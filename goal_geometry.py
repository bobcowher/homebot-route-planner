"""Geometry helpers for the honest greedy reacher eval.

Stateless and unit-testable without a gym import.
"""
import math
import numpy as np

# Env trash pickup: robot.RADIUS(15) + tile_size(32) * _TRASH_RANGE(0.5) = 31 px.
# Match it so eval "reached" agrees with the env's own pickup distance.
GOAL_RADIUS = 31.0
ROBOT_STEP_PX = 4.0      # homebot DISCRETE_SPEED
EVAL_BUDGET_MULT = 3

# Spinning / limit-cycle detection (scripts/spin_metric.py + the in-train metric).
SPIN_WINDOW = 8


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two points in pixel space."""
    return math.hypot(bx - ax, by - ay)


def reach_reward(achieved, desired, radius):
    """Sparse 0/1 reach reward at a PARAMETRIC radius -- the success-radius
    curriculum knob. Mirrors the env's compute_reward (1.0 within `radius`, else 0)
    but with a settable radius, so the curriculum is a pure HER/relabel artifact:
    shrinking `radius` over training teaches a tighter terminal approach with no env
    change. Handles single (x,y) or batched (...,2) inputs (HER passes arrays)."""
    diff = np.asarray(achieved, dtype=np.float32) - np.asarray(desired, dtype=np.float32)
    dist = np.linalg.norm(diff, axis=-1)
    return (dist <= radius).astype(np.float32)


def reach_radius_at(episode, start, end, anneal_start, anneal_end):
    """Linear success-radius schedule: hold `start` until `anneal_start`, anneal to
    `end` by `anneal_end`, hold `end` after. Anneal during the high-hindsight phase
    (before HER's k-anneal) so the tighter radius always has dense relabel signal."""
    if episode <= anneal_start:
        return start
    if episode >= anneal_end:
        return end
    frac = (episode - anneal_start) / max(1, anneal_end - anneal_start)
    return start + (end - start) * frac


def spin_thresholds(window: int = SPIN_WINDOW):
    """Default (move_min, net_max) for spin_fraction, in pixels. A window 'spins'
    when it walked >= half the window's worth of steps but ended within ~2 steps
    of where it began. Defaults track ROBOT_STEP_PX so they follow the step size."""
    return 0.5 * window * ROBOT_STEP_PX, 2.0 * ROBOT_STEP_PX


def spin_fraction(positions, window, move_min, net_max):
    """Fraction of steps inside a 'moving but not progressing' window -- the
    signature of a limit cycle. positions: list of (x, y) per step.

    A step t (t >= window) spins when, over the trailing `window` steps, the path
    walked is >= move_min (really moved, so not a wall-stick) yet the net
    displacement from window-start is <= net_max (ended ~where it began). Returns
    0.0 for traces shorter than the window."""
    n = len(positions)
    if n <= window:
        return 0.0
    spin = 0
    for t in range(window, n):
        net = distance(positions[t - window][0], positions[t - window][1],
                       positions[t][0], positions[t][1])
        path = sum(distance(positions[i - 1][0], positions[i - 1][1],
                            positions[i][0], positions[i][1])
                   for i in range(t - window + 1, t + 1))
        if path >= move_min and net <= net_max:
            spin += 1
    return spin / (n - window)


def ego_vector(rx: float, ry: float, rtheta: float, gx: float, gy: float) -> np.ndarray:
    """Goal displacement expressed in the robot's egocentric frame.

    Rung 2 variable. Takes the world displacement (gx-rx, gy-ry) and rotates it
    by -rtheta into the robot frame: x = forward component, y = left component.
    MAGNITUDE IS PRESERVED (this is a rotation) — range still leaks through, so
    this isolates the allocentric->egocentric change from the range-stripping
    that Rung 3 adds on top.
    """
    dx = gx - rx
    dy = gy - ry
    c, s = math.cos(rtheta), math.sin(rtheta)
    x_ego = dx * c + dy * s
    y_ego = -dx * s + dy * c
    return np.array([x_ego, y_ego], dtype=np.float32)


def world_vector(rx: float, ry: float, gx: float, gy: float) -> np.ndarray:
    """Goal displacement in the WORLD frame (no heading rotation).

    The discrete action space is 8 fixed compass directions (world-frame), and the
    observation viewport is north-up (world-frame). ego_vector rotated the goal by
    -heading into a frame matching neither — and heading isn't even an input — so
    the net got a goal direction scrambled by an unobservable rotation. Returning
    the raw world displacement puts goal, image, and actions all in one frame.
    """
    return np.array([gx - rx, gy - ry], dtype=np.float32)


def noisy_world_vector(rx: float, ry: float, gx: float, gy: float,
                       noise_std: float = 0.0) -> np.ndarray:
    """Goal displacement in the WORLD frame with Gaussian localization noise.

    Returns [dx, dy] = [gx-rx, gy-ry] + N(0, noise_std²). noise_std=0 reproduces
    world_vector exactly. The noise simulates the shaky-map reality: a real robot
    has a vague position estimate, not a precise one, so the bearing/distance to
    the goal is uncertain. This breaks the memorization key (same position gives
    a different noisy vector each visit) and forces the network to learn a robust
    approach skill rather than a position→action lookup.

    30px ≈ 1 tile ≈ ~47cm — realistic for indoor localization without a tight
    SLAM stack. At close range (31px trash collect) the noise dominates the
    vector, which is itself realistic: close-range navigation should come from
    the observation (visual servoing), not the noisy goal vector.
    """
    dx = gx - rx
    dy = gy - ry
    if noise_std > 0:
        dx += float(np.random.randn() * noise_std)
        dy += float(np.random.randn() * noise_std)
    return np.array([dx, dy], dtype=np.float32)


def world_coords(rx: float, ry: float, gx: float, gy: float) -> np.ndarray:
    """Goal as raw ABSOLUTE coordinates: [robot_x, robot_y, goal_x, goal_y].

    Coordinate reframing (vs world_vector). Instead of handing the network the
    pre-computed displacement (gx-rx, gy-ry), feed both the robot's absolute pose
    and the goal's absolute position and let the net learn the relationship. The
    egocentric viewport hides absolute robot position from the image, so the pose
    must come in through this vector or the net can't localize itself. A real
    robot knows its own pose (odometry/localization) and the goal coordinate, so
    this stays within the env-realism rule. More compositional than world_vector
    (the net computes the subtraction + direction), which is why it's the natural
    place to test head depth.
    """
    return np.array([rx, ry, gx, gy], dtype=np.float32)


def eval_step_budget(init_dist: float) -> int:
    """Step budget for greedy eval.

    Budget = EVAL_BUDGET_MULT * ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX).
    Tight enough that a circling policy cannot sweep-and-pass; grows linearly
    with spawn distance so far-away goals aren't penalised unfairly.
    """
    return EVAL_BUDGET_MULT * max(1, math.ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX))
