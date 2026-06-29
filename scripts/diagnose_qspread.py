"""Credit-assignment / value-funnel diagnostic for a trained discrete-SAC checkpoint.

Reads a TRAINED critic+actor and measures — directly, in seconds, without re-training —
the quantities the actor-plateau hypotheses turn on:

  1. Δ   = per-action Q-spread  max_a minQ - min_a minQ
  2. directional advantage  minQ[toward-goal] - minQ[away-from-goal]
  3. V(r) = Σπ·minQ vs distance-to-goal r  -> the spatial extent of the value field
            (the "funnel"): how far from the goal the critic still gives a usable gradient.

Probe states are built by teleporting the robot to rings around the trash and re-rendering,
so the goal cue (trash sprite) is in-frame exactly as during training.

Usage:
    python scripts/diagnose_qspread.py [CHECKPOINT_DIR]
CHECKPOINT_DIR defaults to ./checkpoints; pass a Beekeeper run dir to probe a specific run,
e.g. .../persistent/runs/run_394/checkpoints

This is the fast success metric for any value-shape lever (n-step, gamma, HER horizon,
dueling head): change the lever, train briefly, dump a checkpoint, re-run this, and check
whether the funnel WIDENED — instead of waiting ~1500 episodes for an entropy/reach trend.
"""
import sys
import numpy as np
import torch
import gymnasium as gym
import homebot  # noqa: F401
import cv2

from model import DiscreteQNet, DiscretePolicy
from goal_geometry import world_vector, GOAL_RADIUS
from homebot.robot import _DIRS

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRS = np.array(_DIRS, dtype=np.float32)                       # (8,2) screen coords, y-down
DIR_UNIT = DIRS / np.linalg.norm(DIRS, axis=1, keepdims=True)

N_EPISODES = 10
RADII = [8.0, 16.0, 24.0, 32.0, 48.0, 80.0, 120.0, 160.0, 240.0, 360.0, 480.0]   # px (32≈GOAL_RADIUS)
N_DIRS = 16


def load_nets():
    critic = DiscreteQNet(8, checkpoint_dir=CKPT, name="critic", head_layers=4, head_hidden=512).to(DEVICE)
    actor  = DiscretePolicy(8, checkpoint_dir=CKPT, name="actor", head_layers=4, head_hidden=512).to(DEVICE)
    critic.load_checkpoint(); actor.load_checkpoint()
    critic.eval(); actor.eval()
    return critic, actor


def obs_at(env, rx, ry):
    env._robot.x, env._robot.y = float(rx), float(ry)
    o = env._get_obs()
    o = cv2.resize(o, (96, 96), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(o).permute(2, 0, 1)


def main():
    env = gym.make("HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=1, max_steps=1000, map_name="default",
                   goals=["collect_trash"], random_start=True).unwrapped
    critic, actor = load_nets()

    all_delta, all_dir_adv, all_actor_hit, all_q = [], [], [], []
    V_by_radius = {r: [] for r in RADII}
    maxq_by_radius = {r: [] for r in RADII}
    sample_dumps = []

    for ep in range(N_EPISODES):
        env.reset()
        gx, gy = float(env._desired_goal[0]), float(env._desired_goal[1])
        imgs, goals, motions, meta = [], [], [], []
        for r in RADII:
            for k in range(N_DIRS):
                th = 2 * np.pi * k / N_DIRS
                rx, ry = gx + r * np.cos(th), gy + r * np.sin(th)
                imgs.append(obs_at(env, rx, ry))
                goals.append(world_vector(rx, ry, gx, gy))
                motions.append(np.zeros(4, dtype=np.float32))
                meta.append((r, rx, ry))

        img_t = torch.stack(imgs).float().to(DEVICE) / 255.0
        goal_t = torch.as_tensor(np.array(goals), dtype=torch.float32, device=DEVICE)
        mot_t = torch.as_tensor(np.array(motions), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            q1, q2 = critic(img_t, goal_t, mot_t)
            minq = torch.min(q1, q2).cpu().numpy()
            probs, _ = actor(img_t, goal_t, mot_t)
            probs = probs.cpu().numpy()

        for i, (r, rx, ry) in enumerate(meta):
            q = minq[i]
            gdir = np.array([gx - rx, gy - ry], dtype=np.float32)
            gdir /= (np.linalg.norm(gdir) + 1e-8)
            ideal = int(np.argmax(DIR_UNIT @ gdir))
            opp = (ideal + 4) % 8
            all_delta.append(float(q.max() - q.min()))
            all_dir_adv.append(float(q[ideal] - q[opp]))
            all_q.append(float(np.abs(q).mean()))
            actor_choice = int(np.argmax(probs[i]))
            ring_diff = min((actor_choice - ideal) % 8, (ideal - actor_choice) % 8)
            all_actor_hit.append(1.0 if ring_diff <= 1 else 0.0)
            V_by_radius[r].append(float((probs[i] * q).sum()))
            maxq_by_radius[r].append(float(q.max()))
            if ep == 0 and r in (16.0, 80.0) and i % N_DIRS == 0:
                sample_dumps.append((r, q.copy(), ideal, opp, probs[i].copy()))

    d = np.array(all_delta); da = np.array(all_dir_adv); qs = np.array(all_q)
    print("=" * 72)
    print(f"Q-SPREAD / VALUE-FUNNEL DIAGNOSTIC  |  ckpt={CKPT}")
    print(f"{N_EPISODES} goals x {len(RADII)} radii x {N_DIRS} dirs = {len(d)} probe states")
    print("=" * 72)
    print(f"|Q| scale (mean abs minQ)               : {qs.mean():.4f}")
    print(f"Δ per-action spread  max-min            : mean {d.mean():.5f}  median {np.median(d):.5f}  "
          f"p90 {np.percentile(d,90):.5f}")
    print(f"Δ / |Q| (relative spread)               : {d.mean()/(qs.mean()+1e-8):.4f}")
    print(f"Directional adv Q[toward]-Q[away]       : mean {da.mean():+.5f}  "
          f"frac>0 {float((da>0).mean()):.3f} (chance 0.50)")
    print(f"Actor argmax within ±1 of toward-goal   : {np.mean(all_actor_hit):.3f} (chance 0.375)")
    print("-" * 72)
    print("VALUE FUNNEL — V=Σπ·minQ and max_aQ vs distance (want V positive far out):")
    for r in RADII:
        v = np.mean(V_by_radius[r]); mq = np.mean(maxq_by_radius[r])
        geff = (mq ** (4.0 / r)) if mq > 0.02 else float("nan")
        tag = "  <-- inside GOAL_RADIUS" if r <= GOAL_RADIUS else ""
        print(f"   r={r:6.1f}px   V={v:+.4f}   max_aQ={mq:+.4f}   γ_eff={geff:.4f}{tag}")
    print("-" * 72)
    print("Sample raw minQ(s,·) (8 actions N..NW):")
    for r, q, ideal, opp, p in sample_dumps:
        print(f"   r={r:5.1f}  Q=[" + " ".join(f"{x:+.3f}" for x in q) + f"]  ideal_a={ideal} "
              f"argmaxQ={int(np.argmax(q))} π_max_a={int(np.argmax(p))}")
    print("=" * 72)


if __name__ == "__main__":
    main()
