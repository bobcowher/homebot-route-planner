"""Dump real observations + goal geometry to answer: does the obs contain
enough information to act, or is the goal frequently outside the viewport?

Saves obs_<i>.png (96x96 obs upscaled 4x, goal marked if in view) and prints
per-reset geometry: robot pos, goal pos, distance, goal-in-viewport.
"""

import cv2
import gymnasium as gym
import numpy as np

import homebot  # noqa: F401  (side-effect env registration)


def make_env():
    env = gym.make(
        "HomeBot2D-Goal-V1",
        render_mode="rgb_array",
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=2,
        max_steps=1000,
        map_name="default",
        goals=["collect_trash"],
    )
    return env


def main():
    env = make_env()
    base = env.unwrapped
    renderer = base._renderer
    vw, vh = renderer._viewport_w, renderer._viewport_h
    mw, mh = base._map.pixel_width, base._map.pixel_height
    print(f"Map: {mw}x{mh} px | viewport: {vw}x{vh} px "
          f"({100 * vw / mw:.0f}% x {100 * vh / mh:.0f}% of map)")

    in_view_count = 0
    n = 8
    for i in range(n):
        raw_obs, _ = env.reset()
        obs = raw_obs["observation"]          # (96, 96, 3)
        goal = raw_obs["desired_goal"]        # absolute map px
        robot = raw_obs["achieved_goal"]      # absolute map px

        # Same clamped viewport origin the renderer uses.
        vx = max(0, min(int(robot[0] - vw / 2), mw - vw))
        vy = max(0, min(int(robot[1] - vh / 2), mh - vh))
        in_view = (vx <= goal[0] <= vx + vw) and (vy <= goal[1] <= vy + vh)
        in_view_count += in_view
        dist = float(np.linalg.norm(goal - robot))

        img = cv2.resize(obs, (384, 384), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if in_view:
            gx = int((goal[0] - vx) / vw * 384)
            gy = int((goal[1] - vy) / vh * 384)
            cv2.circle(img, (gx, gy), 12, (0, 0, 255), 2)
        cv2.imwrite(f"obs_{i}.png", img)

        print(f"reset {i}: robot=({robot[0]:.0f},{robot[1]:.0f}) "
              f"goal=({goal[0]:.0f},{goal[1]:.0f}) dist={dist:.0f}px "
              f"goal_in_viewport={in_view}")

    print(f"\nGoal visible at spawn: {in_view_count}/{n}")


if __name__ == "__main__":
    main()
