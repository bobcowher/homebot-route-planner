"""Does the V1 image observation uniquely identify the robot's position?

A deterministic policy maps view -> action. If two distant positions render
identical (or near-identical) 96x96 observations, the policy is forced to act
identically there — structural looping. This sweeps the robot over a tile grid
on a trash-free map (constant everything except position) and measures pairwise
observation distance vs physical distance.
"""

import gymnasium as gym
import numpy as np

import homebot  # noqa: F401  (side-effect env registration)


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=0,
                max_steps=1000,
                map_name="default",
                goals=[],  # no trash/package spawns -> static scene
            )
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def main():
    env = make_env()
    env.reset()
    base = env.unwrapped
    robot = base._robot
    game_map = base._map
    ts = game_map.tile_size
    mw, mh = game_map.pixel_width, game_map.pixel_height

    base._steps = 0       # freeze recliner animation frame
    robot.angle = 0.0     # constant heading -> constant robot sprite

    positions, obs_list = [], []
    for col, row in game_map.valid_floor_tiles():
        x, y = game_map.tile_to_pixel(col, row)
        robot.x, robot.y = float(x), float(y)
        positions.append((x, y))
        obs_list.append(base._get_obs().astype(np.float32).ravel())

    n = len(positions)
    print(f"Free-tile positions sampled: {n}")
    obs_arr = np.stack(obs_list)          # (n, 27648)
    pos_arr = np.array(positions)         # (n, 2)

    # Pairwise distances.
    obs_d = np.linalg.norm(obs_arr[:, None] - obs_arr[None], axis=2)   # L2 in obs space
    pos_d = np.linalg.norm(pos_arr[:, None] - pos_arr[None], axis=2)   # px on map
    iu = np.triu_indices(n, k=1)
    obs_d, pos_d = obs_d[iu], pos_d[iu]

    # Per-pixel RMS difference makes the obs distance interpretable (0-255 scale).
    rms = obs_d / np.sqrt(obs_arr.shape[1])

    exact = int((obs_d == 0).sum())
    print(f"\nExact-duplicate observation pairs: {exact}")

    for far in (64, 128, 256):
        mask = pos_d >= far
        m = rms[mask]
        print(f"\npositions >= {far}px apart ({mask.sum()} pairs):")
        print(f"  min per-pixel RMS diff: {m.min():.2f}  (255 scale)")
        worst = np.argsort(m)[:3]
        idx = np.where(mask)[0][worst]
        for k in idx:
            i, j = iu[0][k], iu[1][k]
            print(f"  closest views: {tuple(pos_arr[i])} vs {tuple(pos_arr[j])} "
                  f"map-dist={pos_d[k]:.0f}px rms={rms[k]:.2f}")

    # And the local picture: how much does one tile of movement change the view?
    near = pos_d <= ts * 1.5
    print(f"\nadjacent tiles (<= {ts * 1.5:.0f}px, {near.sum()} pairs): "
          f"median per-pixel RMS diff: {np.median(rms[near]):.2f}")


if __name__ == "__main__":
    main()
