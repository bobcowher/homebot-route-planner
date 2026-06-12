"""Save one raw HomeBot2D-V1 observation (96x96, upscaled 4x) to obs_v1.png."""

import cv2
import gymnasium as gym

import homebot  # noqa: F401  (side-effect env registration)


def main():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=1000,
                map_name="default",
                goals=["trash"],
            )
            break
        except gym.error.Error:
            continue

    obs, _ = env.reset()
    img = cv2.resize(obs, (384, 384), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("obs_v1.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved obs_v1.png | raw obs shape: {obs.shape}")


if __name__ == "__main__":
    main()
