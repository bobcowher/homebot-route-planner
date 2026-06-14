# HomeBot env — desired improvements (running log)

Core rule: no baked-in config or hack a real planning agent couldn't do if this
were the real world. The sim is a stepping stone to real-home-robot techniques.

## Status note (2026-06-14)

The run-260 "top-down RGB drowns the goal" diagnosis was RETRACTED — the same
conv+goal architecture (`her` branch) hit ~43% greedy / 90% noisy. run 260
changed 3 things at once (range→none, allocentric→egocentric, fixed→random
spawn), so it wasn't a clean test. We're now running a one-variable ablation
ladder off `her` to find the real culprit before touching the env. Raycasts may
or may not turn out to be needed.

## Candidate env work (only if the ablation says we need it)

- **Raycast primitive (2D).** Feasible in the current PyGame 2D engine — Robert
  is happy to add it if the ablation shows we need range/obstacle sensing.
  Enables both a raw LiDAR-style distance vector AND a robot-built local
  occupancy costmap (honest top-down, rendered from the robot's own rays).
- **Forward camera / FPV image obs.** PARKED — this is a *game-engine* change,
  not a game change. Current sim is PyGame 2D for simplicity; true 3D first-person
  view is a full rewrite in something like Raylib. Robert is not against it, but
  it's a large lift, not a quick env tweak.
- **`visible(a, b)` / line-of-sight helper** — for honest line-of-sight-gated
  range / the B-tier `bearing_to` grounding stub.

## Medium

- Unified target registry (semantic name -> fixture position) for the grounding layer.
- Expose doorway / passage waypoints for B/A layers (NOT for C).

## Done our-side (no env change needed)

- Random spawn each episode — agent-side `_random_spawn()` (this is an ablation
  variable, not a fixed decision).
