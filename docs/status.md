# Project Status

Last updated: 2026-04-07

## Current branch state

- Branch: `main`
- Remote tracking: `origin/main`
- Local setup status: Robosuite + Ollama + PyRoKi are working on this machine

## Current benchmark readout on this machine

- Panda cube-lift baseline is recovered locally
- Shape generalization is restored locally after replacing the dead historical `lift_shape` dependency
- Green-target clutter is recovered locally
- YCB real-object lift and YCB target clutter both run end-to-end locally with downloaded ManiSkill assets

## Active benchmark milestones from the repo history

- Robosuite privileged Panda baseline was previously recorded at `20/20`
- Phase 2 shape generalization was previously recorded at `25/30`
- Phase 2.75 green-target clutter was previously recorded at `24/30`
- Phase 3 YCB target-clutter was previously recorded at `9/30`

## Current focus

- Preserve Panda as the reference benchmark path
- Keep hand configuration support only at the simulation / instantiation level for now
- Keep the restored synthetic benchmark path stable while moving the same grasp semantics into YCB
- Remove the old project-local Inspire prompt / preset benchmark direction and restart hand work from a smaller surface area later
- Pause new hand-grasp implementation work until a constrained primitive interface is chosen

## Current local artifacts

- Panda setup and workflow docs have been updated for this machine
- The shape-generalization env no longer depends on an unreachable upstream Robosuite module
- The env registry now skips optional broken imports instead of disabling the whole Robosuite family
- A dedicated hand-config smoke test now verifies that `PandaDexRH` can instantiate and step in simulation
- A hand-preset articulation harness and candidate preset video set were generated for the current Inspire smoke path
- The local shell helper on this machine exports `MANISKILL_ASSET_DIR=/root/.maniskill/data` for the YCB path

## Current local readout

- Panda cube-lift: recovered locally
- Shape generalization: recovered in a short `5/5` run
- Green-target clutter: recovered at `24/30`
- YCB single-object lift: `13/20`, average reward `0.750`
- YCB target clutter: `12/30`, average reward `0.402`
- Current YCB clutter result is above the earlier recorded `9/30`

## Fixed-scene YCB clutter checkpoint

- Added a configurable fixed-target YCB clutter path that can place `10` random distractors plus `1` pre-identified target object
- Current stable scene uses target `005_tomato_soup_can` with `10` distractors drawn from a smaller tabletop-safe YCB pool
- Scene render helper added at `scripts/render_ycb_target_clutter_scene.py`
- On the fixed `11`-object scene:
  - attempt 1: single-shot sampled grasp failed
  - attempt 2: a retry loop with re-sampled grasp poses succeeded
  - attempt 3: a more staged pregrasp / midpoint loop failed

## Current YCB object breakdown

- Strongest category so far: `005_tomato_soup_can`
- Mixed but promising: `006_mustard_bottle`, `010_potted_meat_can`, `009_gelatin_box`
- Weak categories to focus on next: `008_pudding_box`, `004_sugar_box`
- The remaining YCB gap now looks concentrated in a few flatter box-like objects rather than a broad setup failure

## What still needs attention

- Decide whether the weak flat YCB packages need a small category-specific grasp bias or a slightly different preshape / orientation
- Re-run the YCB benchmarks after any object-specific grasp refinement
- Keep hand work deferred until the maintained Panda / shape / clutter / YCB path is stable

## Paused hand-work checkpoint

- The current Inspire hand sim path is validated only at the instantiation and articulation level, not as a reliable grasp benchmark path
- Candidate named hand presets were added and exported as videos, but several looked too similar to count as validated distinct grasps
- The current preset harness keeps the wrist vertical, so it validates finger articulation only, not full grasp primitives
- The next hand iteration should combine hand shape with wrist orientation and approach geometry
- Public prior art split found so far:
  - Allegro provides a small named-grasp software interface plus saved custom poses
  - RH56DFX public code provides grasp families and geometric / force-control planning rather than a static preset catalog
- The most promising restart direction is a constrained primitive API:
  - object class supplied before planning
  - small menu such as `open`, `soft_close`, `close` or a few named primitives per class
  - optional PPO-discovered class-conditioned presets later
