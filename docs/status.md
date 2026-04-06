# Project Status

Last updated: 2026-04-06

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

## Current local artifacts

- Panda setup and workflow docs have been updated for this machine
- The shape-generalization env no longer depends on an unreachable upstream Robosuite module
- The env registry now skips optional broken imports instead of disabling the whole Robosuite family
- A dedicated hand-config smoke test now verifies that `PandaDexRH` can instantiate and step in simulation
- The local shell helper on this machine exports `MANISKILL_ASSET_DIR=/root/.maniskill/data` for the YCB path

## Current local readout

- Panda cube-lift: recovered locally
- Shape generalization: recovered in a short `5/5` run
- Green-target clutter: recovered at `24/30`
- YCB single-object lift: improved to `3/5` in a short run after restoring center grasps
- YCB target clutter: improved to `3/5` in a short run after restoring center grasps

## What still needs attention

- Re-run the YCB benchmarks at larger trial counts and compare them against the earlier milestone
- Decide whether YCB needs any category-specific grasp heuristics beyond the restored center-grasp baseline
- Keep hand work deferred until the maintained Panda / shape / clutter / YCB path is stable
