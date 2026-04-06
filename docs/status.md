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
- YCB single-object lift: `13/20`, average reward `0.750`
- YCB target clutter: `12/30`, average reward `0.402`
- Current YCB clutter result is above the earlier recorded `9/30`

## Current YCB object breakdown

- Strongest category so far: `005_tomato_soup_can`
- Mixed but promising: `006_mustard_bottle`, `010_potted_meat_can`, `009_gelatin_box`
- Weak categories to focus on next: `008_pudding_box`, `004_sugar_box`
- The remaining YCB gap now looks concentrated in a few flatter box-like objects rather than a broad setup failure

## What still needs attention

- Decide whether the weak flat YCB packages need a small category-specific grasp bias or a slightly different preshape / orientation
- Re-run the YCB benchmarks after any object-specific grasp refinement
- Keep hand work deferred until the maintained Panda / shape / clutter / YCB path is stable
