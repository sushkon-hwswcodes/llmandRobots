# Project Status

Last updated: 2026-04-06

## Current branch state

- Branch: `main`
- Remote tracking: `origin/main`
- Local setup status: Robosuite + Ollama + PyRoKi are working on this machine

## Current benchmark readout on this machine

- Panda cube-lift benchmark runs end-to-end with local Ollama Qwen output capture
- Panda cube-lift local readout is currently much worse than the documented historical baseline
- Green-target clutter also runs end-to-end locally, but is currently far below the earlier recorded result
- A short local oracle cube-lift run also underperformed, which suggests the current gap is not purely a model issue

## Active benchmark milestones from the repo history

- Robosuite privileged Panda baseline was previously recorded at `20/20`
- Phase 2 shape generalization was previously recorded at `25/30`
- Phase 2.75 green-target clutter was previously recorded at `24/30`
- Phase 3 YCB target-clutter was previously recorded at `9/30`

## Current focus

- Preserve Panda as the reference benchmark path
- Re-establish a trustworthy Panda local baseline on this machine
- Keep hand configuration support only at the simulation / instantiation level for now
- Remove the old project-local Inspire prompt / preset benchmark direction and restart hand work from a smaller surface area later

## Current local artifacts

- Panda setup and workflow docs have been updated for this machine
- The env registry now skips optional broken imports instead of disabling the whole Robosuite family
- A dedicated hand-config smoke test now verifies that `PandaDexRH` can instantiate and step in simulation

## Known local issue

The current local Panda reward gap likely comes from one or more of:

1. low-level Panda control drift after the configurable-hand refactor
2. reachable Robosuite revision drift versus the earlier benchmarked state
3. model/backend differences in the local Ollama path

## What still needs attention

- Compare the current Panda control path against the earlier `20/20` milestone more directly
- Restore a trustworthy Panda local oracle result before starting a new hand direction
- Decide on the next hand experiment only after Panda is behaving predictably again
