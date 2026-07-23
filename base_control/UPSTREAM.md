# Upstream provenance

This package contains the small subset of
[`tkkim-robot/safe_control`](https://github.com/tkkim-robot/safe_control)
needed by this repository. It was imported from commit
`b8b0f201173c6929ee50eb46cd50d7b5d86149b4`, which was the public submodule
revision recorded by this project before the private fork was introduced.

The code is kept in-tree so the experiments do not depend on an inaccessible
private submodule. Unsupported robot models, planners, demonstrations, and
shielding modules were intentionally omitted. Project-specific occlusion
controllers remain under `position_control/`.
