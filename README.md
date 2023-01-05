# Deep Equilibrium Models
### Final project for Deep Learning Systemc CMU course

This is an attempt to implement (pretty dummy) DEQ models based on the `needle` framework.
Things that were implemented are:
- `all()` and `any()` support both in `cpu` and `cuda` backends (without reduction for now). These were needed to define `__bool__` in `needle.array` that was used in `solver` functionality (i.e. comparing norms with thresholds)
- `FixedPoint` operation that encapsulates implicit layer functionality.
- `ModuleOp` operation which is aimed to map any `needle.nn.Module` to `needle.ops.TensorOp`. This allows to use modules in `FixedPoint`.
- `BaseSolver` interface and `ForwardIteration` algorithm implementation for finding fixed points.
- Two models: `tanh(linear(Z) + X)` and ResNet-ish dummy DEQ

Demonstration can be found in [this](project.ipynb) notebook.
