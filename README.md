# General Manipulator Kinematics

This package is in very early development stages and may not yet be fully functional.

## Description

### General Kinematics

Kinematics of kinematic chains can be expressed in implicit functions[^gosselinSingularityAnalysisClosedloop1990]. This package expands the implicit function to Lie group, and utilize JAX for automatic differentiation with respect to Lie algebra.

Lie group implementation is provided through the use of the [jaxlie](https://github.com/brentyi/jaxlie) library.

### Manipulator Visualizer

Helper to render a manipulator in Matplotlib or PyQtGraph.

### Torus Visualization

Helper to visualized 2 or 3 SO2 rotations.

## Installation

- pip
  ```bash
  pip install .
  ```
- poetry
  ```bash
  poetry install
  ```

## Examples

See [examples](./examples/) for example scripts.

```bash
python -m examples {filename without .py extension}
```

[^gosselinSingularityAnalysisClosedloop1990]: Singularity Analysis of Closed-Loop Kinematic Chains, Gosselin, C. and Angeles, J., IEEE Transactions on Robotics and Automation, 1990, [DOI:10.1109/70.56660](https://doi.org/10.1109/70.56660)
