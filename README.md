# General Manipulator Kinematics

This package is in very early development stages and may not yet be fully functional.

## Description

### General Kinematics

Kinematics of kinematic chains can be expressed in implicit functions[^gosselinSingularityAnalysisClosedloop1990]. This package expands the implicit function to Lie group, and utilize JAX for automatic differentiation with respect to Lie algebra.

Lie group implementation is provided through the use of the [jaxlie](https://github.com/brentyi/jaxlie) library.

### Manipulator Visualizer

Helpers to render a manipulator in Matplotlib or PyQtGraph.

### Torus Visualization

Helpers to visualize 2 or 3 SO2 rotations.

## Installation

This package follows the [PEP 517](https://peps.python.org/pep-0517/) standard for building and installing Python packages. Install it with your favorite package manager.

- core functionality:
  ```bash
  pip install "general_manipulator_kinematics @ git+https://github.com/EzekielDaun/General-Manipulator-Kinematics.git"
  ```
- with Matplotlib visualization:
  ```bash
  pip install "general_manipulator_kinematics[matplotlib-viz] @ git+https://github.com/EzekielDaun/General-Manipulator-Kinematics.git"
  ```
- with PyQtGraph visualization:
  ```bash
  pip install "general_manipulator_kinematics[pyqtgraph-viz] @ git+https://github.com/EzekielDaun/General-Manipulator-Kinematics.git"
  ```

## Examples

See [examples](./examples/) for example scripts.

Visualization examples need this package to be installed with the `matplotlib-viz` and `pyqtgraph-viz` extras.

```bash
uv pip install -e ".[matplotlib-viz,pyqtgraph-viz]"
```

### Run Examples

```bash
python -m examples {filename without .py extension}
```

[^gosselinSingularityAnalysisClosedloop1990]: Singularity Analysis of Closed-Loop Kinematic Chains, Gosselin, C. and Angeles, J., IEEE Transactions on Robotics and Automation, 1990, [DOI:10.1109/70.56660](https://doi.org/10.1109/70.56660)
