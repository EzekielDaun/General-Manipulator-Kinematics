import importlib.util

if importlib.util.find_spec("matplotlib") or importlib.util.find_spec("pyqtgraph"):
    from .__core import (
        JointAttributes,
        JointType,
        ManipulatorVisualizer,
    )

    __all__ = [
        "JointAttributes",
        "JointType",
        "ManipulatorVisualizer",
    ]

    if importlib.util.find_spec("matplotlib"):
        from .__manipulator_visualizer_mpl import ManipulatorVisualizerMPL

        __all__.append("ManipulatorVisualizerMPL")

    if importlib.util.find_spec("pyqtgraph") is not None:
        from .__manipulator_visualizer_pqg import ManipulatorVisualizerPQG
        __all__.append("ManipulatorVisualizerPQG")
