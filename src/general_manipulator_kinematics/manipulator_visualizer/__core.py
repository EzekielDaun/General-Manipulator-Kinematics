from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Optional, TypeVar

import networkx as nx
from jaxlie import SE3


class JointType(Enum):
    FIXED = auto()
    SPHERICAL = auto()
    REVOLUTE = auto()
    PRISMATIC = auto()


@dataclass(frozen=True)
class JointAttributes:
    name_id: str
    joint_type: JointType
    axis: Optional[tuple[float, float, float]] = None  # Axis of rotation or translation
    limit: Optional[tuple] = None  # (lower, upper)


T = TypeVar("T")
J = TypeVar("J")


@dataclass(frozen=True)
class ManipulatorVisualizer(Generic[T, J], ABC):
    """An abstract visualizer class including default implementations to visualize manipulators in Matplotlib and PyQtGraph."""

    joint_attributes: set[JointAttributes]
    geometry_graph: nx.Graph
    kinematic_graph: nx.Graph
    characteristic_length: float
    orientation_axis_factor: float  # Length of axis which only represent orientation

    @abstractmethod
    def joint_info(
        self, ext_task_coord: Optional[T], joint_coord: Optional[J]
    ) -> dict[JointAttributes, SE3]:
        """Return a dictionary mapping each joint to its SE3 transformation.
        If joint has an axis, then the SE3 transformation applied to this axis should produce the right axis orientation in world frame.
        """
        raise NotImplementedError
