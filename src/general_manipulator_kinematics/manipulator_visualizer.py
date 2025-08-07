from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import jax.numpy as jnp
import networkx as nx
import numpy as np
from jaxlie import SE3
from mpl_toolkits.mplot3d import Axes3D
from OpenGL.GL import (
    GL_LINE_SMOOTH,
    GL_LINE_SMOOTH_HINT,
    GL_LINES,
    GL_NICEST,
    glBegin,
    glColor4f,
    glEnable,
    glEnd,
    glHint,
    glLineWidth,
    glVertex3f,
)
from pyqtgraph.opengl import GLGraphItem, GLGridItem, GLLinePlotItem, GLTextItem
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from PySide6.QtGui import QMatrix4x4


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


class GLAxisWidthItem(GLGraphicsItem):
    """GLAxisWidthItem

    3D axis with width
    """

    def __init__(
        self,
        *args,
        size=(1, 1, 1),
        colors=((1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)),
        line_width: float = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__size = size
        self.__colors = colors
        self.__line_width = line_width
        self.update()

    def paint(self):
        self.setupGLState()

        glLineWidth(self.__line_width)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glBegin(GL_LINES)

        x, y, z = self.__size
        glColor4f(*self.__colors[2])
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, z)

        glColor4f(*self.__colors[1])
        glVertex3f(0, 0, 0)
        glVertex3f(0, y, 0)

        glColor4f(*self.__colors[0])
        glVertex3f(0, 0, 0)
        glVertex3f(x, 0, 0)
        glEnd()


@dataclass(frozen=True)
class ManipulatorVisualizerMPLConfig:
    revolute_axis_color = "c"
    prismatic_axis_color = "m"


@dataclass(frozen=True)
class ManipulatorVisualizerPQGConfig:
    # TODO: adjust as needed
    pass


@dataclass(frozen=True)
class ManipulatorVisualizer[T, J](ABC):
    """An abstract visualizer class including default implementations to visualize manipulators in Matplotlib and PyQtGraph."""

    joint_attributes: set[JointAttributes]
    geometry_graph: nx.Graph
    kinematic_graph: nx.Graph
    characteristic_length: float
    orientation_axis_factor: float  # Length of axis which only represent orientation
    mpl_config: (
        ManipulatorVisualizerMPLConfig  # Parameters for matplotlib visualization
    )
    pqg_config: ManipulatorVisualizerPQGConfig  # Parameters for pyqtgraph visualization

    @abstractmethod
    def joint_info(
        self, ext_task_coord: Optional[T], joint_coord: Optional[J]
    ) -> dict[JointAttributes, SE3]:
        """Return a dictionary mapping each joint to its SE3 transformation.
        If joint has an axis, then the SE3 transformation applied to this axis should produce the right axis orientation in world frame.
        """
        raise NotImplementedError

    def mpl_plot(
        self,
        ax_3d: Axes3D,
        ext_task_coord: Optional[T],
        joint_coord: Optional[J],
        end_effector_pose: SE3,
    ):
        """Plot the manipulator's joints and their connections in Axes3D."""

        joint_info = self.joint_info(ext_task_coord, joint_coord)

        # render end effector pose
        end_effector_position = end_effector_pose.translation()
        end_effector_rotation_matrix = end_effector_pose.rotation().as_matrix()
        axis_colors = ["r", "g", "b"]
        for i, c in enumerate(axis_colors):
            ax_3d.quiver(
                end_effector_position[0],
                end_effector_position[1],
                end_effector_position[2],
                self.characteristic_length
                * self.orientation_axis_factor
                * end_effector_rotation_matrix[0, i],
                self.characteristic_length
                * self.orientation_axis_factor
                * end_effector_rotation_matrix[1, i],
                self.characteristic_length
                * self.orientation_axis_factor
                * end_effector_rotation_matrix[2, i],
                color=c,
            )

        # render constant geometry profile
        for j1, j2 in self.geometry_graph.edges():
            j1: JointAttributes
            j2: JointAttributes
            p1 = joint_info[j1].translation()
            p2 = joint_info[j2].translation()
            ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black")

        # render kinematic graph
        for j1, j2 in self.kinematic_graph.edges():
            j1: JointAttributes
            j2: JointAttributes
            p1 = joint_info[j1].translation()
            p2 = joint_info[j2].translation()
            ax_3d.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color="gray",
                alpha=0.5,
            )

        # render joints
        for joint in self.joint_attributes:
            joint_position = joint_info[joint].translation()
            ax_3d.plot(*joint_position, marker=".", linestyle="None", color="black")
            ax_3d.text3D(
                joint_position[0],
                joint_position[1],
                joint_position[2],
                joint.name_id,
                fontsize=8,
                color="black",
                ha="center",
                va="bottom",
            )

            # render joint axis if applicable
            if joint.joint_type in {JointType.REVOLUTE, JointType.PRISMATIC}:
                joint_orientation = jnp.array(joint.axis) / jnp.linalg.norm(
                    jnp.array(joint.axis)
                )

                if joint.joint_type == JointType.REVOLUTE:
                    joint_orientation *= (
                        self.orientation_axis_factor * self.characteristic_length
                    )
                    render_end_point = joint_info[joint].apply(joint_orientation)
                    ax_3d.plot(
                        [joint_position[0], render_end_point[0]],
                        [joint_position[1], render_end_point[1]],
                        [joint_position[2], render_end_point[2]],
                        color=self.mpl_config.revolute_axis_color,
                    )
                elif joint.joint_type == JointType.PRISMATIC:
                    if joint.limit is not None:
                        render_start_point = joint_info[joint].apply(
                            joint.limit[0]
                            * jnp.array(joint.axis)
                            / jnp.linalg.norm(jnp.array(joint.axis))
                        )
                        render_end_point = joint_info[joint].apply(
                            joint.limit[1]
                            * jnp.array(joint.axis)
                            / jnp.linalg.norm(jnp.array(joint.axis))
                        )
                        ax_3d.plot(
                            [
                                render_start_point[0],
                                render_end_point[0],
                            ],
                            [
                                render_start_point[1],
                                render_end_point[1],
                            ],
                            [
                                render_start_point[2],
                                render_end_point[2],
                            ],
                            color=self.mpl_config.prismatic_axis_color,
                        )

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_box_aspect([1, 1, 1])  # X:Y:Z = 1:1:1
        ax_3d.set_aspect("equal")

    def gl_graphics_items(
        self,
        ext_task_coord: Optional[T],
        joint_coord: Optional[J],
        end_effector_pose: SE3,
    ):
        """Generates OpenGL graphics items for the manipulator."""

        joint_info = self.joint_info(ext_task_coord, joint_coord)

        def build_gl_graph_item(graph: nx.Graph) -> GLGraphItem:
            joint_list = list(graph.nodes())
            joint_index = {j: i for i, j in enumerate(joint_list)}
            node_positions = np.array(
                [np.asarray(joint_info[j].translation()) for j in joint_list]
            )
            edges = np.array(
                [[joint_index[j1], joint_index[j2]] for j1, j2 in graph.edges()]
            )
            return GLGraphItem(edges=edges, nodePositions=node_positions)

        geometry_graph_item = build_gl_graph_item(self.geometry_graph)
        kinematic_graph_item = build_gl_graph_item(self.kinematic_graph)

        # Render end effector axis
        pose_axis_item = GLAxisWidthItem(
            size=(
                self.characteristic_length * 0.25,
                self.characteristic_length * 0.25,
                self.characteristic_length * 0.25,
            ),
            colors=(
                (0, 1, 1, 0.5),
                (1, 0, 1, 0.5),
                (1, 1, 0, 0.5),
            ),
            line_width=2.0,
        )
        pose_axis_item.setTransform(QMatrix4x4(end_effector_pose.as_matrix().flatten()))

        # Render origin axis
        origin_axis_item = GLAxisWidthItem(
            size=(
                self.characteristic_length * 0.15,
                self.characteristic_length * 0.15,
                self.characteristic_length * 0.15,
            ),
            line_width=4.0,
        )

        # Render ground plane
        grid_item = GLGridItem()
        grid_item.rotate(90, 0, 0, 1)
        grid_item.setSpacing(0.1, 0.1, 0.1)

        # Render joints
        joint_axis_line_plot_items: list[GLLinePlotItem] = []
        joint_name_text_items: list[GLTextItem] = []
        for joint, pose in joint_info.items():
            joint_name_text_items.append(
                GLTextItem(text=joint.name_id, pos=pose.translation())
            )
            if joint.joint_type in {JointType.REVOLUTE, JointType.PRISMATIC}:
                joint_orientation = jnp.array(joint.axis) / jnp.linalg.norm(
                    jnp.array(joint.axis)
                )
                if joint.joint_type == JointType.REVOLUTE:
                    joint_orientation *= (
                        self.orientation_axis_factor * self.characteristic_length
                    )
                    render_end_point = joint_info[joint].apply(joint_orientation)
                    pos = np.array(
                        [
                            pose.translation(),
                            render_end_point,
                        ]
                    )
                    color = (1.0, 0.5, 0.0, 0.5)

                elif joint.joint_type == JointType.PRISMATIC:
                    if joint.limit is not None:
                        render_start_point = joint_info[joint].apply(
                            joint.limit[0]
                            * jnp.array(joint.axis)
                            / jnp.linalg.norm(jnp.array(joint.axis))
                        )
                        render_end_point = joint_info[joint].apply(
                            joint.limit[1]
                            * jnp.array(joint.axis)
                            / jnp.linalg.norm(jnp.array(joint.axis))
                        )
                        pos = np.array(
                            [
                                render_start_point,
                                render_end_point,
                            ]
                        )
                        color = (0.5, 0.0, 0.5, 0.5)

                joint_axis_line_plot_items.append(GLLinePlotItem(pos=pos, color=color))

        return (
            [
                geometry_graph_item,
                kinematic_graph_item,
                pose_axis_item,
                origin_axis_item,
                grid_item,
            ],
            joint_axis_line_plot_items,
            joint_name_text_items,
        )
