from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import networkx as nx
import numpy as np
from jaxlie import SE3
from OpenGL.GL import (  # type: ignore
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
from pyqtgraph.opengl import (  # type: ignore
    GLGraphItem,
    GLGridItem,
    GLLinePlotItem,
    GLTextItem,
)
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem  # type: ignore
from PySide6.QtGui import QMatrix4x4  # type: ignore

from .__core import JointType, ManipulatorVisualizer


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
class ManipulatorVisualizerPQG[T, J](ManipulatorVisualizer):
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
