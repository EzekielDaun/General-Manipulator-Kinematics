import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import networkx as nx
from jaxlie import SE3, SO2, SO3

from examples.lib.pppsr import PPPSR, JointCoordR9, SRPlatformSE3SO23, TaskCoordSE3SO23
from general_manipulator_kinematics.manipulator_visualizer import (
    JointAttributes,
    JointType,
    ManipulatorVisualizer,
    ManipulatorVisualizerMPLConfig,
    ManipulatorVisualizerPQGConfig,
)


@dataclass(frozen=True)
class PPPSRVisualizer(ManipulatorVisualizer[TaskCoordSE3SO23, JointCoordR9]):
    """Visualizes the 3-PPPSR manipulator."""

    pppsr: PPPSR
    a_i_z_prismatic_joints: list[JointAttributes]
    a_i_y_prismatic_joints: list[JointAttributes]
    a_i_x_prismatic_joints: list[JointAttributes]
    r_i_spherical_joints: list[JointAttributes]
    b_i_revolute_joints: list[JointAttributes]

    @classmethod
    def from_robot(cls, pppsr: PPPSR):
        a_i_z_prismatic_joints = [
            JointAttributes(
                f"a{i+1}_z", JointType.PRISMATIC, axis=(0, 0, 1), limit=(-0.5, 0.5)
            )
            for i in range(pppsr.ai_SE3.get_batch_axes()[0])
        ]

        a_i_y_prismatic_joints = [
            JointAttributes(
                f"a{i+1}_y", JointType.PRISMATIC, axis=(0, 1, 0), limit=(-0.5, 0.5)
            )
            for i in range(pppsr.ai_SE3.get_batch_axes()[0])
        ]

        a_i_x_prismatic_joints = [
            JointAttributes(
                f"a{i+1}_x", JointType.PRISMATIC, axis=(1, 0, 0), limit=(-0.5, 0.0)
            )
            for i in range(pppsr.ai_SE3.get_batch_axes()[0])
        ]

        r_i_spherical_joints = [
            JointAttributes(f"r{i+1}", JointType.SPHERICAL)
            for i in range(pppsr.ai_SE3.get_batch_axes()[0])
        ]

        b_i_revolute_joints = [
            JointAttributes(f"b{i+1}", JointType.REVOLUTE, axis=(0, 0, 1))
            for i in range(pppsr.ai_SE3.get_batch_axes()[0])
        ]

        kinematic_graph = nx.Graph()
        for z, y in zip(a_i_z_prismatic_joints, a_i_y_prismatic_joints):
            kinematic_graph.add_edge(z, y)
        for y, x in zip(a_i_y_prismatic_joints, a_i_x_prismatic_joints):
            kinematic_graph.add_edge(y, x)
        for x, r in zip(a_i_x_prismatic_joints, r_i_spherical_joints):
            kinematic_graph.add_edge(x, r)

        geometry_graph = nx.Graph()
        for b1, b2 in zip(b_i_revolute_joints, b_i_revolute_joints):
            geometry_graph.add_edge(b1, b2)
        b_i_deque = deque(b_i_revolute_joints)
        b_i_deque.rotate(-1)  # Rotate to connect to next b_i
        for b1, b2 in zip(b_i_revolute_joints, b_i_deque):
            geometry_graph.add_edge(b1, b2)

        for r, b in zip(r_i_spherical_joints, b_i_revolute_joints):
            geometry_graph.add_edge(r, b)

        return cls(
            joint_attributes={
                *a_i_z_prismatic_joints,
                *a_i_y_prismatic_joints,
                *a_i_x_prismatic_joints,
                *r_i_spherical_joints,
                *b_i_revolute_joints,
            },
            geometry_graph=geometry_graph,
            kinematic_graph=kinematic_graph,
            characteristic_length=float(
                jnp.average(jnp.linalg.norm(pppsr.ai_SE3.translation(), axis=1))
            ),
            orientation_axis_factor=1e-1,
            mpl_config=ManipulatorVisualizerMPLConfig(),
            pqg_config=ManipulatorVisualizerPQGConfig(),
            pppsr=pppsr,
            a_i_z_prismatic_joints=a_i_z_prismatic_joints,
            a_i_y_prismatic_joints=a_i_y_prismatic_joints,
            a_i_x_prismatic_joints=a_i_x_prismatic_joints,
            r_i_spherical_joints=r_i_spherical_joints,
            b_i_revolute_joints=b_i_revolute_joints,
        )

    def joint_info(
        self,
        ext_task_coord: TaskCoordSE3SO23,
        _joint_coord: Optional[JointCoordR9] = None,
    ) -> dict[JointAttributes, SE3]:
        joint_coord = self.pppsr.ik(ext_task_coord).r9.squeeze().reshape(3, -1)

        joint_info: dict[JointAttributes, SE3] = {}

        for i, a_i_z in enumerate(self.a_i_z_prismatic_joints):
            joint_info[a_i_z] = SE3(self.pppsr.ai_SE3.parameters()[i])
        for i, a_i_y in enumerate(self.a_i_y_prismatic_joints):
            joint_info[a_i_y] = SE3(
                self.pppsr.ai_SE3.parameters()[i]
            ) @ SE3.from_translation(jnp.array([0, 0, joint_coord[i, 2]]))
        for i, a_i_x in enumerate(self.a_i_x_prismatic_joints):
            joint_info[a_i_x] = SE3(
                self.pppsr.ai_SE3.parameters()[i]
            ) @ SE3.from_translation(
                jnp.array([0, joint_coord[i, 1], joint_coord[i, 2]])
            )
        for i, r_i in enumerate(self.r_i_spherical_joints):
            joint_info[r_i] = SE3(
                self.pppsr.ai_SE3.parameters()[i]
            ) @ SE3.from_translation(
                jnp.array([joint_coord[i, 0], joint_coord[i, 1], joint_coord[i, 2]])
            )

        for i, b_i in enumerate(self.b_i_revolute_joints):
            joint_info[b_i] = SE3(ext_task_coord.pose.parameters()) @ SE3(
                self.pppsr.sr_platform.vi_SE3.parameters()[i]
            )

        return joint_info


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pyqtgraph.opengl import GLViewWidget
    from PySide6.QtWidgets import QApplication

    ai_SE3 = SE3.from_rotation_and_translation(
        rotation=(r := SO3.from_z_radians(jnp.deg2rad(jnp.array([0.0, 120.0, 240.0])))),
        translation=r.apply(jnp.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])),
    )
    vi_SE3 = SE3.from_rotation_and_translation(
        rotation=r @ SO3.from_x_radians(jnp.deg2rad(0.0)),
        translation=r.apply(jnp.array([[0.5, 0, 0], [0.5, 0, 0], [0.5, 0, 0]])),
    ) @ SE3.from_translation(jnp.array([-0.0, 0.0, 0.0]))

    robot = PPPSR(
        ai_SE3_params=tuple(ai_SE3.parameters().flatten().tolist()),
        sr_platform=SRPlatformSE3SO23(
            vi_SE3_params=tuple(vi_SE3.parameters().flatten().tolist()),
            li_tuple=(0.1, 0.1, 0.1),
        ),
    )
    task_coord = TaskCoordSE3SO23(
        pose=SE3.identity() @ SE3.from_rotation(SO3.from_x_radians(jnp.deg2rad(30.0))),
        rdof=SO2.from_radians(jnp.deg2rad(jnp.array([90.0, 90.0, 90.0]))),
    )

    visualizer = PPPSRVisualizer.from_robot(robot)

    fig = plt.figure()
    ax_3d: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
    visualizer.mpl_plot(
        ax_3d,
        ext_task_coord=task_coord,
        joint_coord=None,
        end_effector_pose=task_coord.pose,
    )
    plt.show()

    app = QApplication(sys.argv)

    widget = GLViewWidget()
    for lst in visualizer.gl_graphics_items(
        ext_task_coord=task_coord,
        joint_coord=None,
        end_effector_pose=task_coord.pose,
    ):
        for i in lst:
            widget.addItem(i)

    widget.show()

    sys.exit(app.exec())
