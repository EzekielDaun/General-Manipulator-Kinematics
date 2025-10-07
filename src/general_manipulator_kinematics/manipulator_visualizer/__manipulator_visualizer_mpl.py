from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jaxlie import SE3
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

from .__core import JointAttributes, JointType, ManipulatorVisualizer


@dataclass(frozen=True)
class ManipulatorVisualizerMPL[T, J](ManipulatorVisualizer):
    revolute_axis_color = "c"
    prismatic_axis_color = "m"

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
                        color=self.revolute_axis_color,
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
                            color=self.prismatic_axis_color,
                        )

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_box_aspect([1, 1, 1])  # X:Y:Z = 1:1:1
        ax_3d.set_aspect("equal")
