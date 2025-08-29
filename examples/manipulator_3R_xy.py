from functools import partial

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxlie import SE2, SO2
from jaxtyping import Float

from general_manipulator_kinematics.general_kinematics import (
    AbstractLieGroupTree,
    AbstractManipulator,
)


@pytree_dataclass(frozen=True, slots=True)
class JointCoord(AbstractLieGroupTree):
    theta_batched3: SO2
    xy: Float  # redundant, no meaning


@pytree_dataclass(frozen=True, slots=True)
class TaskCoord(AbstractLieGroupTree):
    pose: SE2
    xy: Float  # redundant, no meaning


class Manipulator3R_XY(AbstractManipulator[TaskCoord, JointCoord]):
    """A planar 3R Manipulator in SE2, with extra unused XY coordinates"""

    @partial(jax.jit, static_argnames=("self"))
    def kinematic_constraints(
        self, task_coord: TaskCoord, joint_coord: JointCoord
    ) -> Float:
        fk_se2 = (
            SE2.from_translation(
                SO2(joint_coord.theta_batched3.parameters()[0]).apply(
                    jnp.array([0.9, 0])
                )
            )
            @ SE2.from_translation(
                SO2(joint_coord.theta_batched3.parameters()[1]).apply(
                    jnp.array([1.0, 0])
                )
            )
            @ SE2.from_translation(
                SO2(joint_coord.theta_batched3.parameters()[2]).apply(
                    jnp.array([0.1, 0])
                ),
            )
            @ SE2.from_rotation(SO2(joint_coord.theta_batched3.parameters()[2]))
        )

        g_delta = task_coord.pose.inverse() @ fk_se2

        return jnp.concatenate(
            [SE2.log(g_delta).squeeze(), (task_coord.xy - joint_coord.xy).squeeze()]
        )


def main():
    robot_arm_3r = Manipulator3R_XY()
    task_coord = TaskCoord(
        pose=SE2.from_rotation_and_translation(
            SO2.identity() @ SO2.from_radians(jnp.pi / 2),
            jnp.array([0.9, 1.1]),
        ),
        xy=jnp.array([0.0, 0.0]),
    )
    joint_coord = JointCoord(
        theta_batched3=SO2.from_radians(jnp.array([0.0, jnp.pi / 2, jnp.pi / 2])),
        xy=jnp.array([0.0, 0.0]),
    )

    print("FK Jacobian:")
    print(robot_arm_3r.fk_jacobian(task_coord, joint_coord))
    print("IK Jacobian:")
    print(robot_arm_3r.ik_jacobian(task_coord, joint_coord))

    det = jnp.linalg.det(
        robot_arm_3r.fk_jacobian(task_coord, joint_coord)
        @ robot_arm_3r.ik_jacobian(task_coord, joint_coord)
    )
    assert jnp.isclose(det, 1.0)
