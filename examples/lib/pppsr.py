from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
from jax_dataclasses import pytree_dataclass
from jaxlie import SE3, SO2, SO3
from jaxtyping import Float

from general_manipulator_kinematics.general_kinematics import (
    AbstractLieGroupTree,
    AbstractManipulator,
)


@pytree_dataclass(frozen=True)
class JointCoordR9(AbstractLieGroupTree):
    r9: Float


@pytree_dataclass(frozen=True)
class TaskCoordSE3SO23(AbstractLieGroupTree):
    pose: SE3
    rdof: SO2


@pytree_dataclass(frozen=True)
class SRPlatformSE3SO23:
    vi_SE3_params: tuple[float, ...]
    li_tuple: tuple[float, ...]

    @property
    def vi_SE3(self) -> SE3:
        return SE3(jnp.array(self.vi_SE3_params).reshape((3, 7)))

    @property
    def li(self) -> Float:
        return jnp.array(self.li_tuple, dtype=jnp.float32).reshape((3,))

    @property
    def li_pad_3d(self) -> Float:
        return jnp.column_stack(
            [self.li, jnp.zeros_like(self.li), jnp.zeros_like(self.li)]
        )

    @partial(jax.jit, static_argnames=("self",))
    def ik(self, pose: SE3, rdof: SO2) -> Float:
        r_in_v = SO3.from_z_radians(rdof.as_radians()).apply(self.li_pad_3d)
        return pose @ self.vi_SE3 @ r_in_v


@pytree_dataclass(frozen=True)
class PPPSR(AbstractManipulator):
    """3-PPPSR Manipulator

    Reference: https://doi.org/10.1007/978-3-031-95489-4_18
    """

    sr_platform: SRPlatformSE3SO23
    ai_SE3_params: tuple[float, ...]

    @property
    def ai_SE3(self) -> SE3:
        return SE3(jnp.array(self.ai_SE3_params).reshape((3, 7)))

    @partial(jax.jit, static_argnames=("self",))
    def ik(self, task_coord: TaskCoordSE3SO23) -> JointCoordR9:
        r = self.sr_platform.ik(task_coord.pose, task_coord.rdof)
        return JointCoordR9((self.ai_SE3.inverse() @ r).flatten())

    @partial(jax.jit, static_argnames=("self",))
    def kinematic_constraints(
        self, task_coord: TaskCoordSE3SO23, joint_coord: JointCoordR9
    ) -> Float:
        return (self.ik(task_coord).r9 - joint_coord.r9).squeeze()

    @partial(jax.jit, static_argnames=("self",))
    def loss(self, task_coord: TaskCoordSE3SO23) -> Float:
        joint_coord = self.ik(task_coord)
        ik_jac = self.ik_jacobian(task_coord, joint_coord)
        return -jnp.log(jnp.linalg.det(ik_jac @ ik_jac.T))

    @partial(jax.jit, static_argnames=("self",))
    def loss_grad_wrt_task_log(self, task_coord: TaskCoordSE3SO23):
        def tangent_func(delta):
            return self.loss(jaxlie.manifold.rplus(task_coord, delta))

        return jax.grad(tangent_func)(jaxlie.manifold.zero_tangents(task_coord))

    @partial(jax.jit, static_argnames=("self",))
    def loss_hessian_wrt_task_log(self, task_coord: TaskCoordSE3SO23) -> Float:
        def tangent_func(delta):
            return self.loss(jaxlie.manifold.rplus(task_coord, delta))

        hessian_tree = jax.hessian(tangent_func)(
            jaxlie.manifold.zero_tangents(task_coord)
        )
        return jnp.block(
            [
                [
                    hessian_tree.pose.pose.squeeze(),
                    hessian_tree.pose.rdof.squeeze(),
                ],
                [
                    hessian_tree.rdof.pose.squeeze(),
                    hessian_tree.rdof.rdof.squeeze(),
                ],
            ]
        )
