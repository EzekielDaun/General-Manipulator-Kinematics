from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
from jax.tree_util import tree_reduce
from jax_dataclasses import pytree_dataclass
from jaxtyping import Float


@pytree_dataclass(frozen=True, slots=True)
class AbstractLieGroupTree(ABC):
    """Abstract base class for coordinate composed by Lie groups.
    Members must be MatrixLieGroup or 1D JAX arrays.
    """

    pass


class AbstractManipulator[T: AbstractLieGroupTree, J: AbstractLieGroupTree](ABC):
    @abstractmethod
    def kinematic_constraints(self, task_coord: T, joint_coord: J) -> Float:
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self"))
    def jacobian_wrt_joint_log_tree(self, task_coord: T, joint_coord: J):
        def tangent_func(delta):
            return self.kinematic_constraints(
                task_coord, jaxlie.manifold.rplus(joint_coord, delta)
            )

        return jax.jacobian(tangent_func)(jaxlie.manifold.zero_tangents(joint_coord))

    @partial(jax.jit, static_argnames=("self"))
    def jacobian_wrt_joint_log_matrix(self, task_coord: T, joint_coord: J):
        return tree_reduce(
            lambda acc, x: jnp.hstack([acc.squeeze(), x.squeeze()]),
            self.jacobian_wrt_joint_log_tree(task_coord, joint_coord),
            is_leaf=lambda x: isinstance(x, jaxlie.MatrixLieGroup),
        )

    @partial(jax.jit, static_argnames=("self"))
    def jacobian_wrt_task_log_tree(self, task_coord: T, joint_coord: J):
        def tangent_func(delta):
            return self.kinematic_constraints(
                jaxlie.manifold.rplus(task_coord, delta), joint_coord
            )

        return jax.jacobian(tangent_func)(jaxlie.manifold.zero_tangents(task_coord))

    @partial(jax.jit, static_argnames=("self"))
    def jacobian_wrt_task_log_matrix(self, task_coord: T, joint_coord: J):
        return tree_reduce(
            lambda acc, x: jnp.hstack([acc.squeeze(), x.squeeze()]),
            self.jacobian_wrt_task_log_tree(task_coord, joint_coord),
            is_leaf=lambda x: isinstance(x, jaxlie.MatrixLieGroup),
        )

    @partial(jax.jit, static_argnames=("self"))
    def ik_jacobian(self, task_coord: T, joint_coord: J):
        jacobian_wrt_joint_log_matrix = self.jacobian_wrt_joint_log_matrix(
            task_coord, joint_coord
        )
        jacobian_wrt_task_log_matrix = self.jacobian_wrt_task_log_matrix(
            task_coord, joint_coord
        )

        return (
            -jnp.linalg.pinv(jacobian_wrt_joint_log_matrix)
            @ jacobian_wrt_task_log_matrix
        )

    @partial(jax.jit, static_argnames=("self"))
    def fk_jacobian(self, task_coord: T, joint_coord: J):
        jacobian_wrt_joint_log_matrix = self.jacobian_wrt_joint_log_matrix(
            task_coord, joint_coord
        )
        jacobian_wrt_task_log_matrix = self.jacobian_wrt_task_log_matrix(
            task_coord, joint_coord
        )

        return (
            -jnp.linalg.pinv(jacobian_wrt_task_log_matrix)
            @ jacobian_wrt_joint_log_matrix
        )
