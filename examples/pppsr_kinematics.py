import jax
import jax.numpy as jnp
import jaxlie
from jax.tree_util import tree_reduce
from jaxlie import SE3, SO2, SO3, MatrixLieGroup

from examples.lib.pppsr import PPPSR, SRPlatformSE3SO23, TaskCoordSE3SO23


def main():
    """3-PPPSR Kinematics Example

    Since this robot is isotropic, SE3 pose does not effect singularity. In this example, we will assign a trajectory for one of the SO2 redundant DoF, and try to optimize the other DoFs accordingly to avoid singularity, using gradient descent with the help from JAX auto-differentiation.

    """
    import matplotlib.pyplot as plt

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
        pose=SE3.identity() @ SE3.from_rotation(SO3.from_z_radians(jnp.deg2rad(0.0))),
        rdof=SO2.from_radians(jnp.deg2rad(jnp.array([90, 90.0, 90.0]))),
        # rdof=SO2.from_radians(jnp.deg2rad(jnp.array([70.0, 70.0, 70.0]))),
    )

    angle_deg_array = jnp.arange(
        jnp.rad2deg(task_coord.rdof.as_radians()[0]),
        jnp.rad2deg(task_coord.rdof.as_radians()[0]) + 360 * 3,
        0.02,
    )

    def update_fn(task_coord, angle_deg):
        grad_wrt_task_log = robot.loss_grad_wrt_task_log(task_coord)
        log_1d = tree_reduce(
            lambda acc, x: jnp.concatenate((acc, x.flatten())),
            grad_wrt_task_log,
            is_leaf=lambda x: isinstance(x, MatrixLieGroup),
        )
        hessian = robot.loss_hessian_wrt_task_log(task_coord)

        delta_log = -jnp.linalg.solve(hessian + 1e3 * jnp.eye(hessian.shape[0]), log_1d)
        delta_log_tree = TaskCoordSE3SO23(
            pose=delta_log[:6],
            rdof=(
                jnp.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ]
                )
                @ delta_log[6:]
            ).reshape((3, 1)),
        )
        updated = jaxlie.manifold.rplus(task_coord, delta_log_tree)

        # overwrite the first angle with the assigned angle
        updated = TaskCoordSE3SO23(
            pose=updated.pose,
            rdof=SO2(
                jnp.vstack(
                    [
                        SO2.from_radians(jnp.deg2rad(angle_deg)).parameters(),
                        updated.rdof.parameters()[1:],
                    ]
                )
            ),
        )

        angle_output = jnp.rad2deg(updated.rdof.as_radians()).squeeze()
        loss_val = robot.loss(updated)
        return updated, (angle_output, loss_val)

    # Iterate through the angles and compute the outputs
    final_task_coord, (angle_outputs, loss_history) = jax.lax.scan(
        update_fn, task_coord, angle_deg_array
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    for i in range(3):
        ax1.plot(angle_deg_array, angle_outputs[:, i], label=labels[i])
    ax1.set_ylabel("Redundant DoF (deg)")
    ax1.set_title("Redundant DoF Evolution")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(angle_deg_array, loss_history, color="black")
    ax2.set_xlabel(r"Fixed $\theta_1$ (deg)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Evolution")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
