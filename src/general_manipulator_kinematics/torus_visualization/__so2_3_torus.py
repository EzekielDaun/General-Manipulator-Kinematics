from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from jaxlie import SO2
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_so2_3_torus(
    ax: Axes3D,
    so2_3: SO2,
    R=2.0,
    r=1.0,
    alpha=0.2,
    grid_density=32,
    lie_algebra_grad: Optional[np.ndarray] = None,
):
    """Draws a torus representing SO(2) x SO(2)

    Args:
        R (float): major radius
        r (float): minor radius
        alpha (float): transparency level
        grid_density (int): number of mesh points per dimension
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Axes to plot on. Creates new if None.
        so2_a (Optional[SO2]): first SO2 element to highlight on the torus
        so2_b (Optional[SO2]): second SO2 element to highlight on the torus
        so2_c (Optional[SO2]): optional SO2 element to rotate the torus
    """

    u_mesh, v_mesh = np.meshgrid(
        np.linspace(0, 2 * np.pi, grid_density), np.linspace(0, 2 * np.pi, grid_density)
    )

    x = (R + r * np.cos(v_mesh)) * np.cos(u_mesh)
    y = (R + r * np.cos(v_mesh)) * np.sin(u_mesh)
    z = r * np.sin(v_mesh)

    u_angle, v_angle, w_angle = so2_3.as_radians()

    cos_a, sin_a = np.cos(w_angle), np.sin(w_angle)
    # Rotate around Y axis
    x_new = cos_a * x + sin_a * z
    y_new = y
    z_new = -sin_a * x + cos_a * z

    ax.plot_surface(
        x_new,
        y_new,
        z_new,
        rstride=1,
        cstride=1,
        color="skyblue",
        edgecolor=(0, 0, 0, 0.3),
        linewidth=0.5,
        alpha=alpha,
        antialiased=True,
    )

    # Add angular ticks like a globe using u (theta_1) and v (theta_2)
    tick_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    tick_labels = [f"{int(np.round(np.rad2deg(a)))}°" for a in tick_angles]

    for angle, label in zip(tick_angles, tick_labels):
        # Theta_1 ticks (u direction): vary u, fix v = 0
        x0 = (R + r * np.cos(0)) * np.cos(angle)
        y0 = (R + r * np.cos(0)) * np.sin(angle)
        z0 = r * np.sin(0)
        x0_new = cos_a * x0 + sin_a * z0
        y0_new = y0
        z0_new = -sin_a * x0 + cos_a * z0
        ax.text(x0_new, y0_new, z0_new, label, fontsize=10, ha="center", va="center")

        # Theta_2 ticks (v direction): vary v, fix u = 0
        x1 = (R + r * np.cos(angle)) * np.cos(0)
        y1 = (R + r * np.cos(angle)) * np.sin(0)
        z1 = r * np.sin(angle)
        x1_new = cos_a * x1 + sin_a * z1
        y1_new = y1
        z1_new = -sin_a * x1 + cos_a * z1
        ax.text(x1_new, y1_new, z1_new, label, fontsize=10, ha="center", va="center")

    x_pt = (R + r * np.cos(v_angle)) * np.cos(u_angle)
    y_pt = (R + r * np.cos(v_angle)) * np.sin(u_angle)
    z_pt = r * np.sin(v_angle)

    x_pt_new = cos_a * x_pt + sin_a * z_pt
    y_pt_new = y_pt
    z_pt_new = -sin_a * x_pt + cos_a * z_pt

    # Visualize the torus' local X and Z axes after Y rotation
    axis_length = 1.2 * (R + r)
    x_axis = np.array([axis_length, 0, 0])
    z_axis = np.array([0, 0, axis_length])

    def rotate_y(vec, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y, z = vec
        return np.array([cos_a * x + sin_a * z, y, -sin_a * x + cos_a * z])

    x_rot = rotate_y(x_axis, w_angle)
    z_rot = rotate_y(z_axis, w_angle)
    ax.quiver(0, 0, 0, *x_rot, color="blue", linewidth=2)
    ax.quiver(0, 0, 0, *z_rot, color="green", linewidth=2)

    # Draw XY plane center circle for the major ring (SO(2)_1 sweep)
    theta_center = np.linspace(0, 2 * np.pi, 200)
    x_center = R * np.cos(theta_center)
    y_center = R * np.sin(theta_center)
    z_center = np.zeros_like(theta_center)
    x_center_rot = cos_a * x_center + sin_a * z_center
    y_center_rot = y_center
    z_center_rot = -sin_a * x_center + cos_a * z_center
    ax.plot(
        x_center_rot,
        y_center_rot,
        z_center_rot,
        color="black",
        alpha=0.5,
        linewidth=1,
    )

    # Draw line representing second SO2 from main circle to surface point
    x_center_pt = R * np.cos(u_angle)
    y_center_pt = R * np.sin(u_angle)
    z_center_pt = 0.0
    x_center_pt_rot = cos_a * x_center_pt + sin_a * z_center_pt
    y_center_pt_rot = y_center_pt
    z_center_pt_rot = -sin_a * x_center_pt + cos_a * z_center_pt
    # Draw connection from origin to SO(2)_1 point on main circle
    ax.plot(
        [0.0, x_center_pt_rot],
        [0.0, y_center_pt_rot],
        [0.0, z_center_pt_rot],
        color="darkblue",
        linestyle="-",
        linewidth=1.5,
    )
    ax.plot(
        [x_center_pt_rot, x_pt_new],
        [y_center_pt_rot, y_pt_new],
        [z_center_pt_rot, z_pt_new],
        color="darkred",
        linestyle="-",
        linewidth=1.5,
    )

    # Draw a circular arc in XZ plane at Y=0 for all possible z-axis arrow tips
    arc_radius = np.linalg.norm(z_rot)
    theta_arc = np.linspace(0, 2 * np.pi, 200)
    x_arc = arc_radius * np.cos(theta_arc)
    y_arc = np.zeros_like(theta_arc)
    z_arc = arc_radius * np.sin(theta_arc)
    ax.plot(x_arc, y_arc, z_arc, color="gray", linestyle="--", linewidth=1)

    # Add tick marks and labels around the circle (XZ plane)
    tick_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    tick_labels = [f"{int(np.round(np.rad2deg(a)))}°" for a in tick_angles]
    for angle, label in zip(tick_angles, tick_labels):
        x_tick = arc_radius * np.cos(angle)
        z_tick = arc_radius * -np.sin(angle)
        ax.text(x_tick, 0.0, z_tick + 0.1, label, fontsize=9, ha="center", va="bottom")

    ax.scatter(
        [x_pt_new],
        [y_pt_new],
        [z_pt_new],  # type: ignore
        color="red",
        s=50,
    )

    # Draw gradient arrows if provided
    if lie_algebra_grad is not None:
        # Gradient components
        du, dv, dw = lie_algebra_grad

        # Compute tangent vectors at the surface point
        du_vec = np.array(
            [
                -(R + r * np.cos(v_angle)) * np.sin(u_angle),
                (R + r * np.cos(v_angle)) * np.cos(u_angle),
                0,
            ]
        )  # partial derivative w.r.t. u

        dv_vec = np.array(
            [
                -r * np.sin(v_angle) * np.cos(u_angle),
                -r * np.sin(v_angle) * np.sin(u_angle),
                r * np.cos(v_angle),
            ]
        )  # partial derivative w.r.t. v

        # Normalize for direction, then scale by gradient magnitude
        du_arrow = du * du_vec / np.linalg.norm(du_vec)
        dv_arrow = dv * dv_vec / np.linalg.norm(dv_vec)

        # Rotate these arrows by Y rotation
        du_arrow_rot = rotate_y(du_arrow, w_angle)
        dv_arrow_rot = rotate_y(dv_arrow, w_angle)

        # Draw gradient arrows at the red point
        ax.quiver(
            x_pt_new,
            y_pt_new,
            z_pt_new,
            *du_arrow_rot,
            color=default_colors[0],
            linewidth=3,
            arrow_length_ratio=0.3,
        )
        ax.quiver(
            x_pt_new,
            y_pt_new,
            z_pt_new,
            *dv_arrow_rot,
            color=default_colors[1],
            linewidth=3,
            arrow_length_ratio=0.3,
        )

        # Draw third component from (1, 0, 0) in rotated frame
        base = x_rot
        neg_z_axis_dir = rotate_y(np.array([0, 0, -1.0]), w_angle)
        dw_arrow = (
            dw * np.linalg.norm(z_rot) * neg_z_axis_dir / np.linalg.norm(neg_z_axis_dir)
        )
        ax.quiver(
            *base,
            *dw_arrow,
            color=default_colors[2],
            linewidth=3,
            arrow_length_ratio=0.3,
        )

    ax.set_xlim([-axis_length, axis_length])
    ax.set_ylim([-axis_length, axis_length])
    ax.set_zlim([-axis_length, axis_length])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_aspect("equal")


if __name__ == "__main__":
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
    for angle in np.arange(-45, 45, 5):
        plot_so2_3_torus(
            ax,
            so2_3=SO2.from_radians(
                np.deg2rad(np.array([90.0 + angle, 180.0 - 2 * angle, angle]))
            ),
            lie_algebra_grad=np.array(
                [1.0 * angle / 45, 0.5 * angle / 45, 0.2 * angle / 45.0]
            ),
        )
        plt.tight_layout()
        # plt.pause(0)
        plt.pause(0.5)
        ax.clear()
