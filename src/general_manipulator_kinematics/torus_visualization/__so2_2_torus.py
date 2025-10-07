from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from jaxlie import SO2
from mpl_toolkits.mplot3d import Axes3D  # type: ignore


@dataclass(frozen=True)
class SO22TorusPlotter:
    R = 2.0
    r = 1.0
    alpha = 0.2
    grid_density = 32
    _curve_overlays: List[tuple[list[SO2], dict]] = field(
        default_factory=list, init=False, repr=False
    )

    def add_curve(self, so2_list: list[SO2], **style):
        self._curve_overlays.append((so2_list, style))
        return self  # enable chaining

    def clear_overlays(self):
        self._curve_overlays.clear()
        return self

    def _so2_to_xyz(self, so2: SO2) -> np.ndarray:
        angles = so2.as_radians()
        if len(angles) == 2:
            u, v = angles
            x = (self.R + self.r * np.cos(v)) * np.cos(u)
            y = (self.R + self.r * np.cos(v)) * np.sin(u)
            z = self.r * np.sin(v)
            return np.array([x, y, z])
        else:
            raise ValueError(
                "SO2 array must be of length 2, got {}".format(len(angles))
            )

    def plot_torus(self, ax_3d: Axes3D):
        # Create mesh for torus surface
        u = np.linspace(0, 2 * np.pi, self.grid_density)
        v = np.linspace(0, 2 * np.pi, self.grid_density)
        u_mesh, v_mesh = np.meshgrid(u, v)

        x = (self.R + self.r * np.cos(v_mesh)) * np.cos(u_mesh)
        y = (self.R + self.r * np.cos(v_mesh)) * np.sin(u_mesh)
        z = self.r * np.sin(v_mesh)

        ax_3d.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            color="lightgray",
            edgecolor="gray",
            linewidth=0.4,
            alpha=self.alpha,
            antialiased=True,
        )

        # Draw longitude (constant u, sweep v) and latitude (constant v, sweep u) grid lines
        # Choose about 8 lines in each direction
        step = max(1, self.grid_density // 8)
        for i in range(0, self.grid_density, step):
            u_fixed = u[i]
            v_sweep = np.linspace(0, 2 * np.pi, 200)
            x_line = (self.R + self.r * np.cos(v_sweep)) * np.cos(u_fixed)
            y_line = (self.R + self.r * np.cos(v_sweep)) * np.sin(u_fixed)
            z_line = self.r * np.sin(v_sweep)
            ax_3d.plot(x_line, y_line, z_line, color="gray", linewidth=0.5, alpha=0.7)

        for j in range(0, self.grid_density, step):
            v_fixed = v[j]
            u_sweep = np.linspace(0, 2 * np.pi, 200)
            x_line = (self.R + self.r * np.cos(v_fixed)) * np.cos(u_sweep)
            y_line = (self.R + self.r * np.cos(v_fixed)) * np.sin(u_sweep)
            z_line = self.r * np.sin(v_fixed) * np.ones_like(u_sweep)
            ax_3d.plot(x_line, y_line, z_line, color="gray", linewidth=0.5, alpha=0.7)

        tick_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        tick_labels = [f"{int(np.round(np.rad2deg(a)))}°" for a in tick_angles]
        for angle, label in zip(tick_angles, tick_labels):
            x0 = (self.R + self.r * np.cos(0)) * np.cos(angle)
            y0 = (self.R + self.r * np.cos(0)) * np.sin(angle)
            z0 = self.r * np.sin(0)
            ax_3d.text(x0, y0, z0, label, fontsize=9, ha="center", va="center")

        for angle, label in zip(tick_angles, tick_labels):
            x1 = (self.R + self.r * np.cos(angle)) * np.cos(0)
            y1 = (self.R + self.r * np.cos(angle)) * np.sin(0)
            z1 = self.r * np.sin(angle)
            ax_3d.text(x1, y1, z1, label, fontsize=9, ha="center", va="center")

        # # Add global coordinate axes
        # axis_len = 1.2 * (self.R + self.r)
        # ax_3d.quiver(0, 0, 0, axis_len, 0, 0, color="red", linewidth=1.5)
        # ax_3d.quiver(0, 0, 0, 0, axis_len, 0, color="green", linewidth=1.5)
        # ax_3d.quiver(0, 0, 0, 0, 0, axis_len, color="blue", linewidth=1.5)
        # ax_3d.text(axis_len, 0, 0, "X", color="red", fontsize=10, ha="center")
        # ax_3d.text(0, axis_len, 0, "Y", color="green", fontsize=10, ha="center")
        # ax_3d.text(0, 0, axis_len, "Z", color="blue", fontsize=10, ha="center")

        # Render overlay curves
        for so2_list, style in self._curve_overlays:
            coords = np.array([self._so2_to_xyz(so2) for so2 in so2_list])
            ax_3d.plot(coords[:, 0], coords[:, 1], coords[:, 2], **style)

        axis_length = 1.2 * (self.R + self.r)
        ax_3d.set_xlim([-axis_length, axis_length])
        ax_3d.set_ylim([-axis_length, axis_length])
        ax_3d.set_zlim([-axis_length, axis_length])
        ax_3d.set_box_aspect([1, 1, 1])
        ax_3d.set_axis_off()
        ax_3d.set_aspect("equal")


if __name__ == "__main__":
    fig = plt.figure()
    ax_3d: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    plotter = SO22TorusPlotter()

    # Example curve
    curve_angles = [
        SO2.from_radians(np.array([2 * np.pi * i / 100, 4 * np.pi * i / 100]))
        for i in range(100)
    ]

    plotter.add_curve(curve_angles, color="purple", linewidth=2.0)
    plotter.plot_torus(ax_3d)

    plt.show()
