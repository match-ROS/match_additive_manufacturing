#!/usr/bin/env python3
import argparse
import numpy as np
import rosbag
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

bagfile = "record_20251210_141133_GUI-PC.bag"


def read_poses_from_bag(bag_path, topic):
    xs, ys, zs = [], [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            p = msg.pose.position
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
    return np.array(xs), np.array(ys), np.array(zs)


def split_layers_by_z(z, z_jump_threshold):
    """
    Teilt die Zeitreihe in Layer auf, wenn der z-Sprung größer als z_jump_threshold ist.
    Gibt eine Liste von slice-Objekten zurück.
    """
    layers = []
    if len(z) == 0:
        return layers

    start_idx = 0
    for i in range(1, len(z)):
        if abs(z[i] - z[i - 1]) > z_jump_threshold:
            layers.append(slice(start_idx, i))
            start_idx = i
    layers.append(slice(start_idx, len(z)))
    return layers


def resample_layer_xy(x, y, slc, n_samples):
    """Resample eines Layers auf n_samples Punkte entlang der Bogenlänge in XY."""
    x_layer = x[slc]
    y_layer = y[slc]

    if len(x_layer) < 2:
        # zu wenige Punkte zum Resamplen
        s = np.array([0.0])
        return np.repeat(x_layer, n_samples), np.repeat(y_layer, n_samples)

    dx = np.diff(x_layer)
    dy = np.diff(y_layer)
    ds = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    # Einheitliches s-Raster
    s_new = np.linspace(0, s[-1], n_samples)
    x_new = np.interp(s_new, s, x_layer)
    y_new = np.interp(s_new, s, y_layer)
    return x_new, y_new


def compute_centerline_and_spread(x, y, layer_slices, n_samples=200):
    """Resample alle Layer, bilde Mittellinie und Streuung."""
    layer_x_resampled = []
    layer_y_resampled = []

    for slc in layer_slices:
        xr, yr = resample_layer_xy(x, y, slc, n_samples)
        layer_x_resampled.append(xr)
        layer_y_resampled.append(yr)

    layer_x_resampled = np.stack(layer_x_resampled, axis=0)  # [n_layer, n_samples]
    layer_y_resampled = np.stack(layer_y_resampled, axis=0)

    center_x = layer_x_resampled.mean(axis=0)
    center_y = layer_y_resampled.mean(axis=0)

    # Radiale Abweichung jedes Layers von der Mittellinie
    dx = layer_x_resampled - center_x[None, :]
    dy = layer_y_resampled - center_y[None, :]
    r = np.sqrt(dx**2 + dy**2)
    spread = r.std(axis=0)  # Standardabweichung des radialen Fehlers

    return center_x, center_y, spread, layer_x_resampled, layer_y_resampled


def plot_layers_and_centerline(x, y, layer_slices,
                               center_x, center_y, spread,
                               layer_x_resampled, layer_y_resampled,
                               spread_scale=2.0, output=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Original-Layer (dünn grau)
    for slc in layer_slices:
        ax.plot(x[slc], y[slc], linewidth=0.7, alpha=0.4, linestyle="-", zorder=1)

    # Resample-Layer (optional, leicht dunkler, um zu sehen, was gemittelt wurde)
    # Kannst du auch auskommentieren, wenn dir das zu viel ist.
    # for i in range(layer_x_resampled.shape[0]):
    #     ax.plot(layer_x_resampled[i], layer_y_resampled[i],
    #             linewidth=0.8, alpha=0.5, linestyle=":", zorder=2)

    # Mittellinie (schwarz gestrichelt)
    ax.plot(center_x, center_y, "k--", linewidth=1.5, label="Mittellinie", zorder=3)

    # Streuungs-Band: Kreise um die Mittellinie
    # Radius = spread_scale * spread (z.B. 2 * sigma)
    for cx, cy, s in zip(center_x, center_y, spread):
        if s <= 0:
            continue
        radius = spread_scale * s
        circle = Circle((cx, cy), radius,
                        edgecolor="none",
                        facecolor="gray",
                        alpha=0.15,
                        zorder=2)
        ax.add_patch(circle)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("TCP-Bahn mit Layern, Mittellinie und Streuungsband")
    ax.legend()
    ax.grid(True)

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot TCP Position in XY-Ebene aus ROS-Bag mit Layer-Erkennung."
    )
    parser.add_argument(
        "--topic",
        default="/mur620c/UR10_r/global_tcp_pose_mocap",
        help="Topic mit PoseStamped-Nachrichten (default: %(default)s)",
    )
    parser.add_argument(
        "--z_jump_threshold",
        type=float,
        default=0.4e-3,  # z.B. 0.4 mm
        help="Schwellwert in m für Z-Sprung zur Layer-Erkennung (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Anzahl Resample-Punkte entlang der Bahn (default: %(default)s)",
    )
    parser.add_argument(
        "--spread_scale",
        type=float,
        default=2.0,
        help="Faktor für den Streuungsradius (z.B. 2*sigma) (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        help="Optionaler Dateiname zum Speichern des Plots (PNG)",
    )

    args = parser.parse_args()

    print(f"Lese Bag: {bagfile}")
    x, y, z = read_poses_from_bag(bagfile, args.topic)
    print(f"Anzahl empfangener Posen: {len(x)}")

    if len(x) == 0:
        print("Keine Daten im angegebenen Topic gefunden.")
        return

    layer_slices = split_layers_by_z(z, args.z_jump_threshold)
    print(f"Erkannte Layer: {len(layer_slices)}")
    for i, slc in enumerate(layer_slices):
        print(f"  Layer {i}: Indizes {slc.start} bis {slc.stop} (n={slc.stop - slc.start})")

    center_x, center_y, spread, layer_x_resampled, layer_y_resampled = \
        compute_centerline_and_spread(x, y, layer_slices, n_samples=args.samples)

    plot_layers_and_centerline(
        x, y, layer_slices,
        center_x, center_y, spread,
        layer_x_resampled, layer_y_resampled,
        spread_scale=args.spread_scale,
        output=args.output,
    )


if __name__ == "__main__":
    main()
