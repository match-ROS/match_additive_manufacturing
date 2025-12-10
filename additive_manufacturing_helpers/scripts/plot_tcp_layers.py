#!/usr/bin/env python3
import argparse
import numpy as np
import rosbag
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#bag_path = "record_20251210_141133_MuR.bag"
bag_path = "record_20251209_171429_GUI-PC.bag"
bag_tcp = "record_20251209_171429_GUI-PC.bag"

def read_poses_from_bag(bag_path, topic):
    xs, ys, zs = [], [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            p = msg.pose.position
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
    return np.array(xs), np.array(ys), np.array(zs)


def split_layers_by_start_proximity(x, y, start_radius, min_points_between_layers):
    """
    Teilt die Zeitreihe in Layer auf, basierend darauf,
    dass die Bahn immer wieder in die Nähe der Startposition kommt.

    - Startpunkt = (x[0], y[0])
    - start_radius: Abstand, bei dem wir „nahe am Start“ sind
    - min_points_between_layers: Mindestanzahl Punkte zwischen zwei Lagenwechseln
    """
    n = len(x)
    if n == 0:
        return []

    x0, y0 = x[0], y[0]
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)

    inside = r < start_radius
    start_indices = [0]  # erster Layer startet bei 0

    was_outside = False

    for i in range(1, n):
        if not inside[i]:
            was_outside = True
        else:
            # wir sind im Startbereich
            if was_outside:
                # wir kommen gerade von draußen wieder rein
                if (i - start_indices[-1]) >= min_points_between_layers:
                    start_indices.append(i)
                was_outside = False

    layers = []
    for k in range(len(start_indices) - 1):
        layers.append(slice(start_indices[k], start_indices[k + 1]))
    layers.append(slice(start_indices[-1], n))

    return layers


def resample_layer_xy(x, y, slc, n_samples):
    """Resample eines Layers auf n_samples Punkte entlang der Bogenlänge in XY."""
    x_layer = x[slc]
    y_layer = y[slc]

    if len(x_layer) < 2:
        return np.repeat(x_layer, n_samples), np.repeat(y_layer, n_samples)

    dx = np.diff(x_layer)
    dy = np.diff(y_layer)
    ds = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(ds)))

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

    dx = layer_x_resampled - center_x[None, :]
    dy = layer_y_resampled - center_y[None, :]
    r = np.sqrt(dx**2 + dy**2)
    spread = r.std(axis=0)

    return center_x, center_y, spread, layer_x_resampled, layer_y_resampled


# ---------- UR-Pfad (Groundtruth) einlesen ----------

def split_path_layers_by_z(z, z_jump_threshold):
    """
    Teilt den UR-Pfad in Layer auf, basierend auf Z-Sprüngen.
    Eignet sich gut für /ur_path_original mit diskreten Layerhöhen.
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


def read_ur_path_layer2(bag_path, topic="/ur_path_original", z_jump_threshold=0.0005):
    """
    Liest /ur_path_original aus einer Bag und gibt den zweiten Layer (XY) zurück.
    Falls weniger als 2 Layer vorhanden sind, wird der gesamte Pfad zurückgegeben.
    """
    print(f"Lese UR-Pfad aus Bag: {bag_path}")
    with rosbag.Bag(bag_path, "r") as bag:
        path_msg = None
        for _, msg, _ in bag.read_messages(topics=[topic]):
            path_msg = msg  # letzte Nachricht verwenden

    if path_msg is None:
        print("  Keine /ur_path_original Nachricht gefunden.")
        return None, None

    if not path_msg.poses:
        print("  ur_path_original enthält keine Posen.")
        return None, None

    xs, ys, zs = [], [], []
    for ps in path_msg.poses:
        p = ps.pose.position
        xs.append(p.x)
        ys.append(p.y)
        zs.append(p.z)

    x = np.array(xs)
    y = np.array(ys)
    z = np.array(zs)

    layer_slices = split_path_layers_by_z(z, z_jump_threshold)
    print(f"  UR-Pfad: erkannte Layer: {len(layer_slices)}")

    if len(layer_slices) >= 2:
        sl = layer_slices[1]  # zweiter Layer
        print(f"  Nutze UR-Pfad Layer 2: Indizes {sl.start} bis {sl.stop}")
        return x[sl], y[sl]
    else:
        print("  UR-Pfad hat weniger als 2 Layer – verwende gesamten Pfad.")
        return x, y


# ---------- Plot ----------

def plot_layers_and_centerline(x, y, layer_slices,
                               center_x, center_y, spread,
                               layer_x_resampled, layer_y_resampled,
                               spread_scale=2.0,
                               gt_x=None, gt_y=None,
                               output=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Original-Layer (dünn grau)
    for slc in layer_slices:
        ax.plot(x[slc], y[slc],
                linewidth=0.7, alpha=0.4,
                linestyle="-", zorder=1)

    # Mittellinie (schwarz gestrichelt)
    ax.plot(center_x, center_y, "k--",
            linewidth=1.5, label="Mittellinie", zorder=3)

    # Streuungs-Band: Kreise um die Mittellinie
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

    # Groundtruth-Pfad (UR-Pfad, Layer 2)
    if gt_x is not None and gt_y is not None:
        ax.plot(gt_x, gt_y,
                linewidth=1.5,
                linestyle="-",
                label="UR-Pfad (Layer 2)",
                zorder=4)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("TCP-Bahn mit Layern, Mittellinie, Streuungsband und UR-Pfad")
    ax.legend()
    ax.grid(True)

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot TCP Position in XY-Ebene mit Layer-Erkennung + UR-Pfad-Groundtruth."
    )
    parser.add_argument(
        "--topic",
        default="/mur620c/UR10_r/global_tcp_pose_mocap",
        help="Topic mit PoseStamped-Nachrichten (TCP-Mocap) (default: %(default)s)",
    )
    parser.add_argument(
        "--start_radius",
        type=float,
        default=0.005,  # 5 mm
        help="Radius um die Startposition zur Erkennung eines Lagenwechsels (default: %(default)s)",
    )
    parser.add_argument(
        "--min_points_between_layers",
        type=int,
        default=200,
        help="Mindestanzahl Punkte zwischen zwei Lagenwechseln (default: %(default)s)",
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
        "--path_z_jump_threshold",
        type=float,
        default=0.005,
        help="Z-Sprungschwelle für Layer-Trennung im UR-Pfad (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="layer_plot.pdf",
        help="Dateiname zum Speichern des Plots (z.B. .pdf oder .png) (default: %(default)s)",
    )

    args = parser.parse_args()

    tcp_bag = bag_tcp
    path_bag = bag_path if bag_path is not None else bag_tcp

    # --- TCP-Daten einlesen ---
    print(f"Lese TCP-Bag: {tcp_bag}")
    x, y, z = read_poses_from_bag(tcp_bag, args.topic)
    print(f"Anzahl empfangener Posen: {len(x)}")

    if len(x) == 0:
        print("Keine Daten im angegebenen Topic gefunden.")
        return

    layer_slices = split_layers_by_start_proximity(
        x, y,
        start_radius=args.start_radius,
        min_points_between_layers=args.min_points_between_layers,
    )

    print(f"Erkannte Layer (inkl. erster/letzter): {len(layer_slices)}")

    # Ersten und letzten Layer verwerfen
    if len(layer_slices) > 2:
        layer_slices = layer_slices[1:-1]
    else:
        print("Zu wenig Layer, um ersten und letzten zu verwerfen.")

    print(f"Verbleibende Layer: {len(layer_slices)}")
    for i, slc in enumerate(layer_slices):
        print(f"  Layer {i}: Indizes {slc.start} bis {slc.stop} (n={slc.stop - slc.start})")

    center_x, center_y, spread, layer_x_resampled, layer_y_resampled = \
        compute_centerline_and_spread(x, y, layer_slices, n_samples=args.samples)

    # --- UR-Pfad (Groundtruth, Layer 2) einlesen ---
    gt_x, gt_y = read_ur_path_layer2(
        path_bag,
        topic="/ur_path_original",
        z_jump_threshold=args.path_z_jump_threshold,
    )

    # --- Plot ---
    plot_layers_and_centerline(
        x, y, layer_slices,
        center_x, center_y, spread,
        layer_x_resampled, layer_y_resampled,
        spread_scale=args.spread_scale,
        gt_x=gt_x, gt_y=gt_y,
        output=args.output,
    )


if __name__ == "__main__":
    main()
