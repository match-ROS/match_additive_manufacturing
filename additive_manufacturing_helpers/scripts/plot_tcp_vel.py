#!/usr/bin/env python3
import rosbag
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt

def smooth_positions(positions, window_size=20):
    """
    Gleicht x- und y-Position mit gleitendem Mittelwert.
    positions: Liste von (x, y)
    """
    positions = np.array(positions)  # shape: (N, 2)
    x = positions[:, 0]
    y = positions[:, 1]

    kernel = np.ones(window_size) / window_size

    x_smooth = np.convolve(x, kernel, mode='valid')
    y_smooth = np.convolve(y, kernel, mode='valid')

    # Die zugehörigen Zeitstempel müssen entsprechend gekürzt werden
    return x_smooth, y_smooth

def calculate_velocity_in_xy_plane(bag_file_path, window_size_pos=100):
    bag = rosbag.Bag(bag_file_path, 'r')
    print(bag)

    positions = []
    timestamps = []

    for topic, msg, t in bag.read_messages(topics='/mur620c/UR10_r/global_tcp_pose_mocap'):
        x = msg.pose.position.x
        y = msg.pose.position.y
        positions.append((x, y))
        timestamps.append(t.to_sec())

    bag.close()

    print("Anzahl der Positionen:", len(positions))
    if len(positions) < window_size_pos + 1:
        raise RuntimeError("Zu wenige Samples für das Glättungsfenster.")

    # --- 1) Positionen glätten ---
    x_smooth, y_smooth = smooth_positions(positions, window_size=window_size_pos)

    # Zeitstempel zu den geglätteten Positionen (align: Ende des Fensters)
    timestamps_smooth = np.array(timestamps[window_size_pos - 1:])

    # --- 2) Geschwindigkeit aus geglätteten Positionen ableiten ---
    velocities = []
    vel_timestamps = []

    for i in range(1, len(x_smooth)):
        x1, y1 = x_smooth[i-1], y_smooth[i-1]
        x2, y2 = x_smooth[i],   y_smooth[i]
        t1 = timestamps_smooth[i-1]
        t2 = timestamps_smooth[i]

        delta_x = x2 - x1
        delta_y = y2 - y1
        #delta_t = t2 - t1
        delta_t = 1.0/595.5

        if delta_t > 0:
            speed = np.sqrt((delta_x / delta_t) ** 2 + (delta_y / delta_t) ** 2)
            velocities.append(speed)
            vel_timestamps.append(t2)  # Zeit zum zweiten Punkt

    return vel_timestamps, velocities

# --- Beispielaufruf ---
bag_file_path = 'record_20251209_164623_GUI-PC.bag'

# Fenstergröße für Positions-Glättung (anpassen nach Bedarf)
window_size_pos = 200

timestamps, velocities = calculate_velocity_in_xy_plane(
    bag_file_path,
    window_size_pos=window_size_pos
)

print("Anzahl der Geschwindigkeitswerte:", len(velocities))

plt.figure(figsize=(10, 6))
plt.plot(timestamps, velocities, label='Geschwindigkeit aus geglätteten Positionen (XY)',)
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Geschwindigkeit (m/s)')
plt.title('Geschwindigkeit des Kartons in der XY-Ebene (Positionsdaten vorgefiltert)')
plt.legend()
plt.grid(True)
plt.show()
