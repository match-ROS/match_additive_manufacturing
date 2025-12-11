#!/usr/bin/env python3
import rosbag
from geometry_msgs.msg import PoseStamped, Twist
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

    # mittlere Schrittweite als dt
    delta_t = np.mean(np.diff(timestamps_smooth))

    # --- 2) Geschwindigkeit aus geglätteten Positionen ableiten ---
    velocities = []
    vel_timestamps = []

    for i in range(1, len(x_smooth)):
        x1, y1 = x_smooth[i-1], y_smooth[i-1]
        x2, y2 = x_smooth[i],   y_smooth[i]
        t2 = timestamps_smooth[i]

        delta_x = x2 - x1
        delta_y = y2 - y1

        if delta_t > 0:
            speed = np.sqrt((delta_x / delta_t) ** 2 + (delta_y / delta_t) ** 2)
            velocities.append(speed)
            vel_timestamps.append(t2)  # Zeit zum zweiten Punkt

    return np.array(vel_timestamps), np.array(velocities)

def read_offset_velocities(bag_file_path, topic='/laser_profile_offset_cmd_vel'):
    """
    Liest die seitliche Korrekturgeschwindigkeit aus dem zweiten Bag
    und berechnet den Betrag der linearen Geschwindigkeit.
    """
    bag = rosbag.Bag(bag_file_path, 'r')

    timestamps = []
    speeds = []

    for tpc, msg, t in bag.read_messages(topics=topic):
        # Betrag der linearen Geschwindigkeit (anderes KS -> nur Norm interessiert)
        v = np.sqrt(
            msg.linear.x**2 +
            msg.linear.y**2 +
            msg.linear.z**2
        )
        timestamps.append(t.to_sec())
        speeds.append(v)

    bag.close()

    print("Anzahl der Offset-Velocity-Samples:", len(speeds))
    return np.array(timestamps), np.array(speeds)

def interpolate_and_subtract(vel_timestamps, velocities,
                             offset_timestamps, offset_speeds):
    """
    Interpoliert die Offset-Geschwindigkeit auf die Zeitbasis von vel_timestamps
    und zieht sie von den TCP-Geschwindigkeiten ab.
    """
    # Nur den gemeinsamen Zeitraum betrachten
    t_min = max(vel_timestamps[0], offset_timestamps[0])
    t_max = min(vel_timestamps[-1], offset_timestamps[-1])

    mask = (vel_timestamps >= t_min) & (vel_timestamps <= t_max)

    vel_ts_common = vel_timestamps[mask]
    vel_common = velocities[mask]

    # Offset-Geschwindigkeit auf vel_ts_common interpolieren
    offset_interp = np.interp(vel_ts_common, offset_timestamps, offset_speeds)

    # Korrigierte Geschwindigkeit (negativ vermeiden)
    corrected_speed = np.sqrt(np.clip(vel_common**2 - offset_interp**2, 0.0, None))
    corrected_speed = np.clip(corrected_speed, 0.0, None)

    return vel_ts_common, vel_common, offset_interp, corrected_speed

# --- Beispielaufruf ---

bag_tcp_path    = 'record_20251210_141133_GUI-PC.bag'   # TCP-Pose / Mocap
bag_offset_path = 'record_20251210_141133_MuR.bag'      # Offset-Vel / Twist

# Fenstergröße für Positions-Glättung (anpassen nach Bedarf)
window_size_pos = 200

# 1) TCP-Geschwindigkeit aus Bag 1
vel_timestamps, velocities = calculate_velocity_in_xy_plane(
    bag_tcp_path,
    window_size_pos=window_size_pos
)

print("Anzahl der Geschwindigkeitswerte (TCP):", len(velocities))

# 2) Offset-Geschwindigkeiten aus Bag 2
offset_ts, offset_speeds = read_offset_velocities(bag_offset_path)

# 3) Interpolation + Subtraktion
common_ts, vel_common, offset_interp, corrected_speed = interpolate_and_subtract(
    vel_timestamps, velocities, offset_ts, offset_speeds
)

# 4) Plot
plt.figure(figsize=(10, 6))
#plt.plot(vel_timestamps, velocities, label='TCP speed XY (roh)')
#plt.plot(common_ts, offset_interp, label='|offset vel| (interpoliert)')
plt.plot(common_ts, corrected_speed, label='TCP speed - |offset vel|')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Geschwindigkeit (m/s)')
plt.title('TCP-Geschwindigkeit mit Abzug der seitlichen Korrektur')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
