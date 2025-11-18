#!/usr/bin/env python3
import rosbag
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt

def calculate_velocity_in_xy_plane(bag_file_path):
    # ROS-Bag öffnen
    bag = rosbag.Bag(bag_file_path, 'r')
    print(bag)

    # Listen für Positionen und Zeitstempel
    positions = []
    timestamps = []

    # Das gewünschte Topic abonnieren
    for topic, msg, t in bag.read_messages(topics='/qualisys/karton/pose'):
        # Position (x, y) extrahieren und Zeitstempel speichern
        x = msg.pose.position.x
        y = msg.pose.position.y
        positions.append((x, y))
        timestamps.append(t.to_sec())  # Zeitstempel in Sekunden

    bag.close()

    print("Anzahl der Positionen:", len(positions))

    # Berechnung der Geschwindigkeit in der XY-Ebene
    velocities = []
    for i in range(1, len(positions)):
        # Positionen und Zeitdifferenz berechnen
        x1, y1 = positions[i-1]
        x2, y2 = positions[i]
        t1 = timestamps[i-1]
        t2 = timestamps[i]

        # Berechnung der Differenzen
        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_t = t2 - t1

        if delta_t > 0:
            # Berechnung der Geschwindigkeit (Betrag der Geschwindigkeit in XY-Ebene)
            speed = np.sqrt((delta_x / delta_t) ** 2 + (delta_y / delta_t) ** 2)
            velocities.append(speed)

    return timestamps[1:], velocities  # Rückgabe der Zeit und Geschwindigkeit

def smooth_velocity(velocities, window_size=20):
    # Einfache gleitende Durchschnittsglättung
    return np.convolve(velocities, np.ones(window_size)/window_size, mode='valid')

# Beispielaufruf der Funktion
bag_file_path = '/home/rosmatch/2025-11-18-15-59-21.bag'
timestamps, velocities = calculate_velocity_in_xy_plane(bag_file_path)

# Glättung der Geschwindigkeit
smoothed_velocities = smooth_velocity(velocities, window_size=20)

# Die Zeitstempel für den geglätteten Verlauf anpassen (wegen des gleitenden Durchschnitts)
smoothed_timestamps = timestamps[:len(smoothed_velocities)]

# Plotten der Geschwindigkeit über die Zeit
plt.figure(figsize=(10, 6))
plt.plot(smoothed_timestamps, smoothed_velocities, label='Geglättete Geschwindigkeit in XY-Ebene', color='b')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Geschwindigkeit (m/s)')
plt.title('Geglättete Geschwindigkeit des Kartons in der XY-Ebene')
plt.legend()
plt.grid(True)
plt.show()
