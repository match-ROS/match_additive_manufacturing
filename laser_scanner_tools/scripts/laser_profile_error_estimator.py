#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Float32


class LaserProfileErrorEstimator(object):
    def __init__(self):
        rospy.init_node("laser_profile_error_estimator", anonymous=True)

        # --- Parameters ---
        self.profile_topic = rospy.get_param("~profile_topic", "/profiles_float")

        # Ziel-Schichthöhe
        self.target_layer_height = rospy.get_param("~target_layer_height", -10.0)  # z.B. mm
        self.height_half_width = rospy.get_param("~height_half_width", 50)         # Punkte links/rechts

        # Scanner-zu-Düse-Geometrie / Zeitversatz
        self.distance_scanner_to_nozzle = rospy.get_param("~distance_scanner_to_nozzle", 0.05)  # m
        self.target_speed = rospy.get_param("~target_speed", 0.02)                              # m/s
        self.profile_rate = rospy.get_param("~profile_rate", 40.0)                             # Hz

        # Filterparameter
        self.filter_half_width = rospy.get_param("~filter_half_width", 5)       # für Höhenfehler (symmetrisch)
        self.lateral_ma_window = rospy.get_param("~lateral_ma_window", 5)       # gleitender Mittelwert lateral
        self.min_expected_height = rospy.get_param("~min_expected_height", -30.0)

        # Geschwindigkeitsoverride
        self.velocity_override_topic = rospy.get_param(
            "~velocity_override_topic", "/velocity_override"
        )

        # --- Publisher ---
        self.lateral_error_pub = rospy.Publisher("/laser_profile/lateral_error", Float32, queue_size=1)
        self.height_error_pub = rospy.Publisher("/laser_profile/height_error", Float32, queue_size=1)

        # --- State ---
        self.velocity_override = 1.0
        self.raw_lateral = []   # ungefilterte laterale Fehler (Index)
        self.raw_height = []    # ungefilterte Höhenfehler (target - ist)

        # --- Subscriber ---
        rospy.Subscriber(self.profile_topic, Float32MultiArray,
                         self.profile_callback, queue_size=1)
        rospy.Subscriber(self.velocity_override_topic, Float32,
                         self.velocity_override_callback, queue_size=1)

        rospy.loginfo("LaserProfileErrorEstimator started.")
        rospy.spin()

    # ------------------------------------------------------------------

    def velocity_override_callback(self, msg: Float32):
        self.velocity_override = max(msg.data, 0.0)

    def compute_delay_steps(self):
        """Verzögerung in Profil-Zyklen (nur für Höhe)."""
        v_eff = self.target_speed * max(self.velocity_override, 1e-3)  # 0 vermeiden
        t_delay = self.distance_scanner_to_nozzle / v_eff              # s
        steps = int(round(t_delay * self.profile_rate))
        return max(0, steps)

    def symmetric_filtered_value(self, buffer, center_idx):
        """Symmetrische Mittelung um center_idx (für phasenarmen Höhen-Filter)."""
        if center_idx < 0 or center_idx >= len(buffer):
            return None

        start = max(0, center_idx - self.filter_half_width)
        end = min(len(buffer) - 1, center_idx + self.filter_half_width)
        window = buffer[start:end + 1]
        if not window:
            return None
        return float(np.mean(window))

    # ------------------------------------------------------------------

    def profile_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.size == 0:
            return

        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            rospy.logwarn_throttle(1.0, "No valid laser points")
            return

        valid_values = data[valid_mask]
        valid_indices = np.nonzero(valid_mask)[0]

        # Maximaler Wert (wie bisher)
        min_idx_rel = np.argmax(valid_values)
        min_idx = int(valid_indices[min_idx_rel])

        center_idx = (data.size - 1) / 2.0
        lateral_raw = float(min_idx - center_idx)  # Index-Abweichung

        # --- Höhe aus Mittel der Punkte links/rechts vom Maximum ---
        ist_height = None
        hw = int(self.height_half_width)

        if min_idx - hw >= 0 and min_idx + hw < data.size:
            left = data[min_idx - hw:min_idx]
            right = data[min_idx + 1:min_idx + 1 + hw]
            both = np.concatenate([left, right])
            both = both[np.isfinite(both)]
            if both.size > 0:
                ist_height = float(np.mean(both))

        # Fallback, falls zu randnah / keine gültigen Werte
        if ist_height is None:
            ist_height = float(data[min_idx])

        height_error_raw = float(self.target_layer_height - ist_height)

        # --- Rohwerte in Buffer schieben ---
        self.raw_lateral.append(lateral_raw)
        self.raw_height.append(height_error_raw)

        # --- LATERAL: nur gleitender Mittelwert, KEIN künstlicher Delay ---
        if self.lateral_ma_window > 1 and len(self.raw_lateral) >= 1:
            w = min(self.lateral_ma_window, len(self.raw_lateral))
            lat_filtered = float(np.mean(self.raw_lateral[-w:]))
        else:
            lat_filtered = lateral_raw

        # --- HÖHE: Delay + symmetrischer Filter über mehrere Profile ---
        delay_steps = self.compute_delay_steps()
        idx_center = len(self.raw_height) - 1 - delay_steps

        h_filtered = None
        if idx_center >= 0:
            h_filtered = self.symmetric_filtered_value(self.raw_height, idx_center)

        # --- Seitlichen Fehler sofort publishen ---
        self.lateral_error_pub.publish(Float32(data=lat_filtered))

        # Höhe nur publishen, wenn genügend Daten für Delay/Filter da sind
        if h_filtered is None:
            return

        current_height = self.target_layer_height - h_filtered
        if current_height < self.min_expected_height:
            rospy.logwarn_throttle(
                5.0,
                "Profile point too low: %.2f < %.2f",
                current_height,
                self.min_expected_height
            )
            # Seitliche Korrektur bleibt, Höhenfehler kannst du trotzdem loggen
            # (h_filtered wird trotzdem publiziert)

        self.height_error_pub.publish(Float32(data=h_filtered))


if __name__ == "__main__":
    try:
        LaserProfileErrorEstimator()
    except rospy.ROSInterruptException:
        pass
