#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

class MirTimestampWarping(object):
    def __init__(self):
        rospy.init_node("mir_timestamp_warping")

        # Parameter
        self.max_ur_mir_distance = rospy.get_param("~max_ur_mir_distance", 1.2)   # [m]
        self.curvature_gain_max  = rospy.get_param("~curvature_gain_max", 5.0)    # Obergrenze der "Kurvenverlangsamung"
        self.timer_period        = rospy.get_param("~timer_period", 0.5)          # [s]

        self.mir_path = None      # nav_msgs/Path
        self.ur_path  = None      # nav_msgs/Path
        self.t_orig   = None      # np.array der Original-Timestamps
        self.t_mod    = None      # np.array der modifizierten Timestamps

        # Subscriber
        rospy.Subscriber("/mir_path_original", Path, self.cb_mir_path, queue_size=1)
        rospy.Subscriber("/ur_path_original",  Path, self.cb_ur_path, queue_size=1)
        rospy.Subscriber("/mir_path_timestamps", Float32MultiArray, self.cb_timestamps, queue_size=1)

        # Publisher (gelatched, damit neue Subscriber die Daten bekommen)
        self.pub_t_mod = rospy.Publisher(
            "/mir_path_timestamps_modified",
            Float32MultiArray,
            queue_size=1,
            latch=True
        )

        # Timer, der wartet bis alle Daten da sind und dann einmalig rechnet
        self.computation_done = False
        rospy.Timer(rospy.Duration(self.timer_period), self.on_timer)

    # ---------------------------- Callbacks ----------------------------

    def cb_mir_path(self, msg):
        self.mir_path = msg

    def cb_ur_path(self, msg):
        self.ur_path = msg

    def cb_timestamps(self, msg):
        # Annahme: vollständiger Zeitstempel-Vektor für die MiR
        self.t_orig = np.array(msg.data, dtype=float)

    # ---------------------------- Hilfsfunktionen ----------------------------

    @staticmethod
    def _truncate_to_common_length(path_a, path_b, t_arr):
        """Alle drei auf die gleiche Länge kürzen (falls nötig)."""
        n_a = len(path_a.poses)
        n_b = len(path_b.poses)
        n_t = len(t_arr)
        n   = min(n_a, n_b, n_t)
        return n

    @staticmethod
    def compute_path_curvature(path_msg):
        """
        Krümmung pro Pfadpunkt (2D in x/y).
        Erste und letzte Stelle werden 0 gesetzt.
        """
        pts = path_msg.poses
        N = len(pts)
        if N < 3:
            return np.zeros(N)

        curvature = np.zeros(N)

        for i in range(1, N - 1):
            p_prev = np.array([pts[i-1].pose.position.x,
                               pts[i-1].pose.position.y])
            p      = np.array([pts[i].pose.position.x,
                               pts[i].pose.position.y])
            p_next = np.array([pts[i+1].pose.position.x,
                               pts[i+1].pose.position.y])

            a = np.linalg.norm(p - p_prev)
            b = np.linalg.norm(p_next - p)
            c = np.linalg.norm(p_next - p_prev)

            if a < 1e-6 or b < 1e-6 or c < 1e-6:
                curvature[i] = 0.0
                continue

            s = 0.5 * (a + b + c)
            A = max(0.0, s * (s - a) * (s - b) * (s - c)) ** 0.5

            curvature[i] = 4.0 * A / (a * b * c)

        return curvature

    @staticmethod
    def idx_for_time(t_arr, T):
        """
        Index der letzten Zeit <= T.
        """
        idx = np.searchsorted(t_arr, T, side="right") - 1
        if idx < 0:
            idx = 0
        if idx >= len(t_arr):
            idx = len(t_arr) - 1
        return idx

    def build_timestamps_for_gain(self, gain, curvature, t_orig):
        """
        Baut einen neuen Zeitstempelvektor basierend auf:
        - Original-Zeitdifferenzen
        - Krümmungsgewichtung
        - 'gain' für die Verstärkung in Kurven
        Gesamtfahrzeit bleibt erhalten.
        """
        t_orig = np.asarray(t_orig, dtype=float)
        N = len(t_orig)
        if N < 2:
            return t_orig.copy()

        # Segmentweise Δt
        dt_orig = np.diff(t_orig)

        # Normierte Krümmung (0..1)
        max_c = np.max(curvature)
        if max_c <= 1e-8:
            # Kein nennenswerter Kurvenanteil
            return t_orig.copy()

        c_norm = curvature / max_c

        # Segmentgewichtung: Krümmung am Zielpunkt des Segments (i)
        weights = 1.0 + gain * c_norm[1:]
        weights = np.maximum(weights, 1e-3)

        # Gesamtzeit beibehalten
        T_total = t_orig[-1] - t_orig[0]
        dt_hat = dt_orig * weights
        sum_hat = np.sum(dt_hat)
        if sum_hat <= 1e-9:
            return t_orig.copy()

        scale = T_total / sum_hat
        dt_mod = dt_hat * scale

        t_mod = np.empty_like(t_orig)
        t_mod[0] = t_orig[0]
        t_mod[1:] = t_mod[0] + np.cumsum(dt_mod)
        return t_mod

    def max_ur_mir_separation(self, t_orig, t_mod, mir_path, ur_path):
        """
        Maximaler XY-Abstand zwischen UR (Original-Zeitplan t_orig)
        und MiR (modifizierter Zeitplan t_mod) über die gemeinsame Fahrzeit.
        """
        t_orig = np.asarray(t_orig, dtype=float)
        t_mod  = np.asarray(t_mod, dtype=float)

        # Falls die Arrays nicht gleich lang sind, vorher kürzen:
        n = self._truncate_to_common_length(mir_path, ur_path, t_orig)
        t_orig = t_orig[:n]
        t_mod  = t_mod[:n]
        mir_pts = mir_path.poses[:n]
        ur_pts  = ur_path.poses[:n]

        # Zeitpunkte, an denen wir prüfen (Vereinigung der Stützstellen)
        times = np.unique(np.concatenate((t_orig, t_mod)))
        if len(times) == 0:
            return 0.0

        max_d = 0.0
        for T in times:
            i_m = self.idx_for_time(t_mod, T)
            i_u = self.idx_for_time(t_orig, T)

            pm = mir_pts[i_m].pose.position
            pu = ur_pts[i_u].pose.position

            dx = pm.x - pu.x
            dy = pm.y - pu.y
            d = np.hypot(dx, dy)

            if d > max_d:
                max_d = d

        return max_d

    def compute_modified_timestamps(self):
        """
        Hauptlogik:
        - Krümmung berechnen
        - über Gain (0..curvature_gain_max) suchen
        - größtmöglichen Gain mit Distanz<=max_ur_mir_distance wählen
        """
        if self.mir_path is None or self.ur_path is None or self.t_orig is None:
            return None

        # Auf gemeinsame Länge kürzen
        n = self._truncate_to_common_length(self.mir_path, self.ur_path, self.t_orig)
        if n < 3:
            rospy.logwarn("Pfad zu kurz für sinnvolle Krümmungsanpassung. Verwende Original-Timestamps.")
            return self.t_orig[:n].copy()

        t_orig = self.t_orig[:n].copy()

        # Krümmung der MiR-Bahn
        curvature = self.compute_path_curvature(self.mir_path)
        curvature = curvature[:n]

        if np.max(curvature) <= 1e-8:
            # praktisch gerade Strecke
            rospy.loginfo("Krümmung ~0, verwende Original-Timestamps.")
            return t_orig

        # Falls selbst bei maximalem Gain die Distanz ok ist, nehmen wir ihn
        t_max_gain = self.build_timestamps_for_gain(self.curvature_gain_max, curvature, t_orig)
        d_max_gain = self.max_ur_mir_separation(t_orig, t_max_gain, self.mir_path, self.ur_path)
        rospy.loginfo("Max-Gain Trennung: %.3f m", d_max_gain)

        if d_max_gain <= self.max_ur_mir_distance:
            rospy.loginfo("Verwende maximalen Gain = %.3f", self.curvature_gain_max)
            return t_max_gain

        # Sonst binäre Suche in [0, curvature_gain_max]
        low = 0.0
        high = self.curvature_gain_max
        best_gain = 0.0
        best_t    = t_orig.copy()

        for _ in range(20):  # 2^-20 ~ 1e-6 Auflösung
            mid = 0.5 * (low + high)
            t_mid = self.build_timestamps_for_gain(mid, curvature, t_orig)
            d_mid = self.max_ur_mir_separation(t_orig, t_mid, self.mir_path, self.ur_path)

            if d_mid <= self.max_ur_mir_distance:
                best_gain = mid
                best_t    = t_mid
                low = mid
            else:
                high = mid

        rospy.loginfo("Gewählter Gain = %.4f, max UR-MiR-Distanz = %.3f m",
                      best_gain,
                      self.max_ur_mir_separation(t_orig, best_t, self.mir_path, self.ur_path))

        return best_t

    # ---------------------------- Timer ----------------------------

    def on_timer(self, event):
        if self.computation_done:
            return

        if self.mir_path is None or self.ur_path is None or self.t_orig is None:
            # Warten bis alle Daten da sind
            return

        rospy.loginfo("Alle Daten verfügbar, berechne modifizierte Timestamps...")
        t_mod = self.compute_modified_timestamps()
        if t_mod is None:
            rospy.logwarn("Konnte keine modifizierten Timestamps berechnen.")
            self.computation_done = True
            return

        self.t_mod = t_mod

        msg = Float32MultiArray()
        msg.data = t_mod.tolist()
        self.pub_t_mod.publish(msg)
        rospy.loginfo("Publiziere /mir_path_timestamps_modified mit %d Einträgen.", len(t_mod))

        self.computation_done = True


if __name__ == "__main__":
    MirTimestampWarping()
    rospy.spin()
