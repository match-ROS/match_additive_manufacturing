#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np

import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

# Optional plotting (headless): saves PNG if enabled
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def yaw_from_quat(q):
    """Yaw from geometry_msgs/Quaternion."""
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def path_to_xyz_yaw(path_msg: Path):
    n = len(path_msg.poses)
    xyz = np.zeros((n, 3), dtype=np.float64)
    yaw = np.zeros((n,), dtype=np.float64)
    for i, ps in enumerate(path_msg.poses):
        p = ps.pose.position
        o = ps.pose.orientation
        xyz[i, :] = [p.x, p.y, p.z]
        yaw[i] = yaw_from_quat(o)
    return xyz, yaw


def interp_along_t(xyz: np.ndarray, t_knots: np.ndarray, t_query: np.ndarray):
    """
    Linear interpolation of xyz along time knots.
    xyz: (N,3), t_knots: (N,), t_query: (M,)
    Returns (M,3).
    """
    # clamp queries to range
    tq = np.clip(t_query, t_knots[0], t_knots[-1])
    out = np.zeros((len(tq), 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(tq, t_knots, xyz[:, d])
    return out


def interp_yaw(yaw: np.ndarray, t_knots: np.ndarray, t_query: np.ndarray):
    """
    Interpolate yaw robustly via sin/cos to avoid +-pi jumps.
    """
    tq = np.clip(t_query, t_knots[0], t_knots[-1])
    s = np.interp(tq, t_knots, np.sin(yaw))
    c = np.interp(tq, t_knots, np.cos(yaw))
    return np.arctan2(s, c)


class LocalRetimingOptimizerNode:
    """
    NEUE VARIANTE:
    - UR path & timing: UR-Timestamps weiter trivial aus MiR-Start/Ende abgeleitet.
    - MiR-Zeitstempel bleiben UNVERÄNDERT (ts0).
    - Es wird ein Index-Offset di[k] optimiert, so dass der effektive Index
      i_eff[k] = k + di[k] bei fixer Zeit ts0[k] gefahren wird.
    - Ziel: Reduktion von Geschwindigkeitsspitzen bei Einhaltung der Reach-Constraint.
    """

    def __init__(self):
        rospy.init_node("mir_index_offset_optimizer", anonymous=False)

        # Topics
        self.mir_path_topic = rospy.get_param("~mir_path_topic", "/mur620c/mir_path_original")
        self.ur_path_topic = rospy.get_param("~ur_path_topic", "/mur620c/ur_path_original")
        self.mir_ts_topic = rospy.get_param("~mir_timestamps_topic", "/mur620c/mir_path_timestamps")

        self.out_ts_topic = rospy.get_param("~out_timestamps_topic", "/mur620c/mir_path_timestamps_optimized")
        self.out_dt_topic = rospy.get_param("~out_deltas_topic", "/mur620c/mir_path_dt_optimized")  # optional

        # Mount offset of UR base in MiR base frame (meters) - only XY used for reach
        # Your given xyz="0.549 -0.318 0.49" -> we use x,y
        self.mount_x = float(rospy.get_param("~ur_mount_x", 0.549))
        self.mount_y = float(rospy.get_param("~ur_mount_y", -0.318))

        # Äquivalenter Radius für Drehbewegung (ca. halber Radabstand)
        self.equiv_radius = float(rospy.get_param("~equiv_radius", 0.45))

        # Maximaler äquivalenter Rotations-"Speed" in m/s
        # verhindert, dass einzelne Dreh-Spikes das Mapping dominieren
        self.max_rot_equiv_speed = float(rospy.get_param("~max_rot_equiv_speed", 0.6))

        # Schwelle, ab der wir sagen: der Roboter "fährt wirklich"
        # (darüber ignorieren wir v_rot und glätten nur v_lin)
        self.rotation_only_threshold = float(
            rospy.get_param("~rotation_only_threshold", 0.02)
        )  # m/s

        # Anzahl der vorderen Pfadpunkte, die von der Index-Optimierung
        # explizit ausgenommen werden sollen (di = 0)
        self.ignore_prefix_points = int(rospy.get_param("~ignore_prefix_points", 100))


        # Constraint (XY only)
        self.reach_xy_max = float(rospy.get_param("~reach_xy_max", 1.30))

        # Optimization params
        self.max_iters = int(rospy.get_param("~max_iters", 1600))

        # Diese Parameter nutzen wir jetzt für die Index-Optimierung:
        self.k_fast_frac = float(rospy.get_param("~k_fast_frac", 0.10))  # top 10% segments
        self.k_slow_frac = float(rospy.get_param("~k_slow_frac", 0.20))  # bottom 20% segments

        # Grenzen im Indexraum
        self.di_max = float(rospy.get_param("~di_max", 1000.0))      # max |index offset|
        self.min_step = float(rospy.get_param("~min_step", 0.1))   # min i_eff-Schritt
        self.max_step = float(rospy.get_param("~max_step", 1.5))   # max i_eff-Schritt

        # Objective
        self.obj_mode = rospy.get_param("~objective", "l2")  # "peak" or "l2"
        self.accept_tol = float(rospy.get_param("~accept_tol", 1e-9))

        # Plotting
        self.save_plot = bool(rospy.get_param("~save_plot", True))
        self.plot_path = rospy.get_param("~plot_path", "/tmp/mir_index_offset_speeds.png")

        # CSV exports
        self.export_csv = bool(rospy.get_param("~export_csv", True))
        self.csv_timestamps_path = rospy.get_param("~csv_timestamps_path", "/tmp/mir_timestamps_original.csv")
        self.csv_index_offset_path = rospy.get_param("~csv_index_offset_path", "/tmp/mir_index_offset.csv")

        # XY path plot (first layer)
        self.xy_plot_path = rospy.get_param("~xy_plot_path", "/tmp/mir_ur_xy_first_layer_index_offset.png")
        self.layer_z_eps = float(rospy.get_param("~layer_z_eps", 1e-4))

        # Publishers (hier weiterhin original timestamps, falls benötigt)
        self.pub_ts = rospy.Publisher(self.out_ts_topic, Float32MultiArray, queue_size=1, latch=True)
        self.pub_dt = rospy.Publisher(self.out_dt_topic, Float32MultiArray, queue_size=1, latch=True)

    # -------------------------------------------------------------------------
    # Allgemeine Utilities
    # -------------------------------------------------------------------------

    def wait_inputs(self):
        rospy.loginfo("Waiting for live topics...")
        mir_path = rospy.wait_for_message(self.mir_path_topic, Path, timeout=None)
        ur_path = rospy.wait_for_message(self.ur_path_topic, Path, timeout=None)
        mir_ts_msg = rospy.wait_for_message(self.mir_ts_topic, Float32MultiArray, timeout=None)

        mir_ts = np.array(mir_ts_msg.data, dtype=np.float64)
        if len(mir_ts) != len(mir_path.poses):
            rospy.logwarn("MiR timestamps length != MiR path length. Truncating to min length.")
            n = min(len(mir_ts), len(mir_path.poses))
            mir_ts = mir_ts[:n]
            mir_path.poses = mir_path.poses[:n]

        if len(ur_path.poses) != len(mir_path.poses):
            rospy.logwarn("UR path length != MiR path length. Proceeding anyway (UR timestamps are built separately).")

        return mir_path, ur_path, mir_ts

    def build_ur_time(self, mir_ts, n_ur):
        # UR timing fixed: same start/end as MiR, uniform steps over UR points
        t0 = float(mir_ts[0])
        t1 = float(mir_ts[-1])
        if n_ur < 2:
            return np.array([t0], dtype=np.float64)
        return np.linspace(t0, t1, n_ur, dtype=np.float64)

    def objective(self, v: np.ndarray) -> float:
        if self.obj_mode == "l2":
            return float(np.mean(v * v))
        return float(np.max(v))

    # -------------------------------------------------------------------------
    # Index-Raum Interpolation
    # -------------------------------------------------------------------------

    def _interp_along_index(self, values: np.ndarray, idx_eff: np.ndarray) -> np.ndarray:
        """
        Lineare Interpolation in Indexrichtung.
        values: (N, D) oder (N,) – Bahn über Index k
        idx_eff: (N,)             – kontinuierlicher Index i_eff[k]
        """
        N = len(values)
        s = np.arange(N, dtype=np.float64)
        idx_clamped = np.clip(idx_eff, 0.0, float(N - 1))

        if values.ndim == 1:
            return np.interp(idx_clamped, s, values.astype(np.float64))

        out = np.zeros((len(idx_eff), values.shape[1]), dtype=np.float64)
        for d in range(values.shape[1]):
            out[:, d] = np.interp(idx_clamped, s, values[:, d].astype(np.float64))
        return out

    def _interp_yaw_along_index(self, yaw: np.ndarray, idx_eff: np.ndarray) -> np.ndarray:
        """Yaw-Interpolation im Indexraum (sin/cos, robust gg. +-pi)."""
        N = len(yaw)
        s = np.arange(N, dtype=np.float64)
        idx_clamped = np.clip(idx_eff, 0.0, float(N - 1))

        s_y = np.interp(idx_clamped, s, np.sin(yaw))
        c_y = np.interp(idx_clamped, s, np.cos(yaw))
        return np.arctan2(s_y, c_y)

    # -------------------------------------------------------------------------
    # Geschwindigkeiten & Reach mit Indexoffset
    # -------------------------------------------------------------------------

    def compute_speeds_with_offset(self,
                                   mir_xyz: np.ndarray,
                                   ts0: np.ndarray,
                                   di: np.ndarray) -> np.ndarray:
        """
        v[k] für effektive Pfadpunkte mit Indexoffset di[k] bei festen Zeiten ts0[k].
        """
        idx_eff = np.arange(len(ts0), dtype=np.float64) + di
        mir_eff_xy = self._interp_along_index(mir_xyz[:, :2], idx_eff)

        dp = np.linalg.norm(np.diff(mir_eff_xy, axis=0), axis=1)
        dt = np.diff(ts0)
        v = dp / np.maximum(dt, 1e-9)
        return v

    def check_reach_xy_with_offset(self,
                                   mir_xyz: np.ndarray,
                                   mir_yaw: np.ndarray,
                                   ts0: np.ndarray,
                                   di: np.ndarray,
                                   ur_tcp_xyz: np.ndarray,
                                   ur_ts: np.ndarray) -> tuple[bool, float]:
        """
        Reach-Constraint für Offset-Trajektorie bei festen Zeiten ts0.
        """
        idx_eff = np.arange(len(ts0), dtype=np.float64) + di
        mir_eff_xy = self._interp_along_index(mir_xyz[:, :2], idx_eff)
        yaw_eff = self._interp_yaw_along_index(mir_yaw, idx_eff)

        # UR TCP an denselben Zeiten
        tcp_xy = interp_along_t(ur_tcp_xyz, ur_ts, ts0)[:, :2]

        # UR-Base aus MiR + Mount-Offset
        c = np.cos(yaw_eff)
        s = np.sin(yaw_eff)
        off_x = self.mount_x * c - self.mount_y * s
        off_y = self.mount_x * s + self.mount_y * c

        ur_base_xy = np.stack([mir_eff_xy[:, 0] + off_x,
                               mir_eff_xy[:, 1] + off_y], axis=1)

        d = np.linalg.norm(tcp_xy - ur_base_xy, axis=1)
        dmax = float(np.max(d)) if len(d) else 0.0
        return dmax <= self.reach_xy_max, dmax

    # -------------------------------------------------------------------------
    # Reparatur & Schrittvorschlag im Indexraum
    # -------------------------------------------------------------------------

    def _repair_di(self,
                   di: np.ndarray,
                   di_max: float | None = None,
                   min_step: float | None = None,
                   max_step: float | None = None) -> np.ndarray:
        """
        Erzwingt:
        - |di[k]| <= di_max
        - i_eff[k] = k + di[k] in [0, N-1]
        - monotone i_eff mit Schrittweite in [min_step, max_step]
        """
        if di_max is None:
            di_max = self.di_max
        if min_step is None:
            min_step = self.min_step
        if max_step is None:
            max_step = self.max_step

        N = len(di)
        k = np.arange(N, dtype=np.float64)

        # Roh i_eff, begrenze globalen Offset
        i_eff = k + np.clip(di, -di_max, di_max)
        i_eff[0] = 0.0  # erster Punkt = Start

        for n in range(1, N):
            min_i = i_eff[n - 1] + min_step
            max_i = min(i_eff[n - 1] + max_step, float(N - 1))
            i_eff[n] = np.clip(i_eff[n], min_i, max_i)

        di_repaired = i_eff - k
        return di_repaired
    
    def repair_di_const(self,di: np.ndarray, di_max: float) -> np.ndarray:
        """
        Sanfte Reparatur nur für die Konstantgeschwindigkeits-Lösung:
        - begrenzt |di[k]| auf di_max
        - erzwingt monotones i_eff (keine Rückwärtsbewegung)
        - KEIN max_step-Clamping -> keine dauerhafte 2.0-Sättigung
        """
        N = len(di)
        k = np.arange(N, dtype=np.float64)

        # globaler Offset-Clip
        i_eff = k + np.clip(di, -di_max, di_max)

        # nur monotone Steigerung erzwingen
        i_eff = np.maximum.accumulate(i_eff)

        return i_eff - k


    def propose_offset_step(self, di: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Lokale Anpassung von di:
        - schnelle Segmente: Δi_eff verkleinern (di[k+1] Richtung di[k])
        - langsame Segmente: Δi_eff vergrößern
        """
        N = len(di)
        if N < 3:
            return di.copy()

        k_fast = max(1, int(round(self.k_fast_frac * (N - 1))))
        k_slow = max(1, int(round(self.k_slow_frac * (N - 1))))

        idx_fast = np.argsort(-v)[:k_fast]  # höchste v
        idx_slow = np.argsort(v)[:k_slow]   # niedrigste v

        di_new = di.copy()
        eps_idx = 0.05  # Schrittgröße im Indexraum pro Iteration

        # schnelle Segmente: Δi_eff = (di[k+1]-di[k]) kleiner machen
        for j in idx_fast:
            if j + 1 < N:
                di_new[j + 1] -= eps_idx

        # langsame Segmente: Δi_eff größer machen
        for j in idx_slow:
            if j + 1 < N:
                di_new[j + 1] += eps_idx

        # Reparieren: Monotonie, Grenzen, Schrittweite
        di_new = self._repair_di(di_new)
        return di_new

    # -------------------------------------------------------------------------
    # Hauptoptimierer im Index-Offset-Raum (hier ggf. optional)
    # -------------------------------------------------------------------------

    def optimize_index_offset(self,
                              mir_xyz: np.ndarray,
                              mir_yaw: np.ndarray,
                              ts0: np.ndarray,
                              ur_tcp_xyz: np.ndarray,
                              ur_ts: np.ndarray,
                              di_init: np.ndarray | None = None) -> np.ndarray:
        """
        Iterative Optimierung von di[k]:
        - Startet von di_init (oder 0, falls None)
        - reduziert Peak- oder L2-Geschwindigkeit
        - beachtet Reach-Constraint in jeder Iteration
        """
        N = len(ts0)

        if di_init is None:
            di_best = np.zeros((N,), dtype=np.float64)
        else:
            di_best = np.array(di_init, dtype=np.float64)
            if len(di_best) != N:
                raise ValueError("di_init length mismatch")

        # Sicherstellen, dass Startlösung gültig ist:
        di_best = self._repair_di(di_best)
        v0 = self.compute_speeds_with_offset(mir_xyz, ts0, di_best)
        obj_best = self.objective(v0)

        ok0, dmax0 = self.check_reach_xy_with_offset(
            mir_xyz, mir_yaw, ts0, di_best, ur_tcp_xyz, ur_ts
        )
        rospy.loginfo(f"Initial in optimize_index_offset: obj={obj_best:.4f} "
                      f"reach_ok={ok0} dmax={dmax0:.3f} m")

        accepted = 0
        last_log = time.time()

        for it in range(1, self.max_iters + 1):
            v = self.compute_speeds_with_offset(mir_xyz, ts0, di_best)
            di_prop = self.propose_offset_step(di_best, v)

            ok, dmax = self.check_reach_xy_with_offset(
                mir_xyz, mir_yaw, ts0, di_prop, ur_tcp_xyz, ur_ts
            )
            if not ok:
                continue

            v_prop = self.compute_speeds_with_offset(mir_xyz, ts0, di_prop)
            obj_prop = self.objective(v_prop)

            if obj_prop + self.accept_tol < obj_best:
                di_best = di_prop
                obj_best = obj_prop
                accepted += 1

            now = time.time()
            if now - last_log > 1.0:
                rospy.loginfo(
                    f"[it {it:4d}/{self.max_iters}] obj={obj_best:.4f} "
                    f"dmax={dmax:.3f}m accepted={accepted}"
                )
                last_log = now

        rospy.loginfo(f"Done index-offset optimization. best_obj={obj_best:.4f}, "
                      f"accepted={accepted}")
        return di_best


    def _equivalent_arc_length(self,
                               mir_xyz: np.ndarray,
                               mir_yaw: np.ndarray,
                               ts0: np.ndarray) -> np.ndarray:
        """
        Äquivalente kumulative Weglänge s[k]:

        - wenn v_lin >= rotation_only_threshold:
              -> nur v_lin (wie ursprüngliche XY-Metrik)
        - wenn v_lin <  rotation_only_threshold:
              -> max(v_lin, v_rot) mit geclipptem v_rot
        """
        N = len(ts0)
        s = np.zeros((N,), dtype=np.float64)
        if N < 2:
            return s

        mir_xy = mir_xyz[:, :2].astype(np.float64)

        # Translation
        dp = np.linalg.norm(np.diff(mir_xy, axis=0), axis=1)  # N-1

        # Yaw-Differenzen mit Wrap auf [-pi, pi]
        dyaw_raw = np.diff(mir_yaw)  # N-1
        dyaw = np.arctan2(np.sin(dyaw_raw), np.cos(dyaw_raw))

        # Zeitdifferenzen
        dt = np.diff(ts0)
        dt = np.maximum(dt, 1e-9)

        v_lin = dp / dt                     # reine Translationsgeschwindigkeit
        w = np.abs(dyaw) / dt
        v_rot = w * self.equiv_radius       # Rotations-"Speed" in m/s-Äquivalent
        v_rot = np.minimum(v_rot, self.max_rot_equiv_speed)

        # Maske: wo "fährt" der Roboter wirklich?
        move_mask = v_lin >= self.rotation_only_threshold

        # Basis: überall v_lin
        v_eff = v_lin.copy()

        # Nur dort, wo kaum Translation ist, Rotation berücksichtigen
        v_eff[~move_mask] = np.maximum(v_lin[~move_mask], v_rot[~move_mask])

        ds = v_eff * dt
        s[1:] = np.cumsum(ds)
        return s


    def compute_constant_speed_index_offset(self,
                                            mir_xyz: np.ndarray,
                                            mir_yaw: np.ndarray,
                                            ts0: np.ndarray) -> np.ndarray:
        """
        Berechnet di[k], so dass der Pfad mit (annähernd) konstanter
        äquivalenter Geschwindigkeit (Translation + Rotation) über die
        reale Zeit ts0[k] durchlaufen wird.

        Äquivalente Weglänge s berücksichtigt:
            v_lin = Δp / Δt
            v_rot = |Δyaw/Δt| * equiv_radius (geclippt)
            v_eff = max(v_lin, v_rot)
        """
        N = len(ts0)
        if N < 2:
            return np.zeros((N,), dtype=np.float64)

        # Äquivalente kumulative Weglänge (Translation + Rotation)
        s = self._equivalent_arc_length(mir_xyz, mir_yaw, ts0)
        s_total = float(s[-1])
        if s_total <= 1e-12:
            # Pfad hat praktisch keine Bewegung -> kein Offset nötig
            return np.zeros((N,), dtype=np.float64)

        # --- NEU: Ziel-Weglänge nach echter Zeit, nicht nach Index ---
        t0 = float(ts0[0])
        t1 = float(ts0[-1])
        T = max(t1 - t0, 1e-9)

        # Normierte Zeit tau[k] von 0..1
        tau = (ts0 - t0) / T

        # Ziel-Weglänge linear in der Zeit
        s_target = s_total * tau

        # Invertiere s(k): für jedes s_target finde Index u[k]
        k_idx = np.arange(N, dtype=np.float64)
        u = np.interp(s_target, s, k_idx)  # u[k] ist kontinuierlicher Index

        di = u - k_idx
        return di


    def scale_di_for_reach(self,
                           di: np.ndarray,
                           mir_xyz: np.ndarray,
                           mir_yaw: np.ndarray,
                           ts0: np.ndarray,
                           ur_tcp_xyz: np.ndarray,
                           ur_ts: np.ndarray,
                           max_iter: int = 20) -> np.ndarray:
        """
        Skaliert den Offset di mit einem Faktor alpha ∈ [0,1], sodass die Reach-Constraint
        erfüllt bleibt. Falls bereits ok: alpha = 1.0.

        Einfache binäre Suche auf alpha.
        """
        ok, dmax = self.check_reach_xy_with_offset(mir_xyz, mir_yaw, ts0, di,
                                                   ur_tcp_xyz, ur_ts)
        if ok:
            rospy.loginfo(f"Reach OK mit alpha=1.0 (dmax={dmax:.3f} m)")
            return di

        # Falls nicht ok: binäre Suche auf alpha
        lo, hi = 0.0, 1.0
        best_alpha = 0.0
        for _ in range(max_iter):
            alpha = 0.5 * (lo + hi)
            di_scaled = alpha * di
            ok_alpha, dmax_alpha = self.check_reach_xy_with_offset(
                mir_xyz, mir_yaw, ts0, di_scaled, ur_tcp_xyz, ur_ts
            )
            if ok_alpha:
                best_alpha = alpha
                lo = alpha
            else:
                hi = alpha

        di_scaled = best_alpha * di
        rospy.loginfo(f"Reach enforced via alpha={best_alpha:.3f}")
        return di_scaled

    # -------------------------------------------------------------------------
    # Publishing / CSV / Plots
    # -------------------------------------------------------------------------

    def publish(self, ts_orig: np.ndarray):
        """
        Publish original timestamps (unverändert) und deren dt – optional.
        """
        msg_ts = Float32MultiArray()
        msg_ts.data = [float(x) for x in ts_orig]
        self.pub_ts.publish(msg_ts)

        msg_dt = Float32MultiArray()
        msg_dt.data = [float(x) for x in np.diff(ts_orig)]
        self.pub_dt.publish(msg_dt)

        rospy.loginfo(f"Published ORIGINAL timestamps to {self.out_ts_topic} and dt to {self.out_dt_topic}")

    def export_csv_lines(self, ts0: np.ndarray, index_offset: np.ndarray):
        if not self.export_csv:
            return
        try:
            # 1) originale timestamps
            with open(self.csv_timestamps_path, "w", encoding="utf-8") as f:
                f.write(",".join([f"{float(x):.9f}" for x in ts0]))

            # 2) index offset mit Vorzeichen
            with open(self.csv_index_offset_path, "w", encoding="utf-8") as f:
                f.write(",".join([f"{float(x):+.6f}" for x in index_offset]))

            rospy.loginfo(f"Exported CSV: {self.csv_timestamps_path} and {self.csv_index_offset_path}")
        except Exception as e:
            rospy.logwarn(f"CSV export failed: {e}")

    def _first_layer_end_index(self, z: np.ndarray, eps_z: float = 1e-4) -> int:
        """Return end index (exclusive) for the first layer, based on first significant z-change."""
        if len(z) < 2:
            return len(z)
        z0 = float(z[0])
        dz = np.abs(z - z0)
        idx = np.where(dz > eps_z)[0]
        if len(idx) == 0:
            return len(z)
        return int(idx[0])

    def plot_debug(self, mir_xyz: np.ndarray, ts0: np.ndarray, di: np.ndarray):
        """
        Plot Geschwindigkeiten: original (di=0) vs. mit Indexoffset.
        """
        if not (self.save_plot and HAS_PLOT):
            return

        di_zero = np.zeros_like(di)
        v0 = self.compute_speeds_with_offset(mir_xyz, ts0, di_zero)
        v1 = self.compute_speeds_with_offset(mir_xyz, ts0, di)

        plt.figure()
        plt.plot(v0, label="orig v_xy (di=0)")
        plt.plot(v1, label="opt  v_xy (with di)")
        plt.xlabel("segment index")
        plt.ylabel("speed [m/s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=1600)
        rospy.loginfo(f"Saved speed plot: {self.plot_path}")

    def plot_xy_first_layer(self,
                            mir_xyz: np.ndarray,
                            ur_xyz: np.ndarray,
                            ts0: np.ndarray,
                            di: np.ndarray):
        """
        Plot MiR & UR XY (first layer only). MiR (mit Offset) farblich nach Geschwindigkeitsänderung.
        """
        if not (self.save_plot and HAS_PLOT):
            return

        n_mir = self._first_layer_end_index(mir_xyz[:, 2])
        n_ur = self._first_layer_end_index(ur_xyz[:, 2])
        n = max(2, min(n_mir, n_ur, len(mir_xyz), len(ur_xyz), len(ts0)))

        mir_xy_orig = mir_xyz[:n, :2]
        ur_xy = ur_xyz[:n, :2]
        ts0_n = ts0[:n]
        di_n = di[:n]

        # effektive MiR-Bahn mit Indexoffset
        idx_eff = np.arange(n, dtype=np.float64) + di_n
        mir_eff_xy = self._interp_along_index(mir_xyz[:, :2], idx_eff)

        # Geschwindigkeiten original vs. offset
        dp0 = np.linalg.norm(np.diff(mir_xy_orig, axis=0), axis=1)
        dp1 = np.linalg.norm(np.diff(mir_eff_xy, axis=0), axis=1)
        dt = np.diff(ts0_n)
        v0 = dp0 / np.maximum(dt, 1e-9)
        v1 = dp1 / np.maximum(dt, 1e-9)
        ratio = v1 / np.maximum(v0, 1e-12)

        # Map ratio -> diverging colormap centered at 1.0
        r = np.clip(np.log(ratio + 1e-12), -1.0, 1.0)  # symmetric around 0
        r_norm = (r - (-1.0)) / 2.0  # [0,1]

        # Segmente der effektiven MiR-Bahn
        segs = np.stack([mir_eff_xy[:-1], mir_eff_xy[1:]], axis=1)  # (n-1,2,2)
        lc = LineCollection(segs, array=r_norm, cmap="bwr", linewidths=2.0)

        plt.figure()
        ax = plt.gca()
        ax.add_collection(lc)
        ax.plot(ur_xy[:, 0], ur_xy[:, 1], linestyle="-", linewidth=1.5, label="UR TCP (first layer)")
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.legend()
        cbar = plt.colorbar(lc)
        cbar.set_label("MiR speed change: red=slower, blue=faster (log ratio)")
        plt.tight_layout()
        plt.savefig(self.xy_plot_path, dpi=160)
        rospy.loginfo(f"Saved XY plot (first layer, index offset): {self.xy_plot_path}")

    # -------------------------------------------------------------------------
    # Zusätzliche Plot-Funktionen aus dem Original
    # -------------------------------------------------------------------------

    def compute_speed_with_index_offset(self, mir_xyz, t0, di):
        """
        Wrapper, kompatibel zur Original-Funktion:
        nutzt intern compute_speeds_with_offset.
        """
        return self.compute_speeds_with_offset(mir_xyz, t0, di)

    def debug_offset_effects(self, mir_xyz, t0, di):
        """
        Plot:
        - v_orig vs. v_applied (mit Indexoffset)
        - di[k] (Indexoffset)
        - grad_di[k] = di[k+1] - di[k]
        """
        if not (self.save_plot and HAS_PLOT):
            return

        v_orig = np.linalg.norm(np.diff(mir_xyz[:, :2], axis=0), axis=1) / np.maximum(np.diff(t0), 1e-9)
        v_appl = self.compute_speed_with_index_offset(mir_xyz, t0, di)

        grad_di = np.diff(di)

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(v_orig, label="v_orig")
        axs[0].plot(v_appl, label="v_applied")
        axs[0].set_ylabel("v [m/s]")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(di, label="di (index offset)")
        axs[1].set_ylabel("di")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(grad_di, label="di[k+1] - di[k]")
        axs[2].axhline(0.0, linestyle="--")
        axs[2].set_ylabel("grad_di")
        axs[2].set_xlabel("Index k")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        out_path = self.plot_path.replace(".png", "_debug_offset_effects.png")
        plt.savefig(out_path, dpi=1600)
        rospy.loginfo(f"Saved debug offset effects plot: {out_path}")

    def plot_xy_first_layer_index_gradient(self,
                                           mir_xyz: np.ndarray,
                                           ur_xyz: np.ndarray,
                                           mir_ts0: np.ndarray,
                                           index_offset: np.ndarray):
        """
        Plot MiR & UR XY (first layer only).
        MiR-Bahn wird farbcodiert nach der Ableitung des Index-Offsets:
            grad_di[k] = di[k+1] - di[k]
        - grad_di > 0: Offset wird aufgebaut (MiR "zieht vor")
        - grad_di < 0: Offset wird abgebaut (MiR "wartet nach")
        Farben symmetrisch um 0 normalisiert.
        """
        if not (self.save_plot and HAS_PLOT):
            return

        # First layer bestimmen
        n_mir = self._first_layer_end_index(mir_xyz[:, 2])
        n_ur = self._first_layer_end_index(ur_xyz[:, 2])
        n = max(3, min(n_mir, n_ur, len(mir_xyz), len(ur_xyz), len(index_offset)))

        mir_xy = mir_xyz[:n, :2]
        ur_xy = ur_xyz[:n, :2]
        di = index_offset[:n]

        # Ableitung des Index-Offsets pro Segment
        grad_di = np.diff(di)  # Länge n-1, gehört zu Segmenten [k -> k+1]

        # symmetrische Normalisierung um 0
        if len(grad_di) > 0:
            max_abs = float(np.max(np.abs(grad_di)))
        else:
            max_abs = 1.0
        if max_abs < 1e-9:
            max_abs = 1.0  # alles ~0 => neutrales Mittelgrau

        grad_norm = grad_di / max_abs      # in [-1,1]
        c_vals = (grad_norm + 1.0) / 2.0   # auf [0,1] für Cmap

        # Segmente der MiR-Bahn
        segs = np.stack([mir_xy[:-1], mir_xy[1:]], axis=1)  # (n-1,2,2)
        lc = LineCollection(segs, array=c_vals, cmap="bwr", linewidths=2.0)

        plt.figure()
        ax = plt.gca()
        ax.add_collection(lc)
        ax.plot(ur_xy[:, 0], ur_xy[:, 1], linestyle="-", linewidth=1.5, label="UR TCP (first layer)")
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.legend()
        cbar = plt.colorbar(lc)
        cbar.set_label("Index-Gradient Δdi = di[k+1] - di[k] (rot<0, blau>0)")
        plt.tight_layout()

        out_path = self.xy_plot_path.replace(".png", "_index_grad.png")
        plt.savefig(out_path, dpi=160)
        rospy.loginfo(f"Saved XY plot (first layer, index gradient): {out_path}")

    def plot_xy_first_layer_index_offset(self,
                                         mir_xyz: np.ndarray,
                                         ur_xyz: np.ndarray,
                                         mir_ts0: np.ndarray,
                                         index_offset: np.ndarray):
        """
        Plot MiR & UR XY (first layer only).
        MiR-Bahn wird farbcodiert nach Index-Offset di[k] = i_eff[k] - k:
        - di > 0: MiR ist "voraus" (z.B. blau)
        - di < 0: MiR ist "hinten"  (z.B. rot)
        Farben werden symmetrisch um 0 normalisiert.
        """
        if not (self.save_plot and HAS_PLOT):
            return

        # First layer bestimmen
        n_mir = self._first_layer_end_index(mir_xyz[:, 2])
        n_ur = self._first_layer_end_index(ur_xyz[:, 2])
        n = max(2, min(n_mir, n_ur, len(mir_xyz), len(ur_xyz), len(index_offset)))

        mir_xy = mir_xyz[:n, :2]
        ur_xy = ur_xyz[:n, :2]
        di = index_offset[:n]

        # Index-Offset pro Segment (Mittelwert der Endpunkte)
        di_seg = 0.5 * (di[:-1] + di[1:])

        # symmetrische Normalisierung um 0
        if len(di_seg) > 0:
            max_abs = float(np.max(np.abs(di_seg)))
        else:
            max_abs = 1.0
        if max_abs < 1e-9:
            max_abs = 1.0  # alles ~0 => neutrales Mittelgrau

        di_norm = di_seg / max_abs        # in [-1,1]
        # auf [0,1] mappen für evtl. weitere Nutzung
        # (für LineCollection reicht di_seg mit symmetrischem Cmap)
        _c_vals = (di_norm + 1.0) / 2.0

        # Segmente der MiR-Bahn
        segs = np.stack([mir_xy[:-1], mir_xy[1:]], axis=1)  # (n-1,2,2)
        lc = LineCollection(segs, array=di_seg, cmap="bwr", linewidths=2.0)

        plt.figure()
        ax = plt.gca()
        ax.add_collection(lc)
        ax.plot(ur_xy[:, 0], ur_xy[:, 1], linestyle="-", linewidth=1.5, label="UR TCP (first layer)")
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.legend()
        cbar = plt.colorbar(lc)
        cbar.set_label("Index offset di (relativ, rot<0, blau>0)")
        plt.tight_layout()

        out_path = self.xy_plot_path.replace(".png", "_index_offset.png")
        plt.savefig(out_path, dpi=160)
        rospy.loginfo(f"Saved XY plot (first layer, index offset): {out_path}")

    # -------------------------------------------------------------------------
    # Main
    # -------------------------------------------------------------------------

    def run(self):
        mir_path, ur_path, mir_ts0 = self.wait_inputs()

        rospy.loginfo(f"mir frame: {mir_path.header.frame_id}")
        rospy.loginfo(f"ur  frame: {ur_path.header.frame_id}")

        mir_xyz, mir_yaw = path_to_xyz_yaw(mir_path)
        ur_tcp_xyz, _ = path_to_xyz_yaw(ur_path)

        ur_ts = self.build_ur_time(mir_ts0, len(ur_tcp_xyz))

        # 1) Konstantgeschwindigkeits-Offset als Startlösung
        di_const = self.compute_constant_speed_index_offset(mir_xyz, mir_yaw, mir_ts0)

        # 1a) optional: leicht reparieren (Grenzen, Monotonie)
        di_const = self._repair_di(di_const, min_step=0.0)

        # 1b) Reach-Constraint global einhalten
        di_init = self.scale_di_for_reach(di_const, mir_xyz, mir_yaw,
                                          mir_ts0, ur_tcp_xyz, ur_ts)

        rospy.loginfo(f"Initial di_init: min={float(np.min(di_init)):.3f}, "
                      f"max={float(np.max(di_init)):.3f}")

        # 2) Iterative lokale Optimierung (DAS ist jetzt die eigentliche Optimierung)
        di_best = self.optimize_index_offset(
            mir_xyz, mir_yaw, mir_ts0, ur_tcp_xyz, ur_ts, di_init=di_init
        )

        rospy.loginfo(f"Optimized di_best: min={float(np.min(di_best)):.3f}, "
                      f"max={float(np.max(di_best)):.3f}")

        # 3) Publish: originale Zeitstempel (unverändert)
        self.publish(mir_ts0)

        # 4) CSV-Export: originale Zeitstempel + finaler Indexoffset
        self.export_csv_lines(mir_ts0, di_best)

        # 5) Plots mit dem FINALEN di_best
        if self.save_plot and HAS_PLOT:
            self.plot_debug(mir_xyz, mir_ts0, di_best)
            self.plot_xy_first_layer(mir_xyz, ur_tcp_xyz, mir_ts0, di_best)
            self.debug_offset_effects(mir_xyz, mir_ts0, di_best)
            self.plot_xy_first_layer_index_offset(mir_xyz, ur_tcp_xyz, mir_ts0, di_best)
            self.plot_xy_first_layer_index_gradient(mir_xyz, ur_tcp_xyz, mir_ts0, di_best)



if __name__ == "__main__":
    try:
        node = LocalRetimingOptimizerNode()
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Optimizer crashed: {e}")
        raise
