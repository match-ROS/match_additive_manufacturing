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
    Local, iterative MiR retiming:
    - UR path & timing fixed (computed from MiR start/end + equal steps)
    - only MiR timestamps are modified
    - reach constraint checked in XY between UR base (MiR + rotated mount offset) and UR TCP
    """

    def __init__(self):
        rospy.init_node("mir_local_retiming_optimizer", anonymous=False)

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

        # Constraint (XY only)
        self.reach_xy_max = float(rospy.get_param("~reach_xy_max", 1.10))

        # Optimization params
        self.max_iters = int(rospy.get_param("~max_iters", 400))
        self.eps = float(rospy.get_param("~eps_dt", 0.005))  # seconds per accepted step
        self.k_fast_frac = float(rospy.get_param("~k_fast_frac", 0.10))  # top 10% segments
        self.k_slow_frac = float(rospy.get_param("~k_slow_frac", 0.20))  # bottom 20% segments
        self.min_dt = float(rospy.get_param("~min_dt", 0.002))  # 2 ms safety
        self.sample_stride = int(rospy.get_param("~reach_sample_stride", 1))  # 1 = every UR knot

        # Objective
        self.obj_mode = rospy.get_param("~objective", "peak")  # "peak" or "l2"
        self.accept_tol = float(rospy.get_param("~accept_tol", 1e-9))

        # Plotting
        self.save_plot = bool(rospy.get_param("~save_plot", True))
        self.plot_path = rospy.get_param("~plot_path", "/tmp/mir_retiming.png")

        # CSV exports
        self.export_csv = bool(rospy.get_param("~export_csv", True))
        self.csv_timestamps_path = rospy.get_param("~csv_timestamps_path", "/tmp/mir_timestamps_optimized.csv")
        self.csv_index_offset_path = rospy.get_param("~csv_index_offset_path", "/tmp/mir_index_offset.csv")

        # XY path plot (first layer)
        self.xy_plot_path = rospy.get_param("~xy_plot_path", "/tmp/mir_ur_xy_first_layer.png")
        self.layer_z_eps = float(rospy.get_param("~layer_z_eps", 1e-4))

        # Publishers
        self.pub_ts = rospy.Publisher(self.out_ts_topic, Float32MultiArray, queue_size=1, latch=True)
        self.pub_dt = rospy.Publisher(self.out_dt_topic, Float32MultiArray, queue_size=1, latch=True)

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
            rospy.logwarn("UR path length != MiR path length. This is OK only if you expect interpolation; "
                          "but here we assume same number of steps for 'trivial UR timestamps'. "
                          "Proceeding anyway with UR knots based on its own length.")

        return mir_path, ur_path, mir_ts

    def build_ur_time(self, mir_ts, n_ur):
        # UR timing fixed: same start/end as MiR, uniform steps over UR points
        t0 = float(mir_ts[0])
        t1 = float(mir_ts[-1])
        if n_ur < 2:
            return np.array([t0], dtype=np.float64)
        return np.linspace(t0, t1, n_ur, dtype=np.float64)

    def compute_mir_speeds(self, mir_xyz, mir_ts):
        dt = np.diff(mir_ts)
        dp = np.linalg.norm(np.diff(mir_xyz[:, :2], axis=0), axis=1)  # XY distance
        # Avoid div by 0
        dt_safe = np.maximum(dt, 1e-9)
        v = dp / dt_safe
        return v, dt

    def objective(self, v):
        if self.obj_mode == "l2":
            return float(np.mean(v * v))
        return float(np.max(v))

    def check_reach_xy(self, mir_xyz, mir_yaw, mir_ts, ur_tcp_xyz, ur_ts):
        """
        Sample times (UR knots by default). For each time t:
        - MiR pose interpolated at t (xyz + yaw)
        - UR TCP interpolated at t (given UR path over ur_ts)
        - UR base = MiR_xy + R(yaw)*mount_xy
        - check ||(tcp_xy - ur_base_xy)|| <= reach_xy_max
        """
        # sample times: take ur_ts (or strided)
        t_s = ur_ts[:: max(1, self.sample_stride)].copy()

        mir_xy = interp_along_t(mir_xyz, mir_ts, t_s)[:, :2]
        yaw_s = interp_yaw(mir_yaw, mir_ts, t_s)

        tcp_xy = interp_along_t(ur_tcp_xyz, ur_ts, t_s)[:, :2]

        # rotate mount offset
        c = np.cos(yaw_s)
        s = np.sin(yaw_s)
        off_x = self.mount_x * c - self.mount_y * s
        off_y = self.mount_x * s + self.mount_y * c

        ur_base_xy = np.stack([mir_xy[:, 0] + off_x, mir_xy[:, 1] + off_y], axis=1)

        d = np.linalg.norm(tcp_xy - ur_base_xy, axis=1)  # XY only
        dmax = float(np.max(d)) if len(d) else 0.0
        return dmax <= self.reach_xy_max, dmax

    def propose_step(self, dt, v):
        """
        Stretch fast segments, compress slow segments to keep sum(dt) constant.
        dt has length N-1.
        """
        n = len(dt)
        if n < 3:
            return dt.copy()

        k_fast = max(1, int(round(self.k_fast_frac * n)))
        k_slow = max(1, int(round(self.k_slow_frac * n)))

        idx_fast = np.argsort(-v)[:k_fast]       # highest speeds
        idx_slow = np.argsort(v)[:k_slow]        # lowest speeds

        dt_new = dt.copy()

        # Stretch fast
        dt_new[idx_fast] += self.eps

        # Compress slow to balance total duration
        total_add = self.eps * len(idx_fast)
        # distribute proportionally (uniform)
        per = total_add / float(len(idx_slow))

        dt_new[idx_slow] -= per

        # Enforce min_dt by pulling back uniformly from fast segments if needed
        if np.any(dt_new < self.min_dt):
            # Simple repair: clamp, then re-balance to keep sum constant.
            dt_clamped = np.maximum(dt_new, self.min_dt)
            # restore total duration (sum(dt) must equal sum(original dt))
            target_sum = float(np.sum(dt))
            cur_sum = float(np.sum(dt_clamped))
            diff = cur_sum - target_sum  # positive => need to remove time
            if abs(diff) > 1e-12:
                # remove diff from the currently largest dt segments (usually safe)
                idx_desc = np.argsort(-dt_clamped)
                for j in idx_desc:
                    if diff <= 0:
                        break
                    reducible = dt_clamped[j] - self.min_dt
                    take = min(reducible, diff)
                    dt_clamped[j] -= take
                    diff -= take
                # if still diff>0, can't satisfy min_dt; return None
                if diff > 1e-9:
                    return None
            dt_new = dt_clamped

        # Final sanity: preserve sum close
        if abs(float(np.sum(dt_new)) - float(np.sum(dt))) > 1e-6:
            # numerical drift: correct by nudging the last segment (if possible)
            drift = float(np.sum(dt_new) - np.sum(dt))
            j = n - 1
            dt_new[j] = max(self.min_dt, dt_new[j] - drift)

        return dt_new

    def optimize(self, mir_xyz, mir_yaw, mir_ts0, ur_tcp_xyz, ur_ts):
        v0, dt0 = self.compute_mir_speeds(mir_xyz, mir_ts0)
        obj0 = self.objective(v0)

        ok0, dmax0 = self.check_reach_xy(mir_xyz, mir_yaw, mir_ts0, ur_tcp_xyz, ur_ts)
        rospy.loginfo(f"Initial: objective={obj0:.4f}  reach_ok={ok0}  reach_dmax_xy={dmax0:.3f} m")

        if not ok0:
            rospy.logwarn("Initial solution violates reach_xy_max. Local retiming cannot fix reach if geometry is wrong.")
            return mir_ts0

        best_ts = mir_ts0.copy()
        best_obj = obj0
        best_dmax = dmax0

        dt_best = np.diff(best_ts)

        last_log = time.time()
        accepted = 0

        for it in range(1, self.max_iters + 1):
            v, dt = self.compute_mir_speeds(mir_xyz, best_ts)

            dt_prop = self.propose_step(dt, v)
            if dt_prop is None:
                rospy.logwarn("Cannot satisfy min_dt with current eps; stopping.")
                break

            ts_prop = np.concatenate([[best_ts[0]], best_ts[0] + np.cumsum(dt_prop)])

            # constraint check
            ok, dmax = self.check_reach_xy(mir_xyz, mir_yaw, ts_prop, ur_tcp_xyz, ur_ts)
            if not ok:
                # reject; try smaller eps occasionally
                continue

            v_prop, _ = self.compute_mir_speeds(mir_xyz, ts_prop)
            obj_prop = self.objective(v_prop)

            if obj_prop + self.accept_tol < best_obj:
                best_ts = ts_prop
                best_obj = obj_prop
                best_dmax = dmax
                accepted += 1

            # progress / heartbeat
            now = time.time()
            if now - last_log > 1.0:
                rospy.loginfo(f"[it {it:4d}/{self.max_iters}] best_obj={best_obj:.4f} "
                              f"reach_dmax_xy={best_dmax:.3f}m accepted={accepted}")
                last_log = now

        rospy.loginfo(f"Done. best_obj={best_obj:.4f} accepted_steps={accepted}")
        return best_ts

    def publish(self, ts_opt):
        msg_ts = Float32MultiArray()
        msg_ts.data = [float(x) for x in ts_opt]
        self.pub_ts.publish(msg_ts)

        msg_dt = Float32MultiArray()
        msg_dt.data = [float(x) for x in np.diff(ts_opt)]
        self.pub_dt.publish(msg_dt)

        rospy.loginfo(f"Published optimized timestamps to {self.out_ts_topic} and dt to {self.out_dt_topic}")

    def plot_debug(self, mir_xyz, ts0, ts1):
        if not (self.save_plot and HAS_PLOT):
            return
        v0, _ = self.compute_mir_speeds(mir_xyz, ts0)
        v1, _ = self.compute_mir_speeds(mir_xyz, ts1)

        plt.figure()
        plt.plot(v0, label="orig v_xy")
        plt.plot(v1, label="opt  v_xy")
        plt.xlabel("segment index")
        plt.ylabel("speed [m/s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=1600)
        rospy.loginfo(f"Saved plot: {self.plot_path}")

    def compute_ur_base_world_at_mir_samples(self, mir_xyz: np.ndarray, mir_yaw: np.ndarray) -> np.ndarray:
        """UR base position in world at each MiR sample (same indexing as MiR path)."""
        c = np.cos(mir_yaw)
        s = np.sin(mir_yaw)
        off_x = self.mount_x * c - self.mount_y * s
        off_y = self.mount_x * s + self.mount_y * c
        ur_base = mir_xyz.copy()
        ur_base[:, 0] = mir_xyz[:, 0] + off_x
        ur_base[:, 1] = mir_xyz[:, 1] + off_y
        return ur_base

    def export_csv_lines(self, ts_opt: np.ndarray, index_offset: np.ndarray):
        if not self.export_csv:
            return
        try:
            # 1) timestamps
            with open(self.csv_timestamps_path, "w", encoding="utf-8") as f:
                f.write(",".join([f"{float(x):.9f}" for x in ts_opt]))

            # 2) index offset with explicit sign
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

    def plot_xy_first_layer(self,
                            mir_xyz: np.ndarray,
                            ur_xyz: np.ndarray,
                            mir_ts0: np.ndarray,
                            ts_opt: np.ndarray):
        """Plot MiR & UR XY (first layer only). MiR colored by speed ratio (opt vs orig)."""
        if not (self.save_plot and HAS_PLOT):
            return

        n_mir = self._first_layer_end_index(mir_xyz[:, 2])
        n_ur = self._first_layer_end_index(ur_xyz[:, 2])
        n = max(2, min(n_mir, n_ur, len(mir_xyz), len(ur_xyz)))

        mir_xy = mir_xyz[:n, :2]
        ur_xy = ur_xyz[:n, :2]

        # speed ratio per segment (XY)
        dp = np.linalg.norm(np.diff(mir_xy, axis=0), axis=1)
        dt0 = np.diff(mir_ts0[:n])
        dt1 = np.diff(ts_opt[:n])
        v0 = dp / np.maximum(dt0, 1e-9)
        v1 = dp / np.maximum(dt1, 1e-9)
        ratio = v1 / np.maximum(v0, 1e-12)

        # Map ratio -> diverging colormap centered at 1.0
        r = np.clip(np.log(ratio + 1e-12), -1.0, 1.0)  # symmetric around 0
        r_norm = (r - (-1.0)) / (2.0)  # [0,1]

        # Build colored segments for MiR
        segs = np.stack([mir_xy[:-1], mir_xy[1:]], axis=1)  # (n-1,2,2)
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
        rospy.loginfo(f"Saved XY plot (first layer): {self.xy_plot_path}")

    def _interp_index_at_times(self,t_query: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """
        Map times -> continuous path index via 1D interpolation.
        Returns u in [0, N-1].
        """
        N = len(t_grid)
        idx = np.arange(N, dtype=np.float64)
        tq = np.clip(t_query, float(t_grid[0]), float(t_grid[-1]))
        return np.interp(tq, t_grid, idx)

    def _ur_tcp_xy_at_times(self,t_query: np.ndarray, ts0: np.ndarray, ur_p: np.ndarray) -> np.ndarray:
        """
        UR is FIXED. Interpolate TCP position at given times, return (N,2) XY.
        """
        tq = np.clip(t_query, float(ts0[0]), float(ts0[-1]))
        x = np.interp(tq, ts0, ur_p[:, 0].astype(np.float64))
        y = np.interp(tq, ts0, ur_p[:, 1].astype(np.float64))
        return np.stack([x, y], axis=1)

    def compute_reach_xy(self,ts0: np.ndarray,
                        t_opt: np.ndarray,
                        ur_p: np.ndarray,
                        ur_base_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        d0_xy[k] = reach XY at original time t0[k]
        d1_xy[k] = reach XY at optimized time t_opt[k]
        ur_base_world is at MiR sample k (same indexing as paths).
        """
        tcp0_xy = self._ur_tcp_xy_at_times(ts0, ts0, ur_p)
        tcp1_xy = self._ur_tcp_xy_at_times(t_opt, ts0, ur_p)

        base_xy = ur_base_world[:, :2].astype(np.float64)
        d0 = np.linalg.norm(tcp0_xy - base_xy, axis=1)
        d1 = np.linalg.norm(tcp1_xy - base_xy, axis=1)
        return d0, d1

    def compute_mir_index_deviation(self,ts0: np.ndarray, t_opt: np.ndarray) -> np.ndarray:
        """
        For each original global index k (time ts0[k]),
        compute continuous MiR index reached under optimized timing minus k.
        """
        i_opt_at_t0 = self._interp_index_at_times(ts0, t_opt)
        k = np.arange(len(ts0), dtype=np.float64)
        return i_opt_at_t0 - k

    def compute_index_offset_mir_vs_ur(self, ts_opt: np.ndarray, ur_ts: np.ndarray) -> np.ndarray:
        """At MiR index k (time ts_opt[k]), compute offset = k - u_ur(ts_opt[k]).

        Positive => MiR is ahead (larger index) compared to UR at that time.
        Negative => MiR is behind.
        """
        u = self._interp_index_at_times(ts_opt, ur_ts)
        k = np.arange(len(ts_opt), dtype=np.float64)
        return k - u

    def plot_reach_and_index_offset(self,ts0: np.ndarray,
                                t_opt: np.ndarray,
                                ur_p: np.ndarray,
                                ur_base_world: np.ndarray,
                                max_reach_xy: float = 1.10):
        d0_xy, d1_xy = self.compute_reach_xy(ts0, t_opt, ur_p, ur_base_world)
        di = self.compute_mir_index_deviation(ts0, t_opt)

        plt.figure()
        plt.plot(d0_xy, label="reach XY (orig timing)")
        plt.plot(d1_xy, label="reach XY (optimized timing)")
        plt.axhline(max_reach_xy, linestyle="--", label="max_reach_xy")
        plt.xlabel("global index k")
        plt.ylabel("distance in XY [m]")
        plt.grid(True)
        plt.legend()

        # Save reach plot
        plot_path1 = self.plot_path.replace(".png", "_reach_xy.png")
        plt.savefig(plot_path1, dpi=1600)

        plt.figure()
        plt.plot(di, label="MiR index deviation at original time: i_opt(t0[k]) - k")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("global index k (original)")
        plt.ylabel("index deviation [idx]")
        plt.grid(True)
        plt.legend()

        # Save combined plot
        plot_path2 = self.plot_path.replace(".png", "_reach_index.png")
        plt.savefig(plot_path2, dpi=1600)


    def run(self):
        mir_path, ur_path, mir_ts0 = self.wait_inputs()

        rospy.loginfo(f"mir frame: {mir_path.header.frame_id}")
        rospy.loginfo(f"ur  frame: {ur_path.header.frame_id}")

        mir_xyz, mir_yaw = path_to_xyz_yaw(mir_path)
        ur_tcp_xyz, _ = path_to_xyz_yaw(ur_path)

        ur_ts = self.build_ur_time(mir_ts0, len(ur_tcp_xyz))

        ts_opt = self.optimize(mir_xyz, mir_yaw, mir_ts0, ur_tcp_xyz, ur_ts)

        # 1) Optimierte Timestamps kannst du weiter plotten/debuggen,
        #    aber für die eigentliche Anwendung interessieren uns jetzt
        #    die Offsets bei den ORIGINALZEITEN ts0:
        mir_index_offset_at_ts0 = self.compute_mir_index_deviation(mir_ts0, ts_opt)
        # di[k] > 0 heißt: bei Zeit ts0[k] sollte die MiR weiter vorne auf der Bahn sein.

        # Optional: weiterhin publish der optimierten Zeitstempel (nur zu Debugzwecken)
        self.publish(ts_opt)

        # UR base für Reach-Plot
        ur_base_world = self.compute_ur_base_world_at_mir_samples(mir_xyz, mir_yaw)

        # CSV-Export jetzt mit ORIGINALZEITEN und Index-Offset bei diesen Zeiten:
        self.export_csv_lines(mir_ts0, mir_index_offset_at_ts0)

        # Plots wie bisher, nur aufpassen, welche Zeitachse du für was nimmst
        self.plot_debug(mir_xyz, mir_ts0, ts_opt)
        if self.save_plot and HAS_PLOT:
            self.plot_reach_and_index_offset(mir_ts0, ts_opt, ur_tcp_xyz, ur_base_world=ur_base_world,
                                            max_reach_xy=self.reach_xy_max)
            self.plot_xy_first_layer(mir_xyz, ur_tcp_xyz, mir_ts0, ts_opt)



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