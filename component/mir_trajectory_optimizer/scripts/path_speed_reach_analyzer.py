#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, Float32
from scipy.optimize import minimize
import time

# Optional plotting (only if you want plots on the robot PC)
import matplotlib.pyplot as plt


def yaw_from_quat(q):
    # q: geometry_msgs/Quaternion
    # Returns yaw (Z) assuming ENU / ROS standard
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def path_to_np(path_msg):
    # Returns positions (N,3) and yaws (N,)
    poses = path_msg.poses
    N = len(poses)
    p = np.zeros((N, 3), dtype=np.float64)
    yaw = np.zeros((N,), dtype=np.float64)
    for i, ps in enumerate(poses):
        p[i, 0] = ps.pose.position.x
        p[i, 1] = ps.pose.position.y
        p[i, 2] = ps.pose.position.z
        yaw[i] = yaw_from_quat(ps.pose.orientation)
    return p, yaw


def interp_path_pos(p, u):
    """
    Linear interpolation on discrete path positions.
    p: (N,3), u: float in [0, N-1]
    """
    N = p.shape[0]
    u = np.clip(u, 0.0, float(N - 1))
    i0 = int(np.floor(u))
    i1 = min(i0 + 1, N - 1)
    a = u - float(i0)
    return (1.0 - a) * p[i0] + a * p[i1]


class MirUrTimestampOptimizer:
    def __init__(self):
        # Topics
        self.topic_mir_path = rospy.get_param("~mir_path_topic", "/mur620c/mir_path_original")
        self.topic_ur_path = rospy.get_param("~ur_path_topic", "/mur620c/ur_path_original")
        self.topic_mir_ts = rospy.get_param("~mir_ts_topic", "/mur620c/mir_path_timestamps")

        # Output topics
        self.pub_delta_t = rospy.Publisher("~opt_delta_t", Float32MultiArray, queue_size=1, latch=True)
        self.pub_offset = rospy.Publisher("~opt_index_offset", Float32MultiArray, queue_size=1, latch=True)
        self.pub_progress = rospy.Publisher("trajectory_opt/progress", Float32, queue_size=10)
        self.pub_best_cost = rospy.Publisher("trajectory_opt/best_cost", Float32, queue_size=10)

        # Mount transform UR base in MiR base frame (meters)
        # Given: xyz="0.549 -0.318 0.49"
        self.t_mir_to_urbase = np.array([0.549, -0.318, 0.49], dtype=np.float64)

        # Hard limit
        self.max_reach = rospy.get_param("~max_reach_m", 1.10)
        self.eps_dt = rospy.get_param("~min_dt", 1e-3)

        # Weights (tune)
        self.w_ur_speed = rospy.get_param("~w_ur_speed", 10.0)
        self.w_ur_speed_smooth = rospy.get_param("~w_ur_speed_smooth", 2.0)
        self.w_mir_speed = rospy.get_param("~w_mir_speed", 1.0)
        self.w_mir_acc = rospy.get_param("~w_mir_acc", 0.2)
        self.w_u_smooth = rospy.get_param("~w_u_smooth", 1.0)
        self.w_dt_smooth = rospy.get_param("~w_dt_smooth", 0.2)
        self.w_dt_dev = rospy.get_param("~w_dt_dev", 1e-3)

        # Optional relative bounds around original dt (keeps dt within [lo*dt0, hi*dt0])
        self.use_dt_ratio_bounds = rospy.get_param("~use_dt_ratio_bounds", False)
        self.dt_ratio_lo = rospy.get_param("~dt_ratio_lo", 0.2)
        self.dt_ratio_hi = rospy.get_param("~dt_ratio_hi", 5.0)

    def _wait_inputs(self):
        rospy.loginfo("Waiting for live topics...")
        mir_path_msg = rospy.wait_for_message(self.topic_mir_path, Path)
        ur_path_msg = rospy.wait_for_message(self.topic_ur_path, Path)
        mir_ts_msg = rospy.wait_for_message(self.topic_mir_ts, Float32MultiArray)

        mir_p, mir_yaw = path_to_np(mir_path_msg)
        ur_p, _ = path_to_np(ur_path_msg)

        ts = np.array(mir_ts_msg.data, dtype=np.float64)
        if ts.ndim != 1:
            raise RuntimeError("mir_path_timestamps must be 1D")

        N = mir_p.shape[0]
        if ur_p.shape[0] != N:
            raise RuntimeError(f"MiR and UR path must have same length. Got {N} vs {ur_p.shape[0]}")
        if ts.shape[0] != N:
            # raise RuntimeError(f"Timestamps length must match path length. Got {ts.shape[0]} vs {N}")
            ts = ts[:N]

        # Normalize timestamps: t0=0
        ts = ts - ts[0]
        T = float(ts[-1])
        if T <= 0.0:
            raise RuntimeError("Total duration must be > 0")

        return mir_p, mir_yaw, ur_p, ts, T

    def _compute_ur_base_pos(self, mir_p, mir_yaw):
        # UR base position in world for each MiR sample
        N = mir_p.shape[0]
        out = np.zeros((N, 3), dtype=np.float64)
        for k in range(N):
            R = rotz(mir_yaw[k])
            out[k] = mir_p[k] + R @ self.t_mir_to_urbase
        return out

    
    def optimize_once(self, do_plot=True):
        """Optimize ONLY MiR timing (dt). UR path & UR timing are FIXED.
        We only optimize dt[0..N-2] (positive) with sum(dt)=T.
        The implied continuous UR index u_k is derived from the new MiR time t_k via
        u_k = interp(t_k, t_ur, arange(N)), and the index offset is o_k = u_k - k.
        """

        mir_p, mir_yaw, ur_p, ts0, T = self._wait_inputs()
        N = mir_p.shape[0]

        # Precompute geometry (MiR segment lengths)
        mir_seg = np.linalg.norm(mir_p[1:] - mir_p[:-1], axis=1)  # (N-1,)

        # MiR->UR base transform in world (depends on MiR yaw at each point)
        ur_base = self._compute_ur_base_pos(mir_p, mir_yaw)

        # Original dt as warm-start
        dt0 = np.diff(ts0)
        dt0 = np.maximum(dt0, self.eps_dt)
        dt0 *= (T / float(np.sum(dt0)))  # enforce sum(dt)=T
        x0 = dt0.copy()

        # Bounds: dt in [eps_dt, +inf)
        bnds = [(self.eps_dt, None)] * (N - 1)

        # Vectorized interpolation helpers for UR as function of time
        ur_idx = np.arange(N, dtype=np.float64)
        ur_x = ur_p[:, 0].astype(np.float64)
        ur_y = ur_p[:, 1].astype(np.float64)
        ur_z = ur_p[:, 2].astype(np.float64)

        def ur_tcp_at_times(t_vec):
            # Clamp into UR time range
            t = np.clip(t_vec, float(ts0[0]), float(ts0[-1]))
            x = np.interp(t, ts0, ur_x)
            y = np.interp(t, ts0, ur_y)
            z = np.interp(t, ts0, ur_z)
            return np.stack([x, y, z], axis=1)

        def ur_u_at_times(t_vec):
            t = np.clip(t_vec, float(ts0[0]), float(ts0[-1]))
            return np.interp(t, ts0, ur_idx)

        # Progress / heartbeat counters
        global eval_k, last_eval_t
        eval_k = 0
        last_eval_t = time.time()

        def cost(dt):
            """Smoothness cost for MiR speed/acc + mild regularization to dt0."""
            global eval_k, last_eval_t
            eval_k += 1
            last_eval_t = time.time()

            # MiR speed and acceleration (discrete)
            v = mir_seg / dt  # (N-1,)
            a = np.diff(v) / np.maximum(dt[1:], self.eps_dt)  # (N-2,)

            # Smooth dt
            dt_d = np.diff(dt)  # (N-2,)

            J = 0.0
            J += self.w_mir_speed * float(np.sum(v ** 2))
            J += self.w_mir_acc * float(np.sum(a ** 2))
            J += self.w_dt_smooth * float(np.sum(dt_d ** 2))
            # Keep solution near original timing to avoid wild warps
            J += self.w_dt_dev * float(np.sum((dt - dt0) ** 2))

            if eval_k % 500 == 0:
                rospy.loginfo(f"[SLSQP] cost eval #{eval_k}  J={J:.3e}")
            return J

        # Constraints
        cons = []

        # sum(dt) = T
        cons.append({"type": "eq", "fun": lambda dt: float(np.sum(dt) - T)})

        # Reachability: max_reach - distance >= 0 for each k
        def reach_ineq(dt):
            t = np.concatenate([[0.0], np.cumsum(dt)])  # (N,)
            tcp = ur_tcp_at_times(t)  # (N,3)
            d = np.linalg.norm(tcp[:, :2] - ur_base[:, :2], axis=1)  # (N,)
            print(d)
            return (self.max_reach - d)

        cons.append({"type": "ineq", "fun": reach_ineq})

        # Optional: keep dt not too small / too large relative to original (softly as constraints)
        if self.use_dt_ratio_bounds:
            lo = self.dt_ratio_lo
            hi = self.dt_ratio_hi
            cons.append({"type": "ineq", "fun": lambda dt: dt - lo * dt0})
            cons.append({"type": "ineq", "fun": lambda dt: hi * dt0 - dt})

        # Callback (per SLSQP iteration, not per eval)
        global cb_iter, last_cb_t
        cb_iter = 0
        last_cb_t = time.time()

        def cb(dt_k):
            global cb_iter, last_cb_t
            cb_iter += 1
            last_cb_t = time.time()
            rospy.loginfo(f"[SLSQP] iter={cb_iter}  dt_mean={float(np.mean(dt_k)):.4f}s  dt_min={float(np.min(dt_k)):.4f}s")

        # Heartbeat: tells you whether we are still evaluating cost/constraints
        def heartbeat(_evt):
            now = time.time()
            age = now - max(last_cb_t, last_eval_t)
            rospy.loginfo(f"[HB] running  evals={eval_k}  it={cb_iter}  idle={age:.1f}s")

        hb_timer = rospy.Timer(rospy.Duration(2.0), heartbeat)

        rospy.loginfo("Starting optimization (SLSQP, dt-only)...")
        res = minimize(
            cost,
            x0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            callback=cb,
            options={"maxiter": 200, "ftol": 1e-6, "disp": True}
        )

        hb_timer.shutdown()

        if not res.success:
            rospy.logwarn(f"Optimization did not fully converge: {res.message}")

        dt_opt = np.maximum(np.array(res.x, dtype=np.float64), self.eps_dt)
        dt_opt *= (T / float(np.sum(dt_opt)))  # re-normalize to keep sum(dt)=T

        # Derived continuous UR index u_k from optimized time grid (for your controller)
        t_opt = np.concatenate([[0.0], np.cumsum(dt_opt)])
        u_opt = ur_u_at_times(t_opt)  # (N,)
        offset_opt = u_opt - np.arange(N, dtype=np.float64)

        # Publish results (latched)
        msg_dt = Float32MultiArray(data=dt_opt.astype(np.float32).tolist())
        msg_off = Float32MultiArray(data=offset_opt.astype(np.float32).tolist())
        self.pub_delta_t.publish(msg_dt)
        self.pub_offset.publish(msg_off)
        rospy.loginfo("Published ~opt_delta_t (decision vars) and ~opt_index_offset (derived from timing; latched).")

        if do_plot:
            self._plot_compare_fixed_ur(mir_p, mir_yaw, ur_p, ur_base, ts0, dt_opt)

        return dt_opt, offset_opt, res

    def _plot_compare_fixed_ur(self, mir_p, mir_yaw, ur_p, ur_base, ts0, dt_opt):
        N = mir_p.shape[0]
        t0 = ts0
        t1 = np.concatenate([[0.0], np.cumsum(dt_opt)])

        # MiR speed profiles
        mir_seg = np.linalg.norm(mir_p[1:] - mir_p[:-1], axis=1)
        v0 = mir_seg / np.maximum(np.diff(t0), self.eps_dt)
        v1 = mir_seg / np.maximum(dt_opt, self.eps_dt)

        # Reachability (orig vs opt) using UR fixed in time
        ur_x = ur_p[:, 0].astype(np.float64)
        ur_y = ur_p[:, 1].astype(np.float64)
        ur_z = ur_p[:, 2].astype(np.float64)

        def ur_tcp_at_times(t_vec):
            t = np.clip(t_vec, float(ts0[0]), float(ts0[-1]))
            x = np.interp(t, ts0, ur_x)
            y = np.interp(t, ts0, ur_y)
            z = np.interp(t, ts0, ur_z)
            return np.stack([x, y, z], axis=1)

        tcp0 = ur_tcp_at_times(t0)
        tcp1 = ur_tcp_at_times(t1)
        d0 = np.linalg.norm(tcp0 - ur_base, axis=1)
        d1 = np.linalg.norm(tcp1 - ur_base, axis=1)

        plt.figure()
        plt.plot(v0, label="MiR speed (orig)")
        plt.plot(v1, label="MiR speed (opt)")
        plt.xlabel("segment k")
        plt.ylabel("speed [m/s]")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(d0, label="Reach dist (orig)")
        plt.plot(d1, label="Reach dist (opt)")
        plt.axhline(self.max_reach, linestyle="--", label="max_reach")
        plt.xlabel("k")
        plt.ylabel("distance [m]")
        plt.legend()
        plt.grid(True)

        plt.show()

    def _plot_compare(self, mir_p, ur_p, ur_base, ts0, dt_opt, u_opt):
        N = mir_p.shape[0]
        t0 = ts0
        t1 = np.concatenate([[0.0], np.cumsum(dt_opt)])

        # UR speed in index-space
        u0 = np.arange(N, dtype=np.float64)
        du0 = np.diff(u0) / np.diff(t0)
        du1 = np.diff(u_opt) / np.diff(t1)

        # MiR speed
        mir_seg = np.linalg.norm(mir_p[1:] - mir_p[:-1], axis=1)
        v0 = mir_seg / np.diff(t0)
        v1 = mir_seg / np.diff(t1)

        # Reachability margins
        m0 = []
        m1 = []
        for k in range(N):
            tcp0 = interp_path_pos(ur_p, u0[k])
            tcp1 = interp_path_pos(ur_p, u_opt[k])
            d0 = np.linalg.norm(tcp0 - ur_base[k])
            d1 = np.linalg.norm(tcp1 - ur_base[k])
            m0.append(d0)
            m1.append(d1)
        m0 = np.array(m0)
        m1 = np.array(m1)

        plt.figure()
        plt.plot(du0, label="UR du/dt (orig)")
        plt.plot(du1, label="UR du/dt (opt)")
        plt.xlabel("segment k")
        plt.ylabel("index rate [1/s]")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(v0, label="MiR speed (orig)")
        plt.plot(v1, label="MiR speed (opt)")
        plt.xlabel("segment k")
        plt.ylabel("m/s")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(m0, label="Reach dist (orig)")
        plt.plot(m1, label="Reach dist (opt)")
        plt.axhline(self.max_reach, linestyle="--", label="max_reach")
        plt.xlabel("k")
        plt.ylabel("distance [m]")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(u_opt - np.arange(N), label="offset o_k = u_k - k (opt)")
        plt.xlabel("k")
        plt.ylabel("index offset")
        plt.legend()
        plt.grid(True)

        plt.show()


def main():
    rospy.init_node("mir_ur_timestamp_optimizer")
    opt = MirUrTimestampOptimizer()

    # Run once on startup (simple workflow).
    # If you want periodic re-optimization, wrap in a timer.
    try:
        opt.optimize_once(do_plot=True)
    except Exception as e:
        rospy.logerr(str(e))


if __name__ == "__main__":
    main()