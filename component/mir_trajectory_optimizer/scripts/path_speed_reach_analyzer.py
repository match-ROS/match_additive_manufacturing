#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
from scipy.optimize import minimize

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
        self.pub_u = rospy.Publisher("~opt_u_index", Float32MultiArray, queue_size=1, latch=True)

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
        mir_p, mir_yaw, ur_p, ts0, T = self._wait_inputs()
        N = mir_p.shape[0]

        # Precompute geometry (MiR segment lengths)
        mir_seg = np.linalg.norm(mir_p[1:] - mir_p[:-1], axis=1)  # (N-1,)

        ur_base = self._compute_ur_base_pos(mir_p, mir_yaw)

        # Initial guess:
        # delta_t from original, u_k = k (no offset)
        dt0 = np.diff(ts0)
        dt0 = np.maximum(dt0, self.eps_dt)
        dt0 *= (T / float(np.sum(dt0)))  # enforce sum(dt)=T
        u0 = np.arange(N, dtype=np.float64)

        # Decision vector x = [dt(0..N-2), u(0..N-1)]
        x0 = np.concatenate([dt0, u0], axis=0)

        # Bounds:
        # dt in [eps, inf), u in [0, N-1]
        bnds = [(self.eps_dt, None)] * (N - 1) + [(0.0, float(N - 1))] * N

        r = float(N - 1) / T  # desired du/dt for constant UR progress

        def unpack(x):
            dt = x[:N - 1]
            u = x[N - 1:]
            return dt, u

        def cost(x):
            dt, u = unpack(x)

            # Enforce sum(dt)=T softly (we also add equality constraint)
            # UR "speed" in index-space
            du = np.diff(u)
            ur_speed = du / dt  # (N-1,)

            # MiR speed/acc (geometry / dt)
            mir_speed = mir_seg / dt
            mir_acc = np.diff(mir_speed) / np.maximum(dt[1:], self.eps_dt)

            # Smoothness of u (second difference)
            u_dd = u[2:] - 2.0 * u[1:-1] + u[:-2]

            # Smoothness of dt
            dt_dd = dt[2:] - 2.0 * dt[1:-1] + dt[:-2] if (N - 1) >= 3 else np.array([0.0])

            J = 0.0
            J += self.w_ur_speed * float(np.sum((ur_speed - r) ** 2))
            J += self.w_ur_speed_smooth * float(np.sum(np.diff(ur_speed) ** 2))
            J += self.w_mir_speed * float(np.sum(mir_speed ** 2))
            J += self.w_mir_acc * float(np.sum(mir_acc ** 2))
            J += self.w_u_smooth * float(np.sum(u_dd ** 2))
            J += self.w_dt_smooth * float(np.sum(dt_dd ** 2))
            return J

        # Constraints:
        cons = []

        # sum(dt) = T
        cons.append({
            "type": "eq",
            "fun": lambda x: float(np.sum(x[:N - 1]) - T)
        })

        # u_0 = 0, u_{N-1} = N-1
        cons.append({"type": "eq", "fun": lambda x: float(x[N - 1 + 0] - 0.0)})
        cons.append({"type": "eq", "fun": lambda x: float(x[N - 1 + (N - 1)] - float(N - 1))})

        # Monotonic u: du >= 0
        # SLSQP uses inequality fun(x) >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda x: np.diff(x[N - 1:])  # du
        })

        # Reachability: max_reach - distance >= 0 for each k
        def reach_ineq(x):
            _, u = unpack(x)
            margin = np.zeros((N,), dtype=np.float64)
            for k in range(N):
                tcp = interp_path_pos(ur_p, u[k])
                d = np.linalg.norm(tcp - ur_base[k])
                margin[k] = self.max_reach - d
            return margin

        cons.append({"type": "ineq", "fun": reach_ineq})

        rospy.loginfo("Starting optimization (SLSQP)...")
        res = minimize(
            cost, x0, method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-6, "disp": True}
        )

        if not res.success:
            rospy.logwarn(f"Optimization did not fully converge: {res.message}")

        dt_opt, u_opt = unpack(res.x)

        # Publish results
        msg_dt = Float32MultiArray(data=[float(v) for v in dt_opt])
        msg_u = Float32MultiArray(data=[float(v) for v in u_opt])
        self.pub_delta_t.publish(msg_dt)
        self.pub_u.publish(msg_u)
        rospy.loginfo("Published ~opt_delta_t and ~opt_u_index (latched).")

        if do_plot:
            self._plot_compare(mir_p, ur_p, ur_base, ts0, dt_opt, u_opt)

        return dt_opt, u_opt, res

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
