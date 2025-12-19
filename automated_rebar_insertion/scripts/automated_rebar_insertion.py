#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import rospy
import roslaunch
import numpy as np
import tf

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import tf.transformations as tr

# MoveIt
import moveit_commander
from moveit_msgs.msg import Constraints, JointConstraint

# UR SetIO (ur_robot_driver / ur_msgs)
try:
    from ur_msgs.srv import SetIO, SetIORequest
except Exception:
    SetIO = None


class RebarAutomationNode:
    def __init__(self):
        rospy.init_node("rebar_automation_node", anonymous=False)

        # --- Params
        self.robot_name = rospy.get_param("~robot_name", "mur620d")
        self.UR_prefix = rospy.get_param("~UR_prefix", "UR10_r")
        self.start_index = int(rospy.get_param("~start_index", 600))
        self.step = int(rospy.get_param("~step", 50))
        self.max_cycles = int(rospy.get_param("~max_cycles", 999999))  # safety
        self.cm_switch_srv = f"/{self.robot_name}/{self.UR_prefix}/controller_manager/switch_controller"
        self.arm_controller_name = rospy.get_param("~arm_controller_name", "arm_controller")
        self.twist_controller_name = rospy.get_param("~twist_controller_name", "twist_controller")
        self.twist_cmd_topic = rospy.get_param("~twist_cmd_topic", f"/{self.robot_name}/{self.UR_prefix}/twist_controller/command_collision_free")
        self.twist_frame_id = rospy.get_param("~twist_frame_id", f"{self.robot_name}/base_link")
        self.twist_rate_hz = int(rospy.get_param("~twist_rate_hz", 250))
        self.path_topic = rospy.get_param("~path_topic", "/ur_path_original")

        self.spray_distance_up = float(rospy.get_param("~spray_distance_up", 0.65))
        self.insert_delta = float(rospy.get_param("~insert_delta", 0.05))  # 50 mm
        self.spray_distance_down = self.spray_distance_up - self.insert_delta

        self.cart_step = float(rospy.get_param("~cart_step", 0.002))          # eef_step [m]
        self.cart_jump_thresh = float(rospy.get_param("~cart_jump_thresh", 0.0))  # disable jump check
        self.cart_fraction_min = float(rospy.get_param("~cart_fraction_min", 0.98))

        self.node_start_delay = float(rospy.get_param("~node_start_delay", 0.0))

        # Launch files (absolute paths recommended)
        self.mir_launch = rospy.get_param("~mir_launch_file", "move_mir_to_start_pose.launch")
        self.ur_launch = rospy.get_param("~ur_launch_file", "move_ur_to_start_pose.launch")

        # MoveIt
        self.move_group_ns = rospy.get_param("~move_group_ns", f"/{self.robot_name}/move_group")
        self.planning_group = rospy.get_param("~planning_group", "UR_arm_r")
        self.prepos_joints = rospy.get_param("~prepos_joints", [0.298, -0.764, 1.431, -2.741, -1.572, 2.328])
        self.mag_joints    = rospy.get_param("~magazine_joints", [0.3, -1.308, 1.5, -1.764, -1.579])
        self.manipulator_base_link = rospy.get_param('~manipulator_base_link', 'base_footprint')
        self.manipulator_tcp_link = rospy.get_param('~manipulator_tcp_link', 'UR10_r/tool0')

        # MiR "stillstand" detection
        self.mir_still_pos_eps = float(rospy.get_param("~mir_still_pos_eps", 0.003))  # [m] e.g. 3 mm
        self.mir_still_time = float(rospy.get_param("~mir_still_time", 1.0))          # [s] must be stable this long
        self.mir_pose_timeout = float(rospy.get_param("~mir_pose_timeout", 2.0))      # [s] wait for pose updates
        self.mir_post_wait_max = float(rospy.get_param("~mir_post_wait_max", 30.0))   # [s] overall wait cap

        # UR I/O
        self.io_service = rospy.get_param("~io_service", f"/{self.robot_name}/{self.UR_prefix}/ur_hardware_interface/set_io")
        self.io_pin = int(rospy.get_param("~io_pin", 1))
        self.io_close_value = float(rospy.get_param("~io_close_value", 1.0))
        self.io_open_value = float(rospy.get_param("~io_open_value", 0.0))
        self.io_wait_s = float(rospy.get_param("~io_wait_s", 10.0))

        # Optional: pose topics (currently not strictly required; kept for later checks)
        self.mir_pose_topic = rospy.get_param("~mir_pose_topic", f"/{self.robot_name}/mir_pose_simple")
        self.ur_tcp_topic = rospy.get_param("~ur_tcp_topic", f"/{self.robot_name}/UR10_r/global_tcp_pose")

        self._mir_pose = None
        self._ur_tcp = None
        self.ur_path = None
        rospy.Subscriber(self.mir_pose_topic, Pose, self._cb_mir_pose, queue_size=1)
        rospy.Subscriber(self.ur_tcp_topic, PoseStamped, self._cb_ur_tcp, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self._cb_path, queue_size=1)
        self.twist_pub = rospy.Publisher(self.twist_cmd_topic, Twist, queue_size=1)
        self.tf_listener = tf.TransformListener()
        
        # Wait for initial messages / setup
        rospy.loginfo("RebarAutomationNode: waiting for initial topics...")
        rospy.wait_for_message(self.path_topic, Path, timeout=5.0)  # oder warten bis self.ur_path != None
        rospy.loginfo("RebarAutomationNode: initial topics received.")

        # roslaunch UUID
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

        # MoveIt init
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander(self.planning_group , robot_description= '/' + self.robot_name + '/robot_description', ns=self.robot_name)

        rospy.loginfo("RebarAutomationNode ready. robot=%s start_index=%d step=%d up=%.3f down=%.3f",
                      self.robot_name, self.start_index, self.step,
                      self.spray_distance_up, self.spray_distance_down)

    def _cb_mir_pose(self, msg: Pose):
        self._mir_pose = msg

    def _cb_ur_tcp(self, msg: PoseStamped):
        self._ur_tcp = msg

    def _cb_path(self, msg: Path):
        self.ur_path = msg

    def run(self):
        idx = self.start_index
        # Initial move MiR to start index
        self._run_mir_to_index(idx)

        for cycle in range(self.max_cycles):
            if rospy.is_shutdown():
                break

            rospy.loginfo("=== Cycle %d | index=%d ===", cycle, idx)
            # 0) Reset gripper to open
            self._set_ur_digital_out(self.io_pin, self.io_open_value)

            # 1) Move UR to index (start pose)
            self._wait_mir_stopped() # ensure MiR is stable before continuing 
            #self._run_ur_to_index(idx, self.spray_distance_up)
            self.move_ur_to_index_fast(idx, self.spray_distance_up)

            # 2) Insert fast (down)
            self.move_relative_z_twist_c2(dz_m=-0.15, duration_s=1.35)  # 50 mm runter

            # 3) Retract fast (up)
            self.move_relative_z_twist_c2(dz_m=+0.15, duration_s=1.35)  # 50 mm hoch

            # 4) Move MiR to next index
            idx += self.step
            self._run_mir_to_index(idx)

            # 5) Go to magazine sequence + close gripper
            #self._moveit_joints(self.prepos_joints)
            self._moveit_joints_partial(self.mag_joints)
            self._set_ur_digital_out(self.io_pin, self.io_close_value)
            rospy.sleep(1.0) 
            self._set_ur_digital_out(self.io_pin, self.io_open_value)
            rospy.sleep(10.0)
            
            # 6) Back to prepos
            #self._moveit_joints(self.prepos_joints)


        rospy.loginfo("RebarAutomationNode finished.")

    # ---------- Actions ----------
    def _run_mir_to_index(self, index: int):
        args = {
            "robot_name": self.robot_name,
            "initial_path_index": str(index),
            # "path_topic": "/mir_path_transformed"  # keep default from launch unless needed
        }
        self._run_launch_blocking(self.mir_launch, args, "MiR->index")

    def _run_ur_to_index(self, index: int, spray_distance: float):
        args = {
            "robot_name": self.robot_name,
            "initial_path_index": str(index),
            "spray_distance": f"{spray_distance:.3f}",
            "node_start_delay": f"{self.node_start_delay:.3f}",
            # "planning_group": self.planning_group,  # keep default unless needed
        }
        self._run_launch_blocking(self.ur_launch, args, f"UR->index (spray_distance={spray_distance:.3f})")

    def _moveit_joints(self, joint_values, wait: bool = True):
        """
        joint_values: list[float] in the MoveGroup joint order
                    (use group.get_active_joints() to verify order)
        """
        rospy.loginfo("MoveIt: go joint target (%d joints)", len(joint_values))
        self.group.set_joint_value_target(joint_values)
        ok = self.group.go(wait=wait)
        self.group.stop()
        self.group.clear_pose_targets()
        if not ok:
            raise RuntimeError("MoveIt failed to reach joint target")

    def _moveit_joints_partial(self, joint_values_first5, wait: bool = True):
            """
            Provide only the first 5 joint targets (in MoveGroup order).
            Joint 6 stays at its current value.
            """
            if len(joint_values_first5) != 5:
                raise ValueError("Expected exactly 5 joint values")

            current = self.read_q_from_joint_states(f"/{self.robot_name}/joint_states", timeout=2.0)
            if len(current) < 6:
                raise RuntimeError(f"MoveGroup has only {len(current)} joints, expected >= 6")

            target = current[:]                 # keep all as-is
            target[:5] = joint_values_first5    # overwrite first 5

            rospy.loginfo("MoveIt: go partial joint target (first 5 fixed, last kept)")
            self.group.set_joint_value_target(target)
            ok = self.group.go(wait=wait)
            self.group.stop()
            self.group.clear_pose_targets()
            if not ok:
                raise RuntimeError("MoveIt failed to reach partial joint target")


    def _minimum_jerk_v(self, t, T, dz):
        """C2: v(t) for minimum-jerk position from 0..dz in time T."""
        s = max(0.0, min(1.0, t / T))
        # x(s) = dz*(10s^3 - 15s^4 + 6s^5)
        # v(s) = dz/T*(30s^2 - 60s^3 + 30s^4)
        return (dz / T) * (30*s*s - 60*s*s*s + 30*s*s*s*s)

    def _switch_controllers(self, start_list, stop_list, timeout=3.0):
        rospy.wait_for_service(self.cm_switch_srv, timeout=timeout)
        sw = rospy.ServiceProxy(self.cm_switch_srv, SwitchController)

        req = SwitchControllerRequest()
        req.start_controllers = start_list
        req.stop_controllers = stop_list
        req.strictness = 0 #SwitchControllerRequest.STRICT
        req.start_asap = True

        # FIX: timeout is float64 seconds in ROS1
        req.timeout = float(timeout)

        resp = sw(req)
        if not resp.ok:
            raise RuntimeError(f"Controller switch failed. start={start_list}, stop={stop_list}")

    def move_relative_z_twist_c2(self, dz_m: float, duration_s: float = 0.35):
        """
        Switch to twist controller, move dz in z with C2 profile, then switch back.
        dz_m: +up / -down (meters)
        duration_s: total motion time
        """

        if duration_s <= 0.05:
            duration_s = 0.05

        # 1) switch to twist controller
        self._switch_controllers(
            start_list=[self.twist_controller_name],
            stop_list=[self.arm_controller_name],
            timeout=5.0
        )

        rospy.sleep(0.05)  # tiny settle

        # 2) run minimum-jerk velocity profile
        rate = rospy.Rate(self.twist_rate_hz)
        t0 = rospy.Time.now()
        last = t0

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            t = (now - t0).to_sec()
            if t >= duration_s:
                break

            vz = self._minimum_jerk_v(t, duration_s, dz_m)

            msg = Twist()
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = vz
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0

            self.twist_pub.publish(msg)
            last = now
            rate.sleep()

        # 3) send a few zero commands to stop cleanly
        for _ in range(int(0.1 * self.twist_rate_hz)):
            now = rospy.Time.now()
            msg = Twist()
            self.twist_pub.publish(msg)
            rate.sleep()

        # 4) switch back to arm controller (MoveIt continues)
        self._switch_controllers(
            start_list=[self.arm_controller_name],
            stop_list=[self.twist_controller_name],
            timeout=5.0
        )

    def _ur_target_pose_from_index(self, idx: int, spray_distance: float):
        if self.ur_path is None:
            raise RuntimeError("UR path not received")

        p = self.ur_path.poses[idx].pose.position

        # Zielpunkt in map (inkl. offsets)
        target_map = np.array([p.x, p.y, p.z  + spray_distance])

        # TF: map -> robot/base_link
        now = rospy.Time(0)
        base_tf = f"{self.robot_name}/{self.manipulator_base_link}"  
        self.tf_listener.waitForTransform("map", base_tf, now, rospy.Duration(1.0))
        trans, rot = self.tf_listener.lookupTransform("map", base_tf, now)

        base_pos_map = np.array(trans)

        # relative Position in map
        rel_map = target_map - base_pos_map

        # in Base-Frame rotieren (inverse rot)
        R = tr.quaternion_matrix(tr.quaternion_inverse(rot))
        rel_base = (R[:3, :3] @ rel_map.reshape(3, 1)).reshape(3)

        # PoseStamped im Base-Frame
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = self.manipulator_base_link
        ps.pose.position.x = float(rel_base[0])
        ps.pose.position.y = float(rel_base[1])
        ps.pose.position.z = float(rel_base[2])

        # Orientation: konstant (Yaw egal)
        # z.B. tool "nach unten": roll=pi, pitch=0, yaw=0
        q = tr.quaternion_from_euler(np.pi, 0.0, 0.0)
        ps.pose.orientation = Quaternion(*q)
        return ps

    def move_ur_to_index_fast(self, idx: int, spray_distance: float):
        target = self._ur_target_pose_from_index(idx, spray_distance)
        self.group.set_pose_target(target, end_effector_link=self.manipulator_tcp_link)

        constraints = Constraints()
        # Ellbogen oben (z. B. nahe -2.0 rad)
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/shoulder_lift_joint",
            position=-0.5,
            tolerance_above=1.0,
            tolerance_below=1.0,
            weight=1.0
        ))
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/shoulder_pan_joint",
            position=0.0,
            tolerance_above=2.5,
            tolerance_below=0.5,
            weight=1.0
        ))
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/wrist_1_joint",
            position=-2.1,
            tolerance_above=1.1,
            tolerance_below=1.1,
            weight=1.0
        ))
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/wrist_2_joint",
            position=-1.5,
            tolerance_above=1.1,
            tolerance_below=1.1,
            weight=1.0
        ))

        self.group.set_path_constraints(constraints)

        ok = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        if not ok:
            raise RuntimeError("MoveIt failed to reach UR target pose")


    def _wait_mir_stopped(self):
        """
        Wait until MiR pose is stable for mir_still_time seconds.
        Uses position delta threshold to tolerate localization noise.
        """
        rospy.loginfo("MiR: waiting for standstill (eps=%.4fm for %.2fs)",
                    self.mir_still_pos_eps, self.mir_still_time)

        # wait for first pose
        t_start = time.time()
        while not rospy.is_shutdown() and self._mir_pose is None:
            if time.time() - t_start > self.mir_pose_timeout:
                raise TimeoutError("MiR: no pose received on mir_pose_topic")
            rospy.sleep(0.05)

        last_pose = self._mir_pose
        last_change_t = time.time()
        t0 = time.time()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            cur = self._mir_pose
            if cur is None:
                rospy.sleep(0.05)
                continue

            dx = cur.position.x - last_pose.position.x
            dy = cur.position.y - last_pose.position.y
            d = (dx*dx + dy*dy) ** 0.5

            if d > self.mir_still_pos_eps:
                last_change_t = time.time()
                last_pose = cur

            if (time.time() - last_change_t) >= self.mir_still_time:
                rospy.loginfo("MiR: standstill confirmed.")
                return

            if (time.time() - t0) > self.mir_post_wait_max:
                raise TimeoutError("MiR: standstill not reached within mir_post_wait_max")

            rate.sleep()


    def _set_ur_digital_out(self, pin: int, value: float):
        if SetIO is None:
            raise RuntimeError("ur_msgs/SetIO not available. Install ur_msgs or adjust to your driver interface.")

        rospy.loginfo("UR IO: set DO%d=%.1f via %s", pin, value, self.io_service)
        rospy.wait_for_service(self.io_service, timeout=10.0)
        srv = rospy.ServiceProxy(self.io_service, SetIO)

        req = SetIORequest()
        req.fun = SetIORequest.FUN_SET_DIGITAL_OUT
        req.pin = pin
        req.state = value
        srv(req)

    def read_q_from_joint_states(self, joint_states_topic: str, timeout=2.0):
        """
        joint_names_order: e.g. self.group.get_active_joints() -> q1..q6 order
        returns: list[float] of joint positions in that order
        """
        msg = rospy.wait_for_message(joint_states_topic, JointState, timeout=timeout)

        # Build name -> index map
        idx = {n: i for i, n in enumerate(msg.name)}

        q = []
        missing = []
        joint_names = self.group.get_active_joints()
        print("MoveIt: joint names:", ", ".join(joint_names))
        for jn in joint_names:
            i = idx.get(jn, None)
            if i is None:
                missing.append(jn)
                q.append(float("nan"))
            else:
                # JointState position might be shorter in broken msgs; guard it
                if i >= len(msg.position):
                    missing.append(jn)
                    q.append(float("nan"))
                else:
                    q.append(msg.position[i])

        if missing:
            rospy.logwarn("Missing joints in %s: %s", joint_states_topic, ", ".join(missing))

        return q

    # ---------- roslaunch helper ----------
    def _run_launch_blocking(self, launch_file: str, args: dict, label: str, timeout_s: float = 300.0):
        launch_file = os.path.expanduser(launch_file)
        if not os.path.isabs(launch_file):
            # Allow using $(find pkg)/... style by resolving via roslaunch
            resolved = roslaunch.rlutil.resolve_launch_arguments([launch_file])[0]
            launch_file = resolved

        cli_args = [launch_file] + [f"{k}:={v}" for k, v in args.items()]
        roslaunch_files = [(launch_file, cli_args[1:])]

        parent = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_files, is_core=False)

        rospy.loginfo("Launch start: %s | %s", label, " ".join(cli_args))
        parent.start()

        t0 = time.time()
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown():
                # If the launched node exits, pm should have no alive processes
                alive = True
                try:
                    alive = parent.pm.is_alive()
                except Exception:
                    # fallback: assume alive for a short while
                    alive = True

                if not alive:
                    break

                if (time.time() - t0) > timeout_s:
                    raise TimeoutError(f"Timeout in launch '{label}' after {timeout_s}s")

                rate.sleep()
        finally:
            rospy.loginfo("Launch shutdown: %s", label)
            parent.shutdown()


if __name__ == "__main__":
    try:
        node = RebarAutomationNode()
        node.run()
    except Exception as e:
        rospy.logerr("RebarAutomationNode error: %s", str(e))
        raise
