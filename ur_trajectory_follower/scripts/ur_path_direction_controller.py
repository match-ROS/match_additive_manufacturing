#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32, Bool
import numpy as np
import tf.transformations as tft

class DirectionController:
    def __init__(self):
        # Spray-axis controller: fixed dt assumed
        self.nozzle_height_default = rospy.get_param("~nozzle_height_default", 0.1)
        self.nozzle_height_override = 0.0
        self.kp_z = rospy.get_param("~kp_z", 0.0)
        self.ki_z = rospy.get_param("~ki_z", 0.0)
        self.kd_z = rospy.get_param("~kd_z", 0.0)
        rospy.loginfo(f"Spray-axis gains: kp={self.kp_z}, ki={self.ki_z}, kd={self.kd_z}")
        self.spray_axis_source = rospy.get_param("~spray_axis_source", "tool_z")
        self.spray_axis_sign = float(rospy.get_param("~spray_axis_sign", 1.0))
        self.joint_state_topic = rospy.get_param("~joint_state_topic", "/mur620c/joint_states")
        self.lift_joint_name = rospy.get_param("~lift_joint_name", "right_lift_joint")
        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.0)  # between 0 and 1
        self.integral_z = 0
        self.prev_error_z = 0
        

        self.path: Path = Path()
        self.command_old_twist = Twist()
        self.current_index = 1  # Start at the first waypoint in the path
        self.trajectory_velocity=0
        self.velocity_override=1.0 # in percent
        self.current_lift_height = 0.0
        self.current_pose = None
        self.node_ready = False
        self.from_index_offset = int(rospy.get_param("~from_index_offset", -1))
        self.goal_index_offset = int(rospy.get_param("~goal_index_offset", 0))
        self.start_condition_topic = rospy.get_param("~start_condition_topic", "/start_condition")
        self.wait_for_start_condition = rospy.get_param("~wait_for_start_condition", True)
        self.control_enabled = not self.wait_for_start_condition
        self.path_index_topic = rospy.get_param("~path_index_topic", "/path_index")
        self.initial_path_index = self._parse_initial_path_index(rospy.get_param("~initial_path_index", -1))
        if self.initial_path_index is not None:
            self.current_index = self.initial_path_index
            rospy.loginfo(f"Using initial path index {self.current_index} from parameter.")
        
        self.pub_ur_velocity_world = rospy.Publisher("/ur_twist_world", Twist, queue_size=10)
        rospy.sleep(0.1)  # allow publisher to set up

        self.path = rospy.wait_for_message("/path", Path)        

        rospy.Subscriber(self.path_index_topic, Int32, self.index_callback)
        rospy.Subscriber("/current_pose", PoseStamped, self.ee_pose_callback)
        rospy.Subscriber("/velocity_override", Float32, self.velocity_override_callback)
        rospy.Subscriber("/nozzle_height_override", Float32, self.nozzle_height_callback)        # lift height
        rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        rospy.Subscriber(self.start_condition_topic, Bool, self.start_condition_callback, queue_size=1)

        rospy.wait_for_message(self.joint_state_topic, JointState)
        if self.initial_path_index is None:
            rospy.wait_for_message(self.path_index_topic, Int32)
        else:
            rospy.loginfo(f"Skipping wait for {self.path_index_topic}; initial index provided via parameter.")
        
        self.node_ready = True
        rospy.loginfo("UR Direction Controller node initialized.")
        self.index_callback(Int32(data=self.current_index))  # initial calculation

    @staticmethod
    def _parse_initial_path_index(raw_value):
        try:
            idx = int(raw_value)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, f"Invalid initial_path_index value '{raw_value}', ignoring.")
            return None
        return idx if idx >= 0 else None

    def nozzle_height_callback(self, height_msg: Float32):
        self.nozzle_height_override = height_msg.data

    def joint_state_callback(self, msg: JointState):
        try:
            idx = msg.name.index(self.lift_joint_name)
            self.current_lift_height = msg.position[idx]
        except ValueError:
            pass  # joint not found in this message


    def index_callback(self, index_msg: Int32):
        self.current_index = index_msg.data
        if not self.node_ready:
            return
        if not self.control_enabled:
            return

        self.get_traj_velocity(self.from_index_offset, self.goal_index_offset)
        self.calculate_twist(self.from_index_offset, self.goal_index_offset)

    def start_condition_callback(self, msg: Bool):
        new_state = bool(msg.data) or not self.wait_for_start_condition
        if new_state == self.control_enabled:
            return

        if new_state:
            rospy.loginfo("Start condition fulfilled – enabling UR direction controller output.")
        else:
            rospy.loginfo("Start condition reset – holding UR direction controller output.")
            self.command_old_twist = Twist()
            self.integral_z = 0
            self.prev_error_z = 0
            self.pub_ur_velocity_world.publish(Twist())

        self.control_enabled = new_state

    def velocity_override_callback(self, velocity_msg: Float32):
        self.velocity_override = velocity_msg.data

    def ee_pose_callback(self, pose_msg: PoseStamped):
        self.current_pose = pose_msg
        if not self.node_ready:
            return
        self.calculate_twist(self.from_index_offset, self.goal_index_offset)

    def _clamp_path_index(self, target_index: int) -> int:
        if not self.path.poses:
            return 0
        return max(0, min(target_index, len(self.path.poses) - 1))

    def get_traj_velocity(self, from_offset: int, goal_offset: int):
        """Calculate the velocity of the robot along the trajectory.
        The velocity is calculated as the distance between the last waypoint and the next waypoint divided by the time.
        """
        if not self.path.poses:
            rospy.logwarn("Received empty path; trajectory velocity set to zero.")
            self.trajectory_velocity = 0.0
            return
        last_idx = self._clamp_path_index(self.current_index + from_offset)
        next_idx = self._clamp_path_index(self.current_index + goal_offset)
        if last_idx == next_idx:
            rospy.logwarn("Configured waypoint offsets select identical indices; trajectory velocity set to zero.")
            self.trajectory_velocity = 0.0
            return

        # Get the last waypoint and the next waypoint
        last_waypoint = self.path.poses[last_idx]
        next_waypoint = self.path.poses[next_idx]

        # Compute the absolute velocity from the last waypoint to the next waypoint by dividing the distance by the time
        # The time is the difference between the timestamps of the two waypoints
        distance = ((next_waypoint.pose.position.x - last_waypoint.pose.position.x)**2 + (next_waypoint.pose.position.y - last_waypoint.pose.position.y)**2)**0.5
        dt = (next_waypoint.header.stamp - last_waypoint.header.stamp).to_sec()
        if dt > 0:
            self.trajectory_velocity = distance / dt
        else:
            rospy.logwarn("time difference <=0 encountered in trajectory velocity calculation.")
            self.trajectory_velocity = 0.0

    def _get_goal_pose(self, goal_offset: int) -> PoseStamped:
        if not self.path.poses:
            return None
        goal_idx = self._clamp_path_index(self.current_index + goal_offset)
        return self.path.poses[goal_idx]

    @staticmethod
    def _normalize_vector(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return fallback
        return vec / norm

    def _get_spray_axis(self, goal_pose: PoseStamped) -> np.ndarray:
        orientation = goal_pose.pose.orientation
        quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=float)
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6:
            return np.array([0.0, 0.0, 1.0])
        quat /= quat_norm
        rotation = tft.quaternion_matrix(quat)
        axes = {
            "tool_x": rotation[0:3, 0],
            "tool_y": rotation[0:3, 1],
            "tool_z": rotation[0:3, 2],
        }
        axis = axes.get(self.spray_axis_source, axes["tool_z"])
        axis = self._normalize_vector(axis, np.array([0.0, 0.0, 1.0]))
        if self.spray_axis_sign < 0.0:
            axis = -axis
        return axis

    def get_direction(self, from_offset: int, goal_offset: int):
        """Get the motion direction orthogonal to the spray axis and the spray-axis error.
        Returns:
            direction_plane_norm (np.array): normalized direction orthogonal to spray axis (world frame).
            error_spray (float): signed error along the spray axis.
            spray_axis (np.array): normalized spray axis in world frame.
        """
        if not self.path.poses or self.current_pose is None:
            return np.zeros(3), 0.0, np.array([0.0, 0.0, 1.0])

        goal_pose = self._get_goal_pose(goal_offset)
        if goal_pose is None:
            return np.zeros(3), 0.0, np.array([0.0, 0.0, 1.0])

        direction = np.array([
            goal_pose.pose.position.x - self.current_pose.pose.position.x,
            goal_pose.pose.position.y - self.current_pose.pose.position.y,
            goal_pose.pose.position.z - self.current_pose.pose.position.z,
        ])
        spray_axis = self._get_spray_axis(goal_pose)
        error_spray = float(np.dot(direction, spray_axis))
        direction_plane = direction - error_spray * spray_axis
        direction_plane_norm = self._normalize_vector(direction_plane, np.zeros(3))
        return direction_plane_norm, error_spray, spray_axis

    def smooth_output(self, control_command: Twist):
        """Smooth the output command using exponential moving average."""
        smoothed_command = Twist()
        smoothed_command.linear.x = (self.output_smoothing_coeff * self.command_old_twist.linear.x +
                                     (1 - self.output_smoothing_coeff) * control_command.linear.x)
        smoothed_command.linear.y = (self.output_smoothing_coeff * self.command_old_twist.linear.y +
                                     (1 - self.output_smoothing_coeff) * control_command.linear.y)
        smoothed_command.linear.z = (self.output_smoothing_coeff * self.command_old_twist.linear.z +
                                     (1 - self.output_smoothing_coeff) * control_command.linear.z)
        self.command_old_twist = smoothed_command

        return smoothed_command


    def calculate_twist(self, from_offset: int, goal_offset: int):
        """Control the direction of the robot to follow the path."""

        rospy.logdebug(f"Calculating twist at index {self.current_index} with from_offset {from_offset} and goal_offset {goal_offset}.")
        if self.current_pose is None:
            rospy.logwarn("No current pose received yet.")
            return
        if not self.control_enabled:
            return
        
        direction_plane_norm, error_spray, spray_axis = self.get_direction(from_offset, goal_offset)
        v_plane = direction_plane_norm * self.trajectory_velocity * self.velocity_override
        
        # Spray-axis PID controller (fixed dt assumed).
        error_spray += self.nozzle_height_default + self.nozzle_height_override
        v_spray = (
            error_spray * self.kp_z
            + self.integral_z * self.ki_z
            + (error_spray - self.prev_error_z) * self.kd_z
        )
        self.integral_z += error_spray
        self.prev_error_z = error_spray

        v_spray_vec = spray_axis * v_spray
        v_cmd = v_plane + v_spray_vec

        # Create a Twist message to publish the control command (world_frame)
        control_command = Twist()
        control_command.linear.x = v_cmd[0]
        control_command.linear.y = v_cmd[1]
        control_command.linear.z = v_cmd[2]
        control_command_smoothed = self.smooth_output(control_command)

        self.pub_ur_velocity_world.publish(control_command_smoothed)

if __name__ == "__main__":
    rospy.init_node("ur_direction_controller")
    direction_controller = DirectionController()
    rospy.spin()