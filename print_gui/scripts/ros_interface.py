import subprocess
import yaml
import threading
import rospy
from geometry_msgs.msg import PoseStamped
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import QTimer
import tf.transformations as tf_trans
from rosgraph_msgs.msg import Log
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import BatteryState

import rospy
from geometry_msgs.msg import PoseStamped

class ROSInterface:
    def __init__(self, gui):
        self.gui = gui
        self.updated_poses = {}
        self.virtual_object_pose = None
        self.battery_states = {}  
        self.active_battery_subs = set()  
        self.current_index = 0
        # … your existing init …
        rospy.Subscriber('/path_index', Int32, self._path_idx_cb, queue_size=10)

    def _path_idx_cb(self, msg: Int32):
        self.current_index = msg.data
        
    def subscribe_to_relative_poses(self):
        """Abonniert die relativen Posen der ausgewählten Roboter und speichert sie in YAML."""
        selected_robots = self.gui.get_selected_robots()
        selected_urs = self.gui.get_selected_urs()

        if not selected_robots or not selected_urs:
            print("No robots or URs selected. Skipping update.")
            return

        if not rospy.core.is_initialized():
            rospy.init_node("update_relative_poses", anonymous=True, disable_signals=True)
        self.updated_poses = {}

        def callback(data, robot_ur):
            """Receives relative pose and extracts both position (x, y, z) and orientation (Rx, Ry, Rz) in Euler angles."""
            
            # Extract position
            position = data.pose.position
            x, y, z = position.x, position.y, position.z

            # Extract orientation as quaternion
            orientation = data.pose.orientation
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            rx, ry, rz = tf_trans.euler_from_quaternion(quaternion)

            # Store the values in the dictionary
            self.updated_poses[robot_ur] = {"x": x, "y": y, "z": z, "rx": rx, "ry": ry, "rz": rz}

            print(f"✅ Received pose for {robot_ur}: {self.updated_poses[robot_ur]}")

            # Check if all poses have been received, then trigger save
            if len(self.updated_poses) >= len(selected_robots) * len(selected_urs):
                rospy.signal_shutdown("Pose update complete")
                self.save_poses_to_yaml()



        for robot in selected_robots:
            for ur in selected_urs:
                topic_name = f"/{robot}/{ur}/relative_pose"
                rospy.Subscriber(topic_name, PoseStamped, callback, (robot, ur))
                print(f"Subscribed to {topic_name}")

        rospy.spin()

    def start_roscore(self):
        """Starts roscore on the roscore PC."""
        command = "ssh -t -t roscore 'source ~/.bashrc; source /opt/ros/noetic/setup.bash; roscore; exec bash'"
        subprocess.Popen(["terminator", "--title=Roscore Terminal", "-x", f"{command}; exec bash"])

    def start_mocap(self):
        """Starts the motion capture system on the roscore PC."""
        command = "ssh -t -t roscore 'source ~/.bashrc; source /opt/ros/noetic/setup.bash; source ~/catkin_ws/devel/setup.bash; roslaunch launch_mocap mocap_launch.launch; exec bash'"
        subprocess.Popen(["terminator", "--title=Mocap", "-x", f"{command}; exec bash"])

    def start_sync(self):
        """Starts file synchronization between workspace and selected robots."""
        selected_robots = self.gui.get_selected_robots()
        self.workspace_name = self.gui.get_workspace_name()
        self.gui.btn_sync.setStyleSheet("background-color: lightgreen;")  # Mark sync as active
        
        for robot in selected_robots:
            command = f"while inotifywait -r -e modify,create,delete,move ~/{self.workspace_name}/src; do \n" \
                      f"rsync --delete -avzhe ssh ~/{self.workspace_name}/src rosmatch@{robot}:~/{self.workspace_name}/ \n" \
                      "done"
            subprocess.Popen(["terminator", f"--title=Sync to {robot}", "-x", f"{command}; exec bash"]) 

    def update_button_status(self):
        """Checks if roscore and mocap are running and updates button colors."""
        roscore_running = self.is_ros_node_running("/rosout")
        mocap_running = self.is_ros_node_running("/qualisys")

        self.gui.btn_roscore.setStyleSheet("background-color: lightgreen;" if roscore_running else "background-color: lightgray;")
        self.gui.btn_mocap.setStyleSheet("background-color: lightgreen;" if mocap_running else "background-color: lightgray;")

    def is_ros_node_running(self, node_name):
        """Checks if a specific ROS node is running by using `rosnode list`."""
        try:
            output = subprocess.check_output("rosnode list", shell=True).decode()
            return node_name in output.split("\n")
        except subprocess.CalledProcessError:
            return False
        
    def make_ur_callback(self, robot):
        def callback(msg):
            if robot not in self.battery_states:
                self.battery_states[robot] = {}
            self.battery_states[robot]["ur"] = msg.data
            self.update_battery_display(robot)
        return callback

    def make_mir_callback(self, robot):
        def callback(msg):
            if robot not in self.battery_states:
                self.battery_states[robot] = {}
            self.battery_states[robot]["mir"] = msg.percentage * 100  # falls 0.0–1.0
            self.update_battery_display(robot)
        return callback

    def update_battery_display(self, robot):
        mir_label, ur_label = self.gui.battery_labels.get(robot, (None, None))
        if not mir_label or not ur_label:
            return

        state = self.battery_states.get(robot, {})

        if "mir" in state:
            val = state["mir"]
            mir_label.setText(f"MiR: {val:.0f}%")
            mir_label.setStyleSheet(f"background-color: {self.get_color(val)}; padding: 2px;")

        if "ur" in state:
            val = state["ur"]
            ur_label.setText(f"UR: {val:.0f}%")
            ur_label.setStyleSheet(f"background-color: {self.get_color(val)}; padding: 2px;")


    def check_and_subscribe_battery(self):
        if not rospy.core.is_initialized():
            rospy.init_node("battery_status_gui", anonymous=True, disable_signals=True)

        for robot in self.gui.get_selected_robots():
            if robot in self.active_battery_subs:
                continue  # bereits abonniert

            rospy.Subscriber(f"/{robot}/bms_status/SOC", Float32, self.make_ur_callback(robot), queue_size=1)
            rospy.Subscriber(f"/{robot}/battery_state", BatteryState, self.make_mir_callback(robot))
            self.active_battery_subs.add(robot)
            print(f"🔌 Subscribed to battery topics for {robot}")

    def get_color(self, val):
        if val >= 50:
            return "lightgreen"
        elif val >= 20:
            return "orange"
        else:
            return "red"

    def launch_keyence_scanner(self):
        """Launches the Keyence scanner on the robots PC."""
        selected_robots = self.gui.get_selected_robots()
        workspace_name = self.gui.get_workspace_name()

        for robot in selected_robots:
            workspace = workspace_name
            command = f"ssh -t -t {robot} 'source ~/.bashrc; export ROS_MASTER_URI=http://roscore:11311/; source /opt/ros/noetic/setup.bash; source ~/{workspace}/devel/setup.bash; roslaunch laser_scanner_tools keyence_scanner.launch; exec bash'"

            # Open a new terminal with SSH session + driver launch + keep open
            subprocess.Popen([
                "terminator",
                f"--title=Driver {robot}",      # Set the window title to "Mur Driver" :contentReference[oaicite:0]{index=0}
                "-x",                       # Execute the following command inside the terminal :contentReference[oaicite:1]{index=1}
                f"{command}; exec bash"
            ])


def launch_ros(gui, package, launch_file):
    selected_robots = gui.get_selected_robots()
    robot_names_str = "[" + ",".join(f"'{r}'" for r in selected_robots) + "]"

    command = f"roslaunch {package} {launch_file} robot_names:={robot_names_str}"
    print(f"Executing: {command}")
    subprocess.Popen(command, shell=True)





def start_status_update(gui):
    threading.Thread(target=update_status, args=(gui,), daemon=True).start()

def update_status(gui):
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()
    active_counts = {"force_torque_sensor_controller": 0, "twist_controller": 0, "arm_controller": 0, "admittance": 0}
    total_count = len(selected_robots) * len(selected_urs)
    
    for robot in selected_robots:
        for ur in selected_urs:
            service_name = f"/{robot}/{ur}/controller_manager/list_controllers"
            try:
                output = subprocess.check_output(f"rosservice call {service_name}", shell=True).decode()
                controllers = yaml.safe_load(output).get("controller", [])
                for controller in controllers:
                    if controller.get("state") == "running":
                        active_counts[controller["name"]] += 1
            except Exception:
                pass
    
    status_text = """
    Force/Torque Sensor: {}/{} {}
    Twist Controller: {}/{} {}
    Arm Controller: {}/{} {}
    Admittance Controller: {}/{} {}
    """.format(
        active_counts["force_torque_sensor_controller"], total_count, get_status_symbol(active_counts["force_torque_sensor_controller"], total_count),
        active_counts["twist_controller"], total_count, get_status_symbol(active_counts["twist_controller"], total_count),
        active_counts["arm_controller"], total_count, get_status_symbol(active_counts["arm_controller"], total_count),
        active_counts["admittance"], total_count, get_status_symbol(active_counts["admittance"], total_count),
    )
    
    gui.status_label.setText(status_text)

def get_status_symbol(active, total):
    if active == total:
        return "✅"
    elif active > 0:
        return "⚠️"
    return "❌"

def open_rviz():
    command = "roslaunch print_gui launch_rviz.launch"
    subprocess.Popen(command, shell=True)

def quit_drivers():
    print("Stopping all drivers...")
    subprocess.Popen("pkill -f 'roslaunch'", shell=True)



def turn_on_arm_controllers(gui):
    """Turns on all arm controllers for the selected robots."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping launch.")
        return

    robot_names_str = '"' + str(selected_robots).replace("'", '"') + '"'
    ur_prefixes_str = '"' + str(selected_urs).replace("'", '"') + '"'

    command = f"roslaunch print_gui turn_on_all_arm_controllers.launch robot_names:={robot_names_str} UR_prefixes:={ur_prefixes_str}"
    print(f"Executing: {command}")
    subprocess.Popen(command, shell=True)

def turn_on_twist_controllers(gui):
    """Turns on all twist controllers for the selected robots."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping launch.")
        return

    robot_names_str = '"' + str(selected_robots).replace("'", '"') + '"'
    ur_prefixes_str = '"' + str(selected_urs).replace("'", '"') + '"'

    command = f"roslaunch print_gui turn_on_all_twist_controllers.launch robot_names:={robot_names_str} UR_prefixes:={ur_prefixes_str}"
    print(f"Executing: {command}")
    subprocess.Popen(command, shell=True)

def enable_all_urs(gui):
    """Enables all UR robots for the selected configurations."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping launch.")
        return

    robot_names_str = '"' + str(selected_robots).replace("'", '"') + '"'
    ur_prefixes_str = '"' + str(selected_urs).replace("'", '"') + '"'

    command = f"roslaunch print_gui enable_all_URs.launch robot_names:={robot_names_str} UR_prefixes:={ur_prefixes_str}"
    print(f"Executing: {command}")
    subprocess.Popen(command, shell=True)


def launch_drivers(gui):
    """SSH into the selected robots and start the drivers in separate terminals."""
    selected_robots = gui.get_selected_robots()
    workspace_name = gui.get_workspace_name()

    for robot in selected_robots:
        workspace = workspace_name
        # command = f"ssh -t -t {robot} 'source ~/.bashrc; export ROS_MASTER_URI=http://roscore:11311/; source /opt/ros/noetic/setup.bash; source ~/{workspace}/devel/setup.bash; roslaunch mur_launch_hardware {robot}.launch; exec bash'"
        selected_urs = gui.get_selected_urs()
        launch_suffix=""
        if "UR10_l" in selected_urs:
            launch_suffix += " launch_ur_l:=true"
        else:
            launch_suffix += " launch_ur_l:=false"
        if "UR10_r" in selected_urs:
            launch_suffix += " launch_ur_r:=true"
        else:
            launch_suffix += " launch_ur_r:=false"
        
        command = f"ssh -t -t {robot} 'source ~/.bashrc; export ROS_MASTER_URI=http://roscore:11311/; source /opt/ros/noetic/setup.bash; source ~/{workspace}/devel/setup.bash; roslaunch mur_launch_hardware {robot}.launch"+launch_suffix+"; exec bash'"

        # Open a new terminal with SSH session + driver launch + keep open
        subprocess.Popen([
            "terminator",
            f"--title=Driver {robot}",      # Set the window title to "Mur Driver" :contentReference[oaicite:0]{index=0}
            "-x",                       # Execute the following command inside the terminal :contentReference[oaicite:1]{index=1}
            f"{command}; exec bash"
        ])

def quit_drivers():
    """Terminates all running driver sessions and closes terminals."""
    print("Stopping all driver sessions...")
    try:
        subprocess.Popen("pkill -f 'ssh -t -t'", shell=True)
        subprocess.Popen("pkill -f 'gnome-terminal'", shell=True)
    except Exception as e:
        print(f"Error stopping processes: {e}")

def move_to_home_pose(gui, UR_prefix):
    """Moves the selected robots to the initial pose with the correct namespace and move_group_name."""
    selected_robots = gui.get_selected_robots()

    # Set move_group_name based on UR_prefix
    move_group_name = "UR_arm_l" if UR_prefix == "UR10_l" else "UR_arm_r"

    for robot in selected_robots:
        # Special case for mur620c with UR10_r
        if robot == "mur620c" and UR_prefix == "UR10_r":
            home_position = "Home_custom"
        elif robot in ["mur620a", "mur620b"]:
            home_position = "Home_custom"
        else:  # Default case for mur620c, mur620d
            home_position = "Home_custom"

        # ROS launch command with namespace
        command = f"ROS_NAMESPACE={robot} roslaunch ur_utilities move_UR_to_home_pose.launch tf_prefix:={robot} UR_prefix:={UR_prefix} home_position:={home_position} move_group_name:={move_group_name}"
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

def parse_mir_path(gui):
    command = "roslaunch parse_mir_path parse_mir_path.launch"
    subprocess.Popen(command, shell=True)

def parse_ur_path(gui):
    command = "roslaunch parse_ur_path parse_ur_path.launch"
    subprocess.Popen(command, shell=True)

def move_mir_to_start_pose(gui):
    """Moves the MIR robot to the start pose."""
    selected_robots = gui.get_selected_robots()
    if selected_robots is None:
        print("MIR robot not selected. Skipping move to start pose.")
        return

    # Ensure only one MIR robot is selected
    if len(selected_robots) != 1:
        print("Please select only the MIR robot to move to the start pose.")
        return
    command = f"roslaunch move_mir_to_start_pose move_mir_to_start_pose.launch robot_name:={selected_robots[0]}"
    print(f"Executing: {command}")
    subprocess.Popen(command, shell=True)

def move_ur_to_start_pose(gui):
    """Moves the UR robot to the start pose."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping move to start pose.")
        return

    # Ensure only one UR and one mir robot is selected
    if len(selected_robots) != 1 or len(selected_urs) != 1:
        print("Please select exactly one UR and one MIR robot to move to the start pose.")
        return

    # ToDo: to risky to use for now
    if selected_urs[0] == "UR10_l":
        move_group_name = "UR_arm_l"
        planning_group = "UR10_l/tool0"
    else:
        move_group_name = "UR_arm_r"
        planning_group = "UR10_r/tool0"


    for robot in selected_robots:
        for ur in selected_urs:
            command = f"roslaunch move_ur_to_start_pose move_ur_to_start_pose.launch robot_name:={robot}"
            print(f"Executing: {command}")
            subprocess.Popen(command, shell=True)

def mir_follow_trajectory(gui):
    """Moves the MIR robot along a predefined trajectory."""
    selected_robots = gui.get_selected_robots()
    if not selected_robots:
        print("No MIR robot selected. Skipping follow trajectory.")
        return
    # Ensure only one MIR robot is selected
    if len(selected_robots) != 1:
        print("Please select only the MIR robot to follow the trajectory.")
        return
    command = f"roslaunch mir_trajectory_follower mir_trajectory_follower.launch robot_name:={selected_robots[0]}"
    print(f"Executing: {command}")
    subprocess.Popen(command, shell=True)

def increment_path_index(gui):
    """Increments the path index for the MIR robot."""
    command = f"roslaunch print_gui increment_path_index.launch"
    print(f"Executing: {command}")
    subprocess.Popen(command, shell=True)


def stop_mir_motion(self):
    """Stops any running Lissajous motion by killing the process."""
    command = "pkill -f mir_trajectory_follower"
    print(f"Stopping MiR motion with command: {command}")
    subprocess.Popen(command, shell=True)

    command = "pkill -f increment_path_index"
    print(f"Stopping path index increment with command: {command}")
    subprocess.Popen(command, shell=True)

def ur_follow_trajectory(gui, ur_follow_settings: dict):
    """Moves the UR robot along a predefined trajectory."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()
    metric = ur_follow_settings.get("idx_metric")
    threshold = ur_follow_settings.get("threshold")
    rospy.loginfo(f"Selected metric: {metric}")

    if not selected_robots or not selected_urs:
        print("No UR robot selected. Skipping follow trajectory.")
        return

    # Ensure only one UR and one MIR robot is selected
    if len(selected_robots) != 1 or len(selected_urs) != 1:
        print("Please select exactly one UR and one MIR robot to follow the trajectory.")
        return

    for robot in selected_robots:
        for ur in selected_urs:
            command = f"roslaunch print_hw complete_ur_trajectory_follower_ff_only.launch robot_name:={robot} prefix_ur:={ur}/ metric:='{metric}' threshold:={threshold}"
            print(f"Executing: {command}")
            subprocess.Popen(command, shell=True)

def stop_idx_advancer(gui):
    """Stops any running UR motion by killing the process."""
    command = "pkill -f path_index_advancer"
    print(f"Stopping path_index_advancer wih command: {command}")
    subprocess.Popen(command, shell=True)

    command = "pkill -f increment_path_index"
    print(f"Stopping increment_path_index with command: {command}")
    subprocess.Popen(command, shell=True)
