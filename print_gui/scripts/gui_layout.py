import threading
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLineEdit, QHBoxLayout, QPushButton, QLabel, QTableWidget, QCheckBox, QTableWidgetItem, QGroupBox, QTabWidget, QDoubleSpinBox, QTextEdit, QComboBox, QDoubleSpinBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
from ros_interface import start_status_update, ur_follow_trajectory, open_rviz, launch_drivers, quit_drivers, turn_on_arm_controllers, turn_on_twist_controllers, stop_mir_motion
from ros_interface import enable_all_urs, move_to_home_pose, parse_mir_path, parse_ur_path, move_mir_to_start_pose, move_ur_to_start_pose, mir_follow_trajectory, increment_path_index
from ros_interface import ROSInterface
import os


class ROSGui(QWidget):
    def __init__(self):
        super().__init__()
        self.ros_interface = ROSInterface(self)
        self.setWindowTitle("Multi-Robot Demo")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), '../img/Logo.png')))
        self.setGeometry(100, 100, 1000, 600)  # Increased width
        
        main_layout = QHBoxLayout()
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.ros_interface.update_button_status)
        self.status_timer.start(5000)  # Check status every 5 seconds
        
        # Left Side (Status & Buttons)
        left_layout = QVBoxLayout()
        self.status_label = QLabel("Controller Status: Not Checked")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("border: 1px solid black; padding: 5px;")
        left_layout.addWidget(self.status_label)

        # Battery Status
        self.battery_group = QGroupBox("Battery Status")
        self.battery_layout = QVBoxLayout()
        self.battery_group.setLayout(self.battery_layout)
        left_layout.addWidget(self.battery_group)

        self.battery_labels = {}  # z.B. {"mur620a": (mir_label, ur_label)}

        # Robot and UR selection
        selection_group = QGroupBox("Robot and UR Selection")
        selection_layout = QHBoxLayout()

        robot_layout = QVBoxLayout()
        self.robots = {
            "mur620a": QCheckBox("mur620a"),
            "mur620b": QCheckBox("mur620b"),
            "mur620c": QCheckBox("mur620c"),
            "mur620d": QCheckBox("mur620d"),
        }
        for checkbox in self.robots.values():
            robot_layout.addWidget(checkbox)

        for robot_name, checkbox in self.robots.items():
            checkbox.stateChanged.connect(lambda _, r=robot_name: self.ros_interface.check_and_subscribe_battery())


        for robot in self.robots.keys():
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{robot}"))
            
            mir_label = QLabel("MiR: –")
            ur_label = QLabel("UR: –")
            row.addWidget(mir_label)
            row.addWidget(ur_label)

            self.battery_labels[robot] = (mir_label, ur_label)
            self.battery_layout.addLayout(row)


        ur_layout = QVBoxLayout()
        ur_layout.addWidget(QLabel("Select URs:"))
        self.ur10_l = QCheckBox("UR10_l")
        self.ur10_r = QCheckBox("UR10_r")
        self.ur10_l.setChecked(False)
        self.ur10_r.setChecked(True)
        ur_layout.addWidget(self.ur10_l)
        ur_layout.addWidget(self.ur10_r)

        selection_layout.addLayout(robot_layout)
        selection_layout.addLayout(ur_layout)
        selection_group.setLayout(selection_layout)
        left_layout.addWidget(selection_group)
        

                # Override Slider
        override_layout = QVBoxLayout()
        override_label = QLabel("Override (%)")
        override_label.setAlignment(Qt.AlignCenter)

        self.override_slider = QSlider(Qt.Horizontal)
        self.override_slider.setMinimum(0)
        self.override_slider.setMaximum(100)
        self.override_slider.setValue(50)
        self.override_slider.setTickInterval(10)
        self.override_slider.setTickPosition(QSlider.TicksBelow)

        self.override_value_label = QLabel("50%")
        self.override_value_label.setAlignment(Qt.AlignCenter)

        self.override_slider.valueChanged.connect(
            lambda value: self.override_value_label.setText(f"{value}%")
        )

        override_layout.addWidget(override_label)
        override_layout.addWidget(self.override_slider)
        override_layout.addWidget(self.override_value_label)

        left_layout.addLayout(override_layout)


       
        # Setup Functions Group
        setup_group = QGroupBox("Setup Functions")
        setup_layout = QVBoxLayout()
        setup_buttons = {
            "Check Status": lambda: start_status_update(self),
            "Launch Drivers": lambda: launch_drivers(self),
            "Quit Drivers": lambda: quit_drivers(),
            "Open RVIZ": open_rviz,
            "Start Roscore": lambda: self.ros_interface.start_roscore(),
            "Start Mocap": lambda: self.ros_interface.start_mocap(),
            "Start Sync": lambda: self.ros_interface.start_sync(),
        }

        for text, function in setup_buttons.items():
            btn = QPushButton(text)
            
            # Speichert spezielle Buttons für Status-Updates
            if text == "Start Roscore":
                self.btn_roscore = btn
            elif text == "Start Mocap":
                self.btn_mocap = btn
            elif text == "Start Sync":
                self.btn_sync = btn

            btn.clicked.connect(lambda checked, f=function: f())
            btn.setStyleSheet("background-color: lightgray;")  # Standardfarbe
            setup_layout.addWidget(btn)

        self.workspace_input = QLineEdit()
        default_path = self.get_relative_workspace_path()
        self.workspace_input.setText(default_path)
        self.workspace_input.setPlaceholderText("Enter workspace name")
        setup_layout.addWidget(QLabel("Workspace Name:"))
        setup_layout.addWidget(self.workspace_input)


        setup_group.setLayout(setup_layout)
        left_layout.addWidget(setup_group)
        
               
        main_layout.addLayout(left_layout)
        
    
        
        right_layout = QVBoxLayout()


        # Buttons für "Save Poses" und "Update Poses"
        pose_button_layout = QVBoxLayout()
        right_layout.addLayout(pose_button_layout)

        # Erstelle die "Controller Functions" Gruppe und füge sie rechts hinzu
        controller_group = QGroupBox("Controller Functions")
        controller_layout = QVBoxLayout()
        controller_buttons = {
            "Enable all URs": lambda: enable_all_urs(self),
            "Turn on Arm Controllers": lambda: turn_on_arm_controllers(self),
            "Turn on Twist Controllers": lambda: turn_on_twist_controllers(self),
            "Move to Home Pose Left": lambda: move_to_home_pose(self, "UR10_l"),
            "Move to Home Pose Right": lambda: move_to_home_pose(self, "UR10_r"),
        }

        for text, function in controller_buttons.items():
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, f=function: f())
            controller_layout.addWidget(btn)

        controller_group.setLayout(controller_layout)
        
        # Print Functions GroupBox
        print_functions_group = QGroupBox("Print Functions")
        print_functions_layout = QVBoxLayout()
        print_function_buttons = {
            "Parse MiR Path": lambda: parse_mir_path(self),
            "Parse UR Path": lambda: parse_ur_path(self),
            "Move MiR to Start Pose": lambda: move_mir_to_start_pose(self),
            "Move UR to Start Pose": lambda: move_ur_to_start_pose(self),
            "MiR follow Trajectory": lambda: mir_follow_trajectory(self),
            "Increment Path Index": lambda: increment_path_index(self),
            "Stop MiR Motion": lambda: stop_mir_motion(self),
            "UR Follow Trajectory": lambda: ur_follow_trajectory(self),
        }

        for text, function in print_function_buttons.items():
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, f=function: f())
            print_functions_layout.addWidget(btn)
        
        self.idx_metric = "virtual line"
        dropdown_idx_metric = QComboBox()
        dropdown_idx_metric.addItems(["virtual line", "radius", "collinear"])
        dropdown_idx_metric.setCurrentIndex(0)  # Set default index to 0
        dropdown_idx_metric.setStyleSheet("background-color: lightgray;")
        dropdown_idx_metric.currentTextChanged.connect(lambda text: self.set_idx_metric(text))
        print_functions_layout.addWidget(dropdown_idx_metric)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 0.2)  # Set range for the spin box
        self.spin_threshold.setSingleStep(0.002)
        self.spin_threshold.setValue(0.010)
        self.spin_threshold.setDecimals(3)  # Set number of decimal places
        self.spin_threshold.setSuffix(" m")
        self.spin_threshold.setStyleSheet("background-color: lightgray;")
        print_functions_layout.addWidget(QLabel("Threshold:"))
        print_functions_layout.addWidget(self.spin_threshold)

        print_functions_group.setLayout(print_functions_layout)
        

        right_layout.addWidget(controller_group)
        right_layout.addWidget(print_functions_group)

        main_layout.addLayout(right_layout)  # Fügt das Layout auf der rechten Seite hinzu

        self.setLayout(main_layout)

    def set_idx_metric(self, text):
            self.idx_metric = text

    def get_full_workspace_path(self):
        import os
        current_path = os.path.abspath(__file__)

        while current_path != "/":
            if os.path.basename(current_path) == "src":
                return os.path.dirname(current_path)  # Absoluter Pfad zum Workspace
            current_path = os.path.dirname(current_path)

        return os.path.expanduser("~/catkin_ws")  # Fallback


    def get_relative_workspace_path(self):
        full_path = self.get_full_workspace_path()
        home_path = os.path.expanduser("~")
        if full_path.startswith(home_path):
            return os.path.relpath(full_path, home_path)
        return full_path

    def update_virtual_object_pose(self, pose):
        """Updates the GUI table with the latest virtual object pose."""
        for col in range(6):
            self.table.setItem(8, col, QTableWidgetItem(str(round(pose[col], 4))))
    
    def get_selected_robots(self):
        return [name for name, checkbox in self.robots.items() if checkbox.isChecked()]

    def get_selected_urs(self):
        ur_prefixes = []
        if self.ur10_l.isChecked():
            ur_prefixes.append("UR10_l")
        if self.ur10_r.isChecked():
            ur_prefixes.append("UR10_r")
        return ur_prefixes

    def save_relative_poses(self, updated_poses=None):
        """Collects values from the table and saves them. If updated_poses is provided, those values are used first."""
        poses = {}

        # Convert `updated_poses` keys to match the table format ("mur620c/UR10_l")
        if updated_poses:
            formatted_updated_poses = {f"{robot}/{ur}": pos for (robot, ur), pos in updated_poses.items()}
        else:
            formatted_updated_poses = {}

        for row in range(self.table.rowCount()):
            row_label = self.table.verticalHeaderItem(row).text()

            # Use updated pose values if available; otherwise, keep existing table values
            if row_label in formatted_updated_poses:
                poses[row_label] = formatted_updated_poses[row_label]
            else:
                poses[row_label] = [
                    float(self.table.item(row, col).text()) if self.table.item(row, col) else 0.0
                    for col in range(6)  # Jetzt für X, Y, Z, Rx, Ry, Rz
                ]

        # Save values to poses.yaml
        relative_poses = RelativePoses()
        relative_poses.save_poses(poses)



    def load_relative_poses(self):
        """Lädt die gespeicherten Posen und setzt sie in die Tabelle ein."""
        relative_poses = RelativePoses()  # Instanz erstellen
        poses = relative_poses.load_poses()  # Geladene Posen als Dictionary

        for row in range(self.table.rowCount()):
            row_label = self.table.verticalHeaderItem(row).text()
            if row_label in poses:
                for col in range(self.table.columnCount()):
                    print(f"Setting {row_label} at {col} to {poses[row_label][col]}")
                    print("coloncount", self.table.columnCount())
                    value = poses[row_label][col] if col < len(poses[row_label]) else 0.0
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))

                    if row_label == "Virtual Object":
                        self.ros_interface.virtual_object_pose = poses[row_label]  # Ensure it's loaded properly

    def get_relative_pose(self, robot, ur):
        """Retrieves the relative pose [x, y, z] from the table for the given robot and UR arm."""
        row_label = f"{robot}/{ur}"
        
        for row in range(self.table.rowCount()):
            if self.table.verticalHeaderItem(row).text() == row_label:
                return [
                    float(self.table.item(row, col).text()) if self.table.item(row, col) else 0.0
                    for col in range(6)  # Jetzt für X, Y, Z, Rx, Ry, Rz
                ]
        
        # Default value if no match is found
        return [0.0, 0.0, 0.0]

    def get_workspace_name(self):
        return self.workspace_input.text().strip()

    def get_override_value(self):
        return self.override_slider.value()
