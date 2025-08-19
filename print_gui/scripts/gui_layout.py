from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLineEdit, QHBoxLayout, QPushButton, QLabel, QTableWidget, QCheckBox, QTableWidgetItem, QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox, QTextEdit, QComboBox, QDoubleSpinBox, QDialogButtonBox, QFormLayout, QDialog
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon
from ros_interface import start_status_update, ur_follow_trajectory, open_rviz, launch_drivers, quit_drivers, turn_on_arm_controllers, turn_on_twist_controllers, stop_mir_motion, stop_idx_advancer, stop_ur_motion
from ros_interface import enable_all_urs, move_to_home_pose, parse_mir_path, parse_ur_path, move_mir_to_start_pose, move_ur_to_start_pose, mir_follow_trajectory, increment_path_index
from ros_interface import ROSInterface
import os


class ROSGui(QWidget):
    path_idx = pyqtSignal(int)
    
    def __init__(self):
        self.ur_follow_settings = {
            'idx_metric': 'virtual line',
            'threshold': 0.010,
        }

        super().__init__()
        self.path_idx.connect(self._update_spinbox)
        
        self.ros_interface = ROSInterface(self)
        self.setWindowTitle("Additive Manufacturing GUI")
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
        self.override_slider.setValue(100)
        self.override_slider.setTickInterval(10)
        self.override_slider.setTickPosition(QSlider.TicksBelow)

        self.override_value_label = QLabel("100%")
        self.override_value_label.setAlignment(Qt.AlignCenter)

        self.ros_interface.init_override_velocity_slider()

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
            "Launch Keyence Scanner": lambda: self.ros_interface.launch_keyence_scanner(),
            "Start Dynamixel Driver": lambda: self.ros_interface.start_dynamixel_driver(),
            "Stop Dynamixel Driver": lambda: self.ros_interface.stop_dynamixel_driver(),
            #"Quit Drivers": lambda: quit_drivers(),
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
            "Stop UR Motion": lambda: stop_ur_motion(self)
        }

        for text, function in print_function_buttons.items():
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, f=function: f())
            print_functions_layout.addWidget(btn)

        # Print Functions UR:
        ur_btn = QPushButton("UR Follow Trajectory")
        ur_btn.clicked.connect(lambda _, f=ur_follow_trajectory: f(self, self.ur_follow_settings))

        # 1) Add a settings button next to the UR Follow Trajectory button
        ur_settings_btn = QPushButton("Settings")
        ur_settings_btn.clicked.connect(self.open_ur_settings)
        ur_settings_btn.setStyleSheet("background-color: lightgray;")
         # horizontal layout for side-by-side placement
        hbox = QHBoxLayout()
        hbox.addWidget(ur_btn)
        hbox.addWidget(ur_settings_btn)
        print_functions_layout.addLayout(hbox)

        # ── current‐index display + stop button ──
        idx_box = QHBoxLayout()
        idx_box.addWidget(QLabel("Index:"))
        # self.idx_label = QLabel("000")
        # idx_box.addWidget(self.idx_label)
        self.idx_spin = QSpinBox()
        self.idx_spin.setRange(0, 10000)
        self.idx_spin.setValue(0)
        # self.idx_spin.valueChanged.connect(
        idx_box.addWidget(self.idx_spin)
        
        stop_idx_btn = QPushButton("Stop Index Advancer")
        stop_idx_btn.clicked.connect(lambda: stop_idx_advancer(self))
        idx_box.addWidget(stop_idx_btn)
        print_functions_layout.addLayout(idx_box)

        # Dynamixel Servo Target Controls (added under Print Functions)
        servo_box = QGroupBox("Dynamixel Servo Targets")
        servo_layout = QHBoxLayout()
        # Left servo
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Left target"))
        self.servo_left_spin = QDoubleSpinBox()
        self.servo_left_spin.setRange(-1000.0, 1000.0)
        self.servo_left_spin.setSingleStep(1.0)
        self.servo_left_spin.setDecimals(2)
        self.servo_left_spin.setValue(0.0)
        left_col.addWidget(self.servo_left_spin)
        # Right servo
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Right target"))
        self.servo_right_spin = QDoubleSpinBox()
        self.servo_right_spin.setRange(-1000.0, 1000.0)
        self.servo_right_spin.setSingleStep(1.0)
        self.servo_right_spin.setDecimals(2)
        self.servo_right_spin.setValue(0.0)
        right_col.addWidget(self.servo_right_spin)
        # Send button
        send_col = QVBoxLayout()
        send_btn = QPushButton("Send Targets")
        send_btn.clicked.connect(lambda: self.ros_interface.publish_servo_targets(self.servo_left_spin.value(), self.servo_right_spin.value()))
        send_col.addWidget(QLabel(" "))
        send_col.addWidget(send_btn)
            
        servo_layout.addLayout(left_col)
        servo_layout.addLayout(right_col)
        servo_layout.addLayout(send_col)
        servo_box.setLayout(servo_layout)
        print_functions_layout.addWidget(servo_box)

        print_functions_group.setLayout(print_functions_layout)
            
        right_layout.addWidget(controller_group)
        right_layout.addWidget(print_functions_group)

        main_layout.addLayout(right_layout)  # Fügt das Layout auf der rechten Seite hinzu

        self.setLayout(main_layout)
    

    @pyqtSlot(int)
    def _update_spinbox(self, idx):
        # guaranteed to run in Qt (GUI) thread
        self.idx_spin.setValue(idx)
        
    def open_ur_settings(self):
        dlg = URFollowSettingsDialog(self, initial_settings=self.ur_follow_settings)
        if dlg.exec_() == QDialog.Accepted:
            # Retrieve and store new settings
            self.ur_follow_settings = dlg.getValues()
            print("New UR Follow settings:", self.ur_follow_settings)

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

    def get_workspace_name(self):
        return self.workspace_input.text().strip()

    def get_override_value(self):
        return self.override_slider.value()

class URFollowSettingsDialog(QDialog):
    def __init__(self, parent=None, initial_settings=None):
        super().__init__(parent)
        self.setWindowTitle("UR Follow Trajectory Settings")

        # Default settings
        init = initial_settings or {}
        idx_metrics = ["virtual line", "radius", "collinear"]
        idx_metric = init.get('idx_metric', idx_metrics[0])
        threshold = init.get('threshold', 0.010)

        form = QFormLayout()        
        self.dropdown_idx_metric = QComboBox()
        self.dropdown_idx_metric.addItems(idx_metrics)
        self.dropdown_idx_metric.setCurrentText(idx_metric)  # Set default text
        self.dropdown_idx_metric.setStyleSheet("background-color: lightgray;")
        # self.dropdown_idx_metric.currentTextChanged.connect(lambda text: self.set_idx_metric(text))
        form.addRow("Index Metric:", self.dropdown_idx_metric)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 0.2)  # Set range for the spin box
        self.spin_threshold.setSingleStep(0.002)
        self.spin_threshold.setDecimals(3)  # Set number of decimal places
        self.spin_threshold.setValue(threshold)
        self.spin_threshold.setSuffix(" m")
        self.spin_threshold.setStyleSheet("background-color: lightgray;")
        form.addRow("Threshold:", self.spin_threshold)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def getValues(self):
        return {
            'idx_metric': self.dropdown_idx_metric.currentText(),
            'threshold': self.spin_threshold.value(),
        }

