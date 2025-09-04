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
        super().__init__()
        # state
        self.ur_follow_settings = {'idx_metric': 'virtual line', 'threshold': 0.010}
        self.servo_calib = {'left': {'min': 0, 'zero': 0, 'max': 4095}, 'right': {'min': 0, 'zero': 0, 'max': 4095}}
        # ROS + window
        self.path_idx.connect(self._update_spinbox)
        self.ros_interface = ROSInterface(self)
        self.setWindowTitle("Additive Manufacturing GUI")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), '../img/Logo.png')))
        self.setGeometry(100, 100, 1000, 600)
        main_layout = QHBoxLayout()
        # Left column
        left_layout = QVBoxLayout()
        self.status_label = QLabel("Controller Status: Not Checked")
        self.status_label.setStyleSheet("border: 1px solid black; padding: 5px;")
        left_layout.addWidget(self.status_label)
        # Battery
        self.battery_group = QGroupBox("Battery Status"); self.battery_layout = QVBoxLayout(); self.battery_group.setLayout(self.battery_layout); left_layout.addWidget(self.battery_group); self.battery_labels = {}
        # Selection
        selection_group = QGroupBox("Robot and UR Selection"); selection_layout = QHBoxLayout(); robot_layout = QVBoxLayout()
        self.robots = {n: QCheckBox(n) for n in ["mur620a", "mur620b", "mur620c", "mur620d"]}
        for cb in self.robots.values(): robot_layout.addWidget(cb)
        for name, cb in self.robots.items(): cb.stateChanged.connect(lambda _, r=name: self.ros_interface.check_and_subscribe_battery())
        for robot in self.robots.keys():
            row = QHBoxLayout(); row.addWidget(QLabel(robot)); mir_label = QLabel("MiR: –"); ur_label = QLabel("UR: –"); row.addWidget(mir_label); row.addWidget(ur_label); self.battery_labels[robot] = (mir_label, ur_label); self.battery_layout.addLayout(row)
        ur_layout = QVBoxLayout(); ur_layout.addWidget(QLabel("Select URs:")); self.ur10_l = QCheckBox("UR10_l"); self.ur10_r = QCheckBox("UR10_r"); self.ur10_l.setChecked(False); self.ur10_r.setChecked(True); ur_layout.addWidget(self.ur10_l); ur_layout.addWidget(self.ur10_r)
        selection_layout.addLayout(robot_layout); selection_layout.addLayout(ur_layout); selection_group.setLayout(selection_layout); left_layout.addWidget(selection_group)
        # Override
        override_layout = QVBoxLayout(); override_label = QLabel("Override (%)"); self.override_slider = QSlider(); self.override_slider.setRange(0,100); self.override_slider.setValue(100); self.override_slider.setTickInterval(10); self.override_slider.setTickPosition(QSlider.TicksBelow); self.override_value_label = QLabel("100%"); self.ros_interface.init_override_velocity_slider(); override_layout.addWidget(override_label); override_layout.addWidget(self.override_slider); override_layout.addWidget(self.override_value_label); left_layout.addLayout(override_layout)
        # Setup
        setup_group = QGroupBox("Setup Functions"); setup_layout = QVBoxLayout();
        setup_buttons = {
            "Check Status": lambda: start_status_update(self),
            "Launch Drivers": lambda: launch_drivers(self),
            "Launch Keyence Scanner": lambda: self.ros_interface.launch_keyence_scanner(),
            "Start Dynamixel Driver": lambda: self.ros_interface.start_dynamixel_driver(),
            "Stop Dynamixel Driver": lambda: self.ros_interface.stop_dynamixel_driver(),
            "Open RVIZ": open_rviz,
            "Start Roscore": lambda: self.ros_interface.start_roscore(),
            "Start Mocap": lambda: self.ros_interface.start_mocap(),
            "Start Sync": lambda: self.ros_interface.start_sync(),
        }
        for text, fn in setup_buttons.items():
            b = QPushButton(text);
            if text == "Start Roscore": self.btn_roscore = b
            elif text == "Start Mocap": self.btn_mocap = b
            elif text == "Start Sync": self.btn_sync = b
            b.clicked.connect(lambda _, f=fn: f()); b.setStyleSheet("background-color: lightgray;"); setup_layout.addWidget(b)
        self.workspace_input = QLineEdit(); default_path = self.get_relative_workspace_path(); self.workspace_input.setText(default_path); self.workspace_input.setPlaceholderText("Enter workspace name"); setup_layout.addWidget(QLabel("Workspace Name:")); setup_layout.addWidget(self.workspace_input); setup_group.setLayout(setup_layout); left_layout.addWidget(setup_group)
        main_layout.addLayout(left_layout)
        # Right column
        right_layout = QVBoxLayout(); controller_group = QGroupBox("Controller Functions"); controller_layout = QVBoxLayout(); controller_buttons = {
            "Enable all URs": lambda: enable_all_urs(self),
            "Turn on Arm Controllers": lambda: turn_on_arm_controllers(self),
            "Turn on Twist Controllers": lambda: turn_on_twist_controllers(self),
            "Move to Home Pose Left": lambda: move_to_home_pose(self, "UR10_l"),
            "Move to Home Pose Right": lambda: move_to_home_pose(self, "UR10_r"),
        }
        for text, fn in controller_buttons.items(): btn = QPushButton(text); btn.clicked.connect(lambda _, f=fn: f()); controller_layout.addWidget(btn)
        controller_group.setLayout(controller_layout); right_layout.addWidget(controller_group)
        print_functions_group = QGroupBox("Print Functions"); print_functions_layout = QVBoxLayout(); print_function_buttons = {
            "Parse MiR Path": lambda: parse_mir_path(self),
            "Parse UR Path": lambda: parse_ur_path(self),
            "Move MiR to Start Pose": lambda: move_mir_to_start_pose(self),
            "Move UR to Start Pose": lambda: move_ur_to_start_pose(self),
            "MiR follow Trajectory": lambda: mir_follow_trajectory(self),
            "Increment Path Index": lambda: increment_path_index(self),
            "Stop MiR Motion": lambda: stop_mir_motion(self),
            "Stop UR Motion": lambda: stop_ur_motion(self),
        }
        for text, fn in print_function_buttons.items(): btn = QPushButton(text); btn.clicked.connect(lambda _, f=fn: f()); print_functions_layout.addWidget(btn)
        ur_btn = QPushButton("UR Follow Trajectory"); ur_btn.clicked.connect(lambda _, f=ur_follow_trajectory: f(self, self.ur_follow_settings)); ur_settings_btn = QPushButton("Settings"); ur_settings_btn.clicked.connect(self.open_ur_settings); ur_settings_btn.setStyleSheet("background-color: lightgray;"); hbox = QHBoxLayout(); hbox.addWidget(ur_btn); hbox.addWidget(ur_settings_btn); print_functions_layout.addLayout(hbox)
        idx_box = QHBoxLayout(); idx_box.addWidget(QLabel("Index:")); self.idx_spin = QSpinBox(); self.idx_spin.setRange(0,10000); self.idx_spin.setValue(0); idx_box.addWidget(self.idx_spin); stop_idx_btn = QPushButton("Stop Index Advancer"); stop_idx_btn.clicked.connect(lambda: stop_idx_advancer(self)); idx_box.addWidget(stop_idx_btn); print_functions_layout.addLayout(idx_box)
        # Servo section
        servo_box = QGroupBox("Dynamixel Servo Targets"); servo_outer_layout = QVBoxLayout(); targets_row = QHBoxLayout();
        left_col = QVBoxLayout(); left_col.addWidget(QLabel("Left target (%)")); self.servo_left_slider = QSlider(); self.servo_left_slider.setRange(0,100); self.servo_left_slider.setTickInterval(10); self.servo_left_slider.setTickPosition(QSlider.TicksBelow); self.servo_left_spin = QSpinBox(); self.servo_left_spin.setRange(-100,200); self.servo_left_spin.setValue(0); self.servo_left_slider.valueChanged.connect(self.servo_left_spin.setValue); self.servo_left_spin.valueChanged.connect(lambda v: 0 <= v <= 100 and self.servo_left_slider.setValue(v)); left_col.addWidget(self.servo_left_slider); left_col.addWidget(self.servo_left_spin)
        right_col = QVBoxLayout(); right_col.addWidget(QLabel("Right target (%)")); self.servo_right_slider = QSlider(); self.servo_right_slider.setRange(0,100); self.servo_right_slider.setTickInterval(10); self.servo_right_slider.setTickPosition(QSlider.TicksBelow); self.servo_right_spin = QSpinBox(); self.servo_right_spin.setRange(-100,200); self.servo_right_spin.setValue(0); self.servo_right_slider.valueChanged.connect(self.servo_right_spin.setValue); self.servo_right_spin.valueChanged.connect(lambda v: 0 <= v <= 100 and self.servo_right_slider.setValue(v)); right_col.addWidget(self.servo_right_slider); right_col.addWidget(self.servo_right_spin)
        send_col = QVBoxLayout(); send_btn = QPushButton("Send Targets"); send_btn.clicked.connect(self._send_percent_targets); send_col.addWidget(QLabel(" ")); send_col.addWidget(send_btn)
        targets_row.addLayout(left_col); targets_row.addLayout(right_col); targets_row.addLayout(send_col); servo_outer_layout.addLayout(targets_row)
        zero_row = QHBoxLayout(); zl = QVBoxLayout(); zl.addWidget(QLabel("Left zero (raw)")); self.servo_left_zero_spin = QSpinBox(); self.servo_left_zero_spin.setRange(0,4095); self.servo_left_zero_spin.setValue(self.servo_calib['left']['zero']); zl.addWidget(self.servo_left_zero_spin); zr = QVBoxLayout(); zr.addWidget(QLabel("Right zero (raw)")); self.servo_right_zero_spin = QSpinBox(); self.servo_right_zero_spin.setRange(0,4095); self.servo_right_zero_spin.setValue(self.servo_calib['right']['zero']); zr.addWidget(self.servo_right_zero_spin); zb = QVBoxLayout(); send_zero_btn = QPushButton("Send Zero Position"); send_zero_btn.clicked.connect(self._send_zero_positions); calib_btn = QPushButton("Servo Calibration..."); calib_btn.clicked.connect(self.open_servo_calibration); zb.addWidget(QLabel(" ")); zb.addWidget(send_zero_btn); zb.addWidget(calib_btn); zero_row.addLayout(zl); zero_row.addLayout(zr); zero_row.addLayout(zb); servo_outer_layout.addLayout(zero_row); servo_box.setLayout(servo_outer_layout); print_functions_layout.addWidget(servo_box)
        print_functions_group.setLayout(print_functions_layout); right_layout.addWidget(print_functions_group); main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
        # Timer
        self.status_timer = QTimer(); self.status_timer.timeout.connect(self.ros_interface.update_button_status); self.status_timer.start(5000)

    def _percent_to_raw(self, percent: float, which: str) -> int:
        c = self.servo_calib[which]
        span = c['max'] - c['min']
        if span == 0:
            return int(c['min'])
        raw = c['min'] + (percent / 100.0) * span
        return max(0, min(4095, int(round(raw))))

    def _send_percent_targets(self):
        left_p = self.servo_left_spin.value()
        right_p = self.servo_right_spin.value()
        left_raw = self._percent_to_raw(left_p, 'left')
        right_raw = self._percent_to_raw(right_p, 'right')
        self.ros_interface.publish_servo_targets(left_raw, right_raw)

    def _send_zero_positions(self):
        # Use currently stored zero raw values (spin boxes display them)
        self.ros_interface.publish_servo_targets(
            int(self.servo_left_zero_spin.value()),
            int(self.servo_right_zero_spin.value())
        )

    def open_servo_calibration(self):
        dlg = ServoCalibrationDialog(self, self.servo_calib)
        if dlg.exec_() == QDialog.Accepted:
            self.servo_calib = dlg.get_values()
            # update zero spin boxes
            self.servo_left_zero_spin.setValue(self.servo_calib['left']['zero'])
            self.servo_right_zero_spin.setValue(self.servo_calib['right']['zero'])
            # Optionally reset sliders/spins to zero percent
            # self.servo_left_slider.setValue(0); self.servo_left_spin.setValue(0)
            # self.servo_right_slider.setValue(0); self.servo_right_spin.setValue(0)
    

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


class ServoCalibrationDialog(QDialog):
    def __init__(self, parent, calib):
        super().__init__(parent)
        self.setWindowTitle("Servo Calibration")
        self._orig = calib
        self._data = {
            'left': calib['left'].copy(),
            'right': calib['right'].copy()
        }
        form = QFormLayout()
        # Left
        self.left_min = QSpinBox(); self.left_min.setRange(0,4095); self.left_min.setValue(self._data['left']['min'])
        self.left_zero = QSpinBox(); self.left_zero.setRange(0,4095); self.left_zero.setValue(self._data['left']['zero'])
        self.left_max = QSpinBox(); self.left_max.setRange(0,4095); self.left_max.setValue(self._data['left']['max'])
        form.addRow(QLabel("Left Min"), self.left_min)
        form.addRow(QLabel("Left Zero"), self.left_zero)
        form.addRow(QLabel("Left Max"), self.left_max)
        # Right
        self.right_min = QSpinBox(); self.right_min.setRange(0,4095); self.right_min.setValue(self._data['right']['min'])
        self.right_zero = QSpinBox(); self.right_zero.setRange(0,4095); self.right_zero.setValue(self._data['right']['zero'])
        self.right_max = QSpinBox(); self.right_max.setRange(0,4095); self.right_max.setValue(self._data['right']['max'])
        form.addRow(QLabel("Right Min"), self.right_min)
        form.addRow(QLabel("Right Zero"), self.right_zero)
        form.addRow(QLabel("Right Max"), self.right_max)
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def _on_accept(self):
        # Basic validation: min <= zero <= max
        if not (self.left_min.value() <= self.left_zero.value() <= self.left_max.value()):
            # silently clamp
            z = min(max(self.left_zero.value(), self.left_min.value()), self.left_max.value())
            self.left_zero.setValue(z)
        if not (self.right_min.value() <= self.right_zero.value() <= self.right_max.value()):
            z = min(max(self.right_zero.value(), self.right_min.value()), self.right_max.value())
            self.right_zero.setValue(z)
        self.accept()

    def get_values(self):
        return {
            'left': {
                'min': self.left_min.value(),
                'zero': self.left_zero.value(),
                'max': self.left_max.value(),
            },
            'right': {
                'min': self.right_min.value(),
                'zero': self.right_zero.value(),
                'max': self.right_max.value(),
            }
        }

