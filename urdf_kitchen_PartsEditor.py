"""
File Name: urdf_kitchen_PartsEditor.py
Description: A Python script for configuring connection points of parts for urdf_kitchen_Assembler.py.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Update.     : Jan 22, 2026
Version     : 0.1.0
License     : MIT License
URL         : https://github.com/Ninagawa123/URDF_kitchen_beta
Copyright (c) 2024 Ninagawa123

python3.11
pip install --upgrade pip
pip install numpy
pip install PySide6
pip install vtk
pip install NodeGraphQt
pip install trimesh
pip install pycollada
pip install networkx
"""

import sys
import signal
import math
import os
import numpy as np
import traceback
import base64
import shutil
import datetime
import xml.etree.ElementTree as ET

import vtk

# Import trimesh for COLLADA (.dae) file support (REQUIRED)
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("ERROR: trimesh is required but not installed.")
    print("Install with: pip install trimesh")
    print("Or install with: pip install trimesh pycollada")
    sys.exit(1)

from Qt import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QLabel, QGridLayout,
    QTextEdit, QButtonGroup, QRadioButton, QDialog, QMessageBox, QFrame
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextOption, QColor, QPalette
from PySide6.QtNetwork import QLocalServer, QLocalSocket

# Import URDF Kitchen utilities
from urdf_kitchen_utils import (
    OffscreenRenderer, CameraController, AnimatedCameraRotation,
    AdaptiveMarkerSize, create_crosshair_marker, MouseDragState,
    calculate_arrow_key_step, calculate_inertia_tensor,
    calculate_inertia_with_trimesh,
    get_mesh_file_filter,
    load_mesh_to_polydata, save_polydata_to_mesh,
    setup_signal_handlers, setup_signal_processing_timer, setup_dark_theme,
    VTKViewerBase, KitchenColorPicker
)

# pip install numpy
# pip install PySide6
# pip install vtk
# pip install NodeGraphQt

# ============================================================================
# CONSTANTS
# ============================================================================

# Camera settings
CAMERA_ROTATION_SENSITIVITY = 0.5
CAMERA_PAN_SPEED_FACTOR = 0.001
CAMERA_ZOOM_FACTOR = 0.1
CAMERA_ZOOM_IN_SCALE = 0.9
CAMERA_ZOOM_OUT_SCALE = 1.1
MOUSE_ZOOM_PAN_SCALE = 2.0

# Point marker settings
POINT_MARKER_SIZE = 0.1
POINT_LABEL_OFFSET_Z = 0.15

# Axis colors (RGB 0-1 range)
AXIS_COLOR_X = (1.0, 0.0, 0.0)  # Red
AXIS_COLOR_Y = (0.0, 1.0, 0.0)  # Green
AXIS_COLOR_Z = (0.0, 0.0, 1.0)  # Blue

# UI colors (hex format for labels)
UI_COLOR_X = '#FF6B6B'  # Light Red
UI_COLOR_Y = '#90EE90'  # Light Green
UI_COLOR_Z = '#00BFFF'  # Sky Blue

# Default values
DEFAULT_MASS = 1.0
DEFAULT_DENSITY = 1000.0
DEFAULT_POINT_NAME_PREFIX = "Point"

def apply_dark_theme(app):
    """Apply dark theme"""
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(70, 80, 80))
    palette.setColor(QPalette.WindowText, QColor(240, 240, 237))
    palette.setColor(QPalette.Base, QColor(240, 240, 237))
    palette.setColor(QPalette.AlternateBase, QColor(230, 230, 227))
    palette.setColor(QPalette.ToolTipBase, QColor(240, 240, 237))
    palette.setColor(QPalette.ToolTipText, QColor(51, 51, 51))
    palette.setColor(QPalette.Text, QColor(51, 51, 51))
    palette.setColor(QPalette.Button, QColor(240, 240, 237))
    palette.setColor(QPalette.ButtonText, QColor(51, 51, 51))
    palette.setColor(QPalette.Highlight, QColor(150, 150, 150))
    palette.setColor(QPalette.HighlightedText, QColor(240, 240, 237))
    app.setPalette(palette)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #404244;
        }
        QPushButton {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #5a5a5a, stop:1 #3a3a3a);
            color: #ffffff;
            border: 1px solid #707070;
            border-radius: 5px;
            padding: 2px 8px;
            margin: 3px 2px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #6a6a6a, stop:1 #4a4a4a);
        }
        QPushButton:pressed {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #3a3a3a, stop:1 #5a5a5a);
        }
        QLineEdit {
            background-color: #F0F0ED;
            border: 1px solid #BBBBB7;
            color: #333333;
            padding: 1px;
            border-radius: 2px;
            min-height: 12px;
            max-height: 12px;
        }
        QLineEdit:focus {
            border: 1px solid #999999;
            background-color: #FFFFFF;
        }
        QLabel {
            color: #F0F0ED;
        }
        QCheckBox {
            color: #F0F0ED;
            spacing: 10px;
        }
        QCheckBox::indicator {
            width: 12px;
            height: 12px;
            background-color: #F0F0ED;
            border: 1px solid #BBBBB7;
            border-radius: 2px;
        }
        QCheckBox::indicator:checked {
            background-color: #87CEEB;
            border: 1px solid #4682B4;
        }
        QCheckBox#point_checkbox::indicator:checked {
            background-color: #BA55D3;
            border: 1px solid #8B008B;
        }
        QRadioButton {
            color: #F0F0ED;
            spacing: 2px;
        }
        QRadioButton::indicator {
            width: 12px;
            height: 12px;
            background-color: #F0F0ED;
            border: 1px solid #BBBBB7;
            border-radius: 2px;
        }
        QRadioButton::indicator:checked {
            background-color: #87CEEB;
            border: 1px solid #4682B4;
        }
        QFileDialog {
            background-color: #404244;
        }
        QFileDialog QLabel {
            color: #F0F0ED;
        }
        QFileDialog QLineEdit {
            background-color: #F0F0ED;
            color: #333333;
            border: 1px solid #BBBBB7;
        }
        QFileDialog QPushButton {
            background-color: #F0F0ED;
            color: #333333;
            border: 1px solid #BBBBB7;
        }
        QFileDialog QTreeView {
            background-color: #F0F0ED;
            color: #333333;
        }
        QFileDialog QComboBox {
            background-color: #F0F0ED;
            color: #333333;
            border: 1px solid #BBBBB7;
        }
    """)


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        super(CustomInteractorStyle, self).__init__()
        self.parent = parent
        self.AddObserver("CharEvent", self.on_char_event)
        self.AddObserver("KeyPressEvent", self.on_key_press)

    def on_char_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "t":
            print("[T] Toggle wireframe.")
            self.toggle_wireframe()
        elif key == "r":
            print("[R] Reset camera.")
            if self.parent:
                self.parent.reset_camera()
        elif key == "a":
            print("[A] Rotate 90° left.")
            if self.parent:
                self.parent.rotate_camera(90, 'yaw')
        elif key == "d":
            print("[D] Rotate 90° right.")
            if self.parent:
                self.parent.rotate_camera(-90, 'yaw')
        elif key == "w":
            print("[W] Rotate 90° up.")
            if self.parent:
                self.parent.rotate_camera(-90, 'pitch')
        elif key == "s":
            print("[S] Rotate 90° down.")
            if self.parent:
                self.parent.rotate_camera(90, 'pitch')
        elif key == "q":
            print("[Q] Rotate 90° counterclockwise.")
            if self.parent:
                self.parent.rotate_camera(90, 'roll')
        elif key == "e":
            print("[E] Rotate 90° clockwise.")
            if self.parent:
                self.parent.rotate_camera(-90, 'roll')
        else:
            self.OnChar()
            
    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        shift_pressed = self.GetInteractor().GetShiftKey()
        ctrl_pressed = self.GetInteractor().GetControlKey()
        
        step = 0.01
        if shift_pressed and ctrl_pressed:
            step = 0.0001
        elif shift_pressed:
            step = 0.001
        
        if self.parent:
            horizontal_axis, vertical_axis, screen_right, screen_up = self.parent.get_screen_axes()
            for i, checkbox in enumerate(self.parent.point_checkboxes):
                if checkbox.isChecked():
                    if key == "Up":
                        self.parent.move_point_screen(i, screen_up, step)
                    elif key == "Down":
                        self.parent.move_point_screen(i, screen_up, -step)
                    elif key == "Left":
                        self.parent.move_point_screen(i, screen_right, -step)
                    elif key == "Right":
                        self.parent.move_point_screen(i, screen_right, step)
        
        self.OnKeyPress()
        
    def toggle_wireframe(self):
        # Note: This method is no longer used in offscreen rendering mode
        # Wireframe toggle is handled by MainWindow directly
        if self.parent:
            self.parent.toggle_wireframe()
    
    def on_mouse_move(self, obj, event):
        if self.parent:
            x, y = self.GetInteractor().GetEventPosition()
            for i, checkbox in enumerate(self.parent.point_checkboxes):
                if checkbox.isChecked():
                    self.parent.update_point_position(i, x, y)
        self.OnMouseMove()
    
class MainWindow(VTKViewerBase, QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("URDF Kitchen - PartsEditor v0.1.0 -")
        self.setGeometry(0, 0, 1200, 600)
        self.camera_rotation = [0, 0, 0]
        self.absolute_origin = [0, 0, 0]
        self.initial_camera_position = [10, 0, 0]
        self.initial_camera_focal_point = [0, 0, 0]
        self.initial_camera_view_up = [0, 0, 1]

        self.num_points = 8
        self.point_coords = [list(self.absolute_origin) for _ in range(self.num_points)]
        self.point_angles = [[0.0, 0.0, 0.0] for _ in range(self.num_points)]
        self.point_actors = [None] * self.num_points
        self.point_checkboxes = []
        self.point_inputs = []

        self.com_coords = [0.0, 0.0, 0.0]
        self.com_sphere_actor = None
        self.com_cursor_actor = None

        self.color_manually_changed = False
        self.mesh_color = None

        self.pending_stl_file = None

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.file_name_label = QLabel("File: No file loaded")
        main_layout.addWidget(self.file_name_label)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        content_layout.addWidget(left_widget, 1)

        # VTK display widget (offscreen rendering for Mac compatibility)
        self.vtk_display = QLabel(central_widget)
        self.vtk_display.setMinimumSize(800, 600)
        self.vtk_display.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #555;
            }
            QLabel:focus {
                border: 2px solid #00aaff;
            }
        """)
        self.vtk_display.setAlignment(Qt.AlignCenter)
        self.vtk_display.setText("3D View\n\n(Load 3D model file to display)")
        self.vtk_display.setScaledContents(False)
        self.vtk_display.setFocusPolicy(Qt.StrongFocus)
        content_layout.addWidget(self.vtk_display, 4)

        self.setup_ui()

        self.vtk_initialized = False

        # Rendering lock to prevent re-entry
        self._is_rendering = False
        self._render_counter = 0

        self.model_bounds = None
        self.stl_actor = None
        self.current_rotation = 0

        self.stl_center = list(self.absolute_origin)

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_rotation)
        self.animation_frames = 0
        self.total_animation_frames = 12
        self.rotation_per_frame = 0
        self.target_rotation = 0
        self.is_animating = False  # Flag to block input during animation
        self.target_angle = 0  # Target rotation angle for precise stopping
        self.rotation_axis = None  # Axis for current rotation
        self.pending_rotation_update = None  # Pending camera_rotation update after animation completes

        self.rotation_types = {'yaw': 0, 'pitch': 1, 'roll': 2}

        # Mouse interaction state
        self.mouse_pressed = False
        self.middle_mouse_pressed = False
        self.last_mouse_pos = None

        # Install event filter for mouse events
        self.vtk_display.installEventFilter(self)
        self.vtk_display.setMouseTracking(True)

        # Setup IPC server for receiving file load requests from Assembler
        self.ipc_server = QLocalServer(self)
        server_name = "URDFKitchen_PartsEditor"
        # Remove any previous server instance
        QLocalServer.removeServer(server_name)
        if self.ipc_server.listen(server_name):
            print(f"IPC Server started: {server_name}")
            self.ipc_server.newConnection.connect(self._handle_ipc_connection)
        else:
            print(f"Failed to start IPC server: {self.ipc_server.errorString()}")

    def _handle_ipc_connection(self):
        """Handle incoming IPC connection from Assembler"""
        socket = self.ipc_server.nextPendingConnection()
        if socket:
            socket.readyRead.connect(lambda: self._handle_ipc_data(socket))
            socket.disconnected.connect(socket.deleteLater)

    def _handle_ipc_data(self, socket):
        """Handle IPC data received from Assembler"""
        try:
            data = socket.readAll().data().decode('utf-8')
            print(f"Received IPC request: {data[:100]}...")

            # Parse the file path from the request
            if data.startswith("LOAD:"):
                file_path = data[5:].strip()
                if os.path.exists(file_path):
                    # Load the file
                    self.load_file_from_external(file_path)
                    # Bring window to front and focus
                    self.raise_()
                    self.activateWindow()
                    socket.write(b"OK")
                else:
                    socket.write(b"ERROR: File not found")
            elif data.startswith("LOAD_JSON:"):
                import json
                json_str = data[10:].strip()
                try:
                    message_data = json.loads(json_str)
                    stl_file = message_data.get('stl_file')
                    collider_info = message_data.get('collider')
                    
                    if stl_file and os.path.exists(stl_file):
                        # Load the file with collider data
                        self.load_file_from_external(stl_file, collider_info)
                        # Bring window to front and focus
                        self.raise_()
                        self.activateWindow()
                        socket.write(b"OK")
                    else:
                        socket.write(b"ERROR: File not found")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    socket.write(b"ERROR: Invalid JSON format")
            else:
                socket.write(b"ERROR: Unknown command")

            socket.flush()
        except Exception as e:
            print(f"Error handling IPC data: {e}")
            import traceback
            traceback.print_exc()

    def load_file_from_external(self, stl_path, collider_info=None):
        """Load 3D model and XML file from external request (Assembler)
        
        Args:
            stl_path: Path to STL file
            collider_info: Optional collider information dict with 'type' and 'xml_path' or 'mesh_path'
        """
        try:
            print(f"Loading file from external request: {stl_path}")

            # Load the 3D model file
            self.file_name_label.setText(f"File: {stl_path}")
            self.show_stl(stl_path)

            # Try to load corresponding XML file
            xml_path = os.path.splitext(stl_path)[0] + '.xml'

            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # Load parameters from XML
                    has_parameters = self.load_parameters_from_xml(root)

                    # Recalculate if no parameters in XML
                    if not has_parameters:
                        self.calculate_and_update_properties()

                    # Load point data
                    points_with_data = self._load_points_from_xml(root)

                    print(f"XML file loaded: {xml_path}")
                    if points_with_data:
                        print(f"Loaded {len(points_with_data)} points")

                    # Refresh view
                    self.refresh_view()

                    # Reset camera
                    self.reset_camera()

                except ET.ParseError as e:
                    print(f"Error parsing XML file: {xml_path} - {e}")
                except Exception as e:
                    print(f"Error processing XML file: {str(e)}")
                    traceback.print_exc()
            else:
                print(f"No corresponding XML file found: {xml_path}")
                # Just load the mesh without XML
                self.calculate_and_update_properties()
                self.refresh_view()
                self.reset_camera()

            if collider_info:
                if collider_info.get('type') == 'primitive' and collider_info.get('xml_path'):
                    collider_xml_path = collider_info['xml_path']
                    if os.path.exists(collider_xml_path):
                        print(f"Collider XML file created by Assembler: {collider_xml_path}")
                        print(f"  This file will be available when saving from PartsEditor")
                    else:
                        print(f"Warning: Collider XML file not found: {collider_xml_path}")
                elif collider_info.get('type') == 'mesh' and collider_info.get('mesh_path'):
                    print(f"Collider mesh specified: {os.path.basename(collider_info['mesh_path'])}")

            # Check for collider XML file in the same directory
            stl_dir = os.path.dirname(stl_path)
            stl_basename = os.path.splitext(os.path.basename(stl_path))[0]
            auto_collider_xml_path = os.path.join(stl_dir, f"{stl_basename}_collider.xml")
            
            if os.path.exists(auto_collider_xml_path):
                print(f"Found collider XML file: {os.path.basename(auto_collider_xml_path)}")
                print(f"  This collider data will be preserved when saving from PartsEditor")

        except Exception as e:
            print(f"Error loading file from external request: {e}")
            import traceback
            traceback.print_exc()

    def showEvent(self, event):
        super().showEvent(event)
        if not self.vtk_initialized:
            QTimer.singleShot(100, self._initialize_vtk_delayed)
            self.vtk_initialized = True

    def _initialize_vtk_delayed(self):
        try:
            self.setup_vtk()
            QTimer.singleShot(100, self._vtk_setup_step2)
        except Exception as e:
            print(f"ERROR in VTK step 1: {e}")
            import traceback
            traceback.print_exc()

    def _vtk_setup_step2(self):
        try:
            self.setup_camera()
            QTimer.singleShot(100, self._vtk_setup_step3)
        except Exception as e:
            print(f"ERROR in VTK step 2: {e}")

    def _vtk_setup_step3(self):
        try:
            self.axes_widget = self.add_axes_widget()
            self.add_axes()
            self.add_instruction_text()
            for i in range(self.num_points):
                if self.point_checkboxes[i].isChecked():
                    self.show_point(i)
            QTimer.singleShot(100, self._vtk_setup_final)
        except Exception as e:
            print(f"ERROR in VTK step 3: {e}")
            import traceback
            traceback.print_exc()

    def _vtk_setup_final(self):
        try:
            QTimer.singleShot(200, self.render_to_image)
            QTimer.singleShot(300, lambda: self.vtk_display.setFocus())

            if self.pending_stl_file:
                QTimer.singleShot(400, self._load_pending_stl)
        except Exception as e:
            print(f"ERROR in VTK final step: {e}")
            import traceback
            traceback.print_exc()

    def setup_ui(self):
        self.setup_buttons()
        self.setup_stl_properties_ui()
        self.setup_points_ui()

    def setup_buttons(self):
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        first_row_widget = QWidget()
        first_row_layout = QHBoxLayout()
        first_row_layout.setContentsMargins(0, 0, 0, 0)
        first_row_layout.setSpacing(5)

        first_row = QHBoxLayout()
        button_layout.addLayout(first_row)

        self.load_button = QPushButton("Import Mesh")
        self.load_button.clicked.connect(self.load_stl_file)
        first_row.addWidget(self.load_button)

        self.load_xml_button = QPushButton("Load XML")
        self.load_xml_button.clicked.connect(self.load_xml_file)
        first_row.addWidget(self.load_xml_button)

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_files)
        first_row.addWidget(self.reload_button)

        first_row_widget.setLayout(first_row_layout)

        self.load_stl_xml_button = QPushButton("Load Mesh with XML")
        self.load_stl_xml_button.clicked.connect(self.load_stl_with_xml)
        button_layout.addWidget(self.load_stl_xml_button)

        spacer_top = QWidget()
        button_layout.addWidget(spacer_top)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #707070; }")
        button_layout.addWidget(separator)

        spacer_bottom = QWidget()
        button_layout.addWidget(spacer_bottom)

        self.mesh_sourcer_button = QPushButton("MeshSourcer")
        self.mesh_sourcer_button.clicked.connect(self.open_mesh_sourcer)
        button_layout.addWidget(self.mesh_sourcer_button)

        self.left_layout.addLayout(button_layout)

    def open_mesh_sourcer(self):
        """Open MeshSourcer with current mesh file"""
        import subprocess
        if not hasattr(self, 'stl_file_path') or not self.stl_file_path:
            QMessageBox.warning(
                self,
                "No Mesh Loaded",
                "Please load a mesh file first before opening MeshSourcer."
            )
            return

        if not os.path.exists(self.stl_file_path):
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The mesh file does not exist:\n{self.stl_file_path}"
            )
            return

        mesh_sourcer_path = os.path.join(os.path.dirname(__file__), "urdf_kitchen_MeshSourcer.py")

        if not os.path.exists(mesh_sourcer_path):
            QMessageBox.warning(
                self,
                "MeshSourcer Not Found",
                f"MeshSourcer script not found:\n{mesh_sourcer_path}"
            )
            return

        try:
            subprocess.Popen([sys.executable, mesh_sourcer_path, self.stl_file_path])
            print(f"Launched MeshSourcer with file: {self.stl_file_path}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch MeshSourcer:\n{str(e)}"
            )

    def setup_stl_properties_ui(self):
        grid_layout = QGridLayout()

        grid_layout.setVerticalSpacing(3)
        grid_layout.setHorizontalSpacing(5)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        grid_layout.setColumnMinimumWidth(0, 15)

        properties = [
            ("Volume (m^3):", "volume"),
            ("Density (kg/m^3):", "density"),
            ("Mass (kg):", "mass")
        ]

        current_row = 0
        for i, (label_text, prop_name) in enumerate(properties):
            checkbox = QCheckBox()
            setattr(self, f"{prop_name}_checkbox", checkbox)

            label = QLabel(label_text)
            input_field = QLineEdit("0.000000")
            setattr(self, f"{prop_name}_input", input_field)
            if prop_name == "volume":
                input_field.returnPressed.connect(self.apply_volume_value)
            elif prop_name == "density":
                input_field.returnPressed.connect(self.apply_density_value)
            elif prop_name == "mass":
                input_field.returnPressed.connect(self.apply_mass_value)

            grid_layout.addWidget(checkbox, current_row, 0)
            grid_layout.addWidget(label, current_row, 1)
            grid_layout.addWidget(input_field, current_row, 2)
            current_row += 1

        com_checkbox = QCheckBox()
        com_checkbox.stateChanged.connect(self.toggle_com)
        self.com_checkbox = com_checkbox
        grid_layout.addWidget(com_checkbox, current_row, 0)

        com_label = QLabel("Center of Mass:")
        grid_layout.addWidget(com_label, current_row, 1)
        com_layout = QHBoxLayout()
        com_layout.setSpacing(5)
        com_layout.setContentsMargins(0, 0, 0, 0)

        self.com_inputs = []
        for j, axis in enumerate(['X', 'Y', 'Z']):
            h_layout = QHBoxLayout()
            h_layout.setSpacing(2)
            h_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"{axis}:")
            label.setFixedWidth(15)
            h_layout.addWidget(label)

            input_field = QLineEdit("0.000000")
            input_field.setFixedWidth(80)
            input_field.returnPressed.connect(self.on_com_input_return)
            h_layout.addWidget(input_field)

            self.com_inputs.append(input_field)
            com_layout.addLayout(h_layout)

        grid_layout.addLayout(com_layout, current_row, 2)
        current_row += 1

        inertia_label = QLabel("Inertia Tensor:")
        grid_layout.addWidget(inertia_label, current_row, 1)

        self.inertia_tensor_input = QTextEdit()
        self.inertia_tensor_input.setReadOnly(True)
        self.inertia_tensor_input.setFixedHeight(40)
        self.inertia_tensor_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inertia_tensor_input.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inertia_tensor_input.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)

        font = self.inertia_tensor_input.font()
        font.setPointSize(10)
        self.inertia_tensor_input.setFont(font)

        grid_layout.addWidget(self.inertia_tensor_input, current_row, 2)
        current_row += 1

        self.density_input.setText("1.000000")

        pre_calculate_spacer = QWidget()
        pre_calculate_spacer.setFixedHeight(2)
        grid_layout.addWidget(pre_calculate_spacer, current_row, 0, 1, 3)
        current_row += 1

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_and_update_properties)
        grid_layout.addWidget(self.calculate_button, current_row, 1, 1, 2)
        current_row += 1

        spacer = QWidget()
        spacer.setFixedHeight(16)
        grid_layout.addWidget(spacer, current_row, 0, 1, 3)
        current_row += 1

        # Color layout using KitchenColorPicker
        color_layout = QHBoxLayout()

        color_layout.addWidget(QLabel("Color:"))

        # Add RGBA labels
        for label in ['R:', 'G:', 'B:', 'A:']:
            color_layout.addWidget(QLabel(label))

        # Create KitchenColorPicker instance
        self.color_picker = KitchenColorPicker(
            parent_widget=self,
            initial_color=[1.0, 1.0, 1.0, 1.0],  # White with full opacity
            enable_alpha=True,  # Enable alpha for transparency control
            on_color_changed=self._on_color_changed
        )

        # Add color picker widgets to layout
        self.color_picker.add_to_layout(color_layout)

        # Create aliases for backward compatibility
        self.color_inputs = self.color_picker.color_inputs
        self.color_sample = self.color_picker.color_sample

        # Connect Enter key to apply color
        for color_input in self.color_inputs:
            color_input.returnPressed.connect(self.apply_color_to_stl)

        color_layout.addStretch()

        grid_layout.addLayout(color_layout, current_row, 0, 1, 3)
        current_row += 1

        axis_layout = QHBoxLayout()

        self.axis_group = QButtonGroup(self)
        axis_label = QLabel("Axis:")
        axis_layout.addWidget(axis_label)

        radio_texts = ["X:roll", "Y:pitch", "Z:yaw", "fixed"]
        self.radio_buttons = []
        for i, text in enumerate(radio_texts):
            radio = QRadioButton(text)
            self.axis_group.addButton(radio, i)
            axis_layout.addWidget(radio)
            self.radio_buttons.append(radio)
        self.radio_buttons[0].setChecked(True)

        self.rotate_test_button = QPushButton("Rotate Test")
        self.rotate_test_button.pressed.connect(self.start_rotation_test)
        self.rotate_test_button.released.connect(self.stop_rotation_test)
        axis_layout.addWidget(self.rotate_test_button)

        grid_layout.addLayout(axis_layout, current_row, 0, 1, 3)
        current_row += 1

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle (deg):"))

        angle_layout.addWidget(QLabel("X:"))
        self.angle_x_input = QLineEdit()
        self.angle_x_input.setFixedWidth(60)
        self.angle_x_input.setText("0.0")
        self.angle_x_input.setToolTip("Body initial rotation around X axis (degrees)")
        angle_layout.addWidget(self.angle_x_input)

        angle_layout.addWidget(QLabel("Y:"))
        self.angle_y_input = QLineEdit()
        self.angle_y_input.setFixedWidth(60)
        self.angle_y_input.setText("0.0")
        self.angle_y_input.setToolTip("Body initial rotation around Y axis (degrees)")
        angle_layout.addWidget(self.angle_y_input)

        angle_layout.addWidget(QLabel("Z:"))
        self.angle_z_input = QLineEdit()
        self.angle_z_input.setFixedWidth(60)
        self.angle_z_input.setText("0.0")
        self.angle_z_input.setToolTip("Body initial rotation around Z axis (degrees)")
        angle_layout.addWidget(self.angle_z_input)

        self.angle_x_input.returnPressed.connect(self.save_current_point_angles)
        self.angle_y_input.returnPressed.connect(self.save_current_point_angles)
        self.angle_z_input.returnPressed.connect(self.save_current_point_angles)

        angle_layout.addStretch()
        grid_layout.addLayout(angle_layout, current_row, 0, 1, 3)
        current_row += 1

        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.update_test_rotation)
        self.original_transform = None
        self.test_rotation_angle = 0

        self.left_layout.addLayout(grid_layout)

    def setup_points_ui(self):
        points_layout = QGridLayout()

        points_layout.setContentsMargins(0, 0, 0, 0)
        points_layout.setVerticalSpacing(3)
        points_layout.setHorizontalSpacing(15)

        for i in range(self.num_points):
            row = i

            checkbox = QCheckBox(f"Point {i+1}")
            checkbox.setObjectName("point_checkbox")
            checkbox.setMinimumWidth(80)
            checkbox.stateChanged.connect(lambda state, index=i: self.toggle_point(state, index))
            self.point_checkboxes.append(checkbox)
            points_layout.addWidget(checkbox, row, 0)

            inputs = []
            axis_colors = [UI_COLOR_X, UI_COLOR_Y, UI_COLOR_Z]

            for j, axis in enumerate(['X', 'Y', 'Z']):
                h_layout = QHBoxLayout()
                h_layout.setSpacing(2)
                h_layout.setContentsMargins(0, 0, 0, 0)

                label = QLabel(f"{axis}:")
                label.setFixedWidth(15)
                label.setStyleSheet(f"QLabel {{ color: {axis_colors[j]}; font-weight: bold; }}")
                h_layout.addWidget(label)

                input_field = QLineEdit(str(self.point_coords[i][j]))
                input_field.setFixedWidth(80)
                input_field.returnPressed.connect(lambda idx=i: self.set_point(idx))

                h_layout.addWidget(input_field)
                h_layout.addStretch()

                container = QWidget()
                container.setLayout(h_layout)
                points_layout.addWidget(container, row, j + 1)

                inputs.append(input_field)

            self.point_inputs.append(inputs)

        points_layout.setColumnStretch(0, 1)
        points_layout.setColumnStretch(1, 1)
        points_layout.setColumnStretch(2, 1)
        points_layout.setColumnStretch(3, 1)

        button_row = self.num_points
        reset_button = QPushButton("Reset Point")
        reset_button.clicked.connect(self.handle_reset_only)
        char_width = reset_button.fontMetrics().averageCharWidth()
        reset_button.setFixedWidth(reset_button.fontMetrics().boundingRect("Reset Point").width() + 20 + char_width * 4)
        points_layout.addWidget(reset_button, button_row, 0, 1, 4, Qt.AlignRight)

        self.left_layout.addLayout(points_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #707070; }")
        self.left_layout.addWidget(separator)

        export_layout = QVBoxLayout()
        export_layout.setSpacing(5)

        export_row1 = QHBoxLayout()
        export_row1.setSpacing(5)

        self.export_urdf_button = QPushButton("Export XML")
        self.export_urdf_button.clicked.connect(self.export_urdf)
        export_row1.addWidget(self.export_urdf_button)

        self.export_mirror_button = QPushButton("Export Mirror Mesh with XML")
        self.export_mirror_button.clicked.connect(self.export_mirror_stl_xml)
        export_row1.addWidget(self.export_mirror_button)

        export_layout.addLayout(export_row1)

        self.bulk_convert_button = QPushButton("Batch Mirror \"l_\" to \"r_\" Meshes and XMLs")
        self.bulk_convert_button.clicked.connect(self.bulk_convert_l_to_r)
        export_layout.addWidget(self.bulk_convert_button)

        self.left_layout.addLayout(export_layout)
        
    def set_point(self, index):
        try:
            x = float(self.point_inputs[index][0].text())
            y = float(self.point_inputs[index][1].text())
            z = float(self.point_inputs[index][2].text())
            self.point_coords[index] = [x, y, z]
            
            if self.point_checkboxes[index].isChecked():
                self.show_point(index)
            else:
                self.update_point_display(index)
            
            print(f"Point {index+1} set to: ({x}, {y}, {z})")
        except ValueError:
            print(f"Invalid input for Point {index+1}. Please enter valid numbers for coordinates.")

    def render_to_image(self):
        """Render VTK scene offscreen and display as image in QLabel"""
        self.offscreen_renderer.update_display(self.vtk_display, restore_focus=True)
        self._render_counter += 1

    def reset_point_to_origin(self, index):
        self.point_coords[index] = list(self.absolute_origin)
        self.update_point_display(index)
        if self.point_checkboxes[index].isChecked():
            self.show_point(index)
        print(f"Point {index+1} reset to origin {self.absolute_origin}")

    def reset_camera(self):
        position = [self.absolute_origin[i] + self.initial_camera_position[i] for i in range(3)]
        self.camera_controller.reset_camera(position=position, view_up=self.initial_camera_view_up)

        self.camera_rotation = [0, 0, 0]
        self.current_rotation = 0

        self.render_to_image()
        self.update_all_points()

    def _load_pending_stl(self):
        """Load STL file from command line arguments after VTK initialization"""
        if not self.pending_stl_file or not os.path.exists(self.pending_stl_file):
            return

        try:
            self.show_stl(self.pending_stl_file)

            xml_path = os.path.splitext(self.pending_stl_file)[0] + '.xml'
            if os.path.exists(xml_path):
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    has_parameters = self.load_parameters_from_xml(root)

                    if not has_parameters:
                        self.calculate_and_update_properties()
                    self._load_points_from_xml(root)
                    self.refresh_view()
                    self.reset_camera()

                    print(f"Loaded: {self.pending_stl_file}")
                    print(f"Loaded: {xml_path}")
                except Exception as e:
                    print(f"Error loading XML: {str(e)}")
            else:
                print(f"Loaded: {self.pending_stl_file}")
        except Exception as e:
            print(f"Error loading STL file: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.pending_stl_file = None

    def toggle_wireframe(self):
        """Toggle wireframe mode for STL actor"""
        if not self.stl_actor:
            return

        property = self.stl_actor.GetProperty()

        if property.GetRepresentation() == vtk.VTK_SURFACE:
            property.SetRepresentationToWireframe()
            print("[T] Wireframe mode ON")
        else:
            property.SetRepresentationToSurface()
            print("[T] Wireframe mode OFF")

        self.stl_actor.Modified()
        self.renderer.Modified()
        self.render_to_image()

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.isAutoRepeat():
            return

        key = event.key()
        modifiers = event.modifiers()
        shift_pressed = modifiers & Qt.ShiftModifier
        ctrl_pressed = modifiers & Qt.ControlModifier

        # Block WASDQER keys during animation
        if self.is_animating and key in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E, Qt.Key_R]:
            return

        if key == Qt.Key_W:
            print("[W] Rotate 90° up (pitch)")
            self.rotate_camera(-90, 'pitch')
        elif key == Qt.Key_S:
            print("[S] Rotate 90° down (pitch)")
            self.rotate_camera(90, 'pitch')
        elif key == Qt.Key_A:
            print("[A] Rotate 90° left (yaw)")
            self.rotate_camera(90, 'yaw')
        elif key == Qt.Key_D:
            print("[D] Rotate 90° right (yaw)")
            self.rotate_camera(-90, 'yaw')
        elif key == Qt.Key_Q:
            print("[Q] Rotate 90° counterclockwise (roll)")
            self.rotate_camera(90, 'roll')
        elif key == Qt.Key_E:
            print("[E] Rotate 90° clockwise (roll)")
            self.rotate_camera(-90, 'roll')
        elif key == Qt.Key_R:
            print("[R] Reset camera")
            self.reset_camera()
        elif key == Qt.Key_T:
            self.toggle_wireframe()
        elif key in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            step = 0.01  # 10mm
            if shift_pressed and ctrl_pressed:
                step = 0.0001  # 0.1mm
            elif shift_pressed:
                step = 0.001  # 1mm

            horizontal_axis, vertical_axis, screen_right, screen_up = self.get_screen_axes()

            if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
                if key == Qt.Key_Up:
                    self.move_com_screen(screen_up, step)
                elif key == Qt.Key_Down:
                    self.move_com_screen(screen_up, -step)
                elif key == Qt.Key_Left:
                    self.move_com_screen(screen_right, -step)
                elif key == Qt.Key_Right:
                    self.move_com_screen(screen_right, step)
                self.render_to_image()

            for i, checkbox in enumerate(self.point_checkboxes):
                if checkbox.isChecked():
                    if key == Qt.Key_Up:
                        self.move_point_screen(i, screen_up, step)
                    elif key == Qt.Key_Down:
                        self.move_point_screen(i, screen_up, -step)
                    elif key == Qt.Key_Left:
                        self.move_point_screen(i, screen_right, -step)
                    elif key == Qt.Key_Right:
                        self.move_point_screen(i, screen_right, step)
                    self.render_to_image()

        super().keyPressEvent(event)

    def update_point_position(self, index, x, y):
        renderer = self.renderer
        camera = renderer.GetActiveCamera()

        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        coordinate.SetValue(x, y, 0)
        world_pos = coordinate.GetComputedWorldValue(renderer)

        camera_pos = np.array(camera.GetPosition())
        focal_point = np.array(camera.GetFocalPoint())
        view_direction = focal_point - camera_pos
        view_direction /= np.linalg.norm(view_direction)

        current_z = self.point_coords[index][2]
        t = (current_z - camera_pos[2]) / view_direction[2]
        new_pos = camera_pos + t * view_direction

        self.point_coords[index] = [new_pos[0], new_pos[1], current_z]
        self.update_point_display(index)

        print(f"Point {index+1} moved to: ({new_pos[0]:.6f}, {new_pos[1]:.6f}, {current_z:.6f})")

    def update_inertia_from_mass(self, mass):
        inertia = mass * 0.1
        self.inertia_input.setText(f"{inertia:.12f}")

    def update_properties(self):
        priority_order = ['mass', 'volume', 'inertia', 'density']
        values = {}

        for prop in priority_order:
            checkbox = getattr(self, f"{prop}_checkbox")
            input_field = getattr(self, f"{prop}_input")
            if checkbox.isChecked():
                try:
                    values[prop] = float(input_field.text())
                except ValueError:
                    print(f"Invalid input for {prop}")
                    return

        if 'mass' in values and 'volume' in values:
            values['density'] = values['mass'] / values['volume']
        elif 'mass' in values and 'density' in values:
            values['volume'] = values['mass'] / values['density']
        elif 'volume' in values and 'density' in values:
            values['mass'] = values['volume'] * values['density']

        if 'mass' in values and 'volume' in values:
            side_length = np.cbrt(values['volume'])
            values['inertia'] = (1/6) * values['mass'] * side_length**2

        for prop in priority_order:
            input_field = getattr(self, f"{prop}_input")
            if prop in values:
                input_field.setText(f"{values[prop]:.12f}")

    def update_all_points_size(self, obj=None, event=None):
        """Update point sizes"""
        for index, actor in enumerate(self.point_actors):
            if actor:
                is_checked = self.point_checkboxes[index].isChecked()
                self.renderer.RemoveActor(actor)
                self.point_actors[index] = create_crosshair_marker(
                    coords=[0, 0, 0],
                    radius_scale=self.calculate_sphere_radius()
                )
                self.point_actors[index].SetPosition(self.point_coords[index])
                
                if is_checked:
                    self.renderer.AddActor(self.point_actors[index])
                    self.point_actors[index].VisibilityOn()
                else:
                    self.point_actors[index].VisibilityOff()

        self.render_to_image()

    def update_all_points(self):
        """Update all point displays"""
        for i in range(self.num_points):
            if self.point_actors[i]:
                if self.point_checkboxes[i].isChecked():
                    self.point_actors[i].SetPosition(self.point_coords[i])
                    self.point_actors[i].VisibilityOn()
                    self.renderer.AddActor(self.point_actors[i])
                else:
                    self.point_actors[i].VisibilityOff()
                    self.renderer.RemoveActor(self.point_actors[i])

        self.render_to_image()

    def calculate_sphere_radius(self):
        viewport_size = self.renderer.GetSize()
        diagonal = math.sqrt(viewport_size[0]**2 + viewport_size[1]**2)
        radius = (diagonal * 0.1) / 2

        camera = self.renderer.GetActiveCamera()
        parallel_scale = camera.GetParallelScale()
        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / (viewport[3] - viewport[1])

        scaled_radius = (radius / viewport_size[1]) * parallel_scale * 2

        if aspect_ratio > 1:
            scaled_radius /= aspect_ratio

        return scaled_radius

    def calculate_properties(self):
        priority_order = ['volume', 'density', 'mass', 'inertia']
        values = {}

        for prop in priority_order:
            checkbox = getattr(self, f"{prop}_checkbox")
            input_field = getattr(self, f"{prop}_input")
            if checkbox.isChecked():
                try:
                    values[prop] = float(input_field.text())
                except ValueError:
                    print(f"Invalid input for {prop}")
                    return

        if 'volume' in values and 'density' in values:
            values['mass'] = values['volume'] * values['density']
        elif 'mass' in values and 'volume' in values:
            values['density'] = values['mass'] / values['volume']
        elif 'mass' in values and 'density' in values:
            values['volume'] = values['mass'] / values['density']

        if 'mass' in values and 'volume' in values:
            side_length = np.cbrt(values['volume'])
            values['inertia'] = (1/6) * values['mass'] * side_length**2

        for prop in priority_order:
            input_field = getattr(self, f"{prop}_input")
            if prop in values:
                input_field.setText(f"{values[prop]:.12f}")

    def calculate_and_update_properties(self):
        """
        Calculate physical properties based on checkbox states.

        Rules:
        1. If no checkboxes checked: Calculate Inertia Tensor from Mass and Center of Mass
        2. If Volume+Mass checked, Density unchecked: Calculate Density
        3. General: Calculate unchecked properties from checked properties
        4. Checked items are fixed/given values - never update them
        5. Inertia Tensor is always recalculated
        6. Center of Mass: If checked, use given value; if unchecked, calculate from STL
        """
        try:
            # Get checkbox states
            volume_checked = self.volume_checkbox.isChecked()
            density_checked = self.density_checkbox.isChecked()
            mass_checked = self.mass_checkbox.isChecked()
            com_checked = self.com_checkbox.isChecked()

            # Get STL volume (needed for calculations when volume is unchecked)
            stl_volume = None
            if hasattr(self, 'stl_actor') and self.stl_actor:
                poly_data = self.stl_actor.GetMapper().GetInput()
                mass_properties = vtk.vtkMassProperties()
                mass_properties.SetInputData(poly_data)
                stl_volume = mass_properties.GetVolume()

            # Count how many properties are checked
            checked_count = sum([volume_checked, density_checked, mass_checked])

            # Get values for checked properties
            volume_value = None
            density_value = None
            mass_value = None

            if volume_checked:
                try:
                    volume_value = float(self.volume_input.text())
                except ValueError:
                    print("Error: Invalid volume input")
                    return None

            if density_checked:
                try:
                    density_value = float(self.density_input.text())
                except ValueError:
                    print("Error: Invalid density input")
                    return None

            if mass_checked:
                try:
                    mass_value = float(self.mass_input.text())
                except ValueError:
                    print("Error: Invalid mass input")
                    return None

            # Apply calculation rules based on checked properties
            if checked_count >= 2:
                # Case: Two or more properties are checked - calculate the unchecked one(s)

                if volume_checked and mass_checked and not density_checked:
                    # Volume + Mass → Calculate Density
                    density_value = mass_value / volume_value
                    self.density_input.setText(f"{density_value:.12f}")
                    print(f"Calculated Density: {density_value:.6f}")

                elif volume_checked and density_checked and not mass_checked:
                    # Volume + Density → Calculate Mass
                    mass_value = volume_value * density_value
                    self.mass_input.setText(f"{mass_value:.12f}")
                    print(f"Calculated Mass: {mass_value:.6f}")

                elif density_checked and mass_checked and not volume_checked:
                    # Density + Mass → Calculate Volume
                    volume_value = mass_value / density_value
                    self.volume_input.setText(f"{volume_value:.12f}")
                    print(f"Calculated Volume: {volume_value:.6f}")

                elif volume_checked and density_checked and mass_checked:
                    # All three checked - verify consistency
                    calculated_mass = volume_value * density_value
                    if abs(calculated_mass - mass_value) > 1e-6:
                        print(f"Warning: Inconsistent values. Volume×Density={calculated_mass:.6f} but Mass={mass_value:.6f}")

            elif checked_count == 1:
                # Case: Only one property is checked - use STL volume to calculate others

                if stl_volume is None:
                    print("Warning: STL model required to calculate properties from single input")
                else:
                    if volume_checked:
                        # Volume checked, but we need at least one more value
                        print("Warning: Need at least Density or Mass to calculate remaining properties")

                    elif density_checked:
                        # Density only → First calculate Volume from mesh, then Mass
                        volume_value = stl_volume
                        self.volume_input.setText(f"{volume_value:.12f}")
                        print(f"Step 1: Calculated Volume from mesh: {volume_value:.6f}")

                        mass_value = volume_value * density_value
                        self.mass_input.setText(f"{mass_value:.12f}")
                        print(f"Step 2: Using Density {density_value:.6f}, calculated Mass: {mass_value:.6f}")

                    elif mass_checked:
                        # Mass only → First calculate Volume from mesh, then Density
                        volume_value = stl_volume
                        self.volume_input.setText(f"{volume_value:.12f}")
                        print(f"Step 1: Calculated Volume from mesh: {volume_value:.6f}")

                        density_value = mass_value / volume_value
                        self.density_input.setText(f"{density_value:.12f}")
                        print(f"Step 2: Using Mass {mass_value:.6f}, calculated Density: {density_value:.6f}")

            else:
                # Case: No checkboxes checked - calculate Volume from mesh, then calculate other properties
                print("No properties checked - calculating Volume from mesh first")

                if stl_volume is None:
                    print("Error: STL model required to calculate Volume")
                else:
                    # Step 1: Calculate and set Volume from mesh
                    volume_value = stl_volume
                    self.volume_input.setText(f"{volume_value:.12f}")
                    print(f"Calculated Volume from mesh: {volume_value:.6f}")

                    # Step 2: Try to read existing Density or Mass values from input fields
                    existing_density = None
                    existing_mass = None

                    try:
                        if self.density_input.text():
                            existing_density = float(self.density_input.text())
                    except ValueError:
                        pass

                    try:
                        if self.mass_input.text():
                            existing_mass = float(self.mass_input.text())
                    except ValueError:
                        pass

                    # Step 3: Calculate missing properties based on available values
                    if existing_density is not None and existing_mass is not None:
                        # Both Density and Mass exist - verify consistency
                        calculated_mass = volume_value * existing_density
                        if abs(calculated_mass - existing_mass) > 1e-6:
                            print(f"Warning: Inconsistent values. Volume×Density={calculated_mass:.6f} but existing Mass={existing_mass:.6f}")
                            print(f"Using Density to recalculate Mass")
                            mass_value = calculated_mass
                            self.mass_input.setText(f"{mass_value:.12f}")
                        else:
                            density_value = existing_density
                            mass_value = existing_mass
                            print(f"Using existing Density: {density_value:.6f}, Mass: {mass_value:.6f}")

                    elif existing_density is not None:
                        # Only Density exists - calculate Mass
                        density_value = existing_density
                        mass_value = volume_value * density_value
                        self.mass_input.setText(f"{mass_value:.12f}")
                        print(f"Using existing Density: {density_value:.6f}, Calculated Mass: {mass_value:.6f}")

                    elif existing_mass is not None:
                        # Only Mass exists - calculate Density
                        mass_value = existing_mass
                        density_value = mass_value / volume_value
                        self.density_input.setText(f"{density_value:.12f}")
                        print(f"Using existing Mass: {mass_value:.6f}, Calculated Density: {density_value:.6f}")

                    else:
                        # Neither Density nor Mass exists - cannot proceed with full calculation
                        print("Warning: No Density or Mass values found. Please provide at least one value.")
                        print("Inertia Tensor calculation may use default or previous values.")

            # Handle Center of Mass according to checkbox state
            if com_checked:
                # COM is checked - use the given value (DO NOT recalculate)
                try:
                    center_of_mass = [float(self.com_inputs[i].text()) for i in range(3)]
                    self.com_coords = list(center_of_mass)
                    print(f"Using fixed Center of Mass: [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]")

                    # Update visualization only (don't recalculate)
                    self.update_com_visualization(center_of_mass)

                except (ValueError, IndexError) as e:
                    print(f"Error parsing Center of Mass input: {e}")
                    return None
            else:
                # COM is unchecked - calculate from STL mesh
                self.calculate_center_of_mass()

            # Always recalculate Inertia Tensor
            print("Recalculating Inertia Tensor...")
            self.calculate_inertia_tensor()

            print("Properties calculation completed successfully")
            return True

        except (ValueError, ZeroDivisionError) as e:
            print(f"An error occurred during calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def update_com_visualization(self, center_of_mass):
        """
        Update the Center of Mass sphere visualization without recalculating the value.

        Args:
            center_of_mass: List/tuple of [x, y, z] coordinates
        """
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No 3D model has been loaded.")
            return

        # Update com_coords
        self.com_coords = list(center_of_mass)

        # Remove existing red sphere actor
        if hasattr(self, 'com_sphere_actor') and self.com_sphere_actor:
            self.renderer.RemoveActor(self.com_sphere_actor)

        # Create COM visualization (red sphere)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center_of_mass)
        sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        self.com_sphere_actor = vtk.vtkActor()
        self.com_sphere_actor.SetMapper(mapper)
        self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red color
        self.com_sphere_actor.GetProperty().SetOpacity(0.7)

        # Show/hide based on checkbox state
        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            self.com_sphere_actor.VisibilityOff()
        else:
            self.com_sphere_actor.VisibilityOn()

        self.renderer.AddActor(self.com_sphere_actor)
        self.render_to_image()

    def calculate_center_of_mass(self):
        """Calculate and display center of mass using tetrahedral decomposition"""
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No 3D model has been loaded.")
            return

        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            try:
                center_of_mass = [float(self.com_inputs[i].text()) for i in range(3)]
                print(f"Using manual Center of Mass: {center_of_mass}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing Center of Mass input: {e}")
                return None
        else:
            print("Calculating Center of Mass using tetrahedral decomposition (uniform density assumed)...")
            poly_data = self.stl_actor.GetMapper().GetInput()
            total_volume = 0.0
            weighted_com = np.zeros(3)

            num_cells = poly_data.GetNumberOfCells()

            for i in range(num_cells):
                cell = poly_data.GetCell(i)
                if cell.GetCellType() == vtk.VTK_TRIANGLE:
                    # Get triangle vertices
                    v0 = np.array(cell.GetPoints().GetPoint(0))
                    v1 = np.array(cell.GetPoints().GetPoint(1))
                    v2 = np.array(cell.GetPoints().GetPoint(2))

                    # Signed volume of tetrahedron formed by origin and triangle
                    # V = (1/6) * v0 · (v1 × v2)
                    tet_volume = np.dot(v0, np.cross(v1, v2)) / 6.0

                    # Skip degenerate triangles
                    if abs(tet_volume) < 1e-12:
                        continue

                    # Center of mass of tetrahedron with one vertex at origin
                    # COM = (v0 + v1 + v2) / 4
                    tet_com = (v0 + v1 + v2) / 4.0

                    # Accumulate volume-weighted center of mass
                    total_volume += tet_volume
                    weighted_com += tet_volume * tet_com

            if abs(total_volume) < 1e-12:
                print("Error: Total volume is zero or nearly zero")
                return None

            # Calculate final center of mass
            center_of_mass = weighted_com / total_volume

            # Verify with VTK's calculation
            mass_properties = vtk.vtkMassProperties()
            mass_properties.SetInputData(poly_data)
            mass_properties.Update()
            vtk_volume = mass_properties.GetVolume()

            print(f"Volume verification: Tetrahedral={total_volume:.6f}, VTK={vtk_volume:.6f}")
            volume_ratio = total_volume / vtk_volume if vtk_volume > 0 else 0
            if abs(volume_ratio - 1.0) > 0.01:
                print(f"Warning: Volume mismatch! Ratio = {volume_ratio:.4f}")

            for i in range(3):
                self.com_inputs[i].setText(f"{center_of_mass[i]:.6f}")
            print(f"Calculated Center of Mass (volumetric): [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]")

        self.com_coords = list(center_of_mass)

        if hasattr(self, 'com_sphere_actor') and self.com_sphere_actor:
            self.renderer.RemoveActor(self.com_sphere_actor)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center_of_mass)
        sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        self.com_sphere_actor = vtk.vtkActor()
        self.com_sphere_actor.SetMapper(mapper)
        self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)
        self.com_sphere_actor.GetProperty().SetOpacity(0.7)

        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            self.com_sphere_actor.VisibilityOff()
        else:
            self.com_sphere_actor.VisibilityOn()

        self.renderer.AddActor(self.com_sphere_actor)
        self.render_to_image()

        return center_of_mass

    def calculate_inertia_tensor(self):
        """Calculate inertia tensor using trimesh/VTK hybrid method"""
        if not hasattr(self, 'stl_file_path') or not self.stl_file_path:
            print("No 3D model file is loaded.")
            return None

        # Get density (try from input field, fall back to default)
        try:
            density = float(self.density_input.text())
        except (ValueError, AttributeError):
            print("Warning: Could not read density, using 1.0")
            density = 1.0

        center_of_mass = None
        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            center_of_mass = self.get_center_of_mass()
            if center_of_mass is None:
                print("Error getting center of mass from UI")
                return None

        result = calculate_inertia_with_trimesh(
            mesh_file_path=self.stl_file_path,
            mass=None,
            center_of_mass=center_of_mass,
            density=density,
            auto_repair=True
        )

        if not result['success']:
            print(f"Error calculating inertia: {result['error_message']}")
            return None

        print(f"Volume: {result['volume']:.6f}, Density: {density:.6f}, Mass: {result['mass']:.6f}")
        print(f"Center of Mass (used): [{result['center_of_mass'][0]:.6f}, {result['center_of_mass'][1]:.6f}, {result['center_of_mass'][2]:.6f}]")
        print(f"Center of Mass (trimesh): [{result['trimesh_com'][0]:.6f}, {result['trimesh_com'][1]:.6f}, {result['trimesh_com'][2]:.6f}]")
        print(f"Watertight: {'Yes' if result['is_watertight'] else 'No'}")
        if result['repair_performed']:
            print("Mesh repair was performed")

        inertia_tensor = result['inertia_tensor']
        print("\nCalculated Inertia Tensor (about Center of Mass):")
        print(inertia_tensor)

        urdf_inertia = self.format_inertia_for_urdf(inertia_tensor)
        if hasattr(self, 'inertia_tensor_input'):
            self.inertia_tensor_input.setText(urdf_inertia)
            print("\nInertia tensor has been updated in UI")
        else:
            print("Warning: inertia_tensor_input not found")

        return inertia_tensor

    def export_urdf(self):
        if not hasattr(self, 'stl_file_path'):
            print("No 3D model file has been loaded.")
            return

        stl_dir = os.path.dirname(self.stl_file_path)
        stl_filename = os.path.basename(self.stl_file_path)
        stl_name_without_ext = os.path.splitext(stl_filename)[0]

        default_urdf_filename = f"{stl_name_without_ext}.xml"
        urdf_file_path, _ = QFileDialog.getSaveFileName(self, "Export XML File", os.path.join(
            stl_dir, default_urdf_filename), "XML Files (*.xml)")

        if not urdf_file_path:
            return

        try:
            rgb_values = [float(input.text()) for input in self.color_inputs]
            hex_color = '#{:02X}{:02X}{:02X}'.format(
                int(rgb_values[0] * 255),
                int(rgb_values[1] * 255),
                int(rgb_values[2] * 255)
            )
            rgba_str = f"{rgb_values[0]:.6f} {rgb_values[1]:.6f} {rgb_values[2]:.6f} 1.0"

            try:
                com_values = [float(self.com_inputs[i].text()) for i in range(3)]
                center_of_mass_str = f"{com_values[0]:.6f} {com_values[1]:.6f} {com_values[2]:.6f}"
            except (ValueError, IndexError):
                print("Warning: Invalid center of mass format, using default values")
                center_of_mass_str = "0.000000 0.000000 0.000000"

            axis_options = ["1 0 0", "0 1 0", "0 0 1", "0 0 0"]
            checked_id = self.axis_group.checkedId()
            if 0 <= checked_id < len(axis_options):
                axis_vector = axis_options[checked_id]
            else:
                print("Warning: No axis selected, using default X axis")
                axis_vector = "1 0 0"
            urdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urdf_part>
    <material name="{hex_color}">
        <color rgba="{rgba_str}" />
    </material>
    <link name="{stl_name_without_ext}">
        <visual>
            <origin xyz="{center_of_mass_str}" rpy="0 0 0"/>
            <material name="{hex_color}" />
        </visual>
        <inertial>
            <origin xyz="{center_of_mass_str}"/>
            <mass value="{self.mass_input.text()}"/>
            <volume value="{self.volume_input.text()}"/>
            {self.inertia_tensor_input.toPlainText().strip()}
        </inertial>
        <center_of_mass>{center_of_mass_str}</center_of_mass>
    </link>"""

            for i, checkbox in enumerate(self.point_checkboxes):
                if checkbox.isChecked():
                    x, y, z = self.point_coords[i]
                    angle_x, angle_y, angle_z = self.point_angles[i]
                    urdf_content += f"""
    <point name="point{i+1}" type="fixed">
        <point_xyz>{x:.6f} {y:.6f} {z:.6f}</point_xyz>
        <point_angle>{angle_x:.6f} {angle_y:.6f} {angle_z:.6f}</point_angle>
    </point>"""

            urdf_content += f"""
    <joint>
        <axis xyz="{axis_vector}" />
    </joint>
</urdf_part>"""

            with open(urdf_file_path, "w") as f:
                f.write(urdf_content)
            print(f"URDF file saved: {urdf_file_path}")

        except Exception as e:
            print(f"Error during URDF export: {str(e)}")
            traceback.print_exc()

    def get_center_of_mass(self):
        """Get center of mass from UI or calculate from mesh"""
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No 3D model is loaded.")
            return None

        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            try:
                center_of_mass = np.array([float(self.com_inputs[i].text()) for i in range(3)])
                print(f"Using manual Center of Mass: {center_of_mass}")
                return center_of_mass
            except (ValueError, IndexError) as e:
                print(f"Error parsing Center of Mass input: {e}")
                return None

        try:
            poly_data = self.stl_actor.GetMapper().GetInput()
            com_filter = vtk.vtkCenterOfMass()
            com_filter.SetInputData(poly_data)
            com_filter.SetUseScalarsAsWeights(False)
            com_filter.Update()
            center_of_mass = np.array(com_filter.GetCenter())
            print(f"Using calculated Center of Mass: {center_of_mass}")
            return center_of_mass
        except Exception as e:
            print(f"Error calculating center of mass: {e}")
            return None

    def format_inertia_for_urdf(self, inertia_tensor):
        """Format inertia tensor to URDF string"""
        threshold = 1e-10

        ixx = inertia_tensor[0][0] if abs(inertia_tensor[0][0]) > threshold else 0
        iyy = inertia_tensor[1][1] if abs(inertia_tensor[1][1]) > threshold else 0
        izz = inertia_tensor[2][2] if abs(inertia_tensor[2][2]) > threshold else 0

        ixy = inertia_tensor[0][1] if abs(inertia_tensor[0][1]) > threshold else 0
        ixz = inertia_tensor[0][2] if abs(inertia_tensor[0][2]) > threshold else 0
        iyz = inertia_tensor[1][2] if abs(inertia_tensor[1][2]) > threshold else 0

        return f'<inertia ixx="{ixx:.8f}" ixy="{ixy:.8f}" ixz="{ixz:.8f}" iyy="{iyy:.8f}" iyz="{iyz:.8f}" izz="{izz:.8f}"/>'

    def add_axes(self):
        if not hasattr(self, 'axes_actors'):
            self.axes_actors = []

        for actor in self.axes_actors:
            self.renderer.RemoveActor(actor)
        self.axes_actors.clear()

        axis_length = 5
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for i, color in enumerate(colors):
            for direction in [1, -1]:
                line_source = vtk.vtkLineSource()
                line_source.SetPoint1(*self.absolute_origin)
                end_point = np.array(self.absolute_origin)
                end_point[i] += axis_length * direction
                line_source.SetPoint2(*end_point)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(line_source.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetLineWidth(2)

                self.renderer.AddActor(actor)
                self.axes_actors.append(actor)

        self.render_to_image()

    def load_stl_file(self):
        # Use common utility function for file filter
        file_filter = get_mesh_file_filter(trimesh_available=True)

        file_path, _ = QFileDialog.getOpenFileName(self, "Open 3D Model File", "", file_filter)
        if file_path:
            self.file_name_label.setText(f"File: {file_path}")
            self.show_stl(file_path)

    def show_stl(self, file_path):
        # Remove old actors
        if hasattr(self, 'stl_actor') and self.stl_actor:
            self.renderer.RemoveActor(self.stl_actor)
        if hasattr(self, 'com_actor') and self.com_actor:
            self.renderer.RemoveActor(self.com_actor)
            self.com_actor = None

        self.renderer.Clear()
        self.axes_widget = self.add_axes_widget()
        self.stl_file_path = file_path

        poly_data, volume, extracted_color = load_mesh_to_polydata(file_path)

        if extracted_color is not None:
            print(f"Extracted color from .dae file: RGBA({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f}, {extracted_color[3]:.3f})")
            self.mesh_color = extracted_color
            self.color_manually_changed = False
            if hasattr(self, 'color_button'):
                color = QtGui.QColor(int(extracted_color[0]*255), int(extracted_color[1]*255), int(extracted_color[2]*255))
                self.color_button.setStyleSheet(f"background-color: {color.name()};")

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        self.stl_actor = vtk.vtkActor()
        self.stl_actor.SetMapper(mapper)

        if extracted_color is not None:
            self.stl_actor.GetProperty().SetColor(extracted_color[0], extracted_color[1], extracted_color[2])
            self.stl_actor.GetProperty().SetOpacity(extracted_color[3])
            print(f"Applied color to 3D view: RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")

        self.model_bounds = poly_data.GetBounds()
        self.renderer.AddActor(self.stl_actor)

        self.volume_input.setText(f"{volume:.12f}")

        density = float(self.density_input.text())
        mass = volume * density
        self.mass_input.setText(f"{mass:.12f}")

        self.fit_camera_to_model()
        self.update_all_points()
        self.calculate_and_update_properties()

        print(f"STL model bounding box: [{self.model_bounds[0]:.6f}, {self.model_bounds[1]:.6f}], [{self.model_bounds[2]:.6f}, {self.model_bounds[3]:.6f}], [{self.model_bounds[4]:.6f}, {self.model_bounds[5]:.6f}]")

        self.show_absolute_origin()
        self.file_name_label.setText(f"File: {file_path}")
        self.reset_camera()

    def show_absolute_origin(self):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(0, 0, 0)
        sphere.SetRadius(0.0005)
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 0)

        self.renderer.AddActor(actor)
        self.render_to_image()

    def show_point(self, index):
        """Show point (also used for XML loading)"""
        if not self.point_checkboxes[index].isChecked():
            print(f"DEBUG: show_point({index}) - checkbox not checked")
            return

        if self.point_actors[index] is None:
            print(f"DEBUG: Creating new actor for Point {index+1}")
            self.point_actors[index] = create_crosshair_marker(
                coords=[0, 0, 0],
                radius_scale=self.calculate_sphere_radius()
            )
            self.renderer.AddActor(self.point_actors[index])
            print(f"DEBUG: Actor created and added for Point {index+1}")

        self.point_actors[index].SetPosition(self.point_coords[index])
        self.point_actors[index].VisibilityOn()
        self.update_point_display(index)
        print(f"DEBUG: Point {index+1} shown at {self.point_coords[index]}")

    def rotate_camera(self, angle, rotation_type):
        # Don't start new animation if one is already running
        if self.is_animating:
            return

        if self.animated_rotation.start_rotation(angle, rotation_type):
            self.target_rotation = (self.current_rotation + angle) % 360
            self.animation_frames = 0
            self.current_rotation_type = self.rotation_types[rotation_type]
            self.target_angle = angle  # Store target angle for precise completion
            self.is_animating = True  # Block further input
            self.animation_timer.start(1000 // 60)
            self.camera_rotation[self.rotation_types[rotation_type]] += angle
            self.camera_rotation[self.rotation_types[rotation_type]] %= 360

    def animate_rotation(self):
        self.animation_frames += 1

        # Delegate to utility class for rotation
        animation_continues = self.animated_rotation.animate_frame()

        # Render using offscreen rendering
        self.render_to_image()

        # Stop animation when utility class indicates completion
        if not animation_continues:
            self.animation_timer.stop()
            self.current_rotation = self.target_rotation
            self.is_animating = False  # Allow new input

    def toggle_point(self, state, index):
        """Toggle point visibility"""
        print(f"DEBUG: toggle_point({state}, {index}) - Point {index+1}")
        if state == Qt.CheckState.Checked.value:
            print(f"DEBUG: Checked - creating/showing Point {index+1}")
            if self.point_actors[index] is None:
                print(f"DEBUG: Creating new crosshair marker for Point {index+1}")
                self.point_actors[index] = create_crosshair_marker(
                    coords=[0, 0, 0],
                    radius_scale=self.calculate_sphere_radius()
                )
                self.renderer.AddActor(self.point_actors[index])
                print(f"DEBUG: Crosshair marker created and added for Point {index+1}")
            self.point_actors[index].SetPosition(self.point_coords[index])
            # Apply rotation from saved angles
            self.apply_marker_rotation(index)
            self.point_actors[index].VisibilityOn()
            self.renderer.AddActor(self.point_actors[index])
            print(f"DEBUG: Point {index+1} positioned at {self.point_coords[index]}")

            # Update Angle UI fields with the selected point's angles (degrees for UI)
            self.angle_x_input.setText(f"{math.degrees(self.point_angles[index][0]):.2f}")
            self.angle_y_input.setText(f"{math.degrees(self.point_angles[index][1]):.2f}")
            self.angle_z_input.setText(f"{math.degrees(self.point_angles[index][2]):.2f}")
            print(f"DEBUG: Updated Angle UI to Point {index+1}: {self.point_angles[index]} (rad)")
        else:
            print(f"DEBUG: Unchecked - hiding Point {index+1}")
            if self.point_actors[index]:
                self.point_actors[index].VisibilityOff()
                self.renderer.RemoveActor(self.point_actors[index])

        self.render_to_image()

    def toggle_com(self, state):
        """Toggle Center of Mass visibility"""
        if state == Qt.CheckState.Checked.value:
            # When checked: hide red sphere, show cross-circle (red)
            if self.com_sphere_actor:
                self.com_sphere_actor.VisibilityOff()

            if self.com_cursor_actor is None:
                self.com_cursor_actor = vtk.vtkAssembly()
                origin = [0, 0, 0]
                axis_length = self.calculate_sphere_radius() * 36
                circle_radius = self.calculate_sphere_radius()

                colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
                for i, color in enumerate(colors):
                    for direction in [1, -1]:
                        line_source = vtk.vtkLineSource()
                        line_source.SetPoint1(
                            origin[0] - (axis_length / 2) * (i == 0) * direction,
                            origin[1] - (axis_length / 2) * (i == 1) * direction,
                            origin[2] - (axis_length / 2) * (i == 2) * direction
                        )
                        line_source.SetPoint2(
                            origin[0] + (axis_length / 2) * (i == 0) * direction,
                            origin[1] + (axis_length / 2) * (i == 1) * direction,
                            origin[2] + (axis_length / 2) * (i == 2) * direction
                        )
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(line_source.GetOutputPort())
                        actor = vtk.vtkActor()
                        actor.SetMapper(mapper)
                        actor.GetProperty().SetColor(color)
                        actor.GetProperty().SetLineWidth(2)
                        self.com_cursor_actor.AddPart(actor)

                for i in range(3):
                    circle = vtk.vtkRegularPolygonSource()
                    circle.SetNumberOfSides(50)
                    circle.SetRadius(circle_radius)
                    circle.SetCenter(origin[0], origin[1], origin[2])
                    if i == 0:
                        circle.SetNormal(0, 0, 1)
                    elif i == 1:
                        circle.SetNormal(0, 1, 0)
                    else:
                        circle.SetNormal(1, 0, 0)

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(circle.GetOutputPort())
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(1, 0, 0)
                    actor.GetProperty().SetRepresentationToWireframe()
                    actor.GetProperty().SetLineWidth(6)
                    actor.GetProperty().SetOpacity(0.7)
                    self.com_cursor_actor.AddPart(actor)

                self.renderer.AddActor(self.com_cursor_actor)
                print("DEBUG: Center of Mass red marker created")

            self.com_cursor_actor.SetPosition(self.com_coords)
            self.com_cursor_actor.VisibilityOn()
            self.renderer.AddActor(self.com_cursor_actor)
            print(f"DEBUG: Center of Mass marker shown at {self.com_coords}")
        else:
            if self.com_cursor_actor:
                self.com_cursor_actor.VisibilityOff()
                self.renderer.RemoveActor(self.com_cursor_actor)

            try:
                self.com_coords = [float(self.com_inputs[i].text()) for i in range(3)]
                print(f"Updated Center of Mass from input: {self.com_coords}")
            except (ValueError, IndexError) as e:
                print(f"Error reading Center of Mass input: {e}")

            if self.com_sphere_actor:
                self.renderer.RemoveActor(self.com_sphere_actor)

                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(self.com_coords)
                sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())

                self.com_sphere_actor.SetMapper(mapper)
                self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)
                self.com_sphere_actor.GetProperty().SetOpacity(0.7)

                self.renderer.AddActor(self.com_sphere_actor)
                self.com_sphere_actor.VisibilityOn()

        self.render_to_image()

    def save_current_point_angles(self):
        """Save angle values for currently selected point"""
        selected_index = None
        for i, checkbox in enumerate(self.point_checkboxes):
            if checkbox.isChecked():
                selected_index = i
                break

        if selected_index is None:
            print("DEBUG: No point selected, cannot save angles")
            return

        try:
            angle_x = float(self.angle_x_input.text()) if self.angle_x_input.text() else 0.0
            angle_y = float(self.angle_y_input.text()) if self.angle_y_input.text() else 0.0
            angle_z = float(self.angle_z_input.text()) if self.angle_z_input.text() else 0.0

            self.point_angles[selected_index] = [
                math.radians(angle_x),
                math.radians(angle_y),
                math.radians(angle_z)
            ]
            print(f"DEBUG: Saved angles for Point {selected_index+1}: {self.point_angles[selected_index]} (rad)")

            if self.point_actors[selected_index]:
                self.apply_marker_rotation(selected_index)
                self.render_to_image()
        except ValueError as e:
            print(f"ERROR: Invalid angle value: {e}")

    def apply_marker_rotation(self, index):
        """Apply rotation to marker for specified point"""
        if not self.point_actors[index]:
            return

        angles = self.point_angles[index]
        angles_deg = [math.degrees(a) for a in angles]
        self.point_actors[index].SetOrientation(0, 0, 0)
        self.point_actors[index].RotateX(angles_deg[0])
        self.point_actors[index].RotateY(angles_deg[1])  # Y-axis rotation
        self.point_actors[index].RotateZ(angles_deg[2])  # Z-axis rotation
        print(f"DEBUG: Applied rotation to Point {index+1} marker: {angles_deg} (deg)")

    def create_com_coordinate(self, assembly, coords):
        """Create cross-circle for Center of Mass (red color)"""
        origin = coords
        axis_length = self.calculate_sphere_radius() * 36  # Use 18x diameter (6x * 3) as axis length
        circle_radius = self.calculate_sphere_radius()

        # Create XYZ axes
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, green, blue
        for i, color in enumerate(colors):
            for direction in [1, -1]:  # Both positive and negative directions
                line_source = vtk.vtkLineSource()
                line_source.SetPoint1(
                    origin[0] - (axis_length / 2) * (i == 0) * direction,
                    origin[1] - (axis_length / 2) * (i == 1) * direction,
                    origin[2] - (axis_length / 2) * (i == 2) * direction
                )
                line_source.SetPoint2(
                    origin[0] + (axis_length / 2) * (i == 0) * direction,
                    origin[1] + (axis_length / 2) * (i == 1) * direction,
                    origin[2] + (axis_length / 2) * (i == 2) * direction
                )

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(line_source.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetLineWidth(2)

                assembly.AddPart(actor)

        # Create circles on XY, XZ, YZ planes (red color)
        for i in range(3):
            circle = vtk.vtkRegularPolygonSource()
            circle.SetNumberOfSides(50)
            circle.SetRadius(circle_radius)
            circle.SetCenter(origin[0], origin[1], origin[2])
            if i == 0:  # XY plane
                circle.SetNormal(0, 0, 1)
            elif i == 1:  # XZ plane
                circle.SetNormal(0, 1, 0)
            else:  # YZ plane
                circle.SetNormal(1, 0, 0)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(circle.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Set circle properties (red color)
            actor.GetProperty().SetColor(1, 0, 0)  # Red
            actor.GetProperty().SetRepresentationToWireframe()  # Always wireframe display
            actor.GetProperty().SetLineWidth(6)  # Line width set to 6
            actor.GetProperty().SetOpacity(0.7)  # Reduce opacity for better visibility

            # Set UserTransform for tagging
            transform = vtk.vtkTransform()
            actor.SetUserTransform(transform)

            assembly.AddPart(actor)

    def get_axis_length(self):
        if self.model_bounds:
            size = max([
                self.model_bounds[1] - self.model_bounds[0],
                self.model_bounds[3] - self.model_bounds[2],
                self.model_bounds[5] - self.model_bounds[4]
            ])
            return size * 0.5
        else:
            return 5  # Default length

    def move_com_screen(self, direction, step):
        """Move Center of Mass in screen coordinates"""
        move_vector = direction * step
        new_position = [
            self.com_coords[0] + move_vector[0],
            self.com_coords[1] + move_vector[1],
            self.com_coords[2] + move_vector[2]
        ]
        self.com_coords = new_position
        self.update_com_display()
        print(f"Center of Mass moved to: ({new_position[0]:.6f}, {new_position[1]:.6f}, {new_position[2]:.6f})")

    def update_com_display(self):
        """Update Center of Mass display"""
        # Update input fields
        for i in range(3):
            self.com_inputs[i].setText(f"{self.com_coords[i]:.6f}")

        # Update cursor actor position
        if self.com_cursor_actor:
            self.com_cursor_actor.SetPosition(self.com_coords)

        self.render_to_image()

    def on_com_input_return(self):
        """Handle Return key press in Center of Mass input field"""
        try:
            # Read values from input fields
            self.com_coords = [float(self.com_inputs[i].text()) for i in range(3)]
            print(f"Center of Mass updated from input: {self.com_coords}")

            # Update display based on checkbox state
            if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
                # When checked: update cursor actor position
                if self.com_cursor_actor:
                    self.com_cursor_actor.SetPosition(self.com_coords)
            else:
                # When unchecked: update red sphere position
                if self.com_sphere_actor:
                    # Remove existing red sphere
                    self.renderer.RemoveActor(self.com_sphere_actor)

                    # Create red sphere at new position
                    sphere = vtk.vtkSphereSource()
                    sphere.SetCenter(self.com_coords)
                    sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(sphere.GetOutputPort())

                    self.com_sphere_actor.SetMapper(mapper)
                    self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red
                    self.com_sphere_actor.GetProperty().SetOpacity(0.7)

                    # Re-add to renderer
                    self.renderer.AddActor(self.com_sphere_actor)
                    self.com_sphere_actor.VisibilityOn()

            # Update display
            self.render_to_image()

        except (ValueError, IndexError) as e:
            print(f"Error parsing Center of Mass input: {e}")

    def apply_volume_value(self):
        """Handle Return key press in Volume input field"""
        try:
            volume_value = float(self.volume_input.text())
            print(f"Volume value applied: {volume_value:.6f}")
            # Value is applied and stored as valid
            # Trigger calculation if needed
            if self.volume_checkbox.isChecked():
                # When checked, value is treated as fixed
                pass
        except ValueError:
            print(f"Error: Invalid volume input. Please enter a valid number.")

    def apply_density_value(self):
        """Handle Return key press in Density input field"""
        try:
            density_value = float(self.density_input.text())
            print(f"Density value applied: {density_value:.6f}")
            # Value is applied and stored as valid
            # Trigger calculation if needed
            if self.density_checkbox.isChecked():
                # When checked, value is treated as fixed
                pass
        except ValueError:
            print(f"Error: Invalid density input. Please enter a valid number.")

    def apply_mass_value(self):
        """Handle Return key press in Mass input field"""
        try:
            mass_value = float(self.mass_input.text())
            print(f"Mass value applied: {mass_value:.6f}")
            # Value is applied and stored as valid
            # Trigger calculation if needed
            if self.mass_checkbox.isChecked():
                # When checked, value is treated as fixed
                pass
        except ValueError:
            print(f"Error: Invalid mass input. Please enter a valid number.")

    def fit_camera_to_model(self):
        """Adjust camera distance so STL model fits on screen"""
        if not self.model_bounds:
            return

        camera = self.renderer.GetActiveCamera()

        # Calculate model center
        center = [(self.model_bounds[i] + self.model_bounds[i+1]) / 2 for i in range(0, 6, 2)]

        # Calculate model size
        size = max([
            self.model_bounds[1] - self.model_bounds[0],
            self.model_bounds[3] - self.model_bounds[2],
            self.model_bounds[5] - self.model_bounds[4]
        ])

        # Add 20% margin
        size *= 1.4  # 1.0 + 0.2 + 0.2 = 1.4

        # Preserve current camera direction vector
        current_position = np.array(camera.GetPosition())
        focal_point = np.array(center)  # Set model center as focal point
        direction = current_position - focal_point

        # Normalize direction vector
        direction = direction / np.linalg.norm(direction)

        # Calculate new position (preserve direction, adjust distance only)
        new_position = focal_point + direction * size

        # Update camera position (keep direction unchanged)
        camera.SetPosition(new_position)
        camera.SetFocalPoint(*center)  # Look at model center

        # Get viewport aspect ratio
        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / (viewport[3] - viewport[1])

        # Set parallel scale so model fits on screen
        if aspect_ratio > 1:  # Wide screen
            camera.SetParallelScale(size / 2)
        else:  # Tall screen
            camera.SetParallelScale(size / (2 * aspect_ratio))

        self.renderer.ResetCameraClippingRange()
        self.render_to_image()

    def handle_close(self, event):
        print("Window is closing...")
        self.vtk_widget.GetRenderWindow().Finalize()
        self.vtk_widget.close()
        event.accept()

    def get_screen_axes(self):
        camera = self.renderer.GetActiveCamera()
        view_up = np.array(camera.GetViewUp())
        forward = np.array(camera.GetDirectionOfProjection())
        
        # Use NumPy vector operations
        right = np.cross(forward, view_up)
        
        screen_right = right
        screen_up = view_up

        # Use NumPy for dot product calculation
        if abs(np.dot(screen_right, [1, 0, 0])) > abs(np.dot(screen_right, [0, 0, 1])):
            horizontal_axis = 'x'
            vertical_axis = 'z' if abs(np.dot(screen_up, [0, 0, 1])) > abs(np.dot(screen_up, [0, 1, 0])) else 'y'
        else:
            horizontal_axis = 'z'
            vertical_axis = 'y'

        return horizontal_axis, vertical_axis, screen_right, screen_up

    def handle_reset_only(self):
        """Reset checked points to origin"""
        for i, checkbox in enumerate(self.point_checkboxes):
            if checkbox.isChecked():
                self.reset_point_to_origin(i)

        self.update_all_points_size()
        self.render_to_image()

    def export_mirror_stl_xml(self):
        """Mirror 3D model file on Y-axis and generate corresponding XML file (STL/DAE support)"""
        if not hasattr(self, 'stl_file_path') or not self.stl_file_path:
            print("No 3D model file has been loaded.")
            return

        try:
            # Get original file path and filename
            original_dir = os.path.dirname(self.stl_file_path)
            original_filename = os.path.basename(self.stl_file_path)
            name, ext = os.path.splitext(original_filename)

            # Generate new filename (flip L/R)
            if name.startswith('L_'):
                new_name = 'R_' + name[2:]
            elif name.startswith('l_'):
                new_name = 'r_' + name[2:]
            elif name.startswith('R_'):
                new_name = 'L_' + name[2:]
            elif name.startswith('r_'):
                new_name = 'l_' + name[2:]
            else:
                new_name = 'mirrored_' + name

            # Set mirrored file paths
            mirrored_stl_path = os.path.join(original_dir, new_name + ext)
            mirrored_xml_path = os.path.join(original_dir, new_name + '.xml')

            # Check existing files and show dialog
            if os.path.exists(mirrored_stl_path) or os.path.exists(mirrored_xml_path):
                existing_files = []
                if os.path.exists(mirrored_stl_path):
                    existing_files.append(f"Mesh: {mirrored_stl_path}")
                if os.path.exists(mirrored_xml_path):
                    existing_files.append(f"XML: {mirrored_xml_path}")
                
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setText("Following files already exist:")
                msg.setInformativeText("\n".join(existing_files) + "\n\nDo you want to overwrite them?")
                msg.setWindowTitle("Confirm Overwrite")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                
                if msg.exec() == QMessageBox.No:
                    print("Operation cancelled by user")
                    return

            print("\nStarting mirror export process...")
            print(f"Source file: {self.stl_file_path}")

            # Check file extension to determine loader
            file_ext = os.path.splitext(self.stl_file_path)[1].lower()

            if file_ext == '.dae':
                # Load COLLADA file using trimesh
                mesh = trimesh.load(self.stl_file_path, force='mesh')

                # Convert trimesh to VTK PolyData
                vertices = mesh.vertices
                faces = mesh.faces

                # Create VTK points
                points = vtk.vtkPoints()
                for vertex in vertices:
                    points.InsertNextPoint(vertex)

                # Create VTK cells (triangles)
                triangles = vtk.vtkCellArray()
                for face in faces:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, int(face[0]))
                    triangle.GetPointIds().SetId(1, int(face[1]))
                    triangle.GetPointIds().SetId(2, int(face[2]))
                    triangles.InsertNextCell(triangle)

                # Create VTK PolyData
                poly_data = vtk.vtkPolyData()
                poly_data.SetPoints(points)
                poly_data.SetPolys(triangles)

                # Create transform to flip on Y-axis
                transform = vtk.vtkTransform()
                transform.Scale(1, -1, 1)

                # Apply transform
                transform_filter = vtk.vtkTransformPolyDataFilter()
                transform_filter.SetInputData(poly_data)
                transform_filter.SetTransform(transform)
                transform_filter.Update()

            else:
                # Load STL file using VTK
                reader = vtk.vtkSTLReader()
                reader.SetFileName(self.stl_file_path)
                reader.Update()

                # Create transform to flip on Y-axis
                transform = vtk.vtkTransform()
                transform.Scale(1, -1, 1)

                # Apply transform
                transform_filter = vtk.vtkTransformPolyDataFilter()
                transform_filter.SetInputConnection(reader.GetOutputPort())
                transform_filter.SetTransform(transform)
                transform_filter.Update()

            # Fix normals
            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.SetInputData(transform_filter.GetOutput())
            normal_generator.ConsistencyOn()
            normal_generator.AutoOrientNormalsOn()
            normal_generator.ComputeCellNormalsOn()
            normal_generator.ComputePointNormalsOn()
            normal_generator.Update()

            # Check and load XML file
            xml_path = os.path.splitext(self.stl_file_path)[0] + '.xml'
            xml_data = None
            mass_value_str = "0.0"
            volume_value_str = "0.0"
            mass = 0.0
            volume = 0.0
            center_of_mass = [0.0, 0.0, 0.0]
            rgba_str = "1.0 1.0 1.0 1.0"
            hex_color = "#FFFFFF"
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    xml_data = tree.getroot()
                    print(f"Found and loaded XML file: {xml_path}")

                    # Get physical parameters from XML (preserve original format)
                    mass_element = xml_data.find(".//mass")
                    mass_value_str = mass_element.get('value') if mass_element is not None else "0.0"
                    mass = float(mass_value_str)  # For calculation

                    volume_element = xml_data.find(".//volume")
                    volume_value_str = volume_element.get('value') if volume_element is not None else "0.0"
                    volume = float(volume_value_str)  # For calculation

                    # Get center of mass position (from center_of_mass element)
                    com_element = xml_data.find(".//center_of_mass")
                    if com_element is not None and com_element.text:
                        x, y, z = map(float, com_element.text.strip().split())
                        center_of_mass = [x, -y, z]  # Flip Y coordinate only
                    else:
                        # Get from inertial origin element
                        inertial_origin = xml_data.find(".//inertial/origin")
                        if inertial_origin is not None:
                            xyz = inertial_origin.get('xyz')
                            x, y, z = map(float, xyz.split())
                            center_of_mass = [x, -y, z]  # Flip Y coordinate only
                        else:
                            print("Warning: No center of mass information found in XML")
                            center_of_mass = [0, 0, 0]

                    print(f"Original mass: {mass:.6f}, volume: {volume:.6f}")
                    print(f"Original center of mass: {center_of_mass}")

                    # Get color information
                    color_element = xml_data.find(".//material/color")
                    if color_element is not None:
                        rgba_str = color_element.get('rgba')
                        hex_color = xml_data.find(".//material").get('name')
                    else:
                        rgba_str = "1.0 1.0 1.0 1.0"
                        hex_color = "#FFFFFF"

                except ET.ParseError as e:
                    print(f"Error parsing XML file: {xml_path}")
                    print(f"Error details: {str(e)}")
                    return

            # Save mirrored file
            print(f"\nSaving mirrored file to: {mirrored_stl_path}")

            # Determine output format from extension
            output_ext = os.path.splitext(mirrored_stl_path)[1].lower()

            if output_ext == '.dae':
                # Export as COLLADA using trimesh
                num_points = normal_generator.GetOutput().GetNumberOfPoints()
                num_cells = normal_generator.GetOutput().GetNumberOfCells()

                vertices = np.zeros((num_points, 3))
                for i in range(num_points):
                    vertices[i] = normal_generator.GetOutput().GetPoint(i)

                faces = []
                for i in range(num_cells):
                    cell = normal_generator.GetOutput().GetCell(i)
                    if cell.GetNumberOfPoints() == 3:
                        faces.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)])

                faces = np.array(faces)

                # Create trimesh object and export
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                # Apply color to the mesh only if manually changed by user
                if hasattr(self, 'color_manually_changed') and self.color_manually_changed:
                    if hasattr(self, 'mesh_color') and self.mesh_color is not None:
                        # Convert color from 0-1 range to 0-255 range for trimesh
                        color_rgba = [
                            int(self.mesh_color[0] * 255),
                            int(self.mesh_color[1] * 255),
                            int(self.mesh_color[2] * 255),
                            int(self.mesh_color[3] * 255) if len(self.mesh_color) > 3 else 255
                        ]
                        # Create a single material for the entire mesh
                        mesh.visual = trimesh.visual.ColorVisuals(mesh)
                        mesh.visual.material = trimesh.visual.material.SimpleMaterial(
                            diffuse=color_rgba,
                            ambient=color_rgba,
                            specular=[50, 50, 50, 255]
                        )
                        print(f"Applied manually set color to mirrored .dae file: RGBA({self.mesh_color[0]:.3f}, {self.mesh_color[1]:.3f}, {self.mesh_color[2]:.3f}, {self.mesh_color[3] if len(self.mesh_color) > 3 else 1.0:.3f})")
                else:
                    print("Preserving original .dae file color in mirrored file (no manual color change)")

                mesh.export(mirrored_stl_path)

            else:
                # Export as STL using VTK
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(mirrored_stl_path)
                writer.SetInputData(normal_generator.GetOutput())
                writer.Write()

            # Get inertia tensor and apply mirror transformation
            print("\nProcessing inertia tensor for mirrored model...")
            inertia_element = None
            if xml_data is not None:
                inertia_element = xml_data.find(".//inertia")

            if inertia_element is not None:
                # Get inertia tensor from original XML and apply mirror transformation
                ixx = float(inertia_element.get('ixx', 0))
                iyy = float(inertia_element.get('iyy', 0))
                izz = float(inertia_element.get('izz', 0))
                ixy = float(inertia_element.get('ixy', 0))
                ixz = float(inertia_element.get('ixz', 0))
                iyz = float(inertia_element.get('iyz', 0))
                # For Y-axis mirror, flip sign of ixy and iyz
                inertia_str = f'ixx="{ixx:.12f}" ixy="{-ixy:.12f}" ixz="{ixz:.12f}" iyy="{iyy:.12f}" iyz="{-iyz:.12f}" izz="{izz:.12f}"'
                print(f"Mirrored inertia tensor from XML: ixx={ixx:.6f}, iyy={iyy:.6f}, izz={izz:.6f}, ixy={-ixy:.6f}, ixz={ixz:.6f}, iyz={-iyz:.6f}")
            else:
                # If original XML has no inertia data, calculate from mesh
                print("Warning: No inertia data in XML, calculating from mesh...")
                inertia_tensor = self.calculate_inertia_tensor_for_mirrored(
                    normal_generator.GetOutput(), mass, center_of_mass)
                # Remove <inertia and /> from format_inertia_for_urdf return value
                inertia_formatted = self.format_inertia_for_urdf(inertia_tensor)
                inertia_str = inertia_formatted.replace('<inertia ', '').replace('/>', '').strip()

            # Generate XML file content
            print(f"\nGenerating XML content...")
            urdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urdf_part>
    <material name="{hex_color}">
        <color rgba="{rgba_str}" />
    </material>
    <link name="{new_name}">
        <visual>
            <origin xyz="{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}" rpy="0 0 0"/>
            <material name="{hex_color}" />
        </visual>
        <inertial>
            <origin xyz="{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}"/>
            <mass value="{mass_value_str}"/>
            <volume value="{volume_value_str}"/>
            <inertia {inertia_str} />
        </inertial>
        <center_of_mass>{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}</center_of_mass>
    </link>"""

            # Flip and copy point data
            if xml_data is not None:
                print("Processing point data...")
                points = xml_data.findall('.//point')
                for point in points:
                    xyz_element = point.find('point_xyz')
                    if xyz_element is not None and xyz_element.text:
                        try:
                            x, y, z = map(float, xyz_element.text.strip().split())
                            mirrored_y = -y  # Flip Y coordinate only
                            point_name = point.get('name')

                            # Process point_angle (rotation transform for Y-axis mirror)
                            angle_element = point.find('point_angle')
                            point_angle_str = ""
                            if angle_element is not None and angle_element.text:
                                try:
                                    angle_x, angle_y, angle_z = map(float, angle_element.text.strip().split())
                                    # For Y-axis mirror, flip sign of X and Z axis rotations
                                    mirrored_angle_x = -angle_x
                                    mirrored_angle_y = angle_y
                                    mirrored_angle_z = -angle_z
                                    point_angle_str = f"\n        <point_angle>{mirrored_angle_x:.6f} {mirrored_angle_y:.6f} {mirrored_angle_z:.6f}</point_angle>"
                                    print(f"Mirrored point_angle for {point_name}: [{mirrored_angle_x:.2f}, {mirrored_angle_y:.2f}, {mirrored_angle_z:.2f}]")
                                except ValueError:
                                    print(f"Warning: Invalid point_angle format for {point_name}")

                            urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>{point_angle_str}
    </point>"""
                            print(f"Processed point: {point_name}")
                        except ValueError:
                            print(f"Error processing point coordinates in XML")

            # Get and apply axis information
            if xml_data is not None:
                print("Processing axis information...")
                axis_element = xml_data.find('.//joint/axis')
                if axis_element is not None:
                    axis_str = axis_element.get('xyz')
                    mirrored_axis = self.mirror_axis_value(axis_str)
                else:
                    mirrored_axis = "1 0 0"
            else:
                mirrored_axis = "1 0 0"

            # Get joint angle limits (swap for roll/yaw axis)
            joint_limit_str = ""
            if xml_data is not None:
                limit_element = xml_data.find('.//joint/limit')
                if limit_element is not None:
                    lower = limit_element.get('lower')
                    upper = limit_element.get('upper')
                    effort = limit_element.get('effort')
                    velocity = limit_element.get('velocity')

                    if lower is not None and upper is not None:
                        # Determine axis type (1 0 0 = Roll, 0 1 0 = Pitch, 0 0 1 = Yaw)
                        axis_xyz = [float(x) for x in mirrored_axis.split()]
                        is_roll = abs(axis_xyz[0]) > 0.5  # Large X component = Roll
                        is_yaw = abs(axis_xyz[2]) > 0.5   # Large Z component = Yaw

                        if is_roll or is_yaw:
                            # For Roll/Yaw axis, swap min/max and negate
                            # Example: lower=-10, upper=190 -> lower=-190, upper=10
                            lower_val = float(lower)
                            upper_val = float(upper)
                            lower = str(-upper_val)
                            upper = str(-lower_val)
                            print(f"Swapped and negated joint limits for {'Roll' if is_roll else 'Yaw'} axis: lower={lower}, upper={upper}")

                        joint_limit_str = f'\n        <limit lower="{lower}" upper="{upper}"'
                        if effort is not None:
                            joint_limit_str += f' effort="{effort}"'
                        if velocity is not None:
                            joint_limit_str += f' velocity="{velocity}"'
                        joint_limit_str += ' />'

            urdf_content += f"""
    <joint>
        <axis xyz="{mirrored_axis}" />{joint_limit_str}
    </joint>
</urdf_part>"""

            # Save XML file
            print(f"Saving XML to: {mirrored_xml_path}")
            with open(mirrored_xml_path, "w") as f:
                f.write(urdf_content)

            # Process corresponding collider XML file
            # Generate collider XML path from original filename
            original_name_base = os.path.splitext(original_filename)[0]
            collider_xml_path = os.path.join(original_dir, original_name_base + '_collider.xml')
            
            if os.path.exists(collider_xml_path):
                print(f"\nFound collider XML: {collider_xml_path}")
                try:
                    # Generate new collider XML filename
                    new_collider_xml_name = new_name + '_collider.xml'
                    new_collider_xml_path = os.path.join(original_dir, new_collider_xml_name)

                    # Load collider XML file
                    tree = ET.parse(collider_xml_path)
                    root = tree.getroot()
                    
                    if root.tag != 'urdf_kitchen_collider':
                        print(f"  ⚠ Warning: Invalid collider XML format (expected 'urdf_kitchen_collider'), skipping")
                    else:
                        collider_elem = root.find('collider')
                        if collider_elem is None:
                            print(f"  ⚠ Warning: No collider element found, skipping")
                        else:
                            # Get collider type
                            collider_type = collider_elem.get('type', 'box')

                            # Get geometry information
                            geometry_elem = collider_elem.find('geometry')
                            geometry_attrs = {}
                            if geometry_elem is not None:
                                geometry_attrs = dict(geometry_elem.attrib)

                            # Get position and flip on XZ plane (flip Y coordinate)
                            position_elem = collider_elem.find('position')
                            if position_elem is not None:
                                x = float(position_elem.get('x', '0.0'))
                                y = float(position_elem.get('y', '0.0'))
                                z = float(position_elem.get('z', '0.0'))
                                mirrored_y = -y  # Flip Y coordinate
                            else:
                                x, y, z = 0.0, 0.0, 0.0
                                mirrored_y = 0.0

                            # Get rotation (flip Roll and Yaw, keep Pitch)
                            rotation_elem = collider_elem.find('rotation')
                            if rotation_elem is not None:
                                roll = float(rotation_elem.get('roll', '0.0'))
                                pitch = float(rotation_elem.get('pitch', '0.0'))
                                yaw = float(rotation_elem.get('yaw', '0.0'))
                                # Flip Roll and Yaw, keep Pitch
                                mirrored_roll = -roll
                                mirrored_pitch = pitch  # Don't flip Pitch
                                mirrored_yaw = -yaw
                            else:
                                roll, pitch, yaw = 0.0, 0.0, 0.0
                                mirrored_roll = 0.0
                                mirrored_pitch = 0.0
                                mirrored_yaw = 0.0
                            
                            # Create new collider XML file
                            new_root = ET.Element('urdf_kitchen_collider')
                            new_collider_elem = ET.SubElement(new_root, 'collider')
                            new_collider_elem.set('type', collider_type)

                            # Add geometry element
                            if geometry_attrs:
                                new_geometry_elem = ET.SubElement(new_collider_elem, 'geometry')
                                for key, value in geometry_attrs.items():
                                    new_geometry_elem.set(key, value)

                            # Add position element (flip Y coordinate)
                            new_position_elem = ET.SubElement(new_collider_elem, 'position')
                            new_position_elem.set('x', f"{x:.6f}")
                            new_position_elem.set('y', f"{mirrored_y:.6f}")
                            new_position_elem.set('z', f"{z:.6f}")

                            # Add rotation element (flip Roll and Yaw, keep Pitch)
                            new_rotation_elem = ET.SubElement(new_collider_elem, 'rotation')
                            new_rotation_elem.set('roll', f"{mirrored_roll:.6f}")
                            new_rotation_elem.set('pitch', f"{mirrored_pitch:.6f}")
                            new_rotation_elem.set('yaw', f"{mirrored_yaw:.6f}")
                            print(f"  Mirrored rotation: Roll={roll:.4f}->{mirrored_roll:.4f}, Pitch={pitch:.4f}->{mirrored_pitch:.4f}, Yaw={yaw:.4f}->{mirrored_yaw:.4f}")

                            # Copy mesh_file element if exists (replace l_ with r_)
                            mesh_file_elem = collider_elem.find('mesh_file')
                            if mesh_file_elem is not None and mesh_file_elem.text:
                                new_mesh_file_elem = ET.SubElement(new_collider_elem, 'mesh_file')
                                original_mesh_name = mesh_file_elem.text
                                if 'l_' in original_mesh_name.lower():
                                    mirrored_mesh_name = 'r_' + original_mesh_name[2:] if original_mesh_name.startswith('l_') else 'R_' + original_mesh_name[2:]
                                    new_mesh_file_elem.text = mirrored_mesh_name
                                    print(f"  Mirrored mesh_file reference: {original_mesh_name} -> {mirrored_mesh_name}")
                                else:
                                    new_mesh_file_elem.text = original_mesh_name
                            
                            # Save XML file
                            new_tree = ET.ElementTree(new_root)
                            ET.indent(new_tree, space="    ")
                            new_tree.write(new_collider_xml_path, encoding='utf-8', xml_declaration=True)
                            print(f"  ✓ Created collider XML: {new_collider_xml_name}")
                except Exception as e:
                    print(f"  ✗ Error processing collider XML: {str(e)}")
                    traceback.print_exc()
            else:
                print(f"\nNo collider XML found: {collider_xml_path}")

            print("\nMirror export completed successfully:")
            file_type = "DAE file" if output_ext == '.dae' else "STL file"
            print(f"{file_type}: {mirrored_stl_path}")
            print(f"XML file: {mirrored_xml_path}")

            # Show output contents in dialog box
            dialog = ResultDialog(mirrored_stl_path, mirrored_xml_path, self)
            dialog.exec()

        except Exception as e:
            print(f"\nAn error occurred during mirror export: {str(e)}")
            traceback.print_exc()
            # Show error dialog
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during mirror export:\n{str(e)}"
            )
        

    def calculate_inertia_tensor_for_mirrored(self, poly_data, mass, center_of_mass):
        """
        Calculate inertia tensor for mirrored model

        Args:
            poly_data: vtkPolyData object
            mass: float mass value
            center_of_mass: list[float] center of mass coordinates [x, y, z]

        Returns:
            numpy.ndarray: 3x3 inertia tensor matrix
        """
        # Use shared triangle-based method with mirroring from urdf_kitchen_utils
        inertia_tensor = calculate_inertia_tensor(
            poly_data, mass, center_of_mass, is_mirrored=True
        )

        print("\nCalculated Inertia Tensor:")
        print(inertia_tensor)
        return inertia_tensor



    def _load_points_from_xml(self, root):
        """Load point data from XML"""
        points_with_data = set()
        # Use './/point' to search for point tags at any level
        points = root.findall('.//point')
        print(f"Found {len(points)} points in XML")

        for i, point in enumerate(points):
            if i >= len(self.point_checkboxes):  # Array bounds check
                break

            xyz_element = point.find('point_xyz')
            if xyz_element is not None and xyz_element.text:
                try:
                    x, y, z = map(float, xyz_element.text.strip().split())
                    print(f"Loading point {i+1}: ({x}, {y}, {z})")

                    # Set coordinates
                    self.point_inputs[i][0].setText(f"{x:.6f}")
                    self.point_inputs[i][1].setText(f"{y:.6f}")
                    self.point_inputs[i][2].setText(f"{z:.6f}")
                    self.point_coords[i] = [x, y, z]
                    points_with_data.add(i)

                    # Load point_angle (in radians)
                    angle_element = point.find('point_angle')
                    if angle_element is not None and angle_element.text:
                        try:
                            angle_x, angle_y, angle_z = map(float, angle_element.text.strip().split())
                            # Backward compatibility: if values look like degrees, convert to radians
                            if any(abs(v) > 3.5 for v in [angle_x, angle_y, angle_z]):
                                angle_x = math.radians(angle_x)
                                angle_y = math.radians(angle_y)
                                angle_z = math.radians(angle_z)
                            self.point_angles[i] = [angle_x, angle_y, angle_z]
                            print(f"Loaded point_angle for point {i+1}: ({angle_x}, {angle_y}, {angle_z}) [rad]")
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Invalid point_angle format for point {i+1}, using default [0, 0, 0]: {e}")
                            self.point_angles[i] = [0.0, 0.0, 0.0]
                    else:
                        self.point_angles[i] = [0.0, 0.0, 0.0]
                        print(f"No point_angle found for point {i+1}, using default [0, 0, 0]")

                    # Turn on checkbox
                    self.point_checkboxes[i].setChecked(True)

                    # Set point display
                    if self.point_actors[i] is None:
                        self.point_actors[i] = create_crosshair_marker(
                            coords=[0, 0, 0],
                            radius_scale=self.calculate_sphere_radius()
                        )

                    self.point_actors[i].SetPosition(self.point_coords[i])
                    # Apply rotation from loaded angles
                    self.apply_marker_rotation(i)
                    self.renderer.AddActor(self.point_actors[i])
                    self.point_actors[i].VisibilityOn()

                    print(f"Successfully loaded and visualized point {i+1}")

                except (ValueError, IndexError) as e:
                    print(f"Error processing point {i+1}: {e}")
                    continue

        if not points_with_data:
            print("No valid points found in XML")
        else:
            print(f"Successfully loaded {len(points_with_data)} points")

            # Update UI with Angle values of first checked point
            first_point_index = min(points_with_data)
            self.angle_x_input.setText(f"{self.point_angles[first_point_index][0]:.2f}")
            self.angle_y_input.setText(f"{self.point_angles[first_point_index][1]:.2f}")
            self.angle_z_input.setText(f"{self.point_angles[first_point_index][2]:.2f}")
            print(f"Updated Angle UI to Point {first_point_index+1}: {self.point_angles[first_point_index]}")

        return points_with_data

    def _apply_color_from_xml(self, root):
        """Apply color information from XML"""
        color_element = root.find(".//material/color")
        if color_element is not None:
            rgba_str = color_element.get('rgba')
            if rgba_str:
                try:
                    r, g, b, a = map(float, rgba_str.split())

                    self.color_inputs[0].setText(f"{r:.3f}")
                    self.color_inputs[1].setText(f"{g:.3f}")
                    self.color_inputs[2].setText(f"{b:.3f}")
                    if len(self.color_inputs) >= 4:
                        self.color_inputs[3].setText(f"{a:.3f}")

                    if self.stl_actor:
                        self.stl_actor.GetProperty().SetColor(r, g, b)
                        self.stl_actor.GetProperty().SetOpacity(a)
                        self.render_to_image()

                    print(f"Material color loaded and applied: RGBA({r:.3f}, {g:.3f}, {b:.3f}, {a:.3f})")
                except (ValueError, IndexError) as e:
                    print(f"Warning: Invalid color format in XML: {rgba_str}")
                    print(f"Error details: {e}")

    def _refresh_display(self):
        """Refresh display"""
        self.renderer.ResetCamera()
        self.fit_camera_to_model()
        self.update_all_points_size()
        self.update_all_points()
        self.calculate_center_of_mass()
        self.add_axes()
        self.renderer.ResetCameraClippingRange()
        self.render_to_image()

    def load_parameters_from_xml(self, root):
        """Common process to load and set parameters from XML"""
        try:
            # First reset all points
            for i in range(self.num_points):
                # Reset coordinates to 0
                self.point_coords[i] = [0, 0, 0]
                for j in range(3):
                    self.point_inputs[i][j].setText("0.000000")
                # Uncheck checkbox
                self.point_checkboxes[i].setChecked(False)
                # Remove existing actor
                if self.point_actors[i]:
                    self.point_actors[i].VisibilityOff()
                    self.renderer.RemoveActor(self.point_actors[i])
                    self.point_actors[i] = None

            has_parameters = False

            # Load color information
            material_element = root.find(".//material")
            if material_element is not None:
                color_element = material_element.find("color")
                if color_element is not None:
                    rgba_str = color_element.get('rgba')
                    if rgba_str:
                        try:
                            r, g, b, a = map(float, rgba_str.split())
                            # Set color information to input fields
                            self.color_inputs[0].setText(f"{r:.3f}")
                            self.color_inputs[1].setText(f"{g:.3f}")
                            self.color_inputs[2].setText(f"{b:.3f}")
                            if len(self.color_inputs) >= 4:
                                self.color_inputs[3].setText(f"{a:.3f}")

                            # Apply color to STL model
                            if hasattr(self, 'stl_actor') and self.stl_actor:
                                self.stl_actor.GetProperty().SetColor(r, g, b)
                                self.stl_actor.GetProperty().SetOpacity(a)
                                self.render_to_image()

                            has_parameters = True
                            print(f"Loaded color: RGBA({r:.3f}, {g:.3f}, {b:.3f}, {a:.3f})")
                        except ValueError as e:
                            print(f"Error parsing color values: {e}")

            # Load axis information
            joint_element = root.find(".//joint/axis")
            if joint_element is not None:
                axis_str = joint_element.get('xyz')
                if axis_str:
                    try:
                        x, y, z = map(float, axis_str.split())
                        # Select corresponding radio button
                        if x == 1:
                            self.radio_buttons[0].setChecked(True)
                            print("Set axis to X (roll)")
                        elif y == 1:
                            self.radio_buttons[1].setChecked(True)
                            print("Set axis to Y (pitch)")
                        elif z == 1:
                            self.radio_buttons[2].setChecked(True)
                            print("Set axis to Z (yaw)")
                        else:
                            self.radio_buttons[3].setChecked(True)  # fixed
                            print("Set axis to fixed")
                        has_parameters = True
                    except ValueError as e:
                        print(f"Error parsing axis values: {e}")

            # Get and set volume
            volume_element = root.find(".//volume")
            if volume_element is not None:
                volume = volume_element.get('value')
                self.volume_input.setText(volume)
                self.volume_checkbox.setChecked(True)
                has_parameters = True

            # Get and set mass
            mass_element = root.find(".//mass")
            if mass_element is not None:
                mass = mass_element.get('value')
                self.mass_input.setText(mass)
                self.mass_checkbox.setChecked(True)
                has_parameters = True

            # Get and set center of mass (with priority)
            com_str = None

            # First check <center_of_mass> tag
            com_element = root.find(".//center_of_mass")
            if com_element is not None and com_element.text:
                com_str = com_element.text.strip()

            # Next check inertial origin element
            if com_str is None:
                inertial_origin = root.find(".//inertial/origin")
                if inertial_origin is not None:
                    xyz = inertial_origin.get('xyz')
                    if xyz:
                        com_str = xyz

            # Finally check visual origin element
            if com_str is None:
                visual_origin = root.find(".//visual/origin")
                if visual_origin is not None:
                    xyz = visual_origin.get('xyz')
                    if xyz:
                        com_str = xyz

            # Set center of mass value
            if com_str:
                try:
                    x, y, z = map(float, com_str.split())
                    self.com_inputs[0].setText(f"{x:.6f}")
                    self.com_inputs[1].setText(f"{y:.6f}")
                    self.com_inputs[2].setText(f"{z:.6f}")
                    self.com_coords = [x, y, z]
                    print(f"Loaded center of mass: ({x:.6f}, {y:.6f}, {z:.6f})")

                    # Update Center of Mass red sphere
                    if hasattr(self, 'renderer'):
                        # Remove existing red sphere
                        if hasattr(self, 'com_sphere_actor') and self.com_sphere_actor:
                            self.renderer.RemoveActor(self.com_sphere_actor)

                        # Create red sphere at new position
                        sphere = vtk.vtkSphereSource()
                        sphere.SetCenter(self.com_coords)
                        sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(sphere.GetOutputPort())

                        self.com_sphere_actor = vtk.vtkActor()
                        self.com_sphere_actor.SetMapper(mapper)
                        self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red
                        self.com_sphere_actor.GetProperty().SetOpacity(0.7)

                        # Check state to set visibility
                        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
                            self.com_sphere_actor.VisibilityOff()
                        else:
                            self.com_sphere_actor.VisibilityOn()

                        self.renderer.AddActor(self.com_sphere_actor)
                        print(f"Center of Mass sphere updated at: {self.com_coords}")

                    has_parameters = True
                except ValueError as e:
                    print(f"Error parsing center of mass values: {e}")

            # Set inertia tensor
            inertia_element = root.find(".//inertia")
            if inertia_element is not None:
                inertia_str = ET.tostring(inertia_element, encoding='unicode')
                self.inertia_tensor_input.setText(inertia_str)
                has_parameters = True

            # Load point data
            points = root.findall(".//point")
            for i, point in enumerate(points):
                if i >= len(self.point_checkboxes):
                    break

                xyz_element = point.find("point_xyz")
                if xyz_element is not None and xyz_element.text:
                    try:
                        x, y, z = map(float, xyz_element.text.strip().split())
                        # Set coordinate values
                        self.point_inputs[i][0].setText(f"{x:.6f}")
                        self.point_inputs[i][1].setText(f"{y:.6f}")
                        self.point_inputs[i][2].setText(f"{z:.6f}")
                        self.point_coords[i] = [x, y, z]

                        # Turn on checkbox
                        self.point_checkboxes[i].setChecked(True)

                        # Display point
                        if self.point_actors[i] is None:
                            self.point_actors[i] = vtk.vtkAssembly()
                            self.create_point_coordinate(self.point_actors[i], [0, 0, 0])
                        self.point_actors[i].SetPosition(self.point_coords[i])
                        self.renderer.AddActor(self.point_actors[i])
                        self.point_actors[i].VisibilityOn()
                        
                        print(f"Loaded point {i+1}: ({x:.6f}, {y:.6f}, {z:.6f})")
                        has_parameters = True
                    except ValueError as e:
                        print(f"Error parsing point {i+1} coordinates: {e}")

            # Reset camera
            self.reset_camera()

            return has_parameters

        except Exception as e:
            print(f"Error loading parameters: {str(e)}")
            traceback.print_exc()
            return False

    def load_xml_file(self):
        """Load XML file only and apply parameters"""
        try:
            xml_path, _ = QFileDialog.getOpenFileName(self, "Open XML File", "", "XML Files (*.xml)")
            if not xml_path:
                return

            # Save XML file path
            self.xml_file_path = xml_path

            # Parse XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            print("Processing XML file...")

            # Load parameters from XML
            has_parameters = self.load_parameters_from_xml(root)

            # Recalculate only if parameters not found in XML
            if not has_parameters:
                self.calculate_and_update_properties()

            # Reset all points
            print("Resetting all points...")
            for i in range(self.num_points):
                # Clear text fields
                self.point_inputs[i][0].setText("0.000000")
                self.point_inputs[i][1].setText("0.000000")
                self.point_inputs[i][2].setText("0.000000")

                # Reset internal coordinate data
                self.point_coords[i] = [0, 0, 0]

                # Uncheck checkbox
                self.point_checkboxes[i].setChecked(False)

                # Hide 3D view points and remove actors
                if self.point_actors[i]:
                    self.point_actors[i].VisibilityOff()
                    self.renderer.RemoveActor(self.point_actors[i])
                    self.point_actors[i] = None
            
            print("All points have been reset")

            # Track points with data set
            points_with_data = set()

            # Load coordinates for each point
            points = root.findall('./point')
            print(f"Found {len(points)} points in XML")

            for i, point in enumerate(points):
                xyz_element = point.find('point_xyz')
                if xyz_element is not None and xyz_element.text:
                    try:
                        # Split coordinate text and convert to numbers
                        x, y, z = map(float, xyz_element.text.strip().split())
                        print(f"Point {i+1}: {x}, {y}, z")

                        # Set values to text fields
                        self.point_inputs[i][0].setText(f"{x:.6f}")
                        self.point_inputs[i][1].setText(f"{y:.6f}")
                        self.point_inputs[i][2].setText(f"{z:.6f}")

                        # Update internal coordinate data
                        self.point_coords[i] = [x, y, z]

                        # Load point_angle (in radians)
                        angle_element = point.find('point_angle')
                        if angle_element is not None and angle_element.text:
                            try:
                                angle_x, angle_y, angle_z = map(float, angle_element.text.strip().split())
                                # Backward compatibility: if values look like degrees, convert to radians
                                if any(abs(v) > 3.5 for v in [angle_x, angle_y, angle_z]):
                                    angle_x = math.radians(angle_x)
                                    angle_y = math.radians(angle_y)
                                    angle_z = math.radians(angle_z)
                                self.point_angles[i] = [angle_x, angle_y, angle_z]
                                print(f"Loaded point_angle for point {i+1}: ({angle_x}, {angle_y}, {angle_z}) [rad]")
                            except Exception as e:
                                print(f"Warning: Invalid point_angle format for point {i+1}, using default [0, 0, 0]: {e}")
                                self.point_angles[i] = [0.0, 0.0, 0.0]
                        else:
                            self.point_angles[i] = [0.0, 0.0, 0.0]

                        # Enable checkbox
                        self.point_checkboxes[i].setChecked(True)

                        # Set point display
                        if self.point_actors[i] is None:
                            self.point_actors[i] = vtk.vtkAssembly()
                            self.create_point_coordinate(self.point_actors[i], [0, 0, 0])
                        
                        self.point_actors[i].SetPosition(self.point_coords[i])
                        self.renderer.AddActor(self.point_actors[i])
                        self.point_actors[i].VisibilityOn()
                        
                        points_with_data.add(i)
                        print(f"Set point {i+1} coordinates: x={x:.6f}, y={y:.6f}, z={z:.6f}")
                    except Exception as e:
                        print(f"Error processing point {i+1}: {e}")

            # Apply color only if STL model is loaded
            if hasattr(self, 'stl_actor') and self.stl_actor:
                color_element = root.find(".//material/color")
                if color_element is not None:
                    rgba_str = color_element.get('rgba')
                    if rgba_str:
                        try:
                            r, g, b, a = map(float, rgba_str.split())

                            # Set values to input fields
                            self.color_inputs[0].setText(f"{r:.3f}")
                            self.color_inputs[1].setText(f"{g:.3f}")
                            self.color_inputs[2].setText(f"{b:.3f}")
                            if len(self.color_inputs) >= 4:
                                self.color_inputs[3].setText(f"{a:.3f}")

                            # Apply color to STL model
                            self.stl_actor.GetProperty().SetColor(r, g, b)
                            self.stl_actor.GetProperty().SetOpacity(a)
                            self.render_to_image()

                            print(f"Material color loaded and applied: RGBA({r:.3f}, {g:.3f}, {b:.3f}, {a:.3f})")
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Invalid color format in XML: {rgba_str}")
                            print(f"Error details: {e}")

            # Process axis information
            axis_element = root.find(".//axis")
            if axis_element is not None:
                xyz_str = axis_element.get('xyz')
                if xyz_str:
                    try:
                        x, y, z = map(float, xyz_str.split())
                        if x == 1:
                            self.radio_buttons[0].setChecked(True)
                        elif y == 1:
                            self.radio_buttons[1].setChecked(True)
                        elif z == 1:
                            self.radio_buttons[2].setChecked(True)
                    except ValueError:
                        print(f"Warning: Invalid axis format in XML: {xyz_str}")

            # Update display
            if hasattr(self, 'renderer'):
                self.renderer.ResetCamera()
                self.update_all_points()

                # Fit camera if STL model exists
                if hasattr(self, 'stl_actor') and self.stl_actor:
                    self.fit_camera_to_model()

                self.renderer.ResetCameraClippingRange()
                self.render_to_image()

            print(f"XML file has been loaded: {xml_path}")
            print(f"Number of set points: {len(points_with_data)}")

        except Exception as e:
            print(f"An error occurred while loading the XML file: {str(e)}")
            traceback.print_exc()

    def reload_files(self):
        """Reload currently loaded Mesh and XML files"""
        try:
            print("\n" + "="*60)
            print("RELOADING FILES")
            print("="*60)

            # Save current file paths
            mesh_path = getattr(self, 'stl_file_path', None)
            xml_path = getattr(self, 'xml_file_path', None)

            if not mesh_path and not xml_path:
                print("No files to reload. Please load files first.")
                return

            reload_count = 0

            # Reload Mesh file
            if mesh_path and os.path.exists(mesh_path):
                print(f"\n[1/2] Reloading mesh file: {mesh_path}")
                self.show_stl(mesh_path)
                reload_count += 1
            elif mesh_path:
                print(f"Warning: Mesh file not found: {mesh_path}")
            else:
                print("No mesh file to reload")

            # Reload XML file
            if xml_path and os.path.exists(xml_path):
                print(f"\n[2/2] Reloading XML file: {xml_path}")

                try:
                    # Parse XML file
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # Load parameters from XML
                    has_parameters = self.load_parameters_from_xml(root)

                    # Recalculate only if parameters not found in XML
                    if not has_parameters:
                        self.calculate_and_update_properties()

                    # Load point data (using existing helper function)
                    points_with_data = self._load_points_from_xml(root)

                    # Apply color information (using existing helper function)
                    self._apply_color_from_xml(root)

                    # Update display
                    self._refresh_display()

                    print(f"✓ XML file reloaded successfully")
                    if points_with_data:
                        print(f"  Loaded {len(points_with_data)} points")
                    reload_count += 1

                except ET.ParseError as e:
                    print(f"Error: Failed to parse XML file: {e}")
                except Exception as e:
                    print(f"Error: Failed to load XML file: {e}")
                    traceback.print_exc()
            elif xml_path:
                print(f"Warning: XML file not found: {xml_path}")
            else:
                print("No XML file to reload")

            # Completion message
            print("\n" + "="*60)
            if reload_count > 0:
                print(f"✓ Reload completed: {reload_count} file(s) reloaded")
            else:
                print("⚠ No files were reloaded")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n✗ An error occurred while reloading files: {str(e)}")
            traceback.print_exc()

    def refresh_view(self):
        """Refresh view and fit camera"""
        if hasattr(self, 'renderer'):
            self.renderer.ResetCamera()
            self.update_all_points()
            # Fit camera if STL model exists
            if hasattr(self, 'stl_actor') and self.stl_actor:
                self.fit_camera_to_model()
            self.renderer.ResetCameraClippingRange()
            self.render_to_image()

    def load_stl_with_xml(self):
        """Load 3D model file (STL/OBJ/DAE) together with XML file"""
        try:
            # Use common utility function for file filter
            file_filter = get_mesh_file_filter(trimesh_available=True)
            stl_path, _ = QFileDialog.getOpenFileName(self, "Open 3D Model File", "", file_filter)
            if not stl_path:
                return

            # Load 3D model file
            self.show_stl(stl_path)

            # Generate corresponding XML file path
            xml_path = os.path.splitext(stl_path)[0] + '.xml'

            if not os.path.exists(xml_path):
                print(f"Corresponding XML file not found: {xml_path}")
                return

            # Parse XML file
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Load parameters from XML
                has_parameters = self.load_parameters_from_xml(root)

                # Recalculate only if parameters not found in XML
                if not has_parameters:
                    self.calculate_and_update_properties()

                # Load point data
                points_with_data = self._load_points_from_xml(root)

                print(f"XML file loaded: {xml_path}")
                if points_with_data:
                    print(f"Loaded {len(points_with_data)} points")

                # Update display
                self.refresh_view()

                # Reset camera (same as R key)
                self.reset_camera()

            except ET.ParseError:
                print(f"Error parsing XML file: {xml_path}")
            except Exception as e:
                print(f"Error processing XML file: {str(e)}")
                traceback.print_exc()

        except Exception as e:
            print(f"An error occurred while loading the file: {str(e)}")
            traceback.print_exc()

    def bulk_convert_l_to_r(self):
        """
        Process STL/OBJ/DAE files starting with 'l_' or 'L_' in folder,
        generate left-right symmetric 'r_' or 'R_' files and XML centered on XZ plane.
        Existing files will be overwritten.
        """
        try:
            import re

            # Show folder selection dialog
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder for Bulk Conversion")
            if not folder_path:
                return

            print(f"Selected folder: {folder_path}")

            # Track number of processed files
            processed_count = 0
            collider_count = 0
            # Save list of generated files
            generated_files = []

            # l_*_collider pattern (starts with l_ and ends with _collider.extension)
            # Example: l_hipjoint_upper_collider.stl, l_elbow_collider.dae
            collider_pattern = re.compile(r'^l_.+_collider\.(stl|obj|dae)$', re.IGNORECASE)

            # Search all STL/OBJ/DAE files in folder
            print("\n=== Searching for l_ prefix files ===")
            print("Collider pattern: l_*_collider.(stl|obj|dae)")
            print("Examples: l_hipjoint_upper_collider.stl, l_elbow_collider.dae")
            print("")

            for file_name in os.listdir(folder_path):
                if file_name.lower().startswith(('l_', 'L_')) and file_name.lower().endswith(('.stl', '.obj', '.dae')):
                    # Determine if this is a collider file
                    is_collider = collider_pattern.match(file_name.lower())
                    file_type = "COLLIDER" if is_collider else "mesh"

                    print(f"✓ Found {file_type}: {file_name}")
                    stl_path = os.path.join(folder_path, file_name)

                    # Generate new filename (convert l_ to r_)
                    new_name = 'R_' + file_name[2:] if file_name.startswith('L_') else 'r_' + file_name[2:]
                    new_name_without_ext = os.path.splitext(new_name)[0]
                    new_stl_path = os.path.join(folder_path, new_name)
                    new_xml_path = os.path.splitext(new_stl_path)[0] + '.xml'

                    print(f"  → Mesh: {new_name}")
                    print(f"  → XML:  {os.path.basename(new_xml_path)}")
                    if is_collider:
                        # For collider, explicitly show XML filename pattern
                        print(f"  → Pattern match: r_.*_collider.xml ✓")

                    # Overwrite if existing files exist
                    if os.path.exists(new_stl_path) or os.path.exists(new_xml_path):
                        print(f"  ⚠ Overwriting existing files")

                    print(f"\nProcessing: {stl_path}")

                    try:
                        # Use common utility function to load mesh
                        poly_data, volume_unused, extracted_color = load_mesh_to_polydata(stl_path)

                        # Set Y-axis flip transform
                        transform = vtk.vtkTransform()
                        transform.Scale(1, -1, 1)

                        # Transform vertices
                        transformer = vtk.vtkTransformPolyDataFilter()
                        transformer.SetInputData(poly_data)
                        transformer.SetTransform(transform)
                        transformer.Update()

                        # Fix normals
                        normal_generator = vtk.vtkPolyDataNormals()
                        normal_generator.SetInputData(transformer.GetOutput())
                        normal_generator.ConsistencyOn()
                        normal_generator.AutoOrientNormalsOn()
                        normal_generator.ComputeCellNormalsOn()
                        normal_generator.ComputePointNormalsOn()
                        normal_generator.Update()

                        # Search for corresponding XML file
                        xml_path = os.path.splitext(stl_path)[0] + '.xml'
                        xml_data = None
                        mass_value_str = "0.0"
                        volume_value_str = "0.0"
                        mass = 0.0
                        volume = 0.0
                        center_of_mass = [0.0, 0.0, 0.0]
                        rgba_str = "1.0 1.0 1.0 1.0"
                        hex_color = "#FFFFFF"
                        if os.path.exists(xml_path):
                            try:
                                tree = ET.parse(xml_path)
                                xml_data = tree.getroot()
                                print(f"Found and loaded XML file: {xml_path}")

                                # Get physical parameters from XML (preserve original format)
                                mass_element = xml_data.find(".//mass")
                                mass_value_str = mass_element.get('value') if mass_element is not None else "0.0"
                                mass = float(mass_value_str)  # For calculation

                                volume_element = xml_data.find(".//volume")
                                volume_value_str = volume_element.get('value') if volume_element is not None else "0.0"
                                volume = float(volume_value_str)  # For calculation

                                # Get center of mass position (from center_of_mass element)
                                com_element = xml_data.find(".//center_of_mass")
                                if com_element is not None and com_element.text:
                                    x, y, z = map(float, com_element.text.strip().split())
                                    center_of_mass = [x, -y, z]  # Flip Y coordinate only
                                else:
                                    # Get from inertial origin element
                                    inertial_origin = xml_data.find(".//inertial/origin")
                                    if inertial_origin is not None:
                                        xyz = inertial_origin.get('xyz')
                                        x, y, z = map(float, xyz.split())
                                        center_of_mass = [x, -y, z]  # Flip Y coordinate only

                                # Get color information
                                color_element = xml_data.find(".//material/color")
                                if color_element is not None:
                                    rgba_str = color_element.get('rgba')
                                    hex_color = xml_data.find(".//material").get('name')
                                else:
                                    rgba_str = "1.0 1.0 1.0 1.0"
                                    hex_color = "#FFFFFF"

                            except ET.ParseError:
                                print(f"Error parsing XML file: {xml_path}")
                                continue

                        # Use common utility function to save mesh
                        save_polydata_to_mesh(new_stl_path, normal_generator.GetOutput())

                        # Get inertia tensor and apply mirror transformation
                        inertia_element = None
                        if xml_data is not None:
                            inertia_element = xml_data.find(".//inertia")

                        if inertia_element is not None:
                            # Get inertia tensor from original XML and apply mirror transformation
                            ixx = float(inertia_element.get('ixx', 0))
                            iyy = float(inertia_element.get('iyy', 0))
                            izz = float(inertia_element.get('izz', 0))
                            ixy = float(inertia_element.get('ixy', 0))
                            ixz = float(inertia_element.get('ixz', 0))
                            iyz = float(inertia_element.get('iyz', 0))
                            # For Y-axis mirror, flip sign of ixy and iyz
                            inertia_str = f'ixx="{ixx:.12f}" ixy="{-ixy:.12f}" ixz="{ixz:.12f}" iyy="{iyy:.12f}" iyz="{-iyz:.12f}" izz="{izz:.12f}"'
                        else:
                            # If original XML has no inertia data, calculate from mesh
                            print("Warning: No inertia data in XML, calculating from mesh...")
                            inertia_tensor = self.calculate_inertia_tensor_for_mirrored(
                                normal_generator.GetOutput(), mass, center_of_mass)
                            # Remove <inertia and /> from format_inertia_for_urdf return value
                            inertia_formatted = self.format_inertia_for_urdf(inertia_tensor)
                            inertia_str = inertia_formatted.replace('<inertia ', '').replace('/>', '').strip()

                        # Generate XML file content
                        urdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urdf_part>
    <material name="{hex_color}">
        <color rgba="{rgba_str}" />
    </material>
    <link name="{new_name_without_ext}">
        <visual>
            <origin xyz="{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}" rpy="0 0 0"/>
            <material name="{hex_color}" />
        </visual>
        <inertial>
            <origin xyz="{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}"/>
            <mass value="{mass_value_str}"/>
            <volume value="{volume_value_str}"/>
            <inertia {inertia_str} />
        </inertial>
        <center_of_mass>{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}</center_of_mass>
    </link>"""

                        # Flip and copy point data
                        if xml_data is not None:
                            points = xml_data.findall('.//point')
                            for point in points:
                                xyz_element = point.find('point_xyz')
                                if xyz_element is not None and xyz_element.text:
                                    try:
                                        x, y, z = map(float, xyz_element.text.strip().split())
                                        mirrored_y = -y  # Flip Y coordinate only
                                        point_name = point.get('name')

                                        # Process point_angle (rotation transform for Y-axis mirror)
                                        angle_element = point.find('point_angle')
                                        point_angle_str = ""
                                        if angle_element is not None and angle_element.text:
                                            try:
                                                angle_x, angle_y, angle_z = map(float, angle_element.text.strip().split())
                                                # For Y-axis mirror, flip sign of X and Z axis rotations
                                                mirrored_angle_x = -angle_x
                                                mirrored_angle_y = angle_y
                                                mirrored_angle_z = -angle_z
                                                point_angle_str = f"\n        <point_angle>{mirrored_angle_x:.6f} {mirrored_angle_y:.6f} {mirrored_angle_z:.6f}</point_angle>"
                                            except ValueError:
                                                pass  # Ignore if point_angle is invalid

                                        urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>{point_angle_str}
    </point>"""
                                    except ValueError:
                                        print(f"Error processing point coordinates in XML")

                        # Get and apply axis information
                        if xml_data is not None:
                            axis_element = xml_data.find('.//joint/axis')
                            if axis_element is not None:
                                axis_str = axis_element.get('xyz')
                                mirrored_axis = self.mirror_axis_value(axis_str)
                            else:
                                mirrored_axis = "1 0 0"
                        else:
                            mirrored_axis = "1 0 0"

                        urdf_content += f"""
    <joint>
        <axis xyz="{mirrored_axis}" />
    </joint>
</urdf_part>"""

                        # Save XML file
                        with open(new_xml_path, "w") as f:
                            f.write(urdf_content)

                        processed_count += 1
                        if is_collider:
                            collider_count += 1
                        # Add generated files to list
                        generated_files.append({
                            'mesh': new_stl_path,
                            'xml': new_xml_path
                        })
                        print(f"Converted {file_type}: {file_name} -> {new_name}")
                        print(f"Created XML: {new_xml_path}")

                    except Exception as e:
                        print(f"Error processing file {file_name}: {str(e)}")
                        traceback.print_exc()
                        continue

            # ===== Phase 2: Process standalone XML files (without mesh) =====
            print("\n=== Searching for standalone collider XML files ===")
            xml_collider_pattern = re.compile(r'^l_.+_collider\.xml$', re.IGNORECASE)
            xml_only_count = 0

            # List of already processed XMLs (processed with mesh)
            processed_xml_names = set()
            for item in generated_files:
                if 'xml' in item:
                    processed_xml_names.add(os.path.basename(item['xml']))

            for file_name in os.listdir(folder_path):
                if xml_collider_pattern.match(file_name.lower()):
                    # Check if already processed with mesh
                    output_xml_name = 'R_' + file_name[2:] if file_name.startswith('L_') else 'r_' + file_name[2:]
                    if output_xml_name in processed_xml_names:
                        continue  # Already processed

                    xml_path = os.path.join(folder_path, file_name)
                    new_xml_name = output_xml_name
                    new_xml_path = os.path.join(folder_path, new_xml_name)

                    print(f"✓ Found standalone XML: {file_name}")
                    print(f"  → Will create: {new_xml_name}")

                    try:
                        # Load XML file
                        tree = ET.parse(xml_path)
                        xml_data = tree.getroot()

                        # Get physical parameters
                        mass_element = xml_data.find(".//mass")
                        volume_element = xml_data.find(".//volume")

                        mass = float(mass_element.get('value')) if mass_element is not None else 1.0
                        volume = float(volume_element.get('value')) if volume_element is not None else 1.0

                        # Get center of mass position and flip
                        com_element = xml_data.find(".//center_of_mass")
                        if com_element is not None and com_element.text:
                            x, y, z = map(float, com_element.text.strip().split())
                            center_of_mass = [x, -y, z]  # Flip Y coordinate only
                        else:
                            # Get from inertial origin element
                            inertial_origin = xml_data.find(".//inertial/origin")
                            if inertial_origin is not None:
                                xyz = inertial_origin.get('xyz')
                                x, y, z = map(float, xyz.split())
                                center_of_mass = [x, -y, z]
                            else:
                                center_of_mass = [0, 0, 0]

                        # Get color information
                        color_element = xml_data.find(".//material/color")
                        if color_element is not None:
                            rgba_str = color_element.get('rgba')
                            hex_color = xml_data.find(".//material").get('name')
                        else:
                            rgba_str = "1.0 1.0 1.0 1.0"
                            hex_color = "#FFFFFF"

                        # Get inertia tensor and flip
                        inertia_element = xml_data.find(".//inertia")
                        if inertia_element is not None:
                            ixx = float(inertia_element.get('ixx', 0))
                            iyy = float(inertia_element.get('iyy', 0))
                            izz = float(inertia_element.get('izz', 0))
                            ixy = float(inertia_element.get('ixy', 0))
                            ixz = float(inertia_element.get('ixz', 0))
                            iyz = float(inertia_element.get('iyz', 0))
                            # For Y-axis mirror, flip sign of ixy and iyz
                            inertia_str = f'ixx="{ixx:.12f}" ixy="{-ixy:.12f}" ixz="{ixz:.12f}" iyy="{iyy:.12f}" iyz="{-iyz:.12f}" izz="{izz:.12f}"'
                        else:
                            inertia_str = 'ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"'

                        # Generate link name
                        base_name = os.path.splitext(new_xml_name)[0]

                        # Generate XML file content
                        urdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urdf_part>
    <material name="{hex_color}">
        <color rgba="{rgba_str}" />
    </material>
    <link name="{base_name}">
        <visual>
            <origin xyz="{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}" rpy="0 0 0"/>
            <material name="{hex_color}" />
        </visual>
        <inertial>
            <origin xyz="{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}"/>
            <mass value="{mass:.12f}"/>
            <volume value="{volume:.12f}"/>
            <inertia {inertia_str} />
        </inertial>
        <center_of_mass>{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}</center_of_mass>
    </link>"""

                        # Flip and copy point data
                        points = xml_data.findall('.//point')
                        for point in points:
                            xyz_element = point.find('point_xyz')
                            if xyz_element is not None and xyz_element.text:
                                try:
                                    x, y, z = map(float, xyz_element.text.strip().split())
                                    mirrored_y = -y  # Flip Y coordinate only
                                    point_name = point.get('name')
                                    urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>
    </point>"""
                                except ValueError:
                                    print(f"Error processing point coordinates in XML")

                        # Get and apply axis information
                        axis_element = xml_data.find('.//joint/axis')
                        if axis_element is not None:
                            axis_str = axis_element.get('xyz')
                            mirrored_axis = self.mirror_axis_value(axis_str)
                        else:
                            mirrored_axis = "1 0 0"

                        urdf_content += f"""
    <joint>
        <axis xyz="{mirrored_axis}" />
    </joint>
</urdf_part>"""

                        # Save XML file
                        with open(new_xml_path, "w") as f:
                            f.write(urdf_content)

                        xml_only_count += 1
                        generated_files.append({
                            'mesh': None,
                            'xml': new_xml_path
                        })
                        print(f"Converted standalone XML: {file_name} -> {new_xml_name}")

                    except Exception as e:
                        print(f"Error processing standalone XML {file_name}: {str(e)}")
                        traceback.print_exc()
                        continue

            # ===== Phase 3: Process collider XML files (l_*_collider.xml) =====
            print("\n=== Searching for collider XML files (l_*_collider.xml) ===")
            collider_xml_pattern = re.compile(r'^l_.+_collider\.xml$', re.IGNORECASE)
            collider_xml_count = 0

            for file_name in os.listdir(folder_path):
                if collider_xml_pattern.match(file_name.lower()):
                    collider_xml_path = os.path.join(folder_path, file_name)
                    new_collider_xml_name = 'R_' + file_name[2:] if file_name.startswith('L_') else 'r_' + file_name[2:]
                    new_collider_xml_path = os.path.join(folder_path, new_collider_xml_name)

                    print(f"✓ Found collider XML: {file_name}")
                    print(f"  → Will create: {new_collider_xml_name}")

                    try:
                        # Load collider XML file
                        tree = ET.parse(collider_xml_path)
                        root = tree.getroot()

                        if root.tag != 'urdf_kitchen_collider':
                            print(f"  ⚠ Warning: Invalid collider XML format (expected 'urdf_kitchen_collider'), skipping")
                            continue

                        collider_elem = root.find('collider')
                        if collider_elem is None:
                            print(f"  ⚠ Warning: No collider element found, skipping")
                            continue

                        # Get collider type
                        collider_type = collider_elem.get('type', 'box')

                        # Get geometry information
                        geometry_elem = collider_elem.find('geometry')
                        geometry_attrs = {}
                        if geometry_elem is not None:
                            geometry_attrs = dict(geometry_elem.attrib)

                        # Get position and flip on XZ plane (flip Y coordinate)
                        position_elem = collider_elem.find('position')
                        if position_elem is not None:
                            x = float(position_elem.get('x', '0.0'))
                            y = float(position_elem.get('y', '0.0'))
                            z = float(position_elem.get('z', '0.0'))
                            mirrored_y = -y  # Flip Y coordinate
                        else:
                            x, y, z = 0.0, 0.0, 0.0
                            mirrored_y = 0.0

                        # Get rotation and flip on XZ plane
                        # For XZ plane flip, keep roll and yaw, flip pitch
                        rotation_elem = collider_elem.find('rotation')
                        if rotation_elem is not None:
                            roll = float(rotation_elem.get('roll', '0.0'))
                            pitch = float(rotation_elem.get('pitch', '0.0'))
                            yaw = float(rotation_elem.get('yaw', '0.0'))
                            # Flip Roll and Yaw, keep Pitch
                            mirrored_roll = -roll
                            mirrored_pitch = pitch  # Don't flip Pitch
                            mirrored_yaw = -yaw
                        else:
                            roll, pitch, yaw = 0.0, 0.0, 0.0
                            mirrored_roll = 0.0
                            mirrored_pitch = 0.0
                            mirrored_yaw = 0.0

                        # Create new collider XML file
                        new_root = ET.Element('urdf_kitchen_collider')
                        new_collider_elem = ET.SubElement(new_root, 'collider')
                        new_collider_elem.set('type', collider_type)

                        # Add geometry element
                        if geometry_attrs:
                            new_geometry_elem = ET.SubElement(new_collider_elem, 'geometry')
                            for key, value in geometry_attrs.items():
                                new_geometry_elem.set(key, value)

                        # Add position element (flip Y coordinate)
                        new_position_elem = ET.SubElement(new_collider_elem, 'position')
                        new_position_elem.set('x', f"{x:.6f}")
                        new_position_elem.set('y', f"{mirrored_y:.6f}")
                        new_position_elem.set('z', f"{z:.6f}")

                        # Add rotation element (flip Roll and Yaw, keep Pitch)
                        new_rotation_elem = ET.SubElement(new_collider_elem, 'rotation')
                        new_rotation_elem.set('roll', f"{mirrored_roll:.6f}")
                        new_rotation_elem.set('pitch', f"{mirrored_pitch:.6f}")
                        new_rotation_elem.set('yaw', f"{mirrored_yaw:.6f}")
                        print(f"  Mirrored rotation: Roll={roll:.4f}->{mirrored_roll:.4f}, Pitch={pitch:.4f}->{mirrored_pitch:.4f}, Yaw={yaw:.4f}->{mirrored_yaw:.4f}")

                        # Save XML file
                        new_tree = ET.ElementTree(new_root)
                        ET.indent(new_tree, space="    ")
                        new_tree.write(new_collider_xml_path, encoding='utf-8', xml_declaration=True)
                        print(f"  ✓ Created collider XML: {new_collider_xml_name}")

                        collider_xml_count += 1
                        generated_files.append({
                            'mesh': None,
                            'xml': new_collider_xml_path
                        })

                    except Exception as e:
                        print(f"  ✗ Error processing collider XML {file_name}: {str(e)}")
                        traceback.print_exc()
                        continue

            # Processing complete message
            total_processed = processed_count + xml_only_count + collider_xml_count
            if total_processed > 0:
                regular_count = processed_count - collider_count
                print(f"\n=== Bulk conversion completed ===")
                print(f"Total processed: {total_processed} files")
                print(f"  - Regular meshes: {regular_count} files")
                print(f"  - Colliders (mesh+XML): {collider_count} files")
                if xml_only_count > 0:
                    print(f"  - Standalone collider XMLs: {xml_only_count} files")
                if collider_xml_count > 0:
                    print(f"  - Collider XML files (l_*_collider.xml): {collider_xml_count} files")
                print(f"All files mirrored across XZ plane (Y-axis flip)")
                # Show generated file list in dialog box
                dialog = BulkConversionCompleteDialog(generated_files, folder_path, self)
                dialog.exec()
            else:
                print("\n=== No matching files found ===")
                print("Looking for files with 'l_' prefix and extensions: .stl, .obj, .dae")
                QMessageBox.information(self, "No Files Found",
                    "No files with 'l_' or 'L_' prefix were found in the selected folder.\nSupported formats: STL, OBJ, DAE")

        except Exception as e:
            print(f"Error during bulk conversion: {str(e)}")
            traceback.print_exc()

    def mirror_axis_value(self, axis_str):
        """
        Process axis information for left-right mirroring
        Rotation axis direction is not changed for mirroring

        Args:
            axis_str (str): axis information in "x y z" format

        Returns:
            str: transformed axis information
        """
        try:
            x, y, z = map(float, axis_str.split())
            # Return as-is without changing axis direction
            return f"{x:.1f} {y:.1f} {z:.1f}"
        except ValueError:
            print(f"Error parsing axis values: {axis_str}")
            return "1 0 0"  # Default value


    def start_rotation_test(self):
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            return

        self.original_transform = vtk.vtkTransform()
        self.original_transform.DeepCopy(self.stl_actor.GetUserTransform()
                                    if self.stl_actor.GetUserTransform()
                                    else vtk.vtkTransform())

        # Save original positions for 3D view rotation
        self.original_point_positions = []
        for i in range(self.num_points):
            if self.point_actors[i]:
                self.original_point_positions.append(list(self.point_actors[i].GetPosition()))
            else:
                self.original_point_positions.append(list(self.point_coords[i]))
        self.original_com_position = list(self.com_coords)

        self.test_rotation_angle = 0
        self.rotation_timer.start(16)

    def stop_rotation_test(self):
        self.rotation_timer.stop()

        if self.stl_actor and self.original_transform:
            self.stl_actor.SetUserTransform(self.original_transform)

        # Restore original positions (3D view only, input fields unchanged)
        if hasattr(self, 'original_point_positions'):
            for i in range(self.num_points):
                if self.point_actors[i] and i < len(self.original_point_positions):
                    self.point_actors[i].SetPosition(self.original_point_positions[i])
        if hasattr(self, 'original_com_position'):
            if self.com_sphere_actor:
                self.com_sphere_actor.SetPosition(self.original_com_position)
            if self.com_cursor_actor:
                self.com_cursor_actor.SetPosition(self.original_com_position)

        self.render_to_image()

    def update_test_rotation(self):
        if not self.stl_actor:
            return

        axis_index = self.axis_group.checkedId()
        if axis_index == 3:  # Fixed axis selected
            return

        rotation_axis = [0, 0, 0]
        rotation_axis[axis_index] = 1
        self.test_rotation_angle += 2

        transform = vtk.vtkTransform()
        transform.DeepCopy(self.original_transform)
        transform.RotateWXYZ(self.test_rotation_angle, *rotation_axis)
        self.stl_actor.SetUserTransform(transform)

        # Apply rotation to points and Center of Mass (3D view only)
        angle_rad = math.radians(self.test_rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        def rotate_point(pos, axis_idx):
            x, y, z = pos
            if axis_idx == 0:
                return [x, y * cos_a - z * sin_a, y * sin_a + z * cos_a]
            elif axis_idx == 1:
                return [x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a]
            else:
                return [x * cos_a - y * sin_a, x * sin_a + y * cos_a, z]

        if hasattr(self, 'original_point_positions'):
            for i in range(self.num_points):
                if self.point_actors[i] and i < len(self.original_point_positions):
                    rotated_pos = rotate_point(self.original_point_positions[i], axis_index)
                    self.point_actors[i].SetPosition(rotated_pos)

        if hasattr(self, 'original_com_position'):
            rotated_com = rotate_point(self.original_com_position, axis_index)
            if self.com_sphere_actor:
                self.com_sphere_actor.SetPosition(rotated_com)
            if self.com_cursor_actor:
                self.com_cursor_actor.SetPosition(rotated_com)

        self.render_to_image()

    def _on_color_changed(self, rgba_color):
        """
        KitchenColorPicker callback when color changes.

        Args:
            rgba_color: RGBA color list [r, g, b, a] in 0-1 range
        """
        # Apply color to 3D model automatically
        self.apply_color_to_stl()

    def apply_color_to_stl(self):
        """Apply selected color to 3D model (RGBA supported)"""
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No 3D model has been loaded.")
            return

        try:
            # Get RGBA values (0-1 range)
            rgba_values = [float(input.text()) for input in self.color_inputs]

            # Range check for values
            rgba_values = [max(0.0, min(1.0, value)) for value in rgba_values]

            # Save as mesh color (RGBA format)
            if len(rgba_values) == 3:
                # RGB only case, add Alpha=1.0
                self.mesh_color = rgba_values + [1.0]
            else:
                # RGBA case, save as-is
                self.mesh_color = rgba_values

            # Mark that color was manually changed
            self.color_manually_changed = True

            # Change STL model color (RGB)
            self.stl_actor.GetProperty().SetColor(*rgba_values[:3])

            # Set opacity (Alpha)
            if len(rgba_values) >= 4:
                self.stl_actor.GetProperty().SetOpacity(rgba_values[3])
                print(f"Applied color: RGBA({rgba_values[0]:.3f}, {rgba_values[1]:.3f}, "
                      f"{rgba_values[2]:.3f}, {rgba_values[3]:.3f})")
            else:
                self.stl_actor.GetProperty().SetOpacity(1.0)
                print(f"Applied color: RGB({rgba_values[0]:.3f}, {rgba_values[1]:.3f}, {rgba_values[2]:.3f})")

            self.render_to_image()

        except ValueError as e:
            print(f"Error: Invalid color value - {str(e)}")
        except Exception as e:
            print(f"Error applying color: {str(e)}")

    def add_axes_widget(self):
        """Add axes widget (disabled in offscreen rendering mode)"""
        # Note: vtkOrientationMarkerWidget requires an interactor,
        # which is not available in offscreen rendering mode
        # Return None to maintain compatibility
        return None

    def add_instruction_text(self):
        """Display instruction text on screen"""
        # Top-left text
        text_actor_top = vtk.vtkTextActor()
        text_actor_top.SetInput(
            "[W/S]: Up/Down Rotate\n"
            "[A/D]: Left/Right Rotate\n"
            "[Q/E]: Roll\n"
            "[R]: Reset Camera\n"
            "[T]: Wireframe\n\n"
            "[Drag]: Rotate\n"
            "[Shift + Drag]: Move View\n"
        )
        text_actor_top.GetTextProperty().SetFontSize(14)
        text_actor_top.GetTextProperty().SetColor(0.3, 0.8, 1.0)  # Cyan
        text_actor_top.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor_top.SetPosition(0.03, 0.97)  # Position at top-left
        text_actor_top.GetTextProperty().SetJustificationToLeft()
        text_actor_top.GetTextProperty().SetVerticalJustificationToTop()
        self.renderer.AddActor(text_actor_top)

        # Bottom-left text
        text_actor_bottom = vtk.vtkTextActor()
        text_actor_bottom.SetInput(
            "[Arrows] : Move Point 10mm\n"
            " +[Shift]: Move Point 1mm\n"
            "  +[Ctrl]: Move Point 0.1mm\n\n"
        )
        text_actor_bottom.GetTextProperty().SetFontSize(14)
        text_actor_bottom.GetTextProperty().SetColor(0.3, 0.8, 1.0)  # Cyan
        text_actor_bottom.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor_bottom.SetPosition(0.03, 0.03)  # Position at bottom-left
        text_actor_bottom.GetTextProperty().SetJustificationToLeft()
        text_actor_bottom.GetTextProperty().SetVerticalJustificationToBottom()
        self.renderer.AddActor(text_actor_bottom)

    def process_mirror_properties(self, xml_data, reverse_output, density=1.0):
        """
        Process physical properties for mirrored model
        Args:
            xml_data: original XML data
            reverse_output: vtkPolyData after mirroring
            density: default density (used when original XML has no mass info)
        Returns:
            tuple: (volume, mass, center_of_mass, inertia_tensor)
        """
        # Calculate volume (from new geometry)
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(reverse_output)
        volume = mass_properties.GetVolume()

        # Get mass from original XML (calculate as volume * density if not found)
        if xml_data is not None:
            mass_element = xml_data.find(".//mass")
            if mass_element is not None:
                mass = float(mass_element.get('value'))
            else:
                mass = volume * density
        else:
            mass = volume * density

        # Calculate center of mass
        com_filter = vtk.vtkCenterOfMass()
        com_filter.SetInputData(reverse_output)
        com_filter.SetUseScalarsAsWeights(False)
        com_filter.Update()
        center_of_mass = list(com_filter.GetCenter())

        # Flip Y coordinate only
        center_of_mass[1] = -center_of_mass[1]

        # Calculate inertia tensor (considering mass)
        inertia_tensor = np.zeros((3, 3))
        poly_data = reverse_output
        num_cells = poly_data.GetNumberOfCells()

        # Calculate inertia tensor using actual mass
        density_for_inertia = mass / volume  # Back-calculate density from actual mass

        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                p1, p2, p3 = [np.array(cell.GetPoints().GetPoint(j)) for j in range(3)]
                centroid = (p1 + p2 + p3) / 3
                r = centroid - np.array(center_of_mass)
                area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

                # Inertia tensor calculation
                inertia_tensor[0, 0] += area * (r[1]**2 + r[2]**2)
                inertia_tensor[1, 1] += area * (r[0]**2 + r[2]**2)
                inertia_tensor[2, 2] += area * (r[0]**2 + r[1]**2)
                inertia_tensor[0, 1] -= area * r[0] * r[1]
                inertia_tensor[0, 2] -= area * r[0] * r[2]
                inertia_tensor[1, 2] -= area * r[1] * r[2]

        # Fill lower triangle using symmetry
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]

        # Scale inertia tensor based on actual mass
        inertia_tensor *= density_for_inertia

        return volume, mass, center_of_mass, inertia_tensor

    def eventFilter(self, obj, event):
        """Handle mouse events on vtk_display"""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent

        if obj == self.vtk_display:
            if event.type() == QEvent.MouseButtonPress:
                if isinstance(event, QMouseEvent):
                    # Handle left button for drag rotation
                    if event.button() == Qt.LeftButton:
                        self.vtk_display.setFocus()
                        self.mouse_pressed = True
                        self.last_mouse_pos = event.pos()
                        self.vtk_display.grabMouse()
                        return True
                    # Handle middle button (wheel button) for pan
                    elif event.button() == Qt.MiddleButton:
                        self.vtk_display.setFocus()
                        self.middle_mouse_pressed = True
                        self.last_mouse_pos = event.pos()
                        self.vtk_display.grabMouse()
                        return True

            elif event.type() == QEvent.MouseButtonRelease:
                if isinstance(event, QMouseEvent):
                    # Release on left button stops rotation
                    if event.button() == Qt.LeftButton:
                        self.mouse_pressed = False
                        self.last_mouse_pos = None
                        self.vtk_display.releaseMouse()
                        return True
                    # Release on middle button stops pan
                    elif event.button() == Qt.MiddleButton:
                        self.middle_mouse_pressed = False
                        self.last_mouse_pos = None
                        self.vtk_display.releaseMouse()
                        return True

            elif event.type() == QEvent.MouseMove:
                if isinstance(event, QMouseEvent):
                    # Handle left button drag
                    if self.mouse_pressed and self.last_mouse_pos:
                        current_pos = event.pos()
                        dx = current_pos.x() - self.last_mouse_pos.x()
                        dy = current_pos.y() - self.last_mouse_pos.y()

                        # Check if Shift is pressed for pan, otherwise rotate
                        if event.modifiers() & Qt.ShiftModifier:
                            # Pan (move camera)
                            self.pan_camera(dx, dy)
                        else:
                            # Rotate camera
                            self.rotate_camera_mouse(dx, dy)

                        self.last_mouse_pos = current_pos
                        return True
                    # Handle middle button drag (always pan)
                    elif self.middle_mouse_pressed and self.last_mouse_pos:
                        current_pos = event.pos()
                        dx = current_pos.x() - self.last_mouse_pos.x()
                        dy = current_pos.y() - self.last_mouse_pos.y()

                        # Middle button always pans
                        self.pan_camera(dx, dy)

                        self.last_mouse_pos = current_pos
                        return True

            elif event.type() == QEvent.Wheel:
                # Handle trackpad/mouse wheel - ZOOM ONLY
                delta_y = event.angleDelta().y()

                # Only handle vertical scroll for zoom
                # Ignore horizontal scroll to prevent unwanted rotation
                if delta_y != 0:
                    # Get mouse position for zoom center
                    mouse_pos = event.position() if hasattr(event, 'position') else event.pos()
                    self.zoom_camera(delta_y, mouse_pos)

                return True

        return super().eventFilter(obj, event)

    def pan_camera(self, dx, dy):
        """Pan camera based on mouse drag with Shift"""
        self.camera_controller.pan(dx, dy, pan_speed_factor=CAMERA_PAN_SPEED_FACTOR)
        self.render_to_image()

    def zoom_camera(self, delta, mouse_pos=None):
        """Zoom camera based on mouse wheel, centered on mouse position"""
        # If mouse position not provided, use simple zoom from CameraController
        if mouse_pos is None:
            self.camera_controller.zoom(delta, zoom_factor=CAMERA_ZOOM_FACTOR)
            self.render_to_image()
            return

        # Mouse-centered zoom implementation
        camera = self.renderer.GetActiveCamera()
        current_scale = camera.GetParallelScale()

        if delta > 0:
            new_scale = current_scale * CAMERA_ZOOM_IN_SCALE
            zoom_factor = CAMERA_ZOOM_IN_SCALE
        else:
            new_scale = current_scale * CAMERA_ZOOM_OUT_SCALE
            zoom_factor = CAMERA_ZOOM_OUT_SCALE

        # Get display size
        display_width = self.vtk_display.width()
        display_height = self.vtk_display.height()

        # Convert mouse position to normalized coordinates
        mouse_x = mouse_pos.x()
        mouse_y = mouse_pos.y()

        # Calculate offset from center
        center_x = display_width / 2.0
        center_y = display_height / 2.0
        offset_x = (mouse_x - center_x) / display_width
        offset_y = (center_y - mouse_y) / display_height  # Y is inverted

        # Get camera orientation vectors
        focal_point = camera.GetFocalPoint()
        position = camera.GetPosition()
        view_up = camera.GetViewUp()

        # Calculate right and up vectors (camera coordinate system)
        view_dir = np.array([focal_point[i] - position[i] for i in range(3)])
        view_dir = view_dir / np.linalg.norm(view_dir)
        up_vec = np.array(view_up)
        right_vec = np.cross(view_dir, up_vec)
        right_vec = right_vec / np.linalg.norm(right_vec)

        # Recalculate up vector to ensure orthogonality
        up_vec = np.cross(right_vec, view_dir)
        up_vec = up_vec / np.linalg.norm(up_vec)

        # Calculate pan offset based on current scale and mouse offset
        pan_scale = current_scale * MOUSE_ZOOM_PAN_SCALE
        pan_offset_x = offset_x * pan_scale * (1.0 - zoom_factor)
        pan_offset_y = offset_y * pan_scale * (1.0 - zoom_factor)

        # Move focal point and position towards mouse cursor
        new_focal = [
            focal_point[0] + right_vec[0] * pan_offset_x + up_vec[0] * pan_offset_y,
            focal_point[1] + right_vec[1] * pan_offset_x + up_vec[1] * pan_offset_y,
            focal_point[2] + right_vec[2] * pan_offset_x + up_vec[2] * pan_offset_y
        ]

        new_position = [
            position[0] + right_vec[0] * pan_offset_x + up_vec[0] * pan_offset_y,
            position[1] + right_vec[1] * pan_offset_x + up_vec[1] * pan_offset_y,
            position[2] + right_vec[2] * pan_offset_x + up_vec[2] * pan_offset_y
        ]

        camera.SetFocalPoint(new_focal)
        camera.SetPosition(new_position)

        camera.SetParallelScale(new_scale)
        self.renderer.ResetCameraClippingRange()
        self.render_to_image()

    def closeEvent(self, event):
        """Override QMainWindow's closeEvent to properly cleanup VTK resources"""
        print("Window is closing...")
        try:
            if hasattr(self, 'render_window') and self.render_window:
                self.render_window.Finalize()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        event.accept()


class ResultDialog(QDialog):
    def __init__(self, stl_path: str, xml_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Complete")
        self.setModal(True)

        # Set larger window size
        self.resize(400, 250)  # Increase width to 600, height to 200

        # Create layout
        layout = QVBoxLayout()
        layout.setSpacing(10)  # Set spacing between widgets

        # Create message label
        title_label = QLabel("Following files have been saved:")
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)

        # Display 3D model file path
        stl_label = QLabel(f"Mesh: {stl_path}")
        stl_label.setWordWrap(True)  # Enable word wrap for long paths
        layout.addWidget(stl_label)

        # Display XML file path
        xml_label = QLabel(f"XML: {xml_path}")
        xml_label.setWordWrap(True)  # Enable word wrap for long paths
        layout.addWidget(xml_label)

        # Add spacer
        layout.addSpacing(20)

        # Create Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setFixedWidth(100)

        # Horizontal layout to center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Allow closing dialog with Enter key
        close_button.setDefault(True)

class BulkConversionCompleteDialog(QDialog):
    """Dialog to display list of generated files after Batch Mirror processing"""
    def __init__(self, generated_files: list, folder_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Conversion Complete")
        self.setModal(True)

        # Set window size
        self.resize(600, 400)

        # Create layout
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Count by file type
        mesh_count = sum(1 for f in generated_files if f['mesh'] is not None)
        xml_only_count = sum(1 for f in generated_files if f['mesh'] is None)

        # Create message label
        title_label = QLabel(f"✓ Batch Conversion Complete!")
        title_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #2ecc71;")
        layout.addWidget(title_label)

        # Display statistics
        stats_text = f"Total: {len(generated_files)} pair(s) created"
        if mesh_count > 0:
            stats_text += f"\n  • Mesh files: {mesh_count}"
        if xml_only_count > 0:
            stats_text += f"\n  • Standalone XMLs: {xml_only_count}"
        stats_text += "\n  • Mirrored across XZ plane (Y-axis flip)"

        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-size: 11pt; margin: 5px 0px;")
        layout.addWidget(stats_label)

        # Display directory path
        dir_label = QLabel(f"Directory: {folder_path}")
        dir_label.setStyleSheet("font-size: 10pt;")
        dir_label.setWordWrap(True)
        layout.addWidget(dir_label)

        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # File list header
        list_header = QLabel("Generated Files:")
        list_header.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 5px;")
        layout.addWidget(list_header)

        # Create scrollable text area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("QTextEdit { font-family: monospace; background-color: #f5f5f5; }")

        # Build generated file list as text (filenames only)
        file_list_text = ""
        for i, file_pair in enumerate(generated_files, 1):
            # Add only if mesh file exists (None for standalone XML)
            if file_pair['mesh'] is not None:
                mesh_filename = os.path.basename(file_pair['mesh'])
                file_list_text += f"{mesh_filename}\n"

            xml_filename = os.path.basename(file_pair['xml'])
            file_list_text += f"{xml_filename}\n"
            # Add blank line between file pairs (except last set)
            if i < len(generated_files):
                file_list_text += "\n"

        text_edit.setPlainText(file_list_text)
        layout.addWidget(text_edit)

        # Create Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setFixedWidth(100)

        # Horizontal layout to center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Allow closing dialog with Enter key
        close_button.setDefault(True)

# signal_handler moved to urdf_kitchen_utils.py
# Now using setup_signal_handlers()

if __name__ == "__main__":

    # Set up Ctrl+C signal handler (using utils function)
    setup_signal_handlers()

    app = QApplication(sys.argv)
    apply_dark_theme(app)  # Custom theme with extensive widget styling

    window = MainWindow()
    window.show()

    # Get STL file path from command line arguments
    # Before VTK initialization, save to pending_stl_file for loading after VTK init
    if len(sys.argv) > 1:
        stl_file_path = sys.argv[1]
        if os.path.exists(stl_file_path):
            window.pending_stl_file = stl_file_path
        else:
            print(f"File not found: {stl_file_path}")

    # Timer for signal processing (using utils function)
    timer = setup_signal_processing_timer(app)

    try:
        sys.exit(app.exec())
    except SystemExit:
        print("Exiting application...")
