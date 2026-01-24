"""
File Name: urdf_kitchen_Assembler.py
Description: A Python script to assembling files configured with urdf_kitchen_PartsEditor.py into a URDF file.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Update.     : Jan 24, 2026
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
import traceback
import subprocess
from Qt import QtWidgets, QtCore, QtGui
from NodeGraphQt import NodeGraph, BaseNode
import vtk
from PySide6.QtWidgets import QFileDialog, QLabel
from PySide6.QtCore import QPointF, QRegularExpression, QTimer, Qt
from PySide6.QtGui import QDoubleValidator, QRegularExpressionValidator, QPalette, QColor, QImage, QPixmap
from PySide6.QtNetwork import QLocalSocket
import os
import xml.etree.ElementTree as ET
import base64
import shutil
import datetime
import numpy as np
import trimesh
import math

# Import URDF Kitchen utilities for M4 Mac compatibility
from urdf_kitchen_utils import (
    OffscreenRenderer, CameraController, MouseDragState,
    setup_signal_handlers, setup_signal_processing_timer, setup_dark_theme,
    load_mesh_to_polydata, save_polydata_to_mesh,
    calculate_inertia_tensor, calculate_inertia_with_trimesh,
    get_mesh_file_filter,
    mirror_physical_properties_y_axis, calculate_mirrored_physical_properties_from_mesh,
    mirror_inertia_tensor_left_right, mirror_center_of_mass_left_right,
    euler_to_quaternion, quaternion_to_euler, quaternion_to_matrix, format_float_no_exp,
    KitchenColorPicker, CustomColorDialog,
    ConversionUtils,
    validate_inertia_tensor,
    is_mesh_reversed_check, create_cumulative_coord
)

# Import parser classes and import functions from Importer module
from urdf_kitchen_Importer import ImporterWindow

# M4 Mac (Apple Silicon) compatibility
import platform
IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'

# Default value constants
DEFAULT_JOINT_EFFORT = 1.37  # N*m
DEFAULT_MAX_EFFORT = 1.37  # N*m
DEFAULT_JOINT_VELOCITY = 7.0  # rad/s
DEFAULT_MAX_VELOCITY = 7.48  # rad/s
DEFAULT_MARGIN = 0.01  # m
DEFAULT_ARMATURE = 0.01  # kg*m^2
DEFAULT_FRICTIONLOSS = 0.01  # N*m
DEFAULT_STIFFNESS_KP = 100.0  # N*m/rad
DEFAULT_DAMPING_KV = 1.0  # N*m*s/rad
DEFAULT_ANGLE_RANGE = 3.14159  # rad (+/- π)
DEFAULT_BASE_LINK_HEIGHT = 0.5  # Default z coordinate for MJCF base_link (m)
DEFAULT_NODE_GRID_ENABLED = True  # Enable/disable node grid snapping
DEFAULT_NODE_GRID_SIZE = 50  # Node grid size (pixels)
# Legacy constants for backward compatibility (to be removed)
DEFAULT_JOINT_LOWER = -180.0
DEFAULT_JOINT_UPPER = 180.0
DEFAULT_JOINT_FRICTION = 0.05
DEFAULT_JOINT_ACTUATION_LAG = 0.05
DEFAULT_JOINT_DAMPING = 0.18
DEFAULT_JOINT_STIFFNESS = 50.0
DEFAULT_COLOR_WHITE = [1.0, 1.0, 1.0]
DEFAULT_HIGHLIGHT_COLOR = "#80CCFF"  # Light blue (0.5, 0.8, 1.0)
DEFAULT_COLLISION_COLOR = [1.0, 0.0, 0.0, 0.5]  # Red 50% transparent (R, G, B, A)
DEFAULT_COORDS_ZERO = [0.0, 0.0, 0.0]
DEFAULT_INERTIA_ZERO = {
    'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
    'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
}
DEFAULT_ORIGIN_ZERO = {
    'xyz': [0.0, 0.0, 0.0],
    'rpy': [0.0, 0.0, 0.0]
}

# Unified button style (common across all UI)
UNIFIED_BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #5a5a5a;
        border-radius: 7px;
        padding: 5px;
        text-align: center;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
        color: #000000;
        border: 1px solid #6a6a6a;
    }
    QPushButton:pressed {
        background-color: #1a3a5a;
        border: 1px solid #2a5a8a;
    }
    QPushButton:disabled {
        background-color: #2a2a2a;
        color: #5a5a5a;
        border: 1px solid #3a3a3a;
    }
"""

# CustomColorDialog is now imported from urdf_kitchen_utils
# format_float_no_exp() is now imported from urdf_kitchen_utils

def init_node_properties(node, graph=None):
    """Initialize common properties for a node

    Args:
        node: Node to initialize
        graph: CustomNodeGraph instance (to get Settings default values)
               If None, constants are used
    """
    node.volume_value = 0.0
    node.mass_value = 0.0
    node.inertia = DEFAULT_INERTIA_ZERO.copy()
    node.inertial_origin = {
        'xyz': DEFAULT_ORIGIN_ZERO['xyz'].copy(),
        'rpy': DEFAULT_ORIGIN_ZERO['rpy'].copy()
    }
    node.visual_origin = {
        'xyz': DEFAULT_ORIGIN_ZERO['xyz'].copy(),
        'rpy': DEFAULT_ORIGIN_ZERO['rpy'].copy()
    }
    node.stl_file = None
    # Multiple colliders support: list of collider dictionaries
    # Each collider dict has: {'type': 'primitive'|'mesh', 'enabled': bool, 'data': dict, 'mesh': str, 'mesh_scale': [x,y,z], 'position': [x,y,z], 'rotation': [rx,ry,rz]}
    node.colliders = []
    node.is_mesh_reversed = False  # Flag for reversed/mirrored mesh (for MJCF export)
    node.node_color = DEFAULT_COLOR_WHITE.copy()
    node.mesh_original_color = None  # Original color extracted from mesh file (DAE/OBJ/STL)
    node.rotation_axis = 0  # 0: X, 1: Y, 2: Z
    node.body_angle = [0.0, 0.0, 0.0]  # Body initial rotation in radians [X, Y, Z]
    node.current_joint_angle = 0.0  # Current joint angle in radians (for rotation test)
    node.joint_lower = math.radians(DEFAULT_JOINT_LOWER)  # Convert from Degree to Radian and store
    node.joint_upper = math.radians(DEFAULT_JOINT_UPPER)  # Convert from Degree to Radian and store

    # Use Settings default values (when graph is specified)
    if graph is not None:
        node.joint_effort = getattr(graph, 'default_joint_effort', DEFAULT_JOINT_EFFORT)
        node.joint_damping = getattr(graph, 'default_damping_kv', DEFAULT_DAMPING_KV)
        node.joint_stiffness = getattr(graph, 'default_stiffness_kp', DEFAULT_STIFFNESS_KP)
        node.joint_velocity = getattr(graph, 'default_joint_velocity', DEFAULT_JOINT_VELOCITY)
        node.joint_margin = getattr(graph, 'default_margin', DEFAULT_MARGIN)
        node.joint_armature = getattr(graph, 'default_armature', DEFAULT_ARMATURE)
        node.joint_frictionloss = getattr(graph, 'default_frictionloss', DEFAULT_FRICTIONLOSS)
    else:
        # Use constants (for backward compatibility)
        node.joint_effort = DEFAULT_JOINT_EFFORT
        node.joint_damping = DEFAULT_DAMPING_KV
        node.joint_stiffness = DEFAULT_STIFFNESS_KP
        node.joint_velocity = DEFAULT_JOINT_VELOCITY
        node.joint_margin = DEFAULT_MARGIN
        node.joint_armature = DEFAULT_ARMATURE
        node.joint_frictionloss = DEFAULT_FRICTIONLOSS
    
    node.massless_decoration = False
    node.hide_mesh = False  # Default is mesh visible

def create_point_data(index):
    """Create point data"""
    return {
        'name': f'point_{index}',
        'type': 'fixed',
        'xyz': DEFAULT_COORDS_ZERO.copy(),
        'rpy': [0.0, 0.0, 0.0],  # Joint rotation (radians)
        'angle': [0.0, 0.0, 0.0]  # Default angle (radians)
    }

# is_mesh_reversed_check and create_cumulative_coord moved to urdf_kitchen_utils.py
# apply_dark_theme moved to urdf_kitchen_utils.py
# Now using: setup_dark_theme(app, theme='assembler')

class BaseLinkNode(BaseNode):
    """Base link node class - Special link with no parameters"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'BaseLinkNode'

    def __init__(self):
        super(BaseLinkNode, self).__init__()
        self.add_output('out')

        self.output_count = 1  # Output port is always 1 (fixed)

        # BaseLinkNode is a special link with no parameters
        # Initialize only points and cumulative coordinates
        base_point = create_point_data(1)
        base_point['name'] = 'base_link_point1'
        self.points = [base_point]
        self.cumulative_coords = [create_cumulative_coord(0)]

        # Set up double click event (does not open inspector)
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self.node_double_clicked

    def add_input(self, name='', **kwargs):
        """Prohibit adding input ports"""
        return None

    def add_output(self, name='out_1', **kwargs):
        """Do not add if output port already exists"""
        if not self.has_output(name):
            return super(BaseLinkNode, self).add_output(name, **kwargs)
        return None

    def remove_output(self):
        """Prohibit removing output port (base_link always has 1 port)"""
        return None

    def has_output(self, name):
        """Check if output port with specified name exists"""
        return name in [p.name() for p in self.output_ports()]

    def node_double_clicked(self, event):
        """Handle double click on BaseLinkNode (does not open inspector)"""
        # base_link is a special link so do not open inspector
        pass

class FooNode(BaseNode):
    """General purpose node class"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'FooNode'
    
    def __init__(self):
        super(FooNode, self).__init__()
        self.add_input('in', color=(180, 80, 0))

        self.output_count = 0

        # Initialize common properties
        # Pass None here as graph may be set later
        # Settings values are applied in create_node
        init_node_properties(self, graph=None)

        # FooNode-specific points and cumulative coordinates (start empty)
        self.points = []
        self.cumulative_coords = []

        # Add output port
        self._add_output()

        self.set_port_deletion_allowed(True)
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self.node_double_clicked

        # Initial state (no input connection) is light gray
        self.set_color(74, 84, 85)

    def _add_output(self):
        """Add output port

        Returns:
            str: Name of added port
        """
        self.output_count += 1
        port_name = f'out_{self.output_count}'

        # Add output port
        self.add_output(port_name, color=(180, 80, 0))

        # Add corresponding point data
        point_data = create_point_data(self.output_count)
        self.points.append(point_data)

        # Add cumulative coordinate
        cumulative_coord = create_cumulative_coord(self.output_count - 1)
        self.cumulative_coords.append(cumulative_coord)

        return port_name

    def remove_output(self):
        """Remove output port (can be reduced to 0)"""
        if self.output_count > 0:
            port_name = f'out_{self.output_count}'
            output_port = self.get_output(port_name)
            if output_port:
                try:
                    # Clear all connections
                    output_port.clear_connections()

                    # Remove corresponding point data
                    if len(self.points) >= self.output_count:
                        self.points.pop()

                    # Remove cumulative coordinate
                    if len(self.cumulative_coords) >= self.output_count:
                        self.cumulative_coords.pop()

                    # Delete port
                    self.delete_output(output_port)
                    self.output_count -= 1

                    # Update view
                    self.view.update()

                except Exception as e:
                    traceback.print_exc()
            else:
                pass
        else:
            pass

    def node_double_clicked(self, event):
        if hasattr(self.graph, 'show_inspector'):
            try:
                # Get graph view correctly
                graph_view = self.graph.viewer()  # Use viewer() method in NodeGraphQt

                # Convert scene coordinates to view coordinates
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)

                self.graph.show_inspector(self, screen_pos)

            except Exception as e:
                traceback.print_exc()
                # Fallback: show inspector without position
                self.graph.show_inspector(self)
        else:
            pass

class ClosedLoopJointNode(BaseNode):
    """Closed-loop joint node class - Represents ball, gearbox, screw joints"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'ClosedLoopJointNode'

    def __init__(self):
        super(ClosedLoopJointNode, self).__init__()

        # Input port (from parent link) - dark cyan
        self.add_input('in', color=(0, 180, 180))

        # Output port (to child link) - dark cyan
        self.add_output('out', color=(0, 180, 180))

        # Closed-loop joint metadata
        self.joint_name = ""
        self.joint_type = "ball"  # ball, gearbox, screw
        self.parent_link = ""
        self.child_link = ""
        self.origin_xyz = [0.0, 0.0, 0.0]
        self.origin_rpy = [0.0, 0.0, 0.0]
        self.gearbox_ratio = 1.0
        self.gearbox_reference_body = None

        # Closed-loop node displayed in special color (purple)
        self.set_color(120, 80, 140)

        # Set up double click event
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self.node_double_clicked

    def node_double_clicked(self, event):
        """Handle double click on node"""
        if hasattr(self.graph, 'show_closed_loop_inspector'):
            try:
                # Get graph view correctly
                graph_view = self.graph.viewer()

                # Convert scene coordinates to view coordinates
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)

                self.graph.show_closed_loop_inspector(self, screen_pos)

            except Exception as e:
                traceback.print_exc()
                # Fallback: show inspector without position
                self.graph.show_closed_loop_inspector(self)
        else:
            pass

class InspectorWindow(QtWidgets.QWidget):
    
    def __init__(self, parent=None, stl_viewer=None):
        super(InspectorWindow, self).__init__(parent)
        self.setWindowTitle("Node Inspector")
        self.setMinimumWidth(450)
        self.setMinimumHeight(450)
        self.resize(600, 700)  # Default size (50px increased)

        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.current_node = None
        self.stl_viewer = stl_viewer
        self.port_widgets = []

        # Debounce timer for color input fields
        self.color_update_timer = QTimer()
        self.color_update_timer.setSingleShot(True)
        self.color_update_timer.timeout.connect(self._apply_color_from_inputs)

        # Initialize UI
        self.setup_ui()

        # Set to receive keyboard focus
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def setup_ui(self):
        """Initialize UI"""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)  # Reduce overall margin
        main_layout.setContentsMargins(10, 5, 10, 5)  # Adjust top/bottom margins

        # Unified button style (use global constant)
        self.button_style = UNIFIED_BUTTON_STYLE

        # Scroll area settings
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Widget for scroll content
        scroll_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(scroll_content)
        content_layout.setSpacing(6)  # Compact section spacing
        content_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # File Controls section (placed at top)
        file_layout = QtWidgets.QHBoxLayout()
        self.import_mesh_btn = QtWidgets.QPushButton("Import Mesh")
        self.import_mesh_btn.setStyleSheet(self.button_style)
        self.load_xml_btn = QtWidgets.QPushButton("Load XML")
        self.load_xml_btn.setStyleSheet(self.button_style)
        self.load_xml_with_stl_btn = QtWidgets.QPushButton("Load XML with Mesh")
        self.load_xml_with_stl_btn.setStyleSheet(self.button_style)
        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.reload_btn.setStyleSheet(self.button_style)
        file_layout.addWidget(self.import_mesh_btn)
        file_layout.addWidget(self.load_xml_btn)
        file_layout.addWidget(self.load_xml_with_stl_btn)
        file_layout.addWidget(self.reload_btn)
        self.import_mesh_btn.clicked.connect(self.import_mesh)
        self.load_xml_btn.clicked.connect(self.load_xml)
        self.load_xml_with_stl_btn.clicked.connect(self.load_xml_with_stl)
        self.reload_btn.clicked.connect(self.reload_node_files)
        content_layout.addLayout(file_layout)

        # Node Name section (horizontal)
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Node Name:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Enter node name")
        self.name_edit.editingFinished.connect(self.update_node_name)
        name_layout.addWidget(self.name_edit)

        content_layout.addLayout(name_layout)

        # Massless Decoration checkbox and Hide Mesh checkbox
        massless_layout = QtWidgets.QHBoxLayout()
        self.massless_checkbox = QtWidgets.QCheckBox("Massless Decoration")
        self.massless_checkbox.setChecked(False)  # Default is off
        massless_layout.addWidget(self.massless_checkbox)

        massless_layout.addSpacing(20)  # Fixed space

        # Hide Mesh checkbox
        self.hide_mesh_checkbox = QtWidgets.QCheckBox("Hide Mesh")
        self.hide_mesh_checkbox.setChecked(False)  # Default is off (visible)
        massless_layout.addWidget(self.hide_mesh_checkbox)

        massless_layout.addStretch()  # Add margin on right
        content_layout.addLayout(massless_layout)

        # Connect checkbox state change handlers
        self.massless_checkbox.stateChanged.connect(self.update_massless_decoration)
        self.hide_mesh_checkbox.stateChanged.connect(self.update_hide_mesh)

        # Physical Properties section (Volume and Mass in one row)
        physics_layout = QtWidgets.QHBoxLayout()
        physics_layout.addWidget(QtWidgets.QLabel("Volume(m^3):"))
        self.volume_input = QtWidgets.QLineEdit()
        self.volume_input.setReadOnly(True)
        self.volume_input.setFixedWidth(100)
        physics_layout.addWidget(self.volume_input)

        physics_layout.addSpacing(10)  # Fixed space

        physics_layout.addWidget(QtWidgets.QLabel("Mass(kg):"))
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setValidator(QtGui.QDoubleValidator())
        self.mass_input.setFixedWidth(100)
        self.mass_input.textChanged.connect(self.update_mass)
        self.mass_input.returnPressed.connect(self.update_mass)
        physics_layout.addWidget(self.mass_input)

        physics_layout.addStretch()  # Right margin

        # Parts Editor button (right aligned)
        self.parts_editor_button = QtWidgets.QPushButton("Parts Editor")
        self.parts_editor_button.setStyleSheet(self.button_style)
        self.parts_editor_button.clicked.connect(self.open_parts_editor)
        self.parts_editor_button.setFixedWidth(110)
        physics_layout.addWidget(self.parts_editor_button)

        content_layout.addLayout(physics_layout)

        # Inertial title
        inertial_title = QtWidgets.QLabel("Inertial")
        inertial_title.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(inertial_title)
        content_layout.addSpacing(3)

        # Inertial Origin section (x, y, z, r, p, y in one row)
        origin_layout = QtWidgets.QHBoxLayout()

        # x
        x_label = QtWidgets.QLabel("x:")
        x_label.setFixedWidth(10)
        x_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        origin_layout.addWidget(x_label)
        self.inertial_x_input = QtWidgets.QLineEdit()
        self.inertial_x_input.setFixedWidth(75)
        self.inertial_x_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_x_input.setPlaceholderText("0.0")
        self.inertial_x_input.textChanged.connect(self.update_inertial_origin)
        self.inertial_x_input.returnPressed.connect(self.update_inertial_origin)
        origin_layout.addWidget(self.inertial_x_input)
        origin_layout.addSpacing(5)

        # y
        y_label = QtWidgets.QLabel("y:")
        y_label.setFixedWidth(10)
        y_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        origin_layout.addWidget(y_label)
        self.inertial_y_input = QtWidgets.QLineEdit()
        self.inertial_y_input.setFixedWidth(75)
        self.inertial_y_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_y_input.setPlaceholderText("0.0")
        self.inertial_y_input.textChanged.connect(self.update_inertial_origin)
        self.inertial_y_input.returnPressed.connect(self.update_inertial_origin)
        origin_layout.addWidget(self.inertial_y_input)
        origin_layout.addSpacing(5)

        # z
        z_label = QtWidgets.QLabel("z:")
        z_label.setFixedWidth(10)
        z_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        origin_layout.addWidget(z_label)
        self.inertial_z_input = QtWidgets.QLineEdit()
        self.inertial_z_input.setFixedWidth(75)
        self.inertial_z_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_z_input.setPlaceholderText("0.0")
        self.inertial_z_input.textChanged.connect(self.update_inertial_origin)
        self.inertial_z_input.returnPressed.connect(self.update_inertial_origin)
        origin_layout.addWidget(self.inertial_z_input)
        origin_layout.addSpacing(5)

        # r
        r_label = QtWidgets.QLabel("r:")
        r_label.setFixedWidth(10)
        r_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        origin_layout.addWidget(r_label)
        self.inertial_r_input = QtWidgets.QLineEdit()
        self.inertial_r_input.setFixedWidth(60)
        self.inertial_r_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_r_input.setPlaceholderText("0.0")
        self.inertial_r_input.textChanged.connect(self.update_inertial_origin)
        self.inertial_r_input.returnPressed.connect(self.update_inertial_origin)
        origin_layout.addWidget(self.inertial_r_input)
        origin_layout.addSpacing(5)

        # p
        p_label = QtWidgets.QLabel("p:")
        p_label.setFixedWidth(10)
        p_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        origin_layout.addWidget(p_label)
        self.inertial_p_input = QtWidgets.QLineEdit()
        self.inertial_p_input.setFixedWidth(60)
        self.inertial_p_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_p_input.setPlaceholderText("0.0")
        self.inertial_p_input.textChanged.connect(self.update_inertial_origin)
        self.inertial_p_input.returnPressed.connect(self.update_inertial_origin)
        origin_layout.addWidget(self.inertial_p_input)
        origin_layout.addSpacing(5)

        # y (yaw)
        y_rpy_label = QtWidgets.QLabel("y:")
        y_rpy_label.setFixedWidth(10)
        y_rpy_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        origin_layout.addWidget(y_rpy_label)
        self.inertial_y_rpy_input = QtWidgets.QLineEdit()
        self.inertial_y_rpy_input.setFixedWidth(60)
        self.inertial_y_rpy_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_y_rpy_input.setPlaceholderText("0.0")
        self.inertial_y_rpy_input.textChanged.connect(self.update_inertial_origin)
        self.inertial_y_rpy_input.returnPressed.connect(self.update_inertial_origin)
        origin_layout.addWidget(self.inertial_y_rpy_input)

        origin_layout.addStretch()  # Right margin
        content_layout.addLayout(origin_layout)
        content_layout.addSpacing(3)

        # Inertia Tensor section (ixx, ixy, ixz in row 1, iyy, iyz, izz in row 2)
        inertia_layout = QtWidgets.QVBoxLayout()

        # Row 1: ixx, ixy, ixz
        inertia_row1 = QtWidgets.QHBoxLayout()

        # ixx
        ixx_label = QtWidgets.QLabel("ixx:")
        ixx_label.setFixedWidth(25)
        ixx_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        inertia_row1.addWidget(ixx_label)
        self.ixx_input = QtWidgets.QLineEdit()
        self.ixx_input.setFixedWidth(140)
        self.ixx_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.ixx_input.setPlaceholderText("0.0")
        self.ixx_input.textChanged.connect(self.update_inertia)
        self.ixx_input.returnPressed.connect(self.update_inertia)
        inertia_row1.addWidget(self.ixx_input)
        inertia_row1.addSpacing(5)

        # ixy
        ixy_label = QtWidgets.QLabel("ixy:")
        ixy_label.setFixedWidth(25)
        ixy_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        inertia_row1.addWidget(ixy_label)
        self.ixy_input = QtWidgets.QLineEdit()
        self.ixy_input.setFixedWidth(140)
        self.ixy_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.ixy_input.setPlaceholderText("0.0")
        self.ixy_input.textChanged.connect(self.update_inertia)
        self.ixy_input.returnPressed.connect(self.update_inertia)
        inertia_row1.addWidget(self.ixy_input)
        inertia_row1.addSpacing(5)

        # ixz
        ixz_label = QtWidgets.QLabel("ixz:")
        ixz_label.setFixedWidth(25)
        ixz_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        inertia_row1.addWidget(ixz_label)
        self.ixz_input = QtWidgets.QLineEdit()
        self.ixz_input.setFixedWidth(140)
        self.ixz_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.ixz_input.setPlaceholderText("0.0")
        self.ixz_input.textChanged.connect(self.update_inertia)
        self.ixz_input.returnPressed.connect(self.update_inertia)
        inertia_row1.addWidget(self.ixz_input)
        inertia_row1.addStretch()  # Right margin
        inertia_layout.addLayout(inertia_row1)

        # Row 2: iyy, iyz, izz
        inertia_row2 = QtWidgets.QHBoxLayout()

        # iyy
        iyy_label = QtWidgets.QLabel("iyy:")
        iyy_label.setFixedWidth(25)
        iyy_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        inertia_row2.addWidget(iyy_label)
        self.iyy_input = QtWidgets.QLineEdit()
        self.iyy_input.setFixedWidth(140)
        self.iyy_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.iyy_input.setPlaceholderText("0.0")
        self.iyy_input.textChanged.connect(self.update_inertia)
        self.iyy_input.returnPressed.connect(self.update_inertia)
        inertia_row2.addWidget(self.iyy_input)
        inertia_row2.addSpacing(5)

        # iyz
        iyz_label = QtWidgets.QLabel("iyz:")
        iyz_label.setFixedWidth(25)
        iyz_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        inertia_row2.addWidget(iyz_label)
        self.iyz_input = QtWidgets.QLineEdit()
        self.iyz_input.setFixedWidth(140)
        self.iyz_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.iyz_input.setPlaceholderText("0.0")
        self.iyz_input.textChanged.connect(self.update_inertia)
        self.iyz_input.returnPressed.connect(self.update_inertia)
        inertia_row2.addWidget(self.iyz_input)
        inertia_row2.addSpacing(5)

        # izz
        izz_label = QtWidgets.QLabel("izz:")
        izz_label.setFixedWidth(25)
        izz_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        inertia_row2.addWidget(izz_label)
        self.izz_input = QtWidgets.QLineEdit()
        self.izz_input.setFixedWidth(140)
        self.izz_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.izz_input.setPlaceholderText("0.0")
        self.izz_input.textChanged.connect(self.update_inertia)
        self.izz_input.returnPressed.connect(self.update_inertia)
        inertia_row2.addWidget(self.izz_input)
        inertia_row2.addStretch()  # Right margin
        inertia_layout.addLayout(inertia_row2)

        content_layout.addLayout(inertia_layout)
        content_layout.addSpacing(5)

        # Inertia related buttons
        inertia_button_layout = QtWidgets.QHBoxLayout()
        inertia_button_layout.addStretch()

        # Show CoM toggle (left side)
        self.look_inertial_origin_toggle = QtWidgets.QPushButton("Show CoM")
        self.look_inertial_origin_toggle.setCheckable(True)
        self.look_inertial_origin_toggle.setStyleSheet(self.button_style)
        self.look_inertial_origin_toggle.setFixedWidth(90)
        self.look_inertial_origin_toggle.toggled.connect(self.toggle_inertial_origin_view)
        inertia_button_layout.addWidget(self.look_inertial_origin_toggle)

        # Recalc CoM button (left center)
        recalc_com_button = QtWidgets.QPushButton("Recalc CoM")
        recalc_com_button.setStyleSheet(self.button_style)
        recalc_com_button.setFixedWidth(100)
        recalc_com_button.clicked.connect(self.recalculate_com)
        inertia_button_layout.addWidget(recalc_com_button)

        # Recalc Inertia button (right center)
        recalc_inertia_button = QtWidgets.QPushButton("Recalc Inertia")
        recalc_inertia_button.setStyleSheet(self.button_style)
        recalc_inertia_button.setFixedWidth(110)
        recalc_inertia_button.clicked.connect(self.recalculate_inertia)
        inertia_button_layout.addWidget(recalc_inertia_button)

        # Zero off-diag button (right of Recalc Inertia)
        zero_offdiag_button = QtWidgets.QPushButton("Zero off-diag")
        zero_offdiag_button.setStyleSheet(self.button_style)
        zero_offdiag_button.setFixedWidth(110)
        zero_offdiag_button.clicked.connect(self.zero_off_diagonal_inertia)
        inertia_button_layout.addWidget(zero_offdiag_button)

        content_layout.addLayout(inertia_button_layout)

        # Rotation Axis section (horizontal)
        rotation_layout = QtWidgets.QHBoxLayout()
        rotation_layout.addWidget(QtWidgets.QLabel("Rotation Axis:   "))
        self.axis_group = QtWidgets.QButtonGroup(self)
        for i, axis in enumerate(['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']):  # Fixed added
            radio = QtWidgets.QRadioButton(axis)
            self.axis_group.addButton(radio, i)  # i is 0,1,2,3 (3 is Fixed)
            rotation_layout.addWidget(radio)
        rotation_layout.addStretch()  # Add margin on right
        content_layout.addLayout(rotation_layout)

        # Angle section (Body initial rotation angle, in degrees)
        angle_layout = QtWidgets.QHBoxLayout()
        angle_layout.addWidget(QtWidgets.QLabel("Angle offset (deg):"))
        angle_layout.addSpacing(10)

        # X axis rotation
        angle_layout.addWidget(QtWidgets.QLabel("X:"))
        self.angle_x_input = QtWidgets.QLineEdit()
        self.angle_x_input.setFixedWidth(60)
        self.angle_x_input.setText("0.0")
        self.angle_x_input.setToolTip("Body initial rotation around X axis (degrees)")
        self.angle_x_input.textChanged.connect(self.update_body_angle)
        angle_layout.addWidget(self.angle_x_input)

        # Y axis rotation
        angle_layout.addSpacing(5)
        angle_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.angle_y_input = QtWidgets.QLineEdit()
        self.angle_y_input.setFixedWidth(60)
        self.angle_y_input.setText("0.0")
        self.angle_y_input.setToolTip("Body initial rotation around Y axis (degrees)")
        self.angle_y_input.textChanged.connect(self.update_body_angle)
        angle_layout.addWidget(self.angle_y_input)

        # Z axis rotation
        angle_layout.addSpacing(5)
        angle_layout.addWidget(QtWidgets.QLabel("Z:"))
        self.angle_z_input = QtWidgets.QLineEdit()
        self.angle_z_input.setFixedWidth(60)
        self.angle_z_input.setText("0.0")
        self.angle_z_input.setToolTip("Body initial rotation around Z axis (degrees)")
        self.angle_z_input.textChanged.connect(self.update_body_angle)
        angle_layout.addWidget(self.angle_z_input)

        angle_layout.addStretch()
        content_layout.addLayout(angle_layout)

        # Min Angle and Max Angle (left aligned)
        angle_limits_layout = QtWidgets.QHBoxLayout()

        angle_limits_layout.addWidget(QtWidgets.QLabel("Min Angle (deg):"))
        self.lower_limit_input = QtWidgets.QLineEdit()
        self.lower_limit_input.setValidator(QDoubleValidator(-360.0, 360.0, 5))
        self.lower_limit_input.setPlaceholderText("-180")
        self.lower_limit_input.setFixedWidth(50)
        self.lower_limit_input.textChanged.connect(self.update_joint_limits_realtime)
        self.lower_limit_input.returnPressed.connect(self.set_joint_limits)
        self.lower_limit_input.returnPressed.connect(self.look_lower_limit)
        angle_limits_layout.addWidget(self.lower_limit_input)

        angle_limits_layout.addSpacing(10)  # Fixed space

        angle_limits_layout.addWidget(QtWidgets.QLabel("Max Angle (deg):"))
        self.upper_limit_input = QtWidgets.QLineEdit()
        self.upper_limit_input.setValidator(QDoubleValidator(-360.0, 360.0, 5))
        self.upper_limit_input.setPlaceholderText("180")
        self.upper_limit_input.setFixedWidth(50)
        self.upper_limit_input.textChanged.connect(self.update_joint_limits_realtime)
        self.upper_limit_input.returnPressed.connect(self.set_joint_limits)
        self.upper_limit_input.returnPressed.connect(self.look_upper_limit)
        angle_limits_layout.addWidget(self.upper_limit_input)

        angle_limits_layout.addStretch()  # Right margin
        content_layout.addLayout(angle_limits_layout)

        # Buttons (right aligned)
        joint_buttons_layout = QtWidgets.QHBoxLayout()
        joint_buttons_layout.addStretch()

        look_lower_button = QtWidgets.QPushButton("Show Min")
        look_lower_button.setStyleSheet(self.button_style)
        look_lower_button.setFixedWidth(90)
        look_lower_button.clicked.connect(self.look_lower_limit)
        joint_buttons_layout.addWidget(look_lower_button)

        look_upper_button = QtWidgets.QPushButton("Show Max")
        look_upper_button.setStyleSheet(self.button_style)
        look_upper_button.setFixedWidth(90)
        look_upper_button.clicked.connect(self.look_upper_limit)
        joint_buttons_layout.addWidget(look_upper_button)

        look_zero_button = QtWidgets.QPushButton("Show zero")
        look_zero_button.setStyleSheet(self.button_style)
        look_zero_button.setFixedWidth(90)
        look_zero_button.clicked.connect(self.look_zero_limit)
        joint_buttons_layout.addWidget(look_zero_button)

        content_layout.addLayout(joint_buttons_layout)

        # Inherit to Subnodes checkbox and Rotation Test button (right aligned)
        inherit_rotation_layout = QtWidgets.QHBoxLayout()
        inherit_rotation_layout.addStretch()

        checkbox_container = QtWidgets.QWidget()
        checkbox_container_layout = QtWidgets.QHBoxLayout(checkbox_container)
        checkbox_container_layout.setContentsMargins(-30, 0, 0, 0)
        checkbox_container_layout.setSpacing(0)

        self.follow_checkbox = QtWidgets.QCheckBox("Inherit to Subnodes")
        self.follow_checkbox.setChecked(True)
        self.follow_checkbox.setToolTip("Child nodes rotate together with this node")
        self.follow_checkbox.setStyleSheet("""
            QCheckBox {
                text-indent: -10px;
            }
        """)
        checkbox_container_layout.addWidget(self.follow_checkbox)

        inherit_rotation_layout.addWidget(checkbox_container)

        # Rotation Test button
        self.rotation_test_button = QtWidgets.QPushButton("Rotation Test")
        self.rotation_test_button.setStyleSheet(self.button_style)
        self.rotation_test_button.setFixedWidth(120)
        self.rotation_test_button.pressed.connect(self.start_rotation_test)
        self.rotation_test_button.released.connect(self.stop_rotation_test)
        inherit_rotation_layout.addWidget(self.rotation_test_button)

        content_layout.addLayout(inherit_rotation_layout)

        # Effort, Damping(kv), Stiffness(kp) (left aligned)
        joint_params_row1 = QtWidgets.QHBoxLayout()
        joint_params_row1.setSpacing(5)

        joint_params_row1.addWidget(QtWidgets.QLabel("Effort:"))
        self.effort_input = QtWidgets.QLineEdit()
        self.effort_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.effort_input.setPlaceholderText("1.37")
        self.effort_input.setMaximumWidth(60)
        self.effort_input.textChanged.connect(self.update_joint_params)
        self.effort_input.returnPressed.connect(self.update_joint_params)
        joint_params_row1.addWidget(self.effort_input)

        joint_params_row1.addWidget(QtWidgets.QLabel("Damping(kv):"))
        self.damping_input = QtWidgets.QLineEdit()
        self.damping_input.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.damping_input.setPlaceholderText("0.18")
        self.damping_input.setMaximumWidth(60)
        self.damping_input.textChanged.connect(self.update_joint_params)
        self.damping_input.returnPressed.connect(self.update_joint_params)
        joint_params_row1.addWidget(self.damping_input)

        joint_params_row1.addWidget(QtWidgets.QLabel("Stiffness(kp):"))
        self.stiffness_input = QtWidgets.QLineEdit()
        self.stiffness_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.stiffness_input.setPlaceholderText("50")
        self.stiffness_input.setMaximumWidth(60)
        self.stiffness_input.textChanged.connect(self.update_joint_params)
        self.stiffness_input.returnPressed.connect(self.update_joint_params)
        joint_params_row1.addWidget(self.stiffness_input)

        joint_params_row1.addStretch()

        content_layout.addLayout(joint_params_row1)

        # Velocity, Margin, Armature, Frictionloss (left aligned)
        joint_params_row2 = QtWidgets.QHBoxLayout()
        joint_params_row2.setSpacing(5)

        joint_params_row2.addWidget(QtWidgets.QLabel("Velocity:"))
        self.velocity_input = QtWidgets.QLineEdit()
        self.velocity_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.velocity_input.setPlaceholderText("7.0")
        self.velocity_input.setMaximumWidth(60)
        self.velocity_input.textChanged.connect(self.update_joint_params)
        self.velocity_input.returnPressed.connect(self.update_joint_params)
        joint_params_row2.addWidget(self.velocity_input)

        joint_params_row2.addWidget(QtWidgets.QLabel("Margin:"))
        self.margin_input = QtWidgets.QLineEdit()
        self.margin_input.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.margin_input.setPlaceholderText("0.0")
        self.margin_input.setMaximumWidth(60)
        self.margin_input.textChanged.connect(self.update_joint_params)
        self.margin_input.returnPressed.connect(self.update_joint_params)
        joint_params_row2.addWidget(self.margin_input)

        joint_params_row2.addWidget(QtWidgets.QLabel("Armature:"))
        self.armature_input = QtWidgets.QLineEdit()
        self.armature_input.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.armature_input.setPlaceholderText("0.0")
        self.armature_input.setMaximumWidth(60)
        self.armature_input.textChanged.connect(self.update_joint_params)
        self.armature_input.returnPressed.connect(self.update_joint_params)
        joint_params_row2.addWidget(self.armature_input)

        joint_params_row2.addWidget(QtWidgets.QLabel("Frictionloss:"))
        self.frictionloss_input = QtWidgets.QLineEdit()
        self.frictionloss_input.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.frictionloss_input.setPlaceholderText("0.0")
        self.frictionloss_input.setMaximumWidth(60)
        self.frictionloss_input.textChanged.connect(self.update_joint_params)
        self.frictionloss_input.returnPressed.connect(self.update_joint_params)
        joint_params_row2.addWidget(self.frictionloss_input)

        joint_params_row2.addStretch()

        content_layout.addLayout(joint_params_row2)

        # Color section
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(QtWidgets.QLabel("Color:"))

        # Add RGBA labels
        color_layout.addWidget(QtWidgets.QLabel("   R:"))
        for label in ['G:', 'B:', 'A:']:
            color_layout.addWidget(QtWidgets.QLabel(label))

        # Create KitchenColorPicker instance
        self.color_picker = KitchenColorPicker(
            parent_widget=self,
            initial_color=[1.0, 1.0, 1.0, 1.0],  # White with full opacity
            enable_alpha=True,  # Enable alpha for transparency
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
            # Also apply immediately when color input field changes (with debounce)
            color_input.textChanged.connect(self._on_color_input_changed)

        # Add Original button (right of Pick button)
        self.original_color_button = QtWidgets.QPushButton("Original")
        self.original_color_button.setStyleSheet(self.button_style)
        self.original_color_button.setAutoDefault(False)  # Prevent accidental triggering with Return key
        self.original_color_button.clicked.connect(self.apply_original_mesh_color)
        self.original_color_button.setFixedWidth(70)
        color_layout.addWidget(self.original_color_button)

        color_layout.addStretch()  # Fill right margin
        content_layout.addLayout(color_layout)

        # Separator line (before Colliders)
        separator_colliders = QtWidgets.QFrame()
        separator_colliders.setFrameShape(QtWidgets.QFrame.HLine)
        separator_colliders.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator_colliders)

        # Collider Mesh section (multiple colliders support)
        collider_section_layout = QtWidgets.QVBoxLayout()
        # Title row (Colliders: and Mesh Sourcer button on same line)
        collider_title_layout = QtWidgets.QHBoxLayout()
        collider_section_label = QtWidgets.QLabel("Colliders:")
        collider_title_layout.addWidget(collider_section_label)
        collider_title_layout.addStretch()

        self.collider_mesh_sourcer_button = QtWidgets.QPushButton("Mesh Sourcer")
        self.collider_mesh_sourcer_button.setStyleSheet(self.button_style)
        self.collider_mesh_sourcer_button.setFixedWidth(110)
        self.collider_mesh_sourcer_button.clicked.connect(self.open_mesh_sourcer_for_current_collider_row)
        collider_title_layout.addWidget(self.collider_mesh_sourcer_button)

        collider_section_layout.addLayout(collider_title_layout)

        # Container for collider rows (scrollable)
        self.collider_rows_container = QtWidgets.QWidget()
        self.collider_rows_layout = QtWidgets.QVBoxLayout()
        self.collider_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.collider_rows_layout.setSpacing(5)
        self.collider_rows_container.setLayout(self.collider_rows_layout)

        # Initialize collider rows list
        self.collider_rows = []

        # Colliders do not have individual scroll, extend based on row count
        # (Delegate to Inspector-wide scroll. Same behavior as Add outport)
        collider_section_layout.addWidget(self.collider_rows_container)

        # Add/Remove buttons are placed on each collider row (right of Attach)
        content_layout.addLayout(collider_section_layout)

        # Separator line (before Output Ports)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator)

        # Output Ports section
        ports_layout = QtWidgets.QVBoxLayout()
        self.ports_layout = QtWidgets.QVBoxLayout()  # Parent layout for dynamically added ports
        ports_layout.addLayout(self.ports_layout)
        content_layout.addLayout(ports_layout)

        # Initialize port widgets list
        self.port_widgets = []

        # Point Controls section (buttons right aligned)
        point_layout = QtWidgets.QHBoxLayout()
        point_layout.addStretch()  # Add margin on left for right alignment
        self.add_point_btn = QtWidgets.QPushButton("Add outport")
        self.add_point_btn.setStyleSheet(self.button_style)
        self.remove_point_btn = QtWidgets.QPushButton("Remove outport")
        self.remove_point_btn.setStyleSheet(self.button_style)
        point_layout.addWidget(self.add_point_btn)
        point_layout.addWidget(self.remove_point_btn)
        self.add_point_btn.clicked.connect(self.add_point)
        self.remove_point_btn.clicked.connect(self.remove_point)
        content_layout.addLayout(point_layout)

        # Separator line (before File Controls)
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator2)

        # Clear All and Save XML button layout
        set_button_layout = QtWidgets.QHBoxLayout()
        set_button_layout.addStretch()

        # Clear All button (for BaseLinkNode only)
        self.clear_all_button = QtWidgets.QPushButton("Clear All")
        self.clear_all_button.setStyleSheet(self.button_style)
        self.clear_all_button.clicked.connect(self.clear_all_parameters)
        self.clear_all_button.setFixedWidth(110)
        self.clear_all_button.setVisible(False)  # Hidden by default
        set_button_layout.addWidget(self.clear_all_button)

        save_xml_button = QtWidgets.QPushButton("Save XML")
        save_xml_button.setStyleSheet(self.button_style)
        save_xml_button.clicked.connect(self.save_xml)
        # Save XML button width doubled (about 2x normal button width, 220px)
        save_xml_button.setFixedWidth(220)
        set_button_layout.addWidget(save_xml_button)
        content_layout.addLayout(set_button_layout)

        # Consolidate window resize margin at bottom
        content_layout.addStretch()

        # Set content to scroll area
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # Make Node Inspector buttons look like Pick button (UNIFIED_BUTTON_STYLE)
        # (sizeHint changes, so apply style first then adjust height/width)
        self._apply_pick_like_button_style()
        # Reduce button height ratio in Node Inspector
        self._apply_compact_button_heights()
        # After style change, rescue buttons that are too narrow
        self._ensure_buttons_not_squeezed()

        # Set spacing on existing layouts
        name_layout.setSpacing(2)
        physics_layout.setSpacing(2)
        rotation_layout.setSpacing(2)
        color_layout.setSpacing(2)
        ports_layout.setSpacing(2)
        point_layout.setSpacing(2)
        file_layout.setSpacing(2)

        for line_edit in self.findChildren(QtWidgets.QLineEdit):
            line_edit.setStyleSheet("QLineEdit { padding-left: 2px; padding-top: 0px; padding-bottom: 0px; }")

    def _apply_compact_button_heights(self, ratio: float = 0.9, min_px: int = 18):
        """Reduce button height only for buttons under InspectorWindow (width respects existing settings)"""
        try:
            for btn in self.findChildren(QtWidgets.QPushButton):
                # Exclude Pick button as its shape is easily broken (keep original appearance)
                if btn.text().strip() == "Pick":
                    continue
                # Also shrink buttons with fixed height based on current value
                h = btn.sizeHint().height()
                target_h = max(min_px, int(round(h * ratio)))
                btn.setFixedHeight(target_h)
        except Exception as e:
            print(f"Warning: Failed to apply compact button heights: {e}")

    def _apply_pick_like_button_style(self):
        """Make buttons under InspectorWindow look like Pick button (UNIFIED_BUTTON_STYLE)"""
        try:
            for btn in self.findChildren(QtWidgets.QPushButton):
                btn.setStyleSheet(self.button_style)
        except Exception as e:
            print(f"Warning: Failed to apply pick-like button style: {e}")

    def _ensure_buttons_not_squeezed(self):
        """Prevent button shape from being crushed when fixed width is too narrow (match Qt default sizeHint)"""
        try:
            for btn in self.findChildren(QtWidgets.QPushButton):
                # Keep Pick button as is
                if btn.text().strip() == "Pick":
                    continue
                hint_w = btn.sizeHint().width()
                if hint_w <= 0:
                    continue

                # If setFixedWidth() is used (min==max) and smaller than hint, expand it
                min_w = btn.minimumWidth()
                max_w = btn.maximumWidth()
                if min_w > 0 and max_w > 0 and min_w == max_w:
                    if min_w < hint_w:
                        btn.setFixedWidth(hint_w)
                else:
                    # Even without fixed width, guarantee minimum width if too small
                    if min_w < hint_w:
                        btn.setMinimumWidth(hint_w)
        except Exception as e:
            print(f"Warning: Failed to ensure button widths: {e}")

    def setup_validators(self):
        """Set validators for numeric input fields"""
        try:
            # Validator for Mass input field
            mass_validator = QtGui.QDoubleValidator()
            mass_validator.setBottom(0.0)  # Prohibit negative values
            self.mass_input.setValidator(mass_validator)

            # Validator for Volume input field
            volume_validator = QtGui.QDoubleValidator()
            volume_validator.setBottom(0.0)  # Prohibit negative values
            self.volume_input.setValidator(volume_validator)

            # Validator for RGB input fields
            rgb_validator = QtGui.QDoubleValidator(
                0.0, 1.0, 3)  # 0.0 to 1.0, 3 decimal places
            for color_input in self.color_inputs:
                color_input.setValidator(rgb_validator)

            # Validator for Output Ports
            coord_validator = QtGui.QDoubleValidator()
            for port_widget in self.port_widgets:
                for input_field in port_widget.findChildren(QtWidgets.QLineEdit):
                    input_field.setValidator(coord_validator)


        except Exception as e:
            import traceback
            traceback.print_exc()

    def _on_color_changed(self, rgba_color):
        """
        KitchenColorPicker callback when color changes.

        Args:
            rgba_color: RGBA color list [r, g, b, a] in 0-1 range
        """
        try:
            if self.current_node:
                # Update node color with RGBA values
                self.current_node.node_color = rgba_color

                # Immediately apply color to 3D view
                if self.stl_viewer and hasattr(self.stl_viewer, 'apply_color_to_node'):
                    self.stl_viewer.apply_color_to_node(self.current_node)
        except Exception as e:
            print(f"Error updating node color: {str(e)}")
            traceback.print_exc()
    
    def _on_color_input_changed(self):
        """Callback when color input field changes (with debounce)"""
        # Reset timer (apply after 300ms)
        self.color_update_timer.stop()
        self.color_update_timer.start(300)

    def _apply_color_from_inputs(self):
        """Get values from color input fields and apply color (executed after debounce)"""
        if not self.current_node:
            return

        try:
            # Get RGB values (0-1 range)
            rgb_values = []
            for input_field in self.color_inputs:
                text = input_field.text().strip()
                if not text:
                    return  # Do not apply if empty
                try:
                    value = float(text)
                    rgb_values.append(max(0.0, min(1.0, value)))
                except ValueError:
                    return  # Do not apply if invalid value

            if len(rgb_values) < 3:
                return  # At least 3 values required

            # Update node color information
            if len(rgb_values) == 3:
                self.current_node.node_color = rgb_values + [1.0]  # Add Alpha=1.0
            else:
                self.current_node.node_color = rgb_values[:4]  # Maximum 4 elements

            # Immediately apply color to 3D view
            if self.stl_viewer and hasattr(self.stl_viewer, 'apply_color_to_node'):
                self.stl_viewer.apply_color_to_node(self.current_node)
        except Exception as e:
            print(f"Error applying color from inputs: {str(e)}")
            traceback.print_exc()

    def attach_collider_mesh(self):
        """Attach a separate collision mesh file or XML collider definition"""
        if not self.current_node:
            return

        # Get the directory of the visual mesh
        visual_mesh = getattr(self.current_node, 'stl_file', None)
        if visual_mesh and os.path.exists(visual_mesh):
            start_dir = os.path.dirname(visual_mesh)
        else:
            start_dir = ""

        # Open file dialog with mesh and XML filter
        file_filter = "All Collider Files (*.xml *.stl *.dae *.obj);;XML Collider (*.xml);;Mesh Files (*.stl *.dae *.obj);;STL Files (*.stl);;DAE Files (*.dae);;OBJ Files (*.obj)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Collision Mesh or XML",
            start_dir,
            file_filter
        )

        if file_path:
            # Check if it's an XML file
            if file_path.lower().endswith('.xml'):
                filename = os.path.basename(file_path)
                print(f"✓ Attached collider XML: {filename}")

                # Parse XML collider
                collider_data = self.parse_collider_xml(file_path)
                if collider_data:
                    self.collider_mesh_input.setText(f"Primitive {collider_data['type'].capitalize()}")
                    print(f"  Type: {collider_data['type']}")
                    print(f"  Position: {collider_data['position']}")
                    print(f"  Rotation: {collider_data['rotation']}")

                    # Automatically enable checkbox
                    self.collider_enabled_checkbox.setChecked(True)

                    # Update colliders list
                    if not hasattr(self.current_node, 'colliders'):
                        self.current_node.colliders = []
                    self.current_node.colliders = [{
                        'type': 'primitive',
                        'enabled': True,
                        'data': collider_data,
                        'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                        'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0]),
                        'mesh': None,
                        'mesh_scale': [1.0, 1.0, 1.0]
                    }]
                    print(f"  Collider enabled: True")

                    # Refresh collider display
                    if self.stl_viewer:
                        self.stl_viewer.refresh_collider_display()
                        print(f"  Display refreshed")
                else:
                    print(f"  ✗ Failed to parse XML collider")
                return

            # Mesh file collider
            filename = os.path.basename(file_path)

            # Determine path
            mesh_path = file_path
            if visual_mesh:
                visual_dir = os.path.dirname(visual_mesh)
                try:
                    relative_path = os.path.relpath(file_path, visual_dir)
                    mesh_path = relative_path
                    if relative_path == filename:
                        self.collider_mesh_input.setText(filename)
                    else:
                        self.collider_mesh_input.setText(relative_path)
                    print(f"✓ Attached collider mesh: {filename}")
                    print(f"  Path: {relative_path}")
                except ValueError:
                    self.collider_mesh_input.setText(filename)
                    print(f"✓ Attached collider mesh: {filename}")
                    print(f"  Path (absolute): {file_path}")
            else:
                self.collider_mesh_input.setText(filename)
                print(f"✓ Attached collider mesh: {filename}")
                print(f"  Path (absolute): {file_path}")

            # Automatically enable checkbox
            self.collider_enabled_checkbox.setChecked(True)

            # Update colliders list
            if not hasattr(self.current_node, 'colliders'):
                self.current_node.colliders = []
            self.current_node.colliders = [{
                'type': 'mesh',
                'enabled': True,
                'data': None,
                'position': [0.0, 0.0, 0.0],
                'rotation': [0.0, 0.0, 0.0],
                'mesh': mesh_path,
                'mesh_scale': [1.0, 1.0, 1.0]
            }]
            print(f"  Collider enabled: True")

            # Refresh collider display
            if self.stl_viewer:
                self.stl_viewer.refresh_collider_display()
                print(f"  Display refreshed")

    def auto_load_collider_xml(self, mesh_path):
        """Auto-load collider XML if it exists (meshname_collider.xml)"""
        if not self.current_node or not mesh_path:
            return

        # Generate expected collider XML path
        mesh_dir = os.path.dirname(mesh_path)
        mesh_basename = os.path.splitext(os.path.basename(mesh_path))[0]
        collider_xml_path = os.path.join(mesh_dir, f"{mesh_basename}_collider.xml")

        if os.path.exists(collider_xml_path):
            collider_data = self.parse_collider_xml(collider_xml_path)
            if collider_data:
                self.collider_mesh_input.setText(f"Primitive {collider_data['type'].capitalize()}")
                self.collider_enabled_checkbox.setChecked(True)
                print(f"Auto-loaded collider XML: {collider_xml_path}")
                print(f"  Type: {collider_data['type']}")

                # Update colliders list
                if not hasattr(self.current_node, 'colliders'):
                    self.current_node.colliders = []
                self.current_node.colliders = [{
                    'type': 'primitive',
                    'enabled': True,
                    'data': collider_data,
                    'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                    'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0]),
                    'mesh': None,
                    'mesh_scale': [1.0, 1.0, 1.0]
                }]

                # Refresh collider display if enabled
                if self.stl_viewer:
                    self.stl_viewer.refresh_collider_display()

    def parse_collider_xml(self, xml_path):
        """Parse collider XML file and return collider data"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            if root.tag != 'urdf_kitchen_collider':
                print(f"Invalid collider XML format: Root element should be 'urdf_kitchen_collider', got '{root.tag}'")
                return None

            collider_elem = root.find('collider')
            if collider_elem is None:
                print("No collider element found in XML")
                return None

            collider_data = {}
            collider_data['type'] = collider_elem.get('type', 'box')

            # Parse geometry
            geometry_elem = collider_elem.find('geometry')
            if geometry_elem is not None:
                collider_data['geometry'] = dict(geometry_elem.attrib)

            # Parse position
            position_elem = collider_elem.find('position')
            if position_elem is not None:
                collider_data['position'] = [
                    float(position_elem.get('x', '0.0')),
                    float(position_elem.get('y', '0.0')),
                    float(position_elem.get('z', '0.0'))
                ]
            else:
                collider_data['position'] = [0.0, 0.0, 0.0]

            # Parse rotation (in degrees)
            rotation_elem = collider_elem.find('rotation')
            if rotation_elem is not None:
                collider_data['rotation'] = [
                    float(rotation_elem.get('roll', '0.0')),
                    float(rotation_elem.get('pitch', '0.0')),
                    float(rotation_elem.get('yaw', '0.0'))
                ]
            else:
                collider_data['rotation'] = [0.0, 0.0, 0.0]

            return collider_data

        except Exception as e:
            print(f"Error parsing collider XML: {str(e)}")
            return None

    def create_collider_row(self, collider_index=0, collider_data=None):
        """Create a single collider row UI
        
        Args:
            collider_index: Index of the collider in the list
            collider_data: Dictionary with collider data, or None for new collider
        Returns:
            Dictionary containing the row widgets
        """
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(5)
        
        # Checkbox
        enabled_checkbox = QtWidgets.QCheckBox()
        enabled_checkbox.setChecked(collider_data.get('enabled', False) if collider_data else False)
        enabled_checkbox.stateChanged.connect(lambda: self.update_collider_from_row(collider_index))
        row_layout.addWidget(enabled_checkbox)
        # Add 10px spacing between checkbox and input
        row_layout.addSpacing(10)

        # Input field
        mesh_input = QtWidgets.QLineEdit()
        mesh_input.setReadOnly(True)
        # Remove border
        mesh_input.setStyleSheet("QLineEdit { border: none; }")
        
        # Set palette for placeholder text
        palette = mesh_input.palette()
        palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#cccccc"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#000000"))
        mesh_input.setPalette(palette)
        
        # Set initial text
        if collider_data:
            if collider_data.get('type') == 'primitive':
                data = collider_data.get('data', {})
                primitive_type = data.get('type', 'unknown').capitalize()
                mesh_input.setText(f"Primitive {primitive_type}")
            elif collider_data.get('type') == 'mesh':
                mesh = collider_data.get('mesh', '')
                if mesh:
                    if os.path.isabs(mesh):
                        mesh_input.setText(os.path.basename(mesh))
                    else:
                        mesh_input.setText(mesh)
                else:
                    # Show input field with "Not set" even when collider is not set
                    mesh_input.setText("Not set")
        else:
            # Show input field with "Not set" even when collider is not set
            mesh_input.setText("Not set")

        # Disable input field when enabled=false (also dim appearance)
        is_enabled = collider_data.get('enabled', False) if collider_data else False
        mesh_input.setEnabled(is_enabled)
        
        row_layout.addWidget(mesh_input)
        
        # Attach button
        attach_button = QtWidgets.QPushButton("Attach")
        attach_button.setStyleSheet(self.button_style)
        attach_button.setFixedWidth(60)
        attach_button.clicked.connect(lambda: self.attach_collider_mesh_to_row(collider_index))
        row_layout.addWidget(attach_button)

        # [+] / [-] buttons (row-level)
        add_button = QtWidgets.QPushButton("+")
        add_button.setStyleSheet(self.button_style)
        add_button.setFixedWidth(30)
        add_button.clicked.connect(lambda: self.add_collider_row_at(collider_index))
        row_layout.addWidget(add_button)

        remove_button = QtWidgets.QPushButton("-")
        remove_button.setStyleSheet(self.button_style)
        remove_button.setFixedWidth(30)
        remove_button.clicked.connect(lambda: self.remove_collider_row_at(collider_index))
        row_layout.addWidget(remove_button)
        
        row_widget.setLayout(row_layout)
        
        return {
            'widget': row_widget,
            'enabled_checkbox': enabled_checkbox,
            'mesh_input': mesh_input,
            'attach_button': attach_button,
            'add_button': add_button,
            'remove_button': remove_button,
            'index': collider_index
        }

    def open_mesh_sourcer_for_current_collider_row(self):
        """Called from Mesh Sourcer button in Colliders section title row"""
        # Since there is no "selected row" in current UI, target the first row (future: row selection UI can be added)
        idx = 0
        try:
            if self.collider_rows:
                idx = self.collider_rows[0].get('index', 0)
        except Exception:
            idx = 0
        self.open_mesh_sourcer_for_row(idx)

    def add_collider_row_at(self, after_index: int):
        """Add collider row after specified row"""
        if not self.current_node:
            return
        if not hasattr(self.current_node, 'colliders'):
            self.current_node.colliders = []

        insert_index = max(0, min(after_index + 1, len(self.current_node.colliders)))
        new_collider = {
            'type': None,
            'enabled': False,
            'data': None,
            'mesh': None,
            'mesh_scale': [1.0, 1.0, 1.0],
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0]
        }
        self.current_node.colliders.insert(insert_index, new_collider)
        self.update_collider_rows(self.current_node)

    def remove_collider_row_at(self, index: int):
        """Remove collider row at specified index (keep at least 1 row)"""
        if not self.current_node or not hasattr(self.current_node, 'colliders'):
            return
        if len(self.current_node.colliders) <= 1:
            # Always keep at least 1 row (for "Not set" row)
            self.current_node.colliders[0] = {
                'type': None,
                'enabled': False,
                'data': None,
                'mesh': None,
                'mesh_scale': [1.0, 1.0, 1.0]
            }
            self.update_collider_rows(self.current_node)
            return
        if 0 <= index < len(self.current_node.colliders):
            self.current_node.colliders.pop(index)
            self.update_collider_rows(self.current_node)
    
    def add_collider_row(self):
        """Add a new collider row"""
        if not self.current_node:
            return
        
        # Initialize colliders list if not exists
        if not hasattr(self.current_node, 'colliders'):
            self.current_node.colliders = []
        
        # Add new empty collider
        new_collider = {
            'type': None,
            'enabled': False,
            'data': None,
            'mesh': None,
            'mesh_scale': [1.0, 1.0, 1.0],
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0]
        }
        self.current_node.colliders.append(new_collider)
        
        # Create UI row
        row_index = len(self.current_node.colliders) - 1
        row_data = self.create_collider_row(row_index, new_collider)
        self.collider_rows.append(row_data)
        self.collider_rows_layout.addWidget(row_data['widget'])
        
        print(f"Added collider row {row_index}")
    
    def remove_collider_row(self):
        """Remove the last collider row"""
        if not self.current_node or not hasattr(self.current_node, 'colliders'):
            return
        
        if len(self.current_node.colliders) == 0:
            return
        
        # Remove from node
        self.current_node.colliders.pop()
        
        # Remove UI row
        if self.collider_rows:
            row_data = self.collider_rows.pop()
            self.collider_rows_layout.removeWidget(row_data['widget'])
            row_data['widget'].deleteLater()
        
        # Update indices
        for i, row_data in enumerate(self.collider_rows):
            row_data['index'] = i
        
        print(f"Removed collider row, {len(self.current_node.colliders)} remaining")
        
        # Refresh collider display
        if self.stl_viewer:
            self.stl_viewer.refresh_collider_display()
    
    def update_collider_from_row(self, collider_index):
        """Update collider enabled state from row checkbox"""
        if not self.current_node or not hasattr(self.current_node, 'colliders'):
            return
        
        if collider_index >= len(self.current_node.colliders):
            return
        
        row_data = self.collider_rows[collider_index]
        is_enabled = row_data['enabled_checkbox'].isChecked()
        self.current_node.colliders[collider_index]['enabled'] = is_enabled

        # UI: Keep input field even when unchecked, but disable and show "Not set"
        mesh_input = row_data.get('mesh_input')
        if mesh_input is not None:
            mesh_input.setEnabled(is_enabled)
            if not is_enabled:
                c = self.current_node.colliders[collider_index]
                has_any_value = bool(c.get('mesh')) or bool(c.get('data')) or bool(c.get('type'))
                if not has_any_value:
                    mesh_input.setText("Not set")
            else:
                # Clear "Not set" text when enabled to treat as visual mesh
                if mesh_input.text().strip() == "Not set":
                    mesh_input.setText("")
        
        # Refresh collider display
        if self.stl_viewer:
            self.stl_viewer.refresh_collider_display()
    
    def attach_collider_mesh_to_row(self, collider_index):
        """Attach collider mesh to a specific row"""
        if not self.current_node:
            return
        
        # Initialize colliders list if not exists
        if not hasattr(self.current_node, 'colliders'):
            self.current_node.colliders = []
        
        # Ensure collider exists at this index
        while len(self.current_node.colliders) <= collider_index:
            self.current_node.colliders.append({
                'type': None,
                'enabled': False,
                'data': None,
                'mesh': None,
                'mesh_scale': [1.0, 1.0, 1.0],
                'position': [0.0, 0.0, 0.0],
                'rotation': [0.0, 0.0, 0.0]
            })
        
        # Get the directory of the visual mesh
        visual_mesh = getattr(self.current_node, 'stl_file', None)
        if visual_mesh and os.path.exists(visual_mesh):
            start_dir = os.path.dirname(visual_mesh)
        else:
            start_dir = ""
        
        # Open file dialog with mesh and XML filter
        file_filter = "All Collider Files (*.xml *.stl *.dae *.obj);;XML Collider (*.xml);;Mesh Files (*.stl *.dae *.obj);;STL Files (*.stl);;DAE Files (*.dae);;OBJ Files (*.obj)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Collision Mesh or XML",
            start_dir,
            file_filter
        )
        
        if file_path:
            collider = self.current_node.colliders[collider_index]
            row_data = self.collider_rows[collider_index]
            
            # Check if it's an XML file
            if file_path.lower().endswith('.xml'):
                filename = os.path.basename(file_path)
                print(f"✓ Attached collider XML to row {collider_index}: {filename}")
                
                # Parse XML collider
                collider_data = self.parse_collider_xml(file_path)
                if collider_data:
                    collider['type'] = 'primitive'
                    collider['data'] = collider_data
                    collider['enabled'] = True
                    # Set position and rotation directly under collider (for display and export)
                    collider['position'] = collider_data.get('position', [0.0, 0.0, 0.0])
                    collider['rotation'] = collider_data.get('rotation', [0.0, 0.0, 0.0])
                    row_data['mesh_input'].setText(f"Primitive {collider_data['type'].capitalize()}")
                    row_data['enabled_checkbox'].setChecked(True)
                    print(f"  Type: {collider_data['type']}")
                    print(f"  Position: {collider['position']}")
                    print(f"  Rotation: {collider['rotation']}")
                    
                    # Refresh collider display
                    if self.stl_viewer:
                        self.stl_viewer.refresh_collider_display()
                else:
                    print(f"  ✗ Failed to parse XML collider")
                return
            
            # Mesh file collider
            collider['type'] = 'mesh'
            filename = os.path.basename(file_path)
            
            # Initialize position and rotation for mesh collider (default to origin)
            if 'position' not in collider:
                collider['position'] = [0.0, 0.0, 0.0]
            if 'rotation' not in collider:
                collider['rotation'] = [0.0, 0.0, 0.0]
            
            # Save and display path
            if visual_mesh:
                visual_dir = os.path.dirname(visual_mesh)
                try:
                    # Try relative path
                    relative_path = os.path.relpath(file_path, visual_dir)
                    collider['mesh'] = relative_path
                    if relative_path == filename:
                        row_data['mesh_input'].setText(filename)
                    else:
                        row_data['mesh_input'].setText(relative_path)
                    print(f"✓ Attached collider mesh to row {collider_index}: {filename}")
                    print(f"  Path: {relative_path}")
                except ValueError:
                    # Use absolute path for different drives
                    collider['mesh'] = file_path
                    row_data['mesh_input'].setText(filename)
                    print(f"✓ Attached collider mesh to row {collider_index}: {filename}")
                    print(f"  Path (absolute): {file_path}")
            else:
                # Use absolute path if no visual mesh
                collider['mesh'] = file_path
                row_data['mesh_input'].setText(filename)
                print(f"✓ Attached collider mesh to row {collider_index}: {filename}")
                print(f"  Path (absolute): {file_path}")
            
            collider['enabled'] = True
            row_data['enabled_checkbox'].setChecked(True)
            print(f"  Collider enabled: True")
            print(f"  Collider type: mesh")
            print(f"  Position: {collider['position']}")
            print(f"  Rotation: {collider['rotation']}")
            
            # Refresh collider display
            if self.stl_viewer:
                self.stl_viewer.refresh_collider_display()
    
    def open_mesh_sourcer_for_row(self, collider_index):
        """Open mesh sourcer for a specific collider row"""
        # TODO: Implement mesh sourcer for specific row
        print(f"Mesh Sourcer for row {collider_index} (not yet implemented)")
    
    def update_collider_rows(self, node):
        """Update collider rows UI from node data"""
        # Clear existing rows
        for row_data in self.collider_rows:
            self.collider_rows_layout.removeWidget(row_data['widget'])
            row_data['widget'].deleteLater()
        self.collider_rows.clear()
        
        # Initialize colliders list if not exists
        if not hasattr(node, 'colliders'):
            node.colliders = []

        # Show at least one "Not set" row in UI even when there are no colliders
        if len(node.colliders) == 0:
            node.colliders.append({
                'type': None,
                'enabled': False,
                'data': None,
                'mesh': None,
                'mesh_scale': [1.0, 1.0, 1.0],
                'position': [0.0, 0.0, 0.0],
                'rotation': [0.0, 0.0, 0.0]
            })
        
        # Create UI rows for each collider
        for i, collider in enumerate(node.colliders):
            row_data = self.create_collider_row(i, collider)
            self.collider_rows.append(row_data)
            self.collider_rows_layout.addWidget(row_data['widget'])
    
    def open_mesh_sourcer(self):
        """Open MeshSourcer with the current node's mesh file and collider information"""
        if not self.current_node:
            print("No node selected")
            return

        # Get the current mesh file path
        mesh_file = getattr(self.current_node, 'stl_file', None)
        if not mesh_file:
            print("No mesh file loaded in current node")
            return

        # Check if file exists
        if not os.path.exists(mesh_file):
            print(f"Mesh file not found: {mesh_file}")
            return

        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mesh_sourcer_path = os.path.join(script_dir, "urdf_kitchen_MeshSourcer.py")

            # Check if MeshSourcer script exists
            if not os.path.exists(mesh_sourcer_path):
                print(f"MeshSourcer not found at: {mesh_sourcer_path}")
                return

            # Prepare command arguments
            cmd_args = [sys.executable, mesh_sourcer_path, mesh_file]

            # Check if node has collider primitive information from colliders list
            if hasattr(self.current_node, 'colliders') and self.current_node.colliders:
                for collider in self.current_node.colliders:
                    if collider.get('type') == 'primitive' and collider.get('data'):
                        # Serialize collider data to JSON and pass as command line argument
                        import json
                        collider_json = json.dumps({
                            'type': 'primitive',
                            'data': collider['data']
                        })
                        cmd_args.append(collider_json)
                        print(f"Passing collider information: type={collider['data'].get('type', 'unknown')}")
                        break  # Only pass the first primitive collider

            # Launch MeshSourcer as a separate process with the mesh file path and optional collider info
            subprocess.Popen(cmd_args)

            print(f"Launched MeshSourcer with: {mesh_file}")
        except Exception as e:
            print(f"Error launching MeshSourcer: {e}")
            import traceback
            traceback.print_exc()

    def update_port_coordinate(self, port_index, coord_index, value):
        """Update port coordinate"""
        try:
            if self.current_node and hasattr(self.current_node, 'points'):
                if 0 <= port_index < len(self.current_node.points):
                    try:
                        new_value = float(value)
                        self.current_node.points[port_index]['xyz'][coord_index] = new_value
                        print(
                            f"Updated port {port_index+1} coordinate {coord_index} to {new_value}")
                    except ValueError:
                        print("Invalid coordinate value")
        except Exception as e:
            print(f"Error updating coordinate: {str(e)}")

    def update_port_angle(self, port_index, angle_index, value):
        """Update port angle (UI in degrees, stored internally in radians)"""
        try:
            if self.current_node and hasattr(self.current_node, 'points'):
                if 0 <= port_index < len(self.current_node.points):
                    try:
                        new_value_deg = float(value)
                        # Convert from degrees to radians and store
                        new_value_rad = math.radians(new_value_deg)
                        # Initialize angle key if not exists
                        if 'angle' not in self.current_node.points[port_index]:
                            self.current_node.points[port_index]['angle'] = [0.0, 0.0, 0.0]
                        self.current_node.points[port_index]['angle'][angle_index] = new_value_rad
                        print(f"Updated port {port_index+1} angle {angle_index} to {new_value_deg} degrees ({new_value_rad} rad)")

                        # NOTE: Do not sync with body_angle
                        # body_angle is for MJCF ref attribute (reference angle) only
                        # point['angle'] is for joint origin rotation (origin rpy)
                        # These have different meanings, so manage them separately

                        # Update 3D view (update child node rotations)
                        if self.stl_viewer:
                            self.stl_viewer.update_3d_view()
                    except ValueError:
                        print("Invalid angle value")
        except Exception as e:
            print(f"Error updating angle: {str(e)}")

    def _set_inertial_origin_ui(self, xyz, rpy):
        """Set values to Inertial Origin UI input fields (display with high precision, no exponential notation)"""
        self.inertial_x_input.setText(format_float_no_exp(xyz[0]))
        self.inertial_y_input.setText(format_float_no_exp(xyz[1]))
        self.inertial_z_input.setText(format_float_no_exp(xyz[2]))
        self.inertial_r_input.setText(format_float_no_exp(rpy[0]))
        self.inertial_p_input.setText(format_float_no_exp(rpy[1]))
        self.inertial_y_rpy_input.setText(format_float_no_exp(rpy[2]))

    def _set_inertia_ui(self, inertia_dict):
        """Set values to Inertia Tensor UI input fields (display with high precision, no exponential notation)"""
        self.ixx_input.setText(format_float_no_exp(inertia_dict.get('ixx', 0.0)))
        self.ixy_input.setText(format_float_no_exp(inertia_dict.get('ixy', 0.0)))
        self.ixz_input.setText(format_float_no_exp(inertia_dict.get('ixz', 0.0)))
        self.iyy_input.setText(format_float_no_exp(inertia_dict.get('iyy', 0.0)))
        self.iyz_input.setText(format_float_no_exp(inertia_dict.get('iyz', 0.0)))
        self.izz_input.setText(format_float_no_exp(inertia_dict.get('izz', 0.0)))

    def _set_color_ui(self, color_values):
        """Set values to color UI input fields (RGB or RGBA)"""
        # Accept RGB (3 elements) or RGBA (4 elements)
        num_values = min(len(color_values), len(self.color_inputs))
        for i in range(num_values):
            self.color_inputs[i].setText(f"{color_values[i]:.3f}")

        # Set Alpha=1.0 for RGB
        if len(color_values) == 3 and len(self.color_inputs) >= 4:
            self.color_inputs[3].setText("1.0")

    def update_info(self, node):
        """Update node information"""
        self.current_node = node

        try:
            # Node Name
            self.name_edit.setText(node.name())

            # Volume & Mass (high precision, no exponential notation)
            if hasattr(node, 'volume_value'):
                self.volume_input.setText(format_float_no_exp(node.volume_value))

            if hasattr(node, 'mass_value'):
                self.mass_input.setText(format_float_no_exp(node.mass_value))

            # Set Inertia
            if hasattr(node, 'inertia') and isinstance(node.inertia, dict):
                self._set_inertia_ui(node.inertia)
            else:
                # Set default values
                node.inertia = DEFAULT_INERTIA_ZERO.copy()
                self._set_inertia_ui(node.inertia)

            # Set Inertial Origin
            if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                xyz = node.inertial_origin.get('xyz', DEFAULT_COORDS_ZERO)
                rpy = node.inertial_origin.get('rpy', DEFAULT_COORDS_ZERO)
                self._set_inertial_origin_ui(xyz, rpy)
            else:
                # Set default values
                node.inertial_origin = DEFAULT_ORIGIN_ZERO.copy()
                node.inertial_origin['xyz'] = DEFAULT_COORDS_ZERO.copy()
                node.inertial_origin['rpy'] = DEFAULT_COORDS_ZERO.copy()
                self._set_inertial_origin_ui(node.inertial_origin['xyz'], node.inertial_origin['rpy'])

            # Rotation Axis - check node's rotation_axis attribute and set
            if hasattr(node, 'rotation_axis'):
                axis_button = self.axis_group.button(node.rotation_axis)
                if axis_button:
                    axis_button.setChecked(True)
            else:
                # Default to X axis
                node.rotation_axis = 0
                if self.axis_group.button(0):
                    self.axis_group.button(0).setChecked(True)

            # Set Body Angle (display in degrees)
            # Get and apply Ang value from connected parent node's outport
            parent_angle = None
            if hasattr(node, 'graph'):
                graph = node.graph
                # Find parent from node's input port
                for input_port in node.input_ports():
                    connected_ports = input_port.connected_ports()
                    if connected_ports:
                        parent_node = connected_ports[0].node()
                        parent_port_name = connected_ports[0].name()

                        # Calculate point index from port name (out_1 -> 0, out_2 -> 1, etc.)
                        point_index = 0  # Default
                        if parent_port_name.startswith('out_'):
                            try:
                                port_num = int(parent_port_name.split('_')[1])
                                point_index = port_num - 1
                            except (ValueError, IndexError):
                                pass
                        elif parent_port_name == 'out':
                            point_index = 0

                        # Get angle from parent node's points
                        if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                            point_data = parent_node.points[point_index]
                            parent_angle = point_data.get('angle', [0.0, 0.0, 0.0])
                        break

            # Use parent node's angle value if available, otherwise use node.body_angle
            if parent_angle is not None:
                # NOTE: Do not overwrite if body_angle is already set (e.g., MJCF ref)
                existing_body_angle = getattr(node, 'body_angle', [0.0, 0.0, 0.0])
                if not any(a != 0.0 for a in existing_body_angle):
                    # Sync only if body_angle is not set
                    node.body_angle = list(parent_angle)
                # Display parent node's angle value in degrees in UI
                self.angle_x_input.setText(str(round(math.degrees(parent_angle[0]), 2)))
                self.angle_y_input.setText(str(round(math.degrees(parent_angle[1]), 2)))
                self.angle_z_input.setText(str(round(math.degrees(parent_angle[2]), 2)))
            elif hasattr(node, 'body_angle'):
                # Display in degrees in UI (convert from radians)
                self.angle_x_input.setText(str(round(math.degrees(node.body_angle[0]), 2)))
                self.angle_y_input.setText(str(round(math.degrees(node.body_angle[1]), 2)))
                self.angle_z_input.setText(str(round(math.degrees(node.body_angle[2]), 2)))
            else:
                # Set default values (0 in radians)
                node.body_angle = [0.0, 0.0, 0.0]
                self.angle_x_input.setText("0.0")
                self.angle_y_input.setText("0.0")
                self.angle_z_input.setText("0.0")

            # Set Massless Decoration state
            if hasattr(node, 'massless_decoration'):
                self.massless_checkbox.setChecked(node.massless_decoration)
            else:
                node.massless_decoration = False
                self.massless_checkbox.setChecked(False)

            # Set Hide Mesh state
            if hasattr(node, 'hide_mesh'):
                self.hide_mesh_checkbox.setChecked(node.hide_mesh)
            else:
                node.hide_mesh = False
                self.hide_mesh_checkbox.setChecked(False)

            # Set Joint Limits (convert from Radian to Degree for display)
            if hasattr(node, 'joint_lower'):
                # Convert Radian to Degree (round to 2 decimal places)
                self.lower_limit_input.setText(str(round(math.degrees(node.joint_lower), 2)))
            else:
                # DEFAULT_JOINT_LOWER is already in Degree
                node.joint_lower = math.radians(DEFAULT_JOINT_LOWER)
                self.lower_limit_input.setText(str(DEFAULT_JOINT_LOWER))

            if hasattr(node, 'joint_upper'):
                # Convert Radian to Degree (round to 2 decimal places)
                self.upper_limit_input.setText(str(round(math.degrees(node.joint_upper), 2)))
            else:
                # DEFAULT_JOINT_UPPER is already in Degree
                node.joint_upper = math.radians(DEFAULT_JOINT_UPPER)
                self.upper_limit_input.setText(str(DEFAULT_JOINT_UPPER))

            if hasattr(node, 'joint_effort'):
                self.effort_input.setText(str(node.joint_effort))
            else:
                # Use graph default value
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_effort'):
                    node.joint_effort = node.graph.default_joint_effort
                else:
                    node.joint_effort = DEFAULT_JOINT_EFFORT
                self.effort_input.setText(str(node.joint_effort))

            if hasattr(node, 'joint_velocity'):
                self.velocity_input.setText(str(node.joint_velocity))
            else:
                # Use graph default value
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_velocity'):
                    node.joint_velocity = node.graph.default_joint_velocity
                else:
                    node.joint_velocity = DEFAULT_JOINT_VELOCITY
                self.velocity_input.setText(str(node.joint_velocity))

            # Set Damping
            if hasattr(node, 'joint_damping'):
                self.damping_input.setText(str(node.joint_damping))
            else:
                node.joint_damping = DEFAULT_DAMPING_KV
                self.damping_input.setText(str(node.joint_damping))

            # Set Stiffness
            if hasattr(node, 'joint_stiffness'):
                self.stiffness_input.setText(str(node.joint_stiffness))
            else:
                node.joint_stiffness = DEFAULT_STIFFNESS_KP
                self.stiffness_input.setText(str(node.joint_stiffness))

            # Set Margin
            if hasattr(node, 'joint_margin'):
                self.margin_input.setText(str(node.joint_margin))
            else:
                node.joint_margin = DEFAULT_MARGIN
                self.margin_input.setText(str(node.joint_margin))

            # Set Armature
            if hasattr(node, 'joint_armature'):
                self.armature_input.setText(str(node.joint_armature))
            else:
                node.joint_armature = DEFAULT_ARMATURE
                self.armature_input.setText(str(node.joint_armature))

            # Set Frictionloss
            if hasattr(node, 'joint_frictionloss'):
                self.frictionloss_input.setText(str(node.joint_frictionloss))
            else:
                node.joint_frictionloss = DEFAULT_FRICTIONLOSS
                self.frictionloss_input.setText(str(node.joint_frictionloss))

            # Color settings - check node's node_color attribute and set
            if hasattr(node, 'node_color') and node.node_color:
                self._set_color_ui(node.node_color)

                # Update color sample chip
                rgb_display = [int(v * 255) for v in node.node_color[:3]]
                self.color_sample.setStyleSheet(
                    f"background-color: rgb({rgb_display[0]},{rgb_display[1]},{rgb_display[2]}); "
                    f"border: 1px solid black;"
                )
                # Apply color to STL model
                self.apply_color_to_stl()
            else:
                # Set default color (white)
                node.node_color = DEFAULT_COLOR_WHITE.copy()
                self._set_color_ui(node.node_color)
                self.color_sample.setStyleSheet(
                    "background-color: rgb(255,255,255); border: 1px solid black;"
                )

            # Collider Mesh settings
            # Update collider rows
            self.update_collider_rows(node)

            # Connect signal to update rotation axis selection
            for button in self.axis_group.buttons():
                button.clicked.connect(lambda checked, btn=button: self.update_rotation_axis(btn))

            # Output Ports
            self.update_output_ports(node)

            # Set radio button event handler
            self.axis_group.buttonClicked.connect(self.on_axis_selection_changed)

            # Set validators
            self.setup_validators()

            # Refresh collider display after all data is loaded
            # This ensures colliders display correctly when Node Inspector is reopened
            if self.stl_viewer:
                self.stl_viewer.refresh_collider_display()

        except Exception as e:
            print(f"Error updating inspector info: {str(e)}")
            traceback.print_exc()

    def update_rotation_axis(self, button):
        """Handle rotation axis selection change"""
        if self.current_node:
            self.current_node.rotation_axis = self.axis_group.id(button)

    def on_axis_selection_changed(self, button):
        """Event handler when rotation axis selection changes"""
        if self.current_node:
            # Update rotation axis
            axis_id = self.axis_group.id(button)
            self.current_node.rotation_axis = axis_id

            # Determine and display axis type
            axis_types = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']
            if 0 <= axis_id < len(axis_types):
                print(f"Rotation axis changed to: {axis_types[axis_id]}")

            # Update entire graph layout and reapply all transforms including point_angle
            if hasattr(self, 'graph') and self.graph:
                self.graph.update_node_layout()
            elif self.stl_viewer:
                # Render directly if graph is not accessible
                self.stl_viewer.render_to_image()

    def update_node_name(self):
        """Update node name"""
        if self.current_node:
            new_name = self.name_edit.text()
            old_name = self.current_node.name()
            if new_name != old_name:
                self.current_node.set_name(new_name)

    def add_point(self):
        """Add point"""
        if self.current_node and hasattr(self.current_node, '_add_output'):
            new_port_name = self.current_node._add_output()
            if new_port_name:
                self.update_info(self.current_node)

    def remove_point(self):
        """Remove point"""
        if self.current_node and hasattr(self.current_node, 'remove_output'):
            self.current_node.remove_output()
            self.update_info(self.current_node)

    def import_mesh(self):
        """Import mesh file"""
        if self.current_node:
            file_filter = get_mesh_file_filter(trimesh_available=True)
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Mesh File", "", file_filter)
            if file_name:
                self.current_node.stl_file = file_name
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                    # Update 3D view
                    self.stl_viewer.render_to_image()

                # Auto-load collider XML if exists
                self.auto_load_collider_xml(file_name)

                # Execute same effect as Recalc Positions
                if hasattr(self.current_node, 'graph') and self.current_node.graph:
                    self.current_node.graph.recalculate_all_positions()

    def closeEvent(self, event):
        """Handle window close event"""
        try:
            # Clear highlight
            if self.stl_viewer:
                self.stl_viewer.clear_highlight()

            # Explicitly delete all widgets
            for widget in self.findChildren(QtWidgets.QWidget):
                if widget is not self:
                    widget.setParent(None)
                    widget.deleteLater()

            # Clear references
            self.current_node = None
            self.stl_viewer = None
            self.port_widgets.clear()

            # Accept event
            event.accept()

        except Exception as e:
            print(f"Error in closeEvent: {str(e)}")
            event.accept()

    def _load_xml_common_properties(self, root, xml_dir):
        """Load common properties from XML file (shared by load_xml and load_xml_with_stl)

        Args:
            root: XML root element
            xml_dir: Directory containing the XML file
        """
        # Get and set link name
        link_elem = root.find('link')
        if link_elem is not None:
            link_name = link_elem.get('name')
            if link_name:
                self.current_node.set_name(link_name)
                self.name_edit.setText(link_name)

            # Set physical properties
            inertial_elem = link_elem.find('inertial')
            if inertial_elem is not None:
                # Set volume
                volume_elem = inertial_elem.find('volume')
                if volume_elem is not None:
                    volume = float(volume_elem.get('value', '0.0'))
                    self.current_node.volume_value = volume
                    self.volume_input.setText(format_float_no_exp(volume))

                # Set mass
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value', '0.0'))
                    self.current_node.mass_value = mass
                    self.mass_input.setText(format_float_no_exp(mass))

                # Set Inertial Origin
                origin_elem = inertial_elem.find('origin')
                if origin_elem is not None:
                    origin_xyz = origin_elem.get('xyz', '0 0 0').split()
                    origin_rpy = origin_elem.get('rpy', '0 0 0').split()
                    self.current_node.inertial_origin = {
                        'xyz': [float(x) for x in origin_xyz],
                        'rpy': [float(x) for x in origin_rpy]
                    }
                    # Update UI
                    self._set_inertial_origin_ui(
                        self.current_node.inertial_origin['xyz'],
                        self.current_node.inertial_origin['rpy']
                    )

                # Set inertia tensor
                inertia_elem = inertial_elem.find('inertia')
                if inertia_elem is not None:
                    self.current_node.inertia = {
                        'ixx': float(inertia_elem.get('ixx', '0')),
                        'ixy': float(inertia_elem.get('ixy', '0')),
                        'ixz': float(inertia_elem.get('ixz', '0')),
                        'iyy': float(inertia_elem.get('iyy', '0')),
                        'iyz': float(inertia_elem.get('iyz', '0')),
                        'izz': float(inertia_elem.get('izz', '0'))
                    }
                    # Update UI
                    self._set_inertia_ui(self.current_node.inertia)

            # Set Center of Mass
            center_of_mass_elem = link_elem.find('center_of_mass')
            if center_of_mass_elem is not None:
                com_xyz = center_of_mass_elem.text.strip().split()

        # Process color information
        material_elem = root.find('.//material/color')
        if material_elem is not None:
            rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
            rgba_values = [float(x) for x in rgba[:4]]
            self.current_node.node_color = rgba_values
            self._set_color_ui(rgba_values)

        # Load massless_decoration and hide_mesh
        massless_elem = root.find('massless_decoration')
        if massless_elem is not None:
            try:
                massless_value = massless_elem.text.lower() == 'true' if massless_elem.text else False
                self.current_node.massless_decoration = massless_value
                if hasattr(self, 'massless_checkbox'):
                    self.massless_checkbox.setChecked(massless_value)
            except Exception as e:
                print(f"Error parsing massless_decoration: {e}")

        hide_mesh_elem = root.find('hide_mesh')
        if hide_mesh_elem is not None:
            try:
                hide_mesh_value = hide_mesh_elem.text.lower() == 'true' if hide_mesh_elem.text else False
                self.current_node.hide_mesh = hide_mesh_value
                if hasattr(self, 'hide_mesh_checkbox'):
                    self.hide_mesh_checkbox.setChecked(hide_mesh_value)
            except Exception as e:
                print(f"Error parsing hide_mesh: {e}")

        # Collider情報の読み込み
        collider_elem = root.find('collider')
        if collider_elem is not None:
            collider_type = collider_elem.get('type')
            collider_file = collider_elem.get('file')

            if collider_type == 'primitive' and collider_file:
                collider_xml_path = os.path.join(xml_dir, collider_file)
                if os.path.exists(collider_xml_path):
                    collider_data = self.parse_collider_xml(collider_xml_path)
                    if collider_data:
                        if hasattr(self, 'collider_mesh_input'):
                            self.collider_mesh_input.setText(f"Primitive {collider_data['type'].capitalize()}")
                        if hasattr(self, 'collider_enabled_checkbox'):
                            self.collider_enabled_checkbox.setChecked(True)

                        # Update colliders list
                        if not hasattr(self.current_node, 'colliders'):
                            self.current_node.colliders = []
                        self.current_node.colliders = [{
                            'type': 'primitive',
                            'enabled': True,
                            'data': collider_data,
                            'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                            'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0]),
                            'mesh': None,
                            'mesh_scale': [1.0, 1.0, 1.0]
                        }]

            elif collider_type == 'mesh' and collider_file:
                collider_mesh_path = os.path.join(xml_dir, collider_file)
                if os.path.exists(collider_mesh_path):
                    if hasattr(self, 'collider_mesh_input'):
                        self.collider_mesh_input.setText(os.path.basename(collider_mesh_path))
                    if hasattr(self, 'collider_enabled_checkbox'):
                        self.collider_enabled_checkbox.setChecked(True)

                    # Update colliders list
                    if not hasattr(self.current_node, 'colliders'):
                        self.current_node.colliders = []
                    self.current_node.colliders = [{
                        'type': 'mesh',
                        'enabled': True,
                        'data': None,
                        'position': [0.0, 0.0, 0.0],
                        'rotation': [0.0, 0.0, 0.0],
                        'mesh': collider_mesh_path,
                        'mesh_scale': [1.0, 1.0, 1.0]
                    }]

        # Process collision mesh (legacy XML format support)
        collision_mesh_elem = link_elem.find('collision_mesh') if link_elem is not None else None
        if collision_mesh_elem is not None and collision_mesh_elem.text:
            collision_mesh_path = os.path.join(xml_dir, collision_mesh_elem.text.strip())
            if os.path.exists(collision_mesh_path):
                if hasattr(self, 'collider_mesh_input'):
                    self.collider_mesh_input.setText(os.path.basename(collision_mesh_path))
                # Update colliders list
                if not hasattr(self.current_node, 'colliders'):
                    self.current_node.colliders = []
                self.current_node.colliders = [{
                    'type': 'mesh',
                    'enabled': True,
                    'data': None,
                    'position': [0.0, 0.0, 0.0],
                    'rotation': [0.0, 0.0, 0.0],
                    'mesh': collision_mesh_path,
                    'mesh_scale': [1.0, 1.0, 1.0]
                }]



    def load_xml(self):
        """Load XML file"""
        if not self.current_node:
            return

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open XML File", "", "XML Files (*.xml)")

        if not file_name:
            return

        try:
            tree = ET.parse(file_name)
            root = tree.getroot()

            if root.tag != 'urdf_part':
                print("Invalid XML format: Root element should be 'urdf_part'")
                return


            # Load common properties
            xml_dir = os.path.dirname(file_name)
            self._load_xml_common_properties(root, xml_dir)

            # Process rotation axis
            joint_elem = root.find('joint')
            if joint_elem is not None:
                # Check joint type attribute
                joint_type = joint_elem.get('type', '')
                if joint_type == 'fixed':
                    self.current_node.rotation_axis = 3  # Use 3 for Fixed
                    if self.axis_group.button(3):  # If Fixed button exists
                        self.axis_group.button(3).setChecked(True)
                else:
                    # Process rotation axis
                    axis_elem = joint_elem.find('axis')
                    if axis_elem is not None:
                        axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                        axis_values = [float(x) for x in axis_xyz]
                        if axis_values[2] == 1:  # Z axis
                            self.current_node.rotation_axis = 2
                            self.axis_group.button(2).setChecked(True)
                        elif axis_values[1] == 1:  # Y axis
                            self.current_node.rotation_axis = 1
                            self.axis_group.button(1).setChecked(True)
                        else:  # X axis (default)
                            self.current_node.rotation_axis = 0
                            self.axis_group.button(0).setChecked(True)

                # Process Joint limits
                limit_elem = joint_elem.find('limit')
                if limit_elem is not None:
                    # Read from XML as Radian values
                    lower_rad = float(limit_elem.get('lower', -3.14159))
                    upper_rad = float(limit_elem.get('upper', 3.14159))
                    effort = float(limit_elem.get('effort', 10.0))
                    velocity = float(limit_elem.get('velocity', 3.0))
                    margin = float(limit_elem.get('margin', DEFAULT_MARGIN))
                    armature = float(limit_elem.get('armature', DEFAULT_ARMATURE))
                    frictionloss = float(limit_elem.get('frictionloss', DEFAULT_FRICTIONLOSS))

                    # Store as Radian values in node
                    self.current_node.joint_lower = lower_rad
                    self.current_node.joint_upper = upper_rad
                    self.current_node.joint_effort = effort
                    self.current_node.joint_velocity = velocity
                    self.current_node.joint_margin = margin
                    self.current_node.joint_armature = armature
                    self.current_node.joint_frictionloss = frictionloss

                    # Convert to Degree for UI display
                    self.lower_limit_input.setText(str(round(math.degrees(lower_rad), 2)))
                    self.upper_limit_input.setText(str(round(math.degrees(upper_rad), 2)))
                    self.effort_input.setText(format_float_no_exp(effort))
                    self.velocity_input.setText(format_float_no_exp(velocity))
                    self.margin_input.setText(format_float_no_exp(margin))
                    self.armature_input.setText(format_float_no_exp(armature))
                    self.frictionloss_input.setText(format_float_no_exp(frictionloss))


            # Process points
            points = root.findall('point')
            num_points = len(points)

            # Compare current port count with required port count
            current_ports = len(self.current_node.output_ports())

            # Adjust port count
            if isinstance(self.current_node, FooNode):
                # Clear connections for ports to be deleted before removing them
                if current_ports > num_points:
                    for i in range(num_points + 1, current_ports + 1):
                        port_name = f'out_{i}'
                        port = self.current_node.get_output(port_name)
                        if port:
                            port.clear_connections()

                while current_ports < num_points:
                    self.current_node._add_output()
                    current_ports += 1

                while current_ports > num_points:
                    self.current_node.remove_output()
                    current_ports -= 1

                # Update point data
                self.current_node.points = []
                for point_elem in points:
                    point_name = point_elem.get('name')
                    point_type = point_elem.get('type')
                    point_xyz_elem = point_elem.find('point_xyz')
                    point_angle_elem = point_elem.find('point_angle')

                    if point_xyz_elem is not None and point_xyz_elem.text:
                        xyz_values = [float(x) for x in point_xyz_elem.text.strip().split()]
                        # Load point_angle
                        angle_values = [0.0, 0.0, 0.0]
                        if point_angle_elem is not None and point_angle_elem.text:
                            try:
                                angle_values = [float(x) for x in point_angle_elem.text.strip().split()]
                                if len(angle_values) != 3:
                                    angle_values = [0.0, 0.0, 0.0]
                            except ValueError:
                                angle_values = [0.0, 0.0, 0.0]
                        self.current_node.points.append({
                            'name': point_name,
                            'type': point_type,
                            'xyz': xyz_values,
                            'angle': angle_values
                        })

                # Update cumulative coordinates
                self.current_node.cumulative_coords = []
                for i in range(len(self.current_node.points)):
                    self.current_node.cumulative_coords.append(create_cumulative_coord(i))

                # Update output_count
                self.current_node.output_count = len(self.current_node.points)

            # Update UI
            self.update_info(self.current_node)

            # Update 3D view
            if self.stl_viewer:
                self.stl_viewer.render_to_image()
                # Also update Collider display
                self.stl_viewer.refresh_collider_display()

            # Execute same effect as Recalc Positions
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()

            # Save XML filename
            self.current_node.xml_file = file_name

        except Exception as e:
            print(f"Error loading XML: {str(e)}")
            import traceback
            traceback.print_exc()


            # Execute same effect as Recalc Positions
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()

            # Save XML filename
            self.current_node.xml_file = file_name

        except Exception as e:
            print(f"Error loading XML: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_xml_with_stl(self):
        """Load XML file and corresponding STL file"""
        if not self.current_node:
            return

        # Select XML file
        xml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open XML File", "", "XML Files (*.xml)")

        if not xml_file:
            return

        try:
            # Generate corresponding STL file path
            xml_dir = os.path.dirname(xml_file)
            xml_name = os.path.splitext(os.path.basename(xml_file))[0]
            stl_path = os.path.join(xml_dir, f"{xml_name}.stl")

            # Load XML file first
            tree = ET.parse(xml_file)
            root = tree.getroot()

            if root.tag != 'urdf_part':
                print("Invalid XML format: Root element should be 'urdf_part'")
                return

            # Load common properties
            xml_dir = os.path.dirname(xml_file)
            self._load_xml_common_properties(root, xml_dir)

            # Process rotation axis and joint limits
            joint_elem = root.find('joint')
            if joint_elem is not None:
                # Process rotation axis
                axis_elem = joint_elem.find('axis')
                if axis_elem is not None:
                    axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                    axis_values = [float(x) for x in axis_xyz]
                    if axis_values[2] == 1:  # Z axis
                        self.current_node.rotation_axis = 2
                        self.axis_group.button(2).setChecked(True)
                    elif axis_values[1] == 1:  # Y axis
                        self.current_node.rotation_axis = 1
                        self.axis_group.button(1).setChecked(True)
                    else:  # X axis (default)
                        self.current_node.rotation_axis = 0
                        self.axis_group.button(0).setChecked(True)
                    print(f"Set rotation axis: {self.current_node.rotation_axis} from xyz: {axis_xyz}")

                # Process Joint limits
                limit_elem = joint_elem.find('limit')
                if limit_elem is not None:
                    # Read from XML as Radian values
                    lower_rad = float(limit_elem.get('lower', -3.14159))
                    upper_rad = float(limit_elem.get('upper', 3.14159))
                    effort = float(limit_elem.get('effort', 10.0))
                    velocity = float(limit_elem.get('velocity', 3.0))
                    damping = float(limit_elem.get('damping', DEFAULT_DAMPING_KV))
                    stiffness = float(limit_elem.get('stiffness', DEFAULT_STIFFNESS_KP))
                    margin = float(limit_elem.get('margin', DEFAULT_MARGIN))
                    armature = float(limit_elem.get('armature', DEFAULT_ARMATURE))
                    frictionloss = float(limit_elem.get('frictionloss', DEFAULT_FRICTIONLOSS))

                    # Store as Radian values in node
                    self.current_node.joint_lower = lower_rad
                    self.current_node.joint_upper = upper_rad
                    self.current_node.joint_effort = effort
                    self.current_node.joint_velocity = velocity
                    self.current_node.joint_damping = damping
                    self.current_node.joint_stiffness = stiffness
                    self.current_node.joint_margin = margin
                    self.current_node.joint_armature = armature
                    self.current_node.joint_frictionloss = frictionloss

                    # Convert to Degree for UI display
                    self.lower_limit_input.setText(str(round(math.degrees(lower_rad), 2)))
                    self.upper_limit_input.setText(str(round(math.degrees(upper_rad), 2)))
                    self.effort_input.setText(format_float_no_exp(effort))
                    self.velocity_input.setText(format_float_no_exp(velocity))
                    if hasattr(self, 'damping_input'):
                        self.damping_input.setText(format_float_no_exp(damping))
                    if hasattr(self, 'stiffness_input'):
                        self.stiffness_input.setText(format_float_no_exp(stiffness))
                    self.margin_input.setText(format_float_no_exp(margin))
                    self.armature_input.setText(format_float_no_exp(armature))
                    self.frictionloss_input.setText(format_float_no_exp(frictionloss))

                # Process Joint dynamics (load with priority)
                dynamics_elem = joint_elem.find('dynamics')
                if dynamics_elem is not None:
                    if dynamics_elem.get('damping'):
                        self.current_node.joint_damping = float(dynamics_elem.get('damping', DEFAULT_DAMPING_KV))
                        if hasattr(self, 'damping_input'):
                            self.damping_input.setText(format_float_no_exp(self.current_node.joint_damping))
                    if dynamics_elem.get('stiffness'):
                        self.current_node.joint_stiffness = float(dynamics_elem.get('stiffness', DEFAULT_STIFFNESS_KP))
                        if hasattr(self, 'stiffness_input'):
                            self.stiffness_input.setText(format_float_no_exp(self.current_node.joint_stiffness))
                    if dynamics_elem.get('margin'):
                        self.current_node.joint_margin = float(dynamics_elem.get('margin', DEFAULT_MARGIN))
                        self.margin_input.setText(format_float_no_exp(self.current_node.joint_margin))
                    if dynamics_elem.get('armature'):
                        self.current_node.joint_armature = float(dynamics_elem.get('armature', DEFAULT_ARMATURE))
                        self.armature_input.setText(format_float_no_exp(self.current_node.joint_armature))
                    if dynamics_elem.get('frictionloss'):
                        self.current_node.joint_frictionloss = float(dynamics_elem.get('frictionloss', DEFAULT_FRICTIONLOSS))
                        self.frictionloss_input.setText(format_float_no_exp(self.current_node.joint_frictionloss))

            # Process points
            points = root.findall('point')
            num_points = len(points)

            # Adjust port count only for FooNode
            if isinstance(self.current_node, FooNode):
                # Get current port count correctly
                current_ports = len(self.current_node.output_ports())

                # Clear connections for ports to be deleted before removing them
                if current_ports > num_points:
                    for i in range(num_points + 1, current_ports + 1):
                        port_name = f'out_{i}'
                        port = self.current_node.get_output(port_name)
                        if port:
                            port.clear_connections()

                while current_ports < num_points:
                    self.current_node._add_output()
                    current_ports += 1

                while current_ports > num_points:
                    self.current_node.remove_output()
                    current_ports -= 1

                # Update point data
                self.current_node.points = []
                for point_elem in points:
                    point_name = point_elem.get('name')
                    point_type = point_elem.get('type')
                    point_xyz_elem = point_elem.find('point_xyz')
                    point_angle_elem = point_elem.find('point_angle')

                    if point_xyz_elem is not None and point_xyz_elem.text:
                        xyz_values = [float(x) for x in point_xyz_elem.text.strip().split()]
                        # Load point_angle
                        angle_values = [0.0, 0.0, 0.0]
                        if point_angle_elem is not None and point_angle_elem.text:
                            try:
                                angle_values = [float(x) for x in point_angle_elem.text.strip().split()]
                                if len(angle_values) != 3:
                                    angle_values = [0.0, 0.0, 0.0]
                            except ValueError:
                                angle_values = [0.0, 0.0, 0.0]
                        self.current_node.points.append({
                            'name': point_name,
                            'type': point_type,
                            'xyz': xyz_values,
                            'angle': angle_values
                        })

                # Update cumulative coordinates
                self.current_node.cumulative_coords = []
                for i in range(len(self.current_node.points)):
                    self.current_node.cumulative_coords.append(create_cumulative_coord(i))

                # Update output_count
                self.current_node.output_count = len(self.current_node.points)

            # Process STL file
            mesh_file = None
            if os.path.exists(stl_path):
                mesh_file = stl_path
            else:
                # If STL not found, look for DAE file
                dae_path = os.path.join(xml_dir, f"{xml_name}.dae")
                if os.path.exists(dae_path):
                    mesh_file = dae_path
                else:
                    # If neither found, show dialog
                    print(f"Warning: Neither STL nor DAE file found: {stl_path}, {dae_path}")
                    msg_box = QtWidgets.QMessageBox()
                    msg_box.setIcon(QtWidgets.QMessageBox.Warning)
                    msg_box.setWindowTitle("Mesh File Not Found")
                    msg_box.setText("Neither STL nor DAE file found in the same directory.")
                    msg_box.setInformativeText("Would you like to select the mesh file manually?")
                    msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    msg_box.setDefaultButton(QtWidgets.QMessageBox.Yes)

                    if msg_box.exec() == QtWidgets.QMessageBox.Yes:
                        file_filter = get_mesh_file_filter(trimesh_available=True)
                        mesh_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                            self, "Select Mesh File", xml_dir, file_filter)
                        if mesh_file:
                            pass
                        else:
                            pass
                    else:
                        pass

            # Load if mesh file was found or selected
            if mesh_file:
                self.current_node.stl_file = mesh_file
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                    # Apply color to STL model
                    self.apply_color_to_stl()

            # Auto-detect and load Collider Mesh
            collider_xml_path = os.path.join(xml_dir, f"{xml_name}_collider.xml")
            if os.path.exists(collider_xml_path):
                try:
                    print(f"Found collider XML: {collider_xml_path}")

                    # Parse Collider XML
                    collider_data = self.parse_collider_xml(collider_xml_path)
                    if collider_data:
                        # Update colliders list
                        if not hasattr(self.current_node, 'colliders'):
                            self.current_node.colliders = []
                        self.current_node.colliders = [{
                            'type': 'primitive',
                            'enabled': True,
                            'data': collider_data,
                            'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                            'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0]),
                            'mesh': None,
                            'mesh_scale': [1.0, 1.0, 1.0]
                        }]

                        # Update UI
                        if hasattr(self, 'collider_mesh_input'):
                            self.collider_mesh_input.setText(f"Primitive {collider_data['type'].capitalize()}")
                        if hasattr(self, 'collider_enabled_checkbox'):
                            self.collider_enabled_checkbox.setChecked(True)

                        print(f"✓ Collider mesh automatically loaded: {collider_xml_path}")
                        print(f"  Type: {collider_data['type']}")
                    else:
                        print(f"Warning: Failed to parse collider XML: {collider_xml_path}")

                except Exception as e:
                    print(f"Warning: Failed to load collider XML: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Update UI
            self.update_info(self.current_node)

            # Update 3D view
            if self.stl_viewer:
                self.stl_viewer.render_to_image()
                # Also update Collider display
                self.stl_viewer.refresh_collider_display()

            # Execute same effect as Recalc Positions
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()

            # Save XML filename
            self.current_node.xml_file = xml_file

        except Exception as e:
            print(f"Error loading XML with STL: {str(e)}")
            import traceback
            traceback.print_exc()

    # ========== Helper Methods for Code Consolidation ==========

    def _get_node_file_path(self, attr_name):
        """Safely get file path from node"""
        if not self.current_node:
            return None
        return getattr(self.current_node, attr_name, None) if hasattr(self.current_node, attr_name) else None

    def _show_message(self, title, message, msg_type='info'):
        """Unified message box display"""
        if msg_type == 'warning':
            QtWidgets.QMessageBox.warning(self, title, message)
        elif msg_type == 'error':
            QtWidgets.QMessageBox.critical(self, title, message)
        else:
            QtWidgets.QMessageBox.information(self, title, message)

    class _OperationGuard:
        """Context manager for preventing duplicate execution"""
        def __init__(self, parent, flag_name):
            self.parent = parent
            self.flag_name = flag_name

        def __enter__(self):
            if hasattr(self.parent, self.flag_name) and getattr(self.parent, self.flag_name):
                return False  # Already in progress
            setattr(self.parent, self.flag_name, True)
            return True

        def __exit__(self, exc_type, exc_val, exc_tb):
            setattr(self.parent, self.flag_name, False)

    def save_xml(self):
        """Overwrite save current node parameters to XML file"""
        # Prevent duplicate execution
        with self._OperationGuard(self, '_save_xml_in_progress') as can_proceed:
            if not can_proceed:
                print("Save XML already in progress, ignoring duplicate call")
                return

            self._save_xml_impl()


    def clear_all_parameters(self):
        """Reset all BaseLinkNode parameters to default values"""
        if not self.current_node or not isinstance(self.current_node, BaseLinkNode):
            return

        # Reset to default values
        self.current_node.mass_value = 0.0
        self.current_node.inertia = DEFAULT_INERTIA_ZERO.copy()
        self.current_node.inertial_origin = {
            'xyz': DEFAULT_ORIGIN_ZERO['xyz'].copy(),
            'rpy': DEFAULT_ORIGIN_ZERO['rpy'].copy()
        }
        self.current_node.stl_file = None
        self.current_node.node_color = DEFAULT_COLOR_WHITE.copy()
        self.current_node.rotation_axis = 3  # Fixed
        self.current_node.joint_lower = 0.0
        self.current_node.joint_upper = 0.0
        self.current_node.joint_effort = DEFAULT_JOINT_EFFORT
        self.current_node.joint_velocity = DEFAULT_JOINT_VELOCITY
        self.current_node.joint_damping = DEFAULT_JOINT_DAMPING
        self.current_node.joint_stiffness = DEFAULT_JOINT_STIFFNESS
        self.current_node.joint_margin = DEFAULT_MARGIN
        self.current_node.joint_armature = DEFAULT_ARMATURE
        self.current_node.joint_frictionloss = DEFAULT_FRICTIONLOSS

        # Set output port count to 1
        while self.current_node.output_count > 1:
            self.current_node.remove_output()
        while self.current_node.output_count < 1:
            self.current_node._add_output()

        # Update UI
        self.update_info(self.current_node)

        # Clear 3D viewer
        if self.stl_viewer:
            self.stl_viewer.clear_all_polydata()

        print("BaseLinkNode parameters reset to default values")

    def open_parts_editor(self):
        """Open PartsEditor and load current Mesh and XML"""
        if not self.current_node:
            self._show_message("PartsEditor - Warning", "No node selected.", 'warning')
            return

        # STLファイルのパスを取得とバリデーション
        stl_file = self._get_node_file_path('stl_file')
        if not stl_file:
            self._show_message("PartsEditor - Warning",
                             "No mesh file loaded for this node.\n\n"
                             "Please load a mesh file first using 'Import Mesh' button.",
                             'warning')
            return

        if not os.path.exists(stl_file):
            self._show_message("PartsEditor - Warning",
                             f"Mesh file not found:\n{stl_file}\n\n"
                             "The file may have been moved or deleted.",
                             'warning')
            return

        if not os.access(stl_file, os.R_OK):
            self._show_message("PartsEditor - Warning",
                             f"Cannot read mesh file:\n{stl_file}\n\n"
                             "Please check file permissions.",
                             'warning')
            return

        # PartsEditorのパスを取得
        try:
            assembler_dir = os.path.dirname(os.path.abspath(__file__))
            parts_editor_path = os.path.join(assembler_dir, 'urdf_kitchen_PartsEditor.py')
        except Exception as e:
            self._show_message("PartsEditor - Error",
                             f"Could not determine PartsEditor path:\n\n{str(e)}",
                             'error')
            return

        if not os.path.exists(parts_editor_path):
            self._show_message("PartsEditor - Error",
                             f"PartsEditor not found at:\n{parts_editor_path}\n\n"
                             "Please ensure urdf_kitchen_PartsEditor.py is in the same directory.",
                             'error')
            return

        # Prepare collider data (from colliders list)
        collider_info = None
        collider_xml_path = None

        if hasattr(self.current_node, 'colliders') and self.current_node.colliders:
            for collider in self.current_node.colliders:
                if not collider.get('enabled', False):
                    continue

                if collider.get('type') == 'primitive' and collider.get('data'):
                    # For primitive collider, create temporary collider XML file
                    try:
                        collider_data = collider['data']
                        # Create temporary collider XML file in same directory as STL file
                        stl_dir = os.path.dirname(stl_file)
                        stl_basename = os.path.splitext(os.path.basename(stl_file))[0]
                        collider_xml_path = os.path.join(stl_dir, f"{stl_basename}_collider.xml")

                        # Create collider XML file
                        root = ET.Element('urdf_kitchen_collider')
                        collider_elem = ET.SubElement(root, 'collider')
                        collider_elem.set('type', collider_data.get('type', 'box'))

                        # Add geometry element
                        geometry = collider_data.get('geometry', {})
                        if geometry:
                            geometry_elem = ET.SubElement(collider_elem, 'geometry')
                            geom_type = collider_data.get('type', 'box')

                            if geom_type == 'box':
                                geometry_elem.set('size_x', str(geometry.get('size_x', 1.0)))
                                geometry_elem.set('size_y', str(geometry.get('size_y', 1.0)))
                                geometry_elem.set('size_z', str(geometry.get('size_z', 1.0)))
                            elif geom_type == 'sphere':
                                geometry_elem.set('radius', str(geometry.get('radius', 0.5)))
                            elif geom_type == 'cylinder':
                                geometry_elem.set('radius', str(geometry.get('radius', 0.5)))
                                geometry_elem.set('length', str(geometry.get('length', 1.0)))
                            elif geom_type == 'capsule':
                                geometry_elem.set('radius', str(geometry.get('radius', 0.5)))
                                geometry_elem.set('length', str(geometry.get('length', 1.0)))

                        # Add position element
                        position = collider.get('position', collider_data.get('position', [0.0, 0.0, 0.0]))
                        position_elem = ET.SubElement(collider_elem, 'position')
                        position_elem.set('x', str(position[0]))
                        position_elem.set('y', str(position[1]))
                        position_elem.set('z', str(position[2]))

                        # Add rotation element (degrees remain as degrees)
                        rotation = collider.get('rotation', collider_data.get('rotation', [0.0, 0.0, 0.0]))
                        rotation_elem = ET.SubElement(collider_elem, 'rotation')
                        rotation_elem.set('roll', str(rotation[0]))
                        rotation_elem.set('pitch', str(rotation[1]))
                        rotation_elem.set('yaw', str(rotation[2]))

                        # Save XML file
                        tree = ET.ElementTree(root)
                        tree.write(collider_xml_path, encoding='utf-8', xml_declaration=True)
                        print(f"Created temporary collider XML: {collider_xml_path}")

                        collider_info = {
                            'type': 'primitive',
                            'xml_path': collider_xml_path
                        }
                        break  # Use only the first enabled primitive collider
                    except Exception as e:
                        print(f"Error creating collider XML: {e}")
                        import traceback
                        traceback.print_exc()

                elif collider.get('type') == 'mesh' and collider.get('mesh'):
                    collider_info = {
                        'type': 'mesh',
                        'mesh_path': collider['mesh']
                    }
                    print(f"Preparing collider mesh for PartsEditor: {os.path.basename(collider['mesh'])}")
                    break  # Use only the first enabled mesh collider

        # Try to connect to existing PartsEditor instance first
        socket = QLocalSocket()
        server_name = "URDFKitchen_PartsEditor"
        socket.connectToServer(server_name)

        if socket.waitForConnected(1000):  # Wait up to 1 second
            # Existing PartsEditor found, send file path and collider data to load
            try:
                import json
                print(f"Connected to existing PartsEditor, sending file: {stl_file}")
                
                # Build message in JSON format
                message_data = {
                    'stl_file': stl_file,
                    'collider': collider_info
                }
                message_json = json.dumps(message_data)
                message = f"LOAD_JSON:{message_json}".encode('utf-8')
                
                socket.write(message)
                socket.flush()

                # Wait for response
                if socket.waitForReadyRead(3000):  # Wait up to 3 seconds
                    response = socket.readAll().data().decode('utf-8')
                    print(f"PartsEditor response: {response}")
                    if response == "OK":
                        print("File and collider data loaded successfully in existing PartsEditor")
                    else:
                        print(f"PartsEditor error: {response}")

                socket.disconnectFromServer()
                return

            except Exception as e:
                print(f"Error communicating with PartsEditor: {e}")
                import traceback
                traceback.print_exc()
                socket.disconnectFromServer()

        # No existing PartsEditor, launch new process
        try:
            print("No existing PartsEditor found, launching new instance")
            import subprocess
            import sys
            import json
            import time

            python_exe = sys.executable
            if not python_exe or not os.path.exists(python_exe):
                raise RuntimeError(f"Python executable not found: {python_exe}")

            process = subprocess.Popen(
                [python_exe, parts_editor_path, stl_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Verify process startup
            time.sleep(0.1)
            poll = process.poll()
            if poll is not None:
                stderr = process.stderr.read().decode('utf-8', errors='replace')
                raise RuntimeError(f"PartsEditor exited immediately.\n\nError output:\n{stderr[:500]}")

            # If collider data exists, send via message after startup
            if collider_info:
                # Wait a bit for PartsEditor to start
                time.sleep(0.5)

                # Attempt to connect and send collider data
                socket = QLocalSocket()
                socket.connectToServer(server_name)
                
                if socket.waitForConnected(2000):  # Wait up to 2 seconds
                    try:
                        message_data = {
                            'stl_file': stl_file,
                            'collider': collider_info
                        }
                        message_json = json.dumps(message_data)
                        message = f"LOAD_JSON:{message_json}".encode('utf-8')
                        
                        socket.write(message)
                        socket.flush()
                        
                        if socket.waitForReadyRead(3000):
                            response = socket.readAll().data().decode('utf-8')
                            print(f"PartsEditor response (collider data): {response}")
                        
                        socket.disconnectFromServer()
                    except Exception as e:
                        print(f"Error sending collider data to PartsEditor: {e}")
                        socket.disconnectFromServer()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_message("PartsEditor - Error",
                             f"Failed to launch PartsEditor:\n\n{str(e)}",
                             'error')

    def reload_node_files(self):
        """Reload XML file and Mesh file for current node"""
        with self._OperationGuard(self, '_reload_in_progress') as can_proceed:
            if not can_proceed:
                print("Reload already in progress, ignoring duplicate call")
                return

            self._reload_node_files_impl()

    def create_port_widget(self, port_number, x=0.0, y=0.0, z=0.0, angle_x=0.0, angle_y=0.0, angle_z=0.0):
        """Create widget for Output Port"""
        port_layout = QtWidgets.QHBoxLayout()  # Changed from GridLayout to HBoxLayout
        port_layout.setSpacing(5)
        port_layout.setContentsMargins(0, 1, 0, 1)

        # Port number
        port_name = QtWidgets.QLabel(f"out_{port_number}")
        port_name.setFixedWidth(40)
        port_layout.addWidget(port_name)

        # Create coordinate input pairs
        coords = []
        for label, value in [('X:', x), ('Y:', y), ('Z:', z)]:
            # Create each coordinate pair with HBoxLayout
            coord_pair = QtWidgets.QHBoxLayout()
            coord_pair.setSpacing(0)

            # Label
            coord_label = QtWidgets.QLabel(label)
            coord_label.setFixedWidth(15)
            coord_pair.addWidget(coord_label)

            # Input field
            coord_input = QtWidgets.QLineEdit(f"{value:.6f}")
            coord_input.setFixedWidth(70)
            coord_input.setFixedHeight(20)
            coord_input.setStyleSheet("QLineEdit { padding-left: 2px; padding-top: 0px; padding-bottom: 0px; }")
            coord_input.setValidator(QtGui.QDoubleValidator())
            coord_input.textChanged.connect(
                lambda text, idx=port_number-1, coord=len(coords):
                self.update_port_coordinate(idx, coord, text))
            coord_pair.addWidget(coord_input)
            coords.append(coord_input)

            # Add pair to main layout
            port_layout.addLayout(coord_pair)

            # # Add spacing between pairs
            # if label != 'Z:':  # Add space after all pairs except the last one
            #     port_layout.addSpacing(5)

        # Create angle input pairs
        port_layout.addSpacing(2)  # 座標とAngleの間にスペース

        # Angle ラベル
        angle_label = QtWidgets.QLabel("Ang")
        angle_label.setFixedWidth(24)
        port_layout.addWidget(angle_label)

        angles = []
        for label, value in [('X:', angle_x), ('Y:', angle_y), ('Z:', angle_z)]:
            # Create each angle pair with HBoxLayout
            angle_pair = QtWidgets.QHBoxLayout()
            angle_pair.setSpacing(2)

            # Label
            angle_label_item = QtWidgets.QLabel(label)
            angle_label_item.setFixedWidth(10)
            angle_pair.addWidget(angle_label_item)

            # Input field
            angle_input = QtWidgets.QLineEdit(f"{value:.2f}")
            angle_input.setFixedWidth(45)
            angle_input.setFixedHeight(20)
            angle_input.setStyleSheet("QLineEdit { padding-left: 2px; padding-top: 0px; padding-bottom: 0px; }")
            angle_input.setValidator(QtGui.QDoubleValidator())
            angle_input.textChanged.connect(
                lambda text, idx=port_number-1, angle_idx=len(angles):
                self.update_port_angle(idx, angle_idx, text))
            angle_pair.addWidget(angle_input)
            angles.append(angle_input)

            # Add pair to main layout
            port_layout.addLayout(angle_pair)

            # Add spacing between pairs
            # if label != 'Z:':  # Add space after all pairs except the last one
            #     port_layout.addSpacing(1)

        # Right margin
        port_layout.addStretch()

        # Wrap widget
        port_widget = QtWidgets.QWidget()
        port_widget.setFixedHeight(25)
        port_widget.setLayout(port_layout)
        return port_widget, coords, angles

    def update_output_ports(self, node):
        """Update Output Ports section"""
        # Clear existing port widgets
        for widget in self.port_widgets:
            self.ports_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self.port_widgets.clear()

        # Create widget for each port of the node
        if hasattr(node, 'points'):
            for i, point in enumerate(node.points):
                # Get point_angle (internally radians, displayed as degrees in UI)
                angle_rad = point.get('angle', [0.0, 0.0, 0.0])
                # Convert from radians to degrees for UI
                angle_deg = [math.degrees(a) for a in angle_rad]
                port_widget, _, _ = self.create_port_widget(
                    i + 1,
                    point['xyz'][0],
                    point['xyz'][1],
                    point['xyz'][2],
                    angle_deg[0],
                    angle_deg[1],
                    angle_deg[2]
                )
                self.ports_layout.addWidget(port_widget)
                self.port_widgets.append(port_widget)

    def apply_original_mesh_color(self):
        """Apply original color data (mesh_original_color) that was initially set for the Mesh"""
        if not self.current_node:
            return

        try:
            # mesh_original_colorを確認
            if not hasattr(self.current_node, 'mesh_original_color') or self.current_node.mesh_original_color is None:
                print(f"Warning: No original mesh color found for node '{self.current_node.name()}'")
                return

            # Get mesh_original_color
            original_color = self.current_node.mesh_original_color

            # Convert to RGBA format (add Alpha=1.0 if 3 elements)
            if len(original_color) == 3:
                rgba_values = list(original_color) + [1.0]
            else:
                rgba_values = list(original_color[:4])  # 最大4要素まで

            # Update node color (RGBA)
            self.current_node.node_color = rgba_values

            # Update UI
            self._set_color_ui(rgba_values)

            # Update color sample chip
            rgb_display = [int(v * 255) for v in rgba_values[:3]]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_display[0]},{rgb_display[1]},{rgb_display[2]}); "
                f"border: 1px solid black;"
            )

            # Apply color to STL model
            self.apply_color_to_stl()

            print(f"Applied original mesh color to node '{self.current_node.name()}': RGBA({rgba_values[0]:.3f}, {rgba_values[1]:.3f}, {rgba_values[2]:.3f}, {rgba_values[3]:.3f})")
        except Exception as e:
            print(f"Error applying original mesh color: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_color_to_stl(self):
        """Apply selected color to STL model (RGBA support)"""
        if not self.current_node:
            return

        try:
            # Get RGBA values
            rgba_values = [float(input.text()) for input in self.color_inputs]
            rgba_values = [max(0.0, min(1.0, value)) for value in rgba_values]

            # Update node color (RGBA)
            self.current_node.node_color = rgba_values

            if self.stl_viewer and hasattr(self.stl_viewer, 'stl_actors'):
                if self.current_node in self.stl_viewer.stl_actors:
                    actor = self.stl_viewer.stl_actors[self.current_node]
                    # Set RGB
                    actor.GetProperty().SetColor(*rgba_values[:3])
                    # Set Alpha
                    if len(rgba_values) >= 4:
                        actor.GetProperty().SetOpacity(rgba_values[3])
                        print(f"Applied color: RGBA({rgba_values[0]:.3f}, {rgba_values[1]:.3f}, "
                              f"{rgba_values[2]:.3f}, {rgba_values[3]:.3f})")
                    else:
                        actor.GetProperty().SetOpacity(1.0)
                        print(f"Applied color: RGB({rgba_values[0]:.3f}, {rgba_values[1]:.3f}, {rgba_values[2]:.3f})")
                    self.stl_viewer.render_to_image()
        except ValueError as e:
            print(f"Error: Invalid color value - {str(e)}")

    def update_massless_decoration(self, state):
        """Update Massless Decoration state"""
        if self.current_node:
            self.current_node.massless_decoration = bool(state)

    def update_hide_mesh(self, state):
        """Update Hide Mesh state and show/hide mesh in 3D view"""
        if self.current_node:
            hide = bool(state)
            self.current_node.hide_mesh = hide

            # Hide/Show mesh in 3D viewer
            if self.stl_viewer and hasattr(self.stl_viewer, 'stl_actors'):
                if self.current_node in self.stl_viewer.stl_actors:
                    actor = self.stl_viewer.stl_actors[self.current_node]
                    # If hide=True, hide (VisibilityOff); if hide=False, show (VisibilityOn)
                    actor.SetVisibility(not hide)
                    self.stl_viewer.render_to_image()

    def update_blanklink(self, state):
        """Update Blanklink state (for BaseLinkNode)"""
        if self.current_node and isinstance(self.current_node, BaseLinkNode):
            self.current_node.blank_link = bool(state)

    def update_mass(self):
        """Update mass (realtime + return key)"""
        if not self.current_node:
            return
        try:
            mass_text = self.mass_input.text()
            if mass_text:
                mass = float(mass_text)
                if mass >= 0:
                    self.current_node.mass_value = mass
        except ValueError:
            pass  # Ignore invalid values

    def update_inertial_origin(self):
        """Update inertial origin (realtime + return key)"""
        if not self.current_node:
            return
        try:
            origin_xyz = [
                float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0,
                float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0,
                float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0
            ]
            origin_rpy = [
                float(self.inertial_r_input.text()) if self.inertial_r_input.text() else 0.0,
                float(self.inertial_p_input.text()) if self.inertial_p_input.text() else 0.0,
                float(self.inertial_y_rpy_input.text()) if self.inertial_y_rpy_input.text() else 0.0
            ]
            if not hasattr(self.current_node, 'inertial_origin'):
                self.current_node.inertial_origin = {}
            self.current_node.inertial_origin['xyz'] = origin_xyz
            self.current_node.inertial_origin['rpy'] = origin_rpy

            # Look CoMが有効な場合、3Dビューを即座に更新
            if hasattr(self, 'look_inertial_origin_toggle') and self.look_inertial_origin_toggle.isChecked():
                if self.stl_viewer:
                    self.stl_viewer.show_inertial_origin(self.current_node, origin_xyz)
                    self.stl_viewer.render_to_image()

        except ValueError:
            pass  # 無効な値は無視

    def update_inertia(self):
        """慣性テンソルの更新（リアルタイム + リターンキー）"""
        if not self.current_node:
            return
        try:
            inertia_values = {
                'ixx': float(self.ixx_input.text()) if self.ixx_input.text() else 0.0,
                'ixy': float(self.ixy_input.text()) if self.ixy_input.text() else 0.0,
                'ixz': float(self.ixz_input.text()) if self.ixz_input.text() else 0.0,
                'iyy': float(self.iyy_input.text()) if self.iyy_input.text() else 0.0,
                'iyz': float(self.iyz_input.text()) if self.iyz_input.text() else 0.0,
                'izz': float(self.izz_input.text()) if self.izz_input.text() else 0.0
            }
            self.current_node.inertia = inertia_values
        except ValueError:
            pass  # 無効な値は無視

    def update_joint_params(self):
        """ジョイントパラメータの更新（リアルタイム + リターンキー）"""
        if not self.current_node:
            return
        try:
            # Effort
            if self.effort_input.text():
                self.current_node.joint_effort = float(self.effort_input.text())
            # Velocity
            if self.velocity_input.text():
                self.current_node.joint_velocity = float(self.velocity_input.text())
            # Damping
            if self.damping_input.text():
                self.current_node.joint_damping = float(self.damping_input.text())
            # Stiffness
            if self.stiffness_input.text():
                self.current_node.joint_stiffness = float(self.stiffness_input.text())
            # Margin
            if self.margin_input.text():
                self.current_node.joint_margin = float(self.margin_input.text())
            # Armature
            if self.armature_input.text():
                self.current_node.joint_armature = float(self.armature_input.text())
            # Frictionloss
            if self.frictionloss_input.text():
                self.current_node.joint_frictionloss = float(self.frictionloss_input.text())
        except ValueError:
            pass  # 無効な値は無視

    def update_joint_limits_realtime(self):
        """ジョイントリミットのリアルタイム更新"""
        if not self.current_node:
            return
        try:
            # Lower limitの保存（DegreeからRadianに変換）
            lower_text = self.lower_limit_input.text()
            if lower_text:
                self.current_node.joint_lower = math.radians(float(lower_text))
            # Upper limitの保存（DegreeからRadianに変換）
            upper_text = self.upper_limit_input.text()
            if upper_text:
                self.current_node.joint_upper = math.radians(float(upper_text))
        except ValueError:
            pass  # 無効な値は無視

    def update_body_angle(self):
        """Body Angleのリアルタイム更新"""
        if not self.current_node:
            return
        try:
            # X, Y, Z 軸の回転角度を取得（度数法で入力、ラジアンで保存）
            angle_x_deg = float(self.angle_x_input.text()) if self.angle_x_input.text() else 0.0
            angle_y_deg = float(self.angle_y_input.text()) if self.angle_y_input.text() else 0.0
            angle_z_deg = float(self.angle_z_input.text()) if self.angle_z_input.text() else 0.0

            # Degreeから Radianに変換して保存
            self.current_node.body_angle = [math.radians(angle_x_deg), math.radians(angle_y_deg), math.radians(angle_z_deg)]

            # 親ノードの接続outポートのAng値を更新
            if hasattr(self.current_node, 'graph'):
                graph = self.current_node.graph
                # ノードの入力ポートから親を探す
                for input_port in self.current_node.input_ports():
                    connected_ports = input_port.connected_ports()
                    if connected_ports:
                        parent_node = connected_ports[0].node()
                        parent_port_name = connected_ports[0].name()

                        # ポート名からポイントインデックスを計算（out_1 -> 0, out_2 -> 1, etc.）
                        point_index = 0  # デフォルト
                        if parent_port_name.startswith('out_'):
                            try:
                                port_num = int(parent_port_name.split('_')[1])
                                point_index = port_num - 1
                            except (ValueError, IndexError):
                                pass
                        elif parent_port_name == 'out':
                            point_index = 0

                        # 親ノードのpointsのangle値を更新（ラジアンで保存）
                        if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                            if 'angle' not in parent_node.points[point_index]:
                                parent_node.points[point_index]['angle'] = [0.0, 0.0, 0.0]
                            parent_node.points[point_index]['angle'] = [math.radians(angle_x_deg), math.radians(angle_y_deg), math.radians(angle_z_deg)]
                            print(f"Updated parent node {parent_node.name()} port {point_index+1} angle to [{angle_x_deg}, {angle_y_deg}, {angle_z_deg}] degrees")
                        break

            # 3Dビューの更新をトリガー
            if self.stl_viewer:
                self.stl_viewer.render_to_image()
        except ValueError:
            pass  # 無効な値は無視

    def moveEvent(self, event):
        """ウィンドウ移動イベントの処理"""
        super(InspectorWindow, self).moveEvent(event)
        # グラフオブジェクトが存在し、last_inspector_positionを保存可能な場合
        if hasattr(self, 'graph') and self.graph:
            self.graph.last_inspector_position = self.pos()

    def keyPressEvent(self, event):
        """キープレスイベントの処理"""
        # ESCキーが押されたかどうかを確認
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        # Cmd+W (macOS) または Ctrl+W (Windows/Linux) で閉じる
        elif event.key() == QtCore.Qt.Key.Key_W and (
            event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or
            event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier
        ):
            self.close()
        else:
            # 他のキーイベントは通常通り処理
            super(InspectorWindow, self).keyPressEvent(event)

    def start_rotation_test(self):
        """回転テスト開始"""
        if self.current_node and self.stl_viewer:
            # Inherit to Subnodesチェックボックスの状態を取得
            follow = self.follow_checkbox.isChecked()
            self.stl_viewer.follow_children = follow

            # 現在の変換を保存
            self.stl_viewer.store_current_transform(self.current_node)
            # 回転開始
            self.stl_viewer.start_rotation_test(self.current_node)

    def stop_rotation_test(self):
        """回転テスト終了"""
        if self.current_node and self.stl_viewer:
            # 回転停止と元の角度に戻す
            self.stl_viewer.stop_rotation_test(self.current_node)

    def look_lower_limit(self):
        """Lower limitの角度を表示"""
        if self.current_node and self.stl_viewer:
            try:
                # Inherit to Subnodesチェックボックスの状態を取得して設定
                follow = self.follow_checkbox.isChecked()
                self.stl_viewer.follow_children = follow

                # インプットフィールドから値を取得（Degree表示）
                lower_text = self.lower_limit_input.text()
                if not lower_text:
                    lower_text = self.lower_limit_input.placeholderText()

                lower_deg = float(lower_text)
                lower_rad = math.radians(lower_deg)

                # 現在の変換を保存（子ノードの変換も保存）
                self.stl_viewer.store_current_transform(self.current_node)
                # 指定角度を表示
                self.stl_viewer.show_angle(self.current_node, lower_rad)
            except ValueError:
                pass

    def look_upper_limit(self):
        """Upper limitの角度を表示"""
        if self.current_node and self.stl_viewer:
            try:
                # Inherit to Subnodesチェックボックスの状態を取得して設定
                follow = self.follow_checkbox.isChecked()
                self.stl_viewer.follow_children = follow

                # インプットフィールドから値を取得（Degree表示）
                upper_text = self.upper_limit_input.text()
                if not upper_text:
                    upper_text = self.upper_limit_input.placeholderText()

                upper_deg = float(upper_text)
                upper_rad = math.radians(upper_deg)

                # 現在の変換を保存
                self.stl_viewer.store_current_transform(self.current_node)
                # 指定角度を表示
                self.stl_viewer.show_angle(self.current_node, upper_rad)
            except ValueError:
                pass

    def look_zero_limit(self):
        """0度の角度を表示"""
        if self.current_node and self.stl_viewer:
            # Inherit to Subnodesチェックボックスの状態を取得して設定
            follow = self.follow_checkbox.isChecked()
            self.stl_viewer.follow_children = follow

            # 現在の変換を保存
            self.stl_viewer.store_current_transform(self.current_node)
            # 0ラジアンを表示
            self.stl_viewer.show_angle(self.current_node, 0.0)

    def toggle_inertial_origin_view(self, checked):
        """Inertial Originの表示/非表示を切り替え"""
        if self.current_node and self.stl_viewer:
            if checked:
                # Inertial Origin座標を取得
                try:
                    x = float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0
                    y = float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0
                    z = float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0

                    # 3Dビューに座標系を表示
                    self.stl_viewer.show_inertial_origin(self.current_node, [x, y, z])
                except ValueError:
                    self.look_inertial_origin_toggle.setChecked(False)
            else:
                # 座標系を非表示
                self.stl_viewer.hide_inertial_origin(self.current_node)

    def set_joint_limits(self):
        """Joint limitsの値をノードに保存"""
        if not self.current_node:
            return

        try:
            # Lower limitの保存（DegreeからRadianに変換）
            lower_text = self.lower_limit_input.text()
            if lower_text:
                self.current_node.joint_lower = math.radians(float(lower_text))

            # Upper limitの保存（DegreeからRadianに変換）
            upper_text = self.upper_limit_input.text()
            if upper_text:
                self.current_node.joint_upper = math.radians(float(upper_text))

            # Effortの保存
            effort_text = self.effort_input.text()
            if effort_text:
                self.current_node.joint_effort = float(effort_text)

            # Velocityの保存
            velocity_text = self.velocity_input.text()
            if velocity_text:
                self.current_node.joint_velocity = float(velocity_text)

            # Dampingの保存
            damping_text = self.damping_input.text()
            if damping_text:
                self.current_node.joint_damping = float(damping_text)

            # Stiffnessの保存
            stiffness_text = self.stiffness_input.text()
            if stiffness_text:
                self.current_node.joint_stiffness = float(stiffness_text)

            # Marginの保存
            margin_text = self.margin_input.text()
            if margin_text:
                self.current_node.joint_margin = float(margin_text)

            # Armatureの保存
            armature_text = self.armature_input.text()
            if armature_text:
                self.current_node.joint_armature = float(armature_text)

            # Frictionlossの保存
            frictionloss_text = self.frictionloss_input.text()
            if frictionloss_text:
                self.current_node.joint_frictionloss = float(frictionloss_text)

            print(f"Joint limits set: lower={math.degrees(self.current_node.joint_lower):.2f}° ({self.current_node.joint_lower:.5f} rad), upper={math.degrees(self.current_node.joint_upper):.2f}° ({self.current_node.joint_upper:.5f} rad), effort={self.current_node.joint_effort}, velocity={self.current_node.joint_velocity}, damping={self.current_node.joint_damping}, stiffness={self.current_node.joint_stiffness}, margin={self.current_node.joint_margin}, armature={self.current_node.joint_armature}, frictionloss={self.current_node.joint_frictionloss}")

            QtWidgets.QMessageBox.information(
                self,
                "Joint Limits Set",
                f"Joint limits have been set successfully.\n\n"
                f"Lower: {math.degrees(self.current_node.joint_lower):.2f}° ({self.current_node.joint_lower:.5f} rad)\n"
                f"Upper: {math.degrees(self.current_node.joint_upper):.2f}° ({self.current_node.joint_upper:.5f} rad)\n"
                f"Effort: {self.current_node.joint_effort}\n"
                f"Damping: {self.current_node.joint_damping}\n"
                f"Stiffness: {self.current_node.joint_stiffness}\n"
                f"Velocity: {self.current_node.joint_velocity}\n"
                f"Margin: {self.current_node.joint_margin}\n"
                f"Armature: {self.current_node.joint_armature}\n"
                f"Frictionloss: {self.current_node.joint_frictionloss}"
            )
        except ValueError as e:
            print(f"Error setting joint limits: {str(e)}")
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values."
            )

    def set_inertia(self):
        """InertiaとInertial Originの値をノードに保存"""
        if not self.current_node:
            return

        try:
            # Inertial Origin値を取得
            origin_xyz = [
                float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0,
                float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0,
                float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0
            ]
            origin_rpy = [
                float(self.inertial_r_input.text()) if self.inertial_r_input.text() else 0.0,
                float(self.inertial_p_input.text()) if self.inertial_p_input.text() else 0.0,
                float(self.inertial_y_rpy_input.text()) if self.inertial_y_rpy_input.text() else 0.0
            ]

            # Inertia値を取得
            inertia_values = {
                'ixx': float(self.ixx_input.text()) if self.ixx_input.text() else 0.0,
                'ixy': float(self.ixy_input.text()) if self.ixy_input.text() else 0.0,
                'ixz': float(self.ixz_input.text()) if self.ixz_input.text() else 0.0,
                'iyy': float(self.iyy_input.text()) if self.iyy_input.text() else 0.0,
                'iyz': float(self.iyz_input.text()) if self.iyz_input.text() else 0.0,
                'izz': float(self.izz_input.text()) if self.izz_input.text() else 0.0
            }

            # ノードに保存
            self.current_node.inertial_origin = {
                'xyz': origin_xyz,
                'rpy': origin_rpy
            }
            self.current_node.inertia = inertia_values


            QtWidgets.QMessageBox.information(
                self,
                "Inertial Set",
                f"Inertial values have been set successfully.\n\n"
                f"Origin xyz: {origin_xyz}\n"
                f"Origin rpy: {origin_rpy}\n\n"
                f"ixx: {inertia_values['ixx']}\n"
                f"ixy: {inertia_values['ixy']}\n"
                f"ixz: {inertia_values['ixz']}\n"
                f"iyy: {inertia_values['iyy']}\n"
                f"iyz: {inertia_values['iyz']}\n"
                f"izz: {inertia_values['izz']}"
            )
        except ValueError as e:
            print(f"Error setting inertia: {str(e)}")
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values for inertia."
            )

    def recalculate_com(self):
        """STLファイルからtrimeshを使用してCenter of Massを再計算"""
        if not self.current_node:
            QtWidgets.QMessageBox.warning(
                self,
                "No Node Selected",
                "Please select a node first."
            )
            return

        # STLファイルが存在するか確認
        if not hasattr(self.current_node, 'stl_file') or not self.current_node.stl_file:
            QtWidgets.QMessageBox.warning(
                self,
                "No STL File",
                "This node has no STL file attached.\nPlease load an STL file first."
            )
            return

        stl_path = self.current_node.stl_file
        if not os.path.exists(stl_path):
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"STL file not found:\n{stl_path}"
            )
            return

        try:
            # Trimeshでメッシュを読み込み
            print(f"\n=== Recalculating Center of Mass ===")
            mesh = trimesh.load(stl_path)

            # メッシュが複数ある場合は結合
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            print(f"Mesh loaded successfully")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Volume: {mesh.volume:.6f}")
            print(f"  Is watertight: {mesh.is_watertight}")

            # メッシュが閉じていない場合は自動修復を試みる
            repair_performed = False
            original_watertight = mesh.is_watertight

            if not mesh.is_watertight:
                print("\n⚠ Mesh is not watertight. Attempting automatic repair...")

                # メモリ上でメッシュを修復（元ファイルは変更しない）
                try:
                    # 法線の修正
                    try:
                        print("  - Fixing normals...")
                        mesh.fix_normals()
                    except AttributeError:
                        print("  - Skipping normals fixing (method not available)")

                    # 重複面の削除（trimeshの古いバージョンでは利用不可）
                    try:
                        print("  - Removing duplicate faces...")
                        mesh.remove_duplicate_faces()
                    except AttributeError:
                        print("  - Skipping duplicate faces removal (method not available)")

                    # 退化面の削除
                    try:
                        print("  - Removing degenerate faces...")
                        mesh.remove_degenerate_faces()
                    except AttributeError:
                        print("  - Skipping degenerate faces removal (method not available)")

                    # 穴の修復
                    try:
                        print("  - Filling holes...")
                        mesh.fill_holes()
                    except AttributeError:
                        print("  - Skipping holes filling (method not available)")

                    repair_performed = True

                    # 修復後の状態を確認
                    print(f"\n修復後の状態:")
                    print(f"  Vertices: {len(mesh.vertices)}")
                    print(f"  Faces: {len(mesh.faces)}")
                    print(f"  Is watertight: {mesh.is_watertight}")

                    if mesh.is_watertight:
                        print("✓ Mesh successfully repaired and is now watertight!")
                    else:
                        print("⚠ Mesh repair completed but still not watertight")

                except Exception as repair_error:
                    print(f"⚠ Mesh repair failed: {str(repair_error)}")
                    import traceback
                    traceback.print_exc()

                    QtWidgets.QMessageBox.warning(
                        self,
                        "Mesh Repair Warning",
                        f"Automatic mesh repair failed:\n{str(repair_error)}\n\n"
                        "Calculating center of mass from original mesh."
                    )

            # Center of Massを計算
            center_of_mass = mesh.center_mass
            print(f"\nCalculated center of mass: {center_of_mass}")

            # UIフィールドに設定（高精度、指数表記なし）
            self.inertial_x_input.setText(format_float_no_exp(center_of_mass[0]))
            self.inertial_y_input.setText(format_float_no_exp(center_of_mass[1]))
            self.inertial_z_input.setText(format_float_no_exp(center_of_mass[2]))

            # 成功メッセージ（ダイアログ非表示）
            # repair_msg = ""
            # if repair_performed:
            #     repair_msg = f"\n\nMesh Repair: Performed (in memory only)"
            #     repair_msg += f"\n  Before: Watertight = {original_watertight}"
            #     repair_msg += f"\n  After: Watertight = {mesh.is_watertight}"

            # QtWidgets.QMessageBox.information(
            #     self,
            #     "COM Calculated",
            #     f"Center of Mass successfully calculated!\n\n"
            #     f"Center of mass: [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]\n"
            #     f"Volume: {mesh.volume:.6f} m³\n"
            #     f"Watertight: {'Yes' if mesh.is_watertight else 'No'}"
            #     f"{repair_msg}\n\n"
            #     f"The Inertial Origin has been updated with the calculated COM."
            # )

            print("✓ Center of Mass calculation completed")

        except Exception as e:
            print(f"Error calculating center of mass: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                "Calculation Error",
                f"Failed to calculate center of mass:\n\n{str(e)}"
            )

    def recalculate_inertia(self):
        """STLファイルからtrimeshを使用して慣性を計算（統合関数を使用）"""
        if not self.current_node:
            QtWidgets.QMessageBox.warning(
                self,
                "No Node Selected",
                "Please select a node first."
            )
            return

        # STLファイルが存在するか確認
        if not hasattr(self.current_node, 'stl_file') or not self.current_node.stl_file:
            QtWidgets.QMessageBox.warning(
                self,
                "No STL File",
                "This node has no STL file attached.\nPlease load an STL file first."
            )
            return

        stl_path = self.current_node.stl_file
        if not os.path.exists(stl_path):
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"STL file not found:\n{stl_path}"
            )
            return

        # 質量を取得
        try:
            mass_text = self.mass_input.text()
            if not mass_text or float(mass_text) <= 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Mass",
                    "Please enter a valid mass value (> 0) before calculating inertia."
                )
                return
            mass = float(mass_text)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Mass",
                "Please enter a valid numeric mass value."
            )
            return

        # 既存のInertial Origin値を取得（ユーザー指定の値を使用）
        try:
            center_of_mass = [
                float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0,
                float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0,
                float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0
            ]
        except ValueError:
            center_of_mass = None  # Noneの場合はtrimeshで自動計算

        try:
            # 統合関数を使用して慣性テンソルを計算
            result = calculate_inertia_with_trimesh(
                mesh_file_path=stl_path,
                mass=mass,
                center_of_mass=center_of_mass,
                auto_repair=True
            )

            # エラーチェック
            if not result['success']:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Calculation Failed",
                    f"Failed to calculate inertia:\n\n{result['error_message']}"
                )
                return

            # メッシュ修復失敗時の確認ダイアログ
            if not result['is_watertight'] and result['repair_performed']:
                response = QtWidgets.QMessageBox.question(
                    self,
                    "Mesh Not Watertight",
                    "The mesh is not watertight even after automatic repair.\n"
                    "The calculated inertia may not be accurate.\n\n"
                    "Do you want to continue anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if response == QtWidgets.QMessageBox.No:
                    return

            # 慣性テンソルの検証
            inertia_tensor = result['inertia_tensor']
            validation_result = validate_inertia_tensor(inertia_tensor, mass)

            if not validation_result['valid']:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Validation Warning",
                    f"The calculated inertia tensor may not be physically valid:\n\n"
                    f"{validation_result['message']}\n\n"
                    f"The values will be set anyway, but please verify them."
                )

            # UIフィールドに慣性値を設定（高精度、指数表記なし）
            self.ixx_input.setText(format_float_no_exp(inertia_tensor[0, 0]))
            self.ixy_input.setText(format_float_no_exp(inertia_tensor[0, 1]))
            self.ixz_input.setText(format_float_no_exp(inertia_tensor[0, 2]))
            self.iyy_input.setText(format_float_no_exp(inertia_tensor[1, 1]))
            self.iyz_input.setText(format_float_no_exp(inertia_tensor[1, 2]))
            self.izz_input.setText(format_float_no_exp(inertia_tensor[2, 2]))

            # Inertial Originは既存の値を維持（変更しない）

            # 成功メッセージ（ダイアログ非表示）
            # repair_msg = ""
            # if result['repair_performed']:
            #     repair_msg = "Mesh Repair: Performed (in memory only)\n"

            # QtWidgets.QMessageBox.information(
            #     self,
            #     "Inertia Calculated",
            #     f"Inertia tensor successfully calculated and applied!\n\n"
            #     f"Mass: {result['mass']:.6f} kg\n"
            #     f"Volume: {result['volume']:.9f} m³\n"
            #     f"Density: {result['density']:.6f} kg/m³\n"
            #     f"Watertight: {'Yes' if result['is_watertight'] else 'No'}\n"
            #     f"{repair_msg}"
            #     f"\nCenter of mass (used): [{result['center_of_mass'][0]:.6f}, {result['center_of_mass'][1]:.6f}, {result['center_of_mass'][2]:.6f}]\n"
            #     f"Center of mass (trimesh): [{result['trimesh_com'][0]:.6f}, {result['trimesh_com'][1]:.6f}, {result['trimesh_com'][2]:.6f}]\n\n"
            #     f"Inertia tensor diagonal:\n"
            #     f"  Ixx: {inertia_tensor[0, 0]:.6f}\n"
            #     f"  Iyy: {inertia_tensor[1, 1]:.6f}\n"
            #     f"  Izz: {inertia_tensor[2, 2]:.6f}\n\n"
            #     f"{validation_result['message']}"
            # )

            # 計算結果を即座に適用
            self.set_inertia()

        except Exception as e:
            print(f"Error calculating inertia: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                "Calculation Error",
                f"Failed to calculate inertia:\n\n{str(e)}"
            )

    def zero_off_diagonal_inertia(self):
        """Zero out off-diagonal elements of inertia tensor (Ixy, Ixz, Iyz)"""
        if not self.current_node:
            QtWidgets.QMessageBox.warning(
                self,
                "No Node Selected",
                "Please select a node first."
            )
            return

        # Set off-diagonal elements to zero in UI fields
        self.ixy_input.setText("0.0")
        self.ixz_input.setText("0.0")
        self.iyz_input.setText("0.0")

        # Update internal parameters by calling update_inertia
        self.update_inertia()

        print(f"Zeroed off-diagonal inertia elements for node: {self.current_node.name()}")

class SettingsDialog(QtWidgets.QDialog):
    """設定ダイアログ"""
    def __init__(self, graph, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.graph = graph
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        """UIの初期化"""
        import math
        layout = QtWidgets.QVBoxLayout(self)

        # 統一されたボタンスタイル（グローバル定数を使用）
        self.button_style = UNIFIED_BUTTON_STYLE

        # Default Joint Settings セクション
        group_box = QtWidgets.QGroupBox("Default Joint Settings")
        group_layout = QtWidgets.QGridLayout()

        row = 0

        # Default Effort
        group_layout.addWidget(QtWidgets.QLabel("Default Effort:"), row, 0)
        self.effort_input = QtWidgets.QLineEdit()
        self.effort_input.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        self.effort_input.setText(f"{self.graph.default_joint_effort:.3f}")
        self.effort_input.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        group_layout.addWidget(self.effort_input, row, 1)
        group_layout.addWidget(QtWidgets.QLabel("N*m"), row, 2)
        group_layout.addWidget(QtWidgets.QLabel("Max:"), row, 3)
        self.max_effort_input = QtWidgets.QLineEdit()
        self.max_effort_input.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        self.max_effort_input.setText(f"{self.graph.default_max_effort:.3f}")
        group_layout.addWidget(self.max_effort_input, row, 4)
        group_layout.addWidget(QtWidgets.QLabel("N*m"), row, 5)
        row += 1

        # Default Velocity (Maxも同じ行に)
        group_layout.addWidget(QtWidgets.QLabel("Default Velocity:"), row, 0)
        self.velocity_rad_input = QtWidgets.QLineEdit()
        self.velocity_rad_input.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        self.velocity_rad_input.setText(f"{self.graph.default_joint_velocity:.4f}")
        self.velocity_rad_input.returnPressed.connect(lambda: self._convert_rad_to_deg(
            self.velocity_rad_input, self.velocity_deg_input))
        group_layout.addWidget(self.velocity_rad_input, row, 1)
        group_layout.addWidget(QtWidgets.QLabel("rad/s"), row, 2)

        self.velocity_deg_input = QtWidgets.QLineEdit()
        self.velocity_deg_input.setValidator(QDoubleValidator(0.0, 100000.0, 3))
        self.velocity_deg_input.setText(f"{math.degrees(self.graph.default_joint_velocity):.3f}")
        self.velocity_deg_input.returnPressed.connect(lambda: self._convert_deg_to_rad(
            self.velocity_deg_input, self.velocity_rad_input))
        group_layout.addWidget(QtWidgets.QLabel("("), row, 3, QtCore.Qt.AlignmentFlag.AlignRight)
        group_layout.addWidget(self.velocity_deg_input, row, 4)
        group_layout.addWidget(QtWidgets.QLabel("deg/s)"), row, 5)
        
        # Max Velocity (同じ行に)
        group_layout.addWidget(QtWidgets.QLabel("Max:"), row, 6, QtCore.Qt.AlignmentFlag.AlignRight)
        self.max_velocity_rad_input = QtWidgets.QLineEdit()
        self.max_velocity_rad_input.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        self.max_velocity_rad_input.setText(f"{self.graph.default_max_velocity:.4f}")
        self.max_velocity_rad_input.returnPressed.connect(lambda: self._convert_rad_to_deg(
            self.max_velocity_rad_input, self.max_velocity_deg_input))
        group_layout.addWidget(self.max_velocity_rad_input, row, 7)
        group_layout.addWidget(QtWidgets.QLabel("rad/s"), row, 8)

        self.max_velocity_deg_input = QtWidgets.QLineEdit()
        self.max_velocity_deg_input.setValidator(QDoubleValidator(0.0, 100000.0, 3))
        self.max_velocity_deg_input.setText(f"{math.degrees(self.graph.default_max_velocity):.3f}")
        self.max_velocity_deg_input.returnPressed.connect(lambda: self._convert_deg_to_rad(
            self.max_velocity_deg_input, self.max_velocity_rad_input))
        group_layout.addWidget(QtWidgets.QLabel("("), row, 9, QtCore.Qt.AlignmentFlag.AlignRight)
        group_layout.addWidget(self.max_velocity_deg_input, row, 10)
        group_layout.addWidget(QtWidgets.QLabel("deg/s)"), row, 11)
        row += 1

        # Default Margin
        group_layout.addWidget(QtWidgets.QLabel("Default Margin:"), row, 0)
        self.margin_rad_input = QtWidgets.QLineEdit()
        self.margin_rad_input.setValidator(QDoubleValidator(0.0, 10.0, 4))
        self.margin_rad_input.setText(f"{self.graph.default_margin:.4f}")
        self.margin_rad_input.returnPressed.connect(lambda: self._convert_rad_to_deg(
            self.margin_rad_input, self.margin_deg_input))
        group_layout.addWidget(self.margin_rad_input, row, 1)
        group_layout.addWidget(QtWidgets.QLabel("rad"), row, 2)

        self.margin_deg_input = QtWidgets.QLineEdit()
        self.margin_deg_input.setValidator(QDoubleValidator(0.0, 360.0, 3))
        self.margin_deg_input.setText(f"{math.degrees(self.graph.default_margin):.3f}")
        self.margin_deg_input.returnPressed.connect(lambda: self._convert_deg_to_rad(
            self.margin_deg_input, self.margin_rad_input))
        group_layout.addWidget(QtWidgets.QLabel("("), row, 3, QtCore.Qt.AlignmentFlag.AlignRight)
        group_layout.addWidget(self.margin_deg_input, row, 4)
        group_layout.addWidget(QtWidgets.QLabel("deg)"), row, 5)
        row += 1

        # Default Armature
        group_layout.addWidget(QtWidgets.QLabel("Default Armature:"), row, 0)
        self.armature_input = QtWidgets.QLineEdit()
        self.armature_input.setValidator(QDoubleValidator(0.0, 100.0, 4))
        self.armature_input.setText(f"{self.graph.default_armature:.4f}")
        group_layout.addWidget(self.armature_input, row, 1)
        group_layout.addWidget(QtWidgets.QLabel("kg*m²"), row, 2)
        row += 1

        # Default Frictionloss
        group_layout.addWidget(QtWidgets.QLabel("Default Frictionloss:"), row, 0)
        self.frictionloss_input = QtWidgets.QLineEdit()
        self.frictionloss_input.setValidator(QDoubleValidator(0.0, 100.0, 4))
        self.frictionloss_input.setText(f"{self.graph.default_frictionloss:.4f}")
        group_layout.addWidget(self.frictionloss_input, row, 1)
        group_layout.addWidget(QtWidgets.QLabel("N*m"), row, 2)
        row += 1

        # Default Stiffness (kp)
        group_layout.addWidget(QtWidgets.QLabel("Default Stiffness (kp):"), row, 0)
        self.stiffness_kp_input = QtWidgets.QLineEdit()
        self.stiffness_kp_input.setValidator(QDoubleValidator(0.0, 10000.0, 3))
        self.stiffness_kp_input.setText(f"{self.graph.default_stiffness_kp:.3f}")
        group_layout.addWidget(self.stiffness_kp_input, row, 1)
        group_layout.addWidget(QtWidgets.QLabel("N*m/rad"), row, 2)
        row += 1

        # Default Damping (kv)
        group_layout.addWidget(QtWidgets.QLabel("Default Damping (kv):"), row, 0)
        self.damping_kv_input = QtWidgets.QLineEdit()
        self.damping_kv_input.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        self.damping_kv_input.setText(f"{self.graph.default_damping_kv:.3f}")
        group_layout.addWidget(self.damping_kv_input, row, 1)
        group_layout.addWidget(QtWidgets.QLabel("N*m*s/rad"), row, 2)
        row += 1

        # Apply to All Nodes ボタン（右寄せ）
        apply_all_button = QtWidgets.QPushButton("Apply to All Nodes")
        apply_all_button.setStyleSheet(self.button_style)
        apply_all_button.setAutoDefault(False)
        apply_all_button.clicked.connect(self.apply_to_all_nodes)
        group_layout.addWidget(apply_all_button, row, 9, 1, 3, QtCore.Qt.AlignmentFlag.AlignRight)
        row += 1

        # Default Angle Range
        group_layout.addWidget(QtWidgets.QLabel("Default Angle Range:"), row, 0)
        group_layout.addWidget(QtWidgets.QLabel("+/-"), row, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        self.angle_range_rad_input = QtWidgets.QLineEdit()
        self.angle_range_rad_input.setValidator(QDoubleValidator(0.0, 10.0, 4))
        self.angle_range_rad_input.setText(f"{self.graph.default_angle_range:.4f}")
        self.angle_range_rad_input.returnPressed.connect(lambda: self._convert_rad_to_deg(
            self.angle_range_rad_input, self.angle_range_deg_input))
        group_layout.addWidget(self.angle_range_rad_input, row, 2)
        group_layout.addWidget(QtWidgets.QLabel("rad"), row, 3)

        group_layout.addWidget(QtWidgets.QLabel("+/-"), row, 4, QtCore.Qt.AlignmentFlag.AlignRight)
        self.angle_range_deg_input = QtWidgets.QLineEdit()
        self.angle_range_deg_input.setValidator(QDoubleValidator(0.0, 360.0, 3))
        self.angle_range_deg_input.setText(f"{math.degrees(self.graph.default_angle_range):.3f}")
        self.angle_range_deg_input.returnPressed.connect(lambda: self._convert_deg_to_rad(
            self.angle_range_deg_input, self.angle_range_rad_input))
        group_layout.addWidget(self.angle_range_deg_input, row, 5)
        group_layout.addWidget(QtWidgets.QLabel("deg"), row, 6)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        # MJCF Settings セクション
        mjcf_group = QtWidgets.QGroupBox("MJCF Export Settings")
        mjcf_layout = QtWidgets.QHBoxLayout()  # GridLayoutからHBoxLayoutに変更

        # Base Link Height設定
        # [Default base_link height(m)] [15px][インプットフィールド][右余白]
        mjcf_layout.addWidget(QtWidgets.QLabel("Default base_link height (m):"))
        mjcf_layout.addSpacing(15)  # 15pxのスペース
        self.base_link_height_input = QtWidgets.QLineEdit()
        self.base_link_height_input.setValidator(QDoubleValidator(0.0, 10.0, 3))
        self.base_link_height_input.setText(f"{self.graph.default_base_link_height:.4f}")
        self.base_link_height_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)  # 左寄せ固定
        mjcf_layout.addWidget(self.base_link_height_input)
        mjcf_layout.addStretch()  # 右余白

        mjcf_group.setLayout(mjcf_layout)
        layout.addWidget(mjcf_group)

        # Node Grid セクション
        grid_group = QtWidgets.QGroupBox("Node Grid")
        grid_layout = QtWidgets.QHBoxLayout()

        # チェックボックス
        self.grid_enabled_checkbox = QtWidgets.QCheckBox()
        self.grid_enabled_checkbox.setChecked(self.graph.node_grid_enabled)
        grid_layout.addWidget(self.grid_enabled_checkbox)

        grid_layout.addWidget(QtWidgets.QLabel("Grid Size:"))

        # グリッドサイズ入力
        self.grid_size_input = QtWidgets.QLineEdit()
        self.grid_size_input.setValidator(QtGui.QIntValidator(1, 1000))
        self.grid_size_input.setText(str(self.graph.node_grid_size))
        self.grid_size_input.setFixedWidth(80)
        grid_layout.addWidget(self.grid_size_input)

        grid_layout.addWidget(QtWidgets.QLabel("pixels"))

        # Snap All to Gridボタン
        snap_all_button = QtWidgets.QPushButton("Snap All to Grid")
        snap_all_button.setStyleSheet(self.button_style)
        snap_all_button.setAutoDefault(False)
        snap_all_button.clicked.connect(self.snap_all_nodes_to_grid)
        grid_layout.addWidget(snap_all_button)

        grid_layout.addStretch()

        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # Mesh Highlight セクション
        highlight_group = QtWidgets.QGroupBox("Mesh Highlight")
        highlight_layout = QtWidgets.QHBoxLayout()

        highlight_layout.addWidget(QtWidgets.QLabel("Highlight Color:"))

        # カラーボックス（色を表示）
        self.highlight_color_box = QtWidgets.QLabel()
        self.highlight_color_box.setFixedSize(60, 30)
        self.highlight_color_box.setStyleSheet(
            f"background-color: {self.graph.highlight_color}; border: 1px solid black;"
        )
        highlight_layout.addWidget(self.highlight_color_box)

        # Pickボタン
        pick_button = QtWidgets.QPushButton("Pick")
        pick_button.setStyleSheet(self.button_style)
        pick_button.setAutoDefault(False)  # Returnキーで誤って起動されないように
        pick_button.clicked.connect(self.pick_highlight_color)
        highlight_layout.addWidget(pick_button)

        highlight_layout.addStretch()
        highlight_group.setLayout(highlight_layout)
        layout.addWidget(highlight_group)

        # Collision Color セクション
        collision_group = QtWidgets.QGroupBox("Collision Color")
        collision_layout = QtWidgets.QHBoxLayout()

        # KitchenColorPicker
        self.collision_color_picker = KitchenColorPicker(
            parent_widget=self,
            initial_color=self.graph.collision_color,
            enable_alpha=True,
            on_color_changed=self._on_collision_color_changed
        )

        # レイアウト: Collision Color:[カラーボックス] alpha: [alpha] [Pick]ボタン
        collision_layout.addWidget(QtWidgets.QLabel("Collision Color:"))
        collision_layout.addWidget(self.collision_color_picker.color_sample)
        collision_layout.addWidget(QtWidgets.QLabel("alpha:"))
        # RGBのインプットフィールドは非表示（alphaのみ表示）
        if len(self.collision_color_picker.color_inputs) >= 4:
            # RGBのインプットフィールドを非表示
            for i in range(3):
                self.collision_color_picker.color_inputs[i].setVisible(False)
            collision_layout.addWidget(self.collision_color_picker.color_inputs[3])  # alphaのみ
        self.collision_color_picker.pick_button.setAutoDefault(False)  # Returnキーで誤って起動されないように
        collision_layout.addWidget(self.collision_color_picker.pick_button)
        collision_layout.addStretch()

        collision_group.setLayout(collision_layout)
        layout.addWidget(collision_group)

        # ボタン
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        ok_button = QtWidgets.QPushButton("OK")
        ok_button.setAutoDefault(False)  # Returnキーでウィンドウが閉じないように
        ok_button.setStyleSheet(self.button_style)  # 統一スタイルを適用
        ok_button.clicked.connect(self.accept_settings)
        button_layout.addWidget(ok_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.setAutoDefault(False)
        cancel_button.setStyleSheet(self.button_style)  # 統一スタイルを適用
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        # すべての数値入力フィールドに指数表記を防ぐフォーマッターを設定
        self.effort_input.editingFinished.connect(
            lambda: self._format_number_input(self.effort_input, 3))
        self.max_effort_input.editingFinished.connect(
            lambda: self._format_number_input(self.max_effort_input, 3))
        self.velocity_rad_input.editingFinished.connect(
            lambda: self._format_number_input(self.velocity_rad_input, 4))
        self.velocity_deg_input.editingFinished.connect(
            lambda: self._format_number_input(self.velocity_deg_input, 3))
        self.max_velocity_rad_input.editingFinished.connect(
            lambda: self._format_number_input(self.max_velocity_rad_input, 4))
        self.max_velocity_deg_input.editingFinished.connect(
            lambda: self._format_number_input(self.max_velocity_deg_input, 3))
        self.margin_rad_input.editingFinished.connect(
            lambda: self._format_number_input(self.margin_rad_input, 4))
        self.margin_deg_input.editingFinished.connect(
            lambda: self._format_number_input(self.margin_deg_input, 3))
        self.armature_input.editingFinished.connect(
            lambda: self._format_number_input(self.armature_input, 4))
        self.frictionloss_input.editingFinished.connect(
            lambda: self._format_number_input(self.frictionloss_input, 4))
        self.stiffness_kp_input.editingFinished.connect(
            lambda: self._format_number_input(self.stiffness_kp_input, 3))
        self.damping_kv_input.editingFinished.connect(
            lambda: self._format_number_input(self.damping_kv_input, 3))
        self.angle_range_rad_input.editingFinished.connect(
            lambda: self._format_number_input(self.angle_range_rad_input, 4))
        self.angle_range_deg_input.editingFinished.connect(
            lambda: self._format_number_input(self.angle_range_deg_input, 3))
        self.base_link_height_input.editingFinished.connect(
            lambda: self._format_number_input(self.base_link_height_input, 4))
        
        # すべての入力フィールドの横幅を現在の6割のサイズに変更
        # まず、すべての入力フィールドをリストにまとめる
        input_fields = [
            self.effort_input,
            self.max_effort_input,
            self.velocity_rad_input,
            self.velocity_deg_input,
            self.max_velocity_rad_input,
            self.max_velocity_deg_input,
            self.margin_rad_input,
            self.margin_deg_input,
            self.armature_input,
            self.frictionloss_input,
            self.stiffness_kp_input,
            self.damping_kv_input,
            self.angle_range_rad_input,
            self.angle_range_deg_input,
            self.base_link_height_input
        ]
        
        # 各入力フィールドの現在のサイズを取得して、6割に設定
        for field in input_fields:
            # 現在のサイズヒントを取得（デフォルトサイズが設定されていない場合は100を基準とする）
            current_width = field.sizeHint().width()
            if current_width <= 0:
                current_width = 100  # デフォルト値
            # 6割のサイズに設定
            new_width = int(current_width * 0.6)
            field.setMaximumWidth(new_width)
            field.setMinimumWidth(new_width)

    def pick_highlight_color(self):
        """カラーピッカーを開いてハイライトカラーを選択"""
        # 現在の色を取得
        current_color = QtGui.QColor(self.graph.highlight_color)

        # カスタムカラーダイアログを使用
        dialog = CustomColorDialog(current_color, self)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            color = dialog.currentColor()
            if color.isValid():
                # 色を#RRGGBB形式で保存
                hex_color = color.name()
                self.graph.highlight_color = hex_color
                # カラーボックスを更新
                self.highlight_color_box.setStyleSheet(
                    f"background-color: {hex_color}; border: 1px solid black;"
                )

    def _convert_rad_to_deg(self, rad_input, deg_input):
        """radの入力をdegに変換"""
        import math
        try:
            rad_value = float(rad_input.text())
            # 小数点4桁で丸める
            rad_value = round(rad_value, 4)
            rad_input.setText(f"{rad_value:.4f}")
            deg_value = math.degrees(rad_value)
            deg_input.setText(f"{deg_value:.3f}")
        except ValueError:
            pass

    def _convert_deg_to_rad(self, deg_input, rad_input):
        """degの入力をradに変換"""
        import math
        try:
            deg_value = float(deg_input.text())
            rad_value = math.radians(deg_value)
            # 小数点4桁で丸める
            rad_value = round(rad_value, 4)
            rad_input.setText(f"{rad_value:.4f}")
        except ValueError:
            pass

    def _format_number_input(self, line_edit, decimal_places):
        """入力フィールドの数値を指数表記を使わずにフォーマット"""
        try:
            value = float(line_edit.text())
            # 指数表記を避けて通常の小数表記でフォーマット
            line_edit.setText(f"{value:.{decimal_places}f}")
        except (ValueError, AttributeError):
            pass

    def _on_collision_color_changed(self, rgba_color):
        """Collision Colorが変更されたときのコールバック"""
        # graphのcollision_colorを更新
        self.graph.collision_color = rgba_color.copy()
        print(f"Collision color updated: RGBA={rgba_color}")

    def accept_settings(self):
        """設定を適用"""
        try:
            # 新しい設定値を取得
            effort = float(self.effort_input.text())
            max_effort = float(self.max_effort_input.text())
            velocity_rad = float(self.velocity_rad_input.text())
            max_velocity_rad = float(self.max_velocity_rad_input.text())
            margin_rad = float(self.margin_rad_input.text())
            armature = float(self.armature_input.text())
            frictionloss = float(self.frictionloss_input.text())
            stiffness_kp = float(self.stiffness_kp_input.text())
            damping_kv = float(self.damping_kv_input.text())
            angle_range_rad = float(self.angle_range_rad_input.text())

            # グラフに設定を適用
            self.graph.default_joint_effort = effort
            self.graph.default_max_effort = max_effort
            self.graph.default_joint_velocity = velocity_rad
            self.graph.default_max_velocity = max_velocity_rad
            self.graph.default_margin = margin_rad
            self.graph.default_armature = armature
            self.graph.default_frictionloss = frictionloss
            self.graph.default_stiffness_kp = stiffness_kp
            self.graph.default_damping_kv = damping_kv
            self.graph.default_angle_range = angle_range_rad

            # MJCF設定
            base_link_height = float(self.base_link_height_input.text())
            self.graph.default_base_link_height = base_link_height

            # Node Grid設定
            grid_enabled = self.grid_enabled_checkbox.isChecked()
            grid_size = int(self.grid_size_input.text())
            self.graph.node_grid_enabled = grid_enabled
            self.graph.node_grid_size = grid_size

            # グリッド表示を更新
            self.graph.update_grid_display()

            # highlight_colorは既にpick_highlight_colorで更新済み

            self.accept()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values."
            )

    def snap_all_nodes_to_grid(self):
        """全てのノードを最寄りのグリッドにスナップ"""
        try:
            # Grid Size入力から現在の値を取得
            grid_size = int(self.grid_size_input.text())

            # 全てのノードに対してスナップ処理を実行
            snapped_count = 0
            for node in self.graph.all_nodes():
                node_pos = node.pos()
                if isinstance(node_pos, (list, tuple)):
                    current_x, current_y = node_pos[0], node_pos[1]
                else:
                    current_x, current_y = node_pos.x(), node_pos.y()

                # グリッドにスナップ
                snapped_x = round(current_x / grid_size) * grid_size
                snapped_y = round(current_y / grid_size) * grid_size

                # 位置が変わった場合のみ更新
                if abs(snapped_x - current_x) > 0.1 or abs(snapped_y - current_y) > 0.1:
                    node.set_pos(snapped_x, snapped_y)
                    snapped_count += 1

            print(f"Snapped {snapped_count} nodes to grid (size: {grid_size})")

        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Grid Size",
                "Please enter a valid grid size value."
            )
        except Exception as e:
            print(f"Error snapping nodes to grid: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_to_all_nodes(self):
        """全ノードの該当パラメータに設定値を適用"""
        try:
            # 入力フィールドから値を取得
            effort = float(self.effort_input.text())
            velocity_rad = float(self.velocity_rad_input.text())
            damping_kv = float(self.damping_kv_input.text())
            stiffness_kp = float(self.stiffness_kp_input.text())
            margin_rad = float(self.margin_rad_input.text())
            armature = float(self.armature_input.text())
            frictionloss = float(self.frictionloss_input.text())

            # 全ノードに値を適用
            updated_count = 0
            for node in self.graph.all_nodes():
                # 各ノードが該当パラメータを持っているかチェックして適用
                if hasattr(node, 'joint_effort'):
                    node.joint_effort = effort
                if hasattr(node, 'joint_velocity'):
                    node.joint_velocity = velocity_rad
                if hasattr(node, 'joint_damping'):
                    node.joint_damping = damping_kv
                if hasattr(node, 'joint_stiffness'):
                    node.joint_stiffness = stiffness_kp
                if hasattr(node, 'joint_margin'):
                    node.joint_margin = margin_rad
                if hasattr(node, 'joint_armature'):
                    node.joint_armature = armature
                if hasattr(node, 'joint_frictionloss'):
                    node.joint_frictionloss = frictionloss
                updated_count += 1

            print(f"Applied settings to {updated_count} nodes: effort={effort}, velocity={velocity_rad}, damping={damping_kv}, stiffness={stiffness_kp}, margin={margin_rad}, armature={armature}, frictionloss={frictionloss}")

            QtWidgets.QMessageBox.information(
                self,
                "Settings Applied",
                f"Applied settings to {updated_count} nodes:\n\n"
                f"Effort: {effort} N*m\n"
                f"Velocity: {velocity_rad:.4f} rad/s\n"
                f"Damping: {damping_kv} N*m*s/rad\n"
                f"Stiffness: {stiffness_kp} N*m/rad\n"
                f"Margin: {margin_rad:.4f} rad\n"
                f"Armature: {armature} kg*m²\n"
                f"Frictionloss: {frictionloss} N*m"
            )

        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values for all parameters."
            )
        except Exception as e:
            print(f"Error applying settings to all nodes: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                f"An error occurred while applying settings:\n{str(e)}"
            )


class CircularProgressBar(QtWidgets.QWidget):
    """円形プログレスバー (100から0へカウントダウン)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 100  # 初期値を100に変更
        self.setFixedSize(100, 100)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor, QConicalGradient, QFont
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background circle
        pen = QPen(QColor(50, 50, 50, 180))
        pen.setWidth(8)
        painter.setPen(pen)
        painter.drawEllipse(10, 10, 80, 80)

        # Progress arc (light blue) - 残り処理量を表示
        gradient = QConicalGradient(50, 50, 90)
        gradient.setColorAt(0, QColor(100, 180, 255, 200))
        gradient.setColorAt(1, QColor(150, 220, 255, 200))

        pen = QPen(gradient, 8)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        span_angle = int(self.value * 360 / 100 * 16)
        painter.drawArc(10, 10, 80, 80, 90 * 16, -span_angle)

        # 残り処理パーセントを中央に表示
        painter.setPen(QColor(200, 200, 200, 220))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 10, 80, 80, QtCore.Qt.AlignmentFlag.AlignCenter, f"{int(self.value)}%")


class STLViewerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(STLViewerWidget, self).__init__(parent)
        self.stl_actors = {}
        self.transforms = {}
        self.base_connected_node = None
        self.text_actors = []
        self.inertial_origin_actors = {}  # Inertial Origin表示用のアクター
        self.collider_actors = {}  # Collider表示用のアクター
        self.collider_display_enabled = False  # Collider表示のON/OFF状態

        layout = QtWidgets.QVBoxLayout(self)

        # Progress bar (initially hidden)
        self.progress_bar = CircularProgressBar(self)
        self.progress_bar.hide()

        # Use QLabel instead of QVTKRenderWindowInteractor for M4 Mac compatibility
        self.vtk_display = QLabel(self)
        self.vtk_display.setMinimumSize(100, 1)  # 最小幅100px、高さは1pxまで縮小可能
        self.vtk_display.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #555;
            }
        """)
        self.vtk_display.setAlignment(Qt.AlignCenter)
        self.vtk_display.setText("3D Viewer - Loading...")
        self.vtk_display.setScaledContents(False)
        self.vtk_display.setMouseTracking(True)
        self.vtk_display.setFocusPolicy(Qt.StrongFocus)

        layout.addWidget(self.vtk_display)

        # Create offscreen VTK render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(800, 600)

        self.renderer = vtk.vtkRenderer()
        # レンダラーのビューポートをウィンドウ全体に設定（デフォルトは0,0,1,1だが明示）
        self.renderer.SetViewport(0, 0, 1, 1)
        self.render_window.AddRenderer(self.renderer)

        # No more interactor needed - using offscreen rendering
        self.iren = None

        # Initialize offscreen renderer utility
        self.offscreen_renderer = OffscreenRenderer(self.render_window, self.renderer)

        # Initialize camera controller
        self.camera_controller = CameraController(self.renderer, origin=[0, 0, 0])

        # Initialize mouse drag state
        self.mouse_drag = MouseDragState(self.vtk_display)

        # Install event filter for mouse events
        self.vtk_display.installEventFilter(self)

        # 統一されたボタンスタイル（グローバル定数を使用）
        button_style = UNIFIED_BUTTON_STYLE

        # ボタンとスライダーのレイアウト
        button_layout = QtWidgets.QVBoxLayout()  # 垂直レイアウトに変更
        button_layout.setSpacing(2)  # ボタン間のスペースを狭く設定

        # 1行目: Front, Side, Topボタン
        first_row_layout = QtWidgets.QHBoxLayout()

        # Frontボタン
        self.front_button = QtWidgets.QPushButton("Front")
        self.front_button.setStyleSheet(button_style)
        self.front_button.setFixedWidth(50)
        self.front_button.clicked.connect(self.reset_camera_front)
        first_row_layout.addWidget(self.front_button)

        # Sideボタン
        self.side_button = QtWidgets.QPushButton("Side")
        self.side_button.setStyleSheet(button_style)
        self.side_button.setFixedWidth(50)
        self.side_button.clicked.connect(self.reset_camera_side)
        first_row_layout.addWidget(self.side_button)

        # Topボタン
        self.top_button = QtWidgets.QPushButton("Top")
        self.top_button.setStyleSheet(button_style)
        self.top_button.setFixedWidth(50)
        self.top_button.clicked.connect(self.reset_camera_top)
        first_row_layout.addWidget(self.top_button)

        first_row_layout.addStretch()

        button_layout.addLayout(first_row_layout)
        button_layout.addSpacing(10)  # Front Side Top行とMesh Wireframe Collider行の間に10pxのスペース

        # 2行目: Mesh, Wireframe, Colliderボタン
        second_row_layout = QtWidgets.QHBoxLayout()

        # トグルボタン用の統一スタイル
        # ON状態: #0055ff、押下時: 深い青（#1a3a5a）
        on_bg_color = "#0055ff"  # ON状態の色
        on_border_color = "#0055ff"  # ON状態のボーダー色
        pressed_bg_color = "#1a3a5a"  # 現在のON色を押下時に使用
        pressed_border_color = "#2a5a8a"  # 現在のONボーダー色を押下時に使用
        toggle_button_style = f"""
            QPushButton {{
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #5a5a5a;
                border-radius: 7px;
                padding: 3px 8px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: #4a4a4a;
            }}
            QPushButton:pressed {{
                background-color: {pressed_bg_color};
                border: 1px solid {pressed_border_color};
                color: #ffffff;
            }}
            QPushButton:checked {{
                background-color: {on_bg_color};
                border: 1px solid {on_border_color};
                color: #ffffff;
            }}
            QPushButton:checked:hover {{
                background-color: {on_bg_color};
                border: 1px solid {on_border_color};
                opacity: 0.9;
            }}
            QPushButton:checked:pressed {{
                background-color: {pressed_bg_color};
                border: 1px solid {pressed_border_color};
                color: #ffffff;
            }}
        """

        # Meshスイッチ（デフォルトON）
        self.mesh_toggle = QtWidgets.QPushButton("Mesh")
        self.mesh_toggle.setCheckable(True)
        self.mesh_toggle.setChecked(True)  # デフォルトでON
        self.mesh_toggle.setFixedWidth(80)
        self.mesh_toggle.setStyleSheet(toggle_button_style)
        self.mesh_toggle.toggled.connect(self.toggle_mesh)
        second_row_layout.addWidget(self.mesh_toggle)

        # Wireframeスイッチ（デフォルトOFF）
        self.wireframe_toggle = QtWidgets.QPushButton("Wireframe")
        self.wireframe_toggle.setCheckable(True)
        self.wireframe_toggle.setChecked(False)  # デフォルトでOFF
        self.wireframe_toggle.setFixedWidth(80)
        self.wireframe_toggle.setStyleSheet(toggle_button_style)
        self.wireframe_toggle.toggled.connect(self.toggle_wireframe)
        second_row_layout.addWidget(self.wireframe_toggle)

        # Colliderスイッチ（デフォルトOFF）- oldファイルと同じスタイルを使用
        self.collider_toggle = QtWidgets.QPushButton("Collider")
        self.collider_toggle.setCheckable(True)
        self.collider_toggle.setChecked(False)  # デフォルトでOFF
        self.collider_toggle.setFixedWidth(80)
        self.collider_toggle.setStyleSheet(toggle_button_style)
        self.collider_toggle.toggled.connect(self.toggle_collider_display)
        second_row_layout.addWidget(self.collider_toggle)

        second_row_layout.addStretch()

        button_layout.addLayout(second_row_layout)
        button_layout.addSpacing(5)  # Back-ground-color行の前に5pxのスペースを追加

        # Background スライダーのレイアウト
        bg_layout = QtWidgets.QHBoxLayout()
        bg_label = QtWidgets.QLabel("background-color:")
        bg_layout.addWidget(bg_label)

        self.bg_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_slider.setMinimum(-100)  # 黒
        self.bg_slider.setMaximum(100)   # 白
        self.bg_slider.setValue(-80)      # デフォルト値を暗めに
        self.bg_slider.valueChanged.connect(self.update_background)
        bg_layout.addWidget(self.bg_slider)

        button_layout.addLayout(bg_layout)
        layout.addLayout(button_layout)

        self.setup_camera()
        self.coordinate_axes_actor = self.create_coordinate_axes()
        self.renderer.AddActor(self.coordinate_axes_actor)

        self.rotation_timer = QtCore.QTimer()
        self.rotation_timer.timeout.connect(self.update_rotation)
        self.rotating_node = None
        self.original_transforms = {}
        self.current_angle = 0
        self.rotation_direction = 1  # 1: 増加方向, -1: 減少方向
        self.rotation_paused = False  # 限界点での一時停止フラグ
        self.pause_counter = 0  # 一時停止カウンター
        self.follow_children = True  # 子ノードも一緒に回転するか

        # ライティングの設定
        # 上45度前方からのメインライト
        light1 = vtk.vtkLight()
        light1.SetPosition(0.5, 0.5, 1.0)
        light1.SetIntensity(0.7)
        light1.SetLightTypeToSceneLight()
        
        # 左後方からの補助ライト
        light2 = vtk.vtkLight()
        light2.SetPosition(-1.0, -0.5, 0.2)
        light2.SetIntensity(0.7)
        light2.SetLightTypeToSceneLight()
        
        # 右後方からの補助ライト
        light3 = vtk.vtkLight()
        light3.SetPosition(0.3, -1.0, 0.2)
        light3.SetIntensity(0.7)
        light3.SetLightTypeToSceneLight()

        # 正面からの補助ライト
        light4 = vtk.vtkLight()
        light4.SetPosition(1.0, 0.0, 0.3)
        light4.SetIntensity(0.3)
        light4.SetLightTypeToSceneLight()

        # バランスの取れたアンビエント光
        self.renderer.SetAmbient(0.7, 0.7, 0.7)
        self.renderer.LightFollowCameraOff()
        self.renderer.AddLight(light1)
        self.renderer.AddLight(light2)
        self.renderer.AddLight(light3)
        self.renderer.AddLight(light4)


        # 初期の背景色を設定（暗めのグレー）
        initial_bg = (-80 + 100) / 200.0  # -80のスライダー値を0-1の範囲に変換
        self.renderer.SetBackground(initial_bg, initial_bg, initial_bg)

        # ハイライト関連の変数
        self.highlighted_node = None
        self.original_color = None
        self.highlight_timer = QTimer(self)
        self.highlight_timer.timeout.connect(self._toggle_highlight)
        self.highlight_state = False  # 点滅の状態

        # Delay initial render to avoid blocking
        QTimer.singleShot(100, self.render_to_image)

    def show_progress(self, show=True):
        """Show or hide the progress bar"""
        if show:
            # Center the progress bar
            x = (self.vtk_display.width() - self.progress_bar.width()) // 2
            y = (self.vtk_display.height() - self.progress_bar.height()) // 2
            self.progress_bar.move(x, y)
            self.progress_bar.raise_()
            self.progress_bar.show()
        else:
            self.progress_bar.hide()

    def resizeEvent(self, event):
        """Reposition progress bar on resize"""
        super().resizeEvent(event)
        if self.progress_bar.isVisible():
            x = (self.vtk_display.width() - self.progress_bar.width()) // 2
            y = (self.vtk_display.height() - self.progress_bar.height()) // 2
            self.progress_bar.move(x, y)

    def render_to_image(self):
        """Render VTK scene offscreen and display as image in QLabel"""
        try:
            # 重要: VTKウィンドウサイズをQLabelの実際のサイズに合わせる
            widget_size = self.vtk_display.size()
            widget_width = max(widget_size.width(), 10)  # 最小サイズ10
            widget_height = max(widget_size.height(), 10)

            # VTKウィンドウサイズを更新
            current_size = self.render_window.GetSize()
            if current_size[0] != widget_width or current_size[1] != widget_height:
                self.render_window.SetSize(widget_width, widget_height)
                # ウィンドウサイズ変更時はクリッピング範囲も更新
                self.renderer.ResetCameraClippingRange()

            # デバッグ: カメラとウィンドウの状態を表示（最初の数回のみ）
            if not hasattr(self, '_render_count'):
                self._render_count = 0

            if self._render_count < 3:
                camera = self.renderer.GetActiveCamera()
                win_size = self.render_window.GetSize()
                print(f"\n=== Render #{self._render_count} ===")
                print(f"VTK Window Size: {win_size}")
                print(f"QLabel Widget Size: {widget_width}x{widget_height}")
                print(f"Camera Position: {camera.GetPosition()}")
                print(f"Camera FocalPoint: {camera.GetFocalPoint()}")
                print(f"Camera WindowCenter: {camera.GetWindowCenter()}")
                print(f"Camera ParallelScale: {camera.GetParallelScale()}")
                print(f"Renderer Viewport: {self.renderer.GetViewport()}")
                self._render_count += 1

            self.offscreen_renderer.update_display(self.vtk_display, restore_focus=False)
        except Exception as e:
            print(f"Render error: {e}")

    def eventFilter(self, obj, event):
        """Handle mouse events on vtk_display"""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent

        if obj == self.vtk_display:
            if event.type() == QEvent.MouseButtonPress:
                if isinstance(event, QMouseEvent):
                    # Shift+左ボタンまたはホイールボタンでパンモード
                    if (event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier) or \
                       event.button() == Qt.MiddleButton:
                        # 他のドラッグ状態をクリア
                        self.mouse_drag.end_left_drag()
                        self.mouse_drag.start_middle_drag(event.pos())
                        return True
                    elif event.button() == Qt.LeftButton:
                        # 他のドラッグ状態をクリア
                        self.mouse_drag.end_middle_drag()
                        self.mouse_drag.start_left_drag(event.pos())
                        return True

            elif event.type() == QEvent.MouseButtonRelease:
                if isinstance(event, QMouseEvent):
                    if event.button() == Qt.LeftButton:
                        self.mouse_drag.end_left_drag()
                        return True
                    elif event.button() == Qt.MiddleButton:
                        self.mouse_drag.end_middle_drag()
                        return True

            elif event.type() == QEvent.MouseMove:
                if isinstance(event, QMouseEvent):
                    if self.mouse_drag.left_button_pressed or self.mouse_drag.middle_button_pressed:
                        dx, dy = self.mouse_drag.update_pos(event.pos())

                        if self.mouse_drag.middle_button_pressed:
                            # Use CameraController for panning
                            self.camera_controller.pan(dx, dy)
                        else:
                            # Use CameraController for rotation
                            self.camera_controller.rotate_azimuth_elevation(dx, dy)

                        self.render_to_image()
                        return True

            elif event.type() == QEvent.Wheel:
                delta_y = event.angleDelta().y()
                # Use CameraController for zooming
                self.camera_controller.zoom(delta_y)
                self.render_to_image()
                return True

        return super().eventFilter(obj, event)

    def highlight_node(self, node):
        """ノードを選択したときにハイライト表示し、点滅させる"""
        # 既存のハイライトをクリア
        self.clear_highlight()

        if node not in self.stl_actors:
            return

        actor = self.stl_actors[node]

        # 元の色を保存
        self.original_color = actor.GetProperty().GetColor()
        self.highlighted_node = node

        # 設定されたハイライトカラーを取得（hex -> RGB）
        if hasattr(self, 'graph') and hasattr(self.graph, 'highlight_color'):
            color = QtGui.QColor(self.graph.highlight_color)
            highlight_rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
        else:
            # デフォルトのライトブルー
            highlight_rgb = (0.5, 0.8, 1.0)

        # ハイライトカラーに設定
        actor.GetProperty().SetColor(*highlight_rgb)
        self.render_to_image()

        # 点滅タイマーを開始（500msごと）
        self.highlight_state = True
        self.highlight_timer.start(500)

    def _toggle_highlight(self):
        """ハイライトの点滅を切り替え"""
        if not self.highlighted_node or self.highlighted_node not in self.stl_actors:
            self.highlight_timer.stop()
            return

        actor = self.stl_actors[self.highlighted_node]

        if self.highlight_state:
            # 元の色に戻す
            actor.GetProperty().SetColor(*self.original_color)
        else:
            # 設定されたハイライトカラーを取得（hex -> RGB）
            if hasattr(self, 'graph') and hasattr(self.graph, 'highlight_color'):
                color = QtGui.QColor(self.graph.highlight_color)
                highlight_rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            else:
                # デフォルトのライトブルー
                highlight_rgb = (0.5, 0.8, 1.0)
            actor.GetProperty().SetColor(*highlight_rgb)

        self.highlight_state = not self.highlight_state
        self.render_to_image()

    def clear_highlight(self):
        """ハイライトをクリア"""
        self.highlight_timer.stop()

        if self.highlighted_node and self.highlighted_node in self.stl_actors:
            actor = self.stl_actors[self.highlighted_node]
            if self.original_color:
                actor.GetProperty().SetColor(*self.original_color)
            self.render_to_image()

        self.highlighted_node = None
        self.original_color = None
        self.highlight_state = False

    def store_current_transform(self, node):
        """現在の変換を保存"""
        if node in self.transforms:
            current_transform = vtk.vtkTransform()
            current_transform.DeepCopy(self.transforms[node])
            self.original_transforms[node] = current_transform

            # 子ノードの変換も常に保存（Inherit to Subnodesの状態に関係なく）
            # これにより、show_angle()で親ノードの角度を変更した後、
            # Inherit to Subnodesが無効な場合に子ノードを元の変換に戻せる
            self._store_children_transforms(node)

    def _store_children_transforms(self, parent_node):
        """子孫ノードの変換を再帰的に保存"""
        for output_port in parent_node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                if child_node in self.transforms and child_node not in self.original_transforms:
                    current_transform = vtk.vtkTransform()
                    current_transform.DeepCopy(self.transforms[child_node])
                    self.original_transforms[child_node] = current_transform

                    # 再帰的に孫ノードも保存
                    self._store_children_transforms(child_node)

    def start_rotation_test(self, node):
        """回転テスト開始（PartsEditorと同じアプローチ）"""
        if node in self.stl_actors:
            # 現在の変換を保存（PartsEditorと同じ）
            self.store_current_transform(node)
            
            # 回転テストを有効化
            self.rotation_test_active = True
            self.rotating_node = node
            # 表示上の角度0度からスタート（Angle offsetを加味したzero度を基準）
            self.current_angle = 0.0
            self.rotation_direction = 1  # 増加方向からスタート
            self.rotation_paused = False  # 一時停止状態をリセット
            self.pause_counter = 0
            self.rotation_timer.start(16)  # 約60FPS

    def stop_rotation_test(self, node):
        """回転テスト終了 - ボタンがオフになった瞬間に角度を0に戻す（PartsEditorと同じアプローチ）"""
        # PartsEditorと同じ：タイマーを停止
        self.rotation_timer.stop()
        
        # 回転テストを無効化（stop後にtimeoutが来ても何もしないようにする）
        self.rotation_test_active = False
        
        # 回転中のノードを取得
        target_node = node if node else self.rotating_node
        
        # 元の色を復元
        if target_node and target_node in self.stl_actors:
            if hasattr(target_node, 'node_color') and target_node.node_color:
                # SetColor()はRGB（3要素）のみを受け取るため、最初の3要素を使用
                self.stl_actors[target_node].GetProperty().SetColor(*target_node.node_color[:3])
                # 透明度がある場合はSetOpacity()で設定
                if len(target_node.node_color) >= 4:
                    self.stl_actors[target_node].GetProperty().SetOpacity(target_node.node_color[3])
                else:
                    self.stl_actors[target_node].GetProperty().SetOpacity(1.0)
        
        # PartsEditorと同じ：保存された全ての変換を復元（元の位置に戻す）
        if self.original_transforms:
            nodes_to_restore = list(self.original_transforms.keys())
            for restore_node in nodes_to_restore:
                if restore_node in self.transforms and restore_node in self.original_transforms:
                    # original_transformsから元の変換を復元
                    self.transforms[restore_node].DeepCopy(self.original_transforms[restore_node])
                    if restore_node in self.stl_actors:
                        self.stl_actors[restore_node].SetUserTransform(self.transforms[restore_node])
                del self.original_transforms[restore_node]
        
        # PartsEditorと同じ：3Dビューを更新
        self.render_to_image()
        
        # 状態をリセット
        self.rotating_node = None
        self.rotation_paused = False
        self.pause_counter = 0
        self.current_angle = 0

    def show_angle(self, node, angle_rad):
        """指定された角度でSTLモデルを表示（静止）"""
        import math

        if node not in self.stl_actors:
            return

        # 回転タイマーを停止（もし動いていたら）
        self.rotation_timer.stop()

        # 現在の関節角度を保存（回転テストで使用）
        node.current_joint_angle = angle_rad

        # 角度をラジアンから度数に変換
        angle_deg = math.degrees(angle_rad)

        # transformを取得
        transform = self.transforms[node]

        # 親の変換と、ジョイントのorigin XYZ/RPY、親のpoint_angleを取得
        parent_transform = None
        joint_origin_xyz = None
        joint_origin_rpy = None
        parent_point_angle = None

        if hasattr(node, 'graph'):
            graph = node.graph
            # ノードの入力ポートから親を探す
            for input_port in node.input_ports():
                connected_ports = input_port.connected_ports()
                if connected_ports:
                    parent_node = connected_ports[0].node()
                    parent_port_name = connected_ports[0].name()

                    # 親のポートインデックスを取得（ポート名からインデックスを計算）
                    parent_output_ports = list(parent_node.output_ports())
                    for port_idx, port in enumerate(parent_output_ports):
                        if port.name() == parent_port_name:
                            # ポート名からポイントインデックスを計算（out_1 -> 0, out_2 -> 1, etc.）
                            point_index = port_idx  # デフォルト
                            if parent_port_name.startswith('out_'):
                                try:
                                    port_num = int(parent_port_name.split('_')[1])
                                    point_index = port_num - 1
                                except (ValueError, IndexError):
                                    pass
                            elif parent_port_name == 'out':
                                point_index = 0
                            # 親ノードのpointsからXYZ、RPY、point_angleを取得
                            if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                                point_data = parent_node.points[point_index]
                                joint_origin_xyz = point_data.get('xyz', [0, 0, 0])
                                joint_origin_rpy = point_data.get('rpy', [0, 0, 0])
                                parent_point_angle = point_data.get('angle', [0.0, 0.0, 0.0])

                            # 親の変換を取得
                            if parent_node in self.transforms:
                                parent_transform = self.transforms[parent_node]
                            break
                    break

        # 変換をリセット
        transform.Identity()

        # 親の変換を適用
        if parent_transform is not None:
            transform.Concatenate(parent_transform)

        # ジョイントの位置を適用
        if joint_origin_xyz:
            transform.Translate(joint_origin_xyz[0], joint_origin_xyz[1], joint_origin_xyz[2])

        # ジョイントのorigin RPYを適用（URDF仕様: Z-Y-X順）
        if joint_origin_rpy and len(joint_origin_rpy) == 3:
            roll_deg = math.degrees(joint_origin_rpy[0])
            pitch_deg = math.degrees(joint_origin_rpy[1])
            yaw_deg = math.degrees(joint_origin_rpy[2])
            transform.RotateZ(yaw_deg)    # Yaw
            transform.RotateY(pitch_deg)  # Pitch
            transform.RotateX(roll_deg)   # Roll

        # 親のpoint_angleを適用（radianからdegreeに変換してVTKへ渡す、Z-Y-X順）- Rotation Axisの回転基準となる座標系を定義
        if parent_point_angle and any(a != 0.0 for a in parent_point_angle):
            parent_point_angle_deg = [math.degrees(a) for a in parent_point_angle]
            transform.RotateZ(parent_point_angle_deg[2])  # Z軸回転
            transform.RotateY(parent_point_angle_deg[1])  # Y軸回転
            transform.RotateX(parent_point_angle_deg[0])  # X軸回転
            print(f"Applied parent point_angle: X={parent_point_angle_deg[0]}, Y={parent_point_angle_deg[1]}, Z={parent_point_angle_deg[2]} degrees")

        # Angle offsetを取得（body_angleから回転軸に応じた角度を取得）
        # body_angleはradianで保持されているので、degreeに変換
        angle_offset_deg = 0.0
        if hasattr(node, 'body_angle') and hasattr(node, 'rotation_axis'):
            body_angle = node.body_angle
            rotation_axis = node.rotation_axis
            if rotation_axis == 0:  # X軸
                angle_offset_deg = math.degrees(body_angle[0])
            elif rotation_axis == 1:  # Y軸
                angle_offset_deg = math.degrees(body_angle[1])
            elif rotation_axis == 2:  # Z軸
                angle_offset_deg = math.degrees(body_angle[2])

        # 回転軸に基づいて回転（親のpoint_angleで定義された座標系での回転）
        # angle_radは表示上の角度（Angle offsetを加味したzero度を基準）
        # 実際のjoint角度 = angle_rad + angle_offset_rad
        actual_angle_deg = angle_deg + angle_offset_deg
        if hasattr(node, 'rotation_axis'):
            if node.rotation_axis == 0:    # X軸
                transform.RotateX(actual_angle_deg)
            elif node.rotation_axis == 1:  # Y軸
                transform.RotateY(actual_angle_deg)
            elif node.rotation_axis == 2:  # Z軸
                transform.RotateZ(actual_angle_deg)

        self.stl_actors[node].SetUserTransform(transform)

        # Inherit to Subnodesが有効な場合、子孫ノードも一緒に回転
        if self.follow_children and hasattr(node, 'graph'):
            self._rotate_children(node, transform)
        else:
            # Inherit to Subnodesが無効な場合、子ノードを元の変換に戻す
            if hasattr(node, 'graph'):
                self._restore_children_transforms(node)

        self.render_to_image()

        print(f"Showing angle: {angle_rad} rad ({angle_deg} deg)")

    def show_inertial_origin(self, node, xyz):
        """Inertial Originの位置に赤い点とXYZ座標軸を表示（ローカル座標系）"""
        # 既存の表示があれば削除
        self.hide_inertial_origin(node)

        # ノードのtransformを取得
        if node not in self.transforms:
            print(f"Node {node.name()} has no transform")
            return

        node_transform = self.transforms[node]

        # アクターのリストを作成
        actors = []

        # 1. 赤い球を作成（ローカル座標）
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(xyz[0], xyz[1], xyz[2])
        sphere.SetRadius(0.005)  # 小さな球
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 赤色
        sphere_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(sphere_actor)
        actors.append(sphere_actor)

        # 2. XYZ軸を作成（ローカル座標）
        axis_length = 0.03  # 軸の長さ

        # X軸（赤）
        x_line = vtk.vtkLineSource()
        x_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        x_line.SetPoint2(xyz[0] + axis_length, xyz[1], xyz[2])

        x_mapper = vtk.vtkPolyDataMapper()
        x_mapper.SetInputConnection(x_line.GetOutputPort())

        x_actor = vtk.vtkActor()
        x_actor.SetMapper(x_mapper)
        x_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 赤
        x_actor.GetProperty().SetLineWidth(3)
        x_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(x_actor)
        actors.append(x_actor)

        # Y軸（緑）
        y_line = vtk.vtkLineSource()
        y_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        y_line.SetPoint2(xyz[0], xyz[1] + axis_length, xyz[2])

        y_mapper = vtk.vtkPolyDataMapper()
        y_mapper.SetInputConnection(y_line.GetOutputPort())

        y_actor = vtk.vtkActor()
        y_actor.SetMapper(y_mapper)
        y_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 緑
        y_actor.GetProperty().SetLineWidth(3)
        y_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(y_actor)
        actors.append(y_actor)

        # Z軸（青）
        z_line = vtk.vtkLineSource()
        z_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        z_line.SetPoint2(xyz[0], xyz[1], xyz[2] + axis_length)

        z_mapper = vtk.vtkPolyDataMapper()
        z_mapper.SetInputConnection(z_line.GetOutputPort())

        z_actor = vtk.vtkActor()
        z_actor.SetMapper(z_mapper)
        z_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 青
        z_actor.GetProperty().SetLineWidth(3)
        z_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(z_actor)
        actors.append(z_actor)

        # アクターを保存
        self.inertial_origin_actors[node] = actors

        # 再描画
        self.render_to_image()

    def hide_inertial_origin(self, node):
        """Inertial Originの表示を削除"""
        if node in self.inertial_origin_actors:
            for actor in self.inertial_origin_actors[node]:
                self.renderer.RemoveActor(actor)
            del self.inertial_origin_actors[node]
            self.render_to_image()

    def update_rotation(self):
        """回転更新"""
        if self.rotating_node and self.rotating_node in self.stl_actors:
            node = self.rotating_node
            transform = self.transforms[node]

            # 現在の位置を保持
            position = transform.GetPosition()

            # Fixedモードかどうかをチェック
            is_fixed = hasattr(node, 'rotation_axis') and node.rotation_axis == 3

            # Fixedモードの場合は点滅のみ
            if is_fixed:
                # 点滅効果（400msごとに切り替え）
                # フレームレート60FPSとして、24フレーム（約400ms）ごとに切り替え
                is_red = (self.current_angle // 24) % 2 == 0
                if is_red:
                    self.stl_actors[node].GetProperty().SetColor(1.0, 0.0, 0.0)  # 赤
                else:
                    self.stl_actors[node].GetProperty().SetColor(1.0, 1.0, 1.0)  # 白
            else:
                # Joint limitを取得（ラジアンから度数に変換）
                import math
                lower_deg = math.degrees(getattr(node, 'joint_lower', -3.14159))
                upper_deg = math.degrees(getattr(node, 'joint_upper', 3.14159))

                # Angle offsetを取得（表示上の角度0度を基準にするため）
                # body_angleはradianで保持されているので、degreeに変換
                angle_offset_deg = 0.0
                if hasattr(node, 'body_angle') and hasattr(node, 'rotation_axis'):
                    body_angle = node.body_angle
                    rotation_axis = node.rotation_axis
                    if rotation_axis == 0:  # X軸
                        angle_offset_deg = math.degrees(body_angle[0])
                    elif rotation_axis == 1:  # Y軸
                        angle_offset_deg = math.degrees(body_angle[1])
                    elif rotation_axis == 2:  # Z軸
                        angle_offset_deg = math.degrees(body_angle[2])
                
                # 表示上の角度範囲を計算（実際のjoint範囲からangle_offsetを引く）
                # 表示上の角度0度 = 実際のjoint角度がangle_offset度の状態
                display_lower_deg = lower_deg - angle_offset_deg
                display_upper_deg = upper_deg - angle_offset_deg

                # 一時停止中の処理
                if self.rotation_paused:
                    self.pause_counter += 1
                    # 0.5秒（60FPS × 0.5 = 30フレーム）経過したら再開
                    if self.pause_counter >= 30:
                        self.rotation_paused = False
                        self.pause_counter = 0
                    # 一時停止中は角度を更新せず、現在の角度を維持
                else:
                    # 角度更新（往復運動）- 表示上の角度0度を基準に
                    angle_step = 2.0  # 1フレームあたりの角度変化（度）
                    self.current_angle += angle_step * self.rotation_direction

                    # 範囲チェックと方向反転（表示上の角度範囲でチェック）
                    if self.current_angle >= display_upper_deg:
                        self.current_angle = display_upper_deg
                        self.rotation_direction = -1  # 減少方向へ
                        self.rotation_paused = True  # 一時停止開始
                        self.pause_counter = 0
                    elif self.current_angle <= display_lower_deg:
                        self.current_angle = display_lower_deg
                        self.rotation_direction = 1   # 増加方向へ
                        self.rotation_paused = True  # 一時停止開始
                        self.pause_counter = 0

                # 通常の回転処理
                transform.Identity()  # 変換をリセット

                # 親の変換と、ジョイントのorigin XYZ/RPY、親のpoint_angleを復元
                # ノードの親接続を探す
                parent_transform = None
                joint_origin_xyz = None
                joint_origin_rpy = None
                parent_point_angle = None

                if hasattr(node, 'graph'):
                    graph = node.graph
                    # ノードの入力ポートから親を探す
                    for input_port in node.input_ports():
                        connected_ports = input_port.connected_ports()
                        if connected_ports:
                            parent_node = connected_ports[0].node()
                            parent_port_name = connected_ports[0].name()

                            # 親のポートインデックスを取得（ポート名からインデックスを計算）
                            # ポート名からポイントインデックスを計算（out_1 -> 0, out_2 -> 1, etc.）
                            point_index = 0  # デフォルト
                            if parent_port_name.startswith('out_'):
                                try:
                                    port_num = int(parent_port_name.split('_')[1])
                                    point_index = port_num - 1
                                except (ValueError, IndexError):
                                    pass
                            elif parent_port_name == 'out':
                                point_index = 0

                            # 親ノードのpointsからXYZ、RPY、point_angleを取得
                            if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                                point_data = parent_node.points[point_index]
                                joint_origin_xyz = point_data.get('xyz', [0, 0, 0])
                                joint_origin_rpy = point_data.get('rpy', [0, 0, 0])
                                parent_point_angle = point_data.get('angle', [0.0, 0.0, 0.0])

                            # 親の変換を取得
                            if parent_node in self.transforms:
                                parent_transform = self.transforms[parent_node]
                            break

                # 親の変換を適用
                if parent_transform is not None:
                    transform.Concatenate(parent_transform)

                # ジョイントの位置を適用
                if joint_origin_xyz:
                    transform.Translate(joint_origin_xyz[0], joint_origin_xyz[1], joint_origin_xyz[2])

                # ジョイントのorigin RPYを適用（URDF仕様: Z-Y-X順）
                if joint_origin_rpy and len(joint_origin_rpy) == 3:
                    roll_deg = math.degrees(joint_origin_rpy[0])
                    pitch_deg = math.degrees(joint_origin_rpy[1])
                    yaw_deg = math.degrees(joint_origin_rpy[2])
                    transform.RotateZ(yaw_deg)    # Yaw
                    transform.RotateY(pitch_deg)  # Pitch
                    transform.RotateX(roll_deg)   # Roll

                # 親のpoint_angleを適用（radianからdegreeに変換してVTKへ渡す、Z-Y-X順）- Rotation Axisの回転基準となる座標系を定義
                if parent_point_angle and any(a != 0.0 for a in parent_point_angle):
                    parent_point_angle_deg = [math.degrees(a) for a in parent_point_angle]
                    transform.RotateZ(parent_point_angle_deg[2])  # Z軸回転
                    transform.RotateY(parent_point_angle_deg[1])  # Y軸回転
                    transform.RotateX(parent_point_angle_deg[0])  # X軸回転

                # Angle offsetを取得（body_angleはradianで保持されているので、degreeに変換）
                angle_offset_deg = 0.0
                if hasattr(node, 'body_angle') and hasattr(node, 'rotation_axis'):
                    body_angle = node.body_angle
                    rotation_axis = node.rotation_axis
                    if rotation_axis == 0:  # X軸
                        angle_offset_deg = math.degrees(body_angle[0])
                    elif rotation_axis == 1:  # Y軸
                        angle_offset_deg = math.degrees(body_angle[1])
                    elif rotation_axis == 2:  # Z軸
                        angle_offset_deg = math.degrees(body_angle[2])
                
                # 回転軸に基づいて回転テストの角度を適用（親のpoint_angleで定義された座標系での回転）
                # current_angleは表示上の角度（Angle offsetを加味したzero度を基準）
                # 実際のjoint角度 = current_angle + angle_offset
                actual_angle_deg = self.current_angle + angle_offset_deg
                if hasattr(node, 'rotation_axis'):
                    if node.rotation_axis == 0:    # X軸
                        transform.RotateX(actual_angle_deg)
                    elif node.rotation_axis == 1:  # Y軸
                        transform.RotateY(actual_angle_deg)
                    elif node.rotation_axis == 2:  # Z軸
                        transform.RotateZ(actual_angle_deg)

                self.stl_actors[node].SetUserTransform(transform)

                # Inherit to Subnodesが有効な場合、子孫ノードも一緒に回転
                if self.follow_children and hasattr(node, 'graph'):
                    self._rotate_children(node, transform)

            self.render_to_image()

    def _rotate_children(self, parent_node, parent_transform):
        """子孫ノードを親の回転に追従させて回転（複数分岐対応）"""
        import math
        import vtk

        # 親ノードの全出力ポートをチェック
        for port_idx, output_port in enumerate(parent_node.output_ports()):
            # 接続されている子ノードを取得
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                # 子ノードがSTLを持っている場合のみ処理
                if child_node not in self.stl_actors or child_node not in self.transforms:
                    continue

                # ポート名からポイントインデックスを計算（out_1 -> 0, out_2 -> 1, etc.）
                port_name = output_port.name() if hasattr(output_port, 'name') else ''
                point_index = port_idx  # デフォルト
                if port_name.startswith('out_'):
                    try:
                        port_num = int(port_name.split('_')[1])
                        point_index = port_num - 1
                    except (ValueError, IndexError):
                        pass
                elif port_name == 'out':
                    point_index = 0

                # 子ノードのジョイント情報とpoint_angleを取得
                child_xyz = [0, 0, 0]
                child_rpy = [0, 0, 0]
                parent_point_angle = [0.0, 0.0, 0.0]

                if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                    point_data = parent_node.points[point_index]
                    child_xyz = point_data.get('xyz', [0, 0, 0])
                    child_rpy = point_data.get('rpy', [0, 0, 0])
                    parent_point_angle = point_data.get('angle', [0.0, 0.0, 0.0])

                # 子ノードの変換を更新（新しい変換オブジェクトを作成して、親の変換をコピー）
                # これにより、各分岐が独立して処理される
                child_transform = vtk.vtkTransform()
                child_transform.Identity()
                
                # 親の変換を適用（回転テストの回転を含む）
                # DeepCopyを使用して、親の変換をコピー（参照ではなく）
                child_transform.DeepCopy(parent_transform)

                # ジョイントの位置を適用
                child_transform.Translate(child_xyz[0], child_xyz[1], child_xyz[2])

                # ジョイントのorigin RPYを適用
                if len(child_rpy) == 3:
                    roll_deg = math.degrees(child_rpy[0])
                    pitch_deg = math.degrees(child_rpy[1])
                    yaw_deg = math.degrees(child_rpy[2])
                    child_transform.RotateZ(yaw_deg)
                    child_transform.RotateY(pitch_deg)
                    child_transform.RotateX(roll_deg)

                # 親のpoint_angleを適用（radianからdegreeに変換してVTKへ渡す、Z-Y-X順）
                if any(a != 0.0 for a in parent_point_angle):
                    parent_point_angle_deg = [math.degrees(a) for a in parent_point_angle]
                    child_transform.RotateZ(parent_point_angle_deg[2])
                    child_transform.RotateY(parent_point_angle_deg[1])
                    child_transform.RotateX(parent_point_angle_deg[0])

                # 子ノードのbody_angleを適用（radianからdegreeに変換してVTKへ渡す）
                # NOTE: body_angleがMJCFのref angle（単一の回転軸に対する参照角度）の場合、
                # rotation_axisに応じた単一の軸のみを回転に適用する必要がある
                child_body_angle = getattr(child_node, 'body_angle', [0.0, 0.0, 0.0])
                if any(a != 0.0 for a in child_body_angle):
                    child_body_angle_deg = [math.degrees(a) for a in child_body_angle]
                    # rotation_axisが設定されている場合、その軸のみを回転
                    if hasattr(child_node, 'rotation_axis'):
                        if child_node.rotation_axis == 0 and child_body_angle_deg[0] != 0.0:  # X軸
                            child_transform.RotateX(child_body_angle_deg[0])
                        elif child_node.rotation_axis == 1 and child_body_angle_deg[1] != 0.0:  # Y軸
                            child_transform.RotateY(child_body_angle_deg[1])
                        elif child_node.rotation_axis == 2 and child_body_angle_deg[2] != 0.0:  # Z軸
                            child_transform.RotateZ(child_body_angle_deg[2])
                    else:
                        # rotation_axisが設定されていない場合、従来通りZ-Y-X順で適用
                        child_transform.RotateZ(child_body_angle_deg[2])
                        child_transform.RotateY(child_body_angle_deg[1])
                        child_transform.RotateX(child_body_angle_deg[0])

                # 子ノードの現在の関節角度を適用（回転テスト時に表示角度を維持）
                child_joint_angle = getattr(child_node, 'current_joint_angle', 0.0)
                if child_joint_angle != 0.0:
                    child_angle_deg = math.degrees(child_joint_angle)
                    if hasattr(child_node, 'rotation_axis'):
                        if child_node.rotation_axis == 0:    # X軸
                            child_transform.RotateX(child_angle_deg)
                        elif child_node.rotation_axis == 1:  # Y軸
                            child_transform.RotateY(child_angle_deg)
                        elif child_node.rotation_axis == 2:  # Z軸
                            child_transform.RotateZ(child_angle_deg)

                # 変換を適用（self.transformsも更新）
                self.transforms[child_node].DeepCopy(child_transform)
                self.stl_actors[child_node].SetUserTransform(child_transform)

                # 再帰的に孫ノードも回転（各分岐が独立して処理される）
                self._rotate_children(child_node, child_transform)

    def _restore_children_transforms(self, parent_node):
        """子孫ノードの変換を元に戻す（再帰的）"""
        for output_port in parent_node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                # 子ノードがSTLを持っている場合のみ処理
                if child_node not in self.stl_actors or child_node not in self.transforms:
                    continue

                # 保存された変換があれば復元
                if child_node in self.original_transforms:
                    self.transforms[child_node].DeepCopy(self.original_transforms[child_node])
                    self.stl_actors[child_node].SetUserTransform(self.transforms[child_node])

                # 再帰的に孫ノードも復元
                self._restore_children_transforms(child_node)

    def _get_scene_bounds_and_center(self):
        """シーンのバウンディングボックスと中心を計算"""
        if not self.renderer.GetActors().GetNumberOfItems():
            return None, None

        bounds = [float('inf'), float('-inf'),
                float('inf'), float('-inf'),
                float('inf'), float('-inf')]

        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextActor()
        while actor:
            actor_bounds = actor.GetBounds()
            bounds[0] = min(bounds[0], actor_bounds[0])
            bounds[1] = max(bounds[1], actor_bounds[1])
            bounds[2] = min(bounds[2], actor_bounds[2])
            bounds[3] = max(bounds[3], actor_bounds[3])
            bounds[4] = min(bounds[4], actor_bounds[4])
            bounds[5] = max(bounds[5], actor_bounds[5])
            actor = actors.GetNextActor()

        center = [(bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2]

        diagonal = ((bounds[1] - bounds[0]) ** 2 +
                (bounds[3] - bounds[2]) ** 2 +
                (bounds[5] - bounds[4]) ** 2) ** 0.5

        return center, diagonal

    def reset_camera_front(self):
        """Front view（正面図）- 原点を中心に表示"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # デフォルト値

        # 原点(0,0,0)を中心に設定
        distance = max(diagonal, 1.0)  # 最低距離を確保
        parallel_scale = max(diagonal * 0.7, 0.1)  # 最小スケールを確保

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[distance, 0, 0],  # X軸方向から見る
            view_up=[0, 0, 1],
            focal_point=[0, 0, 0],  # 原点を注視
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Front view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera_side(self):
        """Side view（側面図）- 原点を中心に表示"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # デフォルト値

        # 原点(0,0,0)を中心に設定
        distance = max(diagonal, 1.0)  # 最低距離を確保
        parallel_scale = max(diagonal * 0.7, 0.1)  # 最小スケールを確保

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[0, distance, 0],  # Y軸方向から見る
            view_up=[0, 0, 1],
            focal_point=[0, 0, 0],  # 原点を注視
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Side view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera_top(self):
        """Top view（上面図）- 原点を中心に表示"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # デフォルト値

        # 原点(0,0,0)を中心に設定
        distance = max(diagonal, 1.0)  # 最低距離を確保
        parallel_scale = max(diagonal * 0.7, 0.1)  # 最小スケールを確保

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[0, 0, distance],  # Z軸方向から見る
            view_up=[0, 1, 0],  # Top viewではY軸が上
            focal_point=[0, 0, 0],  # 原点を注視
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Top view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera(self):
        """カメラビューをリセット（Front viewと同じ）"""
        self.reset_camera_front()

    def reset_view_to_fit(self):
        """すべてのSTLモデルが見えるようにビューをリセットして調整"""
        self.reset_camera()
        self.render_to_image()

    def toggle_mesh(self, checked):
        """メッシュ表示の切り替え"""
        self.update_display_mode()

    def toggle_wireframe(self, checked):
        """Wireframe表示モードの切り替え"""
        self.update_display_mode()

    def update_display_mode(self):
        """メッシュとワイヤーフレームの表示モードを更新"""
        mesh_on = self.mesh_toggle.isChecked()
        wireframe_on = self.wireframe_toggle.isChecked()

        for node, actor in self.stl_actors.items():
            if mesh_on and wireframe_on:
                # 面 + エッジ表示
                actor.SetVisibility(True)
                actor.GetProperty().SetRepresentationToSurface()
                actor.GetProperty().EdgeVisibilityOn()
                actor.GetProperty().SetLineWidth(1)
            elif mesh_on and not wireframe_on:
                # 面のみ表示（通常のメッシュ表示）
                actor.SetVisibility(True)
                actor.GetProperty().SetRepresentationToSurface()
                actor.GetProperty().EdgeVisibilityOff()
            elif not mesh_on and wireframe_on:
                # ワイヤーフレームのみ（面なし、エッジのみ）
                actor.SetVisibility(True)
                actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetLineWidth(1)
            else:  # not mesh_on and not wireframe_on
                # 非表示
                actor.SetVisibility(False)

        # 表示モードを出力
        if mesh_on and wireframe_on:
            mode = "Surface + Edges"
        elif mesh_on and not wireframe_on:
            mode = "Surface only"
        elif not mesh_on and wireframe_on:
            mode = "Wireframe only"
        else:
            mode = "Hidden"

        print(f"Display mode updated: {mode} (Mesh={mesh_on}, Wireframe={wireframe_on})")

        # 再描画
        self.render_to_image()

    def toggle_collider_display(self, checked):
        """Collider表示の切り替え"""
        self.collider_display_enabled = checked

        if checked:
            # Colliderを表示
            self.show_all_colliders()
            print("Collider display ON")
        else:
            # Colliderを非表示
            self.hide_all_colliders()
            print("Collider display OFF")

        # 再描画
        self.render_to_image()

    def show_all_colliders(self):
        """全てのノードのColliderを表示"""
        print("=== show_all_colliders() called ===")
        # 既存のColliderアクターをクリア
        self.hide_all_colliders()

        # グラフから全ノードを取得
        if hasattr(self, 'graph') and self.graph:
            nodes = self.graph.all_nodes()
            print(f"Total nodes in graph: {len(nodes)}")
            for node in nodes:
                self.create_collider_actor_for_node(node)
            
            # コライダーアクターを作成した後、すべてのコライダーアクターのtransformを更新
            print("Updating collider transforms...")
            for node in nodes:
                if node in self.collider_actors and node in self.transforms:
                    self.update_collider_transform(node)
        
        print("=== show_all_colliders() finished ===")

    def hide_all_colliders(self):
        """全てのColliderアクターを削除"""
        for node, actors in list(self.collider_actors.items()):
            if isinstance(actors, list):
                for actor in actors:
                    self.renderer.RemoveActor(actor)
            else:
                self.renderer.RemoveActor(actors)
        self.collider_actors.clear()

    def create_collider_actor_for_node(self, node):
        """ノードのColliderアクターを作成"""
        node_name = getattr(node, 'name', 'Unknown')
        
        # 複数コライダー対応: node.colliders リストをチェック
        colliders = getattr(node, 'colliders', [])
        if colliders:
            # 新しい形式: colliders リストを使用
            actors = []
            for i, collider in enumerate(colliders):
                if not collider.get('enabled', False):
                    continue
                
                collider_type = collider.get('type')
                print(f"  Creating collider[{i}] for {node_name}: type={collider_type}")
                
                if collider_type == 'primitive':
                    collider_data = collider.get('data')
                    if collider_data:
                        # position と rotation は collider 直下にある
                        position = collider.get('position', [0, 0, 0])
                        rotation = collider.get('rotation', [0, 0, 0])
                        print(f"    → Creating primitive collider: {collider_data.get('type', 'unknown')}")
                        print(f"       position: {position}, rotation: {rotation}")
                        actor = self.create_primitive_collider_actor(collider_data, node, position=position, rotation=rotation)
                        if actor:
                            self.renderer.AddActor(actor)
                            actors.append(actor)
                            print(f"    ✓ Primitive collider actor created and added")
                        else:
                            print(f"    ✗ Failed to create primitive collider actor")
                    else:
                        print(f"    ✗ No collider_data found")
                
                elif collider_type == 'mesh':
                    collider_mesh = collider.get('mesh')
                    collider_mesh_scale = collider.get('mesh_scale', [1.0, 1.0, 1.0])
                    if collider_mesh:
                        print(f"    → Creating mesh collider: {os.path.basename(collider_mesh)}")
                        actor = self.create_mesh_collider_actor(node, collider_mesh, mesh_scale=collider_mesh_scale)
                        if actor:
                            self.renderer.AddActor(actor)
                            actors.append(actor)
                            print(f"    ✓ Mesh collider actor created and added")
                        else:
                            print(f"    ✗ Failed to create mesh collider actor")
                    else:
                        # メッシュが指定されていない場合は、visual meshを使用
                        if node.stl_file:
                            print(f"    → Creating mesh collider from visual mesh: {os.path.basename(node.stl_file)}")
                            actor = self.create_mesh_collider_actor(node, node.stl_file, mesh_scale=collider_mesh_scale)
                            if actor:
                                self.renderer.AddActor(actor)
                                actors.append(actor)
                                print(f"    ✓ Mesh collider actor created from visual mesh")
                            else:
                                print(f"    ✗ Failed to create mesh collider actor from visual mesh")
                        else:
                            print(f"    ✗ No collider_mesh or visual mesh found")
                else:
                    print(f"    ✗ Unknown collider_type: {collider_type}")
            
            if actors:
                # 複数のアクターがある場合はリストとして保存
                if len(actors) == 1:
                    self.collider_actors[node] = actors[0]
                else:
                    self.collider_actors[node] = actors
                print(f"  ✓ Created {len(actors)} collider actor(s) for {node_name}")
            else:
                print(f"  ✗ No enabled colliders found for {node_name}")

    def create_primitive_collider_actor(self, collider_data, node=None, position=None, rotation=None):
        """プリミティブコライダーのアクターを作成
        
        Args:
            collider_data: プリミティブ形状データ（type, geometryなど）
            node: 親ノード（オプション）
            position: コライダーの位置 [x, y, z]（オプション、デフォルト [0,0,0]）
            rotation: コライダーの回転 [rx, ry, rz] in degrees（オプション、デフォルト [0,0,0]）
        """
        geom_type = collider_data.get('type', 'box')
        geometry = collider_data.get('geometry', {})
        # 引数で渡された position/rotation を優先、なければ collider_data から、それもなければデフォルト
        if position is None:
            position = collider_data.get('position', [0, 0, 0])
        if rotation is None:
            rotation = collider_data.get('rotation', [0, 0, 0])  # degrees

        # ジオメトリソースを作成
        source = None

        if geom_type == 'box':
            size_x = float(geometry.get('size_x', 1.0))
            size_y = float(geometry.get('size_y', 1.0))
            size_z = float(geometry.get('size_z', 1.0))
            source = vtk.vtkCubeSource()
            source.SetXLength(size_x)
            source.SetYLength(size_y)
            source.SetZLength(size_z)

        elif geom_type == 'sphere':
            radius = float(geometry.get('radius', 0.5))
            source = vtk.vtkSphereSource()
            source.SetRadius(radius)
            source.SetThetaResolution(30)
            source.SetPhiResolution(30)

        elif geom_type == 'cylinder':
            radius = float(geometry.get('radius', 0.5))
            # SDF import historically stored cylinder/capsule length under 'height'.
            # Prefer 'length', but fall back to 'height' for backward compatibility.
            length = float(geometry.get('length', geometry.get('height', 1.0)))

            # アペンドフィルターで結合
            append = vtk.vtkAppendPolyData()

            # シリンダー本体（キャップなし）- VTKデフォルトはY軸、URDF/MuJoCoデフォルトはZ軸
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(radius)
            cylinder.SetHeight(length)
            cylinder.SetResolution(30)
            cylinder.SetCapping(0)  # キャップなし

            # Y軸→Z軸への補正回転を適用
            cyl_transform = vtk.vtkTransform()
            cyl_transform.RotateX(90)
            cyl_filter = vtk.vtkTransformPolyDataFilter()
            cyl_filter.SetInputConnection(cylinder.GetOutputPort())
            cyl_filter.SetTransform(cyl_transform)
            append.AddInputConnection(cyl_filter.GetOutputPort())

            # 上端のキャップ（Z軸正方向）
            top_cap = vtk.vtkDiskSource()
            top_cap.SetInnerRadius(0.0)
            top_cap.SetOuterRadius(radius)
            top_cap.SetRadialResolution(1)
            top_cap.SetCircumferentialResolution(30)

            top_cap_transform = vtk.vtkTransform()
            top_cap_transform.Translate(0, 0, length / 2)  # Z軸方向に配置
            top_cap_filter = vtk.vtkTransformPolyDataFilter()
            top_cap_filter.SetInputConnection(top_cap.GetOutputPort())
            top_cap_filter.SetTransform(top_cap_transform)
            append.AddInputConnection(top_cap_filter.GetOutputPort())

            # 下端のキャップ（Z軸負方向）
            bottom_cap = vtk.vtkDiskSource()
            bottom_cap.SetInnerRadius(0.0)
            bottom_cap.SetOuterRadius(radius)
            bottom_cap.SetRadialResolution(1)
            bottom_cap.SetCircumferentialResolution(30)

            bottom_cap_transform = vtk.vtkTransform()
            bottom_cap_transform.RotateY(180)  # ディスクを反転
            bottom_cap_transform.Translate(0, 0, -length / 2)  # Z軸負方向に配置
            bottom_cap_filter = vtk.vtkTransformPolyDataFilter()
            bottom_cap_filter.SetInputConnection(bottom_cap.GetOutputPort())
            bottom_cap_filter.SetTransform(bottom_cap_transform)
            append.AddInputConnection(bottom_cap_filter.GetOutputPort())

            append.Update()
            source = append

        elif geom_type == 'capsule':
            # カプセルはシリンダー（両端開放） + 2つの半球で構成
            radius = float(geometry.get('radius', 0.5))
            # SDF import historically stored capsule length under 'height'.
            length = float(geometry.get('length', geometry.get('height', 1.0)))
            
            # DEBUG: Print capsule dimensions
            if node:
                print(f"[CAPSULE_DEBUG] Node: {node.name()}, radius={radius}, length={length}, total_length={length + 2 * radius}")

            # アペンドフィルターで結合
            append = vtk.vtkAppendPolyData()

            # 中央のシリンダー（キャップなし）- VTKデフォルトはY軸、URDF/MuJoCoデフォルトはZ軸
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(radius)
            cylinder.SetHeight(length)
            cylinder.SetResolution(30)
            cylinder.SetCapping(0)  # 両端を開ける

            # Y軸→Z軸への補正回転を適用
            cyl_transform = vtk.vtkTransform()
            cyl_transform.RotateX(90)
            cyl_filter = vtk.vtkTransformPolyDataFilter()
            cyl_filter.SetInputConnection(cylinder.GetOutputPort())
            cyl_filter.SetTransform(cyl_transform)
            append.AddInputConnection(cyl_filter.GetOutputPort())

            # 上半球（Z軸正方向）
            top_sphere = vtk.vtkSphereSource()
            top_sphere.SetRadius(radius)
            top_sphere.SetThetaResolution(30)
            top_sphere.SetPhiResolution(30)
            top_sphere.SetStartTheta(0)
            top_sphere.SetEndTheta(360)
            top_sphere.SetStartPhi(0)
            top_sphere.SetEndPhi(90)

            top_transform = vtk.vtkTransform()
            top_transform.Translate(0, 0, length / 2)  # Z軸正方向に配置
            top_filter = vtk.vtkTransformPolyDataFilter()
            top_filter.SetInputConnection(top_sphere.GetOutputPort())
            top_filter.SetTransform(top_transform)
            append.AddInputConnection(top_filter.GetOutputPort())

            # 下半球（Z軸負方向）
            bottom_sphere = vtk.vtkSphereSource()
            bottom_sphere.SetRadius(radius)
            bottom_sphere.SetThetaResolution(30)
            bottom_sphere.SetPhiResolution(30)
            bottom_sphere.SetStartTheta(0)
            bottom_sphere.SetEndTheta(360)
            bottom_sphere.SetStartPhi(90)
            bottom_sphere.SetEndPhi(180)

            bottom_transform = vtk.vtkTransform()
            bottom_transform.Translate(0, 0, -length / 2)  # Z軸負方向に配置
            bottom_filter = vtk.vtkTransformPolyDataFilter()
            bottom_filter.SetInputConnection(bottom_sphere.GetOutputPort())
            bottom_filter.SetTransform(bottom_transform)
            append.AddInputConnection(bottom_filter.GetOutputPort())

            append.Update()
            source = append

        if not source:
            return None

        # マッパーとアクターを作成
        mapper = vtk.vtkPolyDataMapper()
        if hasattr(source, 'GetOutputPort'):
            mapper.SetInputConnection(source.GetOutputPort())
        else:
            mapper.SetInputData(source.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Settingsで設定したCollision Colorを適用
        if hasattr(self, 'graph') and hasattr(self.graph, 'collision_color'):
            collision_color = self.graph.collision_color
            actor.GetProperty().SetColor(*collision_color[:3])  # RGB
            if len(collision_color) >= 4:
                actor.GetProperty().SetOpacity(collision_color[3])  # Alpha
            else:
                actor.GetProperty().SetOpacity(1.0)
        else:
            # デフォルト値
            actor.GetProperty().SetColor(*DEFAULT_COLLISION_COLOR[:3])
            actor.GetProperty().SetOpacity(DEFAULT_COLLISION_COLOR[3])

        # コライダーのローカル変換を作成
        collider_local_transform = vtk.vtkTransform()
        collider_local_transform.PostMultiply()

        # Euler angles (degrees) -> Quaternion -> Rotation Matrix
        # URDF RPY: rotate by yaw around Z, then pitch around Y, then roll around X
        quat = euler_to_quaternion(rotation[0], rotation[1], rotation[2])
        w, x, y, z = quat

        # Convert quaternion to rotation matrix
        rot_matrix = vtk.vtkMatrix4x4()
        rot_matrix.SetElement(0, 0, 1 - 2*(y*y + z*z))
        rot_matrix.SetElement(0, 1, 2*(x*y - w*z))
        rot_matrix.SetElement(0, 2, 2*(x*z + w*y))
        rot_matrix.SetElement(1, 0, 2*(x*y + w*z))
        rot_matrix.SetElement(1, 1, 1 - 2*(x*x + z*z))
        rot_matrix.SetElement(1, 2, 2*(y*z - w*x))
        rot_matrix.SetElement(2, 0, 2*(x*z - w*y))
        rot_matrix.SetElement(2, 1, 2*(y*z + w*x))
        rot_matrix.SetElement(2, 2, 1 - 2*(x*x + y*y))

        # Apply rotation matrix
        collider_local_transform.Concatenate(rot_matrix)

        # Apply translation
        collider_local_transform.Translate(position[0], position[1], position[2])

        # ノードの変換と結合
        if node and node in self.transforms:
            combined_transform = vtk.vtkTransform()
            combined_transform.PostMultiply()
            # まずコライダーのローカル変換を適用
            combined_transform.Concatenate(collider_local_transform)
            # 次にノードの変換を適用
            combined_transform.Concatenate(self.transforms[node])
            actor.SetUserTransform(combined_transform)
        else:
            # ノード変換がない場合はコライダーのローカル変換のみ
            actor.SetUserTransform(collider_local_transform)

        return actor

    def create_mesh_collider_actor(self, node, collider_mesh, mesh_scale=None):
        """メッシュコライダーのアクターを作成

        Args:
            node: ノードオブジェクト
            collider_mesh: コライダーメッシュファイルのパス
            mesh_scale: メッシュスケール [x, y, z] (オプション、デフォルトは [1.0, 1.0, 1.0])
        """
        # コライダーメッシュのパスを解決
        if os.path.isabs(collider_mesh):
            # 絶対パスの場合はそのまま使用
            collider_path = collider_mesh
            print(f"      Using absolute path: {collider_path}")
        else:
            # 相対パスの場合、ビジュアルメッシュと同じディレクトリから読み込む
            visual_mesh = getattr(node, 'stl_file', None)
            if not visual_mesh:
                print(f"      ✗ No visual mesh found for relative path resolution")
                return None

            visual_dir = os.path.dirname(visual_mesh)
            collider_path = os.path.join(visual_dir, collider_mesh)
            print(f"      Resolved relative path: {collider_path}")

        if not os.path.exists(collider_path):
            print(f"      ✗ Collider mesh not found: {collider_path}")
            return None

        print(f"      ✓ Collider mesh file exists: {os.path.basename(collider_path)}")

        # メッシュファイルを読み込む
        polydata, _ = self.load_mesh_file(collider_path)
        if not polydata:
            print(f"      ✗ Failed to load mesh file")
            return None

        print(f"      ✓ Mesh loaded: {polydata.GetNumberOfPoints()} points, {polydata.GetNumberOfCells()} cells")

        # collision mesh scale を PolyData に適用（SDF <collision><mesh><scale> 対応）
        try:
            # mesh_scale パラメータが指定されている場合はそれを使用、なければデフォルト
            scale = mesh_scale if mesh_scale is not None else [1.0, 1.0, 1.0]
            
            if isinstance(scale, (list, tuple)) and len(scale) == 3:
                # default [1,1,1] 以外ならスケール適用
                if scale != [1.0, 1.0, 1.0]:
                    mesh_tf = vtk.vtkTransform()
                    mesh_tf.PostMultiply()
                    mesh_tf.Scale(float(scale[0]), float(scale[1]), float(scale[2]))
                    tf_filter = vtk.vtkTransformPolyDataFilter()
                    tf_filter.SetTransform(mesh_tf)
                    tf_filter.SetInputData(polydata)
                    tf_filter.Update()
                    polydata = tf_filter.GetOutput()
                    print(f"      ✓ Applied collider mesh scale: {scale}")
        except Exception as e:
            print(f"      Warning: Failed to apply collider mesh scale: {e}")

        # マッパーとアクターを作成
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Settingsで設定したCollision Colorを適用
        if hasattr(self, 'graph') and hasattr(self.graph, 'collision_color'):
            collision_color = self.graph.collision_color
            actor.GetProperty().SetColor(*collision_color[:3])  # RGB
            if len(collision_color) >= 4:
                actor.GetProperty().SetOpacity(collision_color[3])  # Alpha
            else:
                actor.GetProperty().SetOpacity(1.0)
            print(f"      ✓ Actor created with color: RGB={collision_color[:3]}, opacity: {collision_color[3] if len(collision_color) >= 4 else 1.0}")
        else:
            # デフォルト値
            actor.GetProperty().SetColor(*DEFAULT_COLLISION_COLOR[:3])
            actor.GetProperty().SetOpacity(DEFAULT_COLLISION_COLOR[3])
            print(f"      ✓ Actor created with default collision color")

        # ノードの位置と回転を適用
        self.apply_node_transform_to_collider(node, actor)
        print(f"      ✓ Transform applied to collider actor")

        return actor

    def apply_node_transform_to_collider(self, node, actor):
        """ノードの変換をコライダーアクターに適用（メッシュコライダー用）"""
        if node in self.transforms:
            # 既存のtransformをコピー
            node_transform = self.transforms[node]
            collider_transform = vtk.vtkTransform()
            collider_transform.DeepCopy(node_transform)
            actor.SetUserTransform(collider_transform)
    
    def update_collider_transform(self, node):
        """コライダーアクターのtransformを更新"""
        if node not in self.collider_actors or node not in self.transforms:
            return
        
        node_transform = self.transforms[node]
        actors = self.collider_actors[node]

        # 複数コライダー対応: node.colliders がある場合はそれを優先して更新する
        colliders = getattr(node, 'colliders', None)
        if isinstance(colliders, list) and len(colliders) > 0:
            # actors を常にリストとして扱う
            actor_list = actors if isinstance(actors, list) else [actors]

            actor_idx = 0
            for collider in colliders:
                if not collider.get('enabled', False):
                    continue

                if actor_idx >= len(actor_list):
                    break

                actor = actor_list[actor_idx]
                actor_idx += 1

                collider_type = collider.get('type')

                # primitive: ローカルpos/rot + node_transform を合成
                if collider_type == 'primitive':
                    # position と rotation は collider 直下にある（collider_data 内ではない）
                    position = collider.get('position', [0, 0, 0])
                    rotation = collider.get('rotation', [0, 0, 0])  # degrees

                    collider_local_transform = vtk.vtkTransform()
                    collider_local_transform.PostMultiply()

                    quat = euler_to_quaternion(rotation[0], rotation[1], rotation[2])
                    w, x, y, z = quat
                    rot_matrix = vtk.vtkMatrix4x4()
                    rot_matrix.SetElement(0, 0, 1 - 2*(y*y + z*z))
                    rot_matrix.SetElement(0, 1, 2*(x*y - w*z))
                    rot_matrix.SetElement(0, 2, 2*(x*z + w*y))
                    rot_matrix.SetElement(1, 0, 2*(x*y + w*z))
                    rot_matrix.SetElement(1, 1, 1 - 2*(x*x + z*z))
                    rot_matrix.SetElement(1, 2, 2*(y*z - w*x))
                    rot_matrix.SetElement(2, 0, 2*(x*z - w*y))
                    rot_matrix.SetElement(2, 1, 2*(y*z + w*x))
                    rot_matrix.SetElement(2, 2, 1 - 2*(x*x + y*y))
                    collider_local_transform.Concatenate(rot_matrix)
                    collider_local_transform.Translate(position[0], position[1], position[2])

                    combined_transform = vtk.vtkTransform()
                    combined_transform.PostMultiply()
                    combined_transform.Concatenate(collider_local_transform)
                    combined_transform.Concatenate(node_transform)
                    actor.SetUserTransform(combined_transform)
                else:
                    # mesh (and others): node_transform + ローカルpos/rot を合成
                    collider_position = collider.get('position', [0, 0, 0])
                    collider_rotation = collider.get('rotation', [0, 0, 0])  # degrees
                    
                    print(f"  [COLLIDER_TRANSFORM_DEBUG] Updating mesh collider transform:")
                    print(f"    collider_position: {collider_position}")
                    print(f"    collider_rotation (deg): {collider_rotation}")
                    
                    # コライダーのローカル変換を作成
                    collider_local_transform = vtk.vtkTransform()
                    collider_local_transform.PostMultiply()
                    
                    # 回転を適用（度数からクォータニオンに変換）
                    if collider_rotation != [0, 0, 0]:
                        quat = euler_to_quaternion(collider_rotation[0], collider_rotation[1], collider_rotation[2])
                        w, x, y, z = quat
                        rot_matrix = vtk.vtkMatrix4x4()
                        rot_matrix.SetElement(0, 0, 1 - 2*(y*y + z*z))
                        rot_matrix.SetElement(0, 1, 2*(x*y - w*z))
                        rot_matrix.SetElement(0, 2, 2*(x*z + w*y))
                        rot_matrix.SetElement(1, 0, 2*(x*y + w*z))
                        rot_matrix.SetElement(1, 1, 1 - 2*(x*x + z*z))
                        rot_matrix.SetElement(1, 2, 2*(y*z - w*x))
                        rot_matrix.SetElement(2, 0, 2*(x*z - w*y))
                        rot_matrix.SetElement(2, 1, 2*(y*z + w*x))
                        rot_matrix.SetElement(2, 2, 1 - 2*(x*x + y*y))
                        collider_local_transform.Concatenate(rot_matrix)
                        print(f"    Applied local rotation: {collider_rotation} deg")
                    
                    # 位置を適用
                    if collider_position != [0, 0, 0]:
                        collider_local_transform.Translate(collider_position[0], collider_position[1], collider_position[2])
                        print(f"    Applied local translation: {collider_position}")
                    
                    # ノードの変換と結合
                    combined_transform = vtk.vtkTransform()
                    combined_transform.PostMultiply()
                    combined_transform.Concatenate(collider_local_transform)
                    combined_transform.Concatenate(node_transform)
                    actor.SetUserTransform(combined_transform)
                    print(f"    Combined transform applied to mesh collider actor")

    def refresh_collider_display(self):
        """Collider表示が有効な場合、表示を更新"""
        if self.collider_display_enabled:
            self.show_all_colliders()
            self.render_to_image()

    def create_coordinate_axes(self):
        """座標軸の作成（線と独立したテキスト）"""
        base_assembly = vtk.vtkAssembly()
        length = 0.1
        text_offset = 0.02
        
        # ラインの作成部分は変更なし
        for i, (color, _) in enumerate([
            ((1,0,0), "X"),
            ((0,1,0), "Y"),
            ((0,0,1), "Z")
        ]):
            line = vtk.vtkLineSource()
            line.SetPoint1(0, 0, 0)
            end_point = [0, 0, 0]
            end_point[i] = length
            line.SetPoint2(*end_point)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(2)
            
            base_assembly.AddPart(actor)

        # テキスト部分をvtkBillboardTextActor3Dに変更
        for i, (color, label) in enumerate([
            ((1,0,0), "X"),
            ((0,1,0), "Y"),
            ((0,0,1), "Z")
        ]):
            text_position = [0, 0, 0]
            text_position[i] = length + text_offset
            
            text_actor = vtk.vtkBillboardTextActor3D()  # vtkTextActor3Dから変更
            text_actor.SetInput(label)
            text_actor.SetPosition(*text_position)
            text_actor.GetTextProperty().SetColor(*color)
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.GetTextProperty().SetJustificationToCentered()
            text_actor.GetTextProperty().SetVerticalJustificationToCentered()
            text_actor.SetScale(0.02)  # 単一の値でスケールを設定
            
            self.renderer.AddActor(text_actor)
            if not hasattr(self, 'text_actors'):
                self.text_actors = []
            self.text_actors.append(text_actor)
        
        return base_assembly

    def update_coordinate_axes(self, position):
        """座標軸とテキストの位置を更新"""
        # ラインの位置を更新
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.Translate(position[0], position[1], position[2])
        self.coordinate_axes_actor.SetUserTransform(transform)
        
        # テキストの位置を更新
        if hasattr(self, 'text_actors'):
            for i, text_actor in enumerate(self.text_actors):
                original_pos = list(text_actor.GetPosition())
                text_actor.SetPosition(
                    original_pos[0] + position[0],
                    original_pos[1] + position[1],
                    original_pos[2] + position[2]
                )
        
        self.render_to_image()

    def update_stl_transform(self, node, point_xyz, point_rpy=None, parent_transform=None):
        """STLの位置と回転を更新"""
        # base_linkでblank_linkがTrueの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node in self.stl_actors and node in self.transforms:
            print(f"Updating transform for node {node.name()} to position {point_xyz}, rotation {point_rpy}")
            transform = self.transforms[node]
            transform.Identity()

            # 親の変換を適用（累積変換用）
            if parent_transform is not None:
                transform.Concatenate(parent_transform)

            # ジョイントの位置を適用
            transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])

            # ジョイントの回転を適用（RPY: Roll-Pitch-Yaw）
            if point_rpy is not None and len(point_rpy) == 3:
                # RPYはラジアン単位なので度に変換
                import math
                roll_deg = math.degrees(point_rpy[0])
                pitch_deg = math.degrees(point_rpy[1])
                yaw_deg = math.degrees(point_rpy[2])

                # RPY順序で回転を適用（URDF仕様: Z-Y-X順）
                transform.RotateZ(yaw_deg)    # Yaw
                transform.RotateY(pitch_deg)  # Pitch
                transform.RotateX(roll_deg)   # Roll

            self.stl_actors[node].SetUserTransform(transform)

            # コライダーアクターのtransformも更新
            self.update_collider_transform(node)

            # base_linkに接続された最初のノードの場合、座標軸も更新
            if hasattr(node, 'graph'):
                base_node = node.graph.get_node_by_name('base_link')
                if base_node:
                    for port in base_node.output_ports():
                        for connected_port in port.connected_ports():
                            if connected_port.node() == node:
                                self.base_connected_node = node
                                self.update_coordinate_axes(point_xyz)
                                break

            self.render_to_image()
        else:
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No STL actor or transform found for node {node.name()}")

    def reset_stl_transform(self, node):
        """STLの位置をリセット"""
        # base_linkでblank_linkがTrueの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node in self.transforms:
            print(f"Resetting transform for node {node.name()}")
            transform = self.transforms[node]
            transform.Identity()
            # actor may not exist if it was removed earlier; handle gracefully
            actor = self.stl_actors.get(node)
            if actor is not None:
                try:
                    actor.SetUserTransform(transform)
                except Exception as e:
                    print(f"Warning: Failed to set transform on actor for {node.name()}: {e}")
            else:
                print(f"Warning: Transform exists for node {node.name()} but no STL actor found; removing stale transform.")
                try:
                    del self.transforms[node]
                except Exception:
                    pass
                # 座標軸のリセット（必要な場合）
                if node == self.base_connected_node:
                    self.update_coordinate_axes([0, 0, 0])
                    self.base_connected_node = None
                self.render_to_image()
                return

            # 座標軸のリセット（必要な場合）
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None
            
            self.render_to_image()
        else:
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No transform found for node {node.name()}")

    def load_mesh_file(self, file_path):
        """
        メッシュファイル（.stl, .obj, .dae）を読み込んでVTK PolyDataと色情報を返す

        Returns:
            tuple: (polydata, color) - colorはRGBA配列またはNone
        """
        try:
            # Use common utility function to load mesh
            poly_data, volume, extracted_color = load_mesh_to_polydata(file_path)

            print(f"Mesh file loaded: {file_path}")
            print(f"PolyData: {poly_data.GetNumberOfPoints()} points, {poly_data.GetNumberOfCells()} cells")

            if extracted_color:
                print(f"Color extracted from file: RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")

            # Convert RGBA to RGB for Assembler (Assembler uses RGB format)
            if extracted_color and len(extracted_color) >= 3:
                color_rgb = extracted_color[:3]
            else:
                color_rgb = None

            return poly_data, color_rgb

        except Exception as e:
            print(f"Error loading mesh file: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def load_stl_for_node(self, node, show_progress=True):
        """ノード用のメッシュファイル（.stl, .obj, .dae）を読み込む（色の適用を含む）"""
        # base_linkでblank_linkがTrueの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node.stl_file:
            # ファイルサイズを取得して処理の重み付けを計算
            try:
                file_size = os.path.getsize(node.stl_file)
                # ファイルサイズに基づいて読み込み工程の重みを調整
                # 小さいファイル(< 1MB): 読み込み50%, その他50%
                # 中程度(1-10MB): 読み込み70%, その他30%
                # 大きいファイル(> 10MB): 読み込み85%, その他15%
                file_size_mb = file_size / (1024 * 1024)
                if file_size_mb < 1:
                    load_weight = 50
                elif file_size_mb < 10:
                    load_weight = 70
                else:
                    load_weight = 85
            except:
                load_weight = 60  # デフォルト値
                file_size_mb = 0

            # プログレスバー表示開始 (100%から開始)
            if show_progress:
                self.show_progress(True)
                self.progress_bar.setValue(100)
                QtWidgets.QApplication.processEvents()

            # ファイル読み込み開始
            if show_progress:
                remaining = 100 - (load_weight * 0.3)  # 読み込み開始で30%消費
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            # メッシュファイルを読み込み（色情報も取得）
            polydata, extracted_color = self.load_mesh_file(node.stl_file)

            if show_progress:
                remaining = 100 - load_weight  # 読み込み完了
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            if polydata is None:
                print(f"ERROR: Failed to load mesh: {node.stl_file}")
                if show_progress:
                    self.show_progress(False)
                return

            # メッシュの元の色情報を内部的に保持
            if extracted_color is not None:
                # メッシュから抽出された色情報を保持
                node.mesh_original_color = extracted_color.copy()
                print(f"Stored mesh original color for node '{node.name()}': RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")
            else:
                # 色が抽出されなかった場合はNoneを設定
                if not hasattr(node, 'mesh_original_color'):
                    node.mesh_original_color = None
            
            # FooNodeの場合は読み込み直後にカラーを自動適用しない
            # メッシュの色情報は内部的に保持されているので、後で必要に応じて適用可能
            if isinstance(node, FooNode):
                # FooNodeの場合は、node_colorを更新しない（既存の色を維持）
                if not hasattr(node, 'node_color') or node.node_color is None:
                    node.node_color = DEFAULT_COLOR_WHITE.copy()
                print(f"FooNode '{node.name()}': Skipped automatic color application (mesh color stored in mesh_original_color)")
            else:
                # BaseLinkNodeなどの場合は、従来通りカラーを適用
                if extracted_color is not None:
                    node.node_color = extracted_color
                    print(f"Applied color from .dae file to node '{node.name()}': RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")
                elif not hasattr(node, 'node_color') or node.node_color is None:
                    node.node_color = DEFAULT_COLOR_WHITE.copy()

            # メッシュのスケールをポリデータに適用（URDF左右対称対応）
            # ジョイント位置には影響させず、メッシュの形状のみをスケール
            if hasattr(node, 'mesh_scale'):
                mesh_scale = node.mesh_scale
                if mesh_scale != [1.0, 1.0, 1.0]:
                    # vtkTransformを使ってスケール変換を作成
                    scale_transform = vtk.vtkTransform()
                    scale_transform.Scale(mesh_scale[0], mesh_scale[1], mesh_scale[2])

                    # vtkTransformPolyDataFilterでポリデータにスケールを適用
                    transform_filter = vtk.vtkTransformPolyDataFilter()
                    transform_filter.SetInputData(polydata)
                    transform_filter.SetTransform(scale_transform)
                    transform_filter.Update()

                    polydata = transform_filter.GetOutput()
                    print(f"Applied mesh scale {mesh_scale} to polydata for node '{node.name()}'")

            # Visual originをポリデータに適用（メッシュの位置・回転オフセット）
            if hasattr(node, 'visual_origin'):
                visual_origin = node.visual_origin
                xyz = visual_origin.get('xyz', [0.0, 0.0, 0.0])
                rpy = visual_origin.get('rpy', [0.0, 0.0, 0.0])

                # XYZまたはRPYが非ゼロの場合のみ変換を適用
                if xyz != [0.0, 0.0, 0.0] or rpy != [0.0, 0.0, 0.0]:
                    import math
                    # vtkTransformを使って回転と平行移動を作成
                    visual_transform = vtk.vtkTransform()

                    # まず平行移動を適用
                    visual_transform.Translate(xyz[0], xyz[1], xyz[2])

                    # 次にRPY回転を適用（Yaw, Pitch, Roll の逆順）
                    # URDFのRPY: 固定軸回転 R = Rz(yaw) * Ry(pitch) * Rx(roll)
                    # VTKはPostMultiplyなので、最終的な変換 M = T * Rz * Ry * Rx を得るために逆順で書く
                    # VTKの回転は度単位、URDFはラジアン単位なので変換が必要
                    if rpy[2] != 0.0:  # Yaw (Z軸周り) - 最後に適用される
                        visual_transform.RotateZ(math.degrees(rpy[2]))
                    if rpy[1] != 0.0:  # Pitch (Y軸周り)
                        visual_transform.RotateY(math.degrees(rpy[1]))
                    if rpy[0] != 0.0:  # Roll (X軸周り) - 最初に適用される
                        visual_transform.RotateX(math.degrees(rpy[0]))

                    # vtkTransformPolyDataFilterでポリデータに変換を適用
                    visual_transform_filter = vtk.vtkTransformPolyDataFilter()
                    visual_transform_filter.SetInputData(polydata)
                    visual_transform_filter.SetTransform(visual_transform)
                    visual_transform_filter.Update()

                    polydata = visual_transform_filter.GetOutput()
                    print(f"Applied visual origin xyz={xyz}, rpy={rpy} (radians) to polydata for node '{node.name()}'")

            # マッパーとアクター作成
            if show_progress:
                processing_weight = (100 - load_weight) * 0.6
                remaining = 100 - load_weight - processing_weight
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            # カラフルな表示のため、スカラー値（頂点カラーや面カラー）を使用するように設定
            # スカラー値がある場合はそれを使用、ない場合はノードの色を使用
            if polydata.GetPointData().GetScalars() is not None:
                # 頂点カラーがある場合
                mapper.SetScalarModeToUsePointData()
                mapper.SetColorModeToDefault()  # スカラー値を使用
                print(f"Using vertex colors for node '{node.name()}'")
            elif polydata.GetCellData().GetScalars() is not None:
                # 面カラーがある場合
                mapper.SetScalarModeToUseCellData()
                mapper.SetColorModeToDefault()  # スカラー値を使用
                print(f"Using face colors for node '{node.name()}'")
            else:
                # スカラー値がない場合は通常の色設定を使用
                mapper.SetScalarModeToDefault()
                mapper.SetColorModeToDefault()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # ジョイント位置・回転用のtransform（スケールは含まない）
            transform = vtk.vtkTransform()
            transform.Identity()
            actor.SetUserTransform(transform)

            # レンダラーに追加
            if show_progress:
                processing_weight = (100 - load_weight) * 0.6
                render_weight = (100 - load_weight - processing_weight) * 0.5
                remaining = 100 - load_weight - processing_weight - render_weight
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            if node in self.stl_actors:
                self.renderer.RemoveActor(self.stl_actors[node])

            self.stl_actors[node] = actor
            self.transforms[node] = transform
            self.renderer.AddActor(actor)

            # ノードの色情報を適用
            # FooNodeの場合は読み込み直後にカラーを自動適用しない
            if not isinstance(node, FooNode):
                self.apply_color_to_node(node)
            else:
                # FooNodeの場合は、メッシュの元の色情報を保持しているので、必要に応じて後で適用可能
                print(f"FooNode '{node.name()}': Skipped automatic color application in apply_color_to_node")

            # Hide Mesh状態を確認
            if hasattr(node, 'hide_mesh') and node.hide_mesh:
                actor.SetVisibility(False)
                print(f"Applied hide_mesh on load: {node.name()} - mesh hidden")
            else:
                # Hide Meshでない場合、現在のMesh/Wireframe表示モードを適用
                if hasattr(self, 'mesh_toggle') and hasattr(self, 'wireframe_toggle'):
                    # ボタンの状態に基づいて表示モードを設定
                    mesh_on = self.mesh_toggle.isChecked()
                    wireframe_on = self.wireframe_toggle.isChecked()

                    if mesh_on and wireframe_on:
                        actor.SetVisibility(True)
                        actor.GetProperty().SetRepresentationToSurface()
                        actor.GetProperty().EdgeVisibilityOn()
                        actor.GetProperty().SetLineWidth(1)
                    elif mesh_on and not wireframe_on:
                        actor.SetVisibility(True)
                        actor.GetProperty().SetRepresentationToSurface()
                        actor.GetProperty().EdgeVisibilityOff()
                    elif not mesh_on and wireframe_on:
                        actor.SetVisibility(True)
                        actor.GetProperty().SetRepresentationToWireframe()
                        actor.GetProperty().SetLineWidth(1)
                    else:
                        actor.SetVisibility(False)

                    print(f"Applied display mode on load: {node.name()} - Mesh={mesh_on}, Wireframe={wireframe_on}")

            # 最終レンダリング
            if show_progress:
                remaining = 5  # 最終工程
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            self.reset_camera()
            self.render_to_image()

            # 完了 (0%に到達)
            if show_progress:
                self.progress_bar.setValue(0)
                QtWidgets.QApplication.processEvents()

                # プログレスバー非表示
                QTimer.singleShot(200, lambda: self.show_progress(False))

            # ログ出力にファイルサイズ情報を追加
            print(f"Loaded: {node.stl_file} ({file_size_mb:.2f} MB)")

    def apply_color_to_node(self, node):
        """ノードのSTLモデルに色を適用（RGBA対応）"""
        if node in self.stl_actors:
            actor = self.stl_actors[node]
            mapper = actor.GetMapper()
            
            # スカラー値（頂点カラーや面カラー）がある場合は、それを使用（カラフルな表示）
            if mapper and mapper.GetInput():
                polydata = mapper.GetInput()
                has_vertex_colors = polydata.GetPointData().GetScalars() is not None
                has_face_colors = polydata.GetCellData().GetScalars() is not None
                
                if has_vertex_colors or has_face_colors:
                    # スカラー値を使用する場合は、ノードの色を適用しない（カラフルな表示を維持）
                    print(f"Node '{node.name()}' has vertex/face colors, skipping uniform color application")
                    # 透明度のみ設定（必要に応じて）
                    if hasattr(node, 'node_color') and node.node_color is not None and len(node.node_color) >= 4:
                        actor.GetProperty().SetOpacity(node.node_color[3])
                    else:
                        actor.GetProperty().SetOpacity(1.0)
                    self.render_to_image()
                    return
            
            # スカラー値がない場合は、通常通りノードの色を適用
            # デフォルトの色を設定（色情報がない場合）
            if not hasattr(node, 'node_color') or node.node_color is None:
                node.node_color = [1.0, 1.0, 1.0, 1.0]  # 白色（RGBA）をデフォルトに

            # 色の適用
            # RGB設定（最初の3要素のみ）
            actor.GetProperty().SetColor(*node.node_color[:3])

            # Alpha設定（4番目の要素があれば）
            if len(node.node_color) >= 4:
                actor.GetProperty().SetOpacity(node.node_color[3])
            else:
                actor.GetProperty().SetOpacity(1.0)

            self.render_to_image()

    def remove_stl_for_node(self, node):
        """ノードのSTLとColliderを削除"""
        # STLアクターを削除
        if node in self.stl_actors:
            self.renderer.RemoveActor(self.stl_actors[node])
            del self.stl_actors[node]
            if node in self.transforms:
                del self.transforms[node]

            # 座標軸のリセット（必要な場合）
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None

        # Colliderアクターも削除
        if node in self.collider_actors:
            actors = self.collider_actors[node]
            # actorsはリストまたは単一のアクター
            if isinstance(actors, list):
                for actor in actors:
                    self.renderer.RemoveActor(actor)
            else:
                self.renderer.RemoveActor(actors)
            del self.collider_actors[node]
            print(f"Removed Collider for node: {node.name()}")

        self.render_to_image()
        print(f"Removed STL for node: {node.name()}")

    def setup_camera(self):
        """カメラの初期設定 - 原点(0,0,0)を中心に表示"""
        camera = self.renderer.GetActiveCamera()

        # 平行投影を有効化
        camera.ParallelProjectionOn()

        # 重要: カメラパラメータは正しい順序で設定する必要がある
        # 1. 焦点（どこを見るか）を先に設定
        camera.SetFocalPoint(0, 0, 0)  # 原点を注視

        # 2. カメラ位置（どこから見るか）を設定
        camera.SetPosition(0.3, 0, 0)  # X軸方向から見る（距離0.3）

        # 3. 上方向ベクトルを設定
        camera.SetViewUp(0, 0, 1)  # Z軸が上

        # 4. WindowCenter設定 - 投影の中心をビューポートの中央に明示的に配置
        # (0, 0)がビューポートの中心、デフォルト値だが明示的に設定
        camera.SetWindowCenter(0.0, 0.0)

        # 5. ParallelScale設定（ビューポートの高さの半分、ワールド座標単位）
        # 座標軸が0.1単位なので、0.15に設定してパディングを確保
        camera.SetParallelScale(0.15)

        # 6. クリッピング範囲をリセット（全てが見えるように）
        self.renderer.ResetCameraClippingRange()

        print(f"Camera setup: Position={camera.GetPosition()}, FocalPoint={camera.GetFocalPoint()}, WindowCenter={camera.GetWindowCenter()}")

    def cleanup(self):
        """STLビューアのリソースをクリーンアップ"""
        # VTKオブジェクトの解放
        if hasattr(self, 'renderer'):
            if self.renderer:
                # アクターの削除
                for actor in self.renderer.GetActors():
                    self.renderer.RemoveActor(actor)

                # テキストアクターの削除
                for actor in self.text_actors:
                    self.renderer.RemoveActor(actor)
                self.text_actors.clear()

        # レンダーウィンドウのクリーンアップ
        if hasattr(self, 'render_window'):
            if self.render_window:
                self.render_window.Finalize()

        # 参照の解放
        self.stl_actors.clear()
        self.transforms.clear()

    def __del__(self):
        """デストラクタでクリーンアップを実行"""
        self.cleanup()

    def update_rotation_axis(self, node, axis_id):
        """ノードの回転軸を更新"""
        try:
            print(f"Updating rotation axis for node {node.name()} to axis {axis_id}")
            
            if node in self.stl_actors and node in self.transforms:
                transform = self.transforms[node]
                actor = self.stl_actors[node]
                
                # 現在の位置を保持
                current_position = list(actor.GetPosition())
                
                # 変換をリセット
                transform.Identity()
                
                # 位置を再設定
                transform.Translate(*current_position)
                
                # 新しい回転軸に基づいて回転を設定
                # 必要に応じてここに回転の処理を追加
                
                # 変換を適用
                actor.SetUserTransform(transform)
                
                # ビューを更新
                self.render_to_image()
                print(f"Successfully updated rotation axis for node {node.name()}")
            else:
                print(f"No STL actor or transform found for node {node.name()}")
                
        except Exception as e:
            print(f"Error updating rotation axis: {str(e)}")
            traceback.print_exc()

    def update_background(self, value):
        """背景色をスライダーの値に基づいて更新"""
        # -100から100の値を0から1の範囲に変換
        normalized_value = (value + 100) / 200.0
        self.renderer.SetBackground(normalized_value, normalized_value, normalized_value)
        self.render_to_image()

class ClosedLoopInspectorWindow(QtWidgets.QWidget):
    """閉リンクジョイントノード専用のインスペクタウィンドウ"""

    def __init__(self, parent=None, graph=None):
        super(ClosedLoopInspectorWindow, self).__init__(parent)
        self.setWindowTitle("Closed-Loop Joint Inspector")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.resize(450, 500)

        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.current_node = None
        self.graph = graph

        # UIの初期化
        self.setup_ui()

        # キーボードフォーカスを受け取れるように設定
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def setup_ui(self):
        """UIの初期化"""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 統一されたボタンスタイル
        self.button_style = UNIFIED_BUTTON_STYLE

        # スクロールエリアの設定
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # スクロールの中身となるウィジェット
        scroll_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(scroll_content)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Joint Name
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Joint Name:"))
        self.joint_name_input = QtWidgets.QLineEdit()
        self.joint_name_input.editingFinished.connect(self.update_joint_name)
        name_layout.addWidget(self.joint_name_input)
        content_layout.addLayout(name_layout)

        # Joint Type
        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Joint Type:"))
        self.joint_type_combo = QtWidgets.QComboBox()
        self.joint_type_combo.addItems(['ball', 'gearbox', 'screw'])
        # プルダウンの文字を黒に設定
        self.joint_type_combo.setStyleSheet("QComboBox { color: black; background-color: white; }")
        self.joint_type_combo.currentTextChanged.connect(self.update_joint_type)
        type_layout.addWidget(self.joint_type_combo)
        content_layout.addLayout(type_layout)

        # Parent Link (読み取り専用)
        parent_layout = QtWidgets.QHBoxLayout()
        parent_layout.addWidget(QtWidgets.QLabel("Parent Link:"))
        self.parent_link_label = QtWidgets.QLabel("")
        self.parent_link_label.setStyleSheet("QLabel { color: #aaaaaa; }")
        parent_layout.addWidget(self.parent_link_label)
        parent_layout.addStretch()
        content_layout.addLayout(parent_layout)

        # Child Link (読み取り専用)
        child_layout = QtWidgets.QHBoxLayout()
        child_layout.addWidget(QtWidgets.QLabel("Child Link:"))
        self.child_link_label = QtWidgets.QLabel("")
        self.child_link_label.setStyleSheet("QLabel { color: #aaaaaa; }")
        child_layout.addWidget(self.child_link_label)
        child_layout.addStretch()
        content_layout.addLayout(child_layout)

        # 区切り線
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.HLine)
        separator1.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator1)

        # Origin XYZ
        content_layout.addWidget(QtWidgets.QLabel("Origin Position (XYZ):"))
        xyz_layout = QtWidgets.QHBoxLayout()
        self.origin_x_input = QtWidgets.QLineEdit()
        self.origin_y_input = QtWidgets.QLineEdit()
        self.origin_z_input = QtWidgets.QLineEdit()
        for inp in [self.origin_x_input, self.origin_y_input, self.origin_z_input]:
            inp.setValidator(QDoubleValidator(-10000.0, 10000.0, 6))
            inp.editingFinished.connect(self.update_origin_xyz)
            inp.setMaximumWidth(80)
        xyz_layout.addWidget(QtWidgets.QLabel("X:"))
        xyz_layout.addWidget(self.origin_x_input)
        xyz_layout.addWidget(QtWidgets.QLabel("Y:"))
        xyz_layout.addWidget(self.origin_y_input)
        xyz_layout.addWidget(QtWidgets.QLabel("Z:"))
        xyz_layout.addWidget(self.origin_z_input)
        xyz_layout.addStretch()
        content_layout.addLayout(xyz_layout)

        # Origin RPY (度数法で表示)
        content_layout.addWidget(QtWidgets.QLabel("Origin Rotation (RPY in degrees):"))
        rpy_layout = QtWidgets.QHBoxLayout()
        self.origin_r_input = QtWidgets.QLineEdit()
        self.origin_p_input = QtWidgets.QLineEdit()
        self.origin_yaw_input = QtWidgets.QLineEdit()  # 変数名をorigin_yaw_inputに変更
        for inp in [self.origin_r_input, self.origin_p_input, self.origin_yaw_input]:
            inp.setValidator(QDoubleValidator(-360.0, 360.0, 3))
            inp.editingFinished.connect(self.update_origin_rpy)
            inp.setMaximumWidth(80)
        rpy_layout.addWidget(QtWidgets.QLabel("R:"))
        rpy_layout.addWidget(self.origin_r_input)
        rpy_layout.addWidget(QtWidgets.QLabel("P:"))
        rpy_layout.addWidget(self.origin_p_input)
        rpy_layout.addWidget(QtWidgets.QLabel("Y:"))
        rpy_layout.addWidget(self.origin_yaw_input)
        rpy_layout.addStretch()
        content_layout.addLayout(rpy_layout)

        # 区切り線
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator2)

        # Gearbox専用パラメータ（動的に表示/非表示）
        self.gearbox_widget = QtWidgets.QWidget()
        gearbox_layout = QtWidgets.QVBoxLayout(self.gearbox_widget)
        gearbox_layout.setContentsMargins(0, 0, 0, 0)

        # Gearbox Ratio
        ratio_layout = QtWidgets.QHBoxLayout()
        ratio_layout.addWidget(QtWidgets.QLabel("Gearbox Ratio:"))
        self.gearbox_ratio_input = QtWidgets.QLineEdit()
        self.gearbox_ratio_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.gearbox_ratio_input.editingFinished.connect(self.update_gearbox_ratio)
        self.gearbox_ratio_input.setMaximumWidth(100)
        ratio_layout.addWidget(self.gearbox_ratio_input)
        ratio_layout.addStretch()
        gearbox_layout.addLayout(ratio_layout)

        # Gearbox Reference Body
        ref_layout = QtWidgets.QHBoxLayout()
        ref_layout.addWidget(QtWidgets.QLabel("Reference Body:"))
        self.gearbox_ref_input = QtWidgets.QLineEdit()
        self.gearbox_ref_input.editingFinished.connect(self.update_gearbox_reference)
        ref_layout.addWidget(self.gearbox_ref_input)
        gearbox_layout.addLayout(ref_layout)

        content_layout.addWidget(self.gearbox_widget)
        self.gearbox_widget.setVisible(False)  # デフォルトは非表示

        content_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # 閉じるボタン
        close_button = QtWidgets.QPushButton("Close")
        close_button.setStyleSheet(self.button_style)
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button)

    def set_node(self, node):
        """ノードの情報を表示"""
        self.current_node = node
        if not node:
            return

        # Joint Name
        self.joint_name_input.setText(node.joint_name)

        # Joint Type
        index = self.joint_type_combo.findText(node.joint_type)
        if index >= 0:
            self.joint_type_combo.setCurrentIndex(index)

        # Parent/Child Links
        self.parent_link_label.setText(node.parent_link)
        self.child_link_label.setText(node.child_link)

        # Origin XYZ
        # デバッグ: origin_xyzの値を確認
        print(f"Setting origin_xyz for node {node.joint_name}: {node.origin_xyz}")
        self.origin_x_input.setText(f"{node.origin_xyz[0]}")
        self.origin_y_input.setText(f"{node.origin_xyz[1]}")
        self.origin_z_input.setText(f"{node.origin_xyz[2]}")

        # Origin RPY (radianから度数法に変換、小数点4桁に丸める)
        self.origin_r_input.setText(str(round(math.degrees(node.origin_rpy[0]), 4)))
        self.origin_p_input.setText(str(round(math.degrees(node.origin_rpy[1]), 4)))
        self.origin_yaw_input.setText(str(round(math.degrees(node.origin_rpy[2]), 4)))

        # Gearbox専用パラメータ
        if node.joint_type == 'gearbox':
            self.gearbox_widget.setVisible(True)
            self.gearbox_ratio_input.setText(str(node.gearbox_ratio))
            self.gearbox_ref_input.setText(node.gearbox_reference_body or "")
        else:
            self.gearbox_widget.setVisible(False)

    def update_joint_name(self):
        """Joint Name更新"""
        if self.current_node:
            self.current_node.joint_name = self.joint_name_input.text()
            print(f"Updated joint name to: {self.current_node.joint_name}")

    def update_joint_type(self, joint_type):
        """Joint Type更新"""
        if self.current_node:
            self.current_node.joint_type = joint_type
            # Gearbox専用パラメータの表示/非表示を切り替え
            self.gearbox_widget.setVisible(joint_type == 'gearbox')
            print(f"Updated joint type to: {joint_type}")

    def update_origin_xyz(self):
        """Origin XYZ更新"""
        if self.current_node:
            try:
                x = float(self.origin_x_input.text() or 0)
                y = float(self.origin_y_input.text() or 0)
                z = float(self.origin_z_input.text() or 0)
                self.current_node.origin_xyz = [x, y, z]
                print(f"Updated origin XYZ to: {self.current_node.origin_xyz}")
            except ValueError:
                print("Invalid XYZ values")

    def update_origin_rpy(self):
        """Origin RPY更新（度数法→radian）"""
        if self.current_node:
            try:
                r_deg = float(self.origin_r_input.text() or 0)
                p_deg = float(self.origin_p_input.text() or 0)
                y_deg = float(self.origin_yaw_input.text() or 0)
                self.current_node.origin_rpy = [math.radians(r_deg), math.radians(p_deg), math.radians(y_deg)]
                print(f"Updated origin RPY to: {self.current_node.origin_rpy} (radians)")
            except ValueError:
                print("Invalid RPY values")

    def update_gearbox_ratio(self):
        """Gearbox Ratio更新"""
        if self.current_node:
            try:
                ratio = float(self.gearbox_ratio_input.text() or 1.0)
                self.current_node.gearbox_ratio = ratio
                print(f"Updated gearbox ratio to: {ratio}")
            except ValueError:
                print("Invalid gearbox ratio")

    def update_gearbox_reference(self):
        """Gearbox Reference Body更新"""
        if self.current_node:
            self.current_node.gearbox_reference_body = self.gearbox_ref_input.text()
            print(f"Updated gearbox reference body to: {self.current_node.gearbox_reference_body}")

    def keyPressEvent(self, event):
        """キープレスイベントの処理"""
        # ESCキーが押されたかどうかを確認
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        # Cmd+W (macOS) または Ctrl+W (Windows/Linux) で閉じる
        elif event.key() == QtCore.Qt.Key.Key_W and (
            event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or
            event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier
        ):
            self.close()
        else:
            # 他のキーイベントは通常通り処理
            super(ClosedLoopInspectorWindow, self).keyPressEvent(event)

class CustomNodeGraph(NodeGraph):
    def __init__(self, stl_viewer):
        super(CustomNodeGraph, self).__init__()
        self.stl_viewer = stl_viewer
        self.robot_name = "robot_x"
        self.project_dir = None
        self.meshes_dir = None
        self.last_save_dir = None
        self.mjcf_eulerseq = 'xyz'  # MJCFのEuler回転順序（インポート時に更新）
        self.closed_loop_joints = []  # 閉リンクジョイント情報

        # グローバルデフォルト値（ジョイント制限パラメータ）
        self.default_joint_effort = DEFAULT_JOINT_EFFORT
        self.default_max_effort = DEFAULT_MAX_EFFORT
        self.default_joint_velocity = DEFAULT_JOINT_VELOCITY
        self.default_max_velocity = DEFAULT_MAX_VELOCITY
        self.default_margin = DEFAULT_MARGIN
        self.default_armature = DEFAULT_ARMATURE
        self.default_frictionloss = DEFAULT_FRICTIONLOSS
        self.default_stiffness_kp = DEFAULT_STIFFNESS_KP
        self.default_damping_kv = DEFAULT_DAMPING_KV
        self.default_angle_range = DEFAULT_ANGLE_RANGE

        # 後方互換性のための古いパラメータ（使用しない）
        # self.default_joint_friction = DEFAULT_JOINT_FRICTION
        # self.default_joint_actuation_lag = DEFAULT_JOINT_ACTUATION_LAG
        self.default_joint_damping = DEFAULT_JOINT_DAMPING
        self.default_joint_stiffness = DEFAULT_JOINT_STIFFNESS

        # MJCF base_linkのデフォルトz座標
        self.default_base_link_height = DEFAULT_BASE_LINK_HEIGHT

        # Node Gridの設定
        self.node_grid_enabled = DEFAULT_NODE_GRID_ENABLED
        self.node_grid_size = DEFAULT_NODE_GRID_SIZE

        # ハイライトカラー設定
        self.highlight_color = DEFAULT_HIGHLIGHT_COLOR

        # コリジョンカラー設定 (RGBA)
        self.collision_color = DEFAULT_COLLISION_COLOR.copy()

        # ポート接続/切断のシグナルを接続
        self.port_connected.connect(self.on_port_connected)
        self.port_disconnected.connect(self.on_port_disconnected)

        # ノードタイプの登録
        try:
            # BaseLinkNodeの登録
            self.register_node(BaseLinkNode)
            print(f"Registered node type: {BaseLinkNode.NODE_NAME}")

            # FooNodeの登録
            self.register_node(FooNode)
            print(f"Registered node type: {FooNode.NODE_NAME}")

            # ClosedLoopJointNodeの登録
            self.register_node(ClosedLoopJointNode)
            print(f"Registered node type: {ClosedLoopJointNode.NODE_NAME}")

        except Exception as e:
            print(f"Error registering node types: {str(e)}")
            import traceback
            traceback.print_exc()

        # 他の初期化コード...
        self._cleanup_handlers = []
        self._cached_positions = {}
        self._selection_cache = set()

        # 選択関連の変数を初期化
        self._selection_start = None
        self._is_selecting = False

        # パン操作関連の変数を初期化
        self._is_panning = False
        self._pan_start = None

        # ノードのクリップボード（コピー/ペースト用）
        self._node_clipboard = []

        # ビューの設定
        self._view = self.widget

        # 実際のQGraphicsViewにアクセス（NodeGraphWidgetはQTabWidget）
        self._viewer = self._view.currentWidget()  # NodeViewerを取得

        # NodeGraphQtの組み込み矩形選択の無効化を試みる
        # （PySide6互換性問題を回避するため）
        try:
            if self._viewer:
                self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
                print("NodeGraphQt rubber band selection disabled")
            else:
                print("Could not access internal viewer, will handle errors in event handlers")
        except Exception as e:
            print(f"Could not disable NodeGraphQt drag mode: {e}")

        # ラバーバンドの作成
        self._rubber_band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle,
            self._view
        )

        # オリジナルのイベントハンドラを保存
        if self._viewer:
            self._original_handlers = {
                'press': self._viewer.mousePressEvent,
                'move': self._viewer.mouseMoveEvent,
                'release': self._viewer.mouseReleaseEvent,
                'keyPress': self._viewer.keyPressEvent
            }

            # 新しいイベントハンドラを設定
            self._viewer.mousePressEvent = self.custom_mouse_press
            self._viewer.mouseMoveEvent = self.custom_mouse_move
            self._viewer.mouseReleaseEvent = self.custom_mouse_release
            self._viewer.keyPressEvent = self.custom_key_press
        else:
            self._original_handlers = {}

        # インスペクタウィンドウの初期化
        self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)

        # 選択状態監視用のタイマーを設定
        self.last_selected_node = None
        self.selection_monitor_timer = QTimer()
        self.selection_monitor_timer.timeout.connect(self._check_selection_change)
        self.selection_monitor_timer.start(100)  # 100msごとにチェック

        # グリッド表示の初期化（ビューアーが完全に初期化された後に実行）
        QtCore.QTimer.singleShot(100, self.update_grid_display)

    def _check_selection_change(self):
        """選択状態の変化を監視"""
        selected_nodes = self.selected_nodes()

        if selected_nodes:
            # 最初に選択されたノードを取得
            current_selected = selected_nodes[0]

            # 前回と異なるノードが選択された場合
            if current_selected != self.last_selected_node:
                self.last_selected_node = current_selected
                if self.stl_viewer:
                    self.stl_viewer.highlight_node(current_selected)
        else:
            # 何も選択されていない場合
            if self.last_selected_node is not None:
                self.last_selected_node = None
                if self.stl_viewer:
                    self.stl_viewer.clear_highlight()

    def custom_mouse_press(self, event):
        """カスタムマウスプレスイベントハンドラ"""
        try:






            # 中ボタンでパン操作を開始（カスタム実装）
            if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                print(">>> Starting pan operation (Middle Button Drag) - using custom panning")
                self._is_panning = True
                self._pan_start = event.position().toPoint()
                self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                # オリジナルハンドラーを呼び出してQt状態を初期化
                self._original_handlers['press'](event)
                return

            # 左ボタンの処理
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # Option (Alt) + 左ボタンでパン操作を開始（Qt ScrollHandDrag使用）
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    print(">>> Starting pan operation (Option+Drag) - using Qt ScrollHandDrag")
                    self._is_panning = True
                    self._pan_start = None  # ScrollHandDragを使用する場合はNone
                    # Qt組み込みのドラッグパンモードに切り替え
                    self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
                    self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    # 元のハンドラを呼び出してQt標準の処理を実行
                    self._original_handlers['press'](event)
                    return

                # Shift + 左ボタンでパン操作を開始（カスタムパン実装）
                # 注: NodeGraphQtはShiftを複数選択に使用しているため、競合する可能性あり
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    print(">>> Starting pan operation (Shift+Drag) - using custom panning")
                    self._is_panning = True
                    self._pan_start = event.position().toPoint()
                    self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    # オリジナルハンドラーを呼び出してQt状態を初期化
                    self._original_handlers['press'](event)
                    return

                # クリック位置にアイテムがあるかチェック
                pos = event.position().toPoint()
                print(f"Click position (view): {pos}")

                scene_pos = self._viewer.mapToScene(pos)
                print(f"Click position (scene): {scene_pos}")

                item_at_pos = self._viewer.scene().itemAt(scene_pos, self._viewer.transform())
                print(f"Item at position: {item_at_pos}")
                print(f"Item type: {type(item_at_pos)}")

                # 空白部分をクリックした場合のみラバーバンド選択を開始
                if item_at_pos is None or item_at_pos == self._viewer.scene():
                    print(">>> Starting rubber band selection")
                    self._selection_start = pos
                    self._is_selecting = True

                    # ラバーバンドの設定
                    if self._rubber_band:
                        rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                        self._rubber_band.setGeometry(rect)
                        self._rubber_band.show()
                        print(f"Rubber band shown at: {rect}")

                    # Ctrlキーが押されていない場合は選択をクリア
                    if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                        print("Clearing previous selection")
                        for node in self.selected_nodes():
                            node.set_selected(False)

                    # ラバーバンド選択中はオリジナルハンドラを呼ばない
                    return
                else:
                    print(">>> Item found at click position, not starting rubber band selection")

            # ラバーバンド選択以外の場合はオリジナルのイベントハンドラを呼び出し
            print("Calling original press handler")
            try:
                self._original_handlers['press'](event)
            except TypeError as te:
                # NodeGraphQtのPySide6互換性問題を無視
                print(f"Ignoring NodeGraphQt compatibility error: {te}")

        except Exception as e:
            print(f"Error in mouse press: {str(e)}")
            import traceback
            traceback.print_exc()

    def custom_mouse_move(self, event):
        """カスタムマウス移動イベントハンドラ"""
        try:
            # パン操作中の処理
            if self._is_panning:
                # カスタムパン実装（中ボタンまたはShift+左ボタン）
                if self._pan_start is not None:
                    current_pos = event.position().toPoint()

                    # ビュー座標をシーン座標にマッピング
                    previous_scene = self._viewer.mapToScene(self._pan_start)
                    current_scene = self._viewer.mapToScene(current_pos)
                    delta = previous_scene - current_scene

                    print(f"Custom pan: delta=({delta.x()}, {delta.y()})")

                    # NodeGraphQtの内部パンメソッドを使用
                    self._viewer._set_viewer_pan(delta.x(), delta.y())

                    self._pan_start = current_pos
                    return
                else:
                    # Qt標準のScrollHandDragを使用（Option+左ボタン）
                    try:
                        self._original_handlers['move'](event)
                    except TypeError:
                        pass
                    return

            if self._is_selecting and self._selection_start:
                current_pos = event.position().toPoint()
                rect = QtCore.QRect(self._selection_start,
                                    current_pos).normalized()
                if self._rubber_band:
                    self._rubber_band.setGeometry(rect)
                    print(f"Rubber band updated: {rect}")

                # ラバーバンド選択中はオリジナルハンドラを呼ばない
                return

            # ラバーバンド選択中でない場合はオリジナルのイベントハンドラを呼び出し
            try:
                self._original_handlers['move'](event)
            except TypeError as te:
                # NodeGraphQtのPySide6互換性問題を無視
                pass  # Move中は大量のエラーが出るので表示しない

        except Exception as e:
            print(f"Error in mouse move: {str(e)}")
            import traceback
            traceback.print_exc()

    def custom_mouse_release(self, event):
        """カスタムマウスリリースイベントハンドラ"""
        try:


            print(f"Is selecting: {self._is_selecting}")
            print(f"Is panning: {self._is_panning}")

            # パン操作の終了（中ボタンまたはShift/Option+左ボタン）
            if self._is_panning and (event.button() == QtCore.Qt.MouseButton.MiddleButton or
                                      event.button() == QtCore.Qt.MouseButton.LeftButton):
                print(">>> Ending pan operation")
                # 元のハンドラを呼び出してQt標準の処理を完了
                try:
                    self._original_handlers['release'](event)
                except TypeError:
                    pass
                # ドラッグモードを元に戻す
                self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
                self._viewer.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                self._is_panning = False
                self._pan_start = None
                return

            if event.button() == QtCore.Qt.MouseButton.LeftButton and self._is_selecting:
                if self._rubber_band and self._selection_start:
                    # 選択範囲の処理
                    rect = self._rubber_band.geometry()
                    scene_rect = self._viewer.mapToScene(rect).boundingRect()
                    print(f"Selection rect (view): {rect}")
                    print(f"Selection rect (scene): {scene_rect}")

                    # 範囲内のノードを選択
                    selected_count = 0
                    for node in self.all_nodes():
                        node_pos = node.pos()
                        if isinstance(node_pos, (list, tuple)):
                            node_point = QtCore.QPointF(
                                node_pos[0], node_pos[1])
                        else:
                            node_point = node_pos

                        print(f"Checking node '{node.name()}' at position: {node_point}")

                        if scene_rect.contains(node_point):
                            node.set_selected(True)
                            selected_count += 1
                            print(f"  -> Selected!")

                    print(f"Total nodes selected: {selected_count}")

                    # ラバーバンドを隠す
                    self._rubber_band.hide()

                # 選択状態をリセット
                self._selection_start = None
                self._is_selecting = False

                # ラバーバンド選択を実行した場合はオリジナルハンドラを呼ばない
                return

            # グリッドスナップ: ノードをドラッグした後、グリッドに吸着
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self.node_grid_enabled:
                for node in self.selected_nodes():
                    node_pos = node.pos()
                    if isinstance(node_pos, (list, tuple)):
                        current_x, current_y = node_pos[0], node_pos[1]
                    else:
                        current_x, current_y = node_pos.x(), node_pos.y()

                    snapped_x, snapped_y = self.snap_to_grid(current_x, current_y)

                    # 位置が変わった場合のみ更新
                    if abs(snapped_x - current_x) > 0.1 or abs(snapped_y - current_y) > 0.1:
                        node.set_pos(snapped_x, snapped_y)
                        print(f"Snapped node '{node.name()}' to grid: ({current_x:.1f}, {current_y:.1f}) -> ({snapped_x}, {snapped_y})")

            # ラバーバンド選択以外の場合はオリジナルのイベントハンドラを呼び出し
            print("Calling original release handler")
            try:
                self._original_handlers['release'](event)
            except TypeError as te:
                # NodeGraphQtのPySide6互換性問題を無視
                print(f"Ignoring NodeGraphQt compatibility error: {te}")

        except Exception as e:
            print(f"Error in mouse release: {str(e)}")
            import traceback
            traceback.print_exc()

    def copy_nodes(self):
        """選択されたノードをクリップボードにコピー"""
        selected = self.selected_nodes()
        if not selected:
            print("No nodes selected to copy")
            return

        # BaseLinkNodeは除外
        nodes_to_copy = [node for node in selected if not isinstance(node, BaseLinkNode)]
        if not nodes_to_copy:
            print("Cannot copy BaseLinkNode")
            return

        self._node_clipboard = []
        for node in nodes_to_copy:
            # ノードの位置を取得（リストに変換）
            pos = node.pos()
            if isinstance(pos, (list, tuple)):
                original_pos = [float(pos[0]), float(pos[1])]
            else:
                # QPointFなどの場合
                original_pos = [float(pos.x()), float(pos.y())]

            # ノードのデータをシリアライズ
            node_data = {
                'type': node.__class__.__name__,
                'name': node.name(),
                'original_pos': original_pos,  # オリジナルの位置を保存
                'properties': {}
            }

            # 各プロパティをコピー
            for prop_name in node.model.custom_properties.keys():
                try:
                    node_data['properties'][prop_name] = node.get_property(prop_name)
                except:
                    pass

            self._node_clipboard.append(node_data)

        print(f"Copied {len(self._node_clipboard)} node(s)")

    def paste_nodes(self):
        """クリップボードからノードをペースト"""
        if not self._node_clipboard:
            print("Clipboard is empty")
            return

        # 既存の選択を解除
        for node in self.selected_nodes():
            node.set_selected(False)

        pasted_nodes = []
        offset = 50  # ペースト時のオフセット

        for node_data in self._node_clipboard:
            try:
                # ノードタイプを取得
                node_type = node_data['type']

                # ノードタイプに応じてクラスを取得
                if node_type == 'FooNode':
                    node_class = 'insilico.nodes.FooNode'
                elif node_type == 'ClosedLoopJointNode':
                    node_class = 'insilico.nodes.ClosedLoopJointNode'
                else:
                    print(f"Unknown node type: {node_type}")
                    continue

                # 新しい名前を生成（重複を避ける）
                base_name = node_data['name']
                new_name = base_name
                counter = 1
                existing_names = [n.name() for n in self.all_nodes()]
                while new_name in existing_names:
                    new_name = f"{base_name}_{counter}"
                    counter += 1

                # 新しい位置を計算（常に元の位置から50pxオフセット）
                original_pos = node_data['original_pos']
                new_pos = [original_pos[0] + offset, original_pos[1] + offset]

                # ノードを作成
                new_node = self.create_node(
                    node_class,
                    name=new_name,
                    pos=new_pos
                )

                # プロパティをコピー
                for prop_name, prop_value in node_data['properties'].items():
                    try:
                        new_node.set_property(prop_name, prop_value)
                    except Exception as e:
                        print(f"Could not set property {prop_name}: {e}")

                pasted_nodes.append(new_node)

            except Exception as e:
                print(f"Error pasting node: {e}")
                import traceback
                traceback.print_exc()

        # ペーストしたノードを選択
        for node in pasted_nodes:
            node.set_selected(True)

        print(f"Pasted {len(pasted_nodes)} node(s)")

    def cut_nodes(self):
        """選択されたノードをカット（コピーして削除）"""
        selected = self.selected_nodes()
        if not selected:
            print("No nodes selected to cut")
            return

        # コピー
        self.copy_nodes()

        # 削除（BaseLinkNodeは除外）
        nodes_to_delete = [node for node in selected if not isinstance(node, BaseLinkNode)]
        for node in nodes_to_delete:
            self.delete_node(node)

        print(f"Cut {len(nodes_to_delete)} node(s)")

    def duplicate_nodes(self):
        """選択されたノードを複製"""
        # コピー
        self.copy_nodes()

        # すぐにペースト
        self.paste_nodes()

    def custom_key_press(self, event):
        """カスタムキープレスイベントハンドラ"""
        try:
            # Ctrl/Commandキーが押されているかチェック
            is_ctrl_cmd = (
                event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or
                event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier
            )

            # Ctrl/Command+C でコピー
            if event.key() == QtCore.Qt.Key.Key_C and is_ctrl_cmd:
                print("\n=== Copy Nodes (Ctrl/Cmd+C) ===")
                self.copy_nodes()
                event.accept()
                return

            # Ctrl/Command+V でペースト
            if event.key() == QtCore.Qt.Key.Key_V and is_ctrl_cmd:
                print("\n=== Paste Nodes (Ctrl/Cmd+V) ===")
                self.paste_nodes()
                event.accept()
                return

            # Ctrl/Command+X でカット
            if event.key() == QtCore.Qt.Key.Key_X and is_ctrl_cmd:
                print("\n=== Cut Nodes (Ctrl/Cmd+X) ===")
                self.cut_nodes()
                event.accept()
                return

            # Ctrl/Command+D で複製
            if event.key() == QtCore.Qt.Key.Key_D and is_ctrl_cmd:
                print("\n=== Duplicate Nodes (Ctrl/Cmd+D) ===")
                self.duplicate_nodes()
                event.accept()
                return

            # Ctrl/Command+A でBase以外の全ノードを選択
            if event.key() == QtCore.Qt.Key.Key_A and is_ctrl_cmd:
                print("\n=== Select All Nodes (Ctrl/Cmd+A) ===")
                # Base以外の全ノードを選択
                all_nodes = self.all_nodes()
                selected_count = 0

                for node in all_nodes:
                    # BaseLinkNodeは選択しない
                    if not isinstance(node, BaseLinkNode):
                        node.set_selected(True)
                        selected_count += 1
                    else:
                        # BaseLinkNodeは選択解除
                        node.set_selected(False)

                print(f"Selected {selected_count} nodes (excluding Base)")
                event.accept()
                return

            # DeleteキーまたはBackspaceキーが押された場合
            if event.key() in [QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace]:
                print("\n=== Delete/Backspace Key Pressed ===")
                # 選択されているノードを削除
                delete_selected_node(self)
                # イベントを処理済みとしてマーク
                event.accept()
                return

            # その他のキーイベントは元のハンドラに渡す
            try:
                self._original_handlers['keyPress'](event)
            except (TypeError, KeyError):
                # オリジナルハンドラがない場合は何もしない
                pass

        except Exception as e:
            print(f"Error in key press: {str(e)}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            print("Starting cleanup process...")
            
            # イベントハンドラの復元
            if hasattr(self, '_viewer') and self._viewer:
                if hasattr(self, '_original_handlers'):
                    self._viewer.mousePressEvent = self._original_handlers['press']
                    self._viewer.mouseMoveEvent = self._original_handlers['move']
                    self._viewer.mouseReleaseEvent = self._original_handlers['release']
                    if 'keyPress' in self._original_handlers:
                        self._viewer.keyPressEvent = self._original_handlers['keyPress']
                    print("Restored original event handlers")

            # ラバーバンドのクリーンアップ
            try:
                if hasattr(self, '_rubber_band') and self._rubber_band and not self._rubber_band.isHidden():
                    self._rubber_band.hide()
                    self._rubber_band.setParent(None)
                    self._rubber_band.deleteLater()
                    self._rubber_band = None
                    print("Cleaned up rubber band")
            except Exception as e:
                print(f"Warning: Rubber band cleanup - {str(e)}")
                
            # ノードのクリーンアップ
            for node in self.all_nodes():
                try:
                    # STLデータのクリーンアップ
                    if self.stl_viewer:
                        self.stl_viewer.remove_stl_for_node(node)
                    # ノードの削除
                    self.remove_node(node)
                except Exception as e:
                    print(f"Error cleaning up node: {str(e)}")

            # インスペクタウィンドウのクリーンアップ
            if hasattr(self, 'inspector_window') and self.inspector_window:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                    self.inspector_window = None
                    print("Cleaned up inspector window")
                except Exception as e:
                    print(f"Error cleaning up inspector window: {str(e)}")

            # キャッシュのクリア
            try:
                self._cached_positions.clear()
                self._selection_cache.clear()
                if hasattr(self, '_cleanup_handlers'):
                    self._cleanup_handlers.clear()
                print("Cleared caches")
            except Exception as e:
                print(f"Error clearing caches: {str(e)}")

            print("Cleanup process completed")

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """デストラクタでクリーンアップを実行"""
        self.cleanup()

    def remove_node(self, node):
        """ノード削除時のメモリリーク対策"""
        # キャッシュからノード関連データを削除
        if node in self._cached_positions:
            del self._cached_positions[node]
        self._selection_cache.discard(node)

        # ポート接続の解除
        for port in node.input_ports():
            for connected_port in port.connected_ports():
                self.disconnect_ports(port, connected_port)
        
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                self.disconnect_ports(port, connected_port)

        # STLデータのクリーンアップ
        if self.stl_viewer:
            self.stl_viewer.remove_stl_for_node(node)

        super(CustomNodeGraph, self).remove_node(node)

    def optimize_node_positions(self):
        """ノード位置の計算を最適化"""
        # 位置計算のキャッシュを活用
        for node in self.all_nodes():
            if node not in self._cached_positions:
                pos = self.calculate_node_position(node)
                self._cached_positions[node] = pos
            node.set_pos(*self._cached_positions[node])

    def setup_custom_view(self):
        """ビューのイベントハンドラをカスタマイズ"""
        # オリジナルのイベントハンドラを保存
        self._view.mousePressEvent_original = self._view.mousePressEvent
        self._view.mouseMoveEvent_original = self._view.mouseMoveEvent
        self._view.mouseReleaseEvent_original = self._view.mouseReleaseEvent

        # 新しいイベントハンドラを設定
        self._view.mousePressEvent = lambda event: self.custom_mouse_press(event)
        self._view.mouseMoveEvent = lambda event: self.custom_mouse_move(event)
        self._view.mouseReleaseEvent = lambda event: self.custom_mouse_release(event)

    def eventFilter(self, obj, event):
        """イベントフィルターでマウスイベントを処理"""
        if obj is self._view:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                return self._handle_mouse_press(event)
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                return self._handle_mouse_move(event)
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                return self._handle_mouse_release(event)
        
        return super(CustomNodeGraph, self).eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # ビューの座標系でマウス位置を取得
            view = self.scene().views()[0]
            self._selection_start = view.mapFromGlobal(event.globalPos())
            
            # Ctrlキーが押されていない場合は既存の選択をクリア
            if not event.modifiers() & QtCore.Qt.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)
            
            # ラバーバンドの開始位置を設定
            self._rubber_band.setGeometry(QtCore.QRect(self._selection_start, QtCore.QSize()))
            self._rubber_band.show()
        
        super(CustomNodeGraph, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_start is not None:
            # ビューの座標系で現在位置を取得
            view = self.scene().views()[0]
            current_pos = view.mapFromGlobal(event.globalPos())
            
            # ラバーバンドの領域を更新
            rect = QtCore.QRect(self._selection_start, current_pos).normalized()
            self._rubber_band.setGeometry(rect)
        
        super(CustomNodeGraph, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._selection_start is not None:
            # ビューの座標系でラバーバンドの領域を取得
            view = self.scene().views()[0]
            rubber_band_rect = self._rubber_band.geometry()
            scene_rect = view.mapToScene(rubber_band_rect).boundingRect()
            
            # 範囲内のノードを選択
            for node in self.all_nodes():
                node_center = QtCore.QPointF(node.pos()[0], node.pos()[1])
                if scene_rect.contains(node_center):
                    node.set_selected(True)
            
            # ラバーバンドをクリア
            self._rubber_band.hide()
            self._selection_start = None
        
        super(CustomNodeGraph, self).mouseReleaseEvent(event)

    def create_base_link(self):
        """初期のbase_linkノードを作成"""
        try:
            node_type = f"{BaseLinkNode.__identifier__}.{BaseLinkNode.NODE_NAME}"
            base_node = self.create_node(node_type)
            base_node.set_name('base_link')
            base_node.set_pos(0, 50)
            print("Base Link node created successfully")
            return base_node
        except Exception as e:
            print(f"Error creating base link node: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def register_nodes(self, node_classes):
        """複数のノードクラスを一度に登録"""
        for node_class in node_classes:
            self.register_node(node_class)
            print(f"Registered node type: {node_class.__identifier__}")

    def update_node_color_by_connection(self, node):
        """ノードの入力接続状態に応じて色を更新"""
        # BaseLinkNodeは例外として常に黒
        if isinstance(node, BaseLinkNode):
            node.set_color(45, 45, 45)  # 常に黒
            return

        # 入力ポートの接続をチェック
        has_input_connection = False
        for input_port in node.input_ports():
            if input_port.connected_ports():
                has_input_connection = True
                break

        if has_input_connection:
            # 接続あり：濃い黒
            node.set_color(45, 45, 45)  # 黒
        else:
            # 接続なし：明るめのグレー
            node.set_color(74, 84, 85)  # グレー

    def update_all_node_colors(self):
        """すべてのノードの色を接続状態に応じて更新"""
        for node in self.all_nodes():
            self.update_node_color_by_connection(node)

    def apply_cyan_color_to_connection(self, input_port, output_port):
        """特定の接続に水色を適用"""
        try:
            # ポートのviewオブジェクトから直接パイプにアクセス
            # 出力ポートのview経由でパイプを取得
            if hasattr(output_port, 'view') and output_port.view:
                port_view = output_port.view
                # ポートviewからパイプを取得（connected_pipes属性がある場合）
                if hasattr(port_view, 'connected_pipes'):
                    for pipe in port_view.connected_pipes:
                        # このパイプが目的の接続か確認
                        if hasattr(pipe, 'port_type') or True:  # すべてのパイプを処理
                            # 暗めの青緑に変更 (RGB 0, 180, 180)
                            if hasattr(pipe, 'set_pipe_styling'):
                                pipe.set_pipe_styling(color=(0, 180, 180), width=2, style=0)
                                print(f"  ✓ Applied dark cyan color to closed-loop connection")
                            elif hasattr(pipe, 'color'):
                                pipe.color = (0, 180, 180)
                                print(f"  ✓ Applied dark cyan color to closed-loop connection (via property)")
                    return

            print(f"  ⚠ Warning: Could not access pipe from port view")
        except Exception as e:
            print(f"  ⚠ Warning: Error applying cyan color: {str(e)}")
            traceback.print_exc()

    def apply_cyan_to_closed_loop_connections(self):
        """閉リンクノードへの全ての接続に水色を適用"""
        try:
            cyan_count = 0

            # 全てのノードをチェック
            for node in self.all_nodes():
                # 閉リンクノードかチェック
                if isinstance(node, ClosedLoopJointNode):
                    # 入力ポートと出力ポートの両方をチェック
                    for port in node.input_ports() + node.output_ports():
                        if hasattr(port, 'view') and port.view:
                            port_view = port.view
                            # ポートviewからパイプを取得
                            if hasattr(port_view, 'connected_pipes'):
                                for pipe in port_view.connected_pipes:
                                    # 暗めの青緑に変更 (RGB 0, 180, 180)
                                    if hasattr(pipe, 'set_pipe_styling'):
                                        pipe.set_pipe_styling(color=(0, 180, 180), width=2, style=0)
                                        cyan_count += 1
                                    elif hasattr(pipe, 'color'):
                                        pipe.color = (0, 180, 180)
                                        cyan_count += 1

            print(f"  ✓ Applied dark cyan color to {cyan_count} closed-loop connection(s)")

        except Exception as e:
            print(f"  ⚠ Warning: Error applying cyan color: {str(e)}")
            traceback.print_exc()

    def check_all_inertia(self):
        """全ノードのInertiaをチェックし、MuJoCoの条件を満たしていないノードを赤く染める"""
        import numpy as np
        
        # チェック結果を保存（後で元の色に戻すため）
        invalid_nodes = []
        valid_nodes = []
        
        print("\n=== Checking Inertia for All Nodes ===")
        
        for node in self.all_nodes():
            # BaseLinkNodeはスキップ（通常inertiaを持たない）
            if isinstance(node, BaseLinkNode):
                continue
            
            # ノードのinertiaを取得
            if not hasattr(node, 'inertia') or not node.inertia:
                continue
            
            inertia_dict = node.inertia
            mass = getattr(node, 'mass', 0.0)
            
            # inertia辞書から3x3テンソルを作成
            try:
                ixx = inertia_dict.get('ixx', 0.0)
                iyy = inertia_dict.get('iyy', 0.0)
                izz = inertia_dict.get('izz', 0.0)
                ixy = inertia_dict.get('ixy', 0.0)
                ixz = inertia_dict.get('ixz', 0.0)
                iyz = inertia_dict.get('iyz', 0.0)
                
                # 3x3 inertia tensorを作成
                inertia_tensor = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz]
                ])
                
                # MuJoCoの条件をチェック: A + B >= C (三角不等式)
                # Ixx + Iyy >= Izz, Iyy + Izz >= Ixx, Izz + Ixx >= Iyy
                tolerance = 1e-6
                is_valid = (
                    (ixx + iyy >= izz - tolerance) and
                    (iyy + izz >= ixx - tolerance) and
                    (izz + ixx >= iyy - tolerance)
                )
                
                if not is_valid:
                    invalid_nodes.append(node)
                    print(f"  ✗ {node.name()}: Inertia triangle inequality violated")
                    print(f"    Ixx={ixx:.6f}, Iyy={iyy:.6f}, Izz={izz:.6f}")
                    print(f"    Ixx+Iyy={ixx+iyy:.6f} >= Izz={izz:.6f}: {ixx+iyy >= izz - tolerance}")
                    print(f"    Iyy+Izz={iyy+izz:.6f} >= Ixx={ixx:.6f}: {iyy+izz >= ixx - tolerance}")
                    print(f"    Izz+Ixx={izz+ixx:.6f} >= Iyy={iyy:.6f}: {izz+ixx >= iyy - tolerance}")
                else:
                    valid_nodes.append(node)
                    
            except Exception as e:
                print(f"  ⚠ {node.name()}: Error checking inertia - {str(e)}")
                # エラーが発生した場合も無効として扱う
                invalid_nodes.append(node)
        
        # 無効なノードを赤く染める
        for node in invalid_nodes:
            node.set_color(255, 200, 200)  # 薄い赤
        
        # 有効なノードを元の色に戻す
        for node in valid_nodes:
            self.update_node_color_by_connection(node)
        
        print(f"\n=== Inertia Check Complete ===")
        print(f"  Valid nodes: {len(valid_nodes)}")
        print(f"  Invalid nodes: {len(invalid_nodes)}")
        
        # ダイアログボックスを表示
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Inertia Check Result")
        
        if invalid_nodes:
            # エラーがある場合
            invalid_node_names = [node.name() for node in invalid_nodes]
            invalid_count = len(invalid_nodes)
            
            # メッセージ本文を作成（最大10個まで表示）
            if invalid_count <= 10:
                node_list = "\n".join([f"  • {name}" for name in invalid_node_names])
            else:
                node_list = "\n".join([f"  • {name}" for name in invalid_node_names[:10]])
                node_list += f"\n  ... and {invalid_count - 10} more node(s)"
            
            message = f"⚠ {invalid_count} node(s) have invalid inertia:\n\n{node_list}\n\nThese nodes are highlighted in red."
            
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setText(message)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            
            print(f"\n⚠ {invalid_count} node(s) have invalid inertia (highlighted in red)")
        else:
            # すべてOKの場合
            message = f"✓ All nodes have valid inertia!\n\nChecked {len(valid_nodes)} node(s)."
            
            msg_box.setIcon(QtWidgets.QMessageBox.Information)
            msg_box.setText(message)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            
            print("\n✓ All nodes have valid inertia!")
        
        # ダイアログを表示
        msg_box.exec()

    def snap_to_grid(self, x, y):
        """座標をグリッドにスナップ

        Args:
            x (float): X座標
            y (float): Y座標

        Returns:
            tuple: グリッドスナップされた(x, y)座標
        """
        if not self.node_grid_enabled:
            return (x, y)

        grid_size = self.node_grid_size
        snapped_x = round(x / grid_size) * grid_size
        snapped_y = round(y / grid_size) * grid_size
        return (snapped_x, snapped_y)

    def update_grid_display(self):
        """グリッド表示を更新"""
        try:
            if hasattr(self, '_viewer') and self._viewer:
                # NodeGraphQtのviewerオブジェクトを通じてグリッドサイズを設定
                if hasattr(self._viewer, 'set_grid_size'):
                    self._viewer.set_grid_size(self.node_grid_size)
                    print(f"Grid size updated to: {self.node_grid_size}")
                elif hasattr(self._viewer, '_grid_size'):
                    # 内部プロパティを直接設定（NodeGraphQtのバージョンによる）
                    self._viewer._grid_size = self.node_grid_size
                    # ビューを再描画
                    if hasattr(self._viewer, 'update'):
                        self._viewer.update()
                    print(f"Grid size updated (direct) to: {self.node_grid_size}")
                else:
                    print("Warning: Could not update grid size (viewer does not support grid customization)")
        except Exception as e:
            print(f"Error updating grid display: {str(e)}")

    def on_port_connected(self, input_port, output_port):
        """ポートが接続された時の処理"""
        print(f"**Connecting port: {output_port.name()}")

        # 接続情報の出力
        parent_node = output_port.node()
        child_node = input_port.node()
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")

        try:
            # 閉リンクノードへの接続かチェック
            is_closed_loop_connection = isinstance(parent_node, ClosedLoopJointNode) or isinstance(child_node, ClosedLoopJointNode)

            if is_closed_loop_connection:
                # 接続されたパイプを水色に変更
                try:
                    # 接続後に少し待機してからパイプにアクセス
                    QtCore.QTimer.singleShot(100, lambda: self.apply_cyan_color_to_connection(input_port, output_port))
                    print(f"  ✓ Scheduled cyan color application for closed-loop connection")
                except Exception as pipe_error:
                    print(f"  ⚠ Warning: Could not schedule cyan color: {str(pipe_error)}")

            # 接続されたノード（子ノード）の色を更新
            self.update_node_color_by_connection(child_node)

            # 全ノードの位置を再計算
            print("Recalculating all node positions after connection...")
            self.recalculate_all_positions()

        except Exception as e:
            print(f"Error in port connection: {str(e)}")
            print(f"Detailed connection information:")
            print(f"  Output port: {output_port.name()} from {parent_node.name()}")
            print(f"  Input port: {input_port.name()} from {child_node.name()}")
            traceback.print_exc()

    def on_port_disconnected(self, input_port, output_port):
        """ポートが切断された時の処理"""
        child_node = input_port.node()  # 入力ポートを持つノードが子
        parent_node = output_port.node()  # 出力ポートを持つノードが親

        print(f"\nDisconnecting ports:")
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")

        try:
            # 子ノードの位置をリセット
            if hasattr(child_node, 'current_transform'):
                del child_node.current_transform

            # STLの位置をリセット
            self.stl_viewer.reset_stl_transform(child_node)
            print(f"Reset position for node: {child_node.name()}")

            # 切断されたノード（子ノード）の色を更新
            self.update_node_color_by_connection(child_node)

            # 全ノードの位置を再計算
            print("Recalculating all node positions after disconnection...")
            self.recalculate_all_positions()

        except Exception as e:
            print(f"Error in port disconnection: {str(e)}")
            traceback.print_exc()

    def update_robot_name(self, text):
        """ロボット名を更新するメソッド"""
        self.robot_name = text
        print(f"Robot name updated to: {text}")

        # 必要に応じて追加の処理
        # 例：ウィンドウタイトルの更新
        if hasattr(self, 'widget') and self.widget:
            if self.widget.window():
                title = f"URDF Kitchen - Assembler v0.1.0 -"
                self.widget.window().setWindowTitle(title)

    def get_robot_name(self):
        """
        現在のロボット名を取得するメソッド
        Returns:
            str: 現在のロボット名
        """
        return self.robot_name

    def set_robot_name(self, name):
        """
        ロボット名を設定するメソッド
        Args:
            name (str): 設定するロボット名
        """
        self.robot_name = name
        # 入力フィールドが存在する場合は更新
        if hasattr(self, 'name_input') and self.name_input:
            self.name_input.setText(name)
        print(f"Robot name set to: {name}")

    def clean_robot_name(self, name):
        """ロボット名から_descriptionを除去"""
        if name.endswith('_description'):
            return name[:-12]  # '_description'の長さ(12)を削除
        return name

    def update_robot_name_from_directory(self, dir_path):
        """ディレクトリ名からロボット名を更新"""
        dir_name = os.path.basename(dir_path)
        if dir_name.endswith('_description'):
            robot_name = dir_name[:-12]
            # UI更新
            if hasattr(self, 'name_input') and self.name_input:
                self.name_input.setText(robot_name)
            self.robot_name = robot_name
            return True
        return False

    def _quat_to_rpy(self, quat):
        """クォータニオン (w, x, y, z) をRPY (roll, pitch, yaw) に変換

        Note: This method now delegates to ConversionUtils.quat_to_rpy in utils.py

        Args:
            quat: [w, x, y, z] quaternion (MuJoCo convention)

        Returns:
            [roll, pitch, yaw] in radians
        """
        return ConversionUtils.quat_to_rpy(quat)

    def _euler_to_rpy(self, euler_angles, sequence='xyz'):
        """Euler角（度数法）をRPY（ラジアン）に変換

        Note: This method now delegates to ConversionUtils.euler_to_rpy in utils.py

        Args:
            euler_angles: Euler angles [a1, a2, a3] in degrees
            sequence: Euler rotation sequence ('xyz', 'zyx', etc.)

        Returns:
            [roll, pitch, yaw] in radians
        """
        return ConversionUtils.euler_to_rpy(euler_angles, sequence)

    # ============================================================================
    # XML Parsing Helper Methods
    # ============================================================================

    def _parse_float_list(self, attr_string, default=None):
        """XML属性文字列を数値リストに変換する統一ヘルパー

        Args:
            attr_string: スペース区切りの数値文字列
            default: 変換失敗時のデフォルト値

        Returns:
            floatのリスト、または変換失敗時はdefault
        """
        if not attr_string:
            return default
        try:
            return [float(v) for v in attr_string.split()]
        except (ValueError, AttributeError):
            return default

    def _parse_xyz(self, elem, attr='xyz', default=None):
        """XML要素からxyz属性を取得して数値リストに変換

        Args:
            elem: XML要素
            attr: 属性名（デフォルト: 'xyz'）
            default: 属性がない場合のデフォルト値

        Returns:
            [x, y, z]のリスト、またはdefault
        """
        if elem is None:
            return default if default is not None else [0.0, 0.0, 0.0]
        xyz_str = elem.get(attr, '0 0 0')
        result = self._parse_float_list(xyz_str)
        return result if result and len(result) == 3 else (default if default is not None else [0.0, 0.0, 0.0])

    def _parse_rpy(self, elem, attr='rpy', default=None):
        """XML要素からrpy属性を取得して数値リストに変換

        Args:
            elem: XML要素
            attr: 属性名（デフォルト: 'rpy'）
            default: 属性がない場合のデフォルト値

        Returns:
            [roll, pitch, yaw]のリスト、またはdefault
        """
        if elem is None:
            return default if default is not None else [0.0, 0.0, 0.0]
        rpy_str = elem.get(attr, '0 0 0')
        result = self._parse_float_list(rpy_str)
        return result if result and len(result) == 3 else (default if default is not None else [0.0, 0.0, 0.0])

    # ============================================================================
    # Import Methods
    # ============================================================================

    def _apply_colors_to_all_nodes(self):
        """すべてのノードのカラーを3Dビューに適用（Load Project、URDF、MJCF読み込み後用）
        
        STL読み込み完了後、すべてのノードのカラーを3Dビューに適用します。
        ノードを開いて閉じた時と同じ効果を持たせます。
        """
        if not self.stl_viewer:
            return
        
        print("\nApplying colors to 3D view after import...")
        all_nodes = self.all_nodes()
        applied_count = 0
        skipped_count = 0
        
        for node in all_nodes:
            try:
                node_name = node.name()
                has_stl_file = hasattr(node, 'stl_file') and node.stl_file
                in_actors = node in self.stl_viewer.stl_actors
                has_node_color = hasattr(node, 'node_color') and node.node_color
                
                # ノードにSTLファイルが読み込まれている場合のみカラーを適用
                if has_stl_file and in_actors:
                    # node.node_colorを確認
                    if has_node_color:
                        rgba_values = node.node_color
                        # RGBA値を0-1の範囲に正規化
                        rgba_values = [max(0.0, min(1.0, float(v))) for v in rgba_values[:4]]
                        
                        actor = self.stl_viewer.stl_actors[node]
                        
                        # マッパーを取得して、スカラー値（頂点カラーや面カラー）があるかチェック
                        mapper = actor.GetMapper()
                        has_scalars = False
                        if mapper and mapper.GetInput():
                            polydata = mapper.GetInput()
                            has_vertex_colors = polydata.GetPointData().GetScalars() is not None
                            has_face_colors = polydata.GetCellData().GetScalars() is not None
                            has_scalars = has_vertex_colors or has_face_colors
                        
                        if has_scalars:
                            # スカラー値がある場合は、透明度のみ設定
                            if len(rgba_values) >= 4:
                                actor.GetProperty().SetOpacity(rgba_values[3])
                            else:
                                actor.GetProperty().SetOpacity(1.0)
                            print(f"Node '{node_name}' has vertex/face colors, only opacity applied: {rgba_values[3] if len(rgba_values) >= 4 else 1.0}")
                        else:
                            # スカラー値がない場合は、通常通りノードの色を適用
                            # RGB設定（最初の3要素のみ）
                            actor.GetProperty().SetColor(*rgba_values[:3])
                            # Alpha設定（4番目の要素があれば）
                            if len(rgba_values) >= 4:
                                actor.GetProperty().SetOpacity(rgba_values[3])
                            else:
                                actor.GetProperty().SetOpacity(1.0)
                            print(f"Applied color to node '{node_name}': RGBA{rgba_values[:4]}")
                            applied_count += 1
                    else:
                        # デフォルトの白色を適用
                        actor = self.stl_viewer.stl_actors[node]
                        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                        actor.GetProperty().SetOpacity(1.0)
                        print(f"Applied default white color to node '{node_name}'")
                        applied_count += 1
                else:
                    skipped_count += 1
                    if not has_stl_file:
                        print(f"Skipped node '{node_name}': no STL file")
                    elif not in_actors:
                        print(f"Skipped node '{node_name}': not in stl_actors")
            except Exception as e:
                print(f"Error applying color to node '{node.name()}': {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"Color application completed: {applied_count} applied, {skipped_count} skipped")
        
        # 3Dビューを更新
        if applied_count > 0:
            self.stl_viewer.render_to_image()


    def export_urdf(self):
        """URDFファイルをエクスポート"""
        try:
            # 閉リンクノードから最新の情報を収集
            self.collect_closed_loop_joints_from_nodes()

            # ロボット名を取得（クリーン化）
            robot_base_name = self.robot_name or "robot"
            clean_name = self.clean_robot_name(robot_base_name)

            # 親ディレクトリを選択
            parent_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select parent directory for URDF export",
                os.getcwd()
            )

            if not parent_dir:
                print("URDF export cancelled")
                return False

            # 閉リンクジョイントがある場合は警告を表示
            if self.closed_loop_joints:
                warning_msg = f"This robot contains {len(self.closed_loop_joints)} closed-loop joint(s):\n\n"
                for joint_data in self.closed_loop_joints:
                    joint_name = joint_data['name']
                    joint_type = joint_data.get('original_type', 'unknown')
                    warning_msg += f"  - {joint_name} (type: {joint_type})\n"

                warning_msg += (
                    "\nURDF format only supports tree structures and cannot represent closed-loop constraints.\n"
                    "These joints will be EXCLUDED from the exported URDF file.\n\n"
                    "To preserve closed-loop constraints, please use MJCF export instead.\n\n"
                    "Do you want to continue with URDF export?"
                )

                response = QtWidgets.QMessageBox.warning(
                    self.widget,
                    "Closed-Loop Joints Detected",
                    warning_msg,
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )

                if response == QtWidgets.QMessageBox.No:
                    print("URDF export cancelled due to closed-loop joints")
                    return False

                print(f"User chose to continue URDF export, {len(self.closed_loop_joints)} closed-loop joint(s) will be excluded")

            # メッシュファイル形式を選択
            mesh_format_dialog = QtWidgets.QDialog(self.widget)
            mesh_format_dialog.setWindowTitle("Select Mesh Format")
            mesh_format_dialog.setModal(True)
            layout = QtWidgets.QVBoxLayout()
            
            label = QtWidgets.QLabel("Select mesh file format for export:")
            layout.addWidget(label)
            
            format_group = QtWidgets.QButtonGroup()
            stl_radio = QtWidgets.QRadioButton(".stl (STL)")
            stl_radio.setChecked(True)  # デフォルトは.stl
            dae_radio = QtWidgets.QRadioButton(".dae (COLLADA)")
            format_group.addButton(stl_radio, 0)
            format_group.addButton(dae_radio, 1)
            
            layout.addWidget(stl_radio)
            layout.addWidget(dae_radio)
            
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(mesh_format_dialog.accept)
            button_box.rejected.connect(mesh_format_dialog.reject)
            layout.addWidget(button_box)
            
            mesh_format_dialog.setLayout(layout)
            
            if mesh_format_dialog.exec() != QtWidgets.QDialog.Accepted:
                print("URDF export cancelled (format selection)")
                return False
            
            # 選択された形式を取得
            selected_format = ".stl" if stl_radio.isChecked() else ".dae"
            print(f"Selected mesh format: {selected_format}")

            # [ロボット名]_descriptionディレクトリを作成
            description_dir = os.path.join(parent_dir, f"{clean_name}_description")

            try:
                os.makedirs(description_dir, exist_ok=True)
                print(f"Created description directory: {description_dir}")
            except Exception as e:
                print(f"Error creating description directory: {str(e)}")
                QtWidgets.QMessageBox.critical(
                    self.widget,
                    "Export Error",
                    f"Failed to create description directory:\n{str(e)}"
                )
                return False

            # urdfディレクトリのパスを作成（meshesと同じ階層）
            urdf_dir = os.path.join(description_dir, 'urdf')

            # urdfディレクトリが存在しない場合は作成
            if not os.path.exists(urdf_dir):
                try:
                    os.makedirs(urdf_dir)
                    print(f"Created URDF directory: {urdf_dir}")
                except Exception as e:
                    print(f"Error creating URDF directory: {str(e)}")
                    return False

            # meshesディレクトリのパスを作成
            meshes_dir = os.path.join(description_dir, 'meshes')

            # meshesディレクトリが存在しない場合は作成
            if not os.path.exists(meshes_dir):
                try:
                    os.makedirs(meshes_dir)
                    print(f"Created meshes directory: {meshes_dir}")
                except Exception as e:
                    print(f"Error creating meshes directory: {str(e)}")
                    return False

            # メッシュファイルを収集してコピー
            stl_files_copied = []
            stl_files_failed = []

            for node in self.all_nodes():
                # Hide Meshにチェックが入っているノードはスキップ
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                if hasattr(node, 'stl_file') and node.stl_file:
                    source_path = node.stl_file
                    if os.path.exists(source_path):
                        stl_filename = os.path.basename(source_path)
                        dest_path = os.path.join(meshes_dir, stl_filename)
                        file_ext = os.path.splitext(stl_filename)[1].lower()

                        try:
                            # メッシュファイルを読み込んで、選択された形式で保存
                            poly_data, volume, extracted_color = load_mesh_to_polydata(source_path)
                            
                            # ファイル名の拡張子を選択された形式に変更
                            base_name = os.path.splitext(stl_filename)[0]
                            new_filename = f"{base_name}{selected_format}"
                            dest_path = os.path.join(meshes_dir, new_filename)
                            
                            # ノードの色情報を取得
                            mesh_color = None
                            color_manually_changed = False
                            if hasattr(node, 'node_color'):
                                mesh_color = node.node_color
                                color_manually_changed = True
                            
                            # 選択された形式で保存
                            save_polydata_to_mesh(
                                dest_path, 
                                poly_data, 
                                mesh_color=mesh_color,
                                color_manually_changed=color_manually_changed
                            )
                            
                            stl_files_copied.append(new_filename)
                            print(f"Converted and saved mesh: {stl_filename} -> {new_filename}")
                        except Exception as e:
                            stl_files_failed.append((stl_filename, str(e)))
                            print(f"Failed to convert mesh {stl_filename}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        stl_files_failed.append((os.path.basename(source_path), "Source file not found"))
                        print(f"Mesh file not found: {source_path}")

                # コリジョンメッシュファイルも変換して保存（collidersリストから）
                if hasattr(node, 'colliders') and node.colliders:
                    for collider in node.colliders:
                        if not collider.get('enabled', False):
                            continue
                        if collider.get('type') == 'mesh' and collider.get('mesh'):
                            collider_mesh = collider['mesh']
                            # collider_meshは相対パスまたはファイル名の可能性がある
                            if hasattr(node, 'stl_file') and node.stl_file:
                                visual_dir = os.path.dirname(node.stl_file)
                                collider_source_path = os.path.join(visual_dir, collider_mesh)
                            else:
                                collider_source_path = collider_mesh

                            if os.path.exists(collider_source_path):
                                collider_original_filename = os.path.basename(collider_source_path)
                                # ファイル名の拡張子を選択された形式に変更
                                collider_base_name = os.path.splitext(collider_original_filename)[0]
                                collider_new_filename = f"{collider_base_name}{selected_format}"
                                collider_dest_path = os.path.join(meshes_dir, collider_new_filename)

                                # 既にコピー済みでない場合のみ変換
                                if collider_new_filename not in stl_files_copied:
                                    try:
                                        # メッシュファイルを読み込んで、選択された形式で保存
                                        collider_poly_data, collider_volume, _ = load_mesh_to_polydata(collider_source_path)
                                        save_polydata_to_mesh(collider_dest_path, collider_poly_data)
                                        stl_files_copied.append(collider_new_filename)
                                        print(f"Converted and saved collider mesh: {collider_original_filename} -> {collider_new_filename}")
                                    except Exception as e:
                                        stl_files_failed.append((collider_original_filename, str(e)))
                                        print(f"Failed to convert collider mesh {collider_original_filename}: {str(e)}")
                                        import traceback
                                        traceback.print_exc()
                            else:
                                stl_files_failed.append((os.path.basename(collider_source_path), "Collider mesh file not found"))
                                print(f"Collider mesh file not found: {collider_source_path}")

            print(f"\nMesh files copied: {len(stl_files_copied)}")
            if stl_files_failed:
                print(f"Mesh files failed: {len(stl_files_failed)}")
                for filename, error in stl_files_failed:
                    print(f"  - {filename}: {error}")

            # URDFファイルのパスを設定（クリーンな名前を使用）
            urdf_file = os.path.join(urdf_dir, f"{clean_name}.urdf")

            # 以下、URDFファイルの書き込み処理
            with open(urdf_file, 'w', encoding='utf-8') as f:
                # ヘッダー（クリーンな名前を使用）
                f.write('<?xml version="1.0"?>\n')
                f.write(f'<robot name="{clean_name}">\n\n')

                # マテリアル定義の収集
                materials = {}
                for node in self.all_nodes():
                    if hasattr(node, 'node_color'):
                        rgb = node.node_color
                        if len(rgb) >= 3:
                            hex_color = '#{:02x}{:02x}{:02x}'.format(
                                int(rgb[0] * 255),
                                int(rgb[1] * 255),
                                int(rgb[2] * 255)
                            )
                            materials[hex_color] = rgb
                
                # マテリアルの書き出し
                f.write('<!-- material color setting -->\n')
                for hex_color, rgb in materials.items():
                    f.write(f'<material name="{hex_color}">\n')
                    f.write(f'  <color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"/>\n')
                    f.write('</material>\n')
                f.write('\n')

                # base_linkから開始して、ツリー構造を順番に出力
                visited_nodes = set()
                base_node = self.get_node_by_name('base_link')
                if base_node:
                    self._write_tree_structure(f, base_node, None, visited_nodes, materials)
                
                f.write('</robot>\n')

                print(f"URDF exported to: {urdf_file}")

                # 完了メッセージを作成
                export_summary = "✓ URDF Export Completed Successfully\n"
                export_summary += "=" * 50 + "\n\n"
                export_summary += f"Robot Name: {clean_name}\n\n"
                export_summary += f"Output Directory:\n{description_dir}\n\n"
                export_summary += f"URDF File:\n{urdf_file}\n\n"
                export_summary += f"Meshes Directory:\n{meshes_dir}\n\n"
                export_summary += f"Mesh Files Copied: {len(stl_files_copied)}\n"

                if stl_files_copied:
                    export_summary += "\nCopied Mesh Files:\n"
                    for filename in stl_files_copied[:10]:  # 最大10個まで表示
                        export_summary += f"  • {filename}\n"
                    if len(stl_files_copied) > 10:
                        export_summary += f"  ... and {len(stl_files_copied) - 10} more\n"

                if stl_files_failed:
                    export_summary += f"\n⚠ Warning: {len(stl_files_failed)} file(s) failed to copy\n"
                    for filename, error in stl_files_failed[:5]:  # 最大5個まで表示
                        export_summary += f"  • {filename}: {error}\n"
                    if len(stl_files_failed) > 5:
                        export_summary += f"  ... and {len(stl_files_failed) - 5} more\n"

                export_summary += "\n" + "=" * 50

                QtWidgets.QMessageBox.information(
                    self.widget,
                    "URDF Export Complete",
                    export_summary
                )

                return True

        except Exception as e:
            error_msg = f"Error exporting URDF: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            QtWidgets.QMessageBox.critical(
                self.widget,
                "Export Error",
                error_msg
            )
            return False

    def _write_tree_structure(self, file, node, parent_node, visited_nodes, materials, mesh_format=".stl"):
        """ツリー構造を順番に出力"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless Decorationノードはスキップ（親ノードの<visual>として処理済み）
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        # Hide Meshにチェックが入っているノードはスキップ
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            print(f"Skipping node with hide_mesh=True: {node.name()}")
            return

        if node.name() == "base_link":
            # base_linkの出力
            self._write_base_link(file)
        
        # 現在のノードに接続されているジョイントとリンクを処理
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                if child_node not in visited_nodes:
                    # Massless Decorationでないノードのみジョイントとリンクを出力
                    if not (hasattr(child_node, 'massless_decoration') and child_node.massless_decoration):
                        # まずジョイントを出力
                        self._write_joint(file, node, child_node)
                        file.write('\n')
                        
                        # 次にリンクを出力
                        self._write_link(file, child_node, materials, mesh_format)
                        file.write('\n')
                    
                    # 再帰的に子ノードを処理
                    self._write_tree_structure(file, child_node, node, visited_nodes, materials, mesh_format)

    def _is_base_link_at_defaults(self, base_node):
        """base_linkがすべてデフォルト値かチェック"""
        if not base_node:
            return True

        # Mass check
        if hasattr(base_node, 'mass_value') and base_node.mass_value != 0.0:
            return False

        # Inertia check
        if hasattr(base_node, 'inertia') and base_node.inertia:
            for value in base_node.inertia.values():
                if value != 0.0:
                    return False

        # Mesh file check
        if hasattr(base_node, 'stl_file') and base_node.stl_file:
            return False

        # Joint parameters check
        if hasattr(base_node, 'rotation_axis') and base_node.rotation_axis != 3:  # Fixed
            return False
        if hasattr(base_node, 'joint_lower') and base_node.joint_lower != 0.0:
            return False
        if hasattr(base_node, 'joint_upper') and base_node.joint_upper != 0.0:
            return False

        # Color check (white is default)
        if hasattr(base_node, 'node_color') and base_node.node_color != DEFAULT_COLOR_WHITE:
            return False

        # Output port count check (1 is default)
        if hasattr(base_node, 'output_count') and base_node.output_count != 1:
            return False

        return True

    def _write_base_link(self, file):
        """base_linkの出力"""
        base_node = self.get_node_by_name('base_link')

        # Blanklinkフラグ または すべてデフォルト値の場合は空のリンクとして出力
        is_blank = (base_node and hasattr(base_node, 'blank_link') and base_node.blank_link)
        is_all_defaults = self._is_base_link_at_defaults(base_node)

        if base_node and not is_blank and not is_all_defaults:
            # Blanklinkがオフの場合、通常のリンクとして出力（パラメータ付き）
            file.write('  <link name="base_link">\n')

            # 慣性パラメータ
            if hasattr(base_node, 'mass_value') and hasattr(base_node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial Originの出力
                if hasattr(base_node, 'inertial_origin') and isinstance(base_node.inertial_origin, dict):
                    xyz = base_node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    rpy = base_node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                    file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
                else:
                    file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write(f'      <mass value="{format_float_no_exp(base_node.mass_value)}"/>\n')
                file.write('      <inertia')
                for key, value in base_node.inertia.items():
                    file.write(f' {key}="{format_float_no_exp(value)}"')
                file.write('/>\n')
                file.write('    </inertial>\n')

            # ビジュアルとコリジョン
            if hasattr(base_node, 'stl_file') and base_node.stl_file:
                mesh_dir_name = "meshes"
                if self.meshes_dir:
                    dir_name = os.path.basename(self.meshes_dir)
                    if dir_name.startswith('mesh'):
                        mesh_dir_name = dir_name

                stl_filename = os.path.basename(base_node.stl_file)
                package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{stl_filename}"

                # メインのビジュアル
                file.write('    <visual>\n')
                # Visual originを出力（デフォルト値でない場合のみ）
                file.write(self._format_visual_origin(base_node))
                file.write('      <geometry>\n')
                # Mesh scaleを出力（デフォルト値でない場合のみ）
                scale_attr = self._format_mesh_scale(base_node)
                file.write(f'        <mesh filename="{package_path}"{scale_attr}/>\n')
                file.write('      </geometry>\n')

                # カラー情報を追加
                if hasattr(base_node, 'node_color') and len(base_node.node_color) >= 3:
                    rgb = base_node.node_color
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(rgb[0] * 255),
                        int(rgb[1] * 255),
                        int(rgb[2] * 255)
                    )
                    file.write(f'      <material name="{hex_color}"/>\n')

                file.write('    </visual>\n')

                # コリジョン (新しいヘルパー関数を使用)
                self._write_urdf_collision(file, base_node, package_path, mesh_dir_name)

            file.write('  </link>\n\n')
        else:
            # Blanklinkがオンの場合、パラメータなしのリンクとして出力（デフォルト動作）
            file.write('  <link name="base_link"/>\n\n')

    def generate_tree_text(self, node, level=0):
        tree_text = "  " * level + node.name() + "\n"
        for output_port in node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()
                tree_text += self.generate_tree_text(child_node, level + 1)
        return tree_text

    def get_node_by_name(self, name):
        for node in self.all_nodes():
            if node.name() == name:
                return node
        return None

    def update_last_stl_directory(self, file_path):
        self.last_stl_directory = os.path.dirname(file_path)

    def show_inspector(self, node, screen_pos=None):
        """
        ノードのインスペクタウィンドウを表示
        """
        try:
            # 既存のインスペクタウィンドウをクリーンアップ
            if hasattr(self, 'inspector_window') and self.inspector_window is not None:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                except Exception:
                    pass
                self.inspector_window = None

            # 新しいインスペクタウィンドウを作成
            self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)
            
            # ウィンドウサイズを取得
            inspector_size = self.inspector_window.sizeHint()

            if self.widget and self.widget.window():
                # 保存された位置があればそれを使用し、なければデフォルト位置を計算
                if hasattr(self, 'last_inspector_position') and self.last_inspector_position:
                    x = self.last_inspector_position.x()
                    y = self.last_inspector_position.y()
                    
                    # スクリーンの情報を取得して位置を検証
                    screen = QtWidgets.QApplication.primaryScreen()
                    screen_geo = screen.availableGeometry()
                    
                    # 画面外にはみ出していないか確認
                    if x < screen_geo.x() or x + inspector_size.width() > screen_geo.right() or \
                    y < screen_geo.y() or y + inspector_size.height() > screen_geo.bottom():
                        # 画面外の場合はデフォルト位置を使用
                        main_geo = self.widget.window().geometry()
                        x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                        y = main_geo.y() + 50
                else:
                    # デフォルトの位置を計算
                    main_geo = self.widget.window().geometry()
                    x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                    y = main_geo.y() + 50

                # ウィンドウの初期設定と表示
                self.inspector_window.setWindowTitle(f"Node Inspector - {node.name()}")
                self.inspector_window.current_node = node
                self.inspector_window.graph = self
                self.inspector_window.update_info(node)
                
                self.inspector_window.move(x, y)
                self.inspector_window.show()
                self.inspector_window.raise_()
                self.inspector_window.activateWindow()

                # インスペクター表示時は点滅を停止
                if self.stl_viewer:
                    self.stl_viewer.clear_highlight()

                print(f"Inspector window displayed for node: {node.name()}")

        except Exception as e:
            print(f"Error showing inspector: {str(e)}")
            traceback.print_exc()

    def show_closed_loop_inspector(self, node, screen_pos=None):
        """
        閉リンクノードのインスペクタウィンドウを表示
        """
        try:
            # 既存の閉リンクインスペクタウィンドウをクリーンアップ
            if hasattr(self, 'closed_loop_inspector_window') and self.closed_loop_inspector_window is not None:
                try:
                    self.closed_loop_inspector_window.close()
                    self.closed_loop_inspector_window.deleteLater()
                except Exception:
                    pass
                self.closed_loop_inspector_window = None

            # 新しい閉リンクインスペクタウィンドウを作成
            self.closed_loop_inspector_window = ClosedLoopInspectorWindow(graph=self)

            # ウィンドウサイズを取得
            inspector_size = self.closed_loop_inspector_window.sizeHint()

            if self.widget and self.widget.window():
                # デフォルトの位置を計算（画面中央）
                main_geo = self.widget.window().geometry()
                x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                y = main_geo.y() + 50

                # ウィンドウの初期設定と表示
                self.closed_loop_inspector_window.setWindowTitle(f"Closed-Loop Joint Inspector - {node.name()}")
                self.closed_loop_inspector_window.set_node(node)

                self.closed_loop_inspector_window.move(x, y)
                self.closed_loop_inspector_window.show()
                self.closed_loop_inspector_window.raise_()
                self.closed_loop_inspector_window.activateWindow()

                print(f"Closed-loop inspector window displayed for node: {node.name()}")

        except Exception as e:
            print(f"Error showing closed-loop inspector: {str(e)}")
            traceback.print_exc()

    def create_node(self, node_type, name=None, pos=None):
        new_node = super(CustomNodeGraph, self).create_node(node_type, name)

        if pos is None:
            pos = QPointF(0, 0)
        elif isinstance(pos, (tuple, list)):
            pos = QPointF(*pos)

        print(f"Initial position for new node: {pos}")  # デバッグ情報

        adjusted_pos = self.find_non_overlapping_position(pos)
        print(f"Adjusted position for new node: {adjusted_pos}")  # デバッグ情報

        # グリッドスナップを適用
        snapped_x, snapped_y = self.snap_to_grid(adjusted_pos.x(), adjusted_pos.y())
        print(f"Grid-snapped position: ({snapped_x}, {snapped_y})")  # デバッグ情報

        new_node.set_pos(snapped_x, snapped_y)

        # Settingsのデフォルト値を新しいノードに適用
        # init_node_propertiesで既に初期化されているが、Settingsの値で上書き
        if hasattr(new_node, 'joint_effort'):
            new_node.joint_effort = self.default_joint_effort
        if hasattr(new_node, 'joint_velocity'):
            new_node.joint_velocity = self.default_joint_velocity
        if hasattr(new_node, 'joint_damping'):
            # Settingsのdefault_damping_kvを使用（default_joint_dampingは古い定数）
            new_node.joint_damping = self.default_damping_kv
        if hasattr(new_node, 'joint_stiffness'):
            # Settingsのdefault_stiffness_kpを使用（default_joint_stiffnessは古い定数）
            new_node.joint_stiffness = self.default_stiffness_kp
        if hasattr(new_node, 'joint_margin'):
            new_node.joint_margin = self.default_margin
        if hasattr(new_node, 'joint_armature'):
            new_node.joint_armature = self.default_armature
        if hasattr(new_node, 'joint_frictionloss'):
            new_node.joint_frictionloss = self.default_frictionloss
        
        # joint_lowerとjoint_upperもSettingsのangle_rangeから設定
        if hasattr(new_node, 'joint_lower'):
            new_node.joint_lower = -self.default_angle_range
        if hasattr(new_node, 'joint_upper'):
            new_node.joint_upper = self.default_angle_range
        
        print(f"Applied Settings default values to new node: effort={self.default_joint_effort}, velocity={self.default_joint_velocity}, damping={self.default_damping_kv}, stiffness={self.default_stiffness_kp}, margin={self.default_margin}, armature={self.default_armature}, frictionloss={self.default_frictionloss}, angle_range={self.default_angle_range}")
        
        # === 必須ログ: ノード作成時の慣性値を確認 ===
        node_name = new_node.name() if hasattr(new_node, 'name') else 'unknown'
        if 'arm_lower' in node_name.lower():
            if hasattr(new_node, 'inertia') and new_node.inertia:
                print(f"\n[CREATE_NODE] link_name={node_name}, node_type={node_type}")
                print(f"  Initial node.inertia: ixx={new_node.inertia.get('ixx', 0):.9e}, ixy={new_node.inertia.get('ixy', 0):.9e}, ixz={new_node.inertia.get('ixz', 0):.9e}")
                print(f"                        iyy={new_node.inertia.get('iyy', 0):.9e}, iyz={new_node.inertia.get('iyz', 0):.9e}, izz={new_node.inertia.get('izz', 0):.9e}")
            else:
                print(f"\n[CREATE_NODE] link_name={node_name}, node_type={node_type}")
                print(f"  WARNING: node.inertia is not set after creation!")

        return new_node

    def find_non_overlapping_position(self, pos, offset_x=50, offset_y=30, items_per_row=16):
        all_nodes = self.all_nodes()
        current_node_count = len(all_nodes)
        
        # 現在の行を計算
        row = current_node_count // items_per_row
        
        # 行内での位置を計算
        position_in_row = current_node_count % items_per_row
        
        # 基準となるX座標を計算（各行の開始X座標）
        base_x = pos.x()
        
        # 基準となるY座標を計算
        # 新しい行は前の行の開始位置から200ポイント下
        base_y = pos.y() + (row * 200)
        
        # 現在のノードのX,Y座標を計算
        new_x = base_x + (position_in_row * offset_x)
        new_y = base_y + (position_in_row * offset_y)
        
        new_pos = QPointF(new_x, new_y)
        
        print(f"Positioning node {current_node_count + 1}")
        print(f"Row: {row + 1}, Position in row: {position_in_row + 1}")
        print(f"Position: ({new_pos.x()}, {new_pos.y()})")
        
        # オーバーラップチェックと位置の微調整
        iteration = 0
        while any(self.nodes_overlap(new_pos, node.pos()) for node in all_nodes):
            new_pos += QPointF(5, 5)  # 微小なオフセットで調整
            iteration += 1
            if iteration > 10:
                break
        
        return new_pos

    def nodes_overlap(self, pos1, pos2, threshold=5):
        pos1 = self.ensure_qpointf(pos1)
        pos2 = self.ensure_qpointf(pos2)
        overlap = (abs(pos1.x() - pos2.x()) < threshold and
                abs(pos1.y() - pos2.y()) < threshold)
        # デバッグ出力を条件付きに
        if overlap:
            print(f"Overlap detected: pos1={pos1}, pos2={pos2}")
        return overlap

    def ensure_qpointf(self, pos):
        if isinstance(pos, QPointF):
            return pos
        elif isinstance(pos, (tuple, list)):
            return QPointF(*pos)
        else:
            print(f"Warning: Unsupported position type: {type(pos)}")  # デバッグ情報
            return QPointF(0, 0)  # デフォルト値を返す


    def _save_node_data(self, node, project_dir):
        """ノードデータをXML要素として保存
        
        Args:
            node: 保存するノード
            project_dir: プロジェクトディレクトリ（相対パス計算用）
            
        Returns:
            ET.Element: ノードデータのXML要素
        """
        node_elem = ET.Element("node")
        
        # 基本情報
        ET.SubElement(node_elem, "name").text = node.name()
        ET.SubElement(node_elem, "type").text = node.type_
        
        # 位置情報 (QPointFまたはlist/tupleに対応)
        pos = node.pos()
        try:
            # Try to normalize to QPointF using helper if available
            if hasattr(self, 'ensure_qpointf'):
                pos_q = self.ensure_qpointf(pos)
            else:
                if isinstance(pos, (list, tuple)):
                    pos_q = QPointF(pos[0], pos[1])
                else:
                    pos_q = pos

            ET.SubElement(node_elem, "pos_x").text = str(pos_q.x())
            ET.SubElement(node_elem, "pos_y").text = str(pos_q.y())
        except Exception:
            # Fallback: try index access or string conversion
            try:
                if isinstance(pos, (list, tuple)):
                    ET.SubElement(node_elem, "pos_x").text = str(pos[0])
                    ET.SubElement(node_elem, "pos_y").text = str(pos[1])
                else:
                    ET.SubElement(node_elem, "pos_x").text = str(getattr(pos, 'x', lambda: 0)())
                    ET.SubElement(node_elem, "pos_y").text = str(getattr(pos, 'y', lambda: 0)())
            except Exception:
                ET.SubElement(node_elem, "pos_x").text = "0"
                ET.SubElement(node_elem, "pos_y").text = "0"
        
        # カスタムプロパティ
        if hasattr(node, 'stl_file') and node.stl_file:
            # 相対パスに変換
            try:
                rel_path = os.path.relpath(node.stl_file, project_dir)
                ET.SubElement(node_elem, "stl_file").text = rel_path
            except (ValueError, TypeError):
                ET.SubElement(node_elem, "stl_file").text = node.stl_file
        
        # その他の共通プロパティ
        if hasattr(node, 'mass_value'):
            ET.SubElement(node_elem, "mass").text = str(node.mass_value)
        if hasattr(node, 'volume_value'):
            ET.SubElement(node_elem, "volume").text = str(node.volume_value)
        if hasattr(node, 'node_color'):
            color_str = ' '.join(str(c) for c in node.node_color)
            ET.SubElement(node_elem, "color").text = color_str
        if hasattr(node, 'rotation_axis'):
            ET.SubElement(node_elem, "rotation_axis").text = str(node.rotation_axis)
        if hasattr(node, 'xml_file') and node.xml_file:
            try:
                rel_path = os.path.relpath(node.xml_file, project_dir)
                ET.SubElement(node_elem, "xml_file").text = rel_path
            except (ValueError, TypeError):
                ET.SubElement(node_elem, "xml_file").text = node.xml_file
        
        # Inertial情報
        if hasattr(node, 'inertia') and node.inertia:
            inertia_elem = ET.SubElement(node_elem, "inertia")
            for key, value in node.inertia.items():
                ET.SubElement(inertia_elem, key).text = str(value)
        
        if hasattr(node, 'inertial_origin') and node.inertial_origin:
            io_elem = ET.SubElement(node_elem, "inertial_origin")
            if 'xyz' in node.inertial_origin:
                xyz_str = ' '.join(str(v) for v in node.inertial_origin['xyz'])
                ET.SubElement(io_elem, "xyz").text = xyz_str
            if 'rpy' in node.inertial_origin:
                rpy_str = ' '.join(str(v) for v in node.inertial_origin['rpy'])
                ET.SubElement(io_elem, "rpy").text = rpy_str
        
        # Visual Origin情報
        if hasattr(node, 'visual_origin') and node.visual_origin:
            vo_elem = ET.SubElement(node_elem, "visual_origin")
            if 'xyz' in node.visual_origin:
                xyz_str = ' '.join(str(v) for v in node.visual_origin['xyz'])
                ET.SubElement(vo_elem, "xyz").text = xyz_str
            if 'rpy' in node.visual_origin:
                rpy_str = ' '.join(str(v) for v in node.visual_origin['rpy'])
                ET.SubElement(vo_elem, "rpy").text = rpy_str
        
        # Joint情報
        if hasattr(node, 'joint_lower'):
            ET.SubElement(node_elem, "joint_lower").text = str(node.joint_lower)
        if hasattr(node, 'joint_upper'):
            ET.SubElement(node_elem, "joint_upper").text = str(node.joint_upper)
        if hasattr(node, 'joint_effort'):
            ET.SubElement(node_elem, "joint_effort").text = str(node.joint_effort)
        if hasattr(node, 'joint_velocity'):
            ET.SubElement(node_elem, "joint_velocity").text = str(node.joint_velocity)
        if hasattr(node, 'joint_damping'):
            ET.SubElement(node_elem, "joint_damping").text = str(node.joint_damping)
        if hasattr(node, 'joint_stiffness'):
            ET.SubElement(node_elem, "joint_stiffness").text = str(node.joint_stiffness)
        if hasattr(node, 'joint_margin'):
            ET.SubElement(node_elem, "joint_margin").text = str(node.joint_margin)
        if hasattr(node, 'joint_armature'):
            ET.SubElement(node_elem, "joint_armature").text = str(node.joint_armature)
        if hasattr(node, 'joint_frictionloss'):
            ET.SubElement(node_elem, "joint_frictionloss").text = str(node.joint_frictionloss)
        
        # Body Angle（子ノードの初期回転角度、ラジアンで保存）
        if hasattr(node, 'body_angle'):
            body_angle_str = ' '.join(str(v) for v in node.body_angle)
            ET.SubElement(node_elem, "body_angle").text = body_angle_str
        
        # メッシュ属性
        if hasattr(node, 'is_mesh_reversed'):
            ET.SubElement(node_elem, "is_mesh_reversed").text = str(node.is_mesh_reversed)
        if hasattr(node, 'mesh_original_color') and node.mesh_original_color:
            color_str = ' '.join(str(c) for c in node.mesh_original_color)
            ET.SubElement(node_elem, "mesh_original_color").text = color_str

        # リンク属性
        if hasattr(node, 'blank_link'):
            ET.SubElement(node_elem, "blank_link").text = str(node.blank_link)
        if hasattr(node, 'massless_decoration'):
            ET.SubElement(node_elem, "massless_decoration").text = str(node.massless_decoration)
        if hasattr(node, 'hide_mesh'):
            ET.SubElement(node_elem, "hide_mesh").text = str(node.hide_mesh)
        
        # Collider情報（新形式：複数コライダー対応）
        if hasattr(node, 'colliders') and node.colliders:
            colliders_elem = ET.SubElement(node_elem, "colliders")
            for collider in node.colliders:
                collider_elem = ET.SubElement(colliders_elem, "collider")
                
                # タイプ
                ET.SubElement(collider_elem, "type").text = collider.get('type') or 'primitive'
                
                # 有効/無効
                ET.SubElement(collider_elem, "enabled").text = str(collider.get('enabled', True))
                
                # メッシュファイル
                if 'mesh' in collider and collider['mesh']:
                    try:
                        rel_path = os.path.relpath(collider['mesh'], project_dir)
                        ET.SubElement(collider_elem, "mesh").text = rel_path
                    except (ValueError, TypeError):
                        ET.SubElement(collider_elem, "mesh").text = collider['mesh']
                
                # メッシュスケール
                if 'mesh_scale' in collider and collider['mesh_scale']:
                    scale_str = ' '.join(str(v) for v in collider['mesh_scale'])
                    ET.SubElement(collider_elem, "mesh_scale").text = scale_str
                
                # プリミティブデータ
                if 'data' in collider and collider['data']:
                    data_elem = ET.SubElement(collider_elem, "data")
                    data = collider['data']
                    
                    # タイプ
                    if 'type' in data:
                        ET.SubElement(data_elem, "type").text = data['type']
                    
                    # ジオメトリ（辞書を文字列に変換）
                    if 'geometry' in data and data['geometry']:
                        ET.SubElement(data_elem, "geometry").text = str(data['geometry'])
                
                # 位置（collider直下、常に保存）
                position = collider.get('position', [0.0, 0.0, 0.0])
                if not position:  # 空リストの場合
                    position = [0.0, 0.0, 0.0]
                pos_str = ' '.join(str(v) for v in position)
                ET.SubElement(collider_elem, "position").text = pos_str
                
                # 回転（collider直下、ラジアン、常に保存）
                rotation = collider.get('rotation', [0.0, 0.0, 0.0])
                if not rotation:  # 空リストの場合
                    rotation = [0.0, 0.0, 0.0]
                rot_str = ' '.join(str(v) for v in rotation)
                ET.SubElement(collider_elem, "rotation").text = rot_str

        # Points (FooNodeの場合)
        if hasattr(node, 'points') and isinstance(node, FooNode):
            # 実際のポート数とpointsの数を同期
            # output_ports()とoutput_countの両方を参照して、より安全にポート数を取得
            actual_port_count = len(node.output_ports())
            output_count = getattr(node, 'output_count', 0)
            points_count = len(node.points) if node.points else 0
            
            # ポート数を決定（output_ports()とoutput_countの最大値を使用）
            # 末端ノードの場合、output_ports()が空を返す可能性があるため、output_countも参照
            final_port_count = max(actual_port_count, output_count, points_count)
            
            # 不一致がある場合は調整
            if final_port_count != points_count:
                if actual_port_count != points_count or output_count != points_count:
                    print(f"Warning: Port count mismatch for {node.name()}: ports={actual_port_count}, output_count={output_count}, points={points_count}, using={final_port_count}")
                # pointsの数を最終的なポート数に合わせる
                if final_port_count > points_count:
                    # ポートが多い場合は、不足分のポイントを追加
                    for i in range(points_count, final_port_count):
                        point_data = create_point_data(i + 1)
                        node.points.append(point_data)
                elif final_port_count < points_count:
                    # ポイントが多い場合は、余分なポイントを削除
                    node.points = node.points[:final_port_count]
                # output_countも更新
                node.output_count = final_port_count
            
            # ポイントデータを保存（末端ノードでもoutポートは保存する）
            if node.points:
                points_elem = ET.SubElement(node_elem, "points")
                for point in node.points:
                    point_elem = ET.SubElement(points_elem, "point")
                    if 'name' in point:
                        ET.SubElement(point_elem, "name").text = point['name']
                    if 'type' in point:
                        ET.SubElement(point_elem, "type").text = point['type']
                    if 'xyz' in point:
                        xyz_str = ' '.join(str(v) for v in point['xyz'])
                        ET.SubElement(point_elem, "xyz").text = xyz_str
                    if 'rpy' in point:
                        rpy_str = ' '.join(str(v) for v in point['rpy'])
                        ET.SubElement(point_elem, "rpy").text = rpy_str
                    if 'angle' in point:
                        angle_str = ' '.join(str(v) for v in point['angle'])
                        ET.SubElement(point_elem, "angle").text = angle_str
        
        return node_elem


    def _load_node_data(self, node_elem, connected_ports=None):
        """XML要素からノードデータを読み込み
        
        Args:
            node_elem: ノードデータのXML要素
            
        Returns:
            読み込まれたノード、またはNone
        """
        try:
            node_name = node_elem.find("name").text
            node_type = node_elem.find("type").text

            # Normalize node_type: if a short name was saved (e.g. 'FooNode' or 'BaseLinkNode'),
            # map it to the full NodeGraphQt identifier like 'insilico.nodes.FooNode'.
            try:
                if node_type and '.' not in node_type:
                    # If class with that name exists in this module, use its __identifier__
                    cls = globals().get(node_type)
                    if cls is None:
                        # Also try NODE_NAME matching
                        for g in globals().values():
                            try:
                                if getattr(g, 'NODE_NAME', None) == node_type:
                                    cls = g
                                    break
                            except Exception:
                                continue
                    if cls and hasattr(cls, '__identifier__'):
                        node_type = f"{cls.__identifier__}.{cls.__name__}"
            except Exception:
                pass
            
            # base_linkの場合、既存のbase_linkがあるかチェック
            existing_base_link = None
            if node_name == 'base_link':
                existing_base_link = self.get_node_by_name('base_link')
            
            # ノードを作成
            node = self.create_node(node_type, name=node_name)
            if not node:
                print(f"Warning: Could not create node of type {node_type}")
                return None
            
            # 位置情報
            pos_x_elem = node_elem.find("pos_x")
            pos_y_elem = node_elem.find("pos_y")
            if pos_x_elem is not None and pos_y_elem is not None:
                node.set_pos(float(pos_x_elem.text), float(pos_y_elem.text))
            
            # base_linkの場合、データがあるかどうかをチェック
            is_base_link_with_data = False
            if node_name == 'base_link':
                # データがあるかチェック
                mass_elem = node_elem.find("mass")
                stl_elem = node_elem.find("stl_file")
                inertia_elem = node_elem.find("inertia")
                collider_type_elem = node_elem.find("collider_type")
                collider_enabled_elem = node_elem.find("collider_enabled")
                points_elem = node_elem.find("points")
                
                has_mass = mass_elem is not None and float(mass_elem.text) > 0.0 if mass_elem is not None and mass_elem.text else False
                has_stl = stl_elem is not None and stl_elem.text
                has_inertia = False
                if inertia_elem is not None:
                    for key in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                        key_elem = inertia_elem.find(key)
                        if key_elem is not None and float(key_elem.text) != 0.0:
                            has_inertia = True
                            break
                has_collider = (collider_type_elem is not None and collider_type_elem.text) or \
                              (collider_enabled_elem is not None and collider_enabled_elem.text.lower() == 'true')
                has_points = points_elem is not None and len(points_elem.findall("point")) > 0
                
                is_base_link_with_data = has_mass or has_stl or has_inertia or has_collider or has_points
                
                print(f"  Load: Checking base_link data:")
                print(f"    has_mass: {has_mass}, has_stl: {has_stl}, has_inertia: {has_inertia}, has_collider: {has_collider}, has_points: {has_points}")
                print(f"    is_base_link_with_data: {is_base_link_with_data}, existing_base_link: {existing_base_link is not None}")
                
                # 既存のbase_linkがあり、データがある場合はbase_link_subを作成
                if existing_base_link and is_base_link_with_data:
                    print(f"  Load: base_link has data and existing base_link found, creating base_link_sub")
                    base_link_pos = existing_base_link.pos()
                    # pos()がリストかQPointFか判定
                    if isinstance(base_link_pos, (list, tuple)):
                        base_link_x = base_link_pos[0]
                        base_link_y = base_link_pos[1]
                    else:
                        base_link_x = base_link_pos.x()
                        base_link_y = base_link_pos.y()
                    grid_spacing_value = 150
                    base_link_sub_pos = QtCore.QPointF(base_link_x + grid_spacing_value, base_link_y)
                    
                    base_link_sub_node = self.create_node(
                        'insilico.nodes.FooNode',
                        name='base_link_sub',
                        pos=base_link_sub_pos
                    )
                    
                    # 初期化時に追加されたポートとポイントをクリア（必ず実行）
                    current_ports = len(base_link_sub_node.output_ports())
                    # すべての出力ポートの接続をクリアしてから削除
                    for i in range(1, current_ports + 1):
                        port_name = f'out_{i}'
                        port = base_link_sub_node.get_output(port_name)
                        if port:
                            port.clear_connections()
                    
                    # すべての出力ポートを削除
                    while current_ports > 0:
                        base_link_sub_node.remove_output()
                        current_ports -= 1
                    
                    # ポイントデータと累積座標をクリア
                    base_link_sub_node.points = []
                    base_link_sub_node.cumulative_coords = []
                    base_link_sub_node.output_count = 0
                    
                    # 読み込んだbase_linkのデータをbase_link_subに設定
                    # STLファイル
                    if stl_elem is not None and stl_elem.text:
                        stl_path = os.path.join(self.project_dir, stl_elem.text)
                        if os.path.exists(stl_path):
                            base_link_sub_node.stl_file = stl_path
                    
                    # その他のプロパティ
                    if mass_elem is not None:
                        base_link_sub_node.mass_value = float(mass_elem.text)
                    
                    volume_elem = node_elem.find("volume")
                    if volume_elem is not None:
                        base_link_sub_node.volume_value = float(volume_elem.text)
                    
                    color_elem = node_elem.find("color")
                    if color_elem is not None:
                        base_link_sub_node.node_color = [float(c) for c in color_elem.text.split()]
                    
                    rotation_axis_elem = node_elem.find("rotation_axis")
                    if rotation_axis_elem is not None:
                        base_link_sub_node.rotation_axis = int(rotation_axis_elem.text)
                    else:
                        base_link_sub_node.rotation_axis = 3  # Fixed
                    
                    # Inertial情報
                    if inertia_elem is not None:
                        base_link_sub_node.inertia = {}
                        for key in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                            key_elem = inertia_elem.find(key)
                            if key_elem is not None:
                                base_link_sub_node.inertia[key] = float(key_elem.text)
                    
                    io_elem = node_elem.find("inertial_origin")
                    if io_elem is not None:
                        base_link_sub_node.inertial_origin = {}
                        xyz_elem = io_elem.find("xyz")
                        if xyz_elem is not None:
                            base_link_sub_node.inertial_origin['xyz'] = [float(v) for v in xyz_elem.text.split()]
                        rpy_elem = io_elem.find("rpy")
                        if rpy_elem is not None:
                            base_link_sub_node.inertial_origin['rpy'] = [float(v) for v in rpy_elem.text.split()]
                    
                    # Visual Origin情報
                    vo_elem = node_elem.find("visual_origin")
                    if vo_elem is not None:
                        base_link_sub_node.visual_origin = {}
                        xyz_elem = vo_elem.find("xyz")
                        if xyz_elem is not None:
                            base_link_sub_node.visual_origin['xyz'] = [float(v) for v in xyz_elem.text.split()]
                        rpy_elem = vo_elem.find("rpy")
                        if rpy_elem is not None:
                            base_link_sub_node.visual_origin['rpy'] = [float(v) for v in rpy_elem.text.split()]
                    elif not hasattr(base_link_sub_node, 'visual_origin'):
                        base_link_sub_node.visual_origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
                    
                    # Joint情報
                    joint_lower_elem = node_elem.find("joint_lower")
                    if joint_lower_elem is not None:
                        base_link_sub_node.joint_lower = float(joint_lower_elem.text)
                    
                    joint_upper_elem = node_elem.find("joint_upper")
                    if joint_upper_elem is not None:
                        base_link_sub_node.joint_upper = float(joint_upper_elem.text)
                    
                    joint_effort_elem = node_elem.find("joint_effort")
                    if joint_effort_elem is not None:
                        base_link_sub_node.joint_effort = float(joint_effort_elem.text)
                    
                    joint_velocity_elem = node_elem.find("joint_velocity")
                    if joint_velocity_elem is not None:
                        base_link_sub_node.joint_velocity = float(joint_velocity_elem.text)
                    
                    joint_damping_elem = node_elem.find("joint_damping")
                    if joint_damping_elem is not None:
                        base_link_sub_node.joint_damping = float(joint_damping_elem.text)
                    
                    joint_stiffness_elem = node_elem.find("joint_stiffness")
                    if joint_stiffness_elem is not None:
                        base_link_sub_node.joint_stiffness = float(joint_stiffness_elem.text)
                    
                    joint_margin_elem = node_elem.find("joint_margin")
                    if joint_margin_elem is not None:
                        base_link_sub_node.joint_margin = float(joint_margin_elem.text)
                    
                    joint_armature_elem = node_elem.find("joint_armature")
                    if joint_armature_elem is not None:
                        base_link_sub_node.joint_armature = float(joint_armature_elem.text)
                    
                    joint_frictionloss_elem = node_elem.find("joint_frictionloss")
                    if joint_frictionloss_elem is not None:
                        base_link_sub_node.joint_frictionloss = float(joint_frictionloss_elem.text)
                    
                    # Body Angle（子ノードの初期回転角度、ラジアンで保存されている）
                    body_angle_elem = node_elem.find("body_angle")
                    if body_angle_elem is not None:
                        base_link_sub_node.body_angle = [float(v) for v in body_angle_elem.text.split()]
                    elif not hasattr(base_link_sub_node, 'body_angle'):
                        # body_angleが保存されていない場合のデフォルト値
                        base_link_sub_node.body_angle = [0.0, 0.0, 0.0]
                    
                    # メッシュ属性
                    is_mesh_reversed_elem = node_elem.find("is_mesh_reversed")
                    if is_mesh_reversed_elem is not None:
                        base_link_sub_node.is_mesh_reversed = is_mesh_reversed_elem.text.lower() == 'true'
                    
                    mesh_original_color_elem = node_elem.find("mesh_original_color")
                    if mesh_original_color_elem is not None:
                        base_link_sub_node.mesh_original_color = [float(c) for c in mesh_original_color_elem.text.split()]

                    # リンク属性
                    blank_link_elem = node_elem.find("blank_link")
                    if blank_link_elem is not None:
                        base_link_sub_node.blank_link = blank_link_elem.text.lower() == 'true'
                    
                    massless_decoration_elem = node_elem.find("massless_decoration")
                    if massless_decoration_elem is not None:
                        base_link_sub_node.massless_decoration = massless_decoration_elem.text.lower() == 'true'
                    
                    hide_mesh_elem = node_elem.find("hide_mesh")
                    if hide_mesh_elem is not None:
                        base_link_sub_node.hide_mesh = hide_mesh_elem.text.lower() == 'true'
                    
                    # Collider情報（新形式：複数コライダー対応を優先）
                    colliders_elem = node_elem.find("colliders")
                    if colliders_elem is not None:
                        # 新形式：colliders配列
                        base_link_sub_node.colliders = []
                        for collider_elem in colliders_elem.findall("collider"):
                            collider = {}
                            
                            # タイプ
                            type_elem = collider_elem.find("type")
                            if type_elem is not None:
                                collider['type'] = type_elem.text
                            else:
                                collider['type'] = 'primitive'
                            
                            # 有効/無効
                            enabled_elem = collider_elem.find("enabled")
                            if enabled_elem is not None:
                                collider['enabled'] = enabled_elem.text.lower() == 'true'
                            else:
                                collider['enabled'] = True
                            
                            # メッシュファイル
                            mesh_elem = collider_elem.find("mesh")
                            if mesh_elem is not None and mesh_elem.text:
                                mesh_path = os.path.join(self.project_dir, mesh_elem.text)
                                if os.path.exists(mesh_path):
                                    collider['mesh'] = mesh_path
                                else:
                                    collider['mesh'] = mesh_elem.text
                            else:
                                collider['mesh'] = None
                            
                            # メッシュスケール
                            scale_elem = collider_elem.find("mesh_scale")
                            if scale_elem is not None:
                                collider['mesh_scale'] = [float(v) for v in scale_elem.text.split()]
                            else:
                                collider['mesh_scale'] = [1.0, 1.0, 1.0]
                            
                            # プリミティブデータ
                            data_elem = collider_elem.find("data")
                            if data_elem is not None:
                                collider['data'] = {}
                                
                                # タイプ
                                type_elem = data_elem.find("type")
                                if type_elem is not None:
                                    collider['data']['type'] = type_elem.text
                                
                                # ジオメトリ（文字列から辞書に変換）
                                geometry_elem = data_elem.find("geometry")
                                if geometry_elem is not None:
                                    try:
                                        collider['data']['geometry'] = eval(geometry_elem.text)
                                    except (SyntaxError, NameError):
                                        print(f"Warning: Could not parse geometry string: {geometry_elem.text}")
                                        collider['data']['geometry'] = {}
                            else:
                                collider['data'] = None
                            
                            # 位置（collider直下）
                            pos_elem = collider_elem.find("position")
                            if pos_elem is not None:
                                collider['position'] = [float(v) for v in pos_elem.text.split()]
                            else:
                                collider['position'] = [0.0, 0.0, 0.0]
                            
                            # 回転（collider直下、ラジアン）
                            rot_elem = collider_elem.find("rotation")
                            if rot_elem is not None:
                                collider['rotation'] = [float(v) for v in rot_elem.text.split()]
                            else:
                                collider['rotation'] = [0.0, 0.0, 0.0]
                            
                            base_link_sub_node.colliders.append(collider)

                    # Points (FooNodeの場合)
                    if points_elem is not None:
                        # ポイントデータの復元（空のpointをフィルタリング）
                        import re
                        for point_elem in points_elem.findall("point"):
                            point = {}
                            name_elem = point_elem.find("name")
                            if name_elem is not None:
                                point['name'] = name_elem.text
                            type_elem = point_elem.find("type")
                            if type_elem is not None:
                                point['type'] = type_elem.text
                            xyz_elem = point_elem.find("xyz")
                            if xyz_elem is not None:
                                point['xyz'] = [float(v) for v in xyz_elem.text.split()]
                            rpy_elem = point_elem.find("rpy")
                            if rpy_elem is not None:
                                point['rpy'] = [float(v) for v in rpy_elem.text.split()]
                            else:
                                # rpyが保存されていない場合のデフォルト値
                                point['rpy'] = [0.0, 0.0, 0.0]
                            angle_elem = point_elem.find("angle")
                            if angle_elem is not None:
                                point['angle'] = [float(v) for v in angle_elem.text.split()]
                            else:
                                # angleが保存されていない場合のデフォルト値
                                point['angle'] = [0.0, 0.0, 0.0]
                            
                            # 空のpoint（xyzが0.0 0.0 0.0で汎用名）をフィルタリング
                            point_name = point.get('name', '')
                            point_xyz = point.get('xyz', [0.0, 0.0, 0.0])
                            is_empty_point = (
                                len(point_xyz) == 3 and
                                all(abs(v) < 1e-9 for v in point_xyz) and
                                re.match(r'^point_\d+$', point_name, re.IGNORECASE)
                            )
                            
                            if not is_empty_point:
                                base_link_sub_node.points.append(point)
                        
                        # 必要な数のポートを追加（_add_output()を使わず、直接ポートを追加）
                        num_points = len(base_link_sub_node.points)
                        for i in range(num_points):
                            base_link_sub_node.output_count += 1
                            port_name = f'out_{base_link_sub_node.output_count}'
                            # 出力ポートを追加（ポイントは既に読み込まれているため追加しない）
                            base_link_sub_node.add_output(port_name, color=(180, 80, 0))
                            # 累積座標を追加
                            cumulative_coord = create_cumulative_coord(i)
                            base_link_sub_node.cumulative_coords.append(cumulative_coord)
                        
                        # 整合性チェック：pointsの数と実際のポート数が一致しているか確認
                        actual_port_count = len(base_link_sub_node.output_ports())
                        if actual_port_count != num_points:
                            print(f"Warning: Port count mismatch after load for {base_link_sub_node.name()}: ports={actual_port_count}, points={num_points}")
                            # pointsの数を実際のポート数に合わせる
                            if actual_port_count > num_points:
                                # ポートが多い場合は、不足分のポイントを追加
                                for i in range(num_points, actual_port_count):
                                    point_data = create_point_data(i + 1)
                                    base_link_sub_node.points.append(point_data)
                            elif actual_port_count < num_points:
                                # ポイントが多い場合は、余分なポイントを削除
                                base_link_sub_node.points = base_link_sub_node.points[:actual_port_count]
                            # output_countも更新
                            base_link_sub_node.output_count = actual_port_count
                    
                    # base_linkとbase_link_subを接続
                    try:
                        base_output_port = existing_base_link.get_output('out')
                        base_link_sub_input_port = base_link_sub_node.get_input('in')
                        if base_output_port and base_link_sub_input_port:
                            base_output_port.connect_to(base_link_sub_input_port)
                            print(f"  ✓ Connected base_link.out to base_link_sub.in")
                    except Exception as e:
                        print(f"  ✗ ERROR: Could not connect base_link to base_link_sub: {str(e)}")
                    
                    # 読み込んだbase_linkノードを削除（base_link_subにデータを移したため）
                    self.remove_node(node)
                    print(f"  ✓ Created base_link_sub and removed loaded base_link node")
                    return base_link_sub_node
            
            # STLファイル
            stl_elem = node_elem.find("stl_file")
            if stl_elem is not None and stl_elem.text:
                stl_path = os.path.join(self.project_dir, stl_elem.text)
                if os.path.exists(stl_path):
                    node.stl_file = stl_path
            
            # その他のプロパティ
            mass_elem = node_elem.find("mass")
            if mass_elem is not None:
                node.mass_value = float(mass_elem.text)
            
            volume_elem = node_elem.find("volume")
            if volume_elem is not None:
                node.volume_value = float(volume_elem.text)
            
            color_elem = node_elem.find("color")
            if color_elem is not None:
                node.node_color = [float(c) for c in color_elem.text.split()]
            
            rotation_axis_elem = node_elem.find("rotation_axis")
            if rotation_axis_elem is not None:
                node.rotation_axis = int(rotation_axis_elem.text)
            
            xml_file_elem = node_elem.find("xml_file")
            if xml_file_elem is not None and xml_file_elem.text:
                xml_path = os.path.join(self.project_dir, xml_file_elem.text)
                if os.path.exists(xml_path):
                    node.xml_file = xml_path
            
            # Inertial情報
            inertia_elem = node_elem.find("inertia")
            if inertia_elem is not None:
                node.inertia = {}
                for key in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                    key_elem = inertia_elem.find(key)
                    if key_elem is not None:
                        node.inertia[key] = float(key_elem.text)
            
            io_elem = node_elem.find("inertial_origin")
            if io_elem is not None:
                node.inertial_origin = {}
                xyz_elem = io_elem.find("xyz")
                if xyz_elem is not None:
                    node.inertial_origin['xyz'] = [float(v) for v in xyz_elem.text.split()]
                rpy_elem = io_elem.find("rpy")
                if rpy_elem is not None:
                    node.inertial_origin['rpy'] = [float(v) for v in rpy_elem.text.split()]
            
            # Visual Origin情報
            vo_elem = node_elem.find("visual_origin")
            if vo_elem is not None:
                node.visual_origin = {}
                xyz_elem = vo_elem.find("xyz")
                if xyz_elem is not None:
                    node.visual_origin['xyz'] = [float(v) for v in xyz_elem.text.split()]
                rpy_elem = vo_elem.find("rpy")
                if rpy_elem is not None:
                    node.visual_origin['rpy'] = [float(v) for v in rpy_elem.text.split()]
            elif not hasattr(node, 'visual_origin'):
                # visual_originが保存されていない場合のデフォルト値
                node.visual_origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
            
            # Joint情報
            joint_lower_elem = node_elem.find("joint_lower")
            if joint_lower_elem is not None:
                node.joint_lower = float(joint_lower_elem.text)
            
            joint_upper_elem = node_elem.find("joint_upper")
            if joint_upper_elem is not None:
                node.joint_upper = float(joint_upper_elem.text)
            
            joint_effort_elem = node_elem.find("joint_effort")
            if joint_effort_elem is not None:
                node.joint_effort = float(joint_effort_elem.text)
            
            joint_velocity_elem = node_elem.find("joint_velocity")
            if joint_velocity_elem is not None:
                node.joint_velocity = float(joint_velocity_elem.text)
            
            joint_damping_elem = node_elem.find("joint_damping")
            if joint_damping_elem is not None:
                node.joint_damping = float(joint_damping_elem.text)
            
            joint_stiffness_elem = node_elem.find("joint_stiffness")
            if joint_stiffness_elem is not None:
                node.joint_stiffness = float(joint_stiffness_elem.text)
            
            joint_margin_elem = node_elem.find("joint_margin")
            if joint_margin_elem is not None:
                node.joint_margin = float(joint_margin_elem.text)
            
            joint_armature_elem = node_elem.find("joint_armature")
            if joint_armature_elem is not None:
                node.joint_armature = float(joint_armature_elem.text)
            
            joint_frictionloss_elem = node_elem.find("joint_frictionloss")
            if joint_frictionloss_elem is not None:
                node.joint_frictionloss = float(joint_frictionloss_elem.text)
            
            # Body Angle（子ノードの初期回転角度、ラジアンで保存されている）
            body_angle_elem = node_elem.find("body_angle")
            if body_angle_elem is not None:
                node.body_angle = [float(v) for v in body_angle_elem.text.split()]
            elif not hasattr(node, 'body_angle'):
                # body_angleが保存されていない場合のデフォルト値
                node.body_angle = [0.0, 0.0, 0.0]
            
            # メッシュ属性
            is_mesh_reversed_elem = node_elem.find("is_mesh_reversed")
            if is_mesh_reversed_elem is not None:
                node.is_mesh_reversed = is_mesh_reversed_elem.text.lower() == 'true'
            
            mesh_original_color_elem = node_elem.find("mesh_original_color")
            if mesh_original_color_elem is not None:
                node.mesh_original_color = [float(c) for c in mesh_original_color_elem.text.split()]

            # リンク属性
            blank_link_elem = node_elem.find("blank_link")
            if blank_link_elem is not None:
                node.blank_link = blank_link_elem.text.lower() == 'true'
            
            massless_decoration_elem = node_elem.find("massless_decoration")
            if massless_decoration_elem is not None:
                node.massless_decoration = massless_decoration_elem.text.lower() == 'true'
            
            hide_mesh_elem = node_elem.find("hide_mesh")
            if hide_mesh_elem is not None:
                node.hide_mesh = hide_mesh_elem.text.lower() == 'true'
            
            # Collider情報（新形式：複数コライダー対応を優先）
            colliders_elem = node_elem.find("colliders")
            if colliders_elem is not None:
                # 新形式：colliders配列
                node.colliders = []
                for collider_elem in colliders_elem.findall("collider"):
                    collider = {}
                    
                    # タイプ
                    type_elem = collider_elem.find("type")
                    if type_elem is not None:
                        collider['type'] = type_elem.text
                    else:
                        collider['type'] = 'primitive'
                    
                    # 有効/無効
                    enabled_elem = collider_elem.find("enabled")
                    if enabled_elem is not None:
                        collider['enabled'] = enabled_elem.text.lower() == 'true'
                    else:
                        collider['enabled'] = True
                    
                    # メッシュファイル
                    mesh_elem = collider_elem.find("mesh")
                    if mesh_elem is not None and mesh_elem.text:
                        mesh_path = os.path.join(self.project_dir, mesh_elem.text)
                        if os.path.exists(mesh_path):
                            collider['mesh'] = mesh_path
                        else:
                            collider['mesh'] = mesh_elem.text
                    else:
                        collider['mesh'] = None
                    
                    # メッシュスケール
                    scale_elem = collider_elem.find("mesh_scale")
                    if scale_elem is not None:
                        collider['mesh_scale'] = [float(v) for v in scale_elem.text.split()]
                    else:
                        collider['mesh_scale'] = [1.0, 1.0, 1.0]
                    
                    # プリミティブデータ
                    data_elem = collider_elem.find("data")
                    if data_elem is not None:
                        collider['data'] = {}
                        
                        # タイプ
                        type_elem = data_elem.find("type")
                        if type_elem is not None:
                            collider['data']['type'] = type_elem.text
                        
                        # ジオメトリ（文字列から辞書に変換）
                        geometry_elem = data_elem.find("geometry")
                        if geometry_elem is not None:
                            try:
                                collider['data']['geometry'] = eval(geometry_elem.text)
                            except (SyntaxError, NameError):
                                print(f"Warning: Could not parse geometry string: {geometry_elem.text}")
                                collider['data']['geometry'] = {}
                    else:
                        collider['data'] = None
                    
                    # 位置（collider直下）
                    pos_elem = collider_elem.find("position")
                    if pos_elem is not None:
                        collider['position'] = [float(v) for v in pos_elem.text.split()]
                    else:
                        collider['position'] = [0.0, 0.0, 0.0]
                    
                    # 回転（collider直下、ラジアン）
                    rot_elem = collider_elem.find("rotation")
                    if rot_elem is not None:
                        collider['rotation'] = [float(v) for v in rot_elem.text.split()]
                    else:
                        collider['rotation'] = [0.0, 0.0, 0.0]
                    
                    node.colliders.append(collider)

            # Points (FooNodeの場合)
            points_elem = node_elem.find("points")
            if points_elem is not None and isinstance(node, FooNode):
                points = points_elem.findall("point")
                
                # 既存のポートとポイントをクリア（初期化時に追加されたものを削除）
                current_ports = len(node.output_ports())
                # すべての出力ポートの接続をクリアしてから削除
                for i in range(1, current_ports + 1):
                    port_name = f'out_{i}'
                    port = node.get_output(port_name)
                    if port:
                        port.clear_connections()
                
                # すべての出力ポートを削除
                while current_ports > 0:
                    node.remove_output()
                    current_ports -= 1
                
                # ポイントデータと累積座標をクリア
                node.points = []
                node.cumulative_coords = []
                node.output_count = 0
                
                # ポイントデータの復元（空のpointをフィルタリング）
                import re
                for point_index, point_elem in enumerate(points, 1):
                    point = {}
                    name_elem = point_elem.find("name")
                    if name_elem is not None:
                        point['name'] = name_elem.text
                    type_elem = point_elem.find("type")
                    if type_elem is not None:
                        point['type'] = type_elem.text
                    xyz_elem = point_elem.find("xyz")
                    if xyz_elem is not None:
                        point['xyz'] = [float(v) for v in xyz_elem.text.split()]
                    rpy_elem = point_elem.find("rpy")
                    if rpy_elem is not None:
                        point['rpy'] = [float(v) for v in rpy_elem.text.split()]
                    else:
                        # rpyが保存されていない場合のデフォルト値
                        point['rpy'] = [0.0, 0.0, 0.0]
                    angle_elem = point_elem.find("angle")
                    if angle_elem is not None:
                        point['angle'] = [float(v) for v in angle_elem.text.split()]
                    else:
                        # angleが保存されていない場合のデフォルト値
                        point['angle'] = [0.0, 0.0, 0.0]
                    
                    # 空のpoint（xyzが0.0 0.0 0.0で汎用名）をフィルタリング
                    # ただし、接続されているポートは削除しない
                    point_name = point.get('name', '')
                    point_xyz = point.get('xyz', [0.0, 0.0, 0.0])
                    is_empty_point = (
                        len(point_xyz) == 3 and
                        all(abs(v) < 1e-9 for v in point_xyz) and
                        re.match(r'^point_\d+$', point_name, re.IGNORECASE)
                    )
                    
                    # 接続情報を確認（ポート名は out_1, out_2 などの形式）
                    # ポイントのインデックスはXMLファイル内での出現順序（1から始まる）
                    port_name = f'out_{point_index}'
                    is_connected = connected_ports and port_name in connected_ports
                    
                    # 接続されているポイント、または空でないポイントは保持
                    if is_connected or not is_empty_point:
                        node.points.append(point)
                    else:
                        print(f"Filtered out empty point '{point_name}' from node '{node.name()}' (not connected, port: {port_name})")
                
                num_points = len(node.points)
                
                # 必要な数のポートを追加（_add_output()を使わず、直接ポートを追加）
                for i in range(num_points):
                    node.output_count += 1
                    port_name = f'out_{node.output_count}'
                    # 出力ポートを追加（ポイントは既に読み込まれているため追加しない）
                    node.add_output(port_name, color=(180, 80, 0))
                    # 累積座標を追加
                    cumulative_coord = create_cumulative_coord(i)
                    node.cumulative_coords.append(cumulative_coord)
                
                # 整合性チェック：pointsの数と実際のポート数が一致しているか確認
                actual_port_count = len(node.output_ports())
                if actual_port_count != num_points:
                    print(f"Warning: Port count mismatch after load for {node.name()}: ports={actual_port_count}, points={num_points}")
                    # pointsの数を実際のポート数に合わせる
                    if actual_port_count > num_points:
                        # ポートが多い場合は、不足分のポイントを追加
                        for i in range(num_points, actual_port_count):
                            point_data = create_point_data(i + 1)
                            node.points.append(point_data)
                    elif actual_port_count < num_points:
                        # ポイントが多い場合は、余分なポイントを削除
                        node.points = node.points[:actual_port_count]
                    # output_countも更新
                    node.output_count = actual_port_count
            
            # base_linkの場合、既存のbase_linkがある場合はデータを設定しない（上書きを禁止）
            if node_name == 'base_link' and existing_base_link:
                print(f"  Load: base_link data found but existing base_link exists, skipping data assignment")
                # 読み込んだbase_linkノードを削除（既存のbase_linkを保持するため）
                self.remove_node(node)
                return existing_base_link
            
            return node
            
        except Exception as e:
            print(f"Error loading node data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


    def save_project(self, file_path=None):
        """プロジェクトの保存（循環参照対策版）"""
        print("\n=== Starting Project Save ===")
        try:
            # STLビューアの状態を一時的にバックアップ
            stl_viewer_state = None
            if hasattr(self, 'stl_viewer'):
                print("Backing up STL viewer state...")
                stl_viewer_state = {
                    'actors': dict(self.stl_viewer.stl_actors),
                    'transforms': dict(self.stl_viewer.transforms)
                }
                # STLビューアの参照を一時的にクリア
                self.stl_viewer.stl_actors.clear()
                self.stl_viewer.transforms.clear()

            # ファイルパスの取得
            if not file_path:
                default_filename = f"urdf_pj_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.xml"
                default_dir = self.last_save_dir or self.meshes_dir or os.getcwd()
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    None,
                    "Save Project",
                    os.path.join(default_dir, default_filename),
                    "XML Files (*.xml)"
                )
                if not file_path:
                    print("Save cancelled by user")
                    return False

            self.project_dir = os.path.dirname(os.path.abspath(file_path))
            self.last_save_dir = self.project_dir
            print(f"Project will be saved to: {file_path}")

            # XMLツリーの作成
            print("Creating XML structure...")
            root = ET.Element("project")
            
            # ロボット名の保存
            robot_name_elem = ET.SubElement(root, "robot_name")
            robot_name_elem.text = self.robot_name
            print(f"Saving robot name: {self.robot_name}")
            
            if self.meshes_dir:
                try:
                    meshes_rel_path = os.path.relpath(self.meshes_dir, self.project_dir)
                    ET.SubElement(root, "meshes_directory").text = meshes_rel_path
                    print(f"Added meshes directory reference: {meshes_rel_path}")
                except ValueError:
                    ET.SubElement(root, "meshes_directory").text = self.meshes_dir
                    print(f"Added absolute meshes path: {self.meshes_dir}")
            
            # base_link_heightの保存（MJCFインポート時の元の高さ）
            if hasattr(self, 'base_link_height'):
                ET.SubElement(root, "base_link_height").text = str(self.base_link_height)
                print(f"Saving base_link_height: {self.base_link_height}")

            # ノード情報の保存
            print("\nSaving nodes...")
            nodes_elem = ET.SubElement(root, "nodes")
            total_nodes = len(self.all_nodes())
            
            for i, node in enumerate(self.all_nodes(), 1):
                print(f"Processing node {i}/{total_nodes}: {node.name()}")
                # 一時的にSTLビューアの参照を削除
                stl_viewer_backup = node.stl_viewer if hasattr(node, 'stl_viewer') else None
                if hasattr(node, 'stl_viewer'):
                    delattr(node, 'stl_viewer')
                
                node_elem = self._save_node_data(node, self.project_dir)
                nodes_elem.append(node_elem)
                
                # STLビューアの参照を復元
                if stl_viewer_backup is not None:
                    node.stl_viewer = stl_viewer_backup

            # 接続情報の保存
            print("\nSaving connections...")
            connections = ET.SubElement(root, "connections")
            connection_count = 0
            
            for node in self.all_nodes():
                for port in node.output_ports():
                    for connected_port in port.connected_ports():
                        conn = ET.SubElement(connections, "connection")
                        ET.SubElement(conn, "from_node").text = node.name()
                        ET.SubElement(conn, "from_port").text = port.name()
                        ET.SubElement(conn, "to_node").text = connected_port.node().name()
                        ET.SubElement(conn, "to_port").text = connected_port.name()
                        connection_count += 1
                        print(f"Added connection: {node.name()}.{port.name()} -> "
                            f"{connected_port.node().name()}.{connected_port.name()}")

            print(f"Total connections saved: {connection_count}")

            # カスタムカラーパレットの保存
            print("\nSaving custom color palette...")
            custom_colors_elem = ET.SubElement(root, "custom_colors")
            for i in range(16):
                color = QtWidgets.QColorDialog.customColor(i)
                if color.isValid():
                    color_elem = ET.SubElement(custom_colors_elem, "color")
                    color_elem.set("index", str(i))
                    color_elem.set("r", str(color.red()))
                    color_elem.set("g", str(color.green()))
                    color_elem.set("b", str(color.blue()))
                    color_elem.set("a", str(color.alpha()))
                    print(f"Saved custom color {i}: {color.name()}")
            print(f"Total custom colors saved: 16")

            # ハイライトカラーの保存
            print("\nSaving highlight color...")
            highlight_color_elem = ET.SubElement(root, "highlight_color")
            highlight_color_elem.text = self.highlight_color
            print(f"Saved highlight color: {self.highlight_color}")

            # コリジョンカラーの保存
            print("\nSaving collision color...")
            collision_color_elem = ET.SubElement(root, "collision_color")
            collision_color_elem.text = " ".join(str(v) for v in self.collision_color)
            print(f"Saved collision color: {self.collision_color}")

            # デフォルトジョイント設定の保存
            print("\nSaving default joint settings...")
            settings_elem = ET.SubElement(root, "default_joint_settings")
            ET.SubElement(settings_elem, "effort").text = str(self.default_joint_effort)
            ET.SubElement(settings_elem, "velocity").text = str(self.default_joint_velocity)
            ET.SubElement(settings_elem, "damping").text = str(self.default_joint_damping)
            ET.SubElement(settings_elem, "stiffness").text = str(self.default_joint_stiffness)
            ET.SubElement(settings_elem, "margin").text = str(self.default_margin)
            ET.SubElement(settings_elem, "armature").text = str(self.default_armature)
            ET.SubElement(settings_elem, "frictionloss").text = str(self.default_frictionloss)
            print(f"Saved default joint settings: effort={self.default_joint_effort}, "
                  f"velocity={self.default_joint_velocity}, damping={self.default_joint_damping}, "
                  f"stiffness={self.default_joint_stiffness}, margin={self.default_margin}, "
                  f"armature={self.default_armature}, frictionloss={self.default_frictionloss}")

            # ファイルの保存
            print("\nWriting to file...")
            tree = ET.ElementTree(root)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)

            # STLビューアの状態を復元
            if stl_viewer_state and hasattr(self, 'stl_viewer'):
                print("Restoring STL viewer state...")
                self.stl_viewer.stl_actors = stl_viewer_state['actors']
                self.stl_viewer.transforms = stl_viewer_state['transforms']
                self.stl_viewer.render_to_image()

            print(f"\nProject successfully saved to: {file_path}")
            
            QtWidgets.QMessageBox.information(
                None,
                "Save Complete",
                f"Project saved successfully to:\n{file_path}"
            )

            return True

        except Exception as e:
            error_msg = f"Error saving project: {str(e)}"
            print(f"\nERROR: {error_msg}")
            print("Traceback:")
            traceback.print_exc()
            
            # エラー時もSTLビューアの状態を復元
            if 'stl_viewer_state' in locals() and stl_viewer_state and hasattr(self, 'stl_viewer'):
                print("Restoring STL viewer state after error...")
                self.stl_viewer.stl_actors = stl_viewer_state['actors']
                self.stl_viewer.transforms = stl_viewer_state['transforms']
                self.stl_viewer.render_to_image()
            
            QtWidgets.QMessageBox.critical(
                None,
                "Save Error",
                error_msg
            )
            return False

    def _restore_stl_viewer_state(self, backup):
        """STLビューアの状態を復元"""
        if not backup or not hasattr(self, 'stl_viewer'):
            return
            
        print("Restoring STL viewer state...")
        try:
            self.stl_viewer.stl_actors = dict(backup['actors'])
            self.stl_viewer.transforms = dict(backup['transforms'])
            print("STL viewer state restored successfully")
        except Exception as e:
            print(f"Error restoring STL viewer state: {e}")

    def detect_meshes_directory(self):
        """meshesディレクトリの検出"""
        for node in self.all_nodes():
            if hasattr(node, 'stl_file') and node.stl_file:
                current_dir = os.path.dirname(os.path.abspath(node.stl_file))
                while current_dir and os.path.basename(current_dir).lower() != 'meshes':
                    current_dir = os.path.dirname(current_dir)
                if current_dir and os.path.basename(current_dir).lower() == 'meshes':
                    self.meshes_dir = current_dir
                    print(f"Found meshes directory: {self.meshes_dir}")
                    return

    def load_project(self, file_path=None):
        """プロジェクトの読み込み（コンソール出力版）"""
        print("\n=== Starting Project Load ===")
        try:
            if not file_path:
                file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    None,
                    "Load Project",
                    self.last_save_dir or "",
                    "XML Files (*.xml)"
                )
                
            if not file_path:
                print("Load cancelled by user")
                return False

            print(f"Loading project from: {file_path}")
            
            self.project_dir = os.path.dirname(os.path.abspath(file_path))
            self.last_save_dir = self.project_dir
            
            # XMLファイルの解析
            print("Parsing XML file...")
            tree = ET.parse(file_path)
            root = tree.getroot()

            # ロボット名の読み込み
            robot_name_elem = root.find("robot_name")
            if robot_name_elem is not None and robot_name_elem.text:
                self.robot_name = robot_name_elem.text
                # UI上の名前入力フィールドを更新
                if hasattr(self, 'name_input') and self.name_input:
                    self.name_input.setText(self.robot_name)
                print(f"Loaded robot name: {self.robot_name}")
            else:
                print("No robot name found in project file")
            
            # base_link_heightの読み込み
            base_link_height_elem = root.find("base_link_height")
            if base_link_height_elem is not None and base_link_height_elem.text:
                self.base_link_height = float(base_link_height_elem.text)
                print(f"Loaded base_link_height: {self.base_link_height}")
            else:
                # 見つからない場合はデフォルト値を使用
                if not hasattr(self, 'base_link_height'):
                    self.base_link_height = self.default_base_link_height
                print(f"Using default base_link_height: {self.base_link_height}")

            # 既存のノードをクリア（base_linkは保持）
            print("Clearing existing nodes (except base_link)...")
            existing_base_link = self.get_node_by_name('base_link')
            self.clear_graph()
            # base_linkを再作成（デフォルトのbase_linkを保持）
            if existing_base_link:
                print("Recreating default base_link after clear_graph...")
                default_base_link = self.create_node(
                    'insilico.nodes.BaseLinkNode',
                    name='base_link',
                    pos=QtCore.QPointF(50, 0)
                )
                # デフォルト値を設定
                default_base_link.mass_value = 0.0
                default_base_link.inertia = DEFAULT_INERTIA_ZERO.copy()
                default_base_link.inertial_origin = {
                    'xyz': DEFAULT_ORIGIN_ZERO['xyz'].copy(),
                    'rpy': DEFAULT_ORIGIN_ZERO['rpy'].copy()
                }
                default_base_link.stl_file = None
                default_base_link.node_color = DEFAULT_COLOR_WHITE.copy()
                default_base_link.rotation_axis = 3  # Fixed
                if hasattr(default_base_link, 'blank_link'):
                    default_base_link.blank_link = True
                print("Default base_link recreated")

            # カスタムカラーパレットの復元
            print("\nRestoring custom color palette...")
            custom_colors_elem = root.find("custom_colors")
            if custom_colors_elem is not None:
                color_count = 0
                for color_elem in custom_colors_elem.findall("color"):
                    try:
                        index = int(color_elem.get("index"))
                        r = int(color_elem.get("r"))
                        g = int(color_elem.get("g"))
                        b = int(color_elem.get("b"))
                        a = int(color_elem.get("a"))
                        color = QtGui.QColor(r, g, b, a)
                        QtWidgets.QColorDialog.setCustomColor(index, color)
                        color_count += 1
                    except Exception as e:
                        print(f"Error restoring custom color: {e}")
                print(f"Total custom colors restored: {color_count}")
            else:
                print("No custom colors found in project file")

            # ハイライトカラーの復元
            print("\nRestoring highlight color...")
            highlight_color_elem = root.find("highlight_color")
            if highlight_color_elem is not None and highlight_color_elem.text:
                self.highlight_color = highlight_color_elem.text
                print(f"Restored highlight color: {self.highlight_color}")
            else:
                print("No highlight color found in project file, using default")

            # コリジョンカラーの復元
            print("\nRestoring collision color...")
            collision_color_elem = root.find("collision_color")
            if collision_color_elem is not None and collision_color_elem.text:
                try:
                    self.collision_color = [float(v) for v in collision_color_elem.text.split()]
                    print(f"Restored collision color: {self.collision_color}")
                except (ValueError, IndexError) as e:
                    print(f"Error parsing collision color, using default: {e}")
                    self.collision_color = DEFAULT_COLLISION_COLOR.copy()
            else:
                print("No collision color found in project file, using default")
                self.collision_color = DEFAULT_COLLISION_COLOR.copy()

            # デフォルトジョイント設定の復元
            print("\nRestoring default joint settings...")
            settings_elem = root.find("default_joint_settings")
            if settings_elem is not None:
                try:
                    effort_elem = settings_elem.find("effort")
                    if effort_elem is not None and effort_elem.text:
                        self.default_joint_effort = float(effort_elem.text)

                    velocity_elem = settings_elem.find("velocity")
                    if velocity_elem is not None and velocity_elem.text:
                        self.default_joint_velocity = float(velocity_elem.text)

                    damping_elem = settings_elem.find("damping")
                    if damping_elem is not None and damping_elem.text:
                        self.default_joint_damping = float(damping_elem.text)

                    stiffness_elem = settings_elem.find("stiffness")
                    if stiffness_elem is not None and stiffness_elem.text:
                        self.default_joint_stiffness = float(stiffness_elem.text)

                    margin_elem = settings_elem.find("margin")
                    if margin_elem is not None and margin_elem.text:
                        self.default_margin = float(margin_elem.text)

                    armature_elem = settings_elem.find("armature")
                    if armature_elem is not None and armature_elem.text:
                        self.default_armature = float(armature_elem.text)

                    frictionloss_elem = settings_elem.find("frictionloss")
                    if frictionloss_elem is not None and frictionloss_elem.text:
                        self.default_frictionloss = float(frictionloss_elem.text)

                    # 後方互換性: 古いパラメータを読み込む（使用しないが、エラーを避けるため）
                    friction_elem = settings_elem.find("friction")
                    if friction_elem is not None and friction_elem.text:
                        pass  # 読み込むが使用しない
                    actuation_lag_elem = settings_elem.find("actuation_lag")
                    if actuation_lag_elem is not None and actuation_lag_elem.text:
                        pass  # 読み込むが使用しない

                    print(f"Restored default joint settings: effort={self.default_joint_effort}, "
                          f"velocity={self.default_joint_velocity}, damping={self.default_joint_damping}, "
                          f"stiffness={self.default_joint_stiffness}, margin={self.default_margin}, "
                          f"armature={self.default_armature}, frictionloss={self.default_frictionloss}")
                except (ValueError, TypeError) as e:
                    print(f"Error parsing default joint settings, using defaults: {e}")
            else:
                print("No default joint settings found in project file, using defaults")

            # meshesディレクトリの解決
            print("Resolving meshes directory...")
            meshes_dir_elem = root.find("meshes_directory")
            if meshes_dir_elem is not None and meshes_dir_elem.text:
                meshes_path = os.path.normpath(os.path.join(self.project_dir, meshes_dir_elem.text))
                if os.path.exists(meshes_path):
                    self.meshes_dir = meshes_path
                    print(f"Found meshes directory: {meshes_path}")
                else:
                    response = QtWidgets.QMessageBox.question(
                        None,
                        "Meshes Directory Not Found",
                        "The original meshes directory was not found. Would you like to select it?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    
                    if response == QtWidgets.QMessageBox.Yes:
                        self.meshes_dir = QtWidgets.QFileDialog.getExistingDirectory(
                            None,
                            "Select Meshes Directory",
                            self.project_dir
                        )
                        if self.meshes_dir:
                            print(f"Selected new meshes directory: {self.meshes_dir}")
                        else:
                            print("Meshes directory selection cancelled")

            # 接続情報を先に読み込んで、接続されているポートを記録
            print("\nPre-loading connection information...")
            connections_info = {}  # {node_name: {port_name: True, ...}, ...}
            for conn in root.findall(".//connection"):
                from_node_name = conn.find("from_node")
                from_port_name = conn.find("from_port")
                if from_node_name is not None and from_port_name is not None:
                    node_name = from_node_name.text
                    port_name = from_port_name.text
                    if node_name not in connections_info:
                        connections_info[node_name] = set()
                    connections_info[node_name].add(port_name)
            print(f"Pre-loaded connection info for {len(connections_info)} nodes")
            
            # ノードの復元
            print("\nRestoring nodes...")
            nodes_elem = root.find("nodes")
            total_nodes = len(nodes_elem.findall("node")) if nodes_elem is not None else 0
            
            # STLファイルを読み込む必要があるノードの総数を先に数える
            nodes_with_stl = 0
            for node_elem in nodes_elem.findall("node") if nodes_elem is not None else []:
                stl_elem = node_elem.find("stl_file")
                if stl_elem is not None and stl_elem.text:
                    nodes_with_stl += 1
            
            # 処理の総数 = ノード読み込み + STL読み込み
            total_operations = total_nodes + nodes_with_stl
            print(f"Total operations: {total_nodes} node loads + {nodes_with_stl} STL loads = {total_operations}")
            
            # 進捗バーの設定
            if total_operations > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                self.stl_viewer.show_progress(True)
                self.stl_viewer.progress_bar.setValue(100)  # 100%から開始
                QtWidgets.QApplication.processEvents()
            
            nodes_dict = {}
            processed_operations = 0
            
            # ノードの読み込み（接続情報を渡す）
            for i, node_elem in enumerate(nodes_elem.findall("node"), 1):
                node_name_elem = node_elem.find("name")
                node_name = node_name_elem.text if node_name_elem is not None else None
                node_connections = connections_info.get(node_name, set()) if node_name else set()
                node = self._load_node_data(node_elem, connected_ports=node_connections)
                if node:
                    nodes_dict[node.name()] = node
                
                processed_operations += 1
                # 進捗バーの更新（1操作ごとに）
                if total_operations > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                    # 残り操作数をパーセンテージで計算（100%から減らしていく）
                    remaining_percent = 100 - int((processed_operations / total_operations) * 100)
                    self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                    QtWidgets.QApplication.processEvents()

            # 接続の復元
            print("\nRestoring connections...")
            connection_count = 0
            for conn in root.findall(".//connection"):
                from_node = nodes_dict.get(conn.find("from_node").text)
                to_node = nodes_dict.get(conn.find("to_node").text)
                
                if from_node and to_node:
                    from_port = from_node.get_output(conn.find("from_port").text)
                    to_port = to_node.get_input(conn.find("to_port").text)
                    
                    if from_port and to_port:
                        self.connect_ports(from_port, to_port)
                        connection_count += 1

            # 位置の再計算とビューの更新
            print("\nRecalculating positions...")
            self.recalculate_all_positions()

            print("Updating 3D view...")
            if self.stl_viewer:
                # ビューをリセット
                self.stl_viewer.reset_view_to_fit()

                # Hide Mesh状態を全ノードに適用（recalculate_all_positions後に確実に実行）
                print("\nApplying hide_mesh states after position recalculation...")
                for node in nodes_dict.values():
                    if hasattr(node, 'hide_mesh') and node.hide_mesh:
                        if node in self.stl_viewer.stl_actors:
                            actor = self.stl_viewer.stl_actors[node]
                            actor.SetVisibility(False)
                            print(f"Applied hide_mesh: {node.name()} - mesh hidden")

                # 3Dビューを更新
                self.stl_viewer.render_to_image()

                # Collider表示を更新（表示はOFFのまま）
                print("\nApplying collider display states...")
                # コライダー表示は自動でONにしない（OFFのまま）
                # コライダー情報は復元されているが、表示はユーザーが手動でONにするまでOFFのまま
                if self.stl_viewer.collider_display_enabled:
                    # 既にONの場合は更新
                    self.stl_viewer.refresh_collider_display()
                    print("Collider display updated (already enabled)")
                else:
                    # OFFの場合は、コライダーアクターを非表示にする
                    self.stl_viewer.hide_all_colliders()
                    print("Collider display remains OFF (user must enable manually)")
                
                # 注意: 色の適用はSTL読み込み完了後に行う（STL読み込み前ではstl_actorsにノードが存在しないため）
                
                # STL読み込み完了後、すべてのノードのカラーを3Dビューに適用
                print("\nApplying colors to 3D view after project load...")
                self._apply_colors_to_all_nodes()

            # 進捗バーはSTL読み込みが完了するまで表示し続ける（外部関数で非表示にする）
            # ここでは非表示にしない

            print(f"\nProject successfully loaded from: {file_path}")
            return True

        except Exception as e:
            error_msg = f"Error loading project: {str(e)}"
            print(f"\nERROR: {error_msg}")
            print("Traceback:")
            traceback.print_exc()
            
            # エラー時にも進捗バーを非表示
            if hasattr(self, 'stl_viewer') and self.stl_viewer:
                self.stl_viewer.show_progress(False)
            
            QtWidgets.QMessageBox.critical(
                None,
                "Load Error",
                error_msg
            )
            return False


    def clear_graph(self):
        for node in self.all_nodes():
            self.remove_node(node)

    def connect_ports(self, from_port, to_port):
        """指定された2つのポートを接続"""
        if from_port and to_port:
            try:
                # 利用可能なメソッドを探して接続を試みる
                if hasattr(self, 'connect_nodes'):
                    connection = self.connect_nodes(
                        from_port.node(), from_port.name(),
                        to_port.node(), to_port.name())
                elif hasattr(self, 'add_edge'):
                    connection = self.add_edge(
                        from_port.node().id, from_port.name(),
                        to_port.node().id, to_port.name())
                elif hasattr(from_port, 'connect_to'):
                    connection = from_port.connect_to(to_port)
                else:
                    raise AttributeError("No suitable connection method found")

                if connection:
                    print(
                        f"Connected {from_port.node().name()}.{from_port.name()} to {to_port.node().name()}.{to_port.name()}")
                    return True
                else:
                    print("Failed to connect ports: Connection not established")
                    return False
            except Exception as e:
                print(f"Error connecting ports: {str(e)}")
                return False
        else:
            print("Failed to connect ports: Invalid port(s)")
            return False

    def calculate_cumulative_coordinates(self, node):
        """ノードの累積座標を計算（ルートからのパスを考慮）"""
        if isinstance(node, BaseLinkNode):
            return [0, 0, 0]  # base_linkは原点

        # 親ノードとの接続情報を取得
        input_port = node.input_ports()[0]  # 最初の入力ポート
        if not input_port.connected_ports():
            return [0, 0, 0]  # 接続されていない場合は原点

        parent_port = input_port.connected_ports()[0]
        parent_node = parent_port.node()
        
        # 親ノードの累積座標を再帰的に計算
        parent_coords = self.calculate_cumulative_coordinates(parent_node)
        
        # 接続されているポートのインデックスを取得
        port_name = parent_port.name()
        if '_' in port_name:
            port_index = int(port_name.split('_')[1]) - 1
        else:
            port_index = 0
            
        # 親ノードのポイント座標を取得
        if 0 <= port_index < len(parent_node.points):
            point_xyz = parent_node.points[port_index]['xyz']
            
            # 累積座標の計算
            return [
                parent_coords[0] + point_xyz[0],
                parent_coords[1] + point_xyz[1],
                parent_coords[2] + point_xyz[2]
            ]
        return parent_coords

    def _find_mesh_file(self, folder_path, base_name):
        """フォルダ内でベース名に対応するメッシュファイルを検索

        Args:
            folder_path: 検索対象のフォルダパス
            base_name: ファイルのベース名（拡張子なし）

        Returns:
            str: 見つかったメッシュファイルのパス、見つからない場合はNone
        """
        # 優先順位: dae > obj > stl
        extensions = ['.dae', '.obj', '.stl']

        for ext in extensions:
            mesh_file = base_name + ext
            mesh_path = os.path.join(folder_path, mesh_file)
            if os.path.exists(mesh_path):
                return mesh_path

        return None

    def _find_collider_file(self, folder_path, base_name):
        """フォルダ内でベース名に対応するコライダーファイルを検索

        Args:
            folder_path: 検索対象のフォルダパス
            base_name: ファイルのベース名（拡張子なし）

        Returns:
            tuple: (collider_path, collider_type)
                   collider_pathは見つかったファイルのパス、見つからない場合はNone
                   collider_typeは'xml'（プリミティブ）または'mesh'、見つからない場合はNone
        """
        # 1. XMLコライダーを探す（プリミティブ形状）
        collider_xml = base_name + '_collider.xml'
        collider_xml_path = os.path.join(folder_path, collider_xml)
        if os.path.exists(collider_xml_path):
            return (collider_xml_path, 'xml')

        # 2. メッシュコライダーを探す（優先順位: dae > obj > stl）
        mesh_extensions = ['.dae', '.obj', '.stl']
        for ext in mesh_extensions:
            collider_mesh = base_name + '_collider' + ext
            collider_mesh_path = os.path.join(folder_path, collider_mesh)
            if os.path.exists(collider_mesh_path):
                return (collider_mesh_path, 'mesh')

        return (None, None)

    def import_xmls_from_folder(self):
        """フォルダ内のすべてのXMLファイルを読み込む"""
        message_box = QtWidgets.QMessageBox()
        message_box.setIcon(QtWidgets.QMessageBox.Information)
        message_box.setWindowTitle("Select Directory")
        message_box.setText("Please select the meshes directory.")
        message_box.exec_()

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select meshes Directory Containing XML Files")
        
        if not folder_path:
            return

        print(f"Importing XMLs from folder: {folder_path}")

        # フォルダ名からロボット名を抽出
        try:
            # 選択されたディレクトリ名を取得
            robot_name = os.path.basename(folder_path)

            # _descriptionが末尾にある場合は削除
            if robot_name.endswith('_description'):
                robot_name = robot_name[:-12]
                print(f"Removed '_description' suffix from robot name")

            # ロボット名を更新
            self.robot_name = robot_name
            if hasattr(self, 'name_input') and self.name_input:
                self.name_input.setText(robot_name)
            print(f"Set robot name to: {robot_name}")
        except Exception as e:
            print(f"Error extracting robot name: {str(e)}")
        
        # フォルダ内のXMLファイルを検索（*_collider.xmlを除外）
        all_xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
        xml_files = [f for f in all_xml_files if not f.endswith('_collider.xml')]

        if not xml_files:
            print("No valid XML files found in the selected folder")
            print("(Note: *_collider.xml files are used as collider definitions and don't create nodes)")
            return

        # ファイルを優先順位でソート: c_*, l_*, r_*, その他
        def get_file_sort_priority(filename):
            """
            ファイル名の優先順位を返す
            c_* = 0, l_* = 1, r_* = 2, その他 = 3
            """
            lower_name = filename.lower()
            if lower_name.startswith('c_'):
                return (0, filename)
            elif lower_name.startswith('l_'):
                return (1, filename)
            elif lower_name.startswith('r_'):
                return (2, filename)
            else:
                return (3, filename)

        xml_files.sort(key=get_file_sort_priority)

        print(f"Found {len(xml_files)} XML files to import")
        if len(all_xml_files) > len(xml_files):
            print(f"(Skipped {len(all_xml_files) - len(xml_files)} collider XML files)")
        print(f"Import order: c_* → l_* → r_* → others")

        # 進捗バーの設定
        total_files = len(xml_files)
        if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
            self.stl_viewer.show_progress(True)
            self.stl_viewer.progress_bar.setValue(100)  # 100%から開始
            QtWidgets.QApplication.processEvents()

        # ノード配置用の変数
        current_group = None
        node_y_position = 0
        node_spacing = 5  # ノード間の基本間隔
        group_spacing = 30  # グループ間の追加間隔

        for file_index, xml_file in enumerate(xml_files):
            try:
                xml_path = os.path.join(folder_path, xml_file)

                # 現在のファイルのグループを判定
                file_group = get_file_sort_priority(xml_file)[0]

                # グループが変わったら間隔を追加
                if current_group is not None and file_group != current_group:
                    node_y_position += group_spacing
                    print(f"\n{'─'*60}")
                    group_names = {0: 'Center (c_*)', 1: 'Left (l_*)', 2: 'Right (r_*)', 3: 'Others'}
                    print(f"▼ Starting new group: {group_names.get(file_group, 'Others')}")
                    print(f"{'─'*60}")

                current_group = file_group

                print(f"\n{'='*60}")
                print(f"Processing: {xml_file}")

                # 新しいノードを作成（Y位置を設定）
                new_node = self.create_node(
                    'insilico.nodes.FooNode',
                    name=f'Node_{len(self.all_nodes())}',
                    pos=QtCore.QPointF(0, node_y_position)
                )

                # 次のノードのためにY位置を更新
                node_y_position += node_spacing
                
                # XMLファイルを読み込む
                tree = ET.parse(xml_path)
                root = tree.getroot()

                if root.tag != 'urdf_part':
                    print(f"Warning: Invalid XML format in {xml_file}")
                    continue

                # リンク情報の処理
                link_elem = root.find('link')
                if link_elem is not None:
                    # リンク名の設定
                    link_name = link_elem.get('name')
                    if link_name:
                        new_node.set_name(link_name)
                    else:
                        link_name = new_node.name()  # リンク名が無い場合はノード名を使用

                    # 慣性関連の処理
                    inertial_elem = link_elem.find('inertial')
                    if inertial_elem is not None:
                        # ボリュームの設定
                        volume_elem = inertial_elem.find('volume')
                        if volume_elem is not None:
                            new_node.volume_value = float(volume_elem.get('value', '0.0'))

                        # 質量の設定
                        mass_elem = inertial_elem.find('mass')
                        if mass_elem is not None:
                            new_node.mass_value = float(mass_elem.get('value', '0.0'))

                        # Inertial Originの設定
                        origin_elem = inertial_elem.find('origin')
                        if origin_elem is not None:
                            origin_xyz = origin_elem.get('xyz', '0 0 0').split()
                            origin_rpy = origin_elem.get('rpy', '0 0 0').split()
                            new_node.inertial_origin = {
                                'xyz': [float(x) for x in origin_xyz],
                                'rpy': [float(x) for x in origin_rpy]
                            }

                        # 慣性テンソルの設定
                        inertia_elem = inertial_elem.find('inertia')
                        if inertia_elem is not None:
                            new_node.inertia = {
                                'ixx': float(inertia_elem.get('ixx', '0.0')),
                                'ixy': float(inertia_elem.get('ixy', '0.0')),
                                'ixz': float(inertia_elem.get('ixz', '0.0')),
                                'iyy': float(inertia_elem.get('iyy', '0.0')),
                                'iyz': float(inertia_elem.get('iyz', '0.0')),
                                'izz': float(inertia_elem.get('izz', '0.0'))
                            }
                            
                            # === 必須ログ: どのXMLファイルからinertialを読んだか ===
                            print(f"\n[XML_INERTIAL_SOURCE] link_name={link_name}, source_xml_path={xml_path}")
                            print(f"  mass={new_node.mass_value:.9e}")
                            print(f"  origin_xyz={new_node.inertial_origin.get('xyz', [0,0,0])}")
                            print(f"  origin_rpy={new_node.inertial_origin.get('rpy', [0,0,0])}")
                            print(f"  ixx={new_node.inertia['ixx']:.9e}, ixy={new_node.inertia['ixy']:.9e}, ixz={new_node.inertia['ixz']:.9e}")
                            print(f"  iyy={new_node.inertia['iyy']:.9e}, iyz={new_node.inertia['iyz']:.9e}, izz={new_node.inertia['izz']:.9e}")
                        else:
                            print(f"\n[XML_INERTIAL_SOURCE] link_name={link_name}, source_xml_path={xml_path}")
                            print(f"  WARNING: <inertia> element not found in <inertial> - will use fallback/estimation")
                    else:
                        print(f"\n[XML_INERTIAL_SOURCE] link_name={link_name}, source_xml_path={xml_path}")
                        print(f"  WARNING: <inertial> element not found - will use fallback/estimation")

                # Center of Massの設定（link要素直下）
                center_of_mass_elem = link_elem.find('center_of_mass')
                if center_of_mass_elem is not None and center_of_mass_elem.text:
                    com_xyz = center_of_mass_elem.text.strip().split()
                    if len(com_xyz) == 3:
                        new_node.center_of_mass = [float(x) for x in com_xyz]
                        print(f"Set center of mass: {new_node.center_of_mass}")

                # 色情報の処理
                material_elem = root.find('.//material/color')
                if material_elem is not None:
                    rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                    rgba_values = [float(x) for x in rgba[:4]]  # RGBA使用
                    new_node.node_color = rgba_values
                else:
                    new_node.node_color = DEFAULT_COLOR_WHITE.copy()
                    print("Using default color: white")

                # 回転軸とjoint limitsの処理
                joint_elem = root.find('joint')
                if joint_elem is not None:
                    # ジョイントタイプの確認
                    joint_type = joint_elem.get('type', '')
                    if joint_type == 'fixed':
                        new_node.rotation_axis = 3  # Fixed
                    else:
                        # 回転軸の処理
                        axis_elem = joint_elem.find('axis')
                        if axis_elem is not None:
                            axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                            axis_values = [float(x) for x in axis_xyz]
                            if axis_values[2] == 1:      # Z軸
                                new_node.rotation_axis = 2
                            elif axis_values[1] == 1:    # Y軸
                                new_node.rotation_axis = 1
                            else:                        # X軸（デフォルト）
                                new_node.rotation_axis = 0
                            print(f"Set rotation axis: {new_node.rotation_axis} from xyz: {axis_xyz}")

                    # Joint limitsの処理
                    limit_elem = joint_elem.find('limit')
                    if limit_elem is not None:
                        # XMLからはRadian値で読み込む
                        lower_rad = float(limit_elem.get('lower', -3.14159))
                        upper_rad = float(limit_elem.get('upper', 3.14159))
                        effort = float(limit_elem.get('effort', 10.0))
                        velocity = float(limit_elem.get('velocity', 3.0))
                        damping = float(limit_elem.get('damping', DEFAULT_DAMPING_KV))
                        stiffness = float(limit_elem.get('stiffness', DEFAULT_STIFFNESS_KP))
                        margin = float(limit_elem.get('margin', DEFAULT_MARGIN))
                        armature = float(limit_elem.get('armature', DEFAULT_ARMATURE))
                        frictionloss = float(limit_elem.get('frictionloss', DEFAULT_FRICTIONLOSS))

                        # ノードにはRadian値で保存
                        new_node.joint_lower = lower_rad
                        new_node.joint_upper = upper_rad
                        new_node.joint_effort = effort
                        new_node.joint_velocity = velocity
                        new_node.joint_damping = damping
                        new_node.joint_stiffness = stiffness
                        new_node.joint_margin = margin
                        new_node.joint_armature = armature
                        new_node.joint_frictionloss = frictionloss

                    # Joint dynamicsの処理（優先的に読み込む）
                    dynamics_elem = joint_elem.find('dynamics')
                    if dynamics_elem is not None:
                        if dynamics_elem.get('damping'):
                            new_node.joint_damping = float(dynamics_elem.get('damping', DEFAULT_DAMPING_KV))
                        if dynamics_elem.get('stiffness'):
                            new_node.joint_stiffness = float(dynamics_elem.get('stiffness', DEFAULT_STIFFNESS_KP))
                        if dynamics_elem.get('margin'):
                            new_node.joint_margin = float(dynamics_elem.get('margin', DEFAULT_MARGIN))
                        if dynamics_elem.get('armature'):
                            new_node.joint_armature = float(dynamics_elem.get('armature', DEFAULT_ARMATURE))
                        if dynamics_elem.get('frictionloss'):
                            new_node.joint_frictionloss = float(dynamics_elem.get('frictionloss', DEFAULT_FRICTIONLOSS))
                else:
                    new_node.rotation_axis = 0
                    print("Using default rotation axis: X")

                # massless_decorationとhide_meshの読み込み
                massless_elem = root.find('massless_decoration')
                if massless_elem is not None:
                    try:
                        massless_value = massless_elem.text.lower() == 'true' if massless_elem.text else False
                        new_node.massless_decoration = massless_value
                        print(f"Loaded massless_decoration: {massless_value}")
                    except Exception as e:
                        print(f"Error parsing massless_decoration: {e}")

                hide_mesh_elem = root.find('hide_mesh')
                if hide_mesh_elem is not None:
                    try:
                        hide_mesh_value = hide_mesh_elem.text.lower() == 'true' if hide_mesh_elem.text else False
                        new_node.hide_mesh = hide_mesh_value
                        print(f"Loaded hide_mesh: {hide_mesh_value}")
                    except Exception as e:
                        print(f"Error parsing hide_mesh: {e}")

                # ポイントの処理
                point_elements = root.findall('point')
                num_points = len(point_elements)

                # FooNodeの場合のみポート数を調整
                if isinstance(new_node, FooNode):
                    # 現在のポート数を正しく取得
                    current_ports = len(new_node.output_ports())

                    # ポートを削除する前に、削除対象のポートの接続をすべてクリア
                    if current_ports > num_points:
                        for i in range(num_points + 1, current_ports + 1):
                            port_name = f'out_{i}'
                            port = new_node.get_output(port_name)
                            if port:
                                port.clear_connections()

                    while current_ports < num_points:
                        new_node._add_output()
                        current_ports += 1

                    while current_ports > num_points:
                        new_node.remove_output()
                        current_ports -= 1

                    # ポイントデータの更新
                    new_node.points = []
                    for point_elem in point_elements:
                        point_name = point_elem.get('name')
                        point_type = point_elem.get('type')
                        point_xyz_elem = point_elem.find('point_xyz')
                        point_angle_elem = point_elem.find('point_angle')

                        if point_xyz_elem is not None and point_xyz_elem.text:
                            xyz_values = [float(x) for x in point_xyz_elem.text.strip().split()]
                            # point_angleの読み込み
                            angle_values = [0.0, 0.0, 0.0]
                            if point_angle_elem is not None and point_angle_elem.text:
                                try:
                                    angle_values = [float(x) for x in point_angle_elem.text.strip().split()]
                                    if len(angle_values) != 3:
                                        angle_values = [0.0, 0.0, 0.0]
                                except ValueError:
                                    angle_values = [0.0, 0.0, 0.0]
                            new_node.points.append({
                                'name': point_name,
                                'type': point_type,
                                'xyz': xyz_values,
                                'angle': angle_values
                            })

                    # 累積座標の更新
                    new_node.cumulative_coords = []
                    for i in range(len(new_node.points)):
                        new_node.cumulative_coords.append(create_cumulative_coord(i))

                    # output_countを更新
                    new_node.output_count = len(new_node.points)

                # ベース名を取得（拡張子なし）
                base_name = xml_file[:-4]

                # メッシュファイルの処理（優先順位: dae > obj > stl）
                mesh_path = self._find_mesh_file(folder_path, base_name)
                if mesh_path:
                    mesh_ext = os.path.splitext(mesh_path)[1]
                    print(f"Loading mesh file: {os.path.basename(mesh_path)} {mesh_ext}")
                    new_node.stl_file = mesh_path
                    if self.stl_viewer:
                        # 全体の進捗バーを使用するため、個別の進捗バーは非表示
                        self.stl_viewer.load_stl_for_node(new_node, show_progress=False)
                        # モデルに色を適用
                        if hasattr(new_node, 'node_color'):
                            self.stl_viewer.apply_color_to_node(new_node)
                else:
                    print(f"Warning: No mesh file found for {base_name}")

                # Collider情報の読み込み（XMLファイル内の要素から）
                collider_elem = root.find('collider')
                if collider_elem is not None:
                    collider_type = collider_elem.get('type')
                    collider_file = collider_elem.get('file')

                    if collider_type == 'primitive' and collider_file:
                        # プリミティブコライダーの場合、ファイルパスから読み込む
                        collider_xml_path = os.path.join(folder_path, collider_file)

                        if os.path.exists(collider_xml_path):
                            collider_data = self.inspector_window.parse_collider_xml(collider_xml_path)
                            if collider_data:
                                # Update colliders list
                                if not hasattr(new_node, 'colliders'):
                                    new_node.colliders = []
                                new_node.colliders = [{
                                    'type': 'primitive',
                                    'enabled': True,
                                    'data': collider_data,
                                    'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                                    'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0]),
                                    'mesh': None,
                                    'mesh_scale': [1.0, 1.0, 1.0]
                                }]
                                print(f"Loaded collider XML: {collider_xml_path}")
                        else:
                            print(f"Warning: Collider XML file not found: {collider_xml_path}")

                    elif collider_type == 'mesh' and collider_file:
                        # メッシュコライダーの場合、相対パスから読み込む
                        collider_mesh_path = os.path.join(folder_path, collider_file)

                        if os.path.exists(collider_mesh_path):
                            # Update colliders list
                            if not hasattr(new_node, 'colliders'):
                                new_node.colliders = []
                            new_node.colliders = [{
                                'type': 'mesh',
                                'enabled': True,
                                'data': None,
                                'position': [0.0, 0.0, 0.0],
                                'rotation': [0.0, 0.0, 0.0],
                                'mesh': collider_mesh_path,
                                'mesh_scale': [1.0, 1.0, 1.0]
                            }]
                            print(f"Loaded collider mesh: {collider_mesh_path}")
                        else:
                            print(f"Warning: Collider mesh file not found: {collider_mesh_path}")

                # Collision mesh の処理（後方互換性のため）
                collision_mesh_elem = link_elem.find('collision_mesh') if link_elem is not None else None
                if collision_mesh_elem is not None and collision_mesh_elem.text:
                    collision_mesh_path = os.path.join(folder_path, collision_mesh_elem.text.strip())
                    if os.path.exists(collision_mesh_path):
                        # Update colliders list
                        if not hasattr(new_node, 'colliders'):
                            new_node.colliders = []
                        new_node.colliders = [{
                            'type': 'mesh',
                            'enabled': True,
                            'data': None,
                            'position': [0.0, 0.0, 0.0],
                            'rotation': [0.0, 0.0, 0.0],
                            'mesh': collision_mesh_path,
                            'mesh_scale': [1.0, 1.0, 1.0]
                        }]
                        print(f"Loaded collider mesh (legacy): {collision_mesh_path}")

                # コライダーファイルの自動検出（XMLファイル内にcollider要素がない場合）
                if not collider_elem and not collision_mesh_elem:
                    collider_path, collider_type = self._find_collider_file(folder_path, base_name)

                    if collider_path and collider_type == 'xml':
                        # XMLコライダー（プリミティブ形状）
                        print(f"Loading collider XML (auto-detected): {os.path.basename(collider_path)}")
                        collider_data = self.inspector_window.parse_collider_xml(collider_path)
                        if collider_data:
                            # Update colliders list
                            if not hasattr(new_node, 'colliders'):
                                new_node.colliders = []
                            new_node.colliders = [{
                                'type': 'primitive',
                                'enabled': True,
                                'data': collider_data,
                                'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                                'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0]),
                                'mesh': None,
                                'mesh_scale': [1.0, 1.0, 1.0]
                            }]
                            print(f"  → Primitive collider: {collider_data['type']}")
                        else:
                            print(f"  → Warning: Failed to parse collider XML")

                    elif collider_path and collider_type == 'mesh':
                        # メッシュコライダー
                        print(f"Loading collider mesh (auto-detected): {os.path.basename(collider_path)}")
                        # Update colliders list
                        if not hasattr(new_node, 'colliders'):
                            new_node.colliders = []
                        new_node.colliders = [{
                            'type': 'mesh',
                            'enabled': True,
                            'data': None,
                            'position': [0.0, 0.0, 0.0],
                            'rotation': [0.0, 0.0, 0.0],
                            'mesh': collider_path,
                            'mesh_scale': [1.0, 1.0, 1.0]
                        }]
                        print(f"  → Mesh collider assigned")

                    else:
                        # コライダーなし：ビジュアルメッシュを使用（初期値はアンチェック）
                        if mesh_path:
                            print(f"  → No dedicated collider found, will use visual mesh when enabled")
                            # Update colliders list（無効状態で）
                            if not hasattr(new_node, 'colliders'):
                                new_node.colliders = []
                            new_node.colliders = [{
                                'type': 'mesh',
                                'enabled': False,
                                'data': None,
                                'position': [0.0, 0.0, 0.0],
                                'rotation': [0.0, 0.0, 0.0],
                                'mesh': mesh_path,
                                'mesh_scale': [1.0, 1.0, 1.0]
                            }]
                        else:
                            print(f"  → No collider available")

                # ファイル名が_decまたは_dec[数字]で終わる場合の自動設定
                import re
                if re.search(r'_dec\d*$', base_name):
                    new_node.massless_decoration = True
                    # collidersリストのenabledをFalseに設定
                    if hasattr(new_node, 'colliders') and new_node.colliders:
                        for collider in new_node.colliders:
                            collider['enabled'] = False
                    new_node.rotation_axis = 3  # Fixed
                    print(f"  → Auto-configured: Massless Decoration=ON, Collider=OFF, Rotation Axis=Fixed (filename ends with '_dec')")

                print(f"✓ Successfully imported: {xml_file}")
                
                # === 必須ログ: 読み込み完了後のノードの慣性値を確認 ===
                final_link_name = new_node.name()  # ノード名を取得
                if 'arm_lower' in final_link_name.lower():
                    print(f"\n[XML_IMPORT_COMPLETE] link_name={final_link_name}, source_xml_path={xml_path}")
                    if hasattr(new_node, 'inertia') and new_node.inertia:
                        print(f"  Final node.inertia: ixx={new_node.inertia.get('ixx', 0):.9e}, ixy={new_node.inertia.get('ixy', 0):.9e}, ixz={new_node.inertia.get('ixz', 0):.9e}")
                        print(f"                      iyy={new_node.inertia.get('iyy', 0):.9e}, iyz={new_node.inertia.get('iyz', 0):.9e}, izz={new_node.inertia.get('izz', 0):.9e}")
                    else:
                        print(f"  WARNING: node.inertia is not set!")

                # 進捗バーの更新（1ファイル処理ごとに）
                if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                    # 処理済みファイル数を計算
                    processed_files = file_index + 1
                    # 残りファイル数をパーセンテージで計算（100%から減らしていく）
                    remaining_percent = 100 - int((processed_files / total_files) * 100)
                    self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                    QtWidgets.QApplication.processEvents()

            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                traceback.print_exc()
                # エラーが発生しても進捗を更新
                if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                    processed_files = file_index + 1
                    remaining_percent = 100 - int((processed_files / total_files) * 100)
                    self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                    QtWidgets.QApplication.processEvents()
                continue

        # すべてのノードの色を接続状態に応じて更新
        self.update_all_node_colors()

        # 進捗バーを非表示
        if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
            self.stl_viewer.progress_bar.setValue(0)
            QtWidgets.QApplication.processEvents()
            from PySide6.QtCore import QTimer
            QTimer.singleShot(200, lambda: self.stl_viewer.show_progress(False))

        print("\nImport process completed")

    def recalculate_all_positions(self):
        """すべてのノードの位置を再計算"""
        print("Starting position recalculation for all nodes...")
        
        try:
            # base_linkノードを探す
            base_node = None
            for node in self.all_nodes():
                if isinstance(node, BaseLinkNode):
                    base_node = node
                    break
            
            if not base_node:
                print("Error: Base link node not found")
                return
            
            # 処理するノードの総数を先に数える（再帰的に）
            def count_nodes(node, visited):
                if node in visited:
                    return 0
                visited.add(node)
                count = 1
                for port in node.output_ports():
                    for connected_port in port.connected_ports():
                        child_node = connected_port.node()
                        count += count_nodes(child_node, visited)
                return count
            
            total_nodes = count_nodes(base_node, set())
            print(f"Total nodes to process: {total_nodes}")
            
            # 進捗バーの設定
            if total_nodes > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                # 進捗バーが既に表示されている場合はそのまま使用、そうでない場合は表示
                if not self.stl_viewer.progress_bar.isVisible():
                    self.stl_viewer.show_progress(True)
                self.stl_viewer.progress_bar.setValue(100)  # 100%から開始
                QtWidgets.QApplication.processEvents()
            
            # 再帰的に位置を更新（進捗バーを更新しながら）
            visited_nodes = set()
            processed_count = [0]  # リストでラップして参照渡し
            
            print(f"Starting from base node: {base_node.name()}")
            self._recalculate_node_positions(base_node, [0, 0, 0], visited_nodes, None, total_nodes, processed_count)

            # 閉ループ拘束を適用
            print("\n=== Enforcing Closed-Loop Constraints ===")
            self._enforce_closed_loop_constraints()
            print("=== Closed-Loop Constraints Enforced ===\n")

            # STLビューアの更新
            if hasattr(self, 'stl_viewer'):
                self.stl_viewer.render_to_image()
            
            # 進捗バーを非表示（処理が完了）
            if total_nodes > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                self.stl_viewer.progress_bar.setValue(0)
                QtWidgets.QApplication.processEvents()
                from PySide6.QtCore import QTimer
                QTimer.singleShot(200, lambda: self.stl_viewer.show_progress(False))

            print("Position recalculation completed")

        except Exception as e:
            print(f"Error during position recalculation: {str(e)}")
            traceback.print_exc()
            # エラー時にも進捗バーを非表示
            if hasattr(self, 'stl_viewer') and self.stl_viewer:
                self.stl_viewer.show_progress(False)

    def _recalculate_node_positions(self, node, parent_coords, visited, parent_transform=None, total_nodes=0, processed_count=None):
        """再帰的にノードの位置と回転を計算"""
        if node in visited:
            return
        visited.add(node)
        
        # 進捗バーの更新
        if processed_count is not None and total_nodes > 0:
            processed_count[0] += 1
            if hasattr(self, 'stl_viewer') and self.stl_viewer:
                remaining_percent = 100 - int((processed_count[0] / total_nodes) * 100)
                self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                QtWidgets.QApplication.processEvents()

        print(f"\nProcessing node: {node.name()}")
        print(f"Parent coordinates: {parent_coords}")

        # デバッグ：ノードタイプと出力ポート接続状態を確認
        if isinstance(node, ClosedLoopJointNode):
            print(f"  ⚠ DEBUG: This is a ClosedLoopJointNode")
            print(f"  ⚠ DEBUG: Output ports: {[p.name() for p in node.output_ports()]}")
            for port in node.output_ports():
                connected = port.connected_ports()
                if connected:
                    print(f"  ⚠ DEBUG: Port {port.name()} is connected to: {[f'{p.node().name()}.{p.name()}' for p in connected]}")
                else:
                    print(f"  ⚠ DEBUG: Port {port.name()} has no connections")

            # 閉リンクノードはツリー構造に含まれないため、子ノードの処理をスキップ
            print(f"  ⚠ DEBUG: Skipping child node processing for closed-loop node")
            return

        # 現在のノードがまだ3Dビューに読み込まれていない場合は、先に読み込む
        if hasattr(self, 'stl_viewer'):
            if node not in self.stl_viewer.stl_actors or node not in self.stl_viewer.transforms:
                # メッシュファイルが設定されている場合のみ読み込み
                if hasattr(node, 'stl_file') and node.stl_file:
                    print(f"  ℹ Node {node.name()} not loaded yet, loading now...")
                    self.stl_viewer.load_stl_for_node(node, show_progress=False)
                # メッシュが無いノードの場合は変換のみ作成
                elif node not in self.stl_viewer.transforms:
                    import vtk
                    self.stl_viewer.transforms[node] = vtk.vtkTransform()
                    self.stl_viewer.transforms[node].Identity()
                    print(f"  ℹ Created transform for meshless node {node.name()}")

        # 現在のノードのHide Meshがオンの場合は3Dビューで非表示にする
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            if hasattr(self, 'stl_viewer') and node in self.stl_viewer.stl_actors:
                actor = self.stl_viewer.stl_actors[node]
                actor.SetVisibility(False)
                print(f"Applied hide_mesh: {node.name()} - mesh hidden in 3D view")

        try:
            # デバッグ: ノードのポートとポイント情報を出力
            if node.name() == 'base_link_sub':
                print(f"\n*** DEBUG base_link_sub ***")
                print(f"  Output ports: {[p.name() for p in node.output_ports()]}")
                print(f"  Points count: {len(node.points) if hasattr(node, 'points') else 0}")
                if hasattr(node, 'points'):
                    for i, pt in enumerate(node.points):
                        print(f"  points[{i}]: name={pt.get('name')}, xyz={pt.get('xyz')}")
                print(f"*** END DEBUG ***\n")

            # 出力ポートを処理
            for port_idx, output_port in enumerate(node.output_ports()):
                for connected_port in output_port.connected_ports():
                    child_node = connected_port.node()

                    # ポイントデータの確認
                    # ポート名からインデックスを計算（out_1 -> 0, out_2 -> 1, etc.）
                    port_name = output_port.name() if hasattr(output_port, 'name') else 'unknown'
                    point_index = port_idx  # デフォルトはenumインデックス
                    if port_name.startswith('out_'):
                        try:
                            port_num = int(port_name.split('_')[1])
                            point_index = port_num - 1  # out_1 -> 0, out_2 -> 1, etc.
                        except (ValueError, IndexError):
                            pass
                    elif port_name == 'out':
                        point_index = 0  # BaseLinkNodeの'out'は0

                    print(f"\n=== Processing connection: {node.name()}[{port_name}] (enum_idx={port_idx}, point_index={point_index}) -> {child_node.name()} ===")
                    print(f"  has_points: {hasattr(node, 'points')}, points_count: {len(node.points) if hasattr(node, 'points') else 0}")
                    if hasattr(node, 'points') and point_index < len(node.points):
                        point_data = node.points[point_index]
                        print(f"  point_data: {point_data}")
                        point_xyz = point_data.get('xyz', [0, 0, 0])
                        point_rpy = point_data.get('rpy', [0, 0, 0])
                        point_angle = point_data.get('angle', [0.0, 0.0, 0.0])  # radians

                        # 新しい位置を計算（簡易版 - 実際には累積変換が必要）
                        new_position = [
                            parent_coords[0] + point_xyz[0],
                            parent_coords[1] + point_xyz[1],
                            parent_coords[2] + point_xyz[2]
                        ]

                        print(f"\n=== Transform Debug for {child_node.name()} ===")
                        print(f"  Point XYZ: {point_xyz}")
                        print(f"  Point RPY (rad): {point_rpy}")
                        print(f"  Point Angle (rad): {point_angle}")
                        print(f"  Point Angle zero?: {not any(a != 0.0 for a in point_angle)}")

                        # 累積変換行列を作成
                        import vtk
                        import math

                        # ジョイントのローカル変換を作成（親のローカル座標系での変換）
                        joint_transform = vtk.vtkTransform()
                        joint_transform.Identity()

                        # URDF/SDF の同次変換行列: [R(rpy)  xyz]
                        #                          [  0      1  ]
                        # 点の変換: p' = R * p + xyz (まず回転、次に平行移動)
                        # VTK は post-multiply なので、後で追加した変換が先に適用される
                        # したがって、Translate を先に、Rotate を後に追加する

                        # 1. 平行移動を追加（後で適用される）
                        joint_transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])

                        # 2. RPY 回転を追加（先に適用される）
                        if point_rpy and len(point_rpy) == 3:
                            roll_deg = math.degrees(point_rpy[0])
                            pitch_deg = math.degrees(point_rpy[1])
                            yaw_deg = math.degrees(point_rpy[2])
                            # 固定軸回転: Rz(yaw) * Ry(pitch) * Rx(roll)
                            # VTK で右から乗算されるので、Z, Y, X の順に追加
                            joint_transform.RotateZ(yaw_deg)
                            joint_transform.RotateY(pitch_deg)
                            joint_transform.RotateX(roll_deg)
                            print(f"  Applied RPY rotation: Roll={roll_deg}, Pitch={pitch_deg}, Yaw={yaw_deg} degrees")

                        # 3. point_angle を joint_transform に追加（親のローカル座標系での body orientation）
                        # MJCF の場合、point_angle には body の quat から計算された orientation が入ってる
                        if point_angle and any(a != 0.0 for a in point_angle):
                            point_angle_deg = [math.degrees(a) for a in point_angle]
                            joint_transform.RotateZ(point_angle_deg[2])  # Z軸回転
                            joint_transform.RotateY(point_angle_deg[1])  # Y軸回転
                            joint_transform.RotateX(point_angle_deg[0])  # X軸回転
                            print(f"  Applied point_angle to joint_transform: X={point_angle_deg[0]}, Y={point_angle_deg[1]}, Z={point_angle_deg[2]} degrees")

                        # 親の変換とジョイント変換を合成
                        child_transform = vtk.vtkTransform()
                        if parent_transform is not None:
                            child_transform.Concatenate(parent_transform)
                        child_transform.Concatenate(joint_transform)

                        # 子ノードのbody_angleを取得して累積（radianからdegreeに変換してVTKへ渡す）
                        child_body_angle = getattr(child_node, 'body_angle', [0.0, 0.0, 0.0])
                        print(f"  Child body_angle (rad): {child_body_angle}")
                        print(f"  Child body_angle zero?: {not any(a != 0.0 for a in child_body_angle)}")
                        if child_body_angle and any(a != 0.0 for a in child_body_angle):
                            child_body_angle_deg = [math.degrees(a) for a in child_body_angle]
                            child_transform.RotateZ(child_body_angle_deg[2])  # Z軸回転
                            child_transform.RotateY(child_body_angle_deg[1])  # Y軸回転
                            child_transform.RotateX(child_body_angle_deg[0])  # X軸回転
                            print(f"  ✓ Applied child body_angle: X={child_body_angle_deg[0]}, Y={child_body_angle_deg[1]}, Z={child_body_angle_deg[2]} degrees")
                        print("=== End Transform Debug ===\n")

                        # 計算した変換を子ノードに適用
                        # ノードがまだ3Dビューに読み込まれていない場合は、先に読み込む
                        if child_node not in self.stl_viewer.stl_actors or child_node not in self.stl_viewer.transforms:
                            # メッシュファイルが設定されている場合のみ読み込み
                            if hasattr(child_node, 'stl_file') and child_node.stl_file:
                                print(f"  ℹ Node {child_node.name()} not loaded yet, loading now...")
                                self.stl_viewer.load_stl_for_node(child_node, show_progress=False)
                        
                        # 再度確認して変換を適用
                        if child_node in self.stl_viewer.stl_actors and child_node in self.stl_viewer.transforms:
                            # 新しいトランスフォームを作成してコピー（DeepCopyの代わりに新規作成）
                            new_transform = vtk.vtkTransform()
                            new_transform.DeepCopy(child_transform)
                            self.stl_viewer.transforms[child_node] = new_transform
                            self.stl_viewer.stl_actors[child_node].SetUserTransform(new_transform)
                            print(f"  ✓ Applied transform to 3D actor for {child_node.name()}")
                            # コライダーアクターのtransformも更新
                            self.stl_viewer.update_collider_transform(child_node)
                        else:
                            # メッシュファイルが無い場合（base_linkなど）は警告を表示しない
                            if hasattr(child_node, 'stl_file') and child_node.stl_file:
                                print(f"  ✗ WARNING: Cannot apply transform to {child_node.name()} even after loading")
                            # BaseLinkNodeやメッシュを持たないノードは変換のみ保存
                            # 新しいトランスフォームを作成してコピー
                            new_transform = vtk.vtkTransform()
                            new_transform.DeepCopy(child_transform)
                            self.stl_viewer.transforms[child_node] = new_transform

                        # Hide Meshがオンの場合は3Dビューで非表示にする
                        if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                            if child_node in self.stl_viewer.stl_actors:
                                actor = self.stl_viewer.stl_actors[child_node]
                                actor.SetVisibility(False)
                                print(f"Applied hide_mesh: {child_node.name()} - mesh hidden in 3D view")

                        # 子ノードの累積座標を更新
                        if hasattr(child_node, 'cumulative_coords'):
                            for coord in child_node.cumulative_coords:
                                # 辞書形式かリスト形式かをチェックして適切に処理
                                if isinstance(coord, dict):
                                    coord['xyz'] = new_position.copy()
                                elif isinstance(coord, list):
                                    # リスト形式の場合は辞書形式に変換
                                    coord_idx = child_node.cumulative_coords.index(coord)
                                    child_node.cumulative_coords[coord_idx] = {
                                        'point_index': coord_idx,
                                        'xyz': new_position.copy()
                                    }

                        # 再帰的に子ノードを処理（累積変換を渡す）
                        self._recalculate_node_positions(child_node, new_position, visited, child_transform, total_nodes, processed_count)
                    else:
                        print(f"Warning: No point data found for port {port_name} (point_index={point_index}) in node {node.name()}")

        except Exception as e:
            print(f"Error processing node {node.name()}: {str(e)}")
            traceback.print_exc()

    def _enforce_closed_loop_constraints(self):
        """閉ループ拘束を適用して、閉ループジョイントで接続されたリンクの位置を修正"""
        if not hasattr(self, 'stl_viewer') or not self.stl_viewer:
            print("STL viewer not available, skipping closed-loop constraint enforcement")
            return

        import vtk
        import math
        import numpy as np

        # すべてのノードをチェックして閉ループジョイントノードを見つける
        closed_loop_nodes = []
        all_nodes_dict = {}  # ノード名からノードオブジェクトへのマッピング

        for node in self.all_nodes():
            all_nodes_dict[node.name()] = node
            if isinstance(node, ClosedLoopJointNode):
                closed_loop_nodes.append(node)

        if not closed_loop_nodes:
            print("No closed-loop joints found")
            return

        print(f"Found {len(closed_loop_nodes)} closed-loop joint(s)")

        for cl_node in closed_loop_nodes:
            joint_name = cl_node.joint_name
            parent_link_name = cl_node.parent_link
            child_link_name = cl_node.child_link
            origin_xyz = cl_node.origin_xyz
            origin_rpy = cl_node.origin_rpy

            print(f"\nProcessing closed-loop joint: {joint_name}")
            print(f"  Parent link: {parent_link_name}")
            print(f"  Child link: {child_link_name}")
            print(f"  Joint origin XYZ: {origin_xyz}")
            print(f"  Joint origin RPY (rad): {origin_rpy}")

            # 親リンクと子リンクのノードを取得
            if parent_link_name not in all_nodes_dict:
                print(f"  ✗ WARNING: Parent link '{parent_link_name}' not found in nodes")
                continue
            if child_link_name not in all_nodes_dict:
                print(f"  ✗ WARNING: Child link '{child_link_name}' not found in nodes")
                continue

            parent_node = all_nodes_dict[parent_link_name]
            child_node = all_nodes_dict[child_link_name]

            # 親ノードの変換を取得
            if parent_node not in self.stl_viewer.transforms:
                print(f"  ✗ WARNING: Parent node '{parent_link_name}' has no transform")
                continue
            if child_node not in self.stl_viewer.transforms:
                print(f"  ✗ WARNING: Child node '{child_link_name}' has no transform")
                continue

            parent_transform = self.stl_viewer.transforms[parent_node]

            # ジョイント変換を作成
            joint_transform = vtk.vtkTransform()
            joint_transform.Identity()

            # 平行移動を追加
            joint_transform.Translate(origin_xyz[0], origin_xyz[1], origin_xyz[2])

            # RPY回転を追加
            if origin_rpy and len(origin_rpy) == 3:
                roll_deg = math.degrees(origin_rpy[0])
                pitch_deg = math.degrees(origin_rpy[1])
                yaw_deg = math.degrees(origin_rpy[2])
                joint_transform.RotateZ(yaw_deg)
                joint_transform.RotateY(pitch_deg)
                joint_transform.RotateX(roll_deg)

            # 目標となる子ノードの変換を計算: target = parent @ joint
            target_child_transform = vtk.vtkTransform()
            target_child_transform.Concatenate(parent_transform)
            target_child_transform.Concatenate(joint_transform)

            # 現在の子ノードの位置と目標位置を取得
            current_child_transform = self.stl_viewer.transforms[child_node]

            current_pos = current_child_transform.GetPosition()
            target_pos = target_child_transform.GetPosition()

            distance = math.sqrt(
                (target_pos[0] - current_pos[0])**2 +
                (target_pos[1] - current_pos[1])**2 +
                (target_pos[2] - current_pos[2])**2
            )

            print(f"  Current child position: [{current_pos[0]:.6f}, {current_pos[1]:.6f}, {current_pos[2]:.6f}]")
            print(f"  Target child position:  [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
            print(f"  Distance: {distance:.6f} meters")

            # 子ノードとそのサブツリーを目標位置に移動
            # 補正変換を計算: correction = target @ inv(current)
            correction_transform = vtk.vtkTransform()
            correction_transform.Concatenate(target_child_transform)

            inverse_current = vtk.vtkTransform()
            inverse_current.DeepCopy(current_child_transform)
            inverse_current.Inverse()
            correction_transform.Concatenate(inverse_current)

            # 子ノードとそのすべての子孫に補正を適用
            print(f"  Applying correction to child node and its descendants...")
            self._apply_transform_correction_to_subtree(child_node, correction_transform, set())

            print(f"  ✓ Closed-loop constraint applied for {joint_name}")

        print(f"\nTotal {len(closed_loop_nodes)} closed-loop constraint(s) enforced")

    def _apply_transform_correction_to_subtree(self, node, correction_transform, visited):
        """ノードとそのサブツリーに変換補正を適用"""
        if node in visited:
            return
        visited.add(node)

        # このノードの変換を補正
        if node in self.stl_viewer.transforms and node in self.stl_viewer.stl_actors:
            import vtk
            current_transform = self.stl_viewer.transforms[node]

            # 新しい変換 = correction @ current
            new_transform = vtk.vtkTransform()
            new_transform.Concatenate(correction_transform)
            new_transform.Concatenate(current_transform)

            # 変換を適用
            self.stl_viewer.transforms[node].DeepCopy(new_transform)
            self.stl_viewer.stl_actors[node].SetUserTransform(self.stl_viewer.transforms[node])

            # コライダーも更新
            self.stl_viewer.update_collider_transform(node)

            print(f"    ✓ Applied correction to {node.name()}")

        # 子ノードに再帰的に適用
        for output_port in node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()
                # 閉ループノードはスキップ
                if not isinstance(child_node, ClosedLoopJointNode):
                    self._apply_transform_correction_to_subtree(child_node, correction_transform, visited)

    def build_r_from_l(self):
        """左系統（l_）のノードから右系統（r_）のノードを自動生成"""
        print("Building right side (r_) from left side (l_)...")

        try:
            # 左系統のノードを収集
            l_nodes = {}
            for node in self.all_nodes():
                node_name = node.name()
                if node_name.startswith('l_'):
                    l_nodes[node_name] = node
                    print(f"Found left node: {node_name}")

            if not l_nodes:
                print("No left side nodes (l_) found")
                return

            # 既存の右系統ノードを収集（l_ノードに対応するr_ノードを全て収集）
            existing_r_nodes = {}
            # 既存のr_ノードのコライダー設定を保存（削除前に保存）
            existing_r_collider_settings = {}
            # 既存のr_ノードのメッシュファイル設定を保存（削除前に保存）
            existing_r_mesh_settings = {}
            for node in self.all_nodes():
                node_name = node.name()
                if node_name.startswith('r_'):
                    # r_をl_に置換して、対応するl_ノードが存在するかチェック
                    corresponding_l_name = node_name.replace('r_', 'l_', 1)
                    if corresponding_l_name in l_nodes:
                        existing_r_nodes[node_name] = node
                        print(f"Found existing right node: {node_name} (corresponds to {corresponding_l_name})")
                        
                        # コライダー設定を保存（r_用のコライダーを優先するため）
                        collider_settings = {}
                        import copy
                        # collidersリストを保存
                        if hasattr(node, 'colliders') and node.colliders:
                            collider_settings['colliders'] = copy.deepcopy(node.colliders)

                        if collider_settings:
                            existing_r_collider_settings[corresponding_l_name] = collider_settings
                            print(f"  Saved collider settings for {node_name}")
                            if 'colliders' in collider_settings:
                                print(f"    Saved {len(collider_settings['colliders'])} collider(s) in new format")
                        
                        # メッシュファイル設定を保存（既存のメッシュを保持するため）
                        mesh_settings = {}
                        if hasattr(node, 'stl_file') and node.stl_file:
                            mesh_settings['stl_file'] = node.stl_file
                            print(f"  Saved existing mesh file for {node_name}: {node.stl_file}")
                        if hasattr(node, 'mesh_scale'):
                            mesh_settings['mesh_scale'] = node.mesh_scale.copy() if hasattr(node.mesh_scale, 'copy') else node.mesh_scale
                        if hasattr(node, 'visual_origin') and node.visual_origin:
                            import copy
                            mesh_settings['visual_origin'] = copy.deepcopy(node.visual_origin)
                        if hasattr(node, 'is_mesh_reversed'):
                            mesh_settings['is_mesh_reversed'] = node.is_mesh_reversed
                            print(f"  Saved is_mesh_reversed flag: {node.is_mesh_reversed}")
                        
                        if mesh_settings:
                            existing_r_mesh_settings[corresponding_l_name] = mesh_settings
                            print(f"  Saved mesh settings for {node_name}")

            # 既存の右系統ノードを削除
            for r_node in existing_r_nodes.values():
                print(f"Removing existing node: {r_node.name()}")
                self.remove_node(r_node)
            
            # 削除後に再度全ノードを確認して、残っているr_ノードがないかチェック
            remaining_r_nodes = []
            for node in self.all_nodes():
                node_name = node.name()
                if node_name.startswith('r_'):
                    corresponding_l_name = node_name.replace('r_', 'l_', 1)
                    if corresponding_l_name in l_nodes:
                        remaining_r_nodes.append(node)
            
            if remaining_r_nodes:
                print(f"Warning: Found {len(remaining_r_nodes)} remaining r_ nodes after deletion, removing them...")
                for r_node in remaining_r_nodes:
                    print(f"Removing remaining node: {r_node.name()}")
                    self.remove_node(r_node)

            # 新規作成するr_ノードの配置位置を計算
            # 既存のすべてのノードの最右端を取得
            max_x = -float('inf')
            min_x = float('inf')
            for node in self.all_nodes():
                pos = node.pos()
                x = pos.x() if hasattr(pos, 'x') else pos[0]
                max_x = max(max_x, x)
                min_x = min(min_x, x)

            # l_ノードの範囲を取得
            l_min_x = float('inf')
            l_max_x = -float('inf')
            for l_node in l_nodes.values():
                pos = l_node.pos()
                x = pos.x() if hasattr(pos, 'x') else pos[0]
                l_min_x = min(l_min_x, x)
                l_max_x = max(l_max_x, x)

            # r_ノードを右側に配置するためのオフセットを計算
            # 既存ノードの右端から200ピクセル離す
            x_offset = max_x - l_min_x + 200

            print(f"Positioning r_ nodes with X offset: {x_offset}")

            # 左系統ノードから右系統ノードを作成
            l_to_r_mapping = {}
            for l_name, l_node in l_nodes.items():
                r_name = l_name.replace('l_', 'r_', 1)

                print(f"\nCreating {r_name} from {l_name}")
                # ノードタイプを取得（__identifier__を使用）
                if hasattr(l_node, '__identifier__'):
                    node_type = l_node.__identifier__ + '.' + type(l_node).__name__
                else:
                    node_type = type(l_node).__module__ + '.' + type(l_node).__name__

                print(f"  Node type: {node_type}")

                # グラフビュー上の位置を計算（l_ノードからの相対位置を保持し、X方向にオフセット）
                l_pos = l_node.pos()
                # pos()がリストかQPointFか判定
                if isinstance(l_pos, list):
                    # l_ノードからの相対位置を計算してオフセットを適用
                    r_pos = QtCore.QPointF(l_pos[0] + x_offset, l_pos[1])
                else:
                    # l_ノードからの相対位置を計算してオフセットを適用
                    r_pos = QtCore.QPointF(l_pos.x() + x_offset, l_pos.y())

                print(f"  Position: ({r_pos.x()}, {r_pos.y()})")

                # 新しいノードを作成
                r_node = self.create_node(node_type, name=r_name, pos=r_pos)

                l_to_r_mapping[l_node] = r_node

                # プロパティをコピー
                # r_のメッシュファイルが見つかったかどうかのフラグ
                r_mesh_found = False
                
                # 既存のr_ノードにメッシュが設定されている場合は、それを保持する
                if l_name in existing_r_mesh_settings and 'stl_file' in existing_r_mesh_settings[l_name]:
                    # 既存のメッシュファイルを保持
                    existing_mesh_file = existing_r_mesh_settings[l_name]['stl_file']
                    r_node.stl_file = existing_mesh_file
                    print(f"  Keeping existing mesh file for {r_name}: {existing_mesh_file}")
                    
                    # 既存のmesh_scaleとvisual_originも保持
                    if 'mesh_scale' in existing_r_mesh_settings[l_name]:
                        r_node.mesh_scale = existing_r_mesh_settings[l_name]['mesh_scale']
                        print(f"  Keeping existing mesh_scale: {r_node.mesh_scale}")
                    if 'visual_origin' in existing_r_mesh_settings[l_name]:
                        r_node.visual_origin = existing_r_mesh_settings[l_name]['visual_origin']
                        print(f"  Keeping existing visual_origin: {r_node.visual_origin}")
                    if 'is_mesh_reversed' in existing_r_mesh_settings[l_name]:
                        r_node.is_mesh_reversed = existing_r_mesh_settings[l_name]['is_mesh_reversed']
                        print(f"  Keeping existing is_mesh_reversed flag: {r_node.is_mesh_reversed}")
                    else:
                        # is_mesh_reversedフラグが保存されていない場合は、mesh_scaleとvisual_originから再計算
                        r_node.is_mesh_reversed = is_mesh_reversed_check(
                            r_node.visual_origin if hasattr(r_node, 'visual_origin') and r_node.visual_origin else {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                            r_node.mesh_scale if hasattr(r_node, 'mesh_scale') else [1.0, 1.0, 1.0]
                        )
                        print(f"  Calculated is_mesh_reversed flag: {r_node.is_mesh_reversed}")
                    
                    # メッシュファイルを3Dビューアに読み込む（既存のメッシュを表示）
                    if hasattr(self, 'stl_viewer') and self.stl_viewer:
                        try:
                            self.stl_viewer.load_stl_for_node(r_node)
                            print(f"  Loaded existing mesh for {r_name}")
                        except Exception as e:
                            print(f"  Warning: Could not load existing mesh for {r_name}: {str(e)}")
                elif hasattr(l_node, 'stl_file') and l_node.stl_file:
                    # 既存のメッシュがない場合、r_の該当メッシュを探す
                    stl_file = l_node.stl_file
                    
                    # まず、r_の該当メッシュファイルを探す
                    if stl_file and 'l_' in os.path.basename(stl_file):
                        # l_をr_に置換してr_のメッシュファイルパスを生成
                        r_stl_file = stl_file.replace('/l_', '/r_').replace('\\l_', '\\r_')
                        # ファイル名自体にl_が含まれる場合も置換
                        dirname = os.path.dirname(r_stl_file)
                        basename = os.path.basename(r_stl_file)
                        basename = basename.replace('l_', 'r_', 1)
                        r_stl_file = os.path.join(dirname, basename)
                        
                        # r_のメッシュファイルが存在するか確認
                        if os.path.exists(r_stl_file):
                            r_node.stl_file = r_stl_file
                            r_mesh_found = True
                            print(f"  Found r_ mesh file: {r_stl_file}")
                            
                            # メッシュファイルを3Dビューアに読み込む
                            if hasattr(self, 'stl_viewer') and self.stl_viewer:
                                try:
                                    self.stl_viewer.load_stl_for_node(r_node)
                                    print(f"  Loaded r_ mesh for {r_name}")
                                except Exception as e:
                                    print(f"  Warning: Could not load r_ mesh for {r_name}: {str(e)}")
                    
                    # r_のメッシュファイルが見つからない場合のみ、l_のメッシュをミラーリングして使用
                    if not r_mesh_found:
                        # l_のメッシュをそのまま使用（ミラーリングはmesh_scaleで処理）
                        r_node.stl_file = stl_file
                        print(f"  r_ mesh not found, using l_ mesh with mirroring: {stl_file}")
                        
                        # メッシュファイルを3Dビューアに読み込む
                        if hasattr(self, 'stl_viewer') and self.stl_viewer:
                            try:
                                self.stl_viewer.load_stl_for_node(r_node)
                                print(f"  Loaded l_ mesh (mirrored) for {r_name}")
                            except Exception as e:
                                print(f"  Warning: Could not load l_ mesh for {r_name}: {str(e)}")

                # 物理プロパティをコピー
                if hasattr(l_node, 'volume_value'):
                    r_node.volume_value = l_node.volume_value
                if hasattr(l_node, 'mass_value'):
                    r_node.mass_value = l_node.mass_value

                # 慣性プロパティのミラーリング
                # 優先順位: URDF由来の慣性 > メッシュから再計算 > フォールバック変換
                
                # URDF由来の慣性が存在する場合は、それを優先してミラーリング
                has_urdf_inertia = (hasattr(l_node, 'inertia') and 
                                   l_node.inertia and 
                                   any(abs(v) > 1e-12 for v in l_node.inertia.values() if isinstance(v, (int, float))))
                
                if has_urdf_inertia:
                    print(f"  Using URDF-derived inertia (priority method)")
                    print(f"  [BUILD_R_FROM_L] Creating {r_name} from {l_name}")
                    print(f"    Source l_node.inertia: ixx={l_node.inertia.get('ixx', 0):.9e}, ixy={l_node.inertia.get('ixy', 0):.9e}, ixz={l_node.inertia.get('ixz', 0):.9e}")
                    print(f"                          iyy={l_node.inertia.get('iyy', 0):.9e}, iyz={l_node.inertia.get('iyz', 0):.9e}, izz={l_node.inertia.get('izz', 0):.9e}")
                    # 新しい共通関数を使用してミラーリング（URDF由来の値を保持）
                    mirrored_inertia = mirror_inertia_tensor_left_right(l_node.inertia)
                    if mirrored_inertia:
                        r_node.inertia = mirrored_inertia
                        print(f"  ✓ Mirrored URDF inertia tensor (negated ixy, iyz)")
                        print(f"    Original: ixx={l_node.inertia.get('ixx', 0):.9e}, ixy={l_node.inertia.get('ixy', 0):.9e}")
                        print(f"    Mirrored: ixx={mirrored_inertia.get('ixx', 0):.9e}, ixy={mirrored_inertia.get('ixy', 0):.9e}")
                        print(f"  [BUILD_R_FROM_L] Set r_node.inertia for {r_name}: ixx={mirrored_inertia.get('ixx', 0):.9e}, ixy={mirrored_inertia.get('ixy', 0):.9e}, ixz={mirrored_inertia.get('ixz', 0):.9e}")
                        print(f"                                          iyy={mirrored_inertia.get('iyy', 0):.9e}, iyz={mirrored_inertia.get('iyz', 0):.9e}, izz={mirrored_inertia.get('izz', 0):.9e}")
                    
                    # Center of Massのミラーリング
                    if hasattr(l_node, 'inertial_origin') and l_node.inertial_origin:
                        if not hasattr(r_node, 'inertial_origin'):
                            r_node.inertial_origin = {}
                        else:
                            r_node.inertial_origin = l_node.inertial_origin.copy()
                        
                        if 'xyz' in r_node.inertial_origin:
                            original_xyz = r_node.inertial_origin['xyz']
                            mirrored_xyz = mirror_center_of_mass_left_right(original_xyz)
                            r_node.inertial_origin['xyz'] = mirrored_xyz
                            print(f"  ✓ Mirrored COM: Y={mirrored_xyz[1]:.6f} (original: {original_xyz[1]:.6f})")
                        if 'rpy' not in r_node.inertial_origin and 'rpy' in l_node.inertial_origin:
                            r_node.inertial_origin['rpy'] = l_node.inertial_origin['rpy'].copy()
                
                # 方法2: メッシュファイルから再計算（URDF由来の慣性がない場合のみ）
                use_mesh_recalculation = False
                if not has_urdf_inertia:
                    if hasattr(l_node, 'stl_file') and l_node.stl_file and os.path.exists(l_node.stl_file):
                        if hasattr(l_node, 'mass_value') and l_node.mass_value > 0:
                            print(f"  Attempting to recalculate mirrored properties from mesh (URDF inertia not available)...")
                            print(f"  ⚠ FALLBACK_INERTIA_USED: Calculating from mesh for {r_name}")
                            mirrored_props = calculate_mirrored_physical_properties_from_mesh(
                                l_node.stl_file, l_node.mass_value
                            )
                            if mirrored_props is not None:
                                # メッシュから計算成功
                                r_node.volume_value = mirrored_props['volume']
                                r_node.mass_value = mirrored_props['mass']
                                r_node.inertia = mirrored_props['inertia']
                                if not hasattr(r_node, 'inertial_origin'):
                                    r_node.inertial_origin = {}
                                r_node.inertial_origin['xyz'] = mirrored_props['center_of_mass']
                                if 'rpy' not in r_node.inertial_origin:
                                    r_node.inertial_origin['rpy'] = [0.0, 0.0, 0.0]
                                use_mesh_recalculation = True
                                print(f"  ✓ Recalculated from mirrored mesh (fallback)")
                                print(f"    COM: {mirrored_props['center_of_mass']}")
                                print(f"    FALLBACK_INERTIA_USED: ixx={mirrored_props['inertia'].get('ixx', 0):.9e}")

                # 方法3: 既存のプロパティを変換（最後のフォールバック）
                if not has_urdf_inertia and not use_mesh_recalculation:
                    print(f"  Using property transformation (last fallback method)")
                    print(f"  ⚠ FALLBACK_INERTIA_USED: Using property transformation for {r_name}")
                    # 新しい共通関数を使用してミラーリング
                    if hasattr(l_node, 'inertia') and l_node.inertia:
                        mirrored_inertia = mirror_inertia_tensor_left_right(l_node.inertia)
                        if mirrored_inertia:
                            r_node.inertia = mirrored_inertia
                            print(f"  ✓ Mirrored inertia tensor (negated ixy, iyz)")
                            print(f"    FALLBACK_INERTIA_USED: ixx={mirrored_inertia.get('ixx', 0):.9e}")
                    
                    # Center of Massのミラーリング
                    if hasattr(l_node, 'inertial_origin') and l_node.inertial_origin:
                        if not hasattr(r_node, 'inertial_origin'):
                            r_node.inertial_origin = {}
                        else:
                            r_node.inertial_origin = l_node.inertial_origin.copy()
                        
                        if 'xyz' in r_node.inertial_origin:
                            original_xyz = r_node.inertial_origin['xyz']
                            mirrored_xyz = mirror_center_of_mass_left_right(original_xyz)
                            r_node.inertial_origin['xyz'] = mirrored_xyz
                            print(f"  ✓ Mirrored COM: Y={mirrored_xyz[1]:.6f} (original: {original_xyz[1]:.6f})")
                        if 'rpy' not in r_node.inertial_origin and 'rpy' in l_node.inertial_origin:
                            r_node.inertial_origin['rpy'] = l_node.inertial_origin['rpy'].copy()
                if hasattr(l_node, 'node_color'):
                    r_node.node_color = l_node.node_color
                if hasattr(l_node, 'rotation_axis'):
                    r_node.rotation_axis = l_node.rotation_axis

                # Body Angle (Angle offset X, Y, Z) のコピー
                if hasattr(l_node, 'body_angle'):
                    r_node.body_angle = l_node.body_angle.copy()
                    print(f"  Copied body_angle: {r_node.body_angle}")

                # Mesh scale と Visual origin の処理
                # 既存のr_ノードにメッシュが設定されている場合は既に処理済み
                # r_のメッシュファイルが見つかった場合もミラーリング不要
                if l_name not in existing_r_mesh_settings and not r_mesh_found:
                    # r_のメッシュファイルが見つからず、l_のメッシュをミラーリングして使用する場合
                    # Mesh scale のコピー（Y軸は反転）
                    if hasattr(l_node, 'mesh_scale'):
                        r_node.mesh_scale = [l_node.mesh_scale[0], -l_node.mesh_scale[1], l_node.mesh_scale[2]]
                        print(f"  Copied mesh_scale with Y-axis mirrored: {l_node.mesh_scale} -> {r_node.mesh_scale}")

                    # Visual origin のコピー（Y座標は反転）
                    if hasattr(l_node, 'visual_origin') and l_node.visual_origin:
                        r_node.visual_origin = {}
                        if 'xyz' in l_node.visual_origin:
                            xyz = l_node.visual_origin['xyz']
                            r_node.visual_origin['xyz'] = [xyz[0], -xyz[1], xyz[2]]
                            print(f"  Copied visual_origin xyz with Y mirrored: {xyz} -> {r_node.visual_origin['xyz']}")
                        if 'rpy' in l_node.visual_origin:
                            r_node.visual_origin['rpy'] = l_node.visual_origin['rpy'].copy()
                            print(f"  Copied visual_origin rpy: {r_node.visual_origin['rpy']}")
                elif r_mesh_found:
                    # r_のメッシュファイルが見つかった場合、mesh_scaleとvisual_originをそのままコピー（ミラーリング不要）
                    if hasattr(l_node, 'mesh_scale'):
                        r_node.mesh_scale = l_node.mesh_scale.copy() if hasattr(l_node.mesh_scale, 'copy') else l_node.mesh_scale
                        print(f"  Copied mesh_scale (no mirroring): {r_node.mesh_scale}")
                    if hasattr(l_node, 'visual_origin') and l_node.visual_origin:
                        import copy
                        r_node.visual_origin = copy.deepcopy(l_node.visual_origin)
                        print(f"  Copied visual_origin (no mirroring): {r_node.visual_origin}")
                
                # メッシュ反転判定（ミラーリング後のmesh_scaleとvisual_originから判定）
                r_node.is_mesh_reversed = is_mesh_reversed_check(
                    r_node.visual_origin if hasattr(r_node, 'visual_origin') and r_node.visual_origin else {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                    r_node.mesh_scale if hasattr(r_node, 'mesh_scale') else [1.0, 1.0, 1.0]
                )
                if r_node.is_mesh_reversed:
                    print(f"  Set is_mesh_reversed flag to True for {r_name} (for MJCF export)")

                # Joint limits: swap and negate for Roll(0) or Yaw(2) axes
                if hasattr(l_node, 'joint_lower') and hasattr(l_node, 'joint_upper'):
                    rotation_axis = getattr(l_node, 'rotation_axis', 1)  # Default to Pitch if not set
                    if rotation_axis == 0 or rotation_axis == 2:  # Roll or Yaw
                        # Swap lower and upper, and negate for left-right symmetry
                        # Example: lower=-10, upper=190 -> lower=-190, upper=10
                        r_node.joint_lower = -l_node.joint_upper
                        r_node.joint_upper = -l_node.joint_lower
                        print(f"  Swapped and negated joint limits for {['Roll', 'Pitch', 'Yaw'][rotation_axis]} axis: {l_node.joint_lower:.3f},{l_node.joint_upper:.3f} -> {r_node.joint_lower:.3f},{r_node.joint_upper:.3f}")
                    else:  # Pitch or other
                        r_node.joint_lower = l_node.joint_lower
                        r_node.joint_upper = l_node.joint_upper
                elif hasattr(l_node, 'joint_lower'):
                    r_node.joint_lower = l_node.joint_lower
                elif hasattr(l_node, 'joint_upper'):
                    r_node.joint_upper = l_node.joint_upper
                if hasattr(l_node, 'joint_effort'):
                    r_node.joint_effort = l_node.joint_effort
                if hasattr(l_node, 'joint_velocity'):
                    r_node.joint_velocity = l_node.joint_velocity
                if hasattr(l_node, 'joint_damping'):
                    r_node.joint_damping = l_node.joint_damping
                if hasattr(l_node, 'joint_stiffness'):
                    r_node.joint_stiffness = l_node.joint_stiffness
                if hasattr(l_node, 'joint_margin'):
                    r_node.joint_margin = l_node.joint_margin
                if hasattr(l_node, 'joint_armature'):
                    r_node.joint_armature = l_node.joint_armature
                if hasattr(l_node, 'joint_frictionloss'):
                    r_node.joint_frictionloss = l_node.joint_frictionloss
                if hasattr(l_node, 'massless_decoration'):
                    r_node.massless_decoration = l_node.massless_decoration
                if hasattr(l_node, 'hide_mesh'):
                    r_node.hide_mesh = l_node.hide_mesh

                # コライダー設定を適用（既存のr_用コライダー設定を優先）
                if l_name in existing_r_collider_settings:
                    # 既存のr_用コライダー設定を優先して適用
                    collider_settings = existing_r_collider_settings[l_name]
                    print(f"  Applying existing r_ collider settings for {r_name}")
                    import copy

                    # 新形式: collidersリストを優先して復元
                    # 有効なコライダーがあるかチェック（type: Noneは除外）
                    has_valid_colliders = (
                        'colliders' in collider_settings and collider_settings['colliders'] and
                        any(c.get('type') is not None for c in collider_settings['colliders'])
                    )

                    if has_valid_colliders:
                        r_node.colliders = copy.deepcopy(collider_settings['colliders'])
                        print(f"    Restored {len(r_node.colliders)} collider(s) from saved settings")
                else:
                    # 既存のr_用コライダー設定がない場合、l_のコライダー設定をコピー（l_をr_に置換）
                    import copy

                    # 新形式: collidersリストをl_からコピー（メッシュパスのl_をr_に変換）
                    # 有効なコライダーがあるかチェック（type: Noneは除外）
                    l_has_valid_colliders = (
                        hasattr(l_node, 'colliders') and l_node.colliders and
                        any(c.get('type') is not None for c in l_node.colliders)
                    )

                    if l_has_valid_colliders:
                        r_node.colliders = copy.deepcopy(l_node.colliders)
                        # メッシュパス内のl_をr_に変換
                        for collider in r_node.colliders:
                            if collider.get('mesh') and 'l_' in collider['mesh']:
                                original_mesh = collider['mesh']
                                collider['mesh'] = original_mesh.replace('l_', 'r_', 1)
                                print(f"    Collider mesh path converted: {original_mesh} -> {collider['mesh']}")
                        print(f"  Copied {len(r_node.colliders)} collider(s) from l_ node")

                    # Colliderファイルの自動検出と適用
                    r_collider_found = False

                    # r_用のSTLファイルに対応する _collider.xml を探す
                    if hasattr(r_node, 'stl_file') and r_node.stl_file:
                        r_stl_path = r_node.stl_file
                        if os.path.exists(r_stl_path):
                            mesh_dir = os.path.dirname(r_stl_path)
                            mesh_basename = os.path.splitext(os.path.basename(r_stl_path))[0]
                            r_collider_xml_path = os.path.join(mesh_dir, f"{mesh_basename}_collider.xml")

                            if os.path.exists(r_collider_xml_path):
                                print(f"  Found r_ collider XML: {r_collider_xml_path}")
                                # Collider XMLを解析して適用
                                if hasattr(self, 'inspector_window') and self.inspector_window:
                                    collider_data = self.inspector_window.parse_collider_xml(r_collider_xml_path)
                                    if collider_data:
                                        # Update colliders list
                                        r_node.colliders = [{
                                            'type': 'primitive',
                                            'enabled': True,
                                            'data': collider_data,
                                            'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                                            'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0]),
                                            'mesh': None,
                                            'mesh_scale': [1.0, 1.0, 1.0]
                                        }]
                                        r_collider_found = True
                                        print(f"  ✓ Attached r_ collider XML: {os.path.basename(r_collider_xml_path)}")
                                        print(f"    Type: {collider_data.get('type', 'unknown')}")

                # ポイントをコピー（Y座標を反転してミラーリング）
                if hasattr(l_node, 'points') and hasattr(r_node, 'points'):
                    # 必要な出力ポート数を計算（複数の要素から最大値を取る）
                    l_port_count = len(l_node.output_ports())
                    l_points_count = len(l_node.points) if hasattr(l_node, 'points') else 0

                    # 実際に接続が使用している最大ポートインデックス + 1
                    max_used_port = -1
                    for port_idx, output_port in enumerate(l_node.output_ports()):
                        if output_port.connected_ports():
                            max_used_port = port_idx
                    required_port_count = max_used_port + 1 if max_used_port >= 0 else 0

                    # 最大値を使用（XML、現在のポート数、実際の接続で使用されているポート数の最大）
                    target_port_count = max(l_port_count, l_points_count, required_port_count)

                    print(f"  Port count - Current: {l_port_count}, Points: {l_points_count}, Required: {required_port_count}, Target: {target_port_count}")

                    # ポートが足りない場合は追加
                    while len(r_node.output_ports()) < target_port_count:
                        if hasattr(r_node, '_add_output'):
                            r_node._add_output()
                            print(f"  Added output port to {r_name} (now {len(r_node.output_ports())} ports)")

                    # ポートが多すぎる場合は削除
                    while len(r_node.output_ports()) > target_port_count:
                        if hasattr(r_node, 'remove_output'):
                            # 接続をクリアしてから削除
                            if r_node.output_ports():
                                last_port = r_node.output_ports()[-1]
                                last_port.clear_connections()
                            r_node.remove_output()
                            print(f"  Removed output port from {r_name} (now {len(r_node.output_ports())} ports)")

                    r_node.points = []
                    for point in l_node.points:
                        r_point = point.copy()
                        # Y座標を反転
                        if 'xyz' in r_point:
                            xyz = r_point['xyz']
                            r_point['xyz'] = [xyz[0], -xyz[1], xyz[2]]
                        r_node.points.append(r_point)

                    # cumulative_coordsも更新
                    if hasattr(r_node, 'cumulative_coords'):
                        r_node.cumulative_coords = []
                        for i in range(len(r_node.points)):
                            r_node.cumulative_coords.append({'point_index': i, 'xyz': [0, 0, 0]})

                print(f"  Created {r_name} successfully")

            # 接続をミラーリング
            print("\nMirroring connections...")
            connection_count = 0
            failed_connections = []

            for l_node, r_node in l_to_r_mapping.items():
                # 出力ポートの接続をミラーリング
                for port_idx, output_port in enumerate(l_node.output_ports()):
                    for connected_port in output_port.connected_ports():
                        connected_node = connected_port.node()
                        connected_node_name = connected_node.name()

                        # 接続先に対応するr_ノードを探す
                        r_connected_node = None

                        # 1. 接続先が左系統ノード (l_) の場合
                        if connected_node in l_to_r_mapping:
                            r_connected_node = l_to_r_mapping[connected_node]
                            print(f"  Found in mapping: {connected_node_name} -> {r_connected_node.name()}")

                        # 2. 接続先が右系統ノード (r_) の場合、対応するl_ノードを探してそのr_版を使用
                        elif connected_node_name.startswith('r_'):
                            # r_をl_に変換して探す（末尾のスペースや数字も考慮）
                            import re
                            # 末尾のスペースと数字を削除
                            base_name = re.sub(r'\s+\d+$', '', connected_node_name)
                            l_version_name = base_name.replace('r_', 'l_', 1)

                            # 完全一致で探す
                            for l_n, r_n in l_to_r_mapping.items():
                                if l_n.name() == l_version_name:
                                    r_connected_node = r_n
                                    print(f"  Found r_ node {connected_node_name} -> using r_ version: {r_n.name()}")
                                    break

                            # 見つからない場合、元の名前でも試す
                            if r_connected_node is None:
                                l_version_name_original = connected_node_name.replace('r_', 'l_', 1)
                                for l_n, r_n in l_to_r_mapping.items():
                                    if l_n.name() == l_version_name_original:
                                        r_connected_node = r_n
                                        print(f"  Found r_ node {connected_node_name} (exact match) -> using r_ version: {r_n.name()}")
                                        break

                        # 3. 見つからない場合、全ノードから名前で直接探す
                        if r_connected_node is None:
                            # l_をr_に変換した名前で探す
                            target_name = connected_node_name.replace('l_', 'r_', 1) if 'l_' in connected_node_name else 'r_' + connected_node_name
                            for node in self.all_nodes():
                                if node.name() == target_name:
                                    r_connected_node = node
                                    print(f"  Found by name search: {target_name}")
                                    break

                        if r_connected_node:
                            # 対応するポートを取得
                            if port_idx < len(r_node.output_ports()):
                                r_output_port = r_node.output_ports()[port_idx]

                                # 入力ポートが存在するか確認
                                if not r_connected_node.input_ports():
                                    print(f"  Warning: {r_connected_node.name()} has no input ports")
                                    failed_connections.append(f"{r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()} (no input port)")
                                    continue

                                r_input_port = r_connected_node.input_ports()[0]

                                # 既に接続されているかチェック
                                if r_input_port in r_output_port.connected_ports():
                                    print(f"  Already connected: {r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()}.{r_input_port.name()}")
                                    connection_count += 1
                                    continue

                                # 接続を試行
                                try:
                                    print(f"  Connecting {r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()}.{r_input_port.name()}")
                                    r_output_port.connect_to(r_input_port)
                                    connection_count += 1
                                    print(f"    ✓ Successfully connected")
                                except Exception as e:
                                    error_msg = f"{r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()}.{r_input_port.name()}: {str(e)}"
                                    failed_connections.append(error_msg)
                                    print(f"    ✗ Failed to connect: {str(e)}")
                            else:
                                error_msg = f"{r_node.name()} port {port_idx} out of range (has {len(r_node.output_ports())} ports)"
                                failed_connections.append(error_msg)
                                print(f"  Warning: {error_msg}")
                        else:
                            # 対応するr_ノードが見つからなかった
                            error_msg = f"{r_node.name()}.out_{port_idx+1} -> {connected_node_name}: No corresponding r_ node found"
                            failed_connections.append(error_msg)
                            print(f"  Warning: {error_msg}")

            print(f"\nConnection summary: {connection_count} connections established")
            if failed_connections:
                print(f"Failed connections ({len(failed_connections)}):")
                for fc in failed_connections:
                    print(f"  - {fc}")

            # 親ノードに接続されていない先頭のr_ノードを、対応するl_ノードの親ノードに接続
            print("\nConnecting orphaned r_ nodes to parent nodes...")
            orphan_connection_count = 0
            for l_node, r_node in l_to_r_mapping.items():
                # r_ノードが親ノードに接続されていないかチェック
                r_input_ports = r_node.input_ports()
                is_orphaned = True
                if r_input_ports:
                    for input_port in r_input_ports:
                        if input_port.connected_ports():
                            is_orphaned = False
                            break
                
                if is_orphaned:
                    print(f"  Found orphaned r_ node: {r_node.name()}")
                    
                    # 対応するl_ノードの親ノードを探す
                    l_input_ports = l_node.input_ports()
                    l_parent_node = None
                    l_parent_port_idx = None
                    l_connected_point_idx = None
                    l_connected_point_xyz = None
                    
                    if l_input_ports:
                        for input_port in l_input_ports:
                            connected_ports = input_port.connected_ports()
                            if connected_ports:
                                l_parent_port = connected_ports[0]
                                l_parent_node = l_parent_port.node()
                                # 親ノードのどのポートに接続されているか確認
                                parent_output_ports = l_parent_node.output_ports()
                                for port_idx, output_port in enumerate(parent_output_ports):
                                    if output_port == l_parent_port:
                                        l_parent_port_idx = port_idx
                                        # このポートに対応するポイントを探す
                                        if hasattr(l_parent_node, 'points') and l_parent_node.points:
                                            if port_idx < len(l_parent_node.points):
                                                l_connected_point_idx = port_idx
                                                l_connected_point_xyz = l_parent_node.points[port_idx].get('xyz', [0, 0, 0])
                                                break
                                        break
                                break
                    
                    if l_parent_node and l_parent_node.name().startswith('l_'):
                        # l_ノードの親ノードがl_系統の場合、対応するr_ノードを探す
                        l_parent_name = l_parent_node.name()
                        r_parent_name = l_parent_name.replace('l_', 'r_', 1)
                        r_parent_node = None
                        for node in self.all_nodes():
                            if node.name() == r_parent_name:
                                r_parent_node = node
                                break
                        
                        if r_parent_node:
                            print(f"    Found r_ parent node: {r_parent_name}")
                            
                            # 親ノードのpointsから、x,zは同じでyが正負逆転しているポイントを探す
                            if hasattr(r_parent_node, 'points') and r_parent_node.points and l_connected_point_xyz:
                                target_point_idx = None
                                l_x, l_y, l_z = l_connected_point_xyz
                                
                                for point_idx, point in enumerate(r_parent_node.points):
                                    point_xyz = point.get('xyz', [0, 0, 0])
                                    r_x, r_y, r_z = point_xyz
                                    
                                    # x, zは同じで、yが正負逆転しているかチェック（許容誤差1e-6）
                                    if (abs(r_x - l_x) < 1e-6 and 
                                        abs(r_z - l_z) < 1e-6 and 
                                        abs(r_y + l_y) < 1e-6):
                                        target_point_idx = point_idx
                                        print(f"    Found matching point at index {point_idx}: xyz=({r_x:.6f}, {r_y:.6f}, {r_z:.6f}) (l_ point: ({l_x:.6f}, {l_y:.6f}, {l_z:.6f}))")
                                        break
                                
                                if target_point_idx is not None:
                                    # 対応する出力ポートを取得
                                    if target_point_idx < len(r_parent_node.output_ports()):
                                        r_parent_output_port = r_parent_node.output_ports()[target_point_idx]
                                        
                                        # r_ノードの入力ポートを取得
                                        if r_node.input_ports():
                                            r_input_port = r_node.input_ports()[0]
                                            
                                            # 既に接続されているかチェック
                                            if r_input_port in r_parent_output_port.connected_ports():
                                                print(f"    Already connected: {r_parent_name}.out_{target_point_idx+1} -> {r_node.name()}")
                                                orphan_connection_count += 1
                                            else:
                                                # 接続を試行
                                                try:
                                                    print(f"    Connecting {r_parent_name}.out_{target_point_idx+1} -> {r_node.name()}")
                                                    r_parent_output_port.connect_to(r_input_port)
                                                    orphan_connection_count += 1
                                                    print(f"      ✓ Successfully connected orphaned r_ node")
                                                except Exception as e:
                                                    print(f"      ✗ Failed to connect: {str(e)}")
                                        else:
                                            print(f"    Warning: {r_node.name()} has no input ports")
                                    else:
                                        print(f"    Warning: Parent node {r_parent_name} has no output port at index {target_point_idx}")
                                else:
                                    print(f"    No matching point found (x,z same, y negated) for l_ point ({l_x:.6f}, {l_y:.6f}, {l_z:.6f})")
                            else:
                                print(f"    Warning: Parent node {r_parent_name} has no points or l_ point info not available")
                        else:
                            print(f"    Warning: Could not find r_ parent node: {r_parent_name}")
                    elif l_parent_node:
                        # l_ノードの親ノードがl_系統でない場合（base_linkなど）、そのまま接続
                        print(f"    l_ parent node is not l_ system: {l_parent_node.name()}")
                        
                        # 親ノードのpointsから、x,zは同じでyが正負逆転しているポイントを探す
                        if hasattr(l_parent_node, 'points') and l_parent_node.points and l_connected_point_xyz:
                            target_point_idx = None
                            l_x, l_y, l_z = l_connected_point_xyz
                            
                            for point_idx, point in enumerate(l_parent_node.points):
                                point_xyz = point.get('xyz', [0, 0, 0])
                                p_x, p_y, p_z = point_xyz
                                
                                # x, zは同じで、yが正負逆転しているかチェック（許容誤差1e-6）
                                if (abs(p_x - l_x) < 1e-6 and 
                                    abs(p_z - l_z) < 1e-6 and 
                                    abs(p_y + l_y) < 1e-6):
                                    target_point_idx = point_idx
                                    print(f"    Found matching point at index {point_idx}: xyz=({p_x:.6f}, {p_y:.6f}, {p_z:.6f}) (l_ point: ({l_x:.6f}, {l_y:.6f}, {l_z:.6f}))")
                                    break
                            
                            if target_point_idx is not None:
                                # 対応する出力ポートを取得
                                if target_point_idx < len(l_parent_node.output_ports()):
                                    parent_output_port = l_parent_node.output_ports()[target_point_idx]
                                    
                                    # r_ノードの入力ポートを取得
                                    if r_node.input_ports():
                                        r_input_port = r_node.input_ports()[0]
                                        
                                        # 既に接続されているかチェック
                                        if r_input_port in parent_output_port.connected_ports():
                                            print(f"    Already connected: {l_parent_node.name()}.out_{target_point_idx+1} -> {r_node.name()}")
                                            orphan_connection_count += 1
                                        else:
                                            # 接続を試行
                                            try:
                                                print(f"    Connecting {l_parent_node.name()}.out_{target_point_idx+1} -> {r_node.name()}")
                                                parent_output_port.connect_to(r_input_port)
                                                orphan_connection_count += 1
                                                print(f"      ✓ Successfully connected orphaned r_ node")
                                            except Exception as e:
                                                print(f"      ✗ Failed to connect: {str(e)}")
                                else:
                                    print(f"    Warning: Parent node {l_parent_node.name()} has no output port at index {target_point_idx}")
                            else:
                                print(f"    No matching point found (x,z same, y negated) for l_ point ({l_x:.6f}, {l_y:.6f}, {l_z:.6f})")
                        else:
                            print(f"    Warning: Parent node {l_parent_node.name()} has no points or l_ point info not available")
                    else:
                        print(f"    l_ node {l_node.name()} has no parent node")
            
            if orphan_connection_count > 0:
                print(f"  Connected {orphan_connection_count} orphaned r_ nodes to parent nodes")

            # r_ノードをl_ノードの相対位置関係を保ってレイアウト
            print("\nRearranging r_ nodes with mirrored layout from l_ nodes...")
            if l_to_r_mapping:
                # l_ノードの境界ボックスを計算
                l_positions = {}
                l_min_x = float('inf')
                l_max_x = -float('inf')
                l_min_y = float('inf')
                l_max_y = -float('inf')

                for l_node in l_to_r_mapping.keys():
                    l_pos = l_node.pos()
                    x = l_pos.x() if hasattr(l_pos, 'x') else l_pos[0]
                    y = l_pos.y() if hasattr(l_pos, 'y') else l_pos[1]
                    l_positions[l_node] = (x, y)
                    l_min_x = min(l_min_x, x)
                    l_max_x = max(l_max_x, x)
                    l_min_y = min(l_min_y, y)
                    l_max_y = max(l_max_y, y)

                print(f"  l_ bounding box: X({l_min_x:.1f}, {l_max_x:.1f}), Y({l_min_y:.1f}, {l_max_y:.1f})")

                # 既存の全ノードの境界ボックスを計算（r_ノードを除く）
                all_min_x = float('inf')
                all_max_x = -float('inf')
                all_min_y = float('inf')
                all_max_y = -float('inf')

                for node in self.all_nodes():
                    # r_ノードは除外（これから配置する）
                    if node not in l_to_r_mapping.values():
                        node_pos = node.pos()
                        x = node_pos.x() if hasattr(node_pos, 'x') else node_pos[0]
                        y = node_pos.y() if hasattr(node_pos, 'y') else node_pos[1]
                        all_min_x = min(all_min_x, x)
                        all_max_x = max(all_max_x, x)
                        all_min_y = min(all_min_y, y)
                        all_max_y = max(all_max_y, y)

                # r_ノードの配置開始位置を計算（既存ノードと重ならないように）
                # 既存ノードの右端から300ピクセル離す
                r_base_x = all_max_x + 300
                r_base_y = l_min_y  # l_系と同じY開始位置

                print(f"  r_ base position: ({r_base_x:.1f}, {r_base_y:.1f})")

                # 各r_ノードをl_ノードからの相対位置を保って配置
                for l_node, r_node in l_to_r_mapping.items():
                    l_x, l_y = l_positions[l_node]
                    # l_ノードの相対位置を計算（l_系の左上基準）
                    rel_x = l_x - l_min_x
                    rel_y = l_y - l_min_y
                    # r_ノードの絶対位置を計算
                    new_x = r_base_x + rel_x
                    new_y = r_base_y + rel_y
                    r_node.set_pos(new_x, new_y)
                    print(f"  Repositioned {r_node.name()} to ({new_x:.1f}, {new_y:.1f}) (offset: {rel_x:.1f}, {rel_y:.1f})")

            # 位置を再計算
            self.recalculate_all_positions()

            # すべてのノードの色を接続状態に応じて更新
            self.update_all_node_colors()

            # 全てのr_ノードの色を3Dビューに反映・更新
            if self.stl_viewer:
                for r_node in l_to_r_mapping.values():
                    if hasattr(self.stl_viewer, 'apply_color_to_node'):
                        self.stl_viewer.apply_color_to_node(r_node)
                print(f"  Applied colors to {len(l_to_r_mapping)} r_ nodes in 3D view")

            # コライダー表示を更新
            if self.stl_viewer:
                self.stl_viewer.refresh_collider_display()

            print(f"\nSuccessfully created {len(l_to_r_mapping)} right side nodes from left side")

        except Exception as e:
            print(f"Error building right side from left side: {str(e)}")
            import traceback
            traceback.print_exc()

    def disconnect_ports(self, from_port, to_port):
        """ポートの接続を解除"""
        try:
            print(f"Disconnecting ports: {from_port.node().name()}.{from_port.name()} -> {to_port.node().name()}.{to_port.name()}")
            
            # 接続を解除する前に位置情報をリセット
            child_node = to_port.node()
            if child_node:
                self.stl_viewer.reset_stl_transform(child_node)
            
            # 利用可能なメソッドを探して接続解除を試みる
            if hasattr(self, 'disconnect_nodes'):
                success = self.disconnect_nodes(
                    from_port.node(), from_port.name(),
                    to_port.node(), to_port.name())
            elif hasattr(from_port, 'disconnect_from'):
                success = from_port.disconnect_from(to_port)
            else:
                success = False
                print("No suitable disconnection method found")
                
            if success:
                print("Ports disconnected successfully")
                # on_port_disconnectedイベントを呼び出す
                self.on_port_disconnected(to_port, from_port)
                return True
            else:
                print("Failed to disconnect ports")
                return False
                
        except Exception as e:
            print(f"Error disconnecting ports: {str(e)}")
            return False

    def _write_joint(self, file, parent_node, child_node):
        """ジョイントの出力"""
        try:
            # 親ノードのポイント情報から原点を取得
            origin_xyz = [0, 0, 0]  # デフォルト値を設定
            origin_rpy = [0.0, 0.0, 0.0]  # デフォルト値を設定
            for port in parent_node.output_ports():
                for connected_port in port.connected_ports():
                    if connected_port.node() == child_node:
                        try:
                            port_name = port.name()
                            if '_' in port_name:
                                parts = port_name.split('_')
                                if len(parts) > 1 and parts[1].isdigit():
                                    port_idx = int(parts[1]) - 1
                                    if port_idx < len(parent_node.points):
                                        origin_xyz = parent_node.points[port_idx]['xyz']
                                        # Use angle if available (UI-edited value), otherwise fallback to rpy
                                        # Both are stored in radians
                                        origin_rpy = parent_node.points[port_idx].get('angle',
                                                     parent_node.points[port_idx].get('rpy', [0.0, 0.0, 0.0]))
                        except Exception as e:
                            print(f"Warning: Error processing port {port.name()}: {str(e)}")
                        break

            joint_name = f"{parent_node.name()}_to_{child_node.name()}"

            # 回転軸の値に基づいてジョイントタイプを決定
            if hasattr(child_node, 'rotation_axis'):
                if child_node.rotation_axis == 3:  # Fixed
                    file.write(f'  <joint name="{joint_name}" type="fixed">\n')
                    file.write(f'    <origin xyz="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" rpy="{origin_rpy[0]} {origin_rpy[1]} {origin_rpy[2]}"/>\n')
                    file.write(f'    <parent link="{parent_node.name()}"/>\n')
                    file.write(f'    <child link="{child_node.name()}"/>\n')
                    file.write('  </joint>\n')
                else:
                    # 回転ジョイントの処理
                    file.write(f'  <joint name="{joint_name}" type="revolute">\n')
                    axis = [0, 0, 0]
                    if child_node.rotation_axis == 0:    # X軸
                        axis = [1, 0, 0]
                    elif child_node.rotation_axis == 1:  # Y軸
                        axis = [0, 1, 0]
                    else:                          # Z軸
                        axis = [0, 0, 1]

                    file.write(f'    <origin xyz="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" rpy="{origin_rpy[0]} {origin_rpy[1]} {origin_rpy[2]}"/>\n')
                    file.write(f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n')
                    file.write(f'    <parent link="{parent_node.name()}"/>\n')
                    file.write(f'    <child link="{child_node.name()}"/>\n')

                    # Joint limitパラメータを取得（デフォルト値を使用）
                    lower = getattr(child_node, 'joint_lower', -3.14159)
                    upper = getattr(child_node, 'joint_upper', 3.14159)
                    effort = getattr(child_node, 'joint_effort', 10.0)
                    velocity = getattr(child_node, 'joint_velocity', 3.0)

                    # URDF標準のlimit要素（effort, velocityのみ）
                    file.write(f'    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>\n')

                    # dynamics要素（damping, friction）
                    damping = getattr(child_node, 'damping', 0.0)
                    friction = getattr(child_node, 'friction', 0.0)
                    file.write(f'    <dynamics damping="{damping}" friction="{friction}"/>\n')

                    file.write('  </joint>\n')

        except Exception as e:
            print(f"Error writing joint: {str(e)}")
            traceback.print_exc()

    def _format_visual_origin(self, node):
        """Visual originをフォーマット（デフォルト値でない場合のみ）"""
        if hasattr(node, 'visual_origin') and isinstance(node.visual_origin, dict):
            xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
            rpy = node.visual_origin.get('rpy', [0.0, 0.0, 0.0])
            # デフォルト値でない場合のみ出力
            if xyz != [0.0, 0.0, 0.0] or rpy != [0.0, 0.0, 0.0]:
                return f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n'
        return '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'

    def _format_mesh_scale(self, node):
        """Mesh scaleをフォーマット（デフォルト値でない場合のみ）"""
        if hasattr(node, 'mesh_scale') and isinstance(node.mesh_scale, (list, tuple)) and len(node.mesh_scale) == 3:
            scale = node.mesh_scale
            # デフォルト値[1.0, 1.0, 1.0]でない場合のみ出力
            if scale != [1.0, 1.0, 1.0]:
                return f' scale="{scale[0]} {scale[1]} {scale[2]}"'
        return ''

    def _write_urdf_collision(self, file, node, package_path, mesh_dir_name=None, mesh_format=".stl", unity_mode=False):
        """Write collision geometry for URDF (supports multiple colliders)

        Args:
            file: Output file handle
            node: Node object
            package_path: Package path for visual mesh fallback
            mesh_dir_name: Mesh directory name (not used in unity_mode)
            mesh_format: Mesh file format extension (not used in unity_mode)
            unity_mode: If True, use Unity-style paths (package://meshes/)
        """
        # Get colliders list
        colliders = []
        if hasattr(node, 'colliders') and node.colliders:
            colliders = node.colliders

        # Write each enabled collider
        for collider in colliders:
            if not collider.get('enabled', False):
                continue

            file.write('    <collision>\n')

            if collider.get('type') == 'primitive' and collider.get('data'):
                # Primitive collider
                data = collider['data']
                
                # Prioritize collider['position'] over data['position'] (collider['position'] is set from UI/XML)
                pos = collider.get('position', data.get('position', [0, 0, 0]))
                
                # Add visual_origin offset to collider position (collider should follow visual mesh position)
                if hasattr(node, 'visual_origin') and node.visual_origin:
                    visual_xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
                    pos = [
                        pos[0] + visual_xyz[0],
                        pos[1] + visual_xyz[1],
                        pos[2] + visual_xyz[2]
                    ]
                
                # Prioritize collider['rotation'] over data['rotation'] (collider['rotation'] is set from UI/XML)
                rot_deg = collider.get('rotation', data.get('rotation', [0, 0, 0]))
                # Convert degrees to radians for URDF
                rot_rad = [math.radians(r) for r in rot_deg]
                file.write(f'      <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{rot_rad[0]} {rot_rad[1]} {rot_rad[2]}"/>\n')
                file.write('      <geometry>\n')

                geom_type = data['type']
                geom = data.get('geometry', {})

                if geom_type == 'box':
                    size = geom.get('size', None)
                    if isinstance(size, str):
                        file.write(f'        <box size="{size}"/>\n')
                    else:
                        sx = float(geom.get('size_x', geom.get('x', 1.0)))
                        sy = float(geom.get('size_y', geom.get('y', 1.0)))
                        sz = float(geom.get('size_z', geom.get('z', 1.0)))
                        file.write(f'        <box size="{sx} {sy} {sz}"/>\n')
                elif geom_type == 'sphere':
                    radius = float(geom.get('radius', 0.5))
                    file.write(f'        <sphere radius="{radius}"/>\n')
                elif geom_type == 'cylinder':
                    radius = float(geom.get('radius', 0.5))
                    length = float(geom.get('length', 1.0))
                    file.write(f'        <cylinder radius="{radius}" length="{length}"/>\n')
                elif geom_type == 'capsule':
                    # URDF doesn't have native capsule, approximate with cylinder
                    radius = float(geom.get('radius', 0.5))
                    length = float(geom.get('length', 1.0))
                    file.write(f'        <cylinder radius="{radius}" length="{length}"/>\n')

                file.write('      </geometry>\n')

            elif collider.get('type') == 'mesh':
                # Mesh collider
                collider_mesh = collider.get('mesh')
                if collider_mesh:
                    # Get position and rotation from collider
                    pos = collider.get('position', [0, 0, 0])
                    
                    # Add visual_origin offset to collider position (collider should follow visual mesh position)
                    if hasattr(node, 'visual_origin') and node.visual_origin:
                        visual_xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
                        pos = [
                            pos[0] + visual_xyz[0],
                            pos[1] + visual_xyz[1],
                            pos[2] + visual_xyz[2]
                        ]
                    
                    rot_deg = collider.get('rotation', [0, 0, 0])
                    # Convert degrees to radians for URDF
                    rot_rad = [math.radians(r) for r in rot_deg]
                    file.write(f'      <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{rot_rad[0]} {rot_rad[1]} {rot_rad[2]}"/>\n')
                    file.write('      <geometry>\n')

                    # Build collider mesh path based on mode
                    if unity_mode:
                        # Unity: use original filename with package://meshes/
                        collider_filename = os.path.basename(collider_mesh)
                        collider_package_path = f"package://meshes/{collider_filename}"
                    else:
                        # Standard: convert format and use full package path
                        visual_dir = os.path.dirname(node.stl_file) if node.stl_file else ""
                        collider_absolute = os.path.join(visual_dir, collider_mesh) if visual_dir else collider_mesh
                        collider_original_filename = os.path.basename(collider_absolute)
                        collider_base_name = os.path.splitext(collider_original_filename)[0]
                        collider_filename = f"{collider_base_name}{mesh_format}"
                        collider_package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{collider_filename}"

                    mesh_scale = collider.get('mesh_scale', [1.0, 1.0, 1.0])
                    scale_attr = ''
                    if mesh_scale != [1.0, 1.0, 1.0]:
                        scale_attr = f' scale="{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}"'
                    file.write(f'        <mesh filename="{collider_package_path}"{scale_attr}/>\n')
                    file.write('      </geometry>\n')
                else:
                    # Default: use visual mesh as collider
                    # Get position and rotation from collider (even if no explicit mesh set)
                    pos = collider.get('position', [0, 0, 0])
                    
                    # Add visual_origin offset to collider position (collider should follow visual mesh position)
                    if hasattr(node, 'visual_origin') and node.visual_origin:
                        visual_xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
                        pos = [
                            pos[0] + visual_xyz[0],
                            pos[1] + visual_xyz[1],
                            pos[2] + visual_xyz[2]
                        ]
                    
                    rot_deg = collider.get('rotation', [0, 0, 0])
                    # Convert degrees to radians for URDF
                    rot_rad = [math.radians(r) for r in rot_deg]
                    file.write(f'      <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{rot_rad[0]} {rot_rad[1]} {rot_rad[2]}"/>\n')
                    file.write('      <geometry>\n')
                    file.write(f'        <mesh filename="{package_path}"/>\n')
                    file.write('      </geometry>\n')

            file.write('    </collision>\n')

    def _write_link(self, file, node, materials, mesh_format=".stl"):
        """リンクの出力"""
        try:
            file.write(f'  <link name="{node.name()}">\n')

            # 慣性パラメータ
            if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial Originの出力
                if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                    xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    rpy = node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                    file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
                else:
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write(f'      <mass value="{format_float_no_exp(node.mass_value)}"/>\n')
                file.write('      <inertia')
                for key, value in node.inertia.items():
                    file.write(f' {key}="{format_float_no_exp(value)}"')
                file.write('/>\n')
                file.write('    </inertial>\n')

            # メインのビジュアルとコリジョン
            if hasattr(node, 'stl_file') and node.stl_file:
                try:
                    mesh_dir_name = "meshes"
                    if self.meshes_dir:
                        dir_name = os.path.basename(self.meshes_dir)
                        if dir_name.startswith('mesh'):
                            mesh_dir_name = dir_name

                    # メインのビジュアル
                    # 元のファイル名の拡張子を選択された形式に変更
                    original_filename = os.path.basename(node.stl_file)
                    base_name = os.path.splitext(original_filename)[0]
                    mesh_filename = f"{base_name}{mesh_format}"
                    package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{mesh_filename}"

                    file.write('    <visual>\n')
                    # Visual originを出力（デフォルト値でない場合のみ）
                    file.write(self._format_visual_origin(node))
                    file.write('      <geometry>\n')
                    # Mesh scaleを出力（デフォルト値でない場合のみ）
                    scale_attr = self._format_mesh_scale(node)
                    file.write(f'        <mesh filename="{package_path}"{scale_attr}/>\n')
                    file.write('      </geometry>\n')

                    # カラー情報を追加
                    if hasattr(node, 'node_color') and len(node.node_color) >= 3:
                        rgb = node.node_color
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(rgb[0] * 255),
                            int(rgb[1] * 255),
                            int(rgb[2] * 255)
                        )
                        file.write(f'      <material name="{hex_color}"/>\n')

                    file.write('    </visual>\n')

                    # 装飾パーツのビジュアルを追加
                    for port_index, port in enumerate(node.output_ports()):
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if hasattr(dec_node, 'stl_file') and dec_node.stl_file:
                                    # 装飾パーツのファイル名も選択された形式に変更
                                    dec_original = os.path.basename(dec_node.stl_file)
                                    dec_base_name = os.path.splitext(dec_original)[0]
                                    dec_mesh_filename = f"{dec_base_name}{mesh_format}"
                                    dec_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{dec_mesh_filename}"

                                    # ポイント座標を取得（ジョイント位置）
                                    origin_xyz = "0 0 0"
                                    if hasattr(node, 'points') and port_index < len(node.points):
                                        point_data = node.points[port_index]
                                        if 'xyz' in point_data:
                                            xyz = point_data['xyz']
                                            origin_xyz = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                                    file.write('    <visual>\n')
                                    file.write(f'      <origin xyz="{origin_xyz}" rpy="0 0 0"/>\n')
                                    file.write('      <geometry>\n')
                                    file.write(f'        <mesh filename="{dec_path}"/>\n')
                                    file.write('      </geometry>\n')
                                    if hasattr(dec_node, 'node_color'):
                                        dec_color = '#{:02x}{:02x}{:02x}'.format(
                                            int(dec_node.node_color[0] * 255),
                                            int(dec_node.node_color[1] * 255),
                                            int(dec_node.node_color[2] * 255)
                                        )
                                        file.write(f'      <material name="{dec_color}"/>\n')
                                    file.write('    </visual>\n')

                    # コリジョン (新しいヘルパー関数を使用)
                    self._write_urdf_collision(file, node, package_path, mesh_dir_name, mesh_format)

                except Exception as e:
                    print(f"Error processing STL file for node {node.name()}: {str(e)}")
                    traceback.print_exc()

            file.write('  </link>\n')

        except Exception as e:
            print(f"Error writing link: {str(e)}")
            traceback.print_exc()

    def _write_link_unity(self, file, node, materials, unity_dir_name):
        """Unity用のリンク出力"""
        try:
            file.write(f'  <link name="{node.name()}">\n')

            # 慣性パラメータ
            if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial Originの出力
                if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                    xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    rpy = node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                    file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
                else:
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write(f'      <mass value="{format_float_no_exp(node.mass_value)}"/>\n')
                file.write('      <inertia')
                for key, value in node.inertia.items():
                    file.write(f' {key}="{format_float_no_exp(value)}"')
                file.write('/>\n')
                file.write('    </inertial>\n')

            # ビジュアルとコリジョン
            if hasattr(node, 'stl_file') and node.stl_file:
                try:
                    # メインのビジュアル
                    stl_filename = os.path.basename(node.stl_file)
                    # Unity用のパスは直接meshesを指定
                    package_path = f"package://meshes/{stl_filename}"

                    file.write('    <visual>\n')
                    # Visual originを出力（デフォルト値でない場合のみ）
                    file.write(self._format_visual_origin(node))
                    file.write('      <geometry>\n')
                    # Mesh scaleを出力（デフォルト値でない場合のみ）
                    scale_attr = self._format_mesh_scale(node)
                    file.write(f'        <mesh filename="{package_path}"{scale_attr}/>\n')
                    file.write('      </geometry>\n')
                    if hasattr(node, 'node_color') and len(node.node_color) >= 3:
                        rgb = node.node_color
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(rgb[0] * 255),
                            int(rgb[1] * 255),
                            int(rgb[2] * 255)
                        )
                        file.write(f'      <material name="{hex_color}"/>\n')
                    file.write('    </visual>\n')

                    # 装飾パーツのビジュアルを追加
                    for port_index, port in enumerate(node.output_ports()):
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if hasattr(dec_node, 'stl_file') and dec_node.stl_file:
                                    dec_stl = os.path.basename(dec_node.stl_file)
                                    dec_path = f"package://meshes/{dec_stl}"

                                    # ポイント座標を取得（ジョイント位置）
                                    origin_xyz = "0 0 0"
                                    if hasattr(node, 'points') and port_index < len(node.points):
                                        point_data = node.points[port_index]
                                        if 'xyz' in point_data:
                                            xyz = point_data['xyz']
                                            origin_xyz = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                                    file.write('    <visual>\n')
                                    file.write(f'      <origin xyz="{origin_xyz}" rpy="0 0 0"/>\n')
                                    file.write('      <geometry>\n')
                                    file.write(f'        <mesh filename="{dec_path}"/>\n')
                                    file.write('      </geometry>\n')
                                    if hasattr(dec_node, 'node_color'):
                                        dec_color = '#{:02x}{:02x}{:02x}'.format(
                                            int(dec_node.node_color[0] * 255),
                                            int(dec_node.node_color[1] * 255),
                                            int(dec_node.node_color[2] * 255)
                                        )
                                        file.write(f'      <material name="{dec_color}"/>\n')
                                    file.write('    </visual>\n')

                    # コリジョン (Unity用のパスで)
                    self._write_urdf_collision(file, node, package_path, unity_mode=True)

                except Exception as e:
                    print(f"Error processing STL file for node {node.name()}: {str(e)}")
                    traceback.print_exc()

            file.write('  </link>\n')

        except Exception as e:
            print(f"Error writing link for Unity: {str(e)}")
            traceback.print_exc()

    def export_for_unity(self):
        """Unityプロジェクト用のファイル構造を作成しエクスポート"""
        try:
            # ディレクトリ選択ダイアログを表示
            message_box = QtWidgets.QMessageBox()
            message_box.setIcon(QtWidgets.QMessageBox.Information)
            message_box.setWindowTitle("Select Directory")
            message_box.setText("Please select the directory where you want to create the Unity project structure.")
            message_box.exec_()

            base_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select Base Directory for Unity Export"
            )

            if not base_dir:
                print("Unity export cancelled")
                return False

            # ロボット名からディレクトリ名を生成
            robot_name = self.get_robot_name()
            unity_dir_name = f"{robot_name}_unity_description"
            unity_dir_path = os.path.join(base_dir, unity_dir_name)

            # メインディレクトリの作成
            os.makedirs(unity_dir_path, exist_ok=True)
            print(f"Created Unity description directory: {unity_dir_path}")

            # meshesディレクトリの作成
            meshes_dir = os.path.join(unity_dir_path, "meshes")
            os.makedirs(meshes_dir, exist_ok=True)
            print(f"Created meshes directory: {meshes_dir}")

            # STLファイルのコピー
            copied_files = []
            for node in self.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    if os.path.exists(node.stl_file):
                        # ファイル名のみを取得
                        stl_filename = os.path.basename(node.stl_file)
                        # コピー先のパスを生成
                        dest_path = os.path.join(meshes_dir, stl_filename)
                        # ファイルをコピー
                        shutil.copy2(node.stl_file, dest_path)
                        copied_files.append(stl_filename)
                        print(f"Copied mesh file: {stl_filename}")

            # URDFファイルの生成
            urdf_file = os.path.join(unity_dir_path, f"{robot_name}.urdf")
            with open(urdf_file, 'w', encoding='utf-8') as f:
                # ヘッダー
                f.write('<?xml version="1.0"?>\n')
                f.write(f'<robot name="{robot_name}">\n\n')

                # マテリアル定義の収集
                materials = {}
                for node in self.all_nodes():
                    if hasattr(node, 'node_color'):
                        rgb = node.node_color
                        if len(rgb) >= 3:
                            hex_color = '#{:02x}{:02x}{:02x}'.format(
                                int(rgb[0] * 255),
                                int(rgb[1] * 255),
                                int(rgb[2] * 255)
                            )
                            materials[hex_color] = rgb

                # マテリアルの書き出し
                f.write('<!-- material color setting -->\n')
                for hex_color, rgb in materials.items():
                    f.write(f'<material name="{hex_color}">\n')
                    f.write(f'  <color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"/>\n')
                    f.write('</material>\n')
                f.write('\n')

                # base_linkから開始して、ツリー構造を順番に出力
                visited_nodes = set()
                base_node = self.get_node_by_name('base_link')
                if base_node:
                    self._write_tree_structure_unity(f, base_node, None, visited_nodes, materials, unity_dir_name)

                f.write('</robot>\n')

            print(f"Unity export completed successfully:")
            print(f"- Directory: {unity_dir_path}")
            print(f"- URDF file: {urdf_file}")
            print(f"- Copied {len(copied_files)} mesh files")

            QtWidgets.QMessageBox.information(
                self.widget,
                "Unity Export Complete",
                f"URDF files have been exported for Unity URDF-Importer:\n\n"
                f"Directory Path:\n{unity_dir_path}\n\n"
                f"URDF File:\n{urdf_file}\n\n"
                f"The files are ready to be imported using Unity URDF-Importer."
            )

            return True

        except Exception as e:
            error_msg = f"Error exporting for Unity: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            QtWidgets.QMessageBox.critical(
                self.widget,
                "Export Error",
                error_msg
            )
            return False

    def _write_tree_structure_unity(self, file, node, parent_node, visited_nodes, materials, unity_dir_name):
        """Unity用のツリー構造を順番に出力"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless Decorationノードはスキップ（親ノードの<visual>として処理済み）
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        if node.name() == "base_link":
            # base_linkの出力
            self._write_base_link(file)
        
        # 現在のノードに接続されているジョイントとリンクを処理
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                if child_node not in visited_nodes:
                    # Massless Decorationでないノードのみジョイントとリンクを出力
                    if not (hasattr(child_node, 'massless_decoration') and child_node.massless_decoration):
                        # まずジョイントを出力
                        self._write_joint(file, node, child_node)
                        file.write('\n')
                        
                        # 次にリンクを出力（Unity用のパスで）
                        self._write_link_unity(file, child_node, materials, unity_dir_name)
                        file.write('\n')
                        
                        # 再帰的に子ノードを処理
                        self._write_tree_structure_unity(file, child_node, node, visited_nodes, materials, unity_dir_name)

    def export_mjcf(self):
        # === 必須ログ: MJCFエクスポート開始時に全ノードの慣性値を確認 ===
        print(f"\n{'='*80}")
        print(f"[MJCF_EXPORT_START] Checking all nodes' inertia before export")
        print(f"{'='*80}")
        for node in self.all_nodes():
            node_name = node.name()
            if 'arm_lower' in node_name.lower():
                if hasattr(node, 'inertia') and node.inertia:
                    print(f"  {node_name}: ixx={node.inertia.get('ixx', 0):.9e}, ixy={node.inertia.get('ixy', 0):.9e}, ixz={node.inertia.get('ixz', 0):.9e}")
                    print(f"            iyy={node.inertia.get('iyy', 0):.9e}, iyz={node.inertia.get('iyz', 0):.9e}, izz={node.inertia.get('izz', 0):.9e}")
                else:
                    print(f"  {node_name}: WARNING - node.inertia is not set!")
        print(f"{'='*80}\n")
        """MuJoCo MJCF形式でエクスポート（モジュラー構造）"""
        try:
            # 閉リンクノードから最新の情報を収集
            self.collect_closed_loop_joints_from_nodes()

            # ロボット名を取得し、制御文字を除去
            import re
            import shutil
            robot_name = self.robot_name or "robot"
            # サニタイズ処理（制御文字除去、空白置換、予約語回避）
            robot_name = self._sanitize_name(robot_name)

            # カスタムダイアログを作成（ディレクトリ名とbase_link heightの両方を入力）
            dialog = QtWidgets.QDialog(self.widget)
            dialog.setWindowTitle("MJCF Export - Settings")
            dialog.setMinimumWidth(400)
            
            layout = QtWidgets.QVBoxLayout(dialog)
            
            # ディレクトリ名入力
            dir_label = QtWidgets.QLabel("Enter directory name for MJCF export:")
            layout.addWidget(dir_label)
            dir_input = QtWidgets.QLineEdit()
            dir_input.setText(f"{robot_name}_mjcf")
            layout.addWidget(dir_input)
            
            # base_link height入力
            height_label = QtWidgets.QLabel("Default base_link height (m):")
            layout.addWidget(height_label)
            height_input = QtWidgets.QLineEdit()
            # Settingsに設定された値を初期値として表示
            default_height = getattr(self, 'default_base_link_height', DEFAULT_BASE_LINK_HEIGHT)
            if hasattr(self, 'graph') and hasattr(self.graph, 'default_base_link_height'):
                default_height = self.graph.default_base_link_height
            height_input.setText(f"{default_height:.4f}")
            # 数値のみ入力可能にする
            height_input.setValidator(QDoubleValidator(0.0, 100.0, 6))
            layout.addWidget(height_input)

            # Fix Base to Ground
            fix_base_checkbox = QtWidgets.QCheckBox("Fix Base to Ground")
            fix_base_checkbox.setChecked(False)
            layout.addWidget(fix_base_checkbox)
            
            # ボタン
            button_layout = QtWidgets.QHBoxLayout()
            ok_button = QtWidgets.QPushButton("OK")
            ok_button.setDefault(True)
            cancel_button = QtWidgets.QPushButton("Cancel")
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                print("MJCF export cancelled")
                return False
            
            dir_name = dir_input.text().strip()
            base_link_height_str = height_input.text().strip()
            
            if not dir_name:
                print("MJCF export cancelled: directory name is empty")
                return False
            
            # base_link heightを取得（デフォルト値を使用）
            try:
                base_link_height = float(base_link_height_str) if base_link_height_str else default_height
            except ValueError:
                print(f"Warning: Invalid base_link height '{base_link_height_str}', using default {default_height}")
                base_link_height = default_height

            # Fix Base to Ground
            fix_base_to_ground = fix_base_checkbox.isChecked()
            
            # ディレクトリ名をサニタイズ
            dir_name = self._sanitize_name(dir_name)
            
            # base_link heightを保存（次回の出力時に使用）
            if hasattr(self, 'graph'):
                self.graph.default_base_link_height = base_link_height

            # 保存先の親ディレクトリを選択
            parent_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select Parent Directory for MJCF Export"
            )

            if not parent_dir:
                print("MJCF export cancelled")
                return False

            # MJCFディレクトリ構造を作成
            mjcf_dir = os.path.join(parent_dir, dir_name)
            assets_dir = os.path.join(mjcf_dir, "assets")

            os.makedirs(mjcf_dir, exist_ok=True)
            os.makedirs(assets_dir, exist_ok=True)

            print(f"\n=== Exporting MJCF to {mjcf_dir} ===")
            print(f"MJCF directory: {mjcf_dir}")
            print(f"Assets directory: {assets_dir}")

            # メッシュファイルをコピーしてマッピングを作成
            node_to_mesh = {}
            mesh_names = {}
            mesh_file_to_name = {}  # (mesh_filename, scale_tuple) → メッシュ名のマッピング（重複回避用、scaleも考慮）
            mesh_file_to_scale = {}  # (mesh_filename, scale_tuple) → スケールのマッピング（後方互換性のため残す）
            node_to_mesh_scale_key = {}  # node → (mesh_filename, scale_tuple) のマッピング
            collider_file_to_name = {}  # コライダーファイル → メッシュ名のマッピング（重複回避用）
            mesh_counter = 0

            for node in self.all_nodes():
                # Hide Meshにチェックが入っているノードはスキップ
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                # stl_file属性をチェック（.stl, .dae, .objすべてをサポート）
                mesh_file_path = None
                if hasattr(node, 'stl_file') and node.stl_file:
                    mesh_file_path = node.stl_file
                
                if mesh_file_path and os.path.exists(mesh_file_path):
                    # 元のメッシュファイル名を取得
                    original_filename = os.path.basename(mesh_file_path)
                    file_ext = os.path.splitext(original_filename)[1].lower()
                    
                    # サポートされている拡張子をチェック
                    supported_extensions = ['.stl', '.dae', '.obj']
                    if file_ext not in supported_extensions:
                        print(f"Warning: Unsupported mesh file extension '{file_ext}' for '{original_filename}'. Skipping.")
                        continue

                    # mesh_scaleを事前に取得（Y軸ミラーリング判定用）
                    mesh_scale = getattr(node, 'mesh_scale', [1.0, 1.0, 1.0])
                    needs_y_mirror = False
                    if len(mesh_scale) >= 2 and mesh_scale[1] < 0:
                        needs_y_mirror = True

                    # メッシュファイルを検証してOBJ形式に変換
                    try:
                        import trimesh
                        
                        # .daeファイルの場合、特別な処理が必要な場合がある
                        if file_ext == '.dae':
                            # COLLADAファイルを読み込み（Sceneの可能性があるため、force='mesh'を指定）
                            try:
                                mesh = trimesh.load(mesh_file_path, force='mesh')
                                # Sceneの場合、最初のメッシュを取得
                                if hasattr(mesh, 'geometry'):
                                    # Sceneオブジェクトの場合
                                    if len(mesh.geometry) > 0:
                                        # 最初のメッシュを取得
                                        mesh = list(mesh.geometry.values())[0]
                                    else:
                                        print(f"Warning: DAE file '{original_filename}' has no geometry. Skipping.")
                                        continue
                            except Exception as e:
                                print(f"Warning: Could not load DAE file '{original_filename}': {e}")
                                # フォールバック: 通常の読み込みを試行
                                try:
                                    mesh = trimesh.load(mesh_file_path)
                                    if hasattr(mesh, 'geometry'):
                                        if len(mesh.geometry) > 0:
                                            mesh = list(mesh.geometry.values())[0]
                                        else:
                                            print(f"Warning: DAE file '{original_filename}' has no geometry. Skipping.")
                                            continue
                                except:
                                    print(f"Warning: Failed to load DAE file '{original_filename}'. Skipping.")
                                    continue
                        else:
                            # .stl または .obj ファイル
                            mesh = trimesh.load(mesh_file_path)

                        # 面の数を確認（MuJoCoは1〜200000の範囲を要求）
                        num_faces = 0
                        if hasattr(mesh, 'faces'):
                            num_faces = len(mesh.faces)
                        elif hasattr(mesh, 'triangles'):
                            num_faces = len(mesh.triangles)

                        if num_faces < 1:
                            print(f"Warning: Skipping mesh '{original_filename}' - no faces found (file may be empty or invalid)")
                            continue
                        elif num_faces > 200000:
                            print(f"Warning: Skipping mesh '{original_filename}' - too many faces: {num_faces} (MuJoCo limit: 200000)")
                            continue

                        # Y軸が負の場合はメッシュをY軸でミラーリング
                        if needs_y_mirror:
                            import numpy as np
                            # 頂点座標のY座標を反転
                            if hasattr(mesh, 'vertices'):
                                mesh.vertices[:, 1] = -mesh.vertices[:, 1]
                            # 法線ベクトルのY座標も反転
                            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                                # read-only配列の場合はコピーを作成
                                if not mesh.vertex_normals.flags.writeable:
                                    mesh.vertex_normals = mesh.vertex_normals.copy()
                                mesh.vertex_normals[:, 1] = -mesh.vertex_normals[:, 1]
                            # 面の頂点順序を反転（法線の向きを維持）
                            if hasattr(mesh, 'faces'):
                                mesh.faces = np.flip(mesh.faces, axis=1)
                            print(f"  Mirrored mesh along Y-axis for node '{node.name()}' (mesh_scale Y: {mesh_scale[1]})")

                        # OBJファイル名を生成（拡張子を.objに変更）
                        mesh_filename = os.path.splitext(original_filename)[0] + '.obj'
                        dest_mesh_path = os.path.join(assets_dir, mesh_filename)

                        # OBJ形式でエクスポート（MuJoCoとの互換性向上）
                        mesh.export(dest_mesh_path, file_type='obj')
                        if needs_y_mirror:
                            print(f"Converted and mirrored mesh: {original_filename} -> {mesh_filename} ({num_faces} faces)")
                        else:
                            print(f"Converted mesh: {original_filename} -> {mesh_filename} ({num_faces} faces)")

                    except Exception as e:
                        print(f"Warning: Could not process mesh '{original_filename}': {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"Skipping this mesh file.")
                        continue

                    # MJCF内で使用するファイル名（meshdirで参照）
                    node_to_mesh[node] = mesh_filename

                    # メッシュ名を生成（拡張子なし）
                    base_mesh_name = os.path.splitext(mesh_filename)[0]

                    # スケール情報を正規化（Y軸が負の場合は既にメッシュをミラーリング済みなので、Y軸を正の値に戻す）
                    if needs_y_mirror and len(mesh_scale) >= 2:
                        # Y軸が負の場合は、メッシュをミラーリング済みなので、Y軸を正の値に戻す
                        mesh_scale_for_mjcf = [mesh_scale[0], abs(mesh_scale[1]), mesh_scale[2]]
                    else:
                        # Y軸が正の場合は、X/Z軸の負の値も含めてそのまま保存（MJCFのscale属性で表現）
                        mesh_scale_for_mjcf = mesh_scale
                    
                    # scaleを含めたキーを作成（同じファイルでもscaleが異なる場合は別のmeshとして扱う）
                    scale_key = tuple(mesh_scale_for_mjcf)
                    mesh_key = (mesh_filename, scale_key)
                    
                    # 同じメッシュファイル+スケールの組み合わせが既に登録されているかチェック
                    if mesh_key not in mesh_file_to_name:
                        # 新規メッシュ+スケールの組み合わせ: mesh_file_to_nameに登録
                        # scaleが[1,1,1]でない場合は、mesh名にscale情報を含める
                        if mesh_scale_for_mjcf != [1.0, 1.0, 1.0]:
                            # scale情報を文字列に変換（負の値も含む）
                            scale_suffix = f"-scale-{mesh_scale_for_mjcf[0]}-{mesh_scale_for_mjcf[1]}-{mesh_scale_for_mjcf[2]}"
                            # 負の値を表現するため、マイナス記号をmに置換（ファイル名として安全）
                            scale_suffix = scale_suffix.replace('-', 'm').replace('.', 'd')
                            unique_mesh_name = f"{base_mesh_name}{scale_suffix}"
                        else:
                            unique_mesh_name = base_mesh_name
                        
                        mesh_file_to_name[mesh_key] = unique_mesh_name
                        mesh_file_to_scale[mesh_key] = mesh_scale_for_mjcf
                        mesh_counter += 1
                        print(f"  Registered unique mesh: {unique_mesh_name} -> {mesh_filename} (scale: {mesh_scale_for_mjcf})")
                    else:
                        # 既に登録されているメッシュ+スケールの組み合わせ
                        existing_mesh_name = mesh_file_to_name[mesh_key]
                        print(f"  Reusing existing mesh: {existing_mesh_name} -> {mesh_filename} (scale: {mesh_scale_for_mjcf})")

                    # ノード → メッシュ名のマッピング（scaleを含めたキーを使用）
                    mesh_names[node] = mesh_file_to_name[mesh_key]
                    node_to_mesh_scale_key[node] = mesh_key

            # node.colliders リストの各コライダーメッシュを処理
            for node in self.all_nodes():
                # Hide Meshにチェックが入っているノードはスキップ
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue
                
                if hasattr(node, 'colliders') and node.colliders:
                    for collider in node.colliders:
                        if collider.get('type') == 'mesh' and collider.get('mesh'):
                            collider_mesh_path = collider['mesh']
                            
                            # 相対パスの場合はビジュアルメッシュのディレクトリから解決
                            if not os.path.isabs(collider_mesh_path):
                                visual_mesh = getattr(node, 'stl_file', None)
                                if visual_mesh and os.path.exists(visual_mesh):
                                    visual_dir = os.path.dirname(visual_mesh)
                                    collider_source_path = os.path.join(visual_dir, collider_mesh_path)
                                else:
                                    collider_source_path = collider_mesh_path
                            else:
                                collider_source_path = collider_mesh_path
                            
                            if os.path.exists(collider_source_path):
                                try:
                                    import trimesh
                                    
                                    collider_file_ext = os.path.splitext(collider_source_path)[1].lower()
                                    
                                    # .daeファイルの場合、特別な処理が必要な場合がある
                                    if collider_file_ext == '.dae':
                                        try:
                                            collider_mesh = trimesh.load(collider_source_path, force='mesh')
                                            if hasattr(collider_mesh, 'geometry'):
                                                if len(collider_mesh.geometry) > 0:
                                                    collider_mesh = list(collider_mesh.geometry.values())[0]
                                                else:
                                                    print(f"Warning: Collider DAE file '{collider_mesh_path}' has no geometry. Skipping.")
                                                    continue
                                        except Exception as e:
                                            try:
                                                collider_mesh = trimesh.load(collider_source_path)
                                                if hasattr(collider_mesh, 'geometry'):
                                                    if len(collider_mesh.geometry) > 0:
                                                        collider_mesh = list(collider_mesh.geometry.values())[0]
                                                    else:
                                                        print(f"Warning: Collider DAE file '{collider_mesh_path}' has no geometry. Skipping.")
                                                        continue
                                            except:
                                                print(f"Warning: Failed to load collider DAE file '{collider_mesh_path}'. Skipping.")
                                                continue
                                    else:
                                        collider_mesh = trimesh.load(collider_source_path)
                                    
                                    # 面の数を確認
                                    num_faces = 0
                                    if hasattr(collider_mesh, 'faces'):
                                        num_faces = len(collider_mesh.faces)
                                    elif hasattr(collider_mesh, 'triangles'):
                                        num_faces = len(collider_mesh.triangles)
                                    
                                    if num_faces < 1:
                                        print(f"Warning: Skipping collider mesh '{collider_mesh_path}' - no faces found")
                                        continue
                                    elif num_faces > 200000:
                                        print(f"Warning: Skipping collider mesh '{collider_mesh_path}' - too many faces: {num_faces}")
                                        continue
                                    
                                    # OBJファイル名を生成
                                    collider_filename = os.path.basename(collider_mesh_path)
                                    collider_filename = os.path.splitext(collider_filename)[0] + '.obj'
                                    
                                    # まずビジュアルメッシュとして既に登録されているかチェック
                                    # mesh_file_to_name のキーは (mesh_filename, scale_tuple) なので、scale_tuple を考慮する必要がある
                                    found_mesh_name = None
                                    for mesh_key, mesh_name in mesh_file_to_name.items():
                                        if isinstance(mesh_key, tuple) and mesh_key[0] == collider_filename:
                                            found_mesh_name = mesh_name
                                            break
                                        elif mesh_key == collider_filename:
                                            found_mesh_name = mesh_name
                                            break
                                    
                                    if found_mesh_name:
                                        # ビジュアルメッシュと同じファイル: 同じ名前を再利用
                                        collider['_mesh_name'] = found_mesh_name
                                        print(f"  Reusing visual mesh for collider in node '{node.name()}': {found_mesh_name} ({collider_filename})")
                                    elif collider_filename in collider_file_to_name:
                                        # 既にコライダーとして登録済み: 既存の名前を使用
                                        collider['_mesh_name'] = collider_file_to_name[collider_filename]
                                        print(f"  Reusing collider mesh in node '{node.name()}': {collider['_mesh_name']} ({collider_filename})")
                                    else:
                                        # 新規コライダーファイル: ファイルをエクスポートして登録
                                        collider_dest_path = os.path.join(assets_dir, collider_filename)
                                        collider_mesh.export(collider_dest_path, file_type='obj')
                                        print(f"Converted collider mesh: {collider_mesh_path} -> {collider_filename} ({num_faces} faces)")
                                        
                                        # メッシュ名を生成（拡張子なし）
                                        collider_mesh_name = os.path.splitext(collider_filename)[0]
                                        collider_file_to_name[collider_filename] = collider_mesh_name
                                        collider['_mesh_name'] = collider_mesh_name
                                        print(f"  Registered unique collider in node '{node.name()}': {collider_mesh_name} -> {collider_filename}")
                                        
                                        # ただし、生成したmesh名が実際に<asset>に存在するかは保証されない
                                        # （visual meshが複数のmeshに分割されてる場合など）
                                        # そのため、_write_mjcf_geom で visual mesh名にフォールバックする処理がある
                                    
                                except Exception as e:
                                    print(f"Warning: Could not process collider mesh '{collider_mesh_path}' in node '{node.name()}': {e}")
                                    continue
                            else:
                                print(f"Warning: Collider mesh file not found in node '{node.name()}': {collider_source_path}")

            # 作成されたジョイントのリストを追跡
            created_joints = []

            # base_linkを探す
            base_node = self.get_node_by_name('base_link')

            # 1. ロボット本体ファイル（{dir_name}.xml）を作成
            robot_file_path = os.path.join(mjcf_dir, f"{dir_name}.xml")
            robot_file_basename = os.path.basename(robot_file_path)  # 実際に生成したファイル名
            self._write_mjcf_robot_file(robot_file_path, dir_name, base_node, mesh_names, node_to_mesh, created_joints, mesh_file_to_name, mesh_file_to_scale, collider_file_to_name, node_to_mesh_scale_key, fix_base_to_ground)

            # 2. モデルのz軸全長を計算
            model_z_height = self._calculate_model_z_height(base_node, node_to_mesh)
            
            # 3. scene.xml を作成（ロボットファイルをinclude）
            scene_path = os.path.join(mjcf_dir, "scene.xml")
            self._write_mjcf_scene(scene_path, robot_file_basename, model_z_height, base_link_height, fix_base_to_ground)

            print(f"MJCF export completed: {robot_file_path}")
            print(f"Total mesh files copied: {len(node_to_mesh)}")

            QtWidgets.QMessageBox.information(
                self.widget,
                "Export Successful",
                f"MJCF files have been exported successfully:\n\n"
                f"{mjcf_dir}/\n"
                f"├─ {dir_name}.xml (robot)\n"
                f"├─ scene.xml\n"
                f"└─ assets/ ({len(node_to_mesh)} mesh files)"
            )

            return True

        except Exception as e:
            error_msg = f"Error exporting MJCF: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            QtWidgets.QMessageBox.critical(
                self.widget,
                "Export Error",
                error_msg
            )
            return False

    def _sanitize_name(self, name):
        """MuJoCo用に名前をサニタイズ（空白除去、制御文字除去、予約語回避）"""
        import re
        # 制御文字を除去
        name = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', name)
        # 空白をアンダースコアに置換
        name = name.replace(' ', '_')
        # 予約語を避ける（'main'はMuJoCoの予約クラス名）
        muoco_reserved = ['main', 'default', 'world', 'body', 'joint', 'geom', 'site', 'sensor', 'actuator', 'equality', 'tendon', 'contact', 'asset', 'option', 'compiler', 'visual', 'statistic']
        if name.lower() in muoco_reserved:
            name = f"{name}_obj"
        # 空文字列の場合はデフォルト名を使用
        if not name:
            name = "robot"
        return name

    def collect_closed_loop_joints_from_nodes(self):
        """グラフ内の閉リンクノードから最新の情報を収集"""
        collected_joints = []

        for node in self.all_nodes():
            if isinstance(node, ClosedLoopJointNode):
                joint_data = {
                    'name': node.joint_name,
                    'original_type': node.joint_type,
                    'parent': node.parent_link,
                    'child': node.child_link,
                    'origin_xyz': node.origin_xyz.copy() if isinstance(node.origin_xyz, list) else list(node.origin_xyz),
                    'origin_rpy': node.origin_rpy.copy() if isinstance(node.origin_rpy, list) else list(node.origin_rpy),
                    'gearbox_ratio': node.gearbox_ratio,
                    'gearbox_reference_body': node.gearbox_reference_body
                }
                collected_joints.append(joint_data)

        # 収集した情報でclosed_loop_jointsを更新
        self.closed_loop_joints = collected_joints
        print(f"Collected {len(collected_joints)} closed-loop joint(s) from nodes")

        return collected_joints

    def _write_mjcf_robot_file(self, file_path, model_name, base_node, mesh_names, node_to_mesh, created_joints, mesh_file_to_name, mesh_file_to_scale, collider_file_to_name, node_to_mesh_scale_key, fix_base_to_ground=False):
        """ロボット本体ファイルを作成（すべての要素を含む単一ファイル）
        
        Args:
            fix_base_to_ground: Trueの場合、base_linkから<freejoint>を削除して固定リンクにする
        """
        with open(file_path, 'w') as f:
            # モデル名をサニタイズ（予約語回避）
            sanitized_model_name = self._sanitize_name(model_name)
            # ヘッダー
            f.write(f'<mujoco model="{sanitized_model_name}">\n')

            # compiler設定
            f.write('  <compiler angle="radian" meshdir="assets" autolimits="true" />\n\n')

            # option設定
            f.write('  <option cone="elliptic" impratio="100" />\n\n')

            # default設定（最初の<default>が"main"クラスになる）
            f.write('  <default>\n')
            f.write('    <!-- 共通設定 -->\n')
            f.write('    <joint damping="0.1" armature="0.01" frictionloss="0.2"/>\n')
            f.write('    <motor ctrlrange="-23.7 23.7"/>\n')
            f.write('    <geom friction="0.4" margin="0.001" condim="3"/>\n')
            
            # コライダー用のdefault class（ネスト）
            f.write('    <!-- コライダー用：group=0 -->\n')
            f.write('    <default class="collision">\n')
            f.write('      <geom group="0"/>\n')
            f.write('    </default>\n')
            
            # ビジュアル用のdefault class（ネスト）
            f.write('    <!-- ビジュアル用：group=1（衝突しない） -->\n')
            f.write('    <default class="visual">\n')
            f.write('      <geom contype="0" conaffinity="0" group="1"/>\n')
            f.write('    </default>\n')
            
            f.write('  </default>\n\n')

            # asset設定（メッシュとマテリアル）
            f.write('  <asset>\n')
            f.write('    <material name="metal" rgba=".9 .95 .95 1" />\n')
            f.write('    <material name="black" rgba="0 0 0 1" />\n')
            f.write('    <material name="white" rgba="1 1 1 1" />\n')
            f.write('    <material name="gray" rgba="0.671705 0.692426 0.774270 1" />\n\n')

            # メッシュ定義（mesh_file_to_nameを使用して重複を避ける、scaleも考慮）
            processed_mesh_keys = set()  # 既に処理した(mesh_filename, scale_tuple)の組み合わせを追跡
            used_mesh_names = set()  # 出力済みのmesh名を追跡（一意性保証のため）
            processed_collider_meshes = set()  # 既に処理したコリジョンメッシュファイル名を追跡
            used_collider_names = set()  # 出力済みのコライダーmesh名を追跡
            
            for node in self.all_nodes():
                # Hide Meshにチェックが入っているノードはスキップ
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue
                
                if node in node_to_mesh and node in node_to_mesh_scale_key:
                    mesh_key = node_to_mesh_scale_key[node]
                    if mesh_key not in processed_mesh_keys:
                        processed_mesh_keys.add(mesh_key)
                        mesh_filename, scale_tuple = mesh_key
                        mesh_name = mesh_file_to_name.get(mesh_key, os.path.splitext(mesh_filename)[0])
                        mesh_scale = mesh_file_to_scale.get(mesh_key, [1.0, 1.0, 1.0])
                        
                        # mesh名の一意性を保証（重複する場合はサフィックスを追加）
                        unique_mesh_name = mesh_name
                        counter = 1
                        while unique_mesh_name in used_mesh_names:
                            unique_mesh_name = f"{mesh_name}_{counter}"
                            counter += 1
                        used_mesh_names.add(unique_mesh_name)
                        
                        # デバッグ出力: mesh_scaleの値を確認
                        node_name = node.name() if hasattr(node, 'name') else 'unknown'
                        if unique_mesh_name != mesh_name:
                            print(f"  ⚠ Mesh name '{mesh_name}' already exists, renamed to '{unique_mesh_name}'")
                        print(f"  Writing mesh '{unique_mesh_name}' for node '{node_name}': scale={mesh_scale}")
                        
                        # mesh_file_to_nameを更新（後続の参照用）
                        mesh_file_to_name[mesh_key] = unique_mesh_name
                        
                        # Scale属性が[1, 1, 1]でない場合のみ出力
                        if mesh_scale != [1.0, 1.0, 1.0]:
                            scale_str = f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}"
                            f.write(f'    <mesh name="{unique_mesh_name}" scale="{scale_str}" file="{mesh_filename}" />\n')
                            print(f"    ✓ Added scale attribute: {scale_str}")
                        else:
                            f.write(f'    <mesh name="{unique_mesh_name}" file="{mesh_filename}" />\n')
                
                # コリジョンメッシュも追加（collider_file_to_nameを使用）
                if hasattr(node, '_collider_mesh_name'):
                    collider_mesh_name = node._collider_mesh_name
                    # collider_file_to_nameからファイル名を逆引き
                    collider_filename = None
                    for filename, name in collider_file_to_name.items():
                        if name == collider_mesh_name:
                            collider_filename = filename
                            break
                    
                    if collider_filename and collider_filename not in processed_collider_meshes:
                        processed_collider_meshes.add(collider_filename)
                        
                        # コライダーmesh名の一意性を保証
                        unique_collider_name = collider_mesh_name
                        counter = 1
                        while unique_collider_name in used_collider_names or unique_collider_name in used_mesh_names:
                            unique_collider_name = f"{collider_mesh_name}_{counter}"
                            counter += 1
                        used_collider_names.add(unique_collider_name)
                        
                        if unique_collider_name != collider_mesh_name:
                            print(f"  ⚠ Collider mesh name '{collider_mesh_name}' already exists, renamed to '{unique_collider_name}'")
                            # collider_file_to_nameを更新
                            collider_file_to_name[collider_filename] = unique_collider_name
                            node._collider_mesh_name = unique_collider_name
                        
                        f.write(f'    <mesh name="{unique_collider_name}" file="{collider_filename}" />\n')
            
            # mesh_namesを更新（mesh_file_to_nameが更新されているので、すべてのノードのmesh名を同期）
            for node in self.all_nodes():
                if node in node_to_mesh_scale_key:
                    mesh_key = node_to_mesh_scale_key[node]
                    if mesh_key in mesh_file_to_name:
                        mesh_names[node] = mesh_file_to_name[mesh_key]

            f.write('  </asset>\n\n')

            # worldbody
            f.write('  <worldbody>\n')
            if base_node:
                visited_nodes = set()
                used_body_names = set()  # body名の一意性を保証するためのセット
                used_joint_names = set()  # joint名の一意性を保証するためのセット
                self._write_mjcf_body(f, base_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=4, fix_base_to_ground=fix_base_to_ground, used_body_names=used_body_names, used_joint_names=used_joint_names)
            f.write('  </worldbody>\n\n')

            # equality constraints (閉リンクジョイント)
            # ノード名マッピングを作成
            nodes_map = {node.name(): node for node in self.all_nodes()}
            self._write_mjcf_equality_constraints(f, nodes_map)

            # actuator (position control)
            if created_joints:
                f.write('  <actuator>\n')
                for joint_info in created_joints:
                    joint_name = joint_info['joint_name']
                    actuator_name = joint_info['motor_name'].replace('_motor', '_actuator')
                    
                    # 位置制御用のパラメータ
                    # kp: 位置ゲイン（stiffness）、デフォルト値を使用
                    kp = joint_info.get('kp', 100.0)  # デフォルトの位置ゲイン
                    
                    # ctrlrange: ジョイントの可動範囲（<compiler angle="radian"> なので radians で出力）
                    if joint_info.get('range_values'):
                        lower, upper = joint_info['range_values']
                        ctrlrange = f"{lower} {upper}"
                    else:
                        # デフォルト範囲（±π radians）
                        ctrlrange = "-3.14159 3.14159"
                    
                    # 位置制御アクチュエーター（gear="1" を明示的に設定して1:1の伝達比にする）
                    f.write(f'    <position name="{actuator_name}" joint="{joint_name}" gear="1" kp="{kp}" ctrlrange="{ctrlrange}"/>\n')
                f.write('  </actuator>\n\n')

            # sensor
            f.write('  <sensor>\n')
            f.write('    <!-- Add sensors here if needed -->\n')
            f.write('  </sensor>\n')

            f.write('</mujoco>\n')
        print(f"Created robot file: {file_path}")

    def _calculate_model_z_height(self, base_node, node_to_mesh):
        """
        モデルのz軸全長を計算（すべてのノードの累積座標とメッシュのバウンディングボックスを考慮）
        
        Returns:
            float: モデルのz軸全長（メートル）
        """
        try:
            import trimesh
            import numpy as np
            
            z_min = float('inf')
            z_max = float('-inf')
            
            # すべてのノードを再帰的に走査
            visited_nodes = set()

            def traverse_node(node, parent_coords=[0, 0, 0]):
                nonlocal z_min, z_max
                if node in visited_nodes:
                    return
                visited_nodes.add(node)
                
                # ノードの累積座標を計算
                if isinstance(node, BaseLinkNode):
                    node_coords = [0, 0, 0]
                else:
                    # 親ノードからの相対座標を取得
                    input_port = node.input_ports()[0] if node.input_ports() else None
                    if input_port and input_port.connected_ports():
                        parent_port = input_port.connected_ports()[0]
                        parent_node = parent_port.node()
                        port_name = parent_port.name()
                        port_index = int(port_name.split('_')[1]) - 1 if '_' in port_name else 0
                        if 0 <= port_index < len(parent_node.points):
                            point_xyz = parent_node.points[port_index]['xyz']
                            node_coords = [
                                parent_coords[0] + point_xyz[0],
                                parent_coords[1] + point_xyz[1],
                                parent_coords[2] + point_xyz[2]
                            ]
                        else:
                            node_coords = parent_coords
                    else:
                        node_coords = parent_coords
                
                # メッシュファイルがある場合、バウンディングボックスを取得
                if node in node_to_mesh:
                    # node.stl_fileから元のメッシュファイルを読み込む
                    if hasattr(node, 'stl_file') and node.stl_file and os.path.exists(node.stl_file):
                        try:
                            mesh = trimesh.load(node.stl_file)
                            if hasattr(mesh, 'bounds'):
                                mesh_bounds = mesh.bounds
                                # メッシュのローカル座標系でのz座標範囲
                                mesh_z_min = mesh_bounds[0][2]
                                mesh_z_max = mesh_bounds[1][2]
                                # 累積座標を考慮したz座標範囲
                                global_z_min = node_coords[2] + mesh_z_min
                                global_z_max = node_coords[2] + mesh_z_max
                                z_min = min(z_min, global_z_min)
                                z_max = max(z_max, global_z_max)
                        except Exception as e:
                            print(f"Warning: Could not load mesh for z-height calculation: {node.stl_file}, error: {e}")
                            # メッシュが読み込めない場合、ノードの座標のみを使用
                            z_min = min(z_min, node_coords[2])
                            z_max = max(z_max, node_coords[2])
                    else:
                        # メッシュファイルがない場合、ノードの座標のみを使用
                        z_min = min(z_min, node_coords[2])
                        z_max = max(z_max, node_coords[2])
                else:
                    # メッシュがない場合、ノードの座標のみを使用
                    z_min = min(z_min, node_coords[2])
                    z_max = max(z_max, node_coords[2])
                
                # 子ノードを再帰的に処理
                for port in node.output_ports():
                    for connected_port in port.connected_ports():
                        child_node = connected_port.node()
                        traverse_node(child_node, node_coords)
            
            # base_linkから開始
            if base_node:
                traverse_node(base_node)
            
            # z軸全長を計算
            if z_min != float('inf') and z_max != float('-inf'):
                z_height = z_max - z_min
                print(f"Model z-axis height: {z_height:.6f} m (min: {z_min:.6f}, max: {z_max:.6f})")
                return z_height
            else:
                print("Warning: Could not calculate model z-height, using default 0.5 m")
                return 0.5
                
        except ImportError:
            print("Warning: trimesh not available, using default z-height 0.5 m")
            return 0.5
        except Exception as e:
            print(f"Warning: Error calculating model z-height: {e}, using default 0.5 m")
            import traceback
            traceback.print_exc()
            return 0.5

    def _write_mjcf_scene(self, file_path, robot_file_basename, model_z_height=None, base_link_height=None, fix_base_to_ground=False):
        """scene.xmlを作成（ロボットファイルをinclude）
        
        Args:
            file_path: 出力先ファイルパス
            robot_file_basename: ロボットファイル名（basenameのみ）
            model_z_height: モデルのz軸全長（オプション）
            base_link_height: base_linkの高さ（m）。Noneの場合はSettingsの値を使用
            fix_base_to_ground: Trueの場合、base_linkをworldに固定するequality weldを追加
        """
        with open(file_path, 'w') as f:
            # ロボット名をサニタイズ（予約語回避）
            sanitized_robot_name = self._sanitize_name(robot_file_basename.replace('.xml', ''))
            f.write(f'<mujoco model="{sanitized_robot_name} scene">\n')

            # カメラの視点中心をロボットの初期位置（base_link_height）に設定
            if base_link_height is None:
                # Settingsから取得
                if hasattr(self, 'graph') and hasattr(self.graph, 'default_base_link_height'):
                    camera_center_z = self.graph.default_base_link_height
                else:
                    camera_center_z = DEFAULT_BASE_LINK_HEIGHT
            else:
                camera_center_z = base_link_height
            print(f"Setting camera center to z={camera_center_z:.6f} m (using base_link_height)")

            # 実際に生成したrobotファイル名を使用（basenameのみ）
            f.write(f'  <include file="{robot_file_basename}"/>\n\n')

            # base_link を地面（world）に固定
            # NOTE: base_link がロボットのルートbody名として出力される前提
            if fix_base_to_ground:
                f.write('  <equality>\n')
                f.write('    <weld name="fix_base_to_ground" body1="base_link" body2="world"/>\n')
                f.write('  </equality>\n\n')

            f.write(f'  <statistic center="0 0 {camera_center_z:.6f}" extent="{max(0.8, model_z_height * 1.2 if model_z_height else 0.8):.6f}"/>\n\n')
            f.write('  <visual>\n')
            f.write('    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n')
            f.write('    <rgba haze="0.15 0.25 0.35 1"/>\n')
            f.write('    <global azimuth="-130" elevation="-20"/>\n')
            f.write('  </visual>\n\n')
            f.write('  <asset>\n')
            f.write('    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n')
            f.write('    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n')
            f.write('      markrgb="0.8 0.8 0.8" width="300" height="300"/>\n')
            f.write('    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n')
            f.write('  </asset>\n\n')
            f.write('  <worldbody>\n')
            f.write('    <!-- Ground/Environment -->\n')
            f.write('    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n')
            f.write('    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" group="0"/>\n')
            f.write('  </worldbody>\n')
            f.write('</mujoco>\n')
        print(f"Created scene file: {file_path}")

    def _write_mjcf_defaults(self, file_path):
        """defaults.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <default>\n')
            f.write('    <joint damping="0.1" armature="0.01" />\n')
            f.write('    <geom contype="1" conaffinity="1" condim="3" friction="0.9 0.1 0.1" />\n')
            f.write('    <motor ctrllimited="true" />\n')
            f.write('  </default>\n')
            f.write('</mujoco>\n')
        print(f"Created defaults file: {file_path}")

    def _write_mjcf_body_file(self, file_path, base_node, mesh_names, node_to_mesh, created_joints):
        """body.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <worldbody>\n')

            if base_node:
                visited_nodes = set()
                self._write_mjcf_body(f, base_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=4, fix_base_to_ground=False)

            f.write('  </worldbody>\n')
            f.write('</mujoco>\n')
        print(f"Created body file: {file_path}")

    def _write_mjcf_actuators(self, file_path, created_joints):
        """actuators.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <actuator>\n')

            for joint_info in created_joints:
                joint_name = joint_info['joint_name']
                motor_name = joint_info['motor_name']
                effort = joint_info.get('effort', 10.0)
                stiffness = joint_info.get('stiffness', 100.0)
                damping = joint_info.get('damping', 15.0)

                # <position>要素として出力
                # kp: Stiffnessの値
                # kv: Dampingの値
                # forcerange: Effortの値を -値, +値 でプロット
                # forcelimited: 常にtrue
                # gear: 伝達比を1:1に設定（デフォルトだと半分になる問題を回避）
                kp_str = format_float_no_exp(stiffness)
                kv_str = format_float_no_exp(damping)
                forcerange = f"-{format_float_no_exp(effort)} {format_float_no_exp(effort)}"

                f.write(f'    <position name="{motor_name}" joint="{joint_name}" gear="1" kp="{kp_str}" kv="{kv_str}" forcerange="{forcerange}" forcelimited="true" />\n')

            f.write('  </actuator>\n')
            f.write('</mujoco>\n')
        print(f"Created actuators file: {file_path}")

    def _write_mjcf_equality_constraints(self, file, nodes_map):
        """閉リンクジョイントをMJCFのequality制約として出力

        Args:
            file: 出力先ファイルオブジェクト
            nodes_map: {link_name: node}のマッピング辞書
        """
        if not self.closed_loop_joints:
            return

        file.write('  <equality>\n')

        for joint_data in self.closed_loop_joints:
            joint_name = joint_data['name']
            original_type = joint_data.get('original_type', 'ball')
            parent_link = joint_data['parent']
            child_link = joint_data['child']
            origin_xyz = joint_data.get('origin_xyz', [0.0, 0.0, 0.0])

            # リンク名をサニタイズ
            parent_sanitized = self._sanitize_name(parent_link)
            child_sanitized = self._sanitize_name(child_link)

            if original_type == 'ball':
                # ball jointは<connect>として出力
                # anchor: 接続点の座標（子リンクのローカル座標系）
                anchor_str = f"{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}"
                file.write(f'    <connect body1="{parent_sanitized}" body2="{child_sanitized}" anchor="{anchor_str}"/>\n')
                print(f"  Added ball joint constraint: {joint_name} ({parent_link} <-> {child_link})")

            elif original_type == 'gearbox':
                # gearbox jointは<joint>として出力（2つのジョイントを連動）
                gearbox_ratio = joint_data.get('gearbox_ratio', 1.0)
                gearbox_reference_body = joint_data.get('gearbox_reference_body')

                if gearbox_reference_body:
                    # 参照ジョイント名を構築（命名規則に従う）
                    ref_sanitized = self._sanitize_name(gearbox_reference_body)
                    joint1_name = f"{ref_sanitized}_joint"
                    joint2_name = f"{child_sanitized}_joint"

                    # polycoef: [offset, ratio] - ギア比を表現
                    file.write(f'    <joint joint1="{joint1_name}" joint2="{joint2_name}" polycoef="0 {gearbox_ratio}"/>\n')
                    print(f"  Added gearbox joint constraint: {joint_name} (ratio: {gearbox_ratio})")
                else:
                    print(f"  Warning: gearbox joint '{joint_name}' missing reference_body, skipping")

            elif original_type == 'screw':
                # screw jointも<joint>として出力（並進と回転を連動）
                # TODO: screw jointの詳細な実装が必要な場合は追加
                print(f"  Warning: screw joint '{joint_name}' not fully implemented, skipping")

        file.write('  </equality>\n\n')

    def _write_mjcf_sensors(self, file_path):
        """sensors.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <sensor>\n')
            f.write('    <!-- Add sensors here if needed -->\n')
            f.write('  </sensor>\n')
            f.write('</mujoco>\n')
        print(f"Created sensors file: {file_path}")

    def _write_mjcf_materials(self, file_path, node_to_mesh, mesh_names):
        """assets/materials.xmlを作成（メッシュ定義を含む）"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <asset>\n')

            # メッシュ定義
            for node in self.all_nodes():
                if node in node_to_mesh and node in mesh_names:
                    mesh_path = node_to_mesh[node]
                    # ファイル名のみ抽出（meshdir="assets/meshes"を指定しているため）
                    mesh_filename = os.path.basename(mesh_path)
                    mesh_name = mesh_names[node]

                    # Mesh scale属性を取得（デフォルトは[1, 1, 1]）
                    mesh_scale = getattr(node, 'mesh_scale', [1.0, 1.0, 1.0])

                    # Scale属性が[1, 1, 1]でない場合のみ出力
                    if mesh_scale != [1.0, 1.0, 1.0]:
                        scale_str = f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}"
                        f.write(f'    <mesh name="{mesh_name}" scale="{scale_str}" file="{mesh_filename}" />\n')
                    else:
                        f.write(f'    <mesh name="{mesh_name}" file="{mesh_filename}" />\n')

            f.write('  </asset>\n')
            f.write('</mujoco>\n')
        print(f"Created materials file: {file_path}")

    # _convert_rpy_to_quaternion() is now euler_to_quaternion() from urdf_kitchen_utils

    def _rpy_to_quat(self, rpy):
        """
        Convert RPY (roll, pitch, yaw) in radians to quaternion.
        
        URDF RPY convention (ZYX extrinsic / XYZ intrinsic):
        - Rotate by yaw around Z axis
        - Then pitch around Y axis
        - Then roll around X axis
        
        Args:
            rpy: [roll, pitch, yaw] in radians
            
        Returns:
            np.ndarray: Quaternion [w, x, y, z]
        """
        roll, pitch, yaw = rpy
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        # Quaternion multiplication: qZ(yaw) * qY(pitch) * qX(roll)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = cy * sp * cr + sy * cp * sr
        z = sy * cp * cr - cy * sp * sr
        
        return np.array([w, x, y, z])

    def _rpy_to_rotation_matrix(self, rpy):
        """
        Convert RPY (roll, pitch, yaw) in radians to 3x3 rotation matrix.
        
        URDF RPY convention (ZYX extrinsic / XYZ intrinsic):
        - Rotate by yaw around Z axis
        - Then pitch around Y axis
        - Then roll around X axis
        
        Args:
            rpy: [roll, pitch, yaw] in radians
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        roll, pitch, yaw = rpy
        
        # Precompute trigonometric values
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        # Rotation matrix: R = R_z(yaw) * R_y(pitch) * R_x(roll)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        
        return R

    def _transform_inertia_to_body_frame(self, inertia_dict, rpy):
        """
        Transform inertia tensor from inertial frame to body frame.
        
        URDF specifies inertia in the inertial frame (defined by <inertial origin>).
        MJCF <inertial> specifies inertia in the body frame.
        Transformation: I_body = R * I_inertial * R^T
        
        Args:
            inertia_dict: Dictionary with keys 'ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'
            rpy: [roll, pitch, yaw] in radians (from <inertial origin rpy>)
            
        Returns:
            np.ndarray: 3x3 inertia tensor in body frame
        """
        # Build 3x3 inertia matrix from URDF format
        I_inertial = np.array([
            [inertia_dict.get('ixx', 0.0), inertia_dict.get('ixy', 0.0), inertia_dict.get('ixz', 0.0)],
            [inertia_dict.get('ixy', 0.0), inertia_dict.get('iyy', 0.0), inertia_dict.get('iyz', 0.0)],
            [inertia_dict.get('ixz', 0.0), inertia_dict.get('iyz', 0.0), inertia_dict.get('izz', 0.0)]
        ])
        
        # Get rotation matrix from RPY
        R = self._rpy_to_rotation_matrix(rpy)
        
        # Transform: I_body = R * I_inertial * R^T
        I_body = R @ I_inertial @ R.T
        
        return I_body

    def _ensure_symmetric_positive_definite(self, I):
        """
        Ensure inertia matrix is symmetric, positive semi-definite, and satisfies triangle inequality.
        
        MuJoCo requires: Ixx + Iyy >= Izz, Iyy + Izz >= Ixx, Izz + Ixx >= Iyy
        
        Args:
            I: 3x3 inertia matrix
            
        Returns:
            np.ndarray: Symmetric, positive semi-definite 3x3 inertia matrix satisfying triangle inequality
        """
        # Force symmetry: I = 0.5 * (I + I^T)
        I_sym = 0.5 * (I + I.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(I_sym)
        
        # Clip negative eigenvalues to zero (with small tolerance)
        eigenvals_clipped = np.maximum(eigenvals, 0.0)
        
        # Reconstruct matrix: I = V * diag(eigenvals) * V^T
        I_spd = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
        
        # Ensure symmetry again after reconstruction
        I_final = 0.5 * (I_spd + I_spd.T)
        
        # Ensure triangle inequality: Ixx + Iyy >= Izz, Iyy + Izz >= Ixx, Izz + Ixx >= Iyy
        # MuJoCo's balanceinertia algorithm: iteratively adjust diagonal elements
        # Use iterative approach to handle all three inequalities simultaneously
        Ixx = I_final[0, 0]
        Iyy = I_final[1, 1]
        Izz = I_final[2, 2]
        
        # Small margin for numerical stability
        epsilon = 1e-8
        
        # Iteratively fix triangle inequalities (max 10 iterations to avoid infinite loops)
        max_iterations = 10
        for iteration in range(max_iterations):
            violations = 0
            
            # Check and fix: Ixx + Iyy >= Izz
            if Ixx + Iyy < Izz - epsilon:
                violations += 1
                # Increase Ixx and Iyy proportionally
                if Ixx + Iyy > 0:
                    ratio_xx = Ixx / (Ixx + Iyy)
                    ratio_yy = Iyy / (Ixx + Iyy)
                    target_sum = Izz + epsilon
                    Ixx = target_sum * ratio_xx
                    Iyy = target_sum * ratio_yy
                else:
                    # Both are zero or very small, distribute equally
                    target_sum = Izz + epsilon
                    Ixx = target_sum / 2.0
                    Iyy = target_sum / 2.0
            
            # Check and fix: Iyy + Izz >= Ixx
            if Iyy + Izz < Ixx - epsilon:
                violations += 1
                if Iyy + Izz > 0:
                    ratio_yy = Iyy / (Iyy + Izz)
                    ratio_zz = Izz / (Iyy + Izz)
                    target_sum = Ixx + epsilon
                    Iyy = target_sum * ratio_yy
                    Izz = target_sum * ratio_zz
                else:
                    target_sum = Ixx + epsilon
                    Iyy = target_sum / 2.0
                    Izz = target_sum / 2.0
            
            # Check and fix: Izz + Ixx >= Iyy
            if Izz + Ixx < Iyy - epsilon:
                violations += 1
                if Izz + Ixx > 0:
                    ratio_zz = Izz / (Izz + Ixx)
                    ratio_xx = Ixx / (Izz + Ixx)
                    target_sum = Iyy + epsilon
                    Izz = target_sum * ratio_zz
                    Ixx = target_sum * ratio_xx
                else:
                    target_sum = Iyy + epsilon
                    Izz = target_sum / 2.0
                    Ixx = target_sum / 2.0
            
            # If no violations, we're done
            if violations == 0:
                break
        
        # Update diagonal elements
        I_final[0, 0] = Ixx
        I_final[1, 1] = Iyy
        I_final[2, 2] = Izz
        
        # Ensure symmetry one more time after diagonal adjustment
        I_final = 0.5 * (I_final + I_final.T)
        
        # Final verification (optional, for debugging)
        final_Ixx = I_final[0, 0]
        final_Iyy = I_final[1, 1]
        final_Izz = I_final[2, 2]
        if not (final_Ixx + final_Iyy >= final_Izz - epsilon and
                final_Iyy + final_Izz >= final_Ixx - epsilon and
                final_Izz + final_Ixx >= final_Iyy - epsilon):
            print(f"Warning: Triangle inequality still violated after correction: "
                  f"Ixx={final_Ixx:.9e}, Iyy={final_Iyy:.9e}, Izz={final_Izz:.9e}")
        
        return I_final

    def _format_inertia_for_mjcf(self, I_body):
        """
        Format inertia tensor for MJCF fullinertia attribute.
        
        MJCF fullinertia format: "Ixx Iyy Izz Ixy Ixz Iyz"
        (URDF format is: ixx ixy ixz iyy iyz izz)
        
        Args:
            I_body: 3x3 inertia matrix in body frame
            
        Returns:
            str: Formatted string for fullinertia attribute
        """
        # Extract components in MJCF order: Ixx Iyy Izz Ixy Ixz Iyz
        Ixx = I_body[0, 0]
        Iyy = I_body[1, 1]
        Izz = I_body[2, 2]
        Ixy = I_body[0, 1]
        Ixz = I_body[0, 2]
        Iyz = I_body[1, 2]
        
        # Format with consistent precision (use format_float_no_exp for consistency)
        return (f"{format_float_no_exp(Ixx)} {format_float_no_exp(Iyy)} {format_float_no_exp(Izz)} "
                f"{format_float_no_exp(Ixy)} {format_float_no_exp(Ixz)} {format_float_no_exp(Iyz)}")

    def _write_mjcf_geom(self, file, node, mesh_name, color_str, indent_str):
        """Write geom elements for MJCF (visual + collision)

        MuJoCoの仕様に従い、<body>内に直接<geom>要素を配置します。
        - group 0: 床面
        - group 1: ビジュアル用のメッシュ（Massless Decorationも含む）
        - group 3: コライダー（衝突判定用）
        """
        # ビジュアルgeom（group="1"でビジュアル用メッシュ）
        file.write(f'{indent_str}  <!-- ビジュアル（表示用） -->\n')

        # visual_origin の pos と quat 属性を取得
        pos_attr = ""
        quat_attr = ""
        if hasattr(node, 'visual_origin') and node.visual_origin:
            xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
            rpy = node.visual_origin.get('rpy', [0.0, 0.0, 0.0])
            
            print(f"[MJCF_EXPORT_DEBUG] Node '{node.name()}' visual_origin:")
            print(f"  xyz: {xyz}")
            print(f"  rpy (rad): {rpy}")
            
            # XYZ（位置）がゼロでない場合、pos属性を追加
            if xyz != [0.0, 0.0, 0.0]:
                pos_attr = f' pos="{xyz[0]} {xyz[1]} {xyz[2]}"'
                print(f"  → pos_attr SET: {pos_attr}")
            else:
                print(f"  → pos_attr SKIPPED (xyz is zero)")
            
            # RPY（編集された値）をquatに変換して出力
            if rpy != [0.0, 0.0, 0.0]:
                # RPY → Quat 変換
                from urdf_kitchen_utils import ConversionUtils
                quat = ConversionUtils.rpy_to_quat(rpy)
                
                quat_attr = f' quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"'
                print(f"  → quat_attr SET (converted from rpy): {quat_attr}")
                print(f"  [MJCF Export] Node '{node.name()}': visual_origin rpy={rpy} (rad) -> quat={quat}")
            else:
                print(f"  → rotation SKIPPED (identity rotation)")
        else:
            print(f"[MJCF_EXPORT_DEBUG] Node '{node.name()}' has NO visual_origin")

        geom_line = f'{indent_str}  <geom class="visual" type="mesh" mesh="{mesh_name}"{pos_attr}{quat_attr} rgba="{color_str}" group="1"/>\n'
        print(f"  → Final geom line: {geom_line.strip()}")
        file.write(geom_line)
        
        # Massless Decorationの場合はコライダーを出力しない（ビジュアルのみ）
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return
        
        # Get colliders list
        colliders = []
        if hasattr(node, 'colliders') and node.colliders:
            colliders = node.colliders

        # Write each enabled collider
        if colliders:
            file.write(f'{indent_str}  <!-- コライダー（衝突判定用） -->\n')
        
        for collider in colliders:
            if not collider.get('enabled', False):
                continue
            
            if collider.get('type') == 'primitive' and collider.get('data'):
                # Primitive collider
                data = collider['data']
                
                # Prioritize collider['position'] over data['position'] (collider['position'] is set from UI/XML)
                pos = collider.get('position', data.get('position', [0, 0, 0]))
                
                # Add visual_origin offset to collider position (collider should follow visual mesh position)
                if hasattr(node, 'visual_origin') and node.visual_origin:
                    visual_xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
                    pos = [
                        pos[0] + visual_xyz[0],
                        pos[1] + visual_xyz[1],
                        pos[2] + visual_xyz[2]
                    ]
                    print(f"  [Collider Export] Primitive collider: Added visual_origin xyz {visual_xyz} to pos")
                
                # Prioritize collider['rotation'] over data['rotation'] (collider['rotation'] is set from UI/XML)
                rot_deg = collider.get('rotation', data.get('rotation', [0, 0, 0]))
                # Convert URDF RPY (ZYX) to quaternion for MuJoCo
                quat = euler_to_quaternion(rot_deg[0], rot_deg[1], rot_deg[2])
                pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
                quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

                geom_type = data['type']
                geom = data.get('geometry', {})

                if geom_type == 'box':
                    # MuJoCo box: size is half-sizes
                    if 'size' in geom:
                        size_str = geom['size']
                        # Parse "x y z" and convert to half-sizes
                        sizes = [float(s)/2 for s in size_str.split()]
                        size_str = f"{sizes[0]} {sizes[1]} {sizes[2]}"
                    else:
                        sx = float(geom.get('x', geom.get('size_x', 1.0))) / 2
                        sy = float(geom.get('y', geom.get('size_y', 1.0))) / 2
                        sz = float(geom.get('z', geom.get('size_z', 1.0))) / 2
                        size_str = f"{sx} {sy} {sz}"
                    file.write(f'{indent_str}  <geom class="collision" type="box" size="{size_str}" pos="{pos_str}" quat="{quat_str}" group="3"/>\n')

                elif geom_type == 'sphere':
                    radius = float(geom.get('radius', 0.5))
                    file.write(f'{indent_str}  <geom class="collision" type="sphere" size="{radius}" pos="{pos_str}" quat="{quat_str}" group="3"/>\n')

                elif geom_type == 'cylinder':
                    radius = float(geom.get('radius', 0.5))
                    length = float(geom.get('length', 1.0)) / 2  # MuJoCo uses half-length
                    file.write(f'{indent_str}  <geom class="collision" type="cylinder" size="{radius} {length}" pos="{pos_str}" quat="{quat_str}" group="3"/>\n')

                elif geom_type == 'capsule':
                    radius = float(geom.get('radius', 0.5))
                    length = float(geom.get('length', 1.0)) / 2  # MuJoCo uses half-length
                    file.write(f'{indent_str}  <geom class="collision" type="capsule" size="{radius} {length}" pos="{pos_str}" quat="{quat_str}" group="3"/>\n')

            elif collider.get('type') == 'mesh':
                # Mesh collider
                collider_mesh = collider.get('mesh')
                
                # Get position from collider data
                collider_pos = collider.get('position', [0.0, 0.0, 0.0])
                
                # Add visual_origin offset to collider position (mesh collider should follow visual mesh position)
                if hasattr(node, 'visual_origin') and node.visual_origin:
                    visual_xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
                    collider_pos = [
                        collider_pos[0] + visual_xyz[0],
                        collider_pos[1] + visual_xyz[1],
                        collider_pos[2] + visual_xyz[2]
                    ]
                    print(f"  [Collider Export] Added visual_origin xyz {visual_xyz} to collider pos")
                
                collider_pos_str = f"{collider_pos[0]} {collider_pos[1]} {collider_pos[2]}"
                
                # Get rotation from collider data
                collider_rotation_deg = collider.get('rotation', [0.0, 0.0, 0.0])
                if collider_rotation_deg != [0.0, 0.0, 0.0]:
                    # Convert rotation (degrees) to quat
                    from urdf_kitchen_utils import ConversionUtils
                    import math
                    collider_rpy = [math.radians(r) for r in collider_rotation_deg]
                    collider_quat = ConversionUtils.rpy_to_quat(collider_rpy)
                    collider_quat_attr = f' quat="{collider_quat[0]} {collider_quat[1]} {collider_quat[2]} {collider_quat[3]}"'
                else:
                    collider_quat_attr = ""
                
                print(f"  [Collider Export] Mesh collider pos={collider_pos}, rotation_deg={collider_rotation_deg}, quat_attr={collider_quat_attr}")
                
                if collider_mesh:
                    # export_mjcf で処理済みの _mesh_name を優先使用
                    # ただし、_mesh_name が visual mesh名と一致しない場合（例: "link2" vs "link2_0"）、
                    # visual mesh名にフォールバックする（visual mesh名は必ず<asset>に存在する）
                    if '_mesh_name' in collider and collider['_mesh_name']:
                        collider_mesh_name = collider['_mesh_name']
                        # visual mesh名と一致する場合はそのまま使用、一致しない場合はvisual mesh名にフォールバック
                        # （visual meshが複数のmeshに分割されてる場合など、_mesh_nameがassetに存在しない可能性があるため）
                        if collider_mesh_name == mesh_name:
                            file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{collider_mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                        else:
                            # visual mesh名にフォールバック（安全策）
                            file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                    else:
                        # 旧形式: node._collider_mesh_name を使用（後方互換性）
                        if hasattr(node, '_collider_mesh_name') and node._collider_mesh_name:
                            collider_mesh_name = node._collider_mesh_name
                            # visual mesh名と一致する場合はそのまま使用、一致しない場合はvisual mesh名にフォールバック
                            if collider_mesh_name == mesh_name:
                                file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{collider_mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                            else:
                                # visual mesh名にフォールバック（安全策）
                                file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                        else:
                            # collider['_mesh_name'] が設定されてない場合、必ず visual mesh にフォールバック
                            # （ファイル名から生成したmesh名がassetに存在しない可能性があるため）
                            file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                else:
                    # Default: visual and collision use same mesh
                    file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')

    def _calculate_model_lowest_point(self, base_node, visited_nodes=None):
        """モデル全体の最低点（最小z座標）を計算
        
        Args:
            base_node: ベースノード
            visited_nodes: 訪問済みノードのセット
            
        Returns:
            float: モデルの最低点のz座標（ワールド座標系）
        """
        if visited_nodes is None:
            visited_nodes = set()
        
        min_z = 0.0  # デフォルト値
        
        # 全ノードを走査してコライダーの最低点を探す
        def traverse_nodes(node, current_z=0.0):
            nonlocal min_z
            
            if node in visited_nodes:
                return
            visited_nodes.add(node)
            
            # このノードのコライダーから最低点を取得
            node_min_z = self._get_node_lowest_point(node)
            total_min_z = current_z + node_min_z
            
            if total_min_z < min_z:
                min_z = total_min_z
            
            # 子ノードを再帰的に処理
            for output_port in node.output_ports():
                for connected_port in output_port.connected_ports():
                    child_node = connected_port.node()
                    
                    # 子ノードのz座標オフセットを取得
                    child_z_offset = 0.0
                    if hasattr(node, 'points'):
                        port_index = node.output_ports().index(output_port)
                        if port_index < len(node.points):
                            point = node.points[port_index]
                            if 'xyz' in point and len(point['xyz']) >= 3:
                                child_z_offset = point['xyz'][2]
                    
                    traverse_nodes(child_node, current_z + child_z_offset)
        
        traverse_nodes(base_node)
        return min_z
    
    def _get_node_lowest_point(self, node):
        """ノードの最低点（コライダーまたはメッシュ）を取得
        
        Args:
            node: 対象ノード
            
        Returns:
            float: ノードのローカル座標系での最低点のz座標
        """
        min_z = 0.0
        
        # コライダーがある場合、コライダーの最低点を使用
        if hasattr(node, 'colliders') and node.colliders:
            for collider in node.colliders:
                if not collider.get('enabled', False):
                    continue
                
                collider_type = collider.get('type')
                position = collider.get('position', [0, 0, 0])
                
                if collider_type == 'primitive' and 'data' in collider:
                    data = collider['data']
                    prim_type = data.get('type', 'box')
                    
                    # プリミティブ形状の最低点を計算
                    if prim_type == 'sphere':
                        radius = data.get('radius', 0.5)
                        collider_min = position[2] - radius
                    elif prim_type == 'box':
                        geometry = data.get('geometry', {})
                        size_z = geometry.get('size_z', 1.0)
                        collider_min = position[2] - size_z / 2
                    elif prim_type == 'cylinder':
                        radius = data.get('radius', 0.5)
                        collider_min = position[2] - radius
                    elif prim_type == 'capsule':
                        radius = data.get('radius', 0.5)
                        length = data.get('length', 1.0)
                        collider_min = position[2] - length / 2 - radius
                    else:
                        collider_min = position[2]
                    
                    if collider_min < min_z:
                        min_z = collider_min
        
        # コライダーがない場合、デフォルトで少し下に
        # （実際のメッシュのバウンディングボックスを計算するのは複雑なため簡略化）
        if min_z == 0.0:
            min_z = -0.1  # デフォルトオフセット
        
        return min_z

    def _write_mjcf_body(self, file, node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=2, joint_info=None, fix_base_to_ground=False, used_body_names=None, used_joint_names=None):
        """MJCF bodyを再帰的に出力
        
        Args:
            fix_base_to_ground: Trueの場合、base_linkから<freejoint>を削除して固定リンクにする
            used_body_names: 使用済みのbody名を追跡するセット（一意性保証のため）
            used_joint_names: 使用済みのjoint名を追跡するセット（一意性保証のため）
        """
        if node in visited_nodes:
            return
        visited_nodes.add(node)
        
        # used_body_namesが渡されていない場合は新規作成
        if used_body_names is None:
            used_body_names = set()
        
        # used_joint_namesが渡されていない場合は新規作成
        if used_joint_names is None:
            used_joint_names = set()

        # Massless Decorationノードはスキップ
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        # Hide Meshにチェックが入っているノードはスキップ
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            print(f"Skipping node with hide_mesh=True: {node.name()}")
            return

        indent_str = ' ' * indent

        # base_linkの場合は特別処理
        if node.name() == 'base_link':
            sanitized_name = self._sanitize_name(node.name())
            
            # body名の一意性を保証（重複する場合はサフィックスを追加）
            unique_name = sanitized_name
            counter = 1
            while unique_name in used_body_names:
                unique_name = f"{sanitized_name}_{counter}"
                counter += 1
            used_body_names.add(unique_name)
            
            if unique_name != sanitized_name:
                print(f"  ⚠ Body name '{sanitized_name}' already exists, renamed to '{unique_name}'")

            # すべてデフォルト値かチェック
            is_all_defaults = self._is_base_link_at_defaults(node)

            # Note: メッシュの反転はscale属性で表現するため、_reversedサフィックスは不要
            # base_linkボディを開始（z座標を設定して地面に埋まらないようにする）
            if fix_base_to_ground:
                # Fix Base to Ground: モデルの最低点を計算してz=0に接地
                min_z = self._calculate_model_lowest_point(node, visited_nodes.copy())
                z_pos = max(0, -min_z)  # 最低点がz=0になるように調整
                print(f"Fix Base to Ground: model lowest point = {min_z:.6f}, base z_pos = {z_pos:.6f}")
            else:
                # 通常: 保存されたbase_link_heightを使用（なければデフォルト値）
                z_pos = getattr(self, 'base_link_height', getattr(self, 'default_base_link_height', DEFAULT_BASE_LINK_HEIGHT))
            file.write(f'{indent_str}<body name="{unique_name}" pos="0 0 {z_pos}">\n')

            # freejoint（base_linkは自由に動ける）
            # fix_base_to_groundがTrueの場合は<freejoint>を出力しない（固定リンクにする）
            if not fix_base_to_ground:
                file.write(f'{indent_str}  <freejoint />\n')

            # moving body（freejointを持つ）なので、inertialの有無をチェック
            has_inertial = False
            if hasattr(node, 'mass_value') and node.mass_value > 0:
                has_inertial = True

            # すべてデフォルト値の場合は、geomは出力しない（子ノードは処理する）
            if not is_all_defaults:
                # ジオメトリ（メッシュ）
                if node in mesh_names:
                    mesh_name = mesh_names[node]
                    color_str = "0.8 0.8 0.8 1.0"
                    if hasattr(node, 'node_color') and node.node_color:
                        r, g, b = node.node_color[:3]
                        color_str = f"{r} {g} {b} 1.0"

                    # 新しいヘルパー関数を使用
                    self._write_mjcf_geom(file, node, mesh_name, color_str, indent_str)

            # base_linkの子ノードを処理
            for port in node.output_ports():
                for connected_port in port.connected_ports():
                    child_node = connected_port.node()

                    # Massless Decorationノードは<geom class="visual">として出力
                    if hasattr(child_node, 'massless_decoration') and child_node.massless_decoration:
                        if child_node in mesh_names:
                            dec_mesh_name = mesh_names[child_node]
                            # 色情報
                            dec_color_str = "0.8 0.8 0.8 1.0"
                            if hasattr(child_node, 'node_color') and child_node.node_color:
                                r, g, b = child_node.node_color[:3]
                                dec_color_str = f"{r} {g} {b} 1.0"

                            # ポイント座標を取得（装飾パーツの位置）
                            port_index = list(node.output_ports()).index(port)
                            pos_str = "0 0 0"
                            if hasattr(node, 'points') and port_index < len(node.points):
                                point_data = node.points[port_index]
                                if 'xyz' in point_data:
                                    xyz = point_data['xyz']
                                    pos_str = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                            # Massless Decorationは<geom class="visual">を使用
                            file.write(f'{indent_str}  <geom class="visual" type="mesh" mesh="{dec_mesh_name}" rgba="{dec_color_str}" pos="{pos_str}" group="2"/>\n')
                        continue

                    # Hide Meshにチェックが入っているノードはスキップ（ジョイント情報も作成しない）
                    if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                        continue

                    port_index = list(node.output_ports()).index(port)
                    child_joint_info = self._get_joint_info(node, child_node, port_index, created_joints)
                    self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent + 2, child_joint_info, fix_base_to_ground, used_body_names, used_joint_names)

            # moving body（freejointを持つ）にinertialがない場合、自動で追加
            if not has_inertial:
                # MuJoCoのmjMINVALより十分大きい値を設定（mass: 0.001, inertia: 1e-6）
                file.write(f'{indent_str}  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>\n')
                print(f"  Auto-added inertial to moving body (base_link) to avoid MuJoCo load error")

            # base_linkボディを閉じる
            file.write(f'{indent_str}</body>\n')
            return

        # 名前をサニタイズ
        sanitized_name = self._sanitize_name(node.name())
        
        # body名の一意性を保証（重複する場合はサフィックスを追加）
        unique_name = sanitized_name
        counter = 1
        while unique_name in used_body_names:
            unique_name = f"{sanitized_name}_{counter}"
            counter += 1
        used_body_names.add(unique_name)
        
        if unique_name != sanitized_name:
            print(f"  ⚠ Body name '{sanitized_name}' already exists, renamed to '{unique_name}'")

        # Note: メッシュの反転はscale属性で表現するため、_reversedサフィックスは不要
        # body開始（joint_infoがあれば位置情報とorientation情報を追加）
        pos_attr = f' pos="{joint_info["pos"]}"' if joint_info else ''

        # Body orientation: joint_infoのRPYを優先（URDFのjoint origin RPYが親から子への相対回転を表す）
        # body_angleは補助的な初期回転として扱う
        orientation_attr = ""
        rpy_to_use = None
        
        # 1. joint_infoのRPYを優先（URDFのjoint origin RPY）
        if joint_info and 'rpy' in joint_info:
            rpy_to_use = joint_info['rpy']
        # 2. body_angleをフォールバック（bodyの初期回転）
        elif hasattr(node, 'body_angle') and node.body_angle != [0.0, 0.0, 0.0]:
            rpy_to_use = node.body_angle
        
        # RPYからquaternionまたはxyaxesを生成
        if rpy_to_use and rpy_to_use != [0.0, 0.0, 0.0]:
            # quaternionを使用（より正確）
            quat = self._rpy_to_quat(rpy_to_use)
            quat_str = f"{format_float_no_exp(quat[0])} {format_float_no_exp(quat[1])} {format_float_no_exp(quat[2])} {format_float_no_exp(quat[3])}"
            orientation_attr = f' quat="{quat_str}"'

        file.write(f'{indent_str}<body name="{unique_name}"{pos_attr}{orientation_attr}>\n')

        # ジョイントを最初に出力（子ボディ内に配置）
        # body posで親からのオフセットを設定済みなので、joint posは常に"0 0 0"（body座標系の原点）
        is_moving_body = False
        if joint_info:
            # joint名の一意性を保証（重複する場合はサフィックスを追加）
            original_joint_name = joint_info["name"]
            unique_joint_name = original_joint_name
            counter = 1
            while unique_joint_name in used_joint_names:
                unique_joint_name = f"{original_joint_name}_{counter}"
                counter += 1
            used_joint_names.add(unique_joint_name)
            
            if unique_joint_name != original_joint_name:
                print(f"  ⚠ Joint name '{original_joint_name}' already exists, renamed to '{unique_joint_name}'")
                # created_jointsのjoint_nameも更新（actuator生成で使用される）
                joint_info["name"] = unique_joint_name
            
            # range, limited, margin, armature, frictionloss, damping, stiffness, refを出力（velocityはMJCFに存在しないため削除）
            joint_attrs = f'{joint_info["range"]}{joint_info["limited"]}{joint_info["margin"]}{joint_info["armature"]}{joint_info["frictionloss"]}{joint_info["damping"]}{joint_info["stiffness"]}{joint_info["ref"]}'
            file.write(f'{indent_str}  <joint name="{unique_joint_name}" type="{joint_info["type"]}" pos="0 0 0" axis="{joint_info["axis"]}"{joint_attrs} />\n')
            is_moving_body = True  # jointを持つbodyはmoving body

        # 慣性プロパティ
        has_inertial = False
        if hasattr(node, 'mass_value') and node.mass_value > 0:
            # 質量の最小閾値を設定（数値的安定性のため）
            MIN_MASS = 0.001  # 1g
            # MIN_INERTIAを削除 - URDF由来の値を尊重するため、最小閾値を適用しない

            mass = max(node.mass_value, MIN_MASS)

            # 慣性テンソルを処理
            if hasattr(node, 'inertia') and isinstance(node.inertia, dict) and node.inertia:
                # === 詳細ログ出力（原因特定用） ===
                node_name = node.name()
                is_target = 'arm_lower' in node_name.lower()  # l_arm_lower / r_arm_lower を検出

                if is_target:
                    print(f"\n{'='*80}")
                    print(f"MJCF Inertia Output Debug: {node_name}")
                    print(f"{'='*80}")
                    print(f"  Mass: {node.mass_value} (raw) -> {mass} (after MIN_MASS)")
                
                # Inertial Origin（COM位置と回転）を取得
                com_pos = "0 0 0"
                rpy = [0.0, 0.0, 0.0]  # Default: no rotation
                
                if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                    xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    com_pos = f"{format_float_no_exp(xyz[0])} {format_float_no_exp(xyz[1])} {format_float_no_exp(xyz[2])}"
                    rpy = node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                
                if is_target:
                    print(f"  Inertial Origin:")
                    print(f"    xyz: {xyz}")
                    print(f"    rpy: {rpy} (rad)")
                    print(f"  URDF Inertia (from node.inertia):")
                    print(f"    ixx={node.inertia.get('ixx', 0):.9e}, ixy={node.inertia.get('ixy', 0):.9e}, ixz={node.inertia.get('ixz', 0):.9e}")
                    print(f"    iyy={node.inertia.get('iyy', 0):.9e}, iyz={node.inertia.get('iyz', 0):.9e}, izz={node.inertia.get('izz', 0):.9e}")
                
                # Build I_inertial matrix for logging
                I_inertial = np.array([
                    [node.inertia.get('ixx', 0.0), node.inertia.get('ixy', 0.0), node.inertia.get('ixz', 0.0)],
                    [node.inertia.get('ixy', 0.0), node.inertia.get('iyy', 0.0), node.inertia.get('iyz', 0.0)],
                    [node.inertia.get('ixz', 0.0), node.inertia.get('iyz', 0.0), node.inertia.get('izz', 0.0)]
                ])
                
                if is_target:
                    print(f"  I_inertial matrix:")
                    print(f"    {I_inertial}")
                
                # Transform inertia from inertial frame to body frame
                # URDF inertia is specified in the inertial frame (defined by <inertial origin>)
                # MJCF <inertial> requires inertia in the body frame
                I_body = self._transform_inertia_to_body_frame(node.inertia, rpy)
                
                if is_target:
                    R = self._rpy_to_rotation_matrix(rpy)
                    print(f"  Rotation matrix R (from rpy):")
                    print(f"    {R}")
                    print(f"  I_body = R @ I_inertial @ R^T:")
                    print(f"    {I_body}")
                
                # Ensure symmetric and positive semi-definite
                I_body_before_spd = I_body.copy()
                I_body = self._ensure_symmetric_positive_definite(I_body)
                
                if is_target:
                    max_diff = np.max(np.abs(I_body - I_body_before_spd))
                    if max_diff > 1e-12:
                        print(f"  SPD correction applied: max_diff = {max_diff:.3e}")
                
                # MIN_INERTIA閾値を削除 - URDF由来の値を尊重
                # ただし、完全にゼロの場合は最小値を設定（MuJoCoの数値的安定性のため）
                ZERO_THRESHOLD = 1e-12
                if abs(I_body[0, 0]) < ZERO_THRESHOLD:
                    I_body[0, 0] = ZERO_THRESHOLD
                if abs(I_body[1, 1]) < ZERO_THRESHOLD:
                    I_body[1, 1] = ZERO_THRESHOLD
                if abs(I_body[2, 2]) < ZERO_THRESHOLD:
                    I_body[2, 2] = ZERO_THRESHOLD
                
                # Ensure symmetry again after threshold application
                I_body = 0.5 * (I_body + I_body.T)
                
                # ZERO_THRESHOLD適用後、三角不等式を再度チェック・修正
                # (対角成分の変更が三角不等式に影響する可能性があるため)
                Ixx = I_body[0, 0]
                Iyy = I_body[1, 1]
                Izz = I_body[2, 2]
                epsilon = 1e-8
                
                # 三角不等式をチェック・修正（簡易版）
                max_iterations = 5
                for iteration in range(max_iterations):
                    violations = 0
                    
                    if Ixx + Iyy < Izz - epsilon:
                        violations += 1
                        if Ixx + Iyy > 0:
                            ratio_xx = Ixx / (Ixx + Iyy)
                            ratio_yy = Iyy / (Ixx + Iyy)
                            target_sum = Izz + epsilon
                            Ixx = target_sum * ratio_xx
                            Iyy = target_sum * ratio_yy
                        else:
                            target_sum = Izz + epsilon
                            Ixx = target_sum / 2.0
                            Iyy = target_sum / 2.0
                    
                    if Iyy + Izz < Ixx - epsilon:
                        violations += 1
                        if Iyy + Izz > 0:
                            ratio_yy = Iyy / (Iyy + Izz)
                            ratio_zz = Izz / (Iyy + Izz)
                            target_sum = Ixx + epsilon
                            Iyy = target_sum * ratio_yy
                            Izz = target_sum * ratio_zz
                        else:
                            target_sum = Ixx + epsilon
                            Iyy = target_sum / 2.0
                            Izz = target_sum / 2.0
                    
                    if Izz + Ixx < Iyy - epsilon:
                        violations += 1
                        if Izz + Ixx > 0:
                            ratio_zz = Izz / (Izz + Ixx)
                            ratio_xx = Ixx / (Izz + Ixx)
                            target_sum = Iyy + epsilon
                            Izz = target_sum * ratio_zz
                            Ixx = target_sum * ratio_xx
                        else:
                            target_sum = Iyy + epsilon
                            Izz = target_sum / 2.0
                            Ixx = target_sum / 2.0
                    
                    if violations == 0:
                        break
                
                # 対角成分を更新
                I_body[0, 0] = Ixx
                I_body[1, 1] = Iyy
                I_body[2, 2] = Izz
                
                # 最終的な対称性確保
                I_body = 0.5 * (I_body + I_body.T)
                
                # 最終的な三角不等式チェック（デバッグ用）
                final_Ixx = I_body[0, 0]
                final_Iyy = I_body[1, 1]
                final_Izz = I_body[2, 2]
                final_epsilon = 1e-10  # MuJoCoの許容誤差より少し大きめ
                if not (final_Ixx + final_Iyy >= final_Izz - final_epsilon and
                        final_Iyy + final_Izz >= final_Ixx - final_epsilon and
                        final_Izz + final_Ixx >= final_Iyy - final_epsilon):
                    print(f"  ⚠ WARNING: Triangle inequality violated for {node_name} after all corrections!")
                    print(f"     Ixx={final_Ixx:.12e}, Iyy={final_Iyy:.12e}, Izz={final_Izz:.12e}")
                    print(f"     Ixx+Iyy={final_Ixx+final_Iyy:.12e} >= Izz={final_Izz:.12e}? {final_Ixx+final_Iyy >= final_Izz - final_epsilon}")
                    print(f"     Iyy+Izz={final_Iyy+final_Izz:.12e} >= Ixx={final_Ixx:.12e}? {final_Iyy+final_Izz >= final_Ixx - final_epsilon}")
                    print(f"     Izz+Ixx={final_Izz+final_Ixx:.12e} >= Iyy={final_Iyy:.12e}? {final_Izz+final_Ixx >= final_Iyy - final_epsilon}")
                    # 強制的に修正（最後の手段）
                    if final_Ixx + final_Iyy < final_Izz - final_epsilon:
                        target_sum = final_Izz + final_epsilon
                        if final_Ixx + final_Iyy > 0:
                            ratio = final_Ixx / (final_Ixx + final_Iyy)
                            final_Ixx = target_sum * ratio
                            final_Iyy = target_sum * (1.0 - ratio)
                        else:
                            final_Ixx = target_sum / 2.0
                            final_Iyy = target_sum / 2.0
                    if final_Iyy + final_Izz < final_Ixx - final_epsilon:
                        target_sum = final_Ixx + final_epsilon
                        if final_Iyy + final_Izz > 0:
                            ratio = final_Iyy / (final_Iyy + final_Izz)
                            final_Iyy = target_sum * ratio
                            final_Izz = target_sum * (1.0 - ratio)
                        else:
                            final_Iyy = target_sum / 2.0
                            final_Izz = target_sum / 2.0
                    if final_Izz + final_Ixx < final_Iyy - final_epsilon:
                        target_sum = final_Iyy + final_epsilon
                        if final_Izz + final_Ixx > 0:
                            ratio = final_Izz / (final_Izz + final_Ixx)
                            final_Izz = target_sum * ratio
                            final_Ixx = target_sum * (1.0 - ratio)
                        else:
                            final_Izz = target_sum / 2.0
                            final_Ixx = target_sum / 2.0
                    I_body[0, 0] = final_Ixx
                    I_body[1, 1] = final_Iyy
                    I_body[2, 2] = final_Izz
                    I_body = 0.5 * (I_body + I_body.T)
                    print(f"     Fixed: Ixx={I_body[0,0]:.12e}, Iyy={I_body[1,1]:.12e}, Izz={I_body[2,2]:.12e}")
                
                # 異常値検出と警告
                max_inertia = np.max(np.abs(np.diag(I_body)))
                if mass > 0 and max_inertia / mass > 10.0:  # 異常に大きな慣性（例: mass=0.03, inertia=0.01）
                    print(f"  ⚠ WARNING: Suspiciously large inertia for {node_name}")
                    print(f"     mass={mass:.6f}, max_inertia={max_inertia:.6e}, ratio={max_inertia/mass:.2f}")
                
                # Format for MJCF fullinertia (preferred over diaginertia)
                fullinertia_str = self._format_inertia_for_mjcf(I_body)
                mass_str = format_float_no_exp(mass)
                
                if is_target:
                    print(f"  Output fullinertia: {fullinertia_str}")
                    print(f"  Output mass: {mass_str}")
                    print(f"{'='*80}\n")
                
                # Output with fullinertia (includes off-diagonal terms)
                file.write(f'{indent_str}  <inertial pos="{com_pos}" mass="{mass_str}" fullinertia="{fullinertia_str}" />\n')
                has_inertial = True

        # ジオメトリ（メッシュ）
        if node in mesh_names:
            mesh_name = mesh_names[node]
            # 色情報
            color_str = "0.8 0.8 0.8 1.0"
            if hasattr(node, 'node_color') and node.node_color:
                r, g, b = node.node_color[:3]
                color_str = f"{r} {g} {b} 1.0"

            # 新しいヘルパー関数を使用
            self._write_mjcf_geom(file, node, mesh_name, color_str, indent_str)

        # 子ノードを処理
        for port_index, port in enumerate(node.output_ports()):
            for connected_port in port.connected_ports():
                child_node = connected_port.node()

                # Massless Decorationノードの場合、<geom class="visual">として追加してからスキップ
                if hasattr(child_node, 'massless_decoration') and child_node.massless_decoration:
                    if child_node in mesh_names:
                        dec_mesh_name = mesh_names[child_node]
                        # 色情報
                        dec_color_str = "0.8 0.8 0.8 1.0"
                        if hasattr(child_node, 'node_color') and child_node.node_color:
                            r, g, b = child_node.node_color[:3]
                            dec_color_str = f"{r} {g} {b} 1.0"

                        # ポイント座標を取得（装飾パーツの位置）
                        pos_str = "0 0 0"
                        if hasattr(node, 'points') and port_index < len(node.points):
                            point_data = node.points[port_index]
                            if 'xyz' in point_data:
                                xyz = point_data['xyz']
                                pos_str = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                        # Massless Decorationは<geom class="visual">を使用
                        file.write(f'{indent_str}  <geom class="visual" type="mesh" mesh="{dec_mesh_name}" rgba="{dec_color_str}" pos="{pos_str}"/>\n')
                    continue

                # Hide Meshにチェックが入っているノードはスキップ（ジョイント情報も作成しない）
                if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                    continue

                # 子ノードのジョイント情報を取得（port_indexは既にenumerateで取得済み）
                child_joint_info = self._get_joint_info(node, child_node, port_index, created_joints)

                # 再帰的に子ボディを出力
                self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent + 2, child_joint_info, fix_base_to_ground, used_body_names, used_joint_names)

        # moving body（jointを持つ）にinertialがない場合、自動で追加
        if is_moving_body and not has_inertial:
            # MuJoCoのmjMINVALより十分大きい値を設定（mass: 0.001, inertia: 1e-6）
            file.write(f'{indent_str}  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>\n')
            print(f"  Auto-added inertial to moving body ({node.name()}) to avoid MuJoCo load error")

        # body終了
        file.write(f'{indent_str}</body>\n')

    def _get_joint_info(self, parent_node, child_node, port_index, created_joints):
        """ジョイント情報を取得して辞書として返す"""
        joint_xyz = [0, 0, 0]
        joint_rpy = [0, 0, 0]
        joint_axis = [1, 0, 0]

        # ジョイント位置とRPYを取得
        # Use angle if available (UI-edited value), otherwise fallback to rpy
        if hasattr(parent_node, 'points') and port_index < len(parent_node.points):
            point_data = parent_node.points[port_index]
            joint_xyz = point_data.get('xyz', [0, 0, 0])
            joint_rpy = point_data.get('angle', point_data.get('rpy', [0, 0, 0]))

        # ジョイント軸を取得
        if hasattr(child_node, 'rotation_axis'):
            if child_node.rotation_axis == 0:
                joint_axis = [1, 0, 0]
            elif child_node.rotation_axis == 1:
                joint_axis = [0, 1, 0]
            elif child_node.rotation_axis == 2:
                joint_axis = [0, 0, 1]

        # ジョイントタイプ
        joint_type = "hinge"
        if hasattr(child_node, 'rotation_axis') and child_node.rotation_axis == 3:
            joint_type = "fixed"

        # fixedジョイントの場合はNoneを返す（出力しない）
        if joint_type == "fixed":
            return None

        # ジョイント名
        child_sanitized_name = self._sanitize_name(child_node.name())
        joint_name = f"{child_sanitized_name}_joint"
        motor_name = f"{child_sanitized_name}_motor"

        # ジョイント制限（Min Angle(deg)とMax Angle(deg)をrad変換した値）
        # MJCF <compiler angle="radian"> なので、radians で出力
        range_str = ""
        if hasattr(child_node, 'joint_lower') and hasattr(child_node, 'joint_upper'):
            lower = child_node.joint_lower  # radians で保存されている
            upper = child_node.joint_upper  # radians で保存されている
            # MJCF requires range[0] < range[1], so swap if needed
            if lower >= upper:
                # If lower >= upper, use default range or swap values
                # Default: ±π (3.14159 rad)
                if abs(lower - upper) < 1e-6:
                    # If they're equal, use default range
                    lower = -3.14159
                    upper = 3.14159
                    print(f"  Warning: Joint '{joint_name}' has equal lower/upper limits ({child_node.joint_lower:.6f}), using default range [-π, π]")
                else:
                    # Swap if lower > upper
                    lower, upper = upper, lower
                    print(f"  Warning: Joint '{joint_name}' has lower >= upper ({child_node.joint_lower:.6f} >= {child_node.joint_upper:.6f}), swapped to [{lower:.6f}, {upper:.6f}]")
            # Output as radians (already in radians)
            range_str = f' range="{format_float_no_exp(lower)} {format_float_no_exp(upper)}"'
        else:
            # If joint_lower/upper are not set, use default range for limited joints
            # This is especially important for closed-loop joints (_CL_joint)
            # Default: ±π radians
            lower = -3.14159
            upper = 3.14159
            range_str = f' range="{format_float_no_exp(lower)} {format_float_no_exp(upper)}"'
            print(f"  Warning: Joint '{joint_name}' has no joint_lower/upper limits, using default range [-π, π] radians")

        # limited: 常にtrue
        limited_str = ' limited="true"'

        # margin: marginの値
        margin_str = ""
        if hasattr(child_node, 'joint_margin'):
            margin_str = f' margin="{format_float_no_exp(child_node.joint_margin)}"'

        # armature: Armatureの値
        armature_str = ""
        if hasattr(child_node, 'joint_armature'):
            armature_str = f' armature="{format_float_no_exp(child_node.joint_armature)}"'

        # frictionloss: Frictionlossの値
        frictionloss_str = ""
        if hasattr(child_node, 'joint_frictionloss'):
            frictionloss_str = f' frictionloss="{format_float_no_exp(child_node.joint_frictionloss)}"'

        # damping: dampingの値
        damping_str = ""
        if hasattr(child_node, 'joint_damping'):
            damping_str = f' damping="{format_float_no_exp(child_node.joint_damping)}"'
        
        # stiffness: stiffnessの値（MJCFではジョイント要素の属性）
        # Note: joint stiffness と actuator kp の相互作用を避けるため、joint stiffness は出力しない
        # actuator の kp で制御する
        stiffness_str = ""
        # if hasattr(child_node, 'joint_stiffness') and child_node.joint_stiffness > 0:
        #     stiffness_str = f' stiffness="{format_float_no_exp(child_node.joint_stiffness)}"'
        
        # ref: body_angleからref属性を生成（デフォルト姿勢/参照角度）
        # body_angleは[x_rad, y_rad, z_rad]の形式で、rotation_axisに対応する軸のみが非ゼロ
        # MJCF では <compiler angle="radian"> が設定されているため、ref も radians で出力
        ref_str = ""
        if hasattr(child_node, 'body_angle') and hasattr(child_node, 'rotation_axis'):
            body_angle = child_node.body_angle
            rotation_axis = child_node.rotation_axis
            
            # rotation_axisに対応する軸の角度を取得（radians のまま出力）
            if rotation_axis in [0, 1, 2] and any(a != 0.0 for a in body_angle):
                ref_angle_rad = body_angle[rotation_axis]
                if abs(ref_angle_rad) > 1e-6:  # ゼロでない場合のみ出力
                    ref_str = f' ref="{format_float_no_exp(ref_angle_rad)}"'

        # 作成されたジョイントをリストに追加（actuator用）
        joint_effort = getattr(child_node, 'joint_effort', 10.0)
        joint_stiffness = getattr(child_node, 'joint_stiffness', 100.0)
        joint_damping = getattr(child_node, 'joint_damping', 15.0)
        # range情報を抽出（ctrlrange用）
        # MJCF <compiler angle="radian"> なので、radians で出力
        range_values = None
        if hasattr(child_node, 'joint_lower') and hasattr(child_node, 'joint_upper'):
            lower = child_node.joint_lower  # radians で保存されている
            upper = child_node.joint_upper  # radians で保存されている
            # MJCF requires range[0] < range[1], so swap if needed
            if lower >= upper:
                # If lower >= upper, use default range or swap values
                # Default: ±π radians
                if abs(lower - upper) < 1e-6:
                    # If they're equal, use default range
                    lower = -3.14159
                    upper = 3.14159
                else:
                    # Swap if lower > upper
                    lower, upper = upper, lower
            # Output as radians (already in radians)
            range_values = (lower, upper)
        else:
            # If joint_lower/upper are not set, use default range
            # This is especially important for closed-loop joints (_CL_joint)
            # Default: ±π radians
            range_values = (-3.14159, 3.14159)
        created_joints.append({
            'joint_name': joint_name,
            'motor_name': motor_name,
            'effort': joint_effort,
            'stiffness': joint_stiffness,
            'damping': joint_damping,
            'range': range_str,
            'range_values': range_values
        })

        return {
            'name': joint_name,
            'type': joint_type,
            'pos': f"{joint_xyz[0]} {joint_xyz[1]} {joint_xyz[2]}",
            'rpy': joint_rpy,  # RPY情報を追加（body orientation用）
            'axis': f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}",
            'range': range_str,
            'limited': limited_str,
            'margin': margin_str,
            'armature': armature_str,
            'frictionloss': frictionloss_str,
            'damping': damping_str,
            'stiffness': stiffness_str,
            'ref': ref_str
        }

    def calculate_inertia_tensor_for_mirrored(self, poly_data, mass, center_of_mass):
        """
        ミラーリングされたモデルの慣性テンソルを計算。
        CustomNodeGraphクラスのメソッド。
        """
        try:
            print("\nCalculating inertia tensor for mirrored model...")
            print(f"Mass: {mass:.6f}")
            print(f"Center of Mass (before mirroring): {center_of_mass}")

            # Y座標を反転した重心を使用
            mirrored_com = [center_of_mass[0], -center_of_mass[1], center_of_mass[2]]
            print(f"Center of Mass (after mirroring): {mirrored_com}")

            # 慣性テンソルを計算（ミラーリングモードで）
            # utils.pyの calculate_inertia_tensor を直接使用
            inertia_tensor = calculate_inertia_tensor(
                poly_data, mass, mirrored_com, is_mirrored=True)

            print("\nMirrored model inertia tensor calculated successfully")
            return inertia_tensor

        except Exception as e:
            print(f"Error calculating mirrored inertia tensor: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# ユーティリティ関数
def load_project(graph):
    """プロジェクトを読み込み"""
    try:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 
            "Load Project", 
            "", 
            "XML Files (*.xml)"
        )
        
        if not file_path:
            print("Load cancelled")
            return

        project_base_dir = os.path.dirname(file_path)
        print(f"Project base directory: {project_base_dir}")

        # XMLファイルを解析
        tree = ET.parse(file_path)
        root = tree.getroot()

        # meshesディレクトリのパスを取得
        meshes_dir = None
        meshes_dir_elem = root.find("meshes_dir")
        if meshes_dir_elem is not None and meshes_dir_elem.text:
            meshes_dir = os.path.normpath(os.path.join(project_base_dir, meshes_dir_elem.text))
            if not os.path.exists(meshes_dir):
                # meshesディレクトリが見つからない場合、ユーザーに選択を求める
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Question)
                msg.setText("Meshes directory not found")
                msg.setInformativeText("Would you like to select the meshes directory location?")
                msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
                
                if msg.exec() == QtWidgets.QMessageBox.Yes:
                    meshes_dir = QtWidgets.QFileDialog.getExistingDirectory(
                        None,
                        "Select Meshes Directory",
                        project_base_dir,
                        QtWidgets.QFileDialog.ShowDirsOnly
                    )
                    if not meshes_dir:
                        print("Meshes directory selection cancelled")
                        meshes_dir = None
                    else:
                        print(f"Selected meshes directory: {meshes_dir}")

        # 現在のグラフをクリア
        graph.clear_graph()

        # プロジェクトファイルを読み込み
        success = graph.load_project(file_path)

        if success:
            print("Project loaded, resolving STL paths...")
            # STLファイルを読み込む必要があるノードのリストを作成
            nodes_to_load_stl = []
            for node in graph.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    nodes_to_load_stl.append(node)
            
            total_stl_loads = len(nodes_to_load_stl)
            print(f"Total STL files to load: {total_stl_loads}")
            
            # ノード読み込みが完了した時点での進捗を取得
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer and hasattr(graph.stl_viewer, 'progress_bar'):
                current_progress = graph.stl_viewer.progress_bar.value
                print(f"Current progress after node loading: {current_progress}%")
            else:
                current_progress = 100  # デフォルト値
            
            # STLファイルのパスを解決して読み込む
            for stl_index, node in enumerate(nodes_to_load_stl, 1):
                if hasattr(node, 'stl_file') and node.stl_file:
                    try:
                        stl_path = node.stl_file
                        if not os.path.isabs(stl_path):
                            # まずmeshesディレクトリからの相対パスを試す
                            if meshes_dir:
                                abs_stl_path = os.path.normpath(os.path.join(meshes_dir, stl_path))
                                if os.path.exists(abs_stl_path):
                                    node.stl_file = abs_stl_path
                                    print(f"Found STL file in meshes dir for node {node.name()}: {abs_stl_path}")
                                    if graph.stl_viewer:
                                        # 進捗バーを更新（STL読み込み時）
                                        if total_stl_loads > 0 and hasattr(graph.stl_viewer, 'progress_bar'):
                                            # ノード読み込み後の進捗から、STL読み込みの進捗を減算
                                            # current_progressから0%までをSTL読み込みで消費
                                            stl_progress = int((stl_index / total_stl_loads) * current_progress)
                                            remaining_percent = current_progress - stl_progress
                                            graph.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                                            QtWidgets.QApplication.processEvents()
                                        graph.stl_viewer.load_stl_for_node(node, show_progress=False)  # 個別の進捗バーは非表示
                                    continue

                            # プロジェクトディレクトリからの相対パスを試す
                            abs_stl_path = os.path.normpath(os.path.join(project_base_dir, stl_path))
                            if os.path.exists(abs_stl_path):
                                node.stl_file = abs_stl_path
                                print(f"Found STL file in project dir for node {node.name()}: {abs_stl_path}")
                                if graph.stl_viewer:
                                    # 進捗バーを更新（STL読み込み時）
                                    if total_stl_loads > 0 and hasattr(graph.stl_viewer, 'progress_bar'):
                                        stl_progress = int((stl_index / total_stl_loads) * current_progress)
                                        remaining_percent = current_progress - stl_progress
                                        graph.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                                        QtWidgets.QApplication.processEvents()
                                    graph.stl_viewer.load_stl_for_node(node, show_progress=False)
                            else:
                                print(f"Warning: Could not find STL file for node {node.name()}: {stl_path}")
                        else:
                            if os.path.exists(stl_path):
                                print(f"Using absolute STL path for node {node.name()}: {stl_path}")
                                if graph.stl_viewer:
                                    # 進捗バーを更新（STL読み込み時）
                                    if total_stl_loads > 0 and hasattr(graph.stl_viewer, 'progress_bar'):
                                        stl_progress = int((stl_index / total_stl_loads) * current_progress)
                                        remaining_percent = current_progress - stl_progress
                                        graph.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                                        QtWidgets.QApplication.processEvents()
                                    graph.stl_viewer.load_stl_for_node(node, show_progress=False)
                            else:
                                print(f"Warning: STL file not found: {stl_path}")

                    except Exception as e:
                        print(f"Error resolving STL path for node {node.name()}: {str(e)}")
                        traceback.print_exc()

            print(f"Project loaded successfully from: {file_path}")
            
            # 進捗バーを非表示（すべての処理が完了）
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
                graph.stl_viewer.progress_bar.setValue(0)
                QtWidgets.QApplication.processEvents()
                from PySide6.QtCore import QTimer
                QTimer.singleShot(200, lambda: graph.stl_viewer.show_progress(False))

            # 位置を再計算
            graph.recalculate_all_positions()
            
            # STL読み込み完了後、すべてのノードのカラーを3Dビューに適用（ノードを開いて閉じた時と同じ効果）
            print("\n[DEBUG] Applying colors to 3D view after STL loading...")
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
                all_nodes = graph.all_nodes()
                print(f"[DEBUG] Total nodes: {len(all_nodes)}")
                applied_count = 0
                skipped_count = 0
                
                for node in all_nodes:
                    try:
                        node_name = node.name()
                        has_stl_file = hasattr(node, 'stl_file') and node.stl_file
                        in_actors = node in graph.stl_viewer.stl_actors
                        has_node_color = hasattr(node, 'node_color') and node.node_color
                        
                        print(f"[DEBUG] Node '{node_name}': has_stl_file={has_stl_file}, in_actors={in_actors}, has_node_color={has_node_color}")
                        
                        # ノードにSTLファイルが読み込まれている場合のみカラーを適用
                        if has_stl_file and in_actors:
                            # node.node_colorを確認
                            if has_node_color:
                                rgba_values = node.node_color
                                # RGBA値を0-1の範囲に正規化
                                rgba_values = [max(0.0, min(1.0, float(v))) for v in rgba_values[:4]]
                                
                                actor = graph.stl_viewer.stl_actors[node]
                                
                                # マッパーを取得して、スカラー値（頂点カラーや面カラー）があるかチェック
                                mapper = actor.GetMapper()
                                has_scalars = False
                                if mapper and mapper.GetInput():
                                    polydata = mapper.GetInput()
                                    has_vertex_colors = polydata.GetPointData().GetScalars() is not None
                                    has_face_colors = polydata.GetCellData().GetScalars() is not None
                                    has_scalars = has_vertex_colors or has_face_colors
                                
                                if has_scalars:
                                    # スカラー値がある場合は、透明度のみ設定
                                    if len(rgba_values) >= 4:
                                        actor.GetProperty().SetOpacity(rgba_values[3])
                                    else:
                                        actor.GetProperty().SetOpacity(1.0)
                                    print(f"[DEBUG] Node '{node_name}' has vertex/face colors, only opacity applied: {rgba_values[3] if len(rgba_values) >= 4 else 1.0}")
                                else:
                                    # スカラー値がない場合は、通常通りノードの色を適用
                                    # RGB設定（最初の3要素のみ）
                                    actor.GetProperty().SetColor(*rgba_values[:3])
                                    # Alpha設定（4番目の要素があれば）
                                    if len(rgba_values) >= 4:
                                        actor.GetProperty().SetOpacity(rgba_values[3])
                                    else:
                                        actor.GetProperty().SetOpacity(1.0)
                                    print(f"[DEBUG] Applied color to node '{node_name}': RGBA{rgba_values[:4]}")
                                    applied_count += 1
                            else:
                                # デフォルトの白色を適用
                                actor = graph.stl_viewer.stl_actors[node]
                                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                                actor.GetProperty().SetOpacity(1.0)
                                print(f"[DEBUG] Applied default white color to node '{node_name}'")
                                applied_count += 1
                        else:
                            skipped_count += 1
                            if not has_stl_file:
                                print(f"[DEBUG] Skipped node '{node_name}': no STL file")
                            elif not in_actors:
                                print(f"[DEBUG] Skipped node '{node_name}': not in stl_actors")
                    except Exception as e:
                        print(f"[DEBUG] Warning: Failed to apply color to node '{node.name()}': {e}")
                        import traceback
                        traceback.print_exc()
                
                # 3Dビューを最終更新
                graph.stl_viewer.render_to_image()
                print(f"[DEBUG] Colors applied: {applied_count} nodes, {skipped_count} skipped")
                print("[DEBUG] Colors applied to 3D view after STL loading")

            # 3Dビューをリセット後、hide_mesh状態を再適用
            if graph.stl_viewer:
                def reset_and_apply_hide():
                    graph.stl_viewer.reset_view_to_fit()
                    # すべてのノードのhide_mesh状態を再適用
                    for node in graph.all_nodes():
                        if hasattr(node, 'hide_mesh') and node.hide_mesh:
                            if node in graph.stl_viewer.stl_actors:
                                actor = graph.stl_viewer.stl_actors[node]
                                actor.SetVisibility(False)
                                print(f"Re-applied hide_mesh after view reset: {node.name()}")
                    graph.stl_viewer.render_to_image()

                QtCore.QTimer.singleShot(500, reset_and_apply_hide)

        else:
            print("Failed to load project")

    except Exception as e:
        print(f"Error loading project: {str(e)}")
        traceback.print_exc()

def delete_selected_node(graph):
    selected_nodes = graph.selected_nodes()
    if selected_nodes:
        for node in selected_nodes:
            # BaseLinkNodeは削除不可
            if isinstance(node, BaseLinkNode):
                print("Cannot delete Base Link node")
                continue

            # 3DビューからSTLメッシュを削除
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
                if node in graph.stl_viewer.stl_actors:
                    actor = graph.stl_viewer.stl_actors[node]
                    graph.stl_viewer.renderer.RemoveActor(actor)
                    del graph.stl_viewer.stl_actors[node]
                    print(f"Removed STL mesh for node: {node.name()}")
                    # 3Dビューを更新
                    graph.stl_viewer.render_to_image()

            # グラフからノードを削除
            graph.remove_node(node)
        print(f"Deleted {len(selected_nodes)} node(s)")
    else:
        print("No node selected for deletion")

def show_settings_dialog(graph, parent=None):
    """設定ダイアログを表示"""
    dialog = SettingsDialog(graph, parent)
    result = dialog.exec_()

    # 設定が適用された場合、コライダー表示を更新
    if result == QtWidgets.QDialog.Accepted:
        if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
            stl_viewer = graph.stl_viewer
            # コライダー表示が有効な場合、再表示して色を反映
            if hasattr(stl_viewer, 'collider_display_enabled') and stl_viewer.collider_display_enabled:
                print("Settings updated - refreshing collider display...")
                stl_viewer.show_all_colliders()
                stl_viewer.render_to_image()
                print("Collider display refreshed with new collision color")

def open_importer_window(graph):
    """Model Importerウィンドウを開く"""
    # graphオブジェクトにウィンドウの参照を保持（ガベージコレクション防止）
    if not hasattr(graph, 'importer_window') or graph.importer_window is None:
        graph.importer_window = ImporterWindow(graph)

    # ウィンドウを表示して前面に
    graph.importer_window.show()
    graph.importer_window.raise_()
    graph.importer_window.activateWindow()

def cleanup_and_exit():
    """アプリケーションのクリーンアップと終了"""
    print("Cleaning up application resources...")
    try:
        # グラフのクリーンアップ
        if 'graph' in globals():
            try:
                graph.cleanup()
            except Exception as e:
                print(f"Error cleaning up graph: {str(e)}")

        # STLビューアのクリーンアップ
        if 'stl_viewer' in globals():
            try:
                stl_viewer.cleanup()
            except Exception as e:
                print(f"Error cleaning up STL viewer: {str(e)}")

        # 全てのウィンドウを閉じて削除
        app = QtWidgets.QApplication.instance()
        if app:
            for window in QtWidgets.QApplication.topLevelWidgets():
                try:
                    if window and window.isVisible():
                        window.close()
                        window.deleteLater()
                except Exception as e:
                    print(f"Error closing window: {str(e)}")

    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
    finally:
        print("Cleanup completed.")

def signal_handler(_signum, _frame):
    """Ctrl+Cシグナルのハンドラ"""
    print("\nCtrl+C detected, closing application...")

    def do_shutdown():
        """実際のシャットダウン処理（Qtイベントループ内で実行）"""
        try:
            app = QtWidgets.QApplication.instance()
            if app:
                # 全てのウィンドウを閉じる
                for window in QtWidgets.QApplication.topLevelWidgets():
                    try:
                        if window and window.isVisible():
                            window.close()
                            window.deleteLater()
                    except Exception as e:
                        print(f"Error closing window: {str(e)}")

                # アプリケーションの終了を要求
                QtCore.QTimer.singleShot(100, app.quit)

                # さらに100ms後に強制終了
                QtCore.QTimer.singleShot(200, lambda: sys.exit(0))
        except Exception as e:
            print(f"Error during application shutdown: {str(e)}")
            sys.exit(0)

    # Qtイベントループ内でシャットダウンを実行
    try:
        if QtWidgets.QApplication.instance():
            QtCore.QTimer.singleShot(0, do_shutdown)
        else:
            sys.exit(0)
    except Exception:
        sys.exit(0)

def center_window_top_left(window):
    """ウィンドウを画面の左上に配置"""
    window.move(0, 0)


if __name__ == '__main__':
    try:
        # Ctrl+Cシグナルハンドラの設定（カスタムシャットダウンロジックのため個別実装）
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app = QtWidgets.QApplication(sys.argv)
        setup_dark_theme(app, theme='assembler')

        # アプリケーション終了時のクリーンアップ設定
        app.aboutToQuit.connect(cleanup_and_exit)

        # シグナル処理用タイマー（utils関数使用）
        timer = setup_signal_processing_timer(app)

        # メインウィンドウの作成
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("URDF Kitchen - Assembler v0.1.0 -")
        main_window.resize(1200, 600)

        # セントラルウィジェットの設定
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # STLビューアとグラフの設定（先に作成）
        stl_viewer = STLViewerWidget(central_widget)
        stl_viewer.setMinimumWidth(100)  # 3Dビューの最小幅を100pxに設定
        graph = CustomNodeGraph(stl_viewer)
        stl_viewer.graph = graph  # STLビューアにグラフへの参照を設定
        graph.setup_custom_view()

        # base_linkノードの作成
        base_node = graph.create_base_link()

        # 左パネルの設定
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(145)
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # 名前入力フィールドの設定
        name_label = QtWidgets.QLabel("Name:")
        left_layout.addWidget(name_label)
        name_input = QtWidgets.QLineEdit("robot_x")
        name_input.setFixedWidth(120)
        name_input.setStyleSheet("QLineEdit { padding-left: 3px; padding-top: 0px; padding-bottom: 0px; }")
        left_layout.addWidget(name_input)

        # 名前入力フィールドとグラフを接続（graphが定義された後に接続）
        name_input.textChanged.connect(graph.update_robot_name)

        # ボタンの作成と設定
        buttons = {
            "--spacer1--": None,  # スペーサー用のダミーキー
            "Import XMLs": None,
            "Import MODEL": None,
            "--spacer2--": None,  # スペーサー用のダミーキー
            "Add Node": None,
            "Delete Node": None,
            "Check Inertia": None,
            "Build r_ from l_": None,
            "Recalc Positions": None,
            "--spacer3--": None,  # スペーサー用のダミーキー
            "Load Project": None,
            "Save Project": None,
            "--spacer4--": None,  # スペーサー用のダミーキー
            "Export URDF": None,
            "Export for Unity": None,
            "Export MJCF": None,
            "--spacer5--": None,  # スペーサー用のダミーキー
            "open urdf-loaders": None,
            "Settings": None
        }

        # 統一されたボタンスタイル（グローバル定数を使用）
        button_style = UNIFIED_BUTTON_STYLE

        for button_text in buttons.keys():
            if button_text.startswith("--spacer"):
                # スペーサーの追加
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(1)  # スペースの高さを1ピクセルに設定
                left_layout.addWidget(spacer)
            else:
                # 通常のボタンの追加
                button = QtWidgets.QPushButton(button_text)
                button.setFixedWidth(120)
                button.setStyleSheet(button_style)  # 統一スタイルを適用
                left_layout.addWidget(button)
                buttons[button_text] = button

        left_layout.addStretch()

        # ボタンのコネクション設定
        buttons["Import XMLs"].clicked.connect(graph.import_xmls_from_folder)
        buttons["Add Node"].clicked.connect(
            lambda: graph.create_node(
                'insilico.nodes.FooNode',
                name=f'Node_{len(graph.all_nodes())}',
                pos=QtCore.QPointF(0, 0)
            )
        )
        buttons["Delete Node"].clicked.connect(
            lambda: delete_selected_node(graph))
        buttons["Check Inertia"].clicked.connect(
            lambda: graph.check_all_inertia())
        buttons["Build r_ from l_"].clicked.connect(
            graph.build_r_from_l)
        buttons["Recalc Positions"].clicked.connect(
            graph.recalculate_all_positions)
        buttons["Save Project"].clicked.connect(graph.save_project)
        buttons["Load Project"].clicked.connect(lambda: load_project(graph))
        buttons["Import MODEL"].clicked.connect(lambda: open_importer_window(graph))
        buttons["Export URDF"].clicked.connect(lambda: graph.export_urdf())
        buttons["Export for Unity"].clicked.connect(graph.export_for_unity)
        buttons["Export MJCF"].clicked.connect(graph.export_mjcf)
        buttons["open urdf-loaders"].clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl(
                    "https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/")
            )
        )
        buttons["Settings"].clicked.connect(
            lambda: show_settings_dialog(graph, main_window))

        # スプリッターの設定（3パネル：左サイドバー、中央グラフ、右3Dビューア）
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(graph.widget)
        splitter.addWidget(stl_viewer)
        # 初期サイズ: 左パネル200px、中央700px、右300px
        splitter.setSizes([200, 700, 300])

        # メインレイアウトの設定
        main_layout.addWidget(splitter)

        # セントラルウィジェットの設定
        main_window.setCentralWidget(central_widget)

        # グラフに名前入力フィールドを関連付け
        graph.name_input = name_input

        # ウィンドウを画面の左上に配置して表示
        center_window_top_left(main_window)
        main_window.show()

        print("Application started. Double-click on a node to open the inspector.")
        print("Click 'Add Node' button to add new nodes.")
        print("Select a node and click 'Delete Node' to remove it.")
        print("Use 'Save' and 'Load' buttons to save and load your project.")
        print("Press Ctrl+C in the terminal to close all windows and exit.")

        # タイマーの設定（シグナル処理のため）
        timer = QtCore.QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)
        
        # アプリケーションの実行
        sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        cleanup_and_exit()
        sys.exit(1)
