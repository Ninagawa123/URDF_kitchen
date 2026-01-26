"""
File Name: urdf_kitchen_Assembler.py
Description: A Python script to assembling files configured with urdf_kitchen_PartsEditor.py into a URDF file.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Update.     : Jan 25, 2026
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

        # Load Collider
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

        # Get path STL
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

        # Get PartsEditor path PartsEditor
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
        port_layout.addSpacing(2)  # coordinatesanglebetweenspacing

        # Angle angle
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
            # Confirm mesh_original_color
            if not hasattr(self.current_node, 'mesh_original_color') or self.current_node.mesh_original_color is None:
                print(f"Warning: No original mesh color found for node '{self.current_node.name()}'")
                return

            # Get mesh_original_color
            original_color = self.current_node.mesh_original_color

            # Convert to RGBA format (add Alpha=1.0 if 3 elements)
            if len(original_color) == 3:
                rgba_values = list(original_color) + [1.0]
            else:
                rgba_values = list(original_color[:4])  # Max4elementsup to

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

            # Update Look CoM enable if Look CoM 3D
            if hasattr(self, 'look_inertial_origin_toggle') and self.look_inertial_origin_toggle.isChecked():
                if self.stl_viewer:
                    self.stl_viewer.show_inertial_origin(self.current_node, origin_xyz)
                    self.stl_viewer.render_to_image()

        except ValueError:
            pass  # Ignore invalid values

    def update_inertia(self):
        """inertia tensortextupdate(real-time + Enter key)"""
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
            pass  # Ignore invalid values

    def update_joint_params(self):
        """jointparameterstextupdate(real-time + Enter key)"""
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
            pass  # Ignore invalid values

    def update_joint_limits_realtime(self):
        """jointlimitstextreal-timeupdate"""
        if not self.current_node:
            return
        try:
            # Save Lower limit Degree Radian transform Lower Degree Radian
            lower_text = self.lower_limit_input.text()
            if lower_text:
                self.current_node.joint_lower = math.radians(float(lower_text))
            # Save Upper limit Degree Radian transform Upper Degree Radian
            upper_text = self.upper_limit_input.text()
            if upper_text:
                self.current_node.joint_upper = math.radians(float(upper_text))
        except ValueError:
            pass  # Ignore invalid values

    def update_body_angle(self):
        """Body Angletextreal-timeupdate"""
        if not self.current_node:
            return
        try:
            # Get X Y Z input save X Y Z
            angle_x_deg = float(self.angle_x_input.text()) if self.angle_x_input.text() else 0.0
            angle_y_deg = float(self.angle_y_input.text()) if self.angle_y_input.text() else 0.0
            angle_z_deg = float(self.angle_z_input.text()) if self.angle_z_input.text() else 0.0

            # Degree radian degree radian
            self.current_node.body_angle = [math.radians(angle_x_deg), math.radians(angle_y_deg), math.radians(angle_z_deg)]

            # Update out Ang
            if hasattr(self.current_node, 'graph'):
                graph = self.current_node.graph
                # Find parent from node input port
                for input_port in self.current_node.input_ports():
                    connected_ports = input_port.connected_ports()
                    if connected_ports:
                        parent_node = connected_ports[0].node()
                        parent_port_name = connected_ports[0].name()

                        # Compute point index from port name (out_1->0, out_2->1, etc.)
                        point_index = 0  # Default
                        if parent_port_name.startswith('out_'):
                            try:
                                port_num = int(parent_port_name.split('_')[1])
                                point_index = port_num - 1
                            except (ValueError, IndexError):
                                pass
                        elif parent_port_name == 'out':
                            point_index = 0

                        # Update points save angle
                        if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                            if 'angle' not in parent_node.points[point_index]:
                                parent_node.points[point_index]['angle'] = [0.0, 0.0, 0.0]
                            parent_node.points[point_index]['angle'] = [math.radians(angle_x_deg), math.radians(angle_y_deg), math.radians(angle_z_deg)]
                            print(f"Updated parent node {parent_node.name()} port {point_index+1} angle to [{angle_x_deg}, {angle_y_deg}, {angle_z_deg}] degrees")
                        break

            # Update 3D
            if self.stl_viewer:
                self.stl_viewer.render_to_image()
        except ValueError:
            pass  # Ignore invalid values

    def moveEvent(self, event):
        """windowtexteventtexthandle"""
        super(InspectorWindow, self).moveEvent(event)
        # Last_inspector_position if
        if hasattr(self, 'graph') and self.graph:
            self.graph.last_inspector_position = self.pos()

    def keyPressEvent(self, event):
        """keytexteventtexthandle"""
        # Confirm ESC
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        # Cmd+w macos ctrl+w windows/linux cmd+w macos ctrl+w windows/linux
        elif event.key() == QtCore.Qt.Key.Key_W and (
            event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or
            event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier
        ):
            self.close()
        else:
            # Todo
            super(InspectorWindow, self).keyPressEvent(event)

    def start_rotation_test(self):
        """textstart"""
        if self.current_node and self.stl_viewer:
            # Get Inherit to Inherit Subnodes
            follow = self.follow_checkbox.isChecked()
            self.stl_viewer.follow_children = follow

            # Save current transform
            self.stl_viewer.store_current_transform(self.current_node)
            # Todo
            self.stl_viewer.start_rotation_test(self.current_node)

    def stop_rotation_test(self):
        """textend"""
        if self.current_node and self.stl_viewer:
            # Angle
            self.stl_viewer.stop_rotation_test(self.current_node)

    def look_lower_limit(self):
        """Lower limittextangletextdisplay"""
        if self.current_node and self.stl_viewer:
            try:
                # Get/set "Inherit to Subnodes" checkbox state
                follow = self.follow_checkbox.isChecked()
                self.stl_viewer.follow_children = follow

                # Get value Degree
                lower_text = self.lower_limit_input.text()
                if not lower_text:
                    lower_text = self.lower_limit_input.placeholderText()

                lower_deg = float(lower_text)
                lower_rad = math.radians(lower_deg)

                # Save current transform transform
                self.stl_viewer.store_current_transform(self.current_node)
                # Show TODO
                self.stl_viewer.show_angle(self.current_node, lower_rad)
            except ValueError:
                pass

    def look_upper_limit(self):
        """Upper limittextangletextdisplay"""
        if self.current_node and self.stl_viewer:
            try:
                # Get/set "Inherit to Subnodes" checkbox state
                follow = self.follow_checkbox.isChecked()
                self.stl_viewer.follow_children = follow

                # Get value Degree
                upper_text = self.upper_limit_input.text()
                if not upper_text:
                    upper_text = self.upper_limit_input.placeholderText()

                upper_deg = float(upper_text)
                upper_rad = math.radians(upper_deg)

                # Save current transform
                self.stl_viewer.store_current_transform(self.current_node)
                # Show TODO
                self.stl_viewer.show_angle(self.current_node, upper_rad)
            except ValueError:
                pass

    def look_zero_limit(self):
        """0textangletextdisplay"""
        if self.current_node and self.stl_viewer:
            # Get/set "Inherit to Subnodes" checkbox state
            follow = self.follow_checkbox.isChecked()
            self.stl_viewer.follow_children = follow

            # Save current transform
            self.stl_viewer.store_current_transform(self.current_node)
            # Show 0
            self.stl_viewer.show_angle(self.current_node, 0.0)

    def toggle_inertial_origin_view(self, checked):
        """Inertial Origintextdisplay/textdisplaytext"""
        if self.current_node and self.stl_viewer:
            if checked:
                # Get Inertial Inertial Origin
                try:
                    x = float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0
                    y = float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0
                    z = float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0

                    # Show 3D
                    self.stl_viewer.show_inertial_origin(self.current_node, [x, y, z])
                except ValueError:
                    self.look_inertial_origin_toggle.setChecked(False)
            else:
                # Hide TODO
                self.stl_viewer.hide_inertial_origin(self.current_node)

    def set_joint_limits(self):
        """Joint limitstextnodetextsave"""
        if not self.current_node:
            return

        try:
            # Save Lower limit Degree Radian transform Lower Degree Radian
            lower_text = self.lower_limit_input.text()
            if lower_text:
                self.current_node.joint_lower = math.radians(float(lower_text))

            # Save Upper limit Degree Radian transform Upper Degree Radian
            upper_text = self.upper_limit_input.text()
            if upper_text:
                self.current_node.joint_upper = math.radians(float(upper_text))

            # Save Effort Effort
            effort_text = self.effort_input.text()
            if effort_text:
                self.current_node.joint_effort = float(effort_text)

            # Save Velocity Velocity
            velocity_text = self.velocity_input.text()
            if velocity_text:
                self.current_node.joint_velocity = float(velocity_text)

            # Save Damping Damping
            damping_text = self.damping_input.text()
            if damping_text:
                self.current_node.joint_damping = float(damping_text)

            # Save Stiffness Stiffness
            stiffness_text = self.stiffness_input.text()
            if stiffness_text:
                self.current_node.joint_stiffness = float(stiffness_text)

            # Save Margin Margin
            margin_text = self.margin_input.text()
            if margin_text:
                self.current_node.joint_margin = float(margin_text)

            # Save Armature Armature
            armature_text = self.armature_input.text()
            if armature_text:
                self.current_node.joint_armature = float(armature_text)

            # Save Frictionloss Frictionloss
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
        """InertiatextInertial Origintextnodetextsave"""
        if not self.current_node:
            return

        try:
            # Get Inertial Inertial Origin
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

            # Get Inertia
            inertia_values = {
                'ixx': float(self.ixx_input.text()) if self.ixx_input.text() else 0.0,
                'ixy': float(self.ixy_input.text()) if self.ixy_input.text() else 0.0,
                'ixz': float(self.ixz_input.text()) if self.ixz_input.text() else 0.0,
                'iyy': float(self.iyy_input.text()) if self.iyy_input.text() else 0.0,
                'iyz': float(self.iyz_input.text()) if self.iyz_input.text() else 0.0,
                'izz': float(self.izz_input.text()) if self.izz_input.text() else 0.0
            }

            # Save node
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
        """STLtexttrimeshtextCenter of Masstext"""
        if not self.current_node:
            QtWidgets.QMessageBox.warning(
                self,
                "No Node Selected",
                "Please select a node first."
            )
            return

        # Confirm STL
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
            # Load Trimesh mesh Trimesh
            print(f"\n=== Recalculating Center of Mass ===")
            mesh = trimesh.load(stl_path)

            # Mesh
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            print(f"Mesh loaded successfully")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Volume: {mesh.volume:.6f}")
            print(f"  Is watertight: {mesh.is_watertight}")

            # Mesh
            repair_performed = False
            original_watertight = mesh.is_watertight

            if not mesh.is_watertight:
                print("\n⚠ Mesh is not watertight. Attempting automatic repair...")

                # Mesh
                try:
                    # Todo
                    try:
                        print("  - Fixing normals...")
                        mesh.fix_normals()
                    except AttributeError:
                        print("  - Skipping normals fixing (method not available)")

                    # Remove trimesh
                    try:
                        print("  - Removing duplicate faces...")
                        mesh.remove_duplicate_faces()
                    except AttributeError:
                        print("  - Skipping duplicate faces removal (method not available)")

                    # Remove TODO
                    try:
                        print("  - Removing degenerate faces...")
                        mesh.remove_degenerate_faces()
                    except AttributeError:
                        print("  - Skipping degenerate faces removal (method not available)")

                    # Todo
                    try:
                        print("  - Filling holes...")
                        mesh.fill_holes()
                    except AttributeError:
                        print("  - Skipping holes filling (method not available)")

                    repair_performed = True

                    # Confirm TODO
                    print(f"\ntext:")
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

            # Compute Center of Mass Center Mass
            center_of_mass = mesh.center_mass
            print(f"\nCalculated center of mass: {center_of_mass}")

            # Set UI
            self.inertial_x_input.setText(format_float_no_exp(center_of_mass[0]))
            self.inertial_y_input.setText(format_float_no_exp(center_of_mass[1]))
            self.inertial_z_input.setText(format_float_no_exp(center_of_mass[2]))

            # Todo
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
        """STLtexttrimeshtext(text)"""
        if not self.current_node:
            QtWidgets.QMessageBox.warning(
                self,
                "No Node Selected",
                "Please select a node first."
            )
            return

        # Confirm STL
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

        # Get TODO
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

        # Get existing Inertial value Inertial Origin
        try:
            center_of_mass = [
                float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0,
                float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0,
                float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0
            ]
        except ValueError:
            center_of_mass = None  # If None,trimeshauto-compute

        try:
            # Compute TODO
            result = calculate_inertia_with_trimesh(
                mesh_file_path=stl_path,
                mass=mass,
                center_of_mass=center_of_mass,
                auto_repair=True
            )

            # Todo
            if not result['success']:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Calculation Failed",
                    f"Failed to calculate inertia:\n\n{result['error_message']}"
                )
                return

            # Todo
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

            # Todo
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

            # Set UI
            self.ixx_input.setText(format_float_no_exp(inertia_tensor[0, 0]))
            self.ixy_input.setText(format_float_no_exp(inertia_tensor[0, 1]))
            self.ixz_input.setText(format_float_no_exp(inertia_tensor[0, 2]))
            self.iyy_input.setText(format_float_no_exp(inertia_tensor[1, 1]))
            self.iyz_input.setText(format_float_no_exp(inertia_tensor[1, 2]))
            self.izz_input.setText(format_float_no_exp(inertia_tensor[2, 2]))

            # Inertial origin existing value inertial origin

            # Todo
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

            # Apply TODO
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
    """settext"""
    def __init__(self, graph, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.graph = graph
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        """UItextinitialize"""
        import math
        layout = QtWidgets.QVBoxLayout(self)

        # Unified button style (global constants)
        self.button_style = UNIFIED_BUTTON_STYLE

        # Default joint settings default joint settings
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

        # Default velocity max default velocity max
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
        
        # Max velocity max velocity
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

        # Apply to all nodes button apply all nodes
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

        # Mjcf settings mjcf settings
        mjcf_group = QtWidgets.QGroupBox("MJCF Export Settings")
        mjcf_layout = QtWidgets.QHBoxLayout()  # GridLayoutHBoxLayoutswitch

        # Base link base link height
        # Default base_link height m 15px default
        mjcf_layout.addWidget(QtWidgets.QLabel("Default base_link height (m):"))
        mjcf_layout.addSpacing(15)  # 15pxspacing
        self.base_link_height_input = QtWidgets.QLineEdit()
        self.base_link_height_input.setValidator(QDoubleValidator(0.0, 10.0, 3))
        self.base_link_height_input.setText(f"{self.graph.default_base_link_height:.4f}")
        self.base_link_height_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)  # left-alignedfixed
        mjcf_layout.addWidget(self.base_link_height_input)
        mjcf_layout.addStretch()  # right margin

        mjcf_group.setLayout(mjcf_layout)
        layout.addWidget(mjcf_group)

        # Node grid node grid
        grid_group = QtWidgets.QGroupBox("Node Grid")
        grid_layout = QtWidgets.QHBoxLayout()

        # Checkbox
        self.grid_enabled_checkbox = QtWidgets.QCheckBox()
        self.grid_enabled_checkbox.setChecked(self.graph.node_grid_enabled)
        grid_layout.addWidget(self.grid_enabled_checkbox)

        grid_layout.addWidget(QtWidgets.QLabel("Grid Size:"))

        # Todo
        self.grid_size_input = QtWidgets.QLineEdit()
        self.grid_size_input.setValidator(QtGui.QIntValidator(1, 1000))
        self.grid_size_input.setText(str(self.graph.node_grid_size))
        self.grid_size_input.setFixedWidth(80)
        grid_layout.addWidget(self.grid_size_input)

        grid_layout.addWidget(QtWidgets.QLabel("pixels"))

        # Snap all to snap all grid
        snap_all_button = QtWidgets.QPushButton("Snap All to Grid")
        snap_all_button.setStyleSheet(self.button_style)
        snap_all_button.setAutoDefault(False)
        snap_all_button.clicked.connect(self.snap_all_nodes_to_grid)
        grid_layout.addWidget(snap_all_button)

        grid_layout.addStretch()

        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # Mesh highlight mesh highlight
        highlight_group = QtWidgets.QGroupBox("Mesh Highlight")
        highlight_layout = QtWidgets.QHBoxLayout()

        highlight_layout.addWidget(QtWidgets.QLabel("Highlight Color:"))

        # Show color
        self.highlight_color_box = QtWidgets.QLabel()
        self.highlight_color_box.setFixedSize(60, 30)
        self.highlight_color_box.setStyleSheet(
            f"background-color: {self.graph.highlight_color}; border: 1px solid black;"
        )
        highlight_layout.addWidget(self.highlight_color_box)

        # Pick
        pick_button = QtWidgets.QPushButton("Pick")
        pick_button.setStyleSheet(self.button_style)
        pick_button.setAutoDefault(False)  # Prevent accidental activation with the Return key
        pick_button.clicked.connect(self.pick_highlight_color)
        highlight_layout.addWidget(pick_button)

        highlight_layout.addStretch()
        highlight_group.setLayout(highlight_layout)
        layout.addWidget(highlight_group)

        # Collision color collision color
        collision_group = QtWidgets.QGroupBox("Collision Color")
        collision_layout = QtWidgets.QHBoxLayout()

        # KitchenColorPicker
        self.collision_color_picker = KitchenColorPicker(
            parent_widget=self,
            initial_color=self.graph.collision_color,
            enable_alpha=True,
            on_color_changed=self._on_collision_color_changed
        )

        # Collision color alpha alpha pick button : collision color: alpha: pick
        collision_layout.addWidget(QtWidgets.QLabel("Collision Color:"))
        collision_layout.addWidget(self.collision_color_picker.color_sample)
        collision_layout.addWidget(QtWidgets.QLabel("alpha:"))
        # Hide RGB alpha RGB
        if len(self.collision_color_picker.color_inputs) >= 4:
            # Hide RGB RGB
            for i in range(3):
                self.collision_color_picker.color_inputs[i].setVisible(False)
            collision_layout.addWidget(self.collision_color_picker.color_inputs[3])  # alphaonly
        self.collision_color_picker.pick_button.setAutoDefault(False)  # Prevent accidental activation with the Return key
        collision_layout.addWidget(self.collision_color_picker.pick_button)
        collision_layout.addStretch()

        collision_group.setLayout(collision_layout)
        layout.addWidget(collision_group)

        # Button
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        ok_button = QtWidgets.QPushButton("OK")
        ok_button.setAutoDefault(False)  # Prevent the window from closing via the Return key
        ok_button.setStyleSheet(self.button_style)  # Apply consistent style
        ok_button.clicked.connect(self.accept_settings)
        button_layout.addWidget(ok_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.setAutoDefault(False)
        cancel_button.setStyleSheet(self.button_style)  # Apply consistent style
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        # Set TODO
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
        
        # Change current size 6
        # List
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
        
        # Set current size 6
        for field in input_fields:
            # Get current 100
            current_width = field.sizeHint().width()
            if current_width <= 0:
                current_width = 100  # Default value
            # Set size 6
            new_width = int(current_width * 0.6)
            field.setMaximumWidth(new_width)
            field.setMinimumWidth(new_width)

    def pick_highlight_color(self):
        """textーtextーtextーtext"""
        # Get current color
        current_color = QtGui.QColor(self.graph.highlight_color)

        # Todo
        dialog = CustomColorDialog(current_color, self)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            color = dialog.currentColor()
            if color.isValid():
                # Save color RRGGBB
                hex_color = color.name()
                self.graph.highlight_color = hex_color
                # Update TODO
                self.highlight_color_box.setStyleSheet(
                    f"background-color: {hex_color}; border: 1px solid black;"
                )

    def _convert_rad_to_deg(self, rad_input, deg_input):
        """radtextdegtext"""
        import math
        try:
            rad_value = float(rad_input.text())
            # 4
            rad_value = round(rad_value, 4)
            rad_input.setText(f"{rad_value:.4f}")
            deg_value = math.degrees(rad_value)
            deg_input.setText(f"{deg_value:.3f}")
        except ValueError:
            pass

    def _convert_deg_to_rad(self, deg_input, rad_input):
        """degtextradtext"""
        import math
        try:
            deg_value = float(deg_input.text())
            rad_value = math.radians(deg_value)
            # 4
            rad_value = round(rad_value, 4)
            rad_input.setText(f"{rad_value:.4f}")
        except ValueError:
            pass

    def _format_number_input(self, line_edit, decimal_places):
        """textーtextーtext"""
        try:
            value = float(line_edit.text())
            # Todo
            line_edit.setText(f"{value:.{decimal_places}f}")
        except (ValueError, AttributeError):
            pass

    def _on_collision_color_changed(self, rgba_color):
        """Collision Colortextーtext"""
        # Update graph collision_color
        self.graph.collision_color = rgba_color.copy()
        print(f"Collision color updated: RGBA={rgba_color}")

    def accept_settings(self):
        """settext"""
        try:
            # Get TODO
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

            # Set apply
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

            # Mjcf
            base_link_height = float(self.base_link_height_input.text())
            self.graph.default_base_link_height = base_link_height

            # Node node grid
            grid_enabled = self.grid_enabled_checkbox.isChecked()
            grid_size = int(self.grid_size_input.text())
            self.graph.node_grid_enabled = grid_enabled
            self.graph.node_grid_size = grid_size

            # Update TODO
            self.graph.update_grid_display()

            # Highlight_color pick_highlight_color

            self.accept()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values."
            )

    def snap_all_nodes_to_grid(self):
        """textnodetext"""
        try:
            # Get Grid current value Grid Size
            grid_size = int(self.grid_size_input.text())

            # Run all node
            snapped_count = 0
            for node in self.graph.all_nodes():
                node_pos = node.pos()
                if isinstance(node_pos, (list, tuple)):
                    current_x, current_y = node_pos[0], node_pos[1]
                else:
                    current_x, current_y = node_pos.x(), node_pos.y()

                # Todo
                snapped_x = round(current_x / grid_size) * grid_size
                snapped_y = round(current_y / grid_size) * grid_size

                # Position
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
        """textnodetextparameterstextsettext"""
        try:
            # Get value
            effort = float(self.effort_input.text())
            velocity_rad = float(self.velocity_rad_input.text())
            damping_kv = float(self.damping_kv_input.text())
            stiffness_kp = float(self.stiffness_kp_input.text())
            margin_rad = float(self.margin_rad_input.text())
            armature = float(self.armature_input.text())
            frictionloss = float(self.frictionloss_input.text())

            # Apply value
            updated_count = 0
            for node in self.graph.all_nodes():
                # Todo
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
    """textー (100text0text)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 100  # initial value100switch
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

        # Show Progress arc light blue - Progress
        gradient = QConicalGradient(50, 50, 90)
        gradient.setColorAt(0, QColor(100, 180, 255, 200))
        gradient.setColorAt(1, QColor(150, 220, 255, 200))

        pen = QPen(gradient, 8)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        span_angle = int(self.value * 360 / 100 * 16)
        painter.drawArc(10, 10, 80, 80, 90 * 16, -span_angle)

        # Show TODO
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
        self.inertial_origin_actors = {}  # Inertial Originactor for display
        self.collider_actors = {}  # Collideractor for display
        self.collider_display_enabled = False  # Colliderdisplay ON/OFF state

        layout = QtWidgets.QVBoxLayout(self)

        # Progress bar (initially hidden)
        self.progress_bar = CircularProgressBar(self)
        self.progress_bar.hide()

        # Use QLabel instead of QVTKRenderWindowInteractor for M4 Mac compatibility
        self.vtk_display = QLabel(self)
        self.vtk_display.setMinimumSize(100, 1)  # min width100px、height1pxup tocan shrink
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
        # Set 0 0 1
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

        # Unified button style (global constants)
        button_style = UNIFIED_BUTTON_STYLE

        # Button
        button_layout = QtWidgets.QVBoxLayout()  # vertical layoutswitch
        button_layout.setSpacing(2)  # between buttonsspacingset narrower

        # Front side 1 : front side top
        first_row_layout = QtWidgets.QHBoxLayout()

        # Front
        self.front_button = QtWidgets.QPushButton("Front")
        self.front_button.setStyleSheet(button_style)
        self.front_button.setFixedWidth(50)
        self.front_button.clicked.connect(self.reset_camera_front)
        first_row_layout.addWidget(self.front_button)

        # Side
        self.side_button = QtWidgets.QPushButton("Side")
        self.side_button.setStyleSheet(button_style)
        self.side_button.setFixedWidth(50)
        self.side_button.clicked.connect(self.reset_camera_side)
        first_row_layout.addWidget(self.side_button)

        # Top
        self.top_button = QtWidgets.QPushButton("Top")
        self.top_button.setStyleSheet(button_style)
        self.top_button.setFixedWidth(50)
        self.top_button.clicked.connect(self.reset_camera_top)
        first_row_layout.addWidget(self.top_button)

        first_row_layout.addStretch()

        button_layout.addLayout(first_row_layout)
        button_layout.addSpacing(10)  # Front Side Toprow andMesh Wireframe Colliderbetween10pxspacing

        # Mesh wireframe 2 : mesh wireframe collider
        second_row_layout = QtWidgets.QHBoxLayout()

        # Todo
        # #0055ff #1a3a5a on : 0055ff 1a3a5a
        on_bg_color = "#0055ff"  # ONstatecolor
        on_border_color = "#0055ff"  # ONstatebordercolor
        pressed_bg_color = "#1a3a5a"  # currentONcolorused when pressed
        pressed_border_color = "#2a5a8a"  # currentONbordercolorused when pressed
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

        # Mesh on
        self.mesh_toggle = QtWidgets.QPushButton("Mesh")
        self.mesh_toggle.setCheckable(True)
        self.mesh_toggle.setChecked(True)  # ON
        self.mesh_toggle.setFixedWidth(80)
        self.mesh_toggle.setStyleSheet(toggle_button_style)
        self.mesh_toggle.toggled.connect(self.toggle_mesh)
        second_row_layout.addWidget(self.mesh_toggle)

        # Wireframe off
        self.wireframe_toggle = QtWidgets.QPushButton("Wireframe")
        self.wireframe_toggle.setCheckable(True)
        self.wireframe_toggle.setChecked(False)  # Off by default
        self.wireframe_toggle.setFixedWidth(80)
        self.wireframe_toggle.setStyleSheet(toggle_button_style)
        self.wireframe_toggle.toggled.connect(self.toggle_wireframe)
        second_row_layout.addWidget(self.wireframe_toggle)

        # - collider off old
        self.collider_toggle = QtWidgets.QPushButton("Collider")
        self.collider_toggle.setCheckable(True)
        self.collider_toggle.setChecked(False)  # Off by default
        self.collider_toggle.setFixedWidth(80)
        self.collider_toggle.setStyleSheet(toggle_button_style)
        self.collider_toggle.toggled.connect(self.toggle_collider_display)
        second_row_layout.addWidget(self.collider_toggle)

        second_row_layout.addStretch()

        button_layout.addLayout(second_row_layout)
        button_layout.addSpacing(5)  # Back-ground-color5pxspacing

        # Background background
        bg_layout = QtWidgets.QHBoxLayout()
        bg_label = QtWidgets.QLabel("background-color:")
        bg_layout.addWidget(bg_label)

        self.bg_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_slider.setMinimum(-100)  # NOTE
        self.bg_slider.setMaximum(100)   # NOTE
        self.bg_slider.setValue(-80)      # NOTE
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
        self.rotation_direction = 1  # 1: , -1:
        self.rotation_paused = False  # NOTE
        self.pause_counter = 0  # NOTE
        self.follow_children = True  # NOTE

        # Set TODO
        # 45
        light1 = vtk.vtkLight()
        light1.SetPosition(0.5, 0.5, 1.0)
        light1.SetIntensity(0.7)
        light1.SetLightTypeToSceneLight()
        
        # Todo
        light2 = vtk.vtkLight()
        light2.SetPosition(-1.0, -0.5, 0.2)
        light2.SetIntensity(0.7)
        light2.SetLightTypeToSceneLight()
        
        # Todo
        light3 = vtk.vtkLight()
        light3.SetPosition(0.3, -1.0, 0.2)
        light3.SetIntensity(0.7)
        light3.SetLightTypeToSceneLight()

        # Todo
        light4 = vtk.vtkLight()
        light4.SetPosition(1.0, 0.0, 0.3)
        light4.SetIntensity(0.3)
        light4.SetLightTypeToSceneLight()

        # Todo
        self.renderer.SetAmbient(0.7, 0.7, 0.7)
        self.renderer.LightFollowCameraOff()
        self.renderer.AddLight(light1)
        self.renderer.AddLight(light2)
        self.renderer.AddLight(light3)
        self.renderer.AddLight(light4)


        # Set TODO
        initial_bg = (-80 + 100) / 200.0  # -800-1
        self.renderer.SetBackground(initial_bg, initial_bg, initial_bg)

        # Var
        self.highlighted_node = None
        self.original_color = None
        self.highlight_timer = QTimer(self)
        self.highlight_timer.timeout.connect(self._toggle_highlight)
        self.highlight_state = False  # state

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
            # Qlabel size : vtk qlabel
            widget_size = self.vtk_display.size()
            widget_width = max(widget_size.width(), 10)  # 10
            widget_height = max(widget_size.height(), 10)

            # Update VTK
            current_size = self.render_window.GetSize()
            if current_size[0] != widget_width or current_size[1] != widget_height:
                self.render_window.SetSize(widget_width, widget_height)
                # Update TODO
                self.renderer.ResetCameraClippingRange()

            # Show camera window :
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
                    # Shift+
                    if (event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier) or \
                       event.button() == Qt.MiddleButton:
                        # Todo
                        self.mouse_drag.end_left_drag()
                        self.mouse_drag.start_middle_drag(event.pos())
                        return True
                    elif event.button() == Qt.LeftButton:
                        # Todo
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
        """nodetextdisplaytext、text"""
        # Existing
        self.clear_highlight()

        if node not in self.stl_actors:
            return

        actor = self.stl_actors[node]

        # Save color
        self.original_color = actor.GetProperty().GetColor()
        self.highlighted_node = node

        # Get hex RGB - RGB
        if hasattr(self, 'graph') and hasattr(self.graph, 'highlight_color'):
            color = QtGui.QColor(self.graph.highlight_color)
            highlight_rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
        else:
            # Todo
            highlight_rgb = (0.5, 0.8, 1.0)

        # Set TODO
        actor.GetProperty().SetColor(*highlight_rgb)
        self.render_to_image()

        # Start 500ms
        self.highlight_state = True
        self.highlight_timer.start(500)

    def _toggle_highlight(self):
        """text"""
        if not self.highlighted_node or self.highlighted_node not in self.stl_actors:
            self.highlight_timer.stop()
            return

        actor = self.stl_actors[self.highlighted_node]

        if self.highlight_state:
            # Color
            actor.GetProperty().SetColor(*self.original_color)
        else:
            # Get hex RGB - RGB
            if hasattr(self, 'graph') and hasattr(self.graph, 'highlight_color'):
                color = QtGui.QColor(self.graph.highlight_color)
                highlight_rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            else:
                # Todo
                highlight_rgb = (0.5, 0.8, 1.0)
            actor.GetProperty().SetColor(*highlight_rgb)

        self.highlight_state = not self.highlight_state
        self.render_to_image()

    def clear_highlight(self):
        """text"""
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
        """textsave"""
        if node in self.transforms:
            current_transform = vtk.vtkTransform()
            current_transform.DeepCopy(self.transforms[node])
            self.original_transforms[node] = current_transform

            # Save transform Inherit to Subnodes Inherit Subnodes
            # Show_angle angle
            # Disable Inherit to Subnodes if transform Inherit Subnodes
            self._store_children_transforms(node)

    def _store_children_transforms(self, parent_node):
        """textnodetextsave"""
        for output_port in parent_node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                if child_node in self.transforms and child_node not in self.original_transforms:
                    current_transform = vtk.vtkTransform()
                    current_transform.DeepCopy(self.transforms[child_node])
                    self.original_transforms[child_node] = current_transform

                    # Save TODO
                    self._store_children_transforms(child_node)

    def start_rotation_test(self, node):
        """textstart(PartsEditortextーtext)"""
        if node in self.stl_actors:
            # Save current transform PartsEditor PartsEditor
            self.store_current_transform(node)
            
            # Todo
            self.rotation_test_active = True
            self.rotating_node = node
            # Angle offset 0 angle zero
            self.current_angle = 0.0
            self.rotation_direction = 1  # NOTE
            self.rotation_paused = False  # state
            self.pause_counter = 0
            self.rotation_timer.start(16)  # 60FPS

    def stop_rotation_test(self, node):
        """textend - textangletext0text(PartsEditortextーtext)"""
        # Stop PartsEditor timer PartsEditor
        self.rotation_timer.stop()
        
        # Stop timeout
        self.rotation_test_active = False
        
        # Get node
        target_node = node if node else self.rotating_node
        
        # Color
        if target_node and target_node in self.stl_actors:
            if hasattr(target_node, 'node_color') and target_node.node_color:
                # Setcolor rgb setcolor rgb 3
                self.stl_actors[target_node].GetProperty().SetColor(*target_node.node_color[:3])
                # Set SetOpacity SetOpacity
                if len(target_node.node_color) >= 4:
                    self.stl_actors[target_node].GetProperty().SetOpacity(target_node.node_color[3])
                else:
                    self.stl_actors[target_node].GetProperty().SetOpacity(1.0)
        
        # Partseditor transform position partseditor
        if self.original_transforms:
            nodes_to_restore = list(self.original_transforms.keys())
            for restore_node in nodes_to_restore:
                if restore_node in self.transforms and restore_node in self.original_transforms:
                    # Original_transforms transform
                    self.transforms[restore_node].DeepCopy(self.original_transforms[restore_node])
                    if restore_node in self.stl_actors:
                        self.stl_actors[restore_node].SetUserTransform(self.transforms[restore_node])
                del self.original_transforms[restore_node]
        
        # Update PartsEditor PartsEditor 3D
        self.render_to_image()
        
        # Todo
        self.rotating_node = None
        self.rotation_paused = False
        self.pause_counter = 0
        self.current_angle = 0

    def show_angle(self, node, angle_rad):
        """textangletextSTLmodeltextdisplay(text)"""
        import math

        if node not in self.stl_actors:
            return

        # Stop TODO
        self.rotation_timer.stop()

        # Save current
        node.current_joint_angle = angle_rad

        # Angle transform
        angle_deg = math.degrees(angle_rad)

        # Get transform
        transform = self.transforms[node]

        # Get parent transform joint origin XYZ/RPY parent point_angle XYZ/RPY
        parent_transform = None
        joint_origin_xyz = None
        joint_origin_rpy = None
        parent_point_angle = None

        if hasattr(node, 'graph'):
            graph = node.graph
            # Find parent from node input port
            for input_port in node.input_ports():
                connected_ports = input_port.connected_ports()
                if connected_ports:
                    parent_node = connected_ports[0].node()
                    parent_port_name = connected_ports[0].name()

                    # Get parent compute
                    parent_output_ports = list(parent_node.output_ports())
                    for port_idx, port in enumerate(parent_output_ports):
                        if port.name() == parent_port_name:
                            # Compute point index from port name (out_1->0, out_2->1, etc.)
                            point_index = port_idx  # Default
                            if parent_port_name.startswith('out_'):
                                try:
                                    port_num = int(parent_port_name.split('_')[1])
                                    point_index = port_num - 1
                                except (ValueError, IndexError):
                                    pass
                            elif parent_port_name == 'out':
                                point_index = 0
                            # Get points XYZ RPY point_angle XYZ RPY
                            if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                                point_data = parent_node.points[point_index]
                                joint_origin_xyz = point_data.get('xyz', [0, 0, 0])
                                joint_origin_rpy = point_data.get('rpy', [0, 0, 0])
                                parent_point_angle = point_data.get('angle', [0.0, 0.0, 0.0])

                            # Get parent transform
                            if parent_node in self.transforms:
                                parent_transform = self.transforms[parent_node]
                            break
                    break

        # Transform
        transform.Identity()

        # Apply parent transform
        if parent_transform is not None:
            transform.Concatenate(parent_transform)

        # Apply joint position
        if joint_origin_xyz:
            transform.Translate(joint_origin_xyz[0], joint_origin_xyz[1], joint_origin_xyz[2])

        # Apply joint origin RPY RPY URDF : Z-Y-X
        if joint_origin_rpy and len(joint_origin_rpy) == 3:
            roll_deg = math.degrees(joint_origin_rpy[0])
            pitch_deg = math.degrees(joint_origin_rpy[1])
            yaw_deg = math.degrees(joint_origin_rpy[2])
            transform.RotateZ(yaw_deg)    # Yaw
            transform.RotateY(pitch_deg)  # Pitch
            transform.RotateX(roll_deg)   # Roll

        # Apply parent point_angle radian degree - Rotation Axis VTK Z-Y-X Rotation Axis
        if parent_point_angle and any(a != 0.0 for a in parent_point_angle):
            parent_point_angle_deg = [math.degrees(a) for a in parent_point_angle]
            transform.RotateZ(parent_point_angle_deg[2])  # Z-axis rotation
            transform.RotateY(parent_point_angle_deg[1])  # Y-axis rotation
            transform.RotateX(parent_point_angle_deg[0])  # X-axis rotation
            print(f"Applied parent point_angle: X={parent_point_angle_deg[0]}, Y={parent_point_angle_deg[1]}, Z={parent_point_angle_deg[2]} degrees")

        # Get Angle offset body_angle Angle
        # Body_angle radian degree transform
        angle_offset_deg = 0.0
        if hasattr(node, 'body_angle') and hasattr(node, 'rotation_axis'):
            body_angle = node.body_angle
            rotation_axis = node.rotation_axis
            if rotation_axis == 0:  # X-axis
                angle_offset_deg = math.degrees(body_angle[0])
            elif rotation_axis == 1:  # Y-axis
                angle_offset_deg = math.degrees(body_angle[1])
            elif rotation_axis == 2:  # Z-axis
                angle_offset_deg = math.degrees(body_angle[2])

        # Parent point_angle rotate
        # Angle_rad angle angle offset zero
        # Angle_rad + angle_offset_rad joint
        actual_angle_deg = angle_deg + angle_offset_deg
        if hasattr(node, 'rotation_axis'):
            if node.rotation_axis == 0:    # X-axis
                transform.RotateX(actual_angle_deg)
            elif node.rotation_axis == 1:  # Y-axis
                transform.RotateY(actual_angle_deg)
            elif node.rotation_axis == 2:  # Z-axis
                transform.RotateZ(actual_angle_deg)

        self.stl_actors[node].SetUserTransform(transform)

        # Enable Inherit to Subnodes if rotate Inherit Subnodes
        if self.follow_children and hasattr(node, 'graph'):
            self._rotate_children(node, transform)
        else:
            # Disable Inherit to Subnodes if transform Inherit Subnodes
            if hasattr(node, 'graph'):
                self._restore_children_transforms(node)

        self.render_to_image()

        print(f"Showing angle: {angle_rad} rad ({angle_deg} deg)")

    def show_inertial_origin(self, node, xyz):
        """Inertial OrigintextXYZtransformstextdisplay(textーtexttransformstext)"""
        # Show existing
        self.hide_inertial_origin(node)

        # Get node transform
        if node not in self.transforms:
            print(f"Node {node.name()} has no transform")
            return

        node_transform = self.transforms[node]

        # Create list
        actors = []

        # Create 1 1.
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(xyz[0], xyz[1], xyz[2])
        sphere.SetRadius(0.005)  # NOTE
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # color
        sphere_actor.SetUserTransform(node_transform)  # Apply node transform

        self.renderer.AddActor(sphere_actor)
        actors.append(sphere_actor)

        # Create 2 2. XYZ
        axis_length = 0.03  # NOTE

        # X
        x_line = vtk.vtkLineSource()
        x_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        x_line.SetPoint2(xyz[0] + axis_length, xyz[1], xyz[2])

        x_mapper = vtk.vtkPolyDataMapper()
        x_mapper.SetInputConnection(x_line.GetOutputPort())

        x_actor = vtk.vtkActor()
        x_actor.SetMapper(x_mapper)
        x_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # NOTE
        x_actor.GetProperty().SetLineWidth(3)
        x_actor.SetUserTransform(node_transform)  # Apply node transform

        self.renderer.AddActor(x_actor)
        actors.append(x_actor)

        # Y
        y_line = vtk.vtkLineSource()
        y_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        y_line.SetPoint2(xyz[0], xyz[1] + axis_length, xyz[2])

        y_mapper = vtk.vtkPolyDataMapper()
        y_mapper.SetInputConnection(y_line.GetOutputPort())

        y_actor = vtk.vtkActor()
        y_actor.SetMapper(y_mapper)
        y_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # NOTE
        y_actor.GetProperty().SetLineWidth(3)
        y_actor.SetUserTransform(node_transform)  # Apply node transform

        self.renderer.AddActor(y_actor)
        actors.append(y_actor)

        # Z
        z_line = vtk.vtkLineSource()
        z_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        z_line.SetPoint2(xyz[0], xyz[1], xyz[2] + axis_length)

        z_mapper = vtk.vtkPolyDataMapper()
        z_mapper.SetInputConnection(z_line.GetOutputPort())

        z_actor = vtk.vtkActor()
        z_actor.SetMapper(z_mapper)
        z_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # NOTE
        z_actor.GetProperty().SetLineWidth(3)
        z_actor.SetUserTransform(node_transform)  # Apply node transform

        self.renderer.AddActor(z_actor)
        actors.append(z_actor)

        # Save TODO
        self.inertial_origin_actors[node] = actors

        # Redraw
        self.render_to_image()

    def hide_inertial_origin(self, node):
        """Inertial Origintextdisplaytextremove"""
        if node in self.inertial_origin_actors:
            for actor in self.inertial_origin_actors[node]:
                self.renderer.RemoveActor(actor)
            del self.inertial_origin_actors[node]
            self.render_to_image()

    def update_rotation(self):
        """textupdate"""
        if self.rotating_node and self.rotating_node in self.stl_actors:
            node = self.rotating_node
            transform = self.transforms[node]

            # Current position
            position = transform.GetPosition()

            # Check fixed
            is_fixed = hasattr(node, 'rotation_axis') and node.rotation_axis == 3

            # If fixed
            if is_fixed:
                # 400ms
                # 60fps 24 400ms
                is_red = (self.current_angle // 24) % 2 == 0
                if is_red:
                    self.stl_actors[node].GetProperty().SetColor(1.0, 0.0, 0.0)  # NOTE
                else:
                    self.stl_actors[node].GetProperty().SetColor(1.0, 1.0, 1.0)  # NOTE
            else:
                # Get Joint limit transform Joint
                import math
                lower_deg = math.degrees(getattr(node, 'joint_lower', -3.14159))
                upper_deg = math.degrees(getattr(node, 'joint_upper', 3.14159))

                # Get Angle offset Angle 0
                # Body_angle radian degree transform
                angle_offset_deg = 0.0
                if hasattr(node, 'body_angle') and hasattr(node, 'rotation_axis'):
                    body_angle = node.body_angle
                    rotation_axis = node.rotation_axis
                    if rotation_axis == 0:  # X-axis
                        angle_offset_deg = math.degrees(body_angle[0])
                    elif rotation_axis == 1:  # Y-axis
                        angle_offset_deg = math.degrees(body_angle[1])
                    elif rotation_axis == 2:  # Z-axis
                        angle_offset_deg = math.degrees(body_angle[2])
                
                # Compute angle_offset joint
                # 0 joint angle_offset
                display_lower_deg = lower_deg - angle_offset_deg
                display_upper_deg = upper_deg - angle_offset_deg

                # Process
                if self.rotation_paused:
                    self.pause_counter += 1
                    # 0 60fps × 0 5 0.5 60fps 30
                    if self.pause_counter >= 30:
                        self.rotation_paused = False
                        self.pause_counter = 0
                    # Angle current angle
                else:
                    # - 0
                    angle_step = 2.0  # 1（）
                    self.current_angle += angle_step * self.rotation_direction

                    # Check
                    if self.current_angle >= display_upper_deg:
                        self.current_angle = display_upper_deg
                        self.rotation_direction = -1  # NOTE
                        self.rotation_paused = True  # NOTE
                        self.pause_counter = 0
                    elif self.current_angle <= display_lower_deg:
                        self.current_angle = display_lower_deg
                        self.rotation_direction = 1   # NOTE
                        self.rotation_paused = True  # NOTE
                        self.pause_counter = 0

                # Todo
                transform.Identity()  # NOTE

                # Parent transform joint origin xyz/rpy parent point_angle xyz/rpy
                # Node
                parent_transform = None
                joint_origin_xyz = None
                joint_origin_rpy = None
                parent_point_angle = None

                if hasattr(node, 'graph'):
                    graph = node.graph
                    # Find parent from node input port
                    for input_port in node.input_ports():
                        connected_ports = input_port.connected_ports()
                        if connected_ports:
                            parent_node = connected_ports[0].node()
                            parent_port_name = connected_ports[0].name()

                            # Get parent compute
                            # Compute point index from port name (out_1->0, out_2->1, etc.)
                            point_index = 0  # Default
                            if parent_port_name.startswith('out_'):
                                try:
                                    port_num = int(parent_port_name.split('_')[1])
                                    point_index = port_num - 1
                                except (ValueError, IndexError):
                                    pass
                            elif parent_port_name == 'out':
                                point_index = 0

                            # Get points XYZ RPY point_angle XYZ RPY
                            if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                                point_data = parent_node.points[point_index]
                                joint_origin_xyz = point_data.get('xyz', [0, 0, 0])
                                joint_origin_rpy = point_data.get('rpy', [0, 0, 0])
                                parent_point_angle = point_data.get('angle', [0.0, 0.0, 0.0])

                            # Get parent transform
                            if parent_node in self.transforms:
                                parent_transform = self.transforms[parent_node]
                            break

                # Apply parent transform
                if parent_transform is not None:
                    transform.Concatenate(parent_transform)

                # Apply joint position
                if joint_origin_xyz:
                    transform.Translate(joint_origin_xyz[0], joint_origin_xyz[1], joint_origin_xyz[2])

                # Apply joint origin RPY RPY URDF : Z-Y-X
                if joint_origin_rpy and len(joint_origin_rpy) == 3:
                    roll_deg = math.degrees(joint_origin_rpy[0])
                    pitch_deg = math.degrees(joint_origin_rpy[1])
                    yaw_deg = math.degrees(joint_origin_rpy[2])
                    transform.RotateZ(yaw_deg)    # Yaw
                    transform.RotateY(pitch_deg)  # Pitch
                    transform.RotateX(roll_deg)   # Roll

                # Apply parent point_angle radian degree - Rotation Axis VTK Z-Y-X Rotation Axis
                if parent_point_angle and any(a != 0.0 for a in parent_point_angle):
                    parent_point_angle_deg = [math.degrees(a) for a in parent_point_angle]
                    transform.RotateZ(parent_point_angle_deg[2])  # Z-axis rotation
                    transform.RotateY(parent_point_angle_deg[1])  # Y-axis rotation
                    transform.RotateX(parent_point_angle_deg[0])  # X-axis rotation

                # Get Angle offset body_angle radian degree transform Angle
                angle_offset_deg = 0.0
                if hasattr(node, 'body_angle') and hasattr(node, 'rotation_axis'):
                    body_angle = node.body_angle
                    rotation_axis = node.rotation_axis
                    if rotation_axis == 0:  # X-axis
                        angle_offset_deg = math.degrees(body_angle[0])
                    elif rotation_axis == 1:  # Y-axis
                        angle_offset_deg = math.degrees(body_angle[1])
                    elif rotation_axis == 2:  # Z-axis
                        angle_offset_deg = math.degrees(body_angle[2])
                
                # Apply angle parent point_angle rotate
                # Current_angle angle angle offset zero
                # Current_angle + angle_offset joint
                actual_angle_deg = self.current_angle + angle_offset_deg
                if hasattr(node, 'rotation_axis'):
                    if node.rotation_axis == 0:    # X-axis
                        transform.RotateX(actual_angle_deg)
                    elif node.rotation_axis == 1:  # Y-axis
                        transform.RotateY(actual_angle_deg)
                    elif node.rotation_axis == 2:  # Z-axis
                        transform.RotateZ(actual_angle_deg)

                self.stl_actors[node].SetUserTransform(transform)

                # Enable Inherit to Subnodes if rotate Inherit Subnodes
                if self.follow_children and hasattr(node, 'graph'):
                    self._rotate_children(node, transform)

            self.render_to_image()

    def _rotate_children(self, parent_node, parent_transform):
        """textnodetext(text)"""
        import math
        import vtk

        # Check
        for port_idx, output_port in enumerate(parent_node.output_ports()):
            # Get TODO
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                # Stl stl
                if child_node not in self.stl_actors or child_node not in self.transforms:
                    continue

                # Compute point index from port name (out_1->0, out_2->1, etc.)
                port_name = output_port.name() if hasattr(output_port, 'name') else ''
                point_index = port_idx  # Default
                if port_name.startswith('out_'):
                    try:
                        port_num = int(port_name.split('_')[1])
                        point_index = port_num - 1
                    except (ValueError, IndexError):
                        pass
                elif port_name == 'out':
                    point_index = 0

                # Get point_angle
                child_xyz = [0, 0, 0]
                child_rpy = [0, 0, 0]
                parent_point_angle = [0.0, 0.0, 0.0]

                if hasattr(parent_node, 'points') and point_index < len(parent_node.points):
                    point_data = parent_node.points[point_index]
                    child_xyz = point_data.get('xyz', [0, 0, 0])
                    child_rpy = point_data.get('rpy', [0, 0, 0])
                    parent_point_angle = point_data.get('angle', [0.0, 0.0, 0.0])

                # Update transform parent transform
                # Todo
                child_transform = vtk.vtkTransform()
                child_transform.Identity()
                
                # Apply parent transform rotate
                # Deepcopy parent transform deepcopy
                child_transform.DeepCopy(parent_transform)

                # Apply joint position
                child_transform.Translate(child_xyz[0], child_xyz[1], child_xyz[2])

                # Apply joint origin RPY RPY
                if len(child_rpy) == 3:
                    roll_deg = math.degrees(child_rpy[0])
                    pitch_deg = math.degrees(child_rpy[1])
                    yaw_deg = math.degrees(child_rpy[2])
                    child_transform.RotateZ(yaw_deg)
                    child_transform.RotateY(pitch_deg)
                    child_transform.RotateX(roll_deg)

                # Apply parent point_angle radian degree VTK Z-Y-X
                if any(a != 0.0 for a in parent_point_angle):
                    parent_point_angle_deg = [math.degrees(a) for a in parent_point_angle]
                    child_transform.RotateZ(parent_point_angle_deg[2])
                    child_transform.RotateY(parent_point_angle_deg[1])
                    child_transform.RotateX(parent_point_angle_deg[0])

                # Apply body_angle radian degree VTK
                # Note body_angle mjcf ref angle if note: mjcf
                # Rotation_axis rotate
                child_body_angle = getattr(child_node, 'body_angle', [0.0, 0.0, 0.0])
                if any(a != 0.0 for a in child_body_angle):
                    child_body_angle_deg = [math.degrees(a) for a in child_body_angle]
                    # Rotation_axis rotate
                    if hasattr(child_node, 'rotation_axis'):
                        if child_node.rotation_axis == 0 and child_body_angle_deg[0] != 0.0:  # X-axis
                            child_transform.RotateX(child_body_angle_deg[0])
                        elif child_node.rotation_axis == 1 and child_body_angle_deg[1] != 0.0:  # Y-axis
                            child_transform.RotateY(child_body_angle_deg[1])
                        elif child_node.rotation_axis == 2 and child_body_angle_deg[2] != 0.0:  # Z-axis
                            child_transform.RotateZ(child_body_angle_deg[2])
                    else:
                        # Apply rotation_axis Z-Y-X
                        child_transform.RotateZ(child_body_angle_deg[2])
                        child_transform.RotateY(child_body_angle_deg[1])
                        child_transform.RotateX(child_body_angle_deg[0])

                # Apply current
                child_joint_angle = getattr(child_node, 'current_joint_angle', 0.0)
                if child_joint_angle != 0.0:
                    child_angle_deg = math.degrees(child_joint_angle)
                    if hasattr(child_node, 'rotation_axis'):
                        if child_node.rotation_axis == 0:    # X-axis
                            child_transform.RotateX(child_angle_deg)
                        elif child_node.rotation_axis == 1:  # Y-axis
                            child_transform.RotateY(child_angle_deg)
                        elif child_node.rotation_axis == 2:  # Z-axis
                            child_transform.RotateZ(child_angle_deg)

                # Update transform apply self transforms self.transforms
                self.transforms[child_node].DeepCopy(child_transform)
                self.stl_actors[child_node].SetUserTransform(child_transform)

                # Rotate
                self._rotate_children(child_node, child_transform)

    def _restore_children_transforms(self, parent_node):
        """textnodetext(text)"""
        for output_port in parent_node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                # Stl stl
                if child_node not in self.stl_actors or child_node not in self.transforms:
                    continue

                # Todo
                if child_node in self.original_transforms:
                    self.transforms[child_node].DeepCopy(self.original_transforms[child_node])
                    self.stl_actors[child_node].SetUserTransform(self.transforms[child_node])

                # Todo
                self._restore_children_transforms(child_node)

    def _get_scene_bounds_and_center(self):
        """textーtextbounding boxtext"""
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
        """Front view(text)- textdisplay"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # Default value

        # Center on origin (0,0,0)
        distance = max(diagonal, 1.0)  # Enforce minimum distance
        parallel_scale = max(diagonal * 0.7, 0.1)  # Enforce minimum scale

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[distance, 0, 0],  # X
            view_up=[0, 0, 1],
            focal_point=[0, 0, 0],  # Look at the origin
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Front view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera_side(self):
        """Side view(text)- textdisplay"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # Default value

        # Center on origin (0,0,0)
        distance = max(diagonal, 1.0)  # Enforce minimum distance
        parallel_scale = max(diagonal * 0.7, 0.1)  # Enforce minimum scale

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[0, distance, 0],  # Y
            view_up=[0, 0, 1],
            focal_point=[0, 0, 0],  # Look at the origin
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Side view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera_top(self):
        """Top view(text)- textdisplay"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # Default value

        # Center on origin (0,0,0)
        distance = max(diagonal, 1.0)  # Enforce minimum distance
        parallel_scale = max(diagonal * 0.7, 0.1)  # Enforce minimum scale

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[0, 0, distance],  # Z
            view_up=[0, 1, 0],  # Top viewY
            focal_point=[0, 0, 0],  # Look at the origin
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Top view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera(self):
        """textーtextset(Front viewtext)"""
        self.reset_camera_front()

    def reset_view_to_fit(self):
        """alltextSTLmodeltextーtextsettext"""
        self.reset_camera()
        self.render_to_image()

    def toggle_mesh(self, checked):
        """meshdisplaytext"""
        self.update_display_mode()

    def toggle_wireframe(self, checked):
        """Wireframedisplaytextーtext"""
        self.update_display_mode()

    def update_display_mode(self):
        """meshtextーtextーtextdisplaytextーtextupdate"""
        mesh_on = self.mesh_toggle.isChecked()
        wireframe_on = self.wireframe_toggle.isChecked()

        for node, actor in self.stl_actors.items():
            if mesh_on and wireframe_on:
                # +
                actor.SetVisibility(True)
                actor.GetProperty().SetRepresentationToSurface()
                actor.GetProperty().EdgeVisibilityOn()
                actor.GetProperty().SetLineWidth(1)
            elif mesh_on and not wireframe_on:
                # Todo
                actor.SetVisibility(True)
                actor.GetProperty().SetRepresentationToSurface()
                actor.GetProperty().EdgeVisibilityOff()
            elif not mesh_on and wireframe_on:
                # Todo
                actor.SetVisibility(True)
                actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetLineWidth(1)
            else:  # not mesh_on and not wireframe_on
                # Hide TODO
                actor.SetVisibility(False)

        # Output
        if mesh_on and wireframe_on:
            mode = "Surface + Edges"
        elif mesh_on and not wireframe_on:
            mode = "Surface only"
        elif not mesh_on and wireframe_on:
            mode = "Wireframe only"
        else:
            mode = "Hidden"

        print(f"Display mode updated: {mode} (Mesh={mesh_on}, Wireframe={wireframe_on})")

        # Redraw
        self.render_to_image()

    def toggle_collider_display(self, checked):
        """Colliderdisplaytext"""
        self.collider_display_enabled = checked

        if checked:
            # Show Collider Collider
            self.show_all_colliders()
            print("Collider display ON")
        else:
            # Hide Collider Collider
            self.hide_all_colliders()
            print("Collider display OFF")

        # Redraw
        self.render_to_image()

    def show_all_colliders(self):
        """textnodetextCollidertextdisplay"""
        print("=== show_all_colliders() called ===")
        # Existing collider
        self.hide_all_colliders()

        # Get TODO
        if hasattr(self, 'graph') and self.graph:
            nodes = self.graph.all_nodes()
            print(f"Total nodes in graph: {len(nodes)}")
            for node in nodes:
                self.create_collider_actor_for_node(node)
            
            # Update transform
            print("Updating collider transforms...")
            for node in nodes:
                if node in self.collider_actors and node in self.transforms:
                    self.update_collider_transform(node)
        
        print("=== show_all_colliders() finished ===")

    def hide_all_colliders(self):
        """textCollidertextーtextremove"""
        for node, actors in list(self.collider_actors.items()):
            if isinstance(actors, list):
                for actor in actors:
                    self.renderer.RemoveActor(actor)
            else:
                self.renderer.RemoveActor(actors)
        self.collider_actors.clear()

    def create_collider_actor_for_node(self, node):
        """nodetextCollidertextーtext"""
        node_name = getattr(node, 'name', 'Unknown')
        
        # Node colliders list check : node.colliders
        colliders = getattr(node, 'colliders', [])
        if colliders:
            # Colliders list :
            actors = []
            for i, collider in enumerate(colliders):
                if not collider.get('enabled', False):
                    continue
                
                collider_type = collider.get('type')
                print(f"  Creating collider[{i}] for {node_name}: type={collider_type}")
                
                if collider_type == 'primitive':
                    collider_data = collider.get('data')
                    if collider_data:
                        # Position rotation collider
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
                        # Mesh visual mesh
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
                # Save list
                if len(actors) == 1:
                    self.collider_actors[node] = actors[0]
                else:
                    self.collider_actors[node] = actors
                print(f"  ✓ Created {len(actors)} collider actor(s) for {node_name}")
            else:
                print(f"  ✗ No enabled colliders found for {node_name}")

    def create_primitive_collider_actor(self, collider_data, node=None, position=None, rotation=None):
        """textーtextーtext
        
        Args:
            collider_data: textーtext(type, geometrytext)
            node: textnode(text)
            position: textーtext [x, y, z](text、text [0,0,0])
            rotation: textーtext [rx, ry, rz] in degrees(text、text [0,0,0])
        """
        geom_type = collider_data.get('type', 'box')
        geometry = collider_data.get('geometry', {})
        # Args position/rotation collider_data
        if position is None:
            position = collider_data.get('position', [0, 0, 0])
        if rotation is None:
            rotation = collider_data.get('rotation', [0, 0, 0])  # degrees

        # Create TODO
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

            # Todo
            append = vtk.vtkAppendPolyData()

            # - vtk y urdf/mujoco z
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(radius)
            cylinder.SetHeight(length)
            cylinder.SetResolution(30)
            cylinder.SetCapping(0)  # NOTE

            # Apply Y Z
            cyl_transform = vtk.vtkTransform()
            cyl_transform.RotateX(90)
            cyl_filter = vtk.vtkTransformPolyDataFilter()
            cyl_filter.SetInputConnection(cylinder.GetOutputPort())
            cyl_filter.SetTransform(cyl_transform)
            append.AddInputConnection(cyl_filter.GetOutputPort())

            # Z
            top_cap = vtk.vtkDiskSource()
            top_cap.SetInnerRadius(0.0)
            top_cap.SetOuterRadius(radius)
            top_cap.SetRadialResolution(1)
            top_cap.SetCircumferentialResolution(30)

            top_cap_transform = vtk.vtkTransform()
            top_cap_transform.Translate(0, 0, length / 2)  # Z
            top_cap_filter = vtk.vtkTransformPolyDataFilter()
            top_cap_filter.SetInputConnection(top_cap.GetOutputPort())
            top_cap_filter.SetTransform(top_cap_transform)
            append.AddInputConnection(top_cap_filter.GetOutputPort())

            # Z
            bottom_cap = vtk.vtkDiskSource()
            bottom_cap.SetInnerRadius(0.0)
            bottom_cap.SetOuterRadius(radius)
            bottom_cap.SetRadialResolution(1)
            bottom_cap.SetCircumferentialResolution(30)

            bottom_cap_transform = vtk.vtkTransform()
            bottom_cap_transform.RotateY(180)  # NOTE
            bottom_cap_transform.Translate(0, 0, -length / 2)  # Z
            bottom_cap_filter = vtk.vtkTransformPolyDataFilter()
            bottom_cap_filter.SetInputConnection(bottom_cap.GetOutputPort())
            bottom_cap_filter.SetTransform(bottom_cap_transform)
            append.AddInputConnection(bottom_cap_filter.GetOutputPort())

            append.Update()
            source = append

        elif geom_type == 'capsule':
            # + 2
            radius = float(geometry.get('radius', 0.5))
            # SDF import historically stored capsule length under 'height'.
            length = float(geometry.get('length', geometry.get('height', 1.0)))
            
            # DEBUG: Print capsule dimensions
            if node:
                print(f"[CAPSULE_DEBUG] Node: {node.name()}, radius={radius}, length={length}, total_length={length + 2 * radius}")

            # Todo
            append = vtk.vtkAppendPolyData()

            # - vtk y urdf/mujoco z
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(radius)
            cylinder.SetHeight(length)
            cylinder.SetResolution(30)
            cylinder.SetCapping(0)  # NOTE

            # Apply Y Z
            cyl_transform = vtk.vtkTransform()
            cyl_transform.RotateX(90)
            cyl_filter = vtk.vtkTransformPolyDataFilter()
            cyl_filter.SetInputConnection(cylinder.GetOutputPort())
            cyl_filter.SetTransform(cyl_transform)
            append.AddInputConnection(cyl_filter.GetOutputPort())

            # Z
            top_sphere = vtk.vtkSphereSource()
            top_sphere.SetRadius(radius)
            top_sphere.SetThetaResolution(30)
            top_sphere.SetPhiResolution(30)
            top_sphere.SetStartTheta(0)
            top_sphere.SetEndTheta(360)
            top_sphere.SetStartPhi(0)
            top_sphere.SetEndPhi(90)

            top_transform = vtk.vtkTransform()
            top_transform.Translate(0, 0, length / 2)  # Z
            top_filter = vtk.vtkTransformPolyDataFilter()
            top_filter.SetInputConnection(top_sphere.GetOutputPort())
            top_filter.SetTransform(top_transform)
            append.AddInputConnection(top_filter.GetOutputPort())

            # Z
            bottom_sphere = vtk.vtkSphereSource()
            bottom_sphere.SetRadius(radius)
            bottom_sphere.SetThetaResolution(30)
            bottom_sphere.SetPhiResolution(30)
            bottom_sphere.SetStartTheta(0)
            bottom_sphere.SetEndTheta(360)
            bottom_sphere.SetStartPhi(90)
            bottom_sphere.SetEndPhi(180)

            bottom_transform = vtk.vtkTransform()
            bottom_transform.Translate(0, 0, -length / 2)  # Z
            bottom_filter = vtk.vtkTransformPolyDataFilter()
            bottom_filter.SetInputConnection(bottom_sphere.GetOutputPort())
            bottom_filter.SetTransform(bottom_transform)
            append.AddInputConnection(bottom_filter.GetOutputPort())

            append.Update()
            source = append

        if not source:
            return None

        # Create TODO
        mapper = vtk.vtkPolyDataMapper()
        if hasattr(source, 'GetOutputPort'):
            mapper.SetInputConnection(source.GetOutputPort())
        else:
            mapper.SetInputData(source.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Apply Settings Color Settings Collision Color
        if hasattr(self, 'graph') and hasattr(self.graph, 'collision_color'):
            collision_color = self.graph.collision_color
            actor.GetProperty().SetColor(*collision_color[:3])  # RGB
            if len(collision_color) >= 4:
                actor.GetProperty().SetOpacity(collision_color[3])  # Alpha
            else:
                actor.GetProperty().SetOpacity(1.0)
        else:
            # Todo
            actor.GetProperty().SetColor(*DEFAULT_COLLISION_COLOR[:3])
            actor.GetProperty().SetOpacity(DEFAULT_COLLISION_COLOR[3])

        # Create TODO
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

        # Node transform
        if node and node in self.transforms:
            combined_transform = vtk.vtkTransform()
            combined_transform.PostMultiply()
            # Apply TODO
            combined_transform.Concatenate(collider_local_transform)
            # Apply next node transform
            combined_transform.Concatenate(self.transforms[node])
            actor.SetUserTransform(combined_transform)
        else:
            # Todo
            actor.SetUserTransform(collider_local_transform)

        return actor

    def create_mesh_collider_actor(self, node, collider_mesh, mesh_scale=None):
        """meshtextーtextーtext

        Args:
            node: nodetext
            collider_mesh: textーmeshtext
            mesh_scale: meshtextーtext [x, y, z] (text、text [1.0, 1.0, 1.0])
        """
        # Path
        if os.path.isabs(collider_mesh):
            # If
            collider_path = collider_mesh
            print(f"      Using absolute path: {collider_path}")
        else:
            # If
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

        # Todo
        polydata, _ = self.load_mesh_file(collider_path)
        if not polydata:
            print(f"      ✗ Failed to load mesh file")
            return None

        print(f"      ✓ Mesh loaded: {polydata.GetNumberOfPoints()} points, {polydata.GetNumberOfCells()} cells")

        # Apply collision mesh scale PolyData SDF <collision><mesh><scale> PolyData SDF
        try:
            # Mesh_scale
            scale = mesh_scale if mesh_scale is not None else [1.0, 1.0, 1.0]
            
            if isinstance(scale, (list, tuple)) and len(scale) == 3:
                # Default 1 1 1
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

        # Create TODO
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Apply Settings Color Settings Collision Color
        if hasattr(self, 'graph') and hasattr(self.graph, 'collision_color'):
            collision_color = self.graph.collision_color
            actor.GetProperty().SetColor(*collision_color[:3])  # RGB
            if len(collision_color) >= 4:
                actor.GetProperty().SetOpacity(collision_color[3])  # Alpha
            else:
                actor.GetProperty().SetOpacity(1.0)
            print(f"      ✓ Actor created with color: RGB={collision_color[:3]}, opacity: {collision_color[3] if len(collision_color) >= 4 else 1.0}")
        else:
            # Todo
            actor.GetProperty().SetColor(*DEFAULT_COLLISION_COLOR[:3])
            actor.GetProperty().SetOpacity(DEFAULT_COLLISION_COLOR[3])
            print(f"      ✓ Actor created with default collision color")

        # Apply node position rotate
        self.apply_node_transform_to_collider(node, actor)
        print(f"      ✓ Transform applied to collider actor")

        return actor

    def apply_node_transform_to_collider(self, node, actor):
        """nodetextーtextーtext(meshtextーtext)"""
        if node in self.transforms:
            # Existing transform
            node_transform = self.transforms[node]
            collider_transform = vtk.vtkTransform()
            collider_transform.DeepCopy(node_transform)
            actor.SetUserTransform(collider_transform)
    
    def update_collider_transform(self, node):
        """textーtextーtexttransformtextupdate"""
        if node not in self.collider_actors or node not in self.transforms:
            return
        
        node_transform = self.transforms[node]
        actors = self.collider_actors[node]

        # Node colliders : node.colliders
        colliders = getattr(node, 'colliders', None)
        if isinstance(colliders, list) and len(colliders) > 0:
            # Actors list
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

                # Primitive + node_transform primitive: pos/rot
                if collider_type == 'primitive':
                    # Position rotation collider collider_data
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
                    # Mesh and others node_transform + : pos/rot
                    collider_position = collider.get('position', [0, 0, 0])
                    collider_rotation = collider.get('rotation', [0, 0, 0])  # degrees
                    
                    print(f"  [COLLIDER_TRANSFORM_DEBUG] Updating mesh collider transform:")
                    print(f"    collider_position: {collider_position}")
                    print(f"    collider_rotation (deg): {collider_rotation}")
                    
                    # Create TODO
                    collider_local_transform = vtk.vtkTransform()
                    collider_local_transform.PostMultiply()
                    
                    # Apply rotate transform
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
                    
                    # Apply position
                    if collider_position != [0, 0, 0]:
                        collider_local_transform.Translate(collider_position[0], collider_position[1], collider_position[2])
                        print(f"    Applied local translation: {collider_position}")
                    
                    # Node transform
                    combined_transform = vtk.vtkTransform()
                    combined_transform.PostMultiply()
                    combined_transform.Concatenate(collider_local_transform)
                    combined_transform.Concatenate(node_transform)
                    actor.SetUserTransform(combined_transform)
                    print(f"    Combined transform applied to mesh collider actor")

    def refresh_collider_display(self):
        """Colliderdisplaytext、displaytextupdate"""
        if self.collider_display_enabled:
            self.show_all_colliders()
            self.render_to_image()

    def create_coordinate_axes(self):
        """transformstext(text)"""
        base_assembly = vtk.vtkAssembly()
        length = 0.1
        text_offset = 0.02
        
        # Change TODO
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

        # Change vtkBillboardTextActor3D vtkBillboardTextActor3D
        for i, (color, label) in enumerate([
            ((1,0,0), "X"),
            ((0,1,0), "Y"),
            ((0,0,1), "Z")
        ]):
            text_position = [0, 0, 0]
            text_position[i] = length + text_offset
            
            text_actor = vtk.vtkBillboardTextActor3D()  # vtkTextActor3Dswitch
            text_actor.SetInput(label)
            text_actor.SetPosition(*text_position)
            text_actor.GetTextProperty().SetColor(*color)
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.GetTextProperty().SetJustificationToCentered()
            text_actor.GetTextProperty().SetVerticalJustificationToCentered()
            text_actor.SetScale(0.02)  # NOTE
            
            self.renderer.AddActor(text_actor)
            if not hasattr(self, 'text_actors'):
                self.text_actors = []
            self.text_actors.append(text_actor)
        
        return base_assembly

    def update_coordinate_axes(self, position):
        """transformstextupdate"""
        # Update position
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.Translate(position[0], position[1], position[2])
        self.coordinate_axes_actor.SetUserTransform(transform)
        
        # Update position
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
        """STLtextupdate"""
        # Base_link blank_link true if process skip true
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node in self.stl_actors and node in self.transforms:
            print(f"Updating transform for node {node.name()} to position {point_xyz}, rotation {point_rpy}")
            transform = self.transforms[node]
            transform.Identity()

            # Apply parent transform
            if parent_transform is not None:
                transform.Concatenate(parent_transform)

            # Apply joint position
            transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])

            # Apply joint rotate RPY Roll-Pitch-Yaw RPY: Roll-Pitch-Yaw
            if point_rpy is not None and len(point_rpy) == 3:
                # Rpy transform rpy
                import math
                roll_deg = math.degrees(point_rpy[0])
                pitch_deg = math.degrees(point_rpy[1])
                yaw_deg = math.degrees(point_rpy[2])

                # Apply rotate RPY URDF : Z-Y-X
                transform.RotateZ(yaw_deg)    # Yaw
                transform.RotateY(pitch_deg)  # Pitch
                transform.RotateX(roll_deg)   # Roll

            self.stl_actors[node].SetUserTransform(transform)

            # Update transform
            self.update_collider_transform(node)

            # Update base_link node if
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
            # Show node if base_link
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No STL actor or transform found for node {node.name()}")

    def reset_stl_transform(self, node):
        """STLtextset"""
        # Base_link blank_link true if process skip true
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
                # If
                if node == self.base_connected_node:
                    self.update_coordinate_axes([0, 0, 0])
                    self.base_connected_node = None
                self.render_to_image()
                return

            # If
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None
            
            self.render_to_image()
        else:
            # Show node if base_link
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No transform found for node {node.name()}")

    def load_mesh_file(self, file_path):
        """
        meshtext(.stl, .obj, .dae)textVTK PolyDatatext

        Returns:
            tuple: (polydata, color) - colortextRGBAtextNone
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
        """nodetextmeshtext(.stl, .obj, .dae)text(text)"""
        # Base_link blank_link true if process skip true
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node.stl_file:
            # Compute TODO
            try:
                file_size = os.path.getsize(node.stl_file)
                # Todo
                # < 1mb 1mb : 50
                # 1-10mb 1-10mb : 70 30
                # > 10mb 10mb : 85 15
                file_size_mb = file_size / (1024 * 1024)
                if file_size_mb < 1:
                    load_weight = 50
                elif file_size_mb < 10:
                    load_weight = 70
                else:
                    load_weight = 85
            except:
                load_weight = 60  # Default value
                file_size_mb = 0

            # Start 100% 100
            if show_progress:
                self.show_progress(True)
                self.progress_bar.setValue(100)
                QtWidgets.QApplication.processEvents()

            # Todo
            if show_progress:
                remaining = 100 - (load_weight * 0.3)  # 30%
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            # Get load
            polydata, extracted_color = self.load_mesh_file(node.stl_file)

            if show_progress:
                remaining = 100 - load_weight  # NOTE
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            if polydata is None:
                print(f"ERROR: Failed to load mesh: {node.stl_file}")
                if show_progress:
                    self.show_progress(False)
                return

            # Mesh
            if extracted_color is not None:
                # Mesh
                node.mesh_original_color = extracted_color.copy()
                print(f"Stored mesh original color for node '{node.name()}': RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")
            else:
                # Set color None None
                if not hasattr(node, 'mesh_original_color'):
                    node.mesh_original_color = None
            
            # Foonode if foonode
            # Mesh
            if isinstance(node, FooNode):
                # Foonode if node_color existing color foonode
                if not hasattr(node, 'node_color') or node.node_color is None:
                    node.node_color = DEFAULT_COLOR_WHITE.copy()
                print(f"FooNode '{node.name()}': Skipped automatic color application (mesh color stored in mesh_original_color)")
            else:
                # Apply BaseLinkNode if BaseLinkNode
                if extracted_color is not None:
                    node.node_color = extracted_color
                    print(f"Applied color from .dae file to node '{node.name()}': RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")
                elif not hasattr(node, 'node_color') or node.node_color is None:
                    node.node_color = DEFAULT_COLOR_WHITE.copy()

            # Apply mesh scale URDF
            # Mesh scale
            if hasattr(node, 'mesh_scale'):
                mesh_scale = node.mesh_scale
                if mesh_scale != [1.0, 1.0, 1.0]:
                    # Create vtkTransform vtkTransform
                    scale_transform = vtk.vtkTransform()
                    scale_transform.Scale(mesh_scale[0], mesh_scale[1], mesh_scale[2])

                    # Apply vtkTransformPolyDataFilter scale vtkTransformPolyDataFilter
                    transform_filter = vtk.vtkTransformPolyDataFilter()
                    transform_filter.SetInputData(polydata)
                    transform_filter.SetTransform(scale_transform)
                    transform_filter.Update()

                    polydata = transform_filter.GetOutput()
                    print(f"Applied mesh scale {mesh_scale} to polydata for node '{node.name()}'")

            # Apply Visual origin mesh position Visual
            if hasattr(node, 'visual_origin'):
                visual_origin = node.visual_origin
                xyz = visual_origin.get('xyz', [0.0, 0.0, 0.0])
                rpy = visual_origin.get('rpy', [0.0, 0.0, 0.0])

                # Apply XYZ RPY if XYZ RPY
                if xyz != [0.0, 0.0, 0.0] or rpy != [0.0, 0.0, 0.0]:
                    import math
                    # Create vtkTransform vtkTransform
                    visual_transform = vtk.vtkTransform()

                    # Apply TODO
                    visual_transform.Translate(xyz[0], xyz[1], xyz[2])

                    # Apply next Yaw Pitch Roll RPY Yaw Pitch Roll
                    # Urdf rpy r rz yaw * ry pitch * rx roll urdf rpy: r rz ry rx
                    # Vtk postmultiply transform m t * rz * ry * rx vtk postmultiply m t rz ry rx
                    # Vtk rotate urdf transform vtk urdf
                    if rpy[2] != 0.0:  # Yaw (Z) -
                        visual_transform.RotateZ(math.degrees(rpy[2]))
                    if rpy[1] != 0.0:  # Pitch (Y)
                        visual_transform.RotateY(math.degrees(rpy[1]))
                    if rpy[0] != 0.0:  # Roll (X) -
                        visual_transform.RotateX(math.degrees(rpy[0]))

                    # Apply vtkTransformPolyDataFilter transform vtkTransformPolyDataFilter
                    visual_transform_filter = vtk.vtkTransformPolyDataFilter()
                    visual_transform_filter.SetInputData(polydata)
                    visual_transform_filter.SetTransform(visual_transform)
                    visual_transform_filter.Update()

                    polydata = visual_transform_filter.GetOutput()
                    print(f"Applied visual origin xyz={xyz}, rpy={rpy} (radians) to polydata for node '{node.name()}'")

            # Todo
            if show_progress:
                processing_weight = (100 - load_weight) * 0.6
                remaining = 100 - load_weight - processing_weight
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            # Set show
            # Node color
            if polydata.GetPointData().GetScalars() is not None:
                # Todo
                mapper.SetScalarModeToUsePointData()
                mapper.SetColorModeToDefault()  # NOTE
                print(f"Using vertex colors for node '{node.name()}'")
            elif polydata.GetCellData().GetScalars() is not None:
                # Todo
                mapper.SetScalarModeToUseCellData()
                mapper.SetColorModeToDefault()  # NOTE
                print(f"Using face colors for node '{node.name()}'")
            else:
                # Todo
                mapper.SetScalarModeToDefault()
                mapper.SetColorModeToDefault()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Transform scale
            transform = vtk.vtkTransform()
            transform.Identity()
            actor.SetUserTransform(transform)

            # Add TODO
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

            # Apply node
            # Foonode if foonode
            if not isinstance(node, FooNode):
                self.apply_color_to_node(node)
            else:
                # Foonode if mesh foonode
                print(f"FooNode '{node.name()}': Skipped automatic color application in apply_color_to_node")

            # Confirm Hide Hide Mesh
            if hasattr(node, 'hide_mesh') and node.hide_mesh:
                actor.SetVisibility(False)
                print(f"Applied hide_mesh on load: {node.name()} - mesh hidden")
            else:
                # Apply Hide Mesh current Hide Mesh Mesh/Wireframe
                if hasattr(self, 'mesh_toggle') and hasattr(self, 'wireframe_toggle'):
                    # Set button
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

            # Todo
            if show_progress:
                remaining = 5  # NOTE
                self.progress_bar.setValue(int(remaining))
                QtWidgets.QApplication.processEvents()

            self.reset_camera()
            self.render_to_image()

            # 0% 0
            if show_progress:
                self.progress_bar.setValue(0)
                QtWidgets.QApplication.processEvents()

                # Todo
                QTimer.singleShot(200, lambda: self.show_progress(False))

            # Add TODO
            print(f"Loaded: {node.stl_file} ({file_size_mb:.2f} MB)")

    def apply_color_to_node(self, node):
        """nodetextSTLmodeltext(RGBAtext)"""
        if node in self.stl_actors:
            actor = self.stl_actors[node]
            mapper = actor.GetMapper()
            
            # Show TODO
            if mapper and mapper.GetInput():
                polydata = mapper.GetInput()
                has_vertex_colors = polydata.GetPointData().GetScalars() is not None
                has_face_colors = polydata.GetCellData().GetScalars() is not None
                
                if has_vertex_colors or has_face_colors:
                    # Show node color
                    print(f"Node '{node.name()}' has vertex/face colors, skipping uniform color application")
                    # Todo
                    if hasattr(node, 'node_color') and node.node_color is not None and len(node.node_color) >= 4:
                        actor.GetProperty().SetOpacity(node.node_color[3])
                    else:
                        actor.GetProperty().SetOpacity(1.0)
                    self.render_to_image()
                    return
            
            # Apply color
            # Set color
            if not hasattr(node, 'node_color') or node.node_color is None:
                node.node_color = [1.0, 1.0, 1.0, 1.0]  # color（RGBA）

            # Apply color
            # Rgb 3
            actor.GetProperty().SetColor(*node.node_color[:3])

            # Alpha 4
            if len(node.node_color) >= 4:
                actor.GetProperty().SetOpacity(node.node_color[3])
            else:
                actor.GetProperty().SetOpacity(1.0)

            self.render_to_image()

    def remove_stl_for_node(self, node):
        """nodetextSTLtextCollidertextremove"""
        # Remove STL
        if node in self.stl_actors:
            self.renderer.RemoveActor(self.stl_actors[node])
            del self.stl_actors[node]
            if node in self.transforms:
                del self.transforms[node]

            # If
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None

        # Remove Collider
        if node in self.collider_actors:
            actors = self.collider_actors[node]
            # Actors list
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
        """textset - text(0,0,0)textdisplay"""
        camera = self.renderer.GetActiveCamera()

        # Todo
        camera.ParallelProjectionOn()

        # :
        # Set 1 1.
        camera.SetFocalPoint(0, 0, 0)  # Look at the origin

        # Set 2 2.
        camera.SetPosition(0.3, 0, 0)  # X（0.3）

        # Set 3 3.
        camera.SetViewUp(0, 0, 1)  # Z

        # 4 - center 4. windowcenter
        # Set 0 0 center
        camera.SetWindowCenter(0.0, 0.0)

        # 5 5. parallelscale
        # 0 0 15 0.1 0.15
        camera.SetParallelScale(0.15)

        # 6 all 6.
        self.renderer.ResetCameraClippingRange()

        print(f"Camera setup: Position={camera.GetPosition()}, FocalPoint={camera.GetFocalPoint()}, WindowCenter={camera.GetWindowCenter()}")

    def cleanup(self):
        """STLtextーtextーtextーtext"""
        # Vtk
        if hasattr(self, 'renderer'):
            if self.renderer:
                # Remove TODO
                for actor in self.renderer.GetActors():
                    self.renderer.RemoveActor(actor)

                # Remove TODO
                for actor in self.text_actors:
                    self.renderer.RemoveActor(actor)
                self.text_actors.clear()

        # Todo
        if hasattr(self, 'render_window'):
            if self.render_window:
                self.render_window.Finalize()

        # Todo
        self.stl_actors.clear()
        self.transforms.clear()

    def __del__(self):
        """textーtext"""
        self.cleanup()

    def update_rotation_axis(self, node, axis_id):
        """nodetextupdate"""
        try:
            print(f"Updating rotation axis for node {node.name()} to axis {axis_id}")
            
            if node in self.stl_actors and node in self.transforms:
                transform = self.transforms[node]
                actor = self.stl_actors[node]
                
                # Current position
                current_position = list(actor.GetPosition())
                
                # Transform
                transform.Identity()
                
                # Position
                transform.Translate(*current_position)
                
                # Set TODO
                # Add rotate process
                
                # Apply transform
                actor.SetUserTransform(transform)
                
                # Update view
                self.render_to_image()
                print(f"Successfully updated rotation axis for node {node.name()}")
            else:
                print(f"No STL actor or transform found for node {node.name()}")
                
        except Exception as e:
            print(f"Error updating rotation axis: {str(e)}")
            traceback.print_exc()

    def update_background(self, value):
        """textーtextupdate"""
        # -100 100 value 0 1 transform
        normalized_value = (value + 100) / 200.0
        self.renderer.SetBackground(normalized_value, normalized_value, normalized_value)
        self.render_to_image()

class ClosedLoopInspectorWindow(QtWidgets.QWidget):
    """textjointnodetextwindow"""

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

        # Initialize UI UI
        self.setup_ui()

        # Set TODO
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def setup_ui(self):
        """UItextinitialize"""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Todo
        self.button_style = UNIFIED_BUTTON_STYLE

        # Set TODO
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Todo
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
        # Set TODO
        self.joint_type_combo.setStyleSheet("QComboBox { color: black; background-color: white; }")
        self.joint_type_combo.currentTextChanged.connect(self.update_joint_type)
        type_layout.addWidget(self.joint_type_combo)
        content_layout.addLayout(type_layout)

        # Parent link parent link
        parent_layout = QtWidgets.QHBoxLayout()
        parent_layout.addWidget(QtWidgets.QLabel("Parent Link:"))
        self.parent_link_label = QtWidgets.QLabel("")
        self.parent_link_label.setStyleSheet("QLabel { color: #aaaaaa; }")
        parent_layout.addWidget(self.parent_link_label)
        parent_layout.addStretch()
        content_layout.addLayout(parent_layout)

        # Child link child link
        child_layout = QtWidgets.QHBoxLayout()
        child_layout.addWidget(QtWidgets.QLabel("Child Link:"))
        self.child_link_label = QtWidgets.QLabel("")
        self.child_link_label.setStyleSheet("QLabel { color: #aaaaaa; }")
        child_layout.addWidget(self.child_link_label)
        child_layout.addStretch()
        content_layout.addLayout(child_layout)

        # Todo
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

        # Show Origin RPY Origin RPY
        content_layout.addWidget(QtWidgets.QLabel("Origin Rotation (RPY in degrees):"))
        rpy_layout = QtWidgets.QHBoxLayout()
        self.origin_r_input = QtWidgets.QLineEdit()
        self.origin_p_input = QtWidgets.QLineEdit()
        self.origin_yaw_input = QtWidgets.QLineEdit()  # origin_yaw_inputswitch
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

        # Todo
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator2)

        # Gearbox /
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
        self.gearbox_widget.setVisible(False)  # display

        content_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # Todo
        close_button = QtWidgets.QPushButton("Close")
        close_button.setStyleSheet(self.button_style)
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button)

    def set_node(self, node):
        """nodetextdisplay"""
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
        # Confirm origin_xyz value :
        print(f"Setting origin_xyz for node {node.joint_name}: {node.origin_xyz}")
        self.origin_x_input.setText(f"{node.origin_xyz[0]}")
        self.origin_y_input.setText(f"{node.origin_xyz[1]}")
        self.origin_z_input.setText(f"{node.origin_xyz[2]}")

        # Origin rpy radian transform origin rpy 4
        self.origin_r_input.setText(str(round(math.degrees(node.origin_rpy[0]), 4)))
        self.origin_p_input.setText(str(round(math.degrees(node.origin_rpy[1]), 4)))
        self.origin_yaw_input.setText(str(round(math.degrees(node.origin_rpy[2]), 4)))

        # Gearbox
        if node.joint_type == 'gearbox':
            self.gearbox_widget.setVisible(True)
            self.gearbox_ratio_input.setText(str(node.gearbox_ratio))
            self.gearbox_ref_input.setText(node.gearbox_reference_body or "")
        else:
            self.gearbox_widget.setVisible(False)

    def update_joint_name(self):
        """Joint Nameupdate"""
        if self.current_node:
            self.current_node.joint_name = self.joint_name_input.text()
            print(f"Updated joint name to: {self.current_node.joint_name}")

    def update_joint_type(self, joint_type):
        """Joint Typeupdate"""
        if self.current_node:
            self.current_node.joint_type = joint_type
            # Gearbox /
            self.gearbox_widget.setVisible(joint_type == 'gearbox')
            print(f"Updated joint type to: {joint_type}")

    def update_origin_xyz(self):
        """Origin XYZupdate"""
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
        """Origin RPYupdate(text→radian)"""
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
        """Gearbox Ratioupdate"""
        if self.current_node:
            try:
                ratio = float(self.gearbox_ratio_input.text() or 1.0)
                self.current_node.gearbox_ratio = ratio
                print(f"Updated gearbox ratio to: {ratio}")
            except ValueError:
                print("Invalid gearbox ratio")

    def update_gearbox_reference(self):
        """Gearbox Reference Bodyupdate"""
        if self.current_node:
            self.current_node.gearbox_reference_body = self.gearbox_ref_input.text()
            print(f"Updated gearbox reference body to: {self.current_node.gearbox_reference_body}")

    def keyPressEvent(self, event):
        """keytexteventtexthandle"""
        # Confirm ESC
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        # Cmd+w macos ctrl+w windows/linux cmd+w macos ctrl+w windows/linux
        elif event.key() == QtCore.Qt.Key.Key_W and (
            event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or
            event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier
        ):
            self.close()
        else:
            # Todo
            super(ClosedLoopInspectorWindow, self).keyPressEvent(event)

class CustomNodeGraph(NodeGraph):
    def __init__(self, stl_viewer):
        super(CustomNodeGraph, self).__init__()
        self.stl_viewer = stl_viewer
        self.robot_name = "robot_x"
        self.project_dir = None
        self.meshes_dir = None
        self.last_save_dir = None
        self.mjcf_eulerseq = 'xyz'  # MJCFEuler（）
        self.closed_loop_joints = []  # NOTE

        # Todo
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

        # Todo
        # self.default_joint_friction = DEFAULT_JOINT_FRICTION
        # self.default_joint_actuation_lag = DEFAULT_JOINT_ACTUATION_LAG
        self.default_joint_damping = DEFAULT_JOINT_DAMPING
        self.default_joint_stiffness = DEFAULT_JOINT_STIFFNESS

        # Mjcf base_link mjcf z
        self.default_base_link_height = DEFAULT_BASE_LINK_HEIGHT

        # Set Node Grid Node Grid
        self.node_grid_enabled = DEFAULT_NODE_GRID_ENABLED
        self.node_grid_size = DEFAULT_NODE_GRID_SIZE

        # Todo
        self.highlight_color = DEFAULT_HIGHLIGHT_COLOR

        # Rgba rgba
        self.collision_color = DEFAULT_COLLISION_COLOR.copy()

        # Connect /
        self.port_connected.connect(self.on_port_connected)
        self.port_disconnected.connect(self.on_port_disconnected)

        # Todo
        try:
            # Baselinknode baselinknode
            self.register_node(BaseLinkNode)
            print(f"Registered node type: {BaseLinkNode.NODE_NAME}")

            # Foonode foonode
            self.register_node(FooNode)
            print(f"Registered node type: {FooNode.NODE_NAME}")

            # Closedloopjointnode closedloopjointnode
            self.register_node(ClosedLoopJointNode)
            print(f"Registered node type: {ClosedLoopJointNode.NODE_NAME}")

        except Exception as e:
            print(f"Error registering node types: {str(e)}")
            import traceback
            traceback.print_exc()

        # ...
        self._cleanup_handlers = []
        self._cached_positions = {}
        self._selection_cache = set()

        # Initialize var
        self._selection_start = None
        self._is_selecting = False

        # Initialize var
        self._is_panning = False
        self._pan_start = None

        # Node /
        self._node_clipboard = []

        # Set view
        self._view = self.widget

        # Qgraphicsview nodegraphwidget qtabwidget qgraphicsview nodegraphwidget qtabwidget
        self._viewer = self._view.currentWidget()  # NodeViewer

        # Nodegraphqt nodegraphqt
        # Pyside6
        try:
            if self._viewer:
                self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
                print("NodeGraphQt rubber band selection disabled")
            else:
                print("Could not access internal viewer, will handle errors in event handlers")
        except Exception as e:
            print(f"Could not disable NodeGraphQt drag mode: {e}")

        # Create TODO
        self._rubber_band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle,
            self._view
        )

        # Save TODO
        if self._viewer:
            self._original_handlers = {
                'press': self._viewer.mousePressEvent,
                'move': self._viewer.mouseMoveEvent,
                'release': self._viewer.mouseReleaseEvent,
                'keyPress': self._viewer.keyPressEvent
            }

            # Set TODO
            self._viewer.mousePressEvent = self.custom_mouse_press
            self._viewer.mouseMoveEvent = self.custom_mouse_move
            self._viewer.mouseReleaseEvent = self.custom_mouse_release
            self._viewer.keyPressEvent = self.custom_key_press
        else:
            self._original_handlers = {}

        # Initialize TODO
        self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)

        # Set timer
        self.last_selected_node = None
        self.selection_monitor_timer = QTimer()
        self.selection_monitor_timer.timeout.connect(self._check_selection_change)
        self.selection_monitor_timer.start(100)  # 100ms

        # Initialize run
        QtCore.QTimer.singleShot(100, self.update_grid_display)

    def _check_selection_change(self):
        """text"""
        selected_nodes = self.selected_nodes()

        if selected_nodes:
            # Get TODO
            current_selected = selected_nodes[0]

            # Todo
            if current_selected != self.last_selected_node:
                self.last_selected_node = current_selected
                if self.stl_viewer:
                    self.stl_viewer.highlight_node(current_selected)
        else:
            # Todo
            if self.last_selected_node is not None:
                self.last_selected_node = None
                if self.stl_viewer:
                    self.stl_viewer.clear_highlight()

    def custom_mouse_press(self, event):
        """texteventtext"""
        try:






            # Start TODO
            if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                print(">>> Starting pan operation (Middle Button Drag) - using custom panning")
                self._is_panning = True
                self._pan_start = event.position().toPoint()
                self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                # Initialize Qt
                self._original_handlers['press'](event)
                return

            # Process
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # Start Option Alt + Qt Option Alt Qt ScrollHandDrag
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    print(">>> Starting pan operation (Option+Drag) - using Qt ScrollHandDrag")
                    self._is_panning = True
                    self._pan_start = None  # ScrollHandDragNone
                    # Qt
                    self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
                    self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    # Run process Qt
                    self._original_handlers['press'](event)
                    return

                # Start Shift + Shift
                # Nodegraphqt shift : nodegraphqt shift
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    print(">>> Starting pan operation (Shift+Drag) - using custom panning")
                    self._is_panning = True
                    self._pan_start = event.position().toPoint()
                    self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    # Initialize Qt
                    self._original_handlers['press'](event)
                    return

                # Check
                pos = event.position().toPoint()
                print(f"Click position (view): {pos}")

                scene_pos = self._viewer.mapToScene(pos)
                print(f"Click position (scene): {scene_pos}")

                item_at_pos = self._viewer.scene().itemAt(scene_pos, self._viewer.transform())
                print(f"Item at position: {item_at_pos}")
                print(f"Item type: {type(item_at_pos)}")

                # Start TODO
                if item_at_pos is None or item_at_pos == self._viewer.scene():
                    print(">>> Starting rubber band selection")
                    self._selection_start = pos
                    self._is_selecting = True

                    # Set TODO
                    if self._rubber_band:
                        rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                        self._rubber_band.setGeometry(rect)
                        self._rubber_band.show()
                        print(f"Rubber band shown at: {rect}")

                    # Select ctrl
                    if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                        print("Clearing previous selection")
                        for node in self.selected_nodes():
                            node.set_selected(False)

                    # Todo
                    return
                else:
                    print(">>> Item found at click position, not starting rubber band selection")

            # If
            print("Calling original press handler")
            try:
                self._original_handlers['press'](event)
            except TypeError as te:
                # Nodegraphqt nodegraphqt pyside6
                print(f"Ignoring NodeGraphQt compatibility error: {te}")

        except Exception as e:
            print(f"Error in mouse press: {str(e)}")
            import traceback
            traceback.print_exc()

    def custom_mouse_move(self, event):
        """texteventtext"""
        try:
            # Process
            if self._is_panning:
                # Shift+
                if self._pan_start is not None:
                    current_pos = event.position().toPoint()

                    # Todo
                    previous_scene = self._viewer.mapToScene(self._pan_start)
                    current_scene = self._viewer.mapToScene(current_pos)
                    delta = previous_scene - current_scene

                    print(f"Custom pan: delta=({delta.x()}, {delta.y()})")

                    # Nodegraphqt nodegraphqt
                    self._viewer._set_viewer_pan(delta.x(), delta.y())

                    self._pan_start = current_pos
                    return
                else:
                    # Scrollhanddrag qt scrollhanddrag option+
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

                # Todo
                return

            # Todo
            try:
                self._original_handlers['move'](event)
            except TypeError as te:
                # Nodegraphqt nodegraphqt pyside6
                pass  # Movedisplay

        except Exception as e:
            print(f"Error in mouse move: {str(e)}")
            import traceback
            traceback.print_exc()

    def custom_mouse_release(self, event):
        """textーtexteventtext"""
        try:


            print(f"Is selecting: {self._is_selecting}")
            print(f"Is panning: {self._is_panning}")

            # Quit Shift/Option+
            if self._is_panning and (event.button() == QtCore.Qt.MouseButton.MiddleButton or
                                      event.button() == QtCore.Qt.MouseButton.LeftButton):
                print(">>> Ending pan operation")
                # Process qt
                try:
                    self._original_handlers['release'](event)
                except TypeError:
                    pass
                # Todo
                self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
                self._viewer.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                self._is_panning = False
                self._pan_start = None
                return

            if event.button() == QtCore.Qt.MouseButton.LeftButton and self._is_selecting:
                if self._rubber_band and self._selection_start:
                    # Process
                    rect = self._rubber_band.geometry()
                    scene_rect = self._viewer.mapToScene(rect).boundingRect()
                    print(f"Selection rect (view): {rect}")
                    print(f"Selection rect (scene): {scene_rect}")

                    # Node select
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

                    # Todo
                    self._rubber_band.hide()

                # Todo
                self._selection_start = None
                self._is_selecting = False

                # Todo
                return

            # Node :
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self.node_grid_enabled:
                for node in self.selected_nodes():
                    node_pos = node.pos()
                    if isinstance(node_pos, (list, tuple)):
                        current_x, current_y = node_pos[0], node_pos[1]
                    else:
                        current_x, current_y = node_pos.x(), node_pos.y()

                    snapped_x, snapped_y = self.snap_to_grid(current_x, current_y)

                    # Position
                    if abs(snapped_x - current_x) > 0.1 or abs(snapped_y - current_y) > 0.1:
                        node.set_pos(snapped_x, snapped_y)
                        print(f"Snapped node '{node.name()}' to grid: ({current_x:.1f}, {current_y:.1f}) -> ({snapped_x}, {snapped_y})")

            # If
            print("Calling original release handler")
            try:
                self._original_handlers['release'](event)
            except TypeError as te:
                # Nodegraphqt nodegraphqt pyside6
                print(f"Ignoring NodeGraphQt compatibility error: {te}")

        except Exception as e:
            print(f"Error in mouse release: {str(e)}")
            import traceback
            traceback.print_exc()

    def copy_nodes(self):
        """textnodetextーtextー"""
        selected = self.selected_nodes()
        if not selected:
            print("No nodes selected to copy")
            return

        # Baselinknode baselinknode
        nodes_to_copy = [node for node in selected if not isinstance(node, BaseLinkNode)]
        if not nodes_to_copy:
            print("Cannot copy BaseLinkNode")
            return

        self._node_clipboard = []
        for node in nodes_to_copy:
            # Get node position list transform
            pos = node.pos()
            if isinstance(pos, (list, tuple)):
                original_pos = [float(pos[0]), float(pos[1])]
            else:
                # Qpointf if qpointf
                original_pos = [float(pos.x()), float(pos.y())]

            # Node
            node_data = {
                'type': node.__class__.__name__,
                'name': node.name(),
                'original_pos': original_pos,  # NOTE
                'properties': {}
            }

            # Todo
            for prop_name in node.model.custom_properties.keys():
                try:
                    node_data['properties'][prop_name] = node.get_property(prop_name)
                except:
                    pass

            self._node_clipboard.append(node_data)

        print(f"Copied {len(self._node_clipboard)} node(s)")

    def paste_nodes(self):
        """textーtextnodetextーtext"""
        if not self._node_clipboard:
            print("Clipboard is empty")
            return

        # Disconnect existing select
        for node in self.selected_nodes():
            node.set_selected(False)

        pasted_nodes = []
        offset = 50  # NOTE

        for node_data in self._node_clipboard:
            try:
                # Get TODO
                node_type = node_data['type']

                # Get TODO
                if node_type == 'FooNode':
                    node_class = 'insilico.nodes.FooNode'
                elif node_type == 'ClosedLoopJointNode':
                    node_class = 'insilico.nodes.ClosedLoopJointNode'
                else:
                    print(f"Unknown node type: {node_type}")
                    continue

                # Generate TODO
                base_name = node_data['name']
                new_name = base_name
                counter = 1
                existing_names = [n.name() for n in self.all_nodes()]
                while new_name in existing_names:
                    new_name = f"{base_name}_{counter}"
                    counter += 1

                # Compute position 50px
                original_pos = node_data['original_pos']
                new_pos = [original_pos[0] + offset, original_pos[1] + offset]

                # Create node
                new_node = self.create_node(
                    node_class,
                    name=new_name,
                    pos=new_pos
                )

                # Todo
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

        # Select
        for node in pasted_nodes:
            node.set_selected(True)

        print(f"Pasted {len(pasted_nodes)} node(s)")

    def cut_nodes(self):
        """textnodetext(textーtextremove)"""
        selected = self.selected_nodes()
        if not selected:
            print("No nodes selected to cut")
            return

        # Todo
        self.copy_nodes()

        # Remove BaseLinkNode BaseLinkNode
        nodes_to_delete = [node for node in selected if not isinstance(node, BaseLinkNode)]
        for node in nodes_to_delete:
            self.delete_node(node)

        print(f"Cut {len(nodes_to_delete)} node(s)")

    def duplicate_nodes(self):
        """textnodetext"""
        # Todo
        self.copy_nodes()

        # Todo
        self.paste_nodes()

    def custom_key_press(self, event):
        """textkeytexteventtext"""
        try:
            # Check ctrl/command
            is_ctrl_cmd = (
                event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or
                event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier
            )

            # Ctrl/command+c ctrl/command+c
            if event.key() == QtCore.Qt.Key.Key_C and is_ctrl_cmd:
                print("\n=== Copy Nodes (Ctrl/Cmd+C) ===")
                self.copy_nodes()
                event.accept()
                return

            # Ctrl/command+v ctrl/command+v
            if event.key() == QtCore.Qt.Key.Key_V and is_ctrl_cmd:
                print("\n=== Paste Nodes (Ctrl/Cmd+V) ===")
                self.paste_nodes()
                event.accept()
                return

            # Ctrl/command+x ctrl/command+x
            if event.key() == QtCore.Qt.Key.Key_X and is_ctrl_cmd:
                print("\n=== Cut Nodes (Ctrl/Cmd+X) ===")
                self.cut_nodes()
                event.accept()
                return

            # Ctrl/command+d ctrl/command+d
            if event.key() == QtCore.Qt.Key.Key_D and is_ctrl_cmd:
                print("\n=== Duplicate Nodes (Ctrl/Cmd+D) ===")
                self.duplicate_nodes()
                event.accept()
                return

            # Ctrl/command+a select ctrl/command+a base
            if event.key() == QtCore.Qt.Key.Key_A and is_ctrl_cmd:
                print("\n=== Select All Nodes (Ctrl/Cmd+A) ===")
                # Select base
                all_nodes = self.all_nodes()
                selected_count = 0

                for node in all_nodes:
                    # Baselinknode baselinknode
                    if not isinstance(node, BaseLinkNode):
                        node.set_selected(True)
                        selected_count += 1
                    else:
                        # Baselinknode baselinknode
                        node.set_selected(False)

                print(f"Selected {selected_count} nodes (excluding Base)")
                event.accept()
                return

            # Delete backspace
            if event.key() in [QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace]:
                print("\n=== Delete/Backspace Key Pressed ===")
                # Remove TODO
                delete_selected_node(self)
                # Event
                event.accept()
                return

            # Todo
            try:
                self._original_handlers['keyPress'](event)
            except (TypeError, KeyError):
                # Todo
                pass

        except Exception as e:
            print(f"Error in key press: {str(e)}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """textーtextーtext"""
        try:
            print("Starting cleanup process...")
            
            # Todo
            if hasattr(self, '_viewer') and self._viewer:
                if hasattr(self, '_original_handlers'):
                    self._viewer.mousePressEvent = self._original_handlers['press']
                    self._viewer.mouseMoveEvent = self._original_handlers['move']
                    self._viewer.mouseReleaseEvent = self._original_handlers['release']
                    if 'keyPress' in self._original_handlers:
                        self._viewer.keyPressEvent = self._original_handlers['keyPress']
                    print("Restored original event handlers")

            # Todo
            try:
                if hasattr(self, '_rubber_band') and self._rubber_band and not self._rubber_band.isHidden():
                    self._rubber_band.hide()
                    self._rubber_band.setParent(None)
                    self._rubber_band.deleteLater()
                    self._rubber_band = None
                    print("Cleaned up rubber band")
            except Exception as e:
                print(f"Warning: Rubber band cleanup - {str(e)}")
                
            # Node
            for node in self.all_nodes():
                try:
                    # Stl
                    if self.stl_viewer:
                        self.stl_viewer.remove_stl_for_node(node)
                    # Remove node
                    self.remove_node(node)
                except Exception as e:
                    print(f"Error cleaning up node: {str(e)}")

            # Todo
            if hasattr(self, 'inspector_window') and self.inspector_window:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                    self.inspector_window = None
                    print("Cleaned up inspector window")
                except Exception as e:
                    print(f"Error cleaning up inspector window: {str(e)}")

            # Todo
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
        """textーtext"""
        self.cleanup()

    def remove_node(self, node):
        """noderemovetextーtext"""
        # Remove TODO
        if node in self._cached_positions:
            del self._cached_positions[node]
        self._selection_cache.discard(node)

        # Disconnect TODO
        for port in node.input_ports():
            for connected_port in port.connected_ports():
                self.disconnect_ports(port, connected_port)
        
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                self.disconnect_ports(port, connected_port)

        # Stl
        if self.stl_viewer:
            self.stl_viewer.remove_stl_for_node(node)

        super(CustomNodeGraph, self).remove_node(node)

    def optimize_node_positions(self):
        """nodetext"""
        # Todo
        for node in self.all_nodes():
            if node not in self._cached_positions:
                pos = self.calculate_node_position(node)
                self._cached_positions[node] = pos
            node.set_pos(*self._cached_positions[node])

    def setup_custom_view(self):
        """textーtexteventtext"""
        # Save TODO
        self._view.mousePressEvent_original = self._view.mousePressEvent
        self._view.mouseMoveEvent_original = self._view.mouseMoveEvent
        self._view.mouseReleaseEvent_original = self._view.mouseReleaseEvent

        # Set TODO
        self._view.mousePressEvent = lambda event: self.custom_mouse_press(event)
        self._view.mouseMoveEvent = lambda event: self.custom_mouse_move(event)
        self._view.mouseReleaseEvent = lambda event: self.custom_mouse_release(event)

    def eventFilter(self, obj, event):
        """eventtextーtexteventtexthandle"""
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
            # Get view
            view = self.scene().views()[0]
            self._selection_start = view.mapFromGlobal(event.globalPos())
            
            # Existing select ctrl
            if not event.modifiers() & QtCore.Qt.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)
            
            # Set TODO
            self._rubber_band.setGeometry(QtCore.QRect(self._selection_start, QtCore.QSize()))
            self._rubber_band.show()
        
        super(CustomNodeGraph, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_start is not None:
            # Get view
            view = self.scene().views()[0]
            current_pos = view.mapFromGlobal(event.globalPos())
            
            # Update TODO
            rect = QtCore.QRect(self._selection_start, current_pos).normalized()
            self._rubber_band.setGeometry(rect)
        
        super(CustomNodeGraph, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._selection_start is not None:
            # Get view
            view = self.scene().views()[0]
            rubber_band_rect = self._rubber_band.geometry()
            scene_rect = view.mapToScene(rubber_band_rect).boundingRect()
            
            # Node select
            for node in self.all_nodes():
                node_center = QtCore.QPointF(node.pos()[0], node.pos()[1])
                if scene_rect.contains(node_center):
                    node.set_selected(True)
            
            # Todo
            self._rubber_band.hide()
            self._selection_start = None
        
        super(CustomNodeGraph, self).mouseReleaseEvent(event)

    def create_base_link(self):
        """textbase_linknodetext"""
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
        """textnodetext"""
        for node_class in node_classes:
            self.register_node(node_class)
            print(f"Registered node type: {node_class.__identifier__}")

    def update_node_color_by_connection(self, node):
        """nodetextupdate"""
        # Baselinknode baselinknode
        if isinstance(node, BaseLinkNode):
            node.set_color(45, 45, 45)  # NOTE
            return

        # Connect check
        has_input_connection = False
        for input_port in node.input_ports():
            if input_port.connected_ports():
                has_input_connection = True
                break

        if has_input_connection:
            # Todo
            node.set_color(45, 45, 45)  # NOTE
        else:
            # Connect TODO
            node.set_color(74, 84, 85)  # NOTE

    def update_all_node_colors(self):
        """alltextnodetextupdate"""
        for node in self.all_nodes():
            self.update_node_color_by_connection(node)

    def apply_cyan_color_to_connection(self, input_port, output_port):
        """text"""
        try:
            # Port view
            # Get view
            if hasattr(output_port, 'view') and output_port.view:
                port_view = output_port.view
                # Get view connected_pipes
                if hasattr(port_view, 'connected_pipes'):
                    for pipe in port_view.connected_pipes:
                        # Connect confirm
                        if hasattr(pipe, 'port_type') or True:  # NOTE
                            # Change RGB 0 180 180 RGB
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
        """textnodetext"""
        try:
            cyan_count = 0

            # All node check
            for node in self.all_nodes():
                # Check
                if isinstance(node, ClosedLoopJointNode):
                    # Check
                    for port in node.input_ports() + node.output_ports():
                        if hasattr(port, 'view') and port.view:
                            port_view = port.view
                            # Get view
                            if hasattr(port_view, 'connected_pipes'):
                                for pipe in port_view.connected_pipes:
                                    # Change RGB 0 180 180 RGB
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
        """textnodetextInertiatext、MuJoCotextnodetext"""
        import numpy as np
        
        # Save color
        invalid_nodes = []
        valid_nodes = []
        
        print("\n=== Checking Inertia for All Nodes ===")
        
        for node in self.all_nodes():
            # Baselinknode skip baselinknode inertia
            if isinstance(node, BaseLinkNode):
                continue
            
            # Get node inertia
            if not hasattr(node, 'inertia') or not node.inertia:
                continue
            
            inertia_dict = node.inertia
            mass = getattr(node, 'mass', 0.0)
            
            # Create inertia 3x3
            try:
                ixx = inertia_dict.get('ixx', 0.0)
                iyy = inertia_dict.get('iyy', 0.0)
                izz = inertia_dict.get('izz', 0.0)
                ixy = inertia_dict.get('ixy', 0.0)
                ixz = inertia_dict.get('ixz', 0.0)
                iyz = inertia_dict.get('iyz', 0.0)
                
                # Create 3x3 inertia tensor
                inertia_tensor = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz]
                ])
                
                # Mujoco check a + b > c mujoco : a b c
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
                # Disable error
                invalid_nodes.append(node)
        
        # Disable node
        for node in invalid_nodes:
            node.set_color(255, 200, 200)  # NOTE
        
        # Enable node color
        for node in valid_nodes:
            self.update_node_color_by_connection(node)
        
        print(f"\n=== Inertia Check Complete ===")
        print(f"  Valid nodes: {len(valid_nodes)}")
        print(f"  Invalid nodes: {len(invalid_nodes)}")
        
        # Show TODO
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Inertia Check Result")
        
        if invalid_nodes:
            # Error
            invalid_node_names = [node.name() for node in invalid_nodes]
            invalid_count = len(invalid_nodes)
            
            # Create show 10
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
            # If ok
            message = f"✓ All nodes have valid inertia!\n\nChecked {len(valid_nodes)} node(s)."
            
            msg_box.setIcon(QtWidgets.QMessageBox.Information)
            msg_box.setText(message)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            
            print("\n✓ All nodes have valid inertia!")
        
        # Show dialog
        msg_box.exec()

    def snap_to_grid(self, x, y):
        """transformstext

        Args:
            x (float): Xtransforms
            y (float): Ytransforms

        Returns:
            tuple: text(x, y)transforms
        """
        if not self.node_grid_enabled:
            return (x, y)

        grid_size = self.node_grid_size
        snapped_x = round(x / grid_size) * grid_size
        snapped_y = round(y / grid_size) * grid_size
        return (snapped_x, snapped_y)

    def update_grid_display(self):
        """textdisplaytextupdate"""
        try:
            if hasattr(self, '_viewer') and self._viewer:
                # Set NodeGraphQt NodeGraphQt viewer
                if hasattr(self._viewer, 'set_grid_size'):
                    self._viewer.set_grid_size(self.node_grid_size)
                    print(f"Grid size updated to: {self.node_grid_size}")
                elif hasattr(self._viewer, '_grid_size'):
                    # Nodegraphqt nodegraphqt
                    self._viewer._grid_size = self.node_grid_size
                    # View
                    if hasattr(self._viewer, 'update'):
                        self._viewer.update()
                    print(f"Grid size updated (direct) to: {self.node_grid_size}")
                else:
                    print("Warning: Could not update grid size (viewer does not support grid customization)")
        except Exception as e:
            print(f"Error updating grid display: {str(e)}")

    def on_port_connected(self, input_port, output_port):
        """textーtexthandle"""
        print(f"**Connecting port: {output_port.name()}")

        # Output
        parent_node = output_port.node()
        child_node = input_port.node()
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")

        try:
            # Connect check
            is_closed_loop_connection = isinstance(parent_node, ClosedLoopJointNode) or isinstance(child_node, ClosedLoopJointNode)

            if is_closed_loop_connection:
                # Change TODO
                try:
                    # Todo
                    QtCore.QTimer.singleShot(100, lambda: self.apply_cyan_color_to_connection(input_port, output_port))
                    print(f"  ✓ Scheduled cyan color application for closed-loop connection")
                except Exception as pipe_error:
                    print(f"  ⚠ Warning: Could not schedule cyan color: {str(pipe_error)}")

            # Update color
            self.update_node_color_by_connection(child_node)

            # Position
            print("Recalculating all node positions after connection...")
            self.recalculate_all_positions()

        except Exception as e:
            print(f"Error in port connection: {str(e)}")
            print(f"Detailed connection information:")
            print(f"  Output port: {output_port.name()} from {parent_node.name()}")
            print(f"  Input port: {input_port.name()} from {child_node.name()}")
            traceback.print_exc()

    def on_port_disconnected(self, input_port, output_port):
        """textーtexthandle"""
        child_node = input_port.node()  # NOTE
        parent_node = output_port.node()  # NOTE

        print(f"\nDisconnecting ports:")
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")

        try:
            # Position
            if hasattr(child_node, 'current_transform'):
                del child_node.current_transform

            # Stl position stl
            self.stl_viewer.reset_stl_transform(child_node)
            print(f"Reset position for node: {child_node.name()}")

            # Update color
            self.update_node_color_by_connection(child_node)

            # Position
            print("Recalculating all node positions after disconnection...")
            self.recalculate_all_positions()

        except Exception as e:
            print(f"Error in port disconnection: {str(e)}")
            traceback.print_exc()

    def update_robot_name(self, text):
        """textupdatetext"""
        self.robot_name = text
        print(f"Robot name updated to: {text}")

        # Process
        # Update TODO
        if hasattr(self, 'widget') and self.widget:
            if self.widget.window():
                title = f"URDF Kitchen - Assembler v0.1.0 -"
                self.widget.window().setWindowTitle(title)

    def get_robot_name(self):
        """
        textgettext
        Returns:
            str: text
        """
        return self.robot_name

    def set_robot_name(self, name):
        """
        textsettext
        Args:
            name (str): settext
        """
        self.robot_name = name
        # Update TODO
        if hasattr(self, 'name_input') and self.name_input:
            self.name_input.setText(name)
        print(f"Robot name set to: {name}")

    def clean_robot_name(self, name):
        """text_descriptiontext"""
        if name.endswith('_description'):
            return name[:-12]  # '_description'(12)
        return name

    def update_robot_name_from_directory(self, dir_path):
        """textupdate"""
        dir_name = os.path.basename(dir_path)
        if dir_name.endswith('_description'):
            robot_name = dir_name[:-12]
            # Ui
            if hasattr(self, 'name_input') and self.name_input:
                self.name_input.setText(robot_name)
            self.robot_name = robot_name
            return True
        return False

    def _quat_to_rpy(self, quat):
        """textーtext (w, x, y, z) textRPY (roll, pitch, yaw) text

        Note: This method now delegates to ConversionUtils.quat_to_rpy in utils.py

        Args:
            quat: [w, x, y, z] quaternion (MuJoCo convention)

        Returns:
            [roll, pitch, yaw] in radians
        """
        return ConversionUtils.quat_to_rpy(quat)

    def _euler_to_rpy(self, euler_angles, sequence='xyz'):
        """Eulertext(text)textRPY(text)text

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
        """XMLtextー

        Args:
            attr_string: textーtext
            default: text

        Returns:
            floattext、textdefault
        """
        if not attr_string:
            return default
        try:
            return [float(v) for v in attr_string.split()]
        except (ValueError, AttributeError):
            return default

    def _parse_xyz(self, elem, attr='xyz', default=None):
        """XMLtextxyztextgettext

        Args:
            elem: XMLtext
            attr: text(text: 'xyz')
            default: text

        Returns:
            [x, y, z]text、textdefault
        """
        if elem is None:
            return default if default is not None else [0.0, 0.0, 0.0]
        xyz_str = elem.get(attr, '0 0 0')
        result = self._parse_float_list(xyz_str)
        return result if result and len(result) == 3 else (default if default is not None else [0.0, 0.0, 0.0])

    def _parse_rpy(self, elem, attr='rpy', default=None):
        """XMLtextrpytextgettext

        Args:
            elem: XMLtext
            attr: text(text: 'rpy')
            default: text

        Returns:
            [roll, pitch, yaw]text、textdefault
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
        """alltextnodetextーtext3Dtextーtext(Load Project、URDF、MJCFloadtext)
        
        STLloadtext、alltextnodetextーtext3Dtextーtext。
        nodetext。
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
                
                # Apply node STL
                if has_stl_file and in_actors:
                    # Confirm node node_color node.node_color
                    if has_node_color:
                        rgba_values = node.node_color
                        # 0-1 rgba
                        rgba_values = [max(0.0, min(1.0, float(v))) for v in rgba_values[:4]]
                        
                        actor = self.stl_viewer.stl_actors[node]
                        
                        # Check
                        mapper = actor.GetMapper()
                        has_scalars = False
                        if mapper and mapper.GetInput():
                            polydata = mapper.GetInput()
                            has_vertex_colors = polydata.GetPointData().GetScalars() is not None
                            has_face_colors = polydata.GetCellData().GetScalars() is not None
                            has_scalars = has_vertex_colors or has_face_colors
                        
                        if has_scalars:
                            # Todo
                            if len(rgba_values) >= 4:
                                actor.GetProperty().SetOpacity(rgba_values[3])
                            else:
                                actor.GetProperty().SetOpacity(1.0)
                            print(f"Node '{node_name}' has vertex/face colors, only opacity applied: {rgba_values[3] if len(rgba_values) >= 4 else 1.0}")
                        else:
                            # Apply color
                            # Rgb 3
                            actor.GetProperty().SetColor(*rgba_values[:3])
                            # Alpha 4
                            if len(rgba_values) >= 4:
                                actor.GetProperty().SetOpacity(rgba_values[3])
                            else:
                                actor.GetProperty().SetOpacity(1.0)
                            print(f"Applied color to node '{node_name}': RGBA{rgba_values[:4]}")
                            applied_count += 1
                    else:
                        # Apply TODO
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
        
        # Update 3D
        if applied_count > 0:
            self.stl_viewer.render_to_image()


    def export_urdf(self):
        """URDFtextーtext"""
        try:
            # Todo
            self.collect_closed_loop_joints_from_nodes()

            # Get TODO
            robot_base_name = self.robot_name or "robot"
            clean_name = self.clean_robot_name(robot_base_name)

            # Select
            parent_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select parent directory for URDF export",
                os.getcwd()
            )

            if not parent_dir:
                print("URDF export cancelled")
                return False

            # Show warning
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

            # Select
            mesh_format_dialog = QtWidgets.QDialog(self.widget)
            mesh_format_dialog.setWindowTitle("Select Mesh Format")
            mesh_format_dialog.setModal(True)
            layout = QtWidgets.QVBoxLayout()
            
            label = QtWidgets.QLabel("Select mesh file format for export:")
            layout.addWidget(label)
            
            format_group = QtWidgets.QButtonGroup()
            stl_radio = QtWidgets.QRadioButton(".stl (STL)")
            stl_radio.setChecked(True)  # .stl
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
            
            # Get TODO
            selected_format = ".stl" if stl_radio.isChecked() else ".dae"
            print(f"Selected mesh format: {selected_format}")

            # Create _description
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

            # Create path meshes urdf
            urdf_dir = os.path.join(description_dir, 'urdf')

            # Create urdf
            if not os.path.exists(urdf_dir):
                try:
                    os.makedirs(urdf_dir)
                    print(f"Created URDF directory: {urdf_dir}")
                except Exception as e:
                    print(f"Error creating URDF directory: {str(e)}")
                    return False

            # Create path meshes
            meshes_dir = os.path.join(description_dir, 'meshes')

            # Create meshes
            if not os.path.exists(meshes_dir):
                try:
                    os.makedirs(meshes_dir)
                    print(f"Created meshes directory: {meshes_dir}")
                except Exception as e:
                    print(f"Error creating meshes directory: {str(e)}")
                    return False

            # Todo
            stl_files_copied = []
            stl_files_failed = []

            for node in self.all_nodes():
                # Skip nodes with "Hide Mesh" enabled
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                if hasattr(node, 'stl_file') and node.stl_file:
                    source_path = node.stl_file
                    if os.path.exists(source_path):
                        stl_filename = os.path.basename(source_path)
                        dest_path = os.path.join(meshes_dir, stl_filename)
                        file_ext = os.path.splitext(stl_filename)[1].lower()

                        try:
                            # Save TODO
                            poly_data, volume, extracted_color = load_mesh_to_polydata(source_path)
                            
                            # Change TODO
                            base_name = os.path.splitext(stl_filename)[0]
                            new_filename = f"{base_name}{selected_format}"
                            dest_path = os.path.join(meshes_dir, new_filename)
                            
                            # Get node
                            mesh_color = None
                            color_manually_changed = False
                            if hasattr(node, 'node_color'):
                                mesh_color = node.node_color
                                color_manually_changed = True
                            
                            # Save TODO
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

                # Colliders
                if hasattr(node, 'colliders') and node.colliders:
                    for collider in node.colliders:
                        if not collider.get('enabled', False):
                            continue
                        if collider.get('type') == 'mesh' and collider.get('mesh'):
                            collider_mesh = collider['mesh']
                            # Collider_mesh
                            if hasattr(node, 'stl_file') and node.stl_file:
                                visual_dir = os.path.dirname(node.stl_file)
                                collider_source_path = os.path.join(visual_dir, collider_mesh)
                            else:
                                collider_source_path = collider_mesh

                            if os.path.exists(collider_source_path):
                                collider_original_filename = os.path.basename(collider_source_path)
                                # Change TODO
                                collider_base_name = os.path.splitext(collider_original_filename)[0]
                                collider_new_filename = f"{collider_base_name}{selected_format}"
                                collider_dest_path = os.path.join(meshes_dir, collider_new_filename)

                                # Todo
                                if collider_new_filename not in stl_files_copied:
                                    try:
                                        # Save TODO
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

            # Set path name URDF
            urdf_file = os.path.join(urdf_dir, f"{clean_name}.urdf")

            # Urdf
            with open(urdf_file, 'w', encoding='utf-8') as f:
                # Name
                f.write('<?xml version="1.0"?>\n')
                f.write(f'<robot name="{clean_name}">\n\n')

                # Todo
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
                
                # Export material
                f.write('<!-- material color setting -->\n')
                for hex_color, rgb in materials.items():
                    f.write(f'<material name="{hex_color}">\n')
                    f.write(f'  <color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"/>\n')
                    f.write('</material>\n')
                f.write('\n')

                # Base_link output
                visited_nodes = set()
                base_node = self.get_node_by_name('base_link')
                if base_node:
                    self._write_tree_structure(f, base_node, None, visited_nodes, materials)
                
                f.write('</robot>\n')

                print(f"URDF exported to: {urdf_file}")

                # Create TODO
                export_summary = "✓ URDF Export Completed Successfully\n"
                export_summary += "=" * 50 + "\n\n"
                export_summary += f"Robot Name: {clean_name}\n\n"
                export_summary += f"Output Directory:\n{description_dir}\n\n"
                export_summary += f"URDF File:\n{urdf_file}\n\n"
                export_summary += f"Meshes Directory:\n{meshes_dir}\n\n"
                export_summary += f"Mesh Files Copied: {len(stl_files_copied)}\n"

                if stl_files_copied:
                    export_summary += "\nCopied Mesh Files:\n"
                    for filename in stl_files_copied[:10]:  # Max10up todisplay
                        export_summary += f"  • {filename}\n"
                    if len(stl_files_copied) > 10:
                        export_summary += f"  ... and {len(stl_files_copied) - 10} more\n"

                if stl_files_failed:
                    export_summary += f"\n⚠ Warning: {len(stl_files_failed)} file(s) failed to copy\n"
                    for filename, error in stl_files_failed[:5]:  # Max5up todisplay
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
        """textーtext"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless skip <visual> massless decoration visual
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        # Skip nodes with "Hide Mesh" enabled
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            print(f"Skipping node with hide_mesh=True: {node.name()}")
            return

        if node.name() == "base_link":
            # Base_link output
            self._write_base_link(file)
        
        # Current node link process
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                if child_node not in visited_nodes:
                    # Massless decoration link output massless decoration
                    if not (hasattr(child_node, 'massless_decoration') and child_node.massless_decoration):
                        # Output
                        self._write_joint(file, node, child_node)
                        file.write('\n')
                        
                        # Next link output
                        self._write_link(file, child_node, materials, mesh_format)
                        file.write('\n')
                    
                    # Process
                    self._write_tree_structure(file, child_node, node, visited_nodes, materials, mesh_format)

    def _is_base_link_at_defaults(self, base_node):
        """base_linktextalltext"""
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
        """base_linktext"""
        base_node = self.get_node_by_name('base_link')

        # If link output blanklink
        is_blank = (base_node and hasattr(base_node, 'blank_link') and base_node.blank_link)
        is_all_defaults = self._is_base_link_at_defaults(base_node)

        if base_node and not is_blank and not is_all_defaults:
            # Blanklink if link output blanklink
            file.write('  <link name="base_link">\n')

            # Todo
            if hasattr(base_node, 'mass_value') and hasattr(base_node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial origin output inertial origin
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

            # Todo
            if hasattr(base_node, 'stl_file') and base_node.stl_file:
                mesh_dir_name = "meshes"
                if self.meshes_dir:
                    dir_name = os.path.basename(self.meshes_dir)
                    if dir_name.startswith('mesh'):
                        mesh_dir_name = dir_name

                stl_filename = os.path.basename(base_node.stl_file)
                package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{stl_filename}"

                # Todo
                file.write('    <visual>\n')
                # Visual origin output visual
                file.write(self._format_visual_origin(base_node))
                file.write('      <geometry>\n')
                # Mesh scale output mesh
                scale_attr = self._format_mesh_scale(base_node)
                file.write(f'        <mesh filename="{package_path}"{scale_attr}/>\n')
                file.write('      </geometry>\n')

                # Add TODO
                if hasattr(base_node, 'node_color') and len(base_node.node_color) >= 3:
                    rgb = base_node.node_color
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(rgb[0] * 255),
                        int(rgb[1] * 255),
                        int(rgb[2] * 255)
                    )
                    file.write(f'      <material name="{hex_color}"/>\n')

                file.write('    </visual>\n')

                # Todo
                self._write_urdf_collision(file, base_node, package_path, mesh_dir_name)

            file.write('  </link>\n\n')
        else:
            # Blanklink if link output blanklink
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
        nodetextwindowtextdisplay
        """
        try:
            # Existing
            if hasattr(self, 'inspector_window') and self.inspector_window is not None:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                except Exception:
                    pass
                self.inspector_window = None

            # Create TODO
            self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)
            
            # Get TODO
            inspector_size = self.inspector_window.sizeHint()

            if self.widget and self.widget.window():
                # Compute TODO
                if hasattr(self, 'last_inspector_position') and self.last_inspector_position:
                    x = self.last_inspector_position.x()
                    y = self.last_inspector_position.y()
                    
                    # Todo
                    screen = QtWidgets.QApplication.primaryScreen()
                    screen_geo = screen.availableGeometry()
                    
                    # Confirm TODO
                    if x < screen_geo.x() or x + inspector_size.width() > screen_geo.right() or \
                    y < screen_geo.y() or y + inspector_size.height() > screen_geo.bottom():
                        # If
                        main_geo = self.widget.window().geometry()
                        x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                        y = main_geo.y() + 50
                else:
                    # Compute position
                    main_geo = self.widget.window().geometry()
                    x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                    y = main_geo.y() + 50

                # Show window
                self.inspector_window.setWindowTitle(f"Node Inspector - {node.name()}")
                self.inspector_window.current_node = node
                self.inspector_window.graph = self
                self.inspector_window.update_info(node)
                
                self.inspector_window.move(x, y)
                self.inspector_window.show()
                self.inspector_window.raise_()
                self.inspector_window.activateWindow()

                # Stop TODO
                if self.stl_viewer:
                    self.stl_viewer.clear_highlight()

                print(f"Inspector window displayed for node: {node.name()}")

        except Exception as e:
            print(f"Error showing inspector: {str(e)}")
            traceback.print_exc()

    def show_closed_loop_inspector(self, node, screen_pos=None):
        """
        textnodetextwindowtextdisplay
        """
        try:
            # Existing
            if hasattr(self, 'closed_loop_inspector_window') and self.closed_loop_inspector_window is not None:
                try:
                    self.closed_loop_inspector_window.close()
                    self.closed_loop_inspector_window.deleteLater()
                except Exception:
                    pass
                self.closed_loop_inspector_window = None

            # Create TODO
            self.closed_loop_inspector_window = ClosedLoopInspectorWindow(graph=self)

            # Get TODO
            inspector_size = self.closed_loop_inspector_window.sizeHint()

            if self.widget and self.widget.window():
                # Compute position
                main_geo = self.widget.window().geometry()
                x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                y = main_geo.y() + 50

                # Show window
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

        print(f"Initial position for new node: {pos}")  # Debug info

        adjusted_pos = self.find_non_overlapping_position(pos)
        print(f"Adjusted position for new node: {adjusted_pos}")  # Debug info

        # Apply TODO
        snapped_x, snapped_y = self.snap_to_grid(adjusted_pos.x(), adjusted_pos.y())
        print(f"Grid-snapped position: ({snapped_x}, {snapped_y})")  # Debug info

        new_node.set_pos(snapped_x, snapped_y)

        # Apply Settings Settings
        # Init_node_properties settings value settings
        if hasattr(new_node, 'joint_effort'):
            new_node.joint_effort = self.default_joint_effort
        if hasattr(new_node, 'joint_velocity'):
            new_node.joint_velocity = self.default_joint_velocity
        if hasattr(new_node, 'joint_damping'):
            # Settings default_damping_kv default_joint_damping settings
            new_node.joint_damping = self.default_damping_kv
        if hasattr(new_node, 'joint_stiffness'):
            # Settings default_stiffness_kp default_joint_stiffness settings
            new_node.joint_stiffness = self.default_stiffness_kp
        if hasattr(new_node, 'joint_margin'):
            new_node.joint_margin = self.default_margin
        if hasattr(new_node, 'joint_armature'):
            new_node.joint_armature = self.default_armature
        if hasattr(new_node, 'joint_frictionloss'):
            new_node.joint_frictionloss = self.default_frictionloss
        
        # Set joint_lower joint_upper Settings angle_range Settings
        if hasattr(new_node, 'joint_lower'):
            new_node.joint_lower = -self.default_angle_range
        if hasattr(new_node, 'joint_upper'):
            new_node.joint_upper = self.default_angle_range
        
        print(f"Applied Settings default values to new node: effort={self.default_joint_effort}, velocity={self.default_joint_velocity}, damping={self.default_damping_kv}, stiffness={self.default_stiffness_kp}, margin={self.default_margin}, armature={self.default_armature}, frictionloss={self.default_frictionloss}, angle_range={self.default_angle_range}")
        
        # Confirm :
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
        
        # Compute current
        row = current_node_count // items_per_row
        
        # Compute position
        position_in_row = current_node_count % items_per_row
        
        # Compute X X
        base_x = pos.x()
        
        # Compute Y
        # Previous 200
        base_y = pos.y() + (row * 200)
        
        # Compute current node X X Y
        new_x = base_x + (position_in_row * offset_x)
        new_y = base_y + (position_in_row * offset_y)
        
        new_pos = QPointF(new_x, new_y)
        
        print(f"Positioning node {current_node_count + 1}")
        print(f"Row: {row + 1}, Position in row: {position_in_row + 1}")
        print(f"Position: ({new_pos.x()}, {new_pos.y()})")
        
        # Position
        iteration = 0
        while any(self.nodes_overlap(new_pos, node.pos()) for node in all_nodes):
            new_pos += QPointF(5, 5)  # NOTE
            iteration += 1
            if iteration > 10:
                break
        
        return new_pos

    def nodes_overlap(self, pos1, pos2, threshold=5):
        pos1 = self.ensure_qpointf(pos1)
        pos2 = self.ensure_qpointf(pos2)
        overlap = (abs(pos1.x() - pos2.x()) < threshold and
                abs(pos1.y() - pos2.y()) < threshold)
        # Todo
        if overlap:
            print(f"Overlap detected: pos1={pos1}, pos2={pos2}")
        return overlap

    def ensure_qpointf(self, pos):
        if isinstance(pos, QPointF):
            return pos
        elif isinstance(pos, (tuple, list)):
            return QPointF(*pos)
        else:
            print(f"Warning: Unsupported position type: {type(pos)}")  # Debug info
            return QPointF(0, 0)  # NOTE


    def _save_node_data(self, node, project_dir):
        """nodetextーtextXMLtextsave
        
        Args:
            node: savetextnode
            project_dir: text(text)
            
        Returns:
            ET.Element: nodetextーtextXMLtext
        """
        node_elem = ET.Element("node")
        
        # Todo
        ET.SubElement(node_elem, "name").text = node.name()
        ET.SubElement(node_elem, "type").text = node.type_
        
        # Qpointf list/tuple qpointf
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
        
        # Todo
        if hasattr(node, 'stl_file') and node.stl_file:
            # Transform
            try:
                rel_path = os.path.relpath(node.stl_file, project_dir)
                ET.SubElement(node_elem, "stl_file").text = rel_path
            except (ValueError, TypeError):
                ET.SubElement(node_elem, "stl_file").text = node.stl_file
        
        # Todo
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
        
        # Inertial
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
        
        # Visual visual origin
        if hasattr(node, 'visual_origin') and node.visual_origin:
            vo_elem = ET.SubElement(node_elem, "visual_origin")
            if 'xyz' in node.visual_origin:
                xyz_str = ' '.join(str(v) for v in node.visual_origin['xyz'])
                ET.SubElement(vo_elem, "xyz").text = xyz_str
            if 'rpy' in node.visual_origin:
                rpy_str = ' '.join(str(v) for v in node.visual_origin['rpy'])
                ET.SubElement(vo_elem, "rpy").text = rpy_str
        
        # Joint
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
        
        # Save Body Angle Body Angle
        if hasattr(node, 'body_angle'):
            body_angle_str = ' '.join(str(v) for v in node.body_angle)
            ET.SubElement(node_elem, "body_angle").text = body_angle_str
        
        # Todo
        if hasattr(node, 'is_mesh_reversed'):
            ET.SubElement(node_elem, "is_mesh_reversed").text = str(node.is_mesh_reversed)
        if hasattr(node, 'mesh_original_color') and node.mesh_original_color:
            color_str = ' '.join(str(c) for c in node.mesh_original_color)
            ET.SubElement(node_elem, "mesh_original_color").text = color_str

        # Todo
        if hasattr(node, 'blank_link'):
            ET.SubElement(node_elem, "blank_link").text = str(node.blank_link)
        if hasattr(node, 'massless_decoration'):
            ET.SubElement(node_elem, "massless_decoration").text = str(node.massless_decoration)
        if hasattr(node, 'hide_mesh'):
            ET.SubElement(node_elem, "hide_mesh").text = str(node.hide_mesh)
        
        # Collider
        if hasattr(node, 'colliders') and node.colliders:
            colliders_elem = ET.SubElement(node_elem, "colliders")
            for collider in node.colliders:
                collider_elem = ET.SubElement(colliders_elem, "collider")
                
                # Type
                ET.SubElement(collider_elem, "type").text = collider.get('type') or 'primitive'
                
                # /
                ET.SubElement(collider_elem, "enabled").text = str(collider.get('enabled', True))
                
                # Todo
                if 'mesh' in collider and collider['mesh']:
                    try:
                        rel_path = os.path.relpath(collider['mesh'], project_dir)
                        ET.SubElement(collider_elem, "mesh").text = rel_path
                    except (ValueError, TypeError):
                        ET.SubElement(collider_elem, "mesh").text = collider['mesh']
                
                # Todo
                if 'mesh_scale' in collider and collider['mesh_scale']:
                    scale_str = ' '.join(str(v) for v in collider['mesh_scale'])
                    ET.SubElement(collider_elem, "mesh_scale").text = scale_str
                
                # Todo
                if 'data' in collider and collider['data']:
                    data_elem = ET.SubElement(collider_elem, "data")
                    data = collider['data']
                    
                    # Type
                    if 'type' in data:
                        ET.SubElement(data_elem, "type").text = data['type']
                    
                    # Transform
                    if 'geometry' in data and data['geometry']:
                        ET.SubElement(data_elem, "geometry").text = str(data['geometry'])
                
                # Save position collider
                position = collider.get('position', [0.0, 0.0, 0.0])
                if not position:  # NOTE
                    position = [0.0, 0.0, 0.0]
                pos_str = ' '.join(str(v) for v in position)
                ET.SubElement(collider_elem, "position").text = pos_str
                
                # Save rotate collider
                rotation = collider.get('rotation', [0.0, 0.0, 0.0])
                if not rotation:  # NOTE
                    rotation = [0.0, 0.0, 0.0]
                rot_str = ' '.join(str(v) for v in rotation)
                ET.SubElement(collider_elem, "rotation").text = rot_str

        # Points foonode if points foonode
        if hasattr(node, 'points') and isinstance(node, FooNode):
            # Points
            # Get output_ports output_count
            actual_port_count = len(node.output_ports())
            output_count = getattr(node, 'output_count', 0)
            points_count = len(node.points) if node.points else 0
            
            # Output_ports output_count
            # If output_ports output_count
            final_port_count = max(actual_port_count, output_count, points_count)
            
            # Todo
            if final_port_count != points_count:
                if actual_port_count != points_count or output_count != points_count:
                    print(f"Warning: Port count mismatch for {node.name()}: ports={actual_port_count}, output_count={output_count}, points={points_count}, using={final_port_count}")
                # Points
                if final_port_count > points_count:
                    # Add port
                    for i in range(points_count, final_port_count):
                        point_data = create_point_data(i + 1)
                        node.points.append(point_data)
                elif final_port_count < points_count:
                    # Remove TODO
                    node.points = node.points[:final_port_count]
                # Update output_count
                node.output_count = final_port_count
            
            # Save out
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
        """XMLtextnodetextーtextload
        
        Args:
            node_elem: nodetextーtextXMLtext
            
        Returns:
            textnode、textNone
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
            
            # Base_link if existing base_link check
            existing_base_link = None
            if node_name == 'base_link':
                existing_base_link = self.get_node_by_name('base_link')
            
            # Create node
            node = self.create_node(node_type, name=node_name)
            if not node:
                print(f"Warning: Could not create node of type {node_type}")
                return None
            
            # Todo
            pos_x_elem = node_elem.find("pos_x")
            pos_y_elem = node_elem.find("pos_y")
            if pos_x_elem is not None and pos_y_elem is not None:
                node.set_pos(float(pos_x_elem.text), float(pos_y_elem.text))
            
            # Base_link if check
            is_base_link_with_data = False
            if node_name == 'base_link':
                # Check
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
                
                # Create existing base_link base_link_sub
                if existing_base_link and is_base_link_with_data:
                    print(f"  Load: base_link has data and existing base_link found, creating base_link_sub")
                    base_link_pos = existing_base_link.pos()
                    # Pos list qpointf qpointf
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
                    
                    # Todo
                    current_ports = len(base_link_sub_node.output_ports())
                    # Remove connect
                    for i in range(1, current_ports + 1):
                        port_name = f'out_{i}'
                        port = base_link_sub_node.get_output(port_name)
                        if port:
                            port.clear_connections()
                    
                    # Remove TODO
                    while current_ports > 0:
                        base_link_sub_node.remove_output()
                        current_ports -= 1
                    
                    # Todo
                    base_link_sub_node.points = []
                    base_link_sub_node.cumulative_coords = []
                    base_link_sub_node.output_count = 0
                    
                    # Set base_link_sub base_link
                    # Stl
                    if stl_elem is not None and stl_elem.text:
                        stl_path = os.path.join(self.project_dir, stl_elem.text)
                        if os.path.exists(stl_path):
                            base_link_sub_node.stl_file = stl_path
                    
                    # Todo
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
                    
                    # Inertial
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
                    
                    # Visual visual origin
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
                    
                    # Joint
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
                    
                    # Body angle body angle
                    body_angle_elem = node_elem.find("body_angle")
                    if body_angle_elem is not None:
                        base_link_sub_node.body_angle = [float(v) for v in body_angle_elem.text.split()]
                    elif not hasattr(base_link_sub_node, 'body_angle'):
                        # Body_angle
                        base_link_sub_node.body_angle = [0.0, 0.0, 0.0]
                    
                    # Todo
                    is_mesh_reversed_elem = node_elem.find("is_mesh_reversed")
                    if is_mesh_reversed_elem is not None:
                        base_link_sub_node.is_mesh_reversed = is_mesh_reversed_elem.text.lower() == 'true'
                    
                    mesh_original_color_elem = node_elem.find("mesh_original_color")
                    if mesh_original_color_elem is not None:
                        base_link_sub_node.mesh_original_color = [float(c) for c in mesh_original_color_elem.text.split()]

                    # Todo
                    blank_link_elem = node_elem.find("blank_link")
                    if blank_link_elem is not None:
                        base_link_sub_node.blank_link = blank_link_elem.text.lower() == 'true'
                    
                    massless_decoration_elem = node_elem.find("massless_decoration")
                    if massless_decoration_elem is not None:
                        base_link_sub_node.massless_decoration = massless_decoration_elem.text.lower() == 'true'
                    
                    hide_mesh_elem = node_elem.find("hide_mesh")
                    if hide_mesh_elem is not None:
                        base_link_sub_node.hide_mesh = hide_mesh_elem.text.lower() == 'true'
                    
                    # Collider
                    colliders_elem = node_elem.find("colliders")
                    if colliders_elem is not None:
                        # Colliders
                        base_link_sub_node.colliders = []
                        for collider_elem in colliders_elem.findall("collider"):
                            collider = {}
                            
                            # Type
                            type_elem = collider_elem.find("type")
                            if type_elem is not None:
                                collider['type'] = type_elem.text
                            else:
                                collider['type'] = 'primitive'
                            
                            # /
                            enabled_elem = collider_elem.find("enabled")
                            if enabled_elem is not None:
                                collider['enabled'] = enabled_elem.text.lower() == 'true'
                            else:
                                collider['enabled'] = True
                            
                            # Todo
                            mesh_elem = collider_elem.find("mesh")
                            if mesh_elem is not None and mesh_elem.text:
                                mesh_path = os.path.join(self.project_dir, mesh_elem.text)
                                if os.path.exists(mesh_path):
                                    collider['mesh'] = mesh_path
                                else:
                                    collider['mesh'] = mesh_elem.text
                            else:
                                collider['mesh'] = None
                            
                            # Todo
                            scale_elem = collider_elem.find("mesh_scale")
                            if scale_elem is not None:
                                collider['mesh_scale'] = [float(v) for v in scale_elem.text.split()]
                            else:
                                collider['mesh_scale'] = [1.0, 1.0, 1.0]
                            
                            # Todo
                            data_elem = collider_elem.find("data")
                            if data_elem is not None:
                                collider['data'] = {}
                                
                                # Type
                                type_elem = data_elem.find("type")
                                if type_elem is not None:
                                    collider['data']['type'] = type_elem.text
                                
                                # Transform
                                geometry_elem = data_elem.find("geometry")
                                if geometry_elem is not None:
                                    try:
                                        collider['data']['geometry'] = eval(geometry_elem.text)
                                    except (SyntaxError, NameError):
                                        print(f"Warning: Could not parse geometry string: {geometry_elem.text}")
                                        collider['data']['geometry'] = {}
                            else:
                                collider['data'] = None
                            
                            # Position collider
                            pos_elem = collider_elem.find("position")
                            if pos_elem is not None:
                                collider['position'] = [float(v) for v in pos_elem.text.split()]
                            else:
                                collider['position'] = [0.0, 0.0, 0.0]
                            
                            # Rotate collider
                            rot_elem = collider_elem.find("rotation")
                            if rot_elem is not None:
                                collider['rotation'] = [float(v) for v in rot_elem.text.split()]
                            else:
                                collider['rotation'] = [0.0, 0.0, 0.0]
                            
                            base_link_sub_node.colliders.append(collider)

                    # Points foonode if points foonode
                    if points_elem is not None:
                        # Point
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
                                # Rpy
                                point['rpy'] = [0.0, 0.0, 0.0]
                            angle_elem = point_elem.find("angle")
                            if angle_elem is not None:
                                point['angle'] = [float(v) for v in angle_elem.text.split()]
                            else:
                                # Angle
                                point['angle'] = [0.0, 0.0, 0.0]
                            
                            # Point xyz 0 0 0 0 0 0 0.0
                            point_name = point.get('name', '')
                            point_xyz = point.get('xyz', [0.0, 0.0, 0.0])
                            is_empty_point = (
                                len(point_xyz) == 3 and
                                all(abs(v) < 1e-9 for v in point_xyz) and
                                re.match(r'^point_\d+$', point_name, re.IGNORECASE)
                            )
                            
                            if not is_empty_point:
                                base_link_sub_node.points.append(point)
                        
                        # Add port _add_output
                        num_points = len(base_link_sub_node.points)
                        for i in range(num_points):
                            base_link_sub_node.output_count += 1
                            port_name = f'out_{base_link_sub_node.output_count}'
                            # Add TODO
                            base_link_sub_node.add_output(port_name, color=(180, 80, 0))
                            # Add TODO
                            cumulative_coord = create_cumulative_coord(i)
                            base_link_sub_node.cumulative_coords.append(cumulative_coord)
                        
                        # Confirm points
                        actual_port_count = len(base_link_sub_node.output_ports())
                        if actual_port_count != num_points:
                            print(f"Warning: Port count mismatch after load for {base_link_sub_node.name()}: ports={actual_port_count}, points={num_points}")
                            # Points
                            if actual_port_count > num_points:
                                # Add port
                                for i in range(num_points, actual_port_count):
                                    point_data = create_point_data(i + 1)
                                    base_link_sub_node.points.append(point_data)
                            elif actual_port_count < num_points:
                                # Remove TODO
                                base_link_sub_node.points = base_link_sub_node.points[:actual_port_count]
                            # Update output_count
                            base_link_sub_node.output_count = actual_port_count
                    
                    # Connect base_link base_link_sub
                    try:
                        base_output_port = existing_base_link.get_output('out')
                        base_link_sub_input_port = base_link_sub_node.get_input('in')
                        if base_output_port and base_link_sub_input_port:
                            base_output_port.connect_to(base_link_sub_input_port)
                            print(f"  ✓ Connected base_link.out to base_link_sub.in")
                    except Exception as e:
                        print(f"  ✗ ERROR: Could not connect base_link to base_link_sub: {str(e)}")
                    
                    # Remove base_link_sub base_link
                    self.remove_node(node)
                    print(f"  ✓ Created base_link_sub and removed loaded base_link node")
                    return base_link_sub_node
            
            # Stl
            stl_elem = node_elem.find("stl_file")
            if stl_elem is not None and stl_elem.text:
                stl_path = os.path.join(self.project_dir, stl_elem.text)
                if os.path.exists(stl_path):
                    node.stl_file = stl_path
            
            # Todo
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
            
            # Inertial
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
            
            # Visual visual origin
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
                # Visual_origin
                node.visual_origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
            
            # Joint
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
            
            # Body angle body angle
            body_angle_elem = node_elem.find("body_angle")
            if body_angle_elem is not None:
                node.body_angle = [float(v) for v in body_angle_elem.text.split()]
            elif not hasattr(node, 'body_angle'):
                # Body_angle
                node.body_angle = [0.0, 0.0, 0.0]
            
            # Todo
            is_mesh_reversed_elem = node_elem.find("is_mesh_reversed")
            if is_mesh_reversed_elem is not None:
                node.is_mesh_reversed = is_mesh_reversed_elem.text.lower() == 'true'
            
            mesh_original_color_elem = node_elem.find("mesh_original_color")
            if mesh_original_color_elem is not None:
                node.mesh_original_color = [float(c) for c in mesh_original_color_elem.text.split()]

            # Todo
            blank_link_elem = node_elem.find("blank_link")
            if blank_link_elem is not None:
                node.blank_link = blank_link_elem.text.lower() == 'true'
            
            massless_decoration_elem = node_elem.find("massless_decoration")
            if massless_decoration_elem is not None:
                node.massless_decoration = massless_decoration_elem.text.lower() == 'true'
            
            hide_mesh_elem = node_elem.find("hide_mesh")
            if hide_mesh_elem is not None:
                node.hide_mesh = hide_mesh_elem.text.lower() == 'true'
            
            # Collider
            colliders_elem = node_elem.find("colliders")
            if colliders_elem is not None:
                # Colliders
                node.colliders = []
                for collider_elem in colliders_elem.findall("collider"):
                    collider = {}
                    
                    # Type
                    type_elem = collider_elem.find("type")
                    if type_elem is not None:
                        collider['type'] = type_elem.text
                    else:
                        collider['type'] = 'primitive'
                    
                    # /
                    enabled_elem = collider_elem.find("enabled")
                    if enabled_elem is not None:
                        collider['enabled'] = enabled_elem.text.lower() == 'true'
                    else:
                        collider['enabled'] = True
                    
                    # Todo
                    mesh_elem = collider_elem.find("mesh")
                    if mesh_elem is not None and mesh_elem.text:
                        mesh_path = os.path.join(self.project_dir, mesh_elem.text)
                        if os.path.exists(mesh_path):
                            collider['mesh'] = mesh_path
                        else:
                            collider['mesh'] = mesh_elem.text
                    else:
                        collider['mesh'] = None
                    
                    # Todo
                    scale_elem = collider_elem.find("mesh_scale")
                    if scale_elem is not None:
                        collider['mesh_scale'] = [float(v) for v in scale_elem.text.split()]
                    else:
                        collider['mesh_scale'] = [1.0, 1.0, 1.0]
                    
                    # Todo
                    data_elem = collider_elem.find("data")
                    if data_elem is not None:
                        collider['data'] = {}
                        
                        # Type
                        type_elem = data_elem.find("type")
                        if type_elem is not None:
                            collider['data']['type'] = type_elem.text
                        
                        # Transform
                        geometry_elem = data_elem.find("geometry")
                        if geometry_elem is not None:
                            try:
                                collider['data']['geometry'] = eval(geometry_elem.text)
                            except (SyntaxError, NameError):
                                print(f"Warning: Could not parse geometry string: {geometry_elem.text}")
                                collider['data']['geometry'] = {}
                    else:
                        collider['data'] = None
                    
                    # Position collider
                    pos_elem = collider_elem.find("position")
                    if pos_elem is not None:
                        collider['position'] = [float(v) for v in pos_elem.text.split()]
                    else:
                        collider['position'] = [0.0, 0.0, 0.0]
                    
                    # Rotate collider
                    rot_elem = collider_elem.find("rotation")
                    if rot_elem is not None:
                        collider['rotation'] = [float(v) for v in rot_elem.text.split()]
                    else:
                        collider['rotation'] = [0.0, 0.0, 0.0]
                    
                    node.colliders.append(collider)

            # Points foonode if points foonode
            points_elem = node_elem.find("points")
            if points_elem is not None and isinstance(node, FooNode):
                points = points_elem.findall("point")
                
                # Remove existing port
                current_ports = len(node.output_ports())
                # Remove connect
                for i in range(1, current_ports + 1):
                    port_name = f'out_{i}'
                    port = node.get_output(port_name)
                    if port:
                        port.clear_connections()
                
                # Remove TODO
                while current_ports > 0:
                    node.remove_output()
                    current_ports -= 1
                
                # Todo
                node.points = []
                node.cumulative_coords = []
                node.output_count = 0
                
                # Point
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
                        # Rpy
                        point['rpy'] = [0.0, 0.0, 0.0]
                    angle_elem = point_elem.find("angle")
                    if angle_elem is not None:
                        point['angle'] = [float(v) for v in angle_elem.text.split()]
                    else:
                        # Angle
                        point['angle'] = [0.0, 0.0, 0.0]
                    
                    # Point xyz 0 0 0 0 0 0 0.0
                    # Todo
                    point_name = point.get('name', '')
                    point_xyz = point.get('xyz', [0.0, 0.0, 0.0])
                    is_empty_point = (
                        len(point_xyz) == 3 and
                        all(abs(v) < 1e-9 for v in point_xyz) and
                        re.match(r'^point_\d+$', point_name, re.IGNORECASE)
                    )
                    
                    # Confirm out_1 out_2
                    # 1 xml
                    port_name = f'out_{point_index}'
                    is_connected = connected_ports and port_name in connected_ports
                    
                    # Todo
                    if is_connected or not is_empty_point:
                        node.points.append(point)
                    else:
                        print(f"Filtered out empty point '{point_name}' from node '{node.name()}' (not connected, port: {port_name})")
                
                num_points = len(node.points)
                
                # Add port _add_output
                for i in range(num_points):
                    node.output_count += 1
                    port_name = f'out_{node.output_count}'
                    # Add TODO
                    node.add_output(port_name, color=(180, 80, 0))
                    # Add TODO
                    cumulative_coord = create_cumulative_coord(i)
                    node.cumulative_coords.append(cumulative_coord)
                
                # Confirm points
                actual_port_count = len(node.output_ports())
                if actual_port_count != num_points:
                    print(f"Warning: Port count mismatch after load for {node.name()}: ports={actual_port_count}, points={num_points}")
                    # Points
                    if actual_port_count > num_points:
                        # Add port
                        for i in range(num_points, actual_port_count):
                            point_data = create_point_data(i + 1)
                            node.points.append(point_data)
                    elif actual_port_count < num_points:
                        # Remove TODO
                        node.points = node.points[:actual_port_count]
                    # Update output_count
                    node.output_count = actual_port_count
            
            # Base_link if existing base_link
            if node_name == 'base_link' and existing_base_link:
                print(f"  Load: base_link data found but existing base_link exists, skipping data assignment")
                # Remove existing base_link
                self.remove_node(node)
                return existing_base_link
            
            return node
            
        except Exception as e:
            print(f"Error loading node data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


    def save_project(self, file_path=None):
        """textsave(text)"""
        print("\n=== Starting Project Save ===")
        try:
            # Stl
            stl_viewer_state = None
            if hasattr(self, 'stl_viewer'):
                print("Backing up STL viewer state...")
                stl_viewer_state = {
                    'actors': dict(self.stl_viewer.stl_actors),
                    'transforms': dict(self.stl_viewer.transforms)
                }
                # Stl
                self.stl_viewer.stl_actors.clear()
                self.stl_viewer.transforms.clear()

            # Get TODO
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

            # Create XML
            print("Creating XML structure...")
            root = ET.Element("project")
            
            # Save TODO
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
            
            # Save base_link_height MJCF
            if hasattr(self, 'base_link_height'):
                ET.SubElement(root, "base_link_height").text = str(self.base_link_height)
                print(f"Saving base_link_height: {self.base_link_height}")

            # Save TODO
            print("\nSaving nodes...")
            nodes_elem = ET.SubElement(root, "nodes")
            total_nodes = len(self.all_nodes())
            
            for i, node in enumerate(self.all_nodes(), 1):
                print(f"Processing node {i}/{total_nodes}: {node.name()}")
                # Remove STL
                stl_viewer_backup = node.stl_viewer if hasattr(node, 'stl_viewer') else None
                if hasattr(node, 'stl_viewer'):
                    delattr(node, 'stl_viewer')
                
                node_elem = self._save_node_data(node, self.project_dir)
                nodes_elem.append(node_elem)
                
                # Stl
                if stl_viewer_backup is not None:
                    node.stl_viewer = stl_viewer_backup

            # Save TODO
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

            # Save TODO
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

            # Save TODO
            print("\nSaving highlight color...")
            highlight_color_elem = ET.SubElement(root, "highlight_color")
            highlight_color_elem.text = self.highlight_color
            print(f"Saved highlight color: {self.highlight_color}")

            # Save TODO
            print("\nSaving collision color...")
            collision_color_elem = ET.SubElement(root, "collision_color")
            collision_color_elem.text = " ".join(str(v) for v in self.collision_color)
            print(f"Saved collision color: {self.collision_color}")

            # Save TODO
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

            # Save file
            print("\nWriting to file...")
            tree = ET.ElementTree(root)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)

            # Stl
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
            
            # Stl
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
        """STLtextーtext"""
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
        """meshestext"""
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
        """textload(textーtext)"""
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
            
            # Xml
            print("Parsing XML file...")
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Load TODO
            robot_name_elem = root.find("robot_name")
            if robot_name_elem is not None and robot_name_elem.text:
                self.robot_name = robot_name_elem.text
                # Update UI
                if hasattr(self, 'name_input') and self.name_input:
                    self.name_input.setText(self.robot_name)
                print(f"Loaded robot name: {self.robot_name}")
            else:
                print("No robot name found in project file")
            
            # Load base_link_height
            base_link_height_elem = root.find("base_link_height")
            if base_link_height_elem is not None and base_link_height_elem.text:
                self.base_link_height = float(base_link_height_elem.text)
                print(f"Loaded base_link_height: {self.base_link_height}")
            else:
                # Todo
                if not hasattr(self, 'base_link_height'):
                    self.base_link_height = self.default_base_link_height
                print(f"Using default base_link_height: {self.base_link_height}")

            # Existing node base_link
            print("Clearing existing nodes (except base_link)...")
            existing_base_link = self.get_node_by_name('base_link')
            self.clear_graph()
            # Base_link base_link
            if existing_base_link:
                print("Recreating default base_link after clear_graph...")
                default_base_link = self.create_node(
                    'insilico.nodes.BaseLinkNode',
                    name='base_link',
                    pos=QtCore.QPointF(50, 0)
                )
                # Set TODO
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

            # Todo
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

            # Todo
            print("\nRestoring highlight color...")
            highlight_color_elem = root.find("highlight_color")
            if highlight_color_elem is not None and highlight_color_elem.text:
                self.highlight_color = highlight_color_elem.text
                print(f"Restored highlight color: {self.highlight_color}")
            else:
                print("No highlight color found in project file, using default")

            # Todo
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

            # Todo
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

                    # Error :
                    friction_elem = settings_elem.find("friction")
                    if friction_elem is not None and friction_elem.text:
                        pass  # NOTE
                    actuation_lag_elem = settings_elem.find("actuation_lag")
                    if actuation_lag_elem is not None and actuation_lag_elem.text:
                        pass  # NOTE

                    print(f"Restored default joint settings: effort={self.default_joint_effort}, "
                          f"velocity={self.default_joint_velocity}, damping={self.default_joint_damping}, "
                          f"stiffness={self.default_joint_stiffness}, margin={self.default_margin}, "
                          f"armature={self.default_armature}, frictionloss={self.default_frictionloss}")
                except (ValueError, TypeError) as e:
                    print(f"Error parsing default joint settings, using defaults: {e}")
            else:
                print("No default joint settings found in project file, using defaults")

            # Meshes
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

            # Todo
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
            
            # Node
            print("\nRestoring nodes...")
            nodes_elem = root.find("nodes")
            total_nodes = len(nodes_elem.findall("node")) if nodes_elem is not None else 0
            
            # Stl
            nodes_with_stl = 0
            for node_elem in nodes_elem.findall("node") if nodes_elem is not None else []:
                stl_elem = node_elem.find("stl_file")
                if stl_elem is not None and stl_elem.text:
                    nodes_with_stl += 1
            
            # Process + stl
            total_operations = total_nodes + nodes_with_stl
            print(f"Total operations: {total_nodes} node loads + {nodes_with_stl} STL loads = {total_operations}")
            
            # Set TODO
            if total_operations > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                self.stl_viewer.show_progress(True)
                self.stl_viewer.progress_bar.setValue(100)  # Start from 100%
                QtWidgets.QApplication.processEvents()
            
            nodes_dict = {}
            processed_operations = 0
            
            # Load node
            for i, node_elem in enumerate(nodes_elem.findall("node"), 1):
                node_name_elem = node_elem.find("name")
                node_name = node_name_elem.text if node_name_elem is not None else None
                node_connections = connections_info.get(node_name, set()) if node_name else set()
                node = self._load_node_data(node_elem, connected_ports=node_connections)
                if node:
                    nodes_dict[node.name()] = node
                
                processed_operations += 1
                # Update 1
                if total_operations > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                    # Compute 100% 100
                    remaining_percent = 100 - int((processed_operations / total_operations) * 100)
                    self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                    QtWidgets.QApplication.processEvents()

            # Connect TODO
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

            # Update position view
            print("\nRecalculating positions...")
            self.recalculate_all_positions()

            print("Updating 3D view...")
            if self.stl_viewer:
                # View
                self.stl_viewer.reset_view_to_fit()

                # Apply Hide run Hide Mesh recalculate_all_positions
                print("\nApplying hide_mesh states after position recalculation...")
                for node in nodes_dict.values():
                    if hasattr(node, 'hide_mesh') and node.hide_mesh:
                        if node in self.stl_viewer.stl_actors:
                            actor = self.stl_viewer.stl_actors[node]
                            actor.SetVisibility(False)
                            print(f"Applied hide_mesh: {node.name()} - mesh hidden")

                # Update 3D
                self.stl_viewer.render_to_image()

                # Update show OFF Collider OFF
                print("\nApplying collider display states...")
                # On off on off
                # Show ON OFF ON OFF
                if self.stl_viewer.collider_display_enabled:
                    # Update ON if ON
                    self.stl_viewer.refresh_collider_display()
                    print("Collider display updated (already enabled)")
                else:
                    # Hide OFF if OFF
                    self.stl_viewer.hide_all_colliders()
                    print("Collider display remains OFF (user must enable manually)")
                
                # Apply note color stl_actors node : STL STL
                
                # Apply node STL 3D
                print("\nApplying colors to 3D view after project load...")
                self._apply_colors_to_all_nodes()

            # Hide STL
            # Hide TODO

            print(f"\nProject successfully loaded from: {file_path}")
            return True

        except Exception as e:
            error_msg = f"Error loading project: {str(e)}"
            print(f"\nERROR: {error_msg}")
            print("Traceback:")
            traceback.print_exc()
            
            # Hide TODO
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
        """text2textーtext"""
        if from_port and to_port:
            try:
                # Todo
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
        """nodetextcumulativetransformstext(textーtextconsider)"""
        if isinstance(node, BaseLinkNode):
            return [0, 0, 0]  # base_link

        # Get TODO
        input_port = node.input_ports()[0]  # NOTE
        if not input_port.connected_ports():
            return [0, 0, 0]  # NOTE

        parent_port = input_port.connected_ports()[0]
        parent_node = parent_port.node()
        
        # Compute TODO
        parent_coords = self.calculate_cumulative_coordinates(parent_node)
        
        # Get TODO
        port_name = parent_port.name()
        if '_' in port_name:
            port_index = int(port_name.split('_')[1]) - 1
        else:
            port_index = 0
            
        # Get TODO
        if 0 <= port_index < len(parent_node.points):
            point_xyz = parent_node.points[port_index]['xyz']
            
            # Compute TODO
            return [
                parent_coords[0] + point_xyz[0],
                parent_coords[1] + point_xyz[1],
                parent_coords[2] + point_xyz[2]
            ]
        return parent_coords

    def _find_mesh_file(self, folder_path, base_name):
        """textーtextmeshtext

        Args:
            folder_path: text
            base_name: textーtext(text)

        Returns:
            str: textmeshtext、textNone
        """
        # Dae > obj > stl :
        extensions = ['.dae', '.obj', '.stl']

        for ext in extensions:
            mesh_file = base_name + ext
            mesh_path = os.path.join(folder_path, mesh_file)
            if os.path.exists(mesh_path):
                return mesh_path

        return None

    def _find_collider_file(self, folder_path, base_name):
        """textーtextーtext

        Args:
            folder_path: text
            base_name: textーtext(text)

        Returns:
            tuple: (collider_path, collider_type)
                   collider_pathtext、textNone
                   collider_typetext'xml'(text)text'mesh'、textNone
        """
        # 1 1. xml
        collider_xml = base_name + '_collider.xml'
        collider_xml_path = os.path.join(folder_path, collider_xml)
        if os.path.exists(collider_xml_path):
            return (collider_xml_path, 'xml')

        # 2 dae > obj > stl 2. :
        mesh_extensions = ['.dae', '.obj', '.stl']
        for ext in mesh_extensions:
            collider_mesh = base_name + '_collider' + ext
            collider_mesh_path = os.path.join(folder_path, collider_mesh)
            if os.path.exists(collider_mesh_path):
                return (collider_mesh_path, 'mesh')

        return (None, None)

    def import_xmls_from_folder(self):
        """textalltextXMLtext"""
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

        # Todo
        try:
            # Get TODO
            robot_name = os.path.basename(folder_path)

            # Remove _description
            if robot_name.endswith('_description'):
                robot_name = robot_name[:-12]
                print(f"Removed '_description' suffix from robot name")

            # Update TODO
            self.robot_name = robot_name
            if hasattr(self, 'name_input') and self.name_input:
                self.name_input.setText(robot_name)
            print(f"Set robot name to: {robot_name}")
        except Exception as e:
            print(f"Error extracting robot name: {str(e)}")
        
        # Search *_collider xml _collider.xml
        all_xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
        xml_files = [f for f in all_xml_files if not f.endswith('_collider.xml')]

        if not xml_files:
            print("No valid XML files found in the selected folder")
            print("(Note: *_collider.xml files are used as collider definitions and don't create nodes)")
            return

        # File c_* l_* r_* : c_ l_ r_
        def get_file_sort_priority(filename):
            """
            text
            c_* = 0, l_* = 1, r_* = 2, text = 3
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

        # Set TODO
        total_files = len(xml_files)
        if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
            self.stl_viewer.show_progress(True)
            self.stl_viewer.progress_bar.setValue(100)  # Start from 100%
            QtWidgets.QApplication.processEvents()

        # Var
        current_group = None
        node_y_position = 0
        node_spacing = 5  # NOTE
        group_spacing = 30  # NOTE

        for file_index, xml_file in enumerate(xml_files):
            try:
                xml_path = os.path.join(folder_path, xml_file)

                # Current file
                file_group = get_file_sort_priority(xml_file)[0]

                # Add TODO
                if current_group is not None and file_group != current_group:
                    node_y_position += group_spacing
                    print(f"\n{'─'*60}")
                    group_names = {0: 'Center (c_*)', 1: 'Left (l_*)', 2: 'Right (r_*)', 3: 'Others'}
                    print(f"▼ Starting new group: {group_names.get(file_group, 'Others')}")
                    print(f"{'─'*60}")

                current_group = file_group

                print(f"\n{'='*60}")
                print(f"Processing: {xml_file}")

                # Set create Y
                new_node = self.create_node(
                    'insilico.nodes.FooNode',
                    name=f'Node_{len(self.all_nodes())}',
                    pos=QtCore.QPointF(0, node_y_position)
                )

                # Update next node Y
                node_y_position += node_spacing
                
                # Xml
                tree = ET.parse(xml_path)
                root = tree.getroot()

                if root.tag != 'urdf_part':
                    print(f"Warning: Invalid XML format in {xml_file}")
                    continue

                # Process
                link_elem = root.find('link')
                if link_elem is not None:
                    # Set TODO
                    link_name = link_elem.get('name')
                    if link_name:
                        new_node.set_name(link_name)
                    else:
                        link_name = new_node.name()  # NOTE

                    # Process
                    inertial_elem = link_elem.find('inertial')
                    if inertial_elem is not None:
                        # Set TODO
                        volume_elem = inertial_elem.find('volume')
                        if volume_elem is not None:
                            new_node.volume_value = float(volume_elem.get('value', '0.0'))

                        # Set TODO
                        mass_elem = inertial_elem.find('mass')
                        if mass_elem is not None:
                            new_node.mass_value = float(mass_elem.get('value', '0.0'))

                        # Set Inertial Origin Inertial Origin
                        origin_elem = inertial_elem.find('origin')
                        if origin_elem is not None:
                            origin_xyz = origin_elem.get('xyz', '0 0 0').split()
                            origin_rpy = origin_elem.get('rpy', '0 0 0').split()
                            new_node.inertial_origin = {
                                'xyz': [float(x) for x in origin_xyz],
                                'rpy': [float(x) for x in origin_rpy]
                            }

                        # Set TODO
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
                            
                            # Inertial : xml
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

                # Set Center of Mass Center Mass link
                center_of_mass_elem = link_elem.find('center_of_mass')
                if center_of_mass_elem is not None and center_of_mass_elem.text:
                    com_xyz = center_of_mass_elem.text.strip().split()
                    if len(com_xyz) == 3:
                        new_node.center_of_mass = [float(x) for x in com_xyz]
                        print(f"Set center of mass: {new_node.center_of_mass}")

                # Process
                material_elem = root.find('.//material/color')
                if material_elem is not None:
                    rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                    rgba_values = [float(x) for x in rgba[:4]]  # RGBA
                    new_node.node_color = rgba_values
                else:
                    new_node.node_color = DEFAULT_COLOR_WHITE.copy()
                    print("Using default color: white")

                # Joint limits process
                joint_elem = root.find('joint')
                if joint_elem is not None:
                    # Confirm TODO
                    joint_type = joint_elem.get('type', '')
                    if joint_type == 'fixed':
                        new_node.rotation_axis = 3  # Fixed
                    else:
                        # Process
                        axis_elem = joint_elem.find('axis')
                        if axis_elem is not None:
                            axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                            axis_values = [float(x) for x in axis_xyz]
                            if axis_values[2] == 1:      # Z-axis
                                new_node.rotation_axis = 2
                            elif axis_values[1] == 1:    # Y-axis
                                new_node.rotation_axis = 1
                            else:                        # X（）
                                new_node.rotation_axis = 0
                            print(f"Set rotation axis: {new_node.rotation_axis} from xyz: {axis_xyz}")

                    # Joint limits process joint
                    limit_elem = joint_elem.find('limit')
                    if limit_elem is not None:
                        # Xml xml radian
                        lower_rad = float(limit_elem.get('lower', -3.14159))
                        upper_rad = float(limit_elem.get('upper', 3.14159))
                        effort = float(limit_elem.get('effort', 10.0))
                        velocity = float(limit_elem.get('velocity', 3.0))
                        damping = float(limit_elem.get('damping', DEFAULT_DAMPING_KV))
                        stiffness = float(limit_elem.get('stiffness', DEFAULT_STIFFNESS_KP))
                        margin = float(limit_elem.get('margin', DEFAULT_MARGIN))
                        armature = float(limit_elem.get('armature', DEFAULT_ARMATURE))
                        frictionloss = float(limit_elem.get('frictionloss', DEFAULT_FRICTIONLOSS))

                        # Save node Radian
                        new_node.joint_lower = lower_rad
                        new_node.joint_upper = upper_rad
                        new_node.joint_effort = effort
                        new_node.joint_velocity = velocity
                        new_node.joint_damping = damping
                        new_node.joint_stiffness = stiffness
                        new_node.joint_margin = margin
                        new_node.joint_armature = armature
                        new_node.joint_frictionloss = frictionloss

                    # Joint dynamics process joint
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

                # Load massless_decoration hide_mesh
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

                # Process
                point_elements = root.findall('point')
                num_points = len(point_elements)

                # Foonode if foonode
                if isinstance(new_node, FooNode):
                    # Current
                    current_ports = len(new_node.output_ports())

                    # Connect port port
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

                    # Update TODO
                    new_node.points = []
                    for point_elem in point_elements:
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
                            new_node.points.append({
                                'name': point_name,
                                'type': point_type,
                                'xyz': xyz_values,
                                'angle': angle_values
                            })

                    # Update TODO
                    new_node.cumulative_coords = []
                    for i in range(len(new_node.points)):
                        new_node.cumulative_coords.append(create_cumulative_coord(i))

                    # Update output_count
                    new_node.output_count = len(new_node.points)

                # Get TODO
                base_name = xml_file[:-4]

                # Process dae > obj > stl :
                mesh_path = self._find_mesh_file(folder_path, base_name)
                if mesh_path:
                    mesh_ext = os.path.splitext(mesh_path)[1]
                    print(f"Loading mesh file: {os.path.basename(mesh_path)} {mesh_ext}")
                    new_node.stl_file = mesh_path
                    if self.stl_viewer:
                        # Hide TODO
                        self.stl_viewer.load_stl_for_node(new_node, show_progress=False)
                        # Apply model color
                        if hasattr(new_node, 'node_color'):
                            self.stl_viewer.apply_color_to_node(new_node)
                else:
                    print(f"Warning: No mesh file found for {base_name}")

                # Load Collider XML
                collider_elem = root.find('collider')
                if collider_elem is not None:
                    collider_type = collider_elem.get('type')
                    collider_file = collider_elem.get('file')

                    if collider_type == 'primitive' and collider_file:
                        # If
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
                        # If
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

                # Collision mesh process collision
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

                # Xml collider
                if not collider_elem and not collision_mesh_elem:
                    collider_path, collider_type = self._find_collider_file(folder_path, base_name)

                    if collider_path and collider_type == 'xml':
                        # Xml
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
                        # Todo
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
                        # Todo
                        if mesh_path:
                            print(f"  → No dedicated collider found, will use visual mesh when enabled")
                            # Update colliders list update
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

                # _dec _dec
                import re
                if re.search(r'_dec\d*$', base_name):
                    new_node.massless_decoration = True
                    # Set enabled False colliders False
                    if hasattr(new_node, 'colliders') and new_node.colliders:
                        for collider in new_node.colliders:
                            collider['enabled'] = False
                    new_node.rotation_axis = 3  # Fixed
                    print(f"  → Auto-configured: Massless Decoration=ON, Collider=OFF, Rotation Axis=Fixed (filename ends with '_dec')")

                print(f"✓ Successfully imported: {xml_file}")
                
                # Confirm node :
                final_link_name = new_node.name()  # NOTE
                if 'arm_lower' in final_link_name.lower():
                    print(f"\n[XML_IMPORT_COMPLETE] link_name={final_link_name}, source_xml_path={xml_path}")
                    if hasattr(new_node, 'inertia') and new_node.inertia:
                        print(f"  Final node.inertia: ixx={new_node.inertia.get('ixx', 0):.9e}, ixy={new_node.inertia.get('ixy', 0):.9e}, ixz={new_node.inertia.get('ixz', 0):.9e}")
                        print(f"                      iyy={new_node.inertia.get('iyy', 0):.9e}, iyz={new_node.inertia.get('iyz', 0):.9e}, izz={new_node.inertia.get('izz', 0):.9e}")
                    else:
                        print(f"  WARNING: node.inertia is not set!")

                # Update 1
                if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                    # Compute TODO
                    processed_files = file_index + 1
                    # Compute 100% 100
                    remaining_percent = 100 - int((processed_files / total_files) * 100)
                    self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                    QtWidgets.QApplication.processEvents()

            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                traceback.print_exc()
                # Update error
                if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                    processed_files = file_index + 1
                    remaining_percent = 100 - int((processed_files / total_files) * 100)
                    self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                    QtWidgets.QApplication.processEvents()
                continue

        # Node color
        self.update_all_node_colors()

        # Hide TODO
        if total_files > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
            self.stl_viewer.progress_bar.setValue(0)
            QtWidgets.QApplication.processEvents()
            from PySide6.QtCore import QTimer
            QTimer.singleShot(200, lambda: self.stl_viewer.show_progress(False))

        print("\nImport process completed")

    def recalculate_all_positions(self):
        """alltextnodetext"""
        print("Starting position recalculation for all nodes...")
        
        try:
            # Base_link
            base_node = None
            for node in self.all_nodes():
                if isinstance(node, BaseLinkNode):
                    base_node = node
                    break
            
            if not base_node:
                print("Error: Base link node not found")
                return
            
            # Todo
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
            
            # Set TODO
            if total_nodes > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                # Show TODO
                if not self.stl_viewer.progress_bar.isVisible():
                    self.stl_viewer.show_progress(True)
                self.stl_viewer.progress_bar.setValue(100)  # Start from 100%
                QtWidgets.QApplication.processEvents()
            
            # Update position
            visited_nodes = set()
            processed_count = [0]  # NOTE
            
            print(f"Starting from base node: {base_node.name()}")
            self._recalculate_node_positions(base_node, [0, 0, 0], visited_nodes, None, total_nodes, processed_count)

            # Apply TODO
            print("\n=== Enforcing Closed-Loop Constraints ===")
            self._enforce_closed_loop_constraints()
            print("=== Closed-Loop Constraints Enforced ===\n")

            # Update STL
            if hasattr(self, 'stl_viewer'):
                self.stl_viewer.render_to_image()
            
            # Hide process
            if total_nodes > 0 and hasattr(self, 'stl_viewer') and self.stl_viewer:
                self.stl_viewer.progress_bar.setValue(0)
                QtWidgets.QApplication.processEvents()
                from PySide6.QtCore import QTimer
                QTimer.singleShot(200, lambda: self.stl_viewer.show_progress(False))

            print("Position recalculation completed")

        except Exception as e:
            print(f"Error during position recalculation: {str(e)}")
            traceback.print_exc()
            # Hide TODO
            if hasattr(self, 'stl_viewer') and self.stl_viewer:
                self.stl_viewer.show_progress(False)

    def _recalculate_node_positions(self, node, parent_coords, visited, parent_transform=None, total_nodes=0, processed_count=None):
        """textnodetext"""
        if node in visited:
            return
        visited.add(node)
        
        # Update TODO
        if processed_count is not None and total_nodes > 0:
            processed_count[0] += 1
            if hasattr(self, 'stl_viewer') and self.stl_viewer:
                remaining_percent = 100 - int((processed_count[0] / total_nodes) * 100)
                self.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                QtWidgets.QApplication.processEvents()

        print(f"\nProcessing node: {node.name()}")
        print(f"Parent coordinates: {parent_coords}")

        # Confirm TODO
        if isinstance(node, ClosedLoopJointNode):
            print(f"  ⚠ DEBUG: This is a ClosedLoopJointNode")
            print(f"  ⚠ DEBUG: Output ports: {[p.name() for p in node.output_ports()]}")
            for port in node.output_ports():
                connected = port.connected_ports()
                if connected:
                    print(f"  ⚠ DEBUG: Port {port.name()} is connected to: {[f'{p.node().name()}.{p.name()}' for p in connected]}")
                else:
                    print(f"  ⚠ DEBUG: Port {port.name()} has no connections")

            # Process skip
            print(f"  ⚠ DEBUG: Skipping child node processing for closed-loop node")
            return

        # Current node 3d
        if hasattr(self, 'stl_viewer'):
            if node not in self.stl_viewer.stl_actors or node not in self.stl_viewer.transforms:
                # Todo
                if hasattr(node, 'stl_file') and node.stl_file:
                    print(f"  ℹ Node {node.name()} not loaded yet, loading now...")
                    self.stl_viewer.load_stl_for_node(node, show_progress=False)
                # Mesh if transform
                elif node not in self.stl_viewer.transforms:
                    import vtk
                    self.stl_viewer.transforms[node] = vtk.vtkTransform()
                    self.stl_viewer.transforms[node].Identity()
                    print(f"  ℹ Created transform for meshless node {node.name()}")

        # Hide current node Hide Mesh if Mesh 3D
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            if hasattr(self, 'stl_viewer') and node in self.stl_viewer.stl_actors:
                actor = self.stl_viewer.stl_actors[node]
                actor.SetVisibility(False)
                print(f"Applied hide_mesh: {node.name()} - mesh hidden in 3D view")

        try:
            # Node port output :
            if node.name() == 'base_link_sub':
                print(f"\n*** DEBUG base_link_sub ***")
                print(f"  Output ports: {[p.name() for p in node.output_ports()]}")
                print(f"  Points count: {len(node.points) if hasattr(node, 'points') else 0}")
                if hasattr(node, 'points'):
                    for i, pt in enumerate(node.points):
                        print(f"  points[{i}]: name={pt.get('name')}, xyz={pt.get('xyz')}")
                print(f"*** END DEBUG ***\n")

            # Process
            for port_idx, output_port in enumerate(node.output_ports()):
                for connected_port in output_port.connected_ports():
                    child_node = connected_port.node()

                    # Confirm TODO
                    # Compute out_1 0 out_2 1 etc - etc.
                    port_name = output_port.name() if hasattr(output_port, 'name') else 'unknown'
                    point_index = port_idx  # enum
                    if port_name.startswith('out_'):
                        try:
                            port_num = int(port_name.split('_')[1])
                            point_index = port_num - 1  # out_1 -> 0, out_2 -> 1, etc.
                        except (ValueError, IndexError):
                            pass
                    elif port_name == 'out':
                        point_index = 0  # BaseLinkNode'out'0

                    print(f"\n=== Processing connection: {node.name()}[{port_name}] (enum_idx={port_idx}, point_index={point_index}) -> {child_node.name()} ===")
                    print(f"  has_points: {hasattr(node, 'points')}, points_count: {len(node.points) if hasattr(node, 'points') else 0}")
                    if hasattr(node, 'points') and point_index < len(node.points):
                        point_data = node.points[point_index]
                        print(f"  point_data: {point_data}")
                        point_xyz = point_data.get('xyz', [0, 0, 0])
                        point_rpy = point_data.get('rpy', [0, 0, 0])
                        point_angle = point_data.get('angle', [0.0, 0.0, 0.0])  # radians

                        # Compute -
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

                        # Create TODO
                        import vtk
                        import math

                        # Create joint parent transform
                        joint_transform = vtk.vtkTransform()
                        joint_transform.Identity()

                        # Urdf/sdf r rpy xyz urdf/sdf : r
                        #                          [  0      1  ]
                        # Transform p r * p + xyz next : r
                        # Vtk post-multiply vtk
                        # Translate rotate translate rotate

                        # Add 1 1.
                        joint_transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])

                        # Add 2 RPY rotate 2. RPY
                        if point_rpy and len(point_rpy) == 3:
                            roll_deg = math.degrees(point_rpy[0])
                            pitch_deg = math.degrees(point_rpy[1])
                            yaw_deg = math.degrees(point_rpy[2])
                            # Rz yaw * ry pitch * rx roll : rz ry rx
                            # Add VTK Z Y X VTK Z Y X
                            joint_transform.RotateZ(yaw_deg)
                            joint_transform.RotateY(pitch_deg)
                            joint_transform.RotateX(roll_deg)
                            print(f"  Applied RPY rotation: Roll={roll_deg}, Pitch={pitch_deg}, Yaw={yaw_deg} degrees")

                        # Add 3 point_angle joint_transform parent body orientation 3.
                        # Mjcf if point_angle body quat orientation mjcf
                        if point_angle and any(a != 0.0 for a in point_angle):
                            point_angle_deg = [math.degrees(a) for a in point_angle]
                            joint_transform.RotateZ(point_angle_deg[2])  # Z-axis rotation
                            joint_transform.RotateY(point_angle_deg[1])  # Y-axis rotation
                            joint_transform.RotateX(point_angle_deg[0])  # X-axis rotation
                            print(f"  Applied point_angle to joint_transform: X={point_angle_deg[0]}, Y={point_angle_deg[1]}, Z={point_angle_deg[2]} degrees")

                        # Parent transform
                        child_transform = vtk.vtkTransform()
                        if parent_transform is not None:
                            child_transform.Concatenate(parent_transform)
                        child_transform.Concatenate(joint_transform)

                        # Body_angle radian degree vtk
                        child_body_angle = getattr(child_node, 'body_angle', [0.0, 0.0, 0.0])
                        print(f"  Child body_angle (rad): {child_body_angle}")
                        print(f"  Child body_angle zero?: {not any(a != 0.0 for a in child_body_angle)}")
                        if child_body_angle and any(a != 0.0 for a in child_body_angle):
                            child_body_angle_deg = [math.degrees(a) for a in child_body_angle]
                            child_transform.RotateZ(child_body_angle_deg[2])  # Z-axis rotation
                            child_transform.RotateY(child_body_angle_deg[1])  # Y-axis rotation
                            child_transform.RotateX(child_body_angle_deg[0])  # X-axis rotation
                            print(f"  ✓ Applied child body_angle: X={child_body_angle_deg[0]}, Y={child_body_angle_deg[1]}, Z={child_body_angle_deg[2]} degrees")
                        print("=== End Transform Debug ===\n")

                        # Apply TODO
                        # Node 3d
                        if child_node not in self.stl_viewer.stl_actors or child_node not in self.stl_viewer.transforms:
                            # Todo
                            if hasattr(child_node, 'stl_file') and child_node.stl_file:
                                print(f"  ℹ Node {child_node.name()} not loaded yet, loading now...")
                                self.stl_viewer.load_stl_for_node(child_node, show_progress=False)
                        
                        # Apply TODO
                        if child_node in self.stl_viewer.stl_actors and child_node in self.stl_viewer.transforms:
                            # Deepcopy deepcopy
                            new_transform = vtk.vtkTransform()
                            new_transform.DeepCopy(child_transform)
                            self.stl_viewer.transforms[child_node] = new_transform
                            self.stl_viewer.stl_actors[child_node].SetUserTransform(new_transform)
                            print(f"  ✓ Applied transform to 3D actor for {child_node.name()}")
                            # Update transform
                            self.stl_viewer.update_collider_transform(child_node)
                        else:
                            # Base_link warning
                            if hasattr(child_node, 'stl_file') and child_node.stl_file:
                                print(f"  ✗ WARNING: Cannot apply transform to {child_node.name()} even after loading")
                            # Baselinknode mesh transform baselinknode
                            # Todo
                            new_transform = vtk.vtkTransform()
                            new_transform.DeepCopy(child_transform)
                            self.stl_viewer.transforms[child_node] = new_transform

                        # Hide Hide Mesh if Mesh 3D
                        if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                            if child_node in self.stl_viewer.stl_actors:
                                actor = self.stl_viewer.stl_actors[child_node]
                                actor.SetVisibility(False)
                                print(f"Applied hide_mesh: {child_node.name()} - mesh hidden in 3D view")

                        # Update TODO
                        if hasattr(child_node, 'cumulative_coords'):
                            for coord in child_node.cumulative_coords:
                                # Process
                                if isinstance(coord, dict):
                                    coord['xyz'] = new_position.copy()
                                elif isinstance(coord, list):
                                    # If transform
                                    coord_idx = child_node.cumulative_coords.index(coord)
                                    child_node.cumulative_coords[coord_idx] = {
                                        'point_index': coord_idx,
                                        'xyz': new_position.copy()
                                    }

                        # Process
                        self._recalculate_node_positions(child_node, new_position, visited, child_transform, total_nodes, processed_count)
                    else:
                        print(f"Warning: No point data found for port {port_name} (point_index={point_index}) in node {node.name()}")

        except Exception as e:
            print(f"Error processing node {node.name()}: {str(e)}")
            traceback.print_exc()

    def _enforce_closed_loop_constraints(self):
        """textーtext、textーtextjointtext"""
        if not hasattr(self, 'stl_viewer') or not self.stl_viewer:
            print("STL viewer not available, skipping closed-loop constraint enforcement")
            return

        import vtk
        import math
        import numpy as np

        # Node
        closed_loop_nodes = []
        all_nodes_dict = {}  # NOTE

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

            # Get node
            if parent_link_name not in all_nodes_dict:
                print(f"  ✗ WARNING: Parent link '{parent_link_name}' not found in nodes")
                continue
            if child_link_name not in all_nodes_dict:
                print(f"  ✗ WARNING: Child link '{child_link_name}' not found in nodes")
                continue

            parent_node = all_nodes_dict[parent_link_name]
            child_node = all_nodes_dict[child_link_name]

            # Get transform
            if parent_node not in self.stl_viewer.transforms:
                print(f"  ✗ WARNING: Parent node '{parent_link_name}' has no transform")
                continue
            if child_node not in self.stl_viewer.transforms:
                print(f"  ✗ WARNING: Child node '{child_link_name}' has no transform")
                continue

            parent_transform = self.stl_viewer.transforms[parent_node]

            # Create TODO
            joint_transform = vtk.vtkTransform()
            joint_transform.Identity()

            # Add TODO
            joint_transform.Translate(origin_xyz[0], origin_xyz[1], origin_xyz[2])

            # Add RPY
            if origin_rpy and len(origin_rpy) == 3:
                roll_deg = math.degrees(origin_rpy[0])
                pitch_deg = math.degrees(origin_rpy[1])
                yaw_deg = math.degrees(origin_rpy[2])
                joint_transform.RotateZ(yaw_deg)
                joint_transform.RotateY(pitch_deg)
                joint_transform.RotateX(roll_deg)

            # Compute transform target parent @ joint :
            target_child_transform = vtk.vtkTransform()
            target_child_transform.Concatenate(parent_transform)
            target_child_transform.Concatenate(joint_transform)

            # Get current position
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

            # Move
            # Compute correction target @ inv current :
            correction_transform = vtk.vtkTransform()
            correction_transform.Concatenate(target_child_transform)

            inverse_current = vtk.vtkTransform()
            inverse_current.DeepCopy(current_child_transform)
            inverse_current.Inverse()
            correction_transform.Concatenate(inverse_current)

            # Apply TODO
            print(f"  Applying correction to child node and its descendants...")
            self._apply_transform_correction_to_subtree(child_node, correction_transform, set())

            print(f"  ✓ Closed-loop constraint applied for {joint_name}")

        print(f"\nTotal {len(closed_loop_nodes)} closed-loop constraint(s) enforced")

    def _apply_transform_correction_to_subtree(self, node, correction_transform, visited):
        """nodetextーtext"""
        if node in visited:
            return
        visited.add(node)

        # Node transform
        if node in self.stl_viewer.transforms and node in self.stl_viewer.stl_actors:
            import vtk
            current_transform = self.stl_viewer.transforms[node]

            # Correction @ current
            new_transform = vtk.vtkTransform()
            new_transform.Concatenate(correction_transform)
            new_transform.Concatenate(current_transform)

            # Apply transform
            self.stl_viewer.transforms[node].DeepCopy(new_transform)
            self.stl_viewer.stl_actors[node].SetUserTransform(self.stl_viewer.transforms[node])

            # Update TODO
            self.stl_viewer.update_collider_transform(node)

            print(f"    ✓ Applied correction to {node.name()}")

        # Apply TODO
        for output_port in node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()
                # Skip
                if not isinstance(child_node, ClosedLoopJointNode):
                    self._apply_transform_correction_to_subtree(child_node, correction_transform, visited)

    def build_r_from_l(self):
        """text(l_)textnodetext(r_)textnodetextcreate"""
        print("Building right side (r_) from left side (l_)...")

        try:
            # Node
            l_nodes = {}
            for node in self.all_nodes():
                node_name = node.name()
                if node_name.startswith('l_'):
                    l_nodes[node_name] = node
                    print(f"Found left node: {node_name}")

            if not l_nodes:
                print("No left side nodes (l_) found")
                return

            # Existing l_ r_
            existing_r_nodes = {}
            # Save existing r_
            existing_r_collider_settings = {}
            # Save existing r_
            existing_r_mesh_settings = {}
            for node in self.all_nodes():
                node_name = node.name()
                if node_name.startswith('r_'):
                    # R_ l_ check
                    corresponding_l_name = node_name.replace('r_', 'l_', 1)
                    if corresponding_l_name in l_nodes:
                        existing_r_nodes[node_name] = node
                        print(f"Found existing right node: {node_name} (corresponds to {corresponding_l_name})")
                        
                        # Save r_
                        collider_settings = {}
                        import copy
                        # Save colliders
                        if hasattr(node, 'colliders') and node.colliders:
                            collider_settings['colliders'] = copy.deepcopy(node.colliders)

                        if collider_settings:
                            existing_r_collider_settings[corresponding_l_name] = collider_settings
                            print(f"  Saved collider settings for {node_name}")
                            if 'colliders' in collider_settings:
                                print(f"    Saved {len(collider_settings['colliders'])} collider(s) in new format")
                        
                        # Save existing mesh
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

            # Remove existing
            for r_node in existing_r_nodes.values():
                print(f"Removing existing node: {r_node.name()}")
                self.remove_node(r_node)
            
            # Check r_
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

            # Compute r_
            # Get existing node
            max_x = -float('inf')
            min_x = float('inf')
            for node in self.all_nodes():
                pos = node.pos()
                x = pos.x() if hasattr(pos, 'x') else pos[0]
                max_x = max(max_x, x)
                min_x = min(min_x, x)

            # Get l_
            l_min_x = float('inf')
            l_max_x = -float('inf')
            for l_node in l_nodes.values():
                pos = l_node.pos()
                x = pos.x() if hasattr(pos, 'x') else pos[0]
                l_min_x = min(l_min_x, x)
                l_max_x = max(l_max_x, x)

            # Compute offset r_
            # 200
            x_offset = max_x - l_min_x + 200

            print(f"Positioning r_ nodes with X offset: {x_offset}")

            # Create TODO
            l_to_r_mapping = {}
            for l_name, l_node in l_nodes.items():
                r_name = l_name.replace('l_', 'r_', 1)

                print(f"\nCreating {r_name} from {l_name}")
                # Get __identifier__
                if hasattr(l_node, '__identifier__'):
                    node_type = l_node.__identifier__ + '.' + type(l_node).__name__
                else:
                    node_type = type(l_node).__module__ + '.' + type(l_node).__name__

                print(f"  Node type: {node_type}")

                # Compute position offset l_ X
                l_pos = l_node.pos()
                # Pos list qpointf qpointf
                if isinstance(l_pos, list):
                    # Apply l_
                    r_pos = QtCore.QPointF(l_pos[0] + x_offset, l_pos[1])
                else:
                    # Apply l_
                    r_pos = QtCore.QPointF(l_pos.x() + x_offset, l_pos.y())

                print(f"  Position: ({r_pos.x()}, {r_pos.y()})")

                # Create TODO
                r_node = self.create_node(node_type, name=r_name, pos=r_pos)

                l_to_r_mapping[l_node] = r_node

                # Todo
                # R_
                r_mesh_found = False
                
                # Existing mesh r_
                if l_name in existing_r_mesh_settings and 'stl_file' in existing_r_mesh_settings[l_name]:
                    # Existing
                    existing_mesh_file = existing_r_mesh_settings[l_name]['stl_file']
                    r_node.stl_file = existing_mesh_file
                    print(f"  Keeping existing mesh file for {r_name}: {existing_mesh_file}")
                    
                    # Existing mesh_scale visual_origin
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
                        # Mesh_scale visual_origin is_mesh_reversed
                        r_node.is_mesh_reversed = is_mesh_reversed_check(
                            r_node.visual_origin if hasattr(r_node, 'visual_origin') and r_node.visual_origin else {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                            r_node.mesh_scale if hasattr(r_node, 'mesh_scale') else [1.0, 1.0, 1.0]
                        )
                        print(f"  Calculated is_mesh_reversed flag: {r_node.is_mesh_reversed}")
                    
                    # Show existing mesh 3D
                    if hasattr(self, 'stl_viewer') and self.stl_viewer:
                        try:
                            self.stl_viewer.load_stl_for_node(r_node)
                            print(f"  Loaded existing mesh for {r_name}")
                        except Exception as e:
                            print(f"  Warning: Could not load existing mesh for {r_name}: {str(e)}")
                elif hasattr(l_node, 'stl_file') and l_node.stl_file:
                    # Existing mesh r_
                    stl_file = l_node.stl_file
                    
                    # R_
                    if stl_file and 'l_' in os.path.basename(stl_file):
                        # Generate l_ r_
                        r_stl_file = stl_file.replace('/l_', '/r_').replace('\\l_', '\\r_')
                        # L_
                        dirname = os.path.dirname(r_stl_file)
                        basename = os.path.basename(r_stl_file)
                        basename = basename.replace('l_', 'r_', 1)
                        r_stl_file = os.path.join(dirname, basename)
                        
                        # Confirm r_
                        if os.path.exists(r_stl_file):
                            r_node.stl_file = r_stl_file
                            r_mesh_found = True
                            print(f"  Found r_ mesh file: {r_stl_file}")
                            
                            # 3d
                            if hasattr(self, 'stl_viewer') and self.stl_viewer:
                                try:
                                    self.stl_viewer.load_stl_for_node(r_node)
                                    print(f"  Loaded r_ mesh for {r_name}")
                                except Exception as e:
                                    print(f"  Warning: Could not load r_ mesh for {r_name}: {str(e)}")
                    
                    # R_ l_ mesh
                    if not r_mesh_found:
                        # L_ mesh mesh_scale process
                        r_node.stl_file = stl_file
                        print(f"  r_ mesh not found, using l_ mesh with mirroring: {stl_file}")
                        
                        # 3d
                        if hasattr(self, 'stl_viewer') and self.stl_viewer:
                            try:
                                self.stl_viewer.load_stl_for_node(r_node)
                                print(f"  Loaded l_ mesh (mirrored) for {r_name}")
                            except Exception as e:
                                print(f"  Warning: Could not load l_ mesh for {r_name}: {str(e)}")

                # Todo
                if hasattr(l_node, 'volume_value'):
                    r_node.volume_value = l_node.volume_value
                if hasattr(l_node, 'mass_value'):
                    r_node.mass_value = l_node.mass_value

                # Todo
                # > mesh > : urdf
                
                # Urdf
                has_urdf_inertia = (hasattr(l_node, 'inertia') and 
                                   l_node.inertia and 
                                   any(abs(v) > 1e-12 for v in l_node.inertia.values() if isinstance(v, (int, float))))
                
                if has_urdf_inertia:
                    print(f"  Using URDF-derived inertia (priority method)")
                    print(f"  [BUILD_R_FROM_L] Creating {r_name} from {l_name}")
                    print(f"    Source l_node.inertia: ixx={l_node.inertia.get('ixx', 0):.9e}, ixy={l_node.inertia.get('ixy', 0):.9e}, ixz={l_node.inertia.get('ixz', 0):.9e}")
                    print(f"                          iyy={l_node.inertia.get('iyy', 0):.9e}, iyz={l_node.inertia.get('iyz', 0):.9e}, izz={l_node.inertia.get('izz', 0):.9e}")
                    # Value urdf
                    mirrored_inertia = mirror_inertia_tensor_left_right(l_node.inertia)
                    if mirrored_inertia:
                        r_node.inertia = mirrored_inertia
                        print(f"  ✓ Mirrored URDF inertia tensor (negated ixy, iyz)")
                        print(f"    Original: ixx={l_node.inertia.get('ixx', 0):.9e}, ixy={l_node.inertia.get('ixy', 0):.9e}")
                        print(f"    Mirrored: ixx={mirrored_inertia.get('ixx', 0):.9e}, ixy={mirrored_inertia.get('ixy', 0):.9e}")
                        print(f"  [BUILD_R_FROM_L] Set r_node.inertia for {r_name}: ixx={mirrored_inertia.get('ixx', 0):.9e}, ixy={mirrored_inertia.get('ixy', 0):.9e}, ixz={mirrored_inertia.get('ixz', 0):.9e}")
                        print(f"                                          iyy={mirrored_inertia.get('iyy', 0):.9e}, iyz={mirrored_inertia.get('iyz', 0):.9e}, izz={mirrored_inertia.get('izz', 0):.9e}")
                    
                    # Center of mass center mass
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
                
                # 2: urdf
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
                                # Mesh
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

                # Existing transform 3:
                if not has_urdf_inertia and not use_mesh_recalculation:
                    print(f"  Using property transformation (last fallback method)")
                    print(f"  ⚠ FALLBACK_INERTIA_USED: Using property transformation for {r_name}")
                    # Todo
                    if hasattr(l_node, 'inertia') and l_node.inertia:
                        mirrored_inertia = mirror_inertia_tensor_left_right(l_node.inertia)
                        if mirrored_inertia:
                            r_node.inertia = mirrored_inertia
                            print(f"  ✓ Mirrored inertia tensor (negated ixy, iyz)")
                            print(f"    FALLBACK_INERTIA_USED: ixx={mirrored_inertia.get('ixx', 0):.9e}")
                    
                    # Center of mass center mass
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

                # Body angle angle offset x y z body angle angle x y z
                if hasattr(l_node, 'body_angle'):
                    r_node.body_angle = l_node.body_angle.copy()
                    print(f"  Copied body_angle: {r_node.body_angle}")

                # Mesh scale visual origin process mesh visual
                # Existing mesh r_
                # R_
                if l_name not in existing_r_mesh_settings and not r_mesh_found:
                    # R_ l_ mesh
                    # Mesh scale mesh y
                    if hasattr(l_node, 'mesh_scale'):
                        r_node.mesh_scale = [l_node.mesh_scale[0], -l_node.mesh_scale[1], l_node.mesh_scale[2]]
                        print(f"  Copied mesh_scale with Y-axis mirrored: {l_node.mesh_scale} -> {r_node.mesh_scale}")

                    # Visual origin visual y
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
                    # R_ mesh_scale visual_origin
                    if hasattr(l_node, 'mesh_scale'):
                        r_node.mesh_scale = l_node.mesh_scale.copy() if hasattr(l_node.mesh_scale, 'copy') else l_node.mesh_scale
                        print(f"  Copied mesh_scale (no mirroring): {r_node.mesh_scale}")
                    if hasattr(l_node, 'visual_origin') and l_node.visual_origin:
                        import copy
                        r_node.visual_origin = copy.deepcopy(l_node.visual_origin)
                        print(f"  Copied visual_origin (no mirroring): {r_node.visual_origin}")
                
                # Mesh_scale visual_origin
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

                # Apply existing r_
                if l_name in existing_r_collider_settings:
                    # Existing r_
                    collider_settings = existing_r_collider_settings[l_name]
                    print(f"  Applying existing r_ collider settings for {r_name}")
                    import copy

                    # : colliders
                    # Enable check type None type: None
                    has_valid_colliders = (
                        'colliders' in collider_settings and collider_settings['colliders'] and
                        any(c.get('type') is not None for c in collider_settings['colliders'])
                    )

                    if has_valid_colliders:
                        r_node.colliders = copy.deepcopy(collider_settings['colliders'])
                        print(f"    Restored {len(r_node.colliders)} collider(s) from saved settings")
                else:
                    # Existing l_ l_ r_
                    import copy

                    # L_ l_ r_ transform : colliders
                    # Enable check type None type: None
                    l_has_valid_colliders = (
                        hasattr(l_node, 'colliders') and l_node.colliders and
                        any(c.get('type') is not None for c in l_node.colliders)
                    )

                    if l_has_valid_colliders:
                        r_node.colliders = copy.deepcopy(l_node.colliders)
                        # L_ r_ transform
                        for collider in r_node.colliders:
                            if collider.get('mesh') and 'l_' in collider['mesh']:
                                original_mesh = collider['mesh']
                                collider['mesh'] = original_mesh.replace('l_', 'r_', 1)
                                print(f"    Collider mesh path converted: {original_mesh} -> {collider['mesh']}")
                        print(f"  Copied {len(r_node.colliders)} collider(s) from l_ node")

                    # Apply Collider
                    r_collider_found = False

                    # _collider xml r_ stl _collider.xml
                    if hasattr(r_node, 'stl_file') and r_node.stl_file:
                        r_stl_path = r_node.stl_file
                        if os.path.exists(r_stl_path):
                            mesh_dir = os.path.dirname(r_stl_path)
                            mesh_basename = os.path.splitext(os.path.basename(r_stl_path))[0]
                            r_collider_xml_path = os.path.join(mesh_dir, f"{mesh_basename}_collider.xml")

                            if os.path.exists(r_collider_xml_path):
                                print(f"  Found r_ collider XML: {r_collider_xml_path}")
                                # Collider xml collider xml
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

                # Y
                if hasattr(l_node, 'points') and hasattr(r_node, 'points'):
                    # Compute TODO
                    l_port_count = len(l_node.output_ports())
                    l_points_count = len(l_node.points) if hasattr(l_node, 'points') else 0

                    # Connect + 1
                    max_used_port = -1
                    for port_idx, output_port in enumerate(l_node.output_ports()):
                        if output_port.connected_ports():
                            max_used_port = port_idx
                    required_port_count = max_used_port + 1 if max_used_port >= 0 else 0

                    # Connect XML current XML
                    target_port_count = max(l_port_count, l_points_count, required_port_count)

                    print(f"  Port count - Current: {l_port_count}, Points: {l_points_count}, Required: {required_port_count}, Target: {target_port_count}")

                    # Add port
                    while len(r_node.output_ports()) < target_port_count:
                        if hasattr(r_node, '_add_output'):
                            r_node._add_output()
                            print(f"  Added output port to {r_name} (now {len(r_node.output_ports())} ports)")

                    # Remove port
                    while len(r_node.output_ports()) > target_port_count:
                        if hasattr(r_node, 'remove_output'):
                            # Remove connect
                            if r_node.output_ports():
                                last_port = r_node.output_ports()[-1]
                                last_port.clear_connections()
                            r_node.remove_output()
                            print(f"  Removed output port from {r_name} (now {len(r_node.output_ports())} ports)")

                    r_node.points = []
                    for point in l_node.points:
                        r_point = point.copy()
                        # Y
                        if 'xyz' in r_point:
                            xyz = r_point['xyz']
                            r_point['xyz'] = [xyz[0], -xyz[1], xyz[2]]
                        r_node.points.append(r_point)

                    # Update cumulative_coords
                    if hasattr(r_node, 'cumulative_coords'):
                        r_node.cumulative_coords = []
                        for i in range(len(r_node.points)):
                            r_node.cumulative_coords.append({'point_index': i, 'xyz': [0, 0, 0]})

                print(f"  Created {r_name} successfully")

            # Connect TODO
            print("\nMirroring connections...")
            connection_count = 0
            failed_connections = []

            for l_node, r_node in l_to_r_mapping.items():
                # Connect TODO
                for port_idx, output_port in enumerate(l_node.output_ports()):
                    for connected_port in output_port.connected_ports():
                        connected_node = connected_port.node()
                        connected_node_name = connected_node.name()

                        # R_
                        r_connected_node = None

                        # 1 l_ if 1.
                        if connected_node in l_to_r_mapping:
                            r_connected_node = l_to_r_mapping[connected_node]
                            print(f"  Found in mapping: {connected_node_name} -> {r_connected_node.name()}")

                        # 2 r_ if 2. l_
                        elif connected_node_name.startswith('r_'):
                            # R_ l_
                            import re
                            # Remove TODO
                            base_name = re.sub(r'\s+\d+$', '', connected_node_name)
                            l_version_name = base_name.replace('r_', 'l_', 1)

                            # Todo
                            for l_n, r_n in l_to_r_mapping.items():
                                if l_n.name() == l_version_name:
                                    r_connected_node = r_n
                                    print(f"  Found r_ node {connected_node_name} -> using r_ version: {r_n.name()}")
                                    break

                            # Name
                            if r_connected_node is None:
                                l_version_name_original = connected_node_name.replace('r_', 'l_', 1)
                                for l_n, r_n in l_to_r_mapping.items():
                                    if l_n.name() == l_version_name_original:
                                        r_connected_node = r_n
                                        print(f"  Found r_ node {connected_node_name} (exact match) -> using r_ version: {r_n.name()}")
                                        break

                        # 3 name 3.
                        if r_connected_node is None:
                            # L_ r_
                            target_name = connected_node_name.replace('l_', 'r_', 1) if 'l_' in connected_node_name else 'r_' + connected_node_name
                            for node in self.all_nodes():
                                if node.name() == target_name:
                                    r_connected_node = node
                                    print(f"  Found by name search: {target_name}")
                                    break

                        if r_connected_node:
                            # Get TODO
                            if port_idx < len(r_node.output_ports()):
                                r_output_port = r_node.output_ports()[port_idx]

                                # Confirm TODO
                                if not r_connected_node.input_ports():
                                    print(f"  Warning: {r_connected_node.name()} has no input ports")
                                    failed_connections.append(f"{r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()} (no input port)")
                                    continue

                                r_input_port = r_connected_node.input_ports()[0]

                                # Check
                                if r_input_port in r_output_port.connected_ports():
                                    print(f"  Already connected: {r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()}.{r_input_port.name()}")
                                    connection_count += 1
                                    continue

                                # Connect TODO
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
                            # R_
                            error_msg = f"{r_node.name()}.out_{port_idx+1} -> {connected_node_name}: No corresponding r_ node found"
                            failed_connections.append(error_msg)
                            print(f"  Warning: {error_msg}")

            print(f"\nConnection summary: {connection_count} connections established")
            if failed_connections:
                print(f"Failed connections ({len(failed_connections)}):")
                for fc in failed_connections:
                    print(f"  - {fc}")

            # Connect r_ l_
            print("\nConnecting orphaned r_ nodes to parent nodes...")
            orphan_connection_count = 0
            for l_node, r_node in l_to_r_mapping.items():
                # Check r_
                r_input_ports = r_node.input_ports()
                is_orphaned = True
                if r_input_ports:
                    for input_port in r_input_ports:
                        if input_port.connected_ports():
                            is_orphaned = False
                            break
                
                if is_orphaned:
                    print(f"  Found orphaned r_ node: {r_node.name()}")
                    
                    # L_
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
                                # Confirm port
                                parent_output_ports = l_parent_node.output_ports()
                                for port_idx, output_port in enumerate(parent_output_ports):
                                    if output_port == l_parent_port:
                                        l_parent_port_idx = port_idx
                                        # Port
                                        if hasattr(l_parent_node, 'points') and l_parent_node.points:
                                            if port_idx < len(l_parent_node.points):
                                                l_connected_point_idx = port_idx
                                                l_connected_point_xyz = l_parent_node.points[port_idx].get('xyz', [0, 0, 0])
                                                break
                                        break
                                break
                    
                    if l_parent_node and l_parent_node.name().startswith('l_'):
                        # If l_ r_
                        l_parent_name = l_parent_node.name()
                        r_parent_name = l_parent_name.replace('l_', 'r_', 1)
                        r_parent_node = None
                        for node in self.all_nodes():
                            if node.name() == r_parent_name:
                                r_parent_node = node
                                break
                        
                        if r_parent_node:
                            print(f"    Found r_ parent node: {r_parent_name}")
                            
                            # Points x z y
                            if hasattr(r_parent_node, 'points') and r_parent_node.points and l_connected_point_xyz:
                                target_point_idx = None
                                l_x, l_y, l_z = l_connected_point_xyz
                                
                                for point_idx, point in enumerate(r_parent_node.points):
                                    point_xyz = point.get('xyz', [0, 0, 0])
                                    r_x, r_y, r_z = point_xyz
                                    
                                    # X z y check 1e-6
                                    if (abs(r_x - l_x) < 1e-6 and 
                                        abs(r_z - l_z) < 1e-6 and 
                                        abs(r_y + l_y) < 1e-6):
                                        target_point_idx = point_idx
                                        print(f"    Found matching point at index {point_idx}: xyz=({r_x:.6f}, {r_y:.6f}, {r_z:.6f}) (l_ point: ({l_x:.6f}, {l_y:.6f}, {l_z:.6f}))")
                                        break
                                
                                if target_point_idx is not None:
                                    # Get TODO
                                    if target_point_idx < len(r_parent_node.output_ports()):
                                        r_parent_output_port = r_parent_node.output_ports()[target_point_idx]
                                        
                                        # Get r_
                                        if r_node.input_ports():
                                            r_input_port = r_node.input_ports()[0]
                                            
                                            # Check
                                            if r_input_port in r_parent_output_port.connected_ports():
                                                print(f"    Already connected: {r_parent_name}.out_{target_point_idx+1} -> {r_node.name()}")
                                                orphan_connection_count += 1
                                            else:
                                                # Connect TODO
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
                        # Base_link l_
                        print(f"    l_ parent node is not l_ system: {l_parent_node.name()}")
                        
                        # Points x z y
                        if hasattr(l_parent_node, 'points') and l_parent_node.points and l_connected_point_xyz:
                            target_point_idx = None
                            l_x, l_y, l_z = l_connected_point_xyz
                            
                            for point_idx, point in enumerate(l_parent_node.points):
                                point_xyz = point.get('xyz', [0, 0, 0])
                                p_x, p_y, p_z = point_xyz
                                
                                # X z y check 1e-6
                                if (abs(p_x - l_x) < 1e-6 and 
                                    abs(p_z - l_z) < 1e-6 and 
                                    abs(p_y + l_y) < 1e-6):
                                    target_point_idx = point_idx
                                    print(f"    Found matching point at index {point_idx}: xyz=({p_x:.6f}, {p_y:.6f}, {p_z:.6f}) (l_ point: ({l_x:.6f}, {l_y:.6f}, {l_z:.6f}))")
                                    break
                            
                            if target_point_idx is not None:
                                # Get TODO
                                if target_point_idx < len(l_parent_node.output_ports()):
                                    parent_output_port = l_parent_node.output_ports()[target_point_idx]
                                    
                                    # Get r_
                                    if r_node.input_ports():
                                        r_input_port = r_node.input_ports()[0]
                                        
                                        # Check
                                        if r_input_port in parent_output_port.connected_ports():
                                            print(f"    Already connected: {l_parent_node.name()}.out_{target_point_idx+1} -> {r_node.name()}")
                                            orphan_connection_count += 1
                                        else:
                                            # Connect TODO
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

            # R_ l_
            print("\nRearranging r_ nodes with mirrored layout from l_ nodes...")
            if l_to_r_mapping:
                # Compute l_
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

                # Compute existing r_
                all_min_x = float('inf')
                all_max_x = -float('inf')
                all_min_y = float('inf')
                all_max_y = -float('inf')

                for node in self.all_nodes():
                    # R_
                    if node not in l_to_r_mapping.values():
                        node_pos = node.pos()
                        x = node_pos.x() if hasattr(node_pos, 'x') else node_pos[0]
                        y = node_pos.y() if hasattr(node_pos, 'y') else node_pos[1]
                        all_min_x = min(all_min_x, x)
                        all_max_x = max(all_max_x, x)
                        all_min_y = min(all_min_y, y)
                        all_max_y = max(all_max_y, y)

                # Compute r_
                # 300
                r_base_x = all_max_x + 300
                r_base_y = l_min_y  # l_Y

                print(f"  r_ base position: ({r_base_x:.1f}, {r_base_y:.1f})")

                # R_ l_
                for l_node, r_node in l_to_r_mapping.items():
                    l_x, l_y = l_positions[l_node]
                    # Compute l_
                    rel_x = l_x - l_min_x
                    rel_y = l_y - l_min_y
                    # Compute r_
                    new_x = r_base_x + rel_x
                    new_y = r_base_y + rel_y
                    r_node.set_pos(new_x, new_y)
                    print(f"  Repositioned {r_node.name()} to ({new_x:.1f}, {new_y:.1f}) (offset: {rel_x:.1f}, {rel_y:.1f})")

            # Position
            self.recalculate_all_positions()

            # Node color
            self.update_all_node_colors()

            # Update all color r_ 3D
            if self.stl_viewer:
                for r_node in l_to_r_mapping.values():
                    if hasattr(self.stl_viewer, 'apply_color_to_node'):
                        self.stl_viewer.apply_color_to_node(r_node)
                print(f"  Applied colors to {len(l_to_r_mapping)} r_ nodes in 3D view")

            # Update TODO
            if self.stl_viewer:
                self.stl_viewer.refresh_collider_display()

            print(f"\nSuccessfully created {len(l_to_r_mapping)} right side nodes from left side")

        except Exception as e:
            print(f"Error building right side from left side: {str(e)}")
            import traceback
            traceback.print_exc()

    def disconnect_ports(self, from_port, to_port):
        """textーtext"""
        try:
            print(f"Disconnecting ports: {from_port.node().name()}.{from_port.name()} -> {to_port.node().name()}.{to_port.name()}")
            
            # Connect TODO
            child_node = to_port.node()
            if child_node:
                self.stl_viewer.reset_stl_transform(child_node)
            
            # Todo
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
                # On_port_disconnected
                self.on_port_disconnected(to_port, from_port)
                return True
            else:
                print("Failed to disconnect ports")
                return False
                
        except Exception as e:
            print(f"Error disconnecting ports: {str(e)}")
            return False

    def _write_joint(self, file, parent_node, child_node):
        """jointtext"""
        try:
            # Get origin
            origin_xyz = [0, 0, 0]  # NOTE
            origin_rpy = [0.0, 0.0, 0.0]  # NOTE
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

            # Value
            if hasattr(child_node, 'rotation_axis'):
                if child_node.rotation_axis == 3:  # Fixed
                    file.write(f'  <joint name="{joint_name}" type="fixed">\n')
                    file.write(f'    <origin xyz="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" rpy="{origin_rpy[0]} {origin_rpy[1]} {origin_rpy[2]}"/>\n')
                    file.write(f'    <parent link="{parent_node.name()}"/>\n')
                    file.write(f'    <child link="{child_node.name()}"/>\n')
                    file.write('  </joint>\n')
                else:
                    # Process
                    file.write(f'  <joint name="{joint_name}" type="revolute">\n')
                    axis = [0, 0, 0]
                    if child_node.rotation_axis == 0:    # X-axis
                        axis = [1, 0, 0]
                    elif child_node.rotation_axis == 1:  # Y-axis
                        axis = [0, 1, 0]
                    else:                          # Z-axis
                        axis = [0, 0, 1]

                    file.write(f'    <origin xyz="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" rpy="{origin_rpy[0]} {origin_rpy[1]} {origin_rpy[2]}"/>\n')
                    file.write(f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n')
                    file.write(f'    <parent link="{parent_node.name()}"/>\n')
                    file.write(f'    <child link="{child_node.name()}"/>\n')

                    # Get Joint Joint limit
                    lower = getattr(child_node, 'joint_lower', -3.14159)
                    upper = getattr(child_node, 'joint_upper', 3.14159)
                    effort = getattr(child_node, 'joint_effort', 10.0)
                    velocity = getattr(child_node, 'joint_velocity', 3.0)

                    # Effort velocity urdf limit
                    file.write(f'    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>\n')

                    # Damping friction dynamics
                    damping = getattr(child_node, 'damping', 0.0)
                    friction = getattr(child_node, 'friction', 0.0)
                    file.write(f'    <dynamics damping="{damping}" friction="{friction}"/>\n')

                    file.write('  </joint>\n')

        except Exception as e:
            print(f"Error writing joint: {str(e)}")
            traceback.print_exc()

    def _format_visual_origin(self, node):
        """Visual origintextーtext(text)"""
        if hasattr(node, 'visual_origin') and isinstance(node.visual_origin, dict):
            xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
            rpy = node.visual_origin.get('rpy', [0.0, 0.0, 0.0])
            # Todo
            if xyz != [0.0, 0.0, 0.0] or rpy != [0.0, 0.0, 0.0]:
                return f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n'
        return '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'

    def _format_mesh_scale(self, node):
        """Mesh scaletextーtext(text)"""
        if hasattr(node, 'mesh_scale') and isinstance(node.mesh_scale, (list, tuple)) and len(node.mesh_scale) == 3:
            scale = node.mesh_scale
            # 1 0 1 0 1 0 1.0
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
        """text"""
        try:
            file.write(f'  <link name="{node.name()}">\n')

            # Todo
            if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial origin output inertial origin
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

            # Todo
            if hasattr(node, 'stl_file') and node.stl_file:
                try:
                    mesh_dir_name = "meshes"
                    if self.meshes_dir:
                        dir_name = os.path.basename(self.meshes_dir)
                        if dir_name.startswith('mesh'):
                            mesh_dir_name = dir_name

                    # Todo
                    # Change TODO
                    original_filename = os.path.basename(node.stl_file)
                    base_name = os.path.splitext(original_filename)[0]
                    mesh_filename = f"{base_name}{mesh_format}"
                    package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{mesh_filename}"

                    file.write('    <visual>\n')
                    # Visual origin output visual
                    file.write(self._format_visual_origin(node))
                    file.write('      <geometry>\n')
                    # Mesh scale output mesh
                    scale_attr = self._format_mesh_scale(node)
                    file.write(f'        <mesh filename="{package_path}"{scale_attr}/>\n')
                    file.write('      </geometry>\n')

                    # Add TODO
                    if hasattr(node, 'node_color') and len(node.node_color) >= 3:
                        rgb = node.node_color
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(rgb[0] * 255),
                            int(rgb[1] * 255),
                            int(rgb[2] * 255)
                        )
                        file.write(f'      <material name="{hex_color}"/>\n')

                    file.write('    </visual>\n')

                    # Add TODO
                    for port_index, port in enumerate(node.output_ports()):
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if hasattr(dec_node, 'stl_file') and dec_node.stl_file:
                                    # Change TODO
                                    dec_original = os.path.basename(dec_node.stl_file)
                                    dec_base_name = os.path.splitext(dec_original)[0]
                                    dec_mesh_filename = f"{dec_base_name}{mesh_format}"
                                    dec_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{dec_mesh_filename}"

                                    # Get TODO
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

                    # Todo
                    self._write_urdf_collision(file, node, package_path, mesh_dir_name, mesh_format)

                except Exception as e:
                    print(f"Error processing STL file for node {node.name()}: {str(e)}")
                    traceback.print_exc()

            file.write('  </link>\n')

        except Exception as e:
            print(f"Error writing link: {str(e)}")
            traceback.print_exc()

    def _write_link_unity(self, file, node, materials, unity_dir_name):
        """Unitytext"""
        try:
            file.write(f'  <link name="{node.name()}">\n')

            # Todo
            if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial origin output inertial origin
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

            # Todo
            if hasattr(node, 'stl_file') and node.stl_file:
                try:
                    # Todo
                    stl_filename = os.path.basename(node.stl_file)
                    # Path unity meshes
                    package_path = f"package://meshes/{stl_filename}"

                    file.write('    <visual>\n')
                    # Visual origin output visual
                    file.write(self._format_visual_origin(node))
                    file.write('      <geometry>\n')
                    # Mesh scale output mesh
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

                    # Add TODO
                    for port_index, port in enumerate(node.output_ports()):
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if hasattr(dec_node, 'stl_file') and dec_node.stl_file:
                                    dec_stl = os.path.basename(dec_node.stl_file)
                                    dec_path = f"package://meshes/{dec_stl}"

                                    # Get TODO
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

                    # Path unity
                    self._write_urdf_collision(file, node, package_path, unity_mode=True)

                except Exception as e:
                    print(f"Error processing STL file for node {node.name()}: {str(e)}")
                    traceback.print_exc()

            file.write('  </link>\n')

        except Exception as e:
            print(f"Error writing link for Unity: {str(e)}")
            traceback.print_exc()

    def export_for_unity(self):
        """Unitytextーtext"""
        try:
            # Show TODO
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

            # Generate TODO
            robot_name = self.get_robot_name()
            unity_dir_name = f"{robot_name}_unity_description"
            unity_dir_path = os.path.join(base_dir, unity_dir_name)

            # Create TODO
            os.makedirs(unity_dir_path, exist_ok=True)
            print(f"Created Unity description directory: {unity_dir_path}")

            # Create meshes
            meshes_dir = os.path.join(unity_dir_path, "meshes")
            os.makedirs(meshes_dir, exist_ok=True)
            print(f"Created meshes directory: {meshes_dir}")

            # Stl
            copied_files = []
            for node in self.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    if os.path.exists(node.stl_file):
                        # Get TODO
                        stl_filename = os.path.basename(node.stl_file)
                        # Generate path
                        dest_path = os.path.join(meshes_dir, stl_filename)
                        # File
                        shutil.copy2(node.stl_file, dest_path)
                        copied_files.append(stl_filename)
                        print(f"Copied mesh file: {stl_filename}")

            # Generate URDF
            urdf_file = os.path.join(unity_dir_path, f"{robot_name}.urdf")
            with open(urdf_file, 'w', encoding='utf-8') as f:
                # Todo
                f.write('<?xml version="1.0"?>\n')
                f.write(f'<robot name="{robot_name}">\n\n')

                # Todo
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

                # Export material
                f.write('<!-- material color setting -->\n')
                for hex_color, rgb in materials.items():
                    f.write(f'<material name="{hex_color}">\n')
                    f.write(f'  <color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"/>\n')
                    f.write('</material>\n')
                f.write('\n')

                # Base_link output
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
        """Unitytextーtext"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless skip <visual> massless decoration visual
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        if node.name() == "base_link":
            # Base_link output
            self._write_base_link(file)
        
        # Current node link process
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                if child_node not in visited_nodes:
                    # Massless decoration link output massless decoration
                    if not (hasattr(child_node, 'massless_decoration') and child_node.massless_decoration):
                        # Output
                        self._write_joint(file, node, child_node)
                        file.write('\n')
                        
                        # Next link output path unity
                        self._write_link_unity(file, child_node, materials, unity_dir_name)
                        file.write('\n')
                        
                        # Process
                        self._write_tree_structure_unity(file, child_node, node, visited_nodes, materials, unity_dir_name)

    def export_mjcf(self):
        # Confirm : MJCF
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
        """MuJoCo MJCFtextーtext(textーtext)"""
        try:
            # Todo
            self.collect_closed_loop_joints_from_nodes()

            # Todo
            import re
            import shutil
            robot_name = self.robot_name or "robot"
            # Todo
            robot_name = self._sanitize_name(robot_name)

            # Create base_link height input
            dialog = QtWidgets.QDialog(self.widget)
            dialog.setWindowTitle("MJCF Export - Settings")
            dialog.setMinimumWidth(400)
            
            layout = QtWidgets.QVBoxLayout(dialog)
            
            # Todo
            dir_label = QtWidgets.QLabel("Enter directory name for MJCF export:")
            layout.addWidget(dir_label)
            dir_input = QtWidgets.QLineEdit()
            dir_input.setText(f"{robot_name}_mjcf")
            layout.addWidget(dir_input)
            
            # Base_link height
            height_label = QtWidgets.QLabel("Default base_link height (m):")
            layout.addWidget(height_label)
            height_input = QtWidgets.QLineEdit()
            # Show Settings Settings
            default_height = getattr(self, 'default_base_link_height', DEFAULT_BASE_LINK_HEIGHT)
            if hasattr(self, 'graph') and hasattr(self.graph, 'default_base_link_height'):
                default_height = self.graph.default_base_link_height
            height_input.setText(f"{default_height:.4f}")
            # Todo
            height_input.setValidator(QDoubleValidator(0.0, 100.0, 6))
            layout.addWidget(height_input)

            # Fix Base to Ground
            fix_base_checkbox = QtWidgets.QCheckBox("Fix Base to Ground")
            fix_base_checkbox.setChecked(False)
            layout.addWidget(fix_base_checkbox)
            
            # Button
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
            
            # Get base_link height
            try:
                base_link_height = float(base_link_height_str) if base_link_height_str else default_height
            except ValueError:
                print(f"Warning: Invalid base_link height '{base_link_height_str}', using default {default_height}")
                base_link_height = default_height

            # Fix Base to Ground
            fix_base_to_ground = fix_base_checkbox.isChecked()
            
            # Todo
            dir_name = self._sanitize_name(dir_name)
            
            # Save base_link height
            if hasattr(self, 'graph'):
                self.graph.default_base_link_height = base_link_height

            # Select
            parent_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select Parent Directory for MJCF Export"
            )

            if not parent_dir:
                print("MJCF export cancelled")
                return False

            # Create MJCF
            mjcf_dir = os.path.join(parent_dir, dir_name)
            assets_dir = os.path.join(mjcf_dir, "assets")

            os.makedirs(mjcf_dir, exist_ok=True)
            os.makedirs(assets_dir, exist_ok=True)

            print(f"\n=== Exporting MJCF to {mjcf_dir} ===")
            print(f"MJCF directory: {mjcf_dir}")
            print(f"Assets directory: {assets_dir}")

            # Create TODO
            node_to_mesh = {}
            mesh_names = {}
            mesh_file_to_name = {}  # (mesh_filename, scale_tuple) → （、scale）
            mesh_file_to_scale = {}  # (mesh_filename, scale_tuple) → （）
            node_to_mesh_scale_key = {}  # node → (mesh_filename, scale_tuple)
            collider_file_to_name = {}  # → （）
            mesh_counter = 0

            # Determine which intermediate nodes should be skipped
            # (base_link is always skipped, base_link_sub is skipped if base_link_mjcf exists)
            base_link_mjcf_exists = self.get_node_by_name('base_link_mjcf') is not None

            for node in self.all_nodes():
                # Skip nodes with "Hide Mesh" enabled
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                # Skip base_link node (it's always skipped in MJCF export)
                if node.name() == 'base_link':
                    continue

                # Skip base_link_sub if base_link_mjcf exists (base_link_sub is intermediate)
                if node.name() == 'base_link_sub' and base_link_mjcf_exists:
                    continue

                # Check stl dae stl_file .stl .dae .obj
                mesh_file_path = None
                if hasattr(node, 'stl_file') and node.stl_file:
                    mesh_file_path = node.stl_file
                
                if mesh_file_path and os.path.exists(mesh_file_path):
                    # Get TODO
                    original_filename = os.path.basename(mesh_file_path)
                    file_ext = os.path.splitext(original_filename)[1].lower()
                    
                    # Check
                    supported_extensions = ['.stl', '.dae', '.obj']
                    if file_ext not in supported_extensions:
                        print(f"Warning: Unsupported mesh file extension '{file_ext}' for '{original_filename}'. Skipping.")
                        continue

                    # Get mesh_scale Y
                    mesh_scale = getattr(node, 'mesh_scale', [1.0, 1.0, 1.0])
                    needs_y_mirror = False
                    if len(mesh_scale) >= 2 and mesh_scale[1] < 0:
                        needs_y_mirror = True

                    # Transform obj
                    try:
                        import trimesh
                        
                        # If process if .dae
                        if file_ext == '.dae':
                            # Load Scene force mesh COLLADA Scene
                            try:
                                mesh = trimesh.load(mesh_file_path, force='mesh')
                                # Get Scene if mesh Scene
                                if hasattr(mesh, 'geometry'):
                                    # If scene
                                    if len(mesh.geometry) > 0:
                                        # Get mesh
                                        mesh = list(mesh.geometry.values())[0]
                                    else:
                                        print(f"Warning: DAE file '{original_filename}' has no geometry. Skipping.")
                                        continue
                            except Exception as e:
                                print(f"Warning: Could not load DAE file '{original_filename}': {e}")
                                # Load :
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
                            # Stl obj file .stl .obj
                            mesh = trimesh.load(mesh_file_path)

                        # Confirm MuJoCo 1 200000 MuJoCo
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

                        # If mesh y y
                        if needs_y_mirror:
                            import numpy as np
                            # Y
                            if hasattr(mesh, 'vertices'):
                                mesh.vertices[:, 1] = -mesh.vertices[:, 1]
                            # Y
                            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                                # Create if read-only
                                if not mesh.vertex_normals.flags.writeable:
                                    mesh.vertex_normals = mesh.vertex_normals.copy()
                                mesh.vertex_normals[:, 1] = -mesh.vertex_normals[:, 1]
                            # Todo
                            if hasattr(mesh, 'faces'):
                                mesh.faces = np.flip(mesh.faces, axis=1)
                            print(f"  Mirrored mesh along Y-axis for node '{node.name()}' (mesh_scale Y: {mesh_scale[1]})")

                        # Generate obj change .obj
                        mesh_filename = os.path.splitext(original_filename)[0] + '.obj'
                        dest_mesh_path = os.path.join(assets_dir, mesh_filename)

                        # Mujoco obj mujoco
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

                    # Meshdir mjcf
                    node_to_mesh[node] = mesh_filename

                    # Generate TODO
                    base_mesh_name = os.path.splitext(mesh_filename)[0]

                    # If mesh value y y
                    if needs_y_mirror and len(mesh_scale) >= 2:
                        # If mesh value y y
                        mesh_scale_for_mjcf = [mesh_scale[0], abs(mesh_scale[1]), mesh_scale[2]]
                    else:
                        # If value mjcf y x/z mjcf scale
                        mesh_scale_for_mjcf = mesh_scale
                    
                    # Create scale scale mesh
                    scale_key = tuple(mesh_scale_for_mjcf)
                    mesh_key = (mesh_filename, scale_key)
                    
                    # Check +
                    if mesh_key not in mesh_file_to_name:
                        # Mesh_file_to_name + :
                        # Scale 1 1 1 mesh
                        if mesh_scale_for_mjcf != [1.0, 1.0, 1.0]:
                            # Transform value scale
                            scale_suffix = f"-scale-{mesh_scale_for_mjcf[0]}-{mesh_scale_for_mjcf[1]}-{mesh_scale_for_mjcf[2]}"
                            # Value m
                            scale_suffix = scale_suffix.replace('-', 'm').replace('.', 'd')
                            unique_mesh_name = f"{base_mesh_name}{scale_suffix}"
                        else:
                            unique_mesh_name = base_mesh_name
                        
                        mesh_file_to_name[mesh_key] = unique_mesh_name
                        mesh_file_to_scale[mesh_key] = mesh_scale_for_mjcf
                        mesh_counter += 1
                        print(f"  Registered unique mesh: {unique_mesh_name} -> {mesh_filename} (scale: {mesh_scale_for_mjcf})")
                    else:
                        # +
                        existing_mesh_name = mesh_file_to_name[mesh_key]
                        print(f"  Reusing existing mesh: {existing_mesh_name} -> {mesh_filename} (scale: {mesh_scale_for_mjcf})")

                    # Node scale
                    mesh_names[node] = mesh_file_to_name[mesh_key]
                    node_to_mesh_scale_key[node] = mesh_key

            # Node colliders list process node.colliders
            for node in self.all_nodes():
                # Skip nodes with "Hide Mesh" enabled
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                # Skip base_link node (it's always skipped in MJCF export)
                if node.name() == 'base_link':
                    continue

                # Skip base_link_sub if base_link_mjcf exists (base_link_sub is intermediate)
                if node.name() == 'base_link_sub' and base_link_mjcf_exists:
                    continue

                if hasattr(node, 'colliders') and node.colliders:
                    for collider in node.colliders:
                        if collider.get('type') == 'mesh' and collider.get('mesh'):
                            collider_mesh_path = collider['mesh']
                            
                            # If
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
                                    
                                    # If process if .dae
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
                                    
                                    # Confirm TODO
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
                                    
                                    # Generate OBJ
                                    collider_filename = os.path.basename(collider_mesh_path)
                                    collider_filename = os.path.splitext(collider_filename)[0] + '.obj'
                                    
                                    # Check
                                    # Mesh_file_to_name mesh_filename scale_tuple scale_tuple
                                    found_mesh_name = None
                                    for mesh_key, mesh_name in mesh_file_to_name.items():
                                        if isinstance(mesh_key, tuple) and mesh_key[0] == collider_filename:
                                            found_mesh_name = mesh_name
                                            break
                                        elif mesh_key == collider_filename:
                                            found_mesh_name = mesh_name
                                            break
                                    
                                    if found_mesh_name:
                                        # :
                                        collider['_mesh_name'] = found_mesh_name
                                        print(f"  Reusing visual mesh for collider in node '{node.name()}': {found_mesh_name} ({collider_filename})")
                                    elif collider_filename in collider_file_to_name:
                                        # Existing name :
                                        collider['_mesh_name'] = collider_file_to_name[collider_filename]
                                        print(f"  Reusing collider mesh in node '{node.name()}': {collider['_mesh_name']} ({collider_filename})")
                                    else:
                                        # File :
                                        collider_dest_path = os.path.join(assets_dir, collider_filename)
                                        collider_mesh.export(collider_dest_path, file_type='obj')
                                        print(f"Converted collider mesh: {collider_mesh_path} -> {collider_filename} ({num_faces} faces)")
                                        
                                        # Generate TODO
                                        collider_mesh_name = os.path.splitext(collider_filename)[0]
                                        collider_file_to_name[collider_filename] = collider_mesh_name
                                        collider['_mesh_name'] = collider_mesh_name
                                        print(f"  Registered unique collider in node '{node.name()}': {collider_mesh_name} -> {collider_filename}")
                                        
                                        # <asset> mesh asset
                                        # Visual mesh mesh
                                        # _write_mjcf_geom visual mesh
                                    
                                except Exception as e:
                                    print(f"Warning: Could not process collider mesh '{collider_mesh_path}' in node '{node.name()}': {e}")
                                    continue
                            else:
                                print(f"Warning: Collider mesh file not found in node '{node.name()}': {collider_source_path}")

            # List
            created_joints = []

            # Determine actual root node for MJCF (skip base_link, base_link_sub as needed)
            root_node, rename_to_base_link = self._determine_mjcf_root_node()
            if not root_node:
                raise ValueError("Could not determine MJCF root node. Please check the node structure.")

            # Create 1 dir_name xml 1. .xml
            robot_file_path = os.path.join(mjcf_dir, f"{dir_name}.xml")
            robot_file_basename = os.path.basename(robot_file_path)  # NOTE
            self._write_mjcf_robot_file(robot_file_path, dir_name, root_node, mesh_names, node_to_mesh, created_joints, mesh_file_to_name, mesh_file_to_scale, collider_file_to_name, node_to_mesh_scale_key, fix_base_to_ground, rename_to_base_link)

            # Compute 2 model 2. z
            model_z_height = self._calculate_model_z_height(root_node, node_to_mesh)
            
            # Create 3 scene xml include 3. scene.xml
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
        """MuJoCotext(text、text、text)"""
        import re
        # Todo
        name = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', name)
        # Todo
        name = name.replace(' ', '_')
        # Main mujoco mujoco
        muoco_reserved = ['main', 'default', 'world', 'body', 'joint', 'geom', 'site', 'sensor', 'actuator', 'equality', 'tendon', 'contact', 'asset', 'option', 'compiler', 'visual', 'statistic']
        if name.lower() in muoco_reserved:
            name = f"{name}_obj"
        # If
        if not name:
            name = "robot"
        return name

    def _determine_mjcf_root_node(self):
        """Determine the actual root node for MJCF export.

        Rules:
        1. base_link is ignored, use first connected child
        2. If base_link → base_link_sub → base_link_mjcf: use base_link_mjcf, rename to "base_link"
        3. If base_link → base_link_sub → (other): use base_link_sub, rename to "base_link"
        4. If base_link → (not base_link_sub): use that node, keep original name

        Returns:
            tuple: (root_node, rename_to_base_link: bool)
                   root_node: The actual node to use as MJCF root
                   rename_to_base_link: True if root should be renamed to "base_link"
        """
        base_link = self.get_node_by_name('base_link')
        if not base_link:
            print("Warning: No base_link found")
            return None, False

        # Get first child of base_link
        first_child = None
        for output_port in base_link.output_ports():
            for connected_port in output_port.connected_ports():
                first_child = connected_port.node()
                break
            if first_child:
                break

        if not first_child:
            print("Warning: base_link has no children")
            return None, False

        # Check if first child is base_link_sub
        if first_child.name() == 'base_link_sub':
            # Look for base_link_mjcf in base_link_sub's children
            base_link_mjcf = None
            for output_port in first_child.output_ports():
                for connected_port in output_port.connected_ports():
                    child = connected_port.node()
                    if child.name() == 'base_link_mjcf':
                        base_link_mjcf = child
                        break
                if base_link_mjcf:
                    break

            if base_link_mjcf:
                # Rule 2: base_link → base_link_sub → base_link_mjcf
                print(f"MJCF Root: Using base_link_mjcf (renamed to base_link)")
                return base_link_mjcf, True
            else:
                # Rule 3: base_link → base_link_sub → (other)
                print(f"MJCF Root: Using base_link_sub (renamed to base_link)")
                return first_child, True
        else:
            # Rule 4: base_link → (not base_link_sub)
            print(f"MJCF Root: Using {first_child.name()} (keeping original name)")
            return first_child, False

    def collect_closed_loop_joints_from_nodes(self):
        """textnodetext"""
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

        # Update closed_loop_joints
        self.closed_loop_joints = collected_joints
        print(f"Collected {len(collected_joints)} closed-loop joint(s) from nodes")

        return collected_joints

    def _write_mjcf_robot_file(self, file_path, model_name, base_node, mesh_names, node_to_mesh, created_joints, mesh_file_to_name, mesh_file_to_scale, collider_file_to_name, node_to_mesh_scale_key, fix_base_to_ground=False, rename_to_base_link=False):
        """text(alltext)

        Args:
            fix_base_to_ground: Truetext、base_linktext<freejoint>textremovetext
            rename_to_base_link: True if root node should be renamed to "base_link"
        """
        # Check if base_link_mjcf exists (for skipping intermediate nodes)
        base_link_mjcf_exists = self.get_node_by_name('base_link_mjcf') is not None

        with open(file_path, 'w') as f:
            # Todo
            sanitized_model_name = self._sanitize_name(model_name)
            # Todo
            f.write(f'<mujoco model="{sanitized_model_name}">\n')

            # Compiler
            f.write('  <compiler angle="radian" meshdir="assets" autolimits="true" />\n\n')

            # Option
            f.write('  <option cone="elliptic" impratio="100" />\n\n')

            # <default> main class default
            f.write('  <default>\n')
            f.write('    <!-- textset -->\n')
            f.write('    <joint damping="0.1" armature="0.01" frictionloss="0.2"/>\n')
            f.write('    <motor ctrlrange="-23.7 23.7"/>\n')
            f.write('    <geom friction="0.4" margin="0.001" condim="3"/>\n')
            
            # Default class
            f.write('    <!-- textーtext：group=0 -->\n')
            f.write('    <default class="collision">\n')
            f.write('      <geom group="0"/>\n')
            f.write('    </default>\n')
            
            # Default class
            f.write('    <!-- text：group=1(text) -->\n')
            f.write('    <default class="visual">\n')
            f.write('      <geom contype="0" conaffinity="0" group="1"/>\n')
            f.write('    </default>\n')
            
            f.write('  </default>\n\n')

            # Mesh material asset
            f.write('  <asset>\n')
            f.write('    <material name="metal" rgba=".9 .95 .95 1" />\n')
            f.write('    <material name="black" rgba="0 0 0 1" />\n')
            f.write('    <material name="white" rgba="1 1 1 1" />\n')
            f.write('    <material name="gray" rgba="0.671705 0.692426 0.774270 1" />\n\n')

            # Mesh_file_to_name scale
            processed_mesh_keys = set()  # (mesh_filename, scale_tuple)
            used_mesh_names = set()  # mesh（）
            processed_collider_meshes = set()  # NOTE
            used_collider_names = set()  # mesh
            
            for node in self.all_nodes():
                # Skip nodes with "Hide Mesh" enabled
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                # Skip base_link node (it's always skipped in MJCF export)
                if node.name() == 'base_link':
                    continue

                # Skip base_link_sub if base_link_mjcf exists (base_link_sub is intermediate)
                if node.name() == 'base_link_sub' and base_link_mjcf_exists:
                    continue

                if node in node_to_mesh and node in node_to_mesh_scale_key:
                    mesh_key = node_to_mesh_scale_key[node]
                    if mesh_key not in processed_mesh_keys:
                        processed_mesh_keys.add(mesh_key)
                        mesh_filename, scale_tuple = mesh_key
                        mesh_name = mesh_file_to_name.get(mesh_key, os.path.splitext(mesh_filename)[0])
                        mesh_scale = mesh_file_to_scale.get(mesh_key, [1.0, 1.0, 1.0])
                        
                        # Add mesh
                        unique_mesh_name = mesh_name
                        counter = 1
                        while unique_mesh_name in used_mesh_names:
                            unique_mesh_name = f"{mesh_name}_{counter}"
                            counter += 1
                        used_mesh_names.add(unique_mesh_name)
                        
                        # Confirm mesh_scale value :
                        node_name = node.name() if hasattr(node, 'name') else 'unknown'
                        if unique_mesh_name != mesh_name:
                            print(f"  ⚠ Mesh name '{mesh_name}' already exists, renamed to '{unique_mesh_name}'")
                        print(f"  Writing mesh '{unique_mesh_name}' for node '{node_name}': scale={mesh_scale}")
                        
                        # Update mesh_file_to_name
                        mesh_file_to_name[mesh_key] = unique_mesh_name
                        
                        # 1 1 1 scale
                        if mesh_scale != [1.0, 1.0, 1.0]:
                            scale_str = f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}"
                            f.write(f'    <mesh name="{unique_mesh_name}" scale="{scale_str}" file="{mesh_filename}" />\n')
                            print(f"    ✓ Added scale attribute: {scale_str}")
                        else:
                            f.write(f'    <mesh name="{unique_mesh_name}" file="{mesh_filename}" />\n')
                
                # Add collider_file_to_name
                if hasattr(node, '_collider_mesh_name'):
                    collider_mesh_name = node._collider_mesh_name
                    # Collider_file_to_name
                    collider_filename = None
                    for filename, name in collider_file_to_name.items():
                        if name == collider_mesh_name:
                            collider_filename = filename
                            break
                    
                    if collider_filename and collider_filename not in processed_collider_meshes:
                        processed_collider_meshes.add(collider_filename)
                        
                        # Mesh
                        unique_collider_name = collider_mesh_name
                        counter = 1
                        while unique_collider_name in used_collider_names or unique_collider_name in used_mesh_names:
                            unique_collider_name = f"{collider_mesh_name}_{counter}"
                            counter += 1
                        used_collider_names.add(unique_collider_name)
                        
                        if unique_collider_name != collider_mesh_name:
                            print(f"  ⚠ Collider mesh name '{collider_mesh_name}' already exists, renamed to '{unique_collider_name}'")
                            # Update collider_file_to_name
                            collider_file_to_name[collider_filename] = unique_collider_name
                            node._collider_mesh_name = unique_collider_name
                        
                        f.write(f'    <mesh name="{unique_collider_name}" file="{collider_filename}" />\n')
            
            # Update mesh_names mesh_file_to_name node mesh
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
                used_body_names = set()  # Set to ensure unique body names
                used_joint_names = set()  # joint
                self._write_mjcf_body(f, base_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=4, fix_base_to_ground=fix_base_to_ground, used_body_names=used_body_names, used_joint_names=used_joint_names, is_root=True, rename_to_base_link=rename_to_base_link)
            f.write('  </worldbody>\n\n')

            # Equality constraints
            # Create TODO
            nodes_map = {node.name(): node for node in self.all_nodes()}
            self._write_mjcf_equality_constraints(f, nodes_map)

            # actuator (position control)
            if created_joints:
                f.write('  <actuator>\n')
                for joint_info in created_joints:
                    joint_name = joint_info['joint_name']
                    actuator_name = joint_info['motor_name'].replace('_motor', '_actuator')
                    
                    # Todo
                    # Kp stiffness kp:
                    kp = joint_info.get('kp', 100.0)  # NOTE
                    
                    # Ctrlrange joint <compiler angle radian > radians output ctrlrange: compiler
                    if joint_info.get('range_values'):
                        lower, upper = joint_info['range_values']
                        ctrlrange = f"{lower} {upper}"
                    else:
                        # ±π radians
                        ctrlrange = "-3.14159 3.14159"
                    
                    # Gear 1 1 1:1
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
        Calculate total Z-axis height (using cumulative node transforms and mesh bounding boxes)
        
        Returns:
            float: Total Z-axis height (meters)
        """
        try:
            import trimesh
            import numpy as np
            
            z_min = float('inf')
            z_max = float('-inf')
            
            # Node
            visited_nodes = set()

            def traverse_node(node, parent_coords=[0, 0, 0]):
                nonlocal z_min, z_max
                if node in visited_nodes:
                    return
                visited_nodes.add(node)
                
                # Compute node
                if isinstance(node, BaseLinkNode):
                    node_coords = [0, 0, 0]
                else:
                    # Get TODO
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
                
                # Get TODO
                if node in node_to_mesh:
                    # Node stl_file node.stl_file
                    if hasattr(node, 'stl_file') and node.stl_file and os.path.exists(node.stl_file):
                        try:
                            mesh = trimesh.load(node.stl_file)
                            if hasattr(mesh, 'bounds'):
                                mesh_bounds = mesh.bounds
                                # Mesh z
                                mesh_z_min = mesh_bounds[0][2]
                                mesh_z_max = mesh_bounds[1][2]
                                # Z
                                global_z_min = node_coords[2] + mesh_z_min
                                global_z_max = node_coords[2] + mesh_z_max
                                z_min = min(z_min, global_z_min)
                                z_max = max(z_max, global_z_max)
                        except Exception as e:
                            print(f"Warning: Could not load mesh for z-height calculation: {node.stl_file}, error: {e}")
                            # Mesh node coords
                            z_min = min(z_min, node_coords[2])
                            z_max = max(z_max, node_coords[2])
                    else:
                        # Node coords
                        z_min = min(z_min, node_coords[2])
                        z_max = max(z_max, node_coords[2])
                else:
                    # Mesh node coords
                    z_min = min(z_min, node_coords[2])
                    z_max = max(z_max, node_coords[2])
                
                # Process
                for port in node.output_ports():
                    for connected_port in port.connected_ports():
                        child_node = connected_port.node()
                        traverse_node(child_node, node_coords)
            
            # Start base_link
            if base_node:
                traverse_node(base_node)
            
            # Compute z
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
        """scene.xmltext(textinclude)
        
        Args:
            file_path: text
            robot_file_basename: text(basenametext)
            model_z_height: modeltextZ axistotal length(text)
            base_link_height: base_linktextheight(m)。NonetextSettingstext
            fix_base_to_ground: Truetext、base_linktextworldtextequality weldtextadd
        """
        with open(file_path, 'w') as f:
            # Todo
            sanitized_robot_name = self._sanitize_name(robot_file_basename.replace('.xml', ''))
            f.write(f'<mujoco model="{sanitized_robot_name} scene">\n')

            # Set camera base_link_height
            if base_link_height is None:
                # Get Settings Settings
                if hasattr(self, 'graph') and hasattr(self.graph, 'default_base_link_height'):
                    camera_center_z = self.graph.default_base_link_height
                else:
                    camera_center_z = DEFAULT_BASE_LINK_HEIGHT
            else:
                camera_center_z = base_link_height
            print(f"Setting camera center to z={camera_center_z:.6f} m (using base_link_height)")

            # Basename robot
            f.write(f'  <include file="{robot_file_basename}"/>\n\n')

            # Base_link world
            # Note base_link note: body
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
        """defaults.xmltext"""
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

    def _write_mjcf_body_file(self, file_path, base_node, mesh_names, node_to_mesh, created_joints, rename_to_base_link=False):
        """body.xmltext"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <worldbody>\n')

            if base_node:
                visited_nodes = set()
                self._write_mjcf_body(f, base_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=4, fix_base_to_ground=False, is_root=True, rename_to_base_link=rename_to_base_link)

            f.write('  </worldbody>\n')
            f.write('</mujoco>\n')
        print(f"Created body file: {file_path}")

    def _write_mjcf_actuators(self, file_path, created_joints):
        """actuators.xmltext"""
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

                # Output position
                # Kp stiffness value kp: stiffness
                # Kv damping value kv: damping
                # Forcerange effort value forcerange: effort - +
                # Forcelimited true forcelimited:
                # Set gear 1 1 gear: 1:1
                kp_str = format_float_no_exp(stiffness)
                kv_str = format_float_no_exp(damping)
                forcerange = f"-{format_float_no_exp(effort)} {format_float_no_exp(effort)}"

                f.write(f'    <position name="{motor_name}" joint="{joint_name}" gear="1" kp="{kp_str}" kv="{kv_str}" forcerange="{forcerange}" forcelimited="true" />\n')

            f.write('  </actuator>\n')
            f.write('</mujoco>\n')
        print(f"Created actuators file: {file_path}")

    def _write_mjcf_equality_constraints(self, file, nodes_map):
        """textjointtextMJCFtextequalitytext

        Args:
            file: text
            nodes_map: {link_name: node}text
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

            # Todo
            parent_sanitized = self._sanitize_name(parent_link)
            child_sanitized = self._sanitize_name(child_link)

            if original_type == 'ball':
                # Connect ball joint <connect> output
                # Anchor coords anchor:
                anchor_str = f"{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}"
                file.write(f'    <connect body1="{parent_sanitized}" body2="{child_sanitized}" anchor="{anchor_str}"/>\n')
                print(f"  Added ball joint constraint: {joint_name} ({parent_link} <-> {child_link})")

            elif original_type == 'gearbox':
                # Gearbox joint <joint> output joint 2
                gearbox_ratio = joint_data.get('gearbox_ratio', 1.0)
                gearbox_reference_body = joint_data.get('gearbox_reference_body')

                if gearbox_reference_body:
                    # Todo
                    ref_sanitized = self._sanitize_name(gearbox_reference_body)
                    joint1_name = f"{ref_sanitized}_joint"
                    joint2_name = f"{child_sanitized}_joint"

                    # Polycoef offset ratio - polycoef:
                    file.write(f'    <joint joint1="{joint1_name}" joint2="{joint2_name}" polycoef="0 {gearbox_ratio}"/>\n')
                    print(f"  Added gearbox joint constraint: {joint_name} (ratio: {gearbox_ratio})")
                else:
                    print(f"  Warning: gearbox joint '{joint_name}' missing reference_body, skipping")

            elif original_type == 'screw':
                # Screw joint <joint> output rotate
                # Add screw joint if TODO:
                print(f"  Warning: screw joint '{joint_name}' not fully implemented, skipping")

        file.write('  </equality>\n\n')

    def _write_mjcf_sensors(self, file_path):
        """sensors.xmltext"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <sensor>\n')
            f.write('    <!-- Add sensors here if needed -->\n')
            f.write('  </sensor>\n')
            f.write('</mujoco>\n')
        print(f"Created sensors file: {file_path}")

    def _write_mjcf_materials(self, file_path, node_to_mesh, mesh_names):
        """assets/materials.xmltext(meshtext)"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <asset>\n')

            # Todo
            for node in self.all_nodes():
                if node in node_to_mesh and node in mesh_names:
                    mesh_path = node_to_mesh[node]
                    # Meshdir assets/meshes
                    mesh_filename = os.path.basename(mesh_path)
                    mesh_name = mesh_names[node]

                    # Get Mesh 1 1 1 Mesh scale
                    mesh_scale = getattr(node, 'mesh_scale', [1.0, 1.0, 1.0])

                    # 1 1 1 scale
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

        MuJoCotext、<body>text<geom>text。
        - group 0: text
        - group 1: textmesh(Massless Decorationtext)
        - group 3: textー(text)
        """
        # Skip nodes with "Hide Mesh" enabled - safety check
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            print(f"Skipping geom output for node with hide_mesh=True: {node.name()}")
            return

        # Group 1 geom
        file.write(f'{indent_str}  <!-- text(displaytext) -->\n')

        # Get visual_origin pos quat
        pos_attr = ""
        quat_attr = ""
        if hasattr(node, 'visual_origin') and node.visual_origin:
            xyz = node.visual_origin.get('xyz', [0.0, 0.0, 0.0])
            rpy = node.visual_origin.get('rpy', [0.0, 0.0, 0.0])
            
            print(f"[MJCF_EXPORT_DEBUG] Node '{node.name()}' visual_origin:")
            print(f"  xyz: {xyz}")
            print(f"  rpy (rad): {rpy}")
            
            # Add XYZ position XYZ pos
            if xyz != [0.0, 0.0, 0.0]:
                pos_attr = f' pos="{xyz[0]} {xyz[1]} {xyz[2]}"'
                print(f"  → pos_attr SET: {pos_attr}")
            else:
                print(f"  → pos_attr SKIPPED (xyz is zero)")
            
            # Rpy quat rpy
            if rpy != [0.0, 0.0, 0.0]:
                # Rpy quat transform rpy quat
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
        
        # Massless decoration if massless decoration
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return
        
        # Get colliders list
        colliders = []
        if hasattr(node, 'colliders') and node.colliders:
            colliders = node.colliders

        # Write each enabled collider
        if colliders:
            file.write(f'{indent_str}  <!-- textー(text) -->\n')
        
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
                    # Export_mjcf _mesh_name
                    # _mesh_name visual link2 vs link2_0 mesh :
                    # Visual visual mesh asset
                    if '_mesh_name' in collider and collider['_mesh_name']:
                        collider_mesh_name = collider['_mesh_name']
                        # Visual visual mesh
                        # Visual mesh mesh _mesh_name asset
                        if collider_mesh_name == mesh_name:
                            file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{collider_mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                        else:
                            # Visual mesh
                            file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                    else:
                        # Node _collider_mesh_name : node._collider_mesh_name
                        if hasattr(node, '_collider_mesh_name') and node._collider_mesh_name:
                            collider_mesh_name = node._collider_mesh_name
                            # Visual visual mesh
                            if collider_mesh_name == mesh_name:
                                file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{collider_mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                            else:
                                # Visual mesh
                                file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                        else:
                            # Collider _mesh_name visual mesh
                            # Asset mesh
                            file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')
                else:
                    # Default: visual and collision use same mesh
                    file.write(f'{indent_str}  <geom class="collision" type="mesh" mesh="{mesh_name}" pos="{collider_pos_str}"{collider_quat_attr} group="3"/>\n')

    def _calculate_model_lowest_point(self, base_node, visited_nodes=None):
        """modeltext(textztransforms)text
        
        Args:
            base_node: textーtextnode
            visited_nodes: textnodetextset
            
        Returns:
            float: modeltextztransforms(textーtexttransformstext)
        """
        if visited_nodes is None:
            visited_nodes = set()
        
        min_z = 0.0  # Default value
        
        # Todo
        def traverse_nodes(node, current_z=0.0):
            nonlocal min_z
            
            if node in visited_nodes:
                return
            visited_nodes.add(node)
            
            # Get node
            node_min_z = self._get_node_lowest_point(node)
            total_min_z = current_z + node_min_z
            
            if total_min_z < min_z:
                min_z = total_min_z
            
            # Process
            for output_port in node.output_ports():
                for connected_port in output_port.connected_ports():
                    child_node = connected_port.node()
                    
                    # Get z
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
        """nodetext(textーtextmesh)textget
        
        Args:
            node: textnode
            
        Returns:
            float: nodetextーtexttransformstextztransforms
        """
        min_z = 0.0
        
        # Todo
        if hasattr(node, 'colliders') and node.colliders:
            for collider in node.colliders:
                if not collider.get('enabled', False):
                    continue
                
                collider_type = collider.get('type')
                position = collider.get('position', [0, 0, 0])
                
                if collider_type == 'primitive' and 'data' in collider:
                    data = collider['data']
                    prim_type = data.get('type', 'box')
                    
                    # Compute TODO
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
        
        # Todo
        # Mesh
        if min_z == 0.0:
            min_z = -0.1  # NOTE
        
        return min_z

    def _write_mjcf_body(self, file, node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=2, joint_info=None, fix_base_to_ground=False, used_body_names=None, used_joint_names=None, is_root=False, rename_to_base_link=False):
        """MJCF bodytext

        Args:
            fix_base_to_ground: Truetext、base_linktext<freejoint>textremovetext
            used_body_names: textbody nametextset(uniquenessensuretext)
            used_joint_names: textjointtextset(uniquenessensuretext)
            is_root: True if this node is the MJCF root (gets freejoint or fixed)
            rename_to_base_link: True if root should be renamed to "base_link"
        """
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Used_body_names
        if used_body_names is None:
            used_body_names = set()

        # Used_joint_names
        if used_joint_names is None:
            used_joint_names = set()

        # Massless skip massless decoration
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        # Skip nodes with "Hide Mesh" enabled
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            print(f"Skipping node with hide_mesh=True: {node.name()}")
            return

        indent_str = ' ' * indent

        # Root node handling (replaces old base_link check)
        if is_root:
            # Use "base_link" as name if rename_to_base_link is True, otherwise use node's actual name
            if rename_to_base_link:
                sanitized_name = self._sanitize_name("base_link")
            else:
                sanitized_name = self._sanitize_name(node.name())
            
            # Add body
            unique_name = sanitized_name
            counter = 1
            while unique_name in used_body_names:
                unique_name = f"{sanitized_name}_{counter}"
                counter += 1
            used_body_names.add(unique_name)
            
            if unique_name != sanitized_name:
                print(f"  ⚠ Body name '{sanitized_name}' already exists, renamed to '{unique_name}'")

            # Check
            is_all_defaults = self._is_base_link_at_defaults(node)

            # Note mesh note: scale _reversed
            # Start base_link z
            if fix_base_to_ground:
                # Fix base to ground model 0 fix base ground: z
                min_z = self._calculate_model_lowest_point(node, visited_nodes.copy())
                z_pos = max(0, -min_z)  # z=0
                print(f"Fix Base to Ground: model lowest point = {min_z:.6f}, base z_pos = {z_pos:.6f}")
            else:
                # : base_link_height
                z_pos = getattr(self, 'base_link_height', getattr(self, 'default_base_link_height', DEFAULT_BASE_LINK_HEIGHT))
            file.write(f'{indent_str}<body name="{unique_name}" pos="0 0 {z_pos}">\n')

            # Freejoint base_link
            # Fix_base_to_ground true if <freejoint> true freejoint
            if not fix_base_to_ground:
                file.write(f'{indent_str}  <freejoint />\n')

            # Moving body freejoint inertial check
            has_inertial = False
            if hasattr(node, 'mass_value') and node.mass_value > 0:
                has_inertial = True

            # If geom
            if not is_all_defaults:
                # Mesh
                if node in mesh_names:
                    mesh_name = mesh_names[node]
                    color_str = "0.8 0.8 0.8 1.0"
                    if hasattr(node, 'node_color') and node.node_color:
                        r, g, b = node.node_color[:3]
                        color_str = f"{r} {g} {b} 1.0"

                    # Todo
                    self._write_mjcf_geom(file, node, mesh_name, color_str, indent_str)

            # Base_link process
            for port in node.output_ports():
                for connected_port in port.connected_ports():
                    child_node = connected_port.node()

                    # Massless <geom class visual > output massless decoration geom
                    if hasattr(child_node, 'massless_decoration') and child_node.massless_decoration:
                        if child_node in mesh_names:
                            dec_mesh_name = mesh_names[child_node]
                            # Todo
                            dec_color_str = "0.8 0.8 0.8 1.0"
                            if hasattr(child_node, 'node_color') and child_node.node_color:
                                r, g, b = child_node.node_color[:3]
                                dec_color_str = f"{r} {g} {b} 1.0"

                            # Get position
                            port_index = list(node.output_ports()).index(port)
                            pos_str = "0 0 0"
                            if hasattr(node, 'points') and port_index < len(node.points):
                                point_data = node.points[port_index]
                                if 'xyz' in point_data:
                                    xyz = point_data['xyz']
                                    pos_str = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                            # Massless decoration <geom class visual > massless decoration geom
                            file.write(f'{indent_str}  <geom class="visual" type="mesh" mesh="{dec_mesh_name}" rgba="{dec_color_str}" pos="{pos_str}" group="2"/>\n')
                        continue

                    # Hide mesh check skip hide mesh
                    if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                        continue

                    port_index = list(node.output_ports()).index(port)
                    child_joint_info = self._get_joint_info(node, child_node, port_index, created_joints)
                    self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent + 2, child_joint_info, fix_base_to_ground, used_body_names, used_joint_names, is_root=False, rename_to_base_link=False)

            # Add moving body freejoint inertial
            if not has_inertial:
                # Set MuJoCo mjMINVAL mass 0 001 inertia 1e-6 MuJoCo mjMINVAL mass: 0.001 inertia:
                file.write(f'{indent_str}  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>\n')
                print(f"  Auto-added inertial to moving body (root) to avoid MuJoCo load error")

            # Close root body
            file.write(f'{indent_str}</body>\n')
            return

        # Name
        sanitized_name = self._sanitize_name(node.name())
        
        # Add body
        unique_name = sanitized_name
        counter = 1
        while unique_name in used_body_names:
            unique_name = f"{sanitized_name}_{counter}"
            counter += 1
        used_body_names.add(unique_name)
        
        if unique_name != sanitized_name:
            print(f"  ⚠ Body name '{sanitized_name}' already exists, renamed to '{unique_name}'")

        # Note mesh note: scale _reversed
        # Add joint_info body orientation
        pos_attr = f' pos="{joint_info["pos"]}"' if joint_info else ''

        # Body orientation joint_info rpy urdf joint origin rpy parent child body orientation: rpy urdf rpy
        # Body_angle
        orientation_attr = ""
        rpy_to_use = None
        
        # 1 joint_info rpy urdf joint origin rpy 1. rpy urdf rpy
        if joint_info and 'rpy' in joint_info:
            rpy_to_use = joint_info['rpy']
        # 2 body_angle body 2.
        elif hasattr(node, 'body_angle') and node.body_angle != [0.0, 0.0, 0.0]:
            rpy_to_use = node.body_angle
        
        # Generate RPY quaternion xyaxes RPY
        if rpy_to_use and rpy_to_use != [0.0, 0.0, 0.0]:
            # Quaternion
            quat = self._rpy_to_quat(rpy_to_use)
            quat_str = f"{format_float_no_exp(quat[0])} {format_float_no_exp(quat[1])} {format_float_no_exp(quat[2])} {format_float_no_exp(quat[3])}"
            orientation_attr = f' quat="{quat_str}"'

        file.write(f'{indent_str}<body name="{unique_name}"{pos_attr}{orientation_attr}>\n')

        # Joint output
        # Body pos parent offset joint pos 0 0 0 origin
        is_moving_body = False
        if joint_info:
            # Add joint
            original_joint_name = joint_info["name"]
            unique_joint_name = original_joint_name
            counter = 1
            while unique_joint_name in used_joint_names:
                unique_joint_name = f"{original_joint_name}_{counter}"
                counter += 1
            used_joint_names.add(unique_joint_name)
            
            if unique_joint_name != original_joint_name:
                print(f"  ⚠ Joint name '{original_joint_name}' already exists, renamed to '{unique_joint_name}'")
                # Update created_joints joint_name actuator
                joint_info["name"] = unique_joint_name
            
            # Remove range limited margin armature frictionloss damping stiffness ref output velocity MJCF MJCF
            joint_attrs = f'{joint_info["range"]}{joint_info["limited"]}{joint_info["margin"]}{joint_info["armature"]}{joint_info["frictionloss"]}{joint_info["damping"]}{joint_info["stiffness"]}{joint_info["ref"]}'
            file.write(f'{indent_str}  <joint name="{unique_joint_name}" type="{joint_info["type"]}" pos="0 0 0" axis="{joint_info["axis"]}"{joint_attrs} />\n')
            is_moving_body = True  # jointbodymoving body

        # Todo
        has_inertial = False
        if hasattr(node, 'mass_value') and node.mass_value > 0:
            # Set TODO
            MIN_MASS = 0.001  # 1g
            # Remove MIN_INERTIA - value MIN_INERTIA URDF

            mass = max(node.mass_value, MIN_MASS)

            # Process
            if hasattr(node, 'inertia') and isinstance(node.inertia, dict) and node.inertia:
                # Todo
                node_name = node.name()
                is_target = 'arm_lower' in node_name.lower()  # l_arm_lower / r_arm_lower

                if is_target:
                    print(f"\n{'='*80}")
                    print(f"MJCF Inertia Output Debug: {node_name}")
                    print(f"{'='*80}")
                    print(f"  Mass: {node.mass_value} (raw) -> {mass} (after MIN_MASS)")
                
                # Get Inertial Origin rotate Inertial Origin COM
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
                
                # Remove - value MIN_INERTIA URDF
                # Set if MuJoCo MuJoCo
                ZERO_THRESHOLD = 1e-12
                if abs(I_body[0, 0]) < ZERO_THRESHOLD:
                    I_body[0, 0] = ZERO_THRESHOLD
                if abs(I_body[1, 1]) < ZERO_THRESHOLD:
                    I_body[1, 1] = ZERO_THRESHOLD
                if abs(I_body[2, 2]) < ZERO_THRESHOLD:
                    I_body[2, 2] = ZERO_THRESHOLD
                
                # Ensure symmetry again after threshold application
                I_body = 0.5 * (I_body + I_body.T)
                
                # Zero_threshold
                # Change TODO
                Ixx = I_body[0, 0]
                Iyy = I_body[1, 1]
                Izz = I_body[2, 2]
                epsilon = 1e-8
                
                # Check
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
                
                # Update TODO
                I_body[0, 0] = Ixx
                I_body[1, 1] = Iyy
                I_body[2, 2] = Izz
                
                # Todo
                I_body = 0.5 * (I_body + I_body.T)
                
                # Todo
                final_Ixx = I_body[0, 0]
                final_Iyy = I_body[1, 1]
                final_Izz = I_body[2, 2]
                final_epsilon = 1e-10  # MuJoCo
                if not (final_Ixx + final_Iyy >= final_Izz - final_epsilon and
                        final_Iyy + final_Izz >= final_Ixx - final_epsilon and
                        final_Izz + final_Ixx >= final_Iyy - final_epsilon):
                    print(f"  ⚠ WARNING: Triangle inequality violated for {node_name} after all corrections!")
                    print(f"     Ixx={final_Ixx:.12e}, Iyy={final_Iyy:.12e}, Izz={final_Izz:.12e}")
                    print(f"     Ixx+Iyy={final_Ixx+final_Iyy:.12e} >= Izz={final_Izz:.12e}? {final_Ixx+final_Iyy >= final_Izz - final_epsilon}")
                    print(f"     Iyy+Izz={final_Iyy+final_Izz:.12e} >= Ixx={final_Ixx:.12e}? {final_Iyy+final_Izz >= final_Ixx - final_epsilon}")
                    print(f"     Izz+Ixx={final_Izz+final_Ixx:.12e} >= Iyy={final_Iyy:.12e}? {final_Izz+final_Ixx >= final_Iyy - final_epsilon}")
                    # Todo
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
                
                # Warning
                max_inertia = np.max(np.abs(np.diag(I_body)))
                if mass > 0 and max_inertia / mass > 10.0:  # （: mass=0.03, inertia=0.01）
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

        # Mesh
        if node in mesh_names:
            mesh_name = mesh_names[node]
            # Todo
            color_str = "0.8 0.8 0.8 1.0"
            if hasattr(node, 'node_color') and node.node_color:
                r, g, b = node.node_color[:3]
                color_str = f"{r} {g} {b} 1.0"

            # Todo
            self._write_mjcf_geom(file, node, mesh_name, color_str, indent_str)

        # Process
        for port_index, port in enumerate(node.output_ports()):
            for connected_port in port.connected_ports():
                child_node = connected_port.node()

                # Massless if <geom class visual > skip massless decoration geom
                if hasattr(child_node, 'massless_decoration') and child_node.massless_decoration:
                    if child_node in mesh_names:
                        dec_mesh_name = mesh_names[child_node]
                        # Todo
                        dec_color_str = "0.8 0.8 0.8 1.0"
                        if hasattr(child_node, 'node_color') and child_node.node_color:
                            r, g, b = child_node.node_color[:3]
                            dec_color_str = f"{r} {g} {b} 1.0"

                        # Get position
                        pos_str = "0 0 0"
                        if hasattr(node, 'points') and port_index < len(node.points):
                            point_data = node.points[port_index]
                            if 'xyz' in point_data:
                                xyz = point_data['xyz']
                                pos_str = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                        # Massless decoration <geom class visual > massless decoration geom
                        file.write(f'{indent_str}  <geom class="visual" type="mesh" mesh="{dec_mesh_name}" rgba="{dec_color_str}" pos="{pos_str}"/>\n')
                    continue

                # Hide mesh check skip hide mesh
                if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                    continue

                # Get port_index enumerate
                child_joint_info = self._get_joint_info(node, child_node, port_index, created_joints)

                # Output
                self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent + 2, child_joint_info, fix_base_to_ground, used_body_names, used_joint_names, is_root=False, rename_to_base_link=False)

        # Add moving body joint inertial
        if is_moving_body and not has_inertial:
            # Set MuJoCo mjMINVAL mass 0 001 inertia 1e-6 MuJoCo mjMINVAL mass: 0.001 inertia:
            file.write(f'{indent_str}  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>\n')
            print(f"  Auto-added inertial to moving body ({node.name()}) to avoid MuJoCo load error")

        # Body
        file.write(f'{indent_str}</body>\n')

    def _get_joint_info(self, parent_node, child_node, port_index, created_joints):
        """jointtextgettext"""
        joint_xyz = [0, 0, 0]
        joint_rpy = [0, 0, 0]
        joint_axis = [1, 0, 0]

        # Get RPY RPY
        # Use angle if available (UI-edited value), otherwise fallback to rpy
        if hasattr(parent_node, 'points') and port_index < len(parent_node.points):
            point_data = parent_node.points[port_index]
            joint_xyz = point_data.get('xyz', [0, 0, 0])
            joint_rpy = point_data.get('angle', point_data.get('rpy', [0, 0, 0]))

        # Get TODO
        if hasattr(child_node, 'rotation_axis'):
            if child_node.rotation_axis == 0:
                joint_axis = [1, 0, 0]
            elif child_node.rotation_axis == 1:
                joint_axis = [0, 1, 0]
            elif child_node.rotation_axis == 2:
                joint_axis = [0, 0, 1]

        # Todo
        joint_type = "hinge"
        if hasattr(child_node, 'rotation_axis') and child_node.rotation_axis == 3:
            joint_type = "fixed"

        # If none fixed none
        if joint_type == "fixed":
            return None

        # Generate joint name with axis suffix (roll/pitch/yaw based on rotation_axis)
        child_sanitized_name = self._sanitize_name(child_node.name())
        # Determine axis suffix based on rotation_axis
        axis_suffix = "_roll"  # Default: X axis (rotation_axis == 0)
        if hasattr(child_node, 'rotation_axis'):
            if child_node.rotation_axis == 0:
                axis_suffix = "_roll"    # X axis
            elif child_node.rotation_axis == 1:
                axis_suffix = "_pitch"   # Y axis
            elif child_node.rotation_axis == 2:
                axis_suffix = "_yaw"     # Z axis
            elif child_node.rotation_axis == 3:
                axis_suffix = "_fixed"   # Fixed (though this case returns None above)
        joint_name = f"{child_sanitized_name}{axis_suffix}"
        motor_name = f"{child_sanitized_name}_motor"

        # Min angle deg max angle deg min angle max angle rad
        # Mjcf <compiler angle radian > radians output mjcf compiler
        range_str = ""
        if hasattr(child_node, 'joint_lower') and hasattr(child_node, 'joint_upper'):
            lower = child_node.joint_lower  # Stored in radians
            upper = child_node.joint_upper  # Stored in radians
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

        # Limited true limited:
        limited_str = ' limited="true"'

        # Margin margin value margin:
        margin_str = ""
        if hasattr(child_node, 'joint_margin'):
            margin_str = f' margin="{format_float_no_exp(child_node.joint_margin)}"'

        # Armature armature value armature:
        armature_str = ""
        if hasattr(child_node, 'joint_armature'):
            armature_str = f' armature="{format_float_no_exp(child_node.joint_armature)}"'

        # Frictionloss frictionloss value frictionloss:
        frictionloss_str = ""
        if hasattr(child_node, 'joint_frictionloss'):
            frictionloss_str = f' frictionloss="{format_float_no_exp(child_node.joint_frictionloss)}"'

        # Damping damping value damping:
        damping_str = ""
        if hasattr(child_node, 'joint_damping'):
            damping_str = f' damping="{format_float_no_exp(child_node.joint_damping)}"'
        
        # Stiffness stiffness value mjcf stiffness: mjcf
        # Note joint stiffness actuator kp joint stiffness note:
        # Actuator kp
        stiffness_str = ""
        # if hasattr(child_node, 'joint_stiffness') and child_node.joint_stiffness > 0:
        #     stiffness_str = f' stiffness="{format_float_no_exp(child_node.joint_stiffness)}"'
        
        # Generate ref body_angle ref: /
        # Body_angle x_rad y_rad z_rad rotation_axis
        # Mjcf <compiler angle radian > ref radians output mjcf compiler
        ref_str = ""
        if hasattr(child_node, 'body_angle') and hasattr(child_node, 'rotation_axis'):
            body_angle = child_node.body_angle
            rotation_axis = child_node.rotation_axis
            
            # Get rotation_axis angle radians
            if rotation_axis in [0, 1, 2] and any(a != 0.0 for a in body_angle):
                ref_angle_rad = body_angle[rotation_axis]
                if abs(ref_angle_rad) > 1e-6:  # only
                    ref_str = f' ref="{format_float_no_exp(ref_angle_rad)}"'

        # Add list actuator
        joint_effort = getattr(child_node, 'joint_effort', 10.0)
        joint_stiffness = getattr(child_node, 'joint_stiffness', 100.0)
        joint_damping = getattr(child_node, 'joint_damping', 15.0)
        # Range ctrlrange
        # Mjcf <compiler angle radian > radians output mjcf compiler
        range_values = None
        if hasattr(child_node, 'joint_lower') and hasattr(child_node, 'joint_upper'):
            lower = child_node.joint_lower  # Stored in radians
            upper = child_node.joint_upper  # Stored in radians
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
            'rpy': joint_rpy,  # RPY（body orientation）
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
        textーtextmodeltextinertia tensortext。
        CustomNodeGraphtext。
        """
        try:
            print("\nCalculating inertia tensor for mirrored model...")
            print(f"Mass: {mass:.6f}")
            print(f"Center of Mass (before mirroring): {center_of_mass}")

            # Y
            mirrored_com = [center_of_mass[0], -center_of_mass[1], center_of_mass[2]]
            print(f"Center of Mass (after mirroring): {mirrored_com}")

            # Compute TODO
            # Utils py calculate_inertia_tensor utils.py
            inertia_tensor = calculate_inertia_tensor(
                poly_data, mass, mirrored_com, is_mirrored=True)

            print("\nMirrored model inertia tensor calculated successfully")
            return inertia_tensor

        except Exception as e:
            print(f"Error calculating mirrored inertia tensor: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Todo
def load_project(graph):
    """textload"""
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

        # Xml
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Get path meshes
        meshes_dir = None
        meshes_dir_elem = root.find("meshes_dir")
        if meshes_dir_elem is not None and meshes_dir_elem.text:
            meshes_dir = os.path.normpath(os.path.join(project_base_dir, meshes_dir_elem.text))
            if not os.path.exists(meshes_dir):
                # Select meshes
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

        # Current
        graph.clear_graph()

        # Load TODO
        success = graph.load_project(file_path)

        if success:
            print("Project loaded, resolving STL paths...")
            # Create list STL
            nodes_to_load_stl = []
            for node in graph.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    nodes_to_load_stl.append(node)
            
            total_stl_loads = len(nodes_to_load_stl)
            print(f"Total STL files to load: {total_stl_loads}")
            
            # Get TODO
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer and hasattr(graph.stl_viewer, 'progress_bar'):
                current_progress = graph.stl_viewer.progress_bar.value
                print(f"Current progress after node loading: {current_progress}%")
            else:
                current_progress = 100  # Default value
            
            # Path stl
            for stl_index, node in enumerate(nodes_to_load_stl, 1):
                if hasattr(node, 'stl_file') and node.stl_file:
                    try:
                        stl_path = node.stl_file
                        if not os.path.isabs(stl_path):
                            # Meshes
                            if meshes_dir:
                                abs_stl_path = os.path.normpath(os.path.join(meshes_dir, stl_path))
                                if os.path.exists(abs_stl_path):
                                    node.stl_file = abs_stl_path
                                    print(f"Found STL file in meshes dir for node {node.name()}: {abs_stl_path}")
                                    if graph.stl_viewer:
                                        # Update STL
                                        if total_stl_loads > 0 and hasattr(graph.stl_viewer, 'progress_bar'):
                                            # Stl
                                            # Current_progress 0% 0 stl
                                            stl_progress = int((stl_index / total_stl_loads) * current_progress)
                                            remaining_percent = current_progress - stl_progress
                                            graph.stl_viewer.progress_bar.setValue(max(0, remaining_percent))
                                            QtWidgets.QApplication.processEvents()
                                        graph.stl_viewer.load_stl_for_node(node, show_progress=False)  # display
                                    continue

                            # Todo
                            abs_stl_path = os.path.normpath(os.path.join(project_base_dir, stl_path))
                            if os.path.exists(abs_stl_path):
                                node.stl_file = abs_stl_path
                                print(f"Found STL file in project dir for node {node.name()}: {abs_stl_path}")
                                if graph.stl_viewer:
                                    # Update STL
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
                                    # Update STL
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
            
            # Hide process
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
                graph.stl_viewer.progress_bar.setValue(0)
                QtWidgets.QApplication.processEvents()
                from PySide6.QtCore import QTimer
                QTimer.singleShot(200, lambda: graph.stl_viewer.show_progress(False))

            # Position
            graph.recalculate_all_positions()
            
            # Apply node node STL 3D
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
                        
                        # Apply node STL
                        if has_stl_file and in_actors:
                            # Confirm node node_color node.node_color
                            if has_node_color:
                                rgba_values = node.node_color
                                # 0-1 rgba
                                rgba_values = [max(0.0, min(1.0, float(v))) for v in rgba_values[:4]]
                                
                                actor = graph.stl_viewer.stl_actors[node]
                                
                                # Check
                                mapper = actor.GetMapper()
                                has_scalars = False
                                if mapper and mapper.GetInput():
                                    polydata = mapper.GetInput()
                                    has_vertex_colors = polydata.GetPointData().GetScalars() is not None
                                    has_face_colors = polydata.GetCellData().GetScalars() is not None
                                    has_scalars = has_vertex_colors or has_face_colors
                                
                                if has_scalars:
                                    # Todo
                                    if len(rgba_values) >= 4:
                                        actor.GetProperty().SetOpacity(rgba_values[3])
                                    else:
                                        actor.GetProperty().SetOpacity(1.0)
                                    print(f"[DEBUG] Node '{node_name}' has vertex/face colors, only opacity applied: {rgba_values[3] if len(rgba_values) >= 4 else 1.0}")
                                else:
                                    # Apply color
                                    # Rgb 3
                                    actor.GetProperty().SetColor(*rgba_values[:3])
                                    # Alpha 4
                                    if len(rgba_values) >= 4:
                                        actor.GetProperty().SetOpacity(rgba_values[3])
                                    else:
                                        actor.GetProperty().SetOpacity(1.0)
                                    print(f"[DEBUG] Applied color to node '{node_name}': RGBA{rgba_values[:4]}")
                                    applied_count += 1
                            else:
                                # Apply TODO
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
                
                # 3d
                graph.stl_viewer.render_to_image()
                print(f"[DEBUG] Colors applied: {applied_count} nodes, {skipped_count} skipped")
                print("[DEBUG] Colors applied to 3D view after STL loading")

            # 3d hide_mesh
            if graph.stl_viewer:
                def reset_and_apply_hide():
                    graph.stl_viewer.reset_view_to_fit()
                    # Node hide_mesh
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
            # Baselinknode baselinknode
            if isinstance(node, BaseLinkNode):
                print("Cannot delete Base Link node")
                continue

            # Remove 3D STL
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
                if node in graph.stl_viewer.stl_actors:
                    actor = graph.stl_viewer.stl_actors[node]
                    graph.stl_viewer.renderer.RemoveActor(actor)
                    del graph.stl_viewer.stl_actors[node]
                    print(f"Removed STL mesh for node: {node.name()}")
                    # Update 3D
                    graph.stl_viewer.render_to_image()

            # Remove node
            graph.remove_node(node)
        print(f"Deleted {len(selected_nodes)} node(s)")
    else:
        print("No node selected for deletion")

def show_settings_dialog(graph, parent=None):
    """settextdisplay"""
    dialog = SettingsDialog(graph, parent)
    result = dialog.exec_()

    # Set update
    if result == QtWidgets.QDialog.Accepted:
        if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
            stl_viewer = graph.stl_viewer
            # Enable if
            if hasattr(stl_viewer, 'collider_display_enabled') and stl_viewer.collider_display_enabled:
                print("Settings updated - refreshing collider display...")
                stl_viewer.show_all_colliders()
                stl_viewer.render_to_image()
                print("Collider display refreshed with new collision color")

def open_importer_window(graph):
    """Model Importerwindowtext"""
    # Window graph
    if not hasattr(graph, 'importer_window') or graph.importer_window is None:
        graph.importer_window = ImporterWindow(graph)

    # Window
    graph.importer_window.show()
    graph.importer_window.raise_()
    graph.importer_window.activateWindow()

def cleanup_and_exit():
    """textーtextーtextend"""
    print("Cleaning up application resources...")
    try:
        # Todo
        if 'graph' in globals():
            try:
                graph.cleanup()
            except Exception as e:
                print(f"Error cleaning up graph: {str(e)}")

        # Stl
        if 'stl_viewer' in globals():
            try:
                stl_viewer.cleanup()
            except Exception as e:
                print(f"Error cleaning up STL viewer: {str(e)}")

        # All window
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
    """Ctrl+Ctext"""
    print("\nCtrl+C detected, closing application...")

    def do_shutdown():
        """texthandle(Qteventtextーtext)"""
        try:
            app = QtWidgets.QApplication.instance()
            if app:
                # All window
                for window in QtWidgets.QApplication.topLevelWidgets():
                    try:
                        if window and window.isVisible():
                            window.close()
                            window.deleteLater()
                    except Exception as e:
                        print(f"Error closing window: {str(e)}")

                # Quit TODO
                QtCore.QTimer.singleShot(100, app.quit)

                # 100ms
                QtCore.QTimer.singleShot(200, lambda: sys.exit(0))
        except Exception as e:
            print(f"Error during application shutdown: {str(e)}")
            sys.exit(0)

    # Run Qt
    try:
        if QtWidgets.QApplication.instance():
            QtCore.QTimer.singleShot(0, do_shutdown)
        else:
            sys.exit(0)
    except Exception:
        sys.exit(0)

def center_window_top_left(window):
    """windowtext"""
    window.move(0, 0)


if __name__ == '__main__':
    try:
        # Set Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app = QtWidgets.QApplication(sys.argv)
        setup_dark_theme(app, theme='assembler')

        # Todo
        app.aboutToQuit.connect(cleanup_and_exit)

        # Utils
        timer = setup_signal_processing_timer(app)

        # Create TODO
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("URDF Kitchen - Assembler v0.1.0 -")
        main_window.resize(1200, 600)

        # Set TODO
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Set create STL
        stl_viewer = STLViewerWidget(central_widget)
        stl_viewer.setMinimumWidth(100)  # 3Dmin width100px
        graph = CustomNodeGraph(stl_viewer)
        stl_viewer.graph = graph  # STL
        graph.setup_custom_view()

        # Create base_link
        base_node = graph.create_base_link()

        # Set TODO
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(145)
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # Set TODO
        name_label = QtWidgets.QLabel("Name:")
        left_layout.addWidget(name_label)
        name_input = QtWidgets.QLineEdit("robot_x")
        name_input.setFixedWidth(120)
        name_input.setStyleSheet("QLineEdit { padding-left: 3px; padding-top: 0px; padding-bottom: 0px; }")
        left_layout.addWidget(name_input)

        # Connect graph
        name_input.textChanged.connect(graph.update_robot_name)

        # Set button create
        buttons = {
            "--spacer1--": None,  # Dummy key for spacer
            "Import XMLs": None,
            "Import MODEL": None,
            "--spacer2--": None,  # Dummy key for spacer
            "Add Node": None,
            "Delete Node": None,
            "Check Inertia": None,
            "Build r_ from l_": None,
            "Recalc Positions": None,
            "--spacer3--": None,  # Dummy key for spacer
            "Load Project": None,
            "Save Project": None,
            "--spacer4--": None,  # Dummy key for spacer
            "Export URDF": None,
            "Export for Unity": None,
            "Export MJCF": None,
            "--spacer5--": None,  # Dummy key for spacer
            "open urdf-loaders": None,
            "Settings": None
        }

        # Unified button style (global constants)
        button_style = UNIFIED_BUTTON_STYLE

        for button_text in buttons.keys():
            if button_text.startswith("--spacer"):
                # Add TODO
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(1)  # spacingheight1
                left_layout.addWidget(spacer)
            else:
                # Add button
                button = QtWidgets.QPushButton(button_text)
                button.setFixedWidth(120)
                button.setStyleSheet(button_style)  # Apply consistent style
                left_layout.addWidget(button)
                buttons[button_text] = button

        left_layout.addStretch()

        # Button
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

        # Set 3 3D
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(graph.widget)
        splitter.addWidget(stl_viewer)
        # : 200px 700px 300px
        splitter.setSizes([200, 700, 300])

        # Set TODO
        main_layout.addWidget(splitter)

        # Set TODO
        main_window.setCentralWidget(central_widget)

        # Todo
        graph.name_input = name_input

        # Window
        center_window_top_left(main_window)
        main_window.show()

        print("Application started. Double-click on a node to open the inspector.")
        print("Click 'Add Node' button to add new nodes.")
        print("Select a node and click 'Delete Node' to remove it.")
        print("Use 'Save' and 'Load' buttons to save and load your project.")
        print("Press Ctrl+C in the terminal to close all windows and exit.")

        # Set timer
        timer = QtCore.QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)
        
        # Run TODO
        sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        cleanup_and_exit()
        sys.exit(1)
