"""
File Name: urdf_kitchen_utils.py
Description: Shared utilities for URDF Kitchen tools (Assembler, MeshSourcer, PartsEditor).

Author      : Ninagawa123
Created On  : Dec 28, 2025
Update.     : Jan 18, 2026
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
pip install xacrodoc
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import os
import math
import xml.etree.ElementTree as ET
import subprocess


# ============================================================================
# COLOR DIALOG UTILITIES
# ============================================================================

class CustomColorDialog(QtWidgets.QColorDialog):
    """Color dialog with custom color box selection feature.

    This class extends QColorDialog to add a selection border on the custom
    color box, allowing users to visually confirm which custom color slot
    they are saving colors to.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_custom_color_index = 0  # Index of selected custom color
        self.custom_color_well_array = None  # QWellArray for custom colors
        self._setup_done = False  # Setup completion flag

    def showEvent(self, event):
        """Setup custom color box when dialog is shown"""
        super().showEvent(event)
        if not self._setup_done:
            # Setup after dialog is fully displayed
            QtCore.QTimer.singleShot(300, self._setup_custom_color_boxes)

    def _setup_custom_color_boxes(self):
        """Find custom color QWellArray and set event filter"""
        def find_custom_well_array(widget, depth=0):
            """Recursively search for custom color QWellArray"""
            class_name = widget.metaObject().className()

            # QtPrivate::QWellArray with size 224x48 is the custom color box
            if class_name == 'QtPrivate::QWellArray':
                size = widget.size()
                if size.height() == 48:  # Custom color box height is 48
                    self.custom_color_well_array = widget
                    # Install event filter
                    widget.installEventFilter(self)
                    return True

            # Search child widgets
            for child in widget.children():
                if isinstance(child, QtWidgets.QWidget):
                    if find_custom_well_array(child, depth + 1):
                        return True
            return False

        if find_custom_well_array(self):
            self._setup_done = True
            # Show selection border on first cell initially
            self._draw_selection_border()
            # Find and override "Add to Custom Colors" button
            self._setup_add_button()
        else:
            # Retry with delay if not found
            if not self._setup_done:
                QtCore.QTimer.singleShot(500, self._setup_custom_color_boxes)

    def _setup_add_button(self):
        """Find Add to Custom Colors button and override click handler"""
        # Search all QPushButtons for the appropriate button
        buttons = self.findChildren(QtWidgets.QPushButton)
        for button in buttons:
            # Identify Add to Custom Colors button by text or position
            # Usually the button below the custom color QWellArray
            if button.text() or True:  # Check all buttons regardless of text
                # Check if button is near custom color QWellArray
                if self.custom_color_well_array:
                    well_array_geo = self.custom_color_well_array.geometry()
                    button_geo = button.geometry()
                    # Button below custom color (similar Y coordinate)
                    if abs(button_geo.y() - (well_array_geo.y() + well_array_geo.height())) < 50:
                        if button_geo.width() > 100:  # Wide button
                            # Override click handler
                            button.clicked.disconnect()
                            button.clicked.connect(self._add_custom_color)
                            self._add_button = button
                            break

    def eventFilter(self, obj, event):
        """Monitor click events on custom color QWellArray"""
        if obj == self.custom_color_well_array:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                # Compute cell index from click position
                pos = event.position().toPoint()
                # QWellArray is typically 8 columns x 2 rows (16 colors) grid
                # Get actual widget size
                width = self.custom_color_well_array.width()
                height = self.custom_color_well_array.height()

                # Compute cell size (accounting for borders)
                cell_width = width / 8.0
                cell_height = height / 2.0

                col = int(pos.x() / cell_width)
                row = int(pos.y() / cell_height)

                # Bounds check
                col = max(0, min(7, col))
                row = max(0, min(1, row))

                # Compute index (Qt custom colors stored in column-major order)
                # index = col * rows + row (rows = 2)
                index = col * 2 + row
                self.selected_custom_color_index = index
                # Redraw selection border
                self._draw_selection_border()
            elif event.type() == QtCore.QEvent.Paint:
                # Draw selection border after paint event
                QtCore.QTimer.singleShot(0, self._draw_selection_border)

        return super().eventFilter(obj, event)

    def _add_custom_color(self):
        """Add current color to selected custom color slot"""
        current_color = self.currentColor()
        # Set color at selected index
        QtWidgets.QColorDialog.setCustomColor(self.selected_custom_color_index, current_color)
        # Force QWellArray update
        if self.custom_color_well_array:
            self.custom_color_well_array.update()
            self.custom_color_well_array.repaint()
        # Do not change selected index

    def _draw_selection_border(self):
        """Draw border on selected custom color cell"""
        if not self.custom_color_well_array:
            return

        # Get QWellArray size
        width = self.custom_color_well_array.width()
        height = self.custom_color_well_array.height()

        # Compute cell size (float for precision)
        cell_width = width / 8.0
        cell_height = height / 2.0

        # Compute selected cell position
        # Qt custom colors stored in column-major order
        # index = col * 2 + row, so inversely: col = index // 2, row = index % 2
        col = self.selected_custom_color_index // 2
        row = self.selected_custom_color_index % 2

        # Each QWellArray cell has borders
        # Place frame inside the cell
        BORDER_WIDTH = 2  # Cell border width

        # Compute position (inside border)
        x = int(col * cell_width) + BORDER_WIDTH
        y = int(row * cell_height) + BORDER_WIDTH

        # Compute size (excluding border area)
        next_x = int((col + 1) * cell_width)
        next_y = int((row + 1) * cell_height)
        frame_width = next_x - x - BORDER_WIDTH
        frame_height = next_y - y - BORDER_WIDTH

        # Use QFrame as child widget for border instead of overriding paint event
        if not hasattr(self, '_selection_frame'):
            self._selection_frame = QtWidgets.QFrame(self.custom_color_well_array)
            self._selection_frame.setStyleSheet("border: 3px solid #4080FF; background: transparent;")
            self._selection_frame.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        # Update selection frame position and size
        self._selection_frame.setGeometry(x, y, frame_width, frame_height)
        self._selection_frame.show()
        self._selection_frame.raise_()

    def setCustomColor(self, index, color):
        """Set custom color (uses selected index)"""
        # Set color at selected index
        super().setCustomColor(self.selected_custom_color_index, color)
        # Do not change selected index (repeated clicks set same slot)


# ============================================================================
# RENDERING UTILITIES (Mac M4 Compatibility)
# ============================================================================

class OffscreenRenderer:
    """
    Manages offscreen VTK rendering to QLabel for Mac M4 compatibility.

    This class handles the conversion of VTK render window output to QPixmap
    for display in a QLabel widget, enabling VTK rendering on Apple Silicon
    where native VTK widgets may have compatibility issues.
    """

    def __init__(self, render_window, renderer):
        """
        Initialize offscreen renderer.

        Args:
            render_window: vtkRenderWindow instance (must have SetOffScreenRendering(1))
            renderer: vtkRenderer instance
        """
        self.render_window = render_window
        self.renderer = renderer
        self._is_rendering = False
        self._render_counter = 0

    def render_to_qpixmap(self):
        """
        Render VTK scene offscreen and return QPixmap.

        Returns:
            QPixmap: Rendered image or None if rendering is locked
        """
        if self._is_rendering:
            return None

        self._is_rendering = True
        try:
            self.render_window.Modified()
            self.renderer.Modified()
            self.render_window.Render()

            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.render_window)
            w2i.ReadFrontBufferOff()
            w2i.ShouldRerenderOn()
            w2i.Modified()
            w2i.Update()

            vtk_image = w2i.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()

            # Convert VTK array to numpy array (recommended method)
            arr = vtk_to_numpy(vtk_array)
            arr = arr.reshape(height, width, components)
            arr = np.flip(arr, axis=0)
            arr = np.ascontiguousarray(arr)

            if components == 3:
                qimage = QImage(arr.data, width, height, width * 3, QImage.Format_RGB888)
            else:
                qimage = QImage(arr.data, width, height, width * 4, QImage.Format_RGBA8888)

            pixmap = QPixmap.fromImage(qimage.copy())
            self._render_counter += 1
            return pixmap

        finally:
            self._is_rendering = False

    def update_display(self, qlabel_widget, restore_focus=True):
        """
        Render and update QLabel display.

        Args:
            qlabel_widget: QLabel widget to update with rendered image
            restore_focus: Whether to restore focus to qlabel_widget after rendering
        """
        pixmap = self.render_to_qpixmap()
        if pixmap:
            qlabel_widget.setPixmap(pixmap)
            qlabel_widget.update()

            if restore_focus:
                from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit, QPlainTextEdit
                focus_widget = QApplication.focusWidget()
                if not isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
                    qlabel_widget.setFocus(Qt.OtherFocusReason)


# ============================================================================
# CAMERA UTILITIES
# ============================================================================

class CameraController:
    """
    Manages VTK camera operations (rotation, pan, zoom, reset).

    Provides unified camera control interface for parallel projection cameras
    with methods for rotation, panning, zooming, and fitting to model bounds.
    """

    def __init__(self, renderer, origin=None):
        """
        Initialize camera controller.

        Args:
            renderer: vtkRenderer instance
            origin: Camera focal point [x, y, z], defaults to [0, 0, 0]
        """
        self.renderer = renderer
        self.origin = origin if origin is not None else [0, 0, 0]
        self.rotation = [0, 0, 0]  # [yaw, pitch, roll]

    def setup_parallel_camera(self, position=None, view_up=None,
                            focal_point=None, parallel_scale=5):
        """
        Setup camera with parallel projection.

        Args:
            position: Camera position [x, y, z], defaults to [10, 0, 0]
            view_up: Camera up vector [x, y, z], defaults to [0, 0, 1]
            focal_point: Camera focal point [x, y, z], defaults to origin
            parallel_scale: Parallel projection scale
        """
        if position is None:
            position = [10, 0, 0]
        if view_up is None:
            view_up = [0, 0, 1]
        if focal_point is None:
            focal_point = self.origin

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(*focal_point)
        camera.SetViewUp(*view_up)
        camera.SetParallelScale(parallel_scale)
        camera.ParallelProjectionOn()
        self.renderer.ResetCameraClippingRange()

    def reset_camera(self, position=None, view_up=None):
        """
        Reset camera to default position.

        Args:
            position: Camera position [x, y, z], defaults to [10, 0, 0]
            view_up: Camera up vector [x, y, z], defaults to [0, 0, 1]
        """
        if position is None:
            position = [10, 0, 0]
        if view_up is None:
            view_up = [0, 0, 1]

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(*self.origin)
        camera.SetViewUp(*view_up)
        self.rotation = [0, 0, 0]
        self.renderer.ResetCameraClippingRange()

    def fit_to_bounds(self, bounds, padding=1.4):
        """
        Fit camera to model bounds.

        Args:
            bounds: Model bounds [xmin, xmax, ymin, ymax, zmin, zmax]
            padding: Size multiplier for fitting (default 1.4)
        """
        camera = self.renderer.GetActiveCamera()
        center = [(bounds[i] + bounds[i+1]) / 2 for i in range(0, 6, 2)]

        size = max([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ])
        size *= padding

        camera.SetPosition(center[0] + size, center[1], center[2])
        camera.SetFocalPoint(*center)
        camera.SetViewUp(0, 0, 1)

        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / (viewport[3] - viewport[1])

        if aspect_ratio > 1:
            camera.SetParallelScale(size / 2)
        else:
            camera.SetParallelScale(size / (2 * aspect_ratio))

        self.renderer.ResetCameraClippingRange()

    def rotate_azimuth_elevation(self, dx, dy, sensitivity=0.5):
        """
        Rotate camera using azimuth/elevation (mouse drag).

        Args:
            dx: Horizontal mouse movement delta
            dy: Vertical mouse movement delta
            sensitivity: Rotation sensitivity multiplier
        """
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(-dx * sensitivity)
        camera.Elevation(dy * sensitivity)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()

    def pan(self, dx, dy, pan_speed_factor=0.001):
        """
        Pan camera (Shift + drag) with 1:1 mouse-to-view mapping.

        Args:
            dx: Horizontal mouse movement delta in pixels
            dy: Vertical mouse movement delta in pixels
            pan_speed_factor: Pan speed multiplier (deprecated, kept for compatibility)
        """
        camera = self.renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()

        # Get camera coordinate system
        view_up = np.array(camera.GetViewUp())
        view_direction = np.array(focal_point) - np.array(position)
        distance = np.linalg.norm(view_direction)
        view_direction = view_direction / distance

        # Calculate right and up vectors
        right = np.cross(view_direction, view_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)

        # Calculate 1:1 mapping between mouse pixels and world space
        # Get viewport size in pixels
        render_window = self.renderer.GetRenderWindow()
        window_size = render_window.GetSize()
        viewport_height = window_size[1]

        # For parallel projection, parallel_scale is half the view height in world units
        # Convert pixel movement to world space for 1:1 visual mapping
        parallel_scale = camera.GetParallelScale()
        world_per_pixel = (2.0 * parallel_scale) / viewport_height

        # Calculate pan offset with 1:1 mapping
        offset = -right * dx * world_per_pixel + up * dy * world_per_pixel

        # Move camera and focal point
        new_position = np.array(position) + offset
        new_focal = np.array(focal_point) + offset

        camera.SetPosition(new_position)
        camera.SetFocalPoint(new_focal)
        self.renderer.ResetCameraClippingRange()

    def zoom(self, delta, zoom_factor=0.1):
        """
        Zoom camera (mouse wheel).

        Args:
            delta: Wheel delta (positive = zoom in, negative = zoom out)
            zoom_factor: Zoom speed (0.1 = 10% per step)
        """
        camera = self.renderer.GetActiveCamera()
        current_scale = camera.GetParallelScale()

        if delta > 0:
            new_scale = current_scale * (1 - zoom_factor)
        else:
            new_scale = current_scale * (1 + zoom_factor)

        camera.SetParallelScale(new_scale)
        self.renderer.ResetCameraClippingRange()

    def get_screen_axes(self):
        """
        Get screen-space coordinate axes for arrow key movement.

        Returns:
            tuple: (horizontal_axis, vertical_axis, screen_right, screen_up)
                - horizontal_axis: 'x' or 'z'
                - vertical_axis: 'y' or 'z'
                - screen_right: Right vector in world coordinates
                - screen_up: Up vector in world coordinates
        """
        camera = self.renderer.GetActiveCamera()
        view_up = np.array(camera.GetViewUp())
        forward = np.array(camera.GetDirectionOfProjection())
        right = np.cross(forward, view_up)

        screen_right = right
        screen_up = view_up

        if abs(np.dot(screen_right, [1, 0, 0])) > abs(np.dot(screen_right, [0, 0, 1])):
            horizontal_axis = 'x'
            vertical_axis = 'z' if abs(np.dot(screen_up, [0, 0, 1])) > abs(np.dot(screen_up, [0, 1, 0])) else 'y'
        else:
            horizontal_axis = 'z'
            vertical_axis = 'y'

        return horizontal_axis, vertical_axis, screen_right, screen_up


# ============================================================================
# ANIMATED CAMERA ROTATION
# ============================================================================

class AnimatedCameraRotation:
    """
    Handles 90-degree animated camera rotations (WASDQE keys).

    Provides smooth animated rotations for keyboard-triggered camera movements
    with precise angle control to avoid floating-point drift.
    """

    def __init__(self, renderer, origin=None, frames=12):
        """
        Initialize animated rotation controller.

        Args:
            renderer: vtkRenderer instance
            origin: Rotation center point [x, y, z], defaults to [0, 0, 0]
            frames: Total animation frames (12 frames = 60fps → 200ms)
        """
        self.renderer = renderer
        self.origin = origin if origin is not None else [0, 0, 0]
        self.total_frames = frames
        self.is_animating = False
        self.current_frame = 0
        self.rotation_per_frame = 0
        self.target_angle = 0
        self.rotation_type = None

    def start_rotation(self, angle, rotation_type):
        """
        Start animated rotation.

        Args:
            angle: Rotation angle in degrees (e.g., 90, -90)
            rotation_type: 'yaw', 'pitch', or 'roll'

        Returns:
            bool: True if started, False if already animating
        """
        if self.is_animating:
            return False

        self.is_animating = True
        self.current_frame = 0
        self.rotation_per_frame = angle / self.total_frames
        self.target_angle = angle
        self.rotation_type = rotation_type
        return True

    def animate_frame(self):
        """
        Animate one frame of rotation.

        Returns:
            bool: True if animation continues, False if complete
        """
        if not self.is_animating:
            return False

        self.current_frame += 1
        camera = self.renderer.GetActiveCamera()
        position = list(camera.GetPosition())
        focal_point = self.origin
        view_up = list(camera.GetViewUp())

        # Calculate rotation axis
        forward = [focal_point[i] - position[i] for i in range(3)]
        right = [
            view_up[1] * forward[2] - view_up[2] * forward[1],
            view_up[2] * forward[0] - view_up[0] * forward[2],
            view_up[0] * forward[1] - view_up[1] * forward[0]
        ]

        if self.rotation_type == 'yaw':
            axis = view_up
        elif self.rotation_type == 'pitch':
            axis = right
        else:  # roll
            axis = forward

        # Calculate rotation angle for this frame
        if self.current_frame >= self.total_frames:
            # Last frame - apply exact remaining angle
            remaining_angle = self.target_angle - (self.rotation_per_frame * (self.current_frame - 1))
            rotation_angle = remaining_angle
        else:
            rotation_angle = self.rotation_per_frame

        # Apply rotation
        rotation_matrix = vtk.vtkTransform()
        rotation_matrix.Translate(*focal_point)
        rotation_matrix.RotateWXYZ(rotation_angle, axis)
        rotation_matrix.Translate(*[-x for x in focal_point])

        new_position = rotation_matrix.TransformPoint(position)
        new_up = rotation_matrix.TransformVector(view_up)

        camera.SetPosition(new_position)
        camera.SetViewUp(new_up)

        # Check if animation is complete
        if self.current_frame >= self.total_frames:
            self.is_animating = False
            return False

        return True


# ============================================================================
# MARKER RENDERING UTILITIES
# ============================================================================

class AdaptiveMarkerSize:
    """
    Calculate adaptive marker size based on camera zoom level.

    Ensures markers remain visible and proportional regardless of camera zoom.
    """

    @staticmethod
    def calculate_sphere_radius(renderer):
        """
        Calculate adaptive sphere radius based on camera parallel scale.

        Args:
            renderer: vtkRenderer instance

        Returns:
            float: Radius in world coordinates
        """
        camera = renderer.GetActiveCamera()
        parallel_scale = camera.GetParallelScale()
        viewport = renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / (viewport[3] - viewport[1])
        radius = parallel_scale * 0.05
        if aspect_ratio > 1:
            radius /= aspect_ratio
        return radius


# ============================================================================
# MESH AND COORDINATE UTILITIES
# ============================================================================

def is_mesh_reversed_check(visual_origin, mesh_scale):
    """
    Check if mesh is reversed (mirrored)

    Args:
        visual_origin: visual_origin dict {'xyz': [...], 'rpy': [...]}
        mesh_scale: mesh_scale list [x, y, z]

    Returns:
        bool: True if reversed
    """
    PI = math.pi
    PI_TOLERANCE = 0.01

    # Consider reversed if any RPY axis is close to PI
    if visual_origin:
        rpy = visual_origin.get('rpy', [0.0, 0.0, 0.0])
        for angle in rpy:
            if abs(abs(angle) - PI) < PI_TOLERANCE:
                return True

    # Consider reversed if any mesh_scale axis is negative
    if mesh_scale:
        for scale in mesh_scale:
            if scale < 0:
                return True

    return False


def create_cumulative_coord(index):
    """Create cumulative coordinate data"""
    return {
        'point_index': index,
        'xyz': [0.0, 0.0, 0.0]
    }


def create_crosshair_marker(coords, radius_scale=1.0):
    """
    Create a 3D crosshair marker with circles (3-axis crosshair + 3 circles).

    Args:
        coords: Position [x, y, z]
        radius_scale: Scale factor for marker size

    Returns:
        vtkAssembly: Assembly containing the marker components
    """
    assembly = vtk.vtkAssembly()
    origin = coords
    axis_length = radius_scale * 36  # Axis line length
    circle_radius = radius_scale

    # Create 3-axis crosshair lines (RGB = XYZ)
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
            assembly.AddPart(actor)

    # Create 3 circles (XY, XZ, YZ planes)
    for i in range(3):
        circle = vtk.vtkRegularPolygonSource()
        circle.SetNumberOfSides(50)
        circle.SetRadius(circle_radius)
        circle.SetCenter(origin[0], origin[1], origin[2])
        if i == 0:
            circle.SetNormal(0, 0, 1)  # XY plane
        elif i == 1:
            circle.SetNormal(0, 1, 0)  # XZ plane
        else:
            circle.SetNormal(1, 0, 0)  # YZ plane

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(circle.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 1)  # Magenta
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(6)
        actor.GetProperty().SetOpacity(0.7)

        assembly.AddPart(actor)

    return assembly


# ============================================================================
# MOUSE EVENT HANDLING
# ============================================================================

class MouseDragState:
    """
    Manages mouse drag state for VTK interaction with QLabel.

    Tracks mouse button states and positions for implementing drag operations
    like rotation and panning.
    """

    def __init__(self, qlabel_widget):
        """
        Initialize mouse drag state manager.

        Args:
            qlabel_widget: QLabel widget to grab/release mouse
        """
        self.widget = qlabel_widget
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.last_pos = None

    def start_left_drag(self, pos):
        """Start left button drag."""
        self.left_button_pressed = True
        self.last_pos = pos
        self.widget.grabMouse()

    def start_middle_drag(self, pos):
        """Start middle button drag."""
        self.middle_button_pressed = True
        self.last_pos = pos
        self.widget.grabMouse()

    def update_pos(self, pos):
        """
        Update last position and return delta.

        Args:
            pos: Current mouse position

        Returns:
            tuple: (dx, dy) mouse movement delta
        """
        if self.last_pos:
            dx = pos.x() - self.last_pos.x()
            dy = pos.y() - self.last_pos.y()
            self.last_pos = pos
            return dx, dy
        return 0, 0

    def end_left_drag(self):
        """End left button drag."""
        self.left_button_pressed = False
        self.last_pos = None
        self.widget.releaseMouse()

    def end_middle_drag(self):
        """End middle button drag."""
        self.middle_button_pressed = False
        self.last_pos = None
        self.widget.releaseMouse()

    def cancel_all(self):
        """Cancel all drag operations (fallback for errors)."""
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.last_pos = None
        try:
            self.widget.releaseMouse()
        except:
            pass


# ============================================================================
# KEYBOARD STEP UTILITIES
# ============================================================================

def calculate_arrow_key_step(shift_pressed, ctrl_pressed):
    """
    Calculate movement step for arrow keys.

    Args:
        shift_pressed: Shift modifier state
        ctrl_pressed: Ctrl modifier state

    Returns:
        float: Step size in meters
            - Default: 0.01 (10mm)
            - Shift: 0.001 (1mm)
            - Shift+Ctrl: 0.0001 (0.1mm)
    """
    if shift_pressed and ctrl_pressed:
        return 0.0001  # 0.1mm
    elif shift_pressed:
        return 0.001   # 1mm
    else:
        return 0.01    # 10mm


# ============================================================================
# QUATERNION AND EULER ANGLE CONVERSION UTILITIES
# ============================================================================

def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """
    Convert URDF RPY (roll, pitch, yaw) in degrees to quaternion.

    URDF RPY convention:
    - Rotate by yaw around Z axis
    - Then pitch around Y axis
    - Then roll around X axis
    (ZYX extrinsic / XYZ intrinsic rotation order)

    Args:
        roll_deg: Rotation around X axis in degrees
        pitch_deg: Rotation around Y axis in degrees
        yaw_deg: Rotation around Z axis in degrees

    Returns:
        np.ndarray: Quaternion [w, x, y, z]

    Example:
        >>> q = euler_to_quaternion(0, 90, 0)  # 90 degree pitch
        >>> print(q)
        [0.707107, 0.0, 0.707107, 0.0]
    """
    import numpy as np

    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

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


def quaternion_to_euler(q):
    """
    Convert quaternion to URDF RPY (roll, pitch, yaw) in degrees.

    Args:
        q: Quaternion as array-like [w, x, y, z]

    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw] in degrees

    Example:
        >>> q = np.array([0.707107, 0.0, 0.707107, 0.0])
        >>> euler = quaternion_to_euler(q)
        >>> print(euler)
        [0.0, 90.0, 0.0]
    """
    import numpy as np

    w, x, y, z = q

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # Gimbal lock case
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


def quaternion_to_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion as array-like [w, x, y, z]

    Returns:
        np.ndarray: 3x3 rotation matrix

    Example:
        >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        >>> m = quaternion_to_matrix(q)
        >>> print(m)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    import numpy as np

    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def format_float_no_exp(value, max_decimals=15):
    """
    Format float without exponential notation, removing trailing zeros.

    Used for URDF XML generation to ensure human-readable numbers.

    Args:
        value: Float value to format
        max_decimals: Maximum number of decimal places (default: 15)

    Returns:
        str: Formatted string representation

    Example:
        >>> format_float_no_exp(0.0001)
        '0.0001'
        >>> format_float_no_exp(1.0)
        '1'
        >>> format_float_no_exp(1.23000)
        '1.23'
    """
    # Format with max decimals
    formatted = f"{value:.{max_decimals}f}"

    # Remove trailing zeros and decimal point if not needed
    formatted = formatted.rstrip('0').rstrip('.')

    # Return '0' if empty string
    return formatted if formatted else '0'


# ============================================================================
# INERTIA TENSOR CALCULATION UTILITIES
# ============================================================================

def calculate_inertia_tensor(poly_data, mass, center_of_mass, is_mirrored=False):
    """
    Calculate inertia tensor for a 3D mesh using triangle-based method.

    This method is used by both PartsEditor and Assembler for consistent
    inertia calculations. It uses a triangle-based approach with parallel
    axis theorem.

    Args:
        poly_data: vtkPolyData object containing the mesh
        mass: float - Total mass of the object
        center_of_mass: array-like [x, y, z] - Center of mass coordinates
        is_mirrored: bool - If True, applies Y-axis mirroring transformation

    Returns:
        numpy.ndarray: 3x3 inertia tensor matrix

    Example:
        >>> import vtk
        >>> import numpy as np
        >>> # Load mesh
        >>> reader = vtk.vtkSTLReader()
        >>> reader.SetFileName("model.stl")
        >>> reader.Update()
        >>> poly_data = reader.GetOutput()
        >>>
        >>> # Calculate inertia
        >>> mass = 1.0
        >>> com = [0, 0, 0]
        >>> inertia = calculate_inertia_tensor(poly_data, mass, com)
    """
    # Calculate volume
    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(poly_data)
    mass_properties.Update()
    total_volume = mass_properties.GetVolume()

    # Calculate density from mass and volume
    density = mass / total_volume if total_volume > 0 else 0
    print(f"Calculated density: {density:.6f} from mass: {mass:.6f} and volume: {total_volume:.6f}")

    # Initialize inertia tensor
    inertia_tensor = np.zeros((3, 3))
    num_cells = poly_data.GetNumberOfCells()
    print(f"Processing {num_cells} triangles for inertia tensor calculation...")

    for i in range(num_cells):
        cell = poly_data.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            # Get triangle vertices (relative to center of mass)
            points = [np.array(cell.GetPoints().GetPoint(j)) - np.array(center_of_mass) for j in range(3)]

            # Apply Y-axis mirroring if needed
            if is_mirrored:
                points = [np.array([p[0], -p[1], p[2]]) for p in points]

            # Calculate triangle area and normal vector
            v1 = np.array(points[1]) - np.array(points[0])
            v2 = np.array(points[2]) - np.array(points[0])
            normal = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(normal)

            if area < 1e-10:  # Skip degenerate triangles
                continue

            # Calculate triangle centroid
            tri_centroid = np.mean(points, axis=0)

            # Calculate local inertia tensor for the triangle
            covariance = np.zeros((3, 3))
            for p in points:
                r_squared = np.sum(p * p)
                for a in range(3):
                    for b in range(3):
                        if a == b:
                            # Diagonal components
                            covariance[a, a] += (r_squared - p[a] * p[a]) * area / 12.0
                        else:
                            # Off-diagonal components
                            covariance[a, b] -= (p[a] * p[b]) * area / 12.0

            # Apply parallel axis theorem
            r_squared = np.sum(tri_centroid * tri_centroid)
            parallel_axis_term = np.zeros((3, 3))
            for a in range(3):
                for b in range(3):
                    if a == b:
                        parallel_axis_term[a, a] = r_squared * area
                    else:
                        parallel_axis_term[a, b] = tri_centroid[a] * tri_centroid[b] * area

            # Combine local inertia and parallel axis term
            local_inertia = covariance + parallel_axis_term

            # Add to total inertia tensor
            inertia_tensor += local_inertia

    # Apply density to get final inertia tensor
    inertia_tensor *= density

    # Clean up numerical errors
    threshold = 1e-10
    inertia_tensor[np.abs(inertia_tensor) < threshold] = 0.0

    # Ensure symmetry
    inertia_tensor = 0.5 * (inertia_tensor + inertia_tensor.T)

    # Ensure positive diagonal elements
    for i in range(3):
        if inertia_tensor[i, i] <= 0:
            print(f"Warning: Non-positive diagonal element detected at position ({i},{i})")
            inertia_tensor[i, i] = abs(inertia_tensor[i, i])

    return inertia_tensor


def calculate_inertia_tetrahedral(poly_data, density, center_of_mass):
    """
    Calculate inertia tensor using tetrahedral decomposition method.

    This implementation uses the method described in:
    "Fast and Accurate Computation of Polyhedral Mass Properties" by Brian Mirtich (1996)

    This method is more accurate than triangle-based methods for closed meshes
    and is used by PartsEditor for high-precision calculations.

    Args:
        poly_data: vtkPolyData object containing the mesh
        density: float - Material density
        center_of_mass: array-like [x, y, z] - Center of mass coordinates

    Returns:
        tuple: (inertia_tensor, volume_integral)
            - inertia_tensor: numpy.ndarray - 3x3 inertia tensor matrix
            - volume_integral: float - Total volume from integration

    Example:
        >>> import vtk
        >>> import numpy as np
        >>> # Load mesh
        >>> reader = vtk.vtkSTLReader()
        >>> reader.SetFileName("model.stl")
        >>> reader.Update()
        >>> poly_data = reader.GetOutput()
        >>>
        >>> # Calculate inertia
        >>> density = 1000.0  # kg/m^3
        >>> com = [0, 0, 0]
        >>> inertia, volume = calculate_inertia_tetrahedral(poly_data, density, com)
    """
    com = np.array(center_of_mass)

    # Initialize integrals
    volume_integral = 0.0
    inertia_integral = np.zeros((3, 3))

    # Process all triangles
    num_cells = poly_data.GetNumberOfCells()
    print(f"Processing {num_cells} triangles using tetrahedral decomposition...")

    for i in range(num_cells):
        cell = poly_data.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            # Get triangle vertices (shifted so COM is at origin)
            v0 = np.array(cell.GetPoints().GetPoint(0)) - com
            v1 = np.array(cell.GetPoints().GetPoint(1)) - com
            v2 = np.array(cell.GetPoints().GetPoint(2)) - com

            # Signed volume of tetrahedron (origin, v0, v1, v2)
            # V = (1/6) * v0 · (v1 × v2)
            tet_volume = np.dot(v0, np.cross(v1, v2)) / 6.0

            # Skip degenerate triangles
            if abs(tet_volume) < 1e-12:
                continue

            # Accumulate volume
            volume_integral += tet_volume

            # Calculate inertia integral for tetrahedron
            # I_ij = ∫∫∫ ρ(r²δ_ij - x_i*x_j) dV

            # Vertex contributions
            for vi in [v0, v1, v2]:
                for vj in [v0, v1, v2]:
                    r_squared = np.dot(vi, vi)
                    for a in range(3):
                        for b in range(3):
                            if a == b:
                                # Diagonal: I_aa = ∫(y² + z²)dm for x-axis, etc.
                                inertia_integral[a, b] += tet_volume * (r_squared - vi[a] * vj[a]) / 20.0
                            else:
                                # Off-diagonal: I_ab = -∫(x*y)dm
                                inertia_integral[a, b] -= tet_volume * vi[a] * vj[b] / 20.0

            # Vertex pair contributions
            pairs = [(v0, v1), (v1, v2), (v2, v0)]
            for vi, vj in pairs:
                r_squared_i = np.dot(vi, vi)
                r_squared_j = np.dot(vj, vj)

                for a in range(3):
                    for b in range(3):
                        if a == b:
                            inertia_integral[a, b] += tet_volume * (r_squared_i + r_squared_j - vi[a] * vj[a]) / 60.0
                        else:
                            inertia_integral[a, b] -= tet_volume * (vi[a] * vj[b] + vj[a] * vi[b]) / 60.0

    # Convert to inertia tensor with density scaling
    inertia_tensor = inertia_integral * density

    # Ensure symmetry
    inertia_tensor = 0.5 * (inertia_tensor + inertia_tensor.T)

    # Clean up numerical noise
    threshold = 1e-12
    inertia_tensor[np.abs(inertia_tensor) < threshold] = 0.0

    # Ensure positive diagonal elements
    diagonal = np.diag(inertia_tensor)
    if np.any(diagonal <= 0):
        print("\nWarning: Non-positive diagonal elements detected!")
        print(f"Diagonal: [{diagonal[0]:.6e}, {diagonal[1]:.6e}, {diagonal[2]:.6e}]")
        print("This may indicate:")
        print("  - Mesh has inverted normals")
        print("  - Mesh is not closed")
        print("  - Numerical precision issues")

        # Fix by taking absolute values
        for i in range(3):
            if inertia_tensor[i, i] <= 0:
                inertia_tensor[i, i] = abs(inertia_tensor[i, i])

    # Triangle inequality check: I_xx + I_yy >= I_zz (and cyclic permutations)
    diagonal = np.diag(inertia_tensor)
    if not (diagonal[0] + diagonal[1] >= diagonal[2] - 1e-6 and
            diagonal[1] + diagonal[2] >= diagonal[0] - 1e-6 and
            diagonal[2] + diagonal[0] >= diagonal[1] - 1e-6):
        print("\nWarning: Triangle inequality violated!")
        print("Inertia tensor may not be physically valid")

    return inertia_tensor, volume_integral


def validate_inertia_tensor(inertia_tensor, mass):
    """
    Validate physical validity of inertia tensor.

    Args:
        inertia_tensor: 3x3 numpy array
        mass: Mass

    Returns:
        dict: {'valid': bool, 'message': str}
    """
    try:
        ixx = inertia_tensor[0, 0]
        iyy = inertia_tensor[1, 1]
        izz = inertia_tensor[2, 2]
        ixy = inertia_tensor[0, 1]
        ixz = inertia_tensor[0, 2]
        iyz = inertia_tensor[1, 2]

        messages = []
        valid = True

        # 1. Diagonal elements must be positive
        if ixx <= 0 or iyy <= 0 or izz <= 0:
            messages.append("⚠ Diagonal elements must be positive")
            valid = False

        # 2. Triangle inequality check (Ixx + Iyy >= Izz, etc.)
        if not (ixx + iyy >= izz - 1e-6 and
               iyy + izz >= ixx - 1e-6 and
               izz + ixx >= iyy - 1e-6):
            messages.append("⚠ Triangle inequality violated")
            valid = False

        # 3. Tensor symmetry check
        if not np.allclose(inertia_tensor, inertia_tensor.T, rtol=1e-5):
            messages.append("⚠ Tensor is not symmetric")
            valid = False

        # 4. All eigenvalues must be positive
        eigenvalues = np.linalg.eigvals(inertia_tensor)
        if np.any(eigenvalues <= 0):
            messages.append("⚠ Tensor has non-positive eigenvalues")
            valid = False

        # 5. Consistency check with mass (rough range check)
        # Inertia magnitude depends on mass and shape size
        trace = ixx + iyy + izz
        if trace < 0:
            messages.append("⚠ Trace of tensor is negative")
            valid = False

        if valid:
            messages.append("✓ Inertia tensor validation passed")

        return {
            'valid': valid,
            'message': '\n'.join(messages)
        }

    except Exception as e:
        return {
            'valid': False,
            'message': f"Validation error: {str(e)}"
        }


# ============================================================================
# PARALLEL AXIS THEOREM
# ============================================================================

def parallel_axis_transform(inertia_at_com, mass, displacement):
    """
    Transform inertia tensor at center of mass to inertia at arbitrary point using parallel axis theorem.

    Formula:
        I_P = I_COM + m * (||d||² * E - d ⊗ d)

    Args:
        inertia_at_com: numpy.ndarray - 3x3 inertia tensor at center of mass
        mass: float - Mass (kg)
        displacement: array-like - Vector from COM to target point [dx, dy, dz] (m)

    Returns:
        numpy.ndarray: 3x3 inertia tensor at target point

    Example:
        >>> I_com = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> mass = 1.0
        >>> d = [1.0, 0.0, 0.0]  # 1m displacement along X axis
        >>> I_p = parallel_axis_transform(I_com, mass, d)
    """
    import numpy as np

    d = np.array(displacement, dtype=float)
    d_squared = np.dot(d, d)  # ||d||²

    # Identity matrix
    E = np.eye(3)

    # d ⊗ d (outer product, dyadic product)
    d_outer_d = np.outer(d, d)

    # Steiner term: m * (||d||² * E - d ⊗ d)
    steiner_term = mass * (d_squared * E - d_outer_d)

    # Transform
    inertia_at_point = inertia_at_com + steiner_term

    return inertia_at_point


def inverse_parallel_axis_transform(inertia_at_point, mass, displacement):
    """
    Inverse parallel axis transform: convert inertia at arbitrary point to inertia at center of mass.

    Formula:
        I_COM = I_P - m * (||d||² * E - d ⊗ d)

    Args:
        inertia_at_point: numpy.ndarray - 3x3 inertia tensor at arbitrary point
        mass: float - Mass (kg)
        displacement: array-like - Vector from COM to target point [dx, dy, dz] (m)

    Returns:
        numpy.ndarray: 3x3 inertia tensor at center of mass

    Example:
        >>> I_p = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        >>> mass = 1.0
        >>> d = [1.0, 0.0, 0.0]
        >>> I_com = inverse_parallel_axis_transform(I_p, mass, d)
    """
    import numpy as np

    d = np.array(displacement, dtype=float)
    d_squared = np.dot(d, d)

    E = np.eye(3)
    d_outer_d = np.outer(d, d)

    # Subtract Steiner term
    steiner_term = mass * (d_squared * E - d_outer_d)
    inertia_at_com = inertia_at_point - steiner_term

    return inertia_at_com


# ============================================================================
# MIRRORING UTILITIES
# ============================================================================

def mirror_inertia_tensor_y_axis(inertia_tensor, center_of_mass):
    """
    Transform inertia tensor and COM for Y-axis mirroring (tensor transform only).

    Y-axis mirror transform:
        R = diag(1, -1, 1)
        I' = R @ I @ R.T
        COM' = [x, -y, z]

    Sign changes:
        - ixy and iyz are negated
        - Diagonal components (ixx, iyy, izz) are unchanged
        - ixz is unchanged

    Args:
        inertia_tensor: numpy.ndarray - 3x3 inertia tensor
        center_of_mass: array-like - Center of mass coordinates [x, y, z]

    Returns:
        tuple: (mirrored_inertia_tensor, mirrored_com)
            - mirrored_inertia_tensor: 3x3 numpy.ndarray
            - mirrored_com: list [x, -y, z]

    Example:
        >>> I = np.array([[1, 0.1, 0.2], [0.1, 2, 0.3], [0.2, 0.3, 3]])
        >>> com = [1.0, 2.0, 3.0]
        >>> I_mirror, com_mirror = mirror_inertia_tensor_y_axis(I, com)
        >>> # I_mirror has ixy and iyz negated
        >>> # com_mirror = [1.0, -2.0, 3.0]
    """
    import numpy as np

    # Mirror transform matrix (Y-axis flip)
    R = np.diag([1.0, -1.0, 1.0])

    # Tensor transform: I' = R @ I @ R.T
    mirrored_tensor = R @ inertia_tensor @ R.T

    # COM transform: [x, y, z] → [x, -y, z]
    com = np.array(center_of_mass, dtype=float)
    mirrored_com = [com[0], -com[1], com[2]]

    return mirrored_tensor, mirrored_com


# ============================================================================
# UNIFIED INERTIA CALCULATION WITH TRIMESH
# ============================================================================

def calculate_inertia_with_trimesh(
    mesh_file_path,
    mass=None,
    density=None,
    reference_point=None,
    auto_repair=True,
    unit_scale=1.0,
    # For backward compatibility (deprecated)
    center_of_mass=None,
    use_trimesh_com=False
):
    """
    Unified inertia tensor calculation using trimesh library (revised).

    Provides physically accurate inertia tensor computation:
    - Uses parallel axis theorem for inertia at arbitrary origin
    - Warns when unit_scale is unspecified for Scene (unit ambiguity)
    - Does not fabricate physical quantities for non-watertight meshes

    Processing flow:
    1. Load mesh file with trimesh (Scene defaults to unit_scale=1.0)
    2. Auto-repair mesh if needed
    3. Compute volume and center of mass with trimesh (physical basis)
    4. Compute inertia tensor with VTK (tetrahedral decomposition)
    5. Transform to arbitrary origin via parallel axis theorem

    Args:
        mesh_file_path: str - Path to mesh file (.stl, .obj, .dae)
        mass: float or None - Mass (kg). Computed from density if None
        density: float or None - Density (kg/m³). Error if both mass and density are None
        reference_point: array-like or None - Inertia tensor origin [x, y, z] (m)
            None means center of mass is origin
        auto_repair: bool - Whether to auto-repair mesh (default: True)
        unit_scale: float - Unit scale for Scene (e.g. 0.001 for mm to m)
            Default: 1.0 (no scaling). Warns if unspecified for Scene file.
        center_of_mass: array-like or None - **Deprecated** Use reference_point instead
        use_trimesh_com: bool - **Deprecated** Same as reference_point=None

    Returns:
        dict: {
            'success': bool - Computation success flag
            'inertia_tensor': numpy.ndarray - 3x3 inertia matrix (around reference_point)
            'inertia_at_com': numpy.ndarray - 3x3 inertia matrix (around COM)
            'volume': float - Volume (m³)
            'mass': float - Mass (kg)
            'density': float - Density (kg/m³)
            'center_of_mass': list - COM coordinates [x, y, z] (m)
            'reference_point': list - Inertia tensor origin [x, y, z] (m)
            'is_watertight': bool - Whether mesh is closed
            'repair_performed': bool - Whether repair was performed
            'is_scene': bool - Whether loaded as Scene
            'unit_scale_used': float or None - Unit scale used
            'error_message': str or None - Error message
        }

    Raises:
        ValueError: unit_scale not specified for Scene
        ValueError: Both mass and density are None
        ValueError: Non-watertight mesh (after auto_repair failure)

    Example:
        >>> # Basic usage (inertia around COM)
        >>> result = calculate_inertia_with_trimesh('part.stl', mass=1.0)

        >>> # Inertia around arbitrary origin
        >>> result = calculate_inertia_with_trimesh(
        ...     'part.stl', mass=1.0, reference_point=[0, 0, 0])

        >>> # Scene (unit_scale required)
        >>> result = calculate_inertia_with_trimesh(
        ...     'assembly.dae', density=1000.0, unit_scale=0.001)  # mm→m

        >>> # Backward compatibility (deprecated)
        >>> result = calculate_inertia_with_trimesh(
        ...     'part.stl', mass=1.0, center_of_mass=[0, 0, 0])  # Prefer reference_point
    """
    import os
    import numpy as np

    # Backward compatibility handling
    if center_of_mass is not None:
        if reference_point is None:
            reference_point = center_of_mass
            print("⚠ Warning: 'center_of_mass' parameter is deprecated. Use 'reference_point' instead.")
        else:
            print("⚠ Warning: Both 'center_of_mass' and 'reference_point' specified. Using 'reference_point'.")

    if use_trimesh_com and reference_point is not None:
        print("⚠ Warning: 'use_trimesh_com=True' is ignored when 'reference_point' is specified.")

    # Result dictionary
    result = {
        'success': False,
        'inertia_tensor': None,
        'inertia_at_com': None,
        'volume': 0.0,
        'mass': 0.0,
        'density': 0.0,
        'center_of_mass': [0.0, 0.0, 0.0],
        'trimesh_com': [0.0, 0.0, 0.0],  # Backward compatibility
        'reference_point': [0.0, 0.0, 0.0],
        'is_watertight': False,
        'repair_performed': False,
        'is_scene': False,
        'unit_scale_used': None,
        'error_message': None
    }

    try:
        # Check trimesh import
        try:
            import trimesh
        except ImportError:
            result['error_message'] = "trimesh library not available. Install with: pip install trimesh"
            return result

        # Input validation
        if not os.path.exists(mesh_file_path):
            result['error_message'] = f"Mesh file not found: {mesh_file_path}"
            return result

        if mass is None and density is None:
            result['error_message'] = "Either mass or density must be provided"
            return result

        print(f"\n{'='*60}")
        print(f"INERTIA CALCULATION WITH TRIMESH")
        print(f"{'='*60}")
        print(f"File: {mesh_file_path}")
        print(f"Mass: {mass}")
        print(f"Density: {density}")
        print(f"Reference point: {reference_point}")
        print(f"Auto-repair: {auto_repair}")
        print(f"Unit scale: {unit_scale}")

        # Step 1: Load mesh (strict Scene handling)
        loaded_mesh = trimesh.load(mesh_file_path)

            # Scene handling (unit_scale check)
        if isinstance(loaded_mesh, trimesh.Scene):
            result['is_scene'] = True
            print(f"\n⚠ Loaded as Scene (contains multiple meshes or materials)")

            # Warn if unit_scale is default value
            if unit_scale == 1.0:
                print(f"  ⚠ WARNING: unit_scale not specified (using default 1.0)")
                print(f"     Scene files may have ambiguous units.")
                print(f"     Please verify units are correct, or specify unit_scale explicitly.")
                print(f"     Example: unit_scale=0.001 for mm→m conversion")

            print(f"  Unit scale: {unit_scale}")
            result['unit_scale_used'] = unit_scale

            # Combine Scene into mesh
            mesh = loaded_mesh.dump(concatenate=True)

            # Apply scale (explicit even for 1.0)
            mesh.apply_scale(unit_scale)
            if unit_scale != 1.0:
                print(f"  ✓ Applied unit scale: {unit_scale}")
            else:
                print(f"  No scaling applied (unit_scale=1.0)")
        else:
            # Single mesh
            mesh = loaded_mesh
            print(f"\n✓ Loaded as single Mesh")

        print(f"\nMesh properties:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Volume (trimesh): {mesh.volume:.9f} m³")
        print(f"  Is watertight: {mesh.is_watertight}")

        result['is_watertight'] = mesh.is_watertight

        # Step 2: Auto-repair
        if auto_repair and not mesh.is_watertight:
            print(f"\n{'='*60}")
            print(f"MESH REPAIR")
            print(f"{'='*60}")
            result['repair_performed'] = True

            try:
                if hasattr(mesh, 'fix_normals'):
                    print("  ✓ Fixing normals...")
                    mesh.fix_normals()

                if hasattr(mesh, 'remove_duplicate_faces'):
                    print("  ✓ Removing duplicate faces...")
                    mesh.remove_duplicate_faces()

                if hasattr(mesh, 'remove_degenerate_faces'):
                    print("  ✓ Removing degenerate faces...")
                    mesh.remove_degenerate_faces()

                if hasattr(mesh, 'fill_holes'):
                    print("  ✓ Filling holes...")
                    mesh.fill_holes()

                result['is_watertight'] = mesh.is_watertight
                print(f"\nRepair result:")
                print(f"  Vertices: {len(mesh.vertices)}")
                print(f"  Faces: {len(mesh.faces)}")
                print(f"  Is watertight: {mesh.is_watertight}")

                if mesh.is_watertight:
                    print("  ✓ Mesh successfully repaired!")
                else:
                    print("  ⚠ Mesh still not watertight after repair")

            except Exception as repair_error:
                print(f"  ✗ Mesh repair failed: {str(repair_error)}")

        # Warn if not watertight (computation continues)
        if not mesh.is_watertight:
            print(f"\n⚠ WARNING: Mesh is not watertight!")
            print(f"  Physical properties may not be accurate.")

        # Step 3: Convert to VTK polydata for volume computation
        # Convert trimesh mesh directly to VTK (unit_scale reflected)
        poly_data, vtk_volume = trimesh_to_vtk_polydata(mesh)

        print(f"\n{'='*60}")
        print(f"VOLUME CALCULATION")
        print(f"{'='*60}")
        print(f"  Trimesh volume: {mesh.volume:.9f} m³")
        print(f"  VTK volume: {vtk_volume:.9f} m³")

        # Use VTK volume (more reliable)
        result['volume'] = vtk_volume

        # Step 4: Compute center of mass with trimesh
        trimesh_com = mesh.center_mass
        print(f"\nCenter of mass (trimesh): [{trimesh_com[0]:.6f}, {trimesh_com[1]:.6f}, {trimesh_com[2]:.6f}]")

        result['center_of_mass'] = list(trimesh_com)
        result['trimesh_com'] = list(trimesh_com)  # Backward compatibility

        # Step 5: Mass and density computation
        if mass is None:
            mass = vtk_volume * density
            print(f"\nMass calculated from density: {mass:.6f} kg")
        elif density is None:
            density = mass / vtk_volume if vtk_volume > 0 else 0
            print(f"\nDensity calculated from mass: {density:.6f} kg/m³")

        result['mass'] = mass
        result['density'] = density

        # Step 6: Compute inertia tensor around COM (VTK tetrahedral decomposition)
        print(f"\n{'='*60}")
        print(f"INERTIA TENSOR CALCULATION (at COM)")
        print(f"{'='*60}")
        print(f"  Method: Tetrahedral decomposition (VTK)")
        print(f"  Mass: {mass:.6f} kg")
        print(f"  Density: {density:.6f} kg/m³")

        inertia_at_com, volume_integral = calculate_inertia_tetrahedral(
            poly_data, density, list(trimesh_com)
        )

        # Volume validation
        volume_ratio = volume_integral / vtk_volume if vtk_volume > 0 else 0
        print(f"\nVolume verification:")
        print(f"  Volume from integration: {volume_integral:.9f} m³")
        print(f"  VTK volume: {vtk_volume:.9f} m³")
        print(f"  Ratio: {volume_ratio:.4f}")

        if abs(volume_ratio - 1.0) > 0.01:
            print(f"  ⚠ Volume mismatch detected (ratio = {volume_ratio:.4f})")

        result['inertia_at_com'] = inertia_at_com

        print(f"\nInertia tensor (at COM):")
        print(inertia_at_com)

        # Step 7: Transform via parallel axis theorem if reference_point specified
        if reference_point is not None:
            ref_pt = np.array(reference_point, dtype=float)
            displacement = ref_pt - np.array(trimesh_com)

            print(f"\n{'='*60}")
            print(f"PARALLEL AXIS TRANSFORM")
            print(f"{'='*60}")
            print(f"  Reference point: [{ref_pt[0]:.6f}, {ref_pt[1]:.6f}, {ref_pt[2]:.6f}]")
            print(f"  Center of mass:  [{trimesh_com[0]:.6f}, {trimesh_com[1]:.6f}, {trimesh_com[2]:.6f}]")
            print(f"  Displacement:    [{displacement[0]:.6f}, {displacement[1]:.6f}, {displacement[2]:.6f}]")

            inertia_at_ref = parallel_axis_transform(inertia_at_com, mass, displacement)

            result['inertia_tensor'] = inertia_at_ref
            result['reference_point'] = list(ref_pt)

            print(f"\nInertia tensor (at reference point):")
            print(inertia_at_ref)
        else:
            # Use tensor around COM if reference_point not specified
            result['inertia_tensor'] = inertia_at_com
            result['reference_point'] = list(trimesh_com)

        result['success'] = True

        print(f"\n{'='*60}")
        print(f"✓ CALCULATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")

        return result

    except Exception as e:
        result['error_message'] = str(e)
        print(f"\n{'='*60}")
        print(f"✗ ERROR")
        print(f"{'='*60}")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        return result


# ============================================================================
# MESH FILE I/O UTILITIES
# ============================================================================

def get_mesh_file_filter(trimesh_available=True):
    """
    Get file filter string for mesh file dialogs.

    Supports STL, OBJ, and DAE (COLLADA) formats.
    DAE support requires trimesh library.

    Args:
        trimesh_available: bool - Whether trimesh library is available

    Returns:
        str: File filter string for QFileDialog

    Example:
        >>> from PySide6.QtWidgets import QFileDialog
        >>> filter_str = get_mesh_file_filter(trimesh_available=True)
        >>> file_path, _ = QFileDialog.getOpenFileName(self, "Open Mesh", "", filter_str)
    """
    if trimesh_available:
        return "3D Model Files (*.stl *.obj *.dae);;STL Files (*.stl);;OBJ Files (*.obj);;COLLADA Files (*.dae);;All Files (*)"
    else:
        return "3D Model Files (*.stl *.obj);;STL Files (*.stl);;OBJ Files (*.obj);;All Files (*)"


def trimesh_to_vtk_polydata(trimesh_mesh):
    """
    Convert trimesh mesh object to VTK PolyData.

    This function converts a trimesh mesh (after Scene processing and unit_scale)
    directly to VTK PolyData, preserving all transformations.

    Args:
        trimesh_mesh: trimesh.Trimesh - Trimesh mesh object

    Returns:
        tuple: (poly_data, volume)
            - poly_data: vtkPolyData - VTK mesh
            - volume: float - Mesh volume (m^3)

    Example:
        >>> import trimesh
        >>> mesh = trimesh.load('model.stl')
        >>> mesh.apply_scale(0.001)  # mm to m
        >>> poly_data, volume = trimesh_to_vtk_polydata(mesh)
    """
    import vtk
    import numpy as np

    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for vertex in trimesh_mesh.vertices:
        vtk_points.InsertNextPoint(vertex)

    # Create VTK cells (triangles)
    vtk_cells = vtk.vtkCellArray()
    for face in trimesh_mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, int(face[0]))
        triangle.GetPointIds().SetId(1, int(face[1]))
        triangle.GetPointIds().SetId(2, int(face[2]))
        vtk_cells.InsertNextCell(triangle)

    # Create PolyData
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(vtk_cells)

    # Calculate volume
    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(poly_data)
    mass_properties.Update()
    volume = mass_properties.GetVolume()

    return poly_data, volume


def load_mesh_to_polydata(file_path):
    """
    Load mesh file (STL/OBJ/DAE) and return VTK PolyData.

    Supports:
    - STL: VTK native reader
    - OBJ: VTK native reader
    - DAE: trimesh library (with color extraction)

    Args:
        file_path: str - Path to mesh file

    Returns:
        tuple: (poly_data, volume, extracted_color)
            - poly_data: vtkPolyData - Loaded mesh
            - volume: float - Mesh volume (m^3)
            - extracted_color: list or None - RGBA color [r, g, b, a] (0-1 range) for DAE files

    Raises:
        ImportError: If DAE file is loaded but trimesh is not available
        FileNotFoundError: If file does not exist

    Example:
        >>> poly_data, volume, color = load_mesh_to_polydata("model.stl")
        >>> print(f"Volume: {volume:.6f} m^3")
    """
    import os
    import vtk

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    extracted_color = None

    if file_ext == '.dae':
        # Load COLLADA file using trimesh
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh library is required for loading DAE files. Install with: pip install trimesh")

        mesh = trimesh.load(file_path, force='mesh')

        # Extract color information from .dae file
        # Check vertex/face colors first (for colorful display)
        vertex_colors = None
        face_colors = None
        extracted_color = None  # Initialize
        
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            # Get vertex colors
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vertex_colors = mesh.visual.vertex_colors
                print(f"Found vertex colors in .dae file: {len(vertex_colors)} vertices")
                
                # If vertex colors exist, set representative color (first vertex or average) as extracted_color
                import numpy as np
                if not isinstance(vertex_colors, np.ndarray):
                    vertex_colors = np.array(vertex_colors)
                
                if len(vertex_colors) > 0:
                    # Use first vertex color (or compute average)
                    first_color = vertex_colors[0]
                    # Convert from 0-255 range to 0-1 range
                    if isinstance(first_color, np.ndarray):
                        if first_color.max() > 1.0:
                            # 0-255 range
                            extracted_color = [
                                float(first_color[0]) / 255.0,
                                float(first_color[1]) / 255.0,
                                float(first_color[2]) / 255.0,
                                float(first_color[3]) / 255.0 if len(first_color) > 3 else 1.0
                            ]
                        else:
                            # 0-1 range
                            extracted_color = [
                                float(first_color[0]),
                                float(first_color[1]),
                                float(first_color[2]),
                                float(first_color[3]) if len(first_color) > 3 else 1.0
                            ]
                    else:
                        # List or tuple
                        if max(first_color[:3]) > 1.0:
                            extracted_color = [
                                float(first_color[0]) / 255.0,
                                float(first_color[1]) / 255.0,
                                float(first_color[2]) / 255.0,
                                float(first_color[3]) / 255.0 if len(first_color) > 3 else 1.0
                            ]
                        else:
                            extracted_color = [
                                float(first_color[0]),
                                float(first_color[1]),
                                float(first_color[2]),
                                float(first_color[3]) if len(first_color) > 3 else 1.0
                            ]
                    print(f"Extracted representative color from vertex colors: RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")
            
            # Get face colors (when vertex colors absent)
            if vertex_colors is None:
                if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                    face_colors = mesh.visual.face_colors
                    print(f"Found face colors in .dae file: {len(face_colors)} faces")
                    
                    # If face colors exist, set representative color (first face) as extracted_color
                    import numpy as np
                    if not isinstance(face_colors, np.ndarray):
                        face_colors = np.array(face_colors)
                    
                    if len(face_colors) > 0:
                        # Use first face color
                        first_color = face_colors[0]
                        # Convert from 0-255 range to 0-1 range
                        if isinstance(first_color, np.ndarray):
                            if first_color.max() > 1.0:
                                # 0-255 range
                                extracted_color = [
                                    float(first_color[0]) / 255.0,
                                    float(first_color[1]) / 255.0,
                                    float(first_color[2]) / 255.0,
                                    float(first_color[3]) / 255.0 if len(first_color) > 3 else 1.0
                                ]
                            else:
                                # 0-1 range
                                extracted_color = [
                                    float(first_color[0]),
                                    float(first_color[1]),
                                    float(first_color[2]),
                                    float(first_color[3]) if len(first_color) > 3 else 1.0
                                ]
                        else:
                            # List or tuple
                            if max(first_color[:3]) > 1.0:
                                extracted_color = [
                                    float(first_color[0]) / 255.0,
                                    float(first_color[1]) / 255.0,
                                    float(first_color[2]) / 255.0,
                                    float(first_color[3]) / 255.0 if len(first_color) > 3 else 1.0
                                ]
                            else:
                                extracted_color = [
                                    float(first_color[0]),
                                    float(first_color[1]),
                                    float(first_color[2]),
                                    float(first_color[3]) if len(first_color) > 3 else 1.0
                                ]
                        print(f"Extracted representative color from face colors: RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")
            
            # If no vertex/face colors, get single color from material
            if extracted_color is None:
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    if hasattr(mesh.visual.material, 'diffuse'):
                        diffuse = mesh.visual.material.diffuse
                        if diffuse is not None and len(diffuse) >= 3:
                            # Normalize RGBA to 0-1 range
                            extracted_color = [
                                diffuse[0] / 255.0,
                                diffuse[1] / 255.0,
                                diffuse[2] / 255.0,
                                diffuse[3] / 255.0 if len(diffuse) > 3 else 1.0
                            ]
                            print(f"Extracted color from material diffuse: RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")

        # Convert trimesh to VTK PolyData
        vertices = mesh.vertices
        faces = mesh.faces

        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex)

        triangles = vtk.vtkCellArray()
        for face in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, int(face[0]))
            triangle.GetPointIds().SetId(1, int(face[1]))
            triangle.GetPointIds().SetId(2, int(face[2]))
            triangles.InsertNextCell(triangle)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(triangles)

        # Set vertex colors on VTK PolyData
        if vertex_colors is not None and len(vertex_colors) > 0:
            import numpy as np
            # Convert vertex colors to VTK format
            vtk_colors = vtk.vtkUnsignedCharArray()
            vtk_colors.SetNumberOfComponents(3)  # RGB
            vtk_colors.SetName("Colors")
            
            # Convert to numpy array
            if not isinstance(vertex_colors, np.ndarray):
                vertex_colors = np.array(vertex_colors)
            
            # Check vertex count matches color count
            num_vertices = len(vertices)
            num_colors = len(vertex_colors)
            
            if num_colors == num_vertices:
                # Set color for each vertex
                for i in range(num_vertices):
                    color = vertex_colors[i]
                    # For numpy array, access elements directly
                    if isinstance(color, np.ndarray):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    elif isinstance(color, (list, tuple)):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    else:
                        r = g = b = 255  # Default white
                    
                    # Scale to 0-255 if value is in 0-1 range
                    if r <= 1.0 and g <= 1.0 and b <= 1.0:
                        r = int(r * 255)
                        g = int(g * 255)
                        b = int(b * 255)
                    
                    vtk_colors.InsertNextTuple3(r, g, b)
                
                poly_data.GetPointData().SetScalars(vtk_colors)
                print(f"Applied vertex colors to VTK PolyData: {num_colors} vertices")
            else:
                print(f"Warning: Vertex count ({num_vertices}) doesn't match color count ({num_colors})")
        
        # Set face colors on VTK PolyData (when vertex colors absent)
        elif face_colors is not None and len(face_colors) > 0:
            import numpy as np
            # Convert face colors to VTK format
            vtk_colors = vtk.vtkUnsignedCharArray()
            vtk_colors.SetNumberOfComponents(3)  # RGB
            vtk_colors.SetName("Colors")
            
            # Convert to numpy array
            if not isinstance(face_colors, np.ndarray):
                face_colors = np.array(face_colors)
            
            # Check face count matches color count
            num_faces = len(faces)
            num_colors = len(face_colors)
            
            if num_colors == num_faces:
                # Set color for each face
                for i in range(num_faces):
                    color = face_colors[i]
                    # For numpy array, access elements directly
                    if isinstance(color, np.ndarray):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    elif isinstance(color, (list, tuple)):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    else:
                        r = g = b = 255  # Default white
                    
                    # Multiply by 255 if in 0-1 range
                    if r <= 1.0 and g <= 1.0 and b <= 1.0:
                        r = int(r * 255)
                        g = int(g * 255)
                        b = int(b * 255)
                    
                    vtk_colors.InsertNextTuple3(r, g, b)
                
                poly_data.GetCellData().SetScalars(vtk_colors)
                print(f"Applied face colors to VTK PolyData: {num_colors} faces")
            else:
                print(f"Warning: Face count ({num_faces}) doesn't match color count ({num_colors})")

        # Clean the mesh
        # Save scalar values (vertex/face colors) before Clean for preservation
        saved_point_scalars = poly_data.GetPointData().GetScalars()
        saved_cell_scalars = poly_data.GetCellData().GetScalars()
        
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(poly_data)
        clean.SetTolerance(1e-5)
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.PointMergingOn()
        # Preserve scalar values (default, but set explicitly)
        clean.Update()

        poly_data = clean.GetOutput()
        
        # Restore scalar values after Clean (if needed)
        # Note: vtkCleanPolyData may merge vertices, so scalar remapping may be needed
        if saved_point_scalars is not None:
            # If vertex colors exist, restore to match post-Clean vertex count
            num_points_after = poly_data.GetNumberOfPoints()
            num_points_before = saved_point_scalars.GetNumberOfTuples()
            
            if num_points_after == num_points_before:
                # If vertex count unchanged, restore as-is
                poly_data.GetPointData().SetScalars(saved_point_scalars)
                print(f"Restored vertex colors after cleaning: {num_points_after} vertices")
            else:
                # If vertex count changed, use first N colors
                vtk_colors = vtk.vtkUnsignedCharArray()
                vtk_colors.SetNumberOfComponents(3)
                vtk_colors.SetName("Colors")
                for i in range(min(num_points_after, num_points_before)):
                    color = saved_point_scalars.GetTuple3(i)
                    vtk_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))
                # Use last color for remaining vertices
                if num_points_after > num_points_before and num_points_before > 0:
                    last_color = saved_point_scalars.GetTuple3(num_points_before - 1)
                    for i in range(num_points_before, num_points_after):
                        vtk_colors.InsertNextTuple3(int(last_color[0]), int(last_color[1]), int(last_color[2]))
                poly_data.GetPointData().SetScalars(vtk_colors)
                print(f"Adjusted vertex colors after cleaning: {num_points_before} -> {num_points_after} vertices")
        
        if saved_cell_scalars is not None:
            # If face colors exist, restore to match post-Clean face count
            num_cells_after = poly_data.GetNumberOfCells()
            num_cells_before = saved_cell_scalars.GetNumberOfTuples()
            
            if num_cells_after == num_cells_before:
                # If face count unchanged, restore as-is
                poly_data.GetCellData().SetScalars(saved_cell_scalars)
                print(f"Restored face colors after cleaning: {num_cells_after} cells")
            else:
                # If face count changed, use first N colors
                vtk_colors = vtk.vtkUnsignedCharArray()
                vtk_colors.SetNumberOfComponents(3)
                vtk_colors.SetName("Colors")
                for i in range(min(num_cells_after, num_cells_before)):
                    color = saved_cell_scalars.GetTuple3(i)
                    vtk_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))
                # Use last color for remaining faces
                if num_cells_after > num_cells_before and num_cells_before > 0:
                    last_color = saved_cell_scalars.GetTuple3(num_cells_before - 1)
                    for i in range(num_cells_before, num_cells_after):
                        vtk_colors.InsertNextTuple3(int(last_color[0]), int(last_color[1]), int(last_color[2]))
                poly_data.GetCellData().SetScalars(vtk_colors)
                print(f"Adjusted face colors after cleaning: {num_cells_before} -> {num_cells_after} cells")

    elif file_ext == '.obj':
        # Load OBJ file using VTK
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_path)
        reader.Update()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(reader.GetOutputPort())
        clean.SetTolerance(1e-5)
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.PointMergingOn()
        clean.Update()

        poly_data = clean.GetOutput()

    else:
        # Load STL file using VTK (default)
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(reader.GetOutputPort())
        clean.SetTolerance(1e-5)
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.PointMergingOn()
        clean.Update()

        poly_data = clean.GetOutput()

    # Calculate volume
    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(poly_data)
    volume = mass_properties.GetVolume()

    return poly_data, volume, extracted_color


def save_polydata_to_mesh(file_path, poly_data, mesh_color=None, color_manually_changed=False):
    """
    Save VTK PolyData to mesh file (STL/OBJ/DAE).

    Supports:
    - STL: Binary format
    - OBJ: Standard format
    - DAE: With color preservation (requires trimesh)

    Args:
        file_path: str - Output file path
        poly_data: vtkPolyData - Mesh data to save
        mesh_color: list or None - RGBA color [r, g, b, a] (0-1 range) for DAE export
        color_manually_changed: bool - If True, apply mesh_color to DAE file

    Raises:
        ImportError: If DAE file is requested but trimesh is not available

    Example:
        >>> import vtk
        >>> # Create or load poly_data
        >>> save_polydata_to_mesh("output.stl", poly_data)
        >>> # Save DAE with color
        >>> save_polydata_to_mesh("output.dae", poly_data,
        ...                       mesh_color=[1.0, 0.0, 0.0, 1.0],
        ...                       color_manually_changed=True)
    """
    import os
    import vtk

    file_ext = os.path.splitext(file_path)[1].lower()

    # Ensure file has an extension
    if not file_ext:
        file_path += '.stl'
        file_ext = '.stl'

    if file_ext == '.dae':
        # Export as COLLADA using trimesh
        try:
            import trimesh
            import numpy as np
        except ImportError:
            raise ImportError("trimesh and numpy are required for DAE export. Install with: pip install trimesh numpy")

        # Convert VTK PolyData to numpy arrays
        num_points = poly_data.GetNumberOfPoints()
        num_cells = poly_data.GetNumberOfCells()

        vertices = np.zeros((num_points, 3))
        for i in range(num_points):
            vertices[i] = poly_data.GetPoint(i)

        faces = []
        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetNumberOfPoints() == 3:
                faces.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)])

        faces = np.array(faces)

        # Create trimesh object and export geometry
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(file_path)

        # Determine diffuse color (0-1 range) for COLLADA material
        if color_manually_changed and mesh_color is not None:
            r = float(mesh_color[0])
            g = float(mesh_color[1])
            b = float(mesh_color[2])
            a = float(mesh_color[3]) if len(mesh_color) > 3 else 1.0
        else:
            r, g, b, a = 1.0, 1.0, 1.0, 1.0

        # Post-process COLLADA XML to embed correct unit and diffuse/ambient color
        import xml.etree.ElementTree as ET
        ns = 'http://www.collada.org/2005/11/COLLADASchema'
        ET.register_namespace('', ns)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Ensure <asset> has correct <unit> element (meters)
        asset = root.find(f'{{{ns}}}asset')
        if asset is not None:
            unit = asset.find(f'{{{ns}}}unit')
            if unit is None:
                unit = ET.SubElement(asset, f'{{{ns}}}unit')
            unit.set('name', 'meter')
            unit.set('meter', '1.0')

        color_str = f"{r:.6f} {g:.6f} {b:.6f} {a:.1f}"
        ambient_str = f"{r * 0.3:.6f} {g * 0.3:.6f} {b * 0.3:.6f} {a:.1f}"
        specular_str = f"{0.196:.6f} {0.196:.6f} {0.196:.6f} 1.0"

        for phong in root.iter(f'{{{ns}}}phong'):
            # Set diffuse
            diffuse_el = phong.find(f'{{{ns}}}diffuse')
            if diffuse_el is not None:
                color_el = diffuse_el.find(f'{{{ns}}}color')
                if color_el is not None:
                    color_el.text = color_str

            # Set ambient
            ambient_el = phong.find(f'{{{ns}}}ambient')
            if ambient_el is not None:
                color_el = ambient_el.find(f'{{{ns}}}color')
                if color_el is not None:
                    color_el.text = ambient_str

            # Set specular
            specular_el = phong.find(f'{{{ns}}}specular')
            if specular_el is not None:
                color_el = specular_el.find(f'{{{ns}}}color')
                if color_el is not None:
                    color_el.text = specular_str

            # Set shininess
            shininess_el = phong.find(f'{{{ns}}}shininess')
            if shininess_el is not None:
                float_el = shininess_el.find(f'{{{ns}}}float')
                if float_el is not None:
                    float_el.text = "20.0"

        tree.write(file_path, xml_declaration=False, encoding='unicode')

    elif file_ext == '.obj':
        # Export as OBJ using VTK
        obj_writer = vtk.vtkOBJWriter()
        obj_writer.SetFileName(file_path)
        obj_writer.SetInputData(poly_data)
        obj_writer.Write()

    else:
        # Export as STL using VTK (default)
        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetFileName(file_path)
        stl_writer.SetInputData(poly_data)
        stl_writer.SetFileTypeToBinary()
        stl_writer.Write()


# ============================================================================
# Qt Application Setup Utilities
# ============================================================================

def setup_signal_handlers(verbose=True, include_sigterm=True):
    """
    Setup signal handlers for graceful application shutdown.

    Args:
        verbose: If True, print message when signal received
        include_sigterm: If True, also handle SIGTERM signal

    Usage:
        setup_signal_handlers()  # Call before QApplication.exec()
    """
    import signal
    from PySide6.QtWidgets import QApplication

    def signal_handler(signum, frame):
        """Signal handler for SIGINT (Ctrl+C) and SIGTERM"""
        if verbose:
            print("\nCtrl+C detected, closing application...")
        app = QApplication.instance()
        if app:
            app.quit()

    signal.signal(signal.SIGINT, signal_handler)
    if include_sigterm:
        signal.signal(signal.SIGTERM, signal_handler)


def setup_signal_processing_timer(app, interval_ms=500):
    """
    Create a QTimer to allow Python signal processing in Qt event loop.

    Qt's event loop can block signal processing, so this timer periodically
    interrupts the loop to allow signals like SIGINT to be processed.

    Args:
        app: QApplication instance
        interval_ms: Timer interval in milliseconds (default 500)

    Returns:
        QTimer instance (keep reference to prevent garbage collection)

    Usage:
        app = QApplication(sys.argv)
        timer = setup_signal_processing_timer(app)
        # ... rest of setup ...
        sys.exit(app.exec())
    """
    from PySide6.QtCore import QTimer

    timer = QTimer()
    timer.start(interval_ms)
    timer.timeout.connect(lambda: None)  # Dummy function to interrupt event loop
    return timer


def is_apple_silicon():
    """
    Detect if running on Apple Silicon (M1/M2/M3/M4).

    Returns:
        bool: True if running on Apple Silicon, False otherwise
    """
    import platform
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


def setup_qt_environment_for_apple_silicon():
    """
    Configure Qt environment variables for Apple Silicon compatibility.

    Should be called before creating QApplication on Apple Silicon Macs.

    Usage:
        if is_apple_silicon():
            setup_qt_environment_for_apple_silicon()
        app = QApplication(sys.argv)
    """
    import os
    os.environ['QT_MAC_WANTS_LAYER'] = '1'


# VTK Display Stylesheet Constant
VTK_DISPLAY_STYLESHEET = """
    QLabel {
        background-color: #1a1a1a;
        border: 2px solid #555;
    }
    QLabel:focus {
        border: 2px solid #00aaff;
    }
"""


def create_vtk_display_widget(parent, min_width=800, min_height=600, text="3D View\n\n(Load file to display)"):
    """
    Create a QLabel widget configured for VTK rendering display.

    Args:
        parent: Parent widget
        min_width: Minimum width in pixels (default 800)
        min_height: Minimum height in pixels (default 600)
        text: Initial text to display (default "3D View\\n\\n(Load file to display)")

    Returns:
        QLabel: Configured widget ready for VTK rendering

    Usage:
        self.vtk_display = create_vtk_display_widget(self, min_width=800, min_height=600)
        layout.addWidget(self.vtk_display)
    """
    from PySide6.QtWidgets import QLabel
    from PySide6.QtCore import Qt

    vtk_display = QLabel(parent)
    vtk_display.setMinimumSize(min_width, min_height)
    vtk_display.setStyleSheet(VTK_DISPLAY_STYLESHEET)
    vtk_display.setAlignment(Qt.AlignCenter)
    vtk_display.setText(text)
    vtk_display.setScaledContents(False)
    vtk_display.setFocusPolicy(Qt.StrongFocus)

    return vtk_display


def setup_dark_theme(app, theme='default', custom_styles=None):
    """
    Apply dark theme to Qt application with optional theme presets.

    Args:
        app: QApplication instance
        theme: Theme preset - 'default', 'parts_editor', 'assembler', 'mesh_sourcer'
        custom_styles: Optional dict of custom stylesheet rules to append

    Usage:
        app = QApplication(sys.argv)
        setup_dark_theme(app, theme='parts_editor')
        # Or with custom styles:
        setup_dark_theme(app, theme='assembler', custom_styles={
            'QTextEdit': 'background-color: #2a2a2a;'
        })
    """
    from PySide6.QtGui import QPalette, QColor

    if theme == 'parts_editor':
        # PartsEditor theme: warm gray tones with extensive styling
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

        stylesheet = """
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
            }
        """
        app.setStyleSheet(stylesheet)

    elif theme == 'assembler':
        # Assembler theme: cooler dark tones with minimal styling
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(42, 42, 42))
        palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        palette.setColor(QPalette.PlaceholderText, QColor(180, 180, 180))  # Placeholder text color
        app.setPalette(palette)

        # Set button pressed color in stylesheet (deep blue regardless of focus)
        stylesheet = """
            QPushButton:pressed {
                background-color: #1a3a5a;
                border: 1px solid #2a5a8a;
                color: #ffffff;
            }
            QPushButton:pressed:!active {
                background-color: #1a3a5a;
                border: 1px solid #2a5a8a;
                color: #ffffff;
            }
            QPushButton:checked:!active {
                background-color: #1a3a5a;
                border: 1px solid #2a5a8a;
                color: #ffffff;
            }
            QCheckBox::indicator:checked:!active {
                background-color: #1a3a5a;
                border: 1px solid #2a5a8a;
            }
            QRadioButton::indicator:checked:!active {
                background-color: #1a3a5a;
                border: 1px solid #2a5a8a;
            }
        """
        app.setStyleSheet(stylesheet)

    elif theme == 'mesh_sourcer':
        # MeshSourcer theme: similar to assembler but may have custom tweaks
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(42, 42, 42))
        palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        app.setPalette(palette)

        # Set button pressed color in stylesheet (deep blue regardless of focus)
        stylesheet = """
            QPushButton:pressed {
                background-color: #1a3a5a;
                border: 1px solid #2a5a8a;
            }
        """
        app.setStyleSheet(stylesheet)

    else:  # 'default' or any other value
        # Default dark theme (same as assembler)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(42, 42, 42))
        palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        app.setPalette(palette)

        # Set button pressed color in stylesheet (deep blue regardless of focus)
        stylesheet = """
            QPushButton:pressed {
                background-color: #1a3a5a;
                border: 1px solid #2a5a8a;
            }
        """
        app.setStyleSheet(stylesheet)

    # Apply custom styles if provided
    if custom_styles:
        additional_stylesheet = "\n".join([f"{selector} {{ {rules} }}"
                                          for selector, rules in custom_styles.items()])
        current_stylesheet = app.styleSheet()
        app.setStyleSheet(current_stylesheet + "\n" + additional_stylesheet)


# ==============================================================================
# Physical Properties Mirroring (Y-axis)
# ==============================================================================

def mirror_physical_properties_y_axis(inertia_dict, inertial_origin_dict):
    """
    Transform physical properties (inertia tensor and center of mass) for Y-axis mirroring.

    This function correctly mirrors physical properties of left-right symmetric parts.
    Standardized from the method used in PartsEditor.

    Y-axis mirroring transformations:
    - Center of mass (COM): Invert Y coordinate [x, y, z] -> [x, -y, z]
    - Inertia tensor:
      * Diagonal (ixx, iyy, izz): unchanged
      * Off-diagonal: negate ixy and iyz, ixz unchanged

    Args:
        inertia_dict: dict - Inertia tensor dict
            {'ixx': float, 'iyy': float, 'izz': float,
             'ixy': float, 'ixz': float, 'iyz': float}
        inertial_origin_dict: dict - Center of mass dict
            {'xyz': [x, y, z], 'rpy': [r, p, y]}

    Returns:
        tuple: (mirrored_inertia_dict, mirrored_inertial_origin_dict)

    Example:
        >>> inertia = {'ixx': 0.001, 'iyy': 0.002, 'izz': 0.003,
        ...            'ixy': 0.0001, 'ixz': 0.0002, 'iyz': 0.0003}
        >>> origin = {'xyz': [0.1, 0.2, 0.3], 'rpy': [0, 0, 0]}
        >>> mir_inertia, mir_origin = mirror_physical_properties_y_axis(inertia, origin)
        >>> print(mir_origin['xyz'])  # [0.1, -0.2, 0.3]
        >>> print(mir_inertia['ixy'])  # -0.0001
    """
    # Mirror inertia tensor
    mirrored_inertia = {}
    if inertia_dict is not None and isinstance(inertia_dict, dict):
        mirrored_inertia = inertia_dict.copy()
        # Y-axis mirroring: negate ixy and iyz
        if 'ixy' in mirrored_inertia:
            mirrored_inertia['ixy'] = -inertia_dict['ixy']
        if 'iyz' in mirrored_inertia:
            mirrored_inertia['iyz'] = -inertia_dict['iyz']
        # ixx, iyy, izz, ixz unchanged

    # Mirror center of mass (COM)
    mirrored_origin = {}
    if inertial_origin_dict is not None and isinstance(inertial_origin_dict, dict):
        mirrored_origin = inertial_origin_dict.copy()
        if 'xyz' in mirrored_origin and len(mirrored_origin['xyz']) >= 3:
            xyz = mirrored_origin['xyz']
            mirrored_origin['xyz'] = [xyz[0], -xyz[1], xyz[2]]
        # RPY unchanged (rotation handling is complex; adjust if needed)

    return mirrored_inertia, mirrored_origin


def mirror_inertia_tensor_left_right(inertia_dict):
    """
    Negate inertia tensor components for left-right flip (Y-axis mirroring).

    Used when creating left-right symmetric parts (l_* and r_*).
    For Y-axis mirroring, negate:
    - ixy: negate
    - iyz: negate
    - ixx, iyy, izz, ixz: unchanged

    Args:
        inertia_dict: dict - Inertia tensor dict
            {'ixx': float, 'iyy': float, 'izz': float,
             'ixy': float, 'ixz': float, 'iyz': float}

    Returns:
        dict: Mirrored inertia tensor dict (new dict, original unchanged)

    Example:
        >>> inertia = {'ixx': 0.001, 'iyy': 0.002, 'izz': 0.003,
        ...            'ixy': 0.0001, 'ixz': 0.0002, 'iyz': 0.0003}
        >>> mirrored = mirror_inertia_tensor_left_right(inertia)
        >>> print(mirrored['ixy'])  # -0.0001
        >>> print(mirrored['iyz'])  # -0.0003
        >>> print(mirrored['ixx'])  # 0.001 (unchanged)
    """
    if inertia_dict is None or not isinstance(inertia_dict, dict):
        return None

    # Create new dict (do not modify original)
    mirrored_inertia = inertia_dict.copy()

    # Y-axis mirroring: negate ixy and iyz
    if 'ixy' in mirrored_inertia:
        mirrored_inertia['ixy'] = -inertia_dict['ixy']
    if 'iyz' in mirrored_inertia:
        mirrored_inertia['iyz'] = -inertia_dict['iyz']
    # ixx, iyy, izz, ixz unchanged

    return mirrored_inertia


def mirror_center_of_mass_left_right(center_of_mass):
    """
    Negate Y coordinate of center of mass for left-right flip (Y-axis mirroring).

    Used when creating left-right symmetric parts (l_* and r_*).
    For Y-axis mirroring, negate only Y: [x, y, z] -> [x, -y, z]

    Args:
        center_of_mass: list or array-like - Center of mass coordinates [x, y, z]

    Returns:
        list: Mirrored center of mass [x, -y, z] (new list, original unchanged)

    Example:
        >>> com = [0.1, 0.2, 0.3]
        >>> mirrored_com = mirror_center_of_mass_left_right(com)
        >>> print(mirrored_com)  # [0.1, -0.2, 0.3]
    """
    if center_of_mass is None:
        return None

    # Convert list or array to list
    if hasattr(center_of_mass, '__iter__') and not isinstance(center_of_mass, str):
        com_list = list(center_of_mass)
        if len(com_list) >= 3:
            # Create new list (do not modify original)
            return [com_list[0], -com_list[1], com_list[2]]
        else:
            return com_list.copy()
    else:
        return center_of_mass


def calculate_mirrored_physical_properties_from_mesh(mesh_file_path, mass, density=1.0):
    """
    Compute physical properties from Y-axis mirrored mesh file.

    Mirrors STL/OBJ/DAE mesh along Y-axis and recomputes physical properties
    (volume, center of mass, inertia tensor) from the mirrored geometry.
    Standard method used in PartsEditor.

    Args:
        mesh_file_path: str - Path to mesh file (.stl, .obj, .dae)
        mass: float - Mass (kg)
        density: float - Density (kg/m³), used when mass is not specified

    Returns:
        dict: Mirrored physical properties
            {
                'volume': float,
                'mass': float,
                'center_of_mass': [x, y, z],
                'inertia': {
                    'ixx': float, 'iyy': float, 'izz': float,
                    'ixy': float, 'ixz': float, 'iyz': float
                }
            }
        Returns None on error.

    Example:
        >>> props = calculate_mirrored_physical_properties_from_mesh(
        ...     'l_arm.stl', mass=0.5)
        >>> print(props['center_of_mass'])  # Center of mass with Y inverted
        >>> print(props['inertia']['ixy'])  # Negated off-diagonal component
    """
    import vtk
    import numpy as np

    try:
        # Load mesh file
        poly_data, volume_unused, extracted_color = load_mesh_to_polydata(mesh_file_path)

        # Set Y-axis flip transform
        transform = vtk.vtkTransform()
        transform.Scale(1, -1, 1)

        # Apply transform
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

        mirrored_poly_data = normal_generator.GetOutput()

        # Compute volume
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(mirrored_poly_data)
        mass_properties.Update()
        volume = mass_properties.GetVolume()

        # Compute mass from volume * density if not specified
        if mass is None or mass <= 0:
            mass = volume * density

        # Compute center of mass
        com_filter = vtk.vtkCenterOfMass()
        com_filter.SetInputData(mirrored_poly_data)
        com_filter.SetUseScalarsAsWeights(False)
        com_filter.Update()
        center_of_mass = list(com_filter.GetCenter())

        # Compute inertia tensor (is_mirrored=True accounts for mirroring)
        inertia_tensor = calculate_inertia_tensor(
            mirrored_poly_data, mass, center_of_mass, is_mirrored=True
        )

        # Return as dict
        return {
            'volume': volume,
            'mass': mass,
            'center_of_mass': center_of_mass,
            'inertia': {
                'ixx': float(inertia_tensor[0, 0]),
                'iyy': float(inertia_tensor[1, 1]),
                'izz': float(inertia_tensor[2, 2]),
                'ixy': float(inertia_tensor[0, 1]),
                'ixz': float(inertia_tensor[0, 2]),
                'iyz': float(inertia_tensor[1, 2])
            }
        }

    except Exception as e:
        print(f"Error calculating mirrored properties from mesh: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# VTK VIEWER BASE CLASS
# ============================================================================

class VTKViewerBase:
    """
    Base class for VTK viewer applications (PartsEditor and MeshSourcer).

    Provides common functionality for:
    - Camera control (rotation, mouse interaction)
    - Point manipulation (move, hide, display)
    - VTK rendering coordination

    Subclasses must initialize these attributes before calling methods:
    - self.renderer: vtk.vtkRenderer
    - self.camera_rotation: list [yaw, pitch, roll]
    - self.absolute_origin: list [x, y, z]
    - self.point_actors: list of vtkAssembly
    - self.point_coords: list of [x, y, z]
    - self.point_inputs: list of UI input fields
    - self.camera_controller: CameraController instance

    Subclasses must implement:
    - render_to_image(): Update the display after rendering
    - _update_point_visibility(index): Optional, customize point visibility logic
    """

    # Class variable for camera rotation sensitivity
    CAMERA_ROTATION_SENSITIVITY = 0.5

    def setup_vtk(self):
        """
        Setup VTK offscreen rendering (Mac compatible).

        Initializes renderer, render window, and utility objects.
        Subclasses may override to add additional initialization.

        Requires these attributes to be set before calling:
        - self.absolute_origin: focal point for camera
        - self.vtk_display: QLabel widget for mouse tracking
        """
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(800, 600)
        self.render_window.AddRenderer(self.renderer)
        self.render_window_interactor = None

        # Initialize utility classes
        self.offscreen_renderer = OffscreenRenderer(self.render_window, self.renderer)
        self.camera_controller = CameraController(self.renderer, self.absolute_origin)
        self.animated_rotation = AnimatedCameraRotation(self.renderer, self.absolute_origin)
        self.mouse_state = MouseDragState(self.vtk_display)

    def setup_camera(self):
        """
        Setup camera initial configuration.

        Requires these attributes to be set before calling:
        - self.absolute_origin: focal point for camera
        - self.initial_camera_position: relative camera position [x, y, z]
        - self.initial_camera_view_up: camera up vector [x, y, z]
        """
        position = [self.absolute_origin[i] + self.initial_camera_position[i] for i in range(3)]
        self.camera_controller.setup_parallel_camera(
            position=position,
            view_up=self.initial_camera_view_up,
            focal_point=self.absolute_origin,
            parallel_scale=5
        )

    def apply_camera_rotation(self, camera):
        """
        Apply accumulated camera rotation to the given camera.

        Uses self.camera_rotation [yaw, pitch, roll] and self.absolute_origin
        to rotate the camera around the focal point.

        Args:
            camera: vtkCamera instance to apply rotation to
        """
        # Get current camera position and focal point
        position = list(camera.GetPosition())
        focal_point = self.absolute_origin

        # Create rotation matrix
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Translate(*focal_point)
        transform.RotateZ(self.camera_rotation[2])  # Roll
        transform.RotateX(self.camera_rotation[0])  # Pitch
        transform.RotateY(self.camera_rotation[1])  # Yaw
        transform.Translate(*[-x for x in focal_point])

        # Rotate camera position
        new_position = transform.TransformPoint(position)
        camera.SetPosition(new_position)

        # Update camera up vector
        up = [0, 0, 1]
        new_up = transform.TransformVector(up)
        camera.SetViewUp(new_up)

    def rotate_camera_mouse(self, dx, dy):
        """
        Rotate camera based on mouse drag.

        Args:
            dx: Mouse movement in x direction
            dy: Mouse movement in y direction
        """
        self.camera_controller.rotate_azimuth_elevation(
            dx, dy, sensitivity=self.CAMERA_ROTATION_SENSITIVITY
        )
        self.render_to_image()

    def move_point(self, index, dx, dy, dz):
        """
        Move a point by the specified delta.

        Args:
            index: Point index
            dx, dy, dz: Delta movement in each axis
        """
        new_position = [
            self.point_coords[index][0] + dx,
            self.point_coords[index][1] + dy,
            self.point_coords[index][2] + dz
        ]
        self.point_coords[index] = new_position
        self.update_point_display(index)
        print(f"Point {index+1} moved to: ({new_position[0]:.6f}, {new_position[1]:.6f}, {new_position[2]:.6f})")

    def move_point_screen(self, index, direction, step):
        """
        Move a point in screen-space direction.

        Args:
            index: Point index
            direction: numpy array representing direction vector
            step: Movement step size
        """
        move_vector = direction * step
        new_position = [
            self.point_coords[index][0] + move_vector[0],
            self.point_coords[index][1] + move_vector[1],
            self.point_coords[index][2] + move_vector[2]
        ]
        self.point_coords[index] = new_position
        self.update_point_display(index)
        print(f"Point {index+1} moved to: ({new_position[0]:.6f}, {new_position[1]:.6f}, {new_position[2]:.6f})")

    def hide_point(self, index):
        """
        Hide a point actor.

        Args:
            index: Point index
        """
        if self.point_actors[index]:
            self.point_actors[index].VisibilityOff()
        self.render_to_image()

    def update_point_display(self, index):
        """
        Update point display (position and input fields).

        Uses template method pattern - subclasses can override
        _update_point_visibility() for custom visibility logic.

        Args:
            index: Point index
        """
        if self.point_actors[index]:
            self.point_actors[index].SetPosition(self.point_coords[index])

            # Allow subclasses to customize visibility logic
            self._update_point_visibility(index)

        # Update coordinate input fields
        for i, coord in enumerate(self.point_coords[index]):
            self.point_inputs[index][i].setText(f"{coord:.6f}")

        self.render_to_image()

    def _update_point_visibility(self, index):
        """
        Template method for point visibility customization.

        Default implementation: always show the point.
        Subclasses can override for custom logic (e.g., checkbox-based visibility).

        Args:
            index: Point index
        """
        self.point_actors[index].VisibilityOn()

    def render_to_image(self):
        """
        Update the display after rendering changes.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement render_to_image()")


# ============================================================================
# KITCHEN COLOR PICKER
# ============================================================================

class KitchenColorPicker:
    """
    Reusable color picker widget with RGBA inputs and visual preview.

    This color picker provides:
    - Four RGBA input fields (0-1 range)
    - Visual color sample display with alpha support
    - "Pick" button to open Qt color dialog with alpha channel
    - Callback support for color changes

    Used across PartsEditor, MeshSourcer, and Assembler for consistent
    color selection interface.

    Example:
        >>> from PySide6.QtWidgets import QHBoxLayout
        >>> # Create color picker with RGB only
        >>> color_picker = KitchenColorPicker(
        ...     parent_widget=self,
        ...     initial_color=[1.0, 0.0, 0.0],  # Red, alpha defaults to 1.0
        ...     on_color_changed=self.handle_color_change
        ... )
        >>> # Create color picker with RGBA
        >>> color_picker = KitchenColorPicker(
        ...     parent_widget=self,
        ...     initial_color=[1.0, 0.0, 0.0, 0.5],  # Red with 50% transparency
        ...     enable_alpha=True,
        ...     on_color_changed=self.handle_color_change
        ... )
        >>> # Add to layout
        >>> layout = QHBoxLayout()
        >>> color_picker.add_to_layout(layout)
    """

    def __init__(self, parent_widget, initial_color=None, enable_alpha=True, on_color_changed=None):
        """
        Initialize Kitchen Color Picker.

        Args:
            parent_widget: Parent Qt widget (for dialog parent)
            initial_color: Initial color as list [r, g, b] or [r, g, b, a] (0-1 range), defaults to white
            enable_alpha: If True, show alpha input field and enable alpha in color dialog
            on_color_changed: Callback function(rgba_list) called when color changes
        """
        from PySide6.QtWidgets import QLabel, QPushButton, QLineEdit, QHBoxLayout
        from PySide6.QtGui import QDoubleValidator
        from PySide6.QtCore import QLocale

        self.parent_widget = parent_widget
        self.on_color_changed = on_color_changed
        self.enable_alpha = enable_alpha

        # Set initial color (default to opaque white)
        if initial_color is None:
            initial_color = [1.0, 1.0, 1.0, 1.0]
        elif len(initial_color) == 3:
            initial_color = list(initial_color) + [1.0]  # Add default alpha

        self.current_color = list(initial_color)[:4]  # Ensure RGBA

        # Create UI components
        self.color_inputs = []
        self.color_sample = None
        self.pick_button = None

        self._create_widgets()

    def _create_widgets(self):
        """Create RGBA input fields, color sample, and Pick button."""
        from PySide6.QtWidgets import QLabel, QPushButton, QLineEdit
        from PySide6.QtGui import QDoubleValidator
        from PySide6.QtCore import QLocale

        # Create RGBA input fields (show alpha only if enabled)
        num_inputs = 4 if self.enable_alpha else 3
        for i in range(num_inputs):
            color_input = QLineEdit()
            color_input.setFixedWidth(60)
            color_input.setText(f"{self.current_color[i]:.3f}")

            # Validator for 0-1 range
            validator = QDoubleValidator(0.0, 1.0, 3)
            validator.setLocale(QLocale.c())
            validator.setNotation(QDoubleValidator.StandardNotation)
            color_input.setValidator(validator)

            # Connect to update handler
            color_input.textChanged.connect(self._on_input_changed)

            self.color_inputs.append(color_input)

        # Create color sample display
        self.color_sample = QLabel()
        self.color_sample.setFixedSize(30, 20)
        self._update_color_sample()

        # Create Pick button
        self.pick_button = QPushButton("Pick")
        self.pick_button.clicked.connect(self.show_color_picker)

    def add_to_layout(self, layout):
        """
        Add color picker widgets to a layout.

        Args:
            layout: QLayout to add widgets to (typically QHBoxLayout or QGridLayout)

        Usage:
            >>> color_layout = QHBoxLayout()
            >>> color_picker.add_to_layout(color_layout)
            >>> parent_layout.addLayout(color_layout)
        """
        for color_input in self.color_inputs:
            layout.addWidget(color_input)
        layout.addWidget(self.color_sample)
        layout.addWidget(self.pick_button)

    def _update_color_sample(self):
        """Update the color sample display to match current RGBA values."""
        try:
            rgba = [max(0, min(255, int(c * 255))) for c in self.current_color]

            if self.enable_alpha:
                # Use rgba() with alpha channel
                self.color_sample.setStyleSheet(
                    f"background-color: rgba({rgba[0]},{rgba[1]},{rgba[2]},{self.current_color[3]:.3f}); "
                    f"border: 1px solid black;"
                )
            else:
                # Use rgb() without alpha
                self.color_sample.setStyleSheet(
                    f"background-color: rgb({rgba[0]},{rgba[1]},{rgba[2]}); "
                    f"border: 1px solid black;"
                )
        except (ValueError, IndexError):
            pass

    def _on_input_changed(self):
        """Handle RGBA input field changes."""
        try:
            # Read values from input fields
            new_color = [float(input_field.text()) for input_field in self.color_inputs]

            # If alpha is disabled, preserve current alpha value
            if not self.enable_alpha:
                new_color.append(self.current_color[3])

            # Validate range
            new_color = [max(0.0, min(1.0, value)) for value in new_color]

            # Update internal state
            self.current_color = new_color

            # Update visual sample
            self._update_color_sample()

            # Trigger callback
            if self.on_color_changed:
                self.on_color_changed(self.current_color)

        except ValueError:
            # Invalid input, skip update
            pass

    def show_color_picker(self):
        """
        Show CustomColorDialog with selection border for custom colors.

        Opens a custom color dialog with the current color selected.
        Updates RGBA inputs and triggers callback if a new color is chosen.
        Supports alpha channel if enable_alpha is True.
        """
        from PySide6.QtGui import QColor

        try:
            # Create QColor from current RGBA values
            current_qcolor = QColor(
                *[min(255, max(0, int(c * 255))) for c in self.current_color]
            )
        except (ValueError, IndexError):
            current_qcolor = QColor(255, 255, 255, 255)

        # Use CustomColorDialog instead of standard QColorDialog
        dialog = CustomColorDialog(current_qcolor, self.parent_widget)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)
        if self.enable_alpha:
            dialog.setOption(QtWidgets.QColorDialog.ShowAlphaChannel, True)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            color = dialog.currentColor()
            if color.isValid():
                # Update only RGB values, preserve existing alpha
                new_color = [
                    color.red() / 255.0,
                    color.green() / 255.0,
                    color.blue() / 255.0,
                    self.current_color[3]  # Preserve existing alpha
                ]

                # Update input fields (only RGB - never update alpha from color picker)
                for i in range(3):
                    self.color_inputs[i].setText(f"{new_color[i]:.3f}")

                # Update internal state
                self.current_color = new_color

                # Update visual sample
                self._update_color_sample()

                # Trigger callback
                if self.on_color_changed:
                    self.on_color_changed(self.current_color)

    def get_color(self):
        """
        Get current RGB color (without alpha).

        Returns:
            list: Current color as [r, g, b] in 0-1 range
        """
        return self.current_color[:3].copy()

    def get_color_rgba(self):
        """
        Get current color as RGBA.

        Returns:
            list: Color as [r, g, b, a] in 0-1 range
        """
        return self.current_color.copy()

    def get_alpha(self):
        """
        Get current alpha value.

        Returns:
            float: Alpha value in 0-1 range
        """
        return self.current_color[3]

    def set_color(self, color):
        """
        Set color programmatically (RGB or RGBA).

        Args:
            color: Color as list [r, g, b] or [r, g, b, a] in 0-1 range
        """
        # Convert to RGBA if RGB provided
        if len(color) == 3:
            new_color = list(color) + [self.current_color[3]]  # Preserve alpha
        else:
            new_color = list(color)[:4]

        # Validate and clamp values
        new_color = [max(0.0, min(1.0, float(c))) for c in new_color]

        # Update input fields (only RGB if alpha is disabled)
        num_inputs = 4 if self.enable_alpha else 3
        for i in range(num_inputs):
            self.color_inputs[i].setText(f"{new_color[i]:.3f}")

        # Update internal state
        self.current_color = new_color

        # Update visual sample
        self._update_color_sample()

    def set_alpha(self, alpha):
        """
        Set alpha value programmatically.

        Args:
            alpha: Alpha value in 0-1 range
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        self.current_color[3] = alpha

        # Update alpha input field if enabled
        if self.enable_alpha and len(self.color_inputs) >= 4:
            self.color_inputs[3].setText(f"{alpha:.3f}")

        # Update visual sample
        self._update_color_sample()


# ============================================================================
# COORDINATE CONVERSION UTILITIES
# ============================================================================

class ConversionUtils:
    """Coordinate and rotation conversion utilities for URDF/MJCF parsing.

    This class provides static methods for converting between different
    rotation representations (quaternions, RPY, Euler angles).
    """

    @staticmethod
    def rpy_to_quat(rpy):
        """Convert RPY (roll, pitch, yaw) to quaternion (w, x, y, z).

        Args:
            rpy: List of [roll, pitch, yaw] in radians

        Returns:
            List of [w, x, y, z] quaternion (MuJoCo convention)
        """
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

        # Calculate half angles
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        # Quaternion components
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [w, x, y, z]

    @staticmethod
    def quat_to_rpy(quat):
        """Convert quaternion (w, x, y, z) to RPY (roll, pitch, yaw).

        Args:
            quat: List of [w, x, y, z] quaternion (MuJoCo convention)

        Returns:
            List of [roll, pitch, yaw] in radians
        """
        # Use rotation matrix extraction to stay consistent with rpy_to_quat()
        # (R = Rz(yaw) * Ry(pitch) * Rx(roll))
        R = ConversionUtils.quat_to_rotation_matrix(quat)
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0

        return [roll, pitch, yaw]

    @staticmethod
    def euler_to_rpy(euler_angles, sequence='xyz'):
        """Convert Euler angles (degrees) to RPY (radians).

        Args:
            euler_angles: List of [a1, a2, a3] in degrees
            sequence: Euler rotation sequence ('xyz', 'zyx', etc.)

        Returns:
            List of [roll, pitch, yaw] in radians
        """
        # Convert degrees to radians
        angles_rad = [math.radians(a) for a in euler_angles]

        # Rotation matrices for each axis
        def Rx(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        def Ry(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def Rz(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        rot_funcs = {'x': Rx, 'y': Ry, 'z': Rz}

        # Multiply rotation matrices according to sequence
        R = np.eye(3)
        for i, axis in enumerate(sequence):
            R = R @ rot_funcs[axis](angles_rad[i])

        # Extract RPY (XYZ order) from rotation matrix
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0

        return [roll, pitch, yaw]

    @staticmethod
    def xyaxes_to_quat(xyaxes_str):
        """Convert xyaxes attribute to quaternion (w, x, y, z).
        
        xyaxes format: "x1 x2 x3 y1 y2 y3" (two 3D vectors)
        These define the x-axis and y-axis of the coordinate frame.
        The z-axis is computed as the cross product: z = x × y
        
        Args:
            xyaxes_str: String with 6 values "x1 x2 x3 y1 y2 y3"
        
        Returns:
            List of [w, x, y, z] quaternion (MuJoCo convention)
        """
        import numpy as np
        
        values = [float(v) for v in xyaxes_str.split()]
        if len(values) != 6:
            raise ValueError(f"xyaxes must have 6 values, got {len(values)}")
        
        x_axis = np.array(values[0:3])
        y_axis = np.array(values[3:6])
        
        # Normalize axes
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Compute z-axis as cross product
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Re-orthogonalize y-axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Build rotation matrix
        # NOTE: In MuJoCo, xyaxes defines x and y axes in the parent coordinate frame.
        # The rotation matrix R transforms vectors from body local frame to parent frame.
        # Standard rotation matrix: columns are body local axes expressed in parent frame.
        # xyaxes provides x and y axes (first two columns), z is computed via cross product.
        R = np.array([
            [x_axis[0], y_axis[0], z_axis[0]],  # x-axis as column vector (parent frame)
            [x_axis[1], y_axis[1], z_axis[1]],  # y-axis as column vector (parent frame)
            [x_axis[2], y_axis[2], z_axis[2]]   # z-axis as column vector (parent frame)
        ])
        
        # Convert rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return [w, x, y, z]

    @staticmethod
    def quat_to_rotation_matrix(quat):
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
        
        Args:
            quat: List of [w, x, y, z] quaternion (MuJoCo convention)
        
        Returns:
            3x3 numpy array rotation matrix
        """
        import numpy as np
        
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Normalize quaternion
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Build rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R

    @staticmethod
    def quat_multiply(q1, q2):
        """Multiply two quaternions (w, x, y, z format).

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]

        Returns:
            Result quaternion [w, x, y, z] = q1 * q2
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ]

    @staticmethod
    def check_scale_reversal(scale):
        """Check if scale contains negative values indicating mesh reversal.
        
        Args:
            scale: List of [x, y, z] scale values
        
        Returns:
            bool: True if any scale component is negative (indicating reversal)
        """
        if not scale:
            return False
        return any(s < 0 for s in scale)

    @staticmethod
    def zaxis_to_quat(zaxis_str):
        """Convert zaxis attribute to quaternion (w, x, y, z).
        
        zaxis format: "z1 z2 z3" (one 3D vector)
        This defines the z-axis of the coordinate frame.
        x and y axes are computed to form a right-handed coordinate system.
        
        Args:
            zaxis_str: String with 3 values "z1 z2 z3"
        
        Returns:
            List of [w, x, y, z] quaternion (MuJoCo convention)
        """
        import numpy as np
        
        values = [float(v) for v in zaxis_str.split()]
        if len(values) != 3:
            raise ValueError(f"zaxis must have 3 values, got {len(values)}")
        
        z_axis = np.array(values)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Choose a perpendicular vector for x-axis
        if abs(z_axis[0]) < 0.9:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = np.array([0.0, 1.0, 0.0])
        
        # Make x-axis perpendicular to z-axis
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Compute y-axis as cross product
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Build rotation matrix
        R = np.array([
            [x_axis[0], y_axis[0], z_axis[0]],
            [x_axis[1], y_axis[1], z_axis[1]],
            [x_axis[2], y_axis[2], z_axis[2]]
        ])
        
        # Convert rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return [w, x, y, z]

    @staticmethod
    def axisangle_to_quat(axisangle_str):
        """Convert axisangle attribute to quaternion (w, x, y, z).
        
        axisangle format: "ax ay az angle" (axis vector + angle in degrees)
        
        Args:
            axisangle_str: String with 4 values "ax ay az angle"
        
        Returns:
            List of [w, x, y, z] quaternion (MuJoCo convention)
        """
        import numpy as np
        
        values = [float(v) for v in axisangle_str.split()]
        if len(values) != 4:
            raise ValueError(f"axisangle must have 4 values, got {len(values)}")
        
        axis = np.array(values[0:3])
        angle_deg = values[3]
        angle_rad = math.radians(angle_deg)
        
        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        
        axis = axis / axis_norm
        
        # Convert to quaternion
        half_angle = angle_rad * 0.5
        w = math.cos(half_angle)
        sin_half = math.sin(half_angle)
        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half
        
        return [w, x, y, z]

    @staticmethod
    def normalize_pose_attributes(elem, eulerseq='xyz'):
        """Normalize pose attributes (pos/quat/euler/axisangle/xyaxes/zaxis) to position and rotation_quat.
        
        Args:
            elem: XML element with pose attributes
            eulerseq: Euler sequence from compiler settings
        
        Returns:
            dict with 'position' [x,y,z] and 'rotation_quat' [w,x,y,z]
        """
        import numpy as np
        
        # Parse position
        pos_str = elem.get('pos', '0 0 0')
        position = [float(v) for v in pos_str.split()]
        if len(position) < 3:
            position = position + [0.0] * (3 - len(position))
        position = position[:3]
        
        # Parse rotation (try in order: quat, euler, axisangle, xyaxes, zaxis)
        rotation_quat = [1.0, 0.0, 0.0, 0.0]  # Default identity
        
        quat_str = elem.get('quat')
        if quat_str:
            rotation_quat = [float(v) for v in quat_str.split()]
        else:
            euler_str = elem.get('euler')
            if euler_str:
                euler_degrees = [float(v) for v in euler_str.split()]
                euler_rad = [math.radians(e) for e in euler_degrees]
                # Convert euler to quat using eulerseq
                rpy = ConversionUtils.euler_to_rpy(euler_degrees, eulerseq)
                rotation_quat = ConversionUtils.rpy_to_quat(rpy)
            else:
                axisangle_str = elem.get('axisangle')
                if axisangle_str:
                    rotation_quat = ConversionUtils.axisangle_to_quat(axisangle_str)
                else:
                    xyaxes_str = elem.get('xyaxes')
                    if xyaxes_str:
                        rotation_quat = ConversionUtils.xyaxes_to_quat(xyaxes_str)
                    else:
                        zaxis_str = elem.get('zaxis')
                        if zaxis_str:
                            rotation_quat = ConversionUtils.zaxis_to_quat(zaxis_str)
        
        return {
            'position': position,
            'rotation_quat': rotation_quat
        }


# ============================================================================
# FILE PATH RESOLUTION UTILITIES
# ============================================================================

def normalize_name_for_matching(name):
    """Normalize name for matching.
    
    Enables comparison ignoring case, hyphens, underscores, and dots.
    
    Args:
        name: Name to normalize
    
    Returns:
        Normalized name
    """
    return name.lower().replace('_', '').replace('-', '').replace('.', '').strip()


def match_package_name(package_name, dir_name):
    """Check if package name matches directory name.
    
    Tries multiple matching strategies:
    1. Exact match (after normalization)
    2. Partial match (after normalization) - package name contained in dir name
    3. Partial match on original names (case insensitive)
    
    Args:
        package_name: Package name
        dir_name: Directory name
    
    Returns:
        True if match, False otherwise
    """
    if not package_name or not dir_name:
        return False
    
    package_normalized = normalize_name_for_matching(package_name)
    dir_normalized = normalize_name_for_matching(dir_name)

    # Exact match
    if package_normalized == dir_normalized:
        return True

    # Partial match (package in dir or vice versa)
    # e.g. 'premaidai_description' and 'premaidai_description-master'
    if package_normalized in dir_normalized or dir_normalized in package_normalized:
        return True

    # Partial match on original names (case insensitive)
    package_lower = package_name.lower().strip()
    dir_lower = dir_name.lower().strip()

    # Package name contained in dir name (e.g. 'premaidai_description' in 'premaidai_description-master')
    if package_lower in dir_lower:
        return True

    # Dir name contained in package name (e.g. 'premaidai' in 'premaidai_description')
    if dir_lower in package_lower:
        return True

    # Check common prefix (more flexible matching)
    # First part of package name matches (e.g. 'premaidai' in both)
    package_words = package_lower.replace('_', ' ').replace('-', ' ').split()
    dir_words = dir_lower.replace('_', ' ').replace('-', ' ').split()
    
    if package_words and dir_words:
        # First word matches
        if package_words[0] == dir_words[0] and len(package_words[0]) >= 3:
            return True
        # Major part of package name contained in dir name
        for word in package_words:
            if len(word) >= 3 and word in dir_lower:
                return True
    
    return False


def find_package_root(xacro_file_path, package_name, max_search_depth=10, verbose=False):
    """Guess package root directory from package name.
    
    Generic package search algorithm:
    1. Search parent directories from xacro file's directory
    2. Match package name with directory name at each level
    3. Return first match found
    
    Args:
        xacro_file_path: Path to xacro file
        package_name: Package name (e.g. 'premaidai_description')
        max_search_depth: Max search depth (default: 10)
        verbose: Whether to output verbose logs
    
    Returns:
        Package root path, or None if not found
    """
    if not package_name or not package_name.strip():
        if verbose:
            print("  Warning: Empty package name provided")
        return None
    
    package_name = package_name.strip()
    xacro_abs_path = os.path.abspath(xacro_file_path)
    current_path = os.path.dirname(xacro_abs_path)
    root_path = os.path.abspath(os.sep)  # System root path

    if verbose:
        print(f"  Searching for package '{package_name}' starting from: {current_path}")

    # Search parent directories
    search_depth = 0
    while current_path and current_path != root_path and search_depth < max_search_depth:
        if os.path.isdir(current_path):
            dir_name = os.path.basename(current_path)
            
            if verbose:
                print(f"    Checking directory: {dir_name} at {current_path}")

            # Match package name with directory name
            if match_package_name(package_name, dir_name):
                if verbose:
                    print(f"  Found package root: {current_path} (matched '{package_name}' with '{dir_name}')")
                return current_path

        # Move to parent directory
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:  # Reached root
            break
        current_path = parent_path
        search_depth += 1
    
    if verbose:
        print(f"  Could not find package root for: {package_name} (searched {search_depth} levels)")
    return None


def resolve_ros_find_syntax(path_with_find, xacro_file_path, verbose=False):
    """Resolve ROS $(find package_name) syntax.
    
    Args:
        path_with_find: Path in form $(find package_name)/path/to/file
        xacro_file_path: Path to xacro file
        verbose: Whether to output verbose logs
    
    Returns:
        Resolved path, or original path if resolution fails
    """
    import re

    # Search for $(find package_name) pattern
    find_pattern = r'\$\(find\s+([^)]+)\)'

    def replace_find(match):
        package_name = match.group(1).strip()
        package_root = find_package_root(xacro_file_path, package_name, max_search_depth=10, verbose=verbose)

        if package_root:
            # Replace $(find package_name) with package root
            if verbose:
                print(f"  Resolved $(find {package_name}) -> {package_root}")
            return package_root
        else:
            # Keep original if not found
            if verbose:
                print(f"  Warning: Could not resolve $(find {package_name})")
            return match.group(0)

    # Replace all $(find ...)
    resolved_path = re.sub(find_pattern, replace_find, path_with_find)

    # Normalize path
    resolved_path = os.path.normpath(resolved_path)
    
    return resolved_path


def generate_search_candidates(relative_path, base_dir, parent_depth=1, child_depth=1, include_siblings=True):
    """Generate search candidates for file path resolution.
    
    Generic search candidate algorithm:
    1. Relative path from base directory
    2. Relative path from parent directories (up to specified depth)
    3. Relative path from child directories (up to specified depth)
    4. Relative path from sibling directories (optional)
    
    Args:
        relative_path: Relative path to resolve
        base_dir: Base directory
        parent_depth: Parent directory search depth (default: 1)
        child_depth: Child directory search depth (default: 1)
        include_siblings: Whether to include sibling directories (default: True)
    
    Returns:
        List of search candidates (no duplicates, normalized)
    """
    candidates = []
    base_dir = os.path.abspath(base_dir)

    # 1. Relative path from base directory
    candidates.append(os.path.normpath(os.path.join(base_dir, relative_path)))

    # 2. Relative path from parent directories (up to specified depth)
    current_parent = base_dir
    for _ in range(parent_depth):
        parent_dir = os.path.dirname(current_parent)
        if parent_dir == current_parent:  # Reached root
            break
        candidates.append(os.path.normpath(os.path.join(parent_dir, relative_path)))

        # For paths starting with ../, also try path with ../ stripped
        if relative_path.startswith('../'):
            path_without_parent = relative_path
            depth = 0
            while path_without_parent.startswith('../'):
                path_without_parent = path_without_parent[3:]
                depth += 1
            if depth > 0:
                # Resolve relative path from parent directory
                for i in range(min(depth, parent_depth)):
                    parent_candidate = os.path.normpath(os.path.join(parent_dir, path_without_parent))
                    candidates.append(parent_candidate)
        
        current_parent = parent_dir

    # 3. Relative path from child directories (up to specified depth)
    def collect_child_dirs(directory, max_depth, current_depth=0):
        """Recursively collect child directories"""
        child_dirs = []
        if current_depth >= max_depth:
            return child_dirs
        
        if os.path.isdir(directory):
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isdir(item_path):
                        child_dirs.append(item_path)
                        # Recursively collect child directories
                        child_dirs.extend(collect_child_dirs(item_path, max_depth, current_depth + 1))
            except (OSError, PermissionError):
                pass
        return child_dirs
    
    child_dirs = collect_child_dirs(base_dir, child_depth)
    for child_dir in child_dirs:
        candidates.append(os.path.normpath(os.path.join(child_dir, relative_path)))

    # 4. Relative path from sibling directories (optional)
    if include_siblings:
        parent_dir = os.path.dirname(base_dir)
        if os.path.isdir(parent_dir):
            try:
                for item in os.listdir(parent_dir):
                    sibling_path = os.path.join(parent_dir, item)
                    if os.path.isdir(sibling_path) and sibling_path != base_dir:
                        candidates.append(os.path.normpath(os.path.join(sibling_path, relative_path)))
            except (OSError, PermissionError):
                pass

    # 5. If filename only, search in each directory
    if os.path.basename(relative_path) == relative_path:
        # For filename-only, search directly in each directory
        all_dirs = [base_dir]
        all_dirs.extend([os.path.dirname(base_dir)] if parent_depth > 0 else [])
        all_dirs.extend(child_dirs)
        
        for search_dir in all_dirs:
            candidates.append(os.path.normpath(os.path.join(search_dir, relative_path)))

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    
    return unique_candidates


def resolve_file_path_aggressive(relative_path, base_file_path, parent_depth=1, child_depth=1, verbose=False):
    """Aggressively resolve file path.
    
    Generic file path resolution algorithm:
    1. Resolve ROS $(find ...) syntax
    2. Generate search candidates (parent/child/sibling directories)
    3. Check existence of each candidate
    
    Args:
        relative_path: Relative path to resolve (filename or relative path)
        base_file_path: Base file path (e.g. xacro file)
        parent_depth: Parent directory search depth (default: 1)
        child_depth: Child directory search depth (default: 1)
        verbose: Whether to output verbose logs
    
    Returns:
        Absolute path of found file, or None if not found
    """
    if not relative_path:
        return None

    # Resolve $(find package_name) syntax
    if '$(' in relative_path and 'find' in relative_path:
        relative_path = resolve_ros_find_syntax(relative_path, base_file_path, verbose)
        # After resolution, path may be absolute
        if os.path.isabs(relative_path) and os.path.exists(relative_path) and os.path.isfile(relative_path):
            if verbose:
                print(f"  Resolved ROS find syntax: {relative_path}")
            return relative_path

    # Base file directory
    base_dir = os.path.dirname(os.path.abspath(base_file_path))

    # Generate search candidates
    candidates = generate_search_candidates(relative_path, base_dir, parent_depth, child_depth)

    # Check existence of each candidate
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isfile(candidate):
            if verbose:
                print(f"  Resolved path: {relative_path} -> {candidate}")
            return candidate
    
    if verbose:
        print(f"  Could not resolve path: {relative_path} (searched {len(candidates)} candidates)")
    return None


def resolve_path_in_xml_element(elem, attr_names, xacro_file_path, verbose=False):
    """Resolve path in specified attributes of XML element.
    
    Generic XML attribute path resolution:
    1. Get value from specified attribute names
    2. Resolve $(find ...) syntax
    3. Resolve relative path
    4. Set resolved path to attribute
    
    Args:
        elem: XML element
        attr_names: List of attribute names that may contain paths (e.g. ['filename', 'file', 'uri'])
        xacro_file_path: Path to xacro file
        verbose: Whether to output verbose logs
    
    Returns:
        True if path was resolved and modified, False otherwise
    """
    modified = False

    # Get value from specified attribute names
    path_value = None
    attr_name = None
    for name in attr_names:
        if elem.get(name):
            path_value = elem.get(name)
            attr_name = name
            break
    
    if not path_value:
        return False
    
    original_path = path_value

    # Resolve $(find ...) syntax
    if '$(' in path_value and 'find' in path_value:
        resolved_path = resolve_ros_find_syntax(path_value, xacro_file_path, verbose)
        if resolved_path != path_value:
            elem.set(attr_name, resolved_path)
            modified = True
            if verbose:
                print(f"  Resolved ROS find in {attr_name}: {original_path} -> {resolved_path}")
            path_value = resolved_path

    # Resolve relative path (package:// excluded)
    if (not os.path.isabs(path_value) and 
        not path_value.startswith('package://') and
        not path_value.startswith('file://')):
        resolved = resolve_file_path_aggressive(path_value, xacro_file_path, verbose)
        if resolved:
            elem.set(attr_name, resolved)
            modified = True
            if verbose:
                print(f"  Resolved {attr_name} attribute: {original_path} -> {resolved}")
    
    return modified


# ============================================================================
# URDF PARSER
# ============================================================================

