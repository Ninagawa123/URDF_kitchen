"""
File Name: vtk_utils.py
Description: Shared VTK utilities for URDF Kitchen tools (PartsEditor, Assembler, STLViewerWidget)

Author      : Ninagawa123
Created On  : Dec 25, 2025
Version     : 0.1.0
License     : MIT License
URL         : https://github.com/Ninagawa123/URDF_kitchen_beta
Copyright (c) 2025 Ninagawa123

VTK Utilities for URDF Kitchen Tools
====================================

This module provides shared utilities for PartsEditor, Assembler, and STLViewerWidget:

1. **OffscreenRenderer** - Mac M4 compatible offscreen rendering to QLabel
2. **CameraController** - Camera operations (rotate, pan, zoom, reset, fit)
3. **AnimatedCameraRotation** - 90-degree animated rotations for WASD keys
4. **AdaptiveMarkerSize** - Dynamic marker sizing based on zoom level
5. **create_crosshair_marker** - 3D crosshair marker visualization
6. **MouseDragState** - Mouse interaction state management
7. **calculate_arrow_key_step** - Arrow key movement step calculation

Usage Example:
--------------
```python
from vtk_utils import OffscreenRenderer, CameraController

# Setup offscreen rendering
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
render_window.AddRenderer(renderer)

offscreen = OffscreenRenderer(render_window, renderer)
camera = CameraController(renderer, origin=[0, 0, 0])

# Render to QLabel
offscreen.update_display(my_qlabel)

# Control camera
camera.reset_camera()
camera.zoom(delta=120)  # Zoom in
camera.pan(dx=10, dy=5)  # Pan
```

Dependencies:
- vtk
- numpy
- PySide6
"""

import vtk
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


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

            arr = np.frombuffer(vtk_array, dtype=np.uint8)
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
        Pan camera (Shift + drag).

        Args:
            dx: Horizontal mouse movement delta
            dy: Vertical mouse movement delta
            pan_speed_factor: Pan speed multiplier
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

        # Calculate pan offset
        pan_speed = distance * pan_speed_factor
        offset = -right * dx * pan_speed + up * dy * pan_speed

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
