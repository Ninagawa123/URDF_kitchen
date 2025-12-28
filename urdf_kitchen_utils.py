"""
File Name: vtk_utils.py
Description: Shared VTK utilities for URDF Kitchen tools (PartsEditor, Assembler, STLViewerWidget)

Author      : Ninagawa123
Created On  : Dec 28, 2025
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
8. **calculate_inertia_tensor** - Inertia tensor calculation for 3D meshes
9. **calculate_inertia_tetrahedral** - Tetrahedral decomposition method for inertia

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
        if hasattr(mesh, 'visual') and mesh.visual is not None:
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

        # Clean the mesh
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(poly_data)
        clean.SetTolerance(1e-5)
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.PointMergingOn()
        clean.Update()

        poly_data = clean.GetOutput()

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

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Apply color if manually changed
        if color_manually_changed and mesh_color is not None:
            # Convert color from 0-1 range to 0-255 range for trimesh
            color_rgba = [
                int(mesh_color[0] * 255),
                int(mesh_color[1] * 255),
                int(mesh_color[2] * 255),
                int(mesh_color[3] * 255) if len(mesh_color) > 3 else 255
            ]
            mesh.visual = trimesh.visual.ColorVisuals(mesh)
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(
                diffuse=color_rgba,
                ambient=color_rgba,
                specular=[50, 50, 50, 255]
            )

        mesh.export(file_path)

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
