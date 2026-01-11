"""
File Name: urdf_kitchen_utils.py
Description: Shared utilities for URDF Kitchen tools (Assembler, MeshSourcer, PartsEditor).

Author      : Ninagawa123
Created On  : Dec 28, 2025
Update.     : Jan 11, 2026
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
    """カスタムカラーボックスの選択機能を持つカラーダイアログ

    このクラスは、QColorDialogを拡張して、カスタムカラーボックスに
    選択枠を表示する機能を追加します。ユーザーがどのカスタムカラー
    スロットに色を保存するかを視覚的に確認できます。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_custom_color_index = 0  # 選択されたカスタムカラーのインデックス
        self.custom_color_well_array = None  # カスタムカラーのQWellArray
        self._setup_done = False  # セットアップ完了フラグ

    def showEvent(self, event):
        """ダイアログが表示されたときにカスタムカラーボックスをセットアップ"""
        super().showEvent(event)
        if not self._setup_done:
            # ダイアログが完全に表示された後にセットアップ
            QtCore.QTimer.singleShot(300, self._setup_custom_color_boxes)

    def _setup_custom_color_boxes(self):
        """カスタムカラーのQWellArrayを見つけてイベントフィルタを設定"""
        def find_custom_well_array(widget, depth=0):
            """再帰的にカスタムカラーのQWellArrayを探す"""
            class_name = widget.metaObject().className()

            # QtPrivate::QWellArrayで、サイズが224x48のものがカスタムカラー
            if class_name == 'QtPrivate::QWellArray':
                size = widget.size()
                if size.height() == 48:  # カスタムカラーの高さは48
                    self.custom_color_well_array = widget
                    # イベントフィルタをインストール
                    widget.installEventFilter(self)
                    return True

            # 子ウィジェットも探索
            for child in widget.children():
                if isinstance(child, QtWidgets.QWidget):
                    if find_custom_well_array(child, depth + 1):
                        return True
            return False

        if find_custom_well_array(self):
            self._setup_done = True
            # 初期状態で最初のセルに選択枠を表示
            self._draw_selection_border()
            # "Add to Custom Colors"ボタンを見つけてオーバーライド
            self._setup_add_button()
        else:
            # 見つからない場合、もう一度遅延して試す
            if not self._setup_done:
                QtCore.QTimer.singleShot(500, self._setup_custom_color_boxes)

    def _setup_add_button(self):
        """Add to Custom Colorsボタンを見つけてクリックハンドラをオーバーライド"""
        # QPushButtonを全て探して、適切なボタンを見つける
        buttons = self.findChildren(QtWidgets.QPushButton)
        for button in buttons:
            # ボタンのテキストや位置でAdd to Custom Colorsボタンを特定
            # 通常、カスタムカラーのQWellArrayの下にあるボタン
            if button.text() or True:  # テキストの有無に関わらず全ボタンをチェック
                # ボタンの位置がカスタムカラーQWellArrayの近くかチェック
                if self.custom_color_well_array:
                    well_array_geo = self.custom_color_well_array.geometry()
                    button_geo = button.geometry()
                    # カスタムカラーの下にあるボタン（Y座標が近い）
                    if abs(button_geo.y() - (well_array_geo.y() + well_array_geo.height())) < 50:
                        if button_geo.width() > 100:  # 幅が広いボタン
                            # クリック時の処理をオーバーライド
                            button.clicked.disconnect()
                            button.clicked.connect(self._add_custom_color)
                            self._add_button = button
                            break

    def eventFilter(self, obj, event):
        """カスタムカラーQWellArrayのクリックイベントを監視"""
        if obj == self.custom_color_well_array:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                # クリック位置からセルのインデックスを計算
                pos = event.position().toPoint()
                # QWellArrayは通常8列x2行（16色）のグリッド
                # ウィジェットの実際のサイズを取得
                width = self.custom_color_well_array.width()
                height = self.custom_color_well_array.height()

                # セルサイズを計算（境界線を考慮）
                cell_width = width / 8.0
                cell_height = height / 2.0

                col = int(pos.x() / cell_width)
                row = int(pos.y() / cell_height)

                # 範囲チェック
                col = max(0, min(7, col))
                row = max(0, min(1, row))

                # インデックスを計算
                # Qtのカスタムカラーは列優先順序（column-major）で格納されている
                # index = col * rows + row (rows = 2)
                index = col * 2 + row
                self.selected_custom_color_index = index
                # 選択枠を再描画
                self._draw_selection_border()
            elif event.type() == QtCore.QEvent.Paint:
                # ペイントイベントの後に選択枠を描画
                QtCore.QTimer.singleShot(0, self._draw_selection_border)

        return super().eventFilter(obj, event)

    def _add_custom_color(self):
        """現在の色を選択中のカスタムカラースロットに追加"""
        current_color = self.currentColor()
        # 選択中のインデックスにカラーを設定
        QtWidgets.QColorDialog.setCustomColor(self.selected_custom_color_index, current_color)
        # QWellArrayを強制的に更新
        if self.custom_color_well_array:
            self.custom_color_well_array.update()
            self.custom_color_well_array.repaint()
        # 選択インデックスは変更しない

    def _draw_selection_border(self):
        """選択されたカスタムカラーセルに枠を描画"""
        if not self.custom_color_well_array:
            return

        # QWellArrayのサイズを取得
        width = self.custom_color_well_array.width()
        height = self.custom_color_well_array.height()

        # セルサイズを計算（float型で正確に）
        cell_width = width / 8.0
        cell_height = height / 2.0

        # 選択されたセルの位置を計算
        # Qtのカスタムカラーは列優先順序（column-major）で格納
        # index = col * 2 + row なので逆算すると:
        col = self.selected_custom_color_index // 2
        row = self.selected_custom_color_index % 2

        # QWellArrayの各セルには境界線がある
        # 枠をセルの内側に配置
        BORDER_WIDTH = 2  # セルの境界線の幅

        # 位置を計算（境界線の内側に配置）
        x = int(col * cell_width) + BORDER_WIDTH
        y = int(row * cell_height) + BORDER_WIDTH

        # サイズを計算（境界線を除いた領域）
        next_x = int((col + 1) * cell_width)
        next_y = int((row + 1) * cell_height)
        frame_width = next_x - x - BORDER_WIDTH
        frame_height = next_y - y - BORDER_WIDTH

        # QWellArrayに直接描画するためにペイントイベントをオーバーライド
        # 代わりに、子ウィジェットとして枠を表示するQFrameを作成
        if not hasattr(self, '_selection_frame'):
            self._selection_frame = QtWidgets.QFrame(self.custom_color_well_array)
            self._selection_frame.setStyleSheet("border: 3px solid #4080FF; background: transparent;")
            self._selection_frame.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        # 選択枠の位置とサイズを更新
        self._selection_frame.setGeometry(x, y, frame_width, frame_height)
        self._selection_frame.show()
        self._selection_frame.raise_()

    def setCustomColor(self, index, color):
        """カスタムカラーを設定（選択中のインデックスを使用）"""
        # 選択中のインデックスにカラーを設定
        super().setCustomColor(self.selected_custom_color_index, color)
        # 選択インデックスは変更しない（連打しても同じ場所に設定される）


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

            # VTK配列をnumpy配列に変換（推奨方法）
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
    慣性テンソルの物理的妥当性を検証

    Args:
        inertia_tensor: 3x3 numpy配列
        mass: 質量

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

        # 1. 対角成分は正でなければならない
        if ixx <= 0 or iyy <= 0 or izz <= 0:
            messages.append("⚠ Diagonal elements must be positive")
            valid = False

        # 2. 三角不等式の確認 (Ixx + Iyy >= Izz, etc.)
        if not (ixx + iyy >= izz - 1e-6 and
               iyy + izz >= ixx - 1e-6 and
               izz + ixx >= iyy - 1e-6):
            messages.append("⚠ Triangle inequality violated")
            valid = False

        # 3. テンソルの対称性確認
        if not np.allclose(inertia_tensor, inertia_tensor.T, rtol=1e-5):
            messages.append("⚠ Tensor is not symmetric")
            valid = False

        # 4. 固有値が全て正であることを確認
        eigenvalues = np.linalg.eigvals(inertia_tensor)
        if np.any(eigenvalues <= 0):
            messages.append("⚠ Tensor has non-positive eigenvalues")
            valid = False

        # 5. 質量との整合性チェック（大まかな範囲チェック）
        # 慣性の大きさは質量と形状サイズに依存
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
# PARALLEL AXIS THEOREM (平行軸の定理)
# ============================================================================

def parallel_axis_transform(inertia_at_com, mass, displacement):
    """
    平行軸の定理を使用して、重心における慣性テンソルを任意の点における慣性テンソルに変換する。

    Formula:
        I_P = I_COM + m * (||d||² * E - d ⊗ d)

    Args:
        inertia_at_com: numpy.ndarray - 重心における3x3慣性テンソル
        mass: float - 質量 (kg)
        displacement: array-like - 重心から目標点へのベクトル [dx, dy, dz] (m)

    Returns:
        numpy.ndarray: 目標点における3x3慣性テンソル

    Example:
        >>> I_com = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> mass = 1.0
        >>> d = [1.0, 0.0, 0.0]  # X軸方向に1m移動
        >>> I_p = parallel_axis_transform(I_com, mass, d)
    """
    import numpy as np

    d = np.array(displacement, dtype=float)
    d_squared = np.dot(d, d)  # ||d||²

    # 単位行列
    E = np.eye(3)

    # d ⊗ d (外積、dyadic product)
    d_outer_d = np.outer(d, d)

    # Steiner項: m * (||d||² * E - d ⊗ d)
    steiner_term = mass * (d_squared * E - d_outer_d)

    # 変換
    inertia_at_point = inertia_at_com + steiner_term

    return inertia_at_point


def inverse_parallel_axis_transform(inertia_at_point, mass, displacement):
    """
    平行軸の定理の逆変換：任意の点における慣性テンソルを重心における慣性テンソルに変換する。

    Formula:
        I_COM = I_P - m * (||d||² * E - d ⊗ d)

    Args:
        inertia_at_point: numpy.ndarray - 任意の点における3x3慣性テンソル
        mass: float - 質量 (kg)
        displacement: array-like - 重心から目標点へのベクトル [dx, dy, dz] (m)

    Returns:
        numpy.ndarray: 重心における3x3慣性テンソル

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

    # Steiner項を引く
    steiner_term = mass * (d_squared * E - d_outer_d)
    inertia_at_com = inertia_at_point - steiner_term

    return inertia_at_com


# ============================================================================
# MIRRORING UTILITIES (ミラーリング変換)
# ============================================================================

def mirror_inertia_tensor_y_axis(inertia_tensor, center_of_mass):
    """
    Y軸ミラーリングに対応する慣性テンソルとCOMの変換（テンソル変換のみ）。

    Y軸ミラー変換:
        R = diag(1, -1, 1)
        I' = R @ I @ R.T
        COM' = [x, -y, z]

    符号変化:
        - ixy と iyz の符号が反転
        - 対角成分 (ixx, iyy, izz) は不変
        - ixz は不変

    Args:
        inertia_tensor: numpy.ndarray - 3x3慣性テンソル
        center_of_mass: array-like - 重心座標 [x, y, z]

    Returns:
        tuple: (mirrored_inertia_tensor, mirrored_com)
            - mirrored_inertia_tensor: 3x3 numpy.ndarray
            - mirrored_com: list [x, -y, z]

    Example:
        >>> I = np.array([[1, 0.1, 0.2], [0.1, 2, 0.3], [0.2, 0.3, 3]])
        >>> com = [1.0, 2.0, 3.0]
        >>> I_mirror, com_mirror = mirror_inertia_tensor_y_axis(I, com)
        >>> # I_mirror の ixy と iyz が符号反転
        >>> # com_mirror = [1.0, -2.0, 3.0]
    """
    import numpy as np

    # ミラー変換行列 (Y軸反転)
    R = np.diag([1.0, -1.0, 1.0])

    # テンソル変換: I' = R @ I @ R.T
    mirrored_tensor = R @ inertia_tensor @ R.T

    # COM変換: [x, y, z] → [x, -y, z]
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
    # 後方互換性のため（非推奨）
    center_of_mass=None,
    use_trimesh_com=False
):
    """
    trimeshライブラリを使用した統合的な慣性テンソル計算関数（改訂版）。

    物理的に正確な慣性テンソル計算を提供します：
    - 平行軸の定理を使用して任意の原点での慣性テンソルを計算
    - Sceneの場合はunit_scale未指定時に警告（単位系の曖昧さを注意喚起）
    - watertightでないメッシュでは物理量を捏造しない

    処理フロー:
    1. trimeshでメッシュファイルを読み込み（Sceneはunit_scale=1.0をデフォルト）
    2. 必要に応じてメッシュを自動修復
    3. trimeshで体積と重心を計算（物理量の基準）
    4. VTKで慣性テンソルを計算（四面体分解法）
    5. 平行軸の定理で任意の原点に変換

    Args:
        mesh_file_path: str - メッシュファイルのパス (.stl, .obj, .dae)
        mass: float or None - 質量 (kg)。Noneの場合はdensityから計算
        density: float or None - 密度 (kg/m³)。massとdensityの両方がNoneの場合はエラー
        reference_point: array-like or None - 慣性テンソルの原点 [x, y, z] (m)
            Noneの場合は重心が原点
        auto_repair: bool - メッシュの自動修復を行うかどうか (default: True)
        unit_scale: float - Sceneの場合の単位スケール (例: 0.001でmmをmに変換)
            デフォルト: 1.0 (スケーリングなし)。Sceneファイルで未指定の場合は警告を表示。
        center_of_mass: array-like or None - **非推奨** reference_pointを使用してください
        use_trimesh_com: bool - **非推奨** reference_point=Noneで同じ動作

    Returns:
        dict: {
            'success': bool - 計算成功フラグ
            'inertia_tensor': numpy.ndarray - 3x3慣性テンソル行列（reference_point周り）
            'inertia_at_com': numpy.ndarray - 3x3慣性テンソル行列（重心周り）
            'volume': float - 体積 (m³)
            'mass': float - 質量 (kg)
            'density': float - 密度 (kg/m³)
            'center_of_mass': list - 重心座標 [x, y, z] (m)
            'reference_point': list - 慣性テンソルの原点 [x, y, z] (m)
            'is_watertight': bool - メッシュが閉じているか
            'repair_performed': bool - 修復が実行されたか
            'is_scene': bool - Sceneとして読み込まれたか
            'unit_scale_used': float or None - 使用された単位スケール
            'error_message': str or None - エラーメッセージ
        }

    Raises:
        ValueError: Sceneの場合にunit_scaleが指定されていない
        ValueError: massとdensityの両方がNone
        ValueError: watertightでないメッシュ（auto_repair失敗後）

    Example:
        >>> # 基本的な使用例（重心周りの慣性テンソル）
        >>> result = calculate_inertia_with_trimesh('part.stl', mass=1.0)

        >>> # 任意の原点周りの慣性テンソル
        >>> result = calculate_inertia_with_trimesh(
        ...     'part.stl', mass=1.0, reference_point=[0, 0, 0])

        >>> # Sceneの場合（unit_scale必須）
        >>> result = calculate_inertia_with_trimesh(
        ...     'assembly.dae', density=1000.0, unit_scale=0.001)  # mm→m

        >>> # 後方互換性（非推奨）
        >>> result = calculate_inertia_with_trimesh(
        ...     'part.stl', mass=1.0, center_of_mass=[0, 0, 0])  # reference_pointを推奨
    """
    import os
    import numpy as np

    # 後方互換性の処理
    if center_of_mass is not None:
        if reference_point is None:
            reference_point = center_of_mass
            print("⚠ Warning: 'center_of_mass' parameter is deprecated. Use 'reference_point' instead.")
        else:
            print("⚠ Warning: Both 'center_of_mass' and 'reference_point' specified. Using 'reference_point'.")

    if use_trimesh_com and reference_point is not None:
        print("⚠ Warning: 'use_trimesh_com=True' is ignored when 'reference_point' is specified.")

    # 結果を格納する辞書
    result = {
        'success': False,
        'inertia_tensor': None,
        'inertia_at_com': None,
        'volume': 0.0,
        'mass': 0.0,
        'density': 0.0,
        'center_of_mass': [0.0, 0.0, 0.0],
        'trimesh_com': [0.0, 0.0, 0.0],  # 後方互換性
        'reference_point': [0.0, 0.0, 0.0],
        'is_watertight': False,
        'repair_performed': False,
        'is_scene': False,
        'unit_scale_used': None,
        'error_message': None
    }

    try:
        # trimeshのインポート確認
        try:
            import trimesh
        except ImportError:
            result['error_message'] = "trimesh library not available. Install with: pip install trimesh"
            return result

        # 入力検証
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

        # Step 1: メッシュ読み込み（Scene処理を厳格化）
        loaded_mesh = trimesh.load(mesh_file_path)

        # Scene処理（unit_scale確認）
        if isinstance(loaded_mesh, trimesh.Scene):
            result['is_scene'] = True
            print(f"\n⚠ Loaded as Scene (contains multiple meshes or materials)")

            # unit_scaleがデフォルト値の場合は警告
            if unit_scale == 1.0:
                print(f"  ⚠ WARNING: unit_scale not specified (using default 1.0)")
                print(f"     Scene files may have ambiguous units.")
                print(f"     Please verify units are correct, or specify unit_scale explicitly.")
                print(f"     Example: unit_scale=0.001 for mm→m conversion")

            print(f"  Unit scale: {unit_scale}")
            result['unit_scale_used'] = unit_scale

            # Sceneをメッシュに結合
            mesh = loaded_mesh.dump(concatenate=True)

            # スケール適用（1.0でも明示的に適用）
            mesh.apply_scale(unit_scale)
            if unit_scale != 1.0:
                print(f"  ✓ Applied unit scale: {unit_scale}")
            else:
                print(f"  No scaling applied (unit_scale=1.0)")
        else:
            # 単一メッシュ
            mesh = loaded_mesh
            print(f"\n✓ Loaded as single Mesh")

        print(f"\nMesh properties:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Volume (trimesh): {mesh.volume:.9f} m³")
        print(f"  Is watertight: {mesh.is_watertight}")

        result['is_watertight'] = mesh.is_watertight

        # Step 2: 自動修復
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

        # watertightでない場合は警告（計算は続行）
        if not mesh.is_watertight:
            print(f"\n⚠ WARNING: Mesh is not watertight!")
            print(f"  Physical properties may not be accurate.")

        # Step 3: VTKポリデータに変換して体積計算
        # trimeshメッシュオブジェクトを直接VTKに変換（unit_scaleが反映される）
        poly_data, vtk_volume = trimesh_to_vtk_polydata(mesh)

        print(f"\n{'='*60}")
        print(f"VOLUME CALCULATION")
        print(f"{'='*60}")
        print(f"  Trimesh volume: {mesh.volume:.9f} m³")
        print(f"  VTK volume: {vtk_volume:.9f} m³")

        # VTK体積を使用（より信頼性が高い）
        result['volume'] = vtk_volume

        # Step 4: trimeshで重心を計算
        trimesh_com = mesh.center_mass
        print(f"\nCenter of mass (trimesh): [{trimesh_com[0]:.6f}, {trimesh_com[1]:.6f}, {trimesh_com[2]:.6f}]")

        result['center_of_mass'] = list(trimesh_com)
        result['trimesh_com'] = list(trimesh_com)  # 後方互換性

        # Step 5: 質量と密度の計算
        if mass is None:
            mass = vtk_volume * density
            print(f"\nMass calculated from density: {mass:.6f} kg")
        elif density is None:
            density = mass / vtk_volume if vtk_volume > 0 else 0
            print(f"\nDensity calculated from mass: {density:.6f} kg/m³")

        result['mass'] = mass
        result['density'] = density

        # Step 6: 重心周りの慣性テンソルを計算（VTK四面体分解法）
        print(f"\n{'='*60}")
        print(f"INERTIA TENSOR CALCULATION (at COM)")
        print(f"{'='*60}")
        print(f"  Method: Tetrahedral decomposition (VTK)")
        print(f"  Mass: {mass:.6f} kg")
        print(f"  Density: {density:.6f} kg/m³")

        inertia_at_com, volume_integral = calculate_inertia_tetrahedral(
            poly_data, density, list(trimesh_com)
        )

        # 体積の検証
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

        # Step 7: reference_pointが指定されている場合は平行軸の定理で変換
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
            # reference_pointが指定されていない場合は重心周りのテンソルを使用
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
        # まず頂点カラーや面カラーを確認（カラフルな表示のため）
        vertex_colors = None
        face_colors = None
        extracted_color = None  # 初期化
        
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            # 頂点カラーを取得
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vertex_colors = mesh.visual.vertex_colors
                print(f"Found vertex colors in .dae file: {len(vertex_colors)} vertices")
                
                # 頂点カラーがある場合、代表色（最初の頂点の色または平均色）をextracted_colorとして設定
                import numpy as np
                if not isinstance(vertex_colors, np.ndarray):
                    vertex_colors = np.array(vertex_colors)
                
                if len(vertex_colors) > 0:
                    # 最初の頂点の色を使用（または平均色を計算）
                    first_color = vertex_colors[0]
                    # 0-255範囲の場合は0-1範囲に変換
                    if isinstance(first_color, np.ndarray):
                        if first_color.max() > 1.0:
                            # 0-255範囲の場合
                            extracted_color = [
                                float(first_color[0]) / 255.0,
                                float(first_color[1]) / 255.0,
                                float(first_color[2]) / 255.0,
                                float(first_color[3]) / 255.0 if len(first_color) > 3 else 1.0
                            ]
                        else:
                            # 0-1範囲の場合
                            extracted_color = [
                                float(first_color[0]),
                                float(first_color[1]),
                                float(first_color[2]),
                                float(first_color[3]) if len(first_color) > 3 else 1.0
                            ]
                    else:
                        # リストやタプルの場合
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
            
            # 面カラーを取得（頂点カラーがない場合）
            if vertex_colors is None:
                if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                    face_colors = mesh.visual.face_colors
                    print(f"Found face colors in .dae file: {len(face_colors)} faces")
                    
                    # 面カラーがある場合、代表色（最初の面の色）をextracted_colorとして設定
                    import numpy as np
                    if not isinstance(face_colors, np.ndarray):
                        face_colors = np.array(face_colors)
                    
                    if len(face_colors) > 0:
                        # 最初の面の色を使用
                        first_color = face_colors[0]
                        # 0-255範囲の場合は0-1範囲に変換
                        if isinstance(first_color, np.ndarray):
                            if first_color.max() > 1.0:
                                # 0-255範囲の場合
                                extracted_color = [
                                    float(first_color[0]) / 255.0,
                                    float(first_color[1]) / 255.0,
                                    float(first_color[2]) / 255.0,
                                    float(first_color[3]) / 255.0 if len(first_color) > 3 else 1.0
                                ]
                            else:
                                # 0-1範囲の場合
                                extracted_color = [
                                    float(first_color[0]),
                                    float(first_color[1]),
                                    float(first_color[2]),
                                    float(first_color[3]) if len(first_color) > 3 else 1.0
                                ]
                        else:
                            # リストやタプルの場合
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
            
            # 頂点カラーや面カラーがない場合、materialから単一色を取得
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

        # 頂点カラーをVTK PolyDataに設定
        if vertex_colors is not None and len(vertex_colors) > 0:
            import numpy as np
            # 頂点カラーをVTK形式に変換
            vtk_colors = vtk.vtkUnsignedCharArray()
            vtk_colors.SetNumberOfComponents(3)  # RGB
            vtk_colors.SetName("Colors")
            
            # numpy配列に変換
            if not isinstance(vertex_colors, np.ndarray):
                vertex_colors = np.array(vertex_colors)
            
            # 頂点数とカラー数が一致するか確認
            num_vertices = len(vertices)
            num_colors = len(vertex_colors)
            
            if num_colors == num_vertices:
                # 各頂点の色を設定
                for i in range(num_vertices):
                    color = vertex_colors[i]
                    # numpy配列の場合、要素に直接アクセス
                    if isinstance(color, np.ndarray):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    elif isinstance(color, (list, tuple)):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    else:
                        r = g = b = 255  # デフォルト白色
                    
                    # 0-1範囲の場合は255倍する
                    if r <= 1.0 and g <= 1.0 and b <= 1.0:
                        r = int(r * 255)
                        g = int(g * 255)
                        b = int(b * 255)
                    
                    vtk_colors.InsertNextTuple3(r, g, b)
                
                poly_data.GetPointData().SetScalars(vtk_colors)
                print(f"Applied vertex colors to VTK PolyData: {num_colors} vertices")
            else:
                print(f"Warning: Vertex count ({num_vertices}) doesn't match color count ({num_colors})")
        
        # 面カラーをVTK PolyDataに設定（頂点カラーがない場合）
        elif face_colors is not None and len(face_colors) > 0:
            import numpy as np
            # 面カラーをVTK形式に変換
            vtk_colors = vtk.vtkUnsignedCharArray()
            vtk_colors.SetNumberOfComponents(3)  # RGB
            vtk_colors.SetName("Colors")
            
            # numpy配列に変換
            if not isinstance(face_colors, np.ndarray):
                face_colors = np.array(face_colors)
            
            # 面数とカラー数が一致するか確認
            num_faces = len(faces)
            num_colors = len(face_colors)
            
            if num_colors == num_faces:
                # 各面の色を設定
                for i in range(num_faces):
                    color = face_colors[i]
                    # numpy配列の場合、要素に直接アクセス
                    if isinstance(color, np.ndarray):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    elif isinstance(color, (list, tuple)):
                        r = int(color[0]) if color[0] <= 255 else 255
                        g = int(color[1]) if color[1] <= 255 else 255
                        b = int(color[2]) if color[2] <= 255 else 255
                    else:
                        r = g = b = 255  # デフォルト白色
                    
                    # 0-1範囲の場合は255倍する
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
        # スカラー値（頂点カラーや面カラー）を保持するために、Clean前に保存
        saved_point_scalars = poly_data.GetPointData().GetScalars()
        saved_cell_scalars = poly_data.GetCellData().GetScalars()
        
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(poly_data)
        clean.SetTolerance(1e-5)
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.PointMergingOn()
        # スカラー値を保持する設定（デフォルトで保持されるが、明示的に設定）
        clean.Update()

        poly_data = clean.GetOutput()
        
        # Clean後にスカラー値を再設定（必要に応じて）
        # 注意: vtkCleanPolyDataは頂点をマージする可能性があるため、
        # スカラー値の再マッピングが必要な場合がある
        if saved_point_scalars is not None:
            # 頂点カラーがある場合、Clean後の頂点数に合わせて再設定
            num_points_after = poly_data.GetNumberOfPoints()
            num_points_before = saved_point_scalars.GetNumberOfTuples()
            
            if num_points_after == num_points_before:
                # 頂点数が同じ場合はそのまま再設定
                poly_data.GetPointData().SetScalars(saved_point_scalars)
                print(f"Restored vertex colors after cleaning: {num_points_after} vertices")
            else:
                # 頂点数が変わった場合は、最初のN個の色を使用
                vtk_colors = vtk.vtkUnsignedCharArray()
                vtk_colors.SetNumberOfComponents(3)
                vtk_colors.SetName("Colors")
                for i in range(min(num_points_after, num_points_before)):
                    color = saved_point_scalars.GetTuple3(i)
                    vtk_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))
                # 残りの頂点には最後の色を使用
                if num_points_after > num_points_before and num_points_before > 0:
                    last_color = saved_point_scalars.GetTuple3(num_points_before - 1)
                    for i in range(num_points_before, num_points_after):
                        vtk_colors.InsertNextTuple3(int(last_color[0]), int(last_color[1]), int(last_color[2]))
                poly_data.GetPointData().SetScalars(vtk_colors)
                print(f"Adjusted vertex colors after cleaning: {num_points_before} -> {num_points_after} vertices")
        
        if saved_cell_scalars is not None:
            # 面カラーがある場合、Clean後の面数に合わせて再設定
            num_cells_after = poly_data.GetNumberOfCells()
            num_cells_before = saved_cell_scalars.GetNumberOfTuples()
            
            if num_cells_after == num_cells_before:
                # 面数が同じ場合はそのまま再設定
                poly_data.GetCellData().SetScalars(saved_cell_scalars)
                print(f"Restored face colors after cleaning: {num_cells_after} cells")
            else:
                # 面数が変わった場合は、最初のN個の色を使用
                vtk_colors = vtk.vtkUnsignedCharArray()
                vtk_colors.SetNumberOfComponents(3)
                vtk_colors.SetName("Colors")
                for i in range(min(num_cells_after, num_cells_before)):
                    color = saved_cell_scalars.GetTuple3(i)
                    vtk_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))
                # 残りの面には最後の色を使用
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
        palette.setColor(QPalette.PlaceholderText, QColor(180, 180, 180))  # プレースホルダーテキスト色
        app.setPalette(palette)

        # ボタンの押下時の色をスタイルシートで設定（フォーカスの有無に関わらず深い青）
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

        # ボタンの押下時の色をスタイルシートで設定（フォーカスの有無に関わらず深い青）
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

        # ボタンの押下時の色をスタイルシートで設定（フォーカスの有無に関わらず深い青）
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
    Y軸でミラーリングする際の物理プロパティ変換（慣性テンソルと慣性中心）

    この関数は、左右対称なパーツの物理特性を正しくミラーリングします。
    PartsEditorで使用されている方法を標準化したものです。

    Y軸ミラーリングの物理的変換:
    - 慣性中心（COM）: Y座標を反転 [x, y, z] → [x, -y, z]
    - 慣性テンソル:
      * 対角成分（ixx, iyy, izz）: 変化なし
      * 非対角成分: ixy, iyz を符号反転、ixz は変化なし

    Args:
        inertia_dict: dict - 慣性テンソルの辞書
            {'ixx': float, 'iyy': float, 'izz': float,
             'ixy': float, 'ixz': float, 'iyz': float}
        inertial_origin_dict: dict - 慣性中心の辞書
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
    # 慣性テンソルのミラーリング
    mirrored_inertia = {}
    if inertia_dict is not None and isinstance(inertia_dict, dict):
        mirrored_inertia = inertia_dict.copy()
        # Y軸ミラーリング: ixyとiyzの符号を反転
        if 'ixy' in mirrored_inertia:
            mirrored_inertia['ixy'] = -inertia_dict['ixy']
        if 'iyz' in mirrored_inertia:
            mirrored_inertia['iyz'] = -inertia_dict['iyz']
        # ixx, iyy, izz, ixzはそのまま

    # 慣性中心（COM）のミラーリング
    mirrored_origin = {}
    if inertial_origin_dict is not None and isinstance(inertial_origin_dict, dict):
        mirrored_origin = inertial_origin_dict.copy()
        if 'xyz' in mirrored_origin and len(mirrored_origin['xyz']) >= 3:
            xyz = mirrored_origin['xyz']
            mirrored_origin['xyz'] = [xyz[0], -xyz[1], xyz[2]]
        # RPYはそのまま（回転の扱いは複雑なため、必要に応じて調整）

    return mirrored_inertia, mirrored_origin


def mirror_inertia_tensor_left_right(inertia_dict):
    """
    左右反転（Y軸ミラーリング）のための慣性テンソルの符号反転
    
    左右対称なパーツ（l_* と r_*）を作成する際に使用します。
    Y軸でミラーリングする場合、以下の符号反転を行います：
    - ixy: 符号反転（-1倍）
    - iyz: 符号反転（-1倍）
    - ixx, iyy, izz, ixz: 変化なし
    
    Args:
        inertia_dict: dict - 慣性テンソルの辞書
            {'ixx': float, 'iyy': float, 'izz': float,
             'ixy': float, 'ixz': float, 'iyz': float}
    
    Returns:
        dict: 左右反転後の慣性テンソル辞書（新しい辞書を返す）
    
    Example:
        >>> inertia = {'ixx': 0.001, 'iyy': 0.002, 'izz': 0.003,
        ...            'ixy': 0.0001, 'ixz': 0.0002, 'iyz': 0.0003}
        >>> mirrored = mirror_inertia_tensor_left_right(inertia)
        >>> print(mirrored['ixy'])  # -0.0001
        >>> print(mirrored['iyz'])  # -0.0003
        >>> print(mirrored['ixx'])  # 0.001 (変化なし)
    """
    if inertia_dict is None or not isinstance(inertia_dict, dict):
        return None
    
    # 新しい辞書を作成（元の辞書を変更しない）
    mirrored_inertia = inertia_dict.copy()
    
    # Y軸ミラーリング: ixyとiyzの符号を反転
    if 'ixy' in mirrored_inertia:
        mirrored_inertia['ixy'] = -inertia_dict['ixy']
    if 'iyz' in mirrored_inertia:
        mirrored_inertia['iyz'] = -inertia_dict['iyz']
    # ixx, iyy, izz, ixzはそのまま
    
    return mirrored_inertia


def mirror_center_of_mass_left_right(center_of_mass):
    """
    左右反転（Y軸ミラーリング）のためのCenter of MassのY座標符号反転
    
    左右対称なパーツ（l_* と r_*）を作成する際に使用します。
    Y軸でミラーリングする場合、Y座標のみを符号反転します：
    [x, y, z] → [x, -y, z]
    
    Args:
        center_of_mass: list or array-like - Center of Mass座標 [x, y, z]
    
    Returns:
        list: 左右反転後のCenter of Mass座標 [x, -y, z]（新しいリストを返す）
    
    Example:
        >>> com = [0.1, 0.2, 0.3]
        >>> mirrored_com = mirror_center_of_mass_left_right(com)
        >>> print(mirrored_com)  # [0.1, -0.2, 0.3]
    """
    if center_of_mass is None:
        return None
    
    # リストまたは配列をリストに変換
    if hasattr(center_of_mass, '__iter__') and not isinstance(center_of_mass, str):
        com_list = list(center_of_mass)
        if len(com_list) >= 3:
            # 新しいリストを作成（元のリストを変更しない）
            return [com_list[0], -com_list[1], com_list[2]]
        else:
            return com_list.copy()
    else:
        return center_of_mass


def calculate_mirrored_physical_properties_from_mesh(mesh_file_path, mass, density=1.0):
    """
    メッシュファイルからY軸ミラーリングされた物理プロパティを計算

    この関数は、STL/OBJ/DAEメッシュファイルをY軸でミラーリングし、
    ミラーリングされたジオメトリから物理プロパティ（体積、重心、慣性テンソル）を
    再計算します。PartsEditorで使用されている標準的な方法です。

    Args:
        mesh_file_path: str - メッシュファイルのパス（.stl, .obj, .dae）
        mass: float - 質量（kg）
        density: float - 密度（kg/m³）、massが指定されていない場合に使用

    Returns:
        dict: ミラーリングされた物理プロパティ
            {
                'volume': float,
                'mass': float,
                'center_of_mass': [x, y, z],
                'inertia': {
                    'ixx': float, 'iyy': float, 'izz': float,
                    'ixy': float, 'ixz': float, 'iyz': float
                }
            }
        エラー時はNoneを返す

    Example:
        >>> props = calculate_mirrored_physical_properties_from_mesh(
        ...     'l_arm.stl', mass=0.5)
        >>> print(props['center_of_mass'])  # Y座標が反転された重心
        >>> print(props['inertia']['ixy'])  # 符号反転された非対角成分
    """
    import vtk
    import numpy as np

    try:
        # メッシュファイルを読み込む
        poly_data, volume_unused, extracted_color = load_mesh_to_polydata(mesh_file_path)

        # Y軸反転の変換を設定
        transform = vtk.vtkTransform()
        transform.Scale(1, -1, 1)

        # 変換を適用
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetInputData(poly_data)
        transformer.SetTransform(transform)
        transformer.Update()

        # 法線の修正
        normal_generator = vtk.vtkPolyDataNormals()
        normal_generator.SetInputData(transformer.GetOutput())
        normal_generator.ConsistencyOn()
        normal_generator.AutoOrientNormalsOn()
        normal_generator.ComputeCellNormalsOn()
        normal_generator.ComputePointNormalsOn()
        normal_generator.Update()

        mirrored_poly_data = normal_generator.GetOutput()

        # 体積を計算
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(mirrored_poly_data)
        mass_properties.Update()
        volume = mass_properties.GetVolume()

        # 質量が指定されていない場合は体積×密度で計算
        if mass is None or mass <= 0:
            mass = volume * density

        # 重心を計算
        com_filter = vtk.vtkCenterOfMass()
        com_filter.SetInputData(mirrored_poly_data)
        com_filter.SetUseScalarsAsWeights(False)
        com_filter.Update()
        center_of_mass = list(com_filter.GetCenter())

        # 慣性テンソルを計算（is_mirrored=Trueでミラーリングを考慮）
        inertia_tensor = calculate_inertia_tensor(
            mirrored_poly_data, mass, center_of_mass, is_mirrored=True
        )

        # 辞書形式で返す
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
        # カメラの現在の位置と焦点を取得
        position = list(camera.GetPosition())
        focal_point = self.absolute_origin

        # 回転行列を作成
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Translate(*focal_point)
        transform.RotateZ(self.camera_rotation[2])  # Roll
        transform.RotateX(self.camera_rotation[0])  # Pitch
        transform.RotateY(self.camera_rotation[1])  # Yaw
        transform.Translate(*[-x for x in focal_point])

        # カメラの位置を回転
        new_position = transform.TransformPoint(position)
        camera.SetPosition(new_position)

        # カメラの上方向を更新
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
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

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


# ============================================================================
# FILE PATH RESOLUTION UTILITIES
# ============================================================================

def normalize_name_for_matching(name):
    """名前をマッチング用に正規化する。
    
    大文字小文字、ハイフン、アンダースコア、ドットを無視して比較できるようにする。
    
    Args:
        name: 正規化する名前
    
    Returns:
        正規化された名前
    """
    return name.lower().replace('_', '').replace('-', '').replace('.', '').strip()


def match_package_name(package_name, dir_name):
    """パッケージ名とディレクトリ名が一致するかチェックする。
    
    複数のマッチング方法を試行：
    1. 完全一致（正規化後）
    2. 部分一致（正規化後）- パッケージ名がディレクトリ名の先頭に含まれる
    3. 元の名前での部分一致（大文字小文字を無視）
    
    Args:
        package_name: パッケージ名
        dir_name: ディレクトリ名
    
    Returns:
        一致する場合はTrue、そうでない場合はFalse
    """
    if not package_name or not dir_name:
        return False
    
    package_normalized = normalize_name_for_matching(package_name)
    dir_normalized = normalize_name_for_matching(dir_name)
    
    # 完全一致
    if package_normalized == dir_normalized:
        return True
    
    # 部分一致（パッケージ名がディレクトリ名の先頭に含まれる、またはその逆）
    # 例: 'premaidai_description' と 'premaidai_description-master'
    if package_normalized in dir_normalized or dir_normalized in package_normalized:
        return True
    
    # 元の名前での部分一致（大文字小文字を無視）
    package_lower = package_name.lower().strip()
    dir_lower = dir_name.lower().strip()
    
    # パッケージ名がディレクトリ名の先頭に含まれる（例: 'premaidai_description' in 'premaidai_description-master'）
    if package_lower in dir_lower:
        return True
    
    # ディレクトリ名がパッケージ名の先頭に含まれる（例: 'premaidai' in 'premaidai_description'）
    if dir_lower in package_lower:
        return True
    
    # 共通のプレフィックスをチェック（より柔軟なマッチング）
    # パッケージ名の最初の部分が一致する場合（例: 'premaidai' が両方に含まれる）
    package_words = package_lower.replace('_', ' ').replace('-', ' ').split()
    dir_words = dir_lower.replace('_', ' ').replace('-', ' ').split()
    
    if package_words and dir_words:
        # 最初の単語が一致する場合
        if package_words[0] == dir_words[0] and len(package_words[0]) >= 3:
            return True
        # パッケージ名の主要部分がディレクトリ名に含まれる
        for word in package_words:
            if len(word) >= 3 and word in dir_lower:
                return True
    
    return False


def find_package_root(xacro_file_path, package_name, max_search_depth=10, verbose=False):
    """パッケージ名からパッケージのルートディレクトリを推測する。
    
    汎用的なパッケージ検索アルゴリズム：
    1. xacroファイルのディレクトリから親ディレクトリを探索
    2. 各階層でパッケージ名とディレクトリ名をマッチング
    3. 見つかった最初のマッチを返す
    
    Args:
        xacro_file_path: xacroファイルのパス
        package_name: パッケージ名（例: 'premaidai_description'）
        max_search_depth: 最大探索階層数（デフォルト: 10）
        verbose: 詳細なログを出力するかどうか
    
    Returns:
        パッケージのルートディレクトリのパス、見つからない場合はNone
    """
    if not package_name or not package_name.strip():
        if verbose:
            print("  Warning: Empty package name provided")
        return None
    
    package_name = package_name.strip()
    xacro_abs_path = os.path.abspath(xacro_file_path)
    current_path = os.path.dirname(xacro_abs_path)
    root_path = os.path.abspath(os.sep)  # システムのルートパス
    
    if verbose:
        print(f"  Searching for package '{package_name}' starting from: {current_path}")
    
    # 親ディレクトリを探索
    search_depth = 0
    while current_path and current_path != root_path and search_depth < max_search_depth:
        if os.path.isdir(current_path):
            dir_name = os.path.basename(current_path)
            
            if verbose:
                print(f"    Checking directory: {dir_name} at {current_path}")
            
            # パッケージ名とディレクトリ名をマッチング
            if match_package_name(package_name, dir_name):
                if verbose:
                    print(f"  Found package root: {current_path} (matched '{package_name}' with '{dir_name}')")
                return current_path
        
        # 親ディレクトリに移動
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:  # ルートに到達
            break
        current_path = parent_path
        search_depth += 1
    
    if verbose:
        print(f"  Could not find package root for: {package_name} (searched {search_depth} levels)")
    return None


def resolve_ros_find_syntax(path_with_find, xacro_file_path, verbose=False):
    """ROSの$(find package_name)構文を解決する。
    
    Args:
        path_with_find: $(find package_name)/path/to/file 形式のパス
        xacro_file_path: xacroファイルのパス
        verbose: 詳細なログを出力するかどうか
    
    Returns:
        解決されたパス、解決できない場合は元のパス
    """
    import re
    
    # $(find package_name) パターンを検索
    find_pattern = r'\$\(find\s+([^)]+)\)'
    
    def replace_find(match):
        package_name = match.group(1).strip()
        package_root = find_package_root(xacro_file_path, package_name, max_search_depth=10, verbose=verbose)
        
        if package_root:
            # $(find package_name) をパッケージルートに置き換え
            if verbose:
                print(f"  Resolved $(find {package_name}) -> {package_root}")
            return package_root
        else:
            # 見つからない場合は元のまま
            if verbose:
                print(f"  Warning: Could not resolve $(find {package_name})")
            return match.group(0)
    
    # すべての$(find ...)を置き換え
    resolved_path = re.sub(find_pattern, replace_find, path_with_find)
    
    # パスを正規化
    resolved_path = os.path.normpath(resolved_path)
    
    return resolved_path


def generate_search_candidates(relative_path, base_dir, parent_depth=1, child_depth=1, include_siblings=True):
    """ファイルパス解決のための探索候補を生成する。
    
    汎用的な探索候補生成アルゴリズム：
    1. 基準ディレクトリからの相対パス
    2. 親ディレクトリ（指定階層まで）からの相対パス
    3. 子ディレクトリ（指定階層まで）からの相対パス
    4. 兄弟ディレクトリからの相対パス（オプション）
    
    Args:
        relative_path: 解決したい相対パス
        base_dir: 基準ディレクトリ
        parent_depth: 親ディレクトリの探索階層数（デフォルト: 1）
        child_depth: 子ディレクトリの探索階層数（デフォルト: 1）
        include_siblings: 兄弟ディレクトリを含めるかどうか（デフォルト: True）
    
    Returns:
        探索候補のリスト（重複なし、正規化済み）
    """
    candidates = []
    base_dir = os.path.abspath(base_dir)
    
    # 1. 基準ディレクトリからの相対パス
    candidates.append(os.path.normpath(os.path.join(base_dir, relative_path)))
    
    # 2. 親ディレクトリからの相対パス（指定階層まで）
    current_parent = base_dir
    for _ in range(parent_depth):
        parent_dir = os.path.dirname(current_parent)
        if parent_dir == current_parent:  # ルートに到達
            break
        candidates.append(os.path.normpath(os.path.join(parent_dir, relative_path)))
        
        # ../で始まるパスの場合、../を除去したパスも試す
        if relative_path.startswith('../'):
            path_without_parent = relative_path
            depth = 0
            while path_without_parent.startswith('../'):
                path_without_parent = path_without_parent[3:]
                depth += 1
            if depth > 0:
                # 親ディレクトリから相対パスを解決
                for i in range(min(depth, parent_depth)):
                    parent_candidate = os.path.normpath(os.path.join(parent_dir, path_without_parent))
                    candidates.append(parent_candidate)
        
        current_parent = parent_dir
    
    # 3. 子ディレクトリからの相対パス（指定階層まで）
    def collect_child_dirs(directory, max_depth, current_depth=0):
        """子ディレクトリを再帰的に収集する"""
        child_dirs = []
        if current_depth >= max_depth:
            return child_dirs
        
        if os.path.isdir(directory):
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isdir(item_path):
                        child_dirs.append(item_path)
                        # 再帰的に子ディレクトリを収集
                        child_dirs.extend(collect_child_dirs(item_path, max_depth, current_depth + 1))
            except (OSError, PermissionError):
                pass
        return child_dirs
    
    child_dirs = collect_child_dirs(base_dir, child_depth)
    for child_dir in child_dirs:
        candidates.append(os.path.normpath(os.path.join(child_dir, relative_path)))
    
    # 4. 兄弟ディレクトリからの相対パス（オプション）
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
    
    # 5. ファイル名のみの場合、各ディレクトリで検索
    if os.path.basename(relative_path) == relative_path:
        # ファイル名のみの場合は、各ディレクトリで直接検索
        all_dirs = [base_dir]
        all_dirs.extend([os.path.dirname(base_dir)] if parent_depth > 0 else [])
        all_dirs.extend(child_dirs)
        
        for search_dir in all_dirs:
            candidates.append(os.path.normpath(os.path.join(search_dir, relative_path)))
    
    # 重複を除去
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    
    return unique_candidates


def resolve_file_path_aggressive(relative_path, base_file_path, parent_depth=1, child_depth=1, verbose=False):
    """強力にファイルパスを解決する関数。
    
    汎用的なファイルパス解決アルゴリズム：
    1. ROSの$(find ...)構文を解決
    2. 探索候補を生成（親/子ディレクトリ、兄弟ディレクトリ）
    3. 各候補の存在を確認
    
    Args:
        relative_path: 解決したい相対パス（ファイル名または相対パス）
        base_file_path: 基準となるファイルのパス（xacroファイルなど）
        parent_depth: 親ディレクトリの探索階層数（デフォルト: 1）
        child_depth: 子ディレクトリの探索階層数（デフォルト: 1）
        verbose: 詳細なログを出力するかどうか
    
    Returns:
        見つかったファイルの絶対パス、見つからない場合はNone
    """
    if not relative_path:
        return None
    
    # $(find package_name)構文を解決
    if '$(' in relative_path and 'find' in relative_path:
        relative_path = resolve_ros_find_syntax(relative_path, base_file_path, verbose)
        # 解決後、絶対パスになっている可能性がある
        if os.path.isabs(relative_path) and os.path.exists(relative_path) and os.path.isfile(relative_path):
            if verbose:
                print(f"  Resolved ROS find syntax: {relative_path}")
            return relative_path
    
    # 基準ファイルのディレクトリ
    base_dir = os.path.dirname(os.path.abspath(base_file_path))
    
    # 探索候補を生成
    candidates = generate_search_candidates(relative_path, base_dir, parent_depth, child_depth)
    
    # 各候補の存在を確認
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isfile(candidate):
            if verbose:
                print(f"  Resolved path: {relative_path} -> {candidate}")
            return candidate
    
    if verbose:
        print(f"  Could not resolve path: {relative_path} (searched {len(candidates)} candidates)")
    return None


def resolve_path_in_xml_element(elem, attr_names, xacro_file_path, verbose=False):
    """XML要素の指定された属性のパスを解決する。
    
    汎用的なXML属性パス解決関数：
    1. 指定された属性名のリストから値を取得
    2. $(find ...)構文を解決
    3. 相対パスを解決
    4. 解決されたパスを属性に設定
    
    Args:
        elem: XML要素
        attr_names: パスを含む可能性のある属性名のリスト（例: ['filename', 'file', 'uri']）
        xacro_file_path: xacroファイルのパス
        verbose: 詳細なログを出力するかどうか
    
    Returns:
        パスが解決されて変更された場合はTrue、そうでない場合はFalse
    """
    modified = False
    
    # 指定された属性名から値を取得
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
    
    # $(find ...)構文を解決
    if '$(' in path_value and 'find' in path_value:
        resolved_path = resolve_ros_find_syntax(path_value, xacro_file_path, verbose)
        if resolved_path != path_value:
            elem.set(attr_name, resolved_path)
            modified = True
            if verbose:
                print(f"  Resolved ROS find in {attr_name}: {original_path} -> {resolved_path}")
            path_value = resolved_path
    
    # 相対パスを解決（package://は除外）
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

class URDFParser:
    """Parser for URDF (Unified Robot Description Format) files.

    This class provides methods to parse URDF XML files and extract
    robot structure information including links, joints, materials,
    and mesh file paths.

    Usage:
        parser = URDFParser()
        result = parser.parse_urdf('/path/to/robot.urdf')
        
        # Access parsed data
        robot_name = result['robot_name']
        links_data = result['links']
        joints_data = result['joints']
        materials_data = result['materials']
    """

    def __init__(self, verbose=True):
        """Initialize URDF parser.

        Args:
            verbose: If True, print parsing progress and information
        """
        self.verbose = verbose

    def _expand_xacro(self, xacro_file_path):
        """Expand xacro file to URDF XML string using xacrodoc.

        Args:
            xacro_file_path: Path to xacro file

        Returns:
            str: Expanded URDF XML string

        Raises:
            FileNotFoundError: If xacro file does not exist
            RuntimeError: If xacrodoc is not available or expansion fails
        """
        if not os.path.exists(xacro_file_path):
            raise FileNotFoundError(f"Xacro file not found: {xacro_file_path}")

        if self.verbose:
            print(f"Expanding xacro file using xacrodoc: {xacro_file_path}")

        # Try xacrodoc library
        try:
            from xacrodoc import XacroDoc
            
            if self.verbose:
                print("Using xacrodoc library")
            
            # xacroファイルからXacroDocオブジェクトを作成
            xacro_file_abs = os.path.abspath(xacro_file_path)
            doc = XacroDoc.from_file(xacro_file_abs)
            
            # URDF文字列に変換
            urdf_xml_string = doc.to_urdf_string()
            
            # 生成されたXMLが正しくパースできるか確認
            if urdf_xml_string:
                try:
                    root = ET.fromstring(urdf_xml_string)
                    # robot要素の存在を確認
                    if root.tag != 'robot':
                        if self.verbose:
                            print(f"  Warning: Root element is '{root.tag}', expected 'robot'")
                        # robot要素が見つからない場合は、子要素を探す
                        robot_elem = root.find('robot')
                        if robot_elem is None:
                            # ルート要素がrobotでない場合は警告を出すが、続行
                            if self.verbose:
                                print(f"  Warning: No 'robot' element found in expanded xacro")
                    else:
                        if self.verbose:
                            print(f"  Successfully expanded xacro to URDF (robot element found)")
                    
                    # デバッグ: 展開されたXMLの構造を確認
                    if self.verbose:
                        link_count = len(root.findall('link'))
                        joint_count = len(root.findall('joint'))
                        print(f"  Found {link_count} link elements, {joint_count} joint elements in expanded XML")
                        if link_count == 0:
                            print(f"  WARNING: No links found in expanded xacro!")
                            # XMLの最初の500文字を出力してデバッグ
                            print(f"  First 500 chars of expanded XML:")
                            print(f"  {urdf_xml_string[:500]}")
                    
                    return urdf_xml_string
                except ET.ParseError as parse_err:
                    if self.verbose:
                        print(f"  Warning: Generated XML has parse error: {parse_err}")
                        print(f"  First 200 chars: {urdf_xml_string[:200]}")
                    raise RuntimeError(f"xacrodoc generated invalid XML: {parse_err}")
            else:
                raise RuntimeError("xacrodoc returned empty result")
                
        except ImportError:
            error_msg = (
                "xacrodoc library not found. Please install it using:\n\n"
                "  pip install xacrodoc\n\n"
                f"Xacro file: {xacro_file_path}"
            )
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = (
                f"Failed to expand xacro file using xacrodoc: {e}\n"
                f"File: {xacro_file_path}\n"
            )
            if self.verbose:
                import traceback
                traceback.print_exc()
            raise RuntimeError(error_msg)

    def parse_urdf(self, urdf_file_path):
        """Parse URDF file and extract robot structure information.
        
        Supports both .urdf and .xacro files. If a .xacro file is provided,
        it will be automatically expanded before parsing.

        Args:
            urdf_file_path: Path to URDF or xacro file

        Returns:
            Dictionary containing:
                - robot_name: Name of the robot
                - links: Dictionary of link data
                - joints: List of joint data
                - materials: Dictionary of material colors
                - missing_meshes: List of meshes that could not be found

        Raises:
            FileNotFoundError: If URDF/xacro file does not exist
            ET.ParseError: If URDF file is not valid XML
            RuntimeError: If xacro expansion fails
        """
        if not os.path.exists(urdf_file_path):
            raise FileNotFoundError(f"URDF/xacro file not found: {urdf_file_path}")

        # Check if file is a xacro file
        file_ext = os.path.splitext(urdf_file_path)[1].lower()
        is_xacro = file_ext in ['.xacro', '.xacro.urdf']

        if is_xacro:
            if self.verbose:
                print(f"Detected xacro file: {urdf_file_path}")
            
            # Expand xacro to URDF
            urdf_xml_string = self._expand_xacro(urdf_file_path)
            
            # 展開されたXML文字列が有効か確認
            if not urdf_xml_string or not urdf_xml_string.strip():
                raise RuntimeError("xacro expansion returned empty string")
            
            if self.verbose:
                print(f"  Expanded xacro XML length: {len(urdf_xml_string)} characters")
            
            # Parse expanded URDF XML string
            try:
                root = ET.fromstring(urdf_xml_string)
                
                # デバッグ: 展開されたXMLの構造を確認
                if self.verbose:
                    print(f"  Expanded XML root tag: {root.tag}")
                    # 最初の数個のリンクとジョイントを確認
                    link_count = len(root.findall('link'))
                    joint_count = len(root.findall('joint'))
                    print(f"  Found {link_count} link elements, {joint_count} joint elements in expanded XML")
                    if link_count == 0:
                        print(f"  WARNING: No links found in expanded xacro!")
                        # XMLの最初の500文字を出力してデバッグ
                        print(f"  First 500 chars of expanded XML:")
                        print(f"  {urdf_xml_string[:500]}")
                
                # robot要素の存在を確認
                if root.tag != 'robot':
                    if self.verbose:
                        print(f"  Warning: Root element is '{root.tag}', expected 'robot'")
                    # robot要素が見つからない場合は、子要素を探す
                    robot_elem = root.find('robot')
                    if robot_elem is not None:
                        root = robot_elem
                        if self.verbose:
                            print(f"  Found 'robot' element as child, using it")
                    else:
                        if self.verbose:
                            print(f"  Warning: No 'robot' element found, but continuing with root element '{root.tag}'")
                else:
                    if self.verbose:
                        print(f"  Successfully parsed expanded xacro XML (robot element found)")
            except ET.ParseError as e:
                if self.verbose:
                    print(f"Error parsing expanded xacro XML:")
                    print(f"  First 500 chars: {urdf_xml_string[:500]}")
                    print(f"  Last 200 chars: {urdf_xml_string[-200:]}")
                raise RuntimeError(f"Failed to parse expanded xacro XML: {e}\nFirst 500 chars: {urdf_xml_string[:500]}")
        else:
            if self.verbose:
                print(f"Parsing URDF from: {urdf_file_path}")
            
            # Parse URDF file directly
            tree = ET.parse(urdf_file_path)
            root = tree.getroot()
            
            # robot要素の存在を確認
            if root.tag != 'robot':
                if self.verbose:
                    print(f"  Warning: Root element is '{root.tag}', expected 'robot'")
                # robot要素が見つからない場合は、子要素を探す
                robot_elem = root.find('robot')
                if robot_elem is not None:
                    root = robot_elem
                    if self.verbose:
                        print(f"  Found 'robot' element as child, using it")

        # Get robot name from robot element or file name
        robot_name = root.get('name')
        if not robot_name:
            # robot要素にname属性がない場合は、ファイル名から取得
            robot_name = os.path.splitext(os.path.basename(urdf_file_path))[0]
            if self.verbose:
                print(f"  Robot name not found in XML, using filename: {robot_name}")
        else:
            if self.verbose:
                print(f"Robot name from XML: {robot_name}")

        # Extract data
        materials_data = self._parse_materials(root)
        links_data, missing_meshes = self._parse_links(root, urdf_file_path, materials_data)
        joints_data = self._parse_joints(root, links_data)

        return {
            'robot_name': robot_name,
            'links': links_data,
            'joints': joints_data,
            'materials': materials_data,
            'missing_meshes': missing_meshes
        }

    def _parse_materials(self, root):
        """Parse material information from URDF.

        Args:
            root: XML root element

        Returns:
            Dictionary mapping material names to RGB colors
        """
        materials_data = {}

        for material_elem in root.findall('material'):
            mat_name = material_elem.get('name')
            color_elem = material_elem.find('color')
            if color_elem is not None:
                rgba_str = color_elem.get('rgba', '1.0 1.0 1.0 1.0')
                rgba = [float(v) for v in rgba_str.split()]
                materials_data[mat_name] = rgba[:3]  # RGB only

        if self.verbose:
            print(f"Parsed {len(materials_data)} materials")

        return materials_data

    def _parse_links(self, root, urdf_file_path, materials_data):
        """Parse link information from URDF.

        Args:
            root: XML root element
            urdf_file_path: Path to URDF file (for resolving mesh paths)
            materials_data: Dictionary of material colors

        Returns:
            Tuple of (links_data dict, missing_meshes list)
        """
        links_data = {}
        missing_meshes = []

        # デバッグ: リンク要素の数を確認
        link_elems = root.findall('link')
        if self.verbose:
            print(f"\n[URDFParser] Found {len(link_elems)} link elements in URDF")

        for link_elem in link_elems:
            link_name = link_elem.get('name')
            if not link_name:
                if self.verbose:
                    print(f"  Warning: Found link element without name attribute, skipping")
                continue
            
            if self.verbose:
                print(f"  Parsing link: {link_name}")
            link_data = {
                'name': link_name,
                'mass': 0.0,
                'inertia': {'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0},
                'inertial_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                'visual_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                'stl_file': None,
                'color': [1.0, 1.0, 1.0],
                'stl_filename_original': None,
                'mesh_scale': [1.0, 1.0, 1.0],
                'decorations': [],
                'collision_mesh': None
            }

            # Parse inertial information
            inertial_elem = link_elem.find('inertial')
            if inertial_elem is not None:
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    link_data['mass'] = float(mass_elem.get('value', 0.0))

                inertia_elem = inertial_elem.find('inertia')
                if inertia_elem is not None:
                    for key in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                        link_data['inertia'][key] = float(inertia_elem.get(key, 0.0))

                origin_elem = inertial_elem.find('origin')
                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0 0 0')
                    rpy_str = origin_elem.get('rpy', '0 0 0')
                    link_data['inertial_origin']['xyz'] = [float(v) for v in xyz_str.split()]
                    link_data['inertial_origin']['rpy'] = [float(v) for v in rpy_str.split()]

                if self.verbose:
                    print(f"\n[URDF_INERTIAL] link_name={link_name}")
                    print(f"  mass={link_data['mass']:.9e}")
                    print(f"  origin_xyz={link_data['inertial_origin']['xyz']}")
                    print(f"  origin_rpy={link_data['inertial_origin']['rpy']}")

            # Parse visual information (multiple visual tags supported)
            visual_elems = link_elem.findall('visual')
            for visual_idx, visual_elem in enumerate(visual_elems):
                is_main_visual = (visual_idx == 0)

                current_stl_path = None
                current_color = [1.0, 1.0, 1.0]
                current_visual_origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}

                # Parse visual origin
                visual_origin_elem = visual_elem.find('origin')
                if visual_origin_elem is not None:
                    xyz_str = visual_origin_elem.get('xyz', '0 0 0')
                    rpy_str = visual_origin_elem.get('rpy', '0 0 0')
                    try:
                        xyz_values = [float(v) for v in xyz_str.split()]
                        rpy_values = [float(v) for v in rpy_str.split()]
                        if len(xyz_values) == 3:
                            current_visual_origin['xyz'] = xyz_values
                        if len(rpy_values) == 3:
                            current_visual_origin['rpy'] = rpy_values
                    except ValueError as e:
                        if self.verbose:
                            print(f"Warning: Invalid visual origin values: {e}")

                # Parse geometry
                geometry_elem = visual_elem.find('geometry')
                mesh_scale = [1.0, 1.0, 1.0]
                
                if geometry_elem is not None:
                    mesh_elem = geometry_elem.find('mesh')
                    if mesh_elem is not None:
                        mesh_filename = mesh_elem.get('filename', '')
                        
                        # Parse scale attribute
                        scale_str = mesh_elem.get('scale', '')
                        if scale_str:
                            try:
                                scale_values = [float(v) for v in scale_str.split()]
                                if len(scale_values) == 3:
                                    mesh_scale = scale_values
                            except ValueError:
                                if self.verbose:
                                    print(f"Warning: Invalid scale attribute '{scale_str}'")

                        # Resolve mesh file path
                        resolved_path = self._resolve_mesh_path(mesh_filename, urdf_file_path)
                        
                        if resolved_path:
                            current_stl_path = resolved_path
                        else:
                            if self.verbose:
                                print(f"Could not find mesh file for link {link_name}: {mesh_filename}")
                            if is_main_visual:
                                link_data['stl_filename_original'] = mesh_filename
                                mesh_basename = os.path.basename(mesh_filename)
                                missing_meshes.append({
                                    'link_name': link_name,
                                    'filename': mesh_filename,
                                    'basename': mesh_basename
                                })

                # Parse material color
                material_elem = visual_elem.find('material')
                if material_elem is not None:
                    mat_name = material_elem.get('name')
                    if mat_name in materials_data:
                        current_color = materials_data[mat_name]
                    else:
                        color_elem = material_elem.find('color')
                        if color_elem is not None:
                            rgba_str = color_elem.get('rgba', '1.0 1.0 1.0 1.0')
                            rgba = [float(v) for v in rgba_str.split()]
                            current_color = rgba[:3]
                        elif mat_name and mat_name.startswith('#'):
                            # Hex color code
                            hex_color = mat_name[1:]
                            if len(hex_color) == 6:
                                r = int(hex_color[0:2], 16) / 255.0
                                g = int(hex_color[2:4], 16) / 255.0
                                b = int(hex_color[4:6], 16) / 255.0
                                current_color = [r, g, b]

                # Store data
                if is_main_visual:
                    if current_stl_path:
                        link_data['stl_file'] = current_stl_path
                    link_data['color'] = current_color
                    link_data['mesh_scale'] = mesh_scale
                    link_data['visual_origin'] = current_visual_origin
                else:
                    # Decoration visual
                    if current_stl_path:
                        stl_name = os.path.splitext(os.path.basename(current_stl_path))[0]
                        decoration_data = {
                            'name': stl_name,
                            'stl_file': current_stl_path,
                            'color': current_color,
                            'mesh_scale': mesh_scale,
                            'visual_origin': current_visual_origin
                        }
                        link_data['decorations'].append(decoration_data)

            # Parse collision information
            collision_elems = link_elem.findall('collision')
            for collision_elem in collision_elems:
                geometry_elem = collision_elem.find('geometry')
                if geometry_elem is not None:
                    # Check for mesh collision
                    mesh_elem = geometry_elem.find('mesh')
                    if mesh_elem is not None:
                        mesh_filename = mesh_elem.get('filename', '')
                        resolved_path = self._resolve_mesh_path(mesh_filename, urdf_file_path)
                        if resolved_path:
                            link_data['collision_mesh'] = resolved_path
                            link_data['collider_type'] = 'mesh'
                            link_data['collider_enabled'] = True
                            break
                    
                    # Check for primitive collision geometries
                    box_elem = geometry_elem.find('box')
                    sphere_elem = geometry_elem.find('sphere')
                    cylinder_elem = geometry_elem.find('cylinder')
                    
                    if box_elem is not None:
                        # Parse box collision
                        size_str = box_elem.get('size', '')
                        if size_str:
                            sizes = [float(s) for s in size_str.split()]
                            if len(sizes) >= 3:
                                collider_data = {
                                    'type': 'box',
                                    'geometry': {
                                        'size_x': sizes[0],
                                        'size_y': sizes[1],
                                        'size_z': sizes[2]
                                    },
                                    'position': [0.0, 0.0, 0.0],
                                    'rotation': [0.0, 0.0, 0.0]  # degrees
                                }
                                # Parse origin if present
                                origin_elem = collision_elem.find('origin')
                                if origin_elem is not None:
                                    xyz_str = origin_elem.get('xyz', '0 0 0')
                                    rpy_str = origin_elem.get('rpy', '0 0 0')
                                    collider_data['position'] = [float(v) for v in xyz_str.split()]
                                    rpy_rad = [float(v) for v in rpy_str.split()]
                                    collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]
                                
                                link_data['collider_data'] = collider_data
                                link_data['collider_type'] = 'primitive'
                                link_data['collider_enabled'] = True
                                break
                    
                    elif sphere_elem is not None:
                        # Parse sphere collision
                        radius_str = sphere_elem.get('radius', '0.5')
                        radius = float(radius_str)
                        collider_data = {
                            'type': 'sphere',
                            'geometry': {
                                'radius': radius
                            },
                            'position': [0.0, 0.0, 0.0],
                            'rotation': [0.0, 0.0, 0.0]  # degrees
                        }
                        # Parse origin if present
                        origin_elem = collision_elem.find('origin')
                        if origin_elem is not None:
                            xyz_str = origin_elem.get('xyz', '0 0 0')
                            rpy_str = origin_elem.get('rpy', '0 0 0')
                            collider_data['position'] = [float(v) for v in xyz_str.split()]
                            rpy_rad = [float(v) for v in rpy_str.split()]
                            collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]
                        
                        link_data['collider_data'] = collider_data
                        link_data['collider_type'] = 'primitive'
                        link_data['collider_enabled'] = True
                        break
                    
                    elif cylinder_elem is not None:
                        # Parse cylinder collision
                        radius_str = cylinder_elem.get('radius', '0.5')
                        length_str = cylinder_elem.get('length', '1.0')
                        radius = float(radius_str)
                        length = float(length_str)
                        collider_data = {
                            'type': 'cylinder',
                            'geometry': {
                                'radius': radius,
                                'length': length
                            },
                            'position': [0.0, 0.0, 0.0],
                            'rotation': [0.0, 0.0, 0.0]  # degrees
                        }
                        # Parse origin if present
                        origin_elem = collision_elem.find('origin')
                        if origin_elem is not None:
                            xyz_str = origin_elem.get('xyz', '0 0 0')
                            rpy_str = origin_elem.get('rpy', '0 0 0')
                            collider_data['position'] = [float(v) for v in xyz_str.split()]
                            rpy_rad = [float(v) for v in rpy_str.split()]
                            collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]
                        
                        link_data['collider_data'] = collider_data
                        link_data['collider_type'] = 'primitive'
                        link_data['collider_enabled'] = True
                        break

            links_data[link_name] = link_data

        if self.verbose:
            print(f"\n[URDFParser] Successfully parsed {len(links_data)} links")
            if missing_meshes:
                print(f"  Warning: {len(missing_meshes)} mesh files could not be found")
                for missing in missing_meshes[:5]:  # 最初の5つだけ表示
                    print(f"    - {missing['link_name']}: {missing['basename']}")
                if len(missing_meshes) > 5:
                    print(f"    ... and {len(missing_meshes) - 5} more")

        return links_data, missing_meshes

    def _resolve_mesh_path(self, mesh_filename, urdf_file_path):
        """Resolve mesh file path from URDF reference.

        Tries multiple methods to find the mesh file:
        1. package:// paths relative to description directory
        2. Absolute paths
        3. Relative paths from URDF directory
        4. Search in common mesh directories

        Args:
            mesh_filename: Mesh file reference from URDF
            urdf_file_path: Path to URDF file

        Returns:
            Absolute path to mesh file, or None if not found
        """
        urdf_dir = os.path.dirname(os.path.abspath(urdf_file_path))
        urdf_file_abs = os.path.abspath(urdf_file_path)
        description_dir = os.path.dirname(urdf_dir)  # メソッドの最初で定義
        mesh_basename = os.path.basename(mesh_filename)

        # Handle package:// paths
        if mesh_filename.startswith('package://'):
            parts = mesh_filename.split('/')
            if len(parts) > 2:
                package_name = parts[2]
                relative_path = '/'.join(parts[3:]) if len(parts) > 3 else ''
                
                # パッケージルートを探す（xacroファイルの場所から）
                package_root = find_package_root(urdf_file_abs, package_name, max_search_depth=10, verbose=self.verbose)
                
                if package_root:
                    # パッケージルートからの相対パス
                    if relative_path:
                        candidate = os.path.join(package_root, relative_path)
                    else:
                        candidate = os.path.join(package_root, mesh_basename)
                    
                    if os.path.exists(candidate):
                        if self.verbose:
                            print(f"  Found mesh (package://): {candidate}")
                        return candidate
                
                # パッケージルートが見つからない場合、従来の方法を試す
                if relative_path:
                    relative_path_full = relative_path
                else:
                    relative_path_full = os.path.join('meshes', mesh_basename)

                # Try multiple locations
                candidates = [
                    os.path.join(package_root, relative_path_full) if package_root else None,
                    os.path.join(description_dir, relative_path_full),
                    os.path.join(urdf_dir, relative_path_full),
                    os.path.join(description_dir, 'meshes', mesh_basename),
                    os.path.normpath(os.path.join(urdf_dir, '..', 'meshes', mesh_basename)),
                    os.path.join(os.path.dirname(description_dir), relative_path_full),
                    # xacroファイルと同じディレクトリ構造を試す
                    os.path.normpath(os.path.join(urdf_dir, '..', '..', 'meshes', mesh_basename)),
                    os.path.normpath(os.path.join(urdf_dir, '..', '..', package_name, 'meshes', mesh_basename)) if package_name else None,
                ]

                for candidate in candidates:
                    if candidate and os.path.exists(candidate):
                        if self.verbose:
                            print(f"  Found mesh (package:// fallback): {candidate}")
                        return candidate

        else:
            # Absolute or relative paths
            candidates = []
            
            # Try absolute path
            if os.path.isabs(mesh_filename) and os.path.exists(mesh_filename):
                return mesh_filename

            # Try relative paths
            candidates = [
                os.path.join(urdf_dir, mesh_filename),
                os.path.join(description_dir, mesh_filename),
                os.path.join(description_dir, 'meshes', mesh_basename),
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    if self.verbose:
                        print(f"  Found mesh: {candidate}")
                    return candidate

        return None

    def _parse_joints(self, root, links_data):
        """Parse joint information from URDF.

        Args:
            root: XML root element
            links_data: Dictionary of link data

        Returns:
            List of joint data dictionaries
        """
        joints_data = []

        # デバッグ: ジョイント要素の数を確認
        joint_elems = root.findall('joint')
        if self.verbose:
            print(f"\n[URDFParser] Found {len(joint_elems)} joint elements in URDF")

        for joint_elem in joint_elems:
            joint_data = {
                'name': joint_elem.get('name'),
                'type': joint_elem.get('type', 'fixed'),
                'parent': None,
                'child': None,
                'origin_xyz': [0.0, 0.0, 0.0],
                'origin_rpy': [0.0, 0.0, 0.0],
                'axis': [1.0, 0.0, 0.0],
                'limit': {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 'velocity': 3.0, 'friction': 0.05},
                'dynamics': {'damping': 0.0, 'friction': 0.0}
            }

            # Parse parent and child links
            parent_elem = joint_elem.find('parent')
            if parent_elem is not None:
                joint_data['parent'] = parent_elem.get('link')

            child_elem = joint_elem.find('child')
            if child_elem is not None:
                joint_data['child'] = child_elem.get('link')

            # Parse origin
            origin_elem = joint_elem.find('origin')
            if origin_elem is not None:
                xyz_str = origin_elem.get('xyz', '0 0 0')
                rpy_str = origin_elem.get('rpy', '0 0 0')
                joint_data['origin_xyz'] = [float(v) for v in xyz_str.split()]
                joint_data['origin_rpy'] = [float(v) for v in rpy_str.split()]

            # Parse axis
            axis_elem = joint_elem.find('axis')
            if axis_elem is not None:
                axis_str = axis_elem.get('xyz', '1 0 0')
                joint_data['axis'] = [float(v) for v in axis_str.split()]

            # Parse limit
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                joint_data['limit']['lower'] = float(limit_elem.get('lower', -3.14159))
                joint_data['limit']['upper'] = float(limit_elem.get('upper', 3.14159))
                joint_data['limit']['effort'] = float(limit_elem.get('effort', 10.0))
                joint_data['limit']['velocity'] = float(limit_elem.get('velocity', 3.0))
                joint_data['limit']['friction'] = float(limit_elem.get('friction', 0.05))

            # Parse dynamics
            dynamics_elem = joint_elem.find('dynamics')
            if dynamics_elem is not None:
                joint_data['dynamics']['damping'] = float(dynamics_elem.get('damping', 0.0))
                joint_data['dynamics']['friction'] = float(dynamics_elem.get('friction', 0.0))

            joints_data.append(joint_data)

        if self.verbose:
            print(f"[URDFParser] Successfully parsed {len(joints_data)} joints")

        # Detect root links and connect to base_link
        if self.verbose:
            print("\n=== Detecting root links ===")

        child_links = set()
        for joint_data in joints_data:
            if joint_data['child']:
                child_links.add(joint_data['child'])

        root_links = []
        for link_name in links_data.keys():
            if link_name not in child_links and link_name not in ['base_link', 'BaseLink']:
                root_links.append(link_name)
                if self.verbose:
                    print(f"  Found root link: {link_name}")

        # Create synthetic joints from base_link to root links
        if root_links:
            if self.verbose:
                print(f"  Connecting {len(root_links)} root link(s) to base_link")

            for root_link_name in root_links:
                synthetic_joint = {
                    'name': f'base_to_{root_link_name}',
                    'type': 'fixed',
                    'parent': 'base_link',
                    'child': root_link_name,
                    'origin_xyz': [0.0, 0.0, 0.0],
                    'origin_rpy': [0.0, 0.0, 0.0],
                    'axis': [1.0, 0.0, 0.0],
                    'limit': {'lower': 0.0, 'upper': 0.0, 'effort': 0.0, 'velocity': 0.0, 'friction': 0.0},
                    'dynamics': {'damping': 0.0, 'friction': 0.0}
                }
                joints_data.append(synthetic_joint)
                if self.verbose:
                    print(f"  Created synthetic joint: base_link -> {root_link_name}")

        if self.verbose:
            print(f"Parsed {len(joints_data)} joints (including synthetic joints)")

        return joints_data


# ============================================================================
# MJCF PARSER
# ============================================================================

class MJCFParser:
    """Parser for MJCF (MuJoCo Model Format) files.

    This class provides methods to parse MJCF XML files and extract
    robot structure information including bodies, joints, meshes,
    and default classes.

    Usage:
        parser = MJCFParser()
        result = parser.parse_mjcf('/path/to/robot.xml', working_dir='/tmp')
        
        # Access parsed data
        robot_name = result['robot_name']
        bodies_data = result['bodies']
        joints_data = result['joints']
        meshes_data = result['meshes']
    """

    def __init__(self, verbose=True):
        """Initialize MJCF parser.

        Args:
            verbose: If True, print parsing progress and information
        """
        self.verbose = verbose
        self.conversion_utils = ConversionUtils()

    def parse_mjcf(self, mjcf_file_path, working_dir=None):
        """Parse MJCF file and extract robot structure information.

        Args:
            mjcf_file_path: Path to MJCF XML file
            working_dir: Working directory for resolving mesh paths (optional)

        Returns:
            Dictionary containing:
                - robot_name: Name of the robot
                - bodies: List of body data
                - joints: List of joint data
                - meshes: Dictionary of mesh information
                - eulerseq: Euler sequence from compiler settings
                - default_classes: Dictionary of default class settings

        Raises:
            FileNotFoundError: If MJCF file does not exist
            ET.ParseError: If MJCF file is not valid XML
            ValueError: If root element is not 'mujoco'
        """
        if not os.path.exists(mjcf_file_path):
            raise FileNotFoundError(f"MJCF file not found: {mjcf_file_path}")

        if self.verbose:
            print(f"Parsing MJCF from: {mjcf_file_path}")

        # Parse MJCF file
        tree = ET.parse(mjcf_file_path)
        root = tree.getroot()

        if root.tag != 'mujoco':
            raise ValueError("Root element must be 'mujoco' for valid MJCF file")

        # Get robot name from file name
        robot_name = os.path.splitext(os.path.basename(mjcf_file_path))[0]
        if self.verbose:
            print(f"Robot name: {robot_name}")

        # Parse compiler settings
        eulerseq = self._parse_compiler(root)

        # Parse default classes
        default_classes = self._parse_defaults(root)

        # Parse asset meshes
        if working_dir is None:
            working_dir = os.path.dirname(mjcf_file_path)
        meshes_data = self._parse_assets(root, mjcf_file_path, working_dir)

        # Parse body hierarchy
        bodies_data = []
        joints_data = []
        worldbody = root.find('worldbody')
        if worldbody is not None:
            for body_elem in worldbody.findall('body'):
                self._parse_body(body_elem, None, 0, bodies_data, joints_data, 
                                meshes_data, default_classes, eulerseq)

        if self.verbose:
            print(f"Parsed {len(bodies_data)} bodies and {len(joints_data)} joints")

        return {
            'robot_name': robot_name,
            'bodies': bodies_data,
            'joints': joints_data,
            'meshes': meshes_data,
            'eulerseq': eulerseq,
            'default_classes': default_classes
        }

    def _parse_compiler(self, root):
        """Parse compiler settings from MJCF.

        Args:
            root: XML root element

        Returns:
            Euler sequence string (default 'xyz')
        """
        eulerseq = 'xyz'  # Default
        compiler_elem = root.find('compiler')
        if compiler_elem is not None:
            eulerseq_attr = compiler_elem.get('eulerseq')
            if eulerseq_attr:
                eulerseq = eulerseq_attr.lower()
                if self.verbose:
                    print(f"Compiler eulerseq: {eulerseq}")

        return eulerseq

    def _parse_defaults(self, root):
        """Parse default classes from MJCF.

        Args:
            root: XML root element

        Returns:
            Dictionary of default class settings
        """
        default_classes = {}

        def parse_defaults_recursive(default_elem, parent_class_name=None):
            """Recursively parse default classes."""
            class_name = default_elem.get('class', parent_class_name)

            # Inherit from parent class
            class_defaults = {}
            if parent_class_name and parent_class_name in default_classes:
                class_defaults = default_classes[parent_class_name].copy()

            # Get joint defaults
            joint_elem = default_elem.find('joint')
            if joint_elem is not None:
                axis_str = joint_elem.get('axis')
                if axis_str:
                    class_defaults['joint_axis'] = [float(v) for v in axis_str.split()]

                range_str = joint_elem.get('range')
                if range_str:
                    class_defaults['joint_range'] = [float(v) for v in range_str.split()]

                damping_str = joint_elem.get('damping')
                if damping_str:
                    class_defaults['joint_damping'] = float(damping_str)

                armature_str = joint_elem.get('armature')
                if armature_str:
                    class_defaults['joint_armature'] = float(armature_str)

                frictionloss_str = joint_elem.get('frictionloss')
                if frictionloss_str:
                    class_defaults['joint_frictionloss'] = float(frictionloss_str)

                stiffness_str = joint_elem.get('stiffness')
                if stiffness_str:
                    class_defaults['joint_stiffness'] = float(stiffness_str)

            # Save class if it has a name
            if class_name:
                default_classes[class_name] = class_defaults
                if self.verbose:
                    print(f"Default class '{class_name}': {class_defaults}")

            # Parse child default classes recursively
            for child_default in default_elem.findall('default'):
                parse_defaults_recursive(child_default, class_name)

        # Find and parse all default elements
        for default_elem in root.findall('default'):
            parse_defaults_recursive(default_elem)

        if self.verbose:
            print(f"Parsed {len(default_classes)} default classes")

        return default_classes

    def _parse_assets(self, root, mjcf_file_path, working_dir):
        """Parse asset mesh information from MJCF.

        Args:
            root: XML root element
            mjcf_file_path: Path to MJCF file
            working_dir: Working directory for mesh search

        Returns:
            Dictionary mapping mesh names to mesh info (path and scale)
        """
        meshes_data = {}
        asset_elem = root.find('asset')
        
        if asset_elem is None:
            return meshes_data

        mjcf_dir = os.path.dirname(mjcf_file_path)

        # Mesh search directories
        search_dirs = [
            mjcf_dir,
            os.path.join(mjcf_dir, 'assets'),
            os.path.join(mjcf_dir, 'meshes'),
            os.path.join(mjcf_dir, '..', 'assets'),
            os.path.join(mjcf_dir, '..', 'meshes'),
            working_dir,
            os.path.join(working_dir, 'assets'),
            os.path.join(working_dir, 'meshes'),
        ]

        for mesh_elem in asset_elem.findall('mesh'):
            mesh_name = mesh_elem.get('name')
            mesh_file = mesh_elem.get('file', '')

            # Derive mesh name from filename if not specified
            if not mesh_name and mesh_file:
                mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
                if self.verbose:
                    print(f"Mesh name derived from file: {mesh_name}")

            # Parse mesh scale attribute
            mesh_scale = [1.0, 1.0, 1.0]
            scale_str = mesh_elem.get('scale')
            if scale_str:
                scale_values = [float(v) for v in scale_str.split()]
                if len(scale_values) == 3:
                    mesh_scale = scale_values
                elif len(scale_values) == 1:
                    mesh_scale = [scale_values[0]] * 3
                if self.verbose:
                    print(f"  Mesh '{mesh_name}' scale: {mesh_scale}")

            # Resolve mesh file path
            if mesh_file:
                mesh_path = None
                mesh_basename = os.path.basename(mesh_file)

                # Method 1: Try original path relative to MJCF dir
                candidate = os.path.join(mjcf_dir, mesh_file)
                if os.path.exists(candidate):
                    mesh_path = candidate
                    if self.verbose:
                        print(f"Found mesh: {mesh_name} -> {mesh_path}")

                # Method 2: Try in search directories with original structure
                if not mesh_path:
                    for search_dir in search_dirs:
                        if not search_dir or not os.path.exists(search_dir):
                            continue
                        candidate = os.path.join(search_dir, mesh_file)
                        if os.path.exists(candidate):
                            mesh_path = candidate
                            if self.verbose:
                                print(f"Found mesh: {mesh_name} -> {mesh_path}")
                            break

                # Method 3: Search for basename in directories
                if not mesh_path:
                    for search_dir in search_dirs:
                        if not search_dir or not os.path.exists(search_dir):
                            continue
                        candidate = os.path.join(search_dir, mesh_basename)
                        if os.path.exists(candidate):
                            mesh_path = candidate
                            if self.verbose:
                                print(f"Found mesh: {mesh_name} -> {mesh_path}")
                            break

                if mesh_path:
                    meshes_data[mesh_name] = {
                        'path': mesh_path,
                        'scale': mesh_scale
                    }
                else:
                    if self.verbose:
                        print(f"Warning: Could not find mesh file: {mesh_file}")

        if self.verbose:
            print(f"Parsed {len(meshes_data)} mesh assets")

        return meshes_data

    def _parse_body(self, body_elem, parent_name, level, bodies_data, joints_data, 
                    meshes_data, default_classes, eulerseq):
        """Recursively parse body element from MJCF.

        Args:
            body_elem: Body XML element
            parent_name: Name of parent body (None for root)
            level: Recursion level for indentation
            bodies_data: List to append body data to
            joints_data: List to append joint data to
            meshes_data: Dictionary of mesh information
            default_classes: Dictionary of default class settings
            eulerseq: Euler sequence from compiler settings
        """
        body_name = body_elem.get('name', f'body_{len(bodies_data)}')

        body_data = {
            'name': body_name,
            'parent': parent_name,
            'mass': 0.0,
            'inertia': {'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0},
            'inertial_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
            'stl_file': None,
            'visuals': [],
            'color': [1.0, 1.0, 1.0],
            'collision_mesh': None,
            'mesh_scale': [1.0, 1.0, 1.0],
            'visual_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
            'pos': [0.0, 0.0, 0.0],
            'quat': [1.0, 0.0, 0.0, 0.0],
            'rpy': [0.0, 0.0, 0.0]
        }

        # Parse position
        pos_str = body_elem.get('pos', '0 0 0')
        body_data['pos'] = [float(v) for v in pos_str.split()]

        # Parse quaternion and convert to RPY
        quat_str = body_elem.get('quat')
        if quat_str:
            body_data['quat'] = [float(v) for v in quat_str.split()]
            body_data['rpy'] = self.conversion_utils.quat_to_rpy(body_data['quat'])
            if self.verbose:
                print(f"{'  ' * level}  Body quat: {body_data['quat']} -> rpy: {body_data['rpy']}")

        # Parse euler angles
        euler_str = body_elem.get('euler')
        if euler_str:
            euler_degrees = [float(v) for v in euler_str.split()]
            body_data['rpy'] = [math.radians(e) for e in euler_degrees]
            if self.verbose:
                print(f"{'  ' * level}  Body euler (deg): {euler_degrees} -> rpy (rad): {body_data['rpy']}")

        # Parse inertial information
        inertial_elem = body_elem.find('inertial')
        if inertial_elem is not None:
            mass_str = inertial_elem.get('mass')
            if mass_str:
                body_data['mass'] = float(mass_str)

            inertial_pos = inertial_elem.get('pos', '0 0 0')
            body_data['inertial_origin']['xyz'] = [float(v) for v in inertial_pos.split()]

            diaginertia_str = inertial_elem.get('diaginertia')
            if diaginertia_str:
                diag = [float(v) for v in diaginertia_str.split()]
                if len(diag) >= 3:
                    body_data['inertia']['ixx'] = diag[0]
                    body_data['inertia']['iyy'] = diag[1]
                    body_data['inertia']['izz'] = diag[2]

        # Parse geometry (geom elements)
        geom_elems = body_elem.findall('geom')
        if geom_elems and self.verbose:
            print(f"{'  ' * level}  Found {len(geom_elems)} geom element(s)")

        for idx, geom_elem in enumerate(geom_elems):
            mesh_name = geom_elem.get('mesh')
            
            # Determine geom type (visual or collision)
            geom_class = geom_elem.get('class', '')
            geom_group = geom_elem.get('group', '')
            contype = geom_elem.get('contype')
            conaffinity = geom_elem.get('conaffinity')

            is_collision_geom = (geom_class == 'collision' or 
                                geom_group in ['0', '3'])
            is_visual_geom = (geom_class == 'visual' or 
                             geom_group in ['1', '2'] or
                             (contype == '0') or (conaffinity == '0'))

            if not is_collision_geom and not is_visual_geom:
                is_visual_geom = True
                is_collision_geom = True

            if self.verbose:
                print(f"{'  ' * level}    Geom[{idx}] type: visual={is_visual_geom}, collision={is_collision_geom}")

            if mesh_name and mesh_name in meshes_data:
                mesh_info = meshes_data[mesh_name]
                mesh_path = mesh_info['path']
                asset_mesh_scale = mesh_info['scale']

                # Parse geom position
                geom_pos_str = geom_elem.get('pos', '0 0 0')
                geom_pos = [float(v) for v in geom_pos_str.split()]

                # Parse geom rotation (quat or euler)
                geom_quat_str = geom_elem.get('quat')
                geom_euler_str = geom_elem.get('euler')
                geom_rpy = [0.0, 0.0, 0.0]
                geom_quat = [1.0, 0.0, 0.0, 0.0]

                if geom_quat_str:
                    geom_quat = [float(v) for v in geom_quat_str.split()]
                    geom_rpy = self.conversion_utils.quat_to_rpy(geom_quat)
                elif geom_euler_str:
                    euler_degrees = [float(v) for v in geom_euler_str.split()]
                    geom_rpy = self.conversion_utils.euler_to_rpy(euler_degrees, eulerseq)
                    # Convert RPY back to quat for storage
                    geom_quat = self.conversion_utils.rpy_to_quat(geom_rpy)

                # Parse geom mesh scale
                geom_meshscale = [1.0, 1.0, 1.0]
                meshscale_str = geom_elem.get('meshscale')
                if meshscale_str:
                    scale_values = [float(v) for v in meshscale_str.split()]
                    if len(scale_values) == 3:
                        geom_meshscale = scale_values
                    elif len(scale_values) == 1:
                        geom_meshscale = [scale_values[0]] * 3

                # Combine scales
                combined_scale = [
                    asset_mesh_scale[0] * geom_meshscale[0],
                    asset_mesh_scale[1] * geom_meshscale[1],
                    asset_mesh_scale[2] * geom_meshscale[2]
                ]

                # Parse color
                geom_color = [1.0, 1.0, 1.0]
                rgba_str = geom_elem.get('rgba')
                if rgba_str:
                    rgba = [float(v) for v in rgba_str.split()]
                    geom_color = rgba[:3]

                # Add to visuals list if visual geom
                if is_visual_geom:
                    visual_data = {
                        'mesh': mesh_path,
                        'pos': geom_pos,
                        'quat': geom_quat,
                        'color': geom_color
                    }
                    body_data['visuals'].append(visual_data)

                    # Set first visual as main stl_file for backward compatibility
                    if body_data['stl_file'] is None:
                        body_data['stl_file'] = mesh_path
                        body_data['color'] = geom_color
                        body_data['mesh_scale'] = combined_scale
                        body_data['visual_origin'] = {
                            'xyz': geom_pos,
                            'rpy': geom_rpy
                        }

                # Set as collision mesh if collision geom
                if is_collision_geom and body_data['collision_mesh'] is None:
                    body_data['collision_mesh'] = mesh_path

        bodies_data.append(body_data)

        # Parse joints in this body
        joint_elems = body_elem.findall('joint')
        for joint_elem in joint_elems:
            joint_name = joint_elem.get('name', f'joint_{len(joints_data)}')
            joint_type = joint_elem.get('type', 'hinge')
            
            # Map MJCF joint types to URDF types
            type_mapping = {
                'hinge': 'revolute',
                'slide': 'prismatic',
                'ball': 'spherical',
                'free': 'floating'
            }
            urdf_type = type_mapping.get(joint_type, 'revolute')

            joint_data = {
                'name': joint_name,
                'type': urdf_type,
                'parent': parent_name if parent_name else 'base_link',
                'child': body_name,
                'origin_xyz': body_data['pos'],
                'origin_rpy': body_data['rpy'],
                'axis': [1.0, 0.0, 0.0],
                'limit': {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 
                         'velocity': 3.0, 'friction': 0.05},
                'dynamics': {'damping': 0.0, 'friction': 0.0}
            }

            # Parse axis
            axis_str = joint_elem.get('axis')
            if axis_str:
                joint_data['axis'] = [float(v) for v in axis_str.split()]

            # Parse range (limits)
            range_str = joint_elem.get('range')
            if range_str:
                range_vals = [float(v) for v in range_str.split()]
                if len(range_vals) >= 2:
                    joint_data['limit']['lower'] = range_vals[0]
                    joint_data['limit']['upper'] = range_vals[1]

            # Parse damping
            damping_str = joint_elem.get('damping')
            if damping_str:
                joint_data['dynamics']['damping'] = float(damping_str)

            # Parse friction
            frictionloss_str = joint_elem.get('frictionloss')
            if frictionloss_str:
                joint_data['dynamics']['friction'] = float(frictionloss_str)

            joints_data.append(joint_data)

        # Recursively parse child bodies
        for child_body_elem in body_elem.findall('body'):
            self._parse_body(child_body_elem, body_name, level + 1, bodies_data, 
                           joints_data, meshes_data, default_classes, eulerseq)

    def parse_primitive_geom(self, geom_elem, geom_type, level=0):
        """Parse MJCF primitive geom and convert to Assembler collider_data format.

        Args:
            geom_elem: MJCF geom element
            geom_type: Primitive type ('cylinder', 'box', 'sphere', 'capsule')
            level: Logging indentation level

        Returns:
            collider_data: Dictionary in Assembler collider_data format,
                          or None if size attribute is missing
        """
        collider_data = {
            'type': geom_type,
            'geometry': {},
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0]  # degrees
        }

        # Parse position
        pos_str = geom_elem.get('pos', '0 0 0')
        collider_data['position'] = [float(v) for v in pos_str.split()]

        # Parse rotation (quat or euler to RPY in degrees)
        quat_str = geom_elem.get('quat')
        euler_str = geom_elem.get('euler')

        if quat_str:
            quat = [float(v) for v in quat_str.split()]
            rpy_rad = self.conversion_utils.quat_to_rpy(quat)
            collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]
        elif euler_str:
            euler_deg = [float(v) for v in euler_str.split()]
            collider_data['rotation'] = euler_deg  # Already in degrees

        # Parse size (convert from MJCF format to Assembler format)
        size_str = geom_elem.get('size')
        if not size_str:
            if self.verbose:
                print(f"{'  ' * level}      Warning: No size attribute for primitive geom")
            return None

        size_values = [float(v) for v in size_str.split()]

        if geom_type == 'cylinder':
            # MJCF: [radius, half_length] → Assembler: {radius, length}
            if len(size_values) >= 2:
                collider_data['geometry']['radius'] = str(size_values[0])
                collider_data['geometry']['length'] = str(size_values[1] * 2.0)  # half → full
                if self.verbose:
                    print(f"{'  ' * level}      Cylinder: radius={size_values[0]}, length={size_values[1]*2}")

        elif geom_type == 'box':
            # MJCF: [half_x, half_y, half_z] → Assembler: {size_x, size_y, size_z}
            if len(size_values) >= 3:
                collider_data['geometry']['size_x'] = str(size_values[0] * 2.0)
                collider_data['geometry']['size_y'] = str(size_values[1] * 2.0)
                collider_data['geometry']['size_z'] = str(size_values[2] * 2.0)
                if self.verbose:
                    print(f"{'  ' * level}      Box: x={size_values[0]*2}, y={size_values[1]*2}, z={size_values[2]*2}")

        elif geom_type == 'sphere':
            # MJCF: [radius] → Assembler: {radius}
            if len(size_values) >= 1:
                collider_data['geometry']['radius'] = str(size_values[0])
                if self.verbose:
                    print(f"{'  ' * level}      Sphere: radius={size_values[0]}")

        elif geom_type == 'capsule':
            # MJCF: [radius, half_length] → Assembler: {radius, length}
            if len(size_values) >= 2:
                collider_data['geometry']['radius'] = str(size_values[0])
                collider_data['geometry']['length'] = str(size_values[1] * 2.0)
                if self.verbose:
                    print(f"{'  ' * level}      Capsule: radius={size_values[0]}, length={size_values[1]*2}")

        if self.verbose:
            print(f"{'  ' * level}      Position: {collider_data['position']}")
            print(f"{'  ' * level}      Rotation: {collider_data['rotation']} (degrees)")

        return collider_data
