"""
File Name: urdf_kitchen_StlSourcer.py
Description: A Python script for reconfiguring the center coordinates and axis directions of STL and dae files.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Created On  : Dec 25, 2025


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
"""

import sys
import signal
import vtk
import numpy as np

# Import trimesh for COLLADA (.dae) file support
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. COLLADA (.dae) support disabled.")
    print("Install with: pip install trimesh")

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QLabel, QGridLayout
)
from PySide6.QtCore import QTimer, Qt, QObject

# M4 Mac (Apple Silicon) compatibility
import platform
IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'

# Always use QVTKRenderWindowInteractor (QVTKOpenGLNativeWidget not available in this VTK build)
USE_NATIVE_WIDGET = False
QVTKRenderWindowInteractor = None

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    try:
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except ImportError:
        print("Error: Could not import QVTKRenderWindowInteractor.")
        print("Please ensure VTK is installed with Qt support:")
        print("  pip install --upgrade vtk")
        import sys
        sys.exit(1)

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

        step = 0.01  # 10mm
        if shift_pressed and ctrl_pressed:
            step = 0.0001  # 0.1mm
        elif shift_pressed:
            step = 0.001  # 1mm

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
        if not self.GetInteractor():
            return
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        if not renderer:
            return
        actors = renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextItem()
        while actor:
            if not actor.GetUserTransform():
                if actor.GetProperty().GetRepresentation() == vtk.VTK_SURFACE:
                    actor.GetProperty().SetRepresentationToWireframe()
                else:
                    actor.GetProperty().SetRepresentationToSurface()
            actor = actors.GetNextItem()
        # SKIP Render on M4 Mac - it blocks the UI thread
        # self.GetInteractor().GetRenderWindow().Render()


class GlobalKeyEventFilter(QObject):
    """Global event filter for WASD keyboard controls"""
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        """Route WASD key events to main window when not in input widgets"""
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QApplication

        # Only process KeyPress events
        if event.type() == QEvent.KeyPress:
            # Check if focus is on an input widget
            focus_widget = QApplication.focusWidget()
            if isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
                # Let input widget handle the event
                return False

            # Route WASD keys to main window
            key = event.key()
            if key in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E,
                      Qt.Key_R, Qt.Key_T, Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
                # Forward to main window's keyPressEvent
                self.main_window.keyPressEvent(event)
                return True  # Consume the event

        return False  # Let event propagate normally


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("URDF kitchen - StlDaeSourcer v0.1.0")
        self.setGeometry(100, 100, 700, 700)

        self.camera_rotation = [0, 0, 0]  # [yaw, pitch, roll]
        self.absolute_origin = [0, 0, 0]  # 大原点の設定
        self.initial_camera_position = [10, 0, 0]  # 初期カメラ位置
        self.initial_camera_focal_point = [0, 0, 0]  # 初期焦点
        self.initial_camera_view_up = [0, 0, 1]  # 初期の上方向

        self.num_points = 1  # ポイントの数を1に設定
        self.point_coords = [list(self.absolute_origin) for _ in range(self.num_points)]
        self.point_actors = [None] * self.num_points
        self.point_checkboxes = []
        self.point_inputs = []

        # Track loaded STL file path for save dialog defaults
        self.current_stl_path = None

        # メインウィジェットとレイアウトの設定
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.set_ui_style()

        # ファイル名表示用のラベル
        self.file_name_label = QLabel("File: No file loaded")
        main_layout.addWidget(self.file_name_label)

        # LOADボタンとEXPORTボタン
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load .stl or .dae file")
        self.load_button.setFocusPolicy(Qt.NoFocus)  # Don't steal focus from vtk_display
        self.load_button.clicked.connect(self.load_stl_file)
        self.load_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(self.load_button)

        self.export_stl_button = QPushButton("Save as reoriented 3D model")
        self.export_stl_button.setFocusPolicy(Qt.NoFocus)  # Don't steal focus from vtk_display
        self.export_stl_button.clicked.connect(self.export_stl_with_new_origin)
        self.export_stl_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(self.export_stl_button)

        main_layout.addLayout(button_layout)

        # STL表示画面 - Use QLabel instead of QVTKRenderWindowInteractor for M4 Mac compatibility
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
        self.vtk_display.setText("3D View\n\n(Load STL file to display)\n\nKeyboard controls are enabled")
        self.vtk_display.setScaledContents(False)
        self.vtk_display.setMouseTracking(True)  # Enable mouse tracking

        # Make vtk_display focusable so it can receive keyboard events
        self.vtk_display.setFocusPolicy(Qt.StrongFocus)

        # Mouse interaction state
        self.mouse_pressed = False
        self.last_mouse_pos = None

        # Install event filter for mouse events
        self.vtk_display.installEventFilter(self)

        main_layout.addWidget(self.vtk_display)

        # Create offscreen VTK render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(800, 600)

        # No more QVTKRenderWindowInteractor - using offscreen rendering

        # Volume表示
        self.volume_label = QLabel("Volume (m^3): 0.000000")
        main_layout.addWidget(self.volume_label)

        # Points UI
        self.setup_points_ui(main_layout)

        # DON'T setup VTK yet - delay it until after window is shown
        self.vtk_initialized = False

        self.model_bounds = None
        self.stl_actor = None
        self.current_rotation = 0

        self.stl_center = list(self.absolute_origin)  # STLモデルの中心を大原点に初期化

        # Wireframe mode state
        self.wireframe_mode = False
        self.wireframe_actor = None
        self.original_stl_color = None  # Store original color
        self._toggling_wireframe = False  # Lock for wireframe toggle

        # Rendering lock to prevent re-entry
        self._is_rendering = False
        self._render_counter = 0  # Counter to force Qt pixmap updates
        self._pending_render_timers = []  # Track pending QTimer instances

        # Mouse drag state management (robust handling)
        self.mouse_pressed = False
        self.middle_mouse_pressed = False
        self.last_mouse_pos = None

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_rotation)
        self.animation_frames = 0
        self.total_animation_frames = 12
        self.rotation_per_frame = 0
        self.target_rotation = 0
        self.is_animating = False  # Flag to block input during animation
        self.target_angle = 0  # Target rotation angle for precise stopping
        self.rotation_axis = None  # Axis for current rotation

        self.rotation_types = {'yaw': 0, 'pitch': 1, 'roll': 2}

        # Set focus to vtk_display immediately after UI construction
        QTimer.singleShot(0, lambda: self.vtk_display.setFocus(Qt.OtherFocusReason))

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure focus is on vtk_display when window is shown
        QTimer.singleShot(0, lambda: self.vtk_display.setFocus(Qt.OtherFocusReason))
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
            self.axes_widget = None
            self.add_axes()
            for i in range(self.num_points):
                self.show_point(i)
            self.add_instruction_text()
            QTimer.singleShot(100, self._vtk_setup_final)
        except Exception as e:
            print(f"ERROR in VTK step 3: {e}")
            import traceback
            traceback.print_exc()

    def _vtk_setup_final(self):
        try:
            QTimer.singleShot(200, self.render_to_image)
            QTimer.singleShot(300, lambda: self.vtk_display.setFocus())
        except Exception as e:
            print(f"ERROR in VTK final step: {e}")
            import traceback
            traceback.print_exc()

    def _delayed_render(self):
        try:
            self.render_to_image()
        except Exception as e:
            print(f"_delayed_render: Render failed: {e}")
            import traceback
            traceback.print_exc()

    def render_to_image(self):
        """Render VTK scene offscreen and display as image in QLabel"""
        from PySide6.QtGui import QImage, QPixmap

        if self._is_rendering:
            return

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

            import numpy as np
            arr = np.frombuffer(vtk_array, dtype=np.uint8)
            arr = arr.reshape(height, width, components)
            arr = np.flip(arr, axis=0)
            arr = np.ascontiguousarray(arr)

            if components == 3:
                qimage = QImage(arr.data, width, height, width * 3, QImage.Format_RGB888)
            else:
                qimage = QImage(arr.data, width, height, width * 4, QImage.Format_RGBA8888)

            # Simplified rendering without aggressive cache-busting to prevent flickering
            self._render_counter += 1

            # Create pixmap from image
            pixmap = QPixmap.fromImage(qimage.copy())

            # Simply set the pixmap - no clearing, no visibility toggling
            self.vtk_display.setPixmap(pixmap)

            # Single update call is sufficient
            self.vtk_display.update()

            # Restore focus to vtk_display if focus is not on input widget
            from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit, QPlainTextEdit
            focus_widget = QApplication.focusWidget()
            if not isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
                self.vtk_display.setFocus(Qt.OtherFocusReason)

        finally:
            self._is_rendering = False

    def _cancel_mouse_drag(self):
        """Cancel mouse drag state - call on abnormal termination (Leave, WindowDeactivate, etc.)"""
        self.mouse_pressed = False
        self.middle_mouse_pressed = False
        self.last_mouse_pos = None
        try:
            self.vtk_display.releaseMouse()
        except:
            pass

    def set_ui_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #5a5a5a, stop:1 #3a3a3a);
                color: #ffffff;
                border: 1px solid #707070;
                border-radius: 5px;
                padding: 3px 8px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #6a6a6a, stop:1 #4a4a4a);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #3a3a3a, stop:1 #5a5a5a);
                padding-top: 6px;
                padding-bottom: 4px;
            }
            QLineEdit {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #5a5a5a;
                border-radius: 3px;
                padding: 2px;
            }
            QCheckBox {
                color: #ffffff;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #5a5a5a;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #5a5a5a;
                background-color: #4a90e2;
            }
        """)

    def setup_points_ui(self, layout):
        points_layout = QGridLayout()

        for i in range(self.num_points):
            # Replace checkbox with "Center Position" label
            label = QLabel("Center Position")
            points_layout.addWidget(label, i, 0)

            # Create hidden checkbox for backward compatibility (always checked)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.setVisible(False)  # Hide the checkbox
            self.point_checkboxes.append(checkbox)

            inputs = []
            for j, axis in enumerate(['X', 'Y', 'Z']):
                input_field = QLineEdit(str(self.point_coords[i][j]))
                # Only focus when explicitly clicked, don't steal keyboard focus
                input_field.setFocusPolicy(Qt.ClickFocus)
                # Return focus to vtk_display when editing is done
                input_field.editingFinished.connect(lambda: self.vtk_display.setFocus())
                inputs.append(input_field)
                points_layout.addWidget(QLabel(f"{axis}:"), i, j*2+1)
                points_layout.addWidget(input_field, i, j*2+2)
            self.point_inputs.append(inputs)

        layout.addLayout(points_layout)

        # SET と RESET ボタン
        button_layout = QHBoxLayout()
        set_button = QPushButton("Set Marker")
        set_button.setFocusPolicy(Qt.NoFocus)  # Don't steal focus from vtk_display
        set_button.clicked.connect(self.handle_set_reset)
        set_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(set_button)

        reset_button = QPushButton("Reset Marker")
        reset_button.setFocusPolicy(Qt.NoFocus)  # Don't steal focus from vtk_display
        reset_button.clicked.connect(self.handle_set_reset)
        reset_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(reset_button)

        set_front_button = QPushButton("Set Front as X")
        set_front_button.setFocusPolicy(Qt.NoFocus)  # Don't steal focus from vtk_display
        set_front_button.clicked.connect(self.handle_set_reset)
        set_front_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(set_front_button)

        layout.addLayout(button_layout)

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
            print(
                f"Invalid input for Point {index+1}. Please enter valid numbers for coordinates.")

    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.render_window.AddRenderer(self.renderer)
        self.render_window_interactor = None

    def setup_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(self.absolute_origin[0] + self.initial_camera_position[0],
                           self.absolute_origin[1] +
                           self.initial_camera_position[1],
                           self.absolute_origin[2] + self.initial_camera_position[2])
        camera.SetFocalPoint(*self.absolute_origin)
        camera.SetViewUp(*self.initial_camera_view_up)
        camera.SetParallelScale(5)
        camera.ParallelProjectionOn()
        self.renderer.ResetCameraClippingRange()

    def reset_point_to_origin(self, index):
        self.point_coords[index] = list(self.absolute_origin)
        self.update_point_display(index)
        if self.point_checkboxes[index].isChecked():
            self.show_point(index)
        print(f"Point {index+1} reset to origin {self.absolute_origin}")

    def reset_points(self):
        for i in range(self.num_points):
            if self.point_checkboxes[i].isChecked():
                self.point_coords[i] = list(self.absolute_origin)
                self.update_point_display(i)
                print(f"Point {i+1} reset to origin {self.absolute_origin}")

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(10, 0, 0)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        self.camera_rotation = [0, 0, 0]
        self.current_rotation = 0

        if hasattr(self, 'axes_widget') and self.axes_widget is not None:
            self.renderer.RemoveViewProp(self.axes_widget.GetOrientationMarker())
        self.axes_widget = None

        self.renderer.ResetCameraClippingRange()

        if hasattr(self, 'point_actors') and self.point_actors is not None:
            self.update_all_points_size()

        print("Camera reset to default position")
        print("View direction: Looking from +X towards origin")
        print("Up direction: +Z")
        print("Right direction: +Y")

    def update_point_position(self, index, x, y):
        renderer = self.renderer
        camera = renderer.GetActiveCamera()

        # スクリーン座標からワールド座標への変換
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        coordinate.SetValue(x, y, 0)
        world_pos = coordinate.GetComputedWorldValue(renderer)

        # カメラの向きに基づいて、z座標を現在のポイントのz座標に保つ
        camera_pos = np.array(camera.GetPosition())
        focal_point = np.array(camera.GetFocalPoint())
        view_direction = focal_point - camera_pos
        view_direction /= np.linalg.norm(view_direction)

        current_z = self.point_coords[index][2]
        t = (current_z - camera_pos[2]) / view_direction[2]
        new_pos = camera_pos + t * view_direction

        self.point_coords[index] = [new_pos[0], new_pos[1], current_z]
        self.update_point_display(index)

        print(
            f"Point {index+1} moved to: ({new_pos[0]:.4f}, {new_pos[1]:.4f}, {current_z:.4f})")

    def update_point_display(self, index):
        if self.point_actors[index]:
            self.point_actors[index].SetPosition(self.point_coords[index])

        for i, coord in enumerate(self.point_coords[index]):
            self.point_inputs[index][i].setText(f"{coord:.4f}")

    def update_properties(self):
        # 優先順位: Mass > Volume > Inertia > Density
        priority_order = ['mass', 'volume', 'inertia', 'density']
        values = {}

        # チェックされているプロパティの値を取得
        for prop in priority_order:
            checkbox = getattr(self, f"{prop}_checkbox")
            input_field = getattr(self, f"{prop}_input")
            if checkbox.isChecked():
                try:
                    values[prop] = float(input_field.text())
                except ValueError:
                    print(f"Invalid input for {prop}")
                    return

        # 値の計算
        if 'mass' in values and 'volume' in values:
            values['density'] = values['mass'] / values['volume']
        elif 'mass' in values and 'density' in values:
            values['volume'] = values['mass'] / values['density']
        elif 'volume' in values and 'density' in values:
            values['mass'] = values['volume'] * values['density']

        # Inertiaの計算 (簡略化した例: 立方体と仮定)
        if 'mass' in values and 'volume' in values:
            side_length = np.cbrt(values['volume'])
            values['inertia'] = (1/6) * values['mass'] * side_length**2

        # 結果を入力フィールドに反映
        for prop in priority_order:
            input_field = getattr(self, f"{prop}_input")
            if prop in values:
                input_field.setText(f"{values[prop]:.12f}")

    def update_all_points_size(self, obj=None, event=None):
        for index, actor in enumerate(self.point_actors):
            if actor:
                self.renderer.RemoveActor(actor)
                self.point_actors[index] = vtk.vtkAssembly()
                self.create_point_coordinate(self.point_actors[index], [0, 0, 0])
                self.point_actors[index].SetPosition(self.point_coords[index])
                self.renderer.AddActor(self.point_actors[index])

    def update_all_points(self):
        pass

    def create_point_coordinate(self, assembly, coords):
        origin = coords
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
                assembly.AddPart(actor)

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
            actor.GetProperty().SetColor(1, 0, 1)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(6)
            actor.GetProperty().SetOpacity(0.7)

            transform = vtk.vtkTransform()
            actor.SetUserTransform(transform)
            assembly.AddPart(actor)

    def calculate_sphere_radius(self):
        camera = self.renderer.GetActiveCamera()
        parallel_scale = camera.GetParallelScale()
        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / (viewport[3] - viewport[1])
        radius = parallel_scale * 0.05
        if aspect_ratio > 1:
            radius /= aspect_ratio
        return radius

    def calculate_properties(self):
        try:
            volume = float(self.volume_label.text().split(': ')[1])
            density = 1.0
            mass = volume * density
            side_length = np.cbrt(volume)
            inertia = (1/6) * mass * side_length**2

            print(f"Volume: {volume:.6f} m^3")
            print(f"Density: {density:.6f} kg/m^3")
            print(f"Mass: {mass:.6f} kg")
            print(f"Inertia: {inertia:.6f} kg·m^2")

        except ValueError:
            print("Error calculating properties. Please ensure the volume is a valid number.")

    def apply_camera_rotation(self, camera):
        position = list(camera.GetPosition())
        focal_point = self.absolute_origin

        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Translate(*focal_point)
        transform.RotateZ(self.camera_rotation[2])
        transform.RotateX(self.camera_rotation[0])
        transform.RotateY(self.camera_rotation[1])
        transform.Translate(*[-x for x in focal_point])

        new_position = transform.TransformPoint(position)
        camera.SetPosition(new_position)

        up = [0, 0, 1]
        new_up = transform.TransformVector(up)
        camera.SetViewUp(new_up)

    def add_axes(self):
        if not hasattr(self, 'axis_actors'):
            self.axis_actors = []

        for actor in self.renderer.GetActors():
            if actor != self.stl_actor and actor not in self.point_actors:
                self.renderer.RemoveActor(actor)

        self.axis_actors = []
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
                self.axis_actors.append(actor)

    def add_instruction_text(self):
        if not hasattr(self, 'text_actors'):
            self.text_actors = []

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
        text_actor_top.GetTextProperty().SetColor(0.0, 0.8, 0.8)
        text_actor_top.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor_top.SetPosition(0.03, 0.97)
        text_actor_top.GetTextProperty().SetJustificationToLeft()
        text_actor_top.GetTextProperty().SetVerticalJustificationToTop()
        self.renderer.AddActor(text_actor_top)
        self.text_actors.append(text_actor_top)

        text_actor_bottom = vtk.vtkTextActor()
        text_actor_bottom.SetInput(
            "[Arrows] : Move Marker 10mm\n"
            "  +[Shift] : Move Marker 1mm\n"
            "   +[Ctrl] : Move Marker 0.1mm\n"
        )
        text_actor_bottom.GetTextProperty().SetFontSize(14)
        text_actor_bottom.GetTextProperty().SetColor(0.0, 0.8, 0.8)
        text_actor_bottom.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor_bottom.SetPosition(0.03, 0.03)
        text_actor_bottom.GetTextProperty().SetJustificationToLeft()
        text_actor_bottom.GetTextProperty().SetVerticalJustificationToBottom()
        self.renderer.AddActor(text_actor_bottom)
        self.text_actors.append(text_actor_bottom)

    def load_stl_file(self):
        print("Opening file dialog...")
        # Support both STL and DAE (COLLADA) files
        if TRIMESH_AVAILABLE:
            file_filter = "3D Model Files (*.stl *.dae);;STL Files (*.stl);;COLLADA Files (*.dae);;All Files (*)"
        else:
            file_filter = "STL Files (*.stl);;All Files (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model File", "", file_filter)
        if file_path:
            print(f"Loading STL file: {file_path}")
            self.file_name_label.setText(f"File: {file_path}")
            # Save the loaded STL file path for save dialog defaults
            self.current_stl_path = file_path

            try:
                self.show_stl(file_path)
                self.reset_camera()

                # Reset all Center Position points to origin
                for i in range(self.num_points):
                    self.point_coords[i] = [0.0, 0.0, 0.0]
                    # Update input fields to show 0.0
                    for j in range(3):
                        self.point_inputs[i][j].setText("0.0")
                    # Update point display
                    if self.point_actors[i]:
                        self.update_point_display(i)

                # Render to show updated points
                self.render_to_image()

                # Set focus to vtk_display after loading STL
                self.vtk_display.setFocus()
            except Exception as e:
                print(f"Error loading STL: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("File dialog cancelled")

    def show_stl(self, file_path):
        if hasattr(self, 'stl_actor') and self.stl_actor:
            self.renderer.RemoveActor(self.stl_actor)

        # Check file extension to determine loader
        import os
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.dae' and TRIMESH_AVAILABLE:
            # Load COLLADA file using trimesh
            mesh = trimesh.load(file_path, force='mesh')

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

            # Create clean filter
            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(poly_data)
            clean.SetTolerance(1e-5)
            clean.ConvertPolysToLinesOff()
            clean.ConvertStripsToPolysOff()
            clean.PointMergingOn()
            clean.Update()

        else:
            # Load STL file using VTK
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

        remover = vtk.vtkDecimatePro()
        remover.SetInputConnection(clean.GetOutputPort())
        remover.SetTargetReduction(0.0)
        remover.PreserveTopologyOn()
        remover.Update()

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(remover.GetOutputPort())
        triangulate.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(triangulate.GetOutputPort())
        self.stl_actor = vtk.vtkActor()
        self.stl_actor.SetMapper(mapper)

        self.model_bounds = triangulate.GetOutput().GetBounds()
        self.renderer.AddActor(self.stl_actor)

        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputConnection(triangulate.GetOutputPort())
        volume = mass_properties.GetVolume()
        self.volume_label.setText(f"Volume (m^3): {volume:.6f}")

        self.fit_camera_to_model()

        print(f"Model loaded: {file_path}")
        print(f"Bounds: [{self.model_bounds[0]:.4f}, {self.model_bounds[1]:.4f}], [{self.model_bounds[2]:.4f}, {self.model_bounds[3]:.4f}], [{self.model_bounds[4]:.4f}, {self.model_bounds[5]:.4f}]")

        QTimer.singleShot(500, self._delayed_render)

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

    def show_point(self, index):
        if self.point_actors[index] is None:
            self.point_actors[index] = vtk.vtkAssembly()
            self.create_point_coordinate(
                self.point_actors[index], self.point_coords[index])
            self.renderer.AddActor(self.point_actors[index])
        self.point_actors[index].VisibilityOn()
        self.update_point_display(index)

    def rotate_camera(self, angle, rotation_type):
        # Don't start new animation if one is already running
        if self.is_animating:
            return

        self.target_rotation = (self.current_rotation + angle) % 360
        self.rotation_per_frame = angle / self.total_animation_frames
        self.animation_frames = 0
        self.current_rotation_type = self.rotation_types[rotation_type]
        self.target_angle = angle  # Store target angle for precise completion
        self.is_animating = True  # Block further input
        self.animation_timer.start(1000 // 60)
        self.camera_rotation[self.rotation_types[rotation_type]] += angle
        self.camera_rotation[self.rotation_types[rotation_type]] %= 360

    def animate_rotation(self):
        self.animation_frames += 1

        camera = self.renderer.GetActiveCamera()
        position = list(camera.GetPosition())
        focal_point = self.absolute_origin
        view_up = list(camera.GetViewUp())

        forward = [focal_point[i] - position[i] for i in range(3)]
        right = [
            view_up[1] * forward[2] - view_up[2] * forward[1],
            view_up[2] * forward[0] - view_up[0] * forward[2],
            view_up[0] * forward[1] - view_up[1] * forward[0]
        ]

        if self.current_rotation_type == self.rotation_types['yaw']:
            axis = view_up
        elif self.current_rotation_type == self.rotation_types['pitch']:
            axis = right
        else:  # roll
            axis = forward

        # On the last frame, apply the exact remaining angle to ensure precise 90-degree rotation
        if self.animation_frames >= self.total_animation_frames:
            # Calculate exact remaining angle
            remaining_angle = self.target_angle - (self.rotation_per_frame * (self.animation_frames - 1))
            rotation_angle = remaining_angle
        else:
            rotation_angle = self.rotation_per_frame

        rotation_matrix = vtk.vtkTransform()
        rotation_matrix.Translate(*focal_point)
        rotation_matrix.RotateWXYZ(rotation_angle, axis)
        rotation_matrix.Translate(*[-x for x in focal_point])

        new_position = rotation_matrix.TransformPoint(position)
        new_up = rotation_matrix.TransformVector(view_up)

        camera.SetPosition(new_position)
        camera.SetViewUp(new_up)

        # Render using offscreen rendering
        self.render_to_image()

        # Stop animation after last frame
        if self.animation_frames >= self.total_animation_frames:
            self.animation_timer.stop()
            self.current_rotation = self.target_rotation
            self.is_animating = False  # Allow new input

    def export_urdf(self):
        print("URDF export functionality will be implemented here")

    def get_axis_length(self):
        if self.model_bounds:
            size = max([
                self.model_bounds[1] - self.model_bounds[0],
                self.model_bounds[3] - self.model_bounds[2],
                self.model_bounds[5] - self.model_bounds[4]
            ])
            return size * 0.5
        else:
            return 5

    def hide_point(self, index):
        if self.point_actors[index]:
            self.point_actors[index].VisibilityOff()

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
            print(
                f"Invalid input for Point {index+1}. Please enter valid numbers for coordinates.")

    def move_point(self, index, dx, dy, dz):
        new_position = [
            self.point_coords[index][0] + dx,
            self.point_coords[index][1] + dy,
            self.point_coords[index][2] + dz
        ]
        self.point_coords[index] = new_position
        self.update_point_display(index)
        print(
            f"Point {index+1} moved to: ({new_position[0]:.4f}, {new_position[1]:.4f}, {new_position[2]:.4f})")

    def move_point_screen(self, index, direction, step):
        move_vector = direction * step
        new_position = [
            self.point_coords[index][0] + move_vector[0],
            self.point_coords[index][1] + move_vector[1],
            self.point_coords[index][2] + move_vector[2]
        ]
        self.point_coords[index] = new_position
        self.update_point_display(index)
        print(
            f"Point {index+1} moved to: ({new_position[0]:.4f}, {new_position[1]:.4f}, {new_position[2]:.4f})")

    def fit_camera_to_model(self):
        if not self.model_bounds:
            return

        camera = self.renderer.GetActiveCamera()
        center = [(self.model_bounds[i] + self.model_bounds[i+1]) / 2 for i in range(0, 6, 2)]

        size = max([
            self.model_bounds[1] - self.model_bounds[0],
            self.model_bounds[3] - self.model_bounds[2],
            self.model_bounds[5] - self.model_bounds[4]
        ])
        size *= 1.4

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

    def keyPressEvent(self, event):
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
            self.render_to_image()
        elif key == Qt.Key_T:
            # Prevent rapid toggle - ignore if already toggling
            if self._toggling_wireframe:
                return

            self._toggling_wireframe = True

            # Cancel any pending render timers from previous toggle
            for timer in self._pending_render_timers:
                if timer.isActive():
                    timer.stop()
            self._pending_render_timers.clear()

            self.toggle_wireframe()

            # Single render is sufficient - excessive renders cause flickering
            self.render_to_image()

            # Release lock quickly to allow next toggle
            QTimer.singleShot(100, lambda: setattr(self, '_toggling_wireframe', False))

        elif key in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            step = 0.01
            if shift_pressed and ctrl_pressed:
                step = 0.0001
            elif shift_pressed:
                step = 0.001

            horizontal_axis, vertical_axis, screen_right, screen_up = self.get_screen_axes()

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

    def toggle_wireframe(self):
        if not self.stl_actor:
            return

        property = self.stl_actor.GetProperty()

        if not self.wireframe_mode:
            if self.original_stl_color is None:
                self.original_stl_color = property.GetColor()

            property.SetRepresentationToWireframe()
            property.SetColor(1.0, 1.0, 1.0)  # White wireframe
            property.SetLineWidth(1.0)  # Thin 1-pixel lines
            property.SetOpacity(1.0)
            # Keep background consistent - don't change it
            # self.renderer.SetBackground(0.0, 0.0, 0.0)  # Black background
            self.wireframe_mode = True
        else:
            property.SetRepresentationToSurface()
            if self.original_stl_color is not None:
                property.SetColor(self.original_stl_color)
            else:
                property.SetColor(0.8, 0.8, 0.8)
            property.SetOpacity(1.0)
            # Keep background consistent - don't change it
            # self.renderer.SetBackground(0.2, 0.2, 0.2)  # Gray background
            self.wireframe_mode = False

        self.stl_actor.Modified()
        self.renderer.Modified()
        self.render_window.Modified()

    def eventFilter(self, obj, event):
        """Handle mouse events on vtk_display with robust state management"""
        from PySide6.QtCore import QEvent, Qt as QtCore
        from PySide6.QtGui import QMouseEvent
        try:
            from PySide6.QtGui import QNativeGestureEvent
        except ImportError:
            QNativeGestureEvent = None

        if obj == self.vtk_display:
            # Fallback cancellation - release drag on abnormal events
            # NOTE: Leave events occur frequently on Mac even with grabMouse(), so we ignore them
            # Only cancel on WindowDeactivate (user switches to another app)
            if event.type() == QEvent.WindowDeactivate:
                if self.mouse_pressed or self.middle_mouse_pressed:
                    self._cancel_mouse_drag()
                    return True

            # Double click cancels drag (prevents stuck state)
            if event.type() == QEvent.MouseButtonDblClick:
                if self.mouse_pressed or self.middle_mouse_pressed:
                    self._cancel_mouse_drag()
                return True

            if event.type() == QEvent.MouseButtonPress:
                if isinstance(event, QMouseEvent):
                    # Handle left button for drag rotation
                    if event.button() == QtCore.LeftButton:
                        # Set focus to vtk_display to enable keyboard controls
                        self.vtk_display.setFocus()
                        self.mouse_pressed = True
                        self.last_mouse_pos = event.pos()
                        # CRITICAL: Grab mouse to receive events even outside widget
                        self.vtk_display.grabMouse()
                        return True
                    # Handle middle button (wheel button) for pan
                    elif event.button() == QtCore.MiddleButton:
                        self.vtk_display.setFocus()
                        self.middle_mouse_pressed = True
                        self.last_mouse_pos = event.pos()
                        self.vtk_display.grabMouse()
                        return True

            elif event.type() == QEvent.MouseButtonRelease:
                if isinstance(event, QMouseEvent):
                    # Release on left button stops rotation
                    if event.button() == QtCore.LeftButton:
                        self.mouse_pressed = False
                        self.last_mouse_pos = None
                        # CRITICAL: Release mouse grab
                        self.vtk_display.releaseMouse()
                        return True
                    # Release on middle button stops pan
                    elif event.button() == QtCore.MiddleButton:
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
                # Handle trackpad/mouse wheel
                delta_y = event.angleDelta().y()
                delta_x = event.angleDelta().x()

                # Vertical scroll = Zoom
                if abs(delta_y) > abs(delta_x):
                    self.zoom_camera(delta_y)
                # Horizontal scroll = Rotate around Z axis
                elif abs(delta_x) > 0:
                    # Small rotation based on horizontal scroll
                    self.rotate_camera_mouse(delta_x * 0.2, 0)

                return True

            elif event.type() == QEvent.NativeGesture:
                # Handle Mac trackpad gestures (pinch zoom, rotate)
                if QNativeGestureEvent and isinstance(event, QNativeGestureEvent):
                    try:
                        gesture_type = event.gestureType()

                        # Pinch to zoom
                        if hasattr(QtCore, 'ZoomNativeGesture') and gesture_type == QtCore.ZoomNativeGesture:
                            zoom_value = event.value()
                            # Scale zoom value appropriately
                            self.zoom_camera(zoom_value * 500)
                            return True

                        # Rotate gesture
                        elif hasattr(QtCore, 'RotateNativeGesture') and gesture_type == QtCore.RotateNativeGesture:
                            rotation_angle = event.value()
                            # Apply rotation around view direction
                            self.rotate_camera_mouse(rotation_angle * 2, 0)
                            return True
                    except:
                        pass

            elif event.type() == QEvent.KeyPress:
                # Forward key press events to main window's keyPressEvent
                self.keyPressEvent(event)
                return True

        return super().eventFilter(obj, event)

    def rotate_camera_mouse(self, dx, dy):
        """Rotate camera based on mouse drag"""
        camera = self.renderer.GetActiveCamera()

        # Horizontal drag rotates around up vector (yaw)
        # Vertical drag rotates around right vector (pitch)
        sensitivity = 0.5

        # Yaw rotation (around Z axis)
        camera.Azimuth(-dx * sensitivity)

        # Pitch rotation
        camera.Elevation(dy * sensitivity)

        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.render_to_image()

    def pan_camera(self, dx, dy):
        """Pan camera based on mouse drag with Shift"""
        camera = self.renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()

        # Get camera coordinate system
        view_up = np.array(camera.GetViewUp())
        view_direction = np.array(focal_point) - np.array(position)
        distance = np.linalg.norm(view_direction)
        view_direction = view_direction / distance

        # Right vector
        right = np.cross(view_direction, view_up)
        right = right / np.linalg.norm(right)

        # Recalculate up vector
        up = np.cross(right, view_direction)

        # Pan speed based on distance to focal point
        pan_speed = distance * 0.001

        # Calculate pan offset
        offset = -right * dx * pan_speed + up * dy * pan_speed

        # Move camera and focal point
        new_position = np.array(position) + offset
        new_focal = np.array(focal_point) + offset

        camera.SetPosition(new_position)
        camera.SetFocalPoint(new_focal)

        self.renderer.ResetCameraClippingRange()
        self.render_to_image()

    def zoom_camera(self, delta):
        """Zoom camera based on mouse wheel"""
        camera = self.renderer.GetActiveCamera()

        # For parallel projection, use SetParallelScale instead of Dolly
        current_scale = camera.GetParallelScale()

        if delta > 0:
            # Zoom in (make objects larger by decreasing scale)
            new_scale = current_scale * 0.9
        else:
            # Zoom out (make objects smaller by increasing scale)
            new_scale = current_scale * 1.1

        camera.SetParallelScale(new_scale)
        self.renderer.ResetCameraClippingRange()
        self.render_to_image()

    def closeEvent(self, event):
        """Override QMainWindow's closeEvent to properly cleanup VTK resources"""
        print("Window is closing...")
        try:
            if hasattr(self, 'render_window') and self.render_window:
                self.render_window.Finalize()
            # No vtk_widget to close in offscreen mode
        except Exception as e:
            print(f"Error during cleanup: {e}")
        event.accept()

    def get_screen_axes(self):
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

    def export_stl_with_new_origin(self):
        if not self.stl_actor or not any(self.point_actors):
            print("3D model or points are not set.")
            return

        # Use the loaded file path as default (same directory and filename)
        default_path = self.current_stl_path if self.current_stl_path else ""

        # Support both STL and DAE (COLLADA) files for export
        if TRIMESH_AVAILABLE:
            file_filter = "STL Files (*.stl);;COLLADA Files (*.dae);;All Files (*)"
        else:
            file_filter = "STL Files (*.stl);;All Files (*)"

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save 3D Model File", default_path, file_filter)
        if not file_path:
            return

        try:
            import os
            poly_data = self.stl_actor.GetMapper().GetInput()
            origin_index = next(i for i, actor in enumerate(self.point_actors) if actor and actor.GetVisibility())
            origin_point = self.point_coords[origin_index]

            transform = vtk.vtkTransform()
            transform.Translate(-origin_point[0], -origin_point[1], -origin_point[2])

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(poly_data)
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            # Determine file format from extension
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.dae' and TRIMESH_AVAILABLE:
                # Export as COLLADA using trimesh
                transformed_poly = transform_filter.GetOutput()

                # Convert VTK PolyData to numpy arrays
                num_points = transformed_poly.GetNumberOfPoints()
                num_cells = transformed_poly.GetNumberOfCells()

                vertices = np.zeros((num_points, 3))
                for i in range(num_points):
                    vertices[i] = transformed_poly.GetPoint(i)

                faces = []
                for i in range(num_cells):
                    cell = transformed_poly.GetCell(i)
                    if cell.GetNumberOfPoints() == 3:
                        faces.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)])

                faces = np.array(faces)

                # Create trimesh object and export
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.export(file_path)

                print(f"COLLADA file has been saved: {file_path}")

            else:
                # Export as STL using VTK
                # Add .stl extension if not present
                if not file_ext:
                    file_path += '.stl'

                stl_writer = vtk.vtkSTLWriter()
                stl_writer.SetFileName(file_path)
                stl_writer.SetInputData(transform_filter.GetOutput())
                stl_writer.Write()

                print(f"STL file has been saved: {file_path}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def handle_set_front_as_x(self):
        button_text = self.sender().text()
        if button_text == "Set Front as X":
            self.transform_stl_to_camera_view()

    def transform_stl_to_front_view(self):
        if not self.stl_actor:
            print("No STL model is loaded.")
            return

        current_transform = self.stl_actor.GetMatrix()
        current_x = [current_transform.GetElement(0, 0),
                    current_transform.GetElement(1, 0),
                    current_transform.GetElement(2, 0)]

        needs_flip = current_x[0] < 0
        transform_matrix = np.eye(4)

        if needs_flip:
            transform_matrix[0, 0] = -1

        center = np.array(self.stl_actor.GetCenter())

        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Translate(-center[0], -center[1], -center[2])

        if needs_flip:
            transform.RotateY(180)

        transform.Translate(center[0], center[1], center[2])

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(self.stl_actor.GetMapper().GetInput())
        transform_filter.Update()

        new_mapper = vtk.vtkPolyDataMapper()
        new_mapper.SetInputData(transform_filter.GetOutput())
        self.stl_actor.SetMapper(new_mapper)

    def handle_set_reset(self):
        sender = self.sender()
        button_text = sender.text()

        if button_text == "Set Marker":
            for i, checkbox in enumerate(self.point_checkboxes):
                if checkbox.isChecked():
                    try:
                        new_coords = [float(self.point_inputs[i][j].text()) for j in range(3)]
                        self.point_coords[i] = new_coords
                        # Always show the point when Set Marker is pressed
                        self.show_point(i)
                        print(f"Point {i+1} set to: {new_coords}")
                    except ValueError:
                        print(f"Invalid input for Point {i+1}. Please enter valid numbers.")
            # Update display after setting markers
            self.render_to_image()
        elif button_text == "Set Front as X":
            self.handle_set_front_as_x()
            # Update display after transforming STL to camera view
            self.render_to_image()
        else:
            for i, checkbox in enumerate(self.point_checkboxes):
                if checkbox.isChecked():
                    self.reset_point_to_origin(i)
            self.update_all_points_size()
            # Update display after resetting markers
            self.render_to_image()

    def add_axes_widget(self):
        return None

    def update_axes_widget(self, new_x, new_y, new_z):
        if hasattr(self, 'axes_widget') and self.axes_widget is not None:
            self.renderer.RemoveViewProp(self.axes_widget.GetOrientationMarker())
        self.axes_widget = None

    def transform_stl_to_camera_view(self):
        if not self.stl_actor:
            print("No STL model is loaded.")
            return

        camera = self.renderer.GetActiveCamera()
        camera_pos = np.array(camera.GetPosition())
        focal_point = np.array(camera.GetFocalPoint())
        camera_up = np.array(camera.GetViewUp())

        z_axis = camera_up
        z_axis = z_axis / np.linalg.norm(z_axis)

        view_dir = camera_pos - focal_point
        x_axis = view_dir / np.linalg.norm(view_dir)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        poly_data = self.stl_actor.GetMapper().GetInput()
        points = poly_data.GetPoints()
        n_points = points.GetNumberOfPoints()

        new_points = vtk.vtkPoints()
        for i in range(n_points):
            point = np.array(points.GetPoint(i))
            new_x = np.dot(point, x_axis)
            new_y = np.dot(point, y_axis)
            new_z = np.dot(point, z_axis)
            new_points.InsertNextPoint(new_x, new_y, new_z)

        new_poly_data = vtk.vtkPolyData()
        new_poly_data.SetPoints(new_points)
        new_poly_data.SetPolys(poly_data.GetPolys())

        new_mapper = vtk.vtkPolyDataMapper()
        new_mapper.SetInputData(new_poly_data)
        self.stl_actor.SetMapper(new_mapper)

        self.update_axes_widget(x_axis, y_axis, z_axis)
        self.reset_camera()

def signal_handler(sig, frame):
    QApplication.instance().quit()


if __name__ == "__main__":
    import os

    # M4 Mac (Apple Silicon) detection
    if IS_APPLE_SILICON:
        # Set environment variable for better compatibility
        os.environ['QT_MAC_WANTS_LAYER'] = '1'

    # Set Qt attributes BEFORE creating QApplication
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    # Create QApplication
    app = QApplication(sys.argv)

    # Ctrl+Cのシグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)

    try:
        window = MainWindow()

        # Install global event filter for WASD keyboard controls
        global_filter = GlobalKeyEventFilter(window)
        app.installEventFilter(global_filter)

        # Move window to center of screen
        from PySide6.QtGui import QScreen
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            window_geometry = window.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            window.move(window_geometry.topLeft())

        # Show window
        window.show()
        window.raise_()
        window.activateWindow()

    except Exception as e:
        print(f"Error creating window: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # タイマーを設定してシグナルを処理できるようにする
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    sys.exit(app.exec())
