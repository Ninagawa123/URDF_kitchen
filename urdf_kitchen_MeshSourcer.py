"""
File Name: urdf_kitchen_MeshSourcer.py
Description: A Python script for reconfiguring the center coordinates and axis directions of STL and dae files.

Author      : Ninagawa123
Created On  : Nov 24, 2024
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
pip install networkx
"""

import sys
import os
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
    QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QLabel, QGridLayout,
    QComboBox, QGroupBox, QScrollArea, QButtonGroup, QRadioButton, QSizePolicy
)
from PySide6.QtCore import QTimer, Qt, QObject

# Import URDF Kitchen utilities
from urdf_kitchen_utils import (
    OffscreenRenderer, CameraController, AnimatedCameraRotation,
    AdaptiveMarkerSize, create_crosshair_marker, MouseDragState,
    calculate_arrow_key_step, get_mesh_file_filter, load_mesh_to_polydata,
    save_polydata_to_mesh, calculate_inertia_with_trimesh,
    is_apple_silicon, setup_qt_environment_for_apple_silicon,
    setup_signal_handlers, setup_signal_processing_timer,
    euler_to_quaternion, quaternion_to_euler, quaternion_to_matrix,
    VTKViewerBase
)

# M4 Mac (Apple Silicon) compatibility
IS_APPLE_SILICON = is_apple_silicon()

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

# Camera control constants (matching PartsEditor)
CAMERA_ROTATION_SENSITIVITY = 0.5
CAMERA_PAN_SPEED_FACTOR = 0.001
CAMERA_ZOOM_FACTOR = 0.1
CAMERA_ZOOM_IN_SCALE = 0.9
CAMERA_ZOOM_OUT_SCALE = 1.1
MOUSE_ZOOM_PAN_SCALE = 2.0

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

        step = calculate_arrow_key_step(shift_pressed, ctrl_pressed)

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


class GlobalKeyEventFilter(QObject):
    """Global event filter for WASD keyboard controls"""
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        """Route WASD key events to main window when not in input widgets"""
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QApplication

        if event.type() == QEvent.KeyPress:
            focus_widget = QApplication.focusWidget()
            if isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
                return False
            key = event.key()
            if key in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E,
                      Qt.Key_R, Qt.Key_T, Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
                self.main_window.keyPressEvent(event)
                return True

        return False


class MainWindow(VTKViewerBase, QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("URDF kitchen - MeshSourcer v0.1.0 -")
        self.resize(1000, 680)  # Wider window, 680px height

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

        # メインウィジェットとレイアウトの設定（横分割）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.set_ui_style()

        # 左側：3Dビュー
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # STL表示画面 - Use QLabel instead of QVTKRenderWindowInteractor for M4 Mac compatibility
        self.vtk_display = QLabel()
        self.vtk_display.setMinimumSize(600, 600)
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
        self.vtk_display.setMouseTracking(True)
        self.vtk_display.setFocusPolicy(Qt.StrongFocus)

        self.mouse_pressed = False
        self.last_mouse_pos = None
        self.vtk_display.installEventFilter(self)

        left_layout.addWidget(self.vtk_display)
        main_layout.addWidget(left_widget, 70)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setFixedWidth(400)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(4)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll)

        self.file_name_label = QLabel("File: No file loaded")
        self.file_name_label.setWordWrap(True)
        self.file_name_label.setMaximumWidth(380)
        self.file_name_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        right_layout.addWidget(self.file_name_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)
        self.load_button = QPushButton("Load Mesh")
        self.load_button.setFocusPolicy(Qt.NoFocus)
        self.load_button.clicked.connect(self.load_stl_file)
        self.load_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(self.load_button)

        right_layout.addLayout(button_layout)

        # Initialize collider state before setting up UI
        self.collider_type = "box"
        self.collider_params = {"box": [1.0, 1.0, 1.0], "sphere": [0.5], "cylinder": [0.5, 1.0], "capsule": [0.5, 1.0]}
        self.collider_position = [0.0, 0.0, 0.0]
        self.collider_rotation = [0.0, 0.0, 0.0]  # RPY in radians (internal)
        self.collider_rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.collider_actor = None
        self.collider_surface_actor = None
        self.collider_type_initialized = {"box": False, "sphere": False, "cylinder": False, "capsule": False}
        self.collider_show = False
        self.collider_first_show = True

        self.collider_blink_timer = QTimer()
        self.collider_blink_timer.timeout.connect(self.toggle_collider_surface)
        self.collider_blink_state = False

        self.center_position_active = True
        self.collider_position_active = False
        self.collider_size_active = False
        self.collider_radius_length_active = False
        self.collider_rotation_active = False

        right_layout.addSpacing(10)
        self.setup_reorient_mesh_ui(right_layout)

        right_layout.addSpacing(10)
        self.setup_collider_ui(right_layout)

        right_layout.addSpacing(10)
        self.setup_batch_converter_ui(right_layout)

        right_layout.addStretch()

        # Delay VTK setup until after window is shown
        self.vtk_initialized = False
        self.vtk_fully_ready = False  # VTK初期化完了フラグ
        self.pending_file_to_load = None  # 初期化後に読み込むファイル

        self.model_bounds = None
        self.stl_actor = None
        self.current_rotation = 0

        self.stl_center = list(self.absolute_origin)  # STLモデルの中心を大原点に初期化

        # Wireframe mode state
        self.wireframe_mode = False
        self.original_stl_color = None  # Store original color
        self._toggling_wireframe = False  # Lock for wireframe toggle

        # Help text visibility toggle
        self.help_visible = True  # Show help text by default

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
            # VTK初期化完了後、保留中のファイルがあれば読み込む
            QTimer.singleShot(400, self._load_pending_file)
        except Exception as e:
            print(f"ERROR in VTK final step: {e}")
            import traceback
            traceback.print_exc()

    def _load_pending_file(self):
        """VTK初期化完了後、保留中のファイルを読み込む"""
        self.vtk_fully_ready = True
        if self.pending_file_to_load:
            print(f"VTK ready. Loading pending file: {self.pending_file_to_load}")
            try:
                self.show_stl(self.pending_file_to_load)
                # Update file name label and current path
                self.file_name_label.setText(f"File: {self.pending_file_to_load}")
                self.current_stl_path = self.pending_file_to_load
                self.pending_file_to_load = None
                # Render to display the loaded file
                self.render_to_image()
            except Exception as e:
                print(f"Error loading pending file: {e}")
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
        # Dynamically adjust RenderWindow size to match QLabel size
        # This prevents text clipping and scale issues at different window sizes
        label_width = self.vtk_display.width()
        label_height = self.vtk_display.height()
        if label_width > 0 and label_height > 0:
            self.render_window.SetSize(label_width, label_height)

        self.offscreen_renderer.update_display(self.vtk_display, restore_focus=True)
        self._render_counter += 1

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
            QScrollArea {
                background-color: #2b2b2b;
                border: none;
            }
            QWidget {
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
            QComboBox {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #5a5a5a;
                border-radius: 3px;
                padding: 2px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #5a5a5a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #4a90e2;
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

    def setup_points_ui(self):
        """Create and return the Center Position layout (no longer adds to parent layout)"""
        points_layout = QGridLayout()
        points_layout.setSpacing(4)  # Reduce spacing between grid cells
        points_layout.setVerticalSpacing(5)  # Vertical spacing: 5px
        points_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        for i in range(self.num_points):
            # Checkbox and Target Marker label on the same row
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, idx=i: self.on_center_position_checkbox_changed(idx, state))
            self.point_checkboxes.append(checkbox)
            points_layout.addWidget(checkbox, i*2, 0)

            label = QLabel("Target Marker Position:")
            points_layout.addWidget(label, i*2, 1, 1, 7)  # Span remaining columns

            # Input fields on the next row
            inputs = []
            for j, axis in enumerate(['X', 'Y', 'Z']):
                input_field = QLineEdit(str(self.point_coords[i][j]))
                input_field.setMaximumWidth(80)  # Prevent horizontal expansion
                # Only focus when explicitly clicked, don't steal keyboard focus
                input_field.setFocusPolicy(Qt.ClickFocus)
                # Return focus to vtk_display when editing is done
                input_field.editingFinished.connect(lambda idx=i: self.on_center_position_changed(idx))
                input_field.editingFinished.connect(lambda: self.vtk_display.setFocus())
                inputs.append(input_field)
                points_layout.addWidget(QLabel(f"{axis}:"), i*2+1, j*2)
                points_layout.addWidget(input_field, i*2+1, j*2+1)
            self.point_inputs.append(inputs)

        return points_layout

    def setup_reorient_mesh_ui(self, layout):
        """Setup Origin Marker UI group box"""
        # Create group box for reorient mesh section
        reorient_group = QGroupBox("Origin Coordinates")
        reorient_layout = QVBoxLayout()
        reorient_layout.setSpacing(4)  # Reduce vertical spacing
        reorient_layout.setContentsMargins(8, 8, 8, 8)  # Reduce margins

        # Volume表示
        self.volume_label = QLabel("Volume (m^3): 0.000000")
        self.volume_label.setMaximumWidth(380)  # Prevent horizontal expansion
        reorient_layout.addWidget(self.volume_label)

        # Center Position (from setup_points_ui)
        points_layout = self.setup_points_ui()
        reorient_layout.addLayout(points_layout)

        # Add spacing before Clean Mesh checkbox
        reorient_layout.addSpacing(2)

        # Clean Mesh checkbox for save operation (right-aligned)
        clean_mesh_layout = QHBoxLayout()
        clean_mesh_layout.addStretch()  # Push checkbox to the right
        self.reorient_clean_mesh_checkbox = QCheckBox("Clean Mesh")
        self.reorient_clean_mesh_checkbox.setFocusPolicy(Qt.NoFocus)
        self.reorient_clean_mesh_checkbox.setChecked(False)
        clean_mesh_layout.addWidget(self.reorient_clean_mesh_checkbox)
        reorient_layout.addLayout(clean_mesh_layout)

        # Add spacing after Clean Mesh checkbox
        reorient_layout.addSpacing(2)

        # Save, Reset Marker, and Set Front as X buttons (horizontal layout)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)  # Reduce horizontal spacing

        reset_button = QPushButton("Reset Marker")
        reset_button.setFocusPolicy(Qt.NoFocus)
        reset_button.clicked.connect(self.handle_set_reset)
        reset_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(reset_button)

        set_front_button = QPushButton("Set Front as X")
        set_front_button.setFocusPolicy(Qt.NoFocus)
        set_front_button.clicked.connect(self.handle_set_reset)
        set_front_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(set_front_button)

        self.export_stl_button = QPushButton("Save")
        self.export_stl_button.setFocusPolicy(Qt.NoFocus)
        self.export_stl_button.clicked.connect(self.export_stl_with_new_origin)
        self.export_stl_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(self.export_stl_button)

        reorient_layout.addLayout(button_layout)

        reorient_group.setLayout(reorient_layout)
        layout.addWidget(reorient_group)

    def setup_collider_ui(self, layout):
        """Setup collider design UI"""
        # Create group box for collider section
        collider_group = QGroupBox("Collider Design")
        collider_layout = QVBoxLayout()
        collider_layout.setSpacing(4)  # Reduce vertical spacing
        collider_layout.setContentsMargins(8, 8, 8, 8)  # Reduce margins

        # Show checkbox and Type selection
        type_layout = QHBoxLayout()
        type_layout.setSpacing(4)  # Reduce horizontal spacing
        self.collider_show_checkbox = QCheckBox("Show Collider")
        self.collider_show_checkbox.setChecked(False)  # Default: unchecked
        self.collider_show_checkbox.stateChanged.connect(self.on_collider_show_changed)
        type_layout.addWidget(self.collider_show_checkbox)

        type_layout.addWidget(QLabel("Type:"))
        self.collider_type_combo = QComboBox()
        self.collider_type_combo.addItems(["box", "sphere", "cylinder", "capsule"])
        self.collider_type_combo.setFocusPolicy(Qt.NoFocus)
        self.collider_type_combo.currentTextChanged.connect(self.on_collider_type_changed)
        type_layout.addWidget(self.collider_type_combo)

        # Add spacer to push Load Collider button to the right
        type_layout.addStretch()

        # Load Collider button on the right end
        load_collider_button = QPushButton("Load Collider")
        load_collider_button.setFocusPolicy(Qt.NoFocus)
        load_collider_button.clicked.connect(self.load_collider)
        load_collider_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        type_layout.addWidget(load_collider_button)

        collider_layout.addLayout(type_layout)

        # Position inputs with checkbox (moved to be right after Show/Type)
        position_layout = QGridLayout()
        self.collider_position_checkbox = QCheckBox()
        self.collider_position_checkbox.setChecked(False)
        self.collider_position_checkbox.stateChanged.connect(self.on_collider_position_checkbox_changed)
        position_layout.addWidget(self.collider_position_checkbox, 0, 0)
        position_layout.addWidget(QLabel("Position:"), 0, 1)
        self.collider_position_inputs = []
        for i, axis in enumerate(['X', 'Y', 'Z']):
            position_layout.addWidget(QLabel(f"{axis}:"), 0, i*2+2)
            input_field = QLineEdit("0.0")
            input_field.setMaximumWidth(60)  # Prevent horizontal expansion
            input_field.setFocusPolicy(Qt.ClickFocus)
            input_field.editingFinished.connect(lambda idx=i: self.on_collider_position_input_changed(idx))
            input_field.editingFinished.connect(lambda: self.vtk_display.setFocus())
            position_layout.addWidget(input_field, 0, i*2+3)
            self.collider_position_inputs.append(input_field)
        collider_layout.addLayout(position_layout)

        # Parameters section (dynamic based on type)
        self.collider_params_layout = QGridLayout()
        self.collider_param_labels = []
        self.collider_param_inputs = []
        collider_layout.addLayout(self.collider_params_layout)

        # Rotation inputs with checkbox
        rotation_layout = QGridLayout()
        self.collider_rotation_checkbox = QCheckBox()
        self.collider_rotation_checkbox.setChecked(False)
        self.collider_rotation_checkbox.stateChanged.connect(self.on_collider_rotation_checkbox_changed)
        rotation_layout.addWidget(self.collider_rotation_checkbox, 0, 0)
        rotation_layout.addWidget(QLabel("Rotation (deg):"), 0, 1)
        self.collider_rotation_inputs = []
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            rotation_layout.addWidget(QLabel(f"{axis}:"), 0, i*2+2)
            input_field = QLineEdit("0.0")
            input_field.setMaximumWidth(60)  # Prevent horizontal expansion
            input_field.setFocusPolicy(Qt.ClickFocus)
            input_field.editingFinished.connect(lambda idx=i: self.on_collider_rotation_input_changed(idx))
            input_field.editingFinished.connect(lambda: self.vtk_display.setFocus())
            rotation_layout.addWidget(input_field, 0, i*2+3)
            self.collider_rotation_inputs.append(input_field)
        collider_layout.addLayout(rotation_layout)

        # Add spacing before buttons
        collider_layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()

        draft_button = QPushButton("Rough Fit")
        draft_button.setFocusPolicy(Qt.NoFocus)
        draft_button.clicked.connect(self.draft_collider)
        draft_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(draft_button)

        reset_button = QPushButton("Reset Collider")
        reset_button.setFocusPolicy(Qt.NoFocus)
        reset_button.clicked.connect(self.reset_and_fit_collider)
        reset_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(reset_button)

        export_collider_button = QPushButton("Export Collider")
        export_collider_button.setFocusPolicy(Qt.NoFocus)
        export_collider_button.clicked.connect(self.export_collider)
        export_collider_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        button_layout.addWidget(export_collider_button)

        collider_layout.addLayout(button_layout)

        collider_group.setLayout(collider_layout)
        layout.addWidget(collider_group)

        # Initialize parameter inputs for default type (box)
        self.update_collider_param_inputs()

    def setup_batch_converter_ui(self, layout):
        """Setup Batch Mesh Converter UI group box"""
        # Create group box for batch converter section
        converter_group = QGroupBox("Batch Mesh Converter")
        converter_layout = QVBoxLayout()
        converter_layout.setSpacing(4)  # Reduce vertical spacing
        converter_layout.setContentsMargins(8, 8, 8, 8)  # Remove bottom margin

        # Radio button style for white text
        radio_style = "QRadioButton { color: white; }"

        # Input format selection (label and radio buttons on same line)
        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)

        input_label = QLabel("Input:")
        input_layout.addWidget(input_label)

        self.input_format_group = QButtonGroup()
        self.input_stl_radio = QRadioButton(".stl")
        self.input_dae_radio = QRadioButton(".dae")
        self.input_obj_radio = QRadioButton(".obj")
        self.input_stl_radio.setChecked(True)  # Default: .stl

        # Apply white text style
        self.input_stl_radio.setStyleSheet(radio_style)
        self.input_dae_radio.setStyleSheet(radio_style)
        self.input_obj_radio.setStyleSheet(radio_style)

        self.input_format_group.addButton(self.input_stl_radio, 0)
        self.input_format_group.addButton(self.input_dae_radio, 1)
        self.input_format_group.addButton(self.input_obj_radio, 2)

        input_layout.addWidget(self.input_stl_radio)
        input_layout.addWidget(self.input_dae_radio)
        input_layout.addWidget(self.input_obj_radio)
        input_layout.addStretch()  # Push everything to the left

        # Add Clean Mesh checkbox to the right side of Input line
        self.clean_mesh_checkbox = QCheckBox("Clean Mesh")
        self.clean_mesh_checkbox.setFocusPolicy(Qt.NoFocus)
        self.clean_mesh_checkbox.setChecked(False)
        input_layout.addWidget(self.clean_mesh_checkbox)

        converter_layout.addLayout(input_layout)

        # Output format selection (label and radio buttons on same line)
        output_layout = QHBoxLayout()
        output_layout.setSpacing(4)

        output_label = QLabel("Output:")
        output_layout.addWidget(output_label)

        self.output_format_group = QButtonGroup()
        self.output_stl_radio = QRadioButton(".stl")
        self.output_dae_radio = QRadioButton(".dae")
        self.output_obj_radio = QRadioButton(".obj")
        self.output_dae_radio.setChecked(True)  # Default: .dae

        # Apply white text style
        self.output_stl_radio.setStyleSheet(radio_style)
        self.output_dae_radio.setStyleSheet(radio_style)
        self.output_obj_radio.setStyleSheet(radio_style)

        self.output_format_group.addButton(self.output_stl_radio, 0)
        self.output_format_group.addButton(self.output_dae_radio, 1)
        self.output_format_group.addButton(self.output_obj_radio, 2)

        output_layout.addWidget(self.output_stl_radio)
        output_layout.addWidget(self.output_dae_radio)
        output_layout.addWidget(self.output_obj_radio)
        output_layout.addStretch()  # Push everything to the left
        converter_layout.addLayout(output_layout)

        # Convert button
        convert_button = QPushButton("Select Directory and Convert")
        convert_button.setFocusPolicy(Qt.NoFocus)
        convert_button.clicked.connect(self.batch_convert_meshes)
        convert_button.clicked.connect(lambda: QTimer.singleShot(100, lambda: self.vtk_display.setFocus()))
        converter_layout.addWidget(convert_button)

        converter_group.setLayout(converter_layout)
        layout.addWidget(converter_group)

    def clean_polydata(self, polydata):
        """Apply VTK cleaning filters to polydata"""
        import vtk

        if polydata is None:
            return polydata

        original_poly_count = polydata.GetNumberOfPolys()

        # Step 1: Clean polydata - remove duplicate points and degenerate cells
        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputData(polydata)
        clean_filter.PointMergingOn()
        clean_filter.ToleranceIsAbsoluteOn()
        clean_filter.SetAbsoluteTolerance(1e-6)
        clean_filter.Update()

        # Step 2: Ensure triangles to keep OBJ writer stable (especially for DAE -> OBJ)
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(clean_filter.GetOutputPort())
        triangle_filter.Update()

        # Step 3: Compute and fix normals
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputConnection(triangle_filter.GetOutputPort())
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOn()
        normals_filter.ConsistencyOn()  # Make normals consistent
        normals_filter.AutoOrientNormalsOn()  # Auto-orient normals outward
        normals_filter.SplittingOff()  # Don't split vertices (smooth shading)
        normals_filter.Update()

        # Step 4: Re-triangulate after normals to ensure OBJ writer compatibility
        post_triangle_filter = vtk.vtkTriangleFilter()
        post_triangle_filter.SetInputConnection(normals_filter.GetOutputPort())
        post_triangle_filter.Update()

        cleaned = post_triangle_filter.GetOutput()
        # Safety: if cleaning dropped all polys, fall back to original
        if original_poly_count > 0 and cleaned.GetNumberOfPolys() == 0:
            print("Warning: CleanMesh removed all polygons; using original mesh data.")
            return polydata

        return cleaned

    def clean_polydata_conservative(self, polydata):
        """Apply conservative VTK cleaning for OBJ files (preserves normals and vertices)"""
        import vtk

        if polydata is None:
            return polydata

        original_poly_count = polydata.GetNumberOfPolys()

        # Step 1: Remove only degenerate cells (zero-area triangles)
        # Use vtkCleanPolyData with PointMergingOff to preserve vertex structure
        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputData(polydata)
        clean_filter.PointMergingOff()  # Do NOT merge vertices - preserves OBJ vertex normals
        clean_filter.ConvertLinesToPointsOff()
        clean_filter.ConvertPolysToLinesOff()
        clean_filter.ConvertStripsToPolysOff()
        clean_filter.Update()

        # Step 2: Ensure triangles for format compatibility
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(clean_filter.GetOutputPort())
        triangle_filter.PassVertsOff()
        triangle_filter.PassLinesOff()
        triangle_filter.Update()

        cleaned = triangle_filter.GetOutput()

        # Safety check: if cleaning removed too many polygons, use original
        if original_poly_count > 0:
            new_poly_count = cleaned.GetNumberOfPolys()
            # If more than 10% of faces were removed, something went wrong
            if new_poly_count < original_poly_count * 0.9:
                print(f"Warning: Conservative clean removed too many faces ({original_poly_count} -> {new_poly_count}), using original mesh.")
                return polydata

        return cleaned

    def save_obj_with_trimesh(self, polydata, file_path):
        """Save OBJ using trimesh (more robust for cleaned DAE meshes)."""
        try:
            import trimesh
            import numpy as np
            import vtk
        except ImportError:
            return False

        if polydata is None:
            return False

        # Ensure triangles
        tri_filter = vtk.vtkTriangleFilter()
        tri_filter.SetInputData(polydata)
        tri_filter.Update()
        tri_poly = tri_filter.GetOutput()

        num_points = tri_poly.GetNumberOfPoints()
        num_cells = tri_poly.GetNumberOfCells()
        if num_points == 0 or num_cells == 0:
            return False

        vertices = np.zeros((num_points, 3))
        for i in range(num_points):
            vertices[i] = tri_poly.GetPoint(i)

        faces = []
        for i in range(num_cells):
            cell = tri_poly.GetCell(i)
            if cell.GetNumberOfPoints() == 3:
                faces.append([
                    cell.GetPointId(0),
                    cell.GetPointId(1),
                    cell.GetPointId(2)
                ])

        if not faces:
            return False

        faces = np.array(faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(file_path)
        return True

    def convert_dae_to_obj_with_trimesh_clean(self, dae_path, obj_path):
        """Clean and convert DAE to OBJ using trimesh directly (avoid VTK cleanup issues)."""
        try:
            import trimesh
        except ImportError:
            return False

        try:
            mesh = trimesh.load(str(dae_path), force='mesh')
            # Handle Scene fallback
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    return False
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

            if mesh is None:
                return False

            # Clean operations
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            try:
                mesh.merge_vertices(epsilon=1e-6)
            except Exception:
                pass
            try:
                mesh.process(validate=True)
            except Exception:
                pass

            mesh.export(str(obj_path))
            return True
        except Exception:
            return False

    def batch_convert_meshes(self):
        """Batch convert mesh files from input format to output format"""
        input_ext = None
        if self.input_stl_radio.isChecked():
            input_ext = ".stl"
        elif self.input_dae_radio.isChecked():
            input_ext = ".dae"
        elif self.input_obj_radio.isChecked():
            input_ext = ".obj"

        output_ext = None
        if self.output_stl_radio.isChecked():
            output_ext = ".stl"
        elif self.output_dae_radio.isChecked():
            output_ext = ".dae"
        elif self.output_obj_radio.isChecked():
            output_ext = ".obj"

        clean_mesh_enabled = self.clean_mesh_checkbox.isChecked()

        if input_ext == output_ext and not clean_mesh_enabled:
            print(f"Input and output formats are the same ({input_ext}). No conversion needed.")
            print("Tip: Enable 'Clean Mesh' to clean files without changing format.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Batch Conversion")
        if not directory:
            return

        from pathlib import Path
        input_files = list(Path(directory).glob(f"*{input_ext}"))

        if not input_files:
            print(f"No {input_ext} files found in {directory}")
            return

        print(f"Found {len(input_files)} {input_ext} file(s) in {directory}")

        existing_files = []
        for input_file in input_files:
            output_file = input_file.with_suffix(output_ext)
            if output_file.exists():
                existing_files.append(output_file.name)

        overwrite = True
        if existing_files:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Overwrite Existing Files?",
                f"{len(existing_files)} file(s) already exist:\n{', '.join(existing_files[:5])}"
                + (f"\n...and {len(existing_files) - 5} more" if len(existing_files) > 5 else "")
                + "\n\nDo you want to overwrite all existing files?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            overwrite = (reply == QMessageBox.Yes)
            if not overwrite:
                print("Conversion cancelled by user.")
                return

        success_count = 0
        error_count = 0
        success_files = []
        error_files = []

        from urdf_kitchen_utils import load_mesh_to_polydata, save_polydata_to_mesh

        for input_file in input_files:
            output_file = input_file.with_suffix(output_ext)
            try:
                polydata, volume, color = load_mesh_to_polydata(str(input_file))
                if polydata:
                    if clean_mesh_enabled:
                        # DAE -> OBJ with Clean Mesh: use trimesh direct path for stability
                        if input_ext == ".dae" and output_ext == ".obj":
                            if self.convert_dae_to_obj_with_trimesh_clean(input_file, output_file):
                                action = "Cleaned"
                                print(f"{action}: {input_file.name} → {output_file.name}")
                                success_files.append(input_file.name)
                                success_count += 1
                                continue
                        # Use conservative cleaning for OBJ input to preserve vertex normals
                        if input_ext == ".obj":
                            polydata = self.clean_polydata_conservative(polydata)
                        else:
                            polydata = self.clean_polydata(polydata)

                    # For cleaned DAE -> OBJ, use trimesh export to avoid VTK OBJ writer issues
                    saved = False
                    if output_ext == ".obj" and clean_mesh_enabled:
                        saved = self.save_obj_with_trimesh(polydata, str(output_file))
                    if not saved:
                        save_polydata_to_mesh(str(output_file), polydata, mesh_color=color)

                    action = "Cleaned" if (input_ext == output_ext and clean_mesh_enabled) else "Converted"
                    print(f"{action}: {input_file.name} → {output_file.name}")
                    success_files.append(input_file.name)
                    success_count += 1
                else:
                    print(f"Error loading: {input_file.name}")
                    error_files.append(input_file.name)
                    error_count += 1
            except Exception as e:
                print(f"Error converting {input_file.name}: {e}")
                error_files.append(input_file.name)
                error_count += 1

        from PySide6.QtWidgets import QMessageBox

        operation = "Batch Cleaning" if (input_ext == output_ext and clean_mesh_enabled) else "Batch Conversion"
        result_msg = f"{operation} Complete\n\n"
        result_msg += f"Total files: {len(input_files)}\n"
        result_msg += f"✓ Success: {success_count} file(s)\n"
        result_msg += f"✗ Errors: {error_count} file(s)\n"
        if clean_mesh_enabled:
            result_msg += "\n[Mesh cleaning applied: normals fixed, duplicates removed]\n"

        if success_count > 0 and success_count <= 5:
            result_msg += f"\nConverted files:\n"
            result_msg += "\n".join([f"  • {f}" for f in success_files])
        elif success_count > 5:
            result_msg += f"\nConverted files (first 5):\n"
            result_msg += "\n".join([f"  • {f}" for f in success_files[:5]])
            result_msg += f"\n  ... and {success_count - 5} more"

        if error_count > 0 and error_count <= 5:
            result_msg += f"\n\nFailed files:\n"
            result_msg += "\n".join([f"  • {f}" for f in error_files])
        elif error_count > 5:
            result_msg += f"\n\nFailed files (first 5):\n"
            result_msg += "\n".join([f"  • {f}" for f in error_files[:5]])
            result_msg += f"\n  ... and {error_count - 5} more"

        op_name = "Cleaning" if (input_ext == output_ext and clean_mesh_enabled) else "Conversion"
        if error_count == 0:
            icon = QMessageBox.Information
            title = f"{op_name} Successful"
        elif success_count == 0:
            icon = QMessageBox.Critical
            title = f"{op_name} Failed"
        else:
            icon = QMessageBox.Warning
            title = f"{op_name} Completed with Errors"

        QMessageBox(icon, title, result_msg, QMessageBox.Ok, self).exec()

        print(f"\nBatch {op_name.lower()} complete:")
        print(f"  Success: {success_count} file(s)")
        print(f"  Errors: {error_count} file(s)")

    def update_collider_param_inputs(self):
        """Update parameter input fields based on selected collider type"""
        # Clear existing parameter inputs by removing all items from layout
        while self.collider_params_layout.count():
            item = self.collider_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Clear lists
        self.collider_param_labels.clear()
        self.collider_param_inputs.clear()
        if not hasattr(self, 'collider_param_checkboxes'):
            self.collider_param_checkboxes = []
        else:
            self.collider_param_checkboxes.clear()

        # Add new parameter inputs based on type
        collider_type = self.collider_type
        params = self.collider_params[collider_type]

        col = 0

        # Add one checkbox at the beginning for all types
        checkbox = QCheckBox()
        # Set checkbox state based on current state variable
        if collider_type == "box":
            checkbox.setChecked(self.collider_size_active)
            checkbox.stateChanged.connect(self.on_collider_size_checkbox_changed)
        elif collider_type in ["sphere", "cylinder", "capsule"]:
            checkbox.setChecked(self.collider_radius_length_active)
            checkbox.stateChanged.connect(self.on_collider_radius_length_checkbox_changed)
        self.collider_param_checkboxes.append(checkbox)
        self.collider_params_layout.addWidget(checkbox, 0, col)
        col += 1

        # Add parameter labels and inputs
        if collider_type == "box":
            labels = ["Size:  X:", "Y:", "Z:"]
        elif collider_type == "sphere":
            labels = ["Radius:"]
        elif collider_type in ["cylinder", "capsule"]:
            labels = ["Radius:", "Length:"]
        else:
            labels = []

        for i, (label_text, param_value) in enumerate(zip(labels, params)):
            label = QLabel(label_text)
            self.collider_param_labels.append(label)
            self.collider_params_layout.addWidget(label, 0, col)
            col += 1

            input_field = QLineEdit(str(param_value))
            input_field.setMaximumWidth(60)  # Prevent horizontal expansion
            input_field.setFocusPolicy(Qt.ClickFocus)
            input_field.editingFinished.connect(lambda idx=i: self.on_collider_param_input_changed(idx))
            input_field.editingFinished.connect(lambda: self.vtk_display.setFocus())
            self.collider_param_inputs.append(input_field)
            self.collider_params_layout.addWidget(input_field, 0, col)
            col += 1

    def on_collider_type_changed(self, new_type):
        """Handle collider type change"""
        old_type = self.collider_type
        self.collider_type = new_type

        # Auto-uncheck hidden checkboxes when switching between box and other types
        # Note: Switching within sphere/cylinder/capsule preserves the checkbox state
        # If switching from box to sphere/cylinder/capsule, uncheck Size
        if old_type == "box" and new_type in ["sphere", "cylinder", "capsule"]:
            if self.collider_size_active:
                self.collider_size_active = False
        # If switching from sphere/cylinder/capsule to box, uncheck Radius/Length
        elif old_type in ["sphere", "cylinder", "capsule"] and new_type == "box":
            if self.collider_radius_length_active:
                self.collider_radius_length_active = False
        # Switching within sphere/cylinder/capsule: checkbox state is preserved (no action needed)

        self.update_collider_param_inputs()

        # Only update 3D display if Show is ON
        if self.collider_show:
            # Auto-draft on first selection of this type
            if not self.collider_type_initialized[new_type]:
                self.collider_type_initialized[new_type] = True
                # Only auto-draft if mesh is loaded
                if self.model_bounds:
                    self.draft_collider()
                else:
                    # Even without mesh, display collider with default params
                    self.update_collider_display()
            else:
                # Update display with existing values
                self.update_collider_display()

    def on_center_position_checkbox_changed(self, index, state):
        """Handle Center Position checkbox change"""
        if state == 2:  # Checked
            # Set this active, others inactive (mutual exclusivity)
            self.center_position_active = True
            self.collider_size_active = False
            self.collider_position_active = False
            self.collider_radius_length_active = False
            self.collider_rotation_active = False

            # Uncheck other checkboxes
            if hasattr(self, 'collider_position_checkbox'):
                self.collider_position_checkbox.blockSignals(True)
                self.collider_position_checkbox.setChecked(False)
                self.collider_position_checkbox.blockSignals(False)
            if hasattr(self, 'collider_rotation_checkbox'):
                self.collider_rotation_checkbox.blockSignals(True)
                self.collider_rotation_checkbox.setChecked(False)
                self.collider_rotation_checkbox.blockSignals(False)
            # Uncheck parameter checkboxes (size/radius/length)
            if hasattr(self, 'collider_param_checkboxes'):
                for cb in self.collider_param_checkboxes:
                    if cb:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
        else:  # Unchecked
            self.center_position_active = False

    def on_center_position_changed(self, index):
        """Handle Center Position input change"""
        try:
            for j in range(3):
                value = float(self.point_inputs[index][j].text())
                self.point_coords[index][j] = value
            self.update_point_display(index)
            self.render_to_image()
        except ValueError:
            print(f"Invalid center position value")

    def on_collider_show_changed(self, state):
        """Handle collider show checkbox change"""
        # state is an int: 2=Checked, 0=Unchecked
        self.collider_show = (state == 2)

        if self.collider_show:
            # Show ON: Force mesh to wireframe, show collider
            if self.stl_actor:
                # Force wireframe mode
                if not self.wireframe_mode:
                    self.toggle_wireframe()
                    self.render_to_image()  # Render immediately to apply wireframe

            # First time Show is clicked - auto draft
            if self.collider_first_show:
                self.collider_first_show = False
                if self.model_bounds:
                    # Draft collider based on mesh bounds
                    self.draft_collider()
                else:
                    # No mesh loaded, just display with default params
                    self.update_collider_display()
            else:
                # Not first time, just update display
                self.update_collider_display()
        else:
            # Show OFF: Force mesh to surface mode, hide collider
            if self.stl_actor:
                # Force surface mode
                if self.wireframe_mode:
                    self.toggle_wireframe()
                    self.render_to_image()  # Render immediately to apply surface mode

            # Hide collider
            if self.collider_actor:
                self.renderer.RemoveActor(self.collider_actor)
                self.collider_actor = None
            if self.collider_surface_actor:
                self.renderer.RemoveActor(self.collider_surface_actor)
                self.collider_surface_actor = None
            if self.collider_blink_timer.isActive():
                self.collider_blink_timer.stop()
            self.render_to_image()

    def on_collider_param_input_changed(self, index):
        """Handle collider parameter input change"""
        try:
            value = float(self.collider_param_inputs[index].text())
            self.collider_params[self.collider_type][index] = value
            # Format to 4 decimal places with rounding
            self.collider_param_inputs[index].setText(f"{value:.4f}")
            if self.collider_show:
                self.update_collider_display()
        except ValueError:
            print(f"Invalid parameter value at index {index}")

    def on_collider_position_checkbox_changed(self, state):
        """Handle collider position checkbox change"""
        if state == 2:  # Checked
            # Set this active, others inactive (mutual exclusivity)
            self.center_position_active = False
            self.collider_size_active = False
            self.collider_position_active = True
            self.collider_radius_length_active = False
            self.collider_rotation_active = False

            # Uncheck other checkboxes
            for cb in self.point_checkboxes:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            if hasattr(self, 'collider_rotation_checkbox'):
                self.collider_rotation_checkbox.blockSignals(True)
                self.collider_rotation_checkbox.setChecked(False)
                self.collider_rotation_checkbox.blockSignals(False)
            # Uncheck parameter checkboxes (size/radius/length)
            if hasattr(self, 'collider_param_checkboxes'):
                for cb in self.collider_param_checkboxes:
                    if cb:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
        else:  # Unchecked
            self.collider_position_active = False

    def on_collider_position_input_changed(self, index):
        """Handle collider position input change"""
        try:
            value = float(self.collider_position_inputs[index].text())
            self.collider_position[index] = value
            # Format to 4 decimal places with rounding
            self.collider_position_inputs[index].setText(f"{value:.4f}")
            if self.collider_show:
                self.update_collider_display()
        except ValueError:
            print(f"Invalid position value at index {index}")

    def on_collider_rotation_checkbox_changed(self, state):
        """Handle collider rotation checkbox change"""
        if state == 2:  # Checked
            # Set this active, others inactive (mutual exclusivity)
            self.center_position_active = False
            self.collider_size_active = False
            self.collider_position_active = False
            self.collider_radius_length_active = False
            self.collider_rotation_active = True

            # Uncheck other checkboxes
            for cb in self.point_checkboxes:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            if hasattr(self, 'collider_position_checkbox'):
                self.collider_position_checkbox.blockSignals(True)
                self.collider_position_checkbox.setChecked(False)
                self.collider_position_checkbox.blockSignals(False)
            # Uncheck parameter checkboxes (size/radius/length)
            if hasattr(self, 'collider_param_checkboxes'):
                for cb in self.collider_param_checkboxes:
                    if cb:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
        else:  # Unchecked
            self.collider_rotation_active = False

    def on_collider_rotation_input_changed(self, index):
        """Handle collider rotation input change"""
        try:
            import math
            value_deg = float(self.collider_rotation_inputs[index].text())
            # Store internally in radians
            self.collider_rotation[index] = math.radians(value_deg)
            # Format to 2 decimal places (degrees for UI)
            self.collider_rotation_inputs[index].setText(f"{value_deg:.2f}")

            # Update quaternion from Euler angles (VTK expects degrees for euler_to_quaternion)
            roll_deg = math.degrees(self.collider_rotation[0])
            pitch_deg = math.degrees(self.collider_rotation[1])
            yaw_deg = math.degrees(self.collider_rotation[2])
            self.collider_rotation_quaternion = euler_to_quaternion(
                roll_deg,
                pitch_deg,
                yaw_deg
            )

            if self.collider_show:
                self.update_collider_display()
        except ValueError:
            print(f"Invalid rotation value at index {index}")

    def on_collider_size_checkbox_changed(self, state):
        """Handle collider size checkbox change (for box)"""
        if state == 2:  # Checked
            # Set this active, others inactive (mutual exclusivity)
            self.center_position_active = False
            self.collider_size_active = True
            self.collider_position_active = False
            self.collider_radius_length_active = False
            self.collider_rotation_active = False

            # Uncheck other checkboxes
            for cb in self.point_checkboxes:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            if hasattr(self, 'collider_position_checkbox'):
                self.collider_position_checkbox.blockSignals(True)
                self.collider_position_checkbox.setChecked(False)
                self.collider_position_checkbox.blockSignals(False)
            if hasattr(self, 'collider_rotation_checkbox'):
                self.collider_rotation_checkbox.blockSignals(True)
                self.collider_rotation_checkbox.setChecked(False)
                self.collider_rotation_checkbox.blockSignals(False)
        else:  # Unchecked
            self.collider_size_active = False

    def on_collider_radius_length_checkbox_changed(self, state):
        """Handle collider radius/length checkbox change (shared for sphere/cylinder/capsule)"""
        if state == 2:  # Checked
            # Set this active, others inactive (mutual exclusivity)
            self.center_position_active = False
            self.collider_size_active = False
            self.collider_position_active = False
            self.collider_radius_length_active = True
            self.collider_rotation_active = False

            # Uncheck other checkboxes
            for cb in self.point_checkboxes:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            if hasattr(self, 'collider_position_checkbox'):
                self.collider_position_checkbox.blockSignals(True)
                self.collider_position_checkbox.setChecked(False)
                self.collider_position_checkbox.blockSignals(False)
            if hasattr(self, 'collider_rotation_checkbox'):
                self.collider_rotation_checkbox.blockSignals(True)
                self.collider_rotation_checkbox.setChecked(False)
                self.collider_rotation_checkbox.blockSignals(False)
        else:  # Unchecked
            self.collider_radius_length_active = False

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
        self.camera_controller.reset_camera(position=[10, 0, 0], view_up=[0, 0, 1])
        self.camera_rotation = [0, 0, 0]
        self.current_rotation = 0
        # 回転アニメーションの状態をリセット
        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.stop()
        self.is_animating = False
        if hasattr(self, 'animated_rotation') and self.animated_rotation:
            self.animated_rotation.is_animating = False
            self.animated_rotation.current_frame = 0
            self.animated_rotation.target_angle = 0
            self.animated_rotation.rotation_type = None

        if hasattr(self, 'axes_widget') and self.axes_widget is not None:
            self.renderer.RemoveViewProp(self.axes_widget.GetOrientationMarker())
        self.axes_widget = None

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

        # Inertia計算は削除（精密計算が必要な場合はcalculate_inertia_with_trimesh使用）
        # 簡略化された立方体仮定の計算は使用しない

        # 結果を入力フィールドに反映
        for prop in priority_order:
            input_field = getattr(self, f"{prop}_input")
            if prop in values:
                input_field.setText(f"{values[prop]:.12f}")

    def update_all_points_size(self, obj=None, event=None):
        for index, actor in enumerate(self.point_actors):
            if actor:
                self.renderer.RemoveActor(actor)
                self.point_actors[index] = create_crosshair_marker(
                    coords=[0, 0, 0],
                    radius_scale=self.calculate_sphere_radius()
                )
                self.point_actors[index].SetPosition(self.point_coords[index])
                self.renderer.AddActor(self.point_actors[index])

    def calculate_sphere_radius(self):
        return AdaptiveMarkerSize.calculate_sphere_radius(self.renderer)

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
        """画面上に操作説明を表示"""
        if not hasattr(self, 'text_actors'):
            self.text_actors = []

        # 左上のテキスト
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
        text_actor_top.GetTextProperty().SetColor(0.3, 0.8, 1.0)  # 水色
        text_actor_top.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor_top.SetPosition(0.03, 0.97)  # 左上に配置
        text_actor_top.GetTextProperty().SetJustificationToLeft()
        text_actor_top.GetTextProperty().SetVerticalJustificationToTop()
        self.renderer.AddActor(text_actor_top)
        self.text_actors.append(text_actor_top)

        # 左下のテキスト
        text_actor_bottom = vtk.vtkTextActor()
        text_actor_bottom.SetInput(
            "[Arrows] : Move Point 10mm\n"
            " +[Shift]: Move Point 1mm\n"
            "  +[Ctrl]: Move Point 0.1mm\n\n"
        )
        text_actor_bottom.GetTextProperty().SetFontSize(14)
        text_actor_bottom.GetTextProperty().SetColor(0.3, 0.8, 1.0)  # 水色
        text_actor_bottom.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor_bottom.SetPosition(0.03, 0.03)  # 左下に配置
        text_actor_bottom.GetTextProperty().SetJustificationToLeft()
        text_actor_bottom.GetTextProperty().SetVerticalJustificationToBottom()
        self.renderer.AddActor(text_actor_bottom)
        self.text_actors.append(text_actor_bottom)

    def update_instruction_text_layout(self):
        """Update instruction text visibility (font size is fixed)"""
        if not hasattr(self, 'text_actors') or len(self.text_actors) < 2:
            return

        # Update visibility based on help_visible flag (font size remains fixed)
        for actor in self.text_actors:
            actor.SetVisibility(1 if self.help_visible else 0)

    def load_stl_file(self):
        print("Opening file dialog...")
        # Use common utility function for file filter
        file_filter = get_mesh_file_filter(TRIMESH_AVAILABLE)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model File", "", file_filter)
        if file_path:
            print(f"Loading STL file: {file_path}")
            self.file_name_label.setText(f"File: {file_path}")
            # Save the loaded STL file path for save dialog defaults
            self.current_stl_path = file_path

            try:
                self.show_stl(file_path)
                # 読み込み直後はRと同じカメラ初期化を行う（WASDの90度回転を正確にするため）
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

        # Use common utility function to load mesh
        poly_data, volume, extracted_color = load_mesh_to_polydata(file_path)

        # Apply additional filters
        remover = vtk.vtkDecimatePro()
        remover.SetInputData(poly_data)
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

        self.volume_label.setText(f"Volume (m^3): {volume:.6f}")

        self.fit_camera_to_model()

        print(f"Model loaded: {file_path}")
        print(f"Bounds: [{self.model_bounds[0]:.4f}, {self.model_bounds[1]:.4f}], [{self.model_bounds[2]:.4f}, {self.model_bounds[3]:.4f}], [{self.model_bounds[4]:.4f}, {self.model_bounds[5]:.4f}]")

        # Render is handled by caller (e.g., load_stl_file)

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
            self.point_actors[index] = create_crosshair_marker(
                coords=self.point_coords[index],
                radius_scale=self.calculate_sphere_radius()
            )
            self.renderer.AddActor(self.point_actors[index])
        self.point_actors[index].VisibilityOn()
        self.update_point_display(index)

    def rotate_camera(self, angle, rotation_type):
        # Don't start new animation if one is already running
        if self.is_animating:
            return

        if self.animated_rotation.start_rotation(angle, rotation_type):
            self.target_rotation = (self.current_rotation + angle) % 360
            self.target_angle = angle  # Store target angle for precise completion
            self.is_animating = True  # Block further input
            self.animation_timer.start(1000 // 60)
            self.camera_rotation[self.rotation_types[rotation_type]] += angle
            self.camera_rotation[self.rotation_types[rotation_type]] %= 360

    def animate_rotation(self):
        # Delegate to utility class for rotation
        animation_continues = self.animated_rotation.animate_frame()

        # Render using offscreen rendering
        self.render_to_image()

        # Stop animation when utility class indicates completion
        if not animation_continues:
            self.animation_timer.stop()
            self.current_rotation = self.target_rotation
            self.is_animating = False  # Allow new input

    def quaternion_from_axis_angle(self, axis, angle_deg):
        """Create quaternion from axis and angle (in degrees)"""
        angle_rad = np.radians(angle_deg)
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Normalize
        w = np.cos(angle_rad / 2.0)
        x, y, z = axis * np.sin(angle_rad / 2.0)
        return np.array([w, x, y, z])

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees([roll, pitch, yaw])

    # euler_to_quaternion() and quaternion_to_matrix() are now imported from urdf_kitchen_utils

    def create_collider_actor(self):
        """Create VTK actor for the collider wireframe (lines only)"""
        collider_type = self.collider_type
        params = self.collider_params[collider_type]

        # Create assembly to hold all parts of the collider
        assembly = vtk.vtkAssembly()

        if collider_type == "box":
            # Box collider
            box = vtk.vtkCubeSource()
            box.SetXLength(params[0])
            box.SetYLength(params[1])
            box.SetZLength(params[2])
            box.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(box.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(2)
            assembly.AddPart(actor)

        elif collider_type == "sphere":
            # Sphere collider
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(params[0])
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            sphere.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(2)
            assembly.AddPart(actor)

        elif collider_type == "cylinder":
            # Cylinder collider
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(params[0])
            cylinder.SetHeight(params[1])
            cylinder.SetResolution(20)
            cylinder.Update()

            # Rotate to align with Z-axis (VTK cylinder is along Y by default)
            transform = vtk.vtkTransform()
            transform.RotateX(90)

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(cylinder.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(transform_filter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(2)
            assembly.AddPart(actor)

        elif collider_type == "capsule":
            # Capsule collider (cylinder + 2 hemispheres)
            radius = params[0]
            length = params[1]

            # Cylinder part
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(radius)
            cylinder.SetHeight(length)
            cylinder.SetResolution(20)
            cylinder.Update()

            # Rotate to align with Z-axis
            transform = vtk.vtkTransform()
            transform.RotateX(90)

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(cylinder.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(transform_filter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(2)
            assembly.AddPart(actor)

            # Top hemisphere
            top_sphere = vtk.vtkSphereSource()
            top_sphere.SetRadius(radius)
            top_sphere.SetThetaResolution(20)
            top_sphere.SetPhiResolution(20)
            top_sphere.SetStartPhi(0)
            top_sphere.SetEndPhi(90)
            top_sphere.Update()

            mapper_top = vtk.vtkPolyDataMapper()
            mapper_top.SetInputConnection(top_sphere.GetOutputPort())

            actor_top = vtk.vtkActor()
            actor_top.SetMapper(mapper_top)
            actor_top.SetPosition(0, 0, length / 2)
            actor_top.GetProperty().SetColor(0.0, 0.8, 1.0)
            actor_top.GetProperty().SetRepresentationToWireframe()
            actor_top.GetProperty().SetLineWidth(2)
            assembly.AddPart(actor_top)

            # Bottom hemisphere
            bottom_sphere = vtk.vtkSphereSource()
            bottom_sphere.SetRadius(radius)
            bottom_sphere.SetThetaResolution(20)
            bottom_sphere.SetPhiResolution(20)
            bottom_sphere.SetStartPhi(90)
            bottom_sphere.SetEndPhi(180)
            bottom_sphere.Update()

            mapper_bottom = vtk.vtkPolyDataMapper()
            mapper_bottom.SetInputConnection(bottom_sphere.GetOutputPort())

            actor_bottom = vtk.vtkActor()
            actor_bottom.SetMapper(mapper_bottom)
            actor_bottom.SetPosition(0, 0, -length / 2)
            actor_bottom.GetProperty().SetColor(0.0, 0.8, 1.0)
            actor_bottom.GetProperty().SetRepresentationToWireframe()
            actor_bottom.GetProperty().SetLineWidth(2)
            assembly.AddPart(actor_bottom)

        # Apply position
        assembly.SetPosition(self.collider_position)

        # Apply rotation using quaternion
        transform = vtk.vtkTransform()
        transform.PostMultiply()

        # Move to origin
        transform.Translate(*[-x for x in self.collider_position])

        # Apply rotation from quaternion
        rot_matrix = quaternion_to_matrix(self.collider_rotation_quaternion)
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                vtk_matrix.SetElement(i, j, rot_matrix[i, j])
        transform.Concatenate(vtk_matrix)

        # Move back to position
        transform.Translate(*self.collider_position)

        assembly.SetUserTransform(transform)

        return assembly

    def create_collider_surface_actor(self):
        """Create VTK actor for the collider surface (transparent, for blinking effect)"""
        collider_type = self.collider_type
        params = self.collider_params[collider_type]

        # Create assembly to hold all parts of the collider surface
        assembly = vtk.vtkAssembly()

        if collider_type == "box":
            # Box collider
            box = vtk.vtkCubeSource()
            box.SetXLength(params[0])
            box.SetYLength(params[1])
            box.SetZLength(params[2])
            box.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(box.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetOpacity(0.5)  # 50% transparency
            assembly.AddPart(actor)

        elif collider_type == "sphere":
            # Sphere collider
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(params[0])
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            sphere.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetOpacity(0.5)  # 50% transparency
            assembly.AddPart(actor)

        elif collider_type == "cylinder":
            # Cylinder collider
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(params[0])
            cylinder.SetHeight(params[1])
            cylinder.SetResolution(20)
            cylinder.Update()

            # Rotate to align with Z-axis (VTK cylinder is along Y by default)
            transform = vtk.vtkTransform()
            transform.RotateX(90)

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(cylinder.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(transform_filter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetOpacity(0.5)  # 50% transparency
            assembly.AddPart(actor)

        elif collider_type == "capsule":
            # Capsule collider (cylinder + 2 hemispheres)
            radius = params[0]
            length = params[1]

            # Cylinder part
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(radius)
            cylinder.SetHeight(length)
            cylinder.SetResolution(20)
            cylinder.Update()

            # Rotate to align with Z-axis
            transform = vtk.vtkTransform()
            transform.RotateX(90)

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(cylinder.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(transform_filter.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # Light blue
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetOpacity(0.5)  # 50% transparency
            assembly.AddPart(actor)

            # Top hemisphere
            top_sphere = vtk.vtkSphereSource()
            top_sphere.SetRadius(radius)
            top_sphere.SetThetaResolution(20)
            top_sphere.SetPhiResolution(20)
            top_sphere.SetStartPhi(0)
            top_sphere.SetEndPhi(90)
            top_sphere.Update()

            mapper_top = vtk.vtkPolyDataMapper()
            mapper_top.SetInputConnection(top_sphere.GetOutputPort())

            actor_top = vtk.vtkActor()
            actor_top.SetMapper(mapper_top)
            actor_top.SetPosition(0, 0, length / 2)
            actor_top.GetProperty().SetColor(0.0, 0.8, 1.0)
            actor_top.GetProperty().SetRepresentationToSurface()
            actor_top.GetProperty().SetOpacity(0.5)  # 50% transparency
            assembly.AddPart(actor_top)

            # Bottom hemisphere
            bottom_sphere = vtk.vtkSphereSource()
            bottom_sphere.SetRadius(radius)
            bottom_sphere.SetThetaResolution(20)
            bottom_sphere.SetPhiResolution(20)
            bottom_sphere.SetStartPhi(90)
            bottom_sphere.SetEndPhi(180)
            bottom_sphere.Update()

            mapper_bottom = vtk.vtkPolyDataMapper()
            mapper_bottom.SetInputConnection(bottom_sphere.GetOutputPort())

            actor_bottom = vtk.vtkActor()
            actor_bottom.SetMapper(mapper_bottom)
            actor_bottom.SetPosition(0, 0, -length / 2)
            actor_bottom.GetProperty().SetColor(0.0, 0.8, 1.0)
            actor_bottom.GetProperty().SetRepresentationToSurface()
            actor_bottom.GetProperty().SetOpacity(0.5)  # 50% transparency
            assembly.AddPart(actor_bottom)

        # Apply position
        assembly.SetPosition(self.collider_position)

        # Apply rotation using quaternion
        transform = vtk.vtkTransform()
        transform.PostMultiply()

        # Move to origin
        transform.Translate(*[-x for x in self.collider_position])

        # Apply rotation from quaternion
        rot_matrix = quaternion_to_matrix(self.collider_rotation_quaternion)
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                vtk_matrix.SetElement(i, j, rot_matrix[i, j])
        transform.Concatenate(vtk_matrix)

        # Move back to position
        transform.Translate(*self.collider_position)

        assembly.SetUserTransform(transform)

        return assembly

    def update_collider_display(self):
        """Update collider display in the 3D view"""
        # Remove old collider actors if exist
        if self.collider_actor:
            self.renderer.RemoveActor(self.collider_actor)
            self.collider_actor = None
        if self.collider_surface_actor:
            self.renderer.RemoveActor(self.collider_surface_actor)
            self.collider_surface_actor = None

        # Stop blinking timer (no longer needed)
        if self.collider_blink_timer.isActive():
            self.collider_blink_timer.stop()

        # Create and add new wireframe actor (always visible)
        self.collider_actor = self.create_collider_actor()
        self.renderer.AddActor(self.collider_actor)

        # Create and add new surface actor (always visible with 50% transparency)
        self.collider_surface_actor = self.create_collider_surface_actor()
        self.renderer.AddActor(self.collider_surface_actor)

        # Render the updated scene
        self.render_to_image()

    def toggle_collider_surface(self):
        """Toggle collider surface visibility (deprecated - no longer used)"""
        # This method is no longer used since blinking is disabled
        pass

    def reset_and_fit_collider(self):
        """Reset collider to default state (all zeros) and then fit to mesh"""
        if not self.model_bounds:
            print("No mesh loaded. Please load a mesh first.")
            return

        # Reset position to origin
        self.collider_position = [0.0, 0.0, 0.0]

        # Reset rotation to zero (identity quaternion) - radians internal
        self.collider_rotation = [0.0, 0.0, 0.0]
        self.collider_rotation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Reset parameters to default values
        self.collider_params = {
            "box": [1.0, 1.0, 1.0],
            "sphere": [0.5],
            "cylinder": [0.5, 1.0],
            "capsule": [0.5, 1.0]
        }

        # Update UI with reset values (degrees for display)
        for i in range(3):
            self.collider_position_inputs[i].setText("0.0000")
            self.collider_rotation_inputs[i].setText("0.00")

        # Now perform Rough Fit with reset rotation (identity)
        self.draft_collider()

    def draft_collider(self):
        """Draft a collider from mesh bounding box based on selected type"""
        if not self.model_bounds:
            print("No mesh loaded. Please load a mesh first.")
            return

        # Auto-enable ShowCollider when RoughFit is pressed
        if not self.collider_show:
            self.collider_show_checkbox.setChecked(True)
            print("ShowCollider automatically enabled")

        # Calculate bounding box dimensions (world space, round to 4 decimal places)
        size_x = round(self.model_bounds[1] - self.model_bounds[0], 4)
        size_y = round(self.model_bounds[3] - self.model_bounds[2], 4)
        size_z = round(self.model_bounds[5] - self.model_bounds[4], 4)

        # Calculate center position (round to 4 decimal places)
        center_x = round((self.model_bounds[0] + self.model_bounds[1]) / 2, 4)
        center_y = round((self.model_bounds[2] + self.model_bounds[3]) / 2, 4)
        center_z = round((self.model_bounds[4] + self.model_bounds[5]) / 2, 4)

        # Set position (keep existing rotation)
        self.collider_position = [center_x, center_y, center_z]

        # Calculate parameters based on current collider type
        collider_type = self.collider_type

        if collider_type == "box":
            # Box: use full bounding box dimensions
            self.collider_params["box"] = [size_x, size_y, size_z]
            print(f"Drafted box collider: size=({size_x:.4f}, {size_y:.4f}, {size_z:.4f}), center=({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")

        elif collider_type == "sphere":
            # Sphere: radius that fits inside mesh bounding box (use smallest dimension)
            radius = round(min(size_x, size_y, size_z) / 2, 4)
            self.collider_params["sphere"] = [radius]
            print(f"Drafted sphere collider: radius={radius:.4f}, center=({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")

        elif collider_type in ["cylinder", "capsule"]:
            # For cylinder/capsule, consider rotation to find which world axis aligns with collider's Z-axis
            rot_matrix = quaternion_to_matrix(self.collider_rotation_quaternion)
            collider_z_axis = rot_matrix[:, 2]  # Third column is Z-axis direction in world space

            # Find which world axis is most aligned with collider's Z-axis
            world_axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            mesh_sizes = [size_x, size_y, size_z]

            # Calculate dot products to find most aligned axis
            alignments = [abs(np.dot(collider_z_axis, axis)) for axis in world_axes]
            primary_axis_idx = np.argmax(alignments)

            # Length comes from the most aligned axis
            primary_size = mesh_sizes[primary_axis_idx]

            # Radius comes from the smaller of the two perpendicular axes
            perpendicular_sizes = [mesh_sizes[i] for i in range(3) if i != primary_axis_idx]
            perpendicular_size = min(perpendicular_sizes)

            if collider_type == "cylinder":
                # Cylinder: radius from perpendicular plane, height from aligned axis
                radius = round(perpendicular_size / 2, 4)
                height = round(primary_size, 4)
                self.collider_params["cylinder"] = [radius, height]
                print(f"Drafted cylinder collider: radius={radius:.4f}, length={height:.4f}, center=({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")

            else:  # capsule
                # Capsule: radius from perpendicular plane, length from aligned axis minus hemispheres
                radius = round(perpendicular_size / 2, 4)
                length = round(primary_size - (2 * radius), 4)
                if length < 0:
                    length = 0.0  # Ensure non-negative length
                self.collider_params["capsule"] = [radius, length]
                print(f"Drafted capsule collider: radius={radius:.4f}, length={length:.4f} (cylinder only), center=({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")

        # Update UI (position and parameters only, keep existing rotation)
        self.update_collider_param_inputs()
        for i, value in enumerate([center_x, center_y, center_z]):
            self.collider_position_inputs[i].setText(f"{value:.4f}")

        # Update display
        self.update_collider_display()

    def export_collider(self):
        """Export collider to XML file"""
        if not self.current_stl_path:
            print("No mesh loaded. Please load a mesh first.")
            return

        # Generate default file path (mesh_filename + _collider.xml)
        import os
        base_name = os.path.splitext(self.current_stl_path)[0]
        default_path = base_name + "_collider.xml"

        # Ask user for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Collider", default_path, "XML Files (*.xml)")
        if not file_path:
            return

        try:
            # Create XML content
            import xml.etree.ElementTree as ET
            from xml.dom import minidom

            root = ET.Element("urdf_kitchen_collider")
            root.set("version", "1.0")

            # Add mesh reference
            mesh_elem = ET.SubElement(root, "mesh_file")
            mesh_elem.text = os.path.basename(self.current_stl_path)

            # Add collider info
            collider_elem = ET.SubElement(root, "collider")
            collider_elem.set("type", self.collider_type)

            # Add parameters (4 decimal places for size/radius/length)
            params_elem = ET.SubElement(collider_elem, "geometry")
            params = self.collider_params[self.collider_type]

            if self.collider_type == "box":
                params_elem.set("size_x", f"{round(params[0], 4):.4f}")
                params_elem.set("size_y", f"{round(params[1], 4):.4f}")
                params_elem.set("size_z", f"{round(params[2], 4):.4f}")
            elif self.collider_type == "sphere":
                params_elem.set("radius", f"{round(params[0], 4):.4f}")
            elif self.collider_type in ["cylinder", "capsule"]:
                params_elem.set("radius", f"{round(params[0], 4):.4f}")
                params_elem.set("length", f"{round(params[1], 4):.4f}")

            # Add position (4 decimal places)
            position_elem = ET.SubElement(collider_elem, "position")
            position_elem.set("x", f"{round(self.collider_position[0], 4):.4f}")
            position_elem.set("y", f"{round(self.collider_position[1], 4):.4f}")
            position_elem.set("z", f"{round(self.collider_position[2], 4):.4f}")

            # Add rotation (radians, 6 decimal places)
            rotation_elem = ET.SubElement(collider_elem, "rotation")
            rotation_elem.set("roll", f"{self.collider_rotation[0]:.6f}")
            rotation_elem.set("pitch", f"{self.collider_rotation[1]:.6f}")
            rotation_elem.set("yaw", f"{self.collider_rotation[2]:.6f}")

            # Pretty print XML
            xml_str = ET.tostring(root, encoding='unicode')
            dom = minidom.parseString(xml_str)
            pretty_xml = dom.toprettyxml(indent="  ")

            # Write to file
            with open(file_path, 'w') as f:
                f.write(pretty_xml)

            print(f"Collider exported to: {file_path}")

        except Exception as e:
            print(f"Error exporting collider: {e}")
            import traceback
            traceback.print_exc()

    def load_collider(self):
        """Load collider from XML file"""
        # Ask user for file location
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Collider", "", "XML Files (*.xml)")
        if not file_path:
            return

        try:
            import xml.etree.ElementTree as ET

            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Verify it's a urdf_kitchen_collider file
            if root.tag != "urdf_kitchen_collider":
                print(f"Error: Not a valid urdf_kitchen_collider file")
                return

            # Get collider element
            collider_elem = root.find("collider")
            if collider_elem is None:
                print(f"Error: No collider element found")
                return

            # Get collider type
            collider_type = collider_elem.get("type")
            if collider_type not in ["box", "sphere", "cylinder", "capsule"]:
                print(f"Error: Unknown collider type: {collider_type}")
                return

            # Set collider type in combo box
            index = self.collider_type_combo.findText(collider_type)
            if index >= 0:
                self.collider_type_combo.setCurrentIndex(index)

            # Get geometry parameters
            geometry_elem = collider_elem.find("geometry")
            if geometry_elem is not None:
                if collider_type == "box":
                    size_x = float(geometry_elem.get("size_x", "1.0"))
                    size_y = float(geometry_elem.get("size_y", "1.0"))
                    size_z = float(geometry_elem.get("size_z", "1.0"))
                    self.collider_params[collider_type] = [size_x, size_y, size_z]
                elif collider_type == "sphere":
                    radius = float(geometry_elem.get("radius", "0.5"))
                    self.collider_params[collider_type] = [radius]
                elif collider_type in ["cylinder", "capsule"]:
                    radius = float(geometry_elem.get("radius", "0.5"))
                    length = float(geometry_elem.get("length", "1.0"))
                    self.collider_params[collider_type] = [radius, length]

            # Get position
            position_elem = collider_elem.find("position")
            if position_elem is not None:
                x = float(position_elem.get("x", "0.0"))
                y = float(position_elem.get("y", "0.0"))
                z = float(position_elem.get("z", "0.0"))
                self.collider_position = [x, y, z]

                # Update position input fields
                self.collider_position_inputs[0].setText(f"{x:.4f}")
                self.collider_position_inputs[1].setText(f"{y:.4f}")
                self.collider_position_inputs[2].setText(f"{z:.4f}")

            # Get rotation (radians in file)
            rotation_elem = collider_elem.find("rotation")
            if rotation_elem is not None:
                import math
                roll_rad = float(rotation_elem.get("roll", "0.0"))
                pitch_rad = float(rotation_elem.get("pitch", "0.0"))
                yaw_rad = float(rotation_elem.get("yaw", "0.0"))
                # Backward compatibility: if values look like degrees, convert to radians
                if any(abs(v) > 3.5 for v in [roll_rad, pitch_rad, yaw_rad]):
                    roll_rad = math.radians(roll_rad)
                    pitch_rad = math.radians(pitch_rad)
                    yaw_rad = math.radians(yaw_rad)
                self.collider_rotation = [roll_rad, pitch_rad, yaw_rad]

                # Update rotation input fields (degrees for UI)
                roll_deg = math.degrees(roll_rad)
                pitch_deg = math.degrees(pitch_rad)
                yaw_deg = math.degrees(yaw_rad)
                self.collider_rotation_inputs[0].setText(f"{roll_deg:.2f}")
                self.collider_rotation_inputs[1].setText(f"{pitch_deg:.2f}")
                self.collider_rotation_inputs[2].setText(f"{yaw_deg:.2f}")

                # Convert RPY to quaternion for internal use (degrees for helper)
                self.collider_rotation_quaternion = euler_to_quaternion(
                    roll_deg,
                    pitch_deg,
                    yaw_deg
                )

            # Update parameter input fields
            self.update_collider_param_inputs()

            # Update collider visualization
            self.update_collider_display()

            print(f"Collider loaded from: {file_path}")
            print(f"  Type: {collider_type}")
            print(f"  Position: {self.collider_position}")
            print(f"  Rotation (rad): {self.collider_rotation}")

        except Exception as e:
            print(f"Error loading collider: {e}")
            import traceback
            traceback.print_exc()

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

        # Tab key: cycle through Position, Size/Radius, Rotation checkboxes
        if key == Qt.Key_Tab:
            event.accept()

            # Only work if ShowCollider is checked
            if self.collider_show:
                # Determine available options based on collider type
                if self.collider_type == "box":
                    # Box: Position → Size → Rotation → Position...
                    available_options = ['position', 'size', 'rotation']
                else:  # sphere, cylinder, capsule
                    # Others: Position → Radius&Length → Rotation → Position...
                    available_options = ['position', 'radius_length', 'rotation']

                # Find current active option
                current_active = None
                if self.collider_position_active:
                    current_active = 'position'
                elif self.collider_size_active:
                    current_active = 'size'
                elif self.collider_radius_length_active:
                    current_active = 'radius_length'
                elif self.collider_rotation_active:
                    current_active = 'rotation'

                # If any checkbox is active, cycle to the next one
                if current_active and current_active in available_options:
                    current_index = available_options.index(current_active)
                    next_index = (current_index + 1) % len(available_options)
                    next_option = available_options[next_index]

                    # Activate the next checkbox (this will trigger the mutual exclusivity logic)
                    if next_option == 'position':
                        self.collider_position_checkbox.setChecked(True)
                    elif next_option == 'size':
                        # Find the size checkbox in collider_param_checkboxes
                        if hasattr(self, 'collider_param_checkboxes') and self.collider_param_checkboxes[0]:
                            self.collider_param_checkboxes[0].setChecked(True)
                    elif next_option == 'radius_length':
                        # Find the radius/length checkbox in collider_param_checkboxes
                        if hasattr(self, 'collider_param_checkboxes') and self.collider_param_checkboxes[0]:
                            self.collider_param_checkboxes[0].setChecked(True)
                    elif next_option == 'rotation':
                        self.collider_rotation_checkbox.setChecked(True)

            return

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

        elif key == Qt.Key_H:
            # Toggle help text visibility
            self.help_visible = not self.help_visible
            self.update_instruction_text_layout()
            self.render_to_image()

        elif key in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            step = calculate_arrow_key_step(shift_pressed, ctrl_pressed)
            handled = False

            # Priority 1: Collider Rotation (left/right only, roll direction from camera view)
            if self.collider_rotation_active and key in [Qt.Key_Left, Qt.Key_Right]:
                # Step sizes: 10 degrees (default), 1 degree (shift), 0.1 degree (shift+ctrl)
                angle_step = 10.0  # Default: 10 degrees
                if shift_pressed and not ctrl_pressed:
                    angle_step = 1.0  # Shift: 1 degree
                elif shift_pressed and ctrl_pressed:
                    angle_step = 0.1  # Shift+Ctrl: 0.1 degree

                # Get camera view direction (axis of rotation)
                camera = self.renderer.GetActiveCamera()
                camera_pos = np.array(camera.GetPosition())
                focal_point = np.array(camera.GetFocalPoint())
                view_direction = focal_point - camera_pos
                view_direction = view_direction / np.linalg.norm(view_direction)  # Normalize

                # Determine rotation direction
                if key == Qt.Key_Left:
                    angle_step = -angle_step  # Counter-clockwise from camera view

                # Create rotation quaternion around camera view direction
                delta_quat = self.quaternion_from_axis_angle(view_direction, angle_step)

                # Apply rotation to current quaternion
                self.collider_rotation_quaternion = self.quaternion_multiply(
                    delta_quat, self.collider_rotation_quaternion)

                # Normalize quaternion to prevent drift
                self.collider_rotation_quaternion /= np.linalg.norm(self.collider_rotation_quaternion)

                # Update UI with Euler angles (for reference)
                euler = quaternion_to_euler(self.collider_rotation_quaternion)
                for i in range(3):
                    self.collider_rotation[i] = euler[i]
                    self.collider_rotation_inputs[i].setText(f"{euler[i]:.2f}")

                if self.collider_show:
                    self.update_collider_display()
                handled = True

            # Priority 2: Collider Size (for box, camera-relative)
            elif self.collider_size_active:
                param_step = 0.01 if not shift_pressed else 0.001
                if ctrl_pressed and shift_pressed:
                    param_step = 0.0001

                # Get camera screen axes
                horizontal_axis, vertical_axis, screen_right, screen_up = self.get_screen_axes()

                # Determine which box axis is closest to screen_right and screen_up
                axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]  # X, Y, Z

                # Find axis most aligned with screen_right (for left/right keys)
                right_dots = [abs(np.dot(screen_right, axis)) for axis in axes]
                right_axis_idx = np.argmax(right_dots)

                # Find axis most aligned with screen_up (for up/down keys)
                up_dots = [abs(np.dot(screen_up, axis)) for axis in axes]
                up_axis_idx = np.argmax(up_dots)

                # Left/Right: change size along screen_right direction
                if key in [Qt.Key_Left, Qt.Key_Right]:
                    current_size = self.collider_params[self.collider_type][right_axis_idx]
                    if key == Qt.Key_Left:
                        current_size -= param_step
                    elif key == Qt.Key_Right:
                        current_size += param_step
                    current_size = max(0.001, current_size)
                    self.collider_params[self.collider_type][right_axis_idx] = current_size
                    self.collider_param_inputs[right_axis_idx].setText(f"{current_size:.4f}")
                    handled = True

                # Up/Down: change size along screen_up direction
                if key in [Qt.Key_Up, Qt.Key_Down]:
                    current_size = self.collider_params[self.collider_type][up_axis_idx]
                    if key == Qt.Key_Up:
                        current_size += param_step
                    elif key == Qt.Key_Down:
                        current_size -= param_step
                    current_size = max(0.001, current_size)
                    self.collider_params[self.collider_type][up_axis_idx] = current_size
                    self.collider_param_inputs[up_axis_idx].setText(f"{current_size:.4f}")
                    handled = True

                if handled and self.collider_show:
                    self.update_collider_display()

            # Priority 3: Collider Radius/Length (shared checkbox)
            elif self.collider_radius_length_active:
                param_step = 0.01 if not shift_pressed else 0.001
                if ctrl_pressed and shift_pressed:
                    param_step = 0.0001

                # Radius: left/right
                if key in [Qt.Key_Left, Qt.Key_Right]:
                    radius_idx = 0
                    current_radius = self.collider_params[self.collider_type][radius_idx]
                    if key == Qt.Key_Left:
                        current_radius -= param_step
                    elif key == Qt.Key_Right:
                        current_radius += param_step
                    current_radius = max(0.001, current_radius)  # Minimum radius
                    self.collider_params[self.collider_type][radius_idx] = current_radius
                    self.collider_param_inputs[radius_idx].setText(f"{current_radius:.4f}")
                    if self.collider_show:
                        self.update_collider_display()
                    handled = True

                # Length: up/down (only for cylinder/capsule)
                if self.collider_type in ["cylinder", "capsule"] and key in [Qt.Key_Up, Qt.Key_Down]:
                    length_idx = 1
                    current_length = self.collider_params[self.collider_type][length_idx]
                    if key == Qt.Key_Up:
                        current_length += param_step
                    elif key == Qt.Key_Down:
                        current_length -= param_step
                    current_length = max(0.001, current_length)  # Minimum length
                    self.collider_params[self.collider_type][length_idx] = current_length
                    self.collider_param_inputs[length_idx].setText(f"{current_length:.4f}")
                    if self.collider_show:
                        self.update_collider_display()
                    handled = True

            # Priority 4: Collider Position
            elif self.collider_position_active:
                horizontal_axis, vertical_axis, screen_right, screen_up = self.get_screen_axes()

                if key == Qt.Key_Up:
                    self.collider_position[0] += screen_up[0] * step
                    self.collider_position[1] += screen_up[1] * step
                    self.collider_position[2] += screen_up[2] * step
                elif key == Qt.Key_Down:
                    self.collider_position[0] -= screen_up[0] * step
                    self.collider_position[1] -= screen_up[1] * step
                    self.collider_position[2] -= screen_up[2] * step
                elif key == Qt.Key_Left:
                    self.collider_position[0] -= screen_right[0] * step
                    self.collider_position[1] -= screen_right[1] * step
                    self.collider_position[2] -= screen_right[2] * step
                elif key == Qt.Key_Right:
                    self.collider_position[0] += screen_right[0] * step
                    self.collider_position[1] += screen_right[1] * step
                    self.collider_position[2] += screen_right[2] * step

                # Update UI
                for i in range(3):
                    self.collider_position_inputs[i].setText(f"{self.collider_position[i]:.4f}")
                if self.collider_show:
                    self.update_collider_display()
                handled = True

            # Priority 5: Center Position (default behavior)
            if not handled:
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
                        handled = True

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
                        self.last_mouse_pos = event.position().toPoint()
                        # CRITICAL: Grab mouse to receive events even outside widget
                        self.vtk_display.grabMouse()
                        return True
                    # Handle middle button (wheel button) for pan
                    elif event.button() == QtCore.MiddleButton:
                        self.vtk_display.setFocus()
                        self.middle_mouse_pressed = True
                        self.last_mouse_pos = event.position().toPoint()
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
                        current_pos = event.position().toPoint()
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
                        current_pos = event.position().toPoint()
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

    def move_point_screen(self, point_index, screen_axis, distance):
        """Move a point along a screen axis by a given distance"""
        if point_index >= len(self.point_coords):
            return

        # Get current point coordinates
        current = np.array(self.point_coords[point_index])

        # Move along the screen axis
        new_coords = current + np.array(screen_axis) * distance

        # Update point coordinates
        self.point_coords[point_index] = new_coords.tolist()

        # Update display
        self.update_point_display(point_index)

        # If point is visible, update its position
        if self.point_checkboxes[point_index].isChecked():
            self.show_point(point_index)

        print(f"Point {point_index+1} moved to: ({new_coords[0]:.6f}, {new_coords[1]:.6f}, {new_coords[2]:.6f})")

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

    def resizeEvent(self, event):
        """Handle window resize to update text layout and re-render"""
        super().resizeEvent(event)
        # Debounce resize events to avoid excessive re-rendering
        # Update text layout and trigger re-render after a short delay
        if hasattr(self, 'vtk_fully_ready') and self.vtk_fully_ready:
            QTimer.singleShot(0, self._handle_resize)

    def _handle_resize(self):
        """Internal handler for resize event (debounced)"""
        try:
            # Update instruction text font size based on new window size
            self.update_instruction_text_layout()
            # Re-render with new RenderWindow size
            self.render_to_image()
        except Exception as e:
            pass  # Silently ignore resize errors during initialization

    def get_screen_axes(self):
        return self.camera_controller.get_screen_axes()

    def export_stl_with_new_origin(self):
        if not self.stl_actor or not any(self.point_actors):
            print("3D model or points are not set.")
            return

        # Use the loaded file path as default (same directory and filename)
        default_path = self.current_stl_path if self.current_stl_path else ""

        # Use common utility function for file filter
        file_filter = get_mesh_file_filter(TRIMESH_AVAILABLE)

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

            # Get transformed polydata
            output_polydata = transform_filter.GetOutput()

            # Apply cleaning if checkbox is checked
            if self.reorient_clean_mesh_checkbox.isChecked():
                # Use conservative cleaning for OBJ input to preserve vertex normals
                import os
                input_ext = os.path.splitext(self.current_stl_path)[1].lower() if self.current_stl_path else ""
                if input_ext == ".obj":
                    output_polydata = self.clean_polydata_conservative(output_polydata)
                    print("Mesh cleaning applied (conservative mode for OBJ)")
                else:
                    output_polydata = self.clean_polydata(output_polydata)
                    print("Mesh cleaning applied (normals fixed, duplicates removed)")

            # Use common utility function to save mesh
            save_polydata_to_mesh(file_path, output_polydata)

            print(f"Mesh file has been saved: {file_path}")
            # Update current path for next save
            self.current_stl_path = file_path

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def handle_set_front_as_x(self):
        button_text = self.sender().text()
        if button_text == "Set Front as X":
            self.transform_stl_to_camera_view()

    def handle_set_reset(self):
        sender = self.sender()
        button_text = sender.text()

        if button_text == "Set Front as X":
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

# signal_handler moved to urdf_kitchen_utils.py
# Now using setup_signal_handlers()


if __name__ == "__main__":
    # M4 Mac (Apple Silicon) detection and setup (using utils)
    if IS_APPLE_SILICON:
        setup_qt_environment_for_apple_silicon()

    # Set Qt attributes BEFORE creating QApplication
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    # Create QApplication
    app = QApplication(sys.argv)

    # Ctrl+Cのシグナルハンドラを設定（utils関数使用）
    setup_signal_handlers(verbose=False)

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

        # コマンドライン引数でファイルが指定された場合は、ウィンドウを最前面に表示
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            if os.path.exists(file_path):
                print(f"File specified from command line: {file_path}")
                print("Will load after VTK initialization completes...")
                window.pending_file_to_load = file_path
                # Set window to stay on top temporarily when launched from Assembler
                window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)
            else:
                print(f"Warning: File not found: {file_path}")

        # Show window
        window.show()
        window.raise_()
        window.activateWindow()

    except Exception as e:
        print(f"Error creating window: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # シグナル処理用タイマー（utils関数使用）
    timer = setup_signal_processing_timer(app)

    sys.exit(app.exec())
