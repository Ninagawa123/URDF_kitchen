"""
File Name: urdf_kitchen_PartsEditor.py
Description: A Python script for configuring connection points of parts for urdf_kitchen_Assembler.py.

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
    """シックなダークテーマを適用

    Note: Base theme available in urdf_kitchen_utils.setup_dark_theme(app, 'parts_editor')
    This function extends the base with additional custom widget styling.
    """
    # パレットの設定
    palette = app.palette()
    # メインウィンドウ背景：柔らかいダークグレー
    palette.setColor(QPalette.Window, QColor(70, 80, 80))
    # テキスト：ダークグレー
    palette.setColor(QPalette.WindowText, QColor(240, 240, 237))
    # 入力フィールド背景：オフホワイト
    palette.setColor(QPalette.Base, QColor(240, 240, 237))
    palette.setColor(QPalette.AlternateBase, QColor(230, 230, 227))
    # ツールチップ
    palette.setColor(QPalette.ToolTipBase, QColor(240, 240, 237))
    palette.setColor(QPalette.ToolTipText, QColor(51, 51, 51))
    # 通常のテキスト：ダークグレー
    palette.setColor(QPalette.Text, QColor(51, 51, 51))
    # ボタン
    palette.setColor(QPalette.Button, QColor(240, 240, 237))
    palette.setColor(QPalette.ButtonText, QColor(51, 51, 51))
    # 選択時のハイライト
    palette.setColor(QPalette.Highlight, QColor(150, 150, 150))
    palette.setColor(QPalette.HighlightedText, QColor(240, 240, 237))
    app.setPalette(palette)

    # 追加のスタイル設定
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
        
        step = 0.01  # デフォルトのステップ (10mm)
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
        self.camera_rotation = [0, 0, 0]  # [yaw, pitch, roll]
        self.absolute_origin = [0, 0, 0]  # 大原点の設定
        self.initial_camera_position = [10, 0, 0]  # 初期カメラ位置
        self.initial_camera_focal_point = [0, 0, 0]  # 初期焦点
        self.initial_camera_view_up = [0, 0, 1]  # 初期の上方向

        self.num_points = 8  # ポイントの数を8に設定
        self.point_coords = [list(self.absolute_origin) for _ in range(self.num_points)]
        self.point_angles = [[0.0, 0.0, 0.0] for _ in range(self.num_points)]  # Body angles for each point (radians)
        self.point_actors = [None] * self.num_points
        self.point_checkboxes = []
        self.point_inputs = []

        self.com_coords = [0.0, 0.0, 0.0]  # Center of Mass座標
        self.com_sphere_actor = None  # 赤い球（チェックなし時）
        self.com_cursor_actor = None  # 十字付き円（チェックあり時、赤色）

        # 色の手動変更フラグ（Falseの場合は元の色を保持）
        self.color_manually_changed = False
        self.mesh_color = None  # メッシュの色情報

        # コマンドライン引数から渡されたファイルパスを保持
        self.pending_stl_file = None

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)  # 垂直方向のレイアウトに変更

        # ファイル名表示用のラベル
        self.file_name_label = QLabel("File: No file loaded")
        main_layout.addWidget(self.file_name_label)

        # 水平方向のレイアウトを作成（左側のUIと右側の3D表示用）
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # 左側のUI用ウィジェットとレイアウト
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        content_layout.addWidget(left_widget, 1)  # stretch factorを1に設定

        # 右側のVTK表示用QLabelウィジェット（Mac互換性のためオフスクリーンレンダリング使用）
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
        content_layout.addWidget(self.vtk_display, 4)  # stretch factorを4に設定（UIより広いスペースを確保）

        self.setup_ui()

        # VTK初期化フラグ（StlDaeSourcerと同じパターン）
        self.vtk_initialized = False

        # Rendering lock to prevent re-entry
        self._is_rendering = False
        self._render_counter = 0

        self.model_bounds = None
        self.stl_actor = None
        self.current_rotation = 0

        self.stl_center = list(self.absolute_origin)  # STLモデルの中心を大原点に初期化

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
            print(f"Received IPC request: {data[:100]}...")  # 長いメッセージの場合は最初の100文字のみ表示

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
                # JSON形式でコライダーデータを含むメッセージ
                import json
                json_str = data[10:].strip()  # "LOAD_JSON:"の後の部分
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

            # コライダー情報がある場合、コライダーXMLファイルを確認
            if collider_info:
                if collider_info.get('type') == 'primitive' and collider_info.get('xml_path'):
                    collider_xml_path = collider_info['xml_path']
                    if os.path.exists(collider_xml_path):
                        print(f"Collider XML file created by Assembler: {collider_xml_path}")
                        print(f"  This file will be available when saving from PartsEditor")
                        # コライダーXMLファイルは既にSTLファイルと同じディレクトリに作成されているので、
                        # PartsEditorが保存時にこのファイルを認識できる
                    else:
                        print(f"Warning: Collider XML file not found: {collider_xml_path}")
                elif collider_info.get('type') == 'mesh' and collider_info.get('mesh_path'):
                    print(f"Collider mesh specified: {os.path.basename(collider_info['mesh_path'])}")
                    # メッシュコライダーの場合は、PartsEditorでは直接処理しない
                    # （Assemblerで管理される）
            
            # コライダーXMLファイルが存在する場合、自動的に読み込む（Assemblerから作成されたもの）
            # STLファイルと同じディレクトリに_collider.xmlがあるか確認
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
        # VTKの初期化は最初のshowEvent時に遅延実行（StlDaeSourcerパターン）
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

            # コマンドライン引数からのファイルロードがある場合、VTK初期化後にロード
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
        button_layout = QVBoxLayout()  # メインの垂直レイアウト
        button_layout.setSpacing(10)  # ボタン間の間隔を5ピクセルに設定（デフォルトは約10）

        # first_row をラップするための QWidget を作成
        first_row_widget = QWidget()
        first_row_layout = QHBoxLayout()
        first_row_layout.setContentsMargins(0, 0, 0, 0)  # マージンを0に設定
        first_row_layout.setSpacing(5)  # ボタン間のスペーシングを5ピクセルに設定

        # # 最初の行の水平レイアウト
        first_row = QHBoxLayout()
        
        # # 最初の行を追加
        button_layout.addLayout(first_row)

        # Import Meshボタン
        self.load_button = QPushButton("Import Mesh")
        self.load_button.clicked.connect(self.load_stl_file)
        first_row.addWidget(self.load_button)

        # Load XMLボタン
        self.load_xml_button = QPushButton("Load XML")
        self.load_xml_button.clicked.connect(self.load_xml_file)
        first_row.addWidget(self.load_xml_button)

        # Reloadボタン
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_files)
        first_row.addWidget(self.reload_button)

        # first_row_layout を first_row_widget にセット
        first_row_widget.setLayout(first_row_layout)

        # Import Mesh with XMLボタン
        self.load_stl_xml_button = QPushButton("Import Mesh with XML")
        self.load_stl_xml_button.clicked.connect(self.load_stl_with_xml)
        button_layout.addWidget(self.load_stl_xml_button)

        # 間隔を1.5倍にし、中央に罫線を配置
        # 上部スペース
        spacer_top = QWidget()
        #spacer_top.setFixedHeight(4)  # 元の間隔の半分程度
        button_layout.addWidget(spacer_top)

        # 罫線
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #707070; }")
        button_layout.addWidget(separator)

        # 下部スペース
        spacer_bottom = QWidget()
        #spacer_bottom.setFixedHeight(4)  # 元の間隔の半分程度
        button_layout.addWidget(spacer_bottom)

        # MeshSourcer ボタン
        self.mesh_sourcer_button = QPushButton("MeshSourcer")
        self.mesh_sourcer_button.clicked.connect(self.open_mesh_sourcer)
        button_layout.addWidget(self.mesh_sourcer_button)

        self.left_layout.addLayout(button_layout)

    def open_mesh_sourcer(self):
        """現在のメッシュファイルを読み込んだ状態でMeshSourcerを開く"""
        import subprocess

        # 現在開いているメッシュファイルのパスを確認
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

        # MeshSourcerのスクリプトパスを取得
        mesh_sourcer_path = os.path.join(os.path.dirname(__file__), "urdf_kitchen_MeshSourcer.py")

        if not os.path.exists(mesh_sourcer_path):
            QMessageBox.warning(
                self,
                "MeshSourcer Not Found",
                f"MeshSourcer script not found:\n{mesh_sourcer_path}"
            )
            return

        try:
            # MeshSourcerを起動し、現在のメッシュファイルをコマンドライン引数として渡す
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

        # チェックボックスの列の最小幅を設定
        grid_layout.setColumnMinimumWidth(0, 15)  # この行を追加

        # プロパティの設定（Volume, Density, Mass）
        properties = [
            ("Volume (m^3):", "volume"),
            ("Density (kg/m^3):", "density"),
            ("Mass (kg):", "mass")
        ]

        # Volume, Density, Massの設定
        current_row = 0
        for i, (label_text, prop_name) in enumerate(properties):
            checkbox = QCheckBox()
            setattr(self, f"{prop_name}_checkbox", checkbox)

            label = QLabel(label_text)
            input_field = QLineEdit("0.000000")
            setattr(self, f"{prop_name}_input", input_field)
            
            # リターンキーで値を適用
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

        # Center of Mass（チェックボックス、ラベル、X/Y/Z入力フィールドを1行に配置）
        com_checkbox = QCheckBox()
        com_checkbox.stateChanged.connect(self.toggle_com)
        self.com_checkbox = com_checkbox
        grid_layout.addWidget(com_checkbox, current_row, 0)

        com_label = QLabel("Center of Mass:")
        grid_layout.addWidget(com_label, current_row, 1)

        # Center of MassのX、Y、Z入力フィールド
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

        # Inertia Tensor（Center of Massの次に配置）
        inertia_label = QLabel("Inertia Tensor:")
        grid_layout.addWidget(inertia_label, current_row, 1)

        self.inertia_tensor_input = QTextEdit()
        self.inertia_tensor_input.setReadOnly(True)
        self.inertia_tensor_input.setFixedHeight(40)
        self.inertia_tensor_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inertia_tensor_input.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inertia_tensor_input.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)

        # フォントサイズを10ptに設定
        font = self.inertia_tensor_input.font()
        font.setPointSize(10)
        self.inertia_tensor_input.setFont(font)

        grid_layout.addWidget(self.inertia_tensor_input, current_row, 2)
        current_row += 1

        # デフォルト値の設定
        self.density_input.setText("1.000000")

        # Calculateボタンの前にスペーサーを追加
        pre_calculate_spacer = QWidget()
        pre_calculate_spacer.setFixedHeight(2)  # 2ピクセルの空間
        grid_layout.addWidget(pre_calculate_spacer, current_row, 0, 1, 3)
        current_row += 1

        # Calculate ボタン
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_and_update_properties)
        grid_layout.addWidget(self.calculate_button, current_row, 1, 1, 2)
        current_row += 1

        # スペーサーを追加（小さな空間）
        spacer = QWidget()
        spacer.setFixedHeight(16)  # 4ピクセルの空間
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

        # カラーレイアウトを追加
        grid_layout.addLayout(color_layout, current_row, 0, 1, 3)
        current_row += 1

        # Axis layout
        axis_layout = QHBoxLayout()

        # ラジオボタンのグループを作成
        self.axis_group = QButtonGroup(self)
        axis_label = QLabel("Axis:")
        axis_layout.addWidget(axis_label)

        # ラジオボタンの作成
        radio_texts = ["X:roll", "Y:pitch", "Z:yaw", "fixed"]  # fixedを追加
        self.radio_buttons = []
        for i, text in enumerate(radio_texts):
            radio = QRadioButton(text)
            self.axis_group.addButton(radio, i)
            axis_layout.addWidget(radio)
            self.radio_buttons.append(radio)
        self.radio_buttons[0].setChecked(True)

        # Rotate Testボタン
        self.rotate_test_button = QPushButton("Rotate Test")
        self.rotate_test_button.pressed.connect(self.start_rotation_test)
        self.rotate_test_button.released.connect(self.stop_rotation_test)
        axis_layout.addWidget(self.rotate_test_button)

        grid_layout.addLayout(axis_layout, current_row, 0, 1, 3)
        current_row += 1

        # Angle (deg) layout
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle (deg):"))

        # X軸回転
        angle_layout.addWidget(QLabel("X:"))
        self.angle_x_input = QLineEdit()
        self.angle_x_input.setFixedWidth(60)
        self.angle_x_input.setText("0.0")
        self.angle_x_input.setToolTip("Body initial rotation around X axis (degrees)")
        angle_layout.addWidget(self.angle_x_input)

        # Y軸回転
        angle_layout.addWidget(QLabel("Y:"))
        self.angle_y_input = QLineEdit()
        self.angle_y_input.setFixedWidth(60)
        self.angle_y_input.setText("0.0")
        self.angle_y_input.setToolTip("Body initial rotation around Y axis (degrees)")
        angle_layout.addWidget(self.angle_y_input)

        # Z軸回転
        angle_layout.addWidget(QLabel("Z:"))
        self.angle_z_input = QLineEdit()
        self.angle_z_input.setFixedWidth(60)
        self.angle_z_input.setText("0.0")
        self.angle_z_input.setToolTip("Body initial rotation around Z axis (degrees)")
        angle_layout.addWidget(self.angle_z_input)

        # Connect Enter key to save angles
        self.angle_x_input.returnPressed.connect(self.save_current_point_angles)
        self.angle_y_input.returnPressed.connect(self.save_current_point_angles)
        self.angle_z_input.returnPressed.connect(self.save_current_point_angles)

        angle_layout.addStretch()
        grid_layout.addLayout(angle_layout, current_row, 0, 1, 3)
        current_row += 1

        # 回転テスト用の変数
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.update_test_rotation)
        self.original_transform = None
        self.test_rotation_angle = 0

        # 最後に一度だけgrid_layoutを追加
        self.left_layout.addLayout(grid_layout)

    def setup_points_ui(self):
        points_layout = QGridLayout()

        # グリッドのマージンとスペーシングの設定
        points_layout.setContentsMargins(0, 0, 0, 0)
        points_layout.setVerticalSpacing(3)
        points_layout.setHorizontalSpacing(15)  # 列間のスペースを増やす

        # 座標入力フィールドの最小幅を設定
        #coordinate_input_width = 120  # インプットフィールドの最小幅（ピクセル単位）

        # 各ポイントに対して入力フィールドとチェックボックスを作成
        for i in range(self.num_points):
            row = i

            # チェックボックスはPoint番号のみ表示
            checkbox = QCheckBox(f"Point {i+1}")
            checkbox.setObjectName("point_checkbox")  # ポイント用のチェックボックスに識別用の名前を設定
            checkbox.setMinimumWidth(80)  # チェックボックスの最小幅を設定
            checkbox.stateChanged.connect(lambda state, index=i: self.toggle_point(state, index))
            self.point_checkboxes.append(checkbox)
            points_layout.addWidget(checkbox, row, 0)

            # 座標入力用のウィジェットを作成
            inputs = []
            
            # 座標ラベルとテキストボックスの作成
            # 軸の色定義 (X=淡い赤, Y=淡い緑, Z=明るい青)
            axis_colors = [UI_COLOR_X, UI_COLOR_Y, UI_COLOR_Z]

            for j, axis in enumerate(['X', 'Y', 'Z']):
                # 水平レイアウトを作成して、ラベルとテキストボックスをグループ化
                h_layout = QHBoxLayout()
                h_layout.setSpacing(2)  # ラベルとテキストボックス間の間隔を最小に
                h_layout.setContentsMargins(0, 0, 0, 0)  # マージンを0に

                # ラベルを作成（軸と同じ色）
                label = QLabel(f"{axis}:")
                label.setFixedWidth(15)  # ラベルの幅を固定
                label.setStyleSheet(f"QLabel {{ color: {axis_colors[j]}; font-weight: bold; }}")
                h_layout.addWidget(label)
                
                # テキストボックスを作成
                input_field = QLineEdit(str(self.point_coords[i][j]))
                input_field.setFixedWidth(80)  # テキストボックスの幅を固定

                # リターンキーで3Dビューを更新
                input_field.returnPressed.connect(lambda idx=i: self.set_point(idx))

                h_layout.addWidget(input_field)
                
                # 水平レイアウトを伸縮させないようにする
                h_layout.addStretch()
                
                # 水平レイアウトをコンテナウィジェットに設定
                container = QWidget()
                container.setLayout(h_layout)
                
                # グリッドレイアウトに追加
                points_layout.addWidget(container, row, j + 1)
                
                inputs.append(input_field)
                
            self.point_inputs.append(inputs)

        # グリッドレイアウトの列の伸縮比率を設定
        points_layout.setColumnStretch(0, 1)  # Point番号の列
        points_layout.setColumnStretch(1, 1)  # X座標の列
        points_layout.setColumnStretch(2, 1)  # Y座標の列
        points_layout.setColumnStretch(3, 1)  # Z座標の列

        # RESET ボタン（Point 8の下に配置、テキスト幅サイズ、右寄せ）
        button_row = self.num_points  # Point 8の次の行
        reset_button = QPushButton("Reset Point")
        reset_button.clicked.connect(self.handle_reset_only)
        # 2文字分ずつ左右に広げる（約4文字分の幅を追加）
        char_width = reset_button.fontMetrics().averageCharWidth()
        reset_button.setFixedWidth(reset_button.fontMetrics().boundingRect("Reset Point").width() + 20 + char_width * 4)
        points_layout.addWidget(reset_button, button_row, 0, 1, 4, Qt.AlignRight)  # 全列にまたがって右寄せ

        self.left_layout.addLayout(points_layout)

        # 罫線（Reset PointとExportボタンの間）
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #707070; }")
        self.left_layout.addWidget(separator)

        # Export用のボタン配置
        export_layout = QVBoxLayout()
        export_layout.setSpacing(5)

        # 1行目：Export XML と Export Mirror Mesh with XML を横並び
        export_row1 = QHBoxLayout()
        export_row1.setSpacing(5)

        self.export_urdf_button = QPushButton("Export XML")
        self.export_urdf_button.clicked.connect(self.export_urdf)
        export_row1.addWidget(self.export_urdf_button)

        self.export_mirror_button = QPushButton("Export Mirror Mesh with XML")
        self.export_mirror_button.clicked.connect(self.export_mirror_stl_xml)
        export_row1.addWidget(self.export_mirror_button)

        export_layout.addLayout(export_row1)

        # 2行目：Batch Mirror ボタン
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
        """VTK初期化完了後にコマンドライン引数から渡されたSTLファイルをロード"""
        if not self.pending_stl_file or not os.path.exists(self.pending_stl_file):
            return

        try:
            # STLファイルを読み込む
            self.show_stl(self.pending_stl_file)

            # 対応するXMLファイルがあれば読み込む
            xml_path = os.path.splitext(self.pending_stl_file)[0] + '.xml'
            if os.path.exists(xml_path):
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # XMLからパラメータを読み込む
                    has_parameters = self.load_parameters_from_xml(root)

                    # パラメータがXMLに含まれていない場合のみ再計算を行う
                    if not has_parameters:
                        self.calculate_and_update_properties()

                    # ポイントデータを読み込む
                    self._load_points_from_xml(root)

                    # 表示を更新
                    self.refresh_view()

                    # カメラをリセット
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
            # ロード完了後はクリア
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

            # Center of Massがチェックされている場合
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

            # Pointsの移動
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

        print(f"Point {index+1} moved to: ({new_pos[0]:.6f}, {new_pos[1]:.6f}, {current_z:.6f})")

    def update_inertia_from_mass(self, mass):
        # イナーシャを重さから計算する例（適宜調整してください）
        inertia = mass * 0.1  # 例として、重さの0.1倍をイナーシャとする
        self.inertia_input.setText(f"{inertia:.12f}")

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
        """ポイントのサイズを更新（可視性の厳密な管理を追加）"""
        for index, actor in enumerate(self.point_actors):
            if actor:
                # チェックボックスの状態を確認
                is_checked = self.point_checkboxes[index].isChecked()
                
                # 一旦アクターを削除
                self.renderer.RemoveActor(actor)

                # 新しいアクターを作成
                self.point_actors[index] = create_crosshair_marker(
                    coords=[0, 0, 0],
                    radius_scale=self.calculate_sphere_radius()
                )
                self.point_actors[index].SetPosition(self.point_coords[index])
                
                # チェック状態に応じて可視性を設定
                if is_checked:
                    self.renderer.AddActor(self.point_actors[index])
                    self.point_actors[index].VisibilityOn()
                else:
                    self.point_actors[index].VisibilityOff()
        
        self.render_to_image()

    def update_all_points(self):
        """全ポイントの表示を更新（チェック状態の確認を追加）"""
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
        # ビューポートのサイズを取得
        viewport_size = self.renderer.GetSize()
        
        # 画面の対角線の長さを計算
        diagonal = math.sqrt(viewport_size[0]**2 + viewport_size[1]**2)
        
        # 対角線の10%をサイズとして設定
        radius = (diagonal * 0.1) / 2  # 半径なので2で割る
        
        # カメラのパラレルスケールでスケーリング
        camera = self.renderer.GetActiveCamera()
        parallel_scale = camera.GetParallelScale()
        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / (viewport[3] - viewport[1])
        
        # ビューポートのサイズに基づいて適切なスケールに変換
        scaled_radius = (radius / viewport_size[1]) * parallel_scale * 2
        
        if aspect_ratio > 1:
            scaled_radius /= aspect_ratio
            
        return scaled_radius

    def calculate_properties(self):
        # 優先順位: Volume > Density > Mass > Inertia
        priority_order = ['volume', 'density', 'mass', 'inertia']
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
        if 'volume' in values and 'density' in values:
            values['mass'] = values['volume'] * values['density']
        elif 'mass' in values and 'volume' in values:
            values['density'] = values['mass'] / values['volume']
        elif 'mass' in values and 'density' in values:
            values['volume'] = values['mass'] / values['density']

        # Inertiaの計算 (簡略化した例: 立方体と仮定)
        if 'mass' in values and 'volume' in values:
            side_length = np.cbrt(values['volume'])
            values['inertia'] = (1/6) * values['mass'] * side_length**2

        # 結果を入力フィールドに反映
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
        """
        重心を計算して表示する。
        チェックボックスがオンの場合は入力値を使用し、オフの場合は四面体分解法で正確に計算する。

        Assumes uniform density throughout the mesh.
        Uses tetrahedral decomposition for accurate volumetric center of mass calculation.
        """
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No 3D model has been loaded.")
            return

        # 重心の座標を取得（チェックボックスの状態に応じて）
        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            try:
                # X、Y、Z入力フィールドから座標を取得
                center_of_mass = [float(self.com_inputs[i].text()) for i in range(3)]
                print(f"Using manual Center of Mass: {center_of_mass}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing Center of Mass input: {e}")
                return None
        else:
            # STLから重心を正確に計算（四面体分解法を使用）
            print("Calculating Center of Mass using tetrahedral decomposition (uniform density assumed)...")
            poly_data = self.stl_actor.GetMapper().GetInput()

            # 四面体分解法で体積重心を計算
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

            # 計算された値を入力欄に設定
            for i in range(3):
                self.com_inputs[i].setText(f"{center_of_mass[i]:.6f}")
            print(f"Calculated Center of Mass (volumetric): [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]")

        # com_coordsを更新
        self.com_coords = list(center_of_mass)

        # 既存の赤い球アクターを削除
        if hasattr(self, 'com_sphere_actor') and self.com_sphere_actor:
            self.renderer.RemoveActor(self.com_sphere_actor)

        # 重心を可視化（赤い球）
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center_of_mass)
        sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        self.com_sphere_actor = vtk.vtkActor()
        self.com_sphere_actor.SetMapper(mapper)
        self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)  # 赤色
        self.com_sphere_actor.GetProperty().SetOpacity(0.7)

        # チェック状態に応じて表示/非表示
        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            self.com_sphere_actor.VisibilityOff()
        else:
            self.com_sphere_actor.VisibilityOn()

        self.renderer.AddActor(self.com_sphere_actor)
        self.render_to_image()

        return center_of_mass

    def calculate_inertia_tensor(self):
        """
        三角形メッシュの慣性テンソルを計算する（統合関数を使用）。

        urdf_kitchen_utils.calculate_inertia_with_trimesh() を使用して、
        trimeshとVTKのハイブリッド方式で高精度に計算します。

        Returns:
            numpy.ndarray: 3x3の慣性テンソル行列
            None: エラーが発生した場合
        """
        if not hasattr(self, 'stl_file_path') or not self.stl_file_path:
            print("No 3D model file is loaded.")
            return None

        # Get density (try from input field, fall back to default)
        try:
            density = float(self.density_input.text())
        except (ValueError, AttributeError):
            print("Warning: Could not read density, using 1.0")
            density = 1.0

        # UIから重心を取得（チェックボックスの状態により決定）
        center_of_mass = None
        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            # COMがチェックされている場合は、UIの値を使用
            center_of_mass = self.get_center_of_mass()
            if center_of_mass is None:
                print("Error getting center of mass from UI")
                return None

        # 統合関数を使用して慣性テンソルを計算
        result = calculate_inertia_with_trimesh(
            mesh_file_path=self.stl_file_path,
            mass=None,  # densityから計算
            center_of_mass=center_of_mass,
            density=density,
            auto_repair=True
        )

        # エラーチェック
        if not result['success']:
            print(f"Error calculating inertia: {result['error_message']}")
            return None

        # 結果を表示
        print(f"Volume: {result['volume']:.6f}, Density: {density:.6f}, Mass: {result['mass']:.6f}")
        print(f"Center of Mass (used): [{result['center_of_mass'][0]:.6f}, {result['center_of_mass'][1]:.6f}, {result['center_of_mass'][2]:.6f}]")
        print(f"Center of Mass (trimesh): [{result['trimesh_com'][0]:.6f}, {result['trimesh_com'][1]:.6f}, {result['trimesh_com'][2]:.6f}]")
        print(f"Watertight: {'Yes' if result['is_watertight'] else 'No'}")
        if result['repair_performed']:
            print("Mesh repair was performed")

        inertia_tensor = result['inertia_tensor']
        print("\nCalculated Inertia Tensor (about Center of Mass):")
        print(inertia_tensor)

        # URDFフォーマットに変換してUIを更新
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

        # STLファイルのパスとファイル名を取得
        stl_dir = os.path.dirname(self.stl_file_path)
        stl_filename = os.path.basename(self.stl_file_path)
        stl_name_without_ext = os.path.splitext(stl_filename)[0]

        # デフォルトのURDFファイル名を設定
        default_urdf_filename = f"{stl_name_without_ext}.xml"
        urdf_file_path, _ = QFileDialog.getSaveFileName(self, "Export XML File", os.path.join(
            stl_dir, default_urdf_filename), "XML Files (*.xml)")

        if not urdf_file_path:
            return

        try:
            # 色情報の取得と変換
            rgb_values = [float(input.text()) for input in self.color_inputs]
            hex_color = '#{:02X}{:02X}{:02X}'.format(
                int(rgb_values[0] * 255),
                int(rgb_values[1] * 255),
                int(rgb_values[2] * 255)
            )
            rgba_str = f"{rgb_values[0]:.6f} {rgb_values[1]:.6f} {rgb_values[2]:.6f} 1.0"

            # 重心の座標を取得
            try:
                com_values = [float(self.com_inputs[i].text()) for i in range(3)]
                center_of_mass_str = f"{com_values[0]:.6f} {com_values[1]:.6f} {com_values[2]:.6f}"
            except (ValueError, IndexError):
                print("Warning: Invalid center of mass format, using default values")
                center_of_mass_str = "0.000000 0.000000 0.000000"

            # 軸情報の取得
            axis_options = ["1 0 0", "0 1 0", "0 0 1", "0 0 0"]  # fixedのために"0 0 0"を追加
            checked_id = self.axis_group.checkedId()
            if 0 <= checked_id < len(axis_options):
                axis_vector = axis_options[checked_id]
            else:
                print("Warning: No axis selected, using default X axis")
                axis_vector = "1 0 0"

            # URDFの内容を構築
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

            # ポイント要素の追加
            for i, checkbox in enumerate(self.point_checkboxes):
                if checkbox.isChecked():
                    x, y, z = self.point_coords[i]
                    angle_x, angle_y, angle_z = self.point_angles[i]
                    urdf_content += f"""
    <point name="point{i+1}" type="fixed">
        <point_xyz>{x:.6f} {y:.6f} {z:.6f}</point_xyz>
        <point_angle>{angle_x:.6f} {angle_y:.6f} {angle_z:.6f}</point_angle>
    </point>"""

            # 軸情報の追加
            urdf_content += f"""
    <joint>
        <axis xyz="{axis_vector}" />
    </joint>
</urdf_part>"""

            # ファイルに保存
            with open(urdf_file_path, "w") as f:
                f.write(urdf_content)
            print(f"URDF file saved: {urdf_file_path}")

        except Exception as e:
            print(f"Error during URDF export: {str(e)}")
            traceback.print_exc()


    def get_center_of_mass(self):
        """
        UIまたは計算から重心を取得する

        Returns:
            numpy.ndarray: 重心の座標 [x, y, z]
            None: エラーが発生した場合
        """
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No 3D model is loaded.")
            return None

        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            try:
                # X、Y、Z入力フィールドから座標を取得
                center_of_mass = np.array([float(self.com_inputs[i].text()) for i in range(3)])
                print(f"Using manual Center of Mass: {center_of_mass}")
                return center_of_mass
            except (ValueError, IndexError) as e:
                print(f"Error parsing Center of Mass input: {e}")
                return None

        # チェックされていない場合やエラー時は計算値を使用
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
        """
        慣性テンソルをURDFフォーマットの文字列に変換する
        
        Args:
            inertia_tensor (numpy.ndarray): 3x3の慣性テンソル行列
        
        Returns:
            str: URDF形式の慣性テンソル文字列
        """
        # 値が非常に小さい場合は0とみなす閾値
        threshold = 1e-10

        # 対角成分
        ixx = inertia_tensor[0][0] if abs(inertia_tensor[0][0]) > threshold else 0
        iyy = inertia_tensor[1][1] if abs(inertia_tensor[1][1]) > threshold else 0
        izz = inertia_tensor[2][2] if abs(inertia_tensor[2][2]) > threshold else 0
        
        # 非対角成分
        ixy = inertia_tensor[0][1] if abs(inertia_tensor[0][1]) > threshold else 0
        ixz = inertia_tensor[0][2] if abs(inertia_tensor[0][2]) > threshold else 0
        iyz = inertia_tensor[1][2] if abs(inertia_tensor[1][2]) > threshold else 0

        return f'<inertia ixx="{ixx:.8f}" ixy="{ixy:.8f}" ixz="{ixz:.8f}" iyy="{iyy:.8f}" iyz="{iyz:.8f}" izz="{izz:.8f}"/>'


    def add_axes(self):
        if not hasattr(self, 'axes_actors'):
            self.axes_actors = []

        # 既存の軸アクターを削除
        for actor in self.axes_actors:
            self.renderer.RemoveActor(actor)
        self.axes_actors.clear()

        axis_length = 5  # 固定の軸の長さ
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 赤、緑、青
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
        #古いアクターを削除
        if hasattr(self, 'stl_actor') and self.stl_actor:
            self.renderer.RemoveActor(self.stl_actor)
        if hasattr(self, 'com_actor') and self.com_actor:
            self.renderer.RemoveActor(self.com_actor)
            self.com_actor = None

        # レンダラーをクリア
        self.renderer.Clear()

        # 座標軸の再追加
        self.axes_widget = self.add_axes_widget()

        self.stl_file_path = file_path

        # Use common utility function to load mesh
        poly_data, volume, extracted_color = load_mesh_to_polydata(file_path)

        # Handle color extraction for DAE files
        if extracted_color is not None:
            print(f"Extracted color from .dae file: RGBA({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f}, {extracted_color[3]:.3f})")
            # メッシュカラーとして保存
            self.mesh_color = extracted_color
            # 色が手動で変更されたフラグをリセット（ファイルから読み込んだ色）
            self.color_manually_changed = False
            # UIのカラーボタンも更新
            if hasattr(self, 'color_button'):
                color = QtGui.QColor(int(extracted_color[0]*255), int(extracted_color[1]*255), int(extracted_color[2]*255))
                self.color_button.setStyleSheet(f"background-color: {color.name()};")

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        self.stl_actor = vtk.vtkActor()
        self.stl_actor.SetMapper(mapper)

        # Apply extracted color to the actor
        if extracted_color is not None:
            self.stl_actor.GetProperty().SetColor(extracted_color[0], extracted_color[1], extracted_color[2])
            self.stl_actor.GetProperty().SetOpacity(extracted_color[3])
            print(f"Applied color to 3D view: RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")

        self.model_bounds = poly_data.GetBounds()
        self.renderer.AddActor(self.stl_actor)
        
        # 体積をUIに反映（小数点以下12桁）
        self.volume_input.setText(f"{volume:.12f}")
        
        # デフォルトの密度を取得して質量を計算
        density = float(self.density_input.text())
        mass = volume * density  # 体積 × 密度 = 質量
        self.mass_input.setText(f"{mass:.12f}")

        # カメラのフィッティングと描画更新
        self.fit_camera_to_model()
        self.update_all_points()

        # プロパティを更新（慣性テンソルと重心を計算）
        self.calculate_and_update_properties()
        
        # 境界ボックスを出力
        print(f"STL model bounding box: [{self.model_bounds[0]:.6f}, {self.model_bounds[1]:.6f}], [{self.model_bounds[2]:.6f}, {self.model_bounds[3]:.6f}], [{self.model_bounds[4]:.6f}, {self.model_bounds[5]:.6f}]")

        # 大原点を表示
        self.show_absolute_origin()

        # ファイル名をフルパスで更新
        self.file_name_label.setText(f"File: {file_path}")

        # カメラをリセット（Rキーと同等の処理）
        self.reset_camera()
        
    def show_absolute_origin(self):
        # 大原点を表す球を作成
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(0, 0, 0)
        sphere.SetRadius(0.0005)  # 適切なサイズに調整してください
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 0)  # 黄色

        self.renderer.AddActor(actor)
        self.render_to_image()

    def show_point(self, index):
        """ポイントを表示（XMLロード時にも使用）"""
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
        """ポイントの表示/非表示を切り替え"""
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
        """Center of Massの表示/非表示を切り替え"""
        if state == Qt.CheckState.Checked.value:
            # チェック時：赤い球を非表示、十字付き円（赤）を表示
            if self.com_sphere_actor:
                self.com_sphere_actor.VisibilityOff()

            if self.com_cursor_actor is None:
                # Center of Mass用の赤い円マーカーを作成
                self.com_cursor_actor = vtk.vtkAssembly()
                origin = [0, 0, 0]
                axis_length = self.calculate_sphere_radius() * 36
                circle_radius = self.calculate_sphere_radius()

                # XYZ軸の作成（RGB配色）
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

                # 3つの円を作成（赤色）
                for i in range(3):
                    circle = vtk.vtkRegularPolygonSource()
                    circle.SetNumberOfSides(50)
                    circle.SetRadius(circle_radius)
                    circle.SetCenter(origin[0], origin[1], origin[2])
                    if i == 0:
                        circle.SetNormal(0, 0, 1)  # XY平面
                    elif i == 1:
                        circle.SetNormal(0, 1, 0)  # XZ平面
                    else:
                        circle.SetNormal(1, 0, 0)  # YZ平面

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(circle.GetOutputPort())
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(1, 0, 0)  # 赤色
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
            # チェックなし時：十字付き円を非表示、赤い球を表示
            if self.com_cursor_actor:
                self.com_cursor_actor.VisibilityOff()
                self.renderer.RemoveActor(self.com_cursor_actor)

            # インプットフィールドの値を読み取ってcom_coordsを更新
            try:
                self.com_coords = [float(self.com_inputs[i].text()) for i in range(3)]
                print(f"Updated Center of Mass from input: {self.com_coords}")
            except (ValueError, IndexError) as e:
                print(f"Error reading Center of Mass input: {e}")
                # エラー時は現在の値を維持

            # 赤い球の位置を最新の値に更新
            if self.com_sphere_actor:
                # 既存の赤い球を削除
                self.renderer.RemoveActor(self.com_sphere_actor)

                # 新しい位置で赤い球を作成
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(self.com_coords)
                sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())

                self.com_sphere_actor.SetMapper(mapper)
                self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)  # 赤色
                self.com_sphere_actor.GetProperty().SetOpacity(0.7)

                # レンダラーに再追加
                self.renderer.AddActor(self.com_sphere_actor)
                self.com_sphere_actor.VisibilityOn()

        self.render_to_image()

    def save_current_point_angles(self):
        """現在選択されているポイントのAngle値を保存"""
        # どのポイントが選択されているか確認
        selected_index = None
        for i, checkbox in enumerate(self.point_checkboxes):
            if checkbox.isChecked():
                selected_index = i
                break

        if selected_index is None:
            print("DEBUG: No point selected, cannot save angles")
            return

        # UI入力値を読み取り（deg入力 → rad保存）
        try:
            angle_x = float(self.angle_x_input.text()) if self.angle_x_input.text() else 0.0
            angle_y = float(self.angle_y_input.text()) if self.angle_y_input.text() else 0.0
            angle_z = float(self.angle_z_input.text()) if self.angle_z_input.text() else 0.0

            # 選択されたポイントに角度を保存（radian）
            self.point_angles[selected_index] = [
                math.radians(angle_x),
                math.radians(angle_y),
                math.radians(angle_z)
            ]
            print(f"DEBUG: Saved angles for Point {selected_index+1}: {self.point_angles[selected_index]} (rad)")

            # 3Dビューのマーカーに角度を反映
            if self.point_actors[selected_index]:
                self.apply_marker_rotation(selected_index)
                self.render_to_image()
        except ValueError as e:
            print(f"ERROR: Invalid angle value: {e}")

    def apply_marker_rotation(self, index):
        """指定されたポイントのマーカーに回転を適用"""
        if not self.point_actors[index]:
            return

        angles = self.point_angles[index]
        # VTKでは角度を度数法で設定（内部はradian）
        angles_deg = [math.degrees(a) for a in angles]
        self.point_actors[index].SetOrientation(0, 0, 0)  # リセット
        self.point_actors[index].RotateX(angles_deg[0])  # X軸回転
        self.point_actors[index].RotateY(angles_deg[1])  # Y軸回転
        self.point_actors[index].RotateZ(angles_deg[2])  # Z軸回転
        print(f"DEBUG: Applied rotation to Point {index+1} marker: {angles_deg} (deg)")

    def create_com_coordinate(self, assembly, coords):
        """Center of Mass用の十字付き円を作成（赤色）"""
        origin = coords
        axis_length = self.calculate_sphere_radius() * 36  # 直径の18倍（6倍の3倍）を軸の長さとして使用
        circle_radius = self.calculate_sphere_radius()

        # XYZ軸の作成
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 赤、緑、青
        for i, color in enumerate(colors):
            for direction in [1, -1]:  # 正方向と負方向の両方
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

        # XY, XZ, YZ平面の円を作成（赤色）
        for i in range(3):
            circle = vtk.vtkRegularPolygonSource()
            circle.SetNumberOfSides(50)
            circle.SetRadius(circle_radius)
            circle.SetCenter(origin[0], origin[1], origin[2])
            if i == 0:  # XY平面
                circle.SetNormal(0, 0, 1)
            elif i == 1:  # XZ平面
                circle.SetNormal(0, 1, 0)
            else:  # YZ平面
                circle.SetNormal(1, 0, 0)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(circle.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # 円のプロパティを設定（赤色）
            actor.GetProperty().SetColor(1, 0, 0)  # 赤色
            actor.GetProperty().SetRepresentationToWireframe()  # 常にワイヤーフレーム表示
            actor.GetProperty().SetLineWidth(6)  # 線の太さを6に設定
            actor.GetProperty().SetOpacity(0.7)  # 不透明度を少し下げて見やすくする

            # タグ付けのためにUserTransformを設定
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
            return 5  # デフォルトの長さ

    def move_com_screen(self, direction, step):
        """Center of Massをスクリーン座標系で移動"""
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
        """Center of Massの表示を更新"""
        # 入力フィールドを更新
        for i in range(3):
            self.com_inputs[i].setText(f"{self.com_coords[i]:.6f}")

        # カーソルアクターの位置を更新
        if self.com_cursor_actor:
            self.com_cursor_actor.SetPosition(self.com_coords)

        self.render_to_image()

    def on_com_input_return(self):
        """Center of Massのインプットフィールドでリターンキーが押された時の処理"""
        try:
            # インプットフィールドから値を読み取る
            self.com_coords = [float(self.com_inputs[i].text()) for i in range(3)]
            print(f"Center of Mass updated from input: {self.com_coords}")

            # チェック状態に応じて表示を更新
            if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
                # チェックされている場合：カーソルアクターの位置を更新
                if self.com_cursor_actor:
                    self.com_cursor_actor.SetPosition(self.com_coords)
            else:
                # チェックされていない場合：赤い球の位置を更新
                if self.com_sphere_actor:
                    # 既存の赤い球を削除
                    self.renderer.RemoveActor(self.com_sphere_actor)

                    # 新しい位置で赤い球を作成
                    sphere = vtk.vtkSphereSource()
                    sphere.SetCenter(self.com_coords)
                    sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(sphere.GetOutputPort())

                    self.com_sphere_actor.SetMapper(mapper)
                    self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)  # 赤色
                    self.com_sphere_actor.GetProperty().SetOpacity(0.7)

                    # レンダラーに再追加
                    self.renderer.AddActor(self.com_sphere_actor)
                    self.com_sphere_actor.VisibilityOn()

            # 画面を更新
            self.render_to_image()

        except (ValueError, IndexError) as e:
            print(f"Error parsing Center of Mass input: {e}")

    def apply_volume_value(self):
        """Volumeのインプットフィールドでリターンキーが押された時の処理"""
        try:
            volume_value = float(self.volume_input.text())
            print(f"Volume value applied: {volume_value:.6f}")
            # 値が変数やメモリに採用され有効な値として適用される
            # 必要に応じて計算をトリガー
            if self.volume_checkbox.isChecked():
                # チェックされている場合、値は固定として扱われる
                pass
        except ValueError:
            print(f"Error: Invalid volume input. Please enter a valid number.")

    def apply_density_value(self):
        """Densityのインプットフィールドでリターンキーが押された時の処理"""
        try:
            density_value = float(self.density_input.text())
            print(f"Density value applied: {density_value:.6f}")
            # 値が変数やメモリに採用され有効な値として適用される
            # 必要に応じて計算をトリガー
            if self.density_checkbox.isChecked():
                # チェックされている場合、値は固定として扱われる
                pass
        except ValueError:
            print(f"Error: Invalid density input. Please enter a valid number.")

    def apply_mass_value(self):
        """Massのインプットフィールドでリターンキーが押された時の処理"""
        try:
            mass_value = float(self.mass_input.text())
            print(f"Mass value applied: {mass_value:.6f}")
            # 値が変数やメモリに採用され有効な値として適用される
            # 必要に応じて計算をトリガー
            if self.mass_checkbox.isChecked():
                # チェックされている場合、値は固定として扱われる
                pass
        except ValueError:
            print(f"Error: Invalid mass input. Please enter a valid number.")

    def fit_camera_to_model(self):
        """STLモデルが画面にフィットするようにカメラの距離のみを調整"""
        if not self.model_bounds:
            return

        camera = self.renderer.GetActiveCamera()
        
        # モデルの中心を計算
        center = [(self.model_bounds[i] + self.model_bounds[i+1]) / 2 for i in range(0, 6, 2)]
        
        # モデルの大きさを計算
        size = max([
            self.model_bounds[1] - self.model_bounds[0],
            self.model_bounds[3] - self.model_bounds[2],
            self.model_bounds[5] - self.model_bounds[4]
        ])

        # 20%の余裕を追加
        size *= 1.4  # 1.0 + 0.2 + 0.2 = 1.4

        # 現在のカメラの方向ベクトルを保持
        current_position = np.array(camera.GetPosition())
        focal_point = np.array(center)  # モデルの中心を焦点に
        direction = current_position - focal_point
        
        # 方向ベクトルを正規化
        direction = direction / np.linalg.norm(direction)
        
        # 新しい位置を計算（方向は保持したまま距離のみ調整）
        new_position = focal_point + direction * size

        # カメラの位置を更新（方向は変えない）
        camera.SetPosition(new_position)
        camera.SetFocalPoint(*center)  # モデルの中心を見る

        # ビューポートのアスペクト比を取得
        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / (viewport[3] - viewport[1])

        # モデルが画面にフィットするようにパラレルスケールを設定
        if aspect_ratio > 1:  # 横長の画面
            camera.SetParallelScale(size / 2)
        else:  # 縦長の画面
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
        
        # NumPyのベクトル演算を使用
        right = np.cross(forward, view_up)
        
        screen_right = right
        screen_up = view_up

        # ドット積の計算にNumPyを使用
        if abs(np.dot(screen_right, [1, 0, 0])) > abs(np.dot(screen_right, [0, 0, 1])):
            horizontal_axis = 'x'
            vertical_axis = 'z' if abs(np.dot(screen_up, [0, 0, 1])) > abs(np.dot(screen_up, [0, 1, 0])) else 'y'
        else:
            horizontal_axis = 'z'
            vertical_axis = 'y'

        return horizontal_axis, vertical_axis, screen_right, screen_up

    def handle_reset_only(self):
        """チェックされたポイントを原点にリセット"""
        for i, checkbox in enumerate(self.point_checkboxes):
            if checkbox.isChecked():
                self.reset_point_to_origin(i)

        self.update_all_points_size()
        self.render_to_image()

    def export_mirror_stl_xml(self):
        """3DモデルファイルをY軸でミラーリングし、対応するXMLファイルも生成する（STL/DAE対応）"""
        if not hasattr(self, 'stl_file_path') or not self.stl_file_path:
            print("No 3D model file has been loaded.")
            return

        try:
            # 元のファイルのパスとファイル名を取得
            original_dir = os.path.dirname(self.stl_file_path)
            original_filename = os.path.basename(self.stl_file_path)
            name, ext = os.path.splitext(original_filename)

            # 新しいファイル名を生成（L/R反転）
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

            # ミラー化したファイルのパスを設定
            mirrored_stl_path = os.path.join(original_dir, new_name + ext)
            mirrored_xml_path = os.path.join(original_dir, new_name + '.xml')

            # 既存ファイルのチェックとダイアログ表示
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

                # Y軸に対して反転する変換を作成
                transform = vtk.vtkTransform()
                transform.Scale(1, -1, 1)

                # 変換を適用
                transform_filter = vtk.vtkTransformPolyDataFilter()
                transform_filter.SetInputData(poly_data)
                transform_filter.SetTransform(transform)
                transform_filter.Update()

            else:
                # Load STL file using VTK
                reader = vtk.vtkSTLReader()
                reader.SetFileName(self.stl_file_path)
                reader.Update()

                # Y軸に対して反転する変換を作成
                transform = vtk.vtkTransform()
                transform.Scale(1, -1, 1)

                # 変換を適用
                transform_filter = vtk.vtkTransformPolyDataFilter()
                transform_filter.SetInputConnection(reader.GetOutputPort())
                transform_filter.SetTransform(transform)
                transform_filter.Update()

            # 法線の修正
            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.SetInputData(transform_filter.GetOutput())
            normal_generator.ConsistencyOn()
            normal_generator.AutoOrientNormalsOn()
            normal_generator.ComputeCellNormalsOn()
            normal_generator.ComputePointNormalsOn()
            normal_generator.Update()

            # XMLファイルを確認し読み込む
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

                    # XMLから物理パラメータを取得（元のフォーマットを保持）
                    mass_element = xml_data.find(".//mass")
                    mass_value_str = mass_element.get('value') if mass_element is not None else "0.0"
                    mass = float(mass_value_str)  # 計算用
                    
                    volume_element = xml_data.find(".//volume")
                    volume_value_str = volume_element.get('value') if volume_element is not None else "0.0"
                    volume = float(volume_value_str)  # 計算用
                    
                    # 重心位置を取得（center_of_mass要素から）
                    com_element = xml_data.find(".//center_of_mass")
                    if com_element is not None and com_element.text:
                        x, y, z = map(float, com_element.text.strip().split())
                        center_of_mass = [x, -y, z]  # Y座標のみを反転
                    else:
                        # inertialのorigin要素から取得
                        inertial_origin = xml_data.find(".//inertial/origin")
                        if inertial_origin is not None:
                            xyz = inertial_origin.get('xyz')
                            x, y, z = map(float, xyz.split())
                            center_of_mass = [x, -y, z]  # Y座標のみを反転
                        else:
                            print("Warning: No center of mass information found in XML")
                            center_of_mass = [0, 0, 0]

                    print(f"Original mass: {mass:.6f}, volume: {volume:.6f}")
                    print(f"Original center of mass: {center_of_mass}")

                    # 色情報を取得
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

            # ミラー化したファイルを保存
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

            # イナーシャテンソルを取得してミラーリング変換
            print("\nProcessing inertia tensor for mirrored model...")
            inertia_element = None
            if xml_data is not None:
                inertia_element = xml_data.find(".//inertia")
            
            if inertia_element is not None:
                # 元のXMLからイナーシャテンソルを取得してミラーリング変換
                ixx = float(inertia_element.get('ixx', 0))
                iyy = float(inertia_element.get('iyy', 0))
                izz = float(inertia_element.get('izz', 0))
                ixy = float(inertia_element.get('ixy', 0))
                ixz = float(inertia_element.get('ixz', 0))
                iyz = float(inertia_element.get('iyz', 0))
                # Y軸ミラーの場合、ixyとiyzの符号を反転
                inertia_str = f'ixx="{ixx:.12f}" ixy="{-ixy:.12f}" ixz="{ixz:.12f}" iyy="{iyy:.12f}" iyz="{-iyz:.12f}" izz="{izz:.12f}"'
                print(f"Mirrored inertia tensor from XML: ixx={ixx:.6f}, iyy={iyy:.6f}, izz={izz:.6f}, ixy={-ixy:.6f}, ixz={ixz:.6f}, iyz={-iyz:.6f}")
            else:
                # 元のXMLにイナーシャ情報がない場合、メッシュから計算
                print("Warning: No inertia data in XML, calculating from mesh...")
                inertia_tensor = self.calculate_inertia_tensor_for_mirrored(
                    normal_generator.GetOutput(), mass, center_of_mass)
                # format_inertia_for_urdfの戻り値から<inertia と/>を削除
                inertia_formatted = self.format_inertia_for_urdf(inertia_tensor)
                inertia_str = inertia_formatted.replace('<inertia ', '').replace('/>', '').strip()

            # XMLファイルの内容を生成
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

            # ポイントデータを反転してコピー
            if xml_data is not None:
                print("Processing point data...")
                points = xml_data.findall('.//point')
                for point in points:
                    xyz_element = point.find('point_xyz')
                    if xyz_element is not None and xyz_element.text:
                        try:
                            x, y, z = map(float, xyz_element.text.strip().split())
                            mirrored_y = -y  # Y座標のみ反転
                            point_name = point.get('name')

                            # point_angleの処理（Y軸ミラーリング時の回転変換）
                            angle_element = point.find('point_angle')
                            point_angle_str = ""
                            if angle_element is not None and angle_element.text:
                                try:
                                    angle_x, angle_y, angle_z = map(float, angle_element.text.strip().split())
                                    # Y軸ミラーの場合、X軸とZ軸の回転を符号反転
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

            # 軸情報を取得して適用
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

            # 関節角度制限を取得（ロール/ヨー軸の場合は入れ替え）
            joint_limit_str = ""
            if xml_data is not None:
                limit_element = xml_data.find('.//joint/limit')
                if limit_element is not None:
                    lower = limit_element.get('lower')
                    upper = limit_element.get('upper')
                    effort = limit_element.get('effort')
                    velocity = limit_element.get('velocity')

                    if lower is not None and upper is not None:
                        # 軸の種類を判定（1 0 0 = Roll, 0 1 0 = Pitch, 0 0 1 = Yaw）
                        axis_xyz = [float(x) for x in mirrored_axis.split()]
                        is_roll = abs(axis_xyz[0]) > 0.5  # X軸成分が大きい = Roll
                        is_yaw = abs(axis_xyz[2]) > 0.5   # Z軸成分が大きい = Yaw

                        if is_roll or is_yaw:
                            # Roll/Yaw軸の場合は最小・最大を入れ替え、かつ符号を反転
                            # 例：lower=-10, upper=190 -> lower=-190, upper=10
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

            # XMLファイルを保存
            print(f"Saving XML to: {mirrored_xml_path}")
            with open(mirrored_xml_path, "w") as f:
                f.write(urdf_content)

            # 対応するcollider XMLファイルを処理
            # 元のファイル名からcollider XMLのパスを生成
            original_name_base = os.path.splitext(original_filename)[0]
            collider_xml_path = os.path.join(original_dir, original_name_base + '_collider.xml')
            
            if os.path.exists(collider_xml_path):
                print(f"\nFound collider XML: {collider_xml_path}")
                try:
                    # 新しいcollider XMLファイル名を生成
                    new_collider_xml_name = new_name + '_collider.xml'
                    new_collider_xml_path = os.path.join(original_dir, new_collider_xml_name)
                    
                    # コライダーXMLファイルを読み込み
                    tree = ET.parse(collider_xml_path)
                    root = tree.getroot()
                    
                    if root.tag != 'urdf_kitchen_collider':
                        print(f"  ⚠ Warning: Invalid collider XML format (expected 'urdf_kitchen_collider'), skipping")
                    else:
                        collider_elem = root.find('collider')
                        if collider_elem is None:
                            print(f"  ⚠ Warning: No collider element found, skipping")
                        else:
                            # コライダータイプを取得
                            collider_type = collider_elem.get('type', 'box')
                            
                            # ジオメトリ情報を取得
                            geometry_elem = collider_elem.find('geometry')
                            geometry_attrs = {}
                            if geometry_elem is not None:
                                geometry_attrs = dict(geometry_elem.attrib)
                            
                            # 位置情報を取得してxz平面で反転（y座標を反転）
                            position_elem = collider_elem.find('position')
                            if position_elem is not None:
                                x = float(position_elem.get('x', '0.0'))
                                y = float(position_elem.get('y', '0.0'))
                                z = float(position_elem.get('z', '0.0'))
                                mirrored_y = -y  # Y座標を反転
                            else:
                                x, y, z = 0.0, 0.0, 0.0
                                mirrored_y = 0.0
                            
                            # 回転情報を取得（RollとYawを反転、Pitchはそのまま）
                            rotation_elem = collider_elem.find('rotation')
                            if rotation_elem is not None:
                                roll = float(rotation_elem.get('roll', '0.0'))
                                pitch = float(rotation_elem.get('pitch', '0.0'))
                                yaw = float(rotation_elem.get('yaw', '0.0'))
                                # RollとYawを反転、Pitchはそのまま
                                mirrored_roll = -roll
                                mirrored_pitch = pitch  # Pitchは反転しない
                                mirrored_yaw = -yaw
                            else:
                                roll, pitch, yaw = 0.0, 0.0, 0.0
                                mirrored_roll = 0.0
                                mirrored_pitch = 0.0
                                mirrored_yaw = 0.0
                            
                            # 新しいコライダーXMLファイルを作成
                            new_root = ET.Element('urdf_kitchen_collider')
                            new_collider_elem = ET.SubElement(new_root, 'collider')
                            new_collider_elem.set('type', collider_type)
                            
                            # ジオメトリ要素を追加
                            if geometry_attrs:
                                new_geometry_elem = ET.SubElement(new_collider_elem, 'geometry')
                                for key, value in geometry_attrs.items():
                                    new_geometry_elem.set(key, value)
                            
                            # 位置要素を追加（y座標を反転）
                            new_position_elem = ET.SubElement(new_collider_elem, 'position')
                            new_position_elem.set('x', f"{x:.6f}")
                            new_position_elem.set('y', f"{mirrored_y:.6f}")
                            new_position_elem.set('z', f"{z:.6f}")
                            
                            # 回転要素を追加（RollとYawを反転、Pitchはそのまま）
                            new_rotation_elem = ET.SubElement(new_collider_elem, 'rotation')
                            new_rotation_elem.set('roll', f"{mirrored_roll:.6f}")
                            new_rotation_elem.set('pitch', f"{mirrored_pitch:.6f}")
                            new_rotation_elem.set('yaw', f"{mirrored_yaw:.6f}")
                            print(f"  Mirrored rotation: Roll={roll:.4f}->{mirrored_roll:.4f}, Pitch={pitch:.4f}->{mirrored_pitch:.4f}, Yaw={yaw:.4f}->{mirrored_yaw:.4f}")
                            
                            # mesh_file要素があればコピー（l_をr_に置換）
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
                            
                            # XMLファイルを保存
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

            # ダイアログボックスで出力内容を表示
            dialog = ResultDialog(mirrored_stl_path, mirrored_xml_path, self)
            dialog.exec()

        except Exception as e:
            print(f"\nAn error occurred during mirror export: {str(e)}")
            traceback.print_exc()
            # エラーダイアログも表示
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during mirror export:\n{str(e)}"
            )
        

    def calculate_inertia_tensor_for_mirrored(self, poly_data, mass, center_of_mass):
        """
        ミラーリングされたモデルの慣性テンソルを計算

        Args:
            poly_data: vtkPolyData オブジェクト
            mass: float 質量
            center_of_mass: list[float] 重心座標 [x, y, z]

        Returns:
            numpy.ndarray: 3x3 慣性テンソル行列
        """
        # Use shared triangle-based method with mirroring from urdf_kitchen_utils
        inertia_tensor = calculate_inertia_tensor(
            poly_data, mass, center_of_mass, is_mirrored=True
        )

        print("\nCalculated Inertia Tensor:")
        print(inertia_tensor)
        return inertia_tensor



    def _load_points_from_xml(self, root):
        """XMLからポイントデータを読み込む"""
        points_with_data = set()
        # './/point'とすることで、どの階層にあるpointタグも検索できる
        points = root.findall('.//point')
        print(f"Found {len(points)} points in XML")

        for i, point in enumerate(points):
            if i >= len(self.point_checkboxes):  # 配列の境界チェック
                break

            xyz_element = point.find('point_xyz')
            if xyz_element is not None and xyz_element.text:
                try:
                    x, y, z = map(float, xyz_element.text.strip().split())
                    print(f"Loading point {i+1}: ({x}, {y}, {z})")

                    # 座標の設定
                    self.point_inputs[i][0].setText(f"{x:.6f}")
                    self.point_inputs[i][1].setText(f"{y:.6f}")
                    self.point_inputs[i][2].setText(f"{z:.6f}")
                    self.point_coords[i] = [x, y, z]
                    points_with_data.add(i)

                    # point_angleの読み込み（radian）
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

                    # チェックボックスをオンにする
                    self.point_checkboxes[i].setChecked(True)

                    # ポイントの表示を設定
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

            # 最初にチェックされたポイントのAngle値をUIに反映
            first_point_index = min(points_with_data)
            self.angle_x_input.setText(f"{self.point_angles[first_point_index][0]:.2f}")
            self.angle_y_input.setText(f"{self.point_angles[first_point_index][1]:.2f}")
            self.angle_z_input.setText(f"{self.point_angles[first_point_index][2]:.2f}")
            print(f"Updated Angle UI to Point {first_point_index+1}: {self.point_angles[first_point_index]}")

        return points_with_data

    def _apply_color_from_xml(self, root):
        """XMLからカラー情報を適用"""
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
        """表示を更新する"""
        self.renderer.ResetCamera()
        self.fit_camera_to_model()
        self.update_all_points_size()
        self.update_all_points()
        self.calculate_center_of_mass()
        self.add_axes()
        self.renderer.ResetCameraClippingRange()
        self.render_to_image()

    def load_parameters_from_xml(self, root):
        """XMLからパラメータを読み込んで設定する共通処理"""
        try:
            # まず全てのポイントをリセット
            for i in range(self.num_points):
                # 座標を0にリセット
                self.point_coords[i] = [0, 0, 0]
                for j in range(3):
                    self.point_inputs[i][j].setText("0.000000")
                # チェックボックスを外す
                self.point_checkboxes[i].setChecked(False)
                # 既存のアクターを削除
                if self.point_actors[i]:
                    self.point_actors[i].VisibilityOff()
                    self.renderer.RemoveActor(self.point_actors[i])
                    self.point_actors[i] = None

            has_parameters = False

            # 色情報の読み込み
            material_element = root.find(".//material")
            if material_element is not None:
                color_element = material_element.find("color")
                if color_element is not None:
                    rgba_str = color_element.get('rgba')
                    if rgba_str:
                        try:
                            r, g, b, a = map(float, rgba_str.split())
                            # 色情報を入力フィールドに設定
                            self.color_inputs[0].setText(f"{r:.3f}")
                            self.color_inputs[1].setText(f"{g:.3f}")
                            self.color_inputs[2].setText(f"{b:.3f}")
                            if len(self.color_inputs) >= 4:
                                self.color_inputs[3].setText(f"{a:.3f}")

                            # STLモデルに色を適用
                            if hasattr(self, 'stl_actor') and self.stl_actor:
                                self.stl_actor.GetProperty().SetColor(r, g, b)
                                self.stl_actor.GetProperty().SetOpacity(a)
                                self.render_to_image()

                            has_parameters = True
                            print(f"Loaded color: RGBA({r:.3f}, {g:.3f}, {b:.3f}, {a:.3f})")
                        except ValueError as e:
                            print(f"Error parsing color values: {e}")

            # 軸情報の読み込み
            joint_element = root.find(".//joint/axis")
            if joint_element is not None:
                axis_str = joint_element.get('xyz')
                if axis_str:
                    try:
                        x, y, z = map(float, axis_str.split())
                        # 対応するラジオボタンを選択
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

            # 体積を取得して設定
            volume_element = root.find(".//volume")
            if volume_element is not None:
                volume = volume_element.get('value')
                self.volume_input.setText(volume)
                self.volume_checkbox.setChecked(True)
                has_parameters = True

            # 質量を取得して設定
            mass_element = root.find(".//mass")
            if mass_element is not None:
                mass = mass_element.get('value')
                self.mass_input.setText(mass)
                self.mass_checkbox.setChecked(True)
                has_parameters = True

            # 重心の取得と設定（優先順位付き）
            com_str = None
            
            # まず<center_of_mass>タグを確認
            com_element = root.find(".//center_of_mass")
            if com_element is not None and com_element.text:
                com_str = com_element.text.strip()
            
            # 次にinertialのorigin要素を確認
            if com_str is None:
                inertial_origin = root.find(".//inertial/origin")
                if inertial_origin is not None:
                    xyz = inertial_origin.get('xyz')
                    if xyz:
                        com_str = xyz
            
            # 最後にvisualのorigin要素を確認
            if com_str is None:
                visual_origin = root.find(".//visual/origin")
                if visual_origin is not None:
                    xyz = visual_origin.get('xyz')
                    if xyz:
                        com_str = xyz

            # 重心値を設定
            if com_str:
                try:
                    x, y, z = map(float, com_str.split())
                    self.com_inputs[0].setText(f"{x:.6f}")
                    self.com_inputs[1].setText(f"{y:.6f}")
                    self.com_inputs[2].setText(f"{z:.6f}")
                    self.com_coords = [x, y, z]
                    print(f"Loaded center of mass: ({x:.6f}, {y:.6f}, {z:.6f})")

                    # Center of Massの赤い球を更新
                    if hasattr(self, 'renderer'):
                        # 既存の赤い球を削除
                        if hasattr(self, 'com_sphere_actor') and self.com_sphere_actor:
                            self.renderer.RemoveActor(self.com_sphere_actor)

                        # 新しい位置で赤い球を作成
                        sphere = vtk.vtkSphereSource()
                        sphere.SetCenter(self.com_coords)
                        sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(sphere.GetOutputPort())

                        self.com_sphere_actor = vtk.vtkActor()
                        self.com_sphere_actor.SetMapper(mapper)
                        self.com_sphere_actor.GetProperty().SetColor(1, 0, 0)  # 赤色
                        self.com_sphere_actor.GetProperty().SetOpacity(0.7)

                        # チェック状態を確認して表示/非表示を設定
                        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
                            self.com_sphere_actor.VisibilityOff()
                        else:
                            self.com_sphere_actor.VisibilityOn()

                        self.renderer.AddActor(self.com_sphere_actor)
                        print(f"Center of Mass sphere updated at: {self.com_coords}")

                    has_parameters = True
                except ValueError as e:
                    print(f"Error parsing center of mass values: {e}")

            # 慣性テンソルの設定
            inertia_element = root.find(".//inertia")
            if inertia_element is not None:
                inertia_str = ET.tostring(inertia_element, encoding='unicode')
                self.inertia_tensor_input.setText(inertia_str)
                has_parameters = True

            # ポイントデータの読み込み
            points = root.findall(".//point")
            for i, point in enumerate(points):
                if i >= len(self.point_checkboxes):
                    break

                xyz_element = point.find("point_xyz")
                if xyz_element is not None and xyz_element.text:
                    try:
                        x, y, z = map(float, xyz_element.text.strip().split())
                        # 座標値を設定
                        self.point_inputs[i][0].setText(f"{x:.6f}")
                        self.point_inputs[i][1].setText(f"{y:.6f}")
                        self.point_inputs[i][2].setText(f"{z:.6f}")
                        self.point_coords[i] = [x, y, z]
                        
                        # チェックボックスをオンにする
                        self.point_checkboxes[i].setChecked(True)
                        
                        # ポイントを表示
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

            # カメラをリセット
            self.reset_camera()
            
            return has_parameters

        except Exception as e:
            print(f"Error loading parameters: {str(e)}")
            traceback.print_exc()
            return False

    def load_xml_file(self):
        """XMLファイルのみを読み込み、パラメータを反映する"""
        try:
            xml_path, _ = QFileDialog.getOpenFileName(self, "Open XML File", "", "XML Files (*.xml)")
            if not xml_path:
                return

            # XMLファイルパスを保存
            self.xml_file_path = xml_path

            # XMLファイルを解析
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            print("Processing XML file...")

            # XMLからパラメータを読み込む
            has_parameters = self.load_parameters_from_xml(root)

            # パラメータがXMLに含まれていない場合のみ再計算を行う
            if not has_parameters:
                self.calculate_and_update_properties()

            # 全てのポイントをリセット
            print("Resetting all points...")
            for i in range(self.num_points):
                # テキストフィールドをクリア
                self.point_inputs[i][0].setText("0.000000")
                self.point_inputs[i][1].setText("0.000000")
                self.point_inputs[i][2].setText("0.000000")
                
                # 内部座標データをリセット
                self.point_coords[i] = [0, 0, 0]
                
                # チェックボックスを解除
                self.point_checkboxes[i].setChecked(False)
                
                # 3Dビューのポイントを非表示にし、アクターを削除
                if self.point_actors[i]:
                    self.point_actors[i].VisibilityOff()
                    self.renderer.RemoveActor(self.point_actors[i])
                    self.point_actors[i] = None
            
            print("All points have been reset")

            # データが設定されたポイントを追跡
            points_with_data = set()

            # 各ポイントの座標を読み込む
            points = root.findall('./point')
            print(f"Found {len(points)} points in XML")

            for i, point in enumerate(points):
                xyz_element = point.find('point_xyz')
                if xyz_element is not None and xyz_element.text:
                    try:
                        # 座標テキストを分割して数値に変換
                        x, y, z = map(float, xyz_element.text.strip().split())
                        print(f"Point {i+1}: {x}, {y}, z")

                        # テキストフィールドに値を設定
                        self.point_inputs[i][0].setText(f"{x:.6f}")
                        self.point_inputs[i][1].setText(f"{y:.6f}")
                        self.point_inputs[i][2].setText(f"{z:.6f}")
                        
                        # 内部の座標データを更新
                        self.point_coords[i] = [x, y, z]

                        # point_angleの読み込み（radian）
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

                        # チェックボックスを有効化
                        self.point_checkboxes[i].setChecked(True)
                        
                        # ポイントの表示を設定
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

            # STLモデルが読み込まれている場合のみ色を適用
            if hasattr(self, 'stl_actor') and self.stl_actor:
                color_element = root.find(".//material/color")
                if color_element is not None:
                    rgba_str = color_element.get('rgba')
                    if rgba_str:
                        try:
                            r, g, b, a = map(float, rgba_str.split())

                            # インプットフィールドに値を設定
                            self.color_inputs[0].setText(f"{r:.3f}")
                            self.color_inputs[1].setText(f"{g:.3f}")
                            self.color_inputs[2].setText(f"{b:.3f}")
                            if len(self.color_inputs) >= 4:
                                self.color_inputs[3].setText(f"{a:.3f}")

                            # STLモデルに色を適用
                            self.stl_actor.GetProperty().SetColor(r, g, b)
                            self.stl_actor.GetProperty().SetOpacity(a)
                            self.render_to_image()

                            print(f"Material color loaded and applied: RGBA({r:.3f}, {g:.3f}, {b:.3f}, {a:.3f})")
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Invalid color format in XML: {rgba_str}")
                            print(f"Error details: {e}")

            # 軸情報の処理
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

            # 表示の更新
            if hasattr(self, 'renderer'):
                self.renderer.ResetCamera()
                self.update_all_points()

                # STLモデルが存在する場合、カメラをフィット
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
        """現在読み込まれているMeshファイルとXMLファイルを再読み込みする"""
        try:
            print("\n" + "="*60)
            print("RELOADING FILES")
            print("="*60)

            # 現在のファイルパスを保存
            mesh_path = getattr(self, 'stl_file_path', None)
            xml_path = getattr(self, 'xml_file_path', None)

            if not mesh_path and not xml_path:
                print("No files to reload. Please load files first.")
                return

            reload_count = 0

            # Meshファイルの再読み込み
            if mesh_path and os.path.exists(mesh_path):
                print(f"\n[1/2] Reloading mesh file: {mesh_path}")
                self.show_stl(mesh_path)
                reload_count += 1
            elif mesh_path:
                print(f"Warning: Mesh file not found: {mesh_path}")
            else:
                print("No mesh file to reload")

            # XMLファイルの再読み込み
            if xml_path and os.path.exists(xml_path):
                print(f"\n[2/2] Reloading XML file: {xml_path}")

                try:
                    # XMLファイルを解析
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # XMLからパラメータを読み込む
                    has_parameters = self.load_parameters_from_xml(root)

                    # パラメータがXMLに含まれていない場合のみ再計算を行う
                    if not has_parameters:
                        self.calculate_and_update_properties()

                    # ポイントデータを読み込む（既存のヘルパー関数を使用）
                    points_with_data = self._load_points_from_xml(root)

                    # カラー情報を適用（既存のヘルパー関数を使用）
                    self._apply_color_from_xml(root)

                    # 表示を更新
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

            # 完了メッセージ
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
        """ビューの更新とカメラのフィッティングを行う"""
        if hasattr(self, 'renderer'):
            self.renderer.ResetCamera()
            self.update_all_points()
            # STLモデルが存在する場合、カメラをフィット
            if hasattr(self, 'stl_actor') and self.stl_actor:
                self.fit_camera_to_model()
            self.renderer.ResetCameraClippingRange()
            self.render_to_image()

    def load_stl_with_xml(self):
        """3Dモデルファイル（STL/OBJ/DAE）とXMLファイルを一緒に読み込む"""
        try:
            # Use common utility function for file filter
            file_filter = get_mesh_file_filter(trimesh_available=True)
            stl_path, _ = QFileDialog.getOpenFileName(self, "Open 3D Model File", "", file_filter)
            if not stl_path:
                return

            # 3Dモデルファイルを読み込む
            self.show_stl(stl_path)

            # 対応するXMLファイルのパスを生成
            xml_path = os.path.splitext(stl_path)[0] + '.xml'

            if not os.path.exists(xml_path):
                print(f"Corresponding XML file not found: {xml_path}")
                return

            # XMLファイルを解析
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # XMLからパラメータを読み込む
                has_parameters = self.load_parameters_from_xml(root)
                
                # パラメータがXMLに含まれていない場合のみ再計算を行う
                if not has_parameters:
                    self.calculate_and_update_properties()
                    
                # ポイントデータを読み込む
                points_with_data = self._load_points_from_xml(root)
                
                print(f"XML file loaded: {xml_path}")
                if points_with_data:
                    print(f"Loaded {len(points_with_data)} points")
                
                # 表示を更新
                self.refresh_view()

                # カメラをリセット（Rキーと同等の処理）
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
        フォルダ内の'l_'または'L_'で始まるSTL/OBJ/DAEファイルを処理し、
        XZ平面を中心とした左右対称の'r_'または'R_'ファイルとXMLを生成する。
        既存のファイルは上書きする。
        """
        try:
            import re

            # フォルダ選択ダイアログを表示
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder for Bulk Conversion")
            if not folder_path:
                return

            print(f"Selected folder: {folder_path}")

            # 処理したファイルの数を追跡
            processed_count = 0
            collider_count = 0
            # 生成されたファイルのリストを保存
            generated_files = []

            # l_*_collider パターン（l_で始まり_collider.拡張子で終わる）
            # 例: l_hipjoint_upper_collider.stl, l_elbow_collider.dae
            collider_pattern = re.compile(r'^l_.+_collider\.(stl|obj|dae)$', re.IGNORECASE)

            # フォルダ内のすべてのSTL/OBJ/DAEファイルを検索
            print("\n=== Searching for l_ prefix files ===")
            print("Collider pattern: l_*_collider.(stl|obj|dae)")
            print("Examples: l_hipjoint_upper_collider.stl, l_elbow_collider.dae")
            print("")

            for file_name in os.listdir(folder_path):
                if file_name.lower().startswith(('l_', 'L_')) and file_name.lower().endswith(('.stl', '.obj', '.dae')):
                    # コライダーファイルかどうかを判定
                    is_collider = collider_pattern.match(file_name.lower())
                    file_type = "COLLIDER" if is_collider else "mesh"

                    print(f"✓ Found {file_type}: {file_name}")
                    stl_path = os.path.join(folder_path, file_name)

                    # 新しいファイル名を生成（l_ を r_ に変換）
                    new_name = 'R_' + file_name[2:] if file_name.startswith('L_') else 'r_' + file_name[2:]
                    new_name_without_ext = os.path.splitext(new_name)[0]
                    new_stl_path = os.path.join(folder_path, new_name)
                    new_xml_path = os.path.splitext(new_stl_path)[0] + '.xml'

                    print(f"  → Mesh: {new_name}")
                    print(f"  → XML:  {os.path.basename(new_xml_path)}")
                    if is_collider:
                        # コライダーの場合、XMLファイル名のパターンを明示
                        print(f"  → Pattern match: r_.*_collider.xml ✓")

                    # 既存ファイルがある場合は上書き
                    if os.path.exists(new_stl_path) or os.path.exists(new_xml_path):
                        print(f"  ⚠ Overwriting existing files")

                    print(f"\nProcessing: {stl_path}")

                    try:
                        # Use common utility function to load mesh
                        poly_data, volume_unused, extracted_color = load_mesh_to_polydata(stl_path)

                        # Y軸反転の変換を設定
                        transform = vtk.vtkTransform()
                        transform.Scale(1, -1, 1)

                        # 頂点を変換
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

                        # 対応するXMLファイルを探す
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

                                # XMLから物理パラメータを取得（元のフォーマットを保持）
                                mass_element = xml_data.find(".//mass")
                                mass_value_str = mass_element.get('value') if mass_element is not None else "0.0"
                                mass = float(mass_value_str)  # 計算用
                                
                                volume_element = xml_data.find(".//volume")
                                volume_value_str = volume_element.get('value') if volume_element is not None else "0.0"
                                volume = float(volume_value_str)  # 計算用
                                
                                # 重心位置を取得（center_of_mass要素から）
                                com_element = xml_data.find(".//center_of_mass")
                                if com_element is not None and com_element.text:
                                    x, y, z = map(float, com_element.text.strip().split())
                                    center_of_mass = [x, -y, z]  # Y座標のみを反転
                                else:
                                    # inertialのorigin要素から取得
                                    inertial_origin = xml_data.find(".//inertial/origin")
                                    if inertial_origin is not None:
                                        xyz = inertial_origin.get('xyz')
                                        x, y, z = map(float, xyz.split())
                                        center_of_mass = [x, -y, z]  # Y座標のみを反転

                                # 色情報を取得
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

                        # イナーシャテンソルを取得してミラーリング変換
                        inertia_element = None
                        if xml_data is not None:
                            inertia_element = xml_data.find(".//inertia")
                        
                        if inertia_element is not None:
                            # 元のXMLからイナーシャテンソルを取得してミラーリング変換
                            ixx = float(inertia_element.get('ixx', 0))
                            iyy = float(inertia_element.get('iyy', 0))
                            izz = float(inertia_element.get('izz', 0))
                            ixy = float(inertia_element.get('ixy', 0))
                            ixz = float(inertia_element.get('ixz', 0))
                            iyz = float(inertia_element.get('iyz', 0))
                            # Y軸ミラーの場合、ixyとiyzの符号を反転
                            inertia_str = f'ixx="{ixx:.12f}" ixy="{-ixy:.12f}" ixz="{ixz:.12f}" iyy="{iyy:.12f}" iyz="{-iyz:.12f}" izz="{izz:.12f}"'
                        else:
                            # 元のXMLにイナーシャ情報がない場合、メッシュから計算
                            print("Warning: No inertia data in XML, calculating from mesh...")
                            inertia_tensor = self.calculate_inertia_tensor_for_mirrored(
                                normal_generator.GetOutput(), mass, center_of_mass)
                            # format_inertia_for_urdfの戻り値から<inertia と/>を削除
                            inertia_formatted = self.format_inertia_for_urdf(inertia_tensor)
                            inertia_str = inertia_formatted.replace('<inertia ', '').replace('/>', '').strip()

                        # XMLファイルの内容を生成
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

                        # ポイントデータを反転してコピー
                        if xml_data is not None:
                            points = xml_data.findall('.//point')
                            for point in points:
                                xyz_element = point.find('point_xyz')
                                if xyz_element is not None and xyz_element.text:
                                    try:
                                        x, y, z = map(float, xyz_element.text.strip().split())
                                        mirrored_y = -y  # Y座標のみ反転
                                        point_name = point.get('name')

                                        # point_angleの処理（Y軸ミラーリング時の回転変換）
                                        angle_element = point.find('point_angle')
                                        point_angle_str = ""
                                        if angle_element is not None and angle_element.text:
                                            try:
                                                angle_x, angle_y, angle_z = map(float, angle_element.text.strip().split())
                                                # Y軸ミラーの場合、X軸とZ軸の回転を符号反転
                                                mirrored_angle_x = -angle_x
                                                mirrored_angle_y = angle_y
                                                mirrored_angle_z = -angle_z
                                                point_angle_str = f"\n        <point_angle>{mirrored_angle_x:.6f} {mirrored_angle_y:.6f} {mirrored_angle_z:.6f}</point_angle>"
                                            except ValueError:
                                                pass  # point_angleが不正な場合は無視

                                        urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>{point_angle_str}
    </point>"""
                                    except ValueError:
                                        print(f"Error processing point coordinates in XML")

                        # 軸情報を取得して適用
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

                        # XMLファイルを保存
                        with open(new_xml_path, "w") as f:
                            f.write(urdf_content)

                        processed_count += 1
                        if is_collider:
                            collider_count += 1
                        # 生成されたファイルをリストに追加
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

            # すでに処理されたXMLのリスト（メッシュと一緒に処理されたもの）
            processed_xml_names = set()
            for item in generated_files:
                if 'xml' in item:
                    processed_xml_names.add(os.path.basename(item['xml']))

            for file_name in os.listdir(folder_path):
                if xml_collider_pattern.match(file_name.lower()):
                    # 既にメッシュと一緒に処理されていないかチェック
                    output_xml_name = 'R_' + file_name[2:] if file_name.startswith('L_') else 'r_' + file_name[2:]
                    if output_xml_name in processed_xml_names:
                        continue  # 既に処理済み

                    xml_path = os.path.join(folder_path, file_name)
                    new_xml_name = output_xml_name
                    new_xml_path = os.path.join(folder_path, new_xml_name)

                    print(f"✓ Found standalone XML: {file_name}")
                    print(f"  → Will create: {new_xml_name}")

                    try:
                        # XMLファイルを読み込み
                        tree = ET.parse(xml_path)
                        xml_data = tree.getroot()

                        # 物理パラメータを取得
                        mass_element = xml_data.find(".//mass")
                        volume_element = xml_data.find(".//volume")

                        mass = float(mass_element.get('value')) if mass_element is not None else 1.0
                        volume = float(volume_element.get('value')) if volume_element is not None else 1.0

                        # 重心位置を取得して反転
                        com_element = xml_data.find(".//center_of_mass")
                        if com_element is not None and com_element.text:
                            x, y, z = map(float, com_element.text.strip().split())
                            center_of_mass = [x, -y, z]  # Y座標のみ反転
                        else:
                            # inertialのorigin要素から取得
                            inertial_origin = xml_data.find(".//inertial/origin")
                            if inertial_origin is not None:
                                xyz = inertial_origin.get('xyz')
                                x, y, z = map(float, xyz.split())
                                center_of_mass = [x, -y, z]
                            else:
                                center_of_mass = [0, 0, 0]

                        # 色情報を取得
                        color_element = xml_data.find(".//material/color")
                        if color_element is not None:
                            rgba_str = color_element.get('rgba')
                            hex_color = xml_data.find(".//material").get('name')
                        else:
                            rgba_str = "1.0 1.0 1.0 1.0"
                            hex_color = "#FFFFFF"

                        # イナーシャテンソルを取得して反転
                        inertia_element = xml_data.find(".//inertia")
                        if inertia_element is not None:
                            ixx = float(inertia_element.get('ixx', 0))
                            iyy = float(inertia_element.get('iyy', 0))
                            izz = float(inertia_element.get('izz', 0))
                            ixy = float(inertia_element.get('ixy', 0))
                            ixz = float(inertia_element.get('ixz', 0))
                            iyz = float(inertia_element.get('iyz', 0))
                            # Y軸ミラーの場合、ixyとiyzの符号を反転
                            inertia_str = f'ixx="{ixx:.12f}" ixy="{-ixy:.12f}" ixz="{ixz:.12f}" iyy="{iyy:.12f}" iyz="{-iyz:.12f}" izz="{izz:.12f}"'
                        else:
                            inertia_str = 'ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"'

                        # リンク名を生成
                        base_name = os.path.splitext(new_xml_name)[0]

                        # XMLファイルの内容を生成
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

                        # ポイントデータを反転してコピー
                        points = xml_data.findall('.//point')
                        for point in points:
                            xyz_element = point.find('point_xyz')
                            if xyz_element is not None and xyz_element.text:
                                try:
                                    x, y, z = map(float, xyz_element.text.strip().split())
                                    mirrored_y = -y  # Y座標のみ反転
                                    point_name = point.get('name')
                                    urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>
    </point>"""
                                except ValueError:
                                    print(f"Error processing point coordinates in XML")

                        # 軸情報を取得して適用
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

                        # XMLファイルを保存
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
                        # コライダーXMLファイルを読み込み
                        tree = ET.parse(collider_xml_path)
                        root = tree.getroot()

                        if root.tag != 'urdf_kitchen_collider':
                            print(f"  ⚠ Warning: Invalid collider XML format (expected 'urdf_kitchen_collider'), skipping")
                            continue

                        collider_elem = root.find('collider')
                        if collider_elem is None:
                            print(f"  ⚠ Warning: No collider element found, skipping")
                            continue

                        # コライダータイプを取得
                        collider_type = collider_elem.get('type', 'box')

                        # ジオメトリ情報を取得
                        geometry_elem = collider_elem.find('geometry')
                        geometry_attrs = {}
                        if geometry_elem is not None:
                            geometry_attrs = dict(geometry_elem.attrib)

                        # 位置情報を取得してxz平面で反転（y座標を反転）
                        position_elem = collider_elem.find('position')
                        if position_elem is not None:
                            x = float(position_elem.get('x', '0.0'))
                            y = float(position_elem.get('y', '0.0'))
                            z = float(position_elem.get('z', '0.0'))
                            mirrored_y = -y  # Y座標を反転
                        else:
                            x, y, z = 0.0, 0.0, 0.0
                            mirrored_y = 0.0

                        # 回転情報を取得してxz平面で反転
                        # xz平面で反転する場合、rollとyawはそのまま、pitchは反転
                        rotation_elem = collider_elem.find('rotation')
                        if rotation_elem is not None:
                            roll = float(rotation_elem.get('roll', '0.0'))
                            pitch = float(rotation_elem.get('pitch', '0.0'))
                            yaw = float(rotation_elem.get('yaw', '0.0'))
                            # RollとYawを反転、Pitchはそのまま
                            mirrored_roll = -roll
                            mirrored_pitch = pitch  # Pitchは反転しない
                            mirrored_yaw = -yaw
                        else:
                            roll, pitch, yaw = 0.0, 0.0, 0.0
                            mirrored_roll = 0.0
                            mirrored_pitch = 0.0
                            mirrored_yaw = 0.0

                        # 新しいコライダーXMLファイルを作成
                        new_root = ET.Element('urdf_kitchen_collider')
                        new_collider_elem = ET.SubElement(new_root, 'collider')
                        new_collider_elem.set('type', collider_type)

                        # ジオメトリ要素を追加
                        if geometry_attrs:
                            new_geometry_elem = ET.SubElement(new_collider_elem, 'geometry')
                            for key, value in geometry_attrs.items():
                                new_geometry_elem.set(key, value)

                        # 位置要素を追加（y座標を反転）
                        new_position_elem = ET.SubElement(new_collider_elem, 'position')
                        new_position_elem.set('x', f"{x:.6f}")
                        new_position_elem.set('y', f"{mirrored_y:.6f}")
                        new_position_elem.set('z', f"{z:.6f}")

                        # 回転要素を追加（RollとYawを反転、Pitchはそのまま）
                        new_rotation_elem = ET.SubElement(new_collider_elem, 'rotation')
                        new_rotation_elem.set('roll', f"{mirrored_roll:.6f}")
                        new_rotation_elem.set('pitch', f"{mirrored_pitch:.6f}")
                        new_rotation_elem.set('yaw', f"{mirrored_yaw:.6f}")
                        print(f"  Mirrored rotation: Roll={roll:.4f}->{mirrored_roll:.4f}, Pitch={pitch:.4f}->{mirrored_pitch:.4f}, Yaw={yaw:.4f}->{mirrored_yaw:.4f}")

                        # XMLファイルを保存
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

            # 処理完了メッセージ
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
                # ダイアログボックスで生成されたファイルのリストを表示
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
        軸情報を左右反転する際の処理
        回転軸の向きは左右で変更しない
        
        Args:
            axis_str (str): "x y z" 形式の軸情報
        
        Returns:
            str: 変換後の軸情報
        """
        try:
            x, y, z = map(float, axis_str.split())
            # 軸の向きは変更せずにそのまま返す
            return f"{x:.1f} {y:.1f} {z:.1f}"
        except ValueError:
            print(f"Error parsing axis values: {axis_str}")
            return "1 0 0"  # デフォルト値


    def start_rotation_test(self):
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            return
            
        # 現在の変換行列を保存
        self.original_transform = vtk.vtkTransform()
        self.original_transform.DeepCopy(self.stl_actor.GetUserTransform() 
                                    if self.stl_actor.GetUserTransform() 
                                    else vtk.vtkTransform())
        
        self.test_rotation_angle = 0
        self.rotation_timer.start(16)  # 約60FPS

    def stop_rotation_test(self):
        self.rotation_timer.stop()
        
        # 元の位置に戻す
        if self.stl_actor and self.original_transform:
            self.stl_actor.SetUserTransform(self.original_transform)
            self.render_to_image()

    def update_test_rotation(self):
        if not self.stl_actor:
            return
                
        # 選択された軸を確認
        axis_index = self.axis_group.checkedId()
        
        # fixedが選択されている場合（axis_index == 3）は何もしない
        if axis_index == 3:
            return
                
        # 以下は従来の処理
        rotation_axis = [0, 0, 0]
        rotation_axis[axis_index] = 1
        
        # 回転角度を更新
        self.test_rotation_angle += 2  # 1フレームあたり2度回転
        
        # 回転変換を作成
        transform = vtk.vtkTransform()
        transform.DeepCopy(self.original_transform)
        transform.RotateWXYZ(self.test_rotation_angle, *rotation_axis)
        
        # 変換を適用
        self.stl_actor.SetUserTransform(transform)
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
        """選択された色を3Dモデルに適用（RGBA対応）"""
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No 3D model has been loaded.")
            return

        try:
            # RGBA値を取得（0-1の範囲）
            rgba_values = [float(input.text()) for input in self.color_inputs]

            # 値の範囲チェック
            rgba_values = [max(0.0, min(1.0, value)) for value in rgba_values]

            # メッシュカラーとして保存（RGBA形式）
            if len(rgba_values) == 3:
                # RGBのみの場合、Alpha=1.0を追加
                self.mesh_color = rgba_values + [1.0]
            else:
                # RGBAの場合、そのまま保存
                self.mesh_color = rgba_values

            # 色が手動で変更されたことをマーク
            self.color_manually_changed = True

            # STLモデルの色を変更（RGB）
            self.stl_actor.GetProperty().SetColor(*rgba_values[:3])

            # 透明度を設定（Alpha）
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
        """座標軸を表示するウィジェットを追加（オフスクリーンレンダリングモードでは無効）"""
        # Note: vtkOrientationMarkerWidget requires an interactor,
        # which is not available in offscreen rendering mode
        # Return None to maintain compatibility
        return None

    def add_instruction_text(self):
        """画面上に操作説明を表示"""
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

    def process_mirror_properties(self, xml_data, reverse_output, density=1.0):
        """
        ミラーリングされたモデルの物理プロパティを処理する
        Args:
            xml_data: 元のXMLデータ
            reverse_output: 反転後のvtkPolyData
            density: デフォルトの密度（元のXMLに質量情報がない場合に使用）
        Returns:
            tuple: (volume, mass, center_of_mass, inertia_tensor)
        """
        # 体積を計算（新しいジオメトリから）
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(reverse_output)
        volume = mass_properties.GetVolume()

        # 質量を元のXMLから取得（ない場合は体積×密度で計算）
        if xml_data is not None:
            mass_element = xml_data.find(".//mass")
            if mass_element is not None:
                mass = float(mass_element.get('value'))
            else:
                mass = volume * density
        else:
            mass = volume * density

        # 重心を計算
        com_filter = vtk.vtkCenterOfMass()
        com_filter.SetInputData(reverse_output)
        com_filter.SetUseScalarsAsWeights(False)
        com_filter.Update()
        center_of_mass = list(com_filter.GetCenter())
        
        # Y座標のみを反転
        center_of_mass[1] = -center_of_mass[1]

        # 慣性テンソルを計算（質量を考慮）
        inertia_tensor = np.zeros((3, 3))
        poly_data = reverse_output
        num_cells = poly_data.GetNumberOfCells()

        # 実際の質量を使用して慣性テンソルを計算
        density_for_inertia = mass / volume  # 実際の質量から密度を逆算
        
        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                p1, p2, p3 = [np.array(cell.GetPoints().GetPoint(j)) for j in range(3)]
                centroid = (p1 + p2 + p3) / 3
                r = centroid - np.array(center_of_mass)
                area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

                # 慣性テンソルの計算
                inertia_tensor[0, 0] += area * (r[1]**2 + r[2]**2)
                inertia_tensor[1, 1] += area * (r[0]**2 + r[2]**2)
                inertia_tensor[2, 2] += area * (r[0]**2 + r[1]**2)
                inertia_tensor[0, 1] -= area * r[0] * r[1]
                inertia_tensor[0, 2] -= area * r[0] * r[2]
                inertia_tensor[1, 2] -= area * r[1] * r[2]

        # 対称性を利用して下三角を埋める
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]

        # 実際の質量に基づいて慣性テンソルをスケーリング
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

        # ウィンドウサイズを大きく設定
        self.resize(400, 250)  # 幅を600に、高さを200に増加

        # レイアウトを作成
        layout = QVBoxLayout()
        layout.setSpacing(10)  # ウィジェット間の間隔を設定

        # メッセージラベルを作成
        title_label = QLabel("Following files have been saved:")
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)

        # 3Dモデルファイルパスを表示
        stl_label = QLabel(f"Mesh: {stl_path}")
        stl_label.setWordWrap(True)  # 長いパスの折り返しを有効化
        layout.addWidget(stl_label)

        # XMLファイルパスを表示
        xml_label = QLabel(f"XML: {xml_path}")
        xml_label.setWordWrap(True)  # 長いパスの折り返しを有効化
        layout.addWidget(xml_label)

        # スペーサーを追加
        layout.addSpacing(20)

        # Closeボタンを作成
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setFixedWidth(100)

        # ボタンを中央に配置するための水平レイアウト
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Enterキーでダイアログを閉じられるようにする
        close_button.setDefault(True)

class BulkConversionCompleteDialog(QDialog):
    """Batch Mirror処理完了後に生成されたファイルのリストを表示するダイアログ"""
    def __init__(self, generated_files: list, folder_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Conversion Complete")
        self.setModal(True)

        # ウィンドウサイズを設定
        self.resize(600, 400)

        # レイアウトを作成
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # ファイルタイプごとのカウント
        mesh_count = sum(1 for f in generated_files if f['mesh'] is not None)
        xml_only_count = sum(1 for f in generated_files if f['mesh'] is None)

        # メッセージラベルを作成
        title_label = QLabel(f"✓ Batch Conversion Complete!")
        title_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #2ecc71;")
        layout.addWidget(title_label)

        # 統計情報を表示
        stats_text = f"Total: {len(generated_files)} pair(s) created"
        if mesh_count > 0:
            stats_text += f"\n  • Mesh files: {mesh_count}"
        if xml_only_count > 0:
            stats_text += f"\n  • Standalone XMLs: {xml_only_count}"
        stats_text += "\n  • Mirrored across XZ plane (Y-axis flip)"

        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-size: 11pt; margin: 5px 0px;")
        layout.addWidget(stats_label)

        # ディレクトリパスを表示
        dir_label = QLabel(f"Directory: {folder_path}")
        dir_label.setStyleSheet("font-size: 10pt;")
        dir_label.setWordWrap(True)
        layout.addWidget(dir_label)

        # 区切り線
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # ファイルリストのヘッダー
        list_header = QLabel("Generated Files:")
        list_header.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 5px;")
        layout.addWidget(list_header)

        # スクロール可能なテキストエリアを作成
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("QTextEdit { font-family: monospace; background-color: #f5f5f5; }")

        # 生成されたファイルのリストをテキストとして構築（ファイル名のみ）
        file_list_text = ""
        for i, file_pair in enumerate(generated_files, 1):
            # メッシュファイルがある場合のみ追加（スタンドアロンXMLの場合はNone）
            if file_pair['mesh'] is not None:
                mesh_filename = os.path.basename(file_pair['mesh'])
                file_list_text += f"{mesh_filename}\n"

            xml_filename = os.path.basename(file_pair['xml'])
            file_list_text += f"{xml_filename}\n"
            # ファイルペア間に空行を追加（最後のセット以外）
            if i < len(generated_files):
                file_list_text += "\n"

        text_edit.setPlainText(file_list_text)
        layout.addWidget(text_edit)

        # Closeボタンを作成
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setFixedWidth(100)

        # ボタンを中央に配置するための水平レイアウト
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Enterキーでダイアログを閉じられるようにする
        close_button.setDefault(True)

# signal_handler moved to urdf_kitchen_utils.py
# Now using setup_signal_handlers()

if __name__ == "__main__":

    # Ctrl+Cのシグナルハンドラを設定（utils関数使用）
    setup_signal_handlers()

    app = QApplication(sys.argv)
    apply_dark_theme(app)  # Custom theme with extensive widget styling

    window = MainWindow()
    window.show()

    # コマンドライン引数からSTLファイルのパスを取得
    # VTK初期化前なので、pending_stl_fileに保存してVTK初期化後にロード
    if len(sys.argv) > 1:
        stl_file_path = sys.argv[1]
        if os.path.exists(stl_file_path):
            window.pending_stl_file = stl_file_path
        else:
            print(f"File not found: {stl_file_path}")

    # シグナル処理用タイマー（utils関数使用）
    timer = setup_signal_processing_timer(app)

    try:
        sys.exit(app.exec())
    except SystemExit:
        print("Exiting application...")
