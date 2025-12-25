"""
File Name: urdf_kitchen_PartsEditor.py
Description: A Python script for configuring connection points of parts for urdf_kitchen_Assembler.py.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Update.     : Dec 25, 2025
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
    QTextEdit, QButtonGroup, QRadioButton, QColorDialog, QDialog, QMessageBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextOption, QColor, QPalette

# Import URDF Kitchen utilities
from urdf_kitchen_utils import (
    OffscreenRenderer, CameraController, AnimatedCameraRotation,
    AdaptiveMarkerSize, create_crosshair_marker, MouseDragState,
    calculate_arrow_key_step
)

# pip install numpy
# pip install PySide6
# pip install vtk
# pip install NodeGraphQt

def apply_dark_theme(app):
    """シックなダークテーマを適用"""
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
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("URDF Kitchen - PartsEditor v0.1.0 -")
        self.setGeometry(0, 0, 1200, 600)
        self.camera_rotation = [0, 0, 0]  # [yaw, pitch, roll]
        self.absolute_origin = [0, 0, 0]  # 大原点の設定
        self.initial_camera_position = [10, 0, 0]  # 初期カメラ位置
        self.initial_camera_focal_point = [0, 0, 0]  # 初期焦点
        self.initial_camera_view_up = [0, 0, 1]  # 初期の上方向

        self.num_points = 8  # ポイントの数を8に設定
        self.point_coords = [list(self.absolute_origin) for _ in range(self.num_points)]
        self.point_actors = [None] * self.num_points
        self.point_checkboxes = []
        self.point_inputs = []
        self.point_set_buttons = []
        self.point_reset_buttons = []

        self.com_coords = [0.0, 0.0, 0.0]  # Center of Mass座標
        self.com_sphere_actor = None  # 赤い球（チェックなし時）
        self.com_cursor_actor = None  # 十字付き円（チェックあり時、赤色）

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

        # Load STLボタン
        self.load_button = QPushButton("Load Mesh")
        self.load_button.clicked.connect(self.load_stl_file)
        first_row.addWidget(self.load_button)

        # Load XMLボタン
        self.load_xml_button = QPushButton("Import XML")
        self.load_xml_button.clicked.connect(self.load_xml_file)
        first_row.addWidget(self.load_xml_button)

        # first_row_layout を first_row_widget にセット
        first_row_widget.setLayout(first_row_layout)

        # Load STL with XMLボタン
        self.load_stl_xml_button = QPushButton("Load Mesh with XML")
        self.load_stl_xml_button.clicked.connect(self.load_stl_with_xml)
        button_layout.addWidget(self.load_stl_xml_button)

        # スペーサーを追加
        spacer = QWidget()
        spacer.setFixedHeight(0)  # 20ピクセルの空間を作る
        button_layout.addWidget(spacer)
        
        # Export用のボタンを縦に配置
        self.export_urdf_button = QPushButton("Export XML")
        self.export_urdf_button.clicked.connect(self.export_urdf)
        button_layout.addWidget(self.export_urdf_button)
        
        # ミラーボタン
        self.export_mirror_button = QPushButton("Export Mirror Mesh with XML")
        self.export_mirror_button.clicked.connect(self.export_mirror_stl_xml)
        button_layout.addWidget(self.export_mirror_button)

        # 一括変換ボタン
        self.bulk_convert_button = QPushButton("Batch Mirror \"l_\" to \"r_\" Meshes")
        self.bulk_convert_button.clicked.connect(self.bulk_convert_l_to_r)
        button_layout.addWidget(self.bulk_convert_button)

        # Export STLボタン
        self.export_stl_button = QPushButton("Save Mesh with Point1 as Origin")
        self.export_stl_button.clicked.connect(self.export_stl_with_new_origin)
        button_layout.addWidget(self.export_stl_button)

        self.left_layout.addLayout(button_layout)

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
        spacer.setFixedHeight(2)  # 8ピクセルの空間
        grid_layout.addWidget(spacer, current_row, 0, 1, 3)
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

        # 回転テスト用の変数
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.update_test_rotation)
        self.original_transform = None
        self.test_rotation_angle = 0

        # Color layout
        color_layout = QHBoxLayout()
        
        color_layout.addWidget(QLabel("Color:"))
        
        self.color_inputs = []
        for label in ['R:', 'G:', 'B:']:
            color_layout.addWidget(QLabel(label))
            color_input = QLineEdit("1.0")
            color_input.setFixedWidth(50)
            color_input.textChanged.connect(self.update_color_sample)
            self.color_inputs.append(color_input)
            color_layout.addWidget(color_input)
        
        self.color_sample = QLabel()
        self.color_sample.setFixedSize(30, 20)
        self.color_sample.setStyleSheet("background-color: rgb(255,255,255); border: 1px solid black;")
        color_layout.addWidget(self.color_sample)
        
        pick_button = QPushButton("Pick")
        pick_button.clicked.connect(self.show_color_picker)
        color_layout.addWidget(pick_button)
        
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_color_to_stl)
        color_layout.addWidget(apply_button)
        
        color_layout.addStretch()
        
        # カラーレイアウトを追加
        grid_layout.addLayout(color_layout, current_row, 0, 1, 3)

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
            for j, axis in enumerate(['X', 'Y', 'Z']):
                # 水平レイアウトを作成して、ラベルとテキストボックスをグループ化
                h_layout = QHBoxLayout()
                h_layout.setSpacing(2)  # ラベルとテキストボックス間の間隔を最小に
                h_layout.setContentsMargins(0, 0, 0, 0)  # マージンを0に
                
                # ラベルを作成
                label = QLabel(f"{axis}:")
                label.setFixedWidth(15)  # ラベルの幅を固定
                h_layout.addWidget(label)
                
                # テキストボックスを作成
                input_field = QLineEdit(str(self.point_coords[i][j]))
                input_field.setFixedWidth(80)  # テキストボックスの幅を固定
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

        # SET/RESETボタンの行
        button_row = self.num_points
        # ボタン用の水平レイアウト
        button_layout = QHBoxLayout()
        
        # SET ボタン
        set_button = QPushButton("Set Point")
        set_button.clicked.connect(self.handle_set_reset)
        button_layout.addWidget(set_button)

        # RESET ボタン
        reset_button = QPushButton("Reset Point")
        reset_button.clicked.connect(self.handle_set_reset)
        button_layout.addWidget(reset_button)

        # ボタンレイアウトをグリッドに追加
        button_container = QWidget()
        button_container.setLayout(button_layout)
        points_layout.addWidget(button_container, button_row, 0, 1, 4)

        self.left_layout.addLayout(points_layout)
        
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

    def setup_vtk(self):
        """VTKのオフスクリーンレンダリングを設定（Mac互換性）"""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)  # グレー背景

        # オフスクリーンレンダリングウィンドウを作成
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(800, 600)
        self.render_window.AddRenderer(self.renderer)

        # インタラクタは不要（オフスクリーンレンダリングのため）
        self.render_window_interactor = None

        # Initialize utility classes
        self.offscreen_renderer = OffscreenRenderer(self.render_window, self.renderer)
        self.camera_controller = CameraController(self.renderer, self.absolute_origin)
        self.animated_rotation = AnimatedCameraRotation(self.renderer, self.absolute_origin)
        self.mouse_state = MouseDragState(self.vtk_display)

    def setup_camera(self):
        position = [self.absolute_origin[i] + self.initial_camera_position[i] for i in range(3)]
        self.camera_controller.setup_parallel_camera(
            position=position,
            view_up=self.initial_camera_view_up,
            focal_point=self.absolute_origin,
            parallel_scale=5
        )

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

    def update_point_display(self, index):
        """ポイントの表示を更新（チェック状態の確認を追加）"""
        if self.point_actors[index]:
            if self.point_checkboxes[index].isChecked():
                self.point_actors[index].SetPosition(self.point_coords[index])
                self.point_actors[index].VisibilityOn()
            else:
                self.point_actors[index].VisibilityOff()
                self.renderer.RemoveActor(self.point_actors[index])
        
        for i, coord in enumerate(self.point_coords[index]):
            self.point_inputs[index][i].setText(f"{coord:.6f}")
        
        self.render_to_image()

    def update_all_points_size(self, obj=None, event=None):
        """ポイントのサイズを更新（可視性の厳密な管理を追加）"""
        for index, actor in enumerate(self.point_actors):
            if actor:
                # チェックボックスの状態を確認
                is_checked = self.point_checkboxes[index].isChecked()
                
                # 一旦アクターを削除
                self.renderer.RemoveActor(actor)
                
                # 新しいアクターを作成
                self.point_actors[index] = vtk.vtkAssembly()
                self.create_point_coordinate(self.point_actors[index], [0, 0, 0])
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

    def create_point_coordinate(self, assembly, coords):
        origin = coords
        axis_length = self.calculate_sphere_radius() * 36  # 直径の18倍（6倍の3倍）を軸の長さとして使用
        circle_radius = self.calculate_sphere_radius()

        print(f"Creating point coordinate at {coords}")
        print(f"Axis length: {axis_length}, Circle radius: {circle_radius}")

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
                print(f"Added {['X', 'Y', 'Z'][i]} axis {'positive' if direction == 1 else 'negative'}")

        # XY, XZ, YZ平面の円を作成
        for i in range(3):
            circle = vtk.vtkRegularPolygonSource()
            circle.SetNumberOfSides(50)
            circle.SetRadius(circle_radius)
            circle.SetCenter(origin[0], origin[1], origin[2])
            if i == 0:  # XY平面
                circle.SetNormal(0, 0, 1)
                plane = "XY"
            elif i == 1:  # XZ平面
                circle.SetNormal(0, 1, 0)
                plane = "XZ"
            else:  # YZ平面
                circle.SetNormal(1, 0, 0)
                plane = "YZ"

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(circle.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # 円のプロパティを設定(ponintのカーソル表示用)
            actor.GetProperty().SetColor(1, 0, 1)  # 紫色
            actor.GetProperty().SetRepresentationToWireframe()  # 常にワイヤーフレーム表示
            actor.GetProperty().SetLineWidth(6)  # 線の太さを3倍の6に設定
            actor.GetProperty().SetOpacity(0.7)  # 不透明度を少し下げて見やすくする

            # タグ付けのためにUserTransformを設定
            transform = vtk.vtkTransform()
            actor.SetUserTransform(transform)

            assembly.AddPart(actor)
            print(f"Added {plane} circle")

        print(f"Point coordinate creation completed")

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

    def calculate_screen_diagonal(self):
        viewport_size = self.renderer.GetSize()
        return math.sqrt(viewport_size[0]**2 + viewport_size[1]**2)

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
            print("No STL model has been loaded.")
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
            print("No STL model has been loaded.")
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
        三角形メッシュの慣性テンソルを計算する（四面体分解法を使用）。
        重心位置を考慮し、正確なイナーシャを算出する。

        This implementation uses the tetrahedral decomposition method described in:
        "Fast and Accurate Computation of Polyhedral Mass Properties" by Brian Mirtich (1996)

        Returns:
            numpy.ndarray: 3x3の慣性テンソル行列
            None: エラーが発生した場合
        """
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No STL model is loaded.")
            return None

        # ポリデータを取得
        poly_data = self.stl_actor.GetMapper().GetInput()

        # 体積と質量を取得して表示
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(poly_data)
        mass_properties.Update()
        total_volume = mass_properties.GetVolume()

        # Get density (try from input field, fall back to default)
        try:
            density = float(self.density_input.text())
        except (ValueError, AttributeError):
            print("Warning: Could not read density, using 1.0")
            density = 1.0

        mass = total_volume * density
        print(f"Volume: {total_volume:.6f}, Density: {density:.6f}, Mass: {mass:.6f}")

        # UIまたは計算から重心を取得
        center_of_mass = self.get_center_of_mass()
        if center_of_mass is None:
            print("Error getting center of mass")
            return None

        com = np.array(center_of_mass)
        print(f"Center of Mass: [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")

        # 慣性テンソルの初期化（重心周りで計算）
        # Using tetrahedral decomposition method
        # Each triangle forms a tetrahedron with the origin

        # Canonical moments (for unit density)
        volume_integral = 0.0
        com_integral = np.zeros(3)
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

                # For inertia calculation, we use the formula for a tetrahedron
                # The inertia tensor components can be calculated using:
                # I_ij = ∫∫∫ ρ(r²δ_ij - x_i*x_j) dV

                # For a tetrahedron with vertices at origin and v0, v1, v2:
                # We sum over all combinations of vertices
                for vi in [v0, v1, v2]:
                    for vj in [v0, v1, v2]:
                        # Contribution to inertia integral
                        r_squared = np.dot(vi, vi)
                        for a in range(3):
                            for b in range(3):
                                if a == b:
                                    # Diagonal: I_aa = ∫(y² + z²)dm for x-axis, etc.
                                    inertia_integral[a, b] += tet_volume * (r_squared - vi[a] * vj[a]) / 20.0
                                else:
                                    # Off-diagonal: I_ab = -∫(x*y)dm
                                    inertia_integral[a, b] -= tet_volume * vi[a] * vj[b] / 20.0

                # Additional vertex pair contributions
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

        # Verify volume calculation
        print(f"Volume from integration: {volume_integral:.6f} (expected: {total_volume:.6f})")
        volume_ratio = volume_integral / total_volume if total_volume > 0 else 0
        if abs(volume_ratio - 1.0) > 0.01:
            print(f"Warning: Volume mismatch! Ratio = {volume_ratio:.4f}")
            print("This may indicate non-closed mesh or incorrect orientation")

        # Convert to inertia tensor with proper density scaling
        # The integral gives us the second moment, multiply by density
        inertia_tensor = inertia_integral * density

        # Ensure symmetry (should already be symmetric, but numerical errors can occur)
        inertia_tensor = 0.5 * (inertia_tensor + inertia_tensor.T)

        # Clean up numerical noise
        threshold = 1e-12
        inertia_tensor[np.abs(inertia_tensor) < threshold] = 0.0

        # Display results
        print("\nCalculated Inertia Tensor (about Center of Mass):")
        print(inertia_tensor)

        # Verify physical constraints
        # Diagonal elements should be positive for a physical object
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
        if not (diagonal[0] + diagonal[1] >= diagonal[2] - 1e-6 and
                diagonal[1] + diagonal[2] >= diagonal[0] - 1e-6 and
                diagonal[2] + diagonal[0] >= diagonal[1] - 1e-6):
            print("\nWarning: Inertia tensor violates triangle inequality!")
            print("This indicates a potential calculation error.")

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
            print("No STL file has been loaded.")
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
                    urdf_content += f"""
    <point name="point{i+1}" type="fixed">
        <point_xyz>{x:.6f} {y:.6f} {z:.6f}</point_xyz>
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
            print("No STL model is loaded.")
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


    def apply_camera_rotation(self, camera):
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
        # Support both STL and DAE (COLLADA) files
        file_filter = "3D Model Files (*.stl *.dae);;STL Files (*.stl);;COLLADA Files (*.dae);;All Files (*)"

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

        # Check file extension to determine loader
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.dae':
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

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(clean.GetOutput())
            self.stl_actor = vtk.vtkActor()
            self.stl_actor.SetMapper(mapper)

            self.model_bounds = clean.GetOutput().GetBounds()
            self.renderer.AddActor(self.stl_actor)

            # Get volume
            mass_properties = vtk.vtkMassProperties()
            mass_properties.SetInputData(clean.GetOutput())
            volume = mass_properties.GetVolume()

        else:
            # Load STL file using VTK
            reader = vtk.vtkSTLReader()
            reader.SetFileName(file_path)
            reader.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            self.stl_actor = vtk.vtkActor()
            self.stl_actor.SetMapper(mapper)

            self.model_bounds = reader.GetOutput().GetBounds()
            self.renderer.AddActor(self.stl_actor)

            # STLの体積を取得
            mass_properties = vtk.vtkMassProperties()
            mass_properties.SetInputConnection(reader.GetOutputPort())
            volume = mass_properties.GetVolume()
        
        # 体積をUIに反映（小数点以下12桁）
        self.volume_input.setText(f"{volume:.12f}")
        
        # デフォルトの密度を取得して質量を計算
        density = float(self.density_input.text())
        mass = volume * density  # 体積 × 密度 = 質量
        self.mass_input.setText(f"{mass:.12f}")

        # イナーシャを計算（簡略化：立方体と仮定）
        #side_length = np.cbrt(volume)
        #inertia = (1/6) * mass * side_length**2
        #self.inertia_input.setText(f"{inertia:.12f}")

        # 慣性テンソルを計算
        inertia_tensor = self.calculate_inertia_tensor()

        # カメラのフィッティングと描画更新
        self.fit_camera_to_model()
        self.update_all_points()

        # プロパティを更新
        self.calculate_and_update_properties()

        # 重心を計算して表示
        center_of_mass = self.calculate_center_of_mass()
        # 重心を計算して表示
        self.calculate_center_of_mass()
        
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
            return

        if self.point_actors[index] is None:
            self.point_actors[index] = vtk.vtkAssembly()
            self.create_point_coordinate(self.point_actors[index], [0, 0, 0])
            self.renderer.AddActor(self.point_actors[index])
        
        self.point_actors[index].SetPosition(self.point_coords[index])
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
        self.pending_rotation_update = (self.rotation_types[rotation_type], angle)  # Store for later update
        self.is_animating = True  # Block further input
        self.animation_timer.start(1000 // 60)

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
            # Update camera_rotation only when animation completes
            if hasattr(self, 'pending_rotation_update') and self.pending_rotation_update:
                rotation_type, angle = self.pending_rotation_update
                self.camera_rotation[rotation_type] += angle
                self.camera_rotation[rotation_type] %= 360
                self.pending_rotation_update = None
            self.is_animating = False  # Allow new input

    def toggle_point(self, state, index):
        """ポイントの表示/非表示を切り替え"""
        if state == Qt.CheckState.Checked.value:
            if self.point_actors[index] is None:
                self.point_actors[index] = vtk.vtkAssembly()
                self.create_point_coordinate(self.point_actors[index], [0, 0, 0])
                self.renderer.AddActor(self.point_actors[index])
            self.point_actors[index].SetPosition(self.point_coords[index])
            self.point_actors[index].VisibilityOn()
            self.renderer.AddActor(self.point_actors[index])
        else:
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
                self.com_cursor_actor = vtk.vtkAssembly()
                self.create_com_coordinate(self.com_cursor_actor, [0, 0, 0])
                self.renderer.AddActor(self.com_cursor_actor)
            self.com_cursor_actor.SetPosition(self.com_coords)
            self.com_cursor_actor.VisibilityOn()
            self.renderer.AddActor(self.com_cursor_actor)
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

    def hide_point(self, index):
        if self.point_actors[index]:
            self.point_actors[index].VisibilityOff()
        self.render_to_image()

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

    def move_point(self, index, dx, dy, dz):
        new_position = [
            self.point_coords[index][0] + dx,
            self.point_coords[index][1] + dy,
            self.point_coords[index][2] + dz
        ]
        self.point_coords[index] = new_position
        self.update_point_display(index)
        print(f"Point {index+1} moved to: ({new_position[0]:.6f}, {new_position[1]:.6f}, {new_position[2]:.6f})")

    def move_point_screen(self, index, direction, step):
        move_vector = direction * step
        new_position = [
            self.point_coords[index][0] + move_vector[0],
            self.point_coords[index][1] + move_vector[1],
            self.point_coords[index][2] + move_vector[2]
        ]
        self.point_coords[index] = new_position
        self.update_point_display(index)
        print(f"Point {index+1} moved to: ({new_position[0]:.6f}, {new_position[1]:.6f}, {new_position[2]:.6f})")

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

    def update_all_points(self):
        for i in range(self.num_points):
            if self.point_actors[i]:
                self.update_point_display(i)

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

    def export_stl_with_new_origin(self):
        """
        点1を原点として、3Dモデルファイルを新しい座標系で保存する。
        法線の計算を改善し、品質を保証する。
        """
        if not self.stl_actor or not any(self.point_actors):
            print("STL model or points are not set.")
            return

        # Get the original file extension and path
        if hasattr(self, 'stl_file_path') and self.stl_file_path:
            original_ext = os.path.splitext(self.stl_file_path)[1].lower()
            default_name = self.stl_file_path  # Use same name for overwrite
        else:
            original_ext = '.stl'
            default_name = ""

        # Set file filter based on original file extension
        if original_ext == '.dae':
            file_filter = "COLLADA Files (*.dae);;STL Files (*.stl);;All Files (*)"
            default_ext = '.dae'
        else:
            file_filter = "STL Files (*.stl);;COLLADA Files (*.dae);;All Files (*)"
            default_ext = '.stl'

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save 3D Model File", default_name, file_filter)
        if not file_path:
            return

        # Add default extension if not present
        if not os.path.splitext(file_path)[1]:
            file_path += default_ext

        try:
            # 現在のSTLモデルのポリデータを取得
            poly_data = self.stl_actor.GetMapper().GetInput()

            # 最初に選択されているポイントを新しい原点として使用
            origin_index = next(i for i, actor in enumerate(self.point_actors) if actor and actor.GetVisibility())
            origin_point = self.point_coords[origin_index]

            # Step 1: 選択されたポイントを原点とする平行移動
            translation = vtk.vtkTransform()
            translation.Translate(-origin_point[0], -origin_point[1], -origin_point[2])

            # Step 2: カメラの向きに基づいて座標系を変更
            camera = self.renderer.GetActiveCamera()
            camera_direction = np.array(camera.GetDirectionOfProjection())
            camera_up = np.array(camera.GetViewUp())
            camera_right = np.cross(camera_direction, camera_up)

            # 座標軸の設定
            new_x = -camera_direction  # カメラの向きの逆方向をX軸に
            new_y = camera_right      # カメラの右方向をY軸に
            new_z = camera_up         # カメラの上方向をZ軸に

            # 正規直交基底を確保
            new_x = new_x / np.linalg.norm(new_x)
            new_y = new_y / np.linalg.norm(new_y)
            new_z = new_z / np.linalg.norm(new_z)

            # 変換行列の作成
            rotation_matrix = np.column_stack((new_x, new_y, new_z))
            
            # 直交性を確保
            U, _, Vh = np.linalg.svd(rotation_matrix)
            rotation_matrix = U @ Vh

            # VTKの回転行列に変換
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    vtk_matrix.SetElement(i, j, rotation_matrix[i, j])
            vtk_matrix.SetElement(3, 3, 1.0)

            # 回転変換を作成
            rotation = vtk.vtkTransform()
            rotation.SetMatrix(vtk_matrix)

            # Step 3: 変換を組み合わせる
            transform = vtk.vtkTransform()
            transform.PostMultiply()
            transform.Concatenate(translation)
            transform.Concatenate(rotation)

            # Step 4: 変換を適用
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(poly_data)
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            # Step 5: トライアングルフィルタを適用して面の向きを統一
            triangle_filter = vtk.vtkTriangleFilter()
            triangle_filter.SetInputData(transform_filter.GetOutput())
            triangle_filter.Update()

            # Step 6: クリーンフィルタを適用
            clean_filter = vtk.vtkCleanPolyData()
            clean_filter.SetInputData(triangle_filter.GetOutput())
            clean_filter.Update()

            # Step 7: 法線の再計算
            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.SetInputData(clean_filter.GetOutput())
            
            # 法線計算の設定
            normal_generator.SetFeatureAngle(60.0)  # 特徴エッジの角度閾値
            normal_generator.SetSplitting(False)    # エッジでの分割を無効化
            normal_generator.SetConsistency(True)   # 法線の一貫性を確保
            normal_generator.SetAutoOrientNormals(True)  # 法線の自動配向
            normal_generator.SetComputePointNormals(True)  # 頂点法線の計算
            normal_generator.SetComputeCellNormals(True)   # 面法線の計算
            normal_generator.SetFlipNormals(False)  # 法線の反転を無効化
            normal_generator.NonManifoldTraversalOn()  # 非マニフォールドの処理を有効化
            
            # 法線の計算を実行
            normal_generator.Update()

            # Step 8: 変換後のデータを取得
            transformed_poly_data = normal_generator.GetOutput()

            # Step 9: 出力の品質チェック
            if transformed_poly_data.GetNumberOfPoints() == 0:
                raise ValueError("The transformed model has no vertices.")

            # Determine file format from extension
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.dae':
                # Export as COLLADA using trimesh
                # Convert VTK PolyData to numpy arrays
                num_points = transformed_poly_data.GetNumberOfPoints()
                num_cells = transformed_poly_data.GetNumberOfCells()

                vertices = np.zeros((num_points, 3))
                for i in range(num_points):
                    vertices[i] = transformed_poly_data.GetPoint(i)

                faces = []
                for i in range(num_cells):
                    cell = transformed_poly_data.GetCell(i)
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
                stl_writer.SetInputData(transformed_poly_data)
                stl_writer.SetFileTypeToBinary()  # バイナリ形式で保存
                stl_writer.Write()

                print(f"STL file with corrected normals in the new coordinate system has been saved: {file_path}")

            # メッシュの品質情報を出力
            print(f"Number of vertices: {transformed_poly_data.GetNumberOfPoints()}")
            print(f"Number of faces: {transformed_poly_data.GetNumberOfCells()}")

            # Show success dialog
            file_name = os.path.basename(file_path)
            dir_path = os.path.dirname(file_path)

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("File Saved")
            msg.setText(f"Saved as {file_name} in {dir_path}")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

            # Show error dialog
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()

    def handle_set_reset(self):
        sender = self.sender()
        is_set = sender.text() == "Set Point"

        for i, checkbox in enumerate(self.point_checkboxes):
            if checkbox.isChecked():
                if is_set:
                    try:
                        new_coords = [float(self.point_inputs[i][j].text()) for j in range(3)]
                        if new_coords != self.point_coords[i]:
                            self.point_coords[i] = new_coords
                            self.update_point_display(i)
                            print(f"Point {i+1} set to: {new_coords}")
                        else:
                            print(f"Point {i+1} coordinates unchanged")
                    except ValueError:
                        print(f"Invalid input for Point {i+1}. Please enter valid numbers.")
                else:  # Reset
                    self.reset_point_to_origin(i)

        if not is_set:
            self.update_all_points_size()

        self.render_to_image()

    def get_mirrored_filename(self, original_path):
        dir_path = os.path.dirname(original_path)
        filename = os.path.basename(original_path)
        name, ext = os.path.splitext(filename)
        
        # ファイル名の先頭を確認して適切な新しいファイル名を生成
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
        
        return os.path.join(dir_path, new_name + ext)


    def export_mirror_stl_xml(self):
        """STLファイルをY軸でミラーリングし、対応するXMLファイルも生成する"""
        if not hasattr(self, 'stl_file_path') or not self.stl_file_path:
            print("No STL file has been loaded.")
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
                    existing_files.append(f"STL: {mirrored_stl_path}")
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
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    xml_data = tree.getroot()
                    print(f"Found and loaded XML file: {xml_path}")

                    # XMLから物理パラメータを取得
                    mass = float(xml_data.find(".//mass").get('value'))
                    volume = float(xml_data.find(".//volume").get('value'))
                    
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
                mesh.export(mirrored_stl_path)

            else:
                # Export as STL using VTK
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(mirrored_stl_path)
                writer.SetInputData(normal_generator.GetOutput())
                writer.Write()

            print("\nCalculating inertia tensor for mirrored model...")
            # イナーシャテンソルを計算
            inertia_tensor = self.calculate_inertia_tensor_for_mirrored(
                normal_generator.GetOutput(), mass, center_of_mass)

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
            <mass value="{mass:.12f}"/>
            <volume value="{volume:.12f}"/>
            {self.format_inertia_for_urdf(inertia_tensor)}
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
                            urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>
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

            urdf_content += f"""
    <joint>
        <axis xyz="{mirrored_axis}" />
    </joint>
</urdf_part>"""

            # XMLファイルを保存
            print(f"Saving XML to: {mirrored_xml_path}")
            with open(mirrored_xml_path, "w") as f:
                f.write(urdf_content)

            print("\nMirror export completed successfully:")
            file_type = "DAE file" if output_ext == '.dae' else "STL file"
            print(f"{file_type}: {mirrored_stl_path}")
            print(f"XML file: {mirrored_xml_path}")

        except Exception as e:
            print(f"\nAn error occurred during mirror export: {str(e)}")
            traceback.print_exc()
        

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
        # 体積を計算
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(poly_data)
        mass_properties.Update()
        total_volume = mass_properties.GetVolume()

        # 実際の質量から密度を逆算
        density = mass / total_volume
        print(f"Calculated density: {density:.6f} from mass: {mass:.6f} and volume: {total_volume:.6f}")

        # Y軸ミラーリングの変換行列
        mirror_matrix = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])

        inertia_tensor = np.zeros((3, 3))
        num_cells = poly_data.GetNumberOfCells()
        print(f"Processing {num_cells} triangles for inertia tensor calculation...")

        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                # 三角形の頂点を取得（重心を原点とした座標系で）
                points = [np.array(cell.GetPoints().GetPoint(j)) - np.array(center_of_mass) for j in range(3)]
                
                # 三角形の面積と法線ベクトルを計算
                v1 = points[1] - points[0]
                v2 = points[2] - points[0]
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                
                if area < 1e-10:  # 極小の三角形は無視
                    continue

                # 三角形の重心
                tri_centroid = np.mean(points, axis=0)

                # 三角形の局所的な慣性テンソルを計算
                covariance = np.zeros((3, 3))
                for p in points:
                    # 点をミラーリング
                    p = mirror_matrix @ p
                    r_squared = np.sum(p * p)
                    for a in range(3):
                        for b in range(3):
                            if a == b:
                                covariance[a, a] += (r_squared - p[a] * p[a]) * area / 12.0
                            else:
                                covariance[a, b] -= (p[a] * p[b]) * area / 12.0

                # ミラーリングされた重心
                tri_centroid = mirror_matrix @ tri_centroid
                
                # 平行軸の定理を適用
                r_squared = np.sum(tri_centroid * tri_centroid)
                parallel_axis_term = np.zeros((3, 3))
                for a in range(3):
                    for b in range(3):
                        if a == b:
                            parallel_axis_term[a, a] = r_squared * area
                        else:
                            parallel_axis_term[a, b] = tri_centroid[a] * tri_centroid[b] * area

                # 局所的な慣性テンソルと平行軸の項を合成
                local_inertia = covariance + parallel_axis_term
                
                # 全体の慣性テンソルに加算
                inertia_tensor += local_inertia

        # 密度を考慮して最終的な慣性テンソルを計算
        inertia_tensor *= density

        # Y軸反転による慣性テンソルの変換
        mirror_tensor = np.array([[1, -1, -1],
                                [-1, 1, 1],
                                [-1, 1, 1]])
        inertia_tensor = inertia_tensor * mirror_tensor

        # 数値誤差の処理
        threshold = 1e-10
        inertia_tensor[np.abs(inertia_tensor) < threshold] = 0.0

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

                    # チェックボックスをオンにする
                    self.point_checkboxes[i].setChecked(True)

                    # ポイントの表示を設定
                    if self.point_actors[i] is None:
                        self.point_actors[i] = vtk.vtkAssembly()
                        self.create_point_coordinate(self.point_actors[i], [0, 0, 0])

                    self.point_actors[i].SetPosition(self.point_coords[i])
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

        return points_with_data

    def _apply_color_from_xml(self, root):
        """XMLからカラー情報を適用"""
        color_element = root.find(".//material/color")
        if color_element is not None:
            rgba_str = color_element.get('rgba')
            if rgba_str:
                try:
                    r, g, b, _ = map(float, rgba_str.split())
                    
                    self.color_inputs[0].setText(f"{r:.3f}")
                    self.color_inputs[1].setText(f"{g:.3f}")
                    self.color_inputs[2].setText(f"{b:.3f}")
                    
                    self.update_color_sample()
                    
                    if self.stl_actor:
                        self.stl_actor.GetProperty().SetColor(r, g, b)
                        self.render_to_image()
                    
                    print(f"Material color loaded and applied: R={r:.3f}, G={g:.3f}, B={b:.3f}")
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
                            r, g, b, _ = map(float, rgba_str.split())
                            # 色情報を入力フィールドに設定
                            self.color_inputs[0].setText(f"{r:.3f}")
                            self.color_inputs[1].setText(f"{g:.3f}")
                            self.color_inputs[2].setText(f"{b:.3f}")
                            
                            # カラーサンプルの更新
                            self.update_color_sample()
                            
                            # STLモデルに色を適用
                            if hasattr(self, 'stl_actor') and self.stl_actor:
                                self.stl_actor.GetProperty().SetColor(r, g, b)
                                self.render_to_image()
                            
                            has_parameters = True
                            print(f"Loaded color: R={r:.3f}, G={g:.3f}, B={b:.3f}")
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
                            r, g, b, _ = map(float, rgba_str.split())
                            
                            # インプットフィールドに値を設定
                            self.color_inputs[0].setText(f"{r:.3f}")
                            self.color_inputs[1].setText(f"{g:.3f}")
                            self.color_inputs[2].setText(f"{b:.3f}")
                            
                            # カラーサンプルを更新
                            self.update_color_sample()
                            
                            # STLモデルに色を適用
                            self.stl_actor.GetProperty().SetColor(r, g, b)
                            self.render_to_image()
                            
                            print(f"Material color loaded and applied: R={r:.3f}, G={g:.3f}, B={b:.3f}")
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
        """STLファイルとXMLファイルを一緒に読み込む"""
        try:
            stl_path, _ = QFileDialog.getOpenFileName(self, "Open STL File", "", "STL Files (*.stl)")
            if not stl_path:
                return

            # STLファイルを読み込む
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
        フォルダ内の'l_'または'L_'で始まるSTL/DAEファイルを処理し、
        対応する'r_'または'R_'ファイルを生成する。
        既存のファイルは上書きせずスキップする。
        """
        try:
            # フォルダ選択ダイアログを表示
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder for Bulk Conversion")
            if not folder_path:
                return

            print(f"Selected folder: {folder_path}")

            # 処理したファイルの数を追跡
            processed_count = 0
            skipped_count = 0

            # フォルダ内のすべてのSTL/DAEファイルを検索
            for file_name in os.listdir(folder_path):
                if file_name.lower().startswith(('l_', 'L_')) and file_name.lower().endswith(('.stl', '.dae')):
                    stl_path = os.path.join(folder_path, file_name)

                    # 新しいファイル名を生成
                    new_name = 'R_' + file_name[2:] if file_name.startswith('L_') else 'r_' + file_name[2:]
                    new_name_without_ext = os.path.splitext(new_name)[0]
                    new_stl_path = os.path.join(folder_path, new_name)
                    new_xml_path = os.path.splitext(new_stl_path)[0] + '.xml'

                    # 既存ファイルのチェック
                    if os.path.exists(new_stl_path) or os.path.exists(new_xml_path):
                        print(f"Skipping {file_name} - Target file(s) already exist")
                        skipped_count += 1
                        continue

                    print(f"Processing: {stl_path}")

                    try:
                        # ファイル拡張子を確認
                        file_ext = os.path.splitext(stl_path)[1].lower()

                        if file_ext == '.dae':
                            # DAEファイルをtrimeshで読み込む
                            mesh = trimesh.load(stl_path, force='mesh')

                            # trimeshをVTK PolyDataに変換
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

                            # Y軸反転の変換を設定
                            transform = vtk.vtkTransform()
                            transform.Scale(1, -1, 1)

                            # 頂点を変換
                            transformer = vtk.vtkTransformPolyDataFilter()
                            transformer.SetInputData(poly_data)
                            transformer.SetTransform(transform)
                            transformer.Update()

                        else:
                            # STLファイルを読み込む
                            reader = vtk.vtkSTLReader()
                            reader.SetFileName(stl_path)
                            reader.Update()

                            # Y軸反転の変換を設定
                            transform = vtk.vtkTransform()
                            transform.Scale(1, -1, 1)

                            # 頂点を変換
                            transformer = vtk.vtkTransformPolyDataFilter()
                            transformer.SetInputConnection(reader.GetOutputPort())
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
                        if os.path.exists(xml_path):
                            try:
                                tree = ET.parse(xml_path)
                                xml_data = tree.getroot()
                                print(f"Found and loaded XML file: {xml_path}")

                                # XMLから物理パラメータを取得
                                mass = float(xml_data.find(".//mass").get('value'))
                                volume = float(xml_data.find(".//volume").get('value'))
                                
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

                        # ミラー化したファイルを保存
                        output_ext = os.path.splitext(new_stl_path)[1].lower()

                        if output_ext == '.dae':
                            # DAEファイルとして保存
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

                            # trimeshオブジェクトを作成してエクスポート
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                            mesh.export(new_stl_path)

                        else:
                            # STLファイルとして保存
                            writer = vtk.vtkSTLWriter()
                            writer.SetFileName(new_stl_path)
                            writer.SetInputData(normal_generator.GetOutput())
                            writer.Write()

                        # イナーシャテンソルを計算
                        inertia_tensor = self.calculate_inertia_tensor_for_mirrored(
                            normal_generator.GetOutput(), mass, center_of_mass)

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
            <mass value="{mass:.12f}"/>
            <volume value="{volume:.12f}"/>
            {self.format_inertia_for_urdf(inertia_tensor)}
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
                                        urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>
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
                        print(f"Converted: {file_name} -> {new_name}")
                        print(f"Created XML: {new_xml_path}")

                    except Exception as e:
                        print(f"Error processing file {file_name}: {str(e)}")
                        traceback.print_exc()
                        continue

            # 処理完了メッセージ
            if processed_count > 0 or skipped_count > 0:
                print(f"\nBulk conversion completed.")
                print(f"Processed: {processed_count} files")
                print(f"Skipped: {skipped_count} files (already exist)")
            else:
                print("\nNo files were processed. Make sure there are STL files with 'l_' or 'L_' prefix in the selected folder.")

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

    def update_color_sample(self):
        """カラーサンプルの表示を更新"""
        try:
            rgb_values = [min(255, max(0, int(float(input.text()) * 255))) 
                        for input in self.color_inputs]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_values[0]},{rgb_values[1]},{rgb_values[2]}); "
                f"border: 1px solid black;"
            )

            # STLモデルに色を適用
            if hasattr(self, 'stl_actor') and self.stl_actor:
                rgb_normalized = [v / 255.0 for v in rgb_values]
                self.stl_actor.GetProperty().SetColor(*rgb_normalized)
                self.render_to_image()

        except ValueError:
            pass

    def show_color_picker(self):
        """カラーピッカーを表示"""
        try:
            current_color = QtGui.QColor(
                *[min(255, max(0, int(float(input.text()) * 255))) 
                for input in self.color_inputs]
            )
        except ValueError:
            current_color = QtGui.QColor(255, 255, 255)
        
        color = QtWidgets.QColorDialog.getColor(
            initial=current_color,
            parent=self,
            options=QtWidgets.QColorDialog.DontUseNativeDialog
        )
        
        if color.isValid():
            for i, component in enumerate([color.red(), color.green(), color.blue()]):
                self.color_inputs[i].setText(f"{component / 255:.3f}")
            
            if self.current_node:
                self.current_node.node_color = [
                    color.red() / 255.0,
                    color.green() / 255.0,
                    color.blue() / 255.0
                ]
                self.apply_color_to_stl()

    def apply_color_to_stl(self):
        """選択された色をSTLモデルに適用"""
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No STL model has been loaded.")
            return
        
        try:
            # RGB値を取得（0-1の範囲）
            rgb_values = [float(input.text()) for input in self.color_inputs]
            
            # 値の範囲チェック
            rgb_values = [max(0.0, min(1.0, value)) for value in rgb_values]
            
            # STLモデルの色を変更
            self.stl_actor.GetProperty().SetColor(*rgb_values)
            self.render_to_image()
            print(f"Applied color: RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")
            
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

        # STLファイルパスを表示
        stl_label = QLabel(f"STL: {stl_path}")
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

def signal_handler(sig, frame):
    print("Ctrl+C detected, closing application...")
    QApplication.instance().quit()

if __name__ == "__main__":

    # Ctrl+Cのシグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    # タイマーを設定してシグナルを処理できるようにする
    timer = QTimer()
    timer.start(500)  # 500ミリ秒ごとにイベントループを中断
    timer.timeout.connect(lambda: None)  # ダミー関数を接続

    try:
        sys.exit(app.exec())
    except SystemExit:
        print("Exiting application...")
