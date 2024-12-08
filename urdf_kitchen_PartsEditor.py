"""
File Name: urdf_kitchen_PartsEditor.py
Description: A Python script for configuring connection points of parts for urdf_kitchen_Assembler.py.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Version     : 0.0.1
License     : MIT License
URL         : https://github.com/Ninagawa123/URDF_kitchen_beta
Copyright (c) 2024 Ninagawa123

python3.9
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
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from Qt import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QVBoxLayout, QWidget, 
    QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QLabel, QGridLayout,
    QTextEdit, QButtonGroup, QRadioButton, QColorDialog, QDialog
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextOption, QColor, QPalette

# pip install numpy
# pip install PySide6
# pip install vtk
# pip install NodeGraphQt

def apply_dark_theme(self):
    """シックなダークテーマを適用"""
    # パレットの設定
    palette = self.palette()
    # メインウィンドウ背景：柔らかいダークグレー
    palette.setColor(QPalette.Window, QColor(70, 80, 80))
    # テキスト：ダークグレー
    palette.setColor(QPalette.WindowText, QColor(240, 240, 237))  # この行を修正
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
    self.setPalette(palette)

    # VTKビューポートの背景色をシックなグレーに設定
    if hasattr(self, 'renderer'):
        self.renderer.SetBackground(0.05, 0.05, 0.07)  # よりソフトなダークグレー

    # 追加のスタイル設定
    self.setStyleSheet("""
        QMainWindow {
            background-color: #404244;
        }
        QPushButton {
            background-color: #F0F0ED;
            border: 1px solid #BBBBB7;
            border-radius: 2px;
            padding: 2px 2px;
            color: #333333;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #E6E6E3;
            border: 1px solid #AAAAAA;
        }
        QPushButton:pressed {
            background-color: #DDDDD9;
            padding-top: 4px;
            padding-bottom: 4px;
        }
        QLineEdit {
            background-color: #F0F0ED;
            border: 1px solid #BBBBB7;
            color: #F0F0ED;
            padding: 1px;  # パディングを小さく
            border-radius: 2px;
            min-height: 12px;  # 最小の高さを設定
            max-height: 12px;  # 最大の高さを設定
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
            background-color: #808487;
            border: 1px solid #666666;
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
            background-color: #808487;
            border: 1px solid #666666;
        }
    """)

    # ファイルダイアログのスタイル
    self.setStyleSheet(self.styleSheet() + """
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
        if not self.GetInteractor():
            return
        renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        if not renderer:
            return
        actors = renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextItem()

        while actor:
            # STLアクターの場合のみ表示モードを切り替える
            if actor == self.parent.stl_actor:
                if actor.GetProperty().GetRepresentation() == vtk.VTK_SURFACE:
                    actor.GetProperty().SetRepresentationToWireframe()
                else:
                    actor.GetProperty().SetRepresentationToSurface()
            actor = actors.GetNextItem()
        
        self.GetInteractor().GetRenderWindow().Render()
    
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
        self.setWindowTitle("URDF Kitchen - PartsEditor v0.0.1 -")
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

        self.com_actor = None  # 重心アクターを追跡するための新しい属性

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)  # 垂直方向のレイアウトに変更

        # ファイル名表示用のラベル
        self.file_name_label = QLabel("File:")
        self.file_name_value = QLabel("No file loaded")
        self.file_name_value.setWordWrap(True)  # 長いパスの場合に折り返すように設定
        file_name_layout = QVBoxLayout()  # 垂直レイアウトに変更
        file_name_layout.addWidget(self.file_name_label)
        file_name_layout.addWidget(self.file_name_value)
        main_layout.addLayout(file_name_layout)

        # 水平方向のレイアウトを作成（左側のUIと右側の3D表示用）
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # 左側のUI用ウィジェットとレイアウト
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        content_layout.addWidget(left_widget, 1)  # stretch factorを1に設定

        # 右側のVTKウィジェット
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        content_layout.addWidget(self.vtk_widget, 4)  # stretch factorを4に設定（UIより広いスペースを確保）
        
        self.setup_ui()
        self.setup_vtk()
        self.add_instruction_text()  # 説明テキストを追加

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

        self.rotation_types = {'yaw': 0, 'pitch': 1, 'roll': 2}

        self.setup_camera()
        self.axes_widget = self.add_axes_widget() # 座標軸の追加
        self.add_axes()  # 既存のadd_axesメソッドも保持
        
        self.render_window.Render()
        self.render_window_interactor.Initialize()
        self.add_axes()
        
        self.render_window.Render()
        self.render_window_interactor.Initialize()
        self.vtk_widget.GetRenderWindow().AddObserver("ModifiedEvent", self.update_all_points_size)

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
        self.load_button = QPushButton("Load STL")
        self.load_button.clicked.connect(self.load_stl_file)
        first_row.addWidget(self.load_button)

        # Load XMLボタン
        self.load_xml_button = QPushButton("ImportXML")
        self.load_xml_button.clicked.connect(self.load_xml_file)
        first_row.addWidget(self.load_xml_button)

        # first_row_layout を first_row_widget にセット
        first_row_widget.setLayout(first_row_layout)

        # Load STL with XMLボタン
        self.load_stl_xml_button = QPushButton("Load STL with XML")
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
        self.export_mirror_button = QPushButton("Export mirror STL with XML")
        self.export_mirror_button.clicked.connect(self.export_mirror_stl_xml)
        button_layout.addWidget(self.export_mirror_button)

        # 一括変換ボタン
        self.bulk_convert_button = QPushButton("Batch convert \"l_\" to \"r_\" in /meshes")
        self.bulk_convert_button.clicked.connect(self.bulk_convert_l_to_r)
        button_layout.addWidget(self.bulk_convert_button)

        # Export STLボタン
        self.export_stl_button = QPushButton("Save STL (Point 1 as origin)")
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

        # プロパティの設定
        properties = [
            ("Volume (m^3):", "volume"),
            ("Density (kg/m^3):", "density"),
            ("Mass (kg):", "mass"),
            ("Center of Mass:", "com"),
            ("Inertia Tensor:", "inertia_tensor")
        ]

        # ... checkboxes and inputs setup ...
        for i, (label_text, prop_name) in enumerate(properties):
            if prop_name != "inertia_tensor":
                checkbox = QCheckBox()
                setattr(self, f"{prop_name}_checkbox", checkbox)
                
                label = QLabel(label_text)
                input_field = QLineEdit("0.000000")
                setattr(self, f"{prop_name}_input", input_field)
                
                grid_layout.addWidget(checkbox, i, 0)
                grid_layout.addWidget(label, i, 1)
                grid_layout.addWidget(input_field, i, 2)
            else:
                label = QLabel(label_text)
                grid_layout.addWidget(label, i, 1)

        # Inertia Tensor setup
        self.inertia_tensor_input = QTextEdit()
        self.inertia_tensor_input.setReadOnly(True)
        self.inertia_tensor_input.setFixedHeight(40)
        self.inertia_tensor_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inertia_tensor_input.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inertia_tensor_input.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        grid_layout.addWidget(QLabel("Inertia Tensor:"), len(properties) - 1, 1)
        grid_layout.addWidget(self.inertia_tensor_input, len(properties) - 1, 2)
        
        # フォントサイズを8ptに設定
        font = self.inertia_tensor_input.font()
        font.setPointSize(10)
        self.inertia_tensor_input.setFont(font)

        # デフォルト値の設定
        self.density_input.setText("1.000000")

        # Calculateボタンの前にスペーサーを追加
        pre_calculate_spacer = QWidget()
        pre_calculate_spacer.setFixedHeight(2)  # 2ピクセルの空間
        grid_layout.addWidget(pre_calculate_spacer, len(properties), 0, 1, 3)

        # Calculate ボタン
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_and_update_properties)
        grid_layout.addWidget(self.calculate_button, len(properties) + 1, 1, 1, 2)

        # スペーサーを追加（小さな空間）
        spacer = QWidget()
        spacer.setFixedHeight(2)  # 8ピクセルの空間
        grid_layout.addWidget(spacer, len(properties) + 2, 0, 1, 3)

        # Axis layout
        axis_layout = QHBoxLayout()
        

        # ラジオボタンのグループを作成
        self.axis_group = QButtonGroup(self)
        axis_label = QLabel("Axis:")
        axis_layout.addWidget(axis_label)

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
        
        grid_layout.addLayout(axis_layout, len(properties) + 3, 0, 1, 3)

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
        grid_layout.addLayout(color_layout, len(properties) + 4, 0, 1, 3)

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
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.05)
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        self.render_window_interactor = self.render_window.GetInteractor()

        style = CustomInteractorStyle(self)
        self.render_window_interactor.SetInteractorStyle(style)

    def setup_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(self.absolute_origin[0] + self.initial_camera_position[0],
                           self.absolute_origin[1] + self.initial_camera_position[1],
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

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(self.absolute_origin[0] + self.initial_camera_position[0],
                           self.absolute_origin[1] + self.initial_camera_position[1],
                           self.absolute_origin[2] + self.initial_camera_position[2])
        camera.SetFocalPoint(*self.absolute_origin)
        camera.SetViewUp(*self.initial_camera_view_up)
        
        self.camera_rotation = [0, 0, 0]
        self.current_rotation = 0
        
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()
        self.update_all_points()

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
        
        self.render_window.Render()

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
        
        self.render_window.Render()

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
        
        self.render_window.Render()

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
        try:
            # 現在のチェック状態とテキスト値を取得
            properties = {
                'volume': {
                    'checked': self.volume_checkbox.isChecked(),
                    'value': float(self.volume_input.text()) if self.volume_checkbox.isChecked() else None
                },
                'density': {
                    'checked': self.density_checkbox.isChecked(),
                    'value': float(self.density_input.text()) if self.density_checkbox.isChecked() else None
                },
                'mass': {
                    'checked': self.mass_checkbox.isChecked(),
                    'value': float(self.mass_input.text()) if self.mass_checkbox.isChecked() else None
                }
            }

            # STLからの体積計算（必要な場合に使用）
            stl_volume = None
            if (not properties['volume']['checked'] and 
                ((properties['density']['checked'] and not properties['mass']['checked']) or
                (not properties['density']['checked'] and properties['mass']['checked']))):
                if hasattr(self, 'stl_actor') and self.stl_actor:
                    poly_data = self.stl_actor.GetMapper().GetInput()
                    mass_properties = vtk.vtkMassProperties()
                    mass_properties.SetInputData(poly_data)
                    stl_volume = mass_properties.GetVolume()
                    properties['volume']['value'] = stl_volume
                    self.volume_input.setText(f"{stl_volume:.12f}")

            # 値の計算
            # Case 1: VolumeとMassがチェックされている場合
            if properties['volume']['checked'] and properties['mass']['checked']:
                properties['density']['value'] = properties['mass']['value'] / properties['volume']['value']
                self.density_input.setText(f"{properties['density']['value']:.12f}")

            # Case 2: VolumeとDensityがチェックされている場合
            elif properties['volume']['checked'] and properties['density']['checked']:
                properties['mass']['value'] = properties['volume']['value'] * properties['density']['value']
                self.mass_input.setText(f"{properties['mass']['value']:.12f}")

            # Case 3: DensityとMassがチェックされている場合
            elif properties['density']['checked'] and properties['mass']['checked']:
                properties['volume']['value'] = properties['mass']['value'] / properties['density']['value']
                self.volume_input.setText(f"{properties['volume']['value']:.12f}")

            # Case 4: 単一のチェックケース
            elif stl_volume is not None:
                if properties['density']['checked']:
                    properties['mass']['value'] = stl_volume * properties['density']['value']
                    self.mass_input.setText(f"{properties['mass']['value']:.12f}")
                elif properties['mass']['checked']:
                    properties['density']['value'] = properties['mass']['value'] / stl_volume
                    self.density_input.setText(f"{properties['density']['value']:.12f}")

            # 重心の更新
            # COMチェックボックスの状態に関わらず常に重心を再計算
            self.calculate_center_of_mass()

            # 慣性テンソルを常に更新
            self.calculate_inertia_tensor()

        except (ValueError, ZeroDivisionError) as e:
            print(f"An error occurred during calculation: {str(e)}")
            return None

        return properties

    def calculate_center_of_mass(self):
        """
        重心を計算して表示する。
        チェックボックスがオンの場合は入力値を使用し、オフの場合は計算値を使用する。
        """
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No STL model has been loaded.")
            return

        # 重心の座標を取得（チェックボックスの状態に応じて）
        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            try:
                # 入力テキストから座標を取得
                com_text = self.com_input.text().strip('()').split(',')
                if len(com_text) != 3:
                    raise ValueError("Invalid format: Requires 3 coordinates")
                center_of_mass = [float(x.strip()) for x in com_text]
                print(f"Using manual Center of Mass: {center_of_mass}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing Center of Mass input: {e}")
                return None
        else:
            # STLから重心を計算
            poly_data = self.stl_actor.GetMapper().GetInput()
            com_filter = vtk.vtkCenterOfMass()
            com_filter.SetInputData(poly_data)
            com_filter.SetUseScalarsAsWeights(False)
            com_filter.Update()
            center_of_mass = com_filter.GetCenter()
            # 計算された値を入力欄に設定
            self.com_input.setText(f"({center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f})")
            print(f"Calculated Center of Mass: {center_of_mass}")

        # 既存の重心アクターを削除
        if hasattr(self, 'com_actor') and self.com_actor:
            self.renderer.RemoveActor(self.com_actor)

        # 重心を可視化（赤い点）
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center_of_mass)
        sphere.SetRadius(self.calculate_sphere_radius() * 0.5)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        self.com_actor = vtk.vtkActor()
        self.com_actor.SetMapper(mapper)
        self.com_actor.GetProperty().SetColor(1, 0, 0)  # 赤色
        self.com_actor.GetProperty().SetOpacity(0.7)

        self.renderer.AddActor(self.com_actor)
        self.render_window.Render()

        return center_of_mass

    def calculate_inertia_tensor(self):
        """
        STLモデルの慣性テンソルを計算する。
        重心位置を考慮し、正確なイナーシャを算出する。
        """
        if not hasattr(self, 'stl_actor') or not self.stl_actor:
            print("No STL model is loaded.")
            return None

        # ポリデータを取得
        poly_data = self.stl_actor.GetMapper().GetInput()

        # 体積と質量を取得
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(poly_data)
        mass_properties.Update()
        volume = mass_properties.GetVolume()
        density = float(self.density_input.text())
        mass = volume * density

        # 重心の取得
        if hasattr(self, 'com_checkbox') and self.com_checkbox.isChecked():
            try:
                com_text = self.com_input.text().strip('()').split(',')
                center_of_mass = [float(x.strip()) for x in com_text]
                print(f"Using fixed Center of Mass: {center_of_mass}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing Center of Mass input: {e}")
                return None
        else:
            com_filter = vtk.vtkCenterOfMass()
            com_filter.SetInputData(poly_data)
            com_filter.SetUseScalarsAsWeights(False)
            com_filter.Update()
            center_of_mass = com_filter.GetCenter()
            print(f"Calculated Center of Mass: {center_of_mass}")

        # 慣性テンソルの計算
        inertia_tensor = np.zeros((3, 3))
        num_cells = poly_data.GetNumberOfCells()
        total_volume = 0.0

        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                # 三角形の頂点を取得
                points = [np.array(cell.GetPoints().GetPoint(j)) for j in range(3)]
                
                # 重心からの相対位置を計算
                rel_points = [p - np.array(center_of_mass) for p in points]
                
                # 三角形の面積を計算
                v1 = rel_points[1] - rel_points[0]
                v2 = rel_points[2] - rel_points[0]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                
                # 三角形の重心を計算
                tri_com = sum(rel_points) / 3.0
                
                # 三角形の体積要素を計算（面積 × 厚さの概算）
                thickness = 0.001  # 仮の厚さ
                elem_volume = area * thickness
                total_volume += elem_volume
                
                # 平行軸の定理を使用して慣性テンソルを計算
                for p in rel_points:
                    r_squared = np.sum(p * p)
                    
                    # 対角成分
                    inertia_tensor[0, 0] += elem_volume * (r_squared - p[0] * p[0])
                    inertia_tensor[1, 1] += elem_volume * (r_squared - p[1] * p[1])
                    inertia_tensor[2, 2] += elem_volume * (r_squared - p[2] * p[2])
                    
                    # 非対角成分
                    inertia_tensor[0, 1] -= elem_volume * p[0] * p[1]
                    inertia_tensor[0, 2] -= elem_volume * p[0] * p[2]
                    inertia_tensor[1, 2] -= elem_volume * p[1] * p[2]

        # 体積で正規化し、質量を考慮
        scale_factor = mass / total_volume
        inertia_tensor *= scale_factor

        # 対称性を保証
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]

        print("Inertia_tensor:")
        print(inertia_tensor)

        # URDF形式の慣性テンソル文字列を生成
        urdf_inertia = self.format_inertia_for_urdf(inertia_tensor)

        # UIを更新
        if hasattr(self, 'inertia_tensor_input'):
            self.inertia_tensor_input.setText(urdf_inertia)
            self.com_input.setText(f"({center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f})")
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
        urdf_file_path, _ = QFileDialog.getSaveFileName(self, "Save URDF File", os.path.join(
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
                com_text = self.com_input.text().strip('()').split(',')
                com_values = [float(x.strip()) for x in com_text]
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

    def format_inertia_for_urdf(self, inertia_tensor):
        # 値が非常に小さい場合は0とみなす閾値
        threshold = 1e-10

        ixx = inertia_tensor[0][0] if abs(inertia_tensor[0][0]) > threshold else 0
        iyy = inertia_tensor[1][1] if abs(inertia_tensor[1][1]) > threshold else 0
        izz = inertia_tensor[2][2] if abs(inertia_tensor[2][2]) > threshold else 0
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

        self.render_window.Render()

    def load_stl_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open STL File", "", "STL Files (*.stl)")
        if file_path:
            self.file_name_value.setText(file_path)
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
        self.file_name_value.setText(file_path)
        
        # レンダリングを強制的に更新
        self.render_window.Render()
        
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
        self.render_window.Render()

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
        self.target_rotation = (self.current_rotation + angle) % 360
        self.rotation_per_frame = angle / self.total_animation_frames
        self.animation_frames = 0
        self.current_rotation_type = self.rotation_types[rotation_type]
        self.animation_timer.start(1000 // 60)
        self.camera_rotation[self.rotation_types[rotation_type]] += angle
        self.camera_rotation[self.rotation_types[rotation_type]] %= 360

    def animate_rotation(self):
        self.animation_frames += 1
        if self.animation_frames > self.total_animation_frames:
            self.animation_timer.stop()
            self.current_rotation = self.target_rotation
            return

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

        rotation_matrix = vtk.vtkTransform()
        rotation_matrix.Translate(*focal_point)
        rotation_matrix.RotateWXYZ(self.rotation_per_frame, axis)
        rotation_matrix.Translate(*[-x for x in focal_point])

        new_position = rotation_matrix.TransformPoint(position)
        new_up = rotation_matrix.TransformVector(view_up)

        camera.SetPosition(new_position)
        camera.SetViewUp(new_up)

        self.render_window.Render()

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
        
        self.render_window.Render()

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
        self.render_window.Render()

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
        self.render_window.Render()

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
        点1を原点として、STLファイルを新しい座標系で保存する。
        法線の計算を改善し、品質を保証する。
        """
        if not self.stl_actor or not any(self.point_actors):
            print("STL model or points are not set.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save STL File", "", "STL Files (*.stl)")
        if not file_path:
            return

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

            # STLファイルとして保存
            stl_writer = vtk.vtkSTLWriter()
            stl_writer.SetFileName(file_path)
            stl_writer.SetInputData(transformed_poly_data)
            stl_writer.SetFileTypeToBinary()  # バイナリ形式で保存
            stl_writer.Write()

            print(f"STL file with corrected normals in the new coordinate system has been saved: {file_path}")
            
            # メッシュの品質情報を出力
            print(f"Number of vertices: {transformed_poly_data.GetNumberOfPoints()}")
            print(f"Number of faces: {transformed_poly_data.GetNumberOfCells()}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

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

        self.render_window.Render()




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
        if not hasattr(self, 'stl_file_path') or not self.stl_file_path:
            print("No STL file has been loaded.")
            return

        try:
            # 元のファイルのパスとファイル名を取得
            original_dir = os.path.dirname(self.stl_file_path)
            original_filename = os.path.basename(self.stl_file_path)
            name, ext = os.path.splitext(original_filename)

            # 新しいファイル名を生成
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

            # ミラー化したSTLのパスを設定
            mirrored_stl_path = os.path.join(original_dir, new_name + ext)

            # STLの読み込みとミラー化
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
            # まず連結性を確認
            connect = vtk.vtkPolyDataConnectivityFilter()
            connect.SetInputData(transform_filter.GetOutput())
            connect.Update()

            # 法線の一貫性を確保
            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.SetInputData(connect.GetOutput())
            normal_generator.ConsistencyOn()
            normal_generator.AutoOrientNormalsOn()
            normal_generator.ComputeCellNormalsOn()
            normal_generator.ComputePointNormalsOn()
            normal_generator.SplittingOff()
            normal_generator.NonManifoldTraversalOn()
            normal_generator.Update()

            # 法線が修正されたデータを取得
            mirrored_poly_data = normal_generator.GetOutput()

            # XMLファイルを確認し読み込む
            xml_path = os.path.splitext(self.stl_file_path)[0] + '.xml'
            xml_data = None
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    xml_data = tree.getroot()
                    print(f"Found and loaded XML file: {xml_path}")
                except ET.ParseError:
                    print(f"Error parsing XML file: {xml_path}")

            # 物理プロパティを計算（質量を保持）
            volume, mass, center_of_mass, inertia_tensor = self.process_mirror_properties(
                xml_data,
                mirrored_poly_data,
                density=float(self.density_input.text())
            )

            # ミラー化したSTLを保存
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(mirrored_stl_path)
            writer.SetInputData(mirrored_poly_data)
            writer.Write()

            # 新しいXMLファイルのパスを生成
            new_xml_path = os.path.splitext(mirrored_stl_path)[0] + '.xml'

            # 色情報を取得
            try:
                if xml_data is not None:
                    color_element = xml_data.find(".//material/color")
                    if color_element is not None:
                        rgba_str = color_element.get('rgba', "1.0 1.0 1.0 1.0")
                        try:
                            r, g, b, _ = map(float, rgba_str.split())
                            hex_color = '#{:02X}{:02X}{:02X}'.format(
                                int(r * 255), int(g * 255), int(b * 255))
                        except (ValueError, IndexError):
                            hex_color = '#FFFFFF'
                            rgba_str = "1.0 1.0 1.0 1.0"
                    else:
                        rgb_values = [float(input.text()) for input in self.color_inputs]
                        hex_color = '#{:02X}{:02X}{:02X}'.format(
                            int(rgb_values[0] * 255),
                            int(rgb_values[1] * 255),
                            int(rgb_values[2] * 255)
                        )
                        rgba_str = f"{rgb_values[0]:.6f} {rgb_values[1]:.6f} {rgb_values[2]:.6f} 1.0"
                else:
                    rgb_values = [float(input.text()) for input in self.color_inputs]
                    hex_color = '#{:02X}{:02X}{:02X}'.format(
                        int(rgb_values[0] * 255),
                        int(rgb_values[1] * 255),
                        int(rgb_values[2] * 255)
                    )
                    rgba_str = f"{rgb_values[0]:.6f} {rgb_values[1]:.6f} {rgb_values[2]:.6f} 1.0"
            except ValueError:
                hex_color = '#FFFFFF'
                rgba_str = "1.0 1.0 1.0 1.0"

            # XMLファイルの内容を生成
            urdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urdf_part>
    <material name="{hex_color}">
        <color rgba="{rgba_str}" />
    </material>
    <link name="{new_name}">
        <visual>
            <material name="{hex_color}" />
        </visual>
        <volume value="{volume:.12f}"/>
        <mass value="{mass:.12f}"/>
        {self.format_inertia_for_urdf(inertia_tensor)}</link>"""

            # チェックされているポイントについてpoint要素を追加
            if xml_data is not None:
                points = xml_data.findall('.//point')
                for point in points:
                    xyz_element = point.find('point_xyz')
                    if xyz_element is not None and xyz_element.text:
                        try:
                            x, y, z = map(float, xyz_element.text.strip().split())
                            mirrored_y = -y  # Y座標のみ反転
                            point_name = point.get('name', '').replace('l_', 'r_').replace('L_', 'R_')
                            urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>
    </point>"""
                        except ValueError:
                            print(f"Error processing point coordinates in XML")

            # 選択されている軸に応じてjoint要素を追加
            axis_options = ["1 0 0", "0 1 0", "0 0 1"]
            checked_id = self.axis_group.checkedId()
            if 0 <= checked_id < len(axis_options):
                axis_vector = axis_options[checked_id]
            else:
                axis_vector = "1 0 0"  # デフォルト値

            urdf_content += f"""
    <joint>
        <axis xyz="{axis_vector}" />
    </joint>
</urdf_part>"""

            # XMLファイルを保存
            with open(new_xml_path, "w") as f:
                f.write(urdf_content)

            # 結果ダイアログを表示
            dialog = ResultDialog(mirrored_stl_path, new_xml_path, self)
            dialog.exec()

            print(f"Mirror export completed successfully:")
            print(f"STL file: {mirrored_stl_path}")
            print(f"XML file: {new_xml_path}")

        except Exception as e:
            print(f"An error occurred during mirror export: {str(e)}")
            traceback.print_exc()

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
                        self.render_window.Render()
                    
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
        self.render_window.Render()

    def load_parameters_from_xml(self, root):
        """XMLからパラメータを読み込んで設定する共通処理"""
        try:
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
                                self.render_window.Render()
                            
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
                    self.com_input.setText(f"({x:.6f}, {y:.6f}, {z:.6f})")
                    print(f"Loaded center of mass: ({x:.6f}, {y:.6f}, {z:.6f})")
                    has_parameters = True
                except ValueError as e:
                    print(f"Error parsing center of mass values: {e}")

            # 慣性テンソルの設定
            inertia_element = root.find(".//inertia")
            if inertia_element is not None:
                inertia_str = ET.tostring(inertia_element, encoding='unicode')
                self.inertia_tensor_input.setText(inertia_str)
                has_parameters = True

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
                            self.render_window.Render()
                            
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
                self.render_window.Render()

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
            self.render_window.Render()

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
        try:
            # フォルダ選択ダイアログを表示
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder for Bulk Conversion")
            if not folder_path:
                return

            print(f"Selected folder: {folder_path}")

            # 処理したファイルの数を追跡
            processed_count = 0

            # フォルダ内のすべてのSTLファイルを検索
            for file_name in os.listdir(folder_path):
                if file_name.lower().startswith(('l_', 'L_')) and file_name.lower().endswith('.stl'):
                    stl_path = os.path.join(folder_path, file_name)
                    print(f"Processing: {stl_path}")

                    try:
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

                        # 頂点の順序を反転して法線を修正
                        reverse = vtk.vtkReverseSense()
                        reverse.SetInputConnection(transformer.GetOutputPort())
                        reverse.ReverseCellsOn()
                        reverse.ReverseNormalsOn()
                        reverse.Update()

                        # 対応するXMLファイルを探す
                        xml_path = os.path.splitext(stl_path)[0] + '.xml'
                        xml_data = None
                        if os.path.exists(xml_path):
                            try:
                                tree = ET.parse(xml_path)
                                xml_data = tree.getroot()
                                print(f"Found corresponding XML file: {xml_path}")
                            except ET.ParseError:
                                print(f"Error parsing XML file: {xml_path}")
                                xml_data = None

                        # 物理プロパティを計算（質量を保持）
                        volume, mass, center_of_mass, inertia_tensor = self.process_mirror_properties(
                            xml_data,
                            reverse.GetOutput(),
                            density=float(self.density_input.text())
                        )

                        # 新しいファイル名を生成
                        new_name = 'R_' + file_name[2:] if file_name.startswith('L_') else 'r_' + file_name[2:]
                        new_stl_path = os.path.join(folder_path, new_name)
                        new_name_without_ext = os.path.splitext(new_name)[0]

                        # 変換したSTLを保存
                        writer = vtk.vtkSTLWriter()
                        writer.SetFileName(new_stl_path)
                        writer.SetInputConnection(reverse.GetOutputPort())
                        writer.Write()

                        # 新しいXMLファイルのパスを生成
                        new_xml_path = os.path.splitext(new_stl_path)[0] + '.xml'

                        # 色情報を取得
                        if xml_data is not None:
                            color_element = xml_data.find(".//material/color")
                            if color_element is not None:
                                rgba_str = color_element.get('rgba', "1.0 1.0 1.0 1.0")
                                try:
                                    r, g, b, _ = map(float, rgba_str.split())
                                    hex_color = '#{:02X}{:02X}{:02X}'.format(
                                        int(r * 255), int(g * 255), int(b * 255))
                                except (ValueError, IndexError):
                                    hex_color = '#FFFFFF'
                                    rgba_str = "1.0 1.0 1.0 1.0"
                        else:
                            try:
                                rgb_values = [float(input.text()) for input in self.color_inputs]
                                hex_color = '#{:02X}{:02X}{:02X}'.format(
                                    int(rgb_values[0] * 255),
                                    int(rgb_values[1] * 255),
                                    int(rgb_values[2] * 255)
                                )
                                rgba_str = f"{rgb_values[0]:.6f} {rgb_values[1]:.6f} {rgb_values[2]:.6f} 1.0"
                            except ValueError:
                                hex_color = '#FFFFFF'
                                rgba_str = "1.0 1.0 1.0 1.0"

                        # XMLファイルの内容を生成
                        urdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urdf_part>
    <material name="{hex_color}">
        <color rgba="{rgba_str}" />
    </material>
    <link name="{new_name_without_ext}">
        <visual>
            <material name="{hex_color}" />
        </visual>
        <volume value="{volume:.12f}"/>
        <mass value="{mass:.12f}"/>
        {self.format_inertia_for_urdf(inertia_tensor)}</link>"""

                        # 既存のXMLからポイントデータを取得して反転
                        if xml_data is not None:
                            points = xml_data.findall('.//point')
                            for point in points:
                                xyz_element = point.find('point_xyz')
                                if xyz_element is not None and xyz_element.text:
                                    try:
                                        x, y, z = map(float, xyz_element.text.strip().split())
                                        mirrored_y = -y  # Y座標のみ反転
                                        point_name = point.get('name', '').replace('l_', 'r_').replace('L_', 'R_')
                                        urdf_content += f"""
    <point name="{point_name}" type="fixed">
        <point_xyz>{x:.6f} {mirrored_y:.6f} {z:.6f}</point_xyz>
    </point>"""
                                    except ValueError:
                                        print(f"Error processing point coordinates in XML")

                        # 既存のXMLから回転軸情報を取得
                        axis_vector = "1 0 0"  # デフォルト値
                        if xml_data is not None:
                            axis_element = xml_data.find('.//axis')
                            if axis_element is not None:
                                axis_vector = axis_element.get('xyz', "1 0 0")

                        urdf_content += f"""
    <joint>
        <axis xyz="{axis_vector}" />
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
                        continue

                # 処理完了メッセージ
                if processed_count > 0:
                    print(f"Bulk conversion completed. Processed {processed_count} files.")
                else:
                    print("No files were processed. Make sure there are STL files with 'l_' or 'L_' prefix in the selected folder.")

        except Exception as e:
            print(f"Error during bulk conversion: {str(e)}")
            traceback.print_exc()

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
            self.render_window.Render()

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
        self.render_window.Render()

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
                self.render_window.Render()

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
            self.render_window.Render()
            print(f"Applied color: RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")
            
        except ValueError as e:
            print(f"Error: Invalid color value - {str(e)}")
        except Exception as e:
            print(f"Error applying color: {str(e)}")

    def add_axes_widget(self):
        """座標軸を表示するウィジェットを追加"""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.3, 0.3, 0.3)  # 軸の長さを設定
        axes.SetShaftTypeToLine()
        axes.SetNormalizedShaftLength(1, 1, 1)
        axes.SetNormalizedTipLength(0.1, 0.1, 0.1)
        
        # 座標軸のラベルを設定
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.render_window_interactor)
        widget.SetViewport(0.7, 0.7, 1.0, 1.0)  # 右上に表示
        widget.EnabledOn()
        widget.InteractiveOff()  # インタラクティブな操作を無効化
        
        return widget

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
