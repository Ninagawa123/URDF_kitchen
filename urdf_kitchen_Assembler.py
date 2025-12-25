"""
File Name: urdf_kitchen_Assembler.py
Description: A Python script to assembling files configured with urdf_kitchen_PartsEditor.py into a URDF file.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Update.     : Dec 25, 2025
Version     : 0.0.3
License     : MIT License
URL         : https://github.com/Ninagawa123/URDF_kitchen_beta
Copyright (c) 2024 Ninagawa123

pip install numpy
pip install PySide6
pip install vtk
pip install NodeGraphQt
"""

import sys
import signal
import traceback
from Qt import QtWidgets, QtCore, QtGui
from NodeGraphQt import NodeGraph, BaseNode
import vtk
from PySide6.QtWidgets import QFileDialog, QLabel
from PySide6.QtCore import QPointF, QRegularExpression, QTimer, Qt
from PySide6.QtGui import QDoubleValidator, QRegularExpressionValidator, QPalette, QColor, QImage, QPixmap
import os
import xml.etree.ElementTree as ET
import base64
import shutil
import datetime
import numpy as np
import trimesh

# Import URDF Kitchen utilities for M4 Mac compatibility
from urdf_kitchen_utils import OffscreenRenderer

# M4 Mac (Apple Silicon) compatibility
import platform
IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'

# デフォルト値の定数定義
DEFAULT_JOINT_LOWER = -3.14159
DEFAULT_JOINT_UPPER = 3.14159
DEFAULT_JOINT_EFFORT = 10.0
DEFAULT_JOINT_VELOCITY = 3.0
DEFAULT_COLOR_WHITE = [1.0, 1.0, 1.0]
DEFAULT_COORDS_ZERO = [0.0, 0.0, 0.0]
DEFAULT_INERTIA_ZERO = {
    'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
    'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
}
DEFAULT_ORIGIN_ZERO = {
    'xyz': [0.0, 0.0, 0.0],
    'rpy': [0.0, 0.0, 0.0]
}

def init_node_properties(node):
    """ノードの共通プロパティを初期化"""
    node.volume_value = 0.0
    node.mass_value = 0.0
    node.inertia = DEFAULT_INERTIA_ZERO.copy()
    node.inertial_origin = {
        'xyz': DEFAULT_ORIGIN_ZERO['xyz'].copy(),
        'rpy': DEFAULT_ORIGIN_ZERO['rpy'].copy()
    }
    node.stl_file = None
    node.node_color = DEFAULT_COLOR_WHITE.copy()
    node.rotation_axis = 0  # 0: X, 1: Y, 2: Z
    node.joint_lower = DEFAULT_JOINT_LOWER
    node.joint_upper = DEFAULT_JOINT_UPPER
    node.joint_effort = DEFAULT_JOINT_EFFORT
    node.joint_velocity = DEFAULT_JOINT_VELOCITY
    node.massless_decoration = False

def create_point_data(index):
    """ポイントデータを作成"""
    return {
        'name': f'point_{index}',
        'type': 'fixed',
        'xyz': DEFAULT_COORDS_ZERO.copy()
    }

def create_cumulative_coord(index):
    """累積座標データを作成"""
    return {
        'point_index': index,
        'xyz': DEFAULT_COORDS_ZERO.copy()
    }

def apply_dark_theme(app):
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)

class BaseLinkNode(BaseNode):
    """Base link node class"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'BaseLinkNode'

    def __init__(self):
        super(BaseLinkNode, self).__init__()
        self.add_output('out')

        self.output_count = 1  # 出力ポートのカウント

        # 共通プロパティの初期化
        init_node_properties(self)

        # BaseLinkNode固有のプロパティ
        self.blank_link = True  # デフォルトでBlanklink（パラメータなしのリンク）

        # BaseLinkNode固有のポイントと累積座標
        base_point = create_point_data(1)
        base_point['name'] = 'base_link_point1'  # base_linkは特別な名前を使用
        self.points = [base_point]
        self.cumulative_coords = [create_cumulative_coord(0)]

        # ダブルクリックイベントを設定
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self.node_double_clicked

    def _add_output(self, name=''):
        """出力ポートを追加（複数の子リンクをサポート）"""
        if self.output_count < 8:  # 最大8ポートまで
            self.output_count += 1
            port_name = f'out_{self.output_count}'
            super(BaseLinkNode, self).add_output(port_name)

            # ポイントデータの初期化
            if not hasattr(self, 'points'):
                self.points = []

            # 新しいポイントを追加
            self.points.append(create_point_data(self.output_count))

            # 累積座標の初期化
            if not hasattr(self, 'cumulative_coords'):
                self.cumulative_coords = []

            self.cumulative_coords.append(create_cumulative_coord(self.output_count - 1))

            print(f"Added output port '{port_name}' to BaseLinkNode")
            return port_name

    def add_input(self, name='', **kwargs):
        # 入力ポートの追加を禁止
        print("Base Link node cannot have input ports")
        return None

    def add_output(self, name='out_1', **kwargs):
        # 出力ポートが既に存在する場合は追加しない
        if not self.has_output(name):
            return super(BaseLinkNode, self).add_output(name, **kwargs)
        return None

    def remove_output(self, port=None):
        # 出力ポートの削除を禁止
        print("Cannot remove output port from Base Link node")
        return None

    def has_output(self, name):
        """指定した名前の出力ポートが存在するかチェック"""
        return name in [p.name() for p in self.output_ports()]

    def node_double_clicked(self, event):
        """BaseLinkNodeがダブルクリックされたときの処理"""
        print(f"Node {self.name()} double-clicked!")
        if hasattr(self.graph, 'show_inspector'):
            try:
                # グラフのビューを正しく取得
                graph_view = self.graph.viewer()

                # シーン座標をビュー座標に変換
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)

                print(f"Double click at screen coordinates: ({screen_pos.x()}, {screen_pos.y()})")
                self.graph.show_inspector(self, screen_pos)

            except Exception as e:
                print(f"Error getting mouse position: {str(e)}")
                traceback.print_exc()
                # フォールバック：位置指定なしでインスペクタを表示
                self.graph.show_inspector(self)
        else:
            print("Error: graph does not have show_inspector method")

class FooNode(BaseNode):
    """General purpose node class"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'FooNode'
    
    def __init__(self):
        super(FooNode, self).__init__()
        self.add_input('in', color=(180, 80, 0))

        self.output_count = 0

        # 共通プロパティの初期化
        init_node_properties(self)

        # FooNode固有のポイントと累積座標（空で開始）
        self.points = []
        self.cumulative_coords = []

        # 出力ポートを追加
        self._add_output()

        self.set_port_deletion_allowed(True)
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self.node_double_clicked

    def _add_output(self, name=''):
        if self.output_count < 8:  # 最大8ポートまで
            self.output_count += 1
            port_name = f'out_{self.output_count}'
            super(FooNode, self).add_output(port_name)
            
            # ポイントデータの初期化
            if not hasattr(self, 'points'):
                self.points = []

            # 新しいポイントを追加
            self.points.append(create_point_data(self.output_count))

            # 累積座標の初期化
            if not hasattr(self, 'cumulative_coords'):
                self.cumulative_coords = []

            self.cumulative_coords.append(create_cumulative_coord(self.output_count - 1))
            
            print(f"Added output port '{port_name}' with zero coordinates")
            return port_name

    def remove_output(self):
        """出力ポートの削除（修正版）"""
        if self.output_count > 1:
            port_name = f'out_{self.output_count}'
            output_port = self.get_output(port_name)
            if output_port:
                try:
                    # すべての接続をクリア
                    output_port.clear_connections()
                    print(f"Cleared all connections for port {port_name}")

                    # 対応するポイントデータを削除
                    if len(self.points) >= self.output_count:
                        self.points.pop()
                        print(f"Removed point data for port {port_name}")

                    # 累積座標を削除
                    if len(self.cumulative_coords) >= self.output_count:
                        self.cumulative_coords.pop()
                        print(f"Removed cumulative coordinates for port {port_name}")

                    # ポートの削除
                    self.delete_output(output_port)
                    self.output_count -= 1
                    print(f"Removed port {port_name}")

                    # ビューの更新
                    self.view.update()
                    
                except Exception as e:
                    print(f"Error removing port and associated data: {str(e)}")
                    traceback.print_exc()
            else:
                print(f"Output port {port_name} not found")
        else:
            print("Cannot remove the last output port")

    def node_double_clicked(self, event):
        print(f"Node {self.name()} double-clicked!")
        if hasattr(self.graph, 'show_inspector'):
            try:
                # グラフのビューを正しく取得
                graph_view = self.graph.viewer()  # NodeGraphQtではviewer()メソッドを使用
                
                # シーン座標をビュー座標に変換
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                
                print(f"Double click at screen coordinates: ({screen_pos.x()}, {screen_pos.y()})")
                self.graph.show_inspector(self, screen_pos)
                
            except Exception as e:
                print(f"Error getting mouse position: {str(e)}")
                traceback.print_exc()
                # フォールバック：位置指定なしでインスペクタを表示
                self.graph.show_inspector(self)
        else:
            print("Error: graph does not have show_inspector method")

class InspectorWindow(QtWidgets.QWidget):
    
    def __init__(self, parent=None, stl_viewer=None):
        super(InspectorWindow, self).__init__(parent)
        self.setWindowTitle("Node Inspector")
        self.setMinimumWidth(450)
        self.setMinimumHeight(450)
        self.resize(450, 680)  # デフォルトサイズ

        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.current_node = None
        self.stl_viewer = stl_viewer
        self.port_widgets = []

        # UIの初期化
        self.setup_ui()

        # キーボードフォーカスを受け取れるように設定
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def setup_ui(self):
        """UIの初期化"""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)  # 全体の余白を小さく
        main_layout.setContentsMargins(10, 5, 10, 5)  # 上下の余白も調整

        # スクロールエリアの設定
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # スクロールの中身となるウィジェット
        scroll_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(scroll_content)
        content_layout.setSpacing(6)  # セクション間の間隔をコンパクトに
        content_layout.setContentsMargins(5, 5, 5, 5)  # 余白を小さく

        # Node Name セクション（横一列）
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Node Name:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Enter node name")
        self.name_edit.editingFinished.connect(self.update_node_name)
        name_layout.addWidget(self.name_edit)

        content_layout.addLayout(name_layout)

        # Massless Decorationチェックボックス
        massless_layout = QtWidgets.QHBoxLayout()
        self.massless_checkbox = QtWidgets.QCheckBox("Massless Decoration")
        self.massless_checkbox.setChecked(False)  # デフォルトはオフ
        massless_layout.addWidget(self.massless_checkbox)
        content_layout.addLayout(massless_layout)

        # チェックボックスの状態変更時のハンドラを接続
        self.massless_checkbox.stateChanged.connect(self.update_massless_decoration)

        # Blanklinkチェックボックス（BaseLinkNode用）
        blanklink_layout = QtWidgets.QHBoxLayout()
        self.blanklink_checkbox = QtWidgets.QCheckBox("Blanklink (base_link only)")
        self.blanklink_checkbox.setChecked(True)  # デフォルトはオン
        self.blanklink_checkbox.setToolTip("When checked, outputs base_link without parameters in URDF")
        blanklink_layout.addWidget(self.blanklink_checkbox)
        content_layout.addLayout(blanklink_layout)

        # チェックボックスの状態変更時のハンドラを接続
        self.blanklink_checkbox.stateChanged.connect(self.update_blanklink)

        # Physical Properties セクション（テキストを削除して詰める）
        physics_layout = QtWidgets.QGridLayout()
        physics_layout.setVerticalSpacing(3)
        physics_layout.addWidget(QtWidgets.QLabel("Volume:"), 0, 0)
        self.volume_input = QtWidgets.QLineEdit()
        self.volume_input.setReadOnly(True)
        physics_layout.addWidget(self.volume_input, 0, 1)

        physics_layout.addWidget(QtWidgets.QLabel("Mass:"), 1, 0)
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setValidator(QtGui.QDoubleValidator())
        physics_layout.addWidget(self.mass_input, 1, 1)
        content_layout.addLayout(physics_layout)

        # Inertial タイトル
        inertial_title = QtWidgets.QLabel("Inertial")
        inertial_title.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(inertial_title)
        content_layout.addSpacing(3)

        # Inertial Origin セクション
        origin_layout = QtWidgets.QGridLayout()
        origin_layout.setVerticalSpacing(3)

        # 0行目: x, y, z
        origin_layout.addWidget(QtWidgets.QLabel("x:"), 0, 0)
        self.inertial_x_input = QtWidgets.QLineEdit()
        self.inertial_x_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_x_input.setPlaceholderText("0.0")
        origin_layout.addWidget(self.inertial_x_input, 0, 1)

        origin_layout.addWidget(QtWidgets.QLabel("y:"), 0, 2)
        self.inertial_y_input = QtWidgets.QLineEdit()
        self.inertial_y_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_y_input.setPlaceholderText("0.0")
        origin_layout.addWidget(self.inertial_y_input, 0, 3)

        origin_layout.addWidget(QtWidgets.QLabel("z:"), 0, 4)
        self.inertial_z_input = QtWidgets.QLineEdit()
        self.inertial_z_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_z_input.setPlaceholderText("0.0")
        origin_layout.addWidget(self.inertial_z_input, 0, 5)

        # 1行目: r, p, y
        origin_layout.addWidget(QtWidgets.QLabel("r:"), 1, 0)
        self.inertial_r_input = QtWidgets.QLineEdit()
        self.inertial_r_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_r_input.setPlaceholderText("0.0")
        origin_layout.addWidget(self.inertial_r_input, 1, 1)

        origin_layout.addWidget(QtWidgets.QLabel("p:"), 1, 2)
        self.inertial_p_input = QtWidgets.QLineEdit()
        self.inertial_p_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_p_input.setPlaceholderText("0.0")
        origin_layout.addWidget(self.inertial_p_input, 1, 3)

        origin_layout.addWidget(QtWidgets.QLabel("y:"), 1, 4)
        self.inertial_y_rpy_input = QtWidgets.QLineEdit()
        self.inertial_y_rpy_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.inertial_y_rpy_input.setPlaceholderText("0.0")
        origin_layout.addWidget(self.inertial_y_rpy_input, 1, 5)

        content_layout.addLayout(origin_layout)
        content_layout.addSpacing(3)

        # Inertia Tensor セクション
        inertia_layout = QtWidgets.QGridLayout()
        inertia_layout.setVerticalSpacing(3)

        # 0行目: ixx, ixy, ixz
        inertia_layout.addWidget(QtWidgets.QLabel("ixx:"), 0, 0)
        self.ixx_input = QtWidgets.QLineEdit()
        self.ixx_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.ixx_input.setPlaceholderText("0.0")
        inertia_layout.addWidget(self.ixx_input, 0, 1)

        inertia_layout.addWidget(QtWidgets.QLabel("ixy:"), 0, 2)
        self.ixy_input = QtWidgets.QLineEdit()
        self.ixy_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.ixy_input.setPlaceholderText("0.0")
        inertia_layout.addWidget(self.ixy_input, 0, 3)

        inertia_layout.addWidget(QtWidgets.QLabel("ixz:"), 0, 4)
        self.ixz_input = QtWidgets.QLineEdit()
        self.ixz_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.ixz_input.setPlaceholderText("0.0")
        inertia_layout.addWidget(self.ixz_input, 0, 5)

        # 1行目: iyy, iyz, izz
        inertia_layout.addWidget(QtWidgets.QLabel("iyy:"), 1, 0)
        self.iyy_input = QtWidgets.QLineEdit()
        self.iyy_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.iyy_input.setPlaceholderText("0.0")
        inertia_layout.addWidget(self.iyy_input, 1, 1)

        inertia_layout.addWidget(QtWidgets.QLabel("iyz:"), 1, 2)
        self.iyz_input = QtWidgets.QLineEdit()
        self.iyz_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.iyz_input.setPlaceholderText("0.0")
        inertia_layout.addWidget(self.iyz_input, 1, 3)

        inertia_layout.addWidget(QtWidgets.QLabel("izz:"), 1, 4)
        self.izz_input = QtWidgets.QLineEdit()
        self.izz_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 6))
        self.izz_input.setPlaceholderText("0.0")
        inertia_layout.addWidget(self.izz_input, 1, 5)

        content_layout.addLayout(inertia_layout)
        content_layout.addSpacing(5)

        # Inertia関連ボタン
        inertia_button_layout = QtWidgets.QHBoxLayout()
        inertia_button_layout.addStretch()

        # Look COM スイッチ（左側）
        self.look_inertial_origin_toggle = QtWidgets.QPushButton("Look COM")
        self.look_inertial_origin_toggle.setCheckable(True)
        self.look_inertial_origin_toggle.setFixedWidth(90)
        self.look_inertial_origin_toggle.toggled.connect(self.toggle_inertial_origin_view)
        inertia_button_layout.addWidget(self.look_inertial_origin_toggle)

        # Recalc COM ボタン（左中央）
        recalc_com_button = QtWidgets.QPushButton("Recalc COM")
        recalc_com_button.setFixedWidth(100)
        recalc_com_button.clicked.connect(self.recalculate_com)
        inertia_button_layout.addWidget(recalc_com_button)

        # Recalc Inertia ボタン（右中央）
        recalc_inertia_button = QtWidgets.QPushButton("Recalc Inertia")
        recalc_inertia_button.setFixedWidth(110)
        recalc_inertia_button.clicked.connect(self.recalculate_inertia)
        inertia_button_layout.addWidget(recalc_inertia_button)

        # Set Inertia ボタン（右側）
        set_inertia_button = QtWidgets.QPushButton("Set Inertia")
        set_inertia_button.setFixedWidth(90)
        set_inertia_button.clicked.connect(self.set_inertia)
        inertia_button_layout.addWidget(set_inertia_button)
        content_layout.addLayout(inertia_button_layout)

        # Rotation Axis セクション（横一列）
        rotation_layout = QtWidgets.QHBoxLayout()
        rotation_layout.addWidget(QtWidgets.QLabel("Rotation Axis:   "))
        self.axis_group = QtWidgets.QButtonGroup(self)
        for i, axis in enumerate(['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']):  # Fixedを追加
            radio = QtWidgets.QRadioButton(axis)
            self.axis_group.addButton(radio, i)  # iは0,1,2,3となる（3がFixed）
            rotation_layout.addWidget(radio)
        content_layout.addLayout(rotation_layout)

        # Rotation Testボタンと Followチェックボックスの配置
        rotation_test_layout = QtWidgets.QHBoxLayout()
        rotation_test_layout.addStretch()  # 左側に伸縮可能なスペースを追加

        # Followチェックボックス
        self.follow_checkbox = QtWidgets.QCheckBox("Follow")
        self.follow_checkbox.setChecked(True)  # デフォルトでオン
        self.follow_checkbox.setToolTip("Child nodes rotate together with this node")
        rotation_test_layout.addWidget(self.follow_checkbox)

        # Rotation Testボタン
        self.rotation_test_button = QtWidgets.QPushButton("Rotation Test")
        self.rotation_test_button.setFixedWidth(120)  # 他のボタンと同じ幅に設定
        self.rotation_test_button.pressed.connect(self.start_rotation_test)
        self.rotation_test_button.released.connect(self.stop_rotation_test)
        rotation_test_layout.addWidget(self.rotation_test_button)
        content_layout.addLayout(rotation_test_layout)

        # Joint Limits セクション
        joint_limits_layout = QtWidgets.QGridLayout()
        joint_limits_layout.setVerticalSpacing(3)

        # 0行目: Lower Limit と Upper Limit
        joint_limits_layout.addWidget(QtWidgets.QLabel("Lower Limit:"), 0, 0)
        self.lower_limit_input = QtWidgets.QLineEdit()
        self.lower_limit_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 5))
        self.lower_limit_input.setPlaceholderText("-3.14159")
        joint_limits_layout.addWidget(self.lower_limit_input, 0, 1)

        joint_limits_layout.addWidget(QtWidgets.QLabel("Upper Limit:"), 0, 2)
        self.upper_limit_input = QtWidgets.QLineEdit()
        self.upper_limit_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 5))
        self.upper_limit_input.setPlaceholderText("3.14159")
        joint_limits_layout.addWidget(self.upper_limit_input, 0, 3)

        # 1行目: ボタン（右寄せ）
        joint_buttons_layout = QtWidgets.QHBoxLayout()
        joint_buttons_layout.addStretch()

        look_lower_button = QtWidgets.QPushButton("Look lower")
        look_lower_button.setFixedWidth(90)
        look_lower_button.clicked.connect(self.look_lower_limit)
        joint_buttons_layout.addWidget(look_lower_button)

        look_upper_button = QtWidgets.QPushButton("Look upper")
        look_upper_button.setFixedWidth(90)
        look_upper_button.clicked.connect(self.look_upper_limit)
        joint_buttons_layout.addWidget(look_upper_button)

        set_limits_button = QtWidgets.QPushButton("Set Limits")
        set_limits_button.setFixedWidth(90)
        set_limits_button.clicked.connect(self.set_joint_limits)
        joint_buttons_layout.addWidget(set_limits_button)

        joint_limits_layout.addLayout(joint_buttons_layout, 1, 0, 1, 4)

        # 2行目: Effort と Velocity
        joint_limits_layout.addWidget(QtWidgets.QLabel("Effort:"), 2, 0)
        self.effort_input = QtWidgets.QLineEdit()
        self.effort_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.effort_input.setPlaceholderText("10")
        joint_limits_layout.addWidget(self.effort_input, 2, 1)

        joint_limits_layout.addWidget(QtWidgets.QLabel("Velocity:"), 2, 2)
        self.velocity_input = QtWidgets.QLineEdit()
        self.velocity_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.velocity_input.setPlaceholderText("3")
        joint_limits_layout.addWidget(self.velocity_input, 2, 3)

        content_layout.addLayout(joint_limits_layout)

        # Color セクション
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(QtWidgets.QLabel("Color:"))

        # カラーサンプルチップ
        self.color_sample = QtWidgets.QLabel()
        self.color_sample.setFixedSize(20, 20)
        self.color_sample.setStyleSheet(
        "background-color: rgb(255,255,255); border: 1px solid black;")
        color_layout.addWidget(self.color_sample)

        # R,G,B入力
        color_layout.addWidget(QtWidgets.QLabel("   R:"))
        self.color_inputs = []
        for label in ['', 'G:', 'B:']:  # Rは既に追加したので空文字
            if label:  # G:とB:のみラベルを追加
                color_layout.addWidget(QtWidgets.QLabel(label))
            color_input = QtWidgets.QLineEdit("1.0")
            color_input.setFixedWidth(50)
            color_input.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))
            self.color_inputs.append(color_input)
            color_layout.addWidget(color_input)

        color_layout.addStretch()  # 右側の余白を埋める
        content_layout.addLayout(color_layout)

        # カラーピッカーボタン
        pick_button = QtWidgets.QPushButton("Pick")
        pick_button.clicked.connect(self.show_color_picker)
        pick_button.setFixedWidth(40)
        color_layout.addWidget(pick_button)

        # Applyボタン
        apply_button = QtWidgets.QPushButton("Set")
        apply_button.clicked.connect(self.apply_color_to_stl)
        apply_button.setFixedWidth(40)
        color_layout.addWidget(apply_button)
        
        color_layout.addStretch()
        content_layout.addLayout(color_layout)

        # Output Ports セクション
        ports_layout = QtWidgets.QVBoxLayout()
        self.ports_layout = QtWidgets.QVBoxLayout()  # 動的に追加されるポートのための親レイアウト
        ports_layout.addLayout(self.ports_layout)

        # Set Allボタンレイアウト
        set_button_layout = QtWidgets.QHBoxLayout()
        set_button_layout.addStretch()
        set_button = QtWidgets.QPushButton("Set All")
        set_button.clicked.connect(self.apply_port_values)
        set_button_layout.addWidget(set_button)
        ports_layout.addLayout(set_button_layout)
        content_layout.addLayout(ports_layout)

        # ポートウィジェットを格納するリストを初期化
        self.port_widgets = []

        # Point Controls セクション（横一列にする）
        point_layout = QtWidgets.QHBoxLayout()
        point_layout.addWidget(QtWidgets.QLabel("Point Controls:"))
        self.add_point_btn = QtWidgets.QPushButton("[+] Add")
        self.remove_point_btn = QtWidgets.QPushButton("[-] Remove")
        point_layout.addWidget(self.add_point_btn)
        point_layout.addWidget(self.remove_point_btn)
        self.add_point_btn.clicked.connect(self.add_point)
        self.remove_point_btn.clicked.connect(self.remove_point)
        content_layout.addLayout(point_layout)

        # File Controls セクション（テキストを削除して詰める）
        file_layout = QtWidgets.QHBoxLayout()
        self.load_stl_btn = QtWidgets.QPushButton("Load STL")
        self.load_xml_btn = QtWidgets.QPushButton("Load XML")
        self.load_xml_with_stl_btn = QtWidgets.QPushButton("Load XML with STL")
        file_layout.addWidget(self.load_stl_btn)
        file_layout.addWidget(self.load_xml_btn)
        file_layout.addWidget(self.load_xml_with_stl_btn)
        self.load_stl_btn.clicked.connect(self.load_stl)
        self.load_xml_btn.clicked.connect(self.load_xml)
        self.load_xml_with_stl_btn.clicked.connect(self.load_xml_with_stl)
        content_layout.addLayout(file_layout)

        # ウィンドウリサイズ時の余白を最下部に集約
        content_layout.addStretch()

        # スクロールエリアにコンテンツをセット
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # 既存のレイアウトにも spacing を設定
        name_layout.setSpacing(2)
        physics_layout.setSpacing(2)
        rotation_layout.setSpacing(2)
        color_layout.setSpacing(2)
        ports_layout.setSpacing(2)
        point_layout.setSpacing(2)
        file_layout.setSpacing(2)

        # 既存のグリッドレイアウトの余白調整
        physics_layout.setVerticalSpacing(2)
        physics_layout.setHorizontalSpacing(2)

        for line_edit in self.findChildren(QtWidgets.QLineEdit):
            line_edit.setStyleSheet("QLineEdit { padding-left: 2px; padding-top: 0px; padding-bottom: 0px; }")

    def setup_validators(self):
        """数値入力フィールドにバリデータを設定"""
        try:
            # Mass入力フィールド用のバリデータ
            mass_validator = QtGui.QDoubleValidator()
            mass_validator.setBottom(0.0)  # 負の値を禁止
            self.mass_input.setValidator(mass_validator)

            # Volume入力フィールド用のバリデータ
            volume_validator = QtGui.QDoubleValidator()
            volume_validator.setBottom(0.0)  # 負の値を禁止
            self.volume_input.setValidator(volume_validator)

            # RGB入力フィールド用のバリデータ
            rgb_validator = QtGui.QDoubleValidator(
                0.0, 1.0, 3)  # 0.0から1.0まで、小数点以下3桁
            for color_input in self.color_inputs:
                color_input.setValidator(rgb_validator)

            # Output Ports用のバリデータ
            coord_validator = QtGui.QDoubleValidator()
            for port_widget in self.port_widgets:
                for input_field in port_widget.findChildren(QtWidgets.QLineEdit):
                    input_field.setValidator(coord_validator)

            print("Input validators setup completed")

        except Exception as e:
            print(f"Error setting up validators: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_color_to_stl(self):
        """選択された色をSTLモデルとカラーサンプルに適用"""
        if not self.current_node:
            print("No node selected")
            return
        
        try:
            # RGB値の取得（0-1の範囲）
            rgb_values = [float(input.text()) for input in self.color_inputs]
            
            # 値の範囲チェック
            rgb_values = [max(0.0, min(1.0, value)) for value in rgb_values]
            
            # ノードの色情報を更新
            self.current_node.node_color = rgb_values
            
            # カラーサンプルチップを必ず更新
            rgb_display = [int(v * 255) for v in rgb_values]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_display[0]},{rgb_display[1]},{rgb_display[2]}); "
                f"border: 1px solid black;"
            )
            
            # STLモデルの色を変更
            if self.stl_viewer and hasattr(self.stl_viewer, 'stl_actors'):
                if self.current_node in self.stl_viewer.stl_actors:
                    actor = self.stl_viewer.stl_actors[self.current_node]
                    actor.GetProperty().SetColor(*rgb_values)
                    self.stl_viewer.render_to_image()
                    print(f"Applied color: RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")
                else:
                    print("No STL model found for this node")
            
        except ValueError as e:
            print(f"Error: Invalid color value - {str(e)}")
        except Exception as e:
            print(f"Error applying color: {str(e)}")
            traceback.print_exc()

    def update_color_sample(self):
        """カラーサンプルの表示を更新"""
        try:
            rgb_values = [min(255, max(0, int(float(input.text()) * 255))) 
                        for input in self.color_inputs]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_values[0]},{rgb_values[1]},{rgb_values[2]}); "
                f"border: 1px solid black;"
            )
            
            if self.current_node:
                self.current_node.node_color = [float(input.text()) for input in self.color_inputs]
                
        except ValueError as e:
            print(f"Error updating color sample: {str(e)}")
            traceback.print_exc()

    def update_port_coordinate(self, port_index, coord_index, value):
        """ポート座標の更新"""
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

    def _set_inertial_origin_ui(self, xyz, rpy):
        """Inertial OriginのUI入力フィールドに値を設定"""
        self.inertial_x_input.setText(str(xyz[0]))
        self.inertial_y_input.setText(str(xyz[1]))
        self.inertial_z_input.setText(str(xyz[2]))
        self.inertial_r_input.setText(str(rpy[0]))
        self.inertial_p_input.setText(str(rpy[1]))
        self.inertial_y_rpy_input.setText(str(rpy[2]))

    def _set_inertia_ui(self, inertia_dict):
        """Inertia TensorのUI入力フィールドに値を設定"""
        self.ixx_input.setText(str(inertia_dict.get('ixx', 0.0)))
        self.ixy_input.setText(str(inertia_dict.get('ixy', 0.0)))
        self.ixz_input.setText(str(inertia_dict.get('ixz', 0.0)))
        self.iyy_input.setText(str(inertia_dict.get('iyy', 0.0)))
        self.iyz_input.setText(str(inertia_dict.get('iyz', 0.0)))
        self.izz_input.setText(str(inertia_dict.get('izz', 0.0)))

    def _set_color_ui(self, rgb_values):
        """色のUI入力フィールドに値を設定"""
        for i, value in enumerate(rgb_values[:3]):
            self.color_inputs[i].setText(f"{value:.3f}")

    def update_info(self, node):
        """ノード情報の更新"""
        self.current_node = node

        try:
            # Node Name
            self.name_edit.setText(node.name())

            # Volume & Mass
            if hasattr(node, 'volume_value'):
                self.volume_input.setText(f"{node.volume_value:.6f}")
                print(f"Volume set to: {node.volume_value}")

            if hasattr(node, 'mass_value'):
                self.mass_input.setText(f"{node.mass_value:.6f}")
                print(f"Mass set to: {node.mass_value}")

            # Inertia の設定
            if hasattr(node, 'inertia') and isinstance(node.inertia, dict):
                self._set_inertia_ui(node.inertia)
                print(f"Inertia set: {node.inertia}")
            else:
                # デフォルト値を設定
                node.inertia = DEFAULT_INERTIA_ZERO.copy()
                self._set_inertia_ui(node.inertia)
                print("Default inertia set to zeros")

            # Inertial Origin の設定
            if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                xyz = node.inertial_origin.get('xyz', DEFAULT_COORDS_ZERO)
                rpy = node.inertial_origin.get('rpy', DEFAULT_COORDS_ZERO)
                self._set_inertial_origin_ui(xyz, rpy)
                print(f"Inertial origin set: xyz={xyz}, rpy={rpy}")
            else:
                # デフォルト値を設定
                node.inertial_origin = DEFAULT_ORIGIN_ZERO.copy()
                node.inertial_origin['xyz'] = DEFAULT_COORDS_ZERO.copy()
                node.inertial_origin['rpy'] = DEFAULT_COORDS_ZERO.copy()
                self._set_inertial_origin_ui(node.inertial_origin['xyz'], node.inertial_origin['rpy'])
                print("Default inertial origin set to zeros")

            # Rotation Axis - nodeのrotation_axis属性を確認して設定
            if hasattr(node, 'rotation_axis'):
                axis_button = self.axis_group.button(node.rotation_axis)
                if axis_button:
                    axis_button.setChecked(True)
                    print(f"Rotation axis set to: {node.rotation_axis}")
            else:
                # デフォルトでX軸を選択
                node.rotation_axis = 0
                if self.axis_group.button(0):
                    self.axis_group.button(0).setChecked(True)
                    print("Default rotation axis set to X (0)")

            # Massless Decoration の状態を設定
            if hasattr(node, 'massless_decoration'):
                self.massless_checkbox.setChecked(node.massless_decoration)
                print(f"Massless decoration set to: {node.massless_decoration}")
            else:
                node.massless_decoration = False
                self.massless_checkbox.setChecked(False)
                print("Default massless decoration set to False")

            # Blanklink の状態を設定（BaseLinkNodeの場合のみ）
            if isinstance(node, BaseLinkNode):
                self.blanklink_checkbox.setVisible(True)
                if hasattr(node, 'blank_link'):
                    self.blanklink_checkbox.setChecked(node.blank_link)
                    print(f"Blanklink set to: {node.blank_link}")
                else:
                    node.blank_link = True
                    self.blanklink_checkbox.setChecked(True)
                    print("Default blanklink set to True")
            else:
                # BaseLinkNode以外の場合はBlanklinkチェックボックスを非表示
                self.blanklink_checkbox.setVisible(False)

            # Joint Limits の設定
            if hasattr(node, 'joint_lower'):
                self.lower_limit_input.setText(str(node.joint_lower))
            else:
                node.joint_lower = DEFAULT_JOINT_LOWER
                self.lower_limit_input.setText(str(DEFAULT_JOINT_LOWER))

            if hasattr(node, 'joint_upper'):
                self.upper_limit_input.setText(str(node.joint_upper))
            else:
                node.joint_upper = DEFAULT_JOINT_UPPER
                self.upper_limit_input.setText(str(DEFAULT_JOINT_UPPER))

            if hasattr(node, 'joint_effort'):
                self.effort_input.setText(str(node.joint_effort))
            else:
                # グラフのデフォルト値を使用
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_effort'):
                    node.joint_effort = node.graph.default_joint_effort
                else:
                    node.joint_effort = DEFAULT_JOINT_EFFORT
                self.effort_input.setText(str(node.joint_effort))

            if hasattr(node, 'joint_velocity'):
                self.velocity_input.setText(str(node.joint_velocity))
            else:
                # グラフのデフォルト値を使用
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_velocity'):
                    node.joint_velocity = node.graph.default_joint_velocity
                else:
                    node.joint_velocity = DEFAULT_JOINT_VELOCITY
                self.velocity_input.setText(str(node.joint_velocity))

            # Color settings - nodeのnode_color属性を確認して設定
            if hasattr(node, 'node_color') and node.node_color:
                print(f"Setting color: {node.node_color}")
                self._set_color_ui(node.node_color)

                # カラーサンプルチップの更新
                rgb_display = [int(v * 255) for v in node.node_color[:3]]
                self.color_sample.setStyleSheet(
                    f"background-color: rgb({rgb_display[0]},{rgb_display[1]},{rgb_display[2]}); "
                    f"border: 1px solid black;"
                )
                # STLモデルにも色を適用
                self.apply_color_to_stl()
            else:
                # デフォルトの色を設定（白）
                node.node_color = DEFAULT_COLOR_WHITE.copy()
                self._set_color_ui(node.node_color)
                self.color_sample.setStyleSheet(
                    "background-color: rgb(255,255,255); border: 1px solid black;"
                )
                print("Default color set to white")

            # 回転軸の選択を更新するためのシグナルを接続
            for button in self.axis_group.buttons():
                button.clicked.connect(lambda checked, btn=button: self.update_rotation_axis(btn))

            # Output Ports
            self.update_output_ports(node)

            # ラジオボタンのイベントハンドラを設定
            self.axis_group.buttonClicked.connect(self.on_axis_selection_changed)

            # バリデータの設定
            self.setup_validators()

            print(f"Inspector window updated for node: {node.name()}")

        except Exception as e:
            print(f"Error updating inspector info: {str(e)}")
            traceback.print_exc()

    def update_rotation_axis(self, button):
        """回転軸の選択が変更されたときの処理"""
        if self.current_node:
            self.current_node.rotation_axis = self.axis_group.id(button)
            print(f"Updated rotation axis to: {self.current_node.rotation_axis}")

    def on_axis_selection_changed(self, button):
        """回転軸の選択が変更されたときのイベントハンドラ"""
        if self.current_node:
            # 現在のノードの変換情報を保存
            if self.stl_viewer and self.current_node in self.stl_viewer.transforms:
                current_transform = self.stl_viewer.transforms[self.current_node]
                current_position = current_transform.GetPosition()
            else:
                current_position = [0, 0, 0]

            # 回転軸の更新
            axis_id = self.axis_group.id(button)
            self.current_node.rotation_axis = axis_id

            # 軸のタイプを判定して表示
            axis_types = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']
            if 0 <= axis_id < len(axis_types):
                print(f"Rotation axis changed to: {axis_types[axis_id]}")
            else:
                print(f"Invalid rotation axis ID: {axis_id}")

            # STLモデルの更新
            if self.stl_viewer:
                # 変換の更新
                if self.current_node in self.stl_viewer.transforms:
                    transform = self.stl_viewer.transforms[self.current_node]
                    transform.Identity()  # 変換をリセット
                    transform.Translate(*current_position)  # 元の位置を維持
                    
                    # 回転軸に基づいて現在の角度を設定（必要な場合）
                    if hasattr(self.current_node, 'current_rotation'):
                        angle = self.current_node.current_rotation
                        if axis_id == 0:      # X軸
                            transform.RotateX(angle)
                        elif axis_id == 1:    # Y軸
                            transform.RotateY(angle)
                        elif axis_id == 2:    # Z軸
                            transform.RotateZ(angle)
                    
                    # 変換を適用
                    if self.current_node in self.stl_viewer.stl_actors:
                        self.stl_viewer.stl_actors[self.current_node].SetUserTransform(transform)
                        self.stl_viewer.render_to_image()
                        print(f"Updated transform for node {self.current_node.name()} at position {current_position}")
                        
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
            # RGB値を0-1の範囲に変換してセット
            rgb_values = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]

            # 入力フィールドを更新
            self._set_color_ui(rgb_values)

            # カラーサンプルチップを直接更新
            self.color_sample.setStyleSheet(
                f"background-color: rgb({color.red()},{color.green()},{color.blue()}); "
                f"border: 1px solid black;"
            )
            
            # ノードの色情報を更新
            if self.current_node:
                self.current_node.node_color = rgb_values
                
            # STLモデルに色を適用
            self.apply_color_to_stl()
            
            print(f"Color picker: Selected RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")

    def update_node_name(self):
        """ノード名の更新"""
        if self.current_node:
            new_name = self.name_edit.text()
            old_name = self.current_node.name()
            if new_name != old_name:
                self.current_node.set_name(new_name)
                print(f"Node name updated from '{old_name}' to '{new_name}'")

    def add_point(self):
        """ポイントの追加"""
        if self.current_node and hasattr(self.current_node, '_add_output'):
            new_port_name = self.current_node._add_output()
            if new_port_name:
                self.update_info(self.current_node)
                print(f"Added new port: {new_port_name}")

    def remove_point(self):
        """ポイントの削除"""
        if self.current_node and hasattr(self.current_node, 'remove_output'):
            self.current_node.remove_output()
            self.update_info(self.current_node)
            print("Removed last port")

    def load_stl(self):
        """STLファイルの読み込み"""
        if self.current_node:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Mesh File", "", "Mesh Files (*.stl *.dae);;STL Files (*.stl);;COLLADA Files (*.dae)")
            if file_name:
                self.current_node.stl_file = file_name
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)

    def closeEvent(self, event):
        """ウィンドウが閉じられるときのイベントを処理"""
        try:
            # 全てのウィジェットを明示的に削除
            for widget in self.findChildren(QtWidgets.QWidget):
                if widget is not self:
                    widget.setParent(None)
                    widget.deleteLater()

            # 参照のクリア
            self.current_node = None
            self.stl_viewer = None
            self.port_widgets.clear()

            # イベントを受け入れ
            event.accept()

        except Exception as e:
            print(f"Error in closeEvent: {str(e)}")
            event.accept()

    def load_xml(self):
        """XMLファイルの読み込み"""
        if not self.current_node:
            print("No node selected")
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

            print("Loading XML file...")

            # リンク名の取得と設定
            link_elem = root.find('link')
            if link_elem is not None:
                link_name = link_elem.get('name')
                if link_name:
                    self.current_node.set_name(link_name)
                    self.name_edit.setText(link_name)
                    print(f"Set link name: {link_name}")

                # 物理プロパティの設定
                inertial_elem = link_elem.find('inertial')
                if inertial_elem is not None:
                    # ボリュームの設定
                    volume_elem = inertial_elem.find('volume')
                    if volume_elem is not None:
                        volume = float(volume_elem.get('value', '0.0'))
                        self.current_node.volume_value = volume
                        self.volume_input.setText(f"{volume:.6f}")
                        print(f"Set volume: {volume}")

                    # 質量の設定
                    mass_elem = inertial_elem.find('mass')
                    if mass_elem is not None:
                        mass = float(mass_elem.get('value', '0.0'))
                        self.current_node.mass_value = mass
                        self.mass_input.setText(f"{mass:.6f}")
                        print(f"Set mass: {mass}")

                    # Inertial Originの設定
                    origin_elem = inertial_elem.find('origin')
                    if origin_elem is not None:
                        origin_xyz = origin_elem.get('xyz', '0 0 0').split()
                        origin_rpy = origin_elem.get('rpy', '0 0 0').split()
                        self.current_node.inertial_origin = {
                            'xyz': [float(x) for x in origin_xyz],
                            'rpy': [float(x) for x in origin_rpy]
                        }
                        # UIに反映
                        self._set_inertial_origin_ui(
                            self.current_node.inertial_origin['xyz'],
                            self.current_node.inertial_origin['rpy']
                        )
                        print(f"Set inertial origin: xyz={origin_xyz}, rpy={origin_rpy}")

                    # 慣性モーメントの設定
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
                        # UIに反映
                        self._set_inertia_ui(self.current_node.inertia)
                        print("Set inertia tensor")

                # Center of Massの設定
                center_of_mass_elem = link_elem.find('center_of_mass')
                if center_of_mass_elem is not None:
                    com_xyz = center_of_mass_elem.text.strip().split()
                    print(f"Center of mass: {com_xyz}")

            # 色情報の処理
            material_elem = root.find('.//material/color')
            if material_elem is not None:
                rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                rgb_values = [float(x) for x in rgba[:3]]  # RGBのみ使用
                self.current_node.node_color = rgb_values
                # 色の設定をUIに反映
                self._set_color_ui(rgb_values)
                self.update_color_sample()
                # STLモデルに色を適用
                self.apply_color_to_stl()
                print(f"Set color: RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")

            # 回転軸の処理
            joint_elem = root.find('joint')
            if joint_elem is not None:
                # jointのtype属性を確認
                joint_type = joint_elem.get('type', '')
                if joint_type == 'fixed':
                    self.current_node.rotation_axis = 3  # 3をFixedとして使用
                    if self.axis_group.button(3):  # Fixed用のボタンが存在する場合
                        self.axis_group.button(3).setChecked(True)
                    print("Set rotation axis to Fixed")
                else:
                    # 回転軸の処理
                    axis_elem = joint_elem.find('axis')
                    if axis_elem is not None:
                        axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                        axis_values = [float(x) for x in axis_xyz]
                        if axis_values[2] == 1:  # Z軸
                            self.current_node.rotation_axis = 2
                            self.axis_group.button(2).setChecked(True)
                            print("Set rotation axis to Z")
                        elif axis_values[1] == 1:  # Y軸
                            self.current_node.rotation_axis = 1
                            self.axis_group.button(1).setChecked(True)
                            print("Set rotation axis to Y")
                        else:  # X軸（デフォルト）
                            self.current_node.rotation_axis = 0
                            self.axis_group.button(0).setChecked(True)
                            print("Set rotation axis to X")
                        print(f"Set rotation axis from xyz: {axis_xyz}")

            # ポイントの処理
            points = root.findall('point')
            num_points = len(points)
            print(f"Found {num_points} points in XML")

            # 現在のポート数と必要なポート数を比較
            current_ports = len(self.current_node.output_ports())
            print(f"Current ports: {current_ports}, Required points: {num_points}")

            # ポート数を調整
            if isinstance(self.current_node, FooNode):
                # ポートを削除する前に、削除対象のポートの接続をすべてクリア
                if current_ports > num_points:
                    print(f"Clearing connections for ports to be removed...")
                    for i in range(num_points + 1, current_ports + 1):
                        port_name = f'out_{i}'
                        port = self.current_node.get_output(port_name)
                        if port:
                            port.clear_connections()
                            print(f"Cleared connections for {port_name}")

                while current_ports < num_points:
                    self.current_node._add_output()
                    current_ports += 1
                    print(f"Added new port, total now: {current_ports}")

                while current_ports > num_points:
                    self.current_node.remove_output()
                    current_ports -= 1
                    print(f"Removed port, total now: {current_ports}")

                # ポイントデータの更新
                self.current_node.points = []
                for point_elem in points:
                    point_name = point_elem.get('name')
                    point_type = point_elem.get('type')
                    point_xyz_elem = point_elem.find('point_xyz')

                    if point_xyz_elem is not None and point_xyz_elem.text:
                        xyz_values = [float(x) for x in point_xyz_elem.text.strip().split()]
                        self.current_node.points.append({
                            'name': point_name,
                            'type': point_type,
                            'xyz': xyz_values
                        })
                        print(f"Added point {point_name}: {xyz_values}")

                # 累積座標の更新
                self.current_node.cumulative_coords = []
                for i in range(len(self.current_node.points)):
                    self.current_node.cumulative_coords.append(create_cumulative_coord(i))

                # output_countを更新
                self.current_node.output_count = len(self.current_node.points)
                print(f"Updated output_count to: {self.current_node.output_count}")

            # UI更新
            self.update_info(self.current_node)
            print(f"XML file loaded: {file_name}")

        except Exception as e:
            print(f"Error loading XML: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def load_xml_with_stl(self):
        """XMLファイルとそれに対応するSTLファイルを読み込む"""
        if not self.current_node:
            print("No node selected")
            return

        # XMLファイルの選択
        xml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open XML File", "", "XML Files (*.xml)")

        if not xml_file:
            return

        try:
            # 対応するSTLファイルのパスを生成
            xml_dir = os.path.dirname(xml_file)
            xml_name = os.path.splitext(os.path.basename(xml_file))[0]
            stl_path = os.path.join(xml_dir, f"{xml_name}.stl")

            # まずXMLファイルを読み込む
            tree = ET.parse(xml_file)
            root = tree.getroot()

            if root.tag != 'urdf_part':
                print("Invalid XML format: Root element should be 'urdf_part'")
                return

            # リンク情報の処理
            link_elem = root.find('link')
            if link_elem is not None:
                link_name = link_elem.get('name')
                if link_name:
                    self.current_node.set_name(link_name)
                    self.name_edit.setText(link_name)

                # 物理プロパティの設定（inertial要素内）
                inertial_elem = link_elem.find('inertial')
                if inertial_elem is not None:
                    # ボリュームの設定
                    volume_elem = inertial_elem.find('volume')
                    if volume_elem is not None:
                        volume = float(volume_elem.get('value', '0.0'))
                        self.current_node.volume_value = volume
                        self.volume_input.setText(f"{volume:.6f}")

                    # 質量の設定
                    mass_elem = inertial_elem.find('mass')
                    if mass_elem is not None:
                        mass = float(mass_elem.get('value', '0.0'))
                        self.current_node.mass_value = mass
                        self.mass_input.setText(f"{mass:.6f}")

                    # Inertial Originの設定
                    origin_elem = inertial_elem.find('origin')
                    if origin_elem is not None:
                        origin_xyz = origin_elem.get('xyz', '0 0 0').split()
                        origin_rpy = origin_elem.get('rpy', '0 0 0').split()
                        self.current_node.inertial_origin = {
                            'xyz': [float(x) for x in origin_xyz],
                            'rpy': [float(x) for x in origin_rpy]
                        }
                        # UIに反映
                        self._set_inertial_origin_ui(
                            self.current_node.inertial_origin['xyz'],
                            self.current_node.inertial_origin['rpy']
                        )
                        print(f"Set inertial origin: xyz={origin_xyz}, rpy={origin_rpy}")

                    # 慣性モーメントの設定
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
                        # UIに反映
                        self._set_inertia_ui(self.current_node.inertia)
                        print("Set inertia tensor")

                # Center of Massの設定
                center_of_mass_elem = link_elem.find('center_of_mass')
                if center_of_mass_elem is not None:
                    com_xyz = center_of_mass_elem.text.strip().split()
                    print(f"Center of mass: {com_xyz}")

            # 色情報の処理
            material_elem = root.find('.//material/color')
            if material_elem is not None:
                rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                rgb_values = [float(x) for x in rgba[:3]]  # RGBのみ使用
                self.current_node.node_color = rgb_values
                # 色の設定をUIに反映
                self._set_color_ui(rgb_values)
                self.update_color_sample()
                print(f"Set color: RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")

            # 回転軸の処理
            joint_elem = root.find('.//joint/axis')
            if joint_elem is not None:
                axis_xyz = joint_elem.get('xyz', '1 0 0').split()
                axis_values = [float(x) for x in axis_xyz]
                if axis_values[2] == 1:  # Z軸
                    self.current_node.rotation_axis = 2
                    self.axis_group.button(2).setChecked(True)
                elif axis_values[1] == 1:  # Y軸
                    self.current_node.rotation_axis = 1
                    self.axis_group.button(1).setChecked(True)
                else:  # X軸（デフォルト）
                    self.current_node.rotation_axis = 0
                    self.axis_group.button(0).setChecked(True)
                print(f"Set rotation axis: {self.current_node.rotation_axis} from xyz: {axis_xyz}")

            # ポイントの処理
            points = root.findall('point')
            num_points = len(points)
            print(f"Found {num_points} points")

            # 現在のポート数と必要なポート数を比較
            current_ports = len(self.current_node.points)
            if num_points > current_ports:
                # 不足しているポートを追加
                ports_to_add = num_points - current_ports
                for _ in range(ports_to_add):
                    self.add_point()
            elif num_points < current_ports:
                # 余分なポートを削除する前に接続をクリア
                print(f"Clearing connections for ports to be removed...")
                output_ports = self.current_node.output_ports()
                for i in range(num_points, current_ports):
                    if i < len(output_ports):
                        output_ports[i].clear_connections()
                        print(f"Cleared connections for port {output_ports[i].name()}")

                ports_to_remove = current_ports - num_points
                for _ in range(ports_to_remove):
                    self.remove_point()

            # ポイントデータの更新
            self.current_node.points = []
            for point_elem in points:
                point_name = point_elem.get('name')
                point_type = point_elem.get('type')
                point_xyz_elem = point_elem.find('point_xyz')

                if point_xyz_elem is not None and point_xyz_elem.text:
                    xyz_values = [float(x) for x in point_xyz_elem.text.strip().split()]
                    self.current_node.points.append({
                        'name': point_name,
                        'type': point_type,
                        'xyz': xyz_values
                    })
                    print(f"Added point {point_name}: {xyz_values}")

            # STLファイルの処理
            if os.path.exists(stl_path):
                print(f"Found corresponding STL file: {stl_path}")
                self.current_node.stl_file = stl_path
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                    # STLモデルに色を適用
                    self.apply_color_to_stl()
            else:
                print(f"Warning: STL file not found: {stl_path}")
                msg_box = QtWidgets.QMessageBox()
                msg_box.setIcon(QtWidgets.QMessageBox.Warning)
                msg_box.setWindowTitle("STL File Not Found")
                msg_box.setText("STL file not found in the same directory.")
                msg_box.setInformativeText("Would you like to select the STL file manually?")
                msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                msg_box.setDefaultButton(QtWidgets.QMessageBox.Yes)

                if msg_box.exec() == QtWidgets.QMessageBox.Yes:
                    stl_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                        self, "Select Mesh File", xml_dir, "Mesh Files (*.stl *.dae);;STL Files (*.stl);;COLLADA Files (*.dae)")
                    if stl_file:
                        self.current_node.stl_file = stl_file
                        if self.stl_viewer:
                            self.stl_viewer.load_stl_for_node(self.current_node)
                            # STLモデルに色を適用
                            self.apply_color_to_stl()
                        print(f"Manually selected STL file: {stl_file}")
                    else:
                        print("STL file selection cancelled")
                else:
                    print("STL file loading skipped")

            # UI更新
            self.update_info(self.current_node)
            print(f"XML file loaded: {xml_file}")

        except Exception as e:
            print(f"Error loading XML with STL: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_port_values(self):
        """Inspector内の全ての値を一括保存（Set All機能）"""
        if not self.current_node:
            print("No node selected")
            return

        print("\n=== Applying All Inspector Values ===")
        errors = []

        try:
            # 1. Node Name の保存
            try:
                new_name = self.name_edit.text()
                old_name = self.current_node.name()
                if new_name != old_name and new_name:
                    self.current_node.set_name(new_name)
                    print(f"✓ Node name: '{old_name}' → '{new_name}'")
            except Exception as e:
                errors.append(f"Node name: {str(e)}")

            # 2. Massless Decoration の保存
            try:
                self.current_node.massless_decoration = self.massless_checkbox.isChecked()
                print(f"✓ Massless decoration: {self.current_node.massless_decoration}")
            except Exception as e:
                errors.append(f"Massless decoration: {str(e)}")

            # 3. Mass の保存
            try:
                if self.mass_input.text():
                    mass = float(self.mass_input.text())
                    self.current_node.mass_value = mass
                    print(f"✓ Mass: {mass:.6f} kg")
            except ValueError as e:
                errors.append(f"Mass: Invalid number")

            # 4. Inertia と Inertial Origin の保存
            try:
                # Inertial Origin値を取得
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

                # Inertia値を取得
                inertia_values = {
                    'ixx': float(self.ixx_input.text()) if self.ixx_input.text() else 0.0,
                    'ixy': float(self.ixy_input.text()) if self.ixy_input.text() else 0.0,
                    'ixz': float(self.ixz_input.text()) if self.ixz_input.text() else 0.0,
                    'iyy': float(self.iyy_input.text()) if self.iyy_input.text() else 0.0,
                    'iyz': float(self.iyz_input.text()) if self.iyz_input.text() else 0.0,
                    'izz': float(self.izz_input.text()) if self.izz_input.text() else 0.0
                }

                # ノードに保存
                self.current_node.inertial_origin = {
                    'xyz': origin_xyz,
                    'rpy': origin_rpy
                }
                self.current_node.inertia = inertia_values
                print(f"✓ Inertial origin: xyz={origin_xyz}, rpy={origin_rpy}")
                print(f"✓ Inertia: ixx={inertia_values['ixx']:.6f}, iyy={inertia_values['iyy']:.6f}, izz={inertia_values['izz']:.6f}")
            except ValueError as e:
                errors.append(f"Inertia: {str(e)}")

            # 5. Joint Limits の保存
            try:
                if self.lower_limit_input.text():
                    self.current_node.joint_lower = float(self.lower_limit_input.text())
                if self.upper_limit_input.text():
                    self.current_node.joint_upper = float(self.upper_limit_input.text())
                if self.effort_input.text():
                    self.current_node.joint_effort = float(self.effort_input.text())
                if self.velocity_input.text():
                    self.current_node.joint_velocity = float(self.velocity_input.text())
                print(f"✓ Joint limits: lower={self.current_node.joint_lower}, upper={self.current_node.joint_upper}, effort={self.current_node.joint_effort}, velocity={self.current_node.joint_velocity}")
            except ValueError as e:
                errors.append(f"Joint limits: {str(e)}")

            # ポートウィジェットから値を取得して適用
            for i, port_widget in enumerate(self.port_widgets):
                # ポートの座標入力フィールドを検索
                coord_inputs = []
                for child in port_widget.findChildren(QtWidgets.QLineEdit):
                    coord_inputs.append(child)

                # 座標入力フィールドが3つ（X,Y,Z）あることを確認
                if len(coord_inputs) >= 3:
                    try:
                        # 座標値を取得
                        x = float(coord_inputs[0].text())
                        y = float(coord_inputs[1].text())
                        z = float(coord_inputs[2].text())

                        # ノードのポイントデータを更新
                        if hasattr(self.current_node, 'points') and i < len(self.current_node.points):
                            self.current_node.points[i]['xyz'] = [x, y, z]
                            print(
                                f"Updated point {i+1} coordinates to: ({x:.6f}, {y:.6f}, {z:.6f})")

                            # 累積座標も更新
                            if hasattr(self.current_node, 'cumulative_coords') and i < len(self.current_node.cumulative_coords):
                                if isinstance(self.current_node, BaseLinkNode):
                                    self.current_node.cumulative_coords[i]['xyz'] = [
                                        x, y, z]
                                else:
                                    # base_link以外のノードの場合は相対座標を保持
                                    self.current_node.cumulative_coords[i]['xyz'] = [
                                        0.0, 0.0, 0.0]

                    except ValueError:
                        print(f"Invalid numerical input for point {i+1}")
                        continue

            # 6. Color の保存
            try:
                rgba_values = [float(input.text()) for input in self.color_inputs]
                self.current_node.node_color = rgba_values
                print(f"✓ Color (RGBA): [{rgba_values[0]:.3f}, {rgba_values[1]:.3f}, {rgba_values[2]:.3f}, {rgba_values[3]:.3f}]")

                # STLに色を適用
                if self.stl_viewer and hasattr(self.current_node, 'stl_file') and self.current_node.stl_file:
                    if self.current_node in self.stl_viewer.stl_actors:
                        actor = self.stl_viewer.stl_actors[self.current_node]
                        actor.GetProperty().SetColor(rgba_values[0], rgba_values[1], rgba_values[2])
                        actor.GetProperty().SetOpacity(rgba_values[3])
                        print("  Color applied to STL mesh")
            except (ValueError, IndexError) as e:
                errors.append(f"Color: {str(e)}")

            # ノードの位置を再計算（必要な場合）
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()
                print("✓ Node positions recalculated")

            # STLビューアの更新
            if self.stl_viewer:
                self.stl_viewer.render_to_image()
                print("✓ 3D view updated")

            print("\n=== Set All Completed ===")

            # 完了メッセージ
            if errors:
                error_msg = "\n".join(errors)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Set All - Completed with Warnings",
                    f"All values have been saved, but some errors occurred:\n\n{error_msg}\n\n"
                    f"Please check the console for details."
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Set All - Success",
                    "All Inspector values have been successfully saved to the node!\n\n"
                    "Saved items:\n"
                    "✓ Node Name\n"
                    "✓ Massless Decoration\n"
                    "✓ Mass\n"
                    "✓ Inertia & Inertial Origin\n"
                    "✓ Joint Limits\n"
                    "✓ Port Coordinates\n"
                    "✓ Color (RGBA)"
                )

        except Exception as e:
            print(f"Error in Set All: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                "Set All - Error",
                f"An error occurred while saving values:\n\n{str(e)}"
            )

    def create_port_widget(self, port_number, x=0.0, y=0.0, z=0.0):
        """Output Port用のウィジェットを作成"""
        port_layout = QtWidgets.QHBoxLayout()  # GridLayoutからHBoxLayoutに変更
        port_layout.setSpacing(5)
        port_layout.setContentsMargins(0, 1, 0, 1)

        # ポート番号
        port_name = QtWidgets.QLabel(f"out_{port_number}")
        port_name.setFixedWidth(45)
        port_layout.addWidget(port_name)

        # 座標入力のペアを作成
        coords = []
        for label, value in [('X:', x), ('Y:', y), ('Z:', z)]:
            # 各座標のペアをHBoxLayoutで作成
            coord_pair = QtWidgets.QHBoxLayout()
            coord_pair.setSpacing(2)
            
            # ラベル
            coord_label = QtWidgets.QLabel(label)
            coord_label.setFixedWidth(15)
            coord_pair.addWidget(coord_label)

            # 入力フィールド
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

            # ペアをメインレイアウトに追加
            port_layout.addLayout(coord_pair)
            
            # ペア間にスペースを追加
            if label != 'Z:':  # 最後のペア以外の後にスペースを追加
                port_layout.addSpacing(15)

        # 右側の余白
        port_layout.addStretch()

        # ウィジェットをラップ
        port_widget = QtWidgets.QWidget()
        port_widget.setFixedHeight(25)
        port_widget.setLayout(port_layout)
        return port_widget, coords

    def update_output_ports(self, node):
        """Output Portsセクションを更新"""
        # 既存のポートウィジェットをクリア
        for widget in self.port_widgets:
            self.ports_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self.port_widgets.clear()

        # ノードの各ポートに対してウィジェットを作成
        if hasattr(node, 'points'):
            for i, point in enumerate(node.points):
                port_widget, _ = self.create_port_widget(
                    i + 1,
                    point['xyz'][0],
                    point['xyz'][1],
                    point['xyz'][2]
                )
                self.ports_layout.addWidget(port_widget)
                self.port_widgets.append(port_widget)

    def apply_color_to_stl(self):
        """選択された色をSTLモデルに適用"""
        if not self.current_node:
            return
        
        try:
            rgb_values = [float(input.text()) for input in self.color_inputs]
            rgb_values = [max(0.0, min(1.0, value)) for value in rgb_values]
            
            self.current_node.node_color = rgb_values
            
            if self.stl_viewer and hasattr(self.stl_viewer, 'stl_actors'):
                if self.current_node in self.stl_viewer.stl_actors:
                    actor = self.stl_viewer.stl_actors[self.current_node]
                    actor.GetProperty().SetColor(*rgb_values)
                    self.stl_viewer.render_to_image()
        except ValueError as e:
            print(f"Error: Invalid color value - {str(e)}")

    def update_color_sample(self):
        """カラーサンプルの表示を更新"""
        try:
            rgb_values = [min(255, max(0, int(float(input.text()) * 255))) 
                        for input in self.color_inputs]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_values[0]},{rgb_values[1]},{rgb_values[2]}); "
                f"border: 1px solid black;"
            )
        except ValueError:
            pass

    def update_massless_decoration(self, state):
        """Massless Decorationの状態を更新"""
        if self.current_node:
            self.current_node.massless_decoration = bool(state)
            print(f"Set massless_decoration to {bool(state)} for node: {self.current_node.name()}")

    def update_blanklink(self, state):
        """Blanklinkの状態を更新（BaseLinkNode用）"""
        if self.current_node and isinstance(self.current_node, BaseLinkNode):
            self.current_node.blank_link = bool(state)
            print(f"Set blank_link to {bool(state)} for node: {self.current_node.name()}")

    def moveEvent(self, event):
        """ウィンドウ移動イベントの処理"""
        super(InspectorWindow, self).moveEvent(event)
        # グラフオブジェクトが存在し、last_inspector_positionを保存可能な場合
        if hasattr(self, 'graph') and self.graph:
            self.graph.last_inspector_position = self.pos()

    def keyPressEvent(self, event):
        """キープレスイベントの処理"""
        # ESCキーが押されたかどうかを確認
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        else:
            # 他のキーイベントは通常通り処理
            super(InspectorWindow, self).keyPressEvent(event)

    def start_rotation_test(self):
        """回転テスト開始"""
        if self.current_node and self.stl_viewer:
            # Followチェックボックスの状態を取得
            follow = self.follow_checkbox.isChecked()
            self.stl_viewer.follow_children = follow

            # 現在の変換を保存
            self.stl_viewer.store_current_transform(self.current_node)
            # 回転開始
            self.stl_viewer.start_rotation_test(self.current_node)

    def stop_rotation_test(self):
        """回転テスト終了"""
        if self.current_node and self.stl_viewer:
            # 回転停止と元の角度に戻す
            self.stl_viewer.stop_rotation_test(self.current_node)

    def look_lower_limit(self):
        """Lower limitの角度を表示"""
        if self.current_node and self.stl_viewer:
            try:
                # インプットフィールドから値を取得
                lower_text = self.lower_limit_input.text()
                if not lower_text:
                    lower_text = self.lower_limit_input.placeholderText()

                lower_rad = float(lower_text)

                # 現在の変換を保存
                self.stl_viewer.store_current_transform(self.current_node)
                # 指定角度を表示
                self.stl_viewer.show_angle(self.current_node, lower_rad)
                print(f"Showing lower limit angle: {lower_rad} rad")
            except ValueError:
                print("Invalid lower limit value")

    def look_upper_limit(self):
        """Upper limitの角度を表示"""
        if self.current_node and self.stl_viewer:
            try:
                # インプットフィールドから値を取得
                upper_text = self.upper_limit_input.text()
                if not upper_text:
                    upper_text = self.upper_limit_input.placeholderText()

                upper_rad = float(upper_text)

                # 現在の変換を保存
                self.stl_viewer.store_current_transform(self.current_node)
                # 指定角度を表示
                self.stl_viewer.show_angle(self.current_node, upper_rad)
                print(f"Showing upper limit angle: {upper_rad} rad")
            except ValueError:
                print("Invalid upper limit value")

    def toggle_inertial_origin_view(self, checked):
        """Inertial Originの表示/非表示を切り替え"""
        if self.current_node and self.stl_viewer:
            if checked:
                # Inertial Origin座標を取得
                try:
                    x = float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0
                    y = float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0
                    z = float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0

                    # 3Dビューに座標系を表示
                    self.stl_viewer.show_inertial_origin(self.current_node, [x, y, z])
                    print(f"Showing inertial origin at: [{x}, {y}, {z}]")
                except ValueError:
                    print("Invalid inertial origin values")
                    self.look_inertial_origin_toggle.setChecked(False)
            else:
                # 座標系を非表示
                self.stl_viewer.hide_inertial_origin(self.current_node)
                print("Hiding inertial origin")

    def set_joint_limits(self):
        """Joint limitsの値をノードに保存"""
        if not self.current_node:
            print("No node selected")
            return

        try:
            # Lower limitの保存
            lower_text = self.lower_limit_input.text()
            if lower_text:
                self.current_node.joint_lower = float(lower_text)

            # Upper limitの保存
            upper_text = self.upper_limit_input.text()
            if upper_text:
                self.current_node.joint_upper = float(upper_text)

            # Effortの保存
            effort_text = self.effort_input.text()
            if effort_text:
                self.current_node.joint_effort = float(effort_text)

            # Velocityの保存
            velocity_text = self.velocity_input.text()
            if velocity_text:
                self.current_node.joint_velocity = float(velocity_text)

            print(f"Joint limits set: lower={self.current_node.joint_lower}, upper={self.current_node.joint_upper}, effort={self.current_node.joint_effort}, velocity={self.current_node.joint_velocity}")

            QtWidgets.QMessageBox.information(
                self,
                "Joint Limits Set",
                f"Joint limits have been set successfully.\n\n"
                f"Lower: {self.current_node.joint_lower}\n"
                f"Upper: {self.current_node.joint_upper}\n"
                f"Effort: {self.current_node.joint_effort}\n"
                f"Velocity: {self.current_node.joint_velocity}"
            )
        except ValueError as e:
            print(f"Error setting joint limits: {str(e)}")
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values."
            )

    def set_inertia(self):
        """InertiaとInertial Originの値をノードに保存"""
        if not self.current_node:
            print("No node selected")
            return

        try:
            # Inertial Origin値を取得
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

            # Inertia値を取得
            inertia_values = {
                'ixx': float(self.ixx_input.text()) if self.ixx_input.text() else 0.0,
                'ixy': float(self.ixy_input.text()) if self.ixy_input.text() else 0.0,
                'ixz': float(self.ixz_input.text()) if self.ixz_input.text() else 0.0,
                'iyy': float(self.iyy_input.text()) if self.iyy_input.text() else 0.0,
                'iyz': float(self.iyz_input.text()) if self.iyz_input.text() else 0.0,
                'izz': float(self.izz_input.text()) if self.izz_input.text() else 0.0
            }

            # ノードに保存
            self.current_node.inertial_origin = {
                'xyz': origin_xyz,
                'rpy': origin_rpy
            }
            self.current_node.inertia = inertia_values

            print(f"Inertial origin set: xyz={origin_xyz}, rpy={origin_rpy}")
            print(f"Inertia set: {inertia_values}")

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
        """STLファイルからtrimeshを使用してCenter of Massを再計算"""
        if not self.current_node:
            QtWidgets.QMessageBox.warning(
                self,
                "No Node Selected",
                "Please select a node first."
            )
            return

        # STLファイルが存在するか確認
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
            # Trimeshでメッシュを読み込み
            print(f"\n=== Recalculating Center of Mass ===")
            print(f"Loading STL file: {stl_path}")
            mesh = trimesh.load(stl_path)

            # メッシュが複数ある場合は結合
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            print(f"Mesh loaded successfully")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Volume: {mesh.volume:.6f}")
            print(f"  Is watertight: {mesh.is_watertight}")

            # メッシュが閉じていない場合は自動修復を試みる
            repair_performed = False
            original_watertight = mesh.is_watertight

            if not mesh.is_watertight:
                print("\n⚠ Mesh is not watertight. Attempting automatic repair...")

                # メモリ上でメッシュを修復（元ファイルは変更しない）
                try:
                    # 法線の修正
                    print("  - Fixing normals...")
                    mesh.fix_normals()

                    # 重複面の削除
                    print("  - Removing duplicate faces...")
                    mesh.remove_duplicate_faces()

                    # 退化面の削除
                    print("  - Removing degenerate faces...")
                    mesh.remove_degenerate_faces()

                    # 穴の修復
                    print("  - Filling holes...")
                    mesh.fill_holes()

                    repair_performed = True

                    # 修復後の状態を確認
                    print(f"\n修復後の状態:")
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

            # Center of Massを計算
            center_of_mass = mesh.center_mass
            print(f"\nCalculated center of mass: {center_of_mass}")

            # UIフィールドに設定
            self.inertial_x_input.setText(f"{center_of_mass[0]:.6f}")
            self.inertial_y_input.setText(f"{center_of_mass[1]:.6f}")
            self.inertial_z_input.setText(f"{center_of_mass[2]:.6f}")

            # 成功メッセージ
            repair_msg = ""
            if repair_performed:
                repair_msg = f"\n\nMesh Repair: Performed (in memory only)"
                repair_msg += f"\n  Before: Watertight = {original_watertight}"
                repair_msg += f"\n  After: Watertight = {mesh.is_watertight}"

            QtWidgets.QMessageBox.information(
                self,
                "COM Calculated",
                f"Center of Mass successfully calculated!\n\n"
                f"Center of mass: [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]\n"
                f"Volume: {mesh.volume:.6f} m³\n"
                f"Watertight: {'Yes' if mesh.is_watertight else 'No'}"
                f"{repair_msg}\n\n"
                f"The Inertial Origin has been updated with the calculated COM."
            )

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
        """STLファイルからtrimeshを使用して慣性を計算"""
        if not self.current_node:
            QtWidgets.QMessageBox.warning(
                self,
                "No Node Selected",
                "Please select a node first."
            )
            return

        # STLファイルが存在するか確認
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

        # 質量を取得
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

        try:
            # Trimeshでメッシュを読み込み
            print(f"Loading STL file: {stl_path}")
            mesh = trimesh.load(stl_path)

            # メッシュが複数ある場合は結合
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            print(f"Mesh loaded successfully")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Volume: {mesh.volume:.6f}")
            print(f"  Is watertight: {mesh.is_watertight}")

            # メッシュが閉じていない場合は自動修復を試みる
            repair_performed = False
            if not mesh.is_watertight:
                print("\n⚠ Mesh is not watertight. Attempting automatic repair...")

                # メモリ上でメッシュを修復（元ファイルは変更しない）
                try:
                    # 法線の修正
                    print("  - Fixing normals...")
                    mesh.fix_normals()

                    # 重複面の削除
                    print("  - Removing duplicate faces...")
                    mesh.remove_duplicate_faces()

                    # 退化面の削除
                    print("  - Removing degenerate faces...")
                    mesh.remove_degenerate_faces()

                    # 穴の修復
                    print("  - Filling holes...")
                    mesh.fill_holes()

                    repair_performed = True

                    # 修復後の状態を確認
                    print(f"\n修復後の状態:")
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

                    response = QtWidgets.QMessageBox.question(
                        self,
                        "Mesh Repair Failed",
                        f"Automatic mesh repair failed:\n{str(repair_error)}\n\n"
                        "The mesh is not watertight and repair was unsuccessful.\n"
                        "The calculated inertia may not be accurate.\n\n"
                        "Do you want to continue anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if response == QtWidgets.QMessageBox.No:
                        return

            # 密度を計算
            density = mass / mesh.volume
            print(f"Calculated density: {density:.6f} kg/m³")

            # 既存のInertial Origin値を取得（ユーザー指定の値を使用）
            try:
                inertial_origin = [
                    float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0,
                    float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0,
                    float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0
                ]
            except ValueError:
                inertial_origin = [0.0, 0.0, 0.0]

            print(f"Using inertial origin: {inertial_origin}")

            # 参考情報として重心位置を計算（表示のみ）
            center_of_mass = mesh.center_mass
            print(f"Mesh center of mass (reference): {center_of_mass}")

            # 慣性テンソルを計算（trimeshの組み込み機能を使用）
            # trimeshは重心を中心とした慣性テンソルを返す
            inertia_tensor = mesh.moment_inertia

            # 密度を適用して実際の慣性テンソルに変換
            inertia_tensor = inertia_tensor * density

            print(f"Inertia tensor at center of mass:")
            print(inertia_tensor)

            # 指定されたInertial Originが重心と異なる場合、平行軸の定理を適用
            origin_offset = np.array(inertial_origin) - np.array(center_of_mass)
            if np.linalg.norm(origin_offset) > 1e-6:
                print(f"Applying parallel axis theorem (offset: {origin_offset})")
                # 平行軸の定理: I_new = I_com + m * (d^2 * E - d ⊗ d)
                # ここで d は重心からの距離ベクトル、E は単位行列
                d = origin_offset
                d_squared = np.dot(d, d)
                outer_product = np.outer(d, d)
                inertia_tensor = inertia_tensor + mass * (d_squared * np.eye(3) - outer_product)
                print(f"Inertia tensor at specified origin:")
                print(inertia_tensor)

            # 慣性テンソルの検証
            validation_result = self._validate_inertia_tensor(inertia_tensor, mass)

            if not validation_result['valid']:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Validation Warning",
                    f"The calculated inertia tensor may not be physically valid:\n\n"
                    f"{validation_result['message']}\n\n"
                    f"The values will be set anyway, but please verify them."
                )

            # UIフィールドに慣性値を設定
            self.ixx_input.setText(f"{inertia_tensor[0, 0]:.6f}")
            self.ixy_input.setText(f"{inertia_tensor[0, 1]:.6f}")
            self.ixz_input.setText(f"{inertia_tensor[0, 2]:.6f}")
            self.iyy_input.setText(f"{inertia_tensor[1, 1]:.6f}")
            self.iyz_input.setText(f"{inertia_tensor[1, 2]:.6f}")
            self.izz_input.setText(f"{inertia_tensor[2, 2]:.6f}")

            # Inertial Originは既存の値を維持（変更しない）

            # 成功メッセージ
            repair_msg = ""
            if repair_performed:
                repair_msg = "Mesh Repair: Performed (in memory only)\n"

            # 平行軸の定理が適用されたかチェック
            parallel_axis_applied = np.linalg.norm(origin_offset) > 1e-6
            parallel_axis_msg = ""
            if parallel_axis_applied:
                parallel_axis_msg = f"\nParallel Axis Theorem: Applied (offset = {np.linalg.norm(origin_offset):.4f} m)"

            QtWidgets.QMessageBox.information(
                self,
                "Inertia Calculated",
                f"Inertia successfully calculated using trimesh!\n\n"
                f"Mass: {mass:.6f} kg\n"
                f"Volume: {mesh.volume:.6f} m³\n"
                f"Density: {density:.6f} kg/m³\n"
                f"Watertight: {'Yes' if mesh.is_watertight else 'No'}\n"
                f"{repair_msg}"
                f"\nMesh center of mass (reference): [{center_of_mass[0]:.4f}, {center_of_mass[1]:.4f}, {center_of_mass[2]:.4f}]\n"
                f"Inertial origin (used): [{inertial_origin[0]:.4f}, {inertial_origin[1]:.4f}, {inertial_origin[2]:.4f}]"
                f"{parallel_axis_msg}\n\n"
                f"Inertia tensor diagonal:\n"
                f"  Ixx: {inertia_tensor[0, 0]:.6f}\n"
                f"  Iyy: {inertia_tensor[1, 1]:.6f}\n"
                f"  Izz: {inertia_tensor[2, 2]:.6f}\n\n"
                f"{validation_result['message']}"
            )

        except Exception as e:
            print(f"Error calculating inertia: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                "Calculation Error",
                f"Failed to calculate inertia:\n\n{str(e)}"
            )

    def _validate_inertia_tensor(self, inertia_tensor, mass):
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

    def _calculate_base_inertia_tensor(self, poly_data, mass, center_of_mass, is_mirrored=False):
        """
        基本的な慣性テンソル計算のための共通実装。
        InspectorWindowクラスのメソッド。

        Args:
            poly_data: vtkPolyData オブジェクト
            mass: float 質量
            center_of_mass: list[float] 重心座標 [x, y, z]
            is_mirrored: bool ミラーリングモードかどうか

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

        # 慣性テンソルの初期化
        inertia_tensor = np.zeros((3, 3))
        num_cells = poly_data.GetNumberOfCells()
        print(f"Processing {num_cells} triangles for inertia tensor calculation...")

        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                # 三角形の頂点を取得（重心を原点とした座標系で）
                points = [np.array(cell.GetPoints().GetPoint(j)) - np.array(center_of_mass) for j in range(3)]

                # ミラーリングモードの場合、Y座標を反転
                if is_mirrored:
                    points = [[p[0], -p[1], p[2]] for p in points]

                # 三角形の面積と法線ベクトルを計算
                v1 = np.array(points[1]) - np.array(points[0])
                v2 = np.array(points[2]) - np.array(points[0])
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                
                if area < 1e-10:  # 極小の三角形は無視
                    continue

                # 三角形の重心を計算
                tri_centroid = np.mean(points, axis=0)
                
                # 三角形の局所的な慣性テンソルを計算
                covariance = np.zeros((3, 3))
                for p in points:
                    r_squared = np.sum(p * p)
                    for a in range(3):
                        for b in range(3):
                            if a == b:
                                # 対角成分
                                covariance[a, a] += (r_squared - p[a] * p[a]) * area / 12.0
                            else:
                                # 非対角成分（オフセット項）
                                covariance[a, b] -= (p[a] * p[b]) * area / 12.0

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

        # 数値誤差の処理
        threshold = 1e-10
        inertia_tensor[np.abs(inertia_tensor) < threshold] = 0.0

        # 対称性の確認と強制
        inertia_tensor = 0.5 * (inertia_tensor + inertia_tensor.T)

        # 対角成分が正であることを確認
        for i in range(3):
            if inertia_tensor[i, i] <= 0:
                print(f"Warning: Non-positive diagonal element detected at position ({i},{i})")
                inertia_tensor[i, i] = abs(inertia_tensor[i, i])

        return inertia_tensor

    def calculate_inertia_tensor(self):
        """
        通常モデルの慣性テンソルを計算。
        InspectorWindowクラスのメソッド。
        """
        if not self.current_node or not hasattr(self.current_node, 'stl_file'):
            print("No STL model is loaded.")
            return None

        try:
            # STLデータを取得
            if self.stl_viewer and self.current_node in self.stl_viewer.stl_actors:
                actor = self.stl_viewer.stl_actors[self.current_node]
                poly_data = actor.GetMapper().GetInput()
            else:
                print("No STL actor found for current node")
                return None

            # 体積と質量を取得
            mass_properties = vtk.vtkMassProperties()
            mass_properties.SetInputData(poly_data)
            mass_properties.Update()
            volume = mass_properties.GetVolume()
            density = float(self.density_input.text())
            mass = volume * density

            # 重心を取得
            com_filter = vtk.vtkCenterOfMass()
            com_filter.SetInputData(poly_data)
            com_filter.SetUseScalarsAsWeights(False)
            com_filter.Update()
            center_of_mass = np.array(com_filter.GetCenter())

            print("\nCalculating inertia tensor for normal model...")
            print(f"Volume: {volume:.6f}, Mass: {mass:.6f}")
            print(f"Center of Mass: {center_of_mass}")

            # 慣性テンソルを計算
            inertia_tensor = self._calculate_base_inertia_tensor(
                poly_data, mass, center_of_mass, is_mirrored=False)

            # URDFフォーマットに変換してUIを更新
            urdf_inertia = self.format_inertia_for_urdf(inertia_tensor)
            if hasattr(self, 'inertia_tensor_input'):
                self.inertia_tensor_input.setText(urdf_inertia)
                print("\nInertia tensor has been updated in UI")
            else:
                print("Warning: inertia_tensor_input not found")

            return inertia_tensor

        except Exception as e:
            print(f"Error calculating inertia tensor: {str(e)}")
            traceback.print_exc()
            return None


class SettingsDialog(QtWidgets.QDialog):
    """設定ダイアログ"""
    def __init__(self, graph, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.graph = graph
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        """UIの初期化"""
        layout = QtWidgets.QVBoxLayout(self)

        # Default Joint Limits セクション
        group_box = QtWidgets.QGroupBox("Default Joint Limits")
        group_layout = QtWidgets.QGridLayout()

        # Effort設定
        group_layout.addWidget(QtWidgets.QLabel("Default Effort:"), 0, 0)
        self.effort_input = QtWidgets.QLineEdit()
        self.effort_input.setValidator(QDoubleValidator(0.0, 1000.0, 2))
        self.effort_input.setText(str(self.graph.default_joint_effort))
        group_layout.addWidget(self.effort_input, 0, 1)

        # Velocity設定
        group_layout.addWidget(QtWidgets.QLabel("Default Velocity:"), 1, 0)
        self.velocity_input = QtWidgets.QLineEdit()
        self.velocity_input.setValidator(QDoubleValidator(0.0, 1000.0, 2))
        self.velocity_input.setText(str(self.graph.default_joint_velocity))
        group_layout.addWidget(self.velocity_input, 1, 1)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        # ボタン
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.accept_settings)
        button_layout.addWidget(ok_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def accept_settings(self):
        """設定を適用"""
        try:
            effort = float(self.effort_input.text())
            velocity = float(self.velocity_input.text())

            self.graph.default_joint_effort = effort
            self.graph.default_joint_velocity = velocity

            print(f"Settings updated: effort={effort}, velocity={velocity}")
            self.accept()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values."
            )


class STLViewerWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(STLViewerWidget, self).__init__(parent)
        self.stl_actors = {}
        self.transforms = {}
        self.base_connected_node = None
        self.text_actors = []
        self.inertial_origin_actors = {}  # Inertial Origin表示用のアクター

        layout = QtWidgets.QVBoxLayout(self)

        # Use QLabel instead of QVTKRenderWindowInteractor for M4 Mac compatibility
        self.vtk_display = QLabel(self)
        self.vtk_display.setMinimumSize(10, 10)
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
        self.render_window.AddRenderer(self.renderer)

        # No more interactor needed - using offscreen rendering
        self.iren = None

        # Initialize offscreen renderer utility
        self.offscreen_renderer = OffscreenRenderer(self.render_window, self.renderer)

        # Mouse interaction state
        self.mouse_pressed = False
        self.last_mouse_pos = None

        # Install event filter for mouse events
        self.vtk_display.installEventFilter(self)

        # ボタンとスライダーのレイアウト
        button_layout = QtWidgets.QVBoxLayout()  # 垂直レイアウトに変更

        # Wireframeスイッチ と Reset Angleボタンの横並びレイアウト
        top_buttons_layout = QtWidgets.QHBoxLayout()

        # Wireframeスイッチ（左側）
        self.wireframe_toggle = QtWidgets.QPushButton("Wireframe")
        self.wireframe_toggle.setCheckable(True)
        self.wireframe_toggle.setFixedWidth(100)
        self.wireframe_toggle.toggled.connect(self.toggle_wireframe)
        top_buttons_layout.addWidget(self.wireframe_toggle)

        top_buttons_layout.addStretch()

        # Reset Angleラベルとボタン（右側）
        reset_angle_label = QtWidgets.QLabel("Reset Angle:")
        top_buttons_layout.addWidget(reset_angle_label)

        # Frontボタン
        self.front_button = QtWidgets.QPushButton("Front")
        self.front_button.setFixedWidth(50)
        self.front_button.clicked.connect(self.reset_camera_front)
        top_buttons_layout.addWidget(self.front_button)

        # Sideボタン
        self.side_button = QtWidgets.QPushButton("Side")
        self.side_button.setFixedWidth(50)
        self.side_button.clicked.connect(self.reset_camera_side)
        top_buttons_layout.addWidget(self.side_button)

        # Topボタン
        self.top_button = QtWidgets.QPushButton("Top")
        self.top_button.setFixedWidth(50)
        self.top_button.clicked.connect(self.reset_camera_top)
        top_buttons_layout.addWidget(self.top_button)

        button_layout.addLayout(top_buttons_layout)

        # Background スライダーのレイアウト
        bg_layout = QtWidgets.QHBoxLayout()
        bg_label = QtWidgets.QLabel("background-color:")
        bg_layout.addWidget(bg_label)

        self.bg_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_slider.setMinimum(-100)  # 黒
        self.bg_slider.setMaximum(100)   # 白
        self.bg_slider.setValue(-80)      # デフォルト値を暗めに
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
        self.rotation_direction = 1  # 1: 増加方向, -1: 減少方向
        self.rotation_paused = False  # 限界点での一時停止フラグ
        self.pause_counter = 0  # 一時停止カウンター
        self.follow_children = True  # 子ノードも一緒に回転するか

        # ライティングの設定
        # 上45度前方からのメインライト
        light1 = vtk.vtkLight()
        light1.SetPosition(0.5, 0.5, 1.0)
        light1.SetIntensity(0.7)
        light1.SetLightTypeToSceneLight()
        
        # 左後方からの補助ライト
        light2 = vtk.vtkLight()
        light2.SetPosition(-1.0, -0.5, 0.2)
        light2.SetIntensity(0.7)
        light2.SetLightTypeToSceneLight()
        
        # 右後方からの補助ライト
        light3 = vtk.vtkLight()
        light3.SetPosition(0.3, -1.0, 0.2)
        light3.SetIntensity(0.7)
        light3.SetLightTypeToSceneLight()

        # 正面からの補助ライト
        light4 = vtk.vtkLight()
        light4.SetPosition(1.0, 0.0, 0.3)
        light4.SetIntensity(0.3)
        light4.SetLightTypeToSceneLight()

        # バランスの取れたアンビエント光
        self.renderer.SetAmbient(0.7, 0.7, 0.7)
        self.renderer.LightFollowCameraOff()
        self.renderer.AddLight(light1)
        self.renderer.AddLight(light2)
        self.renderer.AddLight(light3)
        self.renderer.AddLight(light4)


        # 初期の背景色を設定（暗めのグレー）
        initial_bg = (-80 + 100) / 200.0  # -80のスライダー値を0-1の範囲に変換
        self.renderer.SetBackground(initial_bg, initial_bg, initial_bg)

        # Delay initial render to avoid blocking
        QTimer.singleShot(100, self.render_to_image)

    def render_to_image(self):
        """Render VTK scene offscreen and display as image in QLabel"""
        try:
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
                    if event.button() == Qt.LeftButton:
                        self.mouse_pressed = True
                        self.last_mouse_pos = event.pos()
                        self.vtk_display.grabMouse()
                        return True

            elif event.type() == QEvent.MouseButtonRelease:
                if isinstance(event, QMouseEvent):
                    if event.button() == Qt.LeftButton:
                        self.mouse_pressed = False
                        self.last_mouse_pos = None
                        self.vtk_display.releaseMouse()
                        return True

            elif event.type() == QEvent.MouseMove:
                if isinstance(event, QMouseEvent):
                    if self.mouse_pressed and self.last_mouse_pos:
                        current_pos = event.pos()
                        dx = current_pos.x() - self.last_mouse_pos.x()
                        dy = current_pos.y() - self.last_mouse_pos.y()

                        camera = self.renderer.GetActiveCamera()
                        camera.Azimuth(-dx * 0.5)
                        camera.Elevation(dy * 0.5)
                        camera.OrthogonalizeViewUp()
                        self.renderer.ResetCameraClippingRange()

                        self.last_mouse_pos = current_pos
                        self.render_to_image()
                        return True

            elif event.type() == QEvent.Wheel:
                delta_y = event.angleDelta().y()
                camera = self.renderer.GetActiveCamera()
                current_scale = camera.GetParallelScale()

                if delta_y > 0:
                    new_scale = current_scale * 0.9
                else:
                    new_scale = current_scale * 1.1

                camera.SetParallelScale(new_scale)
                self.renderer.ResetCameraClippingRange()
                self.render_to_image()
                return True

        return super().eventFilter(obj, event)

    def store_current_transform(self, node):
        """現在の変換を保存"""
        if node in self.transforms:
            current_transform = vtk.vtkTransform()
            current_transform.DeepCopy(self.transforms[node])
            self.original_transforms[node] = current_transform

            # Followが有効な場合、子孫の変換も保存
            if self.follow_children:
                self._store_children_transforms(node)

    def _store_children_transforms(self, parent_node):
        """子孫ノードの変換を再帰的に保存"""
        for output_port in parent_node.output_ports():
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                if child_node in self.transforms and child_node not in self.original_transforms:
                    current_transform = vtk.vtkTransform()
                    current_transform.DeepCopy(self.transforms[child_node])
                    self.original_transforms[child_node] = current_transform

                    # 再帰的に孫ノードも保存
                    self._store_children_transforms(child_node)

    def start_rotation_test(self, node):
        """回転テスト開始"""
        if node in self.stl_actors:
            self.rotating_node = node
            self.current_angle = 0
            self.rotation_direction = 1  # 増加方向からスタート
            self.rotation_paused = False  # 一時停止状態をリセット
            self.pause_counter = 0
            self.rotation_timer.start(16)  # 約60FPS

    def stop_rotation_test(self, node):
        """回転テスト終了"""
        self.rotation_timer.stop()

        # 元の色と変換を必ず復元
        if self.rotating_node in self.stl_actors:
            # 元の色を復元（必ず実行）
            if hasattr(self.rotating_node, 'node_color'):
                self.stl_actors[self.rotating_node].GetProperty().SetColor(*self.rotating_node.node_color)

            # 保存された全ての変換を復元（親ノードと全子孫ノード）
            nodes_to_restore = list(self.original_transforms.keys())
            for restore_node in nodes_to_restore:
                if restore_node in self.transforms:
                    self.transforms[restore_node].DeepCopy(self.original_transforms[restore_node])
                    if restore_node in self.stl_actors:
                        self.stl_actors[restore_node].SetUserTransform(self.transforms[restore_node])
                del self.original_transforms[restore_node]

            self.render_to_image()

        self.rotating_node = None
        self.rotation_paused = False  # 一時停止状態をリセット
        self.pause_counter = 0

    def show_angle(self, node, angle_rad):
        """指定された角度でSTLモデルを表示（静止）"""
        import math

        if node not in self.stl_actors:
            return

        # 回転タイマーを停止（もし動いていたら）
        self.rotation_timer.stop()

        # 角度をラジアンから度数に変換
        angle_deg = math.degrees(angle_rad)

        # transformを取得
        transform = self.transforms[node]

        # 親の変換と、ジョイントのorigin XYZ/RPYを取得
        parent_transform = None
        joint_origin_xyz = None
        joint_origin_rpy = None

        if hasattr(node, 'graph'):
            graph = node.graph
            # ノードの入力ポートから親を探す
            for input_port in node.input_ports():
                connected_ports = input_port.connected_ports()
                if connected_ports:
                    parent_node = connected_ports[0].node()
                    parent_port_name = connected_ports[0].name()

                    # 親のポートインデックスを取得
                    parent_output_ports = list(parent_node.output_ports())
                    for port_idx, port in enumerate(parent_output_ports):
                        if port.name() == parent_port_name:
                            # 親ノードのpointsからXYZとRPYを取得
                            if hasattr(parent_node, 'points') and port_idx < len(parent_node.points):
                                point_data = parent_node.points[port_idx]
                                joint_origin_xyz = point_data.get('xyz', [0, 0, 0])
                                joint_origin_rpy = point_data.get('rpy', [0, 0, 0])

                            # 親の変換を取得
                            if parent_node in self.transforms:
                                parent_transform = self.transforms[parent_node]
                            break
                    break

        # 変換をリセット
        transform.Identity()

        # 親の変換を適用
        if parent_transform is not None:
            transform.Concatenate(parent_transform)

        # ジョイントの位置を適用
        if joint_origin_xyz:
            transform.Translate(joint_origin_xyz[0], joint_origin_xyz[1], joint_origin_xyz[2])

        # ジョイントのorigin RPYを適用（URDF仕様: Z-Y-X順）
        if joint_origin_rpy and len(joint_origin_rpy) == 3:
            roll_deg = math.degrees(joint_origin_rpy[0])
            pitch_deg = math.degrees(joint_origin_rpy[1])
            yaw_deg = math.degrees(joint_origin_rpy[2])
            transform.RotateZ(yaw_deg)    # Yaw
            transform.RotateY(pitch_deg)  # Pitch
            transform.RotateX(roll_deg)   # Roll

        # 回転軸に基づいて回転
        if hasattr(node, 'rotation_axis'):
            if node.rotation_axis == 0:    # X軸
                transform.RotateX(angle_deg)
            elif node.rotation_axis == 1:  # Y軸
                transform.RotateY(angle_deg)
            elif node.rotation_axis == 2:  # Z軸
                transform.RotateZ(angle_deg)

        self.stl_actors[node].SetUserTransform(transform)
        self.render_to_image()

        print(f"Showing angle: {angle_rad} rad ({angle_deg} deg)")

    def show_inertial_origin(self, node, xyz):
        """Inertial Originの位置に赤い点とXYZ座標軸を表示（ローカル座標系）"""
        # 既存の表示があれば削除
        self.hide_inertial_origin(node)

        # ノードのtransformを取得
        if node not in self.transforms:
            print(f"Node {node.name()} has no transform")
            return

        node_transform = self.transforms[node]

        # アクターのリストを作成
        actors = []

        # 1. 赤い球を作成（ローカル座標）
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(xyz[0], xyz[1], xyz[2])
        sphere.SetRadius(0.005)  # 小さな球
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 赤色
        sphere_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(sphere_actor)
        actors.append(sphere_actor)

        # 2. XYZ軸を作成（ローカル座標）
        axis_length = 0.03  # 軸の長さ

        # X軸（赤）
        x_line = vtk.vtkLineSource()
        x_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        x_line.SetPoint2(xyz[0] + axis_length, xyz[1], xyz[2])

        x_mapper = vtk.vtkPolyDataMapper()
        x_mapper.SetInputConnection(x_line.GetOutputPort())

        x_actor = vtk.vtkActor()
        x_actor.SetMapper(x_mapper)
        x_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 赤
        x_actor.GetProperty().SetLineWidth(3)
        x_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(x_actor)
        actors.append(x_actor)

        # Y軸（緑）
        y_line = vtk.vtkLineSource()
        y_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        y_line.SetPoint2(xyz[0], xyz[1] + axis_length, xyz[2])

        y_mapper = vtk.vtkPolyDataMapper()
        y_mapper.SetInputConnection(y_line.GetOutputPort())

        y_actor = vtk.vtkActor()
        y_actor.SetMapper(y_mapper)
        y_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 緑
        y_actor.GetProperty().SetLineWidth(3)
        y_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(y_actor)
        actors.append(y_actor)

        # Z軸（青）
        z_line = vtk.vtkLineSource()
        z_line.SetPoint1(xyz[0], xyz[1], xyz[2])
        z_line.SetPoint2(xyz[0], xyz[1], xyz[2] + axis_length)

        z_mapper = vtk.vtkPolyDataMapper()
        z_mapper.SetInputConnection(z_line.GetOutputPort())

        z_actor = vtk.vtkActor()
        z_actor.SetMapper(z_mapper)
        z_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 青
        z_actor.GetProperty().SetLineWidth(3)
        z_actor.SetUserTransform(node_transform)  # ノードのtransformを適用

        self.renderer.AddActor(z_actor)
        actors.append(z_actor)

        # アクターを保存
        self.inertial_origin_actors[node] = actors

        # 再描画
        self.render_to_image()

    def hide_inertial_origin(self, node):
        """Inertial Originの表示を削除"""
        if node in self.inertial_origin_actors:
            for actor in self.inertial_origin_actors[node]:
                self.renderer.RemoveActor(actor)
            del self.inertial_origin_actors[node]
            self.render_to_image()

    def update_rotation(self):
        """回転更新"""
        if self.rotating_node and self.rotating_node in self.stl_actors:
            node = self.rotating_node
            transform = self.transforms[node]

            # 現在の位置を保持
            position = transform.GetPosition()

            # Fixedモードかどうかをチェック
            is_fixed = hasattr(node, 'rotation_axis') and node.rotation_axis == 3

            # Fixedモードの場合は点滅のみ
            if is_fixed:
                # 点滅効果（400msごとに切り替え）
                # フレームレート60FPSとして、24フレーム（約400ms）ごとに切り替え
                is_red = (self.current_angle // 24) % 2 == 0
                if is_red:
                    self.stl_actors[node].GetProperty().SetColor(1.0, 0.0, 0.0)  # 赤
                else:
                    self.stl_actors[node].GetProperty().SetColor(1.0, 1.0, 1.0)  # 白
            else:
                # Joint limitを取得（ラジアンから度数に変換）
                import math
                lower_deg = math.degrees(getattr(node, 'joint_lower', -3.14159))
                upper_deg = math.degrees(getattr(node, 'joint_upper', 3.14159))

                # 一時停止中の処理
                if self.rotation_paused:
                    self.pause_counter += 1
                    # 0.5秒（60FPS × 0.5 = 30フレーム）経過したら再開
                    if self.pause_counter >= 30:
                        self.rotation_paused = False
                        self.pause_counter = 0
                    # 一時停止中は角度を更新せず、現在の角度を維持
                else:
                    # 角度更新（往復運動）
                    angle_step = 2.0  # 1フレームあたりの角度変化（度）
                    self.current_angle += angle_step * self.rotation_direction

                    # 範囲チェックと方向反転
                    if self.current_angle >= upper_deg:
                        self.current_angle = upper_deg
                        self.rotation_direction = -1  # 減少方向へ
                        self.rotation_paused = True  # 一時停止開始
                        self.pause_counter = 0
                    elif self.current_angle <= lower_deg:
                        self.current_angle = lower_deg
                        self.rotation_direction = 1   # 増加方向へ
                        self.rotation_paused = True  # 一時停止開始
                        self.pause_counter = 0

                # 通常の回転処理
                transform.Identity()  # 変換をリセット

                # 親の変換と、ジョイントのorigin XYZ/RPYを復元
                # ノードの親接続を探す
                parent_transform = None
                joint_origin_xyz = None
                joint_origin_rpy = None

                if hasattr(node, 'graph'):
                    graph = node.graph
                    # ノードの入力ポートから親を探す
                    for input_port in node.input_ports():
                        connected_ports = input_port.connected_ports()
                        if connected_ports:
                            parent_node = connected_ports[0].node()
                            parent_port_name = connected_ports[0].name()

                            # 親のポートインデックスを取得
                            parent_output_ports = list(parent_node.output_ports())
                            for port_idx, port in enumerate(parent_output_ports):
                                if port.name() == parent_port_name:
                                    # 親ノードのpointsからXYZとRPYを取得
                                    if hasattr(parent_node, 'points') and port_idx < len(parent_node.points):
                                        point_data = parent_node.points[port_idx]
                                        joint_origin_xyz = point_data.get('xyz', [0, 0, 0])
                                        joint_origin_rpy = point_data.get('rpy', [0, 0, 0])

                                    # 親の変換を取得
                                    if parent_node in self.transforms:
                                        parent_transform = self.transforms[parent_node]
                                    break
                            break

                # 親の変換を適用
                if parent_transform is not None:
                    transform.Concatenate(parent_transform)

                # ジョイントの位置を適用
                if joint_origin_xyz:
                    transform.Translate(joint_origin_xyz[0], joint_origin_xyz[1], joint_origin_xyz[2])

                # ジョイントのorigin RPYを適用（URDF仕様: Z-Y-X順）
                if joint_origin_rpy and len(joint_origin_rpy) == 3:
                    roll_deg = math.degrees(joint_origin_rpy[0])
                    pitch_deg = math.degrees(joint_origin_rpy[1])
                    yaw_deg = math.degrees(joint_origin_rpy[2])
                    transform.RotateZ(yaw_deg)    # Yaw
                    transform.RotateY(pitch_deg)  # Pitch
                    transform.RotateX(roll_deg)   # Roll

                # 回転軸に基づいて回転テストの角度を適用
                if hasattr(node, 'rotation_axis'):
                    if node.rotation_axis == 0:    # X軸
                        transform.RotateX(self.current_angle)
                    elif node.rotation_axis == 1:  # Y軸
                        transform.RotateY(self.current_angle)
                    elif node.rotation_axis == 2:  # Z軸
                        transform.RotateZ(self.current_angle)

                self.stl_actors[node].SetUserTransform(transform)

                # Followが有効な場合、子孫ノードも一緒に回転
                if self.follow_children and hasattr(node, 'graph'):
                    self._rotate_children(node, transform)

            self.render_to_image()

    def _rotate_children(self, parent_node, parent_transform):
        """子孫ノードを親の回転に追従させて回転"""
        import math

        # 親ノードの全出力ポートをチェック
        for port_idx, output_port in enumerate(parent_node.output_ports()):
            # 接続されている子ノードを取得
            for connected_port in output_port.connected_ports():
                child_node = connected_port.node()

                # 子ノードがSTLを持っている場合のみ処理
                if child_node not in self.stl_actors or child_node not in self.transforms:
                    continue

                # 子ノードのジョイント情報を取得
                child_xyz = [0, 0, 0]
                child_rpy = [0, 0, 0]

                if hasattr(parent_node, 'points') and port_idx < len(parent_node.points):
                    point_data = parent_node.points[port_idx]
                    child_xyz = point_data.get('xyz', [0, 0, 0])
                    child_rpy = point_data.get('rpy', [0, 0, 0])

                # 子ノードの変換を更新
                child_transform = self.transforms[child_node]
                child_transform.Identity()

                # 親の変換を適用（回転テストの回転を含む）
                child_transform.Concatenate(parent_transform)

                # ジョイントの位置を適用
                child_transform.Translate(child_xyz[0], child_xyz[1], child_xyz[2])

                # ジョイントのorigin RPYを適用
                if len(child_rpy) == 3:
                    roll_deg = math.degrees(child_rpy[0])
                    pitch_deg = math.degrees(child_rpy[1])
                    yaw_deg = math.degrees(child_rpy[2])
                    child_transform.RotateZ(yaw_deg)
                    child_transform.RotateY(pitch_deg)
                    child_transform.RotateX(roll_deg)

                # 変換を適用
                self.stl_actors[child_node].SetUserTransform(child_transform)

                # 再帰的に孫ノードも回転
                self._rotate_children(child_node, child_transform)

    def _get_scene_bounds_and_center(self):
        """シーンのバウンディングボックスと中心を計算"""
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
        """Front view（正面図）"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            self.setup_camera()
            return

        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()

        distance = diagonal
        camera.SetPosition(center[0] + distance, center[1], center[2])
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        camera.SetParallelScale(diagonal * 0.5)

        self.renderer.ResetCameraClippingRange()
        self.render_to_image()
        print("Camera reset to Front view")

    def reset_camera_side(self):
        """Side view（側面図）"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            self.setup_camera()
            return

        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()

        distance = diagonal
        camera.SetPosition(center[0], center[1] + distance, center[2])
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        camera.SetParallelScale(diagonal * 0.5)

        self.renderer.ResetCameraClippingRange()
        self.render_to_image()
        print("Camera reset to Side view")

    def reset_camera_top(self):
        """Top view（上面図）"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            self.setup_camera()
            return

        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()

        distance = diagonal
        camera.SetPosition(center[0], center[1], center[2] + distance)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 1, 0)  # Top viewではY軸が上
        camera.SetParallelScale(diagonal * 0.5)

        self.renderer.ResetCameraClippingRange()
        self.render_to_image()
        print("Camera reset to Top view")

    def reset_camera(self):
        """カメラビューをリセット（Front viewと同じ）"""
        self.reset_camera_front()

    def reset_view_to_fit(self):
        """すべてのSTLモデルが見えるようにビューをリセットして調整"""
        self.reset_camera()
        self.render_to_image()

    def toggle_wireframe(self, checked):
        """Wireframe表示モードの切り替え"""
        if checked:
            # Wireframeモード: エッジのみ表示、面は透明
            for node, actor in self.stl_actors.items():
                actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetLineWidth(1)
            print("Wireframe mode ON")
        else:
            # Surfaceモード: 通常の表示
            for node, actor in self.stl_actors.items():
                actor.GetProperty().SetRepresentationToSurface()
            print("Wireframe mode OFF")

        # 再描画
        self.render_to_image()

    def create_coordinate_axes(self):
        """座標軸の作成（線と独立したテキスト）"""
        base_assembly = vtk.vtkAssembly()
        length = 0.1
        text_offset = 0.02
        
        # ラインの作成部分は変更なし
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

        # テキスト部分をvtkBillboardTextActor3Dに変更
        for i, (color, label) in enumerate([
            ((1,0,0), "X"),
            ((0,1,0), "Y"),
            ((0,0,1), "Z")
        ]):
            text_position = [0, 0, 0]
            text_position[i] = length + text_offset
            
            text_actor = vtk.vtkBillboardTextActor3D()  # vtkTextActor3Dから変更
            text_actor.SetInput(label)
            text_actor.SetPosition(*text_position)
            text_actor.GetTextProperty().SetColor(*color)
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.GetTextProperty().SetJustificationToCentered()
            text_actor.GetTextProperty().SetVerticalJustificationToCentered()
            text_actor.SetScale(0.02)  # 単一の値でスケールを設定
            
            self.renderer.AddActor(text_actor)
            if not hasattr(self, 'text_actors'):
                self.text_actors = []
            self.text_actors.append(text_actor)
        
        return base_assembly

    def update_coordinate_axes(self, position):
        """座標軸とテキストの位置を更新"""
        # ラインの位置を更新
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.Translate(position[0], position[1], position[2])
        self.coordinate_axes_actor.SetUserTransform(transform)
        
        # テキストの位置を更新
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
        """STLの位置と回転を更新"""
        # base_linkでblank_linkがTrueの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node in self.stl_actors and node in self.transforms:
            print(f"Updating transform for node {node.name()} to position {point_xyz}, rotation {point_rpy}")
            transform = self.transforms[node]
            transform.Identity()

            # 親の変換を適用（累積変換用）
            if parent_transform is not None:
                transform.Concatenate(parent_transform)

            # ジョイントの位置を適用
            transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])

            # ジョイントの回転を適用（RPY: Roll-Pitch-Yaw）
            if point_rpy is not None and len(point_rpy) == 3:
                # RPYはラジアン単位なので度に変換
                import math
                roll_deg = math.degrees(point_rpy[0])
                pitch_deg = math.degrees(point_rpy[1])
                yaw_deg = math.degrees(point_rpy[2])

                # RPY順序で回転を適用（URDF仕様: Z-Y-X順）
                transform.RotateZ(yaw_deg)    # Yaw
                transform.RotateY(pitch_deg)  # Pitch
                transform.RotateX(roll_deg)   # Roll

            self.stl_actors[node].SetUserTransform(transform)

            # base_linkに接続された最初のノードの場合、座標軸も更新
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
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No STL actor or transform found for node {node.name()}")

    def reset_stl_transform(self, node):
        """STLの位置をリセット"""
        # base_linkでblank_linkがTrueの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node in self.transforms:
            print(f"Resetting transform for node {node.name()}")
            transform = self.transforms[node]
            transform.Identity()
            
            self.stl_actors[node].SetUserTransform(transform)
            
            # 座標軸のリセット（必要な場合）
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None
            
            self.render_to_image()
        else:
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No transform found for node {node.name()}")

    def load_mesh_file(self, file_path):
        """メッシュファイル（.stl, .dae）を読み込んでVTK PolyDataを返す"""
        import os
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.stl':
            # STLファイルの読み込み
            reader = vtk.vtkSTLReader()
            reader.SetFileName(file_path)
            reader.Update()
            return reader.GetOutput()

        elif file_ext == '.dae':
            # COLLADAファイルの読み込み（trimeshを使用）
            try:
                import trimesh
                import numpy as np

                print(f"\n=== Loading COLLADA file with trimesh ===")
                print(f"File path: {file_path}")

                # trimeshでCOLLADAファイルを読み込み
                mesh = trimesh.load(file_path, force='mesh')

                print(f"Loaded mesh type: {type(mesh)}")

                # 複数のメッシュがある場合は結合
                if isinstance(mesh, trimesh.Scene):
                    print(f"Scene contains {len(mesh.geometry)} geometries")
                    meshes_list = []
                    for name, geom in mesh.geometry.items():
                        print(f"  Geometry '{name}': {type(geom)}, vertices: {len(geom.vertices) if hasattr(geom, 'vertices') else 'N/A'}")
                        if isinstance(geom, trimesh.Trimesh):
                            meshes_list.append(geom)

                    if len(meshes_list) == 0:
                        print("Error: No valid trimesh geometries found in scene")
                        return None

                    print(f"Concatenating {len(meshes_list)} meshes...")
                    mesh = trimesh.util.concatenate(meshes_list)

                # メッシュ情報を表示
                print(f"Final mesh - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

                # メッシュの境界を表示（デバッグ用）
                bounds = mesh.bounds
                print(f"Mesh bounds: min={bounds[0]}, max={bounds[1]}")

                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    print("Error: Mesh has no vertices or faces")
                    return None

                # VTK PolyDataに変換
                print("Converting to VTK PolyData...")
                polydata = vtk.vtkPolyData()

                # 頂点データを設定
                points = vtk.vtkPoints()
                points.SetNumberOfPoints(len(mesh.vertices))
                for i, vertex in enumerate(mesh.vertices):
                    points.SetPoint(i, float(vertex[0]), float(vertex[1]), float(vertex[2]))
                polydata.SetPoints(points)
                print(f"Added {points.GetNumberOfPoints()} points to PolyData")

                # 面データを設定
                cells = vtk.vtkCellArray()
                for face in mesh.faces:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, int(face[0]))
                    triangle.GetPointIds().SetId(1, int(face[1]))
                    triangle.GetPointIds().SetId(2, int(face[2]))
                    cells.InsertNextCell(triangle)
                polydata.SetPolys(cells)
                print(f"Added {polydata.GetNumberOfCells()} cells to PolyData")

                # 法線ベクトルを計算
                print("Computing normals...")
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(polydata)
                normals.ComputePointNormalsOn()
                normals.ComputeCellNormalsOn()
                normals.Update()

                result = normals.GetOutput()
                print(f"Final PolyData: {result.GetNumberOfPoints()} points, {result.GetNumberOfCells()} cells")

                # 境界を確認
                result_bounds = result.GetBounds()
                print(f"PolyData bounds: X({result_bounds[0]:.3f}, {result_bounds[1]:.3f}), Y({result_bounds[2]:.3f}, {result_bounds[3]:.3f}), Z({result_bounds[4]:.3f}, {result_bounds[5]:.3f})")
                print("=== COLLADA loading complete ===\n")

                return result

            except ImportError as e:
                print(f"Error: trimesh library is not installed or missing dependencies: {str(e)}")
                print("Install with: pip install trimesh")
                import traceback
                traceback.print_exc()
                return None
            except Exception as e:
                print(f"Error loading COLLADA file: {str(e)}")
                import traceback
                traceback.print_exc()
                return None

        else:
            print(f"Unsupported file format: {file_ext}")
            return None

    def load_stl_for_node(self, node):
        """ノード用のメッシュファイル（.stl, .dae）を読み込む（色の適用を含む）"""
        # base_linkでblank_linkがTrueの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node.stl_file:
            print(f"\n=== Loading mesh for node ===")
            print(f"Node name: {node.name()}")
            print(f"Mesh file: {node.stl_file}")

            # メッシュファイルを読み込み
            polydata = self.load_mesh_file(node.stl_file)

            if polydata is None:
                print(f"ERROR: Failed to load mesh file: {node.stl_file}")
                return

            print(f"PolyData loaded successfully: {polydata.GetNumberOfPoints()} points, {polydata.GetNumberOfCells()} cells")

            # マッパーを作成
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            print("Mapper created and connected to PolyData")

            # アクターを作成
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            print("Actor created and connected to mapper")

            # 変換行列を作成
            transform = vtk.vtkTransform()
            transform.Identity()
            actor.SetUserTransform(transform)
            print("Transform created and set to identity")

            # 既存のアクターを削除
            if node in self.stl_actors:
                print(f"Removing existing actor for node {node.name()}")
                self.renderer.RemoveActor(self.stl_actors[node])

            # アクターを保存してレンダラーに追加
            self.stl_actors[node] = actor
            self.transforms[node] = transform
            self.renderer.AddActor(actor)
            print(f"Actor added to renderer. Total actors in renderer: {self.renderer.GetActors().GetNumberOfItems()}")

            # アクターが可視かどうかを確認
            print(f"Actor visibility: {actor.GetVisibility()}")
            print(f"Actor opacity: {actor.GetProperty().GetOpacity()}")

            # ノードの色情報を適用
            self.apply_color_to_node(node)

            # カメラをリセットして再レンダリング
            self.reset_camera()
            self.render_to_image()
            print(f"Mesh file loaded and rendered: {node.stl_file}")
            print("=== Loading complete ===\n")

    def apply_color_to_node(self, node):
        """ノードのSTLモデルに色を適用"""
        if node in self.stl_actors:
            # デフォルトの色を設定（色情報がない場合）
            if not hasattr(node, 'node_color') or node.node_color is None:
                node.node_color = DEFAULT_COLOR_WHITE.copy()  # 白色をデフォルトに

            # 色の適用
            actor = self.stl_actors[node]
            actor.GetProperty().SetColor(*node.node_color)
            print(f"Applied color to node {node.name()}: RGB({node.node_color[0]:.3f}, {node.node_color[1]:.3f}, {node.node_color[2]:.3f})")
            self.render_to_image()

    def remove_stl_for_node(self, node):
        """ノードのSTLを削除"""
        if node in self.stl_actors:
            self.renderer.RemoveActor(self.stl_actors[node])
            del self.stl_actors[node]
            if node in self.transforms:
                del self.transforms[node]
                
            # 座標軸のリセット（必要な場合）
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None
                
            self.render_to_image()
            print(f"Removed STL for node: {node.name()}")

    def setup_camera(self):
        """カメラの初期設定"""
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetPosition(1, 0, 0)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)

    def cleanup(self):
        """STLビューアのリソースをクリーンアップ"""
        # VTKオブジェクトの解放
        if hasattr(self, 'renderer'):
            if self.renderer:
                # アクターの削除
                for actor in self.renderer.GetActors():
                    self.renderer.RemoveActor(actor)

                # テキストアクターの削除
                for actor in self.text_actors:
                    self.renderer.RemoveActor(actor)
                self.text_actors.clear()

        # レンダーウィンドウのクリーンアップ
        if hasattr(self, 'render_window'):
            if self.render_window:
                self.render_window.Finalize()

        # 参照の解放
        self.stl_actors.clear()
        self.transforms.clear()

    def __del__(self):
        """デストラクタでクリーンアップを実行"""
        self.cleanup()

    def update_rotation_axis(self, node, axis_id):
        """ノードの回転軸を更新"""
        try:
            print(f"Updating rotation axis for node {node.name()} to axis {axis_id}")
            
            if node in self.stl_actors and node in self.transforms:
                transform = self.transforms[node]
                actor = self.stl_actors[node]
                
                # 現在の位置を保持
                current_position = list(actor.GetPosition())
                
                # 変換をリセット
                transform.Identity()
                
                # 位置を再設定
                transform.Translate(*current_position)
                
                # 新しい回転軸に基づいて回転を設定
                # 必要に応じてここに回転の処理を追加
                
                # 変換を適用
                actor.SetUserTransform(transform)
                
                # ビューを更新
                self.render_to_image()
                print(f"Successfully updated rotation axis for node {node.name()}")
            else:
                print(f"No STL actor or transform found for node {node.name()}")
                
        except Exception as e:
            print(f"Error updating rotation axis: {str(e)}")
            traceback.print_exc()

    def update_background(self, value):
        """背景色をスライダーの値に基づいて更新"""
        # -100から100の値を0から1の範囲に変換
        normalized_value = (value + 100) / 200.0
        self.renderer.SetBackground(normalized_value, normalized_value, normalized_value)
        self.render_to_image()

    def open_urdf_loader_website(self):
        """URDF Loadersのウェブサイトを開く"""
        url = QtCore.QUrl(
            "https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/")
        QtGui.QDesktopServices.openUrl(url)

class CustomNodeGraph(NodeGraph):
    def __init__(self, stl_viewer):
        super(CustomNodeGraph, self).__init__()
        self.stl_viewer = stl_viewer
        self.robot_name = "robot_x"
        self.project_dir = None
        self.meshes_dir = None
        self.last_save_dir = None

        # グローバルデフォルト値（ジョイント制限パラメータ）
        self.default_joint_effort = DEFAULT_JOINT_EFFORT
        self.default_joint_velocity = DEFAULT_JOINT_VELOCITY

        # ポート接続/切断のシグナルを接続
        self.port_connected.connect(self.on_port_connected)
        self.port_disconnected.connect(self.on_port_disconnected)

        # ノードタイプの登録
        try:
            # BaseLinkNodeの登録
            self.register_node(BaseLinkNode)
            print(f"Registered node type: {BaseLinkNode.NODE_NAME}")

            # FooNodeの登録
            self.register_node(FooNode)
            print(f"Registered node type: {FooNode.NODE_NAME}")

        except Exception as e:
            print(f"Error registering node types: {str(e)}")
            import traceback
            traceback.print_exc()

        # 他の初期化コード...
        self._cleanup_handlers = []
        self._cached_positions = {}
        self._selection_cache = set()

        # 選択関連の変数を初期化
        self._selection_start = None
        self._is_selecting = False

        # パン操作関連の変数を初期化
        self._is_panning = False
        self._pan_start = None

        # ビューの設定
        self._view = self.widget

        # 実際のQGraphicsViewにアクセス（NodeGraphWidgetはQTabWidget）
        self._viewer = self._view.currentWidget()  # NodeViewerを取得

        # NodeGraphQtの組み込み矩形選択の無効化を試みる
        # （PySide6互換性問題を回避するため）
        try:
            if self._viewer:
                self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
                print("NodeGraphQt rubber band selection disabled")
            else:
                print("Could not access internal viewer, will handle errors in event handlers")
        except Exception as e:
            print(f"Could not disable NodeGraphQt drag mode: {e}")

        # ラバーバンドの作成
        self._rubber_band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle,
            self._view
        )

        # オリジナルのイベントハンドラを保存
        if self._viewer:
            self._original_handlers = {
                'press': self._viewer.mousePressEvent,
                'move': self._viewer.mouseMoveEvent,
                'release': self._viewer.mouseReleaseEvent,
                'keyPress': self._viewer.keyPressEvent
            }

            # 新しいイベントハンドラを設定
            self._viewer.mousePressEvent = self.custom_mouse_press
            self._viewer.mouseMoveEvent = self.custom_mouse_move
            self._viewer.mouseReleaseEvent = self.custom_mouse_release
            self._viewer.keyPressEvent = self.custom_key_press
        else:
            self._original_handlers = {}

        # インスペクタウィンドウの初期化
        self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)

    def custom_mouse_press(self, event):
        """カスタムマウスプレスイベントハンドラ"""
        try:
            print("\n=== Mouse Press Event ===")
            print(f"Button: {event.button()}")
            print(f"Modifiers: {event.modifiers()}")
            print(f"Alt/Option pressed: {bool(event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)}")

            # 中ボタンでパン操作を開始（カスタム実装）
            if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                print(">>> Starting pan operation (Middle Button Drag) - using custom panning")
                self._is_panning = True
                self._pan_start = event.position().toPoint()
                self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                return

            # 左ボタンの処理
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # Option (Alt) + 左ボタンでもパン操作を開始（トラックパッド用）
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    print(">>> Starting pan operation (Option+Drag) - using Qt ScrollHandDrag")
                    self._is_panning = True
                    # Qt組み込みのドラッグパンモードに切り替え
                    self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
                    self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    # 元のハンドラを呼び出してQt標準の処理を実行
                    self._original_handlers['press'](event)
                    return

                # クリック位置にアイテムがあるかチェック
                pos = event.position().toPoint()
                print(f"Click position (view): {pos}")

                scene_pos = self._viewer.mapToScene(pos)
                print(f"Click position (scene): {scene_pos}")

                item_at_pos = self._viewer.scene().itemAt(scene_pos, self._viewer.transform())
                print(f"Item at position: {item_at_pos}")
                print(f"Item type: {type(item_at_pos)}")

                # 空白部分をクリックした場合のみラバーバンド選択を開始
                if item_at_pos is None or item_at_pos == self._viewer.scene():
                    print(">>> Starting rubber band selection")
                    self._selection_start = pos
                    self._is_selecting = True

                    # ラバーバンドの設定
                    if self._rubber_band:
                        rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                        self._rubber_band.setGeometry(rect)
                        self._rubber_band.show()
                        print(f"Rubber band shown at: {rect}")

                    # Ctrlキーが押されていない場合は選択をクリア
                    if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                        print("Clearing previous selection")
                        for node in self.selected_nodes():
                            node.set_selected(False)

                    # ラバーバンド選択中はオリジナルハンドラを呼ばない
                    return
                else:
                    print(">>> Item found at click position, not starting rubber band selection")

            # ラバーバンド選択以外の場合はオリジナルのイベントハンドラを呼び出し
            print("Calling original press handler")
            try:
                self._original_handlers['press'](event)
            except TypeError as te:
                # NodeGraphQtのPySide6互換性問題を無視
                print(f"Ignoring NodeGraphQt compatibility error: {te}")

        except Exception as e:
            print(f"Error in mouse press: {str(e)}")
            import traceback
            traceback.print_exc()

    def custom_mouse_move(self, event):
        """カスタムマウス移動イベントハンドラ"""
        try:
            # パン操作中の処理（Qt標準のScrollHandDragを使用）
            if self._is_panning:
                # Qt標準のScrollHandDragを使用するため、元のハンドラを呼び出す
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

                # ラバーバンド選択中はオリジナルハンドラを呼ばない
                return

            # ラバーバンド選択中でない場合はオリジナルのイベントハンドラを呼び出し
            try:
                self._original_handlers['move'](event)
            except TypeError as te:
                # NodeGraphQtのPySide6互換性問題を無視
                pass  # Move中は大量のエラーが出るので表示しない

        except Exception as e:
            print(f"Error in mouse move: {str(e)}")
            import traceback
            traceback.print_exc()

    def custom_mouse_release(self, event):
        """カスタムマウスリリースイベントハンドラ"""
        try:
            print("\n=== Mouse Release Event ===")
            print(f"Button: {event.button()}")
            print(f"Is selecting: {self._is_selecting}")
            print(f"Is panning: {self._is_panning}")

            # パン操作の終了（中ボタンまたはOption+左ボタン）
            if self._is_panning and (event.button() == QtCore.Qt.MouseButton.MiddleButton or
                                      event.button() == QtCore.Qt.MouseButton.LeftButton):
                print(">>> Ending pan operation")
                # 元のハンドラを呼び出してQt標準の処理を完了
                try:
                    self._original_handlers['release'](event)
                except TypeError:
                    pass
                # ドラッグモードを元に戻す
                self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
                self._viewer.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                self._is_panning = False
                self._pan_start = None
                return

            if event.button() == QtCore.Qt.MouseButton.LeftButton and self._is_selecting:
                if self._rubber_band and self._selection_start:
                    # 選択範囲の処理
                    rect = self._rubber_band.geometry()
                    scene_rect = self._viewer.mapToScene(rect).boundingRect()
                    print(f"Selection rect (view): {rect}")
                    print(f"Selection rect (scene): {scene_rect}")

                    # 範囲内のノードを選択
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

                    # ラバーバンドを隠す
                    self._rubber_band.hide()

                # 選択状態をリセット
                self._selection_start = None
                self._is_selecting = False

                # ラバーバンド選択を実行した場合はオリジナルハンドラを呼ばない
                return

            # ラバーバンド選択以外の場合はオリジナルのイベントハンドラを呼び出し
            print("Calling original release handler")
            try:
                self._original_handlers['release'](event)
            except TypeError as te:
                # NodeGraphQtのPySide6互換性問題を無視
                print(f"Ignoring NodeGraphQt compatibility error: {te}")

        except Exception as e:
            print(f"Error in mouse release: {str(e)}")
            import traceback
            traceback.print_exc()

    def custom_key_press(self, event):
        """カスタムキープレスイベントハンドラ"""
        try:
            # DeleteキーまたはBackspaceキーが押された場合
            if event.key() in [QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace]:
                print("\n=== Delete/Backspace Key Pressed ===")
                # 選択されているノードを削除
                delete_selected_node(self)
                # イベントを処理済みとしてマーク
                event.accept()
                return

            # その他のキーイベントは元のハンドラに渡す
            try:
                self._original_handlers['keyPress'](event)
            except (TypeError, KeyError):
                # オリジナルハンドラがない場合は何もしない
                pass

        except Exception as e:
            print(f"Error in key press: {str(e)}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            print("Starting cleanup process...")
            
            # イベントハンドラの復元
            if hasattr(self, '_viewer') and self._viewer:
                if hasattr(self, '_original_handlers'):
                    self._viewer.mousePressEvent = self._original_handlers['press']
                    self._viewer.mouseMoveEvent = self._original_handlers['move']
                    self._viewer.mouseReleaseEvent = self._original_handlers['release']
                    if 'keyPress' in self._original_handlers:
                        self._viewer.keyPressEvent = self._original_handlers['keyPress']
                    print("Restored original event handlers")

            # ラバーバンドのクリーンアップ
            try:
                if hasattr(self, '_rubber_band') and self._rubber_band and not self._rubber_band.isHidden():
                    self._rubber_band.hide()
                    self._rubber_band.setParent(None)
                    self._rubber_band.deleteLater()
                    self._rubber_band = None
                    print("Cleaned up rubber band")
            except Exception as e:
                print(f"Warning: Rubber band cleanup - {str(e)}")
                
            # ノードのクリーンアップ
            for node in self.all_nodes():
                try:
                    # STLデータのクリーンアップ
                    if self.stl_viewer:
                        self.stl_viewer.remove_stl_for_node(node)
                    # ノードの削除
                    self.remove_node(node)
                except Exception as e:
                    print(f"Error cleaning up node: {str(e)}")

            # インスペクタウィンドウのクリーンアップ
            if hasattr(self, 'inspector_window') and self.inspector_window:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                    self.inspector_window = None
                    print("Cleaned up inspector window")
                except Exception as e:
                    print(f"Error cleaning up inspector window: {str(e)}")

            # キャッシュのクリア
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
        """デストラクタでクリーンアップを実行"""
        self.cleanup()

    def remove_node(self, node):
        """ノード削除時のメモリリーク対策"""
        # キャッシュからノード関連データを削除
        if node in self._cached_positions:
            del self._cached_positions[node]
        self._selection_cache.discard(node)

        # ポート接続の解除
        for port in node.input_ports():
            for connected_port in port.connected_ports():
                self.disconnect_ports(port, connected_port)
        
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                self.disconnect_ports(port, connected_port)

        # STLデータのクリーンアップ
        if self.stl_viewer:
            self.stl_viewer.remove_stl_for_node(node)

        super(CustomNodeGraph, self).remove_node(node)

    def optimize_node_positions(self):
        """ノード位置の計算を最適化"""
        # 位置計算のキャッシュを活用
        for node in self.all_nodes():
            if node not in self._cached_positions:
                pos = self.calculate_node_position(node)
                self._cached_positions[node] = pos
            node.set_pos(*self._cached_positions[node])

    def setup_custom_view(self):
        """ビューのイベントハンドラをカスタマイズ"""
        # オリジナルのイベントハンドラを保存
        self._view.mousePressEvent_original = self._view.mousePressEvent
        self._view.mouseMoveEvent_original = self._view.mouseMoveEvent
        self._view.mouseReleaseEvent_original = self._view.mouseReleaseEvent
        
        # 新しいイベントハンドラを設定
        self._view.mousePressEvent = lambda event: self._view_mouse_press(event)
        self._view.mouseMoveEvent = lambda event: self._view_mouse_move(event)
        self._view.mouseReleaseEvent = lambda event: self._view_mouse_release(event)

    def eventFilter(self, obj, event):
        """イベントフィルターでマウスイベントを処理"""
        if obj is self._view:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                return self._handle_mouse_press(event)
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                return self._handle_mouse_move(event)
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                return self._handle_mouse_release(event)
        
        return super(CustomNodeGraph, self).eventFilter(obj, event)

    def _handle_mouse_press(self, event):
        """マウスプレスイベントの処理"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Option (Alt)キーが押されている場合はパン操作なのでラバーバンド選択を開始しない
            if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                return

            # パン操作中の場合はラバーバンド選択を開始しない
            if self._is_panning:
                return

            self._selection_start = event.position().toPoint()
            self._is_selecting = True

            # 選択範囲の設定
            if self._rubber_band:
                rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                self._rubber_band.setGeometry(rect)
                self._rubber_band.show()

            # Ctrlキーが押されていない場合は既存の選択をクリア
            if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)

        return False  # イベントを伝播させる

    def _handle_mouse_move(self, event):
        """マウス移動イベントの処理"""
        # パン操作中はラバーバンドを更新しない
        if self._is_panning:
            return False

        if self._is_selecting and self._selection_start is not None and self._rubber_band:
            current_pos = event.position().toPoint()
            rect = QtCore.QRect(self._selection_start,
                                current_pos).normalized()
            self._rubber_band.setGeometry(rect)

        return False  # イベントを伝播させる

    def _handle_mouse_release(self, event):
        """マウスリリースイベントの処理"""
        # パン操作中はラバーバンド処理をしない
        if self._is_panning:
            return False

        if (event.button() == QtCore.Qt.MouseButton.LeftButton and
                self._is_selecting and self._rubber_band):
            try:
                # 選択範囲の取得
                rect = self._rubber_band.geometry()
                scene_rect = self._view.mapToScene(rect).boundingRect()

                # 範囲内のノードを選択
                for node in self.all_nodes():
                    node_pos = node.pos()
                    if isinstance(node_pos, (list, tuple)):
                        node_point = QtCore.QPointF(node_pos[0], node_pos[1])
                    else:
                        node_point = node_pos

                    if scene_rect.contains(node_point):
                        node.set_selected(True)

                # ラバーバンドを隠す
                self._rubber_band.hide()

            except Exception as e:
                print(f"Error in mouse release: {str(e)}")
            finally:
                # 状態をリセット
                self._selection_start = None
                self._is_selecting = False

        return False  # イベントを伝播させる

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # ビューの座標系でマウス位置を取得
            view = self.scene().views()[0]
            self._selection_start = view.mapFromGlobal(event.globalPos())
            
            # Ctrlキーが押されていない場合は既存の選択をクリア
            if not event.modifiers() & QtCore.Qt.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)
            
            # ラバーバンドの開始位置を設定
            self._rubber_band.setGeometry(QtCore.QRect(self._selection_start, QtCore.QSize()))
            self._rubber_band.show()
        
        super(CustomNodeGraph, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_start is not None:
            # ビューの座標系で現在位置を取得
            view = self.scene().views()[0]
            current_pos = view.mapFromGlobal(event.globalPos())
            
            # ラバーバンドの領域を更新
            rect = QtCore.QRect(self._selection_start, current_pos).normalized()
            self._rubber_band.setGeometry(rect)
        
        super(CustomNodeGraph, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._selection_start is not None:
            # ビューの座標系でラバーバンドの領域を取得
            view = self.scene().views()[0]
            rubber_band_rect = self._rubber_band.geometry()
            scene_rect = view.mapToScene(rubber_band_rect).boundingRect()
            
            # 範囲内のノードを選択
            for node in self.all_nodes():
                node_center = QtCore.QPointF(node.pos()[0], node.pos()[1])
                if scene_rect.contains(node_center):
                    node.set_selected(True)
            
            # ラバーバンドをクリア
            self._rubber_band.hide()
            self._selection_start = None
        
        super(CustomNodeGraph, self).mouseReleaseEvent(event)

    def _view_mouse_press(self, event):
        """ビューのマウスプレスイベント"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._selection_start = event.position().toPoint()
            self._is_selecting = True

            # 選択範囲の設定
            if self._rubber_band:
                rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                self._rubber_band.setGeometry(rect)
                self._rubber_band.show()

            # Ctrlキーが押されていない場合は既存の選択をクリア
            if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)

        # 元のイベントハンドラを呼び出し
        if hasattr(self._view, 'mousePressEvent_original'):
            self._view.mousePressEvent_original(event)

    def _view_mouse_move(self, event):
        """ビューのマウス移動イベント"""
        if self._is_selecting and self._selection_start is not None and self._rubber_band:
            current_pos = event.position().toPoint()
            rect = QtCore.QRect(self._selection_start,
                                current_pos).normalized()
            self._rubber_band.setGeometry(rect)

        # 元のイベントハンドラを呼び出し
        if hasattr(self._view, 'mouseMoveEvent_original'):
            self._view.mouseMoveEvent_original(event)

    def _view_mouse_release(self, event):
        """ビューのマウスリリースイベント"""
        if (event.button() == QtCore.Qt.MouseButton.LeftButton and
                self._is_selecting and self._rubber_band):
            try:
                # 選択範囲の取得
                rect = self._rubber_band.geometry()
                scene_rect = self._view.mapToScene(rect).boundingRect()

                # 範囲内のノードを選択
                for node in self.all_nodes():
                    node_pos = node.pos()
                    if isinstance(node_pos, (list, tuple)):
                        node_point = QtCore.QPointF(node_pos[0], node_pos[1])
                    else:
                        node_point = node_pos

                    if scene_rect.contains(node_point):
                        node.set_selected(True)

                # ラバーバンドを隠す
                self._rubber_band.hide()

            except Exception as e:
                print(f"Error in mouse release: {str(e)}")
            finally:
                # 状態をリセット
                self._selection_start = None
                self._is_selecting = False

        # 元のイベントハンドラを呼び出し
        if hasattr(self._view, 'mouseReleaseEvent_original'):
            self._view.mouseReleaseEvent_original(event)

    def create_base_link(self):
        """初期のbase_linkノードを作成"""
        try:
            node_type = f"{BaseLinkNode.__identifier__}.{BaseLinkNode.NODE_NAME}"
            base_node = self.create_node(node_type)
            base_node.set_name('base_link')
            base_node.set_pos(20, 20)
            print("Base Link node created successfully")
            return base_node
        except Exception as e:
            print(f"Error creating base link node: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def register_nodes(self, node_classes):
        """複数のノードクラスを一度に登録"""
        for node_class in node_classes:
            self.register_node(node_class)
            print(f"Registered node type: {node_class.__identifier__}")

    def on_port_connected(self, input_port, output_port):
        """ポートが接続された時の処理"""
        print(f"**Connecting port: {output_port.name()}")
        
        # 接続情報の出力
        parent_node = output_port.node()
        child_node = input_port.node()
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")
        
        try:
            # 全ノードの位置を再計算
            print("Recalculating all node positions after connection...")
            self.recalculate_all_positions()
            
        except Exception as e:
            print(f"Error in port connection: {str(e)}")
            print(f"Detailed connection information:")
            print(f"  Output port: {output_port.name()} from {parent_node.name()}")
            print(f"  Input port: {input_port.name()} from {child_node.name()}")
            traceback.print_exc()

    def on_port_disconnected(self, input_port, output_port):
        """ポートが切断された時の処理"""
        child_node = input_port.node()  # 入力ポートを持つノードが子
        parent_node = output_port.node()  # 出力ポートを持つノードが親
        
        print(f"\nDisconnecting ports:")
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")
        
        try:
            # 子ノードの位置をリセット
            if hasattr(child_node, 'current_transform'):
                del child_node.current_transform
            
            # STLの位置をリセット
            self.stl_viewer.reset_stl_transform(child_node)
            print(f"Reset position for node: {child_node.name()}")

            # 全ノードの位置を再計算
            print("Recalculating all node positions after disconnection...")
            self.recalculate_all_positions()

        except Exception as e:
            print(f"Error in port disconnection: {str(e)}")
            traceback.print_exc()

    def update_robot_name(self, text):
        """ロボット名を更新するメソッド"""
        self.robot_name = text
        print(f"Robot name updated to: {text}")

        # 必要に応じて追加の処理
        # 例：ウィンドウタイトルの更新
        if hasattr(self, 'widget') and self.widget:
            if self.widget.window():
                title = f"URDF Kitchen - Assembler v0.0.1 - {text}"
                self.widget.window().setWindowTitle(title)

    def get_robot_name(self):
        """
        現在のロボット名を取得するメソッド
        Returns:
            str: 現在のロボット名
        """
        return self.robot_name

    def set_robot_name(self, name):
        """
        ロボット名を設定するメソッド
        Args:
            name (str): 設定するロボット名
        """
        self.robot_name = name
        # 入力フィールドが存在する場合は更新
        if hasattr(self, 'name_input') and self.name_input:
            self.name_input.setText(name)
        print(f"Robot name set to: {name}")

    def clean_robot_name(self, name):
        """ロボット名から_descriptionを除去"""
        if name.endswith('_description'):
            return name[:-12]  # '_description'の長さ(12)を削除
        return name

    def update_robot_name_from_directory(self, dir_path):
        """ディレクトリ名からロボット名を更新"""
        dir_name = os.path.basename(dir_path)
        if dir_name.endswith('_description'):
            robot_name = dir_name[:-12]
            # UI更新
            if hasattr(self, 'name_input') and self.name_input:
                self.name_input.setText(robot_name)
            self.robot_name = robot_name
            return True
        return False

    def auto_detect_mesh_directories(self, urdf_file):
        """URDFファイルの親ディレクトリから潜在的なmeshディレクトリを自動検出"""
        potential_dirs = []

        urdf_dir = os.path.dirname(urdf_file)
        parent_dir = os.path.dirname(urdf_dir)

        print(f"\n=== Auto-detecting mesh directories ===")
        print(f"URDF directory: {urdf_dir}")
        print(f"Parent directory: {parent_dir}")

        # 親ディレクトリ内のすべてのサブディレクトリを検索
        try:
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path):
                    # 'mesh' を含むディレクトリ名を検索（大文字小文字を区別しない）
                    if 'mesh' in item.lower():
                        potential_dirs.append(item_path)
                        print(f"  Found potential mesh directory: {item_path}")
        except Exception as e:
            print(f"Error scanning parent directory: {str(e)}")

        print(f"Total potential directories found: {len(potential_dirs)}")
        print("=" * 40 + "\n")

        return potential_dirs

    def search_stl_files_in_directory(self, meshes_dir, missing_stl_files, links_data):
        """指定されたディレクトリ内でSTLファイルを検索"""
        found_count = 0

        print(f"Searching for STL files in: {meshes_dir}")

        for missing_item in missing_stl_files:
            link_name = missing_item['link_name']
            basename = missing_item['basename']

            # すでに見つかっている場合はスキップ
            if links_data[link_name]['stl_file']:
                continue

            # 指定されたディレクトリ内で検索
            candidate_path = os.path.join(meshes_dir, basename)
            if os.path.exists(candidate_path):
                links_data[link_name]['stl_file'] = candidate_path
                print(f"  ✓ Found STL for {link_name}: {candidate_path}")
                found_count += 1
            else:
                # サブディレクトリも検索
                for root_dir, dirs, files in os.walk(meshes_dir):
                    if basename in files:
                        candidate_path = os.path.join(root_dir, basename)
                        links_data[link_name]['stl_file'] = candidate_path
                        print(f"  ✓ Found STL for {link_name} in subdirectory: {candidate_path}")
                        found_count += 1
                        break

        return found_count

    def import_urdf(self):
        """URDFファイルをインポート"""
        try:
            # URDFファイルを選択するダイアログ
            urdf_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.widget,
                "Select URDF file to import",
                os.getcwd(),
                "URDF Files (*.urdf);;All Files (*)"
            )

            if not urdf_file:
                print("URDF import cancelled")
                return False

            print(f"Importing URDF from: {urdf_file}")

            # URDFファイルをパース
            tree = ET.parse(urdf_file)
            root = tree.getroot()

            # ロボット名をURDFファイル名から取得（拡張子なし）
            robot_name = os.path.splitext(os.path.basename(urdf_file))[0]
            self.robot_name = robot_name
            print(f"Robot name set to: {robot_name}")

            # UIのName:フィールドを更新
            if hasattr(self, 'name_input') and self.name_input:
                self.name_input.setText(robot_name)

            # リンクとジョイントの情報を抽出
            links_data = {}
            joints_data = []
            materials_data = {}

            # マテリアル情報を抽出
            for material_elem in root.findall('material'):
                mat_name = material_elem.get('name')
                color_elem = material_elem.find('color')
                if color_elem is not None:
                    rgba_str = color_elem.get('rgba', '1.0 1.0 1.0 1.0')
                    rgba = [float(v) for v in rgba_str.split()]
                    materials_data[mat_name] = rgba[:3]  # RGBのみ

            # リンク情報を抽出
            missing_stl_files = []  # 見つからなかったSTLファイルの情報を保持

            for link_elem in root.findall('link'):
                link_name = link_elem.get('name')
                link_data = {
                    'name': link_name,
                    'mass': 0.0,
                    'inertia': {'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0},
                    'inertial_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                    'stl_file': None,
                    'color': [1.0, 1.0, 1.0],
                    'stl_filename_original': None,  # 元のSTLファイル名を保持
                    'decorations': []  # 2つ目以降のvisualタグ用の装飾情報
                }

                # Inertial情報を抽出
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

                # Visual情報を抽出（STLファイルと色）
                # 複数のvisualタグを処理
                visual_elems = link_elem.findall('visual')
                for visual_idx, visual_elem in enumerate(visual_elems):
                    is_main_visual = (visual_idx == 0)  # 最初のvisualはメイン

                    # STLファイルパスとカラーを一時変数に格納
                    current_stl_path = None
                    current_color = [1.0, 1.0, 1.0]  # デフォルト白

                    geometry_elem = visual_elem.find('geometry')
                    if geometry_elem is not None:
                        mesh_elem = geometry_elem.find('mesh')
                        if mesh_elem is not None:
                            mesh_filename = mesh_elem.get('filename', '')
                            resolved_path = None

                            visual_type = "main" if is_main_visual else "decoration"
                            print(f"Processing {visual_type} mesh for link '{link_name}' (visual {visual_idx}): {mesh_filename}")

                            # メッシュファイル（.stl, .dae等）パスの解決を試みる（複数の方法）
                            urdf_dir = os.path.dirname(urdf_file)
                            description_dir = os.path.dirname(urdf_dir)
                            mesh_basename = os.path.basename(mesh_filename)

                            # package://から始まるパスを処理
                            if mesh_filename.startswith('package://'):
                                # package://robot_name_description/meshes/file.stl → meshes/file.stl
                                parts = mesh_filename.split('/')
                                if len(parts) > 2:
                                    relative_path = '/'.join(parts[2:])

                                    # 方法1: description_dir/相対パス
                                    candidate1 = os.path.join(description_dir, relative_path)
                                    if os.path.exists(candidate1):
                                        resolved_path = candidate1
                                        print(f"Found STL (method 1): {resolved_path}")

                                    # 方法2: urdf_dir/相対パス
                                    if not resolved_path:
                                        candidate2 = os.path.join(urdf_dir, relative_path)
                                        if os.path.exists(candidate2):
                                            resolved_path = candidate2
                                            print(f"Found STL (method 2): {resolved_path}")

                                    # 方法3: description_dir/meshes/basename
                                    if not resolved_path:
                                        candidate3 = os.path.join(description_dir, 'meshes', mesh_basename)
                                        if os.path.exists(candidate3):
                                            resolved_path = candidate3
                                            print(f"Found STL (method 3): {resolved_path}")

                                    # 方法4: urdf_dir/../meshes/basename
                                    if not resolved_path:
                                        candidate4 = os.path.join(urdf_dir, '..', 'meshes', mesh_basename)
                                        candidate4 = os.path.normpath(candidate4)
                                        if os.path.exists(candidate4):
                                            resolved_path = candidate4
                                            print(f"Found mesh (method 4): {resolved_path}")

                                    # 方法5: description_dirの親ディレクトリから検索
                                    if not resolved_path:
                                        parent_dir = os.path.dirname(description_dir)
                                        candidate5 = os.path.join(parent_dir, relative_path)
                                        if os.path.exists(candidate5):
                                            resolved_path = candidate5
                                            print(f"Found STL (method 5): {resolved_path}")

                            else:
                                # 絶対パスまたは相対パスの場合
                                # 方法1: 絶対パスとしてそのまま
                                if os.path.isabs(mesh_filename):
                                    if os.path.exists(mesh_filename):
                                        resolved_path = mesh_filename
                                        print(f"Found STL (absolute path): {resolved_path}")

                                # 方法2: urdf_dirからの相対パス
                                if not resolved_path:
                                    candidate2 = os.path.join(urdf_dir, mesh_filename)
                                    if os.path.exists(candidate2):
                                        resolved_path = candidate2
                                        print(f"Found STL (relative to urdf): {resolved_path}")

                                # 方法3: description_dirからの相対パス
                                if not resolved_path:
                                    candidate3 = os.path.join(description_dir, mesh_filename)
                                    if os.path.exists(candidate3):
                                        resolved_path = candidate3
                                        print(f"Found STL (relative to description): {resolved_path}")

                                # 方法4: meshesディレクトリ内でbasenameを検索
                                if not resolved_path:
                                    meshes_dir = os.path.join(description_dir, 'meshes')
                                    candidate4 = os.path.join(meshes_dir, mesh_basename)
                                    if os.path.exists(candidate4):
                                        resolved_path = candidate4
                                        print(f"Found mesh (meshes dir): {resolved_path}")

                            if resolved_path:
                                # 最終確認: ファイルが本当に存在するか
                                if os.path.exists(resolved_path):
                                    current_stl_path = resolved_path
                                    print(f"  → Successfully resolved: {resolved_path}")
                                else:
                                    print(f"  → ERROR: Path was resolved but file doesn't exist: {resolved_path}")
                                    resolved_path = None

                            if not resolved_path:
                                print(f"  → Could not find STL file for link {link_name}")
                                # メインvisualの場合のみmissing_stl_filesに追加
                                if is_main_visual:
                                    # 元のファイル名を保持して、後で手動検索できるようにする
                                    link_data['stl_filename_original'] = mesh_filename
                                    # すべてのリンクを missing_stl_files に追加
                                    # （base_linkにもSTLファイルがある場合があるため）
                                    missing_stl_files.append({
                                        'link_name': link_name,
                                        'filename': mesh_filename,
                                        'basename': mesh_basename
                                    })
                                    print(f"  → Added to missing list: {mesh_basename}")

                    # カラー情報を取得
                    material_elem = visual_elem.find('material')
                    if material_elem is not None:
                        mat_name = material_elem.get('name')
                        # マテリアルから色を取得
                        if mat_name in materials_data:
                            current_color = materials_data[mat_name]
                        else:
                            # マテリアル要素内にcolor要素がある場合
                            color_elem = material_elem.find('color')
                            if color_elem is not None:
                                rgba_str = color_elem.get('rgba', '1.0 1.0 1.0 1.0')
                                rgba = [float(v) for v in rgba_str.split()]
                                current_color = rgba[:3]
                            elif mat_name and mat_name.startswith('#'):
                                # 16進数カラーコードの場合
                                hex_color = mat_name[1:]
                                if len(hex_color) == 6:
                                    r = int(hex_color[0:2], 16) / 255.0
                                    g = int(hex_color[2:4], 16) / 255.0
                                    b = int(hex_color[4:6], 16) / 255.0
                                    current_color = [r, g, b]

                    # メインvisualかdecorationかで処理を分ける
                    if is_main_visual:
                        # メインvisualの場合は従来通りlink_dataに直接格納
                        if current_stl_path:
                            link_data['stl_file'] = current_stl_path
                        link_data['color'] = current_color
                    else:
                        # decoration visualの場合はdecorationsリストに追加
                        if current_stl_path:
                            # STLファイル名から拡張子を除いた名前を取得
                            stl_name = os.path.splitext(os.path.basename(current_stl_path))[0]
                            decoration_data = {
                                'name': stl_name,
                                'stl_file': current_stl_path,
                                'color': current_color
                            }
                            link_data['decorations'].append(decoration_data)
                            print(f"  → Added decoration: {stl_name}")

                links_data[link_name] = link_data

            # デバッグ: リンクとSTLファイルの状況を出力
            print(f"\n=== STL File Summary ===")
            print(f"Total links: {len(links_data)}")
            for link_name, link_data in links_data.items():
                if link_data['stl_file']:
                    print(f"  ✓ {link_name}: {os.path.basename(link_data['stl_file'])}")
                elif link_data['stl_filename_original']:
                    print(f"  ✗ {link_name}: NOT FOUND ({link_data['stl_filename_original']})")
                else:
                    print(f"  - {link_name}: No STL specified")
            print(f"Missing STL files count: {len(missing_stl_files)}")
            print("=" * 30 + "\n")

            # STLファイルが見つからなかった場合、自動検索してから手動指定
            if missing_stl_files:
                initial_missing_count = len(missing_stl_files)

                # 1. 親ディレクトリ内のmeshフォルダを自動検索
                potential_dirs = self.auto_detect_mesh_directories(urdf_file)

                if potential_dirs:
                    print(f"Trying auto-detected mesh directories...")
                    total_auto_found = 0

                    for mesh_dir in potential_dirs:
                        found_count = self.search_stl_files_in_directory(mesh_dir, missing_stl_files, links_data)
                        total_auto_found += found_count

                    if total_auto_found > 0:
                        print(f"Auto-detection found {total_auto_found} STL file(s)")

                    # missing_stl_filesリストを更新（見つかったものを除去）
                    missing_stl_files = [
                        item for item in missing_stl_files
                        if not links_data[item['link_name']]['stl_file']
                    ]

                # 2. まだ見つからないファイルがあれば、ユーザーに手動指定を促す
                if missing_stl_files:
                    missing_count = len(missing_stl_files)
                    missing_list = '\n'.join([f"  - {item['link_name']}: {item['basename']}" for item in missing_stl_files[:5]])
                    if missing_count > 5:
                        missing_list += f"\n  ... and {missing_count - 5} more"

                    auto_found_count = initial_missing_count - missing_count
                    message = f"Could not find {missing_count} STL file(s):\n\n{missing_list}\n\n"
                    if auto_found_count > 0:
                        message = f"Auto-detected {auto_found_count} file(s), but still missing {missing_count}:\n\n{missing_list}\n\n"

                    response = QtWidgets.QMessageBox.question(
                        self.widget,
                        "STL Files Not Found",
                        message + "Would you like to specify the meshes directory manually?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )

                    if response == QtWidgets.QMessageBox.Yes:
                        # meshesディレクトリを手動で選択
                        meshes_dir = QtWidgets.QFileDialog.getExistingDirectory(
                            self.widget,
                            "Select meshes directory",
                            os.path.dirname(urdf_file)
                        )

                        if meshes_dir:
                            found_count = self.search_stl_files_in_directory(meshes_dir, missing_stl_files, links_data)

                            if found_count > 0:
                                QtWidgets.QMessageBox.information(
                                    self.widget,
                                    "STL Files Found",
                                    f"Found {found_count} out of {missing_count} missing STL file(s) in the specified directory."
                                )
                            else:
                                QtWidgets.QMessageBox.warning(
                                    self.widget,
                                    "No STL Files Found",
                                    f"Could not find any of the missing STL files in the specified directory."
                                )

                            # missing_stl_filesリストを更新（見つかったものを除去）
                            missing_stl_files = [
                                item for item in missing_stl_files
                                if not links_data[item['link_name']]['stl_file']
                            ]
                else:
                    # すべてのSTLファイルが自動検出された
                    if initial_missing_count > 0:
                        QtWidgets.QMessageBox.information(
                            self.widget,
                            "STL Files Found",
                            f"Automatically found all {initial_missing_count} missing STL file(s)!"
                        )

            # ジョイント情報を抽出
            for joint_elem in root.findall('joint'):
                joint_data = {
                    'name': joint_elem.get('name'),
                    'type': joint_elem.get('type', 'fixed'),
                    'parent': None,
                    'child': None,
                    'origin_xyz': [0.0, 0.0, 0.0],
                    'origin_rpy': [0.0, 0.0, 0.0],
                    'axis': [1.0, 0.0, 0.0],
                    'limit': {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 'velocity': 3.0}
                }

                parent_elem = joint_elem.find('parent')
                if parent_elem is not None:
                    joint_data['parent'] = parent_elem.get('link')

                child_elem = joint_elem.find('child')
                if child_elem is not None:
                    joint_data['child'] = child_elem.get('link')

                origin_elem = joint_elem.find('origin')
                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0 0 0')
                    rpy_str = origin_elem.get('rpy', '0 0 0')
                    joint_data['origin_xyz'] = [float(v) for v in xyz_str.split()]
                    joint_data['origin_rpy'] = [float(v) for v in rpy_str.split()]

                axis_elem = joint_elem.find('axis')
                if axis_elem is not None:
                    axis_str = axis_elem.get('xyz', '1 0 0')
                    joint_data['axis'] = [float(v) for v in axis_str.split()]

                limit_elem = joint_elem.find('limit')
                if limit_elem is not None:
                    joint_data['limit']['lower'] = float(limit_elem.get('lower', -3.14159))
                    joint_data['limit']['upper'] = float(limit_elem.get('upper', 3.14159))
                    joint_data['limit']['effort'] = float(limit_elem.get('effort', 10.0))
                    joint_data['limit']['velocity'] = float(limit_elem.get('velocity', 3.0))

                joints_data.append(joint_data)

            # URDFにbase_linkがない場合、ルートリンクを既存のbase_linkに接続
            if 'base_link' not in links_data and 'BaseLink' not in links_data:
                print("\n=== No base_link found in URDF, detecting root links ===")

                # すべてのリンクの中で、どのジョイントの子にもなっていないリンクを見つける
                child_links = set()
                for joint_data in joints_data:
                    if joint_data['child']:
                        child_links.add(joint_data['child'])

                # ルートリンク = links_dataに存在するが、child_linksに存在しないリンク
                root_links = []
                for link_name in links_data.keys():
                    if link_name not in child_links:
                        root_links.append(link_name)
                        print(f"  Found root link: {link_name}")

                # ルートリンクが見つかった場合、base_linkへの接続を作成
                if root_links:
                    print(f"  Connecting {len(root_links)} root link(s) to existing base_link")

                    # 既存のbase_linkノードを取得（グラフ上に存在するはず）
                    # これはlinks_dataには追加せず、接続のためだけに使用

                    for root_link_name in root_links:
                        # base_linkからルートリンクへの合成ジョイントを作成
                        synthetic_joint = {
                            'name': f'base_to_{root_link_name}',
                            'type': 'fixed',
                            'parent': 'base_link',
                            'child': root_link_name,
                            'origin_xyz': [0.0, 0.0, 0.0],
                            'origin_rpy': [0.0, 0.0, 0.0],
                            'axis': [1.0, 0.0, 0.0],
                            'limit': {'lower': 0.0, 'upper': 0.0, 'effort': 0.0, 'velocity': 0.0}
                        }
                        joints_data.append(synthetic_joint)
                        print(f"  Created synthetic joint: base_link -> {root_link_name}")

                print("=" * 50 + "\n")

            # 各リンクに接続する子ジョイントの数を数える（decorationsも含む）
            link_child_counts = {}
            for link_name in links_data.keys():
                link_child_counts[link_name] = 0

            # 子ジョイントの数をカウント
            for joint_data in joints_data:
                parent_link = joint_data['parent']
                # 親リンクがlink_child_countsにない場合は初期化（base_link用）
                if parent_link not in link_child_counts:
                    link_child_counts[parent_link] = 0
                link_child_counts[parent_link] += 1

            # decorationsの数を追加（各decorationも出力ポートが必要）
            for link_name, link_data in links_data.items():
                if link_name in link_child_counts:
                    decoration_count = len(link_data['decorations'])
                    if decoration_count > 0:
                        link_child_counts[link_name] += decoration_count
                        print(f"Link '{link_name}' has {decoration_count} decoration(s)")

            # ノードを作成
            nodes = {}

            # 既存のbase_linkノードを探す（URDFにbase_linkがあるかどうかに関わらず）
            base_node = self.get_node_by_name('base_link')

            # URDFにbase_linkがある場合、またはルートリンクをbase_linkに接続する必要がある場合
            if 'base_link' in links_data or 'base_link' in link_child_counts:
                if base_node:
                    print("Using existing base_link node")
                else:
                    # 既存のbase_linkがない場合は新規作成
                    print("Creating new base_link node")
                    base_node = self.create_node(
                        'insilico.nodes.BaseLinkNode',
                        name='base_link',
                        pos=QtCore.QPointF(0, 0)
                    )

                nodes['base_link'] = base_node

                # base_linkのパラメータを設定（他のリンクと同様に）
                if 'base_link' in links_data:
                    base_link_data = links_data['base_link']
                    base_node.mass_value = base_link_data['mass']
                    base_node.inertia = base_link_data['inertia']
                    base_node.inertial_origin = base_link_data['inertial_origin']
                    base_node.node_color = base_link_data['color']

                    # STLファイルが設定されている場合
                    if base_link_data['stl_file']:
                        base_node.stl_file = base_link_data['stl_file']
                        # STLファイルがある場合はblank_linkをFalseに設定
                        base_node.blank_link = False
                        print(f"Set base_link STL: {base_link_data['stl_file']}")
                        print(f"Set base_link blank_link: False")

                # base_linkの子ジョイント数に応じて出力ポートを追加
                child_count = link_child_counts.get('base_link', 0)
                current_output_count = base_node.output_count if hasattr(base_node, 'output_count') else 1

                # 必要な出力ポート数を計算
                needed_ports = child_count - current_output_count
                if needed_ports > 0:
                    for i in range(needed_ports):
                        base_node._add_output()

            # 他のリンクのノードを作成（グリッドレイアウトで配置）
            grid_spacing = 200  # パネル間の距離を半分に
            current_x = grid_spacing
            current_y = 0
            nodes_per_row = 4
            node_count = 0

            for link_name, link_data in links_data.items():
                if link_name == 'base_link':
                    continue

                # グリッドレイアウトで位置を計算
                row = node_count // nodes_per_row
                col = node_count % nodes_per_row
                pos_x = current_x + col * grid_spacing
                pos_y = current_y + row * grid_spacing

                node = self.create_node(
                    'insilico.nodes.FooNode',
                    name=link_name,
                    pos=QtCore.QPointF(pos_x, pos_y)
                )
                nodes[link_name] = node

                # ノードのパラメータを設定
                node.mass_value = link_data['mass']
                node.inertia = link_data['inertia']
                node.inertial_origin = link_data['inertial_origin']
                node.node_color = link_data['color']
                if link_data['stl_file']:
                    node.stl_file = link_data['stl_file']

                # 子ジョイントの数に応じて出力ポートを追加
                child_count = link_child_counts.get(link_name, 0)
                if child_count > 1:
                    for i in range(1, child_count):
                        node._add_output()

                # decorationノードを作成（このlinkに複数のvisualがある場合）
                decorations = link_data.get('decorations', [])
                for deco_idx, decoration in enumerate(decorations):
                    # decorationノードの位置を親ノードの近くに配置
                    deco_offset_x = 150  # 親の右側に配置
                    deco_offset_y = 100 * (deco_idx + 1)  # 複数ある場合は縦にずらす
                    deco_pos_x = pos_x + deco_offset_x
                    deco_pos_y = pos_y + deco_offset_y

                    deco_node = self.create_node(
                        'insilico.nodes.FooNode',
                        name=decoration['name'],
                        pos=QtCore.QPointF(deco_pos_x, deco_pos_y)
                    )

                    # decorationノードのパラメータを設定
                    deco_node.node_color = decoration['color']
                    deco_node.stl_file = decoration['stl_file']
                    deco_node.massless_decoration = True  # Massless Decorationフラグを設定

                    # 親リンクの参照を保存（接続時に使用）
                    deco_node._parent_link_name = link_name

                    # decorationノードをnodesディクショナリに追加
                    nodes[decoration['name']] = deco_node

                    print(f"  → Created decoration node '{decoration['name']}' for link '{link_name}'")

                node_count += 1

            # ジョイント情報を親ノードのpointsと子ノードのパラメータに反映
            parent_port_indices = {}  # {parent_link: current_port_index}

            for joint_data in joints_data:
                parent_link = joint_data['parent']
                child_link = joint_data['child']

                if parent_link not in nodes or child_link not in nodes:
                    continue

                parent_node = nodes[parent_link]
                child_node = nodes[child_link]

                # 親ノードの現在のポートインデックスを取得
                if parent_link not in parent_port_indices:
                    parent_port_indices[parent_link] = 0
                port_index = parent_port_indices[parent_link]
                parent_port_indices[parent_link] += 1

                # 親ノードのpointsにジョイントのorigin情報を設定
                if port_index < len(parent_node.points):
                    parent_node.points[port_index]['xyz'] = joint_data['origin_xyz']
                    parent_node.points[port_index]['rpy'] = joint_data['origin_rpy']  # RPY情報を追加
                    parent_node.points[port_index]['name'] = joint_data['name']
                    parent_node.points[port_index]['type'] = joint_data['type']

                # 子ノードにジョイント情報を設定
                # 回転軸の設定
                axis = joint_data['axis']
                if joint_data['type'] == 'fixed':
                    child_node.rotation_axis = 3  # Fixed
                elif abs(axis[0]) > 0.9:
                    child_node.rotation_axis = 0  # X軸
                elif abs(axis[1]) > 0.9:
                    child_node.rotation_axis = 1  # Y軸
                else:
                    child_node.rotation_axis = 2  # Z軸

                # ジョイント制限パラメータの設定
                child_node.joint_lower = joint_data['limit']['lower']
                child_node.joint_upper = joint_data['limit']['upper']
                child_node.joint_effort = joint_data['limit']['effort']
                child_node.joint_velocity = joint_data['limit']['velocity']

            # ノードを接続（ジョイントの親子関係に基づく）
            print("\n=== Connecting Nodes ===")
            parent_port_indices = {}  # リセット

            for joint_data in joints_data:
                parent_link = joint_data['parent']
                child_link = joint_data['child']

                if parent_link not in nodes or child_link not in nodes:
                    print(f"Skipping connection: {parent_link} -> {child_link} (node not found)")
                    continue

                parent_node = nodes[parent_link]
                child_node = nodes[child_link]

                # 親ノードの出力ポートインデックスを取得
                if parent_link not in parent_port_indices:
                    parent_port_indices[parent_link] = 0
                port_index = parent_port_indices[parent_link]
                parent_port_indices[parent_link] += 1

                # ポート名を取得
                # isinstance()を使用してノードのクラスを判定
                is_base_link_node = isinstance(parent_node, BaseLinkNode)

                if is_base_link_node:
                    # BaseLinkNodeの場合、最初のポートは'out'、それ以降は'out_2', 'out_3', ...
                    if port_index == 0:
                        output_port_name = 'out'
                    else:
                        output_port_name = f'out_{port_index + 1}'
                else:
                    # FooNodeの場合、'out_1', 'out_2', ... を使用
                    output_port_name = f'out_{port_index + 1}'

                # 子ノードの入力ポート（'in'）を取得
                input_port_name = 'in'

                # デバッグ: 利用可能なポートを表示
                print(f"\nConnecting: {parent_link} -> {child_link}")
                print(f"  Parent node class: {parent_node.__class__.__name__}")
                print(f"  Is BaseLinkNode: {is_base_link_node}")
                print(f"  Port index: {port_index}, Expected output port: {output_port_name}")
                print(f"  Available output ports on {parent_link}: {[p.name() for p in parent_node.output_ports()]}")
                print(f"  Available input ports on {child_link}: {[p.name() for p in child_node.input_ports()]}")

                # ポートを接続
                try:
                    output_port = parent_node.get_output(output_port_name)
                    input_port = child_node.get_input(input_port_name)

                    if output_port and input_port:
                        output_port.connect_to(input_port)
                        print(f"  ✓ Successfully connected {parent_link}.{output_port_name} to {child_link}.{input_port_name}")
                    else:
                        if not output_port:
                            print(f"  ✗ ERROR: Output port '{output_port_name}' not found on {parent_link}")
                        if not input_port:
                            print(f"  ✗ ERROR: Input port '{input_port_name}' not found on {child_link}")
                except Exception as e:
                    print(f"  ✗ ERROR: Exception connecting {parent_link} to {child_link}: {str(e)}")
                    traceback.print_exc()

            # decorationノードを親リンクに接続
            print("\n=== Connecting Decoration Nodes ===")
            for node_name, node in nodes.items():
                # _parent_link_name属性があればdecorationノード
                if hasattr(node, '_parent_link_name'):
                    parent_link = node._parent_link_name
                    if parent_link not in nodes:
                        print(f"Skipping decoration connection: {node_name} -> {parent_link} (parent not found)")
                        continue

                    parent_node = nodes[parent_link]

                    # 親ノードの現在のポートインデックスを取得（joint接続の続きから）
                    if parent_link not in parent_port_indices:
                        parent_port_indices[parent_link] = 0
                    port_index = parent_port_indices[parent_link]
                    parent_port_indices[parent_link] += 1

                    # ポート名を取得
                    is_base_link_node = isinstance(parent_node, BaseLinkNode)

                    if is_base_link_node:
                        # BaseLinkNodeの場合、最初のポートは'out'、それ以降は'out_2', 'out_3', ...
                        if port_index == 0:
                            output_port_name = 'out'
                        else:
                            output_port_name = f'out_{port_index + 1}'
                    else:
                        # FooNodeの場合、'out_1', 'out_2', ... を使用
                        output_port_name = f'out_{port_index + 1}'

                    input_port_name = 'in'

                    print(f"\nConnecting decoration: {parent_link} -> {node_name}")
                    print(f"  Port index: {port_index}, Expected output port: {output_port_name}")

                    # ポートを接続
                    try:
                        output_port = parent_node.get_output(output_port_name)
                        input_port = node.get_input(input_port_name)

                        if output_port and input_port:
                            output_port.connect_to(input_port)
                            print(f"  ✓ Successfully connected {parent_link}.{output_port_name} to {node_name}.{input_port_name}")
                        else:
                            if not output_port:
                                print(f"  ✗ ERROR: Output port '{output_port_name}' not found on {parent_link}")
                            if not input_port:
                                print(f"  ✗ ERROR: Input port '{input_port_name}' not found on {node_name}")
                    except Exception as e:
                        print(f"  ✗ ERROR: Exception connecting {parent_link} to {node_name}: {str(e)}")
                        traceback.print_exc()

            print("=" * 40 + "\n")

            # STLファイルの読み込み状況を確認
            stl_loaded_count = 0
            stl_missing_count = 0
            for link_name, link_data in links_data.items():
                if link_data['stl_file']:
                    stl_loaded_count += 1
                elif link_name != 'base_link':  # base_linkはSTLなしでも問題ない
                    stl_missing_count += 1

            # 全てのノードのSTLファイルを3Dビューに自動読み込み
            print("\n=== Loading STL files to 3D viewer ===")
            stl_viewer_loaded_count = 0
            if self.stl_viewer:
                for link_name, node in nodes.items():
                    # base_linkでblank_linkがTrueの場合はスキップ
                    if link_name == 'base_link':
                        if not hasattr(node, 'blank_link') or node.blank_link:
                            print(f"Skipping base_link (blank_link=True)")
                            continue

                    if hasattr(node, 'stl_file') and node.stl_file:
                        try:
                            print(f"Loading STL to viewer for {link_name}...")
                            self.stl_viewer.load_stl_for_node(node)
                            stl_viewer_loaded_count += 1
                        except Exception as e:
                            print(f"Error loading STL to viewer for {link_name}: {str(e)}")
                            traceback.print_exc()

                print(f"Loaded {stl_viewer_loaded_count} STL files to 3D viewer")
                print("=" * 40 + "\n")
            else:
                print("Warning: STL viewer not available")

            # URDFにbase_linkがない場合、ルートリンクとbase_linkの接続を再確認
            # （Recalc Positionsの直前に実行）
            if 'base_link' not in links_data and 'BaseLink' not in links_data:
                print("\n=== Verifying root link connections to base_link ===")

                # ルートリンクを再検出
                child_links_check = set()
                for joint_data in joints_data:
                    if joint_data['child']:
                        child_links_check.add(joint_data['child'])

                root_links_check = []
                for link_name in links_data.keys():
                    if link_name not in child_links_check:
                        root_links_check.append(link_name)

                # base_linkノードを取得
                if 'base_link' in nodes:
                    base_link_node = nodes['base_link']

                    for root_link_name in root_links_check:
                        if root_link_name in nodes:
                            root_node = nodes[root_link_name]

                            # 既に接続されているか確認
                            already_connected = False
                            for output_port in base_link_node.output_ports():
                                if output_port.connected_ports():
                                    for connected_port in output_port.connected_ports():
                                        if connected_port.node() == root_node:
                                            already_connected = True
                                            break
                                if already_connected:
                                    break

                            if not already_connected:
                                print(f"  Connecting base_link to root link: {root_link_name}")

                                # 最初の利用可能な出力ポートを見つける
                                output_port = None
                                for port in base_link_node.output_ports():
                                    if not port.connected_ports():
                                        output_port = port
                                        break

                                if not output_port and len(base_link_node.output_ports()) > 0:
                                    output_port = base_link_node.output_ports()[0]

                                input_port = root_node.get_input('in')

                                if output_port and input_port:
                                    try:
                                        output_port.connect_to(input_port)
                                        print(f"  ✓ Connected base_link.{output_port.name()} to {root_link_name}.in")
                                    except Exception as e:
                                        print(f"  ✗ Failed to connect: {str(e)}")
                                else:
                                    print(f"  ✗ Could not find ports for connection")
                            else:
                                print(f"  ✓ base_link already connected to {root_link_name}")

                print("=" * 40 + "\n")

            # 全ノードの位置を再計算（Recalc Positionsと同じ処理）
            print("\n=== Recalculating all node positions ===")
            try:
                self.recalculate_all_positions()
                print("Position recalculation completed successfully")
            except Exception as e:
                print(f"Warning: Failed to recalculate positions: {str(e)}")
                traceback.print_exc()
            print("=" * 40 + "\n")

            # 3DビューをFrontビューに設定
            print("=== Setting camera to Front view ===")
            try:
                if self.stl_viewer:
                    self.stl_viewer.reset_camera_front()
                    print("Camera set to Front view successfully")
                else:
                    print("Warning: STL viewer not available")
            except Exception as e:
                print(f"Warning: Failed to set camera view: {str(e)}")
                traceback.print_exc()
            print("=" * 40 + "\n")

            # 完了メッセージ
            import_summary = f"URDF file has been imported:\n{urdf_file}\n\n"
            import_summary += f"Robot name: {robot_name}\n"
            import_summary += f"Links imported: {len(links_data)}\n"
            import_summary += f"Joints imported: {len(joints_data)}\n"
            import_summary += f"Nodes created: {len(nodes)}\n"
            import_summary += f"STL files found: {stl_loaded_count}\n"
            import_summary += f"STL files loaded to 3D viewer: {stl_viewer_loaded_count}\n"
            if stl_missing_count > 0:
                import_summary += f"⚠ Warning: {stl_missing_count} STL file(s) could not be found\n"

            QtWidgets.QMessageBox.information(
                self.widget,
                "Import Complete",
                import_summary
            )

            print("URDF import completed successfully")
            return True

        except Exception as e:
            error_msg = f"Error importing URDF: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            QtWidgets.QMessageBox.critical(
                self.widget,
                "Import Error",
                error_msg
            )
            return False

    def export_urdf(self):
        """URDFファイルをエクスポート"""
        try:
            # # _descriptionを含むディレクトリを探すためのダイアログ
            # description_dir = QtWidgets.QFileDialog.getExistingDirectory(
            #     self.widget,
            #     "Select robot description directory (*_description)",
            #     os.getcwd()
            # )

            # _descriptionを含むディレクトリを探すためのダイアログ
            message_box = QtWidgets.QMessageBox()
            message_box.setIcon(QtWidgets.QMessageBox.Information)
            message_box.setWindowTitle("Select Directory")
            message_box.setText("Please select the *_description directory that will be the root of the URDF.")
            message_box.exec_()

            description_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select robot description directory (*_description)",
                os.getcwd()
            )

            if not description_dir:
                print("URDF export cancelled")
                return False

            # ディレクトリ名が正しいか確認
            dir_name = os.path.basename(description_dir)
            robot_base_name = self.robot_name

            # _descriptionで終わるかチェック
            if dir_name.endswith('_description'):
                # _descriptionを除いた部分を取得
                actual_robot_name = dir_name[:-12]  # '_description'の長さ(12)を削除
                if robot_base_name != actual_robot_name:
                    response = QtWidgets.QMessageBox.question(
                        self.widget,
                        "Robot Name Mismatch",
                        f"Directory suggests robot name '{actual_robot_name}' but current robot name is '{robot_base_name}'.\n"
                        f"Do you want to continue using current name '{robot_base_name}'?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if response == QtWidgets.QMessageBox.No:
                        return False
            else:
                response = QtWidgets.QMessageBox.question(
                    self.widget,
                    "Directory Name Format",
                    f"Selected directory does not end with '_description'.\n"
                    f"Expected format: '*_description'\n"
                    f"Do you want to continue?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if response == QtWidgets.QMessageBox.No:
                    return False

            # ディレクトリ名からロボット名を更新
            if dir_name.endswith('_description'):
                self.update_robot_name_from_directory(description_dir)

            # 現在のロボット名から_descriptionを除去（もし含まれていた場合）
            clean_name = self.clean_robot_name(self.robot_name)

            # urdfディレクトリのパスを作成（meshesと同じ階層）
            urdf_dir = os.path.join(description_dir, 'urdf')

            # urdfディレクトリが存在しない場合は作成
            if not os.path.exists(urdf_dir):
                try:
                    os.makedirs(urdf_dir)
                    print(f"Created URDF directory: {urdf_dir}")
                except Exception as e:
                    print(f"Error creating URDF directory: {str(e)}")
                    return False

            # meshesディレクトリのパスを作成
            meshes_dir = os.path.join(description_dir, 'meshes')

            # meshesディレクトリが存在しない場合は作成
            if not os.path.exists(meshes_dir):
                try:
                    os.makedirs(meshes_dir)
                    print(f"Created meshes directory: {meshes_dir}")
                except Exception as e:
                    print(f"Error creating meshes directory: {str(e)}")
                    return False

            # STLファイルを収集してコピー
            stl_files_copied = []
            stl_files_failed = []

            for node in self.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    source_path = node.stl_file
                    if os.path.exists(source_path):
                        stl_filename = os.path.basename(source_path)
                        dest_path = os.path.join(meshes_dir, stl_filename)

                        try:
                            # STLファイルをコピー
                            shutil.copy2(source_path, dest_path)
                            stl_files_copied.append(stl_filename)
                            print(f"Copied STL: {stl_filename}")
                        except Exception as e:
                            stl_files_failed.append((stl_filename, str(e)))
                            print(f"Failed to copy STL {stl_filename}: {str(e)}")
                    else:
                        stl_files_failed.append((os.path.basename(source_path), "Source file not found"))
                        print(f"STL file not found: {source_path}")

            print(f"\nSTL files copied: {len(stl_files_copied)}")
            if stl_files_failed:
                print(f"STL files failed: {len(stl_files_failed)}")
                for filename, error in stl_files_failed:
                    print(f"  - {filename}: {error}")

            # URDFファイルのパスを設定（クリーンな名前を使用）
            urdf_file = os.path.join(urdf_dir, f"{clean_name}.urdf")

            # 以下、URDFファイルの書き込み処理
            with open(urdf_file, 'w', encoding='utf-8') as f:
                # ヘッダー（クリーンな名前を使用）
                f.write('<?xml version="1.0"?>\n')
                f.write(f'<robot name="{clean_name}">\n\n')

                # マテリアル定義の収集
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
                
                # マテリアルの書き出し
                f.write('<!-- material color setting -->\n')
                for hex_color, rgb in materials.items():
                    f.write(f'<material name="{hex_color}">\n')
                    f.write(f'  <color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"/>\n')
                    f.write('</material>\n')
                f.write('\n')

                # base_linkから開始して、ツリー構造を順番に出力
                visited_nodes = set()
                base_node = self.get_node_by_name('base_link')
                if base_node:
                    self._write_tree_structure(f, base_node, None, visited_nodes, materials)
                
                f.write('</robot>\n')

                print(f"URDF exported to: {urdf_file}")

                # 完了メッセージを作成
                export_summary = f"URDF file has been exported to:\n{urdf_file}\n\n"
                export_summary += f"Meshes directory: {meshes_dir}\n"
                export_summary += f"STL files copied: {len(stl_files_copied)}\n"

                if stl_files_copied:
                    export_summary += "\nCopied files:\n"
                    for filename in stl_files_copied[:10]:  # 最大10個まで表示
                        export_summary += f"  - {filename}\n"
                    if len(stl_files_copied) > 10:
                        export_summary += f"  ... and {len(stl_files_copied) - 10} more\n"

                if stl_files_failed:
                    export_summary += f"\n⚠ Warning: {len(stl_files_failed)} file(s) failed to copy\n"
                    for filename, error in stl_files_failed[:5]:  # 最大5個まで表示
                        export_summary += f"  - {filename}: {error}\n"

                QtWidgets.QMessageBox.information(
                    self.widget,
                    "Export Complete",
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

    def _write_tree_structure(self, file, node, parent_node, visited_nodes, materials):
        """ツリー構造を順番に出力"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless Decorationノードはスキップ（親ノードの<visual>として処理済み）
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        if node.name() == "base_link":
            # base_linkの出力
            self._write_base_link(file)
        
        # 現在のノードに接続されているジョイントとリンクを処理
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                if child_node not in visited_nodes:
                    # Massless Decorationでないノードのみジョイントとリンクを出力
                    if not (hasattr(child_node, 'massless_decoration') and child_node.massless_decoration):
                        # まずジョイントを出力
                        self._write_joint(file, node, child_node)
                        file.write('\n')
                        
                        # 次にリンクを出力
                        self._write_link(file, child_node, materials)
                        file.write('\n')
                    
                    # 再帰的に子ノードを処理
                    self._write_tree_structure(file, child_node, node, visited_nodes, materials)

    def _write_base_link(self, file):
        """base_linkの出力"""
        base_node = self.get_node_by_name('base_link')

        # Blanklinkフラグをチェック
        if base_node and hasattr(base_node, 'blank_link') and not base_node.blank_link:
            # Blanklinkがオフの場合、通常のリンクとして出力（パラメータ付き）
            file.write('  <link name="base_link">\n')

            # 慣性パラメータ
            if hasattr(base_node, 'mass_value') and hasattr(base_node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial Originの出力
                if hasattr(base_node, 'inertial_origin') and isinstance(base_node.inertial_origin, dict):
                    xyz = base_node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    rpy = base_node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                    file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
                else:
                    file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write(f'      <mass value="{base_node.mass_value:.6f}"/>\n')
                file.write('      <inertia')
                for key, value in base_node.inertia.items():
                    file.write(f' {key}="{value:.6f}"')
                file.write('/>\n')
                file.write('    </inertial>\n')

            # ビジュアルとコリジョン
            if hasattr(base_node, 'stl_file') and base_node.stl_file:
                mesh_dir_name = "meshes"
                if self.meshes_dir:
                    dir_name = os.path.basename(self.meshes_dir)
                    if dir_name.startswith('mesh'):
                        mesh_dir_name = dir_name

                stl_filename = os.path.basename(base_node.stl_file)
                package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{stl_filename}"

                # メインのビジュアル
                file.write('    <visual>\n')
                file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write('      <geometry>\n')
                file.write(f'        <mesh filename="{package_path}"/>\n')
                file.write('      </geometry>\n')

                # カラー情報を追加
                if hasattr(base_node, 'node_color') and len(base_node.node_color) >= 3:
                    rgb = base_node.node_color
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(rgb[0] * 255),
                        int(rgb[1] * 255),
                        int(rgb[2] * 255)
                    )
                    file.write(f'      <material name="{hex_color}"/>\n')

                file.write('    </visual>\n')

                # コリジョン
                file.write('    <collision>\n')
                file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write('      <geometry>\n')
                file.write(f'        <mesh filename="{package_path}"/>\n')
                file.write('      </geometry>\n')
                file.write('    </collision>\n')

            file.write('  </link>\n\n')
        else:
            # Blanklinkがオンの場合、パラメータなしのリンクとして出力（デフォルト動作）
            file.write('  <link name="base_link"/>\n\n')

    def _write_urdf_node(self, file, node, parent_node, visited_nodes, materials):
        """再帰的にノードをURDFとして書き出し"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless Decorationフラグのチェック
        is_decoration = hasattr(node, 'massless_decoration') and node.massless_decoration

        if is_decoration:
            # 親ノードに装飾ビジュアルを追加
            if parent_node is not None:
                mesh_dir_name = "meshes"
                if self.meshes_dir:
                    dir_name = os.path.basename(self.meshes_dir)
                    if dir_name.startswith('mesh'):
                        mesh_dir_name = dir_name

                stl_filename = os.path.basename(node.stl_file)
                package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{stl_filename}"

                # 親ノードにビジュアル要素を追加
                file.write(f'''
                <visual>
                    <origin xyz="{node.xyz[0]} {node.xyz[1]} {node.xyz[2]}" rpy="{node.rpy[0]} {node.rpy[1]} {node.rpy[2]}" />
                    <geometry>
                        <mesh filename="{package_path}" />
                    </geometry>
                    <material name="#2e2e2e" />
                </visual>
                ''')
            return  # 装飾要素はこれ以上処理しない

        # 通常ノードの処理
        file.write(f'  <link name="{node.name()}">\n')

        # 慣性パラメータ
        if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
            file.write('    <inertial>\n')
            # Inertial Originの出力
            if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                rpy = node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
            else:
                file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
            file.write(f'      <mass value="{node.mass_value:.6f}"/>\n')
            file.write('      <inertia')
            for key, value in node.inertia.items():
                file.write(f' {key}="{value:.6f}"')
            file.write('/>\n')
            file.write('    </inertial>\n')

        # ビジュアルとコリジョン
        if hasattr(node, 'stl_file') and node.stl_file:
            mesh_dir_name = "meshes"
            if self.meshes_dir:
                dir_name = os.path.basename(self.meshes_dir)
                if dir_name.startswith('mesh'):
                    mesh_dir_name = dir_name

            stl_filename = os.path.basename(node.stl_file)
            package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{stl_filename}"

            # メインのビジュアル
            file.write('    <visual>\n')
            file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
            file.write('      <geometry>\n')
            file.write(f'        <mesh filename="{package_path}"/>\n')
            file.write('      </geometry>\n')
            file.write('    </visual>\n')

            # コリジョン
            file.write('    <collision>\n')
            file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
            file.write('      <geometry>\n')
            file.write(f'        <mesh filename="{package_path}"/>\n')
            file.write('      </geometry>\n')
            file.write('    </collision>\n')

        file.write('  </link>\n\n')

        # ジョイントの書き出し
        if parent_node and not is_decoration:
            origin_xyz = [0, 0, 0]  # デフォルト値
            for port in parent_node.output_ports():
                for connected_port in port.connected_ports():
                    if connected_port.node() == node:
                        origin_xyz = port.get_position()  # ポートの位置を取得
                        break

            joint_name = f"{parent_node.name()}_to_{node.name()}"
            file.write(f'  <joint name="{joint_name}" type="fixed">\n')
            file.write(f'    <origin xyz="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" rpy="0.0 0.0 0.0"/>\n')
            file.write(f'    <parent link="{parent_node.name()}"/>\n')
            file.write(f'    <child link="{node.name()}"/>\n')
            file.write('  </joint>\n\n')

        # 子ノードの処理
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                self._write_urdf_node(file, child_node, node, visited_nodes, materials)

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
        ノードのインスペクタウィンドウを表示
        """
        try:
            # 既存のインスペクタウィンドウをクリーンアップ
            if hasattr(self, 'inspector_window') and self.inspector_window is not None:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                except Exception:
                    pass
                self.inspector_window = None

            # 新しいインスペクタウィンドウを作成
            self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)
            
            # ウィンドウサイズを取得
            inspector_size = self.inspector_window.sizeHint()

            if self.widget and self.widget.window():
                # 保存された位置があればそれを使用し、なければデフォルト位置を計算
                if hasattr(self, 'last_inspector_position') and self.last_inspector_position:
                    x = self.last_inspector_position.x()
                    y = self.last_inspector_position.y()
                    
                    # スクリーンの情報を取得して位置を検証
                    screen = QtWidgets.QApplication.primaryScreen()
                    screen_geo = screen.availableGeometry()
                    
                    # 画面外にはみ出していないか確認
                    if x < screen_geo.x() or x + inspector_size.width() > screen_geo.right() or \
                    y < screen_geo.y() or y + inspector_size.height() > screen_geo.bottom():
                        # 画面外の場合はデフォルト位置を使用
                        main_geo = self.widget.window().geometry()
                        x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                        y = main_geo.y() + 50
                else:
                    # デフォルトの位置を計算
                    main_geo = self.widget.window().geometry()
                    x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                    y = main_geo.y() + 50

                # ウィンドウの初期設定と表示
                self.inspector_window.setWindowTitle(f"Node Inspector - {node.name()}")
                self.inspector_window.current_node = node
                self.inspector_window.graph = self
                self.inspector_window.update_info(node)
                
                self.inspector_window.move(x, y)
                self.inspector_window.show()
                self.inspector_window.raise_()
                self.inspector_window.activateWindow()

                print(f"Inspector window displayed for node: {node.name()}")

        except Exception as e:
            print(f"Error showing inspector: {str(e)}")
            traceback.print_exc()

    def remove_node(self, node):
        self.stl_viewer.remove_stl_for_node(node)
        super(CustomNodeGraph, self).remove_node(node)

    def create_node(self, node_type, name=None, pos=None):
        new_node = super(CustomNodeGraph, self).create_node(node_type, name)

        if pos is None:
            pos = QPointF(0, 0)
        elif isinstance(pos, (tuple, list)):
            pos = QPointF(*pos)

        print(f"Initial position for new node: {pos}")  # デバッグ情報

        adjusted_pos = self.find_non_overlapping_position(pos)
        print(f"Adjusted position for new node: {adjusted_pos}")  # デバッグ情報

        new_node.set_pos(adjusted_pos.x(), adjusted_pos.y())

        # グローバルデフォルト値を新しいノードに適用
        if hasattr(new_node, 'joint_effort'):
            new_node.joint_effort = self.default_joint_effort
        if hasattr(new_node, 'joint_velocity'):
            new_node.joint_velocity = self.default_joint_velocity
        print(f"Applied default values to new node: effort={self.default_joint_effort}, velocity={self.default_joint_velocity}")

        return new_node

    def find_non_overlapping_position(self, pos, offset_x=50, offset_y=30, items_per_row=16):
        all_nodes = self.all_nodes()
        current_node_count = len(all_nodes)
        
        # 現在の行を計算
        row = current_node_count // items_per_row
        
        # 行内での位置を計算
        position_in_row = current_node_count % items_per_row
        
        # 基準となるX座標を計算（各行の開始X座標）
        base_x = pos.x()
        
        # 基準となるY座標を計算
        # 新しい行は前の行の開始位置から200ポイント下
        base_y = pos.y() + (row * 200)
        
        # 現在のノードのX,Y座標を計算
        new_x = base_x + (position_in_row * offset_x)
        new_y = base_y + (position_in_row * offset_y)
        
        new_pos = QPointF(new_x, new_y)
        
        print(f"Positioning node {current_node_count + 1}")
        print(f"Row: {row + 1}, Position in row: {position_in_row + 1}")
        print(f"Position: ({new_pos.x()}, {new_pos.y()})")
        
        # オーバーラップチェックと位置の微調整
        iteration = 0
        while any(self.nodes_overlap(new_pos, node.pos()) for node in all_nodes):
            new_pos += QPointF(5, 5)  # 微小なオフセットで調整
            iteration += 1
            if iteration > 10:
                break
        
        return new_pos

    def nodes_overlap(self, pos1, pos2, threshold=5):
        pos1 = self.ensure_qpointf(pos1)
        pos2 = self.ensure_qpointf(pos2)
        overlap = (abs(pos1.x() - pos2.x()) < threshold and
                abs(pos1.y() - pos2.y()) < threshold)
        # デバッグ出力を条件付きに
        if overlap:
            print(f"Overlap detected: pos1={pos1}, pos2={pos2}")
        return overlap

    def ensure_qpointf(self, pos):
        if isinstance(pos, QPointF):
            return pos
        elif isinstance(pos, (tuple, list)):
            return QPointF(*pos)
        else:
            print(f"Warning: Unsupported position type: {type(pos)}")  # デバッグ情報
            return QPointF(0, 0)  # デフォルト値を返す

    def _save_node_data(self, node, project_dir):
        """ノードデータの保存"""
        print(f"\nStarting _save_node_data for node: {node.name()}")
        node_elem = ET.Element("node")
        
        try:
            # 基本情報
            print(f"  Saving basic info for node: {node.name()}")
            ET.SubElement(node_elem, "id").text = hex(id(node))
            ET.SubElement(node_elem, "name").text = node.name()
            ET.SubElement(node_elem, "type").text = node.__class__.__name__

            # output_count の保存
            if hasattr(node, 'output_count'):
                ET.SubElement(node_elem, "output_count").text = str(node.output_count)
                print(f"  Saved output_count: {node.output_count}")

            # STLファイル情報
            if hasattr(node, 'stl_file') and node.stl_file:
                print(f"  Processing STL file for node {node.name()}: {node.stl_file}")
                stl_elem = ET.SubElement(node_elem, "stl_file")
                
                try:
                    stl_path = os.path.abspath(node.stl_file)
                    print(f"    Absolute STL path: {stl_path}")

                    if self.meshes_dir and stl_path.startswith(self.meshes_dir):
                        rel_path = os.path.relpath(stl_path, self.meshes_dir)
                        stl_elem.set('base_dir', 'meshes')
                        stl_elem.text = os.path.join('meshes', rel_path)
                        print(f"    Using meshes relative path: {rel_path}")
                    else:
                        rel_path = os.path.relpath(stl_path, project_dir)
                        stl_elem.set('base_dir', 'project')
                        stl_elem.text = rel_path
                        print(f"    Using project relative path: {rel_path}")

                except Exception as e:
                    print(f"    Error processing STL file: {str(e)}")
                    stl_elem.set('error', str(e))

            # 位置情報
            pos = node.pos()
            pos_elem = ET.SubElement(node_elem, "position")
            if isinstance(pos, (list, tuple)):
                ET.SubElement(pos_elem, "x").text = str(pos[0])
                ET.SubElement(pos_elem, "y").text = str(pos[1])
            else:
                ET.SubElement(pos_elem, "x").text = str(pos.x())
                ET.SubElement(pos_elem, "y").text = str(pos.y())

            # 物理プロパティ
            if hasattr(node, 'volume_value'):
                ET.SubElement(node_elem, "volume").text = str(node.volume_value)
                print(f"  Saved volume: {node.volume_value}")

            if hasattr(node, 'mass_value'):
                ET.SubElement(node_elem, "mass").text = str(node.mass_value)
                print(f"  Saved mass: {node.mass_value}")

            # 慣性テンソル
            if hasattr(node, 'inertia'):
                inertia_elem = ET.SubElement(node_elem, "inertia")
                for key, value in node.inertia.items():
                    inertia_elem.set(key, str(value))
                print("  Saved inertia tensor")

            # 慣性原点（Inertial Origin / Center of Mass）
            if hasattr(node, 'inertial_origin'):
                inertial_origin_elem = ET.SubElement(node_elem, "inertial_origin")
                xyz_elem = ET.SubElement(inertial_origin_elem, "xyz")
                xyz_elem.text = ' '.join(map(str, node.inertial_origin['xyz']))
                rpy_elem = ET.SubElement(inertial_origin_elem, "rpy")
                rpy_elem.text = ' '.join(map(str, node.inertial_origin['rpy']))
                print(f"  Saved inertial_origin: xyz={node.inertial_origin['xyz']}, rpy={node.inertial_origin['rpy']}")

            # 色情報
            if hasattr(node, 'node_color'):
                color_elem = ET.SubElement(node_elem, "color")
                color_elem.text = ' '.join(map(str, node.node_color))
                print(f"  Saved color: {node.node_color}")

            # 回転軸
            if hasattr(node, 'rotation_axis'):
                ET.SubElement(node_elem, "rotation_axis").text = str(node.rotation_axis)
                print(f"  Saved rotation axis: {node.rotation_axis}")

            # ジョイント制限パラメータ
            if hasattr(node, 'joint_lower'):
                ET.SubElement(node_elem, "joint_lower").text = str(node.joint_lower)
                print(f"  Saved joint_lower: {node.joint_lower}")
            if hasattr(node, 'joint_upper'):
                ET.SubElement(node_elem, "joint_upper").text = str(node.joint_upper)
                print(f"  Saved joint_upper: {node.joint_upper}")
            if hasattr(node, 'joint_effort'):
                ET.SubElement(node_elem, "joint_effort").text = str(node.joint_effort)
                print(f"  Saved joint_effort: {node.joint_effort}")
            if hasattr(node, 'joint_velocity'):
                ET.SubElement(node_elem, "joint_velocity").text = str(node.joint_velocity)
                print(f"  Saved joint_velocity: {node.joint_velocity}")

            # Massless Decoration
            if hasattr(node, 'massless_decoration'):
                ET.SubElement(node_elem, "massless_decoration").text = str(node.massless_decoration)
                print(f"  Saved massless_decoration: {node.massless_decoration}")

            # ポイントデータ
            if hasattr(node, 'points'):
                points_elem = ET.SubElement(node_elem, "points")
                for i, point in enumerate(node.points):
                    point_elem = ET.SubElement(points_elem, "point")
                    point_elem.set('index', str(i))
                    ET.SubElement(point_elem, "name").text = point['name']
                    ET.SubElement(point_elem, "type").text = point['type']
                    ET.SubElement(point_elem, "xyz").text = ' '.join(map(str, point['xyz']))
                print(f"  Saved {len(node.points)} points")

            # 累積座標
            if hasattr(node, 'cumulative_coords'):
                coords_elem = ET.SubElement(node_elem, "cumulative_coords")
                for coord in node.cumulative_coords:
                    coord_elem = ET.SubElement(coords_elem, "coord")
                    ET.SubElement(coord_elem, "point_index").text = str(coord['point_index'])
                    ET.SubElement(coord_elem, "xyz").text = ' '.join(map(str, coord['xyz']))
                print(f"  Saved cumulative coordinates")

            print(f"  Completed saving node data for: {node.name()}")
            return node_elem

        except Exception as e:
            print(f"ERROR in _save_node_data for node {node.name()}: {str(e)}")
            traceback.print_exc()
            raise

    def save_project(self, file_path=None):
        """プロジェクトの保存（循環参照対策版）"""
        print("\n=== Starting Project Save ===")
        try:
            # STLビューアの状態を一時的にバックアップ
            stl_viewer_state = None
            if hasattr(self, 'stl_viewer'):
                print("Backing up STL viewer state...")
                stl_viewer_state = {
                    'actors': dict(self.stl_viewer.stl_actors),
                    'transforms': dict(self.stl_viewer.transforms)
                }
                # STLビューアの参照を一時的にクリア
                self.stl_viewer.stl_actors.clear()
                self.stl_viewer.transforms.clear()

            # ファイルパスの取得
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

            # XMLツリーの作成
            print("Creating XML structure...")
            root = ET.Element("project")
            
            # ロボット名の保存
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

            # ノード情報の保存
            print("\nSaving nodes...")
            nodes_elem = ET.SubElement(root, "nodes")
            total_nodes = len(self.all_nodes())
            
            for i, node in enumerate(self.all_nodes(), 1):
                print(f"Processing node {i}/{total_nodes}: {node.name()}")
                # 一時的にSTLビューアの参照を削除
                stl_viewer_backup = node.stl_viewer if hasattr(node, 'stl_viewer') else None
                if hasattr(node, 'stl_viewer'):
                    delattr(node, 'stl_viewer')
                
                node_elem = self._save_node_data(node, self.project_dir)
                nodes_elem.append(node_elem)
                
                # STLビューアの参照を復元
                if stl_viewer_backup is not None:
                    node.stl_viewer = stl_viewer_backup

            # 接続情報の保存
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

            # ファイルの保存
            print("\nWriting to file...")
            tree = ET.ElementTree(root)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)

            # STLビューアの状態を復元
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
            
            # エラー時もSTLビューアの状態を復元
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
        """STLビューアの状態を復元"""
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
        """meshesディレクトリの検出"""
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
        """プロジェクトの読み込み（コンソール出力版）"""
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
            
            # XMLファイルの解析
            print("Parsing XML file...")
            tree = ET.parse(file_path)
            root = tree.getroot()

            # ロボット名の読み込み
            robot_name_elem = root.find("robot_name")
            if robot_name_elem is not None and robot_name_elem.text:
                self.robot_name = robot_name_elem.text
                # UI上の名前入力フィールドを更新
                if hasattr(self, 'name_input') and self.name_input:
                    self.name_input.setText(self.robot_name)
                print(f"Loaded robot name: {self.robot_name}")
            else:
                print("No robot name found in project file")

            # 既存のノードをクリア
            print("Clearing existing nodes...")
            self.clear_graph()

            # meshesディレクトリの解決
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

            # ノードの復元
            print("\nRestoring nodes...")
            nodes_elem = root.find("nodes")
            total_nodes = len(nodes_elem.findall("node"))
            nodes_dict = {}
            
            for i, node_elem in enumerate(nodes_elem.findall("node"), 1):
                print(f"Processing node {i}/{total_nodes}")
                node = self._load_node_data(node_elem)
                if node:
                    nodes_dict[node.name()] = node
                    print(f"Successfully restored node: {node.name()}")

            # 接続の復元
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
                        print(f"Restored connection: {from_node.name()}.{from_port.name()} -> "
                            f"{to_node.name()}.{to_port.name()}")

            print(f"Total connections restored: {connection_count}")

            # 位置の再計算とビューの更新
            print("\nRecalculating positions...")
            self.recalculate_all_positions()
            
            print("Updating 3D view...")
            if self.stl_viewer:
                QtCore.QTimer.singleShot(500, lambda: self.stl_viewer.reset_view_to_fit())

            print(f"\nProject successfully loaded from: {file_path}")
            return True

        except Exception as e:
            error_msg = f"Error loading project: {str(e)}"
            print(f"\nERROR: {error_msg}")
            print("Traceback:")
            traceback.print_exc()
            
            QtWidgets.QMessageBox.critical(
                None,
                "Load Error",
                error_msg
            )
            return False

    def _load_node_data(self, node_elem):
        """ノードデータの読み込み"""
        try:
            node_type = node_elem.find("type").text
            
            # ノードの作成
            if node_type == "BaseLinkNode":
                node = self.create_base_link()
            else:
                node = self.create_node('insilico.nodes.FooNode')

            # 基本情報の設定
            name_elem = node_elem.find("name")
            if name_elem is not None:
                node.set_name(name_elem.text)
                print(f"Loading node: {name_elem.text}")

            # output_count の復元とポートの追加
            if isinstance(node, FooNode):
                points_elem = node_elem.find("points")
                if points_elem is not None:
                    points = points_elem.findall("point")
                    num_points = len(points)
                    print(f"Found {num_points} points")
                    
                    # 必要な数のポートを追加
                    while len(node.output_ports()) < num_points:
                        node._add_output()
                        print(f"Added output port, total now: {len(node.output_ports())}")

                    # ポイントデータの復元
                    node.points = []
                    for point_elem in points:
                        point_data = {
                            'name': point_elem.find("name").text,
                            'type': point_elem.find("type").text,
                            'xyz': [float(x) for x in point_elem.find("xyz").text.split()]
                        }
                        node.points.append(point_data)
                        print(f"Restored point: {point_data}")

                    # output_countを更新
                    node.output_count = num_points
                    print(f"Set output_count to {num_points}")

            # 位置の設定
            pos_elem = node_elem.find("position")
            if pos_elem is not None:
                x = float(pos_elem.find("x").text)
                y = float(pos_elem.find("y").text)
                node.set_pos(x, y)
                print(f"Set position: ({x}, {y})")

            # 物理プロパティの復元
            volume_elem = node_elem.find("volume")
            if volume_elem is not None:
                node.volume_value = float(volume_elem.text)
                print(f"Restored volume: {node.volume_value}")

            mass_elem = node_elem.find("mass")
            if mass_elem is not None:
                node.mass_value = float(mass_elem.text)
                print(f"Restored mass: {node.mass_value}")

            # 慣性テンソルの復元
            inertia_elem = node_elem.find("inertia")
            if inertia_elem is not None:
                node.inertia = {
                    'ixx': float(inertia_elem.get('ixx', '0.0')),
                    'ixy': float(inertia_elem.get('ixy', '0.0')),
                    'ixz': float(inertia_elem.get('ixz', '0.0')),
                    'iyy': float(inertia_elem.get('iyy', '0.0')),
                    'iyz': float(inertia_elem.get('iyz', '0.0')),
                    'izz': float(inertia_elem.get('izz', '0.0'))
                }
                print(f"Restored inertia tensor")

            # 慣性原点の復元（Inertial Origin / Center of Mass）
            inertial_origin_elem = node_elem.find("inertial_origin")
            if inertial_origin_elem is not None:
                xyz_elem = inertial_origin_elem.find("xyz")
                rpy_elem = inertial_origin_elem.find("rpy")
                if xyz_elem is not None and rpy_elem is not None:
                    node.inertial_origin = {
                        'xyz': [float(x) for x in xyz_elem.text.split()],
                        'rpy': [float(x) for x in rpy_elem.text.split()]
                    }
                    print(f"Restored inertial_origin: xyz={node.inertial_origin['xyz']}, rpy={node.inertial_origin['rpy']}")

            # 色情報の復元
            color_elem = node_elem.find("color")
            if color_elem is not None and color_elem.text:
                node.node_color = [float(x) for x in color_elem.text.split()]
                print(f"Restored color: {node.node_color}")

            # 回転軸の復元
            rotation_axis_elem = node_elem.find("rotation_axis")
            if rotation_axis_elem is not None:
                node.rotation_axis = int(rotation_axis_elem.text)
                print(f"Restored rotation axis: {node.rotation_axis}")

            # ジョイント制限パラメータの復元
            joint_lower_elem = node_elem.find("joint_lower")
            if joint_lower_elem is not None:
                node.joint_lower = float(joint_lower_elem.text)
                print(f"Restored joint_lower: {node.joint_lower}")

            joint_upper_elem = node_elem.find("joint_upper")
            if joint_upper_elem is not None:
                node.joint_upper = float(joint_upper_elem.text)
                print(f"Restored joint_upper: {node.joint_upper}")

            joint_effort_elem = node_elem.find("joint_effort")
            if joint_effort_elem is not None:
                node.joint_effort = float(joint_effort_elem.text)
                print(f"Restored joint_effort: {node.joint_effort}")

            joint_velocity_elem = node_elem.find("joint_velocity")
            if joint_velocity_elem is not None:
                node.joint_velocity = float(joint_velocity_elem.text)
                print(f"Restored joint_velocity: {node.joint_velocity}")

            # Massless Decorationの復元
            massless_dec_elem = node_elem.find("massless_decoration")
            if massless_dec_elem is not None:
                node.massless_decoration = massless_dec_elem.text.lower() == 'true'
                print(f"Restored massless_decoration: {node.massless_decoration}")

            # 累積座標の復元
            coords_elem = node_elem.find("cumulative_coords")
            if coords_elem is not None:
                node.cumulative_coords = []
                for coord_elem in coords_elem.findall("coord"):
                    coord_data = {
                        'point_index': int(coord_elem.find("point_index").text),
                        'xyz': [float(x) for x in coord_elem.find("xyz").text.split()]
                    }
                    node.cumulative_coords.append(coord_data)
                print("Restored cumulative coordinates")

            # STLファイルの設定と処理
            stl_elem = node_elem.find("stl_file")
            if stl_elem is not None and stl_elem.text:
                stl_path = stl_elem.text
                base_dir = stl_elem.get('base_dir', 'project')

                if base_dir == 'meshes' and self.meshes_dir:
                    if stl_path.startswith('meshes/'):
                        stl_path = stl_path[7:]
                    abs_path = os.path.join(self.meshes_dir, stl_path)
                else:
                    abs_path = os.path.join(self.project_dir, stl_path)

                abs_path = os.path.normpath(abs_path)
                if os.path.exists(abs_path):
                    node.stl_file = abs_path
                    if self.stl_viewer:
                        print(f"Loading STL file: {abs_path}")
                        self.stl_viewer.load_stl_for_node(node)
                else:
                    print(f"Warning: STL file not found: {abs_path}")

            print(f"Node {node.name()} loaded successfully")
            return node

        except Exception as e:
            print(f"Error loading node data: {str(e)}")
            traceback.print_exc()
            return None

    def clear_graph(self):
        for node in self.all_nodes():
            self.remove_node(node)

    def connect_ports(self, from_port, to_port):
        """指定された2つのポートを接続"""
        if from_port and to_port:
            try:
                # 利用可能なメソッドを探して接続を試みる
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
        """ノードの累積座標を計算（ルートからのパスを考慮）"""
        if isinstance(node, BaseLinkNode):
            return [0, 0, 0]  # base_linkは原点

        # 親ノードとの接続情報を取得
        input_port = node.input_ports()[0]  # 最初の入力ポート
        if not input_port.connected_ports():
            return [0, 0, 0]  # 接続されていない場合は原点

        parent_port = input_port.connected_ports()[0]
        parent_node = parent_port.node()
        
        # 親ノードの累積座標を再帰的に計算
        parent_coords = self.calculate_cumulative_coordinates(parent_node)
        
        # 接続されているポートのインデックスを取得
        port_name = parent_port.name()
        if '_' in port_name:
            port_index = int(port_name.split('_')[1]) - 1
        else:
            port_index = 0
            
        # 親ノードのポイント座標を取得
        if 0 <= port_index < len(parent_node.points):
            point_xyz = parent_node.points[port_index]['xyz']
            
            # 累積座標の計算
            return [
                parent_coords[0] + point_xyz[0],
                parent_coords[1] + point_xyz[1],
                parent_coords[2] + point_xyz[2]
            ]
        return parent_coords

    def import_xmls_from_folder(self):
        """フォルダ内のすべてのXMLファイルを読み込む"""
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

        # フォルダ名からロボット名を抽出
        try:
            # 2つ上のディレクトリのパスを取得
            parent_dir = os.path.dirname(folder_path)
            robot_name = os.path.basename(parent_dir)

            # _descriptionが末尾にある場合は削除
            if robot_name.endswith('_description'):
                robot_name = robot_name[:-12]
                print(f"Removed '_description' suffix from robot name")

            # ロボット名を更新
            self.robot_name = robot_name
            if hasattr(self, 'name_input') and self.name_input:
                self.name_input.setText(robot_name)
            print(f"Set robot name to: {robot_name}")
        except Exception as e:
            print(f"Error extracting robot name: {str(e)}")
        
        # フォルダ内のXMLファイルを検索
        xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
        
        if not xml_files:
            print("No XML files found in the selected folder")
            return
        
        print(f"Found {len(xml_files)} XML files")

        for xml_file in xml_files:
            try:
                xml_path = os.path.join(folder_path, xml_file)
                stl_path = os.path.join(folder_path, xml_file[:-4] + '.stl')
                
                print(f"\nProcessing: {xml_file}")
                
                # 新しいノードを作成
                new_node = self.create_node(
                    'insilico.nodes.FooNode',
                    name=f'Node_{len(self.all_nodes())}',
                    pos=QtCore.QPointF(0, 0)
                )
                
                # XMLファイルを読み込む
                tree = ET.parse(xml_path)
                root = tree.getroot()

                if root.tag != 'urdf_part':
                    print(f"Warning: Invalid XML format in {xml_file}")
                    continue

                # リンク情報の処理
                link_elem = root.find('link')
                if link_elem is not None:
                    # リンク名の設定
                    link_name = link_elem.get('name')
                    if link_name:
                        new_node.set_name(link_name)
                        print(f"Set link name: {link_name}")

                    # 慣性関連の処理
                    inertial_elem = link_elem.find('inertial')
                    if inertial_elem is not None:
                        # 質量の設定
                        mass_elem = inertial_elem.find('mass')
                        if mass_elem is not None:
                            new_node.mass_value = float(mass_elem.get('value', '0.0'))
                            print(f"Set mass: {new_node.mass_value}")

                        # ボリュームの設定
                        volume_elem = inertial_elem.find('volume')
                        if volume_elem is not None:
                            new_node.volume_value = float(volume_elem.get('value', '0.0'))
                            print(f"Set volume: {new_node.volume_value}")

                        # 慣性テンソルの設定
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
                            print("Set inertia tensor")

                        # 重心位置の設定
                        com_elem = link_elem.find('center_of_mass')
                        if com_elem is not None and com_elem.text:
                            com_values = [float(x) for x in com_elem.text.split()]
                            if len(com_values) == 3:
                                new_node.center_of_mass = com_values
                                print(f"Set center of mass: {com_values}")

                # 色情報の処理
                material_elem = root.find('.//material/color')
                if material_elem is not None:
                    rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                    new_node.node_color = [float(x) for x in rgba[:3]]
                    print(f"Set color: RGB({new_node.node_color})")
                else:
                    new_node.node_color = DEFAULT_COLOR_WHITE.copy()
                    print("Using default color: white")

                # 回転軸の処理
                joint_elem = root.find('.//joint')
                if joint_elem is not None:
                    # ジョイントタイプの確認
                    joint_type = joint_elem.get('type', '')
                    if joint_type == 'fixed':
                        new_node.rotation_axis = 3  # Fixed
                        print("Set rotation axis to Fixed")
                    else:
                        # 回転軸の処理
                        axis_elem = joint_elem.find('axis')
                        if axis_elem is not None:
                            axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                            axis_values = [float(x) for x in axis_xyz]
                            if axis_values[2] == 1:      # Z軸
                                new_node.rotation_axis = 2
                                print("Set rotation axis to Z")
                            elif axis_values[1] == 1:    # Y軸
                                new_node.rotation_axis = 1
                                print("Set rotation axis to Y")
                            else:                        # X軸（デフォルト）
                                new_node.rotation_axis = 0
                                print("Set rotation axis to X")
                else:
                    new_node.rotation_axis = 0
                    print("Using default rotation axis: X")

                # ポイントの処理
                point_elements = root.findall('point')
                
                # 必要な数のポートを追加
                additional_ports_needed = len(point_elements) - 1
                for _ in range(additional_ports_needed):
                    new_node._add_output()

                # ポイントデータの設定
                new_node.points = []
                new_node.cumulative_coords = []
                
                for point_elem in point_elements:
                    point_name = point_elem.get('name')
                    point_type = point_elem.get('type')
                    point_xyz_elem = point_elem.find('point_xyz')
                    
                    if point_xyz_elem is not None and point_xyz_elem.text:
                        xyz_values = [float(x) for x in point_xyz_elem.text.strip().split()]
                        new_node.points.append({
                            'name': point_name,
                            'type': point_type,
                            'xyz': xyz_values
                        })
                        new_node.cumulative_coords.append(
                            create_cumulative_coord(len(new_node.points) - 1)
                        )
                        print(f"Added point: {point_name} at {xyz_values}")

                # STLファイルの処理
                if os.path.exists(stl_path):
                    print(f"Loading corresponding STL file: {stl_path}")
                    new_node.stl_file = stl_path
                    if self.stl_viewer:
                        self.stl_viewer.load_stl_for_node(new_node)
                        # STLモデルに色を適用
                        if hasattr(new_node, 'node_color'):
                            self.stl_viewer.apply_color_to_node(new_node)
                else:
                    print(f"Warning: STL file not found: {stl_path}")

                print(f"Successfully imported: {xml_file}")

            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                traceback.print_exc()
                continue

        print("\nImport process completed")

    def recalculate_all_positions(self):
        """すべてのノードの位置を再計算"""
        print("Starting position recalculation for all nodes...")
        
        try:
            # base_linkノードを探す
            base_node = None
            for node in self.all_nodes():
                if isinstance(node, BaseLinkNode):
                    base_node = node
                    break
            
            if not base_node:
                print("Error: Base link node not found")
                return
            
            # 再帰的に位置を更新
            visited_nodes = set()
            print(f"Starting from base node: {base_node.name()}")
            self._recalculate_node_positions(base_node, [0, 0, 0], visited_nodes)
            
            # STLビューアの更新
            if hasattr(self, 'stl_viewer'):
                self.stl_viewer.render_to_image()
            
            print("Position recalculation completed")

        except Exception as e:
            print(f"Error during position recalculation: {str(e)}")
            traceback.print_exc()

    def _recalculate_node_positions(self, node, parent_coords, visited, parent_transform=None):
        """再帰的にノードの位置と回転を計算"""
        if node in visited:
            return
        visited.add(node)

        print(f"\nProcessing node: {node.name()}")
        print(f"Parent coordinates: {parent_coords}")

        try:
            # 出力ポートを処理
            for port_idx, output_port in enumerate(node.output_ports()):
                for connected_port in output_port.connected_ports():
                    child_node = connected_port.node()

                    # ポイントデータの確認
                    if hasattr(node, 'points') and port_idx < len(node.points):
                        point_data = node.points[port_idx]
                        point_xyz = point_data.get('xyz', [0, 0, 0])
                        point_rpy = point_data.get('rpy', [0, 0, 0])

                        # 新しい位置を計算（簡易版 - 実際には累積変換が必要）
                        new_position = [
                            parent_coords[0] + point_xyz[0],
                            parent_coords[1] + point_xyz[1],
                            parent_coords[2] + point_xyz[2]
                        ]

                        print(f"Child node: {child_node.name()}")
                        print(f"Point XYZ: {point_xyz}, RPY: {point_rpy}")
                        print(f"Calculated position: {new_position}")

                        # 累積変換行列を作成
                        import vtk
                        child_transform = vtk.vtkTransform()
                        if parent_transform is not None:
                            child_transform.DeepCopy(parent_transform)
                        else:
                            child_transform.Identity()

                        # ジョイントの変換を追加
                        child_transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])

                        # 回転を追加（RPY）
                        if point_rpy and len(point_rpy) == 3:
                            import math
                            roll_deg = math.degrees(point_rpy[0])
                            pitch_deg = math.degrees(point_rpy[1])
                            yaw_deg = math.degrees(point_rpy[2])
                            child_transform.RotateZ(yaw_deg)
                            child_transform.RotateY(pitch_deg)
                            child_transform.RotateX(roll_deg)

                        # STL位置と回転を更新
                        self.stl_viewer.update_stl_transform(child_node, point_xyz, point_rpy, parent_transform)

                        # 子ノードの累積座標を更新
                        if hasattr(child_node, 'cumulative_coords'):
                            for coord in child_node.cumulative_coords:
                                coord['xyz'] = new_position.copy()

                        # 再帰的に子ノードを処理（累積変換を渡す）
                        self._recalculate_node_positions(child_node, new_position, visited, child_transform)
                    else:
                        print(f"Warning: No point data found for port {port_idx} in node {node.name()}")

        except Exception as e:
            print(f"Error processing node {node.name()}: {str(e)}")
            traceback.print_exc()

    def disconnect_ports(self, from_port, to_port):
        """ポートの接続を解除"""
        try:
            print(f"Disconnecting ports: {from_port.node().name()}.{from_port.name()} -> {to_port.node().name()}.{to_port.name()}")
            
            # 接続を解除する前に位置情報をリセット
            child_node = to_port.node()
            if child_node:
                self.stl_viewer.reset_stl_transform(child_node)
            
            # 利用可能なメソッドを探して接続解除を試みる
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
                # on_port_disconnectedイベントを呼び出す
                self.on_port_disconnected(to_port, from_port)
                return True
            else:
                print("Failed to disconnect ports")
                return False
                
        except Exception as e:
            print(f"Error disconnecting ports: {str(e)}")
            return False

    def _write_joint(self, file, parent_node, child_node):
        """ジョイントの出力"""
        try:
            # 親ノードのポイント情報から原点を取得
            origin_xyz = [0, 0, 0]  # デフォルト値を設定
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
                        except Exception as e:
                            print(f"Warning: Error processing port {port.name()}: {str(e)}")
                        break

            joint_name = f"{parent_node.name()}_to_{child_node.name()}"
            
            # 回転軸の値に基づいてジョイントタイプを決定
            if hasattr(child_node, 'rotation_axis'):
                if child_node.rotation_axis == 3:  # Fixed
                    file.write(f'  <joint name="{joint_name}" type="fixed">\n')
                    file.write(f'    <origin xyz="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" rpy="0.0 0.0 0.0"/>\n')
                    file.write(f'    <parent link="{parent_node.name()}"/>\n')
                    file.write(f'    <child link="{child_node.name()}"/>\n')
                    file.write('  </joint>\n')
                else:
                    # 回転ジョイントの処理
                    file.write(f'  <joint name="{joint_name}" type="revolute">\n')
                    axis = [0, 0, 0]
                    if child_node.rotation_axis == 0:    # X軸
                        axis = [1, 0, 0]
                    elif child_node.rotation_axis == 1:  # Y軸
                        axis = [0, 1, 0]
                    else:                          # Z軸
                        axis = [0, 0, 1]
                    
                    file.write(f'    <origin xyz="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" rpy="0.0 0.0 0.0"/>\n')
                    file.write(f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n')
                    file.write(f'    <parent link="{parent_node.name()}"/>\n')
                    file.write(f'    <child link="{child_node.name()}"/>\n')

                    # Joint limitパラメータを取得（デフォルト値を使用）
                    lower = getattr(child_node, 'joint_lower', -3.14159)
                    upper = getattr(child_node, 'joint_upper', 3.14159)
                    effort = getattr(child_node, 'joint_effort', 10.0)
                    velocity = getattr(child_node, 'joint_velocity', 3.0)

                    file.write(f'    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>\n')
                    file.write('  </joint>\n')

        except Exception as e:
            print(f"Error writing joint: {str(e)}")
            traceback.print_exc()

    def _write_link(self, file, node, materials):
        """リンクの出力"""
        try:
            file.write(f'  <link name="{node.name()}">\n')

            # 慣性パラメータ
            if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial Originの出力
                if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                    xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    rpy = node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                    file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
                else:
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write(f'      <mass value="{node.mass_value:.6f}"/>\n')
                file.write('      <inertia')
                for key, value in node.inertia.items():
                    file.write(f' {key}="{value:.6f}"')
                file.write('/>\n')
                file.write('    </inertial>\n')

            # メインのビジュアルとコリジョン
            if hasattr(node, 'stl_file') and node.stl_file:
                try:
                    mesh_dir_name = "meshes"
                    if self.meshes_dir:
                        dir_name = os.path.basename(self.meshes_dir)
                        if dir_name.startswith('mesh'):
                            mesh_dir_name = dir_name

                    # メインのビジュアル
                    stl_filename = os.path.basename(node.stl_file)
                    package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{stl_filename}"

                    file.write('    <visual>\n')
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                    file.write('      <geometry>\n')
                    file.write(f'        <mesh filename="{package_path}"/>\n')
                    file.write('      </geometry>\n')
                    if hasattr(node, 'node_color'):
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(node.node_color[0] * 255),
                            int(node.node_color[1] * 255),
                            int(node.node_color[2] * 255)
                        )
                        file.write(f'      <material name="{hex_color}"/>\n')
                    file.write('    </visual>\n')

                    # 装飾パーツのビジュアルを追加
                    for port in node.output_ports():
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if hasattr(dec_node, 'stl_file') and dec_node.stl_file:
                                    dec_stl = os.path.basename(dec_node.stl_file)
                                    dec_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{dec_stl}"
                                    
                                    file.write('    <visual>\n')
                                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
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

                    # コリジョン
                    file.write('    <collision>\n')
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                    file.write('      <geometry>\n')
                    file.write(f'        <mesh filename="{package_path}"/>\n')
                    file.write('      </geometry>\n')
                    file.write('    </collision>\n')

                except Exception as e:
                    print(f"Error processing STL file for node {node.name()}: {str(e)}")
                    traceback.print_exc()

            file.write('  </link>\n')

        except Exception as e:
            print(f"Error writing link: {str(e)}")
            traceback.print_exc()

    def export_for_unity(self):
        """Unityプロジェクト用のファイル構造を作成しエクスポート"""
        try:
            # ディレクトリ選択ダイアログを表示
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

            # ロボット名からディレクトリ名を生成
            robot_name = self.get_robot_name()
            unity_dir_name = f"{robot_name}_unity_description"
            unity_dir_path = os.path.join(base_dir, unity_dir_name)

            # メインディレクトリの作成
            os.makedirs(unity_dir_path, exist_ok=True)
            print(f"Created Unity description directory: {unity_dir_path}")

            # meshesディレクトリの作成
            meshes_dir = os.path.join(unity_dir_path, "meshes")
            os.makedirs(meshes_dir, exist_ok=True)
            print(f"Created meshes directory: {meshes_dir}")

            # STLファイルのコピー
            copied_files = []
            for node in self.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    if os.path.exists(node.stl_file):
                        # ファイル名のみを取得
                        stl_filename = os.path.basename(node.stl_file)
                        # コピー先のパスを生成
                        dest_path = os.path.join(meshes_dir, stl_filename)
                        # ファイルをコピー
                        shutil.copy2(node.stl_file, dest_path)
                        copied_files.append(stl_filename)
                        print(f"Copied STL file: {stl_filename}")

            # URDFファイルの生成
            urdf_file = os.path.join(unity_dir_path, f"{robot_name}.urdf")
            with open(urdf_file, 'w', encoding='utf-8') as f:
                # ヘッダー
                f.write('<?xml version="1.0"?>\n')
                f.write(f'<robot name="{robot_name}">\n\n')

                # マテリアル定義の収集
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

                # マテリアルの書き出し
                f.write('<!-- material color setting -->\n')
                for hex_color, rgb in materials.items():
                    f.write(f'<material name="{hex_color}">\n')
                    f.write(f'  <color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"/>\n')
                    f.write('</material>\n')
                f.write('\n')

                # base_linkから開始して、ツリー構造を順番に出力
                visited_nodes = set()
                base_node = self.get_node_by_name('base_link')
                if base_node:
                    self._write_tree_structure_unity(f, base_node, None, visited_nodes, materials, unity_dir_name)

                f.write('</robot>\n')

            print(f"Unity export completed successfully:")
            print(f"- Directory: {unity_dir_path}")
            print(f"- URDF file: {urdf_file}")
            print(f"- Copied {len(copied_files)} STL files")

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
        """Unity用のツリー構造を順番に出力"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless Decorationノードはスキップ（親ノードの<visual>として処理済み）
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        if node.name() == "base_link":
            # base_linkの出力
            self._write_base_link(file)
        
        # 現在のノードに接続されているジョイントとリンクを処理
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                if child_node not in visited_nodes:
                    # Massless Decorationでないノードのみジョイントとリンクを出力
                    if not (hasattr(child_node, 'massless_decoration') and child_node.massless_decoration):
                        # まずジョイントを出力
                        self._write_joint(file, node, child_node)
                        file.write('\n')
                        
                        # 次にリンクを出力（Unity用のパスで）
                        self._write_link_unity(file, child_node, materials, unity_dir_name)
                        file.write('\n')
                        
                        # 再帰的に子ノードを処理
                        self._write_tree_structure_unity(file, child_node, node, visited_nodes, materials, unity_dir_name)

    def _write_link_unity(self, file, node, materials, unity_dir_name):
        """Unity用のリンク出力"""
        try:
            file.write(f'  <link name="{node.name()}">\n')

            # 慣性パラメータ
            if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial Originの出力
                if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                    xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    rpy = node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                    file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
                else:
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write(f'      <mass value="{node.mass_value:.6f}"/>\n')
                file.write('      <inertia')
                for key, value in node.inertia.items():
                    file.write(f' {key}="{value:.6f}"')
                file.write('/>\n')
                file.write('    </inertial>\n')

            # ビジュアルとコリジョン
            if hasattr(node, 'stl_file') and node.stl_file:
                try:
                    # メインのビジュアル
                    stl_filename = os.path.basename(node.stl_file)
                    # パスは直接meshesを指定
                    package_path = f"package://meshes/{stl_filename}"

                    # メインのビジュアル
                    file.write('    <visual>\n')
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                    file.write('      <geometry>\n')
                    file.write(f'        <mesh filename="{package_path}"/>\n')
                    file.write('      </geometry>\n')
                    if hasattr(node, 'node_color'):
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(node.node_color[0] * 255),
                            int(node.node_color[1] * 255),
                            int(node.node_color[2] * 255)
                        )
                        file.write(f'      <material name="{hex_color}"/>\n')
                    file.write('    </visual>\n')

                    # 装飾パーツのビジュアルを追加
                    for port in node.output_ports():
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if hasattr(dec_node, 'stl_file') and dec_node.stl_file:
                                    dec_stl = os.path.basename(dec_node.stl_file)
                                    dec_path = f"package://meshes/{dec_stl}"
                                    
                                    file.write('    <visual>\n')
                                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
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

                    # コリジョン
                    file.write('    <collision>\n')
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                    file.write('      <geometry>\n')
                    file.write(f'        <mesh filename="{package_path}"/>\n')
                    file.write('      </geometry>\n')
                    file.write('    </collision>\n')

                except Exception as e:
                    print(f"Error processing STL file for node {node.name()}: {str(e)}")
                    traceback.print_exc()

            file.write('  </link>\n')

        except Exception as e:
            print(f"Error writing link: {str(e)}")
            traceback.print_exc()

    def convert_stl_to_obj(self, stl_path, obj_path):
        """STLファイルをOBJ形式に変換（MuJoCo互換性向上）"""
        try:
            # STLファイルを読み込み
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_path)
            reader.Update()

            # OBJ形式で書き出し
            writer = vtk.vtkOBJWriter()
            writer.SetFileName(obj_path)
            writer.SetInputConnection(reader.GetOutputPort())
            writer.Write()

            print(f"Converted STL to OBJ: {os.path.basename(obj_path)}")
            return True

        except Exception as e:
            print(f"Error converting STL to OBJ: {str(e)}")
            traceback.print_exc()
            return False

    def export_mujoco_urdf(self):
        """MuJoCo用のURDFをエクスポート"""
        try:
            # ディレクトリ選択ダイアログを表示
            message_box = QtWidgets.QMessageBox()
            message_box.setIcon(QtWidgets.QMessageBox.Information)
            message_box.setWindowTitle("Select Directory")
            message_box.setText("Please select the directory where you want to create the MuJoCo URDF structure.")
            message_box.exec_()

            base_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select Base Directory for MuJoCo Export"
            )

            if not base_dir:
                print("MuJoCo export cancelled")
                return False

            # ロボット名からディレクトリ名を生成
            robot_name = self.get_robot_name()
            mujoco_dir_name = f"{robot_name}_mujoco_description"
            mujoco_dir_path = os.path.join(base_dir, mujoco_dir_name)

            # メインディレクトリの作成
            os.makedirs(mujoco_dir_path, exist_ok=True)
            print(f"Created MuJoCo description directory: {mujoco_dir_path}")

            # STLファイルをOBJ形式に変換（MuJoCo互換性向上）
            node_to_mesh = {}  # ノードとメッシュファイル名のマッピング
            converted_files = []
            for node in self.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    if os.path.exists(node.stl_file):
                        # ファイル名のみを取得
                        stl_filename = os.path.basename(node.stl_file)
                        obj_filename = stl_filename.replace('.stl', '.obj')

                        # コピー先のパスを生成（URDFと同じディレクトリ）
                        dest_obj_path = os.path.join(mujoco_dir_path, obj_filename)

                        # STLをOBJに変換
                        if self.convert_stl_to_obj(node.stl_file, dest_obj_path):
                            node_to_mesh[node] = obj_filename
                            converted_files.append(obj_filename)
                        else:
                            print(f"Warning: Failed to convert {stl_filename}, skipping...")

            # URDFファイルの生成
            urdf_file = os.path.join(mujoco_dir_path, f"{robot_name}.urdf")
            with open(urdf_file, 'w', encoding='utf-8') as f:
                # ヘッダー
                f.write('<?xml version="1.0"?>\n')
                f.write(f'<robot name="{robot_name}">\n\n')

                # MuJoCo用のコンパイラ設定を追加
                f.write('  <!-- MuJoCo compiler settings -->\n')
                f.write('  <mujoco>\n')
                f.write('    <compiler angle="radian" meshdir="." strippath="true" fusestatic="false" discardvisual="false"/>\n')
                f.write('  </mujoco>\n\n')

                # マテリアル定義の収集
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

                # マテリアルの書き出し
                f.write('  <!-- material color setting -->\n')
                for hex_color, rgb in materials.items():
                    f.write(f'  <material name="{hex_color}">\n')
                    f.write(f'    <color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"/>\n')
                    f.write(f'  </material>\n')
                f.write('\n')

                # base_linkから開始して、ツリー構造を順番に出力
                visited_nodes = set()
                base_node = self.get_node_by_name('base_link')
                if base_node:
                    self._write_tree_structure_mujoco(f, base_node, None, visited_nodes, materials, node_to_mesh)

                f.write('</robot>\n')

            print(f"MuJoCo export completed successfully:")
            print(f"- Directory: {mujoco_dir_path}")
            print(f"- URDF file: {urdf_file}")
            print(f"- Converted {len(converted_files)} mesh files to OBJ format")

            QtWidgets.QMessageBox.information(
                self.widget,
                "MuJoCo Export Complete",
                f"URDF files have been exported for MuJoCo:\n\n"
                f"Directory Path:\n{mujoco_dir_path}\n\n"
                f"URDF File:\n{urdf_file}\n\n"
                f"STL meshes have been converted to OBJ format for better MuJoCo compatibility.\n\n"
                f"The URDF file includes MuJoCo-specific compiler settings:\n"
                f"- angle=\"radian\" (uses radian units)\n"
                f"- meshdir=\".\" (meshes in same directory)\n"
                f"- strippath=\"true\" (removes path from filenames)\n"
                f"- fusestatic=\"false\" (preserves all links)\n"
                f"- discardvisual=\"false\" (preserves visual meshes)"
            )

            return True

        except Exception as e:
            error_msg = f"Error exporting for MuJoCo: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            QtWidgets.QMessageBox.critical(
                self.widget,
                "Export Error",
                error_msg
            )
            return False

    def export_mjcf(self):
        """MuJoCo MJCF形式でエクスポート"""
        try:
            # ロボット名を取得し、制御文字を除去
            import re
            robot_name = self.robot_name or "robot"
            # 制御文字（\x00-\x1F, \x7F-\x9F）を除去
            robot_name = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', robot_name)
            # 空白をアンダースコアに置換
            robot_name = robot_name.replace(' ', '_')
            # 空文字列の場合はデフォルト名を使用
            if not robot_name:
                robot_name = "robot"

            # ファイル保存ダイアログ（保存先を選択）
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.widget,
                "Save MJCF File",
                f"{robot_name}.xml",
                "MJCF Files (*.xml);;All Files (*)"
            )

            if not file_path:
                print("MJCF export cancelled")
                return False

            # 保存先ディレクトリを取得
            save_dir = os.path.dirname(file_path)

            # [Name:] ディレクトリを作成（MuJoCo Menagerie形式）
            mjcf_dir_name = robot_name
            mjcf_dir = os.path.join(save_dir, mjcf_dir_name)
            os.makedirs(mjcf_dir, exist_ok=True)

            # assets ディレクトリを作成
            assets_dir = os.path.join(mjcf_dir, "assets")
            os.makedirs(assets_dir, exist_ok=True)

            # MJCF XMLファイルのパス
            xml_file_path = os.path.join(mjcf_dir, f"{robot_name}.xml")

            print(f"\n=== Exporting MJCF to {xml_file_path} ===")
            print(f"MJCF directory: {mjcf_dir}")
            print(f"Assets directory: {assets_dir}")

            # メッシュファイルをコピーしてマッピングを作成
            node_to_mesh = {}
            import shutil

            for node in self.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file and os.path.exists(node.stl_file):
                    # 元のメッシュファイル名を取得
                    mesh_filename = os.path.basename(node.stl_file)

                    # assetsディレクトリへのコピー先パス
                    dest_mesh_path = os.path.join(assets_dir, mesh_filename)

                    # ファイルをコピー
                    shutil.copy2(node.stl_file, dest_mesh_path)
                    print(f"Copied mesh: {mesh_filename}")

                    # MJCF内で使用する相対パス（assets/filename.stl）
                    rel_mesh_path = os.path.join("assets", mesh_filename)
                    node_to_mesh[node] = rel_mesh_path

            # MJCFファイルを書き込み
            with open(xml_file_path, 'w') as f:
                f.write('<?xml version="1.0" ?>\n')
                f.write(f'<mujoco model="{robot_name}">\n\n')

                # コンパイラ設定（meshdirを"assets"に設定）
                f.write('  <compiler angle="radian" meshdir="assets" />\n\n')

                # オプション設定
                f.write('  <option timestep="0.002" gravity="0 0 -9.81" />\n\n')

                # アセット（メッシュ定義）
                f.write('  <asset>\n')
                mesh_names = {}
                mesh_counter = 0
                for node in self.all_nodes():
                    if node in node_to_mesh:
                        # assets/filename.stl という相対パスを使用
                        mesh_path = node_to_mesh[node]
                        # ファイル名のみ抽出（meshdir="assets"を指定しているため）
                        mesh_filename = os.path.basename(mesh_path)
                        mesh_name = f"mesh_{mesh_counter}"
                        mesh_names[node] = mesh_name
                        f.write(f'    <mesh name="{mesh_name}" file="{mesh_filename}" />\n')
                        mesh_counter += 1
                f.write('  </asset>\n\n')

                # ワールドボディ
                f.write('  <worldbody>\n')

                # 作成されたジョイントのリストを追跡
                created_joints = []

                # base_linkを探す
                base_node = self.get_node_by_name('base_link')
                if base_node:
                    visited_nodes = set()
                    self._write_mjcf_body(f, base_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=2)

                f.write('  </worldbody>\n\n')

                # アクチュエータ（実際に作成されたジョイントのみ）
                f.write('  <actuator>\n')
                for joint_info in created_joints:
                    joint_name = joint_info['joint_name']
                    motor_name = joint_info['motor_name']
                    f.write(f'    <motor name="{motor_name}" joint="{joint_name}" gear="1" />\n')
                f.write('  </actuator>\n\n')

                f.write('</mujoco>\n')

            print(f"MJCF export completed: {xml_file_path}")
            print(f"Total mesh files copied: {len(node_to_mesh)}")

            QtWidgets.QMessageBox.information(
                self.widget,
                "Export Successful",
                f"MJCF file has been exported successfully:\n\n{xml_file_path}\n\nDirectory structure:\n{mjcf_dir_name}/\n  {robot_name}.xml\n  assets/ ({len(node_to_mesh)} files)"
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
        """MuJoCo用に名前をサニタイズ（空白除去、制御文字除去）"""
        import re
        # 制御文字を除去
        name = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', name)
        # 空白をアンダースコアに置換
        name = name.replace(' ', '_')
        return name

    def _write_mjcf_body(self, file, node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=2):
        """MJCF bodyを再帰的に出力"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        indent_str = ' ' * indent

        # base_linkの場合は特別処理
        if node.name() == 'base_link':
            # base_linkの子ノードを処理
            for port in node.output_ports():
                for connected_port in port.connected_ports():
                    child_node = connected_port.node()
                    self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent)
            return

        # 名前をサニタイズ
        sanitized_name = self._sanitize_name(node.name())

        # body開始
        file.write(f'{indent_str}<body name="{sanitized_name}">\n')

        # 慣性プロパティ
        if hasattr(node, 'mass_value') and node.mass_value > 0:
            # 質量の最小閾値を設定（数値的安定性のため）
            MIN_MASS = 0.001  # 1g
            MIN_INERTIA = 1e-6  # 最小慣性モーメント

            mass = max(node.mass_value, MIN_MASS)

            # MuJoCoは慣性を対角成分で指定
            if hasattr(node, 'inertia'):
                # 慣性値を取得し、負の値を絶対値に、最小閾値を適用
                ixx = max(abs(node.inertia.get('ixx', MIN_INERTIA)), MIN_INERTIA)
                iyy = max(abs(node.inertia.get('iyy', MIN_INERTIA)), MIN_INERTIA)
                izz = max(abs(node.inertia.get('izz', MIN_INERTIA)), MIN_INERTIA)
                file.write(f'{indent_str}  <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}" />\n')

        # ジオメトリ（メッシュ）
        if node in mesh_names:
            mesh_name = mesh_names[node]
            # 色情報
            color_str = "0.8 0.8 0.8 1.0"
            if hasattr(node, 'node_color') and node.node_color:
                r, g, b = node.node_color[:3]
                color_str = f"{r} {g} {b} 1.0"
            file.write(f'{indent_str}  <geom type="mesh" mesh="{mesh_name}" rgba="{color_str}" />\n')

        # 子ノードを処理
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()

                # ジョイント情報を取得
                port_index = list(node.output_ports()).index(port)
                joint_xyz = [0, 0, 0]
                joint_rpy = [0, 0, 0]
                joint_axis = [1, 0, 0]

                if hasattr(node, 'points') and port_index < len(node.points):
                    point_data = node.points[port_index]
                    joint_xyz = point_data.get('xyz', [0, 0, 0])
                    joint_rpy = point_data.get('rpy', [0, 0, 0])

                # ジョイント軸を取得
                if hasattr(child_node, 'rotation_axis'):
                    if child_node.rotation_axis == 0:
                        joint_axis = [1, 0, 0]
                    elif child_node.rotation_axis == 1:
                        joint_axis = [0, 1, 0]
                    elif child_node.rotation_axis == 2:
                        joint_axis = [0, 0, 1]

                # 子body開始タグの前にジョイントを定義
                pos_str = f"{joint_xyz[0]} {joint_xyz[1]} {joint_xyz[2]}"

                # ジョイントタイプ
                joint_type = "hinge"
                if hasattr(child_node, 'rotation_axis') and child_node.rotation_axis == 3:
                    joint_type = "fixed"  # MuJoCoではfixedジョイントは省略可能

                if joint_type != "fixed":
                    # 子ノード名をサニタイズしてジョイント名を作成
                    child_sanitized_name = self._sanitize_name(child_node.name())
                    joint_name = f"{child_sanitized_name}_joint"
                    motor_name = f"{child_sanitized_name}_motor"
                    axis_str = f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"

                    # ジョイント制限
                    range_str = ""
                    if hasattr(child_node, 'joint_lower') and hasattr(child_node, 'joint_upper'):
                        range_str = f' range="{child_node.joint_lower} {child_node.joint_upper}"'

                    file.write(f'{indent_str}  <joint name="{joint_name}" type="{joint_type}" pos="{pos_str}" axis="{axis_str}"{range_str} />\n')

                    # 作成されたジョイントをリストに追加
                    created_joints.append({
                        'joint_name': joint_name,
                        'motor_name': motor_name
                    })

                # 再帰的に子ボディを出力
                self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent + 2)

        # body終了
        file.write(f'{indent_str}</body>\n')

    def _write_tree_structure_mujoco(self, file, node, parent_node, visited_nodes, materials, node_to_mesh):
        """MuJoCo用のツリー構造を順番に出力"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless Decorationノードはスキップ（親ノードの<visual>として処理済み）
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        if node.name() == "base_link":
            # base_linkの出力
            self._write_base_link(file)

        # 現在のノードに接続されているジョイントとリンクを処理
        for port in node.output_ports():
            for connected_port in port.connected_ports():
                child_node = connected_port.node()
                if child_node not in visited_nodes:
                    # Massless Decorationでないノードのみジョイントとリンクを出力
                    if not (hasattr(child_node, 'massless_decoration') and child_node.massless_decoration):
                        # まずジョイントを出力
                        self._write_joint(file, node, child_node)
                        file.write('\n')

                        # 次にリンクを出力（MuJoCo用のパスで）
                        self._write_link_mujoco(file, child_node, materials, node_to_mesh)
                        file.write('\n')

                        # 再帰的に子ノードを処理
                        self._write_tree_structure_mujoco(file, child_node, node, visited_nodes, materials, node_to_mesh)

    def _write_link_mujoco(self, file, node, materials, node_to_mesh):
        """MuJoCo用のリンク出力（OBJメッシュを使用）"""
        try:
            file.write(f'  <link name="{node.name()}">\n')

            # 慣性パラメータ
            if hasattr(node, 'mass_value') and hasattr(node, 'inertia'):
                file.write('    <inertial>\n')
                # Inertial Originの出力
                if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                    xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    rpy = node.inertial_origin.get('rpy', [0.0, 0.0, 0.0])
                    file.write(f'      <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n')
                else:
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                file.write(f'      <mass value="{node.mass_value:.6f}"/>\n')
                file.write('      <inertia')
                for key, value in node.inertia.items():
                    file.write(f' {key}="{value:.6f}"')
                file.write('/>\n')
                file.write('    </inertial>\n')

            # ビジュアルとコリジョン
            if node in node_to_mesh:
                try:
                    # メインのビジュアル（OBJファイルを使用）
                    mesh_path = node_to_mesh[node]

                    # メインのビジュアル
                    file.write('    <visual>\n')
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                    file.write('      <geometry>\n')
                    file.write(f'        <mesh filename="{mesh_path}"/>\n')
                    file.write('      </geometry>\n')
                    if hasattr(node, 'node_color'):
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(node.node_color[0] * 255),
                            int(node.node_color[1] * 255),
                            int(node.node_color[2] * 255)
                        )
                        file.write(f'      <material name="{hex_color}"/>\n')
                    file.write('    </visual>\n')

                    # 装飾パーツのビジュアルを追加
                    for port in node.output_ports():
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if dec_node in node_to_mesh:
                                    dec_mesh = node_to_mesh[dec_node]

                                    file.write('    <visual>\n')
                                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                                    file.write('      <geometry>\n')
                                    file.write(f'        <mesh filename="{dec_mesh}"/>\n')
                                    file.write('      </geometry>\n')
                                    if hasattr(dec_node, 'node_color'):
                                        dec_color = '#{:02x}{:02x}{:02x}'.format(
                                            int(dec_node.node_color[0] * 255),
                                            int(dec_node.node_color[1] * 255),
                                            int(dec_node.node_color[2] * 255)
                                        )
                                        file.write(f'      <material name="{dec_color}"/>\n')
                                    file.write('    </visual>\n')

                    # コリジョン
                    file.write('    <collision>\n')
                    file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                    file.write('      <geometry>\n')
                    file.write(f'        <mesh filename="{mesh_path}"/>\n')
                    file.write('      </geometry>\n')
                    file.write('    </collision>\n')

                except Exception as e:
                    print(f"Error processing STL file for node {node.name()}: {str(e)}")
                    traceback.print_exc()

            file.write('  </link>\n')

        except Exception as e:
            print(f"Error writing link: {str(e)}")
            traceback.print_exc()

    def calculate_inertia_tensor_for_mirrored(self, poly_data, mass, center_of_mass):
        """
        ミラーリングされたモデルの慣性テンソルを計算。
        CustomNodeGraphクラスのメソッド。
        """
        try:
            print("\nCalculating inertia tensor for mirrored model...")
            print(f"Mass: {mass:.6f}")
            print(f"Center of Mass (before mirroring): {center_of_mass}")

            # Y座標を反転した重心を使用
            mirrored_com = [center_of_mass[0], -center_of_mass[1], center_of_mass[2]]
            print(f"Center of Mass (after mirroring): {mirrored_com}")

            # インスペクタウィンドウのインスタンスを取得
            if not hasattr(self, 'inspector_window') or not self.inspector_window:
                self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)

            # 慣性テンソルを計算（ミラーリングモードで）
            inertia_tensor = self.inspector_window._calculate_base_inertia_tensor(
                poly_data, mass, mirrored_com, is_mirrored=True)

            print("\nMirrored model inertia tensor calculated successfully")
            return inertia_tensor

        except Exception as e:
            print(f"Error calculating mirrored inertia tensor: {str(e)}")
            traceback.print_exc()
            return None

# ユーティリティ関数
def load_project(graph):
    """プロジェクトを読み込み"""
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

        # XMLファイルを解析
        tree = ET.parse(file_path)
        root = tree.getroot()

        # meshesディレクトリのパスを取得
        meshes_dir = None
        meshes_dir_elem = root.find("meshes_dir")
        if meshes_dir_elem is not None and meshes_dir_elem.text:
            meshes_dir = os.path.normpath(os.path.join(project_base_dir, meshes_dir_elem.text))
            if not os.path.exists(meshes_dir):
                # meshesディレクトリが見つからない場合、ユーザーに選択を求める
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

        # 現在のグラフをクリア
        graph.clear_graph()

        # プロジェクトファイルを読み込み
        success = graph.load_project(file_path)

        if success:
            print("Project loaded, resolving STL paths...")
            # STLファイルのパスを解決
            for node in graph.all_nodes():
                if hasattr(node, 'stl_file') and node.stl_file:
                    try:
                        stl_path = node.stl_file
                        if not os.path.isabs(stl_path):
                            # まずmeshesディレクトリからの相対パスを試す
                            if meshes_dir:
                                abs_stl_path = os.path.normpath(os.path.join(meshes_dir, stl_path))
                                if os.path.exists(abs_stl_path):
                                    node.stl_file = abs_stl_path
                                    print(f"Found STL file in meshes dir for node {node.name()}: {abs_stl_path}")
                                    if graph.stl_viewer:
                                        graph.stl_viewer.load_stl_for_node(node)
                                    continue

                            # プロジェクトディレクトリからの相対パスを試す
                            abs_stl_path = os.path.normpath(os.path.join(project_base_dir, stl_path))
                            if os.path.exists(abs_stl_path):
                                node.stl_file = abs_stl_path
                                print(f"Found STL file in project dir for node {node.name()}: {abs_stl_path}")
                                if graph.stl_viewer:
                                    graph.stl_viewer.load_stl_for_node(node)
                            else:
                                print(f"Warning: Could not find STL file for node {node.name()}: {stl_path}")
                        else:
                            if os.path.exists(stl_path):
                                print(f"Using absolute STL path for node {node.name()}: {stl_path}")
                                if graph.stl_viewer:
                                    graph.stl_viewer.load_stl_for_node(node)
                            else:
                                print(f"Warning: STL file not found: {stl_path}")

                    except Exception as e:
                        print(f"Error resolving STL path for node {node.name()}: {str(e)}")
                        traceback.print_exc()

            print(f"Project loaded successfully from: {file_path}")

            # 位置を再計算
            graph.recalculate_all_positions()

            # 3Dビューをリセット
            if graph.stl_viewer:
                QtCore.QTimer.singleShot(500, lambda: graph.stl_viewer.reset_view_to_fit())

        else:
            print("Failed to load project")

    except Exception as e:
        print(f"Error loading project: {str(e)}")
        traceback.print_exc()

def delete_selected_node(graph):
    selected_nodes = graph.selected_nodes()
    if selected_nodes:
        for node in selected_nodes:
            # BaseLinkNodeは削除不可
            if isinstance(node, BaseLinkNode):
                print("Cannot delete Base Link node")
                continue
            graph.remove_node(node)
        print(f"Deleted {len(selected_nodes)} node(s)")
    else:
        print("No node selected for deletion")
        for connected_port in port.connected_ports():
            child_node = connected_port.node()
            self.print_node_hierarchy(child_node, level + 1)

def show_settings_dialog(graph, parent=None):
    """設定ダイアログを表示"""
    dialog = SettingsDialog(graph, parent)
    dialog.exec_()

def cleanup_and_exit():
    """アプリケーションのクリーンアップと終了"""
    print("Cleaning up application resources...")
    try:
        # グラフのクリーンアップ
        if 'graph' in globals():
            try:
                graph.cleanup()
            except Exception as e:
                print(f"Error cleaning up graph: {str(e)}")

        # STLビューアのクリーンアップ
        if 'stl_viewer' in globals():
            try:
                stl_viewer.cleanup()
            except Exception as e:
                print(f"Error cleaning up STL viewer: {str(e)}")

        # 全てのウィンドウを閉じて削除
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

def signal_handler(signum, frame):
    """Ctrl+Cシグナルのハンドラ"""
    print("\nCtrl+C detected, closing application...")

    def do_shutdown():
        """実際のシャットダウン処理（Qtイベントループ内で実行）"""
        try:
            app = QtWidgets.QApplication.instance()
            if app:
                # 全てのウィンドウを閉じる
                for window in QtWidgets.QApplication.topLevelWidgets():
                    try:
                        if window and window.isVisible():
                            window.close()
                            window.deleteLater()
                    except Exception as e:
                        print(f"Error closing window: {str(e)}")

                # アプリケーションの終了を要求
                QtCore.QTimer.singleShot(100, app.quit)

                # さらに100ms後に強制終了
                QtCore.QTimer.singleShot(200, lambda: sys.exit(0))
        except Exception as e:
            print(f"Error during application shutdown: {str(e)}")
            sys.exit(0)

    # Qtイベントループ内でシャットダウンを実行
    try:
        if QtWidgets.QApplication.instance():
            QtCore.QTimer.singleShot(0, do_shutdown)
        else:
            sys.exit(0)
    except Exception:
        sys.exit(0)

def center_window_top_left(window):
    """ウィンドウを画面の左上に配置"""
    screen = QtWidgets.QApplication.primaryScreen().geometry()
    window.move(0, 0)


if __name__ == '__main__':
    try:
        # Ctrl+Cシグナルハンドラの設定
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app = QtWidgets.QApplication(sys.argv)
        apply_dark_theme(app)

        # アプリケーション終了時のクリーンアップ設定
        app.aboutToQuit.connect(cleanup_and_exit)
        
        timer = QtCore.QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)

        # メインウィンドウの作成
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("URDF Kitchen - Assembler - v0.0.1")
        main_window.resize(1200, 600)

        # セントラルウィジェットの設定
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # STLビューアとグラフの設定（先に作成）
        stl_viewer = STLViewerWidget(central_widget)
        graph = CustomNodeGraph(stl_viewer)
        graph.setup_custom_view()

        # base_linkノードの作成
        base_node = graph.create_base_link()

        # 左パネルの設定
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(145)
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # 名前入力フィールドの設定
        name_label = QtWidgets.QLabel("Name:")
        left_layout.addWidget(name_label)
        name_input = QtWidgets.QLineEdit("robot_x")
        name_input.setFixedWidth(120)
        name_input.setStyleSheet("QLineEdit { padding-left: 3px; padding-top: 0px; padding-bottom: 0px; }")
        left_layout.addWidget(name_input)


        # 名前入力フィールドとグラフを接続（graphが定義された後に接続）
        name_input.textChanged.connect(graph.update_robot_name)




        # ボタンの作成と設定
        buttons = {
            "--spacer1--": None,  # スペーサー用のダミーキー
            "Import XMLs": None,
            "--spacer2--": None,  # スペーサー用のダミーキー
            "Add Node": None,
            "Delete Node": None,
            "Recalc Positions": None,
            "--spacer3--": None,  # スペーサー用のダミーキー
            "Load Project": None,
            "Save Project": None,
            "--spacer4--": None,  # スペーサー用のダミーキー
            "Import URDF": None,
            "Export URDF": None,
            "Export for Unity": None,
            "Exp. MuJoCo URDF": None,
            "Export MJCF": None,
            "--spacer5--": None,  # スペーサー用のダミーキー
            "open urdf-loaders": None,
            "Setting": None
        }

        for button_text in buttons.keys():
            if button_text.startswith("--spacer"):
                # スペーサーの追加
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(1)  # スペースの高さを1ピクセルに設定
                left_layout.addWidget(spacer)
            else:
                # 通常のボタンの追加
                button = QtWidgets.QPushButton(button_text)
                button.setFixedWidth(120)
                left_layout.addWidget(button)
                buttons[button_text] = button

        left_layout.addStretch()

        # ボタンのコネクション設定
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
        buttons["Recalc Positions"].clicked.connect(
            graph.recalculate_all_positions)
        buttons["Save Project"].clicked.connect(graph.save_project)
        buttons["Load Project"].clicked.connect(lambda: load_project(graph))
        buttons["Import URDF"].clicked.connect(lambda: graph.import_urdf())
        buttons["Export URDF"].clicked.connect(lambda: graph.export_urdf())
        buttons["Export for Unity"].clicked.connect(graph.export_for_unity)
        buttons["Exp. MuJoCo URDF"].clicked.connect(graph.export_mujoco_urdf)
        buttons["Exp. MuJoCo URDF"].setStyleSheet("font-size: 9pt;")
        buttons["Export MJCF"].clicked.connect(graph.export_mjcf)
        buttons["open urdf-loaders"].clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl(
                    "https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/")
            )
        )
        buttons["Setting"].clicked.connect(
            lambda: show_settings_dialog(graph, main_window))









        # # ボタンの作成と設定
        # buttons = {
        #     "--spacer1--": None,  # スペーサー用のダミーキー
        #     "Import XMLs": None,
        #     "--spacer2--": None,  # スペーサー用のダミーキー
        #     "Add Node": None,
        #     "Delete Node": None,
        #     "Recalc Positions": None,
        #     "--spacer3--": None,  # スペーサー用のダミーキー
        #     "Load Project": None,
        #     "Save Project": None,
        #     "--spacer4--": None,  # スペーサー用のダミーキー
        #     "Export URDF": None,
        # }


        # for button_text in buttons.keys():
        #     if button_text.startswith("--spacer"):
        #         # スペーサーの追加
        #         spacer = QtWidgets.QWidget()
        #         spacer.setFixedHeight(1)  # スペースの高さを20ピクセルに設定
        #         left_layout.addWidget(spacer)
        #     else:
        #         # 通常のボタンの追加
        #         button = QtWidgets.QPushButton(button_text)
        #         button.setFixedWidth(120)
        #         left_layout.addWidget(button)
        #         buttons[button_text] = button

        # left_layout.addStretch()

        # # ボタンのコネクション設定
        # buttons["Add Node"].clicked.connect(
        #     lambda: graph.create_node(
        #         'insilico.nodes.FooNode',
        #         name=f'Node_{len(graph.all_nodes())}',
        #         pos=QtCore.QPointF(0, 0)
        #     )
        # )
        # buttons["Delete Node"].clicked.connect(
        #     lambda: delete_selected_node(graph))
        # buttons["Import XMLs"].clicked.connect(graph.import_xmls_from_folder)
        # buttons["Export URDF"].clicked.connect(lambda: graph.export_urdf())
        # buttons["Save Project"].clicked.connect(graph.save_project)  # lambdaを使わない直接接続
        # buttons["Load Project"].clicked.connect(lambda: load_project(graph))
        # buttons["Recalc Positions"].clicked.connect(graph.recalculate_all_positions)
        
        # スプリッターの設定（3パネル：左サイドバー、中央グラフ、右3Dビューア）
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(graph.widget)
        splitter.addWidget(stl_viewer)
        # 初期サイズ: 左パネル200px、中央700px、右300px
        splitter.setSizes([200, 700, 300])

        # メインレイアウトの設定
        main_layout.addWidget(splitter)

        # セントラルウィジェットの設定
        main_window.setCentralWidget(central_widget)

        # グラフに名前入力フィールドを関連付け
        graph.name_input = name_input

        # ウィンドウを画面の左上に配置して表示
        center_window_top_left(main_window)
        main_window.show()

        print("Application started. Double-click on a node to open the inspector.")
        print("Click 'Add Node' button to add new nodes.")
        print("Select a node and click 'Delete Node' to remove it.")
        print("Use 'Save' and 'Load' buttons to save and load your project.")
        print("Press Ctrl+C in the terminal to close all windows and exit.")

        # タイマーの設定（シグナル処理のため）
        timer = QtCore.QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)
        
        # アプリケーションの実行
        sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        cleanup_and_exit()
        sys.exit(1)
