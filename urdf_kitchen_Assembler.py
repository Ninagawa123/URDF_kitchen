"""
File Name: urdf_kitchen_Assembler.py
Description: A Python script to assembling files configured with urdf_kitchen_PartsEditor.py into a URDF file.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Update.     : Dec 31, 2025
Version     : 0.0.5
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
from PySide6.QtNetwork import QLocalSocket
import os
import xml.etree.ElementTree as ET
import base64
import shutil
import datetime
import numpy as np
import trimesh
import math

# Import URDF Kitchen utilities for M4 Mac compatibility
from urdf_kitchen_utils import (
    OffscreenRenderer, CameraController, MouseDragState,
    setup_signal_handlers, setup_signal_processing_timer, setup_dark_theme,
    calculate_inertia_tensor, calculate_inertia_tetrahedral,
    get_mesh_file_filter, load_mesh_to_polydata,
    mirror_physical_properties_y_axis, calculate_mirrored_physical_properties_from_mesh
)

# M4 Mac (Apple Silicon) compatibility
import platform
IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'

# デフォルト値の定数定義（Joint LimitsはDegreeで管理）
DEFAULT_JOINT_LOWER = -180.0
DEFAULT_JOINT_UPPER = 180.0
DEFAULT_JOINT_EFFORT = 1.37
DEFAULT_JOINT_VELOCITY = 7.0
DEFAULT_JOINT_FRICTION = 0.05
DEFAULT_JOINT_ACTUATION_LAG = 0.05
DEFAULT_JOINT_DAMPING = 0.18
DEFAULT_JOINT_STIFFNESS = 50.0
DEFAULT_COLOR_WHITE = [1.0, 1.0, 1.0]
DEFAULT_HIGHLIGHT_COLOR = "#80CCFF"  # ライトブルー (0.5, 0.8, 1.0)
DEFAULT_COORDS_ZERO = [0.0, 0.0, 0.0]
DEFAULT_INERTIA_ZERO = {
    'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
    'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
}
DEFAULT_ORIGIN_ZERO = {
    'xyz': [0.0, 0.0, 0.0],
    'rpy': [0.0, 0.0, 0.0]
}

class CustomColorDialog(QtWidgets.QColorDialog):
    """カスタムカラーボックスの選択機能を持つカラーダイアログ"""

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

        print(f"  Array size: {width}x{height}, Cell size: {cell_width}x{cell_height}")
        print(f"  Frame position: ({x}, {y}), size: {frame_width}x{frame_height}")

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

def format_float_no_exp(value, max_decimals=15):
    """
    浮動小数点数を指数表記なしで文字列化する。
    末尾のゼロと不要な小数点も削除する。

    Args:
        value: float - 変換する数値
        max_decimals: int - 最大小数点以下桁数

    Returns:
        str - 指数表記なしの文字列

    Examples:
        >>> format_float_no_exp(0.0000123456789)
        '0.0000123456789'
        >>> format_float_no_exp(1.5)
        '1.5'
        >>> format_float_no_exp(1.0)
        '1'
        >>> format_float_no_exp(0.0)
        '0'
    """
    # 指数表記を避けて文字列化
    formatted = f"{value:.{max_decimals}f}"
    # 末尾のゼロを削除
    formatted = formatted.rstrip('0')
    # 小数点だけが残った場合は削除（例: "1." -> "1"）
    formatted = formatted.rstrip('.')
    # 空文字列の場合は '0' を返す
    return formatted if formatted else '0'

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
    node.collider_mesh = None  # Separate collision mesh file (relative path)
    node.node_color = DEFAULT_COLOR_WHITE.copy()
    node.rotation_axis = 0  # 0: X, 1: Y, 2: Z
    node.joint_lower = math.radians(DEFAULT_JOINT_LOWER)  # DegreeからRadianに変換して保存
    node.joint_upper = math.radians(DEFAULT_JOINT_UPPER)  # DegreeからRadianに変換して保存
    node.joint_effort = DEFAULT_JOINT_EFFORT
    node.joint_velocity = DEFAULT_JOINT_VELOCITY
    node.joint_friction = DEFAULT_JOINT_FRICTION
    node.joint_actuation_lag = DEFAULT_JOINT_ACTUATION_LAG
    node.joint_damping = DEFAULT_JOINT_DAMPING
    node.joint_stiffness = DEFAULT_JOINT_STIFFNESS
    node.massless_decoration = False
    node.hide_mesh = False  # デフォルトはメッシュ表示

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

# apply_dark_theme moved to urdf_kitchen_utils.py
# Now using: setup_dark_theme(app, theme='assembler')

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

            return port_name

    def add_input(self, name='', **kwargs):
        # 入力ポートの追加を禁止
        return None

    def add_output(self, name='out_1', **kwargs):
        # 出力ポートが既に存在する場合は追加しない
        if not self.has_output(name):
            return super(BaseLinkNode, self).add_output(name, **kwargs)
        return None

    def remove_output(self, port=None):
        # 出力ポートの削除を禁止
        return None

    def has_output(self, name):
        """指定した名前の出力ポートが存在するかチェック"""
        return name in [p.name() for p in self.output_ports()]

    def node_double_clicked(self, event):
        """BaseLinkNodeがダブルクリックされたときの処理（Base_linkはインスペクターを開かない）"""
        # Base_linkノードはダブルクリックしてもインスペクターを開かない
        pass

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

        # 初期状態（入力接続なし）は明るめのグレー
        self.set_color(74, 84, 85)

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

                    # 対応するポイントデータを削除
                    if len(self.points) >= self.output_count:
                        self.points.pop()

                    # 累積座標を削除
                    if len(self.cumulative_coords) >= self.output_count:
                        self.cumulative_coords.pop()

                    # ポートの削除
                    self.delete_output(output_port)
                    self.output_count -= 1

                    # ビューの更新
                    self.view.update()
                    
                except Exception as e:
                    traceback.print_exc()
            else:
                pass
        else:
            pass

    def node_double_clicked(self, event):
        if hasattr(self.graph, 'show_inspector'):
            try:
                # グラフのビューを正しく取得
                graph_view = self.graph.viewer()  # NodeGraphQtではviewer()メソッドを使用
                
                # シーン座標をビュー座標に変換
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                
                self.graph.show_inspector(self, screen_pos)
                
            except Exception as e:
                traceback.print_exc()
                # フォールバック：位置指定なしでインスペクタを表示
                self.graph.show_inspector(self)
        else:
            pass

class InspectorWindow(QtWidgets.QWidget):
    
    def __init__(self, parent=None, stl_viewer=None):
        super(InspectorWindow, self).__init__(parent)
        self.setWindowTitle("Node Inspector")
        self.setMinimumWidth(450)
        self.setMinimumHeight(450)
        self.resize(450, 720)  # デフォルトサイズ

        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.current_node = None
        self.stl_viewer = stl_viewer
        self.port_widgets = []
        self.showing_collider = False  # Flag for Visual/Collider mesh display toggle

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

        # File Controls セクション（一番上に配置）
        file_layout = QtWidgets.QHBoxLayout()
        self.load_stl_btn = QtWidgets.QPushButton("Load Mesh")
        self.load_xml_btn = QtWidgets.QPushButton("Load XML")
        self.load_xml_with_stl_btn = QtWidgets.QPushButton("Load XML with Mesh")
        file_layout.addWidget(self.load_stl_btn)
        file_layout.addWidget(self.load_xml_btn)
        file_layout.addWidget(self.load_xml_with_stl_btn)
        self.load_stl_btn.clicked.connect(self.load_stl)
        self.load_xml_btn.clicked.connect(self.load_xml)
        self.load_xml_with_stl_btn.clicked.connect(self.load_xml_with_stl)
        content_layout.addLayout(file_layout)

        # Node Name セクション（横一列）
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Node Name:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Enter node name")
        self.name_edit.editingFinished.connect(self.update_node_name)
        name_layout.addWidget(self.name_edit)

        content_layout.addLayout(name_layout)

        # Massless Decorationチェックボックス と Hide Meshチェックボックス
        massless_layout = QtWidgets.QHBoxLayout()
        self.massless_checkbox = QtWidgets.QCheckBox("Massless Decoration")
        self.massless_checkbox.setChecked(False)  # デフォルトはオフ
        massless_layout.addWidget(self.massless_checkbox)

        # Hide Meshチェックボックス
        self.hide_mesh_checkbox = QtWidgets.QCheckBox("Hide Mesh")
        self.hide_mesh_checkbox.setChecked(False)  # デフォルトはオフ（表示）
        massless_layout.addWidget(self.hide_mesh_checkbox)

        content_layout.addLayout(massless_layout)

        # チェックボックスの状態変更時のハンドラを接続
        self.massless_checkbox.stateChanged.connect(self.update_massless_decoration)
        self.hide_mesh_checkbox.stateChanged.connect(self.update_hide_mesh)

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

        # 0行目: Lower Limit と Upper Limit（Degree表示）
        joint_limits_layout.addWidget(QtWidgets.QLabel("Lower Limit (deg):"), 0, 0)
        self.lower_limit_input = QtWidgets.QLineEdit()
        self.lower_limit_input.setValidator(QDoubleValidator(-360.0, 360.0, 5))
        self.lower_limit_input.setPlaceholderText("-180")
        self.lower_limit_input.returnPressed.connect(self.set_joint_limits)  # リターンキーで即座にリミット値を設定
        joint_limits_layout.addWidget(self.lower_limit_input, 0, 1)

        joint_limits_layout.addWidget(QtWidgets.QLabel("Upper Limit (deg):"), 0, 2)
        self.upper_limit_input = QtWidgets.QLineEdit()
        self.upper_limit_input.setValidator(QDoubleValidator(-360.0, 360.0, 5))
        self.upper_limit_input.setPlaceholderText("180")
        self.upper_limit_input.returnPressed.connect(self.set_joint_limits)  # リターンキーで即座にリミット値を設定
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

        look_zero_button = QtWidgets.QPushButton("Look zero")
        look_zero_button.setFixedWidth(90)
        look_zero_button.clicked.connect(self.look_zero_limit)
        joint_buttons_layout.addWidget(look_zero_button)

        joint_limits_layout.addLayout(joint_buttons_layout, 1, 0, 1, 4)

        # 2行目: Effort, Damping, Stiffness（コンパクト・左寄せ）
        joint_params_row1 = QtWidgets.QHBoxLayout()
        joint_params_row1.setSpacing(5)

        joint_params_row1.addWidget(QtWidgets.QLabel("Effort:"))
        self.effort_input = QtWidgets.QLineEdit()
        self.effort_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.effort_input.setPlaceholderText("1.37")
        self.effort_input.setMaximumWidth(60)
        joint_params_row1.addWidget(self.effort_input)

        joint_params_row1.addWidget(QtWidgets.QLabel("Damping:"))
        self.damping_input = QtWidgets.QLineEdit()
        self.damping_input.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.damping_input.setPlaceholderText("0.18")
        self.damping_input.setMaximumWidth(60)
        joint_params_row1.addWidget(self.damping_input)

        joint_params_row1.addWidget(QtWidgets.QLabel("Stiffness:"))
        self.stiffness_input = QtWidgets.QLineEdit()
        self.stiffness_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.stiffness_input.setPlaceholderText("50")
        self.stiffness_input.setMaximumWidth(60)
        joint_params_row1.addWidget(self.stiffness_input)

        joint_params_row1.addStretch()

        joint_limits_layout.addLayout(joint_params_row1, 2, 0, 1, 6)

        # 3行目: Velocity, ActuationLag（コンパクト・左寄せ）
        joint_params_row2 = QtWidgets.QHBoxLayout()
        joint_params_row2.setSpacing(5)

        joint_params_row2.addWidget(QtWidgets.QLabel("Velocity:"))
        self.velocity_input = QtWidgets.QLineEdit()
        self.velocity_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.velocity_input.setPlaceholderText("7.0")
        self.velocity_input.setMaximumWidth(60)
        joint_params_row2.addWidget(self.velocity_input)

        joint_params_row2.addWidget(QtWidgets.QLabel("Friction:"))
        self.friction_input = QtWidgets.QLineEdit()
        self.friction_input.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.friction_input.setPlaceholderText("0.05")
        self.friction_input.setMaximumWidth(60)
        joint_params_row2.addWidget(self.friction_input)

        joint_params_row2.addWidget(QtWidgets.QLabel("ActuationLag:"))
        self.actuation_lag_input = QtWidgets.QLineEdit()
        self.actuation_lag_input.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.actuation_lag_input.setPlaceholderText("0.05")
        self.actuation_lag_input.setMaximumWidth(60)
        joint_params_row2.addWidget(self.actuation_lag_input)

        joint_params_row2.addStretch()

        joint_limits_layout.addLayout(joint_params_row2, 3, 0, 1, 6)

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

        # Collider Mesh セクション
        collider_layout = QtWidgets.QHBoxLayout()
        collider_layout.addWidget(QtWidgets.QLabel("Collider Mesh:"))

        self.collider_mesh_input = QtWidgets.QLineEdit()
        self.collider_mesh_input.setPlaceholderText("Same as Visual Mesh")
        self.collider_mesh_input.setReadOnly(True)
        collider_layout.addWidget(self.collider_mesh_input)

        attach_button = QtWidgets.QPushButton("Attach")
        attach_button.clicked.connect(self.attach_collider_mesh)
        attach_button.setFixedWidth(60)
        collider_layout.addWidget(attach_button)

        # Toggle button for Visual/Collider display
        self.toggle_collider_button = QtWidgets.QPushButton("Show: Visual")
        self.toggle_collider_button.clicked.connect(self.toggle_collider_display)
        self.toggle_collider_button.setFixedWidth(90)
        self.toggle_collider_button.setEnabled(False)  # Initially disabled
        collider_layout.addWidget(self.toggle_collider_button)

        content_layout.addLayout(collider_layout)

        # 罫線（Output Portsの前）
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator)

        # Output Ports セクション
        ports_layout = QtWidgets.QVBoxLayout()
        self.ports_layout = QtWidgets.QVBoxLayout()  # 動的に追加されるポートのための親レイアウト
        ports_layout.addLayout(self.ports_layout)
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

        # 罫線（File Controlsの前）
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        content_layout.addWidget(separator2)

        # PartsEditor, Save XML, Reload, Set Allボタンレイアウト
        set_button_layout = QtWidgets.QHBoxLayout()
        set_button_layout.addStretch()
        parts_editor_button = QtWidgets.QPushButton("PartsEditor")
        parts_editor_button.clicked.connect(self.open_parts_editor)
        set_button_layout.addWidget(parts_editor_button)
        save_xml_button = QtWidgets.QPushButton("Save XML")
        save_xml_button.clicked.connect(self.save_xml)
        set_button_layout.addWidget(save_xml_button)
        reload_button = QtWidgets.QPushButton("Reload")
        reload_button.clicked.connect(self.reload_node_files)
        set_button_layout.addWidget(reload_button)
        set_button = QtWidgets.QPushButton("Set All")
        set_button.clicked.connect(self.apply_port_values)
        set_button_layout.addWidget(set_button)
        content_layout.addLayout(set_button_layout)

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


        except Exception as e:
            import traceback
            traceback.print_exc()

    def apply_color_to_stl(self):
        """選択された色をSTLモデルとカラーサンプルに適用"""
        if not self.current_node:
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
                else:
                    pass

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

    def attach_collider_mesh(self):
        """Attach a separate collision mesh file"""
        if not self.current_node:
            return

        # Get the directory of the visual mesh
        visual_mesh = getattr(self.current_node, 'stl_file', None)
        if visual_mesh and os.path.exists(visual_mesh):
            start_dir = os.path.dirname(visual_mesh)
        else:
            start_dir = ""

        # Open file dialog with mesh filter
        file_filter = "Mesh Files (*.stl *.dae *.obj);;STL Files (*.stl);;DAE Files (*.dae);;OBJ Files (*.obj)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Collision Mesh",
            start_dir,
            file_filter
        )

        if file_path:
            # Store as relative path from visual mesh directory
            if visual_mesh:
                visual_dir = os.path.dirname(visual_mesh)
                try:
                    relative_path = os.path.relpath(file_path, visual_dir)
                    self.current_node.collider_mesh = relative_path
                    self.collider_mesh_input.setText(relative_path)
                    print(f"Attached collider mesh: {relative_path}")

                    # Enable toggle button
                    self.toggle_collider_button.setEnabled(True)
                except ValueError:
                    # Paths on different drives on Windows
                    self.current_node.collider_mesh = file_path
                    self.collider_mesh_input.setText(file_path)
                    print(f"Attached collider mesh (absolute): {file_path}")
                    self.toggle_collider_button.setEnabled(True)
            else:
                # No visual mesh, store absolute path
                self.current_node.collider_mesh = file_path
                self.collider_mesh_input.setText(file_path)
                print(f"Attached collider mesh (absolute): {file_path}")
                self.toggle_collider_button.setEnabled(True)

    def toggle_collider_display(self):
        """Toggle between Visual and Collider mesh display"""
        if not self.current_node or not hasattr(self.stl_viewer, 'load_stl_for_node'):
            return

        collider_mesh = getattr(self.current_node, 'collider_mesh', None)
        if not collider_mesh:
            return

        self.showing_collider = not self.showing_collider

        if self.showing_collider:
            # Show collider mesh
            self.toggle_collider_button.setText("Show: Collider")
            # Temporarily swap mesh files
            visual_mesh = self.current_node.stl_file
            if visual_mesh:
                visual_dir = os.path.dirname(visual_mesh)
                collider_absolute = os.path.join(visual_dir, collider_mesh)
                if os.path.exists(collider_absolute):
                    # Temporarily set stl_file to collider
                    original_stl = self.current_node.stl_file
                    self.current_node.stl_file = collider_absolute
                    self.stl_viewer.load_stl_for_node(self.current_node)
                    self.current_node.stl_file = original_stl  # Restore original
                    print(f"Displaying collider mesh: {collider_mesh}")
                else:
                    print(f"Warning: Collider mesh not found: {collider_absolute}")
        else:
            # Show visual mesh
            self.toggle_collider_button.setText("Show: Visual")
            self.stl_viewer.load_stl_for_node(self.current_node)
            print(f"Displaying visual mesh")

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
        """Inertial OriginのUI入力フィールドに値を設定（指数表記を使用せず、高精度で表示）"""
        self.inertial_x_input.setText(format_float_no_exp(xyz[0]))
        self.inertial_y_input.setText(format_float_no_exp(xyz[1]))
        self.inertial_z_input.setText(format_float_no_exp(xyz[2]))
        self.inertial_r_input.setText(format_float_no_exp(rpy[0]))
        self.inertial_p_input.setText(format_float_no_exp(rpy[1]))
        self.inertial_y_rpy_input.setText(format_float_no_exp(rpy[2]))

    def _set_inertia_ui(self, inertia_dict):
        """Inertia TensorのUI入力フィールドに値を設定（指数表記を使用せず、高精度で表示）"""
        self.ixx_input.setText(format_float_no_exp(inertia_dict.get('ixx', 0.0)))
        self.ixy_input.setText(format_float_no_exp(inertia_dict.get('ixy', 0.0)))
        self.ixz_input.setText(format_float_no_exp(inertia_dict.get('ixz', 0.0)))
        self.iyy_input.setText(format_float_no_exp(inertia_dict.get('iyy', 0.0)))
        self.iyz_input.setText(format_float_no_exp(inertia_dict.get('iyz', 0.0)))
        self.izz_input.setText(format_float_no_exp(inertia_dict.get('izz', 0.0)))

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

            # Volume & Mass（高精度、指数表記なし）
            if hasattr(node, 'volume_value'):
                self.volume_input.setText(format_float_no_exp(node.volume_value))

            if hasattr(node, 'mass_value'):
                self.mass_input.setText(format_float_no_exp(node.mass_value))

            # Inertia の設定
            if hasattr(node, 'inertia') and isinstance(node.inertia, dict):
                self._set_inertia_ui(node.inertia)
            else:
                # デフォルト値を設定
                node.inertia = DEFAULT_INERTIA_ZERO.copy()
                self._set_inertia_ui(node.inertia)

            # Inertial Origin の設定
            if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                xyz = node.inertial_origin.get('xyz', DEFAULT_COORDS_ZERO)
                rpy = node.inertial_origin.get('rpy', DEFAULT_COORDS_ZERO)
                self._set_inertial_origin_ui(xyz, rpy)
            else:
                # デフォルト値を設定
                node.inertial_origin = DEFAULT_ORIGIN_ZERO.copy()
                node.inertial_origin['xyz'] = DEFAULT_COORDS_ZERO.copy()
                node.inertial_origin['rpy'] = DEFAULT_COORDS_ZERO.copy()
                self._set_inertial_origin_ui(node.inertial_origin['xyz'], node.inertial_origin['rpy'])

            # Rotation Axis - nodeのrotation_axis属性を確認して設定
            if hasattr(node, 'rotation_axis'):
                axis_button = self.axis_group.button(node.rotation_axis)
                if axis_button:
                    axis_button.setChecked(True)
            else:
                # デフォルトでX軸を選択
                node.rotation_axis = 0
                if self.axis_group.button(0):
                    self.axis_group.button(0).setChecked(True)

            # Massless Decoration の状態を設定
            if hasattr(node, 'massless_decoration'):
                self.massless_checkbox.setChecked(node.massless_decoration)
            else:
                node.massless_decoration = False
                self.massless_checkbox.setChecked(False)

            # Hide Mesh の状態を設定
            if hasattr(node, 'hide_mesh'):
                self.hide_mesh_checkbox.setChecked(node.hide_mesh)
            else:
                node.hide_mesh = False
                self.hide_mesh_checkbox.setChecked(False)

            # Blanklink の状態を設定（BaseLinkNodeの場合のみ）
            if isinstance(node, BaseLinkNode):
                self.blanklink_checkbox.setVisible(True)
                if hasattr(node, 'blank_link'):
                    self.blanklink_checkbox.setChecked(node.blank_link)
                else:
                    node.blank_link = True
                    self.blanklink_checkbox.setChecked(True)
            else:
                # BaseLinkNode以外の場合はBlanklinkチェックボックスを非表示
                self.blanklink_checkbox.setVisible(False)

            # Joint Limits の設定（RadianからDegreeに変換して表示）
            if hasattr(node, 'joint_lower'):
                # ノードにはRadian値で保存されているのでDegreeに変換（小数点2桁まで、3桁目を四捨五入）
                self.lower_limit_input.setText(str(round(math.degrees(node.joint_lower), 2)))
            else:
                # DEFAULT_JOINT_LOWERは既にDegree値
                node.joint_lower = math.radians(DEFAULT_JOINT_LOWER)
                self.lower_limit_input.setText(str(DEFAULT_JOINT_LOWER))

            if hasattr(node, 'joint_upper'):
                # ノードにはRadian値で保存されているのでDegreeに変換（小数点2桁まで、3桁目を四捨五入）
                self.upper_limit_input.setText(str(round(math.degrees(node.joint_upper), 2)))
            else:
                # DEFAULT_JOINT_UPPERは既にDegree値
                node.joint_upper = math.radians(DEFAULT_JOINT_UPPER)
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

            if hasattr(node, 'joint_friction'):
                self.friction_input.setText(str(node.joint_friction))
            else:
                # グラフのデフォルト値を使用
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_friction'):
                    node.joint_friction = node.graph.default_joint_friction
                else:
                    node.joint_friction = DEFAULT_JOINT_FRICTION
                self.friction_input.setText(str(node.joint_friction))

            if hasattr(node, 'joint_actuation_lag'):
                self.actuation_lag_input.setText(str(node.joint_actuation_lag))
            else:
                # グラフのデフォルト値を使用
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_actuation_lag'):
                    node.joint_actuation_lag = node.graph.default_joint_actuation_lag
                else:
                    node.joint_actuation_lag = DEFAULT_JOINT_ACTUATION_LAG
                self.actuation_lag_input.setText(str(node.joint_actuation_lag))

            # Dampingの設定
            if hasattr(node, 'joint_damping'):
                self.damping_input.setText(str(node.joint_damping))
            else:
                # グラフのデフォルト値を使用
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_damping'):
                    node.joint_damping = node.graph.default_joint_damping
                else:
                    node.joint_damping = DEFAULT_JOINT_DAMPING
                self.damping_input.setText(str(node.joint_damping))

            # Stiffnessの設定
            if hasattr(node, 'joint_stiffness'):
                self.stiffness_input.setText(str(node.joint_stiffness))
            else:
                # グラフのデフォルト値を使用
                if hasattr(node, 'graph') and hasattr(node.graph, 'default_joint_stiffness'):
                    node.joint_stiffness = node.graph.default_joint_stiffness
                else:
                    node.joint_stiffness = DEFAULT_JOINT_STIFFNESS
                self.stiffness_input.setText(str(node.joint_stiffness))

            # Color settings - nodeのnode_color属性を確認して設定
            if hasattr(node, 'node_color') and node.node_color:
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

            # Collider Mesh settings
            self.showing_collider = False  # Reset display toggle to Visual when switching nodes
            if hasattr(node, 'collider_mesh') and node.collider_mesh:
                self.collider_mesh_input.setText(node.collider_mesh)
                self.toggle_collider_button.setEnabled(True)
            else:
                node.collider_mesh = None
                self.collider_mesh_input.clear()  # Show placeholder "Same as Visual Mesh"
                self.toggle_collider_button.setEnabled(False)
            self.toggle_collider_button.setText("Show: Visual")

            # 回転軸の選択を更新するためのシグナルを接続
            for button in self.axis_group.buttons():
                button.clicked.connect(lambda checked, btn=button: self.update_rotation_axis(btn))

            # Output Ports
            self.update_output_ports(node)

            # ラジオボタンのイベントハンドラを設定
            self.axis_group.buttonClicked.connect(self.on_axis_selection_changed)

            # バリデータの設定
            self.setup_validators()


        except Exception as e:
            print(f"Error updating inspector info: {str(e)}")
            traceback.print_exc()

    def update_rotation_axis(self, button):
        """回転軸の選択が変更されたときの処理"""
        if self.current_node:
            self.current_node.rotation_axis = self.axis_group.id(button)

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
                pass
            else:
                pass

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
                        
    def show_color_picker(self):
        """カラーピッカーを表示"""
        try:
            current_color = QtGui.QColor(
                *[min(255, max(0, int(float(input.text()) * 255)))
                for input in self.color_inputs]
            )
        except ValueError:
            current_color = QtGui.QColor(255, 255, 255)

        # カスタムカラーダイアログを使用
        dialog = CustomColorDialog(current_color, self)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            color = dialog.currentColor()
        else:
            color = QtGui.QColor()  # Invalid color

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
            

    def update_node_name(self):
        """ノード名の更新"""
        if self.current_node:
            new_name = self.name_edit.text()
            old_name = self.current_node.name()
            if new_name != old_name:
                self.current_node.set_name(new_name)

    def add_point(self):
        """ポイントの追加"""
        if self.current_node and hasattr(self.current_node, '_add_output'):
            new_port_name = self.current_node._add_output()
            if new_port_name:
                self.update_info(self.current_node)

    def remove_point(self):
        """ポイントの削除"""
        if self.current_node and hasattr(self.current_node, 'remove_output'):
            self.current_node.remove_output()
            self.update_info(self.current_node)

    def load_stl(self):
        """STLファイルの読み込み"""
        if self.current_node:
            file_filter = get_mesh_file_filter(trimesh_available=True)
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Mesh File", "", file_filter)
            if file_name:
                self.current_node.stl_file = file_name
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                    # 3Dビューを更新
                    self.stl_viewer.render_to_image()

                # Recalc Positionsと同じ効果を実行
                if hasattr(self.current_node, 'graph') and self.current_node.graph:
                    self.current_node.graph.recalculate_all_positions()

    def closeEvent(self, event):
        """ウィンドウが閉じられるときのイベントを処理"""
        try:
            # ハイライトをクリア
            if self.stl_viewer:
                self.stl_viewer.clear_highlight()

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


            # リンク名の取得と設定
            link_elem = root.find('link')
            if link_elem is not None:
                link_name = link_elem.get('name')
                if link_name:
                    self.current_node.set_name(link_name)
                    self.name_edit.setText(link_name)

                # 物理プロパティの設定
                inertial_elem = link_elem.find('inertial')
                if inertial_elem is not None:
                    # ボリュームの設定
                    volume_elem = inertial_elem.find('volume')
                    if volume_elem is not None:
                        volume = float(volume_elem.get('value', '0.0'))
                        self.current_node.volume_value = volume
                        self.volume_input.setText(format_float_no_exp(volume))

                    # 質量の設定
                    mass_elem = inertial_elem.find('mass')
                    if mass_elem is not None:
                        mass = float(mass_elem.get('value', '0.0'))
                        self.current_node.mass_value = mass
                        self.mass_input.setText(format_float_no_exp(mass))

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

                # Center of Massの設定
                center_of_mass_elem = link_elem.find('center_of_mass')
                if center_of_mass_elem is not None:
                    com_xyz = center_of_mass_elem.text.strip().split()

            # メッシュファイルの処理
            mesh_elem = root.find('.//visual/geometry/mesh')
            if mesh_elem is not None:
                mesh_filename = mesh_elem.get('filename')
                if mesh_filename:
                    # XMLファイルと同じディレクトリにあると仮定
                    xml_dir = os.path.dirname(file_name)
                    mesh_path = os.path.join(xml_dir, mesh_filename)
                    if os.path.exists(mesh_path):
                        self.current_node.stl_file = mesh_path

            # 色情報の処理
            material_elem = root.find('.//material/color')
            if material_elem is not None:
                rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                rgb_values = [float(x) for x in rgba[:3]]  # RGBのみ使用
                self.current_node.node_color = rgb_values
                # 色の設定をUIに反映
                self._set_color_ui(rgb_values)
                self.update_color_sample()

            # STLファイルが設定されている場合、ロードして色を適用
            if hasattr(self.current_node, 'stl_file') and self.current_node.stl_file:
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                    # 色は load_stl_for_node() 内で自動的に適用される

            # Collision mesh の処理
            collision_mesh_elem = link_elem.find('collision_mesh') if link_elem is not None else None
            if collision_mesh_elem is not None and collision_mesh_elem.text:
                self.current_node.collider_mesh = collision_mesh_elem.text.strip()
                print(f"Loaded collider mesh: {self.current_node.collider_mesh}")
            else:
                self.current_node.collider_mesh = None

            # 回転軸の処理
            joint_elem = root.find('joint')
            if joint_elem is not None:
                # jointのtype属性を確認
                joint_type = joint_elem.get('type', '')
                if joint_type == 'fixed':
                    self.current_node.rotation_axis = 3  # 3をFixedとして使用
                    if self.axis_group.button(3):  # Fixed用のボタンが存在する場合
                        self.axis_group.button(3).setChecked(True)
                else:
                    # 回転軸の処理
                    axis_elem = joint_elem.find('axis')
                    if axis_elem is not None:
                        axis_xyz = axis_elem.get('xyz', '1 0 0').split()
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

                # Joint limitsの処理
                limit_elem = joint_elem.find('limit')
                if limit_elem is not None:
                    # XMLからはRadian値で読み込む
                    lower_rad = float(limit_elem.get('lower', -3.14159))
                    upper_rad = float(limit_elem.get('upper', 3.14159))
                    effort = float(limit_elem.get('effort', 10.0))
                    velocity = float(limit_elem.get('velocity', 3.0))
                    friction = float(limit_elem.get('friction', 0.05))
                    actuation_lag = float(limit_elem.get('actuationLag', 0.0))

                    # ノードにはRadian値で保存
                    self.current_node.joint_lower = lower_rad
                    self.current_node.joint_upper = upper_rad
                    self.current_node.joint_effort = effort
                    self.current_node.joint_velocity = velocity
                    self.current_node.joint_friction = friction
                    self.current_node.joint_actuation_lag = actuation_lag

                    # UI表示はDegreeに変換
                    self.lower_limit_input.setText(str(round(math.degrees(lower_rad), 2)))
                    self.upper_limit_input.setText(str(round(math.degrees(upper_rad), 2)))
                    self.effort_input.setText(format_float_no_exp(effort))
                    self.velocity_input.setText(format_float_no_exp(velocity))
                    self.friction_input.setText(format_float_no_exp(friction))
                    self.actuation_lag_input.setText(format_float_no_exp(actuation_lag))


            # ポイントの処理
            points = root.findall('point')
            num_points = len(points)

            # 現在のポート数と必要なポート数を比較
            current_ports = len(self.current_node.output_ports())

            # ポート数を調整
            if isinstance(self.current_node, FooNode):
                # ポートを削除する前に、削除対象のポートの接続をすべてクリア
                if current_ports > num_points:
                    for i in range(num_points + 1, current_ports + 1):
                        port_name = f'out_{i}'
                        port = self.current_node.get_output(port_name)
                        if port:
                            port.clear_connections()

                while current_ports < num_points:
                    self.current_node._add_output()
                    current_ports += 1

                while current_ports > num_points:
                    self.current_node.remove_output()
                    current_ports -= 1

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

                # 累積座標の更新
                self.current_node.cumulative_coords = []
                for i in range(len(self.current_node.points)):
                    self.current_node.cumulative_coords.append(create_cumulative_coord(i))

                # output_countを更新
                self.current_node.output_count = len(self.current_node.points)

            # UI更新
            self.update_info(self.current_node)

            # 3Dビューを更新
            if self.stl_viewer:
                self.stl_viewer.render_to_image()

            # Recalc Positionsと同じ効果を実行
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()

            # XMLファイル名を保存
            self.current_node.xml_file = file_name

        except Exception as e:
            print(f"Error loading XML: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_xml_with_stl(self):
        """XMLファイルとそれに対応するSTLファイルを読み込む"""
        if not self.current_node:
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
                        self.volume_input.setText(format_float_no_exp(volume))

                    # 質量の設定
                    mass_elem = inertial_elem.find('mass')
                    if mass_elem is not None:
                        mass = float(mass_elem.get('value', '0.0'))
                        self.current_node.mass_value = mass
                        self.mass_input.setText(format_float_no_exp(mass))

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

                # Center of Massの設定
                center_of_mass_elem = link_elem.find('center_of_mass')
                if center_of_mass_elem is not None:
                    com_xyz = center_of_mass_elem.text.strip().split()

            # 色情報の処理
            material_elem = root.find('.//material/color')
            if material_elem is not None:
                rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                rgb_values = [float(x) for x in rgba[:3]]  # RGBのみ使用
                self.current_node.node_color = rgb_values
                # 色の設定をUIに反映
                self._set_color_ui(rgb_values)
                self.update_color_sample()

            # 回転軸とjoint limitsの処理
            joint_elem = root.find('joint')
            if joint_elem is not None:
                # 回転軸の処理
                axis_elem = joint_elem.find('axis')
                if axis_elem is not None:
                    axis_xyz = axis_elem.get('xyz', '1 0 0').split()
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

                # Joint limitsの処理
                limit_elem = joint_elem.find('limit')
                if limit_elem is not None:
                    # XMLからはRadian値で読み込む
                    lower_rad = float(limit_elem.get('lower', -3.14159))
                    upper_rad = float(limit_elem.get('upper', 3.14159))
                    effort = float(limit_elem.get('effort', 10.0))
                    velocity = float(limit_elem.get('velocity', 3.0))
                    friction = float(limit_elem.get('friction', 0.05))
                    actuation_lag = float(limit_elem.get('actuationLag', 0.0))

                    # ノードにはRadian値で保存
                    self.current_node.joint_lower = lower_rad
                    self.current_node.joint_upper = upper_rad
                    self.current_node.joint_effort = effort
                    self.current_node.joint_velocity = velocity
                    self.current_node.joint_friction = friction
                    self.current_node.joint_actuation_lag = actuation_lag

                    # UI表示はDegreeに変換
                    self.lower_limit_input.setText(str(round(math.degrees(lower_rad), 2)))
                    self.upper_limit_input.setText(str(round(math.degrees(upper_rad), 2)))
                    self.effort_input.setText(format_float_no_exp(effort))
                    self.velocity_input.setText(format_float_no_exp(velocity))
                    self.friction_input.setText(format_float_no_exp(friction))
                    self.actuation_lag_input.setText(format_float_no_exp(actuation_lag))


            # ポイントの処理
            points = root.findall('point')
            num_points = len(points)

            # FooNodeの場合のみポート数を調整
            if isinstance(self.current_node, FooNode):
                # 現在のポート数を正しく取得
                current_ports = len(self.current_node.output_ports())

                # ポートを削除する前に、削除対象のポートの接続をすべてクリア
                if current_ports > num_points:
                    for i in range(num_points + 1, current_ports + 1):
                        port_name = f'out_{i}'
                        port = self.current_node.get_output(port_name)
                        if port:
                            port.clear_connections()

                while current_ports < num_points:
                    self.current_node._add_output()
                    current_ports += 1

                while current_ports > num_points:
                    self.current_node.remove_output()
                    current_ports -= 1

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

                # 累積座標の更新
                self.current_node.cumulative_coords = []
                for i in range(len(self.current_node.points)):
                    self.current_node.cumulative_coords.append(create_cumulative_coord(i))

                # output_countを更新
                self.current_node.output_count = len(self.current_node.points)

            # STLファイルの処理
            mesh_file = None
            if os.path.exists(stl_path):
                mesh_file = stl_path
            else:
                # STLが見つからない場合、DAEファイルを探す
                dae_path = os.path.join(xml_dir, f"{xml_name}.dae")
                if os.path.exists(dae_path):
                    mesh_file = dae_path
                else:
                    # どちらも見つからない場合、ダイアログを表示
                    print(f"Warning: Neither STL nor DAE file found: {stl_path}, {dae_path}")
                    msg_box = QtWidgets.QMessageBox()
                    msg_box.setIcon(QtWidgets.QMessageBox.Warning)
                    msg_box.setWindowTitle("Mesh File Not Found")
                    msg_box.setText("Neither STL nor DAE file found in the same directory.")
                    msg_box.setInformativeText("Would you like to select the mesh file manually?")
                    msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    msg_box.setDefaultButton(QtWidgets.QMessageBox.Yes)

                    if msg_box.exec() == QtWidgets.QMessageBox.Yes:
                        file_filter = get_mesh_file_filter(trimesh_available=True)
                        mesh_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                            self, "Select Mesh File", xml_dir, file_filter)
                        if mesh_file:
                            pass
                        else:
                            pass
                    else:
                        pass

            # メッシュファイルが見つかった、または選択された場合にロード
            if mesh_file:
                self.current_node.stl_file = mesh_file
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                    # STLモデルに色を適用
                    self.apply_color_to_stl()

            # UI更新
            self.update_info(self.current_node)

            # 3Dビューを更新
            if self.stl_viewer:
                self.stl_viewer.render_to_image()

            # Recalc Positionsと同じ効果を実行
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()

            # XMLファイル名を保存
            self.current_node.xml_file = xml_file

        except Exception as e:
            print(f"Error loading XML with STL: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_port_values(self):
        """Inspector内の全ての値を一括保存（Set All機能）"""
        if not self.current_node:
            return

        errors = []

        try:
            # 1. Node Name の保存
            try:
                new_name = self.name_edit.text()
                old_name = self.current_node.name()
                if new_name != old_name and new_name:
                    self.current_node.set_name(new_name)
            except Exception as e:
                errors.append(f"Node name: {str(e)}")

            # 2. Massless Decoration の保存
            try:
                self.current_node.massless_decoration = self.massless_checkbox.isChecked()
            except Exception as e:
                errors.append(f"Massless decoration: {str(e)}")

            # 3. Mass の保存
            try:
                if self.mass_input.text():
                    mass = float(self.mass_input.text())
                    self.current_node.mass_value = mass
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
            except ValueError as e:
                errors.append(f"Inertia: {str(e)}")

            # 5. Joint Limits の保存（DegreeからRadianに変換）
            try:
                if self.lower_limit_input.text():
                    self.current_node.joint_lower = math.radians(float(self.lower_limit_input.text()))
                if self.upper_limit_input.text():
                    self.current_node.joint_upper = math.radians(float(self.upper_limit_input.text()))
                if self.effort_input.text():
                    self.current_node.joint_effort = float(self.effort_input.text())
                if self.velocity_input.text():
                    self.current_node.joint_velocity = float(self.velocity_input.text())
                if self.friction_input.text():
                    self.current_node.joint_friction = float(self.friction_input.text())
                if self.actuation_lag_input.text():
                    self.current_node.joint_actuation_lag = float(self.actuation_lag_input.text())
                if self.damping_input.text():
                    self.current_node.joint_damping = float(self.damping_input.text())
                if self.stiffness_input.text():
                    self.current_node.joint_stiffness = float(self.stiffness_input.text())
                print(f"✓ Joint limits: lower={math.degrees(self.current_node.joint_lower):.2f}° ({self.current_node.joint_lower:.5f} rad), upper={math.degrees(self.current_node.joint_upper):.2f}° ({self.current_node.joint_upper:.5f} rad), effort={self.current_node.joint_effort}, velocity={self.current_node.joint_velocity}, friction={self.current_node.joint_friction}, actuation_lag={self.current_node.joint_actuation_lag}, damping={self.current_node.joint_damping}, stiffness={self.current_node.joint_stiffness}")
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

                # STLに色を適用
                if self.stl_viewer and hasattr(self.current_node, 'stl_file') and self.current_node.stl_file:
                    if self.current_node in self.stl_viewer.stl_actors:
                        actor = self.stl_viewer.stl_actors[self.current_node]
                        actor.GetProperty().SetColor(rgba_values[0], rgba_values[1], rgba_values[2])
                        actor.GetProperty().SetOpacity(rgba_values[3])
            except (ValueError, IndexError) as e:
                errors.append(f"Color: {str(e)}")

            # ノードの位置を再計算（必要な場合）
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()

            # STLビューアの更新
            if self.stl_viewer:
                self.stl_viewer.render_to_image()


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

    # ========== Helper Methods for Code Consolidation ==========

    def _get_node_file_path(self, attr_name):
        """ノードからファイルパスを安全に取得"""
        if not self.current_node:
            return None
        return getattr(self.current_node, attr_name, None) if hasattr(self.current_node, attr_name) else None

    def _show_message(self, title, message, msg_type='info'):
        """統一されたメッセージボックス表示"""
        if msg_type == 'warning':
            QtWidgets.QMessageBox.warning(self, title, message)
        elif msg_type == 'error':
            QtWidgets.QMessageBox.critical(self, title, message)
        else:
            QtWidgets.QMessageBox.information(self, title, message)

    class _OperationGuard:
        """二重実行防止用コンテキストマネージャー"""
        def __init__(self, parent, flag_name):
            self.parent = parent
            self.flag_name = flag_name

        def __enter__(self):
            if hasattr(self.parent, self.flag_name) and getattr(self.parent, self.flag_name):
                return False  # 既に実行中
            setattr(self.parent, self.flag_name, True)
            return True

        def __exit__(self, exc_type, exc_val, exc_tb):
            setattr(self.parent, self.flag_name, False)

    def _load_xml_with_encoding(self, file_path):
        """マルチエンコーディング対応でXMLを読み込み"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'shift_jis']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            raise ValueError(f"Could not decode XML file with any of: {encodings}")

        # XMLパース（BOM除去を試行）
        try:
            return ET.fromstring(content)
        except ET.ParseError:
            content = content.lstrip('\ufeff')
            return ET.fromstring(content)

    def _check_write_permission(self, directory):
        """書き込み権限を確認"""
        if not os.access(directory, os.W_OK):
            self._show_message("Save XML - Permission Error",
                             f"No write permission for directory:\n{directory}",
                             'error')
            return False
        return True

    def _check_disk_space(self, directory, required_mb=10):
        """ディスク容量を確認"""
        try:
            if hasattr(os, 'statvfs'):  # Unix/Linux/Mac
                stat_result = os.statvfs(directory)
                free_space = stat_result.f_bavail * stat_result.f_frsize
                required_space = required_mb * 1024 * 1024

                if free_space < required_space:
                    self._show_message("Save XML - Disk Space Error",
                                     f"Insufficient disk space.\n\n"
                                     f"Available: {free_space / 1024 / 1024:.1f} MB\n"
                                     f"Required: {required_mb} MB (minimum)",
                                     'error')
                    return False
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
        return True

    def _atomic_write_xml(self, tree, file_path):
        """アトミック操作でXMLファイルを保存"""
        import tempfile
        import shutil

        temp_fd = None
        temp_path = None
        backup_path = None

        try:
            # 一時ファイル作成
            file_dir = os.path.dirname(file_path) or '.'
            temp_fd, temp_path = tempfile.mkstemp(suffix='.xml.tmp', dir=file_dir, text=False)

            # 一時ファイルに書き込み
            with os.fdopen(temp_fd, 'wb') as f:
                tree.write(f, encoding='utf-8', xml_declaration=True)
                temp_fd = None

            # バックアップ作成
            if os.path.exists(file_path):
                backup_path = file_path + '.backup'
                try:
                    shutil.copy2(file_path, backup_path)
                except Exception as e:
                    print(f"Warning: Could not create backup: {e}")
                    backup_path = None

            # アトミックリネーム
            if os.name == 'nt':
                if os.path.exists(file_path):
                    os.replace(temp_path, file_path)
                else:
                    os.rename(temp_path, file_path)
            else:
                os.replace(temp_path, file_path)

            temp_path = None

            # バックアップ削除
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass

        finally:
            # クリーンアップ
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _validate_node_state(self, node, operation="operation"):
        """ノードの状態を検証し、問題があればエラーメッセージを返す"""
        if node is None:
            return f"No node selected for {operation}"

        # ノードが削除されていないか確認
        try:
            _ = node.name()  # アクセスしてみる
        except (RuntimeError, AttributeError) as e:
            return f"Node has been deleted or is invalid: {e}"

        # ノード名の検証
        try:
            node_name = node.name()
            if not node_name or not isinstance(node_name, str):
                return "Node has invalid name"
        except Exception as e:
            return f"Cannot access node name: {e}"

        # ノードがグラフに属しているか確認
        try:
            if hasattr(node, 'graph') and node.graph() is None:
                return "Node is not part of a graph"
        except Exception:
            pass  # graph()がない場合もあるので無視

        return None  # 問題なし

    def save_xml(self):
        """現在のノードのパラメータをXMLファイルに上書き保存"""
        # 二重実行防止
        with self._OperationGuard(self, '_save_xml_in_progress') as can_proceed:
            if not can_proceed:
                print("Save XML already in progress, ignoring duplicate call")
                return

            self._save_xml_impl()

    def _save_xml_impl(self):
        """save_xmlの実装本体"""
        # ノード状態の検証
        validation_error = self._validate_node_state(self.current_node, "Save XML")
        if validation_error:
            self._show_message("Save XML - Invalid Node State", validation_error, 'warning')
            return

        # ノード名のバリデーション
        node_name = self.current_node.name()
        if not node_name or not node_name.strip():
            self._show_message("Save XML - Warning",
                             "Node name is empty. Please set a valid node name first.",
                             'warning')
            return

        # 保存前にポート数とpointsを同期（FooNodeのみ）
        if isinstance(self.current_node, FooNode):
            if hasattr(self.current_node, 'points') and hasattr(self.current_node, 'output_ports'):
                try:
                    current_ports = len(self.current_node.output_ports())
                    num_points = len(self.current_node.points)

                    # ポート数がpointsより少ない場合、pointsを切り詰める
                    if current_ports < num_points:
                        self.current_node.points = self.current_node.points[:current_ports]
                        print(f"Adjusted points count from {num_points} to {current_ports}")

                    # ポート数がpointsより多い場合、デフォルトポイントを追加
                    elif current_ports > num_points:
                        for i in range(num_points, current_ports):
                            self.current_node.points.append({
                                'name': f'out_{i+1}',
                                'type': 'revolute',
                                'xyz': [0.0, 0.0, 0.0]
                            })
                        print(f"Adjusted points count from {num_points} to {current_ports}")

                    # output_countを更新
                    if hasattr(self.current_node, 'output_count'):
                        self.current_node.output_count = current_ports

                except Exception as e:
                    print(f"Warning: Could not synchronize ports: {e}")

        # XMLファイルパスの取得
        xml_file = self._get_node_file_path('xml_file')
        if not xml_file:
            # XMLファイルが設定されていない場合は、ファイルダイアログを表示
            safe_name = "".join(c for c in node_name if c.isalnum() or c in (' ', '-', '_')).strip()
            if not safe_name:
                safe_name = "node"
            default_filename = f"{safe_name}.xml"
            xml_file, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save XML File", default_filename, "XML Files (*.xml)")
            if not xml_file:
                return
            self.current_node.xml_file = xml_file

        # ファイルパスのバリデーション
        if not xml_file or not xml_file.strip():
            self._show_message("Save XML - Warning", "Invalid file path.", 'warning')
            return

        # ディレクトリの存在確認
        xml_dir = os.path.dirname(xml_file)
        if xml_dir and not os.path.exists(xml_dir):
            self._show_message("Save XML - Warning",
                             f"Directory does not exist:\n{xml_dir}",
                             'warning')
            return

        # ディスク容量と書き込み権限の確認
        target_dir = xml_dir if xml_dir else '.'
        if not self._check_write_permission(target_dir):
            return
        if not self._check_disk_space(target_dir):
            return

        try:
            # XML構造を作成
            root = ET.Element('urdf_part')

            # link要素を作成
            link_elem = ET.SubElement(root, 'link')
            link_elem.set('name', self.current_node.name())

            # inertial要素を作成
            inertial_elem = ET.SubElement(link_elem, 'inertial')

            # volume
            if hasattr(self.current_node, 'volume_value'):
                volume_elem = ET.SubElement(inertial_elem, 'volume')
                volume_elem.set('value', format_float_no_exp(self.current_node.volume_value))

            # mass
            if hasattr(self.current_node, 'mass_value'):
                mass_elem = ET.SubElement(inertial_elem, 'mass')
                mass_elem.set('value', format_float_no_exp(self.current_node.mass_value))

            # inertial origin
            if hasattr(self.current_node, 'inertial_origin') and self.current_node.inertial_origin:
                origin_elem = ET.SubElement(inertial_elem, 'origin')
                xyz = self.current_node.inertial_origin.get('xyz', [0, 0, 0])
                rpy = self.current_node.inertial_origin.get('rpy', [0, 0, 0])
                origin_elem.set('xyz', ' '.join([format_float_no_exp(v) for v in xyz]))
                origin_elem.set('rpy', ' '.join([format_float_no_exp(v) for v in rpy]))

            # inertia
            if hasattr(self.current_node, 'inertia') and self.current_node.inertia:
                inertia_elem = ET.SubElement(inertial_elem, 'inertia')
                for key in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                    value = self.current_node.inertia.get(key, 0)
                    inertia_elem.set(key, format_float_no_exp(value))

            # center of mass (必要に応じて)
            if hasattr(self.current_node, 'center_of_mass') and self.current_node.center_of_mass:
                com_elem = ET.SubElement(link_elem, 'center_of_mass')
                com_elem.text = ' '.join([format_float_no_exp(v) for v in self.current_node.center_of_mass])

            # visual要素を作成
            visual_elem = ET.SubElement(link_elem, 'visual')

            # geometry
            geometry_elem = ET.SubElement(visual_elem, 'geometry')
            mesh_elem = ET.SubElement(geometry_elem, 'mesh')
            if hasattr(self.current_node, 'stl_file') and self.current_node.stl_file:
                mesh_elem.set('filename', os.path.basename(self.current_node.stl_file))

            # material/color
            material_elem = ET.SubElement(visual_elem, 'material')
            material_elem.set('name', f"{self.current_node.name()}_material")
            color_elem = ET.SubElement(material_elem, 'color')
            if hasattr(self.current_node, 'node_color') and self.current_node.node_color:
                rgba = self.current_node.node_color + [1.0]  # Add alpha
                color_elem.set('rgba', ' '.join([format_float_no_exp(v) for v in rgba]))
            else:
                color_elem.set('rgba', '1.0 1.0 1.0 1.0')

            # collision要素を作成
            collision_elem = ET.SubElement(link_elem, 'collision')
            collision_geometry_elem = ET.SubElement(collision_elem, 'geometry')
            collision_mesh_elem = ET.SubElement(collision_geometry_elem, 'mesh')
            if hasattr(self.current_node, 'stl_file') and self.current_node.stl_file:
                collision_mesh_elem.set('filename', os.path.basename(self.current_node.stl_file))

            # collision_mesh要素を作成（別途コライダーメッシュが設定されている場合）
            if hasattr(self.current_node, 'collider_mesh') and self.current_node.collider_mesh:
                collision_mesh_path_elem = ET.SubElement(link_elem, 'collision_mesh')
                collision_mesh_path_elem.text = self.current_node.collider_mesh

            # joint要素を作成
            joint_elem = ET.SubElement(root, 'joint')
            joint_elem.set('name', f"{self.current_node.name()}_joint")

            # joint type and axis
            if hasattr(self.current_node, 'rotation_axis'):
                if self.current_node.rotation_axis == 3:  # Fixed
                    joint_elem.set('type', 'fixed')
                else:
                    joint_elem.set('type', 'revolute')
                    axis_elem = ET.SubElement(joint_elem, 'axis')
                    if self.current_node.rotation_axis == 0:  # X
                        axis_elem.set('xyz', '1 0 0')
                    elif self.current_node.rotation_axis == 1:  # Y
                        axis_elem.set('xyz', '0 1 0')
                    elif self.current_node.rotation_axis == 2:  # Z
                        axis_elem.set('xyz', '0 0 1')

            # joint limits
            if hasattr(self.current_node, 'joint_lower') and hasattr(self.current_node, 'joint_upper'):
                limit_elem = ET.SubElement(joint_elem, 'limit')
                limit_elem.set('lower', format_float_no_exp(self.current_node.joint_lower))
                limit_elem.set('upper', format_float_no_exp(self.current_node.joint_upper))
                if hasattr(self.current_node, 'joint_effort'):
                    limit_elem.set('effort', format_float_no_exp(self.current_node.joint_effort))
                if hasattr(self.current_node, 'joint_velocity'):
                    limit_elem.set('velocity', format_float_no_exp(self.current_node.joint_velocity))

            # joint dynamics
            if hasattr(self.current_node, 'joint_damping') or hasattr(self.current_node, 'joint_friction'):
                dynamics_elem = ET.SubElement(joint_elem, 'dynamics')
                if hasattr(self.current_node, 'joint_damping'):
                    dynamics_elem.set('damping', format_float_no_exp(self.current_node.joint_damping))
                if hasattr(self.current_node, 'joint_friction'):
                    dynamics_elem.set('friction', format_float_no_exp(self.current_node.joint_friction))

            # output_points要素を作成
            if hasattr(self.current_node, 'points') and self.current_node.points:
                points_elem = ET.SubElement(root, 'output_points')
                for i, point_data in enumerate(self.current_node.points):
                    point_elem = ET.SubElement(points_elem, 'point')
                    point_elem.set('name', point_data.get('name', f'out_{i}'))
                    point_elem.set('type', point_data.get('type', 'revolute'))

                    point_xyz_elem = ET.SubElement(point_elem, 'point_xyz')
                    xyz = point_data.get('xyz', [0, 0, 0])
                    point_xyz_elem.text = ' '.join([format_float_no_exp(v) for v in xyz])

            # XMLを整形して保存（アトミック操作）
            tree = ET.ElementTree(root)
            ET.indent(tree, space='  ')

            try:
                self._atomic_write_xml(tree, xml_file)
                print(f"XML saved to: {xml_file}")
                self._show_message("Save XML - Success",
                                 f"XML file saved successfully:\n{xml_file}",
                                 'info')

            except OSError as e:
                # ディスク容量不足、書き込み権限なしなど
                error_msg = f"Failed to save XML file:\n{xml_file}\n\n"
                if e.errno == 28:  # ENOSPC
                    error_msg += "Error: No space left on device"
                elif e.errno == 13:  # EACCES
                    error_msg += "Error: Permission denied"
                else:
                    error_msg += f"Error: {str(e)}"
                self._show_message("Save XML - File System Error", error_msg, 'error')
                return

        except Exception as e:
            print(f"Error saving XML: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_message("Save XML - Error",
                             f"An error occurred while saving XML:\n\n{str(e)}",
                             'error')

    def open_parts_editor(self):
        """PartsEditorを開き、現在のMeshとXMLを読み込む"""
        if not self.current_node:
            self._show_message("PartsEditor - Warning", "No node selected.", 'warning')
            return

        # STLファイルのパスを取得とバリデーション
        stl_file = self._get_node_file_path('stl_file')
        if not stl_file:
            self._show_message("PartsEditor - Warning",
                             "No mesh file loaded for this node.\n\n"
                             "Please load a mesh file first using 'Load Mesh' button.",
                             'warning')
            return

        if not os.path.exists(stl_file):
            self._show_message("PartsEditor - Warning",
                             f"Mesh file not found:\n{stl_file}\n\n"
                             "The file may have been moved or deleted.",
                             'warning')
            return

        if not os.access(stl_file, os.R_OK):
            self._show_message("PartsEditor - Warning",
                             f"Cannot read mesh file:\n{stl_file}\n\n"
                             "Please check file permissions.",
                             'warning')
            return

        # PartsEditorのパスを取得
        try:
            assembler_dir = os.path.dirname(os.path.abspath(__file__))
            parts_editor_path = os.path.join(assembler_dir, 'urdf_kitchen_PartsEditor.py')
        except Exception as e:
            self._show_message("PartsEditor - Error",
                             f"Could not determine PartsEditor path:\n\n{str(e)}",
                             'error')
            return

        if not os.path.exists(parts_editor_path):
            self._show_message("PartsEditor - Error",
                             f"PartsEditor not found at:\n{parts_editor_path}\n\n"
                             "Please ensure urdf_kitchen_PartsEditor.py is in the same directory.",
                             'error')
            return

        # Try to connect to existing PartsEditor instance first
        socket = QLocalSocket()
        server_name = "URDFKitchen_PartsEditor"
        socket.connectToServer(server_name)

        if socket.waitForConnected(1000):  # Wait up to 1 second
            # Existing PartsEditor found, send file path to load
            try:
                print(f"Connected to existing PartsEditor, sending file: {stl_file}")
                message = f"LOAD:{stl_file}".encode('utf-8')
                socket.write(message)
                socket.flush()

                # Wait for response
                if socket.waitForReadyRead(3000):  # Wait up to 3 seconds
                    response = socket.readAll().data().decode('utf-8')
                    print(f"PartsEditor response: {response}")
                    if response == "OK":
                        print("File loaded successfully in existing PartsEditor")
                    else:
                        print(f"PartsEditor error: {response}")

                socket.disconnectFromServer()
                return

            except Exception as e:
                print(f"Error communicating with PartsEditor: {e}")
                socket.disconnectFromServer()

        # No existing PartsEditor, launch new process
        try:
            print("No existing PartsEditor found, launching new instance")
            import subprocess
            import sys

            python_exe = sys.executable
            if not python_exe or not os.path.exists(python_exe):
                raise RuntimeError(f"Python executable not found: {python_exe}")

            process = subprocess.Popen(
                [python_exe, parts_editor_path, stl_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # プロセス起動確認
            import time
            time.sleep(0.1)
            poll = process.poll()
            if poll is not None:
                stderr = process.stderr.read().decode('utf-8', errors='replace')
                raise RuntimeError(f"PartsEditor exited immediately.\n\nError output:\n{stderr[:500]}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_message("PartsEditor - Error",
                             f"Failed to launch PartsEditor:\n\n{str(e)}",
                             'error')

    def reload_node_files(self):
        """現在のノードのXMLファイルとMeshファイルを再読み込み"""
        with self._OperationGuard(self, '_reload_in_progress') as can_proceed:
            if not can_proceed:
                print("Reload already in progress, ignoring duplicate call")
                return

            self._reload_node_files_impl()

    def _reload_node_files_impl(self):
        """reload_node_filesの実装本体"""
        print("\n" + "="*60)
        print("RELOAD BUTTON CLICKED - Starting reload process...")
        print("="*60)

        # ノード状態の検証
        validation_error = self._validate_node_state(self.current_node, "Reload")
        if validation_error:
            self._show_message("Reload - Invalid Node State", validation_error, 'warning')
            return

        # XMLファイルとSTLファイルのパスを取得
        xml_file = self._get_node_file_path('xml_file')
        stl_file = self._get_node_file_path('stl_file')

        # XMLファイルパスが未設定の場合、STLファイルパスから推測
        if not xml_file and stl_file:
            # STLファイルと同じディレクトリ・同じ名前のXMLファイルを推測
            xml_file = os.path.splitext(stl_file)[0] + '.xml'
            print(f"XML file path inferred from STL: {xml_file}")

        if not xml_file and not stl_file:
            self._show_message("Reload - Warning",
                             "No XML or Mesh file loaded for this node.",
                             'warning')
            return

        # ファイルの存在確認
        xml_exists = xml_file and os.path.exists(xml_file)
        stl_exists = stl_file and os.path.exists(stl_file)

        print(f"File check:")
        print(f"  XML file: {xml_file}")
        print(f"  XML exists: {xml_exists}")
        print(f"  STL file: {stl_file}")
        print(f"  STL exists: {stl_exists}")

        # XMLファイルが見つからない場合、ユーザーに選択してもらう
        if not xml_exists and xml_file:
            # ダイアログで確認
            reply = QtWidgets.QMessageBox.question(
                self,
                "XML File Not Found",
                f"XML file not found:\n{xml_file}\n\nWould you like to select the XML file manually?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )

            if reply == QtWidgets.QMessageBox.Yes:
                # ファイル選択ダイアログを表示
                initial_dir = os.path.dirname(xml_file) if xml_file else os.path.dirname(stl_file) if stl_file else ""
                selected_xml, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "Select XML File",
                    initial_dir,
                    "XML Files (*.xml);;All Files (*)"
                )

                if selected_xml:
                    xml_file = selected_xml
                    xml_exists = os.path.exists(xml_file)
                    print(f"User selected XML file: {xml_file}")
                    print(f"  XML exists: {xml_exists}")
                else:
                    print("User cancelled XML file selection")
                    xml_file = None
                    xml_exists = False
            else:
                print("User chose not to select XML file manually")
                xml_file = None
                xml_exists = False

        if not xml_exists and not stl_exists:
            files_msg = "Files not found:\n"
            if xml_file:
                files_msg += f"XML: {xml_file}\n"
            if stl_file:
                files_msg += f"Mesh: {stl_file}"
            self._show_message("Reload - Warning", files_msg, 'warning')
            return

        try:
            loaded_files = []
            warnings = []

            # XMLファイルが存在する場合は再読み込み
            if xml_exists:
                print(f"\n>>> Loading XML file: {xml_file}")
                try:
                    root = self._load_xml_with_encoding(xml_file)

                    # STLファイルパスをXMLから取得（相対パス・絶対パス両対応）
                    mesh_from_xml = None
                    mesh_elem = root.find('.//visual/geometry/mesh')
                    if mesh_elem is not None:
                        mesh_filename = mesh_elem.get('filename')
                        if mesh_filename:
                            # 絶対パスの場合はそのまま、相対パスの場合はXMLのディレクトリを基準
                            if os.path.isabs(mesh_filename):
                                mesh_from_xml = mesh_filename
                            else:
                                xml_dir = os.path.dirname(xml_file)
                                mesh_from_xml = os.path.join(xml_dir, mesh_filename)

                            # XMLから取得したメッシュファイルが存在するか確認
                            if mesh_from_xml and os.path.exists(mesh_from_xml):
                                stl_file = mesh_from_xml
                                stl_exists = True
                            elif mesh_from_xml:
                                warnings.append(f"Mesh file not found: {mesh_from_xml}")

                    # XMLの全パラメータをロード
                    self._reload_xml_parameters(root, xml_file, stl_file if stl_exists else None)
                    loaded_files.append(xml_file)

                    # XMLファイルパスをノードに保存（次回のReloadで使用）
                    self.current_node.xml_file = xml_file
                    print(f"Saved XML file path to node: {xml_file}")

                except ET.ParseError as e:
                    self._show_message("Reload - XML Parse Error",
                                     f"Failed to parse XML file:\n{xml_file}\n\nError: {str(e)}",
                                     'error')
                    return

            # XMLファイルがなくてSTLファイルのみの場合
            elif stl_exists:
                print(f"\n>>> XML file not found, loading STL only: {stl_file}")
                print("WARNING: RecalcPositions will NOT be executed (XML file required)")
                self.current_node.stl_file = stl_file
                if self.stl_viewer:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                else:
                    warnings.append("STL viewer not initialized")
                loaded_files.append(stl_file)

            # 結果メッセージを表示
            if loaded_files:
                message = "Reloaded:\n" + "\n".join(loaded_files)
                if warnings:
                    message += "\n\nWarnings:\n" + "\n".join(warnings)
                self._show_message("Reload - Success", message, 'info')
            else:
                self._show_message("Reload - Warning", "No files could be reloaded.", 'warning')

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_message("Reload - Error",
                             f"An error occurred while reloading:\n\n{str(e)}",
                             'error')

    def _reload_xml_parameters(self, root, xml_file, stl_file):
        """XMLパラメータを再読み込み（load_xml_with_stlの処理を使用）"""
        import math

        def validate_numeric(value, name, min_val=None, max_val=None, allow_zero=True):
            """数値の検証（NaN、Inf、範囲チェック）"""
            try:
                if math.isnan(value) or math.isinf(value):
                    print(f"Warning: Invalid {name} value (NaN or Inf), using default")
                    return None
                if not allow_zero and value == 0:
                    print(f"Warning: {name} cannot be zero, using default")
                    return None
                if min_val is not None and value < min_val:
                    print(f"Warning: {name} value {value} below minimum {min_val}, using default")
                    return None
                if max_val is not None and value > max_val:
                    print(f"Warning: {name} value {value} above maximum {max_val}, using default")
                    return None
                return value
            except (TypeError, ValueError):
                print(f"Warning: Could not validate {name} value: {value}")
                return None

        # 質量、慣性、volumeの処理
        link_elem = root.find('link')
        if link_elem is not None:
            inertial_elem = link_elem.find('inertial')
            if inertial_elem is not None:
                # Volume
                volume_elem = inertial_elem.find('volume')
                if volume_elem is not None:
                    try:
                        volume_value = float(volume_elem.get('value', '0.001'))
                        validated_volume = validate_numeric(volume_value, 'volume', min_val=0.0, max_val=1000.0)
                        if validated_volume is not None:
                            self.current_node.volume_value = validated_volume
                            if hasattr(self, 'volume_input'):
                                self.volume_input.setText(format_float_no_exp(validated_volume))
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing volume value: {e}")

                # Mass
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    try:
                        mass_value = float(mass_elem.get('value', '1.0'))
                        validated_mass = validate_numeric(mass_value, 'mass', min_val=0.0001, max_val=10000.0, allow_zero=False)
                        if validated_mass is not None:
                            self.current_node.mass_value = validated_mass
                            self.mass_input.setText(format_float_no_exp(validated_mass))
                        else:
                            print("Using default mass value: 1.0")
                            self.current_node.mass_value = 1.0
                            self.mass_input.setText(format_float_no_exp(1.0))
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing mass value: {e}, using default")
                        self.current_node.mass_value = 1.0
                        self.mass_input.setText(format_float_no_exp(1.0))

                # Inertial Origin (COM position)
                origin_elem = inertial_elem.find('origin')
                if origin_elem is not None:
                    try:
                        xyz_text = origin_elem.get('xyz', '0 0 0')
                        xyz_values = [float(x) for x in xyz_text.split()]
                        if len(xyz_values) >= 3:
                            if not hasattr(self.current_node, 'inertial_origin'):
                                self.current_node.inertial_origin = {}
                            self.current_node.inertial_origin['xyz'] = xyz_values[:3]
                            # COM入力フィールドに反映
                            if hasattr(self, 'com_x_input'):
                                self.com_x_input.setText(format_float_no_exp(xyz_values[0]))
                                self.com_y_input.setText(format_float_no_exp(xyz_values[1]))
                                self.com_z_input.setText(format_float_no_exp(xyz_values[2]))

                        # RPY (orientation)
                        rpy_text = origin_elem.get('rpy', '0 0 0')
                        rpy_values = [float(x) for x in rpy_text.split()]
                        if len(rpy_values) >= 3:
                            if not hasattr(self.current_node, 'inertial_origin'):
                                self.current_node.inertial_origin = {}
                            self.current_node.inertial_origin['rpy'] = rpy_values[:3]
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing inertial origin: {e}")

                inertia_elem = inertial_elem.find('inertia')
                if inertia_elem is not None:
                    try:
                        ixx = float(inertia_elem.get('ixx', 0.01))
                        iyy = float(inertia_elem.get('iyy', 0.01))
                        izz = float(inertia_elem.get('izz', 0.01))

                        # 慣性値の検証（正の値のみ）
                        validated_ixx = validate_numeric(ixx, 'ixx', min_val=0.0, max_val=1000.0)
                        validated_iyy = validate_numeric(iyy, 'iyy', min_val=0.0, max_val=1000.0)
                        validated_izz = validate_numeric(izz, 'izz', min_val=0.0, max_val=1000.0)

                        self.current_node.ixx = validated_ixx if validated_ixx is not None else 0.01
                        self.current_node.iyy = validated_iyy if validated_iyy is not None else 0.01
                        self.current_node.izz = validated_izz if validated_izz is not None else 0.01

                        self.ixx_input.setText(format_float_no_exp(self.current_node.ixx))
                        self.iyy_input.setText(format_float_no_exp(self.current_node.iyy))
                        self.izz_input.setText(format_float_no_exp(self.current_node.izz))
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing inertia values: {e}, using defaults")
                        self.current_node.ixx = 0.01
                        self.current_node.iyy = 0.01
                        self.current_node.izz = 0.01
                        self.ixx_input.setText(format_float_no_exp(0.01))
                        self.iyy_input.setText(format_float_no_exp(0.01))
                        self.izz_input.setText(format_float_no_exp(0.01))

        # 色情報の処理（XMLに色情報がある場合のみ更新）
        material_elem = root.find('.//material/color')
        if material_elem is not None:
            rgba_text = material_elem.get('rgba', '')
            if rgba_text:
                try:
                    rgba = rgba_text.strip().split()
                    if len(rgba) >= 3:
                        rgb_values = [float(x) for x in rgba[:3]]
                        # 有効な色値か確認（0.0～1.0の範囲）
                        if all(0.0 <= v <= 1.0 for v in rgb_values):
                            self.current_node.node_color = rgb_values
                            self._set_color_ui(rgb_values)
                            # カラーサンプルも更新
                            try:
                                self.update_color_sample()
                                print(f"Color updated from XML: RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")
                            except Exception as e:
                                print(f"Warning: Could not update color sample: {e}")
                        else:
                            print(f"Warning: Color values out of range (0.0-1.0): {rgb_values}")
                    else:
                        print(f"Warning: Insufficient color values in XML: {rgba}")
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse color from XML: {e}")
            # rgba属性が空の場合は既存の色を保持
        # material要素がない場合も既存の色を保持

        # 回転軸とjoint typeの処理
        joint_elem = root.find('joint')
        if joint_elem is not None:
            # Joint typeの処理
            joint_type = joint_elem.get('type', 'revolute')
            if joint_type == 'fixed':
                # Fixedタイプの場合
                self.current_node.rotation_axis = 3
                self.axis_group.button(3).setChecked(True)
            else:
                # Revoluteタイプの場合、axis要素から軸を読み込む
                axis_elem = joint_elem.find('axis')
                if axis_elem is not None:
                    axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                    axis_values = [float(x) for x in axis_xyz]
                    if axis_values[2] == 1:
                        self.current_node.rotation_axis = 2
                        self.axis_group.button(2).setChecked(True)
                    elif axis_values[1] == 1:
                        self.current_node.rotation_axis = 1
                        self.axis_group.button(1).setChecked(True)
                    else:
                        self.current_node.rotation_axis = 0
                        self.axis_group.button(0).setChecked(True)
                else:
                    # axis要素がない場合はデフォルトでX軸
                    self.current_node.rotation_axis = 0
                    self.axis_group.button(0).setChecked(True)

            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                try:
                    lower_rad = float(limit_elem.get('lower', -3.14159))
                    upper_rad = float(limit_elem.get('upper', 3.14159))
                    effort = float(limit_elem.get('effort', 10.0))
                    velocity = float(limit_elem.get('velocity', 3.0))
                    friction = float(limit_elem.get('friction', 0.05))
                    actuation_lag = float(limit_elem.get('actuationLag', 0.0))

                    # Joint limits の検証
                    validated_lower = validate_numeric(lower_rad, 'joint_lower', min_val=-2*math.pi, max_val=2*math.pi)
                    validated_upper = validate_numeric(upper_rad, 'joint_upper', min_val=-2*math.pi, max_val=2*math.pi)
                    validated_effort = validate_numeric(effort, 'effort', min_val=0.0, max_val=1000.0)
                    validated_velocity = validate_numeric(velocity, 'velocity', min_val=0.0, max_val=100.0)
                    validated_friction = validate_numeric(friction, 'friction', min_val=0.0, max_val=10.0)
                    validated_lag = validate_numeric(actuation_lag, 'actuation_lag', min_val=0.0, max_val=10.0)

                    # デフォルト値を使用
                    self.current_node.joint_lower = validated_lower if validated_lower is not None else -3.14159
                    self.current_node.joint_upper = validated_upper if validated_upper is not None else 3.14159
                    self.current_node.joint_effort = validated_effort if validated_effort is not None else 10.0
                    self.current_node.joint_velocity = validated_velocity if validated_velocity is not None else 3.0
                    self.current_node.joint_friction = validated_friction if validated_friction is not None else 0.05
                    self.current_node.joint_actuation_lag = validated_lag if validated_lag is not None else 0.0

                    # lower > upper の場合は警告して入れ替え
                    if self.current_node.joint_lower > self.current_node.joint_upper:
                        print(f"Warning: joint_lower ({self.current_node.joint_lower}) > joint_upper ({self.current_node.joint_upper}), swapping values")
                        self.current_node.joint_lower, self.current_node.joint_upper = self.current_node.joint_upper, self.current_node.joint_lower

                    self.lower_limit_input.setText(str(round(math.degrees(self.current_node.joint_lower), 2)))
                    self.upper_limit_input.setText(str(round(math.degrees(self.current_node.joint_upper), 2)))
                    self.effort_input.setText(format_float_no_exp(self.current_node.joint_effort))
                    self.velocity_input.setText(format_float_no_exp(self.current_node.joint_velocity))
                    self.friction_input.setText(format_float_no_exp(self.current_node.joint_friction))
                    self.actuation_lag_input.setText(format_float_no_exp(self.current_node.joint_actuation_lag))
                except (ValueError, TypeError) as e:
                    print(f"Error parsing joint limit values: {e}, using defaults")
                    self.current_node.joint_lower = -3.14159
                    self.current_node.joint_upper = 3.14159
                    self.current_node.joint_effort = 10.0
                    self.current_node.joint_velocity = 3.0
                    self.current_node.joint_friction = 0.05
                    self.current_node.joint_actuation_lag = 0.0
                    self.lower_limit_input.setText(format_float_no_exp(-180.0))
                    self.upper_limit_input.setText(format_float_no_exp(180.0))
                    self.effort_input.setText(format_float_no_exp(10.0))
                    self.velocity_input.setText(format_float_no_exp(3.0))
                    self.friction_input.setText(format_float_no_exp(0.05))
                    self.actuation_lag_input.setText(format_float_no_exp(0.0))

        # ポイントの処理（FooNodeのみ）
        if isinstance(self.current_node, FooNode):
            # output_points要素からpoint要素を取得
            output_points_elem = root.find('output_points')
            if output_points_elem is not None:
                points = output_points_elem.findall('point')
            else:
                # 後方互換性のため、直接point要素を探す
                points = root.findall('point')
            num_points = len(points)

            # 現在のポート数を取得
            try:
                current_ports = len(self.current_node.output_ports())
            except Exception as e:
                print(f"Warning: Could not get output ports count: {e}")
                current_ports = 0

            # ポート削除が必要な場合、先に接続をクリア
            if current_ports > num_points:
                for i in range(num_points + 1, current_ports + 1):
                    port_name = f'out_{i}'
                    try:
                        port = self.current_node.get_output(port_name)
                        if port:
                            # 接続を安全にクリア
                            try:
                                port.clear_connections()
                            except Exception as e:
                                print(f"Warning: Could not clear connections for {port_name}: {e}")
                    except Exception as e:
                        print(f"Warning: Could not get port {port_name}: {e}")

            # ポート数を調整
            while current_ports < num_points:
                try:
                    self.current_node._add_output()
                    current_ports += 1
                except Exception as e:
                    print(f"Error adding output port: {e}")
                    break

            while current_ports > num_points:
                try:
                    self.current_node.remove_output()
                    current_ports -= 1
                except Exception as e:
                    print(f"Error removing output port: {e}")
                    break

            # ポイントデータをクリアして再構築
            self.current_node.points = []

            for point_elem in points:
                point_name = point_elem.get('name', 'unnamed')
                point_type = point_elem.get('type', 'revolute')
                point_xyz_elem = point_elem.find('point_xyz')

                if point_xyz_elem is not None and point_xyz_elem.text:
                    try:
                        xyz_text = point_xyz_elem.text.strip()
                        if xyz_text:
                            xyz_values = [float(x) for x in xyz_text.split()]
                            # 3つの値が揃っているか確認
                            if len(xyz_values) == 3:
                                self.current_node.points.append({
                                    'name': point_name,
                                    'type': point_type,
                                    'xyz': xyz_values
                                })
                            else:
                                print(f"Warning: Invalid point coordinates for {point_name}: {xyz_values}")
                                # デフォルト値を追加
                                self.current_node.points.append({
                                    'name': point_name,
                                    'type': point_type,
                                    'xyz': [0.0, 0.0, 0.0]
                                })
                    except ValueError as e:
                        print(f"Warning: Could not parse point coordinates for {point_name}: {e}")
                        # デフォルト値を追加
                        self.current_node.points.append({
                            'name': point_name,
                            'type': point_type,
                            'xyz': [0.0, 0.0, 0.0]
                        })
                else:
                    # point_xyz要素がない場合はデフォルト値
                    self.current_node.points.append({
                        'name': point_name,
                        'type': point_type,
                        'xyz': [0.0, 0.0, 0.0]
                    })

            # 累積座標の計算
            def create_cumulative_coord(index):
                if index == 0:
                    if len(self.current_node.points) > 0:
                        return self.current_node.points[0]['xyz'].copy()
                    else:
                        return [0.0, 0.0, 0.0]
                else:
                    prev_coord = self.current_node.cumulative_coords[index - 1]
                    curr_point = self.current_node.points[index]['xyz']
                    return [
                        prev_coord[0] + curr_point[0],
                        prev_coord[1] + curr_point[1],
                        prev_coord[2] + curr_point[2]
                    ]

            # 累積座標を再計算
            self.current_node.cumulative_coords = []
            for i in range(len(self.current_node.points)):
                self.current_node.cumulative_coords.append(create_cumulative_coord(i))

            # output_countを更新
            self.current_node.output_count = len(self.current_node.points)
        else:
            # FooNode以外の場合は警告を出力
            print(f"Warning: Node type {type(self.current_node).__name__} does not support point management")

        # XMLファイルパスを保存
        self.current_node.xml_file = xml_file

        # STLファイルが指定されている場合は読み込み
        if stl_file and os.path.exists(stl_file):
            self.current_node.stl_file = stl_file
            if self.stl_viewer:
                try:
                    self.stl_viewer.load_stl_for_node(self.current_node)
                except Exception as e:
                    print(f"Warning: Failed to load STL file in viewer: {e}")

        # Output Portsセクションを更新
        try:
            if hasattr(self, 'update_output_ports'):
                print(f"Updating output ports UI (current points count: {len(self.current_node.points) if hasattr(self.current_node, 'points') else 0})")
                self.update_output_ports(self.current_node)
            else:
                print("Warning: update_output_ports method not found")
        except Exception as e:
            print(f"Warning: Failed to update output ports: {e}")
            import traceback
            traceback.print_exc()

        # ノードグラフの更新（色とノード表示の反映）
        if hasattr(self.current_node, 'update') and callable(self.current_node.update):
            try:
                self.current_node.update()
            except Exception as e:
                print(f"Warning: Failed to update node: {e}")

        # 色をSTLモデルに適用（RecalcPositionsの前に実行）
        try:
            if hasattr(self, 'apply_color_to_stl'):
                self.apply_color_to_stl()
                print("Color applied to STL model")
        except Exception as e:
            print(f"Warning: Failed to apply color to STL: {e}")

        # 3Dビューの更新とポジション再計算（LoadXMLと同じ方法で実行）
        print("\n" + "-"*60)
        print("Step: Recalculating all node positions and updating 3D view...")
        print("-"*60)

        try:
            # まずSTLビューアを更新（LoadXMLと同じ順序）
            if self.stl_viewer:
                print("Updating STL viewer...")
                self.stl_viewer.render_to_image()
                print("✓ STL viewer updated")

            # Recalc Positionsと同じ効果を実行（LoadXMLと同じ方法）
            # 優先順位1: current_node.graph を使用（LoadXMLと同じ）
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                print(f"✓ Using current_node.graph: {type(self.current_node.graph).__name__}")
                self.current_node.graph.recalculate_all_positions()
                print("✓ Position recalculation completed successfully")
            # 優先順位2: self.graph を使用（フォールバック）
            elif hasattr(self, 'graph') and self.graph:
                print(f"✓ Using self.graph: {type(self.graph).__name__}")
                self.graph.recalculate_all_positions()
                print("✓ Position recalculation completed successfully")
            else:
                print("Warning: No graph object available for position recalculation")

        except Exception as e:
            print(f"Error: Failed to update 3D view: {e}")
            import traceback
            traceback.print_exc()

        print(f"✓ Reload completed: {xml_file}")

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

            # カラーサンプルボックスを更新
            self.update_color_sample()

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

    def update_hide_mesh(self, state):
        """Hide Meshの状態を更新して3Dビューのメッシュを表示/非表示"""
        if self.current_node:
            hide = bool(state)
            self.current_node.hide_mesh = hide

            # 3DビューアのメッシュをHide/Show
            if self.stl_viewer and hasattr(self.stl_viewer, 'stl_actors'):
                if self.current_node in self.stl_viewer.stl_actors:
                    actor = self.stl_viewer.stl_actors[self.current_node]
                    # hide=Trueなら非表示(VisibilityOff)、hide=Falseなら表示(VisibilityOn)
                    actor.SetVisibility(not hide)
                    self.stl_viewer.render_to_image()

    def update_blanklink(self, state):
        """Blanklinkの状態を更新（BaseLinkNode用）"""
        if self.current_node and isinstance(self.current_node, BaseLinkNode):
            self.current_node.blank_link = bool(state)

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
                # インプットフィールドから値を取得（Degree表示）
                lower_text = self.lower_limit_input.text()
                if not lower_text:
                    lower_text = self.lower_limit_input.placeholderText()

                lower_deg = float(lower_text)
                lower_rad = math.radians(lower_deg)

                # 現在の変換を保存
                self.stl_viewer.store_current_transform(self.current_node)
                # 指定角度を表示
                self.stl_viewer.show_angle(self.current_node, lower_rad)
            except ValueError:
                pass

    def look_upper_limit(self):
        """Upper limitの角度を表示"""
        if self.current_node and self.stl_viewer:
            try:
                # インプットフィールドから値を取得（Degree表示）
                upper_text = self.upper_limit_input.text()
                if not upper_text:
                    upper_text = self.upper_limit_input.placeholderText()

                upper_deg = float(upper_text)
                upper_rad = math.radians(upper_deg)

                # 現在の変換を保存
                self.stl_viewer.store_current_transform(self.current_node)
                # 指定角度を表示
                self.stl_viewer.show_angle(self.current_node, upper_rad)
            except ValueError:
                pass

    def look_zero_limit(self):
        """0度の角度を表示"""
        if self.current_node and self.stl_viewer:
            # 現在の変換を保存
            self.stl_viewer.store_current_transform(self.current_node)
            # 0ラジアンを表示
            self.stl_viewer.show_angle(self.current_node, 0.0)

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
                except ValueError:
                    self.look_inertial_origin_toggle.setChecked(False)
            else:
                # 座標系を非表示
                self.stl_viewer.hide_inertial_origin(self.current_node)

    def set_joint_limits(self):
        """Joint limitsの値をノードに保存"""
        if not self.current_node:
            return

        try:
            # Lower limitの保存（DegreeからRadianに変換）
            lower_text = self.lower_limit_input.text()
            if lower_text:
                self.current_node.joint_lower = math.radians(float(lower_text))

            # Upper limitの保存（DegreeからRadianに変換）
            upper_text = self.upper_limit_input.text()
            if upper_text:
                self.current_node.joint_upper = math.radians(float(upper_text))

            # Effortの保存
            effort_text = self.effort_input.text()
            if effort_text:
                self.current_node.joint_effort = float(effort_text)

            # Velocityの保存
            velocity_text = self.velocity_input.text()
            if velocity_text:
                self.current_node.joint_velocity = float(velocity_text)

            # Frictionの保存
            friction_text = self.friction_input.text()
            if friction_text:
                self.current_node.joint_friction = float(friction_text)

            # ActuationLagの保存
            actuation_lag_text = self.actuation_lag_input.text()
            if actuation_lag_text:
                self.current_node.joint_actuation_lag = float(actuation_lag_text)

            # Dampingの保存
            damping_text = self.damping_input.text()
            if damping_text:
                self.current_node.joint_damping = float(damping_text)

            # Stiffnessの保存
            stiffness_text = self.stiffness_input.text()
            if stiffness_text:
                self.current_node.joint_stiffness = float(stiffness_text)

            print(f"Joint limits set: lower={math.degrees(self.current_node.joint_lower):.2f}° ({self.current_node.joint_lower:.5f} rad), upper={math.degrees(self.current_node.joint_upper):.2f}° ({self.current_node.joint_upper:.5f} rad), effort={self.current_node.joint_effort}, velocity={self.current_node.joint_velocity}, friction={self.current_node.joint_friction}, actuation_lag={self.current_node.joint_actuation_lag}, damping={self.current_node.joint_damping}, stiffness={self.current_node.joint_stiffness}")

            QtWidgets.QMessageBox.information(
                self,
                "Joint Limits Set",
                f"Joint limits have been set successfully.\n\n"
                f"Lower: {math.degrees(self.current_node.joint_lower):.2f}° ({self.current_node.joint_lower:.5f} rad)\n"
                f"Upper: {math.degrees(self.current_node.joint_upper):.2f}° ({self.current_node.joint_upper:.5f} rad)\n"
                f"Effort: {self.current_node.joint_effort}\n"
                f"Damping: {self.current_node.joint_damping}\n"
                f"Stiffness: {self.current_node.joint_stiffness}\n"
                f"Velocity: {self.current_node.joint_velocity}\n"
                f"Friction: {self.current_node.joint_friction}\n"
                f"ActuationLag: {self.current_node.joint_actuation_lag}"
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
                    try:
                        print("  - Fixing normals...")
                        mesh.fix_normals()
                    except AttributeError:
                        print("  - Skipping normals fixing (method not available)")

                    # 重複面の削除（trimeshの古いバージョンでは利用不可）
                    try:
                        print("  - Removing duplicate faces...")
                        mesh.remove_duplicate_faces()
                    except AttributeError:
                        print("  - Skipping duplicate faces removal (method not available)")

                    # 退化面の削除
                    try:
                        print("  - Removing degenerate faces...")
                        mesh.remove_degenerate_faces()
                    except AttributeError:
                        print("  - Skipping degenerate faces removal (method not available)")

                    # 穴の修復
                    try:
                        print("  - Filling holes...")
                        mesh.fill_holes()
                    except AttributeError:
                        print("  - Skipping holes filling (method not available)")

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

            # UIフィールドに設定（高精度、指数表記なし）
            self.inertial_x_input.setText(format_float_no_exp(center_of_mass[0]))
            self.inertial_y_input.setText(format_float_no_exp(center_of_mass[1]))
            self.inertial_z_input.setText(format_float_no_exp(center_of_mass[2]))

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
                    try:
                        print("  - Fixing normals...")
                        mesh.fix_normals()
                    except AttributeError:
                        print("  - Skipping normals fixing (method not available)")

                    # 重複面の削除（trimeshの古いバージョンでは利用不可）
                    try:
                        print("  - Removing duplicate faces...")
                        mesh.remove_duplicate_faces()
                    except AttributeError:
                        print("  - Skipping duplicate faces removal (method not available)")

                    # 退化面の削除
                    try:
                        print("  - Removing degenerate faces...")
                        mesh.remove_degenerate_faces()
                    except AttributeError:
                        print("  - Skipping degenerate faces removal (method not available)")

                    # 穴の修復
                    try:
                        print("  - Filling holes...")
                        mesh.fill_holes()
                    except AttributeError:
                        print("  - Skipping holes filling (method not available)")

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

            # VTKポリデータを取得
            if self.stl_viewer and self.current_node in self.stl_viewer.stl_actors:
                actor = self.stl_viewer.stl_actors[self.current_node]
                poly_data = actor.GetMapper().GetInput()
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No STL Actor",
                    "STL model is not loaded in the viewer."
                )
                return

            # PartsEditorと同じ方法でVTKを使って体積を計算
            mass_properties = vtk.vtkMassProperties()
            mass_properties.SetInputData(poly_data)
            mass_properties.Update()
            vtk_volume = mass_properties.GetVolume()

            print(f"Volume comparison:")
            print(f"  Trimesh volume: {mesh.volume:.9f} m³")
            print(f"  VTK volume: {vtk_volume:.9f} m³")

            # PartsEditorと同じ方法でDensityを計算（VTK体積を使用）
            density = mass / vtk_volume
            print(f"Calculated density: {density:.6f} kg/m³ (using VTK volume)")

            # 既存のInertial Origin値を取得（ユーザー指定の値を使用）
            # Center of Massは再計算せず、UIフィールドの値を使用
            try:
                center_of_mass = [
                    float(self.inertial_x_input.text()) if self.inertial_x_input.text() else 0.0,
                    float(self.inertial_y_input.text()) if self.inertial_y_input.text() else 0.0,
                    float(self.inertial_z_input.text()) if self.inertial_z_input.text() else 0.0
                ]
            except ValueError:
                center_of_mass = [0.0, 0.0, 0.0]

            print(f"Using center of mass from UI: {center_of_mass}")

            # urdf_kitchen_utilsの四面体分解法を使用してInertia Tensorを計算
            # UIで指定されたCenter of Massを使用（PartsEditorと完全に同じ方法）
            print(f"Calculating inertia tensor using tetrahedral decomposition method...")
            inertia_tensor, volume_integral = calculate_inertia_tetrahedral(
                poly_data, density, center_of_mass
            )

            # 体積の検証
            print(f"Volume from integration: {volume_integral:.9f} (VTK volume: {vtk_volume:.9f})")
            volume_ratio = volume_integral / vtk_volume if vtk_volume > 0 else 0
            if abs(volume_ratio - 1.0) > 0.01:
                print(f"Warning: Volume mismatch! Ratio = {volume_ratio:.4f}")
                print("This may indicate non-closed mesh or incorrect orientation")

            print(f"Calculated inertia tensor at specified origin:")
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

            # UIフィールドに慣性値を設定（高精度、指数表記なし）
            self.ixx_input.setText(format_float_no_exp(inertia_tensor[0, 0]))
            self.ixy_input.setText(format_float_no_exp(inertia_tensor[0, 1]))
            self.ixz_input.setText(format_float_no_exp(inertia_tensor[0, 2]))
            self.iyy_input.setText(format_float_no_exp(inertia_tensor[1, 1]))
            self.iyz_input.setText(format_float_no_exp(inertia_tensor[1, 2]))
            self.izz_input.setText(format_float_no_exp(inertia_tensor[2, 2]))

            # Inertial Originは既存の値を維持（変更しない）

            # 成功メッセージ
            repair_msg = ""
            if repair_performed:
                repair_msg = "Mesh Repair: Performed (in memory only)\n"

            QtWidgets.QMessageBox.information(
                self,
                "Inertia Calculated",
                f"Inertia tensor successfully calculated!\n\n"
                f"Mass: {mass:.6f} kg\n"
                f"Volume (VTK): {vtk_volume:.9f} m³\n"
                f"Volume (Trimesh): {mesh.volume:.9f} m³\n"
                f"Density: {density:.6f} kg/m³\n"
                f"Watertight: {'Yes' if mesh.is_watertight else 'No'}\n"
                f"{repair_msg}"
                f"\nCenter of mass (from UI): [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f}]\n\n"
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
        # Use shared triangle-based method from urdf_kitchen_utils
        return calculate_inertia_tensor(poly_data, mass, center_of_mass, is_mirrored)

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
        group_box = QtWidgets.QGroupBox("Default Joint Settings")
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

        # Friction設定
        group_layout.addWidget(QtWidgets.QLabel("Default Friction:"), 2, 0)
        self.friction_input = QtWidgets.QLineEdit()
        self.friction_input.setValidator(QDoubleValidator(0.0, 1000.0, 5))
        self.friction_input.setText(str(self.graph.default_joint_friction))
        group_layout.addWidget(self.friction_input, 2, 1)

        # ActuationLag設定
        group_layout.addWidget(QtWidgets.QLabel("Default ActuationLag:"), 3, 0)
        self.actuation_lag_input = QtWidgets.QLineEdit()
        self.actuation_lag_input.setValidator(QDoubleValidator(0.0, 1000.0, 5))
        self.actuation_lag_input.setText(str(self.graph.default_joint_actuation_lag))
        group_layout.addWidget(self.actuation_lag_input, 3, 1)

        # Damping設定
        group_layout.addWidget(QtWidgets.QLabel("Default Damping:"), 4, 0)
        self.damping_input = QtWidgets.QLineEdit()
        self.damping_input.setValidator(QDoubleValidator(0.0, 1000.0, 5))
        self.damping_input.setText(str(self.graph.default_joint_damping))
        group_layout.addWidget(self.damping_input, 4, 1)

        # Stiffness設定
        group_layout.addWidget(QtWidgets.QLabel("Default Stiffness:"), 5, 0)
        self.stiffness_input = QtWidgets.QLineEdit()
        self.stiffness_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.stiffness_input.setText(str(self.graph.default_joint_stiffness))
        group_layout.addWidget(self.stiffness_input, 5, 1)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        # Mesh Highlight セクション
        highlight_group = QtWidgets.QGroupBox("Mesh Highlight")
        highlight_layout = QtWidgets.QHBoxLayout()

        highlight_layout.addWidget(QtWidgets.QLabel("Highlight Color:"))

        # カラーボックス（色を表示）
        self.highlight_color_box = QtWidgets.QLabel()
        self.highlight_color_box.setFixedSize(60, 30)
        self.highlight_color_box.setStyleSheet(
            f"background-color: {self.graph.highlight_color}; border: 1px solid black;"
        )
        highlight_layout.addWidget(self.highlight_color_box)

        # Pickボタン
        pick_button = QtWidgets.QPushButton("Pick")
        pick_button.clicked.connect(self.pick_highlight_color)
        highlight_layout.addWidget(pick_button)

        highlight_layout.addStretch()
        highlight_group.setLayout(highlight_layout)
        layout.addWidget(highlight_group)

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

    def pick_highlight_color(self):
        """カラーピッカーを開いてハイライトカラーを選択"""
        # 現在の色を取得
        current_color = QtGui.QColor(self.graph.highlight_color)

        # カスタムカラーダイアログを使用
        dialog = CustomColorDialog(current_color, self)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            color = dialog.currentColor()
            if color.isValid():
                # 色を#RRGGBB形式で保存
                hex_color = color.name()
                self.graph.highlight_color = hex_color
                # カラーボックスを更新
                self.highlight_color_box.setStyleSheet(
                    f"background-color: {hex_color}; border: 1px solid black;"
                )

    def accept_settings(self):
        """設定を適用"""
        try:
            effort = float(self.effort_input.text())
            velocity = float(self.velocity_input.text())
            friction = float(self.friction_input.text())
            actuation_lag = float(self.actuation_lag_input.text())
            damping = float(self.damping_input.text())
            stiffness = float(self.stiffness_input.text())

            self.graph.default_joint_effort = effort
            self.graph.default_joint_velocity = velocity
            self.graph.default_joint_friction = friction
            self.graph.default_joint_actuation_lag = actuation_lag
            self.graph.default_joint_damping = damping
            self.graph.default_joint_stiffness = stiffness
            # highlight_colorは既にpick_highlight_colorで更新済み

            self.accept()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values."
            )


class CircularProgressBar(QtWidgets.QWidget):
    """円形プログレスバー (100から0へカウントダウン)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 100  # 初期値を100に変更
        self.setFixedSize(100, 100)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QColor, QConicalGradient, QFont
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background circle
        pen = QPen(QColor(50, 50, 50, 180))
        pen.setWidth(8)
        painter.setPen(pen)
        painter.drawEllipse(10, 10, 80, 80)

        # Progress arc (light blue) - 残り処理量を表示
        gradient = QConicalGradient(50, 50, 90)
        gradient.setColorAt(0, QColor(100, 180, 255, 200))
        gradient.setColorAt(1, QColor(150, 220, 255, 200))

        pen = QPen(gradient, 8)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        span_angle = int(self.value * 360 / 100 * 16)
        painter.drawArc(10, 10, 80, 80, 90 * 16, -span_angle)

        # 残り処理パーセントを中央に表示
        painter.setPen(QColor(200, 200, 200, 220))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 10, 80, 80, QtCore.Qt.AlignmentFlag.AlignCenter, f"{int(self.value)}%")


class STLViewerWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(STLViewerWidget, self).__init__(parent)
        self.stl_actors = {}
        self.transforms = {}
        self.base_connected_node = None
        self.text_actors = []
        self.inertial_origin_actors = {}  # Inertial Origin表示用のアクター

        layout = QtWidgets.QVBoxLayout(self)

        # Progress bar (initially hidden)
        self.progress_bar = CircularProgressBar(self)
        self.progress_bar.hide()

        # Use QLabel instead of QVTKRenderWindowInteractor for M4 Mac compatibility
        self.vtk_display = QLabel(self)
        self.vtk_display.setMinimumSize(100, 1)  # 最小幅100px、高さは1pxまで縮小可能
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
        # レンダラーのビューポートをウィンドウ全体に設定（デフォルトは0,0,1,1だが明示）
        self.renderer.SetViewport(0, 0, 1, 1)
        self.render_window.AddRenderer(self.renderer)

        # No more interactor needed - using offscreen rendering
        self.iren = None

        # Initialize offscreen renderer utility
        self.offscreen_renderer = OffscreenRenderer(self.render_window, self.renderer)

        # Initialize camera controller
        self.camera_controller = CameraController(self.renderer, origin=[0, 0, 0])

        # Initialize mouse drag state
        self.mouse_drag = MouseDragState(self.vtk_display)

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

        # Reset Angleボタン（右側）
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

        # ハイライト関連の変数
        self.highlighted_node = None
        self.original_color = None
        self.highlight_timer = QTimer(self)
        self.highlight_timer.timeout.connect(self._toggle_highlight)
        self.highlight_state = False  # 点滅の状態

        # Delay initial render to avoid blocking
        QTimer.singleShot(100, self.render_to_image)

    def show_progress(self, show=True):
        """Show or hide the progress bar"""
        if show:
            # Center the progress bar
            x = (self.vtk_display.width() - self.progress_bar.width()) // 2
            y = (self.vtk_display.height() - self.progress_bar.height()) // 2
            self.progress_bar.move(x, y)
            self.progress_bar.raise_()
            self.progress_bar.show()
        else:
            self.progress_bar.hide()

    def resizeEvent(self, event):
        """Reposition progress bar on resize"""
        super().resizeEvent(event)
        if self.progress_bar.isVisible():
            x = (self.vtk_display.width() - self.progress_bar.width()) // 2
            y = (self.vtk_display.height() - self.progress_bar.height()) // 2
            self.progress_bar.move(x, y)

    def render_to_image(self):
        """Render VTK scene offscreen and display as image in QLabel"""
        try:
            # 重要: VTKウィンドウサイズをQLabelの実際のサイズに合わせる
            widget_size = self.vtk_display.size()
            widget_width = max(widget_size.width(), 10)  # 最小サイズ10
            widget_height = max(widget_size.height(), 10)

            # VTKウィンドウサイズを更新
            current_size = self.render_window.GetSize()
            if current_size[0] != widget_width or current_size[1] != widget_height:
                self.render_window.SetSize(widget_width, widget_height)
                # ウィンドウサイズ変更時はクリッピング範囲も更新
                self.renderer.ResetCameraClippingRange()

            # デバッグ: カメラとウィンドウの状態を表示（最初の数回のみ）
            if not hasattr(self, '_render_count'):
                self._render_count = 0

            if self._render_count < 3:
                camera = self.renderer.GetActiveCamera()
                win_size = self.render_window.GetSize()
                print(f"\n=== Render #{self._render_count} ===")
                print(f"VTK Window Size: {win_size}")
                print(f"QLabel Widget Size: {widget_width}x{widget_height}")
                print(f"Camera Position: {camera.GetPosition()}")
                print(f"Camera FocalPoint: {camera.GetFocalPoint()}")
                print(f"Camera WindowCenter: {camera.GetWindowCenter()}")
                print(f"Camera ParallelScale: {camera.GetParallelScale()}")
                print(f"Renderer Viewport: {self.renderer.GetViewport()}")
                self._render_count += 1

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
                    # Shift+左ボタンまたはホイールボタンでパンモード
                    if (event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier) or \
                       event.button() == Qt.MiddleButton:
                        # 他のドラッグ状態をクリア
                        self.mouse_drag.end_left_drag()
                        self.mouse_drag.start_middle_drag(event.pos())
                        return True
                    elif event.button() == Qt.LeftButton:
                        # 他のドラッグ状態をクリア
                        self.mouse_drag.end_middle_drag()
                        self.mouse_drag.start_left_drag(event.pos())
                        return True

            elif event.type() == QEvent.MouseButtonRelease:
                if isinstance(event, QMouseEvent):
                    if event.button() == Qt.LeftButton:
                        self.mouse_drag.end_left_drag()
                        return True
                    elif event.button() == Qt.MiddleButton:
                        self.mouse_drag.end_middle_drag()
                        return True

            elif event.type() == QEvent.MouseMove:
                if isinstance(event, QMouseEvent):
                    if self.mouse_drag.left_button_pressed or self.mouse_drag.middle_button_pressed:
                        dx, dy = self.mouse_drag.update_pos(event.pos())

                        if self.mouse_drag.middle_button_pressed:
                            # Use CameraController for panning
                            self.camera_controller.pan(dx, dy)
                        else:
                            # Use CameraController for rotation
                            self.camera_controller.rotate_azimuth_elevation(dx, dy)

                        self.render_to_image()
                        return True

            elif event.type() == QEvent.Wheel:
                delta_y = event.angleDelta().y()
                # Use CameraController for zooming
                self.camera_controller.zoom(delta_y)
                self.render_to_image()
                return True

        return super().eventFilter(obj, event)

    def highlight_node(self, node):
        """ノードを選択したときにハイライト表示し、点滅させる"""
        # 既存のハイライトをクリア
        self.clear_highlight()

        if node not in self.stl_actors:
            return

        actor = self.stl_actors[node]

        # 元の色を保存
        self.original_color = actor.GetProperty().GetColor()
        self.highlighted_node = node

        # 設定されたハイライトカラーを取得（hex -> RGB）
        if hasattr(self, 'graph') and hasattr(self.graph, 'highlight_color'):
            color = QtGui.QColor(self.graph.highlight_color)
            highlight_rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
        else:
            # デフォルトのライトブルー
            highlight_rgb = (0.5, 0.8, 1.0)

        # ハイライトカラーに設定
        actor.GetProperty().SetColor(*highlight_rgb)
        self.render_to_image()

        # 点滅タイマーを開始（500msごと）
        self.highlight_state = True
        self.highlight_timer.start(500)

    def _toggle_highlight(self):
        """ハイライトの点滅を切り替え"""
        if not self.highlighted_node or self.highlighted_node not in self.stl_actors:
            self.highlight_timer.stop()
            return

        actor = self.stl_actors[self.highlighted_node]

        if self.highlight_state:
            # 元の色に戻す
            actor.GetProperty().SetColor(*self.original_color)
        else:
            # 設定されたハイライトカラーを取得（hex -> RGB）
            if hasattr(self, 'graph') and hasattr(self.graph, 'highlight_color'):
                color = QtGui.QColor(self.graph.highlight_color)
                highlight_rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            else:
                # デフォルトのライトブルー
                highlight_rgb = (0.5, 0.8, 1.0)
            actor.GetProperty().SetColor(*highlight_rgb)

        self.highlight_state = not self.highlight_state
        self.render_to_image()

    def clear_highlight(self):
        """ハイライトをクリア"""
        self.highlight_timer.stop()

        if self.highlighted_node and self.highlighted_node in self.stl_actors:
            actor = self.stl_actors[self.highlighted_node]
            if self.original_color:
                actor.GetProperty().SetColor(*self.original_color)
            self.render_to_image()

        self.highlighted_node = None
        self.original_color = None
        self.highlight_state = False

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
        """Front view（正面図）- 原点を中心に表示"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # デフォルト値

        # 原点(0,0,0)を中心に設定
        distance = max(diagonal, 1.0)  # 最低距離を確保
        parallel_scale = max(diagonal * 0.7, 0.1)  # 最小スケールを確保

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[distance, 0, 0],  # X軸方向から見る
            view_up=[0, 0, 1],
            focal_point=[0, 0, 0],  # 原点を注視
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Front view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera_side(self):
        """Side view（側面図）- 原点を中心に表示"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # デフォルト値

        # 原点(0,0,0)を中心に設定
        distance = max(diagonal, 1.0)  # 最低距離を確保
        parallel_scale = max(diagonal * 0.7, 0.1)  # 最小スケールを確保

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[0, distance, 0],  # Y軸方向から見る
            view_up=[0, 0, 1],
            focal_point=[0, 0, 0],  # 原点を注視
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Side view (ParallelScale: {parallel_scale:.3f})")

    def reset_camera_top(self):
        """Top view（上面図）- 原点を中心に表示"""
        center, diagonal = self._get_scene_bounds_and_center()
        if center is None:
            diagonal = 1.0  # デフォルト値

        # 原点(0,0,0)を中心に設定
        distance = max(diagonal, 1.0)  # 最低距離を確保
        parallel_scale = max(diagonal * 0.7, 0.1)  # 最小スケールを確保

        # Use CameraController to reset camera
        self.camera_controller.setup_parallel_camera(
            position=[0, 0, distance],  # Z軸方向から見る
            view_up=[0, 1, 0],  # Top viewではY軸が上
            focal_point=[0, 0, 0],  # 原点を注視
            parallel_scale=parallel_scale
        )

        self.render_to_image()
        print(f"Camera reset to Top view (ParallelScale: {parallel_scale:.3f})")

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
        """
        メッシュファイル（.stl, .obj, .dae）を読み込んでVTK PolyDataと色情報を返す

        Returns:
            tuple: (polydata, color) - colorはRGBA配列またはNone
        """
        try:
            # Use common utility function to load mesh
            poly_data, volume, extracted_color = load_mesh_to_polydata(file_path)

            print(f"Mesh file loaded: {file_path}")
            print(f"PolyData: {poly_data.GetNumberOfPoints()} points, {poly_data.GetNumberOfCells()} cells")

            if extracted_color:
                print(f"Color extracted from file: RGB({extracted_color[0]:.3f}, {extracted_color[1]:.3f}, {extracted_color[2]:.3f})")

            # Convert RGBA to RGB for Assembler (Assembler uses RGB format)
            if extracted_color and len(extracted_color) >= 3:
                color_rgb = extracted_color[:3]
            else:
                color_rgb = None

            return poly_data, color_rgb

        except Exception as e:
            print(f"Error loading mesh file: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def load_stl_for_node(self, node):
        """ノード用のメッシュファイル（.stl, .obj, .dae）を読み込む（色の適用を含む）"""
        # base_linkでblank_linkがTrueの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            if not hasattr(node, 'blank_link') or node.blank_link:
                return

        if node.stl_file:
            # ファイルサイズを取得して処理の重み付けを計算
            try:
                file_size = os.path.getsize(node.stl_file)
                # ファイルサイズに基づいて読み込み工程の重みを調整
                # 小さいファイル(< 1MB): 読み込み50%, その他50%
                # 中程度(1-10MB): 読み込み70%, その他30%
                # 大きいファイル(> 10MB): 読み込み85%, その他15%
                file_size_mb = file_size / (1024 * 1024)
                if file_size_mb < 1:
                    load_weight = 50
                elif file_size_mb < 10:
                    load_weight = 70
                else:
                    load_weight = 85
            except:
                load_weight = 60  # デフォルト値
                file_size_mb = 0

            # プログレスバー表示開始 (100%から開始)
            self.show_progress(True)
            self.progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()

            # ファイル読み込み開始
            remaining = 100 - (load_weight * 0.3)  # 読み込み開始で30%消費
            self.progress_bar.setValue(int(remaining))
            QtWidgets.QApplication.processEvents()

            # メッシュファイルを読み込み（色情報も取得）
            polydata, extracted_color = self.load_mesh_file(node.stl_file)

            remaining = 100 - load_weight  # 読み込み完了
            self.progress_bar.setValue(int(remaining))
            QtWidgets.QApplication.processEvents()

            if polydata is None:
                print(f"ERROR: Failed to load mesh: {node.stl_file}")
                self.show_progress(False)
                return

            # .daeファイルから色が抽出された場合、ノードにまだ色が設定されていなければ適用
            if extracted_color is not None:
                if not hasattr(node, 'node_color') or node.node_color is None or node.node_color == DEFAULT_COLOR_WHITE:
                    node.node_color = extracted_color

            # メッシュのスケールをポリデータに適用（URDF左右対称対応）
            # ジョイント位置には影響させず、メッシュの形状のみをスケール
            if hasattr(node, 'mesh_scale'):
                mesh_scale = node.mesh_scale
                if mesh_scale != [1.0, 1.0, 1.0]:
                    # vtkTransformを使ってスケール変換を作成
                    scale_transform = vtk.vtkTransform()
                    scale_transform.Scale(mesh_scale[0], mesh_scale[1], mesh_scale[2])

                    # vtkTransformPolyDataFilterでポリデータにスケールを適用
                    transform_filter = vtk.vtkTransformPolyDataFilter()
                    transform_filter.SetInputData(polydata)
                    transform_filter.SetTransform(scale_transform)
                    transform_filter.Update()

                    polydata = transform_filter.GetOutput()
                    print(f"Applied mesh scale {mesh_scale} to polydata for node '{node.name()}'")

            # マッパーとアクター作成
            processing_weight = (100 - load_weight) * 0.6
            remaining = 100 - load_weight - processing_weight
            self.progress_bar.setValue(int(remaining))
            QtWidgets.QApplication.processEvents()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # ジョイント位置・回転用のtransform（スケールは含まない）
            transform = vtk.vtkTransform()
            transform.Identity()
            actor.SetUserTransform(transform)

            # レンダラーに追加
            render_weight = (100 - load_weight - processing_weight) * 0.5
            remaining = 100 - load_weight - processing_weight - render_weight
            self.progress_bar.setValue(int(remaining))
            QtWidgets.QApplication.processEvents()

            if node in self.stl_actors:
                self.renderer.RemoveActor(self.stl_actors[node])

            self.stl_actors[node] = actor
            self.transforms[node] = transform
            self.renderer.AddActor(actor)

            # ノードの色情報を適用
            self.apply_color_to_node(node)

            # Hide Mesh状態を確認して適用
            if hasattr(node, 'hide_mesh') and node.hide_mesh:
                actor.SetVisibility(False)
                print(f"Applied hide_mesh on load: {node.name()} - mesh hidden")

            # 最終レンダリング
            remaining = 5  # 最終工程
            self.progress_bar.setValue(int(remaining))
            QtWidgets.QApplication.processEvents()

            self.reset_camera()
            self.render_to_image()

            # 完了 (0%に到達)
            self.progress_bar.setValue(0)
            QtWidgets.QApplication.processEvents()

            # プログレスバー非表示
            QTimer.singleShot(200, lambda: self.show_progress(False))

            # ログ出力にファイルサイズ情報を追加
            print(f"Loaded: {node.stl_file} ({file_size_mb:.2f} MB)")

    def apply_color_to_node(self, node):
        """ノードのSTLモデルに色を適用"""
        if node in self.stl_actors:
            # デフォルトの色を設定（色情報がない場合）
            if not hasattr(node, 'node_color') or node.node_color is None:
                node.node_color = DEFAULT_COLOR_WHITE.copy()  # 白色をデフォルトに

            # 色の適用
            actor = self.stl_actors[node]
            actor.GetProperty().SetColor(*node.node_color)
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
        """カメラの初期設定 - 原点(0,0,0)を中心に表示"""
        camera = self.renderer.GetActiveCamera()

        # 平行投影を有効化
        camera.ParallelProjectionOn()

        # 重要: カメラパラメータは正しい順序で設定する必要がある
        # 1. 焦点（どこを見るか）を先に設定
        camera.SetFocalPoint(0, 0, 0)  # 原点を注視

        # 2. カメラ位置（どこから見るか）を設定
        camera.SetPosition(0.3, 0, 0)  # X軸方向から見る（距離0.3）

        # 3. 上方向ベクトルを設定
        camera.SetViewUp(0, 0, 1)  # Z軸が上

        # 4. WindowCenter設定 - 投影の中心をビューポートの中央に明示的に配置
        # (0, 0)がビューポートの中心、デフォルト値だが明示的に設定
        camera.SetWindowCenter(0.0, 0.0)

        # 5. ParallelScale設定（ビューポートの高さの半分、ワールド座標単位）
        # 座標軸が0.1単位なので、0.15に設定してパディングを確保
        camera.SetParallelScale(0.15)

        # 6. クリッピング範囲をリセット（全てが見えるように）
        self.renderer.ResetCameraClippingRange()

        print(f"Camera setup: Position={camera.GetPosition()}, FocalPoint={camera.GetFocalPoint()}, WindowCenter={camera.GetWindowCenter()}")

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
        self.default_joint_friction = DEFAULT_JOINT_FRICTION
        self.default_joint_actuation_lag = DEFAULT_JOINT_ACTUATION_LAG
        self.default_joint_damping = DEFAULT_JOINT_DAMPING
        self.default_joint_stiffness = DEFAULT_JOINT_STIFFNESS

        # ハイライトカラー設定
        self.highlight_color = DEFAULT_HIGHLIGHT_COLOR

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

        # 選択状態監視用のタイマーを設定
        self.last_selected_node = None
        self.selection_monitor_timer = QTimer()
        self.selection_monitor_timer.timeout.connect(self._check_selection_change)
        self.selection_monitor_timer.start(100)  # 100msごとにチェック

    def _check_selection_change(self):
        """選択状態の変化を監視"""
        selected_nodes = self.selected_nodes()

        if selected_nodes:
            # 最初に選択されたノードを取得
            current_selected = selected_nodes[0]

            # 前回と異なるノードが選択された場合
            if current_selected != self.last_selected_node:
                self.last_selected_node = current_selected
                if self.stl_viewer:
                    self.stl_viewer.highlight_node(current_selected)
        else:
            # 何も選択されていない場合
            if self.last_selected_node is not None:
                self.last_selected_node = None
                if self.stl_viewer:
                    self.stl_viewer.clear_highlight()

    def custom_mouse_press(self, event):
        """カスタムマウスプレスイベントハンドラ"""
        try:






            # 中ボタンでパン操作を開始（カスタム実装）
            if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                print(">>> Starting pan operation (Middle Button Drag) - using custom panning")
                self._is_panning = True
                self._pan_start = event.position().toPoint()
                self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                # オリジナルハンドラーを呼び出してQt状態を初期化
                self._original_handlers['press'](event)
                return

            # 左ボタンの処理
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # Option (Alt) + 左ボタンでパン操作を開始（Qt ScrollHandDrag使用）
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    print(">>> Starting pan operation (Option+Drag) - using Qt ScrollHandDrag")
                    self._is_panning = True
                    self._pan_start = None  # ScrollHandDragを使用する場合はNone
                    # Qt組み込みのドラッグパンモードに切り替え
                    self._viewer.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
                    self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    # 元のハンドラを呼び出してQt標準の処理を実行
                    self._original_handlers['press'](event)
                    return

                # Shift + 左ボタンでパン操作を開始（カスタムパン実装）
                # 注: NodeGraphQtはShiftを複数選択に使用しているため、競合する可能性あり
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    print(">>> Starting pan operation (Shift+Drag) - using custom panning")
                    self._is_panning = True
                    self._pan_start = event.position().toPoint()
                    self._viewer.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    # オリジナルハンドラーを呼び出してQt状態を初期化
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
            # パン操作中の処理
            if self._is_panning:
                # カスタムパン実装（中ボタンまたはShift+左ボタン）
                if self._pan_start is not None:
                    current_pos = event.position().toPoint()

                    # ビュー座標をシーン座標にマッピング
                    previous_scene = self._viewer.mapToScene(self._pan_start)
                    current_scene = self._viewer.mapToScene(current_pos)
                    delta = previous_scene - current_scene

                    print(f"Custom pan: delta=({delta.x()}, {delta.y()})")

                    # NodeGraphQtの内部パンメソッドを使用
                    self._viewer._set_viewer_pan(delta.x(), delta.y())

                    self._pan_start = current_pos
                    return
                else:
                    # Qt標準のScrollHandDragを使用（Option+左ボタン）
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


            print(f"Is selecting: {self._is_selecting}")
            print(f"Is panning: {self._is_panning}")

            # パン操作の終了（中ボタンまたはShift/Option+左ボタン）
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
            # Ctrl/Command+A でBase以外の全ノードを選択
            if event.key() == QtCore.Qt.Key.Key_A and (
                event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or
                event.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier  # MacのCommandキー
            ):
                print("\n=== Select All Nodes (Ctrl/Cmd+A) ===")
                # Base以外の全ノードを選択
                all_nodes = self.all_nodes()
                selected_count = 0

                for node in all_nodes:
                    # BaseLinkNodeは選択しない
                    if not isinstance(node, BaseLinkNode):
                        node.set_selected(True)
                        selected_count += 1
                    else:
                        # BaseLinkNodeは選択解除
                        node.set_selected(False)

                print(f"Selected {selected_count} nodes (excluding Base)")
                event.accept()
                return

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

    def update_node_color_by_connection(self, node):
        """ノードの入力接続状態に応じて色を更新"""
        # BaseLinkNodeは例外として常に黒
        if isinstance(node, BaseLinkNode):
            node.set_color(45, 45, 45)  # 常に黒
            return

        # 入力ポートの接続をチェック
        has_input_connection = False
        for input_port in node.input_ports():
            if input_port.connected_ports():
                has_input_connection = True
                break

        if has_input_connection:
            # 接続あり：濃い黒
            node.set_color(45, 45, 45)  # 黒
        else:
            # 接続なし：明るめのグレー
            node.set_color(74, 84, 85)  # グレー

    def update_all_node_colors(self):
        """すべてのノードの色を接続状態に応じて更新"""
        for node in self.all_nodes():
            self.update_node_color_by_connection(node)

    def on_port_connected(self, input_port, output_port):
        """ポートが接続された時の処理"""
        print(f"**Connecting port: {output_port.name()}")

        # 接続情報の出力
        parent_node = output_port.node()
        child_node = input_port.node()
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")

        try:
            # 接続されたノード（子ノード）の色を更新
            self.update_node_color_by_connection(child_node)

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

            # 切断されたノード（子ノード）の色を更新
            self.update_node_color_by_connection(child_node)

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
                    'mesh_scale': [1.0, 1.0, 1.0],  # メッシュのスケール（URDF左右対称対応）
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

                            # メッシュのscale属性を読み取る（URDF左右対称対応）
                            mesh_scale = [1.0, 1.0, 1.0]
                            scale_str = mesh_elem.get('scale', '')
                            if scale_str:
                                try:
                                    scale_values = [float(v) for v in scale_str.split()]
                                    if len(scale_values) == 3:
                                        mesh_scale = scale_values
                                        print(f"Mesh scale detected: {mesh_scale}")
                                except ValueError:
                                    print(f"Warning: Invalid scale attribute '{scale_str}', using default [1, 1, 1]")

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
                        # メッシュのスケール情報も保存（URDF左右対称対応）
                        if geometry_elem is not None and mesh_elem is not None:
                            link_data['mesh_scale'] = mesh_scale
                    else:
                        # decoration visualの場合はdecorationsリストに追加
                        if current_stl_path:
                            # STLファイル名から拡張子を除いた名前を取得
                            stl_name = os.path.splitext(os.path.basename(current_stl_path))[0]
                            decoration_data = {
                                'name': stl_name,
                                'stl_file': current_stl_path,
                                'color': current_color,
                                'mesh_scale': mesh_scale if geometry_elem is not None and mesh_elem is not None else [1.0, 1.0, 1.0]
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
                        "Mesh Files Not Found",
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
                                    "Mesh Files Found",
                                    f"Found {found_count} out of {missing_count} missing mesh file(s) in the specified directory."
                                )
                            else:
                                QtWidgets.QMessageBox.warning(
                                    self.widget,
                                    "No Mesh Files Found",
                                    f"Could not find any of the missing mesh files in the specified directory."
                                )

                            # missing_stl_filesリストを更新（見つかったものを除去）
                            missing_stl_files = [
                                item for item in missing_stl_files
                                if not links_data[item['link_name']]['stl_file']
                            ]
                else:
                    # すべてのメッシュファイルが自動検出された
                    if initial_missing_count > 0:
                        QtWidgets.QMessageBox.information(
                            self.widget,
                            "Mesh Files Found",
                            f"Automatically found all {initial_missing_count} missing mesh file(s)!"
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
                    'limit': {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 'velocity': 3.0, 'friction': 0.05},
                    'dynamics': {'damping': 0.0, 'friction': 0.0}  # デフォルト値
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
                    joint_data['limit']['friction'] = float(limit_elem.get('friction', 0.05))

                # dynamics要素を読み取る（damping, friction）
                dynamics_elem = joint_elem.find('dynamics')
                if dynamics_elem is not None:
                    joint_data['dynamics']['damping'] = float(dynamics_elem.get('damping', 0.0))
                    joint_data['dynamics']['friction'] = float(dynamics_elem.get('friction', 0.0))

                joints_data.append(joint_data)

            # ルートリンクを検出してbase_linkに接続
            print("\n=== Detecting root links ===")

            # すべてのリンクの中で、どのジョイントの子にもなっていないリンクを見つける
            child_links = set()
            for joint_data in joints_data:
                if joint_data['child']:
                    child_links.add(joint_data['child'])

            # ルートリンク = links_dataに存在するが、child_linksに存在しないリンク
            # ただし、base_linkとBaseLinkは除外
            root_links = []
            for link_name in links_data.keys():
                if link_name not in child_links and link_name not in ['base_link', 'BaseLink']:
                    root_links.append(link_name)
                    print(f"  Found root link: {link_name}")

            # ルートリンクが見つかった場合、base_linkへの接続を作成
            if root_links:
                print(f"  Connecting {len(root_links)} root link(s) to base_link")

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
                        'limit': {'lower': 0.0, 'upper': 0.0, 'effort': 0.0, 'velocity': 0.0, 'friction': 0.0}
                    }
                    joints_data.append(synthetic_joint)
                    print(f"  Created synthetic joint: base_link -> {root_link_name}")
            else:
                print("  No root links found (all links are connected via joints)")

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

                    # メッシュのスケール情報を設定（URDF左右対称対応）
                    base_node.mesh_scale = base_link_data.get('mesh_scale', [1.0, 1.0, 1.0])

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
                # メッシュのスケール情報を設定（URDF左右対称対応）
                node.mesh_scale = link_data.get('mesh_scale', [1.0, 1.0, 1.0])

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
                    # メッシュのスケール情報を設定（URDF左右対称対応）
                    deco_node.mesh_scale = decoration.get('mesh_scale', [1.0, 1.0, 1.0])

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
                child_node.joint_friction = joint_data['limit']['friction']

                # ジョイントdynamicsパラメータの設定（damping, friction）
                if 'dynamics' in joint_data:
                    child_node.damping = joint_data['dynamics']['damping']
                    child_node.friction = joint_data['dynamics']['friction']

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

            # 全てのノードのメッシュファイルを3Dビューに自動読み込み
            print("\n=== Loading mesh files to 3D viewer ===")
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
                            print(f"Loading mesh to viewer for {link_name}...")
                            self.stl_viewer.load_stl_for_node(node)
                            stl_viewer_loaded_count += 1
                        except Exception as e:
                            print(f"Error loading mesh to viewer for {link_name}: {str(e)}")
                            traceback.print_exc()

                print(f"Loaded {stl_viewer_loaded_count} mesh files to 3D viewer")
                print("=" * 40 + "\n")
            else:
                print("Warning: Mesh viewer not available")

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
            import_summary += f"Mesh files found: {stl_loaded_count}\n"
            import_summary += f"Mesh files loaded to 3D viewer: {stl_viewer_loaded_count}\n"
            if stl_missing_count > 0:
                import_summary += f"⚠ Warning: {stl_missing_count} mesh file(s) could not be found\n"

            # すべてのノードの色を接続状態に応じて更新
            self.update_all_node_colors()

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

    def _copy_dae_with_color(self, source_path, dest_path, node_color):
        """
        .daeファイルをコピーし、色情報を埋め込む

        Args:
            source_path: ソースの.daeファイルパス
            dest_path: 保存先の.daeファイルパス
            node_color: ノードの色 [r, g, b, a] (0.0-1.0)
        """
        try:
            import xml.etree.ElementTree as ET

            # .daeファイルを読み込む
            tree = ET.parse(source_path)
            root = tree.getroot()

            # COLLADA名前空間を取得
            namespace = {'collada': 'http://www.collada.org/2005/11/COLLADASchema'}

            # 名前空間を登録
            ET.register_namespace('', 'http://www.collada.org/2005/11/COLLADASchema')

            # library_effectsを検索または作成
            library_effects = root.find('collada:library_effects', namespace)
            if library_effects is None:
                library_effects = ET.SubElement(root, '{http://www.collada.org/2005/11/COLLADASchema}library_effects')

            # 既存のeffectをすべて削除して新しく作成
            for effect in list(library_effects):
                library_effects.remove(effect)

            # 新しいeffectを作成
            effect = ET.SubElement(library_effects, '{http://www.collada.org/2005/11/COLLADASchema}effect')
            effect.set('id', 'material_effect')

            profile_COMMON = ET.SubElement(effect, '{http://www.collada.org/2005/11/COLLADASchema}profile_COMMON')
            technique = ET.SubElement(profile_COMMON, '{http://www.collada.org/2005/11/COLLADASchema}technique')
            technique.set('sid', 'common')

            phong = ET.SubElement(technique, '{http://www.collada.org/2005/11/COLLADASchema}phong')

            # Diffuse color (拡散色)
            diffuse = ET.SubElement(phong, '{http://www.collada.org/2005/11/COLLADASchema}diffuse')
            color_elem = ET.SubElement(diffuse, '{http://www.collada.org/2005/11/COLLADASchema}color')
            color_elem.set('sid', 'diffuse')
            color_elem.text = f"{node_color[0]:.6f} {node_color[1]:.6f} {node_color[2]:.6f} 1.0"

            # library_materialsを検索または作成
            library_materials = root.find('collada:library_materials', namespace)
            if library_materials is None:
                library_materials = ET.SubElement(root, '{http://www.collada.org/2005/11/COLLADASchema}library_materials')

            # 既存のmaterialをすべて削除して新しく作成
            for material in list(library_materials):
                library_materials.remove(material)

            # 新しいmaterialを作成
            material = ET.SubElement(library_materials, '{http://www.collada.org/2005/11/COLLADASchema}material')
            material.set('id', 'material_id')
            material.set('name', 'material')

            instance_effect = ET.SubElement(material, '{http://www.collada.org/2005/11/COLLADASchema}instance_effect')
            instance_effect.set('url', '#material_effect')

            # library_visual_scenesのすべてのinstance_materialを更新
            library_visual_scenes = root.find('collada:library_visual_scenes', namespace)
            if library_visual_scenes is not None:
                for bind_material in library_visual_scenes.findall('.//collada:bind_material', namespace):
                    technique_common = bind_material.find('collada:technique_common', namespace)
                    if technique_common is not None:
                        # 既存のinstance_materialをすべて削除
                        for inst_mat in list(technique_common):
                            technique_common.remove(inst_mat)

                        # 新しいinstance_materialを作成
                        instance_material = ET.SubElement(technique_common, '{http://www.collada.org/2005/11/COLLADASchema}instance_material')
                        instance_material.set('symbol', 'material')
                        instance_material.set('target', '#material_id')

            # 修正した.daeファイルを保存
            tree.write(dest_path, encoding='utf-8', xml_declaration=True)
            print(f"  Applied color {node_color[:3]} to {os.path.basename(dest_path)}")

        except Exception as e:
            print(f"Warning: Failed to apply color to .dae file, copying as-is: {str(e)}")
            # エラーが発生した場合は通常のコピーにフォールバック
            shutil.copy2(source_path, dest_path)

    def import_mjcf(self):
        """MJCFファイルをインポート"""
        try:
            # MJCFファイルまたはZIPファイルを選択するダイアログ
            mjcf_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.widget,
                "Select MJCF file or ZIP archive to import",
                os.getcwd(),
                "MJCF Files (*.xml *.zip);;XML Files (*.xml);;ZIP Files (*.zip);;All Files (*)"
            )

            if not mjcf_file:
                print("MJCF import cancelled")
                return False

            print(f"Importing MJCF from: {mjcf_file}")

            # ZIPファイルの場合、展開する
            working_dir = None
            xml_file_to_load = None

            if mjcf_file.endswith('.zip'):
                import zipfile
                import tempfile

                print("Detected ZIP file, extracting...")

                # 一時ディレクトリに展開
                temp_dir = tempfile.mkdtemp(prefix='mjcf_import_')
                working_dir = temp_dir

                try:
                    with zipfile.ZipFile(mjcf_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    print(f"Extracted to: {temp_dir}")

                    # 展開されたディレクトリ内のXMLファイルを検索
                    xml_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.xml'):
                                xml_files.append(os.path.join(root, file))

                    if not xml_files:
                        QtWidgets.QMessageBox.warning(
                            self.widget,
                            "No XML Files Found",
                            "No XML files found in the ZIP archive."
                        )
                        return False

                    # XMLファイルが複数ある場合、選択させる
                    if len(xml_files) > 1:
                        # ファイル名のリストを作成
                        file_names = [os.path.relpath(f, temp_dir) for f in xml_files]

                        # 選択ダイアログを表示
                        selected_file, ok = QtWidgets.QInputDialog.getItem(
                            self.widget,
                            "Select XML File",
                            "Multiple XML files found. Please select one:",
                            file_names,
                            0,
                            False
                        )

                        if ok and selected_file:
                            xml_file_to_load = os.path.join(temp_dir, selected_file)
                        else:
                            print("XML file selection cancelled")
                            return False
                    else:
                        xml_file_to_load = xml_files[0]

                    print(f"Selected XML file: {xml_file_to_load}")
                    mjcf_file = xml_file_to_load

                except Exception as e:
                    print(f"Error extracting ZIP file: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    QtWidgets.QMessageBox.critical(
                        self.widget,
                        "ZIP Extraction Error",
                        f"Failed to extract ZIP file:\n\n{str(e)}"
                    )
                    return False
            else:
                working_dir = os.path.dirname(mjcf_file)

            # MJCFファイルをパース
            tree = ET.parse(mjcf_file)
            root = tree.getroot()

            if root.tag != 'mujoco':
                QtWidgets.QMessageBox.warning(
                    self.widget,
                    "Invalid MJCF File",
                    "Selected file is not a valid MJCF file (root element should be 'mujoco')"
                )
                return False

            # ロボット名をMJCFファイル名から取得
            robot_name = os.path.splitext(os.path.basename(mjcf_file))[0]
            self.robot_name = robot_name
            print(f"Robot name set to: {robot_name}")

            # デフォルトクラスを解析（ジョイント軸などのデフォルト値を取得）
            default_classes = {}

            def parse_defaults(default_elem, parent_class_name=None):
                """デフォルトクラスを再帰的に解析"""
                class_name = default_elem.get('class', parent_class_name)

                # 親クラスのデフォルト値を継承
                class_defaults = {}
                if parent_class_name and parent_class_name in default_classes:
                    class_defaults = default_classes[parent_class_name].copy()

                # このクラスのジョイントデフォルトを取得
                joint_elem = default_elem.find('joint')
                if joint_elem is not None:
                    axis_str = joint_elem.get('axis')
                    if axis_str:
                        class_defaults['joint_axis'] = [float(v) for v in axis_str.split()]

                # クラス名がある場合は保存
                if class_name:
                    default_classes[class_name] = class_defaults
                    print(f"Default class '{class_name}': {class_defaults}")

                # 子のデフォルトクラスを再帰的に解析
                for child_default in default_elem.findall('default'):
                    parse_defaults(child_default, class_name)

            # ルートのdefault要素を探して解析
            for default_elem in root.findall('default'):
                parse_defaults(default_elem)

            print(f"Parsed {len(default_classes)} default classes")

            # メッシュ情報を抽出（<asset>タグから）
            meshes_data = {}
            asset_elem = root.find('asset')
            if asset_elem is not None:
                mjcf_dir = os.path.dirname(mjcf_file)

                # メッシュ検索用のディレクトリリスト
                search_dirs = [
                    mjcf_dir,  # XMLと同じディレクトリ
                    os.path.join(mjcf_dir, 'assets'),
                    os.path.join(mjcf_dir, 'meshes'),
                    os.path.join(mjcf_dir, '..', 'assets'),
                    os.path.join(mjcf_dir, '..', 'meshes'),
                    working_dir,  # 作業ディレクトリ（ZIPの場合はtempディレクトリ）
                    os.path.join(working_dir, 'assets'),
                    os.path.join(working_dir, 'meshes'),
                ]

                for mesh_elem in asset_elem.findall('mesh'):
                    mesh_name = mesh_elem.get('name')
                    mesh_file = mesh_elem.get('file', '')

                    # mesh_nameがNoneの場合、ファイル名から推測
                    if not mesh_name and mesh_file:
                        # ファイル名から拡張子を除いた部分を名前として使用
                        mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
                        print(f"Mesh name not specified, derived from file: {mesh_name}")

                    # メッシュファイルパスを解決
                    if mesh_file:
                        mesh_path = None
                        mesh_basename = os.path.basename(mesh_file)

                        # 方法1: 元のパスをそのまま試す
                        candidate = os.path.join(mjcf_dir, mesh_file)
                        if os.path.exists(candidate):
                            mesh_path = candidate
                            print(f"Found mesh (method 1): {mesh_name} -> {mesh_path}")

                        # 方法2: 各検索ディレクトリで元のパス構造を試す
                        if not mesh_path:
                            for search_dir in search_dirs:
                                if not search_dir or not os.path.exists(search_dir):
                                    continue
                                candidate = os.path.join(search_dir, mesh_file)
                                if os.path.exists(candidate):
                                    mesh_path = candidate
                                    print(f"Found mesh (method 2): {mesh_name} -> {mesh_path}")
                                    break

                        # 方法3: basenameだけで各検索ディレクトリを再帰的に検索
                        if not mesh_path:
                            for search_dir in search_dirs:
                                if not search_dir or not os.path.exists(search_dir):
                                    continue
                                for root_dir, dirs, files in os.walk(search_dir):
                                    if mesh_basename in files:
                                        mesh_path = os.path.join(root_dir, mesh_basename)
                                        print(f"Found mesh (method 3): {mesh_name} -> {mesh_path}")
                                        break
                                if mesh_path:
                                    break

                        if mesh_path:
                            meshes_data[mesh_name] = mesh_path
                        else:
                            print(f"Warning: Mesh file not found: {mesh_file}")
                            print(f"  Searched in: {mjcf_dir}")
                            print(f"  Basename: {mesh_basename}")
                            meshes_data[mesh_name] = None

            # ボディ（リンク）とジョイント情報を抽出
            bodies_data = {}
            joints_data = []

            # worldbodyを取得
            worldbody = root.find('worldbody')
            if worldbody is None:
                QtWidgets.QMessageBox.warning(
                    self.widget,
                    "Invalid MJCF File",
                    "MJCF file does not contain 'worldbody' element"
                )
                return False

            # クォータニオンからRPY（roll, pitch, yaw）への変換関数
            def quat_to_rpy(quat):
                """
                クォータニオン (w, x, y, z) をRPY (roll, pitch, yaw) に変換
                MuJoCoのクォータニオンは (w, x, y, z) の順序
                返り値はラジアン単位の [roll, pitch, yaw]
                """
                import math

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

            # 再帰的にボディを解析
            def parse_body(body_elem, parent_name=None, level=0):
                """ボディ要素を再帰的に解析"""
                body_name = body_elem.get('name', f'body_{len(bodies_data)}')

                body_data = {
                    'name': body_name,
                    'parent': parent_name,
                    'mass': 0.0,
                    'inertia': {'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0},
                    'inertial_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                    'stl_file': None,  # 最初のメッシュファイル（後方互換性のため）
                    'visuals': [],  # 複数のvisual情報を格納
                    'color': [1.0, 1.0, 1.0],
                    'pos': [0.0, 0.0, 0.0],
                    'quat': [1.0, 0.0, 0.0, 0.0],
                    'rpy': [0.0, 0.0, 0.0]  # クォータニオンから変換されたRPY
                }

                # 位置情報を取得
                pos_str = body_elem.get('pos', '0 0 0')
                body_data['pos'] = [float(v) for v in pos_str.split()]

                # 四元数を取得してRPYに変換
                quat_str = body_elem.get('quat')
                if quat_str:
                    body_data['quat'] = [float(v) for v in quat_str.split()]
                    # クォータニオンをRPYに変換
                    body_data['rpy'] = quat_to_rpy(body_data['quat'])
                    print(f"{'  ' * level}  Body quat: {body_data['quat']} -> rpy: {body_data['rpy']}")

                # eulerタグもチェック（MJCFではeulerで直接指定されることもある）
                euler_str = body_elem.get('euler')
                if euler_str:
                    import math
                    euler_degrees = [float(v) for v in euler_str.split()]
                    # MJCFのeulerは度数なのでラジアンに変換
                    body_data['rpy'] = [math.radians(e) for e in euler_degrees]
                    print(f"{'  ' * level}  Body euler (deg): {euler_degrees} -> rpy (rad): {body_data['rpy']}")

                # Inertial情報を取得
                inertial_elem = body_elem.find('inertial')
                if inertial_elem is not None:
                    mass_str = inertial_elem.get('mass')
                    if mass_str:
                        body_data['mass'] = float(mass_str)

                    # Inertial position
                    inertial_pos = inertial_elem.get('pos', '0 0 0')
                    body_data['inertial_origin']['xyz'] = [float(v) for v in inertial_pos.split()]

                    # Inertia matrix (diaginertia for diagonal elements)
                    diaginertia_str = inertial_elem.get('diaginertia')
                    if diaginertia_str:
                        diag = [float(v) for v in diaginertia_str.split()]
                        if len(diag) >= 3:
                            body_data['inertia']['ixx'] = diag[0]
                            body_data['inertia']['iyy'] = diag[1]
                            body_data['inertia']['izz'] = diag[2]

                # Geometry情報を取得（すべてのgeomをチェック）
                geom_elems = body_elem.findall('geom')
                if geom_elems:
                    print(f"{'  ' * level}  Found {len(geom_elems)} geom element(s)")

                    # すべてのメッシュを持つgeomを収集
                    for idx, geom_elem in enumerate(geom_elems):
                        mesh_name = geom_elem.get('mesh')
                        print(f"{'  ' * level}    Geom[{idx}] - mesh attribute: {mesh_name}")

                        if mesh_name:
                            if mesh_name in meshes_data:
                                mesh_path = meshes_data[mesh_name]

                                # 位置とクォータニオンを取得
                                geom_pos_str = geom_elem.get('pos', '0 0 0')
                                geom_pos = [float(v) for v in geom_pos_str.split()]

                                geom_quat_str = geom_elem.get('quat')
                                geom_quat = [1.0, 0.0, 0.0, 0.0]  # デフォルト
                                if geom_quat_str:
                                    geom_quat = [float(v) for v in geom_quat_str.split()]

                                # 色情報を取得
                                geom_color = [1.0, 1.0, 1.0]
                                rgba_str = geom_elem.get('rgba')
                                if rgba_str:
                                    rgba = [float(v) for v in rgba_str.split()]
                                    geom_color = rgba[:3]  # RGBのみ

                                visual_data = {
                                    'mesh': mesh_path,
                                    'pos': geom_pos,
                                    'quat': geom_quat,
                                    'color': geom_color
                                }
                                body_data['visuals'].append(visual_data)

                                # 最初のメッシュは後方互換性のためstl_fileにも設定
                                if idx == 0:
                                    body_data['stl_file'] = mesh_path
                                    body_data['color'] = geom_color

                                print(f"{'  ' * level}    ✓ Mesh assigned: {mesh_path}")
                            else:
                                print(f"{'  ' * level}    ✗ Mesh '{mesh_name}' not found in meshes_data")
                                print(f"{'  ' * level}      Available meshes: {list(meshes_data.keys())[:5]}...")
                        else:
                            print(f"{'  ' * level}    No mesh attribute (might be primitive shape)")
                else:
                    print(f"{'  ' * level}  No geom element found")

                bodies_data[body_name] = body_data
                print(f"{'  ' * level}Parsed body: {body_name} (parent: {parent_name}, has_mesh: {body_data['stl_file'] is not None})")

                # ジョイント情報を抽出（このボディ内のジョイント）
                for joint_elem in body_elem.findall('joint'):
                    joint_name = joint_elem.get('name', f'joint_{len(joints_data)}')
                    joint_type = joint_elem.get('type', 'hinge')

                    # MJCFのジョイントタイプをURDF形式に変換
                    urdf_joint_type = 'revolute'
                    if joint_type == 'slide':
                        urdf_joint_type = 'prismatic'
                    elif joint_type == 'ball':
                        urdf_joint_type = 'spherical'
                    elif joint_type == 'free':
                        urdf_joint_type = 'floating'

                    joint_data = {
                        'name': joint_name,
                        'type': urdf_joint_type,
                        'parent': parent_name,
                        'child': body_name,
                        'origin_xyz': body_data['pos'],
                        'origin_rpy': body_data['rpy'],  # クォータニオンから変換されたRPY
                        'axis': [1.0, 0.0, 0.0],
                        'limit': {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 'velocity': 3.0},
                        'dynamics': {'damping': 0.0, 'friction': 0.0},  # MJCF対応
                        'stiffness': 0.0,  # MJCF joint.stiffness
                        'actuation_lag': 0.0  # MJCF actuator.dynprm
                    }

                    # 軸情報（class属性から継承またはaxis属性から直接取得）
                    axis_str = joint_elem.get('axis')
                    if axis_str:
                        # axis属性が明示的に指定されている場合
                        joint_data['axis'] = [float(v) for v in axis_str.split()]
                    else:
                        # class属性からデフォルト値を取得
                        joint_class = joint_elem.get('class')
                        if joint_class and joint_class in default_classes:
                            if 'joint_axis' in default_classes[joint_class]:
                                joint_data['axis'] = default_classes[joint_class]['joint_axis']
                                print(f"{'  ' * level}    Inherited axis from class '{joint_class}': {joint_data['axis']}")
                            else:
                                joint_data['axis'] = [1.0, 0.0, 0.0]  # デフォルト
                        else:
                            joint_data['axis'] = [1.0, 0.0, 0.0]  # デフォルト

                    # リミット情報
                    range_str = joint_elem.get('range')
                    if range_str:
                        range_vals = [float(v) for v in range_str.split()]
                        if len(range_vals) >= 2:
                            joint_data['limit']['lower'] = range_vals[0]
                            joint_data['limit']['upper'] = range_vals[1]

                    # MJCF dynamics情報（damping, frictionloss, stiffness）
                    damping_str = joint_elem.get('damping')
                    if damping_str:
                        joint_data['dynamics']['damping'] = float(damping_str)

                    frictionloss_str = joint_elem.get('frictionloss')
                    if frictionloss_str:
                        joint_data['dynamics']['friction'] = float(frictionloss_str)

                    stiffness_str = joint_elem.get('stiffness')
                    if stiffness_str:
                        joint_data['stiffness'] = float(stiffness_str)

                    joints_data.append(joint_data)
                    print(f"{'  ' * level}  Joint: {joint_name} ({urdf_joint_type})")

                # 子ボディを再帰的に処理
                for child_body in body_elem.findall('body'):
                    parse_body(child_body, body_name, level + 1)

            # worldbody直下のすべてのbodyを解析
            worldbody_bodies = worldbody.findall('body')
            root_body_names = []

            if worldbody_bodies:
                print("\n=== Processing worldbody bodies ===")
                for body_elem in worldbody_bodies:
                    body_name = body_elem.get('name', f'body_{len(bodies_data)}')
                    root_body_names.append(body_name)
                    # 親なしで解析（後でbase_linkに接続）
                    parse_body(body_elem, None, 0)
                    print(f"  Root body: {body_name}")

            # worldbody直下のbodyをbase_linkに接続する合成ジョイントを作成
            print("\n=== Creating synthetic joints to connect root bodies to base_link ===")
            for root_body_name in root_body_names:
                # このbodyのfreejointを見つけて削除（freejointはbase_linkとの接続には使用しない）
                joints_to_remove = []
                for i, joint_data in enumerate(joints_data):
                    if joint_data['child'] == root_body_name and joint_data['type'] == 'floating':
                        joints_to_remove.append(i)
                        print(f"  Removing freejoint for {root_body_name}")

                # 逆順で削除（インデックスがずれないように）
                for i in reversed(joints_to_remove):
                    del joints_data[i]

                # base_linkからルートbodyへの固定ジョイントを作成
                if root_body_name in bodies_data:
                    body_data = bodies_data[root_body_name]
                    synthetic_joint = {
                        'name': f'base_to_{root_body_name}',
                        'type': 'fixed',
                        'parent': 'base_link',
                        'child': root_body_name,
                        'origin_xyz': body_data['pos'],
                        'origin_rpy': body_data['rpy'],
                        'axis': [1.0, 0.0, 0.0],
                        'limit': {'lower': 0.0, 'upper': 0.0, 'effort': 0.0, 'velocity': 0.0},
                        'dynamics': {'damping': 0.0, 'friction': 0.0},
                        'stiffness': 0.0,
                        'actuation_lag': 0.0
                    }
                    joints_data.append(synthetic_joint)
                    print(f"  Created synthetic joint: base_link -> {root_body_name}")

            # Actuator情報を処理（forcerange → effort, dynprm → actuation_lag）
            print("\n=== Processing Actuators ===")
            actuator_elem = root.find('actuator')
            if actuator_elem is not None:
                for motor_elem in actuator_elem.findall('motor'):
                    joint_name = motor_elem.get('joint')
                    if not joint_name:
                        continue

                    # 対応するjointを見つける
                    for joint_data in joints_data:
                        if joint_data['name'] == joint_name:
                            # forcerange → effort
                            forcerange_str = motor_elem.get('forcerange')
                            if forcerange_str:
                                range_vals = [float(v) for v in forcerange_str.split()]
                                if len(range_vals) >= 2:
                                    # forcerangeは[-τmax, +τmax]なので、絶対値の最大値をeffortとする
                                    joint_data['limit']['effort'] = max(abs(range_vals[0]), abs(range_vals[1]))
                                    print(f"  Motor '{joint_name}': effort={joint_data['limit']['effort']}")

                            # dynprm → actuation_lag
                            dynprm_str = motor_elem.get('dynprm')
                            if dynprm_str:
                                dynprm_vals = [float(v) for v in dynprm_str.split()]
                                if len(dynprm_vals) >= 1:
                                    joint_data['actuation_lag'] = dynprm_vals[0]
                                    print(f"  Motor '{joint_name}': actuation_lag={joint_data['actuation_lag']}")
                            break
            else:
                print("  No actuator element found in MJCF")

            print("=" * 50 + "\n")

            # ノードを作成
            nodes = {}

            # 既存のbase_linkノードを取得
            base_node = None
            for node in self.all_nodes():
                if isinstance(node, BaseLinkNode):
                    base_node = node
                    nodes['base_link'] = base_node
                    break

            # base_linkが見つからない場合は作成
            if not base_node:
                print("No base_link found, creating new one")
                base_node = self.create_node(
                    'insilico.nodes.BaseLinkNode',
                    name='base_link',
                    pos=QtCore.QPointF(20, 20)
                )
                nodes['base_link'] = base_node

            # 各ボディの子ジョイント数をカウント
            body_child_counts = {}
            for body_name in bodies_data.keys():
                body_child_counts[body_name] = 0
            body_child_counts['base_link'] = 0

            for joint_data in joints_data:
                parent = joint_data['parent']
                if parent in body_child_counts:
                    body_child_counts[parent] += 1

            # base_linkの出力ポートを追加
            if base_node:
                child_count = body_child_counts.get('base_link', 0)
                current_output_count = len(base_node.output_ports())
                needed_ports = child_count - current_output_count + 1
                if needed_ports > 0:
                    for i in range(needed_ports):
                        base_node._add_output()

            # 他のボディのノードを作成
            grid_spacing = 200
            current_x = grid_spacing
            current_y = 0
            nodes_per_row = 4
            node_count = 0

            for body_name, body_data in bodies_data.items():
                # グリッドレイアウトで位置を計算
                row = node_count // nodes_per_row
                col = node_count % nodes_per_row
                pos_x = current_x + col * grid_spacing
                pos_y = current_y + row * grid_spacing

                node = self.create_node(
                    'insilico.nodes.FooNode',
                    name=body_name,
                    pos=QtCore.QPointF(pos_x, pos_y)
                )
                nodes[body_name] = node

                # ノードのパラメータを設定
                node.mass_value = body_data['mass']
                node.inertia = body_data['inertia']
                node.inertial_origin = body_data['inertial_origin']
                node.node_color = body_data['color']

                # メッシュファイルの割り当て（デバッグ出力付き）
                if body_data['stl_file']:
                    node.stl_file = body_data['stl_file']
                    print(f"  ✓ Assigned mesh to node '{body_name}': {body_data['stl_file']}")
                else:
                    print(f"  ✗ No mesh file for node '{body_name}'")

                # 子ジョイントの数に応じて出力ポートを追加
                child_count = body_child_counts.get(body_name, 0)
                if child_count > 1:
                    for i in range(1, child_count):
                        node._add_output()

                # 複数のvisualがある場合、追加のノードを作成
                additional_visuals = body_data.get('visuals', [])[1:]  # 2つ目以降のvisual
                if additional_visuals:
                    print(f"  Body '{body_name}' has {len(additional_visuals)} additional visual(s)")

                    # 追加のvisualの数だけ出力ポートを追加
                    for i in range(len(additional_visuals)):
                        node._add_output()

                    # 追加のvisual用の子ノードを作成して接続
                    for visual_idx, visual_data in enumerate(additional_visuals, start=1):
                        visual_node_name = f"{body_name}_visual_{visual_idx}"

                        # 子ノードの位置を計算（親ノードの近くに配置）
                        visual_pos_x = pos_x + 50 + visual_idx * 30
                        visual_pos_y = pos_y + 100

                        visual_node = self.create_node(
                            'insilico.nodes.FooNode',
                            name=visual_node_name,
                            pos=QtCore.QPointF(visual_pos_x, visual_pos_y)
                        )
                        nodes[visual_node_name] = visual_node

                        # ビジュアルノードにメッシュを設定
                        visual_node.stl_file = visual_data['mesh']
                        visual_node.node_color = visual_data['color']
                        visual_node.mass_value = 0.0  # ビジュアルのみなので質量0

                        # visualのクォータニオンをRPYに変換
                        visual_rpy = quat_to_rpy(visual_data['quat'])

                        # 親ノードのポイント情報を設定
                        if not hasattr(node, 'points'):
                            node.points = []

                        # 子ジョイントの数 + visual_idxの位置にポイントを追加
                        point_index = child_count + visual_idx - 1
                        while len(node.points) <= point_index:
                            node.points.append({
                                'name': f'point_{len(node.points)}',
                                'type': 'fixed',
                                'xyz': [0.0, 0.0, 0.0],
                                'rpy': [0.0, 0.0, 0.0]
                            })

                        # visualの位置と姿勢を設定
                        node.points[point_index] = {
                            'name': f'visual_{visual_idx}_attachment',
                            'type': 'fixed',
                            'xyz': visual_data['pos'],  # geomのローカル位置
                            'rpy': visual_rpy  # geomのローカル姿勢（RPYに変換済み）
                        }
                        print(f"      Visual pos: {visual_data['pos']}, quat: {visual_data['quat']} -> rpy: {visual_rpy}")

                        # ポートを接続
                        is_base_link = isinstance(node, BaseLinkNode)
                        if is_base_link:
                            output_port_name = 'out' if point_index == 0 else f'out_{point_index + 1}'
                        else:
                            output_port_name = f'out_{point_index + 1}'

                        input_port_name = 'in'

                        print(f"    Connecting visual node: {body_name} -> {visual_node_name}")
                        print(f"      Port: {output_port_name} -> {input_port_name}")

                        try:
                            output_port = node.get_output(output_port_name)
                            input_port = visual_node.get_input(input_port_name)

                            if output_port and input_port:
                                output_port.connect_to(input_port)
                                print(f"      ✓ Connected")
                            else:
                                print(f"      ✗ Port not found")
                        except Exception as e:
                            print(f"      ✗ Error: {str(e)}")

                        # ビジュアルノードをfixedジョイントに設定
                        visual_node.rotation_axis = 3  # fixedジョイント

                node_count += 1

            # ジョイント情報を反映して接続
            parent_port_indices = {}

            for joint_data in joints_data:
                parent_name = joint_data['parent']
                child_name = joint_data['child']

                if parent_name not in nodes or child_name not in nodes:
                    continue

                parent_node = nodes[parent_name]
                child_node = nodes[child_name]

                # 親ノードの現在のポートインデックスを取得
                if parent_name not in parent_port_indices:
                    parent_port_indices[parent_name] = 0
                port_index = parent_port_indices[parent_name]
                parent_port_indices[parent_name] += 1

                # ポイント情報を親ノードに追加
                if not hasattr(parent_node, 'points'):
                    parent_node.points = []

                while len(parent_node.points) <= port_index:
                    parent_node.points.append({
                        'name': f'point_{len(parent_node.points)}',
                        'type': 'revolute',
                        'xyz': [0.0, 0.0, 0.0],
                        'rpy': [0.0, 0.0, 0.0]
                    })

                # ジョイントタイプを文字列に変換
                joint_type_str = joint_data['type']

                parent_node.points[port_index] = {
                    'name': joint_data['name'],
                    'type': joint_type_str,
                    'xyz': joint_data['origin_xyz'],
                    'rpy': joint_data['origin_rpy']
                }

                # 子ノードにジョイント情報を設定
                child_node.joint_lower = joint_data['limit']['lower']
                child_node.joint_upper = joint_data['limit']['upper']
                child_node.joint_effort = joint_data['limit']['effort']
                child_node.joint_velocity = joint_data['limit']['velocity']

                # dynamics情報を設定（damping, friction）
                if 'dynamics' in joint_data:
                    child_node.damping = joint_data['dynamics']['damping']
                    child_node.friction = joint_data['dynamics']['friction']

                # stiffnessを設定
                if 'stiffness' in joint_data:
                    child_node.stiffness = joint_data['stiffness']

                # actuation_lagを設定
                if 'actuation_lag' in joint_data:
                    child_node.actuation_lag = joint_data['actuation_lag']

                # 回転軸を設定
                axis = joint_data['axis']
                if abs(axis[2]) > 0.5:  # Z軸
                    child_node.rotation_axis = 2
                    axis_name = "Z"
                elif abs(axis[1]) > 0.5:  # Y軸
                    child_node.rotation_axis = 1
                    axis_name = "Y"
                else:  # X軸
                    child_node.rotation_axis = 0
                    axis_name = "X"

                print(f"  Joint axis: {axis} -> rotation_axis: {child_node.rotation_axis} ({axis_name})")

                # ジョイントタイプがfixedの場合
                if joint_data['type'] == 'fixed':
                    child_node.rotation_axis = 3
                    print(f"  Fixed joint -> rotation_axis: 3")

                # ポートを接続
                is_base_link = isinstance(parent_node, BaseLinkNode)
                if is_base_link:
                    output_port_name = 'out' if port_index == 0 else f'out_{port_index + 1}'
                else:
                    output_port_name = f'out_{port_index + 1}'

                input_port_name = 'in'

                print(f"\nConnecting: {parent_name} -> {child_name}")
                print(f"  Port: {output_port_name} -> {input_port_name}")

                try:
                    output_port = parent_node.get_output(output_port_name)
                    input_port = child_node.get_input(input_port_name)

                    if output_port and input_port:
                        output_port.connect_to(input_port)
                        print(f"  ✓ Connected")
                    else:
                        print(f"  ✗ Port not found")
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")

            # メッシュをSTL viewerにロード
            print("\n=== Loading meshes to 3D viewer ===")
            if self.stl_viewer:
                stl_viewer_loaded_count = 0
                skipped_no_mesh = 0
                skipped_base_link = 0

                for body_name, node in nodes.items():
                    if body_name == 'base_link':
                        print(f"Skipping base_link")
                        skipped_base_link += 1
                        continue

                    # デバッグ: ノードの属性を確認
                    has_stl_attr = hasattr(node, 'stl_file')
                    stl_value = getattr(node, 'stl_file', None) if has_stl_attr else None

                    print(f"\nChecking node: {body_name}")
                    print(f"  has stl_file attr: {has_stl_attr}")
                    print(f"  stl_file value: {stl_value}")

                    if has_stl_attr and stl_value:
                        try:
                            print(f"  → Loading mesh to viewer...")
                            self.stl_viewer.load_stl_for_node(node)
                            stl_viewer_loaded_count += 1
                            print(f"  ✓ Successfully loaded mesh")
                        except Exception as e:
                            print(f"  ✗ Error loading mesh: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"  ✗ Skipped (no mesh file assigned)")
                        skipped_no_mesh += 1

                print(f"\n--- Mesh Loading Summary ---")
                print(f"Successfully loaded: {stl_viewer_loaded_count}")
                print(f"Skipped (no mesh): {skipped_no_mesh}")
                print(f"Skipped (base_link): {skipped_base_link}")
                print("=" * 40 + "\n")
            else:
                print("Warning: Mesh viewer not available")

            # 位置を再計算
            print("\nRecalculating positions...")
            QtCore.QTimer.singleShot(100, self.recalculate_all_positions)

            # 一時ディレクトリのクリーンアップ（ZIPファイルの場合）
            # 注: メッシュファイルは既にノードに読み込まれているため、一時ファイルは削除可能
            # ただし、すぐに削除するとビューアが読み込めない可能性があるため、
            # 数秒後に削除するようスケジュール
            if xml_file_to_load and working_dir != os.path.dirname(mjcf_file):
                import shutil
                def cleanup_temp_dir():
                    try:
                        if os.path.exists(working_dir):
                            shutil.rmtree(working_dir)
                            print(f"Cleaned up temporary directory: {working_dir}")
                    except Exception as e:
                        print(f"Warning: Could not clean up temporary directory: {e}")

                # 5秒後にクリーンアップ（メッシュが読み込まれるのを待つ）
                QtCore.QTimer.singleShot(5000, cleanup_temp_dir)

            QtWidgets.QMessageBox.information(
                self.widget,
                "MJCF Import Complete",
                f"Successfully imported {len(bodies_data)} bodies and {len(joints_data)} joints from MJCF file."
            )

            return True

        except Exception as e:
            print(f"Error importing MJCF: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self.widget,
                "MJCF Import Error",
                f"Failed to import MJCF file:\n\n{str(e)}"
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

            # メッシュファイルを収集してコピー
            stl_files_copied = []
            stl_files_failed = []

            for node in self.all_nodes():
                # Hide Meshにチェックが入っているノードはスキップ
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                if hasattr(node, 'stl_file') and node.stl_file:
                    source_path = node.stl_file
                    if os.path.exists(source_path):
                        stl_filename = os.path.basename(source_path)
                        dest_path = os.path.join(meshes_dir, stl_filename)
                        file_ext = os.path.splitext(stl_filename)[1].lower()

                        try:
                            # .daeファイルの場合は色情報を埋め込んで保存
                            if file_ext == '.dae' and hasattr(node, 'node_color'):
                                self._copy_dae_with_color(source_path, dest_path, node.node_color)
                                stl_files_copied.append(stl_filename)
                                print(f"Copied mesh with color: {stl_filename}")
                            else:
                                # STLファイルやその他のファイルは通常コピー
                                shutil.copy2(source_path, dest_path)
                                stl_files_copied.append(stl_filename)
                                print(f"Copied mesh: {stl_filename}")
                        except Exception as e:
                            stl_files_failed.append((stl_filename, str(e)))
                            print(f"Failed to copy mesh {stl_filename}: {str(e)}")
                    else:
                        stl_files_failed.append((os.path.basename(source_path), "Source file not found"))
                        print(f"Mesh file not found: {source_path}")

            print(f"\nMesh files copied: {len(stl_files_copied)}")
            if stl_files_failed:
                print(f"Mesh files failed: {len(stl_files_failed)}")
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
                export_summary += f"Mesh files copied: {len(stl_files_copied)}\n"

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

        # Hide Meshにチェックが入っているノードはスキップ
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            print(f"Skipping node with hide_mesh=True: {node.name()}")
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

            # コリジョン (collider_meshが設定されている場合はそれを使用)
            file.write('    <collision>\n')
            file.write(f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
            file.write('      <geometry>\n')

            # collider_meshが設定されている場合はそれを使用
            if hasattr(node, 'collider_mesh') and node.collider_mesh:
                # collider_meshは相対パスなので、visual meshと同じディレクトリから取得
                visual_dir = os.path.dirname(node.stl_file)
                collider_absolute = os.path.join(visual_dir, node.collider_mesh)
                collider_filename = os.path.basename(collider_absolute)
                collider_package_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{collider_filename}"
                file.write(f'        <mesh filename="{collider_package_path}"/>\n')
            else:
                # collider_meshが設定されていない場合はvisual meshと同じものを使用
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

                # インスペクター表示時は点滅を停止
                if self.stl_viewer:
                    self.stl_viewer.clear_highlight()

                print(f"Inspector window displayed for node: {node.name()}")

        except Exception as e:
            print(f"Error showing inspector: {str(e)}")
            traceback.print_exc()

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
        if hasattr(new_node, 'joint_friction'):
            new_node.joint_friction = self.default_joint_friction
        print(f"Applied default values to new node: effort={self.default_joint_effort}, velocity={self.default_joint_velocity}, friction={self.default_joint_friction}")

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
            if hasattr(node, 'joint_friction'):
                ET.SubElement(node_elem, "joint_friction").text = str(node.joint_friction)
                print(f"  Saved joint_friction: {node.joint_friction}")

            # Massless Decoration
            if hasattr(node, 'massless_decoration'):
                ET.SubElement(node_elem, "massless_decoration").text = str(node.massless_decoration)
                print(f"  Saved massless_decoration: {node.massless_decoration}")

            # Hide Mesh
            if hasattr(node, 'hide_mesh'):
                ET.SubElement(node_elem, "hide_mesh").text = str(node.hide_mesh)
                print(f"  Saved hide_mesh: {node.hide_mesh}")

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

            # カスタムカラーパレットの保存
            print("\nSaving custom color palette...")
            custom_colors_elem = ET.SubElement(root, "custom_colors")
            for i in range(16):
                color = QtWidgets.QColorDialog.customColor(i)
                if color.isValid():
                    color_elem = ET.SubElement(custom_colors_elem, "color")
                    color_elem.set("index", str(i))
                    color_elem.set("r", str(color.red()))
                    color_elem.set("g", str(color.green()))
                    color_elem.set("b", str(color.blue()))
                    color_elem.set("a", str(color.alpha()))
                    print(f"Saved custom color {i}: {color.name()}")
            print(f"Total custom colors saved: 16")

            # ハイライトカラーの保存
            print("\nSaving highlight color...")
            highlight_color_elem = ET.SubElement(root, "highlight_color")
            highlight_color_elem.text = self.highlight_color
            print(f"Saved highlight color: {self.highlight_color}")

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

            # カスタムカラーパレットの復元
            print("\nRestoring custom color palette...")
            custom_colors_elem = root.find("custom_colors")
            if custom_colors_elem is not None:
                color_count = 0
                for color_elem in custom_colors_elem.findall("color"):
                    try:
                        index = int(color_elem.get("index"))
                        r = int(color_elem.get("r"))
                        g = int(color_elem.get("g"))
                        b = int(color_elem.get("b"))
                        a = int(color_elem.get("a"))
                        color = QtGui.QColor(r, g, b, a)
                        QtWidgets.QColorDialog.setCustomColor(index, color)
                        color_count += 1
                    except Exception as e:
                        print(f"Error restoring custom color: {e}")
                print(f"Total custom colors restored: {color_count}")
            else:
                print("No custom colors found in project file")

            # ハイライトカラーの復元
            print("\nRestoring highlight color...")
            highlight_color_elem = root.find("highlight_color")
            if highlight_color_elem is not None and highlight_color_elem.text:
                self.highlight_color = highlight_color_elem.text
                print(f"Restored highlight color: {self.highlight_color}")
            else:
                print("No highlight color found in project file, using default")

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
                node = self._load_node_data(node_elem)
                if node:
                    nodes_dict[node.name()] = node

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

            # 位置の再計算とビューの更新
            print("\nRecalculating positions...")
            self.recalculate_all_positions()

            print("Updating 3D view...")
            if self.stl_viewer:
                # ビューをリセット
                self.stl_viewer.reset_view_to_fit()

                # Hide Mesh状態を全ノードに適用（recalculate_all_positions後に確実に実行）
                print("\nApplying hide_mesh states after position recalculation...")
                for node in nodes_dict.values():
                    if hasattr(node, 'hide_mesh') and node.hide_mesh:
                        if node in self.stl_viewer.stl_actors:
                            actor = self.stl_viewer.stl_actors[node]
                            actor.SetVisibility(False)
                            print(f"Applied hide_mesh: {node.name()} - mesh hidden")

                # 3Dビューを更新
                self.stl_viewer.render_to_image()

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

            # output_count の復元とポートの追加
            if isinstance(node, FooNode):
                points_elem = node_elem.find("points")
                if points_elem is not None:
                    points = points_elem.findall("point")
                    num_points = len(points)
                    
                    # 必要な数のポートを追加
                    while len(node.output_ports()) < num_points:
                        node._add_output()

                    # ポイントデータの復元
                    node.points = []
                    for point_elem in points:
                        point_data = {
                            'name': point_elem.find("name").text,
                            'type': point_elem.find("type").text,
                            'xyz': [float(x) for x in point_elem.find("xyz").text.split()]
                        }
                        node.points.append(point_data)

                    # output_countを更新
                    node.output_count = num_points

            # 位置の設定
            pos_elem = node_elem.find("position")
            if pos_elem is not None:
                x = float(pos_elem.find("x").text)
                y = float(pos_elem.find("y").text)
                node.set_pos(x, y)

            # 物理プロパティの復元
            volume_elem = node_elem.find("volume")
            if volume_elem is not None:
                node.volume_value = float(volume_elem.text)

            mass_elem = node_elem.find("mass")
            if mass_elem is not None:
                node.mass_value = float(mass_elem.text)

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

            # 色情報の復元
            color_elem = node_elem.find("color")
            if color_elem is not None and color_elem.text:
                node.node_color = [float(x) for x in color_elem.text.split()]

            # 回転軸の復元
            rotation_axis_elem = node_elem.find("rotation_axis")
            if rotation_axis_elem is not None:
                node.rotation_axis = int(rotation_axis_elem.text)

            # ジョイント制限パラメータの復元
            joint_lower_elem = node_elem.find("joint_lower")
            if joint_lower_elem is not None:
                node.joint_lower = float(joint_lower_elem.text)

            joint_upper_elem = node_elem.find("joint_upper")
            if joint_upper_elem is not None:
                node.joint_upper = float(joint_upper_elem.text)

            joint_effort_elem = node_elem.find("joint_effort")
            if joint_effort_elem is not None:
                node.joint_effort = float(joint_effort_elem.text)

            joint_velocity_elem = node_elem.find("joint_velocity")
            if joint_velocity_elem is not None:
                node.joint_velocity = float(joint_velocity_elem.text)

            joint_friction_elem = node_elem.find("joint_friction")
            if joint_friction_elem is not None:
                node.joint_friction = float(joint_friction_elem.text)

            # Massless Decorationの復元
            massless_dec_elem = node_elem.find("massless_decoration")
            if massless_dec_elem is not None:
                node.massless_decoration = massless_dec_elem.text.lower() == 'true'

            # Hide Meshの復元
            hide_mesh_elem = node_elem.find("hide_mesh")
            if hide_mesh_elem is not None:
                node.hide_mesh = hide_mesh_elem.text.lower() == 'true'
            else:
                node.hide_mesh = False

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
                        self.stl_viewer.load_stl_for_node(node)

                        # Apply hide_mesh state to VTK actor
                        if hasattr(node, 'hide_mesh') and node.hide_mesh:
                            if node in self.stl_viewer.stl_actors:
                                actor = self.stl_viewer.stl_actors[node]
                                actor.SetVisibility(False)
                else:
                    print(f"Warning: STL file not found: {abs_path}")

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

                    # 慣性関連の処理
                    inertial_elem = link_elem.find('inertial')
                    if inertial_elem is not None:
                        # 質量の設定
                        mass_elem = inertial_elem.find('mass')
                        if mass_elem is not None:
                            new_node.mass_value = float(mass_elem.get('value', '0.0'))

                        # ボリュームの設定
                        volume_elem = inertial_elem.find('volume')
                        if volume_elem is not None:
                            new_node.volume_value = float(volume_elem.get('value', '0.0'))

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
                    else:
                        # 回転軸の処理
                        axis_elem = joint_elem.find('axis')
                        if axis_elem is not None:
                            axis_xyz = axis_elem.get('xyz', '1 0 0').split()
                            axis_values = [float(x) for x in axis_xyz]
                            if axis_values[2] == 1:      # Z軸
                                new_node.rotation_axis = 2
                            elif axis_values[1] == 1:    # Y軸
                                new_node.rotation_axis = 1
                            else:                        # X軸（デフォルト）
                                new_node.rotation_axis = 0
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

        # すべてのノードの色を接続状態に応じて更新
        self.update_all_node_colors()

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

        # 現在のノードのHide Meshがオンの場合は3Dビューで非表示にする
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            if hasattr(self, 'stl_viewer') and node in self.stl_viewer.stl_actors:
                actor = self.stl_viewer.stl_actors[node]
                actor.SetVisibility(False)
                print(f"Applied hide_mesh: {node.name()} - mesh hidden in 3D view")

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

                        # Hide Meshがオンの場合は3Dビューで非表示にする
                        if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                            if child_node in self.stl_viewer.stl_actors:
                                actor = self.stl_viewer.stl_actors[child_node]
                                actor.SetVisibility(False)
                                print(f"Applied hide_mesh: {child_node.name()} - mesh hidden in 3D view")

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

    def build_r_from_l(self):
        """左系統（l_）のノードから右系統（r_）のノードを自動生成"""
        print("Building right side (r_) from left side (l_)...")

        try:
            # 左系統のノードを収集
            l_nodes = {}
            for node in self.all_nodes():
                node_name = node.name()
                if node_name.startswith('l_'):
                    l_nodes[node_name] = node
                    print(f"Found left node: {node_name}")

            if not l_nodes:
                print("No left side nodes (l_) found")
                return

            # 既存の右系統ノードを収集
            existing_r_nodes = {}
            for node in self.all_nodes():
                node_name = node.name()
                if node_name.startswith('r_') and node_name.replace('r_', 'l_') in l_nodes:
                    existing_r_nodes[node_name] = node
                    print(f"Found existing right node: {node_name}")

            # 既存の右系統ノードの接続をクリア
            for r_node in existing_r_nodes.values():
                print(f"Clearing connections for: {r_node.name()}")
                # 入力ポートの接続をクリア
                for input_port in r_node.input_ports():
                    input_port.clear_connections()
                # 出力ポートの接続をクリア
                for output_port in r_node.output_ports():
                    output_port.clear_connections()

            # 新規作成するr_ノードの配置位置を計算
            # 既存のすべてのノードの最右端を取得
            max_x = -float('inf')
            min_x = float('inf')
            for node in self.all_nodes():
                pos = node.pos()
                x = pos.x() if hasattr(pos, 'x') else pos[0]
                max_x = max(max_x, x)
                min_x = min(min_x, x)

            # l_ノードの範囲を取得
            l_min_x = float('inf')
            l_max_x = -float('inf')
            for l_node in l_nodes.values():
                pos = l_node.pos()
                x = pos.x() if hasattr(pos, 'x') else pos[0]
                l_min_x = min(l_min_x, x)
                l_max_x = max(l_max_x, x)

            # r_ノードを右側に配置するためのオフセットを計算
            # 既存ノードの右端から200ピクセル離す
            x_offset = max_x - l_min_x + 200

            print(f"Positioning r_ nodes with X offset: {x_offset}")

            # 左系統ノードから右系統ノードを作成または更新
            l_to_r_mapping = {}
            for l_name, l_node in l_nodes.items():
                r_name = l_name.replace('l_', 'r_', 1)

                # 既存のノードがあるかチェック
                if r_name in existing_r_nodes:
                    print(f"\nUpdating existing {r_name} from {l_name}")
                    r_node = existing_r_nodes[r_name]
                else:
                    print(f"\nCreating new {r_name} from {l_name}")
                    # ノードタイプを取得（__identifier__を使用）
                    if hasattr(l_node, '__identifier__'):
                        node_type = l_node.__identifier__ + '.' + type(l_node).__name__
                    else:
                        node_type = type(l_node).__module__ + '.' + type(l_node).__name__

                    print(f"  Node type: {node_type}")

                    # グラフビュー上の位置を計算（l_ノードからの相対位置を保持し、X方向にオフセット）
                    l_pos = l_node.pos()
                    # pos()がリストかQPointFか判定
                    if isinstance(l_pos, list):
                        # l_ノードからの相対位置を計算してオフセットを適用
                        r_pos = QtCore.QPointF(l_pos[0] + x_offset, l_pos[1])
                    else:
                        # l_ノードからの相対位置を計算してオフセットを適用
                        r_pos = QtCore.QPointF(l_pos.x() + x_offset, l_pos.y())

                    print(f"  Position: ({r_pos.x()}, {r_pos.y()})")

                    # 新しいノードを作成
                    r_node = self.create_node(node_type, name=r_name, pos=r_pos)

                l_to_r_mapping[l_node] = r_node

                # プロパティをコピー
                if hasattr(l_node, 'stl_file'):
                    # STLファイルパスを変換（l_をr_に置換）
                    stl_file = l_node.stl_file
                    if stl_file and 'l_' in os.path.basename(stl_file):
                        r_stl_file = stl_file.replace('/l_', '/r_').replace('\\l_', '\\r_')
                        # ファイル名自体にl_が含まれる場合も置換
                        dirname = os.path.dirname(r_stl_file)
                        basename = os.path.basename(r_stl_file)
                        basename = basename.replace('l_', 'r_', 1)
                        r_stl_file = os.path.join(dirname, basename)
                        r_node.stl_file = r_stl_file
                        print(f"  STL: {stl_file} -> {r_stl_file}")

                        # メッシュファイルを3Dビューアに読み込む
                        if hasattr(self, 'stl_viewer') and self.stl_viewer:
                            try:
                                self.stl_viewer.load_stl_for_node(r_node)
                                print(f"  Loaded mesh for {r_name}")
                            except Exception as e:
                                print(f"  Warning: Could not load mesh for {r_name}: {str(e)}")

                # 物理プロパティをコピー
                if hasattr(l_node, 'volume_value'):
                    r_node.volume_value = l_node.volume_value
                if hasattr(l_node, 'mass_value'):
                    r_node.mass_value = l_node.mass_value

                # 慣性プロパティのミラーリング
                # 方法1: メッシュファイルから再計算（最も正確）
                use_mesh_recalculation = False
                if hasattr(l_node, 'stl_file') and l_node.stl_file and os.path.exists(l_node.stl_file):
                    if hasattr(l_node, 'mass_value') and l_node.mass_value > 0:
                        print(f"  Attempting to recalculate mirrored properties from mesh...")
                        mirrored_props = calculate_mirrored_physical_properties_from_mesh(
                            l_node.stl_file, l_node.mass_value
                        )
                        if mirrored_props is not None:
                            # メッシュから計算成功
                            r_node.volume_value = mirrored_props['volume']
                            r_node.mass_value = mirrored_props['mass']
                            r_node.inertia = mirrored_props['inertia']
                            if not hasattr(r_node, 'inertial_origin'):
                                r_node.inertial_origin = {}
                            r_node.inertial_origin['xyz'] = mirrored_props['center_of_mass']
                            if 'rpy' not in r_node.inertial_origin:
                                r_node.inertial_origin['rpy'] = [0.0, 0.0, 0.0]
                            use_mesh_recalculation = True
                            print(f"  ✓ Recalculated from mirrored mesh (most accurate)")
                            print(f"    COM: {mirrored_props['center_of_mass']}")

                # 方法2: 既存のプロパティを変換（フォールバック）
                if not use_mesh_recalculation:
                    print(f"  Using property transformation (fallback method)")
                    # 共通関数を使用してミラーリング
                    mirrored_inertia, mirrored_origin = mirror_physical_properties_y_axis(
                        l_node.inertia if hasattr(l_node, 'inertia') else None,
                        l_node.inertial_origin if hasattr(l_node, 'inertial_origin') else None
                    )
                    if mirrored_inertia:
                        r_node.inertia = mirrored_inertia
                        print(f"  ✓ Mirrored inertia tensor (negated ixy, iyz)")
                    if mirrored_origin:
                        r_node.inertial_origin = mirrored_origin
                        if 'xyz' in mirrored_origin:
                            xyz = mirrored_origin['xyz']
                            print(f"  ✓ Mirrored COM: Y={xyz[1]:.6f}")
                if hasattr(l_node, 'node_color'):
                    r_node.node_color = l_node.node_color
                if hasattr(l_node, 'rotation_axis'):
                    r_node.rotation_axis = l_node.rotation_axis

                # Joint limits: swap and negate for Roll(0) or Yaw(2) axes
                if hasattr(l_node, 'joint_lower') and hasattr(l_node, 'joint_upper'):
                    rotation_axis = getattr(l_node, 'rotation_axis', 1)  # Default to Pitch if not set
                    if rotation_axis == 0 or rotation_axis == 2:  # Roll or Yaw
                        # Swap lower and upper, and negate for left-right symmetry
                        # Example: lower=-10, upper=190 -> lower=-190, upper=10
                        r_node.joint_lower = -l_node.joint_upper
                        r_node.joint_upper = -l_node.joint_lower
                        print(f"  Swapped and negated joint limits for {['Roll', 'Pitch', 'Yaw'][rotation_axis]} axis: {l_node.joint_lower:.3f},{l_node.joint_upper:.3f} -> {r_node.joint_lower:.3f},{r_node.joint_upper:.3f}")
                    else:  # Pitch or other
                        r_node.joint_lower = l_node.joint_lower
                        r_node.joint_upper = l_node.joint_upper
                elif hasattr(l_node, 'joint_lower'):
                    r_node.joint_lower = l_node.joint_lower
                elif hasattr(l_node, 'joint_upper'):
                    r_node.joint_upper = l_node.joint_upper
                if hasattr(l_node, 'joint_effort'):
                    r_node.joint_effort = l_node.joint_effort
                if hasattr(l_node, 'joint_velocity'):
                    r_node.joint_velocity = l_node.joint_velocity
                if hasattr(l_node, 'joint_friction'):
                    r_node.joint_friction = l_node.joint_friction
                if hasattr(l_node, 'massless_decoration'):
                    r_node.massless_decoration = l_node.massless_decoration
                if hasattr(l_node, 'hide_mesh'):
                    r_node.hide_mesh = l_node.hide_mesh

                # Collider mesh をコピー（l_をr_に置換）
                if hasattr(l_node, 'collider_mesh') and l_node.collider_mesh:
                    collider_mesh = l_node.collider_mesh
                    # 相対パスのファイル名にl_が含まれる場合は置換
                    if 'l_' in collider_mesh:
                        r_collider_mesh = collider_mesh.replace('l_', 'r_', 1)
                        r_node.collider_mesh = r_collider_mesh
                        print(f"  Collider mesh: {collider_mesh} -> {r_collider_mesh}")
                    else:
                        r_node.collider_mesh = collider_mesh
                        print(f"  Collider mesh: {collider_mesh} (unchanged)")
                else:
                    r_node.collider_mesh = None

                # ポイントをコピー（Y座標を反転してミラーリング）
                if hasattr(l_node, 'points') and hasattr(r_node, 'points'):
                    # 必要な出力ポート数を計算（複数の要素から最大値を取る）
                    l_port_count = len(l_node.output_ports())
                    l_points_count = len(l_node.points) if hasattr(l_node, 'points') else 0

                    # 実際に接続が使用している最大ポートインデックス + 1
                    max_used_port = -1
                    for port_idx, output_port in enumerate(l_node.output_ports()):
                        if output_port.connected_ports():
                            max_used_port = port_idx
                    required_port_count = max_used_port + 1 if max_used_port >= 0 else 0

                    # 最大値を使用（XML、現在のポート数、実際の接続で使用されているポート数の最大）
                    target_port_count = max(l_port_count, l_points_count, required_port_count)

                    print(f"  Port count - Current: {l_port_count}, Points: {l_points_count}, Required: {required_port_count}, Target: {target_port_count}")

                    # ポートが足りない場合は追加
                    while len(r_node.output_ports()) < target_port_count:
                        if hasattr(r_node, '_add_output'):
                            r_node._add_output()
                            print(f"  Added output port to {r_name} (now {len(r_node.output_ports())} ports)")

                    # ポートが多すぎる場合は削除
                    while len(r_node.output_ports()) > target_port_count:
                        if hasattr(r_node, 'remove_output'):
                            # 接続をクリアしてから削除
                            if r_node.output_ports():
                                last_port = r_node.output_ports()[-1]
                                last_port.clear_connections()
                            r_node.remove_output()
                            print(f"  Removed output port from {r_name} (now {len(r_node.output_ports())} ports)")

                    r_node.points = []
                    for point in l_node.points:
                        r_point = point.copy()
                        # Y座標を反転
                        if 'xyz' in r_point:
                            xyz = r_point['xyz']
                            r_point['xyz'] = [xyz[0], -xyz[1], xyz[2]]
                        r_node.points.append(r_point)

                    # cumulative_coordsも更新
                    if hasattr(r_node, 'cumulative_coords'):
                        r_node.cumulative_coords = []
                        for i in range(len(r_node.points)):
                            r_node.cumulative_coords.append({'point_index': i, 'xyz': [0, 0, 0]})

                print(f"  Created {r_name} successfully")

            # 接続をミラーリング
            print("\nMirroring connections...")
            connection_count = 0
            failed_connections = []

            for l_node, r_node in l_to_r_mapping.items():
                # 出力ポートの接続をミラーリング
                for port_idx, output_port in enumerate(l_node.output_ports()):
                    for connected_port in output_port.connected_ports():
                        connected_node = connected_port.node()
                        connected_node_name = connected_node.name()

                        # 接続先に対応するr_ノードを探す
                        r_connected_node = None

                        # 1. 接続先が左系統ノード (l_) の場合
                        if connected_node in l_to_r_mapping:
                            r_connected_node = l_to_r_mapping[connected_node]
                            print(f"  Found in mapping: {connected_node_name} -> {r_connected_node.name()}")

                        # 2. 接続先が右系統ノード (r_) の場合、対応するl_ノードを探してそのr_版を使用
                        elif connected_node_name.startswith('r_'):
                            # r_をl_に変換して探す（末尾のスペースや数字も考慮）
                            import re
                            # 末尾のスペースと数字を削除
                            base_name = re.sub(r'\s+\d+$', '', connected_node_name)
                            l_version_name = base_name.replace('r_', 'l_', 1)

                            # 完全一致で探す
                            for l_n, r_n in l_to_r_mapping.items():
                                if l_n.name() == l_version_name:
                                    r_connected_node = r_n
                                    print(f"  Found r_ node {connected_node_name} -> using r_ version: {r_n.name()}")
                                    break

                            # 見つからない場合、元の名前でも試す
                            if r_connected_node is None:
                                l_version_name_original = connected_node_name.replace('r_', 'l_', 1)
                                for l_n, r_n in l_to_r_mapping.items():
                                    if l_n.name() == l_version_name_original:
                                        r_connected_node = r_n
                                        print(f"  Found r_ node {connected_node_name} (exact match) -> using r_ version: {r_n.name()}")
                                        break

                        # 3. 見つからない場合、全ノードから名前で直接探す
                        if r_connected_node is None:
                            # l_をr_に変換した名前で探す
                            target_name = connected_node_name.replace('l_', 'r_', 1) if 'l_' in connected_node_name else 'r_' + connected_node_name
                            for node in self.all_nodes():
                                if node.name() == target_name:
                                    r_connected_node = node
                                    print(f"  Found by name search: {target_name}")
                                    break

                        if r_connected_node:
                            # 対応するポートを取得
                            if port_idx < len(r_node.output_ports()):
                                r_output_port = r_node.output_ports()[port_idx]

                                # 入力ポートが存在するか確認
                                if not r_connected_node.input_ports():
                                    print(f"  Warning: {r_connected_node.name()} has no input ports")
                                    failed_connections.append(f"{r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()} (no input port)")
                                    continue

                                r_input_port = r_connected_node.input_ports()[0]

                                # 既に接続されているかチェック
                                if r_input_port in r_output_port.connected_ports():
                                    print(f"  Already connected: {r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()}.{r_input_port.name()}")
                                    connection_count += 1
                                    continue

                                # 接続を試行
                                try:
                                    print(f"  Connecting {r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()}.{r_input_port.name()}")
                                    r_output_port.connect_to(r_input_port)
                                    connection_count += 1
                                    print(f"    ✓ Successfully connected")
                                except Exception as e:
                                    error_msg = f"{r_node.name()}.{r_output_port.name()} -> {r_connected_node.name()}.{r_input_port.name()}: {str(e)}"
                                    failed_connections.append(error_msg)
                                    print(f"    ✗ Failed to connect: {str(e)}")
                            else:
                                error_msg = f"{r_node.name()} port {port_idx} out of range (has {len(r_node.output_ports())} ports)"
                                failed_connections.append(error_msg)
                                print(f"  Warning: {error_msg}")
                        else:
                            # 対応するr_ノードが見つからなかった
                            error_msg = f"{r_node.name()}.out_{port_idx+1} -> {connected_node_name}: No corresponding r_ node found"
                            failed_connections.append(error_msg)
                            print(f"  Warning: {error_msg}")

            print(f"\nConnection summary: {connection_count} connections established")
            if failed_connections:
                print(f"Failed connections ({len(failed_connections)}):")
                for fc in failed_connections:
                    print(f"  - {fc}")

            # r_ノードを斜め下に整列
            print("\nRearranging r_ nodes diagonally...")
            if l_to_r_mapping:
                # r_ノードをリストに変換してソート（Y座標でソート）
                r_nodes_list = list(l_to_r_mapping.values())
                r_nodes_list.sort(key=lambda n: (n.pos().y() if hasattr(n.pos(), 'y') else n.pos()[1]))

                # 開始位置を計算（既存ノードの右端から）
                start_x = max_x + 200
                start_y = min([n.pos().y() if hasattr(n.pos(), 'y') else n.pos()[1] for n in r_nodes_list])

                # 斜め下に配置（1ノードごとに右に50、下に75移動）
                x_increment = 50
                y_increment = 75

                for idx, r_node in enumerate(r_nodes_list):
                    new_x = start_x + (idx * x_increment)
                    new_y = start_y + (idx * y_increment)
                    r_node.set_pos(new_x, new_y)
                    print(f"  Repositioned {r_node.name()} to ({new_x}, {new_y})")

            # 位置を再計算
            self.recalculate_all_positions()

            # すべてのノードの色を接続状態に応じて更新
            self.update_all_node_colors()

            print(f"\nSuccessfully created {len(l_to_r_mapping)} right side nodes from left side")

        except Exception as e:
            print(f"Error building right side from left side: {str(e)}")
            import traceback
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

                    # URDF標準のlimit要素（effort, velocityのみ）
                    file.write(f'    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>\n')

                    # dynamics要素（damping, friction）
                    damping = getattr(child_node, 'damping', 0.0)
                    friction = getattr(child_node, 'friction', 0.0)
                    file.write(f'    <dynamics damping="{damping}" friction="{friction}"/>\n')

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
                    for port_index, port in enumerate(node.output_ports()):
                        for connected_port in port.connected_ports():
                            dec_node = connected_port.node()
                            if hasattr(dec_node, 'massless_decoration') and dec_node.massless_decoration:
                                if hasattr(dec_node, 'stl_file') and dec_node.stl_file:
                                    dec_stl = os.path.basename(dec_node.stl_file)
                                    dec_path = f"package://{self.robot_name}_description/{mesh_dir_name}/{dec_stl}"

                                    # ポイント座標を取得（ジョイント位置）
                                    origin_xyz = "0 0 0"
                                    if hasattr(node, 'points') and port_index < len(node.points):
                                        point_data = node.points[port_index]
                                        if 'xyz' in point_data:
                                            xyz = point_data['xyz']
                                            origin_xyz = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                                    file.write('    <visual>\n')
                                    file.write(f'      <origin xyz="{origin_xyz}" rpy="0 0 0"/>\n')
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
                        print(f"Copied mesh file: {stl_filename}")

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
            print(f"- Copied {len(copied_files)} mesh files")

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

    def export_mjcf(self):
        """MuJoCo MJCF形式でエクスポート（モジュラー構造）"""
        try:
            # ロボット名を取得し、制御文字を除去
            import re
            import shutil
            robot_name = self.robot_name or "robot"
            # 制御文字（\x00-\x1F, \x7F-\x9F）を除去
            robot_name = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', robot_name)
            # 空白をアンダースコアに置換
            robot_name = robot_name.replace(' ', '_')
            # 空文字列の場合はデフォルト名を使用
            if not robot_name:
                robot_name = "robot"

            # ディレクトリ名入力ダイアログ
            dir_name, ok = QtWidgets.QInputDialog.getText(
                self.widget,
                "MJCF Export - Directory Name",
                "Enter directory name for MJCF export:",
                QtWidgets.QLineEdit.Normal,
                f"{robot_name}_mjcf"
            )

            if not ok or not dir_name:
                print("MJCF export cancelled")
                return False

            # ディレクトリ名をサニタイズ
            dir_name = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', dir_name)
            dir_name = dir_name.replace(' ', '_')
            if not dir_name:
                dir_name = "robot"

            # 保存先の親ディレクトリを選択
            parent_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.widget,
                "Select Parent Directory for MJCF Export"
            )

            if not parent_dir:
                print("MJCF export cancelled")
                return False

            # MJCFディレクトリ構造を作成
            mjcf_dir = os.path.join(parent_dir, dir_name)
            assets_dir = os.path.join(mjcf_dir, "assets")

            os.makedirs(mjcf_dir, exist_ok=True)
            os.makedirs(assets_dir, exist_ok=True)

            print(f"\n=== Exporting MJCF to {mjcf_dir} ===")
            print(f"MJCF directory: {mjcf_dir}")
            print(f"Assets directory: {assets_dir}")

            # メッシュファイルをコピーしてマッピングを作成
            node_to_mesh = {}
            mesh_names = {}
            mesh_counter = 0

            for node in self.all_nodes():
                # Hide Meshにチェックが入っているノードはスキップ
                if hasattr(node, 'hide_mesh') and node.hide_mesh:
                    continue

                if hasattr(node, 'stl_file') and node.stl_file and os.path.exists(node.stl_file):
                    # 元のメッシュファイル名を取得
                    original_filename = os.path.basename(node.stl_file)

                    # メッシュファイルを検証してOBJ形式に変換
                    try:
                        import trimesh
                        mesh = trimesh.load(node.stl_file)

                        # 面の数を確認（MuJoCoは1〜200000の範囲を要求）
                        num_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0

                        if num_faces < 1 or num_faces > 200000:
                            print(f"Warning: Skipping mesh '{original_filename}' - invalid face count: {num_faces}")
                            continue

                        # OBJファイル名を生成（拡張子を.objに変更）
                        mesh_filename = os.path.splitext(original_filename)[0] + '.obj'
                        dest_mesh_path = os.path.join(assets_dir, mesh_filename)

                        # OBJ形式でエクスポート（MuJoCoとの互換性向上）
                        mesh.export(dest_mesh_path, file_type='obj')
                        print(f"Converted mesh: {original_filename} -> {mesh_filename} ({num_faces} faces)")

                    except Exception as e:
                        print(f"Warning: Could not process mesh '{original_filename}': {e}")
                        print(f"Skipping this mesh file.")
                        continue

                    # MJCF内で使用するファイル名（meshdirで参照）
                    node_to_mesh[node] = mesh_filename

                    # メッシュ名を生成（拡張子なし）
                    mesh_name = os.path.splitext(mesh_filename)[0]
                    mesh_names[node] = mesh_name
                    mesh_counter += 1

            # 作成されたジョイントのリストを追跡
            created_joints = []

            # base_linkを探す
            base_node = self.get_node_by_name('base_link')

            # 1. ロボット本体ファイル（{dir_name}.xml）を作成
            robot_file_path = os.path.join(mjcf_dir, f"{dir_name}.xml")
            self._write_mjcf_robot_file(robot_file_path, dir_name, base_node, mesh_names, node_to_mesh, created_joints)

            # 2. scene.xml を作成（ロボットファイルをinclude）
            scene_path = os.path.join(mjcf_dir, "scene.xml")
            self._write_mjcf_scene(scene_path, dir_name)

            print(f"MJCF export completed: {robot_file_path}")
            print(f"Total mesh files copied: {len(node_to_mesh)}")

            QtWidgets.QMessageBox.information(
                self.widget,
                "Export Successful",
                f"MJCF files have been exported successfully:\n\n"
                f"{mjcf_dir}/\n"
                f"├─ {dir_name}.xml (robot)\n"
                f"├─ scene.xml\n"
                f"└─ assets/ ({len(node_to_mesh)} mesh files)"
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

    def _write_mjcf_robot_file(self, file_path, model_name, base_node, mesh_names, node_to_mesh, created_joints):
        """ロボット本体ファイルを作成（すべての要素を含む単一ファイル）"""
        with open(file_path, 'w') as f:
            # ヘッダー
            f.write(f'<mujoco model="{model_name}">\n')

            # compiler設定
            f.write('  <compiler angle="radian" meshdir="assets" autolimits="true" />\n\n')

            # option設定
            f.write('  <option cone="elliptic" impratio="100" />\n\n')

            # default設定
            f.write('  <default>\n')
            f.write('    <geom friction="0.4" margin="0.001" condim="3"/>\n')
            f.write('    <joint damping="0.1" armature="0.01" frictionloss="0.2"/>\n')
            f.write('    <motor ctrlrange="-23.7 23.7"/>\n')
            f.write('  </default>\n\n')

            # asset設定（メッシュとマテリアル）
            f.write('  <asset>\n')
            f.write('    <material name="metal" rgba=".9 .95 .95 1" />\n')
            f.write('    <material name="black" rgba="0 0 0 1" />\n')
            f.write('    <material name="white" rgba="1 1 1 1" />\n')
            f.write('    <material name="gray" rgba="0.671705 0.692426 0.774270 1" />\n\n')

            # メッシュ定義
            for node in self.all_nodes():
                if node in node_to_mesh and node in mesh_names:
                    mesh_filename = node_to_mesh[node]
                    mesh_name = mesh_names[node]
                    f.write(f'    <mesh name="{mesh_name}" file="{mesh_filename}" />\n')

            f.write('  </asset>\n\n')

            # worldbody
            f.write('  <worldbody>\n')
            if base_node:
                visited_nodes = set()
                self._write_mjcf_body(f, base_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=4)
            f.write('  </worldbody>\n\n')

            # actuator
            if created_joints:
                f.write('  <actuator>\n')
                for joint_info in created_joints:
                    joint_name = joint_info['joint_name']
                    motor_name = joint_info['motor_name']
                    effort = joint_info.get('effort', 10.0)
                    actuation_lag = joint_info.get('actuation_lag', 0.0)

                    # MJCF mapping: gear=1, forcerange=[-effort, +effort], ctrlrange=forcerange
                    gear = 1.0
                    forcerange = f"-{effort} {effort}"
                    ctrlrange = forcerange

                    # dynprm for actuation lag (first-order delay model)
                    if actuation_lag > 0:
                        dynprm = f' dynprm="{actuation_lag} 0 0"'
                    else:
                        dynprm = ''

                    f.write(f'    <motor name="{motor_name}" joint="{joint_name}" gear="{gear}" forcerange="{forcerange}" ctrlrange="{ctrlrange}"{dynprm}/>\n')
                f.write('  </actuator>\n\n')

            # sensor
            f.write('  <sensor>\n')
            f.write('    <!-- Add sensors here if needed -->\n')
            f.write('  </sensor>\n')

            f.write('</mujoco>\n')
        print(f"Created robot file: {file_path}")

    def _write_mjcf_scene(self, file_path, robot_name):
        """scene.xmlを作成（ロボットファイルをinclude）"""
        with open(file_path, 'w') as f:
            f.write(f'<mujoco model="{robot_name} scene">\n')
            f.write(f'  <include file="{robot_name}.xml"/>\n\n')
            f.write('  <statistic center="0 0 0.1" extent="0.8"/>\n\n')
            f.write('  <visual>\n')
            f.write('    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n')
            f.write('    <rgba haze="0.15 0.25 0.35 1"/>\n')
            f.write('    <global azimuth="-130" elevation="-20"/>\n')
            f.write('  </visual>\n\n')
            f.write('  <asset>\n')
            f.write('    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n')
            f.write('    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n')
            f.write('      markrgb="0.8 0.8 0.8" width="300" height="300"/>\n')
            f.write('    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n')
            f.write('  </asset>\n\n')
            f.write('  <worldbody>\n')
            f.write('    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n')
            f.write('    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n')
            f.write('  </worldbody>\n')
            f.write('</mujoco>\n')
        print(f"Created scene file: {file_path}")

    def _write_mjcf_defaults(self, file_path):
        """defaults.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <default>\n')
            f.write('    <joint damping="0.1" armature="0.01" />\n')
            f.write('    <geom contype="1" conaffinity="1" condim="3" friction="0.9 0.1 0.1" />\n')
            f.write('    <motor ctrllimited="true" />\n')
            f.write('  </default>\n')
            f.write('</mujoco>\n')
        print(f"Created defaults file: {file_path}")

    def _write_mjcf_body_file(self, file_path, base_node, mesh_names, node_to_mesh, created_joints):
        """body.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <worldbody>\n')

            if base_node:
                visited_nodes = set()
                self._write_mjcf_body(f, base_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=4)

            f.write('  </worldbody>\n')
            f.write('</mujoco>\n')
        print(f"Created body file: {file_path}")

    def _write_mjcf_actuators(self, file_path, created_joints):
        """actuators.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <actuator>\n')

            for joint_info in created_joints:
                joint_name = joint_info['joint_name']
                motor_name = joint_info['motor_name']
                effort = joint_info.get('effort', 10.0)
                actuation_lag = joint_info.get('actuation_lag', 0.0)

                # MJCF mapping: gear=1, forcerange=[-effort, +effort], ctrlrange=forcerange
                gear = 1.0
                forcerange = f"-{effort} {effort}"
                ctrlrange = forcerange

                # dynprm for actuation lag (first-order delay model)
                if actuation_lag > 0:
                    dynprm = f' dynprm="{actuation_lag} 0 0"'
                else:
                    dynprm = ''

                f.write(f'    <motor name="{motor_name}" joint="{joint_name}" gear="{gear}" forcerange="{forcerange}" ctrlrange="{ctrlrange}"{dynprm} />\n')

            f.write('  </actuator>\n')
            f.write('</mujoco>\n')
        print(f"Created actuators file: {file_path}")

    def _write_mjcf_sensors(self, file_path):
        """sensors.xmlを作成"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <sensor>\n')
            f.write('    <!-- Add sensors here if needed -->\n')
            f.write('  </sensor>\n')
            f.write('</mujoco>\n')
        print(f"Created sensors file: {file_path}")

    def _write_mjcf_materials(self, file_path, node_to_mesh, mesh_names):
        """assets/materials.xmlを作成（メッシュ定義を含む）"""
        with open(file_path, 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<mujoco>\n')
            f.write('  <asset>\n')

            # メッシュ定義
            for node in self.all_nodes():
                if node in node_to_mesh and node in mesh_names:
                    mesh_path = node_to_mesh[node]
                    # ファイル名のみ抽出（meshdir="assets/meshes"を指定しているため）
                    mesh_filename = os.path.basename(mesh_path)
                    mesh_name = mesh_names[node]
                    f.write(f'    <mesh name="{mesh_name}" file="{mesh_filename}" />\n')

            f.write('  </asset>\n')
            f.write('</mujoco>\n')
        print(f"Created materials file: {file_path}")

    def _write_mjcf_body(self, file, node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent=2, joint_info=None):
        """MJCF bodyを再帰的に出力"""
        if node in visited_nodes:
            return
        visited_nodes.add(node)

        # Massless Decorationノードはスキップ
        if hasattr(node, 'massless_decoration') and node.massless_decoration:
            return

        # Hide Meshにチェックが入っているノードはスキップ
        if hasattr(node, 'hide_mesh') and node.hide_mesh:
            print(f"Skipping node with hide_mesh=True: {node.name()}")
            return

        indent_str = ' ' * indent

        # base_linkの場合は特別処理
        if node.name() == 'base_link':
            sanitized_name = self._sanitize_name(node.name())

            # base_linkボディを開始（freejointを持つ）
            file.write(f'{indent_str}<body name="{sanitized_name}" pos="0 0 0.5">\n')

            # 慣性プロパティ（freejointを持つbodyには必須）
            MIN_MASS = 0.001
            MIN_INERTIA = 1e-6

            if hasattr(node, 'mass_value') and node.mass_value > 0:
                mass = max(node.mass_value, MIN_MASS)
                if hasattr(node, 'inertia'):
                    ixx = max(abs(node.inertia.get('ixx', MIN_INERTIA)), MIN_INERTIA)
                    iyy = max(abs(node.inertia.get('iyy', MIN_INERTIA)), MIN_INERTIA)
                    izz = max(abs(node.inertia.get('izz', MIN_INERTIA)), MIN_INERTIA)
                else:
                    # 慣性が設定されていない場合はデフォルト値を使用
                    ixx = iyy = izz = MIN_INERTIA
            else:
                # 質量が設定されていない場合はデフォルト値を使用
                mass = MIN_MASS
                ixx = iyy = izz = MIN_INERTIA

            # base_linkには常に慣性を出力（freejointのため必須）
            file.write(f'{indent_str}  <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}" />\n')

            # freejoint（base_linkは自由に動ける）
            file.write(f'{indent_str}  <freejoint />\n')

            # ジオメトリ（メッシュ）
            if node in mesh_names:
                mesh_name = mesh_names[node]
                color_str = "0.8 0.8 0.8 1.0"
                if hasattr(node, 'node_color') and node.node_color:
                    r, g, b = node.node_color[:3]
                    color_str = f"{r} {g} {b} 1.0"

                # collider_meshが設定されている場合は、visualとcollisionを分離
                if hasattr(node, 'collider_mesh') and node.collider_mesh:
                    # Visual mesh (contype=0, conaffinity=0で衝突しない)
                    file.write(f'{indent_str}  <geom type="mesh" mesh="{mesh_name}" rgba="{color_str}" contype="0" conaffinity="0" />\n')
                    # Collision mesh (別途定義 - ここでは仮のmesh名を使用。実際にはmesh_namesに追加する必要がある)
                    # TODO: collider meshをmesh_namesに登録する処理が必要
                    collider_mesh_name = f"{mesh_name}_collision"
                    file.write(f'{indent_str}  <geom type="mesh" mesh="{collider_mesh_name}" />\n')
                else:
                    # collider_meshが設定されていない場合は通常通り（visual=collision）
                    file.write(f'{indent_str}  <geom type="mesh" mesh="{mesh_name}" rgba="{color_str}" />\n')

            # base_linkの子ノードを処理
            for port in node.output_ports():
                for connected_port in port.connected_ports():
                    child_node = connected_port.node()

                    # Massless Decorationノードはスキップ（ジョイント情報も作成しない）
                    if hasattr(child_node, 'massless_decoration') and child_node.massless_decoration:
                        continue

                    # Hide Meshにチェックが入っているノードはスキップ（ジョイント情報も作成しない）
                    if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                        continue

                    port_index = list(node.output_ports()).index(port)
                    child_joint_info = self._get_joint_info(node, child_node, port_index, created_joints)
                    self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent + 2, child_joint_info)

            # base_linkボディを閉じる
            file.write(f'{indent_str}</body>\n')
            return

        # 名前をサニタイズ
        sanitized_name = self._sanitize_name(node.name())

        # body開始（joint_infoがあれば位置情報を追加）
        pos_attr = f' pos="{joint_info["pos"]}"' if joint_info else ''
        file.write(f'{indent_str}<body name="{sanitized_name}"{pos_attr}>\n')

        # ジョイントを最初に出力（子ボディ内に配置）
        # body posで親からのオフセットを設定済みなので、joint posは常に"0 0 0"（body座標系の原点）
        if joint_info:
            file.write(f'{indent_str}  <joint name="{joint_info["name"]}" type="{joint_info["type"]}" pos="0 0 0" axis="{joint_info["axis"]}"{joint_info["range"]}{joint_info["damping"]}{joint_info["friction"]}{joint_info["stiffness"]} />\n')

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

                # Inertial Origin（COM位置）を取得
                com_pos = "0 0 0"
                if hasattr(node, 'inertial_origin') and isinstance(node.inertial_origin, dict):
                    xyz = node.inertial_origin.get('xyz', [0.0, 0.0, 0.0])
                    com_pos = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                file.write(f'{indent_str}  <inertial pos="{com_pos}" mass="{mass}" diaginertia="{ixx} {iyy} {izz}" />\n')

        # ジオメトリ（メッシュ）
        if node in mesh_names:
            mesh_name = mesh_names[node]
            # 色情報
            color_str = "0.8 0.8 0.8 1.0"
            if hasattr(node, 'node_color') and node.node_color:
                r, g, b = node.node_color[:3]
                color_str = f"{r} {g} {b} 1.0"

            # collider_meshが設定されている場合は、visualとcollisionを分離
            if hasattr(node, 'collider_mesh') and node.collider_mesh:
                # Visual mesh (contype=0, conaffinity=0で衝突しない)
                file.write(f'{indent_str}  <geom type="mesh" mesh="{mesh_name}" rgba="{color_str}" contype="0" conaffinity="0" />\n')
                # Collision mesh (別途定義 - ここでは仮のmesh名を使用。実際にはmesh_namesに追加する必要がある)
                # TODO: collider meshをmesh_namesに登録する処理が必要
                collider_mesh_name = f"{mesh_name}_collision"
                file.write(f'{indent_str}  <geom type="mesh" mesh="{collider_mesh_name}" />\n')
            else:
                # collider_meshが設定されていない場合は通常通り（visual=collision）
                file.write(f'{indent_str}  <geom type="mesh" mesh="{mesh_name}" rgba="{color_str}" />\n')

        # 子ノードを処理
        for port_index, port in enumerate(node.output_ports()):
            for connected_port in port.connected_ports():
                child_node = connected_port.node()

                # Massless Decorationノードの場合、ビジュアル要素として追加してからスキップ
                if hasattr(child_node, 'massless_decoration') and child_node.massless_decoration:
                    if child_node in mesh_names:
                        dec_mesh_name = mesh_names[child_node]
                        # 色情報
                        dec_color_str = "0.8 0.8 0.8 1.0"
                        if hasattr(child_node, 'node_color') and child_node.node_color:
                            r, g, b = child_node.node_color[:3]
                            dec_color_str = f"{r} {g} {b} 1.0"

                        # ポイント座標を取得（装飾パーツの位置）
                        pos_str = "0 0 0"
                        if hasattr(node, 'points') and port_index < len(node.points):
                            point_data = node.points[port_index]
                            if 'xyz' in point_data:
                                xyz = point_data['xyz']
                                pos_str = f"{xyz[0]} {xyz[1]} {xyz[2]}"

                        # ビジュアル専用geomとして追加（contype=0で衝突なし）
                        file.write(f'{indent_str}  <geom type="mesh" mesh="{dec_mesh_name}" rgba="{dec_color_str}" pos="{pos_str}" contype="0" conaffinity="0" />\n')
                    continue

                # Hide Meshにチェックが入っているノードはスキップ（ジョイント情報も作成しない）
                if hasattr(child_node, 'hide_mesh') and child_node.hide_mesh:
                    continue

                # 子ノードのジョイント情報を取得（port_indexは既にenumerateで取得済み）
                child_joint_info = self._get_joint_info(node, child_node, port_index, created_joints)

                # 再帰的に子ボディを出力
                self._write_mjcf_body(file, child_node, visited_nodes, mesh_names, node_to_mesh, created_joints, indent + 2, child_joint_info)

        # body終了
        file.write(f'{indent_str}</body>\n')

    def _get_joint_info(self, parent_node, child_node, port_index, created_joints):
        """ジョイント情報を取得して辞書として返す"""
        joint_xyz = [0, 0, 0]
        joint_axis = [1, 0, 0]

        # ジョイント位置を取得
        if hasattr(parent_node, 'points') and port_index < len(parent_node.points):
            point_data = parent_node.points[port_index]
            joint_xyz = point_data.get('xyz', [0, 0, 0])

        # ジョイント軸を取得
        if hasattr(child_node, 'rotation_axis'):
            if child_node.rotation_axis == 0:
                joint_axis = [1, 0, 0]
            elif child_node.rotation_axis == 1:
                joint_axis = [0, 1, 0]
            elif child_node.rotation_axis == 2:
                joint_axis = [0, 0, 1]

        # ジョイントタイプ
        joint_type = "hinge"
        if hasattr(child_node, 'rotation_axis') and child_node.rotation_axis == 3:
            joint_type = "fixed"

        # fixedジョイントの場合はNoneを返す（出力しない）
        if joint_type == "fixed":
            return None

        # ジョイント名
        child_sanitized_name = self._sanitize_name(child_node.name())
        joint_name = f"{child_sanitized_name}_joint"
        motor_name = f"{child_sanitized_name}_motor"

        # ジョイント制限
        range_str = ""
        if hasattr(child_node, 'joint_lower') and hasattr(child_node, 'joint_upper'):
            range_str = f' range="{child_node.joint_lower} {child_node.joint_upper}"'

        # ジョイントダンピング（MJCF: joint.damping）
        damping_str = ""
        if hasattr(child_node, 'damping'):
            damping_str = f' damping="{child_node.damping}"'

        # ジョイント摩擦（MJCF: joint.frictionloss）
        friction_str = ""
        if hasattr(child_node, 'friction'):
            friction_str = f' frictionloss="{child_node.friction}"'

        # ジョイントスティフネス（MJCF: joint.stiffness）
        stiffness_str = ""
        if hasattr(child_node, 'stiffness'):
            stiffness_str = f' stiffness="{child_node.stiffness}"'

        # 作成されたジョイントをリストに追加（actuator用）
        joint_effort = getattr(child_node, 'joint_effort', 10.0)
        joint_velocity = getattr(child_node, 'joint_velocity', 3.0)
        actuation_lag = getattr(child_node, 'actuation_lag', 0.0)
        created_joints.append({
            'joint_name': joint_name,
            'motor_name': motor_name,
            'effort': joint_effort,
            'velocity': joint_velocity,
            'actuation_lag': actuation_lag
        })

        return {
            'name': joint_name,
            'type': joint_type,
            'pos': f"{joint_xyz[0]} {joint_xyz[1]} {joint_xyz[2]}",
            'axis': f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}",
            'range': range_str,
            'damping': damping_str,
            'friction': friction_str,
            'stiffness': stiffness_str
        }

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

            # 3DビューからSTLメッシュを削除
            if hasattr(graph, 'stl_viewer') and graph.stl_viewer:
                if node in graph.stl_viewer.stl_actors:
                    actor = graph.stl_viewer.stl_actors[node]
                    graph.stl_viewer.renderer.RemoveActor(actor)
                    del graph.stl_viewer.stl_actors[node]
                    print(f"Removed STL mesh for node: {node.name()}")
                    # 3Dビューを更新
                    graph.stl_viewer.render_to_image()

            # グラフからノードを削除
            graph.remove_node(node)
        print(f"Deleted {len(selected_nodes)} node(s)")
    else:
        print("No node selected for deletion")

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
        # Ctrl+Cシグナルハンドラの設定（カスタムシャットダウンロジックのため個別実装）
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app = QtWidgets.QApplication(sys.argv)
        setup_dark_theme(app, theme='assembler')

        # アプリケーション終了時のクリーンアップ設定
        app.aboutToQuit.connect(cleanup_and_exit)

        # シグナル処理用タイマー（utils関数使用）
        timer = setup_signal_processing_timer(app)

        # メインウィンドウの作成
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("URDF Kitchen - Assembler - v0.0.1")
        main_window.resize(1200, 600)

        # セントラルウィジェットの設定
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # STLビューアとグラフの設定（先に作成）
        stl_viewer = STLViewerWidget(central_widget)
        stl_viewer.setMinimumWidth(100)  # 3Dビューの最小幅を100pxに設定
        graph = CustomNodeGraph(stl_viewer)
        stl_viewer.graph = graph  # STLビューアにグラフへの参照を設定
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
            "Import URDF": None,
            "Import MJCF": None,
            "--spacer2--": None,  # スペーサー用のダミーキー
            "Add Node": None,
            "Delete Node": None,
            "Build r_ from l_": None,
            "Recalc Positions": None,
            "--spacer3--": None,  # スペーサー用のダミーキー
            "Load Project": None,
            "Save Project": None,
            "--spacer4--": None,  # スペーサー用のダミーキー
            "Export URDF": None,
            "Export for Unity": None,
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
        buttons["Build r_ from l_"].clicked.connect(
            graph.build_r_from_l)
        buttons["Recalc Positions"].clicked.connect(
            graph.recalculate_all_positions)
        buttons["Save Project"].clicked.connect(graph.save_project)
        buttons["Load Project"].clicked.connect(lambda: load_project(graph))
        buttons["Import URDF"].clicked.connect(lambda: graph.import_urdf())
        buttons["Import MJCF"].clicked.connect(lambda: graph.import_mjcf())
        buttons["Export URDF"].clicked.connect(lambda: graph.export_urdf())
        buttons["Export for Unity"].clicked.connect(graph.export_for_unity)
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
