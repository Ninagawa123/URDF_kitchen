"""
File Name: urdf_kitchen_Assembler.py
Description: A Python script to assembling files configured with urdf_kitchen_PartsEditor.py into a URDF file.

Author      : Ninagawa123
Created On  : Nov 24, 2024
Version     : 0.0.2
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
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QPointF, QRegularExpression
from PySide6.QtGui import QDoubleValidator, QRegularExpressionValidator, QPalette, QColor
import os
import xml.etree.ElementTree as ET
import base64
import shutil
import datetime

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

        self.volume_value = 0.0  # 追加
        self.mass_value = 0.0
        
        self.inertia = {
            'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
            'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
        }
        self.points = [{
            'name': 'base_link_point1',
            'type': 'fixed',
            'xyz': [0.0, 0.0, 0.0]
        }]
        self.cumulative_coords = [{
            'point_index': 0,
            'xyz': [0.0, 0.0, 0.0]
        }]

        self.stl_file = None

        # 色情報を追加
        self.node_color = [1.0, 1.0, 1.0]  # RGBの初期値（白）

        # 回転軸の初期値を追加
        self.rotation_axis = 0  # 0: X, 1: Y, 2: Z

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

class FooNode(BaseNode):
    """General purpose node class"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'FooNode'
    
    def __init__(self):
        super(FooNode, self).__init__()
        self.add_input('in', color=(180, 80, 0))
        
        self.output_count = 0
        self.volume_value = 0.0  # 追加
        self.mass_value = 0.0
        
        # データ属性を初期化
        self.mass_value = 0.0
        self.inertia = {
            'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
            'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
        }
        self.points = []
        self.cumulative_coords = []
        self.stl_file = None
        
        # 色情報を追加
        self.node_color = [1.0, 1.0, 1.0]  # RGBの初期値（白）
        
        # 回転軸の初期値を追加
        self.rotation_axis = 0  # 0: X, 1: Y, 2: Z
        
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
            
            # 新しいポイントを[0,0,0]の座標で追加
            self.points.append({
                'name': f'point_{self.output_count}',
                'type': 'fixed',
                'xyz': [0.0, 0.0, 0.0]
            })
            
            # 累積座標の初期化
            if not hasattr(self, 'cumulative_coords'):
                self.cumulative_coords = []
                
            self.cumulative_coords.append({
                'point_index': self.output_count - 1,
                'xyz': [0.0, 0.0, 0.0]
            })
            
            print(f"Added output port '{port_name}' with zero coordinates")
            return port_name

    def remove_output(self):
        """出力ポートの削除（修正版）"""
        if self.output_count > 1:
            port_name = f'out_{self.output_count}'
            output_port = self.get_output(port_name)
            if output_port:
                try:
                    # 接続されているポートを処理
                    for connected_port in output_port.connected_ports():
                        try:
                            print(f"Disconnecting {port_name} from {connected_port.node().name()}.{connected_port.name()}")
                            # NodeGraphQtの標準メソッドを使用
                            self.graph.disconnect_node(self.id, port_name,
                                                     connected_port.node().id, connected_port.name())
                        except Exception as e:
                            print(f"Error during disconnection: {str(e)}")

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
        self.setMinimumWidth(400)
        self.setMinimumHeight(600)

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
        content_layout.setSpacing(30)  # セクション間の間隔を小さく
        content_layout.setContentsMargins(5, 5, 5, 5)  # 余白を小さく

        # Node Name セクション（横一列）
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Node Name:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Enter node name")
        self.name_edit.editingFinished.connect(self.update_node_name)
        name_layout.addWidget(self.name_edit)

        content_layout.addLayout(name_layout)
        content_layout.addSpacing(5)  # 追加の間隔

        # Physical Properties セクション（テキストを削除して詰める）
        physics_layout = QtWidgets.QGridLayout()
        physics_layout.addWidget(QtWidgets.QLabel("Volume:"), 0, 0)
        self.volume_input = QtWidgets.QLineEdit()
        self.volume_input.setReadOnly(True)
        physics_layout.addWidget(self.volume_input, 0, 1)

        physics_layout.addWidget(QtWidgets.QLabel("Mass:"), 1, 0)
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setValidator(QtGui.QDoubleValidator())
        physics_layout.addWidget(self.mass_input, 1, 1)
        content_layout.addLayout(physics_layout)

        # Rotation Axis セクション（横一列）
        rotation_layout = QtWidgets.QHBoxLayout()
        rotation_layout.addWidget(QtWidgets.QLabel("Rotation Axis:   "))
        self.axis_group = QtWidgets.QButtonGroup(self)
        for i, axis in enumerate(['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']):  # Fixedを追加
            radio = QtWidgets.QRadioButton(axis)
            self.axis_group.addButton(radio, i)  # iは0,1,2,3となる（3がFixed）
            rotation_layout.addWidget(radio)
        content_layout.addLayout(rotation_layout)

        # Rotation Testボタンの配置を修正
        rotation_test_layout = QtWidgets.QHBoxLayout()
        rotation_test_layout.addStretch()  # 左側に伸縮可能なスペースを追加
        self.rotation_test_button = QtWidgets.QPushButton("Rotation Test")
        self.rotation_test_button.setFixedWidth(120)  # 他のボタンと同じ幅に設定
        self.rotation_test_button.pressed.connect(self.start_rotation_test)
        self.rotation_test_button.released.connect(self.stop_rotation_test) 
        rotation_test_layout.addWidget(self.rotation_test_button)
        content_layout.addLayout(rotation_test_layout)

        # Massless Decorationチェックボックスの追加
        massless_layout = QtWidgets.QHBoxLayout()
        self.massless_checkbox = QtWidgets.QCheckBox("Massless Decoration")
        self.massless_checkbox.setChecked(False)  # デフォルトはオフ
        massless_layout.addWidget(self.massless_checkbox)
        content_layout.addLayout(massless_layout)

        # チェックボックスの状態変更時のハンドラを接続
        self.massless_checkbox.stateChanged.connect(self.update_massless_decoration)

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

        # SETボタンレイアウト
        set_button_layout = QtWidgets.QHBoxLayout()
        set_button_layout.addStretch()
        set_button = QtWidgets.QPushButton("SET")
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
                    self.stl_viewer.vtkWidget.GetRenderWindow().Render()
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

            # Color settings - nodeのnode_color属性を確認して設定
            if hasattr(node, 'node_color') and node.node_color:
                print(f"Setting color: {node.node_color}")
                for i, value in enumerate(node.node_color[:3]):
                    self.color_inputs[i].setText(f"{value:.3f}")
                
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
                node.node_color = [1.0, 1.0, 1.0]
                for color_input in self.color_inputs:
                    color_input.setText("1.000")
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
                        self.stl_viewer.vtkWidget.GetRenderWindow().Render()
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
            for i, value in enumerate(rgb_values):
                self.color_inputs[i].setText(f"{value:.3f}")
            
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
                self, "Open STL File", "", "STL Files (*.stl)")
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
                        print("Set inertia tensor")

            # 色情報の処理
            material_elem = root.find('.//material/color')
            if material_elem is not None:
                rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                rgb_values = [float(x) for x in rgba[:3]]  # RGBのみ使用
                self.current_node.node_color = rgb_values
                # 色の設定をUIに反映
                for i, value in enumerate(rgb_values):
                    self.color_inputs[i].setText(f"{value}")
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
                    self.current_node.cumulative_coords.append({
                        'point_index': i,
                        'xyz': [0.0, 0.0, 0.0]
                    })

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

                # 質量の設定
                mass_elem = link_elem.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value', '0.0'))
                    self.current_node.mass_value = mass
                    self.mass_input.setText(f"{mass:.6f}")

                # ボリュームの設定
                volume_elem = root.find('.//volume')
                if volume_elem is not None:
                    volume = float(volume_elem.get('value', '0.0'))
                    self.current_node.volume_value = volume
                    self.volume_input.setText(f"{volume:.6f}")

                # 慣性モーメントの設定
                inertia_elem = link_elem.find('inertia')
                if inertia_elem is not None:
                    self.current_node.inertia = {
                        'ixx': float(inertia_elem.get('ixx', '0')),
                        'ixy': float(inertia_elem.get('ixy', '0')),
                        'ixz': float(inertia_elem.get('ixz', '0')),
                        'iyy': float(inertia_elem.get('iyy', '0')),
                        'iyz': float(inertia_elem.get('iyz', '0')),
                        'izz': float(inertia_elem.get('izz', '0'))
                    }

            # 色情報の処理
            material_elem = root.find('.//material/color')
            if material_elem is not None:
                rgba = material_elem.get('rgba', '1.0 1.0 1.0 1.0').split()
                rgb_values = [float(x) for x in rgba[:3]]  # RGBのみ使用
                self.current_node.node_color = rgb_values
                # 色の設定をUIに反映
                for i, value in enumerate(rgb_values):
                    self.color_inputs[i].setText(f"{value}")
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
                # 余分なポートを削除
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
                        self, "Select STL File", xml_dir, "STL Files (*.stl)")
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
        """Output Portsの値を適用"""
        if not self.current_node:
            print("No node selected")
            return

        try:
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

            # ノードの位置を再計算（必要な場合）
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()
                print("Node positions recalculated")

            # STLビューアの更新
            if self.stl_viewer:
                self.stl_viewer.vtkWidget.GetRenderWindow().Render()
                print("3D view updated")

        except Exception as e:
            print(f"Error applying port values: {str(e)}")
            import traceback
            traceback.print_exc()

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
                    self.stl_viewer.vtkWidget.GetRenderWindow().Render()
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
            # 現在の変換を保存
            self.stl_viewer.store_current_transform(self.current_node)
            # 回転開始
            self.stl_viewer.start_rotation_test(self.current_node)

    def stop_rotation_test(self):
        """回転テスト終了"""
        if self.current_node and self.stl_viewer:
            # 回転停止と元の角度に戻す
            self.stl_viewer.stop_rotation_test(self.current_node)

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


class STLViewerWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(STLViewerWidget, self).__init__(parent)
        self.stl_actors = {}
        self.transforms = {}
        self.base_connected_node = None
        self.text_actors = []

        layout = QtWidgets.QVBoxLayout(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        # インタラクタースタイルを設定
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        # ボタンとスライダーのレイアウト
        button_layout = QtWidgets.QVBoxLayout()  # 垂直レイアウトに変更

        # Reset Angleボタン
        self.reset_button = QtWidgets.QPushButton("Reset Angle")
        self.reset_button.clicked.connect(self.reset_camera)
        button_layout.addWidget(self.reset_button)

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
        
        self.iren.Initialize()

    def store_current_transform(self, node):
        """現在の変換を保存"""
        if node in self.transforms:
            current_transform = vtk.vtkTransform()
            current_transform.DeepCopy(self.transforms[node])
            self.original_transforms[node] = current_transform

    def start_rotation_test(self, node):
        """回転テスト開始"""
        if node in self.stl_actors:
            self.rotating_node = node
            self.current_angle = 0
            self.rotation_timer.start(16)  # 約60FPS

    def stop_rotation_test(self, node):
        """回転テスト終了"""
        self.rotation_timer.stop()
        
        # 元の色と変換を必ず復元
        if self.rotating_node in self.stl_actors:
            # 元の色を復元（必ず実行）
            if hasattr(self.rotating_node, 'node_color'):
                self.stl_actors[self.rotating_node].GetProperty().SetColor(*self.rotating_node.node_color)
            
            # 元の変換を復元
            if self.rotating_node in self.original_transforms:
                self.transforms[self.rotating_node].DeepCopy(self.original_transforms[self.rotating_node])
                self.stl_actors[self.rotating_node].SetUserTransform(self.transforms[self.rotating_node])
                del self.original_transforms[self.rotating_node]
            
            self.vtkWidget.GetRenderWindow().Render()
        
        self.rotating_node = None

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
                # 通常の回転処理
                transform.Identity()  # 変換をリセット
                transform.Translate(position)  # 元の位置を維持
                
                # 回転軸に基づいて回転
                self.current_angle += 5  # 1フレームあたりの回転角度
                if hasattr(node, 'rotation_axis'):
                    if node.rotation_axis == 0:    # X軸
                        transform.RotateX(self.current_angle)
                    elif node.rotation_axis == 1:  # Y軸
                        transform.RotateY(self.current_angle)
                    elif node.rotation_axis == 2:  # Z軸
                        transform.RotateZ(self.current_angle)
                
                self.stl_actors[node].SetUserTransform(transform)
                
            self.vtkWidget.GetRenderWindow().Render()
            self.current_angle += 1

    def reset_camera(self):
        """カメラビューをリセットし、すべてのSTLモデルをビューに収める"""
        if not self.renderer.GetActors().GetNumberOfItems():
            self.setup_camera()
            return

        # すべてのアクターの合計バウンディングボックスを計算
        bounds = [float('inf'), float('-inf'), 
                float('inf'), float('-inf'), 
                float('inf'), float('-inf')]
        
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextActor()
        while actor:
            actor_bounds = actor.GetBounds()
            # X軸の最小値と最大値
            bounds[0] = min(bounds[0], actor_bounds[0])
            bounds[1] = max(bounds[1], actor_bounds[1])
            # Y軸の最小値と最大値
            bounds[2] = min(bounds[2], actor_bounds[2])
            bounds[3] = max(bounds[3], actor_bounds[3])
            # Z軸の最小値と最大値
            bounds[4] = min(bounds[4], actor_bounds[4])
            bounds[5] = max(bounds[5], actor_bounds[5])
            actor = actors.GetNextActor()

        # バウンディングボックスの中心を計算
        center = [(bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2]

        # バウンディングボックスの対角線の長さを計算
        diagonal = ((bounds[1] - bounds[0]) ** 2 +
                (bounds[3] - bounds[2]) ** 2 +
                (bounds[5] - bounds[4]) ** 2) ** 0.5

        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        
        # カメラの位置を設定（バウンディングボックスの対角線の2倍の距離）
        distance = diagonal
        camera.SetPosition(center[0] + distance, center[1], center[2])
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        
        # パラレルスケールを設定してビューに収める
        camera.SetParallelScale(diagonal * 0.5)

        # クリッピング範囲を更新
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

        print("Camera reset complete - All STL models fitted to view")

    def reset_view_to_fit(self):
        """すべてのSTLモデルが見えるようにビューをリセットして調整"""
        self.reset_camera()
        self.vtkWidget.GetRenderWindow().Render()

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
        
        self.vtkWidget.GetRenderWindow().Render()

    def update_stl_transform(self, node, point_xyz):
        """STLの位置を更新"""
        # base_linkの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            return

        if node in self.stl_actors and node in self.transforms:
            print(f"Updating transform for node {node.name()} to position {point_xyz}")
            transform = self.transforms[node]
            transform.Identity()
            transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])
            
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

            self.vtkWidget.GetRenderWindow().Render()
        else:
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No STL actor or transform found for node {node.name()}")

    def reset_stl_transform(self, node):
        """STLの位置をリセット"""
        # base_linkの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
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
            
            self.vtkWidget.GetRenderWindow().Render()
        else:
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No transform found for node {node.name()}")

    def load_stl_for_node(self, node):
        """ノード用のSTLファイルを読み込む（色の適用を含む）"""
        # base_linkの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            return

        if node.stl_file:
            print(f"Loading STL for node: {node.name()}, file: {node.stl_file}")
            reader = vtk.vtkSTLReader()
            reader.SetFileName(node.stl_file)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            transform = vtk.vtkTransform()
            transform.Identity()
            actor.SetUserTransform(transform)

            # 既存のアクターを削除
            if node in self.stl_actors:
                self.renderer.RemoveActor(self.stl_actors[node])

            self.stl_actors[node] = actor
            self.transforms[node] = transform
            self.renderer.AddActor(actor)

            # ノードの色情報を適用
            self.apply_color_to_node(node)

            self.reset_camera()
            self.vtkWidget.GetRenderWindow().Render()
            print(f"STL file loaded and rendered: {node.stl_file}")

    def apply_color_to_node(self, node):
        """ノードのSTLモデルに色を適用"""
        if node in self.stl_actors:
            # デフォルトの色を設定（色情報がない場合）
            if not hasattr(node, 'node_color') or node.node_color is None:
                node.node_color = [1.0, 1.0, 1.0]  # 白色をデフォルトに

            # 色の適用
            actor = self.stl_actors[node]
            actor.GetProperty().SetColor(*node.node_color)
            print(f"Applied color to node {node.name()}: RGB({node.node_color[0]:.3f}, {node.node_color[1]:.3f}, {node.node_color[2]:.3f})")
            self.vtkWidget.GetRenderWindow().Render()

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
                
            self.vtkWidget.GetRenderWindow().Render()
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

        # インタラクターの終了
        if hasattr(self, 'iren'):
            if self.iren:
                self.iren.TerminateApp()

        # レンダーウィンドウのクリーンアップ
        if hasattr(self, 'vtkWidget'):
            if self.vtkWidget:
                self.vtkWidget.close()

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
                self.vtkWidget.GetRenderWindow().Render()
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
        self.vtkWidget.GetRenderWindow().Render()

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

        # ノードタイプの登録
        self.register_node(BaseLinkNode)
        self.register_node(FooNode)

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

        # ビューの設定
        self._view = self.widget

        # ラバーバンドの作成
        self._rubber_band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle,
            self._view
        )

        # オリジナルのイベントハンドラを保存
        self._original_handlers = {
            'press': self._view.mousePressEvent,
            'move': self._view.mouseMoveEvent,
            'release': self._view.mouseReleaseEvent
        }

        # 新しいイベントハンドラを設定
        self._view.mousePressEvent = self.custom_mouse_press
        self._view.mouseMoveEvent = self.custom_mouse_move
        self._view.mouseReleaseEvent = self.custom_mouse_release

        # インスペクタウィンドウの初期化
        self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)

    def custom_mouse_press(self, event):
        """カスタムマウスプレスイベントハンドラ"""
        try:
            # 左ボタンの処理
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._selection_start = event.position().toPoint()
                self._is_selecting = True

                # ラバーバンドの設定
                if self._rubber_band:
                    rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                    self._rubber_band.setGeometry(rect)
                    self._rubber_band.show()

                # Ctrlキーが押されていない場合は選択をクリア
                if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                    for node in self.selected_nodes():
                        node.set_selected(False)

            # オリジナルのイベントハンドラを呼び出し
            self._original_handlers['press'](event)

        except Exception as e:
            print(f"Error in mouse press: {str(e)}")

    def custom_mouse_move(self, event):
        """カスタムマウス移動イベントハンドラ"""
        try:
            if self._is_selecting and self._selection_start:
                current_pos = event.position().toPoint()
                rect = QtCore.QRect(self._selection_start,
                                    current_pos).normalized()
                if self._rubber_band:
                    self._rubber_band.setGeometry(rect)

            # オリジナルのイベントハンドラを呼び出し
            self._original_handlers['move'](event)

        except Exception as e:
            print(f"Error in mouse move: {str(e)}")

    def custom_mouse_release(self, event):
        """カスタムマウスリリースイベントハンドラ"""
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self._is_selecting:
                if self._rubber_band and self._selection_start:
                    # 選択範囲の処理
                    rect = self._rubber_band.geometry()
                    scene_rect = self._view.mapToScene(rect).boundingRect()

                    # 範囲内のノードを選択
                    for node in self.all_nodes():
                        node_pos = node.pos()
                        if isinstance(node_pos, (list, tuple)):
                            node_point = QtCore.QPointF(
                                node_pos[0], node_pos[1])
                        else:
                            node_point = node_pos

                        if scene_rect.contains(node_point):
                            node.set_selected(True)

                    # ラバーバンドを隠す
                    self._rubber_band.hide()

                # 選択状態をリセット
                self._selection_start = None
                self._is_selecting = False

            # オリジナルのイベントハンドラを呼び出し
            self._original_handlers['release'](event)

        except Exception as e:
            print(f"Error in mouse release: {str(e)}")

    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            print("Starting cleanup process...")
            
            # イベントハンドラの復元
            if hasattr(self, '_view') and self._view:
                if hasattr(self, '_original_handlers'):
                    self._view.mousePressEvent = self._original_handlers['press']
                    self._view.mouseMoveEvent = self._original_handlers['move']
                    self._view.mouseReleaseEvent = self._original_handlers['release']
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
        if self._is_selecting and self._selection_start is not None and self._rubber_band:
            current_pos = event.position().toPoint()
            rect = QtCore.QRect(self._selection_start,
                                current_pos).normalized()
            self._rubber_band.setGeometry(rect)

        return False  # イベントを伝播させる

    def _handle_mouse_release(self, event):
        """マウスリリースイベントの処理"""
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
                
                QtWidgets.QMessageBox.information(
                    self.widget,
                    "Export Complete",
                    f"URDF file has been exported to:\n{urdf_file}"
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
        file.write('  <link name="base_link">\n')
        file.write('    <inertial>\n')
        file.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
        file.write('      <mass value="0.0"/>\n')
        file.write('      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>\n')
        file.write('    </inertial>\n')
        file.write('  </link>\n\n')

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
                except:
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

            # 色情報
            if hasattr(node, 'node_color'):
                color_elem = ET.SubElement(node_elem, "color")
                color_elem.text = ' '.join(map(str, node.node_color))
                print(f"  Saved color: {node.node_color}")

            # 回転軸
            if hasattr(node, 'rotation_axis'):
                ET.SubElement(node_elem, "rotation_axis").text = str(node.rotation_axis)
                print(f"  Saved rotation axis: {node.rotation_axis}")

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
                self.stl_viewer.vtkWidget.GetRenderWindow().Render()

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
                self.stl_viewer.vtkWidget.GetRenderWindow().Render()
            
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
                    new_node.node_color = [1.0, 1.0, 1.0]
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
                        new_node.cumulative_coords.append({
                            'point_index': len(new_node.points) - 1,
                            'xyz': [0.0, 0.0, 0.0]
                        })
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
                self.stl_viewer.vtkWidget.GetRenderWindow().Render()
            
            print("Position recalculation completed")

        except Exception as e:
            print(f"Error during position recalculation: {str(e)}")
            traceback.print_exc()

    def _recalculate_node_positions(self, node, parent_coords, visited):
        """再帰的にノードの位置を計算"""
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
                        point_xyz = point_data['xyz']
                        
                        # 新しい位置を計算
                        new_position = [
                            parent_coords[0] + point_xyz[0],
                            parent_coords[1] + point_xyz[1],
                            parent_coords[2] + point_xyz[2]
                        ]
                        
                        print(f"Child node: {child_node.name()}")
                        print(f"Point data: {point_xyz}")
                        print(f"Calculated position: {new_position}")
                        
                        # STL位置を更新
                        self.stl_viewer.update_stl_transform(child_node, new_position)
                        
                        # 子ノードの累積座標を更新
                        if hasattr(child_node, 'cumulative_coords'):
                            for coord in child_node.cumulative_coords:
                                coord['xyz'] = new_position.copy()
                        
                        # 再帰的に子ノードを処理
                        self._recalculate_node_positions(child_node, new_position, visited)
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
                    file.write('    <limit lower="-3.14159" upper="3.14159" effort="0" velocity="0"/>\n')
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
def signal_handler(signum, frame):
    print("\nCtrl+C pressed. Closing all windows and exiting...")
    shutdown()

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

def cleanup_and_exit():
    """アプリケーションのクリーンアップと終了"""
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

        # その他のリソースのクリーンアップ
        for window in QtWidgets.QApplication.topLevelWidgets():
            try:
                window.close()
            except Exception as e:
                print(f"Error closing window: {str(e)}")

    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
    finally:
        # アプリケーションの終了
        if QtWidgets.QApplication.instance():
            QtWidgets.QApplication.instance().quit()

def signal_handler(signum, frame):
    """Ctrl+Cシグナルのハンドラ"""
    print("\nCtrl+C detected, closing application...")
    try:
        # アプリケーションのクリーンアップと終了
        if QtWidgets.QApplication.instance():
            # 全てのウィンドウを閉じる
            for window in QtWidgets.QApplication.topLevelWidgets():
                try:
                    window.close()
                except:
                    pass

            # アプリケーションの終了
            QtWidgets.QApplication.instance().quit()
    except Exception as e:
        print(f"Error during application shutdown: {str(e)}")
    finally:
        # 強制終了
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
            "Export URDF": None,
            "Export for Unity": None,
            "--spacer5--": None,  # スペーサー用のダミーキー
            "open urdf-loaders": None
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
        buttons["Export URDF"].clicked.connect(lambda: graph.export_urdf())
        buttons["Export for Unity"].clicked.connect(graph.export_for_unity)
        buttons["open urdf-loaders"].clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl(
                    "https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/")
            )
        )









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
        
        # スプリッターの設定
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(graph.widget)
        splitter.addWidget(stl_viewer)
        splitter.setSizes([800, 400])

        # メインレイアウトの設定
        main_layout.addWidget(left_panel)
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
