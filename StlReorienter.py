import sys
import signal
import vtk
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QLabel, QGridLayout
)
from PySide6.QtCore import QTimer, Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# python Version: 3.11
# numpy Version: 2.1.1
# PyQt5 Version: 5.15.11
# pvtk　Version: 9.3.1

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        super(CustomInteractorStyle, self).__init__()
        self.parent = parent
        self.AddObserver("CharEvent", self.on_char_event)
        self.AddObserver("KeyPressEvent", self.on_key_press)

    def on_char_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "f":
            print("Fキーが無効化されました。")
        elif key == "t":
            print("Tキーが押されました。ワイヤーフレーム表示を切り替えます。")
            self.toggle_wireframe()
        elif key == "r":
            print("Rキーが押されました。カメラをリセットします。")
            if self.parent:
                self.parent.reset_camera()
        elif key == "a":
            print("Aキーが押されました。左に90度回転します。")
            if self.parent:
                self.parent.rotate_camera(90, 'yaw')
        elif key == "d":
            print("Dキーが押されました。右に90度回転します。")
            if self.parent:
                self.parent.rotate_camera(-90, 'yaw')
        elif key == "w":
            print("Wキーが押されました。上に90度回転します。")
            if self.parent:
                self.parent.rotate_camera(-90, 'pitch')
        elif key == "s":
            print("Sキーが押されました。下に90度回転します。")
            if self.parent:
                self.parent.rotate_camera(90, 'pitch')
        elif key == "q":
            print("Qキーが押されました。反時計回りに90度回転します。")
            if self.parent:
                self.parent.rotate_camera(90, 'roll')
        elif key == "e":
            print("Eキーが押されました。時計回りに90度回転します。")
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
            if not actor.GetUserTransform():
                if actor.GetProperty().GetRepresentation() == vtk.VTK_SURFACE:
                    actor.GetProperty().SetRepresentationToWireframe()
                else:
                    actor.GetProperty().SetRepresentationToSurface()
            actor = actors.GetNextItem()
        self.GetInteractor().GetRenderWindow().Render()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Reorienter")
        self.setGeometry(100, 100, 700, 700)
        self.camera_rotation = [0, 0, 0]  # [yaw, pitch, roll]
        self.absolute_origin = [0, 0, 0]  # 大原点の設定
        self.initial_camera_position = [10, 0, 0]  # 初期カメラ位置
        self.initial_camera_focal_point = [0, 0, 0]  # 初期焦点
        self.initial_camera_view_up = [0, 0, 1]  # 初期の上方向

        self.num_points = 1  # ポイントの数を8に設定
        self.point_coords = [list(self.absolute_origin)
                             for _ in range(self.num_points)]
        self.point_actors = [None] * self.num_points
        self.point_checkboxes = []
        self.point_inputs = []

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
        self.load_button = QPushButton("Load STL File")
        self.load_button.clicked.connect(self.load_stl_file)
        button_layout.addWidget(self.load_button)

        self.export_stl_button = QPushButton("Save as reoriented STL")
        self.export_stl_button.clicked.connect(self.export_stl_with_new_origin)
        button_layout.addWidget(self.export_stl_button)

        main_layout.addLayout(button_layout)

        # STL表示画面
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        main_layout.addWidget(self.vtk_widget)

        # Volume表示
        self.volume_label = QLabel("Volume (m^3): 0.000000")
        main_layout.addWidget(self.volume_label)

        # Points UI
        self.setup_points_ui(main_layout)

        self.setup_vtk()

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
        self.add_axes()

        for i in range(self.num_points):
            self.show_point(i)

        self.add_instruction_text()

        self.render_window.Render()
        self.render_window_interactor.Initialize()

        self.vtk_widget.GetRenderWindow().AddObserver(
            "ModifiedEvent", self.update_all_points_size)

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
            checkbox = QCheckBox(f"Point {i+1}")
            checkbox.setChecked(True)
            self.point_checkboxes.append(checkbox)
            points_layout.addWidget(checkbox, i, 0)

            inputs = []
            for j, axis in enumerate(['X', 'Y', 'Z']):
                input_field = QLineEdit(str(self.point_coords[i][j]))
                inputs.append(input_field)
                points_layout.addWidget(QLabel(f"{axis}:"), i, j*2+1)
                points_layout.addWidget(input_field, i, j*2+2)
            self.point_inputs.append(inputs)

        layout.addLayout(points_layout)

        # SET と RESET ボタン
        button_layout = QHBoxLayout()
        set_button = QPushButton("SET")
        set_button.clicked.connect(self.handle_set_reset)
        button_layout.addWidget(set_button)

        reset_button = QPushButton("RESET")
        reset_button.clicked.connect(self.handle_set_reset)
        button_layout.addWidget(reset_button)

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
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        self.render_window_interactor = self.render_window.GetInteractor()

        style = CustomInteractorStyle(self)
        self.render_window_interactor.SetInteractorStyle(style)

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
                self.point_coords[i] = list(self.absolute_origin)  # 原点にリセット
                self.update_point_display(i)
                print(f"Point {i+1} reset to origin {self.absolute_origin}")

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(self.absolute_origin[0] + self.initial_camera_position[0],
                           self.absolute_origin[1] +
                           self.initial_camera_position[1],
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

        print(
            f"Point {index+1} moved to: ({new_pos[0]:.4f}, {new_pos[1]:.4f}, {current_z:.4f})")

    def update_point_display(self, index):
        # 入力フィールドに座標を反映
        if self.point_actors[index]:
            self.point_actors[index].SetPosition(self.point_coords[index])

        # インプットフィールドを更新
        for j, coord in enumerate(self.point_coords[index]):
            self.point_inputs[index][j].setText(f"{coord:.4f}")

        self.render_window.Render()

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
        if self.point_actors[index]:
            self.point_actors[index].SetPosition(self.point_coords[index])
        for i, coord in enumerate(self.point_coords[index]):
            self.point_inputs[index][i].setText(f"{coord:.4f}")
        self.render_window.Render()

    def update_all_points_size(self, obj=None, event=None):
        for index, actor in enumerate(self.point_actors):
            if actor:
                self.renderer.RemoveActor(actor)
                self.point_actors[index] = vtk.vtkAssembly()
                self.create_point_coordinate(self.point_actors[index], [
                                             0, 0, 0])  # ローカル座標系で作成
                self.point_actors[index].SetPosition(
                    self.point_coords[index])  # 世界座標系で位置を設定
                self.renderer.AddActor(self.point_actors[index])
        self.render_window.Render()

    def update_all_points(self):
        if self.point1_actor:
            self.update_point1_size()

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
                print(
                    f"Added {['X', 'Y', 'Z'][i]} axis {'positive' if direction == 1 else 'negative'}")

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

            # 円のプロパティを設定
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
        camera = self.renderer.GetActiveCamera()
        parallel_scale = camera.GetParallelScale()

        # ビューポートのアスペクト比を取得
        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / \
            (viewport[3] - viewport[1])

        # 画角の10%のサイズを計算（画角の高さの5%を半径とする）
        radius = parallel_scale * 0.05

        # アスペクト比を考慮して、幅と高さの小さい方に合わせる
        if aspect_ratio > 1:
            radius /= aspect_ratio

        return radius

    def calculate_properties(self):
        try:
            volume = float(self.volume_label.text().split(': ')[1])

            # 密度はデフォルト値を使用するか、ユーザー入力を受け付けるUIを追加する必要があります
            density = 1.0  # デフォルト値、必要に応じて変更してください

            mass = volume * density

            # 慣性モーメントの計算（簡略化：立方体と仮定）
            side_length = np.cbrt(volume)
            inertia = (1/6) * mass * side_length**2

            print(f"Volume: {volume:.6f} m^3")
            print(f"Density: {density:.6f} kg/m^3")
            print(f"Mass: {mass:.6f} kg")
            print(f"Inertia: {inertia:.6f} kg·m^2")

        except ValueError:
            print(
                "Error calculating properties. Please ensure the volume is a valid number.")

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
        for actor in self.renderer.GetActors():
            if actor != self.stl_actor and actor != self.point1_actor:
                self.renderer.RemoveActor(actor)

        axis_length = 5  # 固定の軸の長さ
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

    def add_instruction_text(self):
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
        text_actor_top.GetTextProperty().SetColor(0.0, 0.8, 0.8)
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
            "Notice: Press [R] before saving and rotate as needed in 90-degree increments:\n"
            "top of the screen = +Z, right of the screen = +Y, front = +X, with the purple point as the origin."
        )
        text_actor_bottom.GetTextProperty().SetFontSize(14)
        text_actor_bottom.GetTextProperty().SetColor(0.0, 0.8, 0.8)
        text_actor_bottom.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor_bottom.SetPosition(0.03, 0.03)  # 左下に配置
        text_actor_bottom.GetTextProperty().SetJustificationToLeft()
        text_actor_bottom.GetTextProperty().SetVerticalJustificationToBottom()
        self.renderer.AddActor(text_actor_bottom)

    def load_stl_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open STL File", "", "STL Files (*.stl)")
        if file_path:
            self.file_name_label.setText(f"File: {file_path}")
            self.show_stl(file_path)
        self.reset_camera()
        self.update_all_points_size()

    def show_stl(self, file_path):
        if hasattr(self, 'stl_actor') and self.stl_actor:
            self.renderer.RemoveActor(self.stl_actor)

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

        # 体積をUIに反映（小数点以下6桁）
        self.volume_label.setText(f"Volume (m^3): {volume:.6f}")

        # カメラのフィッティングと描画更新
        self.fit_camera_to_model()
        self.update_all_points()

        # 境界ボックスを出力
        print(f"STLモデルの境界ボックス: [{self.model_bounds[0]:.4f}, {self.model_bounds[1]:.4f}], [{self.model_bounds[2]:.4f}, {self.model_bounds[3]:.4f}], [{self.model_bounds[4]:.4f}, {self.model_bounds[5]:.4f}]")

        # 大原点を表示
        self.show_absolute_origin()

        # ファイル名をフルパスで更新
        self.file_name_label.setText(f"File: {file_path}")

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
        if self.point_actors[index] is None:
            self.point_actors[index] = vtk.vtkAssembly()
            self.create_point_coordinate(
                self.point_actors[index], self.point_coords[index])
            self.renderer.AddActor(self.point_actors[index])
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

    def update_all_points(self):
        for i in range(self.num_points):
            if self.point_actors[i]:
                self.update_point_display(i)

    def fit_camera_to_model(self):
        if not self.model_bounds:
            return

        camera = self.renderer.GetActiveCamera()

        # モデルの中心を計算
        center = [(self.model_bounds[i] + self.model_bounds[i+1]) /
                  2 for i in range(0, 6, 2)]

        # モデルの大きさを計算
        size = max([
            self.model_bounds[1] - self.model_bounds[0],
            self.model_bounds[3] - self.model_bounds[2],
            self.model_bounds[5] - self.model_bounds[4]
        ])

        # 20%の余裕を追加
        size *= 1.4  # 1.0 + 0.2 + 0.2 = 1.4

        # カメラの位置を設定 (x軸方向から見る)
        camera.SetPosition(center[0] + size, center[1], center[2])
        camera.SetFocalPoint(*center)  # モデルの中心を見る
        camera.SetViewUp(0, 0, 1)  # z軸を上向きに

        # ビューポートのアスペクト比を取得
        viewport = self.renderer.GetViewport()
        aspect_ratio = (viewport[2] - viewport[0]) / \
            (viewport[3] - viewport[1])

        # モデルが画面にフィットするようにパラレルスケールを設定
        if aspect_ratio > 1:  # 横長の画面
            camera.SetParallelScale(size / 2)
        else:  # 縦長の画面
            camera.SetParallelScale(size / (2 * aspect_ratio))

        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def on_mouse_wheel(self, obj, event):
        self.update_point1_size()

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
            vertical_axis = 'z' if abs(np.dot(screen_up, [0, 0, 1])) > abs(
                np.dot(screen_up, [0, 1, 0])) else 'y'
        else:
            horizontal_axis = 'z'
            vertical_axis = 'y'

        return horizontal_axis, vertical_axis, screen_right, screen_up

    def export_stl_with_new_origin(self):
        if not self.stl_actor or not any(self.point_actors):
            print("STLモデルまたはポイントが設定されていません。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save STL File", "", "STL Files (*.stl)")
        if not file_path:
            return

        try:
            # 現在のSTLモデルのポリデータを取得
            poly_data = self.stl_actor.GetMapper().GetInput()

            # 最初に選択されているポイントを新しい原点として使用
            origin_index = next(i for i, actor in enumerate(
                self.point_actors) if actor and actor.GetVisibility())
            origin_point = self.point_coords[origin_index]

            # Step 1: 選択されたポイントを原点とする平行移動
            translation = vtk.vtkTransform()
            translation.Translate(-origin_point[0], -
                                  origin_point[1], -origin_point[2])

            # Step 2: 画面の向きに基づいた新しい座標系を作成
            camera = self.renderer.GetActiveCamera()

            # 画面手前方向（X軸正方向）
            new_x = np.array(camera.GetDirectionOfProjection())
            new_x = -new_x / np.linalg.norm(new_x)

            # 画面上方向（Z軸正方向）
            new_z = np.array(camera.GetViewUp())
            new_z = new_z / np.linalg.norm(new_z)

            # 画面右方向（Y軸正方向）
            new_y = np.cross(new_z, new_x)
            new_y = new_y / np.linalg.norm(new_y)

            # 新しい座標系の回転行列を作成
            rotation_matrix = np.column_stack((new_x, new_y, new_z))

            # VTKの回転行列に変換
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    vtk_matrix.SetElement(i, j, rotation_matrix[i, j])

            # 回転変換を作成
            rotation = vtk.vtkTransform()
            rotation.SetMatrix(vtk_matrix)

            # 変換を組み合わせる
            transform = vtk.vtkTransform()
            transform.PostMultiply()
            transform.Concatenate(translation)
            transform.Concatenate(rotation)

            # 変換フィルターを適用
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(poly_data)
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            # 変換されたポリデータを取得
            transformed_poly_data = transform_filter.GetOutput()

            # STLファイルとして保存
            stl_writer = vtk.vtkSTLWriter()
            stl_writer.SetFileName(file_path)
            stl_writer.SetInputData(transformed_poly_data)
            stl_writer.Write()

            print(f"新しい座標系でSTLファイルを保存しました: {file_path}")

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")

    def handle_set_reset(self):
        sender = self.sender()
        is_set = sender.text() == "SET"

        for i, checkbox in enumerate(self.point_checkboxes):
            if checkbox.isChecked():
                if is_set:
                    try:
                        new_coords = [
                            float(self.point_inputs[i][j].text()) for j in range(3)]
                        if new_coords != self.point_coords[i]:
                            self.point_coords[i] = new_coords
                            self.update_point_display(i)
                            print(f"Point {i+1} set to: {new_coords}")
                        else:
                            print(f"Point {i+1} coordinates unchanged")
                    except ValueError:
                        print(
                            f"Invalid input for Point {i+1}. Please enter valid numbers.")
                else:  # Reset
                    self.reset_point_to_origin(i)

        if not is_set:
            self.update_all_points_size()

        self.render_window.Render()


def signal_handler(sig, frame):
    print("Ctrl+C detected, closing application...")
    QApplication.instance().quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Ctrl+Cのシグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)

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
