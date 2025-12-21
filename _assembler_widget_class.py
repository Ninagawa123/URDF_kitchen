# Assembler MainWidget クラス定義（urdf_kitchen_Assembler.pyの末尾に追加）

class MainWidget(QtWidgets.QWidget):
    """Assembler用のメインウィジェット（タブ統合用）"""
    
    def __init__(self, event_bus=None, parent=None):
        super().__init__(parent)
        # タブ統合用のイベントバス
        self.event_bus = event_bus
        
        # メインレイアウトの設定
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # STLビューアとグラフの設定（先に作成）
        self.stl_viewer = STLViewerWidget(self)
        self.graph = CustomNodeGraph(self.stl_viewer, parent=self)
        self.graph.setup_custom_view()
        
        # base_linkノードの作成
        base_node = self.graph.create_base_link()
        
        # 左パネルの設定
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(145)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # 名前入力フィールドの設定
        name_label = QtWidgets.QLabel("Name:")
        left_layout.addWidget(name_label)
        self.name_input = QtWidgets.QLineEdit("robot_x")
        self.name_input.setFixedWidth(120)
        self.name_input.setStyleSheet("QLineEdit { padding-left: 3px; padding-top: 0px; padding-bottom: 0px; }")
        left_layout.addWidget(self.name_input)
        
        # 名前入力フィールドとグラフを接続
        self.name_input.textChanged.connect(self.graph.update_robot_name)
        
        # ボタンの作成と設定
        buttons = {
            "--spacer1--": None,
            "Import XMLs": None,
            "Refresh": None,
            "Clear Nodes": None,
            "--spacer2--": None,
            "Add Node": None,
            "Delete Node": None,
            "Recalc Positions": None,
            "--spacer3--": None,
            "Load Project": None,
            "Save Project": None,
            "--spacer4--": None,
            "Export URDF": None,
            "Export for Unity": None,
            "--spacer5--": None,
            "open urdf-loaders": None
        }
        
        self.buttons = {}
        for button_text in buttons.keys():
            if button_text.startswith("--spacer"):
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(1)
                left_layout.addWidget(spacer)
            else:
                button = QtWidgets.QPushButton(button_text)
                button.setFixedWidth(120)
                left_layout.addWidget(button)
                self.buttons[button_text] = button
        
        left_layout.addStretch()
        
        # ボタンのコネクション設定
        self.buttons["Import XMLs"].clicked.connect(self.graph.import_xmls_from_folder)
        self.buttons["Refresh"].clicked.connect(self.graph.refresh_parts)
        self.buttons["Clear Nodes"].clicked.connect(self.graph.clear_all_nodes)
        self.buttons["Add Node"].clicked.connect(
            lambda: self.graph.create_node(
                'insilico.nodes.FooNode',
                name=f'Node_{len(self.graph.all_nodes())}',
                pos=QtCore.QPointF(0, 0)
            )
        )
        self.buttons["Delete Node"].clicked.connect(
            lambda: self.delete_selected_node())
        self.buttons["Recalc Positions"].clicked.connect(
            self.graph.recalculate_all_positions)
        self.buttons["Save Project"].clicked.connect(self.graph.save_project)
        self.buttons["Load Project"].clicked.connect(lambda: self.load_project())
        self.buttons["Export URDF"].clicked.connect(lambda: self.graph.export_urdf())
        self.buttons["Export for Unity"].clicked.connect(self.graph.export_for_unity)
        self.buttons["open urdf-loaders"].clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl(
                    "https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/")
            )
        )
        
        # スプリッターの設定
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.graph.widget)
        splitter.addWidget(self.stl_viewer)
        splitter.setSizes([800, 400])
        
        # メインレイアウトの設定
        main_layout.addWidget(left_panel)
        main_layout.addWidget(splitter)
        
        # グラフに名前入力フィールドを関連付け
        self.graph.name_input = self.name_input
        
        # イベントバスとの連携
        if self.event_bus:
            self.event_bus.file_saved.connect(self.on_file_reload_request)
        
        print("Assembler widget initialized")
    
    def delete_selected_node(self):
        """選択されたノードを削除"""
        selected_nodes = self.graph.selected_nodes()
        if selected_nodes:
            for node in selected_nodes:
                if isinstance(node, BaseLinkNode):
                    print("Cannot delete Base Link node")
                    continue
                self.graph.remove_node(node)
            print(f"Deleted {len(selected_nodes)} node(s)")
        else:
            print("No node selected for deletion")
    
    def load_project(self):
        """プロジェクトを読み込み"""
        from urdf_kitchen_Assembler import load_project as lp
        lp(self.graph)
    
    def on_file_reload_request(self, stl_path, xml_path):
        """PartsEditorからのファイル保存通知を受信してリロード"""
        print(f"File reload request received: {xml_path}")
        # 必要に応じて自動リロード処理を実装
        if self.event_bus:
            self.event_bus.status_message.emit(f"File updated: {os.path.basename(xml_path)}")
    
    def cleanup(self):
        """クリーンアップ処理"""
        try:
            if hasattr(self, 'graph'):
                self.graph.cleanup()
            if hasattr(self, 'stl_viewer'):
                self.stl_viewer.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")
