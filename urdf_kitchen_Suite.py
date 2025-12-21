"""
File Name: urdf_kitchen_Suite.py
Description: Integrated UI for URDF Kitchen tools (StlSourcer, PartsEditor, Assembler) with tab interface.

Author      : yosh617
Created On  : Dec 21, 2025
Version     : 0.1.0
License     : MIT License
URL         : https://github.com/yosh617/URDF_kitchen
Copyright (c) 2025 yosh617

pip install numpy
pip install PySide6
pip install vtk
pip install NodeGraphQt
"""

import sys
import signal
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QStatusBar
from PySide6.QtCore import QTimer, QObject, Signal, Qt
from PySide6.QtGui import QPalette, QColor


# Import the three tools
try:
    from urdf_kitchen_StlSourcer import MainWidget as StlSourcerWidget
except ImportError:
    StlSourcerWidget = None
    print("Warning: Could not import StlSourcer")

try:
    from urdf_kitchen_PartsEditor import MainWidget as PartsEditorWidget
except ImportError:
    PartsEditorWidget = None
    print("Warning: Could not import PartsEditor")

try:
    from urdf_kitchen_Assembler import MainWidget as AssemblerWidget
except ImportError:
    AssemblerWidget = None
    print("Warning: Could not import Assembler")


def apply_dark_theme(app):
    """統合ダークテーマの適用"""
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


class FileEventBus(QObject):
    """タブ間のファイル連携用イベントバス"""
    # PartsEditorからの保存完了シグナル
    file_saved = Signal(str, str)  # (stl_path, xml_path)
    # Assemblerへのリロード要求シグナル
    reload_request = Signal(str)  # (file_path)
    # ステータス通知シグナル
    status_message = Signal(str)  # (message)


class URDFKitchenSuite(QMainWindow):
    """URDF Kitchen 統合メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("URDF Kitchen Suite v0.1.0")
        
        # 画面サイズに応じた適切なウィンドウサイズを設定
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        # 画面の80%のサイズを使用
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        # 画面中央に配置
        x = int((screen.width() - width) / 2)
        y = int((screen.height() - height) / 2)
        self.setGeometry(x, y, width, height)
        
        # 最小サイズを設定
        self.setMinimumSize(1000, 600)
        
        # クリーンアップフラグ
        self._cleaned_up = False
        
        # クリーンアップ対象ウィジェットのリスト
        self.cleanup_widgets = []
        
        # イベントバスの作成
        self.event_bus = FileEventBus()
        
        # タブウィジェットの作成
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # ステータスバーの設定
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.event_bus.status_message.connect(self.show_status_message)
        
        # 各ツールウィジェットの初期化
        self.sourcer_widget = None
        self.parts_editor_widget = None
        self.assembler_widget = None
        
        # タブの作成
        self.init_tabs()
        
        # タブ切り替え時のイベント処理
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # クリーンアップ用のリスト
        self.cleanup_widgets = []
        
    def init_tabs(self):
        """各ツールをタブとして初期化"""
        
        # StlSourcer タブ
        if StlSourcerWidget:
            try:
                self.sourcer_widget = StlSourcerWidget(event_bus=self.event_bus)
                self.tabs.addTab(self.sourcer_widget, "1. STL Sourcer")
                self.cleanup_widgets.append(self.sourcer_widget)
                print("StlSourcer tab initialized")
            except Exception as e:
                print(f"Error initializing StlSourcer: {e}")
                import traceback
                traceback.print_exc()
        
        # PartsEditor タブ
        if PartsEditorWidget:
            try:
                self.parts_editor_widget = PartsEditorWidget(event_bus=self.event_bus)
                self.tabs.addTab(self.parts_editor_widget, "2. Parts Editor")
                self.cleanup_widgets.append(self.parts_editor_widget)
                print("PartsEditor tab initialized")
                
                # ファイル保存シグナルを接続
                if hasattr(self.parts_editor_widget, 'file_saved'):
                    self.parts_editor_widget.file_saved.connect(self.on_file_saved_from_editor)
            except Exception as e:
                print(f"Error initializing PartsEditor: {e}")
                import traceback
                traceback.print_exc()
        
        # Assembler タブ
        if AssemblerWidget:
            try:
                self.assembler_widget = AssemblerWidget(event_bus=self.event_bus, parent=self)
                self.tabs.addTab(self.assembler_widget, "3. Assembler")
                self.cleanup_widgets.append(self.assembler_widget)
                print("Assembler tab initialized")
            except Exception as e:
                print(f"Error initializing Assembler: {e}")
                import traceback
                traceback.print_exc()
    
    def on_tab_changed(self, index):
        """タブ切り替え時の処理"""
        # タブ名を取得してステータスバーに表示
        tab_name = self.tabs.tabText(index)
        self.status_bar.showMessage(f"Switched to: {tab_name}", 2000)
    
    def on_file_saved_from_editor(self, stl_path, xml_path):
        """PartsEditorからファイル保存完了通知を受信"""
        # イベントバスを通じて通知
        self.event_bus.file_saved.emit(stl_path, xml_path)
        
        # ステータスバーに通知
        import os
        filename = os.path.basename(xml_path)
        self.show_status_message(f"Saved: {filename} - Ready to reload in Assembler", 5000)
        
        # Assemblerへのリロード要求（オプション）
        if self.assembler_widget and hasattr(self.assembler_widget, 'on_file_reload_request'):
            self.assembler_widget.on_file_reload_request(xml_path)
    
    def show_status_message(self, message, timeout=3000):
        """ステータスバーにメッセージを表示"""
        self.status_bar.showMessage(message, timeout)
    
    def cleanup(self):
        """アプリケーション終了時のクリーンアップ - VTKリソースを適切な順序で解放"""
        # 既にクリーンアップ済みの場合は何もしない
        if hasattr(self, '_cleaned_up') and self._cleaned_up:
            return
        
        self._cleaned_up = True
        
        # 1. 各ウィジェットのクリーンアップメソッドを呼び出し
        for widget in self.cleanup_widgets:
            if widget and hasattr(widget, 'cleanup'):
                try:
                    widget.cleanup()
                except (RuntimeError, AttributeError):
                    pass
        
        # 2. タブからウィジェットを削除してQt親子関係を切断
        try:
            while self.tabs.count() > 0:
                widget = self.tabs.widget(0)
                self.tabs.removeTab(0)
                if widget:
                    widget.setParent(None)
        except (RuntimeError, AttributeError):
            pass
        
        # 3. クリーンアップリストをクリア
        self.cleanup_widgets.clear()
    
    def closeEvent(self, event):
        """ウィンドウが閉じられるときの処理"""
        self.cleanup()
        event.accept()


def signal_handler(signum, frame):
    """Ctrl+Cシグナルのハンドラ"""
    print("\nCtrl+C detected, closing application...")
    try:
        # アプリケーションのクリーンアップと終了
        if QApplication.instance():
            # 全てのウィンドウを閉じる
            for window in QApplication.topLevelWidgets():
                try:
                    window.close()
                except:
                    pass
            
            # アプリケーションの終了
            QApplication.instance().quit()
    except Exception as e:
        print(f"Error during application shutdown: {e}")
    finally:
        # 強制終了
        sys.exit(0)


def main():
    """メイン関数"""
    try:
        # Ctrl+Cシグナルハンドラの設定
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # アプリケーションの作成
        app = QApplication(sys.argv)
        
        # ダークテーマの適用
        apply_dark_theme(app)
        
        # タイマーを設定してシグナルを処理できるようにする
        timer = QTimer()
        timer.start(500)  # 500ミリ秒ごとにイベントループを中断
        timer.timeout.connect(lambda: None)  # ダミー関数を接続
        
        # メインウィンドウの作成
        main_window = URDFKitchenSuite()
        
        # アプリケーション終了時のクリーンアップ設定
        app.aboutToQuit.connect(main_window.cleanup)
        
        # ウィンドウを表示
        main_window.show()
        
        print("URDF Kitchen Suite started")
        print("Use tabs to switch between tools:")
        print("  1. STL Sourcer - Reconfigure STL center and axes")
        print("  2. Parts Editor - Configure connection points")
        print("  3. Assembler - Assemble parts into URDF")
        print("Press Ctrl+C in the terminal to close all windows and exit.")
        
        # アプリケーションの実行
        sys.exit(app.exec())
        
    except SystemExit:
        print("Application exited")
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
