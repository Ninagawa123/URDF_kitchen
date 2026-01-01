#!/usr/bin/env python3
"""
File Name: urdf_kitchen.py
Description: Small utility window to launch URDF Kitchen applications.

Author      : Ninagawa123
Created On  : Nov 28, 2024
Update.     : Jan  1, 2026
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
"""

import sys
import os
import subprocess
import signal
from PySide6 import QtWidgets, QtCore, QtGui


class URDFKitchenLauncher(QtWidgets.QWidget):
    """Main launcher window for URDF Kitchen applications"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        # Set window title
        self.setWindowTitle("URDF_kitchen")

        # Set window background style (matching PartsEditor)
        self.setStyleSheet("""
            QWidget {
                background-color: #404244;
            }
        """)

        # Create main layout
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(20, 20, 20, 20)

        # Button style matching PartsEditor (with gradient for 3D effect)
        button_style = """
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #5a5a5a, stop:1 #3a3a3a);
                color: #ffffff;
                border: 1px solid #707070;
                border-radius: 5px;
                padding: 2px 8px;
                margin: 1px 2px;
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
        """

        # Application buttons (vertical arrangement)
        mesh_sourcer_btn = QtWidgets.QPushButton("MeshSourcer")
        mesh_sourcer_btn.setStyleSheet(button_style)
        mesh_sourcer_btn.clicked.connect(self.launch_mesh_sourcer)
        layout.addWidget(mesh_sourcer_btn)

        parts_editor_btn = QtWidgets.QPushButton("PartsEditor")
        parts_editor_btn.setStyleSheet(button_style)
        parts_editor_btn.clicked.connect(self.launch_parts_editor)
        layout.addWidget(parts_editor_btn)

        assembler_btn = QtWidgets.QPushButton("Assembler")
        assembler_btn.setStyleSheet(button_style)
        assembler_btn.clicked.connect(self.launch_assembler)
        layout.addWidget(assembler_btn)

        # Add separator line (matching PartsEditor style)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator.setStyleSheet("""
            QFrame {
                color: #707070;
                background-color: #707070;
            }
        """)
        layout.addWidget(separator)

        # Instruction and Close buttons
        instruction_btn = QtWidgets.QPushButton("Instruction")
        instruction_btn.setStyleSheet(button_style)
        instruction_btn.clicked.connect(self.show_instruction)
        layout.addWidget(instruction_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setStyleSheet(button_style)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # Set layout
        self.setLayout(layout)

        # Set fixed size for the window
        self.setFixedSize(250, 300)

        # Position the window at the top-left corner
        self.position_top_left()

    def position_top_left(self):
        """Position the window at the top-left corner with margin"""
        screen = QtGui.QGuiApplication.primaryScreen()
        screen_rect = screen.availableGeometry()

        # Set margin from top and left edges (to avoid desktop menu)
        margin_top = 50
        margin_left = 50

        # Position the window
        self.move(screen_rect.left() + margin_left, screen_rect.top() + margin_top)

    def keyPressEvent(self, event):
        """Handle key press events"""
        # Close window with Ctrl+C
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            self.close()
        else:
            super().keyPressEvent(event)

    def launch_mesh_sourcer(self):
        """Launch MeshSourcer application"""
        script_path = os.path.join(os.path.dirname(__file__), "urdf_kitchen_MeshSourcer.py")
        self.launch_app(script_path, "MeshSourcer")

    def launch_parts_editor(self):
        """Launch PartsEditor application"""
        script_path = os.path.join(os.path.dirname(__file__), "urdf_kitchen_PartsEditor.py")
        self.launch_app(script_path, "PartsEditor")

    def launch_assembler(self):
        """Launch Assembler application"""
        script_path = os.path.join(os.path.dirname(__file__), "urdf_kitchen_Assembler.py")
        self.launch_app(script_path, "Assembler")

    def launch_app(self, script_path, app_name):
        """Launch a Python application"""
        if not os.path.exists(script_path):
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"{app_name} script not found:\n{script_path}"
            )
            return

        try:
            # Launch the application as a separate process
            subprocess.Popen([sys.executable, script_path])
            print(f"Launched {app_name}: {script_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch {app_name}:\n{str(e)}"
            )

    def show_instruction(self):
        """Show instruction message"""
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Instruction")
        msg_box.setText("Hello, this is URDF Kitchen Beta2. I’m currently in the middle of a major update. Instructions are also being prepared.")
        msg_box.setIcon(QtWidgets.QMessageBox.NoIcon)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)

        # Set white background with black text for the message box
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QLabel {
                color: black;
                background-color: white;
            }
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #5a5a5a, stop:1 #3a3a3a);
                color: white;
                border: 1px solid #707070;
                border-radius: 5px;
                padding: 5px 15px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #6a6a6a, stop:1 #4a4a4a);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #3a3a3a, stop:1 #5a5a5a);
            }
        """)

        msg_box.exec()


def main():
    """Main entry point"""
    app = QtWidgets.QApplication(sys.argv)

    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\nClosing URDF Kitchen Launcher...")
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)

    # Create a timer to allow Python to process signals
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    # Create and show launcher window
    launcher = URDFKitchenLauncher()
    launcher.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
