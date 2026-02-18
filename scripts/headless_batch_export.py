#!/usr/bin/env python3
"""
Headless batch wrapper for URDF_kitchen.

What it does:
- Imports a URDF/xacro/SDF model using URDF_kitchen importer logic
- Exports Unity package and MJCF package (via URDF_kitchen exporters)
- Creates normalized mesh export folders: stl/ and dae/

Output structure:
  <out_root>/
    stl/
    dae/
    unity/
    mjcf/
"""

import argparse
import shutil
import os
import sys
import traceback
from pathlib import Path

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Headless Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402
import trimesh  # noqa: E402

from urdf_kitchen_Assembler import STLViewerWidget, CustomNodeGraph  # noqa: E402
from urdf_kitchen_Importer import import_urdf, auto_detect_mesh_directories, search_stl_files_in_directory  # noqa: E402


class DialogPatcher:
    def __init__(self, input_model: Path, out_root: Path, mjcf_dir_name: str):
        self.input_model = str(input_model)
        self.out_root = out_root
        self.mjcf_dir_name = mjcf_dir_name

        self.orig_get_open = None
        self.orig_get_dir = None
        self.orig_dialog_exec = None
        self.orig_info = None
        self.orig_warn = None
        self.orig_critical = None
        self.orig_question = None
        self.orig_msg_exec = None
        self.orig_msg_exec_ = None

    def __enter__(self):
        self.orig_get_open = QtWidgets.QFileDialog.getOpenFileName
        self.orig_get_dir = QtWidgets.QFileDialog.getExistingDirectory
        self.orig_dialog_exec = QtWidgets.QDialog.exec
        self.orig_info = QtWidgets.QMessageBox.information
        self.orig_warn = QtWidgets.QMessageBox.warning
        self.orig_critical = QtWidgets.QMessageBox.critical
        self.orig_question = QtWidgets.QMessageBox.question
        self.orig_msg_exec = QtWidgets.QMessageBox.exec
        self.orig_msg_exec_ = QtWidgets.QMessageBox.exec_

        def fake_get_open(*args, **kwargs):
            # Always return the requested input model for import dialogs
            return self.input_model, ""

        def fake_get_dir(parent=None, caption="", dir="", options=None):
            c = (caption or "").lower()
            if "unity" in c:
                target = self.out_root / "unity"
            elif "mjcf" in c:
                target = self.out_root / "mjcf"
            else:
                # default/fallback
                target = self.out_root
            target.mkdir(parents=True, exist_ok=True)
            return str(target)

        def fake_dialog_exec(dialog_self):
            # Auto-accept export setting dialogs.
            # For MJCF dialog, set first QLineEdit (dir name) to deterministic value.
            try:
                edits = dialog_self.findChildren(QtWidgets.QLineEdit)
                if edits:
                    edits[0].setText(self.mjcf_dir_name)
            except Exception:
                pass
            return QtWidgets.QDialog.Accepted

        def log_box(kind):
            def _inner(parent, title, text, *args, **kwargs):
                print(f"[{kind}] {title}: {text}")
                return QtWidgets.QMessageBox.Ok
            return _inner

        QtWidgets.QFileDialog.getOpenFileName = staticmethod(fake_get_open)
        QtWidgets.QFileDialog.getExistingDirectory = staticmethod(fake_get_dir)
        QtWidgets.QDialog.exec = fake_dialog_exec
        QtWidgets.QMessageBox.information = staticmethod(log_box("INFO"))
        QtWidgets.QMessageBox.warning = staticmethod(log_box("WARN"))
        QtWidgets.QMessageBox.critical = staticmethod(log_box("ERROR"))
        QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No)
        QtWidgets.QMessageBox.exec = lambda self: QtWidgets.QMessageBox.Ok
        QtWidgets.QMessageBox.exec_ = lambda self: QtWidgets.QMessageBox.Ok
        return self

    def __exit__(self, exc_type, exc, tb):
        QtWidgets.QFileDialog.getOpenFileName = self.orig_get_open
        QtWidgets.QFileDialog.getExistingDirectory = self.orig_get_dir
        QtWidgets.QDialog.exec = self.orig_dialog_exec
        QtWidgets.QMessageBox.information = self.orig_info
        QtWidgets.QMessageBox.warning = self.orig_warn
        QtWidgets.QMessageBox.critical = self.orig_critical
        QtWidgets.QMessageBox.question = self.orig_question
        QtWidgets.QMessageBox.exec = self.orig_msg_exec
        QtWidgets.QMessageBox.exec_ = self.orig_msg_exec_


def ensure_dirs(out_root: Path):
    for d in ["stl", "dae", "unity", "mjcf"]:
        (out_root / d).mkdir(parents=True, exist_ok=True)


def export_mesh_variants(graph: CustomNodeGraph, out_root: Path):
    stl_dir = out_root / "stl"
    dae_dir = out_root / "dae"
    stl_dir.mkdir(parents=True, exist_ok=True)
    dae_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    converted = 0
    failed = 0

    for node in graph.all_nodes():
        mesh_path = getattr(node, "stl_file", None)
        if not mesh_path:
            continue
        p = Path(mesh_path)
        if not p.exists():
            continue

        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)

        stem = p.stem
        out_stl = stl_dir / f"{stem}.stl"
        out_dae = dae_dir / f"{stem}.dae"

        try:
            mesh = trimesh.load(str(p), force="mesh")
            mesh.export(str(out_stl))
            mesh.export(str(out_dae))
            converted += 1
        except Exception as e:
            print(f"[WARN] mesh conversion failed for {p}: {e}")
            failed += 1

    return converted, failed




def copy_original_model_to_output_folders(input_model: Path, out_root: Path):
    """Copy original submitted model file into each main output folder."""
    for d in ["stl", "dae", "unity", "mjcf"]:
        target_dir = out_root / d
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / input_model.name
        shutil.copy2(input_model, target)

def make_graph(app: QtWidgets.QApplication):
    container = QtWidgets.QWidget()
    viewer = STLViewerWidget(container)
    graph = CustomNodeGraph(viewer)
    viewer.graph = graph
    graph.setup_custom_view()
    graph.create_base_link()
    return graph


def run(input_model: Path, out_root: Path, robot_name: str, base_height: float):
    ensure_dirs(out_root)

    copy_original_model_to_output_folders(input_model, out_root)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    graph = make_graph(app)

    # Bind importer helper functions expected as graph methods in some versions
    graph.auto_detect_mesh_directories = auto_detect_mesh_directories
    graph.search_stl_files_in_directory = search_stl_files_in_directory

    # Set robot name if possible
    try:
        graph.robot_name = robot_name
        if hasattr(graph, "update_robot_name"):
            graph.update_robot_name(robot_name)
    except Exception:
        pass

    with DialogPatcher(
        input_model=input_model,
        out_root=out_root,
        mjcf_dir_name=f"{robot_name}_mjcf"
    ):
        # Import URDF/xacro/SDF through existing tool logic
        ok = import_urdf(graph)
        if not ok:
            raise RuntimeError("URDF import failed")

        # Keep deterministic default base link height for MJCF exporter
        if hasattr(graph, "default_base_link_height"):
            graph.default_base_link_height = base_height

        unity_ok = graph.export_for_unity()
        if not unity_ok:
            print("[WARN] Unity export returned False")

        mjcf_ok = graph.export_mjcf()
        if not mjcf_ok:
            print("[WARN] MJCF export returned False")

    converted, failed = export_mesh_variants(graph, out_root)

    print("\n=== Batch export summary ===")
    print(f"Input model : {input_model}")
    print(f"Output root : {out_root}")
    print(f"Robot name  : {robot_name}")
    print(f"Meshes ok   : {converted}")
    print(f"Meshes fail : {failed}")
    print("Folders:")
    print(f"  - {out_root / 'stl'}")
    print(f"  - {out_root / 'dae'}")
    print(f"  - {out_root / 'unity'}")
    print(f"  - {out_root / 'mjcf'}")


def main():
    parser = argparse.ArgumentParser(description="Headless batch export wrapper for URDF_kitchen")
    parser.add_argument("--input", required=True, help="Path to URDF/xacro/SDF file")
    parser.add_argument("--out", required=True, help="Output root directory")
    parser.add_argument("--robot-name", default="robot_batch", help="Robot name override")
    parser.add_argument("--base-height", type=float, default=0.01, help="MJCF base_link default height")
    args = parser.parse_args()

    input_model = Path(args.input).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not input_model.exists():
        print(f"ERROR: input file not found: {input_model}")
        sys.exit(2)

    try:
        run(input_model, out_root, args.robot_name, args.base_height)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
