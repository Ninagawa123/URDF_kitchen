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
import os
import sys
import traceback
import re
import shutil
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


def _count_mesh_files(root: Path):
    if not root.exists() or not root.is_dir():
        return 0
    count = 0
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() in {'.stl', '.dae', '.obj', '.fbx', '.gltf', '.glb'}:
            count += 1
    return count


def _backfill_mesh_exports_from_source(input_model: Path, out_root: Path):
    """Populate stl/unity mesh folders from source `meshes/` when exporter outputs are sparse."""
    src_mesh_root = input_model.parent / 'meshes'
    if not src_mesh_root.exists() or not src_mesh_root.is_dir():
        alt_root = input_model.parent.parent / 'meshes'
        if alt_root.exists() and alt_root.is_dir():
            src_mesh_root = alt_root
        else:
            return {'backfilled': 0, 'failed': 0, 'source': str(src_mesh_root), 'used': False}

    stl_mesh_dir = out_root / 'stl' / 'meshes'
    dae_mesh_dir = out_root / 'dae' / 'meshes'
    unity_mesh_dir = out_root / 'unity' / 'meshes'
    for d in (stl_mesh_dir, dae_mesh_dir, unity_mesh_dir):
        d.mkdir(parents=True, exist_ok=True)

    backfilled = 0
    failed = 0

    for src in sorted(src_mesh_root.rglob('*')):
        if not src.is_file():
            continue
        ext = src.suffix.lower()
        if ext not in {'.obj', '.stl', '.dae'}:
            continue
        stem = src.stem
        try:
            mesh = trimesh.load(str(src), force='mesh')
            stl_path = stl_mesh_dir / f'{stem}.stl'
            dae_path = dae_mesh_dir / f'{stem}.dae'
            unity_dae_path = unity_mesh_dir / f'{stem}.dae'
            mesh.export(str(stl_path))
            mesh.export(str(dae_path))
            # Unity usually accepts DAE/FBX; keep DAE here for deterministic pathing.
            shutil.copy2(dae_path, unity_dae_path)
            backfilled += 1
        except Exception as e:
            print(f"[WARN] backfill conversion failed for {src}: {e}")
            failed += 1

    return {'backfilled': backfilled, 'failed': failed, 'source': str(src_mesh_root), 'used': True}


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




def _index_mesh_files(target_dir: Path):
    """Index mesh files under target_dir by (stem, extension) -> relative path."""
    mesh_map = {}
    for path in sorted(target_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in {".stl", ".dae", ".obj"}:
            continue
        rel = path.relative_to(target_dir).as_posix()
        key = (path.stem.lower(), ext)
        # Prefer shallower paths when duplicates exist.
        prev = mesh_map.get(key)
        if prev is None or rel.count("/") < prev.count("/"):
            mesh_map[key] = rel
    return mesh_map


def _normalise_mesh_stem(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r'^(mesh|collision|visual)\d*[_-]?', '', s)
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s


def _resolve_mesh_reference(original_ref: str, mesh_map: dict, preferred_ext: str):
    fallback = {
        ".stl": [".dae", ".obj"],
        ".dae": [".stl", ".obj"],
        ".obj": [".dae", ".stl"],
    }

    original_name = Path(original_ref.replace("\\", "/")).name
    original_stem = Path(original_name).stem.lower()
    original_norm = _normalise_mesh_stem(original_stem)
    original_ext = Path(original_name).suffix.lower()

    ext_order = []
    if preferred_ext in {".stl", ".dae", ".obj"}:
        ext_order.append(preferred_ext)
    if original_ext in {".stl", ".dae", ".obj"} and original_ext not in ext_order:
        ext_order.append(original_ext)
    for ext in fallback.get(preferred_ext, []):
        if ext not in ext_order:
            ext_order.append(ext)

    for ext in ext_order:
        exact = mesh_map.get((original_stem, ext))
        if exact:
            return exact

    candidates = []
    for (stem, ext), rel in mesh_map.items():
        if ext not in ext_order:
            continue
        norm = _normalise_mesh_stem(stem)
        score = -1

        if norm == original_norm and norm:
            score = 120
        elif norm.endswith(original_norm) and original_norm:
            score = 100
        elif original_norm.endswith(norm) and norm:
            score = 95
        elif original_norm and norm and (original_norm in norm or norm in original_norm):
            score = 80
        else:
            a = set(filter(None, original_norm.split('_')))
            b = set(filter(None, norm.split('_')))
            overlap = len(a & b)
            if overlap > 0:
                score = 60 + overlap

        if score >= 0:
            # Prefer shallower path and shorter names for tie-breaks.
            candidates.append((score, -rel.count('/'), -len(rel), rel))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][3]


def _rewrite_urdf_mesh_filenames(urdf_text: str, mesh_map: dict, preferred_ext: str):
    """Rewrite mesh filename= paths to exported meshes in this output folder."""
    updated = 0
    unresolved = 0

    def repl(match):
        nonlocal updated, unresolved
        original = match.group(1)
        rel = _resolve_mesh_reference(original, mesh_map, preferred_ext)
        if rel:
            updated += 1
            return f'filename="{rel}"'

        unresolved += 1
        return match.group(0)

    out = re.sub(r'filename="([^"]+)"', repl, urdf_text)
    return out, updated, unresolved


def _prepare_input_model_for_import(input_model: Path) -> Path:
    """Repair mesh filename links in URDF before import when package paths are legacy/misaligned."""
    ext = input_model.suffix.lower()
    if ext not in {'.urdf', '.xml'}:
        return input_model

    try:
        source_text = input_model.read_text(encoding='utf-8')
    except Exception:
        return input_model

    mesh_map = {}
    # Keep lookup local to the model folder for predictable runtime.
    search_roots = [
        input_model.parent,
        input_model.parent.parent,
        input_model.parent.parent / 'meshes',
    ]

    for root in search_roots:
        for candidate in sorted(root.rglob('*')):
            if not candidate.is_file():
                continue
            ext = candidate.suffix.lower()
            if ext not in {'.stl', '.dae', '.obj'}:
                continue
            rel = os.path.relpath(candidate, input_model.parent).replace('\\', '/')
            key = (candidate.stem.lower(), ext)
            prev = mesh_map.get(key)
            if prev is None or rel.count('/') < prev.count('/'):
                mesh_map[key] = rel

    if not mesh_map:
        return input_model

    preferred_ext = '.obj'
    rewritten, updated, unresolved = _rewrite_urdf_mesh_filenames(source_text, mesh_map, preferred_ext)
    if updated == 0:
        return input_model

    patched = input_model.parent / f"{input_model.stem}.resolved{input_model.suffix}"
    patched.write_text(rewritten, encoding='utf-8')
    print(f"[INFO] repaired input mesh references: {input_model.name} -> {patched.name} (updated={updated}, unresolved={unresolved})")
    return patched



def _iter_possible_mesh_roots(input_model: Path, out_root: Path) -> list[Path]:
    roots = [
        input_model.parent / 'meshes',
        input_model.parent.parent / 'meshes',
        input_model.parent / 'assets',
        input_model.parent.parent / 'assets',
        out_root,
    ]

    for fmt in ["dae", "stl", "unity", "mjcf"]:
        candidate = out_root / fmt
        if candidate.exists() and candidate.is_dir():
            roots.append(candidate)

    fallback = os.getenv('URDF_KITCHEN_MESH_FALLBACK_ROOT', '').strip()
    if fallback:
        for part in fallback.split(os.pathsep):
            part = part.strip()
            if part:
                p = Path(part).expanduser()
                if p.exists() and p.is_dir():
                    roots.append(p)

    return roots


def _fallback_copy_mesh(ref_name: str, preferred_ext: str, input_model: Path, out_root: Path, candidate_roots: list[Path]) -> str | None:
    target_name = Path(ref_name).name
    stem = Path(target_name).stem.lower()

    # Direct name match in known roots/subdirectories.
    check_names = [target_name, f"{stem}{preferred_ext}", f"{stem}.dae", f"{stem}.obj", f"{stem}.stl"]
    seeds = [
        Path(p) for p in candidate_roots
    ]

    for base in seeds:
        if not base.exists() or not base.is_dir():
            continue

        # Known mesh dirs are cheap and deterministic.
        candidate_dirs = []
        for rel_dir in ["", "meshes", "assets", "miro_robot_unity_description", "miro_robot_unity_description/meshes", "realfix_mjcf", "realfix_mjcf/assets", "miro_mjcf", "miro_mjcf/assets"]:
            d = base / rel_dir if rel_dir else base
            if d.exists() and d.is_dir():
                candidate_dirs.append(d)

        # If a root contains nested package libraries, scan 2-level mesh folders.
        # This is intentionally narrow to keep runtime predictable.
        if base.name == "RobotDescriptions":
            candidate_dirs.extend(
                child for child in base.glob('*/*/meshes')
                if child.is_dir()
            )
        elif base.name == "Robots":
            candidate_dirs.extend(
                child for child in base.glob('*/meshes')
                if child.is_dir()
            )

        for d in candidate_dirs:
            for name in check_names:
                c = d / name
                if c.exists() and c.is_file():
                    return str(c)

        # Fallback for MJCF exports that keep assets nested under known package folders.
        for rel_root in ["miro_robot_unity_description", "realfix_mjcf", "miro_mjcf"]:
            asset_root = base / rel_root / 'assets'
            if not asset_root.exists() or not asset_root.is_dir():
                continue
            for name in check_names:
                # direct lookup
                c = asset_root / name
                if c.exists() and c.is_file():
                    return str(c)
                # one-level nested search
                for sub in asset_root.glob('*'):
                    if sub.is_dir():
                        c2 = sub / name
                        if c2.exists() and c2.is_file():
                            return str(c2)

    # Last fallback: bounded scan for files that still match by stem.
    candidates = [
        p for p in out_root.rglob('**/*')
        if p.is_file() and Path(p).suffix.lower() in {'.stl', '.dae', '.obj'} and Path(p).stem.lower() == stem
    ]
    if candidates:
        return str(sorted(candidates)[0])

    return None

def copy_original_model_to_output_folders(input_model: Path, out_root: Path):
    """Copy original URDF into each output folder and rewrite mesh filenames to local exports."""
    # Per-folder target mesh format produced by batch export.
    preferred_mesh_ext = {
        "stl": ".stl",
        "dae": ".dae",
        "unity": ".dae",
        "mjcf": ".obj",
    }

    source_text = input_model.read_text(encoding="utf-8")
    mesh_refs = re.findall(r'filename\s*=\s*"([^"]+)"', source_text)
    fallback_roots = _iter_possible_mesh_roots(input_model, out_root)

    for folder, pref_ext in preferred_mesh_ext.items():
        target_dir = out_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = "source.urdf" if input_model.suffix.lower() == ".urdf" else input_model.name
        target = target_dir / target_name

        mesh_map = _index_mesh_files(target_dir)
        rewritten, updated, unresolved = _rewrite_urdf_mesh_filenames(source_text, mesh_map, pref_ext)

        # Last-mile patch for still-unresolved mesh refs by copying from available roots.
        if unresolved > 0:
            rewritten_refs = {}
            for ref in mesh_refs:
                rel = _resolve_mesh_reference(ref, mesh_map, pref_ext)
                if rel:
                    rewritten_refs[ref] = rel
                    continue

                source_mesh = _fallback_copy_mesh(ref, pref_ext, input_model, out_root, fallback_roots)
                if not source_mesh:
                    continue

                source_path = Path(source_mesh)
                dst_dir = target_dir / 'meshes'
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / source_path.name
                if not dst.exists() and source_path.exists():
                    try:
                        shutil.copy2(source_path, dst)
                    except Exception:
                        continue
                rewritten_refs[ref] = f"meshes/{dst.name}"

            if rewritten_refs:
                def late_repl(match):
                    original = match.group(1)
                    if original in rewritten_refs:
                        return f'filename="{rewritten_refs[original]}"'
                    return match.group(0)
                rewritten = re.sub(r'filename\s*=\s*"([^"]+)"', late_repl, rewritten)

        target.write_text(rewritten, encoding="utf-8")
        print(f"[INFO] wrote {target} (mesh refs updated={updated}, unresolved={unresolved}, pref={pref_ext})")

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
    resolved_input_model = _prepare_input_model_for_import(input_model)

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
        input_model=resolved_input_model,
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

    # If key format folders are still sparse, synthesize exports from source meshes.
    pre_stl = _count_mesh_files(out_root / 'stl' / 'meshes') + _count_mesh_files(out_root / 'stl')
    pre_unity = _count_mesh_files(out_root / 'unity' / 'meshes') + _count_mesh_files(out_root / 'unity')
    backfill_report = {'used': False, 'backfilled': 0, 'failed': 0, 'source': ''}
    if pre_stl == 0 or pre_unity == 0:
        backfill_report = _backfill_mesh_exports_from_source(input_model, out_root)

    copy_original_model_to_output_folders(resolved_input_model, out_root)

    print("\n=== Batch export summary ===")
    print(f"Input model : {input_model}")
    if resolved_input_model != input_model:
        print(f"Resolved input : {resolved_input_model}")
    print(f"Output root : {out_root}")
    print(f"Robot name  : {robot_name}")
    print(f"Meshes ok   : {converted}")
    print(f"Meshes fail : {failed}")
    if backfill_report.get('used'):
        print(f"Backfill ok : {backfill_report.get('backfilled', 0)} from {backfill_report.get('source')}")
        print(f"Backfill fail: {backfill_report.get('failed', 0)}")
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
