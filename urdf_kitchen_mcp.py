#!/usr/bin/env python3
"""
urdf_kitchen MCP Server  (Phase 1+2 - inspect & edit project files)

Usage:
  1. pip install mcp trimesh
  2. claude mcp add urdf-kitchen -- python3 urdf_kitchen_mcp.py  （in urdf_kitchen directory.）
  3. boot claude and talk.
"""

import csv
import glob
import io
import math
import os
import shutil
import datetime
import xml.etree.ElementTree as ET

from mcp.server.mcpserver import MCPServer

URDF_KITCHEN_DIR = os.path.dirname(os.path.abspath(__file__))

mcp = MCPServer("urdf-kitchen")


# ── internal helpers ─────────────────────────────────────────────────────────

def _load(filepath: str) -> tuple[ET.Element, str]:
    """Parse XML project file → (root, project_dir). Raises FileNotFoundError."""
    p = os.path.abspath(os.path.expanduser(filepath))
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")
    return ET.parse(p).getroot(), os.path.dirname(p)


_AXIS_NAME = {0: "X", 1: "Y", 2: "Z", 3: "Fixed", 4: "Free", 5: "Slide"}


def _parse_node(elem: ET.Element, project_dir: str) -> dict:
    """<node> element → structured dict (read-only view)."""

    def txt(tag, default=None):
        e = elem.find(tag)
        return e.text.strip() if (e is not None and e.text) else default

    def origin_dict(tag):
        e = elem.find(tag)
        if e is None:
            return None
        return {
            "xyz": (e.findtext("xyz") or "0 0 0").split(),
            "rpy": (e.findtext("rpy") or "0 0 0").split(),
        }

    rot_idx = int(txt("rotation_axis") or 0)
    mass = float(txt("mass") or 0.0)

    result: dict = {
        "name":             txt("name"),
        "type":             (txt("type") or "").split(".")[-1],
        "stl_file":         txt("stl_file"),
        "mass_kg":          mass,
        "volume_m3":        float(txt("volume") or 0.0),
        "rotation_axis":    _AXIS_NAME.get(rot_idx, str(rot_idx)),
        "joint_lower_deg":  round(math.degrees(float(txt("joint_lower") or 0.0)), 2),
        "joint_upper_deg":  round(math.degrees(float(txt("joint_upper") or 0.0)), 2),
        "joint_effort_Nm":  float(txt("joint_effort") or 0.0),
        "joint_velocity_rads": float(txt("joint_velocity") or 0.0),
        "joint_damping":    float(txt("joint_damping") or 0.0),
        "joint_stiffness":  float(txt("joint_stiffness") or 0.0),
        "joint_armature":   float(txt("joint_armature") or 0.0),
        "body_angle_deg":   [round(math.degrees(float(v)), 2)
                             for v in (txt("body_angle") or "0 0 0").split()],
        "is_imu_site":      txt("is_imu_site") == "True",
        "is_camera_node":   txt("is_camera_node") == "True",
        "massless_decoration": txt("massless_decoration") == "True",
        "hide_mesh":        txt("hide_mesh") == "True",
        "inertial_origin":  origin_dict("inertial_origin"),
        "visual_origin":    origin_dict("visual_origin"),
    }

    # Inertia tensor
    inertia_e = elem.find("inertia")
    if inertia_e is not None:
        result["inertia"] = {
            k: float(inertia_e.findtext(k) or 0)
            for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        }

    # Colliders count + quick summary
    colliders_e = elem.find("colliders")
    if colliders_e is not None:
        result["collider_count"] = len(list(colliders_e))
        result["colliders_summary"] = [
            {
                "type":    col.findtext("type"),
                "enabled": col.findtext("enabled") == "True",
                "shape":   (col.find("data") or ET.Element("_")).findtext("type"),
                "pos":     (col.findtext("position") or "0 0 0").split(),
            }
            for col in colliders_e
        ]
    else:
        result["collider_count"] = 0

    # Mesh existence
    if result["stl_file"]:
        result["mesh_exists"] = os.path.exists(
            os.path.join(project_dir, result["stl_file"])
        )

    return result


def _connections(root: ET.Element) -> list[dict]:
    conns_e = root.find("connections")
    return [
        {"from": c.findtext("from_node"), "to": c.findtext("to_node")}
        for c in (conns_e or [])
    ]


def _kinematic_children(conns: list[dict]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for c in conns:
        children.setdefault(c["from"], []).append(c["to"])
    return children


def _tree_lines(node: str, children: dict, masses: dict, depth: int = 0) -> list[str]:
    prefix = "  " * depth + ("└─ " if depth else "")
    m = masses.get(node, 0.0)
    lines = [f"{prefix}{node}  ({m:.4f} kg)"]
    for ch in children.get(node, []):
        lines += _tree_lines(ch, children, masses, depth + 1)
    return lines


# ── MCP tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def list_projects(directory: str = URDF_KITCHEN_DIR) -> dict:
    """
    指定ディレクトリのXMLプロジェクトファイルを一覧表示する。
    デフォルトは urdf_kitchen ディレクトリ。

    Returns: {directory, count, projects: [{filename, robot_name, node_count, size_kb}]}
    """
    d = os.path.abspath(os.path.expanduser(directory))
    results = []
    for f in sorted(glob.glob(os.path.join(d, "*.xml"))):
        try:
            root, _ = _load(f)
            results.append({
                "filename":   os.path.basename(f),
                "filepath":   f,
                "robot_name": root.findtext("robot_name") or "unknown",
                "node_count": len(list(root.find("nodes") or [])),
                "size_kb":    round(os.path.getsize(f) / 1024, 1),
            })
        except Exception as e:
            results.append({"filename": os.path.basename(f), "error": str(e)})
    return {"directory": d, "count": len(results), "projects": results}


@mcp.tool()
def get_robot_summary(filepath: str) -> dict:
    """
    ロボットの概要を返す。
    - リンク数・接続数・総質量
    - 関節タイプ分布（X回転/Y回転/Fixed/Free 等）
    - IMU・カメラノード数

    Args:
        filepath: XMLプロジェクトファイルのパス
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    nodes = [_parse_node(n, project_dir) for n in (nodes_e or [])]
    conns  = _connections(root)

    total_mass = sum(n["mass_kg"] for n in nodes if not n["massless_decoration"])
    axis_dist: dict[str, int] = {}
    for n in nodes:
        a = n["rotation_axis"]
        axis_dist[a] = axis_dist.get(a, 0) + 1

    djs = root.find("default_joint_settings")
    djs_dict = ({ch.tag: ch.text for ch in djs} if djs is not None else {})

    mj = root.find("mjcf_defaults")
    mj_dict = ({ch.tag: ch.text for ch in mj} if mj is not None else {})

    return {
        "robot_name":         root.findtext("robot_name"),
        "base_link_height_m": float(root.findtext("base_link_height") or 0),
        "total_links":        len(nodes),
        "total_connections":  len(conns),
        "total_mass_kg":      round(total_mass, 4),
        "joint_type_distribution": axis_dist,
        "imu_sites":          sum(1 for n in nodes if n["is_imu_site"]),
        "camera_nodes":       sum(1 for n in nodes if n["is_camera_node"]),
        "default_joint_settings": djs_dict,
        "mjcf_defaults":      mj_dict,
    }


@mcp.tool()
def get_kinematic_tree(filepath: str) -> str:
    """
    ロボットのキネマティクスツリーをテキスト形式で返す（各リンクの質量付き）。

    Args:
        filepath: XMLプロジェクトファイルのパス
    """
    root, _ = _load(filepath)
    nodes_e  = root.find("nodes")
    conns    = _connections(root)

    masses = {n.findtext("name"): float(n.findtext("mass") or 0)
              for n in (nodes_e or [])}
    children = _kinematic_children(conns)
    has_parent = {c["to"] for c in conns}
    roots = [c["from"] for c in conns if c["from"] not in has_parent]
    # fallback: nodes not appearing as 'to'
    all_names = list(masses.keys())
    if not roots:
        roots = [n for n in all_names if n not in has_parent]

    lines = [f"Robot: {root.findtext('robot_name')}"]
    for r in roots:
        lines += _tree_lines(r, children, masses)
    return "\n".join(lines)


@mcp.tool()
def list_nodes(filepath: str) -> dict:
    """
    プロジェクト内の全ノード一覧（名前・タイプ・質量・関節タイプ）を返す。

    Args:
        filepath: XMLプロジェクトファイルのパス
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    nodes = [_parse_node(n, project_dir) for n in (nodes_e or [])]
    summary = [
        {
            "name":          n["name"],
            "type":          n["type"],
            "mass_kg":       n["mass_kg"],
            "rotation_axis": n["rotation_axis"],
            "stl_file":      n["stl_file"],
        }
        for n in nodes
    ]
    return {
        "robot_name": root.findtext("robot_name"),
        "count": len(summary),
        "nodes": summary,
    }


@mcp.tool()
def get_node_details(filepath: str, node_name: str) -> dict:
    """
    特定ノードの詳細情報（質量・慣性テンソル・関節パラメータ・コライダー・親子関係）を返す。

    Args:
        filepath:  XMLプロジェクトファイルのパス
        node_name: ノード名（例: "c_arm_upper_r"）
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    for elem in (nodes_e or []):
        if elem.findtext("name") == node_name:
            result = _parse_node(elem, project_dir)
            conns = _connections(root)
            result["parent_nodes"] = [c["from"] for c in conns if c["to"] == node_name]
            result["child_nodes"]  = [c["to"]   for c in conns if c["from"] == node_name]
            return result
    return {"error": f"Node '{node_name}' not found"}


@mcp.tool()
def list_meshes(filepath: str) -> dict:
    """
    プロジェクトで参照されるメッシュ・XMLファイルを一覧し、存在確認を行う。

    Args:
        filepath: XMLプロジェクトファイルのパス
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    meshes = []
    for n in (nodes_e or []):
        name = n.findtext("name")
        for ref, kind in [(n.findtext("stl_file"), "visual"),
                          (n.findtext("xml_file"),  "parts_xml")]:
            if not ref:
                continue
            abs_p = os.path.join(project_dir, ref)
            exists = os.path.exists(abs_p)
            meshes.append({
                "node":          name,
                "kind":          kind,
                "relative_path": ref,
                "exists":        exists,
                "size_kb": round(os.path.getsize(abs_p) / 1024, 1) if exists else None,
            })
    missing = [m for m in meshes if not m["exists"]]
    return {
        "robot_name":      root.findtext("robot_name"),
        "total_refs":      len(meshes),
        "missing_count":   len(missing),
        "missing":         missing,
        "meshes":          meshes,
    }


@mcp.tool()
def check_project_issues(filepath: str) -> dict:
    """
    プロジェクトの潜在的な問題を検出する。
    - ゼロ質量・ゼロ慣性テンソル（非装飾リンク）
    - 存在しないメッシュファイル
    - 未接続ノード
    - 関節限界の反転（lower > upper）

    Args:
        filepath: XMLプロジェクトファイルのパス
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    nodes   = [_parse_node(n, project_dir) for n in (nodes_e or [])]
    conns   = _connections(root)

    connected = {c["from"] for c in conns} | {c["to"] for c in conns}
    issues: list[dict] = []

    for n in nodes:
        name      = n["name"]
        node_type = n["type"]
        is_deco   = n["massless_decoration"]
        is_base   = node_type == "BaseLinkNode"

        if not is_base and not is_deco and n["mass_kg"] == 0:
            issues.append({"node": name, "issue": "zero_mass",
                           "detail": "mass=0 on active link"})

        if stl := n.get("stl_file"):
            if not n.get("mesh_exists", True):
                issues.append({"node": name, "issue": "missing_mesh", "detail": stl})

        if node_type == "FooNode" and not is_deco:
            inertia = n.get("inertia", {})
            if inertia and all(v == 0 for v in inertia.values()):
                issues.append({"node": name, "issue": "zero_inertia",
                               "detail": "all inertia components = 0"})

        if name not in connected and not is_base:
            issues.append({"node": name, "issue": "disconnected",
                           "detail": "not connected to any other node"})

        lo, hi = n["joint_lower_deg"], n["joint_upper_deg"]
        if lo > hi:
            issues.append({"node": name, "issue": "inverted_joint_limits",
                           "detail": f"lower={lo}° > upper={hi}°"})

    return {
        "robot_name":  root.findtext("robot_name"),
        "status":      "OK" if not issues else "ISSUES_FOUND",
        "issue_count": len(issues),
        "issues":      issues,
    }


@mcp.tool()
def compare_joint_params(filepath: str) -> dict:
    """
    全ジョイントの主要パラメータを比較しやすい表形式で返す。
    設定ミスや統一されていないパラメータを発見するのに有用。

    Args:
        filepath: XMLプロジェクトファイルのパス
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    rows = []
    for n in (nodes_e or []):
        node = _parse_node(n, project_dir)
        if node["type"] == "BaseLinkNode":
            continue
        rows.append({
            "name":          node["name"],
            "axis":          node["rotation_axis"],
            "mass_kg":       node["mass_kg"],
            "effort_Nm":     node["joint_effort_Nm"],
            "velocity_rads": node["joint_velocity_rads"],
            "damping":       node["joint_damping"],
            "stiffness":     node["joint_stiffness"],
            "armature":      node["joint_armature"],
            "lower_deg":     node["joint_lower_deg"],
            "upper_deg":     node["joint_upper_deg"],
        })
    return {
        "robot_name": root.findtext("robot_name"),
        "count": len(rows),
        "joints": rows,
    }


# ── Phase 2: write helpers ────────────────────────────────────────────────────

# Reverse axis-name map for write operations
_AXIS_INDEX = {v: k for k, v in _AXIS_NAME.items()}

# Writable scalar fields: user-facing key → (xml tag, unit conversion in, unit conversion out)
# "in" = value arriving from user (MCP call), "out" = value stored in XML
_SCALAR_FIELDS: dict[str, tuple[str, callable, callable]] = {
    "mass_kg":             ("mass",              float, lambda v: str(v)),
    "volume_m3":           ("volume",            float, lambda v: str(v)),
    "joint_effort_Nm":     ("joint_effort",      float, lambda v: str(v)),
    "joint_velocity_rads": ("joint_velocity",    float, lambda v: str(v)),
    "joint_damping":       ("joint_damping",     float, lambda v: str(v)),
    "joint_stiffness":     ("joint_stiffness",   float, lambda v: str(v)),
    "joint_armature":      ("joint_armature",    float, lambda v: str(v)),
    "joint_frictionloss":  ("joint_frictionloss",float, lambda v: str(v)),
    "joint_kv":            ("joint_kv",          float, lambda v: str(v)),
    "joint_margin":        ("joint_margin",      float, lambda v: str(v)),
    "joint_lower_deg":     ("joint_lower",       float, lambda v: str(math.radians(v))),
    "joint_upper_deg":     ("joint_upper",       float, lambda v: str(math.radians(v))),
    "massless_decoration": ("massless_decoration",bool, lambda v: str(v)),
    "is_imu_site":         ("is_imu_site",       bool,  lambda v: str(v)),
    "is_camera_node":      ("is_camera_node",    bool,  lambda v: str(v)),
    "hide_mesh":           ("hide_mesh",         bool,  lambda v: str(v)),
}


def _backup(filepath: str) -> str:
    """Create timestamped .bak copy; return backup path."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{filepath}.{ts}.bak"
    shutil.copy2(filepath, bak)
    return bak


def _save(root: ET.Element, filepath: str) -> None:
    """Write XML back to file with indentation."""
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    ET.register_namespace("", "")
    tree.write(filepath, encoding="utf-8", xml_declaration=True)


def _find_node_elem(root: ET.Element, node_name: str) -> ET.Element | None:
    nodes_e = root.find("nodes")
    for elem in (nodes_e or []):
        if elem.findtext("name") == node_name:
            return elem
    return None


def _set_text(parent: ET.Element, tag: str, value: str) -> None:
    """Set or create child element text."""
    elem = parent.find(tag)
    if elem is None:
        elem = ET.SubElement(parent, tag)
    elem.text = value


# ── Phase 2 MCP tools ────────────────────────────────────────────────────────

@mcp.tool()
def modify_node(filepath: str, node_name: str, params: dict) -> dict:
    """
    特定ノードのパラメータを変更してXMLに書き戻す（自動バックアップあり）。

    Args:
        filepath:  XMLプロジェクトファイルのパス
        node_name: 変更対象のノード名
        params:    変更するパラメータの辞書。指定可能なキー:
          スカラー値:
            mass_kg, volume_m3,
            joint_effort_Nm, joint_velocity_rads,
            joint_damping, joint_stiffness, joint_armature,
            joint_frictionloss, joint_kv, joint_margin,
            joint_lower_deg, joint_upper_deg,
            massless_decoration, is_imu_site, is_camera_node, hide_mesh
          関節タイプ:
            rotation_axis: "X"|"Y"|"Z"|"Fixed"|"Free"|"Slide"
          慣性テンソル (inertia サブキー):
            inertia: {ixx, ixy, ixz, iyy, iyz, izz}
          原点 (origin サブキー):
            inertial_origin: {xyz: "x y z", rpy: "r p y"}
            visual_origin:   {xyz: "x y z", rpy: "r p y"}

    Returns:
        {status, backup_path, node_name, changed: {field: {before, after}}}

    Example:
        modify_node("roid1.xml", "l_arm_upper",
                    {"mass_kg": 0.12, "joint_damping": 0.20,
                     "inertia": {"ixx": 0.0005, "iyy": 0.0006, "izz": 0.0004}})
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)
    elem = _find_node_elem(root, node_name)
    if elem is None:
        return {"status": "error", "detail": f"Node '{node_name}' not found"}

    bak = _backup(abs_path)
    changed: dict[str, dict] = {}

    for key, value in params.items():
        # ── scalar fields ──────────────────────────────────────────────────
        if key in _SCALAR_FIELDS:
            xml_tag, cast_in, cast_out = _SCALAR_FIELDS[key]
            before = elem.findtext(xml_tag)
            new_text = cast_out(cast_in(value))
            _set_text(elem, xml_tag, new_text)
            changed[key] = {"before": before, "after": new_text}

        # ── rotation_axis ──────────────────────────────────────────────────
        elif key == "rotation_axis":
            idx = _AXIS_INDEX.get(str(value))
            if idx is None:
                return {"status": "error",
                        "detail": f"rotation_axis must be one of {list(_AXIS_INDEX)}"}
            before = elem.findtext("rotation_axis")
            _set_text(elem, "rotation_axis", str(idx))
            changed[key] = {"before": before, "after": str(idx)}

        # ── inertia sub-dict ───────────────────────────────────────────────
        elif key == "inertia" and isinstance(value, dict):
            inertia_e = elem.find("inertia")
            if inertia_e is None:
                inertia_e = ET.SubElement(elem, "inertia")
            for comp in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                if comp in value:
                    before = inertia_e.findtext(comp)
                    _set_text(inertia_e, comp, str(float(value[comp])))
                    changed[f"inertia.{comp}"] = {"before": before,
                                                   "after": str(float(value[comp]))}

        # ── origin sub-dicts ───────────────────────────────────────────────
        elif key in ("inertial_origin", "visual_origin") and isinstance(value, dict):
            origin_e = elem.find(key)
            if origin_e is None:
                origin_e = ET.SubElement(elem, key)
            for sub in ("xyz", "rpy"):
                if sub in value:
                    before = origin_e.findtext(sub)
                    new_text = str(value[sub])
                    _set_text(origin_e, sub, new_text)
                    changed[f"{key}.{sub}"] = {"before": before, "after": new_text}

        else:
            changed[key] = {"before": None, "after": None, "warning": "unknown key, skipped"}

    _save(root, abs_path)
    return {
        "status":      "ok",
        "node_name":   node_name,
        "backup_path": bak,
        "changed":     changed,
    }


@mcp.tool()
def bulk_set_joint_param(
    filepath: str,
    param: str,
    value: float,
    node_names: list[str] | None = None,
    axis_filter: str | None = None,
) -> dict:
    """
    複数ノードの同一関節パラメータを一括更新する（自動バックアップあり）。

    Args:
        filepath:    XMLプロジェクトファイルのパス
        param:       変更するパラメータ名（_SCALAR_FIELDS のキー、例: "joint_damping"）
        value:       設定する値
        node_names:  対象ノード名リスト。None なら全FooNodeが対象
        axis_filter: 関節タイプで絞り込み (例: "X", "Z", "Fixed")。None なら全タイプ

    Returns:
        {status, backup_path, updated_count, skipped_count, updated: [node_name, ...]}
    """
    if param not in _SCALAR_FIELDS:
        return {"status": "error",
                "detail": f"param '{param}' is not writable. Choose from: {list(_SCALAR_FIELDS)}"}

    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)
    bak = _backup(abs_path)

    xml_tag, cast_in, cast_out = _SCALAR_FIELDS[param]
    new_text = cast_out(cast_in(value))

    updated, skipped = [], []
    for elem in (root.find("nodes") or []):
        name = elem.findtext("name") or ""
        node_type = (elem.findtext("type") or "").split(".")[-1]

        if node_type == "BaseLinkNode":
            skipped.append(name)
            continue
        if node_names is not None and name not in node_names:
            skipped.append(name)
            continue
        if axis_filter is not None:
            rot_idx = int(elem.findtext("rotation_axis") or 0)
            if _AXIS_NAME.get(rot_idx) != axis_filter:
                skipped.append(name)
                continue

        _set_text(elem, xml_tag, new_text)
        updated.append(name)

    _save(root, abs_path)
    return {
        "status":        "ok",
        "param":         param,
        "value":         value,
        "backup_path":   bak,
        "updated_count": len(updated),
        "skipped_count": len(skipped),
        "updated":       updated,
    }


@mcp.tool()
def mirror_node_params(
    filepath: str,
    source_prefix: str = "l_",
    target_prefix: str = "r_",
    params: list[str] | None = None,
) -> dict:
    """
    左右対称ノードのパラメータをミラーリングする（l_ → r_ など）。

    コピーされるパラメータ: mass, inertia, joint_effort/velocity/damping/stiffness/armature 等
    慣性テンソル・原点はそのままコピー（軸は対称として扱う）。

    Args:
        filepath:       XMLプロジェクトファイルのパス
        source_prefix:  コピー元プレフィックス（デフォルト "l_"）
        target_prefix:  コピー先プレフィックス（デフォルト "r_"）
        params:         コピーするパラメータキーのリスト。None なら全コピー対象を含む

    Returns:
        {status, backup_path, mirrored: [{source, target, params_copied}], unmatched: [...]}
    """
    # Default params to mirror
    ALL_MIRROR_PARAMS = [
        "mass", "volume", "joint_effort", "joint_velocity",
        "joint_damping", "joint_stiffness", "joint_armature",
        "joint_frictionloss", "joint_kv", "joint_margin",
        "joint_lower", "joint_upper", "rotation_axis",
        "massless_decoration", "hide_mesh",
    ]
    mirror_tags = set(ALL_MIRROR_PARAMS)
    if params:
        # Translate user-facing keys to xml tags where possible
        mirror_tags = set()
        for p in params:
            mirror_tags.add(_SCALAR_FIELDS[p][0] if p in _SCALAR_FIELDS else p)

    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)
    bak = _backup(abs_path)

    # Build name → elem map
    nodes_e = root.find("nodes")
    name_to_elem = {e.findtext("name"): e for e in (nodes_e or [])}

    mirrored, unmatched = [], []
    for name, src_elem in name_to_elem.items():
        if not name.startswith(source_prefix):
            continue
        tgt_name = target_prefix + name[len(source_prefix):]
        tgt_elem = name_to_elem.get(tgt_name)
        if tgt_elem is None:
            unmatched.append({"source": name, "reason": f"no node named '{tgt_name}'"})
            continue

        copied_tags = []
        for tag in mirror_tags:
            src_child = src_elem.find(tag)
            if src_child is None:
                continue
            # Remove existing in target and copy from source
            existing = tgt_elem.find(tag)
            if existing is not None:
                tgt_elem.remove(existing)
            import copy as _copy
            tgt_elem.append(_copy.deepcopy(src_child))
            copied_tags.append(tag)

        # Mirror inertia tensor
        src_inertia = src_elem.find("inertia")
        if src_inertia is not None:
            existing = tgt_elem.find("inertia")
            if existing is not None:
                tgt_elem.remove(existing)
            import copy as _copy
            tgt_elem.append(_copy.deepcopy(src_inertia))
            copied_tags.append("inertia")

        # Mirror origins
        for origin_tag in ("inertial_origin", "visual_origin"):
            src_o = src_elem.find(origin_tag)
            if src_o is not None:
                existing = tgt_elem.find(origin_tag)
                if existing is not None:
                    tgt_elem.remove(existing)
                import copy as _copy
                tgt_elem.append(_copy.deepcopy(src_o))
                copied_tags.append(origin_tag)

        mirrored.append({"source": name, "target": tgt_name, "params_copied": copied_tags})

    _save(root, abs_path)
    return {
        "status":      "ok",
        "backup_path": bak,
        "mirrored_count": len(mirrored),
        "unmatched_count": len(unmatched),
        "mirrored":    mirrored,
        "unmatched":   unmatched,
    }


@mcp.tool()
def suggest_inertia(
    mass_kg: float,
    shape: str,
    dim_a: float = 0.0,
    dim_b: float = 0.0,
    dim_c: float = 0.0,
) -> dict:
    """
    形状と質量から慣性テンソルを計算して返す（modify_node の inertia 引数にそのまま使える）。

    Args:
        mass_kg: 質量 [kg]
        shape:   "box" | "cylinder" | "sphere"
        dim_a:   box→幅X[m] / cylinder→半径[m] / sphere→半径[m]
        dim_b:   box→奥行Y[m] / cylinder→高さ[m] / sphere→(不使用)
        dim_c:   box→高さZ[m] / (他は不使用)

    Returns:
        {shape, mass_kg, dims, inertia: {ixx,ixy,ixz,iyy,iyz,izz}, note}
        ※ inertia 値を modify_node の params["inertia"] にそのまま渡せる

    Examples:
        suggest_inertia(0.1, "box", 0.05, 0.08, 0.06)
        suggest_inertia(0.15, "cylinder", 0.03, 0.12)
        suggest_inertia(0.05, "sphere", 0.025)
    """
    m = float(mass_kg)
    inertia = {"ixy": 0.0, "ixz": 0.0, "iyz": 0.0}

    if shape == "box":
        w, d, h = float(dim_a), float(dim_b), float(dim_c)
        inertia["ixx"] = m / 12.0 * (d**2 + h**2)
        inertia["iyy"] = m / 12.0 * (w**2 + h**2)
        inertia["izz"] = m / 12.0 * (w**2 + d**2)
        dims = {"width_x": w, "depth_y": d, "height_z": h}

    elif shape == "cylinder":
        r, h = float(dim_a), float(dim_b)
        inertia["ixx"] = m / 12.0 * (3 * r**2 + h**2)
        inertia["iyy"] = inertia["ixx"]
        inertia["izz"] = m / 2.0 * r**2
        dims = {"radius": r, "height": h}

    elif shape == "sphere":
        r = float(dim_a)
        v = 2.0 / 5.0 * m * r**2
        inertia["ixx"] = inertia["iyy"] = inertia["izz"] = v
        dims = {"radius": r}

    else:
        return {"status": "error",
                "detail": "shape must be 'box', 'cylinder', or 'sphere'"}

    # Round to 7 significant digits (URDF convention)
    inertia = {k: round(v, 9) for k, v in inertia.items()}

    return {
        "shape":    shape,
        "mass_kg":  m,
        "dims":     dims,
        "inertia":  inertia,
        "note":     "Pass inertia dict directly to modify_node params['inertia']",
    }


@mcp.tool()
def export_joint_csv(filepath: str, output_path: str | None = None) -> dict:
    """
    全関節パラメータをCSVファイルにエクスポートする。

    Args:
        filepath:    XMLプロジェクトファイルのパス
        output_path: 出力CSVパス。None なら プロジェクトと同じディレクトリに自動命名

    Returns:
        {status, output_path, row_count, columns}
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    nodes   = [_parse_node(n, project_dir) for n in (nodes_e or [])]

    rows = []
    for n in nodes:
        if n["type"] == "BaseLinkNode":
            continue
        inertia = n.get("inertia", {})
        rows.append({
            "name":             n["name"],
            "rotation_axis":    n["rotation_axis"],
            "mass_kg":          n["mass_kg"],
            "ixx":              inertia.get("ixx", 0),
            "iyy":              inertia.get("iyy", 0),
            "izz":              inertia.get("izz", 0),
            "ixy":              inertia.get("ixy", 0),
            "ixz":              inertia.get("ixz", 0),
            "iyz":              inertia.get("iyz", 0),
            "joint_effort_Nm":  n["joint_effort_Nm"],
            "joint_velocity_rads": n["joint_velocity_rads"],
            "joint_damping":    n["joint_damping"],
            "joint_stiffness":  n["joint_stiffness"],
            "joint_armature":   n["joint_armature"],
            "joint_frictionloss": n.get("joint_frictionloss", 0),
            "joint_kv":         n.get("joint_kv", 0),
            "joint_lower_deg":  n["joint_lower_deg"],
            "joint_upper_deg":  n["joint_upper_deg"],
            "massless_decoration": n["massless_decoration"],
            "is_imu_site":      n["is_imu_site"],
            "is_camera_node":   n["is_camera_node"],
            "stl_file":         n.get("stl_file") or "",
        })

    if not output_path:
        robot = root.findtext("robot_name") or "robot"
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(project_dir, f"{robot}_joints_{ts}.csv")

    output_path = os.path.abspath(os.path.expanduser(output_path))
    columns = list(rows[0].keys()) if rows else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "status":      "ok",
        "output_path": output_path,
        "row_count":   len(rows),
        "columns":     columns,
    }


@mcp.tool()
def sync_joints_to_defaults(
    filepath: str,
    only_nodes: list[str] | None = None,
) -> dict:
    """
    プロジェクトの default_joint_settings を全ノード（または指定ノード）に適用する。

    個別に設定済みのノードを一括でデフォルトに戻したいときに使う。
    質量・慣性テンソル・関節タイプ・関節限界は変更しない。

    Args:
        filepath:   XMLプロジェクトファイルのパス
        only_nodes: 対象ノード名リスト。None なら全FooNodeが対象

    Returns:
        {status, backup_path, updated_count, defaults_applied}
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)

    djs = root.find("default_joint_settings")
    if djs is None:
        return {"status": "error", "detail": "No default_joint_settings in project"}

    # Build defaults dict (xml_tag → value_text)
    tag_map = {
        "effort":       "joint_effort",
        "velocity":     "joint_velocity",
        "damping":      "joint_damping",
        "stiffness_kp": "joint_stiffness",
        "armature":     "joint_armature",
        "frictionloss": "joint_frictionloss",
        "damping_kv":   "joint_kv",
        "margin":       "joint_margin",
    }
    defaults: dict[str, str] = {}
    for djs_tag, node_tag in tag_map.items():
        val = djs.findtext(djs_tag) or djs.findtext(node_tag)
        if val:
            defaults[node_tag] = val

    bak = _backup(abs_path)
    updated = []
    for elem in (root.find("nodes") or []):
        name = elem.findtext("name") or ""
        node_type = (elem.findtext("type") or "").split(".")[-1]
        if node_type == "BaseLinkNode":
            continue
        if only_nodes is not None and name not in only_nodes:
            continue
        for xml_tag, val in defaults.items():
            _set_text(elem, xml_tag, val)
        updated.append(name)

    _save(root, abs_path)
    return {
        "status":          "ok",
        "backup_path":     bak,
        "defaults_applied": defaults,
        "updated_count":   len(updated),
        "updated":         updated,
    }


@mcp.tool()
def find_mirror_candidates(filepath: str,
                            source_prefix: str = "l_",
                            target_prefix: str = "r_") -> dict:
    """
    ミラーリング可能な左右ペアを検索し、パラメータの差分を報告する。
    mirror_node_params を実行する前の確認に使う。

    Args:
        filepath:       XMLプロジェクトファイルのパス
        source_prefix:  コピー元プレフィックス（デフォルト "l_"）
        target_prefix:  コピー先プレフィックス（デフォルト "r_"）

    Returns:
        {pairs: [{source, target, param_diffs: {param: {source_val, target_val}}}],
         unmatched_sources: [...]}
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    nodes   = {_parse_node(n, project_dir)["name"]: _parse_node(n, project_dir)
               for n in (nodes_e or [])}

    COMPARE_PARAMS = [
        "mass_kg", "joint_effort_Nm", "joint_velocity_rads",
        "joint_damping", "joint_stiffness", "joint_armature",
        "joint_lower_deg", "joint_upper_deg", "rotation_axis",
    ]

    pairs, unmatched = [], []
    for name, src in nodes.items():
        if not name.startswith(source_prefix):
            continue
        tgt_name = target_prefix + name[len(source_prefix):]
        if tgt_name not in nodes:
            unmatched.append(name)
            continue
        tgt = nodes[tgt_name]
        diffs = {}
        for p in COMPARE_PARAMS:
            sv, tv = src.get(p), tgt.get(p)
            if sv != tv:
                diffs[p] = {"source_val": sv, "target_val": tv}
        pairs.append({"source": name, "target": tgt_name,
                      "in_sync": len(diffs) == 0, "param_diffs": diffs})

    return {
        "source_prefix":    source_prefix,
        "target_prefix":    target_prefix,
        "pair_count":       len(pairs),
        "in_sync_count":    sum(1 for p in pairs if p["in_sync"]),
        "out_of_sync_count":sum(1 for p in pairs if not p["in_sync"]),
        "unmatched_count":  len(unmatched),
        "pairs":            pairs,
        "unmatched_sources":unmatched,
    }


# ── Phase 3: full-feature tools ──────────────────────────────────────────────

# ── helpers (phase 3) ────────────────────────────────────────────────────────

def _default_node_xml(name: str, pos_x: float, pos_y: float,
                      rotation_axis: str, defaults: dict) -> ET.Element:
    """Minimal FooNode XML element with default values."""
    axis_idx = _AXIS_INDEX.get(rotation_axis, 2)
    elem = ET.Element("node")
    for tag, val in [
        ("name", name), ("type", "insilico.nodes.FooNode"),
        ("pos_x", str(pos_x)), ("pos_y", str(pos_y)),
        ("mass", "0.0"), ("volume", "0.0"),
        ("rotation_axis", str(axis_idx)), ("slide_axis", "0"),
        ("slide_lower", "-0.05"), ("slide_upper", "0.05"),
        ("joint_lower", str(math.radians(-180))), ("joint_upper", str(math.radians(180))),
        ("joint_effort",      defaults.get("effort",      "2.64")),
        ("joint_velocity",    defaults.get("velocity",    "8.06")),
        ("joint_damping",     defaults.get("damping",     "0.18")),
        ("joint_stiffness",   defaults.get("stiffness_kp","50.0")),
        ("joint_kv",          defaults.get("damping_kv",  "1.0")),
        ("joint_margin",      defaults.get("margin",      "0.0035")),
        ("joint_armature",    defaults.get("armature",    "0.01")),
        ("joint_frictionloss",defaults.get("frictionloss","0.005")),
        ("body_angle", "0.0 0.0 0.0"),
        ("is_mesh_reversed", "False"), ("massless_decoration", "False"),
        ("hide_mesh", "False"), ("is_imu_site", "False"), ("is_camera_node", "False"),
    ]:
        _set_text(elem, tag, val)

    inertia_e = ET.SubElement(elem, "inertia")
    for comp in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        ET.SubElement(inertia_e, comp).text = "0.0"
    for origin_tag in ("inertial_origin", "visual_origin"):
        o = ET.SubElement(elem, origin_tag)
        ET.SubElement(o, "xyz").text = "0.0 0.0 0.0"
        ET.SubElement(o, "rpy").text = "0.0 0.0 0.0"
    ET.SubElement(elem, "colliders")
    return elem


def _read_parts_xml_points(xml_path: str) -> list[dict]:
    """Parse parts XML → list of {xyz:[x,y,z], angle:[r,p,y]} dicts."""
    if not xml_path or not os.path.exists(xml_path):
        return []
    try:
        root = ET.parse(xml_path).getroot()
        points = []
        for pt in root.findall("point"):
            xyz_txt = (pt.findtext("point_xyz") or "0 0 0").split()
            ang_txt = (pt.findtext("point_angle") or "0 0 0").split()
            points.append({
                "name": pt.get("name", ""),
                "xyz":   [float(v) for v in xyz_txt],
                "angle": [float(v) for v in ang_txt],
            })
        return points
    except Exception:
        return []


def _port_name_to_index(port_name: str) -> int:
    """'out_1'→0, 'out_2'→1, 'port_1'→0, 'out'→0, etc."""
    if port_name and "_" in port_name:
        parts = port_name.split("_")
        # "out_1", "out_2", "port_1", "port_2", ...
        if len(parts) > 1 and parts[-1].isdigit():
            return max(0, int(parts[-1]) - 1)
    return 0


def _fmt(v: float) -> str:
    """Format float without unnecessary trailing zeros or exponential notation."""
    if v == 0.0:
        return "0"
    s = f"{v:.9f}".rstrip("0").rstrip(".")
    return s


# ── Phase 3 MCP tools ─────────────────────────────────────────────────────────

@mcp.tool()
def add_node(
    filepath: str,
    name: str,
    rotation_axis: str = "Z",
    stl_file: str | None = None,
    xml_file: str | None = None,
    pos_x: float = 0.0,
    pos_y: float = 0.0,
) -> dict:
    """
    新規 FooNode をプロジェクト XML に追加する（自動バックアップあり）。

    質量・慣性テンソルは後で modify_node または calculate_mesh_properties で設定する。
    関節パラメータはプロジェクトの default_joint_settings が自動適用される。

    Args:
        filepath:       XMLプロジェクトファイルのパス
        name:           新ノード名
        rotation_axis:  "X"|"Y"|"Z"|"Fixed"|"Free"|"Slide"
        stl_file:       ビジュアルメッシュのパス（プロジェクトからの相対パス）
        xml_file:       PartsEditor出力XMLのパス（接続ポイント定義）
        pos_x, pos_y:   ノードグラフ上の表示位置
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)

    # Check duplicate name
    if _find_node_elem(root, name) is not None:
        return {"status": "error", "detail": f"Node '{name}' already exists"}

    # Get project defaults
    djs = root.find("default_joint_settings")
    defaults = {ch.tag: ch.text for ch in djs} if djs is not None else {}

    bak = _backup(abs_path)
    elem = _default_node_xml(name, pos_x, pos_y, rotation_axis, defaults)

    if stl_file:
        _set_text(elem, "stl_file", stl_file)
    if xml_file:
        _set_text(elem, "xml_file", xml_file)

    nodes_e = root.find("nodes")
    if nodes_e is None:
        nodes_e = ET.SubElement(root, "nodes")
    nodes_e.append(elem)

    _save(root, abs_path)
    return {
        "status":      "ok",
        "added_node":  name,
        "backup_path": bak,
        "tip":         "Set mass/inertia with modify_node or calculate_mesh_properties",
    }


@mcp.tool()
def delete_node(filepath: str, node_name: str) -> dict:
    """
    ノードとそのすべての接続を削除する（自動バックアップあり）。

    Args:
        filepath:  XMLプロジェクトファイルのパス
        node_name: 削除するノード名
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)

    elem = _find_node_elem(root, node_name)
    if elem is None:
        return {"status": "error", "detail": f"Node '{node_name}' not found"}

    bak = _backup(abs_path)

    # Remove node
    nodes_e = root.find("nodes")
    nodes_e.remove(elem)

    # Remove connections referencing this node
    conns_e = root.find("connections")
    removed_conns = []
    if conns_e is not None:
        to_remove = [c for c in conns_e
                     if c.findtext("from_node") == node_name
                     or c.findtext("to_node") == node_name]
        for c in to_remove:
            conns_e.remove(c)
            removed_conns.append({
                "from": c.findtext("from_node"),
                "to":   c.findtext("to_node"),
            })

    _save(root, abs_path)
    return {
        "status":            "ok",
        "deleted_node":      node_name,
        "removed_connections": removed_conns,
        "backup_path":       bak,
    }


@mcp.tool()
def connect_nodes(
    filepath: str,
    from_node: str,
    to_node: str,
    from_port: str = "port_1",
    to_port: str = "in",
) -> dict:
    """
    2つのノードを接続する（自動バックアップあり）。

    Args:
        filepath:  XMLプロジェクトファイルのパス
        from_node: 接続元ノード名（親）
        to_node:   接続先ノード名（子）
        from_port: 接続元ポート名（"out" or "port_1", "port_2" ...）
        to_port:   接続先ポート名（通常 "in"）
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)

    # Validate nodes exist
    for n in (from_node, to_node):
        if _find_node_elem(root, n) is None:
            return {"status": "error", "detail": f"Node '{n}' not found"}

    # Check for duplicate connection
    conns_e = root.find("connections")
    if conns_e is None:
        conns_e = ET.SubElement(root, "connections")
    for c in conns_e:
        if (c.findtext("from_node") == from_node and
                c.findtext("to_node") == to_node and
                c.findtext("from_port") == from_port):
            return {"status": "error",
                    "detail": f"Connection {from_node}→{to_node} (port {from_port}) already exists"}

    bak = _backup(abs_path)
    conn = ET.SubElement(conns_e, "connection")
    for tag, val in [("from_node", from_node), ("from_port", from_port),
                     ("to_node", to_node), ("to_port", to_port)]:
        ET.SubElement(conn, tag).text = val

    _save(root, abs_path)
    return {
        "status":      "ok",
        "connection":  f"{from_node}[{from_port}] → {to_node}[{to_port}]",
        "backup_path": bak,
    }


@mcp.tool()
def disconnect_nodes(filepath: str, from_node: str, to_node: str) -> dict:
    """
    2つのノード間の接続をすべて削除する（自動バックアップあり）。

    Args:
        filepath:  XMLプロジェクトファイルのパス
        from_node: 接続元ノード名
        to_node:   接続先ノード名
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)
    bak = _backup(abs_path)

    conns_e = root.find("connections")
    removed = []
    if conns_e is not None:
        to_remove = [c for c in conns_e
                     if c.findtext("from_node") == from_node
                     and c.findtext("to_node") == to_node]
        for c in to_remove:
            conns_e.remove(c)
            removed.append(c.findtext("from_port"))

    _save(root, abs_path)
    return {
        "status":        "ok" if removed else "not_found",
        "removed_ports": removed,
        "backup_path":   bak,
    }


@mcp.tool()
def rename_node(filepath: str, old_name: str, new_name: str) -> dict:
    """
    ノード名を変更する（接続の参照も同時に更新する）。

    Args:
        filepath: XMLプロジェクトファイルのパス
        old_name: 現在のノード名
        new_name: 新しいノード名
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)

    elem = _find_node_elem(root, old_name)
    if elem is None:
        return {"status": "error", "detail": f"Node '{old_name}' not found"}
    if _find_node_elem(root, new_name) is not None:
        return {"status": "error", "detail": f"Node '{new_name}' already exists"}

    bak = _backup(abs_path)
    # Rename node element
    elem.find("name").text = new_name

    # Update connections
    conns_e = root.find("connections")
    updated_conns = 0
    for c in (conns_e or []):
        fn = c.find("from_node")
        tn = c.find("to_node")
        if fn is not None and fn.text == old_name:
            fn.text = new_name
            updated_conns += 1
        if tn is not None and tn.text == old_name:
            tn.text = new_name
            updated_conns += 1

    _save(root, abs_path)
    return {
        "status":           "ok",
        "old_name":         old_name,
        "new_name":         new_name,
        "connections_updated": updated_conns,
        "backup_path":      bak,
    }


@mcp.tool()
def clone_node(
    filepath: str,
    source_name: str,
    new_name: str,
    pos_offset_x: float = 200.0,
    pos_offset_y: float = 0.0,
) -> dict:
    """
    既存ノードを深いコピーで複製する（接続はコピーしない）。

    物理パラメータをすべて引き継いだ新ノードを作成し、その後
    modify_node や connect_nodes で調整するワークフローに向いている。

    Args:
        filepath:      XMLプロジェクトファイルのパス
        source_name:   コピー元ノード名
        new_name:      新ノード名
        pos_offset_x:  グラフ上の X 位置オフセット
        pos_offset_y:  グラフ上の Y 位置オフセット
    """
    import copy as _copy

    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)

    src = _find_node_elem(root, source_name)
    if src is None:
        return {"status": "error", "detail": f"Node '{source_name}' not found"}
    if _find_node_elem(root, new_name) is not None:
        return {"status": "error", "detail": f"Node '{new_name}' already exists"}

    bak = _backup(abs_path)
    new_elem = _copy.deepcopy(src)
    new_elem.find("name").text = new_name

    # Shift position
    for tag, offset in (("pos_x", pos_offset_x), ("pos_y", pos_offset_y)):
        e = new_elem.find(tag)
        if e is not None:
            e.text = str(float(e.text or "0") + offset)

    root.find("nodes").append(new_elem)
    _save(root, abs_path)
    return {
        "status":      "ok",
        "cloned_from": source_name,
        "new_node":    new_name,
        "backup_path": bak,
        "tip":         "Use connect_nodes to wire the new node into the graph",
    }


@mcp.tool()
def update_project_settings(
    filepath: str,
    robot_name: str | None = None,
    base_link_height: float | None = None,
    default_joint_settings: dict | None = None,
    mjcf_defaults: dict | None = None,
) -> dict:
    """
    プロジェクト全体の設定を更新する（自動バックアップあり）。

    Args:
        filepath:               XMLプロジェクトファイルのパス
        robot_name:             新しいロボット名（None で変更なし）
        base_link_height:       MJCF出力用の基準高さ [m]
        default_joint_settings: 更新するキー: effort, velocity, damping,
                                stiffness_kp, margin, armature, frictionloss, damping_kv
        mjcf_defaults:          更新するキー: option_impratio, joint_damping,
                                geom_friction, geom_margin, geom_condim,
                                motor_ctrlrange, option_timestep, option_iterations,
                                mesh_simplify_threshold, mesh_max_faces

    Returns:
        {status, backup_path, changed}
    """
    abs_path = os.path.abspath(os.path.expanduser(filepath))
    root, _ = _load(abs_path)
    bak = _backup(abs_path)
    changed: dict = {}

    if robot_name is not None:
        e = root.find("robot_name")
        if e is None:
            e = ET.SubElement(root, "robot_name")
        changed["robot_name"] = {"before": e.text, "after": robot_name}
        e.text = robot_name

    if base_link_height is not None:
        e = root.find("base_link_height")
        if e is None:
            e = ET.SubElement(root, "base_link_height")
        changed["base_link_height"] = {"before": e.text, "after": str(base_link_height)}
        e.text = str(base_link_height)

    if default_joint_settings:
        djs = root.find("default_joint_settings")
        if djs is None:
            djs = ET.SubElement(root, "default_joint_settings")
        for key, val in default_joint_settings.items():
            _set_text(djs, key, str(val))
            changed[f"default_joint_settings.{key}"] = str(val)

    if mjcf_defaults:
        mj = root.find("mjcf_defaults")
        if mj is None:
            mj = ET.SubElement(root, "mjcf_defaults")
        for key, val in mjcf_defaults.items():
            _set_text(mj, key, str(val))
            changed[f"mjcf_defaults.{key}"] = str(val)

    _save(root, abs_path)
    return {"status": "ok", "backup_path": bak, "changed": changed}


@mcp.tool()
def calculate_mesh_properties(
    mesh_path: str,
    mass_kg: float | None = None,
    density_kg_m3: float | None = None,
) -> dict:
    """
    メッシュファイル (.stl/.dae/.obj) の物理プロパティを trimesh で計算する。

    mass_kg か density_kg_m3 のいずれかを指定すると慣性テンソルを計算できる。
    どちらも指定しない場合は volume と CoM のみ返す。

    Args:
        mesh_path:      メッシュファイルのパス (.stl / .dae / .obj)
        mass_kg:        質量 [kg]（指定すると慣性テンソルを計算）
        density_kg_m3:  密度 [kg/m³]（指定すると質量と慣性テンソルを計算）

    Returns:
        {volume_m3, center_of_mass_xyz, mass_kg, inertia, bounds, surface_area_m2}
        ※ inertia は modify_node の params["inertia"] にそのまま渡せる
    """
    try:
        import trimesh as _tm
    except ImportError:
        return {"status": "error", "detail": "trimesh not installed: pip install trimesh"}

    p = os.path.abspath(os.path.expanduser(mesh_path))
    if not os.path.exists(p):
        return {"status": "error", "detail": f"File not found: {p}"}

    try:
        mesh = _tm.load(p, force="mesh")
        if not isinstance(mesh, _tm.Trimesh):
            # Scene → merge
            mesh = mesh.dump(concatenate=True)
    except Exception as e:
        return {"status": "error", "detail": f"Load failed: {e}"}

    volume  = float(mesh.volume)
    com     = [round(float(v), 6) for v in mesh.center_mass]
    bounds  = {"min": [round(float(v), 6) for v in mesh.bounds[0]],
               "max": [round(float(v), 6) for v in mesh.bounds[1]]}
    size    = [round(float(v), 6) for v in (mesh.bounds[1] - mesh.bounds[0])]

    result: dict = {
        "status":           "ok",
        "mesh_path":        p,
        "volume_m3":        round(volume, 9),
        "center_of_mass_xyz": com,
        "surface_area_m2":  round(float(mesh.area), 6),
        "bounds":           bounds,
        "size_xyz":         size,
        "is_watertight":    bool(mesh.is_watertight),
    }

    # Mass / density
    if density_kg_m3 is not None and mass_kg is None:
        mass_kg = volume * float(density_kg_m3)
    if mass_kg is not None:
        result["mass_kg"] = round(float(mass_kg), 6)
        # Inertia relative to CoM at given mass
        try:
            density = float(mass_kg) / volume if volume > 0 else 0.0
            inertia_tensor = mesh.moment_inertia * density  # 3x3 matrix
            ixx = float(inertia_tensor[0, 0])
            iyy = float(inertia_tensor[1, 1])
            izz = float(inertia_tensor[2, 2])
            ixy = float(inertia_tensor[0, 1])
            ixz = float(inertia_tensor[0, 2])
            iyz = float(inertia_tensor[1, 2])
            result["inertia"] = {
                "ixx": round(ixx, 9), "ixy": round(ixy, 9), "ixz": round(ixz, 9),
                "iyy": round(iyy, 9), "iyz": round(iyz, 9), "izz": round(izz, 9),
            }
            result["note"] = ("Pass result['inertia'] to modify_node params['inertia'], "
                              "and result['center_of_mass_xyz'] to inertial_origin.xyz")
        except Exception as e:
            result["inertia_error"] = str(e)

    return result


@mcp.tool()
def read_parts_xml(xml_path: str) -> dict:
    """
    PartsEditor が出力した部品 XML を読み込み、接続ポイント・慣性情報を返す。

    これらの接続ポイント (point) が URDF エクスポート時の joint origin になる。

    Args:
        xml_path: PartsEditor 出力 XML ファイルのパス

    Returns:
        {link_name, center_of_mass, mass, volume, inertia, points, joint_axis}
    """
    p = os.path.abspath(os.path.expanduser(xml_path))
    if not os.path.exists(p):
        return {"status": "error", "detail": f"File not found: {p}"}

    root = ET.parse(p).getroot()
    link_e = root.find("link")

    result: dict = {"status": "ok", "file": p}
    if link_e is not None:
        result["link_name"] = link_e.get("name")
        inertial_e = link_e.find("inertial")
        if inertial_e is not None:
            result["mass"]   = float(inertial_e.findtext("mass") or 0)
            result["volume"] = float(inertial_e.findtext("volume") or 0)
            com_e = link_e.find("center_of_mass")
            result["center_of_mass"] = (com_e.text or "0 0 0").split() if com_e is not None else None
            inertia_e = inertial_e.find("inertia")
            if inertia_e is not None:
                result["inertia"] = {
                    k: float(inertia_e.get(k, 0))
                    for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                }

    # Connection points
    result["points"] = []
    for i, pt in enumerate(root.findall("point")):
        xyz = (pt.findtext("point_xyz") or "0 0 0").split()
        ang = (pt.findtext("point_angle") or "0 0 0").split()
        result["points"].append({
            "index":    i,
            "name":     pt.get("name", f"point{i+1}"),
            "type":     pt.get("type", ""),
            "xyz":      [float(v) for v in xyz],
            "angle_rad":[float(v) for v in ang],
        })

    joint_e = root.find("joint")
    if joint_e is not None:
        axis_e = joint_e.find("axis")
        result["joint_axis"] = axis_e.get("xyz") if axis_e is not None else None

    result["point_count"] = len(result["points"])
    return result


@mcp.tool()
def get_robot_metrics(filepath: str) -> dict:
    """
    ロボットの質量分布・近似 CoM・慣性サマリーを計算して返す。

    joint_origin が不明のため CoM は link の inertial_origin のみから近似する（参考値）。

    Args:
        filepath: XMLプロジェクトファイルのパス
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    nodes   = [_parse_node(n, project_dir) for n in (nodes_e or [])]

    total_mass = 0.0
    mass_by_node: list[dict] = []
    total_ixx = total_iyy = total_izz = 0.0

    for n in nodes:
        if n["massless_decoration"] or n["type"] == "BaseLinkNode":
            continue
        m = n["mass_kg"]
        total_mass += m
        inertia = n.get("inertia", {})
        total_ixx += inertia.get("ixx", 0)
        total_iyy += inertia.get("iyy", 0)
        total_izz += inertia.get("izz", 0)
        mass_by_node.append({"name": n["name"], "mass_kg": m,
                              "fraction_%": 0.0})  # filled below

    for entry in mass_by_node:
        entry["fraction_%"] = round(entry["mass_kg"] / total_mass * 100, 1) if total_mass else 0

    # Simple body-part grouping by name prefix
    groups: dict[str, float] = {}
    for entry in mass_by_node:
        nm = entry["name"].lower()
        if nm.startswith(("l_", "r_")):
            prefix = nm[:2]
        elif nm.startswith("c_"):
            prefix = "c_"
        else:
            prefix = "other"
        groups[prefix] = groups.get(prefix, 0) + entry["mass_kg"]

    return {
        "robot_name":        root.findtext("robot_name"),
        "total_mass_kg":     round(total_mass, 4),
        "sum_ixx":           round(total_ixx, 6),
        "sum_iyy":           round(total_iyy, 6),
        "sum_izz":           round(total_izz, 6),
        "mass_by_group":     {k: round(v, 4) for k, v in sorted(groups.items())},
        "mass_by_node":      sorted(mass_by_node, key=lambda x: -x["mass_kg"]),
    }


@mcp.tool()
def get_collision_info(filepath: str, node_name: str | None = None) -> dict:
    """
    コライダー詳細情報を返す。node_name を指定すると1ノードのみ、
    None なら全ノードのコライダーサマリーを返す。

    Args:
        filepath:  XMLプロジェクトファイルのパス
        node_name: ノード名（None なら全ノード）
    """
    root, project_dir = _load(filepath)
    nodes_e = root.find("nodes")
    results = []
    for elem in (nodes_e or []):
        nm = elem.findtext("name") or ""
        if node_name and nm != node_name:
            continue
        colliders_e = elem.find("colliders")
        cols = []
        for col in (colliders_e or []):
            data_e  = col.find("data")
            geom_e  = data_e.find("geometry") if data_e else None
            cols.append({
                "type":       col.findtext("type"),
                "enabled":    col.findtext("enabled") == "True",
                "shape":      data_e.findtext("type") if data_e else None,
                "geometry":   geom_e.text if geom_e is not None else None,
                "position":   (col.findtext("position") or "0 0 0").split(),
                "rotation_deg": (col.findtext("rotation") or "0 0 0").split(),
                "mesh_scale": (col.findtext("mesh_scale") or "1 1 1").split(),
            })
        results.append({"node": nm, "collider_count": len(cols), "colliders": cols})

    return {
        "robot_name": root.findtext("robot_name"),
        "queried_node": node_name,
        "nodes": results,
    }


@mcp.tool()
def export_urdf(
    filepath: str,
    output_dir: str,
    mesh_format: str = ".stl",
) -> dict:
    """
    プロジェクト XML から URDF を headless でエクスポートする。

    Parts XML から関節原点 (joint origin) を読み取り、正しい位置に joint を配置する。
    GUI なしで実行できる（Qt/VTK 不要）。

    出力構造:
        {output_dir}/{robot_name}_description/
            urdf/{robot_name}.urdf
            meshes/{*.stl}

    Args:
        filepath:    XMLプロジェクトファイルのパス
        output_dir:  出力先ディレクトリ
        mesh_format: ".stl" (デフォルト) または ".obj"

    Returns:
        {status, urdf_path, mesh_count, warnings}
    """
    root, project_dir = _load(filepath)
    robot_name = (root.findtext("robot_name") or "robot").replace(" ", "_")

    # Collect all nodes
    nodes_e = root.find("nodes")
    node_data: dict[str, dict] = {}
    for elem in (nodes_e or []):
        nd = _parse_node(elem, project_dir)
        nd["_xml_file_abs"] = (
            os.path.join(project_dir, elem.findtext("xml_file"))
            if elem.findtext("xml_file") else None
        )
        nd["_stl_abs"] = (
            os.path.join(project_dir, nd["stl_file"])
            if nd.get("stl_file") else None
        )
        nd["_color"] = (elem.findtext("color") or "0.5 0.5 0.5 1.0").split()
        node_data[nd["name"]] = nd

    # Connections: from_node, from_port, to_node
    conns = _connections(root)
    children_map: dict[str, list[tuple[str, str]]] = {}  # parent → [(child, from_port)]
    for c in conns:
        children_map.setdefault(c["from"], []).append((c["to"], c.get("from_port", "out")))

    # Prepare output directories
    desc_dir   = os.path.join(output_dir, f"{robot_name}_description")
    urdf_dir   = os.path.join(desc_dir, "urdf")
    meshes_dir = os.path.join(desc_dir, "meshes")
    for d in (desc_dir, urdf_dir, meshes_dir):
        os.makedirs(d, exist_ok=True)

    warnings: list[str] = []
    copied_meshes: list[str] = 0
    copied_meshes = []

    # Copy mesh files
    for name, nd in node_data.items():
        stl_abs = nd.get("_stl_abs")
        if stl_abs and os.path.exists(stl_abs):
            dst = os.path.join(meshes_dir, os.path.basename(stl_abs))
            shutil.copy2(stl_abs, dst)
            copied_meshes.append(os.path.basename(stl_abs))
        elif stl_abs:
            warnings.append(f"Mesh not found: {stl_abs}")

    # URDF writer helpers
    def urdf_joint_type(axis_name: str) -> tuple[str, str]:
        """(type_str, axis_xyz_str)"""
        return {
            "X":     ("revolute",  "1 0 0"),
            "Y":     ("revolute",  "0 1 0"),
            "Z":     ("revolute",  "0 0 1"),
            "Fixed": ("fixed",     "0 0 1"),
            "Free":  ("floating",  "0 0 1"),
            "Slide": ("prismatic", "1 0 0"),
        }.get(axis_name, ("revolute", "0 0 1"))

    # DFS tree walk
    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{robot_name}">')
    lines.append("")

    visited: set[str] = set()

    def write_link(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        nd = node_data.get(name)
        if nd is None:
            warnings.append(f"Node '{name}' referenced in connections but not found")
            return

        lines.append(f'  <link name="{name}">')

        # base_link: minimal
        if nd["type"] == "BaseLinkNode":
            lines.append("  </link>")
            lines.append("")
            return

        # inertial
        m = nd["mass_kg"]
        if m > 0 or nd.get("inertia"):
            i = nd.get("inertia", {})
            io = nd.get("inertial_origin") or {}
            io_xyz = " ".join(io.get("xyz", ["0", "0", "0"]))
            io_rpy = " ".join(io.get("rpy", ["0", "0", "0"]))
            lines.append(f'    <inertial>')
            lines.append(f'      <origin xyz="{io_xyz}" rpy="{io_rpy}"/>')
            lines.append(f'      <mass value="{_fmt(m)}"/>')
            lines.append(f'      <inertia ixx="{_fmt(i.get("ixx",0))}" '
                         f'ixy="{_fmt(i.get("ixy",0))}" ixz="{_fmt(i.get("ixz",0))}" '
                         f'iyy="{_fmt(i.get("iyy",0))}" iyz="{_fmt(i.get("iyz",0))}" '
                         f'izz="{_fmt(i.get("izz",0))}"/>')
            lines.append(f'    </inertial>')

        # visual
        stl_abs = nd.get("_stl_abs")
        if stl_abs:
            fn = os.path.basename(stl_abs)
            pkg = f"package://{robot_name}_description/meshes/{fn}"
            vo = nd.get("visual_origin") or {}
            vo_xyz = " ".join(vo.get("xyz", ["0", "0", "0"]))
            vo_rpy = " ".join(vo.get("rpy", ["0", "0", "0"]))
            col_rgb = nd["_color"][:3]
            lines.append(f'    <visual>')
            lines.append(f'      <origin xyz="{vo_xyz}" rpy="{vo_rpy}"/>')
            lines.append(f'      <geometry><mesh filename="{pkg}"/></geometry>')
            lines.append(f'      <material name="color_{name}">'
                         f'<color rgba="{col_rgb[0]} {col_rgb[1]} {col_rgb[2]} 1.0"/>'
                         f'</material>')
            lines.append(f'    </visual>')

            # collision (first enabled primitive collider, fallback to mesh)
            col_written = False
            for col in nd.get("colliders_summary", []):
                if not col.get("enabled"):
                    continue
                shape = col.get("shape")
                pos   = " ".join(col.get("pos", ["0","0","0"]))
                if shape == "box":
                    # geometry text is a dict-like string; skip detailed parse for now
                    lines.append(f'    <collision>')
                    lines.append(f'      <origin xyz="{pos}" rpy="0 0 0"/>')
                    lines.append(f'      <geometry><mesh filename="{pkg}"/></geometry>')
                    lines.append(f'    </collision>')
                    col_written = True
                    break
            if not col_written:
                lines.append(f'    <collision>')
                lines.append(f'      <origin xyz="{vo_xyz}" rpy="{vo_rpy}"/>')
                lines.append(f'      <geometry><mesh filename="{pkg}"/></geometry>')
                lines.append(f'    </collision>')

        lines.append(f'  </link>')
        lines.append("")

    def write_joint(parent: str, child: str, from_port: str) -> None:
        nd_child = node_data.get(child)
        nd_parent = node_data.get(parent)
        if nd_child is None:
            return
        if nd_child["massless_decoration"] or nd_child["is_imu_site"] or nd_child["is_camera_node"]:
            return

        # Joint origin from parent's parts XML
        origin_xyz = [0.0, 0.0, 0.0]
        origin_rpy = [0.0, 0.0, 0.0]
        if nd_parent:
            xml_abs = nd_parent.get("_xml_file_abs")
            points  = _read_parts_xml_points(xml_abs)
            pidx    = _port_name_to_index(from_port)
            if pidx < len(points):
                origin_xyz = points[pidx]["xyz"]
                origin_rpy = points[pidx]["angle"]

        jtype, jaxis = urdf_joint_type(nd_child["rotation_axis"])
        jname = f"{parent}_to_{child}"
        xyz_s = f"{_fmt(origin_xyz[0])} {_fmt(origin_xyz[1])} {_fmt(origin_xyz[2])}"
        rpy_s = f"{_fmt(origin_rpy[0])} {_fmt(origin_rpy[1])} {_fmt(origin_rpy[2])}"

        lines.append(f'  <joint name="{jname}" type="{jtype}">')
        lines.append(f'    <origin xyz="{xyz_s}" rpy="{rpy_s}"/>')
        lines.append(f'    <parent link="{parent}"/>')
        lines.append(f'    <child link="{child}"/>')

        if jtype not in ("fixed", "floating"):
            lines.append(f'    <axis xyz="{jaxis}"/>')
            lo  = math.radians(nd_child["joint_lower_deg"])
            hi  = math.radians(nd_child["joint_upper_deg"])
            eff = nd_child["joint_effort_Nm"]
            vel = nd_child["joint_velocity_rads"]
            dmp = nd_child["joint_damping"]
            fri = nd_child.get("joint_frictionloss", 0.0)
            lines.append(f'    <limit lower="{_fmt(lo)}" upper="{_fmt(hi)}" '
                         f'effort="{_fmt(eff)}" velocity="{_fmt(vel)}"/>')
            lines.append(f'    <dynamics damping="{_fmt(dmp)}" friction="{_fmt(fri)}"/>')
        elif jtype == "prismatic":
            lo  = nd_child.get("slide_lower", -0.05)
            hi  = nd_child.get("slide_upper",  0.05)
            eff = nd_child["joint_effort_Nm"]
            vel = nd_child["joint_velocity_rads"]
            lines.append(f'    <limit lower="{_fmt(lo)}" upper="{_fmt(hi)}" '
                         f'effort="{_fmt(eff)}" velocity="{_fmt(vel)}"/>')

        lines.append(f'  </joint>')
        lines.append("")

    # BFS from base_link
    from collections import deque
    queue = deque(["base_link"])
    write_link("base_link")
    while queue:
        parent = queue.popleft()
        for child, from_port in children_map.get(parent, []):
            write_joint(parent, child, from_port)
            write_link(child)
            queue.append(child)

    lines.append("</robot>")

    urdf_path = os.path.join(urdf_dir, f"{robot_name}.urdf")
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "status":      "ok",
        "urdf_path":   urdf_path,
        "desc_dir":    desc_dir,
        "mesh_count":  len(copied_meshes),
        "meshes":      copied_meshes,
        "warnings":    warnings,
        "warning_count": len(warnings),
    }


@mcp.tool()
def export_mjcf(
    filepath: str,
    output_dir: str,
    base_link_height: float | None = None,
    fix_base_to_ground: bool = False,
    mesh_max_faces: int = 200000,
) -> dict:
    """
    プロジェクト XML から MuJoCo MJCF を headless でエクスポートする。

    STL/DAE/OBJ メッシュを自動で OBJ に変換して assets/ に配置する。
    GUI なしで実行できる（Qt/VTK 不要）。trimesh が必要（pip install trimesh）。

    出力構造:
        {output_dir}/{robot_name}_mjcf/
            model.xml          ← MJCF 本体
            assets/{*.obj}     ← 変換済みメッシュ

    Args:
        filepath:           XMLプロジェクトファイルのパス
        output_dir:         出力先ディレクトリ
        base_link_height:   ルートボディの z 初期位置 [m]。None でプロジェクト設定値を使用
        fix_base_to_ground: True でルートを地面に固定（freejoint を付けない）
        mesh_max_faces:     この面数を超えるメッシュはスキップ（デフォルト 200000）

    Returns:
        {status, mjcf_path, mesh_count, skipped_meshes, actuator_count, warnings}
    """
    try:
        import trimesh as _tm
    except ImportError:
        return {"status": "error", "detail": "trimesh not installed: pip install trimesh"}

    import ast, re

    root, project_dir = _load(filepath)
    robot_name = (root.findtext("robot_name") or "robot").replace(" ", "_")

    # ── project-level defaults ────────────────────────────────────────────────
    mj = root.find("mjcf_defaults") or ET.Element("mjcf_defaults")
    djs = root.find("default_joint_settings") or ET.Element("default_joint_settings")

    def mj_val(tag, fallback):
        v = mj.findtext(tag)
        return float(v) if v else fallback

    def djs_val(tag, fallback):
        v = djs.findtext(tag)
        return float(v) if v else fallback

    opt_timestep   = mj_val("option_timestep",   0.002)
    opt_iterations = int(mj_val("option_iterations", 50))
    opt_impratio   = mj_val("option_impratio",   100.0)
    def_jdamp      = mj_val("joint_damping",      0.1)
    def_gfriction  = mj_val("geom_friction",      0.4)
    def_gmargin    = mj_val("geom_margin",        0.001)
    def_gcondim    = int(mj_val("geom_condim",    3))
    def_armature   = djs_val("armature",          0.01)
    def_frictionloss = djs_val("frictionloss",    0.005)
    def_timeconst  = djs_val("timeconst",         0.01)

    proj_height = float(root.findtext("base_link_height") or 0.5)
    z_root = proj_height if base_link_height is None else base_link_height

    # ── collect nodes ─────────────────────────────────────────────────────────
    nodes_e = root.find("nodes") or ET.Element("nodes")
    node_data: dict[str, dict] = {}
    for elem in nodes_e:
        nd = _parse_node(elem, project_dir)
        nd["_stl_abs"] = (
            os.path.join(project_dir, nd["stl_file"]) if nd.get("stl_file") else None
        )
        nd["_xml_file_abs"] = (
            os.path.join(project_dir, elem.findtext("xml_file"))
            if elem.findtext("xml_file") else None
        )
        nd["_color"] = (elem.findtext("color") or "0.5 0.5 0.5 1.0").split()
        # slide limits
        nd["slide_lower"] = float(elem.findtext("slide_lower") or -0.05)
        nd["slide_upper"] = float(elem.findtext("slide_upper") or  0.05)
        nd["slide_axis"]  = int(elem.findtext("slide_axis") or 0)
        # collider raw elements
        nd["_collider_elems"] = list(elem.find("colliders") or [])
        node_data[nd["name"]] = nd

    # ── connections: parent → [(child, from_port)] ────────────────────────────
    conns_e = root.find("connections") or ET.Element("connections")
    children_map: dict[str, list[tuple[str, str]]] = {}
    for c in conns_e:
        children_map.setdefault(c.findtext("from_node"), []).append(
            (c.findtext("to_node"), c.findtext("from_port") or "out")
        )

    # ── prepare output dirs ───────────────────────────────────────────────────
    mjcf_dir   = os.path.join(output_dir, f"{robot_name}_mjcf")
    assets_dir = os.path.join(mjcf_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    warnings:       list[str] = []
    converted:      list[str] = []
    skipped_meshes: list[str] = []
    mesh_name_map:  dict[str, str] = {}   # stl_abs → asset_name (without ext)

    # ── mesh conversion STL/DAE/OBJ → OBJ ────────────────────────────────────
    for name, nd in node_data.items():
        if nd["massless_decoration"] or nd["hide_mesh"] or nd["is_imu_site"] or nd["is_camera_node"]:
            continue
        stl_abs = nd.get("_stl_abs")
        if not stl_abs or not os.path.exists(stl_abs):
            if stl_abs:
                warnings.append(f"Mesh not found: {stl_abs}")
            continue
        try:
            mesh = _tm.load(stl_abs, force="mesh")
            if hasattr(mesh, "geometry"):           # Scene
                meshes = list(mesh.geometry.values())
                mesh = meshes[0] if meshes else None
            if mesh is None:
                warnings.append(f"Empty mesh: {stl_abs}")
                continue
            nf = len(mesh.faces) if hasattr(mesh, "faces") else 0
            if nf > mesh_max_faces:
                skipped_meshes.append(f"{name} ({nf} faces)")
                warnings.append(f"Skipped (too many faces {nf}): {stl_abs}")
                continue
            stem = os.path.splitext(os.path.basename(stl_abs))[0]
            obj_name = f"{stem}.obj"
            mesh.export(os.path.join(assets_dir, obj_name))
            mesh_name_map[stl_abs] = stem        # asset name without extension
            converted.append(obj_name)
        except Exception as e:
            warnings.append(f"Mesh conversion failed ({name}): {e}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _mjcf_fmt(v: float) -> str:
        if v == 0.0:
            return "0"
        s = f"{v:.9g}"
        return s

    def _parse_geom_dict(geom_text: str) -> dict:
        """Parse Python-dict-like string from collider geometry field."""
        try:
            return ast.literal_eval(geom_text)
        except Exception:
            # fallback: regex key-value extraction
            return {k: v for k, v in re.findall(r"'(\w+)':\s*'([^']*)'", geom_text or "")}

    def _col_geom_str(col_elem: ET.Element) -> str | None:
        """Return '<geom class="collision" ...>' string for a collider element, or None."""
        if col_elem.findtext("enabled") != "True":
            return None
        col_type = col_elem.findtext("type")
        pos  = (col_elem.findtext("position") or "0 0 0")
        rot  = (col_elem.findtext("rotation") or "0 0 0")
        pos_vals = [float(v) for v in pos.split()]
        rot_vals = [float(v) for v in rot.split()]
        pos_str  = " ".join(_mjcf_fmt(v) for v in pos_vals)
        euler_str = " ".join(_mjcf_fmt(v) for v in rot_vals)

        data_e = col_elem.find("data")
        if data_e is None:
            return None
        shape   = data_e.findtext("type") or ""
        geom_e  = data_e.find("geometry")
        geom_txt = geom_e.text if geom_e is not None else ""
        gd = _parse_geom_dict(geom_txt)

        attr = f'class="collision" pos="{pos_str}"'
        if any(v != 0.0 for v in rot_vals):
            attr += f' euler="{euler_str}"'

        if shape == "box":
            sx = float(gd.get("size_x", 0.01)) / 2
            sy = float(gd.get("size_y", 0.01)) / 2
            sz = float(gd.get("size_z", 0.01)) / 2
            return f'<geom {attr} type="box" size="{_mjcf_fmt(sx)} {_mjcf_fmt(sy)} {_mjcf_fmt(sz)}"/>'
        elif shape == "sphere":
            r = float(gd.get("radius", 0.01))
            return f'<geom {attr} type="sphere" size="{_mjcf_fmt(r)}"/>'
        elif shape == "cylinder":
            r = float(gd.get("radius", 0.01))
            h = float(gd.get("height", 0.02)) / 2
            return f'<geom {attr} type="cylinder" size="{_mjcf_fmt(r)} {_mjcf_fmt(h)}"/>'
        elif col_type == "mesh":
            mesh_ref = col_elem.findtext("mesh") or ""
            stem = os.path.splitext(mesh_ref)[0]
            if stem:
                return f'<geom {attr} type="mesh" mesh="{stem}"/>'
        return None

    # ── recursive MJCF body writer ────────────────────────────────────────────
    actuators: list[dict] = []
    imu_sites: list[str]  = []

    def write_body(f, name: str, pos_xyz: list, indent: int, is_root: bool = False) -> None:
        nd = node_data.get(name)
        if nd is None:
            return
        if nd["massless_decoration"] or nd["hide_mesh"]:
            return

        pad = " " * indent
        pos_s = " ".join(_mjcf_fmt(v) for v in pos_xyz)

        # body open
        f.write(f'{pad}<body name="{name}" pos="{pos_s}">\n')

        # root: freejoint or fixed
        if is_root:
            if not fix_base_to_ground:
                f.write(f'{pad}  <freejoint />\n')
        else:
            # joint
            axis_name = nd["rotation_axis"]
            jname = name
            lo  = math.radians(nd["joint_lower_deg"])
            hi  = math.radians(nd["joint_upper_deg"])
            damp = nd["joint_damping"]
            fric = nd.get("joint_frictionloss", 0.0)
            arm  = nd["joint_armature"]

            if axis_name in ("X", "Y", "Z"):
                axis_vec = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}[axis_name]
                f.write(f'{pad}  <joint name="{jname}" type="hinge" '
                        f'axis="{axis_vec}" range="{_mjcf_fmt(lo)} {_mjcf_fmt(hi)}" '
                        f'damping="{_mjcf_fmt(damp)}" frictionloss="{_mjcf_fmt(fric)}" '
                        f'armature="{_mjcf_fmt(arm)}"/>\n')
                actuators.append({
                    "joint":  jname,
                    "name":   f"{jname}_actuator",
                    "kp":     nd["joint_stiffness"],
                    "kv":     nd.get("joint_kv", 1.0),
                    "effort": nd["joint_effort_Nm"],
                    "lo":     lo, "hi": hi,
                })
            elif axis_name == "Slide":
                s_ax = {0: "1 0 0", 1: "0 1 0", 2: "0 0 1"}.get(nd["slide_axis"], "1 0 0")
                sl   = nd["slide_lower"]
                su   = nd["slide_upper"]
                f.write(f'{pad}  <joint name="{jname}" type="slide" '
                        f'axis="{s_ax}" range="{_mjcf_fmt(sl)} {_mjcf_fmt(su)}" '
                        f'damping="{_mjcf_fmt(damp)}" frictionloss="{_mjcf_fmt(fric)}" '
                        f'armature="{_mjcf_fmt(arm)}"/>\n')
                actuators.append({
                    "joint":  jname,
                    "name":   f"{jname}_actuator",
                    "kp":     nd["joint_stiffness"],
                    "kv":     nd.get("joint_kv", 1.0),
                    "effort": nd["joint_effort_Nm"],
                    "lo":     sl, "hi": su,
                })
            elif axis_name == "Free":
                f.write(f'{pad}  <joint name="{jname}" type="free"/>\n')
            # Fixed: no joint tag

        # inertial
        m = nd["mass_kg"]
        inertia = nd.get("inertia", {})
        if m > 0 and inertia:
            io = nd.get("inertial_origin") or {}
            com_xyz = " ".join(io.get("xyz", ["0", "0", "0"]))
            ixx = inertia.get("ixx", 1e-6)
            ixy = inertia.get("ixy", 0)
            ixz = inertia.get("ixz", 0)
            iyy = inertia.get("iyy", 1e-6)
            iyz = inertia.get("iyz", 0)
            izz = inertia.get("izz", 1e-6)
            # MuJoCo 3.x fullinertia order: ixx iyy izz ixy ixz iyz
            fi = (f"{_mjcf_fmt(ixx)} {_mjcf_fmt(iyy)} {_mjcf_fmt(izz)} "
                  f"{_mjcf_fmt(ixy)} {_mjcf_fmt(ixz)} {_mjcf_fmt(iyz)}")
            f.write(f'{pad}  <inertial pos="{com_xyz}" '
                    f'mass="{_mjcf_fmt(m)}" fullinertia="{fi}"/>\n')

        # visual geom
        stl_abs = nd.get("_stl_abs")
        aname = mesh_name_map.get(stl_abs) if stl_abs else None
        if aname:
            col = nd["_color"]
            r, g, b = col[0], col[1], col[2]
            rgba = f"{r} {g} {b} 1.0"
            vo = nd.get("visual_origin") or {}
            vpos = " ".join(vo.get("xyz", ["0", "0", "0"]))
            f.write(f'{pad}  <geom class="visual" type="mesh" mesh="{aname}" '
                    f'rgba="{rgba}" pos="{vpos}"/>\n')

        # collision geoms
        for col_elem in nd["_collider_elems"]:
            g = _col_geom_str(col_elem)
            if g:
                f.write(f'{pad}  {g}\n')

        # children
        for child_name, from_port in children_map.get(name, []):
            child_nd = node_data.get(child_name)
            if child_nd is None:
                continue

            # IMU site → emit <site> here, not a body
            if child_nd["is_imu_site"]:
                pidx = _port_name_to_index(from_port)
                pts  = _read_parts_xml_points(nd.get("_xml_file_abs"))
                spos = pts[pidx]["xyz"] if pidx < len(pts) else [0, 0, 0]
                sp   = " ".join(_mjcf_fmt(v) for v in spos)
                sname = child_name
                f.write(f'{pad}  <site name="{sname}" type="box" '
                        f'size="0.01 0.01 0.01" pos="{sp}"/>\n')
                imu_sites.append(sname)
                continue

            # Camera node → emit <camera> here
            if child_nd["is_camera_node"]:
                pidx = _port_name_to_index(from_port)
                pts  = _read_parts_xml_points(nd.get("_xml_file_abs"))
                cpos = pts[pidx]["xyz"] if pidx < len(pts) else [0, 0, 0]
                cp   = " ".join(_mjcf_fmt(v) for v in cpos)
                f.write(f'{pad}  <camera name="{child_name}" pos="{cp}"/>\n')
                continue

            # skip massless / hidden children (they were added as geoms in parent)
            if child_nd["massless_decoration"] or child_nd["hide_mesh"]:
                continue

            # joint origin: from parent's parts XML point
            pidx     = _port_name_to_index(from_port)
            pts      = _read_parts_xml_points(nd.get("_xml_file_abs"))
            jpos_xyz = pts[pidx]["xyz"] if pidx < len(pts) else [0.0, 0.0, 0.0]

            write_body(f, child_name, jpos_xyz, indent + 2)

        f.write(f'{pad}</body>\n')

    # ── find root node ────────────────────────────────────────────────────────
    has_parent = {c for pairs in children_map.values() for c, _ in pairs}
    raw_roots = [n for n in node_data if n not in has_parent]

    # BaseLinkNode is massless virtual root — promote its children as effective roots
    effective_roots: list[tuple[str, list]] = []
    for r in raw_roots:
        nd_r = node_data.get(r, {})
        if nd_r.get("mass_kg", 0) <= 0 and not nd_r.get("_stl_abs"):
            for child_name, from_port in children_map.get(r, []):
                effective_roots.append((child_name, [0.0, 0.0, z_root]))
        else:
            effective_roots.append((r, [0.0, 0.0, z_root]))

    # ── write MJCF ────────────────────────────────────────────────────────────
    mjcf_path = os.path.join(mjcf_dir, "model.xml")
    with open(mjcf_path, "w", encoding="utf-8") as f:
        f.write(f'<mujoco model="{robot_name}">\n')
        f.write(f'  <compiler angle="radian" meshdir="assets" autolimits="true"/>\n\n')
        f.write(f'  <option timestep="{opt_timestep}" iterations="{opt_iterations}" '
                f'cone="elliptic" impratio="{opt_impratio}"/>\n\n')

        # defaults
        f.write('  <default>\n')
        f.write(f'    <joint damping="{def_jdamp}" armature="{def_armature}" '
                f'frictionloss="{def_frictionloss}"/>\n')
        f.write(f'    <position inheritrange="1" timeconst="{def_timeconst}"/>\n')
        f.write(f'    <geom friction="{def_gfriction}" margin="{def_gmargin}" '
                f'condim="{def_gcondim}"/>\n')
        f.write('    <default class="collision"><geom group="0"/></default>\n')
        f.write('    <default class="visual"><geom contype="0" conaffinity="0" group="1"/></default>\n')
        f.write('  </default>\n\n')

        # asset
        f.write('  <asset>\n')
        f.write('    <material name="metal"  rgba=".9 .95 .95 1"/>\n')
        f.write('    <material name="black"  rgba="0 0 0 1"/>\n')
        f.write('    <material name="white"  rgba="1 1 1 1"/>\n')
        f.write('    <material name="gray"   rgba="0.67 0.69 0.77 1"/>\n')
        for stl_abs, aname in mesh_name_map.items():
            f.write(f'    <mesh name="{aname}" file="{aname}.obj"/>\n')
        f.write('  </asset>\n\n')

        # worldbody
        f.write('  <worldbody>\n')
        for root_name, root_pos in effective_roots:
            write_body(f, root_name, root_pos, indent=4, is_root=True)
        f.write('  </worldbody>\n\n')

        # actuators
        if actuators:
            f.write('  <actuator>\n')
            for act in actuators:
                kp  = _mjcf_fmt(act["kp"])
                kv  = _mjcf_fmt(act["kv"])
                fr  = _mjcf_fmt(act["effort"])
                lo  = _mjcf_fmt(act["lo"])
                hi  = _mjcf_fmt(act["hi"])
                f.write(f'    <position name="{act["name"]}" joint="{act["joint"]}" '
                        f'gear="1" kp="{kp}" kv="{kv}" '
                        f'forcerange="-{fr} {fr}" forcelimited="true"/>\n')
            f.write('  </actuator>\n\n')

        # sensors
        f.write('  <sensor>\n')
        if imu_sites:
            for sname in imu_sites:
                f.write(f'    <accelerometer name="{sname}_accel" site="{sname}"/>\n')
                f.write(f'    <gyro name="{sname}_gyro" site="{sname}"/>\n')
        else:
            f.write('    <!-- Add sensors here if needed -->\n')
        f.write('  </sensor>\n')

        f.write('</mujoco>\n')

    return {
        "status":         "ok",
        "mjcf_path":      mjcf_path,
        "mjcf_dir":       mjcf_dir,
        "mesh_count":     len(converted),
        "actuator_count": len(actuators),
        "imu_sites":      imu_sites,
        "skipped_meshes": skipped_meshes,
        "warnings":       warnings,
        "warning_count":  len(warnings),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
