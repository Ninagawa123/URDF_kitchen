"""Robot joint/link name conversion and rename utilities for LegacyMotionEditor.

Consolidated from robot_name_converter package. Data files are loaded from
a ``data/`` directory located next to this file.

Public API (mirrors the former robot_name_converter package):
  NameConverter                  -- core converter class
  plan_joint_rename(joint_editor)
  apply_joint_rename_plan(...)
  overwrite_loaded_model_file(...)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# stdlib
# ---------------------------------------------------------------------------
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
_MASTER_PATH = Path(__file__).resolve().parent / "robot_label_bridge_master.json"

# ===========================================================================
# models
# ===========================================================================

class EntityType(str, Enum):
    JOINT = "joint"
    LINK = "link"
    AUTO = "auto"


def normalize_entity_type(entity: str | EntityType) -> str:
    """Normalize external entity labels to canonical master entities."""
    if isinstance(entity, EntityType):
        return entity.value
    text = str(entity or EntityType.AUTO.value).strip().lower()
    if text in ("servo", "joint"):
        return EntityType.JOINT.value
    if text == EntityType.LINK.value:
        return EntityType.LINK.value
    return text or EntityType.AUTO.value


class ConversionStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "proposed": 1}


@dataclass
class Candidate:
    target: str
    entity: str
    confidence: str = "medium"
    source: str = ""
    mapping_type: str = "direct"
    notes: str = ""
    score: float = 0.0

    def confidence_rank(self) -> int:
        return CONFIDENCE_RANK.get(self.confidence, 0)


@dataclass
class ConversionResult:
    source: str
    normalized: str
    entity: str
    status: ConversionStatus
    target: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == ConversionStatus.RESOLVED and self.target is not None


@dataclass
class ModelConversionResult:
    joints: dict[str, ConversionResult] = field(default_factory=dict)
    links: dict[str, ConversionResult] = field(default_factory=dict)

    @property
    def resolved_joint_map(self) -> dict[str, str]:
        return {k: v.target for k, v in self.joints.items() if v.resolved and v.target}

    @property
    def resolved_link_map(self) -> dict[str, str]:
        return {k: v.target for k, v in self.links.items() if v.resolved and v.target}

    def unresolved(self) -> list[tuple[str, str, ConversionResult]]:
        out: list[tuple[str, str, ConversionResult]] = []
        for name, result in self.joints.items():
            if not result.resolved:
                out.append(("joint", name, result))
        for name, result in self.links.items():
            if not result.resolved:
                out.append(("link", name, result))
        return out


# ===========================================================================
# normalize
# ===========================================================================

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])|(?<=[a-z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[\s\-/]+")
_MULTI_UNDERSCORE = re.compile(r"_+")
_SUFFIXES = ("_joint", "_link", "joint", "link")


def normalize_name(name: str) -> str:
    if not name:
        return ""
    text = str(name).strip()
    text = _CAMEL_BOUNDARY.sub("_", text)
    text = text.lower()
    text = _NON_ALNUM.sub("_", text)
    text = _MULTI_UNDERSCORE.sub("_", text)
    return text.strip("_")


def normalize_variants(name: str) -> list[str]:
    base = normalize_name(name)
    variants = [base]
    for suffix in _SUFFIXES:
        if base.endswith(suffix) and base not in variants:
            stripped = base[: -len(suffix)].rstrip("_")
            if stripped and stripped not in variants:
                variants.append(stripped)
    return variants


def split_parent_child(name: str) -> tuple[str, str] | None:
    normalized = normalize_name(name)
    if "_to_" not in normalized:
        return None
    parent, child = normalized.split("_to_", 1)
    if parent and child:
        return parent, child
    return None


# ===========================================================================
# policy
# ===========================================================================

PRESERVED_LINK_NAMES = frozenset({"base_link", "c_base_link"})


def is_preserved_link(name: str) -> bool:
    return normalize_name(name) in PRESERVED_LINK_NAMES


def preserved_link_target(name: str) -> str | None:
    normalized = normalize_name(name)
    if normalized in PRESERVED_LINK_NAMES:
        return normalized
    return None


def compact_link_name(name: str) -> str:
    normalized = normalize_name(name)
    if normalized in PRESERVED_LINK_NAMES:
        return normalized
    if normalized.endswith("_link"):
        return normalized[: -len("_link")]
    return normalized


# ===========================================================================
# axis
# ===========================================================================

AXIS_SHORTS = ("xr", "yp", "zy")
AXIS_ROS = ("roll", "pitch", "yaw")
AXIS_KEYWORDS = {
    "xr": ("xroll", "roll", "xr"),
    "yp": ("ypitch", "pitch", "yp"),
    "zy": ("zyaw", "yaw", "zy"),
}


def _normalize_vec(axis: Iterable[float]) -> list[float] | None:
    vals = [float(v) for v in axis]
    if len(vals) < 3:
        return None
    norm = math.sqrt(sum(v * v for v in vals[:3]))
    if norm < 1e-9:
        return None
    return [vals[0] / norm, vals[1] / norm, vals[2] / norm]


def _rotation_matrix_from_rpy(rpy: Iterable[float]) -> list[list[float]]:
    roll, pitch, yaw = (float(v) for v in list(rpy)[:3])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotation_matrix_from_quat(quat: Iterable[float]) -> list[list[float]]:
    w, x, y, z = (float(v) for v in list(quat)[:4])
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _matvec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [
        matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2],
        matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2],
        matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2],
    ]


def resolve_effective_joint_axis(
    axis: Iterable[float] | None,
    *,
    origin_rpy: Iterable[float] | None = None,
    origin_quat: Iterable[float] | None = None,
) -> list[float] | None:
    vec = _normalize_vec(axis) if axis is not None else None
    if vec is None:
        return None
    rotation = None
    if origin_quat is not None and len(list(origin_quat)) >= 4:
        quat = list(origin_quat)[:4]
        if any(abs(v - (1.0 if i == 0 else 0.0)) > 1e-9 for i, v in enumerate(quat)):
            rotation = _rotation_matrix_from_quat(quat)
    elif origin_rpy is not None and len(list(origin_rpy)) >= 3:
        rpy = list(origin_rpy)[:3]
        if any(abs(v) > 1e-9 for v in rpy):
            rotation = _rotation_matrix_from_rpy(rpy)
    if rotation is None:
        return vec
    return _normalize_vec(_matvec(rotation, vec))


def axis_vector_to_short(axis: Iterable[float] | None, master_axis_tokens: dict | None = None) -> str | None:
    vec = _normalize_vec(axis) if axis is not None else None
    if vec is None:
        return None
    if master_axis_tokens:
        best_short = None
        best_dot = -1.0
        for pair_name, info in master_axis_tokens.items():
            ref = info.get("axis")
            if not ref or len(ref) < 3:
                continue
            ref_vec = _normalize_vec(ref)
            if ref_vec is None:
                continue
            dot = abs(sum(a * b for a, b in zip(vec, ref_vec)))
            short = info.get("short")
            if dot > best_dot and short:
                best_dot = dot
                best_short = short
        if best_short:
            return best_short
    abs_vals = [abs(v) for v in vec[:3]]
    idx = abs_vals.index(max(abs_vals))
    return AXIS_SHORTS[idx]


def axis_keyword_to_short(name: str) -> str | None:
    normalized = name.lower()
    for short, keywords in AXIS_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                return short
    return None


def target_matches_axis(target: str, axis_short: str | None) -> bool:
    if not axis_short:
        return True
    return target.endswith(f"_{axis_short}")


# ===========================================================================
# side
# ===========================================================================

_SIDE_PREFIX = re.compile(r"^(l|r|c|left|right|center|centre)_")
_SIDE_TOKEN = re.compile(r"(?:^|_)(l|r|left|right)(?:_|$)")


def detect_side(name: str) -> str | None:
    normalized = name.lower().strip()
    m = _SIDE_PREFIX.match(normalized)
    if m:
        token = m.group(1)
        if token in ("l", "left"):
            return "l"
        if token in ("r", "right"):
            return "r"
        return "c"
    m2 = _SIDE_TOKEN.search(normalized)
    if m2:
        token = m2.group(1)
        return "l" if token in ("l", "left") else "r"
    return None


def apply_side(target_pattern: str, side: str) -> str:
    return (
        target_pattern.replace("l_/r_", f"{side}_")
        .replace("{side}", side)
        .replace("l/r", side)
    )


def default_side(single_arm_as_left: bool) -> str:
    return "l" if single_arm_as_left else "c"


def infer_actuated_side(parent: str | None, child: str | None) -> str | None:
    child_side = detect_side(child) if child else None
    parent_side = detect_side(parent) if parent else None
    if child_side in ("l", "r") and parent_side in (None, "c"):
        return child_side
    if child_side == "c" and parent_side is None:
        return "c"
    if child_side in ("l", "r"):
        return child_side
    if parent_side:
        return parent_side
    return None


# ===========================================================================
# master  (data loaded from ./data/robot_name_conversion_master.json)
# ===========================================================================

@lru_cache(maxsize=1)
def load_master() -> dict[str, Any]:
    path = _MASTER_PATH
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def reload_master() -> dict[str, Any]:
    """Reload master JSON from disk and reset cached converter state."""
    load_master.cache_clear()
    global _nc_singleton
    _nc_singleton = None
    return load_master()


def get_master(use_full: bool = False) -> dict[str, Any]:
    """Return the master data dict. use_full is kept for API compatibility."""
    return load_master()


# ===========================================================================
# NameConverter
# ===========================================================================

_LOGICAL_JOINT_RE = re.compile(
    r"(?:^|_)(shoulder|elbow|wrist|hip|ankle|knee|neck|spine|chest|base)_joint$"
)
_GENERIC_JOINT_RE = re.compile(r"^joint\d+$")
_LEGACY_LINK_ALIASES: dict[str, str] = {
    "c_waist": "c_pelvis",
    "c_waist_link": "c_pelvis",
    "waist": "c_pelvis",
    "waist_link": "c_pelvis",
}
_LEGACY_WAIST_SOURCE_NAMES = frozenset(_LEGACY_LINK_ALIASES)
_LEGACY_WAIST_CANONICAL_CHILDREN = frozenset({"c_pelvis", "pelvis"})


class NameConverter:
    """Convert legacy URDF/MJCF names to canonical joint/link short names."""

    def __init__(
        self,
        *,
        profile: str = "",
        morphology: str = "humanoid",
        single_arm_as_left: bool = True,
        use_full_master: bool = False,  # kept for API compatibility, no-op
        min_confidence: str = "medium",
    ) -> None:
        self.profile = profile
        self.morphology = morphology
        self.single_arm_as_left = single_arm_as_left
        self.min_confidence = min_confidence
        self._master = load_master()
        self._alias_index: dict[str, list[dict[str, Any]]] = self._master.get("alias_index", {})
        self._contextual_aliases: list[dict[str, Any]] = self._master.get("contextual_aliases", [])
        self._axis_tokens = self._master.get("axis_tokens", {})

    # ── public API ────────────────────────────────────────────────────────

    def convert(
        self,
        source_name: str,
        *,
        entity: str | EntityType = EntityType.AUTO,
        axis: Iterable[float] | None = None,
        parent: str | None = None,
        child: str | None = None,
        side: str | None = None,
        origin_rpy: Iterable[float] | None = None,
        origin_quat: Iterable[float] | None = None,
    ) -> ConversionResult:
        normalized = normalize_name(source_name)
        entity_type = normalize_entity_type(self._resolve_entity_type(source_name, normalized, entity))

        preserved = preserved_link_target(source_name)
        if preserved and entity_type == EntityType.LINK.value:
            return self._finalize_link_result(
                ConversionResult(
                    source=source_name,
                    normalized=normalized,
                    entity=EntityType.LINK.value,
                    status=ConversionStatus.RESOLVED,
                    target=preserved,
                    reasons=[],
                    metadata={"policy": "preserved_link"},
                )
            )

        legacy_target = _LEGACY_LINK_ALIASES.get(normalized)
        if legacy_target and entity_type == EntityType.LINK.value:
            return self._finalize_link_result(
                ConversionResult(
                    source=source_name,
                    normalized=normalized,
                    entity=EntityType.LINK.value,
                    status=ConversionStatus.RESOLVED,
                    target=legacy_target,
                    reasons=[],
                    metadata={"policy": "legacy_link_alias"},
                ),
                entity_type=entity_type,
            )

        if self._is_logical_joint_without_dof(normalized, entity_type):
            return ConversionResult(
                source=source_name,
                normalized=normalized,
                entity=entity_type,
                status=ConversionStatus.UNRESOLVED,
                reasons=[
                    "Logical joint name without explicit DOF; refusing to guess axis.",
                ],
                metadata={"policy": "never_guess_logical_joint_dof"},
            )

        if not (parent and child):
            pc = split_parent_child(source_name)
            if pc:
                parent, child = pc

        resolved_side = side or infer_actuated_side(parent, child)
        if resolved_side is None and entity_type in (EntityType.JOINT.value, EntityType.AUTO.value):
            resolved_side = detect_side(normalized) or default_side(self.single_arm_as_left)

        effective_axis = resolve_effective_joint_axis(
            axis,
            origin_rpy=origin_rpy,
            origin_quat=origin_quat,
        )

        candidates = self._collect_candidates(
            source_name,
            normalized,
            entity_type,
            axis=effective_axis,
            parent=parent,
            child=child,
            side=resolved_side,
        )

        if not candidates:
            return self._finalize_link_result(
                ConversionResult(
                    source=source_name,
                    normalized=normalized,
                    entity=entity_type,
                    status=ConversionStatus.UNRESOLVED,
                    reasons=["No alias or heuristic match found."],
                ),
                entity_type=entity_type,
            )

        resolved = self._pick_best(
            candidates,
            entity_type,
            axis=effective_axis,
            parent=parent,
            child=child,
            side=resolved_side,
        )
        if resolved.status == ConversionStatus.RESOLVED:
            resolved.source = source_name
            resolved.normalized = normalized
            return self._finalize_link_result(resolved, entity_type=entity_type)

        return self._finalize_link_result(
            ConversionResult(
                source=source_name,
                normalized=normalized,
                entity=entity_type,
                status=ConversionStatus.AMBIGUOUS,
                candidates=candidates,
                reasons=resolved.reasons or ["Multiple candidates remain after disambiguation."],
            ),
            entity_type=entity_type,
        )

    def convert_model(
        self,
        *,
        joints: list[dict[str, Any]] | None = None,
        links: list[dict[str, Any]] | None = None,
    ) -> ModelConversionResult:
        joint_results: dict[str, ConversionResult] = {}
        link_results: dict[str, ConversionResult] = {}

        for item in joints or []:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            joint_results[name] = self.convert(
                name,
                entity=EntityType.JOINT,
                axis=item.get("axis"),
                parent=item.get("parent"),
                child=item.get("child"),
                side=item.get("side"),
                origin_rpy=item.get("origin_rpy"),
                origin_quat=item.get("origin_quat"),
            )

        for item in links or []:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            link_results[name] = self.convert(
                name,
                entity=EntityType.LINK,
                parent=item.get("parent"),
                child=item.get("child"),
                side=item.get("side"),
            )

        return ModelConversionResult(joints=joint_results, links=link_results)

    def lookup_alias(self, normalized_name: str, entity: str | None = None) -> list[Candidate]:
        return self._lookup_alias_index(normalized_name, entity)

    # ── candidate collection ──────────────────────────────────────────────

    def _collect_candidates(
        self,
        source_name: str,
        normalized: str,
        entity_type: str,
        *,
        axis: Iterable[float] | None,
        parent: str | None,
        child: str | None,
        side: str | None,
    ) -> list[Candidate]:
        found: list[Candidate] = []
        seen: set[tuple[str, str]] = set()

        def add(candidate: Candidate) -> None:
            key = (candidate.target, candidate.entity)
            if key in seen:
                return
            if entity_type != EntityType.AUTO.value and candidate.entity != entity_type:
                return
            seen.add(key)
            found.append(candidate)

        for variant in normalize_variants(source_name):
            for cand in self._lookup_alias_index(variant, entity_type):
                add(cand)

        for cand in self._contextual_pattern_candidates(normalized, entity_type, side):
            add(cand)

        for cand in self._legacy_waist_joint_candidates(
            source_name,
            normalized,
            parent,
            child,
            entity_type,
            axis,
        ):
            add(cand)

        for cand in self._root_waist_candidates(source_name, parent, child, entity_type, axis):
            add(cand)

        for cand in self._heuristic_candidates(source_name, normalized, entity_type, axis, side):
            add(cand)

        if parent and child:
            for cand in self._topology_candidates(parent, child, entity_type, axis, side):
                add(cand)
            for cand in self._link_tree_path_candidates(parent, child, entity_type, axis):
                add(cand)

        return found

    def _lookup_alias_index(self, normalized: str, entity: str | None) -> list[Candidate]:
        rows = self._alias_index.get(normalized, [])
        entity = normalize_entity_type(entity) if entity else None
        out: list[Candidate] = []
        for row in rows:
            ent = normalize_entity_type(row.get("entity", ""))
            if entity and entity != EntityType.AUTO.value and ent != entity:
                continue
            if self.profile and row.get("profile") and row.get("profile") != self.profile:
                continue
            target = row.get("target", "")
            if ent == EntityType.LINK.value:
                target = compact_link_name(target)
            out.append(
                Candidate(
                    target=target,
                    entity=ent,
                    confidence=row.get("confidence", "medium"),
                    source=row.get("source", ""),
                    mapping_type=row.get("mapping_type", "direct"),
                    notes=row.get("notes", ""),
                )
            )
        return [c for c in out if c.target]

    def _contextual_pattern_candidates(
        self,
        normalized: str,
        entity_type: str,
        side: str | None,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        side_val = side or detect_side(normalized) or default_side(self.single_arm_as_left)

        for entry in self._contextual_aliases:
            alias = entry.get("alias")
            alias_pattern = entry.get("alias_pattern")
            ent = entry.get("entity") or entry.get("entity_interpretation", "")
            if entity_type != EntityType.AUTO.value:
                if ent and "link" in ent.lower() and entity_type != EntityType.LINK.value:
                    continue
                if ent and "joint" in ent.lower() and entity_type != EntityType.JOINT.value:
                    continue
                if ent == "link" and entity_type != EntityType.LINK.value:
                    continue

            matched = False
            if alias and normalize_name(alias) == normalized:
                matched = True
            elif alias_pattern and alias_pattern.replace(" ", "_") in normalized:
                matched = True
            elif alias_pattern and normalized.endswith(alias_pattern):
                matched = True

            if not matched:
                continue

            target_pattern = entry.get("target_pattern") or entry.get("target_hint", "")
            if not target_pattern:
                continue
            target = apply_side(target_pattern, side_val)
            if ent == "link":
                target = compact_link_name(target)

            out.append(
                Candidate(
                    target=target,
                    entity=EntityType.LINK.value if "link" in str(ent).lower() else EntityType.JOINT.value,
                    confidence=entry.get("confidence", "medium"),
                    source=entry.get("source", "contextual_aliases"),
                    mapping_type="contextual_pattern",
                    notes=entry.get("notes", ""),
                )
            )
        return out

    def _legacy_waist_topology_keys(
        self,
        *,
        source_normalized: str,
        parent: str | None,
        child: str | None,
    ) -> list[str]:
        if not (parent and child):
            return []
        parent_norm = normalize_name(parent)
        child_norm = normalize_name(child)
        keys = [
            f"{parent_norm}_to_{child_norm}",
            f"{parent_norm}_to_{source_normalized}",
            f"{source_normalized}_to_{child_norm}",
        ]
        if child_norm in _LEGACY_WAIST_CANONICAL_CHILDREN:
            keys.append(f"{parent_norm}_to_c_waist")
            keys.append(f"{parent_norm}_to_waist")
        if parent_norm in _LEGACY_WAIST_CANONICAL_CHILDREN:
            keys.append(f"c_waist_to_{child_norm}")
            keys.append(f"waist_to_{child_norm}")
        if source_normalized in _LEGACY_WAIST_SOURCE_NAMES:
            for legacy in _LEGACY_WAIST_SOURCE_NAMES:
                keys.append(f"{parent_norm}_to_{legacy}")
                keys.append(f"{legacy}_to_{child_norm}")
        deduped: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    def _legacy_waist_joint_candidates(
        self,
        source_name: str,
        normalized: str,
        parent: str | None,
        child: str | None,
        entity_type: str,
        axis: Iterable[float] | None,
    ) -> list[Candidate]:
        if entity_type != EntityType.JOINT.value:
            return []
        if normalized not in _LEGACY_WAIST_SOURCE_NAMES and "waist" not in normalized:
            return []

        axis_short = axis_vector_to_short(axis, self._axis_tokens)
        out: list[Candidate] = []
        seen: set[str] = set()
        for key in self._legacy_waist_topology_keys(
            source_normalized=normalized,
            parent=parent,
            child=child,
        ):
            for cand in self._lookup_alias_index(key, EntityType.JOINT.value):
                if cand.target in seen:
                    continue
                seen.add(cand.target)
                out.append(cand)
        if out:
            if axis_short:
                matched = [c for c in out if target_matches_axis(c.target, axis_short)]
                if matched:
                    return matched
            return out

        parent_norm = normalize_name(parent or "")
        child_norm = normalize_name(child or "")
        legacy_child = (
            normalized in _LEGACY_WAIST_SOURCE_NAMES
            or "waist" in child_norm
            or child_norm in _LEGACY_WAIST_CANONICAL_CHILDREN
        )
        if is_preserved_link(parent_norm) and legacy_child:
            ax = axis_short or "zy"
            return [
                Candidate(
                    target=f"c_pelvis_root_{ax}",
                    entity=EntityType.JOINT.value,
                    confidence="high",
                    source="policy:legacy_waist_root",
                    mapping_type="direct",
                    notes=f"parent={parent}, child={child}, source={source_name}",
                )
            ]

        legacy_parent = (
            parent_norm in _LEGACY_WAIST_CANONICAL_CHILDREN
            or parent_norm in _LEGACY_WAIST_SOURCE_NAMES
            or normalized in _LEGACY_WAIST_SOURCE_NAMES
        )
        if legacy_parent and child_norm and not is_preserved_link(child_norm):
            ax = axis_short or "yp"
            return [
                Candidate(
                    target=f"c_spine_01_{ax}",
                    entity=EntityType.JOINT.value,
                    confidence="high",
                    source="policy:legacy_waist_spine",
                    mapping_type="direct",
                    notes=f"parent={parent}, child={child}, source={source_name}",
                )
            ]
        return []

    def _root_waist_candidates(
        self,
        source_name: str,
        parent: str | None,
        child: str | None,
        entity_type: str,
        axis: Iterable[float] | None = None,
    ) -> list[Candidate]:
        if entity_type != EntityType.JOINT.value:
            return []

        parent_name = parent
        child_name = child
        if not (parent_name and child_name):
            pc = split_parent_child(source_name)
            if not pc:
                return []
            parent_name, child_name = pc

        if not is_preserved_link(parent_name):
            return []
        child_norm = normalize_name(child_name)
        source_norm = normalize_name(source_name)
        legacy_waist_child = (
            "waist" in child_norm
            or child_norm in _LEGACY_WAIST_CANONICAL_CHILDREN
            or source_norm in _LEGACY_WAIST_SOURCE_NAMES
        )
        if not legacy_waist_child:
            return []

        axis_short = axis_vector_to_short(axis, self._axis_tokens) or "zy"
        return [
            Candidate(
                target=f"c_pelvis_root_{axis_short}",
                entity=EntityType.JOINT.value,
                confidence="high",
                source="policy:root_pelvis_attachment",
                mapping_type="direct",
                notes=f"parent={parent_name}, child={child_name}",
            )
        ]

    def _heuristic_candidates(
        self,
        source_name: str,
        normalized: str,
        entity_type: str,
        axis: Iterable[float] | None,
        side: str | None,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        side_val = side or detect_side(normalized) or default_side(self.single_arm_as_left)
        axis_short = axis_vector_to_short(axis, self._axis_tokens) or axis_keyword_to_short(normalized)

        if entity_type in (EntityType.JOINT.value, EntityType.AUTO.value):
            composed = self._compose_joint_from_tokens(normalized, side_val, axis_short)
            if composed:
                out.append(
                    Candidate(
                        target=composed,
                        entity=EntityType.JOINT.value,
                        confidence="medium",
                        source="heuristic_token_compose",
                        mapping_type="heuristic",
                    )
                )

            pc = split_parent_child(source_name)
            if pc:
                parent, child = pc
                parent_side = detect_side(parent) or side_val
                child_side = detect_side(child) or parent_side
                landmark = self._link_segment_token(child)
                if landmark and axis_short:
                    use_side = infer_actuated_side(parent, child) or child_side or parent_side
                    if landmark in {"waist", "neck"} and use_side not in ("l", "r"):
                        use_side = "c"
                    if landmark == "waist" and use_side == "c":
                        target = f"c_pelvis_root_{axis_short}"
                    else:
                        target = f"{use_side}_{landmark}_{axis_short}"
                    out.append(
                        Candidate(
                            target=target,
                            entity=EntityType.JOINT.value,
                            confidence="low",
                            source="heuristic_parent_to_child",
                            mapping_type="heuristic",
                            notes=f"parent={parent}, child={child}",
                        )
                    )

        if entity_type in (EntityType.LINK.value, EntityType.AUTO.value):
            link_target = self._compose_link_from_tokens(normalized, side_val)
            if link_target:
                out.append(
                    Candidate(
                        target=link_target,
                        entity=EntityType.LINK.value,
                        confidence="medium",
                        source="heuristic_link_compose",
                        mapping_type="heuristic",
                    )
                )

        return out

    def _topology_candidates(
        self,
        parent: str,
        child: str,
        entity_type: str,
        axis: Iterable[float] | None,
        side: str | None,
    ) -> list[Candidate]:
        if entity_type != EntityType.JOINT.value:
            return []
        side_val = side or infer_actuated_side(parent, child) or detect_side(child) or detect_side(parent) or default_side(self.single_arm_as_left)
        axis_short = axis_vector_to_short(axis, self._axis_tokens)
        if not axis_short:
            return []
        landmark = self._infer_landmark_from_topology(parent, child, side_val)
        if not landmark:
            return []
        if landmark == "waist" and side_val == "c":
            target = f"c_pelvis_root_{axis_short}"
        else:
            target = f"{side_val}_{landmark}_{axis_short}"
        return [
            Candidate(
                target=target,
                entity=EntityType.JOINT.value,
                confidence="medium",
                source="topology_landmark",
                mapping_type="heuristic",
                notes=f"parent={parent}, child={child}",
            )
        ]

    def _resolve_canonical_link(self, link_name: str) -> str | None:
        if not link_name:
            return None
        if is_preserved_link(link_name):
            return compact_link_name(link_name)
        result = self.convert(link_name, entity=EntityType.LINK)
        if result.resolved and result.target:
            return compact_link_name(result.target)
        compact = compact_link_name(link_name)
        tree = self._master.get("link_tree", {})
        if compact in tree:
            return compact
        return None

    def _link_tree_path_candidates(
        self,
        parent: str,
        child: str,
        entity_type: str,
        axis: Iterable[float] | None,
    ) -> list[Candidate]:
        if entity_type != EntityType.JOINT.value:
            return []
        parent_link = self._resolve_canonical_link(parent)
        child_link = self._resolve_canonical_link(child)
        if not parent_link or not child_link or parent_link == child_link:
            return []

        tree = self._master.get("link_tree", {})
        if parent_link not in tree or child_link not in tree:
            return []

        path = _find_link_tree_path(tree, parent_link, child_link)
        if not path or len(path) < 2:
            return []

        axis_short = axis_vector_to_short(axis, self._axis_tokens)
        out: list[Candidate] = []
        for idx in range(len(path) - 1):
            hop_parent = path[idx]
            hop_child = path[idx + 1]
            via_joint = _link_tree_hop_joint(tree, hop_parent, hop_child)
            if not via_joint:
                continue
            score_note = f"path={' -> '.join(path)}"
            confidence = "high" if idx == 0 and len(path) == 2 else "medium"
            out.append(
                Candidate(
                    target=via_joint,
                    entity=EntityType.JOINT.value,
                    confidence=confidence,
                    source="link_tree_path",
                    mapping_type="topology_alias",
                    notes=f"parent={parent}, child={child}; {score_note}; hop={hop_parent}->{hop_child}",
                )
            )

        if axis_short and out:
            matched = [c for c in out if target_matches_axis(c.target, axis_short)]
            if matched:
                return matched
        return out[:3]

    # ── disambiguation ────────────────────────────────────────────────────

    def _pick_best(
        self,
        candidates: list[Candidate],
        entity_type: str,
        *,
        axis: Iterable[float] | None,
        parent: str | None,
        child: str | None,
        side: str | None,
    ) -> ConversionResult:
        axis_short = axis_vector_to_short(axis, self._axis_tokens) or axis_keyword_to_short(
            candidates[0].target if candidates else ""
        )
        side_val = side or infer_actuated_side(parent, child) or default_side(self.single_arm_as_left)

        scored: list[Candidate] = []
        for cand in candidates:
            score = float(cand.confidence_rank())
            if cand.entity == entity_type or entity_type == EntityType.AUTO.value:
                score += 1.0
            if cand.mapping_type == "direct":
                score += 2.0
            elif cand.mapping_type == "alias":
                score += 1.5
            elif cand.mapping_type == "product_alias":
                score += 2.5
            elif cand.mapping_type == "topology_alias":
                score += 2.25
            elif cand.mapping_type == "functional_alias":
                score += 1.0
            if axis_short and target_matches_axis(cand.target, axis_short):
                score += 2.0
            elif axis_short and cand.entity == EntityType.JOINT.value:
                score -= 1.0
            if cand.target.startswith(f"{side_val}_"):
                score += 0.5
            if self.profile and cand.source and self.profile.lower() in cand.source.lower():
                score += 1.0
            if parent and parent.lower() in cand.notes.lower():
                score += 0.25
            if child and child.lower() in cand.notes.lower():
                score += 0.25
            cand.score = score
            scored.append(cand)

        scored.sort(key=lambda c: c.score, reverse=True)
        if not scored:
            return ConversionResult(
                source="",
                normalized="",
                entity=entity_type,
                status=ConversionStatus.UNRESOLVED,
                reasons=["No scorable candidates."],
            )

        top = scored[0]
        close = [c for c in scored if abs(c.score - top.score) < 0.25]
        high_conf = [c for c in close if c.confidence_rank() >= CONFIDENCE_RANK.get(self.min_confidence, 2)]

        if len(high_conf) == 1:
            winner = high_conf[0]
            return ConversionResult(
                source="",
                normalized="",
                entity=winner.entity,
                status=ConversionStatus.RESOLVED,
                target=winner.target,
                candidates=scored,
                metadata={"winner_score": winner.score, "method": winner.mapping_type},
            )

        if len(scored) == 1 and top.score >= 3.0:
            return ConversionResult(
                source="",
                normalized="",
                entity=top.entity,
                status=ConversionStatus.RESOLVED,
                target=top.target,
                candidates=scored,
                metadata={
                    "winner_score": top.score,
                    "method": top.mapping_type,
                    "policy": "single_candidate_heuristic",
                },
            )

        if len(close) == 1 and top.confidence_rank() >= CONFIDENCE_RANK.get(self.min_confidence, 2):
            return ConversionResult(
                source="",
                normalized="",
                entity=top.entity,
                status=ConversionStatus.RESOLVED,
                target=top.target,
                candidates=scored,
                metadata={"winner_score": top.score, "method": top.mapping_type},
            )

        return ConversionResult(
            source="",
            normalized="",
            entity=entity_type,
            status=ConversionStatus.AMBIGUOUS,
            candidates=scored,
            reasons=[f"Top candidates tied within score window: {[c.target for c in close[:5]]}"],
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _resolve_entity_type(self, source_name: str, normalized: str, entity: str | EntityType) -> str:
        if entity != EntityType.AUTO and entity != EntityType.AUTO.value:
            return normalize_entity_type(entity)
        if normalized in _LEGACY_LINK_ALIASES:
            return EntityType.LINK.value
        alias_entries = self._alias_index.get(normalized, [])
        if alias_entries:
            entities = {str(entry.get("entity", "")).lower() for entry in alias_entries}
            if entities == {EntityType.LINK.value}:
                return EntityType.LINK.value
            if entities == {EntityType.JOINT.value}:
                return EntityType.JOINT.value
        if normalized.endswith("_link") or normalized.endswith("link"):
            return EntityType.LINK.value
        if normalized.endswith("_joint") or "joint" in normalized:
            return EntityType.JOINT.value
        if split_parent_child(source_name):
            return EntityType.JOINT.value
        if any(tok in normalized for tok in ("roll", "pitch", "yaw", "_xr", "_yp", "_zy")):
            return EntityType.JOINT.value
        return EntityType.JOINT.value

    def _is_logical_joint_without_dof(self, normalized: str, entity_type: str) -> bool:
        if entity_type != EntityType.JOINT.value:
            return False
        if _LOGICAL_JOINT_RE.search(normalized):
            return True
        if normalized in {
            "shoulder_joint",
            "elbow_joint",
            "wrist_joint",
            "hip_joint",
            "knee_joint",
            "ankle_joint",
            "neck_joint",
            "chest_joint",
            "base_joint",
        }:
            return True
        return False

    def _compose_joint_from_tokens(self, normalized: str, side: str, axis_short: str | None) -> str | None:
        body = normalized
        for prefix in ("left_", "right_", "l_", "r_", "c_"):
            if body.startswith(prefix):
                body = body[len(prefix):]
                break

        m = re.match(r"^joint(\d+)$", body)
        if m:
            idx = int(m.group(1))
            chain = [
                ("shoulder", "zy"),
                ("shoulder", "yp"),
                ("elbow", "yp"),
                ("wrist", "yp"),
                ("wrist", "zy"),
                ("wrist", "xr"),
                ("wrist", "yp"),
            ]
            if 1 <= idx <= len(chain):
                landmark, ax = chain[idx - 1]
                return f"{side}_{landmark}_{ax}"

        if not axis_short:
            return None

        token_map = {
            "shoulder_pitch": "shoulder",
            "shoulder_roll": "shoulder",
            "shoulder_yaw": "shoulder",
            "shoulder_pan": "shoulder",
            "shoulder_lift": "shoulder",
            "arm_upper_roll": "shoulder",
            "arm_upper_yaw": "shoulder",
            "elbow_pitch": "elbow",
            "elbow_yaw": "elbow",
            "elbow": "elbow",
            "arm_lower_pitch": "elbow",
            "wrist": "wrist",
            "wrist_1": "wrist",
            "wrist_2": "wrist",
            "wrist_3": "wrist",
            "hip_yaw": "hip",
            "hip_roll": "hip",
            "hip_pitch": "hip",
            "hipjoint_upper_yaw": "hip",
            "hipjoint_lower_roll": "hip",
            "leg_upper_pitch": "hip",
            "thigh_pitch": "hip",
            "knee_pitch": "knee",
            "leg_lower_pitch": "knee",
            "ankle_pitch": "ankle",
            "ankle_roll": "ankle",
            "foot_small_roll": "ankle",
            "foot_roll": "ankle",
            "head_yaw": "neck",
            "head_pitch": "neck",
            "chest_yaw": "waist",
            "waist_yaw": "waist",
            "neck_yaw": "neck",
            "neck_pitch": "neck",
        }

        for key, landmark in token_map.items():
            if key in body:
                return f"{side}_{landmark}_{axis_short}"

        return None

    def _compose_link_from_tokens(self, normalized: str, side: str) -> str | None:
        if is_preserved_link(normalized):
            return "base_link"

        base = normalized

        link_map = {
            "upper_arm": "arm_upper",
            "lower_arm": "arm_lower",
            "forearm": "arm_lower",
            "upperarm": "arm_upper",
            "lowerarm": "arm_lower",
            "arm_upper": "arm_upper",
            "arm_lower": "arm_lower",
            "upper_leg": "leg_upper",
            "lower_leg": "leg_lower",
            "thigh": "leg_upper",
            "shank": "leg_lower",
            "leg_upper": "leg_upper",
            "leg_lower": "leg_lower",
            "foot": "foot",
            "hand": "hand",
            "head": "head",
            "neck": "neck",
            "pelvis": "pelvis",
            "torso": "torso",
            "chest": "torso",
            "waist": "pelvis",
            "base": "base",
            "shoulder": "shoulder",
            "hip": "pelvis",
            "hipjoint_upper": "hipjoint_upper",
            "hipjoint_lower": "hipjoint_lower",
            "arm_upper_yaw": "arm_upper_yaw",
            "arm_upper_roll": "arm_upper_roll",
            "elbow": "elbow",
            "wrist": "wrist",
            "knee": "knee",
            "ankle": "ankle",
            "foot_small": "foot",
        }

        body = base
        for prefix in ("left_", "right_", "l_", "r_", "c_"):
            if body.startswith(prefix):
                body = body[len(prefix):]
                break
        body = body.removesuffix("_link")

        for key, segment in sorted(link_map.items(), key=lambda kv: -len(kv[0])):
            if body == key or body.endswith(f"_{key}") or key in body:
                if segment == "base":
                    return "base_link"
                if side == "c" and segment in {"head", "neck", "pelvis", "torso", "base", "mandible"}:
                    return f"c_{segment}"
                return f"{side}_{segment}"
        return None

    def _finalize_link_result(
        self,
        result: ConversionResult,
        *,
        entity_type: str | None = None,
    ) -> ConversionResult:
        link_entity = entity_type or result.entity
        if link_entity != EntityType.LINK.value:
            return result
        if result.target:
            result.target = compact_link_name(result.target)
        for cand in result.candidates:
            if cand.entity == EntityType.LINK.value and cand.target:
                cand.target = compact_link_name(cand.target)
        return result

    def _link_segment_token(self, link_name: str) -> str | None:
        mapping = {
            "hipjoint_upper": "hip",
            "hipjoint_lower": "hip",
            "arm_upper_yaw": "shoulder",
            "arm_upper_roll": "shoulder",
            "arm_upper": "shoulder",
            "shoulder": "shoulder",
            "elbow": "elbow",
            "arm_lower": "elbow",
            "wrist": "wrist",
            "hand": "wrist",
            "leg_upper": "hip",
            "knee": "knee",
            "leg_lower": "knee",
            "ankle": "ankle",
            "foot": "ankle",
            "head": "neck",
            "chest": "waist",
            "waist": "pelvis",
        }
        norm = normalize_name(link_name)
        for key, landmark in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            if key in norm:
                return landmark
        return None

    def _infer_landmark_from_topology(self, parent: str, child: str, side: str) -> str | None:
        child_landmark = self._link_segment_token(child)
        if child_landmark:
            return child_landmark
        parent_landmark = self._link_segment_token(parent)
        return parent_landmark


# ===========================================================================
# importer_bridge
# ===========================================================================

_ACTUATED_JOINT_TYPES = frozenset(
    {"revolute", "continuous", "hinge", "slide", "prismatic", "ball", "spherical"}
)


@dataclass
class ParsedModelConversion:
    """Conversion output for a parsed URDF/MJCF model."""

    model_type: str
    source_path: str = ""
    joints_in: list[dict[str, Any]] = field(default_factory=list)
    links_in: list[dict[str, Any]] = field(default_factory=list)
    conversion: ModelConversionResult = field(default_factory=ModelConversionResult)
    joint_map: dict[str, str] = field(default_factory=dict)
    link_map: dict[str, str] = field(default_factory=dict)
    reverse_joint_map: dict[str, list[str]] = field(default_factory=dict)
    reverse_link_map: dict[str, list[str]] = field(default_factory=dict)

    @property
    def unresolved_joints(self) -> list[tuple[str, ConversionResult]]:
        return [
            (name, result)
            for name, result in self.conversion.joints.items()
            if not result.resolved
        ]

    @property
    def unresolved_links(self) -> list[tuple[str, ConversionResult]]:
        return [
            (name, result)
            for name, result in self.conversion.links.items()
            if not result.resolved
        ]

    def canonical_joint(self, source_name: str) -> str | None:
        return self.joint_map.get(source_name)

    def canonical_link(self, source_name: str) -> str | None:
        return self.link_map.get(source_name)

    def remap_joint_angles(self, angles: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for source, value in angles.items():
            target = self.joint_map.get(source)
            if target:
                out[target] = float(value)
        return out

    def remap_joint_angles_to_source(
        self,
        canonical_angles: dict[str, float],
        *,
        prefer_first: bool = True,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for canonical, value in canonical_angles.items():
            sources = self.reverse_joint_map.get(canonical, [])
            if not sources:
                continue
            chosen = sources[0] if prefer_first else sources[-1]
            out[chosen] = float(value)
        return out


def extract_model_entities(
    parsed_data: dict[str, Any],
    model_type: str,
    *,
    include_fixed_joints: bool = False,
    include_fixed_links: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    joints_out: list[dict[str, Any]] = []
    links_out: list[dict[str, Any]] = []

    if model_type == "mjcf":
        for joint in parsed_data.get("joints", []) or []:
            if not isinstance(joint, dict):
                continue
            jtype = str(joint.get("type", "fixed")).lower()
            if not include_fixed_joints and jtype not in _ACTUATED_JOINT_TYPES:
                continue
            name = str(joint.get("name", "")).strip()
            if not name:
                continue
            joints_out.append(
                {
                    "name": name,
                    "type": jtype,
                    "parent": joint.get("parent"),
                    "child": joint.get("child"),
                    "axis": joint.get("axis"),
                    "origin_rpy": joint.get("origin_rpy"),
                    "origin_quat": joint.get("origin_quat"),
                }
            )

        for body in parsed_data.get("bodies", []) or []:
            if not isinstance(body, dict):
                continue
            name = str(body.get("name", "")).strip()
            if not name:
                continue
            links_out.append({"name": name, "parent": body.get("parent")})
        return joints_out, links_out

    # URDF family
    links_data = parsed_data.get("links", {}) or {}
    if isinstance(links_data, dict):
        for link_name, link_info in links_data.items():
            if not link_name:
                continue
            parent = None
            if isinstance(link_info, dict):
                parent = link_info.get("parent")
            links_out.append({"name": str(link_name), "parent": parent})

    for joint in parsed_data.get("joints", []) or []:
        if not isinstance(joint, dict):
            continue
        jtype = str(joint.get("type", "fixed")).lower()
        if not include_fixed_joints and jtype not in _ACTUATED_JOINT_TYPES:
            continue
        name = str(joint.get("name", "")).strip()
        if not name:
            continue
        joints_out.append(
            {
                "name": name,
                "type": jtype,
                "parent": joint.get("parent"),
                "child": joint.get("child"),
                "axis": joint.get("axis"),
                "origin_rpy": joint.get("origin_rpy"),
                "origin_quat": joint.get("origin_quat"),
            }
        )

    if not include_fixed_links:
        actuated_children = {j.get("child") for j in joints_out if j.get("child")}
        links_out = [lk for lk in links_out if lk["name"] in actuated_children]

    return joints_out, links_out


def convert_parsed_model(
    parsed_data: dict[str, Any],
    model_type: str,
    *,
    source_path: str = "",
    converter: NameConverter | None = None,
    **converter_kwargs: Any,
) -> ParsedModelConversion:
    joints_in, links_in = extract_model_entities(parsed_data, model_type)
    conv = converter or NameConverter(**converter_kwargs)
    conversion = conv.convert_model(joints=joints_in, links=links_in)

    joint_map = dict(conversion.resolved_joint_map)
    link_map = dict(conversion.resolved_link_map)
    reverse_joint_map: dict[str, list[str]] = {}
    reverse_link_map: dict[str, list[str]] = {}
    for src, dst in joint_map.items():
        reverse_joint_map.setdefault(dst, []).append(src)
    for src, dst in link_map.items():
        reverse_link_map.setdefault(dst, []).append(src)

    return ParsedModelConversion(
        model_type=model_type,
        source_path=source_path,
        joints_in=joints_in,
        links_in=links_in,
        conversion=conversion,
        joint_map=joint_map,
        link_map=link_map,
        reverse_joint_map=reverse_joint_map,
        reverse_link_map=reverse_link_map,
    )


def convert_model_file(
    file_path: str,
    *,
    parent_widget: Any = None,
    converter: NameConverter | None = None,
    **converter_kwargs: Any,
) -> tuple[str, str, dict[str, Any], str, ParsedModelConversion] | None:
    from LegacyMotionEditor_Importer import parse_model_file  # lazy: LME app dep

    parsed = parse_model_file(file_path, parent_widget=parent_widget)
    if not parsed:
        return None

    model_path, working_dir, parsed_data, model_type = parsed
    conversion = convert_parsed_model(
        parsed_data,
        model_type,
        source_path=model_path,
        converter=converter,
        **converter_kwargs,
    )
    return model_path, working_dir, parsed_data, model_type, conversion


# ===========================================================================
# model_renamer
# ===========================================================================

_MJCF_ACTUATED_JOINT_TYPES = frozenset(
    {"revolute", "continuous", "hinge", "slide", "prismatic"}
)

_MJCF_JOINT_REF_TAGS = frozenset(
    {"position", "motor", "velocity", "general", "cylinder", "muscle", "adhesion"}
)


class RenameMapValidationError(ValueError):
    """Raised when a rename map would corrupt a model file."""


@dataclass
class RenameMapValidation:
    joint_map: dict[str, str] = field(default_factory=dict)
    link_map: dict[str, str] = field(default_factory=dict)
    blocked_joints: list[tuple[str, str, str]] = field(default_factory=list)
    blocked_links: list[tuple[str, str, str]] = field(default_factory=list)
    missing_joint_sources: list[str] = field(default_factory=list)
    missing_link_sources: list[str] = field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        return bool(
            self.blocked_joints
            or self.blocked_links
            or self.missing_joint_sources
            or self.missing_link_sources
        )


def _is_identity_rename(source: str, target: str) -> bool:
    return normalize_name(source) == normalize_name(target)


def filter_identity_rename_map(rename_map: dict[str, str]) -> dict[str, str]:
    return {
        src: dst
        for src, dst in rename_map.items()
        if not _is_identity_rename(src, dst)
    }


def split_safe_rename_map(
    rename_map: dict[str, str],
    existing_names: set[str],
    *,
    entity_label: str,
) -> tuple[dict[str, str], list[tuple[str, str, str]], list[str]]:
    filtered = filter_identity_rename_map(rename_map)
    blocked: list[tuple[str, str, str]] = []
    missing: list[str] = []
    if not filtered:
        return {}, blocked, missing

    sources = set(filtered.keys())
    by_target: dict[str, list[str]] = defaultdict(list)
    for src, dst in filtered.items():
        by_target[dst].append(src)

    unsafe_sources: set[str] = set()
    for dst, srcs in by_target.items():
        if len(srcs) > 1:
            unsafe_sources.update(srcs)
            blocked.append(
                (
                    ", ".join(sorted(srcs)),
                    dst,
                    f"duplicate {entity_label} target would create multiple '{dst}' names",
                )
            )

    for src, dst in filtered.items():
        if src not in existing_names:
            missing.append(src)
            unsafe_sources.add(src)
            continue
        if dst in existing_names and dst not in sources:
            unsafe_sources.add(src)
            blocked.append(
                (
                    src,
                    dst,
                    f"target {entity_label} '{dst}' already exists in the model",
                )
            )

    safe_map = {
        src: dst
        for src, dst in filtered.items()
        if src not in unsafe_sources
    }
    return safe_map, blocked, missing


def _collect_parsed_joint_names(parsed_data: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for joint in parsed_data.get("joints", []) or []:
        name = str(joint.get("name", "")).strip()
        if name:
            names.add(name)
    return names


def _collect_parsed_link_names(parsed_data: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for link_name in (parsed_data.get("links", {}) or {}).keys():
        if link_name:
            names.add(str(link_name))
    for body in parsed_data.get("bodies", []) or []:
        name = str(body.get("name", "")).strip()
        if name:
            names.add(name)
    return names


def collect_mjcf_element_names(root: ET.Element) -> tuple[set[str], set[str]]:
    joint_names = {
        name
        for name in (elem.get("name") for elem in root.iter("joint"))
        if name
    }
    body_names = {
        name
        for name in (elem.get("name") for elem in root.iter("body"))
        if name
    }
    return joint_names, body_names


def collect_urdf_element_names(root: ET.Element) -> tuple[set[str], set[str]]:
    joint_names = {
        name
        for name in (elem.get("name") for elem in root.findall("joint"))
        if name
    }
    link_names = {
        name
        for name in (elem.get("name") for elem in root.findall("link"))
        if name
    }
    return joint_names, link_names


def validate_rename_maps_for_file(
    file_path: str,
    model_type: str,
    joint_map: dict[str, str] | None = None,
    link_map: dict[str, str] | None = None,
) -> RenameMapValidation:
    tree = ET.parse(file_path)
    root = tree.getroot()
    if model_type == "mjcf":
        joint_names, link_names = collect_mjcf_element_names(root)
    else:
        joint_names, link_names = collect_urdf_element_names(root)

    safe_joint_map, blocked_joints, missing_joint_sources = split_safe_rename_map(
        joint_map or {},
        joint_names,
        entity_label="joint",
    )
    safe_link_map, blocked_links, missing_link_sources = split_safe_rename_map(
        link_map or {},
        link_names,
        entity_label="body",
    )
    return RenameMapValidation(
        joint_map=safe_joint_map,
        link_map=safe_link_map,
        blocked_joints=blocked_joints,
        blocked_links=blocked_links,
        missing_joint_sources=sorted(missing_joint_sources),
        missing_link_sources=sorted(missing_link_sources),
    )


def _format_rename_validation_error(validation: RenameMapValidation) -> str:
    lines = ["モデルファイルの上書きを中止しました。リネーム計画が安全ではありません。"]
    for src, dst, reason in validation.blocked_joints:
        lines.append(f"  joint blocked: {src} -> {dst}: {reason}")
    for src, dst, reason in validation.blocked_links:
        lines.append(f"  body blocked: {src} -> {dst}: {reason}")
    for src in validation.missing_joint_sources:
        lines.append(f"  joint source not found in file: {src}")
    for src in validation.missing_link_sources:
        lines.append(f"  body source not found in file: {src}")
    lines.append(
        "リンク名の衝突がある場合は「リンク名も変換する」を外すか、"
        "衝突するリンク変換をプレビューから除外してから再実行してください。"
    )
    return "\n".join(lines)


def _apply_safe_rename_maps_from_parsed(
    rename_map: dict[str, str],
    existing_names: set[str],
    *,
    entity_label: str,
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    safe_map, blocked, _missing = split_safe_rename_map(
        rename_map,
        existing_names,
        entity_label=entity_label,
    )
    return safe_map, blocked


def _format_ambiguous_reason(result: Any) -> str:
    reasons = "; ".join(result.reasons) if result.reasons else "ambiguous"
    candidates = [c.target for c in result.candidates if c.target][:5]
    if candidates:
        return f"{reasons} (candidates: {', '.join(candidates)})"
    return reasons


def build_joint_rename_map(
    parsed_data: dict[str, Any],
    model_type: str,
    *,
    robot_model: Any | None = None,
    single_arm_as_left: bool = True,
) -> tuple[
    dict[str, str],
    list[tuple[str, str]],
    list[str],
    dict[str, str],
    list[tuple[str, str]],
]:
    conversion = convert_parsed_model(
        parsed_data,
        model_type,
        single_arm_as_left=single_arm_as_left,
    )
    joint_map: dict[str, str] = {}
    unresolved: list[tuple[str, str]] = []
    skipped_fixed: list[str] = []
    unchanged_map: dict[str, str] = {}
    ambiguous: list[tuple[str, str]] = []

    model_joint_names = set()
    if robot_model is not None:
        model_joint_names = set(getattr(robot_model, "joint_order", []) or [])

    for joint in conversion.joints_in:
        source = joint["name"]
        if model_joint_names and source not in model_joint_names:
            continue
        jtype = str(joint.get("type", "fixed")).lower()
        if jtype not in _MJCF_ACTUATED_JOINT_TYPES:
            skipped_fixed.append(source)
            continue
        result = conversion.conversion.joints.get(source)
        if result and result.resolved and result.target:
            if _is_identity_rename(source, result.target):
                unchanged_map[source] = result.target
            else:
                joint_map[source] = result.target
        elif result and result.status == ConversionStatus.AMBIGUOUS:
            ambiguous.append((source, _format_ambiguous_reason(result)))
        else:
            reason = "; ".join(result.reasons) if result and result.reasons else "unresolved"
            unresolved.append((source, reason))

    existing_joint_names = _collect_parsed_joint_names(parsed_data)
    if model_joint_names:
        existing_joint_names &= model_joint_names
    joint_map, blocked = _apply_safe_rename_maps_from_parsed(
        joint_map,
        existing_joint_names,
        entity_label="joint",
    )
    for src, dst, reason in blocked:
        ambiguous.append((src.split(", ")[0], f"{reason} (target: {dst})"))

    return joint_map, unresolved, skipped_fixed, unchanged_map, ambiguous


def build_link_rename_map(
    parsed_data: dict[str, Any],
    model_type: str,
    *,
    robot_model: Any | None = None,
    single_arm_as_left: bool = True,
) -> tuple[
    dict[str, str],
    list[tuple[str, str]],
    dict[str, str],
    dict[str, str],
    list[tuple[str, str]],
]:
    conversion = convert_parsed_model(
        parsed_data,
        model_type,
        single_arm_as_left=single_arm_as_left,
    )
    link_map: dict[str, str] = {}
    unresolved: list[tuple[str, str]] = []
    preserved_link_map: dict[str, str] = {}
    unchanged_map: dict[str, str] = {}
    ambiguous: list[tuple[str, str]] = []

    model_link_names = set()
    if robot_model is not None:
        model_link_names = set(getattr(robot_model, "links", {}) or {})

    for link in conversion.links_in:
        source = str(link.get("name", "")).strip()
        if not source:
            continue
        if is_preserved_link(source):
            preserved = preserved_link_target(source)
            if preserved and (not model_link_names or source in model_link_names):
                preserved_link_map[source] = preserved
            continue
        if model_link_names and source not in model_link_names:
            continue
        result = conversion.conversion.links.get(source)
        if result and result.resolved and result.target:
            if _is_identity_rename(source, result.target):
                unchanged_map[source] = result.target
            else:
                link_map[source] = result.target
        elif result and result.status == ConversionStatus.AMBIGUOUS:
            ambiguous.append((source, _format_ambiguous_reason(result)))
        else:
            reason = "; ".join(result.reasons) if result and result.reasons else "unresolved"
            unresolved.append((source, reason))

    existing_link_names = _collect_parsed_link_names(parsed_data)
    if model_link_names:
        existing_link_names &= model_link_names
    link_map, blocked = _apply_safe_rename_maps_from_parsed(
        link_map,
        existing_link_names,
        entity_label="body",
    )
    for src, dst, reason in blocked:
        ambiguous.append((src.split(", ")[0], f"{reason} (target: {dst})"))

    return link_map, unresolved, preserved_link_map, unchanged_map, ambiguous


def sanitize_link_rename_map(link_map: dict[str, str]) -> dict[str, str]:
    return {
        src: dst
        for src, dst in link_map.items()
        if src not in PRESERVED_LINK_NAMES and not is_preserved_link(src)
    }


def _remap_dict_keys(data: dict[str, Any], key_map: dict[str, str]) -> dict[str, Any]:
    return {key_map.get(key, key): value for key, value in data.items()}


def _remap_group_preset_members(
    presets: list[dict[str, Any]] | None,
    joint_map: dict[str, str],
) -> list[dict[str, Any]]:
    if not presets or not joint_map:
        return presets or []

    remapped: list[dict[str, Any]] = []
    for preset in presets:
        if not isinstance(preset, dict):
            continue
        new_preset = dict(preset)
        new_members: dict[str, dict[str, Any]] = {}
        for old_name, member in (preset.get("members") or {}).items():
            if not isinstance(member, dict):
                continue
            new_name = joint_map.get(old_name, old_name)
            current = new_members.get(new_name, {"enabled": False, "scale": 1.0})
            new_members[new_name] = {
                "enabled": bool(current.get("enabled")) or bool(member.get("enabled", False)),
                "scale": float(member.get("scale", current.get("scale", 1.0))),
            }
        new_preset["members"] = new_members
        remapped.append(new_preset)
    return remapped


def apply_link_rename_to_robot_model(robot_model: Any, link_map: dict[str, str]) -> None:
    if not robot_model or not link_map:
        return

    link_map = sanitize_link_rename_map(link_map)
    link_map = filter_identity_rename_map(link_map)
    if not link_map:
        return

    new_links: dict[str, Any] = {}
    for old_name, link in robot_model.links.items():
        new_name = link_map.get(old_name, old_name)
        link.name = new_name
        new_links[new_name] = link
    robot_model.links = new_links

    for attr in ("link_actors", "link_transforms", "link_visual_transforms"):
        bucket = getattr(robot_model, attr, None)
        if isinstance(bucket, dict) and bucket:
            setattr(robot_model, attr, _remap_dict_keys(bucket, link_map))

    for joint in robot_model.joints.values():
        joint.parent_link = link_map.get(joint.parent_link, joint.parent_link)
        joint.child_link = link_map.get(joint.child_link, joint.child_link)

    robot_model.parent_map = {
        link_map.get(child_link, child_link): joint_name
        for child_link, joint_name in robot_model.parent_map.items()
    }
    robot_model.child_map = {
        link_map.get(parent_link, parent_link): joint_names
        for parent_link, joint_names in robot_model.child_map.items()
    }

    if getattr(robot_model, "root_link", None):
        robot_model.root_link = link_map.get(robot_model.root_link, robot_model.root_link)
        if is_preserved_link(robot_model.root_link):
            robot_model.root_link = "base_link"


def remap_string_dict(data: dict[str, Any], key_map: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        out[key_map.get(key, key)] = value
    return out


def apply_joint_rename_to_robot_model(robot_model: Any, joint_map: dict[str, str]) -> None:
    if not robot_model or not joint_map:
        return

    joint_map = filter_identity_rename_map(joint_map)
    if not joint_map:
        return

    new_joints: dict[str, Any] = {}
    for old_name, joint in robot_model.joints.items():
        new_name = joint_map.get(old_name, old_name)
        joint.name = new_name
        new_joints[new_name] = joint
    robot_model.joints = new_joints
    robot_model.joint_order = [joint_map.get(n, n) for n in robot_model.joint_order]

    robot_model.parent_map = {
        child: joint_map.get(jn, jn)
        for child, jn in robot_model.parent_map.items()
    }
    robot_model.child_map = {
        parent: [joint_map.get(jn, jn) for jn in jnames]
        for parent, jnames in robot_model.child_map.items()
    }
    robot_model.current_angles = remap_string_dict(
        getattr(robot_model, "current_angles", {}) or {},
        joint_map,
    )


def apply_rename_to_robot_model(
    robot_model: Any,
    *,
    joint_map: dict[str, str] | None = None,
    link_map: dict[str, str] | None = None,
) -> None:
    if joint_map:
        apply_joint_rename_to_robot_model(robot_model, joint_map)
    if link_map:
        apply_link_rename_to_robot_model(robot_model, link_map)


def apply_joint_rename_to_joint_editor(
    joint_editor: Any,
    joint_map: dict[str, str],
    *,
    link_map: dict[str, str] | None = None,
) -> None:
    if not joint_editor:
        return
    if not joint_map and not link_map:
        return

    if joint_map:
        joint_editor.joint_settings = remap_string_dict(
            getattr(joint_editor, "joint_settings", {}) or {},
            joint_map,
        )
        for new_name, setting in list(joint_editor.joint_settings.items()):
            if isinstance(setting, dict):
                setting["display_name"] = new_name
        joint_editor.joint_display_order = [
            joint_map.get(n, n) for n in getattr(joint_editor, "joint_display_order", [])
        ]
        joint_editor.joint_display_groups = remap_string_dict(
            getattr(joint_editor, "joint_display_groups", {}) or {},
            joint_map,
        )
        if hasattr(joint_editor, "joint_group_presets"):
            joint_editor.joint_group_presets = _remap_group_preset_members(
                getattr(joint_editor, "joint_group_presets", []) or [],
                joint_map,
            )
        if hasattr(joint_editor, "home_position_angles"):
            joint_editor.home_position_angles = remap_string_dict(
                getattr(joint_editor, "home_position_angles", {}) or {},
                joint_map,
            )

    current_angles = joint_editor.get_angles() if hasattr(joint_editor, "get_angles") else {}
    if joint_map:
        current_angles = remap_string_dict(current_angles, joint_map)
    current_easings = (
        joint_editor.get_joint_easings()
        if hasattr(joint_editor, "get_joint_easings")
        else {}
    )
    if joint_map:
        current_easings = remap_string_dict(current_easings, joint_map)

    robot_model = getattr(joint_editor, "robot_model", None)
    if robot_model:
        apply_rename_to_robot_model(
            robot_model,
            joint_map=joint_map or None,
            link_map=link_map or None,
        )
        if hasattr(joint_editor, "build_from_robot"):
            joint_editor.build_from_robot(robot_model)
        if hasattr(joint_editor, "set_joint_settings"):
            joint_editor.set_joint_settings(joint_editor.joint_settings)
        if hasattr(joint_editor, "set_angles"):
            joint_editor.set_angles(current_angles)
        if hasattr(joint_editor, "set_joint_easings"):
            joint_editor.set_joint_easings(current_easings)


def apply_joint_rename_to_graph(graph: Any, joint_map: dict[str, str]) -> int:
    if not graph or not joint_map:
        return 0

    updated = 0
    nodes = graph.all_nodes() if hasattr(graph, "all_nodes") else []
    for node in nodes:
        changed = False
        if hasattr(node, "angles_deg") and isinstance(node.angles_deg, dict):
            node.angles_deg = remap_string_dict(node.angles_deg, joint_map)
            changed = True
        if hasattr(node, "joint_easings") and isinstance(node.joint_easings, dict):
            node.joint_easings = remap_string_dict(node.joint_easings, joint_map)
            changed = True
        if changed:
            updated += 1
    return updated


def _backup_file(path: str) -> str:
    backup_path = f"{path}.bak"
    idx = 1
    while os.path.exists(backup_path):
        backup_path = f"{path}.bak{idx}"
        idx += 1
    shutil.copy2(path, backup_path)
    return backup_path


def _replace_xml_attr(elem: ET.Element, attr: str, mapping: dict[str, str]) -> None:
    value = elem.get(attr)
    if value and value in mapping:
        elem.set(attr, mapping[value])


def _apply_mjcf_renames_to_tree(
    root: ET.Element,
    *,
    joint_map: dict[str, str] | None = None,
    link_map: dict[str, str] | None = None,
) -> None:
    joint_map = joint_map or {}
    link_map = link_map or {}

    for joint_elem in root.iter("joint"):
        _replace_xml_attr(joint_elem, "name", joint_map)

    for actuator_elem in root.iter("actuator"):
        for child in actuator_elem:
            if child.tag in _MJCF_JOINT_REF_TAGS:
                _replace_xml_attr(child, "joint", joint_map)

    for equality_elem in root.iter("equality"):
        for child in equality_elem:
            _replace_xml_attr(child, "joint1", joint_map)
            _replace_xml_attr(child, "joint2", joint_map)
            _replace_xml_attr(child, "body1", link_map)
            _replace_xml_attr(child, "body2", link_map)

    for contact_elem in root.iter("contact"):
        for exclude_elem in contact_elem.iter("exclude"):
            _replace_xml_attr(exclude_elem, "body1", link_map)
            _replace_xml_attr(exclude_elem, "body2", link_map)

    for tendon_elem in root.iter("tendon"):
        for child in tendon_elem:
            _replace_xml_attr(child, "joint", joint_map)
            _replace_xml_attr(child, "joint1", joint_map)
            _replace_xml_attr(child, "joint2", joint_map)

    for sensor_elem in root.iter("sensor"):
        for child in sensor_elem.iter():
            _replace_xml_attr(child, "joint", joint_map)

    for body_elem in root.iter("body"):
        _replace_xml_attr(body_elem, "name", link_map)


def _apply_urdf_renames_to_tree(
    root: ET.Element,
    *,
    joint_map: dict[str, str] | None = None,
    link_map: dict[str, str] | None = None,
) -> None:
    joint_map = joint_map or {}
    link_map = link_map or {}

    for joint_elem in root.findall("joint"):
        _replace_xml_attr(joint_elem, "name", joint_map)
        parent_elem = joint_elem.find("parent")
        if parent_elem is not None:
            _replace_xml_attr(parent_elem, "link", link_map)
        child_elem = joint_elem.find("child")
        if child_elem is not None:
            _replace_xml_attr(child_elem, "link", link_map)

    for link_elem in root.findall("link"):
        _replace_xml_attr(link_elem, "name", link_map)

    for gazebo_elem in root.iter("gazebo"):
        _replace_xml_attr(gazebo_elem, "reference", joint_map)
        _replace_xml_attr(gazebo_elem, "reference", link_map)

    for transmission_elem in root.findall("transmission"):
        for joint_elem in transmission_elem.findall("joint"):
            _replace_xml_attr(joint_elem, "name", joint_map)
            if joint_elem.text and joint_elem.text.strip() in joint_map:
                joint_elem.text = joint_map[joint_elem.text.strip()]

    for ros2_control in root.iter("ros2_control"):
        for joint_elem in ros2_control.findall("joint"):
            _replace_xml_attr(joint_elem, "name", joint_map)


def _write_xml_tree(tree: ET.ElementTree, file_path: str) -> None:
    tree.write(file_path, encoding="unicode", xml_declaration=True)


def rewrite_mjcf_model_names(
    file_path: str,
    *,
    joint_map: dict[str, str] | None = None,
    link_map: dict[str, str] | None = None,
) -> None:
    tree = ET.parse(file_path)
    _apply_mjcf_renames_to_tree(
        tree.getroot(),
        joint_map=joint_map or {},
        link_map=link_map or {},
    )
    _write_xml_tree(tree, file_path)


def rewrite_urdf_model_names(
    file_path: str,
    *,
    joint_map: dict[str, str] | None = None,
    link_map: dict[str, str] | None = None,
) -> None:
    tree = ET.parse(file_path)
    _apply_urdf_renames_to_tree(
        tree.getroot(),
        joint_map=joint_map or {},
        link_map=link_map or {},
    )
    _write_xml_tree(tree, file_path)


def rewrite_model_file_joint_names(
    file_path: str,
    model_type: str,
    joint_map: dict[str, str],
    *,
    link_map: dict[str, str] | None = None,
    create_backup: bool = True,
) -> str | None:
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    joint_map = filter_identity_rename_map(joint_map)
    if link_map:
        link_map = sanitize_link_rename_map(link_map)
        link_map = filter_identity_rename_map(link_map)
    else:
        link_map = {}

    if not joint_map and not link_map:
        raise ValueError("No rename map provided.")

    validation = validate_rename_maps_for_file(
        file_path,
        model_type,
        joint_map=joint_map,
        link_map=link_map,
    )
    if validation.has_blocking_issues:
        raise RenameMapValidationError(_format_rename_validation_error(validation))

    backup_path = None
    if create_backup:
        backup_path = _backup_file(file_path)

    if model_type == "mjcf":
        rewrite_mjcf_model_names(
            file_path,
            joint_map=validation.joint_map,
            link_map=validation.link_map,
        )
    else:
        rewrite_urdf_model_names(
            file_path,
            joint_map=validation.joint_map,
            link_map=validation.link_map,
        )

    return backup_path


def load_parsed_model(file_path: str, model_type: str) -> dict[str, Any]:
    if model_type == "mjcf":
        from LegacyMotionEditor_Importer import MJCFParser  # lazy: LME app dep

        parser = MJCFParser()
        working_dir = os.path.dirname(os.path.abspath(file_path))
        return parser.parse_mjcf(file_path, working_dir=working_dir)

    from LegacyMotionEditor_Importer import parse_urdf_file  # lazy: LME app dep

    result = parse_urdf_file(file_path, parent_widget=None)
    if not result:
        raise RuntimeError(f"Failed to parse model file: {file_path}")
    _, _, parsed_data = result
    return parsed_data


def detect_model_type_for_path(file_path: str) -> str:
    from LegacyMotionEditor_Importer import detect_model_type  # lazy: LME app dep

    return detect_model_type(file_path) or "urdf"


# ===========================================================================
# simple lookup helpers
# ===========================================================================

_nc_singleton: "NameConverter | None" = None

def _get_nc() -> "NameConverter":
    global _nc_singleton
    if _nc_singleton is None:
        _nc_singleton = NameConverter()
    return _nc_singleton


def canonicalize(name: str, entity: str = "auto") -> str | None:
    """Return the canonical target name for any raw joint/link name.

    Returns the best-match target string, or None if completely unresolved.
    For AMBIGUOUS results the highest-scored candidate is returned.
    """
    def _best(r: "ConversionResult") -> "str | None":
        if r.status == ConversionStatus.RESOLVED:
            return r.target
        if r.status == ConversionStatus.AMBIGUOUS and r.candidates:
            return r.candidates[0].target
        return None

    nc = _get_nc()
    normalized_entity = normalize_entity_type(entity)
    if normalized_entity == EntityType.AUTO.value:
        normalized_entity = EntityType.AUTO.value

    result = nc.convert(name, entity=normalized_entity)
    hit = _best(result)
    if hit:
        return hit

    if entity == "auto":
        hits: dict[str, str] = {}
        for retry in (EntityType.JOINT.value, EntityType.LINK.value):
            h = _best(nc.convert(name, entity=retry))
            if h:
                hits[retry] = h
        norm = re.sub(r"[\s\-/]+", "_", name.strip().lower())
        if "link" in hits and compact_link_name(hits["link"]) == norm:
            return hits["link"]
        return hits.get(EntityType.JOINT.value) or hits.get(EntityType.LINK.value)

    return None


def parent_link(canonical_name: str) -> str | None:
    """Return the parent link of a canonical link name using the stored link_tree.

    Returns None if the name is a root link or is not found in the tree.
    """
    tree: dict = load_master().get("link_tree", {})
    node = tree.get(canonical_name)
    if node is None:
        return None
    return node.get("parent")


def ancestor_links(canonical_name: str) -> list[str]:
    """Return ancestor chain [self, parent, grandparent, ...] from link_tree."""
    tree: dict = load_master().get("link_tree", {})
    chain: list[str] = []
    cur: str | None = canonical_name
    seen: set[str] = set()
    while cur and cur in tree and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = tree[cur].get("parent")
    return chain


def is_ancestor_link(ancestor: str, descendant: str) -> bool:
    """True when ``ancestor`` appears in the link_tree parent chain of ``descendant``."""
    return ancestor in ancestor_links(descendant)


def resolve_urdf_parent_in_tree(child_canonical: str, urdf_parent_canonical: str) -> bool:
    """Check URDF parent against link_tree, walking ancestors when names differ.

    Returns True if ``urdf_parent_canonical`` equals the direct tree parent or any
    ancestor of ``child_canonical`` in link_tree.
    """
    if not child_canonical or not urdf_parent_canonical:
        return False

    def _norm(name: str) -> str:
        compact = compact_link_name(name)
        if compact in {"base_link", "c_base_link"} or is_preserved_link(name):
            return "c_base_link"
        return compact

    child = _norm(child_canonical)
    parent = _norm(urdf_parent_canonical)
    if parent_link(child) == parent:
        return True
    return is_ancestor_link(parent, child)


def _link_tree_hop_joint(tree: dict, parent: str, child: str) -> str | None:
    node = tree.get(parent)
    if not node:
        return None
    for item in node.get("children", []):
        if item.get("link") == child:
            return item.get("via_joint")
    return None


def _find_link_tree_path(tree: dict, start: str, goal: str) -> list[str] | None:
    if start == goal:
        return [start]
    from collections import deque

    queue: deque[list[str]] = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        for child in tree.get(node, {}).get("children", []):
            link = child.get("link")
            if not link or link in seen:
                continue
            next_path = path + [link]
            if link == goal:
                return next_path
            seen.add(link)
            queue.append(next_path)
    return None


def joint_on_link_tree_path(parent: str, child: str) -> list[str]:
    """Return via_joint names for hops from parent to child through link_tree (BFS)."""
    tree: dict = load_master().get("link_tree", {})
    path = _find_link_tree_path(tree, parent, child)
    if not path or len(path) < 2:
        return []
    joints: list[str] = []
    for idx in range(len(path) - 1):
        via = _link_tree_hop_joint(tree, path[idx], path[idx + 1])
        if via:
            joints.append(via)
    return joints


def children_links(canonical_name: str) -> list[str]:
    """Return the direct child link names of a canonical link name."""
    tree: dict = load_master().get("link_tree", {})
    node = tree.get(canonical_name)
    if node is None:
        return []
    return [c["link"] for c in node.get("children", [])]


# ===========================================================================
# lme_integration  (public entry points called by LegacyMotionEditor_Utils)
# ===========================================================================

def _get_motion_context(joint_editor: Any) -> tuple[Any, Any, str, str]:
    graph = getattr(joint_editor, "graph", None)
    robot_model = getattr(joint_editor, "robot_model", None)
    if robot_model is None and graph is not None:
        motion_state = getattr(graph, "motion_state", None) or {}
        robot_model = motion_state.get("robot_model")

    model_path = ""
    model_type = "urdf"
    if robot_model is not None:
        model_path = getattr(robot_model, "urdf_path", "") or ""
        model_type = getattr(robot_model, "model_type", "") or model_type
    if graph is not None:
        motion_state = getattr(graph, "motion_state", None) or {}
        model_path = model_path or motion_state.get("urdf_path", "")
        model_type = motion_state.get("model_type", "") or model_type

    if model_path and (not model_type or model_type == "urdf"):
        model_type = detect_model_type_for_path(model_path)

    return graph, robot_model, model_path, model_type


def plan_joint_rename(joint_editor: Any) -> dict[str, Any]:
    graph, robot_model, model_path, model_type = _get_motion_context(joint_editor)
    if robot_model is None:
        raise RuntimeError("ロボットモデルが読み込まれていません。")
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError("読み込み済みの URDF/MJCF ファイルパスが見つかりません。")

    parsed_data = load_parsed_model(model_path, model_type)
    joint_map, unresolved, skipped_fixed, joint_unchanged_map, joint_ambiguous = build_joint_rename_map(
        parsed_data,
        model_type,
        robot_model=robot_model,
        single_arm_as_left=True,
    )
    link_map, link_unresolved, preserved_link_map, link_unchanged_map, link_ambiguous = build_link_rename_map(
        parsed_data,
        model_type,
        robot_model=robot_model,
        single_arm_as_left=True,
    )
    link_map = sanitize_link_rename_map(link_map)
    return {
        "graph": graph,
        "robot_model": robot_model,
        "model_path": model_path,
        "model_type": model_type,
        "parsed_data": parsed_data,
        "joint_map": joint_map,
        "joint_unchanged_map": joint_unchanged_map,
        "joint_ambiguous": joint_ambiguous,
        "link_map": link_map,
        "link_unchanged_map": link_unchanged_map,
        "preserved_link_map": preserved_link_map,
        "unresolved": unresolved,
        "link_unresolved": link_unresolved,
        "link_ambiguous": link_ambiguous,
        "skipped_fixed": skipped_fixed,
        "include_links": False,
    }


def apply_joint_rename_plan(
    joint_editor: Any,
    plan: dict[str, Any],
    *,
    include_links: bool = False,
) -> dict[str, int]:
    joint_map = filter_identity_rename_map(plan.get("joint_map", {}))
    link_map = plan.get("link_map", {}) if include_links else {}
    link_map = sanitize_link_rename_map(link_map)
    link_map = filter_identity_rename_map(link_map)
    graph = plan.get("graph") or getattr(joint_editor, "graph", None)
    apply_joint_rename_to_joint_editor(
        joint_editor,
        joint_map,
        link_map=link_map,
    )
    pose_updates = apply_joint_rename_to_graph(graph, joint_map)
    plan["include_links"] = include_links
    return {
        "renamed_joints": len(joint_map),
        "renamed_links": len(link_map),
        "pose_nodes_updated": pose_updates,
        "unresolved": len(plan.get("unresolved", [])),
        "link_unresolved": len(plan.get("link_unresolved", [])),
        "joint_ambiguous": len(plan.get("joint_ambiguous", [])),
        "link_ambiguous": len(plan.get("link_ambiguous", [])),
    }


def overwrite_loaded_model_file(joint_editor: Any, plan: dict[str, Any]) -> str | None:
    model_path = plan.get("model_path", "")
    model_type = plan.get("model_type", "urdf")
    joint_map = plan.get("joint_map", {})
    link_map = plan.get("link_map", {}) if plan.get("include_links") else {}
    return rewrite_model_file_joint_names(
        model_path,
        model_type,
        joint_map,
        link_map=link_map,
        create_backup=True,
    )


# ===========================================================================
# __all__
# ===========================================================================

__all__ = [
    "NameConverter",
    "Candidate",
    "ConversionResult",
    "ConversionStatus",
    "EntityType",
    "ModelConversionResult",
    "ParsedModelConversion",
    "RenameMapValidation",
    "RenameMapValidationError",
    "convert_model_file",
    "convert_parsed_model",
    "extract_model_entities",
    "plan_joint_rename",
    "apply_joint_rename_plan",
    "overwrite_loaded_model_file",
    "normalize_name",
    "normalize_variants",
    "split_parent_child",
    "build_joint_rename_map",
    "build_link_rename_map",
    "sanitize_link_rename_map",
    "filter_identity_rename_map",
    "apply_joint_rename_to_robot_model",
    "apply_link_rename_to_robot_model",
    "apply_joint_rename_to_graph",
    "rewrite_model_file_joint_names",
    "load_parsed_model",
    "detect_model_type_for_path",
    "load_master",
    "reload_master",
    "canonicalize",
    "parent_link",
    "ancestor_links",
    "children_links",
    "resolve_urdf_parent_in_tree",
    "normalize_entity_type",
]

__version__ = "1.5.0"


# ===========================================================================
# Qt UI helpers (Viewer + Editor)
# ===========================================================================

import html

from PySide6 import QtCore, QtGui, QtWidgets

_UI_CONF_COLOR = {
    "high": "#4caf50",
    "medium": "#2196f3",
    "low": "#9e9e9e",
    "proposed": "#9e9e9e",
}
_UI_METHOD_COLOR = {
    "direct": "#e0e0e0",
    "alias": "#e0e0e0",
    "product_alias": "#81c784",
    "topology_alias": "#64b5f6",
    "functional_alias": "#ffb74d",
    "heuristic": "#9e9e9e",
    "contextual_pattern": "#ce93d8",
}
_UI_ENTITY_COLOR = {
    "link": "#80deea",   # 水色
    "joint": "#f8bbd0",  # 薄いピンク
}

ALIAS_COLS = ("Alias", "Target", "Entity", "Confidence", "Method", "Source")
COL_TARGET = 1


def normalize_search_query(raw: str) -> str:
    return re.sub(r"[\s\-/]+", "_", raw.strip().lower())


def ui_entity_color(entity: str) -> str:
    return _UI_ENTITY_COLOR.get(entity, "#e0e0e0")


def style_tree_item_entity_colors(item: "QtWidgets.QTreeWidgetItem") -> None:
    """Link 列は水色、via_joint 列は薄いピンク。"""
    item.setForeground(0, QtGui.QColor(_UI_ENTITY_COLOR["link"]))
    joint = item.text(1).strip()
    if joint:
        item.setForeground(1, QtGui.QColor(_UI_ENTITY_COLOR["joint"]))


def ui_html_colored(text: str, entity: str) -> str:
    color = ui_entity_color(entity)
    return f'<span style="color:{color}">{html.escape(text)}</span>'


def apply_fusion_dark_theme(app: "QtWidgets.QApplication") -> None:
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#1e1e1e"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e0e0e0"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#2b2b2b"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#252525"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#e0e0e0"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#3c3c3c"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#e0e0e0"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#0d47a1"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor("#9e9e9e"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QLineEdit, QTextEdit, QPlainTextEdit {
            color: #e0e0e0;
        }
        QLineEdit::placeholder, QTextEdit::placeholder, QPlainTextEdit::placeholder {
            color: #9e9e9e;
        }
    """)


def master_version_label(_master: dict[str, Any] | None = None) -> str:
    return f"Robot Label Bridge v{__version__}"


def collect_alias_rows(
    alias_index: dict[str, list[dict]],
    *,
    query: str = "",
    entity_filter: str = "all",
) -> list[tuple[str, int, dict]]:
    rows: list[tuple[str, int, dict]] = []
    for alias_key, entries in alias_index.items():
        if query and query not in alias_key:
            continue
        for idx, entry in enumerate(entries):
            ent = entry.get("entity", "")
            if entity_filter != "all" and ent != entity_filter:
                continue
            rows.append((alias_key, idx, entry))
    rows.sort(
        key=lambda x: (
            -CONFIDENCE_RANK.get(x[2].get("confidence", ""), 0),
            x[0],
            x[2].get("target", ""),
        )
    )
    return rows


def fill_alias_table_row(
    table: "QtWidgets.QTableWidget",
    row: int,
    alias_key: str,
    entry: dict,
    *,
    editable_target: bool = False,
    edited: bool = False,
) -> None:
    values = (
        alias_key,
        entry.get("target", ""),
        entry.get("entity", ""),
        entry.get("confidence", ""),
        entry.get("mapping_type", ""),
        entry.get("source", ""),
    )
    entity = entry.get("entity", "")
    row_color = ui_entity_color(entity)
    for col, text in enumerate(values):
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        if not editable_target or col != COL_TARGET:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        if col == COL_TARGET and edited:
            item.setForeground(QtGui.QColor("#ff9800"))
        elif col == 3:
            item.setForeground(QtGui.QColor(_UI_CONF_COLOR.get(text, row_color)))
        elif col == 4:
            item.setForeground(QtGui.QColor(_UI_METHOD_COLOR.get(text, row_color)))
        else:
            item.setForeground(QtGui.QColor(row_color))
        table.setItem(row, col, item)


def build_alias_table(*, editable: bool) -> "QtWidgets.QTableWidget":
    table = QtWidgets.QTableWidget(0, len(ALIAS_COLS))
    table.setHorizontalHeaderLabels(ALIAS_COLS)
    hh = table.horizontalHeader()
    hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
    hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
    for col in (2, 3, 4, 5):
        hh.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    if editable:
        table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
    else:
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    return table


class LinkTreePanel(QtWidgets.QWidget):
    """Read-only link_tree browser with search."""

    def __init__(self, parent: "QtWidgets.QWidget | None" = None) -> None:
        super().__init__(parent)
        self._tree_data: dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Filter:"))
        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("link name…")
        self._filter.setClearButtonEnabled(True)
        row.addWidget(self._filter, 1)
        root.addLayout(row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Link", "via_joint"])
        self._tree.setAlternatingRowColors(True)
        splitter.addWidget(self._tree)

        self._detail = QtWidgets.QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText("Select a link…")
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self._filter.textChanged.connect(self._apply_filter)
        self._tree.itemSelectionChanged.connect(self._on_select)

    def set_link_tree(self, tree: dict[str, Any]) -> None:
        self._tree_data = tree or {}
        self._rebuild()

    def _rebuild(self) -> None:
        self._tree.clear()
        if not self._tree_data:
            return
        roots = [k for k, n in self._tree_data.items() if not n.get("parent")]
        roots.sort()
        for root_name in roots:
            item = self._make_item(root_name)
            self._tree.addTopLevelItem(item)
            self._populate(item, root_name)
        self._tree.expandToDepth(1)
        self._apply_filter()

    def _make_item(self, link: str) -> "QtWidgets.QTreeWidgetItem":
        node = self._tree_data.get(link, {})
        parent = node.get("parent") or "(root)"
        item = QtWidgets.QTreeWidgetItem([link, ""])
        item.setData(0, QtCore.Qt.UserRole, link)
        item.setToolTip(0, f"parent: {parent}")
        style_tree_item_entity_colors(item)
        return item

    def _populate(self, parent_item: "QtWidgets.QTreeWidgetItem", link: str) -> None:
        for child in self._tree_data.get(link, {}).get("children", []):
            child_link = child.get("link", "")
            if not child_link:
                continue
            item = QtWidgets.QTreeWidgetItem([child_link, child.get("via_joint", "")])
            item.setData(0, QtCore.Qt.UserRole, child_link)
            style_tree_item_entity_colors(item)
            parent_item.addChild(item)
            if child_link in self._tree_data:
                self._populate(item, child_link)

    def _apply_filter(self) -> None:
        query = normalize_search_query(self._filter.text())

        def walk(item: "QtWidgets.QTreeWidgetItem") -> bool:
            link = item.data(0, QtCore.Qt.UserRole) or item.text(0)
            match = (not query) or (query in str(link).lower())
            child_match = False
            for i in range(item.childCount()):
                if walk(item.child(i)):
                    child_match = True
            visible = match or child_match
            item.setHidden(not visible)
            return visible

        for i in range(self._tree.topLevelItemCount()):
            walk(self._tree.topLevelItem(i))

    def _on_select(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            self._detail.clear()
            return
        link = items[0].data(0, QtCore.Qt.UserRole) or items[0].text(0)
        node = self._tree_data.get(link, {})
        lines = [
            f"link: {ui_html_colored(str(link), 'link')}",
            f"parent: {ui_html_colored(str(node.get('parent') or ''), 'link')}",
            "children:",
        ]
        for child in node.get("children", []):
            child_link = str(child.get("link", ""))
            via_joint = str(child.get("via_joint", ""))
            lines.append(
                f"  -&gt; {ui_html_colored(child_link, 'link')}  "
                f"via {ui_html_colored(via_joint, 'joint')}"
            )
        self._detail.setHtml("<br>".join(lines))


class ConvertPanel(QtWidgets.QWidget):
    """Live conversion preview using NameConverter API."""

    def __init__(
        self,
        convert_fn: Callable[..., Any],
        parent_link_fn: Callable[[str], str | None],
        ancestor_fn: Callable[[str], list[str]],
        parent: "QtWidgets.QWidget | None" = None,
    ) -> None:
        super().__init__(parent)
        self._convert = convert_fn
        self._parent_link = parent_link_fn
        self._ancestor = ancestor_fn
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self._name = QtWidgets.QLineEdit()
        self._name.setPlaceholderText("e.g. l_shoulder_xr / c_waist_to_c_chest / HeadYaw")
        form.addRow("Name:", self._name)

        self._entity = QtWidgets.QComboBox()
        self._entity.addItems(["auto", "joint", "link"])
        form.addRow("Entity:", self._entity)

        self._parent = QtWidgets.QLineEdit()
        self._parent.setPlaceholderText("optional URDF parent link")
        form.addRow("Parent:", self._parent)

        self._child = QtWidgets.QLineEdit()
        self._child.setPlaceholderText("optional URDF child link")
        form.addRow("Child:", self._child)

        root.addLayout(form)
        self._run_btn = QtWidgets.QPushButton("Convert")
        root.addWidget(self._run_btn)
        self._out = QtWidgets.QTextEdit()
        self._out.setReadOnly(True)
        root.addWidget(self._out, 1)
        self._run_btn.clicked.connect(self._run)
        self._name.returnPressed.connect(self._run)

    def _run(self) -> None:
        name = self._name.text().strip()
        if not name:
            self._out.setPlainText("Enter a name.")
            return
        entity = self._entity.currentText()
        parent = self._parent.text().strip() or None
        child = self._child.text().strip() or None
        result = self._convert(name, entity=entity, parent=parent, child=child)
        resolved_entity = str(result.entity)
        target_html = (
            ui_html_colored(str(result.target), resolved_entity)
            if result.target
            else "(none)"
        )
        lines = [
            f"status: {html.escape(str(getattr(result.status, 'value', result.status)))}",
            f"target: {target_html}",
            f"entity: {ui_html_colored(resolved_entity, resolved_entity)}",
        ]
        if result.reasons:
            lines.append(f"reasons: {html.escape('; '.join(result.reasons))}")
        if result.target and entity in ("auto", "link") and resolved_entity == "link":
            pl = self._parent_link(result.target)
            anc = self._ancestor(result.target)
            pl_html = ui_html_colored(str(pl), "link") if pl else "(none)"
            anc_html = " &rarr; ".join(
                ui_html_colored(str(a), "link") for a in anc
            )
            lines.append(f"parent_link: {pl_html}")
            lines.append(f"ancestors: {anc_html}")
        if result.candidates:
            lines.append("<br>candidates:")
            for cand in sorted(
                result.candidates,
                key=lambda c: getattr(c, "score", 0),
                reverse=True,
            )[:8]:
                cand_entity = str(getattr(cand, "entity", resolved_entity))
                lines.append(
                    f"  {ui_html_colored(cand.target, cand_entity)}  "
                    f"score={html.escape(str(getattr(cand, 'score', '?')))}  "
                    f"{html.escape(cand.mapping_type)}  "
                    f"{html.escape(cand.source)}"
                )
        self._out.setHtml("<br>".join(lines))


class BridgeViewer(QtWidgets.QMainWindow):
    """Read-only viewer: alias lookup, link_tree browse, conversion preview."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Robot Label Bridge — Viewer")
        self.resize(980, 640)
        self._master: dict = load_master()
        self._build_ui()
        self._reload_data()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        self._version_lbl = QtWidgets.QLabel("")
        self._version_lbl.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.setFixedWidth(90)
        reload_btn.clicked.connect(self._reload_data)
        header.addWidget(self._version_lbl, 1)
        header.addWidget(reload_btn)
        root.addLayout(header)

        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)

        alias_page = QtWidgets.QWidget()
        alias_layout = QtWidgets.QVBoxLayout(alias_page)
        alias_layout.setContentsMargins(0, 8, 0, 0)
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Search:"))
        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("alias filter…")
        self._search.setClearButtonEnabled(True)
        filter_row.addWidget(self._search, 1)
        filter_row.addWidget(QtWidgets.QLabel("Entity:"))
        self._entity_filter = QtWidgets.QComboBox()
        self._entity_filter.addItems(["all", "joint", "link"])
        filter_row.addWidget(self._entity_filter)
        self._count_lbl = QtWidgets.QLabel("")
        self._count_lbl.setFixedWidth(90)
        self._count_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._count_lbl.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        filter_row.addWidget(self._count_lbl)
        alias_layout.addLayout(filter_row)

        self._alias_table = build_alias_table(editable=False)
        alias_layout.addWidget(self._alias_table, 1)
        tabs.addTab(alias_page, "Aliases")

        self._link_panel = LinkTreePanel()
        tabs.addTab(self._link_panel, "Link Tree")

        nc = _get_nc()
        self._convert_panel = ConvertPanel(
            nc.convert,
            parent_link,
            ancestor_links,
        )
        tabs.addTab(self._convert_panel, "Convert")

        self._search.textChanged.connect(self._refresh_aliases)
        self._entity_filter.currentTextChanged.connect(self._refresh_aliases)

    def _reload_data(self) -> None:
        self._master = reload_master()
        self._version_lbl.setText(master_version_label(self._master))
        self._link_panel.set_link_tree(self._master.get("link_tree", {}))
        self._refresh_aliases()

    def _refresh_aliases(self) -> None:
        query = normalize_search_query(self._search.text())
        entity_filter = self._entity_filter.currentText()
        alias_index = self._master.get("alias_index", {})
        rows = collect_alias_rows(
            alias_index,
            query=query,
            entity_filter=entity_filter,
        )
        self._alias_table.setRowCount(len(rows))
        for row_idx, (alias_key, _idx, entry) in enumerate(rows):
            fill_alias_table_row(
                self._alias_table,
                row_idx,
                alias_key,
                entry,
                editable_target=False,
            )
        count = len(rows)
        if query or entity_filter != "all":
            self._count_lbl.setText(f"{count} match{'es' if count != 1 else ''}")
        else:
            self._count_lbl.setText(f"{count} entries")


def run_viewer() -> None:
    """Launch the read-only Robot Label Bridge viewer."""
    import sys

    app = QtWidgets.QApplication(sys.argv)
    apply_fusion_dark_theme(app)
    win = BridgeViewer()
    win.show()
    sys.exit(app.exec())


# ===========================================================================
# Standalone lookup UI  (python Robot_Label_Bridge.py)
# ===========================================================================

if __name__ == "__main__":
    run_viewer()
