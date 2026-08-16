"""
File Name: RobotLabelBridge.py
Description: Robot joint/link name conversion and rename utilities for LegacyMotionEditor.

Author      : Izumi Ninagawa
License     : MIT License
Copyright (c) 2026 Izumi Ninagawa

Canonical naming data is loaded from ``RobotLabelBridge_Master.json``
(located next to this module). Legacy ``robot_label_bridge_master.json``
is accepted as a fallback path.

Public API:
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
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
_MASTER_PATH = Path(__file__).resolve().parent / "RobotLabelBridge_Master.json"
_LEGACY_MASTER_PATH = Path(__file__).resolve().parent / "robot_label_bridge_master.json"

# ===========================================================================
# models
# ===========================================================================

class EntityType(str, Enum):
    JOINT = "joint"
    LINK = "link"
    LOOP = "loop"
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
    if text in ("loop", "kinematic_loop", "closed_loop"):
        return EntityType.LOOP.value
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
    reason_codes: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    target_entity_id: str | None = None
    morphology: str | None = None

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
    reason_codes: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    target_entity_id: str | None = None

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
# ontology vocab  (controlled vocabulary for lightweight robotics ontology)
# ===========================================================================

class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


class Morphology(str, Enum):
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"
    AVIAN = "avian"
    GENERIC_VERTEBRATE = "generic_vertebrate"
    GENERIC_ROBOT = "generic_robot"


class JointType(str, Enum):
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    SPHERICAL = "spherical"
    FLOATING = "floating"
    PROFILE_DEFINED = "profile_defined"


class MappingConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PROPOSED = "proposed"


class EntityStatus(str, Enum):
    CANONICAL = "canonical"
    PROVISIONAL = "provisional"
    DEPRECATED = "deprecated"
    LEGACY = "legacy"


class LinkKind(str, Enum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"
    AXIS_DECOMPOSITION = "axis_decomposition"
    PROVISIONAL = "provisional"


class MotionClass(str, Enum):
    FLEXION_EXTENSION = "flexion_extension"
    ABDUCTION_ADDUCTION = "abduction_adduction"
    INTERNAL_EXTERNAL_ROTATION = "internal_external_rotation"
    AXIAL_ROTATION = "axial_rotation"
    OPENING_CLOSING = "opening_closing"
    TRANSLATION = "translation"
    PROTRACTION_RETRACTION = "protraction_retraction"
    ELEVATION_DEPRESSION = "elevation_depression"
    INVERSION_EVERSION = "inversion_eversion"
    UNKNOWN = "unknown"


class ReasonCode(str, Enum):
    EXACT_ALIAS = "exact_alias"
    CONTEXTUAL_ALIAS = "contextual_alias"
    TOPOLOGY_MATCH = "topology_match"
    AXIS_MATCH = "axis_match"
    SIDE_MATCH = "side_match"
    TOKEN_HEURISTIC = "token_heuristic"
    PRODUCT_ALIAS = "product_alias"
    FUNCTIONAL_ALIAS = "functional_alias"
    MORPHOLOGY_MATCH = "morphology_match"
    MORPHOLOGY_GENERIC = "morphology_generic"
    MORPHOLOGY_MISMATCH = "morphology_mismatch"
    DIRECT_MAPPING = "direct_mapping"
    POLICY_MATCH = "policy_match"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_LATERALITY_FROM_LABEL = {
    "l": Laterality.LEFT,
    "left": Laterality.LEFT,
    "r": Laterality.RIGHT,
    "right": Laterality.RIGHT,
    "c": Laterality.CENTER,
    "center": Laterality.CENTER,
    "centre": Laterality.CENTER,
}

_LATERALITY_TO_LABEL = {
    Laterality.LEFT: "l",
    Laterality.RIGHT: "r",
    Laterality.CENTER: "c",
}

_MORPHOLOGY_ALIASES = {
    "humanoid": Morphology.HUMANOID,
    "quadruped": Morphology.QUADRUPED,
    "avian": Morphology.AVIAN,
    "generic_vertebrate": Morphology.GENERIC_VERTEBRATE,
    "generic vertebrate": Morphology.GENERIC_VERTEBRATE,
    "generic_robot": Morphology.GENERIC_ROBOT,
    "generic robot": Morphology.GENERIC_ROBOT,
}

_JOINT_TYPE_ALIASES = {
    "revolute": JointType.REVOLUTE,
    "continuous": JointType.CONTINUOUS,
    "prismatic": JointType.PRISMATIC,
    "fixed": JointType.FIXED,
    "spherical": JointType.SPHERICAL,
    "floating": JointType.FLOATING,
    "floating/profile_defined": JointType.FLOATING,
    "profile_defined": JointType.PROFILE_DEFINED,
}

_AXIS_PAIR_TO_SHORT = {"xroll": "xr", "ypitch": "yp", "zyaw": "zy"}
_AXIS_PAIR_TO_VECTOR = {
    "xroll": [1.0, 0.0, 0.0],
    "ypitch": [0.0, 1.0, 0.0],
    "zyaw": [0.0, 0.0, 1.0],
}

_MOTION_CLASS_TABLE: dict[str, MotionClass] = {
    "flexion/extension": MotionClass.FLEXION_EXTENSION,
    "shoulder flexion/extension": MotionClass.FLEXION_EXTENSION,
    "elbow flexion/extension": MotionClass.FLEXION_EXTENSION,
    "knee flexion/extension": MotionClass.FLEXION_EXTENSION,
    "wrist flexion/extension": MotionClass.FLEXION_EXTENSION,
    "digit flexion/extension": MotionClass.FLEXION_EXTENSION,
    "toe flexion/extension": MotionClass.FLEXION_EXTENSION,
    "neck flexion/extension": MotionClass.FLEXION_EXTENSION,
    "neck segment flexion/extension": MotionClass.FLEXION_EXTENSION,
    "trunk flexion/extension": MotionClass.FLEXION_EXTENSION,
    "upper trunk flexion/extension": MotionClass.FLEXION_EXTENSION,
    "head nodding (flexion/extension)": MotionClass.FLEXION_EXTENSION,
    "dorsoventral tail flexion": MotionClass.FLEXION_EXTENSION,
    "wing elbow flexion": MotionClass.FLEXION_EXTENSION,
    "wing wrist flexion": MotionClass.FLEXION_EXTENSION,
    "ankle plantarflexion/dorsiflexion": MotionClass.FLEXION_EXTENSION,
    "abduction/adduction": MotionClass.ABDUCTION_ADDUCTION,
    "shoulder abduction/adduction": MotionClass.ABDUCTION_ADDUCTION,
    "internal/external rotation": MotionClass.INTERNAL_EXTERNAL_ROTATION,
    "limb axial rotation": MotionClass.AXIAL_ROTATION,
    "neck axial rotation": MotionClass.AXIAL_ROTATION,
    "wrist/forearm axial rotation": MotionClass.AXIAL_ROTATION,
    "forearm pronation/supination": MotionClass.AXIAL_ROTATION,
    "shoulder horizontal rotation": MotionClass.AXIAL_ROTATION,
    "head rotation (yaw)": MotionClass.AXIAL_ROTATION,
    "root heading": MotionClass.AXIAL_ROTATION,
    "lateral tail flexion": MotionClass.AXIAL_ROTATION,
    "jaw opening/closing": MotionClass.OPENING_CLOSING,
    "clavicle protraction/retraction": MotionClass.PROTRACTION_RETRACTION,
    "wing protraction/retraction": MotionClass.PROTRACTION_RETRACTION,
    "wing elevation/depression": MotionClass.ELEVATION_DEPRESSION,
    "wing sweep": MotionClass.ELEVATION_DEPRESSION,
    "alula deployment": MotionClass.ELEVATION_DEPRESSION,
    "inversion/eversion": MotionClass.INVERSION_EVERSION,
    "ankle inversion/eversion": MotionClass.INVERSION_EVERSION,
    "scapular rotation": MotionClass.AXIAL_ROTATION,
    "scapular upward/downward rotation": MotionClass.ELEVATION_DEPRESSION,
    "horizontal gaze": MotionClass.AXIAL_ROTATION,
}


def normalize_laterality(value: str | Laterality | None) -> Laterality | None:
    if value is None:
        return None
    if isinstance(value, Laterality):
        return value
    key = str(value).strip().lower()
    # str(Laterality.LEFT) may be "Laterality.LEFT" on some Python builds
    if key.startswith("laterality."):
        key = key.split(".", 1)[1]
    return _LATERALITY_FROM_LABEL.get(key)


def laterality_to_label_prefix(lat: Laterality | str | None) -> str:
    if isinstance(lat, Laterality):
        return _LATERALITY_TO_LABEL.get(lat, "c")
    if isinstance(lat, str):
        resolved = normalize_laterality(lat) or Laterality.CENTER
        return _LATERALITY_TO_LABEL.get(resolved, "c")
    return "c"


def normalize_morphology(value: str | None) -> Morphology | None:
    if value is None:
        return None
    return _MORPHOLOGY_ALIASES.get(str(value).strip().lower())


def normalize_joint_type(value: str | None) -> JointType | None:
    if value is None:
        return None
    return _JOINT_TYPE_ALIASES.get(str(value).strip().lower())


def normalize_mapping_confidence(value: str | None) -> MappingConfidence | None:
    if value is None:
        return None
    try:
        return MappingConfidence(str(value).strip().lower())
    except ValueError:
        return None


def parse_axis_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        try:
            return parse_axis_vector(json.loads(text))
        except json.JSONDecodeError:
            m = re.match(
                r"\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]",
                text,
            )
            if m:
                return [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    return None


def motion_class_from_label(semantic_motion: str | None) -> MotionClass:
    if not semantic_motion:
        return MotionClass.UNKNOWN
    return _MOTION_CLASS_TABLE.get(str(semantic_motion).strip().lower(), MotionClass.UNKNOWN)


def axis_short_from_pair(axis_pair: str | None) -> str | None:
    if not axis_pair:
        return None
    return _AXIS_PAIR_TO_SHORT.get(str(axis_pair).strip().lower()) or axis_keyword_to_short(
        str(axis_pair)
    )


# ===========================================================================
# ontology models
# ===========================================================================

@dataclass
class AxisDefinition:
    """Kinematic axis separated from semantic motion / reference pose."""

    vector: list[float] | None = None
    expressed_in: str = "unknown"
    reference_pose: str | None = None
    axis_pair: str | None = None
    positive_semantic_motion: str | None = None


@dataclass
class DegreeOfFreedom:
    """Actuation-axis entity (serialized DOF), distinct from FunctionalJoint."""

    entity_id: str
    canonical_label: str
    formal_label: str | None = None
    expanded_label: str | None = None
    ros_label: str | None = None
    legacy_canonical_id: str | None = None
    dof_of: str | None = None
    laterality: Laterality | None = None
    morphology: Morphology | None = None
    landmark: str | None = None
    joint_type: JointType | None = None
    axis: AxisDefinition = field(default_factory=AxisDefinition)
    motion_class: MotionClass = MotionClass.UNKNOWN
    semantic_motion_label: str | None = None
    parent_link: str | None = None
    child_link: str | None = None
    status: EntityStatus = EntityStatus.CANONICAL
    mapping_confidence: MappingConfidence = MappingConfidence.MEDIUM
    appendage_location: str | None = None
    chain_name: str | None = None
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class FunctionalJoint:
    """Logical multi-DOF joint concept (ontology layer)."""

    entity_id: str
    canonical_label: str
    laterality: Laterality | None = None
    morphology: Morphology | None = None
    landmark: str | None = None
    dof_ids: list[str] = field(default_factory=list)
    status: EntityStatus = EntityStatus.CANONICAL


@dataclass
class LinkEntity:
    """Physical or virtual kinematic link entity."""

    entity_id: str
    canonical_label: str
    formal_label: str | None = None
    ros_label: str | None = None
    legacy_canonical_id: str | None = None
    laterality: Laterality | None = None
    morphology: Morphology | None = None
    link_kind: LinkKind = LinkKind.PHYSICAL
    status: EntityStatus = EntityStatus.CANONICAL
    parent_attachment: str | None = None
    region: str | None = None
    segment_meaning: str | None = None
    mapping_confidence: MappingConfidence = MappingConfidence.MEDIUM
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LoopMechanismType(str, Enum):
    """Closed-loop mechanism class (attribute, not part of canonical identity)."""

    FOUR_BAR = "four_bar"
    FIVE_BAR = "five_bar"
    SIX_BAR = "six_bar"
    PARALLELOGRAM = "parallelogram"
    PANTOGRAPH = "pantograph"
    PARALLEL = "parallel"
    CROSS_LINK = "cross_link"
    DIFFERENTIAL = "differential"
    SPATIAL = "spatial"
    CABLE_LOOP = "cable_loop"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


_LOOP_MECHANISM_ALIASES = {m.value: m for m in LoopMechanismType}

# Canonical short-name patterns (lp = loop, bNN = branch, cl = closure).
# link_tree remains an acyclic spanning tree; closed chains live only in loop ontology.
_LOOP_NAME_RE = re.compile(
    r"^(?P<side>l|r|c)_(?P<body>[a-z0-9]+(?:_[a-z0-9]+)*)_lp(?:_(?P<idx>[0-9]{2}))?$"
)
_BRANCH_NAME_RE = re.compile(
    r"^(?P<loop>(?:l|r|c)_[a-z0-9]+(?:_[a-z0-9]+)*_lp(?:_[0-9]{2})?)_b(?P<bidx>[0-9]{2})$"
)
_CLOSURE_NAME_RE = re.compile(
    r"^(?P<loop>(?:l|r|c)_[a-z0-9]+(?:_[a-z0-9]+)*_lp(?:_[0-9]{2})?)_cl(?:_(?P<cidx>[0-9]{2}))?$"
)


def normalize_loop_mechanism_type(value: str | None) -> LoopMechanismType | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _LOOP_MECHANISM_ALIASES.get(key)


def make_loop_label(side: str, landmark: str, index: int | None = None) -> str:
    """Build canonical loop label: [l|r|c]_<landmark>_lp[_NN]."""
    lat = normalize_laterality(side)
    if lat is None:
        raise ValueError(f"invalid laterality for loop label: {side!r}")
    body = normalize_name(landmark).strip("_")
    if not body or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", body):
        raise ValueError(f"invalid landmark for loop label: {landmark!r}")
    prefix = laterality_to_label_prefix(lat)
    if index is None:
        label = f"{prefix}_{body}_lp"
    else:
        idx = int(index)
        if idx < 1 or idx > 99:
            raise ValueError(f"loop index out of range 1..99: {index}")
        label = f"{prefix}_{body}_lp_{idx:02d}"
    if not _LOOP_NAME_RE.match(label):
        raise ValueError(f"produced invalid loop label: {label}")
    return label


def make_loop_branch_label(loop_label: str, branch_index: int) -> str:
    """Build branch label: <loop>_bNN (zero-padded, always numbered)."""
    if not _LOOP_NAME_RE.match(str(loop_label)):
        raise ValueError(f"invalid loop label: {loop_label!r}")
    idx = int(branch_index)
    if idx < 1 or idx > 99:
        raise ValueError(f"branch index out of range 1..99: {branch_index}")
    return f"{loop_label}_b{idx:02d}"


def make_loop_closure_label(loop_label: str, closure_index: int | None = None) -> str:
    """Build closure label: <loop>_cl or <loop>_cl_NN."""
    if not _LOOP_NAME_RE.match(str(loop_label)):
        raise ValueError(f"invalid loop label: {loop_label!r}")
    if closure_index is None:
        return f"{loop_label}_cl"
    idx = int(closure_index)
    if idx < 1 or idx > 99:
        raise ValueError(f"closure index out of range 1..99: {closure_index}")
    return f"{loop_label}_cl_{idx:02d}"


def is_valid_loop_label(name: str) -> bool:
    return bool(_LOOP_NAME_RE.match(str(name or "")))


def is_valid_loop_branch_label(name: str) -> bool:
    return bool(_BRANCH_NAME_RE.match(str(name or "")))


def is_valid_loop_closure_label(name: str) -> bool:
    return bool(_CLOSURE_NAME_RE.match(str(name or "")))


class ActuationRole(str, Enum):
    """Whether a KinematicJoint is driven, free, or otherwise constrained."""

    ACTUATED = "actuated"
    PASSIVE = "passive"
    COUPLED = "coupled"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


_ACTUATION_ROLE_ALIASES = {m.value: m for m in ActuationRole}


def normalize_actuation_role(value: str | None) -> ActuationRole | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "driven": ActuationRole.ACTUATED,
        "active": ActuationRole.ACTUATED,
        "servo": ActuationRole.ACTUATED,
        "free": ActuationRole.PASSIVE,
        "unactuated": ActuationRole.PASSIVE,
        "profile_defined": ActuationRole.ACTUATED,
    }
    if key in aliases:
        return aliases[key]
    return _ACTUATION_ROLE_ALIASES.get(key)


class ClosureConstraintType(str, Enum):
    """Semantic class of a closure constraint (NOT a joint type).

    Joint geometry (revolute/prismatic/...) lives on KinematicJoint.joint_type.
    """

    LOOP_CLOSURE = "loop_closure"
    POINT_COINCIDENCE = "point_coincidence"
    WELD = "weld"
    DISTANCE = "distance"
    GEAR = "gear"
    COUPLING = "coupling"
    TENDON = "tendon"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


_CLOSURE_TYPE_ALIASES = {m.value: m for m in ClosureConstraintType}
_CLOSURE_TYPE_ALIASES.update(
    {
        "kinematic": ClosureConstraintType.LOOP_CLOSURE,  # legacy synonym
        "revolute_closure": ClosureConstraintType.LOOP_CLOSURE,
        "prismatic_closure": ClosureConstraintType.LOOP_CLOSURE,
    }
)


def normalize_closure_constraint_type(value: str | None) -> ClosureConstraintType | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _CLOSURE_TYPE_ALIASES.get(key)


@dataclass
class KinematicJoint:
    """Physical/kinematic connection between two links (≠ DegreeOfFreedom, ≠ FunctionalJoint).

    DegreeOfFreedom = controllable / serialized motion coordinate (often a servo axis).
    KinematicJoint = geometric connection in the mechanism graph.
    ActuationRole distinguishes actuated vs passive pivots without subclassing.
    """

    entity_id: str
    canonical_label: str
    parent_link: str
    child_link: str
    joint_type: JointType | None = None
    actuation_role: ActuationRole = ActuationRole.UNKNOWN
    dof_ids: list[str] = field(default_factory=list)  # DegreeOfFreedom canonical labels
    laterality: Laterality | None = None
    morphology: Morphology | None = None
    status: EntityStatus = EntityStatus.CANONICAL
    mapping_confidence: MappingConfidence = MappingConfidence.MEDIUM
    realizes_functional_joint_id: str | None = None
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class MigrationIssue:
    code: str
    message: str
    path: str = ""
    before: Any = None
    after: Any = None


@dataclass
class MigrationReport:
    issues: list[MigrationIssue] = field(default_factory=list)

    def add(self, issue: MigrationIssue) -> None:
        self.issues.append(issue)

    @property
    def rewrite_count(self) -> int:
        return sum(1 for i in self.issues if i.code.endswith("_rewritten") or "rewrite" in i.code)


@dataclass
class LoopBranch:
    """Consecutive direct-edge path belonging to a KinematicLoop (not a PhysicalLink).

    link_path must list immediately adjacent links (tree edge or same-loop closure edge).
    joint_path[i] is the KinematicJoint on the edge link_path[i] -- link_path[i+1].
    """

    entity_id: str
    canonical_label: str
    loop_id: str
    link_path: list[str] = field(default_factory=list)
    joint_path: list[str] = field(default_factory=list)
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ClosureConstraint:
    """Non-tree closure relation (independent of KinematicJoint / DegreeOfFreedom).

    represented_by_joint optionally points at a KinematicJoint that realizes the
    closure edge (often a passive pivot). Absence means an abstract constraint.
    Legacy Master field "via_joint" is accepted as an alias on load.
    """

    entity_id: str
    canonical_label: str
    loop_id: str
    from_link: str
    to_link: str
    constraint_type: ClosureConstraintType = ClosureConstraintType.LOOP_CLOSURE
    represented_by_joint: str | None = None
    status: EntityStatus = EntityStatus.CANONICAL
    mapping_confidence: MappingConfidence = MappingConfidence.MEDIUM
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def via_joint(self) -> str | None:
        """Deprecated alias for represented_by_joint (API compatibility)."""
        return self.represented_by_joint


@dataclass
class KinematicLoop:
    """Closed-chain mechanism ontology entity (spanning-tree + closure constraints)."""

    entity_id: str
    canonical_label: str
    laterality: Laterality | None = None
    landmark: str | None = None
    morphology: Morphology | None = None
    mechanism_type: LoopMechanismType = LoopMechanismType.UNKNOWN
    branch_ids: list[str] = field(default_factory=list)
    closure_ids: list[str] = field(default_factory=list)
    member_link_ids: list[str] = field(default_factory=list)
    member_dof_ids: list[str] = field(default_factory=list)
    member_joint_ids: list[str] = field(default_factory=list)  # KinematicJoint labels
    status: EntityStatus = EntityStatus.CANONICAL
    mapping_confidence: MappingConfidence = MappingConfidence.MEDIUM
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)



@dataclass
class AliasAssertion:
    """Lexical mapping assertion onto a canonical entity (not the entity itself)."""

    alias: str
    target_entity_id: str | None
    target_label: str
    entity_type: str
    alias_type: str = "alias"
    source: str = ""
    confidence: MappingConfidence = MappingConfidence.MEDIUM
    profile: str | None = None
    context_constraints: str | None = None
    notes: str | None = None


@dataclass
class MappingAssertion:
    """Provenance / epistemic metadata about a mapping."""

    source_label: str
    target_entity_id: str | None
    target_label: str
    mapping_type: str
    confidence: MappingConfidence = MappingConfidence.MEDIUM
    source: str = ""
    notes: str | None = None
    profile: str | None = None
    source_document: str | None = None
    source_version: str | None = None
    author: str | None = None
    reviewed_by: str | None = None
    review_status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    evidence: str | None = None


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    path: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    infos: list[ValidationIssue] = field(default_factory=list)
    dangling_references: list[ValidationIssue] = field(default_factory=list)
    duplicate_ids: list[ValidationIssue] = field(default_factory=list)
    vocabulary_violations: list[ValidationIssue] = field(default_factory=list)
    deprecated_references: list[ValidationIssue] = field(default_factory=list)
    provisional_references: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, issue: ValidationIssue) -> None:
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        else:
            self.infos.append(issue)
        if issue.code.startswith("dangling"):
            self.dangling_references.append(issue)
        elif issue.code.startswith("duplicate"):
            self.duplicate_ids.append(issue)
        elif issue.code.startswith("vocab"):
            self.vocabulary_violations.append(issue)
        elif issue.code.startswith("deprecated"):
            self.deprecated_references.append(issue)
        elif issue.code.startswith("provisional"):
            self.provisional_references.append(issue)

    def summary(self) -> str:
        lines = [
            "Master validation",
            "-----------------",
            f"entities (DOF): {self.stats.get('dof_count', '?')}",
            f"functional joints: {self.stats.get('functional_joint_count', '?')}",
            f"links: {self.stats.get('link_count', '?')}",
            f"virtual/axis links: {self.stats.get('virtual_link_count', '?')}",
            f"aliases: {self.stats.get('alias_count', '?')}",
            f"link_tree nodes: {self.stats.get('tree_node_count', '?')}",
            f"kinematic joints (explicit): {self.stats.get('kjoint_count', 0)}",
            f"kinematic loops: {self.stats.get('loop_count', 0)}",
            f"loop branches: {self.stats.get('loop_branch_count', 0)}",
            f"closures: {self.stats.get('closure_count', 0)}",
            f"stage: {self.stats.get('validation_stage', 'migrated')}",
            f"errors: {len(self.errors)}",
            f"warnings: {len(self.warnings)}",
            f"dangling refs: {len(self.dangling_references)}",
            f"invalid vocabulary: {len(self.vocabulary_violations)}",
            f"duplicate IDs: {len(self.duplicate_ids)}",
            f"topology cycles: {self.stats.get('cycle_count', 0)}",
            f"provisional references: {len(self.provisional_references)}",
            f"deprecated references: {len(self.deprecated_references)}",
        ]
        shown = self.errors + self.warnings
        for issue in shown[:50]:
            loc = f" @ {issue.path}" if issue.path else ""
            lines.append(f"  [{issue.severity.value}] {issue.code}: {issue.message}{loc}")
        if len(shown) > 50:
            lines.append(f"  ... ({len(shown) - 50} more)")
        return "\n".join(lines)


@dataclass
class OntologyRegistry:
    """In-memory lightweight ontology registry derived from Master JSON."""

    schema_version: str = "2.1"
    ontology_version: str = "1.0"
    data_version: str = ""
    dofs: dict[str, DegreeOfFreedom] = field(default_factory=dict)
    dofs_by_label: dict[str, DegreeOfFreedom] = field(default_factory=dict)
    functional_joints: dict[str, FunctionalJoint] = field(default_factory=dict)
    links: dict[str, LinkEntity] = field(default_factory=dict)
    links_by_label: dict[str, LinkEntity] = field(default_factory=dict)
    aliases: list[AliasAssertion] = field(default_factory=list)
    alias_by_key: dict[str, list[AliasAssertion]] = field(default_factory=dict)
    kinematic_joints: dict[str, KinematicJoint] = field(default_factory=dict)
    kinematic_joints_by_label: dict[str, KinematicJoint] = field(default_factory=dict)
    loops: dict[str, KinematicLoop] = field(default_factory=dict)
    loops_by_label: dict[str, KinematicLoop] = field(default_factory=dict)
    loop_branches: dict[str, LoopBranch] = field(default_factory=dict)
    closure_constraints: dict[str, ClosureConstraint] = field(default_factory=dict)
    link_tree: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_master(cls, data: dict[str, Any]) -> "OntologyRegistry":
        migrated = migrate_master_v1_to_v2(deepcopy(data))
        reg = cls(
            schema_version=str(migrated.get("schema_version", "2.0")),
            ontology_version=str(migrated.get("ontology_version", "1.0")),
            data_version=str(migrated.get("data_version", "")),
            link_tree=migrated.get("link_tree", {}) or {},
            raw=migrated,
        )
        for row in migrated.get("link_targets", []) or []:
            label = compact_link_name(str(row.get("target", "")))
            if not label:
                continue
            lat = normalize_laterality(row.get("side"))
            morph = normalize_morphology(row.get("morphology"))
            conf = normalize_mapping_confidence(row.get("mapping_confidence")) or MappingConfidence.MEDIUM
            eid = str(row.get("entity_id") or f"link:{laterality_to_label_prefix(lat)}:{label}")
            ent = LinkEntity(
                entity_id=eid,
                canonical_label=label,
                formal_label=row.get("expanded_link_name") or row.get("ros_link_name"),
                ros_label=row.get("ros_link_name"),
                legacy_canonical_id=row.get("canonical_id"),
                laterality=lat,
                morphology=morph,
                link_kind=LinkKind.PHYSICAL,
                status=EntityStatus.CANONICAL,
                parent_attachment=row.get("parent_attachment"),
                region=row.get("region"),
                segment_meaning=row.get("segment_meaning"),
                mapping_confidence=conf,
                notes=row.get("notes"),
                raw=row,
            )
            reg.links[eid] = ent
            reg.links_by_label[label] = ent

        for node_name, node in (migrated.get("link_tree") or {}).items():
            label = compact_link_name(str(node_name))
            kind_raw = (node or {}).get("link_kind")
            status_raw = (node or {}).get("status")
            if label in reg.links_by_label and not kind_raw:
                continue
            if kind_raw or status_raw or re.search(r"_(xr|yp|zy)$", label):
                lat = normalize_laterality(detect_side(label) or "c")
                try:
                    kind = LinkKind(str(kind_raw)) if kind_raw else LinkKind.AXIS_DECOMPOSITION
                except ValueError:
                    kind = LinkKind.VIRTUAL
                try:
                    status = EntityStatus(str(status_raw)) if status_raw else EntityStatus.PROVISIONAL
                except ValueError:
                    status = EntityStatus.PROVISIONAL
                eid = f"link:{laterality_to_label_prefix(lat)}:{label}"
                if label not in reg.links_by_label:
                    ent = LinkEntity(
                        entity_id=eid,
                        canonical_label=label,
                        laterality=lat,
                        link_kind=kind,
                        status=status,
                    )
                    reg.links[eid] = ent
                    reg.links_by_label[label] = ent
                else:
                    existing = reg.links_by_label[label]
                    existing.link_kind = kind
                    existing.status = status

        fj_groups: dict[tuple[str, str], list[str]] = {}
        for row in migrated.get("servo_targets", []) or []:
            label = str(row.get("target", "")).strip()
            if not label:
                continue
            lat = normalize_laterality(row.get("side"))
            morph = normalize_morphology(row.get("morphology"))
            landmark = str(row.get("landmark") or "").strip() or None
            axis_pair = row.get("axis_pair")
            axis_short = axis_short_from_pair(axis_pair) or axis_keyword_to_short(label) or "na"
            eid = str(row.get("entity_id") or f"dof:{laterality_to_label_prefix(lat or normalize_laterality(detect_side(label) or 'c'))}:{label}")
            status = EntityStatus.CANONICAL
            if row.get("entity_status"):
                try:
                    status = EntityStatus(str(row["entity_status"]))
                except ValueError:
                    pass
            elif row.get("mapping_confidence") == "proposed":
                status = EntityStatus.PROVISIONAL
            axis = AxisDefinition(
                vector=parse_axis_vector(row.get("axis_vector_reference_pose"))
                or _AXIS_PAIR_TO_VECTOR.get(str(axis_pair or "").lower()),
                expressed_in="legacy_assumed" if row.get("axis_vector_reference_pose") else "unknown",
                reference_pose=row.get("axis_reference_pose") or row.get("reference_pose"),
                axis_pair=axis_pair,
                positive_semantic_motion=row.get("positive_semantic_motion"),
            )
            sem = row.get("semantic_motion")
            dof = DegreeOfFreedom(
                entity_id=eid,
                canonical_label=label,
                formal_label=row.get("formal_joint_name"),
                expanded_label=row.get("expanded_joint_name"),
                ros_label=row.get("ros_joint_name"),
                legacy_canonical_id=row.get("canonical_id"),
                laterality=lat,
                morphology=morph,
                landmark=landmark,
                joint_type=normalize_joint_type(row.get("joint_type")),
                axis=axis,
                motion_class=motion_class_from_label(sem),
                semantic_motion_label=sem,
                parent_link=compact_link_name(str(row.get("parent_link") or "")) or None,
                child_link=compact_link_name(str(row.get("child_link") or "")) or None,
                status=status,
                mapping_confidence=normalize_mapping_confidence(row.get("mapping_confidence"))
                or MappingConfidence.MEDIUM,
                appendage_location=row.get("appendage_location"),
                chain_name=row.get("chain_name"),
                notes=row.get("notes"),
                raw=row,
            )
            reg.dofs[eid] = dof
            reg.dofs_by_label[label] = dof
            if lat and landmark:
                fj_groups.setdefault((laterality_to_label_prefix(lat), landmark), []).append(eid)

        for (side_p, landmark), dof_ids in fj_groups.items():
            fj_id = f"joint:{side_p}:{landmark}"
            label = f"{side_p}_{landmark}"
            lat = normalize_laterality(side_p)
            morph = reg.dofs[dof_ids[0]].morphology
            if fj_id in reg.functional_joints:
                reg.functional_joints[fj_id].dof_ids = sorted(
                    set(reg.functional_joints[fj_id].dof_ids) | set(dof_ids)
                )
            else:
                reg.functional_joints[fj_id] = FunctionalJoint(
                    entity_id=fj_id,
                    canonical_label=label,
                    laterality=lat,
                    morphology=morph,
                    landmark=landmark,
                    dof_ids=list(dof_ids),
                )
            for did in dof_ids:
                reg.dofs[did].dof_of = fj_id

        # Kinematic joints (explicit + synthesized from tree/servo). Passive joints live here, not in servo_targets.
        def _register_kjoint(kj: KinematicJoint) -> None:
            if kj.canonical_label in reg.kinematic_joints_by_label:
                existing = reg.kinematic_joints_by_label[kj.canonical_label]
                # Prefer richer / explicit rows
                if existing.raw.get("_synthetic") and not kj.raw.get("_synthetic"):
                    del reg.kinematic_joints[existing.entity_id]
                else:
                    return
            reg.kinematic_joints[kj.entity_id] = kj
            reg.kinematic_joints_by_label[kj.canonical_label] = kj

        for row in migrated.get("kinematic_joints", []) or []:
            label = str(row.get("target") or "").strip()
            parent = compact_link_name(str(row.get("parent_link") or ""))
            child = compact_link_name(str(row.get("child_link") or ""))
            if not label or not parent or not child:
                continue
            lat = normalize_laterality(row.get("side")) or normalize_laterality(detect_side(label) or "c")
            eid = str(row.get("entity_id") or f"kjoint:{laterality_to_label_prefix(lat)}:{label}")
            role = normalize_actuation_role(row.get("actuation_role")) or ActuationRole.UNKNOWN
            dof_ids = [str(x) for x in (row.get("dof_ids") or row.get("dofs") or []) if x]
            try:
                status = EntityStatus(str(row.get("status") or EntityStatus.CANONICAL.value))
            except ValueError:
                status = EntityStatus.CANONICAL
            kj = KinematicJoint(
                entity_id=eid,
                canonical_label=label,
                parent_link=parent,
                child_link=child,
                joint_type=normalize_joint_type(row.get("joint_type")),
                actuation_role=role,
                dof_ids=dof_ids,
                laterality=lat,
                morphology=normalize_morphology(row.get("morphology")),
                status=status,
                mapping_confidence=normalize_mapping_confidence(row.get("mapping_confidence"))
                or MappingConfidence.MEDIUM,
                realizes_functional_joint_id=row.get("realizes_functional_joint")
                or row.get("realizes_functional_joint_id"),
                notes=row.get("notes"),
                raw=row,
            )
            _register_kjoint(kj)

        # Synthesize from link_tree via_joint edges (actuated if DOF exists, else unknown/passive-ish provisional)
        for pname, node in (migrated.get("link_tree") or {}).items():
            parent = compact_link_name(str(pname))
            for ch in (node or {}).get("children", []) or []:
                child = compact_link_name(str(ch.get("link") or ""))
                via = str(ch.get("via_joint") or "").strip()
                if not child or not via:
                    continue
                if via in reg.kinematic_joints_by_label:
                    continue
                lat = normalize_laterality(detect_side(via) or detect_side(parent) or "c")
                role = ActuationRole.ACTUATED if via in reg.dofs_by_label else ActuationRole.UNKNOWN
                dof_ids = [via] if via in reg.dofs_by_label else []
                jt = reg.dofs_by_label[via].joint_type if via in reg.dofs_by_label else None
                kj = KinematicJoint(
                    entity_id=f"kjoint:{laterality_to_label_prefix(lat)}:{via}",
                    canonical_label=via,
                    parent_link=parent,
                    child_link=child,
                    joint_type=jt,
                    actuation_role=role,
                    dof_ids=dof_ids,
                    laterality=lat,
                    status=EntityStatus.CANONICAL if dof_ids else EntityStatus.PROVISIONAL,
                    mapping_confidence=MappingConfidence.MEDIUM if dof_ids else MappingConfidence.LOW,
                    realizes_functional_joint_id=(
                        reg.dofs_by_label[via].dof_of if via in reg.dofs_by_label else None
                    ),
                    notes="Synthesized from link_tree via_joint",
                    raw={"_synthetic": True, "source": "link_tree"},
                )
                _register_kjoint(kj)

        # Synthesize from servo_targets parent/child when not already present
        for dof in reg.dofs.values():
            if not dof.parent_link or not dof.child_link:
                continue
            label = dof.canonical_label
            if label in reg.kinematic_joints_by_label:
                kj = reg.kinematic_joints_by_label[label]
                if label not in kj.dof_ids:
                    kj.dof_ids.append(label)
                if kj.actuation_role == ActuationRole.UNKNOWN:
                    kj.actuation_role = ActuationRole.ACTUATED
                continue
            lat = dof.laterality or normalize_laterality(detect_side(label) or "c")
            kj = KinematicJoint(
                entity_id=f"kjoint:{laterality_to_label_prefix(lat)}:{label}",
                canonical_label=label,
                parent_link=dof.parent_link,
                child_link=dof.child_link,
                joint_type=dof.joint_type,
                actuation_role=ActuationRole.ACTUATED,
                dof_ids=[label],
                laterality=lat,
                morphology=dof.morphology,
                status=dof.status,
                mapping_confidence=dof.mapping_confidence,
                realizes_functional_joint_id=dof.dof_of,
                notes="Synthesized from servo_targets DOF endpoints",
                raw={"_synthetic": True, "source": "servo_targets"},
            )
            _register_kjoint(kj)

        # Closed-loop ontology (never stored as link_tree cycles)
        for row in migrated.get("loop_targets", []) or []:
            label = str(row.get("target") or "").strip()
            if not label:
                continue
            lat = normalize_laterality(row.get("side"))
            morph = normalize_morphology(row.get("morphology"))
            mech = normalize_loop_mechanism_type(row.get("mechanism_type")) or LoopMechanismType.UNKNOWN
            conf = normalize_mapping_confidence(row.get("mapping_confidence")) or MappingConfidence.MEDIUM
            try:
                status = EntityStatus(str(row.get("status") or EntityStatus.CANONICAL.value))
            except ValueError:
                status = EntityStatus.CANONICAL
            eid = str(row.get("entity_id") or f"loop:{laterality_to_label_prefix(lat)}:{label}")
            branch_ids: list[str] = []
            closure_ids: list[str] = []
            member_links: set[str] = set()
            member_dofs: set[str] = set()
            member_joints: set[str] = set()

            for bi, br in enumerate(row.get("branches") or [], start=1):
                if not isinstance(br, dict):
                    continue
                blabel = str(br.get("id") or br.get("target") or make_loop_branch_label(label, bi))
                links = [compact_link_name(str(x)) for x in (br.get("links") or []) if x]
                joints = [str(x) for x in (br.get("joints") or []) if x]
                bid = str(br.get("entity_id") or f"loop_branch:{blabel}")
                branch = LoopBranch(
                    entity_id=bid,
                    canonical_label=blabel,
                    loop_id=eid,
                    link_path=links,
                    joint_path=joints,
                    notes=br.get("notes"),
                    raw=br,
                )
                reg.loop_branches[bid] = branch
                branch_ids.append(bid)
                member_links.update(links)
                member_joints.update(joints)
                for jn in joints:
                    kj = reg.kinematic_joints_by_label.get(jn)
                    if kj:
                        member_dofs.update(kj.dof_ids)
                    elif jn in reg.dofs_by_label:
                        member_dofs.add(jn)

            for ci, cl in enumerate(row.get("closures") or [], start=1):
                if not isinstance(cl, dict):
                    continue
                clabel = str(cl.get("id") or cl.get("target") or (
                    make_loop_closure_label(label) if len(row.get("closures") or []) == 1
                    else make_loop_closure_label(label, ci)
                ))
                cid = str(cl.get("entity_id") or f"closure:{clabel}")
                fl = compact_link_name(str(cl.get("from_link") or ""))
                tl = compact_link_name(str(cl.get("to_link") or ""))
                rbj = cl.get("represented_by_joint")
                if rbj is None:
                    rbj = cl.get("via_joint")  # legacy alias
                rbj = str(rbj) if rbj else None
                ctype = normalize_closure_constraint_type(cl.get("constraint_type") or "loop_closure")
                if ctype is None:
                    ctype = ClosureConstraintType.UNKNOWN
                try:
                    cstatus = EntityStatus(str(cl.get("status") or status.value))
                except ValueError:
                    cstatus = status
                closure = ClosureConstraint(
                    entity_id=cid,
                    canonical_label=clabel,
                    loop_id=eid,
                    from_link=fl,
                    to_link=tl,
                    constraint_type=ctype,
                    represented_by_joint=rbj,
                    status=cstatus,
                    mapping_confidence=normalize_mapping_confidence(cl.get("mapping_confidence")) or conf,
                    notes=cl.get("notes"),
                    raw=cl,
                )
                reg.closure_constraints[cid] = closure
                closure_ids.append(cid)
                if fl:
                    member_links.add(fl)
                if tl:
                    member_links.add(tl)
                if rbj:
                    member_joints.add(rbj)
                    kj = reg.kinematic_joints_by_label.get(rbj)
                    if kj:
                        member_dofs.update(kj.dof_ids)
                    elif rbj in reg.dofs_by_label:
                        member_dofs.add(rbj)

            for x in row.get("member_links") or []:
                member_links.add(compact_link_name(str(x)))
            for x in row.get("member_dofs") or []:
                member_dofs.add(str(x))
            for x in row.get("member_joints") or []:
                member_joints.add(str(x))

            loop = KinematicLoop(
                entity_id=eid,
                canonical_label=label,
                laterality=lat,
                landmark=str(row.get("landmark") or "").strip() or None,
                morphology=morph,
                mechanism_type=mech,
                branch_ids=branch_ids,
                closure_ids=closure_ids,
                member_link_ids=sorted(member_links),
                member_dof_ids=sorted(member_dofs),
                member_joint_ids=sorted(member_joints),
                status=status,
                mapping_confidence=conf,
                notes=row.get("notes"),
                raw=row,
            )
            reg.loops[eid] = loop
            reg.loops_by_label[label] = loop

        # Aliases LAST so loop / joint / link entity_ids resolve correctly
        for alias_key, entries in (migrated.get("alias_index") or {}).items():
            for entry in entries or []:
                target = str(entry.get("target") or "")
                ent = normalize_entity_type(entry.get("entity", "joint"))
                if ent == EntityType.LINK.value:
                    target = compact_link_name(target)
                tid = None
                if ent == EntityType.JOINT.value and target in reg.dofs_by_label:
                    tid = reg.dofs_by_label[target].entity_id
                elif ent == EntityType.LINK.value and target in reg.links_by_label:
                    tid = reg.links_by_label[target].entity_id
                elif ent == EntityType.LOOP.value and target in reg.loops_by_label:
                    tid = reg.loops_by_label[target].entity_id
                conf = (
                    normalize_mapping_confidence(entry.get("confidence"))
                    or MappingConfidence.MEDIUM
                )
                assertion = AliasAssertion(
                    alias=str(alias_key),
                    target_entity_id=tid,
                    target_label=target,
                    entity_type=ent,
                    alias_type=str(entry.get("mapping_type") or "alias"),
                    source=str(entry.get("source") or ""),
                    confidence=conf,
                    profile=entry.get("profile"),
                    notes=entry.get("notes"),
                )
                reg.aliases.append(assertion)
                reg.alias_by_key.setdefault(str(alias_key), []).append(assertion)
        return reg

    def get_loop(self, name: str) -> KinematicLoop | None:
        key = str(name or "").strip()
        return self.loops_by_label.get(key) or self.loops.get(key)

    def get_kinematic_joint(self, name: str) -> KinematicJoint | None:
        key = str(name or "").strip()
        return self.kinematic_joints_by_label.get(key) or self.kinematic_joints.get(key)

    def kinematic_joints_between(self, a: str, b: str) -> list[KinematicJoint]:
        la, lb = compact_link_name(a), compact_link_name(b)
        out: list[KinematicJoint] = []
        for kj in self.kinematic_joints.values():
            ends = {kj.parent_link, kj.child_link}
            if ends == {la, lb}:
                out.append(kj)
        return out

    def loops_for_link(self, link: str) -> list[KinematicLoop]:
        label = compact_link_name(link)
        return [lp for lp in self.loops.values() if label in lp.member_link_ids]

    def loops_for_joint(self, joint: str) -> list[KinematicLoop]:
        j = str(joint or "").strip()
        return [
            lp
            for lp in self.loops.values()
            if j in lp.member_dof_ids or j in lp.member_joint_ids
        ]

    def branches_for_loop(self, loop: str) -> list[LoopBranch]:
        lp = self.get_loop(loop)
        if not lp:
            return []
        return [self.loop_branches[i] for i in lp.branch_ids if i in self.loop_branches]

    def closures_for_loop(self, loop: str) -> list[ClosureConstraint]:
        lp = self.get_loop(loop)
        if not lp:
            return []
        return [self.closure_constraints[i] for i in lp.closure_ids if i in self.closure_constraints]

    def parent_of(self, link: str) -> str | None:
        node = self.link_tree.get(compact_link_name(link)) or self.link_tree.get(link)
        if not node:
            return None
        parent = node.get("parent")
        return compact_link_name(parent) if parent else None

    def children_of(self, link: str) -> list[str]:
        node = self.link_tree.get(compact_link_name(link)) or self.link_tree.get(link)
        if not node:
            return []
        return [
            compact_link_name(str(c.get("link")))
            for c in node.get("children", [])
            if c.get("link")
        ]

    def connected_via(self, parent: str, child: str) -> str | None:
        return _link_tree_hop_joint(
            self.link_tree, compact_link_name(parent), compact_link_name(child)
        )


# ===========================================================================
# ontology migration
# ===========================================================================


def _build_dof_alias_index(servo_targets: list[dict[str, Any]]) -> dict[str, str]:
    """Map formal/expanded/ros/canonical/target variants -> canonical DOF label."""
    idx: dict[str, str] = {}
    for row in servo_targets or []:
        target = str(row.get("target") or "").strip()
        if not target:
            continue
        keys = {
            target,
            compact_link_name(target),
            str(row.get("canonical_id") or ""),
            str(row.get("formal_joint_name") or ""),
            str(row.get("expanded_joint_name") or ""),
            str(row.get("ros_joint_name") or ""),
        }
        for key in list(keys):
            if not key:
                continue
            keys.add(key.removesuffix("_joint"))
        cleaned: set[str] = set()
        for key in keys:
            if not key:
                continue
            cleaned.add(key)
            cleaned.add(key.removesuffix("_joint"))
            k2 = (
                key.replace("_ypitch", "_yp")
                .replace("_xroll", "_xr")
                .replace("_zyaw", "_zy")
                .replace("_pitch", "_yp")
                .replace("_roll", "_xr")
                .replace("_yaw", "_zy")
            )
            cleaned.add(k2)
            cleaned.add(k2.removesuffix("_joint"))
        for key in cleaned:
            if key and key not in idx:
                idx[key] = target
    return idx


def _resolve_attachment_or_joint_ref(
    name: str,
    *,
    link_names: set[str],
    dof_index: dict[str, str],
    dof_targets: set[str],
) -> str | None:
    """Resolve legacy parent_attachment / joint-ish refs to a known link or DOF label."""
    if not name:
        return None
    raw = str(name).strip()
    cands: list[str] = [raw, compact_link_name(raw)]
    if raw.startswith("left_"):
        cands.append("l_" + raw[len("left_") :])
    elif raw.startswith("right_"):
        cands.append("r_" + raw[len("right_") :])
    expanded: list[str] = []
    for c in cands:
        expanded.append(c)
        expanded.append(c.removesuffix("_joint"))
        c2 = (
            c.replace("_ypitch", "_yp")
            .replace("_xroll", "_xr")
            .replace("_zyaw", "_zy")
            .replace("_pitch", "_yp")
            .replace("_roll", "_xr")
            .replace("_yaw", "_zy")
        )
        expanded.append(c2)
        expanded.append(c2.removesuffix("_joint"))
    # Prefer DOF matches (attachments often name the connecting joint)
    for c in expanded:
        if c in dof_index:
            return dof_index[c]
        if c in dof_targets:
            return c
    for c in expanded:
        cc = compact_link_name(c)
        if cc in link_names:
            return cc
        if c in link_names:
            return c
    # Unique DOF prefix: l_alula -> l_alula_yp when only one DOF shares prefix
    for c in expanded:
        base = compact_link_name(c.removesuffix("_joint"))
        if not base:
            continue
        hits = sorted(t for t in dof_targets if t == base or t.startswith(base + "_"))
        if len(hits) == 1:
            return hits[0]
    return None


def _rewrite_side_prefix_to_canonical(name: str, known: set[str]) -> str | None:
    if not name:
        return None
    n = str(name)
    candidates = [n]
    if n.startswith("left_"):
        candidates.append("l_" + n[len("left_") :])
    elif n.startswith("right_"):
        candidates.append("r_" + n[len("right_") :])
    if n.startswith("_tbd_"):
        candidates.append(n[len("_tbd_") :])
    for cand in candidates:
        compact = compact_link_name(cand)
        if cand in known or compact in known:
            return compact if compact in known else cand
    return None


def migrate_master_v1_to_v2(
    data: dict[str, Any],
    *,
    report: MigrationReport | None = None,
) -> dict[str, Any]:
    """Normalize Master dict in memory for ontology v2 (caller should deepcopy if needed)."""
    out = data
    # 2.1 additive: closed-loop ontology (loop_targets) while keeping link_tree acyclic.
    out.setdefault("schema_version", "2.2")
    out.setdefault("ontology_version", "1.2")
    out.setdefault("data_version", out.get("data_version") or "2026.08.09")
    out.setdefault("loop_targets", [])
    out.setdefault("kinematic_joints", [])

    link_names: set[str] = set()
    for row in out.get("link_targets", []) or []:
        t = compact_link_name(str(row.get("target", "")))
        if t:
            link_names.add(t)
    for node_name in out.get("link_tree") or {}:
        link_names.add(compact_link_name(str(node_name)))

    for row in out.get("servo_targets", []) or []:
        lat = normalize_laterality(row.get("side"))
        if lat:
            row["side"] = lat.value
        morph = normalize_morphology(row.get("morphology"))
        if morph:
            row["morphology"] = morph.value
        landmark = str(row.get("landmark") or "").strip() or None
        axis_short = (
            axis_short_from_pair(row.get("axis_pair"))
            or axis_keyword_to_short(str(row.get("target", "")))
            or "na"
        )
        # Stable unique id from serialization label (not landmark alone — landmarks collide across sides/morphologies).
        target_label = str(row.get("target") or "").strip()
        side_prefix = laterality_to_label_prefix(lat or normalize_laterality(detect_side(target_label) or "c"))
        new_eid = f"dof:{side_prefix}:{target_label}"
        old_eid = row.get("entity_id")
        if report is not None and old_eid and str(old_eid) != new_eid:
            report.add(
                MigrationIssue(
                    "entity_id_rewritten",
                    f"Rewrote DOF entity_id {old_eid!r} -> {new_eid!r}",
                    f"servo_targets[{target_label}]",
                    before=old_eid,
                    after=new_eid,
                )
            )
        row["entity_id"] = new_eid
        if not row.get("motion_class"):
            row["motion_class"] = motion_class_from_label(row.get("semantic_motion")).value
        for field_name in ("parent_link", "child_link"):
            raw = row.get(field_name)
            if not raw:
                continue
            fixed = _rewrite_side_prefix_to_canonical(str(raw), link_names)
            if fixed:
                row[field_name] = fixed
            elif str(raw).startswith("_tbd_"):
                stripped = str(raw)[len("_tbd_") :]
                if stripped in link_names or compact_link_name(stripped) in link_names:
                    row[field_name] = compact_link_name(stripped)

    dof_targets = {str(r.get("target")) for r in out.get("servo_targets", []) or [] if r.get("target")}
    dof_index = _build_dof_alias_index(out.get("servo_targets", []) or [])

    for row in out.get("link_targets", []) or []:
        lat = normalize_laterality(row.get("side"))
        if lat:
            row["side"] = lat.value
        morph = normalize_morphology(row.get("morphology"))
        if morph:
            row["morphology"] = morph.value
        label = compact_link_name(str(row.get("target", "")))
        new_leid = f"link:{laterality_to_label_prefix(lat)}:{label}"
        old_leid = row.get("entity_id")
        if report is not None and old_leid and str(old_leid) != new_leid:
            report.add(
                MigrationIssue(
                    "entity_id_rewritten",
                    f"Rewrote link entity_id {old_leid!r} -> {new_leid!r}",
                    f"link_targets[{label}]",
                    before=old_leid,
                    after=new_leid,
                )
            )
        row["entity_id"] = new_leid
        pa = row.get("parent_attachment")
        if pa:
            resolved = _resolve_attachment_or_joint_ref(
                str(pa),
                link_names=link_names,
                dof_index=dof_index,
                dof_targets=dof_targets,
            )
            if resolved:
                row["parent_attachment"] = resolved
            else:
                fixed = _rewrite_side_prefix_to_canonical(str(pa), link_names)
                if fixed:
                    row["parent_attachment"] = fixed

    for node_name, node in (out.get("link_tree") or {}).items():
        if not isinstance(node, dict):
            continue
        if re.search(r"_(xr|yp|zy)$", str(node_name)):
            node.setdefault("link_kind", LinkKind.AXIS_DECOMPOSITION.value)
            node.setdefault("status", EntityStatus.PROVISIONAL.value)
            node.setdefault("entity_type", "virtual_link")
    return out


def apply_mechanical_master_fixes(data: dict[str, Any]) -> dict[str, Any]:
    """Persistable mechanical fixes for dangling refs / provisional via_joints."""
    out = migrate_master_v1_to_v2(deepcopy(data))
    out["schema_version"] = "2.2"
    out["ontology_version"] = "1.2"
    out["data_version"] = "2026.08.09"
    out.setdefault("loop_targets", [])
    out.setdefault("kinematic_joints", [])

    link_names = {compact_link_name(str(r.get("target", ""))) for r in out.get("link_targets", [])}
    link_names |= {compact_link_name(k) for k in (out.get("link_tree") or {})}
    servo_by = {str(r["target"]): r for r in out.get("servo_targets", []) if r.get("target")}

    for row in out.get("servo_targets", []):
        for field_name in ("parent_link", "child_link"):
            raw = row.get(field_name)
            if not raw:
                continue
            fixed = _rewrite_side_prefix_to_canonical(str(raw), link_names)
            if fixed:
                row[field_name] = fixed
            else:
                # Normalize left_/right_ even when target is a virtual intermediate not yet registered.
                n = str(raw)
                if n.startswith("left_"):
                    row[field_name] = "l_" + n[len("left_"):]
                elif n.startswith("right_"):
                    row[field_name] = "r_" + n[len("right_"):]
                elif n.startswith("_tbd_"):
                    row[field_name] = n[len("_tbd_"):]

    # Register unresolved axis-decomposition intermediates as provisional virtual tree nodes.
    _AXISISH = re.compile(
        r"_(xr|yp|zy|xroll|ypitch|zyaw|roll|pitch|yaw)$",
        re.I,
    )
    tree = out.setdefault("link_tree", {})
    for row in out.get("servo_targets", []):
        for field_name in ("parent_link", "child_link"):
            ref = compact_link_name(str(row.get(field_name) or ""))
            if not ref or ref in link_names:
                continue
            if not _AXISISH.search(ref):
                continue
            # attach under nearest known physical ancestor inferred from shared prefix if possible
            parent_guess = None
            # e.g. l_arm_lower_roll -> prefer l_arm_lower if present, else l_elbow chain from row
            bare = _AXISISH.sub("", ref)
            if bare in link_names:
                parent_guess = bare
            elif field_name == "child_link":
                parent_guess = compact_link_name(str(row.get("parent_link") or "")) or None
                if parent_guess and parent_guess not in link_names and parent_guess not in tree:
                    parent_guess = None
            node = tree.setdefault(
                ref,
                {
                    "parent": parent_guess,
                    "children": [],
                    "link_kind": LinkKind.AXIS_DECOMPOSITION.value,
                    "status": EntityStatus.PROVISIONAL.value,
                    "entity_type": "virtual_link",
                },
            )
            node.setdefault("link_kind", LinkKind.AXIS_DECOMPOSITION.value)
            node.setdefault("status", EntityStatus.PROVISIONAL.value)
            node.setdefault("entity_type", "virtual_link")
            if parent_guess and not node.get("parent"):
                node["parent"] = parent_guess
            if parent_guess and parent_guess in tree:
                children = tree[parent_guess].setdefault("children", [])
                if not any(c.get("link") == ref for c in children):
                    # via_joint unknown — leave empty rather than invent
                    children.append({"link": ref, "via_joint": row.get("target")})
            link_names.add(ref)

    AXIS_VEC = {"xr": "[1,0,0]", "yp": "[0,1,0]", "zy": "[0,0,1]"}
    AXIS_PAIR = {"xr": "xroll", "yp": "ypitch", "zy": "zyaw"}
    for parent, node in (out.get("link_tree") or {}).items():
        for ch in node.get("children", []) or []:
            child = ch.get("link")
            via = ch.get("via_joint")
            if not via or via in servo_by or not child:
                continue
            m = re.search(r"_(xr|yp|zy)$", str(via))
            if not m:
                continue
            axis_short = m.group(1)
            side_tok = detect_side(via) or detect_side(child) or "c"
            lat = normalize_laterality(side_tok)
            body = re.sub(r"^(l|r|c)_", "", str(via))
            body = re.sub(r"_(xr|yp|zy)$", "", body)
            morph = Morphology.HUMANOID
            for cand in (child, parent):
                for lr in out.get("link_targets", []):
                    if compact_link_name(str(lr.get("target", ""))) == compact_link_name(str(cand)):
                        mm = normalize_morphology(lr.get("morphology"))
                        if mm:
                            morph = mm
                        break
            stub = {
                "target": via,
                "entity_id": f"dof:{laterality_to_label_prefix(lat)}:{via}",
                "canonical_id": via,
                "formal_joint_name": f"{via}_joint",
                "expanded_joint_name": f"{via}_joint",
                "ros_joint_name": f"{via}_joint",
                "axis_pair": AXIS_PAIR[axis_short],
                "axis_vector_reference_pose": AXIS_VEC[axis_short],
                "semantic_motion": "unknown",
                "motion_class": MotionClass.UNKNOWN.value,
                "positive_semantic_motion": "profile_defined",
                "reference_pose": "profile_defined",
                "axis_reference_pose": "profile_defined",
                "side": lat.value if lat else "center",
                "morphology": morph.value,
                "appendage_location": "profile_defined",
                "landmark": body,
                "chain_name": "none",
                "chain_index": None,
                "chain_count": None,
                "parent_link": compact_link_name(str(parent)),
                "child_link": compact_link_name(str(child)),
                "joint_type": "revolute",
                "actuation_role": "profile_defined",
                "mapping_confidence": "proposed",
                "entity_status": EntityStatus.PROVISIONAL.value,
                "compatibility_aliases": [],
                "functional_aliases": [],
                "singleton_chain_rule": None,
                "notes": "Provisional DOF registered from link_tree via_joint edge (topology-evidenced).",
            }
            out.setdefault("servo_targets", []).append(stub)
            servo_by[via] = stub
            out.setdefault("alias_index", {}).setdefault(via, []).append(
                {
                    "target": via,
                    "entity": "joint",
                    "confidence": "proposed",
                    "mapping_type": "topology_alias",
                    "source": "link_tree_via_joint_stub",
                    "notes": "Provisional topology-evidenced DOF",
                }
            )

    # Normalize link parent_attachment to canonical link/DOF labels (no anatomy invention).
    link_names = {compact_link_name(str(r.get("target", ""))) for r in out.get("link_targets", [])}
    link_names |= {compact_link_name(k) for k in (out.get("link_tree") or {})}
    servo_by = {str(r["target"]): r for r in out.get("servo_targets", []) if r.get("target")}
    dof_index = _build_dof_alias_index(out.get("servo_targets", []) or [])
    dof_targets = set(servo_by)
    for row in out.get("link_targets", []) or []:
        pa = row.get("parent_attachment")
        if not pa:
            continue
        resolved = _resolve_attachment_or_joint_ref(
            str(pa),
            link_names=link_names,
            dof_index=dof_index,
            dof_targets=dof_targets,
        )
        if resolved:
            row["parent_attachment"] = resolved

    return migrate_master_v1_to_v2(out)


# ===========================================================================
# ontology validation
# ===========================================================================

def _known_link_labels(data: dict[str, Any]) -> set[str]:
    names: set[str] = {"base_link", "c_base_link"}
    for row in data.get("link_targets", []) or []:
        t = compact_link_name(str(row.get("target", "")))
        if t:
            names.add(t)
    for node_name in data.get("link_tree") or {}:
        names.add(compact_link_name(str(node_name)))
    return names


def _known_dof_labels(data: dict[str, Any]) -> set[str]:
    return {str(r.get("target")) for r in data.get("servo_targets", []) or [] if r.get("target")}


def _detect_link_tree_cycles(tree: dict[str, Any]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def dfs(node: str, stack: list[str]) -> None:
        if node in done:
            return
        if node in visiting:
            if node in stack:
                cycles.append(stack[stack.index(node) :] + [node])
            return
        visiting.add(node)
        stack.append(node)
        for ch in (tree.get(node) or {}).get("children", []) or []:
            cl = ch.get("link")
            if cl:
                dfs(str(cl), stack)
        stack.pop()
        visiting.remove(node)
        done.add(node)

    for name in tree:
        dfs(str(name), [])
    return cycles


def _expected_structured_entity_id(kind: str, side: Any, label: str) -> str | None:
    lat = normalize_laterality(side)
    if lat is None:
        lat = normalize_laterality(detect_side(label) or None)
    if lat is None or not label:
        return None
    return f"{kind}:{laterality_to_label_prefix(lat)}:{label}"


def _check_entity_id_integrity(
    report: ValidationReport,
    *,
    kind: str,
    label: str,
    side: Any,
    entity_id: Any,
    path: str,
) -> None:
    eid = str(entity_id or "").strip()
    if not eid:
        report.add(
            ValidationIssue(
                ValidationSeverity.ERROR,
                "missing_entity_id",
                f"Missing entity_id for '{label}'",
                path,
            )
        )
        return
    parts = eid.split(":")
    if len(parts) < 3:
        report.add(
            ValidationIssue(
                ValidationSeverity.ERROR,
                "malformed_entity_id",
                f"Malformed entity_id '{eid}'",
                path,
            )
        )
        return
    typ, side_p, rest = parts[0], parts[1], ":".join(parts[2:])
    if typ != kind:
        report.add(
            ValidationIssue(
                ValidationSeverity.ERROR,
                "entity_id_type_mismatch",
                f"entity_id type '{typ}' != expected '{kind}' for '{label}'",
                path,
            )
        )
    if rest != label:
        report.add(
            ValidationIssue(
                ValidationSeverity.ERROR,
                "entity_id_label_mismatch",
                f"entity_id label '{rest}' != target '{label}'",
                path,
            )
        )
    expected = _expected_structured_entity_id(kind, side, label)
    if expected and eid != expected:
        report.add(
            ValidationIssue(
                ValidationSeverity.ERROR,
                "entity_id_side_mismatch",
                f"entity_id '{eid}' inconsistent with side/target (expected '{expected}')",
                path,
            )
        )


def find_undirected_tree_path(tree: dict[str, Any], start: str, goal: str) -> list[str] | None:
    """BFS path on the undirected spanning tree (parent/child both traversable)."""
    a = compact_link_name(start)
    b = compact_link_name(goal)
    if not a or not b:
        return None
    if a == b:
        return [a]
    adj: dict[str, set[str]] = {}
    for name, node in (tree or {}).items():
        u = compact_link_name(str(name))
        adj.setdefault(u, set())
        parent = (node or {}).get("parent")
        if parent:
            v = compact_link_name(str(parent))
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        for ch in (node or {}).get("children", []) or []:
            v = compact_link_name(str(ch.get("link") or ""))
            if not v:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    if a not in adj or b not in adj:
        return None
    from collections import deque

    q: deque[list[str]] = deque([[a]])
    seen = {a}
    while q:
        path = q.popleft()
        cur = path[-1]
        if cur == b:
            return path
        for nxt in adj.get(cur, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(path + [nxt])
    return None


def _tree_direct_neighbors(tree: dict[str, Any], a: str, b: str) -> bool:
    la, lb = compact_link_name(a), compact_link_name(b)
    na = tree.get(la) or {}
    nb = tree.get(lb) or {}
    if compact_link_name(str(na.get("parent") or "")) == lb:
        return True
    if compact_link_name(str(nb.get("parent") or "")) == la:
        return True
    for ch in na.get("children", []) or []:
        if compact_link_name(str(ch.get("link") or "")) == lb:
            return True
    for ch in nb.get("children", []) or []:
        if compact_link_name(str(ch.get("link") or "")) == la:
            return True
    return False


def _tree_edge_via_joint(tree: dict[str, Any], a: str, b: str) -> str | None:
    la, lb = compact_link_name(a), compact_link_name(b)
    for src, dst in ((la, lb), (lb, la)):
        node = tree.get(src) or {}
        for ch in node.get("children", []) or []:
            if compact_link_name(str(ch.get("link") or "")) == dst:
                vj = ch.get("via_joint")
                return str(vj) if vj else None
    return None


def _known_kjoint_labels(data: dict[str, Any]) -> set[str]:
    names = {str(r.get("target")) for r in (data.get("kinematic_joints") or []) if r.get("target")}
    # tree via_joints and servo labels also act as kinematic joint identities after synthesis
    for node in (data.get("link_tree") or {}).values():
        for ch in (node or {}).get("children", []) or []:
            vj = ch.get("via_joint")
            if vj:
                names.add(str(vj))
    names |= {str(r.get("target")) for r in (data.get("servo_targets") or []) if r.get("target")}
    return names


def validate_raw_master(data: dict[str, Any] | None = None) -> ValidationReport:
    """Validate on-disk / pre-migration Master without rewriting entity_ids."""
    raw = deepcopy(data) if data is not None else deepcopy(load_master())
    report = ValidationReport()
    report.stats["validation_stage"] = "raw"

    for row in raw.get("servo_targets", []) or []:
        t = str(row.get("target") or "")
        _check_entity_id_integrity(
            report,
            kind="dof",
            label=t,
            side=row.get("side"),
            entity_id=row.get("entity_id"),
            path=f"servo_targets[{t}].entity_id",
        )
    for row in raw.get("link_targets", []) or []:
        t = compact_link_name(str(row.get("target") or ""))
        _check_entity_id_integrity(
            report,
            kind="link",
            label=t,
            side=row.get("side"),
            entity_id=row.get("entity_id"),
            path=f"link_targets[{t}].entity_id",
        )
    for row in raw.get("loop_targets", []) or []:
        t = str(row.get("target") or "")
        if row.get("entity_id"):
            _check_entity_id_integrity(
                report,
                kind="loop",
                label=t,
                side=row.get("side"),
                entity_id=row.get("entity_id"),
                path=f"loop_targets[{t}].entity_id",
            )
    for row in raw.get("kinematic_joints", []) or []:
        t = str(row.get("target") or "")
        if row.get("entity_id"):
            _check_entity_id_integrity(
                report,
                kind="kjoint",
                label=t,
                side=row.get("side") or detect_side(t),
                entity_id=row.get("entity_id"),
                path=f"kinematic_joints[{t}].entity_id",
            )
    report.stats["entity_id_error_count"] = sum(
        1 for i in report.errors if "entity_id" in i.code
    )
    return report


def _validate_loop_ontology(migrated: dict[str, Any], report: ValidationReport) -> None:
    """Strict closed-loop validation: direct edges + closure path membership (not mere connectivity)."""
    link_names = _known_link_labels(migrated)
    dof_names = _known_dof_labels(migrated)
    kjoint_names = _known_kjoint_labels(migrated)
    tree = migrated.get("link_tree") or {}

    loop_rows = migrated.get("loop_targets") or []
    seen_loop_t: dict[str, str] = {}
    seen_loop_eid: dict[str, str] = {}
    seen_branch: dict[str, str] = {}
    seen_closure: dict[str, str] = {}
    branch_count = 0
    closure_count = 0

    for row in loop_rows:
        label = str(row.get("target") or "").strip()
        path = f"loop_targets[{label or '?'}]"
        if not label:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_loop_name", "Empty loop target", path))
            continue
        if not is_valid_loop_label(label):
            report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_loop_name", f"Invalid loop canonical name '{label}'", path))
        if label in seen_loop_t:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_loop_target", f"Duplicate loop target '{label}'", path))
        seen_loop_t[label] = label
        eid = str(row.get("entity_id") or f"loop:{label}")
        if eid in seen_loop_eid:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_loop_entity_id", f"Duplicate loop entity_id '{eid}'", path))
        seen_loop_eid[eid] = label
        if row.get("entity_id"):
            _check_entity_id_integrity(
                report, kind="loop", label=label, side=row.get("side"), entity_id=row.get("entity_id"), path=f"{path}.entity_id"
            )

        if normalize_laterality(row.get("side")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_side", f"Invalid loop side {row.get('side')!r}", f"{path}.side"))
        if row.get("morphology") is not None and normalize_morphology(row.get("morphology")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_morphology", f"Invalid loop morphology {row.get('morphology')!r}", f"{path}.morphology"))
        if normalize_loop_mechanism_type(row.get("mechanism_type")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_loop_mechanism_type", f"Invalid mechanism_type {row.get('mechanism_type')!r}", f"{path}.mechanism_type"))
        if normalize_mapping_confidence(row.get("mapping_confidence") or "high") is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_confidence", f"Invalid mapping_confidence {row.get('mapping_confidence')!r}", f"{path}.mapping_confidence"))

        branches = row.get("branches") or []
        closures = row.get("closures") or []
        if not branches:
            report.add(ValidationIssue(ValidationSeverity.WARNING, "orphan_loop_branch", f"Loop '{label}' has no branches", path))
        if not closures:
            report.add(ValidationIssue(ValidationSeverity.WARNING, "orphan_closure", f"Loop '{label}' has no closures", path))

        closure_edges: set[tuple[str, str]] = set()
        declared_links: set[str] = set()

        for bi, br in enumerate(branches, start=1):
            if not isinstance(br, dict):
                continue
            branch_count += 1
            blabel = str(br.get("id") or br.get("target") or "")
            bpath = f"{path}.branches[{blabel or bi}]"
            if not blabel:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_branch_name", "Empty branch id", bpath))
                continue
            if not is_valid_loop_branch_label(blabel):
                report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_branch_name", f"Invalid branch name '{blabel}'", bpath))
            if not blabel.startswith(label + "_b"):
                report.add(ValidationIssue(ValidationSeverity.ERROR, "orphan_loop_branch", f"Branch '{blabel}' does not belong to loop '{label}'", bpath))
            if blabel in seen_branch:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_loop_branch", f"Duplicate branch id '{blabel}'", bpath))
            seen_branch[blabel] = label
            links = [compact_link_name(str(x)) for x in (br.get("links") or []) if x]
            joints = [str(x) for x in (br.get("joints") or []) if x]
            for lk in links:
                declared_links.add(lk)
                if lk not in link_names:
                    report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_loop_link", f"Branch link '{lk}' missing", bpath))
            # Policy A: consecutive direct edges (tree neighbor OR same-loop closure edge)
            for i in range(len(links) - 1):
                a, b = links[i], links[i + 1]
                tree_ok = _tree_direct_neighbors(tree, a, b)
                # closure edges collected below; provisional check uses closures list
                closure_ok = False
                for cl in closures:
                    if not isinstance(cl, dict):
                        continue
                    fl = compact_link_name(str(cl.get("from_link") or ""))
                    tl = compact_link_name(str(cl.get("to_link") or ""))
                    if {a, b} == {fl, tl}:
                        closure_ok = True
                        break
                if a in tree and b in tree and not tree_ok and not closure_ok:
                    report.add(
                        ValidationIssue(
                            ValidationSeverity.ERROR,
                            "loop_branch_edge_mismatch",
                            f"Branch '{blabel}' links '{a}'-'{b}' are not a direct tree/closure edge",
                            bpath,
                        )
                    )
            if joints:
                if len(joints) != max(0, len(links) - 1):
                    report.add(
                        ValidationIssue(
                            ValidationSeverity.ERROR,
                            "loop_joint_path_mismatch",
                            f"Branch '{blabel}' joint_path length {len(joints)} != edges {max(0, len(links)-1)}",
                            bpath,
                        )
                    )
                for i, jn in enumerate(joints):
                    if jn not in kjoint_names and jn not in dof_names:
                        report.add(
                            ValidationIssue(
                                ValidationSeverity.ERROR,
                                "dangling_loop_joint",
                                f"Branch joint '{jn}' missing from kinematic_joints/servo_targets",
                                bpath,
                            )
                        )
                    if i >= len(links) - 1:
                        continue
                    a, b = links[i], links[i + 1]
                    edge_via = _tree_edge_via_joint(tree, a, b)
                    # explicit kinematic_joints row endpoints
                    endpoint_ok = False
                    for kj in migrated.get("kinematic_joints") or []:
                        if str(kj.get("target")) != jn:
                            continue
                        ends = {
                            compact_link_name(str(kj.get("parent_link") or "")),
                            compact_link_name(str(kj.get("child_link") or "")),
                        }
                        if ends == {a, b}:
                            endpoint_ok = True
                            break
                    if edge_via and edge_via != jn and not endpoint_ok:
                        # DOF label may equal via_joint on that edge
                        if jn != edge_via:
                            report.add(
                                ValidationIssue(
                                    ValidationSeverity.ERROR,
                                    "loop_joint_path_mismatch",
                                    f"Branch '{blabel}' joint '{jn}' does not match edge '{a}'-'{b}' (tree via={edge_via!r})",
                                    bpath,
                                )
                            )
                    elif not edge_via and not endpoint_ok:
                        # closure-edge joint is OK if matches represented_by_joint
                        closure_joint_ok = False
                        for cl in closures:
                            if not isinstance(cl, dict):
                                continue
                            fl = compact_link_name(str(cl.get("from_link") or ""))
                            tl = compact_link_name(str(cl.get("to_link") or ""))
                            rbj = cl.get("represented_by_joint") or cl.get("via_joint")
                            if {a, b} == {fl, tl} and rbj and str(rbj) == jn:
                                closure_joint_ok = True
                                break
                        if not closure_joint_ok and jn in kjoint_names:
                            # still require endpoint match when explicit kjoint exists
                            for kj in migrated.get("kinematic_joints") or []:
                                if str(kj.get("target")) == jn:
                                    ends = {
                                        compact_link_name(str(kj.get("parent_link") or "")),
                                        compact_link_name(str(kj.get("child_link") or "")),
                                    }
                                    if ends != {a, b}:
                                        report.add(
                                            ValidationIssue(
                                                ValidationSeverity.ERROR,
                                                "loop_joint_path_mismatch",
                                                f"Branch '{blabel}' joint '{jn}' endpoints != '{a}'-'{b}'",
                                                bpath,
                                            )
                                        )

        for ci, cl in enumerate(closures, start=1):
            if not isinstance(cl, dict):
                continue
            closure_count += 1
            clabel = str(cl.get("id") or cl.get("target") or "")
            cpath = f"{path}.closures[{clabel or ci}]"
            if not clabel:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_closure_name", "Empty closure id", cpath))
                continue
            if not is_valid_loop_closure_label(clabel):
                report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_closure_name", f"Invalid closure name '{clabel}'", cpath))
            if not (clabel == f"{label}_cl" or clabel.startswith(label + "_cl")):
                report.add(ValidationIssue(ValidationSeverity.ERROR, "orphan_closure", f"Closure '{clabel}' does not belong to loop '{label}'", cpath))
            if clabel in seen_closure:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_loop_closure", f"Duplicate closure id '{clabel}'", cpath))
            seen_closure[clabel] = label
            fl = compact_link_name(str(cl.get("from_link") or ""))
            tl = compact_link_name(str(cl.get("to_link") or ""))
            rbj = cl.get("represented_by_joint")
            if rbj is None:
                rbj = cl.get("via_joint")
            rbj = str(rbj) if rbj else None
            ctype = normalize_closure_constraint_type(cl.get("constraint_type") or "loop_closure")
            if ctype is None:
                report.add(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "invalid_closure_constraint_type",
                        f"Invalid constraint_type {cl.get('constraint_type')!r}",
                        cpath,
                    )
                )
            if not fl or fl not in link_names:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_closure_link", f"closure from_link '{cl.get('from_link')}' missing", cpath))
            if not tl or tl not in link_names:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_closure_link", f"closure to_link '{cl.get('to_link')}' missing", cpath))
            if fl and tl and fl == tl:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_self_closure", f"closure '{clabel}' has identical from/to link", cpath))
            if rbj and rbj not in kjoint_names and rbj not in dof_names:
                report.add(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "dangling_closure_joint",
                        f"closure represented_by_joint '{rbj}' missing",
                        cpath,
                    )
                )
            if fl and tl:
                closure_edges.add(tuple(sorted((fl, tl))))
                declared_links.add(fl)
                declared_links.add(tl)
                tree_path = find_undirected_tree_path(tree, fl, tl)
                if tree_path is None:
                    report.add(
                        ValidationIssue(
                            ValidationSeverity.ERROR,
                            "loop_does_not_close",
                            f"No spanning-tree path between '{fl}' and '{tl}' for closure '{clabel}'",
                            cpath,
                        )
                    )
                else:
                    # Mere connectivity is insufficient: declared members/branches must cover the tree path.
                    missing = [n for n in tree_path if n not in declared_links]
                    if missing:
                        report.add(
                            ValidationIssue(
                                ValidationSeverity.ERROR,
                                "loop_closure_path_mismatch",
                                f"Closure '{clabel}' tree path not covered by branch/member links; missing {missing[:12]}",
                                cpath,
                                {"tree_path": tree_path, "missing": missing},
                            )
                        )
                    # Optional soft check for classic four-bar size
                    mech = normalize_loop_mechanism_type(row.get("mechanism_type"))
                    cycle_len = len(tree_path)  # links on tree path; +closure closes
                    if mech == LoopMechanismType.FOUR_BAR and cycle_len != 4:
                        report.add(
                            ValidationIssue(
                                ValidationSeverity.WARNING,
                                "loop_member_inconsistent",
                                f"four_bar loop '{label}' tree path has {cycle_len} links (expected 4)",
                                path,
                            )
                        )

        # Member validation MUST run even when closures == []
        for mem in row.get("member_links") or []:
            ref = compact_link_name(str(mem))
            if ref not in link_names:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_loop_link", f"member link '{mem}' missing", path))
        for mem in row.get("member_dofs") or []:
            if str(mem) not in dof_names:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_loop_joint", f"member DOF '{mem}' missing", path))
        for mem in row.get("member_joints") or []:
            if str(mem) not in kjoint_names and str(mem) not in dof_names:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_loop_joint", f"member kinematic joint '{mem}' missing", path))

    report.stats["loop_count"] = len(loop_rows)
    report.stats["loop_branch_count"] = branch_count
    report.stats["closure_count"] = closure_count


def validate_migrated_master(data: dict[str, Any] | None = None) -> ValidationReport:
    """Validate after in-memory migration (referential integrity / topology / loops)."""
    raw = data if data is not None else load_master()
    migrated = migrate_master_v1_to_v2(deepcopy(raw))
    report = ValidationReport()
    report.stats["validation_stage"] = "migrated"
    link_names = _known_link_labels(migrated)
    dof_names = _known_dof_labels(migrated)
    tree = migrated.get("link_tree") or {}

    seen_dof: dict[str, str] = {}
    seen_dof_eid: dict[str, str] = {}
    for row in migrated.get("servo_targets", []) or []:
        t = str(row.get("target", ""))
        if t in seen_dof:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_target", f"Duplicate servo target '{t}'", f"servo_targets[{t}]"))
        seen_dof[t] = t
        eid = str(row.get("entity_id") or "")
        if eid:
            if eid in seen_dof_eid:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_entity_id", f"Duplicate entity_id '{eid}'", f"servo_targets[{t}]"))
            seen_dof_eid[eid] = t
            _check_entity_id_integrity(report, kind="dof", label=t, side=row.get("side"), entity_id=eid, path=f"servo_targets[{t}].entity_id")
        if normalize_laterality(row.get("side")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_side", f"Invalid side {row.get('side')!r}", f"servo_targets[{t}].side"))
        elif str(row.get("side")) in ("l", "r", "c"):
            report.add(ValidationIssue(ValidationSeverity.WARNING, "vocab_side", f"Abbreviated side {row.get('side')!r} should be left/right/center", f"servo_targets[{t}].side"))
        if normalize_morphology(row.get("morphology")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_morphology", f"Invalid morphology {row.get('morphology')!r}", f"servo_targets[{t}].morphology"))
        if normalize_joint_type(row.get("joint_type")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_joint_type", f"Invalid joint_type {row.get('joint_type')!r}", f"servo_targets[{t}].joint_type"))
        if normalize_mapping_confidence(row.get("mapping_confidence")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_confidence", f"Invalid mapping_confidence {row.get('mapping_confidence')!r}", f"servo_targets[{t}].mapping_confidence"))
        for field_name in ("parent_link", "child_link"):
            ref = compact_link_name(str(row.get(field_name) or ""))
            if not ref:
                continue
            if ref not in link_names:
                sev = ValidationSeverity.WARNING if str(row.get("mapping_confidence")) == "proposed" else ValidationSeverity.ERROR
                report.add(ValidationIssue(sev, f"dangling_{field_name}", f"{field_name} '{row.get(field_name)}' not in link registry/tree", f"servo_targets[{t}].{field_name}"))
        if row.get("entity_status") == EntityStatus.PROVISIONAL.value or row.get("mapping_confidence") == "proposed":
            report.add(ValidationIssue(ValidationSeverity.INFO, "provisional_dof", f"Provisional DOF '{t}'", f"servo_targets[{t}]"))

    seen_link: dict[str, str] = {}
    for row in migrated.get("link_targets", []) or []:
        t = compact_link_name(str(row.get("target", "")))
        if t in seen_link:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_target", f"Duplicate link target '{t}'", f"link_targets[{t}]"))
        seen_link[t] = t
        if row.get("entity_id"):
            _check_entity_id_integrity(report, kind="link", label=t, side=row.get("side"), entity_id=row.get("entity_id"), path=f"link_targets[{t}].entity_id")
        if normalize_laterality(row.get("side")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_side", f"Invalid side {row.get('side')!r}", f"link_targets[{t}].side"))
        if normalize_morphology(row.get("morphology")) is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "vocab_morphology", f"Invalid morphology {row.get('morphology')!r}", f"link_targets[{t}].morphology"))
        pa = row.get("parent_attachment")
        if pa and str(pa) not in ("", "none", "root", "null"):
            pref = compact_link_name(str(pa))
            if pref not in link_names and str(pa) not in dof_names:
                report.add(ValidationIssue(ValidationSeverity.WARNING, "dangling_parent_attachment", f"parent_attachment '{pa}' not in link/DOF registry", f"link_targets[{t}].parent_attachment"))

    # Explicit kinematic joints (passive etc.)
    seen_kj: dict[str, str] = {}
    for row in migrated.get("kinematic_joints", []) or []:
        t = str(row.get("target") or "")
        kpath = f"kinematic_joints[{t}]"
        if not t:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_kjoint_name", "Empty kinematic joint target", kpath))
            continue
        if t in seen_kj:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "duplicate_kjoint_target", f"Duplicate kinematic joint '{t}'", kpath))
        seen_kj[t] = t
        if row.get("entity_id"):
            _check_entity_id_integrity(report, kind="kjoint", label=t, side=row.get("side") or detect_side(t), entity_id=row.get("entity_id"), path=f"{kpath}.entity_id")
        if normalize_actuation_role(row.get("actuation_role") or "unknown") is None:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "invalid_actuation_role", f"Invalid actuation_role {row.get('actuation_role')!r}", kpath))
        pl = compact_link_name(str(row.get("parent_link") or ""))
        cl = compact_link_name(str(row.get("child_link") or ""))
        if not pl or pl not in link_names:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_kjoint_link", f"parent_link '{row.get('parent_link')}' missing", kpath))
        if not cl or cl not in link_names:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_kjoint_link", f"child_link '{row.get('child_link')}' missing", kpath))
        for did in row.get("dof_ids") or row.get("dofs") or []:
            if str(did) not in dof_names:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_kjoint_dof", f"dof_id '{did}' missing from servo_targets", kpath))

    for alias_key, entries in (migrated.get("alias_index") or {}).items():
        for entry in entries or []:
            target = str(entry.get("target") or "")
            ent = normalize_entity_type(entry.get("entity", "joint"))
            if ent == EntityType.JOINT.value and target and target not in dof_names:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_alias_target", f"Alias '{alias_key}' -> missing joint '{target}'", f"alias_index[{alias_key}]"))
            if ent == EntityType.LINK.value and target:
                if compact_link_name(target) not in link_names and target not in ("base_link", "c_base_link"):
                    report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_alias_target", f"Alias '{alias_key}' -> missing link '{target}'", f"alias_index[{alias_key}]"))
            if ent == EntityType.LOOP.value and target:
                loop_names = {str(r.get("target")) for r in (migrated.get("loop_targets") or []) if r.get("target")}
                if target not in loop_names:
                    report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_alias_target", f"Alias '{alias_key}' -> missing loop '{target}'", f"alias_index[{alias_key}]"))

    cycles = _detect_link_tree_cycles(tree)
    report.stats["cycle_count"] = len(cycles)
    for cyc in cycles:
        report.add(ValidationIssue(ValidationSeverity.ERROR, "topology_cycle", f"Cycle detected: {' -> '.join(cyc)}", "link_tree"))

    for link, node in tree.items():
        parent = (node or {}).get("parent")
        if parent and compact_link_name(str(parent)) not in tree:
            report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_tree_parent", f"Node '{link}' parent '{parent}' missing", f"link_tree[{link}].parent"))
        for ch in (node or {}).get("children", []) or []:
            cl = ch.get("link")
            vj = ch.get("via_joint")
            if cl and compact_link_name(str(cl)) not in tree:
                report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_tree_child", f"Child '{cl}' of '{link}' missing from tree", f"link_tree[{link}]"))
            if cl and compact_link_name(str(cl)) in tree:
                child_parent = (tree.get(compact_link_name(str(cl))) or tree.get(str(cl)) or {}).get("parent")
                if child_parent and compact_link_name(str(child_parent)) != compact_link_name(str(link)):
                    report.add(ValidationIssue(ValidationSeverity.ERROR, "topology_parent_mismatch", f"children/parent mismatch: {link} -> {cl} (child.parent={child_parent})", f"link_tree[{link}]"))
            if vj and str(vj) not in dof_names:
                # tree via_joint may be a passive kinematic joint (not a servo DOF)
                kj_labels = {str(r.get("target")) for r in (migrated.get("kinematic_joints") or []) if r.get("target")}
                if str(vj) not in kj_labels:
                    report.add(ValidationIssue(ValidationSeverity.ERROR, "dangling_via_joint", f"via_joint '{vj}' on {link}->{cl} missing from servo_targets/kinematic_joints", f"link_tree[{link}]"))
            if (node or {}).get("status") == EntityStatus.PROVISIONAL.value or (node or {}).get("link_kind") == LinkKind.AXIS_DECOMPOSITION.value:
                report.add(ValidationIssue(ValidationSeverity.INFO, "provisional_link", f"Provisional/virtual tree node '{link}'", f"link_tree[{link}]"))

    roots = [k for k, n in tree.items() if not (n or {}).get("parent")]
    if "c_base_link" not in roots and "base_link" not in roots:
        report.add(ValidationIssue(ValidationSeverity.WARNING, "topology_root", f"Unexpected roots: {roots}", "link_tree"))

    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for ch in (tree.get(cur) or {}).get("children", []) or []:
            cl = ch.get("link")
            if cl:
                stack.append(str(cl))
    disconnected = [k for k in tree if k not in reachable]
    if disconnected:
        report.add(ValidationIssue(ValidationSeverity.WARNING, "topology_disconnected", f"{len(disconnected)} nodes unreachable from roots", "link_tree", {"nodes": disconnected[:20]}))

    virtual_count = sum(
        1
        for n, node in tree.items()
        if (node or {}).get("link_kind") in (LinkKind.AXIS_DECOMPOSITION.value, LinkKind.VIRTUAL.value)
        or re.search(r"_(xr|yp|zy)$", str(n))
    )

    _validate_loop_ontology(migrated, report)

    report.stats.update(
        {
            "dof_count": len(migrated.get("servo_targets", []) or []),
            "link_count": len(migrated.get("link_targets", []) or []),
            "kjoint_count": len(migrated.get("kinematic_joints", []) or []),
            "virtual_link_count": virtual_count,
            "alias_count": sum(len(v or []) for v in (migrated.get("alias_index") or {}).values()),
            "tree_node_count": len(tree),
            "functional_joint_count": len(
                {(r.get("side"), r.get("landmark")) for r in migrated.get("servo_targets", []) if r.get("landmark")}
            ),
            "schema_version": migrated.get("schema_version"),
            "ontology_version": migrated.get("ontology_version"),
        }
    )
    return report


def validate_master(
    data: dict[str, Any] | None = None,
    *,
    stage: str = "migrated",
) -> ValidationReport:
    """Validate Master JSON.

    stage:
      - "raw": on-disk integrity (entity_id side/label); does NOT run migration rewrite first
      - "migrated": post-migration referential / topology / loop validation (default; backward compatible)
      - "both": attach raw issues into a combined report (raw errors kept; migrated is primary stats)
    """
    stage_n = str(stage or "migrated").strip().lower()
    if stage_n == "raw":
        return validate_raw_master(data)
    if stage_n == "migrated":
        return validate_migrated_master(data)
    if stage_n == "both":
        raw_report = validate_raw_master(data)
        mig_report = validate_migrated_master(data)
        combined = ValidationReport()
        combined.stats = dict(mig_report.stats)
        combined.stats["validation_stage"] = "both"
        combined.stats["raw_error_count"] = len(raw_report.errors)
        combined.stats["raw_warning_count"] = len(raw_report.warnings)
        combined.stats["migrated_error_count"] = len(mig_report.errors)
        combined.stats["migrated_warning_count"] = len(mig_report.warnings)
        for issue in raw_report.errors:
            combined.add(
                ValidationIssue(
                    issue.severity,
                    f"raw:{issue.code}",
                    issue.message,
                    issue.path,
                    issue.context,
                )
            )
        for issue in raw_report.warnings:
            combined.add(
                ValidationIssue(
                    issue.severity,
                    f"raw:{issue.code}",
                    issue.message,
                    issue.path,
                    issue.context,
                )
            )
        for issue in mig_report.errors + mig_report.warnings + mig_report.infos:
            combined.add(issue)
        return combined
    raise ValueError(f"Unknown validation stage: {stage!r}")


def format_validation_report(report: ValidationReport) -> str:
    return report.summary()


def iter_semantic_triples(
    registry: "OntologyRegistry | None" = None,
) -> Iterable[tuple[str, str, str]]:
    """Yield (subject, predicate, object) triples for future RDF/OWL export (no rdflib)."""
    reg = registry or get_ontology_registry()
    for fj in reg.functional_joints.values():
        yield (fj.entity_id, "rdf:type", "FunctionalJoint")
        if fj.laterality:
            yield (fj.entity_id, "hasLaterality", fj.laterality.value)
        if fj.landmark:
            yield (fj.entity_id, "hasLandmark", fj.landmark)
        if fj.morphology:
            yield (fj.entity_id, "hasMorphology", fj.morphology.value)
        for did in fj.dof_ids:
            yield (did, "dofOf", fj.entity_id)
    for dof in reg.dofs.values():
        yield (dof.entity_id, "rdf:type", "DegreeOfFreedom")
        yield (dof.entity_id, "canonicalLabel", dof.canonical_label)
        yield (dof.entity_id, "hasMotionClass", dof.motion_class.value)
        if dof.axis.axis_pair:
            yield (dof.entity_id, "hasAxisPair", str(dof.axis.axis_pair))
        if dof.axis.vector is not None:
            yield (dof.entity_id, "hasAxisVector", json.dumps(dof.axis.vector))
        yield (dof.entity_id, "axisExpressedIn", dof.axis.expressed_in)
        if dof.parent_link:
            yield (dof.entity_id, "parentLink", dof.parent_link)
        if dof.child_link:
            yield (dof.entity_id, "childLink", dof.child_link)
        yield (dof.entity_id, "hasStatus", dof.status.value)
    for link in reg.links.values():
        rtype = "PhysicalLink" if link.link_kind == LinkKind.PHYSICAL else "VirtualLink"
        yield (link.entity_id, "rdf:type", rtype)
        yield (link.entity_id, "canonicalLabel", link.canonical_label)
        yield (link.entity_id, "linkKind", link.link_kind.value)
    for name, node in reg.link_tree.items():
        parent = (node or {}).get("parent")
        if parent:
            yield (f"link:{parent}", "parentOf", f"link:{name}")
            yield (f"link:{name}", "childOf", f"link:{parent}")
        for ch in (node or {}).get("children", []) or []:
            cl = ch.get("link")
            vj = ch.get("via_joint")
            if cl and vj:
                yield (f"link:{name}", "connectedVia", str(vj))
    for alias in reg.aliases:
        if alias.target_entity_id:
            yield (f"alias:{alias.alias}", "mapsTo", alias.target_entity_id)
            yield (f"alias:{alias.alias}", "mappingType", alias.alias_type)
            yield (f"alias:{alias.alias}", "confidence", alias.confidence.value)

    for loop in reg.loops.values():
        yield (loop.entity_id, "rdf:type", "KinematicLoop")
        yield (loop.entity_id, "canonicalLabel", loop.canonical_label)
        yield (loop.entity_id, "hasMechanismType", loop.mechanism_type.value)
        if loop.laterality:
            yield (loop.entity_id, "hasLaterality", loop.laterality.value)
        if loop.landmark:
            yield (loop.entity_id, "hasLandmark", loop.landmark)
        if loop.morphology:
            yield (loop.entity_id, "hasMorphology", loop.morphology.value)
        for bid in loop.branch_ids:
            yield (loop.entity_id, "hasBranch", bid)
        for cid in loop.closure_ids:
            yield (loop.entity_id, "hasClosure", cid)
        for lid in loop.member_link_ids:
            link_eid = reg.links_by_label[lid].entity_id if lid in reg.links_by_label else f"link:{lid}"
            yield (link_eid, "memberOfLoop", loop.entity_id)
        for did_label in loop.member_dof_ids:
            dof_eid = reg.dofs_by_label[did_label].entity_id if did_label in reg.dofs_by_label else f"dof:{did_label}"
            yield (dof_eid, "memberOfLoop", loop.entity_id)
    for branch in reg.loop_branches.values():
        yield (branch.entity_id, "rdf:type", "LoopBranch")
        yield (branch.entity_id, "branchOf", branch.loop_id)
        yield (branch.entity_id, "canonicalLabel", branch.canonical_label)
        for lid in branch.link_path:
            yield (branch.entity_id, "containsLink", lid)
        for jid in branch.joint_path:
            yield (branch.entity_id, "containsKinematicJoint", jid)
    for closure in reg.closure_constraints.values():
        yield (closure.entity_id, "rdf:type", "ClosureConstraint")
        yield (closure.entity_id, "closureOf", closure.loop_id)
        yield (closure.entity_id, "canonicalLabel", closure.canonical_label)
        yield (closure.entity_id, "fromLink", closure.from_link)
        yield (closure.entity_id, "toLink", closure.to_link)
        if closure.represented_by_joint:
            yield (closure.entity_id, "representedByJoint", closure.represented_by_joint)
        yield (closure.entity_id, "hasConstraintType", closure.constraint_type.value)
    for kj in reg.kinematic_joints.values():
        yield (kj.entity_id, "rdf:type", "KinematicJoint")
        yield (kj.entity_id, "canonicalLabel", kj.canonical_label)
        yield (kj.entity_id, "parentLink", kj.parent_link)
        yield (kj.entity_id, "childLink", kj.child_link)
        yield (kj.entity_id, "hasActuationRole", kj.actuation_role.value)
        if kj.joint_type:
            yield (kj.entity_id, "hasJointType", kj.joint_type.value)
        for did in kj.dof_ids:
            yield (kj.entity_id, "hasDOF", did)
        if kj.realizes_functional_joint_id:
            yield (kj.entity_id, "realizesFunctionalJoint", kj.realizes_functional_joint_id)


_registry_singleton: OntologyRegistry | None = None


def get_ontology_registry(force_reload: bool = False) -> OntologyRegistry:
    global _registry_singleton
    if force_reload or _registry_singleton is None:
        _registry_singleton = OntologyRegistry.from_master(load_master())
    return _registry_singleton


def invalidate_ontology_registry() -> None:
    global _registry_singleton
    _registry_singleton = None


# ===========================================================================
# master  (RobotLabelBridge_Master.json)
# ===========================================================================

@lru_cache(maxsize=1)
def load_master() -> dict[str, Any]:
    path = _MASTER_PATH if _MASTER_PATH.exists() else _LEGACY_MASTER_PATH
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def reload_master() -> dict[str, Any]:
    """Reload master JSON from disk and reset cached converter/ontology state."""
    load_master.cache_clear()
    global _nc_singleton
    _nc_singleton = None
    invalidate_ontology_registry()
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
                    if landmark in {"pelvis", "spine_01", "neck"} and use_side not in ("l", "r"):
                        use_side = "c"
                    if landmark == "pelvis" and use_side == "c":
                        target = f"c_pelvis_root_{axis_short}"
                    elif landmark == "spine_01" and use_side == "c":
                        target = f"c_spine_01_{axis_short}"
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
        if landmark == "pelvis" and side_val == "c":
            target = f"c_pelvis_root_{axis_short}"
        elif landmark == "spine_01" and side_val == "c":
            target = f"c_spine_01_{axis_short}"
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
        req_morph = normalize_morphology(self.morphology)
        try:
            registry = get_ontology_registry()
        except Exception:
            registry = None

        scored: list[Candidate] = []
        for cand in candidates:
            components: dict[str, float] = {}
            reasons: list[str] = []
            evidence: list[str] = list(cand.evidence)

            conf_score = float(cand.confidence_rank())
            components["confidence"] = conf_score
            score = conf_score

            if cand.entity == entity_type or entity_type == EntityType.AUTO.value:
                components["entity_match"] = 1.0
                score += 1.0

            mt = cand.mapping_type
            if mt == "direct":
                components["direct_mapping"] = 2.0
                score += 2.0
                reasons.append(ReasonCode.DIRECT_MAPPING.value)
            elif mt == "alias":
                components["alias"] = 1.5
                score += 1.5
                reasons.append(ReasonCode.EXACT_ALIAS.value)
            elif mt == "product_alias":
                components["product_alias"] = 2.5
                score += 2.5
                reasons.append(ReasonCode.PRODUCT_ALIAS.value)
            elif mt == "topology_alias":
                components["topology_alias"] = 2.25
                score += 2.25
                reasons.append(ReasonCode.TOPOLOGY_MATCH.value)
            elif mt == "functional_alias":
                components["functional_alias"] = 1.0
                score += 1.0
                reasons.append(ReasonCode.FUNCTIONAL_ALIAS.value)
            elif mt == "heuristic":
                components["token_heuristic"] = 0.5
                score += 0.5
                reasons.append(ReasonCode.TOKEN_HEURISTIC.value)
            if "policy" in (cand.source or ""):
                components["policy_match"] = 1.25
                score += 1.25
                reasons.append(ReasonCode.POLICY_MATCH.value)
            if "contextual" in (cand.source or "") or mt == "contextual_pattern":
                if ReasonCode.CONTEXTUAL_ALIAS.value not in reasons:
                    reasons.append(ReasonCode.CONTEXTUAL_ALIAS.value)
                    components["contextual_alias"] = components.get("contextual_alias", 0.0) + 0.5
                    score += 0.5

            if axis_short and target_matches_axis(cand.target, axis_short):
                components["axis_match"] = 2.0
                score += 2.0
                reasons.append(ReasonCode.AXIS_MATCH.value)
                evidence.append(f"axis_short={axis_short}")
            elif axis_short and cand.entity == EntityType.JOINT.value:
                components["axis_mismatch"] = -1.0
                score -= 1.0

            if cand.target.startswith(f"{side_val}_"):
                components["side_match"] = 0.5
                score += 0.5
                reasons.append(ReasonCode.SIDE_MATCH.value)

            cand_morph = normalize_morphology(cand.morphology)
            if registry and cand.entity == EntityType.JOINT.value:
                dof = registry.dofs_by_label.get(cand.target)
                if dof:
                    cand.target_entity_id = dof.entity_id
                    cand_morph = cand_morph or dof.morphology
                    if cand_morph:
                        cand.morphology = cand_morph.value
            if registry and cand.entity == EntityType.LINK.value:
                link = registry.links_by_label.get(compact_link_name(cand.target))
                if link:
                    cand.target_entity_id = link.entity_id
                    cand_morph = cand_morph or link.morphology
            if req_morph and cand_morph:
                if cand_morph == req_morph:
                    components["morphology_match"] = 1.0
                    score += 1.0
                    reasons.append(ReasonCode.MORPHOLOGY_MATCH.value)
                elif cand_morph in (Morphology.GENERIC_VERTEBRATE, Morphology.GENERIC_ROBOT):
                    components["morphology_generic"] = 0.25
                    score += 0.25
                    reasons.append(ReasonCode.MORPHOLOGY_GENERIC.value)
                else:
                    components["morphology_mismatch"] = -1.5
                    score -= 1.5
                    reasons.append(ReasonCode.MORPHOLOGY_MISMATCH.value)
                    evidence.append(
                        f"morphology {cand_morph.value} != requested {req_morph.value}"
                    )

            if self.profile and cand.source and self.profile.lower() in cand.source.lower():
                components["profile_match"] = 1.0
                score += 1.0
            if parent and parent.lower() in cand.notes.lower():
                components["parent_note"] = 0.25
                score += 0.25
            if child and child.lower() in cand.notes.lower():
                components["child_note"] = 0.25
                score += 0.25

            cand.score = score
            cand.score_components = components
            cand.reason_codes = list(dict.fromkeys(reasons))
            cand.evidence = evidence
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
        high_conf = [
            c for c in close if c.confidence_rank() >= CONFIDENCE_RANK.get(self.min_confidence, 2)
        ]

        if len(high_conf) == 1:
            winner = high_conf[0]
            return ConversionResult(
                source="",
                normalized="",
                entity=winner.entity,
                status=ConversionStatus.RESOLVED,
                target=winner.target,
                candidates=scored,
                metadata={
                    "winner_score": winner.score,
                    "method": winner.mapping_type,
                    "score_components": winner.score_components,
                },
                reason_codes=list(winner.reason_codes),
                score_components=dict(winner.score_components),
                evidence=list(winner.evidence),
                target_entity_id=winner.target_entity_id,
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
                    "score_components": top.score_components,
                },
                reason_codes=list(top.reason_codes),
                score_components=dict(top.score_components),
                evidence=list(top.evidence),
                target_entity_id=top.target_entity_id,
            )

        if len(close) == 1 and top.confidence_rank() >= CONFIDENCE_RANK.get(self.min_confidence, 2):
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
                    "score_components": top.score_components,
                },
                reason_codes=list(top.reason_codes),
                score_components=dict(top.score_components),
                evidence=list(top.evidence),
                target_entity_id=top.target_entity_id,
            )

        return ConversionResult(
            source="",
            normalized="",
            entity=entity_type,
            status=ConversionStatus.AMBIGUOUS,
            candidates=scored,
            reasons=[f"Top candidates tied within score window: {[c.target for c in close[:5]]}"],
            reason_codes=list(top.reason_codes),
            score_components=dict(top.score_components),
            evidence=list(top.evidence),
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
            "hip_yaw": "hipjoint",
            "hip_roll": "hipjoint",
            "hip_pitch": "hipjoint",
            "hipjoint_upper_yaw": "hipjoint",
            "hipjoint_lower_roll": "hipjoint",
            "leg_upper_pitch": "hipjoint",
            "thigh_pitch": "hipjoint",
            "knee_pitch": "knee",
            "leg_lower_pitch": "knee",
            "ankle_pitch": "ankle",
            "ankle_roll": "ankle",
            "foot_small_roll": "ankle",
            "foot_roll": "ankle",
            "head_yaw": "neck",
            "head_pitch": "neck",
            "chest_yaw": "spine_01",
            "waist_yaw": "pelvis_root",
            "waist_pitch": "spine_01",
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
            "hipjoint_upper": "hipjoint",
            "hipjoint_lower": "hipjoint",
            "arm_upper_yaw": "shoulder",
            "arm_upper_roll": "shoulder",
            "arm_upper": "shoulder",
            "shoulder": "shoulder",
            "elbow": "elbow",
            "arm_lower": "elbow",
            "wrist": "wrist",
            "hand": "wrist",
            "leg_upper": "hipjoint",
            "knee": "knee",
            "leg_lower": "knee",
            "ankle": "ankle",
            "foot": "ankle",
            "head": "neck",
            "chest": "spine_01",
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
    lines = ["Aborted model file overwrite: rename plan is not safe."]
    for src, dst, reason in validation.blocked_joints:
        lines.append(f"  joint blocked: {src} -> {dst}: {reason}")
    for src, dst, reason in validation.blocked_links:
        lines.append(f"  link blocked: {src} -> {dst}: {reason}")
    for src in validation.missing_joint_sources:
        lines.append(f"  joint source not found in file: {src}")
    for src in validation.missing_link_sources:
        lines.append(f"  link source not found in file: {src}")
    lines.append(
        "If link names collide, disable \"also convert link names\" or "
        "exclude the colliding link mappings from the preview, then retry."
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
    """Return a canonical target label (lossy on ambiguity).

    .. deprecated::
        Returns the top candidate even when status is AMBIGUOUS.
        Prefer :func:`canonicalize_strict` or :func:`canonicalize_best_effort`.
    """
    import warnings

    warnings.warn(
        "canonicalize() is lossy on AMBIGUOUS results; "
        "use canonicalize_strict() or canonicalize_best_effort().",
        DeprecationWarning,
        stacklevel=2,
    )
    return canonicalize_best_effort(name, entity=entity)


def canonicalize_best_effort(name: str, entity: str = "auto") -> str | None:
    """Best-effort canonical label: RESOLVED, else top AMBIGUOUS candidate, else None."""

    def _best(r: "ConversionResult") -> "str | None":
        if r.status == ConversionStatus.RESOLVED:
            return r.target
        if r.status == ConversionStatus.AMBIGUOUS and r.candidates:
            return r.candidates[0].target
        return None

    nc = _get_nc()
    normalized_entity = normalize_entity_type(entity)
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


def canonicalize_strict(name: str, entity: str = "auto") -> str | None:
    """Strict canonical label: only RESOLVED; never promote AMBIGUOUS to success."""
    nc = _get_nc()
    normalized_entity = normalize_entity_type(entity)
    result = nc.convert(name, entity=normalized_entity)
    if result.status == ConversionStatus.RESOLVED:
        return result.target
    if result.status == ConversionStatus.AMBIGUOUS:
        return None
    # UNRESOLVED under auto: try joint/link only if exactly one side uniquely RESOLVES
    if entity == "auto" or normalized_entity == EntityType.AUTO.value:
        hits = []
        for retry in (EntityType.JOINT.value, EntityType.LINK.value):
            r = nc.convert(name, entity=retry)
            if r.status == ConversionStatus.RESOLVED and r.target:
                hits.append(r.target)
            elif r.status == ConversionStatus.AMBIGUOUS:
                return None
        if len(set(hits)) == 1:
            return hits[0]
    return None


def convert_with_provenance(name: str, entity: str = "auto", **kwargs: Any) -> ConversionResult:
    """Entity resolution exposing reason_codes / score_components / target_entity_id."""
    return _get_nc().convert(name, entity=entity, **kwargs)


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
        raise RuntimeError("No robot model is loaded.")
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError("Loaded URDF/MJCF file path was not found.")

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
    "canonicalize_strict",
    "canonicalize_best_effort",
    "convert_with_provenance",
    "validate_master",
    "format_validation_report",
    "migrate_master_v1_to_v2",
    "apply_mechanical_master_fixes",
    "get_ontology_registry",
    "OntologyRegistry",
    "iter_semantic_triples",
    "Laterality",
    "Morphology",
    "MotionClass",
    "ReasonCode",
    "parent_link",
    "ancestor_links",
    "children_links",
    "resolve_urdf_parent_in_tree",
    "normalize_entity_type",
]

__version__ = "2.2.0"


# ===========================================================================
# Qt UI helpers (Viewer + Editor)
# ===========================================================================

import html

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    _HAS_QT = True
except ImportError:  # pragma: no cover
    QtCore = QtGui = QtWidgets = None  # type: ignore[misc, assignment]
    _HAS_QT = False

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
    "link": "#80deea",   # cyan
    "joint": "#f8bbd0",  # light pink
}

ALIAS_COLS = ("Alias", "Target", "Entity", "Confidence", "Method", "Source")
COL_TARGET = 1


def normalize_search_query(raw: str) -> str:
    return re.sub(r"[\s\-/]+", "_", raw.strip().lower())


def ui_entity_color(entity: str) -> str:
    return _UI_ENTITY_COLOR.get(entity, "#e0e0e0")


def style_tree_item_entity_colors(item: "QtWidgets.QTreeWidgetItem") -> None:
    """Color the link column cyan and the via_joint column light pink."""
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
    return f"RobotLabelBridge v{__version__}"


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


if _HAS_QT:
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



    class LoopPanel(QtWidgets.QWidget):
        """Read-only closed-loop browser (link_tree remains the spanning tree)."""

        def __init__(self, parent: "QtWidgets.QWidget | None" = None) -> None:
            super().__init__(parent)
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(0, 8, 0, 0)
            split = QtWidgets.QSplitter()
            root.addWidget(split, 1)
            self._list = QtWidgets.QListWidget()
            self._detail = QtWidgets.QTextEdit()
            self._detail.setReadOnly(True)
            split.addWidget(self._list)
            split.addWidget(self._detail)
            split.setStretchFactor(0, 1)
            split.setStretchFactor(1, 2)
            self._list.currentTextChanged.connect(self._on_select)
            self._loops: dict[str, dict] = {}

        def set_loops(self, loop_targets: list[dict] | None) -> None:
            self._loops = {}
            self._list.clear()
            self._detail.clear()
            for row in loop_targets or []:
                label = str(row.get("target") or "")
                if not label:
                    continue
                self._loops[label] = row
                mech = row.get("mechanism_type") or "?"
                self._list.addItem(f"{label}  [{mech}]")
            if self._list.count() == 0:
                self._detail.setPlainText(
                "No loop_targets in Master.\n"
                "Closed loops are modeled as KinematicLoop + ClosureConstraint, "
                "not as cycles in link_tree."
                )

        def _on_select(self, text: str) -> None:
            label = text.split()[0] if text else ""
            row = self._loops.get(label)
            if not row:
                self._detail.clear()
                return
            lines = [
            f"<b>{html.escape(label)}</b>",
            f"mechanism: {html.escape(str(row.get('mechanism_type')))}",
            f"side: {html.escape(str(row.get('side')))}",
            f"landmark: {html.escape(str(row.get('landmark')))}",
            f"morphology: {html.escape(str(row.get('morphology')))}",
            f"status: {html.escape(str(row.get('status')))}",
            f"confidence: {html.escape(str(row.get('mapping_confidence')))}",
            "<br><b>Branches</b>",
            ]
            for br in row.get("branches") or []:
                lines.append(
                f"- {html.escape(str(br.get('id')))}: "
                f"links={html.escape(str(br.get('links')))} "
                f"joints={html.escape(str(br.get('joints')))}"
                )
            lines.append("<br><b>Closures</b>")
            for cl in row.get("closures") or []:
                lines.append(
                f"- {html.escape(str(cl.get('id')))}: "
                f"{html.escape(str(cl.get('from_link')))} -> "
                f"{html.escape(str(cl.get('to_link')))} "
                f"via {html.escape(str(cl.get('via_joint')))}"
                )
            if row.get("notes"):
                lines.append(f"<br>notes: {html.escape(str(row.get('notes')))}")
            self._detail.setHtml("<br>".join(lines))


    class BridgeViewer(QtWidgets.QMainWindow):
        """Read-only viewer: alias lookup, link_tree browse, conversion preview."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("RobotLabelBridge — Viewer")
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
            self._entity_filter.addItems(["all", "joint", "link", "loop"])
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

            self._loop_panel = LoopPanel()
            tabs.addTab(self._loop_panel, "Loops")

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
            self._loop_panel.set_loops(self._master.get("loop_targets", []))
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
        """Launch the read-only RobotLabelBridge viewer."""
        import sys

        app = QtWidgets.QApplication(sys.argv)
        apply_fusion_dark_theme(app)
        win = BridgeViewer()
        win.show()
        sys.exit(app.exec())


# ===========================================================================
# Standalone lookup UI  (python RobotLabelBridge.py)
# ===========================================================================


else:
    def run_viewer() -> None:
        import sys
        print("PySide6 is required for RobotLabelBridge Viewer.", file=sys.stderr)
        sys.exit(2)

    def build_alias_table(*, editable: bool):  # type: ignore[misc]
        raise RuntimeError("PySide6 is required for UI helpers")

def _run_self_test() -> int:
    """Built-in unit checks (no external test runner)."""
    import warnings
    from copy import deepcopy

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    check(normalize_laterality("l") == Laterality.LEFT, "l -> left")
    check(normalize_laterality("r") == Laterality.RIGHT, "r -> right")
    check(
        normalize_morphology("generic vertebrate") == Morphology.GENERIC_VERTEBRATE,
        "morphology alias",
    )

    migrated = migrate_master_v1_to_v2(deepcopy(load_master()))
    hip_sides = [r.get("side") for r in migrated.get("servo_targets", []) if str(r.get("target", "")).startswith("l_hipjoint")]
    check(hip_sides and all(s == "left" for s in hip_sides), f"hipjoint sides normalized: {hip_sides}")

    nc = NameConverter()
    amb = nc.convert("ShoulderJoint", entity="joint")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        best = canonicalize_best_effort("ShoulderJoint", entity="joint")
        strict = canonicalize_strict("ShoulderJoint", entity="joint")
        strict_auto = canonicalize_strict("ShoulderJoint")
    if amb.status == ConversionStatus.AMBIGUOUS:
        check(strict is None, "strict must not resolve AMBIGUOUS ShoulderJoint")
        check(strict_auto is None, "strict auto must not promote ambiguous joint via link fallback")
    check(best is None or isinstance(best, str), "best_effort type")

    report = validate_master()
    check(report.stats.get("cycle_count", 1) == 0, "no topology cycles")

    reg = get_ontology_registry(force_reload=True)
    shoulder_dofs = [
        d
        for d in reg.dofs.values()
        if d.landmark == "shoulder"
        and d.laterality == Laterality.LEFT
        and d.morphology == Morphology.HUMANOID
    ]
    check(len(shoulder_dofs) >= 1, "left shoulder DOFs exist")
    if shoulder_dofs:
        check(bool(shoulder_dofs[0].dof_of) and shoulder_dofs[0].dof_of in reg.functional_joints, "DOF->FunctionalJoint")
        check(shoulder_dofs[0].axis.vector is not None, "axis vector present")
        check(
            shoulder_dofs[0].motion_class != MotionClass.UNKNOWN
            or shoulder_dofs[0].semantic_motion_label is not None,
            "motion metadata present",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        check(canonicalize("l_hip") == "l_hipjoint_yp", "l_hip -> l_hipjoint_yp")
    check(canonicalize_best_effort("l_hip") == "l_hipjoint_yp", "best_effort l_hip")
    check(canonicalize_best_effort("c_waist_yp") is not None, "waist maps")

    res = convert_with_provenance("l_hip", entity="joint")
    if res.candidates:
        check(isinstance(res.candidates[0].score_components, dict), "score_components")

    triples = list(iter_semantic_triples(reg))
    check(len(triples) > 10, "triples exported")

    # Closed-loop naming helpers
    check(make_loop_label("left", "knee") == "l_knee_lp", "loop naming singleton")
    check(make_loop_label("left", "knee", 1) == "l_knee_lp_01", "loop naming indexed")
    check(make_loop_label("left", "knee", 2) == "l_knee_lp_02", "loop naming 02")
    check(make_loop_branch_label("l_knee_lp", 1) == "l_knee_lp_b01", "branch b01")
    check(make_loop_branch_label("l_knee_lp", 2) == "l_knee_lp_b02", "branch b02")
    check(make_loop_closure_label("l_knee_lp") == "l_knee_lp_cl", "closure singleton")
    check(make_loop_closure_label("l_knee_lp", 1) == "l_knee_lp_cl_01", "closure indexed")
    check(make_loop_label("center", "pelvis") == "c_pelvis_lp", "center loop")
    check(normalize_loop_mechanism_type("four_bar") == LoopMechanismType.FOUR_BAR, "mechanism vocab")
    check(normalize_loop_mechanism_type("pantograph") == LoopMechanismType.PANTOGRAPH, "pantograph vocab")
    check(normalize_closure_constraint_type("kinematic") == ClosureConstraintType.LOOP_CLOSURE, "constraint synonym")
    check(normalize_actuation_role("passive") == ActuationRole.PASSIVE, "actuation role")

    # Existing master remains loop-empty and valid (migrated)
    check(isinstance(load_master().get("loop_targets", []), list), "loop_targets defaultable")
    check(isinstance(load_master().get("kinematic_joints", []), list), "kinematic_joints defaultable")
    check(report.stats.get("loop_count", -1) == len(load_master().get("loop_targets") or []), "loop_count matches master")
    check(len(validate_master(stage="raw").errors) == 0, "raw master entity_id integrity")
    check(len(validate_master(stage="migrated").errors) == 0, "migrated master clean")

    # raw vs migrated: broken entity_id surfaces only on raw until migration
    broken = deepcopy(load_master())
    if broken.get("servo_targets"):
        broken["servo_targets"][0]["entity_id"] = "dof:c:WRONG_LABEL"
        raw_bad = validate_master(broken, stage="raw")
        mig_rep = MigrationReport()
        migrated_fixed = migrate_master_v1_to_v2(deepcopy(broken), report=mig_rep)
        mig_bad = validate_master(migrated_fixed, stage="migrated")
        check(any("entity_id" in i.code for i in raw_bad.errors), "raw catches bad entity_id")
        check(mig_rep.rewrite_count >= 1, "migration reports rewrite")
        check(len(mig_bad.errors) == 0 or not any(i.code == "entity_id_label_mismatch" for i in mig_bad.errors), "migrated repairs entity_id")

    def _install_test_subgraph(master: dict) -> dict:
        """Generic test-local links/joints (not written to real Master anatomy)."""
        m = deepcopy(master)
        tree = m.setdefault("link_tree", {})
        # four-bar chain under c_base_link if present else create floating root
        root = "c_base_link" if "c_base_link" in tree else "c_test_root"
        if root not in tree:
            tree[root] = {"parent": None, "children": []}
        nodes = [
            ("c_test_base", root, None),
            ("l_test_a", "c_test_base", "l_test_act_yp"),
            ("l_test_b", "l_test_a", "l_test_p01"),
            ("l_test_c", "l_test_b", "l_test_p02"),
            ("l_test_d", "l_test_c", "l_test_p03"),
            # five-bar / parallel extras
            ("l_test_e", "l_test_a", "l_test_p04"),
            ("l_test_f", "l_test_e", "l_test_p05"),
            # parallelogram / pantograph chain
            ("l_pg_a", "c_test_base", "l_pg_j1"),
            ("l_pg_b", "l_pg_a", "l_pg_j2"),
            ("l_pg_c", "l_pg_b", "l_pg_j3"),
            ("l_pg_d", "c_test_base", "l_pg_j4"),
            # center bilateral
            ("c_bi_a", "c_test_base", "c_bi_j1"),
            ("c_bi_b", "c_bi_a", "c_bi_j2"),
            ("c_bi_c", "c_bi_b", "c_bi_j3"),
        ]
        link_targets = m.setdefault("link_targets", [])
        existing_links = {compact_link_name(str(r.get("target"))) for r in link_targets}
        for name, parent, via in nodes:
            if name not in tree:
                tree[name] = {"parent": parent, "children": []}
            pnode = tree.setdefault(parent, {"parent": None, "children": []})
            if not any(ch.get("link") == name for ch in pnode.get("children", []) or []):
                pnode.setdefault("children", []).append({"link": name, "via_joint": via})
            tree[name]["parent"] = parent
            if name not in existing_links:
                side = "left" if name.startswith("l_") else ("right" if name.startswith("r_") else "center")
                link_targets.append(
                    {
                        "target": name,
                        "side": side,
                        "morphology": "humanoid",
                        "entity_id": f"link:{name[0]}:{name}",
                        "mapping_confidence": "high",
                    }
                )
                existing_links.add(name)
        # actuated DOF
        servos = m.setdefault("servo_targets", [])
        if not any(r.get("target") == "l_test_act_yp" for r in servos):
            servos.append(
                {
                    "target": "l_test_act_yp",
                    "side": "left",
                    "morphology": "humanoid",
                    "landmark": "test_act",
                    "joint_type": "revolute",
                    "axis_pair": "y+/y-",
                    "semantic_motion": "flexion/extension",
                    "parent_link": "c_test_base",
                    "child_link": "l_test_a",
                    "mapping_confidence": "high",
                    "entity_id": "dof:l:l_test_act_yp",
                }
            )
        # passive kinematic joints (NOT in servo_targets)
        kj = m.setdefault("kinematic_joints", [])
        passive = [
            ("l_test_p01", "l_test_a", "l_test_b"),
            ("l_test_p02", "l_test_b", "l_test_c"),
            ("l_test_p03", "l_test_c", "l_test_d"),
            ("l_test_p04", "l_test_a", "l_test_e"),
            ("l_test_p05", "l_test_e", "l_test_f"),
            ("l_test_cl_j", "l_test_d", "c_test_base"),  # closure pivot
            ("l_test_cl2", "l_test_f", "l_test_c"),
            ("l_pg_j1", "c_test_base", "l_pg_a"),
            ("l_pg_j2", "l_pg_a", "l_pg_b"),
            ("l_pg_j3", "l_pg_b", "l_pg_c"),
            ("l_pg_j4", "c_test_base", "l_pg_d"),
            ("l_pg_cl", "l_pg_c", "l_pg_d"),
            ("c_bi_j1", "c_test_base", "c_bi_a"),
            ("c_bi_j2", "c_bi_a", "c_bi_b"),
            ("c_bi_j3", "c_bi_b", "c_bi_c"),
            ("c_bi_cl", "c_bi_c", "c_test_base"),
        ]
        have = {str(r.get("target")) for r in kj}
        for target, parent, child in passive:
            if target in have:
                continue
            side = "left" if target.startswith("l_") else "center"
            kj.append(
                {
                    "target": target,
                    "entity_id": f"kjoint:{side[0] if side!='center' else 'c'}:{target}".replace("kjoint:l:", "kjoint:l:").replace("kjoint:c:", "kjoint:c:"),
                    "side": side,
                    "parent_link": parent,
                    "child_link": child,
                    "joint_type": "revolute",
                    "actuation_role": "passive",
                    "dof_ids": [],
                    "mapping_confidence": "high",
                    "status": "canonical",
                }
            )
            # fix entity_id properly
            kj[-1]["entity_id"] = f"kjoint:{'l' if side=='left' else 'c'}:{target}"
        return m

    fixture = _install_test_subgraph(load_master())

    # A. Passive joint registry
    reg_p = OntologyRegistry.from_master(fixture)
    check(reg_p.get_kinematic_joint("l_test_p01") is not None, "passive joint loads")
    check(reg_p.get_kinematic_joint("l_test_p01").actuation_role == ActuationRole.PASSIVE, "passive role")
    check("l_test_p01" not in reg_p.dofs_by_label, "passive not forged as servo DOF")

    # B. Four-bar valid
    four = deepcopy(fixture)
    four["loop_targets"] = [
        {
            "target": "l_test_lp",
            "entity_id": "loop:l:l_test_lp",
            "side": "left",
            "landmark": "test",
            "morphology": "humanoid",
            "mechanism_type": "four_bar",
            "branches": [
                {
                    "id": "l_test_lp_b01",
                    "links": ["c_test_base", "l_test_a", "l_test_b", "l_test_c", "l_test_d"],
                    "joints": ["l_test_act_yp", "l_test_p01", "l_test_p02", "l_test_p03"],
                },
                {
                    "id": "l_test_lp_b02",
                    "links": ["l_test_d", "c_test_base"],
                    "joints": ["l_test_cl_j"],
                },
            ],
            "closures": [
                {
                    "id": "l_test_lp_cl",
                    "from_link": "l_test_d",
                    "to_link": "c_test_base",
                    "represented_by_joint": "l_test_cl_j",
                    "constraint_type": "loop_closure",
                }
            ],
            "member_links": ["c_test_base", "l_test_a", "l_test_b", "l_test_c", "l_test_d"],
            "mapping_confidence": "high",
            "status": "canonical",
        }
    ]
    four_report = validate_master(four, stage="migrated")
    check(four_report.stats.get("cycle_count", 1) == 0, "four-bar keeps tree acyclic")
    check(
        not any(i.code.startswith("loop_") or i.code.startswith("duplicate_loop") or i.code.startswith("dangling_loop") or i.code.startswith("invalid_loop") for i in four_report.errors),
        f"four-bar accepted: {[i.code for i in four_report.errors[:12]]}",
    )
    reg_f = OntologyRegistry.from_master(four)
    check(len(reg_f.branches_for_loop("l_test_lp")) == 2, "four-bar branches")
    check(any(t[1] == "hasActuationRole" for t in iter_semantic_triples(reg_f)), "kjoint triples")

    # C. Invalid remote closure (connected but not declared)
    remote = deepcopy(fixture)
    # use real anatomy if present
    tree = remote.get("link_tree") or {}
    if "c_pelvis" in tree and "l_foot" in tree:
        remote["loop_targets"] = [
            {
                "target": "l_leg_lp",
                "entity_id": "loop:l:l_leg_lp",
                "side": "left",
                "landmark": "leg",
                "morphology": "humanoid",
                "mechanism_type": "custom",
                "branches": [
                    {"id": "l_leg_lp_b01", "links": ["c_pelvis"], "joints": []},
                    {"id": "l_leg_lp_b02", "links": ["l_foot"], "joints": []},
                ],
                "closures": [
                    {
                        "id": "l_leg_lp_cl",
                        "from_link": "c_pelvis",
                        "to_link": "l_foot",
                        "constraint_type": "loop_closure",
                    }
                ],
                "mapping_confidence": "high",
                "status": "canonical",
            }
        ]
        remote_report = validate_master(remote, stage="migrated")
        check(
            any(i.code == "loop_closure_path_mismatch" for i in remote_report.errors),
            "remote closure rejected",
        )

    # D. Invalid branch path (waypoints / skipped intermediate — not a direct edge)
    bad_branch = deepcopy(four)
    bad_branch["loop_targets"][0]["branches"][0]["links"] = ["l_test_a", "l_test_c"]
    bad_branch["loop_targets"][0]["branches"][0]["joints"] = []
    br_report = validate_master(bad_branch, stage="migrated")
    check(any(i.code == "loop_branch_edge_mismatch" for i in br_report.errors), "non-adjacent branch rejected")

    # E. joint_path mismatch
    bad_j = deepcopy(four)
    bad_j["loop_targets"][0]["branches"][0]["joints"] = ["l_test_p03", "l_test_p01", "l_test_p02", "l_test_act_yp"]
    j_report = validate_master(bad_j, stage="migrated")
    check(any(i.code == "loop_joint_path_mismatch" for i in j_report.errors), "joint_path mismatch rejected")

    # F. member_links with zero closures
    zero_cl = deepcopy(fixture)
    zero_cl["loop_targets"] = [
        {
            "target": "l_test_lp",
            "entity_id": "loop:l:l_test_lp",
            "side": "left",
            "landmark": "test",
            "morphology": "humanoid",
            "mechanism_type": "custom",
            "branches": [{"id": "l_test_lp_b01", "links": ["c_test_base"], "joints": []}],
            "closures": [],
            "member_links": ["no_such_link_zzz"],
            "mapping_confidence": "high",
            "status": "canonical",
        }
    ]
    z_report = validate_master(zero_cl, stage="migrated")
    check(any(i.code == "dangling_loop_link" for i in z_report.errors), "member validated with zero closures")

    # G. Loop alias binding
    alias_fix = deepcopy(four)
    alias_fix.setdefault("alias_index", {})["LeftTestLoop"] = [
        {"target": "l_test_lp", "entity": "loop", "mapping_type": "alias", "confidence": "high", "source": "test"}
    ]
    reg_a = OntologyRegistry.from_master(alias_fix)
    aa = [a for a in reg_a.aliases if a.alias == "LeftTestLoop"]
    check(aa and aa[0].target_entity_id == "loop:l:l_test_lp", f"loop alias bound: {aa[0].target_entity_id if aa else None}")

    # H already covered raw vs migrated above

    # I. duplicate branch / closure codes
    dup = deepcopy(four)
    dup["loop_targets"][0]["branches"].append(deepcopy(dup["loop_targets"][0]["branches"][0]))
    dup["loop_targets"][0]["closures"].append(deepcopy(dup["loop_targets"][0]["closures"][0]))
    dup_report = validate_master(dup, stage="migrated")
    check(any(i.code == "duplicate_loop_branch" for i in dup_report.errors), "duplicate_loop_branch code")
    check(any(i.code == "duplicate_loop_closure" for i in dup_report.errors), "duplicate_loop_closure code")

    # J. legacy conversion regression
    check(canonicalize_best_effort("l_hip") is not None, "legacy convert still works")

    # Additional fixtures: five-bar / parallel, parallelogram, multi-closure, center
    five = deepcopy(fixture)
    five["loop_targets"] = [
        {
            "target": "l_test_lp",
            "entity_id": "loop:l:l_test_lp",
            "side": "left",
            "landmark": "test",
            "morphology": "humanoid",
            "mechanism_type": "five_bar",
            "branches": [
                {
                    "id": "l_test_lp_b01",
                    "links": ["l_test_a", "l_test_b", "l_test_c"],
                    "joints": ["l_test_p01", "l_test_p02"],
                },
                {
                    "id": "l_test_lp_b02",
                    "links": ["l_test_a", "l_test_e", "l_test_f"],
                    "joints": ["l_test_p04", "l_test_p05"],
                },
                {
                    "id": "l_test_lp_b03",
                    "links": ["l_test_f", "l_test_c"],
                    "joints": ["l_test_cl2"],
                },
            ],
            "closures": [
                {
                    "id": "l_test_lp_cl",
                    "from_link": "l_test_f",
                    "to_link": "l_test_c",
                    "represented_by_joint": "l_test_cl2",
                    "constraint_type": "loop_closure",
                }
            ],
            "mapping_confidence": "high",
            "status": "canonical",
        }
    ]
    five_report = validate_master(five, stage="migrated")
    check(
        not any(i.code in ("loop_branch_edge_mismatch", "loop_closure_path_mismatch", "dangling_loop_joint") for i in five_report.errors),
        f"five-bar accepted: {[i.code for i in five_report.errors[:10]]}",
    )

    pg = deepcopy(fixture)
    pg["loop_targets"] = [
        {
            "target": "l_wing_lp",
            "entity_id": "loop:l:l_wing_lp",
            "side": "left",
            "landmark": "wing",
            "morphology": "avian",
            "mechanism_type": "parallelogram",
            "branches": [
                {"id": "l_wing_lp_b01", "links": ["c_test_base", "l_pg_a", "l_pg_b", "l_pg_c"], "joints": ["l_pg_j1", "l_pg_j2", "l_pg_j3"]},
                {"id": "l_wing_lp_b02", "links": ["c_test_base", "l_pg_d"], "joints": ["l_pg_j4"]},
                {"id": "l_wing_lp_b03", "links": ["l_pg_c", "l_pg_d"], "joints": ["l_pg_cl"]},
            ],
            "closures": [
                {
                    "id": "l_wing_lp_cl",
                    "from_link": "l_pg_c",
                    "to_link": "l_pg_d",
                    "represented_by_joint": "l_pg_cl",
                    "constraint_type": "loop_closure",
                }
            ],
            "mapping_confidence": "high",
            "status": "canonical",
        }
    ]
    # morphology avian may not be in CV - use humanoid if needed
    from RobotLabelBridge import normalize_morphology as _nm
    if _nm("avian") is None:
        pg["loop_targets"][0]["morphology"] = "humanoid"
        pg["loop_targets"][0]["mechanism_type"] = "pantograph"
    pg_report = validate_master(pg, stage="migrated")
    check(
        not any(i.code in ("loop_branch_edge_mismatch", "loop_closure_path_mismatch") for i in pg_report.errors),
        f"parallelogram/pantograph accepted: {[i.code for i in pg_report.errors[:10]]}",
    )

    center = deepcopy(fixture)
    center["loop_targets"] = [
        {
            "target": "c_pelvis_lp",
            "entity_id": "loop:c:c_pelvis_lp",
            "side": "center",
            "landmark": "pelvis",
            "morphology": "humanoid",
            "mechanism_type": "parallel",
            "branches": [
                {"id": "c_pelvis_lp_b01", "links": ["c_test_base", "c_bi_a", "c_bi_b", "c_bi_c"], "joints": ["c_bi_j1", "c_bi_j2", "c_bi_j3"]},
                {"id": "c_pelvis_lp_b02", "links": ["c_bi_c", "c_test_base"], "joints": ["c_bi_cl"]},
            ],
            "closures": [
                {
                    "id": "c_pelvis_lp_cl",
                    "from_link": "c_bi_c",
                    "to_link": "c_test_base",
                    "represented_by_joint": "c_bi_cl",
                    "constraint_type": "distance",
                },
                {
                    "id": "c_pelvis_lp_cl_01",
                    "from_link": "c_bi_c",
                    "to_link": "c_test_base",
                    "represented_by_joint": "c_bi_cl",
                    "constraint_type": "loop_closure",
                },
            ],
            "mapping_confidence": "high",
            "status": "canonical",
        }
    ]
    # multi-closure same endpoints is odd but naming should validate; path coverage ok
    c_report = validate_master(center, stage="migrated")
    check(any(is_valid_loop_closure_label("c_pelvis_lp_cl_01") for _ in [0]), "multi closure naming")
    check(
        not any(i.code in ("invalid_closure_name", "loop_closure_path_mismatch") for i in c_report.errors),
        f"center/multi-closure: {[i.code for i in c_report.errors[:10]]}",
    )

    # headless: core symbols exist even when UI optional
    check(callable(validate_master) and callable(OntologyRegistry.from_master), "core API callable")

    if failures:
        print("SELF-TEST FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("SELF-TEST OK")
    print(
        f"  dofs={len(reg.dofs)} functional_joints={len(reg.functional_joints)} "
        f"links={len(reg.links)} loops={len(reg.loops)} triples={len(triples)}"
    )
    return 0


def _main(argv: list[str] | None = None) -> None:
    import argparse
    import sys
    from copy import deepcopy

    parser = argparse.ArgumentParser(description="RobotLabelBridge — ontology-backed naming bridge")
    parser.add_argument("--validate-master", action="store_true", help="Validate Master JSON; exit 1 on ERROR")
    parser.add_argument("--self-test", action="store_true", help="Run built-in unit checks")
    parser.add_argument("--migrate-master", action="store_true", help="Apply mechanical Master fixes and write JSON")
    parser.add_argument("--dump-triples", action="store_true", help="Print sample semantic triples")
    args = parser.parse_args(argv)

    if args.migrate_master:
        fixed = apply_mechanical_master_fixes(load_master())
        ordered: dict[str, Any] = {
            "schema_version": fixed.get("schema_version", "2.1"),
            "ontology_version": fixed.get("ontology_version", "1.1"),
            "data_version": fixed.get("data_version", "2026.08.09"),
        }
        for k, v in fixed.items():
            if k not in ordered:
                ordered[k] = v
        _MASTER_PATH.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reload_master()
        report = validate_master()
        print(report.summary())
        print(f"Wrote {_MASTER_PATH}")
        # allow warnings; fail only on errors
        sys.exit(0 if not report.errors else 1)

    if args.validate_master:
        raw_report = validate_master(stage="raw")
        mig_report = validate_master(stage="migrated")
        print("RAW VALIDATION")
        print("----------------")
        print(raw_report.summary())
        print()
        print("MIGRATED VALIDATION")
        print("-------------------")
        print(mig_report.summary())
        # Exit non-zero if either stage has errors (raw defects must not be hidden)
        sys.exit(0 if not raw_report.errors and not mig_report.errors else 1)

    if args.self_test:
        sys.exit(_run_self_test())

    if args.dump_triples:
        for i, t in enumerate(iter_semantic_triples()):
            print(t)
            if i >= 39:
                break
        sys.exit(0)

    run_viewer()


if __name__ == "__main__":
    _main()
