#!/usr/bin/env python3
"""
File Name: urdf_kitchen_Importer.py
Description: URDF/MJCF Import functionality for URDF Kitchen.
             This module provides parser classes and import methods for
             loading URDF, MJCF, SDF, and SRDF files into the Assembler.

Author      : Ninagawa123
Created On  : Nov 28, 2024
Update.     : Jan 19, 2026
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
pip install xacrodoc
"""

import os
import math
import xml.etree.ElementTree as ET
import subprocess
import traceback
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

# Import utilities from urdf_kitchen_utils
from urdf_kitchen_utils import (
    ConversionUtils,
    resolve_ros_find_syntax,
    resolve_file_path_aggressive,
    resolve_path_in_xml_element,
    find_package_root,
    match_package_name,
    normalize_name_for_matching
)


# ============================================================================
# CONSTANTS (from urdf_kitchen_Assembler.py)
# ============================================================================

DEFAULT_JOINT_EFFORT = 1.37  # N*m
DEFAULT_JOINT_VELOCITY = 7.0  # rad/s
DEFAULT_MARGIN = 0.01  # m
DEFAULT_ARMATURE = 0.01  # kg*m^2
DEFAULT_FRICTIONLOSS = 0.01  # N*m
DEFAULT_STIFFNESS_KP = 100.0  # N*m/rad
DEFAULT_DAMPING_KV = 1.0  # N*m*s/rad
DEFAULT_COLOR_WHITE = [1.0, 1.0, 1.0]
DEFAULT_COORDS_ZERO = [0.0, 0.0, 0.0]
DEFAULT_INERTIA_ZERO = {
    'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
    'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
}
DEFAULT_ORIGIN_ZERO = {
    'xyz': [0.0, 0.0, 0.0],
    'rpy': [0.0, 0.0, 0.0]
}


# ============================================================================
# HELPER FUNCTIONS (from urdf_kitchen_Assembler.py)
# ============================================================================

def is_mesh_reversed_check(visual_origin, mesh_scale):
    """
    メッシュが反転（ミラーリング）されているかを判定する関数

    Args:
        visual_origin: visual_origin辞書 {'xyz': [...], 'rpy': [...]}
        mesh_scale: mesh_scaleリスト [x, y, z]

    Returns:
        bool: 反転している場合True
    """
    PI = math.pi
    PI_TOLERANCE = 0.01

    # RPYのいずれかの軸がπ（PI）に近い場合は反転とみなす
    if visual_origin:
        rpy = visual_origin.get('rpy', [0.0, 0.0, 0.0])
        for angle in rpy:
            if abs(abs(angle) - PI) < PI_TOLERANCE:
                return True

    # mesh_scaleのいずれかの軸が負の場合は反転とみなす
    if mesh_scale:
        for scale in mesh_scale:
            if scale < 0:
                return True

    return False


def create_cumulative_coord(index):
    """累積座標データを作成"""
    return {
        'point_index': index,
        'xyz': DEFAULT_COORDS_ZERO.copy()
    }


# ============================================================================
# URDF PARSER
# ============================================================================

class URDFParser:
    """Parser for URDF (Unified Robot Description Format) files.

    This class provides methods to parse URDF XML files and extract
    robot structure information including links, joints, materials,
    and mesh file paths.

    Usage:
        parser = URDFParser()
        result = parser.parse_urdf('/path/to/robot.urdf')
        
        # Access parsed data
        robot_name = result['robot_name']
        links_data = result['links']
        joints_data = result['joints']
        materials_data = result['materials']
    """

    def __init__(self, verbose=False):
        """Initialize URDF parser.

        Args:
            verbose: If True, print parsing progress and information
        """
        self.verbose = verbose

    def _expand_xacro(self, xacro_file_path):
        """Expand xacro file to URDF XML string using xacrodoc.

        Args:
            xacro_file_path: Path to xacro file

        Returns:
            str: Expanded URDF XML string

        Raises:
            FileNotFoundError: If xacro file does not exist
            RuntimeError: If xacrodoc is not available or expansion fails
        """
        if not os.path.exists(xacro_file_path):
            raise FileNotFoundError(f"Xacro file not found: {xacro_file_path}")

        if self.verbose:
            print(f"Expanding xacro file using xacrodoc: {xacro_file_path}")

        # Try xacrodoc library
        try:
            from xacrodoc import XacroDoc
            
            if self.verbose:
                print("Using xacrodoc library")
            
            # xacroファイルからXacroDocオブジェクトを作成
            xacro_file_abs = os.path.abspath(xacro_file_path)
            doc = XacroDoc.from_file(xacro_file_abs)
            
            # URDF文字列に変換
            urdf_xml_string = doc.to_urdf_string()
            
            # 生成されたXMLが正しくパースできるか確認
            if urdf_xml_string:
                try:
                    root = ET.fromstring(urdf_xml_string)
                    # robot要素の存在を確認
                    if root.tag != 'robot':
                        if self.verbose:
                            print(f"  Warning: Root element is '{root.tag}', expected 'robot'")
                        # robot要素が見つからない場合は、子要素を探す
                        robot_elem = root.find('robot')
                        if robot_elem is None:
                            # ルート要素がrobotでない場合は警告を出すが、続行
                            if self.verbose:
                                print(f"  Warning: No 'robot' element found in expanded xacro")
                    else:
                        if self.verbose:
                            print(f"  Successfully expanded xacro to URDF (robot element found)")
                    
                    # デバッグ: 展開されたXMLの構造を確認
                    if self.verbose:
                        link_count = len(root.findall('link'))
                        joint_count = len(root.findall('joint'))
                        print(f"  Found {link_count} link elements, {joint_count} joint elements in expanded XML")
                        if link_count == 0:
                            print(f"  WARNING: No links found in expanded xacro!")
                            # XMLの最初の500文字を出力してデバッグ
                            print(f"  First 500 chars of expanded XML:")
                            print(f"  {urdf_xml_string[:500]}")
                    
                    return urdf_xml_string
                except ET.ParseError as parse_err:
                    if self.verbose:
                        print(f"  Warning: Generated XML has parse error: {parse_err}")
                        print(f"  First 200 chars: {urdf_xml_string[:200]}")
                    raise RuntimeError(f"xacrodoc generated invalid XML: {parse_err}")
            else:
                raise RuntimeError("xacrodoc returned empty result")
                
        except ImportError:
            error_msg = (
                "xacrodoc library not found. Please install it using:\n\n"
                "  pip install xacrodoc\n\n"
                f"Xacro file: {xacro_file_path}"
            )
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = (
                f"Failed to expand xacro file using xacrodoc: {e}\n"
                f"File: {xacro_file_path}\n"
            )
            if self.verbose:
                traceback.print_exc()
            raise RuntimeError(error_msg)

    def parse_urdf(self, urdf_file_path):
        """Parse URDF file and extract robot structure information.
        
        Supports both .urdf and .xacro files. If a .xacro file is provided,
        it will be automatically expanded before parsing.

        Args:
            urdf_file_path: Path to URDF or xacro file

        Returns:
            Dictionary containing:
                - robot_name: Name of the robot
                - links: Dictionary of link data
                - joints: List of joint data
                - materials: Dictionary of material colors
                - missing_meshes: List of meshes that could not be found

        Raises:
            FileNotFoundError: If URDF/xacro file does not exist
            ET.ParseError: If URDF file is not valid XML
            RuntimeError: If xacro expansion fails
        """
        if not os.path.exists(urdf_file_path):
            raise FileNotFoundError(f"URDF/xacro file not found: {urdf_file_path}")

        # Check if file is a xacro file
        file_ext = os.path.splitext(urdf_file_path)[1].lower()
        is_xacro = file_ext in ['.xacro', '.xacro.urdf']

        if is_xacro:
            if self.verbose:
                print(f"Detected xacro file: {urdf_file_path}")
            
            # Expand xacro to URDF
            urdf_xml_string = self._expand_xacro(urdf_file_path)
            
            # 展開されたXML文字列が有効か確認
            if not urdf_xml_string or not urdf_xml_string.strip():
                raise RuntimeError("xacro expansion returned empty string")
            
            if self.verbose:
                print(f"  Expanded xacro XML length: {len(urdf_xml_string)} characters")
            
            # Parse expanded URDF XML string
            try:
                root = ET.fromstring(urdf_xml_string)
                
                # デバッグ: 展開されたXMLの構造を確認
                if self.verbose:
                    print(f"  Expanded XML root tag: {root.tag}")
                    # 最初の数個のリンクとジョイントを確認
                    link_count = len(root.findall('link'))
                    joint_count = len(root.findall('joint'))
                    print(f"  Found {link_count} link elements, {joint_count} joint elements in expanded XML")
                    if link_count == 0:
                        print(f"  WARNING: No links found in expanded xacro!")
                        # XMLの最初の500文字を出力してデバッグ
                        print(f"  First 500 chars of expanded XML:")
                        print(f"  {urdf_xml_string[:500]}")
                
                # robot要素の存在を確認
                if root.tag != 'robot':
                    if self.verbose:
                        print(f"  Warning: Root element is '{root.tag}', expected 'robot'")
                    # robot要素が見つからない場合は、子要素を探す
                    robot_elem = root.find('robot')
                    if robot_elem is not None:
                        root = robot_elem
                        if self.verbose:
                            print(f"  Found 'robot' element as child, using it")
                    else:
                        if self.verbose:
                            print(f"  Warning: No 'robot' element found, but continuing with root element '{root.tag}'")
                else:
                    if self.verbose:
                        print(f"  Successfully parsed expanded xacro XML (robot element found)")
            except ET.ParseError as e:
                if self.verbose:
                    print(f"Error parsing expanded xacro XML:")
                    print(f"  First 500 chars: {urdf_xml_string[:500]}")
                    print(f"  Last 200 chars: {urdf_xml_string[-200:]}")
                raise RuntimeError(f"Failed to parse expanded xacro XML: {e}\nFirst 500 chars: {urdf_xml_string[:500]}")
        else:
            if self.verbose:
                print(f"Parsing URDF from: {urdf_file_path}")
            
            # Parse URDF file directly
            tree = ET.parse(urdf_file_path)
            root = tree.getroot()
            
            # robot要素の存在を確認
            if root.tag != 'robot':
                if self.verbose:
                    print(f"  Warning: Root element is '{root.tag}', expected 'robot'")
                # robot要素が見つからない場合は、子要素を探す
                robot_elem = root.find('robot')
                if robot_elem is not None:
                    root = robot_elem
                    if self.verbose:
                        print(f"  Found 'robot' element as child, using it")

        # Get robot name from robot element or file name
        robot_name = root.get('name')
        if not robot_name:
            # robot要素にname属性がない場合は、ファイル名から取得
            robot_name = os.path.splitext(os.path.basename(urdf_file_path))[0]
            if self.verbose:
                print(f"  Robot name not found in XML, using filename: {robot_name}")
        else:
            if self.verbose:
                print(f"Robot name from XML: {robot_name}")

        # Extract data
        materials_data = self._parse_materials(root)
        links_data, missing_meshes = self._parse_links(root, urdf_file_path, materials_data)
        joints_data = self._parse_joints(root, links_data)

        return {
            'robot_name': robot_name,
            'links': links_data,
            'joints': joints_data,
            'materials': materials_data,
            'missing_meshes': missing_meshes
        }

    def _parse_materials(self, root):
        """Parse material information from URDF.

        Args:
            root: XML root element

        Returns:
            Dictionary mapping material names to RGBA colors
        """
        materials_data = {}

        for material_elem in root.findall('material'):
            mat_name = material_elem.get('name')
            color_elem = material_elem.find('color')
            if color_elem is not None:
                rgba_str = color_elem.get('rgba', '1.0 1.0 1.0 1.0')
                rgba = [float(v) for v in rgba_str.split()]
                # RGBA（4要素）を保存（Alphaがない場合は1.0を追加）
                if len(rgba) >= 4:
                    materials_data[mat_name] = rgba[:4]
                elif len(rgba) == 3:
                    materials_data[mat_name] = rgba + [1.0]  # Alpha=1.0を追加
                else:
                    # 不正な形式の場合はデフォルト値
                    materials_data[mat_name] = [1.0, 1.0, 1.0, 1.0]

        if self.verbose:
            print(f"Parsed {len(materials_data)} materials")

        return materials_data

    def _parse_links(self, root, urdf_file_path, materials_data):
        """Parse link information from URDF.

        Args:
            root: XML root element
            urdf_file_path: Path to URDF file (for resolving mesh paths)
            materials_data: Dictionary of material colors

        Returns:
            Tuple of (links_data dict, missing_meshes list)
        """
        links_data = {}
        missing_meshes = []

        # デバッグ: リンク要素の数を確認
        link_elems = root.findall('link')
        if self.verbose:
            print(f"\n[URDFParser] Found {len(link_elems)} link elements in URDF")

        for link_elem in link_elems:
            link_name = link_elem.get('name')
            if not link_name:
                if self.verbose:
                    print(f"  Warning: Found link element without name attribute, skipping")
                continue
            
            if self.verbose:
                print(f"  Parsing link: {link_name}")
            link_data = {
                'name': link_name,
                'mass': 0.0,
                'inertia': {'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0},
                'inertial_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                'visual_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                'stl_file': None,
                'color': [1.0, 1.0, 1.0, 1.0],  # RGBA（4要素）に変更
                'stl_filename_original': None,
                'mesh_scale': [1.0, 1.0, 1.0],
                'decorations': [],
                'collision_mesh': None,  # Deprecated, use colliders list
                'colliders': []  # List of collider dictionaries
            }

            # Parse inertial information
            inertial_elem = link_elem.find('inertial')
            if inertial_elem is not None:
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    link_data['mass'] = float(mass_elem.get('value', 0.0))

                inertia_elem = inertial_elem.find('inertia')
                if inertia_elem is not None:
                    for key in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                        link_data['inertia'][key] = float(inertia_elem.get(key, 0.0))

                origin_elem = inertial_elem.find('origin')
                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0 0 0')
                    rpy_str = origin_elem.get('rpy', '0 0 0')
                    link_data['inertial_origin']['xyz'] = [float(v) for v in xyz_str.split()]
                    link_data['inertial_origin']['rpy'] = [float(v) for v in rpy_str.split()]

                if self.verbose:
                    print(f"\n[URDF_INERTIAL] link_name={link_name}")
                    print(f"  mass={link_data['mass']:.9e}")
                    print(f"  origin_xyz={link_data['inertial_origin']['xyz']}")
                    print(f"  origin_rpy={link_data['inertial_origin']['rpy']}")

            # Parse visual information (multiple visual tags supported)
            visual_elems = link_elem.findall('visual')
            for visual_idx, visual_elem in enumerate(visual_elems):
                is_main_visual = (visual_idx == 0)

                current_stl_path = None
                current_color = [1.0, 1.0, 1.0, 1.0]  # RGBA（4要素）に変更
                current_visual_origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}

                # Parse visual origin
                visual_origin_elem = visual_elem.find('origin')
                if visual_origin_elem is not None:
                    xyz_str = visual_origin_elem.get('xyz', '0 0 0')
                    rpy_str = visual_origin_elem.get('rpy', '0 0 0')
                    try:
                        xyz_values = [float(v) for v in xyz_str.split()]
                        rpy_values = [float(v) for v in rpy_str.split()]
                        if len(xyz_values) == 3:
                            current_visual_origin['xyz'] = xyz_values
                        if len(rpy_values) == 3:
                            current_visual_origin['rpy'] = rpy_values
                    except ValueError as e:
                        if self.verbose:
                            print(f"Warning: Invalid visual origin values: {e}")

                # Parse geometry
                geometry_elem = visual_elem.find('geometry')
                mesh_scale = [1.0, 1.0, 1.0]
                
                if geometry_elem is not None:
                    mesh_elem = geometry_elem.find('mesh')
                    if mesh_elem is not None:
                        mesh_filename = mesh_elem.get('filename', '')
                        
                        # Parse scale attribute
                        scale_str = mesh_elem.get('scale', '')
                        if scale_str:
                            try:
                                scale_values = [float(v) for v in scale_str.split()]
                                if len(scale_values) == 3:
                                    mesh_scale = scale_values
                            except ValueError:
                                if self.verbose:
                                    print(f"Warning: Invalid scale attribute '{scale_str}'")

                        # Resolve mesh file path
                        resolved_path = self._resolve_mesh_path(mesh_filename, urdf_file_path)
                        
                        if resolved_path:
                            current_stl_path = resolved_path
                        else:
                            if self.verbose:
                                print(f"Could not find mesh file for link {link_name}: {mesh_filename}")
                            mesh_basename = os.path.basename(mesh_filename)
                            if is_main_visual:
                                link_data['stl_filename_original'] = mesh_filename
                                missing_meshes.append({
                                    'link_name': link_name,
                                    'filename': mesh_filename,
                                    'basename': mesh_basename
                                })
                            else:
                                # Decoration visualのメッシュファイルが見つからない場合も記録
                                missing_meshes.append({
                                    'link_name': link_name,
                                    'filename': mesh_filename,
                                    'basename': mesh_basename,
                                    'is_decoration': True
                                })

                # Parse material color
                material_elem = visual_elem.find('material')
                if material_elem is not None:
                    mat_name = material_elem.get('name')
                    if mat_name in materials_data:
                        # materials_dataからRGBA（4要素）を取得
                        current_color = materials_data[mat_name]
                    else:
                        color_elem = material_elem.find('color')
                        if color_elem is not None:
                            rgba_str = color_elem.get('rgba', '1.0 1.0 1.0 1.0')
                            rgba = [float(v) for v in rgba_str.split()]
                            # RGBA（4要素）を保存（Alphaがない場合は1.0を追加）
                            if len(rgba) >= 4:
                                current_color = rgba[:4]
                            elif len(rgba) == 3:
                                current_color = rgba + [1.0]  # Alpha=1.0を追加
                            else:
                                current_color = [1.0, 1.0, 1.0, 1.0]  # デフォルト値
                        elif mat_name and mat_name.startswith('#'):
                            # Hex color code
                            hex_color = mat_name[1:]
                            if len(hex_color) == 6:
                                r = int(hex_color[0:2], 16) / 255.0
                                g = int(hex_color[2:4], 16) / 255.0
                                b = int(hex_color[4:6], 16) / 255.0
                                current_color = [r, g, b, 1.0]  # Alpha=1.0を追加
                            else:
                                current_color = [1.0, 1.0, 1.0, 1.0]  # デフォルト値
                        else:
                            current_color = [1.0, 1.0, 1.0, 1.0]  # デフォルト値

                # Store data
                if is_main_visual:
                    if current_stl_path:
                        link_data['stl_file'] = current_stl_path
                    link_data['color'] = current_color
                    link_data['mesh_scale'] = mesh_scale
                    link_data['visual_origin'] = current_visual_origin
                else:
                    # Decoration visual
                    # メッシュファイルが見つからない場合でもdecoration_dataを作成
                    # ノード名を一意にするため、リンク名を含める
                    if current_stl_path:
                        stl_name = os.path.splitext(os.path.basename(current_stl_path))[0]
                    else:
                        # メッシュファイルが見つからない場合、ファイル名から推測
                        mesh_basename = os.path.basename(mesh_filename) if mesh_filename else f"decoration_{visual_idx}"
                        stl_name = os.path.splitext(mesh_basename)[0]
                    
                    # ノード名を一意にするため、リンク名を含める（同じメッシュが複数のリンクで使用される場合の衝突を回避）
                    decoration_name = f"{link_name}_{stl_name}"
                    
                    decoration_data = {
                        'name': decoration_name,
                        'stl_file': current_stl_path,  # Noneの可能性もある
                        'color': current_color,
                        'mesh_scale': mesh_scale,
                        'visual_origin': current_visual_origin,
                        'original_name': stl_name  # 元のメッシュ名を保持
                    }
                    link_data['decorations'].append(decoration_data)

            # Parse collision information (multiple colliders support)
            collision_elems = link_elem.findall('collision')
            for collision_elem in collision_elems:
                geometry_elem = collision_elem.find('geometry')
                if geometry_elem is not None:
                    # Parse origin (position and rotation)
                    origin_elem = collision_elem.find('origin')
                    pos = [0.0, 0.0, 0.0]
                    rpy = [0.0, 0.0, 0.0]
                    if origin_elem is not None:
                        xyz_str = origin_elem.get('xyz', '0 0 0')
                        pos = [float(v) for v in xyz_str.split()]
                        rpy_str = origin_elem.get('rpy', '0 0 0')
                        rpy_rad = [float(v) for v in rpy_str.split()]
                        rpy = [math.degrees(r) for r in rpy_rad]  # Convert to degrees
                    
                    # Check for mesh collision
                    mesh_elem = geometry_elem.find('mesh')
                    if mesh_elem is not None:
                        mesh_filename = mesh_elem.get('filename', '')
                        resolved_path = self._resolve_mesh_path(mesh_filename, urdf_file_path)
                        if resolved_path:
                            # Get scale from mesh element
                            scale_str = mesh_elem.get('scale', '1.0 1.0 1.0')
                            scale = [float(v) for v in scale_str.split()]
                            
                            # Add to colliders list
                            link_data['colliders'].append({
                                'type': 'mesh',
                                'enabled': True,
                                'data': None,
                                'mesh': resolved_path,
                                'mesh_scale': scale
                            })
                            
                            # Backward compatibility: store first mesh collider
                            if 'collision_mesh' not in link_data or link_data['collision_mesh'] is None:
                                link_data['collision_mesh'] = resolved_path
                                link_data['collider_type'] = 'mesh'
                                link_data['collider_enabled'] = True
                            continue
                    
                    # Check for primitive collision geometries
                    box_elem = geometry_elem.find('box')
                    sphere_elem = geometry_elem.find('sphere')
                    cylinder_elem = geometry_elem.find('cylinder')
                    
                    if box_elem is not None:
                        # Parse box collision
                        size_str = box_elem.get('size', '')
                        if size_str:
                            sizes = [float(s) for s in size_str.split()]
                            if len(sizes) >= 3:
                                collider_data = {
                                    'type': 'box',
                                    'geometry': {
                                        'size_x': sizes[0],
                                        'size_y': sizes[1],
                                        'size_z': sizes[2]
                                    },
                                    'position': pos,
                                    'rotation': rpy  # degrees
                                }
                                
                                # Add to colliders list
                                link_data['colliders'].append({
                                    'type': 'primitive',
                                    'enabled': True,
                                    'data': collider_data,
                                    'mesh': None,
                                    'mesh_scale': [1.0, 1.0, 1.0]
                                })
                                
                                # Backward compatibility: store first primitive collider
                                if 'collider_data' not in link_data or link_data['collider_data'] is None:
                                    link_data['collider_data'] = collider_data
                                    link_data['collider_type'] = 'primitive'
                                    link_data['collider_enabled'] = True
                    
                    elif sphere_elem is not None:
                        # Parse sphere collision
                        radius_str = sphere_elem.get('radius', '0.5')
                        radius = float(radius_str)
                        collider_data = {
                            'type': 'sphere',
                            'geometry': {
                                'radius': radius
                            },
                            'position': pos,
                            'rotation': rpy  # degrees
                        }
                        
                        # Add to colliders list
                        link_data['colliders'].append({
                            'type': 'primitive',
                            'enabled': True,
                            'data': collider_data,
                            'mesh': None,
                            'mesh_scale': [1.0, 1.0, 1.0]
                        })
                        
                        # Backward compatibility: store first primitive collider
                        if 'collider_data' not in link_data or link_data['collider_data'] is None:
                            link_data['collider_data'] = collider_data
                            link_data['collider_type'] = 'primitive'
                            link_data['collider_enabled'] = True
                    
                    elif cylinder_elem is not None:
                        # Parse cylinder collision
                        radius_str = cylinder_elem.get('radius', '0.5')
                        length_str = cylinder_elem.get('length', '1.0')
                        radius = float(radius_str)
                        length = float(length_str)
                        collider_data = {
                            'type': 'cylinder',
                            'geometry': {
                                'radius': radius,
                                'length': length
                            },
                            'position': pos,
                            'rotation': rpy  # degrees
                        }
                        
                        # Add to colliders list
                        link_data['colliders'].append({
                            'type': 'primitive',
                            'enabled': True,
                            'data': collider_data,
                            'mesh': None,
                            'mesh_scale': [1.0, 1.0, 1.0]
                        })
                        
                        # Backward compatibility: store first primitive collider
                        if 'collider_data' not in link_data or link_data['collider_data'] is None:
                            link_data['collider_data'] = collider_data
                            link_data['collider_type'] = 'primitive'
                            link_data['collider_enabled'] = True
                        
                        link_data['collider_data'] = collider_data
                        link_data['collider_type'] = 'primitive'
                        link_data['collider_enabled'] = True
                        break

            links_data[link_name] = link_data

        if self.verbose:
            print(f"\n[URDFParser] Successfully parsed {len(links_data)} links")
            if missing_meshes:
                print(f"  Warning: {len(missing_meshes)} mesh files could not be found")
                for missing in missing_meshes[:5]:  # 最初の5つだけ表示
                    print(f"    - {missing['link_name']}: {missing['basename']}")
                if len(missing_meshes) > 5:
                    print(f"    ... and {len(missing_meshes) - 5} more")

        return links_data, missing_meshes

    def _resolve_mesh_path(self, mesh_filename, urdf_file_path):
        """Resolve mesh file path from URDF reference.

        Tries multiple methods to find the mesh file:
        1. package:// paths relative to description directory
        2. Absolute paths
        3. Relative paths from URDF directory
        4. Search in common mesh directories

        Args:
            mesh_filename: Mesh file reference from URDF
            urdf_file_path: Path to URDF file

        Returns:
            Absolute path to mesh file, or None if not found
        """
        urdf_dir = os.path.dirname(os.path.abspath(urdf_file_path))
        urdf_file_abs = os.path.abspath(urdf_file_path)
        description_dir = os.path.dirname(urdf_dir)  # メソッドの最初で定義
        mesh_basename = os.path.basename(mesh_filename)

        # Handle package:// paths
        if mesh_filename.startswith('package://'):
            parts = mesh_filename.split('/')
            if len(parts) > 2:
                package_name = parts[2]
                relative_path = '/'.join(parts[3:]) if len(parts) > 3 else ''
                
                # パッケージルートを探す（xacroファイルの場所から）
                package_root = find_package_root(urdf_file_abs, package_name, max_search_depth=10, verbose=self.verbose)
                
                if package_root:
                    # パッケージルートからの相対パス
                    if relative_path:
                        candidate = os.path.join(package_root, relative_path)
                    else:
                        candidate = os.path.join(package_root, mesh_basename)
                    
                    if os.path.exists(candidate):
                        if self.verbose:
                            print(f"  Found mesh (package://): {candidate}")
                        return candidate
                
                # パッケージルートが見つからない場合、従来の方法を試す
                if relative_path:
                    relative_path_full = relative_path
                    # relative_pathから"robots/"プレフィックスを削除（例: "robots/asr_twodof_description/qb_meshes/dae/qb_base_flange_m.dae" -> "asr_twodof_description/qb_meshes/dae/qb_base_flange_m.dae"）
                    if relative_path_full.startswith('robots/'):
                        relative_path_without_robots = relative_path_full[7:]  # "robots/"を削除
                    else:
                        relative_path_without_robots = relative_path_full
                else:
                    relative_path_full = os.path.join('meshes', mesh_basename)
                    relative_path_without_robots = relative_path_full

                # Try multiple locations
                candidates = [
                    os.path.join(package_root, relative_path_full) if package_root else None,
                    os.path.join(description_dir, relative_path_full),
                    os.path.join(description_dir, relative_path_without_robots),  # "robots/"なしのパスも試す
                    os.path.join(urdf_dir, relative_path_full),
                    os.path.join(urdf_dir, relative_path_without_robots),  # "robots/"なしのパスも試す
                    os.path.join(description_dir, 'meshes', mesh_basename),
                    os.path.normpath(os.path.join(urdf_dir, '..', 'meshes', mesh_basename)),
                    os.path.join(os.path.dirname(description_dir), relative_path_full),
                    os.path.join(os.path.dirname(description_dir), relative_path_without_robots),  # "robots/"なしのパスも試す
                    # xacroファイルと同じディレクトリ構造を試す
                    os.path.normpath(os.path.join(urdf_dir, '..', '..', 'meshes', mesh_basename)),
                    os.path.normpath(os.path.join(urdf_dir, '..', '..', package_name, 'meshes', mesh_basename)) if package_name else None,
                    # 相対パスから直接ファイル名を抽出して検索
                    os.path.join(description_dir, mesh_basename),
                    os.path.join(urdf_dir, mesh_basename),
                ]

                for candidate in candidates:
                    if candidate and os.path.exists(candidate):
                        if self.verbose:
                            print(f"  Found mesh (package:// fallback): {candidate}")
                        return candidate

        else:
            # Absolute or relative paths
            candidates = []
            
            # Try absolute path
            if os.path.isabs(mesh_filename) and os.path.exists(mesh_filename):
                return mesh_filename

            # Try relative paths
            candidates = [
                os.path.join(urdf_dir, mesh_filename),
                os.path.join(description_dir, mesh_filename),
                os.path.join(description_dir, 'meshes', mesh_basename),
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    if self.verbose:
                        print(f"  Found mesh: {candidate}")
                    return candidate

        # メッシュが見つからない場合、一つ階層を上り、そこから下位4階層まで探索
        if self.verbose:
            print(f"  Mesh not found in standard locations, searching from parent directory (4 levels deep)...")
        
        # URDFファイルのディレクトリの親ディレクトリを取得
        parent_dir = os.path.dirname(urdf_dir)
        
        if os.path.isdir(parent_dir):
            # 親ディレクトリから4階層まで探索
            parent_dir_depth = len(parent_dir.split(os.sep))
            for root_dir, dirs, files in os.walk(parent_dir):
                # 現在のディレクトリの深さを計算
                current_depth = len(root_dir.split(os.sep)) - parent_dir_depth
                # 4階層まで探索
                if current_depth > 4:
                    dirs[:] = []  # これより深い探索を停止
                    continue
                
                # メッシュファイルのbasenameと一致するファイルを探す
                if mesh_basename in files:
                    candidate = os.path.join(root_dir, mesh_basename)
                    if os.path.exists(candidate):
                        if self.verbose:
                            print(f"  Found mesh (parent directory search, depth {current_depth}): {candidate}")
                        return candidate

        return None

    def _parse_joints(self, root, links_data):
        """Parse joint information from URDF.

        Args:
            root: XML root element
            links_data: Dictionary of link data

        Returns:
            List of joint data dictionaries
        """
        joints_data = []

        # デバッグ: ジョイント要素の数を確認
        joint_elems = root.findall('joint')
        if self.verbose:
            print(f"\n[URDFParser] Found {len(joint_elems)} joint elements in URDF")

        for joint_elem in joint_elems:
            joint_data = {
                'name': joint_elem.get('name'),
                'type': joint_elem.get('type', 'fixed'),
                'parent': None,
                'child': None,
                'origin_xyz': [0.0, 0.0, 0.0],
                'origin_rpy': [0.0, 0.0, 0.0],
                'axis': [1.0, 0.0, 0.0],
                'limit': {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 'velocity': 3.0, 'friction': 0.05},
                'dynamics': {'damping': 0.0, 'friction': 0.0}
            }

            # Parse parent and child links
            parent_elem = joint_elem.find('parent')
            if parent_elem is not None:
                joint_data['parent'] = parent_elem.get('link')

            child_elem = joint_elem.find('child')
            if child_elem is not None:
                joint_data['child'] = child_elem.get('link')

            # Parse origin
            origin_elem = joint_elem.find('origin')
            if origin_elem is not None:
                xyz_str = origin_elem.get('xyz', '0 0 0')
                rpy_str = origin_elem.get('rpy', '0 0 0')
                joint_data['origin_xyz'] = [float(v) for v in xyz_str.split()]
                joint_data['origin_rpy'] = [float(v) for v in rpy_str.split()]

            # Parse axis
            axis_elem = joint_elem.find('axis')
            if axis_elem is not None:
                axis_str = axis_elem.get('xyz', '1 0 0')
                joint_data['axis'] = [float(v) for v in axis_str.split()]

            # Parse limit
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                joint_data['limit']['lower'] = float(limit_elem.get('lower', -3.14159))
                joint_data['limit']['upper'] = float(limit_elem.get('upper', 3.14159))
                joint_data['limit']['effort'] = float(limit_elem.get('effort', 10.0))
                joint_data['limit']['velocity'] = float(limit_elem.get('velocity', 3.0))
                joint_data['limit']['friction'] = float(limit_elem.get('friction', 0.05))

            # Parse dynamics
            dynamics_elem = joint_elem.find('dynamics')
            if dynamics_elem is not None:
                joint_data['dynamics']['damping'] = float(dynamics_elem.get('damping', 0.0))
                joint_data['dynamics']['friction'] = float(dynamics_elem.get('friction', 0.0))

            joints_data.append(joint_data)

        if self.verbose:
            print(f"[URDFParser] Successfully parsed {len(joints_data)} joints")

        # Detect root links and connect to base_link
        if self.verbose:
            print("\n=== Detecting root links ===")

        child_links = set()
        for joint_data in joints_data:
            if joint_data['child']:
                child_links.add(joint_data['child'])

        root_links = []
        for link_name in links_data.keys():
            if link_name not in child_links and link_name not in ['base_link', 'BaseLink']:
                root_links.append(link_name)
                if self.verbose:
                    print(f"  Found root link: {link_name}")

        # Create synthetic joints from base_link to root links
        if root_links:
            if self.verbose:
                print(f"  Connecting {len(root_links)} root link(s) to base_link")

            for root_link_name in root_links:
                synthetic_joint = {
                    'name': f'base_to_{root_link_name}',
                    'type': 'fixed',
                    'parent': 'base_link',
                    'child': root_link_name,
                    'origin_xyz': [0.0, 0.0, 0.0],
                    'origin_rpy': [0.0, 0.0, 0.0],
                    'axis': [1.0, 0.0, 0.0],
                    'limit': {'lower': 0.0, 'upper': 0.0, 'effort': 0.0, 'velocity': 0.0, 'friction': 0.0},
                    'dynamics': {'damping': 0.0, 'friction': 0.0}
                }
                joints_data.append(synthetic_joint)
                if self.verbose:
                    print(f"  Created synthetic joint: base_link -> {root_link_name}")

        if self.verbose:
            print(f"Parsed {len(joints_data)} joints (including synthetic joints)")

        return joints_data


# ============================================================================
# SRDF PARSER
# ============================================================================

class SRDFParser:
    """Parser for SRDF (Semantic Robot Description Format) files.

    This class provides methods to parse SRDF XML files and extract
    semantic information including groups, disabled collisions,
    end effectors, and virtual joints.

    Usage:
        parser = SRDFParser()
        result = parser.parse_srdf('/path/to/robot.srdf')
        
        # Access parsed data
        groups = result['groups']
        disabled_collisions = result['disabled_collisions']
        end_effectors = result['end_effectors']
    """

    def __init__(self, verbose=True):
        """Initialize SRDF parser.

        Args:
            verbose: If True, print parsing progress and information
        """
        self.verbose = verbose

    def parse_srdf(self, srdf_file_path):
        """Parse SRDF file and extract semantic information.

        Args:
            srdf_file_path: Path to SRDF XML file

        Returns:
            Dictionary containing:
                - groups: List of group definitions
                - disabled_collisions: List of disabled collision pairs
                - disabled_links: List of disabled link names
                - disabled_joints: List of disabled joint names
                - end_effectors: List of end effector definitions
                - virtual_joints: List of virtual joint definitions

        Raises:
            FileNotFoundError: If SRDF file does not exist
            ET.ParseError: If SRDF file is not valid XML
            ValueError: If root element is not 'robot'
        """
        if not os.path.exists(srdf_file_path):
            raise FileNotFoundError(f"SRDF file not found: {srdf_file_path}")

        if self.verbose:
            print(f"Parsing SRDF from: {srdf_file_path}")

        # Parse SRDF file
        tree = ET.parse(srdf_file_path)
        root = tree.getroot()

        if root.tag != 'robot':
            raise ValueError("Root element must be 'robot' for valid SRDF file")

        robot_name = root.get('name', 'robot')
        if self.verbose:
            print(f"Robot name: {robot_name}")

        # Parse groups
        groups = self._parse_groups(root)

        # Parse disabled collisions
        disabled_collisions = self._parse_disabled_collisions(root)

        # Parse disabled links
        disabled_links = self._parse_disabled_links(root)

        # Parse disabled joints
        disabled_joints = self._parse_disabled_joints(root)

        # Parse end effectors
        end_effectors = self._parse_end_effectors(root)

        # Parse virtual joints
        virtual_joints = self._parse_virtual_joints(root)

        if self.verbose:
            print(f"Parsed {len(groups)} groups, {len(disabled_collisions)} disabled collisions, "
                  f"{len(disabled_links)} disabled links, {len(disabled_joints)} disabled joints, "
                  f"{len(end_effectors)} end effectors, {len(virtual_joints)} virtual joints")

        return {
            'robot_name': robot_name,
            'groups': groups,
            'disabled_collisions': disabled_collisions,
            'disabled_links': disabled_links,
            'disabled_joints': disabled_joints,
            'end_effectors': end_effectors,
            'virtual_joints': virtual_joints
        }


class SDFParser:
    """Parser for SDF (Simulation Description Format) files.
    
    This class provides methods to parse SDF XML files and convert them
    to URDF-like format for compatibility with the URDF parser.
    
    Usage:
        parser = SDFParser()
        result = parser.parse_sdf('/path/to/robot.sdf')
    """
    
    def __init__(self, verbose=True):
        """Initialize SDF parser.
        
        Args:
            verbose: If True, print parsing progress and information
        """
        self.verbose = verbose
    
    def parse_sdf(self, sdf_file_path):
        """Parse SDF file and convert to URDF-like format.
        
        Args:
            sdf_file_path: Path to SDF XML file
            
        Returns:
            Dictionary containing:
                - robot_name: Name of the robot/model
                - links_data: Dictionary of link data (URDF format)
                - joints_data: List of joint data (URDF format)
                - materials_data: Dictionary of material data
                - missing_meshes: List of missing mesh files
                
        Raises:
            FileNotFoundError: If SDF file does not exist
            ET.ParseError: If SDF file is not valid XML
            ValueError: If root element is not 'sdf'
        """
        if not os.path.exists(sdf_file_path):
            raise FileNotFoundError(f"SDF file not found: {sdf_file_path}")
        
        if self.verbose:
            print(f"Parsing SDF from: {sdf_file_path}")
        
        # Parse SDF file
        tree = ET.parse(sdf_file_path)
        root = tree.getroot()
        
        # SDFファイルのルート要素は'sdf'または'gazebo'
        if root.tag not in ['sdf', 'gazebo']:
            raise ValueError(f"Root element must be 'sdf' or 'gazebo' for valid SDF file, got '{root.tag}'")
        
        # SDFバージョンを取得
        sdf_version = root.get('version', '1.0')
        if self.verbose:
            print(f"SDF version: {sdf_version}")
        
        # <model>要素を探す
        model_elem = root.find('model')
        if model_elem is None:
            # <world>要素の中に<model>がある場合
            world_elem = root.find('world')
            if world_elem is not None:
                model_elem = world_elem.find('model')
        
        if model_elem is None:
            raise ValueError("No <model> element found in SDF file")
        
        robot_name = model_elem.get('name', 'robot')
        if self.verbose:
            print(f"Model name: {robot_name}")
        
        # SDFをURDF形式に変換するために、URDFParserを使用
        # まず、SDFを一時的にURDF形式に変換
        # 注: これは簡易的な実装で、完全なSDF→URDF変換ではない
        # より完全な実装には、sdf2urdfツールの使用を推奨
        
        # SDFの<link>と<joint>をパース
        links_data = {}
        joints_data = []
        closed_loop_joints = []  # 閉リンクジョイント（ball, gearbox, screw）を別管理
        materials_data = {}
        missing_meshes = []
        
        # Parse links
        # まず、リンクのpose情報を保存するための辞書を作成
        link_poses = {}  # {link_name: {'xyz': [...], 'rpy': [...]}}
        
        for link_elem in model_elem.findall('link'):
            link_name = link_elem.get('name')
            if not link_name:
                continue
            
            # Parse link pose (モデル座標系での位置)
            link_pose = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
            pose_elem = link_elem.find('pose')
            if pose_elem is not None:
                pose_str = pose_elem.text
                if pose_str:
                    pose_values = [float(v) for v in pose_str.split()]
                    if len(pose_values) >= 3:
                        link_pose['xyz'] = pose_values[:3]
                    if len(pose_values) >= 6:
                        link_pose['rpy'] = pose_values[3:6]
            link_poses[link_name] = link_pose
            
            link_data = {
                'name': link_name,
                'mass': 0.0,
                'inertia': {'ixx': 0.0, 'iyy': 0.0, 'izz': 0.0, 'ixy': 0.0, 'ixz': 0.0, 'iyz': 0.0},
                'inertial_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                'stl_file': None,
                'stl_filename_original': None,  # 元のメッシュファイル名（見つからない場合に使用）
                'color': [1.0, 1.0, 1.0, 1.0],
                'mesh_scale': [1.0, 1.0, 1.0],
                'visual_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
                'decorations': [],
                'collider_type': None,
                'collider_enabled': False,
                'collider_data': None,
                'collision_mesh': None,  # Deprecated, use colliders list
                'colliders': [],  # List of collider dictionaries
                # SDF <collision><geometry><mesh><scale> ... </scale>
                # Keep separate from visual mesh_scale.
                'collision_mesh_scale': [1.0, 1.0, 1.0]
            }
            
            # Parse inertial
            inertial_elem = link_elem.find('inertial')
            if inertial_elem is not None:
                # Parse pose (inertial origin)
                pose_elem = inertial_elem.find('pose')
                if pose_elem is not None:
                    pose_str = pose_elem.text
                    if pose_str:
                        pose_values = [float(v) for v in pose_str.split()]
                        if len(pose_values) >= 3:
                            link_data['inertial_origin']['xyz'] = pose_values[:3]
                        if len(pose_values) >= 6:
                            link_data['inertial_origin']['rpy'] = pose_values[3:6]
                
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    link_data['mass'] = float(mass_elem.get('value', 0.0))
                
                inertia_elem = inertial_elem.find('inertia')
                if inertia_elem is not None:
                    link_data['inertia'] = {
                        'ixx': float(inertia_elem.get('ixx', 0.0)),
                        'iyy': float(inertia_elem.get('iyy', 0.0)),
                        'izz': float(inertia_elem.get('izz', 0.0)),
                        'ixy': float(inertia_elem.get('ixy', 0.0)),
                        'ixz': float(inertia_elem.get('ixz', 0.0)),
                        'iyz': float(inertia_elem.get('iyz', 0.0))
                    }
            
            # Parse visual
            visual_elems = link_elem.findall('visual')
            for visual_idx, visual_elem in enumerate(visual_elems):
                # Parse pose (visual origin)
                visual_origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
                pose_elem = visual_elem.find('pose')
                if pose_elem is not None:
                    pose_str = pose_elem.text
                    if pose_str:
                        pose_values = [float(v) for v in pose_str.split()]
                        if len(pose_values) >= 3:
                            visual_origin['xyz'] = pose_values[:3]
                        if len(pose_values) >= 6:
                            visual_origin['rpy'] = pose_values[3:6]
                
                # Parse material (color)
                color = [1.0, 1.0, 1.0, 1.0]
                material_elem = visual_elem.find('material')
                if material_elem is not None:
                    ambient_elem = material_elem.find('ambient')
                    if ambient_elem is not None:
                        ambient_str = ambient_elem.text
                        if ambient_str:
                            ambient_values = [float(v) for v in ambient_str.split()]
                            if len(ambient_values) >= 3:
                                color = ambient_values[:3] + [1.0] if len(ambient_values) < 4 else ambient_values[:4]
                
                geometry_elem = visual_elem.find('geometry')
                if geometry_elem is not None:
                    mesh_elem = geometry_elem.find('mesh')
                    if mesh_elem is not None:
                        # Parse scale
                        mesh_scale = [1.0, 1.0, 1.0]
                        scale_elem = mesh_elem.find('scale')
                        if scale_elem is not None:
                            scale_str = scale_elem.text
                            if scale_str:
                                scale_values = [float(v) for v in scale_str.split()]
                                if len(scale_values) >= 3:
                                    mesh_scale = scale_values[:3]
                        
                        uri_elem = mesh_elem.find('uri')
                        if uri_elem is not None:
                            mesh_filename = uri_elem.text
                            if mesh_filename:
                                # package://パスを解決
                                resolved_path = self._resolve_mesh_path(mesh_filename, sdf_file_path)
                                if resolved_path:
                                    if visual_idx == 0:
                                        link_data['stl_file'] = resolved_path
                                        link_data['visual_origin'] = visual_origin
                                        link_data['mesh_scale'] = mesh_scale
                                        link_data['color'] = color
                                    else:
                                        # Decoration
                                        link_data['decorations'].append({
                                            'name': f"{link_name}_{os.path.splitext(os.path.basename(mesh_filename))[0]}",
                                            'stl_file': resolved_path,
                                            'color': color,
                                            'mesh_scale': mesh_scale,
                                            'visual_origin': visual_origin
                                        })
                                else:
                                    # メッシュファイルが見つからない場合、stl_filename_originalを設定
                                    if visual_idx == 0:
                                        link_data['stl_filename_original'] = mesh_filename
                                        link_data['visual_origin'] = visual_origin
                                        link_data['mesh_scale'] = mesh_scale
                                        link_data['color'] = color
                                    missing_meshes.append({
                                        'link_name': link_name,
                                        'filename': mesh_filename,
                                        'basename': os.path.basename(mesh_filename),
                                        'is_decoration': visual_idx > 0
                                    })

            # Parse collision
            collision_elems = link_elem.findall('collision')
            for collision_elem in collision_elems:
                geometry_elem = collision_elem.find('geometry')
                if geometry_elem is not None:
                    # Check for mesh collision
                    mesh_elem = geometry_elem.find('mesh')
                    if mesh_elem is not None:
                        # Parse collision mesh scale (optional)
                        collision_mesh_scale = [1.0, 1.0, 1.0]
                        scale_elem = mesh_elem.find('scale')
                        if scale_elem is not None and scale_elem.text:
                            try:
                                scale_values = [float(v) for v in scale_elem.text.split()]
                                if len(scale_values) >= 3:
                                    collision_mesh_scale = scale_values[:3]
                            except Exception:
                                # Keep default on parse failure
                                pass

                        uri_elem = mesh_elem.find('uri')
                        if uri_elem is not None:
                            mesh_filename = uri_elem.text
                            if mesh_filename:
                                resolved_path = self._resolve_mesh_path(mesh_filename, sdf_file_path)
                                if resolved_path:
                                    link_data['collision_mesh'] = resolved_path
                                    link_data['collision_mesh_scale'] = collision_mesh_scale
                                    link_data['collider_type'] = 'mesh'
                                    link_data['collider_enabled'] = True
                                    break

                    # Check for primitive collision geometries
                    box_elem = geometry_elem.find('box')
                    sphere_elem = geometry_elem.find('sphere')
                    cylinder_elem = geometry_elem.find('cylinder')
                    capsule_elem = geometry_elem.find('capsule')

                    if box_elem is not None:
                        # Parse box collision
                        size_elem = box_elem.find('size')
                        if size_elem is not None:
                            size_str = size_elem.text
                            if size_str:
                                sizes = [float(s) for s in size_str.split()]
                                if len(sizes) >= 3:
                                    collider_data = {
                                        'type': 'box',
                                        'geometry': {
                                            'size_x': sizes[0],
                                            'size_y': sizes[1],
                                            'size_z': sizes[2]
                                        },
                                        'position': [0.0, 0.0, 0.0],
                                        'rotation': [0.0, 0.0, 0.0]
                                    }

                                    # Parse collision pose
                                    pose_elem = collision_elem.find('pose')
                                    if pose_elem is not None:
                                        pose_str = pose_elem.text
                                        if pose_str:
                                            pose_values = [float(v) for v in pose_str.split()]
                                            if len(pose_values) >= 3:
                                                collider_data['position'] = pose_values[:3]
                                            if len(pose_values) >= 6:
                                                # RPY in radians, convert to degrees
                                                rpy_rad = pose_values[3:6]
                                                collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]

                                    link_data['collider_data'] = collider_data
                                    link_data['collider_type'] = 'primitive'
                                    link_data['collider_enabled'] = True
                                    break

                    elif sphere_elem is not None:
                        # Parse sphere collision
                        radius_elem = sphere_elem.find('radius')
                        if radius_elem is not None:
                            radius_str = radius_elem.text
                            if radius_str:
                                radius = float(radius_str)
                                collider_data = {
                                    'type': 'sphere',
                                    'geometry': {
                                        'radius': radius
                                    },
                                    'position': [0.0, 0.0, 0.0],
                                    'rotation': [0.0, 0.0, 0.0]
                                }

                                # Parse collision pose
                                pose_elem = collision_elem.find('pose')
                                if pose_elem is not None:
                                    pose_str = pose_elem.text
                                    if pose_str:
                                        pose_values = [float(v) for v in pose_str.split()]
                                        if len(pose_values) >= 3:
                                            collider_data['position'] = pose_values[:3]
                                        if len(pose_values) >= 6:
                                            rpy_rad = pose_values[3:6]
                                            collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]

                                link_data['collider_data'] = collider_data
                                link_data['collider_type'] = 'primitive'
                                link_data['collider_enabled'] = True
                                break

                    elif cylinder_elem is not None:
                        # Parse cylinder collision
                        radius_elem = cylinder_elem.find('radius')
                        length_elem = cylinder_elem.find('length')
                        if radius_elem is not None and length_elem is not None:
                            radius_str = radius_elem.text
                            length_str = length_elem.text
                            if radius_str and length_str:
                                radius = float(radius_str)
                                length = float(length_str)
                                collider_data = {
                                    'type': 'cylinder',
                                    'geometry': {
                                        'radius': radius,
                                        # NOTE:
                                        # The rest of the toolchain (3D viewer / URDF export) expects 'length'
                                        # for cylinder/capsule primitives. Using 'height' here caused the viewer
                                        # to fall back to default length=1.0, making the collider appear too long.
                                        'length': length
                                    },
                                    'position': [0.0, 0.0, 0.0],
                                    'rotation': [0.0, 0.0, 0.0]
                                }

                                # Parse collision pose
                                pose_elem = collision_elem.find('pose')
                                if pose_elem is not None:
                                    pose_str = pose_elem.text
                                    if pose_str:
                                        pose_values = [float(v) for v in pose_str.split()]
                                        if len(pose_values) >= 3:
                                            collider_data['position'] = pose_values[:3]
                                        if len(pose_values) >= 6:
                                            rpy_rad = pose_values[3:6]
                                            collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]

                                link_data['collider_data'] = collider_data
                                link_data['collider_type'] = 'primitive'
                                link_data['collider_enabled'] = True
                                break

                    elif capsule_elem is not None:
                        # Parse capsule collision
                        radius_elem = capsule_elem.find('radius')
                        length_elem = capsule_elem.find('length')
                        if radius_elem is not None and length_elem is not None:
                            radius_str = radius_elem.text
                            length_str = length_elem.text
                            if radius_str and length_str:
                                radius = float(radius_str)
                                length = float(length_str)
                                collider_data = {
                                    'type': 'capsule',
                                    'geometry': {
                                        'radius': radius,
                                        # Keep key consistent with viewer/exporter expectations (see cylinder note).
                                        'length': length
                                    },
                                    'position': [0.0, 0.0, 0.0],
                                    'rotation': [0.0, 0.0, 0.0]
                                }

                                # Parse collision pose
                                pose_elem = collision_elem.find('pose')
                                if pose_elem is not None:
                                    pose_str = pose_elem.text
                                    if pose_str:
                                        pose_values = [float(v) for v in pose_str.split()]
                                        if len(pose_values) >= 3:
                                            collider_data['position'] = pose_values[:3]
                                        if len(pose_values) >= 6:
                                            rpy_rad = pose_values[3:6]
                                            collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]

                                link_data['collider_data'] = collider_data
                                link_data['collider_type'] = 'primitive'
                                link_data['collider_enabled'] = True
                                break

            links_data[link_name] = link_data
        
        # Parse joints
        for joint_elem in model_elem.findall('joint'):
            joint_name = joint_elem.get('name')
            if not joint_name:
                continue
            
            joint_type = joint_elem.get('type', 'fixed')

            # 閉リンクジョイント（ball, gearbox, screw）かどうかをチェック
            is_closed_loop = joint_type in ['ball', 'gearbox', 'screw']

            # SDFのjoint typeをURDFのjoint typeに変換
            # gearbox, ball, screwなどの特殊なタイプは閉リンクとして別管理
            type_mapping = {
                'revolute': 'revolute',
                'prismatic': 'prismatic',
                'fixed': 'fixed',
                'continuous': 'continuous',
                'ball': 'ball',      # 閉リンクとして保持
                'gearbox': 'gearbox',  # 閉リンクとして保持
                'screw': 'screw'     # 閉リンクとして保持
            }
            urdf_joint_type = type_mapping.get(joint_type, 'fixed')

            # gearboxタイプのジョイントの場合、gearbox_reference_bodyとgearbox_ratioを保存
            gearbox_reference_body = None
            gearbox_ratio = 1.0
            if joint_type == 'gearbox':
                gearbox_ref_elem = joint_elem.find('gearbox_reference_body')
                if gearbox_ref_elem is not None:
                    gearbox_reference_body = gearbox_ref_elem.text
                    if self.verbose:
                        print(f"  Joint '{joint_name}' (gearbox): reference_body = {gearbox_reference_body}")

                # gearbox_ratioを取得
                gearbox_ratio_elem = joint_elem.find('gearbox_ratio')
                if gearbox_ratio_elem is not None and gearbox_ratio_elem.text:
                    gearbox_ratio = float(gearbox_ratio_elem.text)
            
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            
            parent_link = parent_elem.text if parent_elem is not None else None
            child_link = child_elem.text if child_elem is not None else None
            
            # Parse origin
            origin_xyz = [0.0, 0.0, 0.0]
            origin_rpy = [0.0, 0.0, 0.0]
            pose_elem = joint_elem.find('pose')
            if pose_elem is not None:
                # SDF の joint <pose> は child link の座標系で表現される
                # URDF の joint <origin> は parent link の座標系で表現されるため、変換が必要
                pose_str = pose_elem.text
                if pose_str:
                    pose_values = [float(v) for v in pose_str.split()]
                    joint_xyz_in_child = [0.0, 0.0, 0.0]
                    joint_rpy_in_child = [0.0, 0.0, 0.0]
                    if len(pose_values) >= 3:
                        joint_xyz_in_child = pose_values[:3]
                    if len(pose_values) >= 6:
                        joint_rpy_in_child = pose_values[3:6]

                    # Child frame -> Parent frame への変換
                    parent_in_poses = parent_link in link_poses
                    child_in_poses = child_link in link_poses
                    if self.verbose:
                        print(f"  Joint '{joint_name}' has pose in child frame: xyz={joint_xyz_in_child}, rpy={joint_rpy_in_child}")
                        print(f"    Parent '{parent_link}' in link_poses: {parent_in_poses}")
                        print(f"    Child '{child_link}' in link_poses: {child_in_poses}")

                    if parent_link and child_link and parent_in_poses and child_in_poses:
                        import numpy as np

                        # 回転行列変換関数（既存のものを再利用）
                        def rpy_to_matrix(rpy):
                            """RPY（Roll-Pitch-Yaw）から回転行列を作成"""
                            r, p, y = rpy
                            cx, sx = np.cos(r), np.sin(r)
                            cy, sy = np.cos(p), np.sin(p)
                            cz, sz = np.cos(y), np.sin(y)
                            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                            return Rz @ Ry @ Rx

                        def matrix_to_rpy(R):
                            """回転行列からRPY（Roll-Pitch-Yaw）を抽出"""
                            sy = R[2, 0]
                            cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                            if cy > 1e-6:
                                roll = np.arctan2(R[2, 1], R[2, 2])
                                pitch = np.arctan2(-R[2, 0], cy)
                                yaw = np.arctan2(R[1, 0], R[0, 0])
                            else:
                                roll = np.arctan2(-R[1, 2], R[1, 1])
                                pitch = np.arctan2(-R[2, 0], cy)
                                yaw = 0
                            return [roll, pitch, yaw]

                        parent_pose = link_poses[parent_link]
                        child_pose = link_poses[child_link]

                        # 1. Joint の child frame での pose を child link のワールド pose に合成
                        R_child = rpy_to_matrix(child_pose['rpy'])
                        R_joint_in_child = rpy_to_matrix(joint_rpy_in_child)
                        R_joint_world = R_child @ R_joint_in_child

                        joint_xyz_world = np.array(child_pose['xyz']) + R_child @ np.array(joint_xyz_in_child)

                        # 2. Joint のワールド pose を parent link のローカル座標に変換
                        R_parent = rpy_to_matrix(parent_pose['rpy'])
                        R_joint_in_parent = R_parent.T @ R_joint_world
                        origin_rpy = matrix_to_rpy(R_joint_in_parent)

                        world_diff = joint_xyz_world - np.array(parent_pose['xyz'])
                        local_diff = R_parent.T @ world_diff
                        origin_xyz = local_diff.tolist()

                        # デバッグ出力
                        print(f"\n=== SDF Joint Origin Calculation: '{joint_name}' (from joint pose) ===")
                        print(f"  Parent link: '{parent_link}'")
                        print(f"    World pose XYZ: {parent_pose['xyz']}")
                        print(f"    World pose RPY (rad): {parent_pose['rpy']}")
                        print(f"  Child link: '{child_link}'")
                        print(f"    World pose XYZ: {child_pose['xyz']}")
                        print(f"    World pose RPY (rad): {child_pose['rpy']}")
                        print(f"  Joint pose in child frame:")
                        print(f"    XYZ: {joint_xyz_in_child}")
                        print(f"    RPY (rad): {joint_rpy_in_child}")
                        print(f"  Calculated joint origin (parent's local frame):")
                        print(f"    XYZ: {origin_xyz}")
                        print(f"    RPY (rad): {origin_rpy}")
                        print("=== End SDF Joint Origin Calculation ===\n")
                    else:
                        # Link poses がない場合は、そのまま使用（フォールバック）
                        origin_xyz = joint_xyz_in_child
                        origin_rpy = joint_rpy_in_child
            else:
                # <joint>に<pose>がない場合、親リンクと子リンクの<pose>の差から計算
                if parent_link and child_link and parent_link in link_poses and child_link in link_poses:
                    parent_pose = link_poses[parent_link]
                    child_pose = link_poses[child_link]
                    # 親の回転行列を計算してワールド座標からローカル座標に変換
                    import numpy as np

                    # 親のRPYから回転行列を作成
                    def rpy_to_matrix(rpy):
                        """RPY（Roll-Pitch-Yaw）から回転行列を作成"""
                        r, p, y = rpy
                        # Rz(yaw) * Ry(pitch) * Rx(roll)
                        cx, sx = np.cos(r), np.sin(r)
                        cy, sy = np.cos(p), np.sin(p)
                        cz, sz = np.cos(y), np.sin(y)

                        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

                        return Rz @ Ry @ Rx

                    # 回転行列からRPYを抽出
                    def matrix_to_rpy(R):
                        """回転行列からRPY（Roll-Pitch-Yaw）を抽出"""
                        # R = Rz * Ry * Rx の形式からRPYを逆算
                        sy = R[2, 0]
                        cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

                        if cy > 1e-6:  # 特異点でない場合
                            roll = np.arctan2(R[2, 1], R[2, 2])
                            pitch = np.arctan2(-R[2, 0], cy)
                            yaw = np.arctan2(R[1, 0], R[0, 0])
                        else:  # ジンバルロック
                            roll = np.arctan2(-R[1, 2], R[1, 1])
                            pitch = np.arctan2(-R[2, 0], cy)
                            yaw = 0

                        return [roll, pitch, yaw]

                    # 親と子のワールド座標での回転行列
                    R_parent = rpy_to_matrix(parent_pose['rpy'])
                    R_child = rpy_to_matrix(child_pose['rpy'])

                    # 親のローカル座標系での子の回転: R_relative = R_parent^T * R_child
                    R_relative = R_parent.T @ R_child

                    # 相対回転行列からRPYを抽出
                    origin_rpy = matrix_to_rpy(R_relative)

                    # 親のローカル座標系での位置の差を計算
                    # ワールド座標での位置の差
                    world_diff = np.array([
                        child_pose['xyz'][0] - parent_pose['xyz'][0],
                        child_pose['xyz'][1] - parent_pose['xyz'][1],
                        child_pose['xyz'][2] - parent_pose['xyz'][2]
                    ])
                    # 親のローカル座標系に変換
                    local_diff = R_parent.T @ world_diff
                    origin_xyz = local_diff.tolist()

                    # デバッグ出力（常に表示）
                    print(f"\n=== SDF Joint Origin Calculation: '{joint_name}' ===")
                    print(f"  Parent link: '{parent_link}'")
                    print(f"    World pose XYZ: {parent_pose['xyz']}")
                    print(f"    World pose RPY (rad): {parent_pose['rpy']}")
                    print(f"  Child link: '{child_link}'")
                    print(f"    World pose XYZ: {child_pose['xyz']}")
                    print(f"    World pose RPY (rad): {child_pose['rpy']}")
                    print(f"  Calculated joint origin (parent's local frame):")
                    print(f"    XYZ: {origin_xyz}")
                    print(f"    RPY (rad): {origin_rpy}")
                    print("=== End SDF Joint Origin Calculation ===\n")
            
            # Parse axis
            axis = [1.0, 0.0, 0.0]
            axis_elem = joint_elem.find('axis')
            if axis_elem is not None:
                xyz_elem = axis_elem.find('xyz')
                if xyz_elem is not None:
                    axis_str = xyz_elem.text
                    if axis_str:
                        axis = [float(v) for v in axis_str.split()]
            
            # Parse limits
            limit = {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 'velocity': 3.0, 'friction': 0.05}
            if axis_elem is not None:
                limit_elem = axis_elem.find('limit')
                if limit_elem is not None:
                    lower = limit_elem.find('lower')
                    upper = limit_elem.find('upper')
                    effort = limit_elem.find('effort')
                    velocity = limit_elem.find('velocity')
                    
                    if lower is not None:
                        limit['lower'] = float(lower.text) if lower.text else -3.14159
                    if upper is not None:
                        limit['upper'] = float(upper.text) if upper.text else 3.14159
                    if effort is not None:
                        limit['effort'] = float(effort.text) if effort.text else 10.0
                    if velocity is not None:
                        limit['velocity'] = float(velocity.text) if velocity.text else 3.0
            
            joint_data = {
                'name': joint_name,
                'type': urdf_joint_type,
                'parent': parent_link,
                'child': child_link,
                'origin_xyz': origin_xyz,
                'origin_rpy': origin_rpy,
                'axis': axis,
                'limit': limit,
                'dynamics': {'damping': 0.0, 'friction': 0.0},
                'gearbox_reference_body': gearbox_reference_body,  # gearboxタイプの場合のみ設定
                'gearbox_ratio': gearbox_ratio  # gearboxタイプの場合のみ意味を持つ
            }

            # 閉リンクジョイントは別リストに保存、通常ジョイントはjoints_dataに追加
            if is_closed_loop:
                # original_typeフィールドを追加して元のタイプを保存
                joint_data['original_type'] = joint_type
                closed_loop_joints.append(joint_data)
                if self.verbose:
                    print(f"  Closed-loop joint detected: '{joint_name}' (type: {joint_type})")
            else:
                joints_data.append(joint_data)
        
        if self.verbose:
            print(f"Parsed {len(links_data)} links, {len(joints_data)} joints, and {len(closed_loop_joints)} closed-loop joints from SDF")

        return {
            'robot_name': robot_name,
            'links_data': links_data,
            'joints_data': joints_data,
            'closed_loop_joints': closed_loop_joints,  # 閉リンクジョイント情報を追加
            'materials_data': materials_data,
            'missing_meshes': missing_meshes
        }
    
    def _resolve_mesh_path(self, mesh_filename, sdf_file_path):
        """Resolve mesh file path from SDF reference.
        
        Args:
            mesh_filename: Mesh file reference from SDF
            sdf_file_path: Path to SDF file
            
        Returns:
            Absolute path to mesh file, or None if not found
        """
        # URDFParserの_resolve_mesh_pathと同様の処理
        # 簡易実装のため、URDFParserを再利用することも可能
        urdf_dir = os.path.dirname(os.path.abspath(sdf_file_path))
        mesh_basename = os.path.basename(mesh_filename)
        
        # package://パスの処理
        if mesh_filename.startswith('package://'):
            # package://パスを処理（簡易実装）
            parts = mesh_filename.split('/')
            if len(parts) > 2:
                package_name = parts[2]
                relative_path = '/'.join(parts[3:]) if len(parts) > 3 else ''
                
                # パッケージルートを探す（循環参照を避けるため、直接関数を呼び出す）
                package_root = find_package_root(sdf_file_path, package_name, max_search_depth=10, verbose=self.verbose)
                
                if package_root:
                    if relative_path:
                        candidate = os.path.join(package_root, relative_path)
                    else:
                        candidate = os.path.join(package_root, mesh_basename)
                    
                    if os.path.exists(candidate):
                        return candidate
        
        # 相対パスの処理
        candidates = [
            os.path.join(urdf_dir, mesh_filename),
            os.path.join(urdf_dir, 'meshes', mesh_basename),
            os.path.join(os.path.dirname(urdf_dir), 'meshes', mesh_basename),
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        return None

    def _parse_groups(self, root):
        """Parse group definitions from SRDF.

        Args:
            root: XML root element

        Returns:
            List of group dictionaries with 'name', 'links', and 'joints'
        """
        groups = []
        for group_elem in root.findall('group'):
            group_name = group_elem.get('name')
            if not group_name:
                continue

            links = []
            for link_elem in group_elem.findall('link'):
                link_name = link_elem.get('name')
                if link_name:
                    links.append(link_name)

            joints = []
            for joint_elem in group_elem.findall('joint'):
                joint_name = joint_elem.get('name')
                if joint_name:
                    joints.append(joint_name)

            groups.append({
                'name': group_name,
                'links': links,
                'joints': joints
            })

            if self.verbose:
                print(f"  Group '{group_name}': {len(links)} links, {len(joints)} joints")

        return groups

    def _parse_disabled_collisions(self, root):
        """Parse disabled collision pairs from SRDF.

        Args:
            root: XML root element

        Returns:
            List of disabled collision dictionaries with 'link1' and 'link2'
        """
        disabled_collisions = []
        for disable_elem in root.findall('disable_collisions'):
            link1 = disable_elem.get('link1')
            link2 = disable_elem.get('link2')
            if link1 and link2:
                disabled_collisions.append({
                    'link1': link1,
                    'link2': link2,
                    'reason': disable_elem.get('reason', '')
                })

        return disabled_collisions

    def _parse_disabled_links(self, root):
        """Parse disabled links from SRDF.

        Args:
            root: XML root element

        Returns:
            List of disabled link names
        """
        disabled_links = []
        for disable_elem in root.findall('disable_links'):
            for link_elem in disable_elem.findall('link'):
                link_name = link_elem.get('name')
                if link_name:
                    disabled_links.append(link_name)

        return disabled_links

    def _parse_disabled_joints(self, root):
        """Parse disabled joints from SRDF.

        Args:
            root: XML root element

        Returns:
            List of disabled joint names
        """
        disabled_joints = []
        for disable_elem in root.findall('disable_joints'):
            for joint_elem in disable_elem.findall('joint'):
                joint_name = joint_elem.get('name')
                if joint_name:
                    disabled_joints.append(joint_name)

        return disabled_joints

    def _parse_end_effectors(self, root):
        """Parse end effector definitions from SRDF.

        Args:
            root: XML root element

        Returns:
            List of end effector dictionaries
        """
        end_effectors = []
        for ee_elem in root.findall('end_effector'):
            ee_name = ee_elem.get('name')
            if not ee_name:
                continue

            group = ee_elem.get('group')
            parent_link = ee_elem.get('parent_link')
            parent_group = ee_elem.get('parent_group')

            end_effectors.append({
                'name': ee_name,
                'group': group,
                'parent_link': parent_link,
                'parent_group': parent_group
            })

            if self.verbose:
                print(f"  End effector '{ee_name}': group={group}, parent_link={parent_link}")

        return end_effectors

    def _parse_virtual_joints(self, root):
        """Parse virtual joint definitions from SRDF.

        Args:
            root: XML root element

        Returns:
            List of virtual joint dictionaries
        """
        virtual_joints = []
        for vj_elem in root.findall('virtual_joint'):
            vj_name = vj_elem.get('name')
            if not vj_name:
                continue

            parent_frame = vj_elem.get('parent_frame')
            child_link = vj_elem.get('child_link')
            joint_type = vj_elem.get('type', 'fixed')

            virtual_joints.append({
                'name': vj_name,
                'parent_frame': parent_frame,
                'child_link': child_link,
                'type': joint_type
            })

            if self.verbose:
                print(f"  Virtual joint '{vj_name}': type={joint_type}, parent={parent_frame}, child={child_link}")

        return virtual_joints


# ============================================================================
# MJCF PARSER
# ============================================================================

class MJCFParser:
    """Parser for MJCF (MuJoCo Model Format) files.

    This class provides methods to parse MJCF XML files and extract
    robot structure information including bodies, joints, meshes,
    and default classes.

    Usage:
        parser = MJCFParser()
        result = parser.parse_mjcf('/path/to/robot.xml', working_dir='/tmp')
        
        # Access parsed data
        robot_name = result['robot_name']
        bodies_data = result['bodies']
        joints_data = result['joints']
        meshes_data = result['meshes']
    """

    def __init__(self, verbose=True):
        """Initialize MJCF parser.

        Args:
            verbose: If True, print parsing progress and information
        """
        self.verbose = verbose
        self.conversion_utils = ConversionUtils()
        # Track parsed files for include resolution
        self._parsed_files = set()  # Set of absolute paths already parsed
        self._include_stack = []  # Stack of include paths for cycle detection
        # Provenance tracking
        self._provenance = {
            'warnings': [],
            'unknown_tags': set(),
            'unknown_attrs': {}  # {tag_name: set of attribute names}
        }

    def parse_mjcf(self, mjcf_file_path, working_dir=None):
        """Parse MJCF file and extract robot structure information.

        Args:
            mjcf_file_path: Path to MJCF XML file
            working_dir: Working directory for resolving mesh paths (optional)

        Returns:
            Dictionary containing:
                - robot_name: Name of the robot
                - bodies: List of body data
                - joints: List of joint data
                - meshes: Dictionary of mesh information
                - eulerseq: Euler sequence from compiler settings
                - default_classes: Dictionary of default class settings

        Raises:
            FileNotFoundError: If MJCF file does not exist
            ET.ParseError: If MJCF file is not valid XML
            ValueError: If root element is not 'mujoco'
        """
        if not os.path.exists(mjcf_file_path):
            raise FileNotFoundError(f"MJCF file not found: {mjcf_file_path}")

        # if self.verbose:
        #     print(f"Parsing MJCF from: {mjcf_file_path}")

        # Reset provenance tracking for new parse
        self._parsed_files = set()
        self._include_stack = []
        self._provenance = {
            'warnings': [],
            'unknown_tags': set(),
            'unknown_attrs': {}
        }

        # Parse MJCF file with include resolution
        root = self._parse_with_includes(mjcf_file_path, working_dir)

        if root.tag != 'mujoco':
            raise ValueError("Root element must be 'mujoco' for valid MJCF file")

        # Get robot name from file name
        robot_name = os.path.splitext(os.path.basename(mjcf_file_path))[0]
        # if self.verbose:
        #     print(f"Robot name: {robot_name}")

        # Parse compiler settings
        eulerseq, angle_unit = self._parse_compiler(root)

        # Parse default classes
        default_classes = self._parse_defaults(root)

        # Parse asset meshes
        if working_dir is None:
            working_dir = os.path.dirname(mjcf_file_path)
        meshes_data = self._parse_assets(root, mjcf_file_path, working_dir)

        # Parse body hierarchy
        bodies_data = []
        joints_data = []
        worldbody = root.find('worldbody')
        if worldbody is not None:
            for body_elem in worldbody.findall('body'):
                self._parse_body(body_elem, None, 0, bodies_data, joints_data,
                                meshes_data, default_classes, eulerseq, None, angle_unit)

        if self.verbose:
            print(f"Parsed {len(bodies_data)} bodies and {len(joints_data)} joints")

        # Parse equality constraints (closed-loop joints)
        closed_loop_joints = self._parse_equality(root, bodies_data)

        if self.verbose:
            print(f"Parsed {len(closed_loop_joints)} closed-loop joints from equality section")

        # Build IR structure
        ir = self._build_ir(root, robot_name, eulerseq, default_classes, meshes_data, 
                           bodies_data, joints_data, mjcf_file_path)

        return {
            'robot_name': robot_name,
            'bodies': bodies_data,
            'joints': joints_data,
            'meshes': meshes_data,
            'eulerseq': eulerseq,
            'default_classes': default_classes,
            'closed_loop_joints': closed_loop_joints,  # 閉リンクジョイント情報を追加
            'ir': ir  # New IR structure
        }

    def _parse_compiler(self, root):
        """Parse compiler settings from MJCF.

        Args:
            root: XML root element

        Returns:
            Tuple of (eulerseq, angle_unit)
            - eulerseq: Euler sequence string (default 'xyz')
            - angle_unit: 'degree' or 'radian' (default 'degree')
        """
        eulerseq = 'xyz'  # Default
        angle_unit = 'degree'  # Default in MuJoCo

        compiler_elem = root.find('compiler')
        if compiler_elem is not None:
            eulerseq_attr = compiler_elem.get('eulerseq')
            if eulerseq_attr:
                eulerseq = eulerseq_attr.lower()
                # if self.verbose:
                #     print(f"Compiler eulerseq: {eulerseq}")

            angle_attr = compiler_elem.get('angle')
            if angle_attr:
                angle_unit = angle_attr.lower()
                # if self.verbose:
                #     print(f"Compiler angle: {angle_unit}")

        return eulerseq, angle_unit

    def _parse_defaults(self, root):
        """Parse default classes from MJCF.

        Args:
            root: XML root element

        Returns:
            Dictionary of default class settings
        """
        default_classes = {}

        def parse_defaults_recursive(default_elem, parent_class_name=None, ancestor_classes=None):
            """Recursively parse default classes.
            
            Args:
                default_elem: Default XML element
                parent_class_name: Name of parent class (for inheritance)
                ancestor_classes: List of ancestor class names (for multi-level inheritance)
            """
            class_name = default_elem.get('class', parent_class_name)
            
            # Build ancestor classes list
            if ancestor_classes is None:
                ancestor_classes = []
            if parent_class_name:
                ancestor_classes = ancestor_classes + [parent_class_name]

            # Inherit from parent class (if already registered)
            class_defaults = {}
            if parent_class_name and parent_class_name in default_classes:
                class_defaults = default_classes[parent_class_name].copy()

            # Get joint defaults
            joint_elem = default_elem.find('joint')
            if joint_elem is not None:
                axis_str = joint_elem.get('axis')
                if axis_str:
                    class_defaults['joint_axis'] = [float(v) for v in axis_str.split()]

                range_str = joint_elem.get('range')
                if range_str:
                    class_defaults['joint_range'] = [float(v) for v in range_str.split()]

                damping_str = joint_elem.get('damping')
                if damping_str:
                    class_defaults['joint_damping'] = float(damping_str)

                armature_str = joint_elem.get('armature')
                if armature_str:
                    class_defaults['joint_armature'] = float(armature_str)

                frictionloss_str = joint_elem.get('frictionloss')
                if frictionloss_str:
                    class_defaults['joint_frictionloss'] = float(frictionloss_str)

                stiffness_str = joint_elem.get('stiffness')
                if stiffness_str:
                    class_defaults['joint_stiffness'] = float(stiffness_str)

                margin_str = joint_elem.get('margin')
                if margin_str:
                    class_defaults['joint_margin'] = float(margin_str)

            # Get geom defaults
            geom_elem = default_elem.find('geom')
            if geom_elem is not None:
                geom_defaults = {}
                
                # Parse geom attributes
                size_str = geom_elem.get('size')
                if size_str:
                    geom_defaults['size'] = size_str
                
                type_str = geom_elem.get('type')
                if type_str:
                    geom_defaults['type'] = type_str
                
                pos_str = geom_elem.get('pos')
                if pos_str:
                    geom_defaults['pos'] = pos_str
                
                quat_str = geom_elem.get('quat')
                if quat_str:
                    geom_defaults['quat'] = quat_str
                
                euler_str = geom_elem.get('euler')
                if euler_str:
                    geom_defaults['euler'] = euler_str
                
                fromto_str = geom_elem.get('fromto')
                if fromto_str:
                    geom_defaults['fromto'] = fromto_str
                
                # Store geom defaults
                if geom_defaults:
                    class_defaults['geom'] = geom_defaults

            # Save class if it has a name
            if class_name:
                default_classes[class_name] = class_defaults
                # if self.verbose:
                #     print(f"Default class '{class_name}': {class_defaults}")

            # Parse child default classes recursively (pass ancestor classes)
            for child_default in default_elem.findall('default'):
                parse_defaults_recursive(child_default, class_name, ancestor_classes)

        # Find and parse all default elements
        for default_elem in root.findall('default'):
            parse_defaults_recursive(default_elem, None, [])

        # if self.verbose:
        #     print(f"Parsed {len(default_classes)} default classes")

        return default_classes

    def _parse_with_includes(self, mjcf_file_path, working_dir=None):
        """Parse MJCF file with include resolution.
        
        Args:
            mjcf_file_path: Path to MJCF XML file
            working_dir: Working directory for resolving paths
        
        Returns:
            Merged XML root element with includes resolved
        """
        mjcf_file_path = os.path.abspath(mjcf_file_path)
        
        # Check for circular includes
        if mjcf_file_path in self._include_stack:
            cycle = ' -> '.join(self._include_stack + [mjcf_file_path])
            warning = f"Circular include detected: {cycle}"
            self._provenance['warnings'].append(warning)
            if self.verbose:
                print(f"WARNING: {warning}")
            return ET.Element('mujoco')  # Return empty root to break cycle
        
        # Mark as parsed
        self._parsed_files.add(mjcf_file_path)
        self._include_stack.append(mjcf_file_path)
        
        try:
            # Parse main file
            tree = ET.parse(mjcf_file_path)
            root = tree.getroot()
            
            # Resolve includes
            mjcf_dir = os.path.dirname(mjcf_file_path)
            if working_dir is None:
                working_dir = mjcf_dir
            
            # Find and process include elements (use findall with XPath)
            # We need to process includes at all levels
            def process_includes_recursive(elem, parent_elem=None):
                """Recursively process includes in element tree."""
                # Find includes at current level
                includes = list(elem.findall('include'))
                for include_elem in includes:
                    include_file = include_elem.get('file')
                    if not include_file:
                        warning = "Include element missing 'file' attribute"
                        self._provenance['warnings'].append(warning)
                        if self.verbose:
                            print(f"WARNING: {warning}")
                        continue
                    
                    # Resolve include path
                    if os.path.isabs(include_file):
                        include_path = include_file
                    else:
                        # Try relative to current MJCF directory
                        include_path = os.path.join(mjcf_dir, include_file)
                        if not os.path.exists(include_path):
                            # Try relative to working directory
                            include_path = os.path.join(working_dir, include_file)
                    
                    if not os.path.exists(include_path):
                        warning = f"Include file not found: {include_file} (resolved to: {include_path})"
                        self._provenance['warnings'].append(warning)
                        if self.verbose:
                            print(f"WARNING: {warning}")
                        continue
                    
                    # Recursively parse included file
                    included_root = self._parse_with_includes(include_path, working_dir)
                    
                    # Merge included content into current element
                    # MuJoCo includes: merge all child elements of included root into current element
                    for child in included_root:
                        elem.append(child)
                    
                    # Remove include element
                    elem.remove(include_elem)
                
                # Recursively process child elements
                for child in list(elem):
                    process_includes_recursive(child, elem)
            
            # Process includes starting from root
            process_includes_recursive(root)
            
            return root
        
        finally:
            # Remove from include stack
            if self._include_stack and self._include_stack[-1] == mjcf_file_path:
                self._include_stack.pop()

    def _parse_assets(self, root, mjcf_file_path, working_dir):
        """Parse asset mesh information from MJCF.

        Args:
            root: XML root element
            mjcf_file_path: Path to MJCF file
            working_dir: Working directory for mesh search

        Returns:
            Dictionary mapping mesh names to mesh info (path and scale)
        """
        meshes_data = {}
        asset_elem = root.find('asset')
        
        if asset_elem is None:
            return meshes_data

        mjcf_dir = os.path.dirname(mjcf_file_path)
        parent_dir = os.path.dirname(mjcf_dir)  # 上位1階層

        # Mesh search directories（上位1階層、そこから下位2階層まで）
        def collect_search_dirs(base_dir, max_depth=2):
            """指定されたディレクトリから、下位2階層までのディレクトリを収集"""
            dirs = []
            if not os.path.exists(base_dir):
                return dirs
            
            def walk_dirs(current_dir, current_depth):
                if current_depth > max_depth:
                    return
                if current_dir not in dirs:
                    dirs.append(current_dir)
                if current_depth < max_depth:
                    try:
                        for item in os.listdir(current_dir):
                            item_path = os.path.join(current_dir, item)
                            if os.path.isdir(item_path):
                                walk_dirs(item_path, current_depth + 1)
                    except Exception:
                        pass
            
            walk_dirs(base_dir, 0)
            return dirs

        # 上位1階層から下位2階層までのディレクトリを収集
        search_dirs = []
        # MJCFディレクトリとその下位2階層
        search_dirs.extend(collect_search_dirs(mjcf_dir, max_depth=2))
        # 上位1階層とその下位2階層
        if os.path.exists(parent_dir):
            search_dirs.extend(collect_search_dirs(parent_dir, max_depth=2))
        # working_dirとその下位2階層
        if working_dir and os.path.exists(working_dir):
            search_dirs.extend(collect_search_dirs(working_dir, max_depth=2))
        
        # 重複を除去
        search_dirs = list(dict.fromkeys(search_dirs))  # 順序を保持しながら重複除去

        for mesh_elem in asset_elem.findall('mesh'):
            mesh_name = mesh_elem.get('name')
            mesh_file = mesh_elem.get('file', '')

            # Derive mesh name from filename if not specified
            if not mesh_name and mesh_file:
                mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
                if self.verbose:
                    print(f"Mesh name derived from file: {mesh_name}")

            # Parse mesh scale attribute
            mesh_scale = [1.0, 1.0, 1.0]
            scale_str = mesh_elem.get('scale')
            if scale_str:
                scale_values = [float(v) for v in scale_str.split()]
                if len(scale_values) == 3:
                    mesh_scale = scale_values
                elif len(scale_values) == 1:
                    mesh_scale = [scale_values[0]] * 3
                if self.verbose:
                    print(f"  Mesh '{mesh_name}' scale: {mesh_scale}")

            # Resolve mesh file path
            if mesh_file:
                mesh_path = None
                mesh_basename = os.path.basename(mesh_file)

                # Method 1: Try original path relative to MJCF dir
                candidate = os.path.join(mjcf_dir, mesh_file)
                if os.path.exists(candidate):
                    mesh_path = candidate
                    # if self.verbose:
                    #     print(f"Found mesh: {mesh_name} -> {mesh_path}")

                # Method 2: Try in search directories with original structure
                if not mesh_path:
                    for search_dir in search_dirs:
                        if not search_dir or not os.path.exists(search_dir):
                            continue
                        candidate = os.path.join(search_dir, mesh_file)
                        if os.path.exists(candidate):
                            mesh_path = candidate
                            # if self.verbose:
                            #     print(f"Found mesh: {mesh_name} -> {mesh_path}")
                            break

                # Method 3: Search for basename in directories (max 2 levels deep)
                if not mesh_path:
                    for search_dir in search_dirs:
                        if not search_dir or not os.path.exists(search_dir):
                            continue
                        # 直接のパスを試す
                        candidate = os.path.join(search_dir, mesh_basename)
                        if os.path.exists(candidate):
                            mesh_path = candidate
                            # if self.verbose:
                            #     print(f"Found mesh: {mesh_name} -> {mesh_path}")
                            break
                        
                        # 下位2階層まで探索
                        search_dir_depth = len(search_dir.split(os.sep))
                        for root_dir, dirs, files in os.walk(search_dir):
                            # 現在のディレクトリの深さを計算
                            current_depth = len(root_dir.split(os.sep)) - search_dir_depth
                            # 2階層まで探索
                            if current_depth > 2:
                                dirs[:] = []  # これより深い探索を停止
                                continue
                            
                            if mesh_basename in files:
                                candidate = os.path.join(root_dir, mesh_basename)
                                mesh_path = candidate
                                # if self.verbose:
                                #     print(f"Found mesh: {mesh_name} -> {mesh_path} (depth {current_depth})")
                                break
                        
                        if mesh_path:
                            break

                if mesh_path:
                    meshes_data[mesh_name] = {
                        'path': mesh_path,
                        'scale': mesh_scale
                    }
                else:
                    # if self.verbose:
                    #     print(f"Warning: Could not find mesh file: {mesh_file}")
                    pass

        # if self.verbose:
        #     print(f"Parsed {len(meshes_data)} mesh assets")

        return meshes_data

    def _parse_body(self, body_elem, parent_name, level, bodies_data, joints_data,
                    meshes_data, default_classes, eulerseq, parent_childclass=None, angle_unit='degree'):
        """Recursively parse body element from MJCF.

        Args:
            body_elem: Body XML element
            parent_name: Name of parent body (None for root)
            level: Recursion level for indentation
            bodies_data: List to append body data to
            joints_data: List to append joint data to
            meshes_data: Dictionary of mesh information
            default_classes: Dictionary of default class settings
            eulerseq: Euler sequence from compiler settings
            parent_childclass: childclass from parent body (for default class inheritance)
            angle_unit: 'degree' or 'radian' from compiler angle attribute (default 'degree')
        """
        body_name = body_elem.get('name', f'body_{len(bodies_data)}')
        
        # Get childclass from body element (for child elements' default class)
        body_childclass = body_elem.get('childclass', parent_childclass)

        body_data = {
            'name': body_name,
            'parent': parent_name,
            'mass': 0.0,
            'inertia': {'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0},
            'inertial_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
            'stl_file': None,
            'visuals': [],
            'color': [1.0, 1.0, 1.0],
            'collision_mesh': None,  # Deprecated, use colliders list
            'colliders': [],  # List of collider dictionaries
            'mesh_scale': [1.0, 1.0, 1.0],
            'visual_origin': {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
            'pos': [0.0, 0.0, 0.0],
            'quat': [1.0, 0.0, 0.0, 0.0],
            'rpy': [0.0, 0.0, 0.0],
            'is_mesh_reversed': False,
            'rotation_axis': None  # Will be set from joint axis attribute
        }

        # Parse position
        pos_str = body_elem.get('pos', '0 0 0')
        body_data['pos'] = [float(v) for v in pos_str.split()]

        # Parse quaternion and convert to RPY
        # Note: body quat represents the body's orientation, not the visual mesh orientation
        # visual_origin should only be set from geom quat/euler, not from body quat
        quat_str = body_elem.get('quat')
        if quat_str:
            body_data['quat'] = [float(v) for v in quat_str.split()]
            body_data['rpy'] = self.conversion_utils.quat_to_rpy(body_data['quat'])
            # if self.verbose:
            #     print(f"{'  ' * level}  Body quat: {body_data['quat']} -> rpy: {body_data['rpy']}")

        # Parse euler angles (use eulerseq from compiler settings)
        euler_str = body_elem.get('euler')
        if euler_str:
            euler_values = [float(v) for v in euler_str.split()]
            # Convert to degrees if needed (euler_to_rpy expects degrees)
            if angle_unit == 'radian':
                euler_degrees = [math.degrees(v) for v in euler_values]
            else:
                euler_degrees = euler_values
            # Convert euler to RPY using eulerseq
            body_data['rpy'] = self.conversion_utils.euler_to_rpy(euler_degrees, eulerseq)
            # Convert RPY back to quat for storage
            body_data['quat'] = self.conversion_utils.rpy_to_quat(body_data['rpy'])
            # if self.verbose:
            #     print(f"{'  ' * level}  Body euler ({angle_unit}): {euler_values} (seq={eulerseq}) -> rpy (rad): {body_data['rpy']}")

        # Parse xyaxes attribute (coordinate frame definition)
        xyaxes_str = body_elem.get('xyaxes')
        if xyaxes_str:
            try:
                xyaxes_quat = self.conversion_utils.xyaxes_to_quat(xyaxes_str)
                body_data['quat'] = xyaxes_quat
                body_data['rpy'] = self.conversion_utils.quat_to_rpy(xyaxes_quat)
                # if self.verbose:
                #     print(f"{'  ' * level}  Body xyaxes: {xyaxes_str} -> quat: {xyaxes_quat} -> rpy: {body_data['rpy']}")
            except Exception as e:
                if self.verbose:
                    print(f"{'  ' * level}  Warning: Failed to parse xyaxes '{xyaxes_str}': {str(e)}")

        # Parse inertial information
        inertial_elem = body_elem.find('inertial')
        if inertial_elem is not None:
            mass_str = inertial_elem.get('mass')
            if mass_str:
                body_data['mass'] = float(mass_str)

            inertial_pos = inertial_elem.get('pos', '0 0 0')
            body_data['inertial_origin']['xyz'] = [float(v) for v in inertial_pos.split()]

            # Parse inertia tensor
            # MJCF supports two formats:
            # - diaginertia: "ixx iyy izz" (diagonal elements only)
            # - fullinertia: "ixx iyy izz ixy ixz iyz" (full symmetric tensor)
            fullinertia_str = inertial_elem.get('fullinertia')
            diaginertia_str = inertial_elem.get('diaginertia')

            if fullinertia_str:
                # Full inertia tensor with off-diagonal elements
                inertia_values = [float(v) for v in fullinertia_str.split()]
                if len(inertia_values) >= 6:
                    body_data['inertia']['ixx'] = inertia_values[0]
                    body_data['inertia']['iyy'] = inertia_values[1]
                    body_data['inertia']['izz'] = inertia_values[2]
                    body_data['inertia']['ixy'] = inertia_values[3]
                    body_data['inertia']['ixz'] = inertia_values[4]
                    body_data['inertia']['iyz'] = inertia_values[5]
                elif len(inertia_values) >= 3:
                    # Fallback if only 3 values provided
                    body_data['inertia']['ixx'] = inertia_values[0]
                    body_data['inertia']['iyy'] = inertia_values[1]
                    body_data['inertia']['izz'] = inertia_values[2]
            elif diaginertia_str:
                # Diagonal inertia only (no off-diagonal elements)
                diag = [float(v) for v in diaginertia_str.split()]
                if len(diag) >= 3:
                    body_data['inertia']['ixx'] = diag[0]
                    body_data['inertia']['iyy'] = diag[1]
                    body_data['inertia']['izz'] = diag[2]

        # Parse geometry (geom elements)
        geom_elems = body_elem.findall('geom')
        # if geom_elems and self.verbose:
        #     print(f"{'  ' * level}  Found {len(geom_elems)} geom element(s)")

        for idx, geom_elem in enumerate(geom_elems):
            mesh_name = geom_elem.get('mesh')
            
            # Determine geom type (visual or collision)
            geom_class = geom_elem.get('class', '')
            geom_group = geom_elem.get('group', '')
            contype = geom_elem.get('contype')
            conaffinity = geom_elem.get('conaffinity')

            # Check if collision geom (class="collision", "collision-left", "collision-right", etc.)
            is_collision_geom = (geom_class.startswith('collision') or 
                                geom_group in ['0', '3'])
            is_visual_geom = (geom_class == 'visual' or 
                             geom_group in ['1', '2'] or
                             (contype == '0') or (conaffinity == '0'))

            if not is_collision_geom and not is_visual_geom:
                is_visual_geom = True
                is_collision_geom = True

            # if self.verbose:
            #     print(f"{'  ' * level}    Geom[{idx}] type: visual={is_visual_geom}, collision={is_collision_geom}")

            if mesh_name and mesh_name in meshes_data:
                mesh_info = meshes_data[mesh_name]
                mesh_path = mesh_info['path']
                asset_mesh_scale = mesh_info['scale']

                # Parse geom position
                geom_pos_str = geom_elem.get('pos', '0 0 0')
                geom_pos = [float(v) for v in geom_pos_str.split()]

                # Parse geom rotation (quat or euler)
                # Note: geom quat/euler represents the rotation RELATIVE to the body frame
                # If not specified, the geom is aligned with the body frame (no additional rotation)
                geom_quat_str = geom_elem.get('quat')
                geom_euler_str = geom_elem.get('euler')
                # Initialize with identity (no rotation relative to body frame)
                # The body's orientation is already handled in point['angle'] (origin_rpy)
                # visual_origin should only contain the geom's LOCAL rotation within the body frame
                geom_rpy = [0.0, 0.0, 0.0]
                geom_quat = [1.0, 0.0, 0.0, 0.0]

                if geom_quat_str:
                    geom_quat = [float(v) for v in geom_quat_str.split()]
                    geom_rpy = self.conversion_utils.quat_to_rpy(geom_quat)
                elif geom_euler_str:
                    euler_degrees = [float(v) for v in geom_euler_str.split()]
                    geom_rpy = self.conversion_utils.euler_to_rpy(euler_degrees, eulerseq)
                    # Convert RPY back to quat for storage
                    geom_quat = self.conversion_utils.rpy_to_quat(geom_rpy)
                # If geom has no quat/euler, geom_rpy remains [0,0,0] (aligned with body frame)

                # Parse geom mesh scale
                geom_meshscale = [1.0, 1.0, 1.0]
                meshscale_str = geom_elem.get('meshscale')
                if meshscale_str:
                    scale_values = [float(v) for v in meshscale_str.split()]
                    if len(scale_values) == 3:
                        geom_meshscale = scale_values
                    elif len(scale_values) == 1:
                        geom_meshscale = [scale_values[0]] * 3

                # Combine scales
                combined_scale = [
                    asset_mesh_scale[0] * geom_meshscale[0],
                    asset_mesh_scale[1] * geom_meshscale[1],
                    asset_mesh_scale[2] * geom_meshscale[2]
                ]
                
                # Check if scale contains negative values (mesh reversal)
                is_reversed = self.conversion_utils.check_scale_reversal(combined_scale)
                if is_reversed:
                    body_data['is_mesh_reversed'] = True
                    # if self.verbose:
                    #     print(f"{'  ' * level}    Detected mesh reversal from scale: {combined_scale}")

                # Parse color
                geom_color = [1.0, 1.0, 1.0]
                rgba_str = geom_elem.get('rgba')
                if rgba_str:
                    rgba = [float(v) for v in rgba_str.split()]
                    geom_color = rgba[:3]

                # Add to visuals list if visual geom
                if is_visual_geom:
                    visual_data = {
                        'mesh': mesh_path,
                        'pos': geom_pos,
                        'quat': geom_quat,
                        'color': geom_color
                    }
                    body_data['visuals'].append(visual_data)

                    # Set first visual as main stl_file for backward compatibility
                    if body_data['stl_file'] is None:
                        body_data['stl_file'] = mesh_path
                        body_data['color'] = geom_color
                        body_data['mesh_scale'] = combined_scale
                        # visual_origin should use geom's rpy (relative to body frame)
                        # If geom has no quat/euler, it defaults to [0,0,0] (no rotation)
                        body_data['visual_origin'] = {
                            'xyz': geom_pos,
                            'rpy': geom_rpy
                        }

                # Set as collision mesh if collision geom (for backward compatibility)
                if is_collision_geom and body_data['collision_mesh'] is None:
                    body_data['collision_mesh'] = mesh_path
                    # Also add to colliders list with rotation from geom quat/euler
                    collider_rotation_deg = [math.degrees(r) for r in geom_rpy]  # Convert to degrees
                    collider_data = {
                        'type': 'mesh',
                        'enabled': True,
                        'data': None,
                        'mesh': mesh_path,
                        'mesh_scale': combined_scale,
                        'position': geom_pos,
                        'rotation': collider_rotation_deg
                    }
                    body_data['colliders'].append(collider_data)
                    # if self.verbose:
                    #     print(f"{'  ' * level}    [COLLIDER_DEBUG] Added mesh collider:")
                    #     print(f"{'  ' * level}      mesh: {os.path.basename(mesh_path)}")
                    #     print(f"{'  ' * level}      geom_quat_str: {geom_quat_str}")
                    #     print(f"{'  ' * level}      geom_euler_str: {geom_euler_str}")
                    #     print(f"{'  ' * level}      geom_quat: {geom_quat}")
                    #     print(f"{'  ' * level}      geom_rpy (rad): {geom_rpy}")
                    #     print(f"{'  ' * level}      collider_rotation (deg): {collider_rotation_deg}")
                    #     print(f"{'  ' * level}      collider_position: {geom_pos}")
            
            # Handle primitive collision geoms (box, sphere, cylinder, ellipsoid, capsule)
            elif is_collision_geom:
                # Get default class attributes for this geom
                geom_class = geom_elem.get('class', '')
                geom_defaults = {}
                if geom_class and geom_class in default_classes:
                    class_defaults = default_classes[geom_class]
                    if 'geom' in class_defaults:
                        geom_defaults = class_defaults['geom'].copy()
                
                # Merge default class attributes with geom element attributes (geom element attributes take precedence)
                # Get type (from geom element or default class)
                geom_type = geom_elem.get('type', '')
                if not geom_type and 'type' in geom_defaults:
                    geom_type = geom_defaults['type']
                
                # Get position (from geom element or default class)
                geom_pos_str = geom_elem.get('pos')
                if not geom_pos_str and 'pos' in geom_defaults:
                    geom_pos_str = geom_defaults['pos']
                if not geom_pos_str:
                    geom_pos_str = '0 0 0'
                geom_pos = [float(v) for v in geom_pos_str.split()]
                
                # Get rotation (from geom element or default class)
                geom_quat_str = geom_elem.get('quat')
                if not geom_quat_str and 'quat' in geom_defaults:
                    geom_quat_str = geom_defaults['quat']
                
                geom_euler_str = geom_elem.get('euler')
                if not geom_euler_str and 'euler' in geom_defaults:
                    geom_euler_str = geom_defaults['euler']
                has_explicit_geom_rot = bool(geom_quat_str or geom_euler_str)
                
                # Initialize geom_rpy with body's rotation (from xyaxes or euler).
                # NOTE: If fromto is present, we will override this with the fromto direction
                # unless an explicit geom rotation is provided.
                geom_rpy = body_data.get('rpy', [0.0, 0.0, 0.0]).copy()
                
                if geom_quat_str:
                    geom_quat = [float(v) for v in geom_quat_str.split()]
                    geom_rpy = self.conversion_utils.quat_to_rpy(geom_quat)
                elif geom_euler_str:
                    euler_degrees = [float(v) for v in geom_euler_str.split()]
                    geom_rpy = self.conversion_utils.euler_to_rpy(euler_degrees, eulerseq)
                
                # Convert RPY to degrees for collider_data
                geom_rpy_deg = [math.degrees(r) for r in geom_rpy]
                
                # Get size from default class if not in geom element
                size_str = geom_elem.get('size')
                if not size_str and 'size' in geom_defaults:
                    size_str = geom_defaults['size']
                
                # Check for fromto attribute (from geom element or default class)
                fromto_str = geom_elem.get('fromto')
                if not fromto_str and 'fromto' in geom_defaults:
                    fromto_str = geom_defaults['fromto']
                
                # Handle fromto attribute (capsule-like geometry)
                handled_fromto = False
                if fromto_str:
                    fromto_values = [float(v) for v in fromto_str.split()]
                    if len(fromto_values) >= 6:
                        from_point = fromto_values[:3]
                        to_point = fromto_values[3:6]
                        
                        # Calculate capsule parameters from fromto
                        # MuJoCo fromto: 2点間の距離 = 中心軸の長さ（端点間）
                        direction = [to_point[i] - from_point[i] for i in range(3)]
                        fromto_distance = math.sqrt(sum(d*d for d in direction))
                        
                        # Get size (radius) from geom element or default class
                        if not size_str:
                            size_str = '0.01'
                        size_values = [float(v) for v in size_str.split()]
                        radius = size_values[0] if len(size_values) >= 1 else 0.01
                        
                        # Center position
                        center_pos = [(from_point[i] + to_point[i]) / 2.0 for i in range(3)]
                        
                        # MJCFのfromtoは中心軸の長さを指定するため、cylinder部分の長さはそのままdistance
                        cylinder_length = fromto_distance
                        
                        # fromto方向から回転を計算（明示回転がない場合のみ）
                        if fromto_distance > 0 and not has_explicit_geom_rot:
                            # 方向ベクトルを正規化
                            dx, dy, dz = (direction[0] / fromto_distance,
                                          direction[1] / fromto_distance,
                                          direction[2] / fromto_distance)
                            # z軸(0,0,1)からdirectionへの回転を求める
                            dot = max(min(dz, 1.0), -1.0)
                            if dot > 0.999999:
                                fromto_quat = [1.0, 0.0, 0.0, 0.0]
                            elif dot < -0.999999:
                                # 180度回転（x軸回りで反転）
                                fromto_quat = [0.0, 1.0, 0.0, 0.0]
                            else:
                                # 回転軸 = z × direction
                                ax = -dy
                                ay = dx
                                az = 0.0
                                axis_len = math.sqrt(ax * ax + ay * ay + az * az)
                                if axis_len > 0:
                                    ax /= axis_len
                                    ay /= axis_len
                                    az /= axis_len
                                angle = math.acos(dot)
                                half = angle / 2.0
                                s = math.sin(half)
                                fromto_quat = [math.cos(half), ax * s, ay * s, az * s]
                            geom_rpy = self.conversion_utils.quat_to_rpy(fromto_quat)
                            # Update degrees after overriding geom_rpy
                            geom_rpy_deg = [math.degrees(r) for r in geom_rpy]
                        
                        # capsule の length は cylinder 部分の長さとして保存
                        # (両端の半球は別途描画されるため、lengthには含めない)
                        fromto_geom_type = geom_type if geom_type in ['capsule', 'cylinder'] else 'capsule'
                        collider_data = {
                            'type': fromto_geom_type,
                            'geometry': {
                                'radius': radius,
                                'length': cylinder_length  # cylinder部分のみの長さ
                            },
                            'position': center_pos,
                            'rotation': geom_rpy_deg
                        }
                        
                        if self.verbose:
                            print(f"{'  ' * level}      Set {fromto_geom_type} collider from fromto: fromto_distance={fromto_distance:.6f}, radius={radius}, cylinder_length={cylinder_length:.6f} (from class='{geom_class}')")
                        
                        # Add to colliders list
                        body_data['colliders'].append({
                            'type': 'primitive',
                            'enabled': True,
                            'data': collider_data,
                            'mesh': None,
                            'mesh_scale': [1.0, 1.0, 1.0],
                            'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                            'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0])
                        })
                        
                        # Store first collision primitive as collider_data (for backward compatibility)
                        if body_data.get('collider_type') is None:
                            body_data['collider_data'] = collider_data
                            body_data['collider_type'] = 'primitive'
                            body_data['collider_enabled'] = True
                            # if self.verbose:
                            #     print(f"{'  ' * level}    Set capsule collider from fromto: radius={radius}, length={length} (from class='{geom_class}')")
                        handled_fromto = True
                
                if handled_fromto:
                    continue

                # Parse primitive geom types
                # If type is not specified but size is available, infer type from size
                if not fromto_str and size_str:
                    size_values = [float(v) for v in size_str.split()]
                    if not geom_type:
                        # Infer type from size if not specified
                        if len(size_values) == 1:
                            # Single value: sphere (MJCF default)
                            geom_type = 'sphere'
                        elif len(size_values) == 2:
                            # Two values: cylinder or capsule
                            geom_type = 'cylinder'  # Default to cylinder
                        elif len(size_values) == 3:
                            # Three values: box or ellipsoid
                            geom_type = 'box'  # Default to box
                        # if self.verbose:
                        #     print(f"{'  ' * level}    Inferred geom_type='{geom_type}' from size (length={len(size_values)})")
                
                if geom_type in ['box', 'sphere', 'cylinder', 'capsule', 'ellipsoid']:
                    # Create a temporary geom element with merged attributes for parsing
                    temp_geom_elem = ET.Element('geom')
                    if geom_type:
                        temp_geom_elem.set('type', geom_type)
                    if size_str:
                        temp_geom_elem.set('size', size_str)
                    if geom_pos_str and geom_pos_str != '0 0 0':
                        temp_geom_elem.set('pos', geom_pos_str)
                    if geom_quat_str:
                        temp_geom_elem.set('quat', geom_quat_str)
                    elif geom_euler_str:
                        temp_geom_elem.set('euler', geom_euler_str)
                    
                    collider_data = self.parse_primitive_geom(temp_geom_elem, geom_type, level)
                    if collider_data:
                        # Update position and rotation from geom attributes
                        collider_data['position'] = geom_pos
                        collider_data['rotation'] = geom_rpy_deg
                        
                        # Add to colliders list
                        body_data['colliders'].append({
                            'type': 'primitive',
                            'enabled': True,
                            'data': collider_data,
                            'mesh': None,
                            'mesh_scale': [1.0, 1.0, 1.0],
                            'position': collider_data.get('position', [0.0, 0.0, 0.0]),
                            'rotation': collider_data.get('rotation', [0.0, 0.0, 0.0])
                        })
                        
                        # if self.verbose:
                        #     print(f"{'  ' * level}    Added primitive collider to colliders list: {geom_type} (total: {len(body_data['colliders'])})")
                        
                        # Store first collision primitive as collider_data (for backward compatibility)
                        if body_data.get('collider_type') is None:
                            body_data['collider_data'] = collider_data
                            body_data['collider_type'] = 'primitive'
                            body_data['collider_enabled'] = True
                            # if self.verbose:
                            #     print(f"{'  ' * level}    Set primitive collider: {geom_type} (from class='{geom_class}')")
                    else:
                        # if self.verbose:
                        #     print(f"{'  ' * level}    Warning: parse_primitive_geom returned None for {geom_type}")
                        # if self.verbose:
                        #     print(f"{'  ' * level}    Warning: parse_primitive_geom returned None for {geom_type}")
                        pass

        # Check for freejoint element (MJCF's 6-DOF free joint)
        freejoint_elem = body_elem.find('freejoint')
        if freejoint_elem is not None:
            body_data['has_freejoint'] = True
            if self.verbose:
                print(f"{'  ' * level}  Body '{body_name}' has freejoint (6-DOF floating base)")

        # Parse joints in this body (before appending to bodies_data)
        # This ensures rotation_axis is set before body_data is added to the list
        joint_elems = body_elem.findall('joint')
        for joint_elem in joint_elems:
            joint_name = joint_elem.get('name', f'joint_{len(joints_data)}')
            joint_type = joint_elem.get('type', 'hinge')
            
            # Map MJCF joint types to URDF types
            type_mapping = {
                'hinge': 'revolute',
                'slide': 'prismatic',
                'ball': 'spherical',
                'free': 'floating'
            }
            urdf_type = type_mapping.get(joint_type, 'revolute')
            
            # For slide (prismatic) joints, set rotation_axis to 3 (Fixed)
            if joint_type == 'slide':
                body_data['rotation_axis'] = 3  # Fixed
                # if self.verbose:
                #     print(f"{'  ' * level}    Joint '{joint_name}' is slide (prismatic) -> rotation_axis=3 (Fixed)")

            # Get joint class (for default class lookup)
            joint_class = joint_elem.get('class', body_childclass)
            
            # Apply default class settings
            # Start with parent childclass defaults, then apply joint class defaults
            joint_defaults = {}
            if body_childclass and body_childclass in default_classes:
                joint_defaults = default_classes[body_childclass].copy()
            if joint_class and joint_class in default_classes:
                # Merge joint class defaults (overwrite parent defaults)
                for key, value in default_classes[joint_class].items():
                    joint_defaults[key] = value

            # Determine joint axis in body's local coordinate system
            # NOTE:
            #   In MJCF, if the axis attribute is omitted for hinge/slide joints,
            #   the default axis is the Z-axis [0, 0, 1] in the body's local frame.
            #   We must respect this to avoid twisted joints (e.g. Cassie model).
            axis_local = [0.0, 0.0, 1.0]  # Default: Z-axis in body local frame (MJCF spec)
            
            # Apply default class axis if available
            if 'joint_axis' in joint_defaults:
                axis_local = joint_defaults['joint_axis']
            
            # Parse axis (explicit attribute overrides default class)
            axis_str = joint_elem.get('axis')
            if axis_str:
                axis_local = [float(v) for v in axis_str.split()]

            # Calculate axis in parent frame (for URDF joint axis attribute)
            # Transform axis from body local frame to parent frame
            import numpy as np
            body_quat = body_data.get('quat', [1.0, 0.0, 0.0, 0.0])
            axis_parent = axis_local.copy()
            if body_quat != [1.0, 0.0, 0.0, 0.0]:
                R_parent = self.conversion_utils.quat_to_rotation_matrix(body_quat)
                axis_local_np = np.array(axis_local)
                axis_parent_np = R_parent @ axis_local_np
                axis_parent = axis_parent_np.tolist()

            # if self.verbose:
            #     print(f"{'  ' * level}    Joint '{joint_name}' axis: local={axis_local} -> parent={axis_parent}")

            # Set rotation_axis in body_data based on axis_local (only for non-slide joints)
            # For slide joints, rotation_axis is already set to 3 (Fixed) above
            # IMPORTANT: In Rotation Test (Assembler.py line 5578-5616):
            #   1. origin RPY is applied first (transform.RotateZ/Y/X)
            #   2. Then rotation_axis rotation is applied
            #   3. VTK's RotateZ() rotates around the CURRENT coordinate system's Z-axis
            #      (i.e., after origin RPY transformation)
            # MJCF's axis="0 0 1" means Z-axis in body's local frame (after origin RPY).
            # So rotation_axis should be determined from axis_local, not axis_parent.
            # This matches URDF behavior: URDF <axis xyz="0 0 1"/> also means rotation
            # around Z-axis after origin RPY transformation, resulting in rotation_axis=2.
            if joint_type != 'slide':
                # Find the axis with the largest absolute value in LOCAL coordinate system
                abs_axis = [abs(v) for v in axis_local]
                max_idx = abs_axis.index(max(abs_axis))
                # rotation_axis: 0=X, 1=Y, 2=Z (in local frame after origin RPY)
                body_data['rotation_axis'] = max_idx
                # if self.verbose:
                #     axis_names = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
                #     print(f"{'  ' * level}    Joint '{joint_name}' rotation_axis={body_data['rotation_axis']} ({axis_names[body_data['rotation_axis']]} in LOCAL frame)")

            # Joint position: MJCF joint pos is defined in the BODY local frame.
            # Convert it to the parent frame by applying the body's orientation.
            joint_pos_local = [0.0, 0.0, 0.0]
            joint_pos_str = joint_elem.get('pos')
            if joint_pos_str:
                try:
                    joint_pos_local = [float(v) for v in joint_pos_str.split()]
                except Exception:
                    joint_pos_local = [0.0, 0.0, 0.0]

            if any(v != 0.0 for v in joint_pos_local):
                body_pos_np = np.array(body_data['pos'])
                joint_pos_np = np.array(joint_pos_local)
                if body_quat != [1.0, 0.0, 0.0, 0.0]:
                    R_parent = self.conversion_utils.quat_to_rotation_matrix(body_quat)
                    origin_xyz = (body_pos_np + (R_parent @ joint_pos_np)).tolist()
                else:
                    origin_xyz = (body_pos_np + joint_pos_np).tolist()
            else:
                origin_xyz = body_data['pos']

            # if self.verbose and joint_pos_str:
            #     print(f"{'  ' * level}    Joint '{joint_name}' pos (local): {joint_pos_local} -> origin_xyz (parent): {origin_xyz}")

            joint_data = {
                'name': joint_name,
                'type': urdf_type,
                'parent': parent_name if parent_name else 'base_link',
                'child': body_name,
                'origin_xyz': origin_xyz,
                'origin_rpy': body_data['rpy'],
                'origin_quat': body_data.get('quat', [1.0, 0.0, 0.0, 0.0]),
                'axis': axis_parent,  # Axis in parent coordinate system (URDF convention)
                'limit': {'lower': -3.14159, 'upper': 3.14159, 'effort': 10.0, 
                         'velocity': 3.0, 'friction': 0.05},
                'dynamics': {'damping': 0.0, 'friction': 0.0}
            }
            
            # Apply default class range if available
            if 'joint_range' in joint_defaults:
                range_vals = joint_defaults['joint_range']
                if len(range_vals) >= 2:
                    # Convert to radians if angle_unit is degree
                    if angle_unit == 'degree':
                        joint_data['limit']['lower'] = math.radians(range_vals[0])
                        joint_data['limit']['upper'] = math.radians(range_vals[1])
                    else:
                        joint_data['limit']['lower'] = range_vals[0]
                        joint_data['limit']['upper'] = range_vals[1]
            
            # Apply default class damping if available
            if 'joint_damping' in joint_defaults:
                joint_data['dynamics']['damping'] = joint_defaults['joint_damping']
            
            # Apply default class armature if available
            if 'joint_armature' in joint_defaults:
                joint_data['dynamics']['armature'] = joint_defaults['joint_armature']
            
            # Apply default class frictionloss if available
            if 'joint_frictionloss' in joint_defaults:
                joint_data['dynamics']['friction'] = joint_defaults['joint_frictionloss']
            
            # Apply default class stiffness if available
            if 'joint_stiffness' in joint_defaults:
                joint_data['dynamics']['stiffness'] = joint_defaults['joint_stiffness']

            # Parse range (limits)
            range_str = joint_elem.get('range')
            if range_str:
                range_vals = [float(v) for v in range_str.split()]
                if len(range_vals) >= 2:
                    # Convert to radians if angle_unit is degree
                    if angle_unit == 'degree':
                        joint_data['limit']['lower'] = math.radians(range_vals[0])
                        joint_data['limit']['upper'] = math.radians(range_vals[1])
                    else:
                        joint_data['limit']['lower'] = range_vals[0]
                        joint_data['limit']['upper'] = range_vals[1]

            # Parse damping (個別ジョイント属性が優先)
            damping_str = joint_elem.get('damping')
            if damping_str:
                joint_data['dynamics']['damping'] = float(damping_str)

            # Parse frictionloss (個別ジョイント属性が優先)
            frictionloss_str = joint_elem.get('frictionloss')
            if frictionloss_str:
                joint_data['dynamics']['friction'] = float(frictionloss_str)

            # Parse armature (個別ジョイント属性が優先)
            armature_str = joint_elem.get('armature')
            if armature_str:
                joint_data['dynamics']['armature'] = float(armature_str)
            
            # Parse stiffness (個別ジョイント属性が優先)
            stiffness_str = joint_elem.get('stiffness')
            if stiffness_str:
                joint_data['dynamics']['stiffness'] = float(stiffness_str)
            
            # Parse margin (個別ジョイント属性が優先)
            margin_str = joint_elem.get('margin')
            if margin_str:
                joint_data['dynamics']['margin'] = float(margin_str)

            # Parse ref attribute (reference angle / default angle)
            # Note: ref follows the compiler angle attribute (degree or radian)
            ref_str = joint_elem.get('ref')
            if ref_str:
                ref_value = float(ref_str)
                # Convert to degrees for storage (Assembler expects degrees)
                if angle_unit == 'degree':
                    ref_angle_deg = ref_value
                else:
                    ref_angle_deg = math.degrees(ref_value)
                joint_data['ref'] = ref_angle_deg  # 度単位で保存
                # if self.verbose:
                #     print(f"{'  ' * level}    Joint '{joint_name}' ref angle: {ref_value} {angle_unit} = {ref_angle_deg} degrees")

            joints_data.append(joint_data)

        # If this body has no joint element, it is fixed to its parent in MJCF.
        # Create a synthetic fixed joint so the node graph has a proper origin.
        if not joint_elems:
            # Mark as fixed to avoid accidental rotations.
            body_data['rotation_axis'] = 3
            fixed_joint_name = f"{body_name}_fixed"
            fixed_joint = {
                'name': fixed_joint_name,
                'type': 'fixed',
                'parent': parent_name if parent_name else 'base_link',
                'child': body_name,
                'origin_xyz': body_data['pos'],
                'origin_rpy': body_data['rpy'],
                'origin_quat': body_data.get('quat', [1.0, 0.0, 0.0, 0.0]),
                'axis': [0.0, 0.0, 1.0],
                'limit': {'lower': 0.0, 'upper': 0.0, 'effort': 0.0, 'velocity': 0.0, 'friction': 0.0},
                'dynamics': {'damping': 0.0, 'friction': 0.0}
            }
            joints_data.append(fixed_joint)
            # if self.verbose:
            #     print(f"{'  ' * level}    No joint found; created fixed joint '{fixed_joint_name}'")

        # Append body_data to bodies_data after processing joints
        # This ensures rotation_axis is set before body_data is added to the list
        bodies_data.append(body_data)

        # Recursively parse child bodies (pass childclass to children)
        for child_body_elem in body_elem.findall('body'):
            self._parse_body(child_body_elem, body_name, level + 1, bodies_data,
                           joints_data, meshes_data, default_classes, eulerseq, body_childclass, angle_unit)

    def parse_primitive_geom(self, geom_elem, geom_type, level=0):
        """Parse MJCF primitive geom and convert to Assembler collider_data format.

        Args:
            geom_elem: MJCF geom element
            geom_type: Primitive type ('cylinder', 'box', 'sphere', 'capsule')
            level: Logging indentation level

        Returns:
            collider_data: Dictionary in Assembler collider_data format,
                          or None if size attribute is missing
        """
        collider_data = {
            'type': geom_type,
            'geometry': {},
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0]  # degrees
        }

        # Parse position
        pos_str = geom_elem.get('pos', '0 0 0')
        collider_data['position'] = [float(v) for v in pos_str.split()]

        # Parse rotation (quat or euler to RPY in degrees)
        quat_str = geom_elem.get('quat')
        euler_str = geom_elem.get('euler')

        if quat_str:
            quat = [float(v) for v in quat_str.split()]
            rpy_rad = self.conversion_utils.quat_to_rpy(quat)
            collider_data['rotation'] = [math.degrees(r) for r in rpy_rad]
        elif euler_str:
            euler_deg = [float(v) for v in euler_str.split()]
            collider_data['rotation'] = euler_deg  # Already in degrees

        # Parse size (convert from MJCF format to Assembler format)
        size_str = geom_elem.get('size')
        if not size_str:
            if self.verbose:
                print(f"{'  ' * level}      Warning: No size attribute for primitive geom")
            return None

        size_values = [float(v) for v in size_str.split()]

        if geom_type == 'cylinder':
            # MJCF: [radius, half_length] → Assembler: {radius, length}
            if len(size_values) >= 2:
                collider_data['geometry']['radius'] = size_values[0]
                collider_data['geometry']['length'] = size_values[1] * 2.0  # half → full
                if self.verbose:
                    print(f"{'  ' * level}      Cylinder: radius={size_values[0]}, length={size_values[1]*2}")

        elif geom_type == 'box':
            # MJCF: [half_x, half_y, half_z] → Assembler: {size_x, size_y, size_z}
            if len(size_values) >= 3:
                collider_data['geometry']['size_x'] = size_values[0] * 2.0
                collider_data['geometry']['size_y'] = size_values[1] * 2.0
                collider_data['geometry']['size_z'] = size_values[2] * 2.0
                if self.verbose:
                    print(f"{'  ' * level}      Box: x={size_values[0]*2}, y={size_values[1]*2}, z={size_values[2]*2}")

        elif geom_type == 'ellipsoid':
            # MJCF: [half_x, half_y, half_z] → Assembler: {size_x, size_y, size_z} (boxとして扱う)
            if len(size_values) >= 3:
                collider_data['type'] = 'box'  # ellipsoidはboxとして扱う
                collider_data['geometry']['size_x'] = size_values[0] * 2.0
                collider_data['geometry']['size_y'] = size_values[1] * 2.0
                collider_data['geometry']['size_z'] = size_values[2] * 2.0
                if self.verbose:
                    print(f"{'  ' * level}      Ellipsoid (as box): x={size_values[0]*2}, y={size_values[1]*2}, z={size_values[2]*2}")

        elif geom_type == 'sphere':
            # MJCF: [radius] → Assembler: {radius}
            if len(size_values) >= 1:
                collider_data['geometry']['radius'] = size_values[0]
                if self.verbose:
                    print(f"{'  ' * level}      Sphere: radius={size_values[0]}")

        elif geom_type == 'capsule':
            # MJCF: [radius, half_length] → Assembler: {radius, length}
            if len(size_values) >= 2:
                collider_data['geometry']['radius'] = size_values[0]
                collider_data['geometry']['length'] = size_values[1] * 2.0
                if self.verbose:
                    print(f"{'  ' * level}      Capsule: radius={size_values[0]}, length={size_values[1]*2}")

        if self.verbose:
            print(f"{'  ' * level}      Position: {collider_data['position']}")
            print(f"{'  ' * level}      Rotation: {collider_data['rotation']} (degrees)")

        return collider_data

    def _build_ir(self, root, robot_name, eulerseq, default_classes, meshes_data,
                  bodies_data, joints_data, mjcf_file_path):
        """Build Intermediate Representation (IR) structure from parsed MJCF data.
        
        Args:
            root: XML root element
            robot_name: Name of the robot
            eulerseq: Euler sequence from compiler
            default_classes: Dictionary of default class settings
            meshes_data: Dictionary of mesh information
            bodies_data: List of body data
            joints_data: List of joint data
            mjcf_file_path: Path to MJCF file
        
        Returns:
            Dictionary containing IR structure
        """
        ir = {
            'model': {
                'name': robot_name,
                'compiler': self._parse_compiler_ir(root),
                'option': self._parse_option_ir(root),
                'size': self._parse_size_ir(root),
                'visual': self._parse_visual_ir(root),
                'statistic': self._parse_statistic_ir(root),
                'defaults': default_classes
            },
            'assets': {
                'meshes': meshes_data,
                'materials': self._parse_materials_ir(root, mjcf_file_path),
                'textures': self._parse_textures_ir(root, mjcf_file_path)
            },
            'world': {
                'bodies': bodies_data,
                'geoms': [],  # Will be populated from bodies
                'sites': [],
                'cameras': [],
                'lights': []
            },
            'joints': joints_data,
            'actuators': [],
            'sensors': [],
            'tendons': [],
            'equality': [],
            'contact': {
                'excludes': [],
                'pairs': []
            },
            'keyframes': {},
            'includes': list(self._parsed_files),
            'provenance': {
                'warnings': self._provenance['warnings'].copy(),
                'unknown_tags': list(self._provenance['unknown_tags']),
                'unknown_attrs': {k: list(v) for k, v in self._provenance['unknown_attrs'].items()}
            }
        }
        
        # Extract geoms from bodies
        for body in bodies_data:
            if 'visuals' in body:
                for visual in body['visuals']:
                    ir['world']['geoms'].append({
                        'name': visual.get('name', ''),
                        'body': body['name'],
                        'type': 'mesh',
                        'mesh': visual.get('mesh', ''),
                        'pos': visual.get('pos', [0.0, 0.0, 0.0]),
                        'quat': visual.get('quat', [1.0, 0.0, 0.0, 0.0]),
                        'role': ['visual']
                    })
        
        return ir

    def _parse_compiler_ir(self, root):
        """Parse compiler settings for IR.
        
        Returns:
            Dictionary with compiler settings
        """
        compiler_elem = root.find('compiler')
        if compiler_elem is None:
            return {}
        
        compiler = {}
        known_attrs = ['eulerseq', 'angle', 'coordinate', 'fuseprefix', 'balanceinertia',
                      'discardvisual', 'convexhull', 'usethread', 'inertiafromgeom',
                      'boundmass', 'boundinertia', 'settotalmass', 'strippath',
                      'precision', 'timestep', 'gravity']
        
        for attr in known_attrs:
            value = compiler_elem.get(attr)
            if value is not None:
                compiler[attr] = value
        
        # Track unknown attributes
        for attr_name in compiler_elem.attrib:
            if attr_name not in known_attrs:
                if 'compiler' not in self._provenance['unknown_attrs']:
                    self._provenance['unknown_attrs']['compiler'] = set()
                self._provenance['unknown_attrs']['compiler'].add(attr_name)
                if self.verbose:
                    self._provenance['warnings'].append(
                        f"Unknown compiler attribute: {attr_name}")
        
        return compiler

    def _parse_option_ir(self, root):
        """Parse option settings for IR.
        
        Returns:
            Dictionary with option settings
        """
        option_elem = root.find('option')
        if option_elem is None:
            return {}
        
        option = {}
        known_attrs = ['timestep', 'apirate', 'impratio', 'gravity', 'wind',
                      'magnetic', 'density', 'viscosity', 'o_margin', 'o_solref',
                      'o_solimp', 'integrator', 'collision', 'cone', 'jacobian',
                      'solver', 'iterations', 'tolerance', 'noslip_iterations',
                      'noslip_tolerance', 'mpr_iterations', 'mpr_tolerance']
        
        for attr in known_attrs:
            value = option_elem.get(attr)
            if value is not None:
                option[attr] = value
        
        # Track unknown attributes
        for attr_name in option_elem.attrib:
            if attr_name not in known_attrs:
                if 'option' not in self._provenance['unknown_attrs']:
                    self._provenance['unknown_attrs']['option'] = set()
                self._provenance['unknown_attrs']['option'].add(attr_name)
                if self.verbose:
                    self._provenance['warnings'].append(
                        f"Unknown option attribute: {attr_name}")
        
        return option

    def _parse_size_ir(self, root):
        """Parse size settings for IR.
        
        Returns:
            Dictionary with size settings
        """
        size_elem = root.find('size')
        if size_elem is None:
            return {}
        
        size = {}
        known_attrs = ['njmax', 'nconmax', 'nstack', 'nuserdata', 'nkey', 'nuser_body',
                      'nuser_jnt', 'nuser_geom', 'nuser_site', 'nuser_tendon',
                      'nuser_actuator', 'nuser_sensor']
        
        for attr in known_attrs:
            value = size_elem.get(attr)
            if value is not None:
                size[attr] = int(value) if value.isdigit() else value
        
        # Track unknown attributes
        for attr_name in size_elem.attrib:
            if attr_name not in known_attrs:
                if 'size' not in self._provenance['unknown_attrs']:
                    self._provenance['unknown_attrs']['size'] = set()
                self._provenance['unknown_attrs']['size'].add(attr_name)
                if self.verbose:
                    self._provenance['warnings'].append(
                        f"Unknown size attribute: {attr_name}")
        
        return size

    def _parse_visual_ir(self, root):
        """Parse visual settings for IR.
        
        Returns:
            Dictionary with visual settings
        """
        visual_elem = root.find('visual')
        if visual_elem is None:
            return {}
        
        visual = {}
        known_attrs = ['headlight', 'rgba', 'fogstart', 'fogend', 'fogrgba', 'shadows',
                      'flags', 'quality', 'global']
        
        for attr in known_attrs:
            value = visual_elem.get(attr)
            if value is not None:
                visual[attr] = value
        
        # Track unknown attributes
        for attr_name in visual_elem.attrib:
            if attr_name not in known_attrs:
                if 'visual' not in self._provenance['unknown_attrs']:
                    self._provenance['unknown_attrs']['visual'] = set()
                self._provenance['unknown_attrs']['visual'].add(attr_name)
                if self.verbose:
                    self._provenance['warnings'].append(
                        f"Unknown visual attribute: {attr_name}")
        
        return visual

    def _parse_statistic_ir(self, root):
        """Parse statistic settings for IR.
        
        Returns:
            Dictionary with statistic settings
        """
        statistic_elem = root.find('statistic')
        if statistic_elem is None:
            return {}
        
        statistic = {}
        known_attrs = ['meaninertia', 'meanmass', 'meansize', 'extent', 'center']
        
        for attr in known_attrs:
            value = statistic_elem.get(attr)
            if value is not None:
                statistic[attr] = value
        
        # Track unknown attributes
        for attr_name in statistic_elem.attrib:
            if attr_name not in known_attrs:
                if 'statistic' not in self._provenance['unknown_attrs']:
                    self._provenance['unknown_attrs']['statistic'] = set()
                self._provenance['unknown_attrs']['statistic'].add(attr_name)
                if self.verbose:
                    self._provenance['warnings'].append(
                        f"Unknown statistic attribute: {attr_name}")
        
        return statistic

    def _parse_materials_ir(self, root, mjcf_file_path):
        """Parse material definitions for IR.
        
        Returns:
            Dictionary mapping material names to material data
        """
        materials = {}
        asset_elem = root.find('asset')
        if asset_elem is None:
            return materials
        
        for material_elem in asset_elem.findall('material'):
            name = material_elem.get('name')
            if not name:
                continue
            
            material = {'name': name}
            known_attrs = ['class', 'rgba', 'texture', 'texuniform', 'texrepeat',
                          'emission', 'specular', 'shininess', 'reflectance',
                          'roughness', 'metallic']
            
            for attr in known_attrs:
                value = material_elem.get(attr)
                if value is not None:
                    if attr == 'rgba':
                        material[attr] = [float(v) for v in value.split()]
                    else:
                        material[attr] = value
            
            # Track unknown attributes
            for attr_name in material_elem.attrib:
                if attr_name not in known_attrs:
                    if 'material' not in self._provenance['unknown_attrs']:
                        self._provenance['unknown_attrs']['material'] = set()
                    self._provenance['unknown_attrs']['material'].add(attr_name)
                    if self.verbose:
                        self._provenance['warnings'].append(
                            f"Unknown material attribute: {attr_name}")
            
            materials[name] = material
        
        return materials

    def _parse_textures_ir(self, root, mjcf_file_path):
        """Parse texture definitions for IR.
        
        Returns:
            Dictionary mapping texture names to texture data
        """
        textures = {}
        asset_elem = root.find('asset')
        if asset_elem is None:
            return textures
        
        mjcf_dir = os.path.dirname(mjcf_file_path)
        
        for texture_elem in asset_elem.findall('texture'):
            name = texture_elem.get('name')
            if not name:
                continue
            
            texture = {'name': name}
            known_attrs = ['type', 'file', 'builtin', 'rgb1', 'rgb2', 'mark',
                          'markrgb', 'width', 'height', 'hflip', 'vflip']
            
            for attr in known_attrs:
                value = texture_elem.get(attr)
                if value is not None:
                    if attr in ['rgb1', 'rgb2', 'markrgb']:
                        texture[attr] = [float(v) for v in value.split()]
                    elif attr in ['width', 'height']:
                        texture[attr] = int(value) if value.isdigit() else value
                    elif attr == 'file':
                        # Resolve texture file path
                        texture_path = self._resolve_asset_path(value, mjcf_dir)
                        texture[attr] = texture_path
                    else:
                        texture[attr] = value
            
            # Track unknown attributes
            for attr_name in texture_elem.attrib:
                if attr_name not in known_attrs:
                    if 'texture' not in self._provenance['unknown_attrs']:
                        self._provenance['unknown_attrs']['texture'] = set()
                    self._provenance['unknown_attrs']['texture'].add(attr_name)
                    if self.verbose:
                        self._provenance['warnings'].append(
                            f"Unknown texture attribute: {attr_name}")
            
            textures[name] = texture
        
        return textures

    def _parse_equality(self, root, bodies_data):
        """Parse equality constraints (closed-loop joints) from MJCF.

        Args:
            root: XML root element
            bodies_data: List of parsed body data dictionaries

        Returns:
            List of closed-loop joint dictionaries with 'body1', 'body2', 'anchor', etc.

        Note:
            MJCF's <connect> anchor is specified in body1's local frame. However, for
            URDF Kitchen's closed-loop joint representation, we need the origin in the
            parent's local frame (which is body1). So the anchor can be used directly
            as origin_xyz.

            In MuJoCo, both bodies are constrained to have the same anchor point in
            their respective local frames coincide in world space. For visualization
            purposes, we use the anchor as-is for the parent (body1) and let the
            constraint be implicit.
        """
        closed_loop_joints = []
        equality_elem = root.find('equality')

        if equality_elem is None:
            if self.verbose:
                print("No <equality> section found")
            return closed_loop_joints

        # Build body name to body data mapping for quick lookup
        body_dict = {body['name']: body for body in bodies_data}

        # Parse <connect> elements (ball joints connecting two bodies)
        for connect_elem in equality_elem.findall('connect'):
            body1_name = connect_elem.get('body1')
            body2_name = connect_elem.get('body2')
            anchor_str = connect_elem.get('anchor')

            if not body1_name or not body2_name:
                if self.verbose:
                    print(f"Warning: <connect> element missing body1 or body2, skipping")
                continue

            # Verify both bodies exist
            if body1_name not in body_dict:
                if self.verbose:
                    print(f"Warning: body1 '{body1_name}' not found in bodies_data, skipping")
                continue
            if body2_name not in body_dict:
                if self.verbose:
                    print(f"Warning: body2 '{body2_name}' not found in bodies_data, skipping")
                continue

            # Parse anchor position (default to [0, 0, 0])
            # Anchor is specified in body1's local frame in MJCF
            anchor_xyz = [0.0, 0.0, 0.0]
            if anchor_str:
                anchor_values = [float(v) for v in anchor_str.split()]
                if len(anchor_values) >= 3:
                    anchor_xyz = anchor_values[:3]

            # IMPORTANT: In MJCF, the anchor attribute specifies a point in body1's
            # local frame. For URDF Kitchen, we use body1 as the parent of the
            # closed-loop joint, so anchor_xyz can be used directly as origin_xyz
            # (the joint origin in the parent's frame).
            #
            # MuJoCo's connect constraint ensures that this anchor point in body1's
            # frame coincides with the same anchor point in body2's frame in world
            # space. The Assembler will create a ClosedLoopJointNode that connects
            # body1 (parent) to body2 (child).

            # Create closed-loop joint data (similar to SDF ball joint)
            joint_data = {
                'type': 'ball',  # MJCF connect is equivalent to ball joint
                'name': f"{body1_name}_{body2_name}_CL_joint",  # Closed-loop joint name
                'parent': body1_name,
                'child': body2_name,
                'origin_xyz': anchor_xyz,  # Anchor in parent (body1) frame
                'origin_rpy': [0.0, 0.0, 0.0],
                'axis': [0.0, 0.0, 1.0],  # Default axis (not used for ball joints)
                'limit': {},  # Ball joints typically have no limits
                'dynamics': {'damping': 0.0, 'friction': 0.0},
                'original_type': 'ball',  # For compatibility with SDF parser
                'anchor': anchor_xyz  # Store anchor position for reference
            }

            closed_loop_joints.append(joint_data)

            if self.verbose:
                print(f"  Closed-loop joint: {body1_name} <-> {body2_name} at anchor {anchor_xyz} (in {body1_name}'s frame)")

        return closed_loop_joints

    def _resolve_asset_path(self, asset_path, mjcf_dir):
        """Resolve asset file path relative to MJCF directory.
        
        Args:
            asset_path: Relative or absolute path to asset file
            mjcf_dir: Directory containing MJCF file
        
        Returns:
            Resolved absolute path or original path if not found
        """
        if os.path.isabs(asset_path):
            return asset_path
        
        # Try relative to MJCF directory
        candidate = os.path.join(mjcf_dir, asset_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        
        return asset_path

# ============================================================================
# HELPER FUNCTIONS FOR IMPORT OPERATIONS
# ============================================================================

def auto_detect_mesh_directories(urdf_file):
    """URDFファイルの親ディレクトリから潜在的なmeshディレクトリを自動検出
    
    検索範囲：
    - 上位ディレクトリは1階層（URDFファイルの親ディレクトリの親）
    - そこから下位ディレクトリまでは2階層探索
    """
    potential_dirs = []

    urdf_dir = os.path.dirname(urdf_file)
    parent_dir = os.path.dirname(urdf_dir)  # 上位1階層

    print(f"\n=== Auto-detecting mesh directories ===")
    print(f"URDF directory: {urdf_dir}")
    print(f"Parent directory (upper 1 level): {parent_dir}")

    # 上位ディレクトリから下位2階層まで探索
    def search_directories(base_dir, current_depth, max_depth):
        """再帰的にディレクトリを探索（最大深さ制限付き）"""
        if current_depth > max_depth:
            return
        
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # 'mesh' を含むディレクトリ名を検索（大文字小文字を区別しない）
                    if 'mesh' in item.lower():
                        if item_path not in potential_dirs:
                            potential_dirs.append(item_path)
                            print(f"  Found potential mesh directory (depth {current_depth}): {item_path}")
                    
                    # 下位ディレクトリを探索（2階層まで）
                    if current_depth < max_depth:
                        search_directories(item_path, current_depth + 1, max_depth)
        except Exception as e:
            print(f"Error scanning directory {base_dir}: {str(e)}")

    # 上位ディレクトリから探索開始（深さ0から開始、最大2階層まで）
    try:
        search_directories(parent_dir, 0, 2)
    except Exception as e:
        print(f"Error scanning parent directory: {str(e)}")

    print(f"Total potential directories found: {len(potential_dirs)}")
    print("=" * 40 + "\n")

    return potential_dirs


def search_stl_files_in_directory(meshes_dir, missing_stl_files, links_data):
    """指定されたディレクトリ内でSTLファイルを検索
    
    検索範囲：下位ディレクトリまで2階層探索
    """
    found_count = 0

    print(f"Searching for STL files in: {meshes_dir} (max depth: 2 levels)")

    for missing_item in missing_stl_files:
        link_name = missing_item['link_name']
        basename = missing_item['basename']

        # すでに見つかっている場合はスキップ
        if links_data[link_name]['stl_file']:
            continue

        # 指定されたディレクトリ内で検索
        candidate_path = os.path.join(meshes_dir, basename)
        if os.path.exists(candidate_path):
            links_data[link_name]['stl_file'] = candidate_path
            print(f"  ✓ Found STL for {link_name}: {candidate_path}")
            found_count += 1
        else:
            # サブディレクトリも検索（最大2階層まで）
            meshes_dir_depth = len(meshes_dir.split(os.sep))
            for root_dir, dirs, files in os.walk(meshes_dir):
                # 現在のディレクトリの深さを計算
                current_depth = len(root_dir.split(os.sep)) - meshes_dir_depth
                # 2階層まで探索
                if current_depth > 2:
                    # これより深いディレクトリは探索しない
                    dirs[:] = []  # os.walkのdirsを空にすることで、それより深い探索を停止
                    continue
                
                if basename in files:
                    candidate_path = os.path.join(root_dir, basename)
                    links_data[link_name]['stl_file'] = candidate_path
                    print(f"  ✓ Found STL for {link_name} in subdirectory (depth {current_depth}): {candidate_path}")
                    found_count += 1
                    break

    return found_count


# ============================================================================
# MAIN IMPORT FUNCTIONS
# ============================================================================

def import_urdf(graph):
        """URDFファイルをインポート"""
        try:
            # URDFファイル、xacroファイル、SDFファイル、またはSRDFファイルを選択するダイアログ
            # デフォルトで.urdf、.xacro、.sdf、.srdfのすべてを表示
            urdf_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                graph.widget,
                "Select URDF, xacro, SDF, or SRDF file to import",
                os.getcwd(),
                "URDF/Xacro/SDF/SRDF Files (*.urdf;*.xacro;*.sdf;*.srdf);;URDF Files (*.urdf);;Xacro Files (*.xacro);;SDF Files (*.sdf);;SRDF Files (*.srdf);;All Files (*)"
            )

            if not urdf_file:
                print("URDF import cancelled")
                return False

            print(f"Importing URDF from: {urdf_file}")
            
            # ファイルの種類を確認
            file_ext = os.path.splitext(urdf_file)[1].lower()
            is_xacro = file_ext in ['.xacro', '.xacro.urdf']
            is_srdf = file_ext == '.srdf'
            is_sdf = file_ext == '.sdf'
            
            if is_srdf:
                print(f"  Detected SRDF file, will search for corresponding URDF/SDF file")
                # SRDFファイルの場合、対応するURDF/SDFファイルを探す（上1階層、下4階層）
                srdf_dir = os.path.dirname(urdf_file)
                srdf_basename = os.path.splitext(os.path.basename(urdf_file))[0]
                
                # 上1階層、下4階層まで探索する関数
                def search_urdf_sdf_files(base_dir, basename, max_depth=4):
                    """上1階層、下4階層までURDF/SDFファイルを探索"""
                    candidates = []
                    
                    # 上1階層
                    parent_dir = os.path.dirname(base_dir)
                    if os.path.isdir(parent_dir):
                        candidates.extend([
                            os.path.join(parent_dir, f"{basename}.urdf"),
                            os.path.join(parent_dir, f"{basename}.xacro"),
                            os.path.join(parent_dir, f"{basename}.sdf"),
                        ])
                    
                    # 下4階層まで探索
                    base_depth = len(base_dir.split(os.sep))
                    for root_dir, dirs, files in os.walk(base_dir):
                        current_depth = len(root_dir.split(os.sep)) - base_depth
                        if current_depth > max_depth:
                            dirs[:] = []  # これより深い探索を停止
                            continue
                        
                        candidates.extend([
                            os.path.join(root_dir, f"{basename}.urdf"),
                            os.path.join(root_dir, f"{basename}.xacro"),
                            os.path.join(root_dir, f"{basename}.sdf"),
                        ])
                    
                    return candidates
                
                urdf_candidates = search_urdf_sdf_files(srdf_dir, srdf_basename, max_depth=4)
                urdf_file_found = None
                for candidate in urdf_candidates:
                    candidate = os.path.normpath(candidate)
                    if os.path.exists(candidate):
                        urdf_file_found = candidate
                        print(f"  Found corresponding file: {urdf_file_found}")
                        break
                
                if not urdf_file_found:
                    QtWidgets.QMessageBox.warning(
                        graph.widget,
                        "URDF/SDF File Not Found",
                        f"SRDF file selected, but could not find corresponding URDF/SDF file.\n\n"
                        f"Please select the URDF/SDF file manually.\n\n"
                        f"SRDF file: {urdf_file}"
                    )
                    # URDF/SDFファイルを手動で選択させる
                    urdf_file_found, _ = QtWidgets.QFileDialog.getOpenFileName(
                        graph.widget,
                        "Select corresponding URDF, xacro, or SDF file",
                        os.path.dirname(urdf_file),
                        "URDF/Xacro/SDF Files (*.urdf;*.xacro;*.sdf);;All Files (*)"
                    )
                    if not urdf_file_found:
                        print("URDF import cancelled")
                        return False
                
                srdf_file = urdf_file
                urdf_file = urdf_file_found
                # ファイル拡張子を再確認
                file_ext = os.path.splitext(urdf_file)[1].lower()
                is_xacro = file_ext in ['.xacro', '.xacro.urdf']
                is_sdf = file_ext == '.sdf'
            elif is_sdf:
                print(f"  Detected SDF file, will be parsed using SDFParser")
                srdf_file = None
            elif is_xacro:
                print(f"  Detected xacro file, will be expanded to URDF first")
                srdf_file = None
            else:
                print(f"  Detected URDF file, will be parsed directly")
                # URDFファイルの場合、同じディレクトリにあるSRDFファイルを自動検索
                urdf_dir = os.path.dirname(urdf_file)
                urdf_basename = os.path.splitext(os.path.basename(urdf_file))[0]
                srdf_candidate = os.path.join(urdf_dir, f"{urdf_basename}.srdf")
                if os.path.exists(srdf_candidate):
                    srdf_file = srdf_candidate
                    print(f"  Found corresponding SRDF file: {srdf_file}")
                else:
                    srdf_file = None

            # URDFParserまたはSDFParserを使用してパース
            try:
                if is_sdf:
                    sdf_parser = SDFParser(verbose=True)
                    sdf_data = sdf_parser.parse_sdf(urdf_file)
                    # SDFデータをURDF形式に変換
                    robot_name = sdf_data['robot_name']
                    links_data = sdf_data['links_data']
                    joints_data = sdf_data['joints_data']
                    closed_loop_joints_data = sdf_data.get('closed_loop_joints', [])  # 閉リンクジョイント情報を取得
                    materials_data = sdf_data['materials_data']
                    missing_stl_files = sdf_data['missing_meshes']
                    urdf_data = {
                        'robot_name': robot_name,
                        'links': links_data,
                        'joints': joints_data,
                        'closed_loop_joints': closed_loop_joints_data,  # 閉リンクジョイント情報を追加
                        'materials': materials_data,
                        'missing_meshes': missing_stl_files
                    }
                else:
                    urdf_parser = URDFParser(verbose=True)
                    urdf_data = urdf_parser.parse_urdf(urdf_file)
            except FileNotFoundError as e:
                QtWidgets.QMessageBox.critical(
                    graph.widget,
                    "File Not Found",
                    f"Could not find URDF/xacro file:\n{urdf_file}\n\n{str(e)}"
                )
                return False
            except RuntimeError as e:
                error_msg = str(e)
                if "xacro" in error_msg.lower():
                    QtWidgets.QMessageBox.critical(
                        graph.widget,
                        "Xacro Expansion Failed",
                        f"Failed to expand xacro file:\n{urdf_file}\n\n{error_msg}\n\n"
                        "Please ensure xacro is installed:\n"
                        "  pip install xacro"
                    )
                else:
                    QtWidgets.QMessageBox.critical(
                        graph.widget,
                        "URDF Parse Error",
                        f"Failed to parse URDF file:\n{urdf_file}\n\n{error_msg}"
                    )
                return False
            except ET.ParseError as e:
                QtWidgets.QMessageBox.critical(
                    graph.widget,
                    "XML Parse Error",
                    f"Failed to parse XML in file:\n{urdf_file}\n\n{str(e)}"
                )
                return False
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    graph.widget,
                    "Import Error",
                    f"Unexpected error while importing URDF:\n{urdf_file}\n\n{str(e)}"
                )
                traceback.print_exc()
                return False

            # パースされたデータを取得
            robot_name = urdf_data['robot_name']
            links_data = urdf_data['links']
            joints_data = urdf_data['joints']
            closed_loop_joints = urdf_data.get('closed_loop_joints', [])  # 閉リンクジョイント情報を取得
            materials_data = urdf_data.get('materials', {})  # SDFの場合は空の可能性がある
            missing_stl_files = urdf_data.get('missing_meshes', [])  # SDFの場合は空の可能性がある

            # SRDFファイルを読み込む（存在する場合）
            srdf_data = None
            if 'srdf_file' in locals() and srdf_file:
                try:
                    srdf_parser = SRDFParser(verbose=True)
                    srdf_data = srdf_parser.parse_srdf(srdf_file)
                    print(f"\n=== SRDF Parse Summary ===")
                    print(f"Groups: {len(srdf_data['groups'])}")
                    print(f"Disabled collisions: {len(srdf_data['disabled_collisions'])}")
                    print(f"Disabled links: {len(srdf_data['disabled_links'])}")
                    print(f"Disabled joints: {len(srdf_data['disabled_joints'])}")
                    print(f"End effectors: {len(srdf_data['end_effectors'])}")
                    print(f"Virtual joints: {len(srdf_data['virtual_joints'])}")
                    print("=" * 30 + "\n")
                except Exception as e:
                    print(f"Warning: Failed to parse SRDF file: {srdf_file}\n{str(e)}")
                    traceback.print_exc()
                    srdf_data = None

            # デバッグ: パースされたデータの概要を出力
            print(f"\n=== URDF Parse Summary ===")
            print(f"Robot name: {robot_name}")
            print(f"Total links: {len(links_data)}")
            print(f"Total joints: {len(joints_data)}")
            print(f"Total closed-loop joints: {len(closed_loop_joints)}")
            print(f"Total materials: {len(materials_data)}")
            print(f"Missing meshes: {len(missing_stl_files)}")
            if srdf_data:
                print(f"SRDF data loaded: Yes")
            else:
                print(f"SRDF data loaded: No")
            
            # リンクとジョイントが空でないか確認
            if len(links_data) == 0:
                QtWidgets.QMessageBox.warning(
                    graph.widget,
                    "No Links Found",
                    f"No links were found in the URDF file.\n\n"
                    f"This might indicate:\n"
                    f"1. The xacro file was not properly expanded\n"
                    f"2. The URDF structure is invalid\n"
                    f"3. The file is empty or corrupted\n\n"
                    f"File: {urdf_file}"
                )
                return False
            
            if len(joints_data) == 0:
                print(f"  Warning: No joints found in URDF (this might be normal for a single-link robot)")
            
            # リンク名のリストを出力
            print(f"\nLinks found:")
            for link_name in sorted(links_data.keys()):
                print(f"  - {link_name}")
            
            # ジョイント名のリストを出力
            if joints_data:
                print(f"\nJoints found:")
                for joint_data in joints_data:
                    print(f"  - {joint_data['name']}: {joint_data['parent']} -> {joint_data['child']}")

            # 閉リンクジョイント名のリストを出力
            if closed_loop_joints:
                print(f"\nClosed-loop joints found:")
                for joint_data in closed_loop_joints:
                    print(f"  - {joint_data['name']}: {joint_data['parent']} -> {joint_data['child']} (type: {joint_data.get('original_type', 'unknown')})")

            print("=" * 30 + "\n")

            graph.robot_name = robot_name
            graph.closed_loop_joints = closed_loop_joints  # 閉リンクジョイント情報を保存
            print(f"Robot name set to: {robot_name}")

            # UIのName:フィールドを更新
            if hasattr(graph, 'name_input') and graph.name_input:
                graph.name_input.setText(robot_name)

            # 追加のinertialログを出力（Assembler固有）
            for link_name, link_data in links_data.items():
                if link_data['mass'] > 0.0 or any(link_data['inertia'].values()):
                    print(f"\n[URDF_INERTIAL_SOURCE] link_name={link_name}, source_urdf_path={urdf_file}")
                    print(f"  mass={link_data['mass']:.9e}")
                    print(f"  origin_xyz={link_data['inertial_origin']['xyz']}")
                    print(f"  origin_rpy={link_data['inertial_origin']['rpy']}")
                    if any(link_data['inertia'].values()):
                        print(f"  ixx={link_data['inertia']['ixx']:.9e}, ixy={link_data['inertia']['ixy']:.9e}, ixz={link_data['inertia']['ixz']:.9e}")
                        print(f"  iyy={link_data['inertia']['iyy']:.9e}, iyz={link_data['inertia']['iyz']:.9e}, izz={link_data['inertia']['izz']:.9e}")
                    else:
                        print(f"  WARNING: <inertia> element not found in <inertial>")
                else:
                    print(f"\n[URDF_INERTIAL_SOURCE] link_name={link_name}, source_urdf_path={urdf_file}")
                    print(f"  WARNING: <inertial> element not found - will use fallback/estimation")

            # デバッグ: リンクとSTLファイルの状況を出力
            print(f"\n=== STL File Summary ===")
            print(f"Total links: {len(links_data)}")
            for link_name, link_data in links_data.items():
                if link_data.get('stl_file'):
                    print(f"  ✓ {link_name}: {os.path.basename(link_data['stl_file'])}")
                elif link_data.get('stl_filename_original'):
                    print(f"  ✗ {link_name}: NOT FOUND ({link_data['stl_filename_original']})")
                else:
                    print(f"  - {link_name}: No STL specified")
            print(f"Missing STL files count: {len(missing_stl_files)}")
            print("=" * 30 + "\n")

            # STLファイルが見つからなかった場合、自動検索してから手動指定
            if missing_stl_files:
                initial_missing_count = len(missing_stl_files)

                # 1. 親ディレクトリ内のmeshフォルダを自動検索
                potential_dirs = graph.auto_detect_mesh_directories(urdf_file)

                if potential_dirs:
                    print(f"Trying auto-detected mesh directories...")
                    total_auto_found = 0

                    for mesh_dir in potential_dirs:
                        found_count = graph.search_stl_files_in_directory(mesh_dir, missing_stl_files, links_data)
                        total_auto_found += found_count

                    if total_auto_found > 0:
                        print(f"Auto-detection found {total_auto_found} STL file(s)")

                    # missing_stl_filesリストを更新（見つかったものを除去）
                    missing_stl_files = [
                        item for item in missing_stl_files
                        if not links_data[item['link_name']]['stl_file']
                    ]

                # 2. まだ見つからないファイルがあれば、ユーザーに手動指定を促す
                if missing_stl_files:
                    missing_count = len(missing_stl_files)
                    missing_list = '\n'.join([f"  - {item['link_name']}: {item['basename']}" for item in missing_stl_files[:5]])
                    if missing_count > 5:
                        missing_list += f"\n  ... and {missing_count - 5} more"

                    auto_found_count = initial_missing_count - missing_count
                    message = f"Could not find {missing_count} STL file(s):\n\n{missing_list}\n\n"
                    if auto_found_count > 0:
                        message = f"Auto-detected {auto_found_count} file(s), but still missing {missing_count}:\n\n{missing_list}\n\n"

                    response = QtWidgets.QMessageBox.question(
                        graph.widget,
                        "Mesh Files Not Found",
                        message + "Would you like to specify the meshes directory manually?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )

                    if response == QtWidgets.QMessageBox.Yes:
                        # meshesディレクトリを手動で選択
                        meshes_dir = QtWidgets.QFileDialog.getExistingDirectory(
                            graph.widget,
                            "Select meshes directory",
                            os.path.dirname(urdf_file)
                        )

                        if meshes_dir:
                            found_count = graph.search_stl_files_in_directory(meshes_dir, missing_stl_files, links_data)

                            if found_count > 0:
                                QtWidgets.QMessageBox.information(
                                    graph.widget,
                                    "Mesh Files Found",
                                    f"Found {found_count} out of {missing_count} missing mesh file(s) in the specified directory."
                                )
                            else:
                                QtWidgets.QMessageBox.warning(
                                    graph.widget,
                                    "No Mesh Files Found",
                                    f"Could not find any of the missing mesh files in the specified directory."
                                )

                            # missing_stl_filesリストを更新（見つかったものを除去）
                            missing_stl_files = [
                                item for item in missing_stl_files
                                if not links_data[item['link_name']]['stl_file']
                            ]
                else:
                    # すべてのメッシュファイルが自動検出された
                    if initial_missing_count > 0:
                        QtWidgets.QMessageBox.information(
                            graph.widget,
                            "Mesh Files Found",
                            f"Automatically found all {initial_missing_count} missing mesh file(s)!"
                        )

            # 各リンクに接続する子ジョイントの数を数える（decorationsも含む）
            link_child_counts = {}
            for link_name in links_data.keys():
                link_child_counts[link_name] = 0

            # 子ジョイントの数をカウント
            for joint_data in joints_data:
                parent_link = joint_data['parent']
                # 親リンクがlink_child_countsにない場合は初期化（base_link用）
                if parent_link not in link_child_counts:
                    link_child_counts[parent_link] = 0
                link_child_counts[parent_link] += 1

            # 閉リンクジョイントの親リンクもカウント（出力ポートが必要）
            for joint_data in closed_loop_joints:
                parent_link = joint_data['parent']
                if parent_link not in link_child_counts:
                    link_child_counts[parent_link] = 0
                link_child_counts[parent_link] += 1

            # decorationsの数を追加（各decorationも出力ポートが必要）
            # 子ジョイントがないリンクでもdecorationがある場合はカウントする必要がある
            for link_name, link_data in links_data.items():
                decoration_count = len(link_data.get('decorations', []))
                if decoration_count > 0:
                    # link_child_countsに存在しない場合は初期化
                    if link_name not in link_child_counts:
                        link_child_counts[link_name] = 0
                    link_child_counts[link_name] += decoration_count
                    print(f"Link '{link_name}' has {decoration_count} decoration(s) (total children: {link_child_counts[link_name]})")

            # ノードを作成
            nodes = {}

            # 既存のbase_linkノードを探す（URDFにbase_linkがあるかどうかに関わらず）
            base_node = graph.get_node_by_name('base_link')
            base_link_sub_node = None  # base_link_subノード（必要に応じて作成）

            # base_linkノードは常に作成する（URDFにbase_linkがあるかどうかに関わらず）
            if base_node:
                print("Using existing base_link node")
            else:
                # 既存のbase_linkがない場合は新規作成
                print("Creating new base_link node")
                base_node = graph.create_node(
                    'insilico.nodes.BaseLinkNode',
                    name='base_link',
                    pos=QtCore.QPointF(50, 50)  # 中心近くに配置（MJCFと統一）
                )

            nodes['base_link'] = base_node

            # URDFにbase_linkがある場合の処理
            if 'base_link' in links_data or 'base_link' in link_child_counts:

                # URDFのbase_linkが空でないかチェック
                # 既存のbase_linkがある場合、データの上書きを禁止
                # URDFのbase_linkにデータがある場合、または子ノードがある場合はbase_link_subを作成
                base_link_has_children = link_child_counts.get('base_link', 0) > 0
                
                if 'base_link' in links_data:
                    base_link_data = links_data['base_link']
                    # base_linkが空でないかチェック
                    # 以下のいずれかがあれば空でないと判定：
                    # - mass > 0
                    # - inertiaが0でない
                    # - stl_fileがある
                    # - decorationsリストが空でない（複数のvisual要素がある）
                    # - collider_typeが設定されている（collision要素がある）
                    # - collider_enabledがTrue
                    # - 子ノードがある（pointsデータが設定される）
                    is_base_link_non_empty = (
                        base_link_data.get('mass', 0.0) > 0.0 or
                        any(base_link_data.get('inertia', {}).values()) or
                        base_link_data.get('stl_file') is not None or
                        len(base_link_data.get('decorations', [])) > 0 or
                        base_link_data.get('collider_type') is not None or
                        base_link_data.get('collider_enabled', False) is True or
                        base_link_has_children
                    )
                    
                    # デバッグ: base_linkが空でないかチェック
                    print(f"  Checking if base_link is non-empty:")
                    print(f"    mass > 0: {base_link_data.get('mass', 0.0) > 0.0}")
                    print(f"    inertia non-zero: {any(base_link_data.get('inertia', {}).values())}")
                    print(f"    stl_file exists: {base_link_data.get('stl_file') is not None}")
                    print(f"    decorations count: {len(base_link_data.get('decorations', []))}")
                    print(f"    collider_type: {base_link_data.get('collider_type')}")
                    print(f"    collider_enabled: {base_link_data.get('collider_enabled', False)}")
                    print(f"    has_children (points will be set): {base_link_has_children}")
                    print(f"    Result: is_base_link_non_empty = {is_base_link_non_empty}")

                    # 既存のbase_linkがある場合、データの上書きを禁止
                    # URDFのbase_linkにデータがある場合、または子ノードがある場合はbase_link_subを作成
                    if is_base_link_non_empty and base_node:
                        # base_link_subノードを作成
                        print("URDF base_link is not empty, creating base_link_sub node")
                        base_link_pos = base_node.pos()
                        # pos()がリストかQPointFか判定
                        if isinstance(base_link_pos, (list, tuple)):
                            base_link_x = base_link_pos[0]
                            base_link_y = base_link_pos[1]
                        else:
                            base_link_x = base_link_pos.x()
                            base_link_y = base_link_pos.y()
                        grid_spacing_value = 150  # パネル間の距離
                        base_link_sub_pos = QtCore.QPointF(base_link_x + grid_spacing_value, base_link_y)
                        
                        base_link_sub_node = graph.create_node(
                            'insilico.nodes.FooNode',
                            name='base_link_sub',
                            pos=base_link_sub_pos
                        )
                        nodes['base_link_sub'] = base_link_sub_node

                        # 初期化時に追加されたポートとポイントをクリア（必ず実行）
                        current_ports = len(base_link_sub_node.output_ports())
                        # すべての出力ポートの接続をクリアしてから削除
                        for i in range(1, current_ports + 1):
                            port_name = f'out_{i}'
                            port = base_link_sub_node.get_output(port_name)
                            if port:
                                port.clear_connections()
                        
                        # すべての出力ポートを削除
                        while current_ports > 0:
                            base_link_sub_node.remove_output()
                            current_ports -= 1
                        
                        # ポイントデータと累積座標をクリア
                        base_link_sub_node.points = []
                        base_link_sub_node.cumulative_coords = []
                        base_link_sub_node.output_count = 0

                        # URDFのbase_linkデータをbase_link_subに設定
                        base_link_sub_node.mass_value = base_link_data['mass']
                        base_link_sub_node.inertia = base_link_data['inertia'].copy()
                        base_link_sub_node.inertial_origin = base_link_data['inertial_origin'].copy()
                        # カラー情報を設定（RGBA形式に統一）
                        color_data = base_link_data['color']
                        if len(color_data) == 3:
                            base_link_sub_node.node_color = color_data + [1.0]  # Alpha=1.0を追加
                        elif len(color_data) >= 4:
                            base_link_sub_node.node_color = color_data[:4]
                        else:
                            base_link_sub_node.node_color = DEFAULT_COLOR_WHITE.copy()

                        # STLファイルが設定されている場合
                        if base_link_data['stl_file']:
                            base_link_sub_node.stl_file = base_link_data['stl_file']
                            print(f"Set base_link_sub STL: {base_link_data['stl_file']}")

                        # メッシュのスケール情報を設定
                        base_link_sub_node.mesh_scale = base_link_data.get('mesh_scale', [1.0, 1.0, 1.0])
                        # Visual origin情報を設定
                        base_link_sub_node.visual_origin = base_link_data.get('visual_origin', {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]})
                        # Collision情報を設定
                        if base_link_data.get('collider_type') == 'mesh' and base_link_data.get('collision_mesh'):
                            base_link_sub_node.collider_type = 'mesh'
                            base_link_sub_node.collider_mesh = base_link_data['collision_mesh']
                            base_link_sub_node.collider_mesh_scale = base_link_data.get('collision_mesh_scale', [1.0, 1.0, 1.0])
                            base_link_sub_node.collider_enabled = True
                            print(f"Set base_link_sub collision mesh: {os.path.basename(base_link_data['collision_mesh'])}")
                        elif base_link_data.get('collider_type') == 'primitive' and base_link_data.get('collider_data'):
                            base_link_sub_node.collider_type = 'primitive'
                            base_link_sub_node.collider_data = base_link_data['collider_data']
                            base_link_sub_node.collider_enabled = True
                            collider_type = base_link_data['collider_data'].get('type', 'unknown')
                            print(f"Set base_link_sub primitive collider: {collider_type}")
                        
                        # メッシュ反転判定
                        base_link_sub_node.is_mesh_reversed = is_mesh_reversed_check(
                            base_link_sub_node.visual_origin,
                            base_link_sub_node.mesh_scale
                        )
                        if base_link_sub_node.is_mesh_reversed:
                            print(f"Base_link_sub mesh is reversed/mirrored (for MJCF export)")

                        # base_link_subのrotation_axisをFixed（3）に設定
                        base_link_sub_node.rotation_axis = 3  # Fixed
                        print("Set base_link_sub rotation_axis to Fixed")
                        
                        # base_link_subのpointsデータを初期化（子ノードがある場合）
                        if base_link_has_children:
                            child_count = link_child_counts.get('base_link', 0)
                            base_link_sub_node.points = []
                            for i in range(child_count):
                                base_link_sub_node.points.append({
                                    'name': f'point_{i+1}',
                                    'type': 'fixed',
                                    'xyz': [0.0, 0.0, 0.0],
                                    'rpy': [0.0, 0.0, 0.0],
                                    'angle': [0.0, 0.0, 0.0]
                                })
                            print(f"Initialized base_link_sub.points with {child_count} points")
                            
                            # 必要な数のポートを追加（_add_output()を使わず、直接ポートを追加）
                            for i in range(child_count):
                                base_link_sub_node.output_count += 1
                                port_name = f'out_{base_link_sub_node.output_count}'
                                # 出力ポートを追加（ポイントは既に設定されているため追加しない）
                                base_link_sub_node.add_output(port_name, color=(180, 80, 0))
                                # 累積座標を追加
                                cumulative_coord = create_cumulative_coord(i)
                                base_link_sub_node.cumulative_coords.append(cumulative_coord)
                            
                            # output_countを更新
                            base_link_sub_node.output_count = child_count

                        # 既存のbase_linkをデフォルト値にリセット
                        base_node.mass_value = 0.0
                        base_node.inertia = DEFAULT_INERTIA_ZERO.copy()
                        base_node.inertial_origin = {
                            'xyz': DEFAULT_ORIGIN_ZERO['xyz'].copy(),
                            'rpy': DEFAULT_ORIGIN_ZERO['rpy'].copy()
                        }
                        base_node.stl_file = None
                        base_node.node_color = DEFAULT_COLOR_WHITE.copy()
                        base_node.rotation_axis = 3  # Fixed
                        base_node.joint_lower = 0.0
                        base_node.joint_upper = 0.0
                        base_node.joint_effort = DEFAULT_JOINT_EFFORT
                        base_node.joint_velocity = DEFAULT_JOINT_VELOCITY
                        base_node.joint_damping = DEFAULT_DAMPING_KV
                        base_node.joint_stiffness = DEFAULT_STIFFNESS_KP
                        base_node.joint_margin = DEFAULT_MARGIN
                        base_node.joint_armature = DEFAULT_ARMATURE
                        base_node.joint_frictionloss = DEFAULT_FRICTIONLOSS
                        if hasattr(base_node, 'blank_link'):
                            base_node.blank_link = True
                        print("Reset existing base_link to default values")

                        # base_linkの出力ポート数を1に設定（デフォルト）
                        if hasattr(base_node, 'output_count'):
                            while base_node.output_count > 1:
                                if hasattr(base_node, 'remove_output'):
                                    base_node.remove_output()
                                else:
                                    break
                            while base_node.output_count < 1:
                                if hasattr(base_node, '_add_output'):
                                    base_node._add_output()
                                elif hasattr(base_node, 'add_output'):
                                    base_node.output_count = getattr(base_node, 'output_count', 0) + 1
                                    port_name = f'out_{base_node.output_count}'
                                    base_node.add_output(port_name, color=(180, 80, 0))
                    elif is_base_link_non_empty and not base_node:
                        # base_linkが空でないが、既存のbase_linkがない場合は新規作成
                        # この場合はbase_link_subを作成する必要はない（base_link自体が新規作成される）
                        pass  # 通常の処理に任せる（後でbase_linkが作成される）
                    else:
                        # base_linkが空で、既存のbase_linkがある場合は何もしない（既存のbase_linkを保持）
                        if base_node:
                            print("URDF base_link is empty, keeping existing base_link unchanged")
                        # base_linkが空で、既存のbase_linkがない場合は通常通りbase_linkに設定（空のデータ）
                        elif not base_node:
                            # この場合は後でbase_linkが作成されるので、ここでは何もしない
                            pass

                # base_linkの子ジョイント数に応じて出力ポートを追加
                # base_link_subがある場合は、base_link_subの出力ポート数を設定
                target_node = base_link_sub_node if base_link_sub_node else base_node
                child_count = link_child_counts.get('base_link', 0)
                current_output_count = target_node.output_count if hasattr(target_node, 'output_count') else 1

                # 必要な出力ポート数を計算
                needed_ports = child_count - current_output_count
                if needed_ports > 0:
                    for i in range(needed_ports):
                        if hasattr(target_node, '_add_output'):
                            target_node._add_output()
                        elif hasattr(target_node, 'add_output'):
                            target_node.output_count = getattr(target_node, 'output_count', 0) + 1
                            port_name = f'out_{target_node.output_count}'
                            target_node.add_output(port_name, color=(180, 80, 0))

            # 他のリンクのノードを作成（グリッドレイアウトで配置）
            grid_spacing = 150  # パネル間の距離（200から50px狭めた）
            # base_linkノードの位置を基準に子ノードを配置
            if hasattr(base_node, 'pos') and callable(base_node.pos):
                pos = base_node.pos()
                if hasattr(pos, 'x') and callable(pos.x):
                    base_x = pos.x()
                    base_y = pos.y()
                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    base_x = pos[0]
                    base_y = pos[1]
                else:
                    base_x = 50
                    base_y = 50
            else:
                base_x = 50
                base_y = 50
            current_x = base_x + grid_spacing  # base_linkから右にオフセット
            current_y = base_y
            nodes_per_row = 4
            node_count = 0

            for link_name, link_data in links_data.items():
                if link_name == 'base_link':
                    continue

                # グリッドレイアウトで位置を計算
                row = node_count // nodes_per_row
                col = node_count % nodes_per_row
                pos_x = current_x + col * grid_spacing
                pos_y = current_y + row * grid_spacing

                node = graph.create_node(
                    'insilico.nodes.FooNode',
                    name=link_name,
                    pos=QtCore.QPointF(pos_x, pos_y)
                )
                nodes[link_name] = node

                # ノードのパラメータを設定
                node.mass_value = link_data['mass']
                node.inertia = link_data['inertia'].copy()  # コピーを作成して参照を切る
                # inertial_originが存在する場合のみ設定
                if 'inertial_origin' in link_data:
                    node.inertial_origin = link_data['inertial_origin'].copy() if isinstance(link_data['inertial_origin'], dict) else link_data['inertial_origin']
                else:
                    # デフォルト値を設定
                    node.inertial_origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
                
                # === 必須ログ: ノードに設定された慣性値を確認 ===
                print(f"\n[URDF_NODE_SET] link_name={link_name}, source_urdf_path={urdf_file}")
                print(f"  node.mass_value={node.mass_value:.9e}")
                print(f"  node.inertial_origin={node.inertial_origin}")
                print(f"  node.inertia: ixx={node.inertia['ixx']:.9e}, ixy={node.inertia['ixy']:.9e}, ixz={node.inertia['ixz']:.9e}")
                print(f"                iyy={node.inertia['iyy']:.9e}, iyz={node.inertia['iyz']:.9e}, izz={node.inertia['izz']:.9e}")
                # カラー情報を設定（RGBA形式に統一）
                color_data = link_data['color']
                if len(color_data) == 3:
                    node.node_color = color_data + [1.0]  # Alpha=1.0を追加
                elif len(color_data) >= 4:
                    node.node_color = color_data[:4]
                else:
                    node.node_color = DEFAULT_COLOR_WHITE.copy()
                if link_data['stl_file']:
                    node.stl_file = link_data['stl_file']
                # メッシュのスケール情報を設定（URDF左右対称対応）
                node.mesh_scale = link_data.get('mesh_scale', [1.0, 1.0, 1.0])
                # Visual origin情報を設定（メッシュの位置・回転）
                node.visual_origin = link_data.get('visual_origin', {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]})
                # Collision情報を設定（メッシュまたはプリミティブ）
                if link_data.get('collider_type') == 'mesh' and link_data.get('collision_mesh'):
                    node.collider_type = 'mesh'
                    node.collider_mesh = link_data['collision_mesh']
                    node.collider_mesh_scale = link_data.get('collision_mesh_scale', [1.0, 1.0, 1.0])
                    node.collider_enabled = True
                    print(f"Set {link_name} collision mesh: {os.path.basename(link_data['collision_mesh'])}")
                elif link_data.get('collider_type') == 'primitive' and link_data.get('collider_data'):
                    node.collider_type = 'primitive'
                    node.collider_data = link_data['collider_data']
                    node.collider_enabled = True
                    collider_type = link_data['collider_data'].get('type', 'unknown')
                    print(f"Set {link_name} primitive collider: {collider_type}")
                # メッシュ反転判定
                node.is_mesh_reversed = is_mesh_reversed_check(
                    node.visual_origin,
                    node.mesh_scale
                )
                if node.is_mesh_reversed:
                    print(f"Node '{link_name}' mesh is reversed/mirrored (for MJCF export)")

                # 子ジョイントの数に応じて出力ポートを追加
                child_count = link_child_counts.get(link_name, 0)
                current_output_count = node.output_count if hasattr(node, 'output_count') else 1
                needed_ports = child_count - current_output_count
                if needed_ports > 0:
                    for i in range(needed_ports):
                        if hasattr(node, '_add_output'):
                            node._add_output()
                        elif hasattr(node, 'add_output'):
                            # FooNodeのadd_outputメソッドを使用
                            node.output_count = getattr(node, 'output_count', 0) + 1
                            port_name = f'out_{node.output_count}'
                            node.add_output(port_name, color=(180, 80, 0))

                # decorationノードを作成（このlinkに複数のvisualがある場合）
                decorations = link_data.get('decorations', [])
                for deco_idx, decoration in enumerate(decorations):
                    # decorationノードの位置を親ノードの近くに配置
                    deco_offset_x = 150  # 親の右側に配置
                    deco_offset_y = 100 * (deco_idx + 1)  # 複数ある場合は縦にずらす
                    deco_pos_x = pos_x + deco_offset_x
                    deco_pos_y = pos_y + deco_offset_y

                    # decorationノード名を取得（一意な名前が既に設定されている）
                    decoration_name = decoration['name']
                    
                    # ノード名の衝突をチェック
                    if decoration_name in nodes:
                        print(f"  ⚠ Warning: Decoration node name '{decoration_name}' already exists, appending index")
                        decoration_name = f"{decoration_name}_{deco_idx}"
                    
                    deco_node = graph.create_node(
                        'insilico.nodes.FooNode',
                        name=decoration_name,
                        pos=QtCore.QPointF(deco_pos_x, deco_pos_y)
                    )

                    # decorationノードのパラメータを設定
                    # カラー情報を設定（RGBA形式に統一）
                    deco_color_data = decoration['color']
                    if len(deco_color_data) == 3:
                        deco_node.node_color = deco_color_data + [1.0]  # Alpha=1.0を追加
                    elif len(deco_color_data) >= 4:
                        deco_node.node_color = deco_color_data[:4]
                    else:
                        deco_node.node_color = DEFAULT_COLOR_WHITE.copy()
                    
                    # メッシュファイルが存在する場合のみ設定
                    if decoration.get('stl_file'):
                        deco_node.stl_file = decoration['stl_file']
                    else:
                        print(f"  ⚠ Warning: Decoration '{decoration_name}' has no mesh file (stl_file is None)")
                    
                    deco_node.massless_decoration = True  # Massless Decorationフラグを設定
                    # メッシュのスケール情報を設定（URDF左右対称対応）
                    deco_node.mesh_scale = decoration.get('mesh_scale', [1.0, 1.0, 1.0])
                    # Visual origin情報を設定（メッシュの位置・回転）
                    deco_node.visual_origin = decoration.get('visual_origin', {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]})

                    # 親リンクの参照を保存（接続時に使用）
                    deco_node._parent_link_name = link_name

                    # decorationノードをnodesディクショナリに追加
                    nodes[decoration_name] = deco_node

                    print(f"  → Created decoration node '{decoration_name}' for link '{link_name}'")
                    if decoration.get('stl_file'):
                        print(f"    Mesh file: {os.path.basename(decoration['stl_file'])}")
                    else:
                        print(f"    Mesh file: None (not found)")

                node_count += 1

            # ジョイント情報を親ノードのpointsと子ノードのパラメータに反映
            parent_port_indices = {}  # {parent_link: current_port_index}

            for joint_data in joints_data:
                parent_link = joint_data['parent']
                child_link = joint_data['child']

                # base_link_subがある場合、base_linkに接続予定の子ノードをbase_link_subに接続するように変更
                actual_parent_link = parent_link
                if base_link_sub_node and parent_link == 'base_link':
                    actual_parent_link = 'base_link_sub'
                    parent_node = base_link_sub_node
                elif parent_link not in nodes:
                    continue
                else:
                    parent_node = nodes[parent_link]

                if child_link not in nodes:
                    continue

                child_node = nodes[child_link]

                # 親ノードの現在のポートインデックスを取得
                if actual_parent_link not in parent_port_indices:
                    parent_port_indices[actual_parent_link] = 0
                port_index = parent_port_indices[actual_parent_link]
                parent_port_indices[actual_parent_link] += 1

                # 親ノードのpointsにジョイントのorigin情報を設定
                if port_index < len(parent_node.points):
                    origin_rpy = joint_data['origin_rpy']
                    parent_node.points[port_index]['xyz'] = joint_data['origin_xyz']
                    parent_node.points[port_index]['rpy'] = [0.0, 0.0, 0.0]  # Keep rpy as zero (3D view uses angle)
                    parent_node.points[port_index]['name'] = joint_data['name']
                    parent_node.points[port_index]['type'] = joint_data['type']
                    # Set angle for 3D view display and UI editing (radians)
                    parent_node.points[port_index]['angle'] = list(origin_rpy)

                # 子ノードにジョイント情報を設定
                # 回転軸の設定
                axis = joint_data.get('axis', [1.0, 0.0, 0.0])
                if joint_data['type'] == 'fixed':
                    child_node.rotation_axis = 3  # Fixed
                elif len(axis) >= 3 and abs(axis[0]) > 0.9:
                    child_node.rotation_axis = 0  # X軸
                elif len(axis) >= 3 and abs(axis[1]) > 0.9:
                    child_node.rotation_axis = 1  # Y軸
                elif len(axis) >= 3:
                    child_node.rotation_axis = 2  # Z軸
                else:
                    child_node.rotation_axis = 3  # Fixed (デフォルト)

                # ジョイント制限パラメータの設定
                # ファイルに定義されている場合は上書き、未定義の場合はSettingsのデフォルト値を使用
                if 'limit' in joint_data:
                    if 'lower' in joint_data['limit']:
                        child_node.joint_lower = joint_data['limit']['lower']
                    if 'upper' in joint_data['limit']:
                        child_node.joint_upper = joint_data['limit']['upper']
                    if 'effort' in joint_data['limit']:
                        child_node.joint_effort = joint_data['limit']['effort']
                    if 'velocity' in joint_data['limit']:
                        child_node.joint_velocity = joint_data['limit']['velocity']
                # joint_frictionは削除（margin, armature, frictionlossに置き換え）

                # ジョイントdynamicsパラメータの設定（damping, stiffness, margin, armature, frictionloss）
                # ファイルに定義されている場合は上書き、未定義の場合はSettingsのデフォルト値を使用
                if 'dynamics' in joint_data:
                    if 'damping' in joint_data['dynamics']:
                        child_node.joint_damping = joint_data['dynamics']['damping']
                    if 'stiffness' in joint_data['dynamics']:
                        child_node.joint_stiffness = joint_data['dynamics']['stiffness']
                    if 'margin' in joint_data['dynamics']:
                        child_node.joint_margin = joint_data['dynamics']['margin']
                    if 'armature' in joint_data['dynamics']:
                        child_node.joint_armature = joint_data['dynamics']['armature']
                    if 'frictionloss' in joint_data['dynamics']:
                        child_node.joint_frictionloss = joint_data['dynamics']['frictionloss']

            # ノードを接続（ジョイントの親子関係に基づく）
            print("\n=== Connecting Nodes ===")
            parent_port_indices = {}  # リセット

            # base_link_subがある場合、base_linkとbase_link_subを接続
            if base_link_sub_node:
                print("\n=== Connecting base_link to base_link_sub ===")
                try:
                    base_output_port = base_node.get_output('out')
                    base_link_sub_input_port = base_link_sub_node.get_input('in')
                    if base_output_port and base_link_sub_input_port:
                        base_output_port.connect_to(base_link_sub_input_port)
                        print(f"  ✓ Successfully connected base_link.out to base_link_sub.in")
                    else:
                        print(f"  ✗ ERROR: Could not connect base_link to base_link_sub")
                except Exception as e:
                    print(f"  ✗ ERROR: Exception connecting base_link to base_link_sub: {str(e)}")
                    traceback.print_exc()

            for joint_data in joints_data:
                parent_link = joint_data['parent']
                child_link = joint_data['child']

                # base_link_subがある場合、base_linkに接続予定の子ノードをbase_link_subに接続するように変更
                actual_parent_link = parent_link
                if base_link_sub_node and parent_link == 'base_link':
                    actual_parent_link = 'base_link_sub'
                    parent_node = base_link_sub_node
                elif parent_link not in nodes:
                    print(f"Skipping connection: {parent_link} -> {child_link} (node not found)")
                    continue
                else:
                    parent_node = nodes[parent_link]

                if child_link not in nodes:
                    print(f"Skipping connection: {parent_link} -> {child_link} (child node not found)")
                    continue

                child_node = nodes[child_link]

                # 親ノードの出力ポートインデックスを取得
                if actual_parent_link not in parent_port_indices:
                    parent_port_indices[actual_parent_link] = 0
                port_index = parent_port_indices[actual_parent_link]
                parent_port_indices[actual_parent_link] += 1

                # ポート名を取得
                # isinstance()を使用してノードのクラスを判定
                is_base_link_node = parent_node.__class__.__name__ == 'BaseLinkNode'

                if is_base_link_node:
                    # BaseLinkNodeの場合、最初のポートは'out'、それ以降は'out_2', 'out_3', ...
                    if port_index == 0:
                        output_port_name = 'out'
                    else:
                        output_port_name = f'out_{port_index + 1}'
                else:
                    # FooNodeの場合、'out_1', 'out_2', ... を使用
                    output_port_name = f'out_{port_index + 1}'

                # 子ノードの入力ポート（'in'）を取得
                input_port_name = 'in'

                # デバッグ: 利用可能なポートを表示
                print(f"\nConnecting: {parent_link} -> {child_link}")
                print(f"  Parent node class: {parent_node.__class__.__name__}")
                print(f"  Is BaseLinkNode: {is_base_link_node}")
                print(f"  Port index: {port_index}, Expected output port: {output_port_name}")
                print(f"  Available output ports on {parent_link}: {[p.name() for p in parent_node.output_ports()]}")
                print(f"  Available input ports on {child_link}: {[p.name() for p in child_node.input_ports()]}")

                # ポートを接続
                try:
                    output_port = parent_node.get_output(output_port_name)
                    input_port = child_node.get_input(input_port_name)

                    if output_port and input_port:
                        output_port.connect_to(input_port)
                        print(f"  ✓ Successfully connected {parent_link}.{output_port_name} to {child_link}.{input_port_name}")
                        
                        # NOTE: body_angleは同期しない（MJCFのref専用）
                        # point['angle']（origin_rpy）とbody_angle（ref）は別々の回転として3Dビューで適用される
                        # 同期すると二重回転が発生する
                    else:
                        if not output_port:
                            print(f"  ✗ ERROR: Output port '{output_port_name}' not found on {parent_link}")
                        if not input_port:
                            print(f"  ✗ ERROR: Input port '{input_port_name}' not found on {child_link}")
                except Exception as e:
                    print(f"  ✗ ERROR: Exception connecting {parent_link} to {child_link}: {str(e)}")
                    traceback.print_exc()

            # decorationノードを親リンクに接続
            print("\n=== Connecting Decoration Nodes ===")
            for node_name, node in nodes.items():
                # _parent_link_name属性があればdecorationノード
                if hasattr(node, '_parent_link_name'):
                    parent_link = node._parent_link_name
                    
                    # base_link_subがある場合、base_linkに接続予定のdecorationをbase_link_subに接続するように変更
                    actual_parent_link = parent_link
                    if base_link_sub_node and parent_link == 'base_link':
                        actual_parent_link = 'base_link_sub'
                        parent_node = base_link_sub_node
                    elif parent_link not in nodes:
                        print(f"Skipping decoration connection: {node_name} -> {parent_link} (parent not found)")
                        continue
                    else:
                        parent_node = nodes[parent_link]

                    # 親ノードの現在のポートインデックスを取得（joint接続の続きから）
                    # actual_parent_linkを使用してポートインデックスを管理
                    if actual_parent_link not in parent_port_indices:
                        parent_port_indices[actual_parent_link] = 0
                    port_index = parent_port_indices[actual_parent_link]
                    parent_port_indices[actual_parent_link] += 1

                    # ポート名を取得
                    is_base_link_node = parent_node.__class__.__name__ == 'BaseLinkNode'

                    if is_base_link_node:
                        # BaseLinkNodeの場合、最初のポートは'out'、それ以降は'out_2', 'out_3', ...
                        if port_index == 0:
                            output_port_name = 'out'
                        else:
                            output_port_name = f'out_{port_index + 1}'
                    else:
                        # FooNodeの場合、'out_1', 'out_2', ... を使用
                        output_port_name = f'out_{port_index + 1}'

                    input_port_name = 'in'

                    print(f"\nConnecting decoration: {actual_parent_link} -> {node_name}")
                    print(f"  Original parent link: {parent_link}")
                    print(f"  Port index: {port_index}, Expected output port: {output_port_name}")
                    print(f"  Available output ports on {actual_parent_link}: {[p.name() for p in parent_node.output_ports()]}")

                    # ポートを接続
                    try:
                        output_port = parent_node.get_output(output_port_name)
                        
                        # 出力ポートが存在しない場合、動的に追加を試みる
                        if not output_port:
                            print(f"  ⚠ Warning: Output port '{output_port_name}' not found, attempting to add...")
                            # 必要なポート数を計算
                            current_port_count = len(parent_node.output_ports())
                            needed_ports = port_index + 1 - current_port_count
                            
                            if needed_ports > 0:
                                # 不足しているポートを追加
                                for i in range(needed_ports):
                                    if hasattr(parent_node, '_add_output'):
                                        parent_node._add_output()
                                    elif hasattr(parent_node, 'add_output'):
                                        parent_node.output_count = getattr(parent_node, 'output_count', 0) + 1
                                        new_port_name = f'out_{parent_node.output_count}'
                                        parent_node.add_output(new_port_name, color=(180, 80, 0))
                                print(f"  ✓ Added {needed_ports} output port(s) to {actual_parent_link}")
                                # 再度ポートを取得
                                output_port = parent_node.get_output(output_port_name)
                        
                        input_port = node.get_input(input_port_name)

                        if output_port and input_port:
                            output_port.connect_to(input_port)
                            print(f"  ✓ Successfully connected {actual_parent_link}.{output_port_name} to {node_name}.{input_port_name}")
                        else:
                            if not output_port:
                                print(f"  ✗ ERROR: Output port '{output_port_name}' not found on {actual_parent_link}")
                                print(f"    Available ports: {[p.name() for p in parent_node.output_ports()]}")
                                print(f"    Current port count: {len(parent_node.output_ports())}, Required index: {port_index}")
                            if not input_port:
                                print(f"  ✗ ERROR: Input port '{input_port_name}' not found on {node_name}")
                    except Exception as e:
                        print(f"  ✗ ERROR: Exception connecting {actual_parent_link} to {node_name}: {str(e)}")
                        traceback.print_exc()

            # 閉リンクジョイントノードの作成と接続
            print("\n=== Creating Closed-Loop Joint Nodes ===")
            closed_loop_nodes = {}
            for joint_data in closed_loop_joints:
                joint_name = joint_data['name']
                parent_link = joint_data['parent']
                child_link = joint_data['child']

                # 閉リンクノードの名前を決定
                cl_node_name = f"{joint_name}_CL"

                # 親リンクと子リンクがノードとして存在するか確認
                if parent_link not in nodes:
                    print(f"Skipping closed-loop joint '{joint_name}': parent link '{parent_link}' not found")
                    continue
                if child_link not in nodes:
                    print(f"Skipping closed-loop joint '{joint_name}': child link '{child_link}' not found")
                    continue

                parent_node = nodes[parent_link]
                child_node = nodes[child_link]

                # 閉リンクノードを作成（親と子の中間位置に配置）
                parent_pos = parent_node.pos()
                child_pos = child_node.pos()

                if isinstance(parent_pos, (list, tuple)):
                    parent_x, parent_y = parent_pos[0], parent_pos[1]
                else:
                    parent_x, parent_y = parent_pos.x(), parent_pos.y()

                if isinstance(child_pos, (list, tuple)):
                    child_x, child_y = child_pos[0], child_pos[1]
                else:
                    child_x, child_y = child_pos.x(), child_pos.y()

                # 中間位置を計算
                cl_node_x = (parent_x + child_x) / 2
                cl_node_y = (parent_y + child_y) / 2
                cl_node_pos = QtCore.QPointF(cl_node_x, cl_node_y)

                # 閉リンクノードを作成
                print(f"Creating closed-loop node: {cl_node_name} ({parent_link} <-> {child_link})")
                cl_node = graph.create_node(
                    'insilico.nodes.ClosedLoopJointNode',
                    name=cl_node_name,
                    pos=cl_node_pos
                )

                # メタデータを設定
                cl_node.joint_name = joint_name
                cl_node.joint_type = joint_data.get('original_type', 'ball')
                cl_node.parent_link = parent_link
                cl_node.child_link = child_link
                cl_node.origin_xyz = joint_data.get('origin_xyz', [0.0, 0.0, 0.0])
                cl_node.origin_rpy = joint_data.get('origin_rpy', [0.0, 0.0, 0.0])
                cl_node.gearbox_ratio = joint_data.get('gearbox_ratio', 1.0)
                cl_node.gearbox_reference_body = joint_data.get('gearbox_reference_body')

                # 閉リンクノードのpointsデータを初期化（位置再計算で使用される可能性があるため）
                # 閉リンクノードは出力ポートから子に接続されないため、空のpointsでOK
                if not hasattr(cl_node, 'points'):
                    cl_node.points = []

                closed_loop_nodes[cl_node_name] = cl_node

                # 親リンクの出力ポートインデックスを取得
                if parent_link not in parent_port_indices:
                    parent_port_indices[parent_link] = 0
                parent_port_index = parent_port_indices[parent_link]
                parent_port_indices[parent_link] += 1

                # 親ノードのpointsデータが不足している場合は初期化
                if not hasattr(parent_node, 'points'):
                    parent_node.points = []

                # 必要なpointsデータを追加
                while len(parent_node.points) <= parent_port_index:
                    parent_node.points.append({
                        'name': f'point_{len(parent_node.points) + 1}',
                        'type': 'fixed',
                        'xyz': [0.0, 0.0, 0.0],
                        'rpy': [0.0, 0.0, 0.0],
                        'angle': [0.0, 0.0, 0.0]
                    })

                # 親ノードのpointsに閉リンクジョイントのorigin情報を設定
                origin_rpy = joint_data.get('origin_rpy', [0.0, 0.0, 0.0])
                parent_node.points[parent_port_index]['xyz'] = joint_data.get('origin_xyz', [0.0, 0.0, 0.0])
                parent_node.points[parent_port_index]['rpy'] = [0.0, 0.0, 0.0]  # Keep rpy as zero (3D view uses angle)
                parent_node.points[parent_port_index]['name'] = joint_name
                parent_node.points[parent_port_index]['type'] = joint_data.get('original_type', 'ball')
                # Set angle for 3D view display and UI editing (radians)
                parent_node.points[parent_port_index]['angle'] = list(origin_rpy)

                # 親リンクのポート名を取得
                is_base_link_node = parent_node.__class__.__name__ == 'BaseLinkNode'
                if is_base_link_node:
                    if parent_port_index == 0:
                        parent_output_port_name = 'out'
                    else:
                        parent_output_port_name = f'out_{parent_port_index + 1}'
                else:
                    parent_output_port_name = f'out_{parent_port_index + 1}'

                # 親リンク → 閉リンクノードの接続
                try:
                    # デバッグ：親ノードの全出力ポートを確認
                    print(f"  DEBUG: Parent node '{parent_link}' has {len(parent_node.points)} points")
                    print(f"  DEBUG: Using port index {parent_port_index} for closed-loop joint")
                    print(f"  DEBUG: Available output ports on {parent_link}: {[p.name() for p in parent_node.output_ports()]}")

                    parent_output_port = parent_node.get_output(parent_output_port_name)
                    cl_input_port = cl_node.get_input('in')

                    if parent_output_port and cl_input_port:
                        # 閉リンク接続では位置再計算をスキップ（ツリー構造に影響しないため）
                        # 一時的に recalculate_all_positions を無効化
                        old_recalc = graph.recalculate_all_positions
                        graph.recalculate_all_positions = lambda: None

                        parent_output_port.connect_to(cl_input_port)

                        # 元に戻す
                        graph.recalculate_all_positions = old_recalc

                        print(f"  ✓ Connected {parent_link}.{parent_output_port_name} -> {cl_node_name}.in")
                    else:
                        if not parent_output_port:
                            print(f"  ✗ ERROR: Parent output port '{parent_output_port_name}' not found on {parent_link}")
                        if not cl_input_port:
                            print(f"  ✗ ERROR: Input port 'in' not found on {cl_node_name}")
                except Exception as e:
                    print(f"  ✗ ERROR: Exception connecting {parent_link} -> {cl_node_name}: {str(e)}")
                    traceback.print_exc()

                # 閉リンクノード → 子リンクの接続
                # 閉リンクノードはツリー構造に含まれないため、出力ポートから子への接続は行わない
                # 閉リンク制約は親→閉リンクノードの接続と、メタデータで表現される
                print(f"  ℹ Closed-loop constraint: {parent_link} <-> {child_link} (via {cl_node_name})")
                print(f"            No direct connection from {cl_node_name} to {child_link} (tree structure preserved)")

                # デバッグ：閉リンクノードの接続状態を確認
                print(f"  DEBUG: Verifying closed-loop node connections for {cl_node_name}:")
                for port in cl_node.output_ports():
                    connected = port.connected_ports()
                    if connected:
                        print(f"    ⚠ WARNING: Output port {port.name()} is connected to: {[f'{p.node().name()}.{p.name()}' for p in connected]}")
                    else:
                        print(f"    ✓ Output port {port.name()} has no connections (correct)")

            print(f"Created {len(closed_loop_nodes)} closed-loop joint node(s)")

            # 閉リンク接続に水色を適用（少し遅延させてからパイプにアクセス）
            if closed_loop_nodes:
                print("\n=== Applying Cyan Color to Closed-Loop Connections ===")
                QtCore.QTimer.singleShot(200, graph.apply_cyan_to_closed_loop_connections)

            print("=" * 40 + "\n")

            # STLファイルの読み込み状況を確認
            stl_loaded_count = 0
            stl_missing_count = 0
            for link_name, link_data in links_data.items():
                if link_data['stl_file']:
                    stl_loaded_count += 1
                elif link_name != 'base_link':  # base_linkはSTLなしでも問題ない
                    stl_missing_count += 1

            # 全てのノードのメッシュファイルを3Dビューに自動読み込み
            print("\n=== Loading mesh files to 3D viewer ===")
            stl_viewer_loaded_count = 0
            if graph.stl_viewer:
                for link_name, node in nodes.items():
                    # base_linkでblank_linkがTrueの場合はスキップ
                    if link_name == 'base_link':
                        if not hasattr(node, 'blank_link') or node.blank_link:
                            print(f"Skipping base_link (blank_link=True)")
                            continue

                    if hasattr(node, 'stl_file') and node.stl_file:
                        try:
                            print(f"Loading mesh to viewer for {link_name}...")
                            graph.stl_viewer.load_stl_for_node(node)
                            stl_viewer_loaded_count += 1
                        except Exception as e:
                            print(f"Error loading mesh to viewer for {link_name}: {str(e)}")
                            traceback.print_exc()

                print(f"Loaded {stl_viewer_loaded_count} mesh files to 3D viewer")
                print("=" * 40 + "\n")
            else:
                print("Warning: Mesh viewer not available")

            # URDFにbase_linkがない場合、ルートリンクとbase_linkの接続を再確認
            # （Recalc Positionsの直前に実行）
            if 'base_link' not in links_data and 'BaseLink' not in links_data:
                print("\n=== Verifying root link connections to base_link ===")

                # ルートリンクを再検出（親リンクがないリンクを検出）
                child_links_check = set()
                for joint_data in joints_data:
                    if joint_data['child']:
                        child_links_check.add(joint_data['child'])

                root_links_check = []
                for link_name in links_data.keys():
                    if link_name not in child_links_check:
                        root_links_check.append(link_name)

                print(f"  Found {len(root_links_check)} root link(s): {root_links_check}")

                # base_linkノードを取得（必ず存在するはず）
                if 'base_link' in nodes:
                    base_link_node = nodes['base_link']

                    for root_link_name in root_links_check:
                        if root_link_name in nodes:
                            root_node = nodes[root_link_name]

                            # 既に接続されているか確認
                            already_connected = False
                            for output_port in base_link_node.output_ports():
                                if output_port.connected_ports():
                                    for connected_port in output_port.connected_ports():
                                        if connected_port.node() == root_node:
                                            already_connected = True
                                            print(f"  ✓ base_link already connected to {root_link_name}")
                                            break
                                if already_connected:
                                    break

                            if not already_connected:
                                print(f"  Connecting base_link to root link: {root_link_name}")

                                # 最初の利用可能な出力ポートを見つける
                                output_port = None
                                for port in base_link_node.output_ports():
                                    if not port.connected_ports():
                                        output_port = port
                                        break

                                if not output_port and len(base_link_node.output_ports()) > 0:
                                    output_port = base_link_node.output_ports()[0]

                                input_port = root_node.get_input('in')

                                if output_port and input_port:
                                    try:
                                        output_port.connect_to(input_port)
                                        print(f"  ✓ Connected base_link.{output_port.name()} to {root_link_name}.in")
                                    except Exception as e:
                                        print(f"  ✗ Failed to connect: {str(e)}")
                                        traceback.print_exc()
                                else:
                                    print(f"  ✗ Could not find ports for connection")
                                    if not output_port:
                                        print(f"    - base_link output port not found")
                                    if not input_port:
                                        print(f"    - {root_link_name} input port not found")
                        else:
                            print(f"  ✗ Root link '{root_link_name}' node not found in nodes dictionary")
                else:
                    print("  ✗ base_link node not found in nodes dictionary")

                print("=" * 40 + "\n")

            # 全ノードの位置を再計算（Recalc Positionsと同じ処理）
            print("\n=== Recalculating all node positions ===")
            try:
                graph.recalculate_all_positions()
                print("Position recalculation completed successfully")
            except Exception as e:
                print(f"Warning: Failed to recalculate positions: {str(e)}")
                traceback.print_exc()
            print("=" * 40 + "\n")

            # 3DビューをFrontビューに設定

            try:
                if graph.stl_viewer:
                    graph.stl_viewer.reset_camera_front()
                    print("Camera set to Front view successfully")
                    
                    # STL読み込み完了後、すべてのノードのカラーを3Dビューに適用
                    print("\nApplying colors to 3D view after URDF import...")
                    graph._apply_colors_to_all_nodes()
                else:
                    print("Warning: STL viewer not available")
            except Exception as e:
                print(f"Warning: Failed to set camera view: {str(e)}")
                traceback.print_exc()
            print("=" * 40 + "\n")

            # 完了メッセージ
            import_summary = f"URDF file has been imported:\n{urdf_file}\n\n"
            import_summary += f"Robot name: {robot_name}\n"
            import_summary += f"Links imported: {len(links_data)}\n"
            import_summary += f"Joints imported: {len(joints_data)}\n"
            import_summary += f"Nodes created: {len(nodes)}\n"
            import_summary += f"Mesh files found: {stl_loaded_count}\n"
            import_summary += f"Mesh files loaded to 3D viewer: {stl_viewer_loaded_count}\n"
            if stl_missing_count > 0:
                import_summary += f"⚠ Warning: {stl_missing_count} mesh file(s) could not be found\n"

            # すべてのノードの色を接続状態に応じて更新
            graph.update_all_node_colors()

            QtWidgets.QMessageBox.information(
                graph.widget,
                "Import Complete",
                import_summary
            )

            print("URDF import completed successfully")
            return True

        except Exception as e:
            error_msg = f"Error importing URDF: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            QtWidgets.QMessageBox.critical(
                graph.widget,
                "Import Error",
                error_msg
            )
            return False



def import_mjcf(graph):
        """MJCFファイルをインポート"""
        try:
            # MJCFファイルまたはZIPファイルを選択するダイアログ
            mjcf_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                graph.widget,
                "Select MJCF file or ZIP archive to import",
                os.getcwd(),
                "MJCF Files (*.xml *.zip);;XML Files (*.xml);;ZIP Files (*.zip);;All Files (*)"
            )

            if not mjcf_file:
                print("MJCF import cancelled")
                return False

            print(f"Importing MJCF from: {mjcf_file}")

            # ZIPファイルの場合、展開する
            working_dir = None
            xml_file_to_load = None

            if mjcf_file.endswith('.zip'):
                import zipfile
                import tempfile

                print("Detected ZIP file, extracting...")

                # 一時ディレクトリに展開
                temp_dir = tempfile.mkdtemp(prefix='mjcf_import_')
                working_dir = temp_dir

                try:
                    with zipfile.ZipFile(mjcf_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    print(f"Extracted to: {temp_dir}")

                    # 展開されたディレクトリ内のXMLファイルを検索
                    xml_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.xml'):
                                xml_files.append(os.path.join(root, file))

                    if not xml_files:
                        QtWidgets.QMessageBox.warning(
                            graph.widget,
                            "No XML Files Found",
                            "No XML files found in the ZIP archive."
                        )
                        return False

                    # XMLファイルが複数ある場合、選択させる
                    if len(xml_files) > 1:
                        # ファイル名のリストを作成
                        file_names = [os.path.relpath(f, temp_dir) for f in xml_files]

                        # 選択ダイアログを表示
                        selected_file, ok = QtWidgets.QInputDialog.getItem(
                            graph.widget,
                            "Select XML File",
                            "Multiple XML files found. Please select one:",
                            file_names,
                            0,
                            False
                        )

                        if ok and selected_file:
                            xml_file_to_load = os.path.join(temp_dir, selected_file)
                        else:
                            print("XML file selection cancelled")
                            return False
                    else:
                        xml_file_to_load = xml_files[0]

                    print(f"Selected XML file: {xml_file_to_load}")
                    mjcf_file = xml_file_to_load

                except Exception as e:
                    print(f"Error extracting ZIP file: {str(e)}")
                    traceback.print_exc()
                    QtWidgets.QMessageBox.critical(
                        graph.widget,
                        "ZIP Extraction Error",
                        f"Failed to extract ZIP file:\n\n{str(e)}"
                    )
                    return False
            else:
                working_dir = os.path.dirname(mjcf_file)

            # MJCFParserを使用してパース
            try:
                mjcf_parser = MJCFParser(verbose=True)
                mjcf_data = mjcf_parser.parse_mjcf(mjcf_file, working_dir=working_dir)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(
                    graph.widget,
                    "Invalid MJCF File",
                    f"Selected file is not a valid MJCF file:\n{str(e)}"
                )
                return False
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    graph.widget,
                    "MJCF Parse Error",
                    f"Failed to parse MJCF file:\n{str(e)}"
                )
                return False

            # パースされたデータを取得
            robot_name = mjcf_data['robot_name']
            bodies_data_list = mjcf_data['bodies']  # リストとして取得
            joints_data = mjcf_data['joints']
            meshes_data = mjcf_data['meshes']
            eulerseq = mjcf_data['eulerseq']
            default_classes = mjcf_data['default_classes']
            closed_loop_joints = mjcf_data.get('closed_loop_joints', [])  # 閉リンクジョイント情報を取得

            # bodies_dataをリストから辞書に変換（nameをキーとして）
            bodies_data = {}
            for body_data in bodies_data_list:
                body_name = body_data.get('name')
                if body_name:
                    bodies_data[body_name] = body_data
                else:
                    # nameがない場合はスキップまたは警告
                    print(f"Warning: Body data without name found: {body_data}")

            graph.robot_name = robot_name
            print(f"Robot name set to: {robot_name}")

            # Graphにeulerseqを保存（Export時に使用）
            graph.mjcf_eulerseq = eulerseq

            # ノードを作成
            # ノードを作成
            nodes = {}

            # 既存のbase_linkノードを取得
            base_node = None
            for node in graph.all_nodes():
                if node.__class__.__name__ == 'BaseLinkNode':
                    base_node = node
                    nodes['base_link'] = base_node
                    break

            # base_linkが見つからない場合は作成
            if not base_node:
                print("No base_link found, creating new one")
                base_node = graph.create_node(
                    'insilico.nodes.BaseLinkNode',
                    name='base_link',
                    pos=QtCore.QPointF(50, 50)  # 中心近くに配置
                )
                nodes['base_link'] = base_node

            # MJCF用のbase_link_subノードを作成（URDFと同様の中間ノード）
            base_link_sub_node = None
            try:
                base_link_sub_node = graph.get_node_by_name('base_link_sub')
            except Exception:
                base_link_sub_node = None
            if not base_link_sub_node:
                base_link_sub_node = graph.create_node(
                    'insilico.nodes.FooNode',
                    name='base_link_sub',
                    pos=QtCore.QPointF(200, 50)
                )
                # base_link_subは固定ノード
                base_link_sub_node.rotation_axis = 3
                base_link_sub_node.body_angle = [0.0, 0.0, 0.0]
            nodes['base_link_sub'] = base_link_sub_node

            # base_link -> base_link_sub を固定で接続
            try:
                base_output_port = base_node.output_ports()[0]
                base_link_sub_input_port = base_link_sub_node.input_ports()[0]
                base_output_port.connect_to(base_link_sub_input_port)
                print("    ✓ Connected base_link -> base_link_sub")
            except Exception:
                pass

            # MJCFのルートbody（freejoint）をbase_link_subに吸収
            base_link_sub_additional_visuals = []
            mjcf_root_body_name = None
            root_bodies = [b for b in bodies_data_list if not b.get('parent')]
            print(f"\n[DEBUG] Searching for freejoint root body")
            print(f"[DEBUG] Found {len(root_bodies)} root bodies (no parent): {[b.get('name') for b in root_bodies]}")
            for root_body in root_bodies:
                if root_body.get('has_freejoint'):
                    mjcf_root_body_name = root_body.get('name')
                    print(f"[DEBUG] Found freejoint root: {mjcf_root_body_name}")
                    break
            if mjcf_root_body_name and mjcf_root_body_name != 'base_link':
                root_body_data = bodies_data.get(mjcf_root_body_name)
                if root_body_data:
                    print(f"\n  Using root body '{mjcf_root_body_name}' as base_link_sub (freejoint)")
                    # Count children of root body BEFORE absorption
                    root_children = [b for b in bodies_data_list if b.get('parent') == mjcf_root_body_name]
                    print(f"  [DEBUG] Root body '{mjcf_root_body_name}' has {len(root_children)} direct children: {[b.get('name') for b in root_children]}")
                    # base bodyのz座標を保存（MJCF/URDFエクスポート時に使用）
                    if 'pos' in root_body_data and len(root_body_data['pos']) >= 3:
                        graph.base_link_height = root_body_data['pos'][2]
                        print(f"    ✓ Saved base_link_height: {graph.base_link_height} (from MJCF root body pos)")
                    # base_link_subにパラメータを設定
                    base_link_sub_node.mass_value = root_body_data['mass']
                    base_link_sub_node.inertia = root_body_data['inertia']
                    base_link_sub_node.inertial_origin = root_body_data['inertial_origin']
                    base_link_sub_node.node_color = root_body_data['color']
                    base_link_sub_node.stl_file = root_body_data.get('stl_file')
                    base_link_sub_node.mesh_scale = root_body_data.get('mesh_scale', [1.0, 1.0, 1.0])
                    base_link_sub_node.visual_origin = root_body_data.get('visual_origin', {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]})
                    if root_body_data.get('collision_mesh'):
                        base_link_sub_node.collider_mesh = root_body_data['collision_mesh']
                        if root_body_data.get('collider_enabled'):
                            base_link_sub_node.collider_type = root_body_data.get('collider_type', 'mesh')
                            base_link_sub_node.collider_enabled = True
                    if root_body_data.get('collider_data'):
                        base_link_sub_node.collider_data = root_body_data['collider_data']
                        base_link_sub_node.collider_type = 'primitive'
                        base_link_sub_node.collider_enabled = True
                    # 追加visual
                    base_link_sub_additional_visuals = root_body_data.get('visuals', [])[1:]

                    # root bodyを削除（子をbase_link_subへ付け替え）
                    bodies_data.pop(mjcf_root_body_name, None)
                    bodies_data_list = [b for b in bodies_data_list if b.get('name') != mjcf_root_body_name]
                    updated_children_count = 0
                    for body_data in bodies_data_list:
                        if body_data.get('parent') == mjcf_root_body_name:
                            body_data['parent'] = 'base_link_sub'
                            updated_children_count += 1
                            print(f"    ✓ Updated body '{body_data.get('name')}' parent: {mjcf_root_body_name} -> base_link_sub")
                    print(f"  [DEBUG] Updated {updated_children_count} children to point to base_link_sub")
                    
                    # jointの親参照を更新、rootボディへのジョイントは削除
                    new_joints = []
                    updated_joints_count = 0
                    for joint_data in joints_data:
                        if joint_data.get('child') == mjcf_root_body_name:
                            print(f"    ✓ Skipping joint to root body: {joint_data.get('name')}")
                            continue
                        if joint_data.get('parent') == mjcf_root_body_name:
                            joint_data['parent'] = 'base_link_sub'
                            updated_joints_count += 1
                            print(f"    ✓ Updated joint '{joint_data.get('name')}' parent: {mjcf_root_body_name} -> base_link_sub")
                        new_joints.append(joint_data)
                    joints_data = new_joints
                    print(f"  [DEBUG] Updated {updated_joints_count} joints to point to base_link_sub")
                    print(f"  [DEBUG] After absorption: bodies_data has {len(bodies_data)} bodies, bodies_data_list has {len(bodies_data_list)} bodies")

            # MJCFのbase_linkデータがある場合、base_link_mjcfノードを作成
            base_link_mjcf_node = None
            base_link_mjcf_additional_visuals = []
            if 'base_link' in bodies_data:
                base_data = bodies_data['base_link']
                print(f"\n  Creating base_link_mjcf node with MJCF data:")
                print(f"    mass: {base_data['mass']}")
                print(f"    inertia: {base_data['inertia']}")
                print(f"    mesh: {base_data.get('stl_file')}")
                
                # base bodyのz座標を保存（MJCF/URDFエクスポート時に使用）
                if 'pos' in base_data and len(base_data['pos']) >= 3:
                    graph.base_link_height = base_data['pos'][2]  # z座標
                    print(f"    ✓ Saved base_link_height: {graph.base_link_height} (from MJCF base body pos)")
                else:
                    # デフォルト値を設定
                    graph.base_link_height = 0.5
                    print(f"    ✓ Using default base_link_height: {graph.base_link_height}")

                # base_link_mjcfノードを作成（base_linkの近くに配置）
                if hasattr(base_node, 'pos') and callable(base_node.pos):
                    pos = base_node.pos()
                    if hasattr(pos, 'x') and callable(pos.x):
                        base_x = pos.x()
                        base_y = pos.y()
                    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        base_x = pos[0]
                        base_y = pos[1]
                    else:
                        base_x = 50
                        base_y = 50
                else:
                    base_x = 50
                    base_y = 50
                base_link_mjcf_node = graph.create_node(
                    'insilico.nodes.FooNode',
                    name='base_link_mjcf',
                    pos=QtCore.QPointF(base_x + 150, base_y)  # base_linkから右にオフセット
                )
                nodes['base_link_mjcf'] = base_link_mjcf_node

                # パラメータを設定
                base_link_mjcf_node.mass_value = base_data['mass']
                base_link_mjcf_node.inertia = base_data['inertia']
                base_link_mjcf_node.inertial_origin = base_data['inertial_origin']
                base_link_mjcf_node.node_color = base_data['color']

                # Body Angle（初期回転角度）の設定（ラジアンで保持）
                # MJCFインポートの場合、base bodyの姿勢は通常[0,0,0]なので、body_angleは初期化のみ
                # （bodyの姿勢の二重適用を防ぐため）
                base_link_mjcf_node.body_angle = [0.0, 0.0, 0.0]
                if 'rpy' in base_data and any(a != 0.0 for a in base_data['rpy']):
                    print(f"    ℹ Base body orientation from MJCF: {[math.degrees(a) for a in base_data['rpy']]} degrees (not applied to body_angle)")

                # base_link_mjcfはbase_linkに固定されているので、rotation_axisを3（Fixed）に設定
                # これにより、MJCF出力時にジョイントが出力されなくなる
                base_link_mjcf_node.rotation_axis = 3
                print(f"    ✓ Set base_link_mjcf rotation_axis to 3 (Fixed) - no joint will be exported")

                # メッシュファイル
                if base_data.get('stl_file'):
                    base_link_mjcf_node.stl_file = base_data['stl_file']
                    print(f"    ✓ Assigned mesh to base_link_mjcf: {base_data['stl_file']}")

                # メッシュスケールとvisual origin
                base_link_mjcf_node.mesh_scale = base_data.get('mesh_scale', [1.0, 1.0, 1.0])
                base_link_mjcf_node.visual_origin = base_data.get('visual_origin', {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]})

                # Collision mesh
                if base_data.get('collision_mesh'):
                    base_link_mjcf_node.collider_mesh = base_data['collision_mesh']
                    print(f"    ✓ Assigned collision mesh to base_link_mjcf: {base_data['collision_mesh']}")

                    # コライダーが明示的に指定されている場合、有効化
                    if base_data.get('collider_enabled'):
                        base_link_mjcf_node.collider_type = base_data.get('collider_type', 'mesh')
                        base_link_mjcf_node.collider_enabled = True
                        print(f"    ✓ Enabled collider for base_link_mjcf")

                # Primitive collider
                if base_data.get('collider_data'):
                    base_link_mjcf_node.collider_data = base_data['collider_data']
                    base_link_mjcf_node.collider_type = 'primitive'
                    base_link_mjcf_node.collider_enabled = True
                    print(f"    ✓ Assigned primitive collider to base_link_mjcf: {base_data['collider_data']['type']}")

                # base_link_subからbase_link_mjcfへ接続
                try:
                    base_link_sub_output_port = base_link_sub_node.output_ports()[0]
                    base_link_mjcf_input_port = base_link_mjcf_node.input_ports()[0]
                    base_link_sub_output_port.connect_to(base_link_mjcf_input_port)
                    print("    ✓ Connected base_link_sub -> base_link_mjcf")
                except Exception:
                    pass

                # base_link_mjcfの追加visual情報を保存（後で処理するため）
                base_link_mjcf_additional_visuals = base_data.get('visuals', [])[1:]  # 2つ目以降のvisual

                # bodies_dataから'base_link'を削除（後続の処理で重複作成されないように）
                del bodies_data['base_link']

                # joints_dataの親参照を'base_link'から'base_link_mjcf'に更新
                for joint_data in joints_data:
                    if joint_data['parent'] == 'base_link':
                        joint_data['parent'] = 'base_link_mjcf'
                        print(f"    ✓ Updated joint '{joint_data['name']}' parent: base_link -> base_link_mjcf")
                # bodies_data_listの親参照も更新（接続順の安定化）
                for body_data in bodies_data_list:
                    if body_data.get('parent') == 'base_link':
                        body_data['parent'] = 'base_link_mjcf'
            else:
                # base bodyがない場合、デフォルトの高さを設定
                if not hasattr(graph, 'base_link_height'):
                    graph.base_link_height = 0.5
                    print(f"  ✓ Using default base_link_height: {graph.base_link_height} (no base body in MJCF)")

            # base_link_mjcfがない場合、base_link_subに親参照を寄せる
            if base_link_mjcf_node is None:
                for joint_data in joints_data:
                    if joint_data['parent'] == 'base_link':
                        joint_data['parent'] = 'base_link_sub'
                        print(f"    ✓ Updated joint '{joint_data['name']}' parent: base_link -> base_link_sub")
                for body_data in bodies_data_list:
                    if body_data.get('parent') == 'base_link':
                        body_data['parent'] = 'base_link_sub'

            # 各ボディの子ジョイント数をカウント
            body_child_counts = {}
            for body_name in bodies_data.keys():
                body_child_counts[body_name] = 0

            # base_link_mjcfが作成されている場合は追加
            if base_link_mjcf_node is not None:
                body_child_counts['base_link_mjcf'] = 0
            # base_link_subは常にカウント対象
            body_child_counts['base_link_sub'] = 0

            for joint_data in joints_data:
                parent = joint_data['parent']
                if parent in body_child_counts:
                    body_child_counts[parent] += 1

            # 親→子の順序マップ（XML順に基づく）
            child_order_by_parent = {}
            default_root_parent = 'base_link_sub' if base_link_sub_node is not None else 'base_link'
            print(f"\n[DEBUG] Creating child_order_by_parent map")
            print(f"[DEBUG] default_root_parent: {default_root_parent}")
            print(f"[DEBUG] bodies_data_list length: {len(bodies_data_list)}")
            print(f"[DEBUG] bodies_data keys: {list(bodies_data.keys())}")
            for body_data in bodies_data_list:
                child_name = body_data.get('name')
                if not child_name or child_name not in bodies_data:
                    print(f"[DEBUG] Skipping body: child_name={child_name}, in_bodies_data={child_name in bodies_data if child_name else False}")
                    continue
                parent_name = body_data.get('parent') or default_root_parent
                print(f"[DEBUG] Processing body: {child_name}, parent: {parent_name}")
                if parent_name not in child_order_by_parent:
                    child_order_by_parent[parent_name] = []
                if child_name not in child_order_by_parent[parent_name]:
                    child_order_by_parent[parent_name].append(child_name)
            
            print(f"\n[DEBUG] child_order_by_parent created:")
            for parent, children in child_order_by_parent.items():
                print(f"  {parent}: {children}")

            # base_link_mjcfに子ジョイント数に応じた出力ポートを追加
            if base_link_mjcf_node is not None:
                child_count = body_child_counts.get('base_link_mjcf', 0)
                print(f"    base_link_mjcf has {child_count} children")
                if child_count > 1:
                    for i in range(1, child_count):
                        base_link_mjcf_node._add_output()
                        print(f"    ✓ Added output port {i+1} to base_link_mjcf")

                # base_link_mjcfの追加visual処理（base_1.obj以降のメッシュ用）
                if base_link_mjcf_additional_visuals:
                    print(f"  base_link_mjcf has {len(base_link_mjcf_additional_visuals)} additional visual(s)")

                    # 追加のvisualの数だけ出力ポートを追加
                    for i in range(len(base_link_mjcf_additional_visuals)):
                        base_link_mjcf_node._add_output()

                    # 追加のvisual用の子ノードを作成して接続
                    if hasattr(base_link_mjcf_node, 'pos') and callable(base_link_mjcf_node.pos):
                        mjcf_pos = base_link_mjcf_node.pos()
                        if hasattr(mjcf_pos, 'x') and callable(mjcf_pos.x):
                            base_mjcf_pos_x = mjcf_pos.x()
                            base_mjcf_pos_y = mjcf_pos.y()
                        elif isinstance(mjcf_pos, (list, tuple)) and len(mjcf_pos) >= 2:
                            base_mjcf_pos_x = mjcf_pos[0]
                            base_mjcf_pos_y = mjcf_pos[1]
                        else:
                            base_mjcf_pos_x = 200
                            base_mjcf_pos_y = 50
                    else:
                        base_mjcf_pos_x = 200
                        base_mjcf_pos_y = 50

                    for visual_idx, visual_data in enumerate(base_link_mjcf_additional_visuals, start=1):
                        visual_node_name = f"base_link_mjcf_visual_{visual_idx}"

                        # 子ノードの位置を計算（親ノードの近くに配置）
                        visual_pos_x = base_mjcf_pos_x + 50 + visual_idx * 30
                        visual_pos_y = base_mjcf_pos_y + 100

                        visual_node = graph.create_node(
                            'insilico.nodes.FooNode',
                            name=visual_node_name,
                            pos=QtCore.QPointF(visual_pos_x, visual_pos_y)
                        )
                        nodes[visual_node_name] = visual_node

                        # ビジュアルノードにメッシュを設定
                        visual_node.stl_file = visual_data['mesh']
                        visual_node.node_color = visual_data['color']
                        visual_node.mass_value = 0.0  # ビジュアルのみなので質量0

                        print(f"      ✓ Created visual node '{visual_node_name}' with mesh: {os.path.basename(visual_data['mesh'])}")

                        # visualのクォータニオンをRPYに変換
                        visual_rpy = ConversionUtils.quat_to_rpy(visual_data['quat'])

                        # base_link_mjcfのポイント情報を設定
                        if not hasattr(base_link_mjcf_node, 'points'):
                            base_link_mjcf_node.points = []

                        # visual用のポイントを追加（child_count + visual_idxの位置）
                        point_index = child_count + visual_idx - 1
                        while len(base_link_mjcf_node.points) <= point_index:
                            base_link_mjcf_node.points.append({
                                'name': f'point_{len(base_link_mjcf_node.points)}',
                                'type': 'fixed',
                                'xyz': [0.0, 0.0, 0.0],
                                'rpy': [0.0, 0.0, 0.0]
                            })

                        base_link_mjcf_node.points[point_index] = {
                            'name': f'visual_{visual_idx}_attachment',
                            'type': 'fixed',
                            'xyz': visual_data['pos'],  # geomのローカル位置
                            'rpy': [0.0, 0.0, 0.0],  # Keep rpy as zero (3D view uses angle)
                            'angle': list(visual_rpy)  # Set angle for 3D view display (radians)
                        }
                        print(f"      Visual pos: {visual_data['pos']}, quat: {visual_data['quat']} -> rpy: {visual_rpy}")

                        # ポートを接続
                        # 出力ポート番号は、子ジョイント数 + visual_idx
                        output_port_index = child_count + visual_idx - 1
                        output_port_name = f'out_{output_port_index + 1}'
                        input_port_name = 'in'

                        print(f"    Connecting visual node: base_link_mjcf -> {visual_node_name}")
                        print(f"      Port: {output_port_name} -> {input_port_name} (port index: {output_port_index})")

                        # ポート接続を実行
                        output_ports = base_link_mjcf_node.output_ports()
                        if output_port_index < len(output_ports):
                            output_port = output_ports[output_port_index]
                            input_port = visual_node.input_ports()[0]
                            output_port.connect_to(input_port)
                            print(f"      ✓ Connected")
                        else:
                            print(f"      ✗ Output port not found: {output_port_name} (have {len(output_ports)} ports)")

            # base_link_subに必要な出力ポートを追加（子ボディ + 追加visual用）
            base_link_sub_child_count = body_child_counts.get('base_link_sub', 0)
            print(f"\n[DEBUG] base_link_sub port configuration:")
            print(f"  Child count from body_child_counts: {base_link_sub_child_count}")
            print(f"  base_link_sub needs {base_link_sub_child_count} output ports for child bodies")
            
            # 現在の出力ポート数を取得
            current_base_link_sub_ports = len(base_link_sub_node.output_ports())
            print(f"  Current output ports: {current_base_link_sub_ports}")
            needed_ports_for_children = base_link_sub_child_count - current_base_link_sub_ports
            print(f"  Needed ports: {needed_ports_for_children}")
            if needed_ports_for_children > 0:
                for _ in range(needed_ports_for_children):
                    base_link_sub_node._add_output()
                print(f"  ✓ Added {needed_ports_for_children} output port(s) to base_link_sub for children")
            print(f"  Total output ports after addition: {len(base_link_sub_node.output_ports())}")

            # base_link_subの追加visual処理（freejointルートを吸収した場合）
            if base_link_sub_additional_visuals:
                child_count = body_child_counts.get('base_link_sub', 0)
                print(f"  base_link_sub has {len(base_link_sub_additional_visuals)} additional visual(s)")
                for _ in range(len(base_link_sub_additional_visuals)):
                    base_link_sub_node._add_output()

                if hasattr(base_link_sub_node, 'pos') and callable(base_link_sub_node.pos):
                    mjcf_pos = base_link_sub_node.pos()
                    if hasattr(mjcf_pos, 'x') and callable(mjcf_pos.x):
                        base_mjcf_pos_x = mjcf_pos.x()
                        base_mjcf_pos_y = mjcf_pos.y()
                    elif isinstance(mjcf_pos, (list, tuple)) and len(mjcf_pos) >= 2:
                        base_mjcf_pos_x = mjcf_pos[0]
                        base_mjcf_pos_y = mjcf_pos[1]
                    else:
                        base_mjcf_pos_x = 200
                        base_mjcf_pos_y = 50
                else:
                    base_mjcf_pos_x = 200
                    base_mjcf_pos_y = 50

                for visual_idx, visual_data in enumerate(base_link_sub_additional_visuals, start=1):
                    visual_node_name = f"base_link_sub_visual_{visual_idx}"
                    visual_pos_x = base_mjcf_pos_x + 50 + visual_idx * 30
                    visual_pos_y = base_mjcf_pos_y + 100

                    visual_node = graph.create_node(
                        'insilico.nodes.FooNode',
                        name=visual_node_name,
                        pos=QtCore.QPointF(visual_pos_x, visual_pos_y)
                    )
                    nodes[visual_node_name] = visual_node
                    visual_node.stl_file = visual_data['mesh']
                    visual_node.node_color = visual_data['color']
                    visual_node.mass_value = 0.0

                    visual_rpy = ConversionUtils.quat_to_rpy(visual_data['quat'])
                    if not hasattr(base_link_sub_node, 'points'):
                        base_link_sub_node.points = []

                    point_index = child_count + visual_idx - 1
                    while len(base_link_sub_node.points) <= point_index:
                        base_link_sub_node.points.append({
                            'name': f'point_{len(base_link_sub_node.points)}',
                            'type': 'fixed',
                            'xyz': [0.0, 0.0, 0.0],
                            'rpy': [0.0, 0.0, 0.0]
                        })

                    base_link_sub_node.points[point_index] = {
                        'name': f'visual_{visual_idx}_attachment',
                        'type': 'fixed',
                        'xyz': visual_data['pos'],
                        'rpy': [0.0, 0.0, 0.0],  # Keep rpy as zero (3D view uses angle)
                        'angle': list(visual_rpy)  # Set angle for 3D view display (radians)
                    }

                    output_port_index = child_count + visual_idx - 1
                    output_port_name = f'out_{output_port_index + 1}'
                    input_port_name = 'in'
                    output_ports = base_link_sub_node.output_ports()
                    if output_port_index < len(output_ports):
                        output_port = output_ports[output_port_index]
                        input_port = visual_node.input_ports()[0]
                        output_port.connect_to(input_port)
                        print(f"      ✓ Connected base_link_sub -> {visual_node_name}")
                    else:
                        print(f"      ✗ Output port not found on base_link_sub: {output_port_name} (have {len(output_ports)} ports)")

            # 他のボディのノードを作成
            grid_spacing = 200
            # base_linkノードの位置を基準に子ノードを配置
            if hasattr(base_node, 'pos') and callable(base_node.pos):
                pos = base_node.pos()
                if hasattr(pos, 'x') and callable(pos.x):
                    base_x = pos.x()
                    base_y = pos.y()
                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    base_x = pos[0]
                    base_y = pos[1]
                else:
                    base_x = 50
                    base_y = 50
            else:
                base_x = 50
                base_y = 50
            current_x = base_x + grid_spacing  # base_linkから右にオフセット
            current_y = base_y
            nodes_per_row = 4
            node_count = 0

            for body_name, body_data in bodies_data.items():
                # base_linkはスキップ（既に処理済み）
                if body_name == 'base_link':
                    continue
                
                # freejoint rootbodyはbase_link_subに吸収されたのでスキップ
                if mjcf_root_body_name and body_name == mjcf_root_body_name:
                    print(f"  ℹ Skipping node creation for '{body_name}' (freejoint root, absorbed into base_link_sub)")
                    continue

                # グリッドレイアウトで位置を計算
                row = node_count // nodes_per_row
                col = node_count % nodes_per_row
                pos_x = current_x + col * grid_spacing
                pos_y = current_y + row * grid_spacing

                node = graph.create_node(
                    'insilico.nodes.FooNode',
                    name=body_name,
                    pos=QtCore.QPointF(pos_x, pos_y)
                )
                nodes[body_name] = node

                # ノードのパラメータを設定
                node.mass_value = body_data['mass']
                node.inertia = body_data['inertia']
                node.inertial_origin = body_data['inertial_origin']
                node.node_color = body_data['color']

                # Body Angle（初期回転角度）の設定（ラジアンで保持）
                # MJCFインポートの場合、bodyの姿勢（xyaxes/euler/quatから）はparent_node.points[port_index]['rpy']に設定されるので、
                # ここではbody_angleを初期化のみ行う（後でrefがある場合のみ設定される）
                # これにより、bodyの姿勢が二重適用されるのを防ぐ
                node.body_angle = [0.0, 0.0, 0.0]
                if 'rpy' in body_data and any(a != 0.0 for a in body_data['rpy']):
                    print(f"  ℹ Body orientation from MJCF (will be set in parent's point['rpy']): {[math.degrees(a) for a in body_data['rpy']]} degrees")

                # Rotation axisの設定（joint要素のaxis属性から取得）
                if 'rotation_axis' in body_data and body_data['rotation_axis'] is not None:
                    node.rotation_axis = body_data['rotation_axis']
                    axis_names = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']
                    if 0 <= node.rotation_axis < len(axis_names):
                        print(f"  ✓ Set rotation_axis for node '{body_name}': {node.rotation_axis} ({axis_names[node.rotation_axis]})")
                    else:
                        print(f"  ✓ Set rotation_axis for node '{body_name}': {node.rotation_axis}")
                else:
                    # デフォルトはX軸（後方互換性のため）
                    node.rotation_axis = 0

                # メッシュファイルの割り当て（デバッグ出力付き）
                if body_data['stl_file']:
                    node.stl_file = body_data['stl_file']
                    print(f"  ✓ Assigned mesh to node '{body_name}': {body_data['stl_file']}")
                else:
                    print(f"  ✗ No mesh file for node '{body_name}'")

                # メッシュのスケール情報を設定
                node.mesh_scale = body_data.get('mesh_scale', [1.0, 1.0, 1.0])
                # Visual origin情報を設定
                node.visual_origin = body_data.get('visual_origin', {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]})

                # Multiple colliders support
                # Initialize colliders list
                node.colliders = []
                
                # First, check if colliders list exists in body_data
                if 'colliders' in body_data and body_data['colliders']:
                    for collider in body_data['colliders']:
                        collider_copy = collider.copy()
                        node.colliders.append(collider_copy)
                        # Debug: Print collider rotation and position
                        print(f"  [IMPORT_DEBUG] Added collider to node '{body_name}':")
                        print(f"    type: {collider_copy.get('type')}")
                        print(f"    position: {collider_copy.get('position', 'NOT SET')}")
                        print(f"    rotation: {collider_copy.get('rotation', 'NOT SET')}")
                        print(f"    mesh: {collider_copy.get('mesh', 'NOT SET')}")
                    print(f"  ✓ Assigned {len(node.colliders)} collider(s) from colliders list to node '{body_name}'")
                
                # Backward compatibility: migrate old format to colliders list
                # Only migrate if colliders list is empty
                if len(node.colliders) == 0:
                    # Collision mesh情報を設定（旧形式）
                    if body_data.get('collision_mesh'):
                        node.collider_mesh = body_data['collision_mesh']
                        print(f"  ✓ Assigned collision mesh to node '{body_name}': {os.path.basename(body_data['collision_mesh'])}")

                        # コライダーが明示的に指定されている場合、有効化
                        collider_enabled = body_data.get('collider_enabled', False)
                        if body_data.get('collider_type') == 'mesh' or body_data.get('collider_enabled'):
                            node.collider_type = body_data.get('collider_type', 'mesh')
                            node.collider_enabled = collider_enabled
                            # Add to colliders list
                            node.colliders.append({
                                'type': 'mesh',
                                'enabled': collider_enabled,
                                'data': None,
                                'mesh': body_data['collision_mesh'],
                                'mesh_scale': getattr(node, 'collider_mesh_scale', [1.0, 1.0, 1.0])
                            })
                            print(f"  ✓ Enabled mesh collider for node '{body_name}' (migrated from old format)")

                    # Primitive collider（旧形式）
                    if body_data.get('collider_data'):
                        node.collider_data = body_data['collider_data']
                        node.collider_type = 'primitive'
                        node.collider_enabled = True
                        # Add to colliders list
                        node.colliders.append({
                            'type': 'primitive',
                            'enabled': True,
                            'data': body_data['collider_data'],
                            'mesh': None,
                            'mesh_scale': [1.0, 1.0, 1.0]
                        })
                        print(f"  ✓ Assigned primitive collider to node '{body_name}': {body_data['collider_data']['type']} (migrated from old format)")
                
                # If still no colliders, check if collider_enabled is True (use visual mesh as collider)
                if len(node.colliders) == 0 and body_data.get('collider_enabled', False):
                    node.collider_enabled = True
                    node.colliders.append({
                        'type': 'mesh',
                        'enabled': True,
                        'data': None,
                        'mesh': None,  # Use visual mesh
                        'mesh_scale': [1.0, 1.0, 1.0]
                    })
                    print(f"  ✓ Enabled visual mesh as collider for node '{body_name}'")
                
                # Debug: Print collider status
                if len(node.colliders) > 0:
                    print(f"  ✓ Node '{body_name}' has {len(node.colliders)} collider(s)")
                    for i, collider in enumerate(node.colliders):
                        print(f"    Collider[{i}]: type={collider.get('type')}, enabled={collider.get('enabled')}")
                else:
                    print(f"  ✗ Node '{body_name}' has no colliders")
                    print(f"    body_data keys: {list(body_data.keys())}")
                    print(f"    body_data['colliders']: {body_data.get('colliders', 'NOT FOUND')}")
                    print(f"    body_data['collider_enabled']: {body_data.get('collider_enabled', 'NOT FOUND')}")
                    print(f"    body_data['collider_data']: {body_data.get('collider_data', 'NOT FOUND')}")
                    print(f"    body_data['collision_mesh']: {body_data.get('collision_mesh', 'NOT FOUND')}")

                # メッシュ反転判定
                # body_dataからis_mesh_reversedフラグを取得（scaleの負の値から検出済み）
                if 'is_mesh_reversed' in body_data:
                    node.is_mesh_reversed = body_data['is_mesh_reversed']
                else:
                    # フォールバック: visual_originとmesh_scaleから判定
                    node.is_mesh_reversed = is_mesh_reversed_check(
                        node.visual_origin,
                        node.mesh_scale
                    )
                if node.is_mesh_reversed:
                    print(f"  MJCF node '{body_name}' mesh is reversed/mirrored (for MJCF export)")

                # 子ジョイントの数に応じて出力ポートを追加
                child_count = body_child_counts.get(body_name, 0)
                if child_count > 1:
                    for i in range(1, child_count):
                        node._add_output()

                # 複数のvisualがある場合、追加のノードを作成
                additional_visuals = body_data.get('visuals', [])[1:]  # 2つ目以降のvisual
                if additional_visuals:
                    print(f"  Body '{body_name}' has {len(additional_visuals)} additional visual(s)")

                    # 追加のvisualの数だけ出力ポートを追加
                    for i in range(len(additional_visuals)):
                        node._add_output()

                    # 追加のvisual用の子ノードを作成して接続
                    for visual_idx, visual_data in enumerate(additional_visuals, start=1):
                        visual_node_name = f"{body_name}_visual_{visual_idx}"

                        # 子ノードの位置を計算（親ノードの近くに配置）
                        visual_pos_x = pos_x + 50 + visual_idx * 30
                        visual_pos_y = pos_y + 100

                        visual_node = graph.create_node(
                            'insilico.nodes.FooNode',
                            name=visual_node_name,
                            pos=QtCore.QPointF(visual_pos_x, visual_pos_y)
                        )
                        nodes[visual_node_name] = visual_node

                        # ビジュアルノードにメッシュを設定
                        visual_node.stl_file = visual_data['mesh']
                        visual_node.node_color = visual_data['color']
                        visual_node.mass_value = 0.0  # ビジュアルのみなので質量0

                        # visualのクォータニオンをRPYに変換
                        visual_rpy = ConversionUtils.quat_to_rpy(visual_data['quat'])

                        # 親ノードのポイント情報を設定
                        if not hasattr(node, 'points'):
                            node.points = []

                        # 子ジョイントの数 + visual_idxの位置にポイントを追加
                        point_index = child_count + visual_idx - 1
                        while len(node.points) <= point_index:
                            node.points.append({
                                'name': f'point_{len(node.points)}',
                                'type': 'fixed',
                                'xyz': [0.0, 0.0, 0.0],
                                'rpy': [0.0, 0.0, 0.0]
                            })

                        # visualの位置と姿勢を設定
                        node.points[point_index] = {
                            'name': f'visual_{visual_idx}_attachment',
                            'type': 'fixed',
                            'xyz': visual_data['pos'],  # geomのローカル位置
                            'rpy': [0.0, 0.0, 0.0],  # Keep rpy as zero (3D view uses angle)
                            'angle': list(visual_rpy)  # Set angle for 3D view display (radians)
                        }
                        print(f"      Visual pos: {visual_data['pos']}, quat: {visual_data['quat']} -> rpy: {visual_rpy}")

                        # ポートを接続
                        is_base_link = node.__class__.__name__ == 'BaseLinkNode'
                        if is_base_link:
                            output_port_name = 'out' if point_index == 0 else f'out_{point_index + 1}'
                        else:
                            output_port_name = f'out_{point_index + 1}'

                        input_port_name = 'in'

                        print(f"    Connecting visual node: {body_name} -> {visual_node_name}")
                        print(f"      Port: {output_port_name} -> {input_port_name}")

                        try:
                            output_port = node.get_output(output_port_name)
                            input_port = visual_node.get_input(input_port_name)

                            if output_port and input_port:
                                output_port.connect_to(input_port)
                                print(f"      ✓ Connected")
                            else:
                                print(f"      ✗ Port not found")
                        except Exception as e:
                            print(f"      ✗ Error: {str(e)}")

                        # ビジュアルノードをfixedジョイントに設定
                        visual_node.rotation_axis = 3  # fixedジョイント

                node_count += 1

            # 子ボディ数に合わせて出力ポートを追加（MJCF用）
            # NOTE: ベースボディ（例: "base"）が複数の脚を持つ場合、
            # 出力ポートが不足すると脚が接続されない
            for body_name, child_count in body_child_counts.items():
                if body_name not in nodes:
                    continue
                node = nodes[body_name]
                try:
                    current_output_count = len(node.output_ports())
                except Exception:
                    current_output_count = 0
                needed_ports = child_count - current_output_count
                if needed_ports > 0 and hasattr(node, '_add_output'):
                    for _ in range(needed_ports):
                        node._add_output()
                    print(f"  ✓ Added {needed_ports} output port(s) to '{body_name}' (children: {child_count})")

            # ジョイント情報を反映して接続
            parent_port_indices = {}
            connected_pairs = set()

            print(f"\n[DEBUG] Starting joint connections")
            print(f"[DEBUG] Total joints to process: {len(joints_data)}")
            print(f"[DEBUG] Available nodes: {list(nodes.keys())}")

            for joint_data in joints_data:
                parent_name = joint_data['parent']
                child_name = joint_data['child']

                print(f"\n[DEBUG] Processing joint: {joint_data.get('name')}")
                print(f"  parent: {parent_name}, child: {child_name}")

                if parent_name not in nodes or child_name not in nodes:
                    print(f"  [DEBUG] Skipping: parent_in_nodes={parent_name in nodes}, child_in_nodes={child_name in nodes}")
                    continue

                parent_node = nodes[parent_name]
                child_node = nodes[child_name]

                # 既に同じ親子が接続されている場合はスキップ（多自由度ジョイント対策）
                pair_key = (parent_name, child_name)
                if pair_key in connected_pairs:
                    print(f"  ℹ Skip duplicate joint connection: {parent_name} -> {child_name}")
                    continue

                # 親ノードの現在のポートインデックスを取得
                port_index = None
                if parent_name in child_order_by_parent and child_name in child_order_by_parent[parent_name]:
                    port_index = child_order_by_parent[parent_name].index(child_name)
                    print(f"  [DEBUG] port_index from child_order_by_parent: {port_index}")
                if port_index is None:
                    if parent_name not in parent_port_indices:
                        parent_port_indices[parent_name] = 0
                    port_index = parent_port_indices[parent_name]
                    parent_port_indices[parent_name] += 1
                    print(f"  [DEBUG] port_index from parent_port_indices: {port_index}")

                # ポイント情報を親ノードに追加
                if not hasattr(parent_node, 'points'):
                    parent_node.points = []

                while len(parent_node.points) <= port_index:
                    parent_node.points.append({
                        'name': f'point_{len(parent_node.points)}',
                        'type': 'revolute',
                        'xyz': [0.0, 0.0, 0.0],
                        'rpy': [0.0, 0.0, 0.0]
                    })

                # ジョイントタイプを文字列に変換
                joint_type_str = joint_data['type']

                # Set angle for 3D view and UI display, keep rpy as zero to avoid double rotation
                # angle is used for 3D view display and URDF/MJCF export
                parent_node.points[port_index] = {
                    'name': joint_data['name'],
                    'type': joint_type_str,
                    'xyz': joint_data['origin_xyz'],
                    'rpy': [0.0, 0.0, 0.0],  # Keep rpy as zero (3D view uses angle)
                    'angle': list(joint_data['origin_rpy'])  # Set angle for 3D view display (radians)
                }

                # 子ノードにジョイント情報を設定
                # ファイルに定義されている場合は上書き、未定義の場合はSettingsのデフォルト値を使用
                if 'limit' in joint_data:
                    if 'lower' in joint_data['limit']:
                        child_node.joint_lower = joint_data['limit']['lower']
                    if 'upper' in joint_data['limit']:
                        child_node.joint_upper = joint_data['limit']['upper']
                    if 'effort' in joint_data['limit']:
                        child_node.joint_effort = joint_data['limit']['effort']
                    if 'velocity' in joint_data['limit']:
                        child_node.joint_velocity = joint_data['limit']['velocity']

                # dynamics情報を設定（damping, frictionloss, armature）
                # ファイルに定義されている場合は上書き、未定義の場合はSettingsのデフォルト値を使用
                if 'dynamics' in joint_data:
                    if 'damping' in joint_data['dynamics']:
                        child_node.joint_damping = joint_data['dynamics']['damping']
                        print(f"  Set joint_damping: {child_node.joint_damping}")

                    if 'friction' in joint_data['dynamics']:
                        child_node.joint_frictionloss = joint_data['dynamics']['friction']
                        print(f"  Set joint_frictionloss: {child_node.joint_frictionloss}")

                    if 'armature' in joint_data['dynamics']:
                        child_node.joint_armature = joint_data['dynamics']['armature']
                        print(f"  Set joint_armature: {child_node.joint_armature}")
                    
                    if 'stiffness' in joint_data['dynamics']:
                        child_node.joint_stiffness = joint_data['dynamics']['stiffness']
                        print(f"  Set joint_stiffness: {child_node.joint_stiffness}")
                    
                    if 'margin' in joint_data['dynamics']:
                        child_node.joint_margin = joint_data['dynamics']['margin']
                        print(f"  Set joint_margin: {child_node.joint_margin}")

                # stiffnessを設定
                if 'stiffness' in joint_data:
                    child_node.joint_stiffness = joint_data['stiffness']
                    print(f"  Set joint_stiffness: {child_node.joint_stiffness}")

                # actuation_lagを設定（存在する場合）
                if 'actuation_lag' in joint_data and hasattr(child_node, 'actuation_lag'):
                    child_node.actuation_lag = joint_data['actuation_lag']
                    print(f"  Set actuation_lag: {child_node.actuation_lag}")

                # 回転軸を設定
                # NOTE: rotation_axis is already set in body_data from MJCF/URDF parser
                # Do NOT override it here with joint_data['axis'] because:
                # - joint_data['axis'] is in parent coordinate frame (axis_parent)
                # - rotation_axis should be determined from local axis (axis_local)
                # - MJCFParser/URDFParser already calculated the correct rotation_axis
                # Only log the axis for debugging
                axis = joint_data['axis']
                axis_names = {0: 'X', 1: 'Y', 2: 'Z', 3: 'Fixed'}
                if hasattr(child_node, 'rotation_axis'):
                    axis_name = axis_names.get(child_node.rotation_axis, 'Unknown')
                    print(f"  Joint axis (parent frame): {axis} | rotation_axis: {child_node.rotation_axis} ({axis_name})")
                else:
                    print(f"  Joint axis (parent frame): {axis} | rotation_axis: not set")

                # ジョイントタイプがfixedの場合
                if joint_data['type'] == 'fixed':
                    child_node.rotation_axis = 3
                    print(f"  Fixed joint -> rotation_axis: 3")

                # MJCFのjointのref属性（参照角度）をbody_angleとして設定
                # NOTE: ref = ジョイントのデフォルト姿勢/参照点
                # body_angleのみに設定し、point['angle']には設定しない
                # （point['angle']とbody_angleは連動するが、インポート時は片方のみに設定）
                if 'ref' in joint_data and joint_data['type'] != 'fixed':
                    ref_angle_deg = joint_data['ref']  # 度単位
                    rotation_axis = child_node.rotation_axis

                    # 回転軸に応じて適切な軸のangleを設定
                    # NOTE: body_angleは"ラジアン"で保持される
                    # （3Dビジュアライゼーションでmath.degrees()で度に変換されるため）
                    angle_offset_rad = [0.0, 0.0, 0.0]
                    ref_angle_rad = math.radians(ref_angle_deg)
                    if rotation_axis == 0:  # X軸
                        angle_offset_rad[0] = ref_angle_rad
                    elif rotation_axis == 1:  # Y軸
                        angle_offset_rad[1] = ref_angle_rad
                    elif rotation_axis == 2:  # Z軸
                        angle_offset_rad[2] = ref_angle_rad

                    # 子ノードのbody_angleに設定（ラジアンで保存）
                    child_node.body_angle = angle_offset_rad
                    print(f"  ✓ Set body_angle from ref: {ref_angle_deg} degrees ({ref_angle_rad:.4f} radians)")
                    print(f"  ℹ Note: point['angle'] not set (will be synced from body_angle when needed)")

                # ポートを接続
                is_base_link = parent_node.__class__.__name__ == 'BaseLinkNode'
                # 親ノードの出力ポート数が足りない場合は追加（MJCF用の保険）
                if hasattr(parent_node, '_add_output'):
                    try:
                        current_output_count = len(parent_node.output_ports())
                    except Exception:
                        current_output_count = 0
                    needed_ports = (port_index + 1) - current_output_count
                    if needed_ports > 0:
                        for _ in range(needed_ports):
                            parent_node._add_output()
                        print(f"  ✓ Added {needed_ports} output port(s) to '{parent_name}' before connect (port_index={port_index})")
                if is_base_link:
                    output_port_name = 'out' if port_index == 0 else f'out_{port_index + 1}'
                else:
                    output_port_name = f'out_{port_index + 1}'

                input_port_name = 'in'

                print(f"\nConnecting: {parent_name} -> {child_name}")
                print(f"  Port: {output_port_name} -> {input_port_name}")

                try:
                    output_port = parent_node.get_output(output_port_name)
                    input_port = child_node.get_input(input_port_name)

                    if output_port and input_port:
                        output_port.connect_to(input_port)
                        connected_pairs.add(pair_key)
                        print(f"  ✓ Connected")
                        
                        # NOTE: body_angleは同期しない（MJCFのref専用）
                        # point['angle']（origin_rpy）とbody_angle（ref）は別々の回転として3Dビューで適用される
                        # 同期すると二重回転が発生する
                    else:
                        print(f"  ✗ Port not found")
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")

            # baseノードをbase_linkに自動接続
            print("\n=== Connecting base node to base_link ===")
            base_candidate_node = None
            base_candidate_name = None

            # 1. まず'base'という名前のノードを探す
            if 'base' in nodes:
                base_candidate_node = nodes['base']
                base_candidate_name = 'base'
                print(f"  Found 'base' node: {base_candidate_name}")
            else:
                # 2. baseがない場合は、INが接続されていないノードの中で子ノード数が最大のものを探す
                print("  'base' node not found, searching for node with most children and no input connection...")
                
                # INが接続されていないノードを探す
                unconnected_nodes = []
                for body_name, node in nodes.items():
                    # base_link系は除外
                    if body_name in ['base_link', 'base_link_mjcf', 'base_link_sub']:
                        continue
                    
                    # 入力ポートが接続されているかチェック
                    input_ports = node.input_ports()
                    is_connected = False
                    for input_port in input_ports:
                        if input_port.connected_ports():
                            is_connected = True
                            break
                    
                    if not is_connected:
                        child_count = body_child_counts.get(body_name, 0)
                        unconnected_nodes.append((body_name, node, child_count))
                        print(f"    Found unconnected node: {body_name} (children: {child_count})")
                
                # 子ノード数が最大のノードを選択
                if unconnected_nodes:
                    unconnected_nodes.sort(key=lambda x: x[2], reverse=True)  # 子ノード数でソート
                    base_candidate_name, base_candidate_node, max_child_count = unconnected_nodes[0]
                    print(f"  Selected node with most children: {base_candidate_name} (children: {max_child_count})")
                else:
                    print("  No unconnected nodes found")

            # base候補ノードが見つかった場合、base_link_subを優先して接続
            if base_candidate_node and base_candidate_name:
                # 既に接続されている場合はスキップ
                input_ports = base_candidate_node.input_ports()
                is_already_connected = False
                for input_port in input_ports:
                    if input_port.connected_ports():
                        is_already_connected = True
                        break
                
                if not is_already_connected:
                    try:
                        target_parent_node = base_link_sub_node or base_link_mjcf_node or base_node
                        if target_parent_node is None:
                            print("  ✗ No parent node available for base connection")
                        else:
                            # 接続されていない出力ポートを探す
                            output_ports = target_parent_node.output_ports()
                            available_output_port = None
                            for output_port in output_ports:
                                if not output_port.connected_ports():
                                    available_output_port = output_port
                                    break
                            if available_output_port is None and hasattr(target_parent_node, '_add_output'):
                                target_parent_node._add_output()
                                available_output_port = target_parent_node.output_ports()[-1]
                                print(f"  ✓ Added output port to {target_parent_node.name()} for base connection")
                            if available_output_port:
                                base_candidate_input_port = base_candidate_node.input_ports()[0]
                                available_output_port.connect_to(base_candidate_input_port)
                                print(f"  ✓ Connected {target_parent_node.name()} -> {base_candidate_name}")
                            else:
                                print(f"  ✗ No available output port on {target_parent_node.name()}")
                    except Exception as e:
                        print(f"  ✗ Error connecting to {base_candidate_name}: {str(e)}")
                        traceback.print_exc()
                else:
                    print(f"  {base_candidate_name} is already connected, skipping")
            else:
                print("  No base candidate node found")

            # 接続されていないノードをMJCFファイルから調べて接続
            print("\n=== Connecting unconnected nodes from MJCF hierarchy ===")
            
            # bodies_data_listから親情報を取得（元のリスト形式を使用）
            body_parent_map = {}
            for body_data in bodies_data_list:
                body_name = body_data.get('name')
                parent_name = body_data.get('parent')
                if body_name and parent_name:
                    body_parent_map[body_name] = parent_name
                    print(f"  Body hierarchy: {parent_name} -> {body_name}")
            
            # 接続されていないノードを探す
            unconnected_nodes = []
            for body_name, node in nodes.items():
                # base_linkとbase_link_mjcfは除外
                if body_name == 'base_link' or body_name == 'base_link_mjcf':
                    continue
                
                # 入力ポートが接続されているかチェック
                input_ports = node.input_ports()
                is_connected = False
                for input_port in input_ports:
                    if input_port.connected_ports():
                        is_connected = True
                        break
                
                if not is_connected:
                    unconnected_nodes.append((body_name, node))
                    print(f"  Found unconnected node: {body_name}")
            
            # 接続されていないノードについて、親ノードを探して接続
            for body_name, node in unconnected_nodes:
                parent_name = body_parent_map.get(body_name)
                
                if parent_name:
                    print(f"\n  Attempting to connect {body_name} to parent {parent_name}...")
                    
                    # 親ノードを探す
                    parent_node = None
                    if parent_name in nodes:
                        parent_node = nodes[parent_name]
                    elif parent_name == 'base_link':
                        parent_node = base_node
                    elif parent_name == 'base_link_mjcf' and base_link_mjcf_node:
                        parent_node = base_link_mjcf_node
                    
                    if parent_node:
                        try:
                            # 親ノードの出力ポートを探す（接続されていないポートを優先）
                            parent_output_ports = parent_node.output_ports()
                            available_output_port = None
                            
                            # まず接続されていないポートを探す
                            for output_port in parent_output_ports:
                                if not output_port.connected_ports():
                                    available_output_port = output_port
                                    break
                            
                            # 接続されていないポートがない場合は、新しいポートを追加
                            if not available_output_port:
                                if not parent_node.__class__.__name__ == 'BaseLinkNode':
                                    parent_node._add_output()
                                    available_output_port = parent_node.output_ports()[-1]
                                    print(f"    Added new output port to {parent_name}")
                                else:
                                    print(f"    Warning: {parent_name} is BaseLinkNode and all ports are used, skipping")
                                    continue
                            
                            # 子ノードの入力ポートを取得
                            child_input_port = node.input_ports()[0]
                            
                            # 接続
                            if available_output_port and child_input_port:
                                available_output_port.connect_to(child_input_port)
                                
                                # ポート名を取得（デバッグ用）
                                output_port_name = available_output_port.name()
                                print(f"    ✓ Connected {parent_name} ({output_port_name}) -> {body_name}")
                                
                                # ポイント情報を設定（ジョイント情報があれば使用）
                                joint_data_for_node = None
                                for joint_data in joints_data:
                                    if joint_data['parent'] == parent_name and joint_data['child'] == body_name:
                                        joint_data_for_node = joint_data
                                        break
                                
                                if joint_data_for_node:
                                    # 親ノードのpointsに情報を追加
                                    if not hasattr(parent_node, 'points'):
                                        parent_node.points = []
                                    
                                    # 使用されているポートのインデックスを取得
                                    port_index = parent_output_ports.index(available_output_port)
                                    
                                    # points配列を拡張
                                    while len(parent_node.points) <= port_index:
                                        parent_node.points.append({
                                            'name': f'point_{len(parent_node.points)}',
                                            'type': 'revolute',
                                            'xyz': [0.0, 0.0, 0.0],
                                            'rpy': [0.0, 0.0, 0.0]
                                        })
                                    
                                    # ジョイント情報を設定
                                    origin_rpy = joint_data_for_node['origin_rpy']
                                    parent_node.points[port_index] = {
                                        'name': joint_data_for_node['name'],
                                        'type': joint_data_for_node['type'],
                                        'xyz': joint_data_for_node['origin_xyz'],
                                        'rpy': [0.0, 0.0, 0.0],  # Keep rpy as zero (3D view uses angle)
                                        'angle': list(origin_rpy)  # Set angle for 3D view display (radians)
                                    }
                                    
                                    # 子ノードにジョイント情報を設定
                                    # ファイルに定義されている場合は上書き、未定義の場合はSettingsのデフォルト値を使用
                                    if 'limit' in joint_data_for_node:
                                        if 'lower' in joint_data_for_node['limit']:
                                            node.joint_lower = joint_data_for_node['limit']['lower']
                                        if 'upper' in joint_data_for_node['limit']:
                                            node.joint_upper = joint_data_for_node['limit']['upper']
                                        if 'effort' in joint_data_for_node['limit']:
                                            node.joint_effort = joint_data_for_node['limit']['effort']
                                        if 'velocity' in joint_data_for_node['limit']:
                                            node.joint_velocity = joint_data_for_node['limit']['velocity']
                                    
                                    # dynamics情報を設定
                                    # ファイルに定義されている場合は上書き、未定義の場合はSettingsのデフォルト値を使用
                                    if 'dynamics' in joint_data_for_node:
                                        if 'damping' in joint_data_for_node['dynamics']:
                                            node.joint_damping = joint_data_for_node['dynamics']['damping']
                                        if 'friction' in joint_data_for_node['dynamics']:
                                            node.joint_frictionloss = joint_data_for_node['dynamics']['friction']
                                        if 'armature' in joint_data_for_node['dynamics']:
                                            node.joint_armature = joint_data_for_node['dynamics']['armature']
                                        if 'stiffness' in joint_data_for_node['dynamics']:
                                            node.joint_stiffness = joint_data_for_node['dynamics']['stiffness']
                                        if 'margin' in joint_data_for_node['dynamics']:
                                            node.joint_margin = joint_data_for_node['dynamics']['margin']
                                    
                                    # 回転軸を設定
                                    axis = joint_data_for_node['axis']
                                    if abs(axis[2]) > 0.5:  # Z軸
                                        node.rotation_axis = 2
                                    elif abs(axis[1]) > 0.5:  # Y軸
                                        node.rotation_axis = 1
                                    else:  # X軸
                                        node.rotation_axis = 0
                                    
                                    # ジョイントタイプがfixedの場合
                                    if joint_data_for_node['type'] == 'fixed':
                                        node.rotation_axis = 3
                                    
                                    print(f"    ✓ Set joint information for {body_name}")
                            else:
                                print(f"    ✗ Could not find available ports")
                        except Exception as e:
                            print(f"    ✗ Error connecting {body_name} to {parent_name}: {str(e)}")
                            traceback.print_exc()
                    else:
                        print(f"    ✗ Parent node '{parent_name}' not found in nodes")
                else:
                    print(f"  No parent information found for {body_name} in MJCF hierarchy")
            
            # 接続されたノード数をカウント
            connected_count = 0
            for body_name, node in unconnected_nodes:
                input_ports = node.input_ports()
                for input_port in input_ports:
                    if input_port.connected_ports():
                        connected_count += 1
                        break
            print(f"\n  Connected {connected_count} out of {len(unconnected_nodes)} unconnected nodes")

            # 閉リンクジョイントノードの作成と接続（MJCFのequalityセクションから）
            # 位置再計算の前に実行する必要がある
            print("\n=== Creating Closed-Loop Joint Nodes from MJCF equality section ===")
            if closed_loop_joints:
                closed_loop_nodes = {}
                parent_port_indices = {}  # 親ノードごとのポートインデックスを管理
                
                for joint_data in closed_loop_joints:
                    joint_name = joint_data['name']
                    parent_link = joint_data['parent']
                    child_link = joint_data['child']
                    
                    # 閉リンクノードの名前を決定
                    cl_node_name = f"{joint_name}_CL"
                    
                    # 親リンクと子リンクがノードとして存在するか確認
                    if parent_link not in nodes:
                        print(f"Skipping closed-loop joint '{joint_name}': parent link '{parent_link}' not found")
                        continue
                    if child_link not in nodes:
                        print(f"Skipping closed-loop joint '{joint_name}': child link '{child_link}' not found")
                        continue
                    
                    parent_node = nodes[parent_link]
                    child_node = nodes[child_link]
                    
                    # 閉リンクノードを作成（親と子の中間位置に配置）
                    parent_pos = parent_node.pos()
                    child_pos = child_node.pos()
                    
                    if isinstance(parent_pos, (list, tuple)):
                        parent_x, parent_y = parent_pos[0], parent_pos[1]
                    else:
                        parent_x, parent_y = parent_pos.x(), parent_pos.y()
                    
                    if isinstance(child_pos, (list, tuple)):
                        child_x, child_y = child_pos[0], child_pos[1]
                    else:
                        child_x, child_y = child_pos.x(), child_pos.y()
                    
                    # 中間位置を計算
                    cl_node_x = (parent_x + child_x) / 2
                    cl_node_y = (parent_y + child_y) / 2
                    cl_node_pos = QtCore.QPointF(cl_node_x, cl_node_y)
                    
                    # 閉リンクノードを作成
                    print(f"Creating closed-loop node: {cl_node_name} ({parent_link} <-> {child_link})")
                    cl_node = graph.create_node(
                        'insilico.nodes.ClosedLoopJointNode',
                        name=cl_node_name,
                        pos=cl_node_pos
                    )
                    
                    # メタデータを設定
                    cl_node.joint_name = joint_name
                    cl_node.joint_type = joint_data.get('original_type', 'ball')
                    cl_node.parent_link = parent_link
                    cl_node.child_link = child_link
                    cl_node.origin_xyz = joint_data.get('origin_xyz', [0.0, 0.0, 0.0])
                    cl_node.origin_rpy = joint_data.get('origin_rpy', [0.0, 0.0, 0.0])
                    cl_node.gearbox_ratio = joint_data.get('gearbox_ratio', 1.0)
                    cl_node.gearbox_reference_body = joint_data.get('gearbox_reference_body')
                    
                    # 閉リンクノードのpointsデータを初期化
                    if not hasattr(cl_node, 'points'):
                        cl_node.points = []
                    
                    closed_loop_nodes[cl_node_name] = cl_node
                    
                    # 親リンクの出力ポートインデックスを取得
                    if parent_link not in parent_port_indices:
                        parent_port_indices[parent_link] = 0
                    parent_port_index = parent_port_indices[parent_link]
                    parent_port_indices[parent_link] += 1
                    
                    # 親ノードのpointsデータが不足している場合は初期化
                    if not hasattr(parent_node, 'points'):
                        parent_node.points = []
                    
                    # 必要なpointsデータを追加
                    while len(parent_node.points) <= parent_port_index:
                        parent_node.points.append({
                            'name': f'point_{len(parent_node.points) + 1}',
                            'type': 'fixed',
                            'xyz': [0.0, 0.0, 0.0],
                            'rpy': [0.0, 0.0, 0.0],
                            'angle': [0.0, 0.0, 0.0]
                        })
                    
                    # 親ノードのpointsに閉リンクジョイントのorigin情報を設定
                    origin_rpy = joint_data.get('origin_rpy', [0.0, 0.0, 0.0])
                    parent_node.points[parent_port_index]['xyz'] = joint_data.get('origin_xyz', [0.0, 0.0, 0.0])
                    parent_node.points[parent_port_index]['rpy'] = [0.0, 0.0, 0.0]  # Keep rpy as zero (3D view uses angle)
                    parent_node.points[parent_port_index]['name'] = joint_name
                    parent_node.points[parent_port_index]['type'] = joint_data.get('original_type', 'ball')
                    parent_node.points[parent_port_index]['angle'] = list(origin_rpy)  # Set angle for 3D view (radians)
                    
                    # 親リンクのポート名を取得
                    is_base_link_node = parent_node.__class__.__name__ == 'BaseLinkNode'
                    if is_base_link_node:
                        if parent_port_index == 0:
                            parent_output_port_name = 'out'
                        else:
                            parent_output_port_name = f'out_{parent_port_index + 1}'
                    else:
                        parent_output_port_name = f'out_{parent_port_index + 1}'
                    
                    # 親リンク -> 閉リンクノードの接続
                    print(f"  Connecting {parent_link}.{parent_output_port_name} -> {cl_node_name}.in")
                    try:
                        # 親ノードの出力ポートが不足している場合は追加
                        parent_output_ports = parent_node.output_ports()
                        if parent_port_index >= len(parent_output_ports):
                            needed_ports = parent_port_index + 1 - len(parent_output_ports)
                            for i in range(needed_ports):
                                if hasattr(parent_node, '_add_output'):
                                    parent_node._add_output()
                                elif hasattr(parent_node, 'add_output'):
                                    parent_node.output_count = getattr(parent_node, 'output_count', 0) + 1
                                    new_port_name = f'out_{parent_node.output_count}'
                                    parent_node.add_output(new_port_name, color=(180, 80, 0))
                        
                        parent_output_port = parent_node.get_output(parent_output_port_name)
                        cl_input_port = cl_node.get_input('in')
                        
                        if parent_output_port and cl_input_port:
                            parent_output_port.connect_to(cl_input_port)
                            print(f"    ✓ Connected {parent_link} -> {cl_node_name}")
                        else:
                            print(f"    ✗ Port not found")
                    except Exception as e:
                        print(f"    ✗ Error connecting {parent_link} -> {cl_node_name}: {str(e)}")
                        traceback.print_exc()
                    
                    # 閉リンクノード -> 子リンクの接続
                    # NOTE: 閉リンク構造では、子リンクが既に階層構造の親を持っている場合がある
                    # その場合、入力ポートは既に接続されているため、接続をスキップする
                    print(f"  Connecting {cl_node_name}.out -> {child_link}.in")
                    try:
                        cl_output_port = cl_node.get_output('out')
                        child_input_port = child_node.get_input('in')

                        if cl_output_port and child_input_port:
                            # 子リンクの入力ポートが既に接続されているかチェック
                            if child_input_port.connected_ports():
                                print(f"    ⚠ {child_link}.in is already connected (closed-loop structure)")
                                print(f"      Closed-loop joint node created but not connected to child")
                            else:
                                cl_output_port.connect_to(child_input_port)
                                print(f"    ✓ Connected {cl_node_name} -> {child_link}")
                        else:
                            print(f"    ✗ Port not found")
                    except Exception as e:
                        print(f"    ✗ Error connecting {cl_node_name} -> {child_link}: {str(e)}")
                        traceback.print_exc()
                
                print(f"\nCreated {len(closed_loop_nodes)} closed-loop joint nodes")
            else:
                print("No closed-loop joints found in MJCF equality section")

            # メッシュをSTL viewerにロード
            print("\n=== Loading meshes to 3D viewer ===")
            if graph.stl_viewer:
                stl_viewer_loaded_count = 0
                skipped_no_mesh = 0
                skipped_base_link = 0

                for body_name, node in nodes.items():
                    if body_name == 'base_link':
                        print(f"Skipping base_link")
                        skipped_base_link += 1
                        continue

                    # デバッグ: ノードの属性を確認
                    has_stl_attr = hasattr(node, 'stl_file')
                    stl_value = getattr(node, 'stl_file', None) if has_stl_attr else None

                    print(f"\nChecking node: {body_name}")
                    print(f"  has stl_file attr: {has_stl_attr}")
                    print(f"  stl_file value: {stl_value}")

                    if has_stl_attr and stl_value:
                        try:
                            print(f"  → Loading mesh to viewer...")
                            graph.stl_viewer.load_stl_for_node(node)
                            stl_viewer_loaded_count += 1
                            print(f"  ✓ Successfully loaded mesh")
                        except Exception as e:
                            print(f"  ✗ Error loading mesh: {str(e)}")
                            traceback.print_exc()
                    else:
                        print(f"  ✗ Skipped (no mesh file assigned)")
                        skipped_no_mesh += 1

                print(f"\n--- Mesh Loading Summary ---")
                print(f"Successfully loaded: {stl_viewer_loaded_count}")
                print(f"Skipped (no mesh): {skipped_no_mesh}")
                print(f"Skipped (base_link): {skipped_base_link}")
                print("=" * 40 + "\n")
            else:
                print("Warning: Mesh viewer not available")

            # 位置を再計算（閉リンクジョイントノード作成後）
            print("\nRecalculating positions...")
            QtCore.QTimer.singleShot(100, graph.recalculate_all_positions)
            
            # STL読み込み完了後、すべてのノードのカラーを3Dビューに適用
            # タイマーを使って、STL読み込みが完了してから色を適用
            def apply_colors_after_stl_load():
                if graph.stl_viewer:
                    print("\nApplying colors to 3D view after MJCF import...")
                    graph._apply_colors_to_all_nodes()
            
            # STL読み込みが完了するのを待つため、少し遅延を入れる
            QtCore.QTimer.singleShot(500, apply_colors_after_stl_load)

            # 一時ディレクトリのクリーンアップ（ZIPファイルの場合）
            # 注: メッシュファイルは既にノードに読み込まれているため、一時ファイルは削除可能
            # ただし、すぐに削除するとビューアが読み込めない可能性があるため、
            # 数秒後に削除するようスケジュール
            if xml_file_to_load and working_dir != os.path.dirname(mjcf_file):
                import shutil
                def cleanup_temp_dir():
                    try:
                        if os.path.exists(working_dir):
                            shutil.rmtree(working_dir)
                            print(f"Cleaned up temporary directory: {working_dir}")
                    except Exception as e:
                        print(f"Warning: Could not clean up temporary directory: {e}")

                # 5秒後にクリーンアップ（メッシュが読み込まれるのを待つ）
                QtCore.QTimer.singleShot(5000, cleanup_temp_dir)

            QtWidgets.QMessageBox.information(
                graph.widget,
                "MJCF Import Complete",
                f"Successfully imported {len(bodies_data)} bodies and {len(joints_data)} joints from MJCF file."
            )

            # UIのName:フィールドを更新
            if hasattr(graph, 'name_input') and graph.name_input:
                graph.name_input.setText(graph.robot_name)
                print(f"Updated Name field to: {graph.robot_name}")

            return True

        except Exception as e:
            print(f"Error importing MJCF: {str(e)}")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                graph.widget,
                "MJCF Import Error",
                f"Failed to import MJCF file:\n\n{str(e)}"
            )
            return False

# ============================================================================
# IMPORTER WINDOW - Standalone UI for importing URDF/MJCF models
# ============================================================================

class ImporterWindow(QtWidgets.QWidget):
    """Model Importer Window for URDF Kitchen
    
    This window provides a simple UI for importing URDF/SDF and MJCF models
    into the Assembler graph. It acts as a launcher that calls the import
    functions defined in this module.
    """

    def __init__(self, graph):
        """Initialize the Importer Window
        
        Args:
            graph: Reference to the Assembler's CustomNodeGraph instance
        """
        super().__init__()
        self.graph = graph
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        # Set window title
        self.setWindowTitle("URDF Kitchen - Model Importer")

        # Set window background style (matching Assembler)
        self.setStyleSheet("""
            QWidget {
                background-color: #404244;
            }
        """)

        # Create main layout
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Button style matching Launcher
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

        # Title label (combined with line-height control)
        title_label = QtWidgets.QLabel()
        title_label.setText(
            '<div style="text-align: center;">'
            '<p style="font-weight: bold; margin: 0 0 5px 0;">- MODEL Importer -</p>'
            '<p style="margin: 0;">I\'ll do my best :)</p>'
            '</div>'
        )
        title_label.setStyleSheet("color: #ffffff;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)

        # Add some spacing (reduced to half)
        layout.addSpacing(5)

        # Import URDF/SDF button
        urdf_btn = QtWidgets.QPushButton("Import URDF/SDF")
        urdf_btn.setStyleSheet(button_style)
        urdf_btn.clicked.connect(self.on_import_urdf)
        layout.addWidget(urdf_btn)

        # Import MJCF button
        mjcf_btn = QtWidgets.QPushButton("Import MJCF")
        mjcf_btn.setStyleSheet(button_style)
        mjcf_btn.clicked.connect(self.on_import_mjcf)
        layout.addWidget(mjcf_btn)

        # Add separator line
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

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setStyleSheet(button_style)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # Set layout
        self.setLayout(layout)

        # Set fixed size for the window
        self.setFixedSize(250, 240)

        # Position the window at the center of the screen
        self.center_on_screen()

    def center_on_screen(self):
        """Position the window at the center of the screen"""
        screen = QtGui.QGuiApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        
        # Calculate center position
        x = (screen_rect.width() - self.width()) // 2 + screen_rect.left()
        y = (screen_rect.height() - self.height()) // 2 + screen_rect.top()
        
        self.move(x, y)

    def on_import_urdf(self):
        """Handle Import URDF/SDF button click"""
        try:
            result = import_urdf(self.graph)
            if result:
                print("URDF/SDF import completed successfully")
                # Close window after successful import
                QtCore.QTimer.singleShot(0, self.close)
        except Exception as e:
            print(f"Error during URDF/SDF import: {str(e)}")
            traceback.print_exc()

    def on_import_mjcf(self):
        """Handle Import MJCF button click"""
        try:
            result = import_mjcf(self.graph)
            if result:
                print("MJCF import completed successfully")
                # Close window after successful import
                QtCore.QTimer.singleShot(0, self.close)
        except Exception as e:
            print(f"Error during MJCF import: {str(e)}")
            traceback.print_exc()

    def keyPressEvent(self, event):
        """Handle key press events"""
        # Close window with standard close shortcut (Cmd+W / Ctrl+W)
        if event.matches(QtGui.QKeySequence.Close):
            self.close()
            return
        # Close window with Escape key
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event"""
        # グラフオブジェクトの参照をクリア
        if hasattr(self.graph, 'importer_window'):
            self.graph.importer_window = None
        super().closeEvent(event)


# ============================================================================
# STANDALONE EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # For standalone testing, create a mock graph object
    class MockGraph:
        def __init__(self):
            self.widget = None
            self.robot_name = "test_robot"
            self.stl_viewer = None
            self.name_input = None
    
    mock_graph = MockGraph()
    window = ImporterWindow(mock_graph)
    window.show()
    
    sys.exit(app.exec())
