import os
import yaml


def load_config(config_path=None):
    """Load YAML config with defaults for missing keys."""
    defaults = {
        'decomposition': {
            'threshold': 0.05,
            'max_hulls': -1,
            'preprocess': 'auto',
            'resolution': 2000,
            'mcts_nodes': 20,
            'mcts_iterations': 150,
        },
        'cleaning': {
            'fix_normals': True,
            'remove_duplicates': True,
            'fill_holes': False,
            'merge_tolerance': 0.0001,
        },
        'output': {
            'format': 'stl',
            'merge_hulls': True,
            'suffix': '_collision',
        },
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user = yaml.safe_load(f) or {}
        for section in defaults:
            if section in user:
                defaults[section].update(user[section])
    return defaults


def validate_input(path):
    """Check file exists and has a supported extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    supported = {'.stl', '.obj', '.ply', '.off', '.glb', '.gltf'}
    if ext not in supported:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(sorted(supported))}")
    return path


def ensure_output_dir(path):
    """Create parent directories for output path if needed."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def format_filesize(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"
