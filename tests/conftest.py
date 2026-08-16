import pytest
import trimesh
import os
import tempfile


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def cube_stl(tmp_dir):
    path = os.path.join(tmp_dir, "cube.stl")
    cube = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    cube.export(path)
    return path


@pytest.fixture
def l_shape_stl(tmp_dir):
    path = os.path.join(tmp_dir, "l_shape.stl")
    box1 = trimesh.creation.box(extents=[0.1, 0.1, 0.2])
    box2 = trimesh.creation.box(extents=[0.1, 0.2, 0.1])
    box2.apply_translation([0, 0.05, -0.05])
    combined = trimesh.util.concatenate([box1, box2])
    combined.export(path)
    return path
