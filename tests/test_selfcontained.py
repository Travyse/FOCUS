"""Tests that FOCUS runs without the PyTorch3D / geffnet (EfficientNet) / PyAV
pip packages, by exercising the vendored pure-Python code in FOCUS/vendor and
the inlined DSINE / FOUND / FIND model code.

Run from the repository root:  python -m pytest tests/ -q
(or simply: python tests/test_selfcontained.py)
"""

import os
import sys
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Make sure we test THIS repository's code, not any installed copy of `focus`.
sys.path.insert(0, REPO)


class TestImports(unittest.TestCase):
    def test_imports_without_compiled_dependencies(self):
        """The vendored pytorch3d must import without the compiled C++ extension."""
        import FOCUS

        import pytorch3d
        import pytorch3d.transforms
        import pytorch3d.structures
        import pytorch3d.renderer.cameras
        import pytorch3d.renderer.mesh.textures
        import pytorch3d.io
        from pytorch3d import ops

        # Vendored package is in use (unless explicitly skipped):
        if not os.environ.get("FOCUS_SKIP_VENDOR") == "1":
            self.assertIn(
                os.path.join(REPO, "FOCUS", "vendor"),
                pytorch3d.__file__,
                "FOCUS import should activate the vendored pytorch3d",
            )

        import geffnet  # noqa

    def test_pure_pytorch3d_api_surface(self):
        import FOCUS  # noqa

        from pytorch3d.transforms import Transform3d, euler_angles_to_matrix  # noqa
        from pytorch3d.structures import Meshes  # noqa
        from pytorch3d.renderer import TexturesVertex  # noqa
        from pytorch3d.renderer.cameras import (  # noqa
            PerspectiveCameras,
            get_ndc_to_screen_transform,
        )
        from pytorch3d.io import load_obj  # noqa
        from pytorch3d.ops import sample_points_from_meshes  # noqa


class TestCameraMath(unittest.TestCase):
    """Golden values from the reference (compiled) PyTorch3D implementation."""

    def test_perspective_projection_golden(self):
        import FOCUS  # noqa
        import torch

        from FOCUS.utils.camera import Camera
        from FOCUS.utils import triangulation

        cam = Camera(
            R=torch.eye(3)[None],
            T=torch.tensor([[0.0, 0.0, 3.0]]),
            size=torch.tensor([192, 192])[None],
            focal_length=torch.tensor([500.0]),
        )

        # Point at world origin seen from a fronto-parallel camera:
        pix = cam.transform_points_screen(torch.zeros(1, 1, 3), eps=1e-6)[..., :2]
        self.assertTrue(torch.allclose(pix, torch.tensor([[[96.0, 96.0]]])))

        # Point offset by (dx, dy, 0): pix = (cx - f*dx/(z+3), cy - f*dy/(z+3))
        pts = torch.tensor([[[0.3, 0.4, 0.0]]])
        pix = cam.transform_points_screen(pts, eps=1e-6)[..., :2]
        expected = torch.tensor([[[96.0 - 500 * 0.3 / 3, 96.0 - 500 * 0.4 / 3]]])
        self.assertTrue(torch.allclose(pix, expected))

        # Triangulation projection matrices: shape (N, 3, 4), finite.
        P = triangulation.cameras_to_projection_matrix(cam)
        self.assertEqual(P.shape, (1, 3, 4))
        self.assertTrue(torch.isfinite(P).all())


class TestModelsLoad(unittest.TestCase):
    def test_find_model_loads_cpu(self):
        import FOCUS  # noqa
        import torch

        from FOCUS.optim.find.found_model import FIND

        find_dir = os.path.join(REPO, "data", "find")
        if not os.path.exists(os.path.join(find_dir, "model.pth")):
            self.skipTest("data/find model not present")

        find = FIND(find_dir)
        meshes = find()
        self.assertEqual(meshes.verts_padded().shape[0], 1)
        self.assertTrue(torch.isfinite(meshes.verts_padded()).all())

    def test_toc_predictor_builds_no_download(self):
        """FootPredictorModel builds with pretrained encoder disabled (no network)."""
        import FOCUS  # noqa
        import torch

        from FOCUS.toc_prediction.model import FootPredictorModel

        model = FootPredictorModel()
        out = model(torch.rand(1, 3, 256, 256))
        self.assertEqual(tuple(out["TOC"].shape), (1, 3, 256, 256))


if __name__ == "__main__":
    unittest.main()