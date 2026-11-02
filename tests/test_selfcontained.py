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


class TestVideoUtils(unittest.TestCase):
    """ffmpeg resolution + frame handling (PATH install or imageio-ffmpeg fallback)."""

    N_FRAMES = 12

    @classmethod
    def setUpClass(cls):
        import imageio
        import numpy as np
        cls.tmp = __import__("tempfile").mkdtemp(prefix="focus_video_test_")
        cls.video_path = os.path.join(cls.tmp, "synth.mp4")
        # note: mimicwrite consumes RGB frames
        frames = (np.random.RandomState(0).rand(cls.N_FRAMES, 64, 96, 3) * 255).astype("uint8")
        imageio.mimwrite(cls.video_path, frames, fps=cls.N_FRAMES // 2)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_get_ffmpeg_exe_resolves_to_existing_binary(self):
        from FOCUS.utils.video import get_ffmpeg_exe
        exe = get_ffmpeg_exe()
        self.assertTrue(os.path.exists(exe), exe)

    def test_focus_ffmpeg_env_override(self):
        from FOCUS.utils.video import get_ffmpeg_exe
        sentinel = sys.executable  # any runnable file works as a sentinel
        try:
            os.environ["FOCUS_FFMPEG"] = sentinel
            self.assertEqual(get_ffmpeg_exe(), sentinel)
        finally:
            os.environ.pop("FOCUS_FFMPEG", None)

    def test_frame_count_ffprobe_and_fallback_agree(self):
        import shutil as _shutil

        from FOCUS.utils import video as video_utils

        has_ffprobe = _shutil.which("ffprobe") is not None
        count = video_utils.get_frame_count(self.video_path)
        self.assertEqual(count, self.N_FRAMES)

        if has_ffprobe:
            # force the imageio fallback path and check agreement
            orig_which = video_utils.shutil.which

            def fake_which(cmd, *a, **k):
                return None

            video_utils.shutil.which = fake_which
            try:
                self.assertEqual(video_utils.get_frame_count(self.video_path), self.N_FRAMES)
            finally:
                video_utils.shutil.which = orig_which

    def test_frames_from_video_round_trip(self):
        import argparse
        import cv2

        # run_focus imports the fusion chain, which imports pymeshlab; stub it
        # only on platforms where it is not installable (no py39-arm64 wheel).
        try:
            import pymeshlab  # noqa: F401
        except ImportError:
            import types
            stub = types.ModuleType("pymeshlab")
            stub.MeshSet = object
            stub.Mesh = object
            sys.modules["pymeshlab"] = stub

        import FOCUS.run_focus as run_focus_mod

        out_dir = os.path.join(self.tmp, "out")
        args = argparse.Namespace(
            video_path=__import__("pathlib").Path(self.video_path),
            output_folder=__import__("pathlib").Path(out_dir),
            num_frames=4,
        )
        run_focus_mod._frames_from_video(args)

        frames_dir = os.path.join(out_dir, "frames")
        pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        self.assertEqual(len(pngs), 4)
        for f in pngs:
            img = cv2.imread(os.path.join(frames_dir, f))
            self.assertIsNotNone(img)


if __name__ == "__main__":
    unittest.main()