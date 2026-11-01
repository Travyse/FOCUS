# Vendored third-party code

This directory contains third-party code vendored into FOCUS so that the
project runs on secure compute clusters where some dependencies cannot be
installed.

| Package | Version | Source | License | Why |
|---------|---------|--------|---------|-----|
| `pytorch3d` | 0.7.8 (commit `81d82980bc82fd605f27cca87f89ba08af94db3d`) | https://github.com/facebookresearch/pytorch3d | BSD (see `pytorch3d/LICENSE`) | Cannot be compiled/installed on the cluster. Only the pure-Python modules actually used by FOCUS / FOUND / FIND are vendored; all modules that require the compiled C++/CUDA extension (`pytorch3d._C`) have been removed. |
| `geffnet` | 1.0.2 | https://github.com/rwightman/gen-efficientnet-pytorch | Apache-2.0 (see `geffnet/LICENSE`) | Not approved for installation by company policy. Pure Python. The EfficientNet-B5 (TF-weights) encoder used by the TOC prediction model is built from here, with `pretrained=False` so no network download occurs. |

## What was changed in the vendored pytorch3d

Only package `__init__.py` files were pruned to avoid importing compiled-only
modules, plus these small, behavior-preserving shims:

- `ops/interp_face_attrs.py` — `pytorch3d._C` import made optional; falls back
  to the Python implementation when the extension is unavailable.
- `ops/mesh_face_areas_normals.py` — pure PyTorch reimplementation taken from
  the official reference implementation in pytorch3d 0.7.8's test suite
  (`tests/test_face_areas_normals.py::face_areas_normals_python`).
- `ops/packed_to_padded.py` — pure PyTorch reimplementation taken from the
  official reference implementation in pytorch3d 0.7.8's test suite
  (`tests/test_packed_to_padded.py`).
- `ops/sample_points_from_meshes.py` — verbatim except `Fragments` is imported
  from the pure-Python shim `renderer/mesh/fragments.py` instead of
  `renderer/mesh/rasterizer.py`.
- `renderer/mesh/fragments.py` — new file; the `Fragments` dataclass copied
  verbatim from `renderer/mesh/rasterizer.py`.

To force use of a fully installed (compiled) PyTorch3D instead of this vendored
subset, set the environment variable `FOCUS_SKIP_VENDOR=1` before launching.