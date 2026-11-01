# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in vendor tree ('LICENSE').
#
# Vendored pytorch-level fallback reimplemented from the reference
# implementation in pytorch3d 0.7.8's test suite
# (tests/test_face_areas_normals.py::face_areas_normals_python), which is
# verified to match the compiled C++/CUDA version to within numerical
# tolerance. No modification of semantics.

# pyre-unsafe

import torch
import torch.nn.functional as F


def mesh_face_areas_normals(verts, faces):
    """
    Computes the area and normals of each face.
    Args:
        verts: FloatTensor of shape (V, 3), the packed batch verts tensor.
        faces: LongTensor of shape (F, 3), the packed batch faces tensor.
    Returns:
        areas: FloatTensor of shape (F,) with the area of each face.
        normals: FloatTensor of shape (F, 3) with the unit normal of each face.
    """
    if not (verts.dim() == 2):
        raise ValueError("verts need to be of shape Vx3.")
    if not (verts.shape[1] == 3):
        raise ValueError("verts need to be of shape Vx3.")
    if not (faces.dim() == 2):
        raise ValueError("faces need to be of shape Fx3.")
    if not (faces.shape[1] == 3):
        raise ValueError("faces need to be of shape Fx3.")
    if not (faces.dtype == torch.int64):
        raise ValueError("faces need to be of type torch.int64.")
    # TODO(gkioxari) Change cast to floats once we add support for doubles.
    if not (verts.dtype == torch.float32):
        verts = verts.float()

    vertices_faces = verts[faces]  # (F, 3, 3)
    # vector pointing from v0 to v1
    v01 = vertices_faces[:, 1] - vertices_faces[:, 0]
    # vector pointing from v0 to v2
    v02 = vertices_faces[:, 2] - vertices_faces[:, 0]
    normals = torch.cross(v01, v02, dim=1)  # (F, 3)
    face_areas = normals.norm(dim=-1) / 2
    face_normals = F.normalize(normals, p=2, dim=1, eps=1e-6)
    return face_areas, face_normals