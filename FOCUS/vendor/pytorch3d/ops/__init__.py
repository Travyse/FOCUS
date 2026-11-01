# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in vendor tree ('LICENSE').
#
# Vendored from pytorch3d 0.7.8. Pruned pure-Python subset: only the modules
# that do not require the compiled C++/CUDA extension are exposed.

# pyre-unsafe

from .interp_face_attrs import interpolate_face_attributes
from .laplacian_matrices import cot_laplacian, laplacian, norm_laplacian

from .mesh_face_areas_normals import mesh_face_areas_normals
from .packed_to_padded import packed_to_padded, padded_to_packed
from .sample_points_from_meshes import _rand_barycentric_coords, sample_points_from_meshes