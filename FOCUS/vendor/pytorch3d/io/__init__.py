# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in vendor tree ('LICENSE').
#
# Vendored from pytorch3d 0.7.8. Pruned pure-Python subset: only the OBJ
# loaders are exposed (PLY/IO plugins etc. have been removed).

# pyre-unsafe

from .obj_io import load_obj, load_objs_as_meshes, save_obj