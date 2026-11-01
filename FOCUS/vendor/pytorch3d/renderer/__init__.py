# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in vendor tree ('LICENSE').
#
# Vendored from pytorch3d 0.7.8 (commit 81d82980bc82fd605f27cca87f89ba08af94db3d).
# This is a pruned package: only the pure-Python modules required by the FOCUS
# codebase are vendored. Anything requiring the compiled C++/CUDA extension
# (`pytorch3d._C`) - rasterizers, knn, chamfer kernels, etc. - has been removed.
# See FOCUS/vendor/README.md for details.

# pyre-unsafe

from .cameras import (
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    get_ndc_to_screen_transform,
    get_screen_to_ndc_transform,
    get_world_to_view_transform,
    look_at_rotation,
    look_at_view_transform,
    OrthographicCameras,
    PerspectiveCameras,
)
from .mesh.textures import (
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
)
from .utils import (
    TensorProperties,
    convert_to_tensors_and_broadcast,
)