# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in vendor tree ('LICENSE').
#
# Vendored from pytorch3d 0.7.8. Pruned pure-Python subset: only textures are
# exposed; rasterizers and shaders requiring the compiled C++/CUDA extension
# have been removed.

# pyre-unsafe

from .textures import TexturesAtlas, TexturesUV, TexturesVertex