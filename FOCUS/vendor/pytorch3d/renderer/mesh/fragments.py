# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in vendor tree ('LICENSE').
#
# The Fragments dataclass vendored from pytorch3d 0.7.8
# (pytorch3d/renderer/mesh/rasterizer.py) so that it can be imported without
# the compiled C++/CUDA rasterization extension.

# pyre-unsafe

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Fragments:
    """
    A dataclass representing the outputs of a rasterizer. Can be detached from the
    computational graph in order to stop the gradients from flowing through the
    rasterizer.

    Members:
        pix_to_face:
            LongTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the indices of the nearest faces at each pixel, sorted in ascending
            z-order. Concretely ``pix_to_face[n, y, x, k] = f`` means that
            ``faces_verts[f]`` is the kth closest face (in the z-direction) to pixel
            (y, x). Pixels that are hit by fewer than faces_per_pixel are padded with
            -1.

        zbuf:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the NDC z-coordinates of the nearest faces at each pixel, sorted in
            ascending z-order. Concretely, if ``pix_to_face[n, y, x, k] = f`` then
            ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
            faces_per_pixel are padded with -1.

        bary_coords:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel, 3)
            giving the barycentric coordinates in NDC units of the nearest faces at
            each pixel, sorted in ascending z-order. Concretely, if ``pix_to_face[n,
            y, x, k] = f`` then ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives the
            barycentric coords for pixel (y, x) relative to the face defined by
            ``face_verts[f]``. Pixels hit by fewer than faces_per_pixel are padded
            with -1.

        dists:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the signed Euclidean distance (in NDC units) in the x/y plane of each
            point closest to the pixel. Concretely if ``pix_to_face[n, y, x, k] = f``
            then ``pix_dists[n, y, x, k]`` is the squared distance between the pixel
            (y, x) and the face given by vertices ``face_verts[f]``. Pixels hit with
            fewer than ``faces_per_pixel`` are padded with -1.
    """

    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: Optional[torch.Tensor]

    def detach(self) -> "Fragments":
        return Fragments(
            pix_to_face=self.pix_to_face,
            zbuf=self.zbuf.detach(),
            bary_coords=self.bary_coords.detach(),
            dists=self.dists.detach() if self.dists is not None else self.dists,
        )