# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in vendor tree ('LICENSE').
#
# Vendored pytorch-level fallback reimplemented from the reference
# implementation in pytorch3d 0.7.8's test suite
# (tests/test_packed_to_padded.py::packed_to_padded_python /
# padded_to_packed_python), which is verified to match the compiled C++/CUDA
# version to within numerical tolerance. No modification of semantics.

# pyre-unsafe

import torch


def _packed_to_padded_python(inputs, first_idxs, max_size):
    """
    PyTorch implementation of packed_to_padded function.
    """
    num_meshes = first_idxs.size(0)
    if inputs.dim() == 1:
        inputs_padded = torch.zeros(
            (num_meshes, max_size), device=inputs.device, dtype=inputs.dtype
        )
    else:
        inputs_padded = torch.zeros(
            (num_meshes, max_size, *inputs.shape[1:]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
    for m in range(num_meshes):
        s = first_idxs[m]
        if m == num_meshes - 1:
            f = inputs.shape[0]
        else:
            f = first_idxs[m + 1]
        inputs_padded[m, : f - s] = inputs[s:f]

    return inputs_padded


def _padded_to_packed_python(inputs, first_idxs, num_inputs):
    """
    PyTorch implementation of padded_to_packed function.
    """
    num_meshes = inputs.size(0)
    if inputs.dim() == 2:
        inputs_packed = torch.zeros(
            (num_inputs,), device=inputs.device, dtype=inputs.dtype
        )
    else:
        inputs_packed = torch.zeros(
            (num_inputs, *inputs.shape[2:]), device=inputs.device, dtype=inputs.dtype
        )
    for m in range(num_meshes):
        s = first_idxs[m]
        if m == num_meshes - 1:
            f = num_inputs
        else:
            f = first_idxs[m + 1]
        inputs_packed[s:f] = inputs[m, : f - s]

    return inputs_packed


def packed_to_padded(
    inputs: torch.Tensor, first_idxs: torch.LongTensor, max_size: int
) -> torch.Tensor:
    """
    Torch wrapper that handles allowed input shapes. See description below.

    Args:
        inputs: FloatTensor of shape (F,) or (F, ...), representing the packed
            batch tensor, e.g. areas for faces in a batch of meshes.
        first_idxs: LongTensor of shape (N,) where N is the number of
            elements in the batch and `first_idxs[i] = f`
            means that the inputs for batch element i begin at `inputs[f]`.
        max_size: Max length of an element in the batch.

    Returns:
        inputs_padded: FloatTensor of shape (N, max_size) or (N, max_size, ...)
            where max_size is max of `sizes`. The values for batch element i
            which start at `inputs[first_idxs[i]]` will be copied to
            `inputs_padded[i, :]`, with zeros padding out the extra inputs.

    To handle the allowed input shapes, we convert the inputs tensor of shape
    (F,) to (F, 1). We reshape the output back to (N, max_size) from
    (N, max_size, 1).
    """
    # if inputs is of shape (F,), reshape into (F, 1)
    input_shape = inputs.shape
    n_dims = inputs.dim()
    if n_dims == 1:
        inputs = inputs.unsqueeze(1)
    else:
        inputs = inputs.reshape(input_shape[0], -1)
    inputs_padded = _packed_to_padded_python(inputs, first_idxs, max_size)
    # if flat is True, reshape output to (N, max_size) from (N, max_size, 1)
    # else reshape output to (N, max_size, ...)
    if n_dims == 1:
        return inputs_padded.squeeze(2)
    return inputs_padded.view(*inputs_padded.shape[:2], *input_shape[1:])


def padded_to_packed(
    inputs: torch.Tensor,
    first_idxs: torch.LongTensor,
    num_inputs: int,
    max_size_dim: int = 1,
) -> torch.Tensor:
    """
    Torch wrapper that handles allowed input shapes. See description below.

    Args:
        inputs: FloatTensor of shape (N, ..., max_size) or (N, ..., max_size, ...),
            representing the padded tensor, e.g. areas for faces in a batch of
            meshes, where max_size occurs on max_size_dim-th position.
        first_idxs: LongTensor of shape (N,) where N is the number of
            elements in the batch and `first_idxs[i] = f`
            means that the inputs for batch element i begin at `inputs_packed[f]`.
        num_inputs: Number of packed entries (= F)
        max_size_dim: the dimension to be packed

    Returns:
        inputs_packed: FloatTensor of shape (F,) or (F, ...) where
            `inputs_packed[first_idx[i]:first_idx[i+1]] = inputs[i, ..., :delta[i]]`,
            where `delta[i] = first_idx[i+1] - first_idx[i]`.

    To handle the allowed input shapes, we convert the inputs tensor of shape
    (N, max_size) to (N, max_size, 1). We reshape the output back to (F,) from
    (F, 1).
    """
    n_dims = inputs.dim()
    # move the variable dim to position 1
    inputs = inputs.movedim(max_size_dim, 1)

    # if inputs is of shape (N, max_size), reshape into (N, max_size, 1))
    input_shape = inputs.shape
    if n_dims == 2:
        inputs = inputs.unsqueeze(2)
    else:
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    inputs_packed = _padded_to_packed_python(inputs, first_idxs, num_inputs)
    # if flat is True, reshape output to (F,) from (F, 1) or (F, ...)
    if n_dims == 2:
        return inputs_packed.squeeze(1)
    elif n_dims == 3:
        return inputs_packed.view(inputs_packed.shape[0], *input_shape[2:])
    return inputs_packed