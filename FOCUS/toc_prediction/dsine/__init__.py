# DenseDepth encoder-decoder, vendored from DSINE
# (https://github.com/baegwangbin/DSINE, commit at time of vendoring), which
# is an adaptation of DenseDepth (https://github.com/ialhashim/DenseDepth).
# Inlined here so the codebase is self-contained and does not require
# `git submodule` or the `geffnet` pip package.
#
# See LICENSE notices in the individual files and the DSINE license
# (DSINE_LICENSE.md) next to this package.

from FOCUS.toc_prediction.dsine.dense_depth import DenseDepth

__all__ = ["DenseDepth"]