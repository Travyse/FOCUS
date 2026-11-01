# Model code inlined from FOUND (https://github.com/OllieBoyne/FOUND) and its
# nested FIND model (https://github.com/OllieBoyne/FIND), both MIT licensed
# (see FIND_LICENSE.md / FOUND_LICENSE.md). Inlined so the codebase is
# self-contained and does not require git submodules or a compiled PyTorch3D
# package.
#
# Only the model-building and inference code paths used by FOCUS are included;
# training-only code (renderers, losses) is not part of this package.

from FOCUS.optim.find.found_model import FIND

__all__ = ["FIND"]