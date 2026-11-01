import os
import sys

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")

# Make the vendored, pure-Python pytorch3d/geffnet subsets importable under
# their original package names. These are vendored because the FOCUS codebase
# (and its inlined DSINE/FOUND/FIND model code) uses APIs from them, but the
# full packages cannot be installed on some secure compute environments.
# Set FOCUS_SKIP_VENDOR=1 to skip this and use site-packages versions instead
# (e.g. on machines where PyTorch3D is compiled and installed).
if os.environ.get("FOCUS_SKIP_VENDOR", "0") != "1":
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)