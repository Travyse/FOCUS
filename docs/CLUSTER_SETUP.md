# FOCUS on a secure cluster (pip-only setup)

Everything FOCUS needs is installable with `pip` alone - no conda, no compiled
COLMAP binary, no git submodules, no network downloads at run time.

## 1. Environment

Python 3.9+ is required (3.9 and 3.10+ are both supported and tested).

```bash
git clone https://github.com/Travyse/FOCUS.git
cd FOCUS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # includes pycolmap (COLMAP's C++ core)
pip install -e . --no-deps
```

Notes:

- **COLMAP**: nothing to install separately. `pycolmap` (pinned in
  requirements.txt) ships COLMAP's C++ core as a compiled wheel and runs the
  exact mapper pipeline the CLI would. `FOCUS/calibration/run_colmap.py`
  automatically uses the COLMAP executable when one is present and falls back
  to pycolmap when it is not - no configuration needed either way.
- Python 3.9 environments should end up with `numpy<2`-era pins resolved by
  pip (e.g. `numpy==2.0.2`); this combination is tested (pycolmap 0.6.1 +
  torch 2.5.1 + numpy 2.0.2 on CPython 3.9).
- If your platform already has a compiled PyTorch3D installed and you prefer
  to use it instead of the vendored pure-Python subset, set
  `FOCUS_SKIP_VENDOR=1`.

## 2. TOC model checkpoint

The checkpoint ships split inside the repository (GitHub's 100 MB file limit):

```bash
cat data/toc_model/densedepth_toc_predictor.pth.part-* > data/toc_model/densedepth_toc_predictor.pth
sha256sum -c data/toc_model/densedepth_toc_predictor.sha256
```

(`sha256sum -c` needs the GNU coreutils package bundled with every Linux
distribution.)

## 3. One-time sanity check

```bash
python tests/test_selfcontained.py          # from the repository root
```

Expect `Ran 5 tests ... OK`. This verifies the vendored pytorch3d/geffnet
imports, the camera math, the FIND model load, and the TOC predictor build
(no network access required).

## 4. Running on a video

```bash
mkdir -p outputs && cd <repository root>
python FOCUS/run_focus.py \
  --method o \
  --video_path /path/anywhere/video.mp4 \
  --output_folder outputs/run01 \
  --make_predictions \
  --num_frames 30
```

- `--output_folder` must be a fresh directory, and never the video's folder or
  an ancestor of it (`--overwrite` deletes its contents).
- COLMAP and ffmpeg are used automatically: ffmpeg for frame extraction, and
  COLMAP (via pycolmap) for calibration.
- The video file itself is only ever read; all artifacts are written under
  `--output_folder`.
- Add `--render` only on machines where Blender is installed; `--produce_videos`
  additionally needs the `ffmpeg` binary on PATH.