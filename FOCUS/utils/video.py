from __future__ import annotations
import os
import shutil
import subprocess
import json
from pathlib import Path


def get_ffmpeg_exe() -> str:
	"""Resolve an ffmpeg executable.

	Resolution order:
	 1. ``$FOCUS_FFMPEG`` environment variable (explicit override),
	 2. ``ffmpeg`` on PATH (system, brew or .deb installs),
	 3. the ffmpeg binary bundled inside the ``imageio-ffmpeg`` wheel
	    (already a dependency; ships a static build per platform).

	Note: ``ffprobe`` is *not* bundled with imageio-ffmpeg; frame counting
	falls back to ``imageio_ffmpeg.count_frames_and_secs`` in
	`get_frame_count` when ffprobe is unavailable.
	"""
	env_override = os.environ.get("FOCUS_FFMPEG")
	if env_override:
		return env_override

	exe = shutil.which("ffmpeg")
	if exe:
		return exe

	try:
		import imageio_ffmpeg
	except ImportError:
		pass
	else:
		return imageio_ffmpeg.get_ffmpeg_exe()

	raise FileNotFoundError(
		"ffmpeg was not found: it is not on PATH, $FOCUS_FFMPEG is not set, "
		"and `imageio-ffmpeg` is not installed. Install ffmpeg (e.g. `apt "
		"install ffmpeg`) or `pip install imageio-ffmpeg`."
	)


def get_frame_count(video_path: str | Path) -> int:
	"""Calculate number of frames in video file.

	Uses ffprobe when available (preferred), otherwise counts frames via the
	ffmpeg binary resolved by `get_ffmpeg_exe()` - e.g. the one bundled with
	imageio-ffmpeg, which ships without ffprobe. Both methods decode the full
	video and agree on the count.
	"""
	if shutil.which("ffprobe"):
		cmd = [
			"ffprobe",
			"-v", "error",
			"-select_streams", "v:0",
			"-count_frames",
			"-show_entries", "stream=nb_read_frames",
			"-of", "json",
			str(video_path),
		]

		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		info = json.loads(result.stdout)
		return int(info["streams"][0]["nb_read_frames"])

	# ffprobe unavailable (e.g. only imageio-ffmpeg's bundled ffmpeg is
	# present): count frames by decoding the file and parsing output.
	import imageio_ffmpeg
	nframes, _ = imageio_ffmpeg.count_frames_and_secs(str(video_path))
	return int(nframes)