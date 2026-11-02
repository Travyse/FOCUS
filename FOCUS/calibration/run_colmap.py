"""Run COLMAP Bundle Adjustment on a set of images.

Two interchangeable backends:

 1. The COLMAP executable (selected via `--colmap_exe`), used automatically
    whenever it is present. Behaviour identical to the original implementation,
    including per-step log files.

 2. `pycolmap` (https://github.com/colmap/pycolmap), used automatically when
    no executable is available. pycolmap wraps COLMAP's C++ core compiled
    into the pip wheel, so the exact same pipeline runs without installing
    any binary - useful on secure clusters where only `pip` is available.
    The database written by FOCUS (via `toc_matches_to_database`) and the
    sparse model format read afterwards are identical in both backends.

Only the `mapper` stage runs for custom (TOC-based) matches; feature
extraction / matching are replaced by them in both backends.
"""

from pathlib import Path
import os
import subprocess
from tqdm import tqdm
import json
import platform

from FOCUS.calibration.colmap_format_conversion import colmap2pytorch3d
from FOCUS.calibration.custom_matches_colmap import toc_matches_to_database

ACCEPTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Use shell if Windows only
use_shell = platform.system() == 'Windows'


def _check_colmap(colmap_exe: str = 'colmap'):
    try:
        subprocess.check_call([colmap_exe, 'help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=use_shell)
    except FileNotFoundError:
        raise FileNotFoundError(f'`{colmap_exe}` does not run. Make sure COLMAP is installed, and either added to PATH, or the `colmap_exe` argument points to the correct location.')


def _colmap_backend(colmap_exe: str) -> str:
    """Decide between the COLMAP executable and the pycolmap bindings.

    The executable wins whenever it runs; otherwise we fall back to pycolmap
    (raising a helpful error if neither is available)."""
    try:
        _check_colmap(colmap_exe)
        return 'cli'
    except (OSError, subprocess.CalledProcessError):
        # FileNotFoundError (not installed), OSError (e.g. incompatible binary
        # that fails to exec) or non-zero exit (`colmap help`) -> fall back.
        pass

    try:
        import pycolmap  # noqa: F401
    except ImportError:
        raise FileNotFoundError(
            f'`{colmap_exe}` does not run and `pycolmap` is not installed. '
            'Install COLMAP (and add it to PATH or pass `--colmap_exe`), '
            'or install the Python bindings: `pip install pycolmap`.'
        )
    return 'pycolmap'


def _run_mapper_cli(image_dir, output_dir, colmap_exe, predictions_folder,
                    num_correspondences):
    """Original implementation: run COLMAP through the executable."""
    workspace_dir = output_dir / 'colmap'
    workspace_dir.mkdir(parents=True, exist_ok=True)

    database_path = os.path.join(workspace_dir, 'database.db')
    sparse_dir = workspace_dir / 'sparse'
    sparse_dir.mkdir(exist_ok=True)

    img_ids = []
    commands = {}
    if predictions_folder is not None:
        # Use custom matches.
        img_ids = toc_matches_to_database(predictions_folder, workspace_dir, num_correspondences=num_correspondences)
        image_dir = predictions_folder  # Switch to image_dir being the predicted images (so that sizing is consistent
                                        # with predictions).
        commands['mapper'] = {'database_path': database_path, 'image_path': image_dir, 'output_path': sparse_dir,
                              'Mapper.ba_refine_focal_length': '1'}
    else:
        img_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(ACCEPTED_IMAGE_EXTENSIONS)]
        commands['feature_extractor'] = {
            'image_path': image_dir,
            'database_path': database_path,
            'ImageReader.single_camera': '1'
        }
        commands['exhaustive_matcher'] = {'database_path': database_path}
        commands['mapper'] = {'database_path': database_path, 'image_path': image_dir, 'output_path': sparse_dir,
                              'Mapper.ba_refine_focal_length': '1'}

    with tqdm(commands.items()) as progress_bar:
        for method, command in progress_bar:
            progress_bar.set_description(f'COLMAP: Running `{method}`')

            args = [colmap_exe, method]
            for k, v in command.items():
                args += [f'--{k}', str(v)]

            # Run subprocess check call, saving output to log file.
            logfile = workspace_dir / f'{method}_log.txt'
            subprocess.check_call(args, stdout=open(logfile, 'w'), stderr=subprocess.STDOUT, shell=use_shell)

    return img_ids


def _run_mapper_pycolmap(image_dir, output_dir, predictions_folder,
                         num_correspondences):
    """Run the identical COLMAP pipeline through the pycolmap bindings."""
    import pycolmap

    workspace_dir = output_dir / 'colmap'
    workspace_dir.mkdir(parents=True, exist_ok=True)

    database_path = str(workspace_dir / 'database.db')
    sparse_dir = workspace_dir / 'sparse'

    img_ids = []
    if predictions_folder is not None:
        # Use custom matches: keypoints/matches are injected into the database
        # by FOCUS (sqlite) - only the mapper stage remains.
        img_ids = toc_matches_to_database(predictions_folder, workspace_dir, num_correspondences=num_correspondences)
        image_dir = predictions_folder

        with tqdm([('mapper',)]) as progress_bar:
            for method, in progress_bar:
                progress_bar.set_description(f'COLMAP (pycolmap): Running `{method}`')
                maps = pycolmap.incremental_mapping(
                    database_path, str(image_dir), str(sparse_dir),
                    options=pycolmap.IncrementalPipelineOptions())
    else:
        img_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(ACCEPTED_IMAGE_EXTENSIONS)]

        with tqdm([('feature_extractor',), ('exhaustive_matcher',), ('mapper',)]) as progress_bar:
            for method, in progress_bar:
                progress_bar.set_description(f'COLMAP (pycolmap): Running `{method}`')
                if method == 'feature_extractor':
                    pycolmap.extract_features(
                        database_path, str(image_dir),
                        camera_mode=pycolmap.CameraMode.SINGLE,
                        camera_model='SIMPLE_RADIAL')
                elif method == 'exhaustive_matcher':
                    pycolmap.match_exhaustive(database_path)
                else:
                    maps = pycolmap.incremental_mapping(
                        database_path, str(image_dir), str(sparse_dir),
                        options=pycolmap.IncrementalPipelineOptions())

    if not maps:
        raise ValueError(
            'COLMAP reconstructed no models (no images with matches). '
            'Try increasing the number of matches/correspondences.'
        )

    return img_ids


def run_colmap(image_dir: Path, output_dir: Path, colmap_exe: str = 'colmap',
               predictions_folder: Path = None,
               num_correspondences: int = 2500) -> None:
    """
    :param image_dir: Directory containing images.
    :param output_dir: Output directory for COLMAP data.
    :param colmap_exe: Path to COLMAP executable. If it cannot run, the
        pycolmap bindings are used instead.
    :param predictions_folder: If given, use these for custom matches. Otherwise, use COLMAP feature extraction & matching.
    :param num_correspondences: For custom matches, how many correspondences to use.
    """

    backend = _colmap_backend(colmap_exe)

    if backend == 'cli':
        img_ids = _run_mapper_cli(image_dir, output_dir, colmap_exe,
                                  predictions_folder, num_correspondences)
    else:
        img_ids = _run_mapper_pycolmap(image_dir, output_dir,
                                       predictions_folder, num_correspondences)

    workspace_dir = output_dir / 'colmap'
    sparse_dir = workspace_dir / 'sparse'

    # Convert to PyTorch3D format
    output_data = colmap2pytorch3d(str(sparse_dir))

    with open(output_dir / 'colmap.json', 'w') as f:
        json.dump(output_data, f)

    # Export per view.
    failed_views = []
    for view in img_ids:
        view_data = {**output_data['camera']}

        for i in output_data['images']:
            if i['pth'].startswith(view):
                view_data['R'] = i['R']
                view_data['T'] = i['T']
                view_data['C'] = i['C']
                break
        else:
            failed_views.append(view)
            continue

        view_dir = output_dir / view
        view_dir.mkdir(exist_ok=True)
        with open(view_dir / f'colmap.json', 'w') as f:
            json.dump(view_data, f)

    if failed_views:
        raise ValueError(f'Failed to calibrate {len(failed_views)} / {len(img_ids)} images. '
                      'Try increasing --num_colmap_matches.')


if __name__ == '__main__':
    # Small self-check of the chosen backend.
    print('COLMAP backend:', _colmap_backend(os.environ.get('COLMAP_EXE', 'colmap')))