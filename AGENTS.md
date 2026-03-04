# Agent Guide for frustum-pointnets
# Target: agentic coding assistants operating in this repo.

## Scope and overview
- This is a TensorFlow-based research codebase for Frustum PointNets.
- Code mixes Python 2/3 compatibility and TF1-style graphs via `tf.compat.v1`.
- Primary workflows are dataset preparation, training, testing, and evaluation.
- There is no standard packaging, no requirements.txt, and no CI config.
- Scripts live in `scripts/`, training/testing in `train/`, models in `models/`.

## Environment assumptions
- Python is invoked as `python`.
- TensorFlow is expected; many scripts call `tf.compat.v1.disable_v2_behavior()`.
- Data files are expected under `dataset/` and `kitti/` or `sunrgbd/`.
- GPU use is controlled via `--gpu` flags in training/testing scripts.

## Build, lint, test commands
### Data preparation (KITTI)
- Prepare KITTI frustum pickles:
  - `python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection`
  - Script reference: `scripts/command_prep_data.sh`

### Training (KITTI)
- Train v1:
  - `python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/log_v1 --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5`
  - Script reference: `scripts/command_train_v1.sh`
- Train v2:
  - `python train/train.py --gpu 0 --model frustum_pointnets_v2 --log_dir train/log_v2 --num_point 1024 --max_epoch 201 --batch_size 24 --decay_step 800000 --decay_rate 0.5`
  - Script reference: `scripts/command_train_v2.sh`

### Testing and evaluation (KITTI)
- Test v1 and evaluate:
  - `python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v1 --model_path train/log_v1/model.ckpt --output train/detection_results_v1 --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle --from_rgb_detection --idx_path kitti/image_sets/val.txt --from_rgb_detection`
  - `train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v1`
  - Script reference: `scripts/command_test_v1.sh`
- Test v2 and evaluate:
  - `python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v2 --model_path train/log_v2/model.ckpt --output train/detection_results_v2 --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle --from_rgb_detection --idx_path kitti/image_sets/val.txt --from_rgb_detection`
  - `train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v2`
  - Script reference: `scripts/command_test_v2.sh`

### Build native eval tool (KITTI)
- Compile KITTI evaluator:
  - `g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp`
  - Script reference: `train/kitti_eval/compile.sh`

### SUN-RGBD training/testing/eval
- Train:
  - `python sunrgbd/sunrgbd_detection/train_one_hot.py --gpu 0 --model frustum_pointnets_v1_sunrgbd --log_dir log --num_point 2048 --max_epoch 151 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --learning_rate 0.001 --momentum 0.9 --optimizer adam`
- Test:
  - `python sunrgbd/sunrgbd_detection/test_one_hot.py --gpu 0 --num_point 2048 --model frustum_pointnets_v1_sunrgbd --model_path log/model.ckpt --dump_result`
- Evaluate:
  - `python sunrgbd/sunrgbd_detection/evaluate.py --data_path ../sunrgbd_data/fcn_det_val.zip.pickle --result_path test_results_v1_fcn_ft_val.pickle --from_rgb_detection`
  - See `sunrgbd/README.md` for context and data prep notes.

### Linting
- No lint or formatter is configured in this repository.
- Do not introduce new linters without explicit request.

### Single-test guidance
- There is no unified test runner.
- For TF op tests, run the specific test script directly, for example:
  - `python models/tf_ops/grouping/tf_grouping_op_test.py`
  - `python models/tf_ops/3d_interpolation/tf_interpolate_op_test.py`
- For a single evaluation run, call the target script with a minimal dataset or
  a specific index list if the script supports it (e.g., `--idx_path`).

## Code style guidelines
### Imports and module layout
- Standard library imports first, then third-party (numpy/tensorflow), then local.
- Keep `from __future__ import print_function` at top if present in the file.
- Many scripts modify `sys.path` to add `train/` or `models/`; preserve this
  pattern when adding new modules to avoid breaking legacy imports.
- Avoid adding new dependencies unless requested; this repo is dependency-light.

### Formatting and whitespace
- Use 4-space indentation (no tabs).
- Keep line lengths reasonable; follow existing style within the file.
- Use blank lines to separate logical sections (imports, constants, functions).
- Preserve existing docstring style (triple-quoted strings, often brief).

### Naming conventions
- Modules and functions are snake_case.
- Constants are ALL_CAPS in module scope (e.g., `NUM_POINT`, `BN_DECAY_CLIP`).
- TensorFlow tensors and placeholders often end in `_pl` (e.g., `labels_pl`).
- Training flags are stored in `FLAGS` from `argparse`.

### Types and compatibility
- Codebase is TF1-style graph construction; use `tf.compat.v1` APIs when needed.
- Do not introduce Python typing annotations unless a file already uses them.
- Maintain Python 2/3 compatibility patterns where present (e.g., `cPickle`).

### Error handling and assertions
- The code uses lightweight assertions for invariants (shape, range checks).
- Prefer explicit `assert` for data assumptions rather than heavy try/except.
- If adding user-facing errors, raise `ValueError` with a concise message.

### TensorFlow graph patterns
- Use `tf.Graph().as_default()` and device scopes in training/testing scripts.
- Batch norm decay and learning rate are computed with `tf.train.exponential_decay`.
- When adding ops, follow existing `end_points` dict patterns for outputs.
- Avoid eager execution; scripts call `tf.compat.v1.disable_v2_behavior()`.

### Data and file handling
- Dataset files are pickles stored under `kitti/` or `sunrgbd/` paths.
- Do not hardcode new absolute paths; prefer repo-relative paths or args.
- When writing outputs, follow existing `output` or `output_dir` conventions.

### Logging
- Training scripts write logs to `log_dir` with `log_train.txt`.
- Keep logging minimal; use `log_string()` helpers when available.

### Tests and evaluation scripts
- Test scripts are standalone Python files, not pytest modules.
- Some evaluation scripts use interactive `raw_input()`; preserve behavior.

## Repository-specific notes
- There are no Cursor rules or Copilot instruction files in this repo.
- There is no `AGENTS.md` yet; this file is the source of guidance.
- If you add new scripts, consider a `scripts/command_*.sh` entry for parity.

## Safe change practices
- Avoid reformatting large files; keep diffs focused.
- Respect existing datasets and checkpoints; never delete or overwrite unless
  explicitly requested.
- If a change touches training behavior, note the impact in the PR summary.
