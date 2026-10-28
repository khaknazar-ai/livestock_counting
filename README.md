# Livestock Counting

A research and local demo app for livestock counting with CSRNet and MSA-CSRNet density models. It includes image inference, training and data-preparation scripts, and a Flask web interface. This repository does not include datasets or trained checkpoints.

## Run

The pinned dependencies target Windows, Python 3.8, and CUDA 11.3. `requirements.txt` is not a CPU-only install recipe; YOLO support is optional in `requirements-yolo.txt`. Install YOLO extras only when needed; if Ultralytics is installed, app startup tries to load `yolov8n.pt` and may download it when absent.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run inference with an authorized local checkpoint:

```powershell
python inference.py --image path\to\image.jpg --config configs\config.yaml --checkpoint path\to\checkpoint.pth --no_save
```

Start the local web app:

```powershell
python webapp\app.py --config configs\config.yaml --checkpoint path\to\checkpoint.pth
```

The UI can start without a CSRNet checkpoint, but counting is unavailable until a valid checkpoint is supplied.

For local use, set `webapp.host` to `127.0.0.1` in `configs/config.yaml`. The app seeds demo accounts and uses a random session key per run unless LIVESTOCK_SECRET_KEY or a config key is set. Keep the demo accounts for local use only; do not expose the app publicly.

For training, place images and point annotations under `data/livestock/images/` and `data/livestock/annotations/`, then run:

```powershell
python tools\generate_density_map_fast.py --config configs\config.yaml
python train.py --model csrnet --config configs\config.yaml
python train.py --model msa_csrnet --config configs\config_msa.yaml
```

Configuration files and data-conversion commands are in `configs/` and `tools/`. Training writes checkpoints and logs to `experiments/logs/`.

## Results

A saved comparison on the configured test split reports these results (736 images per model):

| Model | N | MAE (count) | RMSE (count) | Mean relative error |
| --- | ---: | ---: | ---: | ---: |
| CSRNet | 736 | 2.4739 | 4.2215 | 8.17% |
| MSA-CSRNet | 736 | 2.6676 | 5.4162 | 10.21% |
| YOLOv8n-ZS | 736 | 36.6617 | 50.7921 | 91.05% |
| YOLOv8n-FT | 736 | 2.0326 | 4.8920 | 4.65% |

Mean relative error is the average per-image absolute count error divided by max(ground-truth count, 1). Source: the local experiments/logs/comparison_test_summary.csv. The CSV, comparison script, dataset, and checkpoints are not included; these saved results are not reproducible from this public repository alone.

## Reuse

No project-level LICENSE is included. Dataset attribution and license notes are retained in baseline/contagem/ and baseline/sheep6tost/; check those notices and third-party terms before reuse.

Dependencies: [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Pillow](https://python-pillow.org/), [Flask](https://flask.palletsprojects.com/), [Werkzeug](https://werkzeug.palletsprojects.com/), [bcrypt](https://pypi.org/project/bcrypt/), [openpyxl](https://openpyxl.readthedocs.io/), [Matplotlib](https://matplotlib.org/), [TensorBoard](https://www.tensorflow.org/tensorboard), [PyYAML](https://pyyaml.org/), [tqdm](https://tqdm.github.io/), and optional [Ultralytics](https://github.com/ultralytics/ultralytics). Exact versions are in the requirements files.
