# Diffusion Image Generation Project

This project uses `diffusers`, `torch`, and `ollama` to generate images using diffusion pipelines.  

---

## Prerequisites

- Windows 10/11
- Python 3.10 (or compatible version)
- GPU with CUDA 12.8 (optional, for faster PyTorch computations)

---

## Setup

1. Clone or download the project folder.
2. Ensure `setup_env.bat` and `requirements.txt` are in the project root folder.
3. Open **Command Prompt** in the project folder.
4. Run the setup script:

```bat
setup_env.bat
````

This script will:

* Create a virtual environment named `venv`
* Activate the virtual environment
* Upgrade `pip`
* Install all required dependencies including PyTorch with CUDA 12.8 support

---

## Usage

1. Activate the virtual environment manually (if not already active):

```bat
call venv\Scripts\activate
```

2. Run your Python scripts, e.g.:

```bat
python main.py
```

> Replace `main.py` with the script where you call your `generate_image` or other pipeline functions.

---

## Installed Packages

* `torch` & `torchvision` — PyTorch with CUDA support
* `diffusers` — Diffusion pipelines
* `Pillow` — Image handling
* `ollama` — Async client for model interaction

---

## Notes

* Make sure your Python version matches the PyTorch wheel (e.g., Python 3.10 → cp310).
* `diffusion_pipeline` is a local module; ensure it’s in the project folder.
* CUDA is optional but recommended for GPU acceleration.

---


