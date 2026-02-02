# CLIP-VG Setup and Inference Guide

## Environment Setup

- Use **Python 3.9**
- Recommended OS: **Ubuntu** or **WSL (Windows Subsystem for Linux)** if you are on Windows

Create and activate a virtual environment:

```bash
python3.9 -m venv clipvg_venv
source clipvg_venv/bin/activate
```

Install dependencies (CPU version):

```bash
pip install -r requirements_cpu.txt
```

Note:
- requirements.txt is incomplete
- It also appears to be intended for GPU usage

---

## Pretrained Weights

Download the pretrained model from Google Drive:

https://drive.google.com/file/d/14b-lc7zNniy4EEcJoBdXY9gNv2d20yxU/view

After downloading, place the files so the directory structure looks like this:

```txt
./pretrained/flickr/best_checkpoint.pth
./pretrained/gref/best_checkpoint.pth
```

Recommended datasets/checkpoints to use:
- unc
- referit

---

## Quick Test

### Test on Sample Images

Run the sample inference script:

```bash
python3 infer_clip_vg_samples.py
```

---

### Test on a Single Image

Example command for running inference on one image:

```bash
python3 infer_clip_vg.py \
  --checkpoint ./pretrained/unc/best_checkpoint.pth \
  --image test_infer_images/humanity_2.png \
  --text "girl swinging from a tree"
```

---

## Notes

This README assumes CPU-only inference and a correctly configured Python 3.9 environment.

This repo is a fork of https://github.com/linhuixiao/CLIP-VG.  
All credit goes to the original author.
