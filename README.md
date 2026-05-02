# Leveraging Mono-Modal Data to Generate Multimodal Representations for Industrial Anomaly Detection (Accepted in COMPSAC 2026)

This repository extends the **MVTec Anomaly Detection (MVTec-AD)** dataset by generating multiple modalities from the original RGB images using a fully automated pipeline (no human refinement).  
The goal is to evaluate whether **synthetically generated modalities** can improve **industrial anomaly detection** performance.

The full methodology and experiments are described in the included preprint manuscript.

---

## Overview

Starting from the original RGB images of MVTec-AD, we generate additional modalities:

- **Captions (Text modality)**
- **Thermal Infrared (TIR) images**
- **Depth Maps**
- **WAV audio signals (raw waveform)**
- **Spectrograms (STFT and CQT)**

Each modality is evaluated independently using supervised **binary classification**.

---

## Modalities

### 1. Original RGB (Preprocessed)
- All images are resized/normalized to **700×700**
- Zero-padding is applied to preserve aspect ratio
- Unified naming convention is applied

---

### 2. Captions (Text)
Captions are generated using:
- **BLIP** (captioning model)
- **CLIP + GPT-2** (feature extraction + caption generation)

Captions are stored in CSV format.

---

### 3. Thermal Infrared (TIR)
Thermal images are generated from RGB using a pretrained **ThermalGAN** model.

---

### 4. Depth Maps
Depth maps are generated using the pretrained **MiDaS** monocular depth estimation model.

---

### 5. WAV Audio Signals
Each RGB image is converted into a **mono-channel WAV signal** using a deterministic pixel-to-waveform transformation:
- Sample rate: **44,100 Hz**
- Output length: ~11 seconds

---

### 6. Spectrograms
Two spectrogram modalities are generated from the WAV signals using Librosa:
- **STFT spectrograms**
- **CQT spectrograms**

---

## Dataset Structure

The dataset is reorganized for supervised learning:
- Split into **80% training**, **10% validation**, **10% testing**
- A CSV file is generated containing:
  - filename
  - binary label (0 = good, 1 = defective)

---

## Experimental Setup

Each modality is evaluated independently:

### Image-based modalities (RGB / TIR / Depth / Spectrograms)
Trained using top pretrained models from `timm`, including:
- ViT-B-p16-384
- EfficientNet-B7 variants
- ResNeXt-101 variants

### Text modality (Captions)
Evaluated using transformer-based models such as:
- BERT
- DistilBERT
- RoBERTa

### Audio modality (WAV)
Evaluated using raw waveform classifiers:
- RawNet
- RawNet2

---

## Results Summary

Key observations from experiments:
- EfficientNet variants perform strongly across most image-based modalities
- Thermal IR and spectrogram modalities can outperform original RGB in some cases
- Caption-based classification achieves competitive performance
- Raw waveform classification remains challenging compared to spectrogram-based representations

---

## Manuscript

The full explanation of preprocessing, modality generation, models, and results is available in:

📄 `preprint_manuscript.pdf`
