# ICBHI Respiratory Sound Segmentation and Classification

This project presents a two-stage deep learning system for automatic respiratory sound analysis using the **Respiratory Sound Database (ICBHI 2017)**.

The system performs two main tasks:

1. **Respiratory cycle boundary detection** in continuous audio recordings using a CRNN model.
2. **Classification of extracted respiratory cycles** into four classes using an Audio Spectrogram Transformer (AST).

The project was developed as part of a master's thesis in Artificial Intelligence.

---

## Dataset

The project uses the **Respiratory Sound Database (ICBHI 2017)**, which contains respiratory sound recordings together with temporal annotations.

The classification task includes four classes:

- `Normal` – normal respiratory sound
- `Crackles` – crackles
- `Wheezes` – wheezes
- `Both` – simultaneous presence of crackles and wheezes

The dataset is split at the patient level to prevent recordings from the same patient from appearing in both the training and evaluation sets.

---

## System Architecture

The proposed system consists of two separate but connected modules.

### 1. Respiratory Cycle Segmentation – CRNN

Respiratory cycle segmentation is performed using a **Convolutional Recurrent Neural Network (CRNN)**.

The architecture contains:

- three convolutional blocks,
- Batch Normalization,
- ReLU activation,
- asymmetric Max Pooling `(2, 1)`,
- a two-layer bidirectional GRU,
- a linear output layer for respiratory cycle boundary prediction.

Main input parameters:

- sampling rate: `4000 Hz`
- number of Mel bands: `64`
- hop length: `100`
- maximum recording length: `20 s`

The segmentation problem is formulated as **respiratory cycle boundary detection**.

---

### 2. Respiratory Sound Classification – AST

The classification module is based on the **Audio Spectrogram Transformer (AST)** architecture.

Pretrained model:

`MIT/ast-finetuned-audioset-10-10-0.4593`

The pretrained AST backbone is adapted to the four-class respiratory sound classification task using a custom classification head.

The classification head consists of:

```text
LayerNorm(768)
Dropout(0.3)
Linear(768 -> 512)
GELU
Dropout(0.3)
Linear(512 -> 128)
GELU
Dropout(0.2)
Linear(128 -> 4)
```

The AST model operates on audio resampled to `16000 Hz`.

The maximum segment length is `5 s`.

---

## Data Preprocessing

The preprocessing pipeline includes:

- resampling,
- Butterworth high-pass filtering,
- Mel-spectrogram generation,
- logarithmic conversion to the decibel scale,
- standardization for the CRNN input,
- `ASTFeatureExtractor` for AST input preparation.

A 4th-order Butterworth high-pass filter with a cutoff frequency of `50 Hz` is used to reduce low-frequency interference.

---

## Data Augmentation

Several augmentation techniques are applied during training:

- Pitch Shifting
- Time Stretching
- Gaussian Noise
- Random Gain
- SpecAugment
- Mixup

Mixup is applied specifically to the least represented `Both` class.

---

## Class Imbalance Handling

The ICBHI dataset is strongly imbalanced, therefore several balancing mechanisms are used:

- `WeightedRandomSampler`
- weighted `CrossEntropyLoss`
- additional weighting of the `Both` class
- Mixup augmentation for the `Both` class

Class weights are calculated inversely proportional to class frequency.

---

## Training

### CRNN

The segmentation model is trained for **30 epochs**.

- Optimizer: `Adam`
- Learning rate: `0.001`
- Loss function: `BCEWithLogitsLoss`

Additional training techniques include:

- gradient clipping,
- Cosine Annealing learning rate scheduling.

### AST

The classification model is trained for **20 epochs**.

- Optimizer: `AdamW`
- Base learning rate: `3e-4`

A lower learning rate is used for the pretrained AST backbone than for the newly initialized classification head.

The training procedure also includes:

- learning rate warmup,
- cosine decay,
- gradient clipping,
- model checkpointing based on ICBHI Score.

---

## Evaluation Results

### Segmentation Model – CRNN

| Metric | Result |
|---|---:|
| Precision | 67.82% |
| Recall | 69.81% |
| F1-Score | 68.80% |
| MAE | 0.112 s |

### Classification Model – AST

| Metric | Result |
|---|---:|
| Sensitivity | 49.28% |
| Specificity | 71.62% |
| ICBHI Score | 60.45% |

The best classification model is selected based on the **ICBHI Score**.

---

## Technologies

The project uses:

- Python
- PyTorch
- Hugging Face Transformers
- Librosa
- SciPy
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- CUDA

Experiments can be run in environments such as:

- Kaggle
- Google Colab

---

## Installation

Install the required dependencies with:

```bash
pip install numpy scipy librosa matplotlib seaborn scikit-learn torch transformers
```

---

## Model Files

During training, the following model files are generated:

```text
best_crnn_model.pth
best_ast_model.pth
```

Model checkpoints may be excluded from the repository due to their file size.

---

## Project Purpose

The project was developed as part of a master's thesis focused on automatic respiratory cycle segmentation and respiratory sound classification using deep learning methods.

The main goal was to build a complete prototype capable of processing continuous respiratory sound recordings by combining a CRNN-based segmentation model with an AST-based classification model.

---

## Author

Kamil Dwornik
