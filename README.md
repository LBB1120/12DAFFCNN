# README

This repository contains the implementation of a dual-branch fusion model, integrating 1D and 2D convolutional branches with an Attention Feature Fusion Module (AFE). The model is designed for tasks requiring the combination of spectral and spatial features.

## Repository Structure

```
├── data_utils
│   ├── data_preprocessing.py  # Functions for dataset loading and preprocessing
│   ├── dataset.py             # Dataset utilities for training and prediction
├── models
│   ├── conv1d_branch.py       # 1D convolutional branch
│   ├── conv2d_branch.py       # 2D convolutional branch
│   ├── dual_branch_model.py   # Dual-branch model integrating Conv1D and Conv2D
├── training
│   ├── train.py               # Training and validation logic
├── prediction
│   ├── predict.py             # Inference logic and result saving
├── main.py                    # Entry point to train and predict
├── requirements.txt           # List of dependencies
└── README.md                  # Project overview and setup instructions
```

## Features

- **1D and 2D Convolutional Networks**: Two branches for processing 1D and 2D data respectively.
- **Attention Feature Fusion Module (AFE)**: Combines spatial and spectral features with attention mechanisms.
- **Pretrained Weights Support**: Load pretrained weights for both branches.
- **Balanced Dataset Support**: Includes functions to create balanced datasets for training and testing.
- **Prediction Pipeline**: Batch prediction support with results saved to an Excel file.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/dual-branch-fusion-model.git
   cd dual-branch-fusion-model
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your dataset and pretrained weights are correctly organized:

   - Dataset: Adjust file paths in `training/train.py` and `prediction/predict.py`.
   - Pretrained weights: Place the weights in the specified locations.

## Usage

### Training

To train the model, run the following command:

```bash
python main.py
```

Training logs and the model's weights will be saved in the specified paths in the script.

### Prediction

To generate predictions:

```bash
python main.py
```

This will save the prediction probabilities to an Excel file.

## File Descriptions

### `data_utils/data_preprocessing.py`

Provides utility functions for loading datasets, creating balanced test sets, and preprocessing data for 1D and 2D branches.

### `data_utils/dataset.py`

Contains dataset classes:

- `PredictionDataset`: Dataset class for inference.

### `models/conv1d_branch.py`

Implements the 1D convolutional branch for spectral feature extraction.

### `models/conv2d_branch.py`

Implements the 2D convolutional branch for spatial feature extraction.

### `models/dual_branch_model.py`

Defines the full dual-branch model with the AFE module.

### `training/train.py`

Handles model training, including:

- Loading datasets.
- Applying pretrained weights.
- Using weighted loss functions to handle class imbalance.

### `prediction/predict.py`

Manages inference:

- Loads pretrained model weights.
- Processes input data and generates predictions.
- Saves results to an Excel file.

### `main.py`

Entry point to train and predict:

- Calls `train_dual_branch_model` for training.
- Calls `predict_and_save_results` for inference.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenPyXL

Install dependencies with:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you find bugs or have suggestions.

## Acknowledgments

This work leverages pretrained weights for both 1D and 2D convolutional networks. Thanks to all contributors and open-source projects that made this work possible.
