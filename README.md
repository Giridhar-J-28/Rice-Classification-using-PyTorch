# Rice Type Classification using PyTorch

A simple neural network built in PyTorch to classify rice grains into one of two varieties (class 1 = Jasmine, class 0 = the other variety) using 10 measurements of each grain's shape.

**Test accuracy: 98.75%**

## Overview

This project covers a complete PyTorch workflow on tabular data:

- Loading and cleaning data with pandas
- Normalising features
- Splitting data into training, validation and test sets
- Writing a custom `Dataset` class and using `DataLoader` for batching
- Building a small feed-forward network with `nn.Module`
- Training with binary cross-entropy loss and the Adam optimizer
- Tracking and plotting loss and accuracy per epoch
- Evaluating on a held-out test set and predicting a new sample

The code automatically uses the best available hardware: an NVIDIA GPU (CUDA), an Apple Silicon GPU (MPS), or the CPU.

## Dataset

`riceClassification.csv` contains 18,185 rice grains (after removing missing values), each described by 10 features:

| Feature | Description |
|---|---|
| Area | Number of pixels inside the grain's outline |
| MajorAxisLength | Length of the longest line through the grain |
| MinorAxisLength | Length of the shortest line through the grain |
| Eccentricity | How elongated the grain's ellipse is |
| ConvexArea | Area of the smallest convex shape containing the grain |
| EquivDiameter | Diameter of a circle with the same area |
| Extent | Ratio of grain area to its bounding box area |
| Perimeter | Length of the grain's outline |
| Roundness | How close the grain is to a circle |
| AspectRation | Ratio of major axis to minor axis |

The classes are fairly balanced: 9,985 grains of class 1 (Jasmine) and 8,200 of class 0.

Source: [ADD DATASET LINK HERE]

## Approach

**Preprocessing:** Rows with missing values and the `id` column are removed. Each column is divided by its maximum absolute value so every feature falls between 0 and 1, which stops large features like `Area` from dominating small ones like `Eccentricity`.

**Data split:**

| Set | Share | Samples |
|---|---|---|
| Training | 70% | 12,729 |
| Validation | 15% | 2,728 |
| Test | 15% | 2,728 |

**Model:**

```
Input (10 features) -> Linear(10, 10) -> Linear(10, 1) -> Sigmoid -> probability
```

The model has 121 trainable parameters.

**Training setup:**

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy (`nn.BCELoss`) |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 32 |
| Epochs | 10 |

## Results

| Set | Accuracy |
|---|---|
| Training | ~98.7% |
| Validation | ~98.7% |
| Test | 98.75% |

Training and validation accuracy stay close throughout, which suggests the model is not overfitting. The test set was never used during training, so the test score is a fair estimate of performance on new data.

## How to run

```bash
git clone https://github.com/Giridhar-J-28/Rice-Classification-using-PyTorch.git
cd Rice-Classification-using-PyTorch
pip install torch torchinfo scikit-learn pandas