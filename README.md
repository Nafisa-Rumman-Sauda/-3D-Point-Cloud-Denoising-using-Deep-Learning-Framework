# Point Cloud Denoising using Deep Learning

This repository contains the code and models for point cloud denoising using advanced deep learning techniques. The goal of this project is to provide a robust solution for denoising 3D point cloud data, improving its quality for various applications like autonomous driving, robotic object recognition, and geometric analysis. This method leverages cutting-edge deep learning methods such as **normalizing flows** and **noise disassociation** to clean noisy point cloud data while preserving critical geometric and topological features.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Repository Structure](#repository-structure)
- [References](#references)

## Introduction

Point clouds are 3D data points representing objects or environments, often captured by devices such as LiDAR sensors or depth cameras. However, these data points are inherently noisy due to sensor limitations and environmental factors. This project presents a deep learning-based model for denoising point clouds that utilizes normalizing flows and noise disassociation techniques.

The model separates latent point representations from noise and uses normalizing flows to map these latent features to the Euclidean plane, resulting in a clean and high-quality point cloud.

### Key Features

- **Deep Learning Framework:** Utilizes deep learning techniques to handle different types of noise in point cloud data.
- **Noise Types Supported:** Supports various noise types such as isotropic Gaussian, non-isotropic Gaussian, uniform, and simulated LiDAR noise.
- **State-of-the-art Performance:** Achieves superior denoising results compared to existing methods.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- PyTorch (>=1.7)
- NumPy
- SciPy
- Matplotlib
- Open3D (for visualizing point clouds)
- TensorFlow (if using TensorFlow-based models)

To install these dependencies, you can create a virtual environment and run:

```bash
pip install -r requirements.txt
```

You can download the `requirements.txt` file from the repository.

---

## Usage

### 1. Prepare Data

Point cloud data should be in PLY, XYZ, or similar formats. Example datasets provided include Paris-CARLA-3D, Paris-rue-Madame, and 3DCSR. Place your point cloud data in the `/dataset` directory.

### 2. Denoising Procedure

Run the following command to denoise a point cloud:

```bash
python denoise.py --input /path/to/input/cloud.ply --output /path/to/output/cloud_denoised.ply
```

This script loads a noisy point cloud and applies the trained denoising model, saving the cleaned output to the specified file.

### 3. Visualize the Result

You can visualize the original and denoised point clouds using the `visualize.py` script:

```bash
python visualize.py --input /path/to/input/cloud.ply --denoised /path/to/output/cloud_denoised.ply
```

This will open a window displaying both the original and denoised point clouds for comparison.

---

## Training

To train the denoising model on your own dataset, follow these steps:

1. **Prepare your dataset**: Ensure the point cloud data is in the correct format (e.g., PLY, XYZ).
2. **Configure training parameters**: Edit the `config.py` file to set the parameters for the training, such as the learning rate, batch size, and number of epochs.
3. **Run training**: Start the training process by running the following command:

```bash
python train.py --train_data /path/to/train_data --valid_data /path/to/valid_data
```

The training script will save the model checkpoints during the training process, and you can resume training if needed.

---

## Evaluation

Once the model is trained, evaluate its performance on a test set:

```bash
python evaluate.py --test_data /path/to/test_data --model /path/to/trained_model
```

This will output quantitative metrics such as CD (Chamfer Distance), P2M (Point-to-Model), and HD (Hausdorff Distance), which assess the quality of the denoised point cloud.

### Metrics

- **Chamfer Distance (CD)**: Measures the average distance between points in the denoised cloud and the clean ground truth.
- **Point-to-Model (P2M)**: Measures the quality of the denoised points relative to the underlying model.
- **Hausdorff Distance (HD)**: Measures the maximum distance between points in the denoised cloud and the ground truth.

---

## Results

### Performance Metrics

This section presents the performance of our method compared to existing denoising techniques across various datasets and noise levels.

- **Table 4.4**: Comparison of deep-learning models trained on different types of noise distributions.
- **Table 4.6**: Performance under simulated LiDAR noise.
- **Visual Results**: Refer to Figures 4.6-4.10 for visual comparisons of denoised patches from different methods.

Our method consistently outperforms traditional methods, achieving lower error rates and preserving fine details in the point clouds.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Repository Structure

Here's a brief overview of the repository structure:

- **`dataset/`**: Contains the datasets used for training, evaluation, and testing.
- **`eval/`**: Contains code for evaluating model performance and generating metrics.
- **`metric/`**: Includes scripts for calculating evaluation metrics such as Chamfer Distance, P2M, and Hausdorff Distance.
- **`models/`**: Contains the deep learning models used for denoising point clouds.
- **`modules/`**: Includes utility modules and functions used across various scripts in the project.
- **`pretrain/`**: Contains pre-trained models for use in evaluation or fine-tuning.

---

## References

1. DMR Denoising [35]
2. ScoreDenoise [36]
3. PCNet [45]
4. 3DCSR Dataset [64]
5. Paris-CARLA-3D Dataset [65]
6. LiDAR Simulation Methods [8]

For additional references, please see the full paper or documentation in the repository.

---

By following the steps outlined above, you can easily train, evaluate, and deploy the point cloud denoising model on your data. Feel free to modify the code and share your results!

---
