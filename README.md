
#  LRP-Based Interpretability for ASTGCN Models

This repository provides an implementation of **Layer-wise Relevance Propagation (LRP)** for interpreting predictions made by **ASTGCN (Attention-based Spatio-Temporal Graph Convolutional Network)** models, specifically applied to spatiotemporal forecasting tasks such as **traffic flow** or **air pollution prediction**.

##  Overview

Understanding deep learning models' internal decision processes is critical, especially in high-stakes domains like urban traffic forecasting or air quality analysis. This repository focuses on:

- Explaining the **importance of each input feature and node** in the model’s prediction.
- Leveraging **gradient-based LRP** to estimate contribution scores.
- Providing **visual insights** via bar plots for top contributing features.

## ⚙ Methodology: Layer-wise Relevance Propagation (LRP)

LRP explains predictions by redistributing the output prediction score backward to input features proportionally. For a given model output `y`, LRP uses:


Where:
- `x`: the input features,
- `∂y/∂x`: gradients of the output with respect to the input,
- `R`: the relevance matrix indicating each input's contribution.

### Why LRP?

Unlike plain gradients, LRP considers both **input strength and gradient sensitivity**, making it **more robust to noisy gradients** and more **interpretable**.

## 📈 What This Code Does

- Accepts a trained ASTGCN model and a mini-batch of input data.
- Performs a forward pass and calculates the relevance matrix `R`.
- Extracts **feature-wise** and **node-wise** relevance by averaging over the temporal dimension.
- Filters out specific features (e.g., `"PM2.5"`) to highlight contextual influences.
- Generates and saves a high-resolution bar plot for interpretability.

## 🧪 Input Format

The model expects a `batch` dictionary with:
```python
{
    'features': torch.Tensor of shape (B, T, N, F),
    ...
}
