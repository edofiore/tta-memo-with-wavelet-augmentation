# Test-Time Adaptation via MEMO with Wavelet Augmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep Learning, 2024 - UniTn

This repository contains the final project developed for the **Deep Learning (2024)** course as part of the Master's degree curriculum at the **University of Trento (UniTn)**.

This project was conducted under the guidance of **Dr. Elisa Ricci** and **Dr. Francesco Tonini**.

Authors:
* [**Stefano Bertolasi**](https://github.com/stefanobertolasi)
* [**Edoardo Fiorentino**](https://github.com/edofiore)

## Project Overview

This repository implements and evaluates **Marginal Entropy Minimization with One Test Point (MEMO)**, an advanced Test-Time Adaptation (TTA) technique designed to mitigate domain shift during inference. Additionally, we introduce **W-MEMO**, an innovative approach that enhances the standard MEMO framework by utilizing **Discrete Wavelet Decomposition** to perform data augmentation in the frequency domain.

Test-Time Adaptation (TTA) dynamically fine-tunes a model during the inference stage using the specific data it is currently processing. This is crucial for real-world AI applications where test samples come from a distribution that deviates significantly from the training data (domain shift).

### Core Approaches Explored:
1. **Baseline Setup:** ResNet-50 pre-trained on `IMAGENET1K_V1` evaluated on the adversarial **ImageNet-A** dataset.
2. **Standard MEMO:** Sampling $M$ random augmentations per test instance via AugMix, computing the marginal entropy across the augmented batch, and performing a single backpropagation step to dynamically update parameters.
3. **Wavelet-Enhanced MEMO (W-MEMO):** Decomposing the image into approximation ($cA$) and detail ($cH, cV, cD$) frequency sub-bands. We manipulate these coefficients to create distinct structural variations of the image before passing them into the TTA pipeline.

---
 
## Test-Time Adaptation (TTA) & MEMO

The Test-Time Adaptation (TTA) framework redefines the classic deployment paradigm: instead of treating a neural network as a static block of frozen weights, the model becomes a dynamically adaptive system during the inference stage. **MEMO** (Marginal Entropy Minimization with One test point) solves this challenge without requiring ground-truth labels (unsupervised) and operates on **a single test point at a time** ($x_0$), completely eliminating any dependency on large batches or sequential data streams.

<p align="center">
  <img src="https://ar5iv.labs.arxiv.org/html/2110.09506/assets/fig/intro.png" alt="Standard MEMO Framework" width="80%" style="max-width: 780px; border-radius: 8px;">
</p>

### The Mathematical Optimization Loop of MEMO:

1. **Batching Perturbations:** Given a single test input $x_0$, we apply a set of $M$ stochastic transformations sampled from an augmentation family $\mathcal{A}$ (configured via *AugMix* in our setup), generating a synthetic batch of data variants $[\mathbf{x}_1, \dots, \mathbf{x}_M]$.
2. **Marginal Average Distribution:** We compute the output logits of the network $f_\theta$ for each variant and extract the marginal average probability distribution across the batch via Softmax:
```math
\overline{p}_\theta(c|\mathbf{x}) = \frac{1}{M}\sum_{i = 1}^M p_\theta(c|\mathbf{x}_i)
```
3. **Entropy Minimization:** To force the network to produce a confident, invariant prediction regardless of the perturbation applied, we optimize the weights by minimizing the **Marginal Entropy**:
```math
\mathcal{H}(\mathbf{X}) = -\sum_{x \in \mathcal{X}} \mathcal{p}(\mathbf{x}) \log \mathcal{p}(\mathbf{x})
```
where $\mathbf{p(x)}$ represents the probability of each discrete outcome $\mathbf{x}$, and the sum is taken over all possible outcomes in the sample space $\mathbf{X}$.

We compute a single backpropagation step to update the model parameters $\theta \rightarrow \theta^\prime$, and then output the final prediction using the adapted weights.

> **Weight Resetting:** To prevent the severe issue of *parameter drift*, where the network overfits to a specific sample and degrades on subsequent ones, our pipeline clones the initial model state and **restores the default parameters after processing each test batch**, ensuring absolute isolation between separate inferences.

---

## Wavelet Decomposition with MEMO (W-MEMO)

While traditional TTA methods apply spatial or chromatic perturbations (e.g., rotations, color jitter), **W-MEMO** introduces variations directly inside the **frequency domain**, modifying the underlying geometric structures of the image.

Using the **2D Discrete Wavelet Transform (DWT)**, an image is mathematically decomposed into four distinct sub-bands:
* **$cA$ (Approximation Coefficients):** Captures the low-frequency components, representing the global context and macro-structures of the image.
* **$cH, cV, cD$ (Detail Coefficients):** Captures the high-frequency components, isolating Horizontal, Vertical, and Diagonal edge patterns, textures, and fine details, respectively.

### Frequency Coefficient Tuning & Augmentation:

Instead of distorting the pixel space uniformly, we apply targeted, stochastic mutations directly to the high-frequency detail arrays before reconstructing the image via the Inverse Discrete Wavelet Transform (IDWT):
* **Horizontal Detail ($cH$):** Injects Normally distributed noise $\mathcal{N}(0, \sigma^2)$ to simulate sensor grain or structural shifting.
* **Vertical Detail ($cV$):** Applies a scalar multiplier to dynamically enhance or suppress prominent vertical edge responses.
* **Diagonal Detail ($cD$):** Uses a binomial random mask (frequency dropout) to selectively strip away noisy high-frequency sub-components.

---

## Experiments & Results

We evaluated our approaches on **ImageNet-A** a highly challenging subset consisting of real-world natural adversarial examples that standard classifiers frequently fail on. Performance was measured using **Top-1 and Top-5 Accuracy (%)** restricted to the target ImageNet-A mask categories.

| Configuration | Augmentation Strategy | Wavelet Function | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :---: | :---: |
| **ResNet-50 Baseline** | None | N/A | 1.8533% | 17.5333% |
| **Standard MEMO** | AugMix ($M=8$) | N/A | 5.1867% | 30.5333% |
| **Test 1_s** | Wavelet $\rightarrow$ AugMix (Single-Level) | Cyclic Batch | 5.0933% | 30.7467% |
| **Test 1_m** | Wavelet $\rightarrow$ AugMix (Multilevel) | Cyclic Batch | 5.3867% | 30.3600% |
| **Test 2_s** | Wavelet $\rightarrow$ $M \times$ AugMix (Single-Level) | Random Uniform | 5.7867% | 30.1333% |
| **Test 2_m** | Wavelet $\rightarrow$ $M \times$ AugMix (Multilevel) | Random Uniform | **5.9067%** | **31.7067%** |

### Key Findings:
* **TTA Works:** Applying MEMO immediately provides a substantial boost over the baseline model.
* **Frequency Augmentation Advantage:** The highest performance (**5.91% Top-1**) was achieved using our **multilevel wavelet decomposition** coupled with multiple AugMix perturbations on the same decomposed variations (Test 2_m).
* **Level Proportionality:** We observed a direct correlation between the number of decomposition levels used and final accuracy.

---

## Requirements & Dependencies

Ensure you have the following core modules installed before processing the notebook:
* `torch` / `torchvision` (Deep learning framework & models)
* `boto3` (AWS SDK for Python used to stream dataset from S3)
* `PyWavelets` (`pywt` package for frequency decomposition)
* `Pillow` (`PIL` module for processing image file bytes)
* `matplotlib` (For plotting sub-bands and generating charts)
* `pandas` (For structured data and logging summaries)
* `numpy` (For array operations and coefficient manipulation)
* `tqdm` (Progress bar tracking during adaptation loops)

---

## Environment & Portability Note

This notebook is specifically configured out-of-the-box to run on **Amazon Web Services (AWS)** using an **Amazon S3** bucket to stream the ImageNet-A dataset via `boto3`. 

If you wish to run this project in a different environment (such as a local machine, a local cluster, or Google Colab):
* You will need to download the **ImageNet-A** dataset manually.
* You must replace the custom `S3ImageFolder` dataset class with PyTorch's standard `torchvision.datasets.ImageFolder` pointed to your local directory path.

---

## Future Enhancements

* **Code Modularization:** While this project is currently self-contained within a unified Jupyter Notebook for experimental visibility, a primary next step is to refactor the codebase into a structured Python package. We plan to move the data loading, wavelet processing, and adaptation logic into standalone Python files (`.py`) featuring a clean command-line interface.

---

## References
1. Wang, H., Ge, S., Xing, E. P., & Lipton, Z. C. (2021). [MEMO: Test Time Robustness via Adaptation and Augmentation](https://arxiv.org/pdf/2110.09506)
2. Y Shimizu, Z Zhang, R Batres. (2007) [The Wavelet Transform in Signal and Image Processing](https://link.springer.com/chapter/10.1007/978-1-84628-955-2_5#citeas)
3. Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan (2020) [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781)
4. Dan Hendrycks et al. (CVPR 2021) [ImageNet-A Dataset: Natural Adversarial Examples](https://arxiv.org/abs/1907.07174)

---

## Full Report
The report (a self-contained notebook with code and text) can be read [here](https://github.com/edofiore/tta-memo-with-wavelet-augmentation/blob/main/DL_Project_AWS.ipynb).
