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

## Experiments & Results

We evaluated our approaches on **ImageNet-A**a highly challenging subset consisting of real-world natural adversarial examples that standard classifiers frequently fail on. Performance was measured using **Top-1 and Top-5 Accuracy (%)** restricted to the target ImageNet-A mask categories.

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

## References
1. Wang, H., Ge, S., Xing, E. P., & Lipton, Z. C. (2021). [MEMO: Test Time Robustness via Adaptation and Augmentation](https://arxiv.org/pdf/2110.09506)
2. Y Shimizu, Z Zhang, R Batres. (2007) [The Wavelet Transform in Signal and Image Processing](https://link.springer.com/chapter/10.1007/978-1-84628-955-2_5#citeas)
3. Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan (2020) [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781)

---

## Full Report
The report (a self-contained notebook with code and text) can be read [here](https://github.com/edofiore/tta-memo-with-wavelet-augmentation/blob/main/DL_Project_AWS.ipynb).
