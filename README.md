# Test-Time Adaptation via MEMO with Wavelet Augmentation
Test-Time Adaptation via Marginal Entropy Minimization (MEMO) using frequency-domain Wavelet Decomposition on ImageNet-A.

# Test-Time Adaptation via MEMO with Wavelet Augmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep Learning, 2024 - UniTn

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

<p align="center">
  <img src="https://ar5iv.labs.arxiv.org/html/2110.09506/assets/fig/intro.png" width="500" alt="MEMO Pipeline Overview">
</p>

---

## Experiments & Results

We evaluated our approaches on **ImageNet-A**a highly challenging subset consisting of real-world natural adversarial examples that standard classifiers frequently fail on. Performance was measured using **Top-1 and Top-5 Accuracy (%)** restricted to the target ImageNet-A mask categories.

| Configuration | Augmentation Strategy | Wavelet Function | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :---: | :---: |
| **ResNet-50 Baseline** | None | N/A | 1.8533% | — |
| **Standard MEMO** | AugMix ($M=8$) | N/A | 5.1867% | — |
| **Test 1_s** | Wavelet $\rightarrow$ AugMix (Single-Level) | Cyclic Batch | 5.0933% | 30.7467% |
| **Test 1_m** | Wavelet $\rightarrow$ AugMix (Multilevel) | Cyclic Batch | 5.3867% | 30.3600% |
| **Test 2_s** | Wavelet $\rightarrow$ $M \times$ AugMix (Single-Level) | Random Uniform | 5.7867% | 30.1333% |
| **Test 2_m** | Wavelet $\rightarrow$ $M \times$ AugMix (Multilevel) | Random Uniform | **5.9067%** | **31.7067%** |

### Key Findings:
* **TTA Works:** Applying MEMO immediately provides a substantial boost over the baseline model.
* **Frequency Augmentation Advantage:** The highest performance (**5.91% Top-1**) was achieved using our **multilevel wavelet decomposition** coupled with multiple AugMix perturbations on the same decomposed variations (Test 2_m).
* **Level Proportionality:** We observed a direct correlation between the number of decomposition levels used and final accuracy.

---

## Repository Structure

```text
├── DL_Project_Colab_AWS.ipynb   # Complete implementation, execution, and logs
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
```

## Full Report
The report (a self-contained notebook with code and text) can be read [here](https://github.com/edofiore/tta-memo-with-wavelet-augmentation/blob/main/DL_Project_AWS.ipynb).
