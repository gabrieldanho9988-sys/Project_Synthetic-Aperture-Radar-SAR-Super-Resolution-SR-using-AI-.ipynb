# SAR Image Super-Resolution using Deep Residual Learning

### Project Overview
This project implements a Deep Learning pipeline to enhance the spatial resolution of **Synthetic Aperture Radar (SAR)** imagery. SAR data is notoriously difficult to process due to speckle noise and complex geometry. This project utilizes a **Very Deep Super Resolution (VDSR)** inspired architecture with custom **Residual Blocks** to reconstruct high-frequency spatial details from low-resolution inputs. The data is sourced from the Capella Space Open Data catalog (IEEE Data Contest collection) and retrieved dynamically via the STAC API.

---

### Technical Highlights
* **Custom Architecture:** A 10-layer Deep Residual Network featuring **Feature Scaling** (factor of 0.1) within residual blocks to stabilize deep gradient flow.
* **SAR-Physics Optimized Loss:** Replaced standard MSE with a weighted combination of **L1 Loss** (for sharp edges) and **SSIM (Structural Similarity)** to preserve the physical geometry of radar returns.
* **Dense Data Engineering:** Implemented high-density patch extraction with a 50% overlap (stride 32) to quadruple the training dataset while maintaining data integrity.
* **Fully Convolutional Inference:** The model supports **Full-Scene Inference**, allowing it to process entire SAR geographic expanses of any size natively without tiling artifacts.

---

### Performance Metrics
| Metric | Baseline (Bicubic) | Baseline (SRCNN) | **Optimized VDSR (Final)** |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~19.50 dB | 19.89 dB | **20.09 dB** |
| **SSIM** | 0.5800 | 0.6208 | **0.6635** |

---

### Project Structure (Notebook Phases)
* **Phase 1:** Environment Setup - Dependency installation and API connection.
* **Phase 2:** Metadata Pipeline - Parallel extraction of STAC metadata via multi-threading.
* **Phase 3:** Data Retrieval - Spatiotemporal filtering and raw TIFF normalization (DN to dB).
* **Phase 4:** Baseline Establishment - Implementation of a 3-layer SRCNN.
* **Phase 5:** Advanced Architecture - Development of the Residual VDSR model.
* **Phase 6:** Data Engineering - Dense overlapping patch extraction and SAR-safe augmentation analysis.
* **Phase 7:** Optimization - The "Goldilocks" training run using L1 + SSIM loss.
* **Phase 8:** Evaluation - Visual validation and full-scene AI enhancement.

---

### How to Run
1. Open the project notebook in **Google Colab**.
2. Ensure GPU acceleration is enabled (**Runtime** -> **Change runtime type** -> **T4 GPU**).
3. Follow the sequential cells to reproduce the training and evaluation results.

### Data Source
This project utilizes the [Capella Space Open Data Catalog](https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json).
* **Dataset:** IEEE Data Contest (GEO SAR Imagery)

---

**Author:** Gabriel Danho  
**Course:** Deep Learning  
**Instructor:**Professor. Amit Kumar Mishra
