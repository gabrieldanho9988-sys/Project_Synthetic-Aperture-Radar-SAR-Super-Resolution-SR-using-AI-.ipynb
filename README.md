# SAR Super Resolution using AI

## Project Description
The goal of this project is to enhance the spatial resolution of Synthetic Aperture Radar (SAR) imagery using Deep Learning techniques. SAR images are notoriously difficult to process due to inherent speckle noise and complex geometric properties (like layover and shadowing). This project implements Super-Resolution models to reconstruct high-frequency spatial details from low-resolution SAR inputs, aiming to improve both visual quality and subsequent image analysis capabilities

## Dataset
The data is sourced from the Capella Space Open Data catalog, specifically the IEEE Data Contest collection.

We utilize high-resolution GEO (Geographically Geocoded) SAR imagery in TIFF format.

The data is retrieved dynamically via the SpatioTemporal Asset Catalog (STAC) API using pystac and stac-asset.

## Method
The project utilizes a "Spiral Approach" to algorithm design, starting with a baseline model and iterating to a deeper architecture:

SRCNN (Super-Resolution Convolutional Neural Network): Used as an initial baseline to establish proof-of-concept for upscaling SAR patches.

VDSR (Very Deep Super Resolution): The primary model used in the final iterations. VDSR uses a deep 9-layer convolutional architecture with residual learning. Instead of predicting the entire high-resolution image directly, the model learns to predict the "residual" (the missing high-frequency sharp details) and adds it back to a bicubic-upscaled low-resolution input.

## Data Preprocessing
SAR data requires careful preprocessing before feeding it into a neural network to prevent vanishing gradients or uniform blank outputs:Radiometric Calibration: Raw Digital Numbers (DN) are scaled using a calibration factor.Logarithmic Transformation: Scaled values are converted to decibels (dB) using 20.log_10(scaled_DN) to handle the massive dynamic range of radar backscatter.Normalization: The image is clipped using the 2nd and 98th percentiles to remove extreme outliers, then min-max normalized to a range of (0 , 1).Patching: To manage RAM and increase the dataset size, the large SAR image is sliced into overlapping 128.128 patches.LR/HR Pairing: The High-Resolution (HR) ground truth patches are downsampled using bicubic interpolation (scale factor 2) to create the Low-Resolution (LR) inputs.

## Training
The model was trained on an 80/20 Train-Validation split using the following parameters:

Optimizer: Adam (Initial Learning Rate: 0.001)

Loss Function: Mean Absolute Error (L1 Loss) was chosen over MSE to encourage sharper edge reconstruction and reduce the blurring effect common in MSE-trained super-resolution models.

Callbacks: ReduceLROnPlateau (factor 0.5, patience 5) to dynamically refine the learning rate when validation loss stagnates, and EarlyStopping (patience 15) to prevent overfitting.

Batch Size: 16

Epochs: Up to 40-50 epochs.

## Results
The performance of the models was evaluated using two Acceptance Test Procedures (ATPs): PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure).

Baseline (Single Patch): PSNR: 23.79 dB | SSIM: 0.8427

Improved VDSR (Dataset Patching & Splitting): PSNR: 24.51 dB | SSIM: 0.8643

(Note: Data augmentation using rotations was tested but ultimately degraded performance due to the directional nature of SAR shadows. The un-augmented patched dataset provided the best metrics).

## Run the Code
This project was developed and executed using Google Colab.

Open the .ipynb notebook in Google Colab.

Crucial: Go to Runtime -> Change runtime type and select T4 GPU (or any available GPU). The model will train incredibly slowly on a CPU.

Run the first cell to install the required dependencies (pystac, stac-asset, rasterio, etc.).

Run the cells sequentially to download the Capella STAC data, preprocess the images, define the model, and begin training.

Repository
https://github.com/gabrieldanho9988-sys/SAR-Super-Resolution-Using-AI.git


## Repository
Link to the notebook.
