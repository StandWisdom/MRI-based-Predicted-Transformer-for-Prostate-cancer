# MRI-based-Predicted-Transformer-for-Prostate-cancer
An MRI-pathology model (MRI-based Predicted Transformer for Prostate cancer (MRI-PTPCa)) was proposed to discover correlations between mp-MRI and tumor regressiveness of PCa and was further deployed for diagnosing non-PCa, PCa, non-CSPCa, CSPCa, and grading of GGG.

The goal of this repository is:
- to help researchers to reproduce the MRI-PTPCa  and expand for other prostate research or relevant research.
- to help researchers to build a end-to-end AI model alone to predicting pathological prostate tumour aggressiveness for assisted non-invasive assessment.
- to provide pre-trained foundation model of prostate mp-MRI for migration of downstream tasks in prostate cancer.

## Installation
1. [python3 with anaconda](https://www.continuum.io/downloads)
2. [pytorch with/out CUDA](http://pytorch.org)

Using requirements.txt
- Install the required Python packages using pip:
`pip install -r requirements.txt`

Using environment.yml
- Create a Conda environment with the specified dependencies:
`conda env create -f environment.yml`

Activate the Conda environment
- Activate the newly created Conda environment:
`conda activate my_python_env`

# MRI-based Predicted Transformer for Prostate Cancer

## **Overview**
This repository provides an implementation of the MRI-based Predicted Transformer for Prostate Cancer (MRI-PTPCa). The code includes the full pipeline for data preparation, model training, testing, and statistical analysis. Below is a step-by-step guide to help users run the project.


## **Key Highlights**

We recommend using **Jupyter Notebook** to run the code. To facilitate sharing and secondary development, we have minimized the coupling between different steps, making the code more flexible and easier to understand.


## **Step 1: Data Preparation**

### 1) **Image File Format Conversion**
   - Run the `dcm2nii` function in `prepare_datasets/dcm2nrrd_MRI`.
   - This function organizes multiple unmarked `.dcm` MRI files into single `.nii` or `.nii.gz` files by sequence name.

### 2) **Generating Image Sequence Lists**
   - Run the `gen_dataset_list` function in `prepare_datasets/dcm2nrrd_MRI`.
   - This step associates `.nii` files with patient IDs and generates a dataset list.
   - For better understanding, we provide a sample in `data/PICAI-seq_list.xlsx`.

### 3) **Associating Clinical Information**
   - Use patient IDs to associate clinical information, such as Gleason scores, to pair mp-MRI data with labels.
   - A sample file is provided in `data/PICAI_clinicInfo.xlsx`.


## **Step 2: Data Loading**

- The `loaddata/dataset2.py` file provides the data I/O interface required for model training and testing.
- Run the script directly (`python dataset2.py`) to display the data and label information loaded during each iteration.
- Dataloader I/O: T2WI, ADC, DWI with high B values, and labels.
- Missing sequences are replaced with matrices of all-zero values of the same dimensions.

## **Step 3: Contrastive Learning Training (MRI-BYOL Network)**

- The `/contrastive_learning/MRIBYOL.py` file provides an example of contrastive learning to handle missing sequences.
- During training, either the ADC or DWI sequence is randomly disabled in the input.
- The training minimizes feature differences across branches to achieve the learning objective.
- The data I/O interface for training uses the `generate_img_batch_BYOL` function from `loaddata/dataset1.py`.
- The first stage of contrastive learning follows the classic BYOL framework, where paired data is prepared by randomly masking image content.

## **Step 4: Training the Tumor Aggressiveness Prediction Model (MIMSViT)**

- We provide examples for training:
  - Single-sequence T2WI: `/supervised_learning/MIMSViT_t2wi.py`
  - Multi-parametric MRI (mp-MRI): `/supervised_learning/MIMSViT_mpMRI.py`

## **Step 5: Testing and Evaluation**

### 1) **Model Testing**
   - The `evaluate/evaluate(mp-MRI).py` script provides an example of model testing.
   - Prediction results are saved in `.xlsx` format.

### 2) **Post-processing**
   - The `evaluate/distribute.py` script converts multi-class prediction results into scores linearly correlated with tumor aggressiveness.

## **Step 6: Visualization and Statistical Analysis**

- The `Statistics` folder contains scripts for generating the following:
  - ROC curves
  - Decision curves
  - Confusion matrices
  - NRI-IDI curves

---
