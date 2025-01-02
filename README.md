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
