# BioSift-Predicting-Molecular-Bioactivity-Using-Physicochemical-Descriptors-and-Machine-Learning
# BioSift

**BioSift** is a machine learning-powered molecular screening application that predicts whether a small molecule is likely to be biologically active or inactive based on physicochemical descriptors.

## Features

* Bioactivity prediction using machine learning
* Interactive Streamlit interface
* Active/Inactive classification
* Probability-based prediction scores
* Preloaded compound examples
* Molecular descriptor guidance

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Joblib

## Descriptors

The model uses the following molecular descriptors:

* Molecular Weight (MolWt)
* LogP
* Hydrogen Bond Donors
* Hydrogen Bond Acceptors
* Topological Polar Surface Area (TPSA)

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Goal

This project demonstrates how machine learning can be applied to molecular descriptor data to support early-stage virtual screening and bioactivity prediction.


## Author

Developed by Judi Yousri.
