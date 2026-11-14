# DSCI 552 — Homework 7

Multi-class and multi-label classification with support vector machines and k-means clustering on the UCI Anuran Calls (MFCCs) data (exact match and Hamming score / loss, binary relevance with one-versus-all Gaussian-kernel SVMs on raw and standardized attributes, L1-penalized linear SVMs, L1 SVMs with SMOTE inside 10-fold cross-validation, and, as extra practice, a classifier chain and multi-label confusion matrices, precision, recall, ROC and AUC; k-means with k chosen by the silhouette and majority-label triplets in a 50-run Monte-Carlo simulation), plus ISLR exercise 12.6.2.

## Structure

```
.
├── README.md
├── .gitignore
├── data
│   └── anuran_calls
│       ├── Frogs_MFCCs.csv
│       └── Readme.txt
├── notebook
│   └── Lu_Yiming_HW7.ipynb
└── requirements.txt
```

## Run

```bash
pip install -r requirements.txt
cd notebook
jupyter notebook Lu_Yiming_HW7.ipynb   # then Kernel -> Restart & Run All
```

The notebook loads data with the relative path `../data/anuran_calls/`. The `RecordID` column is an identifier and is not used as a feature. A full run (10-fold CV over 35 (C, sigma) pairs for each of the three labels with raw and with standardized attributes, two 10-fold CVs of L1 SVMs, and 50 Monte-Carlo runs of k-means for k = 2, ..., 50) takes about 30 minutes on a multi-core machine.
