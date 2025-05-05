# chemDTI
Evaluating different molecular fingerprints and machine learning approaches for Drug-Target interactions prediction

# Data
- Download training  and test sets from:
https://tripod.nih.gov/tox21/challenge/data.jsp

## FpGen files
- generate different molecular fingerprints

## clust_combine.py
- load the calculated fingerprints, calculates tanimoto/jackard similarities between the fingerprints.

## models.py
- trains different ml models on the fingerprint data, compares performances

## deep.py
- training deep learning models on fingerprint data, comparing performances


