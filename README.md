# SemSAD

This repository contains PyTorch code for the SemSAD paper[add link].

SemSAD is a simple and generic framework for detecting examples that lie out-of-distribution (OOD) for a given training set. The approach is based on learning a semantic similarity measure to find for a given test example the semantically closest example in the training set and then using a discriminator to classify whether the two examples show sufficient semantic dissimilarity such that the test example can be rejected as OOD. Here are the results of .

<img src="paper/Figure4.png" width="600">
<img src="paper/Table2.png" width="600">




Download our trained models [here]()

# Package dependencies
listed [here](https://github.com/nimaous/SemSAD/blob/main/package_version.txt)
