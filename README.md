# UNSUPERVISED ANOMALY DETECTION FROM SEMAN- TIC SIMILARITY SCORES

This repository contains PyTorch code for the [SemSAD paper]().

SemSAD is a simple and generic framework for detecting examples that lie out-of-distribution (OOD) for a given training set. Our approach is based on learning a semantic similarity measure to find for a given test example the semantically closest example in the training set and then using a discriminator to classify whether the two examples show sufficient semantic dissimilarity such that the test example can be rejected as OOD.


<figure>
    <img src='paper/Table1.png' />
    <font size="2">
    <figcaption> Table 1: Out-of-distribution detection performance (% AUROC). Reported values for SemSAD are lower bounds.
    </figcaption>
    </font>
</figure>


<figure>
    <img src='paper/Figure4.png' />
    <font size="2">
    <figcaption> 
    </figcaption>
    </font>
</figure>

<figure>
    <img src='paper/Table2.png' />
    <font size="2">
    <figcaption> 
    </figcaption>
    </font>
</figure>

Commands used to train the encoder and the discriminator in the paper [here](https://github.com/nimaous/SemSAD/blob/main/commands.txt)

Download our trained models [here]()

# Package dependencies
listed [here](https://github.com/nimaous/SemSAD/blob/main/package_version.txt)

# Citation


