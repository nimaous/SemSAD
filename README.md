# We are still updating this repo 
## UNSUPERVISED ANOMALY DETECTION FROM SEMANTIC SIMILARITY SCORES

This repository contains PyTorch code for the [SemSAD paper](https://arxiv.org/abs/2012.00461).

SemSAD is a simple and generic framework for detecting examples that lie out-of-distribution (OOD) for a given training set. Our approach is based on learning a semantic similarity measure to find for a given test example the semantically closest example in the training set and then using a discriminator to classify whether the two examples show sufficient semantic dissimilarity such that the test example can be rejected as OOD. 


<html>
  <head>
    <title>Center an Image using text align center</title>
    <style>
      .img-container {
        text-align: center;
        display: block;
      }
    </style>
  </head>
  <body>
    <span class="img-container"> <!-- Inline parent element -->
      <img src="paper/Table1.png" alt="">
    </span>
  </body>
</html>

<figure>
    <img src='paper/Table 2.png' />
    <font size="0.5">
    <figcaption>Table 2: Out-of-distribution detection performance (% AUROC) for Tiny Imagenet dataset as OOD.
    </figcaption>
    </font>
</figure>

<figure>
    <img src='paper/Table 4.png' />
    <font size="0.5">
    <figcaption>Table 4: Out-of-distribution detection performance (% AUROC). Reported values for SemSAD are lower bounds.
    </figcaption>
    </font>
</figure>

<figure>
    <img src='paper/Table 6.png' />
    <font size="0.5">
    <figcaption>Table 6: Out-of-distribution detection performance (% AUROC). Reported values for SemSAD are lower bounds.
    </figcaption>
    </font>
</figure>

<figure>
    <img src='paper/Figure4.png' />
    <font size="2">
    <figcaption>Figure 4: Distributions over the semantic similiarity score, s(x, x′), trained on CIFAR-100 pos/neg pairs (<a href="https://www.codecogs.com/eqnedit.php?latex=P_{pos}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_{pos}" title="P_{pos}" /></a> in blue; <a href="https://www.codecogs.com/eqnedit.php?latex=P_{neg}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_{neg}" title="P_{neg}" /></a> in red) as described in Section 4.1 and applied to semantic nearest- neighbour pairs from the test sets of SVHN and CIFAR-10 (out-distributions) in comparison to semantic nearest-neighbour pairs of the CIFAR-100 test/train sets (in-distributions).
    </figcaption>
    </font>
</figure>



Commands used to train the encoder and the discriminator in the paper [here](https://github.com/nimaous/SemSAD/blob/main/commands.txt)<br/>

Download our trained models [here](https://www.dropbox.com/sh/rsjz3gqswk8xtqn/AAC35v9J2hsHxoBaHVCgN22ua?dl=0)

# Package dependencies
listed [here](https://github.com/nimaous/SemSAD/blob/main/package_version.txt)



