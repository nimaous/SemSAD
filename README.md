# UNSUPERVISED ANOMALY DETECTION FROM SEMANTIC SIMILARITY SCORES

This repository contains PyTorch code for the [SemSAD paper]().

SemSAD is a simple and generic framework for detecting examples that lie out-of-distribution (OOD) for a given training set. Our approach is based on learning a semantic similarity measure to find for a given test example the semantically closest example in the training set and then using a discriminator to classify whether the two examples show sufficient semantic dissimilarity such that the test example can be rejected as OOD. 


<figure>
    <img src='paper/Table1.png' />
    <font size="2">
    <figcaption>&nbspTable 1: Out-of-distribution detection performance (% AUROC). Reported values for SemSAD are lower bounds.
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

<figure>
    <img src='paper/Table2.png' />
    <font size="2">
    <figcaption> 
&nbspTable 2: Average over AUROC values from 5 independent training runs for CIFAR-100/CIFAR10 (in/out distribution) for different setups. The lowest AUROC values among the 5 runs are shown in brackets. Reported AUROC values are lower bounds. We applied gaussian blurring on negative samples (blur), extreme transformations on positive samples (extreme transf.), and using correlated negative pairs <a href="https://www.codecogs.com/eqnedit.php?latex=P_{neg}(x,x')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_{neg}(x,x')" title="P_{neg}(x,x')" /></a> derived from extreme transformations of the same image (correlated neg), and changed the fraction of semantically similar pairs (μ) per minibatch, the sampling range for γ, and the semantic neighbourhood size (N). AUROC is computed for CIFAR-100/10 test sets with 10k examples.<br/>
    </figcaption>
    </font>
</figure>

Commands used to train the encoder and the discriminator in the paper [here](https://github.com/nimaous/SemSAD/blob/main/commands.txt)<br/>

Download our trained models [here]()

# Package dependencies
listed [here](https://github.com/nimaous/SemSAD/blob/main/package_version.txt)

# Citation


