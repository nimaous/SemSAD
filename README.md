## PyTorch-SemSAD

PyTorch implementation of [Unsupervised Anomaly Detection From Semantic Similarity Scores](https://arxiv.org/abs/2012.00461).


[SemSAD](https://arxiv.org/abs/2012.00461) is a simple and generic framework for detecting examples that lie out-of-distribution (OOD) for a given training set. Our approach is based on learning a semantic similarity measure to find for a given test example the semantically closest example in the training set and then using a discriminator to classify whether the two examples show sufficient semantic dissimilarity such that the test example can be rejected as OOD. 


<p align="center">
<img src="figures/Table1.png" width="400px">
<img src="figures/Figure5.png" width="400px">
</p>

<p align="center">
<img src="figures/Table2.png" width="400px"></img>
<p/>

<p align="center">
<img src="figures/Table4.png" width="400px"></img>
<p/>

<p align="center">
<img src="figures/Table6.png" width="400px"></img>
<p/>


Commands used to train the encoder and the discriminator in the paper : [ImageData](https://github.com/nimaous/SemSAD/blob/main/ImageData/commands.txt), [TextData](https://github.com/nimaous/SemSAD/blob/main/TextData/commands.txt) and [AudioData](https://github.com/nimaous/SemSAD/blob/main/AudioData/commands.txt)<br/>

Download our trained models for [ImageData](https://github.com/nimaous/SemSAD/tree/main/ImageData/trained_models), [TextData](https://github.com/nimaous/SemSAD/tree/main/TextData/trained_models) and [AudioData](https://github.com/nimaous/SemSAD/tree/main/AudioData/trained_models)

## Requirements
• [ImageData](https://github.com/nimaous/SemSAD/blob/main/ImageData/requirements.txt)<br/>
• [TextData](https://github.com/nimaous/SemSAD/blob/main/TextData/requirements.txt)<br/>
• [AudioData](https://github.com/nimaous/SemSAD/blob/main/AudioData/requirements.txt)<br/>

## Citation
```
@misc{rafiee2021unsupervised,
      title={Unsupervised Anomaly Detection From Semantic Similarity Scores}, 
      author={Nima Rafiee and Rahil Gholamipoor and Markus Kollmann},
      year={2021},
      eprint={2012.00461},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
