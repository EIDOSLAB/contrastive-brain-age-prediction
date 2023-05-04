# Contrastive learning for regression in multi-site brain age prediction

Carlo Alberto Barbano<sup>1,2</sup>, Benoit Dufumier<sup>1,3</sup>, Edouard Duchesnay<sup>3</sup>, Marco Grangetto<sup>2</sup>, Pietro Gori<sup>1</sup> | [[pdf](https://arxiv.org/pdf/2211.08326.pdf)] [[poster](https://drive.google.com/file/d/1gr45EamhVVClPbMT5T5b1Gy9V50fgw3c/view)]

1<sub>LTCI, Télécom Paris, IP Paris</sub><br>
2<sub>University of Turin, Computer Science dept.</sub><br>
3<sub>NeuroSpin, CEA, Universite Paris-Saclay</sub>
<br/><br/>

![asd](assets/teaser.png)

Building accurate Deep Learning (DL) models for brain age prediction is a very relevant topic in neuroimaging, as it could help better understand neurodegenerative disorders and find new biomarkers. To estimate accurate and generalizable models, large datasets have been collected, which are often multi-site and multi-scanner. This large heterogeneity negatively affects the generalization performance of DL models since they are prone to overfit site-related noise. Recently, contrastive learning approaches have been shown to be more robust against noise in data or labels. For this reason, we propose a novel contrastive learning regression loss for robust brain age prediction using MRI scans. Our method achieves state-of-the-art performance on the OpenBHB challenge, yielding the best generalization capability and robustness to site-related noise.


## Running 


## Citing

For citing our work, please use the following bibtex entry:

```bibtex
@inproceedings{barbano2023contrastive,
    author = {Barbano, Carlo Alberto and Dufumier, Benoit and Duchesnay, Edouard and Grangetto, Marco and Gori, Pietro},
    journal = {International Symposium on Biomedical Imaging (ISBI)},
    title = {Contrastive learning for regression in multi-site brain age prediction},
    year = {2023}
}
```