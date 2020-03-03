# CLRS: Continual Learning Benchmark for Remote Sensing Image Scene Classification

This is a new datasae design for a new task named Continual/Lifelong learning for remote sensing image scene classification. The existing remote sensing image scene classification datasets use static benchmarks and lack the standard to divide the datasets into a number of sequential learning training batches, which largely limits the development of Continual/Lifelong learning in remote sensing image scene classification.

Splitting the training set into a number of batches is essential to train and test Continual/Lifelong learning approaches which are currently receiving much attention. Unfortunately, most of the existing datasets are not well suited to this purpose because they lack a fundamental ingredient: the presence of multiple (unconstrained) views of the same objects taken in different sessions. 

In this dataset, we consider three Continual/Lifelong learning scenarios:

```
* New Instances (NI): new training patterns of the same classes becomes available in subsequent batches with new poses and environment conditions. A good model is expected to incrementally consolidate its knowledge about the known classes without compromising what it learned before.

* New Classes (NC): new training patterns belonging to different classes becomes available in subsequent batches. In this case the model should be able to deal with the new classes without losing accuracy on the previous ones.

* New Instances and Classes (NIC): new training patterns belonging both to known and new classes becomes available in subsequent training batches. A good model is expected to consolidate its knowledge about the known classes and to learn the new ones.
```


# CLRS Dataset
The proposed CLRS data set consists of 15,000 remote sensing images divided into 25 scene classes, namely, airport, bare-land, beach, bridge, commercial, desert, farmland, forest, golf-course, highway, industrial, meadow, mountain, overpass, park, parking, playground, port, railway, railway-station, residential, river, runway, stadium, and storage-tank. Each class has 600 images, and the image size is 256x256. The resolution of the images ranges from 0.26 m to 8.85 m. <br> 
<div align=center><img src="https://github.com/jh101024/Python/blob/master/CLRS-samples.png"/></div>

# CLRS Dataset download
[CLRS Dataset can be downloaded here in BaiduYun](https://pan.baidu.com/s/1NkkaJxPtewW5fQMk8yCAQw)

# Experiment results
1) NI scenario:<br>
<img src="pics/NI.png" width="600px" hight="400px" />

2) NC scenario:<br>
<img src="pics/NC.png" width="600px" hight="400px" />

3) NIC scenario:<br>
<img src="pics/NIC.png" width="600px" hight="400px" />

# Code requirements
* tensorflow
* scipy
* numpy
* matplotlib
* scikit-image

# The manuscript
The manuscript can be visited at https://www.mdpi.com/1424-8220/20/4/1226

If this repo is useful in your research, please kindly consider citing our paper as follow.
```
Bibtex
@Article{s20041226,
AUTHOR = {Li, Haifeng and Jiang, Hao and Gu, Xin and Peng, Jian and Li, Wenbo and Hong, Liang and Tao, Chao},
TITLE = {CLRS: Continual Learning Benchmark for Remote Sensing Image Scene Classification},
JOURNAL = {Sensors},
VOLUME = {20},
YEAR = {2020},
NUMBER = {4},
ARTICLE-NUMBER = {1226},
URL = {https://www.mdpi.com/1424-8220/20/4/1226},
ISSN = {1424-8220},
DOI = {10.3390/s20041226}
}
```
