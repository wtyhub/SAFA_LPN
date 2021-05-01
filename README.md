# SAFA + LPN


### Experiment Dataset
We use two existing dataset to do the experiments

- CVUSA datset: a dataset in America, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.  
	The dataset can be accessed from https://github.com/viibridges/crossnet

- CVACT dataset: a dataset in Australia, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.  
	The dataset can be accessed from https://github.com/Liumouliu/OriCNN


### Dataset Preparation
Please Download the two datasets from above links, and then put them under the director "Data/". The structure of the director "Data/" should be:
"Data/CVUSA/
 Data/ANU_data_small/"

### Models:

There is also an "Initialize" model for your own training step. The VGG16 part in the "Initialize_model" model is initialised by the online model and other parts are initialised randomly. 

Please put them under the director of "Model/" and then you can use them for training or evaluation.


### Codes

1. Training:
	CVUSA: python train_cvusa_lpn.py
	CVACT: python train_cvact_lpn.py

2. Evaluation:
	CVUSA: python test_cvusa.py
	CVACT: python test_cvact.py


### Reference  
[Spatial-Aware Feature Aggregation for Cross-View Image Based Geo-Localization](http://papers.nips.cc/paper/9199-spatial-aware-feature-aggregation-for-image-based-cross-view-geo-localization.pdf)

[github](https://github.com/shiyujiao/cross_view_localization_SAFA.git)
