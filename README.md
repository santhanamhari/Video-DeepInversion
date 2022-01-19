# Binary Masks to Video Frames via DeepInversion
Machine learning models rely on data for training, so they can help make real-world predictions. The acquisition of such training data can be arduous in certain situations. For example, some models are developed and trained using data that is privacy protected. As a result, such datasets become inaccessible to researchers seeking to train new models and make predictions. If we can recover the training data from a pre-trained model, this can greatly aid in potential knowledge transfer. In this thesis, we use a recently developed technique called DeepInversion to synthesize video training data.

<img width="916" alt="Screen Shot 2022-01-19 at 12 36 57 AM" src="https://user-images.githubusercontent.com/40223805/150070495-acfce110-ea39-4036-bbe6-94f5f696b1b3.png">

DeepInversion is applied to invert a Mask R-CNN architecture, in order to produce synthetic frames of videos in the DAVIS dataset. We perform input optimization from random noise to high fidelity frames. Specifically, we optimize a classification loss, defined between ground truth and predicted coarse masks, as well as auxiliary losses that minimize noise and batch normalization statistic differences. We train for 2k iterations with a learning rate of 0.1 and an Adam optimizer. The viability of our method is tested on many first frames of videos in the DAVIS set, with different auxiliary loss parameter scaling values for each frame. Finally, we synthesize many frames of the ’bear’ video and string them together to produce a synthetic video. Ideas developed in this thesis can be greatly beneficial in the domains of federated learning, privacy-protected data acquisition, and lower latency model training.


## Requirements
```setup
- python 3.6
- tensorflow 1.8
- pytorch 0.2.0_3 (pip install https://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl) (this is a version for CUDA 8.0, but I always run this using CUDA 9.0 and it works)
- CUDA 9.0
- cudnn 7
- COCO API for Python 3
```
## Run Code

`python train.py --nr_epochs=2000 --lr=0.1 --var_l2 --var_l1 --l2_norm --bn`


- `nr_epochs` - number of epochs for training
- `lr` - learning rate for the optimizer
- `var_l2` - scaling used for l2 variance pixel loss
- `var_l1` - scaling used for l1 variance pixel loss
- `l2_norm` - scaling used for noise optimization loss
- `bn` - scaling used for batch normalization loss

We train for 2k iterations with a learning rate of 0.1 and an Adam optimizer. 

## Methodology

We use DeepInversion to invert a Mask R-CNN architecture to synthesize training data originally from the DAVIS video dataset. We perform noise optimization and minimize a classification loss as well as two auxiliary losses. The classification loss is defined by comparing a coarse mask to the ground truth mask label. Additionally, the two auxiliary losses are used to minimize noise and force the synthesized images towards the batch normalization statistics.  DeepInversion, with slight adaptions, shows great promise in creating high-fidelity synthetic frames for videos in the DAVIS dataset.

## Example Results

<img width="999" alt="Screen Shot 2022-01-19 at 12 45 39 AM" src="https://user-images.githubusercontent.com/40223805/150071425-211060dd-318d-46f2-81f7-e5010c309d95.png">

