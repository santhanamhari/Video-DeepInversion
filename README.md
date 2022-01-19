# Binary Masks to Video Frames via DeepInversion
DeepInversion is applied to invert a Mask R-CNN architecture, in order to produce synthetic frames of videos in the DAVIS dataset. We perform input optimization from random noise to high fidelity frames. Specifically, we optimize a classification loss, defined between ground truth and predicted coarse masks, as well as auxiliary losses that minimize noise and batch normalization statistic differences. We train for 2k iterations with a learning rate of 0.1 and an Adam optimizer. The viability of our method is tested on many first frames of videos in the DAVIS set, with different auxiliary loss parameter scaling values for each frame. Finally, we synthesize many frames of the ’bear’ video and string them together to produce a synthetic video. 

## Requirements
- python 3.6
- tensorflow 1.8
- pytorch 0.2.0_3 (pip install https://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl) (this is a version for CUDA 8.0, but I always run this using CUDA 9.0 and it works)
- CUDA 9.0
- cudnn 7
- COCO API for Python 3


```setup
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install numpy
pip install Pillow
```


![image_1427](https://user-images.githubusercontent.com/40223805/150068945-1c70f960-f4bb-4118-bfc8-f057d24d707c.png)
