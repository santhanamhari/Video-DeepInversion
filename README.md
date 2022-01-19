# Binary Masks to Video Frames via DeepInversion
DeepInversion is applied to invert a Mask R-CNN architecture, in order to produce synthetic frames of videos in the DAVIS dataset. We perform input optimization from random noise to high fidelity frames. Specifically, we optimize a classification loss, defined between ground truth and predicted coarse masks, as well as auxiliary losses that minimize noise and batch normalization statistic differences. The viability of our method is tested on many first frames of videos in the DAVIS set, with different auxiliary loss parameter scaling values for each frame.
<img width="916" alt="Screen Shot 2022-01-19 at 12 36 57 AM" src="https://user-images.githubusercontent.com/40223805/150070495-acfce110-ea39-4036-bbe6-94f5f696b1b3.png">



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

## Example Results
![image_1427](https://user-images.githubu<img width="387" alt="Screen Shot 2022-01-19 at 12 42 44 AM" src="https://user-images.githubusercontent.com/40223805/150071167-4291fc9c-e04d-4f63-9558-76b5f3692c39.png">
<img width="385" alt="Screen Shot 2022-01-19 at 12 42 49 AM" src="https://user-images.githubusercontent.com/40223805/150071171-95133320-bee4-4d80-98de-53c7264ea5dc.png">
<img width="382" alt="Screen Shot 2022-01-19 at 12 42 56 AM" src="https://user-images.githubusercontent.com/40223805/150071174-9e3dbdaa-8b08-4b0a-a45f-357817199337.png">
<img width="384" alt="Screen Shot 2022-01-19 at 12 43 01 AM" src="https://user-images.githubusercontent.com/40223805/150071177-f143e0e6-afe6-4f65-ae92-123f62e3e7e4.png">
sercontent.com/40223805/150068945-1c70f960-f4bb-4118-bfc8-f057d24d707c.png)

