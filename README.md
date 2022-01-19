# Video-DeepInversion
DeepInversion is applied to invert a Mask R-CNN architecture, in order to produce synthetic frames of videos in the DAVIS dataset. We perform input optimization from random noise to high fidelity frames. Specifically, we optimize a classification loss, defined between ground truth and predicted coarse masks, as well as auxiliary losses that minimize noise and batch normalization statistic differences. We train for 2k iterations with a learning rate of 0.1 and an Adam optimizer. The viability of our method is tested on many first frames of videos in the DAVIS set, with different auxiliary loss parameter scaling values for each frame. Finally, we synthesize many frames of the ’bear’ video and string them together to produce a synthetic video. 

Requirements

![image_1427](https://user-images.githubusercontent.com/40223805/150068945-1c70f960-f4bb-4118-bfc8-f057d24d707c.png)
