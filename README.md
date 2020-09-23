# ResUNet-a-TF2.0
Implementation of Diakogiannis et al.'s [ResUNet-a](https://arxiv.org/abs/1904.00592) deep learning framework in Tensorflow 2.0 - originally written in MXNet

**If you're training this yourself look at increasing batch size; I'm having to train locally on a P4000, which means a batch size of 2 with 256x256 images. Such a low batch size is known to have implications on accuracy, even with "corrective" regularisation elsewhere...**

##### To-do:
- Automate learning rate
- Implement valid Tanimoto loss
- Introduce additional augmentation functions (rotate, brightness, zoom)
- Remove or tune early stopping
