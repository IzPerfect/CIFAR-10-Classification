CIFAR-10-Classification
===
This is CIFAR-10 data classification based on VGG16 architecture.

When using the DNN model, the accuracy did not increase further at about 50%. 

However, when using the VGG model, it shows high accuracy.


Requirement
---
* Python
* Keras
* Python packages : numpy, matplotlib, and so on...

Usage
---
### Command
`python run_main.py`

### Files
* Model Architecture Class
	- DNN : cifar10_DNN.py
	- VGG16 : cifar10_VGG.py
* Use with model imported
	- DNN : cifar10_dnn_ex1.ipynb
	- VGG16 : cifar10_vgg_ex1.ipynb

Reference Implementations
---
+ https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
+ https://github.com/geifmany/cifar-vgg