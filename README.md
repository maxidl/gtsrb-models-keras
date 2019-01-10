
# Train Keras models on German Traffic Sign Recognition Benchmark (GTSRB)

This repository contains a simple and light script to train several CNNs on the [GTRSB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset.
Currently, you can select between an AlexNet, LeNet-5, VGG19 and the ResNet50, but note that the last two are pretrained on ImageNet.

## Requirements
- numpy
- tensorflow
- matplotlib

## Usage

1. First, download GTSRB training and test dataset from http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads 
2. Train a network using the following command:

```
python train_model.py 
-a="<model architecture>" 
-t="<path to train images>" 
-v="<path to validation images>" 
-l="path to validation labels" 
-i=<image size> 
-g="<train on grayscale images>"
```

#### Argparse parameters

| Option | Required | Choices | Default| Option Summary |  
|---------------|----------|---------|--------|----------------|  
| ['-a', '--architecture'] | True | True |  | Model architecture for training. ['alex', 'vgg19', 'resnet50', 'lenet-5'] | 
| ['-t', '--train_path'] | False | False | res/GTSRB/train/Final_Training/Images/" | Input directory for train set | 
| ['-v', '--validation_path'] | False | False | res/GTSRB/test/Final_Test/Images/ | Input directory for validation set | 
| ['-l', '--validation_labels'] | False | False | res/GTSRB/test/Final_Test/GT-final_test.csv | Path to 'GT-final_test.csv' file | 
| ['-i', '--image_dim'] | False | False | 64 | Image width and height in pixel (width == height) | 
| ['-g', '--grayscale'] | False | False | False | Train only on grayscale images | 
| ['-r', '--result_folder'] | False | False | results/ | Recursively walk through all folders in the project directory | 

#### Usage example

For instance, it could look like this:

```
python train_model.py 
-a="alex" 
-t="res/GTSRB/train/Final_Training/Images/" 
-v="res/GTSRB/test/Final_Test/Images/" 
-l="res/GTSRB/test/Final_Test/GT-final_test.csv" 
-i=64 
-g="false"
```

This command will train the network by 10 epochs and produces a keras `model.h5` file keeping the best. In addition, training process will be logged and the accuracy and loss is visualized.

![training and validation accuracy](/results/model_acc.svg)
![training and validation loss](/results/model_loss.svg)
		

