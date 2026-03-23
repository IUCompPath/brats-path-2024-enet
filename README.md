# Brain Tumor Segmentation - Pathology Challenge : IUCompPath codebase 

## Dataset
The dataset should be arranged in this format:
```
Dataset root folder
└───Class-1
│   image1.png
│   image2.png
│   image3.png
|   ...
│   
└───Class-2
│   image1.png
│   image2.png
│   image3.png
|   ...
|
...
```
The name of the parent folder of each image is considered the ground truth label for that image.
The images must be split into training and validation sets before starting training.
The TRAIN_IMAGE_PATHS and VALIDATION_IMAGE_PATHS in the [constants.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/constants.py) file must be paths to the pickle file that has the file paths of all the training and validation splits in the form of a list.

```
all_paths.pkl = ["1/image1.png", "1/image2.png", "1/image3.png", "2/image1.png", "2/image2.png", "2/image3.png", ...]
```

The TRAIN_IMAGE_PATHS and VALIDATION_IMAGE_PATHS in the [cv/constants.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/cv/constants.py) file must be paths to the pickle file that has the file paths of all the training and validation splits in the form of a dictionary for each fold. Example:

```
all_fold_paths.pkl = {
    1: ["1/image1.png", "1/image2.png", "1/image3.png"],
    2: ["2/image1.png", "2/image2.png", "2/image3.png"],
...
}
```
The cross-validation split can be performed using the [split_cv_data.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/cv/split_cv_data.py) script.

## Docker containers
### BraTS-Path 2024 docker
sanyuktaadap/brats-24-inference:latest

### BraTS-Path 2025 docker
sanyuktaadap/brats-25-inference:latest

## Training, Validation and Testing

### Run single training (no cross-validation)

1. In [constants.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/constants.py) script, add the following
- Paths to the train and validation data
- Add class names in the order of needed prediction

2. Run [train.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/train.py) to start training. Check required input arguments for training hyperparameters.

### Run training using Cross-Validation

1. In [cv/constants.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/cv/constants.py) script, add the following
- Paths to the train and validation data
- Add class names in the order of needed prediction

2. Run [cv/train.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/cv/train.py) to start training. Check required input arguments for training hyperparameters.

### Run testing using script

1. Run [test.py](https://github.com/IUCompPath/brats-path-2024-enet/blob/main/scripts/test.py) using the input arguments and the run_id/experiment number used during training.
