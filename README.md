# Image Similarity using Autoencoders

This repository finds the image similarity measure between images using AutoEncoders. For the sake of comparision, OpenCV similarity is also calculated separately.


## Directory structure 

```
|- code/  ## code from the ImageSimilarity.ipynb notebook into different modules
    |- dataset.py  ## to load dataset
    |- image_similarity.py  ## to find similarity between true image and sample images
    |- model.py  ## contains the autoencoder model definition
    |- train.py  ## to train the model
|- data/ #sample and true images
    |- similarity_test_examples ## sample images
    |- true_img.png  ## query image
|- models/ ## trained models
|- results/ ## result images with similarities
|
|- ImageSimilarity.ipynb  ## Notebook with image similarity example and execution
|- opencv_similarity.py  ## Trying out image similarity with OpenCV
```


## Setup

To install the virtual environment and requirements in linux:

```
$ sudo apt-get install virtualenv
$ virtualenv -p python3 env 
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Train and Deploy

To train your model on custom images, follow the above mentioned directory structure. 
```
python train.py
```

To run the Image similarity on trained models, 

```
python image_similarity.py
```