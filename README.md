# Image Similarity using Autoencoder using PyTorch

Finds the image similarity measure between images using AutoEncoder. The encoder allows the image to be embedded into feature maps and the decoder re-creates the image from the feature maps.

The encoder and decoder models are trained on the sample dataset and based on the best encoder model saved, the feature embeddings are created. Using cosine distance, we can get a clear idea of which image is closer, i.e., more similar to the True/ Query image.

For the sake of example, the true image and the resulting similarities of various sample images are shown below.



For the sake of comparision, OpenCV similarity is also calculated separately which can be seen in `results`.


## Directory structure 

```
|- code/                         ## code from the ImageSimilarity.ipynb notebook into different modules
    |- dataset.py                ## to load dataset
    |- image_similarity.py       ## to find similarity between true image and sample images
    |- model.py                  ## contains the autoencoder model definition
    |- train.py                  ## to train the model
|- data/                         ## sample and true images
    |- similarity_test_examples  ## sample images
    |- true_img.png              ## query image
|- models/                       ## trained models
|- results/                      ## result images with similarities
|
|- ImageSimilarity.ipynb         ## Notebook with image similarity example and execution
|- opencv_similarity.py          ## Trying out image similarity with OpenCV
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