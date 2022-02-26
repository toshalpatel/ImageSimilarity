import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from model import IMEncoder
from dataset import SimilarImagesDataset


def create_embedding(encoder, data_loader, embedding_dim, device):
    '''
    create embeddings for finding image similarity
    '''
    encoder.eval()
    embedding = torch.randn(embedding_dim)

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(data_loader):

            train_img = train_img.to(device)
            enc_output = encoder(train_img).cpu()
            embedding = torch.cat((embedding, enc_output), 0)

    return embedding



def compute_similar_images(encoder, image, num_images, embedding, device):

    image_tensor = T.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    flattened_embedding = image_embedding.reshape(
        (image_embedding.shape[0], -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    return indices_list


# Run the training on GPU
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

embedding_shape = (1, 128, 16, 16)
model_path = 'models/encoder_model.pt'

# Normalize and convert to tensor
transforms = T.Compose([T.ToTensor()])

# Load the dataset
data = SimilarImagesDataset('./data/similarity_test_examples/', transform=transforms)
data_loader = DataLoader(data, batch_size=1)

# load model
encoder = IMEncoder(in_c=3, kernel=(3, 3), padding=(1, 1))
encoder.load_state_dict(torch.load(model_path))
encoder.eval()

# save feature representation
embedding = create_embedding(encoder, data_loader, embedding_shape, device)

# Convert embedding to numpy and save them
numpy_embedding = embedding.cpu().detach().numpy()
num_images = numpy_embedding.shape[0]

flattened_embedding = numpy_embedding.reshape((num_images, -1))
#np.save("data_embedding.npy", flattened_embedding)


# Load true image
true_image = Image.open('true_img.png').convert("RGB")
true_image = true_image.resize((512, 512))
true_image = transforms(true_image)
true_image = torch.unsqueeze(true_image, 0)
print(true_image.shape)

# get true image embedding
encoder.eval()
true_image = true_image.to(device)
embedding_true = encoder(true_image).cpu().detach().numpy()
embedding_true = embedding_true.reshape((embedding_true.shape[0], -1))
print(embedding_true.shape)

distances = []
for i in range(num_images):
    distances.append(cosine(embedding_true, flattened_embedding[i]))

# known from the order of loading of the dataset
imgs = ['./data/similarity_test_examples/sample4.png',
        './data/similarity_test_examples/sample2.png',
        './data/similarity_test_examples/sample3.png',
        './data/similarity_test_examples/sample1.png']

# normalize the array
d = np.asarray(distances)
norm = np.linalg.norm(d)
normal_array = d/norm

# similarity
similarities = (1 - normal_array) * 100

# show true image
im = Image.open('true_img.png')
plt.title("True Image")
imgplot = plt.imshow(im)
plt.show()

# plot the images with scores
_, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()

for img_path, ax, s in zip(imgs, axs, similarities):
	name = img_path.split('/')[-1]

	im = Image.open(img_path)
	ax.set_title("{} : {:.5f} %".format(name, s))
	ax.imshow(im)


plt.show()
plt.savefig("result_similarity_image.jpg")
