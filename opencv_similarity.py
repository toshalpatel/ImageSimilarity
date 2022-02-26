import cv2
import glob 
from scipy.spatial.distance import cosine

import numpy as np
import matplotlib.pyplot as plt

# for MacOS
import matplotlib as mpl
mpl.use('tkagg')

# true image
true_img_path = "data/true_img.png"
print("True image: ", true_img_path)

image = cv2.imread(true_img_path, cv2.IMREAD_UNCHANGED)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram_true = cv2.calcHist([image], [0], None, [256], [0, 256])


# Loop through sample images to compare with true image
distances = []
sample_images_path = './data/similarity_test_examples/'

for img_path in glob.glob(sample_images_path+'/*'):
	print("Processing True image vs ", img_path)
	
	image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	histogram_sample = cv2.calcHist([image], [0], None, [256], [0, 256])

	# cosine distance between the two histograms 
	dist = cosine(histogram_sample, histogram_true)
	distances.append(dist)
	print(dist)


# get similarity percentage based on distances
# normalize the array
d = np.asarray(distances)
norm = np.linalg.norm(d)
normal_array = d/norm

# similarity
similarities = (1 - normal_array) * 100

# plot the images with scores
_, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()

for img_path, ax, s in zip(glob.glob(sample_images_path+'/*'), axs, similarities):
	name = img_path.split('/')[-1]

	im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	
	ax.set_title("{} : {:.5f} %".format(name, s))
	ax.imshow(im)
	

#plt.show()
plt.savefig("results/result_similarity_image_opencv.jpg")
