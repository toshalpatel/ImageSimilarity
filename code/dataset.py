import os
from PIL import Image
from torch.utils.data import Dataset


class SimilarImagesDataset(Dataset):

    def __init__(self, im_path, transform=None):
        self.path = im_path
        self.files = self.absolute_file_paths(self.path)
        self.transform = transform

    def absolute_file_paths(self, directory):
        path = os.path.abspath(directory)
        return [entry.path for entry in os.scandir(path) if entry.is_file()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((512, 512))

        if self.transform is not None:
            tensor_image = self.transform(image)
            return tensor_image, tensor_image

        return image
