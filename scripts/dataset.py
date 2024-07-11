from PIL import Image
from torch.utils.data import Dataset
import pickle
import numpy as np

class GBMPathDataset(Dataset):
    def __init__(
        self,
        imgs_path_file=None,
        transforms=None,
        func="test",
        class_map={
            "CT" : 0,
            "PN" : 1,
            "MP" : 2,
            "NC" : 3,
            "IC" : 4,
            "WM" : 5
        }
    ):
        # Path to dataset
        self.imgs_path_file = imgs_path_file
        # Torchvision transforms to apply
        self.transforms = transforms
        self.func = func
        self.class_map = class_map
        # List containing paths to all the images
        self.data = []
        with open(self.imgs_path_file, 'rb') as fp:
            all_paths = pickle.load(fp)
        if self.func == "train":
            np.random.shuffle(all_paths)
        for path in all_paths:
            if self.func == "train":
                class_name = path.split("_")[1]
                self.data.append((path, class_name))
            else:
                self.data.append((path))

    def __len__(self):
        """Gets the length of the dataset
        Returns:
            int: total number of data points
        """
        return len(self.data)

    def __getitem__(self, idx):
        if self.func == "train":
            img_path, class_name = self.data[idx]
            class_id = self.class_map[class_name]
        else:
            img_path = self.data[idx]
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        if self.func == "train":
            return img, class_id
        else:
            return img_path, img