from PIL import Image
from torch.utils.data import Dataset
import pickle
import numpy as np

class GBMPathDataset(Dataset):
    def __init__(
        self,
        imgs_path_file=None,
        # fold=None,
        func="test",
        transforms=None,
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
        # self.fold = fold
        self.func = func
        # Classes dictionary
        self.class_map = class_map
        # Torchvision transforms to apply
        self.transforms = transforms
        # List containing paths to all the images
        self.data = []
        if self.func == "test":
            with open(self.imgs_path_file, 'rb') as fp:
                all_test_paths = pickle.load(fp)
            for path in all_test_paths:
                class_name = path.split("_")[1]
                self.data.append((path, class_name))
        # elif self.func == "val":
        #     with open(self.imgs_path_file, 'rb') as fp:
        #         all_fold_paths = pickle.load(fp)
        #     np.random.shuffle(all_fold_paths[self.fold])
        #     for path in all_fold_paths[self.fold]:
        #         class_name = path.split("_")[1]
        #         self.data.append((path, class_name))
        # elif self.func == "train":
        #     with open(self.imgs_path_file, 'rb') as fp:
        #         all_fold_paths = pickle.load(fp)
        #     for num in all_fold_paths:
        #         if num != self.fold:
        #             np.random.shuffle(all_fold_paths[num])
        #             for path in all_fold_paths[num]:
        #                 class_name = path.split("_")[1]
        #                 self.data.append((path, class_name))
        elif self.func == "train":
            with open(self.imgs_path_file, 'rb') as fp:
                all_fold_paths = pickle.load(fp)
            np.random.shuffle(all_fold_paths)
            for path in all_fold_paths:
                class_name = path.split("_")[1]
                self.data.append((path, class_name))

    def __len__(self):
        """Gets the length of the dataset
        Returns:
            int: total number of data points
        """
        return len(self.data)

    def __getitem__(self, idx):
        # idx - indexing data for accessibility
        img_path, class_name = self.data[idx]
        # Assigning ids to each class (number, not name of the class)
        class_id = self.class_map[class_name]
        # Loads an image from the given image_path
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img, class_id