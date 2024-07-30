from PIL import Image
from torch.utils.data import Dataset
import pickle

class GBMPathDataset(Dataset):
    def __init__(
        self,
        imgs_path_file=None,
        transforms=None,
        func="train",
        fold=5,
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
        # Torchvision transforms to apply
        self.transforms = transforms
        self.func = func
        self.class_map = class_map
        # List containing paths to all the images
        self.data = []

        with open(imgs_path_file, 'rb') as fp:
            all_paths = pickle.load(fp)

        if self.func == "train":
            all_folds = [1, 2, 3, 4, 5]
            selec_folds = [x for x in all_folds if x != fold]
            for selec_fold in selec_folds:
                paths = all_paths[selec_fold]
                for path in paths:
                    class_name = path.split("_")[-2]
                    self.data.append((path, class_name))

        elif self.func  == "val":
            paths = all_paths[fold]
            for path in paths:
                class_name = path.split("_")[-2]
                self.data.append((path, class_name))

        elif self.func == "test":
            for path in all_paths:
                self.data.append((path))

    def __len__(self):
        """Gets the length of the dataset
        Returns:
            int: total number of data points
        """
        return len(self.data)

    def __getitem__(self, idx):
        if self.func != "test":
            img_path, class_name = self.data[idx]
            class_id = self.class_map[class_name]
        else:
            img_path = self.data[idx]
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        if self.func != "test":
            return img, class_id
        else:
            return img_path, img