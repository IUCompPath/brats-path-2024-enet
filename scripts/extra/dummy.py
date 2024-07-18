# Local Imports
import constants as C
from dataset import GBMPathDataset

# Logging
# train_logger = SummaryWriter(log_dir=os.path.join(C.LOG_PATH, "train"))
# val_logger = SummaryWriter(log_dir=os.path.join(C.LOG_PATH, "val"))

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load train data
train_dataset = GBMPathDataset(
    imgs_path_prefix=C.IMAGES_PATH_PREFIX,
    imgs_path_file=C.TRAIN_IMAGES_PATHS,
    transforms=C.train_transforms
)

# Create labels list for StratifiedKFold
data_pts = [pt for pt, _ in train_dataset]
labels = [label for _, label in train_dataset]

label_count_dict = {"CT": 0, "IC": 0, "MP": 0, "NC": 0, "PN": 0, "WM":0}
for label in labels:
    if label in label_count_dict.keys():
        label_count_dict[label] += 1
class_counts = list(label_count_dict.values())


import pickle

# Specify the filename to save the variables
filename = 'dataset_and_labels.pkl'

# Open the file in write-binary mode and use pickle to dump the variables
with open(filename, 'wb') as file:
    pickle.dump({
        'train_dataset': train_dataset,
        'data_pts': data_pts,
        'labels': labels
    }, file)