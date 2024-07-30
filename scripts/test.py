import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import os
import gc

# Torch Imports
import torch
from torch.utils.data import DataLoader
# import torch.nn.functional as nn
from torchvision.transforms import v2
from efficientnet_pytorch import EfficientNet

# Local Imports
from dataset import GBMPathDataset
from densenet import DenseNetClassifier
import constants as C
from core_utils import get_model

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='This script trains the model'
    )
    parser.add_argument('--run_id',
                        help='Exp No.',
                        default=0,
                        type=int)
    parser.add_argument('--chkpt',
                        help='Checkpoint Number',
                        default="20",
                        type=int)
    parser.add_argument("--batch_size",
                        help='Batch Size',
                        default=32,
                        type=int)
    parser.add_argument('--csv_path',
                        help='Predictions CSV Path',
                        default="./predictions_csv",
                        type=str)
    args = parser.parse_args()

with open(f"run_args/{args.run_id}_args.pkl", 'rb') as f:
    model_dict = pickle.load(f)

os.makedirs(args.csv_path, exist_ok=True)

def run_epoch(dataloader,
              model,
              device):

    path_list = []
    pred_list = []

    for path, x in tqdm(dataloader):

        # Moving input to device
        x = x.to(device)

        # Running forward propagation
        y_hat = model(x)
        y_hat = y_hat.cpu()

        prob = torch.softmax(y_hat, dim=1)
        pred_class = torch.argmax(prob, dim=1)

        path_list.extend(path)
        pred_list.extend(pred_class.tolist())

        del x, y_hat, prob, pred_class
        gc.collect()

    return path_list, pred_list


# Load Test data
test_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=model_dict.image_size),
    v2.ToDtype(torch.float, scale=True),
])

test_dataset = GBMPathDataset(
    imgs_path_file=C.TEST_IMAGES_PATHS,
    transforms=test_transforms,
    func="test"
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2
)

model = get_model(model_dict)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
CHECKPOINT_PATH = f"./checkpoints/{args.run_id}/checkpoint{args.chkpt}.pt"
checkpoint = torch.load(CHECKPOINT_PATH)
# Load pre-trained weights
model.load_state_dict(checkpoint)

model = model.to(device)

# Testing
model.eval()

path_list, pred_list = run_epoch(
    test_dataloader,
    model,
    device)

pred_df = pd.DataFrame(
    {'SubjectID': path_list,
     'Prediction': pred_list
    })

pred_df.to_csv(f'{args.csv_path}/{args.run_id}-{args.chkpt}_preds.csv', encoding='utf-8', index=False)