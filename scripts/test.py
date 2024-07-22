import argparse
from tqdm import tqdm
import pandas as pd
import pickle

# Torch Imports
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as nn
from torchvision.transforms import v2
from efficientnet_pytorch import EfficientNet

# Local Imports
from dataset import GBMPathDataset
import constants as C

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='This script trains the model'
    )
    parser.add_argument('--run_id',
                        help='Exp No.',
                        default=0,
                        type=int)
    # parser.add_argument('--run_file',
    #                     help = 'Path to pickle file with arguments',
    #                     default=7,
    #                     type=int)
    parser.add_argument('--chkpt_file',
                        help='Checkpoint File Name',
                        default="checkpoint.pt",
                        type=str)
    parser.add_argument("--batch_size",
                        help='Batch Size',
                        default=32,
                        type=int)
    parser.add_argument('--csv_path',
                        help='Predictions CSV Path',
                        default="./predictions_csv",
                        type=tuple)
    args = parser.parse_args()

with open(f"run_args/{args.run_id}_args.pkl", 'rb') as f:
    model_dict = pickle.load(f)

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

        prob = nn.softmax(y_hat, dim=1)
        _, pred_class = torch.max(prob, dim=1)

        path_list.extend(path)
        pred_list.extend(pred_class)

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
    shuffle=False
)

model = EfficientNet.from_name(f'efficientnet-b{model_dict.enet_model}', num_classes=C.N_CLASSES)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
CHECKPOINT_PATH = f"./checkpoints/{args.run_id}/{args.chkpt_file}"
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

pred_df.to_csv(f'{args.csv_path}/{args.run_id}_test_preds.csv', encoding='utf-8', index=False)