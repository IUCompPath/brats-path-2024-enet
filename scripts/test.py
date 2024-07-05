import os

# Torch Imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

# Local Imports
from dataset import GBMPathDataset
from utility import run_epoch, plot_cm
import constants as C

# Load Test data
test_dataset = GBMPathDataset(
    imgs_path_prefix=C.IMAGES_PATH_PREFIX,
    imgs_path_file=C.TEST_IMAGES_PATHS,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=C.test_batch_size,
    shuffle=False,
    num_workers=2
)

model = EfficientNet.from_name('efficientnet-b4', num_classes=C.N_CLASSES)
opt = optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.lmbda)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define weighted cross-entropy loss
loss_fn = nn.CrossEntropyLoss()

# Load model
CHECKPOINT_FILE = f"checkpoint.pt"
CHECKPOINT_PATH = f"./checkpoints/{C.RUN_ID}/{CHECKPOINT_FILE}"
checkpoint = torch.load(CHECKPOINT_PATH)
# Load pre-trained weights
model.load_state_dict(checkpoint)

model = model.to(device)

# Testing
model.eval()

test_metrics = run_epoch(
    test_dataloader,
    model,
    device,
    loss_fn)

# Plot Confusion Matrix
plot_cm(test_metrics[6], C.PLOTS_PATH, "test.png")

print(
        f"Test: Loss - {test_metrics[0]}, " +
        f"Accuracy - {test_metrics[1]}, " +
        f"Specificity - {test_metrics[2].mean()}, " +
        f"Precision - {test_metrics[3].mean()}, " +
        f"Recall - {test_metrics[4].mean()}, " +
        f"F1 - {test_metrics[5].mean()}"
    )

for i in range(C.N_CLASSES):
    print(
        f"Class-{i}\n\t" +
        f"Specificity - {test_metrics[2][i]}, " +
        f"Precision - {test_metrics[3][i]}, " +
        f"Recall - {test_metrics[4][i]}, " +
        f"F1 - {test_metrics[5][i]}"
    )