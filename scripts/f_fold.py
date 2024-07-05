import os
import argparse

# Torch Imports
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

# Local Imports
import constants as C
from dataset import GBMPathDataset
from utility import run_epoch, plot_cm, EarlyStopping

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='This script runs one fold of training and tests it on the best model.'
    )
    parser.add_argument('-f','--fold',
                        help='Fold number',
                        default=0,
                        type=int)
    args = parser.parse_args()
    fold = args.fold

os.makedirs(C.CKPT_PATH, exist_ok=True)

# Logging
train_logger = SummaryWriter(log_dir=os.path.join(C.LOG_PATH, "train"))
val_logger = SummaryWriter(log_dir=os.path.join(C.LOG_PATH, "val"))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load train data
train_dataset = GBMPathDataset(
    imgs_path_file=C.TRAIN_IMAGES_PATHS,
    fold=fold,
    func="train",
    transforms=C.train_transforms
)

train_dataloader = DataLoader(train_dataset,
                            batch_size=C.batch_size,
                            shuffle=True,
                            num_workers=2)

# Load validation data
val_dataset = GBMPathDataset(
    imgs_path_file=C.TRAIN_IMAGES_PATHS,
    fold=fold,
    func="val",
    transforms=C.train_transforms
)

val_dataloader = DataLoader(val_dataset,
                            batch_size=C.batch_size,
                            shuffle=True,
                            num_workers=2)

# Calculate weights for each class and define weighted cross-entropy loss
class_counts = [2902, 1233, 409, 2511, 821, 325]
total_samples = sum(class_counts)
class_weights = [total_samples / count for count in class_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = class_weights.to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=C.N_CLASSES)
opt = optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.lmbda)
model = model.to(device)

# Logging model graph
x, _ = next(iter(train_dataloader))
train_logger.add_graph(model, x.to(device))

# Initialize early stopping object
early_stopping = EarlyStopping(
    patience=C.patience,
    path=os.path.join(C.CKPT_PATH, f"checkpoint_{fold}.pt")
)

# Training
for i in range(C.n_epochs):
    model.train()
    # run one epoch of training
    train_metrics = run_epoch(
        train_dataloader,
        model,
        device,
        loss_fn,
        train_logger,
        opt=opt,
        step=i*len(train_dataloader)
    )

    model.eval()
    # run one epoch of validation
    val_metrics = run_epoch(
        val_dataloader,
        model,
        device,
        loss_fn,
        val_logger,
        step=i*len(val_dataloader)
    )

    print(f"Fold {fold}, Epoch {i}:")
    print(
        f"Train: Loss - {train_metrics[0]}, " +
        f"Accuracy - {train_metrics[1]}, " +
        f"Specificity - {train_metrics[2].mean()}, " +
        f"Precision - {train_metrics[3].mean()}, " +
        f"Recall - {train_metrics[4].mean()}, " +
        f"F1 - {train_metrics[5].mean()}"
    )

    print(
        f"Val: Loss - {val_metrics[0]}, " +
        f"Accuracy - {val_metrics[1]}, " +
        f"Specificity - {val_metrics[2].mean()}, " +
        f"Precision - {val_metrics[3].mean()}, " +
        f"Recall - {val_metrics[4].mean()}, " +
        f"F1 - {val_metrics[5].mean()}"
    )

    # Logging per epoch for training and validation metrics
    # Because of equal no. of data points on the graph
    train_logger.add_scalar(f"epoch_f{fold}/loss", train_metrics[0], i)
    train_logger.add_scalar(f"epoch_f{fold}/accuracy", train_metrics[1], i)
    train_logger.add_scalar(f"epoch_f{fold}/specificity", train_metrics[2].mean(), i)
    train_logger.add_scalar(f"epoch_f{fold}/precision", train_metrics[3].mean(), i)
    train_logger.add_scalar(f"epoch_f{fold}/recall", train_metrics[4].mean(), i)
    train_logger.add_scalar(f"epoch_f{fold}/f1", train_metrics[5].mean(), i)

    val_logger.add_scalar(f"epoch_f{fold}/loss", val_metrics[0], i)
    val_logger.add_scalar(f"epoch_f{fold}/accuracy", val_metrics[1], i)
    val_logger.add_scalar(f"epoch_f{fold}/specificity", val_metrics[2].mean(), i)
    val_logger.add_scalar(f"epoch_f{fold}/precision", val_metrics[3].mean(), i)
    val_logger.add_scalar(f"epoch_f{fold}/recall", val_metrics[4].mean(), i)
    val_logger.add_scalar(f"epoch_f{fold}/f1", val_metrics[5].mean(), i)

    for j in range(C.N_CLASSES):
        print(
            f"Class-{j}\n\t" +
            f"Specificity - {val_metrics[2][j]}, " +
            f"Precision - {val_metrics[3][j]}, " +
            f"Recall - {val_metrics[4][j]}, " +
            f"F1 - {val_metrics[5][j]}"
        )

    # Check early stopping criteria
    early_stopping(val_metrics[5].mean(), model)

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {i} for fold {fold}\n")
        print(f"Model weights saved")
        break

plot_cm(val_metrics[6], C.PLOTS_PATH, f"val_{fold}.png")


# ----------------Starting testing-----------------
print("Testing model on test data")

# Load Test data
test_dataset = GBMPathDataset(
    imgs_path_file=C.TEST_IMAGES_PATHS,
    fold=fold,
    func="test"
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
CHECKPOINT_PATH = f"{C.CKPT_PATH}/checkpoint_{fold}"
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
plot_cm(test_metrics[6], C.PLOTS_PATH, f"{fold}_test.png")

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
