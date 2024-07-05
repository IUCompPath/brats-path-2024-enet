import os
import argparse

# Torch Imports
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import v2

# Local Imports
import constants as C
from dataset import GBMPathDataset
from utility import run_epoch, plot_cm, EarlyStopping

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='This script trains the model'
    )
    parser.add_argument('--run_id',
                        help='Exp No.',
                        default=0,
                        type=int)
    parser.add_argument('--transforms',
                        help = 'set if use transforms',
                        default=True,
                        action='store_true')
    parser.add_argument('--enet_model',
                        help = 'Efficient Net Model to be used',
                        default=3,
                        type=int)
    parser.add_argument("--batch_size",
                        help='Batch Size',
                        default=32,
                        type=int)
    parser.add_argument("--n_epochs",
                        help='Number of epochs',
                        default=100,
                        type=int)
    parser.add_argument('--image_size',
                        help='Image Size',
                        default=(512, 512),
                        type=tuple)
    parser.add_argument('--lr',
                        help='Learning Rate',
                        default=1e-4,
                        type=float)
    parser.add_argument('--lmbda',
                        help='Lambda/Momentum',
                        default=1e-4,
                        type=float)
    parser.add_argument('--early_stop',
                        help='Early Stopping',
                        default=True,
                        action='store_true')
    parser.add_argument("--patience",
                        help='Early Stopping Patience',
                        default=15,
                        type=int)
    args = parser.parse_args()

CKPT_PATH = os.path.join("./checkpoints/", f'{args.run_id}')
LOG_PATH = os.path.join("./logs", f'{args.run_id}')
PLOTS_PATH = f"./plots/{args.run_id}"

os.makedirs(CKPT_PATH, exist_ok=True)

# Logging
train_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "train"))
# val_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "val"))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASS_COUNTS = []
for cls in C.CLASS_NAMES:
    class_path = os.path.join(C.IMAGES_PATH_PREFIX, cls)
    count = len(os.listdir(class_path))
    CLASS_COUNTS.append(count)

total_samples = sum(CLASS_COUNTS)
class_weights = [total_samples / count for count in CLASS_COUNTS]
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = C.class_weights.to(device)
# Define weighted cross-entropy loss
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

model = EfficientNet.from_pretrained(f'efficientnet-b{args.enet_model}', num_classes=C.N_CLASSES)
opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lmbda)

model = model.to(device)

# Load train data
if args.transforms:
    train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=args.image_size),
    v2.RandomRotation(degrees=20),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float, scale=True),
    ])
else:
    train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=args.image_size),
    v2.ToDtype(torch.float, scale=True),
    ])
train_dataset = GBMPathDataset(
    imgs_path_file=C.TRAIN_IMAGES_PATHS,
    func="train",
    transforms=train_transforms,
)
train_dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2)

# Load validation data
# val_transforms = v2.Compose([
#     v2.ToImage(),
#     v2.Resize(size=args.image_size),
#     v2.ToDtype(torch.float, scale=True),
# ])
# val_dataset = GBMPathDataset(
#     imgs_path_file=C.TEST_IMAGES_PATHS,
#     func="train",
#     transforms=val_transforms)
# val_dataloader = DataLoader(val_dataset,
#                             batch_size=C.batch_size,
#                             shuffle=True,
#                             num_workers=2)

# Logging model graph
x, _ = next(iter(train_dataloader))
train_logger.add_graph(model, x.to(device))

# Initialize early stopping object
if args.early_stop:
    early_stopping = EarlyStopping(
    patience=args.patience,
    path=os.path.join(CKPT_PATH, f"checkpoint.pt")
    )

# Training
for i in range(args.n_epochs):
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

    # model.eval()
    # # run one epoch of validation
    # val_metrics = run_epoch(
    #     val_dataloader,
    #     model,
    #     device,
    #     loss_fn,
    #     val_logger,
    #     step=i*len(val_dataloader)
    # )

    print(f"Epoch {i}:")
    print(
        f"Train: Loss - {train_metrics[0]}, " +
        f"Accuracy - {train_metrics[1]}, " +
        f"Specificity - {train_metrics[2].mean()}, " +
        f"Precision - {train_metrics[3].mean()}, " +
        f"Recall - {train_metrics[4].mean()}, " +
        f"F1 - {train_metrics[5].mean()}"
    )

    # print(
    #     f"Val: Loss - {val_metrics[0]}, " +
    #     f"Accuracy - {val_metrics[1]}, " +
    #     f"Specificity - {val_metrics[2].mean()}, " +
    #     f"Precision - {val_metrics[3].mean()}, " +
    #     f"Recall - {val_metrics[4].mean()}, " +
    #     f"F1 - {val_metrics[5].mean()}"
    # )

    # Logging per epoch so that traning and val can be compared properly
    # Because of equal no. of data points on the graph
    train_logger.add_scalar(f"epoch/loss", train_metrics[0], i)
    train_logger.add_scalar(f"epoch/accuracy", train_metrics[1], i)
    train_logger.add_scalar(f"epoch/specificity", train_metrics[2].mean(), i)
    train_logger.add_scalar(f"epoch/precision", train_metrics[3].mean(), i)
    train_logger.add_scalar(f"epoch/recall", train_metrics[4].mean(), i)
    train_logger.add_scalar(f"epoch/f1", train_metrics[5].mean(), i)

    # val_logger.add_scalar(f"epoch/loss", val_metrics[0], i)
    # val_logger.add_scalar(f"epoch/accuracy", val_metrics[1], i)
    # val_logger.add_scalar(f"epoch/specificity", val_metrics[2].mean(), i)
    # val_logger.add_scalar(f"epoch/precision", val_metrics[3].mean(), i)
    # val_logger.add_scalar(f"epoch/recall", val_metrics[4].mean(), i)
    # val_logger.add_scalar(f"epoch/f1", val_metrics[5].mean(), i)

    for j in range(C.N_CLASSES):
        # print(
        #     f"Val metrics for Class: {j}\n\t" +
        #     f"Precision - {val_metrics[3][j]}, " +
        #     f"Recall - {val_metrics[4][j]}, " +
        #     f"F1 - {val_metrics[5][j]}, " +
        #     f"Specificity - {val_metrics[2][j]}"
        # )

        train_logger.add_scalar(f"epoch/precision/{j}", train_metrics[3][j], i)
        train_logger.add_scalar(f"epoch/recall/{j}", train_metrics[4][j], i)
        train_logger.add_scalar(f"epoch/f1/{j}", train_metrics[5][j], i)
        train_logger.add_scalar(f"epoch/specificity/{j}", train_metrics[2][j], i)

        # val_logger.add_scalar(f"epoch/precision/{j}", val_metrics[3][j], i)
        # val_logger.add_scalar(f"epoch/recall/{j}", val_metrics[4][j], i)
        # val_logger.add_scalar(f"epoch/f1/{j}", val_metrics[5][j], i)
        # val_logger.add_scalar(f"epoch/specificity/{j}", val_metrics[2][j], i)

    # Check early stopping criteria
    if args.early_stop:
        early_stopping(train_metrics[5].mean(), model)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {i}")
            print(f"Model weights saved")
            break

    torch.save(model.state_dict(), os.path.join(CKPT_PATH, f"checkpoint{i}.pt"))

plot_cm(train_metrics[6], PLOTS_PATH, "train.png")