import os
import pickle
import constants as C
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import torch.optim.lr_scheduler as LRS
from torchvision.transforms import v2
from dataset import GBMPathDataset
from torch.utils.data import DataLoader

def save_args(args):
    with open(f'run_args/{args.run_id}_args.pkl', 'wb') as f:
        pickle.dump(args, f)

def get_class_weights():
    CLASS_COUNTS = []
    for cls in C.CLASS_NAMES:
        class_path = os.path.join(C.IMAGES_PATH_PREFIX, cls)
        count = len(os.listdir(class_path))
        CLASS_COUNTS.append(count)

    total_samples = sum(CLASS_COUNTS)
    class_weights = [total_samples / count for count in CLASS_COUNTS]
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return class_weights

def get_loss_fn(args, class_weights):
    if args.loss == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn

def get_model(args):
    model = EfficientNet.from_pretrained(f'efficientnet-b{args.enet_model}', num_classes=C.N_CLASSES)
    return model

def get_opt(args, model):
    if args.opt == "adam":
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lmbda)
    elif args.opt == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lmbda)
    return opt

def get_scheduler(args, opt):
    lrs = args.lrs
    if lrs == "step":
        scheduler = LRS.StepLR(opt,
                            step_size = 10,
                            gamma = args.gamma)
    elif lrs == "multistep":
        scheduler = LRS.MultiStepLR(opt,
                                    milestones = [15, 20, 25, 30],
                                    gamma = args.gamma)
    elif lrs == "exp":
        scheduler = LRS.ExponentialLR(opt,
                                      gamma = args.gamma)
    else:
        scheduler = None
    return scheduler

def create_dataloader(args):
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

    # VALIDATION
    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=args.image_size),
        v2.ToDtype(torch.float, scale=True),
    ])
    val_dataset = GBMPathDataset(
        imgs_path_file=C.VAL_IMAGES_PATHS,
        func="val",
        transforms=val_transforms)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)

    return train_dataloader, val_dataloader

def print_metrics(train_metrics, val_metrics, epoch):
    print(f"Epoch {epoch}:")
    print(
        f"Train: Loss - {train_metrics[0]}, " +
        f"Accuracy - {train_metrics[1]}, " +
        f"Precision - {train_metrics[3].mean()}, " +
        f"Recall - {train_metrics[4].mean()}, " +
        f"F1 - {train_metrics[5].mean()}"
    )

    print(
        f"Val: Loss - {val_metrics[0]}, " +
        f"Accuracy - {val_metrics[1]}, " +
        f"Precision - {val_metrics[3].mean()}, " +
        f"Recall - {val_metrics[4].mean()}, " +
        f"F1 - {val_metrics[5].mean()}"
    )

    for j in range(C.N_CLASSES):
        print(
            f"Val metrics for Class: {j}\n\t" +
            f"Precision - {val_metrics[3][j]}, " +
            f"Recall - {val_metrics[4][j]}, " +
            f"F1 - {val_metrics[5][j]}, "
        )

def log_values(train_logger, train_metrics, val_logger, val_metrics, epoch):
    train_logger.add_scalar(f"epoch/loss", train_metrics[0], epoch)
    train_logger.add_scalar(f"epoch/accuracy", train_metrics[1], epoch)
    train_logger.add_scalar(f"epoch/precision", train_metrics[3].mean(), epoch)
    train_logger.add_scalar(f"epoch/recall", train_metrics[4].mean(), epoch)
    train_logger.add_scalar(f"epoch/f1", train_metrics[5].mean(), epoch)

    val_logger.add_scalar(f"epoch/loss", val_metrics[0], epoch)
    val_logger.add_scalar(f"epoch/accuracy", val_metrics[1], epoch)
    val_logger.add_scalar(f"epoch/precision", val_metrics[3].mean(), epoch)
    val_logger.add_scalar(f"epoch/recall", val_metrics[4].mean(), epoch)
    val_logger.add_scalar(f"epoch/f1", val_metrics[5].mean(), epoch)

    for j in range(C.N_CLASSES):

        train_logger.add_scalar(f"{j}/epoch/precision", train_metrics[3][j], epoch)
        train_logger.add_scalar(f"{j}/epoch/recall", train_metrics[4][j], epoch)
        train_logger.add_scalar(f"{j}/epoch/f1", train_metrics[5][j], epoch)

        val_logger.add_scalar(f"{j}/epoch/precision", val_metrics[3][j], epoch)
        val_logger.add_scalar(f"{j}/epoch/recall", val_metrics[4][j], epoch)
        val_logger.add_scalar(f"{j}/epoch/f1", val_metrics[5][j], epoch)

def save_model(model, path, epoch):
    torch.save(model.state_dict(), os.path.join(path, f"checkpoint{epoch}.pt"))