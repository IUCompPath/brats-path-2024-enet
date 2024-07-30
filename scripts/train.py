import os
import argparse

# Torch Imports
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor

# Local Imports
from train_utils import run_epoch, plot_cm, EarlyStopping
import core_utils as Util

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
                        default=False,
                        action='store_true')
    parser.add_argument('--model',
                        help = 'Model to be used. 0-7 are efficient net models, 8 is densenet, 9 is resnet18, 10 is resnet34, 11 is resnet50',
                        default=4,
                        type=int)
    parser.add_argument("--batch_size",
                        help='Batch Size',
                        default=32,
                        type=int)
    parser.add_argument("--n_epochs",
                        help='Number of epochs',
                        default=50,
                        type=int)
    parser.add_argument('--loss',
                        help='Loss Function',
                        default="ce",
                        type=str)
    parser.add_argument('--opt',
                        help='Optimizer',
                        default="adam",
                        choices=["adam", "sgd"],
                        type=str)
    parser.add_argument('--lrs',
                        help='Learning Rate Scheduler',
                        default=None,
                        choices=["step", "multistep", "exp", None],
                        type=str)
    parser.add_argument('--lr',
                        help='Learning Rate',
                        default=1e-4,
                        type=float)
    parser.add_argument('--lmbda',
                        help='Lambda/Momentum',
                        default=1e-4,
                        type=float)
    parser.add_argument('--gamma',
                        help='Learning Rate Reduction Factor',
                        default=0.9,
                        type=float)
    parser.add_argument('--early_stop',
                        help='Early Stopping- set if true',
                        default=False,
                        action='store_true')
    parser.add_argument("--patience",
                        help='Early Stopping Patience',
                        default=10,
                        type=int)
    args = parser.parse_args()

    if args.model <= 7:
        args.image_size = (512, 512)
    else:
        args.image_size = (224, 224)

Util.save_args(args)

run_id = args.run_id

CKPT_PATH = os.path.join("./checkpoints/", f'{run_id}')
LOG_PATH = os.path.join("./logs", f'{run_id}')
PLOTS_PATH = f"./plots/{run_id}"

os.makedirs(CKPT_PATH, exist_ok=True)

# Logging
train_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "train"))
val_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "val"))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class_weights = Util.get_class_weights()
class_weights = class_weights.to(device)
loss_fn = Util.get_loss_fn(args, class_weights)

model = Util.get_model(args)
opt = Util.get_opt(args, model)
scheduler = Util.get_scheduler(args, opt)

model = model.to(device)

train_dataloader, val_dataloader = Util.create_dataloader(args)

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
    if scheduler != None:
        scheduler.step()

    Util.print_metrics(train_metrics, val_metrics, i)
    Util.log_values(train_logger, train_metrics, val_logger, val_metrics, i)

    # Check early stopping criteria
    if args.early_stop:
        early_stopping(val_metrics[0].mean(), model)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {i}")
            print(f"Model weights saved")
            break

    Util.save_model(model, CKPT_PATH, i)

    plot_cm(val_metrics[6], PLOTS_PATH, f"val_{i}.png")
    plot_cm(train_metrics[6], PLOTS_PATH, f"train_{i}.png")