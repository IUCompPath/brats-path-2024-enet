from tqdm import tqdm
import torch
from metrics import get_metrics
from sklearn.metrics import confusion_matrix
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import constants as C

# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, path='checkpoint.pt'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_max = float('-inf')

    def __call__(self, f1, model):
        if self.best_score is None:
            self.best_score = f1
            self.save_checkpoint(f1, model)
        elif f1 < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = f1
            self.save_checkpoint(f1, model)
            self.counter = 0

    def save_checkpoint(self, f1, model):
        '''Saves model when F1 increases.'''
        torch.save(model.state_dict(), self.path)
        self.f1_max = f1

def create_paths_list(train_data_path):#, val_data_path):
    paths_list = []
    train_classes = os.listdir(train_data_path)
    for cls in train_classes:
        class_path = os.path.join(train_data_path, cls)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            paths_list.append(os.path.join(class_path, image_name))
    # val_classes = os.listdir(val_data_path)
    # for cls in val_classes:
    #     class_path = os.path.join(val_data_path, cls)
    #     image_names = os.listdir(class_path)
    #     for image_name in image_names:
    #         paths_list.append(os.path.join(class_path, image_name))
    random.shuffle(paths_list)
    return paths_list

def plot_cm(cm, path, file_name):
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot
    os.makedirs(path, exist_ok=True)
    plot_path = os.path.join(path, file_name)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def run_epoch(dataloader,
              model,
              device,
              loss_fn,
              logger=None,
              opt=None,
              n_classes=5,
              step=0):

    loss_list = []
    y_list = []
    y_hat_list = []

    for x, y in tqdm(dataloader):

        # Moving input to device
        x = x.to(device)
        y = y.to(device)

        # Running forward propagation
        y_hat = model(x)
        # Compute loss
        loss = loss_fn(y_hat, y)

        # detach removes y_hat from the original computational graph which might be on gpu.
        y_hat = y_hat.detach().cpu()
        y = y.cpu()

        y_list.append(y)
        y_hat_list.append(y_hat)

        if opt is not None:
            # Make all gradients zero.
            opt.zero_grad()

            # Run backpropagation
            loss.backward()

            # Update parameters
            opt.step()

        # Freeing up memory
        del x

        loss_list.append(loss.item())

        # Compute metrics
        acc = get_metrics(y_hat, y, metric="accuracy")
        spec = get_metrics(y_hat, y, metric="specificity")
        prec = get_metrics(y_hat, y, metric="precision")
        rec = get_metrics(y_hat, y, metric="recall")
        f1 = get_metrics(y_hat, y, metric="f1")

        if logger is not None:

            logger.add_scalar(f"loss", loss.item(), step)
            logger.add_scalar(f"accuracy", acc, step)
            logger.add_scalar(f"specificity", spec.mean(), step)
            logger.add_scalar(f"precision", prec.mean(), step)
            logger.add_scalar(f"recall", rec.mean(), step)
            logger.add_scalar(f"f1", f1.mean(), step)

            for j in range(n_classes):
                logger.add_scalar(f"precision/{j}", prec[j], step)
                logger.add_scalar(f"recall/{j}", rec[j], step)
                logger.add_scalar(f"f1/{j}", f1[j], step)
                logger.add_scalar(f"specificity/{j}", spec[j], step)

            step += 1

    # Concatenating each batch's logits and gt into a single row
    y_hat_list = torch.concat(y_hat_list)
    y_list = torch.concat(y_list)

    # Compute Confusion Matrix
    _, pred_labels = torch.max(y_hat_list, 1)

    cm = confusion_matrix(y_list, pred_labels)

    # Compute final metrics
    acc_final = get_metrics(y_hat_list, y_list, metric="accuracy")
    spec_final = get_metrics(y_hat_list, y_list, metric="specificity")
    prec_final = get_metrics(y_hat_list, y_list, metric="precision")
    rec_final = get_metrics(y_hat_list, y_list, metric="recall")
    f1_final = get_metrics(y_hat_list, y_list, metric="f1")

    avg_loss = torch.Tensor(loss_list).mean()

    return avg_loss, acc_final, spec_final, prec_final, rec_final, f1_final, cm