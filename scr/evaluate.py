import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
import random
import csv
import os


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")
    bacc = balanced_accuracy_score(labels, prediction)

    return micro_f1, macro_f1, bacc

class EarlyStopping(object):
    def __init__(self, args, patience=10):

        self.filename = "model_im_"+str(args.im_ratio)+".pth"
        self.filename = args.log_dir  + "/" + self.filename
        self.patience = args.patience
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def step(self, acc, model):
        if self.best_acc is None:
            self.best_acc = acc
            self.save_checkpoint(model)
        elif  (acc < self.best_acc):
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if  (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop, self.counter, self.best_acc

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

def set_random_seed(seed=1024):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms=True



def record(args, results_mic,results_mac,results_bacc):
    write_record  =  [str(round(np.mean(results_mac)*100,2))+"+"+str(round(np.std(results_mac)*100,2)),
                      str(round(np.mean(results_mic)*100,2))+"+"+str(round(np.std(results_mic)*100,2)), 
                      str(round(np.mean(results_bacc)*100,2))+"+"+str(round(np.std(results_bacc)*100,2))]
    with open(args.csv_file ,'a+', newline='')as f:
        writer = csv.writer(f)
        writer.writerow(write_record) 