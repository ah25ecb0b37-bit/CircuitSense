"""Per-class metrics and confusion matrix helper."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ..data.simulator import FAULT_CLASSES

def print_report(y_true, y_pred):
    names = [FAULT_CLASSES[i] for i in range(len(FAULT_CLASSES))]
    print(classification_report(y_true, y_pred, target_names=names))

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
