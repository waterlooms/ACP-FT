import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, auc, hamming_loss
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Filter out UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def compute_metric_binary(labels, outputs, thres = 0.5):
    zero = 1e-10
    tn, fp, fn, tp = confusion_matrix(labels, outputs > thres, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn + zero)
    specificity = tn / (tn + fp + zero)
    precision = tp / (tp + fp + zero)
    recall = tp / (tp + fn + zero)
    f1score = 2 * (precision * recall) / (precision + recall + zero)
    mcc = matthews_corrcoef(labels, outputs > thres)
    fpr, tpr, _ = roc_curve(labels, outputs)
    _precision, _recall, _ = precision_recall_curve(labels, outputs)
    auroc = auc(fpr, tpr)
    auprc = auc(_recall, _precision)

    res = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1score': f1score,
        'MCC': mcc,
    }

    return res

def compute_metric(labels, outputs, thres = 0.5):
    _, num_label = outputs.shape
    dicts = []
    for idx in range(num_label):
        label, output = labels[:, idx], outputs[:, idx]
        dicts.append(compute_metric_binary(label, output, thres))

    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = [value]
            else:
                merged_dict[key].append(value)
    mean_dict = {}
    for k in merged_dict:
        mean_dict[k] = np.mean(merged_dict[k])
    hammingloss = hamming_loss(labels, outputs > thres)
    mean_dict['hamming_loss'] = hammingloss
    return mean_dict, merged_dict

from pyteomics import fasta
import pandas as pd

def convert_fasta_df(file_dir):
    dataset = fasta.read(file_dir)
    description, sequence = [], []
    for x in dataset:
        description.append(x.description)
        sequence.append(x.sequence)
    pd.DataFrame()
    df = pd.DataFrame({
        'label': description,
        'text': sequence,
    })
    return df