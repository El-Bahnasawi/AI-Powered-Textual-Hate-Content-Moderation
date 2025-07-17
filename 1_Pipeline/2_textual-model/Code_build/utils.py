from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
import wandb
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='weighted')
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None)

    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    # Log confusion matrix to W&B as an image
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=preds,
            class_names=["Non-Hate", "Hate"]
        )
    })

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'matthews_corrcoef': mcc,
        'precision_class_0': precision_per_class[0],
        'precision_class_1': precision_per_class[1],
        'recall_class_0': recall_per_class[0],
        'recall_class_1': recall_per_class[1],
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
    }
