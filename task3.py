"""
Mashq 3: Multiclass confusion matrix
Qatorlar = haqiqiy klass, ustunlar = bashorat qilingan klass
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from dataset import EX3_LABELS, load_ex3_results


results = load_ex3_results()
labels = EX3_LABELS
y_true = results["Actual"]
y_pred = results["Predicted"]


cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm,
                     index=[f"Actual {l}" for l in labels],
                     columns=[f"Pred {l}" for l in labels])
print("a) Confusion matrix:")
print(cm_df)

acc_manual = np.trace(cm) / cm.sum()
print(f"\nb) Accuracy (qo'lda) = {np.trace(cm)}/{cm.sum()} = {acc_manual:.3f}")
print(f"   Accuracy (sklearn) = {accuracy_score(y_true, y_pred):.3f}")


def one_vs_rest(cm, labels, cls):
    """Bitta klass uchun TP, FP, FN, TN ni matritsadan topadi."""
    i = labels.index(cls)
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    return tp, fp, fn, tn


def report(cls, name):
    tp, fp, fn, tn = one_vs_rest(cm, labels, cls)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    p_sk = precision_score(y_true, y_pred, labels=[cls], average=None)[0]
    r_sk = recall_score(y_true, y_pred, labels=[cls], average=None)[0]
    print(f"   {name}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"   Precision = {tp}/({tp}+{fp}) = {precision:.3f}  (sklearn: {p_sk:.3f})")
    print(f"   Recall    = {tp}/({tp}+{fn}) = {recall:.3f}  (sklearn: {r_sk:.3f})")


print("\nc) CRITICAL - qiziqish klassi:")
report("C", "CRITICAL")

print("\nd) DEGRADED - qiziqish klassi:")
report("D", "DEGRADED")