"""
Mashq 4: Precision va recall farqi (binary)
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from dataset import load_ex4_matrix


cm = load_ex4_matrix()
print("Berilgan confusion matrix:")
print(cm, "\n")

TP, FN = cm.iloc[0, 0], cm.iloc[0, 1]
FP, TN = cm.iloc[1, 0], cm.iloc[1, 1]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"a) Precision = {TP}/({TP}+{FP}) = {precision:.3f}")
print(f"b) Recall    = {TP}/({TP}+{FN}) = {recall:.3f}")
print(f"c) Accuracy  = ({TP}+{TN})/{TP + TN + FP + FN} = {accuracy:.3f}")


y_true = np.array(["C"] * (TP + FN) + ["NC"] * (FP + TN))
y_pred = np.array(["C"] * TP + ["NC"] * FN + ["C"] * FP + ["NC"] * TN)
print("\nsklearn tekshiruvi:")
print("   confusion_matrix:\n", confusion_matrix(y_true, y_pred, labels=["C", "NC"]))
print(f"   precision = {precision_score(y_true, y_pred, pos_label='C'):.3f}")
print(f"   recall    = {recall_score(y_true, y_pred, pos_label='C'):.3f}")
print(f"   accuracy  = {accuracy_score(y_true, y_pred):.3f}")


print("\nd) Xavfsizlik monitoringida RECALL muhimroq.")
print(f"   Recall = {recall:.3f}: 20 ta haqiqiy xavfli mashinadan {FN} tasi o'tkazib yuborilgan (FN).")
print("   FN - xavfli mashina 'sog'lom' deb qoldirilishi, bu avariya/xavfga olib keladi.")
print(f"   FP ({FP} ta) - sog'lom mashinani ortiqcha tekshirish, bu faqat vaqt/xarajat.")
print("   Shuning uchun FN ni kamaytirish, ya'ni recall'ni oshirish ustuvor.")
print(f"   Accuracy ({accuracy:.3f}) bu yerda chalg'itadi: klasslar nobalans (20 vs 80),")
print("   model ko'pchilik klassni to'g'ri topib ham yuqori accuracy olishi mumkin.")