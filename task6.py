"""
Mashq 6: K-NN xato naqshini talqin qilish
"""
import numpy as np
import pandas as pd

from dataset import EX6_LABELS, load_ex6_matrix


labels = EX6_LABELS
cm = load_ex6_matrix()
cm_df = pd.DataFrame(cm,
                     index=[f"Actual {l}" for l in labels],
                     columns=[f"Pred {l}" for l in labels])
print("Confusion matrix:")
print(cm_df, "\n")

diag = np.diag(cm)
recall = diag / cm.sum(axis=1)
precision = diag / cm.sum(axis=0)
errors = cm.sum(axis=1) - diag

metrics = pd.DataFrame({
    "Support": cm.sum(axis=1),
    "Correct": diag,
    "Errors": errors,
    "Recall": recall.round(3),
    "Precision": precision.round(3),
}, index=labels)
print(metrics.to_string(), "\n")

print(f"a) Eng yuqori recall: {metrics['Recall'].idxmax()} ({metrics['Recall'].max():.3f})")

worst = metrics["Errors"].idxmax()
i = labels.index(worst)
print(f"\nb) Eng ko'p adashtiriladigan klass: {worst} - {errors[i]} ta xato")
for j, l in enumerate(labels):
    if j != i and cm[i, j] > 0:
        print(f"   {worst} -> {l} deb bashorat: {cm[i, j]} ta")
print(f"   Shuningdek, boshqa klasslar ham {worst} deb noto'g'ri bashorat qilingan: "
      f"{cm[:, i].sum() - cm[i, i]} ta")


c = labels.index("CRITICAL")
tp = cm[c, c]
fp = cm[:, c].sum() - tp
fn = cm[c, :].sum() - tp
print(f"\nc) CRITICAL: TP={tp}, FP={fp}, FN={fn}")
print(f"   Precision = {tp}/({tp}+{fp}) = {tp / (tp + fp):.3f}")
print(f"   Recall    = {tp}/({tp}+{fn}) = {tp / (tp + fn):.3f}")


print("\nd) DEGRADED namunalar NORMAL va CRITICAL o'rtasida, 'chegara zonasi'da joylashgan:")
print("   - DEGRADED ikkala tomonga ham adashadi (5 -> NORMAL, 6 -> CRITICAL).")
print("   - NORMAL va CRITICAL bir-biri bilan hech qachon adashmaydi (cm[0,2] = cm[2,0] = 0),")
print("     ya'ni ular feature space'da bir-biridan uzoq.")
print("   - Demak DEGRADED klasterlari ikki qo'shni klass bilan ustma-ust tushadi; bu holat")
print("     tabiiy degradatsiya jarayonining oraliq bosqich ekanini aks ettiradi.")


print("\ne) Tekshirish kerak bo'lgan K-NN o'zgarishlari:")
print("   1) k ni tanlash (masalan, GridSearchCV + cross-validation).")
print("      Chegara zonasida kichik k shovqinga sezgir, katta k esa klasslarni silliqlab yuboradi.")
print("   2) weights='distance' - yaqinroq qo'shnilarga ko'proq og'irlik beradi,")
print("      chegara yaqinidagi nuqtalarda uzoqdagi boshqa klass ovozlarining ta'sirini kamaytiradi.")
print("   Qo'shimcha variantlar: boshqa distance metric (manhattan), feature selection yoki")
print("   DEGRADED ni ajratadigan yangi feature qo'shish, klasslar balansini tekshirish.")