"""
Mashq 2: Standartlashtirishdan keyin nima o'zgaradi
X_tilde = (X - mu) / sigma   (sigma - populyatsiya std, maxraj n)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from dataset import FEATURES, GIVEN_MU, GIVEN_SIGMA, load_pumps, load_query


train = load_pumps()
features = FEATURES
X = train[features].to_numpy(dtype=float)
y = train["Class"].to_numpy()
Q = load_query()


mu = GIVEN_MU
sigma = GIVEN_SIGMA


scaler = StandardScaler().fit(X)
print("StandardScaler hisoblagan mu   :", scaler.mean_.round(3))
print("StandardScaler hisoblagan sigma:", scaler.scale_.round(3))

print("numpy qo'lda sigma (ddof=0)    :", X.std(axis=0, ddof=0).round(3))


X_std = (X - mu) / sigma
Q_std = (Q - mu) / sigma

train["Vib_std"] = X_std[:, 0].round(3)
train["Flow_std"] = X_std[:, 1].round(3)
print("\na) Standartlashgan qiymatlar:")
print(train[["Pump", "Vib_std", "Flow_std", "Class"]].to_string(index=False))
print(f"   Q_std = ({Q_std[0]:.3f}, {Q_std[1]:.3f})")


train["Distance_std"] = np.sqrt(((X_std - Q_std) ** 2).sum(axis=1)).round(3)
print("\nb) Standartlashgan masofalar:")
print(train[["Pump", "Class", "Distance_std"]].to_string(index=False))


k = 3
neighbors = train.nsmallest(k, "Distance_std")
pred_manual = neighbors["Class"].mode()[0]
print(f"\nc) k={k} eng yaqin qo'shnilar:")
print(neighbors[["Pump", "Class", "Distance_std"]].to_string(index=False))
print(f"   Qo'lda bashorat: y_hat(Q) = {pred_manual}")


knn = KNeighborsClassifier(n_neighbors=k).fit(scaler.transform(X), y)
pred_sklearn = knn.predict(scaler.transform([Q]))[0]
print(f"   sklearn bashorati:  y_hat(Q) = {pred_sklearn}")

raw_dist = np.sqrt(((X - Q) ** 2).sum(axis=1))
raw_nn = train["Pump"].to_numpy()[np.argsort(raw_dist)[:k]]
std_nn = neighbors["Pump"].to_numpy()
print("\nd) Solishtirish:")
print(f"   Xom qo'shnilar          : {list(raw_nn)} -> NORMAL")
print(f"   Standartlashgan qo'shnilar: {list(std_nn)} -> {pred_manual}")
print("   Qo'shnilar to'plami butunlay almashdi. Bu xom ma'lumotdagi natija ma'lumotdagi")
print("   haqiqiy o'xshashlikdan emas, FlowRate'ning katta shkalasidan kelib chiqqanini ko'rsatadi.")
print("   Scalingdan keyin har ikki feature teng hissa qo'shadi va Vibration'dagi")
print("   kuchli signal (Q ~ DEGRADED) ko'rinadigan bo'ladi. K-NN masofaga asoslangani uchun")
print("   scaling unda majburiy preprocessing qadam hisoblanadi.")
print("\nEslatma: berilgan sigma yaxlitlangan, shuning uchun sklearn bilan 3-xonada")
print("~0.001 farq bo'lishi mumkin.")