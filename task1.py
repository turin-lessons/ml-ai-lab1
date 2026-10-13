"""
Mashq 1: K-NN'da feature dominance
Masofa xom (scale qilinmagan) ma'lumotda hisoblanadi.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from dataset import FEATURES, load_pumps, load_query


train = load_pumps()
features = FEATURES
X = train[features].to_numpy(dtype=float)
y = train["Class"].to_numpy()
Q = load_query()


diff = X - Q
train["d_vib^2"] = (diff[:, 0] ** 2).round(3)
train["d_flow^2"] = (diff[:, 1] ** 2).round(3)
train["Distance"] = np.sqrt((diff ** 2).sum(axis=1)).round(3)

print("a) Q dan har bir nasosgacha xom masofa:")
print(train[["Pump", "Vibration", "FlowRate", "Class", "d_vib^2", "d_flow^2", "Distance"]]
      .to_string(index=False))

k = 3
neighbors = train.nsmallest(k, "Distance")
pred_manual = neighbors["Class"].mode()[0]

print(f"\nb) k={k} eng yaqin qo'shnilar:")
print(neighbors[["Pump", "Class", "Distance"]].to_string(index=False))
print(f"   Qo'lda bashorat: y_hat(Q) = {pred_manual}")


knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train[features], y)
pred_sklearn = knn.predict(pd.DataFrame([Q], columns=features))[0]
print(f"   sklearn bashorati:  y_hat(Q) = {pred_sklearn}")


ranges = train[features].max() - train[features].min()
flow_share = (diff[:, 1] ** 2) / (diff ** 2).sum(axis=1) * 100

print("\nc) Feature diapazonlari (max - min):")
print(f"   Vibration: {ranges['Vibration']:.3f} mm/s")
print(f"   FlowRate : {ranges['FlowRate']:.3f} L/min")
print("   Har bir masofa kvadratida FlowRate ulushi (%):")
for p, s in zip(train["Pump"], flow_share):
    print(f"     {p}: {s:.3f}%")
print("   Xulosa: FlowRate dominant - uning farqlari yuzlab birlikda,")
print("   Vibration farqlari esa ~0.1-3.6 oralig'ida, shuning uchun masofaga deyarli ta'sir qilmaydi.")


print("\nd) Izoh:")
print("   Formula matematik jihatdan to'g'ri, lekin u faqat sonlarning kattaligini ko'radi,")
print("   o'lchov birligini emas. Q ning vibratsiyasi (4.6) DEGRADED nasoslarga (4.0-5.0)")
print("   juda yaqin, ammo bu ma'lumot FlowRate farqlari ichida 'yo'qoladi'.")
print("   Natijada model aslida birlik/shkala tanloviga qarab qaror qiladi, fizik o'xshashlikka emas.")