"""
Mashq 5: Ikki K-NN modelini solishtirish (CRITICAL klassi)
"""
from dataset import load_ex5_models

models = load_ex5_models()


models["Precision"] = (models["TP"] / (models["TP"] + models["FP"])).round(3)
models["Recall"] = (models["TP"] / (models["TP"] + models["FN"])).round(3)
models["Accuracy"] = ((models["TP"] + models["TN"]) / models[["TP", "FN", "FP", "TN"]].sum(axis=1)).round(3)

print("a) CRITICAL klassi bo'yicha metrikalar:")
print(models.to_string(float_format="{:.3f}".format))

best_recall = models["Recall"].idxmax()
print(f"\nb) Critical mashinani o'tkazib yuborish qimmat bo'lsa: {best_recall}")
print(f"   Sabab: recall yuqori ({models.loc[best_recall, 'Recall']:.3f}), FN atigi "
      f"{models.loc[best_recall, 'FN']} ta.")


best_precision = models["Precision"].idxmax()
print(f"\nc) False alarm qimmat bo'lsa: {best_precision}")
print(f"   Sabab: precision yuqori ({models.loc[best_precision, 'Precision']:.3f}), FP atigi "
      f"{models.loc[best_precision, 'FP']} ta.")

print("\nd) 'A yaxshiroq' yoki 'B yaxshiroq' deyish to'liq emas, chunki:")
print("   - Model A ko'proq xavfli mashinani topadi, lekin ko'p yolg'on signal beradi.")
print("   - Model B kam yolg'on signal beradi, lekin ko'proq xavfli mashinani o'tkazib yuboradi.")
print("   - Bu precision-recall trade-off. Qaysi model 'yaxshi' ekanligi FN va FP xatolarining")
print("     real narxiga (biznes/xavfsizlik konteksti) bog'liq.")
print(f"   - Accuracy ham hal qilmaydi: A={models.loc['Model A', 'Accuracy']:.3f}, "
      f"B={models.loc['Model B', 'Accuracy']:.3f}, ammo bu qaysi xato qimmatligini ko'rsatmaydi.")