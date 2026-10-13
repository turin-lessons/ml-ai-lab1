# K-Nearest Neighbors — Laboratoriya ishi

Feature dominance, scaling, confusion matrix, precision, recall va natijalarni talqin qilish bo'yicha 6 ta mashq.
Har bir mashq avval **numpy** bilan qo'lda (formulalar bo'yicha) hisoblanadi, so'ng **scikit-learn** bilan tekshiriladi.
Natijalar 3 xonagacha yaxlitlanadi.

## Talablar

- Python **3.10+**
- numpy, pandas, scikit-learn (`requirements.txt` da)

## O'rnatish

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Ishga tushirish

Interaktiv menyu:

```bash
python main.py
```

| Kiritish     | Natija                                      |
|--------------|---------------------------------------------|
| `3`          | faqat 3-mashq                               |
| `1 4` / `1,4`| tanlangan mashqlar                          |
| `all` / `0`  | hamma mashqlar + OK/XATO xulosasi           |
| `q`          | chiqish                                     |

Menyusiz ishga tushirish:

```bash
python main.py all      # hammasi
python main.py 2 5      # faqat 2 va 5-mashq
python ex1_feature_dominance.py   # bitta faylni to'g'ridan-to'g'ri
```

## Loyiha tuzilmasi

```
.
├── main.py                       # interaktiv menyu, mashqlarni ishga tushiradi
├── dataset.py                    # barcha mashqlarning ma'lumotlari
├── ex1_feature_dominance.py      # Mashq 1
├── ex2_standardization.py        # Mashq 2
├── ex3_multiclass_confusion.py   # Mashq 3
├── ex4_precision_vs_recall.py    # Mashq 4
├── ex5_compare_models.py         # Mashq 5
├── ex6_error_pattern.py          # Mashq 6
├── requirements.txt
└── README.md
```

Barcha fayllar **bitta papkada** turishi kerak, chunki mashqlar `from dataset import ...` orqali ma'lumot oladi.

## Mashqlar

| # | Mavzu | Asosiy vositalar | Natija |
|---|-------|------------------|--------|
| 1 | Feature dominance (xom masofa) | `np.sqrt`, `KNeighborsClassifier` | k=3 → **NORMAL**; FlowRate masofaning ~100% ini egallaydi |
| 2 | Standartlashtirishdan keyin | `StandardScaler`, `KNeighborsClassifier` | k=3 → **DEGRADED**; qo'shnilar P5, P6, P4 |
| 3 | 3×3 confusion matrix | `confusion_matrix`, `precision_score`, `recall_score` | Accuracy 0.667; CRITICAL P=R=0.750; DEGRADED P=R=0.500 |
| 4 | Precision vs recall (binary) | `precision_score`, `recall_score`, `accuracy_score` | P=0.571, R=0.800, Acc=0.840; xavfsizlikda **recall** muhim |
| 5 | Ikki modelni solishtirish | pandas | A: P=0.500, R=0.900; B: P=0.778, R=0.700 |
| 6 | Xato naqshini talqin qilish | numpy | Eng yuqori recall CRITICAL (0.875); eng ko'p adashuvchi DEGRADED |

## Muhim eslatmalar

- **σ (standart og'ish)** — populyatsiya std (maxraj `n`), `StandardScaler` va `np.std(ddof=0)` bilan bir xil.
- Scaler **faqat training** ma'lumotiga `fit` qilinadi; yangi nasos Q faqat `transform` qilinadi.
- Mashq 2 da topshiriqda berilgan yaxlitlangan μ va σ ishlatiladi, shuning uchun sklearn bilan 3-xonada ±0.001 farq bo'lishi mumkin.
- Confusion matrixlarda **qatorlar = haqiqiy klass**, **ustunlar = bashorat qilingan klass**.