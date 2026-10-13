"""
Barcha mashqlar uchun umumiy ma'lumotlar.
Har bir funksiya har chaqirilganda yangi (toza) nusxa qaytaradi,
shuning uchun skriptlar ustun qo'shsa ham bir-biriga ta'sir qilmaydi.
"""
import numpy as np
import pandas as pd


FEATURES = ["Vibration", "FlowRate"]


def load_pumps() -> pd.DataFrame:
    """P1..P6 training nasoslari."""
    return pd.DataFrame({
        "Pump":      ["P1", "P2", "P3", "P4", "P5", "P6"],
        "Vibration": [1.0, 2.0, 1.5, 4.0, 5.0, 4.5],
        "FlowRate":  [800, 900, 850, 200, 300, 250],
        "Class":     ["NORMAL", "NORMAL", "NORMAL", "DEGRADED", "DEGRADED", "DEGRADED"],
    })


def load_query() -> np.ndarray:
    """Klassi aniqlanishi kerak bo'lgan yangi nasos Q = (Vibration, FlowRate)."""
    return np.array([4.6, 700.0])



GIVEN_MU = np.array([3.000, 550.000])
GIVEN_SIGMA = np.array([1.555, 302.765])



EX3_LABELS = ["N", "D", "C"]


def load_ex3_results() -> pd.DataFrame:
    return pd.DataFrame({
        "Pump":      [f"T{i}" for i in range(1, 13)],
        "Actual":    ["N", "N", "N", "N", "D", "D", "D", "D", "C", "C", "C", "C"],
        "Predicted": ["N", "N", "D", "N", "D", "N", "D", "C", "C", "C", "D", "C"],
    })


def load_ex4_matrix() -> pd.DataFrame:
    return pd.DataFrame([[16, 4],
                         [12, 68]],
                        index=["Actual CRITICAL", "Actual NOT CRITICAL"],
                        columns=["Pred CRITICAL", "Pred NOT CRITICAL"])


def load_ex5_models() -> pd.DataFrame:
    return pd.DataFrame({
        "TP": [18, 14],
        "FN": [2, 6],
        "FP": [18, 4],
        "TN": [62, 76],
    }, index=["Model A", "Model B"])

EX6_LABELS = ["NORMAL", "DEGRADED", "CRITICAL"]


def load_ex6_matrix() -> np.ndarray:
    return np.array([[24, 4, 0],
                     [5, 17, 6],
                     [0, 3, 21]])