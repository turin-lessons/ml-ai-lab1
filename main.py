"""
K-NN mashqlarini ishga tushiruvchi interaktiv menyu.

Foydalanish:
    python main.py          # menyu ochiladi: mashqni tanlang yoki "all"
    python main.py 2 5      # menyusiz, faqat 2 va 5-mashq
    python main.py all      # menyusiz, hammasi
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EXERCISES = {
    1: ("Feature dominance in K-NN",          "task1.py"),
    2: ("What changes after standardization", "task2.py"),
    3: ("Multiclass confusion matrix",        "task3.py"),
    4: ("Precision versus recall",            "task4.py"),
    5: ("Compare two K-NN models",            "task5.py"),
    6: ("Interpret a K-NN error pattern",     "task6.py"),
}

ALL_WORDS = {"all", "a", "0", "hammasi"}
QUIT_WORDS = {"q", "quit", "exit", "chiqish"}


def run_exercise(num: int) -> bool:
    title, filename = EXERCISES[num]
    print("=" * 70)
    print(f"MASHQ {num}: {title}")
    print("=" * 70)
    try:

        runpy.run_path(str(BASE_DIR / filename), run_name="__main__")
        print()
        return True
    except Exception as exc:
        print(f"\n[XATO] Mashq {num} ({filename}): {type(exc).__name__}: {exc}\n")
        return False


def run_many(numbers: list[int]) -> bool:
    results = {n: run_exercise(n) for n in numbers}
    if len(numbers) > 1:
        print("=" * 70)
        print("XULOSA")
        print("=" * 70)
        for n, ok in results.items():
            print(f"  Mashq {n}: {'OK' if ok else 'XATO'}")
        print()
    return all(results.values())


def parse_choice(text: str) -> list[int] | None:
    """'3' -> [3], '1 4' yoki '1,4' -> [1, 4], 'all' -> [1..6]. Noto'g'ri bo'lsa None."""
    text = text.strip().lower()
    if text in ALL_WORDS:
        return list(EXERCISES)
    try:
        numbers = [int(p) for p in text.replace(",", " ").split()]
    except ValueError:
        return None
    if not numbers or any(n not in EXERCISES for n in numbers):
        return None
    return numbers


def print_menu() -> None:
    print("=" * 70)
    print("K-NN LAB - qaysi mashqni tekshirmoqchisiz?")
    print("=" * 70)
    for num, (title, _) in EXERCISES.items():
        print(f"  {num}) {title}")
    print("  all) Hammasini ketma-ket ishga tushirish")
    print("  q)   Chiqish")
    print("  (bir nechtasini birdan ham tanlash mumkin: 1 3 5)")


def interactive_loop() -> None:
    while True:
        print_menu()
        try:
            choice = input("\nTanlovingiz: ")
        except (EOFError, KeyboardInterrupt):
            print("\nChiqildi.")
            return

        if not choice.strip():
            continue

        if choice.strip().lower() in QUIT_WORDS:
            print("Chiqildi.")
            return

        numbers = parse_choice(choice)
        if numbers is None:
            print(f"\nNoto'g'ri tanlov: '{choice.strip()}'. 1-6, 'all' yoki 'q' kiriting.\n")
            continue

        print()
        run_many(numbers)

        try:
            input("Menyuga qaytish uchun Enter bosing...")
        except (EOFError, KeyboardInterrupt):
            print("\nChiqildi.")
            return
        print()


def main(argv: list[str]) -> int:

    sys.path.insert(0, str(BASE_DIR))

    if not argv:
        interactive_loop()
        return 0

    numbers = parse_choice(" ".join(argv))
    if numbers is None:
        print(f"Noto'g'ri argument: {' '.join(argv)}. Misol: python main.py 1 3  yoki  python main.py all")
        return 2
    return 0 if run_many(numbers) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))