"""
Batch-tests your running Vanaushadhi API against a folder of fresh photos
(not your training/validation data), logs results to test_results.csv, and
prints a confusion summary -- which species get mistaken for which, and how
often -- so you can see real patterns instead of debugging one photo at a time.

Setup:
  1. Make sure `python app.py` is already running in another terminal.
  2. Create a folder called `manual_test_images` next to this script.
  3. Inside it, make one subfolder per species using the SAME names as your
     dataset (e.g. Tulsi, Neem, Aloe_Vera...), and put 5+ fresh phone photos
     of each into its folder. For best results: one plant per photo, leaves
     only (no flowers unless your training data included them), plain
     background, decent lighting -- consistent conditions make the results
     meaningful rather than noisy.
     If you don't know the true species for some photos, put them directly
     in `manual_test_images` (not in a subfolder) -- they'll still be
     tested, just not scored for accuracy.

Run:
  pip install requests
  python test_model.py
"""

import os
import csv
from collections import Counter
import requests

API_URL = "http://localhost:5000/predict"
TEST_DIR = "manual_test_images"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def main():
    if not os.path.isdir(TEST_DIR):
        print(f"Create a folder called '{TEST_DIR}' first -- see the instructions at the top of this script.")
        return

    results = []
    correct = 0
    total_labeled = 0

    for root, _, files in os.walk(TEST_DIR):
        true_label = os.path.basename(root) if root != TEST_DIR else "unknown"
        for fname in sorted(files):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "rb") as f:
                    resp = requests.post(API_URL, files={"file": f}, timeout=30)
            except requests.exceptions.ConnectionError:
                print("Could not reach the API. Is `python app.py` running?")
                return

            if resp.status_code != 200:
                print(f"  ERROR on {path}: {resp.text}")
                continue

            data = resp.json()
            predicted = data["species"]
            confidence = data["confidence"]
            predicted_norm = predicted.replace(" ", "_").lower()
            true_norm = true_label.replace(" ", "_").lower()
            is_correct = (true_label != "unknown" and predicted_norm == true_norm)

            if true_label != "unknown":
                total_labeled += 1
                if is_correct:
                    correct += 1

            mark = "correct" if is_correct else ("n/a" if true_label == "unknown" else "WRONG")
            print(f"{path:<55} predicted={predicted:<16} ({confidence:5.1f}%)  actual={true_label:<16} [{mark}]")

            results.append([path, true_label, predicted, confidence, is_correct])

    if not results:
        print(f"No images found. Add some .jpg/.png files inside '{TEST_DIR}' (optionally in per-species subfolders).")
        return

    with open("test_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "actual", "predicted", "confidence", "correct"])
        writer.writerows(results)

    print(f"\nTested {len(results)} images total.")
    if total_labeled > 0:
        acc = correct / total_labeled * 100
        print(f"Real-world accuracy on {total_labeled} labeled photos: {acc:.1f}% ({correct}/{total_labeled} correct)")
        print("Use this number alongside your validation accuracy in your report --")
        print("it's a more honest picture of how the model performs on photos it has never seen.")

        # Confusion summary: which species get mistaken for which, and how often.
        confusions = Counter()
        per_class_total = Counter()
        per_class_correct = Counter()
        for _, actual, predicted, confidence, is_correct in results:
            if actual == "unknown":
                continue
            per_class_total[actual] += 1
            if is_correct:
                per_class_correct[actual] += 1
            else:
                confusions[(actual, predicted)] += 1

        print("\n--- Per-species accuracy ---")
        for cls in sorted(per_class_total):
            t = per_class_total[cls]
            c = per_class_correct[cls]
            print(f"  {cls:<16} {c}/{t} correct ({c/t*100:.0f}%)")

        if confusions:
            print("\n--- Most common confusions (actual -> predicted : count) ---")
            for (actual, predicted), count in confusions.most_common(10):
                print(f"  {actual:<16} -> {predicted:<16} : {count} time(s)")
            print("\nIf the same pair shows up repeatedly, that's a real pattern worth")
            print("investigating (similar leaf shape, overlapping training images, an")
            print("underrepresented class) rather than one-off bad luck -- see the")
            print("retraining checklist in deployment_and_testing_guide.md.")
        else:
            print("\nNo misclassifications in this batch -- nice.")

    print("\nFull results saved to test_results.csv")


if __name__ == "__main__":
    main()