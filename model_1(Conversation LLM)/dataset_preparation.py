#Split → Verify → Save → Tokenization
import json
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

from collections import defaultdict
dataset_folder = Path(__file__).parent / "dataset"
DATA_PATH = dataset_folder / "model1_dummy.json"
# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset loaded successfully.")
print("Python type:", type(data))

# ============================================
# STEP 8A — SPLIT + VERIFY
# ============================================



print("\n" + "=" * 70)
print("STEP 8A — TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

# 80% Train, 10% Validation, 10% Test
train_data, temp_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42
)

# --------------------------------------------
# BASIC SIZE CHECK
# --------------------------------------------

print("\nRECORD COUNTS")
print("-" * 70)

print("Original dataset :", len(data))
print("Train            :", len(train_data))
print("Validation       :", len(val_data))
print("Test             :", len(test_data))

# --------------------------------------------
# ID OVERLAP CHECK
# --------------------------------------------

train_ids = {record["id"] for record in train_data}
val_ids = {record["id"] for record in val_data}
test_ids = {record["id"] for record in test_data}

print("\nID OVERLAP CHECK")
print("-" * 70)

print("Train ∩ Validation:", len(train_ids & val_ids))
print("Train ∩ Test      :", len(train_ids & test_ids))
print("Validation ∩ Test :", len(val_ids & test_ids))

# --------------------------------------------
# TOTAL UNIQUE ID CHECK
# --------------------------------------------

all_split_ids = train_ids | val_ids | test_ids

print("\nUNIQUE ID CHECK")
print("-" * 70)

print("Unique IDs in splits:", len(all_split_ids))
print("Original unique IDs :", len({record["id"] for record in data}))

# --------------------------------------------
# TARGET DISTRIBUTION
# --------------------------------------------

print("\nTARGET COUNTS")
print("-" * 70)

print("Train targets:", len(set(record["target"] for record in train_data)))
print("Validation targets:", len(set(record["target"] for record in val_data)))
print("Test targets:", len(set(record["target"] for record in test_data)))

# --------------------------------------------
# FINAL VERDICT
# --------------------------------------------

print("\n" + "=" * 70)

if (
    len(train_data) == 440
    and len(val_data) == 55
    and len(test_data) == 55
    and len(train_ids & val_ids) == 0
    and len(train_ids & test_ids) == 0
    and len(val_ids & test_ids) == 0
    and len(all_split_ids) == len(data)
):
    print("SPLIT VERIFICATION: PASSED")
else:
    print("SPLIT VERIFICATION: CHECK REQUIRED")

print("=" * 70)


# ============================================
# STEP 8B — SAVE DATASET SPLITS
# ============================================

import os
import json

SPLIT_DIR = "data/splits"

os.makedirs(SPLIT_DIR, exist_ok=True)

# Save training data
with open(
    os.path.join(SPLIT_DIR, "train.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

# Save validation data
with open(
    os.path.join(SPLIT_DIR, "validation.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

# Save test data
with open(
    os.path.join(SPLIT_DIR, "test.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)


print("\n" + "=" * 70)
print("STEP 8B — SPLITS SAVED")
print("=" * 70)

print("\nSaved files:")

print("Train      :", os.path.join(SPLIT_DIR, "train.json"))
print("Validation :", os.path.join(SPLIT_DIR, "validation.json"))
print("Test       :", os.path.join(SPLIT_DIR, "test.json"))

print("\n" + "=" * 70)


# ============================================
# STEP 8C — RELOAD & VERIFY SAVED SPLITS
# ============================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Reload the saved files
train_check = load_json(os.path.join(SPLIT_DIR, "train.json"))
val_check = load_json(os.path.join(SPLIT_DIR, "validation.json"))
test_check = load_json(os.path.join(SPLIT_DIR, "test.json"))


# Get IDs
train_ids = {record["id"] for record in train_check}
val_ids = {record["id"] for record in val_check}
test_ids = {record["id"] for record in test_check}

original_ids = {record["id"] for record in data}


print("\n" + "=" * 70)
print("STEP 8C — RELOAD & VERIFY")
print("=" * 70)

print("\nRELOADED RECORD COUNTS")
print("-" * 70)

print("Train      :", len(train_check))
print("Validation :", len(val_check))
print("Test       :", len(test_check))

print("\nID OVERLAP CHECK")
print("-" * 70)

print("Train ∩ Validation:", len(train_ids & val_ids))
print("Train ∩ Test      :", len(train_ids & test_ids))
print("Validation ∩ Test :", len(val_ids & test_ids))

print("\nORIGINAL ID COVERAGE")
print("-" * 70)

combined_ids = train_ids | val_ids | test_ids

print("Combined split IDs :", len(combined_ids))
print("Original IDs       :", len(original_ids))

print("\nFINAL VERIFICATION")
print("-" * 70)

if (
    len(train_check) == 440
    and len(val_check) == 55
    and len(test_check) == 55
    and len(train_ids & val_ids) == 0
    and len(train_ids & test_ids) == 0
    and len(val_ids & test_ids) == 0
    and combined_ids == original_ids
):
    print("RELOAD VERIFICATION: PASSED")
else:
    print("RELOAD VERIFICATION: FAILED")

print("\n" + "=" * 70)





# ============================================
# TOKENIZATION we will laod model fromhugginf face
# ============================================