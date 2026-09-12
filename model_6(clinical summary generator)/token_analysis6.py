#3 becz preprocessing tehn prepartion tehn this 
from pathlib import Path
import json
import statistics

from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/flan-t5-small"

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "csg_prepared" / "train.jsonl"
VAL_FILE = BASE_DIR / "csg_prepared" / "validation.jsonl"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CSG TOKEN ANALYSIS")
print("=" * 70)

print()
print(f"Model tokenizer : {MODEL_NAME}")

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                records.append(json.loads(line))

    return records


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):

    values = sorted(values)

    index = int(len(values) * p)

    index = min(index, len(values) - 1)

    return values[index]


# ============================================================
# ANALYZE
# ============================================================

def analyze(records, dataset_name):

    input_lengths = []
    target_lengths = []

    print()
    print("=" * 70)
    print(f"ANALYZING {dataset_name}")
    print("=" * 70)

    total = len(records)

    for i, record in enumerate(records, start=1):

        input_text = record["input_text"]
        target_text = record["target_text"]

        input_tokens = tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        target_tokens = tokenizer(
            target_text,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        input_lengths.append(len(input_tokens))
        target_lengths.append(len(target_tokens))

        # Progress every 50 records
        if i % 50 == 0 or i == total:
            print(f"Processed {i:3}/{total} records")

    # ========================================================
    # INPUT RESULTS
    # ========================================================

    print()
    print("INPUT — CONVERSATION")
    print("-" * 70)

    print(f"Minimum tokens     : {min(input_lengths)}")
    print(f"Maximum tokens     : {max(input_lengths)}")
    print(f"Average tokens     : {statistics.mean(input_lengths):.2f}")
    print(f"Median tokens      : {statistics.median(input_lengths):.2f}")
    print(f"90th percentile    : {percentile(input_lengths, 0.90)}")
    print(f"95th percentile    : {percentile(input_lengths, 0.95)}")

    # ========================================================
    # TARGET RESULTS
    # ========================================================

    print()
    print("TARGET — SUMMARY")
    print("-" * 70)

    print(f"Minimum tokens     : {min(target_lengths)}")
    print(f"Maximum tokens     : {max(target_lengths)}")
    print(f"Average tokens     : {statistics.mean(target_lengths):.2f}")
    print(f"Median tokens      : {statistics.median(target_lengths):.2f}")
    print(f"90th percentile    : {percentile(target_lengths, 0.90)}")
    print(f"95th percentile    : {percentile(target_lengths, 0.95)}")

    # ========================================================
    # TRUNCATION CHECK
    # ========================================================

    print()
    print("INPUT TRUNCATION CHECK")
    print("-" * 70)

    for limit in [256, 320, 384, 448, 512]:

        count = sum(
            length > limit
            for length in input_lengths
        )

        percentage = (count / total) * 100

        print(
            f"Input > {limit:3} tokens : "
            f"{count:3}/{total} "
            f"({percentage:.2f}%)"
        )

    print()
    print("TARGET TRUNCATION CHECK")
    print("-" * 70)

    for limit in [96, 128, 160, 192]:

        count = sum(
            length > limit
            for length in target_lengths
        )

        percentage = (count / total) * 100

        print(
            f"Target > {limit:3} tokens : "
            f"{count:3}/{total} "
            f"({percentage:.2f}%)"
        )

    return input_lengths, target_lengths


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading prepared datasets...")

train = load_jsonl(TRAIN_FILE)
validation = load_jsonl(VAL_FILE)

print()
print(f"Training examples   : {len(train)}")
print(f"Validation examples : {len(validation)}")


# ============================================================
# ANALYZE TRAIN
# ============================================================

train_input_lengths, train_target_lengths = analyze(
    train,
    "TRAINING DATA"
)


# ============================================================
# ANALYZE VALIDATION
# ============================================================

val_input_lengths, val_target_lengths = analyze(
    validation,
    "VALIDATION DATA"
)


# ============================================================
# FINAL COMBINED CHECK
# ============================================================

all_input_lengths = (
    train_input_lengths +
    val_input_lengths
)

all_target_lengths = (
    train_target_lengths +
    val_target_lengths
)

print()
print("=" * 70)
print("FINAL COMBINED DATASET ANALYSIS")
print("=" * 70)

print()
print("ALL INPUTS")
print("-" * 70)

print(f"Maximum input tokens : {max(all_input_lengths)}")
print(f"95th percentile     : {percentile(all_input_lengths, 0.95)}")

print()
print("ALL TARGETS")
print("-" * 70)

print(f"Maximum target tokens : {max(all_target_lengths)}")
print(f"95th percentile      : {percentile(all_target_lengths, 0.95)}")


# ============================================================
# RECOMMENDATION
# ============================================================

print()
print("=" * 70)
print("PRELIMINARY RECOMMENDATION")
print("=" * 70)

print()

max_input = max(all_input_lengths)
max_target = max(all_target_lengths)

if max_input <= 256:
    input_recommendation = 256
elif max_input <= 320:
    input_recommendation = 320
elif max_input <= 384:
    input_recommendation = 384
elif max_input <= 448:
    input_recommendation = 448
else:
    input_recommendation = 512


if max_target <= 96:
    target_recommendation = 96
elif max_target <= 128:
    target_recommendation = 128
elif max_target <= 160:
    target_recommendation = 160
else:
    target_recommendation = 192


print(
    f"Suggested max_input_length  : "
    f"{input_recommendation}"
)

print(
    f"Suggested max_target_length : "
    f"{target_recommendation}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TOKEN ANALYSIS COMPLETE")
print("=" * 70)