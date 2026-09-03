import json
import random
from pathlib import Path
from collections import Counter

from collections import defaultdict

# File paths
dataset_folder = Path(__file__).parent / "dataset"
DATA_PATH = dataset_folder / "model1_dummy.json"
# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset loaded successfully.")
print("Python type:", type(data))

if isinstance(data, list):
    print("Number of records:", len(data))
elif isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))

print("\nFirst record:")
print(data[0] if isinstance(data, list) else data)


# ============================================
# STEP 2 — DATASET INSPECTION
# ============================================

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

# ------------------------------------------------
# 1. BASIC INFORMATION
# ------------------------------------------------

print("\n1. BASIC INFORMATION")
print("-" * 40)

print("Total records:", len(data))

# ------------------------------------------------
# 2. CHECK REQUIRED TOP-LEVEL KEYS
# ------------------------------------------------

print("\n2. TOP-LEVEL KEYS")
print("-" * 40)

required_keys = {"id", "messages", "target", "safety"}

records_with_missing_keys = []

for record in data:
    missing = required_keys - set(record.keys())

    if missing:
        records_with_missing_keys.append(
            (record.get("id", "NO_ID"), missing)
        )

print("Expected keys:", required_keys)
print("Records with missing keys:", len(records_with_missing_keys))

if records_with_missing_keys:
    print("Examples:")
    for item in records_with_missing_keys[:10]:
        print(item)

# ------------------------------------------------
# 3. CHECK DUPLICATE IDS
# ------------------------------------------------

print("\n3. DUPLICATE IDS")
print("-" * 40)

ids = [record.get("id") for record in data]
id_counts = Counter(ids)

duplicate_ids = {
    record_id: count
    for record_id, count in id_counts.items()
    if count > 1
}

print("Unique IDs:", len(id_counts))
print("Duplicate IDs:", len(duplicate_ids))

if duplicate_ids:
    print("Examples:")
    for record_id, count in list(duplicate_ids.items())[:10]:
        print(f"  {record_id} -> {count} times")
# ------------------------------------------------
# 4. CHECK MESSAGE STRUCTURE
# ------------------------------------------------

print("\n4. MESSAGE STRUCTURE")
print("-" * 40)

invalid_messages = []
empty_messages = []
invalid_roles = []

allowed_roles = {"user", "assistant"}

message_count_distribution = Counter()

for record in data:

    record_id = record.get("id", "NO_ID")
    messages = record.get("messages")

    if not isinstance(messages, list):
        invalid_messages.append(
            (record_id, "messages is not a list")
        )
        continue

    message_count_distribution[len(messages)] += 1

    for message in messages:

        if not isinstance(message, dict):
            invalid_messages.append(
                (record_id, "message is not a dictionary")
            )
            continue

        if "role" not in message or "content" not in message:
            invalid_messages.append(
                (record_id, "message missing role/content")
            )
            continue

        role = message["role"]
        content = message["content"]

        if role not in allowed_roles:
            invalid_roles.append(
                (record_id, role)
            )

        if not isinstance(content, str) or not content.strip():
            empty_messages.append(record_id)

print("Invalid message structures:", len(invalid_messages))
print("Invalid roles:", len(invalid_roles))
print("Records containing empty messages:", len(set(empty_messages)))

print("\nMessage count distribution:")
for count, frequency in sorted(message_count_distribution.items()):
    print(f"  {count} messages -> {frequency} records")

# ------------------------------------------------
# 5. CHECK TARGETS
# ------------------------------------------------

print("\n5. TARGETS")
print("-" * 40)

targets = [
    record.get("target")
    for record in data
]

missing_targets = [
    record.get("id", "NO_ID")
    for record in data
    if not record.get("target")
]

target_counts = Counter(targets)

print("Unique targets:", len(target_counts))
print("Missing targets:", len(missing_targets))

print("\nTarget distribution:")

for target, count in target_counts.most_common():
    print(f"  {target}: {count}")

# ------------------------------------------------
# 6. CHECK SAFETY / RED FLAG
# ------------------------------------------------

print("\n6. RED-FLAG LABELS")
print("-" * 40)

red_flag_values = Counter()
invalid_safety = []

for record in data:

    record_id = record.get("id", "NO_ID")
    safety = record.get("safety")

    if not isinstance(safety, dict):
        invalid_safety.append(
            (record_id, "safety is not a dictionary")
        )
        continue

    red_flag = safety.get("red_flag")

    red_flag_values[str(red_flag)] += 1

    if not isinstance(red_flag, bool):
        invalid_safety.append(
            (record_id, red_flag)
        )

print("Red-flag distribution:")

for value, count in red_flag_values.items():
    print(f"  {value}: {count}")

print("Invalid safety records:", len(invalid_safety))

# ------------------------------------------------
# 7. CHECK FINAL MESSAGE ROLE
# ------------------------------------------------

print("\n7. FINAL MESSAGE ROLE")
print("-" * 40)

final_role_counts = Counter()
records_not_ending_assistant = []

for record in data:

    record_id = record.get("id", "NO_ID")
    messages = record.get("messages", [])

    if messages:
        final_role = messages[-1].get("role")
        final_role_counts[final_role] += 1

        if final_role != "assistant":
            records_not_ending_assistant.append(
                record_id
            )

print("Final message roles:")

for role, count in final_role_counts.items():
    print(f"  {role}: {count}")

print(
    "Records NOT ending with assistant:",
    len(records_not_ending_assistant)
)

# ------------------------------------------------
# 8. CHECK SIMPLE DUPLICATE CONVERSATIONS
# ------------------------------------------------

print("\n8. EXACT DUPLICATE CONVERSATIONS")
print("-" * 40)

conversation_strings = []

for record in data:

    messages = record.get("messages", [])

    conversation = tuple(
        (m.get("role"), m.get("content"))
        for m in messages
        if isinstance(m, dict)
    )

    conversation_strings.append(conversation)

conversation_counts = Counter(conversation_strings)

duplicate_conversations = {
    conversation: count
    for conversation, count in conversation_counts.items()
    if count > 1
}

print(
    "Exact duplicate conversation groups:",
    len(duplicate_conversations)
)

print("=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)


# ============================================
# STEP 3 — TARGET ↔ FINAL QUESTION SAMPLE
# ============================================


print("\n" + "=" * 60)
print("TARGET FINAL QUESTION SAMPLE")
print("=" * 60)

examples_by_target = defaultdict(list)

for record in data:
    target = record["target"]
    final_question = record["messages"][-1]["content"]

    examples_by_target[target].append({
        "id": record["id"],
        "question": final_question
    })

# Show up to 3 examples for every target
for target in sorted(examples_by_target):

    print(f"\nTARGET: {target}")
    print("-" * 50)

    examples = examples_by_target[target]

    for example in examples[:3]:
        print(f"ID: {example['id']}")
        print(f"Question: {example['question']}")

print("\n" + "=" * 60)
print("END OF TARGET INSPECTION")
print("=" * 60)


# ============================================
# STEP 4 — CONVERSATION FLOW INSPECTION
# ============================================

import random

print("\n" + "=" * 70)
print("STEP 4 - CONVERSATION FLOW INSPECTION")
print("=" * 70)

# Fixed seed so we get the same records every time
random.seed(42)

# Pick 30 records from the dataset
sample_size = min(30, len(data))
sample_records = random.sample(data, sample_size)

for number, record in enumerate(sample_records, start=1):

    print("\n" + "=" * 70)
    print(f"EXAMPLE {number}/30")
    print("=" * 70)

    print("ID:", record["id"])
    print("TARGET:", record["target"])
    print("RED FLAG:", record["safety"]["red_flag"])

    print("\nCONVERSATION:")
    print("-" * 70)

    for i, message in enumerate(record["messages"], start=1):
        role = message["role"].upper()
        content = message["content"]

        print(f"{i}. [{role}] {content}")

    print("\nFINAL QUESTION:")
    print("-" * 70)
    print(record["messages"][-1]["content"])

print("\n" + "=" * 70)
print("FLOW INSPECTION COMPLETE")
print("=" * 70)


# ============================================
# STEP 5 — SYSTEMATIC TARGET VALIDATION
# ============================================

from collections import defaultdict

print("\n" + "=" * 70)
print("STEP 5 - SYSTEMATIC TARGET VALIDATION")
print("=" * 70)

# Group records by target
target_groups = defaultdict(list)

for record in data:
    target_groups[record["target"]].append(record)

print("\nTotal unique targets:", len(target_groups))

# Print every target and all its final questions
for target in sorted(target_groups.keys()):

    records = target_groups[target]

    print("\n" + "=" * 70)
    print(f"TARGET: {target}")
    print(f"NUMBER OF RECORDS: {len(records)}")
    print("=" * 70)

    for number, record in enumerate(records, start=1):

        final_question = record["messages"][-1]["content"]

        print(f"\n{number}. ID: {record['id']}")
        print(f"   FINAL QUESTION: {final_question}")

print("\n" + "=" * 70)
print("SYSTEMATIC TARGET VALIDATION COMPLETE")
print("=" * 70)