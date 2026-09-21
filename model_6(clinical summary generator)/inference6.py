import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "flan_t5_summary_model"
TEST_FILE = BASE_DIR / "preprocessing6_output" / "test.jsonl"


# ============================================================
# CONFIG
# ============================================================

FIELDS = [
    "chief_complaint",
    "onset",
    "duration",
    "location",
    "severity",
    "complaint_characteristics",
    "associated_symptoms",
    "negative_findings",
    "medical_history",
    "medications",
    "allergies",
    "red_flags",
    "triggers",
    "relieving_factors",
    "relevant_context",
]

LIST_FIELDS = {
    "complaint_characteristics",
    "associated_symptoms",
    "negative_findings",
    "medical_history",
    "medications",
    "allergies",
    "triggers",
    "relieving_factors",
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("MODEL 6 - TEST SET EVALUATION")
print("=" * 70)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# PROMPT
# ============================================================

PROMPT = """You are a clinical summary generator.

Read the patient-assistant conversation and produce a structured
clinical summary using EXACTLY these 15 fields:

chief_complaint
onset
duration
location
severity
complaint_characteristics
associated_symptoms
negative_findings
medical_history
medications
allergies
red_flags
triggers
relieving_factors
relevant_context

Rules:
- Output every field.
- Use exactly one field label followed by ":".
- List fields must use semicolon-separated values.
- If a list field has no information, write "none".
- red_flags must use:
  status=<status>; items=<item1> | <item2>
- Do not add explanations.
- Do not add markdown.
- Do not add JSON.
- Preserve information from the conversation.

Conversation:
"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def conversation_to_text(conversation):
    parts = []

    for message in conversation:
        role = message.get("role", "").strip()
        text = message.get("text", "").strip()

        if role and text:
            parts.append(f"{role}: {text}")

    return "\n".join(parts)


def clean_text(value):
    value = value.strip()

    # Remove accidental trailing punctuation
    value = value.strip(" \n\t")

    if value.lower() == "none":
        return ""

    return value


def parse_list(value):
    value = value.strip()

    if not value or value.lower() == "none":
        return []

    items = []

    for item in value.split(";"):
        item = item.strip()

        if not item:
            continue

        if item.lower() == "none":
            continue

        items.append(item)

    return items


def parse_red_flags(value):
    value = value.strip()

    result = {
        "status": "",
        "items": []
    }

    if not value:
        return result

    status_match = re.search(
        r"status\s*=\s*(.*?)(?:;\s*items\s*=|$)",
        value,
        flags=re.IGNORECASE
    )

    if status_match:
        result["status"] = status_match.group(1).strip()

    items_match = re.search(
        r"items\s*=\s*(.*)$",
        value,
        flags=re.IGNORECASE
    )

    if items_match:
        items_text = items_match.group(1).strip()

        if items_text.lower() != "none":
            result["items"] = [
                item.strip()
                for item in items_text.split("|")
                if item.strip()
            ]

    return result


# ============================================================
# PARSER
# ============================================================

def parse_generated_output(generated_text):
    """
    Handles both:
      field1: value
      field2: value

    and:

      field1: value field2: value field3: value
    """

    text = generated_text.strip()

    parsed = {}

    if not text:
        text = ""

    positions = []

    for field in FIELDS:
        marker = field + ":"

        start = 0

        while True:
            position = text.find(marker, start)

            if position == -1:
                break

            positions.append(
                (position, field, len(marker))
            )

            start = position + len(marker)

    positions.sort(key=lambda x: x[0])

    for i, (position, field, marker_length) in enumerate(positions):

        value_start = position + marker_length

        if i + 1 < len(positions):
            next_position = positions[i + 1][0]
            value = text[value_start:next_position]
        else:
            value = text[value_start:]

        parsed[field] = value.strip()

    summary = {}

    for field in FIELDS:

        value = parsed.get(field, "")

        if field == "red_flags":
            summary[field] = parse_red_flags(value)

        elif field in LIST_FIELDS:
            summary[field] = parse_list(value)

        else:
            summary[field] = clean_text(value)

    return summary


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):
    """
    Used only for evaluation comparison.
    Makes comparison case-insensitive and removes
    insignificant whitespace.
    """

    if isinstance(value, str):
        value = value.lower().strip()

        value = re.sub(r"\s+", " ", value)

        return value

    if isinstance(value, list):
        return sorted(
            normalize(x)
            for x in value
        )

    if isinstance(value, dict):
        return {
            key: normalize(val)
            for key, val in value.items()
        }

    return value


# ============================================================
# FIELD COMPARISON
# ============================================================

def field_equal(predicted, expected, field):

    predicted = normalize(predicted)
    expected = normalize(expected)

    return predicted == expected


# ============================================================
# LOAD TEST DATA
# ============================================================

test_examples = []

with open(TEST_FILE, "r", encoding="utf-8") as f:

    for line in f:
        if line.strip():
            test_examples.append(
                json.loads(line)
            )

print(f"\nTest examples: {len(test_examples)}")


# ============================================================
# EVALUATION
# ============================================================

field_correct = {
    field: 0
    for field in FIELDS
}

field_total = {
    field: len(test_examples)
    for field in FIELDS
}

exact_match_count = 0

invalid_structure_count = 0

results = []


print("\nRunning evaluation...")
print("-" * 70)


for index, example in enumerate(test_examples, start=1):

    conversation = example["conversation"]
    expected = example["clinical_summary"]

    conversation_text = conversation_to_text(
        conversation
    )

    prompt = PROMPT + conversation_text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
        )

    raw_output = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    predicted = parse_generated_output(
        raw_output
    )

    # --------------------------------------------------------
    # Structure check
    # --------------------------------------------------------

    structure_valid = all(
        field in predicted
        for field in FIELDS
    )

    if not structure_valid:
        invalid_structure_count += 1

    # --------------------------------------------------------
    # Field comparison
    # --------------------------------------------------------

    all_fields_correct = True

    field_results = {}

    for field in FIELDS:

        expected_value = expected.get(field)

        predicted_value = predicted.get(field)

        correct = field_equal(
            predicted_value,
            expected_value,
            field
        )

        field_results[field] = correct

        if correct:
            field_correct[field] += 1
        else:
            all_fields_correct = False

    if all_fields_correct:
        exact_match_count += 1

    results.append({
        "index": index,
        "id": example.get("id"),
        "field_results": field_results,
        "expected": expected,
        "predicted": predicted,
        "raw_output": raw_output
    })

    # Progress
    if index % 10 == 0:
        print(f"Processed {index}/{len(test_examples)}")


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(f"\nTotal test examples: {len(test_examples)}")

print(
    f"Exact full-summary matches: "
    f"{exact_match_count}/{len(test_examples)} "
    f"({exact_match_count / len(test_examples) * 100:.2f}%)"
)

print(
    f"Invalid structures: "
    f"{invalid_structure_count}/{len(test_examples)}"
)

print("\nFIELD ACCURACY")
print("-" * 70)

for field in FIELDS:

    correct = field_correct[field]
    total = field_total[field]

    accuracy = correct / total * 100

    print(
        f"{field:30s} "
        f"{correct:3d}/{total:<3d} "
        f"({accuracy:6.2f}%)"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = BASE_DIR / "model6_evaluation_results.json"

evaluation_summary = {
    "total_examples": len(test_examples),
    "exact_full_summary_matches": exact_match_count,
    "exact_match_percentage": (
        exact_match_count / len(test_examples) * 100
    ),
    "invalid_structure_count": invalid_structure_count,
    "field_accuracy": {
        field: {
            "correct": field_correct[field],
            "total": field_total[field],
            "accuracy_percentage":
                field_correct[field] /
                field_total[field] *
                100
        }
        for field in FIELDS
    },
    "examples": results
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_summary,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print("=" * 70)
print("Evaluation saved to:")
print(OUTPUT_FILE)
print("=" * 70)