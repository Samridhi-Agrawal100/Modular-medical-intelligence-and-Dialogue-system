
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("model_6(clinical summary generator)\\flan_t5_summary_model")
TEST_FILE = Path("model_6(clinical summary generator)\\flan_t5_data\\test.jsonl")

MAX_SOURCE_LENGTH = 512
MAX_NEW_TOKENS = 256

OUTPUT_FILE = Path("model6_evaluation_results.json")


# ============================================================
# EXPECTED FIELDS
# ============================================================

EXPECTED_FIELDS = [
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


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("MODEL 6 - FULL TEST EVALUATION")
print("=" * 60)

test_examples = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_examples.append(json.loads(line))

print(f"\nTest examples: {len(test_examples)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

model.eval()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)


# ============================================================
# GENERATION
# ============================================================

def generate_summary(conversation):

    prompt = f"""Generate a structured clinical summary from the following patient-assistant conversation.

Return the clinical summary using exactly these 15 fields:
chief_complaint, onset, duration, location, severity, complaint_characteristics, associated_symptoms, negative_findings, medical_history, medications, allergies, red_flags, triggers, relieving_factors, relevant_context.

Do not add information that is not present in the conversation.

Conversation:
{conversation}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            do_sample=False,
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):

    if isinstance(value, str):
        return value.strip().lower()

    if isinstance(value, list):
        return sorted(
            str(x).strip().lower()
            for x in value
        )

    if isinstance(value, dict):
        return {
            str(k).strip().lower(): normalize(v)
            for k, v in value.items()
        }

    return value


def field_matches(predicted, expected):
    return normalize(predicted) == normalize(expected)


# ============================================================
# SAFE JSON REPAIR FOR EVALUATION
# ============================================================

def try_json_parse(text):

    text = text.strip()

    # --------------------------------------------------------
    # 1. Try exact model output
    # --------------------------------------------------------

    try:
        return json.loads(text), "raw"

    except json.JSONDecodeError:
        pass


    # --------------------------------------------------------
    # 2. Add missing outer braces
    #
    # Model is generating:
    #
    # "chief_complaint":"..."
    # ...
    # "relevant_context":"..."
    #
    # instead of:
    #
    # {
    #   "chief_complaint":"..."
    #   ...
    # }
    # --------------------------------------------------------

    repaired = "{" + text + "}"

    try:
        return json.loads(repaired), "outer_braces_added"

    except json.JSONDecodeError:
        pass


    # --------------------------------------------------------
    # 3. Try extracting from the first { to the last }
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate), "extracted_json"

        except json.JSONDecodeError:
            pass


    # --------------------------------------------------------
    # 4. Completely invalid
    # --------------------------------------------------------

    return None, "invalid"


# ============================================================
# COUNTERS
# ============================================================

total = len(test_examples)

raw_valid_json = 0
repaired_valid_json = 0

all_fields_present = 0
exact_match = 0
red_flags_correct = 0

field_correct = {
    field: 0
    for field in EXPECTED_FIELDS
}

failed_examples = []


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting evaluation...\n")

for index, example in enumerate(test_examples, start=1):

    generated_text = generate_summary(
        example["input"]
    )

    expected_json = json.loads(
        example["target"]
    )

    # --------------------------------------------------------
    # Print first 5 outputs
    # --------------------------------------------------------

    if index <= 5:

        print("\n" + "=" * 60)
        print(f"RAW OUTPUT #{index}")
        print("=" * 60)

        print(generated_text)

    # --------------------------------------------------------
    # Try parsing
    # --------------------------------------------------------

    predicted_json, parse_type = try_json_parse(
        generated_text
    )

    if parse_type == "raw":
        raw_valid_json += 1

    if predicted_json is not None:
        repaired_valid_json += 1

        # ----------------------------------------------------
        # Check all 15 fields
        # ----------------------------------------------------

        missing_fields = [
            field
            for field in EXPECTED_FIELDS
            if field not in predicted_json
        ]

        if len(missing_fields) == 0:
            all_fields_present += 1

        # ----------------------------------------------------
        # Field-level accuracy
        # ----------------------------------------------------

        for field in EXPECTED_FIELDS:

            if field in predicted_json:

                if field_matches(
                    predicted_json[field],
                    expected_json[field]
                ):
                    field_correct[field] += 1

        # ----------------------------------------------------
        # Red flags
        # ----------------------------------------------------

        predicted_red_flags = predicted_json.get(
            "red_flags"
        )

        expected_red_flags = expected_json.get(
            "red_flags"
        )

        if (
            isinstance(predicted_red_flags, dict)
            and field_matches(
                predicted_red_flags,
                expected_red_flags
            )
        ):
            red_flags_correct += 1

        # ----------------------------------------------------
        # Complete match
        # ----------------------------------------------------

        if field_matches(
            predicted_json,
            expected_json
        ):
            exact_match += 1

        else:

            failed_examples.append({
                "index": index,
                "parse_type": parse_type,
                "generated": predicted_json,
                "expected": expected_json,
            })

    else:

        failed_examples.append({
            "index": index,
            "parse_type": "invalid",
            "generated_raw": generated_text,
            "expected": expected_json,
        })

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 10 == 0:
        print(f"\nProcessed {index}/{total}")


# ============================================================
# PERCENTAGE
# ============================================================

def pct(value):
    return round(
        (value / total) * 100,
        2
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL 6 EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nRAW VALID JSON:"
    f" {raw_valid_json}/{total}"
    f" ({pct(raw_valid_json)}%)"
)

print(
    f"PARSEABLE AFTER SAFE OUTER-BRACE REPAIR:"
    f" {repaired_valid_json}/{total}"
    f" ({pct(repaired_valid_json)}%)"
)

print(
    f"ALL 15 FIELDS PRESENT:"
    f" {all_fields_present}/{total}"
    f" ({pct(all_fields_present)}%)"
)

print(
    f"EXACT COMPLETE MATCH:"
    f" {exact_match}/{total}"
    f" ({pct(exact_match)}%)"
)

print(
    f"CORRECT RED_FLAGS STRUCTURE:"
    f" {red_flags_correct}/{total}"
    f" ({pct(red_flags_correct)}%)"
)


# ============================================================
# FIELD ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("FIELD-LEVEL ACCURACY")
print("=" * 60)

field_percentages = {}

for field in EXPECTED_FIELDS:

    count = field_correct[field]

    percentage = pct(count)

    field_percentages[field] = percentage

    print(
        f"{field:30s}"
        f"{count:3d}/{total}"
        f"  ({percentage}%)"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "total_examples": total,

    "raw_valid_json": {
        "count": raw_valid_json,
        "percentage": pct(raw_valid_json),
    },

    "parseable_after_outer_brace_repair": {
        "count": repaired_valid_json,
        "percentage": pct(repaired_valid_json),
    },

    "all_15_fields_present": {
        "count": all_fields_present,
        "percentage": pct(all_fields_present),
    },

    "exact_complete_match": {
        "count": exact_match,
        "percentage": pct(exact_match),
    },

    "red_flags_structure": {
        "count": red_flags_correct,
        "percentage": pct(red_flags_correct),
    },

    "field_accuracy": field_percentages,

    "failed_examples": failed_examples,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"\nResults saved to: {OUTPUT_FILE}"
)

