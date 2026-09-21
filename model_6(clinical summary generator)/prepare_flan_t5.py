import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "preprocessing6_output"
OUTPUT_DIR = BASE_DIR / "flan_t5_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FLAN-T5 PROMPT
# ============================================================

PROMPT = """Generate a structured clinical summary from the following patient-assistant conversation.

Return the clinical summary using exactly these 15 fields, one field per line:

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

For list fields, separate multiple items using semicolons (;).
If a list field has no information, write: none

For red_flags, use exactly this format:
red_flags: status=<status>; items=<item1> | <item2>

If there are no red-flag items, write:
red_flags: status=<status>; items=none

Do not add information that is not present in the conversation.
Do not add explanations.
Return only the 15 labelled fields.

Conversation:
"""


# ============================================================
# FIELD DEFINITIONS
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
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Convert a value into clean text for the FLAN-T5 target.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def format_list(values):
    """
    Convert a list into:
        item1; item2; item3

    Empty list becomes:
        none
    """
    if not values:
        return "none"

    cleaned = []

    for value in values:
        value = clean_value(value)

        if value:
            cleaned.append(value)

    if not cleaned:
        return "none"

    return "; ".join(cleaned)


def format_red_flags(red_flags):
    """
    Convert the nested red_flags object into one simple line.

    Example:
    status=assessed_negative; items=fainting | weakness
    """

    if not isinstance(red_flags, dict):
        return "status=unknown; items=none"

    status = clean_value(red_flags.get("status", ""))

    if not status:
        status = "unknown"

    items = red_flags.get("items", [])

    if not isinstance(items, list):
        items = []

    cleaned_items = []

    for item in items:
        item = clean_value(item)

        if item:
            cleaned_items.append(item)

    if cleaned_items:
        items_text = " | ".join(cleaned_items)
    else:
        items_text = "none"

    return f"status={status}; items={items_text}"


def conversation_to_text(conversation):
    """
    Convert conversation list into the text format expected by FLAN-T5.
    """

    lines = []

    for message in conversation:

        role = clean_value(message.get("role", ""))
        text = clean_value(message.get("text", ""))

        if not text:
            continue

        if role.lower() == "assistant":
            lines.append(f"Assistant: {text}")

        elif role.lower() == "patient":
            lines.append(f"Patient: {text}")

        else:
            # Preserve unexpected roles rather than silently dropping them.
            lines.append(f"{role.capitalize()}: {text}")

    return "\n".join(lines)


def create_target(clinical_summary):
    """
    Convert the original 15-field clinical summary JSON
    into a simple labelled text target.

    IMPORTANT:
    The final schema remains the same 15 fields.
    Only the training representation changes.
    """

    lines = []

    for field in FIELDS:

        if field == "red_flags":
            value = format_red_flags(
                clinical_summary.get("red_flags", {})
            )

        elif field in LIST_FIELDS:
            value = format_list(
                clinical_summary.get(field, [])
            )

        else:
            value = clean_value(
                clinical_summary.get(field, "")
            )

        # Keep exactly one labelled line per field.
        lines.append(f"{field}: {value}")

    return "\n".join(lines)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    input_file = INPUT_DIR / f"{split_name}.jsonl"
    output_file = OUTPUT_DIR / f"{split_name}.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_file}"
        )

    count = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line_number, line in enumerate(infile, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {input_file} "
                    f"at line {line_number}: {e}"
                )

            conversation = example.get("conversation", [])
            clinical_summary = example.get("clinical_summary", {})

            if not conversation:
                raise ValueError(
                    f"Missing conversation in {input_file} "
                    f"at line {line_number}"
                )

            if not clinical_summary:
                raise ValueError(
                    f"Missing clinical_summary in {input_file} "
                    f"at line {line_number}"
                )

            # ------------------------------------------------
            # CREATE INPUT
            # ------------------------------------------------

            conversation_text = conversation_to_text(conversation)

            model_input = PROMPT + conversation_text

            # ------------------------------------------------
            # CREATE NEW SIMPLE TARGET
            # ------------------------------------------------

            target = create_target(clinical_summary)

            # ------------------------------------------------
            # OUTPUT RECORD
            # ------------------------------------------------

            output_record = {
                "id": example.get("id", f"{split_name}_{count + 1}"),
                "input": model_input,
                "target": target
            }

            outfile.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                ) + "\n"
            )

            count += 1

    print(f"{split_name}: {count} examples")
    print(f"Saved to: {output_file}")

    return count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FLAN-T5 DATA PREPARATION")
    print("Target format: labelled fields")
    print("=" * 70)

    total = 0

    for split in ["train", "validation", "test"]:
        total += process_split(split)
        print()

    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Total examples: {total}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # SHOW ONE SAMPLE
    # --------------------------------------------------------

    sample_file = OUTPUT_DIR / "train.jsonl"

    with open(sample_file, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    print("SAMPLE INPUT")
    print("-" * 70)
    print(sample["input"])

    print()
    print("SAMPLE TARGET")
    print("-" * 70)
    print(sample["target"])

    print()
    print("=" * 70)
    print("Expected target fields:", len(FIELDS))
    print("Expected fields:", ", ".join(FIELDS))
    print("=" * 70)


if __name__ == "__main__":
    main()