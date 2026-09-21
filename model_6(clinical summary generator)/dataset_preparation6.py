import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("model_6(clinical summary generator)\\preprocessing6_output")
OUTPUT_DIR = Path("flan_t5_data")

TRAIN_INPUT = INPUT_DIR / "train.jsonl"
VALIDATION_INPUT = INPUT_DIR / "validation.jsonl"
TEST_INPUT = INPUT_DIR / "test.jsonl"

TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation.jsonl"
TEST_OUTPUT = OUTPUT_DIR / "test.jsonl"


# ============================================================
# CLINICAL SUMMARY SCHEMA
# ============================================================

SUMMARY_FIELDS = [
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
    "relevant_context"
]


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    examples = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
                examples.append(example)

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {path}: {e}"
                )

    return examples


# ============================================================
# FORMAT CONVERSATION
# ============================================================

def format_conversation(conversation):

    formatted_turns = []

    for turn in conversation:

        role = turn["role"].strip().upper()
        text = turn["text"].strip()

        if role == "PATIENT":

            formatted_turns.append(
                f"Patient: {text}"
            )

        elif role == "ASSISTANT":

            formatted_turns.append(
                f"Assistant: {text}"
            )

        else:

            raise ValueError(
                f"Unknown conversation role: {role}"
            )

    return "\n".join(formatted_turns)


# ============================================================
# CREATE TARGET SUMMARY
# ============================================================

def create_target_summary(clinical_summary):

    # Keep exactly the required 15 fields.
    target = {}

    for field in SUMMARY_FIELDS:

        if field not in clinical_summary:

            raise ValueError(
                f"Missing clinical summary field: {field}"
            )

        target[field] = clinical_summary[field]

    # JSON string becomes the target sequence
    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":")
    )


# ============================================================
# CONVERT ONE EXAMPLE
# ============================================================

def convert_example(example):

    if "conversation" not in example:
        raise ValueError("Example missing conversation")

    if "clinical_summary" not in example:
        raise ValueError("Example missing clinical_summary")

    conversation = format_conversation(
        example["conversation"]
    )

    target = create_target_summary(
        example["clinical_summary"]
    )

    # This is the actual instruction given to FLAN-T5.
    source = (
        "Generate a structured clinical summary from the "
        "following patient-assistant conversation.\n\n"
        "Return the clinical summary using exactly these 15 "
        "fields:\n"
        "chief_complaint, onset, duration, location, severity, "
        "complaint_characteristics, associated_symptoms, "
        "negative_findings, medical_history, medications, "
        "allergies, red_flags, triggers, relieving_factors, "
        "relevant_context.\n\n"
        "Do not add information that is not present in the "
        "conversation.\n\n"
        "Conversation:\n"
        + conversation
    )

    return {
        "input": source,
        "target": target
    }


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(input_file, output_file):

    print(f"\nProcessing: {input_file}")

    examples = load_jsonl(input_file)

    converted = []

    for index, example in enumerate(examples):

        try:

            converted_example = convert_example(
                example
            )

            converted.append(
                converted_example
            )

        except Exception as e:

            raise ValueError(
                f"\nError processing example "
                f"{index + 1} in {input_file}\n"
                f"Example ID: {example.get('id')}\n"
                f"Error: {e}"
            )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for example in converted:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Examples written: {len(converted)}"
    )

    print(
        f"Output: {output_file}"
    )

    return converted


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FLAN-T5 TRAINING DATA PREPARATION")
print("=" * 70)


# Check input files

for path in [
    TRAIN_INPUT,
    VALIDATION_INPUT,
    TEST_INPUT
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{path.resolve()}\n\n"
            f"Current working directory:\n"
            f"{Path.cwd()}"
        )


# Create output directory

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS DATASETS
# ============================================================

train_data = process_split(
    TRAIN_INPUT,
    TRAIN_OUTPUT
)

validation_data = process_split(
    VALIDATION_INPUT,
    VALIDATION_OUTPUT
)

test_data = process_split(
    TEST_INPUT,
    TEST_OUTPUT
)


# ============================================================
# VERIFY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING DATA")
print("=" * 70)

print(
    f"Train examples:      {len(train_data)}"
)

print(
    f"Validation examples: {len(validation_data)}"
)

print(
    f"Test examples:       {len(test_data)}"
)


expected_train = 800
expected_validation = 100
expected_test = 100


if len(train_data) != expected_train:

    raise ValueError(
        f"Expected {expected_train} training examples, "
        f"got {len(train_data)}"
    )


if len(validation_data) != expected_validation:

    raise ValueError(
        f"Expected {expected_validation} validation examples, "
        f"got {len(validation_data)}"
    )


if len(test_data) != expected_test:

    raise ValueError(
        f"Expected {expected_test} test examples, "
        f"got {len(test_data)}"
    )


# ============================================================
# VERIFY FIELDS
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING TARGET STRUCTURE")
print("=" * 70)


for split_name, data in [
    ("train", train_data),
    ("validation", validation_data),
    ("test", test_data)
]:

    for index, example in enumerate(data):

        if "input" not in example:
            raise ValueError(
                f"{split_name} example {index} "
                f"missing input"
            )

        if "target" not in example:
            raise ValueError(
                f"{split_name} example {index} "
                f"missing target"
            )

        target = json.loads(
            example["target"]
        )

        actual_fields = list(target.keys())

        if actual_fields != SUMMARY_FIELDS:

            raise ValueError(
                f"{split_name} example {index} "
                f"has incorrect target fields.\n"
                f"Expected: {SUMMARY_FIELDS}\n"
                f"Got: {actual_fields}"
            )


print("✓ All targets contain exactly 15 fields.")


# ============================================================
# SHOW ONE EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TRAINING EXAMPLE")
print("=" * 70)

sample = train_data[0]

print("\nINPUT:")
print("-" * 70)
print(sample["input"])

print("\nTARGET:")
print("-" * 70)
print(
    json.dumps(
        json.loads(sample["target"]),
        indent=2,
        ensure_ascii=False
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PREPARATION COMPLETE")
print("=" * 70)

print(
    f"""
Output directory:

    {OUTPUT_DIR}

Files:

    {TRAIN_OUTPUT}
    {VALIDATION_OUTPUT}
    {TEST_OUTPUT}

Dataset:

    Train       : {len(train_data)}
    Validation  : {len(validation_data)}
    Test        : {len(test_data)}

Total         : {
    len(train_data)
    + len(validation_data)
    + len(test_data)
}
"""
)

print("=" * 70)
print("READY FOR FLAN-T5 TRAINING")
print("=" * 70)