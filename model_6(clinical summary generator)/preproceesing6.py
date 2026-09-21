import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import random


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("model_6(clinical summary generator)\\dataset\\model6_summary_dataset_expanded_1000.json")

OUTPUT_DIR = Path("preprocessing6_output")

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

CLEAN_DATASET_FILE = OUTPUT_DIR / "clean_dataset.json"
INVALID_FILE = OUTPUT_DIR / "invalid_examples.json"
REPORT_FILE = OUTPUT_DIR / "preprocessing_report.json"


# Expected clinical-summary fields
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
    "relevant_context"
]


# Fields that should contain lists
LIST_FIELDS = [
    "complaint_characteristics",
    "associated_symptoms",
    "negative_findings",
    "medical_history",
    "medications",
    "allergies",
    "triggers",
    "relieving_factors"
]


# Fields that should contain strings
STRING_FIELDS = [
    "chief_complaint",
    "onset",
    "duration",
    "location",
    "severity",
    "relevant_context"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(text):
    """
    Basic text normalization for duplicate detection.
    Does NOT modify the actual dataset text.
    """
    if not isinstance(text, str):
        return ""

    return " ".join(text.lower().strip().split())


def conversation_to_text(conversation):
    """
    Convert structured conversation into one normalized string.

    Example:
    PATIENT: I have a headache.
    ASSISTANT: Where does it hurt?
    """
    parts = []

    for turn in conversation:
        role = str(turn.get("role", "")).strip().lower()
        text = normalize_text(turn.get("text", ""))

        parts.append(f"{role}: {text}")

    return "\n".join(parts)


def summary_fingerprint(summary):
    """
    Create a stable fingerprint for a clinical summary.
    Used only for duplicate analysis.
    """

    canonical = {}

    for field in EXPECTED_FIELDS:
        value = summary.get(field)

        if isinstance(value, list):
            canonical[field] = [
                normalize_text(str(x))
                for x in value
            ]

        elif isinstance(value, dict):
            canonical[field] = value

        else:
            canonical[field] = normalize_text(str(value))

    text = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path, examples):
    """
    Save examples in JSONL format.
    """

    with open(path, "w", encoding="utf-8") as f:

        for example in examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )


def validate_example(example):
    """
    Validate one dataset example.

    Returns:
        (True, None)
    or
        (False, reason)
    """

    if not isinstance(example, dict):
        return False, "Example is not a dictionary"

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if not isinstance(example.get("id"), str):
        return False, "Missing or invalid id"

    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    conversation = example.get("conversation")

    if not isinstance(conversation, list):
        return False, "conversation is not a list"

    if len(conversation) == 0:
        return False, "conversation is empty"

    for i, turn in enumerate(conversation):

        if not isinstance(turn, dict):
            return False, f"conversation turn {i} is not a dictionary"

        if turn.get("role") not in ["assistant", "patient"]:
            return False, f"invalid role in conversation turn {i}"

        if not isinstance(turn.get("text"), str):
            return False, f"invalid text in conversation turn {i}"

        if not turn.get("text").strip():
            return False, f"empty text in conversation turn {i}"

    # --------------------------------------------------------
    # Clinical summary
    # --------------------------------------------------------

    summary = example.get("clinical_summary")

    if not isinstance(summary, dict):
        return False, "clinical_summary is not a dictionary"

    missing_fields = [
        field
        for field in EXPECTED_FIELDS
        if field not in summary
    ]

    if missing_fields:
        return False, f"missing fields: {missing_fields}"

    # --------------------------------------------------------
    # String fields
    # --------------------------------------------------------

    for field in STRING_FIELDS:

        value = summary.get(field)

        if not isinstance(value, str):
            return False, (
                f"{field} should be a string, "
                f"got {type(value).__name__}"
            )

    # --------------------------------------------------------
    # List fields
    # --------------------------------------------------------

    for field in LIST_FIELDS:

        value = summary.get(field)

        if not isinstance(value, list):
            return False, (
                f"{field} should be a list, "
                f"got {type(value).__name__}"
            )

    # --------------------------------------------------------
    # Red flags
    # --------------------------------------------------------

    red_flags = summary.get("red_flags")

    if not isinstance(red_flags, dict):
        return False, "red_flags should be a dictionary"

    if "status" not in red_flags:
        return False, "red_flags missing status"

    if "items" not in red_flags:
        return False, "red_flags missing items"

    if not isinstance(red_flags["items"], list):
        return False, "red_flags.items should be a list"

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = example.get("metadata")

    if not isinstance(metadata, dict):
        return False, "metadata is not a dictionary"

    if not isinstance(metadata.get("scenario_id"), str):
        return False, "missing or invalid scenario_id"

    if not isinstance(metadata.get("paraphrase_id"), int):
        return False, "missing or invalid paraphrase_id"

    return True, None


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE.resolve()}\n\n"
        f"Current working directory:\n{Path.cwd()}"
    )


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)


# ============================================================
# READ EXAMPLES
# ============================================================

if not isinstance(dataset, dict):
    raise ValueError("Top-level dataset must be a dictionary.")

if "examples" not in dataset:
    raise ValueError("Dataset does not contain an 'examples' key.")

examples = dataset["examples"]

if not isinstance(examples, list):
    raise ValueError("'examples' must be a list.")


print(f"Dataset name: {dataset.get('dataset_name')}")
print(f"Total examples: {len(examples)}")

print(
    f"Declared unique scenarios: "
    f"{dataset.get('num_unique_clinical_summary_scenarios')}"
)

print(
    f"Declared paraphrases per scenario: "
    f"{dataset.get('paraphrases_per_scenario')}"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING DATASET")
print("=" * 70)

valid_examples = []
invalid_examples = []

for example in examples:

    valid, reason = validate_example(example)

    if valid:

        valid_examples.append(example)

    else:

        invalid_examples.append({
            "id": example.get("id") if isinstance(example, dict) else None,
            "reason": reason,
            "example": example
        })


print(f"Valid examples: {len(valid_examples)}")
print(f"Invalid examples: {len(invalid_examples)}")


# ============================================================
# STOP IF INVALID DATA
# ============================================================

if invalid_examples:

    print("\nFirst 5 invalid examples:")

    for item in invalid_examples[:5]:
        print(
            f"\nID: {item['id']}"
            f"\nReason: {item['reason']}"
        )


# ============================================================
# DUPLICATE ID ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

id_counter = Counter(
    example["id"]
    for example in valid_examples
)

duplicate_ids = {
    k: v
    for k, v in id_counter.items()
    if v > 1
}


print(f"Duplicate IDs: {len(duplicate_ids)}")


# ============================================================
# CONVERSATION DUPLICATES
# ============================================================

conversation_groups = defaultdict(list)

for example in valid_examples:

    conversation_text = conversation_to_text(
        example["conversation"]
    )

    conversation_groups[conversation_text].append(
        example["id"]
    )


duplicate_conversation_groups = {
    text: ids
    for text, ids in conversation_groups.items()
    if len(ids) > 1
}


examples_in_duplicate_conversations = sum(
    len(ids)
    for ids in duplicate_conversation_groups.values()
)


print(
    f"Unique conversation texts: "
    f"{len(conversation_groups)}"
)

print(
    f"Exact duplicate conversation groups: "
    f"{len(duplicate_conversation_groups)}"
)

print(
    f"Examples involved in exact duplicates: "
    f"{examples_in_duplicate_conversations}"
)


# ============================================================
# CLINICAL SUMMARY DUPLICATES
# ============================================================

summary_groups = defaultdict(list)

for example in valid_examples:

    fingerprint = summary_fingerprint(
        example["clinical_summary"]
    )

    summary_groups[fingerprint].append(
        example["id"]
    )


duplicate_summary_groups = {
    fingerprint: ids
    for fingerprint, ids in summary_groups.items()
    if len(ids) > 1
}


print(
    f"Unique clinical summaries: "
    f"{len(summary_groups)}"
)

print(
    f"Clinical summaries appearing more than once: "
    f"{len(duplicate_summary_groups)}"
)


# ============================================================
# SCENARIO GROUPING
# ============================================================

print("\n" + "=" * 70)
print("CLINICAL SCENARIO GROUPING")
print("=" * 70)

scenario_groups = defaultdict(list)

for example in valid_examples:

    scenario_id = example["metadata"]["scenario_id"]

    scenario_groups[scenario_id].append(
        example
    )


print(
    f"Unique clinical scenarios: "
    f"{len(scenario_groups)}"
)


scenario_sizes = [
    len(group)
    for group in scenario_groups.values()
]


if scenario_sizes:

    print(
        f"Examples per scenario: "
        f"min={min(scenario_sizes)}, "
        f"max={max(scenario_sizes)}, "
        f"average={sum(scenario_sizes) / len(scenario_sizes):.2f}"
    )


# ============================================================
# CHIEF COMPLAINT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CHIEF COMPLAINT DISTRIBUTION")
print("=" * 70)

complaint_counter = Counter(
    example["clinical_summary"]["chief_complaint"]
    for example in valid_examples
)


for complaint, count in sorted(
    complaint_counter.items()
):

    percentage = (
        count / len(valid_examples) * 100
        if valid_examples
        else 0
    )

    print(
        f"{complaint}: "
        f"{count} ({percentage:.2f}%)"
    )


# ============================================================
# FIELD MISSINGNESS
# ============================================================

print("\n" + "=" * 70)
print("FIELD MISSINGNESS")
print("=" * 70)

missingness = {}

for field in EXPECTED_FIELDS:

    missing_count = 0

    for example in valid_examples:

        value = example["clinical_summary"].get(field)

        if value is None:
            missing_count += 1

        elif isinstance(value, str) and not value.strip():
            missing_count += 1

        elif isinstance(value, list) and len(value) == 0:
            missing_count += 1

    percentage = (
        missing_count / len(valid_examples) * 100
        if valid_examples
        else 0
    )

    missingness[field] = {
        "count": missing_count,
        "percentage": round(percentage, 2)
    }

    print(
        f"{field}: "
        f"{missing_count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

print("\n" + "=" * 70)
print("GROUP-AWARE DATA SPLIT")
print("=" * 70)


random.seed(42)

scenario_ids = list(scenario_groups.keys())

random.shuffle(scenario_ids)

num_scenarios = len(scenario_ids)

if num_scenarios < 3:

    raise ValueError(
        "Not enough clinical scenarios for "
        "train/validation/test splitting."
    )


# ------------------------------------------------------------
# 80 / 10 / 10 scenario split
# ------------------------------------------------------------

train_count = int(num_scenarios * 0.80)
validation_count = int(num_scenarios * 0.10)

# Ensure at least one validation and one test scenario
train_count = max(train_count, 1)
validation_count = max(validation_count, 1)

test_count = (
    num_scenarios
    - train_count
    - validation_count
)

if test_count < 1:

    test_count = 1
    train_count -= 1


train_scenarios = set(
    scenario_ids[:train_count]
)

validation_scenarios = set(
    scenario_ids[
        train_count:
        train_count + validation_count
    ]
)

test_scenarios = set(
    scenario_ids[
        train_count + validation_count:
    ]
)


# ------------------------------------------------------------
# Create example splits
# ------------------------------------------------------------

train_examples = []
validation_examples = []
test_examples = []


for scenario_id in train_scenarios:

    train_examples.extend(
        scenario_groups[scenario_id]
    )


for scenario_id in validation_scenarios:

    validation_examples.extend(
        scenario_groups[scenario_id]
    )


for scenario_id in test_scenarios:

    test_examples.extend(
        scenario_groups[scenario_id]
    )


# Shuffle examples inside each split
random.shuffle(train_examples)
random.shuffle(validation_examples)
random.shuffle(test_examples)


print(
    f"Train scenarios: "
    f"{len(train_scenarios)}"
)

print(
    f"Validation scenarios: "
    f"{len(validation_scenarios)}"
)

print(
    f"Test scenarios: "
    f"{len(test_scenarios)}"
)

print()

print(
    f"Train examples: "
    f"{len(train_examples)}"
)

print(
    f"Validation examples: "
    f"{len(validation_examples)}"
)

print(
    f"Test examples: "
    f"{len(test_examples)}"
)


total_split_examples = (
    len(train_examples)
    + len(validation_examples)
    + len(test_examples)
)


print(
    f"Total split examples: "
    f"{total_split_examples}"
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)


train_ids = {
    example["metadata"]["scenario_id"]
    for example in train_examples
}

validation_ids = {
    example["metadata"]["scenario_id"]
    for example in validation_examples
}

test_ids = {
    example["metadata"]["scenario_id"]
    for example in test_examples
}


train_validation_overlap = (
    train_ids & validation_ids
)

train_test_overlap = (
    train_ids & test_ids
)

validation_test_overlap = (
    validation_ids & test_ids
)


print(
    "Train ∩ Validation:",
    len(train_validation_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(validation_test_overlap)
)


if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):

    raise ValueError(
        "SCENARIO LEAKAGE DETECTED!"
    )


print("\n✓ No clinical scenario leakage detected.")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE JSONL FILES
# ============================================================

print("\n" + "=" * 70)
print("SAVING DATASETS")
print("=" * 70)


write_jsonl(
    TRAIN_FILE,
    train_examples
)

write_jsonl(
    VALIDATION_FILE,
    validation_examples
)

write_jsonl(
    TEST_FILE,
    test_examples
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

clean_dataset = {
    "dataset_name": dataset.get("dataset_name"),
    "description": dataset.get("description"),
    "num_examples": len(valid_examples),
    "num_unique_clinical_summary_scenarios": len(
        scenario_groups
    ),
    "schema": EXPECTED_FIELDS,
    "examples": valid_examples
}


with open(
    CLEAN_DATASET_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        clean_dataset,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE INVALID EXAMPLES
# ============================================================

with open(
    INVALID_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        invalid_examples,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PREPROCESSING REPORT
# ============================================================

report = {

    "input_file": str(INPUT_FILE),

    "original_examples": len(examples),

    "valid_examples": len(valid_examples),

    "invalid_examples": len(invalid_examples),

    "duplicate_ids": len(duplicate_ids),

    "unique_conversations": len(
        conversation_groups
    ),

    "exact_duplicate_conversation_groups": len(
        duplicate_conversation_groups
    ),

    "unique_clinical_summaries": len(
        summary_groups
    ),

    "duplicate_clinical_summary_groups": len(
        duplicate_summary_groups
    ),

    "unique_clinical_scenarios": len(
        scenario_groups
    ),

    "chief_complaint_distribution": dict(
        complaint_counter
    ),

    "field_missingness": missingness,

    "split": {

        "train": {
            "examples": len(train_examples),
            "scenarios": len(train_scenarios)
        },

        "validation": {
            "examples": len(validation_examples),
            "scenarios": len(validation_scenarios)
        },

        "test": {
            "examples": len(test_examples),
            "scenarios": len(test_scenarios)
        }
    },

    "leakage": {

        "train_validation_overlap": len(
            train_validation_overlap
        ),

        "train_test_overlap": len(
            train_test_overlap
        ),

        "validation_test_overlap": len(
            validation_test_overlap
        )
    },

    "random_seed": 42
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"""
Input:
    {INPUT_FILE}

Output directory:
    {OUTPUT_DIR}

Files created:

    {TRAIN_FILE}
    {VALIDATION_FILE}
    {TEST_FILE}
    {CLEAN_DATASET_FILE}
    {INVALID_FILE}
    {REPORT_FILE}

Dataset:

    Original examples : {len(examples)}
    Valid examples    : {len(valid_examples)}
    Invalid examples  : {len(invalid_examples)}

Scenarios:

    Total scenarios   : {len(scenario_groups)}
    Train scenarios   : {len(train_scenarios)}
    Validation        : {len(validation_scenarios)}
    Test scenarios    : {len(test_scenarios)}

Examples:

    Train             : {len(train_examples)}
    Validation        : {len(validation_examples)}
    Test              : {len(test_examples)}

Scenario leakage:

    Train/Validation  : {len(train_validation_overlap)}
    Train/Test        : {len(train_test_overlap)}
    Validation/Test   : {len(validation_test_overlap)}
""")

print("=" * 70)
print("DONE")
print("=" * 70)