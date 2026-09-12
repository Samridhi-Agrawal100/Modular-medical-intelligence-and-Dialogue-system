# ================================================================
# MODEL 6 — CLINICAL SUMMARY GENERATOR
# STRUCTURED TARGET PREPROCESSING
# ================================================================

import json
import random
from pathlib import Path


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "dataset" / "model6_dummy_dataset.json"
OUTPUT_DIR = BASE_DIR / "csg_processed"

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
REPORT_FILE = OUTPUT_DIR / "preprocessing_report.json"

TRAIN_SIZE = 450
VALIDATION_SIZE = 50
RANDOM_SEED = 42


# ================================================================
# DISPLAY HELPERS
# ================================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ================================================================
# LOAD RAW DATASET
# ================================================================

def load_raw_dataset(path):
    """
    Supports:
        1. JSON list
        2. {"data": [...]}
        3. {"examples": [...]}
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            records = data["data"]

        elif "examples" in data and isinstance(data["examples"], list):
            records = data["examples"]

        else:
            raise ValueError(
                "JSON object found, but neither 'data' nor 'examples' "
                "contains a list."
            )

    else:
        raise ValueError("Unsupported JSON dataset format.")

    return records


# ================================================================
# FORMAT HELPERS
# ================================================================

def format_scalar(value):
    """
    Formats a scalar field.

    Empty / missing values become:
        Not reported
    """

    if value is None:
        return "Not reported"

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return "Not reported"

        return value

    return str(value)


def format_list(value):
    """
    Formats list-valued clinical fields into bullet points.

    Example:

        ["fever", "cough"]

    becomes:

        - fever
        - cough

    Empty list becomes:

        Not reported
    """

    if value is None:
        return "Not reported"

    if not isinstance(value, list):
        value = [value]

    cleaned = []

    for item in value:
        if item is None:
            continue

        item = str(item).strip()

        if item:
            cleaned.append(item)

    if not cleaned:
        return "Not reported"

    return "\n".join(f"- {item}" for item in cleaned)


# ================================================================
# BUILD STRUCTURED CLINICAL SUMMARY TARGET
# ================================================================

def build_structured_target(clinical_summary):
    """
    Converts the structured clinical_summary object from the raw
    dataset into the exact doctor-readable target format.

    IMPORTANT:
    This function does NOT invent medical information.
    It only formats information already present in the dataset.
    """

    if not isinstance(clinical_summary, dict):
        raise ValueError(
            "clinical_summary must be a dictionary."
        )

    chief_complaint = format_scalar(
        clinical_summary.get("chief_complaint")
    )

    onset = format_scalar(
        clinical_summary.get("onset")
    )

    duration = format_scalar(
        clinical_summary.get("duration")
    )

    location = format_scalar(
        clinical_summary.get("location")
    )

    severity = format_scalar(
        clinical_summary.get("severity")
    )

    symptoms = format_list(
        clinical_summary.get("symptoms")
    )

    associated_symptoms = format_list(
        clinical_summary.get("associated_symptoms")
    )

    negative_findings = format_list(
        clinical_summary.get("negative_findings")
    )

    medical_history = format_list(
        clinical_summary.get("medical_history")
    )

    medications = format_list(
        clinical_summary.get("medications")
    )

    allergies = format_list(
        clinical_summary.get("allergies")
    )

    red_flags = format_list(
        clinical_summary.get("red_flags")
    )

    relevant_context = format_scalar(
        clinical_summary.get("relevant_context")
    )

    target = f"""CLINICAL SUMMARY

Chief Complaint:
{chief_complaint}

Onset:
{onset}

Duration:
{duration}

Location:
{location}

Severity:
{severity}

Symptoms:
{symptoms}

Associated Symptoms:
{associated_symptoms}

Negative Findings:
{negative_findings}

Medical History:
{medical_history}

Medications:
{medications}

Allergies:
{allergies}

Red Flags:
{red_flags}

Relevant Context:
{relevant_context}

Recommendation:
Further clinical evaluation recommended.
"""

    return target.strip()


# ================================================================
# BUILD INPUT TEXT FROM RAW CONVERSATION
# ================================================================

def build_input_text(conversation):
    """
    Converts the raw conversation list into the model input.

    IMPORTANT:
    This function creates the prompt exactly ONCE.

    Expected output:

    Summarize the following medical conversation into a structured
    doctor-readable clinical summary.

    Conversation:
    Patient: ...
    Assistant: ...
    """

    if not isinstance(conversation, list):
        raise ValueError(
            "conversation must be a list."
        )

    lines = []

    # ------------------------------------------------------------
    # ONE instruction only
    # ------------------------------------------------------------

    lines.append(
        "Summarize the following medical conversation into a "
        "structured doctor-readable clinical summary."
    )

    lines.append("")
    lines.append("Conversation:")

    # ------------------------------------------------------------
    # Add conversation turns
    # ------------------------------------------------------------

    for turn in conversation:

        if not isinstance(turn, dict):
            raise ValueError(
                "Each conversation turn must be a dictionary."
            )

        role = str(
            turn.get("role", "")
        ).strip().lower()

        text = str(
            turn.get("text", "")
        ).strip()

        if not text:
            continue

        if role == "patient":
            lines.append(f"Patient: {text}")

        elif role == "assistant":
            lines.append(f"Assistant: {text}")

        else:
            raise ValueError(
                f"Unknown conversation role: {role}"
            )

    return "\n".join(lines)


# ================================================================
# VALIDATE RAW RECORD
# ================================================================

def validate_raw_record(record):
    """
    Validates the raw dataset record BEFORE processing.

    Required:
        id
        conversation
        clinical_summary
    """

    if not isinstance(record, dict):
        return False, "Record is not a dictionary."

    # ------------------------------------------------------------
    # ID
    # ------------------------------------------------------------

    record_id = record.get("id")

    if record_id is None:
        return False, "Missing id."

    if not str(record_id).strip():
        return False, "Empty id."

    # ------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------

    conversation = record.get("conversation")

    if not isinstance(conversation, list):
        return False, "conversation is not a list."

    if len(conversation) == 0:
        return False, "conversation is empty."

    for index, turn in enumerate(conversation):

        if not isinstance(turn, dict):
            return False, (
                f"Conversation turn {index} is not a dictionary."
            )

        role = str(
            turn.get("role", "")
        ).strip().lower()

        text = str(
            turn.get("text", "")
        ).strip()

        if role not in {"patient", "assistant"}:
            return False, (
                f"Invalid role '{role}' at turn {index}."
            )

        if not text:
            return False, (
                f"Empty text at conversation turn {index}."
            )

    # ------------------------------------------------------------
    # Clinical summary
    # ------------------------------------------------------------

    clinical_summary = record.get("clinical_summary")

    if not isinstance(clinical_summary, dict):
        return False, "clinical_summary is missing or invalid."

    if not str(
        clinical_summary.get("chief_complaint", "")
    ).strip():
        return False, "Missing chief_complaint."

    return True, None


# ================================================================
# PROCESS ONE RECORD
# ================================================================

def process_record(record):
    """
    Creates one processed record.

    Output:

    {
        "id": "...",
        "input_text": "...",
        "target_text": "...",
        "clinical_summary": {...}
    }
    """

    conversation = record["conversation"]
    clinical_summary = record["clinical_summary"]

    input_text = build_input_text(
        conversation
    )

    target_text = build_structured_target(
        clinical_summary
    )

    processed = {
        "id": str(record["id"]),
        "input_text": input_text,
        "target_text": target_text,
        "clinical_summary": clinical_summary,
    }

    return processed


# ================================================================
# SAVE JSONL
# ================================================================

def save_jsonl(records, path):

    with open(path, "w", encoding="utf-8") as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "MODEL 6 — CLINICAL SUMMARY GENERATOR\n"
        "STRUCTURED TARGET PREPROCESSING"
    )

    # ============================================================
    # PATHS
    # ============================================================

    print_header("PATHS")

    print(f"Raw dataset : {RAW_FILE}")
    print(f"Output dir  : {OUTPUT_DIR}")

    # ============================================================
    # CHECK RAW DATASET
    # ============================================================

    print_header("CHECKING RAW DATASET")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}"
        )

    print("Raw dataset found.")

    # ============================================================
    # LOAD DATASET
    # ============================================================

    print_header("LOADING RAW DATASET")

    raw_records = load_raw_dataset(
        RAW_FILE
    )

    print(
        f"Raw examples : {len(raw_records)}"
    )

    # ============================================================
    # PROCESS RECORDS
    # ============================================================

    print_header("PROCESSING RECORDS")

    processed_records = []

    invalid_records = []

    seen_ids = set()
    seen_conversations = set()

    duplicate_ids = 0
    duplicate_conversations = 0

    for index, record in enumerate(raw_records):

        # --------------------------------------------------------
        # Validate
        # --------------------------------------------------------

        valid, error = validate_raw_record(
            record
        )

        if not valid:

            invalid_records.append(
                {
                    "index": index,
                    "id": record.get("id")
                    if isinstance(record, dict)
                    else None,
                    "error": error,
                }
            )

            continue

        # --------------------------------------------------------
        # Duplicate ID check
        # --------------------------------------------------------

        record_id = str(
            record["id"]
        ).strip()

        if record_id in seen_ids:
            duplicate_ids += 1
            continue

        seen_ids.add(record_id)

        # --------------------------------------------------------
        # Duplicate conversation check
        # --------------------------------------------------------

        conversation_key = json.dumps(
            record["conversation"],
            sort_keys=True,
            ensure_ascii=False
        )

        if conversation_key in seen_conversations:
            duplicate_conversations += 1
            continue

        seen_conversations.add(
            conversation_key
        )

        # --------------------------------------------------------
        # Process
        # --------------------------------------------------------

        try:

            processed = process_record(
                record
            )

            processed_records.append(
                processed
            )

        except Exception as e:

            invalid_records.append(
                {
                    "index": index,
                    "id": record_id,
                    "error": str(e),
                }
            )

    # ============================================================
    # DATASET VALIDATION
    # ============================================================

    print_header("DATASET VALIDATION")

    print(
        f"Raw examples          : {len(raw_records)}"
    )

    print(
        f"Valid examples        : {len(processed_records)}"
    )

    print(
        f"Invalid examples      : {len(invalid_records)}"
    )

    print(
        f"Duplicate IDs         : {duplicate_ids}"
    )

    print(
        f"Duplicate conversations: {duplicate_conversations}"
    )

    # ============================================================
    # STOP IF DATASET IS INVALID
    # ============================================================

    if len(processed_records) == 0:

        raise ValueError(
            "No valid examples were produced. "
            "Check the raw dataset schema and preprocessing code."
        )

    # ============================================================
    # SPLIT DATASET
    # ============================================================

    print_header("DATASET SPLIT")

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        processed_records
    )

    total_examples = len(
        processed_records
    )

    # ------------------------------------------------------------
    # We expect 450 / 50 for the current 500-example dataset.
    # For a future dataset, the code still works as long as there
    # are enough records.
    # ------------------------------------------------------------

    if total_examples < (
        TRAIN_SIZE + VALIDATION_SIZE
    ):

        raise ValueError(
            f"Not enough valid examples.\n"
            f"Required: {TRAIN_SIZE + VALIDATION_SIZE}\n"
            f"Found: {total_examples}"
        )

    train_records = processed_records[
        :TRAIN_SIZE
    ]

    validation_records = processed_records[
        TRAIN_SIZE:
        TRAIN_SIZE + VALIDATION_SIZE
    ]

    print(
        f"Training examples     : {len(train_records)}"
    )

    print(
        f"Validation examples   : {len(validation_records)}"
    )

    print(
        f"Random seed           : {RANDOM_SEED}"
    )

    # ============================================================
    # CREATE OUTPUT DIRECTORY
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # SAVE PROCESSED DATA
    # ============================================================

    print_header("SAVING PROCESSED DATA")

    save_jsonl(
        train_records,
        TRAIN_FILE
    )

    save_jsonl(
        validation_records,
        VALIDATION_FILE
    )

    print(
        f"Training file   : {TRAIN_FILE}"
    )

    print(
        f"Validation file : {VALIDATION_FILE}"
    )

    # ============================================================
    # SAMPLE RECORD
    # ============================================================

    print_header(
        "SAMPLE STRUCTURED RECORD"
    )

    sample = train_records[0]

    print()
    print("ID:")
    print(sample["id"])

    print()
    print("INPUT:")
    print(sample["input_text"])

    print()
    print("STRUCTURED TARGET:")
    print(sample["target_text"])

    print()
    print("CLINICAL SUMMARY OBJECT:")
    print(
        json.dumps(
            sample["clinical_summary"],
            indent=2,
            ensure_ascii=False
        )
    )

    # ============================================================
    # PREPROCESSING REPORT
    # ============================================================

    report = {
        "dataset_name": "clinical_summary_dummy_dataset",
        "raw_examples": len(raw_records),
        "valid_examples": len(processed_records),
        "invalid_examples": len(invalid_records),
        "duplicate_ids": duplicate_ids,
        "duplicate_conversations": duplicate_conversations,
        "training_examples": len(train_records),
        "validation_examples": len(validation_records),
        "random_seed": RANDOM_SEED,
        "target_type": "structured_clinical_summary",
        "uses_summary_text": False,
        "invalid_records": invalid_records,
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
    # FINAL
    # ============================================================

    print_header(
        "PREPROCESSING COMPLETE"
    )

    print()
    print("Generated:")
    print("  ✓ train.jsonl")
    print("  ✓ validation.jsonl")
    print("  ✓ preprocessing_report.json")

    print()
    print("IMPORTANT:")
    print(
        "The target is now a structured clinical summary."
    )

    print(
        "The old paragraph-style summary_text is NOT used."
    )

    print()
    print("Next step:")
    print(
        "Run dataset_preparation6.py, then perform token analysis."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()