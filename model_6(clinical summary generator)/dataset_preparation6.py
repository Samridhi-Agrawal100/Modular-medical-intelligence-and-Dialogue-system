# ================================================================
# MODEL 6 — CLINICAL SUMMARY GENERATOR
# DATASET PREPARATION
# ================================================================

import json
from pathlib import Path


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "csg_processed"
PREPARED_DIR = BASE_DIR / "csg_prepared"

TRAIN_INPUT = PROCESSED_DIR / "train.jsonl"
VAL_INPUT = PROCESSED_DIR / "validation.jsonl"

TRAIN_OUTPUT = PREPARED_DIR / "train.jsonl"
VAL_OUTPUT = PREPARED_DIR / "validation.jsonl"


# ================================================================
# DISPLAY
# ================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# LOAD JSONL
# ================================================================

def load_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_number} "
                    f"in {path}\n{e}"
                )

            records.append(record)

    return records


# ================================================================
# VALIDATE PROCESSED RECORD
# ================================================================

def validate_record(record, index):

    if not isinstance(record, dict):
        raise ValueError(
            f"Record {index} is not a dictionary."
        )

    required_fields = [
        "id",
        "input_text",
        "target_text",
        "clinical_summary"
    ]

    for field in required_fields:

        if field not in record:
            raise ValueError(
                f"Record {index} ({record.get('id', 'UNKNOWN')}) "
                f"is missing field: {field}"
            )

    if not str(record["input_text"]).strip():
        raise ValueError(
            f"Record {record['id']} has empty input_text."
        )

    if not str(record["target_text"]).strip():
        raise ValueError(
            f"Record {record['id']} has empty target_text."
        )


# ================================================================
# PREPARE ONE RECORD
# ================================================================

def prepare_record(record, index):

    validate_record(
        record,
        index
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT create another prompt here.
    #
    # preproceesing6.py has already created the complete input:
    #
    # Summarize the following medical conversation into a
    # structured doctor-readable clinical summary.
    #
    # Conversation:
    # Patient: ...
    # Assistant: ...
    #
    # We simply reuse it.
    # ------------------------------------------------------------

    input_text = record["input_text"]

    target_text = record["target_text"]

    prepared_record = {
        "id": record["id"],
        "input_text": input_text,
        "target_text": target_text,
        "clinical_summary": record["clinical_summary"]
    }

    return prepared_record


# ================================================================
# PREPARE DATASET
# ================================================================

def prepare_dataset(records):

    prepared_records = []

    for index, record in enumerate(records):

        prepared = prepare_record(
            record,
            index
        )

        prepared_records.append(
            prepared
        )

    return prepared_records


# ================================================================
# SAVE JSONL
# ================================================================

def save_jsonl(records, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ================================================================
# CHECK FOR PROMPT DUPLICATION
# ================================================================

def check_prompt_duplication(records):

    problems = []

    for record in records:

        text = record["input_text"]

        instruction_count = text.count(
            "Summarize the following medical conversation"
        )

        conversation_count = text.count(
            "Conversation:"
        )

        if instruction_count != 1:

            problems.append(
                (
                    record["id"],
                    "instruction_count",
                    instruction_count
                )
            )

        if conversation_count != 1:

            problems.append(
                (
                    record["id"],
                    "conversation_count",
                    conversation_count
                )
            )

    return problems


# ================================================================
# CHECK TARGET FORMAT
# ================================================================

def check_target_format(records):

    required_sections = [
        "CLINICAL SUMMARY",
        "Chief Complaint:",
        "Onset:",
        "Duration:",
        "Location:",
        "Severity:",
        "Symptoms:",
        "Associated Symptoms:",
        "Negative Findings:",
        "Medical History:",
        "Medications:",
        "Allergies:",
        "Red Flags:",
        "Relevant Context:",
        "Recommendation:"
    ]

    problems = []

    for record in records:

        target = record["target_text"]

        missing = []

        for section in required_sections:

            if section not in target:
                missing.append(section)

        if missing:

            problems.append(
                {
                    "id": record["id"],
                    "missing_sections": missing
                }
            )

    return problems


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "CSG DATASET PREPARATION"
    )

    # ============================================================
    # PATHS
    # ============================================================

    print()
    print("Processed training file :")
    print(TRAIN_INPUT)

    print()
    print("Processed validation file :")
    print(VAL_INPUT)

    print()
    print("Prepared training file :")
    print(TRAIN_OUTPUT)

    print()
    print("Prepared validation file :")
    print(VAL_OUTPUT)

    # ============================================================
    # LOAD
    # ============================================================

    print_header(
        "LOADING DATASETS"
    )

    train_records = load_jsonl(
        TRAIN_INPUT
    )

    val_records = load_jsonl(
        VAL_INPUT
    )

    print(
        f"Loaded training examples   : "
        f"{len(train_records)}"
    )

    print(
        f"Loaded validation examples : "
        f"{len(val_records)}"
    )

    # ============================================================
    # PREPARE
    # ============================================================

    print_header(
        "PREPARING TRAINING DATASET"
    )

    prepared_train = prepare_dataset(
        train_records
    )

    print(
        f"Prepared training examples : "
        f"{len(prepared_train)}"
    )

    print_header(
        "PREPARING VALIDATION DATASET"
    )

    prepared_val = prepare_dataset(
        val_records
    )

    print(
        f"Prepared validation examples : "
        f"{len(prepared_val)}"
    )

    # ============================================================
    # PROMPT DUPLICATION CHECK
    # ============================================================

    print_header(
        "CHECKING PROMPT DUPLICATION"
    )

    train_prompt_problems = check_prompt_duplication(
        prepared_train
    )

    val_prompt_problems = check_prompt_duplication(
        prepared_val
    )

    total_prompt_problems = (
        len(train_prompt_problems)
        + len(val_prompt_problems)
    )

    if total_prompt_problems > 0:

        print(
            f"ERROR: {total_prompt_problems} "
            f"records have prompt duplication/format problems."
        )

        for problem in (
            train_prompt_problems[:5]
            + val_prompt_problems[:5]
        ):

            print(problem)

        raise ValueError(
            "Prompt duplication detected. "
            "Preparation stopped."
        )

    print(
        "✓ All records contain exactly one instruction."
    )

    print(
        "✓ All records contain exactly one 'Conversation:' heading."
    )

    # ============================================================
    # TARGET FORMAT CHECK
    # ============================================================

    print_header(
        "CHECKING STRUCTURED TARGETS"
    )

    train_target_problems = check_target_format(
        prepared_train
    )

    val_target_problems = check_target_format(
        prepared_val
    )

    total_target_problems = (
        len(train_target_problems)
        + len(val_target_problems)
    )

    if total_target_problems > 0:

        print(
            f"ERROR: {total_target_problems} "
            f"records have missing target sections."
        )

        for problem in (
            train_target_problems[:5]
            + val_target_problems[:5]
        ):

            print(problem)

        raise ValueError(
            "Structured target validation failed."
        )

    print(
        "✓ All training targets have the required sections."
    )

    print(
        "✓ All validation targets have the required sections."
    )

    # ============================================================
    # SAVE
    # ============================================================

    print_header(
        "SAVING PREPARED DATASETS"
    )

    save_jsonl(
        prepared_train,
        TRAIN_OUTPUT
    )

    save_jsonl(
        prepared_val,
        VAL_OUTPUT
    )

    print(
        f"Train      : {TRAIN_OUTPUT}"
    )

    print(
        f"Validation : {VAL_OUTPUT}"
    )

    # ============================================================
    # SAMPLE
    # ============================================================

    print_header(
        "SAMPLE TRAINING RECORD"
    )

    sample = prepared_train[0]

    print()
    print("ID:")
    print(sample["id"])

    print()
    print("INPUT:")
    print(sample["input_text"])

    print()
    print("TARGET:")
    print(sample["target_text"])

    print()
    print("STRUCTURED CLINICAL SUMMARY:")
    print(
        json.dumps(
            sample["clinical_summary"],
            indent=2,
            ensure_ascii=False
        )
    )

    # ============================================================
    # FINAL COUNTS
    # ============================================================

    print_header(
        "PREPARED DATASET"
    )

    print(
        f"Training examples   : "
        f"{len(prepared_train)}"
    )

    print(
        f"Validation examples : "
        f"{len(prepared_val)}"
    )

    print()
    print(
        "✓ Prompt duplication check passed."
    )

    print(
        "✓ Structured target check passed."
    )

    print(
        "✓ Dataset preparation complete."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Run token_analysis6.py."
    )

    print(
        "Do NOT train until token analysis is complete."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()