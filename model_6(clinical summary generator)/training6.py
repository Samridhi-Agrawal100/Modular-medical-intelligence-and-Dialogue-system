# ================================================================
# MODEL 6 — CLINICAL SUMMARY GENERATOR
# FINAL STRUCTURED TRAINING SCRIPT
# ================================================================

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)


# ================================================================
# 1. PATH CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "csg_prepared" / "train.jsonl"
VALIDATION_FILE = BASE_DIR / "csg_prepared" / "validation.jsonl"

# IMPORTANT:
# This is a NEW directory.
# It will NOT overwrite the old paragraph-summary model.
MODEL_DIR = BASE_DIR / "csg_model_structured"

MODEL_NAME = "google/flan-t5-small"


# ================================================================
# 2. LOCKED TRAINING CONFIGURATION
# ================================================================

MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 192

LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

WEIGHT_DECAY = 0.01

EARLY_STOPPING_PATIENCE = 2

SEED = 42


# ================================================================
# 3. REPRODUCIBILITY
# ================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ================================================================
# 4. STARTUP INFORMATION
# ================================================================

print("=" * 70)
print("MODEL 6 — CLINICAL SUMMARY GENERATOR")
print("STRUCTURED SUMMARY TRAINING")
print("=" * 70)

print()
print("Model name        :", MODEL_NAME)
print("Input max tokens  :", MAX_INPUT_LENGTH)
print("Target max tokens :", MAX_TARGET_LENGTH)
print("Epochs            :", NUM_EPOCHS)
print("Train batch size  :", TRAIN_BATCH_SIZE)
print("Eval batch size   :", EVAL_BATCH_SIZE)
print("Learning rate     :", LEARNING_RATE)
print("Weight decay      :", WEIGHT_DECAY)
print("Seed              :", SEED)

print()
print("PyTorch version   :", torch.__version__)
print("CUDA available    :", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU               :",
        torch.cuda.get_device_name(0)
    )

else:

    print("GPU               : CPU")

print()


# ================================================================
# 5. CHECK FILES
# ================================================================

if not TRAIN_FILE.exists():

    raise FileNotFoundError(
        f"\nTraining dataset not found:\n{TRAIN_FILE}"
    )


if not VALIDATION_FILE.exists():

    raise FileNotFoundError(
        f"\nValidation dataset not found:\n{VALIDATION_FILE}"
    )


# ================================================================
# 6. LOAD JSONL
# ================================================================

def load_jsonl(file_path):

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON at "
                    f"{file_path}, line {line_number}\n"
                    f"{error}"
                )

            records.append(record)

    return records


print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_records = load_jsonl(TRAIN_FILE)
validation_records = load_jsonl(VALIDATION_FILE)

print()
print("Training examples   :", len(train_records))
print("Validation examples :", len(validation_records))
print()


# ================================================================
# 7. VALIDATE RECORDS
# ================================================================

def validate_records(
    records,
    dataset_name
):

    required_fields = [
        "id",
        "input_text",
        "target_text",
        "clinical_summary",
    ]

    print(
        f"Validating {dataset_name} dataset..."
    )

    for index, record in enumerate(records):

        # --------------------------------------------------------
        # Required fields
        # --------------------------------------------------------

        for field in required_fields:

            if field not in record:

                raise ValueError(
                    f"{dataset_name} record {index} "
                    f"is missing field: {field}"
                )

        # --------------------------------------------------------
        # Input
        # --------------------------------------------------------

        if not isinstance(
            record["input_text"],
            str
        ):

            raise ValueError(
                f"{dataset_name} record {index}: "
                f"input_text must be a string"
            )

        # --------------------------------------------------------
        # Target
        # --------------------------------------------------------

        if not isinstance(
            record["target_text"],
            str
        ):

            raise ValueError(
                f"{dataset_name} record {index}: "
                f"target_text must be a string"
            )

        if not record["input_text"].strip():

            raise ValueError(
                f"{dataset_name} record {index}: "
                f"input_text is empty"
            )

        if not record["target_text"].strip():

            raise ValueError(
                f"{dataset_name} record {index}: "
                f"target_text is empty"
            )

        # --------------------------------------------------------
        # Prompt duplication checks
        # --------------------------------------------------------

        instruction_count = record[
            "input_text"
        ].count(
            "Summarize the following medical conversation"
        )

        conversation_count = record[
            "input_text"
        ].count(
            "Conversation:"
        )

        if instruction_count != 1:

            raise ValueError(
                f"{dataset_name} record {index}: "
                f"expected exactly 1 instruction, "
                f"found {instruction_count}"
            )

        if conversation_count != 1:

            raise ValueError(
                f"{dataset_name} record {index}: "
                f"expected exactly 1 Conversation heading, "
                f"found {conversation_count}"
            )

    print(
        f"✓ {dataset_name} dataset valid"
    )

    print()


validate_records(
    train_records,
    "Training"
)

validate_records(
    validation_records,
    "Validation"
)


# ================================================================
# 8. CREATE CLEAN HUGGINGFACE DATASETS
# ================================================================
#
# IMPORTANT:
#
# clinical_summary is intentionally NOT included.
#
# It is nested metadata containing lists and strings.
# The model only needs:
#
#       input_text
#       target_text
#
# ================================================================

train_dataset = Dataset.from_dict({

    "input_text": [
        record["input_text"]
        for record in train_records
    ],

    "target_text": [
        record["target_text"]
        for record in train_records
    ],
})


validation_dataset = Dataset.from_dict({

    "input_text": [
        record["input_text"]
        for record in validation_records
    ],

    "target_text": [
        record["target_text"]
        for record in validation_records
    ],
})


print("=" * 70)
print("DATASETS READY")
print("=" * 70)

print()
print(
    "Training rows      :",
    len(train_dataset)
)

print(
    "Validation rows    :",
    len(validation_dataset)
)

print()


# ================================================================
# 9. LOAD TOKENIZER
# ================================================================

print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print()
print("Tokenizer loaded successfully.")
print("Vocabulary size :", len(tokenizer))
print()


# ================================================================
# 10. LOAD MODEL
# ================================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print()
print("Model loaded successfully.")
print()


# ================================================================
# 11. TOKENIZATION
# ================================================================

def preprocess_function(examples):

    # ------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------

    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    # ------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels[
        "input_ids"
    ]

    return model_inputs


print("=" * 70)
print("TOKENIZING TRAINING DATA")
print("=" * 70)

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=[
        "input_text",
        "target_text",
    ],
    desc="Tokenizing training data",
)

print()
print("Training tokenization complete.")
print()


print("=" * 70)
print("TOKENIZING VALIDATION DATA")
print("=" * 70)

tokenized_validation = validation_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=[
        "input_text",
        "target_text",
    ],
    desc="Tokenizing validation data",
)

print()
print("Validation tokenization complete.")
print()


# ================================================================
# 12. DATA COLLATOR
# ================================================================

data_collator = DataCollatorForSeq2Seq(

    tokenizer=tokenizer,

    model=model,

    padding=True,
)


# ================================================================
# 13. TRAINING ARGUMENTS
# ================================================================
#
# IMPORTANT:
#
# predict_with_generate=False
#
# We do NOT generate summaries during every validation epoch.
#
# Why?
#
# Training should optimize validation loss.
# Clinical factual evaluation will be done separately after training.
#
# This also avoids the tokenizer/generation error encountered earlier.
#
# ================================================================

training_args = Seq2SeqTrainingArguments(

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    output_dir=str(MODEL_DIR),

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    weight_decay=WEIGHT_DECAY,

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    eval_strategy="epoch",

    # ------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    predict_with_generate=False,

    # ------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------

    # IMPORTANT:
    # Previous mixed-precision training caused NaN.
    # Keep both disabled.

    fp16=False,

    bf16=False,

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    logging_strategy="epoch",

    logging_first_step=True,

    report_to="none",

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    seed=SEED,

    data_seed=SEED,

    # ------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------

    dataloader_num_workers=0,

    # ------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------

    remove_unused_columns=True,
)


# ================================================================
# 14. TRAINER
# ================================================================

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    data_collator=data_collator,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=
            EARLY_STOPPING_PATIENCE
        )
    ],
)


# ================================================================
# 15. START TRAINING
# ================================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()
print(
    "Training examples :",
    len(tokenized_train)
)

print(
    "Validation        :",
    len(tokenized_validation)
)

print(
    "Max input length  :",
    MAX_INPUT_LENGTH
)

print(
    "Max target length :",
    MAX_TARGET_LENGTH
)

print("FP16              : False")
print("BF16              : False")
print("Generation eval   : False")
print()


train_result = trainer.train()


# ================================================================
# 16. SAVE BEST MODEL
# ================================================================

print()
print("=" * 70)
print("SAVING BEST MODEL")
print("=" * 70)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

trainer.save_model(
    str(MODEL_DIR)
)

tokenizer.save_pretrained(
    str(MODEL_DIR)
)

print()
print("✓ Model saved.")
print()
print("Model directory:")
print(MODEL_DIR)
print()


# ================================================================
# 17. FINAL VALIDATION LOSS
# ================================================================

print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

final_metrics = trainer.evaluate()

print()

for key, value in final_metrics.items():

    if isinstance(value, float):

        print(
            f"{key:25s}: {value:.6f}"
        )

    else:

        print(
            f"{key:25s}: {value}"
        )


# ================================================================
# 18. PERPLEXITY
# ================================================================

validation_loss = final_metrics.get(
    "eval_loss"
)

if validation_loss is not None:

    try:

        perplexity = math.exp(
            validation_loss
        )

    except OverflowError:

        perplexity = float("inf")

    print()
    print(
        f"Validation perplexity : "
        f"{perplexity:.6f}"
    )


# ================================================================
# 19. TRAINING METRICS
# ================================================================

print()
print("=" * 70)
print("TRAINING RESULTS")
print("=" * 70)

train_metrics = train_result.metrics

if "train_loss" in train_metrics:

    print(
        f"Final train loss      : "
        f"{train_metrics['train_loss']:.6f}"
    )

if validation_loss is not None:

    print(
        f"Final validation loss : "
        f"{validation_loss:.6f}"
    )


# ================================================================
# 20. SAVE METADATA
# ================================================================

metrics_output = {

    "model_name": MODEL_NAME,

    "training_examples": len(
        train_records
    ),

    "validation_examples": len(
        validation_records
    ),

    "max_input_length": MAX_INPUT_LENGTH,

    "max_target_length": MAX_TARGET_LENGTH,

    "learning_rate": LEARNING_RATE,

    "epochs_requested": NUM_EPOCHS,

    "train_batch_size": TRAIN_BATCH_SIZE,

    "eval_batch_size": EVAL_BATCH_SIZE,

    "weight_decay": WEIGHT_DECAY,

    "early_stopping_patience":
        EARLY_STOPPING_PATIENCE,

    "fp16": False,

    "bf16": False,

    "predict_with_generate": False,

    "seed": SEED,

    "train_metrics": train_metrics,

    "final_validation_metrics":
        final_metrics,
}


METRICS_FILE = (
    MODEL_DIR /
    "training_metrics.json"
)


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics_output,
        file,
        indent=4,
        ensure_ascii=False
    )


# ================================================================
# 21. FINAL MESSAGE
# ================================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("✓ Training completed")
print("✓ Best model restored")
print("✓ Validation loss calculated")
print("✓ Model saved")
print("✓ Tokenizer saved")
print("✓ Training metrics saved")

print()
print("MODEL:")
print(MODEL_DIR)

print()
print("METRICS:")
print(METRICS_FILE)

print()
print("=" * 70)