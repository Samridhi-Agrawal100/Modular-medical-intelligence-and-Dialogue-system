from pathlib import Path
import inspect
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "flan_t5_data"
OUTPUT_DIR = BASE_DIR / "flan_t5_summary_model"

TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"


# ============================================================
# 2. MODEL
# ============================================================

MODEL_NAME = "google/flan-t5-small"

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 256


# ============================================================
# 3. CHECK FILES
# ============================================================

for file_path in [TRAIN_FILE, VALIDATION_FILE, TEST_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{file_path}\n\n"
            f"Make sure flan_t5_data contains train.jsonl, "
            f"validation.jsonl and test.jsonl."
        )

print("=" * 70)
print("MODEL 6 - CLINICAL SUMMARY GENERATOR")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"Train file: {TRAIN_FILE}")
print(f"Validation file: {VALIDATION_FILE}")
print(f"Test file: {TEST_FILE}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VALIDATION_FILE),
        "test": str(TEST_FILE),
    }
)

print("\nDataset:")
print(dataset)

print(f"\nTrain examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")
print(f"Test examples: {len(dataset['test'])}")


# ============================================================
# 5. VERIFY TARGET FORMAT BEFORE TRAINING
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING TARGET FORMAT")
print("=" * 70)

sample_target = dataset["train"][0]["target"]

print("\nSample target:")
print("-" * 70)
print(sample_target)

expected_fields = [
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

target_lines = [
    line.strip()
    for line in sample_target.splitlines()
    if line.strip()
]

found_fields = []

for line in target_lines:
    if ":" in line:
        field = line.split(":", 1)[0].strip()
        found_fields.append(field)

missing_fields = [
    field for field in expected_fields
    if field not in found_fields
]

extra_fields = [
    field for field in found_fields
    if field not in expected_fields
]

if missing_fields:
    raise ValueError(
        f"\nERROR: Missing fields in target: {missing_fields}"
    )

if extra_fields:
    raise ValueError(
        f"\nERROR: Unexpected fields in target: {extra_fields}"
    )

if len(found_fields) != 15:
    raise ValueError(
        f"\nERROR: Expected exactly 15 fields, "
        f"but found {len(found_fields)}."
    )

print("\nTarget format check PASSED.")
print("Exactly 15 fields detected.")


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 7. LOAD MODEL
# ============================================================

print("Loading FLAN-T5 model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# ============================================================
# 8. TOKENIZATION
# ============================================================

def preprocess_function(examples):

    inputs = examples["input"]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# ============================================================
# 9. CHECK TOKENIZED LABELS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING TOKENIZED LABELS")
print("=" * 70)

sample_labels = tokenized_dataset["train"][0]["labels"]

print("Number of label tokens:", len(sample_labels))
print("First 30 label token IDs:", sample_labels[:30])

valid_label_tokens = [
    token
    for token in sample_labels
    if token != -100
]

print("Number of valid label tokens:", len(valid_label_tokens))

if len(valid_label_tokens) == 0:
    raise ValueError(
        "ERROR: All target labels became -100. "
        "The model has nothing to learn from."
    )

print("Label check PASSED.")
print("Tokenization complete.")


# ============================================================
# 10. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# 11. DEVICE INFORMATION
# ============================================================

if torch.cuda.is_available():

    device = torch.cuda.get_device_name(0)

    print(f"\nGPU detected: {device}")

else:

    print("\nNo GPU detected. Training will use CPU.")


# IMPORTANT:
# We previously had NaN/gradient instability with FP16.
# Keep FP16 disabled.

use_fp16 = False


# ============================================================
# 12. TRAINING ARGUMENTS
# ============================================================

training_kwargs = {

    "output_dir": str(OUTPUT_DIR),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    "num_train_epochs": 5,

    "learning_rate": 5e-5,

    "weight_decay": 0.01,

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    "per_device_train_batch_size": 4,

    "per_device_eval_batch_size": 4,

    "gradient_accumulation_steps": 4,

    # --------------------------------------------------------
    # Evaluation / saving
    # --------------------------------------------------------

    "save_strategy": "epoch",

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    "logging_strategy": "steps",

    "logging_steps": 25,

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    "load_best_model_at_end": True,

    "metric_for_best_model": "eval_loss",

    "greater_is_better": False,

    # --------------------------------------------------------
    # Generation during evaluation
    # --------------------------------------------------------

    "predict_with_generate": True,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    "seed": 42,

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    "fp16": use_fp16,

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    "save_total_limit": 2,

    # --------------------------------------------------------
    # No external reporting
    # --------------------------------------------------------

    "report_to": "none",
}


# ============================================================
# 13. TRANSFORMERS VERSION COMPATIBILITY
# ============================================================

signature = inspect.signature(
    Seq2SeqTrainingArguments
)

if "eval_strategy" in signature.parameters:

    training_kwargs["eval_strategy"] = "epoch"

else:

    training_kwargs["evaluation_strategy"] = "epoch"


training_args = Seq2SeqTrainingArguments(
    **training_kwargs
)


# ============================================================
# 14. TRAINER
# ============================================================

trainer_kwargs = {
    "model": model,

    "args": training_args,

    "train_dataset": tokenized_dataset["train"],

    "eval_dataset": tokenized_dataset["validation"],

    "data_collator": data_collator,
}


trainer_signature = inspect.signature(
    Seq2SeqTrainer
)

if "processing_class" in trainer_signature.parameters:

    trainer_kwargs["processing_class"] = tokenizer

else:

    trainer_kwargs["tokenizer"] = tokenizer


trainer = Seq2SeqTrainer(
    **trainer_kwargs
)


# ============================================================
# 15. START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING FLAN-T5 TRAINING")
print("=" * 70)

print("\nTraining configuration:")
print("Epochs: 5")
print("Learning rate: 5e-5")
print("Train batch size: 4")
print("Gradient accumulation: 4")
print("FP16: False")
print("Model: google/flan-t5-small")

print("\nThe model is now learning the labelled-field format.")


train_result = trainer.train()


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)

trainer.save_model(str(OUTPUT_DIR))

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

print(f"\nModel saved to:")
print(OUTPUT_DIR)


# ============================================================
# 17. VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

validation_results = trainer.evaluate(
    eval_dataset=tokenized_dataset["validation"]
)

for key, value in validation_results.items():

    if isinstance(value, float):

        print(f"{key}: {value:.4f}")

    else:

        print(f"{key}: {value}")


# ============================================================
# 18. TEST SET LOSS
# ============================================================

print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_results = trainer.evaluate(
    eval_dataset=tokenized_dataset["test"],
    metric_key_prefix="test",
)

for key, value in test_results.items():

    if isinstance(value, float):

        print(f"{key}: {value:.4f}")

    else:

        print(f"{key}: {value}")


# ============================================================
# 19. GENERATE ONE TEST EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE GENERATED CLINICAL SUMMARY")
print("=" * 70)

sample = dataset["test"][0]

input_text = sample["input"]

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_SOURCE_LENGTH,
)


# Move inputs to same device as model
model_device = next(
    model.parameters()
).device

inputs = {
    key: value.to(model_device)
    for key, value in inputs.items()
}


with torch.no_grad():

    generated_ids = model.generate(

        **inputs,

        max_new_tokens=MAX_TARGET_LENGTH,

        num_beams=4,

        do_sample=False,
    )


generated_text = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True,
)


print("\nGENERATED:")
print("-" * 70)
print(generated_text)

print("\nEXPECTED:")
print("-" * 70)
print(sample["target"])


# ============================================================
# 20. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nFinal model directory:")
print(OUTPUT_DIR)

print("\nNext step:")
print("Use the labelled-output inference/parser.")