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

DATA_DIR = Path("model_6(clinical summary generator)\\flan_t5_data")
OUTPUT_DIR = Path("flan_t5_summary_model")

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
            f"File not found: {file_path}\n"
            f"Make sure flan_t5_data is in the current directory."
        )

print("=" * 60)
print("MODEL 6 - CLINICAL SUMMARY GENERATOR")
print("=" * 60)

print(f"\nModel: {MODEL_NAME}")
print(f"Train file: {TRAIN_FILE}")
print(f"Validation file: {VALIDATION_FILE}")
print(f"Test file: {TEST_FILE}")


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
# 5. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 6. LOAD MODEL
# ============================================================

print("Loading FLAN-T5 model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# ============================================================
# 7. TOKENIZATION
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
# CHECK TOKENIZED LABELS
# ============================================================

print("\n" + "=" * 60)
print("CHECKING TOKENIZED LABELS")
print("=" * 60)

sample_labels = tokenized_dataset["train"][0]["labels"]

print("Number of label tokens:", len(sample_labels))
print("First 30 label token IDs:", sample_labels[:30])

valid_label_tokens = [
    token for token in sample_labels
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
# 8. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# 9. DEVICE INFORMATION
# ============================================================

if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    print(f"\nGPU detected: {device}")
else:
    print("\nNo GPU detected.")

# Disable FP16 for now.
# This avoids NaN/gradient instability while we verify training.
use_fp16 = False


# ============================================================
# 10. TRAINING ARGUMENTS
# ============================================================

training_kwargs = {
    "output_dir": str(OUTPUT_DIR),

    # Training
    "num_train_epochs": 5,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,

    # Batch size
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,

    # Evaluation / saving
    "save_strategy": "epoch",

    # Logging
    "logging_strategy": "steps",
    "logging_steps": 25,

    # Best model
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,

    # Generation
    "predict_with_generate": True,

    # Reproducibility
    "seed": 42,

    # Performance
    "fp16": use_fp16,

    # Keep only a few checkpoints
    "save_total_limit": 2,

    # Don't report to external services
    "report_to": "none",
}


# ============================================================
# 11. TRANSFORMERS VERSION COMPATIBILITY
# ============================================================

# Newer Transformers versions use eval_strategy.
# Older versions use evaluation_strategy.

signature = inspect.signature(Seq2SeqTrainingArguments)

if "eval_strategy" in signature.parameters:
    training_kwargs["eval_strategy"] = "epoch"
else:
    training_kwargs["evaluation_strategy"] = "epoch"


training_args = Seq2SeqTrainingArguments(
    **training_kwargs
)


# ============================================================
# 12. TRAINER
# ============================================================

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": tokenized_dataset["train"],
    "eval_dataset": tokenized_dataset["validation"],
    "data_collator": data_collator,
}

# Transformers compatibility:
# newer versions use processing_class,
# older versions use tokenizer.

trainer_signature = inspect.signature(Seq2SeqTrainer)

if "processing_class" in trainer_signature.parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer


trainer = Seq2SeqTrainer(
    **trainer_kwargs
)


# ============================================================
# 13. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING FLAN-T5 TRAINING")
print("=" * 60)

train_result = trainer.train()


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"\nModel saved to:")
print(OUTPUT_DIR)


# ============================================================
# 15. VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

validation_results = trainer.evaluate(
    eval_dataset=tokenized_dataset["validation"]
)

for key, value in validation_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# 16. TEST SET LOSS
# ============================================================

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

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
# 17. GENERATE ONE EXAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE GENERATED CLINICAL SUMMARY")
print("=" * 60)

sample = dataset["test"][0]

input_text = sample["input"]

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_SOURCE_LENGTH,
)

# Move inputs to same device as model
model_device = next(model.parameters()).device
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
print(generated_text)

print("\nEXPECTED:")
print(sample["target"])


# ============================================================
# 18. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nFinal model directory:")
print(OUTPUT_DIR)

print("\nYou can now use this trained model for Model 6.")