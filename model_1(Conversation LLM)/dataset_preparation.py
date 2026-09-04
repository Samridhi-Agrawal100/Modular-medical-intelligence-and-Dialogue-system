#Split → Verify → Save → Tokenization
import json
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

from collections import defaultdict
dataset_folder = Path(__file__).parent / "dataset"
DATA_PATH = dataset_folder / "model1_dummy.json"
# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset loaded successfully.")
print("Python type:", type(data))

# ============================================
# STEP 8A — SPLIT + VERIFY
# ============================================



print("\n" + "=" * 70)
print("STEP 8A — TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

# 80% Train, 10% Validation, 10% Test
train_data, temp_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42
)

# --------------------------------------------
# BASIC SIZE CHECK
# --------------------------------------------

print("\nRECORD COUNTS")
print("-" * 70)

print("Original dataset :", len(data))
print("Train            :", len(train_data))
print("Validation       :", len(val_data))
print("Test             :", len(test_data))

# --------------------------------------------
# ID OVERLAP CHECK
# --------------------------------------------

train_ids = {record["id"] for record in train_data}
val_ids = {record["id"] for record in val_data}
test_ids = {record["id"] for record in test_data}

print("\nID OVERLAP CHECK")
print("-" * 70)

print("Train ∩ Validation:", len(train_ids & val_ids))
print("Train ∩ Test      :", len(train_ids & test_ids))
print("Validation ∩ Test :", len(val_ids & test_ids))

# --------------------------------------------
# TOTAL UNIQUE ID CHECK
# --------------------------------------------

all_split_ids = train_ids | val_ids | test_ids

print("\nUNIQUE ID CHECK")
print("-" * 70)

print("Unique IDs in splits:", len(all_split_ids))
print("Original unique IDs :", len({record["id"] for record in data}))

# --------------------------------------------
# TARGET DISTRIBUTION
# --------------------------------------------

print("\nTARGET COUNTS")
print("-" * 70)

print("Train targets:", len(set(record["target"] for record in train_data)))
print("Validation targets:", len(set(record["target"] for record in val_data)))
print("Test targets:", len(set(record["target"] for record in test_data)))

# --------------------------------------------
# FINAL VERDICT
# --------------------------------------------

print("\n" + "=" * 70)

if (
    len(train_data) == 440
    and len(val_data) == 55
    and len(test_data) == 55
    and len(train_ids & val_ids) == 0
    and len(train_ids & test_ids) == 0
    and len(val_ids & test_ids) == 0
    and len(all_split_ids) == len(data)
):
    print("SPLIT VERIFICATION: PASSED")
else:
    print("SPLIT VERIFICATION: CHECK REQUIRED")

print("=" * 70)


# ============================================
# STEP 8B — SAVE DATASET SPLITS
# ============================================

import os
import json

SPLIT_DIR = "data/splits"

os.makedirs(SPLIT_DIR, exist_ok=True)

# Save training data
with open(
    os.path.join(SPLIT_DIR, "train.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

# Save validation data
with open(
    os.path.join(SPLIT_DIR, "validation.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

# Save test data
with open(
    os.path.join(SPLIT_DIR, "test.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)


print("\n" + "=" * 70)
print("STEP 8B — SPLITS SAVED")
print("=" * 70)

print("\nSaved files:")

print("Train      :", os.path.join(SPLIT_DIR, "train.json"))
print("Validation :", os.path.join(SPLIT_DIR, "validation.json"))
print("Test       :", os.path.join(SPLIT_DIR, "test.json"))

print("\n" + "=" * 70)


# ============================================
# STEP 8C — RELOAD & VERIFY SAVED SPLITS
# ============================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Reload the saved files
train_check = load_json(os.path.join(SPLIT_DIR, "train.json"))
val_check = load_json(os.path.join(SPLIT_DIR, "validation.json"))
test_check = load_json(os.path.join(SPLIT_DIR, "test.json"))


# Get IDs
train_ids = {record["id"] for record in train_check}
val_ids = {record["id"] for record in val_check}
test_ids = {record["id"] for record in test_check}

original_ids = {record["id"] for record in data}


print("\n" + "=" * 70)
print("STEP 8C — RELOAD & VERIFY")
print("=" * 70)

print("\nRELOADED RECORD COUNTS")
print("-" * 70)

print("Train      :", len(train_check))
print("Validation :", len(val_check))
print("Test       :", len(test_check))

print("\nID OVERLAP CHECK")
print("-" * 70)

print("Train ∩ Validation:", len(train_ids & val_ids))
print("Train ∩ Test      :", len(train_ids & test_ids))
print("Validation ∩ Test :", len(val_ids & test_ids))

print("\nORIGINAL ID COVERAGE")
print("-" * 70)

combined_ids = train_ids | val_ids | test_ids

print("Combined split IDs :", len(combined_ids))
print("Original IDs       :", len(original_ids))

print("\nFINAL VERIFICATION")
print("-" * 70)

if (
    len(train_check) == 440
    and len(val_check) == 55
    and len(test_check) == 55
    and len(train_ids & val_ids) == 0
    and len(train_ids & test_ids) == 0
    and len(val_ids & test_ids) == 0
    and combined_ids == original_ids
):
    print("RELOAD VERIFICATION: PASSED")
else:
    print("RELOAD VERIFICATION: FAILED")

print("\n" + "=" * 70)





# ============================================
# TOKENIZATION we will laod model fromhugging face
# ============================================
# ============================================
# STEP 8D-3 — LOAD LLAMA 3.1 TOKENIZER
# ============================================

from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("\n" + "=" * 70)
print("STEP 8D-3 — LOADING LLAMA 3.1 TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("\nTokenizer loaded successfully.")

print("Tokenizer type:", type(tokenizer))
print("Vocabulary size:", tokenizer.vocab_size)

print("\nSpecial tokens:")
print("BOS:", tokenizer.bos_token)
print("EOS:", tokenizer.eos_token)
print("PAD:", tokenizer.pad_token)

print("\n" + "=" * 70)
print("STEP 8D-3 COMPLETE")
print("=" * 70)


# ============================================
# STEP 8D-4 — TEST LLAMA CHAT TEMPLATE
# ============================================

print("\n" + "=" * 70)
print("STEP 8D-4 — TESTING LLAMA CHAT TEMPLATE")
print("=" * 70)

sample_record = data[0]

formatted_text = tokenizer.apply_chat_template(
    sample_record["messages"],
    tokenize=False,
    add_generation_prompt=False
)

print("\nORIGINAL MESSAGES:")
print("-" * 70)

for message in sample_record["messages"]:
    print(f"[{message['role'].upper()}] {message['content']}")

print("\nLLAMA FORMATTED TEXT:")
print("-" * 70)
print(formatted_text)

print("\nCHAT TEMPLATE TEST: PASSED")


# ============================================
# STEP 8D-5 — TEST TOKENIZATION
# ============================================

print("\n" + "=" * 70)
print("STEP 8D-5 — TESTING LLAMA TOKENIZATION")
print("=" * 70)

sample_record = data[0]

tokenized_sample = tokenizer.apply_chat_template(
    sample_record["messages"],
    tokenize=True,
    add_generation_prompt=False,
    return_tensors=None
)

print("\nRECORD ID:", sample_record["id"])
print("TARGET:", sample_record["target"])

print("\nTOKENIZATION RESULTS:")
print("-" * 70)
print("Token count:", len(tokenized_sample))
print("First 20 token IDs:", tokenized_sample[:20])
print("Last 20 token IDs:", tokenized_sample[-20:])

print("\nTOKENIZATION TEST: PASSED")


# ============================================
# STEP 8D-5 — TOKENIZATION + LENGTH ANALYSIS
# ============================================

print("\n" + "=" * 70)
print("STEP 8D-5 — TOKENIZATION + LENGTH ANALYSIS")
print("=" * 70)

# ------------------------------------------------
# PART A — VERIFY ACTUAL TOKENIZATION
# ------------------------------------------------

sample_record = data[0]

sample_encoding = tokenizer.apply_chat_template(
    sample_record["messages"],
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True
)

sample_input_ids = sample_encoding["input_ids"]

print("\nSAMPLE TOKENIZATION")
print("-" * 70)
print("Record ID:", sample_record["id"])
print("Target:", sample_record["target"])
print("Input IDs type:", type(sample_input_ids))
print("Token count:", len(sample_input_ids))
print("First 20 token IDs:", sample_input_ids[:20])
print("Last 20 token IDs:", sample_input_ids[-20:])

print("\nAttention mask length:", len(sample_encoding["attention_mask"]))

# ------------------------------------------------
# PART B — TOKENIZE ALL RECORDS FOR LENGTH ONLY
# ------------------------------------------------

print("\n" + "=" * 70)
print("TOKEN LENGTH ANALYSIS — ALL RECORDS")
print("=" * 70)

token_lengths = []

for record in data:
    encoding = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True
    )

    length = len(encoding["input_ids"])
    token_lengths.append((record["id"], length))

# Extract only lengths
lengths = [length for _, length in token_lengths]

# ------------------------------------------------
# PART C — BASIC STATISTICS
# ------------------------------------------------

print("\nTOTAL RECORDS:", len(lengths))
print("MIN TOKENS:", min(lengths))
print("MAX TOKENS:", max(lengths))
print("AVERAGE TOKENS:", round(sum(lengths) / len(lengths), 2))

sorted_lengths = sorted(lengths)

def percentile(values, p):
    index = int((len(values) - 1) * p)
    return values[index]

print("MEDIAN TOKENS:", percentile(sorted_lengths, 0.50))
print("90th PERCENTILE:", percentile(sorted_lengths, 0.90))
print("95th PERCENTILE:", percentile(sorted_lengths, 0.95))
print("99th PERCENTILE:", percentile(sorted_lengths, 0.99))

# ------------------------------------------------
# PART D — CHECK COMMON MAX_LENGTH VALUES
# ------------------------------------------------

print("\n" + "=" * 70)
print("MAX LENGTH COVERAGE")
print("=" * 70)

for max_length in [256, 512, 1024, 2048, 4096, 8192]:
    truncated = sum(1 for length in lengths if length > max_length)
    percentage = (truncated / len(lengths)) * 100

    print(
        f"max_length={max_length:4d} | "
        f"records over limit: {truncated:3d} | "
        f"percentage: {percentage:.2f}%"
    )

# ------------------------------------------------
# PART E — SHOW LONGEST RECORDS
# ------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 LONGEST RECORDS")
print("=" * 70)

longest_records = sorted(
    token_lengths,
    key=lambda x: x[1],
    reverse=True
)[:20]

for rank, (record_id, length) in enumerate(longest_records, start=1):
    print(f"{rank:2d}. {record_id} -> {length} tokens")

# ------------------------------------------------
# FINAL
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8D-5 COMPLETE")
print("=" * 70)

# ============================================
# STEP 8D-6 — PATIENT ANSWER → NEXT QUESTION
# BEHAVIORAL VALIDATION
# ============================================

print("\n" + "=" * 70)
print("STEP 8D-6 — PATIENT ANSWER → NEXT QUESTION VALIDATION")
print("=" * 70)

# ------------------------------------------------
# PART A — BASIC CONVERSATION CHECK
# ------------------------------------------------

valid_records = []
invalid_records = []

for record in data:

    messages = record["messages"]

    # Need at least:
    # previous user answer + final assistant question
    if len(messages) < 2:
        invalid_records.append(record["id"])
        continue

    # Final message must be assistant
    if messages[-1]["role"] != "assistant":
        invalid_records.append(record["id"])
        continue

    # Message immediately before final assistant question
    previous_message = messages[-2]

    if previous_message["role"] != "user":
        invalid_records.append(record["id"])
        continue

    valid_records.append(record)


print("\nTOTAL RECORDS:", len(data))
print("VALID PATIENT → ASSISTANT PAIRS:", len(valid_records))
print("INVALID RECORDS:", len(invalid_records))


# ------------------------------------------------
# PART B — CHECK TARGET + FINAL QUESTION
# ------------------------------------------------

print("\n" + "=" * 70)
print("TARGET / FINAL QUESTION CHECK")
print("=" * 70)

missing_targets = []
empty_questions = []

for record in valid_records:

    target = record["target"]
    final_question = record["messages"][-1]["content"].strip()

    if not target:
        missing_targets.append(record["id"])

    if not final_question:
        empty_questions.append(record["id"])


print("Missing targets:", len(missing_targets))
print("Empty final questions:", len(empty_questions))


# ------------------------------------------------
# PART C — SHOW REPRESENTATIVE EXAMPLES
# ------------------------------------------------

print("\n" + "=" * 70)
print("REPRESENTATIVE PATIENT → NEXT QUESTION EXAMPLES")
print("=" * 70)

# Use evenly distributed records rather than only first records
sample_count = min(20, len(valid_records))

if sample_count > 0:

    step = max(1, len(valid_records) // sample_count)

    selected_records = valid_records[::step][:sample_count]

    for number, record in enumerate(selected_records, start=1):

        messages = record["messages"]

        patient_answer = messages[-2]["content"]
        next_question = messages[-1]["content"]

        print("\n" + "-" * 70)
        print(f"EXAMPLE {number}/{sample_count}")
        print("-" * 70)

        print("ID:", record["id"])
        print("TARGET:", record["target"])

        print("\nPATIENT'S LATEST ANSWER:")
        print(patient_answer)

        print("\nASSISTANT'S NEXT QUESTION:")
        print(next_question)


# ------------------------------------------------
# PART D — CHECK FOR IMMEDIATE REPETITION
# ------------------------------------------------

print("\n" + "=" * 70)
print("IMMEDIATE QUESTION REPETITION CHECK")
print("=" * 70)

repeated_questions = []

for record in valid_records:

    messages = record["messages"]

    if len(messages) < 4:
        continue

    final_question = messages[-1]["content"].strip().lower()

    previous_assistant_messages = [
        message["content"].strip().lower()
        for message in messages[:-1]
        if message["role"] == "assistant"
    ]

    if final_question in previous_assistant_messages:
        repeated_questions.append(
            (record["id"], final_question)
        )


print("Exact repeated assistant questions:",
      len(repeated_questions))

if repeated_questions:

    print("\nExamples of repetitions:")

    for record_id, question in repeated_questions[:10]:
        print(f"\n{record_id}:")
        print(question)


# ------------------------------------------------
# PART E — TARGET DISTRIBUTION
# ------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

from collections import Counter

target_counts = Counter(
    record["target"]
    for record in valid_records
)

for target, count in target_counts.most_common():
    print(f"{target:<35} {count}")


# ------------------------------------------------
# FINAL
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8D-6 COMPLETE")
print("=" * 70)



# ============================================
# STEP 8D-7 — MODEL-READY CHAT FORMAT CHECK
# ============================================

from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("\n" + "=" * 70)
print("STEP 8D-7 — MODEL-READY CHAT FORMAT CHECK")
print("=" * 70)

# ------------------------------------------------
# LOAD TOKENIZER
# ------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer loaded:", type(tokenizer).__name__)
print("Chat template available:", tokenizer.chat_template is not None)


# ------------------------------------------------
# SELECT SAMPLE RECORD
# ------------------------------------------------

sample_record = data[0]

print("\n" + "=" * 70)
print("SAMPLE RECORD")
print("=" * 70)

print("ID:", sample_record["id"])
print("TARGET:", sample_record["target"])
print("RED FLAG:", sample_record["safety"]["red_flag"])


# ------------------------------------------------
# ORIGINAL MESSAGES
# ------------------------------------------------

print("\n" + "=" * 70)
print("ORIGINAL MESSAGES")
print("=" * 70)

for i, message in enumerate(sample_record["messages"], start=1):
    print(f"\n{i}. [{message['role'].upper()}]")
    print(message["content"])


# ------------------------------------------------
# APPLY LLAMA CHAT TEMPLATE
# ------------------------------------------------

formatted_text = tokenizer.apply_chat_template(
    sample_record["messages"],
    tokenize=False,
    add_generation_prompt=False
)

print("\n" + "=" * 70)
print("LLAMA CHAT-TEMPLATE OUTPUT")
print("=" * 70)

print(formatted_text)


# ------------------------------------------------
# VERIFY FINAL ASSISTANT RESPONSE IS PRESENT
# ------------------------------------------------

final_assistant_text = sample_record["messages"][-1]["content"]

print("\n" + "=" * 70)
print("FINAL ASSISTANT QUESTION CHECK")
print("=" * 70)

print("Expected final assistant question:")
print(final_assistant_text)

print(
    "\nFinal question present in formatted conversation:",
    final_assistant_text in formatted_text
)


# ------------------------------------------------
# TOKENIZE FORMATTED CONVERSATION
# ------------------------------------------------

tokenized = tokenizer(
    formatted_text,
    add_special_tokens=False
)

input_ids = tokenized["input_ids"]

print("\n" + "=" * 70)
print("FINAL TOKENIZATION CHECK")
print("=" * 70)

print("Token count:", len(input_ids))
print("First 20 tokens:", input_ids[:20])
print("Last 20 tokens:", input_ids[-20:])


# ------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8D-7 COMPLETE")
print("=" * 70)



# ============================================
# STEP 8D-8 — ASSISTANT-ONLY LOSS MASK CHECK
# ============================================

print("\n" + "=" * 70)
print("STEP 8D-8 — ASSISTANT-ONLY LOSS MASK CHECK")
print("=" * 70)

# ------------------------------------------------
# SAMPLE RECORD
# ------------------------------------------------

sample_record = data[0]

messages = sample_record["messages"]

print("\nRecord ID:", sample_record["id"])
print("Number of messages:", len(messages))


# ------------------------------------------------
# SHOW WHICH MESSAGES ARE TRAINING TARGETS
# ------------------------------------------------

print("\n" + "=" * 70)
print("MESSAGE LOSS ASSIGNMENT")
print("=" * 70)

for i, message in enumerate(messages, start=1):

    role = message["role"]

    if role == "assistant":
        loss_status = "TRAIN → LOSS COMPUTED"
    else:
        loss_status = "CONTEXT → NO LOSS"

    print(f"\n{i}. [{role.upper()}]")
    print(loss_status)
    print(message["content"])


# ------------------------------------------------
# COUNT MESSAGES
# ------------------------------------------------

assistant_count = sum(
    1 for message in messages
    if message["role"] == "assistant"
)

user_count = sum(
    1 for message in messages
    if message["role"] == "user"
)

print("\n" + "=" * 70)
print("MESSAGE COUNTS")
print("=" * 70)

print("Assistant messages:", assistant_count)
print("User messages:", user_count)


# ------------------------------------------------
# FINAL ASSISTANT QUESTION
# ------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TRAINING TARGET")
print("=" * 70)

print("The final assistant response that the model should learn:")
print(messages[-1]["content"])


# ------------------------------------------------
# FINAL CHECK
# ------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8D-8 COMPLETE")
print("=" * 70)



# ============================================
# STEP 8D-9 — TOKEN-LEVEL ASSISTANT LOSS MASK
# ============================================

from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("\n" + "=" * 70)
print("STEP 8D-9 — TOKEN-LEVEL ASSISTANT LOSS MASK")
print("=" * 70)


# ------------------------------------------------
# LOAD TOKENIZER
# ------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer:", type(tokenizer).__name__)
print("Chat template available:", tokenizer.chat_template is not None)


# ------------------------------------------------
# SAMPLE RECORD
# ------------------------------------------------

record = data[0]
messages = record["messages"]

print("\nRecord ID:", record["id"])
print("Number of messages:", len(messages))


# ------------------------------------------------
# TOKENIZE FULL CONVERSATION
# ------------------------------------------------
#
# We deliberately keep the complete conversation.
# The model needs the entire history as context.
#
# return_assistant_tokens_mask=True asks the
# tokenizer/chat template to identify assistant
# response tokens.
# ------------------------------------------------

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=False,
    return_assistant_tokens_mask=True
)


input_ids = encoded["input_ids"]
assistant_mask = encoded.get("assistant_masks")


# ------------------------------------------------
# VERIFY MASK EXISTS
# ------------------------------------------------

print("\n" + "=" * 70)
print("ASSISTANT MASK CHECK")
print("=" * 70)

print("Input token count:", len(input_ids))
print("Assistant mask exists:", assistant_mask is not None)

if assistant_mask is None:
    print("\nWARNING:")
    print("The tokenizer did not return assistant token masks.")
    print("We will NOT proceed to training until this is handled correctly.")


# ------------------------------------------------
# IF MASK EXISTS, INSPECT IT
# ------------------------------------------------

if assistant_mask is not None:

    print("Assistant mask length:", len(assistant_mask))

    print(
        "Mask length matches input:",
        len(input_ids) == len(assistant_mask)
    )

    # Count tokens belonging to assistant responses
    assistant_token_count = sum(assistant_mask)

    context_token_count = len(assistant_mask) - assistant_token_count

    print("\nAssistant tokens:", assistant_token_count)
    print("Context/non-assistant tokens:", context_token_count)


    # ------------------------------------------------
    # BUILD LABELS
    # ------------------------------------------------
    #
    # -100 = ignored by cross-entropy loss
    # actual token ID = contributes to loss
    # ------------------------------------------------

    labels = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(input_ids, assistant_mask)
    ]


    # ------------------------------------------------
    # VERIFY LABEL COUNTS
    # ------------------------------------------------

    trainable_label_count = sum(
        1 for label in labels
        if label != -100
    )

    ignored_label_count = sum(
        1 for label in labels
        if label == -100
    )

    print("\n" + "=" * 70)
    print("LABEL CHECK")
    print("=" * 70)

    print("Total labels:", len(labels))
    print("Trainable labels:", trainable_label_count)
    print("Ignored labels (-100):", ignored_label_count)

    print(
        "Label count matches token count:",
        len(labels) == len(input_ids)
    )


    # ------------------------------------------------
    # SHOW TOKEN / MASK / LABEL INFORMATION
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKEN-LEVEL INSPECTION")
    print("=" * 70)

    print("\nFirst 80 tokens:")
    print("-" * 70)

    for i in range(min(80, len(input_ids))):

        token_id = input_ids[i]
        mask = assistant_mask[i]
        label = labels[i]

        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False
        )

        print(
            f"{i:4d} | "
            f"token_id={token_id:<8} | "
            f"assistant={mask} | "
            f"label={label:<8} | "
            f"{repr(token_text)}"
        )


    # ------------------------------------------------
    # VERIFY ASSISTANT RESPONSE TEXT
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("ASSISTANT-TOKEN RECONSTRUCTION")
    print("=" * 70)

    assistant_token_ids = [
        token_id
        for token_id, mask in zip(input_ids, assistant_mask)
        if mask == 1
    ]

    reconstructed_assistant = tokenizer.decode(
        assistant_token_ids,
        skip_special_tokens=False
    )

    print("\nReconstructed assistant-token text:")
    print(reconstructed_assistant)

    print("\nExpected assistant messages:")

    for i, message in enumerate(messages):

        if message["role"] == "assistant":

            print("\n--- Assistant message", i + 1, "---")
            print(message["content"])


    # ------------------------------------------------
    # FINAL QUESTION TOKEN CHECK
    # ------------------------------------------------

    final_question = messages[-1]["content"]

    print("\n" + "=" * 70)
    print("FINAL ASSISTANT QUESTION CHECK")
    print("=" * 70)

    print("Expected final question:")
    print(final_question)

    print(
        "\nFinal question appears in assistant-token text:",
        final_question in reconstructed_assistant
    )


print("\n" + "=" * 70)
print("STEP 8D-9 COMPLETE")
print("=" * 70)


# ============================================
# STEP 8D-11 — MANUAL ASSISTANT TOKEN MASK
# ============================================

from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("\n" + "=" * 70)
print("STEP 8D-11 — MANUAL ASSISTANT TOKEN MASK")
print("=" * 70)


# ------------------------------------------------
# LOAD TOKENIZER
# ------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer:", type(tokenizer).__name__)


# ------------------------------------------------
# SAMPLE RECORD
# ------------------------------------------------

record = data[0]
messages = record["messages"]

print("Record ID:", record["id"])
print("Number of messages:", len(messages))


# ------------------------------------------------
# BUILD FULL CHAT
# ------------------------------------------------

full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

full_ids = tokenizer(
    full_text,
    add_special_tokens=False
)["input_ids"]

print("\nFull token count:", len(full_ids))


# ------------------------------------------------
# TOKEN IDs WE NEED
# ------------------------------------------------

assistant_header_ids = tokenizer.encode(
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    add_special_tokens=False
)

eot_id = tokenizer.eos_token_id

print("\nAssistant header token IDs:")
print(assistant_header_ids)

print("EOT token ID:", eot_id)


# ------------------------------------------------
# FIND ASSISTANT SECTIONS
# ------------------------------------------------

assistant_mask = [0] * len(full_ids)

assistant_spans = []


def find_subsequence(sequence, pattern, start=0):
    """
    Find the first occurrence of pattern in sequence
    starting from index 'start'.
    """
    pattern_length = len(pattern)

    for i in range(start, len(sequence) - pattern_length + 1):

        if sequence[i:i + pattern_length] == pattern:
            return i

    return -1


search_position = 0


print("\n" + "=" * 70)
print("ASSISTANT SPAN DETECTION")
print("=" * 70)


for message_index, message in enumerate(messages):

    if message["role"] != "assistant":
        continue

    # --------------------------------------------
    # Find assistant header
    # --------------------------------------------

    header_start = find_subsequence(
        full_ids,
        assistant_header_ids,
        search_position
    )

    if header_start == -1:

        print(
            f"\nMessage {message_index + 1}: "
            "FAIL — assistant header not found"
        )

        continue

    content_start = header_start + len(assistant_header_ids)


    # --------------------------------------------
    # Find EOT after assistant content
    # --------------------------------------------

    eot_position = -1

    for position in range(content_start, len(full_ids)):

        if full_ids[position] == eot_id:

            eot_position = position
            break


    if eot_position == -1:

        print(
            f"\nMessage {message_index + 1}: "
            "FAIL — EOT token not found"
        )

        continue


    # --------------------------------------------
    # IMPORTANT:
    #
    # We train ONLY on assistant CONTENT.
    #
    # We do NOT train on:
    #
    # <|start_header_id|>
    # assistant
    # <|end_header_id|>
    # newlines
    # <|eot_id|>
    # --------------------------------------------

    span_start = content_start
    span_end = eot_position

    assistant_spans.append(
        (span_start, span_end)
    )


    # Mark assistant content tokens

    for position in range(span_start, span_end):

        assistant_mask[position] = 1


    print(
        f"\nAssistant message {message_index + 1}:"
    )

    print(
        f"Header starts at token: {header_start}"
    )

    print(
        f"Content tokens: [{span_start}:{span_end}]"
    )

    print(
        f"Content token count: "
        f"{span_end - span_start}"
    )


    # Move search forward

    search_position = eot_position + 1


# ------------------------------------------------
# MASK SUMMARY
# ------------------------------------------------

print("\n" + "=" * 70)
print("MASK SUMMARY")
print("=" * 70)

assistant_tokens = sum(assistant_mask)

context_tokens = (
    len(assistant_mask) - assistant_tokens
)

print("Total tokens:", len(full_ids))
print("Assistant content tokens:", assistant_tokens)
print("Context tokens:", context_tokens)

print(
    "Mask length matches:",
    len(assistant_mask) == len(full_ids)
)


# ------------------------------------------------
# BUILD LABELS
# ------------------------------------------------

labels = [
    token_id if mask == 1 else -100
    for token_id, mask in zip(full_ids, assistant_mask)
]


# ------------------------------------------------
# LABEL CHECK
# ------------------------------------------------

trainable_labels = sum(
    1 for label in labels
    if label != -100
)

ignored_labels = sum(
    1 for label in labels
    if label == -100
)

print("\n" + "=" * 70)
print("LABEL CHECK")
print("=" * 70)

print("Total labels:", len(labels))
print("Trainable labels:", trainable_labels)
print("Ignored labels:", ignored_labels)

print(
    "Trainable == assistant tokens:",
    trainable_labels == assistant_tokens
)


# ------------------------------------------------
# RECONSTRUCT ASSISTANT CONTENT
# ------------------------------------------------

assistant_only_ids = [
    token_id
    for token_id, mask in zip(
        full_ids,
        assistant_mask
    )
    if mask == 1
]

assistant_text = tokenizer.decode(
    assistant_only_ids,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)

print("\n" + "=" * 70)
print("RECONSTRUCTED ASSISTANT CONTENT")
print("=" * 70)

print(assistant_text)


# ------------------------------------------------
# EXPECTED ASSISTANT CONTENT
# ------------------------------------------------

expected_assistant_text = "\n".join(
    message["content"]
    for message in messages
    if message["role"] == "assistant"
)

print("\n" + "=" * 70)
print("EXPECTED ASSISTANT CONTENT")
print("=" * 70)

print(expected_assistant_text)


# ------------------------------------------------
# FINAL QUESTION CHECK
# ------------------------------------------------

final_question = messages[-1]["content"]

print("\n" + "=" * 70)
print("FINAL QUESTION CHECK")
print("=" * 70)

print("Expected final question:")
print(final_question)

print(
    "\nFinal question found:",
    final_question in assistant_text
)


# ------------------------------------------------
# SANITY CHECK
# ------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SANITY CHECK")
print("=" * 70)


if len(assistant_spans) != sum(
    1 for message in messages
    if message["role"] == "assistant"
):

    print(
        "FAIL — NOT ALL ASSISTANT MESSAGES WERE FOUND"
    )

elif assistant_tokens == 0:

    print(
        "FAIL — ZERO ASSISTANT CONTENT TOKENS"
    )

elif trainable_labels != assistant_tokens:

    print(
        "FAIL — LABEL/MASK COUNT MISMATCH"
    )

elif final_question not in assistant_text:

    print(
        "FAIL — FINAL ASSISTANT QUESTION NOT RECOVERED"
    )

else:

    print(
        "PASS — ASSISTANT CONTENT MASK SUCCESSFULLY CREATED"
    )


print("\n" + "=" * 70)
print("STEP 8D-11 COMPLETE")
print("=" * 70)



# ============================================================
# STEP 8D-12 — BUILD MODEL-READY TOKENIZED DATASET
# ============================================================

import json
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

MAX_LENGTH = 2048

SPLIT_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/tokenized")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "=" * 70)
print("STEP 8D-12 — BUILD MODEL-READY TOKENIZED DATASET")
print("=" * 70)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer:", type(tokenizer).__name__)
print("Vocabulary size:", tokenizer.vocab_size)
print("Max sequence length:", MAX_LENGTH)


# ============================================================
# TOKEN IDs USED FOR ASSISTANT MASKING
# ============================================================

assistant_header_ids = tokenizer.encode(
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    add_special_tokens=False
)

eot_id = tokenizer.eos_token_id

print("\nAssistant header IDs:", assistant_header_ids)
print("EOT token ID:", eot_id)


# ============================================================
# HELPER: FIND SUBSEQUENCE
# ============================================================

def find_subsequence(sequence, pattern, start=0):

    pattern_length = len(pattern)

    for i in range(
        start,
        len(sequence) - pattern_length + 1
    ):

        if sequence[i:i + pattern_length] == pattern:

            return i

    return -1


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(record):

    record_id = record["id"]
    messages = record["messages"]

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not isinstance(messages, list) or len(messages) == 0:

        raise ValueError(
            f"{record_id}: invalid messages"
        )

    if messages[-1]["role"] != "assistant":

        raise ValueError(
            f"{record_id}: final message is not assistant"
        )


    # --------------------------------------------------------
    # Apply official Llama chat template
    # --------------------------------------------------------

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    tokenized = tokenizer(
        formatted_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]


    # --------------------------------------------------------
    # IMPORTANT
    #
    # We need the ORIGINAL untruncated sequence to determine
    # assistant spans correctly.
    # --------------------------------------------------------

    full_ids = tokenizer(
        formatted_text,
        add_special_tokens=False
    )["input_ids"]


    # --------------------------------------------------------
    # Create assistant mask
    # --------------------------------------------------------

    assistant_mask = [0] * len(full_ids)

    search_position = 0

    assistant_message_count = 0
    detected_assistant_count = 0


    for message in messages:

        if message["role"] != "assistant":
            continue

        assistant_message_count += 1


        # Find assistant header

        header_start = find_subsequence(
            full_ids,
            assistant_header_ids,
            search_position
        )

        if header_start == -1:

            raise ValueError(
                f"{record_id}: assistant header not found"
            )


        content_start = (
            header_start +
            len(assistant_header_ids)
        )


        # Find EOT

        eot_position = -1

        for position in range(
            content_start,
            len(full_ids)
        ):

            if full_ids[position] == eot_id:

                eot_position = position
                break


        if eot_position == -1:

            raise ValueError(
                f"{record_id}: assistant EOT not found"
            )


        # Mark ONLY assistant content

        for position in range(
            content_start,
            eot_position
        ):

            assistant_mask[position] = 1


        detected_assistant_count += 1

        search_position = eot_position + 1


    # --------------------------------------------------------
    # Verify assistant message detection
    # --------------------------------------------------------

    if detected_assistant_count != assistant_message_count:

        raise ValueError(
            f"{record_id}: assistant message count mismatch"
        )


    # --------------------------------------------------------
    # Apply truncation to assistant mask
    # --------------------------------------------------------

    assistant_mask = assistant_mask[:MAX_LENGTH]


    # --------------------------------------------------------
    # Create labels
    #
    # Assistant content → train
    # Everything else   → ignore
    # --------------------------------------------------------

    labels = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(
            input_ids,
            assistant_mask
        )
    ]


    # --------------------------------------------------------
    # CRITICAL SAFETY CHECK
    # --------------------------------------------------------

    if len(input_ids) != len(labels):

        raise ValueError(
            f"{record_id}: input/label length mismatch"
        )


    trainable_tokens = sum(
        1
        for label in labels
        if label != -100
    )


    if trainable_tokens == 0:

        raise ValueError(
            f"{record_id}: ZERO trainable assistant tokens"
        )


    # --------------------------------------------------------
    # Check whether final assistant question survived
    # --------------------------------------------------------

    final_question = messages[-1]["content"]

    final_question_ids = tokenizer(
        final_question,
        add_special_tokens=False
    )["input_ids"]


    # Determine whether final question is fully contained
    # in the retained assistant-token region.

    final_question_present = False

    if len(final_question_ids) > 0:

        for start in range(
            len(input_ids) - len(final_question_ids) + 1
        ):

            candidate = input_ids[
                start:
                start + len(final_question_ids)
            ]

            if candidate == final_question_ids:

                # Ensure those tokens are trainable

                if all(
                    labels[start + j] != -100
                    for j in range(len(final_question_ids))
                ):

                    final_question_present = True
                    break


    # --------------------------------------------------------
    # Return model-ready record
    # --------------------------------------------------------

    return {
        "id": record_id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,

        # Metadata retained for evaluation/debugging.
        # These are NOT model inputs.
        "target": record["target"],
        "red_flag": record["safety"]["red_flag"],

        "original_token_count": len(full_ids),
        "final_token_count": len(input_ids),
        "truncated": len(full_ids) > MAX_LENGTH,
        "trainable_tokens": trainable_tokens,
        "final_question_present": final_question_present
    }


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    input_file = SPLIT_DIR / f"{split_name}.json"

    output_file = (
        OUTPUT_DIR /
        f"{split_name}_tokenized.json"
    )


    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print("=" * 70)


    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)


    print("Input records:", len(records))


    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    processed = []

    errors = []


    for index, record in enumerate(records):

        try:

            processed_record = process_record(record)

            processed.append(processed_record)

        except Exception as e:

            errors.append(
                {
                    "index": index,
                    "id": record.get("id", "UNKNOWN"),
                    "error": str(e)
                }
            )


    # --------------------------------------------------------
    # DO NOT silently continue with broken records
    # --------------------------------------------------------

    if errors:

        print("\nERRORS FOUND:")

        for error in errors[:20]:

            print(
                f"  {error['id']}: "
                f"{error['error']}"
            )


        if len(errors) > 20:

            print(
                f"  ... and "
                f"{len(errors) - 20} more"
            )


        raise RuntimeError(
            f"{split_name}: "
            f"{len(errors)} records failed preprocessing."
        )


    # --------------------------------------------------------
    # Dataset-level statistics
    # --------------------------------------------------------

    token_counts = [
        item["final_token_count"]
        for item in processed
    ]

    trainable_counts = [
        item["trainable_tokens"]
        for item in processed
    ]

    truncated_count = sum(
        item["truncated"]
        for item in processed
    )

    final_question_failures = sum(
        not item["final_question_present"]
        for item in processed
    )


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(processed) != len(records):

        raise RuntimeError(
            f"{split_name}: record count changed"
        )


    if final_question_failures > 0:

        raise RuntimeError(
            f"{split_name}: "
            f"{final_question_failures} final questions "
            f"were not preserved."
        )


    if min(trainable_counts) <= 0:

        raise RuntimeError(
            f"{split_name}: zero trainable tokens detected."
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            processed,
            f,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nProcessed records:", len(processed))
    print("Minimum tokens:", min(token_counts))
    print("Maximum tokens:", max(token_counts))
    print(
        "Average tokens:",
        round(
            sum(token_counts) / len(token_counts),
            2
        )
    )

    print(
        "Minimum trainable tokens:",
        min(trainable_counts)
    )

    print(
        "Maximum trainable tokens:",
        max(trainable_counts)
    )

    print(
        "Truncated records:",
        truncated_count
    )

    print(
        "Final-question preservation failures:",
        final_question_failures
    )

    print("\nSaved:")
    print(output_file)


    return processed


# ============================================================
# PROCESS ALL THREE SPLITS
# ============================================================

train_processed = process_split("train")
validation_processed = process_split("validation")
test_processed = process_split("test")


# ============================================================
# FINAL DATASET CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 8D-12 — FINAL PREPROCESSING VERIFICATION")
print("=" * 70)

print("\nTrain:", len(train_processed))
print("Validation:", len(validation_processed))
print("Test:", len(test_processed))

print(
    "\nTotal:",
    len(train_processed)
    + len(validation_processed)
    + len(test_processed)
)


# ------------------------------------------------------------
# Verify IDs against original split structure
# ------------------------------------------------------------

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


original_train = load_json(
    SPLIT_DIR / "train.json"
)

original_validation = load_json(
    SPLIT_DIR / "validation.json"
)

original_test = load_json(
    SPLIT_DIR / "test.json"
)


for name, original, processed in [

    ("train", original_train, train_processed),

    ("validation", original_validation, validation_processed),

    ("test", original_test, test_processed)

]:

    original_ids = {
        item["id"]
        for item in original
    }

    processed_ids = {
        item["id"]
        for item in processed
    }


    if original_ids != processed_ids:

        raise RuntimeError(
            f"{name}: ID mismatch after tokenization"
        )


print("\nID coverage: PASSED")


# ------------------------------------------------------------
# Verify split separation
# ------------------------------------------------------------

train_ids = {
    item["id"]
    for item in train_processed
}

validation_ids = {
    item["id"]
    for item in validation_processed
}

test_ids = {
    item["id"]
    for item in test_processed
}


print(
    "Train ∩ Validation:",
    len(train_ids & validation_ids)
)

print(
    "Train ∩ Test:",
    len(train_ids & test_ids)
)

print(
    "Validation ∩ Test:",
    len(validation_ids & test_ids)
)


if (
    train_ids & validation_ids
    or train_ids & test_ids
    or validation_ids & test_ids
):

    raise RuntimeError(
        "DATA LEAKAGE DETECTED BETWEEN SPLITS"
    )


print("\nSplit separation: PASSED")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 8D-12 COMPLETE")
print("=" * 70)

print("\nMODEL-READY DATASET: SUCCESS")