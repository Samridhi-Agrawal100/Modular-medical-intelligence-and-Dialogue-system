# ============================================================
# MODEL 6 — CLINICAL SUMMARY GENERATOR
# FINAL UNSEEN INFERENCE + ROBUST EVALUATION
# ============================================================

from pathlib import Path
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "csg_model_structured"

RESULTS_FILE = BASE_DIR / "unseen_inference_results.json"

MAX_INPUT_LENGTH = 384
MAX_NEW_TOKENS = 192

NUM_BEAMS = 4


# ============================================================
# UNSEEN EXAMPLES
# ============================================================

UNSEEN_EXAMPLES = [

    {
        "id": "UNSEEN_001",

        "conversation": [
            {
                "role": "patient",
                "text":
                "I've had a burning pain in the upper part of my stomach since yesterday."
            },
            {
                "role": "assistant",
                "text":
                "How severe is the pain?"
            },
            {
                "role": "patient",
                "text":
                "About 6 out of 10, and it becomes worse after eating."
            },
            {
                "role": "assistant",
                "text":
                "Have you had vomiting, fever, or black stools?"
            },
            {
                "role": "patient",
                "text":
                "No vomiting or fever, and my stools look normal."
            }
        ],

        "clinical_summary": {
            "chief_complaint":
                "Burning upper abdominal pain",

            "onset":
                "Yesterday",

            "duration":
                "1 day",

            "location":
                "Upper abdomen",

            "severity":
                "6/10",

            "symptoms": [
                "Burning abdominal pain"
            ],

            "associated_symptoms": [
                "Pain worsens after eating"
            ],

            "negative_findings": [
                "No vomiting",
                "No fever",
                "No black stools"
            ],

            "medical_history": [
                "Not reported"
            ],

            "medications": [
                "Not reported"
            ],

            "allergies": [
                "Not reported"
            ],

            "red_flags":
                "Not assessed",

            "relevant_context":
                "Pain becomes worse after eating."
        }
    },


    {
        "id": "UNSEEN_002",

        "conversation": [
            {
                "role": "patient",
                "text":
                "My left ankle has been swollen and painful for three days."
            },
            {
                "role": "assistant",
                "text":
                "Did anything happen before the pain started?"
            },
            {
                "role": "patient",
                "text":
                "I twisted it while walking down the stairs."
            },
            {
                "role": "assistant",
                "text":
                "Are you able to put weight on the foot?"
            },
            {
                "role": "patient",
                "text":
                "Yes, but walking hurts. I don't have any numbness."
            }
        ],

        "clinical_summary": {
            "chief_complaint":
                "Left ankle pain and swelling",

            "onset":
                "After twisting ankle on stairs",

            "duration":
                "3 days",

            "location":
                "Left ankle",

            "severity":
                "Not reported",

            "symptoms": [
                "Ankle pain",
                "Ankle swelling"
            ],

            "associated_symptoms": [
                "Pain with walking"
            ],

            "negative_findings": [
                "No numbness"
            ],

            "medical_history": [
                "Not reported"
            ],

            "medications": [
                "Not reported"
            ],

            "allergies": [
                "Not reported"
            ],

            "red_flags":
                "Not assessed",

            "relevant_context":
                "Injury occurred after twisting the ankle while walking downstairs."
        }
    },


    {
        "id": "UNSEEN_003",

        "conversation": [
            {
                "role": "patient",
                "text":
                "I've been feeling unusually tired for about two weeks."
            },
            {
                "role": "assistant",
                "text":
                "Have you noticed anything else along with the tiredness?"
            },
            {
                "role": "patient",
                "text":
                "I've been getting dizzy when I stand up and I look paler than usual."
            },
            {
                "role": "assistant",
                "text":
                "Any chest pain or difficulty breathing?"
            },
            {
                "role": "patient",
                "text":
                "No chest pain and no trouble breathing."
            }
        ],

        "clinical_summary": {
            "chief_complaint":
                "Unusual fatigue",

            "onset":
                "Approximately two weeks ago",

            "duration":
                "2 weeks",

            "location":
                "Not applicable",

            "severity":
                "Not reported",

            "symptoms": [
                "Fatigue"
            ],

            "associated_symptoms": [
                "Dizziness on standing",
                "Pale appearance"
            ],

            "negative_findings": [
                "No chest pain",
                "No difficulty breathing"
            ],

            "medical_history": [
                "Not reported"
            ],

            "medications": [
                "Not reported"
            ],

            "allergies": [
                "Not reported"
            ],

            "red_flags":
                "Not assessed",

            "relevant_context":
                "Patient reports dizziness when standing and appearing paler than usual."
        }
    },


    {
        "id": "UNSEEN_004",

        "conversation": [
            {
                "role": "patient",
                "text":
                "I have a sore throat that started four days ago."
            },
            {
                "role": "assistant",
                "text":
                "Have you had a fever or cough?"
            },
            {
                "role": "patient",
                "text":
                "I had a low fever yesterday and I have a dry cough."
            },
            {
                "role": "assistant",
                "text":
                "Any difficulty swallowing or breathing?"
            },
            {
                "role": "patient",
                "text":
                "Swallowing is uncomfortable, but I can swallow. Breathing is normal."
            }
        ],

        "clinical_summary": {
            "chief_complaint":
                "Sore throat",

            "onset":
                "Four days ago",

            "duration":
                "4 days",

            "location":
                "Throat",

            "severity":
                "Not reported",

            "symptoms": [
                "Sore throat"
            ],

            "associated_symptoms": [
                "Low fever",
                "Dry cough",
                "Discomfort when swallowing"
            ],

            "negative_findings": [
                "No difficulty breathing",
                "Able to swallow"
            ],

            "medical_history": [
                "Not reported"
            ],

            "medications": [
                "Not reported"
            ],

            "allergies": [
                "Not reported"
            ],

            "red_flags":
                "Not assessed",

            "relevant_context":
                "Patient reports a low fever and dry cough."
        }
    },


    {
        "id": "UNSEEN_005",

        "conversation": [
            {
                "role": "patient",
                "text":
                "I've been having headaches mostly in the back of my head for a week."
            },
            {
                "role": "assistant",
                "text":
                "How severe are they?"
            },
            {
                "role": "patient",
                "text":
                "Usually around 4 out of 10. They get worse after working at my computer."
            },
            {
                "role": "assistant",
                "text":
                "Do you have vision changes, vomiting, or weakness?"
            },
            {
                "role": "patient",
                "text":
                "No vomiting or weakness. My vision is normal."
            }
        ],

        "clinical_summary": {
            "chief_complaint":
                "Headache",

            "onset":
                "One week ago",

            "duration":
                "1 week",

            "location":
                "Back of the head",

            "severity":
                "4/10",

            "symptoms": [
                "Headache"
            ],

            "associated_symptoms": [
                "Headache worsens after computer work"
            ],

            "negative_findings": [
                "No vomiting",
                "No weakness",
                "No vision changes"
            ],

            "medical_history": [
                "Not reported"
            ],

            "medications": [
                "Not reported"
            ],

            "allergies": [
                "Not reported"
            ],

            "red_flags":
                "Not assessed",

            "relevant_context":
                "Headaches worsen after working at a computer."
        }
    }
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("MODEL 6 — DIRECT UNSEEN INFERENCE")
print("=" * 80)

print(f"Model : {MODEL_DIR}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU   : {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print("✓ Tokenizer loaded")


print("\nLoading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()

print("✓ Model loaded")


# ============================================================
# BUILD INPUT
# ============================================================

def build_input(conversation):

    lines = []

    for message in conversation:

        role = message["role"].strip().lower()

        text = message["text"].strip()

        if role == "patient":

            lines.append(
                f"Patient: {text}"
            )

        elif role == "assistant":

            lines.append(
                f"Assistant: {text}"
            )

    return (
        "Summarize the following medical conversation "
        "into a structured doctor-readable clinical summary.\n\n"
        "Conversation:\n"
        + "\n".join(lines)
    )


# ============================================================
# FIELD HEADERS
# ============================================================

FIELDS = [
    "Chief Complaint",
    "Onset",
    "Duration",
    "Location",
    "Severity",
    "Symptoms",
    "Associated Symptoms",
    "Negative Findings",
    "Medical History",
    "Medications",
    "Allergies",
    "Red Flags",
    "Relevant Context",
]


# ============================================================
# ROBUST PARSER
# ============================================================

def parse_summary(text):

    parsed = {
        field: []
        for field in FIELDS
    }

    # --------------------------------------------------------
    # Remove everything after Recommendation.
    # This prevents repeated Recommendation garbage.
    # --------------------------------------------------------

    text = re.split(
        r"\bRecommendation\s*:",
        text,
        flags=re.IGNORECASE
    )[0]

    # --------------------------------------------------------
    # Extract fields using positions.
    # Works whether output is one line or many lines.
    # --------------------------------------------------------

    pattern = (
        r"(Chief Complaint|Onset|Duration|Location|Severity|"
        r"Symptoms|Associated Symptoms|Negative Findings|"
        r"Medical History|Medications|Allergies|Red Flags|"
        r"Relevant Context)\s*:"
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )

    for i, match in enumerate(matches):

        field_found = match.group(1)

        # Match exact canonical field name
        field = None

        for f in FIELDS:

            if f.lower() == field_found.lower():

                field = f
                break

        if field is None:
            continue

        start = match.end()

        if i + 1 < len(matches):

            end = matches[i + 1].start()

        else:

            end = len(text)

        value = text[start:end].strip()

        # ----------------------------------------------------
        # Split bullets if present
        # ----------------------------------------------------

        value = value.replace(
            "•",
            "\n"
        )

        parts = re.split(
            r"\n+|\s+-\s+",
            value
        )

        cleaned = []

        for part in parts:

            part = part.strip()

            part = part.lstrip("-• ")

            if part:

                cleaned.append(part)

        parsed[field] = cleaned

    return parsed


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace(
        "not reported",
        ""
    )

    text = text.replace(
        "not applicable",
        ""
    )

    text = re.sub(
        r"[^\w\s/]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXPECTED FIELD CONVERSION
# ============================================================

FIELD_MAP = {
    "Chief Complaint": "chief_complaint",
    "Onset": "onset",
    "Duration": "duration",
    "Location": "location",
    "Severity": "severity",
    "Symptoms": "symptoms",
    "Associated Symptoms": "associated_symptoms",
    "Negative Findings": "negative_findings",
    "Medical History": "medical_history",
    "Medications": "medications",
    "Allergies": "allergies",
    "Red Flags": "red_flags",
    "Relevant Context": "relevant_context",
}


def expected_fields(example):

    output = {}

    summary = example[
        "clinical_summary"
    ]

    for field, key in FIELD_MAP.items():

        value = summary.get(
            key,
            "Not reported"
        )

        if isinstance(value, list):

            output[field] = value

        else:

            output[field] = [value]

    return output


# ============================================================
# FIELD MATCHING
# ============================================================

def evaluate_field(expected, predicted):

    expected_norm = [
        normalize(x)
        for x in expected
        if normalize(x)
    ]

    predicted_norm = [
        normalize(x)
        for x in predicted
        if normalize(x)
    ]

    # Exact set
    if set(expected_norm) == set(predicted_norm):

        return "EXACT"

    # Semantic-ish substring matching
    matched = 0

    for expected_item in expected_norm:

        for predicted_item in predicted_norm:

            if (
                expected_item in predicted_item
                or predicted_item in expected_item
            ):

                matched += 1
                break

    if matched > 0:

        return "PARTIAL"

    return "WRONG"


# ============================================================
# GENERATION
# ============================================================

results = []

print("\n")
print("=" * 80)
print("RUNNING UNSEEN INFERENCE")
print("=" * 80)


for example in UNSEEN_EXAMPLES:

    example_id = example["id"]

    print("\n" + "-" * 80)
    print(f"EXAMPLE: {example_id}")

    input_text = build_input(
        example["conversation"]
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.no_grad():

        output_ids = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            num_beams=NUM_BEAMS,

            do_sample=False,

            early_stopping=True,

            no_repeat_ngram_size=3,

            repetition_penalty=1.1,

            length_penalty=1.0
        )

    generated = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()


    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    predicted = parse_summary(
        generated
    )

    expected = expected_fields(
        example
    )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluation = {}

    for field in FIELDS:

        evaluation[field] = evaluate_field(
            expected[field],
            predicted[field]
        )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nGENERATED:")
    print(generated)

    print("\nFIELD RESULTS:")

    for field in FIELDS:

        print(
            f"{field:<22} : "
            f"{evaluation[field]}"
        )


    results.append({

        "id":
            example_id,

        "generated_summary":
            generated,

        "expected_fields":
            expected,

        "predicted_fields":
            predicted,

        "field_evaluation":
            evaluation
    })


# ============================================================
# OVERALL
# ============================================================

counts = {
    "EXACT": 0,
    "PARTIAL": 0,
    "WRONG": 0
}

total = 0


for result in results:

    for status in result[
        "field_evaluation"
    ].values():

        counts[status] += 1

        total += 1


print("\n")
print("=" * 80)
print("OVERALL UNSEEN PERFORMANCE")
print("=" * 80)

print(
    f"Exact   : {counts['EXACT']}/{total} "
    f"({counts['EXACT']/total*100:.2f}%)"
)

print(
    f"Partial : {counts['PARTIAL']}/{total} "
    f"({counts['PARTIAL']/total*100:.2f}%)"
)

print(
    f"Wrong   : {counts['WRONG']}/{total} "
    f"({counts['WRONG']/total*100:.2f}%)"
)


# ============================================================
# SAVE
# ============================================================

final_output = {

    "model":
        str(MODEL_DIR),

    "num_examples":
        len(results),

    "overall":
        counts,

    "results":
        results
}


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_output,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print("=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)

print(
    f"Results saved to:\n{RESULTS_FILE}"
)