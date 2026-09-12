from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_1_DIR = ROOT_DIR / "model_1(Conversation LLM)"
MODEL_2_DIR = ROOT_DIR / "model_2(Medical Dialogue Manager)"

MODEL_1_BASE = "meta-llama/Llama-3.1-8B-Instruct"

MODEL_1_ADAPTER = (
    MODEL_1_DIR
    / "outputs"
    / "model1_qlora"
    / "final"
)

MODEL_2_MODEL = (
    MODEL_2_DIR
    / "outputs"
    / "model2_lightgbm"
    / "final"
    / "model.txt"
)

MODEL_2_LABELS = (
    MODEL_2_DIR
    / "model2_preprocessed_lightgbm"
    / "label_mapping.json"
)

EMBEDDING_MODEL = "all-mpnet-base-v2"


# ============================================================
# MODEL 1
# ============================================================

class Model1:

    def __init__(
        self,
        base_model: str,
        adapter_path: Path,
    ) -> None:

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Model 1 adapter not found:\n{adapter_path}"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Model 1 inference."
            )

        print("\nLoading Model 1...")

        compute_dtype = torch.bfloat16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map={"": 0},
            dtype=compute_dtype,
        )

        self.model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path)
        )

        self.tokenizer.clean_up_tokenization_spaces = False

        self.model.eval()

        print("Model 1 loaded.")


    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 120,
    ) -> str:

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]

        response = self.tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        return response


# ============================================================
# MODEL 2
# ============================================================

class Model2:

    def __init__(
        self,
        model_path: Path,
        label_path: Path,
        embedding_model: str,
    ) -> None:

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model 2 model not found:\n{model_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Model 2 label mapping not found:\n{label_path}"
            )

        print("\nLoading Model 2...")

        with label_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            label_mapping = json.load(f)

        self.id2label = label_mapping["id2label"]

        self.model = lgb.Booster(
            model_file=str(model_path)
        )

        self.embedding_model = SentenceTransformer(
            embedding_model
        )

        print("Warming up Model 2 embeddings...", flush=True)
        self.embedding_model.encode(
            ["warmup"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        print("Model 2 loaded.")


    @staticmethod
    def flatten_context(
        context: list[dict[str, str]],
    ) -> str:

        flattened = []

        for message in context:

            role = message["role"].strip().lower()
            content = message["content"].strip()

            flattened.append(
                f"{role.capitalize()}: {content}"
            )

        return " | ".join(flattened)


    def predict(
        self,
        context: list[dict[str, str]],
        top_k: int = 5,
    ) -> dict[str, Any]:

        text = self.flatten_context(context)

        embedding = self.embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        probabilities = self.model.predict(
            embedding
        )[0]

        top_indices = np.argsort(
            probabilities
        )[::-1][:top_k]

        top_predictions = []

        for class_id in top_indices:

            class_id = int(class_id)

            top_predictions.append(
                {
                    "label": self.id2label[
                        str(class_id)
                    ],
                    "confidence": float(
                        probabilities[class_id]
                    ),
                }
            )

        predicted_class = int(
            np.argmax(probabilities)
        )

        return {
            "predicted_label": self.id2label[
                str(predicted_class)
            ],
            "confidence": float(
                probabilities[predicted_class]
            ),
            "top_predictions": top_predictions,
        }


# ============================================================
# PROMPTS
# ============================================================

OPENING_SYSTEM_PROMPT = """
You are a careful medical dialogue assistant.

Start the conversation naturally.

Greet the patient and ask exactly ONE concise question
about their main symptom.

Do not diagnose.
Do not give medical conclusions.
Use plain and empathetic language.
""".strip()


ACTION_SYSTEM_PROMPT = """
You are the natural-language response generator in a medical
dialogue system.

A separate dialogue manager called Model 2 has already decided
what information should be gathered NEXT.

Your ONLY job is to convert that action into one natural,
patient-friendly response.

SELECTED ACTION:
{label}

CONFIDENCE:
{confidence:.3f}

PREVIOUS ASSISTANT MESSAGE:
{previous_question}

STRICT RULES:

1. Ask exactly ONE concise question.

2. The question MUST correspond to the selected action.

3. Do NOT repeat the previous assistant question.

4. Do NOT ask about a different symptom.

5. Do NOT diagnose.

6. Do NOT mention Model 2.

7. Do NOT mention labels.

8. Do NOT mention confidence.

9. Use the patient's previous answers as context.

10. If the patient has already provided information relevant
to the selected action, ask a natural follow-up instead of
repeating the same question.

11. If the action is RED_FLAG, advise urgent medical attention
instead of asking a routine question.

12. Generate ONLY the patient's-facing response.

CONVERSATION:
""".strip()


# ============================================================
# HELPERS
# ============================================================

def get_last_assistant_message(
    history: list[dict[str, str]],
) -> str:

    for message in reversed(history):

        if message["role"] == "assistant":
            return message["content"].strip()

    return ""


def normalize_text(text: str) -> str:

    return " ".join(
        text.lower().strip().split()
    )


def is_repeated_response(
    response: str,
    history: list[dict[str, str]],
) -> bool:

    response_normalized = normalize_text(
        response
    )

    for message in history:

        if message["role"] != "assistant":
            continue

        previous = normalize_text(
            message["content"]
        )

        if response_normalized == previous:
            return True

    return False


def select_action(
    decision: dict[str, Any],
    used_actions: set[str],
) -> str:

    predictions = decision["top_predictions"]

    for prediction in predictions:

        action = prediction["label"]

        if action not in used_actions:
            return action

    return decision["predicted_label"]


def fallback_response(
    action: str,
    history: list[dict[str, str]],
) -> str:

    questions = {
        "associated_symptoms": "What other symptoms have you noticed?",
        "fever_temperature": "What is the highest temperature you have measured?",
        "fever_progression": "Has the fever been getting better or worse?",
        "rash_character": "What does the rash look and feel like?",
        "rash_location": "Where on your body is the rash?",
        "rash_progression": "Has the rash been spreading or changing?",
        "severity": "How severe are your symptoms right now?",
        "duration": "How long have these symptoms been present?",
    }

    candidate = questions.get(
        action,
        "What else can you tell me about this symptom?",
    )

    if not is_repeated_response(candidate, history):
        return candidate

    return "What has changed since you first noticed this symptom?"


# ============================================================
# CREATE PIPELINE
# ============================================================

def create_pipeline() -> tuple[Model1, Model2]:

    model_1 = Model1(
        MODEL_1_BASE,
        MODEL_1_ADAPTER,
    )

    model_2 = Model2(
        MODEL_2_MODEL,
        MODEL_2_LABELS,
        EMBEDDING_MODEL,
    )

    return model_1, model_2


# ============================================================
# CONVERSATION
# ============================================================

def run_conversation(
    model_1: Model1,
    model_2: Model2,
) -> None:

    history: list[dict[str, str]] = []
    used_actions: set[str] = set()

    # --------------------------------------------------------
    # MODEL 1 OPENS
    # --------------------------------------------------------

    opening = model_1.generate(
        [
            {
                "role": "system",
                "content": OPENING_SYSTEM_PROMPT,
            }
        ]
    )

    history.append(
        {
            "role": "assistant",
            "content": opening,
        }
    )

    print(
        f"\nAssistant: {opening}"
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        try:

            patient_message = input(
                "You: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print(
                "\nConversation ended."
            )
            return

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if patient_message.lower() in {
            "exit",
            "quit",
            "bye",
        }:

            print(
                "Conversation ended."
            )
            return

        if not patient_message:
            continue

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        history.append(
            {
                "role": "user",
                "content": patient_message,
            }
        )

        print("Processing your message...", flush=True)

        # ----------------------------------------------------
        # MODEL 2
        # ----------------------------------------------------

        print("Model 2 is selecting the next question...", flush=True)

        decision = model_2.predict(
            history,
            top_k=5,
        )

        action = select_action(
            decision,
            used_actions,
        )

        used_actions.add(action)

        confidence = decision[
            "confidence"
        ]

        # ----------------------------------------------------
        # PREVIOUS QUESTION
        # ----------------------------------------------------

        previous_question = get_last_assistant_message(
            history[:-1]
        )

        # ----------------------------------------------------
        # MODEL 1
        # ----------------------------------------------------

        print("Model 1 is preparing the response...", flush=True)

        response_prompt = ACTION_SYSTEM_PROMPT.format(
            label=action,
            confidence=confidence,
            previous_question=previous_question,
        )

        response_messages = [
            {
                "role": "system",
                "content": response_prompt,
            },
            *history,
        ]

        assistant_message = model_1.generate(
            response_messages
        )

        # ----------------------------------------------------
        # REPETITION PROTECTION
        # ----------------------------------------------------

        if is_repeated_response(
            assistant_message,
            history,
        ):

            retry_prompt = response_prompt + """

IMPORTANT RETRY:

Your previous response repeated a previous response.

Generate a DIFFERENT question.

The question must still correspond
to the selected action.

Ask exactly ONE question.
""".strip()

            retry_messages = [
                {
                    "role": "system",
                    "content": retry_prompt,
                },
                *history,
            ]

            assistant_message = model_1.generate(
                retry_messages
            )

        if is_repeated_response(
            assistant_message,
            history,
        ):

            assistant_message = fallback_response(
                action,
                history,
            )

        # ----------------------------------------------------
        # SAVE RESPONSE
        # ----------------------------------------------------

        history.append(
            {
                "role": "assistant",
                "content": assistant_message,
            }
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print(
            f"Assistant: {assistant_message}"
        )

        print(
            f"[internal Model 2 action: "
            f"{action} | "
            f"confidence: "
            f"{confidence:.3f}]"
        )

        print(
            "[internal top predictions: "
            + ", ".join(
                f"{item['label']}="
                f"{item['confidence']:.3f}"
                for item in decision[
                    "top_predictions"
                ]
            )
            + "]"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run Model 1 + Model 2 medical "
            "conversation pipeline."
        )
    )

    parser.parse_args()

    print("=" * 70)
    print("MODEL 1 + MODEL 2 MEDICAL CONVERSATION PIPELINE")
    print("=" * 70)

    print("\nLoading models...")

    model_1, model_2 = create_pipeline()

    print("\n" + "=" * 70)
    print("READY")
    print("Type 'exit', 'quit', or 'bye' to stop.")
    print("=" * 70)

    run_conversation(
        model_1,
        model_2,
    )


if __name__ == "__main__":
    main()