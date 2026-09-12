import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "medalpaca/medalpaca-7b"

print("=" * 60)
print("LOADING MEDALPACA 7B")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

print("\nMODEL LOADED SUCCESSFULLY")

prompt = """You are a medical conversational assistant.

Patient: I have had a headache since yesterday and I feel slightly nauseous.

Assistant:"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)
print(response)