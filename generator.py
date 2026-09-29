import sys
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("models/SmolLM2-360M-Instruct")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

def generate_idea(prompt: str) -> str:
	inputs = TOKENIZER.apply_chat_template(
		[{"role": "user", "content": prompt}],
		add_generation_prompt=True,
		return_tensors="pt",
		return_dict=True,
	)

	seed = int(datetime.now().timestamp())  # Seed based on the current time
	torch.manual_seed(seed)
	with torch.inference_mode():
		output = MODEL.generate(
			**inputs,
			max_new_tokens=10000,
			do_sample=True,
			pad_token_id=TOKENIZER.eos_token_id,
		)
	new_tokens = output[0, inputs["input_ids"].shape[1] :]
	return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
	if len(sys.argv) > 1:
		print(generate_idea(sys.argv[1]))
