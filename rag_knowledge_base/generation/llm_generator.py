import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dtype=torch.float16 if torch.cuda.is_available() else torch.float32
llm=AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
llm.eval()
def generate_answer(prompt,max_new_tokens=200):
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt"
    ).to(llm.device)
    with torch.no_grad():
        outputs = llm.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        input_length=inputs["input_ids"].shape[1]
        generated_ids=outputs[0][input_length:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer