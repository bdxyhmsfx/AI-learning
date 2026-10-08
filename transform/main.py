import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

checkpoint="gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

text="Artificial intelligence will"
inputs=tokenizer(text,return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.logits.shape)

for temperature in [0.2, 0.7, 1.2]:
    generate_ids=model.generate(**inputs,max_new_tokens=30,do_sample=True,temperature=temperature)
    new_text=tokenizer.decode(generate_ids[0],skip_special_tokens=True)
    print(new_text)
