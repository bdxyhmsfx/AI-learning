from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
'''
model_inputs = tokenizer(sequences,padding="longest",return_tensors="pt")
model_inputs1 = tokenizer(sequences,max_length=8,truncation=True)

print(model_inputs.input_ids)
print(model_inputs1)
'''
model_inputs = tokenizer(sequences)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequences)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))

