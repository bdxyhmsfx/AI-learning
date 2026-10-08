import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

'''
sequence="I've been waiting for a HuggingFace course my whole life."
tokens=tokenizer.tokenize(sequence)
ids=tokenizer.convert_tokens_to_ids(tokens)
input_ids=torch.tensor([ids,ids])
print(input_ids)
out=model(input_ids)
print(out)
'''
pad_token_id=100
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
attention_mask=[[1,1,1],
                [1,1,0],]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids),attention_mask=torch.tensor(attention_mask)).logits)