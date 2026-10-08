from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text="Héllò hôw are ü?"
print(tokenizer.backend_tokenizer.normalizer.normalize_str(text))
'''Using a Transformer network is simple'''