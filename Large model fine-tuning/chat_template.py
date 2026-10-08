from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
message=[
    {
        "role":"system",
        "content":"you are assistant"
    },
    {
        "role":"user",
        "content":"hello"
    },
    {
        "role":"assistant",
        "content":"Hi,What can I do for you?"
    }
]
print(tokenizer.apply_chat_template(message,tokenize=False))