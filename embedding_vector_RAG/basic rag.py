from transformers import AutoTokenizer,AutoModelForCausalLM
from Retriever import retriever
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name="Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)
llm = AutoModelForCausalLM.from_pretrained(model_name,dtype=dtype).to(device)

def rag(question,k):
    results = retriever(question,k)
    context = "\n\n".join(
        result["text"] for result in results
    )
    prompt = f"""
请只根据下面提供的资料回答问题。
如果资料中没有答案，请明确回答“资料中没有相关信息”。
    
资料：
{context}

问题：
{question}
"""
    messages=[
        {
            "role":"user",
            "content":prompt
        }
    ]
    inputs=tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt"
    ).to(llm.device)

    llm.eval()

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids,skip_special_tokens=True)
    return answer

if __name__ == "__main__":

    test_questions = [
        "LoRA 中的 r 是什么意思？",
        "RAG 是怎么工作的？",
        "CNN 通常用于什么任务？",
        "支持向量机 SVM 是什么？"
    ]

    for question in test_questions:

        print("\n" + "=" * 60)
        print(f"问题：{question}")

        results = retriever(
            question,
            k=3
        )

        print("\n【Retriever 检索结果】")

        for i, result in enumerate(
            results,
            start=1
        ):
            print(
                f"\nTop-{i}"
            )

            print(
                f"distance: "
                f"{result['distance']:.4f}"
            )

            print(
                f"text: "
                f"{result['text']}"
            )

        answer = rag(
            question,
            k=3
        )

        print("\n【RAG 最终回答】")
        print(answer)