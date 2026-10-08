import torch
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTConfig,SFTTrainer

model_name = "Qwen/Qwen3-0.6B"
output_dir = "./qwen3_lora_output"
adapter_dir = "./qwen3_ai_lora_adapter"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备：", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_data = [

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是过拟合？"
            },
            {
                "role": "assistant",
                "content": "【定义】过拟合是模型在训练数据上表现很好，但在未见数据上表现较差的现象。\n【关键点】模型过度记住训练数据，导致泛化能力下降。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是梯度下降？"
            },
            {
                "role": "assistant",
                "content": "【定义】梯度下降是一种用于最小化损失函数的优化算法。\n【关键点】沿损失函数梯度的反方向不断更新模型参数。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Transformer？"
            },
            {
                "role": "assistant",
                "content": "【定义】Transformer 是一种主要基于注意力机制构建的神经网络架构。\n【关键点】它利用 Self-Attention 建模序列中不同位置之间的关系。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Attention？"
            },
            {
                "role": "assistant",
                "content": "【定义】Attention 是一种根据相关性动态聚合信息的机制。\n【关键点】模型通过 Q、K、V 计算不同位置之间的重要程度。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Token？"
            },
            {
                "role": "assistant",
                "content": "【定义】Token 是语言模型处理文本时使用的基本单位。\n【关键点】一个 Token 可以是单词、子词、字符或其他文本片段。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Tokenization？"
            },
            {
                "role": "assistant",
                "content": "【定义】Tokenization 是把原始文本转换成 Token 序列的过程。\n【关键点】随后 Token 会进一步映射成 Token ID 输入模型。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Batch？"
            },
            {
                "role": "assistant",
                "content": "【定义】Batch 是一次送入模型进行训练的一组样本。\n【关键点】Batch Size 决定一次 forward 和 backward 处理多少条数据。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Epoch？"
            },
            {
                "role": "assistant",
                "content": "【定义】Epoch 表示模型完整遍历一次训练数据集。\n【关键点】训练通常会进行多个 Epoch，让模型反复学习数据。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是学习率？"
            },
            {
                "role": "assistant",
                "content": "【定义】学习率控制模型每次参数更新的步长。\n【关键点】学习率过大可能训练不稳定，过小则可能导致学习速度过慢。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是损失函数？"
            },
            {
                "role": "assistant",
                "content": "【定义】损失函数用于衡量模型预测结果与目标结果之间的差异。\n【关键点】训练的目标通常就是尽可能降低损失函数。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Embedding？"
            },
            {
                "role": "assistant",
                "content": "【定义】Embedding 是把离散 Token ID 映射成连续向量的过程。\n【关键点】这些向量能够被神经网络进一步计算和学习。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Logits？"
            },
            {
                "role": "assistant",
                "content": "【定义】Logits 是模型输出的未经归一化的预测分数。\n【关键点】分类任务或语言模型中通常会进一步通过 Softmax 转换成概率分布。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Softmax？"
            },
            {
                "role": "assistant",
                "content": "【定义】Softmax 是把一组实数分数转换成概率分布的函数。\n【关键点】转换后的各项概率都大于等于零，并且总和为一。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是因果掩码？"
            },
            {
                "role": "assistant",
                "content": "【定义】因果掩码用于阻止当前位置看到未来 Token。\n【关键点】它保证 Causal LM 在预测下一个 Token 时只能利用当前位置之前的信息。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 SFT？"
            },
            {
                "role": "assistant",
                "content": "【定义】SFT 是监督微调，即使用高质量输入和标准回答继续训练预训练模型。\n【关键点】它可以让模型更加擅长理解并遵循人类指令。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 LoRA？"
            },
            {
                "role": "assistant",
                "content": "【定义】LoRA 是一种参数高效微调方法。\n【关键点】它冻结基础模型的大部分参数，只训练额外加入的低秩矩阵。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 QLoRA？"
            },
            {
                "role": "assistant",
                "content": "【定义】QLoRA 是结合量化和 LoRA 的参数高效微调方法。\n【关键点】它通常以低比特形式保存基础模型，从而进一步降低显存占用。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 PEFT？"
            },
            {
                "role": "assistant",
                "content": "【定义】PEFT 是参数高效微调方法的统称。\n【关键点】它通过只训练少量参数来降低大模型微调的计算和存储成本。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "BPE 是什么？"
            },
            {
                "role": "assistant",
                "content": "【定义】BPE 是一种常见的子词 Tokenization 算法。\n【关键点】训练过程中反复合并最常出现的相邻 Token 对。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "WordPiece 是什么？"
            },
            {
                "role": "assistant",
                "content": "【定义】WordPiece 是一种子词 Tokenization 方法。\n【关键点】训练时关注 Token 对的关联程度，编码时通常使用最长匹配。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "Unigram Tokenizer 是什么？"
            },
            {
                "role": "assistant",
                "content": "【定义】Unigram 是一种基于概率模型的子词 Tokenization 方法。\n【关键点】它从较大的候选词表开始，通过删除不重要 Token 并寻找全局最优切分。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是预训练？"
            },
            {
                "role": "assistant",
                "content": "【定义】预训练是在大规模数据上提前训练模型的过程。\n【关键点】大语言模型通过预测 Token 学习语言规律、知识以及文本结构。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 DPO？"
            },
            {
                "role": "assistant",
                "content": "【定义】DPO 是一种利用偏好数据直接优化语言模型的方法。\n【关键点】它让模型提高对优选回答的相对概率，并降低被拒绝回答的相对概率。"
            }
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 RLHF？"
            },
            {
                "role": "assistant",
                "content": "【定义】RLHF 是利用人类反馈进行强化学习的方法。\n【关键点】它利用人类偏好帮助语言模型生成更符合人类需求的回答。"
            }
        ]
    },
]
dataset = Dataset.from_list(train_data)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)
model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float16).to(device)

model.config.pad_token_id=tokenizer.pad_token_id

def ask_model(model,question):

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer

test_question_1 = "什么是过拟合？"
test_question_2 = "什么是欠拟合？"

before_answer_1 = ask_model(model, test_question_1)
before_answer_2 = ask_model(model, test_question_2)

peft_config=LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=1,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_length=256,
    packing=False,
    assistant_only_loss=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

del trainer
del model

if torch.cuda.is_available():
    torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float16)
base_model.config.pad_token_id=tokenizer.pad_token_id

lora_model = PeftModel.from_pretrained(base_model,adapter_dir)

after_answer_1 = ask_model(
    lora_model,
    test_question_1
)

after_answer_2 = ask_model(
    lora_model,
    test_question_2
)

print("\n\n" + "=" * 70)
print("微调前 VS 微调后")
print("=" * 70)

print("\n【问题1】")
print(test_question_1)

print("\n--- 微调前 ---")
print(before_answer_1)

print("\n--- 微调后 ---")
print(after_answer_1)


print("\n\n【问题2】")
print(test_question_2)

print("\n--- 微调前 ---")
print(before_answer_2)

print("\n--- 微调后 ---")
print(after_answer_2)