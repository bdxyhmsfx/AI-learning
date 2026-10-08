import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer


# ============================================================
# 1. 基本配置
# ============================================================

model_name = "Qwen/Qwen3-0.6B"

output_dir = "./qwen3_lora_output"
adapter_dir = "./qwen3_ai_lora_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("当前设备：", device)


# ============================================================
# 2. 加载 Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(model_name)


# ============================================================
# 3. 准备 SFT 数据
#
# 每一条数据：
# user      -> 问题
# assistant -> 希望模型学会生成的标准回答
#
# 我故意把回答统一成：
# 【定义】
# 【关键点】
#
# 方便观察微调前后“回答风格”有没有变化
# ============================================================

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


# ============================================================
# 4. 转成 Hugging Face Dataset
# ============================================================

dataset = Dataset.from_list(train_data)

# 划分训练集和验证集
dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

print("\n训练集数量：", len(dataset["train"]))
print("验证集数量：", len(dataset["test"]))

print("\n第一条训练数据：")
print(dataset["train"][0])


# ============================================================
# 5. 加载 Base Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16
)

model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# 6. 定义生成函数
#
# enable_thinking=False：
# 这次只是做简单 SFT 实验，不需要 Qwen3 的 thinking 模式
# ============================================================

def ask_model(model, question):

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
            pad_token_id=tokenizer.eos_token_id
        )

    # 只取模型新生成的部分
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return answer


# ============================================================
# 7. 微调前先测试
# ============================================================

model = model.to(device)

test_question_1 = "什么是过拟合？"
test_question_2 = "什么是欠拟合？"

print("\n" + "=" * 60)
print("微调前：")
print("=" * 60)

before_answer_1 = ask_model(model, test_question_1)
before_answer_2 = ask_model(model, test_question_2)

print("\n问题1：", test_question_1)
print(before_answer_1)

print("\n问题2：", test_question_2)
print(before_answer_2)


# ============================================================
# 8. LoRA 配置
#
# r=8：
#   LoRA 低秩矩阵的 rank
#
# lora_alpha=16：
#   LoRA 更新的缩放因子
#
# target_modules：
#   只在 Attention 的 q_proj 和 v_proj 上加 LoRA
# ============================================================

peft_config = LoraConfig(
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


# ============================================================
# 9. SFT 训练参数
#
# 你的 8GB 显存先采用比较保守的配置
# ============================================================

training_args = SFTConfig(

    # checkpoint 保存位置
    output_dir=output_dir,

    # 训练 5 个 Epoch
    num_train_epochs=5,

    # GPU 一次真正放 1 条数据
    per_device_train_batch_size=1,

    # 连续累计 4 次梯度后再更新一次
    gradient_accumulation_steps=4,

    # LoRA 一般可以使用比 Full Fine-Tuning 更高的学习率
    learning_rate=1e-4,

    # 前 5% 的训练 step 做 warmup
    # warmup_ratio=0.05,
    warmup_steps=1,

    # 每一个 step 打印一次日志
    logging_steps=1,

    # 每个 Epoch 验证一次
    eval_strategy="epoch",

    # 每个 Epoch 保存一次
    save_strategy="epoch",

    # 最大 Token 长度
    max_length=256,

    # 不做 packing，第一次实验保持简单
    packing=False,

    # 只对 assistant 回答部分计算 loss
    assistant_only_loss=True,

    # RTX GPU 使用 FP16
    fp16=torch.cuda.is_available(),

    # 不上传 wandb 等平台
    report_to="none",
)


# ============================================================
# 10. 创建 SFTTrainer
#
# Dataset 本身就是：
#
# {
#   "messages": [...]
# }
#
# 所以当前 TRL 会自动应用 Qwen3 的 Chat Template
# 不需要我们手动 apply_chat_template
# ============================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config
)


# ============================================================
# 11. 查看到底训练多少参数
# ============================================================

print("\n" + "=" * 60)
print("LoRA 可训练参数：")
print("=" * 60)

trainer.model.print_trainable_parameters()


# ============================================================
# 12. 开始真正的 SFT + LoRA
# ============================================================

print("\n开始训练……")

trainer.train()

print("\n训练完成！")


# ============================================================
# 13. 保存 LoRA Adapter
#
# 这里保存的主要不是完整 Qwen3，
# 而是 LoRA 的 A、B 参数 + 配置
# ============================================================

trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print("\nLoRA Adapter 已保存到：")
print(adapter_dir)


# ============================================================
# 14. 删除当前训练模型，释放显存
#
# 这样接下来真正演示：
# Base Model + Adapter 重新加载
# ============================================================

del trainer
del model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 15. 重新加载原始 Base Model
# ============================================================

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16
)

base_model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# 16. 把刚刚训练好的 LoRA Adapter 挂到 Base Model
#
# Base Model
#       +
# LoRA Adapter
#       ↓
# lora_model
# ============================================================

lora_model = PeftModel.from_pretrained(
    base_model,
    adapter_dir
)

lora_model = lora_model.to(device)


# ============================================================
# 17. 微调后测试
# ============================================================

print("\n" + "=" * 60)
print("微调后：")
print("=" * 60)

after_answer_1 = ask_model(
    lora_model,
    test_question_1
)

after_answer_2 = ask_model(
    lora_model,
    test_question_2
)

print("\n问题1：", test_question_1)
print(after_answer_1)

print("\n问题2：", test_question_2)
print(after_answer_2)


# ============================================================
# 18. 最终前后对比
# ============================================================

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