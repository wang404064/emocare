#!/usr/bin/env python3
"""
DPO对齐训练脚本
提升模型在情感支持对话中的表现质量
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from trl import DPOTrainer, DPOConfig
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DPODataset(Dataset):
    """DPO训练数据集"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建prompt
        if "instruction" in item and "input" in item:
            prompt = f"用户: {item['instruction']}\n{item['input']}\n\n助手:"
        else:
            prompt = item.get("prompt", "")

        # 选择和拒绝的回应
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

def load_dpo_data(data_path: str) -> List[Dict]:
    """加载DPO训练数据"""

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"加载了 {len(data)} 条DPO训练数据")
    return data

def train_dpo_model(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    beta: float = 0.1,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-7,
    max_length: int = 1024,
    gradient_accumulation_steps: int = 4
):
    """训练DPO模型"""

    logger.info("开始DPO训练...")

    # 加载数据
    train_data = load_dpo_data(train_data_path)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建数据集
    train_dataset = DPODataset(train_data, tokenizer, max_length)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 参考模型（用于DPO计算）
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # DPO训练配置
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        beta=beta,  # DPO beta参数
        max_prompt_length=max_length // 2,  # prompt最大长度
        max_length=max_length,  # 总最大长度
    )

    # 创建DPO训练器
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    logger.info("开始DPO训练...")
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    logger.info(f"DPO模型已保存到: {output_dir}")

    return model, tokenizer

def create_emotion_support_dpo_data() -> List[Dict[str, Any]]:
    """创建情感支持DPO数据"""

    dpo_data = [
        {
            "instruction": "用户感到焦虑，请提供支持。",
            "input": "我最近总是担心未来，睡不着觉。",
            "chosen": "我完全理解这种对未来的担忧会让人夜不能寐。许多人都有类似的经历，这是一种正常的反应。让我和你一起面对这个：\n\n首先，深呼吸。现在请试着慢慢吸气4秒，屏息4秒，呼气4秒。感觉如何？\n\n关于对未来的担忧，我们可以试着：\n1. 将大问题分解成小步骤\n2. 专注于当下能控制的事情\n3. 建立支持系统\n\n你愿意和我分享具体在担心什么吗？我在这里陪着你。",
            "rejected": "放松点，不就是睡不着吗？很多人都是这样的。多喝点牛奶，早点睡就好了。"
        },
        {
            "instruction": "用户表达自我怀疑，请给予鼓励。",
            "input": "我觉得自己一无是处，做什么都失败。",
            "chosen": "听到你这么说，我的心都为你感到难过。自我怀疑确实会让人感到很痛苦，我理解这有多难受。我想告诉你，你的感受是真实的，但这并不意味着它是事实。\n\n每个人都有价值，每个人都会经历挫折。这并不定义你的全部。\n\n让我问你：你能想起一个你引以为傲的时刻吗？或者一个你帮助过别人的经历？这些都是你力量的证明。\n\n我们一起慢慢来，好吗？",
            "rejected": "别这么想，你又没做什么坏事。振作起来，想想开心的事。"
        },
        {
            "instruction": "用户感到孤独，请提供陪伴。",
            "input": "我感觉很孤独，没有什么朋友。",
            "chosen": "谢谢你愿意分享这个感受。孤独确实是一种很折磨人的情绪，我完全理解这有多难受。许多人都有类似的经历，你并不孤单。\n\n建立连接需要时间，就像种植一棵树。你已经迈出了勇敢的一步，愿意面对这个问题。这本身就是成长的开始。\n\n我们可以一起探索一些可行的方式：\n1. 从小群体的互动开始，比如兴趣小组\n2. 质量胜于数量，专注于建立有意义的关系\n3. 专业支持：考虑参加团体活动或寻求咨询\n\n同时也要善待自己，孤独的时候给自己多一些温柔和耐心。",
            "rejected": "多交朋友就好了。你太宅了，多出去走走。"
        },
        {
            "instruction": "用户感到压力大，请提供解压建议。",
            "input": "工作压力太大，我快撑不住了。",
            "chosen": "我能感受到你现在的压力和疲惫，这真的不容易。工作压力确实会让人感到很煎熬，让我们一起找到一些缓解的方法。\n\n首先，认可你的感受：你已经很努力了，能意识到需要寻求帮助，这已经很棒了。\n\n一些实用的压力管理建议：\n1. **时间管理**：尝试优先级排序，只做最重要的事情\n2. **界限设置**：学会说“不”，保护自己的精力\n3. **放松练习**：深呼吸、冥想，或简单的散步\n4. **寻求支持**：和信任的人聊聊，或考虑专业咨询\n\n你能和我分享一下具体是什么让你感到特别有压力吗？让我们一起面对。",
            "rejected": "工作就是这样的，谁都不容易。你忍忍就过去了。"
        }
    ]

    return dpo_data

def prepare_dpo_dataset(output_path: str = "data/dpo_training_data.json"):
    """准备DPO训练数据集"""

    print("开始准备DPO训练数据...")

    dpo_data = create_emotion_support_dpo_data()

    # 数据增强
    enhanced_data = []
    for item in dpo_data:
        enhanced_data.append(item)

        # 添加温和版本
        if "焦虑" in item["input"]:
            mild_item = item.copy()
            mild_item["input"] = item["input"].replace("总是担心", "偶尔担心")
            enhanced_data.append(mild_item)

    # 保存数据
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

    print(f"DPO训练数据已保存到: {output_file}")
    print(f"数据量: {len(enhanced_data)} 条")

    return enhanced_data

def main():
    parser = argparse.ArgumentParser(description="DPO对齐训练")
    parser.add_argument("--model_name", default="models/merged_emotion_support",
                       help="基础模型路径")
    parser.add_argument("--train_data", default="data/dpo_training_data.json",
                       help="DPO训练数据路径")
    parser.add_argument("--output_dir", default="models/dpo_emotion_support",
                       help="DPO模型输出目录")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta参数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                       help="梯度累积步数")
    parser.add_argument("--prepare_data_only", action="store_true",
                       help="仅准备数据")

    args = parser.parse_args()

    if args.prepare_data_only:
        # 仅准备数据
        prepare_dpo_dataset(args.train_data)
    else:
        # 确保数据存在
        if not Path(args.train_data).exists():
            logger.info("DPO数据不存在，正在准备...")
            prepare_dpo_dataset(args.train_data)

        # 训练DPO模型
        model, tokenizer = train_dpo_model(
            model_name=args.model_name,
            train_data_path=args.train_data,
            output_dir=args.output_dir,
            beta=args.beta,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation
        )

    logger.info("DPO训练完成！")

if __name__ == "__main__":
    main()