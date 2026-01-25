#!/usr/bin/env python3
"""
Qwen3-8B LoRA微调训练脚本
针对中文情感支持场景定制训练
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionSupportDataset(Dataset):
    """情感支持对话数据集"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建对话格式
        if "instruction" in item and "input" in item and "output" in item:
            # 标准指令格式
            prompt = f"用户: {item['instruction']}\n{item['input']}\n\n助手: {item['output']}"
        elif "question" in item and "answer" in item:
            # 问答格式
            prompt = f"问题: {item['question']}\n回答: {item['answer']}"
        else:
            # 通用格式
            prompt = item.get("text", str(item))

        # 编码
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'labels': encodings['input_ids'].flatten()  # 对于语言建模，labels与input_ids相同
        }

def load_training_data(data_path: str) -> List[Dict]:
    """加载训练数据"""

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"加载了 {len(data)} 条训练数据")
    return data

def create_lora_config(model_name: str) -> LoraConfig:
    """创建LoRA配置"""

    # Qwen模型的LoRA配置
    if "qwen" in model_name.lower():
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    else:
        # 其他模型的通用配置
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    lora_config = LoraConfig(
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA alpha参数
        lora_dropout=0.05,  # Dropout率
        bias="none",  # 不训练bias
        task_type=TaskType.CAUSAL_LM,  # 因果语言建模
        target_modules=target_modules,
        modules_to_save=["lm_head"]  # 保存输出层
    )

    return lora_config

def create_peft_model(model_name: str, lora_config: LoraConfig):
    """创建PEFT模型"""

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用半精度以节省显存
        device_map="auto",  # 自动设备映射
        trust_remote_code=True  # 信任远程代码（针对Qwen等模型）
    )

    # 准备模型进行k-bit训练（如果需要量化）
    model = prepare_model_for_kbit_training(model)

    # 获取PEFT模型
    model = get_peft_model(model, lora_config)

    # 打印可训练参数信息
    model.print_trainable_parameters()

    return model

def train_lora_model(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    gradient_accumulation_steps: int = 4
):
    """训练LoRA模型"""

    logger.info("开始LoRA训练...")

    # 加载数据
    train_data = load_training_data(train_data_path)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"  # Qwen模型推荐右填充
    )

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建数据集
    train_dataset = EmotionSupportDataset(train_data, tokenizer, max_length)

    # 创建LoRA配置
    lora_config = create_lora_config(model_name)

    # 创建PEFT模型
    model = create_peft_model(model_name, lora_config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,  # 使用混合精度训练
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",  # 不使用外部日志服务
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # 梯度检查点以节省显存
        max_grad_norm=0.3,  # 梯度裁剪
    )

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 不是MLM任务
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    logger.info(f"模型已保存到: {output_dir}")

    return model, tokenizer

def merge_lora_adapter(base_model_name: str, lora_path: str, output_path: str):
    """合并LoRA适配器到基础模型"""

    logger.info("开始合并LoRA适配器...")

    from peft import PeftModel

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载LoRA模型
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 合并权重
    merged_model = model.merge_and_unload()

    # 保存合并后的模型
    merged_model.save_pretrained(output_path)

    # 保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"合并后的模型已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B LoRA微调训练")
    parser.add_argument("--model_name", default="Qwen/Qwen2-7B-Instruct",
                       help="基础模型名称")
    parser.add_argument("--train_data", default="data/lora_training_data.json",
                       help="训练数据路径")
    parser.add_argument("--output_dir", default="models/lora_emotion_support",
                       help="LoRA模型输出目录")
    parser.add_argument("--merged_output", default="models/merged_emotion_support",
                       help="合并后模型输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                       help="梯度累积步数")
    parser.add_argument("--merge_only", action="store_true",
                       help="仅执行合并步骤")

    args = parser.parse_args()

    if args.merge_only:
        # 仅合并模型
        merge_lora_adapter(args.model_name, args.output_dir, args.merged_output)
    else:
        # 训练LoRA模型
        model, tokenizer = train_lora_model(
            model_name=args.model_name,
            train_data_path=args.train_data,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation
        )

        # 合并模型
        merge_lora_adapter(args.model_name, args.output_dir, args.merged_output)

    logger.info("LoRA训练和合并完成！")

if __name__ == "__main__":
    main()