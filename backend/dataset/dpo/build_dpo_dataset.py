#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO 数据生成脚本（Qwen 专用）
使用 Qwen（通义千问）API 生成情感支持场景下的 DPO 训练数据（chosen vs rejected）
API 通过 DashScope OpenAI 兼容模式调用
"""

import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from dotenv import load_dotenv
import jsonlines

SCRIPT_DIR = Path(__file__).parent.resolve()
load_dotenv(SCRIPT_DIR / ".env")

# === 情感支持场景与策略（来自 ExTES）===
SCENARIOS = [
    "分手或离婚", "导航性别认同和过渡", "冲突或沟通问题", "搬到新的城市或国家",
    "沟通挑战", "职业转型", "应对所爱之人的死亡", "为人父母和育儿挑战",
    "处理宠物死亡", "自尊心低或缺乏自信", "与工作相关的压力和倦怠", "身体形象问题和饮食失调",
    "财务担忧和不确定的未来", "LGBTQ+身份", "文化认同和归属感", "失业或职业挫折",
    "学业压力", "学术压力", "灵性和信仰", "育儿挑战和父母内疚",
    "失业相关压力", "兄弟姐妹间的竞争或家庭冲突", "从身体或情感虐待中幸存和恢复", "从性侵犯或家庭暴力中康复",
    "焦虑和恐慌", "抑郁和低落情绪", "适应新工作或角色", "创伤后应激障碍",
    "应对诊断或医疗治疗", "从虐待中康复", "照顾者支持", "上瘾和康复",
    "寻找生活的意义和目标", "对所爱之人或朋友的支持"
]

STRATEGIES = [
    "情感验证", "提供希望", "肯定", "移情陈述",
    "建议选项", "协同规划", "重新构建消极思想", "使经验正常化"
]


# === LLM 客户端（复用自扩写脚本）===
class CustomAPIClient:
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-max"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("请安装 requests: pip install requests")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000
        }

        for attempt in range(max_retries):
            try:
                response = self.requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
        return ""


def create_qwen_client(model_name: str = "qwen-max") -> CustomAPIClient:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return CustomAPIClient(api_key=api_key, base_url=base_url, model=model_name)


def load_safety_prompt_template() -> str:
    """加载带安全限制的prompt模板"""
    prompt_file = SCRIPT_DIR / "dpo_safety_prompt.txt"
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # 如果文件不存在，使用简化版本（向后兼容）
        return """你是一位专业的情感支持心理咨询师。请根据以下要求生成一条 DPO（Direct Preference Optimization）训练数据：

【场景】：{scenario}
【应使用的支持策略】：{strategy}

请生成：
1. 一个真实、自然的用户输入（表达困扰、情绪或问题）
2. 一个"优质回复"（chosen）：积极、共情、符合策略、提供有效支持
3. 一个"劣质回复"（rejected）：冷漠、敷衍、评判性、或缺乏支持性

输出必须是严格符合以下 JSON 格式的单个对象，不要任何额外文字：
{{
  "prompt": "用户说的话",
  "chosen": "优质回复",
  "rejected": "劣质回复"
}}"""


def generate_dpo_sample(
    client: CustomAPIClient,
    scenario: str,
    strategy: str,
    max_retries: int = 3,
    use_safety_prompt: bool = True
) -> Optional[Dict[str, Any]]:
    """生成DPO样本，支持安全限制prompt"""
    if use_safety_prompt:
        prompt_template = load_safety_prompt_template()
        # 替换场景和策略占位符
        prompt_template = prompt_template.replace("{scenario}", scenario)
        prompt_template = prompt_template.replace("{strategy}", strategy)
    else:
        # 使用简化版本（向后兼容）
        prompt_template = f"""你是一位专业的情感支持心理咨询师。请根据以下要求生成一条 DPO（Direct Preference Optimization）训练数据：

【场景】：{scenario}
【应使用的支持策略】：{strategy}

请生成：
1. 一个真实、自然的用户输入（表达困扰、情绪或问题）
2. 一个"优质回复"（chosen）：积极、共情、符合策略、提供有效支持
3. 一个"劣质回复"（rejected）：冷漠、敷衍、评判性、或缺乏支持性

输出必须是严格符合以下 JSON 格式的单个对象，不要任何额外文字：
{{
  "prompt": "用户说的话",
  "chosen": "优质回复",
  "rejected": "劣质回复"
}}
"""

    for attempt in range(max_retries):
        try:
            raw_response = client.generate(prompt_template)
            content = raw_response.strip()

            # 清理可能的 Markdown 代码块
            if content.startswith("```json"):
                content = content[7:].strip()
            if content.startswith("```"):
                content = content[3:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            data = json.loads(content)

            if not all(k in data for k in ["prompt", "chosen", "rejected"]):
                raise ValueError("缺少必要字段")

            return {
                "prompt": data["prompt"].strip(),
                "chosen": data["chosen"].strip(),
                "rejected": data["rejected"].strip(),
                "metadata": {
                    "scenario": scenario,
                    "strategy": strategy
                }
            }

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"最终失败: {e}")
                return None
            time.sleep(2 ** attempt)

    return None


def main(
    output_file: str = "emo_dpo_dataset.jsonl",
    total_samples: int = 100,
    model: str = "qwen-max",
    delay: float = 1.0,
    use_safety_prompt: bool = True
):
    client = create_qwen_client(model_name=model)

    # 检查是否已有部分结果（断点续传）
    existing_count = 0
    if os.path.exists(output_file):
        with jsonlines.open(output_file, mode='r') as reader:
            existing_count = sum(1 for _ in reader)
        print(f"检测到已有 {existing_count} 条数据，将从第 {existing_count + 1} 条继续生成...")

    if existing_count >= total_samples:
        print("目标数量已达成，无需继续生成。")
        return

    pbar = tqdm(total=total_samples, initial=existing_count, desc="DPO 生成进度")

    with jsonlines.open(output_file, mode='a') as writer:
        for i in range(existing_count, total_samples):
            scenario = random.choice(SCENARIOS)
            strategy = random.choice(STRATEGIES)

            sample = generate_dpo_sample(client, scenario, strategy, use_safety_prompt=use_safety_prompt)
            if sample:
                writer.write(sample)
                pbar.update(1)
            else:
                print(f"[失败] 场景: {scenario} | 策略: {strategy}")

            time.sleep(delay)

    pbar.close()
    print(f"✅ 完成！共生成 {total_samples} 条 DPO 数据，保存至 {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="使用 Qwen 生成情感支持 DPO 数据集")
    parser.add_argument("--output", type=str, default="emo_dpo_dataset.jsonl")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--model", type=str, default="qwen3-max",
                        choices=["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-longcontext"])
    parser.add_argument("--delay", type=float, default=2.0, help="API 调用间隔（秒）")
    parser.add_argument("--no-safety", action="store_true", help="不使用安全限制prompt（使用简化版本）")

    args = parser.parse_args()
    main(
        output_file=args.output,
        total_samples=args.num_samples,
        model=args.model,
        delay=args.delay,
        use_safety_prompt=not args.no_safety
    )