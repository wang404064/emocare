#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_dpo_gen.py — 用 DeepSeek API 生成高质量 DPO 数据
- rejected 由 LLM 生成（非规则模板），质量远高于规则
- 多样性控制：用户画像（年龄/性别/文化背景）、问题细节、情绪表达风格
- 输出格式：jsonl，兼容 LLaMA-Factory DPO 训练
"""

import os
import json
import time
import random
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm

# ── DeepSeek API Client（OpenAI 兼容）──────────────────────────────────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

DEEPEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"   # DeepSeek-V3
# 若要推理增强：deepseek-reasoner


# ── 多样性控制维度 ──────────────────────────────────────────────────────────
AGE_GROUPS = ["青少年(13-17岁)", "大学生(18-22岁)", "年轻职场人(23-30岁)",
              "中年(31-50岁)", "银发族(51-70岁)"]

GENDERS = ["男性", "女性", "非二元性别"]

CULTURES = ["都市白领", "小城镇居民", "农村背景", "海外华人", "跨文化家庭"]

EMOTION_STYLES = [
    "含蓄内敛型（不直接说感受，用叙述暗示）",
    "激烈外放型（情绪强烈，用词直接甚至极端）",
    "自责型（把问题归咎于自己）",
    "理性分析型（试图用逻辑解释情绪，但明显在压抑）",
    "依赖求助型（直接表达无助，强烈需要他人支持）",
    "回避型（轻描淡写，但字里行间透露痛苦）",
]

SCENE_DETAILS = {
    "分手或离婚": [
        "和平分手但还在同一个朋友圈", "对方出轨后发现，不知道怎么面对共同朋友",
        "结婚五年决定离婚，双方父母都反对", "异地恋分手，怀疑是不是距离的问题",
    ],
    "失业相关压力": [
        "35岁被优化，行业整体不景气", "应届生找不到工作，学历还被人质疑",
        "休息一年后返岗，发现岗位全变了", "自由职业收入骤减，房租快交不起了",
    ],
    "学业压力": [
        "考研二战，家里给的压力特别大", "博士资格考试前彻底崩了",
        "挂科两门，担心拿不到毕业证", "语言考试考了三次还是没过",
    ],
    "慢性病或疼痛管理": [
        "腰椎间盘突出压迫神经，走路都困难", "偏头痛十几年，最近频率越来越高",
        "类风湿，手指关节已经变形了", "慢性胃痛，检查不出器质性问题",
    ],
    "育儿挑战和父母内疚": [
        "二胎后大宝行为退行，感到愧疚", "孩子确诊 ADHD，怀疑是不是自己基因问题",
        "单亲妈妈一个人带俩孩子", "孩子在学校被霸凌，不知道怎么帮他",
    ],
    "焦虑和恐慌": [
        "第一次惊恐发作，以为自己要死了", "广场恐惧症，已经三个月没出过门",
        "健康焦虑，每次身体有点不舒服就查百度", "社交后的反刍，反复想自己是不是说错话了",
    ],
    "抑郁和低落情绪": [
        "持续两周起不来床，但对别人说自己挺好的", "产后抑郁，不敢告诉家人怕被说矫情",
        "退休后突然失去生活目标", "长期微笑抑郁，周围人都不知道",
    ],
    "工作倦怠": [
        "互联网大厂996，感觉自己就是个工具", "护士三班倒，已经对病人没有同理心了",
        "教师职业倦怠，对学生彻底失去耐心", "创业失败欠了债，还得假装正常上班",
    ],
    "照顾者支持": [
        "照顾失智症父亲三年，自己身体也垮了", "配偶车祸后半身不遂，心情复杂",
        "一边照顾重病配偶一边带娃", "照顾者内疚：有时希望对方早点解脱",
    ],
    "财务担忧": [
        "房贷+车贷+孩子学费，工资刚到账就没了", "投资失败，背了一身债",
        "信用卡最低还款都快还不上了", "父母生病，医疗费用没有着落",
    ],
}


# ── 系统 Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位专业的心理咨询师数据生成专家。
你的任务是为 DPO（Direct Preference Optimization）训练生成高质量数据对。

## 输出格式
每次输出必须是一个严格的 JSON 对象（不要用 Markdown 代码块包裹），包含三个字段：
- "prompt": 用户（来访者的）求助消息
- "chosen": 优质咨询师回复（共情为先，策略正确）
- "rejected": 劣质咨询师回复（看起来像好回复，但有明确的策略错误）

## chosen 写作要求
1. 必须先有情感验证（Emotional Validation），让用户感到被理解
2. 策略要自然嵌入，不要生硬地"给建议"
3. 语气温暖、真诚，不过度热情也不冷淡
4. 长度 80-150 字为宜

## rejected 写作要求（核心）
rejected 必须是「困难负样本」：
- 语气自然、看起来关心，不是明显的错误回复
- 长度与 chosen 相当（不能太短）
- 错误是策略性的，不是内容错误：
  · 过早建议：跳过共情直接给建议
  · 说教评判：表面理性，暗含"你也有责任"的暗示
  · 正常化过度：快速"大家都这样"，让用户的独特痛苦被忽视
  · 情感否定：用"但是/其实"逻辑翻转，让感受显得不合理
- 与 chosen 的语义相似度应在 0.6-0.85 之间（"看起来差不多，但有明显错误"）

## 重要
每次生成的 prompt 必须包含具体细节（人名、时间、地点、具体数字），不能泛泛而谈。
"""

USER_PROMPT_TEMPLATE = """【场景】：{scenario}
【应使用的支持策略】：{strategy}
【用户画像】：{age} | {gender} | {culture}
【情绪表达风格】：{emotion_style}
【具体情境细节】：{specific_detail}

请生成符合以上条件的 DPO 数据对（prompt + chosen + rejected）。
直接输出 JSON，不要任何额外文字。"""


# ── 主生成逻辑 ────────────────────────────────────────────────────────────────
def create_client(api_key: str, model: str = DEFAULT_MODEL) -> "OpenAI":
    if not HAS_OPENAI:
        raise ImportError("请安装 openai: pip install openai")
    return OpenAI(api_key=api_key, base_url=DEEPEK_BASE_URL), model


def generate_one(
    client: "OpenAI",
    model: str,
    scenario: str,
    strategy: str,
    age: str,
    gender: str,
    culture: str,
    emotion_style: str,
    specific_detail: str,
    max_retries: int = 3,
) -> Optional[Dict]:
    user_msg = USER_PROMPT_TEMPLATE.format(
        scenario=scenario,
        strategy=strategy,
        age=age,
        gender=gender,
        culture=culture,
        emotion_style=emotion_style,
        specific_detail=specific_detail,
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.8,
                top_p=0.95,
                max_tokens=800,
            )
            content = resp.choices[0].message.content.strip()

            # 清理可能的代码块包裹
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            if content.startswith("{"):
                content = content
            # 找到 JSON 边界
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != 0:
                content = content[start:end]

            data = json.loads(content)
            if not all(k in data for k in ("prompt", "chosen", "rejected")):
                raise ValueError("缺少必要字段")

            return {
                "prompt": data["prompt"].strip(),
                "chosen": data["chosen"].strip(),
                "rejected": data["rejected"].strip(),
                "meta": {
                    "scene": scenario,
                    "strategy": strategy,
                    "age": age,
                    "gender": gender,
                    "culture": culture,
                    "emotion_style": emotion_style,
                    "model": model,
                }
            }

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [失败] {scenario}/{strategy}: {e}")
                return None
            time.sleep(2 ** attempt)

    return None


def main():
    parser = argparse.ArgumentParser(description="用 DeepSeek API 生成 DPO 数据")
    parser.add_argument("--api_key", type=str, default="",
                        help="DeepSeek API Key（也可通过环境变量 DEEPEK_API_KEY 设置）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["deepseek-chat", "deepseek-reasoner"],
                        help="DeepSeek 模型名")
    parser.add_argument("--output", type=str, default="deepseek_dpo.jsonl")
    parser.add_argument("--num", type=int, default=1000,
                        help="生成总数")
    parser.add_argument("--per_scene", type=int, default=0,
                        help="每个场景至少生成几条（0=按总数均匀分配）")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="API 调用间隔（秒）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # API Key
    api_key = args.api_key or os.getenv("DEEPEK_API_KEY", "")
    if not api_key:
        print("请提供 DeepSeek API Key：")
        print("  方法1: --api_key sk-xxxx")
        print("  方法2: 设置环境变量 DEEPEK_API_KEY")
        return
    os.environ["DEEPEK_API_KEY"] = api_key  # 写入环境供 create_client 用

    random.seed(args.seed)

    client, model = create_client(api_key, args.model)

    # ── 构建场景×策略矩阵 ──
    all_scenes = list(SCENE_DETAILS.keys()) + [
        "分手或离婚", "导航性别认同和过渡", "沟通挑战", "职业转型",
        "应对所爱之人的死亡", "为人父母和育儿挑战", "处理宠物死亡",
        "自尊心低或缺乏自信", "身体形象问题和饮食失调",
        "LGBTQ+身份", "文化认同和归属感", "适应新工作或角色",
        "创伤后应激障碍", "应对诊断或医疗治疗", "从虐待中康复",
        "上瘾和康复", "寻找生活的意义和目标", "对所爱之人或朋友的支持",
    ]
    all_strategies = [
        "情感验证(EV)", "提供希望(ES)", "肯定(Aff)", "移情陈述(Emp)",
        "建议选项(SI)", "协同规划(CP)", "重新构建消极思想(RF)", "使经验正常化(Norm)",
    ]

    # 去重
    all_scenes = list(dict.fromkeys(all_scenes))

    print(f"场景数: {len(all_scenes)} | 策略数: {len(all_strategies)}")
    print(f"目标生成: {args.num} 条 | 模型: {model}")
    print(f"输出文件: {args.output}")
    print("-" * 60)

    # ── 断点续传 ──
    existing = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        print(f"检测到已有 {existing} 条，将追加写入...")

    # ── 生成任务队列 ──
    # 每个样本 = 随机选场景 + 策略 + 用户画像 + 具体细节
    tasks = []
    for _ in range(args.num):
        scene = random.choice(all_scenes)
        strategy = random.choice(all_strategies)
        age = random.choice(AGE_GROUPS)
        gender = random.choice(GENDERS)
        culture = random.choice(CULTURES)
        emotion = random.choice(EMOTION_STYLES)
        detail = random.choice(
            SCENE_DETAILS.get(scene, ["具体的个人困扰细节，由生成时自由发挥"])
        )
        tasks.append((scene, strategy, age, gender, culture, emotion, detail))

    # ── 执行生成 ──
    pbar = tqdm(total=args.num, initial=existing, desc="DeepSeek DPO 生成")
    success = existing
    failed = 0

    with open(args.output, "a", encoding="utf-8") as fout:
        for i, (scene, strategy, age, gender, culture, emotion, detail) in enumerate(tasks):
            if i < existing:   # 跳过已生成的
                continue

            sample = generate_one(
                client, model, scene, strategy,
                age, gender, culture, emotion, detail
            )

            if sample:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fout.flush()
                success += 1
                pbar.update(1)
            else:
                failed += 1

            time.sleep(args.delay)

    pbar.close()
    print(f"\n✅ 完成！成功: {success} 条 | 失败: {failed} 条")
    print(f"   输出: {args.output}")


if __name__ == "__main__":
    main()
