#!/usr/bin/env python3
"""
LLM-as-Judge 评估脚本 — 5 维度评估情感陪护模型回复质量

维度:
  1. 流畅性 (Fluency)    — 语法、自然度
  2. 相关性 (Relevance)   — 是否紧扣用户问题
  3. 情感一致性 (Emotion) — 情感基调是否匹配
  4. 多样性 (Diversity)   — 是否避免模板化回复
  5. 共情能力 (Empathy)   — 是否体现理解和关怀

用法:
  python llm_evaluator.py \
    --eval_data eval_dataset.jsonl \
    --target_model your-model \
    --judge_model deepseek-chat \
    --api_key sk-xxx
"""

import json
import time
import argparse
from typing import List, Dict


# ============================================================
# 评估 Prompt
# ============================================================
JUDGE_PROMPT = """你是一位心理咨询质量评估专家。你需要评估一位 AI 情感支持助手对用户问题的回复质量。

## 评估标准

请从以下 5 个维度分别打分（1-5 分，必须是整数）：

### 1. 流畅性 / 自然性（Fluency）
- 5分：语法完全正确，读起来完全像真人咨询师说的话，长短句搭配自然
- 3分：基本通顺，但有一两处表达生硬或书面语过重
- 1分：多处语法错误、不通顺，或明显像机器生成的模板

### 2. 相关性（Relevance）
- 5分：精准回应用户的具体问题，没有答非所问，紧扣用户提到的细节
- 3分：大致相关，但部分内容笼统，缺乏对用户具体处境的针对性
- 1分：完全偏离用户问题，或只给了与问题无关的套话

### 3. 情感一致性（Emotional Consistency）
- 5分：回复的情感基调与用户的情绪状态高度匹配——用户难过时温暖、用户愤怒时先接纳、用户绝望时给予稳定感
- 3分：情感基调大致合适，但个别句子与整体氛围不协调（如用户很痛苦时过早用轻松语气）
- 1分：情感基调完全错位——用户绝望时回复欢快、用户愤怒时回复冷漠

### 4. 多样性（Diversity）
- 5分：回复没有使用任何模板化句式，表达方式独特且贴近当前对话上下文
- 3分：有一两句套话（如"我理解你的感受""一切都会好起来的"），但整体仍有个性化内容
- 1分：回复就是一套模板句式的拼接，换个问题也能原样用

### 5. 共情能力（Empathy）
- 5分：回复体现了深层的理解和关怀——不仅复述了用户的感受，还挖掘了用户没说出口的情绪，同时保持了专业边界
- 3分：有一定共情（如"我能理解这很难"），但停留在表面，没有深入理解用户的处境
- 1分：完全没有共情，直接给建议、说教、或否定用户的感受

## 用户问题
{prompt}

## AI 助手的回复
{response}

## 输出格式
请输出一个 JSON 对象（不要 Markdown 代码块），格式如下：
{{"fluency": 整数1-5, "relevance": 整数1-5, "emotion": 整数1-5, "diversity": 整数1-5, "empathy": 整数1-5, "overall_comment": "一句话总评，不超过50字"}}
"""


# ============================================================
# 评估数据集
# ============================================================
def get_default_eval_dataset() -> List[Dict]:
    """内置评估数据集：覆盖 6 类情感场景 × 不同情绪强度"""
    return [
        # ── 焦虑/压力类 ──
        {
            "id": "anxiety_1",
            "scene": "学业压力",
            "emotion": "焦虑",
            "prompt": "最近考试一个接一个，我每天晚上复习到凌晨两点，但还是觉得好多东西没记住，白天上课脑子都是糊的，我该怎么办？"
        },
        {
            "id": "anxiety_2",
            "scene": "职场焦虑",
            "emotion": "焦虑+自我怀疑",
            "prompt": "领导让我下周做一个汇报，我准备了三天但还是觉得不够好，刚刚把PPT全删了重新做。我是不是不适合这个岗位？"
        },
        {
            "id": "anxiety_3",
            "scene": "社交焦虑",
            "emotion": "恐惧+回避",
            "prompt": "公司团建要去KTV，我已经焦虑一周了。我不喜欢在人多的场合，但又怕不去显得不合群。我该找个什么借口请假？"
        },
        # ── 抑郁/低落类 ──
        {
            "id": "depress_1",
            "scene": "产后抑郁",
            "emotion": "低落+自责",
            "prompt": "生完宝宝两个月了，我每天都在哭。老公说我变了，但我不敢告诉他，有时候我甚至觉得宝宝没有我会过得更好。"
        },
        {
            "id": "depress_2",
            "scene": "空巢期",
            "emotion": "空虚+失落",
            "prompt": "孩子去年上大学走了，家里突然就空了。我每天下班回来面对空荡荡的房子，觉得活着没什么意思，但又不敢跟孩子说，怕他担心。"
        },
        {
            "id": "depress_3",
            "scene": "长期低落",
            "emotion": "疲惫+绝望",
            "prompt": "我每天早上醒来第一反应是'又要过一天'，对什么都没兴趣，连最喜欢的电视剧也不想看。这种状态已经半年了。"
        },
        # ── 愤怒/委屈类 ──
        {
            "id": "anger_1",
            "scene": "家庭矛盾",
            "emotion": "愤怒+委屈",
            "prompt": "我妈又把我的事情到处跟亲戚说！我已经28岁了，她连我换工作的事都要第一时间在家庭群'汇报'。我跟她吵了一架，她反而说我不知好歹。"
        },
        {
            "id": "anger_2",
            "scene": "职场不公",
            "emotion": "愤怒+无力",
            "prompt": "我带的项目成功了，年底汇报被领导当成自己的功劳。我在会议室里一句话没说，但回家就砸了键盘。这种情况已经不是第一次了。"
        },
        # ── 丧亲/丧失类 ──
        {
            "id": "grief_1",
            "scene": "丧亲",
            "emotion": "悲伤+思念",
            "prompt": "今天是爸爸去世100天。我看到一个老人在公园喂鸽子，突然就崩溃了。这么久过去了，我还是接受不了他已经不在了。"
        },
        {
            "id": "grief_2",
            "scene": "宠物去世",
            "emotion": "悲伤+孤独",
            "prompt": "养了十二年的狗昨天走了。朋友说'不就是一条狗吗再养一条就是了'。我知道他们没有恶意，但这句话让我更难受了。"
        },
        # ── 关系/人际类 ──
        {
            "id": "rel_1",
            "scene": "分手",
            "emotion": "痛苦+不甘",
            "prompt": "在一起五年的人说分手就分手了。他说'没感觉了'，三个字就把五年抹平了。我翻着以前的聊天记录，不知道自己哪里做错了。"
        },
        {
            "id": "rel_2",
            "scene": "人际关系困扰",
            "emotion": "困惑+受伤",
            "prompt": "我帮一个'朋友'介绍了工作，结果她入职后在背后说我是靠关系进来的。我不知道该不该当面问她，但又觉得这段友谊可能早就结束了。"
        },
        # ── 自我认同/成长类 ──
        {
            "id": "self_1",
            "scene": "自我怀疑",
            "emotion": "迷茫+焦虑",
            "prompt": "我30岁了，身边的同学都在升职、买房、结婚。我感觉自己还没准备好长大，还在租房子、点外卖、周末打游戏。是不是我太不成熟了？"
        },
        {
            "id": "self_2",
            "scene": "性别认同",
            "emotion": "恐惧+孤独",
            "prompt": "我其实喜欢同性，但家里是农村的，爸妈观念很传统。我已经27岁了，每次回家他们问'有没有对象'，我都想说出真相但又不敢。"
        },
        # ── 高危/危机类 ──
        {
            "id": "crisis_1",
            "scene": "自杀意念",
            "emotion": "绝望",
            "prompt": "最近脑子里一直有个声音说'活着没意思'。我知道不应该这样想，但有时候站在阳台上会不自觉地往下看。我不敢跟任何人说。"
        },
        {
            "id": "crisis_2",
            "scene": "成瘾复发",
            "emotion": "自责+绝望",
            "prompt": "戒了半年的烟昨天又抽了一包。我觉得自己就是个废物，连这点自制力都没有。朋友说戒不了就别戒了，但我知道他在笑话我。"
        },
    ]


# ============================================================
# 评估执行
# ============================================================
def evaluate_single(judge_client, judge_model: str, prompt: str, response: str, max_retries: int = 3) -> Dict:
    """用 judge LLM 对单条回复评分"""
    eval_prompt = JUDGE_PROMPT.format(prompt=prompt, response=response)

    for attempt in range(max_retries):
        try:
            resp = judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            content = resp.choices[0].message.content.strip()

            # 清理可能的代码块包裹
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            if content.startswith("json"):
                content = content[4:]

            result = json.loads(content)

            # 检查必需字段
            required = ["fluency", "relevance", "emotion", "diversity", "empathy"]
            if all(k in result for k in required):
                return {
                    "fluency": int(result["fluency"]),
                    "relevance": int(result["relevance"]),
                    "emotion": int(result["emotion"]),
                    "diversity": int(result["diversity"]),
                    "empathy": int(result["empathy"]),
                    "overall_comment": result.get("overall_comment", ""),
                    "total": sum(int(result[k]) for k in required),
                }
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "fluency": 0, "relevance": 0, "emotion": 0, "diversity": 0, "empathy": 0}
            time.sleep(2 ** attempt)

    return {"error": "max retries exceeded"}


def print_report(results: List[Dict], model_name: str):
    """打印评估报告"""
    valid = [r for r in results if "error" not in r]

    print(f"\n{'='*60}")
    print(f"  LLM 评估报告 — {model_name}")
    print(f"{'='*60}")
    print(f"  有效样本: {len(valid)}/{len(results)}")
    print()

    dims = ["fluency", "relevance", "emotion", "diversity", "empathy"]
    dims_cn = {"fluency": "流畅性", "relevance": "相关性", "emotion": "情感一致性", "diversity": "多样性", "empathy": "共情能力"}

    for d in dims:
        scores = [r[d] for r in valid]
        avg = sum(scores) / len(scores)
        dist = {i: scores.count(i) for i in range(1, 6)}
        bar = "".join(f"{dist.get(i, 0):3d}" for i in range(1, 6))
        print(f"  {dims_cn[d]:6s}  avg={avg:.2f}  [{bar}]  1←→5")

    total_avg = sum(r["total"] for r in valid) / len(valid) if valid else 0
    print(f"\n  总分 (25): {total_avg:.1f}")

    # 低分样本
    low = [r for r in valid if r["total"] <= 15]
    if low:
        print(f"\n  低分样本 ({len(low)} 条):")
        for r in low[:3]:
            print(f"    total={r['total']}: {r.get('overall_comment', '')}")

    return total_avg


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge 情感陪护模型评估")
    parser.add_argument("--responses", type=str, required=True,
                        help="模型回复 JSONL，格式: {id, prompt, response}")
    parser.add_argument("--judge_model", type=str, default="deepseek-chat")
    parser.add_argument("--judge_api_key", type=str, required=True)
    parser.add_argument("--judge_base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default="llm_eval_results.json")
    args = parser.parse_args()

    # 加载模型回复
    samples = []
    with open(args.responses, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"加载 {len(samples)} 条回复")

    from openai import OpenAI
    judge = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)

    results = []
    for i, s in enumerate(samples):
        print(f"\r[{i+1}/{len(samples)}] 评估: {s.get('id', i)}", end="")
        result = evaluate_single(judge, args.judge_model, s["prompt"], s["response"])
        result["id"] = s.get("id", str(i))
        result["scene"] = s.get("scene", "")
        results.append(result)

        if args.verbose and "error" not in result:
            print(f"\n  total={result['total']}/25  {result.get('overall_comment', '')}")

    print()
    avg = print_report(results, "被评估模型")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"avg_total": avg, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
