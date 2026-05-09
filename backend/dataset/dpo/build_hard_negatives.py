#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_hard_negatives.py — EmoCare DPO 困难负样本生成器

现状诊断：
  当前 emocare_dpo.jsonl 中 4792 条数据的 rejected 全部是 Type C（空洞通用型）：
  "一切都会好起来的，保持乐观心态..." — 与 chosen 的语义距离过大，
  导致 DPO loss 在训练初期就饱和为 0，模型无法从困难样本中持续学习。

目标：
  为现有 prompt+chosen 对，生成 4 种困难负样本，让 Embedding Cosine Similarity
  落在 0.60-0.85 的硬负样本区间（"看起来都像好回答，但策略逻辑有本质区别"）。

四种困难负样本类型（按难度从低到高）：
  Type A - 过早建议（Premature Advice）
    内容正确，但完全跳过情感验证/共情，直接给实用建议。
    模型常见毛病。Embedding 相似度较高（内容相关）但策略时序错误。

  Type B - 说教评判（Moralizing/Judging）
    语气看似理性客观，但隐含道德评判，违反 AJC 原则。
    比 Type A 更难区分——表面"关心"，实则把问题归因到用户。

  Type D - 正常化过度（Over-normalization）
    把情绪轻描淡写（"这很正常，大家都这样"），或跳过共情直接正常化，
    让用户感到被忽视而非被理解。

  Type E - 情感否定（Emotional Invalidation）
    用逻辑否定感受（"但你想想..."，"其实没那么糟"），
    表面在帮助，实则切断了情感连接。

用法：
  # 仅用规则模板生成（不消耗 API，速度快，优先用于初步扩充）
  python build_hard_negatives.py --mode rule --input emocare_dpo.jsonl --output hard_neg_rule.jsonl

  # 用 LLM 生成高质量困难负样本（消耗 API，适合精细样本）
  python build_hard_negatives.py --mode llm --input emocare_dpo.jsonl --output hard_neg_llm.jsonl --n 500

  # 混合模式（LLM 生成 + 规则补充，推荐）
  python build_hard_negatives.py --mode mix --input emocare_dpo.jsonl --output hard_neg_mix.jsonl --llm_n 1000

  # 合并到主训练集
  python build_hard_negatives.py --mode merge \
    --input emocare_dpo.jsonl \
    --hard_neg hard_neg_mix.jsonl \
    --output emocare_dpo_v2.jsonl \
    --ratio 0.4
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass


# ============================================================
# 场景 → 策略映射（用于生成场景相关的困难负样本）
# ============================================================
SCENE_STRATEGIES = {
    # 场景名关键词 → (rejected 中应该出现的"错误"策略描述)
    "分手": ("你应该快点振作起来，多出去走走", "过早建议"),
    "离婚": ("你要往前看，不要总沉浸在过去", "过早建议"),
    "失业": ("你应该好好反思一下，更新简历，扩大求职范围", "过早建议"),
    "工作": ("你要主动沟通，调整心态，这样压力就会小很多", "过早建议"),
    "考试": ("你要制定复习计划，分散学习，保证睡眠", "过早建议"),
    "学业": ("你应该找到适合自己的学习方法，劳逸结合", "过早建议"),
    "节食": ("你应该咨询营养师，均衡饮食比节食更健康", "过早建议"),
    "减肥": ("你要循序渐进，节食不是长久之计，建议规律运动", "过早建议"),
    "腰疼": ("你应该去看理疗师，同时注意坐姿和运动习惯", "过早建议"),
    "疼痛": ("你要积极配合治疗，保持乐观，多和医生沟通", "过早建议"),
    "焦虑": ("你要学会深呼吸和冥想，这些技巧对焦虑很有效", "过早建议"),
    "抑郁": ("你要试着每天做一件小事，比如出门散步，积累正向体验", "过早建议"),
    "孤独": ("你可以尝试加入兴趣群体，主动认识新朋友", "过早建议"),
    "亲人": ("你要学会放下，逝者已逝，你需要照顾好自己", "过早建议"),
    "宠物": ("你应该让自己忙碌起来，时间会帮助你走出来的", "过早建议"),
    "家庭": ("你们可以约好时间坐下来好好谈谈，沟通是解决问题的关键", "过早建议"),
    "父母": ("你要理解他们的出发点，尝试换位思考", "说教评判"),
    "孩子": ("你需要反思一下自己的教育方式，孩子的问题往往也是家长的问题", "说教评判"),
    "转行": ("你要先想清楚自己真正想要什么，冲动换工作风险很大", "说教评判"),
    "创业": ("创业风险很高，你要理性评估自己的资金和能力，不要盲目行动", "说教评判"),
    "PTSD": ("这都过去了，你要试着放下，不要总想着过去的事情", "情感否定"),
    "暴力": ("已经过去了，你现在安全了，要往前看", "情感否定"),
    "性侵": ("这不是你的错，但你要学会原谅，带着恨意只会伤害自己", "情感否定"),
}

# ============================================================
# 四种类型的规则模板
# ============================================================

# Type A: 过早建议 - 跳过共情直接给建议
TYPE_A_TEMPLATES = [
    "你应该{suggestion}，这样情况会好很多的。",
    "我建议你{suggestion}，试试看，应该会有帮助。",
    "最有效的方法是{suggestion}。你不妨从今天就开始。",
    "其实你只需要{suggestion}，很多人这样做了之后都好多了。",
    "{suggestion}，这是我能给你的最好建议。",
    "说实话，{suggestion}，比一直纠结要有用多了。",
    "你可以试试{suggestion}，同时{suggestion2}，这两个方法效果都不错。",
    "从实际角度来看，{suggestion}是最合理的选择。",
]

# Type B: 说教评判 - 隐含道德评判
TYPE_B_TEMPLATES = [
    "我理解你的感受，但你是不是也应该反思一下{blame_hint}？",
    "这种情况确实不容易，不过{blame_hint}，可能也是原因之一。",
    "你的心情可以理解，但{blame_hint}，你觉得有没有道理？",
    "有时候{blame_hint}，所以才会有这些困扰，你想过这个问题吗？",
    "虽然很难，但你要承认{blame_hint}，这样才能真正解决问题。",
    "我不是批评你，只是觉得{blame_hint}，这点很值得思考。",
    "说句实话，{blame_hint}，但我说这些是希望你能好起来。",
]

# Type D: 正常化过度 - 轻描淡写，跳过个性化共情
TYPE_D_TEMPLATES = [
    "这种感觉很正常，每个人都会经历这些，没什么特别的。",
    "其实大家都有类似的问题，你不是一个人，慢慢就过去了。",
    "这很常见，你不用太在意，时间久了自然就好了。",
    "这种情况太普遍了，身边很多人都是这样过来的，你也会的。",
    "嗯，这些都是正常现象，没必要太纠结，放平心态就好。",
    "说实话，这种事人人都会遇到，你的情况并不特别，都会过去的。",
    "你的情绪很正常，这是普遍现象，大家都能挺过去，你也一样。",
]

# Type E: 情感否定 - 用逻辑否定感受
TYPE_E_TEMPLATES = [
    "我知道你现在难受，但其实{reframe}，你再想想是不是这样？",
    "你的感受可以理解，不过{reframe}，从这个角度看是不是好一些？",
    "虽然你觉得{negative_restate}，但{reframe}，换个角度可能就不一样了。",
    "我觉得你可能想多了，{reframe}，事情没你想的那么糟。",
    "说实话，{reframe}，你现在的感受更多是情绪在放大问题。",
    "其实{reframe}，你好好想想，是不是并没有那么严重？",
    "我理解你难过，但{reframe}，理性来看这件事其实还好。",
]


def get_scene_suggestion(prompt: str) -> Tuple[str, str]:
    """根据 prompt 内容提取场景关键词，返回(建议内容, rejected_type)"""
    for keyword, (suggestion, rtype) in SCENE_STRATEGIES.items():
        if keyword in prompt:
            return suggestion, rtype
    # 默认
    return "调整一下心态，给自己制定一个小目标", "过早建议"


def generate_type_a(prompt: str, chosen: str) -> str:
    """Type A: 过早建议 - 跳过共情直接给建议，但内容看起来关心且合理"""
    suggestion, _ = get_scene_suggestion(prompt)
    
    generic_suggestions = [
        "保持规律的作息时间", "多和信任的朋友倾诉", "尝试每天运动散心",
        "把感受记录在日记里", "给自己设定一个小目标"
    ]
    suggestion2 = random.choice([s for s in generic_suggestions if s[:4] not in suggestion[:4]])
    
    # 生成包含2-3条建议、看起来关心但完全跳过情感验证的回复
    intros = [
        "针对你的情况，我有几点建议：",
        "我觉得你可以从以下几个方面来改善：",
        "处理这种情况，有几个比较有效的方法：",
        "我了解了一下，建议你这样做：",
        "从实际角度来说，你可以尝试：",
    ]
    
    closings = [
        "这些方法都经过验证，你可以从最容易的开始尝试。",
        "很多人用了这些方法之后都有明显改善，希望对你也有帮助。",
        "只要坚持做，情况一定会好转的，加油！",
        "这些都是实用的方法，试试看吧，相信你可以的。",
    ]
    
    intro = random.choice(intros)
    closing = random.choice(closings)
    
    return f"{intro}首先，{suggestion}；其次，{suggestion2}。{closing}"


def generate_type_b(prompt: str, chosen: str) -> str:
    """Type B: 说教评判 - 表面理性，暗含道德评判，语气像在"分析问题"但实则归因于用户"""
    # 根据场景构造责怪短语
    if "孩子" in prompt or "育儿" in prompt or "带娃" in prompt:
        blame_hints = [
            "孩子的问题往往折射出家长的情绪管理方式，你是不是也要审视一下自己的反应模式？",
            "作为父母，我们的情绪和教育方式对孩子影响很大，你有没有想过自己的处理方式是否也需要调整？",
            "带孩子确实辛苦，但有时候我们对孩子的期待或者自身的焦虑情绪也会加重局面，你觉得呢？",
        ]
    elif "工作" in prompt or "职场" in prompt or "领导" in prompt or "同事" in prompt:
        blame_hints = [
            "职场关系出现问题，有时候也要从自身角度分析，自己的沟通方式或者心态是不是也有值得改进的地方？",
            "当然，工作压力很大，但我们自己的抗压能力和情绪管理方式，是不是也需要加强？",
            "遇到这种情况，除了外部环境，你有没有想过自己在这件事上是否也有一些可以改变的地方？",
        ]
    elif "感情" in prompt or "分手" in prompt or "男友" in prompt or "女友" in prompt or "恋爱" in prompt:
        blame_hints = [
            "一段感情出了问题，双方都有责任，你有没有反思过自己在这段关系里的一些行为或者期待方式？",
            "当然他/她有不对的地方，但感情是两个人的事，你自己的沟通方式或者依赖程度是不是也值得想一想？",
            "感情的问题往往是双方共同造成的，你有没有思考过，自己在这段关系里是否也有一些需要成长的地方？",
        ]
    elif "父母" in prompt or "家人" in prompt or "妈妈" in prompt or "爸爸" in prompt:
        blame_hints = [
            "和家人的矛盾需要双向努力，你有没有试着站在他们的角度想想，他们为什么会这样？",
            "理解父母不容易，但他们那个年代有自己的局限，你是不是也需要多一些包容和理解？",
            "家庭关系的修复需要主动的一方先迈出一步，你有没有想过，你自己是否也可以做些什么来改善？",
        ]
    elif "减肥" in prompt or "节食" in prompt or "身材" in prompt or "体重" in prompt:
        blame_hints = [
            "减肥困难，除了方法的问题，意志力和生活习惯的建立也很关键，你觉得自己在这方面有没有需要改进的地方？",
            "你说控制不住想吃东西，这其实是一个信号——你有没有想过，是不是情绪性进食的问题需要先面对？",
            "身体管理是一场持久战，你的方法可能没问题，但你有没有认真审视过自己日常的作息和生活习惯？",
        ]
    elif "焦虑" in prompt or "压力" in prompt or "紧张" in prompt:
        blame_hints = [
            "焦虑感有时候和我们自己的思维模式有关，你有没有想过，自己是不是有一些完美主义倾向或者过度担忧的习惯？",
            "压力是客观存在的，但我们对待压力的方式更重要，你觉得自己的心理建设是不是也需要加强？",
            "焦虑往往来自于我们对结果的控制欲，你有没有想过放下一些执念，允许事情有多种可能？",
        ]
    else:
        blame_hints = [
            "面对这样的情况，除了外部的因素，我们自身的应对方式和心态调整也很重要，你有没有认真想过这个问题？",
            "遇到困难的时候，往往也是我们成长的机会，你有没有从这件事里看到一些自己需要改变的地方？",
            "我理解你很难受，但有时候困境也是一面镜子，你有没有思考过，自己在这件事里扮演了什么角色？",
        ]
    
    blame = random.choice(blame_hints)
    
    # 包裹在"关心"的语气中
    wrappers = [
        f"我听到你说的了，你现在确实很辛苦。不过，{blame}",
        f"你现在的感受我能理解。但说实话，{blame}",
        f"我知道你现在很难，这种压力确实不小。{blame}",
        f"我想认真和你聊聊这个问题。你当然有你的难处，但{blame}",
    ]
    
    return random.choice(wrappers)


def generate_type_d(prompt: str, chosen: str) -> str:
    """Type D: 正常化过度 - 用"大家都这样"来替代真正的个性化共情"""
    
    # 根据场景提供相关的"正常化"语境
    if "分手" in prompt or "感情" in prompt or "失恋" in prompt:
        specific = "每个人在感情里都受过伤，这是成长必须经历的一课。"
    elif "考试" in prompt or "学业" in prompt or "学习" in prompt:
        specific = "学生都会经历考试压力，这是学习阶段很普遍的感受。"
    elif "失业" in prompt or "工作" in prompt or "职场" in prompt:
        specific = "职场压力是现代人都会遇到的问题，大家都在想方设法应对。"
    elif "亲人" in prompt or "去世" in prompt or "离开" in prompt or "宠物" in prompt:
        specific = "失去重要的人或事物，每个人都会经历，这是生命的一部分。"
    elif "焦虑" in prompt or "压力" in prompt or "紧张" in prompt:
        specific = "现代生活节奏快，焦虑和压力是大家都会有的感受。"
    elif "减肥" in prompt or "身材" in prompt or "体重" in prompt:
        specific = "对身材的不满意几乎是每个人都有的困扰，你不是一个人在经历这些。"
    elif "家庭" in prompt or "父母" in prompt:
        specific = "家庭矛盾在每个家庭里都有，这非常普遍。"
    else:
        specific = "这种情况很多人都经历过，并不罕见。"
    
    # 加上"过去了就好了"的收尾，完全没有个性化关注
    templates = [
        f"你说的这些感受其实很正常。{specific}大多数人都会经历这样的阶段，随着时间推移，情绪慢慢就会稳定下来的，你不要太担心。",
        f"听你说完，我觉得你这种情况挺普遍的。{specific}身边很多人都是这样过来的，时间是最好的良药，你也一定能走过去的。",
        f"这种感受是完全正常的，你不用太在意。{specific}给自己一些时间，慢慢就会好起来的，你只是暂时处于一个低谷期而已。",
        f"我理解你的感受。{specific}这是人生中很常见的波折，大家都能挺过去，你肯定也可以的，放宽心。",
    ]
    
    return random.choice(templates)


def generate_type_e(prompt: str, chosen: str) -> str:
    """Type E: 情感否定 - 表面先认可，然后用"但是/其实"逻辑翻转感受"""
    
    # 根据场景构造"其实没那么糟"的逻辑翻转
    if "崩溃" in prompt or "撑不下去" in prompt or "快不行" in prompt:
        acknowledge = "你现在感觉压力很大"
        reframe = "但其实你还在正常运转，说明你的承受能力比你想象的要强。人在压力下总会觉得自己快到极限，但往往都能扛过去的。你可能只是暂时被情绪淹没了，理性看来并没有到真正的临界点。"
    elif "失败" in prompt or "无用" in prompt or "没用" in prompt or "一无是处" in prompt:
        acknowledge = "你觉得自己很失败"
        reframe = "但客观来说，你还在努力、还在寻求帮助，这本身就说明你并没有放弃。失败的感受往往是情绪化的，并不代表真实情况。换个角度，你其实已经做得不错了，只是标准定得太高了而已。"
    elif "害怕" in prompt or "恐惧" in prompt or "不安" in prompt:
        acknowledge = "你感到害怕和不安"
        reframe = "但仔细想想，你现在客观上其实是安全的。这种恐惧感更多来自于你的想象和对未来的担忧，而不是真实的威胁。大脑在保护你，但它有时候会过度反应，让你把风险放大了。"
    elif "孤独" in prompt or "没人" in prompt or "没有人" in prompt:
        acknowledge = "你感到孤独和被忽视"
        reframe = "但你有没有想过，很多时候是我们自己把自己孤立起来了？身边其实可能有很多关心你的人，只是你现在的状态让你更容易关注到负面的部分，忽略了那些积极的连接。"
    elif "没意思" in prompt or "空虚" in prompt or "意义" in prompt:
        acknowledge = "你觉得生活没有意义"
        reframe = "但这往往是因为我们还没有找到真正适合自己的方向，而不是真的没有意义。很多人在某个阶段都会有这样的感受，它只是一个信号，告诉你需要做些改变了，而不是真正的绝望。"
    elif "难受" in prompt or "痛苦" in prompt or "伤心" in prompt or "悲伤" in prompt:
        acknowledge = "你现在很难受"
        reframe = "但情绪是会变化的，今天感觉很痛苦，不代表明天还是这样。我们在情绪低落的时候往往很难客观判断，事情实际上可能并没有你感受到的那么糟糕。给自己一点时间，你会发现情绪会自然慢慢平复的。"
    else:
        acknowledge = "你现在的感受确实不好"
        reframe = "但如果理性来看这个情况，你会发现很多担忧其实是可以解决的，并没有你感受到的那么绝望。情绪有时候会放大问题，让我们看不清楚真实的全貌。"
    
    openers = [
        f"我听到你说的了，{acknowledge}，这是真实的。不过，{reframe}",
        f"你现在的感受我理解，{acknowledge}，这很正常。但你有没有想过，{reframe}",
        f"嗯，{acknowledge}，我不否认这一点。只是……{reframe}",
        f"我明白，{acknowledge}。但换个角度来想一想——{reframe}",
    ]
    
    return random.choice(openers)


# ============================================================
# LLM 生成器
# ============================================================

LLM_SYSTEM_PROMPT = """你是一位专业的心理咨询数据标注专家，负责为 DPO 训练数据集生成高质量的"困难负样本"（Hard Negative）。

## 任务
给定一个用户的心理求助消息（prompt）和一个优质咨询师回复（chosen），你需要生成一个"看起来很像好回答，但在咨询策略上有本质错误"的劣质回复（rejected）。

## 核心要求
- 长度：rejected 应与 chosen 相当（不能太短，太短显然是差的）
- 语气：正式、关心的语气，不能是明显的错误或乱码
- 错误：错误必须是策略性的（时机/方式/态度），不是内容错误
- 相似度：embedding cosine similarity 应在 0.60-0.85 之间（"硬样本区间"）

## 四种类型说明

**Type A - 过早建议（Premature Advice）**
直接跳过情感验证和共情，上来就给具体的实用建议。
核心错误：内容正确，但"时序"完全错了——用户还没被理解，就被推向解决方案。
写法要点：体现关心，给出2-3个合理建议，但完全没有先承认/确认用户的感受。

**Type B - 说教评判（Moralizing）**
回复的表面是理性分析，但隐含着"你也有责任"或"你这样做不对"的道德评判。
核心错误：违反 AJC（Avoid Judgment and Criticism）原则。
写法要点：用"客观分析"的语气，暗示用户在某些方面需要反思，但包裹在"为你好"的措辞里。

**Type D - 正常化过度（Over-normalization）**
快速把用户的痛苦定性为"很正常的事"，轻描淡写，然后说"都会过去的"。
核心错误：正常化之前没有充分确认个体的独特痛苦，让用户感到"被当成了一个普通案例"。
写法要点：先说"大家都会这样"，然后讲一些普遍性的经验，不针对具体情况。

**Type E - 情感否定（Emotional Invalidation）**
用逻辑来"纠正"情感（"但其实..."，"你可能想多了"），让用户的感受显得不理性。
核心错误：切断情感连接，让用户感到自己的情绪是"不合理"的。
写法要点：先承认感受，然后用"但是/其实/换个角度"来翻转，暗示情绪本身有问题。

## 输出格式（严格遵守）
直接输出 rejected 回复的纯文本，不要加任何解释、标签或引号。"""


LLM_USER_TEMPLATE = """用户消息（prompt）：
{prompt}

优质咨询师回复（chosen）：
{chosen}

请生成一个 {type_name} 类型的困难负样本（rejected），要求：
1. 长度与 chosen 相当（不能太短）
2. 语气自然、表面关心
3. 错误是策略性的，不是内容错误
4. 嵌入相似度应在 0.60-0.85 之间

直接输出 rejected 回复，不要加任何说明："""

TYPE_NAMES = {
    "A": "Type A（过早建议型）",
    "B": "Type B（说教评判型）",
    "D": "Type D（正常化过度型）",
    "E": "Type E（情感否定型）",
}


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = os.getenv("DPO_GEN_MODEL", "qwen-max")
        
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def generate(self, prompt_text: str, system: str, max_retries: int = 3) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=0.85,
                    max_tokens=512,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [ERROR] LLM 生成失败: {e}")
                    return None
                wait = 2 ** attempt
                time.sleep(wait)
        return None


# ============================================================
# 规则模式：批量生成
# ============================================================

def generate_rule_based(
    samples: List[Dict],
    types: List[str] = ["A", "B", "D", "E"],
    samples_per_type: int = 1,
    scene_balance: bool = True,
) -> List[Dict]:
    """
    为每个 prompt+chosen 对，用规则生成困难负样本。
    
    Args:
        samples: 原始 DPO 样本列表
        types: 要生成的类型列表
        samples_per_type: 每种类型生成几条（1 = 随机选一种；>1 = 每种都生成）
        scene_balance: 是否按场景均衡采样
    
    Returns:
        新的 DPO 样本列表（每条样本包含一个困难负样本）
    """
    
    type_generators = {
        "A": generate_type_a,
        "B": generate_type_b,
        "D": generate_type_d,
        "E": generate_type_e,
    }
    
    results = []
    type_counts = defaultdict(int)
    
    for sample in tqdm(samples, desc="规则生成困难负样本"):
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        meta = sample.get("meta", {})
        
        if not prompt or not chosen:
            continue
        
        for t in types:
            if t not in type_generators:
                continue
            
            gen_func = type_generators[t]
            rejected = gen_func(prompt, chosen)
            
            # 避免生成的 rejected 和原来的 "一切都会好起来" 重复
            if rejected == sample.get("rejected", ""):
                continue
            
            new_sample = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "rejected_type": f"HARD_TYPE_{t}",
                "meta": {
                    **meta,
                    "hard_negative": True,
                    "generation_method": "rule",
                }
            }
            results.append(new_sample)
            type_counts[t] += 1
    
    print(f"\n规则生成完成：")
    for t, count in sorted(type_counts.items()):
        print(f"  Type {t}: {count} 条")
    print(f"  总计: {len(results)} 条")
    
    return results


# ============================================================
# LLM 模式：高质量生成
# ============================================================

def generate_llm_based(
    samples: List[Dict],
    llm_client: LLMClient,
    n: int = 500,
    types: List[str] = ["A", "B", "D", "E"],
    delay: float = 0.5,
) -> List[Dict]:
    """
    用 LLM 为精选的 prompt 生成高质量困难负样本。
    
    策略：
    - 按场景均衡采样，避免某些场景过度拟合
    - 优先处理 chosen 较长（咨询质量好）的样本
    - 每个 prompt 生成 1-2 种类型（避免过度增强单一 prompt）
    """
    
    # 按场景分组，均衡采样
    by_scene = defaultdict(list)
    for s in samples:
        scene_id = s.get("meta", {}).get("scene_id", "unknown")
        by_scene[scene_id].append(s)
    
    # 每个场景均衡取样
    n_per_scene = max(1, n // max(len(by_scene), 1))
    selected = []
    for scene_id, scene_samples in by_scene.items():
        # 优先选 chosen 较长（质量好）的样本
        sorted_samples = sorted(scene_samples, key=lambda x: len(x.get("chosen", "")), reverse=True)
        selected.extend(sorted_samples[:n_per_scene])
    
    # 截断到目标数量
    random.shuffle(selected)
    selected = selected[:n]
    
    results = []
    type_counts = defaultdict(int)
    
    # 每个 prompt 随机分配 1-2 种类型
    for sample in tqdm(selected, desc="LLM 生成困难负样本"):
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        meta = sample.get("meta", {})
        
        if not prompt or not chosen:
            continue
        
        # 随机选 1-2 种类型
        n_types = random.choices([1, 2], weights=[0.6, 0.4])[0]
        selected_types = random.sample(types, min(n_types, len(types)))
        
        for t in selected_types:
            user_content = LLM_USER_TEMPLATE.format(
                prompt=prompt,
                chosen=chosen,
                type_name=TYPE_NAMES[t],
            )
            
            rejected = llm_client.generate(user_content, LLM_SYSTEM_PROMPT)
            
            if not rejected or len(rejected) < 20:
                continue
            
            # 质量检查：不能和 chosen 开头一样
            if rejected[:10] == chosen[:10]:
                continue
            
            new_sample = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "rejected_type": f"HARD_TYPE_{t}",
                "meta": {
                    **meta,
                    "hard_negative": True,
                    "generation_method": "llm",
                }
            }
            results.append(new_sample)
            type_counts[t] += 1
            
            if delay > 0:
                time.sleep(delay)
    
    print(f"\nLLM 生成完成：")
    for t, count in sorted(type_counts.items()):
        print(f"  Type {t}: {count} 条")
    print(f"  总计: {len(results)} 条")
    
    return results


# ============================================================
# 合并：原始数据 + 困难负样本 → 最终训练集
# ============================================================

def merge_datasets(
    original: List[Dict],
    hard_negatives: List[Dict],
    ratio: float = 0.3,
    shuffle: bool = True,
) -> List[Dict]:
    """
    将原始数据和困难负样本按比例合并。
    
    Args:
        original: 原始 DPO 样本（Type C 空洞通用型）
        hard_negatives: 困难负样本
        ratio: 困难负样本占总数的比例，基于原始数据量计算
               ratio=0.3 表示：困难样本数 = 原始样本数 * 0.3 / (1-0.3)
               即最终 困难样本/(困难+原始) = ratio
        shuffle: 是否打乱
    
    Returns:
        合并后的数据集
    """
    # 保持原始数据全量，计算需要多少困难样本
    n_original = len(original)
    # ratio = n_hard / (n_original + n_hard) → n_hard = n_original * ratio / (1 - ratio)
    n_hard_target = int(n_original * ratio / (1 - ratio))
    
    # 随机采样困难负样本（打乱后取前 n 个，保证类型均衡）
    shuffled_hard = list(hard_negatives)
    random.shuffle(shuffled_hard)
    hard_selected = shuffled_hard[:n_hard_target]
    
    merged = list(original) + hard_selected
    
    if shuffle:
        random.shuffle(merged)
    
    actual_ratio = len(hard_selected) / len(merged)
    
    print(f"\n合并结果：")
    print(f"  原始样本 (Type C): {n_original} 条")
    print(f"  困难负样本 (Type A/B/D/E): {len(hard_selected)} 条")
    print(f"  困难样本占比: {actual_ratio*100:.1f}%")
    print(f"  总计: {len(merged)} 条")
    
    return merged


# ============================================================
# 质量评估（可选，需要 sentence-transformers）
# ============================================================

def compute_margin_distribution(samples: List[Dict], sample_n: int = 200) -> Dict:
    """计算样本的 embedding cosine similarity 分布，评估困难程度"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("[WARN] 未安装 sentence-transformers，跳过 margin 计算")
        print("  安装方法: pip install sentence-transformers")
        return {}
    
    print(f"\n计算 margin 分布（抽样 {sample_n} 条）...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    test_samples = random.sample(samples, min(sample_n, len(samples)))
    
    margins = []
    for s in tqdm(test_samples, desc="计算 embedding"):
        chosen = s.get("chosen", "")
        rejected = s.get("rejected", "")
        if not chosen or not rejected:
            continue
        
        embs = model.encode([chosen, rejected], normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        margins.append(sim)
    
    if not margins:
        return {}
    
    margins_arr = np.array(margins)
    result = {
        "count": len(margins),
        "mean": float(margins_arr.mean()),
        "std": float(margins_arr.std()),
        "min": float(margins_arr.min()),
        "max": float(margins_arr.max()),
        "hard_ratio": float((margins_arr >= 0.6).mean()),      # 0.6-1.0 硬样本
        "optimal_ratio": float(((margins_arr >= 0.6) & (margins_arr <= 0.85)).mean()),  # 最优区间
        "too_easy_ratio": float((margins_arr < 0.4).mean()),    # < 0.4 太简单
        "too_hard_ratio": float((margins_arr > 0.95).mean()),   # > 0.95 太难
    }
    
    print(f"  均值: {result['mean']:.3f} ± {result['std']:.3f}")
    print(f"  最优区间(0.6-0.85): {result['optimal_ratio']*100:.1f}%")
    print(f"  硬样本(0.6-1.0): {result['hard_ratio']*100:.1f}%")
    print(f"  太简单(<0.4): {result['too_easy_ratio']*100:.1f}%")
    
    return result


# ============================================================
# 主程序
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def save_jsonl(samples: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"已保存 {len(samples)} 条到 {path}")


def main():
    parser = argparse.ArgumentParser(description="EmoCare DPO 困难负样本生成器")
    parser.add_argument("--mode", choices=["rule", "llm", "mix", "merge", "eval"], default="rule",
                        help="运行模式: rule=规则生成, llm=LLM生成, mix=混合, merge=合并, eval=评估")
    parser.add_argument("--input", default="emocare_dpo.jsonl", help="输入文件路径")
    parser.add_argument("--output", default="hard_negatives.jsonl", help="输出文件路径")
    parser.add_argument("--hard_neg", default="hard_negatives.jsonl", help="困难负样本文件（merge模式用）")
    parser.add_argument("--types", nargs="+", default=["A", "B", "D", "E"],
                        choices=["A", "B", "D", "E"], help="要生成的类型")
    parser.add_argument("--n", type=int, default=500, help="LLM模式生成总数")
    parser.add_argument("--llm_n", type=int, default=1000, help="混合模式中LLM生成数量")
    parser.add_argument("--ratio", type=float, default=0.4, help="困难样本在最终训练集中的占比")
    parser.add_argument("--delay", type=float, default=0.3, help="LLM调用间隔（秒）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--eval_n", type=int, default=200, help="评估时的抽样数量")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 加载输入数据
    if args.mode not in ["merge"]:
        print(f"加载数据: {args.input}")
        samples = load_jsonl(args.input)
        print(f"  共 {len(samples)} 条样本")
    
    if args.mode == "rule":
        # ── 规则模式 ──
        hard_negs = generate_rule_based(samples, types=args.types)
        save_jsonl(hard_negs, args.output)
        
    elif args.mode == "llm":
        # ── LLM 模式 ──
        llm = LLMClient()
        hard_negs = generate_llm_based(samples, llm, n=args.n, types=args.types, delay=args.delay)
        save_jsonl(hard_negs, args.output)
        
    elif args.mode == "mix":
        # ── 混合模式 ──
        # 先用规则快速生成全量
        print("=== 第一阶段：规则生成 ===")
        rule_negs = generate_rule_based(samples, types=args.types)
        
        # 再用 LLM 生成高质量子集
        print(f"\n=== 第二阶段：LLM 生成 {args.llm_n} 条 ===")
        try:
            llm = LLMClient()
            llm_negs = generate_llm_based(samples, llm, n=args.llm_n, types=args.types, delay=args.delay)
        except Exception as e:
            print(f"[WARN] LLM 初始化失败: {e}，仅使用规则生成结果")
            llm_negs = []
        
        # 合并：LLM 结果优先，规则补充
        all_hard = llm_negs + rule_negs
        
        # 去重（按 prompt+rejected 去重）
        seen = set()
        deduped = []
        for s in all_hard:
            key = (s["prompt"], s["rejected"][:50])
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        
        save_jsonl(deduped, args.output)
        print(f"\n混合模式完成: LLM {len(llm_negs)} + 规则 {len(rule_negs)} → 去重后 {len(deduped)} 条")
        
    elif args.mode == "merge":
        # ── 合并模式 ──
        print(f"加载原始数据: {args.input}")
        original = load_jsonl(args.input)
        print(f"  原始样本: {len(original)} 条")
        
        print(f"加载困难负样本: {args.hard_neg}")
        hard_negs = load_jsonl(args.hard_neg)
        print(f"  困难负样本: {len(hard_negs)} 条")
        
        merged = merge_datasets(original, hard_negs, ratio=args.ratio)
        save_jsonl(merged, args.output)
        
    elif args.mode == "eval":
        # ── 评估模式 ──
        print(f"评估数据集质量: {args.input}")
        result = compute_margin_distribution(samples, sample_n=args.eval_n)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n完成。")


if __name__ == "__main__":
    main()
