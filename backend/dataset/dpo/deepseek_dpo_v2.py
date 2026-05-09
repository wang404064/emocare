#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_dpo_v2.py — 高质量 DPO 数据生成脚本（v3：场景×策略矩阵版）

改进点（v3）：
1. 场景×策略适配矩阵：每个场景有 preferred/optional/forbidden 策略，不再随机组合
2. Chosen 要求策略组合正确，不强制 REBT 术语（SFT阶段已覆盖REBT）
3. Rejected 由 LLM 生成5种策略错误类型，一次调用出3条
4. 语义去重 + 多样性控制（用户画像、情绪风格、具体细节）
5. 禁止所有回复以「听到你...」模板开头

使用方法：
  pip install openai tqdm sentence-transformers
  python deepseek_dpo_v2.py --api_key sk-xxxx --num 3000 --output emocare_dpo_v3.jsonl
"""

import os
import re
import json
import time
import random
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# ── DeepSeek API ────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# ── 语义去重 ────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("[警告] sentence-transformers 未安装，语义去重将跳过")
    print("  安装: pip install sentence-transformers")

# ── 18种策略体系（来自 dpo-场景策略矩阵.md v1.0）─────────────────────────
STRATEGIES = [
    "情感验证(EV)",
    "情感确认(ES)",
    "反思性陈述(RS)",
    "正常化(NE)",
    "澄清探索(Cla)",
    "肯定赋能(Aff)",
    "压力管理(SM)",
    "建议选项(SO)",
    "协同规划(CP)",
    "心理教育(PS)",
    "避免评判(AJC)",
    "不同视角(PDP)",
    "行为激活(BA)",
    "悲伤处理(GP)",
    "安全计划(SP)",
    "动机访谈(MI)",
    "聚焦解决(SF)",
    "意象共情(IS)",
]

# ── 场景 × 策略适配矩阵 ─────────────────────────────────────────────────────
SCENE_STRATEGY_MATRIX = {
    "分手或离婚": {
        "preferred": ["情感验证(EV)", "情感确认(ES)", "澄清探索(Cla)", "正常化(NE)"],
        "preferred_order": ["情感验证(EV)", "情感确认(ES)", "澄清探索(Cla)", "正常化(NE)"],
        "optional": ["避免评判(AJC)"],
        "forbidden": ["建议选项(SO)（过早建议）"],
    },
    "丧亲": {
        "preferred": ["情感验证(EV)", "悲伤处理(GP)", "正常化(NE)", "避免评判(AJC)"],
        "preferred_order": ["情感验证(EV)", "悲伤处理(GP)", "正常化(NE)", "避免评判(AJC)"],
        "optional": ["协同规划(CP)"],
        "forbidden": ["建议选项(SO)", "催放下"],
    },
    "创伤后应激障碍": {
        "preferred": ["情感验证(EV)", "心理教育(PS)", "安全计划(SP)", "避免评判(AJC)"],
        "preferred_order": ["情感验证(EV)", "心理教育(PS)", "安全计划(SP)", "避免评判(AJC)"],
        "optional": [],
        "forbidden": ["澄清探索(Cla)（过早暴露创伤）"],
    },
    "网络性暴力受害者康复": {
        "preferred": ["情感验证(EV)", "安全计划(SP)", "心理教育(PS)", "避免评判(AJC)"],
        "preferred_order": ["情感验证(EV)", "安全计划(SP)", "心理教育(PS)", "避免评判(AJC)"],
        "optional": ["正常化(NE)"],
        "forbidden": ["任何评判", "质疑经历"],
    },
    "急性自杀自残危机": {
        "preferred": ["安全计划(SP)", "情感验证(EV)", "心理教育(PS)", "避免评判(AJC)"],
        "preferred_order": ["安全计划(SP)", "情感验证(EV)", "心理教育(PS)", "避免评判(AJC)"],
        "optional": ["意象共情(IS)"],
        "forbidden": ["讲道理", "否定感受", "转移话题", "忽略自杀风险"],
    },
    "药物滥用成瘾": {
        "preferred": ["情感验证(EV)", "动机访谈(MI)", "避免评判(AJC)", "心理教育(PS)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "动机访谈(MI)", "心理教育(PS)"],
        "optional": ["澄清探索(Cla)"],
        "forbidden": ["道德评判（意志力不够）", "说教"],
    },
    "原生家庭情感忽视": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "心理教育(PS)", "意象共情(IS)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "意象共情(IS)", "心理教育(PS)"],
        "optional": ["澄清探索(Cla)"],
        "forbidden": ["催促和解", "合理化（那个年代都这样）"],
    },
    "职场歧视性骚扰": {
        "preferred": ["情感验证(EV)", "心理教育(PS)", "协同规划(CP)", "避免评判(AJC)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "心理教育(PS)", "协同规划(CP)"],
        "optional": ["澄清探索(Cla)"],
        "forbidden": ["质疑经历", "劝退（辞职算了）", "淡化（职场都这样）"],
    },
    "学业霸凌": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "协同规划(CP)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "协同规划(CP)"],
        "optional": ["安全计划(SP)"],
        "forbidden": ["质疑（你想多了吧）", "淡化霸凌"],
    },
    "婚姻危机": {
        "preferred": ["情感验证(EV)", "情感确认(ES)", "澄清探索(Cla)", "协同规划(CP)"],
        "preferred_order": ["情感验证(EV)", "情感确认(ES)", "澄清探索(Cla)", "协同规划(CP)"],
        "optional": ["避免评判(AJC)"],
        "forbidden": ["催促决定", "评判（你应该原谅/离婚）"],
    },
    "临终关怀复杂丧亲": {
        "preferred": ["情感验证(EV)", "心理教育(PS)", "悲伤处理(GP)", "意象共情(IS)"],
        "preferred_order": ["情感验证(EV)", "悲伤处理(GP)", "意象共情(IS)", "心理教育(PS)"],
        "optional": ["澄清探索(Cla)"],
        "forbidden": ["催放下", "否定悲伤"],
    },
    "社交媒体焦虑网络暴力": {
        "preferred": ["情感验证(EV)", "心理教育(PS)", "压力管理(SM)", "澄清探索(Cla)"],
        "preferred_order": ["情感验证(EV)", "澄清探索(Cla)", "心理教育(PS)", "压力管理(SM)"],
        "optional": ["正常化(NE)"],
        "forbidden": ["否定焦虑", "指责（你不该发那个）"],
    },
    "流产创伤": {
        "preferred": ["情感验证(EV)", "情感确认(ES)", "避免评判(AJC)", "正常化(NE)"],
        "preferred_order": ["情感验证(EV)", "情感确认(ES)", "避免评判(AJC)", "正常化(NE)"],
        "optional": ["悲伤处理(GP)"],
        "forbidden": ["否定（没事以后还会有）", "归因（是不是你...）", "催促放下"],
    },
    "人际冲突沟通问题": {
        "preferred": ["反思性陈述(RS)", "澄清探索(Cla)", "情感验证(EV)", "协同规划(CP)"],
        "preferred_order": ["反思性陈述(RS)", "澄清探索(Cla)", "情感验证(EV)", "协同规划(CP)"],
        "optional": [],
        "forbidden": ["直接给建议", "评判对错"],
    },
    "社交焦虑社交恐惧": {
        "preferred": ["情感验证(EV)", "正常化(NE)", "压力管理(SM)", "澄清探索(Cla)"],
        "preferred_order": ["情感验证(EV)", "正常化(NE)", "澄清探索(Cla)", "压力管理(SM)"],
        "optional": ["协同规划(CP)"],
        "forbidden": ["催促社交", "否定焦虑"],
    },
    "家庭矛盾": {
        "preferred": ["情感验证(EV)", "澄清探索(Cla)", "避免评判(AJC)", "协同规划(CP)"],
        "preferred_order": ["情感验证(EV)", "澄清探索(Cla)", "避免评判(AJC)", "协同规划(CP)"],
        "optional": [],
        "forbidden": ["评判", "站队"],
    },
    "为人父母育儿挑战": {
        "preferred": ["情感验证(EV)", "正常化(NE)", "避免评判(AJC)", "澄清探索(Cla)"],
        "preferred_order": ["情感验证(EV)", "正常化(NE)", "避免评判(AJC)", "澄清探索(Cla)"],
        "optional": [],
        "forbidden": ["比较（别人带三个都不累）", "说教", "评判育儿方式"],
    },
    "兄弟姐妹竞争家庭偏心": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "正常化(NE)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "正常化(NE)"],
        "optional": [],
        "forbidden": ["比较兄弟姐妹", "评判父母"],
    },
    "婆媳姻亲冲突": {
        "preferred": ["情感验证(EV)", "澄清探索(Cla)", "避免评判(AJC)", "协同规划(CP)"],
        "preferred_order": ["情感验证(EV)", "澄清探索(Cla)", "避免评判(AJC)", "协同规划(CP)"],
        "optional": [],
        "forbidden": ["站队", "催促和解"],
    },
    "情感虐待忽视原生家庭": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "心理教育(PS)", "安全计划(SP)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "心理教育(PS)", "安全计划(SP)"],
        "optional": ["澄清探索(Cla)"],
        "forbidden": ["催促和解", "合理化（父母也是为你好）"],
    },
    "从性侵犯家暴中康复": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "安全计划(SP)", "心理教育(PS)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "安全计划(SP)", "心理教育(PS)"],
        "optional": ["正常化(NE)", "意象共情(IS)"],
        "forbidden": ["催促和解", "质疑经历", "受害者有罪推论"],
    },
    "工作倦怠": {
        "preferred": ["情感验证(EV)", "压力管理(SM)", "澄清探索(Cla)", "协同规划(CP)"],
        "preferred_order": ["情感验证(EV)", "压力管理(SM)", "澄清探索(Cla)", "协同规划(CP)"],
        "optional": ["正常化(NE)"],
        "forbidden": ["否定感受", "催促振作"],
    },
    "失业求职压力": {
        "preferred": ["情感验证(EV)", "正常化(NE)", "澄清探索(Cla)", "协同规划(CP)"],
        "preferred_order": ["情感验证(EV)", "正常化(NE)", "澄清探索(Cla)", "协同规划(CP)"],
        "optional": ["心理教育(PS)"],
        "forbidden": ["否定", "比较（别人比你更惨）"],
    },
    "学业压力": {
        "preferred": ["澄清探索(Cla)", "压力管理(SM)", "协同规划(CP)", "不同视角(PDP)"],
        "preferred_order": ["澄清探索(Cla)", "压力管理(SM)", "协同规划(CP)", "不同视角(PDP)"],
        "optional": ["情感验证(EV)"],
        "forbidden": ["催促", "否定压力", "说教"],
    },
    "灵性与信仰危机": {
        "preferred": ["澄清探索(Cla)", "避免评判(AJC)", "意象共情(IS)", "情感验证(EV)"],
        "preferred_order": ["澄清探索(Cla)", "避免评判(AJC)", "意象共情(IS)", "情感验证(EV)"],
        "optional": [],
        "forbidden": ["强行灌输信仰", "否定信仰选择"],
    },
    "适应新工作职场新人": {
        "preferred": ["正常化(NE)", "肯定赋能(Aff)", "澄清探索(Cla)", "情感验证(EV)"],
        "preferred_order": ["正常化(NE)", "肯定赋能(Aff)", "澄清探索(Cla)", "情感验证(EV)"],
        "optional": ["协同规划(CP)"],
        "forbidden": ["比较", "否定适应困难"],
    },
    "财务压力": {
        "preferred": ["情感验证(EV)", "澄清探索(Cla)", "协同规划(CP)", "心理教育(PS)"],
        "preferred_order": ["情感验证(EV)", "澄清探索(Cla)", "协同规划(CP)", "心理教育(PS)"],
        "optional": ["压力管理(SM)"],
        "forbidden": ["评判（你不该乱花钱）", "空洞安慰"],
    },
    "寻找生活意义与目标": {
        "preferred": ["澄清探索(Cla)", "不同视角(PDP)", "聚焦解决(SF)", "情感验证(EV)"],
        "preferred_order": ["澄清探索(Cla)", "不同视角(PDP)", "聚焦解决(SF)", "情感验证(EV)"],
        "optional": ["意象共情(IS)"],
        "forbidden": ["空洞鼓励（你要加油）", "否定意义感探索"],
    },
    "职业转型": {
        "preferred": ["澄清探索(Cla)", "协同规划(CP)", "动机访谈(MI)", "情感验证(EV)"],
        "preferred_order": ["情感验证(EV)", "澄清探索(Cla)", "协同规划(CP)", "动机访谈(MI)"],
        "optional": ["肯定赋能(Aff)"],
        "forbidden": ["催促做决定", "否定转型想法"],
    },
    "双相情感障碍": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "心理教育(PS)", "澄清探索(Cla)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "心理教育(PS)", "澄清探索(Cla)"],
        "optional": ["意象共情(IS)"],
        "forbidden": ["道德评判", "否定医疗必要性"],
    },
    "焦虑与恐慌": {
        "preferred": ["压力管理(SM)", "情感验证(EV)", "心理教育(PS)", "澄清探索(Cla)"],
        "preferred_order": ["情感验证(EV)", "压力管理(SM)", "心理教育(PS)", "澄清探索(Cla)"],
        "optional": ["肯定赋能(Aff)"],
        "forbidden": ["否定焦虑", "催促放松（你只要...就好了）"],
    },
    "抑郁情绪": {
        "preferred": ["情感验证(EV)", "正常化(NE)", "澄清探索(Cla)", "行为激活(BA)"],
        "preferred_order": ["情感验证(EV)", "正常化(NE)", "澄清探索(Cla)", "行为激活(BA)"],
        "optional": ["肯定赋能(Aff)"],
        "forbidden": ["催促振作", "否定抑郁感受", "灌鸡汤"],
    },
    "慢性病疼痛管理": {
        "preferred": ["情感验证(EV)", "意象共情(IS)", "澄清探索(Cla)", "行为激活(BA)"],
        "preferred_order": ["情感验证(EV)", "意象共情(IS)", "澄清探索(Cla)", "行为激活(BA)"],
        "optional": ["肯定赋能(Aff)"],
        "forbidden": ["否定疼痛", "质疑（你是不是想多了）"],
    },
    "医疗诊断应对": {
        "preferred": ["澄清探索(Cla)", "情感验证(EV)", "心理教育(PS)", "意象共情(IS)"],
        "preferred_order": ["澄清探索(Cla)", "情感验证(EV)", "心理教育(PS)", "意象共情(IS)"],
        "optional": ["避免评判(AJC)"],
        "forbidden": ["评判（你怎么不早检查）", "否定恐惧"],
    },
    "身体形象饮食失调": {
        "preferred": ["避免评判(AJC)", "情感验证(EV)", "澄清探索(Cla)", "心理教育(PS)"],
        "preferred_order": ["避免评判(AJC)", "情感验证(EV)", "澄清探索(Cla)", "心理教育(PS)"],
        "optional": ["意象共情(IS)"],
        "forbidden": ["建议饮食/减肥", "评判外貌", "说教"],
    },
    "低自尊自我否定": {
        "preferred": ["肯定赋能(Aff)", "情感验证(EV)", "澄清探索(Cla)", "心理教育(PS)"],
        "preferred_order": ["肯定赋能(Aff)", "情感验证(EV)", "澄清探索(Cla)", "心理教育(PS)"],
        "optional": ["意象共情(IS)"],
        "forbidden": ["说教（你要自信）", "鸡汤（你一定行的）"],
    },
    "外貌性别认同困惑": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "肯定赋能(Aff)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "肯定赋能(Aff)"],
        "optional": ["意象共情(IS)"],
        "forbidden": ["质疑", "替用户做决定", "试图纠正"],
    },
    "搬家适应新环境": {
        "preferred": ["正常化(NE)", "肯定赋能(Aff)", "澄清探索(Cla)", "协同规划(CP)"],
        "preferred_order": ["正常化(NE)", "肯定赋能(Aff)", "澄清探索(Cla)", "协同规划(CP)"],
        "optional": ["情感验证(EV)"],
        "forbidden": ["催促适应", "否定适应困难"],
    },
    "LGBTQ+身份认同": {
        "preferred": ["避免评判(AJC)", "情感验证(EV)", "澄清探索(Cla)", "肯定赋能(Aff)"],
        "preferred_order": ["避免评判(AJC)", "情感验证(EV)", "澄清探索(Cla)", "肯定赋能(Aff)"],
        "optional": ["意象共情(IS)"],
        "forbidden": ["评判", "质疑身份", "试图改变"],
    },
    "文化归属感跨文化适应": {
        "preferred": ["澄清探索(Cla)", "情感验证(EV)", "肯定赋能(Aff)", "意象共情(IS)"],
        "preferred_order": ["澄清探索(Cla)", "情感验证(EV)", "肯定赋能(Aff)", "意象共情(IS)"],
        "optional": ["正常化(NE)"],
        "forbidden": ["否定归属感需求", "比较"],
    },
    "人生转变期身份认同危机": {
        "preferred": ["澄清探索(Cla)", "不同视角(PDP)", "聚焦解决(SF)", "情感验证(EV)"],
        "preferred_order": ["澄清探索(Cla)", "不同视角(PDP)", "聚焦解决(SF)", "情感验证(EV)"],
        "optional": ["协同规划(CP)", "肯定赋能(Aff)"],
        "forbidden": ["催促做决定", "否定转变期痛苦"],
    },
    "照顾者支持": {
        "preferred": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "正常化(NE)"],
        "preferred_order": ["情感验证(EV)", "避免评判(AJC)", "澄清探索(Cla)", "正常化(NE)"],
        "optional": ["协同规划(CP)"],
        "forbidden": ["道德评判", "催促牺牲", "否定照顾者感受"],
    },
}

# ── 多样性控制 ──────────────────────────────────────────────────────────────
AGE_GROUPS = [
    "13-17岁青少年", "18-22岁大学生", "23-30岁年轻职场人",
    "31-50岁中年", "51-70岁银发族",
]

GENDERS = ["男性", "女性", "非二元性别"]

CULTURES = ["都市白领", "小城镇居民", "农村背景", "海外华人", "跨文化家庭"]

EMOTION_STYLES = [
    "含蓄内敛型（不直接说感受，用叙述暗示痛苦）",
    "激烈外放型（情绪强烈，用词直接甚至极端）",
    "自责型（把问题全部归咎于自己）",
    "理性压抑型（试图用逻辑解释情绪，明显在强撑）",
    "依赖求助型（直接表达无助，强烈需要他人支持）",
    "回避轻描型（轻描淡写，但字里行间透露痛苦）",
    "躯体化表达型（不说情绪，只说身体不舒服）",
]

# ── 场景 × 具体细节 ─────────────────────────────────────────────────────────
SCENE_DETAILS = {
    "分手或离婚": [
        "和平分手但还在同一个朋友圈，看到对方动态很难受",
        "对方出轨后发现，不知道怎么面对共同朋友",
        "结婚五年决定离婚，双方父母都反对",
        "异地恋分手，怀疑是不是距离的问题还是不爱了",
    ],
    "丧亲": [
        "父亲去世三个月，还是每天醒来不敢相信",
        "母亲走了，自己一直没机会说再见",
        "孩子流产了，周围人却说没事以后还会有",
        "宠物狗陪伴了十二年，上周被车撞死了",
    ],
    "创伤后应激障碍": [
        "经历过地震，现在一感觉到晃动就心跳加速想逃跑",
        "车祸后不敢坐车，哪怕是出租车都不行",
        "被暴力抢劫后，听到陌生男子快步走近就会本能地害怕",
        "经历严重事故后，每次路过事故现场都会闪回",
    ],
    "网络性暴力受害者康复": [
        "被人在网上人肉搜索，照片和个人信息被到处传播",
        "遭遇网络暴力攻击，因为一句话被骂了好几天",
        "私密照片被前伴侣报复性传播，不知道该怎么面对",
        "在社交平台上被恶意P图侮辱，已经不敢打开评论区",
    ],
    "急性自杀自残危机": [
        "已经三天没睡觉了，觉得活着没意思，手里有药",
        "刚刚和家里大吵一架，现在站在天台边缘",
        "用刀片划手臂已经半年了，最近伤口越来越深",
        "发给心理热线说再见，感觉自己撑不下去了",
    ],
    "药物滥用成瘾": [
        "喝酒才能入睡，现在一瓶都不够了",
        "戒毒一年后复发，觉得自己这辈子没救了",
        "游戏每天玩超过12小时，工作已经丢了",
        "止痛药吃多了，现在没有药根本止不了痛",
    ],
    "原生家庭情感忽视": [
        "从小被父母说你怎么这么笨，现在一犯错就自我攻击",
        "父母的爱是有条件的，考好了才给好脸色",
        "童年被情感忽视，有需求时被说你怎么这么不懂事",
        "父母从来不夸我，别人的父母都会鼓励孩子",
    ],
    "职场歧视性骚扰": [
        "老板每次开会都要拍我的肩膀，说一些让人不舒服的话",
        "同事总是把最脏最累的活推给我，因为我是女生",
        "35岁求职，HR直接说我们更想要90后的",
        "被男同事散布谣言说我和老板有关系才升职",
    ],
    "学业霸凌": [
        "班上同学建了个群，专门发我的丑照",
        "被起侮辱性外号已经三年了，老师也不管",
        "小组作业没人愿意和我一组，说我拖后腿",
        "成绩单被同学故意贴在公告栏最显眼的地方",
    ],
    "婚姻危机": [
        "发现老公出轨半年了，为了孩子一直在忍",
        "夫妻生活已经为零，睡在一张床上像陌生人",
        "老婆说她不需要我了，正在找律师谈离婚",
        "结婚十年，发现我们之间除了孩子已经没有话题了",
    ],
    "临终关怀复杂丧亲": [
        "父亲肝癌晚期，医生说最多还有三个月",
        "母亲已经不认识人了，但我还是每天去养老院陪她",
        "丧亲一年后还是每天梦到逝者，醒来说不出话",
        "亲人走了两年了，还是会把他的事当成新闻讲给别人听",
    ],
    "社交媒体焦虑网络暴力": [
        "发了一条朋友圈，两小时没点赞就开始想是不是写错了",
        "刷到别人的精致生活，觉得自己怎么这么失败",
        "被人在微博上骂了上百条回复，现在已经不敢看手机",
        "每次发照片都要修图两小时，发完还在反复检查评论",
    ],
    "流产创伤": [
        "怀孕三个月胎停了，一直责怪自己是不是哪里没注意好",
        "流产一年后还是会在夜里哭醒，觉得被惩罚了",
        "周围人都说没事你还年轻以后还会有，但我真的很想要这个孩子",
        "试管婴儿两次都失败了，开始怀疑自己是不是不配当妈妈",
    ],
    "人际冲突沟通问题": [
        "和最好的朋友因为误会说开了但关系回不去了",
        "同事在领导面前抢了我的项目功劳，我不敢当面撕",
        "和伴侣每天为小事吵架，不知道是不是该分手了",
        "室友生活习惯差异太大，沟通过几次但完全没改",
    ],
    "社交焦虑社交恐惧": [
        "公司团建要自我介绍，提前一周就开始焦虑",
        "在食堂吃饭必须找个角落，怕被人看到一个人吃",
        "电话铃声响起就心跳加速，能不接就不接",
        "聚会后反复回想自己是不是说错话了，整晚睡不着",
    ],
    "家庭矛盾": [
        "婆婆每天来家里指点家务，老公却说我想多了",
        "父母一直催二胎，说一个孩子太孤单，但我真的不想",
        "哥哥弟弟分家产，父母明显偏心小的，我不敢说什么",
        "岳父母从老家搬来一起住，生活习惯冲突不断",
    ],
    "为人父母育儿挑战": [
        "二胎后大宝行为退行（尿床、吮手指），感到愧疚",
        "孩子确诊ADHD，怀疑是不是自己基因问题",
        "单亲妈妈一个人带俩孩子，大的上初中小的上幼儿园",
        "孩子在学校被霸凌，老师说是同学间的玩笑",
    ],
    "兄弟姐妹竞争家庭偏心": [
        "父母把最好的资源都给了弟弟，我只能自己打工攒学费",
        "哥哥一直是家里的骄傲，我永远被拿来和他比较",
        "姐姐结婚时办了盛大婚礼，轮到我父母说简单点就行",
        "家里出钱给弟弟买了房，我买房时却说没有余钱了",
    ],
    "婆媳姻亲冲突": [
        "婆婆每天都要给我带孩子提建议，不接受就甩脸色",
        "老公的姐姐总介入我们的家务事，老公也不敢得罪她",
        "岳父母从老家搬来住，生活习惯差异大到每天吵架",
        "媳妇不愿意叫我爸妈，老公夹在中间很难做人",
    ],
    "情感虐待忽视原生家庭": [
        "从小被父母冷暴力，做错了事就被完全无视好几天",
        "妈妈总是拿我和别人比，说我哪里都不如别人",
        "父亲脾气暴躁，每次发火我都觉得自己是个废物",
        "父母从来看不到我的努力，只看到我没做到的部分",
    ],
    "从性侵犯家暴中康复": [
        "被前男友家暴，分手两年了还是会在梦里见到他",
        "童年时期被亲戚性侵，现在建立亲密关系非常困难",
        "家暴后离婚了，但前夫还在骚扰，不敢报警怕激怒他",
        "被网友诱骗拍了私密照，现在担心被传播",
    ],
    "工作倦怠": [
        "互联网大厂996，感觉自己就是个工具，没有名字只有工号",
        "护士三班倒，已经对病人没有同理心了，觉得自己很冷血",
        "教师职业倦怠，对学生彻底失去耐心，开始讨厌上课",
        "创业失败欠了50万，白天假装正常上班还债",
    ],
    "失业求职压力": [
        "35岁被优化，投了200份简历全是已读不回",
        "应届生找不到工作，学历还被人质疑是水硕",
        "休息一年后返岗，发现整个行业的技术栈全变了",
        "自由职业收入骤减，这个月房租还差3000",
    ],
    "学业压力": [
        "考研二战，家里给的压力特别大，考不上感觉没脸见人",
        "博士资格考试前彻底崩了，怀疑自己根本不适合搞学术",
        "挂科两门，学费已交，担心拿不到毕业证",
        "语言考试考了三次还是没过，开始怀疑自己语言能力",
    ],
    "灵性与信仰危机": [
        "从小信佛，但经历一次变故后开始怀疑是不是真的有因果",
        "信了十几年基督教，现在突然觉得都是自己骗自己",
        "失去亲人后开始质疑之前所有的信仰，觉得找不到意义",
        "对什么都提不起兴趣，不知道人活着到底是为了什么",
    ],
    "适应新工作职场新人": [
        "入职三个月了，还是觉得自己是个局外人",
        "第一天上班就搞砸了一个小任务，现在见到领导就躲",
        "同事都认识很久了，自己插不进去话",
        "怕自己表现不好过不了试用期，每天失眠",
    ],
    "财务压力": [
        "房贷+车贷+孩子学费，工资刚到账三天就见底了",
        "投资失败，背了30万债，每天被催收电话轰炸",
        "信用卡最低还款都快还不上了，准备去借网贷",
        "父母同时生病，医疗费用没有着落，兄弟还不肯出钱",
    ],
    "寻找生活意义与目标": [
        "每天重复上班下班，不知道这样的意义是什么",
        "孩子上大学走了，突然不知道自己该干什么了",
        "退休后失眠，觉得自己在世界上已经没用了",
        "大学毕业后gap了一年，越来越迷茫不知道该做什么",
    ],
    "职业转型": [
        "想转行做UI设计，但已经30岁了，怕来不及",
        "体制内工作十年，想辞职但不敢，稳定但痛苦",
        "创业失败后不知道该继续还是回去打工",
        "被裁员后不知道该找同行业还是彻底转行",
    ],
    "双相情感障碍": [
        "躁狂期觉得自己无所不能，抑郁期又觉得自己是个废物",
        "因为情绪波动丢了三份工作，开始怀疑自己还能不能正常生活",
        "吃药后体重暴涨20斤，不想吃药但又怕复发",
        "朋友说我情绪不稳定，但我自己觉得没什么不对",
    ],
    "焦虑与恐慌": [
        "第一次惊恐发作，心跳180，以为自己要死了，叫了救护车",
        "广场恐惧症，已经三个月没出过门，外卖都让人放门口",
        "健康焦虑，每次身体有点不舒服就查百度，越查越怕",
        "社交后的反刍，反复想自己是不是说错话了，整晚睡不着",
    ],
    "抑郁情绪": [
        "持续两周起不来床，但对别人说自己挺好的",
        "产后抑郁，不敢告诉家人怕被说矫情",
        "退休后突然失去生活目标，每天坐在沙发上发呆",
        "长期微笑抑郁，周围人都不知道，直到崩溃住院",
    ],
    "慢性病疼痛管理": [
        "腰椎间盘突出压迫神经，走路超过10分钟腿就麻了",
        "偏头痛十几年，最近频率越来越高，怕是自己脑子里长了东西",
        "类风湿，手指关节已经变形，拧瓶盖都困难",
        "慢性胃痛，检查不出器质性问题，但每天隐痛",
    ],
    "医疗诊断应对": [
        "体检发现肺部有结节，等复查的一个月每天都在恐惧中度过",
        "确诊糖尿病，不知道以后生活该怎么安排",
        "医生怀疑是良性肿瘤，等最终结果的那两周度日如年",
        "家族有遗传病史，自己还没症状但每天担心会发病",
    ],
    "身体形象饮食失调": [
        "暴食后催吐，已经持续半年了，牙齿开始被胃酸腐蚀",
        "厌食三个月，月经停了，但照镜子还是觉得自己胖",
        "每次吃东西都算卡路里，吃到超过就崩溃大哭",
        "被男友说你要是再瘦点就好了，之后开始节食",
    ],
    "低自尊自我否定": [
        "每次做完汇报都觉得自己表现很差，虽然别人说很好",
        "被夸的时候第一反应是他们只是客气，不敢接受赞美",
        "总觉得同事在背后议论自己，哪怕没人说过什么",
        "写好的方案不敢提交，总觉得还不够好，改了十几版",
    ],
    "外貌性别认同困惑": [
        "觉得自己穿什么都不对，照镜子就开始嫌弃自己",
        "性别认同困惑，不知道该怎么跟伴侣开口",
        "因为身材被嘲笑后，再也不敢去游泳馆了",
        "整容上瘾，每次整完还是觉得不满意，觉得自己好丑",
    ],
    "搬家适应新环境": [
        "搬到新城市一年了，一个朋友都没交到",
        "最好的朋友搬走了，感觉自己被抛弃了",
        "移民后十年，还是觉得自己是外人",
        "换了新工作，同事都认识很久了，自己插不进去",
    ],
    "LGBTQ+身份认同": [
        "LGBTQ+身份，不敢在家里出柜，每天演戏很累",
        "跨文化家庭中长大，不知道自己到底属于哪里",
        "移民后十年，还是觉得自己是外人",
        "性别认同困惑，不知道该怎么跟伴侣开口",
    ],
    "文化归属感跨文化适应": [
        "从农村考到大城市，和城里同学完全聊不到一起",
        "移民三年了，还是没有归属感，两边都不属于",
        "留学生在国外，每逢节日特别想家但回不去",
        "跨种族恋爱，双方家庭都反对，压力很大",
    ],
    "人生转变期身份认同危机": [
        "退休后不知道自己是谁了，以前的生活只有工作",
        "孩子都上大学走了，家里突然空荡荡，不知道该干嘛",
        "40岁突然被裁员，感觉自己的人生被清零了",
        "大学毕业后在家备考两年，朋友圈越来越小，很孤独",
    ],
    "照顾者支持": [
        "照顾失智症父亲三年，自己身体也垮了，腰痛得直不起来",
        "配偶车祸后半身不遂，一边照顾一边感到愤怒和愧疚",
        "一边照顾重病配偶一边带娃，已经半年没睡过整觉",
        "照顾者内疚：有时希望对方早点解脱，然后又骂自己狠心",
    ],
}


# ── 系统 Prompt（v3：不强制REBT，禁止「听到你」模板）─────────────────────
SYSTEM_PROMPT = """你是一位专业的心理咨询师数据生成专家。

你的任务是为 DPO（Direct Preference Optimization）训练生成高质量数据对。

## 输出格式
每次输出必须是一个严格的 JSON 数组，包含 3 个对象（不要用 Markdown 代码块包裹）。
每个对象包含以下字段：
- "prompt": 用户（来访者的）求助消息
- "chosen": 优质咨询师回复（策略组合正确、共情自然）
- "rejected": 劣质咨询师回复（有策略错误，但看起来像在关心）

## chosen 写作要求（策略正确，不强制REBT）
1. **策略组合正确**：必须自然融入该场景的推荐策略（详见用户消息中的【该场景推荐策略】）
2. **策略顺序严格**：必须按【推荐策略顺序】使用策略，先共情验证，再探索澄清，最后才给建议或规划
3. **完整覆盖所有阶段**：chosen 必须覆盖策略顺序的所有阶段——情感验证→正常化→澄清探索→行动建议。不能停在中间（如只做共情和提问，不给任何实际帮助）
4. **必须包含行动方案**：chosen 的末尾部分必须给出至少一个具体的、可操作的建议或行动方向（如：协同规划、建议选项、心理教育练习、安全计划等）
5. **不强制REBT术语**：不需要显式提到"非理性信念""辩论""理性信念"等REBT术语
6. **开头必须多样化**：禁止以「听到你」「我能感受到」「面对这种情况」「这听起来」开头。这些被过度使用。每次生成3条时，必须使用不同风格的开头，包括但不限于：
   - 直接共情用户的具体处境：如 "熬夜做出来的东西被否定，换谁都会难受。" "照顾这么久还要上班，你真的太累了。"
   - 用身体感受切入：如 "手发抖、胃疼，这些反应都在告诉你，那一刻有多受伤。"
   - 肯定用户的行为或韧性：如 "你改了十几版还在坚持，这份认真本身就值得尊重。"
   - 直接进入正常化：如 "很多人在空巢期都会有这种失落，不是你的问题。"
   - 用沉默/停顿切入：如 "这件事如果说出来很难受，我们可以先停一停。"
   - 直接进入澄清探索：如 "能跟我说说，当时发生了什么吗？"
   **核心原则**：根据prompt的具体内容，用贴近来访者当下的感受开头，不要用模板句式。
7. **语气温暖真诚**，不教条，不列点
8. **长度 150-250 字**（足够覆盖"共情→探索→建议"的完整流程）

## rejected 写作要求（5种策略错误，每次随机选1种）
rejected 必须是「看起来像好回复，但有明确策略错误」的困难负样本：

- **Type A（策略顺序错误）**：使用了正确的策略，但顺序错误。例如：先给建议(SO)再进行情感验证(EV)，或先澄清探索(Cla)再验证情绪。
- **Type B（策略组合错误）**：使用了该场景禁用的策略（forbidden），或使用了完全无关的策略。
- **Type C（过早建议）**：跳过共情和情感验证，直接给建议或解决方案。话术开头就是"你可以..."，没有先建立共情关系。
- **Type D（说教评判）**：表面在共情，但暗含"你也有责任"的暗示，违反避免评判(AJC)原则。例如："我理解你的感受，但也许你也该反思一下..."
- **Type E（空洞通用/情感否定）**：全是正确的废话，没有任何针对性；或用"但是""其实"进行逻辑翻转，让用户的感受显得不合理。

**关键：rejected 长度必须与 chosen 接近（150-250 字），两者长度差控制在 1.5 倍以内。**
不能靠字数差异区分——chosen 可以短但精准，rejected 可以长但错误。区分度必须在内容质量上，不能在长度上。

## prompt 写作要求
- **用户正在跟一个匿名的AI情感支持助手对话**，不是跟某个具体的人（朋友/亲戚/老师等）。prompt中不应出现对助手的称呼（如"张姐""王姨""老师"等）。助手没有名字，是专业的匿名倾听者
- 必须包含用户生活中的具体细节（如：同事的名字、具体数字、时间、地点），让prompt有真实感
- 情绪表达风格按指定类型
- 长度 30-100 字

## 重要提醒
- 每次生成 3 条，分别使用不同的 rejected 错误类型（Type A/B/C/D/E 各选一种，不重复）
- **禁止在 rejected 文本末尾标注错误类型**（不要出现 "Type A" "Type C" "（Type D）" 等字样），错误类型是给你参考的，不能出现在输出中
- 不要生成模板化的回复
- chosen 开头可以多样化：情感验证、澄清问题、正常化表达都可以
- 每条都应有独特细节，避免模板化
"""

USER_PROMPT_TEMPLATE = """【场景】：{scenario}
【该场景推荐策略】：{strategies}
【推荐策略顺序】：{strategy_order}
【禁用策略（不能出现在chosen中）】：{forbidden}
【用户画像】：{age} | {gender} | {culture}
【情绪表达风格】：{emotion_style}
【具体情境细节】：{specific_detail}

请生成 3 条 DPO 数据对（一个 JSON 数组，包含 3 个对象）。
每条使用不同的 rejected 错误类型（Type A/B/C/D/E 各选一种，不重复）。
直接输出 JSON 数组，不要任何额外文字。"""


# ── 语义去重器 ────────────────────────────────────────────────────────────────
class Deduplicator:
    """基于语义相似度的去重器"""
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = None
        self.seen_embeddings: List[np.ndarray] = []
        self.threshold = 0.85
        if HAS_ST:
            print(f"[去重] 加载语义模型: {model_name}")
            self.model = SentenceTransformer(model_name)

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        return self.model.encode(text, convert_to_numpy=True)

    def is_duplicate(self, prompt: str, chosen: str, rejected: str) -> bool:
        """判断是否为语义重复（与已有任何样本相似度 > threshold）"""
        if self.model is None:
            return False

        text = prompt + " " + chosen
        emb = self._get_embedding(text)
        if emb is None:
            return False

        for seen_emb in self.seen_embeddings:
            sim = float(util.cos_sim(emb, seen_emb)[0][0])
            if sim > self.threshold:
                return True
        return False

    def add(self, prompt: str, chosen: str):
        if self.model is None:
            return
        text = prompt + " " + chosen
        emb = self._get_embedding(text)
        if emb is not None:
            self.seen_embeddings.append(emb)


# ── API 调用 ─────────────────────────────────────────────────────────────────
def create_client(api_key: str) -> OpenAI:
    if not HAS_OPENAI:
        raise ImportError("请安装 openai: pip install openai")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def generate_batch(
    client: OpenAI,
    model: str,
    scenario: str,
    strategies: str,
    strategy_order: str,  # 新增参数
    forbidden: str,
    emotion_style: str,
    age: str,
    gender: str,
    culture: str,
    specific_detail: str,
    max_retries: int = 3,
) -> List[Dict]:
    """一次调用生成 3 条数据（提高 API 效率）"""
    user_msg = USER_PROMPT_TEMPLATE.format(
        scenario=scenario,
        strategies=strategies,
        strategy_order=strategy_order,  # 新增：传递给LLM
        forbidden=forbidden,
        emotion_style=emotion_style,
        age=age,
        gender=gender,
        culture=culture,
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
                temperature=0.85,
                top_p=0.95,
                max_tokens=1500,
            )
            content = resp.choices[0].message.content.strip()

            # 去掉可能的代码块包裹
            if "```" in content:
                parts = content.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"):
                        p = p[4:].strip()
                    if p.startswith("["):
                        content = p
                        break
                else:
                    content = parts[-2].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()

            # 找到 JSON 数组边界
            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError(f"未找到 JSON 数组，内容前200字: {content[:200]}")
            content = content[start:end]

            data_list = json.loads(content)
            if not isinstance(data_list, list):
                raise ValueError("返回的不是 JSON 数组")
            if len(data_list) == 0:
                raise ValueError("返回了空数组")

            # 验证每条数据的字段
            results = []
            for item in data_list:
                if not all(k in item for k in ("prompt", "chosen", "rejected")):
                    continue
                results.append({
                    "prompt": item["prompt"].strip(),
                    "chosen": item["chosen"].strip(),
                    "rejected": item["rejected"].strip(),
                })
            return results

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [失败] {scenario}: {e}")
                return []
            time.sleep(2 ** attempt)

    return []


# ── 主流程 ──────────────────────────────────────────────────────────────────
def build_task_queue(num: int) -> List[Tuple]:
    """构建生成任务队列，按场景×策略矩阵分配策略"""
    all_scenes = list(SCENE_STRATEGY_MATRIX.keys())
    tasks = []
    n_tasks = (num + 2) // 3  # 每个任务生成3条

    for _ in range(n_tasks):
        # 1. 随机选择场景（每个场景权重相同，但危机场景可加权）
        scene = random.choice(all_scenes)
        matrix = SCENE_STRATEGY_MATRIX[scene]

        # 2. 从 preferred 中随机选1-2个策略，optional 中选0-1个
        preferred = matrix["preferred"]
        optional = matrix["optional"]
        preferred_order = matrix.get("preferred_order", preferred)  # 新增：获取推荐顺序

        n_pref = min(random.choice([1, 2]), len(preferred))
        chosen_pref = random.sample(preferred, n_pref)
        n_opt = random.choice([0, 1]) if optional else 0
        chosen_opt = random.sample(optional, n_opt) if n_opt > 0 else []

        strategy_str = " + ".join(chosen_pref + chosen_opt)
        strategy_order_str = " → ".join(preferred_order)  # 新增：顺序用箭头连接
        forbidden_str = "；".join(matrix.get("forbidden", []))

        # 3. 随机选择用户画像和具体细节
        detail = random.choice(SCENE_DETAILS.get(scene, ["具体的个人困扰细节"]))

        tasks.append((
            scene,
            strategy_str,
            strategy_order_str,  # 新增：传递给LLM
            forbidden_str,
            random.choice(EMOTION_STYLES),
            random.choice(AGE_GROUPS),
            random.choice(GENDERS),
            random.choice(CULTURES),
            detail,
        ))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="用 DeepSeek API 生成高质量 DPO 数据（场景×策略矩阵版）")
    parser.add_argument("--api_key", type=str, default="",
                        help="DeepSeek API Key（也可通过环境变量 DEEPSEEK_API_KEY 设置）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="DeepSeek 模型名")
    parser.add_argument("--output", type=str, default="emocare_dpo_v3.jsonl")
    parser.add_argument("--num", type=int, default=3000,
                        help="生成总数（会被向上取整到 3 的倍数）")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="API 调用间隔（秒）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_dedup", action="store_true",
                        help="跳过语义去重（加快速度，但可能有重复）")
    args = parser.parse_args()

    # ── API Key ──
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("请提供 DeepSeek API Key：")
        print("  方法1: --api_key sk-xxxx")
        print("  方法2: 设置环境变量 DEEPSEEK_API_KEY")
        return
    os.environ["DEEPSEEK_API_KEY"] = api_key

    random.seed(args.seed)

    client = create_client(api_key)

    # ── 初始化去重器 ──
    deduplicator = None if args.no_dedup else Deduplicator()

    # ── 构建任务 ──
    tasks = build_task_queue(args.num)
    print(f"目标生成: {args.num} 条（{len(tasks)} 个任务，每个任务 3 条）")
    print(f"模型: {args.model} | 输出: {args.output}")
    print(f"语义去重: {'关闭' if args.no_dedup else '开启（阈值 0.85）'}")
    print("-" * 60)

    # ── 断点续传：加载已有数据（用于去重判断）──
    existing = 0
    if os.path.exists(args.output) and deduplicator:
        print("加载已有数据用于去重...")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    deduplicator.add(data["prompt"], data["chosen"])
                    existing += 1
                except Exception:
                    pass
        print(f"  已有 {existing} 条已加载到去重器")

    # ── 生成主循环 ──
    pbar = tqdm(total=args.num, initial=existing, desc="生成 DPO 数据")
    success = existing
    failed_tasks = 0
    duplicate_count = 0

    with open(args.output, "a", encoding="utf-8") as fout:
        for task_idx, (scene, strategies, strategy_order, forbidden, emotion_style, age, gender, culture, detail) in enumerate(tasks):
            batch = generate_batch(
                client, args.model,
                scene, strategies, strategy_order, forbidden, emotion_style, age, gender, culture, detail
            )

            if not batch:
                failed_tasks += 1
                time.sleep(args.delay)
                continue

            # ── 去重过滤 + 写入 ──
            written_this_batch = 0
            for item in batch:
                if success >= args.num:
                    break

                # 过滤：prompt 不应以人名称呼助手
                prompt_text = item.get("prompt", "")
                if re.match(r'^[一-鿿]{1,3}(姐|姨|叔|哥|老师|先生|师傅|阿姨)\W', prompt_text):
                    continue

                # 过滤：chosen 长度检查（至少100字才能覆盖完整策略链）
                chosen_text = item.get("chosen", "")
                rejected_text = item.get("rejected", "")
                if len(chosen_text) < 100 or len(rejected_text) < 100:
                    continue

                # 语义去重检查
                if deduplicator and deduplicator.is_duplicate(
                    item["prompt"], item["chosen"], item["rejected"]
                ):
                    duplicate_count += 1
                    continue

                # 写入
                out = {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "meta": {
                        "scene": scene,
                        "strategies": strategies,
                        "strategy_order": strategy_order,  # 新增：记录策略顺序
                        "forbidden": forbidden,
                        "age": age,
                        "gender": gender,
                        "culture": culture,
                        "emotion_style": emotion_style,
                        "model": args.model,
                    }
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()

                if deduplicator:
                    deduplicator.add(item["prompt"], item["chosen"])
                success += 1
                written_this_batch += 1
                pbar.update(1)

            if written_this_batch == 0:
                failed_tasks += 1

            time.sleep(args.delay)

    pbar.close()
    print(f"\n[OK] 生成完成！")
    print(f"  成功: {success} 条")
    print(f"  失败任务: {failed_tasks}")
    print(f"  语义去重过滤: {duplicate_count} 条")
    print(f"  输出: {args.output}")


if __name__ == "__main__":
    main()
