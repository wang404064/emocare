# 情绪识别器推理脚本
import torch
from transformers import AutoTokenizer
from model import EmotionRiskClassifier
from utils import postprocess_prediction

# === 配置 ===
model_path = "emotion_risk_model_v1"  # 替换为你保存的模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载模型和 tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = EmotionRiskClassifier.from_pretrained(model_path)
model.to(device)
model.eval()

# 情绪标签顺序（与训练一致）
EMOTION_ORDER = [
    "sadness", "anxiety", "anger", "loneliness",
    "shame_guilt", "hopelessness", "hope", "calm", "joy"
]

# === 测试用例：覆盖多种情绪和风险等级 ===
test_cases = [
    # 高风险 + 绝望
    "我太孤独了，我想死",
    "活着没有任何意义，我已经写好遗书了",
    "我真的撑不下去了，想结束这一切",

    # 中风险 + 焦虑/悲伤
    "最近总是睡不着，感觉很焦虑",
    "工作压力太大了，每天都想哭",
    "我觉得没人理解我，好难过",

    # 低风险 + 积极情绪
    "今天阳光真好，心情很愉快",
    "终于完成项目了，感觉很有希望！",
    "和朋友聊完天，心里平静多了",

    # 愤怒主导
    "我气死了，这不公平！",
    "凭什么要这样对我？我受够了！",

    # 孤独但未达高危
    "一个人在家，有点孤单",
    "周末没人约，感觉被世界遗忘",

    # 内疚/羞耻
    "都是我的错，我不该那样做",
    "我让家人失望了，真没脸见人",

    # 希望 vs 绝望对比
    "虽然很难，但我相信明天会更好",
    "一切都完了，我永远好不起来了",

    # 打招呼
    "你好",
    "你是谁呀？"
]

print("=" * 60)
print("🧠 情绪-风险多任务模型推理演示")
print("=" * 60)

for i, text in enumerate(test_cases, 1):
    print(f"\n【{i}】输入: \"{text}\"")

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        result = postprocess_prediction(
            outputs["emotion_logits"][0].cpu(),
            outputs["risk_logits"][0].cpu()
        )

    # 打印情绪强度（格式化为字典）
    emotion_dict = dict(zip(EMOTION_ORDER, result["emotion_intensities"]))
    print("  情绪强度:")
    for emo, score in emotion_dict.items():
        print(f"    {emo:15}: {score:.3f}")

    # 打印风险等级
    risk_level_str = ["low", "medium", "high"][result["risk_level"]]
    print(f"  风险等级: {risk_level_str} ({result['risk_level']})")

    # 可选：打印风险概率分布
    risk_probs = result["risk_probs"]
    print(f"  风险概率: low={risk_probs[0]:.2f}, medium={risk_probs[1]:.2f}, high={risk_probs[2]:.2f}")

print("\n" + "=" * 60)
print("✅ 推理完成！")