#!/usr/bin/env python3
"""
对评估数据集批量生成模型回复。

用法:
  # 方式1: 用 LLaMA-Factory 本地模型
  llamafactory-cli chat --model_name_or_path /path/to/model \
    --adapter_name_or_path /path/to/adapter --template qwen3_nothink

  # 方式2: 用 OpenAI 兼容 API
  python generate_responses.py \
    --eval_data eval_dataset.jsonl \
    --api_url http://localhost:8000/v1 \
    --model_name emocare \
    --output model_responses.jsonl
"""
import json
import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", default="eval_dataset.jsonl")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="not-needed")
    parser.add_argument("--model_name", default="default")
    parser.add_argument("--output", default="model_responses.jsonl")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    with open(args.eval_data, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    with open(args.output, "w", encoding="utf-8") as fout:
        for i, s in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {s['id']}")

            resp = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": s["prompt"]}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            response_text = resp.choices[0].message.content

            fout.write(json.dumps({
                "id": s["id"],
                "scene": s.get("scene", ""),
                "prompt": s["prompt"],
                "response": response_text,
            }, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"完成: {args.output}")


if __name__ == "__main__":
    main()
