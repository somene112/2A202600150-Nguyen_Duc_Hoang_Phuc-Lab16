"""
Download 50 câu từ HotpotQA validation set và convert sang định dạng lab.

Chạy:
    python scripts/download_hotpot.py
    python scripts/download_hotpot.py --num 100 --out data/hotpot_100.json
"""
from __future__ import annotations
import json
import random
import argparse
from pathlib import Path


def download_and_convert(num: int = 50, out: str = "data/hotpot_100.json") -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Cài đặt trước: pip install datasets")

    print(f"Đang tải HotpotQA (distractor, validation) từ HuggingFace...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    print(f"Dataset gốc: {len(ds)} câu hỏi")

    items = []
    for ex in ds:
        # HotpotQA context schema: {"title": [str,...], "sentences": [[str,...], ...]}
        context_chunks = []
        titles = ex["context"]["title"]
        sentences_list = ex["context"]["sentences"]
        for title, sents in zip(titles, sentences_list):
            context_chunks.append({
                "title": title,
                "text": " ".join(sents),
            })

        level = ex.get("level", "medium")
        if level not in ("easy", "medium", "hard"):
            level = "medium"

        items.append({
            "qid": ex["id"],
            "difficulty": level,
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "context": context_chunks,
        })

    # Chia đều theo độ khó
    random.seed(42)
    by_diff: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for item in items:
        by_diff[item["difficulty"]].append(item)
    for v in by_diff.values():
        random.shuffle(v)

    per_bucket = num // 3
    selected = (
        by_diff["easy"][:per_bucket]
        + by_diff["medium"][:per_bucket]
        + by_diff["hard"][:num - 2 * per_bucket]
    )
    # Nếu không đủ hard, bù từ medium
    if len(selected) < num:
        extra = num - len(selected)
        selected += by_diff["medium"][per_bucket: per_bucket + extra]
    selected = selected[:num]
    random.shuffle(selected)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu {len(selected)} câu vào {out_path}")

    counts = {}
    for item in selected:
        counts[item["difficulty"]] = counts.get(item["difficulty"], 0) + 1
    print(f"Phân bố: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--out", type=str, default="data/hotpot_100.json")
    args = parser.parse_args()
    download_and_convert(num=args.num, out=args.out)
