# -*- coding: utf-8 -*-
"""B0 采样池构建脚本（词典 B 全自动构建 · 第一步）

按《docs/词典B全自动构建方案.md》§2 执行：
  1. 从 data/raw/TOXICN/train.json 分层抽样（隐式/显式/转述/冒犯/非毒）；
  2. 长度过滤（>= min_len 字）+ 全等去重 + bge-m3 嵌入近重复去重（cosine > dedup_sim 只留一条）；
  3. 配额缺额如实记录（不许放宽过滤凑数）；
  4. 产出 data/concepts/sample_pool.json（样本 id + 层标签 + 原文），供 B1 与 B5 复用。

约束与约定：
  - 只用本地模型：嵌入模型默认 models/bge-m3，可用 --model_path 指定任意本地路径；
  - 只用数据集真实样本，不构造/不改写任何文本；
  - 固定随机种子（默认 729，组内约定），完全可复现；
  - 去重的跨层优先级 = 层处理顺序（隐式 > 显式 > 转述 > 冒犯 > 非毒），后处理的层让位。

运行（用户执行，本脚本不由 AI 运行）：
  conda activate toxicnew
  python scripts/build_b0_sample_pool.py
  python scripts/build_b0_sample_pool.py --model_path D:/tcbm/models/bge-m3 --seed 729
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Windows 控制台 GBK 兜底：统一按 UTF-8 输出，避免中文打印报错
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

# 层定义：顺序即跨层去重的优先级（先处理的层保留，后处理的层让位）
STRATA = [
    # (key, 中文名, 判定函数)
    ("implicit",    "隐式仇恨", lambda r: r["toxic_type"] == 2 and r["expression"] == 2),
    ("explicit",    "显式仇恨", lambda r: r["toxic_type"] == 2 and r["expression"] == 1),
    ("paraphrase",  "转述仇恨", lambda r: r["toxic_type"] == 2 and r["expression"] == 3),
    ("offensive",   "冒犯",     lambda r: r["toxic_type"] == 1),
    ("non_toxic",   "非毒",     lambda r: r["toxic"] == 0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B0 采样池构建（TOXICN train 分层抽样 + 去重），产出 sample_pool.json"
    )
    p.add_argument("--data_path", type=str,
                   default=str(ROOT / "data" / "raw" / "TOXICN" / "train.json"),
                   help="TOXICN train json 路径")
    p.add_argument("--model_path", type=str,
                   default=str(ROOT / "models" / "bge-m3"),
                   help="本地嵌入模型路径（bge-m3），禁止远程下载")
    p.add_argument("--output_dir", type=str, default=str(ROOT / "data" / "concepts"),
                   help="输出目录（sample_pool.json 落在这里）")
    p.add_argument("--seed", type=int, default=729, help="全局随机种子（组内约定 729）")
    # 各层配额（对应《词典B全自动构建方案.md》§2 表格）
    p.add_argument("--quota_implicit", type=int, default=600)
    p.add_argument("--quota_explicit", type=int, default=400)
    p.add_argument("--quota_paraphrase", type=int, default=200)
    p.add_argument("--quota_offensive", type=int, default=200)
    p.add_argument("--quota_non_toxic", type=int, default=400)
    p.add_argument("--min_len", type=int, default=10, help="最短文本长度（去首尾空白后的字符数）")
    p.add_argument("--dedup_sim", type=float, default=0.95,
                   help="bge-m3 cosine 相似度超过该值视为近重复（> 阈值才删）")
    p.add_argument("--over_provision", type=float, default=2.0,
                   help="每层超采样倍数（为去重损耗预留候选，配额 600 实际编码约 1200 条）")
    p.add_argument("--batch_size", type=int, default=64, help="嵌入编码 batch 大小")
    p.add_argument("--max_length", type=int, default=512, help="tokenizer 截断长度")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def load_train(path: Path) -> list[dict]:
    """加载 TOXICN train.json 并做字段校验。"""
    if not path.exists():
        sys.exit(f"[B0][错误] 数据文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        sys.exit(f"[B0][错误] 数据格式异常（期望非空 list）: {path}")
    required = {"content", "toxic", "toxic_type", "expression"}
    missing = required - set(data[0].keys())
    if missing:
        sys.exit(f"[B0][错误] 数据缺少字段: {missing}")
    return data


def stratify(data: list[dict]) -> dict[str, list[int]]:
    """按层过滤行索引。

    五层判定互斥且完整覆盖本数据集（已核验：tt=0→exp=0，tt=1→exp=0，tt=2→exp∈{1,2,3}）。
    任何不匹配任何层的行（含假想的 tt=2 且 exp∉{1,2,3} 异常值）都会落入末尾警告并被跳过。
    """
    buckets: dict[str, list[int]] = {}
    for idx, row in enumerate(data):
        matched = False
        for key, _name, cond in STRATA:
            if cond(row):
                buckets.setdefault(key, []).append(idx)
                matched = True
                break
        if not matched:
            print(f"[B0][警告] 第 {idx} 条样本不属于任何层，已跳过: "
                  f"toxic={row.get('toxic')}, toxic_type={row.get('toxic_type')}, "
                  f"expression={row.get('expression')}")
    return buckets


def build_candidates(
    data: list[dict],
    idx_list: list[int],
    quota: int,
    min_len: int,
    over_provision: float,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """层内候选生成：长度过滤 → 全等去重 → 超采样（shuffled 顺序即优先级）。

    返回 (候选 idx 列表, 过滤后层内池大小 n_pool)——n_pool 用于缺额归因
    （区分"原始样本不足"与"过滤/去重损耗"）。
    """
    # 1) 长度过滤 + 层内全等去重（同文只留文件序靠前者，确定性）
    seen_text: set[str] = set()
    pool: list[int] = []
    for idx in idx_list:  # 文件序遍历，保证确定性
        text = str(data[idx]["content"]).strip()
        if len(text) < min_len:
            continue
        if text in seen_text:
            continue
        seen_text.add(text)
        pool.append(idx)

    # 2) 超采样：n_cand = min(可用数, ceil(quota * over_provision))
    n_cand = min(len(pool), math.ceil(quota * over_provision))
    if n_cand < len(pool):
        perm = rng.permutation(len(pool))[:n_cand]
        candidates = [pool[i] for i in perm]
    else:
        candidates = pool
    return candidates, len(pool)


class BGEM3Encoder:
    """bge-m3 dense 嵌入：transformers AutoModel + CLS 池化 + L2 归一。

    与 FlagEmbedding 的 BGEM3FlagModel dense_vecs 等价（该库本机未安装，
    用 transformers 直接加载，行为一致且不引入新依赖）。
    """

    def __init__(self, model_path: str, device: str, max_length: int, batch_size: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        p = Path(model_path)
        if not p.exists():
            sys.exit(
                f"[B0][错误] 嵌入模型路径不存在: {p}\n"
                f"  请将 bge-m3 放入 models/ 目录，或用 --model_path 指定本地路径。"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(str(p), trust_remote_code=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"[B0] 嵌入模型加载完成: {p} | device={device}")

    def encode(self, texts: list[str], desc: str = "") -> np.ndarray:
        """返回 (n, d) float32 的 L2 归一化 dense 向量。

        注意：不用 @torch.no_grad() 装饰器——torch 为延迟导入，类定义时该名字
        尚不存在，装饰器会在模块导入期直接 NameError；改用方法内上下文管理器。
        """
        torch = self.torch
        vecs: list[np.ndarray] = []
        try:
            from tqdm import tqdm
            it = tqdm(range(0, len(texts), self.batch_size), desc=desc, ncols=80)
        except ImportError:  # tqdm 不可用时退化为静默进度
            it = range(0, len(texts), self.batch_size)
        with torch.no_grad():
            for start in it:
                batch = texts[start:start + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                cls = out.last_hidden_state[:, 0]  # CLS 池化
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                vecs.append(cls.float().cpu().numpy())
        emb = np.concatenate(vecs, axis=0).astype(np.float32)
        # 数值安全：再归一一次，消除 float32 累积的微小舍入误差
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return emb / norm


def greedy_dedup_fill(
    cand_idx: list[int],
    cand_emb: np.ndarray,
    kept_emb_mat: np.ndarray | None,
    quota: int,
    dedup_sim: float,
) -> tuple[list[int], np.ndarray | None]:
    """贪心去重并填充配额：按候选优先级逐条检查与全局已保留向量的最大 cosine。

    返回 (本层保留的 idx, 更新后的全局向量矩阵)。近重复判定：sim > dedup_sim → 丢弃。
    """
    kept_idx: list[int] = []
    local_mat: np.ndarray | None = None  # 本层已保留向量（增量维护，避免逐候选 stack）
    for i, idx in enumerate(cand_idx):
        if len(kept_idx) >= quota:
            break
        v = cand_emb[i]
        # 与之前各层已保留样本比
        if kept_emb_mat is not None and kept_emb_mat.shape[0] > 0:
            if float((kept_emb_mat @ v).max()) > dedup_sim:
                continue
        # 与本层已保留样本比
        if local_mat is not None:
            if float((local_mat @ v).max()) > dedup_sim:
                continue
        kept_idx.append(idx)
        local_mat = v[None, :] if local_mat is None else np.vstack([local_mat, v[None, :]])
    if local_mat is not None:
        kept_emb_mat = local_mat if kept_emb_mat is None else np.concatenate(
            [kept_emb_mat, local_mat], axis=0)
    return kept_idx, kept_emb_mat


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print("=" * 60)
    print("[B0] 采样池构建开始")
    print(f"  数据: {args.data_path}")
    print(f"  嵌入模型: {args.model_path}")
    print(f"  种子: {args.seed} | 去重阈值: {args.dedup_sim} | 最短长度: {args.min_len}")

    data = load_train(Path(args.data_path))
    print(f"  train 总量: {len(data)}")

    quotas = {
        "implicit": args.quota_implicit,
        "explicit": args.quota_explicit,
        "paraphrase": args.quota_paraphrase,
        "offensive": args.quota_offensive,
        "non_toxic": args.quota_non_toxic,
    }

    # ---- 分层 ----
    buckets = stratify(data)
    name_of = {k: n for k, n, _ in STRATA}
    for key, _name, _cond in STRATA:
        avail = len(buckets.get(key, []))
        print(f"  层[{name_of[key]}] 可用: {avail} | 配额: {quotas[key]}")
        if avail < quotas[key]:
            print(f"    [B0][注意] 该层可用样本少于配额，将全取并记录缺额")

    # ---- 逐层候选（各层独立子种子，层间顺序变化不影响各自抽样）----
    candidates: dict[str, list[int]] = {}
    pool_sizes: dict[str, int] = {}
    for si, (key, _name, _cond) in enumerate(STRATA):
        rng = np.random.default_rng([args.seed, si])  # SeedSequence 子种子
        candidates[key], pool_sizes[key] = build_candidates(
            data, buckets.get(key, []), quotas[key], args.min_len,
            args.over_provision, rng,
        )
    n_total_cand = sum(len(v) for v in candidates.values())
    print(f"  候选总数（超采样后）: {n_total_cand}")

    # ---- 嵌入 + 贪心去重填配额 ----
    encoder = BGEM3Encoder(args.model_path, args.device, args.max_length, args.batch_size)

    kept_emb_mat: np.ndarray | None = None
    per_stratum_stat: dict[str, dict] = {}
    samples: list[dict] = []
    for key, name, _cond in STRATA:
        cand_idx = candidates[key]
        if not cand_idx:
            per_stratum_stat[key] = {
                "name": name, "quota": quotas[key],
                "available_raw": len(buckets.get(key, [])),
                "pool_after_filter": pool_sizes[key],
                "candidates": 0, "kept": 0, "deficit": quotas[key],
            }
            continue
        texts = [str(data[i]["content"]).strip() for i in cand_idx]
        emb = encoder.encode(texts, desc=f"编码[{name}]")
        kept_idx, kept_emb_mat = greedy_dedup_fill(
            cand_idx, emb, kept_emb_mat, quotas[key], args.dedup_sim,
        )
        for idx in kept_idx:
            text = str(data[idx]["content"]).strip()
            samples.append({
                "sample_id": f"toxicn_train_{idx:05d}",
                "src_idx": idx,
                "stratum": key,
                "stratum_name": name,
                "char_len": len(text),
                "text_md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
                "text": text,
            })
        per_stratum_stat[key] = {
            "name": name,
            "quota": quotas[key],
            "available_raw": len(buckets.get(key, [])),
            "pool_after_filter": pool_sizes[key],
            "candidates": len(cand_idx),
            "kept": len(kept_idx),
            "deficit": quotas[key] - len(kept_idx),
        }
        s = per_stratum_stat[key]
        print(f"  层[{name}] 保留 {s['kept']}/{s['quota']}"
              + (f"（缺额 {s['deficit']}）" if s["deficit"] else ""))

    # ---- 落盘 ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_pool.json"
    payload = {
        "metadata": {
            "stage": "B0",
            "purpose": "词典B构建采样池（供B1微概念抽取与B5增补迭代复用）",
            "data_path": str(Path(args.data_path).resolve()),
            "model_path": str(Path(args.model_path).resolve()),
            "seed": args.seed,
            "min_len": args.min_len,
            "dedup_sim": args.dedup_sim,
            "over_provision": args.over_provision,
            "dedup_priority_order": [k for k, _, _ in STRATA],
            "per_stratum": per_stratum_stat,
            "total_kept": len(samples),
            "elapsed_sec": round(time.time() - t0, 1),
            "note": "只使用 TOXICN train 真实样本；sample_id 由源文件行号生成，稳定可复现",
        },
        "samples": samples,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"[B0] 完成: 保留 {len(samples)} 条 → {out_path}")
    print(f"[B0] 用时 {time.time() - t0:.1f}s")
    deficit_total = sum(s["deficit"] for s in per_stratum_stat.values())
    if deficit_total:
        print(f"[B0][注意] 总缺额 {deficit_total} 条（各层缺额详见 metadata.per_stratum）")


if __name__ == "__main__":
    main()
