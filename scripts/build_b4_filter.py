# -*- coding: utf-8 -*-
"""
build_b4_filter.py — 词典B构建 · B4 六规则过滤
（实施细则：docs/词典B全自动构建方案.md §6；执行清单 §10 第 5 步）

流程（对应方案 §6，全部可量化、自动执行，零人工标注）：
  Step1  读取 B3 产物 B_dict_draft.csv；从 B1 原始表聚合每个宏概念的
         sample_ids（触发样本集）与 text_spans（触发片段）。
  Step2  bge-m3 编码：池内 1800 条文本（F5 探针用）+ 概念名与标签词（F1/F2 用）。
  Step3  F1 标签捷径：概念与任务标签词/类别名（非毒/冒犯/仇恨/显式/隐式/转述
         + 8 类别名）句嵌入相似度 > f1_sim → 删。
  Step4  F2 冗余（余弦 > 0.9）与 F3 触发重合（Jaccard > 0.95）产生候选对，
         逐对走 P6 LLM 仲裁（方案 §6 提示词逐字）；merge=true → 删低频者，
         合并其 sample_ids 与 spans；解析失败按量化规则默认合并并告警。
  Step5  F4 频次：freq_agg < tau_freq(5) → 删（池内 1800 ≈ 全 train 19%，
         池内 5 次 ≈ 全 train 27 次量级）。
  Step6  F5 线性可辨识度（CT-CBM g(c) 判据）：标签 = 该概念 text_span 是否
         出现在池样本中（样本级自动弱标签，非人工标注）；bge-m3 嵌入 +
         线性探针（逻辑回归，class_weight=balanced）。评估用 5 折分层交叉
         验证取均值 F1（单次划分在低正例场景下方差过大，0.000 可能只是
         漏掉验证集里唯一的正例），均值 F1 < tau_g(0.30) → 删；
         正例 < 折数或折内无正例 → 跳过探针保留并标记。
  Step7  F6 近常数：触发率 < 0.1%（近全零）或 > 90%（近全一）→ 删。
  Step8  落盘：B_dict_final.csv + filter_report.md（逐条删除/合并证据，
         含词典级诊断：触发率基尼系数）+ b4_stats.json。

产物：
  data/concepts/B_dict_final.csv
  data/concepts/filter_report.md
  data/concepts/b4_stats.json
  data/processed/TOXICN/{llm}/concepts/p6_merge_cache.jsonl   （断点续跑）

复用：B0 的 BGEM3Encoder、B1 的 LLMBackend/JsonlCache/extract_json/enforce_offline。
依赖：scikit-learn（B2 已用其 HDBSCAN，环境已具备）。

离线硬约束（红线）：同 B0-B3——全程 HF_HUB_OFFLINE=1 / local_files_only=True，
仅从本地路径加载模型，任何缺失只报错、绝不联网。

用法（用户执行，本脚本不由 AI 运行）：
  conda activate toxic_env   # 服务器环境
  python scripts/build_b4_filter.py --model_path /root/autodl-tmp/glm-4-9b-chat \
      --encoder_path /root/autodl-tmp/bge-m3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_b0_sample_pool import BGEM3Encoder          # noqa: E402
from build_b1_micro_concepts import (                  # noqa: E402
    LLMBackend, JsonlCache, enforce_offline, extract_json,
)


# ---------------------------------------------------------------------------
# F1 标签词表（任务标签 + 表达方式 + 8 类别名；可被 configs b4.f1_label_words 覆盖）
# ---------------------------------------------------------------------------

DEFAULT_LABEL_WORDS = [
    # TOXICN toxic_type（§10 数据集表：非毒/冒犯/仇恨）
    "非毒", "冒犯", "仇恨", "仇恨言论",
    # expression 标注
    "显式", "隐式", "转述",
    # 8 个类别名（方案 §5.1）
    "群体指称", "侮辱称谓", "威胁恐吓", "刻板印象与偏见主题",
    "行为与事件", "性与身体", "场景与话题", "其它",
]


def build_p6_messages(a: str, def_a: str, b: str, def_b: str) -> list[dict]:
    """P6 边界仲裁（方案 §6，逐字）。"""
    user = (
        "判断两个概念是否语义重合到\"在任何文本中几乎总是同时适用\"。"
        "只输出 JSON：{\"merge\": true|false, \"reason\": \"≤20字\"}。\n"
        f"概念A：{a}（定义：{def_a}）\n"
        f"概念B：{b}（定义：{def_b}）\n"
        "判定标准：若一条文本涉及A时几乎必然也涉及B（或反之），输出 true。"
    )
    return [{"role": "system",
             "content": "你是社交媒体有害言论研究的概念词典过滤仲裁助手。"
                        "严格按用户规则输出 JSON，不要输出任何其它内容。"},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 参数与配置
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="B4 六规则过滤：F1捷径/F2冗余/F3重合/F4频次/F5探针/F6近常数 → 最终词典",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_path", type=str, required=True,
                    help="本地 LLM 目录（P6 仲裁用），禁止远程下载")
    ap.add_argument("--encoder_path", type=str, default="models/bge-m3",
                    help="本地 bge-m3 目录（F1/F2/F5 嵌入用）")
    ap.add_argument("--backend", type=str, default="vllm",
                    choices=["vllm", "transformers"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--dict_input", type=str,
                    default="data/concepts/B_dict_draft.csv")
    ap.add_argument("--b1_input", type=str,
                    default="data/concepts/micro_concepts_raw.jsonl")
    ap.add_argument("--pool_input", type=str,
                    default="data/concepts/sample_pool.json")
    ap.add_argument("--config", type=str, default="configs/dict_b.yaml")
    ap.add_argument("--out_dir", type=str, default="data/concepts")
    ap.add_argument("--cache_root", type=str, default="data/processed/TOXICN")
    ap.add_argument("--encoder_device", type=str, default="auto")
    ap.add_argument("--encoder_max_length", type=int, default=256,
                    help="池文本较长（≤约200字），256 token 足够")
    ap.add_argument("--encoder_batch_size", type=int, default=64)
    return ap.parse_args()


def load_cfg(path: str) -> dict:
    import yaml
    p = Path(path)
    if not p.is_file():
        print(f"[警告] 配置文件不存在: {p}，全部使用内置默认值")
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_params(args: argparse.Namespace, cfg: dict) -> dict:
    b4 = cfg.get("b4", {}) if isinstance(cfg, dict) else {}
    defaults = {
        # 六规则阈值（方案 §6 表）
        "tau_freq": 5,
        "f1_sim": 0.85,
        "f2_sim": 0.9,
        "f3_jaccard": 0.95,
        "tau_g": 0.30,
        "f6_zero_rate": 0.001,
        "f6_one_rate": 0.9,
        "k_max": 150,
        # F1 标签词
        "f1_label_words": DEFAULT_LABEL_WORDS,
        # F5 探针
        "f5_n_folds": 5,
        "f5_max_iter": 1000,
        # P6 仲裁
        "p6_temperature": 0.2,
        "p6_top_p": 0.9,
        "p6_max_new_tokens": 96,
        "p6_retries": 2,
        "seed": 729,
    }
    for k in list(defaults):
        if k in b4:
            defaults[k] = b4[k]
    if "seed" in cfg:
        defaults["seed"] = cfg["seed"]
    return defaults


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def gini(x) -> float:
    """基尼系数（词典级诊断指标，方案 §6 补充说明）。"""
    x = np.sort(np.asarray(list(x), dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2.0 * (cum / cum[-1]).sum()) / n)


class ConceptState:
    """词典概念的可变状态（F2/F3 合并会迁移 ids/spans/members）。"""

    def __init__(self, row: dict, ids: set, spans: list[str]):
        self.concept: str = row["concept"]
        self.definition: str = row["definition"]
        self.category: str = row["category"]
        self.confidence: str = row["confidence"]
        self.type_majority: str = row["type_majority"]
        self.source: str = row["source"]
        self.members: list[str] = row["members"]
        self.ids: set = set(ids)
        self.spans: list[str] = list(spans)
        self.f5_f1: float | None = None
        self.f5_status: str = ""

    @property
    def freq(self) -> int:
        return len(self.ids)

    @property
    def trigger_rate(self) -> float:
        return self.freq  # 除以池大小在落盘时进行


def resolve(rep_of: dict[str, str], c: str) -> str:
    while rep_of[c] != c:
        c = rep_of[c]
    return c


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    enforce_offline()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_cfg(args.config)
    par = resolve_params(args, cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(par["seed"])
    t0 = time.time()

    # ---- Step1 读取输入 ----
    with open(args.dict_input, encoding="utf-8-sig", newline="") as f:
        draft = list(csv.DictReader(f))
    for r in draft:
        r["freq_agg"] = int(r["freq_agg"])
        r["members"] = [m for m in r["members"].split("|") if m]
    raw_rows = [json.loads(l) for l in open(args.b1_input, encoding="utf-8") if l.strip()]
    raw_index = {r["concept"]: r for r in raw_rows}
    pool = json.load(open(args.pool_input, encoding="utf-8"))
    samples = pool["samples"]
    n_pool = len(samples)
    texts = [s["text"] for s in samples]
    print(f"[B4] 草案概念: {len(draft)} 条 | B1 原始微概念: {len(raw_rows)} 条 | "
          f"池样本: {n_pool} 条")

    states: dict[str, ConceptState] = {}
    missing_ids: list[str] = []
    for r in draft:
        ids: set = set()
        spans: list[str] = []
        for m in r["members"]:
            raw = raw_index.get(m)
            if raw is None:
                continue
            ids.update(raw.get("sample_ids", []))
            for sp in raw.get("text_spans", []):
                sp = str(sp).strip()
                if sp and sp not in spans:
                    spans.append(sp)
        if not ids:
            missing_ids.append(r["concept"])
        states[r["concept"]] = ConceptState(r, ids, spans)
    if missing_ids:
        print(f"[B4][警告] {len(missing_ids)} 条概念在 B1 原始表中无 sample_ids "
              f"（freq_agg 将以 CSV 值兜底）: {missing_ids}")
        for c in missing_ids:
            states[c].ids = {f"__freq_only_{i}" for i in range(states[c].freq)}

    delete_log: list[dict] = []   # {concept, rule, evidence}
    merge_log: list[dict] = []    # {kept, dropped, evidence, reason}

    # ---- 总进度条（唯一进度条：样本编码→概念编码→F1→逐对P6→F4→逐概念F5→F6→落盘）----
    # 初始 total = 3(两步编码+F1) + 1(F4) + 1(F6) + 1(落盘)；P6 对数与 F5 概念数
    # 在流程中已知后追加（各子模块内置进度条已 progress=False 关闭）
    bar = tqdm(total=6, desc="B4总进度", unit="step", ncols=100, miniters=1)

    # ---- Step2 编码（一次加载，同时编样本与概念/标签词）----
    bar.set_postfix_str("bge-m3编码(池文本)")
    encoder = BGEM3Encoder(args.encoder_path, args.encoder_device,
                           args.encoder_max_length, args.encoder_batch_size)
    text_emb = encoder.encode(texts, desc="", progress=False)          # (n_pool, d)
    bar.update(1)

    bar.set_postfix_str("bge-m3编码(概念/标签)")
    label_words = list(par["f1_label_words"])
    name_texts = [s.concept for s in states.values()] + label_words
    name_emb = encoder.encode(name_texts, desc="", progress=False)
    concept_emb = name_emb[:len(states)]
    label_emb = name_emb[len(states):]
    concept_names = [s.concept for s in states.values()]
    # 释放编码器显存，给 vLLM 让路
    encoder.model.to("cpu")
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    bar.update(1)

    keep: set[str] = set(states.keys())

    # ---- Step3 F1 标签捷径 ----
    bar.set_postfix_str("F1标签捷径")
    sim_cl = concept_emb @ label_emb.T          # (n_concept, n_label)
    for i, c in enumerate(concept_names):
        j = int(np.argmax(sim_cl[i]))
        if float(sim_cl[i, j]) > float(par["f1_sim"]):
            delete_log.append({"concept": c, "rule": "F1标签捷径",
                               "evidence": f"与标签词\"{label_words[j]}\""
                                           f"cos={sim_cl[i, j]:.3f}>"
                                           f"{par['f1_sim']}"})
            keep.discard(c)
    bar.update(1)
    print(f"\n[B4] F1 删除 {len([d for d in delete_log if d['rule'].startswith('F1')])} 条")

    # ---- Step4 F2/F3 候选对 + P6 仲裁 ----
    keeper_idx = [i for i, c in enumerate(concept_names) if c in keep]
    sim_cc = concept_emb[keeper_idx] @ concept_emb[keeper_idx].T
    f2_pairs = []
    for a in range(len(keeper_idx)):
        for b in range(a + 1, len(keeper_idx)):
            if sim_cc[a, b] > float(par["f2_sim"]):
                f2_pairs.append((concept_names[keeper_idx[a]],
                                 concept_names[keeper_idx[b]],
                                 f"cos={sim_cc[a, b]:.3f}>{par['f2_sim']}"))

    f3_pairs = []
    ks = [c for c in concept_names if c in keep]
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            ia, ib = states[ks[a]].ids, states[ks[b]].ids
            union = ia | ib
            if union and len(ia & ib) / len(union) > float(par["f3_jaccard"]):
                f3_pairs.append((ks[a], ks[b],
                                 f"jaccard={len(ia & ib) / len(union):.3f}>"
                                 f"{par['f3_jaccard']}"))

    # 合并去重（同一对可能同时命中 F2/F3）
    pair_map: dict[tuple[str, str], list[str]] = {}
    for a, b, ev in f2_pairs + f3_pairs:
        key = (a, b) if a < b else (b, a)
        pair_map.setdefault(key, []).append(ev)
    pairs = sorted(pair_map.keys())
    if pairs:
        bar.total += len(pairs)
        bar.refresh()
    print(f"[B4] F2 候选对 {len(f2_pairs)} | F3 候选对 {len(f3_pairs)} | "
          f"去重后 {len(pairs)} 对进入 P6 仲裁")

    # P6 缓存（键 = 概念对 + 双方定义 md5）
    llm_name = Path(args.model_path).expanduser().resolve().name
    cache_dir = Path(args.cache_root) / llm_name / "concepts"
    p6_cache = JsonlCache(cache_dir / "p6_merge_cache.jsonl")

    bar.set_postfix_str("加载LLM(P6)")
    llm = LLMBackend(args.model_path, args.backend, args.device, args.dtype,
                     seed=seed, gpu_memory_utilization=args.gpu_memory_utilization,
                     max_model_len=args.max_model_len)

    rep_of = {c: c for c in keep}
    n_p6_default = 0
    for a, b in pairs:
        ra, rb = resolve(rep_of, a), resolve(rep_of, b)
        if ra == rb:      # 已在先前合并中归并
            bar.update(1)
            bar.set_postfix_str(f"P6 {a}~{b}(已归并)")
            continue
        key = hashlib.md5((ra + "\x1f" + states[ra].definition + "\x1f" +
                           rb + "\x1f" + states[rb].definition
                           ).encode("utf-8")).hexdigest()
        merge = None
        reason = ""
        raw = p6_cache.get(key)
        obj = extract_json(raw) if raw is not None else None
        if isinstance(obj, dict) and isinstance(obj.get("merge"), bool):
            merge = obj["merge"]
            reason = str(obj.get("reason", ""))[:20]
        if merge is None:
            msgs = [build_p6_messages(ra, states[ra].definition,
                                      rb, states[rb].definition)]
            for attempt in range(int(par["p6_retries"]) + 1):
                raws = llm.chat_batch(msgs, par["p6_temperature"], par["p6_top_p"],
                                      par["p6_max_new_tokens"], 1, desc="P6仲裁",
                                      progress=False)
                obj = extract_json(raws[0])
                if isinstance(obj, dict) and isinstance(obj.get("merge"), bool):
                    merge = obj["merge"]
                    reason = str(obj.get("reason", ""))[:20]
                    p6_cache.put(key, raws[0])
                    break
            if merge is None:
                # 解析失败 → 按量化规则默认合并（阈值本身即强证据），并告警
                merge = True
                reason = "P6解析失败，按量化规则默认"
                n_p6_default += 1
                print(f"[B4][P6][警告] 对 ({ra}, {rb}) 解析失败重试耗尽，"
                      f"默认 merge=true")
        if merge:
            # 删低频者；freq 相同保留字典序小的名字（确定性）
            if states[ra].freq > states[rb].freq:
                winner, loser = ra, rb
            elif states[ra].freq < states[rb].freq:
                winner, loser = rb, ra
            else:
                winner, loser = sorted([ra, rb])
            states[winner].ids |= states[loser].ids
            for sp in states[loser].spans:
                if sp not in states[winner].spans:
                    states[winner].spans.append(sp)
            states[winner].members = list(dict.fromkeys(
                states[winner].members + states[loser].members))
            rep_of[loser] = winner
            merge_log.append({"kept": winner, "dropped": loser,
                              "evidence": "; ".join(pair_map[(a, b)]),
                              "reason": reason})
            keep.discard(loser)
            delete_log.append({"concept": loser, "rule": "F2/F3合并(P6)",
                               "evidence": f"并入 {winner}: "
                                           f"{'; '.join(pair_map[(a, b)])}"})
        else:
            merge_log.append({"kept": ra, "dropped": "",
                              "evidence": "; ".join(pair_map[(a, b)]),
                              "reason": reason or "保留两者"})
        bar.update(1)
        bar.set_postfix_str(f"P6 {a}~{b}")

    # ---- Step5 F4 频次 ----
    bar.set_postfix_str("F4频次")
    f4_del = 0
    for c in sorted(keep):
        if states[c].freq < int(par["tau_freq"]):
            delete_log.append({"concept": c, "rule": "F4频次",
                               "evidence": f"freq={states[c].freq}<{par['tau_freq']}"})
            f4_del += 1
    keep -= {d["concept"] for d in delete_log if d["rule"] == "F4频次"}
    bar.update(1)
    print(f"\n[B4] F4 删除 {f4_del} 条")

    # ---- Step6 F5 线性可辨识度 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    f5_list = sorted(keep)
    bar.total += len(f5_list)
    bar.refresh()
    idx = np.arange(n_pool)
    n_splits = int(par["f5_n_folds"])
    n_f5_del = 0
    n_f5_skip = 0
    for c in f5_list:
        bar.set_postfix_str(f"F5探针 {c}")
        span_set = set(states[c].spans)
        if not span_set:
            states[c].f5_status = "跳过(无片段)"
            n_f5_skip += 1
            bar.update(1)
            continue
        y = np.array([any(sp in t for sp in span_set) for t in texts])
        if y.sum() < n_splits or y.sum() >= n_pool - 1:
            states[c].f5_status = "跳过(正例过少/近全一)"
            n_f5_skip += 1
            bar.update(1)
            continue
        # 5 折分层交叉验证：单次划分在低正例场景下方差过大，
        # 均值 F1 才是"线性可辨识度"的稳定估计
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        fold_f1: list[float] = []
        for tr_idx, va_idx in skf.split(idx, y):
            if y[va_idx].sum() == 0:
                continue
            clf = LogisticRegression(max_iter=int(par["f5_max_iter"]),
                                     class_weight="balanced",
                                     random_state=seed)
            clf.fit(text_emb[tr_idx], y[tr_idx])
            fold_f1.append(float(f1_score(y[va_idx],
                                          clf.predict(text_emb[va_idx]))))
        if len(fold_f1) < max(2, n_splits - 1):
            states[c].f5_status = "跳过(有效折不足)"
            n_f5_skip += 1
            bar.update(1)
            continue
        f1 = float(np.mean(fold_f1))
        states[c].f5_f1 = round(f1, 4)
        states[c].f5_status = "probe"
        if f1 < float(par["tau_g"]):
            delete_log.append({"concept": c, "rule": "F5线性探针",
                               "evidence": f"CV均值F1={f1:.3f}<{par['tau_g']} "
                                           f"(正例 {int(y.sum())}/{n_pool}, "
                                           f"有效折 {len(fold_f1)})"})
            n_f5_del += 1
        bar.update(1)
    keep -= {d["concept"] for d in delete_log if d["rule"] == "F5线性探针"}
    print(f"\n[B4] F5 删除 {n_f5_del} 条（跳过探针 {n_f5_skip} 条）")

    # ---- Step7 F6 近常数 ----
    bar.set_postfix_str("F6近常数")
    for c in sorted(keep):
        rate = states[c].freq / n_pool
        if rate < float(par["f6_zero_rate"]) or rate > float(par["f6_one_rate"]):
            delete_log.append({"concept": c, "rule": "F6近常数",
                               "evidence": f"触发率={rate:.4f} "
                                           f"(阈值 {par['f6_zero_rate']}/"
                                           f"{par['f6_one_rate']})"})
    keep -= {d["concept"] for d in delete_log if d["rule"] == "F6近常数"}
    bar.update(1)

    # ---- Step8 落盘 ----
    bar.set_postfix_str("落盘")
    final = sorted(keep, key=lambda c: -states[c].freq)
    if len(final) > int(par["k_max"]):
        print(f"[B4][警告] 最终词典 {len(final)} > k_max={par['k_max']}")

    csv_path = out_dir / "B_dict_final.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "definition", "category", "confidence",
                    "freq_agg", "trigger_rate", "type_majority", "source",
                    "members", "f5_f1", "f5_status"])
        for c in final:
            s = states[c]
            w.writerow([c, s.definition, s.category, s.confidence,
                        s.freq, round(s.freq / n_pool, 5), s.type_majority,
                        s.source, "|".join(s.members),
                        "" if s.f5_f1 is None else s.f5_f1, s.f5_status])

    rates = [states[c].freq / n_pool for c in final]
    g = gini(rates)
    rules_cnt: dict[str, int] = {}
    for d in delete_log:
        rules_cnt[d["rule"]] = rules_cnt.get(d["rule"], 0) + 1

    # filter_report.md（逐条证据，只读审计）
    rep = ["# B4 过滤报告（逐条删除/合并证据，§6 只读审计）", ""]
    rep.append(f"- 输入 {len(draft)} 条 → **最终 {len(final)} 条** "
               f"(k_max={par['k_max']}) | 合并对 {len(merge_log)} | "
               f"P6 解析失败默认合并 {n_p6_default}")
    rep.append(f"- 删除计数: {rules_cnt}")
    rep.append(f"- 词典级诊断: 触发率基尼系数 g={g:.3f} | "
               f"最高触发率 {max(rates):.3f} ({final[0]}) | "
               f"最低触发率 {min(rates):.4f}")
    rep.append("")
    rep.append("## 最终词典（按 freq 降序）")
    rep.append("")
    rep.append("| 概念 | 类别 | freq | 触发率 | F5 F1 |")
    rep.append("|---|---|---|---|---|")
    for c in final:
        s = states[c]
        f1s = "-" if s.f5_f1 is None else f"{s.f5_f1:.3f}"
        rep.append(f"| {c} | {s.category} | {s.freq} | "
                   f"{s.freq / n_pool:.4f} | {f1s} |")
    rep.append("")
    rep.append("## P6 仲裁明细")
    rep.append("")
    if merge_log:
        rep.append("| 保留 | 删除 | 量化证据 | LLM 理由 |")
        rep.append("|---|---|---|---|")
        for m in merge_log:
            rep.append(f"| {m['kept']} | {m['dropped'] or '（保留两者）'} | "
                       f"{m['evidence']} | {m['reason']} |")
    else:
        rep.append("- （无候选对）")
    rep.append("")
    rep.append("## 删除明细")
    rep.append("")
    if delete_log:
        rep.append("| 概念 | 规则 | 证据 |")
        rep.append("|---|---|---|")
        for d in delete_log:
            rep.append(f"| {d['concept']} | {d['rule']} | {d['evidence']} |")
    else:
        rep.append("- （无）")
    (out_dir / "filter_report.md").write_text("\n".join(rep), encoding="utf-8")

    stats = {
        "stage": "B4",
        "elapsed_sec": round(time.time() - t0, 1),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "encoder_path": str(Path(args.encoder_path).expanduser().resolve()),
        "params": {k: (v if not isinstance(v, list) else f"<{len(v)}项>")
                   for k, v in par.items()},
        "input": {"n_draft": len(draft), "n_pool": n_pool},
        "rules": rules_cnt,
        "merges": {"n_pairs": len(pairs), "n_merged": len(merge_log),
                   "n_p6_default": n_p6_default},
        "output": {"k_final": len(final), "k_max": par["k_max"],
                   "gini_trigger_rate": round(g, 4),
                   "max_trigger_rate": round(max(rates), 5) if rates else 0,
                   "min_trigger_rate": round(min(rates), 5) if rates else 0},
    }
    (out_dir / "b4_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    bar.update(1)
    bar.set_postfix_str("完成")
    bar.close()

    print(f"\n[B4] 完成: 最终词典 {len(final)} 条 → {csv_path}")
    print(f"[B4] 删除: {rules_cnt} | 合并对: "
          f"{sum(1 for m in merge_log if m['dropped'])}")
    print(f"[B4] 诊断: 触发率基尼 g={g:.3f} | "
          f"最高 {final[0]}({max(rates):.3f})" if rates else "[B4] 空词典!")
    print(f"[B4] 产物: {out_dir}/B_dict_final.csv, filter_report.md, b4_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
