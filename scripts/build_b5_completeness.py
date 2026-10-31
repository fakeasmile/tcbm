# -*- coding: utf-8 -*-
"""
build_b5_completeness.py — 词典B构建 · B5 完备性迭代与 k 的确定
（实施细则：docs/词典B全自动构建方案.md §7；执行清单 §10 第 6 步）

不动点迭代（每轮，全部自动零人工）：
  1. 弱标签：当前版词典各概念的触发 span（成员 text_spans ∪ 成员名 ∪ 概念名）
     在全量 TOXICN train（9600 条）上字符串匹配 → 样本级"涉及哪些类别概念"
     的 8 类多标签（类别头监督，词典派生，零人工）。
  2. PLM-c：bge-m3 冻结编码（全量 9600 嵌入只算一次，落盘缓存）+ 每类别独立
     线性头（class_weight=balanced），9:1 内部划分（分层于 toxic_type×expression，
     划分落盘复用）之 train 部分训练，val 部分出 8 维类别概率向量。
  3. LLM-c：val 部分逐样本对 8 类别三级打分（1=不相关/2=提及/3=实施，
     含类别定义与各类 top-5 高频成员示例），JSON 输出，缓存键含示例签名
     （词典变化自动失效）。
  4. 一致率：两教师类别向量 argmax 一致率，按 expression 分层报告
     （显式/隐式/转述；expression 仅仇恨类有效）。总体与隐式子集同时
     连续两轮增益 < sat_gain → 饱和停止。
  5. P7 挖缺：两教师余弦距离最大的 top p7_top 条 val 样本 → LLM 提出 0-2 个
     缺失概念/条 → 规范化去重 → P2 词性双通道闸门（jieba+LLM，双形容词删）
     → 单点归簇（与最近宏概念 cos > merge_sim 并入，否则立新）。
  6. 新概念补全与过滤：P4 定义 + P5 归类（复用 B3 提示词与缓存格式）；
     F1（标签捷径）/F2（冗余）/F3（触发重合，池内名称匹配）沿用原规则；
     F4'→教师激活率 ≥ f4p_rate（LLM-k 教师池上三级打分 ≥2 占比，等价换算
     5/1800）；F5'→教师激活弱标签探针（5 折 CV 均值 F1 ≥ tau_g）；
     F6'→激活率 < 0.1% 或 > 90% 删。
  7. v_{t+1} = v_t + 并入成员 + 存活新概念；写 B_dict_v{t}.csv 快照。

停止条件（§7.3 任一）：①一致率饱和（总体+隐式同时）；②k ≥ k_max(150)；
③连续 2 轮存活率 < 20% 且已迭代 ≥2 轮；④安全上限 max_rounds。

产物：
  data/concepts/B_dict_final.csv        最终词典（迭代结束版）
  data/concepts/build_log.md            每轮 k/一致率曲线/增补删除明细
  data/concepts/b5_stats.json           汇总统计
  data/concepts/B_dict_v{t}.csv         各轮快照
  data/processed/TOXICN/split_9to1.json                      9:1 划分（复用）
  data/processed/TOXICN/{encoder}/train_emb.npy              全量嵌入缓存
  data/processed/TOXICN/{llm}/concepts/b5_*.jsonl            LLM 缓存（断点续跑）

复用：B0 BGEM3Encoder；B1 LLMBackend/JsonlCache/extract_json/run_p2_llm/
jieba_pos_flags/normalize_concept；B3 CATEGORY_DEFS/build_p4_messages/
build_p5_messages。

离线硬约束（红线）：同 B0-B4，全程不联网。

用法（用户执行，本脚本不由 AI 运行）：
  conda activate toxic_env
  python scripts/build_b5_completeness.py --model_path /root/autodl-tmp/glm-4-9b-chat \
      --encoder_path /root/autodl-tmp/bge-m3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import OrderedDict, Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_b0_sample_pool import BGEM3Encoder          # noqa: E402
from build_b1_micro_concepts import (                  # noqa: E402
    LLMBackend, JsonlCache, enforce_offline, extract_json,
    run_p2_llm, jieba_pos_flags, normalize_concept, ADJ_POS_LLM,
)
from build_b3_dict_draft import (                      # noqa: E402
    CATEGORY_DEFS, CATEGORY_NAMES, build_p4_messages, build_p5_messages,
)


# ---------------------------------------------------------------------------
# 参数与配置
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="B5 完备性迭代：教师一致率饱和判据 + P7 分歧挖缺增补",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_path", type=str, required=True,
                    help="本地 LLM 目录，禁止远程下载")
    ap.add_argument("--encoder_path", type=str, default="models/bge-m3")
    ap.add_argument("--backend", type=str, default="vllm",
                    choices=["vllm", "transformers"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--train_input", type=str,
                    default="data/raw/TOXICN/train.json")
    ap.add_argument("--pool_input", type=str,
                    default="data/concepts/sample_pool.json")
    ap.add_argument("--b1_input", type=str,
                    default="data/concepts/micro_concepts_raw.jsonl")
    ap.add_argument("--dict_input", type=str,
                    default="data/concepts/B_dict_final.csv")
    ap.add_argument("--config", type=str, default="configs/dict_b.yaml")
    ap.add_argument("--out_dir", type=str, default="data/concepts")
    ap.add_argument("--cache_root", type=str, default="data/processed/TOXICN")
    ap.add_argument("--encoder_device", type=str, default="auto")
    ap.add_argument("--encoder_max_length", type=int, default=256)
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
    b5 = cfg.get("b5", {}) if isinstance(cfg, dict) else {}
    b4 = cfg.get("b4", {}) if isinstance(cfg, dict) else {}
    b3 = cfg.get("b3", {}) if isinstance(cfg, dict) else {}
    defaults = {
        # 判据与停止条件（§7）
        "sat_gain": 0.005,           # 一致率饱和增益阈值（0.5pp）
        "survival_min": 0.2,         # 存活率枯竭阈值
        "max_rounds": 8,             # 安全上限
        "k_max": 150,
        # P7 挖缺
        "p7_top": 200,
        "p7_batch": 10,              # 每次调用分析的样本数
        "p7_temperature": 0.3,
        "p7_top_p": 0.9,
        "p7_max_new_tokens": 600,
        "p7_retries": 2,
        # 归簇
        "merge_sim": 0.75,
        # F1/F2/F3/F6 阈值沿用 B4
        "f1_sim": 0.85, "f2_sim": 0.9, "f3_jaccard": 0.95,
        "tau_g": 0.30, "f6_zero_rate": 0.001, "f6_one_rate": 0.9,
        # F4' 教师激活率（等价 5/1800）
        "f4p_rate": 0.003,
        # LLM-c / LLM-k 打分
        "llm_c_batch": 32,
        "score_temperature": 0.0,
        "score_top_p": 1.0,
        "score_max_new_tokens": 256,
        "score_retries": 2,
        "text_max_chars": 160,
        "examples_top": 5,
        # F5' 探针
        "f5_n_folds": 5,
        "f5_max_iter": 1000,
        # PLM-c 线性头
        "plm_max_iter": 1000,
        # P2 闸门（run_p2_llm 需要）
        "p2_batch_terms": 20,
        "p2_temperature": 0.2,
        "p2_top_p": 0.9,
        "p2_max_new_tokens": 1200,
        # P4/P5 沿用 B3
        "p4_temperature": 0.2, "p4_top_p": 0.9, "p4_max_new_tokens": 256,
        "p5_temperature": 0.2, "p5_top_p": 0.9, "p5_max_new_tokens": 128,
        "p4_batch_size": 16,
        "seed": 729,
    }
    for k in defaults:
        if k in b5:
            defaults[k] = b5[k]
    # 与 B4 同源的阈值：配置里 b4 段为准
    for k in ("f1_sim", "f2_sim", "f3_jaccard", "tau_g",
              "f6_zero_rate", "f6_one_rate", "k_max"):
        if k in b4:
            defaults[k] = b4[k]
    for k in ("p4_temperature", "p4_top_p", "p4_max_new_tokens",
              "p5_temperature", "p5_top_p", "p5_max_new_tokens",
              "p4_batch_size"):
        if k in b3:
            defaults[k] = b3[k]
    if "seed" in cfg:
        defaults["seed"] = cfg["seed"]
    return defaults


# ---------------------------------------------------------------------------
# 数据与缓存
# ---------------------------------------------------------------------------

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_train(path: str) -> list[dict]:
    rows = json.load(open(path, encoding="utf-8"))
    out = [{"text": str(r["content"]), "toxic_type": int(r["toxic_type"]),
            "expression": int(r["expression"])} for r in rows]
    return out


def get_split(n: int, meta: list[dict], path: Path, seed: int):
    """9:1 内部划分（分层于 toxic_type×expression），落盘复用；测试集零接触。"""
    if path.is_file():
        d = json.load(open(path, encoding="utf-8"))
        va = np.array(d["val_idx"], dtype=int)
        print(f"[B5] 复用已有 9:1 划分: val {len(va)} 条")
    else:
        from sklearn.model_selection import train_test_split
        strat = [f"{m['toxic_type']}_{m['expression']}" for m in meta]
        tr, va = train_test_split(np.arange(n), test_size=0.1,
                                  random_state=seed, stratify=strat)
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"val_idx": va.tolist(), "seed": seed,
                   "stratify": "toxic_type_x_expression"},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[B5] 新建 9:1 划分: val {len(va)} 条 → {path}")
    va = set(json.load(open(path, encoding="utf-8"))["val_idx"])
    tr = np.array([i for i in range(n) if i not in va], dtype=int)
    va = np.array(sorted(va), dtype=int)
    return tr, va


def load_or_build_emb(encoder: BGEM3Encoder, texts: list[str],
                      cache_path: Path, enc_key: str) -> np.ndarray:
    side = cache_path.with_suffix(".meta.json")
    if cache_path.is_file() and side.is_file():
        m = json.load(open(side, encoding="utf-8"))
        if m.get("encoder") == enc_key and m.get("n") == len(texts):
            emb = np.load(cache_path)
            print(f"[B5] 复用嵌入缓存: {cache_path.name} {emb.shape}")
            return emb
    print(f"[B5] 编码 {len(texts)} 条文本（一次性，落盘缓存）...")
    emb = encoder.encode(texts, desc="", progress=False)
    np.save(cache_path, emb)
    json.dump({"encoder": enc_key, "n": len(texts), "dim": int(emb.shape[1])},
              open(side, "w", encoding="utf-8"))
    return emb


# ---------------------------------------------------------------------------
# 词典状态与弱标签
# ---------------------------------------------------------------------------

class DictState:
    def __init__(self, concept: str, definition: str, category: str,
                 confidence: str, type_majority: str, source: str,
                 members: list[str]):
        self.concept = concept
        self.definition = definition
        self.category = category
        self.confidence = confidence
        self.type_majority = type_majority
        self.source = source
        self.members = list(members)
        self.spans: list[str] = []       # 弱标签匹配用（成员 spans ∪ 名字）
        self.round_added = 0             # 0=B4 前身；t=第 t 轮增补
        self.freq_agg: int | None = None       # B4 版词典自带（新概念为空）
        self.trigger_rate: float | None = None


def load_dict(path: str, raw_index: dict) -> OrderedDict[str, DictState]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    states: OrderedDict[str, DictState] = OrderedDict()
    for r in rows:
        members = [m for m in r["members"].split("|") if m]
        s = DictState(r["concept"], r["definition"], r["category"],
                      r["confidence"], r["type_majority"], r["source"], members)
        if r.get("freq_agg"):
            s.freq_agg = int(r["freq_agg"])
        if r.get("trigger_rate"):
            s.trigger_rate = float(r["trigger_rate"])
        spans: list[str] = []
        for m in members:
            if m not in spans:
                spans.append(m)                       # 名字本身也参与匹配
            raw = raw_index.get(m, {})
            for sp in raw.get("text_spans", []):
                sp = str(sp).strip()
                if len(sp) >= 2 and sp not in spans:
                    spans.append(sp)
        s.spans = spans
        states[r["concept"]] = s
    return states


def weak_labels(states: OrderedDict, texts: list[str],
                cat_index: dict) -> np.ndarray:
    """全量 train 上 span 匹配 → (n, 8) 类别多标签（0/1）。"""
    span2cats: dict[str, set] = {}
    for s in states.values():
        ci = cat_index[s.category]
        for sp in s.spans:
            span2cats.setdefault(sp, set()).add(ci)
    Y = np.zeros((len(texts), len(CATEGORY_NAMES)), dtype=np.float32)
    for i, t in enumerate(texts):
        cats: set = set()
        for sp, cs in span2cats.items():
            if sp in t:
                cats |= cs
        for ci in cats:
            Y[i, ci] = 1.0
    return Y


def train_plmc(X: np.ndarray, Y: np.ndarray, tr_idx: np.ndarray,
               par: dict, seed: int):
    """每类别独立线性头（bge-m3 冻结，只训头）。返回 heads 与零正例类别集。"""
    from sklearn.linear_model import LogisticRegression
    heads = []
    zero_pos: set[int] = set()
    for j in range(Y.shape[1]):
        y = Y[tr_idx, j]
        if y.sum() == 0:
            heads.append(None)
            zero_pos.add(j)
            continue
        clf = LogisticRegression(max_iter=int(par["plm_max_iter"]),
                                 class_weight="balanced", random_state=seed)
        clf.fit(X[tr_idx], y)
        heads.append(clf)
    return heads, zero_pos


def predict_plmc(heads, X_va: np.ndarray) -> np.ndarray:
    P = np.zeros((X_va.shape[0], len(heads)), dtype=np.float32)
    for j, clf in enumerate(heads):
        if clf is None:
            continue
        P[:, j] = clf.predict_proba(X_va)[:, 1]
    return P


# ---------------------------------------------------------------------------
# LLM 教师
# ---------------------------------------------------------------------------

def build_llm_c_messages(text: str, examples: dict[str, list[str]],
                         defs_json: str, max_chars: int) -> list[dict]:
    t = text[:max_chars]
    ex_json = json.dumps(examples, ensure_ascii=False)
    user = (
        "对下列文本，逐一判断它与每个概念类别的关联程度。等级定义：\n"
        "1=完全不相关；2=文本提及/涉及该类别；3=文本在使用/体现该类别实施行为。\n"
        f"类别定义：{defs_json}\n"
        f"各类别典型成员示例：{ex_json}\n"
        "只输出 JSON：{\"levels\": {\"类别名\": 1|2|3, ...}}，8 个类别全部给出。\n"
        f"文本：{t}"
    )
    return [{"role": "system",
             "content": "你是社交媒体有害言论研究的类别标注助手。"
                        "严格按用户规则输出 JSON，不要输出任何其它内容。"},
            {"role": "user", "content": user}]


def llm_c_scores(llm: LLMBackend, texts: list[str], examples: dict,
                 cache: JsonlCache, par: dict, sig: str,
                 on_progress=None) -> np.ndarray:
    """val 样本 × 8 类别三级打分 → (n,8) int(1-3)。缓存键含示例签名。"""
    defs_json = json.dumps(CATEGORY_DEFS, ensure_ascii=False)
    n_cat = len(CATEGORY_NAMES)
    levels = np.ones((len(texts), n_cat), dtype=np.int8)
    keys = [md5(t[:par["text_max_chars"]] + "\x1f" + sig) for t in texts]
    todo: list[int] = []
    for i, k in enumerate(keys):
        raw = cache.get(k)
        obj = extract_json(raw) if raw is not None else None
        got = False
        if isinstance(obj, dict) and isinstance(obj.get("levels"), dict):
            lv = obj["levels"]
            ok = True
            for c in CATEGORY_NAMES:
                v = lv.get(c)
                if isinstance(v, int) and v in (1, 2, 3):
                    levels[i, CATEGORY_NAMES.index(c)] = v
                else:
                    ok = False
                    break
            got = ok
        if got:
            if on_progress:
                on_progress()
        else:
            todo.append(i)

    bs = int(par["llm_c_batch"])
    for attempt in range(int(par["score_retries"]) + 1):
        if not todo:
            break
        nxt: list[int] = []
        for st in range(0, len(todo), bs):
            chunk = todo[st:st + bs]
            msgs = [build_llm_c_messages(texts[i], examples, defs_json,
                                         par["text_max_chars"]) for i in chunk]
            raws = llm.chat_batch(msgs, par["score_temperature"],
                                  par["score_top_p"],
                                  par["score_max_new_tokens"], bs,
                                  desc="LLM-c类别打分", progress=False)
            for i, raw in zip(chunk, raws):
                obj = extract_json(raw)
                if isinstance(obj, dict) and isinstance(obj.get("levels"), dict):
                    lv = obj["levels"]
                    if all(isinstance(lv.get(c), int) and lv[c] in (1, 2, 3)
                           for c in CATEGORY_NAMES):
                        for ci, c in enumerate(CATEGORY_NAMES):
                            levels[i, ci] = lv[c]
                        cache.put(keys[i], raw)
                        if on_progress:
                            on_progress()
                        continue
                nxt.append(i)
        todo = nxt
        if todo:
            print(f"[B5][LLM-c][警告] {len(todo)} 条解析失败，重试 "
                  f"{attempt + 1}/{par['score_retries']}")
    for i in todo:
        if on_progress:
            on_progress()          # 失败也计入进度（保持 1 级默认值）
    return levels


def build_llm_k_messages(text: str, concepts: list[dict],
                         max_chars: int) -> list[dict]:
    t = text[:max_chars]
    c_json = json.dumps(concepts, ensure_ascii=False)
    user = (
        "对下列文本，逐一判断它与每个概念的关联程度。等级定义：\n"
        "1=完全不相关；2=文本提及/涉及该概念；3=文本在使用/体现该概念实施行为。\n"
        f"概念与定义：{c_json}\n"
        "只输出 JSON：{\"scores\": {\"概念名\": 1|2|3, ...}}，全部概念给出。\n"
        f"文本：{t}"
    )
    return [{"role": "system",
             "content": "你是社交媒体有害言论研究的概念标注助手。"
                        "严格按用户规则输出 JSON，不要输出任何其它内容。"},
            {"role": "user", "content": user}]


def llm_k_scores(llm: LLMBackend, texts: list[str], concepts: list[dict],
                 cache: JsonlCache, par: dict,
                 on_progress=None) -> np.ndarray:
    """池样本 × 新概念三级打分 → (n, len(concepts)) int(1-3)。"""
    n_cat = len(concepts)
    if n_cat == 0:
        return np.zeros((len(texts), 0), dtype=np.int8)
    levels = np.ones((len(texts), n_cat), dtype=np.int8)
    ckey = md5(json.dumps([c["concept"] for c in concepts],
                          ensure_ascii=False))
    keys = [md5(t[:par["text_max_chars"]] + "\x1f" + ckey) for t in texts]
    todo: list[int] = []
    for i, k in enumerate(keys):
        raw = cache.get(k)
        obj = extract_json(raw) if raw is not None else None
        got = False
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            lv = obj["scores"]
            ok = True
            for j, cd in enumerate(concepts):
                v = lv.get(cd["concept"])
                if isinstance(v, int) and v in (1, 2, 3):
                    levels[i, j] = v
                else:
                    ok = False
                    break
            got = ok
        if got:
            if on_progress:
                on_progress()
        else:
            todo.append(i)

    bs = int(par["llm_c_batch"])
    for attempt in range(int(par["score_retries"]) + 1):
        if not todo:
            break
        nxt: list[int] = []
        for st in range(0, len(todo), bs):
            chunk = todo[st:st + bs]
            msgs = [build_llm_k_messages(texts[i], concepts,
                                         par["text_max_chars"]) for i in chunk]
            raws = llm.chat_batch(msgs, par["score_temperature"],
                                  par["score_top_p"],
                                  par["score_max_new_tokens"], bs,
                                  desc="LLM-k概念打分", progress=False)
            for i, raw in zip(chunk, raws):
                obj = extract_json(raw)
                if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
                    lv = obj["scores"]
                    if all(isinstance(lv.get(cd["concept"]), int)
                           and lv[cd["concept"]] in (1, 2, 3)
                           for cd in concepts):
                        for j, cd in enumerate(concepts):
                            levels[i, j] = lv[cd["concept"]]
                        cache.put(keys[i], raw)
                        if on_progress:
                            on_progress()
                        continue
                nxt.append(i)
        todo = nxt
        if todo:
            print(f"[B5][LLM-k][警告] {len(todo)} 条解析失败，重试 "
                  f"{attempt + 1}/{par['score_retries']}")
    for i in todo:
        if on_progress:
            on_progress()
    return levels


# ---------------------------------------------------------------------------
# P7 挖缺
# ---------------------------------------------------------------------------

def build_p7_messages(batch: list[dict], concept_list: list[str]) -> list[dict]:
    user = (
        "当前概念词典尚不能解释下列文本与类别判断之间的矛盾。请为每条文本提出 "
        "0-2 个\"缺失概念\"：\n"
        "规则：\n"
        "1. 概念必须是名词性、2-8字、且是当前词典中不存在的；\n"
        "2. 提出的概念必须能解释该文本为何被判定为该条的 label_a 而非 "
        "label_b（label_a/label_b 见每条样本字段）；\n"
        "3. 只输出 JSON：{\"missing\": [{\"text_id\": ..., \"concept\": \"...\", "
        "\"reason\": \"≤20字\"}]}。\n"
        f"当前词典（仅概念名列表）：{json.dumps(concept_list, ensure_ascii=False)}\n"
        f"待分析文本（含两教师判断）：{json.dumps(batch, ensure_ascii=False)}"
    )
    return [{"role": "system",
             "content": "你是社交媒体有害言论研究的概念缺口挖掘助手。"
                        "严格按用户规则输出 JSON，不要输出任何其它内容。"},
            {"role": "user", "content": user}]


def p7_mine(llm: LLMBackend, samples: list[dict], concept_list: list[str],
            cache: JsonlCache, par: dict, dict_sig: str,
            on_progress=None) -> list[dict]:
    """samples: [{text_id, text, label_a, label_b}] → 候选列表（含去重前）。"""
    out: list[dict] = []
    bs = int(par["p7_batch"])
    batches = [samples[i:i + bs] for i in range(0, len(samples), bs)]
    for bi, batch in enumerate(batches):
        payload = [{"text_id": s["text_id"],
                    "text": s["text"][:par["text_max_chars"]],
                    "label_a": s["label_a"], "label_b": s["label_b"]}
                   for s in batch]
        key = md5(json.dumps(payload, ensure_ascii=False) + "\x1f" + dict_sig)
        raw = cache.get(key)
        obj = extract_json(raw) if raw is not None else None
        got = []
        if isinstance(obj, dict) and isinstance(obj.get("missing"), list):
            for it in obj["missing"]:
                if isinstance(it, dict) and str(it.get("concept", "")).strip():
                    got.append({"text_id": it.get("text_id"),
                                "concept": str(it["concept"]).strip(),
                                "reason": str(it.get("reason", ""))[:20]})
        if not got and raw is None:
            msgs = [build_p7_messages(payload, concept_list)]
            for attempt in range(int(par["p7_retries"]) + 1):
                raws = llm.chat_batch(msgs, par["p7_temperature"],
                                      par["p7_top_p"], par["p7_max_new_tokens"],
                                      1, desc="P7挖缺", progress=False)
                obj = extract_json(raws[0])
                if isinstance(obj, dict) and isinstance(obj.get("missing"), list):
                    for it in obj["missing"]:
                        if isinstance(it, dict) and str(it.get("concept", "")).strip():
                            got.append({"text_id": it.get("text_id"),
                                        "concept": str(it["concept"]).strip(),
                                        "reason": str(it.get("reason", ""))[:20]})
                    cache.put(key, raws[0])
                    break
        out.extend(got)
        if on_progress:
            on_progress()
    return out


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
    n_cat = len(CATEGORY_NAMES)
    cat_index = {c: i for i, c in enumerate(CATEGORY_NAMES)}

    # ---- 数据 ----
    train = load_train(args.train_input)
    texts_all = [r["text"] for r in train]
    n_train = len(train)
    tr_idx, va_idx = get_split(n_train, train,
                               Path(args.cache_root) / "split_9to1.json", seed)
    pool = json.load(open(args.pool_input, encoding="utf-8"))["samples"]
    pool_texts = [s["text"] for s in pool]
    raw_rows = [json.loads(l) for l in open(args.b1_input, encoding="utf-8") if l.strip()]
    raw_index = {r["concept"]: r for r in raw_rows}
    print(f"[B5] 全量 train {n_train}（train部分 {len(tr_idx)} / val {len(va_idx)}）"
          f" | 池 {len(pool_texts)}")

    # ---- 词典 ----
    states = load_dict(args.dict_input, raw_index)
    print(f"[B5] 初始词典: {len(states)} 条")

    # ---- 总进度条（唯一；各轮动态追加 work units）----
    bar = tqdm(total=1, desc="B5总进度", unit="step", ncols=100, miniters=1)

    # ---- 编码器与嵌入缓存（全量编码在 LLM 加载前完成，避显存冲突；
    #      循环内的少量名字/标签词编码一律走 CPU）----
    bar.set_postfix_str("bge-m3编码")
    encoder = BGEM3Encoder(args.encoder_path, args.encoder_device,
                           args.encoder_max_length, args.encoder_batch_size)
    enc_name = Path(args.encoder_path).expanduser().resolve().name
    enc_dir = Path(args.cache_root) / enc_name
    enc_dir.mkdir(parents=True, exist_ok=True)
    X_all = load_or_build_emb(encoder, texts_all,
                              enc_dir / "train_emb.npy",
                              enc_key=enc_name)
    X_pool = load_or_build_emb(encoder, pool_texts,
                               enc_dir / "pool_emb.npy",
                               enc_key=enc_name)
    encoder.model.to("cpu")
    encoder.device = "cpu"       # 循环内增量编码固定 CPU，防 vLLM 显存冲突
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    bar.update(1)

    # ---- LLM 与缓存 ----
    llm_name = Path(args.model_path).expanduser().resolve().name
    cache_dir = Path(args.cache_root) / llm_name / "concepts"
    llm = LLMBackend(args.model_path, args.backend, args.device, args.dtype,
                     seed=seed, gpu_memory_utilization=args.gpu_memory_utilization,
                     max_model_len=args.max_model_len)
    c_cache = JsonlCache(cache_dir / "b5_llmcat_cache.jsonl")
    k_cache = JsonlCache(cache_dir / "b5_llmk_cache.jsonl")
    p7_cache = JsonlCache(cache_dir / "b5_p7_cache.jsonl")
    p4_cache = JsonlCache(cache_dir / "b5_p4_cache.jsonl")
    p5_cache = JsonlCache(cache_dir / "b5_p5_cache.jsonl")
    p2_cache = JsonlCache(cache_dir / "b5_p2_cache.jsonl")

    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    log_lines = ["# B5 完备性迭代日志", ""]
    history: list[dict] = []
    round_stats_all: list[dict] = []
    stop_reason = ""

    for t in range(1, int(par["max_rounds"]) + 1):
        rt0 = time.time()
        print(f"\n===== [B5] 第 {t} 轮 =====")
        k_now = len(states)

        # ---- 停止条件检查（用上一轮历史）----
        if k_now >= int(par["k_max"]):
            stop_reason = f"k={k_now} 达上限 {par['k_max']}"
        elif len(history) >= 3:
            # 连续两轮增益（总体与隐式各自）均 < sat_gain 才算饱和（§7.1）
            pairs = []
            for a, b in ((-3, -2), (-2, -1)):
                pairs.append((history[b]["agree"] - history[a]["agree"],
                              history[b]["agree_impl"] - history[a]["agree_impl"]))
            if all(g < par["sat_gain"] for pair in pairs for g in pair):
                stop_reason = (f"一致率饱和（连续两轮增益均 < {par['sat_gain']}："
                               f"末两轮 总体 {pairs[-1][0]:+.4f} / "
                               f"隐式 {pairs[-1][1]:+.4f}）")
            surv = [h["survival"] for h in history[-2:]]
            if all(s < par["survival_min"] for s in surv):
                stop_reason = (f"连续 2 轮存活率 < {par['survival_min']}"
                               f"（{surv}）")
        if stop_reason:
            print(f"[B5] 停止: {stop_reason}")
            break

        # ---- 1. 弱标签 ----
        bar.set_postfix_str(f"R{t} 弱标签")
        Y_all = weak_labels(states, texts_all, cat_index)
        bar.update(1)

        # ---- 2. PLM-c ----
        bar.set_postfix_str(f"R{t} PLM-c训练")
        heads, zero_pos = train_plmc(X_all, Y_all, tr_idx, par, seed)
        P_va = predict_plmc(heads, X_all[va_idx])
        if zero_pos:
            print(f"[B5][提示] 类别 {[CATEGORY_NAMES[j] for j in zero_pos]} "
                  f"train 部分零正例，头输出恒 0")
        bar.update(1)

        # ---- 3. LLM-c（val 逐样本）----
        top_ex = {c: [] for c in CATEGORY_NAMES}
        for s in sorted(states.values(),
                        key=lambda x: -(x.freq_agg or len(x.members))):
            if len(top_ex[s.category]) < int(par["examples_top"]):
                top_ex[s.category].append(s.concept)
        ex_sig = md5(json.dumps(top_ex, ensure_ascii=False))
        va_texts = [texts_all[i] for i in va_idx]

        bar.total += len(va_texts) + 1   # LLM-c 逐样本 + 一致率
        bar.refresh()
        bar.set_postfix_str(f"R{t} LLM-c打分")
        done = {"n": 0}

        def _c_progress() -> None:
            done["n"] += 1
            bar.update(1)
            if done["n"] % 100 == 0:
                bar.set_postfix_str(f"R{t} LLM-c {done['n']}/{len(va_texts)}")

        L_va = llm_c_scores(llm, va_texts, top_ex, c_cache, par, ex_sig,
                            on_progress=_c_progress)
        L_vec = (L_va.astype(np.float32) - 1.0) / 2.0     # {0,0.5,1}

        # ---- 4. 一致率（分层）----
        bar.set_postfix_str(f"R{t} 一致率")
        llm_arg = L_vec.argmax(axis=1)
        plm_arg = P_va.argmax(axis=1)
        agree_all = float((llm_arg == plm_arg).mean())
        cos_sim = np.array([
            float(np.dot(L_vec[i], P_va[i]) /
                  (np.linalg.norm(L_vec[i]) * np.linalg.norm(P_va[i]) + 1e-9))
            for i in range(len(va_idx))])
        expr = np.array([train[i]["expression"] for i in va_idx])
        subset = {
            "显式": expr == 1, "隐式": expr == 2, "转述": expr == 3,
        }
        agree_sub = {k: float((llm_arg[m] == plm_arg[m]).mean())
                     if m.sum() else float("nan")
                     for k, m in subset.items()}
        cos_mean = float(cos_sim.mean())
        rec = {"round": t, "k": k_now, "agree": agree_all,
               "agree_impl": agree_sub["隐式"], "agree_expl": agree_sub["显式"],
               "agree_para": agree_sub["转述"], "cos_mean": cos_mean}
        history.append(rec)
        print(f"[B5] R{t} 一致率: 总体 {agree_all:.4f} | 隐式 "
              f"{agree_sub['隐式']:.4f} | 显式 {agree_sub['显式']:.4f} | "
              f"转述 {agree_sub['转述']:.4f} | 平均余弦 {cos_mean:.4f}")
        log_lines.append(
            f"- **R{t}**: k={k_now} | 一致率 总体 {agree_all:.4f} / "
            f"隐式 {agree_sub['隐式']:.4f} / 显式 {agree_sub['显式']:.4f} / "
            f"转述 {agree_sub['转述']:.4f} | 余弦 {cos_mean:.4f}")
        bar.update(1)

        # ---- 5. P7 挖缺（分歧 top）----
        dis = 1.0 - cos_sim
        n_top = min(int(par["p7_top"]), len(va_idx))
        order = np.argsort(-dis)[:n_top]
        samples = [{"text_id": int(va_idx[i]),
                    "text": va_texts[i],
                    "label_a": CATEGORY_NAMES[int(llm_arg[i])],
                    "label_b": CATEGORY_NAMES[int(plm_arg[i])]}
                   for i in order]
        dict_sig = md5(json.dumps(list(states.keys()), ensure_ascii=False))
        n_p7_calls = (n_top + int(par["p7_batch"]) - 1) // int(par["p7_batch"])
        bar.total += n_p7_calls + 1    # P7 调用 + 闸门/归簇
        bar.refresh()
        bar.set_postfix_str(f"R{t} P7挖缺")
        cands = p7_mine(llm, samples, list(states.keys()), p7_cache, par,
                        dict_sig, on_progress=lambda: bar.update(1))

        # ---- 6. 闸门 + 归簇 ----
        bar.set_postfix_str(f"R{t} P2闸门/归簇")
        seen: set[str] = set(states.keys())
        for s in states.values():
            seen.update(s.members)
        cand_list: list[str] = []
        for c in cands:
            nm = normalize_concept(str(c["concept"]))
            if not (2 <= len(nm) <= 8) or nm in seen:
                continue
            seen.add(nm)
            cand_list.append(nm)
        # P2 双通道：jieba + LLM，双形容词才删（方案 §4 同规则）
        if cand_list:
            pos_llm = run_p2_llm(cand_list, llm, p2_cache, par)
            kept: list[str] = []
            for nm in cand_list:
                adj_jieba, _ = jieba_pos_flags(
                    nm, {"a", "ad", "d"})
                adj_llm = pos_llm.get(nm) in ADJ_POS_LLM
                if adj_jieba and adj_llm:
                    continue
                kept.append(nm)
        else:
            kept = []
        # 归簇：与最近宏概念 cos > merge_sim → 并入（名字编码走 CPU，量小）
        if kept:
            name_emb_path = enc_dir / "dict_name_emb.npy"
            name_side = name_emb_path.with_suffix(".meta.json")
            dict_names = list(states.keys())
            name_emb = None
            if name_emb_path.is_file() and name_side.is_file():
                m = json.load(open(name_side, encoding="utf-8"))
                if m.get("n") == len(dict_names) and m.get("dict_sig") == dict_sig:
                    name_emb = np.load(name_emb_path)
                    dict_emb = name_emb
                    kept_emb = encoder.encode(kept, desc="", progress=False)
            if name_emb is None:
                emb_all = encoder.encode(dict_names + kept, desc="",
                                         progress=False)
                dict_emb, kept_emb = (emb_all[:len(dict_names)],
                                      emb_all[len(dict_names):])
                name_emb = dict_emb     # 回填：后续 F2 相似度检查依赖此变量
                np.save(name_emb_path, dict_emb)
                json.dump({"n": len(dict_names), "dict_sig": dict_sig},
                          open(name_side, "w", encoding="utf-8"))
            sim_nk = kept_emb @ dict_emb.T           # (n_kept, n_dict)
            merged: dict[str, list[str]] = {}
            standalone: list[str] = []
            for j, nm in enumerate(kept):
                bi = int(np.argmax(sim_nk[j]))
                if float(sim_nk[j, bi]) > float(par["merge_sim"]):
                    merged.setdefault(dict_names[bi], []).append(nm)
                else:
                    standalone.append(nm)
        else:
            merged, standalone = {}, []
        bar.update(1)
        print(f"[B5] P7 提出 {len(cands)} → 规范去重 {len(cand_list)} → "
              f"词性闸门 {len(kept)} → 并入 {sum(len(v) for v in merged.values())} "
              f"/ 立新 {len(standalone)}")

        # ---- 7. 新概念 P4/P5 + 过滤 ----
        n_p4 = len(standalone)
        bar.total += (n_p4 * 2
                      + (len(pool_texts) if standalone else 1)
                      + 1)             # P4+P5 逐概念 + LLM-k 逐池样本 + 过滤落盘
        bar.refresh()
        new_defs: dict[str, str] = {}
        for nm in standalone:
            bar.set_postfix_str(f"R{t} P4定义 {nm}")
            key = md5(nm + "\x1f" + json.dumps([nm], ensure_ascii=False))
            raw = p4_cache.get(key)
            obj = extract_json(raw) if raw is not None else None
            d = str(obj.get("definition", "")).strip() \
                if isinstance(obj, dict) else ""
            if not d:
                msgs = [build_p4_messages(nm, [nm])]
                raws = llm.chat_batch(msgs, par["p4_temperature"],
                                      par["p4_top_p"], par["p4_max_new_tokens"],
                                      1, desc="P4定义", progress=False)
                obj = extract_json(raws[0])
                d = str(obj.get("definition", "")).strip() \
                    if isinstance(obj, dict) else ""
                if d:
                    p4_cache.put(key, raws[0])
            if d:
                new_defs[nm] = d
            bar.update(1)
        new_cat: dict[str, dict] = {}
        for nm in standalone:
            bar.set_postfix_str(f"R{t} P5归类 {nm}")
            if nm not in new_defs:
                bar.update(1)
                continue
            key = md5(nm + "\x1f" + new_defs[nm])
            raw = p5_cache.get(key)
            obj = extract_json(raw) if raw is not None else None
            cat = str(obj.get("category", "")).strip() \
                if isinstance(obj, dict) else ""
            if cat in CATEGORY_NAMES:
                conf = str(obj.get("confidence", "中")).strip()
                new_cat[nm] = {"category": cat,
                               "confidence": conf if conf in ("高", "中", "低") else "中"}
                bar.update(1)
                continue
            msgs = [build_p5_messages(nm, new_defs[nm])]
            raws = llm.chat_batch(msgs, par["p5_temperature"], par["p5_top_p"],
                                  par["p5_max_new_tokens"], 1, desc="P5归类",
                                  progress=False)
            obj = extract_json(raws[0])
            cat = str(obj.get("category", "")).strip() \
                if isinstance(obj, dict) else ""
            if cat in CATEGORY_NAMES:
                conf = str(obj.get("confidence", "中")).strip()
                new_cat[nm] = {"category": cat,
                               "confidence": conf if conf in ("高", "中", "低") else "中"}
                p5_cache.put(key, raws[0])
            bar.update(1)
        # P4 失败者出局
        standalone = [nm for nm in standalone
                      if nm in new_defs and nm in new_cat]

        # 过滤（F1/F2/F3 + 教师激活 F4'/F5'/F6'）
        bar.set_postfix_str(f"R{t} LLM-k池打分")
        new_filt_log: list[dict] = []
        survivors: list[str] = []
        if standalone:
            concepts_payload = [{"concept": nm, "definition": new_defs[nm]}
                                for nm in standalone]
            done_k = {"n": 0}

            def _k_progress() -> None:
                done_k["n"] += 1
                bar.update(1)
                if done_k["n"] % 300 == 0:
                    bar.set_postfix_str(f"R{t} LLM-k {done_k['n']}/{len(pool_texts)}")

            K_pool = llm_k_scores(llm, pool_texts, concepts_payload, k_cache,
                                  par, on_progress=_k_progress)
            act = (K_pool >= 2).mean(axis=0)             # (n_new,) 激活率

            # 名字嵌入（F1/F2）：归簇阶段已为 standalone 编码过 kept_emb
            keep_names = kept
            keep_pos = [keep_names.index(nm) for nm in standalone]
            new_emb = kept_emb[keep_pos]
            label_emb = None
            label_words = ["非毒", "冒犯", "仇恨", "仇恨言论",
                           "显式", "隐式", "转述"] + list(CATEGORY_NAMES)
            # 标签词短文本：CPU 编码（encoder 已固定 CPU，防 vLLM 显存冲突）
            label_emb = encoder.encode(label_words, desc="", progress=False)

            # F1
            sim_nk2 = new_emb @ label_emb.T
            alive = list(range(len(standalone)))
            for j in list(alive):
                jj = int(np.argmax(sim_nk2[j]))
                if float(sim_nk2[j, jj]) > float(par["f1_sim"]):
                    new_filt_log.append({"concept": standalone[j],
                                         "rule": "F1", "evidence":
                                         f"与标签词\"{label_words[jj]}\""
                                         f"cos={sim_nk2[j, jj]:.3f}"})
                    alive.remove(j)
            # F2（与现有词典名 + 新概念彼此）
            dict_names_now = list(states.keys())
            if name_emb is not None and len(dict_emb) == len(dict_names_now):
                sim_nd = new_emb @ dict_emb.T
            else:
                sim_nd = np.zeros((new_emb.shape[0], 0), dtype=np.float32)
            for j in list(alive):
                if sim_nd.shape[1] and float(sim_nd[j].max()) > float(par["f2_sim"]):
                    new_filt_log.append({"concept": standalone[j], "rule": "F2",
                                         "evidence": f"与现有概念"
                                         f"\"{dict_names_now[int(np.argmax(sim_nd[j]))]}\""
                                         f"cos={sim_nd[j].max():.3f}"})
                    alive.remove(j)
            # 新概念彼此（cos>0.9 → 删激活率低者）
            if len(alive) > 1:
                sub = [standalone[j] for j in alive]
                sub_emb = new_emb[alive]
                sims = sub_emb @ sub_emb.T
                for a in range(len(sub)):
                    for b in range(a + 1, len(sub)):
                        if sims[a, b] > float(par["f2_sim"]) and \
                                sub[a] in [standalone[j] for j in alive] and \
                                sub[b] in [standalone[j] for j in alive]:
                            ja = alive[[standalone[x] for x in alive].index(sub[a])]
                            jb = alive[[standalone[x] for x in alive].index(sub[b])]
                            drop = jb if act[ja] >= act[jb] else ja
                            if drop in alive:
                                alive.remove(drop)
                                new_filt_log.append(
                                    {"concept": standalone[drop], "rule": "F2",
                                     "evidence": f"与 \"{sub[a] if drop==jb else sub[b]}\""
                                                 f"cos={sims[a, b]:.3f}，删低激活"})
            # F3（池内名称匹配 Jaccard）
            name_sets: dict[str, set] = {}
            for nm in [standalone[j] for j in alive]:
                name_sets[nm] = {i for i, t in enumerate(pool_texts) if nm in t}
            names_list = list(name_sets.keys())
            for a in range(len(names_list)):
                for b in range(a + 1, len(names_list)):
                    ia, ib = name_sets[names_list[a]], name_sets[names_list[b]]
                    u = ia | ib
                    if u and len(ia & ib) / len(u) > float(par["f3_jaccard"]):
                        ja = alive[[standalone[x] for x in alive].index(names_list[a])]
                        jb = alive[[standalone[x] for x in alive].index(names_list[b])]
                        drop = jb if act[ja] >= act[jb] else ja
                        if drop in alive:
                            alive.remove(drop)
                            new_filt_log.append(
                                {"concept": standalone[drop], "rule": "F3",
                                 "evidence": f"与 \"{names_list[b] if drop==jb else names_list[a]}\""
                                             f"触发重合 jaccard={len(ia & ib) / len(u):.3f}"})
            # F4' 激活率
            for j in list(alive):
                if float(act[j]) < float(par["f4p_rate"]):
                    new_filt_log.append({"concept": standalone[j], "rule": "F4'",
                                         "evidence": f"激活率={act[j]:.4f}<"
                                                     f"{par['f4p_rate']}"})
                    alive.remove(j)
            # F5' 教师激活弱标签探针（5 折 CV）
            for j in list(alive):
                nm = standalone[j]
                y = (K_pool[:, j] >= 2).astype(np.int8)
                if y.sum() < int(par["f5_n_folds"]) or y.sum() >= len(pool_texts) - 1:
                    new_filt_log.append({"concept": nm, "rule": "F5'",
                                         "evidence": "激活标签退化，跳过保留"})
                    continue
                skf = StratifiedKFold(n_splits=int(par["f5_n_folds"]),
                                      shuffle=True, random_state=seed)
                f1s = []
                for tri, vai in skf.split(X_pool, y):
                    if y[vai].sum() == 0:
                        continue
                    clf = LogisticRegression(max_iter=int(par["f5_max_iter"]),
                                             class_weight="balanced",
                                             random_state=seed)
                    clf.fit(X_pool[tri], y[tri])
                    f1s.append(float(f1_score(y[vai],
                                              clf.predict(X_pool[vai]))))
                f1m = float(np.mean(f1s)) if f1s else 0.0
                if f1m < float(par["tau_g"]):
                    new_filt_log.append({"concept": nm, "rule": "F5'",
                                         "evidence": f"CV均值F1={f1m:.3f}<"
                                                     f"{par['tau_g']}"})
                    alive.remove(j)
            # F6'
            for j in list(alive):
                if float(act[j]) < float(par["f6_zero_rate"]) or \
                        float(act[j]) > float(par["f6_one_rate"]):
                    new_filt_log.append({"concept": standalone[j], "rule": "F6'",
                                         "evidence": f"激活率={act[j]:.4f}越界"})
                    alive.remove(j)
            survivors = [standalone[j] for j in alive]
        else:
            bar.update(1)   # LLM-k 打分（空转，占位单元）
        bar.update(1)       # 过滤落盘

        # ---- 8. 更新词典 v_{t+1} ----
        n_merged = 0
        for parent, mems in merged.items():
            if parent in states:
                for m in mems:
                    if m not in states[parent].members:
                        states[parent].members.append(m)
                        if m not in states[parent].spans:
                            states[parent].spans.append(m)
                n_merged += len(mems)
        for nm in survivors:
            s = DictState(nm, new_defs[nm], new_cat[nm]["category"],
                          new_cat[nm]["confidence"], "其它", "b5_new", [nm])
            s.spans = [nm]
            s.round_added = t
            states[nm] = s
        survival = (len(survivors) / len(standalone)) if standalone else 0.0
        history[-1]["survival"] = round(survival, 4)
        round_stats_all.append({
            "round": t, "k_before": k_now, "n_proposed": len(cands),
            "n_gate": len(cand_list), "n_merged": n_merged,
            "n_new": len(standalone), "n_survived": len(survivors),
            "survival": round(survival, 4), "filt": new_filt_log,
            "elapsed": round(time.time() - rt0, 1),
        })
        log_lines.append(
            f"  - 增补: 提出 {len(cands)} → 闸门 {len(cand_list)} → "
            f"并入 {n_merged} / 立新 {len(standalone)} → **存活 "
            f"{len(survivors)}**（存活率 {survival:.0%}）")
        if new_filt_log:
            for f in new_filt_log:
                log_lines.append(f"    - 删 {f['concept']} [{f['rule']}] {f['evidence']}")
        # 快照
        snap = out_dir / f"B_dict_v{t}.csv"
        with open(snap, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["concept", "definition", "category", "confidence",
                        "freq_agg", "trigger_rate", "type_majority", "source",
                        "members", "round_added"])
            for s in states.values():
                w.writerow([s.concept, s.definition, s.category, s.confidence,
                            "" if s.freq_agg is None else s.freq_agg,
                            "" if s.trigger_rate is None else s.trigger_rate,
                            s.type_majority, s.source, "|".join(s.members),
                            s.round_added])
        print(f"[B5] R{t} 完成: k {k_now} → {len(states)} | "
              f"存活新概念 {len(survivors)} | 用时 {time.time() - rt0:.0f}s")

    if not stop_reason:
        stop_reason = f"达安全轮数上限 max_rounds={par['max_rounds']}"
    log_lines.append("")
    log_lines.append(f"**停止原因**: {stop_reason}")
    log_lines.append("")
    log_lines.append("| 轮 | k(前) | 提出 | 闸门 | 并入 | 立新 | 存活 | 存活率 |")
    log_lines.append("|---|---|---|---|---|---|---|---|")
    for r in round_stats_all:
        log_lines.append(
            f"| {r['round']} | {r['k_before']} | {r['n_proposed']} | "
            f"{r['n_gate']} | {r['n_merged']} | {r['n_new']} | "
            f"{r['n_survived']} | {r['survival']:.0%} |")
    (out_dir / "build_log.md").write_text("\n".join(log_lines), encoding="utf-8")

    # ---- 最终词典 ----
    final_path = out_dir / "B_dict_final.csv"
    with open(final_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "definition", "category", "confidence",
                    "freq_agg", "trigger_rate", "type_majority", "source",
                    "members", "round_added"])
        for s in states.values():
            w.writerow([s.concept, s.definition, s.category, s.confidence,
                        "" if s.freq_agg is None else s.freq_agg,
                        "" if s.trigger_rate is None else s.trigger_rate,
                        s.type_majority, s.source, "|".join(s.members),
                        s.round_added])

    agree_curve = [(h["round"], round(h["agree"], 4), round(h["agree_impl"], 4))
                   for h in history]
    stats = {
        "stage": "B5",
        "elapsed_sec": round(time.time() - t0, 1),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "encoder_path": str(Path(args.encoder_path).expanduser().resolve()),
        "params": {k: v for k, v in par.items() if not isinstance(v, list)},
        "rounds": round_stats_all,
        "agreement_curve": agree_curve,
        "stop_reason": stop_reason,
        "k_final": len(states),
    }
    (out_dir / "b5_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    bar.close()
    print(f"\n[B5] 完成: {stop_reason}")
    print(f"[B5] 最终词典 {len(states)} 条 → {final_path}")
    print(f"[B5] 一致率曲线（轮, 总体, 隐式）: {agree_curve}")
    print(f"[B5] 产物: B_dict_final.csv, build_log.md, b5_stats.json, "
          f"B_dict_v*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
