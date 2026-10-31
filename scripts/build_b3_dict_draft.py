# -*- coding: utf-8 -*-
"""
build_b3_dict_draft.py — 词典B构建 · B3 三字段补全（概念，定义，类别）
（实施细则：docs/词典B全自动构建方案.md §5；执行清单 §10 第 4 步）

流程（对应方案 §5，全部自动，零人工标注）：
  Step1  读取 B2 产物 macro_concepts.jsonl（77 个宏概念），并从 B1 产物
         micro_concepts_raw.jsonl 取各成员的 text_spans（原文本触发片段），
         按成员频次优先、去重、截断为 P4 证据片段（n_spans 条）。
  Step2  P4 定义生成：逐概念调用 LLM，提示词与方案 §5.2 逐字一致；
         定义解析失败重试 <=retries 次，仅成功结果写缓存（失败重跑自动补齐）。
  Step3  P5 类别归类：八类操作定义（方案 §5.1 逐字）写入提示词 §5.3，
         输出 {category, confidence}；类别名非法按失败重试，最终兜底
         归入"其它"+confidence=低 并打标（宁可错标监控类，不塞七类）。
  Step4  落盘：
         - B_dict_draft.csv：概念/定义/类别/置信 + 审计列（utf-8-sig，Excel 可开）；
         - b3_stats.json：类别分布、"其它"占比告警（>15% 触发体系修订检查）、
           P4/P5 失败清单、定义风格 lint；
         - b3_review.md：类别体系冻结确认材料（§8 审定点 2：类别定义 +
           各类按 freq_agg 排序的概念清单 + confidence=低 争议清单），
           供导师只读过目。

产物：
  data/concepts/B_dict_draft.csv
  data/concepts/b3_stats.json
  data/concepts/b3_review.md
  data/processed/TOXICN/{llm}/concepts/p4_def_cache.jsonl   （断点续跑）
  data/processed/TOXICN/{llm}/concepts/p5_cat_cache.jsonl   （键含定义，定义变则失效）

复用：B1 的 LLMBackend（vllm 默认后端、强离线、冒烟自愈）/JsonlCache/
extract_json/enforce_offline。

离线硬约束（红线）：同 B1/B2——全程 HF_HUB_OFFLINE=1 / local_files_only=True，
仅从本地路径加载模型，任何缺失只报错、绝不联网。

用法（用户执行，本脚本不由 AI 运行）：
  conda activate toxic_env   # 服务器环境
  python scripts/build_b3_dict_draft.py --model_path /root/autodl-tmp/glm-4-9b-chat
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_b1_micro_concepts import (                  # noqa: E402
    LLMBackend, JsonlCache, enforce_offline, extract_json,
)


# ---------------------------------------------------------------------------
# 类别体系（方案 §5.1，逐字；决策点 1：数据集弱先验锚定，8 类）
# 说明：类别名冻结后不可再改（§5.1），此处为唯一权威定义。
# ---------------------------------------------------------------------------

CATEGORY_DEFS: "OrderedDict[str, str]" = OrderedDict([
    ("群体指称", "指向特定人群的称谓，含中性称谓与蔑称（地域/种族/性别/职业/年龄等）"),
    ("侮辱称谓", "直接贬损、辱骂他人或其亲属性格/能力/品格的词语"),
    ("威胁恐吓", "威胁人身安全、煽动暴力、鼓吹驱逐/清洗的语汇"),
    ("刻板印象与偏见主题", "对群体的概括化负面断言主题（如\"某地人都是骗子\"）"),
    ("行为与事件", "描述冲突、歧视、骚扰等行为动作的名词化表达"),
    ("性与身体", "性暗示、外貌羞辱、身体特征攻击"),
    ("场景与话题", "引发争议的社会话题语境（户籍/婚恋/涉外/贫富等），本身中性"),
    ("其它", "暂无法归入上述七类的概念。仅作监控类：占比>15%触发体系一次性修订；"
             "确实无法归类的保留在此"),
])
CATEGORY_NAMES = list(CATEGORY_DEFS.keys())


# ---------------------------------------------------------------------------
# 提示词（方案 §5.2 / §5.3，逐字）
# ---------------------------------------------------------------------------

def build_p4_messages(concept: str, spans: list[str]) -> list[dict]:
    spans_json = json.dumps(spans, ensure_ascii=False)
    user = (
        "为下列概念写一句可判读的定义，要求：\n"
        "1. 不超过 40 字，说明该概念指什么、在社交媒体有害言论语境中的典型用法；\n"
        "2. 定义必须给出\"可判定边界\"——读完后能明确判断一条文本是否涉及该概念；\n"
        "3. 不得使用\"指……的概念\"这类循环句式，不得引用类别名作定义；\n"
        "4. 只输出 JSON：{\"definition\": \"...\"}。\n"
        f"概念：{concept}\n"
        f"典型成员（原文本触发片段）：{spans_json}"
    )
    return [{"role": "system",
             "content": "你是社交媒体有害言论研究的概念词典构建助手。"
                        "严格按用户规则输出 JSON，不要输出任何其它内容。"},
            {"role": "user", "content": user}]


def build_p5_messages(concept: str, definition: str) -> list[dict]:
    defs_json = json.dumps(CATEGORY_DEFS, ensure_ascii=False, indent=1)
    user = (
        "将概念归入唯一类别。类别定义如下：\n"
        f"{defs_json}\n"
        "规则：只输出 JSON：{\"category\": \"类别名\", \"confidence\": \"高|中|低\"}。\n"
        "若概念同时符合多类，选其在文本中最常出现的功能所属类；"
        "confidence=低 时仍必须给出类别。\n"
        f"概念：{concept}\n"
        f"定义：{definition}"
    )
    return [{"role": "system",
             "content": "你是社交媒体有害言论研究的概念归类助手。"
                        "严格按用户规则输出 JSON，不要输出任何其它内容。"},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 参数与配置
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="B3 三字段补全：P4 定义生成 + P5 类别归类 → B_dict_draft.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_path", type=str, required=True,
                    help="本地 LLM 目录，禁止远程下载")
    ap.add_argument("--backend", type=str, default="vllm",
                    choices=["vllm", "transformers"],
                    help="推理后端（默认 vllm，GLM-4-9B-chat 已验证路径）")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--b2_input", type=str,
                    default="data/concepts/macro_concepts.jsonl")
    ap.add_argument("--b1_input", type=str,
                    default="data/concepts/micro_concepts_raw.jsonl")
    ap.add_argument("--config", type=str, default="configs/dict_b.yaml")
    ap.add_argument("--out_dir", type=str, default="data/concepts")
    ap.add_argument("--cache_root", type=str, default="data/processed/TOXICN")
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
    """配置文件打底（configs/dict_b.yaml 的 b3 段）。"""
    b3 = cfg.get("b3", {}) if isinstance(cfg, dict) else {}
    defaults = {
        "p4_temperature": 0.2,
        "p4_top_p": 0.9,
        "p4_max_new_tokens": 256,    # 40 字定义 + JSON 壳，256 充裕
        "p5_temperature": 0.2,
        "p5_top_p": 0.9,
        "p5_max_new_tokens": 128,    # {"category": "...", "confidence": "高"} 很短
        "retries": 2,                # JSON 解析失败重试次数（§9 失败模式 2）
        "p4_batch_size": 16,         # 每次 chat_batch 的概念数（vllm 连续批处理）
        "n_spans": 20,               # P4 证据片段条数上限
        "span_min_len": 2,           # 过滤单字触发片段（"她"/"女"这类无信息量）
        "other_share_warn": 0.15,    # "其它"占比告警阈值（§5.1）
        "def_max_len": 40,           # 定义长度 lint（方案 §5.2 要求 <=40 字）
        "seed": 729,
    }
    for k in defaults:
        if k in b3:
            defaults[k] = b3[k]
    if "seed" in cfg:
        defaults["seed"] = cfg["seed"]
    return defaults


# ---------------------------------------------------------------------------
# Step1：证据片段收集
# ---------------------------------------------------------------------------

def collect_member_spans(macro: dict, raw_index: dict[str, dict],
                         par: dict) -> list[str]:
    """宏概念 → P4 证据片段：成员按 freq 降序，各取前 2 个 text_span，
    过滤单字片段、去重、截断 n_spans 条。"""
    members = macro.get("members", []) or [macro["concept"]]
    # 成员按其自身 freq 降序（高频成员的片段更能代表典型用法）
    mem_freq = [(m, raw_index.get(m, {}).get("freq", 0)) for m in members]
    mem_freq.sort(key=lambda x: -x[1])
    min_len = int(par["span_min_len"])
    spans: list[str] = []
    seen: set[str] = set()
    for m, _ in mem_freq:
        raw = raw_index.get(m, {})
        cnt = 0
        for sp in raw.get("text_spans", []):
            sp = str(sp).strip()
            if len(sp) < min_len or sp in seen:
                continue
            seen.add(sp)
            spans.append(sp)
            cnt += 1
            if cnt >= 2:
                break
        if len(spans) >= int(par["n_spans"]):
            break
    # 兜底：一条片段都找不到（成员不在 B1 原始表中）→ 用成员名本身
    if not spans:
        spans = members[:int(par["n_spans"])]
    return spans[:int(par["n_spans"])]


# ---------------------------------------------------------------------------
# Step2：P4 定义生成（带重试与缓存）
# ---------------------------------------------------------------------------

def run_p4(macros: list[dict], spans_map: dict[str, list[str]],
           llm: LLMBackend, cache: JsonlCache, par: dict,
           on_progress=None) -> dict[str, str]:
    """返回 concept -> definition（仅成功者）。缓存键 = 概念+片段集 md5。"""
    keys: dict[str, str] = {}
    for m in macros:
        c = m["concept"]
        keys[c] = hashlib.md5(
            (c + "\x1f" + json.dumps(spans_map[c], ensure_ascii=False)
             ).encode("utf-8")).hexdigest()

    defs: dict[str, str] = {}
    todo: list[str] = []
    for m in macros:
        c = m["concept"]
        raw = cache.get(keys[c])
        obj = extract_json(raw) if raw is not None else None
        if isinstance(obj, dict) and str(obj.get("definition", "")).strip():
            defs[c] = str(obj["definition"]).strip()
            if on_progress:
                on_progress(f"{c}(缓存)")
        else:
            todo.append(c)

    bs = int(par["p4_batch_size"])
    for attempt in range(int(par["retries"]) + 1):
        if not todo:
            break
        nxt: list[str] = []
        for st in range(0, len(todo), bs):
            chunk = todo[st:st + bs]
            msgs = [build_p4_messages(c, spans_map[c]) for c in chunk]
            raws = llm.chat_batch(msgs, par["p4_temperature"], par["p4_top_p"],
                                  par["p4_max_new_tokens"], bs, desc="P4定义",
                                  progress=False)
            for c, raw in zip(chunk, raws):
                obj = extract_json(raw)
                d = str(obj.get("definition", "")).strip() \
                    if isinstance(obj, dict) else ""
                if d:
                    cache.put(keys[c], raw)
                    defs[c] = d
                    if on_progress:
                        on_progress(c)
                else:
                    nxt.append(c)
        todo = nxt
        if todo:
            print(f"[B3][P4][警告] {len(todo)} 条定义解析失败，重试 "
                  f"{attempt + 1}/{par['retries']}")
    for c in todo:
        print(f"[B3][P4][失败] 概念 {c} 定义生成失败（本轮弃用，重跑可补齐）")
        if on_progress:
            on_progress(f"{c}(失败)")   # 失败单元也计入总进度条，保证 total 可达
    return defs


# ---------------------------------------------------------------------------
# Step3：P5 类别归类（带重试与缓存）
# ---------------------------------------------------------------------------

def run_p5(concepts: list[str], defs: dict[str, str], llm: LLMBackend,
           cache: JsonlCache, par: dict, on_progress=None) -> dict[str, dict]:
    """返回 concept -> {"category": str, "confidence": str, "fallback": bool}。"""
    keys = {c: hashlib.md5((c + "\x1f" + defs[c]).encode("utf-8")).hexdigest()
            for c in concepts}

    results: dict[str, dict] = {}
    todo: list[str] = []
    for c in concepts:
        raw = cache.get(keys[c])
        obj = extract_json(raw) if raw is not None else None
        if isinstance(obj, dict) and str(obj.get("category", "")).strip() in CATEGORY_NAMES:
            conf = str(obj.get("confidence", "中")).strip()
            results[c] = {"category": str(obj["category"]).strip(),
                          "confidence": conf if conf in ("高", "中", "低") else "中",
                          "fallback": False}
            if on_progress:
                on_progress(f"{c}(缓存)")
        else:
            todo.append(c)

    bs = int(par["p4_batch_size"])
    for attempt in range(int(par["retries"]) + 1):
        if not todo:
            break
        nxt: list[str] = []
        for st in range(0, len(todo), bs):
            chunk = todo[st:st + bs]
            msgs = [build_p5_messages(c, defs[c]) for c in chunk]
            raws = llm.chat_batch(msgs, par["p5_temperature"], par["p5_top_p"],
                                  par["p5_max_new_tokens"], bs, desc="P5归类",
                                  progress=False)
            for c, raw in zip(chunk, raws):
                obj = extract_json(raw)
                if isinstance(obj, dict) and \
                        str(obj.get("category", "")).strip() in CATEGORY_NAMES:
                    conf = str(obj.get("confidence", "中")).strip()
                    results[c] = {"category": str(obj["category"]).strip(),
                                  "confidence": conf if conf in ("高", "中", "低") else "中",
                                  "fallback": False}
                    cache.put(keys[c], raw)
                    if on_progress:
                        on_progress(c)
                else:
                    nxt.append(c)
        todo = nxt
        if todo:
            print(f"[B3][P5][警告] {len(todo)} 条归类非法/解析失败，重试 "
                  f"{attempt + 1}/{par['retries']}")
    # 兜底：重试后仍失败 → 归"其它"+低置信（监控类，不污染七类），不写缓存
    for c in todo:
        results[c] = {"category": "其它", "confidence": "低", "fallback": True}
        print(f"[B3][P5][失败] 概念 {c} 归类兜底为\"其它\"（confidence=低，重跑可补齐）")
        if on_progress:
            on_progress(f"{c}(兜底)")
    return results


# ---------------------------------------------------------------------------
# Step4：落盘
# ---------------------------------------------------------------------------

def write_review(path: Path, macros: list[dict], defs: dict[str, str],
                 cats: dict[str, dict], par: dict, stats: dict) -> None:
    """b3_review.md：§8 审定点 2 的只读材料。"""
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORY_NAMES}
    for m in macros:
        c = m["concept"]
        if c in defs and c in cats:
            by_cat[cats[c]["category"]].append(m)

    lines = ["# B3 类别体系冻结确认材料（只读审阅，§8 审定点 2）", ""]
    lines.append(f"- 宏概念总数: {len(macros)} | 成功补全: {stats['n_ok']} | "
                 f"P4 失败: {stats['n_p4_fail']} | P5 兜底: {stats['n_p5_fallback']}")
    lines.append(f"- \"其它\"占比: {stats['other_share']:.1%} "
                 f"(告警阈值 {par['other_share_warn']:.0%} → "
                 f"{'⚠️ 触发体系修订检查' if stats['other_share'] > par['other_share_warn'] else '未触发'})")
    lines.append("")
    lines.append("| 类别 | 数量 | 概念（按 freq_agg 降序，最多 20） |")
    lines.append("|---|---|---|")
    for cat in CATEGORY_NAMES:
        rows = sorted(by_cat[cat], key=lambda m: -m["freq_agg"])
        names = "、".join(f"{m['concept']}({m['freq_agg']})" for m in rows[:20])
        more = f" …等共 {len(rows)} 个" if len(rows) > 20 else ""
        lines.append(f"| {cat} | {len(rows)} | {names}{more} |")
    lines.append("")
    lines.append("## 类别操作定义（写入提示词的版本，冻结后不改）")
    lines.append("")
    for cat, d in CATEGORY_DEFS.items():
        lines.append(f"- **{cat}**: {d}")
    lines.append("")
    lines.append("## 争议清单（P5 confidence=低，默认按自动规则执行，意见只记录不回改）")
    lines.append("")
    low = [(m, cats[m["concept"]]) for m in macros
           if m["concept"] in cats and cats[m["concept"]]["confidence"] == "低"]
    if low:
        for m, cinfo in low:
            lines.append(f"- {m['concept']} → {cinfo['category']} "
                         f"(freq_agg={m['freq_agg']}, 定义: {defs[m['concept']]})")
    else:
        lines.append("- （无）")
    lines.append("")
    lines.append("## 定义风格 lint（>40 字或含循环句式，仅标记不删除）")
    lines.append("")
    flags = stats.get("def_style_flags", [])
    if flags:
        for c, why in flags:
            lines.append(f"- {c}: {why}（定义: {defs[c]}）")
    else:
        lines.append("- （无）")
    path.write_text("\n".join(lines), encoding="utf-8")


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

    # ---- Step1 读取 B2 宏概念 + B1 原始片段 ----
    macros = [json.loads(l) for l in open(args.b2_input, encoding="utf-8") if l.strip()]
    raw_rows = [json.loads(l) for l in open(args.b1_input, encoding="utf-8") if l.strip()]
    raw_index = {r["concept"]: r for r in raw_rows}
    print(f"[B3] 宏概念: {len(macros)} 个 | B1 原始微概念: {len(raw_rows)} 条")

    spans_map = {m["concept"]: collect_member_spans(m, raw_index, par)
                 for m in macros}
    # 真正无证据的宏概念：成员全部不在 B1 原始表（此时 P4 用成员名兜底）
    n_no_evidence = sum(
        1 for m in macros
        if not any(mem in raw_index and raw_index[mem].get("text_spans")
                   for mem in (m.get("members") or [m["concept"]])))
    if n_no_evidence:
        print(f"[B3][提示] {n_no_evidence} 个宏概念的成员均未在 B1 原始表中找到片段，"
              f"P4 用成员名兜底")

    # ---- 总进度条（唯一进度条：P4逐概念 → P5逐概念 → 落盘）----
    # chat_batch 内置进度条已通过 progress=False 关闭
    bar = tqdm(total=len(macros) * 2 + 1, desc="B3总进度", unit="step",
               ncols=100, miniters=1)

    # ---- Step2 P4 定义 ----
    bar.set_postfix_str("加载LLM")
    llm_name = Path(args.model_path).expanduser().resolve().name
    cache_dir = Path(args.cache_root) / llm_name / "concepts"
    p4_cache = JsonlCache(cache_dir / "p4_def_cache.jsonl")
    llm = LLMBackend(args.model_path, args.backend, args.device, args.dtype,
                     seed=seed, gpu_memory_utilization=args.gpu_memory_utilization,
                     max_model_len=args.max_model_len)

    def _p4_progress(stage: str) -> None:
        bar.update(1)
        bar.set_postfix_str(f"P4定义 {stage}")

    defs = run_p4(macros, spans_map, llm, p4_cache, par, on_progress=_p4_progress)

    # ---- Step3 P5 归类 ----
    p5_cache = JsonlCache(cache_dir / "p5_cat_cache.jsonl")
    ok_concepts = [m["concept"] for m in macros if m["concept"] in defs]

    def _p5_progress(stage: str) -> None:
        bar.update(1)
        bar.set_postfix_str(f"P5归类 {stage}")

    cats = run_p5(ok_concepts, defs, llm, p5_cache, par, on_progress=_p5_progress)

    # ---- Step4 落盘 ----
    bar.set_postfix_str("落盘")
    n_p4_fail = len(macros) - len(defs)
    n_p5_fallback = sum(1 for v in cats.values() if v.get("fallback"))
    cat_counter = Counter(v["category"] for v in cats.values())
    other_share = cat_counter.get("其它", 0) / len(cats) if cats else 0.0

    # 定义风格 lint（仅标记）
    style_flags: list[tuple[str, str]] = []
    for c, d in defs.items():
        why = None
        if len(d) > int(par["def_max_len"]):
            why = f"定义 {len(d)} 字 > {par['def_max_len']}"
        elif "的概念" in d or d.startswith("指……"):
            why = "疑似循环句式"
        if why:
            style_flags.append((c, why))

    # B_dict_draft.csv（utf-8-sig，Excel 直开不乱码）
    csv_path = out_dir / "B_dict_draft.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "definition", "category", "confidence",
                    "freq_agg", "type_majority", "source", "members",
                    "def_style_flag"])
        for m in macros:                      # 保持 B2 的 freq 降序
            c = m["concept"]
            if c not in defs:
                continue                      # P4 失败者不进词典草案
            ci = cats[c]
            w.writerow([c, defs[c], ci["category"], ci["confidence"],
                        m["freq_agg"], m["type_majority"], m["source"],
                        "|".join(m["members"]),
                        dict(style_flags).get(c, "")])

    stats = {
        "stage": "B3",
        "elapsed_sec": round(time.time() - t0, 1),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "backend": args.backend,
        "params": {k: par[k] for k in (
            "p4_temperature", "p4_top_p", "p4_max_new_tokens", "p5_temperature",
            "p5_top_p", "p5_max_new_tokens", "retries", "p4_batch_size",
            "n_spans", "other_share_warn", "def_max_len", "seed")},
        "input": {"n_macro": len(macros)},
        "p4": {"n_ok": len(defs), "n_fail": n_p4_fail,
               "n_style_flag": len(style_flags)},
        "p5": {"n_ok": len(cats), "n_fallback": n_p5_fallback,
               "category_dist": dict(cat_counter),
               "other_share": round(other_share, 4),
               "other_share_exceeded": other_share > par["other_share_warn"],
               "n_low_confidence": sum(1 for v in cats.values()
                                       if v["confidence"] == "低")},
    }
    (out_dir / "b3_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    write_review(out_dir / "b3_review.md", macros, defs, cats, par, stats)

    bar.update(1)
    bar.set_postfix_str("完成")
    bar.close()

    print(f"\n[B3] 完成: 草案词典 {len(defs)} 条 → {csv_path}")
    print(f"[B3] 类别分布: {dict(cat_counter)}")
    if other_share > par["other_share_warn"]:
        print(f"[B3][警告] \"其它\"占比 {other_share:.1%} > "
              f"{par['other_share_warn']:.0%}，触发类别体系修订检查（§5.1）")
    print(f"[B3] P4失败 {n_p4_fail} | P5兜底 {n_p5_fallback} | "
          f"低置信 {stats['p5']['n_low_confidence']} | 用时 {stats['elapsed_sec']}s")
    print(f"[B3] 产物: {out_dir}/B_dict_draft.csv, b3_stats.json, b3_review.md"
          f"（b3_review.md 供导师只读冻结确认）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
