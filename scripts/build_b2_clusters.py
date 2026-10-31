# -*- coding: utf-8 -*-
"""
build_b2_clusters.py — 词典B构建 · B2 聚类与宏概念命名
（实施细则：docs/词典B全自动构建方案.md §4；执行清单 §10 第 3 步）

流程（对应方案 §4，全部自动，零人工标注）：
  Step1  读取 B1 产物 micro_concepts_raw.jsonl，按 min_cluster_freq 分流：
         freq>=2 → 聚类输入；freq=1 → 不编码，存档 singleton_pool.json（供 B5/P7 查重）。
  Step2  bge-m3 编码概念名 → UMAP（n_neighbors=15, min_dist=0.0, metric=cosine）
         → HDBSCAN（min_cluster_size=5, 允许 noise）。
         依赖降级：umap-learn 缺失时回退"原始嵌入 + 余弦预计算距离"路径（573×573
         规模下 HDBSCAN(precomputed) 开销可忽略），并打印降级警告。
  Step3  P3 簇命名：每簇取距质心（原始嵌入空间）最近 m=15 个成员，LLM 归纳上位
         概念（≤2 个，允许标离群）。提示词与方案 §4 逐字一致。
  Step4  分流落盘：
         - 簇 → 宏概念（members, type_majority, freq_agg）→ macro_concepts.jsonl；
         - 未归簇成员（HDBSCAN noise ∪ P3 离群）按频次分流：freq_agg >= tau_freq
           → 单成员宏概念（沿用词面）直接进 B3；否则 → singleton_pool.json 存档。
           "宁可漏、不可滥"：noise 不强行归簇是抗坍缩的源头闸门。

产物：
  data/concepts/macro_concepts.jsonl       每行 {concept, members[], type_majority,
                                           freq_agg, source: cluster|single}
  data/concepts/micro_concepts_clustered.jsonl  全部 freq>=2 微概念 + cluster_id/
                                           macro_concept 标注（审计用）
  data/concepts/singleton_pool.json        弃置单例存档（freq=1 ∪ 低频 noise/离群）
  data/concepts/b2_stats.json              统计
  data/processed/TOXICN/{llm}/concepts/p3_name_cache.jsonl  P3 命名缓存（断点续跑）

复用：B0 的 BGEM3Encoder（transformers 直载 bge-m3）、B1 的 LLMBackend（vllm 默认
后端、强离线、预检自愈）/extract_json/JsonlCache/enforce_offline。

离线硬约束（红线）：同 B1——全程 HF_HUB_OFFLINE=1 / local_files_only=True，
仅从本地路径加载模型，任何缺失只报错、绝不联网。

用法（用户执行，本脚本不由 AI 运行）：
  conda activate toxic_env   # 服务器环境
  python scripts/build_b2_clusters.py --model_path /root/autodl-tmp/glm-4-9b-chat \
      --encoder_path /root/autodl-tmp/bge-m3
  # 依赖 umap-learn（缺失自动降级并告警）；建议 pip install umap-learn
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_b0_sample_pool import BGEM3Encoder          # noqa: E402
from build_b1_micro_concepts import (                  # noqa: E402
    LLMBackend, JsonlCache, enforce_offline, extract_json,
)


# ---------------------------------------------------------------------------
# 参数与配置
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="B2 聚类与宏概念命名：bge-m3 编码 + UMAP + HDBSCAN + P3 命名",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_path", type=str, required=True,
                    help="本地 LLM 目录（P3 命名用），禁止远程下载")
    ap.add_argument("--encoder_path", type=str, default="models/bge-m3",
                    help="本地 bge-m3 目录（概念编码用）")
    ap.add_argument("--backend", type=str, default="vllm",
                    choices=["vllm", "transformers"],
                    help="推理后端（默认 vllm，GLM-4-9B-chat 已验证路径）")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--b1_input", type=str, default="data/concepts/micro_concepts_raw.jsonl")
    ap.add_argument("--config", type=str, default="configs/dict_b.yaml")
    ap.add_argument("--out_dir", type=str, default="data/concepts")
    ap.add_argument("--cache_root", type=str, default="data/processed/TOXICN")
    ap.add_argument("--encoder_device", type=str, default="auto",
                    help="bge-m3 设备：auto/cuda/cpu")
    ap.add_argument("--encoder_max_length", type=int, default=64,
                    help="概念名很短，64 token 足够")
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
    """配置文件打底、CLI 显式传参覆盖。"""
    b2 = cfg.get("b2", {}) if isinstance(cfg, dict) else {}
    b4 = cfg.get("b4", {}) if isinstance(cfg, dict) else {}
    defaults = {
        "min_cluster_freq": 2,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "hdbscan_min_cluster_size": 5,
        "name_top_m": 15,
        "tau_freq": 5,           # 单成员宏概念的频次门槛（与 B4 的 F4 同源）
        "seed": 729,
        "p3_temperature": 0.2,
        "p3_top_p": 0.9,
        "p3_max_new_tokens": 800,
    }
    mapping = {
        "min_cluster_freq": "min_cluster_freq",
        "umap_n_neighbors": "umap_n_neighbors",
        "umap_min_dist": "umap_min_dist",
        "umap_metric": "umap_metric",
        "hdbscan_min_cluster_size": "hdbscan_min_cluster_size",
        "name_top_m": "name_top_m",
    }
    for yk, dk in mapping.items():
        if yk in b2:
            defaults[dk] = b2[yk]
    if "tau_freq" in b4:
        defaults["tau_freq"] = b4["tau_freq"]
    if "seed" in cfg:
        defaults["seed"] = cfg["seed"]
    # CLI 未在 b2 层提供覆盖项，均以配置文件为准
    return defaults


# ---------------------------------------------------------------------------
# Step2：聚类（UMAP → HDBSCAN，含依赖降级）
# ---------------------------------------------------------------------------


def cluster_embeddings(emb_norm: "np.ndarray", par: dict) -> tuple["np.ndarray", dict]:
    """返回 (labels, info)。降级路径：umap-learn 缺失 → 原始嵌入 + 余弦预计算距离。"""
    import numpy as np

    info: dict = {"umap_used": False, "hdbscan_impl": None}
    X = emb_norm
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=par["umap_n_neighbors"], min_dist=par["umap_min_dist"],
            metric=par["umap_metric"], random_state=par["seed"])
        X = np.asarray(reducer.fit_transform(emb_norm), dtype=np.float32)
        info["umap_used"] = True
    except ImportError:
        print("[B2][警告] 未安装 umap-learn（pip install umap-learn），"
              "降级为原始 bge-m3 嵌入 + 余弦预计算距离聚类。"
              "降级路径的簇结构与 UMAP 版可能有差异，建议安装后重跑")

    # HDBSCAN：优先独立 hdbscan 库，缺则用 sklearn >= 1.3 内置实现；
    # 降级路径（无 UMAP）统一用余弦预计算距离——高维原始嵌入 + euclidean 是坏组合
    labels = None
    if info["umap_used"]:
        try:
            import hdbscan  # type: ignore
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=par["hdbscan_min_cluster_size"], metric="euclidean")
            labels = clusterer.fit_predict(X)
            info["hdbscan_impl"] = "hdbscan"
        except ImportError:
            from sklearn.cluster import HDBSCAN as SkHDBSCAN  # noqa: N813
            clusterer = SkHDBSCAN(
                min_cluster_size=par["hdbscan_min_cluster_size"], metric="euclidean")
            labels = clusterer.fit_predict(X)
            info["hdbscan_impl"] = "sklearn"
    else:
        sim = emb_norm @ emb_norm.T
        dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)
        np.fill_diagonal(dist, 0.0)
        try:
            import hdbscan  # type: ignore
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=par["hdbscan_min_cluster_size"],
                metric="precomputed")
            labels = clusterer.fit_predict(dist)
            info["hdbscan_impl"] = "hdbscan(precomputed)"
        except ImportError:
            from sklearn.cluster import HDBSCAN as SkHDBSCAN  # noqa: N813
            clusterer = SkHDBSCAN(
                min_cluster_size=par["hdbscan_min_cluster_size"],
                metric="precomputed")
            labels = clusterer.fit_predict(dist)
            info["hdbscan_impl"] = "sklearn(precomputed)"
    return np.asarray(labels), info


def pick_nearest_members(names: list[str], emb_norm: "np.ndarray",
                         member_idx: list[int], m: int) -> list[str]:
    """取距簇质心（原始嵌入空间）最近的前 m 个成员名。"""
    import numpy as np
    sub = emb_norm[member_idx]
    centroid = sub.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    sims = sub @ centroid
    order = np.argsort(-sims)[:m]
    return [names[member_idx[i]] for i in order]


# ---------------------------------------------------------------------------
# Step3：P3 簇命名
# ---------------------------------------------------------------------------

P3_SYSTEM = ("你是社交媒体有害言论研究的概念归纳助手。严格按用户规则输出 JSON，"
             "不要输出任何其它内容。")


def build_p3_messages(members: list[str]) -> list[dict]:
    members_json = json.dumps(members, ensure_ascii=False)
    user = (
        "下面是从社交媒体文本中聚出的同一组相关概念（均为名词性）。请归纳一个上位概念名，规则：\n"
        "1. 上位概念名必须是 2-8 字的通用名词，能覆盖组内 ≥80% 的成员；\n"
        "2. 若组内存在两类明显不同的语义（例如既有\"地域称谓\"又有\"职业称谓\"），"
        "允许输出最多 2 个上位概念，并对每个成员标注归属；若一个成员两边都不属于，标为\"离群\"；\n"
        "3. 只输出 JSON：{\"names\": [{\"name\": \"...\", \"members\": [\"...\"]}], "
        "\"outliers\": [\"...\"]}。\n"
        f"组内概念：{members_json}"
    )
    return [{"role": "system", "content": P3_SYSTEM},
            {"role": "user", "content": user}]


def p3_name_clusters(clusters: dict[int, list[int]], names: list[str],
                     emb_norm: "np.ndarray", llm: LLMBackend,
                     cache: JsonlCache, par: dict,
                     on_progress=None) -> dict[int, dict]:
    """返回 cluster_id -> {"names": [{name, members}], "outliers": [...]}。

    缓存键 = 簇代表成员（排序后）的 md5，成员集变化自动失效；
    缓存值解析失败视同缺失，重新查询。
    on_progress(stage_str)：每完成一簇（含缓存命中）回调一次，供总进度条更新。
    """
    keys: dict[int, str] = {}
    reprs: dict[int, list[str]] = {}
    for cid, member_idx in clusters.items():
        reprs[cid] = pick_nearest_members(names, emb_norm, member_idx,
                                          par["name_top_m"])
        keys[cid] = hashlib.md5(json.dumps(sorted(reprs[cid]),
                                           ensure_ascii=False).encode("utf-8")
                                ).hexdigest()

    results: dict[int, dict] = {}
    todo: list[int] = []
    for cid in clusters:
        raw = cache.get(keys[cid])
        obj = extract_json(raw) if raw is not None else None
        if isinstance(obj, dict) and isinstance(obj.get("names"), list):
            results[cid] = obj
            if on_progress:
                on_progress(f"簇{cid}(缓存)")
        else:
            todo.append(cid)

    for cid in todo:
        msgs = [build_p3_messages(reprs[cid])]
        raws = llm.chat_batch(msgs, par["p3_temperature"], par["p3_top_p"],
                              par["p3_max_new_tokens"], 1, desc="P3簇命名",
                              progress=False)
        cache.put(keys[cid], raws[0])
        obj = extract_json(raws[0])
        if isinstance(obj, dict) and isinstance(obj.get("names"), list):
            results[cid] = obj
        else:
            print(f"[B2][P3][警告] 簇 {cid} 命名解析失败，本轮跳过（可重跑补齐）")
        if on_progress:
            on_progress(f"簇{cid}")
    return results


# ---------------------------------------------------------------------------
# Step4：分流落盘
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

    # ---- Step1 读取 B1 产物并分流 ----
    rows = [json.loads(l) for l in open(args.b1_input, encoding="utf-8") if l.strip()]
    cluster_rows = [r for r in rows if r["freq"] >= par["min_cluster_freq"]]
    singleton_rows = [r for r in rows if r["freq"] < par["min_cluster_freq"]]
    print(f"[B2] B1 输入: {len(rows)} 条 | freq>={par['min_cluster_freq']}: "
          f"{len(cluster_rows)} 条（聚类） | 单例存档: {len(singleton_rows)} 条")

    names = [r["concept"] for r in cluster_rows]
    sample_sets = [set(r.get("sample_ids", [])) for r in cluster_rows]

    # ---- 总进度条（唯一进度条：编码 → 逐簇P3命名 → 分流落盘）----
    # 各子模块（encode/chat_batch）的内置进度条已通过 progress=False 关闭
    bar = tqdm(total=1, desc="B2总进度", unit="step", ncols=100, miniters=1)

    # ---- Step2 编码 + 聚类 ----
    bar.set_postfix_str("bge-m3编码")
    encoder = BGEM3Encoder(args.encoder_path, args.encoder_device,
                           args.encoder_max_length, 256)
    emb = encoder.encode(names, desc="", progress=False)
    bar.update(1)
    labels, cl_info = cluster_embeddings(emb, par)
    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"\n[B2] 聚类完成: 簇 {n_clusters} 个 | noise {n_noise} 条 | "
          f"UMAP={'是' if cl_info['umap_used'] else '否（降级）'} | "
          f"HDBSCAN 实现: {cl_info['hdbscan_impl']}")
    clusters: dict[int, list[int]] = {}
    for i, lb in enumerate(labels.tolist()):
        if lb != -1:
            clusters.setdefault(lb, []).append(i)

    # ---- Step3 P3 命名 ----
    bar.total = 1 + len(clusters) + 1   # 编码已计入，加：逐簇命名 + 落盘
    bar.refresh()
    bar.set_postfix_str("加载LLM")
    llm_name = Path(args.model_path).expanduser().resolve().name
    cache_dir = Path(args.cache_root) / llm_name / "concepts"
    p3_cache = JsonlCache(cache_dir / "p3_name_cache.jsonl")
    llm = LLMBackend(args.model_path, args.backend, args.device, args.dtype,
                     seed=seed, gpu_memory_utilization=args.gpu_memory_utilization,
                     max_model_len=args.max_model_len)

    def _p3_progress(stage: str) -> None:
        bar.update(1)
        bar.set_postfix_str(f"P3命名 {stage}")

    p3_out = p3_name_clusters(clusters, names, emb, llm, p3_cache, par,
                              on_progress=_p3_progress)
    bar.set_postfix_str("分流落盘")

    # ---- Step4 分流 ----
    # 归属映射：微概念名 -> 归属的宏概念名（P3 未标归属的成员 → outlier 分流）
    macro_records: list[dict] = []
    claimed_all: set[int] = set()      # 所有已被宏概念认领的微概念索引
    used_names: set[str] = set()
    dup_names: list[str] = []

    for cid in sorted(clusters):
        out = p3_out.get(cid)
        member_set = set(clusters[cid])
        if not out or not out.get("names"):
            # 命名失败的簇整体按未归簇处理（宁可漏、不可滥）
            continue
        name_to_members: dict[str, list[int]] = {}
        for entry in out["names"]:
            nm = str(entry.get("name", "")).strip()
            if not nm:
                continue
            for mname in entry.get("members", []):
                mname = str(mname).strip()
                # 归一匹配（去空格）容错回显
                hit = None
                for i in member_set:
                    if names[i].replace(" ", "") == mname.replace(" ", ""):
                        hit = i
                        break
                if hit is not None:
                    name_to_members.setdefault(nm, []).append(hit)

        claimed = set()
        for nm, idxs in name_to_members.items():
            claimed.update(idxs)
            if nm in used_names:  # 跨簇同名 → 合并（同名即同义，避免词典键冲突）
                dup_names.append(nm)
                target = next(m for m in macro_records if m["concept"] == nm)
                target["members"].extend(names[i] for i in idxs)
                target["_src_rows"].extend(cluster_rows[i] for i in idxs)
                continue
            used_names.add(nm)
            rows_sub = [cluster_rows[i] for i in idxs]
            macro_records.append({
                "concept": nm,
                "members": [names[i] for i in idxs],
                "type_majority": Counter(r["type"] for r in rows_sub).most_common(1)[0][0],
                "freq_agg": len(set().union(*(set(r["sample_ids"]) for r in rows_sub))),
                "source": "cluster",
                "_src_rows": rows_sub,
            })
        claimed_all |= claimed
        # P3 标注的离群成员（按名字回显匹配）也保持未归簇；未标注归属的成员同理
        out_names = {str(x).replace(" ", "") for x in out.get("outliers", [])
                     if isinstance(x, str)}
        for i in member_set - claimed:
            if names[i].replace(" ", "") not in out_names:
                print(f"[B2][P3][提示] 簇 {cid} 成员 {names[i]} 未被命名结果认领，"
                      f"按未归簇分流")
        # 两种情况（未认领/标离群）均留在未归簇集合，分流规则相同

    # 未归簇成员 = 全部索引 − 已认领；按频次分流
    unassigned = set(range(len(names))) - claimed_all
    tau = int(par["tau_freq"])
    n_single_promoted = 0
    for i in sorted(unassigned):
        if cluster_rows[i]["freq"] >= tau:
            macro_records.append({
                "concept": names[i],
                "members": [names[i]],
                "type_majority": cluster_rows[i]["type"],
                "freq_agg": cluster_rows[i]["freq"],
                "source": "single",
                "_src_rows": [cluster_rows[i]],
            })
            n_single_promoted += 1
        else:
            singleton_rows.append(cluster_rows[i])

    # 合并同名后重算 freq_agg / type_majority（合并会追加 _src_rows）；
    # 并对成员去重（P3 可能把同一成员归到两个名下）
    for m in macro_records:
        m["_src_rows"] = list({r["concept"]: r for r in m["_src_rows"]}.values())
        m["members"] = list(dict.fromkeys(m["members"]))
        m["freq_agg"] = len(set().union(*(set(r["sample_ids"]) for r in m["_src_rows"])))
        m["type_majority"] = Counter(r["type"] for r in m["_src_rows"]).most_common(1)[0][0]

    macro_records.sort(key=lambda m: (-m["freq_agg"], m["concept"]))

    # ---- 落盘 ----
    with open(out_dir / "macro_concepts.jsonl", "w", encoding="utf-8") as f:
        for m in macro_records:
            f.write(json.dumps({
                "concept": m["concept"], "members": m["members"],
                "type_majority": m["type_majority"],
                "freq_agg": m["freq_agg"], "source": m["source"],
            }, ensure_ascii=False) + "\n")

    # 微概念级标注（审计）
    macro_of: dict[str, str] = {}
    for m in macro_records:
        for mem in m["members"]:
            macro_of[mem] = m["concept"]
    with open(out_dir / "micro_concepts_clustered.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(cluster_rows):
            f.write(json.dumps({
                "concept": r["concept"], "freq": r["freq"], "type": r["type"],
                "cluster_id": int(labels[i]), "macro_concept": macro_of.get(r["concept"]),
            }, ensure_ascii=False) + "\n")

    singletons = sorted({r["concept"] for r in singleton_rows})
    with open(out_dir / "singleton_pool.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(singletons), "concepts": singletons,
                   "note": "freq=1 单例 ∪ 低频 noise/P3 离群；唯一用途：B5/P7 产出查重"},
                  f, ensure_ascii=False, indent=2)

    stats = {
        "stage": "B2",
        "elapsed_sec": round(time.time() - t0, 1),
        "model_path": args.model_path,
        "encoder_path": args.encoder_path,
        "backend": args.backend,
        "params": {k: par[k] for k in ("min_cluster_freq", "umap_n_neighbors",
                                       "umap_min_dist", "umap_metric",
                                       "hdbscan_min_cluster_size", "name_top_m",
                                       "tau_freq", "seed")},
        "clustering": {**cl_info, "n_clusters": n_clusters, "n_noise": n_noise},
        "p3": {"n_clusters_named": len(p3_out), "n_parse_fail":
               len(clusters) - len(p3_out), "dup_names_merged": dup_names},
        "merge": {
            "n_macro": len(macro_records),
            "n_macro_from_cluster": sum(1 for m in macro_records if m["source"] == "cluster"),
            "n_macro_single_promoted": n_single_promoted,
            "n_unassigned": len(unassigned),
            "n_singleton_archived": len(singletons),
        },
    }
    with open(out_dir / "b2_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    bar.update(1)
    bar.set_postfix_str("完成")
    bar.close()
    print(f"\n[B2] 完成: 宏概念 {stats['merge']['n_macro']} 个"
          f"（含单成员晋升 {n_single_promoted}） | 单例存档 "
          f"{len(singletons)} | 用时 {stats['elapsed_sec']}s")
    print(f"[B2] 产物: {out_dir}/macro_concepts.jsonl, "
          f"micro_concepts_clustered.jsonl, singleton_pool.json, b2_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
