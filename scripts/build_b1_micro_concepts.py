# -*- coding: utf-8 -*-
"""
build_b1_micro_concepts.py — 词典B构建 · B1 微概念抽取
（实施细则：docs/词典B全自动构建方案.md §3；执行清单 §10 第 2 步）

流程（三步，全部自动，零人工标注）：
  Step1  P1 微概念抽取：对 data/concepts/sample_pool.json 的 1800 条真实样本逐条调用本地 LLM，
         抽取名词性微概念（text_span → concept → type）。JSON 解析失败自动重试（≤p1_retries），
         仍失败记录弃用并统计失败率（>5% 告警，方案 §9.2）。
  Step2  P2 词性闸门（双通道交叉）：
         通道一 = LLM 批量词性判定（每次 p2_batch_terms 条）；
         通道二 = jieba.posseg 本地词性标注（保留 n/nz/ng/vn 等）。
         合并规则（方案 §3.2）：两通道均判形容词/副词 → 删；仅一通道判形容词 → 进争议清单
         （§8 只读审阅），默认按分词工具结果处理。
  Step3  词面归一合并（方案 §3.3）：规范概念名 strip 归一后完全同形合并（计数累加），
         freq = 触及样本数；变体合并留给 B2 聚类，B1 不做 LLM 同义合并。

产物（全部落 data/concepts/ 与缓存目录）：
  data/concepts/micro_concepts_raw.jsonl   每行 {concept, type, text_spans[], sample_ids[],
                                           freq, type_dist, pos_llm, pos_seg_flags,
                                           decision, flags}
  data/concepts/pos_deleted.jsonl          被 P2 闸门删除的概念（审计用）
  data/concepts/pos_conflicts.json         双通道词性冲突争议清单（§8 只读审阅用）
  data/concepts/b1_stats.json              本步统计（失败率/词性分布/类型分布等）
  data/processed/TOXICN/{llm}/concepts/p1_raw_cache.jsonl   P1 原始输出缓存（sample_id → raw）
  data/processed/TOXICN/{llm}/concepts/p2_pos_cache.jsonl   P2 原始输出缓存（term → raw）

复现性：温度 0.2 采样 + 固定种子（同版本模型/同硬件下近似可复现）；缓存命中部分与
上次运行逐字节一致。断点续跑：P1/P2 缓存逐条落盘，中断后重跑只补缺失项。

离线硬约束（红线）：全程禁止任何远程下载/在线调用——脚本启动即设置
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1，所有 from_pretrained 一律
local_files_only=True；模型仅从 --model_path 指定的本地目录加载，任何文件缺失
只明确报错、绝不联网。原生格式（model_type=glm4/glm）零自定义代码执行；
旧版 chatglm 格式仅执行模型目录内的本地自定义代码（该代码随权重分发、存于本地，
trust_remote_code 的 "remote" 是 transformers 术语，非联网行为）。

用法（用户执行，本脚本不由 AI 运行）：
  conda activate toxic_env   # 服务器环境
  python scripts/build_b1_micro_concepts.py --model_path /root/autodl-tmp/glm-4-9b-chat
  # 默认后端 vllm（fscbm 项目验证过的 GLM-4-9B-chat 加载路径——该模型的旧版自定义
  # 代码与新版 transformers 在加载路径上有已知不兼容，transformers 后端仅建议用于
  # 原生格式模型）；小规模试跑加 --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 离线硬约束（先于一切 transformers 导入生效）
# ---------------------------------------------------------------------------

NATIVE_MODEL_TYPES = {"glm4", "glm"}  # transformers 原生支持的 GLM 格式（无需任何自定义代码）


def enforce_offline() -> None:
    """强离线保证：封死 HF/transformers 的一切网络访问暗道。

    设置后，from_pretrained 在本地文件缺失时只会明确报错，绝不会尝试联网下载；
    配合 local_files_only=True 双保险。红线：禁止任何远程下载/在线调用。
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def resolve_model_format(mp: Path) -> tuple[str, bool]:
    """读取 config.json 判定模型格式。返回 (model_type, trust_remote_code)。

    - 原生格式（glm4/glm）：transformers 内置实现，**不执行任何自定义 Python 代码**；
    - 旧版 chatglm 格式：transformers 无内置实现，必须执行**模型目录内的本地**自定义
      代码（trust_remote_code=True 术语辨析："remote" 指代码由模型作者随权重分发，
      并存于本地模型目录，加载与执行均不联网）。
    """
    cfg_file = mp / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {cfg_file}")
    with open(cfg_file, encoding="utf-8") as f:
        mt = (json.load(f) or {}).get("model_type", "")
    return mt, mt not in NATIVE_MODEL_TYPES

# ---------------------------------------------------------------------------
# 常量：提示词（与方案文档 §3.1 / §3.2 逐字一致，不得随意改动）
# ---------------------------------------------------------------------------

P1_SYSTEM = """你是一名中文有害言论研究的语言学标注助手。你的任务是从给定的社交媒体文本中抽取"概念候选"。
概念指文本中涉及的名词性语义单元，包括：攻击对象、群体指称（含蔑称与代称）、侮辱称谓、
行为动作（名词化，如"家暴""人肉搜索"）、刻板印象主题（如"外来者抢工作"）、话题场景（如"地域矛盾"）。
规则：
1. 只输出名词、名词短语或动名词概念；严禁输出形容词、副词（形容词属于另一部形容词词典，不在你的职责内）；
2. 每个概念给出原文触发片段（text_span，必须是原文中出现的连续字串或其明确代称）和一个规范概念名（concept，2-8字通用名词，不保留蔑称的生僻写法，如"黑乐色"→"黑人蔑称"）；
3. 概念标注微类型 type，取值限于：攻击对象/群体指称/侮辱称谓/行为动作/刻板印象/话题场景/其它；
4. 不评判文本是否有害，不输出任何解释，只输出 JSON；
5. 输出格式：{"concepts": [{"text_span": "...", "concept": "...", "type": "..."}]}，无概念则输出 {"concepts": []}。"""

# 4-shot 示例（真实样本，取自 TOXICN train，方案 §3.1；user 侧带"文本："前缀与正式模板一致）
P1_FEWSHOT = [
    ("文本：好久都没有见过那么干净的白人班级了，一个黑乐色都没有\nJSON：",
     '{"concepts": [{"text_span": "白人", "concept": "白人", "type": "群体指称"},\n'
     '              {"text_span": "黑乐色", "concept": "黑人蔑称", "type": "侮辱称谓"}]}'),
    ("文本：估计已经成了非洲某部落的第36号王妃\nJSON：",
     '{"concepts": [{"text_span": "非洲某部落", "concept": "非洲部落", "type": "群体指称"},\n'
     '              {"text_span": "王妃", "concept": "涉外婚姻", "type": "刻板印象"}]}'),
    ("文本：说明一下，我只关爱自己身边的女性，比如老婆女儿等，别的女人关我鸟事。\nJSON：",
     '{"concepts": [{"text_span": "女性", "concept": "女性", "type": "群体指称"},\n'
     '              {"text_span": "别的女人关我鸟事", "concept": "性别对立言论", "type": "话题场景"}]}'),
    ("文本：就凭你的北京户口我就感觉你比我高级五个阶级，我是河北人\U0001F606\nJSON：",
     '{"concepts": [{"text_span": "北京户口", "concept": "户籍", "type": "话题场景"},\n'
     '              {"text_span": "河北人", "concept": "外地人", "type": "群体指称"}]}'),
]

P1_RETRY_HINT = "上次输出无法解析为 JSON。请严格只输出一个 JSON 对象（格式：{\"concepts\": [...]}），不要输出任何其它文字、解释或代码块标记。"

TYPE_ENUM = ["攻击对象", "群体指称", "侮辱称谓", "行为动作", "刻板印象", "话题场景", "其它"]

P2_INSTRUCTION = ("判断下列每个中文词语或短语的词性，只输出 JSON："
                  "{\"results\": [{\"term\": \"...\", \"pos\": \"名词|动名词|形容词|副词|其它\"}]}。"
                  "若一个词既可作名词又可作形容词（如\"恶心\"），按其在有害言论语境中更常用的用法判断。\n"
                  "待判定：{terms_json}")
P2_SYSTEM = "你是中文语言学词性标注助手，只输出 JSON，不输出任何解释。"

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# ---------------------------------------------------------------------------
# 参数与配置
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="B1 微概念抽取：P1 抽取 + P2 词性闸门 + 词面归一合并",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_path", type=str, required=True,
                    help="本地 LLM 目录（如 models/GLM-4-9B-Chat），禁止远程下载")
    ap.add_argument("--backend", type=str, default="vllm",
                    choices=["vllm", "transformers"],
                    help="推理后端。默认 vllm（fscbm 项目验证过的 GLM-4-9B-chat 加载路径，"
                         "对老版自定义代码兼容）；transformers 仅建议用于原生格式模型")
    ap.add_argument("--device", type=str, default="auto",
                    help="transformers 后端设备：auto/cuda/cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                    help="vllm 后端 GPU 显存占用比例（对齐 fscbm 项目）")
    ap.add_argument("--max_model_len", type=int, default=4096,
                    help="vllm 后端最大序列长度。fscbm 用 1024，但本脚本 P1 为 4-shot "
                         "提示词更长，取 4096 留足余量")
    ap.add_argument("--pool_path", type=str, default="data/concepts/sample_pool.json")
    ap.add_argument("--config", type=str, default="configs/dict_b.yaml",
                    help="超参配置文件（CLI 显式传参时以 CLI 为准）")
    ap.add_argument("--out_dir", type=str, default="data/concepts")
    ap.add_argument("--cache_root", type=str, default="data/processed/TOXICN",
                    help="LLM 推理缓存根目录（实际缓存在其下 {llm名}/concepts/）")
    # CLI 覆盖项（默认 None = 用配置文件值）
    ap.add_argument("--p1_batch_size", type=int, default=None)
    ap.add_argument("--p1_temperature", type=float, default=None)
    ap.add_argument("--p1_max_new_tokens", type=int, default=None)
    ap.add_argument("--p1_retries", type=int, default=None)
    ap.add_argument("--p2_batch_terms", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理池内前 N 条（冒烟测试用；正式跑不要加）")
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
    b1 = cfg.get("b1", {}) if isinstance(cfg, dict) else {}
    defaults = {
        "p1_batch_size": 8,
        "p1_temperature": 0.2,
        "p1_top_p": 0.9,
        "p1_max_new_tokens": 320,
        "p1_retries": 2,
        "concept_len_min": 2,
        "concept_len_max": 8,
        "p2_batch_terms": 50,
        "p2_temperature": 0.2,
        "p2_top_p": 0.9,
        "p2_max_new_tokens": 1600,
        "adj_flags_jieba": ["a", "ad", "d"],
    }
    # YAML 键名与 defaults 的映射（concept_len_range 拆两半）
    y2d = {
        "p1_temperature": "p1_temperature", "p1_top_p": "p1_top_p",
        "p1_max_new_tokens": "p1_max_new_tokens", "p1_batch_size": "p1_batch_size",
        "p1_retries": "p1_retries", "p2_batch_terms": "p2_batch_terms",
        "p2_temperature": "p2_temperature", "p2_max_new_tokens": "p2_max_new_tokens",
        "adj_flags_jieba": "adj_flags_jieba",
    }
    for yk, dk in y2d.items():
        if yk in b1:
            defaults[dk] = b1[yk]
    if "p2_top_p" in b1:
        defaults["p2_top_p"] = b1["p2_top_p"]
    rng = b1.get("concept_len_range")
    if isinstance(rng, list) and len(rng) == 2:
        defaults["concept_len_min"], defaults["concept_len_max"] = int(rng[0]), int(rng[1])
    # CLI 覆盖
    for k in ("p1_batch_size", "p1_temperature", "p1_max_new_tokens", "p1_retries",
              "p2_batch_terms"):
        if getattr(args, k, None) is not None:
            defaults[k] = getattr(args, k)
    return defaults


# ---------------------------------------------------------------------------
# JSON 提取（方案 §9.2：解析失败重试 + 正则兜底）
# ---------------------------------------------------------------------------


def extract_json(text: str):
    """从模型输出中尽力提取第一个 JSON 对象；失败返回 None。"""
    if text is None:
        return None
    s = text.strip()
    # 去掉 ```json ... ``` 围栏
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    m = JSON_BLOCK_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# LLM 后端
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 远端代码预检与自愈（防 trust_remote_code 模型与新 transformers 的兼容问题）
# ---------------------------------------------------------------------------


def preflight_remote_code(model_path: str) -> bool:
    """模型格式判定 + 预检 + 已知兼容问题自愈（fail fast）。返回是否执行本地自定义代码。

    仅当模型为旧版 chatglm 格式（transformers 无内置实现）时才执行模型目录内的
    本地自定义代码（trust_remote_code=True）；原生格式零自定义代码。
    全程 local_files_only + 强离线环境变量，任何文件缺失只报错、不联网。
    """
    mp = Path(model_path).expanduser().resolve()
    model_type, trust = resolve_model_format(mp)
    print(f"[预检] model_type={model_type} | 自定义代码执行: "
          f"{'是（仅模型目录内本地代码，全程离线）' if trust else '否（transformers 原生加载）'}"
          f" | 离线模式: HF_HUB_OFFLINE=1")

    if trust:
        # 旧版 ChatGLM 远端（本地分发）代码在新 transformers 上的已知断裂点：
        # config.max_length 已被移除 → 原地修补为 getattr 兜底（原文件备份 .bak，仅首次）
        mfile = mp / "modeling_chatglm.py"
        if mfile.is_file():
            src = mfile.read_text(encoding="utf-8")
            if "config.max_length" in src:
                bak = mfile.with_name(mfile.name + ".bak")
                if not bak.is_file():
                    bak.write_text(src, encoding="utf-8")
                patched = re.sub(
                    r"config\.max_length\b",
                    'getattr(config, "max_length", 8192)',
                    src,
                )
                mfile.write_text(patched, encoding="utf-8")
                print(f"[预检] 已修补本地模型代码的 config.max_length 兼容问题: {mfile}"
                      f"（原文件备份: {bak.name}；8192 兜底对本任务短文本足够）")
        # GLM 系本地分词代码依赖 tiktoken
        if (mp / "tokenization_chatglm.py").is_file():
            try:
                import tiktoken  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "该模型使用 GLM 本地分词代码（tokenization_chatglm.py），其依赖 "
                    "tiktoken，当前环境未安装。请先执行: pip install tiktoken") from e

    # chat 模板冒烟（本脚本用 apply_chat_template 构建全部提示）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=trust,
                                        local_files_only=True)
    try:
        s = tok.apply_chat_template(
            [{"role": "user", "content": "你好"}],
            tokenize=False, add_generation_prompt=True)
    except Exception as e:
        raise RuntimeError(
            "tokenizer.apply_chat_template 不可用（模型目录未内置 chat 模板且模型代码"
            "未提供兼容实现），本脚本依赖 chat 模板构建批量提示。"
            f"原始错误: {e}") from e
    if not s or "你好" not in s:
        raise RuntimeError("chat 模板输出异常（未包含输入内容），请检查 tokenizer 配置")
    return trust


class LLMBackend:
    """统一 chat 批量推理接口：chat_batch(messages_list) -> list[str]。"""

    def __init__(self, model_path: str, backend: str, device: str, dtype: str,
                 seed: int, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 4096):
        self.backend = backend
        self.model_path = str(Path(model_path).expanduser().resolve())
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"模型目录不存在: {self.model_path}")
        self.trust_remote_code = preflight_remote_code(self.model_path)
        import torch
        self.torch = torch
        torch.manual_seed(seed)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code,
            local_files_only=True)
        self.tokenizer.padding_side = "left"  # 批量生成必须左填充，显式设置防模板路径差异
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if backend == "transformers":
            # 旧版 chatglm 格式 + 新版 transformers 的已知加载不兼容（all_tied_weights_keys
            # 等，出现在 from_pretrained 内部，无法在外部修补）——提前警告，推荐 vllm
            cfg_path = Path(self.model_path) / "config.json"
            mtype = ""
            if cfg_path.is_file():
                try:
                    mtype = str(json.loads(cfg_path.read_text(encoding="utf-8"))
                                .get("model_type", ""))
                except (json.JSONDecodeError, OSError):
                    pass
            if mtype == "chatglm":
                print("[警告] 旧版 chatglm 格式（GLM-4-9B-chat 原始分发）的自定义代码与"
                      "新版 transformers 在模型加载路径上存在已知不兼容"
                      "（如 all_tied_weights_keys 缺失），transformers 后端很可能失败。"
                      "推荐 --backend vllm（fscbm 项目验证过的加载路径），"
                      "或改用原生 glm4 格式的模型目录。仍继续尝试加载……")
            from transformers import AutoModelForCausalLM
            dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                  "float32": torch.float32}[dtype]
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            try:  # transformers >= 4.56 用 dtype，旧版用 torch_dtype（已废弃）
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, dtype=dt,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True)
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, torch_dtype=dt,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True)
            self.model = self.model.to(self.device).eval()
            self._smoke_generate()
        elif backend == "vllm":
            try:
                from vllm import LLM
            except ImportError as e:
                raise RuntimeError(
                    "未安装 vllm。GLM-4-9B-chat 这类旧版自定义代码模型在新版 "
                    "transformers 后端存在加载不兼容（all_tied_weights_keys 等），"
                    "fscbm 项目即用 vllm 加载该模型，请安装 vllm 后重试") from e
            # 参数对齐 fscbm/scripts/generate_scbm_concept.py::load_vllm_model
            # （同款 GLM-4-9B-chat 在本项目服务器上的已验证配置）；差异仅两处：
            # max_model_len 1024→4096（本脚本 P1 为 4-shot 提示词，更长）、显式传 seed
            self.llm = LLM(model=self.model_path,
                           trust_remote_code=self.trust_remote_code,
                           dtype="auto",
                           gpu_memory_utilization=gpu_memory_utilization,
                           enable_prefix_caching=True,
                           max_model_len=max_model_len,
                           max_num_seqs=64,
                           max_num_batched_tokens=16384,
                           seed=seed)
            self._smoke_generate()
        else:  # pragma: no cover - argparse 已约束
            raise ValueError(f"未知后端: {backend}")

    def _to_prompt_strings(self, messages_list: list[list[dict]]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
            for msgs in messages_list
        ]

    def _smoke_generate(self) -> None:
        """加载后立即做一次最小生成（fail fast）：远端代码与 transformers 的
        前向/Cache 等深层不兼容问题在启动 10 秒内暴露，而不是跑到批量推理中段才崩。"""
        torch = self.torch
        try:
            outs = self.chat_batch(
                [[{"role": "user", "content": "回复：OK"}]],
                temperature=0.0, top_p=1.0, max_new_tokens=8,
                batch_size=1, desc="smoke")
        except Exception as e:
            raise RuntimeError(
                "模型加载后冒烟生成失败——通常是模型自带的本地自定义代码"
                "（trust_remote_code）与当前 transformers 版本在生成路径上不兼容"
                "（如 Cache 格式变更）。"
                "可尝试: ① --backend vllm；② 为该环境安装与该代码同期版本的"
                " transformers（如 pip install \"transformers==4.46.3\"）。"
                f"原始错误: {e}") from e
        if not outs or not outs[0]:
            raise RuntimeError("冒烟生成返回空文本，请检查模型完整性")

    def chat_batch(self, messages_list: list[list[dict]], temperature: float,
                   top_p: float, max_new_tokens: int, batch_size: int,
                   desc: str = "") -> list[str]:
        if self.backend == "vllm":
            return self._gen_vllm(messages_list, temperature, top_p, max_new_tokens)
        return self._gen_transformers(messages_list, temperature, top_p,
                                      max_new_tokens, batch_size, desc)

    def _gen_transformers(self, messages_list, temperature, top_p, max_new_tokens,
                          batch_size, desc):
        prompts = self._to_prompt_strings(messages_list)
        outs: list[str] = []
        try:
            from tqdm import tqdm
            it = tqdm(range(0, len(prompts), batch_size), desc=desc, ncols=90)
        except ImportError:  # pragma: no cover
            it = range(0, len(prompts), batch_size)
        torch = self.torch
        for start in it:
            batch = prompts[start:start + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=4096, return_tensors="pt").to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                gen = self.model.generate(**enc, **gen_kwargs)
            new_tokens = gen[:, enc.input_ids.shape[1]:]
            texts = self.tokenizer.batch_decode(new_tokens,
                                                skip_special_tokens=True)
            outs.extend(t.strip() for t in texts)
        return outs

    def _gen_vllm(self, messages_list, temperature, top_p, max_new_tokens):
        from vllm import SamplingParams
        prompts = self._to_prompt_strings(messages_list)
        sp = SamplingParams(temperature=temperature, top_p=top_p,
                            max_tokens=max_new_tokens, seed=None)
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text.strip() for o in outs]


# ---------------------------------------------------------------------------
# 缓存（逐条 jsonl，断点续跑）
# ---------------------------------------------------------------------------


class JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, str] = {}
        if self.path.is_file():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["raw"]
        self._fh = open(self.path, "a", encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, raw: str) -> None:
        self.data[key] = raw
        self._fh.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Step1：P1 微概念抽取
# ---------------------------------------------------------------------------


def build_p1_messages(text: str, retry: bool) -> list[dict]:
    msgs = [{"role": "system", "content": P1_SYSTEM}]
    for u, a in P1_FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"文本：{text}\nJSON："})
    if retry:
        msgs.append({"role": "user", "content": P1_RETRY_HINT})
    return msgs


def run_p1(samples: list[dict], llm: LLMBackend, cache: JsonlCache,
           par: dict, limit: int | None) -> tuple[list[dict], dict]:
    """返回 (抽取记录列表, 统计字典)。抽取记录：
    {sample_id, stratum, text, span, concept_raw, type_raw, is_subspan}
    """
    work = samples if limit is None else samples[:limit]
    todo = [s for s in work if cache.get(s["sample_id"]) is None]
    if todo:
        bs = par["p1_batch_size"]
        # 分块处理：每块生成→重试→立即落缓存；中途崩溃最多损失一块的推理
        chunk_size = max(bs * 8, 64)
        texts_map = {s["sample_id"]: s["text"] for s in work}
        for cstart in range(0, len(todo), chunk_size):
            chunk = todo[cstart:cstart + chunk_size]
            results: dict[str, str] = {}
            ids = [s["sample_id"] for s in chunk]
            msgs = [build_p1_messages(s["text"], False) for s in chunk]
            raws = llm.chat_batch(msgs, par["p1_temperature"], par["p1_top_p"],
                                  par["p1_max_new_tokens"], bs, desc="P1抽取")
            for sid, raw in zip(ids, raws):
                results[sid] = raw
            # 对解析失败的做重试（逐条，最多 p1_retries 轮）
            need_retry = [sid for sid in ids if extract_json(results[sid]) is None]
            for attempt in range(1, par["p1_retries"] + 1):
                if not need_retry:
                    break
                retry_msgs = [build_p1_messages(texts_map[sid], True)
                              for sid in need_retry]
                retry_raws = llm.chat_batch(retry_msgs, par["p1_temperature"],
                                            par["p1_top_p"], par["p1_max_new_tokens"],
                                            bs, desc=f"P1重试r{attempt}")
                still = []
                for sid, raw in zip(need_retry, retry_raws):
                    results[sid] = raw
                    if extract_json(raw) is None:
                        still.append(sid)
                need_retry = still
            for sid, raw in results.items():
                cache.put(sid, raw)

    # 解析与校验（缓存命中也在此统一解析，解析逻辑改进无需重跑推理）
    records: list[dict] = []
    n_ok = n_fail = 0
    for s in work:
        raw = cache.get(s["sample_id"])
        if raw is None:
            continue  # limit 截断之外
        obj = extract_json(raw)
        if not isinstance(obj, dict) or not isinstance(obj.get("concepts"), list):
            n_fail += 1
            continue
        n_ok += 1
        text = s["text"]
        for item in obj["concepts"]:
            if not isinstance(item, dict):
                continue
            span = str(item.get("text_span", "")).strip()
            concept = str(item.get("concept", "")).strip()
            typ = str(item.get("type", "其它")).strip()
            if not concept:  # span 缺失可容忍（concept 为锚），span 缺则记空
                continue
            records.append({
                "sample_id": s["sample_id"],
                "stratum": s["stratum"],
                "text": text,
                "span": span,
                "concept_raw": concept,
                "type_raw": typ if typ in TYPE_ENUM else "其它",
                "is_subspan": bool(span) and span in text,
            })
    stats = {
        "n_input": len(work),
        "n_json_ok": n_ok,
        "n_json_fail": n_fail,
        "json_fail_rate": round(n_fail / max(n_ok + n_fail, 1), 4),
        "n_records": len(records),
    }
    if stats["json_fail_rate"] > 0.05:
        print(f"[B1][警告] P1 JSON 解析失败率 {stats['json_fail_rate']:.1%} > 5%"
              "（方案 §9.2 失败模式2），建议检查模型或提高重试次数")
    return records, stats


# ---------------------------------------------------------------------------
# Step2：P2 词性闸门
# ---------------------------------------------------------------------------


def run_p2_llm(terms: list[str], llm: LLMBackend, cache: JsonlCache,
               par: dict) -> dict[str, str]:
    """返回 term -> pos（名词/动名词/形容词/副词/其它；两轮重试后仍失败的 term 不在返回字典中）。"""

    def query_chunk(chunk: list[str]) -> None:
        terms_json = json.dumps(chunk, ensure_ascii=False)
        msgs = [[{"role": "system", "content": P2_SYSTEM},
                 {"role": "user",
                  "content": P2_INSTRUCTION.replace("{terms_json}", terms_json)}]]
        raws = llm.chat_batch(msgs, par["p2_temperature"], par["p2_top_p"],
                              par["p2_max_new_tokens"], 1, desc="P2词性LLM")
        obj = extract_json(raws[0])
        got: dict[str, str] = {}
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            for it in obj["results"]:
                if isinstance(it, dict) and str(it.get("term", "")).strip():
                    got[str(it["term"]).strip()] = str(it.get("pos", "其它")).strip()
        # 只缓存成功判定且命中请求词条的项；缺失项进入重试轮（不缓存失败）
        for t in chunk:
            if t in got:
                cache.put(t, got[t])

    bs = par["p2_batch_terms"]
    todo = [t for t in terms if cache.get(t) is None]
    for start in range(0, len(todo), bs):
        query_chunk(todo[start:start + bs])

    # 首轮未获判定的词条（模型漏答/解析失败）自动重试一轮
    missing = [t for t in terms if cache.get(t) is None]
    if missing:
        print(f"[B1][P2] {len(missing)} 条词条首轮未获词性判定，重试一轮")
        for start in range(0, len(missing), bs):
            query_chunk(missing[start:start + bs])

    # 读回：从缓存原始文本中解析出每个词条的词性
    out = {}
    for t in terms:
        raw = cache.get(t)
        if raw is None:
            continue
        obj = extract_json(raw)
        pos = None
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            for it in obj["results"]:
                if isinstance(it, dict) and str(it.get("term", "")).strip() == t:
                    pos = str(it.get("pos", "")).strip()
                    break
        if pos:
            out[t] = pos
    return out


def jieba_pos_flags(term: str, adj_flags: set[str]) -> tuple[bool, list[str]]:
    """通道二：jieba.posseg 词性标注。
    返回 (是否形容词侧, 原始 flags)。形容词侧 = 所有 token 的 flag 均在 adj_flags 中。"""
    import jieba.posseg as pseg
    words = [w for w in pseg.cut(term) if w.word.strip()]
    flags = [w.flag for w in words]
    if not flags:
        return False, ["x"]  # 无法切分 → 按名词侧保留（宁可漏不可滥）
    return all(f in adj_flags for f in flags), flags


ADJ_POS_LLM = {"形容词", "副词"}


def pos_gate(concepts: list[str], llm: LLMBackend, p2_cache: JsonlCache,
             par: dict, out_dir: Path) -> tuple[dict[str, dict], dict]:
    """对去重后的概念列表执行双通道闸门。
    返回 (term -> 判定信息, 统计)。判定信息：
    {pos_llm, seg_adjish, seg_flags, decision: keep|deleted_pos, conflict: bool}
    """
    pos_map = run_p2_llm(concepts, llm, p2_cache, par)
    adj_flags = set(par["adj_flags_jieba"])
    decisions: dict[str, dict] = {}
    stats = Counter()
    for t in concepts:
        pos_llm = pos_map.get(t)  # None = LLM 通道失败
        seg_adjish, seg_flags = jieba_pos_flags(t, adj_flags)
        llm_adjish = pos_llm in ADJ_POS_LLM if pos_llm is not None else None
        if llm_adjish is None:
            # LLM 通道失败：只有分词通道可用 → 按 seg 处理（§3.2 默认规则），seg 判形容词则删并记争议
            conflict = seg_adjish
            decision = "deleted_pos" if seg_adjish else "keep"
        elif llm_adjish and seg_adjish:
            decision, conflict = "deleted_pos", False   # 两通道均判形容词/副词 → 删
        elif llm_adjish != seg_adjish:
            # 仅一通道判形容词 → 争议清单，默认按分词工具结果处理（§3.2）
            conflict = True
            decision = "deleted_pos" if seg_adjish else "keep"
        else:
            # 两通道一致判名词侧 → 保留
            decision, conflict = "keep", False
        decisions[t] = {"pos_llm": pos_llm, "seg_adjish": seg_adjish,
                        "seg_flags": seg_flags, "decision": decision,
                        "conflict": conflict}
        stats[decision] += 1
        if conflict:
            stats["conflict"] += 1

    # 争议清单落盘（§8 只读审阅）
    conflicts = [{"term": t, **d} for t, d in decisions.items() if d["conflict"]]
    with open(out_dir / "pos_conflicts.json", "w", encoding="utf-8") as f:
        json.dump(conflicts, f, ensure_ascii=False, indent=2)
    return decisions, dict(stats)


# ---------------------------------------------------------------------------
# Step3：词面归一合并（方案 §3.3）
# ---------------------------------------------------------------------------

_STRIP_CHARS = " \u3000\t\n「」\"'\u201c\u201d\u2018\u2019。.,，、;；:：!！?？"


def normalize_concept(name: str) -> str:
    return name.strip(_STRIP_CHARS)


def merge_records(records: list[dict], decisions: dict[str, dict],
                  par: dict) -> tuple[list[dict], list[dict], dict]:
    """同形合并 → 闸门过滤。返回 (kept 行, deleted 行, 统计)。"""
    lo, hi = par["concept_len_min"], par["concept_len_max"]
    grouped: dict[str, dict] = {}
    for r in records:
        name = normalize_concept(r["concept_raw"])
        if not name:
            continue
        g = grouped.setdefault(name, {
            "concept": name, "spans": [], "sample_ids": [], "types": [],
            "non_subspan": 0,
        })
        if r["span"] and r["span"] not in g["spans"]:
            g["spans"].append(r["span"])
        if r["sample_id"] not in g["sample_ids"]:
            g["sample_ids"].append(r["sample_id"])
        g["types"].append(r["type_raw"])
        if not r["is_subspan"]:
            g["non_subspan"] += 1

    kept, deleted = [], []
    for name, g in grouped.items():
        d = decisions.get(name, {})
        type_dist = dict(Counter(g["types"]))
        base = {
            "concept": g["concept"],
            "type": Counter(g["types"]).most_common(1)[0][0],
            "text_spans": g["spans"],
            "sample_ids": g["sample_ids"],
            "freq": len(g["sample_ids"]),
            "n_occurrences": len(g["types"]),
            "type_dist": type_dist,
            "pos_llm": d.get("pos_llm"),
            "pos_seg_flags": d.get("seg_flags", []),
            "flags": {
                "len_outlier": not (lo <= len(g["concept"]) <= hi),
                "non_subspan": g["non_subspan"] > 0,
                "pos_conflict": bool(d.get("conflict")),
            },
        }
        if d.get("decision") == "deleted_pos":
            base["decision"] = "deleted_pos"
            deleted.append(base)
        else:
            base["decision"] = "keep"
            kept.append(base)

    kept.sort(key=lambda x: (-x["freq"], x["concept"]))
    deleted.sort(key=lambda x: (-x["freq"], x["concept"]))
    freq_dist = Counter(">=2" if x["freq"] >= 2 else "=1" for x in kept)
    stats = {
        "n_unique_concepts": len(grouped),
        "n_kept": len(kept),
        "n_deleted_pos": len(deleted),
        "freq_ge2": freq_dist.get(">=2", 0),
        "freq_eq1": freq_dist.get("=1", 0),
        "n_len_outlier": sum(1 for x in kept if x["flags"]["len_outlier"]),
        "n_non_subspan": sum(1 for x in kept if x["flags"]["non_subspan"]),
    }
    return kept, deleted, stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    enforce_offline()  # 离线硬约束：先于一切 transformers/HF 导入生效

    cfg = load_cfg(args.config)
    par = resolve_params(args, cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # jieba 依赖预检（P2 通道二依赖；缺失则启动即报错，不跑到中段才崩）
    try:
        import jieba  # noqa: F401
    except ImportError:
        print("[错误] 当前环境缺少 jieba（P2 词性通道二依赖）。请先执行: pip install jieba")
        return 1

    # 1) 载入采样池
    with open(args.pool_path, encoding="utf-8") as f:
        pool = json.load(f)
    samples = pool["samples"]
    print(f"[B1] 采样池: {args.pool_path}（{len(samples)} 条）")
    print(f"[B1] LLM: {args.model_path} | 后端: {args.backend} | "
          f"配置: {args.config}")

    # 2) 缓存目录按 LLM 目录名隔离
    llm_name = Path(args.model_path).expanduser().resolve().name
    cache_dir = Path(args.cache_root) / llm_name / "concepts"
    p1_cache = JsonlCache(cache_dir / "p1_raw_cache.jsonl")
    p2_cache = JsonlCache(cache_dir / "p2_pos_cache.jsonl")

    t0 = time.time()
    llm = LLMBackend(args.model_path, args.backend, args.device, args.dtype,
                     seed=int(pool.get("metadata", {}).get("seed", 729)),
                     gpu_memory_utilization=args.gpu_memory_utilization,
                     max_model_len=args.max_model_len)

    # Step1 P1
    records, p1_stats = run_p1(samples, llm, p1_cache, par, args.limit)
    print(f"[B1][Step1] P1 完成: 输入 {p1_stats['n_input']} | JSON 成功 "
          f"{p1_stats['n_json_ok']} | 失败 {p1_stats['n_json_fail']} | "
          f"抽取记录 {p1_stats['n_records']} 条")

    # Step2 P2（对去重后的概念名）
    uniq_names = sorted({normalize_concept(r["concept_raw"]) for r in records
                         if normalize_concept(r["concept_raw"])})
    decisions, gate_stats = pos_gate(uniq_names, llm, p2_cache, par, out_dir)
    print(f"[B1][Step2] P2 完成: 唯一概念 {len(uniq_names)} | 删除 "
          f"{gate_stats.get('deleted_pos', 0)} | 词性冲突 "
          f"{gate_stats.get('conflict', 0)}")

    # Step3 归一合并
    kept, deleted, merge_stats = merge_records(records, decisions, par)
    with open(out_dir / "micro_concepts_raw.jsonl", "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "pos_deleted.jsonl", "w", encoding="utf-8") as f:
        for row in deleted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "stage": "B1",
        "elapsed_sec": round(time.time() - t0, 1),
        "model_path": args.model_path,
        "backend": args.backend,
        "p1": p1_stats,
        "p2_gate": gate_stats,
        "merge": merge_stats,
        "params": {k: v for k, v in par.items()},
    }
    with open(out_dir / "b1_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[B1][Step3] 合并完成: 唯一概念 {merge_stats['n_unique_concepts']} | "
          f"保留 {merge_stats['n_kept']}（freq>=2: {merge_stats['freq_ge2']}）| "
          f"词性删除 {merge_stats['n_deleted_pos']} | 长度越界 "
          f"{merge_stats['n_len_outlier']} | 非原文span "
          f"{merge_stats['n_non_subspan']}")
    print(f"[B1] 完成: {out_dir / 'micro_concepts_raw.jsonl'}（"
          f"{merge_stats['n_kept']} 行）| 用时 {stats['elapsed_sec']}s")

    p1_cache.close()
    p2_cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
