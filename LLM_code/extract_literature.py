# -*- coding: utf-8 -*-
"""
Universal Document Processor — Skeleton (Markdown-table; extended columns incl. Coulombic efficiency; NO Loading; Atomic Rows; 50-file batch subfolders)
- 始终为每个输入文件导出一个独立 CSV（即使没有解析出任何行，也导出“空表头 CSV”）
- SYSTEM_PROMPT 与 PROMPT_TEMPLATE 规则一致，并明确 charge/discharge 分行标注
- 移除未使用的 SINGLE_WINDOW_CHAR_LIMIT
- 合并大表 skeleton_parsed_all.csv 即使无数据也落一个空表头文件
- 新增：按输入顺序每 50 篇分一组，输出到 OUTPUT_DIR 下的子目录（如 1-50、51-100、…、最后一组如 101-135）
"""

import os, re, json, time, logging, sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import pandas as pd
from typing import Optional, List, Tuple, Dict

# ===================== 基本配置（路径/代理/模型） =====================
DIRECTORY_PATH = os.environ.get("LITERATURE_INPUT_DIR", "input_json")
OUTPUT_DIR     = os.environ.get("LITERATURE_OUTPUT_DIR", "extraction_result")
SOCKS_PROXY    = os.environ.get("LITERATURE_SOCKS_PROXY", "")   # 无代理时保持空字符串

OPENAI_MODEL   = 'gpt-4-1106-preview'
TEMPERATURE    = 0.0
MAX_TOKENS_OUT = 2048  # 可按需改到 3072

MAX_RETRIES    = 3
RETRY_SLEEP    = 8
MAX_WORKERS    = 8

# —— 切窗策略（固定长度滑窗 + 重叠）
FORCE_SINGLE_WINDOW      = False
WINDOW_CHARS             = 10_000
WINDOW_OVERLAP           = 600
TRUNCATE_WINDOW_CONTENT_TO: Optional[int] = 5000   # None 表示不截断（仅保存预览）
TRUNCATE_INPUT_CHARS: Optional[int] = None         # 送模前可截断；None 不截断

# —— 运行控制
RUN_MODEL = True

# —— 运行汇总索引：False=仅终端显示；True=同时保存 CSV
SAVE_RUN_INDEX_CSV = False

# —— 解析器导出控制
EXPORT_PER_FILE_CSV = True
EXPORT_EMPTY_PER_FILE_CSV = True   # 即使没有解析行，也为每个文件导出空 CSV
EXPORT_ALL_CSV      = True         # 合并大表：即使无数据也落空表头

# —— 行过滤（可选）
DROP_ROWS_MISSING_REQUIRED = False
REQUIRED_FIELDS = ["Positive", "Negative"]  # 如需更严，可改为 ["Positive","Negative","Solvents","Lithium salt"]

# ===================== 目标列定义（去掉 Loading） =====================
# 与模型输出严格一致：16 列（不包含 Loading）
EXPECTED_COLS = [
    "Positive", "Negative", "Rate", "Cycle", "Capacity", "Retention efficiency", "Coulombic efficiency",
    "Voltage window", "Temperature", "Solvents", "Lithium salt", "Additives",
    "Electrolyte conductivity", "Li-ion mobility", "SEI thickness", "CEI thickness"
]

# —— 表头同义词兜底（顺序越靠前越优先）——已移除 Loading 的同义词
HEADER_SYNONYMS: Dict[str, List[str]] = {
    "Positive":     ["Cathode", "Positive electrode", "Cathode material", "正极", "正极材料"],
    "Negative":     ["Anode", "Negative electrode", "Anode material", "负极", "负极材料"],
    "Rate":         ["C-rate", "Current rate", "Rate capability", "Charge rate", "Discharge rate", "Current density", "倍率", "电流密度"],
    "Cycle":        ["Cycles", "Cycle Count", "Cycle count", "Cycle number", "Cycle life", "Cycle retention", "循环数", "循环圈数"],
    "Capacity":     ["Specific capacity", "Discharge capacity", "容量", "比容量", "放电容量"],
    "Retention efficiency": ["Retention", "Capacity retention", "Retention (%)", "Capacity retention (%)", "RE", "保持率", "容量保持率"],
    "Coulombic efficiency": ["CE", "Coulombic eff.", "Coulombic efficiency (%)", "库伦效率", "CE%", "库伦效率(%)"],
    "Voltage window": ["Voltage", "Voltage range", "Cutoff voltage", "Upper/lower cutoff", "Potential window", "V range", "电压", "电压范围", "截止电压", "电压窗口"],
    "Temperature":   ["Temp", "Operating temperature", "Ambient temperature", "Test temperature", "Room temperature", "RT", "温度", "室温", "25 °C", "25C"],
    "Solvents":      ["Solvent", "溶剂", "溶剂体系", "溶剂组成"],
    "Lithium salt":  ["Lithium salts", "Li salt", "Salt", "锂盐", "锂盐种类", "盐浓度"],
    "Additives":     ["Additive", "添加剂", "功能添加剂", "成膜添加剂"],
    "Electrolyte conductivity": ["Ionic conductivity", "Conductivity", "电导率", "离子电导率", "σ", "mS cm^-1"],
    "Li-ion mobility": ["Li+ transference number", "Transference number", "Mobility", "迁移数", "锂离子迁移数", "t+ (Li)", "tLi+"],
    "SEI thickness": ["SEI layer thickness", "SEI thickness (nm)", "SEI 厚度", "SEI 层厚度"],
    "CEI thickness": ["CEI layer thickness", "CEI thickness (nm)", "CEI 厚度", "CEI 层厚度"],
}

# ===== 系统提示（与 EXPECTED_COLS 完全一致；新增“原子性/一行一条件/充放分行”硬规则）=====
SYSTEM_PROMPT = (
    "You are a meticulous extraction assistant. "
    "Output EXACTLY one of the following and nothing else:\n"
    "1) 'Nothing extracted.'\n"
    "2) 'Succeed in extracting information:' followed by ONE Markdown table with columns:\n"
    "| Positive | Negative | Rate | Cycle | Capacity | Retention efficiency | Coulombic efficiency | Voltage window | Temperature | Solvents | Lithium salt | Additives | Electrolyte conductivity | Li-ion mobility | SEI thickness | CEI thickness |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    "Rules:\n"
    "- The header names must match exactly the line above.\n"
    "- Use a standard Markdown divider line like the one above.\n"
    "- If a cell needs a literal vertical bar, escape it as \\| inside the cell.\n"
    "- Atomicity: One atomic fact per cell. Do NOT concatenate multiple values in any cell using ',', '/', ';', 'and', '与', '、', '以及', etc. "
    "Chemical mixture notation in composition cells (e.g., 'EC/EMC', 'FEC/AN') is allowed as a single atomic concept.\n"
    "- Row = one condition set: A row MUST correspond to a unique set of (Positive, Negative, Voltage window, Temperature, Rate). "
    "If multiple test conditions (e.g., several rates or temperatures) are reported, OUTPUT MULTIPLE ROWS—duplicate invariant cells across rows.\n"
    "- If both charge and discharge conditions exist, output separate rows and annotate Rate as 'X C (charge)' or 'Y C (discharge)'.\n"
    "- No explanations, no extra lines, no code fences."
)

# ===== 任务提示（与 SYSTEM_PROMPT 对齐）=====
PROMPT_TEMPLATE = r"""
You are an expert in lithium battery electrolyte research and are skilled at extracting information from texts.
Your task is to extract information about lithium battery tests from the given text. This is a text that may or may not contain the required information:
'''
[text content]
'''
Your mission is to extract the following information from the given text:

1) The positive and negative electrodes of the batteries;
2) The composition of the electrolyte used in the battery;
3) The charging and discharging rate of the battery;
4) The number of cycles and coulomb efficiency of the battery;
5) The capacity and the capacity retention (as a percentage if given);
6) The test voltage and test temperature of the batteries;
7) The electrical conductivity of the electrolyte and the lithium ion mobility;
8) The thickness of SEI and CEI

Follow these rules:
Please follow these steps to extract information from the given text:
1. Analyze the text, answer this question according to the content of the text: “Whether the text about the information mentioned above or not? Your answer should only be yes or no.”
   Keep in mind that do not output your answer, but execute Step 2.
2. If your answer in Step 1 is “Yes”, continue the process to step 3. If your answer in Step 1 is “no”, end the process and output nothing.
3. Identify the information mentioned previously. IF you find no relevant variables, output nothing.
4. The positive and negative electrodes of each row of batteries must be strictly matched with the electrolyte composition and test information.
5. Always escape vertical bars inside cell values as \| to keep Markdown valid.
6. Each row MUST represent exactly ONE set of test conditions (a unique combination of Rate, Temperature, and Voltage window). If the text reports multiple conditions, CREATE MULTIPLE ROWS instead of listing multiple values inside any cell.
7. One atomic fact per cell: do NOT concatenate multiple values using ',', '/', ';', 'and', '与', '、', '以及' etc. Chemical mixture notation in composition cells (e.g., EC/EMC, FEC/AN) is allowed as a single atomic concept.
8. If both charge and discharge rates exist, output separate rows and annotate Rate as 'X C (charge)' or 'Y C (discharge)' accordingly.
9. Your output MUST be exactly:
   Succeed in extracting information:
   [ONE Markdown table with exactly these columns in this order]
   and MUST include the standard Markdown divider line under the header.
10. No explanations, no extra lines, no code fences before or after the table.

The following is a random three-line sample table corresponding to different batteries (for reference):
'''
| Positive | Negative | Rate | Cycle | Capacity | Retention efficiency | Coulombic efficiency | Voltage window | Temperature | Solvents | Lithium salt | Additives | Electrolyte conductivity | Li-ion mobility | SEI thickness | CEI thickness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NMC811 | Graphite | 10 C (charge) | 1000 | N/A | 70% | 99.9% | 4.4 V | 25 °C | FEC/AN | 1.0 M LiPF6 | N/A | 12.0 mS cm^-1 | N/A | 3 nm | N/A |
| NMC811 | Graphite | 20 C (discharge) | 2500 | N/A | 80% | 99.9% | 4.4 V | 60 °C | FEC/AN | 1.0 M LiPF6 | N/A | 12.0 mS cm^-1 | N/A | 3 nm | N/A |
| Li | Graphite | 6 C | 300 | 300 mAh g^-1 | 80% | 99.9% | 4.3 V | N/A | FEC/AN | 1.0 M LiPF6 | N/A | 12.0 mS cm^-1 | N/A | 3 nm | N/A |
'''
'''
  
Remember: NO other sentences or extra lines besides the two allowed formats.
"""

# ============== 启动信息与日志 ==============
os.makedirs(OUTPUT_DIR, exist_ok=True)
start_time = datetime.now()
log_name   = os.path.join(OUTPUT_DIR, f"doc_framework_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print(f"== Universal Document Processor (Skeleton + Parser) started at {start_time} ==")
print(f"Log file: {log_name}")
print(f"Input dir: {DIRECTORY_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

# ===================== OpenAI 客户端（.env + SOCKS 代理） =====================
_CLIENT: Optional[OpenAI] = None  # 全局复用

def load_api_key() -> str:
    load_dotenv('./API_key.env')
    k = os.getenv('OPENAI_API_KEY')
    if not k and RUN_MODEL:
        raise RuntimeError('OPENAI_API_KEY 未找到（请在 ./API_key.env 中配置）')
    return k or ''

def build_client() -> Optional[OpenAI]:
    if not RUN_MODEL:
        return None
    if SOCKS_PROXY:
        # 仅在启用 SOCKS 代理时加载可选依赖，不影响直连模式。
        try:
            from httpx_socks import SyncProxyTransport
        except ModuleNotFoundError as exc:
            if exc.name != 'httpx_socks':
                raise
            raise RuntimeError(
                '已启用 SOCKS 代理，但未安装 httpx-socks。'
                '请执行 python -m pip install -r requirements-proxy.txt，'
                '或清空 LITERATURE_SOCKS_PROXY / SOCKS_PROXY 后使用直连模式。'
            ) from exc
        transport = SyncProxyTransport.from_url(SOCKS_PROXY)
        http_client = httpx.Client(transport=transport, timeout=httpx.Timeout(60.0, connect=20.0))
        return OpenAI(api_key=load_api_key(), http_client=http_client)
    else:
        return OpenAI(api_key=load_api_key())

# ===================== JSON 文本抽取与清理（通用版） =====================
_REF_HEADERS   = re.compile(r'^\s*(references|参考文献|bibliography)\s*$', re.I)
_DROP_HEADERS  = re.compile(r'^\s*(acknowl(e)?dg(e)?ments?|author information|作者信息|affiliations?|funding|conflict of interest|数据可用性|data availability|supplement(ary)?( information)?|supporting information|graphical abstract|toc|notes?)\s*$', re.I)

def _collect_text_nodes(obj, acc: List[str]):
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            acc.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_text_nodes(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_text_nodes(v, acc)
    else:
        try:
            s = str(obj).strip()
            if s:
                acc.append(s)
        except Exception:
            pass

def _clean_front_matter(text: str) -> str:
    lines = text.splitlines()
    kept = []
    drop_refs = False
    for ln in lines:
        s = ln.strip()
        if _REF_HEADERS.match(s):
            drop_refs = True
            continue
        if drop_refs:
            continue
        if _DROP_HEADERS.match(s):
            continue

        # 去掉图表标题 “Figure/Table X:” 前缀，仅保留描述
        m_cap = re.match(r'^\s*((Figure|Fig\.|Scheme|Table|图|表)\s*\d+[:.\-]?)\s*(.*)$', s, re.I)
        if m_cap:
            rest = m_cap.group(3).strip()
            if rest:

                kept.append(rest)
            continue

        # 常见水印/邮箱/DOI/版权/页脚
        if re.search(r'\bDOI[:\s]', s, re.I): continue
        if re.search(r'@[\w\.-]+\.[\w]+', s): continue
        if ('ORCID' in s) or ('orcid.org' in s): continue
        if re.match(r'^\s*(Received|Accepted|Published online|Manuscript)\b', s, re.I): continue
        if re.match(r'^\s*Page \d+ of \d+\s*$', s, re.I): continue
        if re.match(r'^\s*(This is an open access article|©|\(c\))', s, re.I): continue

        kept.append(ln)

    clean = '\n'.join(kept)
    clean = re.sub(r'[ \t]+', ' ', clean)
    clean = re.sub(r'\n{2,}', '\n\n', clean)
    return clean.strip()

def extract_text_from_json(file_path: str) -> str:
    """递归抽取 JSON 全部文本，并做前置清理。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        acc: List[str] = []
        _collect_text_nodes(data, acc)
        raw = '\n'.join(acc)
        return _clean_front_matter(raw)
    except Exception as e:
        logging.error(f'读取/提纯失败：{file_path} -> {e}')
        return ""

# ===================== 固定长度滑窗（不依赖断句） =====================

def simplified_segmentation(text: str) -> List[Tuple[str, str]]:
    """
    固定长度滑窗 + 重叠；不依赖断句。
    始终按 WINDOW_CHARS 切窗（除非 FORCE_SINGLE_WINDOW=True 或文本极短）。
    """
    if not text:
        return []

    if FORCE_SINGLE_WINDOW or len(text) <= max(WINDOW_CHARS, 1000):
        return [("FullText", text)]

    windows: List[Tuple[str, str]] = []
    step = max(WINDOW_CHARS - WINDOW_OVERLAP, 1)
    n = len(text)
    idx = 0
    widx = 1
    while idx < n:
        end = min(idx + WINDOW_CHARS, n)
        chunk = text[idx:end]
        windows.append((f"Window_{widx}", chunk))
        if end >= n:
            break
        idx += step
        widx += 1
    return windows

# ===================== Markdown 表解析器（加分隔线兜底 + 列数自愈） =====================

def _is_md_divider(line: str) -> bool:
    """严格判定 Markdown 表头分隔线，如: | --- | :---: | ---: |"""
    s = line.strip()
    return bool(re.fullmatch(r'\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?', s))

def _split_md_row(line: str) -> List[str]:
    """
    使用“未转义竖线”分列；把 \| 还原为 |
    这样 'Li\\|\\|graphite' 不会被拆坏。
    """
    s = line.strip().strip('|')
    parts = re.split(r'(?<!\\)\|', s)  # 仅在未转义的 | 处分割
    return [c.replace(r'\|', '|').strip() for c in parts]

def _normalize_header(header: List[str]) -> Dict[str, int]:
    """将解析到的列名映射到 EXPECTED_COLS 的索引。大小写/空格宽松匹配 + 同义词兜底。"""
    idx_map: Dict[str, int] = {}
    norm = [re.sub(r"\s+", " ", h.strip()).lower() for h in header]
    norm_nospace = [x.replace(' ', '') for x in norm]

    for need in EXPECTED_COLS:
        key = need.lower()
        # 1) 完全匹配
        j = norm.index(key) if key in norm else -1
        # 2) 去空格匹配
        if j < 0:
            key2 = key.replace(' ', '')
            j = norm_nospace.index(key2) if key2 in norm_nospace else -1
        # 3) 同义词兜底
        if j < 0:
            syns = HEADER_SYNONYMS.get(need, [])
            for s in syns:
                s_norm = s.lower()
                s2 = s_norm.replace(' ', '')
                if s_norm in norm:
                    j = norm.index(s_norm); break
                if s2 in norm_nospace:
                    j = norm_nospace.index(s2); break
        idx_map[need] = j
    return idx_map


# Markdown 表格解析器（已移除未使用的旧定义）
def parse_markdown_table(md_text: str) -> List[Dict[str, str]]:
    if not isinstance(md_text, str) or not md_text.strip():
        return []
    lines = [ln.rstrip() for ln in md_text.splitlines()]

    start = None
    has_divider = False
    for i in range(len(lines) - 1):
        if lines[i].strip().startswith('|') and _is_md_divider(lines[i + 1]):
            start = i
            has_divider = True
            break
    if start is None:
        for i in range(len(lines)):
            if lines[i].strip().startswith('|'):
                start = i
                break
        if start is None:
            return []

    header = _split_md_row(lines[start])
    idx_map = _normalize_header(header)
    if all(idx_map[c] == -1 for c in EXPECTED_COLS):
        return []

    i = start + (2 if has_divider else 1)
    rows: List[Dict[str, str]] = []
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln:
            i += 1
            continue
        if not ln.strip().startswith('|'):
            break
        cells = _split_md_row(ln)

        if len(cells) > len(header):
            merged = []

            skip = False
            for k, c in enumerate(cells):
                if skip:
                    skip = False
                    continue
                if c == '' and k + 1 < len(cells) and cells[k + 1] == '':
                    merged.append('')
                    skip = True
                else:
                    merged.append(c)
            cells = merged

        row: Dict[str, str] = {}
        for col in EXPECTED_COLS:
            j = idx_map.get(col, -1)
            row[col] = cells[j] if (j >= 0 and j < len(cells)) else 'N/A'
        rows.append(row)
        i += 1
    return rows

def extract_rows_from_model_response(model_resp: Optional[str], meta: Dict[str, str]) -> List[Dict[str, str]]:
    """将单个窗口的 model_response 解析为结构化行，并附加元数据（文件名/窗口）。"""
    if not isinstance(model_resp, str) or not model_resp.strip():
        return []
    text = model_resp.strip()
    if text == 'Nothing extracted.' or text.startswith('Error:'):
        return []
    rows = parse_markdown_table(text)
    for r in rows:
        r.update(meta)  # 附加 file_name/section/window_index
    return rows

# ===================== 模型交互 =====================

def _call_model(client: Optional[OpenAI], prompt_template: str, text_block: str):
    if (not RUN_MODEL) or (not prompt_template.strip()):
        return {"ok": True, "content": None, "tokens": 0, "skipped": True}

    # 输入截断（可选）
    text_for_model = (
        (text_block[:TRUNCATE_INPUT_CHARS] + ' …[truncated]')
        if isinstance(TRUNCATE_INPUT_CHARS, int) and TRUNCATE_INPUT_CHARS > 0 and len(text_block) > TRUNCATE_INPUT_CHARS
        else text_block
    )

    # 用窗口文本替换占位符 [text content]
    user_payload = (
        prompt_template.replace("[text content]", text_for_model)
        if "[text content]" in prompt_template
        else (prompt_template + "\n\n" + text_for_model)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_payload}
    ]
    logging.info(f"MODEL={OPENAI_MODEL}, temperature={TEMPERATURE}, max_tokens={MAX_TOKENS_OUT}")
    logging.info("PROMPT_SENT_TO_GPT_BEGIN\n" + user_payload + "\nPROMPT_SENT_TO_GPT_END")

    last_err = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_OUT,
            )
            content = resp.choices[0].message.content if (resp.choices and resp.choices[0].message) else None
            usage   = getattr(resp, 'usage', None)
            tokens  = getattr(usage, 'total_tokens', 0) if usage else 0
            logging.info('RAW_RESPONSE_BEGIN\n' + (content or '') + '\nRAW_RESPONSE_END')
            return {"ok": True, "content": content, "tokens": tokens, "skipped": False}
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
            else:
                logging.error(f"模型请求失败：{e}")
                return {"ok": False, "content": None, "tokens": 0, "skipped": False, "error": str(last_err)}

def interact_with_gpt4(content: str, prompt_template: str, token_count: int, request_count: int):
    """返回 (response_text_or_None, token_count, request_count)。"""
    global _CLIENT
    try:
        if RUN_MODEL and prompt_template.strip():
            if _CLIENT is None:
                _CLIENT = build_client()
            client = _CLIENT
        else:
            client = None
        r = _call_model(client, prompt_template, content)
        token_count += r.get("tokens", 0)
        request_count += (0 if r.get("skipped") else 1)
        if not r.get("ok"):
            return f"Error: {r.get('error')}", token_count, request_count
        return (r.get("content") if not r.get("skipped") else None), token_count, request_count
    except Exception as e:
        logging.error(f"Error in model interaction: {str(e)}")
        return f"Error: {str(e)}", token_count, request_count

# ===================== 主处理：骨架输出 + 解析CSV =====================

def _maybe_truncate(s: str, limit: Optional[int]) -> str:
    if limit is None or not isinstance(s, str): return s
    return s if len(s) <= limit else (s[:limit] + " …[truncated]")

def _filter_rows_by_required_fields(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not DROP_ROWS_MISSING_REQUIRED or not REQUIRED_FIELDS:
        return rows
    kept = []
    for r in rows:
        ok = True
        for fld in REQUIRED_FIELDS:
            val = (r.get(fld, "") or "").strip()
            if val == "" or val.upper() == "N/A":
                ok = False
                break
        if ok:
            kept.append(r)
    return kept

def process_and_save_final_result(file_path: str, prompt_template: str, output_dir: str):
    """读取→清理→切窗→（可选）调用模型→保存骨架 JSON；解析 Markdown 表并导出 per-file CSV。
    返回 (token_count, n_windows, parsed_rows)。"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    full_text = extract_text_from_json(file_path)
    if not isinstance(full_text, str) or not full_text.strip():
        logging.error(f"Invalid/empty content in file: {base_name}")
        return 0, 0, []

    segments = simplified_segmentation(full_text)
    token_count = 0
    request_count = 0
    skeleton_rows = []
    parsed_rows_all: List[Dict[str, str]] = []

    # 并行跑各窗口
    def run_one(idx_section):
        idx, (section, text) = idx_section
        resp, tk, rq = interact_with_gpt4(text, prompt_template, 0, 0)
        meta = {"file_name": base_name, "section": section, "window_index": idx}
        parsed = extract_rows_from_model_response(resp, meta)
        return {
            "idx": idx,
            "file_name": base_name,
            "section": section,
            "window_index": idx,
            "content": _maybe_truncate(text, TRUNCATE_WINDOW_CONTENT_TO),
            "model_response": resp,  # Markdown 表 / 或 Nothing extracted.
            "tokens": tk,
            "requests": rq,
            "parsed_rows": parsed,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_one, (i, seg)) for i, seg in enumerate(segments, start=1)]
        for fut in as_completed(futures):
            row = fut.result()
            token_count += row["tokens"]
            request_count += row["requests"]
            skeleton_rows.append(row)
            parsed_rows_all.extend(row.get("parsed_rows", []))

    # 可选：过滤必填字段
    parsed_rows_all = _filter_rows_by_required_fields(parsed_rows_all)

    # 排序后写骨架文件（不含 parsed_rows 列）
    skeleton_rows.sort(key=lambda x: x["idx"])
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'parsed_rows'} for r in skeleton_rows])
    out_path = os.path.join(output_dir, f"{base_name}_skeleton.json")
    df.to_json(out_path, orient='records', force_ascii=False)
    logging.info(f"Saved skeleton: {out_path}")

    # === 每个文件都导出它的 CSV（即使没有解析行） ===
    if EXPORT_PER_FILE_CSV:
        dfp_cols = EXPECTED_COLS + ["file_name", "section", "window_index"]
        if parsed_rows_all:
            dfp = pd.DataFrame(parsed_rows_all)
        else:
            # 没有任何解析行：若允许导出空表，则创建空 DataFrame（只有表头）
            if EXPORT_EMPTY_PER_FILE_CSV:
                dfp = pd.DataFrame(columns=dfp_cols)  # 仅表头
            else:
                dfp = None

        if dfp is not None:
            # 补全缺失列并按列顺序输出
            for c in dfp_cols:
                if c not in dfp.columns: dfp[c] = 'N/A'
            dfp = dfp[dfp_cols]
            out_csv = os.path.join(output_dir, f"{base_name}_parsed.csv")
            dfp.to_csv(out_csv, index=False)
            logging.info(f"Saved per-file parsed CSV: {out_csv} (rows={len(dfp)})")

    return token_count, len(skeleton_rows), parsed_rows_all

def end_part_of_script(start_time: datetime, latest_token_count: Optional[int]):
    end_time = datetime.now()
    logging.info(f"Script ended at {end_time}")
    logging.info(f"Total runtime: {end_time - start_time}")
    if latest_token_count is not None:
        logging.info(f"Tokens used in this session: {latest_token_count}")

# ===================== 终端进度显示工具 =====================

def _fmt_td(td: timedelta) -> str:
    s = int(td.total_seconds()); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _render_bar(pct: float, width: int = 40) -> str:
    pct = max(0.0, min(1.0, pct)); fill = int(width * pct)
    return "[" + "#" * fill + "-" * (width - fill) + "]"

def _print_progress(processed: int, total: int, start_dt: datetime, grand_tokens: int, cur_file: str = "", cur_time: float = None):
    elapsed = datetime.now() - start_dt
    pct = (processed / total) if total else 1.0
    bar = _render_bar(pct, 40)
    avg_sec = elapsed.total_seconds() / processed if processed else 0.0
    remain = total - processed
    eta = timedelta(seconds=int(avg_sec * remain)) if processed else timedelta(seconds=0)
    cur_part = f" | last_file='{cur_file}'" if cur_file else ""
    if cur_time is not None: cur_part += f" took {cur_time:.2f}s"
    line = (f"\r{bar} {processed}/{total} ({pct*100:5.1f}%) | elapsed { _fmt_td(elapsed) } "
            f"| ETA { _fmt_td(eta) } | tokens {grand_tokens}{cur_part}   ")
    sys.stdout.write(line); sys.stdout.flush()

def _newline_progress():
    sys.stdout.write("\n"); sys.stdout.flush()

# ===================== 50 篇分组子目录工具 =====================

def _batch_subdir_for_index(i_1based: int, total: int, base_outdir: str) -> str:
    """
    输入的 i_1based 是本次运行中文件的序号（从 1 开始）。
    每 50 篇划一组：1-50, 51-100, ...，最后一组根据 total 动态截尾，例如 101-135。
    """
    batch = (i_1based - 1) // 50
    start = batch * 50 + 1
    end = min((batch + 1) * 50, total)
    name = f"{start}-{end}"
    path = os.path.join(base_outdir, name)
    os.makedirs(path, exist_ok=True)
    return path

# ===================== 入口：批处理 + 汇总 CSV（骨架 & 解析） =====================

def _list_json_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        print('目录不存在：', directory)
        return []
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.json')]
    files.sort()
    return files

def main():
    files = _list_json_files(DIRECTORY_PATH)
    total = len(files)
    if not files:
        print('目录为空：', DIRECTORY_PATH); return

    # —— 预初始化 OpenAI 客户端，避免并发首次调用的重复构建
    global _CLIENT
    if RUN_MODEL and _CLIENT is None:
        _CLIENT = build_client()

    index_rows = []
    grand_tokens = 0
    processed = 0
    all_parsed_rows: List[Dict[str, str]] = []

    print("="*90); print(f"Total files: {total}"); print("="*90)
    _print_progress(0, total, start_time, grand_tokens)

    for i, file_path in enumerate(files, start=1):  # i 从 1 开始
        f = os.path.basename(file_path)
        # 计算当前文件应落到的 50 篇分组子目录
        batch_out_dir = _batch_subdir_for_index(i, total, OUTPUT_DIR)

        t0 = time.time()
        try:
            tokens_used, n_windows, parsed_rows = process_and_save_final_result(file_path, PROMPT_TEMPLATE, batch_out_dir)
            dt = time.time() - t0
            grand_tokens += tokens_used
            processed += 1
            parsed_n = len(parsed_rows)
            index_rows.append({
                "file_name": f,
                "status": "OK",
                "windows": n_windows,
                "tokens": tokens_used,
                "parsed_rows": parsed_n,
                "time_s": round(dt, 2),
                "batch_dir": os.path.basename(batch_out_dir),
            })
            all_parsed_rows.extend(parsed_rows)
            _newline_progress()
            print(f"[OK] {f} -> {os.path.basename(batch_out_dir)} | windows={n_windows} | parsed_rows={parsed_n} | time={dt:.2f}s | tokens+={tokens_used} | total_tokens={grand_tokens}")
            _print_progress(processed, total, start_time, grand_tokens, cur_file=f, cur_time=dt)
        except Exception as e:
            dt = time.time() - t0
            processed += 1
            index_rows.append({
                "file_name": f,
                "status": f"FAIL: {e}",
                "windows": 0, "tokens": 0, "parsed_rows": 0,
                "time_s": round(dt, 2),
                "batch_dir": os.path.basename(batch_out_dir),
            })
            _newline_progress()
            print(f"[FAIL] {f}: {e} | time={dt:.2f}s -> intended batch {os.path.basename(batch_out_dir)}")
            _print_progress(processed, total, start_time, grand_tokens, cur_file=f, cur_time=dt)

    _newline_progress()
    print("\nSummary:")
    if index_rows:
        df = pd.DataFrame(index_rows)
        head_n = min(20, len(df))
        print(df.head(head_n).to_string(index=False))
        if len(df) > head_n:
            print(f"... ({len(df)-head_n} more rows)")
        if SAVE_RUN_INDEX_CSV:
            idx_csv = os.path.join(OUTPUT_DIR, "skeleton_index.csv")
            df.to_csv(idx_csv, index=False)
            print(f"(Saved full index to {idx_csv})")
        else:
            print("(Run index not saved to disk; showing summary only)")

    # —— 合并大表：即使没有任何行也写出空表头
    if EXPORT_ALL_CSV:
        cols = EXPECTED_COLS + ["file_name", "section", "window_index"]
        if all_parsed_rows:
            df_all = pd.DataFrame(all_parsed_rows)
            for c in cols:
                if c not in df_all.columns: df_all[c] = 'N/A'
            df_all = df_all[cols]
        else:
            df_all = pd.DataFrame(columns=cols)  # 空表头
        out_all = os.path.join(OUTPUT_DIR, "skeleton_parsed_all.csv")
        df_all.to_csv(out_all, index=False)
        print(f"Merged parsed CSV -> {out_all} (rows={len(df_all)})")

    print("="*90)
    total_elapsed = datetime.now() - start_time
    print(f"Completed at {datetime.now()} | Runtime: {_fmt_td(total_elapsed)} | Total tokens: {grand_tokens}")
    print("="*90)
    end_part_of_script(start_time, grand_tokens)

if __name__ == "__main__":
    main()

