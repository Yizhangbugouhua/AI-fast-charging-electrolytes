# LLM-assisted literature extraction for battery electrolytes

Prompt templates and Python code for extracting electrolyte formulations and battery-test information from literature text parsed with MinerU. This repository covers the initial structured-extraction step described in the Supplementary Information of the manuscript.

## 仓库用途

处理流程：MinerU解析PDF并生成JSON → 整理文本并按窗口分段 → 调用GPT提取信息 → 将Markdown表格解析为CSV。

完整提示词已包含在`extract_literature.py`的`SYSTEM_PROMPT`和`PROMPT_TEMPLATE`中。运行时，`[text content]`会被实际文本替换。提示词中的示例行用于说明输出格式，不是实际实验结果。

本仓库公开文献信息提取方法。后续人工核验、数据筛选、去重、1C容量配对及 R_x = Q_x / Q_1C 标签计算见SI方法。图片不作为多模态输入提交给GPT，倍率性能图由人工读取与核验。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `extract_literature.py` | 完整提取脚本，含两段原始提示词 |
| `README.md` | 环境、参数、输入输出和操作说明 |
| `requirements.txt` | 基础运行依赖 |
| `requirements-proxy.txt` | 基础依赖及可选SOCKS代理依赖 |
| `API_key.env.example` | API密钥配置示例，不包含真实密钥 |
| `.gitignore` | Git操作时忽略本地密钥、输入文献、输出及缓存 |

## 软件信息

| 软件 | 已核实信息 |
| --- | --- |
| Python | 3.10.18 |
| MinerU | 2.1.10，用于上游PDF解析 |
| OpenAI Python SDK | 1.99.9 |
| pandas | 2.3.3 |
| python-dotenv | 1.1.1 |
| httpx | 0.28.1 |
| httpx-socks | 可选；仅启用SOCKS代理时需要，未锁定历史版本 |

Python、MinerU和OpenAI版本由作者在所用环境中核实；pandas、python-dotenv和httpx版本来自作者2026-11-06提供的查询结果，其中pandas来自用户级安装目录。上述记录用于说明可核实的软件信息，不等同于完整历史环境快照。MinerU不由此脚本调用，因此不列入基础脚本依赖。

`httpx-socks`未在当前查询的环境中检出，其历史版本未确认。公开整理版将它改为按需加载：未启用代理时无需安装；启用代理而缺包时，脚本会给出安装说明，不会悄悄改为直连。此调整保留原始提示词和全部提取规则。

## 提取设置

| 参数 | 设置 |
| --- | --- |
| 模型标识 | `gpt-4-1106-preview` |
| Temperature | `0.0` |
| 最大输出长度 | `2048` tokens |
| 文本窗口长度 | `10000`字符 |
| 相邻窗口重叠长度 | `600`字符 |
| 输入额外截断 | 无 |
| 保存的文本预览长度 | 最多`5000`字符，仅影响预览 |
| 并发窗口任务数 | `8` |
| 请求尝试次数 | 最多`3`次 |
| 重试等待 | `8`秒 |
| 输出格式 | `Nothing extracted.`，或固定16列Markdown表格 |

模型标识用于记录原提取设置。重新运行须具备相应模型的API访问条件；如果修改模型，应另行记录新设置。

## 输入文件

在仓库目录下创建`input_json`文件夹，放入MinerU生成的`.json`文件。脚本只扫描该目录第一层，不递归扫描子目录。

建议每篇文献使用可追溯的文件名，并由作者内部保存文件名与文献题名或DOI的对应关系。脚本保留文件名及窗口索引，不自动生成DOI。

## 安装与运行

以下命令在Linux终端的本仓库目录中执行。运行提取会向API发送实际请求。

1. 安装基础依赖：

```bash
python -m pip install -r requirements.txt
```

2. 复制密钥模板，并在本地编辑生成的`API_key.env`，填入自己的密钥：

```bash
cp API_key.env.example API_key.env
```

3. 创建输入文件夹并放入JSON：

```bash
mkdir -p input_json
```

4. 默认输入目录为`input_json`，输出目录为`extraction_result`。需要更改时可编辑脚本顶部配置，或设置：

```bash
export LITERATURE_INPUT_DIR=/absolute/path/to/jsons
export LITERATURE_OUTPUT_DIR=/absolute/path/to/results
```

5. 运行脚本：

```bash
python extract_literature.py
```

### 可选：SOCKS代理

默认`SOCKS_PROXY`为空，使用直连模式，不导入`httpx_socks`。如果网络环境需要SOCKS代理，再执行：

```bash
python -m pip install -r requirements-proxy.txt
export LITERATURE_SOCKS_PROXY=socks5://127.0.0.1:1080
python extract_literature.py
```

`1080`只是示例端口，须替换为自己的实际代理地址与端口。恢复默认直连模式可执行`unset LITERATURE_SOCKS_PROXY`；若直接修改过脚本的`SOCKS_PROXY`常量，也需将其清空。

如只检查本地处理流程，可在配置中将`RUN_MODEL`改为`False`。该模式跳过API调用，不产生真实模型提取记录。

## 输出文件

| 输出 | 内容 |
| --- | --- |
| `extraction_result/1-50/<文件名>_parsed.csv`等分组目录 | 每篇文献的结构化提取结果，每50篇输入文献分为一组 |
| 同目录的`<文件名>_skeleton.json` | 文本窗口、预览、原始响应及运行元数据 |
| `extraction_result/skeleton_parsed_all.csv` | 本次运行的合并提取结果 |
| `extraction_result/doc_framework_*.log` | 请求提示、原始响应及运行日志 |

输出字段由`EXPECTED_COLS`定义，另附`file_name`、`section`和`window_index`。`Retention efficiency`是原文信息提取字段，不会自动转换为论文定义的 R_x。

窗口重叠和原文重复叙述可能产生重复记录，后续按SI方法清洗与人工核验。

## 公开整理版的改动与验证

- 完整保留两段提示词、模型设置、分段规则、提取字段和响应解析逻辑。
- 删除了被后续同名函数覆盖、未生效的一段旧表格解析函数。
- 将个人服务器路径改为通用路径，支持环境变量覆盖；代理默认关闭。
- 仅在启用SOCKS代理时加载`httpx_socks`，将代理依赖单独列入可选依赖文件。
- 补充环境、运行方法及配置示例。

本次整理通过Python语法检查、原始提示词及提取规则一致性检查，并使用离线替身验证了直连、跳过模型、缺少代理依赖和启用代理四条路径。没有发出真实API请求，也未重新提取文献或重建原数据库。

原上传脚本的SHA-256：

`b6f1ba2dbc14ffe7fb38171651d55505d8b37d9edbf57f559d70ce2c7466dae8`

## 本模块在整篇论文仓库中的位置

本文件夹位于仓库的`LLM_code/`。新建仓库及上传方法见根目录的`README.md`。运行本模块前，先进入本文件夹：

```bash
cd LLM_code
```

然后按上面的安装、密钥配置和运行步骤操作。暂时保留模块自身的工作目录，可以使用现有代码，不需要为仓库重组改写提取流程。

## 论文引用本仓库的表述

上传完成并填入真实网址后，SI方法可写：

“文献信息提取所用的完整提示词、处理脚本及运行说明见代码仓库：【填写实际仓库网址】。”

正文Code availability可写：

“本研究用于文献信息提取的提示词及相关处理脚本可通过以下代码仓库获取：【填写实际仓库网址】。软件依赖、提取参数及运行方法见仓库中的README文件。”

上述表述对应本仓库实际提供的文献提取代码；SMI-TED、MLP及候选空间代码可在后续整理时另行增加并同步更新声明。
