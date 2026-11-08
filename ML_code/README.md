# ML_code — 配方处理、模型训练与候选筛选

当前已加入九份作者提供的工作流代码，另在 external 中保存提供的 SMI-TED loader。整理仅调整文件名、路径配置和输出目录创建；候选生成器中 `x6=0.01` 的注释改为“质量分数，即 1 wt%”，数值不变。编码脚本的标签量纲注释及数组检查脚本的配对/标准化说明已纠正，计算不变。学习算法、网络结构、训练参数、序列构建及采样规则均保留。原文件与仓库文件的 SHA256 和对应关系见 `source_map.json`。

## 1. 文件功能与运行边界

| 仓库文件 | 原文件 | 实际作用 |
| --- | --- | --- |
| `preprocessing/Unify.py` | `Unify(1).py` | 将原表的组成换算为摩尔份额，增加字符串列 `Unified_mole_ratio` |
| `preprocessing/replace.py` | `replace(1).py` | 对已整理的六槽位表，将组分缩写替换为 SMILES |
| `preprocessing/smiles_integrate.py` | `smiles_integrate(1).py` | 六槽位序列化，导出 `seqA,T_C,Rate_C,y_retention` |
| `encode_fastcharge_embeddings.py` | `encode_fastcharge_embeddings(2).py` | 调用实际 SMI-TED encode_raw，拼接温度与倍率，保存训练 X/y |
| `check_npy.py` | `check_npy(1).py` | 辅助检查数组形状、缺失值、目标范围和分布，不生成编码 |
| `train_mlp.py` | `data_fastcharge_mlp(1).py` | 读取现成 X/y 数组，训练 MLP、评估并保存模型三件套 |
| `generate_candidates.py` | `generate_place(2).py` | 在质量分数约束下随机采样，换算 mol%，生成候选序列 |
| `predict_default.py` | `prediction.py` | 读取行内工况；缺失时默认 25 °C、4C，编码后预测并导出 Top20 |
| `predict_5c.py` | `prediction_0530(1).py` | 忽略行内工况，固定 25 °C、5C，编码后预测并导出 Top20 |

## 2. 流程衔接及待补项

1. `Unify.py` 只新增 `Unified_mole_ratio`，不会创建后两份脚本要求的 `solvent1…solvent4,salt1,salt2` 及对应 `*_mol_pct`。需要实际使用的中间处理脚本，或说明这一阶段是如何人工完成的。为避免误接，`replace.py` 默认读取独立的 `formulations_six_slots.xlsx`。
2. 已取得真实 `encode_fastcharge_embeddings.py`：读取 `seqA,T_C,Rate_C,y_retention`，通过 `encode_raw` 生成向量，按 `[embedding,T_C,Rate_C]` 拼接，保存与训练脚本同名的两个 NPY。它与预测文件使用同一函数名、权重名和词表名。实际 `load.py` 已补入 `external/smi_ted_light/`，确认有对应接口；外部依赖和匹配模型文件仍需准备，同名不等于文件内容相同。

原上传编码脚本指向 `data_experiment_smited_fastcharge.csv`；文件名不能证明该次输入包含哪些文献/反馈记录。仓库默认接到上游序列 CSV，可用 `FORMULATION_SEQUENCE_CSV` 指定真实阶段文件。不要据此补造各轮样本数或历史结果。

## 3. 依赖与配置

`requirements.txt` 是根据当前脚本导入整理的**未锁版本依赖清单**，不代表已经恢复作者历史 ML 环境。已从实际 loader 确认 transformers、RDKit、regex、tqdm 和 fast_transformers 等导入；各实际版本与 fast_transformers 对应安装来源仍待核实。确认环境后再建立锁定版本的环境文件。

路径均以仓库根目录为基准，可通过以下环境变量覆盖；无需修改作者服务器路径：

| 环境变量 | 默认相对路径/用途 |
| --- | --- |
| `ELECTROLYTE_PROPERTIES_XLSX` | `data/private/electrolyte_properties.xlsx`，缩写/SMILES/物性表 |
| `FORMULATION_RAW_XLSX` | `data/private/formulations_raw.xlsx`，原配方表 |
| `FORMULATION_MOLES_XLSX` | `outputs/preprocessing/data_mol_ration.xlsx`，统一摩尔比结果 |
| `FORMULATION_SLOTS_XLSX` | `data/private/formulations_six_slots.xlsx`，中间六槽位表 |
| `FORMULATION_SMILES_XLSX` | `outputs/preprocessing/data_replaced.xlsx` |
| `FORMULATION_INTEGRATED_XLSX` | `outputs/preprocessing/data_smiles_integrate.xlsx` |
| `FORMULATION_SEQUENCE_CSV` | `outputs/preprocessing/data_smited_fastcharge.csv` |
| `FASTCHARGE_X_NPY` / `FASTCHARGE_Y_NPY` | `data/private/X_fastcharge_data.npy` / `y_fastcharge_data.npy` |
| `MLP_OUTPUT_DIR` | 默认 `outputs/training/FixedBest_FinalRelease_<时间戳>` |
| `CANDIDATE_MATERIALS_XLSX` | `data/private/candidate_materials.xlsx` |
| `CANDIDATE_CSV` | `outputs/candidates/place.csv` |
| `SMITED_DIR` | `external/smi_ted_light`，作者实际 loader、词表及权重目录 |
| `MLP_MODEL_DIR` | `ML_code/models/final`，同一次训练的三件套 |
| `PREDICTION_OUTPUT_DIR` | 预测默认分别写入 `outputs/predictions/default` 或 `outputs/predictions/5c` |

代码目录可放在任何位置。环境变量应使用对应文件/目录的实际路径。

## 4. 分阶段运行

以下命令在仓库根目录执行，**每段需先准备对应输入**，并非当前就能依次跑通的全流程。

```bash
# 组成换算；随后补齐实际六槽位转换，不能直接跳到下一步。
python ML_code/preprocessing/Unify.py

# 已有完整六槽位表时：
python ML_code/preprocessing/replace.py
python ML_code/preprocessing/smiles_integrate.py

# 已有配方序列 CSV 及实际 SMI-TED loader/权重/词表时：
python ML_code/encode_fastcharge_embeddings.py
python ML_code/check_npy.py

# 已有真实训练编码数组时：
python ML_code/train_mlp.py

# 已有组分参数表时；原设置目标 2,000,000 行，运行前先核对采样规则：
python ML_code/generate_candidates.py

# 已有配套三件套、实际 SMI-TED loader/权重/词表和候选 CSV 时：
python ML_code/predict_5c.py
```

`predict_default.py` 是独立的另一预测入口，不需要先执行它再执行 `predict_5c.py`。它会覆盖自己原有的两个 CSV 输出。5C 入口使用时间戳区分输出，`USE_ROW_T_RATE` 在该版本中只是记录字段；实际始终使用固定工况。

## 5. 已核实的训练设置

- 输入：现成二维特征数组 X、一维目标 y；训练脚本自身不检查是否为 770 维。
- 目标：脚本直接使用 y；序列导出脚本也直接使用 `Retention efficiency`，不计算 `Q_x/Q_1C`，不自动除以 100。需在上游确保是作者定义的比例标签。
- 划分：对 y 做 5 个等宽区间分箱后分层随机划分，目标比例训练/验证/测试为 70/10/20，实际样本数受整数舍入影响；seed=2026。
- StandardScaler 仅在训练集拟合，再用于验证、测试。
- 三个隐藏层 512/512/29，每层 Linear → BatchNorm → ReLU → Dropout；末层 Linear → Sigmoid。
- dropout=0.13254228667616397；AdamW，lr=0.0038369729910356348，weight_decay=1.0931541508233718e-05；batch_size=64（原脚本可能为避免单样本批次调小）；MSELoss。
- 最多 600 epoch，验证 RMSE 连续 100 epoch 不改善即停止；CPU。
- 保存训练/验证/测试误差表及散点图、模型三件套。代码没有执行超参数搜索或自动合并实验反馈数据；每次调用只训练传入的一份 X/y。

网络定义与两个预测脚本在 AST 层面一致。实际 loader 已取得并确认接口。没有获得实际权重，因此尚未验证真实模型能否加载、历史 X/y 实物是否与编码脚本对应及论文指标能否复现。

详细问题与验证范围见 `CODE_REVIEW.md` 和 `verification.json`。

## 6. 训练集编码与辅助检查

- 编码按 CSV 行序进行，batch_size=32。特征顺序为 embedding、T_C、Rate_C；X/y 均保存为 float32。预期为 `[N,770]` 和 `[N]`，实际 embedding 宽度仍以当前 loader 为准，脚本未强制断言 768。
- T_C 缺失填25；Rate_C缺失填1。标签直接转数值，不计算 Qx/Q1C、不自动将百分数除以100。
- 实际 loader 的 load_smi_ted 在加载后调用 model.eval()，extract_embeddings 也调用 encoder.eval()；encode_raw 通过 cpu().detach().numpy() 收集结果，再构造返回 Tensor。因此已确认返回向量脱离梯度图。编码入口未显式 no_grad 属于运行开销问题，不能据此断言输出仍带梯度。
- `check_npy.py` 是辅助诊断，输出警告不等于严格验证通过。等长不证明逐行正确配对；其标准化函数是在整个临时副本重新 fit_transform 后展示统计，并不检查原 X 是否已标准化。该临时 Scaler不会保存，也不会交给训练脚本；实际训练仍只在训练集拟合 Scaler。
