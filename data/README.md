# 数据接口与获取说明

当前不包含 15,824 条初始提取记录、163 条清洗后的文献建模记录或实验原始数据。163 是作者报告的清洗后记录数，是否包含反馈数据及各阶段 X/y 的实际行数仍需以原文件核实；记录数不等于独立配方数。

## 1. 组成换算输入

`Unify.py` 读取 Excel：

- 溶剂：`Solvent1,ration1,…,Solvent4,ration4`。
- 添加剂：`Additive1,Concentration1,Additive2,Concentration2`。
- 盐：`Lithium salt 1,Concentration3,…,Lithium salt 3,Concentration5`。
- 物性表需 `Short Name`，包含 `Specific Gravity` 的密度列及包含 `MW` 的分子量列；默认路径 `data/private/electrolyte_properties.xlsx`。单位按原脚本假定为 g/mL、g/mol。
- 输出保留原表，只增加 `Unified_mole_ratio` 字符串，摩尔份额保留四位小数；不自动整理为六槽位。

换算代码包含不同质量/体积基准的假设，其适用条件和当前边界见 `ML_code/CODE_REVIEW.md`，不能把所有输入都视为已完成严格等价换算。

## 2. 六槽位与序列输入

`replace.py` 要求已有 `solvent1,solvent2,solvent3,solvent4,salt1,salt2` 六个小写列；用物性表 `Short Name→SMILES` 替换。未匹配缩写会原样保留，需要人工检查，避免当作 SMILES 传入。

`smiles_integrate.py` 还需要六个对应的 `*_mol_pct` 列。虽然列名含 pct，**其输入数值实际要求为 0–1 摩尔分数**：代码会乘以 100，再输出五位小数的摩尔百分数。空槽输出 `O<sep>0`，其中 O 是占位符，不表示加入水。序列顺序固定为四个溶剂槽、两个盐槽。需从真实中间表核实添加剂如何并入这些槽位、第三种盐及超过槽位的组分如何处理。

导出 CSV：

| 列 | 含义/行为 |
| --- | --- |
| `seqA` | 六组 `SMILES<sep>摩尔百分数`，以 `<sep>` 连接 |
| `T_C` | 数值温度 °C；不能转为数值的 Temperature 值按原代码填 25 |
| `Rate_C` | 倍率 C，来自 `Rate_C` 或 `Rate`；充电/放电含义需由源记录定义 |
| `y_retention` | 原 `Retention efficiency` 列直接转 float，代码不计算或归一化标签 |

论文标签为同一倍率测试系列的 `R_x=Q_x/Q_1C`。代码列名不能证明数据已完成该计算，也不能证明它不是循环容量保持率。输入必须经过作者实际标签核验。当前 MLP 使用 Sigmoid，训练目标应采用比例表示；不能直接把 72.1 当作 0.721 输入。

## 3. 训练数组与候选

- `X_fastcharge_data.npy`：二维数值特征；已取得的训练编码和预测代码均按 embedding、T_C、Rate_C 顺序拼接；真实数组与 loader 数值仍待验证。
- `y_fastcharge_data.npy`：对应顺序的一维目标。
- `candidate_materials.xlsx`：`Short Name,SMILES,MW (g/mol)`，用于候选质量份额→摩尔百分数换算；这是一张组分参数表，不是完整文献数据库。
- 生成的 `place.csv` 含 `subspace,s1,x1_wt,…,s4,x4_wt,salt1,x5_wt,salt2,x6_wt,mol_sum,seqA`。`x*_wt` 是 0–1 质量分数，`seqA` 使用摩尔百分数，二者不可混用。

默认私有输入放在 `data/private/`；生成结果放在 `outputs/`。两者不随本文件包发布。数据获取条件以作者最终 Data availability 声明为准；未公开数据时，不声称读者可以只凭代码复现训练指标。

`encode_fastcharge_embeddings.py` 将上述序列 CSV 按原行序转换为 X/y，batch_size=32，均保存为 float32。缺失温度填25、缺失倍率填1，标签原样数值化；它不补做文献核验或 Rx计算。`check_npy.py` 只提供诊断，行数匹配不证明逐行配对正确。
