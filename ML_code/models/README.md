# 模型三件套

当前尚未提供实际模型文件。三份脚本中的保存与加载文件名已经对应：

| 文件 | 训练时保存内容 | 预测时用途 |
| --- | --- | --- |
| `best_model_fixed.pth` | `SprintMLP.state_dict()` | 恢复 MLP 权重及 BatchNorm 状态 |
| `best_scaler_fixed.pkl` | 训练集拟合的 StandardScaler | 对完整输入执行相同标准化 |
| `best_features.npy` | `np.arange(X.shape[1])` | 标准化之后按索引取列；当前训练脚本保存的是全部特征索引 |

第三个文件是**特征索引数组**，不是超参数配置文件，也不是“筛出的重要特征”。三个文件须来自同一次训练。放入本目录下的 `final/`，或用 `MLP_MODEL_DIR` 指向原训练输出目录。

预测顺序：SMI-TED `encode_raw` → 拼接温度、倍率 → StandardScaler → 按特征索引选列 → MLP。代码预期通常为 768+1+1=770 维，但仍须由实际训练数组、loader 及文件确认。

仅有相同文件名不能证明模型相同。取得实物后需记录 SHA256、输入维度、模型阶段及对应 loader/编码设置。未收到的旧阶段权重不补造，不用重新训练的新权重替代论文原权重。
