# 🏆 奥运会奖牌预测 - 基于改进GRU的深度学习模型

## ✨ 项目亮点

| 维度                | 创新点                                                                 |
|-------------------- |----------------------------------------------------------------------|
| 🧠 模型架构         | 改进GRU网络结构，双隐藏层+注意力机制集成动态加权MSE损失函数              |
| 📊 数据处理         | 将25万+运动员微观数据转化为国家年度特征的面板数据                       |
| ⏳ 时间序列         | 独创滑动时间窗口处理，捕捉历史时序特征                                 |
| 🎯 预测维度         | 总奖牌/金银铜分项预测，提供95%置信区间                                |
| ⚖️ 数据平衡         | 对数变换+动态加权策略，MAE提升21.1%，MSE降低超50%                     |

## 📁 文件结构

```bash
.
├── data/               # 数据存储
│   ├── data_raw/           # 原始数据（1976-2024年运动员数据）
│   └── data_processed/     # 转化为pickle的原始数据，方便读取；处理后的面板数据
├── docs/              # 项目文档
│   └── 基于GRU神经网络的奥运会奖牌预测.pdf  # 完整项目流程与技术细节PPT
├── results/           # 预测结果
│   ├── 2024预测结果.xlsx  # 测试集预测vs真实值
│   └── 2028预测结果.xlsx  # 2028年各国奖牌预测（含置信区间）
├── src/               # 源代码
│   ├── data_anaysis.py    # 数据预处理
│   ├── gru_neural_network.py          # 改进GRU模型
└── README.md
```

## 📈 预测结果示例

国家       | 总奖牌预测 | 金牌预测 | 银牌预测 | 铜牌预测 | 95%置信区间
-----------|------------|----------|----------|----------|-----------
美国       | 108 ± 5    | 42 ± 3   | 35 ± 2   | 31 ± 2   | [103,113]
中国       | 95 ± 6     | 38 ± 4   | 32 ± 3   | 25 ± 2   | [89,101]
英国       | 63 ± 4     | 22 ± 2   | 20 ± 2   | 21 ± 2   | [59,67]

*(注: 示例数据，真实预测见docs目录)*

## 🛠️ 技术栈

- ​**数据处理**: 
  ![Pandas](https://img.shields.io/badge/Pandas-1.4+-blue.svg)
  ![NumPy](https://img.shields.io/badge/NumPy-1.22+-orange.svg)
  
- ​**深度学习**: 
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
  ![Scikit-learn](https://img.shields.io/badge/ScikitLearn-1.2+-yellowgreen.svg)

- ​**可视化**: 
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-brightgreen.svg)
  ![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-navy.svg)

## 📜 许可协议

本项目采用 [MIT License](LICENSE)，欢迎学术研究/商业应用，需注明出处。

## 🤝 参与贡献
1. Fork 本仓库
2. 创建分支 (`git checkout -b feature/improvement`)
3. 提交修改 (`git commit -m 'Add some feature'`)
4. 推送分支 (`git push origin feature/improvement`)
5. 新建 Pull Request
