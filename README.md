# 利用改进的GRU模型对奥运会奖牌数进行预测

## 项目亮点
- 基于GRU神经网络模型对奥运会奖牌榜进行预测
- 使用1976-2024年的运动员数据，将25万行运动员微观数据转化为国家年份特征的面板数据
- 搭建了超20个维度的特征，利用滑动时间窗口处理##时间序列##数据
- 实现对总奖牌，金牌，银牌，铜牌以及置信区间（95%）的预测，超越了传统XGBoost方法，有较强泛化能力
- 使用对数变换+动态加权MSE损失函数处理数据不平衡问题，MAE精度较基准方法提升21.1%，MSE提升超50%！

## 文件结构
-data：原始数据与处理后数据
-docs：项目的PPT展示，详细展示了这个项目流程，思考过程，具体细节与最终结果，运用了大量的图表展示
-results：2024年测试集的真实值与预测值对比，2028年各个国家奖牌预测（含置信区间）
-scrs：项目Python代码，基于Python3.9开发，数据分析全流程均使用pandas库，利用Pytorch进行训练
