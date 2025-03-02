import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,kurtosis

#利用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''
0.特征工程，加入部分特征，对数变换，赋权
'''
all_data = pd.read_excel(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\特征聚类汇总.xlsx')

#计算2024年奖牌的峰度与偏度
data_2024 = all_data[all_data['Year']==2024]
skew_kurt_list = ['Total_medal','Gold','Silver','Bronze']
for i in skew_kurt_list:
    print(f'{i}的峰度为:{kurtosis(data_2024[i])}')
    print(f'{i}的偏度为:{skew(data_2024[i])}')
#构建东道主特征
host_data = pd.read_excel(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\东道主.xlsx')
data = pd.merge(all_data,host_data,on=['NOC','Year'],how='left')
data['Is_host'] = data['Is_host'].fillna(0)
data.drop(columns=['No medal'],inplace=True)

#构造东道主的一阶提前
data["next_host"] = data.groupby("NOC")["Is_host"].shift(-1)
data["next_host"] = data["next_host"].fillna(0)
#指定2024年美国为东道主已知
data.loc[(data['Year'] == 2024) & (data['NOC'] == 'USA'), 'next_host'] = 1

#计算奖牌效率
data['medal_eff'] = (data['Total_medal'] / data['athlete_num']).fillna(0)
data['gold_eff'] = (data['Gold'] / data['athlete_num']).fillna(0)
data['silver_eff'] = (data['Silver'] / data['athlete_num']).fillna(0)
data['bronze_eff'] = data['Bronze'] / data['athlete_num'].fillna(0)

#对国家进行赋权
def assign_weight(data,column):
    '''
    输入需要赋权的dataframe与需要赋权的列的列表，返回赋权后的dataframe
    '''
    #根据奖牌数直接赋权
    data['weight'] = data[column].apply(lambda x: 40 if x >=75 else (20 if x >= 40 else (10 if x >= 20 else (5 if x >= 10 else (2 if x > 0 else 1)))))
    #将权重列设置为第三列
    weight_column = data.pop('weight')
    data.insert(2, 'weight', weight_column)
    return data

data = assign_weight(data,'Total_medal')

#对特征与目标进行对数变换
def log_transform(data,column):
    '''
    输入需要对数变换的dataframe与需要对数变换的列的列表，返回对数变换后的dataframe
    '''
    #用ln(x+1)进行变换
    data[column] = np.log1p(data[column])
    return data

data = log_transform(data,['Total_medal','Gold','Silver','Bronze','Total_equ_athlete','athlete_num','Event'])
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(0)

feature = data.drop(columns=['NOC','Year','weight'])
correlation_matrix = feature.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')


'''
1.时间序列分割，滑动窗口法
'''
def sliding_window(data,look_back,output_size=1):
    '''
    输入需要创建滑动窗口的数据，窗口大小，输出大小，返回创建的时间窗口样本
    '''
    x,y,weight = [],[],[]
    #遍历每个国家创建滑动窗口
    for i in data['NOC'].unique().tolist():
        NOC_data = data[data['NOC']==i]
        #遍历每个国家的每一届创建滑动窗口
        for j in range(len(NOC_data)-look_back):
            x.append(NOC_data.iloc[j:j+look_back,3:].values) #特征排除国家和年份与权重，获取numpy数组
            y.append(NOC_data.iloc[j+look_back][['Total_medal','Gold','Silver','Bronze']].values.astype(np.float32))#预测奖牌数
            weight.append(NOC_data.iloc[j+look_back]['weight'])
    return np.array(x),np.array(y),np.array(weight)

'''
2.利用滑动窗口法创建训练集，验证集，测试集样本
'''
def train_val_test_split(start,end,val,test,data,look_back):
    '''
    输入时间序列数据，起始年份，结束年份，验证集年份，测试集年份
    返回训练集，验证集，测试集，训练集与验证集权重
    '''

    train_x,train_y = [],[]
    val_x,val_y = [],[]
    test_x,test_y = [],[]
    train_weight = []
    val_weight = []

    look_back_years = look_back*4
    train_data = data.query('Year >= @start and Year < @val')
    val_data = data.query('Year <= @val and Year >= (@val-@look_back_years)')
    test_data = data.query('Year >= (@test-@look_back_years)')

    train_x,train_y,train_weight = sliding_window(train_data,look_back)
    val_x,val_y,val_weight = sliding_window(val_data,look_back)
    test_x,test_y,test_weight = sliding_window(test_data,look_back)

    return train_x,train_y,train_weight,val_x,val_y,val_weight,test_x,test_y,test_weight

#创建训练集，验证集，测试集
train_x,train_y,train_weight,val_x,val_y,val_weight,test_x,test_y,test_weight = train_val_test_split(1976,2016,2020,2024,data,3)

print(f'训练集：{train_x.shape},{train_y.shape}')
print(f'验证集：{val_x.shape},{val_y.shape}')
print(f'测试集：{test_x.shape},{test_y.shape}')
print("train_x 数据类型:", train_x.dtype)
print("train_y 数据类型:", train_y.dtype)
print("val_x 数据类型:", val_x.dtype)
print("val_y 数据类型:", val_y.dtype)

'''
3.利用pytorch构建GRU神经网络
'''
class AttGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers = 2,batch_first=True,dropout = 0.3)
        
        self.attention = nn.Sequential(  # 时间步注意力层
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size, output_size)
      
    def forward(self, x):
        gru_out, _ = self.gru(x)  # (batch, seq, hidden)
        
        att_weights = self.attention(gru_out)  # (batch, seq, 1)
        context = torch.sum(att_weights * gru_out, dim=1)  # (batch, hidden)
        
        return self.fc(context), att_weights

'''
4.定义训练函数，训练模型
'''
def weighted_mse_loss(outputs, targets, weights):
    # 新增动态权重组件
    error = outputs - targets
    adaptive_weights = 1 + torch.log1p(torch.abs(error))
    
    # 组合静态权重和动态权重
    combined_weights = adaptive_weights * weights.unsqueeze(1)
    
    # 修改后的损失计算（保留原损失结构）
    loss = torch.mean(combined_weights * (outputs - targets) ** 2)
    return loss
def train_model(model,X_train,y_train,X_val,y_val,train_weight,val_weight,epochs=8000):
    #定义损失函数和优化器，采取动态学习率衰减策略
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)

    train_losses = []
    val_losses = []
    counter = 0
    patience = 300

    best_val_loss = float('inf')

    #将numpy数组转化为tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    train_weight = torch.tensor(train_weight, dtype=torch.float32).to(device)
    val_weight = torch.tensor(val_weight, dtype=torch.float32).to(device)
    
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)[0]
        train_loss = weighted_mse_loss(outputs, y_train, train_weight)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)[0]
            val_loss = weighted_mse_loss(val_outputs, y_val, val_weight)
            val_losses.append(val_loss.item())
        
        if (epoch-1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            torch.save(best_model,'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        model.load_state_dict(best_model)
    return train_losses,val_losses
        
input_size = train_x.shape[2]
model = AttGRU(input_size).to(device)
print('开始训练')
train_losses, val_losses = train_model(model, train_x, train_y, val_x, val_y,train_weight,val_weight)

'''
5.利用测试集评估模型，解释特征
'''
def evaluate_model(model,X_test,y_test):
    model.eval()
    y_true = []
    y_pred = []
    NOC = all_data['NOC'].unique().tolist()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        outputs, att_weights = model(X_test)
        for i in range(len(y_test)):
            y_true.append(y_test[i].cpu().numpy())
            y_pred.append(outputs[i].cpu().numpy())
    # 注意力分析年份重要性
    time_importance = att_weights.cpu().mean(dim=(0,2)).numpy()  # 各时间步重要性
    print("历史年份重要性排序:", [f"t-{i}" for i in np.argsort(-time_importance)[:3]])

    # 将列表转换为NumPy数组，对预测值取指数
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #计算指标
    medal_list = ['Total_medal','Gold','Silver','Bronze']
    for i,medal in enumerate(medal_list):
        medal_true = y_true[:,i]
        medal_pred = y_pred[:,i]
        mse = mean_squared_error(medal_true,medal_pred)
        mae = mean_absolute_error(medal_true,medal_pred)
        r2 = r2_score(medal_true,medal_pred)
        print(f'{medal} MSE:{mse:.4f} MAE:{mae:.4f} R2:{r2:.4f}')

    total_medal_pred = y_pred[:, 0]
    gold_pred = y_pred[:, 1]
    silver_pred = y_pred[:, 2]
    bronze_pred = y_pred[:, 3]
    test_result = pd.DataFrame({
        'NOC': NOC,
        'Total_medal_True': y_true[:, 0],
        'Total_medal_Pred': total_medal_pred,
        'Gold_True': y_true[:, 1],
        'Gold_Pred': gold_pred,
        'Silver_True': y_true[:, 2],
        'Silver_Pred': silver_pred,
        'Bronze_True': y_true[:, 3],
        'Bronze_Pred': bronze_pred
    })
    #按照预测结果从高到低排列
    
    test_result = test_result.sort_values(by='Total_medal_True',ascending=False)
    return test_result
val_2020_result = evaluate_model(model,val_x,val_y)
print(val_2020_result)
predicting2024_result = evaluate_model(model,test_x,test_y)
print(predicting2024_result)

predicting2024_result.to_excel(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\2024测试结果.xlsx',index = False)

'''
6.利用梯度分析模型
'''
def gradient_analysis(model, X_test):
    X_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True,device = device)
    model.train()
    if X_tensor.grad is not None:
        X_tensor.grad.zero_()

    output = model(X_tensor)[0][:, 0]  # 分析总奖牌预测
    output.sum().backward()
    X_tensor.retain_grad()
    gradients = X_tensor.grad.detach().cpu().numpy()
    return np.mean(np.abs(gradients), axis=(0,1))

# 在评估后调用
grad_importance = gradient_analysis(model, test_x)
grad_df = pd.DataFrame({
    'features': data.columns[3:],
    'gradient_importance': grad_importance
}).sort_values('gradient_importance', ascending=False)

# 新增可视化代码
plt.figure(figsize=(12, 8))
# 取重要性前15的特征，颜色映射使用渐变色
grad_df.head(10).sort_values('gradient_importance', ascending=True).plot(
    kind='barh',
    x='features',
    y='gradient_importance',
    color=plt.cm.Blues(np.linspace(0.4, 1, 15)),  # 使用蓝色渐变
    legend=False
)
plt.title('Feature Importance via Gradient Analysis', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('')  # 隐藏y轴标签
plt.grid(axis='x', linestyle='--', alpha=0.7)  # 添加纵向网格线
plt.tight_layout()
plt.savefig('gradient_importance.png', dpi=300, bbox_inches='tight')

print(grad_df)

'''
7.预测2028年奖牌数
'''
data_to_predict = data.query('Year >= 2016 and Year <= 2024')
predict_x = []
for i in all_data['NOC'].unique().tolist():
    NOC_predict_data = data_to_predict[data_to_predict['NOC']==i]
    predict_x.append(NOC_predict_data.iloc[:,3:].values)

predict_x = np.array(predict_x)
#转化为张量
predict_x = torch.tensor(predict_x, dtype=torch.float32).to(device)
print(predict_x.shape)

#预测2028年奖牌数，蒙特卡洛模拟
def mcdropout_predict(model, X, n_samples=100):
    """执行蒙特卡洛Dropout采样"""
    model.train()  # 保持dropout激活
    with torch.no_grad():
        samples = [model(X)[0].cpu().numpy() for _ in range(n_samples)]
    return np.stack(samples)  # (n_samples, batch_size, output_size)

# 执行预测并计算统计量
samples = mcdropout_predict(model, predict_x, 100)
samples = np.expm1(samples)  # 指数逆变换

# 构建结果DataFrame
medians = np.median(samples, axis=0)
lower = np.percentile(samples, 2.5, axis=0)
upper = np.percentile(samples, 97.5, axis=0)

result_rows = []
for i in range(len(medians)):
    row = [all_data['NOC'].unique().tolist()[i]]  # 国家顺序与输入一致
    for j in range(4):  # 四种奖牌类型
        row += [medians[i,j], lower[i,j], upper[i,j]]
    result_rows.append(row)

columns = ['NOC']
for medal in ['Total_medal', 'Gold', 'Silver', 'Bronze']:
    columns += [f'{medal}_pred', f'{medal}_lower', f'{medal}_upper']

predict_result = pd.DataFrame(result_rows, columns=columns).sort_values(
    by='Total_medal_pred', ascending=False)

#按照预测结果从高到低排列
predict_result = predict_result.sort_values(by='Total_medal_pred',ascending=False)
print(predict_result)
predict_result.to_excel(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\2028预测结果.xlsx',index = False)

''' 
8. 预测从未获奖国家的奖牌概率 
'''
def probility_analysis(year,dataframe):
    '''
    输入年份，对应结果，输出该年份从未获奖国家的预测概率与实际首次获奖国家
    '''
    df = data[data['Year'] == year and data['Year'] == year -4]
    # 提取从未获奖国家
    never_won = df.query(f'Year == {year}-4 and never_won_medal == 0')
    never_won_country = never_won['NOC'].unique().tolist()
    #实际获奖
    win_country = never_won.query(f'Year == {year} and Is_won_medal == 1')['NOC'].unique().tolist()
    #预测获奖

print('代码运行结束')

epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




    


        




