import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

print('开始运行')
#data = pd.read_excel(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\运动员数据v1.xlsx')
#data.to_pickle(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\运动员数据v1.pkl')
data = pd.read_pickle(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\运动员数据v1.pkl')
print('读取完成')
data['Total_medal'] = data['Gold'] + data['Silver'] + data['Bronze']
#按运动员去重，确保每个项目奖牌统计只统计一次

'''
1.有关奖牌数据特征整合
'''
data_athlete_dropdulicates = data.drop_duplicates(subset=['NOC','Year','Sport','Event','Medal'], keep='first')

#转化为每个国家每年的奖牌统计
df_country = data_athlete_dropdulicates.drop(columns=['Name','Sex','Team','Medal'])
df_country = df_country.groupby(['NOC','Year']).agg({'Gold':'sum','Silver':'sum','Bronze':'sum','Total_medal':'sum','No medal':'sum'}).reset_index()
df_country['Total_equ_athlete'] = df_country['Gold'] + df_country['Silver'] + df_country['Bronze']+df_country['No medal']

#补充奥运会未参加国的数据，进行关联与填充

def filter_country(df,column):
    '''
    输入需要补充年份的dataframe与补充的所依据的列名，返回自动补充与剔除的dataframe
    '''
    all_country = df['NOC'].unique()
    all_years = df['Year'].unique()
    all_combinations = pd.MultiIndex.from_product([all_country, all_years], names=['NOC', 'Year'])
    all_combinations = all_combinations.to_frame(index=False)

    #将原始数据与所有组合进行合并
    df = pd.merge(all_combinations, df, on=['NOC', 'Year'], how='left')

    #填充缺失值为0
    df = df.fillna(0)
    df = df.sort_values(by=['NOC','Year'])

    df_noexist = df.query(f"(Year == 2020 or Year == 2024) and {column} == 0")
    country_noexist = df_noexist['NOC'].unique()
    
    df_filtered_country = df[~df['NOC'].isin(country_noexist)]
    return df_filtered_country
df_filtered_country = filter_country(df_country,'Total_equ_athlete')


def growth_rate_cal(columns_list,df):
    '''
    输入需要计算增长率的列名列表，以及需要计算的dataframe
    输出计算好的增长率的dataframe
    '''
    #计算增长率
    for i in columns_list:
        df[i+'_growth_rate'] = df.groupby('NOC')[i].pct_change()
    

    #处理inf,NAN，将inf填充为1，将NAN填充为0
    for column in columns_list:
        df[column+'_growth_rate'] = df[column+'_growth_rate'].replace([float('inf'), float('-inf')], 1)
        df[column+'_growth_rate'] = df[column+'_growth_rate'].fillna(0)

    #把第一行的增长率填充为0
    for column in columns_list:
        df.loc[df.groupby('NOC').head(1).index,column+'_growth_rate'] = pd.NA
    
    return df

#计算增长率
df_filtered_country = growth_rate_cal(['Gold','Silver','Bronze','Total_medal','Total_equ_athlete'],df_filtered_country)
print(df_filtered_country)

#计算是否参赛
df_filtered_country['Is_participate'] = (df_filtered_country['Total_equ_athlete'] > 0).astype(int)

#计算是否获得奖
df_filtered_country['Is_won_medal'] = (df_filtered_country['Total_medal'] > 0).astype(int)

#计算是否从未获得奖牌
df_filtered_country['never_won_medal'] = df_filtered_country.groupby("NOC")['Total_medal'].transform(lambda x:(x.cumsum()!=0).astype(int))

'''
2.有关运动员数据特征整合
'''
data_athlete = data.drop_duplicates(subset=['Name','Sex','NOC','Sport','Event','Year'], keep='first')

#计算运动员数量
df_athelte = data_athlete.groupby(['NOC','Year']).agg({'Name':'count'}).reset_index()
df_athelte = df_athelte.rename(columns={'Name':'athlete_num'})
df_athelte = filter_country(df_athelte,'athlete_num')

#计算运动员数量增长率
df_athelte = growth_rate_cal(['athlete_num'],df_athelte)

#合并运动员数量与奖牌数据
df_filtered_country = pd.merge(df_filtered_country,df_athelte,on=['NOC','Year'],how='left')

'''
3.有关项目特征数据整合
'''
data_event_dropduplicate = data.drop_duplicates(subset=['NOC','Year','Sport','Event'],keep='first')
data_event_dropduplicate = data_event_dropduplicate.groupby(['NOC','Year']).agg({'Event':'count'}).reset_index()
filtered_event_data = filter_country(data_event_dropduplicate,'Event')
event_data = growth_rate_cal(['Event'],filtered_event_data)
#合并数据
df_filtered_country = pd.merge(df_filtered_country,event_data,on=['NOC','Year'],how='left')

#获取每一年新增项目的名称与数量，并计算每个国家新增项目参与数量
data['specific_event'] = data['Sport']+ ' ' +data['Event']

def event(df):
    '''
    输入需要计算的dataframe
    输出计算好的dataframe
    '''
    event_dict = {}
    new_event_dict = {}
    for year in df['Year'].unique().tolist():
        data_specific_year = df.query(f"Year == {year}")
        event_dict[year] = data_specific_year['Sport'].unique().tolist()
    return event_dict

event_df = event(data)
event_df = pd.DataFrame(list(event_df.items()), columns=['Year', 'Sport'])
#按照年份升序，并调整索引
event_df = event_df.sort_values(by='Year').reset_index(drop=True)
print(event_df)
#计算该年的新增项目与废弃项目

def new_event(df):
    '''
    输入需要计算的dataframe
    计算每一年的真正项目
    输出计算好的dataframe
    '''
    #计算每一年的新增项目
    all_events = set()
    new_event_list = []
    df['new_events_name'] = None
    df['new_events_count'] = 0
    for index,row in df.iterrows():
        current_events = set(row['Sport'])
        if row['Year'] == 1896:
            current_events = set(row['Sport'])
            new_events = current_events
        else:
            new_events = current_events - all_events
        all_events = all_events.union(current_events)
        event_df.at[index,'new_events_name'] = list(new_events)
        event_df.at[index,'new_events_count'] = len(new_events)
    return event_df

event_df = new_event(event_df)
print(event_df)

print("\n2020年新增项目详情:")
print(f"年份: 2020")
print(f"新增项目数量: {event_df[event_df['Year'] == 2020]['new_events_count'].values[0]}")
print("具体项目列表:")
for event in event_df[event_df['Year'] == 2020]['new_events_name'].values[0]:
    print(f"- {event}")


df_filtered_country.to_excel(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\特征聚类汇总.xlsx',index=False)

#绘制2024年的所有国家奖牌数，运动员的散点图
year_2024 = df_filtered_country[df_filtered_country['Year'] == 2024]

plt.figure(figsize=(14, 8))
sns.scatterplot(
    data=year_2024,
    x='Total_medal',    # 横轴设为奖牌总数
    y='athlete_num',    # 纵轴设为运动员数量
    size='Total_medal',  # 点大小表示总参赛人数
    sizes=(50, 300),    # 调整点尺寸范围
    alpha=0.7,
    color='#2E86C1',    # 统一颜色
    edgecolor='black'   # 添加描边
)

plt.title('2024 Medal-Athlete Scatter Plot ', fontsize=14)
plt.xlabel('Total medal', fontsize=12)
plt.ylabel('Athlete number', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 保存高清图片
plt.savefig(r'C:\Users\29306\Desktop\数学建模\美赛相关\2025_Problem_C_Data\复盘\2024散点图.png', 
            bbox_inches='tight', dpi=300)
plt.show()

print('代码运行结束')   
    

    








