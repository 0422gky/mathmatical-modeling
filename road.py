import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取附件三
file_path_3 = "./data/附件3.csv"

try:
    # 读取附件三的数据
    data3 = pd.read_csv(file_path_3)
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()

# 创建有向图
G = nx.DiGraph()

# 添加边到图中
for _, row in data3.iterrows():
    origin = row['始发分拣中心']  # 确保列名与附件三中的列名一致
    destination = row['到达分拣中心']  # 确保列名与附件三中的列名一致
    G.add_edge(origin, destination)

# 绘制有向图
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # 使用 spring 布局
nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=10, font_color="black", arrowsize=2)
plt.title("附件三：分拣中心之间的有向图", fontsize=16)
plt.show()