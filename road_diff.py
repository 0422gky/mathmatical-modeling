import pandas as pd

# 读取附件三和附件四
file_path_3 = "./data/附件3.csv"
file_path_4 = "./data/附件4.csv"

try:
    data3 = pd.read_csv(file_path_3)
    data4 = pd.read_csv(file_path_4)
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()

# 提取路径为集合
paths_3 = set(zip(data3['始发分拣中心'], data3['到达分拣中心']))
paths_4 = set(zip(data4['始发分拣中心'], data4['到达分拣中心']))

# 找出新增和删除的路径
added_paths = paths_4 - paths_3  # 附件四中新增的路径
removed_paths = paths_3 - paths_4  # 附件四中删除的路径

# 输出结果
print("新增的路径：")
for path in added_paths:
    print(f"{path[0]} -> {path[1]}")

print("\n删除的路径：")
for path in removed_paths:
    print(f"{path[0]} -> {path[1]}")