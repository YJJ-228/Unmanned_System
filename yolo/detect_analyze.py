import pandas as pd
import matplotlib.pyplot as plt

detect_data_dir = r"runs\detect\train\results.csv"

# yolo
df = pd.read_csv(detect_data_dir)

# 遍历DataFrame的每一列
for column in df.iloc[:, 2:]:
    # 绘制每列的数据
    plt.plot(df.index, df[column], label=column)

# 添加标题和标签
plt.title("Multiple Lines from CSV")
plt.xlabel("X-axis (Index)")
plt.ylabel("Y-axis (Values)")

# 添加图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()


# 准确率视图
plt.plot(df.index, df["metrics/precision(B)"], label="metrics/precision(B)")
# 添加图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
