import pandas as pd
import matplotlib.pyplot as plt

# yolo
df = pd.read_csv(r"runs\classify\train\results.csv")

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
plt.plot(df.index, df["metrics/accuracy_top5"], label="metrics/accuracy_top5")
# 添加图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
