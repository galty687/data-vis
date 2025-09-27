# 西纳图



西纳图（Sina plot，有时也叫 **Sina 图表**、**Sina 图**）其实是一种 **数据分布可视化图**，它是 **小提琴图 (violin plot)** 的变体，融合了 **箱线图/小提琴图 + 散点图** 的优点。



1. **展示数据分布**

   - 像小提琴图一样，能显示数据在不同取值上的分布密度。
   - 能直观看出数据是否对称、是否存在偏态。

2. **避免过度简化**

   - 箱线图只能显示分位数信息，小提琴图只能显示密度曲线；
   - 西纳图会把每个样本点也画出来，避免只看到整体趋势。

3. **对比分组数据**

   - 可以用来对比不同类别（比如不同实验组、不同班级）的数据分布差异。

4. **兼顾整体与细节**

   - 既能看到整体分布形状（类似小提琴图），
   - 又能看到每个具体数据点（类似散点图）。

   

> **西纳图的作用就是在展示分布趋势的同时，把每个数据点也保留下来**，它比单纯的箱线图、小提琴图更全面，也比单纯的散点图更有结构性。



```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 日期处理
df["DATE"] = pd.to_datetime(df["DATE"])
df["MONTH"] = df["DATE"].dt.month

# 转换温度为摄氏
df["TEMP_C"] = (df["TEMP"] - 32) * 5/9

plt.figure(figsize=(12,6))

# 背景小提琴图
sns.violinplot(
    data=df, 
    x="MONTH", 
    y="TEMP_C", 
    inner=None,      # 不显示箱线图，只要分布形状
    color="lightgray"
)

# 前景散点图（模拟 sina）
sns.stripplot(
    data=df, 
    x="MONTH", 
    y="TEMP_C", 
    jitter=True, 
    alpha=0.6, 
    size=2, 
    color="tab:blue"
)

plt.title("Sina-style Chart of Monthly Temperatures in Beijing (2024, °C)")
plt.xlabel("Month")
plt.ylabel("Mean Daily Temperature (°C)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
```



![sina](images/sina.png)