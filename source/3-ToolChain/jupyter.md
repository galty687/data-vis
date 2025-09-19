# Jupyter

[Jupyter Notebook](https://jupyter.org)是一个交互式编程环境，可以在网页界面中运行 Python 代码并即时看到结果。

特点：
  - 支持 **代码 + 文本 + 图表** 混合展示，特别适合数据分析和可视化教学。
  - 文本单元格支持 **Markdown**，方便写说明文档。
  - 广泛应用于数据科学、机器学习、工程管理数据分析。



## 安装方法 1

安装 VS Code 扩展

1. 打开 VS Code。

2. 点击左侧 **扩展 (Extensions)** 图标。

3. 搜索并安装以下扩展：

   - **Python** （微软官方）
   - **Jupyter** （微软官方）


## 安装方法 2

在终端中直接使用 `pip`安装：

```bash
pip install notebook
```

启动：

```bash
jupyter notebook
```





## 常用快捷键

Markdown 和 Code 都可以。

- **Shift + Enter**：运行当前单元格，并跳到下一个单元格
- **Ctrl + Enter**：运行当前单元格，不跳转
- **Alt + Enter**：运行当前单元格，并在下方新建一个单元格



## Markdown 基本语法

```
# 一级标题
## 二级标题
### 三级标题

这是**加粗**，这是*斜体*，这是`代码`。

- 列表项 1
- 列表项 2

1. 有序列表 1
2. 有序列表 2

公式示例：$y = x^2 + 1$

代码块：
```python
print("Hello, Markdown in Jupyter!")
```

