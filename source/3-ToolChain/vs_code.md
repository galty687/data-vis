# Visual Studio Code

[Visual Studio Code](https://code.visualstudio.com) 是微软推出的一款免费、开源、跨平台的代码编辑器。具有如下优点：

  - 轻量、启动快
  - 支持 **Python / Java / C++ / Web 开发** 等多种语言
  - 拥有强大的 **扩展市场 (Extensions Marketplace)**
  - 内置 Git 支持，适合代码协作

  

## 安装 VS Code

下载安装

1. 访问 [VS Code 官网](https://code.visualstudio.com/)
2. 下载对应平台版本：Windows / macOS / Linux
3. 安装后启动即可使用

推荐安装的扩展

- **Python**
- **Jupyter**
- **Pylance**（智能补全与语法提示）
- **GitLens**（Git 可视化工具）
- **Prettier**（代码格式化）



## GitHub Copilot 使用指南

### 安装扩展
1. 打开 **VS Code**  
2. 点击左侧 **扩展 (Extensions)** 图标  
3. 搜索 **GitHub Copilot**，点击安装  
4. 建议同时安装 **GitHub Copilot Chat**（支持对话式问答）  



### 登录 GitHub 账号
- 第一次启用时，VS Code 会提示用 **GitHub 账号登录**  
- 注意：Copilot 是 **付费服务**（学生可申请教育优惠）  
- 登录成功后，状态栏会显示 Copilot 已启用  


### 启用和设置
- 在 VS Code 设置里搜索 **Copilot**，可配置以下选项：  
  - **Inline Suggestions**：输入时自动弹出代码建议  
  - **Accept Suggestion**：默认快捷键 `Tab` 或 `Ctrl + ]`  
  - **Trigger Suggestion**：手动触发 `Alt + ]`  



## Copilot 使用方式

### 自动补全
在代码中输入时，Copilot 会实时给出灰色提示，按 **Tab** 接受。  

```python
# 输入
def fibonacci(n):

# Copilot 自动生成
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```



### 注释生成代码

写注释 → Copilot 会根据注释生成实现。

```markdown
# 生成一个列表，包含前10个平方数
```



### 对话问答（需要 Copilot Chat）

在编辑器右侧打开 Copilot Chat 面板，可以问：
	•	帮我写一个计算平均值的函数
	•	解释这段代码
	•	优化下面的循环
	
	

### Agent 模式（智能代理）

Copilot 最新支持 Agent 模式，不只是给出代码建议，而是像一个“智能助理”一样，能够执行更复杂的任务。
	•	功能：
	•	理解上下文，不仅生成代码，还能修改已有代码
	•	自动安装缺失的依赖包（例如缺少 matplotlib 时提示安装）
	•	跨文件理解：可以回答“这个函数在项目里是干什么的？”
	•	任务分解：可以根据自然语言请求生成多步解决方案

> 用法示例：
> 创建一个术语库管理系统，支持中英文双语与基础检索、分页、导入导出。



- 