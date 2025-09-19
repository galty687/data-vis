# Python 入门



## 安装 Python

- **Windows**
  1. 前往 [Python 官网](https://www.python.org/downloads/) 下载最新版本。
  2. 安装时勾选 **Add Python to PATH**，然后点击 Install。
  3. 打开命令提示符 (cmd)，输入：
     ```bash
     python --version
     ```
     显示版本号则安装成功。

- **macOS / Linux**
  - macOS 通常自带 Python，但版本可能较旧，可以通过 [Homebrew](https://brew.sh/) 安装：
    ```bash
    brew install python
    ```


---

## 常见语法

  ```python
  # 输出
  print("Hello, Python!")

  # 变量
  name = "Alice"
  age = 18

  # 条件语句
  if age >= 18:
      print("Adult")
  else:
      print("Child")

  # 循环
  for i in range(5):
      print(i)

  # 函数
  def greet(person):
      return f"Hello, {person}"

  print(greet(name))
  
  # 字典操作
  person = {"name": "Alice", "age": 18}
  print(person["name"])

# 列表切片
  numbers = [1, 2, 3, 4, 5]
  print(numbers[1:4])  # 输出 [2, 3, 4]
  ```



## 文件读写

```python
# 写入文件
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("这是一个测试文件。\n")

# 读取文件
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)
```



## 安装与使用新库

```python
# 安装 requests 库
pip install requests

# 升级 pip
pip install --upgrade pip

#安装完成后，在代码中使用 import：
import requests

response = requests.get("https://www.python.org")
print(response.status_code)
```

