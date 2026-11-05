# 需求质量评估智能体（实验4 方案B：LangChain + LangGraph 代码版）

## 一、目录结构

```
D:\python_project\req_expert_system
├─ .venv                       项目虚拟环境（Python 3.9.12）
├─ .env                        DeepSeek 密钥与模型配置
├─ .env.example                配置模板（提交或换机时用）
├─ main.py                     命令行版主程序：需求质量评估智能体
├─ app.py                      网页版入口：Gradio 界面（http://127.0.0.1:7860）
├─ 启动网页端.bat               双击启动网页版，自动打开浏览器
├─ requirements.txt            依赖清单
├─ knowledge\
│   └─ 需求工程知识库.txt       本地知识库，用于检索增强（RAG）
└─ logs\
    └─ 对话记录_YYYYMMDD.txt    自动保存的测试记录，写实验报告直接用
```

## 二、在 PyCharm 中打开并绑定这个环境

1. 打开 PyCharm，菜单 File 到 Open，选择文件夹 `D:\python_project\req_expert_system`。
2. 打开后看右下角状态栏，点 Python 解释器，选 Add Interpreter 到 Add Local Interpreter。
3. 左侧选 Virtualenv Environment，类型选 Existing，解释器路径填：
   `D:\python_project\req_expert_system\.venv\Scripts\python.exe`
4. 点 OK 确认。之后 PyCharm 右下角应显示 `Python 3.9 (req_expert_system)`。
5. 如果 PyCharm 底部没有 Terminal，点菜单 View 到 Tool Windows 到 Terminal，确认终端前缀有 `(.venv)`。

## 三、填入密钥并运行命令行版

1. 打开 `.env`，在 `DEEPSEEK_API_KEY=` 后面直接粘贴密钥，不要加引号、不要留空格。
2. 右键 `main.py` 选 Run main，或终端执行 `python main.py`。
3. 输入一条软件需求，观察评估输出。输入 `退出` 结束，输入 `知识` 查看检索命中的知识片段。

## 四、运行网页版

方式一：双击项目里的 `启动网页端.bat`，会自动打开浏览器。
方式二：PyCharm 右键 `app.py` 选 Run app，或终端执行 `python app.py`，然后手动访问 http://127.0.0.1:7860。

界面说明：

| 区域 | 作用 |
|---|---|
| 左侧输入框 | 粘贴软件需求，三个示例按钮可一键填入测试用例 |
| 开始评估 | 流式输出，模型边生成边显示，单次约 8 到 30 秒 |
| 清空会话 | 换一个新 thread_id，等于清空 LangGraph 的多轮记忆并重置计数 |
| 导出记录 | 下载 logs 目录下当天的对话记录 txt |
| 右侧结果区 | 五个维度打分、问题清单、改写建议，Markdown 渲染 |
| 折叠面板 | 展示本次检索命中的知识片段，用来证明 RAG 真的在工作 |

## 五、自检命令

```
python main.py --check
```

输出会显示各组件版本、知识库切片数、密钥是否已配置，用于排查环境问题。

## 六、可调参数（改 .env 即可）

| 变量 | 说明 | 默认值 |
|---|---|---|
| DEEPSEEK_API_KEY | DeepSeek 平台申请的密钥 | 空 |
| DEEPSEEK_BASE_URL | DeepSeek 接口地址 | https://api.deepseek.com |
| MODEL_NAME | 模型名称，可选 deepseek-chat 或 deepseek-reasoner | deepseek-chat |
| THREAD_ID | LangGraph 记忆会话编号，换编号等于重开一段记忆 | 实验4_需求评估 |

## 七、版本约束说明（换机重装时必看）

Python 3.9 决定了部分依赖必须锁版本，否则装出来的组合会直接报错：

| 包 | 锁定版本 | 原因 |
|---|---|---|
| greenlet | 3.1.1 | 3.2.x 没有 cp39 的 Windows 预编译包，会尝试本地编译并失败 |
| gradio | 4.44.1 | gradio 5.x 要求 Python 3.10 以上 |
| huggingface_hub | 0.36.2 | 1.x 删掉了 HfFolder，gradio 4.x 导入即崩 |
| pydantic | 2.9.2 | 2.13 生成的 schema 带 additionalProperties: false，会让 gradio 4.x 的接口信息解析抛 TypeError |
| fastapi | 0.115.6 | 与 gradio 4.44 同期版本，搭配 0.128 时页面接口 500 |

重装顺序：先装 `requirements.txt`，如果 pip 擅自升级上述包，按上表回退。

## 八、实验记录建议

1. 至少跑三组测试需求，控制台与网页各截一次图，日志文件 `logs\对话记录_*.txt` 可直接粘进报告附录。
2. 对比实验：同一条需求分别喂给扣子版（方案A）与这套程序（方案B），比较输出结构与细节差异。
3. RAG 有效性验证：输入一条非软件需求（例如“今天天气怎么样”），智能体应拒答；再输入真实需求，看折叠面板里的命中片段是否与需求相关。
4. 网页版截图建议：输入框 + 评分区 + 命中片段面板三张，能同时体现界面、评估能力、检索增强。
