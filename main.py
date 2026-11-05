import os
import sys
import glob
import datetime

import jieba
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from rank_bm25 import BM25Okapi

项目目录 = os.path.dirname(os.path.abspath(__file__))
日志目录 = os.path.join(项目目录, "logs")
知识目录 = os.path.join(项目目录, "knowledge")

load_dotenv(os.path.join(项目目录, ".env"))

接口密钥 = os.getenv("DEEPSEEK_API_KEY", "").strip()
接口地址 = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
模型名称 = os.getenv("MODEL_NAME", "deepseek-chat").strip()
会话编号 = os.getenv("THREAD_ID", "实验4_需求评估").strip()

评估维度 = ["完整性", "无二义性", "可测试性", "一致性", "可行性"]

知识切片 = []
检索索引 = None


def 读取知识库():
    global 知识切片, 检索索引
    文件清单 = sorted(glob.glob(os.path.join(知识目录, "*.txt")) + glob.glob(os.path.join(知识目录, "*.md")))
    if not 文件清单:
        print("[知识库] 未找到知识文档，将不使用检索增强")
        return
    切分器 = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60, separators=["\n\n", "\n", "。", "；", ""])
    原文清单 = []
    for 文件 in 文件清单:
        with open(文件, "r", encoding="utf-8") as 句柄:
            原文清单.append(句柄.read())
    知识切片 = 切分器.create_documents(原文清单)
    分词结果 = [list(jieba.cut(切片.page_content)) for 切片 in 知识切片]
    检索索引 = BM25Okapi(分词结果)
    print("[知识库] 已加载 %d 个文档，切分为 %d 个片段" % (len(文件清单), len(知识切片)))


def 检索知识(查询文本, 数量=3):
    if 检索索引 is None or not 知识切片:
        return []
    查询分词 = list(jieba.cut(查询文本))
    分值 = 检索索引.get_scores(查询分词)
    排序 = sorted(range(len(分值)), key=lambda 序号: 分值[序号], reverse=True)[:数量]
    命中 = [知识切片[序号].page_content.strip() for 序号 in 排序 if 分值[序号] > 0]
    return 命中


def 拼装系统提示(参考知识):
    参考段落 = "\n".join(["· " + 片段 for 片段 in 参考知识]) if 参考知识 else "（本次未检索到与需求直接相关的知识片段）"
    return """你是资深软件需求工程专家，负责对软件需求文本做质量评估与改写。

【本次从本地知识库检索到的参考依据】
%s

【必须评估的五个维度】
完整性：输入、处理、输出、约束、异常处理是否齐全
无二义性：是否存在模糊词、主观词或可作多种解释的表述
可测试性：能否设计出具体测试用例与可判定的验收标准
一致性：条目之间、术语之间是否矛盾
可行性：在技术、周期、成本、法规约束下是否可实现

【输出格式，严格照此结构，不得增删标题】
一、需求原文
（原样引用用户给出的需求文本）

二、维度评分（每项 0 到 10 分，并给出扣分理由）
完整性：X 分，理由
无二义性：X 分，理由
可测试性：X 分，理由
一致性：X 分，理由
可行性：X 分，理由

三、综合得分
（计算五项平均分，保留一位小数，并给出等级：优秀 8 分以上、合格 6 到 8 分、需重写 6 分以下）

四、现存问题清单
（逐条列出问题，每条格式为 问题类型｜所在表述｜为什么是问题）

五、修改建议
1. 改写后的高质量需求（可直接进入需求规格说明书）
2. 补充建议（需要补哪些缺失信息、验收标准怎么写）

【约束】
只围绕软件需求质量评估作答，不闲聊、不回答与需求工程无关的问题。
若用户输入明显不是软件需求，只回复：“请输入一段软件需求描述，我将为你完成需求质量评估。”
""" % 参考段落


def 构建评估节点():
    模型 = ChatOpenAI(api_key=接口密钥, base_url=接口地址, model=模型名称, temperature=0.2)

    def 评估节点(状态):
        需求文本 = 状态["messages"][-1].content
        参考知识 = 检索知识(需求文本, 3)
        系统提示 = 拼装系统提示(参考知识)
        消息序列 = [SystemMessage(content=系统提示)] + list(状态["messages"])
        回复 = 模型.invoke(消息序列)
        return {"messages": [回复]}

    return 评估节点


def 构建智能体():
    图 = StateGraph(MessagesState)
    图.add_node("评估", 构建评估节点())
    图.add_edge(START, "评估")
    记忆 = MemorySaver()
    return 图.compile(checkpointer=记忆)


def 记录对话(需求文本, 评估结果):
    if not os.path.isdir(日志目录):
        os.makedirs(日志目录)
    日志文件 = os.path.join(日志目录, "对话记录_%s.txt" % datetime.date.today().strftime("%Y%m%d"))
    with open(日志文件, "a", encoding="utf-8") as 句柄:
        句柄.write("=" * 60 + "\n")
        句柄.write("[时间] %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        句柄.write("[输入需求]\n%s\n\n" % 需求文本)
        句柄.write("[智能体输出]\n%s\n\n" % 评估结果)


def 自检查():
    from importlib import metadata
    print("项目目录：%s" % 项目目录)
    print("Python：%s" % sys.version.split()[0])
    for 包名 in ["langchain", "langchain-core", "langchain-openai", "langgraph", "rank-bm25", "jieba", "python-dotenv"]:
        try:
            print("%s：%s" % (包名, metadata.version(包名)))
        except metadata.PackageNotFoundError:
            print("%s：未安装" % 包名)
    读取知识库()
    print("接口地址：%s" % 接口地址)
    print("模型名称：%s" % 模型名称)
    print("接口密钥：%s" % ("已配置，长度 %d" % len(接口密钥) if 接口密钥 else "未配置"))
    return 0


def 主程序():
    if "--check" in sys.argv:
        return 自检查()
    print("=" * 60)
    print("需求质量评估智能体（LangChain + LangGraph + 本地知识库检索）")
    print("=" * 60)
    读取知识库()
    if not 接口密钥:
        print("[错误] 未检测到 DEEPSEEK_API_KEY，请先在 .env 中填入密钥")
        return 2
    智能体 = 构建智能体()
    会话配置 = {"configurable": {"thread_id": 会话编号}}
    print("输入软件需求开始评估，输入 退出 结束，输入 知识 查看本次检索到的知识片段")
    while True:
        try:
            输入内容 = input("\n请输入软件需求：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            break
        if not 输入内容:
            continue
        if 输入内容 in ("退出", "exit", "quit"):
            print("已退出")
            break
        if 输入内容 == "知识":
            片段 = 检索知识("需求质量 评价指标", 2)
            for 序号 in range(len(片段)):
                print("\n[片段 %d]\n%s" % (序号 + 1, 片段[序号]))
            continue
        try:
            结果 = 智能体.invoke({"messages": [HumanMessage(content=输入内容)]}, config=会话配置)
            输出内容 = 结果["messages"][-1].content
        except Exception as 异常信息:
            print("[调用失败] %s" % 异常信息)
            continue
        print("\n" + 输出内容)
        记录对话(输入内容, 输出内容)
    return 0


if __name__ == "__main__":
    sys.exit(主程序())
