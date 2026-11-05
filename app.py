import os
import glob
import time
import threading

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "gradio_temp")

import gradio as gr
from langchain_core.messages import HumanMessage

import main as 核心

智能体锁 = threading.Lock()
会话状态 = {"线程编号": 核心.会话编号, "轮次": 0}

核心.读取知识库()
智能体实例 = 核心.构建智能体()

页面样式 = """
.result-box { max-height: 560px; overflow-y: auto; padding-right: 8px; }
.hit-box { max-height: 260px; overflow-y: auto; }
"""


def 当前配置():
    return {"configurable": {"thread_id": 会话状态["线程编号"]}}


def 状态文本(结尾说明="就绪"):
    return "**模型** %s　｜　**知识库** %d 段　｜　**本次会话** 第 %d 次评估　｜　**状态** %s" % (
        核心.模型名称,
        len(核心.知识切片),
        会话状态["轮次"],
        结尾说明,
    )


def 拼命中(命中清单):
    if not 命中清单:
        return "本次未检索到与需求相关的知识片段。"
    段落 = []
    for 序号, 片段 in enumerate(命中清单, start=1):
        段落.append("**片段 %d**\n\n> %s" % (序号, 片段.replace("\n", "\n> ")))
    return "\n\n".join(段落)


def 评估流程(需求文本):
    if not str(需求文本).strip():
        yield "请先在左侧输入一段软件需求描述。", "尚未检索", 状态文本("等待输入")
        return

    需求文本 = str(需求文本).strip()
    命中清单 = 核心.检索知识(需求文本, 3)
    命中文本 = 拼命中(命中清单)
    累计 = []
    开始时刻 = time.time()

    yield "正在从知识库检索并调用大模型评估，请稍候……", 命中文本, 状态文本("评估中")

    try:
        with 智能体锁:
            for 片段, _元信息 in 智能体实例.stream(
                {"messages": [HumanMessage(content=需求文本)]},
                config=当前配置(),
                stream_mode="messages",
            ):
                正文 = getattr(片段, "content", "")
                if 正文:
                    累计.append(正文)
                    yield "".join(累计), 命中文本, 状态文本("输出中")
    except Exception as 异常信息:
        错误文本 = "调用失败：%s: %s" % (type(异常信息).__name__, 异常信息)
        yield 错误文本, 命中文本, 状态文本("出错")
        return

    全文 = "".join(累计).strip()
    核心.记录对话(需求文本, 全文)
    会话状态["轮次"] += 1
    耗时 = round(time.time() - 开始时刻, 1)
    yield 全文, 命中文本, 状态文本("完成，耗时 %s 秒" % 耗时)


def 清空会话():
    会话状态["线程编号"] = "%s_%d" % (核心.会话编号, int(time.time()))
    会话状态["轮次"] = 0
    return "已清空，左侧输入新需求开始新一轮评估。", "尚未检索", 状态文本("已重置")


def 导出记录():
    文件清单 = sorted(glob.glob(os.path.join(核心.日志目录, "对话记录_*.txt")))
    if not 文件清单:
        return None
    return 文件清单[-1]


with gr.Blocks(title="需求质量评估智能体", css=页面样式, theme=gr.themes.Soft()) as 界面:
    gr.Markdown(
        "# 需求质量评估智能体\n"
        "LangChain + LangGraph + 本地知识库检索（RAG）　·　实验4 方案B"
    )
    with gr.Row():
        with gr.Column(scale=5, min_width=380):
            需求框 = gr.Textbox(
                label="输入软件需求",
                lines=10,
                placeholder="例如：系统需要做一个很好用的图书借阅功能。",
            )
            with gr.Row():
                示例甲 = gr.Button("示例一 · 模糊需求", size="sm")
                示例乙 = gr.Button("示例二 · 概括需求", size="sm")
                示例丙 = gr.Button("示例三 · 较完整需求", size="sm")
            with gr.Row():
                评估钮 = gr.Button("开始评估", variant="primary")
                清空钮 = gr.Button("清空会话")
                导出钮 = gr.Button("导出记录")
            状态栏 = gr.Markdown(状态文本())
            记录文件 = gr.File(label="对话记录下载", visible=False)
        with gr.Column(scale=7, min_width=420):
            结果框 = gr.Markdown(
                "左侧输入一条软件需求，点击「开始评估」。\n\n"
                "系统会从完整性、无二义性、可测试性、一致性、可行性五个维度打分，"
                "列出问题清单并给出可直接落地的改写建议。",
                elem_classes="result-box",
            )
            with gr.Accordion("本次检索命中的知识片段（RAG 检索增强）", open=False):
                命中框 = gr.Markdown("尚未检索", elem_classes="hit-box")

    示例甲.click(lambda: "系统需要做一个很好用的图书借阅功能。", outputs=需求框, api_name="示例一")
    示例乙.click(lambda: "管理员可以对用户账号进行管理。", outputs=需求框, api_name="示例二")
    示例丙.click(
        lambda: "用户在登录页面输入账号和密码，点击登录后进入系统，登录失败要给出提示。",
        outputs=需求框,
        api_name="示例三",
    )
    评估钮.click(评估流程, inputs=需求框, outputs=[结果框, 命中框, 状态栏], api_name="评估")
    清空钮.click(清空会话, outputs=[结果框, 命中框, 状态栏], api_name="清空会话")
    导出钮.click(导出记录, outputs=记录文件, api_name="导出记录").then(
        lambda: gr.update(visible=True), outputs=记录文件
    )


if __name__ == "__main__":
    print("=" * 60)
    print("网页端地址：http://127.0.0.1:7860")
    print("=" * 60)
    界面.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_api=False)
