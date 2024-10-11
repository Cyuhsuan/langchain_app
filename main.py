from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# 設定狀態
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    messages: Annotated[list, add_messages]

# 設定模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 設定記憶體儲存
memory = MemorySaver()

# 設定工具
tool = TavilySearchResults(max_results=2)
tools = [tool]
# model 使用工具
model = model.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}

# 告宣告一個 state 的狀態機
graph_builder = StateGraph(State)
# # 添加聊天機器人節點
graph_builder.add_node("chatbot", chatbot)

# # 添加工具節點
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# 添加邊
# 當使用工具時，返回聊天機器人節點
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# 任何時候使用工具時，返回聊天機器人節點
graph_builder.add_edge("tools", "chatbot")
# 設定入口節點
graph_builder.set_entry_point("chatbot")
# 編譯圖
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
    )

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}, config=config):
        for value in event.values():
            value["messages"][-1].pretty_print()

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break