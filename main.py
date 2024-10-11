from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    messages: Annotated[list, add_messages]

# 設定模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 設定工具
tool = TavilySearchResults(max_results=2)
# tools = [{
#     "type": "function",
#     "function": {
#         "name": tool.name,
#         "description": tool.description,
#         "parameters": tool.args_schema.schema()
#     }
# }]
tools = [tool]
# tool.invoke("What's a 'node' in LangGraph?")
# model 使用工具
model = model.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}

# 告宣告一個 state 的狀態機
graph_builder = StateGraph(State)
# # 添加聊天機器人節點
# graph_builder.add_node("chatbot", chatbot)
# # 添加工具節點
# tool_node = ToolNode(tools=[tool])
# graph_builder.add_node("tools", tool_node)

# 添加邊
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.set_entry_point("chatbot")
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

# 建立 狀態機的樹圖
# 它將是一個節點和邊的樹圖 (start -> (chatbot (if)-> tools -> chatbot) -> end)
# graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

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