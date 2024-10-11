from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
model = ChatOpenAI(model="gpt-3.5-turbo")
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}


# 告宣告一個 state 的狀態機
graph_builder = StateGraph(State)
# 添加節點
graph_builder.add_node("chatbot", chatbot)
# 添加邊
# START 是狀態機的入口
# END 是狀態機的出口
# 從 START 到 chatbot
graph_builder.add_edge(START, "chatbot")
# 從 chatbot 到 END
graph_builder.add_edge("chatbot", END)

# 建立 狀態機的樹圖
# it will be a tree of nodes and edges (start -> chatbot -> end)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input(">>> ")
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