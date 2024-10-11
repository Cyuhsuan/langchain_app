from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# 設定狀態
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    messages: Annotated[list, add_messages]
    ask_human: bool

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


# 設定模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 設定記憶體儲存
memory = MemorySaver()

# 設定工具
tool = TavilySearchResults(max_results=2)
tools = [tool]
# model 使用工具
llm_with_tools = model.bind_tools(tools + [RequestAssistance])

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}

# 告宣告一個 state 的狀態機
graph_builder = StateGraph(State)
# # 添加聊天機器人節點
graph_builder.add_node("chatbot", chatbot)

# # 添加工具節點
graph_builder.add_node("tools", ToolNode(tools=[tool]))

def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }

# 添加人類節點
graph_builder.add_node("human", human_node)

def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


# 添加邊
graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"},
)
# 任何時候使用工具時，返回聊天機器人節點
graph_builder.add_edge("tools", "chatbot")
# 當人類節點返回時，返回聊天機器人節點
graph_builder.add_edge("human", "chatbot")
# 設定入口節點
graph_builder.set_entry_point("chatbot")

# 編譯圖
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"]
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