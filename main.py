from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
model = ChatOpenAI(model="gpt-3.5-turbo")
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
# Define the function that calls the model
# def call_model(state: MessagesState):
#     response = model.invoke(state["messages"])
#     return {"messages": response}

async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

async def main():
    input_text = input('>>> ')
    while input_text.lower() != 'bye':
        if input_text:
            input_messages = [HumanMessage(input_text)]
            output = await app.ainvoke({"messages": input_messages}, config)
            output["messages"][-1].pretty_print()  # output contains all messages in state

        input_text = input('>>> ')

import asyncio

if __name__ == "__main__":
    asyncio.run(main())