from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-3.5-turbo")

input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        # 創建 HumanMessage 對象
        message = HumanMessage(content=input_text)
        # 調用模型並獲取回答
        response = model.invoke([message])
        # 打印回答
        print(response.content)

    input_text = input('>>> ')