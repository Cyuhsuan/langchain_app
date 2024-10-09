from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

model = ChatOpenAI(model="gpt-3.5-turbo")


config = {"configurable": {"thread_id": "abc123"}}

# 新增爬取網站和RAG功能
def scrape_and_rag(url):
    # 爬取網站
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # 使用 RecursiveCharacterTextSplitter 將文檔分割成較小的塊
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    
    # 初始化 OpenAI 嵌入模型
    embeddings = OpenAIEmbeddings()
    
    # 使用 FAISS 創建向量數據庫並加載分割後的文檔
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    
    # 創建用於生成搜索查詢的提示模板
    prompt_search_query = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    # 創建帶有歷史上下文的檢索器
    retriever_chain = create_history_aware_retriever(model, retriever, prompt_search_query)

    # 創建用於獲取答案的提示模板
    prompt_get_answer = ChatPromptTemplate.from_messages([
        ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
        MessagesPlaceholder(variable_name="chat_history"),
        ('user', '{input}'),
    ])
    
    # 創建文檔鏈以生成答案
    document_chain = create_stuff_documents_chain(model, prompt_get_answer)

    # 結合檢索器和文檔鏈，創建檢索鏈
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    return retrieval_chain

# 修改主循環
input_text = input('>>> ')
chat_history = []
while input_text.lower() != 'bye':
    if input_text.startswith("scrape:"):
        # 處理爬取和RAG請求
        _, url = input_text.split("scrape:", 1)
        retrieval_chain = scrape_and_rag(url.strip())
        print("網頁內容已爬取並處理完成。請輸入您的問題:")
    elif input_text:
        # 原有的對話功能
        if 'retrieval_chain' in locals():
            result = retrieval_chain.invoke({"input": input_text, "chat_history": chat_history})
            print(result['answer'])
            chat_history.extend([HumanMessage(input_text), result['answer']])
        else:
            print("請先使用 'scrape:' 命令爬取網頁內容。")

    input_text = input('>>> ')

    # scrape:https://www.explainthis.io/zh-hant/swe/what-is-closure
    # 閉包第一項應用?