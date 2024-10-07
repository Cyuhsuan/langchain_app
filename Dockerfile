# FROM langchain/langchain

# # 安裝 OpenAI 和更新 LangChain
# RUN pip install --no-cache-dir openai langchain-openai langchain-community
# RUN pip install langserve



# FROM python:3.12

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install -r requirements.txt

# COPY . .

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM langchain/langchain

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]