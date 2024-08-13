from langchain.llms import load_llm
from langchain.chains import RetrievalAugmentation
from langchain.configs import load_config

# 加载配置
config = load_config("config.yaml")

# 初始化RAG模型
rag_model = RetrievalAugmentation.from_config(config["models"]["my_rag_model"])

# 示例问题
question = "What is the capital of France?"

# 使用RAG模型生成回答
answer = rag_model(question)

print(f"Question: {question}")
print(f"Answer: {answer}")

