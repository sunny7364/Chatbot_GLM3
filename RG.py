##  pip install -q torch transformers accelerate bitsandbytes transformers sentence-transformers faiss-gpu
##  pip install -q langchain
##  pip install modelscope
##  pip install faiss-gpu
##  pip install bitsandbytes


import torch
from getpass import getpass
from modelscope import snapshot_download
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
#ACCESS_TOKEN = getpass("YOUR_GITHUB_PERSONAL_TOKEN")
#loader = GitHubIssuesLoader(repo="huggingface/peft", access_token=ACCESS_TOKEN, include_prs=False, state="all")
loader = CSVLoader(file_path="drug_name_1000.csv")
docs = loader.load()
chunked_docs = splitter.split_documents(docs)


model_dir = snapshot_download("AI-ModelScope/bge-base-zh-v1.5", revision="master")
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True} # set True to compute cosine similarity
db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name=model_dir))
#db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})


model_name = "ChatGLM3"
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
#model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)


prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
llm_chain = prompt | llm | StrOutputParser()


retriever = db.as_retriever()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain


question = "How do you combine multiple adapters?"
llm_chain.invoke({"context": "", "question": question})
rag_chain.invoke(question)