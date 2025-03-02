from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")

vectorstore_persist_path = "/Users/govin/Projects/sprintdotnext/legal-advisor/vectorstore"

# Initialize the embedding model (must match the one used for indexing)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Load the persisted Chroma vector store
vectorstore = Chroma(
    persist_directory=vectorstore_persist_path, 
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "Qwen/Qwen2.5-3B-Instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
#)

#model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps", max_memory={0: "6GB"})
tokenizer = AutoTokenizer.from_pretrained(model_name)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=200,
    eos_token_id=terminators,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions using provided context.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
If there are any references from the retrieved documents, share that also part of your answer.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

from langchain.memory import ConversationBufferMemory

# Initialize memory to store conversation history
memory = ConversationBufferMemory(memory_key="history")

# Create the RetrievalQA chain with memory
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    memory=memory,
    verbose=True
)

message = input("User: ")
while (message != 'done'):
    response = qa_chain.invoke(message)["result"]
    print(f"Advisor: {response}")
    message  = input("User: ")
