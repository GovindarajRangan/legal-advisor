import os
import json
from unstructured.staging.base import elements_from_json
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import HuggingFaceEmbeddings

vectorstore_persist_path = "/Users/govin/Projects/sprintdotnext/legal-advisor/vectorstore"
unstructured_ingest_path = "/Users/govin/Projects/sprintdotnext/legal-advisor/local-ingest-output"

elements = []
for filename in os.listdir(unstructured_ingest_path):
    filepath = os.path.join(unstructured_ingest_path, filename)
    elements.extend(elements_from_json(filepath))

documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    #metadata["source"] = metadata["filename"]
    metadata["source"] = metadata.get("filename", metadata.get("attached_to_filename", "NoFilename"))
    del metadata["languages"]
    documents.append(Document(page_content=element.text, metadata=metadata))

# ChromaDB doesn't support complex metadata, e.g. lists, so we drop it here.
# If you're using a different vector store, you may not need to do this
docs = filter_complex_metadata(documents)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma.from_documents(
    documents=docs,  # Use the filtered docs
    embedding=embeddings,
    persist_directory=vectorstore_persist_path  # Ensure persistence
)
vectorstore.persist()  # This is optional in Chroma >=0.4.x, but safe to include

