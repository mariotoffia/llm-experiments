from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredURLLoader

loader = UnstructuredFileLoader("app.py")

docs = loader.load()

print(docs)

