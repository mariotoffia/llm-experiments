from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.document_loaders import UnstructuredURLLoader

# https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed

loader = UnstructuredFileLoader(
    file_path="data/training/smarter-heating.md",
    strategy="hi-res", # other option:"fast"
    mode="elements", # single (default), elements, paged (for PDFs)
    post_processors=[clean_extra_whitespace],
)

# loader = UnstructuredURLLoader(
#    urls=["https://www.crossbreed.se"], continue_on_failure=True, show_progress_bar=False, mode="elements"//
# )

docs = loader.load()

result = []
current_question = None
current_answer = []

for doc in docs:
    # Check if metadata and category exist, otherwise treat content as part of the answer
    category = doc.metadata.get("category") if doc.metadata else None
    content = doc.page_content.strip()

    if category is None:
        result.append({"Answer": doc.page_content.strip(), "Metadata": doc.metadata})
        continue

    if category == "Title":
        # If there is an ongoing question, save it before starting a new one
        if current_question is not None:
            result.append({"Question": current_question, "Answer": "\n".join(current_answer), "Metadata": doc.metadata})
            current_answer = []

        current_question = content        
    else:
        if category == "ListItem":
            current_answer.append(f"• {content}")  # Format list items with a bullet
        else:
            # Treat as part of the answer if there's no category or it's not a ListItem
            current_answer.append(content)

# Don't forget to add the last question-answer pair
if current_question is not None:
    result.append({"Question": current_question, "Answer": "\n".join(current_answer), "Metadata": doc.metadata})

for res in result:
    if res.get("Question") is not None:
      print(res.get("Question"))
      print("-" * 50)
    print(res.get("Answer"))
    print("\n")
