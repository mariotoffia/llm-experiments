from typing import List
import re
from pydantic import BaseModel
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class StructuredData(BaseModel):
    Question: str | None
    Answer: str
    Metadata: object
    File: str


def find_files(pattern) -> List[str]:
    """
    Find files matching a pattern
    :param pattern: Pattern to match e.g. 'data/*.txt'
    :return: List of file paths
    """
    import glob

    result: List[str] = []

    for file_path in glob.glob(pattern):
        result.append(file_path)

    return result


def scan_directory(pattern: str) -> List[List[StructuredData]]:  # NOSONAR
    """
    Retrieve structured data from a directory
    :param pattern: Pattern to match e.g. 'data/*.txt'
    :return: List of StructuredData objects
    """

    result: List[List[StructuredData]] = []

    for file_path in find_files(pattern):
        result.append(get_file_data(file_path))

    return result


def get_file_data(file_path: str, chunk_size=1024, chunk_overlap=100) -> List[StructuredData]:
    """
    Retrieve structured data from a file
    :param file_path: Path to the file
    :return: List of StructuredData objects
    """

    mode = "single"

    if re.match(r'.+_qa\..+', file_path):
        mode = 'elements'

    loader = UnstructuredFileLoader(
        file_path=file_path,
        strategy="hi-res",  # other option:"fast"
        mode=mode,  # single (default), elements, paged (for PDFs)
        post_processors=[clean_extra_whitespace],
    )

    docs = loader.load()

    if mode == "single":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        docs = text_splitter.split_documents(docs)
        result: List[StructuredData] = []
        for doc in docs:
            result.append(StructuredData(
                Question=None,
                Answer=doc.page_content.strip(),
                Metadata=doc.metadata,
                File=file_path,
            ))

        return result

    result: List[StructuredData] = []
    current_question = None
    current_answer = []

    for doc in docs:
        # Check if metadata and category exist, otherwise treat content as part of the answer
        category = doc.metadata.get("category") if doc.metadata else None
        content = doc.page_content.strip()

        if category is None:
            result.append(StructuredData(
                Question=None,
                Answer=content,
                Metadata=doc.metadata,
                File=file_path,
            ))
            continue

        if category == "Title":
            if current_question is not None:
                result.append(StructuredData(
                    Question=current_question.strip(),
                    Answer="\n".join(current_answer),
                    Metadata=doc.metadata,
                    File=file_path,
                ))
                current_answer = []

            current_question = content
        else:
            if category == "ListItem":
                current_answer.append(f"• {content}")
            else:
                current_answer.append(content)

    # Don't forget to add the last question-answer pair
    if current_question is not None:
        result.append(StructuredData(
            Question=current_question.strip(),
            Answer="\n".join(current_answer),
            Metadata=doc.metadata,
            File=file_path,
        ))

    return result


# loader = UnstructuredURLLoader(
#    urls=["https://www.crossbreed.se"], continue_on_failure=True, show_progress_bar=False, mode="elements"//
# )
