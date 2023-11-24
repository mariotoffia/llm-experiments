import os
from embeddingsdb import EmbeddingsDb
from scanner import scan_directory, is_binary_file

user_file_path = os.path.join(os.path.dirname(__file__), "data", "user")

def index_files(pattern: str, embeddings: EmbeddingsDb):
  training = scan_directory(pattern)

  for text in training:
    if len(text) == 0:
      continue

    file = text[0].metadata["file"]
    stored = embeddings.store_structured_data(docs=text, id=file)

    if stored:
      print(f'\n*** STORED: {file} ***')
    else:
      print(f'\n*** SKIPPED: {file} ***')

def system_index(embeddings: EmbeddingsDb):
  """
  system_index will index training and knowledge (if needed) and
  delete data/embeddings if you want to reindex
  
  :param embeddings: The embeddings database to use.
  :return: None
  """
  index_files(pattern="data/training/*.md", embeddings=embeddings)
  index_files(pattern="data/knowledge/*.md", embeddings=embeddings)

def user_index(embeddings: EmbeddingsDb):
  """
  user_index will index the user's uploaded files
  
  :param embeddings: The embeddings database to use.
  :return: None
  """
  index_files(pattern="data/user/*.*", embeddings=embeddings)

def add_file_to_user_index(
      file_name: str, 
      embeddings_db: EmbeddingsDb,
      content: bytes | str | None
      ):
      """
      add_file_to_user_index will add a file to the user index
      and reindex the user index

      :param file_name: The name of the file to add.
      :param embeddings_db: The embeddings database to use.
      :param content: The content of the file to add.
      """

      if is_binary_file(file_name):
          with open(os.path.join(user_file_path, file_name), "wb") as f:
              f.write(content)
      else:
          with open(os.path.join(user_file_path, file_name), "w") as f:
              f.write(content.decode("utf-8"))

      user_index(embeddings=embeddings_db)
