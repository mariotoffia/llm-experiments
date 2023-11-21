from embeddingsdb import EmbeddingsDb
from scanner import scan_directory

def index_files(pattern: str, embeddings: EmbeddingsDb):
  training = scan_directory(pattern)

  for text in training:
    if len(text) == 0:
      continue
    
    stored = embeddings.store_structured_data(data=text, id=text[0].File)

    if stored:
      print(f'\n*** STORED: {text[0].File} ***')
    else:
      print(f'\n*** SKIPPED: {text[0].File} ***')

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