from sentence_transformers import SentenceTransformer
import numpy as np
import os

class SimpleRetriever:
    def __init__(self, data_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.data_dir = data_dir
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self._load_data()

    def _load_data(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.documents.append(content)
                    print(f"Loaded document: {filename}")
                    print(f"Content: {content[:100]}...")  # Print first 100 characters