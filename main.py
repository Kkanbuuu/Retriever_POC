from fastapi import FastAPI
from retriever import SimpleRetriever

app = FastAPI()
retriever = SimpleRetriever(data_dir='data')

@app.get("/")
def root():
    return {"message": "Simple Retriever is running!"}

@app.get("/load_data")
def load_data():
    retriever._load_data()
    return {"message": "Data loaded successfully.", "num_documents": len(retriever.documents)}