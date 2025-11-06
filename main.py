from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Simple Retriever is running!"}