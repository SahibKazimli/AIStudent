import numpy as np
import requests
import os
from typing import List, Dict
import dotenv

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = "https://api.openai.com/v1"

class Document:
    """Simple document class to store text and its embedding."""
    def __init__(self, text: str, doc_id: str = None):
        self.text = text
        self.doc_id = doc_id
        self.embedding = None
    
    def __str__(self):
        return f"Document(id={self.doc_id}, text={self.text[:50]}...)"

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between vectors using NumPy."""
    # Handle single vector to matrix comparison
    if len(vec1.shape) == 1:
        vec1 = vec1.reshape(1, -1)
    
    # Compute dot product
    dot_product = np.dot(vec1, vec2.T)
    
    # Compute norms
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    
    # Avoid division by zero
    similarity = dot_product / (np.outer(norm1, norm2) + 1e-8)
    
    return similarity

class RAGSystem:
    """Simplified RAG system using OpenRouter and NumPy."""
    
    def __init__(self, model_name: str = "google/gemini-2.0-pro-exp-02-05:free"):
        self.documents = []
        self.model_name = model_name
        self.embeddings = None
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        }
    
    def add_documents(self, texts: List[str], doc_ids: List[str] = None):
        """Add documents to the RAG system."""
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(texts))]
        
        for text, doc_id in zip(texts, doc_ids):
            self.documents.append(Document(text, doc_id))

        self._compute_embeddings()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using OpenRouter."""
        payload = {
            "model": "text-embedding-3-small",
            "input": text
        }

        headers = self.headers
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

        response = requests.post(
            f"{OPENAI_BASE_URL}/embeddings",
            headers=headers,
            json=payload
        )
        response_data = response.json()
        return np.array(response_data["data"][0]["embedding"])


    def _compute_embeddings(self):
        """Compute embeddings for all documents."""
        all_embeddings = [self._get_embedding(doc.text) for doc in self.documents]
        
        # Store embeddings in documents and as numpy array
        for i, doc in enumerate(self.documents):
            doc.embedding = all_embeddings[i]
        
        self.embeddings = np.array(all_embeddings)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve the top_k most relevant documents for a query."""
        if not self.documents:
            return []
        
        # Compute query embedding
        query_embedding = self._get_embedding(query)
        
        # Compute similarity scores using NumPy-based cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get indices of top_k documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
    
    def generate(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generate a response based on the query and retrieved documents."""
        context = "\n\n".join([doc.text for doc in retrieved_docs])
        
        prompt = f"Answer based on this context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ]
            }

            headers = self.headers
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
            
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I couldn't generate a response due to an error."
    
    def query(self, query: str, top_k: int = 3) -> Dict:
        """End-to-end RAG pipeline: retrieve documents and generate answer."""
        retrieved_docs = self.retrieve(query, top_k)
        answer = self.generate(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {"id": doc.doc_id, "text": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text}
                for doc in retrieved_docs
            ]
        }


# Example usage
if __name__ == "__main__":

    # More realistic knowledge base with medical information
    documents = [
        "Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels over a prolonged period. Symptoms include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many health complications.",
        "Type 1 diabetes results from the pancreas's inability to produce enough insulin due to loss of beta cells. This form was previously referred to as 'insulin-dependent diabetes mellitus' or 'juvenile diabetes'. The loss of beta cells is caused by an autoimmune response.",
        "Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly. As the disease progresses, a lack of insulin may also develop. The most common cause is a combination of excessive body weight and insufficient exercise.",
        "Gestational diabetes is the third main form and occurs when pregnant women without a previous history of diabetes develop high blood sugar levels. It may lead to increased risk of complications during pregnancy and delivery.",
        "Treatment of diabetes focuses on maintaining blood sugar levels as close to normal as possible. Type 1 diabetes requires the person to inject insulin. Type 2 diabetes may be treated with medications with or without insulin.",
        "Lifestyle factors are important in the management of diabetes. Regular physical activity, maintaining a healthy body weight, and following a healthful diet are essential components of treatment plans."
    ]

    rag = RAGSystem(model_name="google/gemini-2.0-pro-exp-02-05:free")
    rag.add_documents(documents)

    result = rag.query("What are the key differences between Type 1 and Type 2 diabetes, and how are they treated?")

    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print("\nRetrieved Documents:")
    for doc in result['retrieved_documents']:
        print(f"- {doc['text']}")
