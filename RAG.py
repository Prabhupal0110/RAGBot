import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

model_checkpoint = "consciousAI/question-answering-roberta-base-s"

class SimpleRAG:
    def __init__(self):
        # Initialize a lightweight sentence transformer for embeddings
        self.retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize a small language model for generation
        self.generator_name = "gpt2"  # Very lightweight model
        self.generator = AutoModelForCausalLM.from_pretrained(self.generator_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.generator_name)
        self.question_answerer = pipeline("question-answering", model=model_checkpoint)
        
        # Initialize empty knowledge base
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs):
        """Add documents to the knowledge base"""
        self.documents.extend(docs)
        # Create embeddings for new documents
        new_embeddings = self.retriever.encode(docs)
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def retrieve(self, query, top_k=10):
        """Retrieve most relevant documents for a query"""
        # Get query embedding
        query_embedding = self.retriever.encode([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k document indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]

    def generate_response(self, query):
        """Generate response based on retrieved documents"""
        # Get relevant documents
        relevant_docs = self.retrieve(query)
        
        # Create prompt
        context = " ".join(relevant_docs)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        print("Promp given to the Generator: ", prompt)
        
        response = self.question_answerer(question=query, context=context)
        
        # # Generate response
        # inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # with torch.no_grad():
        #     outputs = self.generator.generate(
        #         inputs.input_ids,
        #         max_new_tokens=100,
        #         num_return_sequences=1,
        #         temperature=0.7,
        #         do_sample=True
        #     )
        
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return response.replace(prompt, "").strip()
        return response

# Example usage
def main():
    # Create sample knowledge base
    sample_docs = [
        "The Python programming language was created by Guido van Rossum and released in 1991.",
        "Python is known for its simple syntax and readability, following the 'batteries included' philosophy.",
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        "RAG (Retrieval Augmented Generation) combines document retrieval with text generation."
    ]
    
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Add documents to knowledge base
    print("Adding documents to knowledge base...")
    rag.add_documents(sample_docs)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        try:
            print("\nRetrieving relevant documents...")
            relevant_docs = rag.retrieve(query)
            print("Retrieved documents:")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"{i}. {doc}")
            
            print("\nGenerating response...")
            response = rag.generate_response(query)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()