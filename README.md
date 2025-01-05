# RAGbot Project - README

## Project Overview
RAGbot is a lightweight Retrieval-Augmented Generation (RAG) system that combines document retrieval with language generation to answer queries. It uses a small sentence transformer model for efficient document embedding and retrieval, alongside a lightweight language model (RoBERTa) for response generation. This project was created over a weekend to learn the RAG concept for fun. It is ideal for small-scale, efficient RAG implementations.

## Features
- **Document Retrieval**: Uses a lightweight sentence transformer (`all-MiniLM-L6-v2`) to encode and retrieve relevant documents.
- **Language Generation**: Utilizes RoBERTa to generate responses based on the retrieved documents.
- **Question Answering Pipeline**: Enhances responses using a pre-trained question-answering model.
- **Knowledge Base**: Dynamically expands by adding documents and generating embeddings.

## Requirements
- Python 3.7+
- Transformers (`transformers`)
- Sentence Transformers (`sentence-transformers`)
- NumPy (`numpy`)
- scikit-learn (`scikit-learn`)

Install dependencies with:
```bash
pip install transformers sentence-transformers numpy scikit-learn
```

## How It Works
### 1. Initialization
```python
rag = SimpleRAG()
```
- Initializes the retriever (sentence-transformer) and generator (RoBERTa).
- Prepares an empty knowledge base for document storage.

### 2. Adding Documents
```python
documents = ["AI is transforming the world.", "Machine learning powers many AI applications."]
rag.add_documents(documents)
```
- Adds documents to the knowledge base and generates embeddings for efficient retrieval.

### 3. Retrieving Documents
```python
retrieved_docs = rag.retrieve("How is AI changing the world?")
print(retrieved_docs)
```
- Retrieves the most relevant documents based on cosine similarity of embeddings.

### 4. Generating Responses
```python
response = rag.generate_response("How is AI changing the world?")
print(response)
```
- Retrieves relevant documents and generates an answer using the RoBERTa model.

## Class Breakdown
### `SimpleRAG` Class
- **`__init__`**: Initializes the retriever, generator, and question-answering pipeline. Prepares an empty knowledge base.
- **`add_documents`**: Adds documents and their embeddings to the knowledge base.
- **`retrieve`**: Retrieves the top-k documents that match the query using cosine similarity.
- **`generate_response`**: Generates a response by retrieving relevant documents and feeding them into the language model.

## Example Workflow
```python
rag = SimpleRAG()
rag.add_documents(["AI improves healthcare.", "AI helps in self-driving cars."])
response = rag.generate_response("How does AI help healthcare?")
print(response)
```

## Notes
- This project uses RoBERTa for generation, making it lightweight but less sophisticated than larger models.
- Ideal for small datasets and environments with limited resources.
- For better results, consider fine-tuning models or using larger pre-trained models.

## Future Improvements
- Integrate larger models for more accurate generation.
- Implement fine-tuning of the retriever and generator.
- Add multi-turn conversation support.

## License
This project is open-source and can be modified and redistributed under the MIT License.

