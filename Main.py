import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from RAG import SimpleRAG

# Download the tokenizer model
# nltk.download('punkt', force=True)
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    :param pdf_path: Path to the PDF file
    :return: Extracted text as a string
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def pdf_to_sentences(pdf_path):
    """
    Convert PDF text into a list of sentences.
    
    :param pdf_path: Path to the PDF file
    :return: List of sentences
    """
    # Extract text from the PDF
    book_text = extract_text_from_pdf(pdf_path)
    
    # Split text into sentences
    sentences = sent_tokenize(book_text)
    return sentences

# Example usage
if __name__ == "__main__":
    # Path to your PDF book
    pdf_path = "Introduction to Artificial Intelligence.pdf"
    
    # Convert to sentences
    sample_docs = pdf_to_sentences(pdf_path)
    # sample_docsNew = []

    # for idx, sentence in enumerate(sample_docs[:]):  
    #     if len(sentence) > 100:
    #         sample_docsNew.append(sentence)
    
    # Print sentences
    for idx, sentence in enumerate(sample_docs[:10]):  # Print first 10 sentences
        print(f"{idx+1}: {sentence}")

   
    
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
