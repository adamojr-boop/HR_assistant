import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Caricamento configurazioni
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Inizializzazione ChromaDB e Text Splitter
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="hr_resumes")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 3. Ingestione dei CV se la collection è vuota
if collection.count() == 0:
    resumes_dir = "./resumes"
    files = [f for f in os.listdir(resumes_dir) if f.endswith(".txt")]
    
    documents = []
    metadatas = []
    ids = []
    
    for filename in files:
        file_path = os.path.join(resumes_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename})
            ids.append(f"{filename}_chunk_{i}")
            
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Database popolato con {collection.count()} frammenti dai CV.")
else:
    print(f"Database già popolato. Elementi presenti: {collection.count()}")

# 4. Funzione di interrogazione RAG
def ask_hr_assistant(query: string, n_results: int = 2):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""
Sei un assistente HR esperto. Rispondi alla domanda dell'utente basandoti unicamente sul contesto estratto dai curriculum forniti di seguito.

Contesto:
{context}

Domanda: {query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente HR preciso e professionale."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content, sources

if __name__ == "__main__":
    test_query = "Chi ha competenze di Python e lavora nel settore tech?"
    answer, sources = ask_hr_assistant(test_query)
    print("\n--- RISPOSTA FINALE ---")
    print(answer)
    print("\nFonti utilizzate:", set(s["source"] for s in sources))