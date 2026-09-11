import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

db_path = Path("chroma_db")
chroma_client = chromadb.PersistentClient(path=str(db_path))

resumes_path = Path("resumes")
resume_files = [f for f in resumes_path.iterdir() if f.is_file()]

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

collection = chroma_client.get_or_create_collection(name="hr_resumes")

if collection.count() == 0:
    for f in resume_files:
        if f.suffix.lower() == ".txt":
            with open(f, "r", encoding="utf-8") as file:
                content = file.read()
                
            chunks = chunk_text(content)
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{f.name}_chunk_{idx}"
                collection.add(
                    documents=[chunk],
                    ids=[chunk_id],
                    metadatas=[{"source": f.name}]
                )

print("\n... HR ASSISTANT IN FUNZIONE ...")

while True:
    user_query = input("Cosa ti serve?: ")
    
    if user_query.strip().lower() in ["esci", "exit", "quit"]:
        print("Arrivederci!")
        break
        
    if not user_query.strip():
        continue

    results = collection.query(
        query_texts=[user_query],
        n_results=2
    )
    
    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""
Sei un assistente HR esperto. Analizza i curriculum dati e struttura la risposta esattamente in questo modo:
- **Candidato Consigliato:** (Nome del profilo, usa il come del file se non trovi il nome del profilo)
- **Punti di Forza:** (Breve elenco puntato)
- **Motivazione:** (Perché è adatto)

Contesto:
{context}

Domanda: {user_query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente HR preciso e professionale."},
            {"role": "user", "content": prompt}
        ]
    )
    
    print("\n--- MATCH TROVATO ---")
    print(response.choices[0].message.content)
    print("\nFonti utilizzate:", set(s["source"] for s in sources))
    print("-" * 50 + "\n")
    
    #poetry run python main.py  --> Avvia l'assistente HR interattivo