import os
from pathlib import Path
import uuid
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Model and client
#model_name = 'all-MiniLM-L6-v2'
#model_name = 'all-mpnet-base-v2'
model_name = 'BAAI/bge-base-en-v1.5'
model = SentenceTransformer(model_name)
db_path = "./fineweb_index"
chroma_client = chromadb.PersistentClient(path=db_path)

# Read the contents of the text to evaluate
def get_file():
    filepath = input("Enter the name of a file to evaluate:")
    while True:
        try:
            with open(filepath, "r") as file:
                evaluation_text = file.read()
                if evaluation_text: 
                    break
        except IOError:
            filepath = input("File not found. Retype the filename:")
            filename = Path(filepath).stem
    return filepath, evaluation_text

# Store vectorized texts in space with cosine distances
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
# Create a new, larger collection
large_chunk_collection = chroma_client.get_or_create_collection(
    name="fineweb_large_chunks", 
    embedding_function=hf_ef,
    configuration={"hnsw": {"space": "cosine"}} 
)

# Use a sliding frame to put words in chunks to 
# accomodate the model's context window
def chunk_long_text(text: str, words_per_chunk: int = 400, overlap: int = 80):
    words = text.split()
    chunks = []
    
    if len(words) <= words_per_chunk:
        return [text]

    # In each loop, start at the beginning of the overlap
    step = words_per_chunk - overlap
    for i in range(0, len(words), step):
        # Measure out a chunk from the beginning of the overlap
        # and add it to the chunks list
        chunk_words = words[i:i + words_per_chunk]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks

# The stream_fineweb_to_chroma is Gemini's work
def stream_fineweb_to_chroma(max_documents_to_process: int = 50):
    print(f"Initializing stream from Hugging Face (HuggingFaceFW/fineweb-edu)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        streaming=True
    )
    doc_count = 0

    for row in dataset:
        if doc_count >= max_documents_to_process:
            break
        raw_text = row["text"]
        # If there is no ID for the document, generate one
        doc_id = row.get("id", str(uuid.uuid4()))
        text_chunks = chunk_long_text(raw_text, words_per_chunk=400, overlap=80)
        chunk_docs = []
        chunk_ids = []
        chunk_metadatas = []

        for idx, chunk in enumerate(text_chunks):
            chunk_docs.append(chunk)
            chunk_ids.append(f"{doc_id}_chunk_{idx}")
            chunk_metadatas.append({
                "source_doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": len(text_chunks)
            })

        if chunk_docs:
            large_chunk_collection.add(
                documents=chunk_docs,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
        doc_count += 1
        if doc_count % 10 == 0:
            print(f"Successfully processed and stored {doc_count} web documents...")

    print(f"\nStream complete. Total stored chunks in index: {large_chunk_collection.count()}")

stream_fineweb_to_chroma(max_documents_to_process=1000)

def evaluate_ideation_novelty(llm_output_text: str):
    # Find closest match in FineWeb collection
    # Calculate cosine distance (and thus dissimilarity)

    llm_output_chunks = chunk_long_text(llm_output_text, words_per_chunk=400, overlap=80)

    # Query Chroma for the single closest matching document
    results = large_chunk_collection.query(
        query_texts=llm_output_chunks,
        n_results=1
    )
    
    # Chroma returns cosine distance 
    # See https://docs.trychroma.com/docs/collections/configure
    all_distances = [dist_list[0] for dist_list in results['distances'] if dist_list]
    all_documents = [doc_list[0] for doc_list in results['documents'] if doc_list]

    if not all_distances:
        return {"novelty_score": 1.0, "max_similarity_score": 0.0, "closest_match": "", "analyzed_text": llm_output_text}


    # Find the index of the MINIMUM distance (most unoriginal chunk)
    most_unoriginal_idx = all_distances.index(min(all_distances))

    closest_distance = all_distances[most_unoriginal_idx]
    closest_document = all_documents[most_unoriginal_idx]
    
    highest_similarity = 1.0 - closest_distance
    
    # Closest distance gives us novelty
    novelty_score = closest_distance
    
    # Bound the results cleanly between 0.0 and 1.0 to handle floating point anomalies
    novelty_score = max(0.0, min(1.0, novelty_score))
    
    return {
        "analyzed_text": llm_output_text,
        "closest_match": closest_document,
        "max_similarity_score": round(highest_similarity, 4),
        "novelty_score": round(novelty_score, 4)
    }

# Finally, run the evaluation code on the evaluation text
while True: 
    filepath, evaluation_text = get_file()
    result = evaluate_ideation_novelty(evaluation_text)

    # Display Results
    print(f"--- RESULTS ---")

    result_string = f"Analyzed text: {result['analyzed_text']}\n" + f"Closest match: {result['closest_match']}\n" + f"Similarity: {result['max_similarity_score']} | Novelty score: {result['novelty_score']}\n"
    print(result_string)

    text_data = result_string

    filename = Path(filepath).stem

    with open("./output/" + filename + "_evaluation.txt", "w", encoding="utf-8") as file:
        file.write(text_data)

    print("Output writtent to file.\n")