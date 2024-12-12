import re
import pickle
import numpy as np
import pandas as pd
import argparse
import time
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
import math
import torch
from codecarbon import EmissionsTracker
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter
import faiss
from torch.cuda.amp import autocast
from rerankers import Reranker, Document

### UTILS ###

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

### DATA CLEANUP ###

def get_results_discussion(text):
    # extract text between results and references. dotall to account for multiple lines
    return "## RESULTS" + re.search(r"\s*R\s*E\s*S\s*U\s*L\s*T\s*S(.*?)\s*R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S", text,re.DOTALL| re.IGNORECASE).group(1)
    
def clean_text(text):
    # Extract newlines artifacts from PDF extraction
    return text.replace("\n\n", " ")

def get_metadata(text):
    # extract text before abstract, should contain title and authors.
    return re.split(r"\s*A\s*B\s*S\s*T\s*R\s*A\s*C\s*T", text, flags=re.IGNORECASE)[0]


def preprocess_doc(document):
    # Clean up text to include results and discussion only 
    document.metadata['article_info'] = get_metadata(document.text)
    document.excluded_embed_metadata_keys.append("article_info")
    document.excluded_llm_metadata_keys.append("article_info")
    try:
        document.text = get_results_discussion(document.text)
    except: # remove at least references from text, but keep the rest
        document.text = re.split(r"R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S", document.text, flags=re.IGNORECASE)[0]
    document.text = clean_text(document.text)


### CHUNKING ###

def chunk_text_with_sentences(text, max_chunk_size=300):
    # Chunking function that respects sentence boundaries
    # Initialize the SentenceSplitter
    text_parser = SentenceSplitter(
        chunk_size=max_chunk_size,
    )
    sentences = text_parser.split_text(text)  # Split text into sentences
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_docs(documents):
    # Create chunks with metadata
    chunks = []
    for doc in documents:
        chunked_text = chunk_text_with_sentences(doc.text)
        for idx, chunk in enumerate(chunked_text):
            chunks.append({"text": chunk, "metadata": {"source": doc.metadata, "chunk_index": idx}})
    return chunks

### EMBEDDING ###

def embeddings(model_name, chunks, batch_size=8):
    embedding_model = SentenceTransformer(model_name, device="cpu")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=False)
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

### INDEXING ###
def indexing(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    # Add embeddings to the respective indices
    index.add(np.array(embeddings))
    return index

### RETRIEVAL ###

def retrieve_top_bottom(query, embedding_model_name, index, chunks, top_k=25, bottom_k=10):
    # Encode and normalize the query
    embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Search FAISS index
    similarities, indices = index.search(query_embedding, len(chunks))
    
    # Sort indices by decreasing similarity
    sorted_indices = np.argsort(-similarities[0])  # Negate to sort by descending order
    
    # Get top K and bottom K indices
    top_k_indices = sorted_indices[:top_k]
    bottom_k_indices = sorted_indices[-bottom_k:]
    
    # Retrieve chunks with metadata, include similarity score and create Document objects
    top_k_chunks = [Document(text=chunks[i]["text"], metadata={"meta":chunks[i]["metadata"], "similarity": similarities[0][i], "query": query})for i in top_k_indices]
    bottom_k_chunks = [Document(text=chunks[i]["text"], metadata={"meta":chunks[i]["metadata"], "similarity": similarities[0][i], "query": query})for i in bottom_k_indices]
    
    return top_k_chunks, bottom_k_chunks

### RERANKING ###

def reranking(query, model_type, passage_docs):
    ranker = Reranker(model_type, device="cpu", verbose=0)
    with torch.cuda.amp.autocast():
        results = ranker.rank(query=query, docs=passage_docs)
    return results


### SAVING AND TRACKING ###

def save_passage_scores(output_dir, query, model_name, top_k_chunks, bottom_k_chunks):
    """
    Save top and bottom passages with similarity scores for analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name}_passages.csv")

    # Check if file exists
    write_header = not os.path.exists(file_path)
    
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new
        if write_header:
            writer.writerow(["query", "type", "text", "similarity", "metadata"])
        
        # Write top K passages
        for chunk in top_k_chunks:
            writer.writerow([
                query, "top", chunk.text, chunk.metadata["similarity"], chunk.metadata["meta"]
            ])
        
        # Write bottom K passages
        for chunk in bottom_k_chunks:
            writer.writerow([
                query, "bottom", chunk.text, chunk.metadata["similarity"], chunk.metadata["meta"]
            ])

def save_csv_results(output_file, query, passages, method):
    """
    Save top 10 results with metadata, similarity, and relevance scores for each method.
    """
    # Check if file exists
    write_header = not os.path.exists(output_file)

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new
        if write_header:
            writer.writerow(["query", "passage text", "metadata", "method", "similarity score", "relevance score"])
        
        for passage in passages[:10]:  # Top 10
            if method == "embeddings":
                score = passage.metadata["similarity"]
            else:
                score = passage.score
            writer.writerow([
                query, passage.text, passage.metadata["meta"], method,
                passage.metadata["similarity"], score
            ])

def load_chunks_from_csv(file_path, query, chunk_type):
    """
    Load chunks from a CSV file for a specific query and chunk type (e.g., top or bottom).
    """
    chunks = []
    df = pd.read_csv(file_path)
    
    # Filter rows by query and chunk type
    filtered_df = df[(df["query"] == query) & (df["type"] == chunk_type)]
    
    # Reconstruct the chunks as Document objects
    for _, row in filtered_df.iterrows():
        metadata = eval(row["metadata"])  # Convert metadata string back to dictionary
        chunks.append(Document(text=row["text"], metadata={"meta": metadata, "similarity": row["similarity"], "query": query}))
    
    return chunks


### MAIN ###
def main():
    parser = argparse.ArgumentParser(description="Get embeddings ")
    parser.add_argument('--documents', type=str, required=True, help="Path to documents pickle file")
    parser.add_argument('--queries', type=str, required=True, help="Path to csv file containing queries")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save output (data, logs, plots)")
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Extract queries from csv file
    df = pd.read_csv(args.queries)   
    queries = df["query"].tolist()

    # Open documents
    with open(args.documents, "rb") as f:
        documents = pickle.load(f)

    # Preprocess documents
    for doc in documents:
        preprocess_doc(doc)

    # Chunk documents
    chunks = chunk_docs(documents)

    # Carbon tracking
    tracker = EmissionsTracker(output_dir=args.save_dir, allow_multiple_runs=True)
    
    # Embedding stage
    tracker.start()
    start_time = time.time()
    embedding_model_gen = "BAAI/bge-base-en-v1.5"
    embedding_model_bio = "abhinand/MedEmbed-base-v0.1"
    embeddings_gen = embeddings(embedding_model_gen, chunks)
    np.save( os.path.join(args.save_dir,'embeddings_gen.npy'), embeddings_gen)
    embeddings_bio = embeddings(embedding_model_bio, chunks)
    np.save(os.path.join(args.save_dir,'embeddings_bio.npy'), embeddings_bio)
    end_time = time.time()
    embedding_time = end_time - start_time
    embedding_emissions = tracker.stop()
    embeddings_gen = np.load(os.path.join(args.save_dir,'embeddings_gen.npy'))
    embeddings_bio = np.load(os.path.join(args.save_dir,'embeddings_bio.npy'))

    index_gen = indexing(embeddings_gen)
    index_bio = indexing(embeddings_bio)

    output_csv = os.path.join(args.save_dir, "results.csv")
    for query in queries:
        print(query)
        # Retrieve top and bottom passages
        top_k_chunks_gen, bottom_k_chunks_gen = retrieve_top_bottom(query, embedding_model_gen, index_gen, chunks, top_k=50, bottom_k=10)
        top_k_chunks_bio, bottom_k_chunks_bio = retrieve_top_bottom(query, embedding_model_bio, index_bio, chunks, top_k=50, bottom_k=10)
        
        # Save top and bottom passages for both models
        save_passage_scores(args.save_dir, query, "general", top_k_chunks_gen, bottom_k_chunks_gen)
        save_passage_scores(args.save_dir, query, "bio", top_k_chunks_bio, bottom_k_chunks_bio)
        csv_path = os.path.join(args.save_dir, "bio_passages.csv")

        torch.cuda.empty_cache()
        
        # Reranking with cross-encoder
        tracker.start()
        start_time = time.time()
        reranked_cross = reranking(query, "cross-encoder", top_k_chunks_bio)
        cross_time = time.time() - start_time
        cross_emissions = tracker.stop()
        torch.cuda.empty_cache()
        print("crossencod OK")

        # Reranking with ColBERT
        tracker.start()
        start_time = time.time()
        reranked_colbert = reranking(query, "colbert", top_k_chunks_bio)
        colbert_time = time.time() - start_time
        colbert_emissions = tracker.stop()
        torch.cuda.empty_cache()
        print("colbert OK")

        # Save results to CSV
        save_csv_results(output_csv, query, reranked_cross, "cross-encoder")
        save_csv_results(output_csv, query, reranked_colbert, "colbert")
        save_csv_results(output_csv, query, top_k_chunks_bio[:10], "embeddings")
        
        # Save bottom results only once
        save_csv_results(output_csv, query, bottom_k_chunks_bio, "embeddings")
        print("bottom saved")

    # Log time and emissions
    with open(os.path.join(args.save_dir, "timing_emissions.txt"), "w") as f:
        #f.write(f"Embedding Time: {embedding_time:.2f}s, Emissions: {embedding_emissions:.4f}kg CO2\n")
        f.write(f"Cross-Encoder Time: {cross_time:.2f}s, Emissions: {cross_emissions:.4f}kg CO2\n")
        f.write(f"ColBERT Time: {colbert_time:.2f}s, Emissions: {colbert_emissions:.4f}kg CO2\n")

            

if __name__ == "__main__":
    main()
