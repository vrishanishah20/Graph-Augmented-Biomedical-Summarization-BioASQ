import os
import json
import networkx as nx
import spacy
import pandas as pd
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from rouge_score import rouge_scorer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cugraph
import cudf
import re
# automatic GPU acceleration
os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"

# tokenizer = MT5Tokenizer.from_pretrained("/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/clean_mt5_tokenizer", legacy=False)

def load_json_files(file_paths):
    documents, summaries, languages = [], [], []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            documents.append(item['full_text'])
            summaries.append(item['summary'])
            languages.append(item.get('language', 'unknown'))
    return documents, summaries, languages

def clean_comm_text(text):
    # Removing citations
    text = re.sub(r'\[\d+(-\d+)?\]', '', text)
    # Collapse whitespace
    return ' '.join(text.split())

# Chunk documents 
def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
    chunks = []
    for doc in documents:
        for i in range(0, len(doc), chunk_size - overlap_size):
            chunk = doc[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

# Entity extraction 
nlp = spacy.load("en_core_sci_sm")  # Instead of en_core_web_sm for better scientific extraction
def extract_entities(chunk):
    doc = nlp(chunk)
    return [ent.text for ent in doc.ents]

# Building knowledge graph using chunks
def build_graph(chunks):
    G = nx.Graph()
    entity_to_chunks = {}
    for idx, chunk in enumerate(chunks):
        entities = extract_entities(chunk)
        for ent in entities:
            entity_to_chunks.setdefault(ent, set()).add(idx)
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                G.add_edge(entities[i], entities[j])
    return G, entity_to_chunks

#saving as csv to load for neo4j
def export_neo4j_csv(G, output_dir):
    """Export graph to Neo4j-compatible CSVs"""
    os.makedirs(output_dir, exist_ok=True)
    # Export nodes
    nodes = [{"id": node} for node in G.nodes()]
    nodes_df = pd.DataFrame(nodes)
    nodes_path = os.path.join(output_dir, "entities.csv")
    nodes_df.to_csv(nodes_path, index=False)
    print(f"Saved {len(nodes_df)} nodes to {nodes_path}")

    # Export relationships
    edges = [{"source": u, "target": v} for u, v in G.edges()]
    edges_df = pd.DataFrame(edges)
    edges_path = os.path.join(output_dir, "relationships.csv")
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved {len(edges_df)} relationships to {edges_path}")
    
# Community detection 
from networkx.algorithms import community
def detect_communities(G):
    # Convert NetworkX graph to cuGraph format for gpu support
    edges = list(G.edges())
    
    # Create cuDF DataFrame from edges
    df = cudf.DataFrame(edges, columns=['src', 'dst'])
    
    # Build cuGraph graph
    G_cugraph = cugraph.Graph()
    G_cugraph.from_cudf_edgelist(df, source='src', destination='dst')
    
    # Perform Louvain community detection on GPU
    parts, _ = cugraph.louvain(G_cugraph)
    
    # Convert results to community list format
    communities_dict = parts.groupby('partition').agg({'vertex': list})['vertex'].to_pandas().to_dict()
    return list(communities_dict.values())

#Aggregate text for each community
def aggregate_text_for_community(community, chunks, entity_to_chunks):
    relevant_chunk_ids = set()
    for entity in community:
        relevant_chunk_ids.update(entity_to_chunks.get(entity, []))
    texts = [chunks[i] for i in relevant_chunk_ids]
    return " ".join(texts)

# Summarization with fine-tuned mT5 
def summarize_text_mt5(text, tokenizer, model, device, max_length=128):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=2.5,  # Reducing repeated tokens
        no_repeat_ngram_size=3    # Preventing 3-gram repeats
    )
    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True  
    )

# ROUGE evaluation 
def compute_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, summary)

def main(
    json_path,
    model_path,
    language_filter='en',
    chunk_size=600,
    overlap_size=100,
    max_length=128
):
    file_paths = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')]
    if not file_paths:
        raise ValueError(f"No JSON files found in {json_path}")
   
    documents, reference_summaries, languages = load_json_files(file_paths)
    print(f"Loaded {len(documents)} documents.")
   
 
    chunks = split_documents_into_chunks(documents, chunk_size, overlap_size)
    print(f"Created {len(chunks)} chunks.")


    G, entity_to_chunks = build_graph(chunks)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    nx.write_graphml(G, "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/knowledge_graph.graphml")
    nx.write_gpickle(G, "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/knowledge_graph.gpickle")
    export_neo4j_csv(G, "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/neo4j_data")

    del G  # Freeing up memory
    import gc
    gc.collect()

    communities = detect_communities(G)
    print(f"Detected {len(communities)} communities.")

    # Load fine-tuned mT5
    from transformers import MT5Config

    config = MT5Config.from_pretrained(model_path)
    tokenizer = MT5Tokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Summarize and evaluate
    all_scores = []
    for idx, community in enumerate(communities):
        comm_text = aggregate_text_for_community(community, chunks, entity_to_chunks)
        if not comm_text.strip():
            continue
        comm_text = clean_comm_text(comm_text)
        if len(comm_text.split()) < 50:  # Skipping very short texts
            continue
        summary = summarize_text_mt5(comm_text, tokenizer, model, device, max_length)
        
        
        # Compute similarity between the community text and all reference summaries
        texts_to_compare = [comm_text] + reference_summaries
        tfidf = TfidfVectorizer().fit_transform(texts_to_compare)
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        best_match_idx = cosine_similarities.argmax()
        ref_summary = reference_summaries[best_match_idx]
     
        
        # Compute ROUGE
        if ref_summary:
            scores = compute_rouge(ref_summary, summary)
            all_scores.append(scores)
            print(f"\nCommunity {idx+1} summary:\n{summary}")
            print(f"ROUGE: {scores}")

    # Average ROUGE
    if all_scores:
        avg_rouge = {
            'rouge1': sum(s['rouge1'].fmeasure for s in all_scores)/len(all_scores),
            'rouge2': sum(s['rouge2'].fmeasure for s in all_scores)/len(all_scores),
            'rougeL': sum(s['rougeL'].fmeasure for s in all_scores)/len(all_scores)
        }
        print("\nAverage ROUGE Scores:", avg_rouge)
    else:
        print("No summaries to score.")

if __name__ == "__main__":
    json_path = "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/converted_json"
    model_path = "/content/drive/MyDrive/Biomedical-Summarization-Using-GraphRAG/final_model"
    main(json_path, model_path)
