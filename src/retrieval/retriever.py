import lancedb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import pandas as pd

class EnvironmentalRetriever:
    def __init__(self, db_path: str = ".lancedb", table_name: str = "environmental_knowledge"):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query: str, top_k: int = 5, domains: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Executes Hybrid Search (Vector + Full-Text Keyword) using Reciprocal Rank Fusion (RRF).
        """
        # 1. Vector Search (Semantic)
        query_vector = self.embedder.encode(query).tolist()
        vector_query = self.table.search(query_vector).limit(top_k * 2)
        
        # 2. Full-Text Search (Exact Keyword)
        # We wrap the query in an FTS search. Tantivy handles tokenization.
        fts_query = self.table.search(query, query_type="fts").limit(top_k * 2)
        
        # Apply domain filters if requested
        if domains:
            domain_conditions = [f"array_has(domains, '{d}')" for d in domains]
            where_clause = " OR ".join(domain_conditions)
            vector_query = vector_query.where(where_clause)
            fts_query = fts_query.where(where_clause)

        try:
            vec_df = vector_query.to_pandas()
        except Exception:
            vec_df = pd.DataFrame()
            
        try:
            fts_df = fts_query.to_pandas()
        except Exception:
            fts_df = pd.DataFrame()

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (k + rank), where k is typically 60
        k = 60
        scores = {}
        
        # Score Vector Results
        for rank, (_, row) in enumerate(vec_df.iterrows()):
            chunk_id = row['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank + 1))
            
        # Score FTS Results
        for rank, (_, row) in enumerate(fts_df.iterrows()):
            chunk_id = row['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank + 1))
            
        # Combine all unique rows
        all_rows = pd.concat([vec_df, fts_df]).drop_duplicates(subset=['chunk_id'])
        
        # Map the RRF scores back to the dataframe
        all_rows['rrf_score'] = all_rows['chunk_id'].map(scores)
        
        # Sort by RRF score descending and return the exact top_k requested
        final_df = all_rows.sort_values(by='rrf_score', ascending=False).head(top_k)
        
        return final_df

# --- Local Testing ---
if __name__ == "__main__":
    retriever = EnvironmentalRetriever()
    
    print("--- HYBRID SEARCH DIAGNOSTIC ---")
    query = "What is the effect of synthetic pesticides on soil microorganisms?"
    
    # We will look at the top 3 hybrid results
    res = retriever.search(query, top_k=3, domains=["Human Impact", "Biodiversity", "Soil Health"])
    
    for idx, row in res.iterrows():
        print(f"RRF Score: {row['rrf_score']:.4f} | Source: {row['document_title']} (p. {row['printed_page']})")
        print(f"Snippet: {row['text'][:250]}...\n")
        print("-" * 70)