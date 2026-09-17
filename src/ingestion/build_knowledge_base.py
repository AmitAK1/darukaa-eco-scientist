import os
import re
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import lancedb
from typing import Optional

DATA_DIR = Path("data/raw_pdfs")
DB_URI = ".lancedb"
TABLE_NAME = "environmental_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Document registry with domain mappings and max page limits
DOC_REGISTRY = {
    "ipbes": {
        "title": "IPBES Global Assessment Report on Biodiversity and Ecosystem Services",
        "org": "IPBES",
        "year": 2019,
        "domains": ["Biodiversity", "Land Use", "Human Impact"],
        "max_pdf_page": 45  # Covers the complete Summary for Policymakers
    },
    "ipcc": {
        "title": "IPCC Special Report on Climate Change and Land",
        "org": "IPCC",
        "year": 2019,
        "domains": ["Climate", "Land Use", "Soil Health"],
        "max_pdf_page": 45  # Covers SPM
    },
    "pollution": {
        "title": "FAO Global Assessment of Soil Pollution",
        "org": "FAO & UNEP",
        "year": 2021,
        "domains": ["Soil Health", "Human Impact", "Biodiversity"],
        "max_pdf_page": 62  # Ingests Chapters 1-4, cuts off references at p. 51-57
    },
    "recarbonizing": {
        "title": "FAO Recarbonizing Global Soils - A Technical Manual",
        "org": "FAO",
        "year": 2021,
        "domains": ["Soil Health", "Land Use", "Biodiversity"],
        "max_pdf_page": 350 # Ingests management practices (silviculture, rice, urban)
    },
    "soil_carbon": {
        "title": "Soil Carbon Sequestration for Improved Land Management",
        "org": "FAO",
        "year": 2020,
        "domains": ["Soil Health", "Biodiversity"],
        "max_pdf_page": None # Ingest entire document
    },
    "guidelines": {
        "title": "FAO Guidelines for Soil Description",
        "org": "FAO",
        "year": 2006,
        "domains": ["Soil Health"],
        "max_pdf_page": None
    }
}

def resolve_metadata(filename: str):
    name_lower = filename.lower()
    for key, meta in DOC_REGISTRY.items():
        if key in name_lower:
            return meta
    return {
        "title": filename.replace(".pdf", ""),
        "org": "Environmental Agency",
        "year": 2021,
        "domains": ["Soil Health", "Biodiversity"],
        "max_pdf_page": None
    }

def extract_printed_page(raw_lines: list[str]) -> Optional[str]:
    """Detects printed folio numbers in the top 2 or bottom 2 lines of extracted page text."""
    candidate_lines = []
    if len(raw_lines) >= 1:
        candidate_lines.extend(raw_lines[:2])
        candidate_lines.extend(raw_lines[-2:])
    
    for line in candidate_lines:
        line_clean = line.strip()
        # Case 1: Standalone number (e.g., '2' or '45')
        if re.fullmatch(r"\d{1,4}", line_clean):
            return line_clean
        # Case 2: Number followed by running title (e.g., '2 RECARBONIZING GLOBAL SOILS')
        match = re.match(r"^(\d{1,4})\s+[A-Z\s]{4,}", line_clean)
        if match:
            return match.group(1)
        # Case 3: Running title followed by number (e.g., 'GLOBAL ASSESSMENT OF SOIL POLLUTION 12')
        match_end = re.search(r"[A-Z\s]{4,}\s+(\d{1,4})$", line_clean)
        if match_end:
            return match_end.group(1)
            
    return None

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 100:
            chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks

def build_knowledge_base():
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    db = lancedb.connect(DB_URI)
    records = []
    
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files located in {DATA_DIR.resolve()}")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_path in pdf_files:
        meta = resolve_metadata(pdf_path.name)
        print(f"\nProcessing: {pdf_path.name}")
        
        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            max_limit = meta["max_pdf_page"] or total_pages
            end_page = min(max_limit, total_pages)
            
            print(f"  -> Ingesting PDF pages 1 to {end_page} (Total pages in file: {total_pages})")
            
            for page_idx in range(end_page):
                page_obj = reader.pages[page_idx]
                raw_text = page_obj.extract_text() or ""
                
                # Filter image-only pages and blank spacers
                if len(raw_text.strip()) < 150:
                    continue
                
                raw_lines = [l for l in raw_text.splitlines() if l.strip()]
                detected_printed_page = extract_printed_page(raw_lines)
                cleaned = clean_text(raw_text)
                
                chunks = chunk_text(cleaned)
                for chunk_idx, chunk in enumerate(chunks):
                    records.append({
                        "chunk_id": f"{pdf_path.stem}_p{page_idx+1}_c{chunk_idx}",
                        "document_title": meta["title"],
                        "source_org": meta["org"],
                        "year": meta["year"],
                        "pdf_page": page_idx + 1,
                        "printed_page": detected_printed_page or str(page_idx + 1),
                        "domains": meta["domains"],
                        "text": chunk
                    })
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    if not records:
        print("No text chunks were generated.")
        return

    print(f"\nExtracted {len(records)} high-density chunks.")
    print("Generating vector embeddings...")
    
    texts = [r["text"] for r in records]
    vectors = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    
    for i, rec in enumerate(records):
        rec["vector"] = vectors[i].tolist()

    print(f"Persisting chunks to LanceDB at '{DB_URI}' (table: '{TABLE_NAME}')...")
    table = db.create_table(TABLE_NAME, data=records, mode="overwrite")
    print(f"Persisting chunks to LanceDB at '{DB_URI}' (table: '{TABLE_NAME}')...")
    table = db.create_table(TABLE_NAME, data=records, mode="overwrite")
    
    # ADD THIS LINE: Build a Full-Text Search index on the text column
    table.create_fts_index("text", replace=True) 
    
    print(f"Knowledge ingestion complete. {len(table)} records indexed successfully.")


if __name__ == "__main__":
    build_knowledge_base()