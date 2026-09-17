from pydantic import BaseModel, Field
from typing import List, Optional

class KnowledgeChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique ID: doc_pdfpage_chunk")
    document_title: str = Field(..., description="Authoritative publication title")
    source_org: str = Field(..., description="FAO, IPCC, IPBES, UNEP")
    year: int = Field(..., description="Publication year")
    pdf_page: int = Field(..., description="Viewer page number (1-based index)")
    printed_page: Optional[str] = Field(None, description="Printed folio number from document")
    domains: List[str] = Field(..., description="Domain tags: Soil Health, Climate, etc.")
    text: str = Field(..., description="Cleaned chunk text")