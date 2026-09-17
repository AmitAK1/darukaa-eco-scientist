import os
import time
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

# Import our previously built modules
from src.reasoning.state import EnvironmentalState
from src.retrieval.retriever import EnvironmentalRetriever

# --- 1. Define the Output Schema the Judges Want to See ---
class ImpactedMetric(BaseModel):
    metric_name: str = Field(..., description="e.g., Soil Organic Carbon, Microbial Biomass")
    expected_change: str = Field(..., description="e.g., Increase, Stabilize, Decrease")

class ScientificRecommendation(BaseModel):
    action: str = Field(..., description="The exact intervention to perform.")
    causal_reasoning_chain: List[str] = Field(..., description="Step-by-step logic connecting at least 3 environmental variables (e.g., 'Low rainfall + monoculture -> reduces soil moisture', 'Cover crops -> retain moisture', 'Increased moisture -> supports soil fauna').")
    impacted_metrics: List[ImpactedMetric]
    time_horizon: str = Field(..., description="Short, Medium, or Long term.")
    evidence_citation: str = Field(..., description="Exact quote or citation string from the provided source documents. DO NOT hallucinate.")
    confidence_level: str = Field(..., description="High, Medium, or Low based on evidence match.")

class ReasoningEngine:
    def __init__(self):
        self.retriever = EnvironmentalRetriever()
        self.client = genai.Client()

    def generate_recommendation(self, user_query: str, state: EnvironmentalState, keywords: List[str]) -> ScientificRecommendation:
        # 1. Expand the query with our scientific keywords for better Hybrid Search
        expanded_query = f"{user_query} {' '.join(keywords)}"
        print(f"[*] Retrieving evidence for: {expanded_query[:80]}...")
        
        # 2. Retrieve top 4 chunks using RRF Hybrid Search
        evidence_df = self.retriever.search(expanded_query, top_k=4)
        
        # 3. Format the evidence into a readable string for the LLM
        evidence_text = ""
        for idx, row in evidence_df.iterrows():
            evidence_text += f"\n--- Source: {row['document_title']} (Page {row['printed_page']}) ---\n"
            evidence_text += f"Fact: {row['text']}\n"

        # 4. Prompt the LLM to reason across variables using ONLY the evidence
        prompt = f"""
        You are a strict, senior environmental scientist. 
        
        Current Environmental State of the user's land:
        {state.model_dump_json(exclude_none=True)}
        
        Retrieved Scientific Evidence (Use ONLY these facts):
        {evidence_text}
        
        Task: 
        Generate a multi-variable intervention. You MUST connect at least 3 environmental variables from the user's state. 
        Your 'evidence_citation' must explicitly reference the Source Title and Page provided in the evidence above.
        """

        # Fast and high-quota models prioritized first to save retry time
        models_chain = ['gemini-3.5-flash-lite', 'gemini-3.5-flash', 'gemini-flash-latest', 'gemini-3.6-flash']

        for model_name in models_chain:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ScientificRecommendation,
                        temperature=0.2, # Low temp for factual accuracy
                    ),
                )
                if response.parsed:
                    print(f"[*] Successfully generated recommendation using {model_name}")
                    return response.parsed
                
            except Exception as e:
                err_str = str(e)
                print(f"[ReasoningEngine] Model {model_name} failed: {err_str[:100]}")
                continue

        return None

# --- Local Testing ---
if __name__ == "__main__":
    from src.reasoning.state import StateTracker
    
    tracker = StateTracker()
    engine = ReasoningEngine()
    
    query = "I'm noticing a lot of topsoil washing away, and I think pesticides are killing the good bugs in the dirt. My land is monoculture wheat with low rainfall."
    print("Extracting State...")
    state, keywords = tracker.update_state_from_query(query)
    
    print("\nGenerating Scientific Recommendation...")
    recommendation = engine.generate_recommendation(query, state, keywords)
    
    if recommendation:
        print("\n=== FINAL OUTPUT ===")
        print(recommendation.model_dump_json(indent=2))