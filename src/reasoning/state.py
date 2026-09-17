import os
import time
import re
from pydantic import BaseModel, Field
from typing import Optional, List
from google import genai
from google.genai import types

# Define the exact variables Darukaa evaluators are looking for
class EnvironmentalState(BaseModel):
    soil_ph: Optional[str] = Field(None, description="e.g., acidic, 5.5, alkaline")
    soil_organic_carbon: Optional[str] = Field(None, description="e.g., low, 1.2%, high")
    soil_moisture: Optional[str] = Field(None, description="e.g., dry, waterlogged, adequate")
    rainfall: Optional[str] = Field(None, description="e.g., low, 500mm, heavy")
    temperature: Optional[str] = Field(None, description="e.g., arid, 35C, temperate")
    land_use: Optional[str] = Field(None, description="e.g., monoculture wheat, agroforestry, urban")
    human_impact: Optional[str] = Field(None, description="e.g., heavy pesticides, deforestation, pollution")
    biodiversity_status: Optional[str] = Field(None, description="e.g., declining species, habitat fragmentation")

class StateExtractionResponse(BaseModel):
    updated_state: EnvironmentalState
    extracted_keywords: List[str] = Field(..., description="3-5 scientific synonyms based on the user query to improve database retrieval (e.g., if user says 'bugs', output ['microbiota', 'insects', 'microorganisms']).")

class StateTracker:
    def __init__(self):
        # Initializes Gemini Client (automatically picks up GEMINI_API_KEY from .env)
        from dotenv import load_dotenv
        load_dotenv()
        self.client = genai.Client()
        self.current_state = EnvironmentalState()

    def _fallback_extraction(self, user_query: str) -> tuple[EnvironmentalState, List[str]]:
        """
        Rule-based heuristic extraction fallback used when all Gemini API models are unavailable.
        """
        q = user_query.lower()
        new_data = {}
        keywords = []
        
        # pH
        if 'acid' in q or 'ph' in q or 'alkalin' in q:
            if 'acid' in q: new_data['soil_ph'] = 'acidic'
            elif 'alkalin' in q: new_data['soil_ph'] = 'alkaline'
            m = re.search(r'ph\s*(?:is\s*)?(\d+(?:\.\d+)?)', q)
            if m: new_data['soil_ph'] = f"pH {m.group(1)}"
            keywords.append('soil pH')
            
        # Human Impact / Erosion
        impacts = []
        if 'erosion' in q or 'erod' in q: impacts.append('topsoil erosion')
        if 'pesticide' in q: impacts.append('heavy pesticides')
        if 'fertilizer' in q or 'chemical' in q: impacts.append('agrochemical runoff')
        if 'pollution' in q: impacts.append('environmental pollution')
        if 'tillage' in q: impacts.append('intensive tillage')
        if impacts:
            new_data['human_impact'] = ', '.join(impacts)
            keywords.extend(impacts)
            
        # Land Use
        land = []
        if 'monoculture' in q: land.append('monoculture')
        if 'wheat' in q: land.append('wheat farm')
        elif 'crop' in q or 'corn' in q: land.append('cropland')
        if 'pasture' in q: land.append('pasture land')
        if 'agroforestry' in q: land.append('agroforestry')
        if land:
            new_data['land_use'] = ' '.join(land)
            keywords.append('land use')
            
        # Rainfall & Climate
        if 'rainfall' in q or 'rain' in q or 'arid' in q or 'drought' in q:
            if 'low' in q or 'dry' in q or 'drought' in q or 'semi-arid' in q or 'arid' in q:
                new_data['rainfall'] = 'low/arid'
            elif 'heavy' in q or 'high' in q:
                new_data['rainfall'] = 'heavy rainfall'
            keywords.append('rainfall')
            
        if 'temp' in q or 'warm' in q or 'hot' in q or 'cold' in q:
            new_data['temperature'] = 'warm/extreme' if ('hot' in q or 'high' in q or 'warm' in q) else 'moderate'
            keywords.append('temperature')
            
        # Organic Carbon
        if 'carbon' in q or 'soc' in q or 'organic' in q or 'humus' in q:
            new_data['soil_organic_carbon'] = 'low organic carbon' if 'low' in q or 'degrad' in q else 'moderate organic carbon'
            keywords.append('soil organic carbon')
            
        # Moisture
        if 'moist' in q or 'waterlog' in q or 'dry' in q or 'retention' in q:
            new_data['soil_moisture'] = 'waterlogged' if 'waterlog' in q else 'dry/poor retention'
            keywords.append('soil moisture')

        # Biodiversity
        if 'fauna' in q or 'biodivers' in q or 'microb' in q or 'bug' in q or 'insects' in q:
            new_data['biodiversity_status'] = 'declining species/fauna'
            keywords.append('biodiversity')

        # Update current state with non-null values
        curr_dict = self.current_state.model_dump()
        for k, v in new_data.items():
            if v is not None:
                curr_dict[k] = v
        self.current_state = EnvironmentalState(**curr_dict)
        return self.current_state, list(set(keywords))

    def update_state_from_query(self, user_query: str) -> tuple[EnvironmentalState, List[str]]:
        """
        Takes the user's natural language query and extracts environmental metrics 
        to update our structured memory, trying fast & available models first.
        """
        prompt = f"""
        You are an environmental data extractor.
        Review the user's current environmental state: {self.current_state.model_dump_json()}
        
        Analyze this new user input: "{user_query}"
        
        1. Update any known environmental variables based on the new input.
        2. Retain previous variables if they are not explicitly changed.
        3. Generate 3 to 5 highly specific scientific synonyms for the core environmental concepts in the input to help search a scientific database.
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
                        response_schema=StateExtractionResponse,
                        temperature=0.1,
                    ),
                )
                
                extraction = response.parsed
                if extraction:
                    self.current_state = extraction.updated_state
                    return self.current_state, extraction.extracted_keywords
                
            except Exception as e:
                err_str = str(e)
                print(f"[StateTracker] Model {model_name} failed: {err_str[:100]}")
                continue
                
        # If all API models failed or quota was exhausted, use fallback heuristic extraction
        print("[StateTracker] Utilizing fallback keyword extraction...")
        return self._fallback_extraction(user_query)

# --- Local Testing ---
if __name__ == "__main__":
    tracker = StateTracker()
    query1 = "My land is mostly monoculture wheat in a semi-arid region. We have really low rainfall."
    print(f"\nUser: {query1}")
    state, keywords = tracker.update_state_from_query(query1)
    print("--- Updated State ---")
    print(state.model_dump_json(indent=2))
    print(f"Search Keywords: {keywords}")