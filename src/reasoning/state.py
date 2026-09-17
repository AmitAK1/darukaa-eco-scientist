import os
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

    def update_state_from_query(self, user_query: str) -> tuple[EnvironmentalState, List[str]]:
        """
        Takes the user's natural language query and extracts environmental metrics 
        to update our structured memory, while also generating scientific synonyms for retrieval.
        """
        prompt = f"""
        You are an environmental data extractor.
        Review the user's current environmental state: {self.current_state.model_dump_json()}
        
        Analyze this new user input: "{user_query}"
        
        1. Update any known environmental variables based on the new input.
        2. Retain previous variables if they are not explicitly changed.
        3. Generate 3 to 5 highly specific scientific synonyms for the core environmental concepts in the input to help search a scientific database.
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StateExtractionResponse,
                    temperature=0.1, # Low temperature for factual extraction
                ),
            )
            
            # Parse the structured JSON response
            extraction = response.parsed
            
            # Update internal state memory
            self.current_state = extraction.updated_state
            
            return self.current_state, extraction.extracted_keywords
            
        except Exception as e:
            print(f"Extraction failed: {e}")
            return self.current_state, []

# --- Local Testing ---
if __name__ == "__main__":
    tracker = StateTracker()
    
    # Turn 1
    query1 = "My land is mostly monoculture wheat in a semi-arid region. We have really low rainfall."
    print(f"\nUser: {query1}")
    state, keywords = tracker.update_state_from_query(query1)
    print("--- Updated State ---")
    print(state.model_dump_json(indent=2))
    print(f"Search Keywords: {keywords}")

    # Turn 2 (Testing Memory and Synonym Generation)
    query2 = "I'm noticing a lot of topsoil washing away, and I think pesticides are killing the good bugs in the dirt."
    print(f"\nUser: {query2}")
    state, keywords = tracker.update_state_from_query(query2)
    print("--- Updated State ---")
    print(state.model_dump_json(indent=2))
    print(f"Search Keywords: {keywords}")