import streamlit as st
import json
import re
from src.reasoning.state import StateTracker, EnvironmentalState
from src.reasoning.engine import ReasoningEngine

# --- App Configuration ---
st.set_page_config(page_title="Darukaa AI Environmental Scientist", layout="wide")
st.title("🌱 Darukaa AI Environmental Scientist")
st.markdown("I am an AI researcher grounded in FAO and IPCC science. Describe your land's environmental state, and I will provide evidence-backed interventions.")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tracker" not in st.session_state:
    st.session_state.tracker = StateTracker()
if "engine" not in st.session_state:
    st.session_state.engine = ReasoningEngine()

# --- Greeting Response Template ---
GREETING_RESPONSE = """👋 **Welcome to Darukaa AI Environmental Scientist**

I am an AI environmental research assistant grounded in empirical **FAO** and **IPCC** scientific reports. I evaluate multi-metric ecosystem dynamics to recommend evidence-backed interventions.

### 🌐 My Scientific Scope & Evaluation Domains
1. 🧪 **Soil Health**: Soil organic carbon (SOC), pH levels, soil moisture, topsoil erosion.
2. 🌾 **Land Use & Crop Management**: Monoculture, agroforestry, crop rotation, cover crops.
3. 🐝 **Biodiversity**: Soil microbiota, pollinators, fauna species diversity, habitat fragmentation.
4. 🌡️ **Climate & Hydrology**: Rainfall patterns, temperature extremes, drought conditions, runoff.
5. 🚜 **Human Impact & Agrochemicals**: Pesticides, synthetic fertilizers, tillage practices, pollution.

---

### 💡 Suggested Prompts to Try:
- *"I operate a monoculture wheat farm in a low rainfall region. Recently, I've noticed a severe decline in beneficial soil fauna due to heavy pesticide application."*
- *"My degraded pasture land is experiencing severe topsoil erosion. We are in a semi-arid climate, and our soil organic carbon levels are critically low."*
- *"I am dealing with severe topsoil erosion, and my soil pH has become highly acidic."""


def is_greeting(text: str) -> bool:
    """
    Detects if the user query is a generic greeting, small talk, or capability query.
    """
    cleaned = text.lower().strip().strip("!.,?::-_")
    
    # Exact single-word greetings and common conversational fillers
    greeting_words = {
        "hi", "hii", "hiii", "hello", "hlo", "hey", "heyy", "hy",
        "good", "ok", "okay", "cool", "great", "nice", "fine",
        "thanks", "thankyou", "thank", "help", "who", "what"
    }
    if cleaned in greeting_words:
        return True
        
    # Regex patterns for elongated small talk words (e.g. "hiiii", "heyyyy", "helloooo")
    patterns = [
        r"^h+i+$",
        r"^h+e+l+l+o+$",
        r"^h+e+y+$",
        r"^g+o+o+d+$",
        r"^o+k+a*y*$",
        r"^t+h+a+n+k+s*$",
        r"^g+r+e+a+t+$",
        r"^c+o+o+l+$"
    ]
    for pattern in patterns:
        if re.match(pattern, cleaned):
            return True

    # Multi-word common greeting phrases
    greeting_phrases = {
        "who are you", "what can you do?", "what can you do", "good morning", "good evening", 
        "good afternoon", "hi there", "hello there", "what can this app do",
        "what do you do", "how are you", "how do you work", "hello, what can you do?"
    }
    if cleaned in greeting_phrases:
        return True

    greeting_starters = ["hi ", "hello ", "hey ", "what can you ", "who are you", "what do you "]
    for starter in greeting_starters:
        if cleaned.startswith(starter) and len(cleaned.split()) <= 4:
            return True
            
    return False


def is_state_complete(state: EnvironmentalState, last_query: str = "") -> tuple[bool, str]:
    """
    Checks if we have enough environmental variables to perform multi-metric reasoning.
    Requires at least 3 non-empty variables and provides dynamic, context-aware prompts.
    """
    state_dict = state.model_dump()
    filled_vars = {k: v for k, v in state_dict.items() if v is not None}
    
    if len(filled_vars) < 3:
        has_land_or_human = ("land_use" in filled_vars or "human_impact" in filled_vars)
        has_soil = ("soil_ph" in filled_vars or "soil_organic_carbon" in filled_vars or "soil_moisture" in filled_vars)
        has_climate = ("rainfall" in filled_vars or "temperature" in filled_vars)
        
        if has_land_or_human and not (has_soil or has_climate):
            return False, "I notice you've provided land use or human management details, but I lack soil and climate context. Could you share your regional **rainfall**, **temperature**, or **soil characteristics** (e.g., pH, organic carbon)?"
        elif has_soil and not has_land_or_human:
            return False, "I see your soil health parameters, but to understand ecosystem interactions, could you tell me about your **land use practices** (e.g., monoculture vs agroforestry) or **chemical/pesticide inputs**?"
        elif len(filled_vars) > 0:
            return False, "I've recorded your environmental parameters. To perform a multi-metric causal analysis grounded in FAO/IPCC data, I need at least 3 parameters. Could you provide a bit more detail about your **land use**, **soil condition**, or **climate**?"
        else:
            if last_query:
                return False, f"I noted your input: *\"{last_query}\"*. To perform a multi-metric causal analysis grounded in FAO/IPCC data, I need specific environmental parameters. Could you describe your **land use** (e.g., monoculture wheat), **soil health** (pH, organic carbon), or **climate/rainfall**?"
            else:
                return False, "To provide a multi-metric causal analysis grounded in FAO/IPCC data, I need at least 3 environmental parameters. Could you describe your current **land use**, observed **soil conditions**, and regional **climate/rainfall**?"
            
    return True, ""


# --- Layout: Sidebar for State Management & Visualization ---
with st.sidebar:
    st.header("📊 Current Environmental State")
    st.caption("Extracted dynamically from conversation")
    
    if st.button("🔄 Start New Assessment / Clear State", use_container_width=True):
        st.session_state.tracker = StateTracker()
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    st.subheader("Extracted Parameters")
    
    FIELD_LABELS = {
        "soil_ph": "Soil pH",
        "soil_organic_carbon": "Soil Organic Carbon",
        "soil_moisture": "Soil Moisture",
        "rainfall": "Rainfall",
        "temperature": "Temperature",
        "land_use": "Land Use",
        "human_impact": "Human Impact",
        "biodiversity_status": "Biodiversity Status"
    }
    
    current_state = st.session_state.tracker.current_state
    state_dict = current_state.model_dump()
    
    for field_key, field_label in FIELD_LABELS.items():
        val = state_dict.get(field_key)
        if val is not None:
            st.markdown(f"🟢 **{field_label}**: `{val}`")
        else:
            st.markdown(f"⚪ **{field_label}**: *Unknown*")
            
    st.divider()
    with st.expander("Inspect Raw Pydantic State (JSON)", expanded=False):
        current_state_json = current_state.model_dump_json(indent=2)
        st.code(current_state_json, language="json")


# --- Layout: Main Chat Interface & History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Layout: Landing Page / Empty State Starter Scenario Chips ---
starter_prompt = None

if len(st.session_state.messages) == 0:
    st.markdown("""
    ### 🌍 Welcome to Darukaa AI Environmental Scientist
    Grounded in **FAO** and **IPCC** scientific data, this system uses **LanceDB Hybrid RRF (Reciprocal Rank Fusion)** retrieval and Pydantic-validated reasoning to assess ecosystem health across 5 core domains.
    """)
    st.markdown("#### 💡 Select a starter scenario to test the scientific reasoning engine:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌾 **Scenario A: Soil Degradation**\n\nMonoculture wheat with declining soil fauna under low rainfall.", use_container_width=True):
            starter_prompt = "Monoculture wheat farm with declining soil fauna and heavy pesticide use under low rainfall."
    with col2:
        if st.button("🏜️ **Scenario B: Carbon & Land Use**\n\nDegraded pasture land in semi-arid zone with low SOC & erosion.", use_container_width=True):
            starter_prompt = "Degraded pasture land in a semi-arid zone with low soil organic carbon and erosion."
    with col3:
        if st.button("💧 **Scenario C: Agrochemical Runoff**\n\nRunoff affecting microbial biomass & water retention in drought.", use_container_width=True):
            starter_prompt = "Intensive agrochemical runoff affecting microbial biomass and water retention in drought conditions."


# --- Chat Input Handling ---
chat_input_val = st.chat_input("E.g., My monoculture wheat farm has low rainfall and declining biodiversity...")
user_input = starter_prompt or chat_input_val

if user_input:
    # 1. Immediately render user input message so the UI responds instantly on Enter
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Intent Gating: Check for Greetings / Small Talk
    if is_greeting(user_input):
        with st.chat_message("assistant"):
            st.markdown(GREETING_RESPONSE)
        st.session_state.messages.append({"role": "assistant", "content": GREETING_RESPONSE})
        st.rerun()
    else:
        # 3. Scientific Query Processing & State Update with Visual Progress Spinners
        with st.chat_message("assistant"):
            with st.spinner("🧪 Extracting environmental parameters & updating state..."):
                state, keywords = st.session_state.tracker.update_state_from_query(user_input)
                is_ready, clarification_msg = is_state_complete(state, last_query=user_input)
            
            if not is_ready:
                st.markdown(clarification_msg)
                st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
                st.rerun()
            else:
                with st.spinner("🔍 Retrieving FAO/IPCC evidence & generating causal recommendation..."):
                    recommendation = st.session_state.engine.generate_recommendation(user_input, state, keywords)
                    
                    if recommendation:
                        reply = f"### Recommended Action: {recommendation.action}\n\n"
                        reply += f"**Time Horizon:** {recommendation.time_horizon} | **Confidence:** {recommendation.confidence_level}\n\n"
                        
                        reply += "#### Scientific Reasoning Chain\n"
                        for step in recommendation.causal_reasoning_chain:
                            reply += f"- {step}\n"
                            
                        reply += "\n#### Impacted Metrics\n"
                        for metric in recommendation.impacted_metrics:
                            icon = "📈" if metric.expected_change.lower() == "increase" else "📉" if metric.expected_change.lower() == "decrease" else "➖"
                            reply += f"- {icon} **{metric.metric_name}**: {metric.expected_change}\n"
                            
                        reply += f"\n#### Evidence & Citation\n> *\"{recommendation.evidence_citation}\"*\n"
                        
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                    else:
                        err_msg = "⚠️ **Gemini API Quota Exceeded (429)**: The free-tier rate limit for Gemini API was reached. Please wait ~30-60 seconds and submit your query again."
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                        st.rerun()