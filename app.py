import streamlit as st
import json
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

def is_state_complete(state: EnvironmentalState) -> tuple[bool, str]:
    """
    Checks if we have enough environmental variables to perform multi-metric reasoning.
    Requires at least 3 variables to be non-empty.
    """
    filled_vars = [v for k, v in state.model_dump().items() if v is not None]
    
    if len(filled_vars) < 3:
        # Determine what's missing to ask a smart clarifying question
        if state.soil_organic_carbon is None and state.soil_ph is None:
            return False, "To give you a scientifically grounded recommendation, I need a bit more context. Could you tell me about your **soil health** (e.g., organic carbon levels, pH, or moisture) and **rainfall**?"
        elif state.land_use is None:
            return False, "I see the issue. To understand how this interacts with the ecosystem, what is the current **land use** (e.g., monoculture, agroforestry) and what does the **climate/temperature** look like there?"
        else:
            return False, "Could you provide a few more details about the **human impact** (e.g., pollution, pesticides) or **climate** so I can connect the variables?"
            
    return True, ""

# --- Layout: Sidebar for State Visualization ---
with st.sidebar:
    st.header("📊 Current Environmental State")
    st.markdown("*(Extracted dynamically from conversation)*")
    current_state_json = st.session_state.tracker.current_state.model_dump_json(indent=2)
    st.code(current_state_json, language="json")

# --- Layout: Main Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if user_input := st.chat_input("E.g., My monoculture wheat farm has low rainfall and declining biodiversity..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing environmental state..."):
            # 2. Update State
            state, keywords = st.session_state.tracker.update_state_from_query(user_input)
            st.rerun() # Refresh sidebar to show updated JSON state immediately

# --- Logic Processing (Runs after state updates) ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    current_state = st.session_state.tracker.current_state
    
    # 3. Clarifying Question Engine
    is_ready, clarification_msg = is_state_complete(current_state)
    
    with st.chat_message("assistant"):
        if not is_ready:
            st.markdown(clarification_msg)
            st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
        else:
            with st.spinner("Retrieving scientific evidence and reasoning..."):
                # 4. Multi-Metric Reasoning Engine
                # We need the last user input and its keywords. Since st.rerun clears local variables, 
                # we grab the last user message from memory.
                last_user_msg = st.session_state.messages[-1]["content"]
                _, keywords = st.session_state.tracker.update_state_from_query(last_user_msg) 
                
                recommendation = st.session_state.engine.generate_recommendation(last_user_msg, current_state, keywords)
                
                if recommendation:
                    # Format the output beautifully for the judges
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
                else:
                    st.error("Engine failed to generate a scientifically grounded recommendation.")