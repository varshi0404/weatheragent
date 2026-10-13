import streamlit as st

from agent import WeatherAgent


st.set_page_config(page_title="Weather Agent", page_icon="324", layout="centered")

st.title("Weather Agent")
st.caption("Local Ollama-powered weather insights using Open-Meteo data.")

with st.sidebar:
    st.subheader("Settings")
    model_name = st.text_input("Ollama model", value="llama3:8b")
    use_agent = st.checkbox("Use LLM orchestration", value=False)
    verbose = st.checkbox("Verbose output", value=False)
    output_mode = st.selectbox("Output mode", ["report", "mood"], index=0)

st.subheader("City")
city = st.text_input("Enter city name", value="Chennai")

st.subheader("Query (optional)")
query = st.text_input("Custom query", value="")

if st.button("Get Weather Insights", type="primary"):
    if not city.strip():
        st.error("Please enter a city name.")
    else:
        with st.spinner("Generating report..."):
            agent = WeatherAgent(
                model_name=model_name,
                verbose=verbose,
                use_agent=use_agent,
                output_mode=output_mode,
                default_city=city.strip(),
            )
            final_query = query.strip() or f"Weather for {city.strip()}"
            report = agent.run(final_query)
        st.subheader("Report")
        st.text(report)

st.divider()
st.markdown(
    """
**Notes**
- If you enable **Use LLM orchestration**, the model must follow ReAct formatting.
- For best reliability with local models, keep it off.
"""
)
