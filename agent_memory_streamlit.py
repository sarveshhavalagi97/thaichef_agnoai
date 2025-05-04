import streamlit as st
from agno.agent import Agent
from agno.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.groq import Groq
import os
from dotenv import load_dotenv
from agno.agent import AgentKnowledge

# Set the Groq API key
groq_api_key = ""

# Initialize the agent with knowledge base and tools
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key="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"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=HuggingfaceCustomEmbedder(),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

# Load knowledge base
if agent.knowledge is not None:
    agent.knowledge.load()

# Streamlit Interface
st.title("🥘 Thai Cuisine Expert")
st.write("Ask me anything about Thai cuisine! From recipes to history, I've got you covered.")

# User Input
user_input = st.text_input("Enter your question:")

# Display Response
if user_input:
    with st.spinner("Thinking..."):
        response = agent.run(user_input)
        st.markdown(response.get_content_as_string())