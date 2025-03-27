import os
import streamlit as st
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from config import ASSET_PATH, get_logger
from get_product_documents import get_product_documents
from azure.ai.inference.prompts import PromptTemplate
import time
# Initialize logging and tracing objects
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Set Streamlit page config
st.set_page_config(page_title="Azure AI Chatbot", layout="wide")

# Create project client and chat client
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)
chat = project.inference.get_chat_completions_client()

# Load prompt template once
grounded_chat_prompt = PromptTemplate.from_prompty(os.path.join(ASSET_PATH, "grounded_chat.prompty"))

def chat_with_products(messages: list, context: dict = None) -> dict:
    """Handles chatbot response using Azure AI and product document retrieval."""
    if context is None:
        context = {}

    documents = get_product_documents(messages, context)
    system_message = grounded_chat_prompt.create_messages(documents=documents, context=context)
    
    response = chat.complete(
        model=os.environ["CHAT_MODEL"],
        messages=system_message + messages,
        **grounded_chat_prompt.parameters,
    )

    logger.info(f"💬 Response: {response.choices[0].message}")
    return {"message": response.choices[0].message, "context": context}

# Streamlit UI
st.title("🤖 Drilling bot assistant")
st.markdown("Ask anything about Drilling & Oil...")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    role = msg["role"]
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("What would you like to ask?"):
    # Show user message on right
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Generate bot response
    with st.spinner("Thinking..."):
        try:
            response = chat_with_products(st.session_state["messages"])
            bot_reply = response["message"]["content"]
        except Exception as e:
            bot_reply = f"❌ Error: {str(e)}"
            logger.error(f"Error during chat: {str(e)}")

    # Show assistant message on left
    # Show assistant message on left with typing effect
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        typed_text = ""
        for char in bot_reply:
            typed_text += char
            placeholder.markdown(typed_text + "▌")  # Adding a cursor
            time.sleep(0.01)  # Typing speed (adjust if needed)
        placeholder.markdown(typed_text)  # Final text without cursor

    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

