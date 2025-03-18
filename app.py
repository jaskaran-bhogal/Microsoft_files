import os
from flask import Flask, request, jsonify
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from config import ASSET_PATH, get_logger
from get_product_documents import get_product_documents
from azure.ai.inference.prompts import PromptTemplate

# Initialize Flask app
app = Flask(__name__)

# Initialize logging and tracing objects
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Create a project client using environment variables
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# Create a chat client
chat = project.inference.get_chat_completions_client()


def chat_with_products(messages: list, context: dict = None) -> dict:
    """Handles chatbot response using Azure AI and product document retrieval."""
    if context is None:
        context = {}

    documents = get_product_documents(messages, context)

    # Load prompt template
    grounded_chat_prompt = PromptTemplate.from_prompty(os.path.join(ASSET_PATH, "grounded_chat.prompty"))

    # Construct system message
    system_message = grounded_chat_prompt.create_messages(documents=documents, context=context)
    
    # Call Azure AI chat model
    response = chat.complete(
        model=os.environ["CHAT_MODEL"],
        messages=system_message + messages,
        **grounded_chat_prompt.parameters,
    )

    logger.info(f"💬 Response: {response.choices[0].message}")

    # Return formatted response
    return {"message": response.choices[0].message, "context": context}


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """API Endpoint to handle chat requests."""
    try:
        data = request.json
        if not data or "messages" not in data:
            return jsonify({"error": "Invalid request. 'messages' is required."}), 400
        
        messages = data["messages"]
        context = data.get("context", {})

        response = chat_with_products(messages, context)
        
        return jsonify(str(response))
    
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify API is running."""
    print('I am running !')
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=True)