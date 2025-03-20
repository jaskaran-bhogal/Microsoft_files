
import os
import sys
import traceback
from datetime import datetime
from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    TurnContext,
    BotFrameworkAdapter,
)
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.schema import Activity, ActivityTypes
import json

from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from config import ASSET_PATH, get_logger
from get_product_documents import get_product_documents
from azure.ai.inference.prompts import PromptTemplate
import json


  # Ensure this file exists with your bot logic
from config import DefaultConfig  # Ensure this config file exists

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount


class MyBot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    async def on_message_activity(self, turn_context: TurnContext):
        messages = [{"role": "user", "content": turn_context.activity.text}]
        await turn_context.send_activity(chat_with_products(messages,{}).content)

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")


CONFIG = DefaultConfig()

# Create adapter settings for authentication
SETTINGS = BotFrameworkAdapterSettings(CONFIG.APP_ID, CONFIG.APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)
LOG_FILE = "requests_log.jsonl"

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
    return  response.choices[0].message


# Handle errors in the bot
async def on_error(context: TurnContext, error: Exception):
    """Handles errors and logs them"""
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    # Inform the user about the error
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity("Please check the bot's source code for issues.")

    # Send error details as a trace in the Bot Emulator
    if context.activity.channel_id == "emulator":
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        await context.send_activity(trace_activity)


# Assign the error handler
ADAPTER.on_turn_error = on_error

# Create the bot instance
BOT = MyBot()

LOG_FILE = "seccess_log.jsonl"

async def messages(req: Request) -> Response:
    """
    Handles incoming messages from Microsoft Bot Framework
    """

    # Ensure request has JSON content
    if "application/json" not in req.headers.get("Content-Type", ""):
        return Response(status=415, text="Invalid Content-Type")

    try:
        body = await req.json()
        print("Received Payload:", json.dumps(body, indent=4))
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(body, ensure_ascii=False) + "\n")

        activity = Activity().deserialize(body)
        auth_header = req.headers.get("Authorization", "")

        # Process the activity and return response
        response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
        

        if response:
        
            return json_response(data=response.body, status=response.status)
        
        return Response(status=201)

    except Exception as e:
        print(f"Error processing request: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return json_response({"error": str(e)}, status=500)


# Create aiohttp web application
APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    try:
        web.run_app(APP, host="0.0.0.0", port=CONFIG.PORT)
    except Exception as error:
        raise error
