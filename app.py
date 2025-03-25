import os
import sys
import traceback
from datetime import datetime
from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import TurnContext
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes
from botbuilder.core import ActivityHandler
from botbuilder.schema import ChannelAccount

from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import ASSET_PATH, get_logger, DefaultConfig
from get_product_documents import get_product_documents
from azure.ai.inference.prompts import PromptTemplate
import json

# Initialize config
CONFIG = DefaultConfig()

# Logging and tracing
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Create Azure AI project client
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"],
    credential=DefaultAzureCredential()
)

# Chat client from the AI project
chat = project.inference.get_chat_completions_client()

# Prompt-based product chat handler
def chat_with_products(messages: list, context: dict = None) -> dict:
    documents = get_product_documents(messages, context)
    grounded_chat_prompt = PromptTemplate.from_prompty(os.path.join(ASSET_PATH, "grounded_chat.prompty"))
    system_message = grounded_chat_prompt.create_messages(documents=documents, context=context)

    response = chat.complete(
        model=os.environ["CHAT_MODEL"],
        messages=system_message + messages,
        **grounded_chat_prompt.parameters,
    )

    logger.info(f"💬 Response: {response.choices[0].message}")
    return response.choices[0].message


# Bot class handling message activity
class MyBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        messages = [{"role": "user", "content": turn_context.activity.text}]
        await turn_context.send_activity(chat_with_products(messages, {}).content)

    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")


# Azure-authenticated CloudAdapter
ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))

# Error handling
async def on_error(context: TurnContext, error: Exception):
    print(f"\n[on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity("Please check the bot's source code for issues.")

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

ADAPTER.on_turn_error = on_error

# Instantiate bot
BOT = MyBot()

# API endpoint handler
async def messages(req: Request) -> Response:
    try:
        return await ADAPTER.process(req, BOT)
    except Exception as e:
        print(f"Error processing request: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return json_response({"error": str(e)}, status=500)

# Web app and routing
APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    try:
        web.run_app(APP, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    except Exception as error:
        raise error
