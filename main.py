from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AI Chatbot API",
    description="FastAPI endpoint for a Groq-powered AI chatbot with context.",
)


# Initiallizing the Groq client:
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    reply: str


class HistoryManager:

    def __init__(self):
        # The system message, for putting the chatbot in a role.
        self.history = [
            {
                "role": "system",
                "content": "You are a helpful, detailed AI assistant.",
            }
        ]

    def add_pair(self, role: str, content: str):
        # When role=assistant the chatbot's response and user's prompt when role=user.
        new_pair = {
            "role": role,
            "content": content,
        }
        self.history.append(new_pair)
        self.purge_history()

    # Function that keeps the prompt/response history to 2.
    def purge_history(self):

        #  It is written as a pair because it will contain the user's prompt and the chatbot's response, so 4 total messages.
        history_pair_limit = 2

        # 1 is for the already there system message.
        histroy_pair_len = 1 + (2 * history_pair_limit)

        start_index_for_new_messages = len(self.history) - (histroy_pair_len - 1)

        self.history[:] = (
            self.history[0:1] + self.history[start_index_for_new_messages:]
        )

    def get_history(self) -> list:
        return self.history


history_manager = HistoryManager()


# <--------------------UI-------------------->
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Adding prompt/response pair to history:
    history_manager.add_pair(role="user", content=request.prompt)

    # Initialize ai_response with a default message
    ai_response: str = "An unexpected server error occurred."

    try:
        context = history_manager.get_history()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=context,
        )

        content = completion.choices[0].message.content

        if content is None:
            ai_response = "AI returned no content."
        else:
            ai_response = content

    except Exception as e:
        print(f"Chat processing error: {e}")
        ai_response = "AI Chatbot connection Error!"

    history_manager.add_pair(role="assistant", content=ai_response)
    return ChatResponse(reply=ai_response)
