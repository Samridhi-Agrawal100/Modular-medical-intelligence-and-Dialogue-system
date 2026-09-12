from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from conversation_pipeline import (
    ACTION_SYSTEM_PROMPT,
    OPENING_SYSTEM_PROMPT,
    Model1,
    Model2,
    create_pipeline,
    fallback_response,
    get_last_assistant_message,
    is_repeated_response,
    select_action,
)


class StartConversationRequest(BaseModel):
    patient_id: str | None = None


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ConversationState:
    def __init__(self, model_1: Model1, model_2: Model2) -> None:
        self.model_1 = model_1
        self.model_2 = model_2
        self.sessions: dict[str, ConversationSession] = {}
        self.inference_lock = threading.Lock()

    def start(self) -> tuple[str, str]:
        with self.inference_lock:
            opening = self.model_1.generate(
                [{"role": "system", "content": OPENING_SYSTEM_PROMPT}]
            )

        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ConversationSession(
            history=[{"role": "assistant", "content": opening}]
        )
        return session_id, opening

    def respond(self, session_id: str, patient_message: str) -> dict[str, object]:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)

        session.history.append({"role": "user", "content": patient_message})

        with self.inference_lock:
            decision = self.model_2.predict(session.history, top_k=5)
            action = select_action(decision, session.used_actions)
            session.used_actions.add(action)

            previous_question = get_last_assistant_message(session.history[:-1])
            response_prompt = ACTION_SYSTEM_PROMPT.format(
                label=action,
                confidence=decision["confidence"],
                previous_question=previous_question,
            )
            response_messages = [
                {"role": "system", "content": response_prompt},
                *session.history,
            ]
            assistant_message = self.model_1.generate(response_messages)

            if is_repeated_response(assistant_message, session.history):
                assistant_message = fallback_response(action, session.history)

        session.history.append(
            {"role": "assistant", "content": assistant_message}
        )
        return {
            "assistant_message": assistant_message,
            "status": "active",
        }


@dataclass
class ConversationSession:
    history: list[dict[str, str]] = field(default_factory=list)
    used_actions: set[str] = field(default_factory=set)


app = FastAPI(title="Medical Conversation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: ConversationState | None = None


@app.on_event("startup")
def load_models() -> None:
    global pipeline
    model_1, model_2 = create_pipeline()
    pipeline = ConversationState(model_1, model_2)


@app.get("/", include_in_schema=False)
def api_info() -> dict[str, str]:
    return {
        "service": "Medical Conversation API",
        "status": "ready" if pipeline is not None else "loading",
        "health": "/health",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready" if pipeline is not None else "loading"}


@app.post("/conversation/start")
def start_conversation(
    request: StartConversationRequest,
) -> dict[str, str]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models are still loading")

    session_id, opening = pipeline.start()
    return {
        "session_id": session_id,
        "assistant_message": opening,
        "status": "active",
    }


@app.post("/conversation/{session_id}/message")
def send_message(
    session_id: str,
    request: MessageRequest,
) -> dict[str, object]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models are still loading")

    try:
        result = pipeline.respond(session_id, request.message.strip())
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown session_id") from error

    return {"session_id": session_id, **result}