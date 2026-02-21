from dataclasses import dataclass
from typing import Any, Literal

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.runtime import Runtime
from langgraph.store.sqlite import SqliteStore
import sqlite3
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7, max_tokens=2048)
extractor_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

conn = sqlite3.connect("./2-LongTermMemory/memories.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
store = SqliteStore(conn)
store.setup()



@dataclass
class Context:
    user_id: str


class MemoryItem(BaseModel):
    category: Literal["personal", "preferences", "goals", "relationships", "other"]
    key: str = Field(description="Stable unique identifier e.g. 'user_name', 'city', 'job_title'")
    value: str = Field(description="Concise, self-contained fact")
    operation: Literal["upsert", "delete"] = Field(
        default="upsert",
        description="'upsert' to add/update, 'delete' to remove this key"
    )

class ExtractedMemories(BaseModel):
    has_memory: bool = Field(description="True if any facts are worth storing, updating, or deleting")
    memories: list[MemoryItem] = Field(default_factory=list)


EXTRACTOR_SYSTEM_PROMPT = """\
You are a memory extraction assistant. Analyze the user message and decide what facts are worth remembering long-term.

Rules:
- UPSERT: personal info, preferences, goals, relationships, important constraints, ongoing context
- DELETE: user explicitly corrects or retracts a previous fact (e.g. "I moved out of London", "forget I said that")
- Use a stable `key` so the same fact always maps to the same key (e.g. "user_name", "city", "job_title")
- Each `value` should be a concise, self-contained statement
- Return has_memory=false for generic questions, small talk, or messages with no personal information
"""

extractor = extractor_llm.with_structured_output(ExtractedMemories)


def _save_memories(memories: list[MemoryItem], user_id: str):
    for mem in memories:
        namespace = (user_id, mem.category)
        if mem.operation == "upsert":
            store.put(namespace, mem.key, {"value": mem.value})
        elif mem.operation == "delete":
            try:
                store.delete(namespace, mem.key)
            except Exception:
                pass


def _load_all_memories(user_id: str) -> dict[str, list[dict]]:
    all_memories = {}
    for category in ["personal", "preferences", "goals", "relationships", "other"]:
        namespace = (user_id, category)
        try:
            items = store.search(namespace)
            if items:
                all_memories[category] = [
                    {"key": item.key, "value": item.value.get("value", "")}
                    for item in items
                ]
        except Exception:
            pass
    return all_memories


def _format_memories(memories: dict[str, list[dict]]) -> str:
    lines = ["Here is what you know about this user from previous conversations:"]
    for category, items in memories.items():
        if items:
            lines.append(f"\n[{category.upper()}]")
            for item in items:
                lines.append(f"  - {item['key']}: {item['value']}")
    return "\n".join(lines)


@before_model
def inject_memories(state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
    all_memories = _load_all_memories(runtime.context.user_id)
    if not all_memories:
        return None
    memory_msg = SystemMessage(content=_format_memories(all_memories))
    return {"messages": [memory_msg, *state["messages"]]}


@after_model
def extract_memories(state: AgentState, runtime: Runtime[Context]) -> None:
    last_human = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    if not last_human:
        return None

    result: ExtractedMemories = extractor.invoke([
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=last_human.content),
    ])

    if not result.has_memory or not result.memories:
        return None

    _save_memories(result.memories, runtime.context.user_id)


agent = create_agent(
    model=model,
    tools=[],
    middleware=[inject_memories, extract_memories],
    checkpointer=checkpointer,
    store=store,
    context_schema=Context,
)

USER_ID = "user_1"

config: RunnableConfig = {
    "configurable": {
        "thread_id": "1234",
    }
}

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    if user_input.lower() == "quit":
        break

    if user_input.lower() == "memory":
        memories = _load_all_memories(USER_ID)
        print(_format_memories(memories) if memories else "  (no memories stored yet)")
        continue

    results = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        context=Context(user_id=USER_ID),
    )
    print(f"Agent: {results['messages'][-1].content}\n")