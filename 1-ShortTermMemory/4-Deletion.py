from langchain.messages import RemoveMessage, HumanMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=2048,
)


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 4:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:4]]}
    return None


agent = create_agent(
    model=model,
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config:RunnableConfig = {
    'configurable': {
        "thread_id": "1234",
    }
}

while True:
    user_input = input("Enter a message (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    results = agent.invoke({
        "messages": [
            HumanMessage(content=user_input),
        ]
    },config=config)
    print("Agent response:", results['messages'][-1].content)