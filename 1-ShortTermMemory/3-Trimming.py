from langchain.agents import create_agent, AgentState
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=2048,
)

@before_model
def trim_messages_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]

    # if len(messages) <= 3:
    #     return None

    # first_msg = messages[0]
    # recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    # new_messages = [first_msg] + recent_messages
    last_msg = messages[-1]
    recent_messages = trim_messages(
        messages[:-1],
        strategy='last',
        max_tokens=100,
        token_counter=count_tokens_approximately
    )
    

    new_messages = recent_messages + [last_msg]


    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

checkpointer = InMemorySaver()

agent = create_agent(
    model,
    tools=[],
    checkpointer=checkpointer,
    system_prompt="You're name is Helpful Bot",
    middleware=[trim_messages_middleware]
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