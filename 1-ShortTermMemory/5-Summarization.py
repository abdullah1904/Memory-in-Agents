from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=2048,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 3),
            keep=("messages", 2)
        )
    ],
    checkpointer=checkpointer,
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

