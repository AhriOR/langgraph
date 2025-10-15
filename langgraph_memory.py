import os
from dotenv import load_dotenv



load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
model=os.getenv("MODEL_NAME")
url=os.getenv("DEEPSEEK_API_BASE")

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(
    openai_api_key=api_key,
    model=model,
    base_url=url,
    temperature=0.7
)

from langgraph.graph import StateGraph,MessagesState,START
from langgraph.checkpoint.memory import InMemorySaver

def call_model(state:MessagesState):
    response=llm.invoke(state['messages'])
    return {'messages':response}

builder=StateGraph(MessagesState)
builder.add_node('node1',call_model)
builder.add_edge(START,'node1')

checkpointer=InMemorySaver()
graph=builder.compile(checkpointer=checkpointer)

config={
    'configurable':{
        'thread_id':'1'
    }
}

for chunk in graph.stream({'messages':{'role':'user','content':'湖南的省会在哪里？'}},config=config,stream_node='values'):
    print(chunk)
for chunk in graph.stream({'messages':{'role':'user','content':'湖北呢？'}},config=config,stream_node='values'):
    print(chunk)