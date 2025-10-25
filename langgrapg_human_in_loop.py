from operator import add

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START,END
from langgraph.graph import StateGraph
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI



load_dotenv()
model_name=os.getenv("MODEL_NAME")
model_path=os.getenv('DEEPSEEK_API_BASE')
api_key=os.getenv('DEEPSEEK_API_KEY')

llm=ChatOpenAI(
    openai_api_key=api_key,
    model=model_name,
    base_url=model_path,
    temperature=0.7
)

from typing import  Literal,TypedDict,Annotated
from langgraph.types import interrupt,Command

class State(TypedDict):
    messages: Annotated[list,add]

def human_approval(state: State)-> Command[Literal['call_llm',END]]:
    is_approved=interrupt(
        {
            'question':'是否调用大模型'
        }
    )

    if is_approved:
        return Command(goto='call_llm')
    else:
        return Command(goto=END)

def call_llm(state:State):
    response=llm.invoke(state['messages'])
    return {'messages':[response]}

builder=StateGraph(State)
builder.add_node('human_approval',human_approval)
builder.add_node('call_llm',call_llm)

builder.add_edge(START,'human_approval')
checkpointer=InMemorySaver()
graph=builder.compile(checkpointer=checkpointer)

from langchain_core.messages import HumanMessage
thread_config={'configurable':{'thread_id':1}}
graph.invoke({"messages":[HumanMessage('湖南省会在哪里')]},config=thread_config)

final_result=graph.invoke(Command(resume=True),config=thread_config)
print(final_result)