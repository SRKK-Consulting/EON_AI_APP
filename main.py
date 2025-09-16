import chainlit as cl
import uuid

from langgraph.graph import StateGraph
from state import AppState
from nodes import orchestrator_node, agent1_node, agent2_node, agent3_node, synthesizer_node

# Build the Graph (linear, nodes self-skip if not selected)
workflow = StateGraph(AppState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)
workflow.add_node("agent3", agent3_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("orchestrator")

# Step 1: orchestrator → agent1 OR skip to synthesizer
workflow.add_conditional_edges(
    "orchestrator",
    lambda state: "agent1" if "1" in state.get("agents_to_call", []) else "synthesizer",
    {"agent1": "agent1", "synthesizer": "synthesizer"}
)

# Step 2: agent1 → agent2 OR → synthesizer
workflow.add_conditional_edges(
    "agent1",
    lambda state: "agent2" if "2" in state.get("agents_to_call", []) else "synthesizer",
    {"agent2": "agent2", "synthesizer": "synthesizer"}
)

# Step 3: agent2 → agent3 OR → synthesizer
workflow.add_conditional_edges(
    "agent2",
    lambda state: "agent3" if "3" in state.get("agents_to_call", []) else "synthesizer",
    {"agent3": "agent3", "synthesizer": "synthesizer"}
)

# Step 4: agent3 → synthesizer
workflow.add_edge("agent3", "synthesizer")

# Compile
graph = workflow.compile()


@cl.on_chat_start
async def start_chat():
    # Only create if not already set
    if not cl.user_session.get("thread_id"):
        cl.user_session.set("thread_id", str(uuid.uuid4()))

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    # Retrieve existing history or start new
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    input_state: AppState = {
        "user_query": message.content,
        "history": history,
    }
    await cl.Message(content="Starting analysis...").send()

    config = {"configurable": {"thread_id": thread_id}}
    out_state = await graph.ainvoke(input_state, config)

    response = out_state.get("final_response", "⚠️ No synthesized response was generated.")
    history.append({"role": "assistant", "content": response})

    # Save updated history back into Chainlit session
    cl.user_session.set("history", history)

    #await cl.Message(content=response).send()

'''

if __name__ == "__main__":
    # Example user input
    input_state: AppState = {
        "user_query": "Analyze deal of Opportunity Number OP00138598 based on industry news.",
    }
    config = {"configurable": {"thread_id": "1"}}
    out_state = graph.invoke(input_state, config)

    # Pretty-print the synthesized note
    print("\n" + "="*80)
    print("FINAL RESPONSE")
    print("="*80)
    print(out_state.get("final_response", ""))
'''