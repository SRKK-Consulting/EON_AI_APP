from langgraph.graph import StateGraph
from state import AppState
from nodes import orchestrator_node, agent1_node, agent2_node, synthesizer_node

# Build the Graph (linear, nodes self-skip if not selected)
workflow = StateGraph(AppState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("orchestrator")
workflow.add_edge("orchestrator", "agent1")
workflow.add_edge("agent1", "agent2")
workflow.add_edge("agent2", "synthesizer")

# No checkpointing for simple linear workflow (avoids threading issues)
graph = workflow.compile()

if __name__ == "__main__":
    # Example user input
    input_state: AppState = {
        "user_query": "Analyze deals of Opportunity Number OP00138096. Also provide recent industry news.",
    }
    config = {"configurable": {"thread_id": "1"}}
    out_state = graph.invoke(input_state, config)

    # Pretty-print the synthesized note
    print("\n" + "="*80)
    print("FINAL RESPONSE")
    print("="*80)
    print(out_state.get("final_response", ""))