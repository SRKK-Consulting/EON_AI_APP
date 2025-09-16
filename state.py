
from typing import TypedDict, List, Dict, Optional, Any, Annotated
from langgraph.channels import LastValue
from operator import add
from langgraph.channels.topic import Topic

# OPTION 1: Allow accumulating multiple errors (Recommended)
class AppState(TypedDict, total=False):
    user_query: str
    parsed_filters: str
    agents_to_call: List[str]
    sql_query: str
    # These fields can be updated by multiple agents concurrently
    retrieved_deals: Annotated[List[Dict[str, Any]], add]
    explanations: Annotated[List[Dict[str, Any]], add]
    news: Annotated[List[Dict[str, Any]], add]
    final_response: Annotated[str, Topic(str, accumulate=True)]
    errors: Annotated[List[str], add]  
    history: List[Dict[str, str]]