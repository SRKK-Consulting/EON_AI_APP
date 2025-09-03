from typing import TypedDict, List, Dict, Optional, Any

class AppState(TypedDict, total=False):
    user_query: str
    parsed_filters: str
    agents_to_call: List[str]
    sql_query: str
    retrieved_deals: List[Dict[str, Any]]
    explanations: List[Dict[str, Any]]
    final_response: str
    errors: List[str]