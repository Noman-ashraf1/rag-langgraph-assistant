
from typing import TypedDict, List, Dict, Any


class State(TypedDict):
    query: str
    context: str
    docs: List[Dict[str, Any]]
    answer: str
    evaluation: Dict[str, Any]

    # NEW
    history: List[str]