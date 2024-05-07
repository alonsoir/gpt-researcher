from typing import TypedDict


class DraftState(TypedDict):
    task: dict
    title: str
    topic: str
    draft: dict
    review: str
    revision_notes: str
