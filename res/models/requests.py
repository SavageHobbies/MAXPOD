"""This module contains the pydantic models for request bodies"""
from typing import List, Optional
from pydantic import BaseModel


class PatternRequest(BaseModel):
    """A pydantic model for a pattern request"""
    patterns: int
    idea: str
    llm_config: Optional[str] = None


class MockupRequest(BaseModel):
    """A pydantic model for a mockup generation request"""
    templates: List[str]
    designs: List[str]
    category: str
