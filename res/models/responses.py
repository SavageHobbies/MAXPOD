"""This module contains the pydantic models for response bodies"""
from typing import List
from pydantic import BaseModel


class HealthcheckResponse(BaseModel):
    """A pydantic model for a healthcheck response"""
    status: str


class MockupData(BaseModel):
    """A pydantic model for mockup data"""
    id: str
    mockupKey: str
    templateName: str
    designName: str


class MockupResponse(BaseModel):
    """A pydantic model for a mockup generation response"""
    message: str
    mockups: List[MockupData]


class PatternResponse(BaseModel):
    """A pydantic model for a pattern response"""
    message: str
    patterns: List
