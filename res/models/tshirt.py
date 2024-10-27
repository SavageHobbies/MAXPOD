"""This module contains the pydantic models for the tshirt resource"""
import re
import json
from pydantic import BaseModel, field_validator


class TshirtFromAi(BaseModel):
    """A pydantic model for ai generated fields"""
    product_name: str
    description: str
    # tshirt_text is the text that will be printed on the tshirt
    tshirt_text: str
    marketing_tags: list[str]


class TshirtFromAiList(BaseModel):
    """A pydantic model for a list of ai generated fields"""
    patterns: list[TshirtFromAi]

    @classmethod
    def model_validate_json(cls, json_str: str):
        """Custom validation for JSON string from Ollama"""
        # Find the JSON object in the response
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                return cls.model_validate(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")
        raise ValueError("No valid JSON object found in the response")


class TshirtWithIds(TshirtFromAi):
    """A pydantic model that extends TshirtFromAi with a product_id"""
    product_id: str
    image_ids: list[str]
