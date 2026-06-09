# app/schemas.py

from pydantic import BaseModel


class DigitizeResponse(BaseModel):
    layout: str
    signal: list