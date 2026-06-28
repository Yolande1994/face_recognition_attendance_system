from pydantic import BaseModel
from typing import List, Optional


class UserInfo(BaseModel):
    user_id: str
    user_name: str


class VerifyResult(BaseModel):
    user_id: str
    user_name: str
    similarity: float
    threshold: float
    is_success: bool


class CheckinResult(BaseModel):
    is_success: bool
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    similarity: Optional[float] = None
    checkin_type: Optional[str] = None


class CheckinRecord(BaseModel):
    user_id: str
    user_name: str
    checkin_time: str
    similarity: float
    checkin_type: str


class UserListItem(BaseModel):
    user_id: str
    user_name: str
    create_time: str