from typing import List, Optional
from pydantic import BaseModel, Field

class User(BaseModel):
    user_id: str = Field(..., description="사용자 고유 식별자")
    name: str = Field(..., description="사용자 이름")
    goal: str = Field(..., description="사용자 목표")
    skills: List[str] = Field(default_factory=list, description="보유 기술 목록")
    competency_level: str = Field(..., description="현재 역량 수준")
    learning_history: Optional[str] = Field(None, description="학습 이력")
    employment_history: Optional[str] = Field(None, description="취업 이력")
    education_level: str = Field(..., description="교육 수준")
    engagement_metrics: Optional[str] = Field(None, description="참여 지표")
    acquired_badges: List[str] = Field(default_factory=list, description="획득한 배지 목록")

class UserResponse(BaseModel):
    user_id: str = Field(..., description="사용자 고유 식별자")
    name: str = Field(..., description="사용자 이름")
    goal: str = Field(..., description="사용자 목표")
    skills: List[str] = Field(..., description="보유 기술 목록")
    competency_level: str = Field(..., description="현재 역량 수준")
    education_level: str = Field(..., description="교육 수준")
    acquired_badges: List[str] = Field(..., description="획득한 배지 목록") 