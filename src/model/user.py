from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    user_id: str = Field(..., description="사용자의 고유 식별자")
    name: str = Field(..., description="사용자의 이름")
    email: str = Field(..., description="사용자의 이메일 주소")
    goal: str = Field(..., description="사용자의 목표")
    skills: List[str] = Field(default_factory=list, description="보유한 기술 목록")
    competency_level: str = Field(..., description="역량 수준")
    acquired_badges: List[str] = Field(default_factory=list, description="획득한 배지 목록")
    learning_history: Optional[str] = Field(None, description="학습 이력")
    employment_history: Optional[str] = Field(None, description="취업 이력")
    education_level: Optional[str] = Field(None, description="교육 수준")
    engagement_metrics: Optional[str] = Field(None, description="참여도 지표")
    recommendation_history: List[str] = Field(default_factory=list, description="추천 이력")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "U00901",
                "name": "Robert Ochoa",
                "email": "yalvarado@example.org",
                "goal": "Become an AI Expert",
                "skills": ["Cloud Computing"],
                "competency_level": "Beginner",
                "acquired_badges": ["B09997", "B07193"],
                "learning_history": "Completed the 'Deep Learning Master' course at Google",
                "employment_history": "Worked at Rivas LLC for 4 years",
                "education_level": "Bachelor's Degree",
                "engagement_metrics": "Highly Active",
                "recommendation_history": []
            }
        }

class UserResponse(BaseModel):
    user_id: str = Field(..., description="사용자 고유 식별자")
    name: str = Field(..., description="사용자 이름")
    goal: str = Field(..., description="사용자 목표")
    skills: List[str] = Field(..., description="보유 기술 목록")
    competency_level: str = Field(..., description="현재 역량 수준")
    education_level: str = Field(..., description="교육 수준")
    acquired_badges: List[str] = Field(..., description="획득한 배지 목록") 