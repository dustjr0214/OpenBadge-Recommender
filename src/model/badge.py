from typing import List, Optional
from pydantic import BaseModel, Field

class Badge(BaseModel):
    badge_id: str = Field(..., description="배지의 고유 식별자")
    name: str = Field(..., description="배지의 이름")
    issuer: str = Field(..., description="배지를 발급한 기관")
    description: str = Field(..., description="배지에 대한 설명")
    criteria: str = Field(..., description="배지 획득 기준")
    alignment: Optional[str] = Field(None, description="산업 표준 정렬")
    employmentOutcome: Optional[str] = Field(None, description="취업 결과")
    skillsValidated: List[str] = Field(default_factory=list, description="검증된 기술 목록")
    competency: List[str] = Field(default_factory=list, description="역량 목록")
    learningOpportunity: Optional[str] = Field(None, description="학습 기회")
    related_badges: List[str] = Field(default_factory=list, description="관련된 배지 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "badge_id": "B00601",
                "name": "Cloud Computing Guru",
                "issuer": "Udacity",
                "description": "Cloud Computing Guru badge issued by Udacity",
                "criteria": "Passing all module assessments",
                "alignment": "Industry Standard Alignment",
                "employmentOutcome": "Eligible for Machine Learning Engineer positions",
                "skillsValidated": ["Python", "Machine Learning", "SQL"],
                "competency": ["Cloud Computing", "Problem Solving"],
                "learningOpportunity": "Data Science Bootcamp",
                "related_badges": ["B06866", "B01842"]
            }
        }

class BadgeRecommendation(BaseModel):
    badge_id: str = Field(..., description="추천 배지 ID")
    name: str = Field(..., description="추천 배지 이름")
    issuer: str = Field(..., description="배지 발급자")
    skills: List[str] = Field(..., description="필요한 기술")
    competency: str = Field(..., description="필요한 역량 수준")
    similarity_score: float = Field(..., description="유사도 점수")
    recommendation_reason: str = Field(..., description="추천 이유")
    preparation_steps: str = Field(..., description="획득을 위한 준비사항")
    expected_benefits: str = Field(..., description="획득 후 기대효과") 