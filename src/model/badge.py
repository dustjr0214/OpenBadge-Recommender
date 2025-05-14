from typing import List, Optional
from pydantic import BaseModel, Field

class Badge(BaseModel):
    badge_id: str = Field(..., description="배지 고유 식별자")
    name: str = Field(..., description="배지 이름")
    issuer: str = Field(..., description="배지 발급자")
    description: str = Field(..., description="배지 설명")
    criteria: str = Field(..., description="배지 획득 기준")
    alignment: Optional[str] = Field(None, description="배지 정렬 정보")
    employmentOutcome: Optional[str] = Field(None, description="취업 결과")
    skillsValidated: List[str] = Field(default_factory=list, description="검증된 기술 목록")
    competency: str = Field(..., description="필요한 역량 수준")
    learningOpportunity: Optional[str] = Field(None, description="학습 기회")
    related_badges: List[str] = Field(default_factory=list, description="관련된 다른 배지 목록")

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