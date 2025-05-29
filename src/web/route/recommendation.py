from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import sys
import os
from dotenv import load_dotenv

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.recommender import BadgeRecommender
from src.model.badge import BadgeRecommendation
from src.model.user import UserResponse

load_dotenv(verbose=True)

router = APIRouter(
    prefix="/api/recommendations",
    tags=["recommendations"],
    responses={404: {"description": "Not found"}},
)

@router.post("/{user_id}", response_model=Dict[str, List[Dict[str, Any]]])
async def get_badge_recommendations(
    user_id: str, 
    count_recommendation: Optional[int] = None
):
    """
    사용자 ID를 기반으로 오픈배지를 추천합니다.
    
    Args:
        user_id: 사용자 ID
        count_recommendation: 추천할 배지 개수 (1-10개, 기본값: 3)
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: 추천된 배지 목록을 포함한 JSON 응답
    """
    try:
        # 🔍 환경변수 및 디렉토리 확인 (디버깅용)
        print(f"📂 작업 디렉토리: {os.getcwd()}")
        print(f"🔑 환경변수 - ANTHROPIC: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
        print(f"🔑 환경변수 - PINECONE: {bool(os.environ.get('PINECONE_API_KEY'))}")
        
        # 추천 시스템 초기화
        recommender = BadgeRecommender()
        
        if(count_recommendation):
            count_rec = count_recommendation
        else:   
            count_rec = 3   

        # 배지 추천
        recommendation = recommender.recommend_badges(user_id, count_recommendation=count_rec)
        
        if not recommendation or not recommendation.get("recommendations"):
            raise HTTPException(status_code=404, detail="추천 배지를 찾을 수 없습니다.")
            
        return recommendation
        
    except Exception as e:
        print(f"🚨 API 오류 상세: {str(e)}")
        print(f"🚨 오류 타입: {type(e).__name__}")
        
        # 환경변수 관련 오류 체크
        if any(key in str(e) for key in ["ANTHROPIC_API_KEY", "PINECONE_API_KEY"]):
            print("🔑 환경변수 로딩 문제 확인됨!")
        
        raise HTTPException(status_code=500, detail=str(e))

router.get("/user/{user_id}", response_model=UserResponse)
async def get_user_info(user_id: str):
    """
    사용자 정보를 조회합니다.
    
    Args:
        user_id: 사용자 ID
        
    Returns:
        UserResponse: 사용자 정보
    """
    try:
        # 추천 시스템 초기화
        recommender = BadgeRecommender()
        
        # 사용자 정보 조회
        user_info = recommender._get_user_info(user_id)
        
        if not user_info:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
            
        return user_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))