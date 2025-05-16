import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from src.rag.retriever_openbg import DataRetriever
from src.model.badge import BadgeRecommendation
import json
from datetime import datetime

load_dotenv(verbose=True)

class BadgeRecommender:
    def __init__(self):
        """
        오픈배지 추천 시스템 초기화
        """
        # API 키 검증
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 데이터 검색기 초기화
        self.retriever = DataRetriever(pinecone_api_key=self.pinecone_api_key)
        
        # Claude 모델 초기화
        self.llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            temperature=0.7,
            anthropic_api_key=self.anthropic_api_key
        )
        
        # JSON 출력 파서 초기화
        self.output_parser = JsonOutputParser()
        
        # 프롬프트 템플릿 설정
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 오픈배지 추천 전문가입니다. 
            사용자의 프로필과 관심사를 분석하여 가장 적합한 배지를 추천해주세요.
            
            다음 정보를 바탕으로 추천을 해주세요:
            1. 사용자 프로필
            2. 검색된 배지 정보
            3. 사용자의 현재 역량 수준
            
            각 배지에 대해 다음 형식의 JSON을 반환해주세요:
            {{
                "recommendations": [
                    {{
                        "badge_id": "배지 ID",
                        "name": "배지 이름",
                        "issuer": "발급자",
                        "skills": ["필요한 기술 목록"],
                        "competency": "필요한 역량 수준",
                        "similarity_score": 유사도 점수,
                        "recommendation_reason": "추천 이유",
                        "preparation_steps": "획득을 위한 준비사항",
                        "expected_benefits": "획득 후 기대효과"
                    }}
                ]
            }}
            
            답변은 반드시 위 형식의 JSON만 반환해주세요."""),
            ("human", """
            사용자 정보:
            {user_info}
            
            추천 배지 정보:
            {badge_info}
            
            위 정보를 바탕으로 이 사용자에게 가장 적합한 배지를 추천해주세요.
            """)
        ])
        
        # RAG 체인 구성
        self.chain = (
            {"user_info": RunnablePassthrough(), "badge_info": self._get_badge_info}
            | self.prompt
            | self.llm
            | self.output_parser
        )
    
    def _parse_list_string(self, list_str: str) -> List[str]:
        """
        문자열로 된 리스트를 실제 리스트로 변환
        
        Args:
            list_str: 문자열로 된 리스트
            
        Returns:
            실제 리스트
        """
        try:
            # 문자열이 이미 리스트 형태인 경우
            if isinstance(list_str, list):
                return list_str
            
            # 문자열에서 따옴표 제거 및 공백 제거
            cleaned_str = list_str.strip().replace("'", '"')
            
            # JSON 파싱
            return json.loads(cleaned_str)
        except (json.JSONDecodeError, ValueError):
            # 변환 실패 시 빈 리스트 반환
            return []
    
    def _get_badge_info(self, user_id: str) -> str:
        """
        사용자 ID를 기반으로 추천 배지 정보를 가져옴
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            추천 배지 정보 문자열
        """
        # 사용자 정보 가져오기
        user_info = self._get_user_info(user_id)
        
        # 사용자의 기술과 목표를 기반으로 배지 검색
        query = f"""
        목표: {user_info.get('goal', '')}
        기술: {user_info.get('skills', '')}
        역량 수준: {user_info.get('competency_level', '')}
        """
        
        # 이미 획득한 배지 제외 (문자열 리스트를 실제 리스트로 변환)
        acquired_badges = self._parse_list_string(user_info.get('acquired_badges', '[]'))
        
        # 필터 조건 설정
        filter_criteria = None
        if acquired_badges:
            filter_criteria = {
                "id": {"$nin": acquired_badges}
            }
        
        # 배지 검색
        recommended_badges = self.retriever.search_badges(
            query=query,
            top_k=3,
            filter_criteria=filter_criteria
        )
        
        # 배지 정보 포맷팅
        badge_info = []
        for badge in recommended_badges:
            badge_info.append(f"""
            배지 ID: {badge['id']}
            이름: {badge['metadata']['name']}
            발급자: {badge['metadata']['issuer']}
            기술: {badge['metadata']['skills']}
            역량: {badge['metadata']['competency']}
            유사도 점수: {badge['score']:.4f}
            """)
        
        return "\n".join(badge_info)
    
    def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        사용자 정보를 가져옴
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            사용자 정보 딕셔너리
        """
        # 사용자 ID 검증
        if not user_id or not isinstance(user_id, str):
            print(f"Invalid user ID: {user_id}")
            return {}
            
        # 사용자 정보 검색 (정확한 ID 매칭 사용)
        user_results = self.retriever.search_users(
            query=user_id,
            top_k=1,
            exact_id=True  # 정확한 ID 매칭 사용
        )
        
        if not user_results:
            print(f"No user found with ID: {user_id}")
            return {}
        
        user = user_results[0]
        return {
            "name": user['metadata']['name'],
            "goal": user['metadata']['goal'],
            "skills": user['metadata']['skills'],
            "competency_level": user['metadata']['competency_level'],
            "education_level": user['metadata']['education_level'],
            "acquired_badges": user['metadata']['acquired_badges']
        }
    
    def recommend_badges(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        사용자에게 배지 추천
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            추천 결과를 담은 딕셔너리
        """
        # 사용자 정보 가져오기
        user_info = self._get_user_info(user_id)
        
        if not user_info:
            return {"recommendations": []}
        
        # 사용자 정보 포맷팅
        formatted_user_info = f"""
        이름: {user_info['name']}
        목표: {user_info['goal']}
        기술: {user_info['skills']}
        역량 수준: {user_info['competency_level']}
        교육 수준: {user_info['education_level']}
        획득한 배지: {user_info['acquired_badges']}
        """
        
        try:
            # RAG 체인 실행
            recommendation = self.chain.invoke(formatted_user_info)
            return recommendation
        except Exception as e:
            print(f"추천 생성 중 오류 발생: {str(e)}")
            return {"recommendations": []}

def main():
    # 추천 시스템 초기화
    recommender = BadgeRecommender()
    
    # 예시: 특정 사용자에게 배지 추천
    user_id = "U00113"  # 예시 사용자 ID
    recommendation = recommender.recommend_badges(user_id)
    user_info = recommender._get_user_info(user_id)
    
    print("\n=== 사용자 정보 조회 결과 ===")
    print(user_info)

    print("\n=== 배지 추천 결과 ===")
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
    
    # 결과를 JSON 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"other/test/json/recommendation_{user_id}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendation, f, indent=2, ensure_ascii=False)
    
    print(f"\n추천 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 