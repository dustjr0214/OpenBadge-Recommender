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
            max_tokens=2048,
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
            
            중요: 위 정보를 바탕으로 이 사용자에게 가장 적합한 배지를 **정확히 {count_recommendation}개** 추천해주세요.
            반드시 {count_recommendation}개의 배지를 JSON 배열에 포함해야 합니다.
            만약 적절한 배지가 {count_recommendation}개보다 작다면 그 수만큼만 추천해주세요.
            """)
        ])
        
        # RAG 체인 구성
        self.chain = (
            {"user_info": RunnablePassthrough(), "badge_info": lambda x: self._get_badge_recommendation(x["user_id"]), "count_recommendation": lambda x: x["count_recommendation"]}
            | self.prompt
            | self.llm
            | self.output_parser
        )
    
    def _get_badge_recommendation(self, user_id: str, top_k: int = 15) -> str:
        """
        사용자 ID를 기반으로 추천 배지 정보를 가져옴
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            추천 배지 정보 문자열
        """
        # ✅ 개선된 retriever 메소드 사용 - 책임 분리
        # retriever에서 사용자 조회, 쿼리 구성, 필터링까지 모두 처리
        recommended_badges = self.retriever.get_similar_badges_for_user(
            user_id=user_id,
            top_k=top_k
        )
        
        # 배지 정보 포맷팅만 담당
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
    
    def recommend_badges(self, user_id: str, count_recommendation: int = 3) -> Dict[str, List[Dict[str, Any]]]:
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
            chain_input = {
                "user_id": user_id,
                "user_info": formatted_user_info,
                "count_recommendation": count_recommendation
            }
            recommendation = self.chain.invoke(chain_input)
            return recommendation
        except Exception as e:
            print(f"추천 생성 중 오류 발생: {str(e)}")
            return {"recommendations": []}

def main():
    # 추천 시스템 초기화
    recommender = BadgeRecommender()
    
    # 예시: 여러 사용자에게 배지 추천 (개선된 결과 확인)
    test_users = ["U07703", "U10003", "U10051"]  # 실제 존재하는 사용자들
    
    for user_id in test_users:
        print(f"\n{'='*60}")
        print(f"🎯 사용자 {user_id} 배지 추천 테스트")
        print(f"{'='*60}")
        
        # 사용자 정보 조회
        user_info = recommender._get_user_info(user_id)
        print(f"\n📋 사용자 정보:")
        print(f"  - 이름: {user_info.get('name', 'N/A')}")
        print(f"  - 목표: {user_info.get('goal', 'N/A')}")
        print(f"  - 기술: {user_info.get('skills', 'N/A')}")
        print(f"  - 역량 수준: {user_info.get('competency_level', 'N/A')}")
        print(f"  - 획득한 배지: {user_info.get('acquired_badges', 'N/A')}")
        
        # 배지 추천
        recommendation = recommender.recommend_badges(user_id, count_recommendation=4)
        
        print(f"\n🎯 추천 결과:")
        if recommendation.get('recommendations'):
            for i, rec in enumerate(recommendation['recommendations']):
                print(f"  {i+1}. {rec.get('name', 'N/A')} (ID: {rec.get('badge_id', 'N/A')})")
                print(f"     발급자: {rec.get('issuer', 'N/A')}")
                print(f"     유사도: {rec.get('similarity_score', 'N/A')}")
                print(f"     추천 이유: {rec.get('recommendation_reason', 'N/A')[:100]}...")
                print()
        else:
            print("  추천할 배지가 없습니다.")
        
        # 결과를 JSON 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"other/test/json/recommendation_{user_id}_{timestamp}.json"
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(recommendation, f, indent=2, ensure_ascii=False)
        
        print(f"📁 추천 결과가 {output_file}에 저장되었습니다.")
        print("-" * 60)

if __name__ == "__main__":
    main() 