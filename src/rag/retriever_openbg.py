import os
from typing import List, Dict, Any, Optional
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(verbose=True)

class DataRetriever:
    def __init__(self, 
                 pinecone_api_key: str,
                 embedding_model: str = "multilingual-e5-large",
                 index_name: str = "openbadges"):
        """
        데이터 검색을 위한 클래스 초기화
        
        Args:
            pinecone_api_key: Pinecone API 키
            embedding_model: 사용할 임베딩 모델 이름
            index_name: Pinecone 인덱스 이름
        """
        self.embedding_model = embedding_model
        self.index_name = index_name
        
        # 임베딩 모델 초기화
        self.embeddings = PineconeEmbeddings(
            model=embedding_model,
            pinecone_api_key=pinecone_api_key
        )
        
        # Pinecone 초기화
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
    
    def search_badges(self, 
                     query: str, 
                     top_k: int = 5,
                     filter_criteria: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        배지 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_criteria: 메타데이터 필터링 조건
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리 텍스트를 벡터로 변환
        query_vector = self.embeddings.embed_query(query)
        
        # 검색 실행
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace="badge",
            filter=filter_criteria,
            include_metadata=True
        )
        
        return self._format_results(results)
    
    def search_users(self, 
                    query: str, 
                    top_k: int = 5,
                    filter_criteria: Optional[Dict] = None,
                    exact_id: bool = False) -> List[Dict[str, Any]]:
        """
        사용자 검색
        
        Args:
            query: 검색 쿼리 또는 사용자 ID
            top_k: 반환할 결과 수
            filter_criteria: 메타데이터 필터링 조건
            exact_id: 정확한 ID 매칭 여부
            
        Returns:
            검색 결과 리스트
        """
        if exact_id:
            # 정확한 ID 매칭을 위한 검색
            try:
                # ID 값 정제 (공백 제거 및 ASCII 문자만 허용)
                clean_id = query.strip()
                if not clean_id.isascii():
                    print(f"Warning: ID contains non-ASCII characters: {clean_id}")
                    return []
                
                results = self.index.query(
                    id=clean_id,
                    top_k=top_k,
                    namespace="user",
                    include_metadata=True
                )
            except Exception as e:
                print(f"Error querying Pinecone: {e}")
                return []
        else:
            # 의미적 검색
            query_vector = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace="user",
                filter=filter_criteria,
                include_metadata=True
            )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """
        검색 결과 포맷팅
        
        Args:
            results: Pinecone 검색 결과
            
        Returns:
            포맷팅된 결과 리스트
        """
        formatted_results = []
        for match in results.matches:
            formatted_result = {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_similar_badges_for_user(self, 
                                  user_id: str, 
                                  top_k: int = 5) -> List[Dict[str, Any]]:
        """
        특정 사용자에게 추천할 배지 검색
        
        Args:
            user_id: 사용자 ID
            top_k: 반환할 결과 수
            
        Returns:
            추천 배지 리스트
        """
        # 사용자 정보 조회
        user_results = self.index.query(
            id=user_id,
            top_k=1,
            namespace="user",
            include_metadata=True
        )
        
        if not user_results.matches:
            return []
        
        user_data = user_results.matches[0]
        
        # 사용자의 기술과 목표를 기반으로 배지 검색
        query = f"""
        목표: {user_data.metadata.get('goal', '')}
        기술: {user_data.metadata.get('skills', '')}
        역량 수준: {user_data.metadata.get('competency_level', '')}
        """
        
        # 이미 획득한 배지 제외
        acquired_badges = user_data.metadata.get('acquired_badges', [])
        filter_criteria = {
            "id": {"$nin": acquired_badges}
        }
        
        return self.search_badges(query, top_k, filter_criteria)

def main():
    # 환경 변수에서 API 키 가져오기
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 검색기 초기화
    retriever = DataRetriever(pinecone_api_key=pinecone_api_key)
    
    # 예시: 배지 검색
    badge_results = retriever.search_badges(
        query="머신러닝과 데이터 분석에 관심이 있는 초보자를 위한 배지",
        top_k=3
    )
    print("\n=== 배지 검색 결과 ===")
    for result in badge_results:
        print(f"배지 ID: {result['id']}")
        print(f"이름: {result['metadata']['name']}")
        print(f"점수: {result['score']:.4f}")
        print("---")
    
    # 예시: 사용자 검색
    user_results = retriever.search_users(
        query="머신러닝과 데이터 분석에 관심이 있는 사용자",
        top_k=3
    )
    print("\n=== 사용자 검색 결과 ===")
    for result in user_results:
        print(f"사용자 ID: {result['id']}")
        print(f"이름: {result['metadata']['name']}")
        print(f"목표: {result['metadata']['goal']}")
        print(f"점수: {result['score']:.4f}")
        print("---")
    
    # 예시: 특정 사용자에게 추천할 배지
    user_id = "U01199"  # 예시 사용자 ID
    recommended_badges = retriever.get_similar_badges_for_user(user_id, top_k=3)
    print(f"\n=== 사용자 {user_id}에게 추천할 배지 ===")
    for badge in recommended_badges:
        print(f"배지 ID: {badge['id']}")
        print(f"이름: {badge['metadata']['name']}")
        print(f"점수: {badge['score']:.4f}")
        print("---")

if __name__ == "__main__":
    main() 