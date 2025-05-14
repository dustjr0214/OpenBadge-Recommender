import os
import json
import glob
from typing import List, Dict, Any
from pathlib import Path
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

load_dotenv(verbose=True)

class DataEmbedder:
    def __init__(self, 
                 pinecone_api_key: str,
                 embedding_model: str = "multilingual-e5-large",
                 index_name: str = "openbadges"):
        """
        데이터 임베딩을 위한 클래스 초기화
        
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
        self._initialize_index()
        
    def _initialize_index(self):
        """Pinecone 인덱스 초기화"""
        cloud = os.environ.get('PINECONE_CLOUD', 'aws')
        region = os.environ.get('PINECONE_REGION', 'us-east-1')
        spec = ServerlessSpec(cloud=cloud, region=region)
        
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embeddings.dimension,
                metric="cosine",
                spec=spec
            )
            # 인덱스가 준비될 때까지 대기
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
    
    def preprocess_badge(self, badge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        배지 데이터 전처리
        
        Args:
            badge_data: 원본 배지 JSON 데이터
            
        Returns:
            전처리된 배지 데이터
        """
        # 배지 정보를 하나의 텍스트로 결합
        text = f"""
        배지명: {badge_data.get('name', '')}
        발급자: {badge_data.get('issuer', '')}
        설명: {badge_data.get('description', '')}
        기준: {badge_data.get('criteria', '')}
        정렬: {badge_data.get('alignment', '')}
        취업 결과: {badge_data.get('employmentOutcome', '')}
        검증된 기술: {badge_data.get('skillsValidated', '')}
        역량: {badge_data.get('competency', '')}
        학습 기회: {badge_data.get('learningOpportunity', '')}
        """
        
        return {
            "id": badge_data.get('badge_id'),
            "text": text,
            "metadata": {
                "name": badge_data.get('name'),
                "issuer": badge_data.get('issuer'),
                "skills": badge_data.get('skillsValidated'),
                "competency": badge_data.get('competency'),
                "related_badges": badge_data.get('related_badges')
            }
        }
    
    def preprocess_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 데이터 전처리
        
        Args:
            user_data: 원본 사용자 JSON 데이터
            
        Returns:
            전처리된 사용자 데이터
        """
        # 사용자 정보를 하나의 텍스트로 결합
        text = f"""
        이름: {user_data.get('name', '')}
        목표: {user_data.get('goal', '')}
        기술: {user_data.get('skills', '')}
        역량 수준: {user_data.get('competency_level', '')}
        학습 이력: {user_data.get('learning_history', '')}
        취업 이력: {user_data.get('employment_history', '')}
        교육 수준: {user_data.get('education_level', '')}
        참여 지표: {user_data.get('engagement_metrics', '')}
        """
        
        return {
            "id": user_data.get('user_id'),
            "text": text,
            "metadata": {
                "name": user_data.get('name'),
                "goal": user_data.get('goal'),
                "skills": user_data.get('skills'),
                "competency_level": user_data.get('competency_level'),
                "acquired_badges": user_data.get('acquired_badges'),
                "education_level": user_data.get('education_level')
            }
        }
    
    def process_data(self, data_dir: str, data_type: str):
        """
        데이터 파일들을 처리하고 Pinecone에 저장
        
        Args:
            data_dir: 데이터 JSON 파일이 있는 디렉토리 경로
            data_type: 데이터 타입 ('badge' 또는 'user')
        """
        if data_type not in ['badge', 'user']:
            raise ValueError("data_type은 'badge' 또는 'user'여야 합니다.")
        
        # 파일 패턴 설정
        file_pattern = f"{data_type}_*.json"
        data_files = glob.glob(os.path.join(data_dir, file_pattern))
        processed_data = []
        
        # 데이터 전처리
        for data_file in data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data_type == 'badge':
                    processed_item = self.preprocess_badge(data)
                else:
                    processed_item = self.preprocess_user(data)
                processed_data.append(processed_item)
        
        # 텍스트 임베딩 생성 및 Pinecone에 저장
        texts = [item["text"] for item in processed_data]
        metadatas = [item["metadata"] for item in processed_data]
        ids = [item["id"] for item in processed_data]
        
        # Pinecone에 데이터 저장 (네임스페이스 지정)
        index = self.pc.Index(self.index_name)
        for text, metadata, id in zip(texts, metadatas, ids):
            vector = self.embeddings.embed_query(text)
            index.upsert(
                vectors=[(id, vector, metadata)],
                namespace=data_type
            )
        
        print(f"{data_type} 데이터 임베딩 생성 및 저장 완료: {len(texts)}개")

def main():
    # 환경 변수에서 API 키 가져오기
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 데이터 디렉토리 경로
    data_dir = "data/json"
    
    # 임베더 초기화
    embedder = DataEmbedder(pinecone_api_key=pinecone_api_key)
    
    # 배지 데이터 처리
    embedder.process_data(os.path.join(data_dir, "badge"), "badge")
    
    # 사용자 데이터 처리
    embedder.process_data(os.path.join(data_dir, "user"), "user")

if __name__ == "__main__":
    main()