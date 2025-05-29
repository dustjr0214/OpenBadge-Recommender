import os
import json
import glob
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time
import pickle
import threading
from datetime import datetime, timedelta

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
        
        # 삭제된 벡터 백업을 위한 딕셔너리 (메모리 기반)
        self._deleted_vectors_backup = {}
        
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
    
    def _determine_namespace_from_id(self, vector_id: str) -> str:
        """
        벡터 ID를 기반으로 네임스페이스 결정
        
        Args:
            vector_id: 벡터 ID
            
        Returns:
            네임스페이스 ('badge' 또는 'user')
            
        Raises:
            ValueError: ID 형식이 올바르지 않은 경우
        """
        if not vector_id:
            raise ValueError("벡터 ID가 비어있습니다.")
            
        if vector_id.upper().startswith('B'):
            return 'badge'
        elif vector_id.upper().startswith('U'):
            return 'user'
        else:
            raise ValueError(
                f"올바르지 않은 ID 형식입니다. 'B' 또는 'U'로 시작해야 합니다: "
                f"{vector_id}"
            )
    
    def _backup_vector_to_file(self, vector_id: str, namespace: str, 
                               vector_data: Dict[str, Any]):
        """
        삭제될 벡터를 파일로 백업 (30분 후 자동 삭제)
        
        Args:
            vector_id: 벡터 ID
            namespace: 네임스페이스
            vector_data: 벡터 데이터 (id, values, metadata 포함)
        """
        backup_dir = "backup/deleted_vectors"
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_data = {
            'vector_id': vector_id,
            'namespace': namespace,
            'vector_data': vector_data,
            'deleted_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(minutes=30)).isoformat()
        }
        
        backup_file = os.path.join(backup_dir, f"{vector_id}.pkl")
        
        try:
            with open(backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            
            # 30분 후 백업 파일 자동 삭제를 위한 타이머 설정
            timer = threading.Timer(1800, self._cleanup_backup_file, 
                                  args=[backup_file])
            timer.daemon = True
            timer.start()
            
            print(f"🗂️ 벡터 백업 완료: {backup_file}")
            
        except Exception as e:
            print(f"⚠️ 백업 실패: {str(e)}")
    
    def _backup_vector_to_memory(self, vector_id: str, namespace: str, 
                                 vector_data: Dict[str, Any]):
        """
        삭제될 벡터를 메모리에 백업 (30분 후 자동 삭제)
        
        Args:
            vector_id: 벡터 ID
            namespace: 네임스페이스
            vector_data: 벡터 데이터
        """
        backup_data = {
            'vector_id': vector_id,
            'namespace': namespace,
            'vector_data': vector_data,
            'deleted_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(minutes=30)
        }
        
        self._deleted_vectors_backup[vector_id] = backup_data
        
        # 30분 후 메모리에서 자동 삭제
        timer = threading.Timer(1800, self._cleanup_memory_backup, 
                                  args=[vector_id])
        timer.daemon = True
        timer.start()
        
        print(f"💾 벡터 메모리 백업 완료: {vector_id}")
    
    def _cleanup_backup_file(self, backup_file: str):
        """백업 파일 자동 삭제"""
        try:
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"🗑️ 백업 파일 자동 삭제: {backup_file}")
        except Exception as e:
            print(f"⚠️ 백업 파일 삭제 실패: {str(e)}")
    
    def _cleanup_memory_backup(self, vector_id: str):
        """메모리 백업 자동 삭제"""
        try:
            if vector_id in self._deleted_vectors_backup:
                del self._deleted_vectors_backup[vector_id]
                print(f"🗑️ 메모리 백업 자동 삭제: {vector_id}")
        except Exception as e:
            print(f"⚠️ 메모리 백업 삭제 실패: {str(e)}")
    
    def restore_vector(self, vector_id: str) -> Tuple[bool, str]:
        """
        백업된 벡터를 복원
        
        Args:
            vector_id: 복원할 벡터 ID
            
        Returns:
            (성공여부, 메시지)
        """
        try:
            # 메모리 백업에서 확인
            if vector_id in self._deleted_vectors_backup:
                backup_data = self._deleted_vectors_backup[vector_id]
                
                if datetime.now() <= backup_data['expires_at']:
                    # 벡터 복원
                    index = self.pc.Index(self.index_name)
                    vector_data = backup_data['vector_data']
                    
                    index.upsert(
                        vectors=[(
                            vector_data['id'],
                            vector_data['values'],
                            vector_data['metadata']
                        )],
                        namespace=backup_data['namespace']
                    )
                    
                    # 백업에서 제거
                    del self._deleted_vectors_backup[vector_id]
                    
                    return True, f"✅ 벡터 복원 성공: {vector_id}"
                else:
                    del self._deleted_vectors_backup[vector_id]
                    return False, f"❌ 복원 기간이 만료되었습니다: {vector_id}"
            
            # 파일 백업에서 확인
            backup_file = f"backup/deleted_vectors/{vector_id}.pkl"
            if os.path.exists(backup_file):
                with open(backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                
                expires_at = datetime.fromisoformat(backup_data['expires_at'])
                if datetime.now() <= expires_at:
                    # 벡터 복원
                    index = self.pc.Index(self.index_name)
                    vector_data = backup_data['vector_data']
                    
                    index.upsert(
                        vectors=[(
                            vector_data['id'],
                            vector_data['values'],
                            vector_data['metadata']
                        )],
                        namespace=backup_data['namespace']
                    )
                    
                    # 백업 파일 삭제
                    os.remove(backup_file)
                    
                    return True, f"✅ 벡터 복원 성공: {vector_id}"
                else:
                    os.remove(backup_file)
                    return False, f"❌ 복원 기간이 만료되었습니다: {vector_id}"
            
            return False, f"❌ 백업된 벡터를 찾을 수 없습니다: {vector_id}"
            
        except Exception as e:
            return False, f"❌ 벡터 복원 실패: {str(e)}"
    
    def delete_vector(self, vector_id: str, backup_method: str = "memory") -> Tuple[bool, str]:
        """
        Pinecone에서 벡터를 삭제 (30분간 복원 가능하도록 백업)
        
        Args:
            vector_id: 삭제할 벡터 ID (B로 시작하면 badge, U로 시작하면 user)
            backup_method: 백업 방법 ("memory" 또는 "file")
            
        Returns:
            (성공여부, 메시지)
        """
        try:
            # 1. ID로부터 네임스페이스 결정
            namespace = self._determine_namespace_from_id(vector_id)
            data_type = "배지" if namespace == "badge" else "사용자"
            
            # 2. 벡터 존재 여부 확인
            index = self.pc.Index(self.index_name)
            try:
                existing_vector = index.fetch(ids=[vector_id], namespace=namespace)
                if (not existing_vector.vectors or 
                    vector_id not in existing_vector.vectors):
                    return False, (f"❌ 해당 {data_type}가 존재하지 않습니다: "
                                  f"{vector_id}")
                
                # 3. 삭제 전 백업
                vector_data = existing_vector.vectors[vector_id]
                backup_data = {
                    'id': vector_id,
                    'values': vector_data.values,
                    'metadata': vector_data.metadata
                }
                
                if backup_method == "file":
                    self._backup_vector_to_file(vector_id, namespace, 
                                              backup_data)
                else:
                    self._backup_vector_to_memory(vector_id, namespace, 
                                                backup_data)
                
            except Exception as e:
                return False, f"❌ 벡터 확인 중 오류 발생: {str(e)}"
            
            # 4. 벡터 삭제
            try:
                index.delete(ids=[vector_id], namespace=namespace)
                return True, (f"✅ {data_type} 삭제 성공: {vector_id} "
                              f"(30분간 복원 가능)")
                
            except Exception as e:
                return False, f"❌ {data_type} 삭제 실패: {str(e)}"
                
        except ValueError as e:
            return False, f"❌ {str(e)}"
        except Exception as e:
            return False, f"❌ 예상치 못한 오류: {str(e)}"

    def _determine_data_type(self, file_path: str) -> str:
        """
        JSON 파일을 분석하여 데이터 타입(badge 또는 user)을 결정
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            데이터 타입 ('badge' 또는 'user')
            
        Raises:
            ValueError: 데이터 타입을 결정할 수 없는 경우
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # badge 데이터의 특징적인 필드들 확인
            badge_fields = {
                'badge_id', 'name', 'issuer', 'criteria', 'skillsValidated'
            }
            # user 데이터의 특징적인 필드들 확인
            user_fields = {
                'user_id', 'goal', 'competency_level', 'learning_history'
            }
            
            data_keys = set(data.keys())
            
            # badge 필드와의 교집합이 더 많으면 badge
            badge_match = len(badge_fields.intersection(data_keys))
            user_match = len(user_fields.intersection(data_keys))
            
            if badge_match > user_match:
                return 'badge'
            elif user_match > badge_match:
                return 'user'
            else:
                # 파일명으로도 확인
                filename = os.path.basename(file_path).lower()
                if 'badge' in filename:
                    return 'badge'
                elif 'user' in filename:
                    return 'user'
                else:
                    raise ValueError(
                        f"데이터 타입을 결정할 수 없습니다: {file_path}")
                    
        except Exception as e:
            raise ValueError(
                f"파일을 읽거나 분석하는 중 오류가 발생했습니다: {file_path}, "
                f"오류: {str(e)}")
    
    def upsert_vector(self, file_path: str):
        """
        특정 JSON 파일을 처리하여 Pinecone에 벡터를 upsert
        
        Args:
            file_path: 처리할 JSON 파일 경로
        """
        try:
            # 1. 파일 타입 확인
            data_type = self._determine_data_type(file_path)
            print(f"파일 타입 확인: {data_type}")
            
            # 2. 파일 로드 및 전처리
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data_type == 'badge':
                processed_item = self.preprocess_badge(data)
            else:
                processed_item = self.preprocess_user(data)
            
            # 3. 동일한 ID가 있는지 확인
            index = self.pc.Index(self.index_name)
            vector_id = processed_item["id"]
            
            try:
                existing_vector = index.fetch(
                    ids=[vector_id], namespace=data_type
                )
                if existing_vector.vectors:
                    print(f"벡터 업데이트: ID {vector_id} "
                          f"(네임스페이스: {data_type})")
                else:
                    print(f"새 벡터 삽입: ID {vector_id} "
                          f"(네임스페이스: {data_type})")
            except Exception:
                print(f"새 벡터 삽입: ID {vector_id} "
                      f"(네임스페이스: {data_type})")
            
            # 4. 벡터 임베딩 생성
            vector = self.embeddings.embed_query(processed_item["text"])
            
            # 5. Pinecone에 upsert
            try:
                index.upsert(
                    vectors=[(vector_id, vector, processed_item["metadata"])],
                    namespace=data_type
                )
                
                # 성공 메시지 출력
                name = processed_item["metadata"].get("name", "이름 없음")
                print(f"✅ upsert 성공: {name} (ID: {vector_id}, "
                      f"타입: {data_type})")
                
            except Exception as e:
                print(f"❌ upsert 실패: ID {vector_id}, 오류: {str(e)}")
                
        except Exception as e:
            print(f"❌ 파일 처리 실패: {file_path}, 오류: {str(e)}")

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
    
    def upsert_all(self, data_dir: str, data_type: str):
        """
        데이터 파일들을 처리하고 Pinecone에 저장 (기존 process_data에서 이름 변경)
        
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
    """
    /data 의 json 배지/유저 파일 임베딩
    
    """
    # 환경 변수에서 API 키 가져오기
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 데이터 디렉토리 경로
    data_dir = "data/json"
    
    # 임베더 초기화
    embedder = DataEmbedder(pinecone_api_key=pinecone_api_key)
    
    # 배지 데이터 처리
    embedder.upsert_all(os.path.join(data_dir, "badge"), "badge")
    
    # 사용자 데이터 처리
    embedder.upsert_all(os.path.join(data_dir, "user"), "user")

if __name__ == "__main__":
    main()