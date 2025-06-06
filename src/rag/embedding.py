import os
import json
import glob
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time
import pickle
import threading
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass

load_dotenv(verbose=True)

@dataclass
class EmbeddingConfig:
    """임베딩 관련 설정을 관리하는 클래스"""
    pinecone_api_key: str
    embedding_model: str = "multilingual-e5-large"
    index_name: str = "openbadges"
    cloud: str = "aws"
    region: str = "us-east-1"
    backup_retention_minutes: int = 30
    backup_dir: str = "backup/deleted_vectors"
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """환경변수에서 설정을 로드"""
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        return cls(
            pinecone_api_key=pinecone_api_key,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "multilingual-e5-large"),
            index_name=os.environ.get("PINECONE_INDEX_NAME", "openbadges"),
            cloud=os.environ.get("PINECONE_CLOUD", "aws"),
            region=os.environ.get("PINECONE_REGION", "us-east-1"),
            backup_retention_minutes=int(os.environ.get("BACKUP_RETENTION_MINUTES", "30")),
            backup_dir=os.environ.get("BACKUP_DIR", "backup/deleted_vectors")
        )

class PineconeManager:
    """Pinecone 연결 및 인덱스 관리 클래스"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self._index = None
        
    def get_index(self):
        """Pinecone 인덱스 반환 (lazy loading)"""
        if self._index is None:
            self._initialize_index()
            self._index = self.pc.Index(self.config.index_name)
        return self._index
    
    def _initialize_index(self):
        """Pinecone 인덱스 초기화"""
        spec = ServerlessSpec(cloud=self.config.cloud, region=self.config.region)
        
        if self.config.index_name not in self.pc.list_indexes().names():
            # 임베딩 차원을 얻기 위해 임시 임베딩 객체 생성
            temp_embeddings = PineconeEmbeddings(
                model=self.config.embedding_model,
                pinecone_api_key=self.config.pinecone_api_key
            )
            
            self.pc.create_index(
                name=self.config.index_name,
                dimension=temp_embeddings.dimension,
                metric="cosine",
                spec=spec
            )
            # 인덱스가 준비될 때까지 대기
            while not self.pc.describe_index(self.config.index_name).status['ready']:
                time.sleep(1)

class VectorBackupManager:
    """벡터 백업 및 복원 관리 클래스"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._memory_backup = {}
        
    def backup_vector(self, vector_id: str, namespace: str, vector_data: Dict[str, Any], 
                     method: str = "memory"):
        """벡터 백업"""
        if method == "file":
            self._backup_to_file(vector_id, namespace, vector_data)
        else:
            self._backup_to_memory(vector_id, namespace, vector_data)
    
    def _backup_to_file(self, vector_id: str, namespace: str, vector_data: Dict[str, Any]):
        """파일로 백업"""
        os.makedirs(self.config.backup_dir, exist_ok=True)
        
        backup_data = {
            'vector_id': vector_id,
            'namespace': namespace,
            'vector_data': vector_data,
            'deleted_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(minutes=self.config.backup_retention_minutes)).isoformat()
        }
        
        backup_file = os.path.join(self.config.backup_dir, f"{vector_id}.pkl")
        
        try:
            with open(backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            
            # 자동 삭제 타이머 설정
            timer = threading.Timer(
                self.config.backup_retention_minutes * 60, 
                self._cleanup_backup_file, 
                args=[backup_file]
            )
            timer.daemon = True
            timer.start()
            
            print(f"🗂️ 벡터 백업 완료: {backup_file}")
            
        except Exception as e:
            print(f"⚠️ 백업 실패: {str(e)}")
    
    def _backup_to_memory(self, vector_id: str, namespace: str, vector_data: Dict[str, Any]):
        """메모리로 백업"""
        backup_data = {
            'vector_id': vector_id,
            'namespace': namespace,
            'vector_data': vector_data,
            'deleted_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(minutes=self.config.backup_retention_minutes)
        }
        
        self._memory_backup[vector_id] = backup_data
        
        # 자동 삭제 타이머 설정
        timer = threading.Timer(
            self.config.backup_retention_minutes * 60, 
            self._cleanup_memory_backup, 
            args=[vector_id]
        )
        timer.daemon = True
        timer.start()
        
        print(f"💾 벡터 메모리 백업 완료: {vector_id}")
    
    def restore_vector(self, vector_id: str, pinecone_manager: PineconeManager) -> Tuple[bool, str]:
        """백업된 벡터 복원"""
        try:
            # 메모리 백업에서 확인
            if vector_id in self._memory_backup:
                backup_data = self._memory_backup[vector_id]
                if datetime.now() <= backup_data['expires_at']:
                    return self._restore_from_backup(backup_data, pinecone_manager, vector_id)
                else:
                    del self._memory_backup[vector_id]
                    return False, f"❌ 복원 기간이 만료되었습니다: {vector_id}"
            
            # 파일 백업에서 확인
            backup_file = os.path.join(self.config.backup_dir, f"{vector_id}.pkl")
            if os.path.exists(backup_file):
                with open(backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                
                expires_at = datetime.fromisoformat(backup_data['expires_at'])
                if datetime.now() <= expires_at:
                    result = self._restore_from_backup(backup_data, pinecone_manager, vector_id)
                    if result[0]:  # 성공시 백업 파일 삭제
                        os.remove(backup_file)
                    return result
                else:
                    os.remove(backup_file)
                    return False, f"❌ 복원 기간이 만료되었습니다: {vector_id}"
            
            return False, f"❌ 백업된 벡터를 찾을 수 없습니다: {vector_id}"
            
        except Exception as e:
            return False, f"❌ 벡터 복원 실패: {str(e)}"
    
    def _restore_from_backup(self, backup_data: Dict, pinecone_manager: PineconeManager, 
                           vector_id: str) -> Tuple[bool, str]:
        """백업 데이터에서 벡터 복원"""
        try:
            index = pinecone_manager.get_index()
            vector_data = backup_data['vector_data']
            
            index.upsert(
                vectors=[(
                    vector_data['id'],
                    vector_data['values'],
                    vector_data['metadata']
                )],
                namespace=backup_data['namespace']
            )
            
            # 메모리 백업에서 제거
            if vector_id in self._memory_backup:
                del self._memory_backup[vector_id]
            
            return True, f"✅ 벡터 복원 성공: {vector_id}"
            
        except Exception as e:
            return False, f"❌ 벡터 복원 중 오류: {str(e)}"
    
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
            if vector_id in self._memory_backup:
                del self._memory_backup[vector_id]
                print(f"🗑️ 메모리 백업 자동 삭제: {vector_id}")
        except Exception as e:
            print(f"⚠️ 메모리 백업 삭제 실패: {str(e)}")

class DataPreprocessor(ABC):
    """데이터 전처리를 위한 추상 기본 클래스"""
    
    @abstractmethod
    def get_required_fields(self) -> set:
        """각 데이터 타입별 필수 필드 반환"""
        pass
    
    @abstractmethod
    def build_text(self, data: Dict[str, Any]) -> str:
        """데이터를 임베딩을 위한 텍스트로 변환"""
        pass
    
    @abstractmethod
    def build_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 구성"""
        pass
    
    @abstractmethod
    def get_id(self, data: Dict[str, Any]) -> str:
        """ID 필드 추출"""
        pass
    
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """공통 전처리 로직"""
        return {
            "id": self.get_id(data),
            "text": self.build_text(data),
            "metadata": self.build_metadata(data)
        }

class BadgePreprocessor(DataPreprocessor):
    """배지 데이터 전처리 클래스"""
    
    def get_required_fields(self) -> set:
        return {'badge_id', 'name', 'issuer', 'criteria', 'skillsValidated'}
    
    def build_text(self, data: Dict[str, Any]) -> str:
        return f"""
        배지명: {data.get('name', '')}
        발급자: {data.get('issuer', '')}
        설명: {data.get('description', '')}
        기준: {data.get('criteria', '')}
        정렬: {data.get('alignment', '')}
        취업 결과: {data.get('employmentOutcome', '')}
        검증된 기술: {data.get('skillsValidated', '')}
        역량: {data.get('competency', '')}
        학습 기회: {data.get('learningOpportunity', '')}
        """
    
    def build_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": data.get('name'),
            "issuer": data.get('issuer'),
            "skills": data.get('skillsValidated'),
            "competency": data.get('competency'),
            "related_badges": data.get('related_badges')
        }
    
    def get_id(self, data: Dict[str, Any]) -> str:
        return data.get('badge_id')

class UserPreprocessor(DataPreprocessor):
    """사용자 데이터 전처리 클래스"""
    
    def get_required_fields(self) -> set:
        return {'user_id', 'goal', 'competency_level', 'learning_history'}
    
    def build_text(self, data: Dict[str, Any]) -> str:
        return f"""
        이름: {data.get('name', '')}
        목표: {data.get('goal', '')}
        기술: {data.get('skills', '')}
        역량 수준: {data.get('competency_level', '')}
        학습 이력: {data.get('learning_history', '')}
        취업 이력: {data.get('employment_history', '')}
        교육 수준: {data.get('education_level', '')}
        참여 지표: {data.get('engagement_metrics', '')}
        """
    
    def build_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": data.get('name'),
            "goal": data.get('goal'),
            "skills": data.get('skills'),
            "competency_level": data.get('competency_level'),
            "acquired_badges": data.get('acquired_badges'),
            "education_level": data.get('education_level')
        }
    
    def get_id(self, data: Dict[str, Any]) -> str:
        return data.get('user_id')

class DataTypeDetector:
    """데이터 타입 감지 클래스"""
    
    def __init__(self):
        self.badge_preprocessor = BadgePreprocessor()
        self.user_preprocessor = UserPreprocessor()
    
    def detect_data_type(self, data: Dict[str, Any], file_path: str = "") -> str:
        """데이터 타입을 감지"""
        data_keys = set(data.keys())
        
        badge_match = len(self.badge_preprocessor.get_required_fields().intersection(data_keys))
        user_match = len(self.user_preprocessor.get_required_fields().intersection(data_keys))
        
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
                raise ValueError(f"데이터 타입을 결정할 수 없습니다: {file_path}")

class DataEmbedder:
    """
    데이터 임베딩을 위한 메인 클래스 (기존 인터페이스 유지)
    실제로는 다른 클래스들을 조합하여 작동
    """
    
    def __init__(self, 
                 pinecone_api_key: Optional[str] = None,
                 embedding_model: str = "multilingual-e5-large",
                 index_name: str = "openbadges",
                 config: Optional[EmbeddingConfig] = None):
        """
        데이터 임베딩을 위한 클래스 초기화
        기존 인터페이스와 호환성을 위해 개별 파라미터도 지원
        """
        if config is None:
            if pinecone_api_key is None:
                config = EmbeddingConfig.from_env()
            else:
                config = EmbeddingConfig(
                    pinecone_api_key=pinecone_api_key,
                    embedding_model=embedding_model,
                    index_name=index_name
                )
        
        self.config = config
        self.pinecone_manager = PineconeManager(config)
        self.backup_manager = VectorBackupManager(config)
        self.type_detector = DataTypeDetector()
        
        # 임베딩 모델 초기화
        self.embeddings = PineconeEmbeddings(
            model=config.embedding_model,
            pinecone_api_key=config.pinecone_api_key
        )
        
        # 기존 속성들 (하위 호환성을 위해)
        self.embedding_model = config.embedding_model
        self.index_name = config.index_name
        self.pc = self.pinecone_manager.pc
        self._deleted_vectors_backup = self.backup_manager._memory_backup
    
    def _determine_namespace_from_id(self, vector_id: str) -> str:
        """벡터 ID를 기반으로 네임스페이스 결정"""
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
    
    def _determine_data_type(self, file_path: str) -> str:
        """JSON 파일을 분석하여 데이터 타입 결정"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.type_detector.detect_data_type(data, file_path)
        except Exception as e:
            raise ValueError(
                f"파일을 읽거나 분석하는 중 오류가 발생했습니다: {file_path}, "
                f"오류: {str(e)}"
            )
    
    def preprocess_badge(self, badge_data: Dict[str, Any]) -> Dict[str, Any]:
        """배지 데이터 전처리 (하위 호환성)"""
        return BadgePreprocessor().preprocess(badge_data)
    
    def preprocess_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 데이터 전처리 (하위 호환성)"""
        return UserPreprocessor().preprocess(user_data)
    
    def delete_vector(self, vector_id: str, backup_method: str = "memory") -> Tuple[bool, str]:
        """Pinecone에서 벡터를 삭제 (30분간 복원 가능하도록 백업)"""
        try:
            namespace = self._determine_namespace_from_id(vector_id)
            data_type = "배지" if namespace == "badge" else "사용자"
            
            index = self.pinecone_manager.get_index()
            try:
                existing_vector = index.fetch(ids=[vector_id], namespace=namespace)
                if (not existing_vector.vectors or 
                    vector_id not in existing_vector.vectors):
                    return False, (f"❌ 해당 {data_type}가 존재하지 않습니다: "
                                  f"{vector_id}")
                
                # 삭제 전 백업
                vector_data = existing_vector.vectors[vector_id]
                backup_data = {
                    'id': vector_id,
                    'values': vector_data.values,
                    'metadata': vector_data.metadata
                }
                
                self.backup_manager.backup_vector(
                    vector_id, namespace, backup_data, backup_method
                )
                
            except Exception as e:
                return False, f"❌ 벡터 확인 중 오류 발생: {str(e)}"
            
            # 벡터 삭제
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
    
    def restore_vector(self, vector_id: str) -> Tuple[bool, str]:
        """백업된 벡터를 복원"""
        return self.backup_manager.restore_vector(vector_id, self.pinecone_manager)
    
    def upsert_vector(self, file_path: str):
        """특정 JSON 파일을 처리하여 Pinecone에 벡터를 upsert"""
        try:
            # 파일 타입 확인
            data_type = self._determine_data_type(file_path)
            print(f"파일 타입 확인: {data_type}")
            
            # 파일 로드 및 전처리
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data_type == 'badge':
                processed_item = self.preprocess_badge(data)
            else:
                processed_item = self.preprocess_user(data)
            
            # 동일한 ID가 있는지 확인
            index = self.pinecone_manager.get_index()
            vector_id = processed_item["id"]
            
            try:
                existing_vector = index.fetch(ids=[vector_id], namespace=data_type)
                if existing_vector.vectors:
                    print(f"벡터 업데이트: ID {vector_id} (네임스페이스: {data_type})")
                else:
                    print(f"새 벡터 삽입: ID {vector_id} (네임스페이스: {data_type})")
            except Exception:
                print(f"새 벡터 삽입: ID {vector_id} (네임스페이스: {data_type})")
            
            # 벡터 임베딩 생성
            vector = self.embeddings.embed_query(processed_item["text"])
            
            # Pinecone에 upsert
            try:
                index.upsert(
                    vectors=[(vector_id, vector, processed_item["metadata"])],
                    namespace=data_type
                )
                
                # 성공 메시지 출력
                name = processed_item["metadata"].get("name", "이름 없음")
                print(f"✅ upsert 성공: {name} (ID: {vector_id}, 타입: {data_type})")
                
            except Exception as e:
                print(f"❌ upsert 실패: ID {vector_id}, 오류: {str(e)}")
                
        except Exception as e:
            print(f"❌ 파일 처리 실패: {file_path}, 오류: {str(e)}")
    
    def upsert_manually_all(self, data_dir: str, data_type: str):
        """데이터 수동 임베딩을 위해 특정 디렉토리의 모든 json 파일을 임베딩 후 저장"""
        if data_type not in ['badge', 'user']:
            raise ValueError("data_type은 'badge' 또는 'user'여야 합니다.")
        
        # 파일 패턴 설정
        file_pattern = f"{data_type}_*.json"
        data_files = glob.glob(os.path.join(data_dir, file_pattern))
        processed_data = []
        
        # 전처리기 선택
        if data_type == 'badge':
            preprocessor = BadgePreprocessor()
        else:
            preprocessor = UserPreprocessor()
        
        # 데이터 전처리
        for data_file in data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed_item = preprocessor.preprocess(data)
                processed_data.append(processed_item)
        
        # 텍스트 임베딩 생성 및 Pinecone에 저장
        texts = [item["text"] for item in processed_data]
        metadatas = [item["metadata"] for item in processed_data]
        ids = [item["id"] for item in processed_data]
        
        # Pinecone에 데이터 저장
        index = self.pinecone_manager.get_index()
        for text, metadata, id in zip(texts, metadatas, ids):
            vector = self.embeddings.embed_query(text)
            index.upsert(
                vectors=[(id, vector, metadata)],
                namespace=data_type
            )
            print(f"{data_type} 데이터 임베딩 생성 및 저장 완료: {id}")
        
        print(f"{data_type} 데이터 임베딩 생성 및 저장 완료: {len(texts)}개")

    # 기존 메소드들 유지 (하위 호환성)
    def _initialize_index(self):
        """기존 코드 호환성을 위한 메소드"""
        self.pinecone_manager._initialize_index()
    
    def _backup_vector_to_file(self, vector_id: str, namespace: str, vector_data: Dict[str, Any]):
        """기존 코드 호환성을 위한 메소드"""
        self.backup_manager._backup_to_file(vector_id, namespace, vector_data)
    
    def _backup_vector_to_memory(self, vector_id: str, namespace: str, vector_data: Dict[str, Any]):
        """기존 코드 호환성을 위한 메소드"""
        self.backup_manager._backup_to_memory(vector_id, namespace, vector_data)

def main():
    """
    /data 의 json 배지/유저 파일 임베딩
    """
    # 설정을 환경변수에서 로드
    config = EmbeddingConfig.from_env()
    
    # 데이터 디렉토리 경로
    data_dir = "data/json"
    
    # 임베더 초기화
    embedder = DataEmbedder(config=config)
    
    # 배지 데이터 처리
    embedder.upsert_manually_all(os.path.join(data_dir, "badge"), "badge")
    
    # 사용자 데이터 처리
    embedder.upsert_manually_all(os.path.join(data_dir, "user"), "user")

if __name__ == "__main__":
    main()