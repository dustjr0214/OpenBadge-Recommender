# OpenBadge-Recommender
------

RAG로 사용자 정보 기반 오픈배지를 추천하는 프로젝트입니다.


## 실행법 & 요구사항

- 버전 3.12 이상의 python이 필요합니다.
- 실제 실행 시 .env 파일이 필요합니다.



### 설치 방법

1. 저장소 클론
```powershell
git clone [repository-url]
cd openbadge-recommender
```

2. 가상환경 설정
```powershell
# 가상환경 생성 (.venv는 예시이며, 원하는 이름으로 변경 가능)
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate
```

3. 의존성 설치
```powershell
pip install -r requirements.txt
```

### 실행 방법

1. FastAPI 서버 실행
```powershell
uvicorn src.web.main:app --reload
```

2. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc