# Python 3.12 slim 이미지 사용 (용량 최적화)
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# requirements.txt 먼저 복사 (도커 레이어 캐싱 최적화)
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 전체 소스 코드 복사
COPY . .

# 포트 8000 노출
EXPOSE 8000

# 애플리케이션 실행 명령어
# README에서 확인한 명령어: uvicorn src.web.main:app
# 프로덕션 환경을 위해 --reload 제거, 호스트와 포트 설정
CMD ["uvicorn", "src.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
