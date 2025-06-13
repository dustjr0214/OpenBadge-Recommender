from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.web.route.recommendation import router as recommendation_router
import uvicorn

app = FastAPI(
    title="OpenBadge Recommendation API",
    description="오픈배지 추천 시스템 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:8000",
                   "https://openbadgesyu.web.app/",
                   "https://openbadgesyu.firebaseapp.com/"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendation_router)


@app.get("/")
async def root():
    """
    API 루트 엔드포인트
    """
    return {
        "message": "OpenBadge Recommendation API",
        "version": "1.0.0",
        "docs_url": "/docs"
    } 
if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
