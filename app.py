"""AI Engine microservice. Chạy: uvicorn app:app --port 8001"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from inference_week7 import Predictor

predictor = None


@asynccontextmanager
async def lifespan(_):
    global predictor
    predictor = Predictor()  # nạp LSTM-AE + Isolation Forest đúng 1 lần khi khởi động
    yield


app = FastAPI(title="Xiaomi Tier3 AI Engine", lifespan=lifespan)


class PredictRequest(BaseModel):
    user_id: str
    daily_history: list[dict]  # schema ngày: xem inference_spec_week7.md mục 2
    age: Optional[int] = None
    custom_thresholds: Optional[dict] = None  # Ngưỡng bác sĩ gán riêng cho hồ sơ (thay thế Tầng 1)


@app.post("/predict")
def predict(req: PredictRequest):
    # ponytail: def (không async) -> FastAPI chạy trong threadpool, không chặn event loop khi torch tính toán
    return predictor.predict(req.user_id, req.daily_history, req.age, req.custom_thresholds)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}
