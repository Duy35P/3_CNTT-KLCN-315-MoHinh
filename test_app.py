"""Smoke test: python test_app.py  (cần httpx: pip install httpx)"""
from datetime import date, timedelta

from fastapi.testclient import TestClient

from app import app


def day(i, hr=70):
    return {"date": str(date(2026, 1, 1) + timedelta(days=i)), "TotalSteps": 4000 + i * 10,
            "TotalMinutesAsleep": 400, "TotalTimeInBed": 440, "heart_rate_mean": hr,
            "heart_rate_min": 55, "heart_rate_max": 120, "resting_hr_p10": 60, "valid_day": True}


with TestClient(app) as c:  # with -> chạy lifespan, nạp model
    assert c.get("/health").json()["model_loaded"]

    r = c.post("/predict", json={"user_id": "u1", "daily_history": [day(i) for i in range(20)]}).json()
    assert r["status"] == "ok" and r["tier"] == 3 and r["ml"]["status"] == "ok", r

    r = c.post("/predict", json={"user_id": "u1", "daily_history": [day(i) for i in range(19)] + [day(19, hr=130)]}).json()
    assert r["alert"] == "CRITICAL", r  # luật cứng HR trung bình > 100
    print("OK", r["reason"])
