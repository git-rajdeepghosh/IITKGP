from fastapi import APIRouter, HTTPException, Request
from typing import List
from pydantic import BaseModel, Field

# =========================================================
# 1. CONSTANTS & METADATA
# =========================================================

SENSOR_METADATA = {
    "s_1": {"name": "Fan Inlet Temperature", "unit": "°R"},
    "s_2": {"name": "LPC Outlet Temperature", "unit": "°R"},
    "s_3": {"name": "HPC Outlet Temperature", "unit": "°R"},
    "s_4": {"name": "LPT Outlet Temperature", "unit": "°R"},
    "s_5": {"name": "Fan Inlet Pressure", "unit": "psia"},
    "s_6": {"name": "Bypass Duct Pressure", "unit": "psia"},
    "s_7": {"name": "HPC Outlet Pressure", "unit": "psia"},
    "s_8": {"name": "Physical Fan Speed", "unit": "rpm"},
    "s_9": {"name": "Physical Core Speed", "unit": "rpm"},
    "s_10": {"name": "Engine Pressure Ratio", "unit": "-"},
    "s_11": {"name": "HPC Outlet Static Pressure", "unit": "psia"},
    "s_12": {"name": "Fuel Flow Ratio", "unit": "pps/psi"},
    "s_13": {"name": "Corrected Fan Speed", "unit": "rpm"},
    "s_14": {"name": "Corrected Core Speed", "unit": "rpm"},
    "s_15": {"name": "Bypass Ratio", "unit": "-"},
    "s_16": {"name": "Burner Fuel-Air Ratio", "unit": "-"},
    "s_17": {"name": "Bleed Enthalpy", "unit": "-"},
    "s_18": {"name": "Demanded Fan Speed", "unit": "rpm"},
    "s_19": {"name": "Demanded Corrected Fan Speed", "unit": "rpm"},
    "s_20": {"name": "HPT Coolant Bleed", "unit": "lbm/s"},
    "s_21": {"name": "LPT Coolant Bleed", "unit": "lbm/s"},
}

# =========================================================
# 2. SCHEMAS
# =========================================================

class SensorSeries(BaseModel):
    sensor_id: str
    name: str
    unit: str
    data: List[float]

class SensorTrendResponse(BaseModel):
    unit_id: int
    cycles: List[int]
    rul_trend: List[int]
    sensors: List[SensorSeries]

class PredictionResponse(BaseModel):
    unit_id: int
    current_cycle: int
    predicted_rul: int
    health_score: float = Field(..., ge=0, le=100)
    risk_level: str
    confidence: float

class EngineMetadata(BaseModel):
    unit_id: int
    max_cycles: int
    status: str

# =========================================================
# 3. ROUTER
# =========================================================

router = APIRouter()

FAKE_DB_ENGINES = {
    1: {"max_cycles": 192, "status": "Active"},
    2: {"max_cycles": 250, "status": "Retired"},
    5: {"max_cycles": 150, "status": "Active"},
}

# =========================================================
# 4. ENDPOINTS
# =========================================================

@router.get("/engines", response_model=List[EngineMetadata], tags=["Inventory"])
async def list_engines():
    return [
        {"unit_id": k, "max_cycles": v["max_cycles"], "status": v["status"]}
        for k, v in FAKE_DB_ENGINES.items()
    ]


@router.get("/engines/{unit_id}/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rul(unit_id: int, request: Request):
    engine = FAKE_DB_ENGINES.get(unit_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Engine not found")

    if engine["status"] == "Retired":
        raise HTTPException(status_code=422, detail="Engine is retired")

    service = request.app.state.inference_service
    result = await service.predict(engine["max_cycles"])

    return {
        "unit_id": unit_id,
        "current_cycle": engine["max_cycles"] - result["predicted_rul"],
        **result
    }


@router.get("/engines/{unit_id}/trends", response_model=SensorTrendResponse, tags=["Analytics"])
async def get_trends(unit_id: int):
    if unit_id not in FAKE_DB_ENGINES:
        raise HTTPException(status_code=404, detail="Engine not found")

    cycles = [100, 101, 102, 103, 104]
    critical_sensors = ["s_2", "s_11", "s_12", "s_14"]

    sensors = []
    for sid in critical_sensors:
        meta = SENSOR_METADATA[sid]
        sensors.append({
            "sensor_id": sid,
            "name": meta["name"],
            "unit": meta["unit"],
            "data": [1400 + i * 2 for i in range(len(cycles))]
        })

    return {
        "unit_id": unit_id,
        "cycles": cycles,
        "rul_trend": [50, 49, 48, 47, 45],
        "sensors": sensors
    }
