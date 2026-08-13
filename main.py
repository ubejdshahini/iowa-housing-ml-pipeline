from __future__ import annotations

import hmac
import io
import logging
import os
import re
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iowa_model.pkl"
FEATURES_PATH = BASE_DIR / "iowa_features.pkl"
MAX_CSV_SIZE_BYTES = 5 * 1024 * 1024

EXPECTED_FEATURES = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "HouseAge",
    "TotalFlrSF",
]

DEFAULT_DATABRICKS_TABLE = "workspace.default.iowa_preditions"
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

load_dotenv(BASE_DIR / ".env")


def _load_artifacts() -> tuple[object, list[str]]:
    """Load and validate the trusted model artifacts stored with the project."""
    if not MODEL_PATH.is_file() or not FEATURES_PATH.is_file():
        raise RuntimeError(
            "Model artifacts are missing. Expected iowa_model.pkl and "
            "iowa_features.pkl in the project root."
        )

    loaded_model = joblib.load(MODEL_PATH)
    loaded_features = list(joblib.load(FEATURES_PATH))

    if loaded_features != EXPECTED_FEATURES:
        raise RuntimeError(
            "The feature artifact does not match the required 9-feature contract. "
            f"Expected {EXPECTED_FEATURES}, received {loaded_features}."
        )

    if not hasattr(loaded_model, "predict"):
        raise RuntimeError("The loaded model artifact does not provide predict().")

    return loaded_model, loaded_features


model, features = _load_artifacts()

app = FastAPI(
    title="Iowa Housing Price API",
    description="Predict Iowa house prices and optionally save predictions to Databricks.",
    version="1.0.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
    """Require the API key configured in the local environment."""
    configured_key = os.getenv("API_KEY")
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY is not configured on the server.",
        )

    if key is None or not hmac.compare_digest(key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


class PredictionRow(BaseModel):
    """One predicted house row accepted by the Databricks save endpoint."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    LotArea: float = Field(ge=0)
    YearBuilt: int = Field(ge=1800)
    FirstFlrSF: float = Field(alias="1stFlrSF", ge=0)
    SecondFlrSF: float = Field(alias="2ndFlrSF", ge=0)
    FullBath: int = Field(ge=0)
    BedroomAbvGr: int = Field(ge=0)
    TotRmsAbvGrd: int = Field(ge=0)
    HouseAge: int = Field(ge=0)
    TotalFlrSF: float = Field(ge=0)
    predicted_price: float = Field(ge=0)


@app.get("/", tags=["health"])
def health() -> dict[str, object]:
    """Return a lightweight health check without requiring authentication."""
    return {
        "status": "ok",
        "model": type(model).__name__,
        "feature_count": len(features),
    }


def _validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate CSV structure and return numeric features in model order."""
    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded CSV contains no data rows.",
        )

    missing_columns = [feature for feature in features if feature not in frame.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "The CSV is missing required feature columns.",
                "missing_columns": missing_columns,
                "required_columns": features,
            },
        )

    prediction_frame = frame.loc[:, features].copy()
    for column in features:
        prediction_frame[column] = pd.to_numeric(
            prediction_frame[column], errors="coerce"
        )

    invalid_columns = prediction_frame.columns[
        prediction_frame.isna().any()
    ].tolist()
    if invalid_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Feature columns must contain numeric, non-empty values.",
                "invalid_columns": invalid_columns,
            },
        )

    return prediction_frame


@app.post(
    "/predict-csv",
    dependencies=[Depends(verify_api_key)],
    tags=["predictions"],
)
async def predict_csv(file: Annotated[UploadFile, File(...)]) -> list[dict[str, object]]:
    """Predict prices for all rows in an uploaded CSV file."""
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a .csv file.",
        )

    contents = await file.read(MAX_CSV_SIZE_BYTES + 1)
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )
    if len(contents) > MAX_CSV_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The CSV exceeds the 5 MB upload limit.",
        )

    try:
        uploaded_frame = pd.read_csv(io.BytesIO(contents))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file is not a valid CSV.",
        ) from exc

    prediction_frame = _validate_prediction_frame(uploaded_frame)

    try:
        prediction_frame["predicted_price"] = model.predict(prediction_frame)
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model could not generate predictions.",
        ) from exc

    return prediction_frame.to_dict(orient="records")


def _databricks_settings() -> tuple[str, str, str, str]:
    """Return validated Databricks connection settings."""
    variable_names = (
        "DATABRICKS_SERVER_HOSTNAME",
        "DATABRICKS_HTTP_PATH",
        "DATABRICKS_TOKEN",
    )
    missing = [name for name in variable_names if not os.getenv(name)]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Databricks is not configured. Missing: {', '.join(missing)}.",
        )

    table_name = os.getenv("DATABRICKS_TABLE", DEFAULT_DATABRICKS_TABLE)
    if not TABLE_NAME_PATTERN.fullmatch(table_name):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DATABRICKS_TABLE is not a valid table identifier.",
        )

    return (
        os.environ["DATABRICKS_SERVER_HOSTNAME"],
        os.environ["DATABRICKS_HTTP_PATH"],
        os.environ["DATABRICKS_TOKEN"],
        table_name,
    )


@app.post(
    "/save-predictions",
    dependencies=[Depends(verify_api_key)],
    tags=["predictions"],
)
def save_predictions(rows: list[PredictionRow]) -> dict[str, int]:
    """Save previously predicted rows to the configured Databricks table."""
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one prediction row.",
        )

    server_hostname, http_path, access_token, table_name = _databricks_settings()

    try:
        from databricks import sql
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The Databricks SQL connector is not installed.",
        ) from exc

    connection = None
    cursor = None
    try:
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        )
        cursor = connection.cursor()

        insert_statement = f"""
            INSERT INTO {table_name}
            (
                lot_area,
                year_built,
                first_floor_sf,
                second_floor_sf,
                full_bath,
                bedroom_above_gr,
                total_rooms_above_grd,
                house_age,
                total_flr_sf,
                predicted_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for row in rows:
            cursor.execute(
                insert_statement,
                (
                    row.LotArea,
                    row.YearBuilt,
                    row.FirstFlrSF,
                    row.SecondFlrSF,
                    row.FullBath,
                    row.BedroomAbvGr,
                    row.TotRmsAbvGrd,
                    row.HouseAge,
                    row.TotalFlrSF,
                    row.predicted_price,
                ),
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Saving predictions to Databricks failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Predictions could not be saved to Databricks.",
        ) from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    return {"saved": len(rows)}
