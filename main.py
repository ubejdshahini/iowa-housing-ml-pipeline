import io
import os

import joblib
import pandas as pd

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Security,
    HTTPException,
    status,
    Depends,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
from databricks import sql



# Environment variables

load_dotenv()


# FastAPI app

app = FastAPI(
    title="Iowa Housing Price Prediction API",
    description="Predict Iowa house prices using a trained Random Forest model.",
    version="2.0",
)


# 
# Load model and feature list
# 

model = joblib.load("iowa_model.pkl")
features = joblib.load("iowa_features.pkl")


print("Model loaded successfully")
print("Expected features:", features)



# API key authentication

API_KEY = os.environ["API_KEY"]

api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(key: str = Security(api_key_header)):

    if key != API_KEY:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return key



# Health endpoint


@app.get("/")
def health():

    return {
        "status": "ok",
        "model": "Iowa Housing Random Forest v2",
        "features": features,
    }



# Endpoint 1
#
# CSV -> model predictions
# Does NOT save anything to Databricks.


@app.post(
    "/predict-csv",
    dependencies=[Depends(verify_api_key)],
)
async def predict_csv(file: UploadFile = File(...)):

    # Read uploaded CSV
    contents = await file.read()

    df = pd.read_csv(io.BytesIO(contents))

    
    # Check that all required model features exist
    

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_features}",
        )

    
    # Keep EXACT feature order used during training
    

    prediction_df = df[features].copy()

    
    # Predict
    

    predictions = model.predict(prediction_df)

    # Add predictions to result
    result = prediction_df.copy()

    result["predicted_price"] = predictions

    return result.to_dict(orient="records")



# Pydantic structure used when saving predictions


class PredictionRow(BaseModel):

    LotArea: float

    YearBuilt: int

    FirstFlrSF: float = Field(
        alias="1stFlrSF"
    )

    SecondFlrSF: float = Field(
        alias="2ndFlrSF"
    )

    FullBath: int

    BedroomAbvGr: int

    TotRmsAbvGrd: int

    HouseAge: int

    TotalFlrSF: float

    predicted_price: float

    model_config = ConfigDict(
        populate_by_name=True
    )



# Endpoint 2
#
# Save prediction rows to Databricks


@app.post(
    "/save-predictions",
    dependencies=[Depends(verify_api_key)],
)
def save_predictions(rows: list[PredictionRow]):

    connection = sql.connect(

        server_hostname=os.environ[
            "DATABRICKS_SERVER_HOSTNAME"
        ],

        http_path=os.environ[
            "DATABRICKS_HTTP_PATH"
        ],

        access_token=os.environ[
            "DATABRICKS_TOKEN"
        ],
    )

    cursor = connection.cursor()

    try:

        for row in rows:

            # Convert Pydantic model back to original names:
            #
            # FirstFlrSF -> 1stFlrSF
            # SecondFlrSF -> 2ndFlrSF

            d = row.model_dump(
                by_alias=True
            )

            cursor.execute(
                """
                INSERT INTO workspace.default.iowa_preditions
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
                """,
                (
                    d["LotArea"],
                    d["YearBuilt"],
                    d["1stFlrSF"],
                    d["2ndFlrSF"],
                    d["FullBath"],
                    d["BedroomAbvGr"],
                    d["TotRmsAbvGrd"],
                    d["HouseAge"],
                    d["TotalFlrSF"],
                    d["predicted_price"],
                ),
            )

    finally:

        cursor.close()

        connection.close()

    return {
        "saved": len(rows)
    }