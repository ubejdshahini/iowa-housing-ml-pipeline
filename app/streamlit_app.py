import streamlit as st
import requests
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Iowa House Price Predictor",
    page_icon="🏠",
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title("Iowa House Price Predictor")

st.write(
    """
    Upload a CSV containing Iowa house information.
    The trained machine learning model will predict
    the house price using the FastAPI backend.
    """
)


# =========================================================
# FASTAPI CONFIGURATION
# =========================================================

FASTAPI_URL = st.secrets["FASTAPI_URL"]

API_KEY = st.secrets["FASTAPI_API_KEY"]

HEADERS = {
    "X-API-Key": API_KEY
}


# =========================================================
# REQUIRED MODEL FEATURES
# =========================================================

FEATURES = [
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


# =========================================================
# FILE UPLOAD
# =========================================================

st.header("1. Upload house data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)


# =========================================================
# IF FILE WAS UPLOADED
# =========================================================

missing_features = []

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded data")

    st.dataframe(
        df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CHECK REQUIRED FEATURES
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        st.error(
            "The CSV is missing these required columns: "
            + ", ".join(missing_features)
        )

    else:

        st.success(
            "CSV contains all 9 required features."
        )

        # Show detected columns
        st.write("Required features:")

        st.code(
            "\n".join(FEATURES)
        )


# =========================================================
# PREDICTION
# =========================================================

if (
    uploaded_file is not None
    and not missing_features
):

    st.header("2. Generate predictions")

    if st.button(
        "Predict House Prices",
        type="primary"
    ):

        # Reset file position
        uploaded_file.seek(0)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "text/csv"
            )
        }

        try:

            with st.spinner(
                "Sending data to FastAPI..."
            ):

                response = requests.post(
                    f"{FASTAPI_URL}/predict-csv",
                    headers=HEADERS,
                    files=files,
                    timeout=120,
                )

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            if response.status_code == 200:

                predictions = response.json()

                # Store predictions
                st.session_state[
                    "predictions"
                ] = predictions

                st.success(
                    "Predictions generated successfully!"
                )

            # -------------------------------------------------
            # AUTH ERROR
            # -------------------------------------------------

            elif response.status_code == 401:

                st.error(
                    "Authentication failed. "
                    "Check your API key."
                )

            # -------------------------------------------------
            # BAD REQUEST
            # -------------------------------------------------

            elif response.status_code == 400:

                st.error(
                    f"Bad request:\n\n"
                    f"{response.text}"
                )

            # -------------------------------------------------
            # OTHER API ERROR
            # -------------------------------------------------

            else:

                st.error(
                    f"FastAPI returned "
                    f"HTTP {response.status_code}"
                )

                st.code(
                    response.text
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Make sure uvicorn is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The FastAPI request timed out."
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )


# =========================================================
# DISPLAY PREDICTIONS
# =========================================================

if "predictions" in st.session_state:

    st.header("3. Prediction Results")

    prediction_df = pd.DataFrame(
        st.session_state["predictions"]
    )

    st.dataframe(
        prediction_df,
        use_container_width=True
    )



    # -----------------------------------------------------
    # PRICE SUMMARY
    # -----------------------------------------------------

    if "predicted_price" in prediction_df.columns:

        st.subheader("Predicted Prices")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Average Predicted Price",
                f"${prediction_df['predicted_price'].mean():,.2f}"
            )

        with col2:

            st.metric(
                "Minimum Predicted Price",
                f"${prediction_df['predicted_price'].min():,.2f}"
            )

        with col3:

            st.metric(
                "Maximum Predicted Price",
                f"${prediction_df['predicted_price'].max():,.2f}"
            )



    # =====================================================
    # SAVE TO DATABRICKS
    # =====================================================

    st.header("4. Save predictions")

    st.write(
        "Save the generated predictions to Databricks."
    )

    if st.button(
        "💾 Save Predictions to Databricks"
    ):

        try:

            with st.spinner(
                "Saving predictions..."
            ):

                save_response = requests.post(
                    f"{FASTAPI_URL}/save-predictions",
                    headers=HEADERS,
                    json=st.session_state["predictions"],
                    timeout=120,
                )

            if save_response.status_code == 200:

                result = save_response.json()

                st.success(
                    f"Successfully saved "
                    f"{result['saved']} prediction(s) "
                    f"to Databricks."
                )

            elif save_response.status_code == 401:

                st.error(
                    "Authentication failed."
                )

            else:

                st.error(
                    f"Could not save predictions. "
                    f"HTTP {save_response.status_code}"
                )

                st.code(
                    save_response.text
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI."
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )


    csv = prediction_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="iowa_house_predictions.csv",
        mime="text/csv",
    )