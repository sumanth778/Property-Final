from unittest.mock import patch
import pandas as pd
from app import app


@patch("app.joblib.load")
@patch("app.pd.read_csv")
def test_home_route(mock_read_csv, mock_joblib):
    # Mock CSV
    mock_read_csv.return_value = pd.DataFrame({
        "location": ["NY", "CA"],
        "property_type": ["House", "Apartment"]
    })

    # Mock model
    mock_joblib.return_value = None

    tester = app.test_client()
    response = tester.get("/")

    assert response.status_code == 200
    assert b"<" in response.data
