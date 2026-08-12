import os
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

client = TestClient(main.app)

class DummyFREDClient:
    def __init__(self, api_key=None, cache_dir=None):
        assert api_key == "TESTKEY"
        self.api_key = api_key

    def macro_features_for_dates(self, dates):
        macro = pd.DataFrame({"macro_gdp": [1.0] * len(dates)}, index=dates.index)
        return macro


def dummy_attach_macro_features(df, fred_client=None, date_col=None):
    macro = pd.DataFrame({"macro_gdp": [1.0] * len(df)}, index=df.index)
    df_with_macro = pd.concat([df.copy(), macro], axis=1)
    return df_with_macro, ["macro_gdp"]


def test_macro_fetch_requires_server_fred_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    csv_content = "origination_date,amount\n2020-01-15,100\n"
    response = client.post(
        "/data/macro/fetch",
        data={"date_col": "origination_date"},
        files={"file": ("test.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 500
    assert response.json()["detail"].startswith(
        "FRED macroeconomic integration is not configured on the server",
    )


def test_macro_fetch_uses_fred_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "TESTKEY")
    monkeypatch.setattr(main.fred_client, "FREDClient", DummyFREDClient)
    monkeypatch.setattr(main.fred_client, "attach_macro_features", dummy_attach_macro_features)

    csv_content = "origination_date,amount\n2020-01-15,100\n2020-02-15,200\n"
    response = client.post(
        "/data/macro/fetch",
        data={"date_col": "origination_date"},
        files={"file": ("test.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["macro_columns"] == ["macro_gdp"]
    assert "csv_with_macro" in payload
    assert "macro_gdp" in payload["csv_with_macro"]


def test_integration_run_applies_macro_using_server_api_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "TESTKEY")
    monkeypatch.setattr(main.fred_client, "FREDClient", DummyFREDClient)
    monkeypatch.setattr(main.fred_client, "attach_macro_features", dummy_attach_macro_features)

    csv_content = "origination_date,amount\n2020-01-15,100\n2020-02-15,200\n"
    response = client.post(
        "/data/integration/run",
        data={
            "fetch_macro": "true",
            "macro_date_col": "origination_date",
        },
        files={"customer_file": ("customer.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "csv_text" in payload
    assert "macro_gdp" in payload["csv_text"]
    assert payload["integration_report"]["warnings"] == []
