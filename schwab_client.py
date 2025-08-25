# brokers/schwab_client.py
from __future__ import annotations
import enum
import pandas as pd
from io import StringIO
from typing import Optional

class LoadMode(enum.Enum):
    CSV = "CSV upload"
    API = "Schwab API"

class SchwabClient:
    """
    Minimal broker abstraction so you can swap CSV -> API later
    without touching the Streamlit UI.
    """
    def __init__(self, load_mode: str = LoadMode.CSV.value):
        self.mode = load_mode
        self._positions: Optional[pd.DataFrame] = None
        self._transactions: Optional[pd.DataFrame] = None
        self._accounts: Optional[pd.DataFrame] = None

    # ------------- Mode helpers -------------
    @property
    def mode_is_csv(self) -> bool:
        return self.mode == LoadMode.CSV.value

    # ------------- CSV loaders -------------
    def load_from_csv_files(self, positions_file, transactions_file, accounts_file=None):
        """
        Expect Schwab-exported CSVs (you can map headers here).
        We'll use lenient parsers so any reasonable CSV works.
        """
        if positions_file:
            self._positions = self._parse_positions_csv(positions_file.read().decode("utf-8"))
        if transactions_file:
            self._transactions = self._parse_transactions_csv(transactions_file.read().decode("utf-8"))
        if accounts_file:
            self._accounts = self._parse_accounts_csv(accounts_file.read().decode("utf-8"))

        # compute market_value if missing
        if self._positions is not None:
            df = self._positions
            if "market_value" not in df.columns:
                if {"quantity","last_price"}.issubset(df.columns):
                    df["market_value"] = df["quantity"] * df["last_price"]
                elif {"quantity","price"}.issubset(df.columns):
                    df["market_value"] = df["quantity"] * df["price"]

            # unify common column names
            rename = {
                "Symbol": "symbol",
                "Quantity": "quantity",
                "Qty": "quantity",
                "Last Price": "last_price",
                "Price": "price",
                "Cost Basis": "cost_basis_total",
                "Cost Basis Total": "cost_basis_total",
                "Market Value": "market_value",
                "Account": "account",
                "Asset Class": "asset_class",
            }
            self._positions = df.rename(columns=rename)
            # ensure numeric
            for c in ["quantity","last_price","price","market_value","cost_basis_total"]:
                if c in self._positions.columns:
                    self._positions[c] = pd.to_numeric(self._positions[c], errors="coerce")

        if self._transactions is not None:
            if "date" in self._transactions.columns:
                self._transactions["date"] = pd.to_datetime(self._transactions["date"], errors="coerce")

    def _parse_positions_csv(self, text: str) -> pd.DataFrame:
        df = pd.read_csv(StringIO(text))
        return df

    def _parse_transactions_csv(self, text: str) -> pd.DataFrame:
        df = pd.read_csv(StringIO(text))
        # unify likely columns
        rename = {
            "Symbol":"symbol","Action":"action","Quantity":"quantity","Price":"price",
            "Amount":"amount","Date":"date","Description":"description"
        }
        df = df.rename(columns=rename)
        for c in ["quantity","price","amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def _parse_accounts_csv(self, text: str) -> pd.DataFrame:
        df = pd.read_csv(StringIO(text))
        rename = {"Cash":"cash","Account Number":"account_number","Account":"account"}
        df = df.rename(columns=rename)
        for c in ["cash"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # ------------- API loader (stub) -------------
    def load_from_api(self):
        """
        Placeholder for Schwab Trader API integration.
        Steps you’ll implement later:
          - OAuth Authorization Code + PKCE
          - Store refresh/access tokens securely
          - GET /accounts, /positions, /orders, etc.
        For now we just noop.
        """
        # Example (pseudo):
        # self._accounts = pd.DataFrame(api.get_accounts())
        # self._positions = pd.DataFrame(api.get_positions())
        # self._transactions = pd.DataFrame(api.get_transactions())
        return

    # ------------- getters -------------
    def get_positions(self) -> Optional[pd.DataFrame]:
        return self._positions

    def get_transactions(self) -> Optional[pd.DataFrame]:
        return self._transactions

    def get_accounts(self) -> Optional[pd.DataFrame]:
        return self._accounts
