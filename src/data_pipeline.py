"""Data pipeline for loading and preprocessing e-commerce data."""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from itertools import combinations

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MIN_SUPPORT,
    MIN_CONFIDENCE,
    MAX_BUNDLE_SIZE,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline for loading, cleaning, and preprocessing e-commerce data."""

    def __init__(self):
        """Initialize the data pipeline."""
        self.raw_data = None
        self.processed_data = None
        self.products = None
        self.transactions = None
        self.bundles = None

    def download_kaggle_data(self) -> bool:
        """
        Download e-commerce dataset from Kaggle.

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if data already exists
        if os.path.exists(RAW_DATA_PATH):
            logger.info(f"Dataset already exists at: {RAW_DATA_PATH}")
            logger.info("Skipping download and continuing with processing")
            return True

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            dataset_name = "carrie1/ecommerce-data"
            download_path = os.path.dirname(RAW_DATA_PATH)

            logger.info(f"Downloading dataset: {dataset_name}")
            api.dataset_download_files(dataset_name, path=download_path, unzip=True)
            logger.info(f"Dataset downloaded to: {download_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
            return False

    def load_raw_data(self, filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
        """
        Load raw data from CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            pd.DataFrame: Raw data
        """
        if not os.path.exists(filepath):
            logger.error(f"Data file not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading raw data from: {filepath}")
        self.raw_data = pd.read_csv(filepath, encoding="ISO-8859-1")
        logger.info(f"Loaded {len(self.raw_data)} records, {len(self.raw_data.columns)} columns")
        return self.raw_data

    def explore_data(self) -> Dict:
        """
        Explore and describe the dataset.

        Returns:
            dict: Statistics about the dataset
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw_data first.")

        stats = {
            "shape": self.raw_data.shape,
            "columns": self.raw_data.columns.tolist(),
            "dtypes": self.raw_data.dtypes.to_dict(),
            "missing_values": self.raw_data.isnull().sum().to_dict(),
            "duplicates": self.raw_data.duplicated().sum(),
            "date_range": (
                str(pd.to_datetime(self.raw_data["InvoiceDate"]).min()),
                str(pd.to_datetime(self.raw_data["InvoiceDate"]).max()),
            ),
            "unique_customers": self.raw_data["CustomerID"].nunique(),
            "unique_products": self.raw_data["Description"].nunique(),
        }

        logger.info(f"Dataset Statistics: {stats}")
        return stats

    def convert_csv_to_tsv(self, input_filepath: str = RAW_DATA_PATH, output_filepath: Optional[str] = None) -> bool:
        """
        Convert CSV file to TSV format with UTF-8 encoding.

        Args:
            input_filepath: Path to the input CSV file
            output_filepath: Path to save the TSV file. If None, replaces .csv with .tsv

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if output_filepath is None:
                output_filepath = input_filepath.replace(".csv", ".tsv")

            # Load CSV with original encoding
            logger.info(f"Loading CSV from: {input_filepath}")
            df = pd.read_csv(input_filepath, encoding="ISO-8859-1")

            # Save as TSV with UTF-8 encoding
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            df.to_csv(output_filepath, sep="\t", encoding="utf-8", index=False)
            logger.info(f"Successfully converted and saved TSV to: {output_filepath}")
            logger.info(f"TSV file contains {len(df)} records with {len(df.columns)} columns")

            return True
        except Exception as e:
            logger.error(f"Failed to convert CSV to TSV: {e}")
            return False

    def preprocess(self) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.

        Returns:
            pd.DataFrame: Processed data
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw_data first.")

        logger.info("Starting data preprocessing...")
        df = self.raw_data.copy()

        # Remove rows with missing CustomerID
        initial_rows = len(df)
        df = df.dropna(subset=["CustomerID"])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing CustomerID")

        # Remove rows with missing Description
        initial_rows = len(df)
        df = df.dropna(subset=["Description"])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing Description")

        # Remove cancellations (negative quantities)
        df = df[df["Quantity"] > 0]
        logger.info("Removed cancellations (negative quantities)")

        # Remove rows with zero or negative prices
        df = df[df["UnitPrice"] > 0]
        logger.info("Removed rows with zero or negative prices")

        # Convert InvoiceDate to datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Create transaction value
        df["TransactionValue"] = df["Quantity"] * df["UnitPrice"]

        # Standardize product descriptions (lowercase, strip)
        df["Description"] = df["Description"].str.strip().str.lower()

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=["InvoiceNo", "StockCode"])
        logger.info(f"Removed {initial_rows - len(df)} duplicate transactions")

        self.processed_data = df
        logger.info(f"Preprocessing complete. Final dataset: {len(df)} records")

        return df

    def create_transaction_baskets(self) -> pd.DataFrame:
        """
        Create transaction baskets (items purchased together per invoice).

        Returns:
            pd.DataFrame: Transaction baskets with product groups
        """
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess first.")

        logger.info("Creating transaction baskets...")

        # Group by InvoiceNo to get items in each transaction
        baskets = (
            self.processed_data.groupby("InvoiceNo")["Description"].apply(list).reset_index()
        )
        baskets.columns = ["InvoiceNo", "Items"]

        # Add transaction metadata
        invoice_data = self.processed_data.groupby("InvoiceNo").agg(
            {
                "CustomerID": "first",
                "InvoiceDate": "first",
                "TransactionValue": "sum",
                "Quantity": "sum",
            }
        )

        baskets = baskets.merge(invoice_data, left_on="InvoiceNo", right_index=True)

        # Filter baskets with at least 2 items (needed for bundles)
        baskets = baskets[baskets["Items"].apply(len) >= 2]

        logger.info(f"Created {len(baskets)} transaction baskets with multiple items")
        self.transactions = baskets

        return baskets

    def generate_product_bundles(
        self, min_support: float = MIN_SUPPORT, min_confidence: float = MIN_CONFIDENCE, max_size: int = MAX_BUNDLE_SIZE
    ) -> List[Tuple]:
        """
        Generate product bundles using frequent itemset analysis (Apriori-like approach).

        Args:
            min_support: Minimum support threshold (fraction of transactions)
            min_confidence: Minimum confidence threshold
            max_size: Maximum bundle size (can be reduced for speed)

        Returns:
            List of product bundles (tuples of products)
        """
        if self.transactions is None:
            raise ValueError("Transaction baskets not created. Call create_transaction_baskets first.")

        logger.info(
            f"Generating bundles (min_support={min_support}, min_confidence={min_confidence}, max_size={max_size})..."
        )

        # Calculate item frequencies
        item_counts = {}
        total_transactions = len(self.transactions)

        for items in self.transactions["Items"]:
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Filter frequent items
        frequent_items = {
            item: count
            for item, count in item_counts.items()
            if count / total_transactions >= min_support
        }

        logger.info(f"Found {len(frequent_items)} frequent items (support >= {min_support})")

        # Pre-compute transaction item sets for O(1) lookup performance
        transaction_sets = [set(items) for items in self.transactions["Items"]]
        support_threshold = min_support * total_transactions

        # Generate itemsets of increasing size
        bundles = []
        current_itemsets = [[item] for item in frequent_items.keys()]

        for size in range(2, max_size + 1):
            # Generate candidate itemsets using set for O(1) duplicate checking (instead of O(n) list lookup)
            candidates_set = set()
            for i in range(len(current_itemsets)):
                for j in range(i + 1, len(current_itemsets)):
                    union = sorted(list(set(current_itemsets[i]) | set(current_itemsets[j])))
                    if len(union) == size:
                        candidates_set.add(tuple(union))  # Use set for fast O(1) lookup

            if not candidates_set:
                break
            
            logger.info(f"Created candidates_set with {len(candidates_set)} candidates of size {size}")

            # Calculate support for candidates using pre-computed transaction sets
            valid_itemsets = []
            for candidate in candidates_set:
                candidate_set = set(candidate)
                support = sum(1 for trans_set in transaction_sets if candidate_set.issubset(trans_set))
                if support >= support_threshold:
                    valid_itemsets.append(tuple(candidate))
                    bundles.append(tuple(candidate))

            current_itemsets = [list(itemset) for itemset in valid_itemsets]

            if not current_itemsets:
                break

            logger.info(f"Generated {len(valid_itemsets)} bundles of size {size}")

        self.bundles = bundles
        logger.info(f"Total bundles generated: {len(bundles)}")

        return bundles

    def save_processed_data(self, filepath: str = PROCESSED_DATA_PATH) -> bool:
        """
        Save processed data to pickle file.

        Args:
            filepath: Path to save the processed data

        Returns:
            bool: True if successful
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "processed_data": self.processed_data,
                        "transactions": self.transactions,
                        "bundles": self.bundles,
                        "timestamp": datetime.now(),
                    },
                    f,
                )
            logger.info(f"Processed data saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return False

    def load_processed_data(self, filepath: str = PROCESSED_DATA_PATH) -> bool:
        """
        Load processed data from pickle file.

        Args:
            filepath: Path to load the processed data from

        Returns:
            bool: True if successful
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.processed_data = data["processed_data"]
            self.transactions = data["transactions"]
            self.bundles = data["bundles"]
            logger.info(f"Processed data loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return False

    def get_bundle_statistics(self) -> Dict:
        """
        Get statistics about generated bundles.

        Returns:
            dict: Bundle statistics
        """
        if self.bundles is None:
            return {}

        bundle_sizes = [len(bundle) for bundle in self.bundles]

        stats = {
            "total_bundles": len(self.bundles),
            "min_bundle_size": min(bundle_sizes),
            "max_bundle_size": max(bundle_sizes),
            "avg_bundle_size": np.mean(bundle_sizes),
            "bundle_size_distribution": pd.Series(bundle_sizes).value_counts().to_dict(),
            "top_10_bundles": self.bundles[:10],
        }

        return stats
