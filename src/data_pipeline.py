"""Data pipeline for loading and preprocessing e-commerce data."""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
from collections import Counter

from src.config import (
    PipelineConfig,
    load_config,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    PERPLEXITY_API_KEY,
    PERPLEXITY_BASE_URL,
)
from src.llm_client import LLMConfig, LLMClient
from src.utils import (
    enrich_categories_batch_with_llm,
    extract_contexts_with_llm,
    compute_iqr_bounds,
    batch_score_anomalies_with_llm,
    LLMQuotaExceededError,
    load_json_file,
    save_json_file,
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline for loading, cleaning, and preprocessing e-commerce data."""

    def __init__(self, force_reprocess: bool = False, config: Optional[PipelineConfig] = None):
        """Initialize the data pipeline.
        
        Args:
            force_reprocess: If True, bypass all LLM caches and reprocess from scratch
            config: Optional PipelineConfig object. If None, loads config from environment via load_config()
        """
        self.raw_data = None
        self.processed_data = None
        self.products = None
        self.transactions = None
        self.bundles = None
        self.force_reprocess = force_reprocess
        self._config_injected = config is not None
        
        # Load configuration
        if config is None:
            pipeline_config, _, _, _, _ = load_config()
            self.config = pipeline_config
        else:
            self.config = config

    def _get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get the correct API key for the specified LLM provider.
        
        Args:
            provider: LLM provider name ('openai', 'azure', 'gemini', 'anthropic', 'perplexity')
        
        Returns:
            Optional[str]: API key for the provider, or None if not configured
        """
        provider_lower = provider.lower()
        if not self._config_injected:
            if provider_lower == "openai":
                return OPENAI_API_KEY
            elif provider_lower == "azure":
                return AZURE_OPENAI_API_KEY
            elif provider_lower == "gemini":
                return GEMINI_API_KEY
            elif provider_lower == "anthropic":
                return ANTHROPIC_API_KEY
            elif provider_lower == "perplexity":
                return PERPLEXITY_API_KEY
            return None

        llm_config = self.config.llm_config
        if provider_lower == "openai":
            return llm_config.openai_api_key
        elif provider_lower == "azure":
            return llm_config.azure_api_key
        elif provider_lower == "gemini":
            return llm_config.gemini_api_key
        elif provider_lower == "anthropic":
            return llm_config.anthropic_api_key
        elif provider_lower == "perplexity":
            return llm_config.perplexity_api_key
        return None

    def _get_base_url_for_provider(self, provider: str) -> Optional[str]:
        """Get the base URL for the specified LLM provider (if applicable).
        
        Args:
            provider: LLM provider name
        
        Returns:
            Optional[str]: Base URL for the provider, or None if not applicable
        """
        if provider.lower() == "perplexity":
            if not self._config_injected:
                return PERPLEXITY_BASE_URL
            return self.config.llm_config.perplexity_base_url
        return None

    def _build_pipeline_llm_config(self) -> LLMConfig:
        """Build LLMConfig from injected PipelineConfig settings.
        
        Centralizes provider-specific credential mapping for all LLM functions
        in the data pipeline, eliminating duplicate config creation across methods.
        
        Returns:
            LLMConfig: Configured LLM client configuration object
        """
        if not self._config_injected:
            provider = LLM_PROVIDER.lower() if LLM_PROVIDER else "openai"
            return LLMConfig(
                provider=LLM_PROVIDER,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
                openai_api_key=OPENAI_API_KEY if provider == "openai" else None,
                azure_api_key=AZURE_OPENAI_API_KEY if provider == "azure" else None,
                azure_endpoint=AZURE_OPENAI_ENDPOINT if provider == "azure" else None,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT if provider == "azure" else None,
                azure_api_version=AZURE_OPENAI_API_VERSION if provider == "azure" else None,
                gemini_api_key=GEMINI_API_KEY if provider == "gemini" else None,
                anthropic_api_key=ANTHROPIC_API_KEY if provider == "anthropic" else None,
                perplexity_api_key=PERPLEXITY_API_KEY if provider == "perplexity" else None,
                perplexity_base_url=PERPLEXITY_BASE_URL if provider == "perplexity" else None,
            )

        llm_config = self.config.llm_config
        provider = llm_config.provider.lower() if llm_config.provider else "openai"
        return LLMConfig(
            provider=llm_config.provider,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout_seconds=llm_config.timeout_seconds,
            openai_api_key=llm_config.openai_api_key if provider == "openai" else None,
            azure_api_key=llm_config.azure_api_key if provider == "azure" else None,
            azure_endpoint=llm_config.azure_endpoint if provider == "azure" else None,
            azure_deployment=llm_config.azure_deployment if provider == "azure" else None,
            azure_api_version=llm_config.azure_api_version if provider == "azure" else None,
            gemini_api_key=llm_config.gemini_api_key if provider == "gemini" else None,
            anthropic_api_key=llm_config.anthropic_api_key if provider == "anthropic" else None,
            perplexity_api_key=llm_config.perplexity_api_key if provider == "perplexity" else None,
            perplexity_base_url=llm_config.perplexity_base_url if provider == "perplexity" else None,
        )

    def _handle_llm_quota_error_standardized(self, exc: Exception, fallback_strategy: str) -> None:
        """Handle LLM quota exceeded errors with standardized behavior.
        
        Centralizes error handling for quota/rate limit errors, providing
        consistent logging and fallback strategy tracking across all LLM methods.
        
        Args:
            exc: The exception that occurred
            fallback_strategy: Strategy name to use ('fill_nan', 'use_heuristic', 'skip_batch')
        
        Raises:
            LLMQuotaExceededError: Re-raises the quota error for caller to handle
        """
        logger.warning(
            f"LLM quota exceeded or rate limited: {exc}. "
            f"Using fallback strategy: {fallback_strategy}"
        )
        # Re-raise to let calling method implement strategy-specific handling
        raise LLMQuotaExceededError(str(exc)) from exc

    def _validate_and_load_cache(self, cache_path: str, enabled: bool, force_reprocess: bool) -> Dict[str, Any]:
        """Load and validate cache with unified logic for all pipeline methods.
        
        Consolidates cache loading validation logic used by _enrich_categories,
        _flag_anomalies, and _extract_contexts methods.
        
        Args:
            cache_path: Path to the cache JSON file
            enabled: Whether caching is enabled for this operation
            force_reprocess: Whether to bypass cache and reprocess from scratch
        
        Returns:
            Dict[str, Any]: Loaded cache dictionary, empty dict if cache disabled or not found
        """
        if not enabled or force_reprocess:
            return {}
        
        cache = load_json_file(cache_path)
        if cache:
            logger.info(f"Loaded cache from: {cache_path} ({len(cache)} entries)")
        return cache

    def download_kaggle_data(self) -> bool:
        """
        Download e-commerce dataset from Kaggle.

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if data already exists
        raw_data_path = self.config.raw_data_path
        if os.path.exists(raw_data_path):
            logger.info(f"Dataset already exists at: {raw_data_path}")
            logger.info("Skipping download and continuing with processing")
            return True

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            dataset_name = "carrie1/ecommerce-data"
            download_path = os.path.dirname(raw_data_path)

            logger.info(f"Downloading dataset: {dataset_name}")
            api.dataset_download_files(dataset_name, path=download_path, unzip=True)
            logger.info(f"Dataset downloaded to: {download_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
            return False

    def load_raw_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            pd.DataFrame: Raw data
        """
        data_filepath = filepath or self.config.raw_data_path

        if not os.path.exists(data_filepath):
            logger.error(f"Data file not found: {data_filepath}")
            raise FileNotFoundError(f"Data file not found: {data_filepath}")

        logger.info(f"Loading raw data from: {data_filepath}")
        self.raw_data = pd.read_csv(
            data_filepath,
            encoding="ISO-8859-1",
            dtype={"InvoiceNo": str, "StockCode": str},
        )
        logger.info(f"Loaded {len(self.raw_data)} records, {len(self.raw_data.columns)} columns")
        return self.raw_data

    def explore_data(self, data_type: str = "auto") -> Dict:
        """
        Explore and describe the dataset (raw or preprocessed).

        Args:
            data_type: Type of data to analyze. Options:
                - "auto": Use processed_data if available, otherwise raw_data
                - "raw": Force analysis of raw_data
                - "processed": Force analysis of processed_data

        Returns:
            dict: Statistics about the dataset

        Raises:
            ValueError: If requested data type is not available
        """
        # Determine which dataset to analyze
        if data_type == "auto":
            df = self.processed_data if self.processed_data is not None else self.raw_data
            stage = "processed" if self.processed_data is not None else "raw"
        elif data_type == "raw":
            df = self.raw_data
            stage = "raw"
        elif data_type == "processed":
            df = self.processed_data
            stage = "processed"
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Use 'auto', 'raw', or 'processed'")

        if df is None:
            raise ValueError(f"{data_type} data not available. Load or preprocess data first.")

        # Calculate statistics
        stats = {
            "stage": stage,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
        }

        # Add date range if InvoiceDate column exists
        if "InvoiceDate" in df.columns:
            try:
                date_col = pd.to_datetime(df["InvoiceDate"]) if df["InvoiceDate"].dtype != "datetime64[ns]" else df["InvoiceDate"]
                stats["date_range"] = (
                    str(date_col.min()),
                    str(date_col.max()),
                )
            except Exception as e:
                logger.warning(f"Could not parse InvoiceDate for date range: {e}")
                stats["date_range"] = None

        # Add customer and product statistics
        if "CustomerID" in df.columns:
            stats["unique_customers"] = df["CustomerID"].nunique()
        if "Description" in df.columns:
            stats["unique_products"] = df["Description"].nunique()

        # Add additional metrics for processed data
        if stage == "processed":
            if "TransactionValue" in df.columns:
                stats["transaction_value_stats"] = {
                    "min": float(df["TransactionValue"].min()),
                    "max": float(df["TransactionValue"].max()),
                    "mean": float(df["TransactionValue"].mean()),
                    "median": float(df["TransactionValue"].median()),
                }
            if "Quantity" in df.columns:
                stats["quantity_stats"] = {
                    "min": int(df["Quantity"].min()),
                    "max": int(df["Quantity"].max()),
                    "mean": float(df["Quantity"].mean()),
                    "median": float(df["Quantity"].median()),
                }

        logger.info(f"Dataset Statistics ({stage} data): {stats}")
        return stats

    def convert_csv_to_tsv(self, input_filepath: Optional[str] = None, output_filepath: Optional[str] = None) -> bool:
        """
        Convert CSV file to TSV format with UTF-8 encoding.

        Args:
            input_filepath: Path to the input CSV file
            output_filepath: Path to save the TSV file. If None, replaces .csv with .tsv

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            source_filepath = input_filepath or self.config.raw_data_path

            if output_filepath is None:
                output_filepath = source_filepath.replace(".csv", ".tsv")

            # Load CSV with original encoding
            logger.info(f"Loading CSV from: {source_filepath}")
            df = pd.read_csv(source_filepath, encoding="ISO-8859-1")

            # Save as TSV with UTF-8 encoding
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            df.to_csv(output_filepath, sep="\t", encoding="utf-8", index=False)
            logger.info(f"Successfully converted and saved TSV to: {output_filepath}")
            logger.info(f"TSV file contains {len(df)} records with {len(df.columns)} columns")

            return True
        except Exception as e:
            logger.error(f"Failed to convert CSV to TSV: {e}")
            return False

    def _handle_cancellations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process cancellation flags from InvoiceNo."""
        df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")
        
        # Save cancellations to separate file
        cancellations = df[df["IsCancellation"]]
        if len(cancellations) > 0:
            processed_dir = os.path.dirname(self.config.processed_data_path)
            os.makedirs(processed_dir, exist_ok=True)
            cancellations_path = os.path.join(processed_dir, "Cancellations.tsv")
            cancellations.to_csv(cancellations_path, sep="\t", encoding="utf-8", index=False)
            logger.info(f"Saved {len(cancellations)} cancellations to: {cancellations_path}")
        
        # Remove cancellations from main dataframe
        df = df[~df["IsCancellation"]].copy()
        
        logger.info(f"Removed {len(cancellations)} cancellation entries")
        return df

    def _remove_missing_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove and save rows with missing CustomerID."""
        initial_rows = len(df)
        missing_customer_id = df[df["CustomerID"].isna()]
        
        if len(missing_customer_id) > 0:
            processed_dir = os.path.dirname(self.config.processed_data_path)
            os.makedirs(processed_dir, exist_ok=True)
            no_customer_id_path = os.path.join(processed_dir, "no_CustomerID.tsv")
            missing_customer_id.to_csv(no_customer_id_path, sep="\t", encoding="utf-8", index=False)
            logger.info(f"Saved {len(missing_customer_id)} records with missing CustomerID to: {no_customer_id_path}")

        df = df.dropna(subset=["CustomerID"])
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
        logger.info(f"Removed {initial_rows - len(df)} rows with missing CustomerID")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid and duplicate rows."""
        # Remove missing descriptions
        initial_rows = len(df)
        df = df.dropna(subset=["Description"])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing Description")

        # Remove non-positive quantities and prices; not cancellations
        initial_rows = len(df)
        df = df[df["Quantity"] > 0]
        df = df[df["UnitPrice"] > 0]
        logger.info(f"Removed {initial_rows - len(df)} rows with zero/negative quantities and prices (not cancellations)")

        # Remove duplicates and save them to a separate file
        initial_rows = len(df)
        duplicates = df[df.duplicated(subset=["InvoiceNo", "StockCode"], keep=False)]
        
        if len(duplicates) > 0:
            processed_dir = os.path.dirname(self.config.processed_data_path)
            os.makedirs(processed_dir, exist_ok=True)
            duplicates_path = os.path.join(processed_dir, "Duplicate_transactions.tsv")
            duplicates.to_csv(duplicates_path, sep="\t", encoding="utf-8", index=False)
            logger.info(f"Saved {len(duplicates)} duplicate transactions to: {duplicates_path}")
        
        df = df.drop_duplicates(subset=["InvoiceNo", "StockCode"], keep="first")
        logger.info(f"Removed {initial_rows - len(df)} duplicate transactions")

        return df

    def _normalize_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize product descriptions using Python string lowercasing.

        Args:
            df (pd.DataFrame): Input dataframe containing a Description column

        Returns:
            pd.DataFrame: Dataframe with normalized Description values (lowercase, stripped)
        """
        if "Description" not in df.columns:
            return df

        logger.info("Normalizing descriptions with Python string lowercase...")
        
        # Simple normalization: lowercase and strip whitespace
        df["Description"] = df["Description"].astype(str).str.strip().str.lower()
        
        logger.info(f"Normalized {df['Description'].nunique()} unique descriptions")
        return df

    def _enrich_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich product descriptions with category attributes using cache-first LLM tagging.

        Args:
            df (pd.DataFrame): Input dataframe with Description column (should be normalized)

        Returns:
            pd.DataFrame: Dataframe with added category columns (category, material, size, theme)
        """
        if "Description" not in df.columns:
            return df

        if not self.config.enrichment_enabled:
            logger.info("Category enrichment disabled; skipping")
            return df

        # Load cache with unified helper
        cache = self._validate_and_load_cache(
            self.config.cache_config.enrichment_cache_path,
            self.config.cache_config.enrichment_cache_first,
            self.force_reprocess,
        )

        llm_config = self.config.llm_config
        provider = llm_config.provider.lower() if llm_config.provider else "openai"
        
        # Validate LLM credentials using unified client
        config = self._build_pipeline_llm_config()
        client = LLMClient(config)
        llm_available = client.validate_credentials()

        if not llm_available:
            logger.warning(
                "Category enrichment enabled but provider credentials are missing; skipping enrichment"
            )
            for field in self.config.enrichment_fields:
                df[field] = np.nan
            return df

        # Use original (pre-normalization) Description for cache key to avoid re-tagging identical products
        # We'll enrich based on the current (normalized) Description value
        unique_descriptions = df["Description"].dropna().astype(str).unique()
        enrichment_map: Dict[str, Dict[str, str]] = {}

        logger.info(
            f"Enriching {len(unique_descriptions)} unique descriptions "
            f"with {len(self.config.enrichment_fields)} fields "
            f"(batch_size={self.config.enrichment_batch_size})"
        )

        # Separate cached and uncached descriptions
        descriptions_to_enrich = []
        cache_hits = 0
        
        for desc in unique_descriptions:
            if self.config.cache_config.enrichment_cache_first and not self.force_reprocess and desc in cache:
                enrichment_map[desc] = cache[desc]
                cache_hits += 1
            else:
                descriptions_to_enrich.append(desc)

        cache_misses = len(descriptions_to_enrich)
        logger.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")

        # Process uncached descriptions using batch processing
        if descriptions_to_enrich and llm_available:
            try:
                batch_results = enrich_categories_batch_with_llm(
                    texts=descriptions_to_enrich,
                    fields=self.config.enrichment_fields,
                    provider=provider,
                    model=llm_config.model,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    timeout_seconds=llm_config.timeout_seconds,
                    batch_size=self.config.enrichment_batch_size,
                    api_key=self._get_api_key_for_provider(provider),
                    endpoint=llm_config.azure_endpoint,
                    deployment=llm_config.azure_deployment,
                    api_version=llm_config.azure_api_version,
                    base_url=self._get_base_url_for_provider(provider),
                )
                enrichment_map.update(batch_results)
                
                # Update cache with new results
                if self.config.cache_config.enrichment_cache_first:
                    for desc, enriched in batch_results.items():
                        cache[desc] = enriched
                    save_json_file(self.config.cache_config.enrichment_cache_path, cache)
                    logger.info(
                        f"Saved enrichment cache to: {self.config.cache_config.enrichment_cache_path} ({len(cache)} total entries)"
                    )
                    
            except LLMQuotaExceededError as exc:
                self._handle_llm_quota_error_standardized(exc, fallback_strategy="fill_nan")
                # Fill remaining descriptions with NaN
                for desc in descriptions_to_enrich:
                    if desc not in enrichment_map:
                        enrichment_map[desc] = {field: "NaN" for field in self.config.enrichment_fields}
        elif descriptions_to_enrich:
            # LLM not available, fill with NaN
            for desc in descriptions_to_enrich:
                enrichment_map[desc] = {field: "NaN" for field in self.config.enrichment_fields}

        # Add enrichment columns to dataframe
        for field in self.config.enrichment_fields:
            # Create a mapping function that returns None instead of "NaN" string
            def get_field_value(desc, field_name=field):
                enrichment = enrichment_map.get(desc, {})
                value = enrichment.get(field_name, "NaN")
                return None if value == "NaN" else value
            
            # Apply mapping and handle None values properly
            df[field] = df["Description"].apply(get_field_value)

        logger.info(f"Category enrichment complete; added columns: {self.config.enrichment_fields}")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features and standardize data for machine learning pipeline.
        
        This method performs feature engineering on transaction data by:
        - Converting InvoiceDate to datetime format for temporal analysis
        - Extracting season information (1-4) from invoice month for seasonal patterns
        - Extracting day of week information (1=Monday, 7=Sunday) from InvoiceDate for daily patterns
        - Calculating transaction value by multiplying quantity and unit price
        - Normalizing product descriptions to lowercase and removing whitespace
        
        Args:
            df (pd.DataFrame): Input dataframe containing raw transaction data with columns:
                - InvoiceDate: Date of transaction (string or datetime)
                - Quantity: Number of items purchased (numeric)
                - UnitPrice: Price per item (numeric)
                - Description: Product description (string)
        
        Returns:
            pd.DataFrame: Enhanced dataframe with engineered features ready for model training.
        """
        """Create derived features and standardize data."""
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["InvoiceSeason"] = df["InvoiceDate"].dt.month % 12 // 3 + 1  # type: ignore[attr-defined]
        df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.dayofweek + 1  # type: ignore[attr-defined]
        df["TransactionValue"] = df["Quantity"] * df["UnitPrice"]
        if not self.config.normalization_enabled:
            df["Description"] = df["Description"].str.strip().str.lower()
        return df

    def _flag_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag suspicious transactions using IQR-based outliers and LLM-assisted batch scoring.

        Adds columns:
            - check_anomaly (bool)
            - anomaly_type (str)
            - anomaly_reason (str)
        """
        df["check_anomaly"] = False
        df["anomaly_type"] = None
        df["anomaly_reason"] = None

        if not self.config.outlier_enabled:
            logger.info("Outlier detection disabled; skipping")
            return df

        numeric_fields = [field for field in self.config.outlier_fields if field in df.columns]
        if not numeric_fields:
            logger.warning("Outlier detection enabled but no numeric fields available")
            return df

        outlier_indices = set()
        index_reasons: Dict[int, List[str]] = {}

        for field in numeric_fields:
            lower, upper = compute_iqr_bounds(df[field], self.config.outlier_iqr_multiplier)
            mask = df[field].notna() & ((df[field] < lower) | (df[field] > upper))
            for idx in df[mask].index:
                outlier_indices.add(idx)
                index_reasons.setdefault(idx, []).append(f"IQR outlier in {field}")

        if not outlier_indices:
            logger.info("No IQR outliers detected")
            return df

        llm_config = self.config.llm_config
        provider = llm_config.provider.lower() if llm_config.provider else "openai"
        
        # Validate LLM credentials using unified client
        config = self._build_pipeline_llm_config()
        client = LLMClient(config)
        llm_available = client.validate_credentials()

        if not llm_available:
            logger.warning("Outlier detection enabled but provider credentials are missing; using heuristic labels")

        cache = self._validate_and_load_cache(
            self.config.cache_config.outlier_cache_path,
            self.config.cache_config.outlier_cache_first,
            self.force_reprocess,
        )
        cache_updated = False

        def _record_key(row: pd.Series) -> str:
            return "|".join(
                [
                    str(row.get("InvoiceNo", "")),
                    str(row.get("StockCode", "")),
                    str(row.get("CustomerID", "")),
                    str(row.get("InvoiceDate", "")),
                    str(row.get("Quantity", "")),
                    str(row.get("UnitPrice", "")),
                    str(row.get("TransactionValue", "")),
                ]
            )

        candidates = df.loc[list(outlier_indices)].copy()
        candidates["_record_key"] = candidates.apply(_record_key, axis=1)
        candidates["_outlier_reasons"] = candidates.index.map(lambda i: ", ".join(index_reasons.get(i, [])))

        # Prepare batch records for LLM
        pending_records: List[Dict[str, Any]] = []
        pending_keys: List[str] = []
        results: Dict[str, Dict[str, str]] = {}

        for _, row in candidates.iterrows():
            key = row["_record_key"]
            if self.config.cache_config.outlier_cache_first and not self.force_reprocess and key in cache:
                results[key] = cache[key]
                continue

            record = {
                "key": key,
                "description": row.get("Description", ""),
                "quantity": row.get("Quantity", ""),
                "unit_price": row.get("UnitPrice", ""),
                "transaction_value": row.get("TransactionValue", ""),
                "customer_id": row.get("CustomerID", ""),
                "country": row.get("Country", ""),
                "invoice_date": str(row.get("InvoiceDate", "")),
                "outlier_reasons": row.get("_outlier_reasons", ""),
            }
            pending_records.append(record)
            pending_keys.append(key)

        if llm_available and pending_records:
            for i in range(0, len(pending_records), self.config.outlier_batch_size):
                batch = pending_records[i : i + self.config.outlier_batch_size]
                try:
                    batch_results = batch_score_anomalies_with_llm(
                        records=batch,
                        provider=provider,
                        model=llm_config.model,
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens,
                        timeout_seconds=llm_config.timeout_seconds,
                        api_key=self._get_api_key_for_provider(provider),
                        endpoint=llm_config.azure_endpoint,
                        deployment=llm_config.azure_deployment,
                        api_version=llm_config.azure_api_version,
                        base_url=self._get_base_url_for_provider(provider),
                    )
                except LLMQuotaExceededError as exc:
                    llm_available = False
                    self._handle_llm_quota_error_standardized(exc, fallback_strategy="use_heuristic")
                    batch_results = {}
                    results.update(batch_results)
                    break

                results.update(batch_results)

        # Heuristic fallback for any missing results
        for _, row in candidates.iterrows():
            key = row["_record_key"]
            if key in results:
                continue
            reasons = row.get("_outlier_reasons", "")
            if "Quantity" in reasons:
                anomaly_type = "bot-like"
            elif "UnitPrice" in reasons or "TransactionValue" in reasons:
                anomaly_type = "mispriced"
            else:
                anomaly_type = "invalid"
            results[key] = {
                "anomaly_type": anomaly_type,
                "anomaly_reason": reasons,
            }

        # Apply results to dataframe
        for idx, row in candidates.iterrows():
            key = row["_record_key"]
            result = results.get(key, {})
            df.at[idx, "check_anomaly"] = True  # type: ignore[call-overload]
            df.at[idx, "anomaly_type"] = result.get("anomaly_type", "none")  # type: ignore[call-overload]
            df.at[idx, "anomaly_reason"] = result.get("anomaly_reason", row.get("_outlier_reasons", ""))  # type: ignore[call-overload]

            if self.config.cache_config.outlier_cache_first:
                cache[key] = {
                    "anomaly_type": df.at[idx, "anomaly_type"],  # type: ignore[index]
                    "anomaly_reason": df.at[idx, "anomaly_reason"],  # type: ignore[index]
                }
                cache_updated = True

        if self.config.cache_config.outlier_cache_first and cache_updated:
            save_json_file(self.config.cache_config.outlier_cache_path, cache)
            logger.info(f"Saved outlier cache to: {self.config.cache_config.outlier_cache_path} ({len(cache)} entries)")

        anomalies = df[df["check_anomaly"]]
        if len(anomalies) > 0:
            os.makedirs(os.path.dirname(self.config.outlier_output_path), exist_ok=True)
            anomalies.to_csv(self.config.outlier_output_path, sep="\t", encoding="utf-8", index=False)
            logger.info(
                f"Saved {len(anomalies)} suspicious transactions to: {self.config.outlier_output_path}"
            )

        logger.info(f"Flagged {len(anomalies)} suspicious transactions")
        return df

    def _extract_contexts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract usage contexts from product descriptions using cache-first LLM.

        Args:
            df (pd.DataFrame): Input dataframe with Description column

        Returns:
            pd.DataFrame: Dataframe with added 'contexts' column (comma-separated context strings)
        """
        if "Description" not in df.columns:
            return df

        if not self.config.context_enabled:
            logger.info("Context extraction disabled; skipping")
            return df

        cache = self._validate_and_load_cache(
            self.config.cache_config.context_cache_path,
            self.config.cache_config.context_cache_first,
            self.force_reprocess,
        )
        cache_updated = False
        cache_hits = 0
        cache_misses = 0

        llm_config = self.config.llm_config
        provider = llm_config.provider.lower() if llm_config.provider else "openai"
        
        # Validate LLM credentials using unified client
        config = self._build_pipeline_llm_config()
        client = LLMClient(config)
        llm_available = client.validate_credentials()

        if not llm_available:
            logger.warning(
                "Context extraction enabled but provider credentials are missing; skipping extraction"
            )
            df["contexts"] = ""
            return df

        unique_descriptions = df["Description"].dropna().astype(str).unique()
        context_map: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Extracting contexts from {len(unique_descriptions)} unique descriptions")

        for desc in unique_descriptions:
            cache_key = desc

            if self.config.cache_config.context_cache_first and not self.force_reprocess and cache_key in cache:
                context_map[desc] = cache[cache_key]
                cache_hits += 1
                continue

            if llm_available:
                try:
                    result = extract_contexts_with_llm(
                        text=desc,
                        max_contexts=self.config.context_max_contexts,
                        provider=provider,
                        model=llm_config.model,
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens,
                        timeout_seconds=llm_config.timeout_seconds,
                        api_key=self._get_api_key_for_provider(provider),
                        endpoint=llm_config.azure_endpoint,
                        deployment=llm_config.azure_deployment,
                        api_version=llm_config.azure_api_version,
                        base_url=self._get_base_url_for_provider(provider),
                    )
                    cache_misses += 1
                except LLMQuotaExceededError as exc:
                    llm_available = False
                    self._handle_llm_quota_error_standardized(
                        exc,
                        fallback_strategy="skip_batch",
                    )
                    result = {"contexts": []}
            else:
                result = {"contexts": []}

            context_map[desc] = result
            if self.config.cache_config.context_cache_first:
                cache[cache_key] = result
                cache_updated = True

        if self.config.cache_config.context_cache_first and cache_updated:
            save_json_file(self.config.cache_config.context_cache_path, cache)
            logger.info(
                f"Saved context cache to: {self.config.cache_config.context_cache_path} ({cache_hits} hits, {cache_misses} misses)"
            )

        # Add contexts column (comma-separated context strings, filtered by min_confidence)
        def extract_filtered_contexts(desc):
            result = context_map.get(desc, {"contexts": []})
            contexts = result.get("contexts", [])
            
            # Filter by min_confidence and extract context strings
            filtered = [
                ctx.get("context", "")
                for ctx in contexts
                if isinstance(ctx, dict) and ctx.get("confidence", 0) >= self.config.context_min_confidence
            ]
            
            return ", ".join(filtered)

        df["contexts"] = df["Description"].apply(extract_filtered_contexts)
        logger.info(f"Context extraction complete; added {len(df)} context entries")
        return df

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

        df = self._handle_cancellations(df)
        df = self._remove_missing_customers(df)
        df = self._clean_data(df)
        df = self._normalize_descriptions(df)
        df = self._enrich_categories(df)
        df = self._extract_contexts(df)
        df = self._engineer_features(df)
        df = self._flag_anomalies(df)

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

        # Add transaction metadata using vectorized aggregation
        invoice_data = self.processed_data.groupby("InvoiceNo").agg(
            {
                "CustomerID": "first",
                "InvoiceDate": "first",
                "TransactionValue": "sum",
                "Quantity": "sum",
            }
        )

        baskets = baskets.merge(invoice_data, left_on="InvoiceNo", right_index=True)

        # In mixed datasets keep all invoices; in all-single datasets return empty baskets.
        # This preserves invoice-level metadata while avoiding bundle mining on single-item-only data.
        basket_sizes = baskets["Items"].str.len()
        if (basket_sizes > 1).sum() == 0:
            baskets = baskets.iloc[0:0].copy()

        logger.info(f"Created {len(baskets)} transaction baskets with multiple items")
        self.transactions = baskets

        return baskets

    def generate_product_bundles(
        self,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        max_size: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Generate product bundles using frequent itemset analysis (Apriori-like approach).
        Optimized with vectorized matrix operations for support calculation.

        Args:
            min_support: Minimum support threshold (fraction of transactions)
            min_confidence: Minimum confidence threshold
            max_size: Maximum bundle size (can be reduced for speed)

        Returns:
            List of product bundles (tuples of products)
        """
        if self.transactions is None:
            raise ValueError("Transaction baskets not created. Call create_transaction_baskets first.")

        min_support_value = min_support if min_support is not None else self.config.min_support
        min_confidence_value = min_confidence if min_confidence is not None else self.config.min_confidence
        max_size_value = max_size if max_size is not None else self.config.max_bundle_size

        logger.info(
            f"Generating bundles (min_support={min_support_value}, min_confidence={min_confidence_value}, max_size={max_size_value})..."
        )

        # Get all items and create item-to-index mapping
        all_items = [item for items in self.transactions["Items"] for item in items]
        item_counts = dict(Counter(all_items))
        total_transactions = len(self.transactions)

        # Filter frequent items
        frequent_items = {
            item: count
            for item, count in item_counts.items()
            if count / total_transactions >= min_support_value
        }

        logger.info(f"Found {len(frequent_items)} frequent items (support >= {min_support_value})")

        # Create item-to-index mapping for matrix operations
        item_to_idx = {item: idx for idx, item in enumerate(frequent_items.keys())}
        num_items = len(item_to_idx)

        # Create transaction-item matrix (sparse representation)
        # Each row is a transaction, each column is an item
        transaction_matrix = np.zeros((total_transactions, num_items), dtype=np.uint8)
        
        for trans_idx, items in enumerate(self.transactions["Items"]):
            for item in items:
                if item in item_to_idx:
                    transaction_matrix[trans_idx, item_to_idx[item]] = 1

        support_threshold = min_support_value * total_transactions

        # Generate itemsets of increasing size
        bundles = []
        current_itemsets = [[item] for item in frequent_items.keys()]

        for size in range(2, max_size_value + 1):
            # Generate candidate itemsets using Apriori principle
            candidates_set = set()
            
            if size == 2:
                # For size 2, generate ALL pairs of frequent items
                frequent_items_list = sorted(list(frequent_items.keys()))
                for i in range(len(frequent_items_list)):
                    for j in range(i + 1, len(frequent_items_list)):
                        candidate = tuple(sorted([frequent_items_list[i], frequent_items_list[j]]))
                        candidates_set.add(candidate)
                logger.info(f"Generated {len(candidates_set)} candidate pairs (size 2) from {len(frequent_items_list)} frequent items")
            else:
                # For size > 2, use Apriori principle: combine itemsets that share k-1 items
                for i in range(len(current_itemsets)):
                    for j in range(i + 1, len(current_itemsets)):
                        itemset_i = sorted(current_itemsets[i])
                        itemset_j = sorted(current_itemsets[j])
                        
                        # Check if first k-1 items are identical (Apriori pruning)
                        if itemset_i[:-1] == itemset_j[:-1]:
                            union = tuple(sorted(list(set(itemset_i) | set(itemset_j))))
                            if len(union) == size:
                                candidates_set.add(union)
                
                if candidates_set:
                    logger.info(f"Generated {len(candidates_set)} candidate itemsets of size {size} using Apriori principle")

            if not candidates_set:
                logger.info(f"No candidates generated for size {size}. Stopping bundle generation.")
                break

            # Vectorized support calculation using matrix operations
            valid_itemsets = []
            valid_count = 0
            for candidate in candidates_set:
                # Get column indices for items in candidate
                col_indices = [item_to_idx[item] for item in candidate]
                # Calculate support: count transactions where ALL items are present
                support = np.sum(np.all(transaction_matrix[:, col_indices], axis=1))
                support_pct = support / total_transactions if total_transactions > 0 else 0
                
                if support >= support_threshold:
                    valid_itemsets.append(tuple(candidate))
                    bundles.append(tuple(candidate))
                    valid_count += 1

            logger.info(f"Found {valid_count} frequent itemsets of size {size} (support >= {min_support_value:.4f})")
            
            current_itemsets = [list(itemset) for itemset in valid_itemsets]

            if not current_itemsets:
                logger.info(f"No frequent itemsets of size {size}. Stopping bundle generation.")
                break

        self.bundles = bundles
        logger.info(f"Total bundles generated: {len(bundles)}")

        return bundles

    def save_processed_data(self, filepath: Optional[str] = None) -> bool:
        """
        Save processed data to pickle file and a copy to TSV.

        Args:
            filepath: Path to save the processed data

        Returns:
            bool: True if successful
        """
        try:
            output_filepath = filepath or self.config.processed_data_path
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, "wb") as f:
                pickle.dump(
                    {
                        "processed_data": self.processed_data,
                        "transactions": self.transactions,
                        "bundles": self.bundles,
                        "timestamp": datetime.now(),
                    },
                    f,
                )
            logger.info(f"Processed data saved to: {output_filepath}")

            # Also save a copy as TSV for easier inspection
            tsv_filepath = os.path.splitext(output_filepath)[0] + ".tsv"
            try:
                self.processed_data.to_csv(tsv_filepath, sep="\t", index=False)  # type: ignore[union-attr]
                logger.info(f"Processed data also saved to TSV: {tsv_filepath}")
            except Exception as e:
                logger.error(f"Failed to save processed data as TSV: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return False

    def load_processed_data(self, filepath: Optional[str] = None) -> bool:
        """
        Load processed data from pickle file.

        Args:
            filepath: Path to load the processed data from

        Returns:
            bool: True if successful
        """
        input_filepath = filepath or self.config.processed_data_path
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Processed data file not found: {input_filepath}")

        try:
            with open(input_filepath, "rb") as f:
                data = pickle.load(f)
            self.processed_data = data["processed_data"]
            self.transactions = data["transactions"]
            self.bundles = data["bundles"]
            logger.info(f"Processed data loaded from: {input_filepath}")
            return True
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return False

    def get_bundle_statistics(self) -> Dict:
        """
        Get statistics about generated bundles.

        Returns:
            dict: Bundle statistics
        """
        if self.bundles is None or len(self.bundles) == 0:
            return {}

        bundle_sizes = [len(bundle) for bundle in self.bundles]

        stats = {
            "total_bundles": len(self.bundles),
            "min_bundle_size": min(bundle_sizes),
            "max_bundle_size": max(bundle_sizes),
            "avg_bundle_size": np.mean(bundle_sizes),
            "bundle_sizes": bundle_sizes,
            "bundle_size_distribution": pd.Series(bundle_sizes).value_counts().to_dict(),
            "top_10_bundles": self.bundles[:10],
        }

        return stats
