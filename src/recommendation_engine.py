"""Bundle recommendation engine using machine learning algorithms."""

import os
import pickle
import logging
import hashlib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    RANDOM_STATE,
    TRAIN_TEST_SPLIT,
    N_JOBS,
    SVM_KERNEL,
    SVM_C,
    OOS_ENABLED,
    OOS_CACHE_FIRST,
    OOS_CACHE_PATH,
    OOS_INVENTORY_PATH,
    OOS_MAX_ALTERNATIVES,
    OOS_MIN_SCORE,
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
from src.data_splitter import RandomSplit, KFoldSplit, BundleDataPreprocessor
from src.utils import (
    load_inventory_csv,
    load_json_file,
    save_json_file,
    normalize_description_basic,
    select_alternatives_with_llm,
    select_alternative_heuristic,
    LLMQuotaExceededError,
)

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Base class for bundle recommenders."""

    def __init__(self, name: str = "BaseRecommender"):
        """Initialize base recommender."""
        self.name = name
        self.model = None
        self.feature_encoder = None
        self.is_fitted = False
        self.feature_names = None
        self.known_classes = None  # Store known classes from training

    def _filter_unknown_classes(self, transactions: List[List[str]]) -> Tuple[List[List[str]], int]:
        """
        Filter out unknown classes from transactions to prevent sklearn warnings.
        
        Args:
            transactions: List of transaction items (list of lists of strings)
            
        Returns:
            Tuple of (filtered_transactions, num_filtered_items): 
            - filtered_transactions: Transactions with only known classes
            - num_filtered_items: Total number of items filtered out
        """
        # Initialize known_classes from feature_names if not set (for backwards compatibility with pickled models)
        known_classes = getattr(self, 'known_classes', None)
        if known_classes is None:
            if self.feature_names is not None:
                known_classes = set(self.feature_names)
            else:
                return transactions, 0
            
        filtered_transactions = []
        num_filtered = 0
        
        for transaction in transactions:
            filtered_items = [item for item in transaction if item in known_classes]
            num_filtered += len(transaction) - len(filtered_items)
            
            if filtered_items:  # Keep transaction even if some items are filtered
                filtered_transactions.append(filtered_items)
            else:
                # If all items are unknown, keep empty transaction (important for prediction shape)
                filtered_transactions.append([])
        
        if num_filtered > 0:
            logger.debug(f"Filtered out {num_filtered} unknown class items from {len(transactions)} transactions")
        
        return filtered_transactions, num_filtered

    @abstractmethod
    def fit(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        validation_split: float = TRAIN_TEST_SPLIT,
    ) -> Dict:
        """Fit the model with transactions and bundles."""
        pass

    @abstractmethod
    def predict(self, transactions: List[List[str]]) -> np.ndarray:
        """Make predictions from transactions."""
        pass

    @abstractmethod
    def predict_proba(self, transactions: List[List[str]]) -> np.ndarray:
        """Predict probabilities from transactions."""
        pass


class NaiveBayesBundleRecommender(BaseRecommender):
    """Bundle recommender using Naive Bayes algorithm."""

    def __init__(self, model_type: str = "multinomial"):
        """
        Initialize Naive Bayes recommender.

        Args:
            model_type: 'multinomial' or 'gaussian'
        """
        super().__init__(name="NaiveBayesBundleRecommender")
        self.model_type = model_type

        if model_type == "multinomial":
            self.model = MultinomialNB()
        elif model_type == "gaussian":
            self.model = GaussianNB()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.mlb = MultiLabelBinarizer()
        self.feature_names = None

    def fit(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        validation_split: float = TRAIN_TEST_SPLIT,
    ) -> Dict:
        """
        Fit the Naive Bayes model using random train-test split.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            validation_split: Train-test split ratio

        Returns:
            dict: Training metrics
        """
        logger.info(f"Fitting {self.name} with random split...")

        # Prepare features and labels
        X = self.mlb.fit_transform(transactions)
        self.feature_names = self.mlb.classes_
        self.known_classes = set(self.mlb.classes_)  # Store for filtering in predict

        # Create binary labels for each bundle
        y = np.zeros(len(transactions), dtype=int)
        for i, items in enumerate(transactions):
            for bundle in bundles:
                if all(item in items for item in bundle):
                    y[i] = 1
                    break

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - validation_split, random_state=RANDOM_STATE
        )

        # Handle dense/sparse for Gaussian NB
        if self.model_type == "gaussian":
            if hasattr(X_train, "toarray"):
                X_train = X_train.toarray()
            if hasattr(X_test, "toarray"):
                X_test = X_test.toarray()

        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        logger.info(f"Training metrics: {metrics}")
        return metrics

    def fit_with_splitter(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        splitter,
    ) -> Dict:
        """
        Fit the Naive Bayes model with custom data splitter (random or k-fold).

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            splitter: DataSplitter instance (RandomSplit or KFoldSplit)

        Returns:
            dict: Aggregated metrics across all splits
        """
        logger.info(f"Fitting {self.name} with {splitter.get_split_info()['strategy']}...")

        # Prepare features and labels
        X = self.mlb.fit_transform(transactions)
        self.feature_names = self.mlb.classes_
        self.known_classes = set(self.mlb.classes_)  # Store for filtering in predict

        # Create binary labels for each bundle
        y = np.zeros(len(transactions), dtype=int)
        for i, items in enumerate(transactions):
            for bundle in bundles:
                if all(item in items for item in bundle):
                    y[i] = 1
                    break

        # Iterate through splits
        all_metrics = []
        for split_num, (X_train, X_test, y_train, y_test) in enumerate(splitter.split(X, y)):
            logger.info(f"Processing split {split_num + 1}...")

            # Handle dense/sparse for Gaussian NB
            X_train_proc = X_train
            X_test_proc = X_test
            if self.model_type == "gaussian":
                if hasattr(X_train_proc, "toarray"):
                    X_train_proc = X_train_proc.toarray()
                if hasattr(X_test_proc, "toarray"):
                    X_test_proc = X_test_proc.toarray()

            # Fit model
            model = (
                MultinomialNB()
                if self.model_type == "multinomial"
                else GaussianNB()
            )
            model.fit(X_train_proc, y_train)

            # Evaluate
            y_pred = model.predict(X_test_proc)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
            all_metrics.append(metrics)

        # Use the last model as the fitted model
        self.model = model  # type: ignore[possibly-unbound]
        self.is_fitted = True

        # Average metrics across splits
        avg_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "precision": np.mean([m["precision"] for m in all_metrics]),
            "recall": np.mean([m["recall"] for m in all_metrics]),
            "f1": np.mean([m["f1"] for m in all_metrics]),
            "std_accuracy": np.std([m["accuracy"] for m in all_metrics]),
            "std_precision": np.std([m["precision"] for m in all_metrics]),
            "std_recall": np.std([m["recall"] for m in all_metrics]),
            "std_f1": np.std([m["f1"] for m in all_metrics]),
            "n_splits": len(all_metrics),
        }

        logger.info(f"Aggregated metrics: {avg_metrics}")
        return avg_metrics

    def predict(self, transactions: List[List[str]]) -> np.ndarray:
        """
        Predict bundle recommendations.

        Args:
            transactions: List of transaction items

        Returns:
            np.ndarray: Predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.mlb.transform(transactions)
        if self.model_type == "gaussian":
            if hasattr(X, "toarray"):
                X = X.toarray()  # type: ignore[union-attr]

        return self.model.predict(X)

    def predict_proba(self, transactions: List[List[str]]) -> np.ndarray:
        """
        Predict bundle recommendation probabilities.

        Args:
            transactions: List of transaction items

        Returns:
            np.ndarray: Probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Filter unknown classes to prevent sklearn warnings
        filtered_transactions, num_filtered = self._filter_unknown_classes(transactions)
        if num_filtered > 0:
            logger.debug(f"{self.name}: Removed {num_filtered} unknown product(s) before prediction")

        X = self.mlb.transform(filtered_transactions)
        if self.model_type == "gaussian":
            if hasattr(X, "toarray"):
                X = X.toarray()  # type: ignore[union-attr]

        return self.model.predict_proba(X)


class SVMBundleRecommender(BaseRecommender):
    """Bundle recommender using Support Vector Machine algorithm."""

    def __init__(self, kernel: str = "linear", C: float = 1.0):
        """
        Initialize SVM recommender.

        Args:
            kernel: SVM kernel type ('linear', 'rbf', 'poly', etc.)
            C: Regularization parameter
        """
        super().__init__(name="SVMBundleRecommender")
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=RANDOM_STATE)  # type: ignore[arg-type]
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.use_scaling = True

    def fit(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        validation_split: float = TRAIN_TEST_SPLIT,
    ) -> Dict:
        """
        Fit the SVM model using random train-test split.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            validation_split: Train-test split ratio

        Returns:
            dict: Training metrics
        """
        logger.info(f"Fitting {self.name} with kernel={self.kernel}, C={self.C} using random split...")

        # Prepare features and labels
        X = self.mlb.fit_transform(transactions)
        self.feature_names = self.mlb.classes_
        self.known_classes = set(self.mlb.classes_)  # Store for filtering in predict

        # Convert to dense and scale
        X_dense = X.toarray() if hasattr(X, "toarray") else X  # type: ignore[union-attr]
        X_scaled = self.scaler.fit_transform(X_dense)

        # Create binary labels for each bundle
        y = np.zeros(len(transactions), dtype=int)
        for i, items in enumerate(transactions):
            for bundle in bundles:
                if all(item in items for item in bundle):
                    y[i] = 1
                    break

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=1 - validation_split, random_state=RANDOM_STATE
        )

        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        logger.info(f"Training metrics: {metrics}")
        return metrics

    def fit_with_splitter(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        splitter,
    ) -> Dict:
        """
        Fit the SVM model with custom data splitter (random or k-fold).

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            splitter: DataSplitter instance (RandomSplit or KFoldSplit)

        Returns:
            dict: Aggregated metrics across all splits
        """
        logger.info(f"Fitting {self.name} with {splitter.get_split_info()['strategy']}...")

        # Prepare features and labels
        X = self.mlb.fit_transform(transactions)
        self.feature_names = self.mlb.classes_
        self.known_classes = set(self.mlb.classes_)  # Store for filtering in predict

        # Convert to dense and scale
        X_dense = X.toarray() if hasattr(X, "toarray") else X  # type: ignore[union-attr]
        X_scaled = self.scaler.fit_transform(X_dense)

        # Create binary labels for each bundle
        y = np.zeros(len(transactions), dtype=int)
        for i, items in enumerate(transactions):
            for bundle in bundles:
                if all(item in items for item in bundle):
                    y[i] = 1
                    break

        # Iterate through splits
        all_metrics = []
        for split_num, (X_train, X_test, y_train, y_test) in enumerate(splitter.split(X_scaled, y)):
            logger.info(f"Processing split {split_num + 1}...")

            # Fit model
            model = SVC(
                kernel=self.kernel,  # type: ignore[arg-type]
                C=self.C,
                probability=True,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
            all_metrics.append(metrics)

        # Use the last model as the fitted model
        self.model = model  # type: ignore[possibly-unbound]
        self.is_fitted = True

        # Average metrics across splits
        avg_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "precision": np.mean([m["precision"] for m in all_metrics]),
            "recall": np.mean([m["recall"] for m in all_metrics]),
            "f1": np.mean([m["f1"] for m in all_metrics]),
            "std_accuracy": np.std([m["accuracy"] for m in all_metrics]),
            "std_precision": np.std([m["precision"] for m in all_metrics]),
            "std_recall": np.std([m["recall"] for m in all_metrics]),
            "std_f1": np.std([m["f1"] for m in all_metrics]),
            "n_splits": len(all_metrics),
        }

        logger.info(f"Aggregated metrics: {avg_metrics}")
        return avg_metrics

    def predict(self, transactions: List[List[str]]) -> np.ndarray:
        """
        Predict bundle recommendations.

        Args:
            transactions: List of transaction items

        Returns:
            np.ndarray: Predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.mlb.transform(transactions)
        X_dense = X.toarray() if hasattr(X, "toarray") else X  # type: ignore[union-attr]
        X_scaled = self.scaler.transform(X_dense)
        return self.model.predict(X_scaled)

    def predict_proba(self, transactions: List[List[str]]) -> np.ndarray:
        """
        Predict bundle recommendation probabilities.

        Args:
            transactions: List of transaction items

        Returns:
            np.ndarray: Probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Filter unknown classes to prevent sklearn warnings
        filtered_transactions, num_filtered = self._filter_unknown_classes(transactions)
        if num_filtered > 0:
            logger.debug(f"{self.name}: Removed {num_filtered} unknown product(s) before prediction")

        X = self.mlb.transform(filtered_transactions)
        X_dense = X.toarray() if hasattr(X, "toarray") else X  # type: ignore[union-attr]
        X_scaled = self.scaler.transform(X_dense)
        return self.model.predict_proba(X_scaled)


class BundleRecommendationEngine:
    """Main bundle recommendation engine."""

    def __init__(self):
        """Initialize the recommendation engine."""
        self.recommenders = {}
        self.transactions = None
        self.bundles = None
        self.recommendations_cache = {}

    def add_recommender(self, name: str, recommender: BaseRecommender) -> None:
        """
        Add a recommender to the engine.

        Args:
            name: Name identifier for the recommender
            recommender: Recommender instance
        """
        self.recommenders[name] = recommender
        logger.info(f"Added recommender: {name}")

    def fit_all(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        validation_split: float = TRAIN_TEST_SPLIT,
    ) -> Dict[str, Dict]:
        """
        Fit all recommenders using random train-test split.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            validation_split: Train-test split ratio

        Returns:
            dict: Metrics for each recommender
        """
        self.transactions = transactions
        self.bundles = bundles

        metrics = {}
        for name, recommender in self.recommenders.items():
            logger.info(f"Training {name}...")
            metrics[name] = recommender.fit(transactions, bundles, validation_split)

        return metrics

    def fit_all_with_kfold(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        n_splits: int = 10,
    ) -> Dict[str, Dict]:
        """
        Fit all recommenders using k-fold cross-validation.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            n_splits: Number of folds (default: 10)

        Returns:
            dict: Aggregated metrics for each recommender across all folds
        """
        self.transactions = transactions
        self.bundles = bundles

        splitter = KFoldSplit(n_splits=n_splits)
        metrics = {}

        for name, recommender in self.recommenders.items():
            logger.info(f"Training {name} with {n_splits}-fold cross-validation...")
            metrics[name] = recommender.fit_with_splitter(transactions, bundles, splitter)

        return metrics

    def fit_all_with_random_split(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
        test_size: float = 0.2,
    ) -> Dict[str, Dict]:
        """
        Fit all recommenders using single random train-test split.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            test_size: Fraction of data to use for testing (default: 0.2)

        Returns:
            dict: Metrics for each recommender
        """
        self.transactions = transactions
        self.bundles = bundles

        splitter = RandomSplit(test_size=test_size)
        metrics = {}

        for name, recommender in self.recommenders.items():
            logger.info(f"Training {name} with random split (test_size={test_size})...")
            metrics[name] = recommender.fit_with_splitter(transactions, bundles, splitter)

        return metrics

    def recommend_bundles(
        self,
        customer_transaction: List[str],
        recommender_name: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Recommend bundles for a customer.

        Args:
            customer_transaction: Items in customer's current transaction
            recommender_name: Name of recommender to use (if None, use average)
            threshold: Confidence threshold for recommendations

        Returns:
            dict: Recommended bundles with confidence scores
        """
        if not self.recommenders:
            raise ValueError("No recommenders available. Call add_recommender first.")

        if not self.bundles:
            raise ValueError("Bundles not set. Call fit_all first.")

        if recommender_name:
            # Use specific recommender
            if recommender_name not in self.recommenders:
                raise ValueError(f"Recommender '{recommender_name}' not found")

            recommender = self.recommenders[recommender_name]
            proba = recommender.predict_proba([customer_transaction])
            confidence = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]

        else:
            # Use ensemble (average probabilities from all recommenders)
            probas = []
            for recommender in self.recommenders.values():
                proba = recommender.predict_proba([customer_transaction])
                probas.append(proba[0, 1] if proba.shape[1] > 1 else proba[0, 0])
            confidence = np.mean(probas)

        # Initialize recommendations with confidence score
        recommendations = {
            "transaction": customer_transaction,
            "bundles": [],
            "confidence": float(confidence),
            "recommender": recommender_name or "ensemble",
        }

        if confidence >= threshold:
            # Find applicable bundles
            applicable_bundles = []
            for bundle in self.bundles:
                # Check if bundle items overlap with transaction
                overlap = len(set(bundle) & set(customer_transaction))
                if overlap > 0:
                    applicable_bundles.append(bundle)

            recommendations["bundles"] = applicable_bundles[:5]  # Top 5

        # Resolve out-of-stock items with alternatives (LLM-assisted)
        if OOS_ENABLED and recommendations["bundles"]:
            inventory = load_inventory_csv(OOS_INVENTORY_PATH)
            in_stock_items = set()
            for bundle in recommendations["bundles"]:
                for item in bundle:
                    normalized = normalize_description_basic(item)
                    if inventory.get(normalized, True):
                        in_stock_items.add(item)

            candidates = list(in_stock_items)
            cache = load_json_file(OOS_CACHE_PATH) if OOS_CACHE_FIRST else {}
            cache_updated = False

            provider = LLM_PROVIDER.lower() if LLM_PROVIDER else "openai"
            
            # Validate LLM credentials using unified client
            config = LLMConfig(
                provider=provider,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
                openai_api_key=OPENAI_API_KEY,
                azure_api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                azure_api_version=AZURE_OPENAI_API_VERSION,
                gemini_api_key=GEMINI_API_KEY,
                anthropic_api_key=ANTHROPIC_API_KEY,
                perplexity_api_key=PERPLEXITY_API_KEY,
                perplexity_base_url=PERPLEXITY_BASE_URL,
            )
            client = LLMClient(config)
            llm_available = client.validate_credentials()

            candidate_hash = hashlib.sha256("|".join(sorted(candidates)).encode("utf-8")).hexdigest()
            resolved_bundles = []
            substitutions = []

            for bundle in recommendations["bundles"]:
                resolved_bundle = list(bundle)
                bundle_subs = []
                for idx, item in enumerate(bundle):
                    normalized = normalize_description_basic(item)
                    if inventory.get(normalized, True):
                        continue

                    cache_key = f"{normalized}|{candidate_hash}"
                    alternatives = []
                    if OOS_CACHE_FIRST and cache_key in cache:
                        alternatives = cache[cache_key].get("alternatives", [])
                    elif llm_available:
                        try:
                            alternatives = select_alternatives_with_llm(
                                missing_item=item,
                                candidates=candidates,
                                provider=provider,
                                model=LLM_MODEL,
                                temperature=LLM_TEMPERATURE,
                                max_tokens=LLM_MAX_TOKENS,
                                timeout_seconds=LLM_TIMEOUT_SECONDS,
                                max_alternatives=OOS_MAX_ALTERNATIVES,
                                api_key=(
                                    OPENAI_API_KEY
                                    if provider == "openai"
                                    else AZURE_OPENAI_API_KEY
                                    if provider == "azure"
                                    else GEMINI_API_KEY
                                    if provider == "gemini"
                                    else ANTHROPIC_API_KEY
                                    if provider == "anthropic"
                                    else PERPLEXITY_API_KEY
                                ),
                                endpoint=AZURE_OPENAI_ENDPOINT,
                                deployment=AZURE_OPENAI_DEPLOYMENT,
                                api_version=AZURE_OPENAI_API_VERSION,
                                base_url=PERPLEXITY_BASE_URL if provider == "perplexity" else None,
                            )
                        except LLMQuotaExceededError as exc:
                            llm_available = False
                            logger.warning("LLM quota exceeded; falling back to heuristic alternatives")
                            logger.debug(f"Quota error detail: {exc}")
                            alternatives = []

                    if not alternatives:
                        fallback = select_alternative_heuristic(item, candidates)
                        alternatives = [fallback] if fallback else []

                    best = None
                    if alternatives:
                        best = max(alternatives, key=lambda a: float(a.get("score", 0)))

                    if best and float(best.get("score", 0)) >= OOS_MIN_SCORE:
                        resolved_bundle[idx] = best.get("item")
                        bundle_subs.append(
                            {
                                "missing_item": item,
                                "alternative_item": best.get("item"),
                                "score": float(best.get("score", 0)),
                                "reason": best.get("reason", ""),
                            }
                        )

                    if OOS_CACHE_FIRST:
                        cache[cache_key] = {"alternatives": alternatives}
                        cache_updated = True

                resolved_bundles.append(tuple(resolved_bundle))
                if bundle_subs:
                    substitutions.append(
                        {
                            "original_bundle": bundle,
                            "resolved_bundle": tuple(resolved_bundle),
                            "substitutions": bundle_subs,
                        }
                    )

            recommendations["bundles"] = resolved_bundles
            recommendations["bundle_substitutions"] = substitutions

            if OOS_CACHE_FIRST and cache_updated:
                save_json_file(OOS_CACHE_PATH, cache)

        return recommendations

    def get_cross_sell_products(
        self,
        customer_transaction: List[str],
        top_n: int = 5,
        recommender_name: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get cross-sell product recommendations.

        Args:
            customer_transaction: Items in customer's current transaction
            top_n: Number of recommendations to return
            recommender_name: Name of recommender to use

        Returns:
            List of (product, confidence_score) tuples
        """
        if not self.bundles:
            raise ValueError("Bundles not set. Call fit_all first.")

        cross_sell_products = {}

        # Find products that appear with customer's items in bundles
        for bundle in self.bundles:
            bundle_set = set(bundle)
            trans_set = set(customer_transaction)

            # Find products in bundle that aren't in current transaction
            new_products = bundle_set - trans_set

            if new_products:
                for product in new_products:
                    if product not in cross_sell_products:
                        cross_sell_products[product] = 0

                    # Score based on bundle applicability
                    overlap = len(bundle_set & trans_set)
                    bundle_size = len(bundle)
                    score = overlap / bundle_size

                    cross_sell_products[product] += score

        # Sort by score and return top N
        sorted_products = sorted(cross_sell_products.items(), key=lambda x: x[1], reverse=True)
        return sorted_products[:top_n]

    def save_model(self, filepath: str) -> bool:
        """
        Save the trained models.

        Args:
            filepath: Path to save the model

        Returns:
            bool: True if successful
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "recommenders": self.recommenders,
                        "bundles": self.bundles,
                        "timestamp": datetime.now(),
                    },
                    f,
                )
            logger.info(f"Model saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Load trained models.

        Args:
            filepath: Path to load the model from

        Returns:
            bool: True if successful
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.recommenders = data["recommenders"]
            self.bundles = data["bundles"]
            logger.info(f"Model loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_engine_stats(self) -> Dict:
        """
        Get engine statistics.

        Returns:
            dict: Engine statistics
        """
        stats = {
            "num_recommenders": len(self.recommenders),
            "recommender_names": list(self.recommenders.keys()),
            "num_bundles": len(self.bundles) if self.bundles else 0,
            "num_transactions": len(self.transactions) if self.transactions else 0,
        }
        return stats
