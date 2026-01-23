"""Bundle recommendation engine using machine learning algorithms."""

import os
import pickle
import logging
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

from src.config import RANDOM_STATE, TRAIN_TEST_SPLIT, N_JOBS

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

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
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
        Fit the Naive Bayes model.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            validation_split: Train-test split ratio

        Returns:
            dict: Training metrics
        """
        logger.info(f"Fitting {self.name}...")

        # Prepare features and labels
        X = self.mlb.fit_transform(transactions)
        self.feature_names = self.mlb.classes_

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
            X_train = X_train.toarray()
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
            X = X.toarray()

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

        X = self.mlb.transform(transactions)
        if self.model_type == "gaussian":
            X = X.toarray()

        return self.model.predict_proba(X)


class SVMBundleRecommender(BaseRecommender):
    """Bundle recommender using Support Vector Machine algorithm."""

    def __init__(self, kernel: str = "rbf", C: float = 1.0):
        """
        Initialize SVM recommender.

        Args:
            kernel: SVM kernel type ('linear', 'rbf', 'poly', etc.)
            C: Regularization parameter
        """
        super().__init__(name="SVMBundleRecommender")
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=RANDOM_STATE, n_jobs=N_JOBS)
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
        Fit the SVM model.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles
            validation_split: Train-test split ratio

        Returns:
            dict: Training metrics
        """
        logger.info(f"Fitting {self.name} with kernel={self.kernel}, C={self.C}...")

        # Prepare features and labels
        X = self.mlb.fit_transform(transactions)
        self.feature_names = self.mlb.classes_

        # Convert to dense and scale
        X_dense = X.toarray()
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
        X_scaled = self.scaler.transform(X.toarray())
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

        X = self.mlb.transform(transactions)
        X_scaled = self.scaler.transform(X.toarray())
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
        Fit all recommenders.

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

    def recommend_bundles(
        self,
        customer_transaction: List[str],
        recommender_name: str = None,
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

        recommendations = {
            "transaction": customer_transaction,
            "bundles": [],
            "confidence": [],
            "recommender": recommender_name or "ensemble",
        }

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

        if confidence >= threshold:
            # Find applicable bundles
            applicable_bundles = []
            for bundle in self.bundles:
                # Check if bundle items overlap with transaction
                overlap = len(set(bundle) & set(customer_transaction))
                if overlap > 0:
                    applicable_bundles.append(bundle)

            recommendations["bundles"] = applicable_bundles[:5]  # Top 5
            recommendations["confidence"] = float(confidence)

        return recommendations

    def get_cross_sell_products(
        self,
        customer_transaction: List[str],
        top_n: int = 5,
        recommender_name: str = None,
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
