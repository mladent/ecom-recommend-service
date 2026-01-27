"""Data splitting utilities for train-test and k-fold cross-validation."""

import logging
import numpy as np
from typing import List, Tuple, Optional, Generator, Dict, Any
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import RANDOM_STATE, TRAIN_TEST_SPLIT

logger = logging.getLogger(__name__)


class DataSplitter(ABC):
    """Base class for data splitting strategies."""

    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize data splitter.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    @abstractmethod
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate train-test splits.

        Yields:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        pass

    @abstractmethod
    def get_split_info(self) -> Dict[str, Any]:
        """Get information about the splitting strategy."""
        pass


class RandomSplit(DataSplitter):
    """Single random train-test split."""

    def __init__(
        self,
        test_size: float = 1 - TRAIN_TEST_SPLIT,
        random_state: int = RANDOM_STATE,
    ):
        """
        Initialize random splitter.

        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state=random_state)
        self.test_size = test_size
        self.train_size = 1 - test_size

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate a single random train-test split.

        Args:
            X: Feature matrix
            y: Target vector

        Yields:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield X_train, X_test, y_train, y_test

    def get_split_info(self) -> Dict[str, Any]:
        """Get information about the splitting strategy."""
        return {
            "strategy": "random_split",
            "test_size": self.test_size,
            "train_size": self.train_size,
            "n_splits": 1,
            "random_state": self.random_state,
        }


class KFoldSplit(DataSplitter):
    """K-fold cross-validation splitting."""

    def __init__(
        self,
        n_splits: int = 10,
        random_state: int = RANDOM_STATE,
        shuffle: bool = True,
    ):
        """
        Initialize k-fold splitter.

        Args:
            n_splits: Number of folds
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
        """
        super().__init__(random_state=random_state)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate k-fold cross-validation splits.

        Args:
            X: Feature matrix
            y: Target vector

        Yields:
            Tuple of (X_train, X_test, y_train, y_test) for each fold
        """
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(X)):
            logger.info(f"Generating fold {fold + 1}/{self.n_splits}")

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            yield X_train, X_test, y_train, y_test

    def get_split_info(self) -> Dict[str, Any]:
        """Get information about the splitting strategy."""
        return {
            "strategy": "k_fold_cross_validation",
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
        }


class BundleDataPreprocessor:
    """
    Preprocess transaction and bundle data for model training.

    Handles feature encoding and label generation.
    """

    def __init__(self):
        """Initialize preprocessor."""
        self.mlb = MultiLabelBinarizer()
        self.feature_names = None
        self.is_fitted = False

    def preprocess(
        self,
        transactions: List[List[str]],
        bundles: List[Tuple[str, ...]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess transactions and bundles into features and labels.

        Args:
            transactions: List of transaction items
            bundles: List of product bundles

        Returns:
            Tuple of (X, y) where:
                - X: Feature matrix (n_transactions, n_features)
                - y: Binary labels (n_transactions,)
        """
        logger.info(f"Preprocessing {len(transactions)} transactions and {len(bundles)} bundles...")

        # Encode transactions as binary feature vectors
        X = self.mlb.fit_transform(transactions)
        self.feature_names = self.mlb.classes_
        self.is_fitted = True

        # Create binary labels for each bundle
        y = np.zeros(len(transactions), dtype=int)
        for i, items in enumerate(transactions):
            for bundle in bundles:
                if all(item in items for item in bundle):
                    y[i] = 1
                    break

        logger.info(f"Generated features shape: {X.shape}, labels distribution: {np.bincount(y)}")
        return X, y

    def transform(self, transactions: List[List[str]]) -> np.ndarray:
        """
        Transform new transactions using fitted encoder.

        Args:
            transactions: List of transaction items

        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call preprocess() first.")

        return self.mlb.transform(transactions)

    def get_feature_names(self) -> Optional[np.ndarray]:
        """Get the names of features."""
        return self.feature_names
