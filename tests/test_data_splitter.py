"""Unit tests for data splitter module."""

import pytest
import numpy as np
from src.data_splitter import RandomSplit, KFoldSplit, BundleDataPreprocessor


class TestRandomSplit:
    """Test cases for RandomSplit class."""

    def test_random_split_initialization(self):
        """Test RandomSplit initialization."""
        splitter = RandomSplit(test_size=0.2)
        assert splitter.test_size == 0.2
        assert splitter.train_size == 0.8

    def test_random_split_default_values(self):
        """Test RandomSplit with default values."""
        splitter = RandomSplit()
        assert np.isclose(splitter.test_size, 0.2)  # Default is 1 - TRAIN_TEST_SPLIT (0.8)
        assert np.isclose(splitter.train_size, 0.8)

    def test_random_split_generates_single_split(self):
        """Test that RandomSplit generates exactly one split."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        splitter = RandomSplit(test_size=0.2)
        splits = list(splitter.split(X, y))

        assert len(splits) == 1
        X_train, X_test, y_train, y_test = splits[0]
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_random_split_reproducibility(self):
        """Test that RandomSplit is reproducible with same random_state."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        splitter1 = RandomSplit(test_size=0.2, random_state=42)
        X_train1, X_test1, y_train1, y_test1 = list(splitter1.split(X, y))[0]

        splitter2 = RandomSplit(test_size=0.2, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = list(splitter2.split(X, y))[0]

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)

    def test_random_split_info(self):
        """Test get_split_info method."""
        splitter = RandomSplit(test_size=0.2)
        info = splitter.get_split_info()

        assert info["strategy"] == "random_split"
        assert info["test_size"] == 0.2
        assert info["train_size"] == 0.8
        assert info["n_splits"] == 1


class TestKFoldSplit:
    """Test cases for KFoldSplit class."""

    def test_kfold_initialization(self):
        """Test KFoldSplit initialization."""
        splitter = KFoldSplit(n_splits=10)
        assert splitter.n_splits == 10
        assert splitter.shuffle is True

    def test_kfold_invalid_splits(self):
        """Test that KFoldSplit raises error for n_splits < 2."""
        with pytest.raises(ValueError):
            KFoldSplit(n_splits=1)

    def test_kfold_generates_correct_number_of_splits(self):
        """Test that KFoldSplit generates correct number of splits."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        splitter = KFoldSplit(n_splits=5)
        splits = list(splitter.split(X, y))

        assert len(splits) == 5

    def test_kfold_split_sizes(self):
        """Test that KFoldSplit splits have correct sizes."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        splitter = KFoldSplit(n_splits=5)
        splits = list(splitter.split(X, y))

        for X_train, X_test, y_train, y_test in splits:
            assert len(X_train) == 80
            assert len(X_test) == 20
            assert len(y_train) == 80
            assert len(y_test) == 20

    def test_kfold_no_overlap(self):
        """Test that KFoldSplit test sets don't overlap."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        splitter = KFoldSplit(n_splits=5, random_state=42)
        all_test_indices = []

        for X_train, X_test, y_train, y_test in splitter.split(X, y):
            # Collect indices by comparing with original X
            test_indices = [i for i in range(len(X)) if any((X[i] == row).all() for row in X_test)]
            all_test_indices.extend(test_indices)

        # All indices should appear exactly once
        assert len(all_test_indices) == len(X)

    def test_kfold_reproducibility(self):
        """Test that KFoldSplit is reproducible with same random_state."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        splitter1 = KFoldSplit(n_splits=5, random_state=42)
        splits1 = list(splitter1.split(X, y))

        splitter2 = KFoldSplit(n_splits=5, random_state=42)
        splits2 = list(splitter2.split(X, y))

        for (X_train1, X_test1, y_train1, y_test1), (X_train2, X_test2, y_train2, y_test2) in zip(
            splits1, splits2
        ):
            np.testing.assert_array_equal(X_train1, X_train2)
            np.testing.assert_array_equal(X_test1, X_test2)

    def test_kfold_info(self):
        """Test get_split_info method."""
        splitter = KFoldSplit(n_splits=10)
        info = splitter.get_split_info()

        assert info["strategy"] == "k_fold_cross_validation"
        assert info["n_splits"] == 10
        assert info["shuffle"] is True


class TestBundleDataPreprocessor:
    """Test cases for BundleDataPreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test BundleDataPreprocessor initialization."""
        preprocessor = BundleDataPreprocessor()
        assert preprocessor.is_fitted is False
        assert preprocessor.feature_names is None

    def test_preprocess_basic(self):
        """Test basic preprocessing."""
        transactions = [
            ["item_a", "item_b"],
            ["item_b", "item_c"],
            ["item_a", "item_c"],
        ]
        bundles = [("item_a", "item_b")]

        preprocessor = BundleDataPreprocessor()
        X, y = preprocessor.preprocess(transactions, bundles)

        assert X.shape[0] == 3  # 3 transactions
        assert y[0] == 1  # First transaction has the bundle
        assert y[1] == 0  # Second transaction doesn't have the bundle
        assert preprocessor.is_fitted is True

    def test_preprocess_multiple_bundles(self):
        """Test preprocessing with multiple bundles."""
        transactions = [
            ["item_a", "item_b", "item_c"],
            ["item_a", "item_b"],
            ["item_c"],
        ]
        bundles = [("item_a", "item_b"), ("item_a", "item_c")]

        preprocessor = BundleDataPreprocessor()
        X, y = preprocessor.preprocess(transactions, bundles)

        assert y[0] == 1  # First transaction has both bundles
        assert y[1] == 1  # Second transaction has first bundle
        assert y[2] == 0  # Third transaction has neither bundle

    def test_preprocess_feature_names(self):
        """Test that feature names are extracted correctly."""
        transactions = [
            ["item_a", "item_b"],
            ["item_b", "item_c"],
        ]
        bundles = [("item_a", "item_b")]

        preprocessor = BundleDataPreprocessor()
        X, y = preprocessor.preprocess(transactions, bundles)

        feature_names = preprocessor.get_feature_names()
        assert feature_names is not None
        assert "item_a" in feature_names
        assert "item_b" in feature_names
        assert "item_c" in feature_names

    def test_transform_without_fit(self):
        """Test that transform raises error if not fitted."""
        preprocessor = BundleDataPreprocessor()
        transactions = [["item_a", "item_b"]]

        with pytest.raises(ValueError):
            preprocessor.transform(transactions)

    def test_transform_after_fit(self):
        """Test transform after fitting."""
        transactions_train = [
            ["item_a", "item_b"],
            ["item_b", "item_c"],
        ]
        bundles = [("item_a", "item_b")]

        preprocessor = BundleDataPreprocessor()
        preprocessor.preprocess(transactions_train, bundles)

        # Transform new transactions
        transactions_test = [["item_a", "item_c"]]
        X_test = preprocessor.transform(transactions_test)

        assert X_test.shape[0] == 1
        assert X_test.shape[1] == 3  # Same number of features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
