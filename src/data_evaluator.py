"""Data evaluation and statistical analysis module."""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataEvaluator:
    """Comprehensive data evaluation and statistical analysis."""

    def __init__(self, df: pd.DataFrame, data_path: str = "data"):
        """
        Initialize data evaluator.

        Args:
            df: DataFrame to evaluate
            data_path: Path to save reports
        """
        self.df = df
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        self.report = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def evaluate(self) -> Dict:
        """
        Run comprehensive data evaluation.

        Returns:
            dict: Complete evaluation report
        """
        logger.info("Starting comprehensive data evaluation...")

        self.report = {
            "timestamp": self.timestamp,
            "dataset_overview": self._dataset_overview(),
            "columns": {},
            "data_quality": self._data_quality_issues(),
        }

        # Evaluate each column
        for col in self.df.columns:
            logger.info(f"Evaluating column: {col}")
            self.report["columns"][col] = self._evaluate_column(col)

        return self.report

    def _dataset_overview(self) -> Dict:
        """Get basic dataset overview."""
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
        }

    def _data_quality_issues(self) -> Dict:
        """Identify data quality issues."""
        issues = {
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": len(self.df[self.df.duplicated()]),
            "fully_null_columns": [
                col for col in self.df.columns if self.df[col].isnull().all()
            ],
        }
        return issues

    def _evaluate_column(self, col: str) -> Dict:
        """Evaluate a single column."""
        dtype = self.df[col].dtype
        result = {
            "dtype": str(dtype),
            "non_null_count": int(self.df[col].notna().sum()),
            "null_count": int(self.df[col].isnull().sum()),
            "null_percentage": round(
                self.df[col].isnull().sum() / len(self.df) * 100, 2
            ),
            "unique_count": int(self.df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(dtype):
            result.update(self._evaluate_numeric(col))
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            result.update(self._evaluate_text(col))
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            result.update(self._evaluate_datetime(col))

        return result

    def _evaluate_numeric(self, col: str) -> Dict:
        """Evaluate numeric column."""
        data = self.df[col].dropna()

        if len(data) == 0:
            return {}

        stats = {
            "five_number_summary": {
                "min": float(data.min()),
                "q1": float(data.quantile(0.25)),
                "median": float(data.quantile(0.50)),
                "q3": float(data.quantile(0.75)),
                "max": float(data.max()),
            },
            "mean": float(data.mean()),
            "std": float(data.std()),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
        }

        # Detect outliers using IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        stats["outliers"] = {
            "count": int(len(outliers)),
            "percentage": round(len(outliers) / len(data) * 100, 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "sample_outliers": list(outliers.head(10).values),
        }

        # Check for potential errors
        stats["potential_errors"] = self._check_numeric_errors(col, data)

        return stats

    def _check_numeric_errors(self, col: str, data: pd.Series) -> Dict:
        """Check for potential numeric errors."""
        errors = {"negative_values": 0, "zero_values": 0, "extreme_values": 0}

        if (data < 0).any():
            errors["negative_values"] = int((data < 0).sum())
            errors["negative_samples"] = list(data[data < 0].head(5).values)

        if (data == 0).any():
            errors["zero_values"] = int((data == 0).sum())

        # Check for extreme values (beyond 5 standard deviations)
        mean, std = data.mean(), data.std()
        if std > 0:
            extreme = data[np.abs((data - mean) / std) > 5]
            if len(extreme) > 0:
                errors["extreme_values"] = int(len(extreme))
                errors["extreme_samples"] = list(extreme.head(5).values)

        return errors

    def _evaluate_text(self, col: str) -> Dict:
        """Evaluate text column."""
        data = self.df[col].dropna()

        if len(data) == 0:
            return {}

        # Text statistics
        stats = {
            "min_length": int(data.str.len().min()),
            "max_length": int(data.str.len().max()),
            "mean_length": float(data.str.len().mean()),
            "empty_strings": int((data.str.len() == 0).sum()),
        }

        # Compare raw vs cleaned
        cleaned = data.str.lower().str.replace(r"[^a-z\s]", "", regex=True)
        differences = (data.str.lower() != cleaned).sum()
        stats["text_cleaning"] = {
            "rows_with_non_alphanumeric": int(differences),
            "percentage": round(differences / len(data) * 100, 2),
            "case_sensitive_unique": int(data.nunique()),
            "case_insensitive_unique": int(cleaned.nunique()),
            "unique_count_reduction": int(data.nunique()) - int(cleaned.nunique()),
        }

        # Sample unique values
        unique_vals = list(data.unique()[:20])
        stats["sample_values"] = unique_vals

        # Check for suspicious patterns
        stats["suspicious_patterns"] = self._check_text_patterns(col, data)

        # Save all unique categories to file
        self._save_categories(col, data)

        return stats

    def _check_text_patterns(self, col: str, data: pd.Series) -> Dict:
        """Check for suspicious text patterns."""
        patterns = {
            "very_long_values": int((data.str.len() > 200).sum()),
            "contains_urls": int(data.str.contains(r"http|www", na=False).sum()),
            "contains_emails": int(data.str.contains(r"\w+@\w+", na=False, regex=True).sum()),
            "contains_numbers": int(data.str.contains(r"\d", na=False).sum()),
            "all_whitespace": int(data.str.strip().str.len().eq(0).sum()),
        }

        # Sample suspicious values
        if patterns["very_long_values"] > 0:
            patterns["sample_long_values"] = list(
                data[data.str.len() > 200].head(3).values
            )

        return patterns

    def _evaluate_datetime(self, col: str) -> Dict:
        """Evaluate datetime column."""
        data = self.df[col].dropna()

        if len(data) == 0:
            return {}

        stats = {
            "min_date": str(data.min()),
            "max_date": str(data.max()),
            "date_range_days": int((data.max() - data.min()).days),
        }

        return stats

    def _save_categories(self, col: str, data: pd.Series) -> None:
        """Save unique categories to file."""
        unique_vals = data.unique()
        filename = self.data_path / f"categories_{col}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Unique values for column: {col}\n")
            f.write(f"Total unique values: {len(unique_vals)}\n")
            f.write(f"Generated: {self.timestamp}\n")
            f.write("=" * 80 + "\n\n")

            # Sort by frequency
            value_counts = data.value_counts()
            for i, (val, count) in enumerate(value_counts.items(), 1):
                percentage = round(count / len(data) * 100, 2)
                f.write(f"{i:4d}. [{count:6d}] {percentage:6.2f}% | {val}\n")

        logger.info(f"Saved categories to: {filename}")

    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        md = []
        md.append("# Data Quality and Statistical Analysis Report\n")
        md.append(f"**Generated:** {self.timestamp}\n\n")

        # Dataset Overview
        md.append("## Dataset Overview\n")
        overview = self.report["dataset_overview"]
        md.append(
            f"- **Total Rows:** {overview['total_rows']:,}\n"
            f"- **Total Columns:** {overview['total_columns']}\n"
            f"- **Memory Usage:** {overview['memory_usage_mb']:.2f} MB\n\n"
        )

        # Data Quality Issues
        md.append("## Data Quality Summary\n")
        quality = self.report["data_quality"]
        md.append(
            f"- **Duplicate Rows:** {quality['duplicate_rows']}\n"
            f"- **Fully Null Columns:** {len(quality['fully_null_columns'])}\n"
        )
        if quality["fully_null_columns"]:
            md.append(
                f"  - {', '.join(quality['fully_null_columns'])}\n"
            )
        md.append("\n")

        # Missing Values Summary
        md.append("### Missing Values by Column\n")
        md.append("| Column | Missing Count | Missing % |\n")
        md.append("|--------|---------------|----------|\n")
        for col, count in quality["missing_values"].items():
            pct = round(count / len(self.df) * 100, 2)
            md.append(f"| {col} | {count} | {pct}% |\n")
        md.append("\n")

        # Column Analysis
        md.append("## Column Analysis\n\n")

        for col, analysis in self.report["columns"].items():
            md.append(f"### {col}\n")
            md.append(f"- **Data Type:** {analysis['dtype']}\n")
            md.append(
                f"- **Non-Null:** {analysis['non_null_count']} "
                f"({100 - analysis['null_percentage']:.2f}%)\n"
            )
            md.append(f"- **Unique Values:** {analysis['unique_count']}\n")

            # Numeric analysis
            if "five_number_summary" in analysis:
                md.append("\n**Five-Number Summary:**\n")
                fns = analysis["five_number_summary"]
                md.append(
                    f"| Min | Q1 | Median | Q3 | Max |\n"
                    f"|-----|-----|--------|-----|-----|\n"
                    f"| {fns['min']:.4f} | {fns['q1']:.4f} | {fns['median']:.4f} "
                    f"| {fns['q3']:.4f} | {fns['max']:.4f} |\n\n"
                )

                md.append(
                    f"- **Mean:** {analysis['mean']:.4f}\n"
                    f"- **Std Dev:** {analysis['std']:.4f}\n"
                    f"- **Skewness:** {analysis['skewness']:.4f}\n"
                    f"- **Kurtosis:** {analysis['kurtosis']:.4f}\n"
                )

                # Outliers
                if "outliers" in analysis:
                    outliers = analysis["outliers"]
                    md.append(
                        f"\n**Outliers (IQR Method):**\n"
                        f"- **Count:** {outliers['count']} ({outliers['percentage']}%)\n"
                        f"- **Bounds:** [{outliers['lower_bound']:.4f}, "
                        f"{outliers['upper_bound']:.4f}]\n"
                    )

                # Potential errors
                if "potential_errors" in analysis:
                    md.append(f"\n**Potential Data Errors:**\n")
                    errors = analysis["potential_errors"]
                    if errors["negative_values"] > 0:
                        md.append(
                            f"- **Negative Values:** {errors['negative_values']}\n"
                        )
                    if errors["zero_values"] > 0:
                        md.append(f"- **Zero Values:** {errors['zero_values']}\n")
                    if errors["extreme_values"] > 0:
                        md.append(
                            f"- **Extreme Values (>5σ):** {errors['extreme_values']}\n"
                        )

            # Text analysis
            if "text_cleaning" in analysis:
                md.append("\n**Text Analysis:**\n")
                md.append(
                    f"- **Min Length:** {analysis['min_length']}\n"
                    f"- **Max Length:** {analysis['max_length']}\n"
                    f"- **Mean Length:** {analysis['mean_length']:.1f}\n"
                    f"- **Empty Strings:** {analysis['empty_strings']}\n"
                )

                cleaning = analysis["text_cleaning"]
                md.append(f"\n**Text Cleaning Comparison:**\n")
                md.append(
                    f"- **Raw vs Lowercase:** {cleaning['case_sensitive_unique']} "
                    f"vs {cleaning['case_insensitive_unique']} unique values\n"
                    f"- **Non-Alphanumeric Rows:** {cleaning['rows_with_non_alphanumeric']} "
                    f"({cleaning['percentage']}%)\n"
                )

                if "suspicious_patterns" in analysis:
                    patterns = analysis["suspicious_patterns"]
                    md.append(f"\n**Suspicious Patterns:**\n")
                    if patterns["very_long_values"] > 0:
                        md.append(
                            f"- **Very Long Values (>200 chars):** "
                            f"{patterns['very_long_values']}\n"
                        )
                    if patterns["contains_numbers"] > 0:
                        md.append(
                            f"- **Contains Numbers:** {patterns['contains_numbers']}\n"
                        )

            md.append("\n---\n\n")

        return "\n".join(md)

    def save_report(self) -> str:
        """Save report to markdown file."""
        report_path = self.data_path / "data_report.md"
        markdown_content = self.generate_markdown_report()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"Report saved to: {report_path}")
        return str(report_path)
