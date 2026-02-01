# AI Agents Best Practices Guide

## Overview

Guidelines for AI agents working on software development tasks, focusing on non-obvious patterns, agent-specific workflows, and integration best practices.

---

## 1. Coding Best Practices

### 1.1 Architecture Patterns

**Separation of Concerns**
- Pure functions for transformations (easy to test, no side effects)
- Orchestration functions handle I/O and coordination
- Configuration separate from implementation logic

**Example: Testable vs Embedded I/O**
```python
# Hard to test - everything embedded
def process_and_save(filename: str):
    data = pd.read_csv(filename)
    processed = transform(data)
    processed.to_csv("output.csv")

# Better - testable transformation
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return transform(data)
```

**Factory Pattern for Extensibility**
```python
class RecommenderFactory:
    _registry: Dict[str, Type[BaseRecommender]] = {}
    
    @classmethod
    def register(cls, name: str, recommender_class: Type[BaseRecommender]):
        cls._registry[name] = recommender_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseRecommender:
        if name not in cls._registry:
            raise ValueError(f"Unknown recommender: {name}")
        return cls._registry[name](**kwargs)
```

### 1.2 Data Validation Patterns

**Validate at System Boundaries**
- API endpoints: validate incoming requests
- File I/O: validate schema and content
- Model inputs: validate shape, types, ranges

```python
def validate_transaction_data(df: pd.DataFrame) -> bool:
    """Validate transaction data schema and content."""
    required = ["InvoiceNo", "StockCode", "Quantity", "CustomerID"]
    
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing columns: {required}")
    
    if not pd.api.types.is_numeric_dtype(df["Quantity"]):
        raise TypeError("Quantity must be numeric")
    
    # Log warnings for data quality issues
    if df["Quantity"].isna().any():
        logger.warning("Found NaN values in Quantity")
    
    return True
```

**Fail Fast Principle**
```python
def fit_model(transactions: List[List[str]], bundles: List[tuple]):
    # Validate immediately at function entry
    if not transactions:
        raise ValueError("transactions cannot be empty")
    if not bundles:
        raise ValueError("bundles cannot be empty")
    if not self._fitted:
        raise RuntimeError("Engine must be fitted first")
    # ... proceed with logic
```

### 1.3 Configuration Management

**Structured Configuration Objects**
```python
@dataclass
class TrainingConfig:
    n_splits: int = 10
    test_size: float = 0.2
    random_state: int = 42
    enable_cache: bool = True
    cache_dir: str = ".cache"

def train_model(data: pd.DataFrame, config: TrainingConfig):
    """Training with explicit configuration dependency."""
    pass
```

**Environment-Based Configuration**
- Development vs Production settings
- Feature flags for experimental features
- Secrets management (API keys, credentials)

### 1.4 Error Handling Strategies

**Contextual Error Messages**
```python
# Poor
if not data:
    raise ValueError("Invalid data")

# Better - actionable error message
if not data:
    raise ValueError(
        "No data found. Please run: python main.py --prepare"
    )
```

**Graceful Degradation**
```python
def load_cached_results(cache_file: str) -> Optional[dict]:
    """Load cached results with fallback to None."""
    try:
        with open(cache_file) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Cache miss: {e}. Will compute fresh results.")
        return None
```

---

## 2. Documentation Best Practices

### 2.1 Docstring Essentials

**Comprehensive Function Documentation**
```python
def fit_all_with_kfold(
    self,
    transactions: List[List[str]],
    bundles: List[tuple],
    n_splits: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Train all recommenders using k-fold cross-validation.
    
    Args:
        transactions: List of transaction baskets with item IDs
        bundles: List of bundle tuples for training
        n_splits: Number of folds (default: 10)
    
    Returns:
        Dict mapping model names to metrics:
        {'model': {'accuracy': 0.85, 'std_accuracy': 0.02, ...}}
    
    Raises:
        ValueError: If transactions or bundles are empty
    """
```

**When NOT to Document**
- Obvious getters/setters
- Self-explanatory variable names
- Implementation details that may change

### 2.2 Code Comments Philosophy

**Comment the "Why", Not the "What"**
```python
# Good - Explains reasoning
# Sample before bundle generation to avoid O(n²) complexity
# Bundle generation is the bottleneck operation
if quick_mode:
    sample_size = max(25, len(transactions) // 200)

# Avoid - States the obvious
# Increment counter by 1
counter += 1
```

**When to Add Comments**
- Complex algorithms with non-obvious logic
- Performance optimizations
- Workarounds for library limitations
- Business domain knowledge
- Security considerations

### 2.3 README Structure for Projects

```markdown
# Project Name

## Quick Start (30 seconds)
```bash
pip install -r requirements.txt
python main.py --prepare
python main.py --api
```

## Features
- What it does (3-5 bullet points)
- Key capabilities

## Usage Examples
```python
# Minimal working example
```

## Configuration
- Environment variables
- Config file options

## Development
- Running tests
- Code style
- Contributing
```

---

## 3. Validation Best Practices

### 3.1 Testing Strategy

**Test Pyramid**
1. **Unit Tests (70%)**: Pure functions, isolated components
2. **Integration Tests (20%)**: Component interactions
3. **End-to-End Tests (10%)**: Full workflows

**What to Test**
- ✅ Happy path with valid inputs
- ✅ Edge cases (empty, null, boundary values)
- ✅ Error conditions (invalid inputs, missing data)
- ✅ Integration points between components
- ❌ Don't test library code
- ❌ Don't test trivial getters/setters

### 3.2 Pytest Patterns

**Fixtures for Reusability**
```python
@pytest.fixture
def sample_data():
    """Reusable test data across tests."""
    transactions = [["A", "B"], ["B", "C"]]
    bundles = [("A", "B"), ("B", "C")]
    return transactions, bundles

def test_kfold_validation(sample_data):
    transactions, bundles = sample_data
    # Test logic here
```

**Parametrized Tests for Coverage**
```python
@pytest.mark.parametrize("threshold,expected_count", [
    (0.1, 10),
    (0.5, 5),
    (0.9, 1),
])
def test_threshold_filtering(threshold, expected_count):
    results = filter_by_threshold(data, threshold)
    assert len(results) == expected_count
```

### 3.3 Performance Validation

**Timing Critical Paths**
```python
@contextmanager
def timer(name: str):
    start = time.time()
    yield
    logger.info(f"{name}: {time.time() - start:.2f}s")

with timer("K-Fold Training"):
    metrics = engine.fit_all_with_kfold(txns, bundles)
```

**Memory Profiling for Large Datasets**
```python
import tracemalloc

tracemalloc.start()
result = expensive_operation(large_data)
current, peak = tracemalloc.get_traced_memory()
logger.info(f"Peak memory: {peak / 1024**2:.1f} MB")
tracemalloc.stop()
```

---

## 4. AI Agent Workflow Best Practices

### 4.1 Context Gathering Protocol

**Before Writing Code - Research Phase**
1. `semantic_search("relevant concept")` - Find similar implementations
2. `read_file()` on key files - Understand existing patterns
3. `grep_search("function_name")` - Find all usages
4. `list_code_usages("ClassName")` - See how it's used

**Example Workflow:**
```
Task: Add summary tables to k-fold validation example

1. semantic_search("model evaluation metrics aggregation")
   → Found: src/recommendation_engine.py has fit_all_with_kfold()

2. read_file(recommendation_engine.py, lines with metrics)
   → Metrics structure: {model: {metric: value, std_metric: value}}

3. grep_search("format.*table") 
   → Check if table formatting exists elsewhere

4. Now ready to implement with full context
```

### 4.2 Incremental Implementation

**Build → Test → Enhance Cycle**
```
Step 1: Minimal implementation
  ├─ Create function stub with type hints
  ├─ Add basic logic
  └─ Test with get_errors()

Step 2: Core functionality
  ├─ Implement main algorithm
  ├─ Add input validation
  └─ Test with sample data

Step 3: Enhancement
  ├─ Add error handling
  ├─ Optimize performance
  └─ Add comprehensive tests
```

### 4.3 Verification Checklist

**After Implementation - Always Run:**
- [ ] `get_errors(filePath)` - No syntax errors
- [ ] Check imports are available
- [ ] Verify function signatures match usage
- [ ] Test with realistic sample data
- [ ] Check edge cases (empty, None, invalid)

**Integration Verification:**
- [ ] New code follows existing patterns
- [ ] Naming conventions match codebase
- [ ] Configuration is consistent
- [ ] Error handling aligns with project style

### 4.4 Multi-File Changes

**Dependency Order Matters**
```
When modifying multiple files:
1. Update base classes/interfaces first
2. Update implementations second
3. Update call sites third
4. Update tests last

Use multi_replace_string_in_file for atomicity
```

---

## 5. Common Patterns & Anti-Patterns

### 5.1 DRY Principle Applied

**Before - Repetitive Code:**
```python
if models_config["naive_bayes"] and "naive_bayes" in metrics:
    nb_acc = metrics["naive_bayes"]["accuracy"]
    # ... 15 lines of comparison logic

if models_config["svm"] and "svm" in metrics:
    svm_acc = metrics["svm"]["accuracy"]
    # ... same 15 lines duplicated
```

**After - Extracted Function:**
```python
def format_comparison_table(
    metrics1: dict, 
    metrics2: dict, 
    enabled_models: dict
) -> str:
    """Single function handles all models."""
    # Implement once, use for all models
```

### 5.2 Result Objects Over Dicts

**Type-Safe Returns**
```python
@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    std_accuracy: Optional[float] = None
    n_splits: int = 1
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}

# Usage - type hints provide intellisense
result: EvaluationResult = evaluate_model(data)
print(result.accuracy)  # IDE knows this is float
```

### 5.3 Configuration Injection

**Avoid Global Configuration**
```python
# Anti-pattern
config = load_config()  # Global

def process():
    if config["enable_cache"]:  # Implicit dependency
        ...

# Better - explicit dependencies
def process(config: Config):
    if config.enable_cache:
        ...
```

---

## 6. Project-Specific Guidelines

### 6.1 This E-Commerce Recommendation Project

**Key Patterns Used:**
- Engine pattern: `BundleRecommendationEngine` orchestrates multiple models
- Pipeline pattern: `DataPipeline` for ETL operations
- Configuration: `config/settings.yaml` for prompts and schemas
- Caching: JSON files for expensive LLM operations
- Examples: Separate files for different use cases

**When Adding Features:**
1. Check if similar pattern exists (cross-sell, category enrichment)
2. Follow the example file naming: `examples_*.py`
3. Use existing caching mechanisms where appropriate
4. Add documentation to relevant `*_README.md` files

### 6.2 File Organization

```
src/           # Core library code
  ├─ api.py           # FastAPI endpoints
  ├─ recommendation_engine.py  # Main models
  └─ data_pipeline.py # Data processing

config/        # Configuration
  ├─ settings.yaml
  └─ prompts/         # LLM prompts

examples_*.py  # Runnable examples
tests/         # Test suite
```

---

## 7. Summary Checklist

### Pre-Implementation
- [ ] Searched for similar implementations
- [ ] Read relevant existing code
- [ ] Understand data structures and types
- [ ] Know integration points

### During Implementation
- [ ] Follow existing naming conventions
- [ ] Add type hints to all functions
- [ ] Validate inputs at boundaries
- [ ] Handle errors with context
- [ ] Add docstrings to public functions

### Post-Implementation
- [ ] No syntax errors (`get_errors()`)
- [ ] Imports are correct
- [ ] Tested with sample data
- [ ] Edge cases considered
- [ ] Follows DRY principle
- [ ] Documentation updated if needed

### Quality Gates
- [ ] Code is readable (descriptive names)
- [ ] Functions are focused (single responsibility)
- [ ] Error messages are actionable
- [ ] No obvious performance issues
- [ ] Tests cover main scenarios

---

## 8. Agent Communication Best Practices

### 8.1 Status Updates

**Progress Reporting:**
- Report what was found during research phase
- Explain key design decisions
- Highlight any trade-offs or limitations
- Confirm completion with evidence (test output, error-free validation)

### 8.2 Problem Escalation

**When to Ask for Clarification:**
- Ambiguous requirements with multiple valid interpretations
- Conflicts with existing patterns
- Missing dependencies or data
- Performance vs accuracy trade-offs

**When to Proceed Autonomously:**
- Clear implementation path exists
- Following established patterns
- Standard error handling
- Routine refactoring

---

## 9. Testing Best Practices

### 9.1 Test Organization

```python
class TestBundleRecommendationEngine:
    """Group related tests in classes."""
    
    @pytest.fixture
    def engine(self):
        """Shared setup."""
        return BundleRecommendationEngine()
    
    def test_empty_input_raises_error(self, engine):
        """Test error conditions."""
        with pytest.raises(ValueError, match="cannot be empty"):
            engine.fit_all_with_random_split([], [])
    
    def test_kfold_metrics_structure(self, engine):
        """Test return value structure."""
        metrics = engine.fit_all_with_kfold(txns, bundles)
        assert "accuracy" in metrics["model_name"]
        assert "std_accuracy" in metrics["model_name"]
```

### 9.2 Integration Tests

**Test Real Workflows:**
```python
def test_end_to_end_recommendation_workflow():
    """Test complete pipeline from data to prediction."""
    # Setup
    pipeline = DataPipeline()
    pipeline.load_raw_data()
    
    # Process
    transactions = pipeline.create_transaction_baskets()
    bundles = pipeline.generate_product_bundles()
    
    # Train
    engine = BundleRecommendationEngine()
    engine.add_recommender("nb", NaiveBayesBundleRecommender())
    metrics = engine.fit_all_with_random_split(
        [list(t) for t in transactions["Items"]], 
        [tuple(b) for b in bundles]
    )
    
    # Validate
    assert metrics["nb"]["accuracy"] > 0.5
    
    # Predict
    recs = engine.recommend_bundles(["item1", "item2"])
    assert "bundles" in recs
    assert "confidence" in recs
```

---

## 10. References & Resources

**Python Standards:**
- PEP 8: Style Guide
- PEP 257: Docstring Conventions
- PEP 484: Type Hints

**Testing:**
- pytest documentation
- unittest.mock for mocking

**Design Patterns:**
- Factory Pattern for model creation
- Strategy Pattern for interchangeable algorithms
- Pipeline Pattern for data transformations

**Project-Specific:**
- See `ARCHITECTURE.md` for system design
- See `WORKFLOW_DIAGRAMS.md` for process flows
- See `PROJECT_SUMMARY.md` for overview
