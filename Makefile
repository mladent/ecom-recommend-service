.PHONY: test test-unit test-integration test-coverage test-html test-markers test-verbose test-quick test-failed help

# Colors for output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
NC     := \033[0m # No Color

help:
	@echo "$(BLUE)Testing Commands$(NC)"
	@echo ""
	@echo "$(GREEN)make test$(NC)              - Run all tests with coverage report"
	@echo "$(GREEN)make test-quick$(NC)        - Run tests with minimal output"
	@echo "$(GREEN)make test-verbose$(NC)      - Run tests with full output"
	@echo "$(GREEN)make test-unit$(NC)         - Run unit tests only (fast)"
	@echo "$(GREEN)make test-integration$(NC)  - Run integration tests only"
	@echo "$(GREEN)make test-coverage$(NC)     - Show terminal coverage report"
	@echo "$(GREEN)make test-html$(NC)         - Generate HTML coverage report"
	@echo "$(GREEN)make test-markers$(NC)      - List available test markers"
	@echo "$(GREEN)make test-failed$(NC)       - Re-run only previously failed tests"
	@echo "$(GREEN)make test-llm$(NC)          - Run only LLM-related tests"
	@echo "$(GREEN)make test-llm-off$(NC)      - Run tests with LLM disabled"
	@echo "$(GREEN)make test-llm-on$(NC)       - Run tests with LLM enabled (mocked)"
	@echo "$(GREEN)make test-file FILE=...$(NC) - Run specific test file"
	@echo "$(GREEN)make test-func FUNC=...$(NC) - Run specific test function"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make test-file FILE=tests/test_data_pipeline.py"
	@echo "  make test-func FUNC=TestDataPipelineBasics::test_initialization_defaults"

test:
	@echo "$(BLUE)Running all tests with coverage report...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

test-quick:
	@echo "$(BLUE)Running tests (quick mode)...$(NC)"
	pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=80

test-verbose:
	@echo "$(BLUE)Running tests (verbose mode)...$(NC)"
	pytest tests/ -vv --tb=long --cov=src --cov-report=term-missing

test-unit:
	@echo "$(BLUE)Running unit tests only...$(NC)"
	pytest tests/ -v -m "unit" --cov=src --cov-report=term-missing

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -v -m "integration" --cov=src --cov-report=term-missing

test-coverage:
	@echo "$(BLUE)Showing coverage report...$(NC)"
	pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

test-html:
	@echo "$(BLUE)Generating HTML coverage report...$(NC)"
	pytest tests/ -q --cov=src --cov-report=html
	@echo "$(GREEN)✓ Report generated in $(NC)htmlcov/index.html"

test-markers:
	@echo "$(BLUE)Available test markers:$(NC)"
	pytest --markers | grep "^@pytest.mark"

test-failed:
	@echo "$(BLUE)Re-running previously failed tests...$(NC)"
	pytest tests/ -v --lf --cov=src --cov-report=term-missing

test-llm:
	@echo "$(BLUE)Running LLM-related tests...$(NC)"
	pytest tests/ -v -m "llm" --cov=src --cov-report=term-missing

test-llm-off:
	@echo "$(BLUE)Running tests with LLM disabled...$(NC)"
	ENRICHMENT_ENABLED=false OUTLIER_ENABLED=false CONTEXT_ENABLED=false \
	pytest tests/ -v --cov=src --cov-report=term-missing

test-llm-on:
	@echo "$(BLUE)Running tests with LLM enabled (mocked)...$(NC)"
	ENRICHMENT_ENABLED=true OUTLIER_ENABLED=true CONTEXT_ENABLED=true \
	pytest tests/ -v -m "llm" --cov=src --cov-report=term-missing

test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "$(YELLOW)Usage: make test-file FILE=path/to/test_file.py$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running tests from $(FILE)...$(NC)"
	pytest $(FILE) -v --cov=src --cov-report=term-missing

test-func:
	@if [ -z "$(FUNC)" ]; then \
		echo "$(YELLOW)Usage: make test-func FUNC=TestClass::test_function$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running test function $(FUNC)...$(NC)"
	pytest tests/ -v -k "$(FUNC)" --cov=src --cov-report=term-missing

test-watch:
	@echo "$(BLUE)Running tests in watch mode (requires pytest-watch)...$(NC)"
	ptw tests/ -v --cov=src --cov-report=term-missing

test-parallel:
	@echo "$(BLUE)Running tests in parallel (requires pytest-xdist)...$(NC)"
	pytest tests/ -v -n auto --cov=src --cov-report=term-missing

# Additional test utilities
lint:
	@echo "$(BLUE)Running code linting...$(NC)"
	pylint src/ || true
	flake8 src/ tests/ || true

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/ || true
	isort src/ tests/ || true

test-clean:
	@echo "$(BLUE)Cleaning test artifacts...$(NC)"
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf __pycache__/
	find tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

# Combined workflow
ci: lint test
	@echo "$(GREEN)✓ CI workflow complete$(NC)"

all: clean test-clean test
	@echo "$(GREEN)✓ Full test suite complete$(NC)"
