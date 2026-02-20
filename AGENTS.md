# AI Agents Best Practices Guide

## Overview

Guidelines for AI agents working on software development tasks, focusing on non-obvious patterns, agent-specific workflows, and integration best practices.

---

## 1. Coding Best Practices

### 1.1 Architecture Patterns

**Key Principles:**
- **Separation of Concerns**: Pure functions for transformations, orchestration functions for I/O
- **Factory Pattern**: Use for extensible model creation
- **Configuration Objects**: Use dataclasses with explicit dependencies (avoid global config)
- **Fail Fast**: Validate inputs at function entry with actionable error messages
- **Type Hints**: Add to all function signatures

### 1.2 Error Handling

- Provide actionable error messages (e.g., "No data found. Run: python main.py --prepare")
- Validate at system boundaries (API endpoints, file I/O, model inputs)
- Use graceful degradation with fallback values where appropriate

### 1.3 Function Length and Complexity

**Length Guidelines:**
- **20-50 lines**: Ideal range
- **50-75 lines**: Consider refactoring
- **Over 100 lines**: Strong refactoring signal

**Red Flags Indicating Split Needed:**
- Multiple levels of nested loops/conditionals (>3 levels)
- Many local variables (>7-10)
- Multiple distinct responsibilities
- Hard to name accurately (contains "and", "or", "then")
- Difficult to write concise docstring

**Refactoring Techniques:**
- **Extract Method**: Break into focused helper functions
- **Compose Functions**: Chain transformations with `.pipe()` or function composition
- **Helper Functions**: Extract complex conditionals

```python
# Before: 100+ line function
def process_order(order: dict) -> dict:
    # Validation, pricing, inventory, order creation all in one
    ...

# After: Orchestration + focused functions
def process_order(order: dict) -> dict:
    validate_order(order)
    total = calculate_order_total(order)
    check_inventory_availability(order["items"])
    return create_order_record(order, total)
```

**Acceptable Exceptions:**
- Well-documented complex algorithms (single cohesive algorithm)
- Configuration/setup functions (many options, single logical operation)
- Switch/case routing logic (single responsibility despite length)

---

## 2. Documentation Best Practices

### 2.1 Docstring Essentials

- Document public APIs with Args, Returns, Raises
- Skip obvious getters/setters
- Comment the "why" not the "what"
- Add comments for: complex algorithms, performance optimizations, workarounds, business logic

### 2.2 Documentation File Policy

**Avoid Creating New .md Files**
- **DO NOT** create new markdown documentation files unless explicitly requested
- Documentation belongs in code: docstrings, comments, and README.md
- Exception: Project already has established documentation structure (ARCHITECTURE.md, API.md, etc.)

**Where to Document:**
```
✅ In-code docstrings for functions/classes
✅ README.md for project overview and usage
✅ Inline comments for complex logic
✅ Existing .md files if they already exist

❌ New .md files for every feature
❌ Separate documentation for small changes
❌ Summary documents after implementations
❌ "Change log" markdown files
```

---

## 3. Testing Best Practices

- **Test Pyramid**: 70% unit, 20% integration, 10% e2e
- Test: happy path, edge cases, error conditions
- Use pytest fixtures for reusability
- Use parametrized tests for coverage
- Skip testing: library code, trivial getters/setters

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

### 4.5 Environment Setup

**Virtual Environment Best Practice**
- **Always activate the project's virtual environment** before running code or tests
- This ensures imports, dependencies, and tests align with the project's requirements

**Check for venv:**
```bash
# Look for venv directory
ls -la | grep venv

# If exists, activate it:
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

**Why This Matters for AI Agents:**
- Import validation reflects actual project dependencies
- Test execution uses correct package versions
- Prevents false positives/negatives from system Python packages
- Commands like `pytest` and `python` use project-specific versions
- Ensures `get_python_environment_details` returns accurate information

**Workflow Integration:**
```
Before running any Python commands:
1. Check if venv exists in project
2. Activate venv if present
3. Verify activation: `which python` should point to venv
4. Then proceed with testing/development
```

---

## 5. Common Patterns & Anti-Patterns

### 5.1 DRY Principle Applied

Extract repetitive logic into reusable functions. If you see the same code block repeated with minor variations, extract it into a function that handles all cases.

### 5.2 Type-Safe Returns

Use dataclasses for structured returns instead of dicts - provides type safety and IDE support.

### 5.3 Configuration Injection

Avoid global configuration. Pass config objects explicitly as function parameters for testability and clarity.
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
- [ ] Use relative/project paths wherever possible
- [ ] Add type hints to all functions
- [ ] Validate inputs at boundaries
- [ ] Handle errors with context
- [ ] Add docstrings to public functions

### Post-Implementation
- [ ] No syntax errors (`get_errors()`)
- [ ] Imports are correct
- [ ] Tested with sample data
- [ ] Edge cases considered
- [ ] If same bug failed after 3 fix attempts, undo changes and re-analyze with expanded context before a new approach
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

**Progress Reporting:**
- Report findings during research phase
- Explain key design decisions
- Highlight trade-offs or limitations
- Confirm completion with evidence

**When to Ask for Clarification:**
- Ambiguous requirements with multiple valid interpretations
- Conflicts with existing patterns
- Missing dependencies or data

**When to Proceed Autonomously:**
- Clear implementation path exists
- Following established patterns
- Standard error handling

---

## 9. Version Control Best Practices

**Commit Message Structure:**
```
<type>: <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

**Best Practices:**
- Use descriptive, imperative mood ("Add feature" not "Added feature")
- Keep subject under 72 characters
- Explain what and why, not how
- Reference issue numbers (e.g., "Fixes #123")
- **NO emoticons or emojis**
- NO vague messages ("fix stuff", "updates", "wip")

**Branch Naming:**
- `feature/<description>`, `fix/<description>`, `test/<description>`, `docs/<description>`, `refactor/<description>`
- QA/Integration: `qa`, `staging`, `integration`, `develop`

---

## 10. References & Resources

**Project-Specific:**
- See `ARCHITECTURE.md` for system design
- See `WORKFLOW_DIAGRAMS.md` for process flows
- See `PROJECT_SUMMARY.md` for overview
