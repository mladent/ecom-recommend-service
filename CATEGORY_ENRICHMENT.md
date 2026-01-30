# Category Enrichment Feature

## Overview

The category enrichment feature automatically tags products with fine-grained categories and attributes using LLM-based analysis. This enhances product data with structured information for better grouping, filtering, and recommendation quality.

## Features

- **Multi-field extraction**: Automatically extracts category, material, size, and theme attributes
- **Cache-first strategy**: Reduces LLM API calls by caching enrichment results per description
- **Multi-provider support**: Works with OpenAI, Azure OpenAI, Gemini, Anthropic, and Perplexity
- **Configurable fields**: Customize which attributes to extract via YAML configuration
- **Graceful fallback**: Returns NaN for missing attributes instead of failing

## Configuration

### 1. Enable in `config/settings.yaml`

```yaml
category_enrichment:
  enabled: true
  cache_first: true
  cache_path: data/category_enrichment_cache.json
  fields:
    - category
    - material
    - size
    - theme
```

### 2. Set environment variables in `.env`

```bash
# Enable category enrichment
LLM_CATEGORY_ENRICHMENT_ENABLED=true
LLM_CATEGORY_CACHE_FIRST=true
LLM_CATEGORY_CACHE_PATH=./data/category_enrichment_cache.json

# LLM Provider (reuses normalization settings)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_api_key_here
```

### 3. Customize enrichment fields (optional)

Edit `config/settings.yaml` to add or remove fields:

```yaml
category_enrichment:
  fields:
    - category
    - material
    - size
    - theme
    - color        # Add custom fields
    - style
    - occasion
```

## Usage

### Basic Pipeline Integration

Category enrichment runs automatically during preprocessing:

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()
pipeline.load_raw_data('data/data.csv')

# Enrichment happens here (after normalization, before feature engineering)
processed_data = pipeline.preprocess()

# Access enriched columns
print(processed_data[['Description', 'category', 'material', 'size', 'theme']].head())
```

### Run Example Script

```bash
python examples_category_enrichment.py
```

## How It Works

1. **Preprocessing Flow**:
   - Raw data → Cleaning → Normalization → **Category Enrichment** → Feature Engineering

2. **Cache Strategy**:
   - Uses original (normalized) description as cache key
   - Checks cache before calling LLM API
   - Saves new enrichments to JSON cache file
   - Logs cache hits and misses for monitoring

3. **LLM Prompting**:
   - Sends description with list of fields to extract
   - Requests JSON response format
   - Parses JSON and extracts field values
   - Sets "NaN" for attributes that cannot be determined

4. **Dataframe Integration**:
   - Adds new columns for each enrichment field
   - Maps enrichment results to descriptions
   - Converts "NaN" strings to actual pandas NaN values

## Example Output

```
Product: white hanging heart t-light holder
  Category: Home Decor
  Material: Metal
  Size: Small
  Theme: Hearts

Product: white metal lantern
  Category: Lighting
  Material: Metal
  Size: Medium
  Theme: Classic
```

## Cache Management

The enrichment cache is stored in JSON format:

```json
{
  "white hanging heart t-light holder": {
    "category": "Home Decor",
    "material": "Metal",
    "size": "Small",
    "theme": "Hearts"
  },
  "white metal lantern": {
    "category": "Lighting",
    "material": "Metal",
    "size": "Medium",
    "theme": "Classic"
  }
}
```

### Clear Cache

To force re-enrichment, delete or rename the cache file:

```bash
rm data/category_enrichment_cache.json
```

## Performance Considerations

- **First run**: Calls LLM API for each unique description (can be slow and costly)
- **Subsequent runs**: Uses cache for previously enriched descriptions (fast)
- **Quota limits**: Handles quota exceeded errors gracefully and logs warnings
- **Cost optimization**: Enable `cache_first: true` to minimize API calls

## Provider Configuration

Category enrichment reuses the LLM provider configuration from normalization. All supported providers work:

- **OpenAI**: `LLM_PROVIDER=openai`, requires `OPENAI_API_KEY`
- **Azure OpenAI**: `LLM_PROVIDER=azure`, requires endpoint and deployment
- **Gemini**: `LLM_PROVIDER=gemini`, requires `GEMINI_API_KEY`
- **Anthropic**: `LLM_PROVIDER=anthropic`, requires `ANTHROPIC_API_KEY`
- **Perplexity**: `LLM_PROVIDER=perplexity`, requires `PERPLEXITY_API_KEY`

## Troubleshooting

### Enrichment columns are all NaN

- Check that `LLM_CATEGORY_ENRICHMENT_ENABLED=true` in `.env` or `config/settings.yaml`
- Verify provider API credentials are set correctly
- Check logs for error messages

### LLM quota exceeded

- The system will log a warning and skip remaining enrichment calls
- Enrichment cache is still saved for successfully enriched items
- Re-run later when quota resets

### Unexpected field values

- LLM responses may vary; consider adjusting `LLM_TEMPERATURE` (lower = more deterministic)
- Review and curate cache file manually if needed
- Clear cache to force re-enrichment with updated prompts

## Integration with Recommendations

Enriched categories can be used to:

1. **Filter recommendations** by category or theme
2. **Group similar products** by material or size
3. **Cross-sell** complementary categories
4. **Analyze** product distribution by attributes

Access enriched fields in recommendation logic:

```python
# Filter recommendations by category
recommendations = engine.get_recommendations(basket)
filtered = [r for r in recommendations if r['category'] == 'Home Decor']

# Group by material
from collections import defaultdict
by_material = defaultdict(list)
for item in recommendations:
    by_material[item['material']].append(item)
```
