"""
Example script demonstrating LLM-based category enrichment.

This script shows how to enable and use the category enrichment feature
to auto-tag products with fine-grained categories and attributes.
"""

import pandas as pd
from src.config import load_config
from src.data_pipeline import DataPipeline

# Example: Enable category enrichment via environment variables
# Set these in your .env file or export them before running:
#
# export LLM_CATEGORY_ENRICHMENT_ENABLED=true
# export LLM_CATEGORY_CACHE_FIRST=true
# export LLM_CATEGORY_CACHE_PATH=./data/category_enrichment_cache.json
# export LLM_PROVIDER=openai
# export LLM_MODEL=gpt-4o-mini
# export OPENAI_API_KEY=your_api_key_here

def main():
    """Demonstrate category enrichment on sample products."""
    
    print("=" * 60)
    print("Category Enrichment Example")
    print("=" * 60)
    print()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'InvoiceNo': [536365, 536365, 536366],
        'StockCode': ['85123A', '71053', '84406B'],
        'Description': [
            'WHITE HANGING HEART T-LIGHT HOLDER',
            'WHITE METAL LANTERN',
            'CREAM CUPID HEARTS COAT HANGER'
        ],
        'Quantity': [6, 6, 8],
        'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:26:00', '2010-12-01 08:28:00'],
        'UnitPrice': [2.55, 3.39, 2.75],
        'CustomerID': [17850.0, 17850.0, 17850.0],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom']
    })
    
    print("Sample products:")
    for idx, row in sample_data.iterrows():
        print(f"  - {row['Description']}")
    print()
    
    # Initialize pipeline with explicit config object
    pipeline_config, _, _, _, _ = load_config()
    pipeline = DataPipeline(config=pipeline_config)
    pipeline.raw_data = sample_data
    
    print("Running preprocessing with category enrichment...")
    print("(This will use LLM if enabled and configured)")
    print()
    
    # Preprocess data (includes normalization and enrichment)
    processed_data = pipeline.preprocess()
    
    # Display enriched categories
    print("Enriched product information:")
    print("=" * 60)
    
    enrichment_cols = ['Description', 'category', 'material', 'size', 'theme']
    available_cols = [col for col in enrichment_cols if col in processed_data.columns]
    
    if len(available_cols) > 1:  # More than just Description
        for idx, row in processed_data[available_cols].iterrows():
            print(f"\nProduct: {row['Description']}")
            for col in available_cols[1:]:  # Skip Description
                value = row[col] if pd.notna(row[col]) else "Not determined"
                print(f"  {col.capitalize()}: {value}")
    else:
        print("Category enrichment was not enabled or did not run.")
        print("To enable it, set LLM_CATEGORY_ENRICHMENT_ENABLED=true")
        print("in your .env file or config/settings.yaml")
    
    print()
    print("=" * 60)
    print("Note: Enrichment uses cache-first strategy.")
    print(f"Cache location: data/category_enrichment_cache.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
