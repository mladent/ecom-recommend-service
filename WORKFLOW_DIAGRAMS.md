# System Workflow Diagrams

## 1. Data Processing Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    KAGGLE ECOMMERCE DATASET                     │
│         (https://www.kaggle.com/carrie1/ecommerce-data)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
                    ┌────────────────────┐
                    │ Download Kaggle    │
                    │ Dataset (CSV)      │
                    │ data_pipeline.py   │
                    │ .download_kaggle_  │
                    │  data()            │
                    └────────┬───────────┘
                             │
                             ↓
                    ┌────────────────────────────┐
                    │ Load CSV                   │
                    │ .load_raw_data()           │
                    │                            │
                    │ Data Shape:                │
                    │ • ≈540K rows               │
                    │ • 8 columns                │
                    │ • Multiple products/invoice│
                    └────────┬───────────────────┘
                             │
                             ↓
            ┌────────────────────────────────────┐
            │     DATA EXPLORATION PHASE         │
            │                                    │
            │ • Row & column counts              │
            │ • Data types                       │
            │ • Missing values                   │
            │ • Date ranges                      │
            │ • Unique customers/products        │
            │ • Basic statistics                 │
            └────────┬───────────────────────────┘
                     │
                     ↓
           ┌─────────────────────────────────────┐
           │    DATA CLEANING & PREPROCESSING    │
           │   .preprocess()                     │
           │                                     │
           │ ✓ Remove null CustomerIDs          │
           │ ✓ Remove null product descriptions │
           │ ✓ Filter negative quantities       │
           │ ✓ Filter zero/negative prices      │
           │ ✓ Convert dates to datetime        │
           │ ✓ Standardize descriptions (lower) │
           │ ✓ Remove duplicate transactions    │
           │ ✓ Calculate transaction values     │
           └────────┬──────────────────────────┘
                    │
                    ↓
          ┌────────────────────────────────┐
          │ CREATE TRANSACTION BASKETS     │
          │ .create_transaction_baskets()  │
          │                                │
          │ Group items by InvoiceNo       │
          │ Filter baskets: size >= 2      │
          │                                │
          │ Output: varies by filters      │
          │ Average: depends on data       │
          └────────┬─────────────────────┘
                   │
                   ↓
         ┌──────────────────────────────────┐
         │ BUNDLE DISCOVERY (APRIORI)       │
         │ .generate_product_bundles()      │
         │                                  │
         │ Input Parameters:                │
         │ • MIN_SUPPORT: 0.02 (2%)         │
         │ • MIN_CONFIDENCE: 0.5 (50%)      │
         │ • MAX_BUNDLE_SIZE: 5             │
         │                                  │
         │ Process:                         │
         │ 1. Count item frequencies        │
         │ 2. Filter by support threshold   │
         │ 3. Generate k-itemsets           │
         │ 4. Filter by confidence          │
         │                                  │
         │ Output: varies by thresholds     │
         │ Avg size: depends on params      │
         └────────┬──────────────────────┘
                  │
                  ↓
        ┌────────────────────────────┐
        │ CACHE PROCESSED DATA       │
        │ .save_processed_data()     │
        │                            │
        │ Serialized to:             │
        │ data/processed_data.pkl    │
        │                            │
        │ Includes:                  │
        │ • Cleaned transactions     │
        │ • Transaction baskets      │
        │ • Discovered bundles       │
        │ • Timestamp                │
        └────────┬──────────────────┘
                 │
                 ↓
     ┌────────────────────────────────┐
     │ READY FOR MODEL TRAINING       │
     │                                │
      │ Data statistics:               │
      │ • Clean transactions: varies   │
      │ • Usable baskets: varies       │
      │ • Frequent items: varies       │
      │ • Product bundles: varies      │
     └────────────────────────────────┘
```

## 2. Model Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│         PROCESSED DATA (from pipeline above)                    │
│ • Transactions (item lists)                                     │
│ • Baskets (customer purchases)                                  │
│ • Bundles (frequent itemsets)                                   │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────────┐
        │ FEATURE ENCODING PHASE       │
        │                              │
        │ Input: List of product names │
        │ Process:                     │
        │ • MultiLabelBinarizer       │
        │ • Create sparse matrix      │
        │ • Each row = transaction    │
        │ • Each col = product        │
        │ • Cell = 1 if present, 0 if │
        │                              │
        │ Output shape: (N, M)        │
        │ N = transactions            │
        │ M = unique products         │
        └──────────┬───────────────────┘
                   │
        ┌──────────┴──────────┬─────────────────┐
        │                     │                 │
        ↓                     ↓                 ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  NAIVE BAYES 1   │  │  NAIVE BAYES 2   │  │       SVM        │
│                  │  │                  │  │                  │
│  Multinomial NB  │  │   Gaussian NB    │  │ (RBF Kernel)     │
│                  │  │                  │  │                  │
│  Training:       │  │  Training:       │  │ Pre-process:     │
│  • Input: X      │  │  • Convert to    │  │ • StandardScaler │
│  • Target: y     │  │    dense matrix  │  │ • Normalize      │
│  • Direct fit()  │  │  • Fit()         │  │                  │
│                  │  │                  │  │ Training:        │
│  Output:         │  │  Output:         │  │ • Input: X_scaled│
│  • Trained model │  │  • Trained model │  │ • Target: y      │
│  • P(bundle)     │  │  • P(bundle)     │  │ • Find hyperplane│
└──────┬───────────┘  └──────┬───────────┘  │                  │
       │                     │              │ Output:          │
       │                     │              │ • Support vectors│
       │                     │              │ • Trained model  │
       │                     │              └──────┬───────────┘
       │                     │                     │
       └─────────┬───────────┴─────────────────────┘
                 │
                 ↓
      ┌────────────────────────────────┐
      │ DATA SPLITTING (BEFORE TRAINING)│
      │                                │
      │ • RandomSplit (0.8/0.2)        │
      │ • K-Fold (k=10)                │
      │ • Random state: 42             │
      │                                │
      │ Output:                        │
      │ • Train/test splits per fold   │
      │ • X_train, X_test, y_train, y_test
      └────────┬─────────────────────┘
                 │
                 ↓
        ┌────────────────────────────────┐
        │    MODEL EVALUATION            │
        │                                │
        │ For each model:                │
        │ • y_pred = predict(X_test)     │
        │ • Accuracy                     │
        │ • Precision                    │
        │ • Recall                       │
        │ • F1-Score                     │
        │                                │
        │ Store metrics for logging      │
        └────────┬─────────────────────┘
                 │
                 ↓
        ┌────────────────────────────────┐
        │ MODEL SERIALIZATION            │
        │ .save_model()                  │
        │                                │
        │ Save to disk:                  │
        │ models/recommendation_         │
        │ engine.pkl                     │
        │                                │
        │ Contains:                      │
        │ • All 3 trained models         │
        │ • Metadata                     │
        │ • Timestamp                    │
        │ • Bundles reference            │
        └────────┬─────────────────────┘
                 │
                 ↓
     ┌──────────────────────────────────┐
     │ READY FOR INFERENCE              │
     │                                  │
     │ ✓ Naive Bayes (Multinomial)      │
     │ ✓ Naive Bayes (Gaussian)         │
     │ ✓ SVM (RBF)                      │
     │ ✓ Ensemble (averaging)           │
     └──────────────────────────────────┘
```

## 3. Inference (Recommendation) Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│           NEW CUSTOMER TRANSACTION                             │
│         (Items added to shopping cart)                         │
│                                                                │
│  Example:                                                      │
│  ["white hanging heart holder",                               │
│   "regency cakestand 3 tier",                                 │
│   "assorted colour teardrop heart"]                           │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
         ┌────────────────────────────────┐
         │ LOAD TRAINED MODELS            │
         │ engine.load_model()            │
         │                                │
         │ Load from disk:                │
         │ models/recommendation_         │
         │ engine.pkl                     │
         │                                │
         │ Restores:                      │
         │ • Naive Bayes models (2)       │
         │ • SVM models (up to 3 kernels) │
         │ • Feature encoders             │
         │ • Bundle catalog               │
         └────────┬──────────────────────┘
                  │
                  ↓
      ┌───────────────────────────────────┐
      │ FEATURE ENCODING (Same as training)
      │                                   │
      │ Input: Customer transaction       │
      │ Process:                          │
      │ • Use stored MultiLabelBinarizer │
      │ • Transform to binary vector     │
      │ • Maintain same feature space    │
      │                                   │
      │ Output: Sparse matrix (1, M)     │
      │ M = unique products from training│
      └───────┬──────────────────────────┘
              │
        ┌─────┴─────┬──────────┬──────────┐
        │           │          │          │
        ↓           ↓          ↓          ↓
    ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐
    │  NB    │ │  NB    │ │  SVM   │ │Ensemble │
    │Model 1 │ │Model 2 │ │(RBF)   │ │(All)    │
    │        │ │        │ │        │ │         │
    │Predict:│ │Predict:│ │Predict:│ │Predict: │
    │Proba   │ │Proba   │ │Proba   │ │Average  │
    │        │ │        │ │        │ │         │
    │Output: │ │Output: │ │Output: │ │Output:  │
    │0.72    │ │0.68    │ │0.75    │ │0.7166   │
    └────┬───┘ └────┬───┘ └────┬───┘ └────┬────┘
         │           │          │          │
         └─────┬─────┴──────────┴──────────┘
               │
               ↓
      ┌────────────────────────────────┐
      │ CONFIDENCE THRESHOLD CHECK      │
      │                                │
      │ Threshold: 0.5                 │
      │ Ensemble score: 0.7166         │
      │                                │
      │ Is 0.7166 >= 0.5?              │
      │ YES ✓ → Return recommendations │
      │ NO → Empty recommendations    │
      └────────┬─────────────────────┘
               │
               ↓ (if passed threshold)
     ┌──────────────────────────────────┐
     │ FIND APPLICABLE BUNDLES          │
     │                                  │
     │ For each bundle in catalog:      │
     │ • Check overlap with customer    │
     │   items                          │
     │ • Keep bundles with overlap > 0  │
     │ • Rank by match strength         │
     │                                  │
     │ Output: Top 5 matching bundles   │
     └────────┬─────────────────────────┘
              │
              ↓
    ┌─────────────────────────────────┐
    │ CROSS-SELL RECOMMENDATIONS      │
    │ (Products not in transaction)   │
    │                                 │
    │ For each bundle:                │
    │ • Find new products in bundle   │
    │ • Calculate affinity score      │
    │ • Score = overlap/bundle_size   │
    │                                 │
    │ Aggregate across bundles:       │
    │ • Sum scores per product        │
    │ • Sort by total score           │
    │ • Return top 5                  │
    │                                 │
    │ Examples:                        │
    │ 1. candle (2.3)                │
    │ 2. ornament (1.8)              │
    │ 3. decoration (1.5)            │
    │ 4. centerpiece (1.2)           │
    │ 5. holder (0.9)                │
    └────────┬──────────────────────┘
             │
             ↓
  ┌────────────────────────────────────┐
  │ RETURN RECOMMENDATIONS             │
  │                                    │
  │ Output object:                     │
  │ {                                  │
  │   "transaction": [...items...],    │
  │   "bundles": [                     │
  │     (prod1, prod2, prod3),         │
  │     (prod2, prod4, prod5),         │
  │     ...                            │
  │   ],                               │
  │   "confidence": 0.7166,            │
  │   "recommender": "ensemble"        │
  │ }                                  │
  │                                    │
  │ Also available:                    │
  │ • Individual model predictions     │
  │ • Cross-sell products              │
  │ • Cross-sell scores                │
  └────────────────────────────────────┘
```

## 4. System Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │   APPLICATION LAYER             │
                    │  (REST API, Web UI)            │
                    └──────────────┬────────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          │                                                 │
     ┌────▼─────────────────────┐  ┌─────────────────────▼─────┐
     │  RECOMMENDATION ENGINE    │  │  DATA PIPELINE           │
     │                          │  │                          │
     │ • Naive Bayes Models     │  │ • Load raw data          │
     │ • SVM Models             │  │ • Clean & preprocess     │
     │ • Ensemble Manager       │  │ • Create baskets         │
     │ • Cross-sell Engine      │  │ • Generate bundles       │
     │ • Model persistence      │  │ • Save cache             │
     └────┬──────────────────────┘  └────────┬────────────────┘
          │                                   │
          └─────────────────┬─────────────────┘
                           │
                    ┌──────▼──────────┐
                    │  CONFIG LAYER   │
                    │                │
                    │ • Environment   │
                    │ • YAML settings │
                    │ • Defaults      │
                    │ • Validation    │
                    └─────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
   ┌────▼────────────────────┐  ┌───────────▼──────┐
   │  DATA STORAGE           │  │  MODEL STORAGE   │
   │                        │  │                  │
   │ • CSV (raw)            │  │ • PKL (trained)  │
   │ • PKL (processed)      │  │ • Scalers        │
   │ • Statistics           │  │ • Encoders       │
   │                        │  │ • Metadata       │
   └────────────────────────┘  └──────────────────┘
```

## 5. Request/Response Flow

```
Customer Request
      │
      ↓
┌─────────────────────────────────────┐
│  Customer Transaction               │
│  ["item_a", "item_b", "item_c"]    │
└────────┬────────────────────────────┘
         │
         ↓
   ┌──────────────────┐
   │ Load Models      │ (from pickle cache)
   │ Load Encoders    │
   │ Load Bundles     │
   └────────┬─────────┘
            │
            ↓
   ┌──────────────────────┐
   │ Encode Transaction   │
   │ (to binary vector)   │
   └────────┬─────────────┘
            │
      ┌─────┴──────┐
      ↓            ↓
  ┌────────┐  ┌─────────┐
  │ NB1    │  │ SVM/NB2 │
  │Proba   │  │ Proba   │
  │ 0.72   │  │  0.73   │
  └────┬───┘  └────┬────┘
       │           │
       └─────┬─────┘
             │
             ↓
       ┌──────────────┐
       │ Average: 0.72│  (or specific model)
       │ ✓ > threshold│
       └──────┬───────┘
              │
              ↓
       ┌─────────────────────────┐
       │ Find matching bundles   │
       │ (where overlap > 0)     │
       │ Top 5 results           │
       └──────┬──────────────────┘
              │
              ↓
      ┌──────────────────────────┐
      │ Create response:          │
      │ {                        │
      │   "bundles": [...],      │
      │   "confidence": 0.72,    │
      │   "cross_sell": [...],   │
      │   "algorithm": "ensemble"│
      │ }                        │
      └──────┬───────────────────┘
             │
             ↓
     Return to Client/API
```

These diagrams illustrate how the recommendation system works from data ingestion through model training to final recommendations delivery.
