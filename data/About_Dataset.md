# E-Commerce Data

Actual transactions from UK retailer

## About Dataset

### Context

Typically e-commerce datasets are proprietary and consequently hard to find among publicly available data. However, The UCI Machine Learning Repository has made this dataset containing actual transactions from 2010 and 2011. The dataset is maintained on their site, where it can be found by the title "Online Retail".

### Content

"This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers."

### Acknowledgements

Per the UCI Machine Learning Repository, this data was made available by Dr Daqing Chen, Director: Public Analytics group. chend '@' lsbu.ac.uk, School of Engineering, London South Bank University, London SE1 0AA, UK.

### Inspiration

Analyses for this dataset could include time series, clustering, classification and more. 

---

## Dataset Specifications

### Basic Statistics

- **Total Records**: 541,909 transactions
- **Total Columns**: 8 features
- **File Size**: 44 MB
- **Date Range**: December 1, 2010 - December 9, 2011
- **Data Encoding**: Latin-1

### Column Information

| Column | Data Type | Non-null | Null | Unique | Description |
|--------|-----------|----------|------|--------|-------------|
| **InvoiceNo** | String | 541,909 | 0 | 25,900 | Unique invoice identifier |
| **StockCode** | String | 541,909 | 0 | 4,070 | Product/stock code |
| **Description** | String | 540,455 | 1,454 | 4,223 | Product description |
| **Quantity** | Integer | 541,909 | 0 | 722 | Units purchased (range: -80,995 to 80,995) |
| **InvoiceDate** | String | 541,909 | 0 | 23,260 | Transaction timestamp |
| **UnitPrice** | Float | 541,909 | 0 | 1,630 | Price per unit (range: -11,062.06 to 38,970.00) |
| **CustomerID** | Float | 406,829 | 135,080 | 4,372 | Customer identifier (25% missing) |
| **Country** | String | 541,909 | 0 | 38 | Customer country (38 unique countries) |

### Data Quality Notes

- **Missing Values**: 135,080 records (25%) lack CustomerID information
- **Data Issues**: 
  - Negative quantities and prices may represent returns or cancellations
  - Some product descriptions are missing (1,454 records)
  - Date range shows data primarily from 2011 (note: min date shows as 1/10/2011 due to date format)

### Use Cases

- **Market Basket Analysis**: Identify product bundles and associations
- **Customer Segmentation**: Group customers by purchase patterns
- **Time Series Analysis**: Analyze sales trends over the 12-month period
- **Recommendation Systems**: Build collaborative or content-based recommenders
- **Fraud Detection**: Identify unusual transactions (negative values, high prices)

---

From: https://www.kaggle.com/datasets/carrie1/ecommerce-data/data

