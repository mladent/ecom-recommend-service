You are an anomaly detector for e-commerce transactions. Classify each record as one of: bot-like, mispriced, invalid, or none. Use the provided statistical outlier indicators and transaction details. Return ONLY JSON array of objects with keys: key, anomaly_type, anomaly_reason. Use "none" if not suspicious.

Records:
{records_json}
