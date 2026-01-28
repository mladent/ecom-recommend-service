#!/bin/bash

# Query script for data.csv
# Usage: ./query_data.sh [command] [options]

DATA_FILE="data/data.csv"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found!"
    exit 1
fi

# Function to display usage
usage() {
    cat << EOF
Usage: ./query_data.sh [COMMAND] [OPTIONS]

COMMANDS:
  head [N]                  Show first N rows (default: 10)
  tail [N]                  Show last N rows (default: 10)
  count                     Count total number of rows
  columns                   Show column names
  customer [ID]             Filter by CustomerID
  country [NAME]            Filter by Country
  invoice [NO]              Filter by InvoiceNo
  product [CODE]            Filter by StockCode
  search [TEXT]             Search for text in Description
  date [DATE]               Filter by InvoiceDate (format: M/D/YYYY)
  stats                     Show basic statistics
  unique [COLUMN]           Show unique values for a column
  top-products [N]          Show top N products by quantity (default: 10)
  top-customers [N]         Show top N customers by transactions (default: 10)
  top-countries [N]         Show top N countries by transactions (default: 10)

EXAMPLES:
  ./query_data.sh head 20
  ./query_data.sh customer 17850
  ./query_data.sh country "United Kingdom"
  ./query_data.sh search "HEART"
  ./query_data.sh unique Country
  ./query_data.sh top-products 15

EOF
}

# Get command
COMMAND=${1:-help}

case "$COMMAND" in
    help|--help|-h)
        usage
        ;;
    
    head)
        N=${2:-10}
        head -n $((N + 1)) "$DATA_FILE" | column -t -s ','
        ;;
    
    tail)
        N=${2:-10}
        (head -n 1 "$DATA_FILE"; tail -n $N "$DATA_FILE") | column -t -s ','
        ;;
    
    count)
        TOTAL=$(wc -l < "$DATA_FILE")
        echo "Total rows (including header): $TOTAL"
        echo "Data rows: $((TOTAL - 1))"
        ;;
    
    columns)
        head -n 1 "$DATA_FILE" | tr ',' '\n' | nl
        ;;
    
    customer)
        if [ -z "$2" ]; then
            echo "Error: Please provide CustomerID"
            exit 1
        fi
        (head -n 1 "$DATA_FILE"; grep ",$2," "$DATA_FILE") | column -t -s ','
        ;;
    
    country)
        if [ -z "$2" ]; then
            echo "Error: Please provide Country name"
            exit 1
        fi
        (head -n 1 "$DATA_FILE"; grep "$2" "$DATA_FILE") | column -t -s ','
        ;;
    
    invoice)
        if [ -z "$2" ]; then
            echo "Error: Please provide InvoiceNo"
            exit 1
        fi
        (head -n 1 "$DATA_FILE"; grep "^$2," "$DATA_FILE") | column -t -s ','
        ;;
    
    product)
        if [ -z "$2" ]; then
            echo "Error: Please provide StockCode"
            exit 1
        fi
        (head -n 1 "$DATA_FILE"; awk -F',' -v code="$2" '$2 == code' "$DATA_FILE") | column -t -s ','
        ;;
    
    search)
        if [ -z "$2" ]; then
            echo "Error: Please provide search text"
            exit 1
        fi
        (head -n 1 "$DATA_FILE"; grep -i "$2" "$DATA_FILE") | column -t -s ','
        ;;
    
    date)
        if [ -z "$2" ]; then
            echo "Error: Please provide date (format: M/D/YYYY)"
            exit 1
        fi
        (head -n 1 "$DATA_FILE"; grep "$2" "$DATA_FILE") | column -t -s ','
        ;;
    
    stats)
        TOTAL=$(wc -l < "$DATA_FILE")
        DATA_ROWS=$((TOTAL - 1))
        CUSTOMERS=$(tail -n +2 "$DATA_FILE" | cut -d',' -f7 | sort -u | wc -l)
        PRODUCTS=$(tail -n +2 "$DATA_FILE" | cut -d',' -f2 | sort -u | wc -l)
        COUNTRIES=$(tail -n +2 "$DATA_FILE" | cut -d',' -f8 | sort -u | wc -l)
        INVOICES=$(tail -n +2 "$DATA_FILE" | cut -d',' -f1 | sort -u | wc -l)
        
        echo "=== Data Statistics ==="
        echo "Total records: $DATA_ROWS"
        echo "Unique customers: $CUSTOMERS"
        echo "Unique products: $PRODUCTS"
        echo "Unique countries: $COUNTRIES"
        echo "Unique invoices: $INVOICES"
        ;;
    
    unique)
        if [ -z "$2" ]; then
            echo "Error: Please provide column name"
            echo "Available columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country"
            exit 1
        fi
        
        case "$2" in
            InvoiceNo) COL=1 ;;
            StockCode) COL=2 ;;
            Description) COL=3 ;;
            Quantity) COL=4 ;;
            InvoiceDate) COL=5 ;;
            UnitPrice) COL=6 ;;
            CustomerID) COL=7 ;;
            Country) COL=8 ;;
            *)
                echo "Error: Invalid column name: $2"
                echo "Available columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country"
                exit 1
                ;;
        esac
        
        echo "Unique values for $2:"
        tail -n +2 "$DATA_FILE" | cut -d',' -f$COL | sort -u
        ;;
    
    top-products)
        N=${2:-10}
        echo "Top $N Products by Quantity:"
        tail -n +2 "$DATA_FILE" | awk -F',' '{sum[$2] += $4; desc[$2] = $3} END {for (code in sum) print sum[code], code, desc[code]}' | sort -rn | head -n $N | column -t
        ;;
    
    top-customers)
        N=${2:-10}
        echo "Top $N Customers by Transaction Count:"
        tail -n +2 "$DATA_FILE" | cut -d',' -f7 | sort | uniq -c | sort -rn | head -n $N
        ;;
    
    top-countries)
        N=${2:-10}
        echo "Top $N Countries by Transaction Count:"
        tail -n +2 "$DATA_FILE" | cut -d',' -f8 | sort | uniq -c | sort -rn | head -n $N
        ;;
    
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        usage
        exit 1
        ;;
esac
