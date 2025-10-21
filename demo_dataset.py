"""
Apple Stock Data Demo (AI-Ready Dataset)
----------------------------------------

Author: Agourou Stavroula
GitHub: https://github.com/staaaavv

Disclaimer:
- This script downloads publicly available Apple Inc. (AAPL) stock data from Yahoo Finance via the `yfinance` library.
- All processing, feature engineering, visualization, and PDF report generation are performed locally.
- Raw Yahoo Finance data are not included in this repository and may NOT be used for commercial purposes.
- The code is intended strictly for educational, research, and demonstration purposes (e.g., AI model training, financial analysis, forecasting prototypes).
- The author provides NO financial advice and assumes NO responsibility for investment decisions made using this script.
- The Python code is licensed under the Apache License 2.0. See LICENSE file for details.
"""
import yfinance as yf
import pandas as pd

# Δημιουργία αντικειμένου για Apple
ticker = yf.Ticker("AAPL")

# Ιστορικές τιμές 2 χρόνων (OHLCV)
data = ticker.history(period="2y")

# Οικονομικά στοιχεία (fundamentals)
financials = ticker.financials
balance_sheet = ticker.balance_sheet
cashflow = ticker.cashflow

# Προβολή πρώτων γραμμών
print(data.head())
print(financials.head())
# Αφαίρεση διπλών γραμμών
data = data.drop_duplicates()

# Γέμισμα κενών τιμών με την τελευταία διαθέσιμη τιμή
data = data.ffill()
import ta

# SMA και EMA
data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)

# RSI και MACD
data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
data['MACD'] = ta.trend.macd(data['Close'])

# Υπολογισμός ημερήσιας απόδοσης
data['Return'] = data['Close'].pct_change()
# ISO format ημερομηνιών και αποθήκευση σε CSV
data.reset_index(inplace=True)
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
data.to_csv("apple_curated_dataset.csv", index=False, sep=';')

numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in numeric_cols:
    if (data[col] < 0).any():
        print(f"Warning: Negative values found in {col}")

# Αν υπάρχουν αρνητικές τιμές, τις αντικαθιστούμε με NaN και κάνουμε forward fill
for col in numeric_cols:
    data[col] = data[col].apply(lambda x: x if x >= 0 else None)
data = data.ffill()
# Αφαιρούμε τις γραμμές που έχουν NaN στους υπολογισμένους δείκτες
data.dropna(subset=['SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'Return'], inplace=True)

print("Missing values per column:")
print(data.isna().sum())
data.to_csv("apple_curated_dataset_clean.csv", index=False, sep=';')

import matplotlib.pyplot as plt



plt.figure(figsize=(12,5))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], data['SMA_20'], label='SMA 20')
plt.plot(data['Date'], data['EMA_20'], label='EMA 20')
plt.title('Apple Close Prices with SMA & EMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Βρίσκουμε ακραίες ημερήσιες μεταβολές (>10%)
outliers = data[abs(data['Return']) > 0.10]

# Εμφάνιση ημερομηνίας, κλείσιμου και return
print("Potential outliers:")
print(outliers[['Date', 'Close', 'Return']])

plt.figure(figsize=(12,5))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
plt.scatter(outliers['Date'], outliers['Close'], color='red', label='Spike', zorder=5)
plt.title('Apple Close Prices with Spike Highlighted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Συνοπτικά στατιστικά
summary_stats = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].describe()
print(summary_stats)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Δημιουργία PDF
pdf_filename = "apple_summary_report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
styles = getSampleStyleSheet()
elements = []

# Τίτλος
elements.append(Paragraph("Apple Stock Summary Report", styles['Title']))
elements.append(Spacer(1, 12))

# Εισαγωγή
elements.append(Paragraph("Aggregated Statistics (Last 2 Years)", styles['Heading2']))

# Μετατροπή summary σε πίνακα
summary_data = [summary_stats.columns.tolist()] + summary_stats.round(4).values.tolist()
t = Table(summary_data)
t.setStyle([('BACKGROUND', (0,0), (-1,0), colors.lightblue),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)])
elements.append(t)
elements.append(Spacer(1, 12))

# Εισαγωγή outliers
elements.append(Paragraph("Days with Extreme Daily Returns (>10%)", styles['Heading2']))
outlier_table = [["Date", "Close", "Return"]] + outliers[['Date', 'Close', 'Return']].round(4).values.tolist()
t2 = Table(outlier_table)
t2.setStyle([('BACKGROUND', (0,0), (-1,0), colors.lightcoral),
             ('GRID', (0,0), (-1,-1), 0.5, colors.grey)])
elements.append(t2)

# Δημιουργία PDF
doc.build(elements)
print("✅ PDF report saved as:", pdf_filename) 
