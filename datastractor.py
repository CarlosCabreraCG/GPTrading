import yfinance as yf

# Descargar datos M15 para EUR/USD
data = yf.download(
    tickers="EURUSD=X",
    start="2026-08-18",
    end="2026-09-24",
    interval="15m"
)

# Guardar en archivo CSV
data.to_csv("EURUSD_2025_15m.csv")
print("Descarga completada.")