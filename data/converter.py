import pandas as pd

# 1. Ejemplo de datos como el tuyo
data = pd.read_csv("data_files/EURUSD_X_15m_60d.csv")
df = pd.DataFrame(data)

# 2. Convertir la columna datetime a formato datetime de pandas
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')

# 3. Establecer datetime como el índice (requerido para resample)
df.set_index('datetime', inplace=True)

# 4. Definir las reglas de agregación OHLC
ohlc_dict = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
}

# 5. Aplicar resample a 4 horas (4h)
df_4h = df.resample('4h').agg(ohlc_dict)

# 6. Eliminar periodos sin datos (si aplica) y reiniciar índice
df_4h.dropna(inplace=True)
df_4h.reset_index(inplace=True)
df.to_csv("DEFAULT_CSV.csv")
print(df_4h)