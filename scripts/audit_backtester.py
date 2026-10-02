"""
Auditoría del backtester.

Genera un trade log detallado sobre datos reales con una estrategia conocida,
y verifica manualmente:
1. Cada ejecución ocurre al open de la vela siguiente a la señal.
2. PnL bruto cuadra con precios.
3. Costes son los esperados.
4. Cash y equity cuadran.

Uso:
    python -m scripts.audit_backtester \
        --csv data_files/EURUSD_X_15m_5y.csv \
        --output experiments_output/audit \
        --n-bars 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np 
import pandas as pd

from backtest.engine import (
    BacktestConfig,
    logs_to_dataframe,
    run_backtest_with_log,
)
from backtest.execution import TradeRecord
from config.settings import DEFAULT_COSTS
from data.cleaner import load_and_clean
from config.settings import SUPPORTED_SYMBOLS, infer_symbol_from_path

def make_simple_signal(n: int, period: int = 100) -> pd.Series:
    """
    Señal trivial: alterna LONG durante `period` velas y FLAT durante `period` velas.
    No depende de features. Sirve para auditar el motor de ejecución.
    """
    arr = np.zeros(n, dtype=int)
    for i in range(n):
        if (i // period) % 2 == 0:
            arr[i] = 1  # LONG
        else:
            arr[i] = 0  # FLAT
    return pd.Series(arr, name="signal")


def audit_execution_timing(logs_df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """
    Verifica que cada ejecución ocurre al open de t+1 respecto a la señal en t.
    Devuelve las filas donde hay ejecución, con la señal previa.
    """
    executed = logs_df[logs_df["pending_signal"].notna()].copy()
    executed["signal_prev"] = executed["pending_signal"]
    executed["executed_at_open"] = executed["execution_price"]
    executed["open_of_this_bar"] = executed["open"]
    executed["timing_ok"] = (
        executed["executed_at_open"] == executed["open_of_this_bar"]
    )
    return executed


def audit_trade_pnl(trades: list[TradeRecord]) -> pd.DataFrame:
    """Verifica que gross_pnl cuadra con entry/exit prices."""
    rows = []
    for t in trades:
        if t.direction.name == "LONG":
            expected_gross = t.size * (t.exit_price - t.entry_price)
        else:
            expected_gross = t.size * (t.entry_price - t.exit_price)
        diff = abs(expected_gross - t.gross_pnl)
        rows.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction.name,
            "entry": t.entry_price,
            "exit": t.exit_price,
            "size": t.size,
            "gross_pnl": t.gross_pnl,
            "expected_gross": expected_gross,
            "pnl_ok": diff < 1e-9,
            "cost": t.cost,
            "net_pnl": t.net_pnl,
        })
    return pd.DataFrame(rows)


def audit_costs(trades: list[TradeRecord], costs) -> pd.DataFrame:
    """Verifica que cada coste de cierre es el esperado."""
    one_side_close = costs.spread / 2 + costs.slippage
    rows = []
    for t in trades:
        expected = one_side_close * t.size
        diff = abs(expected - t.cost)
        rows.append({
            "exit_time": t.exit_time,
            "size": t.size,
            "cost": t.cost,
            "expected_cost": expected,
            "cost_ok": diff < 1e-12,
        })
    return pd.DataFrame(rows)


def audit_equity(
    equity_final: float,
    initial_capital: float,
    trades: list[TradeRecord],
    total_costs_paid: float,
) -> dict:
    """
    Verifica la identidad contable:
    equity_final = capital + sum(gross) - total_costs_paid
    """
    total_gross = sum(t.gross_pnl for t in trades)
    expected = initial_capital + total_gross - total_costs_paid
    return {
        "initial_capital": initial_capital,
        "total_gross_pnl": total_gross,
        "total_costs_paid": total_costs_paid,
        "expected_equity": expected,
        "actual_equity": equity_final,
        "diff": abs(expected - equity_final),
        "ok": abs(expected - equity_final) < 1e-6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="Símbolo. Si no se pasa, se infiere del nombre del CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bars", type=int, default=2000)
    parser.add_argument("--period", type=int, default=100,
                        help="Velas por ciclo LONG/FLAT")
    args = parser.parse_args()

    if args.symbol is None:
        try:
            symbol = infer_symbol_from_path(args.csv)
        except ValueError as e:
            parser.error(str(e))
    else:
        symbol = args.symbol
        if symbol not in SUPPORTED_SYMBOLS:
            parser.error(f"--symbol {symbol} no está en {SUPPORTED_SYMBOLS}")

    print(f"Símbolo: {symbol}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    df, _ = load_and_clean(args.csv)
    df = df.iloc[:args.n_bars]

    # Señal trivial
    signals = make_simple_signal(len(df), period=args.period)
    signals.index = df.index

    # Backtest con log
    config = BacktestConfig(symbol=symbol, initial_capital=10_000.0, costs=DEFAULT_COSTS, size=1.0)
    result, logs = run_backtest_with_log(df, signals, config)

    logs_df = logs_to_dataframe(logs)

    # Guardar log completo
    logs_df.to_csv(out_dir / "bar_log.csv")
    print(f"Log guardado en: {out_dir / 'bar_log.csv'}")

    # 1. Verificar timing
    exec_df = audit_execution_timing(logs_df, signals)
    n_exec = len(exec_df)
    n_timing_ok = int(exec_df["timing_ok"].sum())
    print(f"\n[1] Timing de ejecución:")
    print(f"  Ejecuciones: {n_exec}")
    print(f"  Al open de t+1: {n_timing_ok}")
    if n_timing_ok != n_exec:
        print(f"  ❌ FALLO: {n_exec - n_timing_ok} ejecuciones no al open")
        print(exec_df[~exec_df["timing_ok"]].head())
    else:
        print(f"  ✅ Correcto")

    # 2. Verificar PnL
    trades_df = audit_trade_pnl(result.trades)
    trades_df.to_csv(out_dir / "trades_audit.csv", index=False)
    n_trades = len(trades_df)
    n_pnl_ok = int(trades_df["pnl_ok"].sum())
    print(f"\n[2] PnL bruto por trade:")
    print(f"  Trades cerrados: {n_trades}")
    print(f"  PnL cuadra: {n_pnl_ok}")
    if n_pnl_ok != n_trades:
        print(f"  ❌ FALLO: {n_trades - n_pnl_ok} trades con PnL mal calculado")
        print(trades_df[~trades_df["pnl_ok"]].head())
    else:
        print(f"  ✅ Correcto")

    # 3. Verificar costes
    costs_df = audit_costs(result.trades, DEFAULT_COSTS)
    n_costs_ok = int(costs_df["cost_ok"].sum())
    print(f"\n[3] Costes:")
    print(f"  Trades: {n_trades}")
    print(f"  Costes cuadran: {n_costs_ok}")
    if n_costs_ok != n_trades:
        print(f"  ❌ FALLO")
        print(costs_df[~costs_df["cost_ok"]].head())
    else:
        print(f"  ✅ Correcto")

    # 4. Verificar equity
    # Coste total pagado = suma de costes de cada transición (no solo cierres)
    # En el log, cada trade registra solo el coste de cierre; el de apertura
    # se descontó al abrir. Reconstruimos el total.
    transitions = logs_df[logs_df["pending_signal"].notna()]
    # Contar transiciones
    n_transitions = int(
        (logs_df["position_before"] != logs_df["position_after"]).sum()
    )
    one_side = DEFAULT_COSTS.spread / 2 + DEFAULT_COSTS.slippage
    total_costs_paid = n_transitions * one_side * config.size

    eq = audit_equity(
        equity_final=float(result.equity.iloc[-1]),
        initial_capital=config.initial_capital,
        trades=result.trades,
        total_costs_paid=total_costs_paid,
    )
    print(f"\n[4] Identidad contable:")
    for k, v in eq.items():
        print(f"  {k}: {v}")
    if eq["ok"]:
        print(f"  ✅ Correcto")
    else:
        print(f"  ❌ FALLO")

    # 5. Mostrar primeros 5 trades para inspección manual
    print(f"\n[5] Primeros 5 trades (para inspección manual):")
    print(trades_df.head(5).to_string())

    # 6. Verificar contra los datos crudos: buscar la vela de ejecución
    print(f"\n[6] Cross-check contra datos crudos (primer trade):")
    if n_trades > 0:
        first = result.trades[0]
        # Buscar la vela cuyo open == entry_price
        mask_entry = np.isclose(df["open"].values, first.entry_price, atol=1e-9)
        mask_exit = np.isclose(df["open"].values, first.exit_price, atol=1e-9)
        entry_bar = df[mask_entry]
        exit_bar = df[mask_exit]
        print(f"  Trade LONG/SHORT: {first.direction.name}")
        print(f"  Entry time registrado: {first.entry_time}")
        print(f"  Entry price registrado: {first.entry_price}")
        if len(entry_bar) > 0:
            print(f"  Vela con ese open: {entry_bar.index[0]}")
        else:
            print(f"  ⚠️ No hay vela con ese open exacto")
        print(f"  Exit time registrado: {first.exit_time}")
        print(f"  Exit price registrado: {first.exit_price}")
        if len(exit_bar) > 0:
            print(f"  Vela con ese open: {exit_bar.index[0]}")
        else:
            print(f"  ⚠️ No hay vela con ese open exacto")


if __name__ == "__main__":
    main()