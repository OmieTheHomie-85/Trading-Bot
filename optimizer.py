import itertools
import pandas as pd
from dataclasses import dataclass
from backtester import Backtester, BacktestResult
from strategy import BaseStrategy, MomentumStrategy, RSIStrategy, DualMAStrategy


@dataclass
class OptimizationResult:
    strategy_class: type
    best_params: dict
    best_result: BacktestResult
    all_results: list[tuple[dict, BacktestResult]]

    def summary(self) -> str:
        lines = [
            f"Best strategy: {self.strategy_class.__name__}",
            f"Best params:   {self.best_params}",
            "",
            self.best_result.summary(),
        ]
        return "\n".join(lines)


class Optimizer:
    """
    Walk-forward optimizer. For each strategy class, tries all parameter
    combinations and ranks by Sharpe ratio on out-of-sample data.

    Walk-forward methodology:
        - Train window: first `train_frac` of the data
        - Test window:  remaining data (never seen during parameter search)

    This prevents overfitting to in-sample data.
    """

    PARAM_GRID = {
        MomentumStrategy: {
            "ma_window": [20, 50, 100],
            "momentum_window": [30, 60, 90],
            "top_n": [1, 2, 3],
        },
        RSIStrategy: {
            "rsi_period": [7, 14, 21],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
            "top_n": [1, 2, 3],
        },
        DualMAStrategy: {
            "fast_window": [10, 20],
            "slow_window": [50, 100],
            "top_n": [1, 2, 3],
        },
    }

    def __init__(
        self,
        price_data: dict[str, pd.Series],
        train_frac: float = 0.7,
        commission: float = 0.001,
        rebalance_freq: str = "W",
    ):
        self.price_data = price_data
        self.train_frac = train_frac
        self.commission = commission
        self.rebalance_freq = rebalance_freq
        self._split_data()

    def _split_data(self):
        df = pd.DataFrame(self.price_data).dropna()
        split = int(len(df) * self.train_frac)
        self.train_data = {col: df[col].iloc[:split] for col in df.columns}
        self.test_data  = {col: df[col].iloc[split:] for col in df.columns}
        print(
            f"Data split — train: {len(df.iloc[:split])} days, "
            f"test: {len(df.iloc[split:])} days"
        )

    def _grid_search(
        self,
        strategy_class: type,
        data: dict[str, pd.Series],
        param_grid: dict,
    ) -> list[tuple[dict, BacktestResult]]:
        """Try all parameter combinations; return sorted results."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            strategy = strategy_class(**params)
            backtester = Backtester(
                data,
                commission=self.commission,
                rebalance_freq=self.rebalance_freq,
            )
            try:
                result = backtester.run(strategy)
                results.append((params, result))
            except Exception as e:
                print(f"  Skipped {params}: {e}")

        results.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)
        return results

    def optimize(
        self,
        strategy_classes: list[type] | None = None,
    ) -> list[OptimizationResult]:
        """
        Run walk-forward optimization across all (or selected) strategy classes.
        Returns results sorted by out-of-sample Sharpe ratio (best first).
        """
        if strategy_classes is None:
            strategy_classes = list(self.PARAM_GRID.keys())

        optimization_results = []

        for cls in strategy_classes:
            print(f"\nOptimizing {cls.__name__}...")
            param_grid = self.PARAM_GRID.get(cls, {})

            if not param_grid:
                print(f"  No param grid defined for {cls.__name__}, skipping.")
                continue

            # Step 1: Grid search on TRAINING data
            train_results = self._grid_search(cls, self.train_data, param_grid)
            if not train_results:
                continue

            best_train_params, _ = train_results[0]
            print(f"  Best train params: {best_train_params}")

            # Step 2: Evaluate best params on TEST data (out-of-sample)
            best_strategy = cls(**best_train_params)
            test_backtester = Backtester(
                self.test_data,
                commission=self.commission,
                rebalance_freq=self.rebalance_freq,
            )
            test_result = test_backtester.run(best_strategy)
            print(f"  Out-of-sample Sharpe: {test_result.sharpe_ratio:.3f}")

            optimization_results.append(
                OptimizationResult(
                    strategy_class=cls,
                    best_params=best_train_params,
                    best_result=test_result,
                    all_results=train_results,
                )
            )

        # Sort by out-of-sample Sharpe
        optimization_results.sort(
            key=lambda r: r.best_result.sharpe_ratio, reverse=True
        )
        return optimization_results

    def best_strategy(self) -> BaseStrategy:
        """Run optimization and return the best strategy instance, ready to trade."""
        results = self.optimize()
        if not results:
            raise RuntimeError("Optimization produced no results.")
        best = results[0]
        print(f"\nWinner: {best.strategy_class.__name__} with {best.best_params}")
        print(best.best_result.summary())
        return best.strategy_class(**best.best_params)