import yaml
from src.data_loader import DataLoader
from src.feature_engineering import create_features
from src.hmm_model import HMMModel
from src.signal_generation import generate_signals
from src.backtesting import Backtest

if __name__ == "__main__":
    # Load configuration
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Step 1: Load Data
    data_loader = DataLoader(
        symbol=config["data"]["symbol"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        save_path=config["data"]["raw_data_path"]
    )
    raw_data = data_loader.load_data()

    # Step 2: Feature Engineering
    processed_data = create_features(raw_data)

    # Step 3: Train HMM Model
    hmm_model = HMMModel(
        n_states=config["model"]["n_states"],
        covariance_type=config["model"]["covariance_type"],
        n_emissions=config["model"]["n_emissions"]
    )
    hmm_model.train(processed_data[["Returns", "Volatility"]].iloc[:500])
    hmm_results = hmm_model.predict(processed_data[["Returns", "Volatility"]].iloc[500:])

    # Step 4: Generate Signals
    signals = generate_signals(processed_data.iloc[500:], hmm_results, config["backtesting"]["favorable_states"])

    # Step 5: Backtesting and Evaluation
    backtest = Backtest(signals, config["backtesting"]["risk_free_rate"], config["backtesting"]["annual_trading_days"])
    benchmark_sharpe, strategy_sharpe = backtest.evaluate()

    # Output Results
    print(f"Benchmark Sharpe: {benchmark_sharpe}")
    print(f"Strategy Sharpe: {strategy_sharpe}")
