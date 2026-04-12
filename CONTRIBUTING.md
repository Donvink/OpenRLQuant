# Contributing to OpenRLQuant

First off, thank you for considering contributing. This is an open research project and every contribution matters.

## Ways to Contribute

**Code contributions**
- Bug fixes — always welcome, open a PR directly
- New features — open an issue first to discuss
- Performance improvements — benchmark before and after
- Test coverage — we need more of this

**Non-code contributions**
- Documenting edge cases you discovered
- Sharing backtest results (good or bad)
- Reporting data quality issues with specific brokers
- Translating documentation

---

## Development Setup

```bash
git clone https://github.com/yourusername/OpenRLQuant.git
cd OpenRLQuant

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements_dev.txt   # pytest, black, ruff

# Run tests
python -m pytest tests/ -v

# Format code
black . && ruff check .
```

---

## Project Structure Conventions

**One responsibility per file.** If a file is doing two unrelated things, split it.

**No silent failures.** Log warnings when data is missing or fallbacks are used.

**Look-ahead bias is a hard error.** Any feature that uses future data to normalize or compute must be caught in review. This is the most common and most damaging mistake in quant backtesting.

**Real costs, not optimistic ones.** When in doubt, use higher slippage estimates.

---

## Pull Request Process

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Run the Phase 1 smoke test: `python run_phase1.py --mode quick --use-synthetic`
6. Open a PR with a clear description of what changed and why

**PR checklist:**
- [ ] Tests pass
- [ ] No look-ahead bias introduced in features
- [ ] New feature has at least one test
- [ ] Docstring added for public functions
- [ ] `run_phase1.py --mode quick` passes

---

## Reporting Bugs

Open a GitHub issue with:
- Python version and OS
- Exact command that caused the error
- Full traceback
- What you expected vs what happened

---

## Areas Most Needing Contribution

These are open and would have meaningful impact:

**Factor research** (`features/`)
- Barra-style fundamental factors (quality, value, momentum)
- Earnings surprise factors
- Short interest data integration

**Environment improvements** (`environment/`)
- Short selling support
- Options overlay (protective puts)
- Intraday VWAP execution simulation

**Alternative data** (`data/`)
- Earnings call transcript sentiment
- Insider filing signals (Form 4)
- Options flow imbalance

**Testing** (`tests/`)
- Integration tests for the full Phase 1 pipeline
- Property-based tests for the risk manager
- Regression tests for known backtest results

---

## Code Style

We use `black` for formatting and `ruff` for linting. CI will reject PRs that don't pass both.

```python
# Good: explicit, typed, documented
def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.05,
) -> pd.Series:
    """
    Compute rolling annualized Sharpe ratio.
    Uses only past data — no look-ahead bias.
    """
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - rf_daily
    return (excess.rolling(window).mean() /
            returns.rolling(window).std()) * np.sqrt(252)

# Bad: implicit, no types, no docs
def sharpe(r, w=63):
    return r.rolling(w).mean() / r.rolling(w).std() * 16
```

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
