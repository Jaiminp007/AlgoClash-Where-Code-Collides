# Original Conservative System Prompt

This file contains the original conservative trading prompt. Use this to restore the original behavior if needed.

## How to Restore

Copy the sections below and replace the corresponding sections in `backend/open_router/algo_gen.py`:

---

## 1. KEY PRINCIPLES (Line ~593)

Replace the "KEY PRINCIPLES - AGGRESSIVE DEGENERATE TRADING MODE" section with:

```
KEY PRINCIPLES:
- Trade SELECTIVELY (quality > quantity) - avoid overtrading
- Use multi-layer signal confirmation
- Implement risk management (position sizing, stop losses)
- Profit in BOTH up AND down markets (use short selling)
- Focus on risk-adjusted returns, not just frequency
```

---

## 2. TRADING RULES (Line ~555)

Replace the "DEGEN TRADING RULES" section with:

```
RETURN VALUE: Must be exactly "BUY", "SELL", or "HOLD" (uppercase string)

TRADING RULES:
- BUY = Predict price will increase
- SELL = Predict price will decrease (short selling allowed)
- HOLD = Uncertain or no clear signal
- Goal: Maximize ROI through selective trades

SESSION CONSTRAINTS:
- Shorting is allowed; selling with zero/negative shares opens or increases a short.
- Be cost-aware; avoid overtrading. If no edge, HOLD.
- Aim to finish the session flat (shares_held == 0) in the final ticks.
```

---

## 3. SHORTING STRATEGIES (Line ~630)

Replace the "DEGEN" shorting strategies with:

```
Strategy A - Momentum Short (Aggressive):
```python
# Detect strong downtrend early
if sma_20 < sma_50 and current_price < sma_20 * 0.98:  # 2% below trend
    if rsi < 45 and shares_held >= 0:  # Bearish momentum, no short yet
        return "SELL"  # Initiate short position

# Cover short when downtrend weakens
elif shares_held < 0 and current_price > sma_20:
    return "BUY"  # Cover short position
```

Strategy B - Breakdown Short (Safer):
```python
# Wait for price to break key support before shorting
support_level = np.min(close_prices[-20:])  # 20-day low
if current_price < support_level * 0.99:  # Broke support
    if shares_held >= 0 and rsi < 50:
        return "SELL"  # Short on breakdown
```

Strategy C - Mean Reversion Short (Contrarian):
```python
# Short when price is overextended to upside in downtrend
if sma_20 < sma_50:  # Overall downtrend
    if current_price > sma_20 * 1.03:  # Temporary rally
        if rsi > 60:  # Overbought in downtrend
            return "SELL"  # Short the rally
```
```

---

## 4. POSITION MANAGEMENT (Line ~668)

Replace "DEGEN POSITION MANAGEMENT" with:

```
POSITION MANAGEMENT WITH SHORTS:
- Start shorts early when downtrend confirmed (don't wait for bottom!)
- Use tight stops above resistance levels
- Cover shorts (BUY) when:
  * Price breaks above SMA20
  * RSI crosses above 50
  * Volume dries up (exhaustion)
  * Bullish divergence appears

RECOMMENDED APPROACH:
Use technical analysis with historical data:
1. Download historical OHLCV data with yfinance (cached at module level)
2. Calculate technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
3. Generate trading signals with multiple confirmations
4. Return BUY/SELL/HOLD based on signal strength

STRATEGY OPTIONS (choose and innovate):
- Mean Reversion: Trade extremes (Bollinger Bands, z-scores)
- Momentum: Follow trends (MACD, ROC, moving average crossovers)
- Volatility: Trade regime changes (ATR, volatility breakouts)
- Multi-factor: Combine multiple uncorrelated signals
```

---

## 5. CODE REJECTION CRITERIA (Line ~599)

Replace with original (remove the last 2 lines about being "too conservative"):

```
YOUR CODE WILL BE REJECTED IF:
❌ Empty function body or only "pass"
❌ Missing return statements
❌ Less than 50 lines
❌ No actual trading logic
❌ yfinance used without end= parameter
❌ Contains only explanation text with no code
```

---

## 6. EXAMPLE TEMPLATE (Line ~767)

Replace the aggressive example with:

```python
        # Trading logic (example: trend following with confirmation)
        # BUY signal: Price above SMA20, SMA20 above SMA50 (uptrend)
        if current_price > sma_20 * 1.01 and sma_20 > sma_50:
            if shares_held <= 0:  # No position or short - open long
                return "BUY"

        # SELL signal: Price below SMA20, SMA20 below SMA50 (downtrend)
        elif current_price < sma_20 * 0.99 and sma_20 < sma_50:
            if shares_held >= 0:  # No position or long - go short
                return "SELL"

        # Exit positions when trend weakens
        elif shares_held > 0 and current_price < sma_20:
            return "SELL"  # Exit long
        elif shares_held < 0 and current_price > sma_20:
            return "BUY"   # Cover short

        return "HOLD"
```

---

## 7. TEMPLATE NOTE (Line ~794)

Replace with:

```
THIS IS A BASIC TEMPLATE - YOU MUST CREATE A MORE SOPHISTICATED STRATEGY!
Add more indicators, better signal confirmation, risk management, etc.
```

---

## 8. RISK PROFILES (Line ~890 in build_diversity_directives)

Replace "DEGEN Risk management profiles" with:

```python
    # Risk management profiles
    risk_profiles = [
        {
            "name": "Conservative",
            "rules": "Maximum 2 trades per 50 ticks. Require 3+ confirmations. Exit any position down >5%. Position size: 20-30% of capital."
        },
        {
            "name": "Balanced",
            "rules": "Maximum 3 trades per 50 ticks. Require 2 confirmations. Exit positions down >8%. Position size: 30-40% of capital."
        },
        {
            "name": "Aggressive",
            "rules": "Maximum 5 trades per 50 ticks. Require 1 strong confirmation. Exit positions down >12%. Position size: 40-60% of capital. Use leverage."
        }
    ]
```

---

## 9. IMPLEMENTATION REQUIREMENTS (Line ~932)

Replace with:

```
1. Implement the EXACT strategy described above
2. Calculate ALL indicators from scratch (RSI, MACD, Bollinger Bands, ATR, etc.)
3. Apply the specified risk management rules
4. Use try/except error handling throughout
5. Cache historical data at module level
6. Make the algorithm profitable in BOTH bull and bear markets
7. Be selective - avoid overtrading (aim for 20-80 trades total, not 200+)
```

---

## 10. SUCCESS FACTORS (Line ~940)

Replace "CRITICAL DEGEN SUCCESS FACTORS" with:

```
CRITICAL SUCCESS FACTORS:
- Multi-layer confirmation: Don't trade on single signals
- Position sizing: Scale based on signal strength
- Stop losses: Exit losing positions before they become disasters
- Profit targets: Take profits when signals reverse
- Market direction detection: Long in uptrends, short in downtrends

Your goal is to WIN the competition by maximizing risk-adjusted returns!
- Trades selectively with high conviction rather than frequently with low conviction
```

---

## 11. REMOVE DEGENERATE PHILOSOPHY SECTION (Line ~518)

Delete the entire "⚡ DEGENERATE TRADER MODE ACTIVATED ⚡" section:

```
# DELETE THIS ENTIRE SECTION:
═══════════════════════════════════════════════════════════════════════════════
⚡ DEGENERATE TRADER MODE ACTIVATED ⚡
═══════════════════════════════════════════════════════════════════════════════

TRADING PHILOSOPHY:
...
(entire section until next separator)
```

---

## 12. REMOVE DEGEN STRATEGIES (in build_diversity_directives around line ~913)

Remove these 4 strategies from the strategies list:
- "DEGEN Momentum Chaser"
- "Ultra-Aggressive Pyramid Scalper"
- "YOLO Volatility Trader"
- "Contrarian Degen Reversal"

---

## Quick Restore Command

To quickly restore the original prompt, you can tell Claude:

> "Please restore the original conservative system prompt using the backup in ORIGINAL_SYSTEM_PROMPT.md. Replace all the DEGEN/aggressive sections with the original conservative versions."

