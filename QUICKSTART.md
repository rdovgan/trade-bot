# Швидкий старт: запуск бота у тестовому режимі

## Крок 1: Встановити Python

Потрібен Python 3.9+. Перевір версію:

```bash
python3 --version
```

## Крок 2: Створити віртуальне середовище

```bash
cd ~/Desktop/projects/trade-bot
python3 -m venv .venv
source .venv/bin/activate
```

Після активації у терміналі з'явиться `(.venv)` на початку рядка.

## Крок 3: Встановити залежності

```bash
pip install -e ".[dev]"
```

Це встановить сам бот + усі бібліотеки (ccxt, pandas, numpy, pydantic, pytest тощо).

## Крок 4: Перевірити, що все встановилось

```bash
pytest
```

Має бути `147 passed`. Якщо тести пройшли — установка коректна.

## Крок 5: Налаштувати ключі біржі

Бот працює через [CCXT](https://github.com/ccxt/ccxt) і підтримує 100+ бірж. За замовчуванням — Binance testnet.

### 5.1 Отримати тестові API ключі Binance

1. Зайди на https://testnet.binancefuture.com/ (ф'ючерси) або https://testnet.binance.vision/ (спот)
2. Залогінься через GitHub
3. Створи API ключ — отримаєш `API Key` та `Secret Key`

### 5.2 Створити файл `.env`

```bash
cp .env.example .env
```

Відкрий `.env` і впиши ключі:

```
EXCHANGE_API_KEY=твій_api_key
EXCHANGE_SECRET=твій_secret_key
```

Решту можна залишити за замовчуванням.

## Крок 6: Запустити бота

```bash
python -m trade_bot.main
```

Бот стартує у **sandbox (testnet) режимі** — торгує фейковими грошима. Він буде:

1. Кожну хвилину отримувати дані ринку (OHLCV, order book)
2. Визначати режим ринку (тренд, mean reversion, висока волатильність)
3. Генерувати сигнали відповідними стратегіями
4. Перевіряти сигнал через risk validator (може заветувати)
5. Виконувати ордер, якщо пройшов валідацію
6. Моніторити позиції, записувати в журнал

Логи пишуться у `trade_bot.log` та в термінал.

### Зупинити бота

`Ctrl+C` — бот коректно закриє всі з'єднання і скасує відкриті ордери.

## Крок 7: Подивитися результати

Журнал торгівлі зберігається у SQLite базі:

```
~/.trade_bot/journal.db
```

Переглянути можна через Python:

```python
from trade_bot.learning.journal import TradeJournal

journal = TradeJournal()

# Останні угоди
trades = journal.get_trades(limit=10)
for t in trades:
    print(f"{t.symbol} {t.side.value} PnL={t.pnl:.2f} R={t.r_multiple:.2f}")

# Статистика за 30 днів
stats = journal.get_performance_stats(days=30)
print(stats)

# Результати по режимах ринку
print(journal.get_regime_performance())

# Експорт у CSV
journal.export_to_csv("my_trades.csv")
```

## Крок 8: Змінити налаштування (опціонально)

### Інша біржа

У `main.py` → `_default_config()` зміни `exchange.name`:

```python
'exchange': {
    'name': 'bybit',    # або 'okx', 'kraken', тощо
    'sandbox': True,
    ...
}
```

### Інші торгові пари

```python
'trading': {
    'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    ...
}
```

### Увімкнути LLM-аналітику

У `.env`:

```
LLM_ENABLED=true
LLM_API_KEY=sk-your-openai-key
```

LLM дає поради (підтвердження режиму, фактори ризику), але **не може** змінювати розмір позиції, знімати стоп-лос або обходити ліміти ризику.

## Що відбувається всередині

```
Кожну хвилину для кожного символу:

1. SCAN      — збір даних (ціна, ATR, волатильність, ліквідність)
2. SIGNAL    — генерація сигналу (залежить від режиму ринку)
3. RESEARCH  — LLM аналіз (якщо увімкнено)
4. DECIDE    — вибір найкращого сигналу
5. RISK      — валідація (може заблокувати угоду)
6. EXECUTE   — виконання ордеру на біржі
7. MONITOR   — відстеження позиції, здоров'я біржі
8. CLOSE     — закриття по SL/TP/trailing stop
9. JOURNAL   — запис результату
10. LEARN    — оцінка стратегії через salience model
```

## Типові проблеми

| Проблема | Рішення |
|---|---|
| `ModuleNotFoundError: No module named 'trade_bot'` | Переконайся, що запустив `pip install -e ".[dev]"` |
| `ccxt.NetworkError` | Перевір інтернет-з'єднання та API ключі |
| Бот нічого не торгує | Це нормально — сигнали генеруються лише при певних умовах ринку. Перевір логи |
| `Exchange unhealthy — skipping` | Біржа не відповідає. Бот почекає і спробує знову |
