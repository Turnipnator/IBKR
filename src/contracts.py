"""
Contract registry for the UCITS-on-LSE universe (v2).

UK retail IBKR accounts cannot trade US-listed ETFs (PRIIPs/KID — Error 201).
Every instrument here is a UCITS ETF/ETC listed on LSE, accessed via SMART
routing with primary_exchange='LSEETF'.

Currency is per-symbol because USD-class doesn't exist for every fund;
where it doesn't (EQQQ, VEUR, IJPN) we use the GBP share class.

Price units matter as much as currency. IBKR quotes some LSE GBP lines in
PENCE (ContractDetails.priceMagnifier == 100) while still labelling the
contract currency "GBP": read-only probe 2026-08-28 — EQQQ close 53,603 and
IJPN 1,865 "GBP" are really £536.03 and £18.65; VEUR (magnifier 1) is a
genuine £43.51. Sizing off the raw number treated EQQQ as a £53k share (0
shares, silently never bought) and got IJPN dropped from the universe on
2026-08-14 as "£1.9k/share, unaffordable". Such lines are registered as
"GBX" (100 GBX = 1 GBP): the engine converts at 0.01 GBP per price unit and
`resolve_contract` still builds the IBKR contract with currency "GBP".
Order prices (stop triggers, trail amounts) stay in pence — that is the
unit IBKR expects for these contracts.
"""
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import Stock

DEFAULT_EXCHANGE = "SMART"
DEFAULT_PRIMARY_EXCHANGE = "LSEETF"

# Registry currency -> the currency IBKR wants on the contract. "GBX" is a
# price-unit tag (pence-quoted GBP line), not an IBKR currency.
IBKR_CURRENCY = {"GBX": "GBP"}
GBX_PER_GBP = 100.0

# symbol -> (currency, primary_exchange)
CONTRACT_REGISTRY: dict[str, tuple[str, str]] = {
    # === equity ===
    "CSPX": ("USD", "LSEETF"),   # iShares Core S&P 500 UCITS USD Acc
    "EQQQ": ("GBX", "LSEETF"),   # Invesco Nasdaq-100 UCITS (USD-class only on EBS) — PENCE
    "RTWO": ("USD", "LSEETF"),   # SPDR Russell 2000 US Small Cap UCITS USD
    "EIMU": ("USD", "LSEETF"),   # iShares Core MSCI EM IMI UCITS USD Acc
    "VEUR": ("GBP", "LSEETF"),   # Vanguard FTSE Developed Europe UCITS
    "IJPN": ("GBX", "LSEETF"),   # iShares MSCI Japan UCITS (USD-class MXJP too thin) — PENCE
    "CNYA": ("USD", "LSEETF"),   # iShares MSCI China A UCITS USD Acc
    # === bond ===
    "DTLA": ("USD", "LSEETF"),   # iShares $ Treasury Bond 20+yr UCITS USD
    "IDTM": ("USD", "LSEETF"),   # iShares $ Treasury Bond 7-10yr UCITS USD
    "IBTA": ("USD", "LSEETF"),   # iShares $ Treasury Bond 1-3yr UCITS USD
    "LQDE": ("USD", "LSEETF"),   # iShares $ Corp Bond UCITS USD
    "IHYU": ("USD", "LSEETF"),   # iShares $ High Yield Corp Bond UCITS USD
    "JPEA": ("USD", "LSEETF"),   # JPM USD EM Sovereign Bond UCITS USD
    "IDTP": ("USD", "LSEETF"),   # iShares $ TIPS UCITS USD
    # === commodity ===
    "IGLN": ("USD", "LSEETF"),   # iShares Physical Gold ETC USD
    "ISLN": ("USD", "LSEETF"),   # iShares Physical Silver ETC USD
    "CRUD": ("USD", "LSEETF"),   # WisdomTree WTI Crude Oil USD
    "NGAS": ("USD", "LSEETF"),   # WisdomTree Natural Gas USD
    "AIGA": ("USD", "LSEETF"),   # WisdomTree Agriculture USD
    "AIGI": ("USD", "LSEETF"),   # WisdomTree Industrial Metals USD
    "CMOD": ("USD", "LSEETF"),   # WisdomTree Broad Commodities USD (PDBC proxy)
    "COPA": ("USD", "LSEETF"),   # WisdomTree Copper USD
    "AIGS": ("USD", "LSEETF"),   # WisdomTree Broad Commodities USD (DBC proxy)
    # === alt ===
    "IDUP": ("USD", "LSEETF"),   # iShares US Property Yield UCITS USD
}


def resolve_contract(symbol: str) -> Stock:
    """Build an unqualified Stock contract for a registered UCITS symbol.

    Callers must still qualify via ib.qualifyContracts() before use.
    Raises KeyError if the symbol is not in the registry — fail loudly rather
    than silently default to USD which would re-trigger PRIIPs rejections.
    """
    if symbol not in CONTRACT_REGISTRY:
        raise KeyError(
            f"Symbol {symbol!r} not in contract registry. "
            f"Known: {sorted(CONTRACT_REGISTRY)}"
        )
    currency, primary = CONTRACT_REGISTRY[symbol]
    currency = IBKR_CURRENCY.get(currency, currency)
    return Stock(symbol, DEFAULT_EXCHANGE, currency, primaryExchange=primary)
