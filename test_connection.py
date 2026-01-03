#!/usr/bin/env python3
"""
IBKR Connection Test Script
Tests connection to TWS or IB Gateway and fetches basic account info.
"""

# Python 3.10+ asyncio compatibility fix - must be before ib_insync import
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock, util
import sys

# Connection settings
# TWS Paper Trading: port 7497
# TWS Live: port 7496
# IB Gateway Paper: port 4002
# IB Gateway Live: port 4001

HOST = '127.0.0.1'
PORT = 7497  # Paper trading TWS - change if using Gateway
CLIENT_ID = 1


def test_connection():
    """Test basic connection to IBKR."""
    ib = IB()

    print(f"Attempting connection to {HOST}:{PORT}...")
    print("-" * 50)

    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
        print("SUCCESS: Connected to IBKR!")
        print(f"Server Version: {ib.client.serverVersion()}")
        print("-" * 50)

        # Get account info
        print("\nAccount Summary:")
        accounts = ib.managedAccounts()
        print(f"Managed Accounts: {accounts}")

        # Get account values
        if accounts:
            account_values = ib.accountSummary()
            # Show key values
            key_tags = ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'AvailableFunds']
            for av in account_values:
                if av.tag in key_tags:
                    print(f"  {av.tag}: {av.value} {av.currency}")

        print("-" * 50)

        # Test market data with a simple stock quote
        print("\nTesting market data (AAPL)...")
        contract = Stock('AAPL', 'SMART', 'USD')
        ib.qualifyContracts(contract)

        # Request snapshot (doesn't require streaming subscription)
        ticker = ib.reqMktData(contract, '', True, False)
        ib.sleep(2)  # Wait for data

        if ticker.last or ticker.close:
            price = ticker.last if ticker.last else ticker.close
            print(f"  AAPL Price: ${price}")
        else:
            print("  Note: Price data may require market data subscription")
            print(f"  Ticker data: bid={ticker.bid}, ask={ticker.ask}")

        print("-" * 50)
        print("\nAll tests passed! Your IBKR connection is working.")

        ib.disconnect()
        return True

    except ConnectionRefusedError:
        print("ERROR: Connection refused!")
        print("\nTroubleshooting steps:")
        print("1. Make sure TWS or IB Gateway is running")
        print("2. In TWS: File > Global Configuration > API > Settings")
        print("   - Enable 'Enable ActiveX and Socket Clients'")
        print("   - Set Socket Port to 7497 (paper) or 7496 (live)")
        print("   - Uncheck 'Read-Only API' if you want to place orders")
        print("3. Add 127.0.0.1 to 'Trusted IPs' or uncheck 'Allow connections from localhost only'")
        return False

    except TimeoutError:
        print("ERROR: Connection timed out!")
        print("\nCheck that:")
        print("1. TWS/Gateway is running and logged in")
        print("2. The port number is correct (7497 for paper TWS)")
        print("3. API connections are enabled in TWS settings")
        return False

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)
