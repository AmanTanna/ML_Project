import requests
import json
import pandas as pd
import certifi
from urllib.request import urlopen
import sys

def get_jsonparsed_data(url, stock_symbol, period):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    # Add debugging information
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 403:
        print("403 Forbidden Error - Possible causes:")
        print("1. Invalid or expired API key")
        print("2. Rate limit exceeded")
        print("3. API key doesn't have access to this endpoint")
        print(f"Response content: {response.text}")
        return None
    
    response.raise_for_status()
    return response.json()

def main(sys_args):
    if len(sys_args) < 3:
        print("Usage: python3 poc.py <stock_symbol> <period>")
        sys.exit(1)

    
    stock_symbol = sys_args[1]
    period = sys_args[2]
    url = f"https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{stock_symbol}?period={period}&limit=50&apikey=d9aOft0Pc62iCklpe7kDRpHyxmrHnmcL"
    
    print(f"Requesting URL: {url}")
    result = get_jsonparsed_data(url, stock_symbol, period)
    
    if result is None:
        print("Failed to fetch data. Please check your API key and try again.")
        sys.exit(1)
    
    # Write result to a JSON file
    with open("output2.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Result saved to output.json")

if __name__ == "__main__":
    main(sys.argv)