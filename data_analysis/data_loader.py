import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import json

# Paths for data storage
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(DATA_DIR, 'stocks_data.csv')
PARQUET_FILE = os.path.join(DATA_DIR, 'stocks_data.parquet')
CHECKPOINT_FILE = os.path.join(DATA_DIR, 'download_checkpoint.json')

# Batch size for processing tickers (adjust based on your system's capabilities)
BATCH_SIZE = 25
# Delay between batches to avoid rate limiting (seconds)
BATCH_DELAY = 2

def get_tickers():
    """Fetch stock tickers from S&P indices and return in manageable batches"""
    print("Fetching ticker lists...")
    try:
        sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        
        # Clean tickers: remove any that contain dots (these often cause issues with yfinance)
        all_tickers = [ticker for ticker in sp500_tickers 
                      if '.' not in ticker and len(ticker) > 0]
        
        # Remove duplicates
        all_tickers = list(set(all_tickers))
        print(f"Found {len(all_tickers)} unique tickers")
        
        # Split tickers into batches
        return [all_tickers[i:i + BATCH_SIZE] for i in range(0, len(all_tickers), BATCH_SIZE)]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []

def download_stock_data(ticker_batch, start_date, end_date):
    """Download data for a batch of tickers"""
    all_data = []
    for ticker in ticker_batch:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                print(f"No data available for {ticker}")
                continue
                
            data.reset_index(inplace=True)
            data['Stock_Ticker'] = ticker
            data['Stock_Name'] = ticker  # Adjust if actual names are available
            data = data[['Date', 'Close', 'Stock_Ticker', 'Stock_Name']]
            data.rename(columns={'Close': 'Price'}, inplace=True)
            data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')
            all_data.append(data)
            print(f'Downloaded data for {ticker}')
        except Exception as e:
            print(f'Failed to download data for {ticker}: {e}')
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def load_checkpoint():
    """Load checkpoint data if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_batches": 0, "last_update": None}

def save_checkpoint(batch_index):
    """Save checkpoint data"""
    checkpoint = {
        "processed_batches": batch_index,
        "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def update_data_files():
    """Update stock data files with incremental processing"""
    # Determine date range
    if os.path.exists(CSV_FILE):
        # Load existing data to get the last date
        try:
            existing_data = pd.read_csv(CSV_FILE)
            existing_data['Date'] = pd.to_datetime(existing_data['Date'], dayfirst=True)
            last_date = existing_data['Date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f'Updating data from {start_date}')
        except Exception as e:
            print(f"Error reading existing data: {e}")
            start_date = '2020-01-01'
    else:
        start_date = '2020-01-01'
        print(f'Downloading initial data from {start_date}')

    end_date = datetime.today().strftime('%Y-%m-%d')

    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print('Data is already up-to-date.')
        return

    # Get ticker batches
    ticker_batches = get_tickers()
    if not ticker_batches:
        print("No tickers available to process.")
        return

    # Load checkpoint to resume if interrupted
    checkpoint = load_checkpoint()
    start_batch = checkpoint["processed_batches"]
    
    # Process each batch of tickers
    for i, ticker_batch in enumerate(ticker_batches[start_batch:], start_batch):
        print(f"Processing batch {i+1} of {len(ticker_batches)} ({len(ticker_batch)} tickers)")
        
        # Download data for this batch
        batch_data = download_stock_data(ticker_batch, start_date, end_date)
        
        if batch_data.empty:
            print('No data found for this batch, moving to next batch.')
            save_checkpoint(i+1)
            continue
            
        # Append or create data files
        if os.path.exists(CSV_FILE):
            # Read existing data
            if i == start_batch:  # First batch in this run
                # We already loaded it earlier
                updated_data = pd.concat([existing_data, batch_data], ignore_index=True)
            else:
                # Just append to the file
                batch_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
                continue
        else:
            updated_data = batch_data
            
        # Save complete data for first batch or if creating files for the first time
        if i == start_batch or not os.path.exists(CSV_FILE):
            updated_data.to_csv(CSV_FILE, index=False)
            updated_data.to_parquet(PARQUET_FILE, engine='pyarrow', index=False)
            print(f"Saved initial/updated data files")
        
        # Save checkpoint
        save_checkpoint(i+1)
        
        # Pause between batches to avoid rate limiting
        if i < len(ticker_batches) - 1:
            print(f"Pausing for {BATCH_DELAY} seconds before next batch...")
            time.sleep(BATCH_DELAY)
    
    print("Data update complete.")
    # Final data consolidation to ensure parquet is updated
    if os.path.exists(CSV_FILE):
        pd.read_csv(CSV_FILE).to_parquet(PARQUET_FILE, engine='pyarrow', index=False)
    
    # Reset checkpoint after successful completion
    save_checkpoint(0)

# Execute the update
if __name__ == '__main__':
    update_data_files()
