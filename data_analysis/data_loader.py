import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths for data storage
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(DATA_DIR, 'stocks_data.csv')
PARQUET_FILE = os.path.join(DATA_DIR, 'stocks_data.parquet')
CHECKPOINT_FILE = os.path.join(DATA_DIR, 'download_checkpoint.json')

# Batch size for processing tickers (reduced to avoid rate limiting)
BATCH_SIZE = 10
# Delay between batches to avoid rate limiting (seconds)
BATCH_DELAY = 0

def get_tickers():
    """Fetch stock tickers from S&P 500 and return in manageable batches"""
    logging.info("Fetching S&P 500 ticker list...")
    try:
        # Get S&P 500 tickers from Wikipedia
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_tickers = tables[0]['Symbol'].tolist()
        
        # Clean tickers: remove any that contain dots or special characters
        clean_tickers = []
        for ticker in sp500_tickers:
            if isinstance(ticker, str) and len(ticker) > 0 and '.' not in ticker:
                clean_tickers.append(ticker.strip())
        
        # Remove duplicates and sort
        clean_tickers = sorted(list(set(clean_tickers)))
        logging.info(f"Found {len(clean_tickers)} unique S&P 500 tickers")
        
        # Split tickers into batches
        batches = [clean_tickers[i:i + BATCH_SIZE] for i in range(0, len(clean_tickers), BATCH_SIZE)]
        return batches
        
    except Exception as e:
        logging.error(f"Error fetching tickers: {e}")
        return []

def download_stock_data(ticker_batch, start_date, end_date):
    """Download comprehensive stock data for a batch of tickers"""
    all_data = []
    
    for ticker in ticker_batch:
        try:
            logging.info(f"Downloading data for {ticker}")
            
            # Download data with all available fields
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if data.empty:
                logging.warning(f"No data available for {ticker}")
                continue
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Add ticker information
            data['symbol'] = ticker
            
            # Rename columns to match your CSV structure
            data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            }, inplace=True)
            
            # Add missing columns with default values
            data['capital_gains'] = 0.0
            data['sector'] = 'Unknown'  # You'd need to fetch this separately
            
            # Select and reorder columns
            columns_order = ['date', 'symbol', 'open', 'high', 'low', 'close', 
                           'volume', 'dividends', 'stock_splits', 'capital_gains', 'sector']
            data = data[columns_order]
            
            # Format date
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            all_data.append(data)
            logging.info(f'Successfully downloaded {len(data)} records for {ticker}')
            
            # Small delay between individual ticker downloads
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f'Failed to download data for {ticker}: {e}')
            continue
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        logging.info(f"Batch complete: {len(combined_data)} total records")
        return combined_data
    else:
        return pd.DataFrame()

def load_checkpoint():
    """Load checkpoint data if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
    return {"processed_batches": 0, "last_update": None}

def save_checkpoint(batch_index, total_batches):
    """Save checkpoint data"""
    checkpoint = {
        "processed_batches": batch_index,
        "total_batches": total_batches,
        "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logging.info(f"Checkpoint saved: batch {batch_index}/{total_batches}")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")

def get_sector_data():
    """Fetch sector information for S&P 500 companies"""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sector_data = tables[0][['Symbol', 'GICS Sector']].copy()
        sector_data.columns = ['symbol', 'sector']
        return dict(zip(sector_data['symbol'], sector_data['sector']))
    except Exception as e:
        logging.error(f"Error fetching sector data: {e}")
        return {}

def update_data_files():
    """Update stock data files with incremental processing"""
    logging.info("Starting data update process...")
    
    # Get sector mapping
    sector_mapping = get_sector_data()
    
    # Determine date range
    if os.path.exists(CSV_FILE):
        try:
            existing_data = pd.read_csv(CSV_FILE)
            existing_data['date'] = pd.to_datetime(existing_data['date'])
            last_date = existing_data['date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            logging.info(f'Updating data from {start_date}')
        except Exception as e:
            logging.error(f"Error reading existing data: {e}")
            start_date = '2020-01-01'
            existing_data = pd.DataFrame()
    else:
        start_date = '2020-01-01'
        existing_data = pd.DataFrame()
        logging.info(f'Downloading initial data from {start_date}')

    end_date = datetime.today().strftime('%Y-%m-%d')

    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        logging.info('Data is already up-to-date.')
        return

    # Get ticker batches
    ticker_batches = get_tickers()
    if not ticker_batches:
        logging.error("No tickers available to process.")
        return

    # Load checkpoint to resume if interrupted
    checkpoint = load_checkpoint()
    start_batch = checkpoint.get("processed_batches", 0)
    
    logging.info(f"Processing {len(ticker_batches)} batches starting from batch {start_batch + 1}")
    
    # Initialize data container
    all_new_data = []
    
    # Process each batch of tickers
    for i, ticker_batch in enumerate(ticker_batches[start_batch:], start_batch):
        logging.info(f"Processing batch {i+1} of {len(ticker_batches)} ({len(ticker_batch)} tickers)")
        
        # Download data for this batch
        batch_data = download_stock_data(ticker_batch, start_date, end_date)
        
        if batch_data.empty:
            logging.warning('No data found for this batch, moving to next batch.')
        else:
            # Update sector information if available
            if sector_mapping:
                batch_data['sector'] = batch_data['symbol'].map(sector_mapping).fillna('Unknown')
            
            all_new_data.append(batch_data)
        
        # Save checkpoint
        save_checkpoint(i + 1, len(ticker_batches))
        
        # Pause between batches to avoid rate limiting
        if i < len(ticker_batches) - 1:
            logging.info(f"Pausing for {BATCH_DELAY} seconds before next batch...")
            time.sleep(BATCH_DELAY)
    
    # Combine all new data
    if all_new_data:
        new_data = pd.concat(all_new_data, ignore_index=True)
        logging.info(f"Downloaded {len(new_data)} new records")
        
        # Convert new_data date column to datetime (it's currently string from download_stock_data)
        new_data['date'] = pd.to_datetime(new_data['date'])
        
        # Combine with existing data
        if not existing_data.empty:
            # existing_data['date'] is already datetime from above
            final_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates based on date and symbol
            final_data = final_data.drop_duplicates(subset=['date', 'symbol'], keep='last')
        else:
            final_data = new_data
        
        # Sort by date and symbol
        final_data = final_data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Convert date back to string for CSV storage - ensure it's datetime first
        if final_data['date'].dtype != 'datetime64[ns]':
            final_data['date'] = pd.to_datetime(final_data['date'])
        final_data['date'] = final_data['date'].dt.strftime('%Y-%m-%d')
        
        # Save files
        try:
            final_data.to_csv(CSV_FILE, index=False)
            final_data.to_parquet(PARQUET_FILE, engine='pyarrow', index=False)
            logging.info(f"Saved {len(final_data)} records to data files")
        except Exception as e:
            logging.error(f"Error saving data files: {e}")
            return
    else:
        logging.warning("No new data was downloaded.")
    
    # Reset checkpoint after successful completion
    save_checkpoint(0, 0)
    logging.info("Data update complete!")

# Execute the update
if __name__ == '__main__':
    update_data_files()
