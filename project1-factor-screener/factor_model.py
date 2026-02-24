import pandas as pd
import numpy as np

def compute_factor_scores(fundamentals_df, momentum_dict):
    df = fundamentals_df.copy()
    
    # Add momentum data
    df['momentum'] = df['ticker'].map(momentum_dict)
    
    # Drop rows with too much missing data (no PE ratio and no momentum)
    df = df.dropna(subset=['pe_ratio', 'momentum'], how='all')

    # === VALUE FACTOR ===
    # Lower P/E and EV/EBITDA = cheaper = better
    # We rank in reverse (rank ascending, so rank 1 = lowest P/E = best)
    # Filter out negative P/E (unprofitable companies)
    df['pe_rank'] = df['pe_ratio'].where(df['pe_ratio'] > 0).rank(
        ascending=True, pct=True  # pct=True gives percentile rank 0 to 1
    )
    df['ev_ebitda_rank'] = df['ev_ebitda'].where(df['ev_ebitda'] > 0).rank(
        ascending=True, pct=True
    )
    df['value_score'] = (df['pe_rank'].fillna(0.5) + 
                         df['ev_ebitda_rank'].fillna(0.5)) / 2
    
    # === MOMENTUM FACTOR ===
    # Higher momentum = better
    df['momentum_score'] = df['momentum'].rank(
        ascending=False, pct=True
    ).fillna(0.5)

    # === QUALITY FACTOR ===
    # Higher ROE and gross margin = better
    df['roe_rank'] = df['roe'].rank(ascending=False, pct=True)
    df['margin_rank'] = df['gross_margin'].rank(ascending=False, pct=True)
    df['quality_score'] = (df['roe_rank'].fillna(0.5) + 
                           df['margin_rank'].fillna(0.5)) / 2

    # === COMPOSITE SCORE ===
    # Equal-weight the three factors (you can adjust these weights)
    df['composite_score'] = (
        df['value_score'] * 0.30 + 
        df['momentum_score'] * 0.40 + 
        df['quality_score'] * 0.30
    )

    # Rank from best (lowest composite) to worst
    df['final_rank'] = df['composite_score'].rank(ascending=True)
    df = df.sort_values('final_rank')
    
    return df

def build_portfolio(scored_df, n_stocks=20):
    portfolio = scored_df.head(n_stocks).copy()
    # Equal-weight portfolio
    portfolio['weight'] = 1.0 / n_stocks
    return portfolio[['ticker', 'name', 'sector', 'pe_ratio', 
                      'ev_ebitda', 'roe', 'gross_margin', 'momentum',
                      'composite_score', 'final_rank', 'weight']]