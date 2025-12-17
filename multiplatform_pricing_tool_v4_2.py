# v4.2 — Ultimate Edition: v2.5's Comprehensive Scraping + v4.1's Fixes + Tier System
import sys
import streamlit as real_st

# ---- Import v2_5 behind a shim to suppress its UI during import ----
class _DummyCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return None
    def columns(self, *a, **k): return [self, self]
    def progress(self, *a, **k): return self
    def empty(self, *a, **k): return self
    def update(self, *a, **k): return None
    def write(self, *a, **k): return None
    def text(self, *a, **k): return None
    def markdown(self, *a, **k): return None
    def dataframe(self, *a, **k): return None
    def caption(self, *a, **k): return None
    def info(self, *a, **k): return None
    def warning(self, *a, **k): return None
    def error(self, *a, **k): return None
    def success(self, *a, **k): return None
    def button(self, *a, **k): return False
    def text_input(self, *a, **k): return ""
    def text_area(self, *a, **k): return ""
    def selectbox(self, *a, **k): return ""
    def multiselect(self, *a, **k): return []
    def toggle(self, *a, **k): return False
    def download_button(self, *a, **k): return None
    def number_input(self, *a, **k): return 1.0

class _Shim(_DummyCtx):
    def __init__(self):
        self.session_state = {}
        self.sidebar = _DummyCtx()
    def set_page_config(self, *a, **k): return None

shim = _Shim()
sys.modules['streamlit'] = shim
import multiplatform_pricing_tool_v2_5 as v25  # noqa: E402
sys.modules['streamlit'] = real_st
v25.st = real_st
st = real_st

# ---- App code ----
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Game Pricing — v4.2 Ultimate", page_icon="🎮", layout="wide")

# ==========================================
# CONFIGURATION - FIXED XBOX URLs FROM v4.1
# ==========================================

# Currency symbols for formatting
CURRENCY_SYMBOLS = {
    'USD': '$', 'CAD': 'CA$', 'EUR': '€', 'GBP': '£', 'AUD': 'A$', 'NZD': 'NZ$',
    'JPY': '¥', 'KRW': '₩', 'CNY': '¥', 'HKD': 'HK$', 'TWD': 'NT$', 'SGD': 'S$',
    'THB': '฿', 'MYR': 'RM', 'IDR': 'Rp', 'PHP': '₱', 'VND': '₫', 'INR': '₹',
    'AED': 'AED', 'SAR': 'SAR', 'ZAR': 'R', 'BRL': 'R$', 'ARS': 'ARS$',
    'CLP': 'CLP$', 'COP': 'COL$', 'MXN': 'Mex$', 'PEN': 'S/', 'UYU': '$U',
    'RUB': '₽', 'TRY': '₺', 'UAH': '₴', 'PLN': 'zł', 'CHF': 'CHF', 'SEK': 'kr',
    'NOK': 'kr', 'DKK': 'kr.', 'CZK': 'Kč', 'HUF': 'Ft', 'ILS': '₪', 'QAR': 'QR'
}

TIER_CONFIGS = {
    "AAA": {
        "Steam": [
            ("The Outer Worlds 2", "https://store.steampowered.com/app/1449110/The_Outer_Worlds_2/", 1.0, 1.0),
            ("Madden NFL 26", "https://store.steampowered.com/app/3230400/Madden_NFL_26/", 1.0, 1.0),
            ("Call of Duty: Black Ops 6", "https://store.steampowered.com/app/2933620/Call_of_Duty_Black_Ops_6/", 1.0, 1.0),
            ("Borderlands 4", "https://store.steampowered.com/app/1285190/Borderlands_4/", 1.0, 1.0),
            ("NBA 2K26", "https://store.steampowered.com/app/3472040/NBA_2K26/", 1.0, 1.0),
        ],
        "Xbox": [
            # ✅ FIXED URLs from v4.1
            ("The Outer Worlds 2", "https://www.xbox.com/en-US/games/store/p/9NSPRSXXZZLG/0017", 1.0, 1.0),
            ("Madden NFL 26", "https://www.xbox.com/en-us/games/store/ea-sports-madden-nfl-26/9nvd16np4j8t", 1.0, 1.0),
            ("Call of Duty: Black Ops 6", "https://www.xbox.com/en-us/games/store/call-of-duty-black-ops-6-cross-gen-bundle/9pf528m6crhq", 1.0, 1.0),
            ("Borderlands 4", "https://www.xbox.com/en-us/games/store/borderlands-4/9m6bhbh5647g", 1.0, 1.0),
            ("NBA 2K26", "https://www.xbox.com/en-us/games/store/nba-2k26-standard-edition/9pj2rvrc0l1x", 1.0, 1.0),
        ],
        "PlayStation": [
            ("The Outer Worlds 2", "https://store.playstation.com/en-us/product/UP6312-PPSA24588_00-0872768154966924", 1.0, 1.0),
            ("Madden NFL 26", "https://store.playstation.com/en-us/product/UP0006-PPSA26127_00-MADDENNFL26GAME0", 1.0, 1.0),
            ("Call of Duty: Black Ops 6", "https://store.playstation.com/en-us/product/UP0002-PPSA01649_00-CODBO6CROSSGEN01", 1.0, 1.0),
            ("Borderlands 4", "https://store.playstation.com/en-us/product/UP1001-PPSA01494_00-000000000000OAK2", 1.0, 1.0),
            ("NBA 2K26", "https://store.playstation.com/en-us/product/UP1001-PPSA28420_00-NBA2K26000000000", 1.0, 1.0),
        ],
    },
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_weights(weights):
    """Auto-normalize weights to percentages"""
    total = sum(weights)
    if total == 0:
        return [0.0] * len(weights)
    return [w / total for w in weights]

def calculate_weighted_recommendations(pricing_data, tier_config, usd_rates):
    """Calculate weighted pricing recommendations using actual pricing data"""
    recommendations = []
    
    for platform in ['Steam', 'Xbox', 'PlayStation']:
        if platform not in pricing_data or not pricing_data[platform]:
            continue
        
        games_config = tier_config[platform]
        scales = [game[2] for game in games_config]
        weights = [game[3] for game in games_config]
        normalized_weights = normalize_weights(weights)
        
        # Group by country
        countries = set()
        for game_prices in pricing_data[platform].values():
            countries.update(game_prices.keys())
        
        for country in countries:
            weighted_sum = 0
            total_weight = 0
            count = 0
            local_currency = None
            
            for idx, (game_name, _, scale, weight) in enumerate(games_config):
                if game_name in pricing_data[platform]:
                    game_prices = pricing_data[platform][game_name]
                    if country in game_prices:
                        price_data = game_prices[country]
                        usd_price = price_data['price_usd']
                        
                        # Skip if price_usd is None
                        if usd_price is None or usd_price <= 0:
                            continue
                        
                        # Apply scale factor and weight
                        weighted_price = usd_price * scale * normalized_weights[idx]
                        weighted_sum += weighted_price
                        total_weight += normalized_weights[idx]
                        count += 1
                        
                        if local_currency is None:
                            local_currency = price_data['currency']
            
            if count > 0 and total_weight > 0:
                # Calculate recommendation in USD
                rec_usd = weighted_sum / total_weight
                
                # Convert back to local currency
                rate = usd_rates.get(local_currency, 1.0)
                rec_local = rec_usd * rate
                
                # Format with proper decimals
                currency_symbol = CURRENCY_SYMBOLS.get(local_currency, '')
                recommendations.append({
                    'Platform': platform,
                    'Country': v25.COUNTRY_NAMES.get(country, country),
                    'Currency': local_currency,
                    'Recommended (USD)': f"${rec_usd:.2f}",
                    'Recommended (Local)': f"{currency_symbol}{rec_local:.2f}",
                    'Games Included': count
                })
    
    return pd.DataFrame(recommendations)

# ==========================================
# MAIN APP
# ==========================================

st.title("🎮 Game Pricing Tool — v4.2 Ultimate Edition")
st.success("""
✨ **v4.2 Features**:
- 📊 Comprehensive Coverage: 54 Steam markets, 47 Xbox markets, 41 PlayStation markets
- 🔧 Fixed Xbox URLs: All 5 AAA games now scrape correctly
- ⚡ Multi-threaded: Fast parallel scraping with progress tracking
- 💰 Weighted Recommendations: Configurable scale factors & weights per game
- 📈 Tier System: AAA competitive set with expansion capability
""")

# Initialize session state
if 'tier_configs' not in st.session_state:
    st.session_state.tier_configs = TIER_CONFIGS.copy()
if 'pricing_data' not in st.session_state:
    st.session_state.pricing_data = {}
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# Tier selector
tier = st.selectbox("Select Tier", ["AAA"], key="tier_selector")

st.divider()

# ==========================================
# COMPETITIVE SET EDITOR
# ==========================================

st.subheader(f"📝 {tier} Tier - Competitive Set Configuration")

for platform in ['Steam', 'Xbox', 'PlayStation']:
    with st.expander(f"{platform} Games", expanded=False):
        games = st.session_state.tier_configs[tier][platform]
        
        cols = st.columns([3, 2, 1, 1, 1])
        cols[0].write("**Game Title**")
        cols[1].write("**URL**")
        cols[2].write("**Scale**")
        cols[3].write("**Weight**")
        cols[4].write("**Action**")
        
        updated_games = []
        
        for idx, game in enumerate(games):
            cols = st.columns([3, 2, 1, 1, 1])
            
            name = cols[0].text_input(f"Name {idx}", value=game[0], key=f"{tier}_{platform}_name_{idx}", label_visibility="collapsed")
            url = cols[1].text_input(f"URL {idx}", value=game[1], key=f"{tier}_{platform}_url_{idx}", label_visibility="collapsed")
            scale = cols[2].number_input(f"Scale {idx}", value=float(game[2]), min_value=0.1, max_value=5.0, step=0.05, key=f"{tier}_{platform}_scale_{idx}", label_visibility="collapsed")
            weight = cols[3].number_input(f"Weight {idx}", value=float(game[3]), min_value=0.0, step=0.1, key=f"{tier}_{platform}_weight_{idx}", label_visibility="collapsed")
            
            if cols[4].button("🗑️", key=f"{tier}_{platform}_delete_{idx}"):
                pass
            else:
                updated_games.append((name, url, scale, weight))
        
        st.session_state.tier_configs[tier][platform] = updated_games
        
        if st.button(f"➕ Add {platform} Game", key=f"{tier}_{platform}_add"):
            st.session_state.tier_configs[tier][platform].append(("New Game", "", 1.0, 1.0))
            st.rerun()

st.divider()

# ==========================================
# PRICE PULLER
# ==========================================

if st.button("🚀 Pull Prices from All Markets", type="primary", use_container_width=True):
    st.info("🔄 Fetching prices across 54 Steam markets, 47 Xbox markets, and 41 PlayStation markets...")
    
    # Parse Steam games
    steam_games = []
    for name, url, _, _ in st.session_state.tier_configs[tier]['Steam']:
        if name and url:
            appid = v25.extract_steam_appid(url)
            if appid:
                steam_games.append((appid, name))
    
    # Parse Xbox games  
    xbox_games = []
    for name, url, _, _ in st.session_state.tier_configs[tier]['Xbox']:
        if name and url:
            store_id = v25.extract_xbox_store_id(url)
            if store_id:
                xbox_games.append((store_id, name))
    
    # Parse PlayStation games
    ps_games = []
    for name, url, _, _ in st.session_state.tier_configs[tier]['PlayStation']:
        if name and url:
            product_id = v25.extract_ps_product_id(url)
            if product_id:
                ps_games.append((product_id, name))
    
    # Use v2.5's comprehensive multi-threaded scraping
    steam_results, xbox_results, ps_results = v25.pull_all_prices(
        steam_games, 
        xbox_games, 
        ps_games, 
        max_workers=20
    )
    
    # Get exchange rates
    rates = v25.fetch_exchange_rates()
    
    # Process results into DataFrame
    all_results = list(steam_results) + list(xbox_results) + list(ps_results)
    st.session_state.processed_df = v25.process_results(all_results, rates)
    
    # Also organize by platform and game for recommendations
    st.session_state.pricing_data[tier] = {'Steam': {}, 'Xbox': {}, 'PlayStation': {}}
    
    for result in steam_results:
        if result.title not in st.session_state.pricing_data[tier]['Steam']:
            st.session_state.pricing_data[tier]['Steam'][result.title] = {}
        # Format the price with currency symbol
        currency_symbol = CURRENCY_SYMBOLS.get(result.currency, '')
        formatted_price = f"{currency_symbol}{result.price:.2f}" if result.price else "N/A"
        st.session_state.pricing_data[tier]['Steam'][result.title][result.country] = {
            'price': result.price,
            'currency': result.currency,
            'price_usd': result.price_usd,
            'formatted': formatted_price
        }
    
    for result in xbox_results:
        if result.title not in st.session_state.pricing_data[tier]['Xbox']:
            st.session_state.pricing_data[tier]['Xbox'][result.title] = {}
        # Format the price with currency symbol
        currency_symbol = CURRENCY_SYMBOLS.get(result.currency, '')
        formatted_price = f"{currency_symbol}{result.price:.2f}" if result.price else "N/A"
        st.session_state.pricing_data[tier]['Xbox'][result.title][result.country] = {
            'price': result.price,
            'currency': result.currency,
            'price_usd': result.price_usd,
            'formatted': formatted_price
        }
    
    for result in ps_results:
        if result.title not in st.session_state.pricing_data[tier]['PlayStation']:
            st.session_state.pricing_data[tier]['PlayStation'][result.title] = {}
        # Format the price with currency symbol
        currency_symbol = CURRENCY_SYMBOLS.get(result.currency, '')
        formatted_price = f"{currency_symbol}{result.price:.2f}" if result.price else "N/A"
        st.session_state.pricing_data[tier]['PlayStation'][result.title][result.country] = {
            'price': result.price,
            'currency': result.currency,
            'price_usd': result.price_usd,
            'formatted': formatted_price
        }
    
    st.success(f"✅ Pulled {len(all_results)} price points! Ready for analysis.")

st.divider()

# ==========================================
# RESULTS TABS
# ==========================================

if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    tabs = st.tabs(["Steam Pricing", "Xbox Pricing", "PlayStation Pricing", "💡 Recommendations"])
    
    df = st.session_state.processed_df
    
    # Steam Tab
    with tabs[0]:
        st.subheader("Steam Pricing")
        steam_df = df[df['Platform'] == 'Steam'] if 'Platform' in df.columns else pd.DataFrame()
        if not steam_df.empty:
            st.caption(f"📊 {len(steam_df)} price points across {steam_df['Country'].nunique()} countries")
            st.dataframe(steam_df, use_container_width=True)
            st.download_button("⬇️ Download Steam CSV", 
                             steam_df.to_csv(index=False).encode("utf-8"),
                             "steam_prices.csv", "text/csv")
        else:
            st.info("No Steam data available")
    
    # Xbox Tab
    with tabs[1]:
        st.subheader("Xbox Pricing")
        xbox_df = df[df['Platform'] == 'Xbox'] if 'Platform' in df.columns else pd.DataFrame()
        if not xbox_df.empty:
            st.caption(f"📊 {len(xbox_df)} price points across {xbox_df['Country'].nunique()} countries")
            st.dataframe(xbox_df, use_container_width=True)
            st.download_button("⬇️ Download Xbox CSV", 
                             xbox_df.to_csv(index=False).encode("utf-8"),
                             "xbox_prices.csv", "text/csv")
        else:
            st.info("No Xbox data available")
    
    # PlayStation Tab
    with tabs[2]:
        st.subheader("PlayStation Pricing")
        ps_df = df[df['Platform'] == 'PlayStation'] if 'Platform' in df.columns else pd.DataFrame()
        if not ps_df.empty:
            st.caption(f"📊 {len(ps_df)} price points across {ps_df['Country'].nunique()} countries")
            st.dataframe(ps_df, use_container_width=True)
            st.download_button("⬇️ Download PlayStation CSV", 
                             ps_df.to_csv(index=False).encode("utf-8"),
                             "playstation_prices.csv", "text/csv")
        else:
            st.info("No PlayStation data available")
    
    # Recommendations Tab
    with tabs[3]:
        st.subheader("💡 Weighted Pricing Recommendations")
        st.caption("Calculated using: (Game1_USD × Scale1 × Weight1 + ...) / Total_Weight")
        
        if tier in st.session_state.pricing_data and st.session_state.pricing_data[tier]:
            rates = v25.fetch_exchange_rates()
            rec_df = calculate_weighted_recommendations(
                st.session_state.pricing_data[tier],
                st.session_state.tier_configs[tier],
                rates
            )
            
            if not rec_df.empty:
                st.caption(f"📊 {len(rec_df)} market recommendations across all platforms")
                st.dataframe(rec_df, use_container_width=True)
                st.download_button("⬇️ Download Recommendations CSV", 
                                 rec_df.to_csv(index=False).encode("utf-8"),
                                 "recommendations.csv", "text/csv")
            else:
                st.warning("No recommendations generated - insufficient price data")
        else:
            st.info("Pull prices first to generate recommendations")
else:
    st.info("👆 Click 'Pull Prices' to fetch comprehensive pricing data from all global markets")

st.divider()
st.caption("Powered by v2.5's comprehensive scraping engine + v4.1's fixes + weighted recommendation system")
