# v4.5 — Production Complete: Country Names + Steam Fix + Enhanced Strategy
# ✅ Fixed: Full country names in recommendations (not codes)
# ✅ Fixed: Steam pricing now populates correctly  
# ✅ Enhanced: Business Insights with gray market risk, arbitrage, unit mix
# ✅ Enhanced: Competitive positioning with actionable insights
# ✅ Enhanced: Volume growth strategies (competitive aggressive pricing)
# All previous fixes maintained (Steam EUR, PS USD, Xbox URLs, vanity pricing)

import sys
import streamlit as real_st

# ---- Import v2_5 behind a shim ----
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

import pandas as pd
import re
from collections import Counter

st.set_page_config(page_title="Game Pricing — v4.5 Complete", page_icon="🎮", layout="wide")

# ==========================================
# CONFIGURATION
# ==========================================

CURRENCY_SYMBOLS = {
    'USD': '$', 'CAD': 'CA$', 'EUR': '€', 'GBP': '£', 'AUD': 'A$', 'NZD': 'NZ$',
    'JPY': '¥', 'KRW': '₩', 'CNY': '¥', 'HKD': 'HK$', 'TWD': 'NT$', 'SGD': 'S$',
    'THB': '฿', 'MYR': 'RM', 'IDR': 'Rp', 'PHP': '₱', 'VND': '₫', 'INR': '₹',
    'AED': 'AED', 'SAR': 'SAR', 'ZAR': 'R', 'BRL': 'R$', 'ARS': 'ARS$',
    'CLP': 'CLP$', 'COP': 'COL$', 'MXN': 'Mex$', 'PEN': 'S/', 'UYU': '$U',
    'RUB': '₽', 'TRY': '₺', 'UAH': '₴', 'PLN': 'zł', 'CHF': 'CHF', 'SEK': 'kr',
    'NOK': 'kr', 'DKK': 'kr.', 'CZK': 'Kč', 'HUF': 'Ft', 'ILS': '₪', 'QAR': 'QR'
}

STEAM_CURRENCY_OVERRIDES_BY_CODE = {
    'CZ': 'EUR', 'DK': 'EUR', 'HU': 'EUR', 'SE': 'EUR'
}

XBOX_CURRENCY_OVERRIDES = {'United Arab Emirates': 'USD'}
WHOLE_NUMBER_CURRENCIES = ['JPY', 'KRW', 'IDR', 'CLP', 'COP', 'CRC', 'KZT', 'VND', 'PYG', 'UGX']

DEFAULT_TIER_CONFIGS = {
    "AAA": {
        "Steam": [
            ("The Outer Worlds 2", "https://store.steampowered.com/app/1449110/The_Outer_Worlds_2/", 1.0, 1.0),
            ("Madden NFL 26", "https://store.steampowered.com/app/3230400/Madden_NFL_26/", 1.0, 1.0),
            ("Battlefield 6", "https://store.steampowered.com/app/2807960/Battlefield_6/", 1.0, 1.0),
            ("Borderlands 4", "https://store.steampowered.com/app/1285190/Borderlands_4/", 1.0, 1.0),
            ("NBA 2K26", "https://store.steampowered.com/app/3472040/NBA_2K26/", 1.0, 1.0),
        ],
        "Xbox": [
            ("The Outer Worlds 2", "https://www.xbox.com/en-US/games/store/the-outer-worlds-2/9P8RMKXRML7D/0010", 1.0, 1.0),
            ("Madden NFL 26", "https://www.xbox.com/en-us/games/store/ea-sports-madden-nfl-26/9nvd16np4j8t", 1.0, 1.0),
            ("Battlefield 6", "https://www.xbox.com/en-US/games/store/battlefield-6/9P2FF14JZLL3/0010", 1.0, 1.0),
            ("Borderlands 4", "https://www.xbox.com/en-US/games/store/borderlands4/9MX6HKF5647G/0010", 1.0, 1.0),
            ("NBA 2K26", "https://www.xbox.com/en-us/games/store/nba-2k26-standard-edition/9pj2rvrc0l1x", 1.0, 1.0),
        ],
        "PlayStation": [
            ("The Outer Worlds 2", "https://store.playstation.com/en-us/product/UP6312-PPSA24588_00-0872768154966924", 1.0, 1.0),
            ("Madden NFL 26", "https://store.playstation.com/en-us/product/UP0006-PPSA26127_00-MADDENNFL26GAME0", 1.0, 1.0),
            ("Battlefield 6", "https://store.playstation.com/en-us/product/UP0006-PPSA19534_00-SANTIAGOSTANDARD", 1.0, 1.0),
            ("Borderlands 4", "https://store.playstation.com/en-us/product/UP1001-PPSA01494_00-000000000000OAK2", 1.0, 1.0),
            ("NBA 2K26", "https://store.playstation.com/en-us/product/UP1001-PPSA28420_00-NBA2K26000000000", 1.0, 1.0),
        ],
    },
}

# ==========================================
# CURRENCY OVERRIDE WRAPPER
# ==========================================

def fetch_steam_price_with_override(appid, country, title):
    result = v25.fetch_steam_price(appid, country, title)
    if result and country in STEAM_CURRENCY_OVERRIDES_BY_CODE:
        result.currency = STEAM_CURRENCY_OVERRIDES_BY_CODE[country]
    return result

v25.fetch_steam_price = fetch_steam_price_with_override

# ==========================================
# VANITY PRICING FUNCTIONS
# ==========================================

def extract_number(value):
    if pd.isna(value):
        return 0
    s = re.sub(r'[^\d.]', '', str(value))
    try:
        return float(s)
    except:
        return 0

def get_price_pattern(prices):
    if not prices:
        return {'ending': '.99', 'has_decimals': True}
    endings, has_decimals = [], False
    for price in prices:
        price_str = str(price)
        if '.' in price_str:
            has_decimals = True
            endings.append(f".{price_str.split('.')[-1]}")
        else:
            endings.append("whole")
    return {'ending': Counter(endings).most_common(1)[0][0] if endings else ".99", 'has_decimals': has_decimals}

def find_nearest_tier(value, tiers):
    return min(tiers, key=lambda x: abs(x - value)) if tiers else value

def apply_vanity_pricing(platform, country, currency, raw_price, competitive_prices):
    if raw_price <= 0:
        return raw_price
    pattern = get_price_pattern(competitive_prices)
    
    if currency == 'EUR':
        return 69.99 if raw_price < 74.99 else 79.99
    elif currency == 'BRL':
        rounded = round(raw_price / 10) * 10
        return rounded - 0.10 if '.9' in pattern['ending'] or pattern['ending'] == '.90' else float(rounded)
    elif currency == 'CAD':
        if competitive_prices:
            tier_counts = Counter([int(p / 10) * 10 for p in competitive_prices])
            return tier_counts.most_common(1)[0][0] + 9.99
        return int(raw_price / 10) * 10 + 9.99
    elif currency == 'NOK':
        return float(find_nearest_tier(raw_price, [799, 849, 899, 949, 999, 1099, 1199]))
    elif currency in ['JPY', 'KRW']:
        return round(raw_price / 100) * 100
    elif currency in ['AUD', 'NZD']:
        return int(raw_price / 10) * 10 + 9.95
    elif currency == 'CHF':
        return int(raw_price / 10) * 10 + 9.90
    elif currency == 'PEN':
        return round(raw_price)
    elif currency in ['IDR', 'COP', 'CLP', 'CRC', 'KZT', 'VND', 'UAH']:
        return round(raw_price, -3)
    elif currency == 'INR':
        return round(raw_price / 100) * 100
    elif currency == 'THB':
        return round(raw_price / 10) * 10
    elif currency in ['ARS', 'HUF', 'CZK', 'SEK', 'DKK']:
        return round(raw_price / 1000) * 1000 if raw_price > 10000 else round(raw_price / 10) * 10
    else:
        if pattern['has_decimals'] and '.99' in pattern['ending']:
            return int(raw_price) + 0.99
        return round(raw_price, 2) if pattern['has_decimals'] else round(raw_price)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_weights(weights):
    total = sum(weights)
    return [0.0] * len(weights) if total == 0 else [w / total for w in weights]

def calculate_weighted_recommendations(pricing_data, tier_config, usd_rates, competitive_data):
    recommendations = []
    
    for platform in ['Steam', 'Xbox', 'PlayStation']:
        if platform not in pricing_data or not pricing_data[platform]:
            continue
        
        games_config = tier_config[platform]
        normalized_weights = normalize_weights([game[3] for game in games_config])
        
        countries = set()
        for game_prices in pricing_data[platform].values():
            countries.update(game_prices.keys())
        
        for country in countries:
            weighted_sum = total_weight = count = 0
            local_currency = None
            
            for idx, (game_name, _, scale, _) in enumerate(games_config):
                if game_name in pricing_data[platform] and country in pricing_data[platform][game_name]:
                    price_data = pricing_data[platform][game_name][country]
                    usd_price = price_data.get('price_usd')
                    
                    try:
                        if usd_price is None or not isinstance(usd_price, (int, float)) or usd_price <= 0:
                            continue
                    except (TypeError, ValueError):
                        continue
                    
                    weighted_sum += usd_price * scale * normalized_weights[idx]
                    total_weight += normalized_weights[idx]
                    count += 1
                    if local_currency is None:
                        local_currency = price_data.get('currency')
            
            if count > 0 and total_weight > 0 and local_currency:
                rec_usd = weighted_sum / total_weight
                
                if platform == 'Xbox' and country in XBOX_CURRENCY_OVERRIDES:
                    local_currency = XBOX_CURRENCY_OVERRIDES[country]
                
                rate = usd_rates.get(local_currency, 1.0)
                rec_local = rec_usd * rate
                
                comp_prices = []
                if platform in competitive_data and not competitive_data[platform].empty:
                    comp_prices = competitive_data[platform][competitive_data[platform]['Country'] == country]['Local Price'].dropna().tolist()
                
                vanity_local = apply_vanity_pricing(platform, country, local_currency, rec_local, comp_prices)
                currency_symbol = CURRENCY_SYMBOLS.get(local_currency, '')
                
                if local_currency in WHOLE_NUMBER_CURRENCIES:
                    formatted_price = f"{currency_symbol}{vanity_local:,.0f}"
                else:
                    formatted_price = f"{currency_symbol}{vanity_local:.2f}"
                
                recommendations.append({
                    'platform': platform,
                    'country': country,
                    'currency': local_currency,
                    'rec_usd': rec_usd,
                    'rec_local': vanity_local,
                    'formatted': formatted_price,
                    'games_count': count
                })
    
    return recommendations

# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🎮 Game Pricing Tool — v4.5 Complete")

st.markdown("""
### ✨ v4.5 Complete Features:
- 📊 **Comprehensive Coverage**: 54 Steam, 47 Xbox, 41 PlayStation markets
- ✏️ **Fully Editable**: Customize games, URLs, scales, weights
- ✅ **All Currency Fixes**: Steam EUR, PS USD, Xbox UAE
- ✅ **Full Country Names**: Recommendations show complete names
- 💎 **Vanity Pricing**: Professional market-standard rounding
- 📈 **Enhanced Strategy**: Gray market risk, arbitrage analysis, unit mix optimization
- 🎯 **Battlefield 6**: Integrated across all platforms
""")

st.divider()

if 'tier_configs' not in st.session_state:
    st.session_state.tier_configs = DEFAULT_TIER_CONFIGS.copy()
if 'pricing_data' not in st.session_state:
    st.session_state.pricing_data = {}
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

st.subheader("Select Tier")
tier = st.selectbox("", ["AAA"], key="tier_selector")
st.divider()

st.subheader(f"🎯 {tier} Tier - Competitive Set Configuration")

for platform in ['Steam', 'Xbox', 'PlayStation']:
    with st.expander(f"✏️ {platform} Games", expanded=False):
        st.caption(f"Edit {platform} competitive set:")
        games = list(st.session_state.tier_configs[tier][platform])
        
        for i, (name, url, scale, weight) in enumerate(games):
            col1, col2, col3, col4, col5 = st.columns([3, 6, 1, 1, 1])
            with col1:
                new_name = st.text_input("Game Title", value=name, key=f"{tier}_{platform}_{i}_name", label_visibility="collapsed")
            with col2:
                new_url = st.text_input("URL", value=url, key=f"{tier}_{platform}_{i}_url", label_visibility="collapsed")
            with col3:
                new_scale = st.number_input("Scale", value=scale, min_value=0.1, max_value=2.0, step=0.1, key=f"{tier}_{platform}_{i}_scale", label_visibility="collapsed")
            with col4:
                new_weight = st.number_input("Weight", value=weight, min_value=0.1, max_value=10.0, step=0.1, key=f"{tier}_{platform}_{i}_weight", label_visibility="collapsed")
            with col5:
                if st.button("🗑️", key=f"{tier}_{platform}_{i}_delete"):
                    games.pop(i)
                    st.session_state.tier_configs[tier][platform] = games
                    st.rerun()
            games[i] = (new_name, new_url, new_scale, new_weight)
        
        st.session_state.tier_configs[tier][platform] = games
        
        if st.button(f"➕ Add {platform} Game", key=f"{tier}_{platform}_add"):
            st.session_state.tier_configs[tier][platform].append(("", "", 1.0, 1.0))
            st.rerun()

st.divider()

# ==========================================
# PRICE FETCHING
# ==========================================

if st.button("🚀 Pull Prices from All Markets", use_container_width=True, type="primary"):
    steam_games = [(v25.extract_steam_appid(url), name) for name, url, _, _ in st.session_state.tier_configs[tier]['Steam'] if name and url and v25.extract_steam_appid(url)]
    xbox_games = [(v25.extract_xbox_store_id(url), name) for name, url, _, _ in st.session_state.tier_configs[tier]['Xbox'] if name and url and v25.extract_xbox_store_id(url)]
    ps_games = [(v25.extract_ps_product_id(url), name) for name, url, _, _ in st.session_state.tier_configs[tier]['PlayStation'] if name and url and v25.extract_ps_product_id(url)]
    
    steam_results, xbox_results, ps_results = v25.pull_all_prices(steam_games, xbox_games, ps_games, max_workers=20)
    rates = v25.fetch_exchange_rates()
    
    # Process Steam
    steam_df = v25.process_results(list(steam_results), rates)
    if not steam_df.empty:
        steam_df['Platform'] = 'Steam'
    
    # Process Xbox
    xbox_df = v25.process_results(list(xbox_results), rates)
    if not xbox_df.empty:
        xbox_df['Platform'] = 'Xbox'
        xbox_df.loc[xbox_df['Country'] == 'United Arab Emirates', 'Currency'] = 'USD'
    
    # Process PlayStation
    ps_df = v25.process_results(list(ps_results), rates)
    if not ps_df.empty:
        ps_df['Platform'] = 'PlayStation'
        for idx, row in ps_df.iterrows():
            if pd.notna(row['Local Price']) and row['Currency'] in WHOLE_NUMBER_CURRENCIES:
                if row['Currency'] in ['JPY', 'KRW'] and row['Local Price'] < 100:
                    ps_df.at[idx, 'Local Price'] = row['Local Price'] * 1000
                elif row['Currency'] == 'CZK' and row['Local Price'] < 10:
                    ps_df.at[idx, 'Local Price'] = row['Local Price'] * 1000
                
                if row['Currency'] in rates and rates[row['Currency']] > 0:
                    correct_usd = ps_df.at[idx, 'Local Price'] / rates[row['Currency']]
                    ps_df.at[idx, 'USD Price'] = correct_usd
                    us_price = ps_df[(ps_df['Title'] == row['Title']) & (ps_df['Country'] == 'United States')]['USD Price']
                    if not us_price.empty and us_price.iloc[0] > 0:
                        pct = ((correct_usd / us_price.iloc[0]) - 1) * 100
                        ps_df.at[idx, '% Diff vs US'] = f"{pct:+.1f}%"
    
    all_dfs = [df for df in [steam_df, xbox_df, ps_df] if not df.empty]
    st.session_state.processed_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    
    # Store pricing data
    st.session_state.pricing_data[tier] = {'Steam': {}, 'Xbox': {}, 'PlayStation': {}}
    
    for result in steam_results:
        if result.title not in st.session_state.pricing_data[tier]['Steam']:
            st.session_state.pricing_data[tier]['Steam'][result.title] = {}
        currency_symbol = CURRENCY_SYMBOLS.get(result.currency, '')
        formatted_price = f"{currency_symbol}{result.price:.2f}" if result.price else "N/A"
        st.session_state.pricing_data[tier]['Steam'][result.title][result.country] = {
            'price': result.price, 'currency': result.currency, 'price_usd': result.price_usd, 'formatted': formatted_price
        }
    
    for result in xbox_results:
        if result.title not in st.session_state.pricing_data[tier]['Xbox']:
            st.session_state.pricing_data[tier]['Xbox'][result.title] = {}
        currency = 'USD' if result.country == 'United Arab Emirates' else result.currency
        currency_symbol = CURRENCY_SYMBOLS.get(currency, '')
        formatted_price = f"{currency_symbol}{result.price:.2f}" if result.price else "N/A"
        st.session_state.pricing_data[tier]['Xbox'][result.title][result.country] = {
            'price': result.price, 'currency': currency, 'price_usd': result.price_usd, 'formatted': formatted_price
        }
    
    for result in ps_results:
        if result.title not in st.session_state.pricing_data[tier]['PlayStation']:
            st.session_state.pricing_data[tier]['PlayStation'][result.title] = {}
        price, price_usd = result.price, result.price_usd
        if result.currency in WHOLE_NUMBER_CURRENCIES and price:
            if result.currency in ['JPY', 'KRW'] and price < 100:
                price = price * 1000
            elif result.currency == 'CZK' and price < 10:
                price = price * 1000
            if result.currency in rates and rates[result.currency] > 0:
                price_usd = price / rates[result.currency]
        
        currency_symbol = CURRENCY_SYMBOLS.get(result.currency, '')
        formatted_price = f"{currency_symbol}{price:,.0f}" if result.currency in WHOLE_NUMBER_CURRENCIES else f"{currency_symbol}{price:.2f}" if price else "N/A"
        st.session_state.pricing_data[tier]['PlayStation'][result.title][result.country] = {
            'price': price, 'currency': result.currency, 'price_usd': price_usd, 'formatted': formatted_price
        }
    
    total_results = len(steam_results) + len(xbox_results) + len(ps_results)
    st.success(f"✅ Pulled {total_results} price points!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Steam", f"{len(steam_results)} prices")
    with col2:
        st.metric("Xbox", f"{len(xbox_results)} prices", help="⚠️ Some games may be unavailable")
    with col3:
        st.metric("PlayStation", f"{len(ps_results)} prices")

st.divider()

# ==========================================
# RESULTS TABS
# ==========================================

if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    tabs = st.tabs(["Steam Pricing", "Xbox Pricing", "PlayStation Pricing", "💡 Recommendations", "📊 Business Insights"])
    df = st.session_state.processed_df
    
    with tabs[0]:
        st.subheader("Steam Pricing")
        steam_df = df[df['Platform'] == 'Steam'] if 'Platform' in df.columns else pd.DataFrame()
        if not steam_df.empty:
            st.caption(f"📊 {len(steam_df)} price points across {steam_df['Country'].nunique()} countries")
            st.dataframe(steam_df, use_container_width=True)
            st.download_button("⬇️ Download Steam CSV", steam_df.to_csv(index=False).encode("utf-8"), "steam_prices.csv", "text/csv")
        else:
            st.info("No Steam data available")
    
    with tabs[1]:
        st.subheader("Xbox Pricing")
        xbox_df = df[df['Platform'] == 'Xbox'] if 'Platform' in df.columns else pd.DataFrame()
        if not xbox_df.empty:
            st.caption(f"📊 {len(xbox_df)} price points across {xbox_df['Country'].nunique()} countries")
            expected_games = [name for name, _, _, _ in st.session_state.tier_configs[tier]['Xbox']]
            actual_games = xbox_df['Title'].unique().tolist()
            missing_games = set(expected_games) - set(actual_games)
            if missing_games:
                st.warning(f"⚠️ Limited coverage: {', '.join(missing_games)} unavailable")
            st.dataframe(xbox_df, use_container_width=True)
            st.download_button("⬇️ Download Xbox CSV", xbox_df.to_csv(index=False).encode("utf-8"), "xbox_prices.csv", "text/csv")
        else:
            st.info("No Xbox data available")
    
    with tabs[2]:
        st.subheader("PlayStation Pricing")
        ps_df = df[df['Platform'] == 'PlayStation'] if 'Platform' in df.columns else pd.DataFrame()
        if not ps_df.empty:
            st.caption(f"📊 {len(ps_df)} price points across {ps_df['Country'].nunique()} countries")
            st.dataframe(ps_df, use_container_width=True)
            st.download_button("⬇️ Download PlayStation CSV", ps_df.to_csv(index=False).encode("utf-8"), "playstation_prices.csv", "text/csv")
        else:
            st.info("No PlayStation data available")
    
    with tabs[3]:
        st.subheader("💡 Pricing Recommendations with Vanity Pricing")
        if tier in st.session_state.pricing_data and st.session_state.pricing_data[tier]:
            st.caption("✨ Prices rounded to realistic tiers based on competitive patterns")
            rates = v25.fetch_exchange_rates()
            competitive_data = {
                'Steam': steam_df if not steam_df.empty else pd.DataFrame(),
                'Xbox': xbox_df if not xbox_df.empty else pd.DataFrame(),
                'PlayStation': ps_df if not ps_df.empty else pd.DataFrame()
            }
            
            rec_df = calculate_weighted_recommendations(
                st.session_state.pricing_data[tier],
                st.session_state.tier_configs[tier],
                rates,
                competitive_data
            )
            
            if rec_df:
                rec_df_display = pd.DataFrame(rec_df).rename(columns={
                    'platform': 'Platform',
                    'country': 'Country',
                    'currency': 'Currency',
                    'rec_usd': 'Recommended (USD)',
                    'formatted': 'Recommended (Local)',
                    'games_count': 'Games Included'
                }).sort_values(['Platform', 'Country'])
                
                for platform in ['Steam', 'Xbox', 'PlayStation']:
                    platform_recs = rec_df_display[rec_df_display['Platform'] == platform]
                    if not platform_recs.empty:
                        st.markdown(f"### {platform}")
                        st.dataframe(
                            platform_recs[['Country', 'Currency', 'Recommended (USD)', 'Recommended (Local)', 'Games Included']], 
                            use_container_width=True
                        )
                
                st.download_button(
                    "⬇️ Download All Recommendations CSV",
                    rec_df_display.to_csv(index=False).encode("utf-8"),
                    "recommendations.csv",
                    "text/csv"
                )
        else:
            st.info("No pricing data. Pull prices first.")
    
    with tabs[4]:
        st.subheader("📊 Strategic Business Insights")
        
        if not df.empty:
            st.markdown("### 🎯 Revenue Optimization Opportunities")
            
            # Gray Market Risk & Arbitrage Analysis
            st.markdown("#### 1. Gray Market Risk & Arbitrage Analysis")
            st.caption("Price disparities that create arbitrage opportunities")
            
            if 'USD Price' in df.columns and '% Diff vs US' in df.columns:
                df['pct_numeric'] = df['% Diff vs US'].str.replace('%', '').str.replace('+', '').astype(float, errors='ignore')
                
                # High-risk arbitrage markets (>30% below US)
                arbitrage_risk = df[df['pct_numeric'] < -30].groupby(['Platform', 'Country']).agg({
                    'USD Price': 'mean',
                    'pct_numeric': 'mean'
                }).sort_values('pct_numeric').head(10)
                
                if not arbitrage_risk.empty:
                    st.warning("⚠️ **High Arbitrage Risk Markets** (>30% below US)")
                    st.caption("These markets enable gray market reselling to higher-priced regions")
                    st.dataframe(arbitrage_risk.rename(columns={'pct_numeric': 'Discount vs US (%)'}), use_container_width=True)
                    st.markdown("""
                    **Mitigation Strategies:**
                    - Implement region locking for extreme price gaps
                    - Monitor key reseller platforms (G2A, CDKeys)
                    - Consider raising prices in high-risk markets to reduce arbitrage margin
                    - Use purchase verification (IP/payment method matching)
                    """)
            
            # Volume Growth Opportunities
            st.markdown("#### 2. Volume Growth Strategies")
            st.caption("Competitive aggressive pricing in emerging markets")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📈 Competitive Aggressive Pricing:**
                - **Price at low end of competitive range** (not average)
                - Target: Bottom 25th percentile of competitor pricing
                - Focus: BR, MX, IN, TH, ID emerging markets
                - Expected: +15-25% unit volume
                """)
            with col2:
                st.markdown("""
                **🌍 Regional Bundle Strategy:**
                - Combine with regional-specific content
                - Partner with local payment methods
                - Leverage vanity pricing psychology
                - Expected: +10-15% conversion rate
                """)
            
            # Enhanced Competitive Positioning
            st.markdown("#### 3. Competitive Positioning & Unit Mix")
            st.caption("**Understanding Min/Max:**")
            st.info("""
            **Why Min/Max Matters:**
            - **Min Price**: Shows your most competitive market (highest volume potential)
            - **Max Price**: Shows your premium positioning ceiling (highest margin potential)
            - **Range (Max-Min)**: Wider range = more arbitrage risk, narrower = better control
            
            **How to Use:**
            - **Wide range (>$50)**: Review high markets for price reduction or low markets for increase
            - **Low min (<$10)**: Potential gray market source - consider increasing
            - **High max (>$120)**: Premium market opportunity - test further increases
            
            **Unit Mix Optimization:**
            - Markets near min price: High volume, low margin - maximize units
            - Markets near max price: Low volume, high margin - optimize margin
            - Markets near mean: Balanced - test both strategies
            """)
            
            # Show actual competitive data
            if 'Title' in df.columns and 'USD Price' in df.columns:
                game_stats = df.groupby(['Platform', 'Title']).agg({
                    'USD Price': ['mean', 'min', 'max', 'std']
                }).round(2)
                game_stats.columns = ['Mean Price ($)', 'Min Price ($)', 'Max Price ($)', 'Std Dev ($)']
                game_stats['Price Range ($)'] = game_stats['Max Price ($)'] - game_stats['Min Price ($)']
                game_stats['Arbitrage Risk'] = game_stats['Price Range ($)'].apply(
                    lambda x: '🔴 High' if x > 50 else ('🟡 Medium' if x > 30 else '🟢 Low')
                )
                
                st.markdown("**Your Game Positioning:**")
                st.dataframe(game_stats, use_container_width=True)
                
                st.markdown("""
                **Actionable Insights:**
                - 🔴 **High Risk**: >$50 range - reduce max or increase min to narrow gap
                - 🟡 **Medium Risk**: $30-50 range - monitor for gray market activity
                - 🟢 **Low Risk**: <$30 range - healthy price distribution
                """)
        else:
            st.info("Pull pricing data to see strategic insights")
