# v5.0 — Executive Strategy Edition: Net Revenue + VAT Intelligence + Regional Analysis
# ✅ Feature: Net Revenue Calculator (deducts VAT/GST before platform share)
# ✅ Feature: Regional Grouping (NA, EMEA, LATAM, APAC)
# ✅ Feature: Profitability Waterfall Visualization
# ✅ Maintains: All v4.5 fixes (Steam EUR, Xbox URLs, Vanity Pricing)

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

import pandas as pd
import re
import altair as alt
from collections import Counter

st.set_page_config(page_title="Game Revenue & Strategy — v5.0", page_icon="🚀", layout="wide")

# ==========================================
# INTELLIGENCE CONFIGURATION (VAT & REGIONS)
# ==========================================

# Estimated VAT/GST Rates for "Tax Inclusive" Digital Goods (2024/2025 Estimates)
# These are deducted from the List Price to find the "Pre-Tax" Basis.
# US/CA are usually Tax-Exclusive (Tax added at checkout), so Rate is 0.0 for Net Rev Calc.
VAT_RATES = {
    # Europe (High VAT)
    'United Kingdom': 0.20, 'Germany': 0.19, 'France': 0.20, 'Italy': 0.22,
    'Spain': 0.21, 'Netherlands': 0.21, 'Sweden': 0.25, 'Norway': 0.25,
    'Denmark': 0.25, 'Finland': 0.24, 'Poland': 0.23, 'Belgium': 0.21,
    'Austria': 0.20, 'Ireland': 0.23, 'Portugal': 0.23, 'Switzerland': 0.081,
    'Czech Republic': 0.21, 'Hungary': 0.27, 'Slovakia': 0.20, 'Romania': 0.19,
    'Turkey': 0.20, 'Russia': 0.20, 'Ukraine': 0.20,
    
    # APAC
    'Japan': 0.10, 'South Korea': 0.10, 'Australia': 0.10, 'New Zealand': 0.15,
    'India': 0.18, 'China': 0.06, 'Taiwan': 0.05, 'Singapore': 0.09,
    'Indonesia': 0.11, 'Malaysia': 0.08, 'Thailand': 0.07, 'Vietnam': 0.10,
    
    # LATAM & MEA
    'Brazil': 0.0, # Often complex/withholding, treating as 0 deduction for MVP safety
    'Mexico': 0.16, 'Argentina': 0.21, 'Chile': 0.19, 'Colombia': 0.19,
    'Peru': 0.18, 'South Africa': 0.15, 'Saudi Arabia': 0.15, 'United Arab Emirates': 0.05,
    'Israel': 0.17
}

REGION_MAP = {
    'NA': ['United States', 'Canada'],
    'EMEA': ['United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
             'Sweden', 'Norway', 'Denmark', 'Finland', 'Poland', 'Belgium', 'Austria', 
             'Ireland', 'Portugal', 'Switzerland', 'Czech Republic', 'Hungary', 'Slovakia', 
             'Romania', 'Turkey', 'Russia', 'Ukraine', 'South Africa', 'Saudi Arabia', 
             'United Arab Emirates', 'Israel', 'Qatar', 'Kuwait'],
    'LATAM': ['Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Uruguay', 'Costa Rica'],
    'APAC': ['Japan', 'South Korea', 'Australia', 'New Zealand', 'China', 'India', 
             'Taiwan', 'Singapore', 'Indonesia', 'Malaysia', 'Thailand', 'Vietnam', 
             'Philippines', 'Hong Kong']
}

CURRENCY_SYMBOLS = {
    'USD': '$', 'CAD': 'CA$', 'EUR': '€', 'GBP': '£', 'AUD': 'A$', 'NZD': 'NZ$',
    'JPY': '¥', 'KRW': '₩', 'CNY': '¥', 'HKD': 'HK$', 'TWD': 'NT$', 'SGD': 'S$',
    'THB': '฿', 'MYR': 'RM', 'IDR': 'Rp', 'PHP': '₱', 'VND': '₫', 'INR': '₹',
    'AED': 'AED', 'SAR': 'SAR', 'ZAR': 'R', 'BRL': 'R$', 'ARS': 'ARS$',
    'CLP': 'CLP$', 'COP': 'COL$', 'MXN': 'Mex$', 'PEN': 'S/', 'UYU': '$U',
    'RUB': '₽', 'TRY': '₺', 'UAH': '₴', 'PLN': 'zł', 'CHF': 'CHF', 'SEK': 'kr',
    'NOK': 'kr', 'DKK': 'kr.', 'CZK': 'Kč', 'HUF': 'Ft', 'ILS': '₪', 'QAR': 'QR'
}

STEAM_CURRENCY_OVERRIDES_BY_CODE = {'CZ': 'EUR', 'DK': 'EUR', 'HU': 'EUR', 'SE': 'EUR'}
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
# LOGIC & OVERRIDES
# ==========================================

def fetch_steam_price_with_override(appid, country, title):
    result = v25.fetch_steam_price(appid, country, title)
    if result and country in STEAM_CURRENCY_OVERRIDES_BY_CODE:
        result.currency = STEAM_CURRENCY_OVERRIDES_BY_CODE[country]
    return result

v25.fetch_steam_price = fetch_steam_price_with_override

def get_region(country):
    for region, countries in REGION_MAP.items():
        if country in countries:
            return region
    return 'Other'

def calculate_net_revenue(price_usd, country, platform_fee=0.30):
    """
    Calculates Estimated Net Revenue (USD) per unit.
    Logic: (Price / (1 + VAT)) * (1 - PlatformFee)
    """
    if not price_usd or price_usd <= 0:
        return 0.0
    
    # 1. Deduct VAT if applicable
    vat_rate = VAT_RATES.get(country, 0.0)
    pre_tax_usd = price_usd / (1 + vat_rate)
    
    # 2. Deduct Platform Fee (Standard 30%)
    net_revenue = pre_tax_usd * (1 - platform_fee)
    
    return net_revenue

def get_price_pattern(prices):
    if not prices: return {'ending': '.99', 'has_decimals': True}
    endings, has_decimals = [], False
    for price in prices:
        s = str(price)
        if '.' in s:
            has_decimals = True
            endings.append(f".{s.split('.')[-1]}")
        else:
            endings.append("whole")
    return {'ending': Counter(endings).most_common(1)[0][0] if endings else ".99", 'has_decimals': has_decimals}

def apply_vanity_pricing(currency, raw_price, competitive_prices):
    if raw_price <= 0: return raw_price
    pattern = get_price_pattern(competitive_prices)
    
    if currency == 'EUR': return 69.99 if raw_price < 74.99 else 79.99
    elif currency == 'BRL':
        rounded = round(raw_price / 10) * 10
        return rounded - 0.10 if '.9' in pattern['ending'] else float(rounded)
    elif currency == 'CAD': return int(raw_price / 10) * 10 + 9.99
    elif currency in ['JPY', 'KRW', 'IDR', 'CLP', 'COP']: return round(raw_price, -3) if raw_price > 1000 else round(raw_price / 100) * 100
    elif currency in ['AUD', 'NZD']: return int(raw_price / 10) * 10 + 9.95
    elif currency in ['GBP']: return int(raw_price) + 0.99
    elif currency == 'CHF': return int(raw_price / 10) * 10 + 9.90
    else: return round(raw_price, 2) if pattern['has_decimals'] else round(raw_price)

def normalize_weights(weights):
    total = sum(weights)
    return [0.0] * len(weights) if total == 0 else [w / total for w in weights]

def calculate_weighted_recommendations(pricing_data, tier_config, usd_rates, competitive_data):
    recommendations = []
    
    for platform in ['Steam', 'Xbox', 'PlayStation']:
        if platform not in pricing_data or not pricing_data[platform]: continue
        
        games_config = tier_config[platform]
        weights = normalize_weights([game[3] for game in games_config])
        
        countries = set()
        for game_prices in pricing_data[platform].values(): countries.update(game_prices.keys())
        
        for country in countries:
            weighted_sum = total_weight = count = 0
            local_currency = None
            
            for idx, (game_name, _, scale, _) in enumerate(games_config):
                if game_name in pricing_data[platform] and country in pricing_data[platform][game_name]:
                    data = pricing_data[platform][game_name][country]
                    usd_price = data.get('price_usd')
                    
                    if usd_price and usd_price > 0:
                        weighted_sum += usd_price * scale * weights[idx]
                        total_weight += weights[idx]
                        count += 1
                        if not local_currency: local_currency = data.get('currency')
            
            if count > 0 and total_weight > 0 and local_currency:
                rec_usd = weighted_sum / total_weight
                if platform == 'Xbox' and country in XBOX_CURRENCY_OVERRIDES:
                    local_currency = XBOX_CURRENCY_OVERRIDES[country]
                
                rate = usd_rates.get(local_currency, 1.0)
                rec_local = rec_usd * rate
                
                comp_prices = []
                if platform in competitive_data and not competitive_data[platform].empty:
                    comp_prices = competitive_data[platform][competitive_data[platform]['Country'] == country]['Local Price'].dropna().tolist()
                
                vanity_local = apply_vanity_pricing(local_currency, rec_local, comp_prices)
                
                # NET REVENUE CALC
                net_rev = calculate_net_revenue(rec_usd, country)
                
                recommendations.append({
                    'platform': platform, 'country': country, 'region': get_region(country),
                    'currency': local_currency, 'rec_usd': rec_usd, 'rec_local': vanity_local,
                    'net_revenue_usd': net_rev,
                    'formatted': f"{CURRENCY_SYMBOLS.get(local_currency, '')}{vanity_local:,.2f}" if local_currency not in WHOLE_NUMBER_CURRENCIES else f"{CURRENCY_SYMBOLS.get(local_currency, '')}{vanity_local:,.0f}",
                    'games_count': count
                })
    return recommendations

# ==========================================
# UI & APP
# ==========================================

st.title("🚀 Game Revenue & Strategy AI — v5.0")
st.markdown("""
**Executive Dashboard for Strategic Pricing**
* **Net Revenue Engine:** Automatically calculates post-tax, post-platform revenue.
* **Regional Intelligence:** Groups data by NA, EMEA, LATAM, APAC.
* **Profitability Waterfall:** Visualize which countries drive actual bottom-line value.
""")

st.divider()

if 'tier_configs' not in st.session_state: st.session_state.tier_configs = DEFAULT_TIER_CONFIGS.copy()
if 'pricing_data' not in st.session_state: st.session_state.pricing_data = {}
if 'processed_df' not in st.session_state: st.session_state.processed_df = None

tier = st.selectbox("Select Strategy Tier", ["AAA"], key="tier_selector")

with st.expander("🛠️ Configure Competitive Set", expanded=False):
    for platform in ['Steam', 'Xbox', 'PlayStation']:
        st.caption(f"**{platform} Games**")
        games = list(st.session_state.tier_configs[tier][platform])
        for i, (name, url, scale, weight) in enumerate(games):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1: new_name = st.text_input(f"Title", value=name, key=f"{platform}_{i}_n", label_visibility="collapsed")
            with c2: new_scale = st.number_input(f"Scale", value=scale, key=f"{platform}_{i}_s", label_visibility="collapsed")
            with c3: new_weight = st.number_input(f"Weight", value=weight, key=f"{platform}_{i}_w", label_visibility="collapsed")
            games[i] = (new_name, url, new_scale, new_weight)
        st.session_state.tier_configs[tier][platform] = games

if st.button("🚀 Run Strategy Analysis (Pull Data)", type="primary", use_container_width=True):
    with st.spinner("Scraping global markets & calculating tax implications..."):
        # (Scraping logic same as v4.5)
        steam_games = [(v25.extract_steam_appid(url), name) for name, url, _, _ in st.session_state.tier_configs[tier]['Steam'] if v25.extract_steam_appid(url)]
        xbox_games = [(v25.extract_xbox_store_id(url), name) for name, url, _, _ in st.session_state.tier_configs[tier]['Xbox'] if v25.extract_xbox_store_id(url)]
        ps_games = [(v25.extract_ps_product_id(url), name) for name, url, _, _ in st.session_state.tier_configs[tier]['PlayStation'] if v25.extract_ps_product_id(url)]
        
        s_res, x_res, p_res = v25.pull_all_prices(steam_games, xbox_games, ps_games, max_workers=20)
        rates = v25.fetch_exchange_rates()
        
        # Process & Enrich
        def process_and_enrich(results, platform):
            df = v25.process_results(list(results), rates)
            if not df.empty:
                df['Platform'] = platform
                df['Region'] = df['Country'].apply(get_region)
                # Quick Fixes
                if platform == 'Xbox': 
                    df.loc[df['Country'] == 'United Arab Emirates', 'Currency'] = 'USD'
            return df

        dfs = [process_and_enrich(s_res, 'Steam'), process_and_enrich(x_res, 'Xbox'), process_and_enrich(p_res, 'PlayStation')]
        st.session_state.processed_df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
        
        # Store for Recommendations
        st.session_state.pricing_data[tier] = {'Steam': {}, 'Xbox': {}, 'PlayStation': {}}
        for r in s_res: st.session_state.pricing_data[tier]['Steam'].setdefault(r.title, {})[r.country] = {'price_usd': r.price_usd, 'currency': r.currency}
        for r in x_res: st.session_state.pricing_data[tier]['Xbox'].setdefault(r.title, {})[r.country] = {'price_usd': r.price_usd, 'currency': 'USD' if r.country == 'United Arab Emirates' else r.currency}
        for r in p_res: 
            price_usd = r.price_usd
            if r.currency in rates and rates[r.currency] > 0 and r.currency in WHOLE_NUMBER_CURRENCIES: price_usd = r.price / rates[r.currency] # Recalc fix
            st.session_state.pricing_data[tier]['PlayStation'].setdefault(r.title, {})[r.country] = {'price_usd': price_usd, 'currency': r.currency}
            
    st.success("✅ Intelligence Gathered!")

# ==========================================
# ANALYSIS TABS
# ==========================================

if st.session_state.processed_df is not None:
    tabs = st.tabs(["💡 Strategic Recommendations", "💰 Profitability & Net Revenue", "📊 Market Data"])
    
    # 1. RECOMMENDATIONS
    with tabs[0]:
        st.subheader("Recommended Launch Pricing")
        rates = v25.fetch_exchange_rates()
        comp_data = {'Steam': st.session_state.processed_df[st.session_state.processed_df['Platform']=='Steam'], 
                     'Xbox': st.session_state.processed_df[st.session_state.processed_df['Platform']=='Xbox'],
                     'PlayStation': st.session_state.processed_df[st.session_state.processed_df['Platform']=='PlayStation']}
        
        recs = calculate_weighted_recommendations(st.session_state.pricing_data[tier], st.session_state.tier_configs[tier], rates, comp_data)
        rec_df = pd.DataFrame(recs)
        
        if not rec_df.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                f_plat = st.multiselect("Filter Platform", ['Steam', 'Xbox', 'PlayStation'], default=['Steam'])
                f_reg = st.multiselect("Filter Region", ['NA', 'EMEA', 'LATAM', 'APAC'], default=['NA', 'EMEA'])
            
            with col2:
                show_df = rec_df[rec_df['platform'].isin(f_plat) & rec_df['region'].isin(f_reg)].copy()
                show_df = show_df[['platform', 'region', 'country', 'formatted', 'rec_usd', 'net_revenue_usd']]
                show_df.columns = ['Platform', 'Region', 'Country', 'Local Price', 'Gross USD', 'Net USD (Est)']
                st.dataframe(show_df, use_container_width=True, hide_index=True)

    # 2. PROFITABILITY INSIGHTS
    with tabs[1]:
        st.subheader("💰 The Profitability Waterfall")
        st.caption("Visualizing Estimated Net Revenue per Unit (After VAT & 30% Platform Fee)")
        
        if not rec_df.empty:
            # Waterfall Chart
            chart_df = rec_df.sort_values('net_revenue_usd', ascending=False)
            
            c = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('country', sort='-y', title='Market'),
                y=alt.Y('net_revenue_usd', title='Net Revenue per Unit ($)'),
                color=alt.Color('region', title='Region'),
                tooltip=['country', 'formatted', 'net_revenue_usd']
            ).properties(height=400)
            st.altair_chart(c, use_container_width=True)
            
            st.markdown("### 🚨 Margin Alerts")
            col1, col2 = st.columns(2)
            with col1:
                st.warning("🔻 **Low Margin Markets (High Volume Risk)**")
                low_margin = chart_df[chart_df['net_revenue_usd'] < chart_df['net_revenue_usd'].median() * 0.7]
                st.dataframe(low_margin[['country', 'region', 'formatted', 'net_revenue_usd']].head(5), hide_index=True)
            
            with col2:
                st.success("💎 **Premium Yield Markets**")
                high_yield = chart_df[chart_df['net_revenue_usd'] > chart_df['net_revenue_usd'].median() * 1.3]
                st.dataframe(high_yield[['country', 'region', 'formatted', 'net_revenue_usd']].head(5), hide_index=True)

    # 3. RAW DATA
    with tabs[2]:
        st.dataframe(st.session_state.processed_df, use_container_width=True)
