# theme.py — MongoDB AI Assistant Design System v2
# Linear / MongoDB Atlas / Vercel Analytics inspired premium dark theme

import streamlit as st

# ═══════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════

# Backgrounds (elevation system)
BG_PAGE     = "#0c0f16"
BG_CARD     = "#131920"
BG_INPUT    = "#1c2333"
BG_ELEVATED = "#242e42"

# Borders
BORDER_DIM   = "rgba(255,255,255,0.06)"
BORDER_STD   = "rgba(255,255,255,0.10)"
BORDER_GREEN = "rgba(0,237,100,0.20)"

# Text
TEXT_100 = "#f1f5f9"
TEXT_200 = "#94a3b8"
TEXT_300 = "#475569"
TEXT_400 = "#1e293b"

# Accent (MongoDB green)
GREEN_100  = "#00ed64"
GREEN_200  = "#00c853"
GREEN_BG   = "rgba(0,237,100,0.08)"
GREEN_RING = "rgba(0,237,100,0.20)"

# Status
STATUS_SUCCESS = "#22c55e"
STATUS_WARNING = "#f59e0b"
STATUS_ERROR   = "#ef4444"
STATUS_INFO    = "#3b82f6"

# Typography
FONT_BASE    = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_DISPLAY = "'Inter Display', Inter, sans-serif"
FONT_MONO    = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace"

# Chart palette
CHART_COLORS = [
    "#00ed64", "#3b82f6", "#f59e0b", "#a78bfa",
    "#f43f5e", "#06b6d4", "#84cc16", "#fb923c",
]

# Legacy aliases for backwards compat
PRIMARY_GREEN  = GREEN_100
BG_BASE        = BG_PAGE
BG_SURFACE     = BG_CARD
TEXT_PRIMARY   = TEXT_100
TEXT_SECONDARY = TEXT_200
TEXT_MUTED     = TEXT_300
BORDER_SUBTLE  = BORDER_DIM
BORDER_ACCENT  = BORDER_GREEN
CHART_PALETTE  = CHART_COLORS


# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS INJECTION
# ═══════════════════════════════════════════════════════════════

def inject_global_css():
    """Inject full design-system stylesheet."""
    css = f"""
    <style>
    /* ── A) Font import ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

    /* ── B) Kill Streamlit chrome ───────────────────────────── */
    #MainMenu, header, footer, .stDeployButton {{display: none !important;}}
    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}
    .stApp {{background: {BG_PAGE} !important;}}
    section[data-testid="stSidebar"] {{
        background: {BG_PAGE} !important;
        border-right: 1px solid {BORDER_DIM} !important;
        width: 240px !important;
        min-width: 240px !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding: 0 !important;
        background: transparent !important;
    }}

    /* ── C) Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar {{width: 4px; height: 4px;}}
    ::-webkit-scrollbar-track {{background: transparent;}}
    ::-webkit-scrollbar-thumb {{background: {BORDER_STD}; border-radius: 99px;}}
    ::-webkit-scrollbar-thumb:hover {{background: {TEXT_300};}}
    * {{scrollbar-width: thin; scrollbar-color: {BORDER_STD} transparent;}}

    /* ── D) Selection ───────────────────────────────────────── */
    ::selection {{background: {GREEN_BG}; color: {GREEN_100};}}

    /* ── E) Base typography ─────────────────────────────────── */
    body, .stApp {{
        font-family: {FONT_BASE} !important;
        color: {TEXT_200} !important;
        -webkit-font-smoothing: antialiased;
        font-size: 14px;
        line-height: 1.5;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: {FONT_DISPLAY} !important;
        color: {TEXT_100} !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    p, li, span, div {{font-family: {FONT_BASE};}}
    hr {{border-color: {BORDER_DIM} !important; margin: 0 !important;}}

    /* ── F) Sidebar navigation ──────────────────────────────── */
    .sidebar-logo {{
        display: flex; align-items: center; gap: 10px;
        padding: 20px 16px 16px;
        border-bottom: 1px solid {BORDER_DIM};
        margin-bottom: 8px;
    }}
    .sidebar-logo-icon {{font-size: 24px;}}
    .sidebar-logo-name {{
        font-size: 14px; font-weight: 700; color: {TEXT_100};
        letter-spacing: -0.01em; line-height: 1;
    }}
    .sidebar-logo-sub {{
        font-size: 11px; color: {GREEN_100}; font-weight: 500;
        line-height: 1; margin-top: 2px;
    }}
    .nav-section-label {{
        padding: 12px 16px 4px;
        font-size: 10px; font-weight: 600; color: {TEXT_300};
        text-transform: uppercase; letter-spacing: 0.08em;
    }}

    /* Sidebar nav buttons — override Streamlit buttons inside sidebar */
    section[data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: {TEXT_200} !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        border-radius: 0 6px 6px 0 !important;
        padding: 8px 12px !important;
        margin: 1px 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        font-family: {FONT_BASE} !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: all 0.12s ease !important;
        width: calc(100% - 16px) !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: {BG_INPUT} !important;
        color: {TEXT_100} !important;
    }}
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {GREEN_BG} !important;
        color: {GREEN_100} !important;
        border-left: 3px solid {GREEN_100} !important;
    }}

    /* ── G) Top bar ─────────────────────────────────────────── */
    .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        height: 48px; padding: 0 24px;
        border-bottom: 1px solid {BORDER_DIM};
        background: {BG_PAGE};
        position: sticky; top: 0; z-index: 10;
    }}
    .topbar-title {{
        font-size: 14px; font-weight: 600; color: {TEXT_100};
        letter-spacing: -0.01em;
    }}
    .topbar-right {{display: flex; align-items: center; gap: 12px;}}
    .db-status-badge {{
        display: flex; align-items: center; gap: 6px;
        padding: 4px 10px;
        background: {BG_CARD}; border: 1px solid {BORDER_DIM};
        border-radius: 99px; font-size: 11px; font-weight: 500; color: {TEXT_200};
    }}
    .db-status-dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: {STATUS_SUCCESS};
        box-shadow: 0 0 6px rgba(34,197,94,0.6);
        animation: pulse 2s ease-in-out infinite;
    }}
    .db-status-dot.error {{
        background: {STATUS_ERROR};
        box-shadow: 0 0 6px rgba(239,68,68,0.6);
    }}
    @keyframes pulse {{
        0%, 100% {{opacity: 1;}}
        50% {{opacity: 0.4;}}
    }}

    /* ── H) Page content wrapper ────────────────────────────── */
    .page-content {{padding: 24px; min-height: calc(100vh - 48px);}}
    .page-header {{
        margin-bottom: 24px; padding-bottom: 20px;
        border-bottom: 1px solid {BORDER_DIM};
    }}
    .page-header-title {{
        font-size: 20px; font-weight: 700; color: {TEXT_100};
        letter-spacing: -0.02em;
        display: flex; align-items: center; gap: 10px;
    }}
    .page-header-subtitle {{
        font-size: 13px; color: {TEXT_300};
        margin-top: 4px; font-weight: 400;
    }}

    /* ── I) Card components ─────────────────────────────────── */
    .card {{
        background: {BG_CARD}; border: 1px solid {BORDER_DIM};
        border-radius: 10px; padding: 16px 20px;
        transition: border-color 0.15s;
    }}
    .card:hover {{border-color: {BORDER_STD};}}
    .card-glass {{
        background: rgba(19,25,32,0.7);
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 20px;
    }}
    .insight-card {{
        background: {BG_CARD}; border: 1px solid {BORDER_DIM};
        border-left: 3px solid {GREEN_100};
        border-radius: 0 10px 10px 0;
        padding: 16px 20px; margin-bottom: 16px;
        transition: border-color 0.15s;
    }}
    .insight-card:hover {{
        border-top-color: {BORDER_GREEN};
        border-right-color: {BORDER_GREEN};
        border-bottom-color: {BORDER_GREEN};
    }}

    /* ── J) KPI cards ───────────────────────────────────────── */
    .kpi-card {{
        background: {BG_CARD}; border: 1px solid {BORDER_DIM};
        border-radius: 10px; padding: 16px;
    }}
    .kpi-label {{
        font-size: 11px; font-weight: 600; color: {TEXT_300};
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 26px; font-weight: 700; color: {TEXT_100};
        letter-spacing: -0.03em; line-height: 1;
        font-family: {FONT_DISPLAY};
    }}
    .kpi-delta {{
        margin-top: 6px; font-size: 12px; font-weight: 500;
        display: flex; align-items: center; gap: 3px;
    }}
    .kpi-delta.up   {{color: {STATUS_SUCCESS};}}
    .kpi-delta.down {{color: {STATUS_ERROR};}}
    .kpi-delta.flat {{color: {TEXT_300};}}
    .kpi-context {{margin-top: 2px; font-size: 11px; color: {TEXT_300};}}

    /* ── K) Chat interface ──────────────────────────────────── */
    .chat-messages {{
        display: flex; flex-direction: column; gap: 12px;
        padding: 16px 0; max-width: 800px; margin: 0 auto;
    }}
    .chat-row {{display: flex; align-items: flex-end; gap: 8px;}}
    .chat-row.user  {{flex-direction: row-reverse;}}
    .chat-row.assistant {{flex-direction: row;}}
    .chat-avatar {{
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; flex-shrink: 0;
    }}
    .chat-avatar.user      {{background: {GREEN_100}; color: {TEXT_400};}}
    .chat-avatar.assistant {{background: {BG_INPUT}; color: {TEXT_100};}}
    .chat-bubble {{
        max-width: 72%; padding: 10px 14px;
        font-size: 13.5px; line-height: 1.6;
        word-wrap: break-word;
    }}
    .chat-bubble.user {{
        background: {GREEN_100}; color: {TEXT_400};
        border-radius: 12px 12px 2px 12px; font-weight: 500;
    }}
    .chat-bubble.assistant {{
        background: {BG_CARD}; color: {TEXT_100};
        border: 1px solid {BORDER_DIM};
        border-radius: 12px 12px 12px 2px;
    }}
    .chat-timestamp {{
        font-size: 10px; color: {TEXT_300}; margin-top: 4px; padding: 0 4px;
    }}

    /* ── L) Takeaway / callout ──────────────────────────────── */
    .takeaway {{
        background: {GREEN_BG}; border-left: 3px solid {GREEN_100};
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        font-size: 13px; color: {TEXT_200}; line-height: 1.6; margin-top: 12px;
    }}
    .takeaway strong {{color: {TEXT_100}; font-weight: 600;}}

    /* ── M) Streamlit widget overrides ──────────────────────── */
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background: {BG_INPUT} !important;
        border: 1px solid {BORDER_STD} !important;
        border-radius: 8px !important;
        color: {TEXT_100} !important;
        font-family: {FONT_BASE} !important;
        font-size: 13px !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {GREEN_100} !important;
        box-shadow: 0 0 0 3px {GREEN_RING} !important;
        outline: none !important;
    }}
    .stTextArea > div > div > textarea {{
        font-family: {FONT_MONO} !important;
        font-size: 12px !important;
    }}
    /* Primary buttons (outside sidebar) */
    div.stButton > button[kind="primary"],
    .stFormSubmitButton > button {{
        background: {GREEN_100} !important;
        color: {TEXT_400} !important;
        font-weight: 600 !important; font-size: 13px !important;
        border: none !important; border-radius: 8px !important;
        padding: 8px 20px !important;
        transition: background 0.15s, box-shadow 0.15s !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background: {GREEN_200} !important;
        box-shadow: 0 4px 20px rgba(0,237,100,0.3) !important;
    }}
    /* Secondary buttons (outside sidebar) */
    div.stButton > button[kind="secondary"] {{
        background: {BG_INPUT} !important;
        color: {TEXT_200} !important;
        border: 1px solid {BORDER_STD} !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        transition: all 0.15s !important;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        background: {BG_ELEVATED} !important;
        color: {TEXT_100} !important;
    }}
    /* Selectbox */
    .stSelectbox > div > div {{
        background: {BG_INPUT} !important;
        border: 1px solid {BORDER_STD} !important;
        border-radius: 8px !important;
        color: {TEXT_100} !important;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 1px solid {BORDER_DIM} !important;
        gap: 0 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {TEXT_300} !important;
        font-size: 13px !important; font-weight: 500 !important;
        padding: 8px 16px !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{color: {TEXT_200} !important;}}
    .stTabs [aria-selected="true"] {{
        color: {GREEN_100} !important;
        border-bottom-color: {GREEN_100} !important;
    }}
    /* Expander */
    .stExpander, div[data-testid="stExpander"] details {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER_DIM} !important;
        border-radius: 8px !important;
    }}
    .stExpander > div > div > div {{color: {TEXT_200} !important; font-size: 13px !important;}}
    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER_DIM} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    /* Metric overrides */
    [data-testid="stMetricValue"] {{
        font-size: 24px !important; font-weight: 700 !important;
        color: {TEXT_100} !important;
        font-family: {FONT_DISPLAY} !important;
        letter-spacing: -0.02em !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 11px !important; font-weight: 600 !important;
        color: {TEXT_300} !important;
        text-transform: uppercase !important; letter-spacing: 0.06em !important;
    }}
    /* Chat input */
    [data-testid="stChatInput"] textarea {{
        background: {BG_INPUT} !important;
        border: 1px solid {BORDER_STD} !important;
        border-radius: 10px !important;
        color: {TEXT_100} !important; font-size: 13px !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {GREEN_100} !important;
        box-shadow: 0 0 0 3px {GREEN_RING} !important;
    }}
    [data-testid="stChatInput"] {{
        background: transparent !important;
        border: none !important;
    }}
    /* Slider */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: {GREEN_100} !important; border-color: {GREEN_100} !important;
    }}
    /* Info/success/warning/error boxes */
    div[data-testid="stAlert"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER_DIM} !important;
        border-radius: 8px !important;
        color: {TEXT_200} !important;
    }}
    /* Markdown text colors */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {{
        color: {TEXT_200} !important;
        font-size: 13px !important;
    }}
    .stMarkdown p {{color: {TEXT_200};}}
    .stMarkdown a {{color: {GREEN_100};}}

    /* ── CHAT INPUT BORDER FIX ─────────────────────────────── */
    /* Kill ALL default Streamlit border colors on chat input */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] > div > div > div,
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInputContainer"] > div,
    [data-testid="stChatInputContainer"] textarea {{
        border-color: {BORDER_STD} !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    /* Override red invalid/error state Streamlit injects */
    [data-testid="stChatInput"] textarea:invalid,
    [data-testid="stChatInput"] textarea:user-invalid,
    [data-testid="stChatInputContainer"] textarea:invalid,
    [data-testid="stChatInputContainer"]:invalid,
    [data-testid="stChatInput"] > div[data-invalid],
    [data-testid="stChatInput"] > div[aria-invalid],
    [data-testid="stChatInput"] [aria-invalid="true"] {{
        border-color: {BORDER_STD} !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    /* Clean focus — ONLY green, no red bleed */
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInputContainer"] textarea:focus,
    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInputContainer"]:focus-within {{
        border-color: {GREEN_100} !important;
        box-shadow: 0 0 0 2px rgba(0,237,100,0.18) !important;
        outline: none !important;
    }}
    /* Inner container focus */
    [data-testid="stChatInput"] > div:focus-within,
    [data-testid="stChatInputContainer"] > div:focus-within {{
        border-color: {GREEN_100} !important;
        box-shadow: 0 0 0 2px rgba(0,237,100,0.18) !important;
    }}
    /* Nuke red pseudo-element borders */
    [data-testid="stChatInput"]::before,
    [data-testid="stChatInput"]::after,
    [data-testid="stChatInputContainer"]::before,
    [data-testid="stChatInputContainer"]::after {{
        display: none !important;
        border: none !important;
        box-shadow: none !important;
    }}
    /* BaseWeb/Emotion styled component overrides */
    [data-baseweb="textarea"],
    [data-baseweb="base-input"],
    [data-baseweb="input"] {{
        border-color: {BORDER_STD} !important;
        box-shadow: none !important;
        background-color: {BG_INPUT} !important;
    }}
    [data-baseweb="textarea"]:focus-within,
    [data-baseweb="base-input"]:focus-within,
    [data-baseweb="input"]:focus-within {{
        border-color: {GREEN_100} !important;
        box-shadow: 0 0 0 2px rgba(0,237,100,0.18) !important;
    }}

    /* ── CHAT INPUT FINAL OVERRIDE ─────────────────────────── */
    /* DEFAULT state — subtle dark border, no glow */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputContainer"] > div,
    [data-baseweb="base-input"],
    [data-baseweb="textarea"] {{
        border: 1px solid {BORDER_STD} !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    /* FOCUSED state — green border + soft glow only when typing */
    [data-testid="stChatInput"]:focus-within > div,
    [data-testid="stChatInputContainer"]:focus-within > div,
    [data-baseweb="base-input"]:focus-within,
    [data-baseweb="textarea"]:focus-within {{
        border: 1px solid {GREEN_100} !important;
        box-shadow: 0 0 0 3px rgba(0,237,100,0.12) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# REUSABLE UI COMPONENTS
# ═══════════════════════════════════════════════════════════════

def render_topbar(icon: str, title: str, subtitle: str,
                  db_connected: bool = True) -> str:
    """Sticky top bar with page title and DB status."""
    dot_cls = "" if db_connected else " error"
    return f"""
    <div class="topbar">
      <div class="topbar-title">{icon}&nbsp; {title}</div>
      <div class="topbar-right">
        <div style="font-size:11px;color:{TEXT_300};">{subtitle}</div>
        <div class="db-status-badge">
          <div class="db-status-dot{dot_cls}"></div>
          ai_test_db
        </div>
      </div>
    </div>"""


def render_page_header(title: str, subtitle: str, icon: str) -> str:
    """Page header block below top bar."""
    return f"""
    <div class="page-header">
      <div class="page-header-title">{icon} {title}</div>
      <div class="page-header-subtitle">{subtitle}</div>
    </div>"""


def render_kpi_card(value: str, label: str, delta: str = None,
                    delta_type: str = "flat", context: str = "") -> str:
    """HTML for a styled KPI metric card."""
    delta_html = ""
    if delta:
        arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
        delta_html = f'<div class="kpi-delta {delta_type}">{arrow} {delta}</div>'
    ctx_html = f'<div class="kpi-context">{context}</div>' if context else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
      {ctx_html}
    </div>"""


def render_takeaway(text: str) -> str:
    """Styled insight takeaway callout (replaces st.success)."""
    return f'<div class="takeaway">💡 {text}</div>'


def render_divider(label: str = "") -> str:
    """Horizontal rule with optional centered label."""
    if label:
        return f"""
        <div style="display:flex;align-items:center;gap:12px;margin:20px 0 12px;">
            <div style="flex:1;height:1px;background:{BORDER_DIM};"></div>
            <span style="color:{TEXT_300};font-size:10px;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.08em;
                         font-family:{FONT_BASE};">{label}</span>
            <div style="flex:1;height:1px;background:{BORDER_DIM};"></div>
        </div>"""
    return f'<div style="height:1px;background:{BORDER_DIM};margin:16px 0;"></div>'


def render_section_header(title: str, subtitle: str = "") -> str:
    """Section title within a page."""
    sub = f'<div style="font-size:12px;color:{TEXT_300};margin-top:3px;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="margin:20px 0 12px;">
      <div style="font-size:14px;font-weight:600;color:{TEXT_100};letter-spacing:-0.01em;">{title}</div>
      {sub}
    </div>"""


def render_badge(label: str, color_hex: str) -> str:
    """Small colored pill badge."""
    return f"""
    <span style="display:inline-flex;align-items:center;padding:2px 8px;
      background:{color_hex}18;color:{color_hex};
      border:1px solid {color_hex}30;
      border-radius:99px;font-size:10px;font-weight:600;
      letter-spacing:0.04em;text-transform:uppercase;">{label}</span>"""


# Keep old name for backwards compat
render_insight_takeaway = render_takeaway


# ═══════════════════════════════════════════════════════════════
# PLOTLY CHART STYLING
# ═══════════════════════════════════════════════════════════════

def style_plotly_chart(fig, height: int = 420):
    """Apply unified premium dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BASE, color=TEXT_200, size=12),
        title=dict(
            font=dict(family=FONT_DISPLAY, size=14, color=TEXT_100),
            x=0.0, y=1.0, xanchor="left",
            pad=dict(b=12),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor=BORDER_DIM,
            tickcolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=TEXT_300, size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor=BORDER_DIM,
            tickfont=dict(color=TEXT_300, size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER_DIM,
            borderwidth=1,
            font=dict(color=TEXT_200, size=11),
        ),
        margin=dict(l=8, r=8, t=44, b=8),
        height=height,
        colorway=CHART_COLORS,
        hoverlabel=dict(
            bgcolor=BG_ELEVATED,
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(family=FONT_BASE, color=TEXT_100, size=12),
        ),
    )
    return fig
