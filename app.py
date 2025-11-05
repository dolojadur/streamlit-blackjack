# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import random

# --------------------------- CONFIG ---------------------------
st.set_page_config(page_title="Blackjack ML", page_icon="🃏", layout="wide")

# Fuerza tema claro y elimina sidebar/toolbar
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

:root{
  --bg:#f7f9fc;        /* fondo claro */
  --felt:#0b8f3a;      /* paño */
  --felt2:#087532;
  --ink:#0f172a;       /* texto */
}

html, body, [class*="css"] { font-family:'Poppins', sans-serif !important; color:var(--ink); }
.stApp { background: var(--bg); }
section.main { background: var(--bg); padding-top: 10px; }

[data-testid="stSidebar"], header, [data-testid="stToolbar"] { display:none !important; }

.container { max-width: 980px; margin: 0 auto 40px; padding: 0 14px; }

/* Título simple sin tarjetas blancas */
.title {
  font-size: 28px; font-weight: 700; margin: 10px 0 4px 0;
}
.subtitle { color:#6b7280; margin-bottom: 18px; }

/* Mesa verde centrada */
.table-wrap{
  background: radial-gradient(ellipse at center, var(--felt) 0%, var(--felt2) 100%);
  border-radius: 22px;
  padding: 28px 26px;
  box-shadow: inset 0 10px 30px rgba(0,0,0,.28);
}

/* Filas y cartas */
.row-label{ color:#e5e7eb; font-weight:600; letter-spacing:.4px; margin-bottom:8px; }
.cards-row{ display:flex; gap:12px; align-items:center; min-height:92px; flex-wrap:wrap; }

.card-face{
  width:68px; height:96px; border-radius:12px; background:#fff;
  box-shadow: 0 6px 14px rgba(0,0,0,.25);
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:22px; position:relative;
}
.card-face.red{ color:#d11a2a; }
.card-face.black{ color:#111827; }

.card-back{
  width:68px; height:96px; border-radius:12px;
  background: repeating-linear-gradient(45deg, #0ea5e9, #0ea5e9 10px, #0284c7 10px, #0284c7 20px);
  border: 2px solid #e2e8f0;
  box-shadow: 0 6px 14px rgba(0,0,0,.25);
}

.totals{ color:#e5e7eb; font-weight:600; margin-top:6px; }

/* Controles en una sola línea */
.controls { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
.controls button[kind="secondary"] { background:#fff !important; }

/* Botones bonitos */
.btn {
  border: 1px solid #e5e7eb; background:#fff; border-radius:12px; padding:10px 14px;
  box-shadow: 0 6px 14px rgba(16,24,40,.08);
}
.btn:disabled{ opacity:.6; cursor:not-allowed; }
.pill{ display:inline-block; border-radius:999px; background:#eef2ff; color:#3730a3; padding:4px 10px; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# --------------------------- HELPERS PIPELINE ---------------------------
def _find_column_transformer(pipe: Pipeline) -> ColumnTransformer | None:
    if isinstance(pipe, ColumnTransformer): return pipe
    if hasattr(pipe, "named_steps"):
        for _, step in pipe.named_steps.items():
            if isinstance(step, ColumnTransformer): return step
            if isinstance(step, Pipeline):
                inner = _find_column_transformer(step)
                if inner is not None: return inner
    return None

def expected_columns_from_ct(ct: ColumnTransformer) -> list[str]:
    cols = []
    for _, _, cols_spec in getattr(ct, "transformers_", []):
        if cols_spec in ("drop", None): continue
        if isinstance(cols_spec, (list, tuple)):
            cols.extend([c for c in cols_spec if isinstance(c, str)])
    # únicos preservando orden
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def ensure_expected_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            if col in ("player_cards","dealer_cards"): df[col] = ""
            elif col in ("step","round_id","hand_number"): df[col] = 1
            elif col == "game_id": df[col] = 0
            elif col in ("bet_mode","strategy_used"): df[col] = ("flat" if col=="bet_mode" else "unknown")
            else: df[col] = np.nan
    return df

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None): self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.columns_to_drop, errors="ignore")

class BlackjackFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_ = X.copy()
        def hand_value(cards):
            card_list = [c.strip().upper() for c in str(cards).split(",") if c.strip()]
            vals, aces = [], 0
            for c in card_list:
                if c in ["J","Q","K"]: vals.append(10)
                elif c == "A": vals.append(11); aces += 1
                else:
                    try: vals.append(int(c))
                    except: vals.append(0)
            total = sum(vals)
            while total>21 and aces>0: total-=10; aces-=1
            return total
        def dealer_value(cards):
            first = str(cards).split(",")[0].strip().upper()
            if first in ["J","Q","K"]: return 10
            if first=="A": return 11
            try: return int(first)
            except: return 0
        X_["player_total"] = X_["player_cards"].apply(hand_value)
        X_["player_aces"]  = X_["player_cards"].apply(lambda s: str(s).upper().split(",").count("A"))
        X_["dealer_visible"] = X_["dealer_cards"].apply(dealer_value)
        return X_

@st.cache_resource
def load_model():
    m = joblib.load("models/blackjack_action_model.joblib")
    ct = _find_column_transformer(m)
    expected = expected_columns_from_ct(ct) if ct is not None else []
    return m, expected

model, expected_cols = load_model()
ACTIONS = ["hit","stand","double","split"]  # split no implementado

# --------------------------- BLACKJACK CORE ---------------------------
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]

def new_shoe(num_decks=4):
    shoe=[]
    for _ in range(num_decks):
        for r in RANKS:
            for s in SUITS: shoe.append((r,s))
    random.shuffle(shoe)
    return shoe

def add_card_str(s, card):  # rank-only para el modelo
    r = card[0]
    return (s + ", " + r) if s.strip() else r

def hand_total(cards_str):
    cards=[c.strip().upper() for c in cards_str.split(",") if c.strip()]
    vals, aces = [], 0
    for c in cards:
        if c in ["J","Q","K"]: vals.append(10)
        elif c=="A": vals.append(11); aces+=1
        else:
            try: vals.append(int(c))
            except: vals.append(0)
    total=sum(vals)
    while total>21 and aces>0: total-=10; aces-=1
    return total

def recommend_action(player_cards, dealer_cards, step=1, extra_cols=None):
    row = {
        "player_cards": player_cards, "dealer_cards": dealer_cards,
        "step": step, "game_id":1, "round_id":1, "hand_number":1,
        "bet_mode":"flat", "strategy_used":"unknown"
    }
    if extra_cols: row.update(extra_cols)
    X = pd.DataFrame([row])
    if expected_cols: X = ensure_expected_columns(X, expected_cols)
    return model.predict(X)[0]

# --------------------------- STATE ---------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("shoe", new_shoe())
    ss.setdefault("player", [])        # lista de (rank,suit)
    ss.setdefault("dealer", [])
    ss.setdefault("player_str", "")
    ss.setdefault("dealer_str", "")
    ss.setdefault("step", 1)
    ss.setdefault("round_over", True)
    ss.setdefault("dealer_revealed", False)
    ss.setdefault("last_rec", "hit")
    ss.setdefault("history", [])       # para deshacer: lista de ("player"|"dealer", carta)
init_state()

# --------------------------- RENDER ---------------------------
def suit_color(s): return "red" if s in ["♥","♦"] else "black"
def render_card(rank, suit, hidden=False):
    if hidden:
        return '<div class="card-back"></div>'
    return f'<div class="card-face {"red" if suit_color(suit)=="red" else "black"}">{rank}{suit}</div>'

def render_hand(hand, hide_second=False):
    html=[]
    for i,(r,s) in enumerate(hand):
        html.append(render_card(r,s, hidden=(hide_second and i==1)))
    return "".join(html) if html else '<span class="pill">Sin cartas</span>'

# --------------------------- FLOW ---------------------------
def deal_new_hand():
    ss=st.session_state
    ss.shoe = new_shoe()
    ss.player, ss.dealer = [], []
    ss.player_str = ""; ss.dealer_str=""
    ss.step = 1; ss.round_over = False; ss.dealer_revealed=False; ss.history=[]
    for _ in range(2):
        cp = ss.shoe.pop(); ss.player.append(cp); ss.player_str = add_card_str(ss.player_str, cp); ss.history.append(("player", cp))
        cd = ss.shoe.pop(); ss.dealer.append(cd); ss.dealer_str = add_card_str(ss.dealer_str, cd); ss.history.append(("dealer", cd))

def player_hit(double=False):
    ss=st.session_state
    c = ss.shoe.pop()
    ss.player.append(c); ss.player_str = add_card_str(ss.player_str, c); ss.history.append(("player", c))
    ss.step += 1
    if hand_total(ss.player_str) > 21:
        ss.dealer_revealed=True; ss.round_over=True
        st.toast("¡Te pasaste! Pierdes la mano.", icon="❌")

def undo_last_hit():
    ss=st.session_state
    # quita la última carta del jugador si la ronda no terminó y hubo hit
    if ss.round_over or not ss.history: return
    # buscar la última carta del jugador en la historia
    while ss.history:
        who, card = ss.history.pop()
        if who == "player" and len(ss.player) > 2:  # no permitimos deshacer debajo de 2 cartas iniciales
            ss.player.pop()
            # reconstruir player_str
            s = ""
            for r,suit in ss.player: s = add_card_str(s, (r,suit))
            ss.player_str = s
            ss.step = max(1, ss.step-1)
            break

def dealer_play_to_17():
    ss=st.session_state
    ss.dealer_revealed=True
    def total_from_list(cards):
        s=""; 
        for c in cards: s = add_card_str(s, c)
        return hand_total(s)
    while total_from_list(ss.dealer) < 17 and ss.shoe:
        c = ss.shoe.pop()
        ss.dealer.append(c); ss.dealer_str = add_card_str(ss.dealer_str, c); ss.history.append(("dealer", c))
    p = hand_total(ss.player_str); d = hand_total(ss.dealer_str)
    if d>21 or p>d: st.toast("¡Ganaste!", icon="✅")
    elif p<d:       st.toast("Perdiste 😢", icon="⚠️")
    else:           st.toast("Empate (push).", icon="🔁")
    ss.round_over=True

# --------------------------- UI ---------------------------
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="title">🃏 Blackjack ML</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Jugar con asistencia del modelo</div>', unsafe_allow_html=True)

# Mesa
st.markdown('<div class="table-wrap">', unsafe_allow_html=True)

# Dealer
dealer_total_display = "?" if (not st.session_state.dealer_revealed and st.session_state.dealer) else (
    hand_total(st.session_state.dealer_str) if st.session_state.dealer else "-"
)
st.markdown('<div class="row-label">DEALER</div>', unsafe_allow_html=True)
st.markdown(f'<div class="cards-row">{render_hand(st.session_state.dealer, hide_second=not st.session_state.dealer_revealed)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="totals">Total: {dealer_total_display}</div>', unsafe_allow_html=True)

# Separador pequeño
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# Player
player_total = hand_total(st.session_state.player_str) if st.session_state.player else "-"
st.markdown('<div class="row-label">JUGADOR</div>', unsafe_allow_html=True)
st.markdown(f'<div class="cards-row">{render_hand(st.session_state.player)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="totals">Total: {player_total}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /table-wrap

# Controles
st.markdown('<div class="controls">', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.button("🂠 Repartir", key="deal", on_click=deal_new_hand, disabled=not st.session_state.round_over)
with c2:
    disabled = st.session_state.round_over or not st.session_state.player
    if st.button("🤖 Recomendar", disabled=disabled):
        rec = recommend_action(st.session_state.player_str, st.session_state.dealer_str, step=st.session_state.step)
        st.session_state.last_rec = rec
        st.toast(f"Modelo: {rec.upper()}", icon="🧠")
with c3:
    st.button("➕ Hit", on_click=player_hit, disabled=disabled)
with c4:
    st.button("✋ Stand", on_click=dealer_play_to_17, disabled=disabled)
with c5:
    st.button("×2 Double", on_click=lambda: (player_hit(True), None) if not st.session_state.round_over else None,
              disabled=(st.session_state.round_over or not st.session_state.player or st.session_state.step!=1))
with c6:
    st.button("↩ Deshacer", on_click=undo_last_hit,
              disabled=(st.session_state.round_over or len(st.session_state.player)<=2))

st.markdown(f'<div class="pill">Paso: {st.session_state.step} • Recomendación: {st.session_state.last_rec.upper()}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # /controls
st.markdown('</div>', unsafe_allow_html=True)  # /container
