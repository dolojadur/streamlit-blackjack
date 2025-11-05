# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import random

# --------------------------- CONFIG & THEME ---------------------------
st.set_page_config(page_title="Blackjack ML – Jugar", page_icon="🃏", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

:root {
  --bg: #f6f7fb;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --shadow: 0 10px 25px rgba(16,24,40,0.08);
  --felt: #0b8f3a; /* verde paño */
  --felt-dark: #087532;
  --chip: #0ea5e9;
}

html, body, [class*="css"]  { font-family: 'Poppins', sans-serif !important; color: var(--text); }
section.main { background: var(--bg); }
header, [data-testid="stSidebar"] { display: none !important; } /* sin sidebar ni header app */

.container {
  max-width: 1100px;
  margin: 24px auto 56px;
  padding: 0 16px;
}
.card {
  background: var(--card);
  box-shadow: var(--shadow);
  border-radius: 18px;
  padding: 22px 24px;
}
.title {
  font-weight: 600;
  font-size: 28px;
  margin-bottom: 6px;
}
.subtitle {
  color: var(--muted);
  margin-bottom: 18px;
}

.table-wrap {
  background: radial-gradient(ellipse at center, var(--felt) 0%, var(--felt-dark) 100%);
  border-radius: 22px;
  padding: 26px;
  box-shadow: inset 0 8px 24px rgba(0,0,0,.25);
}

.row-label {
  color: #e2e8f0;
  font-weight: 600;
  margin-bottom: 8px;
  letter-spacing: .4px;
}

.cards-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  min-height: 84px;
}

.card-face {
  width: 64px;
  height: 90px;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 6px 14px rgba(0,0,0,.25);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 22px;
  position: relative;
}
.card-face.red { color: #d11a2a; }
.card-face.black { color: #111827; }

.card-back {
  width: 64px;
  height: 90px;
  border-radius: 10px;
  background: repeating-linear-gradient(
    45deg,
    #0ea5e9, #0ea5e9 8px,
    #0284c7 8px, #0284c7 16px
  );
  box-shadow: 0 6px 14px rgba(0,0,0,.25);
  border: 2px solid #e2e8f0;
}

.totals {
  display: flex;
  gap: 12px;
  margin-top: 8px;
  color: #e2e8f0;
  font-weight: 500;
}

.controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.controls .btn {
  background: var(--card);
  box-shadow: var(--shadow);
  border-radius: 12px;
  padding: 10px 14px;
  border: 1px solid #e5e7eb;
  cursor: pointer;
}
.controls .btn:disabled {
  opacity: .6; cursor: not-allowed;
}

.pill {
  display: inline-block;
  border-radius: 999px;
  background: #e6f4ff;
  border: 1px solid #cfe8ff;
  color: #0b5ed7;
  padding: 4px 10px;
  font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------- PIPELINE UTILS (tus helpers) ---------------------------
def _find_column_transformer(pipe: Pipeline) -> ColumnTransformer | None:
    if isinstance(pipe, ColumnTransformer):
        return pipe
    if hasattr(pipe, "named_steps"):
        for _, step in pipe.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return step
            if isinstance(step, Pipeline):
                inner = _find_column_transformer(step)
                if inner is not None:
                    return inner
    return None

def expected_columns_from_ct(ct: ColumnTransformer) -> list[str]:
    cols = []
    for name, trans, cols_spec in getattr(ct, "transformers_", []):
        if cols_spec == "drop" or cols_spec is None:
            continue
        if isinstance(cols_spec, (list, tuple)):
            cols.extend([c for c in cols_spec if isinstance(c, str)])
    seen = set()
    unique_cols = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    return unique_cols

def ensure_expected_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            if col in ("player_cards", "dealer_cards"):
                df[col] = ""
            elif col in ("step", "round_id", "hand_number"):
                df[col] = 1
            elif col in ("game_id",):
                df[col] = 0
            elif col in ("bet_mode", "strategy_used"):
                df[col] = "flat" if col == "bet_mode" else "unknown"
            else:
                df[col] = np.nan
    return df

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.columns_to_drop, errors="ignore")

class BlackjackFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_ = X.copy()
        def hand_value(cards):
            card_list = [c.strip().upper() for c in str(cards).split(",") if c.strip()]
            values = []
            for c in card_list:
                if c in ["J","Q","K"]: values.append(10)
                elif c == "A": values.append(11)
                else:
                    try: values.append(int(c))
                    except ValueError: values.append(0)
            total = sum(values)
            aces = card_list.count("A")
            while total > 21 and aces > 0:
                total -= 10; aces -= 1
            return total

        X_["player_total"] = X_["player_cards"].apply(hand_value)
        X_["player_aces"]  = X_["player_cards"].apply(lambda s: str(s).upper().split(",").count("A"))

        def dealer_value(cards):
            first = str(cards).split(",")[0].strip().upper()
            if first in ["J","Q","K"]: return 10
            if first == "A": return 11
            try: return int(first)
            except ValueError: return 0
        X_["dealer_visible"] = X_["dealer_cards"].apply(dealer_value)
        return X_

# --------------------------- MODEL LOAD ---------------------------
@st.cache_resource
def load_model():
    m = joblib.load("models/blackjack_action_model.joblib")
    ct = _find_column_transformer(m)
    expected = expected_columns_from_ct(ct) if ct is not None else []
    return m, expected

model, expected_cols = load_model()
ACTIONS = ["hit","stand","double","split"]  # split no implementado a nivel UI lógicas múltiples

# --------------------------- BLACKJACK ENGINE ---------------------------
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]

def new_shoe(num_decks=4):
    shoe = []
    for _ in range(num_decks):
        for r in RANKS:
            for s in SUITS:
                shoe.append((r, s))
    random.shuffle(shoe)
    return shoe

def card_to_rank(card_tuple):  # para el modelo: solo rank
    return card_tuple[0]

def add_card_str(list_str, card_tuple):
    r = card_to_rank(card_tuple)
    return (list_str + ", " + r) if list_str.strip() else r

def hand_value_from_str(cards_str):
    cards = [c.strip().upper() for c in cards_str.split(",") if c.strip()]
    vals, aces = [], 0
    for c in cards:
        if c in ["J","Q","K"]: vals.append(10)
        elif c == "A": vals.append(11); aces += 1
        else:
            try: vals.append(int(c))
            except: vals.append(0)
    total = sum(vals)
    while total > 21 and aces > 0:
        total -= 10; aces -= 1
    return total

def recommend_action(player_cards, dealer_cards, step=1, extra_cols=None):
    row = {
        "player_cards": player_cards,
        "dealer_cards": dealer_cards,  # el extractor usa sólo la primera carta visible
        "step": step,
        "game_id": 1, "round_id": 1, "hand_number": 1,
        "bet_mode": "flat", "strategy_used": "unknown",
    }
    if extra_cols: row.update(extra_cols)
    X = pd.DataFrame([row])
    try:
        if expected_cols:
            X = ensure_expected_columns(X, expected_cols)
        pred = model.predict(X)[0]
        return pred
    except ValueError as e:
        st.error("Faltan columnas para el pipeline. Ajustá ensure_expected_columns().")
        st.code(str(e))
        raise

# --------------------------- SESSION STATE ---------------------------
def init_state():
    if "shoe" not in st.session_state:
        st.session_state.shoe = new_shoe()
    if "player_hand" not in st.session_state:
        st.session_state.player_hand = []
    if "dealer_hand" not in st.session_state:
        st.session_state.dealer_hand = []
    if "player_str" not in st.session_state:
        st.session_state.player_str = ""
    if "dealer_str" not in st.session_state:
        st.session_state.dealer_str = ""
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "round_over" not in st.session_state:
        st.session_state.round_over = True
    if "dealer_revealed" not in st.session_state:
        st.session_state.dealer_revealed = False
    if "last_rec" not in st.session_state:
        st.session_state.last_rec = "hit"

init_state()

# --------------------------- RENDER HELPERS ---------------------------
def suit_color(s):
    return "red" if s in ["♥","♦"] else "black"

def render_card(rank, suit, hidden=False):
    if hidden:
        return '<div class="card-back"></div>'
    return f'<div class="card-face {suit_color(suit)}">{rank}{suit}</div>'

def render_hand(hand, hide_second=False):
    html = []
    for i, (r, s) in enumerate(hand):
        if hide_second and i == 1:
            html.append(render_card(r, s, hidden=True))
        else:
            html.append(render_card(r, s, hidden=False))
    if not hand:
        html.append('<span class="pill">Sin cartas</span>')
    return "".join(html)

# --------------------------- GAME FLOW ---------------------------
def deal_new_hand():
    st.session_state.shoe = new_shoe()
    st.session_state.player_hand = []
    st.session_state.dealer_hand = []
    st.session_state.player_str = ""
    st.session_state.dealer_str = ""
    st.session_state.step = 1
    st.session_state.round_over = False
    st.session_state.dealer_revealed = False
    # Repartir 2 y 2
    for _ in range(2):
        c_p = st.session_state.shoe.pop()
        c_d = st.session_state.shoe.pop()
        st.session_state.player_hand.append(c_p)
        st.session_state.dealer_hand.append(c_d)
        st.session_state.player_str = add_card_str(st.session_state.player_str, c_p)
        st.session_state.dealer_str = add_card_str(st.session_state.dealer_str, c_d)

def player_hit(double=False):
    c = st.session_state.shoe.pop()
    st.session_state.player_hand.append(c)
    st.session_state.player_str = add_card_str(st.session_state.player_str, c)
    st.session_state.step += 1
    if hand_value_from_str(st.session_state.player_str) > 21:
        st.session_state.dealer_revealed = True
        st.session_state.round_over = True
        st.toast("¡Te pasaste! Pierdes la mano.", icon="❌")

def dealer_play_to_17():
    # Revela y juega
    st.session_state.dealer_revealed = True
    def total_from_hand(hand):
        s = ""
        for c in hand:
            s = add_card_str(s, c)
        return hand_value_from_str(s)
    while total_from_hand(st.session_state.dealer_hand) < 17 and len(st.session_state.shoe) > 0:
        c = st.session_state.shoe.pop()
        st.session_state.dealer_hand.append(c)
        st.session_state.dealer_str = add_card_str(st.session_state.dealer_str, c)

    # Resultado
    p = hand_value_from_str(st.session_state.player_str)
    d = hand_value_from_str(st.session_state.dealer_str)
    if d > 21 or p > d:
        st.toast("¡Ganaste!", icon="✅")
    elif p < d:
        st.toast("Perdiste 😢", icon="⚠️")
    else:
        st.toast("Empate (push).", icon="🔁")
    st.session_state.round_over = True

# --------------------------- UI ---------------------------
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown(f"""
  <div class="card" style="margin-bottom:16px;">
    <div class="title">🃏 Blackjack ML</div>
    <div class="subtitle">Una sola página para jugar con asistencia del modelo</div>
    <span class="pill">Tema claro • Poppins • Mesa verde</span>
  </div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Mesa
st.markdown('<div class="table-wrap">', unsafe_allow_html=True)

# Dealer row
dealer_total_display = (
    ("?" if not st.session_state.dealer_revealed else hand_value_from_str(st.session_state.dealer_str))
    if st.session_state.dealer_hand else "-"
)
st.markdown('<div class="row-label">DEALER</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="cards-row">{render_hand(st.session_state.dealer_hand, hide_second=not st.session_state.dealer_revealed)}</div>',
    unsafe_allow_html=True
)
st.markdown(f'<div class="totals"><div>Total: {dealer_total_display}</div></div>', unsafe_allow_html=True)

# Spacer
st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

# Player row
player_total = hand_value_from_str(st.session_state.player_str) if st.session_state.player_hand else "-"
st.markdown('<div class="row-label">JUGADOR</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="cards-row">{render_hand(st.session_state.player_hand)}</div>',
    unsafe_allow_html=True
)
st.markdown(f'<div class="totals"><div>Total: {player_total}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /table-wrap

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# Controls
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="controls">', unsafe_allow_html=True)
    deal_disabled = not st.session_state.round_over
    st.button("🂠 Repartir", disabled=not st.session_state.round_over, on_click=deal_new_hand)

    rec_disabled = st.session_state.round_over or not st.session_state.player_hand
    if st.button("🤖 Recomendar acción", disabled=rec_disabled):
        rec = recommend_action(
            st.session_state.player_str,
            st.session_state.dealer_str,
            step=st.session_state.step
        )
        st.session_state.last_rec = rec
        st.toast(f"Modelo sugiere: {rec.upper()}", icon="🧠")

    hit_disabled = st.session_state.round_over or not st.session_state.player_hand
    if st.button("➕ Hit", disabled=hit_disabled):
        player_hit(double=False)

    stand_disabled = st.session_state.round_over or not st.session_state.player_hand
    if st.button("✋ Stand", disabled=stand_disabled):
        dealer_play_to_17()

    double_disabled = st.session_state.round_over or not st.session_state.player_hand or st.session_state.step != 1
    if st.button("×2 Double (demo)", disabled=double_disabled):
        # Demos: solo 1 carta extra y terminar turno
        player_hit(double=True)
        if not st.session_state.round_over:
            dealer_play_to_17()

    # Split no implementado en esta demo (múltiples manos)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Panel de estado
    st.markdown("""
    <div class="card">
      <div style="font-weight:600;margin-bottom:8px;">Estado</div>
      <div style="display:flex;gap:10px;flex-direction:column;">
        <div><span class="pill">Paso</span> &nbsp; {step}</div>
        <div><span class="pill">Ronda</span> &nbsp; {ronda}</div>
        <div><span class="pill">Recomendación</span> &nbsp; {rec}</div>
      </div>
    </div>
    """.format(
        step=st.session_state.step,
        ronda=("Finalizada" if st.session_state.round_over else "En juego"),
        rec=st.session_state.last_rec.upper() if st.session_state.last_rec else "-"
    ), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /card
st.markdown('</div>', unsafe_allow_html=True)  # /container
