# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ===================== CONFIG & THEME =====================
st.set_page_config(page_title="Blackjack ML – Play", page_icon="🃏", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

:root {
  --bg: #0e1117;
  --panel: #131a22;
  --panel-2: #1b2330;
  --acc: #8a7dff;
  --acc-2: #00e5ff;
  --text: #eef2f7;
  --muted: #a6b0c3;
  --win: #1fa27a;
  --lose: #e04f56;
  --push: #d1a110;
}

html, body, [class*="css"]  {
  font-family: 'Poppins', sans-serif;
}
.main, .block-container {
  padding-top: 0.5rem;
}
.hero {
  background: radial-gradient(1200px 400px at 10% -10%, rgba(138,125,255,0.15), transparent 60%),
              radial-gradient(1000px 500px at 110% 10%, rgba(0,229,255,0.10), transparent 60%),
              linear-gradient(180deg, #0c0f14 0%, var(--bg) 35%, #0c0f14 100%);
  border-radius: 18px;
  padding: 24px 24px 16px;
  border: 1px solid rgba(138,125,255,0.15);
  box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 0 40px rgba(138,125,255,0.05);
}
.hero h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: 0.4px;
  font-weight: 800;
  color: var(--text);
}
.hero p {
  margin: 6px 0 0;
  color: var(--muted);
}

.table-board {
  background: linear-gradient(180deg, #0a3829 0%, #0b2d23 100%);
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 12px 50px rgba(0,0,0,0.45), inset 0 0 120px rgba(255,255,255,0.06);
  padding: 20px;
  position: relative;
  overflow: hidden;
  min-height: 360px;
}
.glow-ring {
  position: absolute;
  width: 420px; height: 420px;
  border-radius: 50%;
  left: calc(50% - 210px);
  top: -200px;
  background: radial-gradient(closest-side, rgba(255,255,255,0.15), transparent 65%);
  filter: blur(20px);
  opacity: .18;
  pointer-events: none;
}
.section-title {
  color: var(--text);
  font-weight: 700;
  font-size: 18px;
  margin-bottom: 10px;
}

.row {
  display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
}
.badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 10px;
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 999px; color: var(--text);
  font-size: 12px; font-weight: 600;
}

.panel {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 14px;
}

.stat {
  background: rgba(0,0,0,0.25);
  border: 1px dashed rgba(255,255,255,0.15);
  border-radius: 12px;
  padding: 10px 12px;
  color: var(--text);
}
.stat b { font-size: 20px; }

.action-row {
  display: flex; gap: 10px; flex-wrap: wrap;
}
.action-btn {
  background: linear-gradient(180deg, #20293a, #1a2231);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px;
  padding: 10px 14px;
  color: var(--text); font-weight: 700; font-size: 14px;
}
.action-btn:hover { border-color: rgba(138,125,255,0.45); }

.rec-chip {
  background: linear-gradient(180deg, rgba(138,125,255,0.18), rgba(138,125,255,0.06));
  border: 1px solid rgba(138,125,255,0.4);
  color: #dcd8ff;
  padding: 8px 12px; border-radius: 999px; font-weight: 700;
  display: inline-flex; gap: 8px; align-items: center;
}

.result-win { color: var(--win); font-weight: 800; }
.result-lose { color: var(--lose); font-weight: 800; }
.result-push { color: var(--push); font-weight: 800; }

/* ===== Cards ===== */
.cards {
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}
.card {
  width: 64px; height: 90px; border-radius: 10px;
  background: linear-gradient(180deg, #fefefe 0%, #f7f7f7 100%);
  border: 1px solid rgba(0,0,0,0.15);
  box-shadow: 0 8px 20px rgba(0,0,0,.25);
  display: grid; place-items: center;
  position: relative;
}
.card .rank { font-size: 20px; font-weight: 800; }
.card .suit { font-size: 18px; }
.card.red { color: #d12c2c; }
.card.black { color: #101015; }

.card .corner {
  position: absolute; top: 6px; left: 6px; text-align: left; line-height: 1.0;
  font-size: 12px; font-weight: 800;
}
.card .corner .s { font-size: 12px; }

.dealer-area, .player-area {
  background: rgba(0,0,0,0.12);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px;
}

hr.soft {
  border: none; height: 1px; background: linear-gradient(to right, transparent, rgba(255,255,255,0.15), transparent);
  margin: 10px 0 14px;
}
</style>
""", unsafe_allow_html=True)

# ===================== MODEL LOADING W/ COLUMN SAFETY =====================
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
    for _, _, cols_spec in getattr(ct, "transformers_", []):
        if cols_spec == "drop" or cols_spec is None:
            continue
        if isinstance(cols_spec, (list, tuple)):
            cols.extend([c for c in cols_spec if isinstance(c, str)])
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

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
                df[col] = "unknown"
            else:
                df[col] = np.nan
    return df

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.columns_to_drop)

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
            total = sum(values); aces = card_list.count("A")
            while total > 21 and aces > 0:
                total -= 10; aces -= 1
            return total
        def dealer_value(cards):
            first = str(cards).split(",")[0].strip().upper()
            if first in ["J","Q","K"]: return 10
            if first == "A": return 11
            try: return int(first)
            except ValueError: return 0
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

def recommend_action(player_cards, dealer_cards, step=1, extra_cols=None):
    row = {
        "player_cards": player_cards,
        "dealer_cards": dealer_cards,
        "step": step,
        "game_id": 1, "round_id": 1, "hand_number": 1,
        "bet_mode": "flat", "strategy_used": "unknown",
    }
    if extra_cols: row.update(extra_cols)
    X = pd.DataFrame([row])
    if expected_cols: X = ensure_expected_columns(X, expected_cols)
    return model.predict(X)[0]

# ===================== GAME ENGINE =====================
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]

def new_shoe(num_decks=4):
    shoe = [(r, s) for _ in range(num_decks) for r in RANKS for s in SUITS]
    random.shuffle(shoe); return shoe

def card_value_total(cards):
    vals = []
    for r, _ in cards:
        if r in ["J","Q","K"]: vals.append(10)
        elif r == "A": vals.append(11)
        else: vals.append(int(r))
    total = sum(vals); aces = sum(1 for r,_ in cards if r=="A")
    while total > 21 and aces > 0:
        total -= 10; aces -= 1
    return total

def cards_to_rank_csv(cards):
    return ", ".join([r for r,_ in cards])

def suit_symbol(s):
    return {"♠":"♠","♥":"♥","♦":"♦","♣":"♣"}[s]

def card_html(rank, suit):
    is_red = suit in ["♥","♦"]
    return f"""
    <div class="card {'red' if is_red else 'black'}">
      <div class="corner">{rank}<div class="s">{suit_symbol(suit)}</div></div>
      <div class="rank">{rank}</div>
      <div class="suit">{suit_symbol(suit)}</div>
    </div>
    """

def render_hand(label, cards, reveal_all=True):
    total = card_value_total(cards) if reveal_all else card_value_total(cards[:1])
    vis_label = f"{label} • Total: {total if reveal_all else '??'}"
    st.markdown(f'<div class="section-title">{vis_label}</div>', unsafe_allow_html=True)
    st.markdown('<div class="cards">', unsafe_allow_html=True)
    # Render all cards (dealer shows both en resultado final; durante juego, mostramos primera carta en texto aparte)
    for i, (r,s) in enumerate(cards):
        st.markdown(card_html(r, s), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def push_state():
    snap = dict(
        shoe=list(st.session_state.shoe),
        player_hand=list(st.session_state.player_hand),
        dealer_hand=list(st.session_state.dealer_hand),
        step=st.session_state.step,
        round_over=st.session_state.round_over,
        last_rec=st.session_state.get("last_rec", None),
        result=st.session_state.get("result", None),
    )
    st.session_state.history.append(snap)

def pop_state():
    if not st.session_state.history:
        return
    snap = st.session_state.history.pop()
    st.session_state.shoe = list(snap["shoe"])
    st.session_state.player_hand = list(snap["player_hand"])
    st.session_state.dealer_hand = list(snap["dealer_hand"])
    st.session_state.step = snap["step"]
    st.session_state.round_over = snap["round_over"]
    st.session_state.last_rec = snap["last_rec"]
    st.session_state.result = snap["result"]

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("<h1>🃏 Blackjack ML</h1>", unsafe_allow_html=True)
    st.markdown("<p>Jugá una mano y pedile consejo al modelo entrenado.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    num_decks = st.selectbox("N° de mazos", [1,2,4,6,8], index=2)
    st.caption("Tip: podés **Deshacer** el último paso si querés probar otra acción.")

# ===================== STATE INIT =====================
if "shoe" not in st.session_state:
    st.session_state.shoe = new_shoe(num_decks)
if "player_hand" not in st.session_state:
    st.session_state.player_hand = []
if "dealer_hand" not in st.session_state:
    st.session_state.dealer_hand = []
if "step" not in st.session_state:
    st.session_state.step = 1
if "round_over" not in st.session_state:
    st.session_state.round_over = True
if "history" not in st.session_state:
    st.session_state.history = []
if "last_rec" not in st.session_state:
    st.session_state.last_rec = None
if "result" not in st.session_state:
    st.session_state.result = None

# ===================== LAYOUT =====================
left, right = st.columns([3,2], gap="large")

with left:
    st.markdown('<div class="table-board">', unsafe_allow_html=True)
    st.markdown('<div class="glow-ring"></div>', unsafe_allow_html=True)

    # Dealer
    st.markdown('<div class="dealer-area">', unsafe_allow_html=True)
    if st.session_state.dealer_hand:
        render_hand("Crupier", st.session_state.dealer_hand, reveal_all=st.session_state.round_over)
        if not st.session_state.round_over and len(st.session_state.dealer_hand) >= 1:
            st.markdown(f"<div class='badge'>Visible: {st.session_state.dealer_hand[0][0]}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-title'>Crupier</div>", unsafe_allow_html=True)
        st.markdown("<div class='badge'>Sin cartas</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    # Player
    st.markdown('<div class="player-area">', unsafe_allow_html=True)
    if st.session_state.player_hand:
        render_hand("Jugador", st.session_state.player_hand, reveal_all=True)
    else:
        st.markdown("<div class='section-title'>Jugador</div>", unsafe_allow_html=True)
        st.markdown("<div class='badge'>Sin cartas</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-title'>Controles</div>", unsafe_allow_html=True)
    ctrl = st.container()
    with ctrl:
        if st.session_state.round_over:
            c1, c2 = st.columns(2)
            if c1.button("🂠 Repartir"):
                # reset & deal
                st.session_state.history.clear()
                st.session_state.result = None
                st.session_state.shoe = new_shoe(num_decks)
                st.session_state.player_hand = []
                st.session_state.dealer_hand = []
                st.session_state.step = 1
                st.session_state.round_over = False

                # deal 2 & 2
                push_state()
                st.session_state.player_hand.append(st.session_state.shoe.pop())
                st.session_state.dealer_hand.append(st.session_state.shoe.pop())
                st.session_state.player_hand.append(st.session_state.shoe.pop())
                st.session_state.dealer_hand.append(st.session_state.shoe.pop())

            if c2.button("🔄 Nueva mano"):
                st.session_state.history.clear()
                st.session_state.result = None
                st.session_state.shoe = new_shoe(num_decks)
                st.session_state.player_hand = []
                st.session_state.dealer_hand = []
                st.session_state.step = 1
                st.session_state.round_over = True
                st.session_state.last_rec = None
        else:
            # Live controls
            top = st.container()
            with top:
                c1, c2 = st.columns([2,1])
                if c1.button("🤖 Recomendar acción", use_container_width=True):
                    player_csv = cards_to_rank_csv(st.session_state.player_hand)
                    dealer_csv = cards_to_rank_csv(st.session_state.dealer_hand)
                    rec = recommend_action(player_csv, dealer_csv, step=st.session_state.step)
                    st.session_state.last_rec = rec
                    st.toast(f"Modelo sugiere: {rec.upper()}")

                if c2.button("↩️ Deshacer último paso", use_container_width=True):
                    pop_state()

                # Show recommendation chip
                if st.session_state.last_rec:
                    st.markdown(f"<div class='rec-chip'>Sugerencia del modelo: <b>{st.session_state.last_rec.upper()}</b></div>", unsafe_allow_html=True)

            st.write("")
            st.markdown("<div class='section-title'>Acciones</div>", unsafe_allow_html=True)
            ac1, ac2, ac3, ac4 = st.columns(4)

            def player_hit():
                push_state()
                st.session_state.player_hand.append(st.session_state.shoe.pop())
                st.session_state.step += 1
                if card_value_total(st.session_state.player_hand) > 21:
                    st.session_state.result = ("lose", "¡Te pasaste! Pierdes la mano.")
                    st.session_state.round_over = True

            def player_stand():
                push_state()
                # Dealer plays to 17+
                while card_value_total(st.session_state.dealer_hand) < 17 and len(st.session_state.shoe) > 0:
                    st.session_state.dealer_hand.append(st.session_state.shoe.pop())
                p, d = card_value_total(st.session_state.player_hand), card_value_total(st.session_state.dealer_hand)
                if d > 21 or p > d:
                    st.session_state.result = ("win", "¡Ganaste!")
                elif p < d:
                    st.session_state.result = ("lose", "Perdiste 😢")
                else:
                    st.session_state.result = ("push", "Empate (push).")
                st.session_state.round_over = True

            def player_double():
                # Didáctico: lo tratamos como un hit + finalizar turno (si no se pasó)
                push_state()
                st.session_state.player_hand.append(st.session_state.shoe.pop())
                st.session_state.step += 1
                if card_value_total(st.session_state.player_hand) > 21:
                    st.session_state.result = ("lose", "¡Te pasaste con el Doble! Pierdes la mano.")
                    st.session_state.round_over = True
                else:
                    player_stand()

            def player_split():
                # No implementado (requiere múltiples manos). Conservamos botón por completitud.
                st.info("Split no implementado en esta demo.")

            if ac1.button("🖐️ HIT", use_container_width=True): player_hit()
            if ac2.button("✋ STAND", use_container_width=True): player_stand()
            if ac3.button("🟰 DOUBLE", use_container_width=True): player_double()
            if ac4.button("🔀 SPLIT", use_container_width=True): player_split()

            st.write("")
            st.markdown("<div class='section-title'>Estado</div>", unsafe_allow_html=True)
            with st.container():
                p_total = card_value_total(st.session_state.player_hand) if st.session_state.player_hand else 0
                d_visible = st.session_state.dealer_hand[0][0] if st.session_state.dealer_hand else "-"
                cA, cB, cC = st.columns(3)
                cA.markdown(f"<div class='stat'>Paso<br><b>{st.session_state.step}</b></div>", unsafe_allow_html=True)
                cB.markdown(f"<div class='stat'>Total Jugador<br><b>{p_total}</b></div>", unsafe_allow_html=True)
                cC.markdown(f"<div class='stat'>Dealer visible<br><b>{d_visible}</b></div>", unsafe_allow_html=True)

    st.write("")
    # Result banner if round over
    if st.session_state.round_over and st.session_state.result:
        kind, msg = st.session_state.result
        css = "result-win" if kind=="win" else "result-lose" if kind=="lose" else "result-push"
        st.markdown(f"<div class='{css}' style='font-size:18px;'>{msg}</div>", unsafe_allow_html=True)
