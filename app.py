# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Blackjack ML – Play", page_icon="🃏", layout="wide")

# ===================== THEME (LIGHT) & GLOBAL CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

:root{
  --bg: #f6f7fb;
  --panel: #ffffff;
  --panel-2:#f1f3f9;
  --border: rgba(31,36,48,.14);
  --text:#1f2430;
  --muted:#636e8b;
  --primary:#5568ff;
  --felt1:#0e8b5e;
  --felt2:#0b6c48;
  --wood:#b68458;
  --win:#1fa27a;
  --lose:#e04f56;
  --push:#b8860b;
}

html, body, [class*="css"] { background: var(--bg); color: var(--text); font-family: 'Poppins', sans-serif; }
.block-container{ padding-top: 1.0rem; padding-bottom: 1.2rem; }

h1.title{
  margin: 0 0 .6rem 0; font-weight: 800; letter-spacing:.2px;
}

.panel{
  background: var(--panel); border:1px solid var(--border); border-radius:18px;
  box-shadow: 0 12px 36px rgba(17,28,51,.08); padding: 14px 16px;
}

/* =================== Mesa centrada =================== */
.stage-wrap{ width:100%; display:flex; justify-content:center; }
.table{
  width: min(980px, 96%); min-height: 520px;
  background: radial-gradient(1200px 420px at 50% -20%, rgba(255,255,255,.25), transparent 60%),
              linear-gradient(180deg, var(--felt1) 0%, var(--felt2) 100%);
  border-radius: 26px;
  border: 7px solid var(--wood);
  box-shadow: 0 18px 60px rgba(0,0,0,.14), inset 0 0 140px rgba(0,0,0,.16);
  position: relative; overflow: hidden;
}
.table .banner{
  position:absolute; top:10px; left:50%; transform:translateX(-50%);
  color:#ffffff; font-weight:800; opacity:.9; letter-spacing:.4px;
}
.hands{
  position:absolute; inset:64px 24px 24px 24px;
  display:grid; grid-template-rows: 1fr 6px 1fr; align-items:center;
}
.divider{
  height:6px; width:100%; border-radius:999px;
  background: linear-gradient(90deg, rgba(255,255,255,.25), rgba(255,255,255,.55), rgba(255,255,255,.25));
  opacity:.75;
}
.hand-row{ display:flex; align-items:center; justify-content:center; gap:14px; height:100%; }
.hand-label{
  position:absolute; left:16px; top:8px; padding:7px 12px; border-radius:999px;
  background: rgba(255,255,255,.18); color:#fff; font-weight:800; font-size:12px; backdrop-filter: blur(2px);
}
.hand-row.player .hand-label{ top:auto; bottom:8px; }

/* =================== Cartas =================== */
.card{
  width: 82px; height: 114px; border-radius: 12px;
  background: linear-gradient(180deg,#fff 0%,#f3f3f3 100%);
  border:1px solid rgba(0,0,0,.12);
  box-shadow: 0 12px 30px rgba(0,0,0,.18);
  position:relative; display:grid; place-items:center;
}
.card .rank{ font-size: 24px; font-weight: 800; }
.card .suit{ font-size: 18px; }
.card.red{ color:#cc2336; } .card.black{ color:#161b24; }
.card .corner{ position:absolute; top:7px; left:7px; line-height:1.0; font-size:12px; font-weight:800; }
.card .s{ font-size:12px; }
.card.back{
  background: repeating-linear-gradient(45deg, #5568ff 0, #5568ff 6px, #4658e0 6px, #4658e0 12px);
  border-color: rgba(0,0,0,.18);
}

/* =================== Controles =================== */
.controls{ width:min(980px,96%); margin: 10px auto 0; }
.btn{
  display:inline-flex; align-items:center; justify-content:center; gap:8px;
  font-weight:800; font-size:14px; padding:10px 14px; border-radius:12px; border:1px solid var(--border);
  background: var(--panel); box-shadow: 0 8px 22px rgba(17,28,51,.06); color: var(--text);
}
.btn:hover{ border-color: rgba(85,104,255,.55); }
.btn.primary{ background: linear-gradient(180deg,#6b7cff,#5568ff); color:#fff; border-color:transparent; }
.btn.warn{ background:#ffe7e9; color:#b03542; border-color:#ffc7cc; }
.btn.alt{ background:#e6faff; color:#0d5b65; border-color:#b3f1fb; }

.rec-chip{
  display:inline-flex; gap:8px; align-items:center;
  background:#eef0ff; border:1px solid #cad0ff; color:#3842a8;
  padding:8px 12px; border-radius:999px; font-weight:800;
}
.stat{ background:var(--panel-2); border:1px dashed var(--border); border-radius:12px; padding:10px 12px; }
.stat b{ font-size:20px; }

.result-win{ color:var(--win); font-weight:800; }
.result-lose{ color:var(--lose); font-weight:800; }
.result-push{ color:var(--push); font-weight:800; }
</style>
""", unsafe_allow_html=True)

# ===================== MODEL HELPERS =====================
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
    seen, out = set(), []
    for c in cols:
        if c not in seen: seen.add(c); out.append(c)
    return out

def ensure_expected_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            if col in ("player_cards", "dealer_cards"): df[col] = ""
            elif col in ("step", "round_id", "hand_number"): df[col] = 1
            elif col in ("game_id",): df[col] = 0
            elif col in ("bet_mode", "strategy_used"): df[col] = "unknown"
            else: df[col] = np.nan
    return df

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None): self.columns_to_drop = columns_to_drop or []
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
            while total > 21 and aces > 0: total -= 10; aces -= 1
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
        "player_cards": player_cards, "dealer_cards": dealer_cards, "step": step,
        "game_id": 1, "round_id": 1, "hand_number": 1, "bet_mode": "flat", "strategy_used": "unknown",
    }
    if extra_cols: row.update(extra_cols)
    X = pd.DataFrame([row])
    if expected_cols: X = ensure_expected_columns(X, expected_cols)
    return model.predict(X)[0]

# ===================== GAME ENGINE =====================
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]

def new_shoe(num_decks=4):
    shoe = [(r,s) for _ in range(num_decks) for r in RANKS for s in SUITS]
    random.shuffle(shoe); return shoe

def total(cards):
    vals, aces = [], 0
    for r,_ in cards:
        if r in ["J","Q","K"]: vals.append(10)
        elif r == "A": vals.append(11); aces += 1
        else: vals.append(int(r))
    s = sum(vals)
    while s > 21 and aces > 0: s -= 10; aces -= 1
    return s

def cards_to_rank_csv(cards): return ", ".join([r for r,_ in cards])

def card_html(rank, suit, back=False):
    if back:
        return '<div class="card back"></div>'
    is_red = suit in ["♥","♦"]
    return f'''
      <div class="card {"red" if is_red else "black"}'>
        <div class="corner">{rank}<div class="s">{suit}</div></div>
        <div class="rank">{rank}</div>
        <div class="suit">{suit}</div>
      </div>
    '''

def render_table(dealer_cards, player_cards, reveal_dealer=False):
    st.markdown('<div class="stage-wrap"><div class="table">', unsafe_allow_html=True)
    st.markdown('<div class="banner">BLACKJACK</div>', unsafe_allow_html=True)
    st.markdown('<div class="hands">', unsafe_allow_html=True)

    # Dealer (fila superior)
    st.markdown('<div class="hand-row dealer" style="position:relative;">', unsafe_allow_html=True)
    label = f'Crupier • Total: {total(dealer_cards) if (reveal_dealer and dealer_cards) else "??"}'
    st.markdown(f'<div class="hand-label">{label}</div>', unsafe_allow_html=True)
    if not reveal_dealer and len(dealer_cards) >= 2:
        r0,s0 = dealer_cards[0]
        st.markdown(card_html(r0,s0), unsafe_allow_html=True)
        st.markdown(card_html("", "", back=True), unsafe_allow_html=True)
        for r,s in dealer_cards[2:]:
            st.markdown(card_html(r,s), unsafe_allow_html=True)
    else:
        for r,s in dealer_cards:
            st.markdown(card_html(r,s), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Player (fila inferior)
    st.markdown('<div class="hand-row player" style="position:relative;">', unsafe_allow_html=True)
    plabel = f'Jugador • Total: {total(player_cards) if player_cards else 0}'
    st.markdown(f'<div class="hand-label">{plabel}</div>', unsafe_allow_html=True)
    for r,s in player_cards:
        st.markdown(card_html(r,s), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div></div></div>', unsafe_allow_html=True)

# ===================== STATE =====================
if "shoe" not in st.session_state: st.session_state.shoe = new_shoe(4)
if "player_hand" not in st.session_state: st.session_state.player_hand = []
if "dealer_hand" not in st.session_state: st.session_state.dealer_hand = []
if "step" not in st.session_state: st.session_state.step = 1
if "round_over" not in st.session_state: st.session_state.round_over = True
if "history" not in st.session_state: st.session_state.history = []
if "last_rec" not in st.session_state: st.session_state.last_rec = None
if "result" not in st.session_state: st.session_state.result = None

def push_state():
    st.session_state.history.append(dict(
        shoe=list(st.session_state.shoe),
        player=list(st.session_state.player_hand),
        dealer=list(st.session_state.dealer_hand),
        step=st.session_state.step,
        round_over=st.session_state.round_over,
        last_rec=st.session_state.last_rec,
        result=st.session_state.result,
    ))
def pop_state():
    if not st.session_state.history: return
    s = st.session_state.history.pop()
    st.session_state.shoe = list(s["shoe"])
    st.session_state.player_hand = list(s["player"])
    st.session_state.dealer_hand = list(s["dealer"])
    st.session_state.step = s["step"]
    st.session_state.round_over = s["round_over"]
    st.session_state.last_rec = s["last_rec"]
    st.session_state.result = s["result"]

# ===================== UI =====================
st.markdown('<h1 class="title">🃏 Blackjack ML</h1>', unsafe_allow_html=True)

# --- Mesa centrada ---
render_table(
    st.session_state.dealer_hand,
    st.session_state.player_hand,
    reveal_dealer=st.session_state.round_over
)

# --- Controles centrados ---
st.markdown('<div class="controls panel">', unsafe_allow_html=True)

cols_top = st.columns([1,1,1,1,1], gap="medium")
if st.session_state.round_over:
    if cols_top[0].button("🂠 Repartir", use_container_width=True):
        st.session_state.history.clear(); st.session_state.result = None
        st.session_state.shoe = new_shoe(4)
        st.session_state.player_hand = []; st.session_state.dealer_hand = []
        st.session_state.step = 1; st.session_state.round_over = False
        push_state()
        st.session_state.player_hand.append(st.session_state.shoe.pop())
        st.session_state.dealer_hand.append(st.session_state.shoe.pop())
        st.session_state.player_hand.append(st.session_state.shoe.pop())
        st.session_state.dealer_hand.append(st.session_state.shoe.pop())
    if cols_top[1].button("🔄 Nueva mano", use_container_width=True):
        st.session_state.history.clear(); st.session_state.result = None
        st.session_state.shoe = new_shoe(4)
        st.session_state.player_hand = []; st.session_state.dealer_hand = []
        st.session_state.step = 1; st.session_state.round_over = True
        st.session_state.last_rec = None
else:
    if cols_top[0].button("🤖 Recomendar", use_container_width=True):
        player_csv = cards_to_rank_csv(st.session_state.player_hand)
        dealer_csv = cards_to_rank_csv(st.session_state.dealer_hand)
        rec = recommend_action(player_csv, dealer_csv, step=st.session_state.step)
        st.session_state.last_rec = rec
        st.toast(f"Modelo sugiere: {rec.upper()}")
    if cols_top[1].button("↩️ Deshacer", use_container_width=True):
        pop_state()

    if st.session_state.last_rec:
        st.markdown(f'<div class="rec-chip">Sugerencia: <b>{st.session_state.last_rec.upper()}</b></div>', unsafe_allow_html=True)

    st.write("")
    cols_actions = st.columns(4, gap="medium")

    def do_hit():
        push_state()
        st.session_state.player_hand.append(st.session_state.shoe.pop())
        st.session_state.step += 1
        if total(st.session_state.player_hand) > 21:
            st.session_state.result = ("lose","¡Te pasaste! Pierdes la mano.")
            st.session_state.round_over = True

    def do_stand():
        push_state()
        while total(st.session_state.dealer_hand) < 17 and len(st.session_state.shoe) > 0:
            st.session_state.dealer_hand.append(st.session_state.shoe.pop())
        p, d = total(st.session_state.player_hand), total(st.session_state.dealer_hand)
        if d > 21 or p > d: st.session_state.result=("win","¡Ganaste!")
        elif p < d:         st.session_state.result=("lose","Perdiste 😢")
        else:               st.session_state.result=("push","Empate (push).")
        st.session_state.round_over = True

    def do_double():
        push_state()
        st.session_state.player_hand.append(st.session_state.shoe.pop())
        st.session_state.step += 1
        if total(st.session_state.player_hand) > 21:
            st.session_state.result=("lose","¡Te pasaste con el Doble! Pierdes la mano.")
            st.session_state.round_over = True
        else:
            do_stand()

    def do_split():
        st.info("Split no implementado en esta demo.")

    cols_actions[0].button("🖐️ HIT",   use_container_width=True, on_click=do_hit)
    cols_actions[1].button("✋ STAND",  use_container_width=True, on_click=do_stand)
    cols_actions[2].button("🟰 DOUBLE", use_container_width=True, on_click=do_double)
    cols_actions[3].button("🔀 SPLIT",  use_container_width=True, on_click=do_split)

st.write("")
cols_state = st.columns(3, gap="large")
cols_state[0].markdown(f'<div class="stat">Paso<br><b>{st.session_state.step}</b></div>', unsafe_allow_html=True)
cols_state[1].markdown(f'<div class="stat">Total Jugador<br><b>{total(st.session_state.player_hand) if st.session_state.player_hand else 0}</b></div>', unsafe_allow_html=True)
visible = st.session_state.dealer_hand[0][0] if st.session_state.dealer_hand else "-"
cols_state[2].markdown(f'<div class="stat">Dealer visible<br><b>{visible}</b></div>', unsafe_allow_html=True)

if st.session_state.round_over and st.session_state.result:
    kind, msg = st.session_state.result
    cls = "result-win" if kind=="win" else "result-lose" if kind=="lose" else "result-push"
    st.markdown(f"<p class='{cls}' style='margin-top:.6rem'>{msg}</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close .controls
