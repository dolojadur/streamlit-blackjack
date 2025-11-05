# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ===================== CONFIG GENERAL =====================
st.set_page_config(page_title="Blackjack ML – Play", page_icon="🃏", layout="wide")

# ====== Tema claro + UI ======
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

:root {
  --bg: #f6f7fb;
  --panel: #ffffff;
  --panel-2: #f0f2f7;
  --border: rgba(24, 32, 60, 0.12);
  --text: #1f2430;
  --muted: #5b6682;
  --primary: #5b6cff; /* violeta azulado suave */
  --primary-2: #00bcd4; /* celeste */
  --win: #1fa27a;
  --lose: #e04f56;
  --push: #b8860b;
  --felt1: #0d8b5b;      /* verde paño */
  --felt2: #0b6d49;
}

html, body, [class*="css"]  {
  font-family: 'Poppins', sans-serif;
  background: var(--bg);
  color: var(--text);
}
.block-container { padding-top: 0.8rem; }

h1,h2,h3 { letter-spacing: .2px; }

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(17,28,51,0.06);
}

/* =================== Mesa =================== */
.table-wrap {
  display: flex; justify-content: center; width: 100%;
}
.table {
  width: min(1000px, 96%);
  min-height: 460px;
  margin: 8px 0 14px;
  background: radial-gradient(1200px 420px at 50% -20%, rgba(255,255,255,.24), transparent 60%),
              linear-gradient(180deg, var(--felt1) 0%, var(--felt2) 100%);
  border-radius: 24px;
  border: 6px solid #b48658;            /* borde madera */
  box-shadow: 0 18px 60px rgba(0,0,0,.12), inset 0 0 120px rgba(0,0,0,.15);
  position: relative;
  overflow: hidden;
  padding: 22px 18px;
}
.table .title {
  position: absolute; left: 50%; transform: translateX(-50%);
  top: 10px; color: #fff; opacity: .9; font-weight: 700; letter-spacing:.4px;
}
.hands {
  position: absolute; inset: 56px 22px 22px 22px;
  display: grid;
  grid-template-rows: 1fr 4px 1fr;
  align-items: center;
}
.divider {
  height: 4px; width: 100%;
  background: linear-gradient(90deg, rgba(255,255,255,.18), rgba(255,255,255,.35), rgba(255,255,255,.18));
  border-radius: 999px;
  opacity: .65;
}
.hand-row {
  display: flex;
  height: 100%;
  align-items: center;
  justify-content: center;
  gap: 12px;
}
.hand-label {
  position: absolute; left: 18px; top: 8px;
  background: rgba(255,255,255,.18);
  color: #fff; font-weight: 700;
  padding: 6px 10px; border-radius: 999px; font-size: 12px;
  backdrop-filter: blur(2px);
}
.hand-row.player .hand-label { top: auto; bottom: 8px; }

/* =================== Cartas =================== */
.card {
  width: 76px; height: 104px; border-radius: 12px;
  background: linear-gradient(180deg, #fff 0%, #f3f3f3 100%);
  border: 1px solid rgba(0,0,0,.12);
  box-shadow: 0 10px 26px rgba(0,0,0,.18);
  position: relative; display: grid; place-items: center;
}
.card .rank { font-size: 22px; font-weight: 800; }
.card .suit { font-size: 18px; }
.card.red { color: #cc2336; }
.card.black { color: #141821; }
.card .corner {
  position: absolute; top: 6px; left: 6px;
  text-align: left; line-height: 1.0; font-size: 12px; font-weight: 800;
}
.card .s { font-size: 12px; }
.card.back {
  background: repeating-linear-gradient(
      45deg, #5b6cff 0, #5b6cff 6px, #4758e0 6px, #4758e0 12px
  );
  border-color: rgba(0,0,0,.18);
}

/* =================== Controles =================== */
.btn {
  display: inline-flex; align-items: center; justify-content: center;
  gap: 8px; font-weight: 800; font-size: 14px;
  padding: 10px 14px; border-radius: 12px; border: 1px solid var(--border);
  background: var(--panel);
  box-shadow: 0 6px 18px rgba(17,28,51,0.06);
}
.btn:hover { border-color: rgba(91,108,255,.55); }
.btn.primary {
  background: linear-gradient(180deg, #6b7cff, #5568ff);
  color: #fff; border-color: transparent;
}
.btn.warn { background: #ffe7e9; color: #b03542; border-color: #ffc7cc; }
.btn.alt { background: #e6faff; color: #0d5b65; border-color: #b3f1fb; }

.stat {
  background: var(--panel-2);
  border: 1px dashed var(--border);
  border-radius: 12px; padding: 10px 12px;
  color: var(--text);
}
.stat b { font-size: 20px; }
.rec-chip {
  background: #eef0ff;
  border: 1px solid #cad0ff;
  color: #3842a8;
  padding: 8px 12px; border-radius: 999px; font-weight: 800;
  display: inline-flex; gap: 8px; align-items: center;
}
.result-win { color: var(--win); font-weight: 800; }
.result-lose { color: var(--lose); font-weight: 800; }
.result-push { color: var(--push); font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ===================== Utilidades modelo =====================
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
        if cols_spec == "drop" or cols_spec is None: continue
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

# ===================== Motor de juego =====================
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]

def new_shoe(num_decks=4):
    shoe = [(r, s) for _ in range(num_decks) for r in RANKS for s in SUITS]
    random.shuffle(shoe); return shoe

def total(cards):
    vals = []
    aces = 0
    for r, _ in cards:
        if r in ["J","Q","K"]: vals.append(10)
        elif r == "A": vals.append(11); aces += 1
        else: vals.append(int(r))
    s = sum(vals)
    while s > 21 and aces > 0:
        s -= 10; aces -= 1
    return s

def cards_to_rank_csv(cards): return ", ".join([r for r,_ in cards])

def card_html(rank, suit, back=False):
    if back:
        return '<div class="card back"></div>'
    is_red = suit in ["♥","♦"]
    return f"""
      <div class="card {'red' if is_red else 'black'}">
        <div class="corner">{rank}<div class="s">{suit}</div></div>
        <div class="rank">{rank}</div>
        <div class="suit">{suit}</div>
      </div>
    """

def render_table(dealer_cards, player_cards, reveal_dealer=False):
    st.markdown('<div class="table-wrap"><div class="table">', unsafe_allow_html=True)
    st.markdown('<div class="title">BLACKJACK</div>', unsafe_allow_html=True)
    st.markdown('<div class="hands">', unsafe_allow_html=True)

    # Dealer row
    st.markdown('<div class="hand-row dealer" style="position:relative;">', unsafe_allow_html=True)
    lab = f"Crupier • Total: {total(dealer_cards) if reveal_dealer and dealer_cards else '??'}"
    st.markdown(f'<div class="hand-label">{lab}</div>', unsafe_allow_html=True)

    if not reveal_dealer and len(dealer_cards) >= 2:
        # primera boca arriba, segunda boca abajo
        r0,s0 = dealer_cards[0]
        st.markdown(card_html(r0,s0), unsafe_allow_html=True)
        st.markdown(card_html("","", back=True), unsafe_allow_html=True)
        for r,s in dealer_cards[2:]:
            st.markdown(card_html(r,s), unsafe_allow_html=True)
    else:
        for r,s in dealer_cards:
            st.markdown(card_html(r,s), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Player row
    st.markdown('<div class="hand-row player" style="position:relative;">', unsafe_allow_html=True)
    plab = f"Jugador • Total: {total(player_cards) if player_cards else 0}"
    st.markdown(f'<div class="hand-label">{plab}</div>', unsafe_allow_html=True)
    for r,s in player_cards:
        st.markdown(card_html(r,s), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div></div></div>', unsafe_allow_html=True)

# ===================== Sidebar (opciones breves) =====================
with st.sidebar:
    st.markdown("<div class='panel'><h3>🃏 Blackjack ML</h3><p style='color:var(--muted)'>Probá una mano y pedile consejo al modelo entrenado.</p></div>", unsafe_allow_html=True)
    num_decks = st.selectbox("N° de mazos", [1,2,4,6,8], index=2)
    st.caption("Podés **deshacer** el último paso para probar otra acción.")

# ===================== Estado =====================
if "shoe" not in st.session_state: st.session_state.shoe = new_shoe(num_decks)
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

# ===================== Layout principal =====================
left, right = st.columns([4, 2], gap="large")

with left:
    render_table(
        st.session_state.dealer_hand,
        st.session_state.player_hand,
        reveal_dealer=st.session_state.round_over
    )

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    # CONTROLES SUPERIORES
    c1, c2 = st.columns([1,1])
    if st.session_state.round_over:
        if c1.button("🂠 Repartir", use_container_width=True, type="primary"):
            st.session_state.history.clear()
            st.session_state.result = None
            st.session_state.shoe = new_shoe(num_decks)
            st.session_state.player_hand = []
            st.session_state.dealer_hand = []
            st.session_state.step = 1
            st.session_state.round_over = False

            push_state()
            st.session_state.player_hand.append(st.session_state.shoe.pop())
            st.session_state.dealer_hand.append(st.session_state.shoe.pop())
            st.session_state.player_hand.append(st.session_state.shoe.pop())
            st.session_state.dealer_hand.append(st.session_state.shoe.pop())
        if c2.button("🔄 Nueva mano", use_container_width=True):
            st.session_state.history.clear()
            st.session_state.result = None
            st.session_state.shoe = new_shoe(num_decks)
            st.session_state.player_hand = []
            st.session_state.dealer_hand = []
            st.session_state.step = 1
            st.session_state.round_over = True
            st.session_state.last_rec = None
    else:
        if c1.button("🤖 Recomendar acción", use_container_width=True):
            player_csv = cards_to_rank_csv(st.session_state.player_hand)
            dealer_csv = cards_to_rank_csv(st.session_state.dealer_hand)
            rec = recommend_action(player_csv, dealer_csv, step=st.session_state.step)
            st.session_state.last_rec = rec
            st.toast(f"Modelo sugiere: {rec.upper()}")
        if c2.button("↩️ Deshacer último paso", use_container_width=True):
            pop_state()

        if st.session_state.last_rec:
            st.markdown(f"<div class='rec-chip'>Sugerencia: <b>{st.session_state.last_rec.upper()}</b></div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("**Acciones**")
        a1, a2 = st.columns(2)
        a3, a4 = st.columns(2)

        def do_hit():
            push_state()
            st.session_state.player_hand.append(st.session_state.shoe.pop())
            st.session_state.step += 1
            if total(st.session_state.player_hand) > 21:
                st.session_state.result = ("lose", "¡Te pasaste! Pierdes la mano.")
                st.session_state.round_over = True

        def do_stand():
            push_state()
            while total(st.session_state.dealer_hand) < 17 and len(st.session_state.shoe) > 0:
                st.session_state.dealer_hand.append(st.session_state.shoe.pop())
            p, d = total(st.session_state.player_hand), total(st.session_state.dealer_hand)
            if d > 21 or p > d: st.session_state.result = ("win", "¡Ganaste!")
            elif p < d:        st.session_state.result = ("lose", "Perdiste 😢")
            else:              st.session_state.result = ("push", "Empate (push).")
            st.session_state.round_over = True

        def do_double():
            push_state()
            st.session_state.player_hand.append(st.session_state.shoe.pop())
            st.session_state.step += 1
            if total(st.session_state.player_hand) > 21:
                st.session_state.result = ("lose", "¡Te pasaste con el Doble! Pierdes la mano.")
                st.session_state.round_over = True
            else:
                do_stand()

        def do_split():
            st.info("Split no implementado en esta demo.")

        a1.button("🖐️ HIT", use_container_width=True, on_click=do_hit)
        a2.button("✋ STAND", use_container_width=True, on_click=do_stand)
        a3.button("🟰 DOUBLE", use_container_width=True, on_click=do_double)
        a4.button("🔀 SPLIT", use_container_width=True, on_click=do_split)

        st.write("")
        st.markdown("**Estado**")
        s1, s2, s3 = st.columns(3)
        s1.markdown(f"<div class='stat'>Paso<br><b>{st.session_state.step}</b></div>", unsafe_allow_html=True)
        s2.markdown(f"<div class='stat'>Total Jugador<br><b>{total(st.session_state.player_hand) if st.session_state.player_hand else 0}</b></div>", unsafe_allow_html=True)
        vis = st.session_state.dealer_hand[0][0] if st.session_state.dealer_hand else "-"
        s3.markdown(f"<div class='stat'>Dealer visible<br><b>{vis}</b></div>", unsafe_allow_html=True)

    # resultado final
    if st.session_state.round_over and st.session_state.result:
        kind, msg = st.session_state.result
        cls = "result-win" if kind=="win" else "result-lose" if kind=="lose" else "result-push"
        st.markdown(f"<p class='{cls}' style='margin:.6rem 0 0'>{msg}</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
