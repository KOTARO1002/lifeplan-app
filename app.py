
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from datetime import datetime
from pathlib import Path
import base64

# ========================
# 日本語フォント設定（文字化け防止）
# ========================
font_path = "fonts/ipaexg.ttf"  # フォントファイルのパス
fm.fontManager.addfont(font_path)
matplotlib.rcParams["font.family"] = "IPAexGothic"  # フォント内部名

# ========================
# ページ設定 & 全体デザイン
# ========================
st.set_page_config(page_title="ライフプランシミュレーション", layout="wide")

st.markdown(
    """
    <style>
      :root { --accent:#2D7FF9; --accent2:#00C2A8; --bgsoft:#F6F8FB; --text:#0F172A; }
      .app-title{
        font-size: 2.0rem; font-weight: 800; color: var(--text);
        letter-spacing: .02em; margin-bottom: .2rem;
      }
      .app-sub{
        color:#64748B; font-size: .95rem; margin-bottom: .9rem;
      }
      .section-card{
        background: var(--bgsoft);
        border: 1px solid #E2E8F0;
        padding: 1.0rem 1.1rem;
        border-radius: 14px;
        margin-bottom: 1.0rem;
      }
      /* DataFrameの横幅・表示・スクロール調整 */
      div[data-testid="stDataFrame"] { width: 100%; }
      div[data-testid="stDataFrame"] * { font-variant-numeric: tabular-nums; }
      div[data-testid="stDataFrame"] td, 
      div[data-testid="stDataFrame"] th {
        min-width: 110px !important;
        max-width: 220px !important;
        white-space: nowrap !important;
        font-size: 14px !important;
        padding: 6px 8px !important;
      }
      div[data-testid="stDataFrame"] th {
        background: #0B1220 !important;
        color: white !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #111827 !important;
      }
      div[data-testid="stDataFrame"] > div { overflow-y: hidden !important; }
      h2, h3 { letter-spacing: .02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# タイトル
st.markdown('<div class="app-title">ライフプランシミュレーション</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">収入・支出・教育費・投資をまとめて試算できる家計シミュレーター</div>', unsafe_allow_html=True)

# ----------------------------
# 基本情報
# ----------------------------
st.sidebar.header("基本情報")
start_age = st.sidebar.number_input("現在の年齢", 20, 60, 40)
start_year = datetime.today().year
end_age = 65
years = list(range(start_year, start_year + (end_age - start_age) + 1))

# 配偶者（サイドバーに表示）
has_spouse = st.sidebar.checkbox("配偶者あり")
if has_spouse:
    spouse_age = st.sidebar.number_input("配偶者の現在の年齢", 20, 60, 40)
else:
    spouse_age = None

# 年収（手取りベース）
income = st.sidebar.number_input("本人の手取り年収（円）", 0, 30_000_000, 6_000_000)
if has_spouse:
    spouse_income = st.sidebar.number_input("配偶者の手取り年収（円）", 0, 30_000_000, 3_000_000)
else:
    spouse_income = 0

# インフレ率 & 賃金上昇率（基本情報の最下段）
inflation = st.sidebar.slider("インフレ率（年%）", 0.0, 5.0, 1.0) / 100
wage_growth = st.sidebar.slider("賃金上昇率（年%）", 0.0, 5.0, 1.0) / 100

# ----------------------------
# 支出（生活費・住宅関連）
# ----------------------------
st.sidebar.header("支出関連")

life_cost_preset = st.sidebar.selectbox("生活費", ["ミニマム", "標準", "ゆとり", "手入力"])
life_cost_custom = 0
if life_cost_preset == "手入力":
    life_cost_custom = st.sidebar.number_input("生活費（年額・円）", 0, 30_000_000, 3_000_000)

# 住宅ローン：月額入力 → 年額に換算
debt_month = st.sidebar.number_input("住宅ローン返済（月額・円）", 0, 2_000_000, 150_000)
debt_year = debt_month * 12

# 住宅ローン残り年数
loan_years = st.sidebar.number_input("住宅ローンの残り返済年数（年）", 0, 40, 30)

# 管理費・修繕費：月額入力 → 年額に換算
repair_month = st.sidebar.number_input("管理費・修繕費（月額・円）", 0, 1_000_000, 20_000)
repair = repair_month * 12

# ----------------------------
# 教育費
# ----------------------------
st.sidebar.header("教育費（子どもごとに設定）")
children = st.sidebar.number_input("子どもの人数", 0, 3, 1)

child_settings = []
for i in range(int(children)):
    st.sidebar.subheader(f"子ども {i+1}")
    birth_year = st.sidebar.number_input(
        f"子ども{i+1}の誕生年（西暦）",
        start_year - 30,
        start_year + 30,
        start_year - 5,
        key=f"birth_{i}",
    )
    school_type = st.sidebar.selectbox(
        f"子ども{i+1} 中高タイプ", ["公立", "私立"], key=f"type_{i}"
    )
    cram_month = st.sidebar.number_input(
        f"子ども{i+1} 塾・学費（月額・円）", 0, 300_000, 20_000, key=f"cram_{i}"
    )
    uni_type = st.sidebar.selectbox(
        f"子ども{i+1} 進学先",
        ["進学しない", "国公立大学", "私立大学", "専門学校"],
        key=f"uni_{i}",
    )
    dorm = st.sidebar.checkbox(f"子ども{i+1} 下宿する", key=f"dorm_{i}")
    child_settings.append(
        {
            "birth_year": int(birth_year),
            "school_type": school_type,
            "cram_month": cram_month,
            "uni": uni_type,
            "dorm": dorm,
        }
    )

# ----------------------------
# 投資・貯蓄
# ----------------------------
st.sidebar.header("投資・貯蓄")
initial_savings = st.sidebar.number_input(
    "現在の貯蓄額（円）", 0, 300_000_000, 3_000_000
)
invest_principal = st.sidebar.number_input("投資元本（円）", 0, 100_000_000, 1_000_000)
invest_month = st.sidebar.number_input("毎月積立額（円）", 0, 1_000_000, 30_000)
invest_return = st.sidebar.slider("利回り（年率%）", 0.0, 15.0, 3.0) / 100

# ----------------------------
# 特別収入・特別支出（サイドバー最下部 / 万円単位）
# ----------------------------
st.sidebar.header("特別収入・特別支出（万円）")
st.sidebar.caption("学資保険の払戻しや臨時出費などを年ごとに追加できます（万円単位・年額）。")

special_df = pd.DataFrame({
    "西暦": years,
    "特別収入": [0] * len(years),
    "特別支出": [0] * len(years),
})

special_df = st.sidebar.data_editor(
    special_df,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={
        # intのまま表示フォーマットで「年」を付ける → 警告マーク対策
        "西暦": st.column_config.NumberColumn("西暦", disabled=True, format="%d年", width="small"),
        "特別収入": st.column_config.NumberColumn("特別収入", min_value=0, step=10, format="%d", width="small"),
        "特別支出": st.column_config.NumberColumn("特別支出", min_value=0, step=10, format="%d", width="small"),
    },
    key="special_editor_sidebar",
)

# 万円 → 円へ変換して辞書化
special_income_by_year = {int(k): int(v) * 10_000 for k, v in zip(special_df["西暦"], special_df["特別収入"])}
special_expense_by_year = {int(k): int(v) * 10_000 for k, v in zip(special_df["西暦"], special_df["特別支出"])}

# ===============================
# 生活費プリセット（都市部・子ども人数別ざっくりモデル）
# ===============================
LIFE_TABLE = {
    0: {"ミニマム": 2_200_000, "標準": 2_800_000, "ゆとり": 3_400_000},
    1: {"ミニマム": 2_600_000, "標準": 3_200_000, "ゆとり": 3_800_000},
    2: {"ミニマム": 3_000_000, "標準": 3_600_000, "ゆとり": 4_200_000},
    3: {"ミニマム": 3_400_000, "標準": 4_000_000, "ゆとり": 4_600_000},
}

def get_life_cost(num_children: int) -> int:
    if life_cost_preset == "手入力":
        return life_cost_custom
    n = max(0, min(3, int(num_children)))
    return LIFE_TABLE[n][life_cost_preset]

# ===============================
# 教育費モデル（すべて円）
# ===============================
PUBLIC_MH = 400_000
PRIVATE_MH = 900_000
UNI_COST = {
    "国公立大学": 800_000,
    "私立大学": 1_500_000,
    "専門学校": 1_100_000,
}
UNI_YEARS = {"国公立大学": 4, "私立大学": 4, "専門学校": 2}
ENTRANCE_FEE = 300_000
DORM_COST = 800_000

def education_cost_for_year(year_index: int) -> int:
    current_year = start_year + year_index
    total = 0
    for cs in child_settings:
        child_age = current_year - cs["birth_year"]

        if 12 <= child_age <= 18:
            total += PRIVATE_MH if cs["school_type"] == "私立" else PUBLIC_MH

        if 6 <= child_age <= 22 and cs["cram_month"] > 0:
            total += cs["cram_month"] * 12

        if cs["uni"] != "進学しない" and child_age >= 18:
            uy = UNI_YEARS[cs["uni"]]
            if 18 <= child_age < 18 + uy:
                total += UNI_COST[cs["uni"]]
                if child_age == 18:
                    total += ENTRANCE_FEE
                if cs["dorm"]:
                    total += DORM_COST
    return int(total)

# ===============================
# シミュレーション本体
# ===============================
records = []
cumulative = initial_savings
invest_balance = invest_principal
num_children = len(child_settings)

for idx, year in enumerate(years):
    age = start_age + idx
    spouse_age_year = spouse_age + idx if (has_spouse and spouse_age is not None) else None

    # 収入（本人・配偶者を分解）
    person_income = int(income * (1 + wage_growth) ** idx)
    spouse_income_year = int(spouse_income * (1 + wage_growth) ** idx) if has_spouse else 0

    # 特別収入（円）
    special_income = int(special_income_by_year.get(year, 0))

    # 収入合計
    annual_income = person_income + spouse_income_year + special_income

    # 生活費（子どもの人数 & インフレ連動）
    base_life = get_life_cost(num_children)
    life_cost = int(base_life * ((1 + inflation) ** idx))

    # 教育費
    edu_cost = education_cost_for_year(idx)

    # ローンは残り年数を過ぎたら 0 円
    annual_debt = debt_year if idx < loan_years else 0

    # 投資積立（年額）
    invest_annual = invest_month * 12

    # 特別支出（円）
    special_expense = int(special_expense_by_year.get(year, 0))

    # 純粋な「消費的支出」
    pure_expense = life_cost + annual_debt + repair + edu_cost + special_expense

    # 投資積立まで含めたキャッシュアウト
    annual_expense = pure_expense + invest_annual

    # 年間収支
    annual_balance = annual_income - annual_expense

    # 現金ベースの累積貯蓄
    cumulative += annual_balance

    # 投資残高：複利＋積立
    invest_balance = invest_balance * (1 + invest_return) + invest_annual

    # 累積貯蓄がマイナスになった場合は投資残高から補填（切り崩し）
    if cumulative < 0:
        deficit = -cumulative
        invest_balance -= deficit
        cumulative = 0

    # 総資産
    total_asset = cumulative + invest_balance

    row = {
        "年齢": age,
        "西暦": year,
        "配偶者年齢": spouse_age_year,
        "本人収入": person_income,
        "特別収入": special_income,
        "収入合計": annual_income,
        "生活費": life_cost,
        "住宅ローン": annual_debt,
        "管理費・修繕費": repair,
        "教育費": edu_cost,
        "投資積立額": invest_annual,
        "特別支出": special_expense,
        "支出合計": annual_expense,
        "年間収支": annual_balance,
        "累積貯蓄": cumulative,
        "投資残高": invest_balance,
        "総資産": total_asset,
    }

    if has_spouse:
        row["配偶者収入"] = spouse_income_year

    for j, cs in enumerate(child_settings):
        child_age_raw = year - cs["birth_year"]
        row[f"子{j+1}年齢"] = max(0, child_age_raw)

    records.append(row)

df = pd.DataFrame(records)

# ===============================
# キャッシュフロー表（横向き）
# ===============================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("キャッシュフロー表")

child_age_cols = [f"子{i+1}年齢" for i in range(num_children)]
base_cols = ["西暦", "年齢", "配偶者年齢"]

# 収入ブロック
income_cols = ["本人収入"] + (["配偶者収入"] if has_spouse else []) + ["特別収入", "収入合計"]

# 支出ブロック
expense_cols = [
    "生活費", "住宅ローン", "管理費・修繕費",
    "教育費", "投資積立額", "特別支出", "支出合計",
]

# 資産ブロック
asset_cols = ["年間収支", "累積貯蓄", "投資残高", "総資産"]

rest_cols = income_cols + expense_cols + asset_cols
show_cols = base_cols + child_age_cols + rest_cols

df_show = df[show_cols].copy()

# 配偶者なしの場合は「配偶者年齢」「配偶者収入」を削除
if not has_spouse:
    if "配偶者年齢" in df_show.columns:
        df_show = df_show.drop(columns=["配偶者年齢"])
    if "配偶者収入" in df_show.columns:
        df_show = df_show.drop(columns=["配偶者収入"])

numeric_cols = [c for c in df_show.columns if c not in ["配偶者年齢", "西暦"]]
df_show[numeric_cols] = df_show[numeric_cols].round(0).astype(int)

st.caption("※ 収入は手取りベース、金額の単位はすべて円です。投資積立額も支出に含めた後の年間収支・累積貯蓄を表示しています。")

def highlight_row(row):
    name = row.name
    INCOME_ROWS = ["本人収入", "配偶者収入", "特別収入", "収入合計"]
    EXPENSE_ROWS = ["生活費", "住宅ローン", "管理費・修繕費", "教育費", "投資積立額", "特別支出", "支出合計"]
    ASSET_ROWS = ["年間収支", "累積貯蓄", "投資残高", "総資産"]

    if name in INCOME_ROWS:
        style = "background-color: #D7EEFF;"
        if name == "収入合計":
            style += " font-weight: 800 !important"
        return [style] * len(row)

    if name in EXPENSE_ROWS:
        style = "background-color: #FFE4E1; font-weight: 700;"
        if name == "支出合計":
            style += " font-weight: 800 !important"
        return [style] * len(row)

    if name in ASSET_ROWS:
        style = "background-color: #E9FFE7;"
        if name == "総資産":
            style = "background-color: #DFF7DD; font-weight: 800 !important"
        return [style] * len(row)

    return [""] * len(row)

def color_negative(v):
    try:
        return "color: #E11D48; font-weight: 700" if float(v) < 0 else ""
    except Exception:
        return ""

def bold_key_rows(row):
    key_rows = {"収入合計", "支出合計", "総資産"}
    if row.name in key_rows:
        return ["font-weight: 800 !important"] * len(row)
    return [""] * len(row)

years_header = df_show["西暦"].astype(int).values
df_t = df_show.drop(columns=["西暦"]).T
df_t.columns = years_header

table_height = int((len(df_t.index) + 1) * 35 + 20)

styler = (
    df_t.style
        .apply(highlight_row, axis=1)
        .apply(bold_key_rows, axis=1)
        .applymap(color_negative)
        .format("{:,.0f}")
)

st.dataframe(styler, height=table_height, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# CSVダウンロード
# ===============================
csv_data = df_show.to_csv(index=False).encode("utf-8-sig")
st.download_button("CSVダウンロード", csv_data, "cashflow.csv", "text/csv")

# ===============================
# 資産・貯蓄・投資残高の推移グラフ（万円表示）
# ===============================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("資産・貯蓄・投資残高の推移")

fig, ax = plt.subplots(figsize=(10, 5))
yen_to_10k = 10_000

ax.plot(df["年齢"], df["累積貯蓄"] / yen_to_10k, label="累積貯蓄（現金）")
ax.plot(df["年齢"], df["投資残高"] / yen_to_10k, label="投資残高")
ax.plot(df["年齢"], df["総資産"] / yen_to_10k, label="総資産", linewidth=3)
ax.set_xlabel("年齢")
ax.set_ylabel("金額（万円）")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# モデルの前提説明（生活費・教育費）
# ===============================
st.markdown(
    """
<div class="section-card">

### 生活費・教育費モデルの前提

- **生活費**  
  - 総務省「家計調査」による勤労者世帯（2人以上）の消費支出データをベースに、都市部（大阪圏）の水準を参考にしたモデルです。  
  - 「ミニマム」：統計値のおおよそ80〜85％程度（かなり節約寄り）  
  - 「標準」　　：統計値に近い水準（一般的な暮らしイメージ）  
  - 「ゆとり」　：統計値の115〜120％程度（外食やレジャー多め）  
  - 子どもの人数（0〜3人）に応じて世帯の生活費が段階的に増える前提としています。

- **教育費（中学校〜高校）**  
  - 文部科学省「子供の学習費調査」などを参考に、  
    公立・私立別の年間学費＋塾・習い事費を目安としてモデル化しています。  

- **大学・専門学校の費用**  
  - 各種統計に基づく平均的な授業料・施設費等から、  
    「国公立大学」「私立大学」「専門学校」ごとの年間費用を設定。  
  - 入学初年度には入学金（30万円）を加算し、  
    「下宿あり」の場合は下宿費（年80万円）を上乗せしています。

※ 実際の家計・学校選択・進路によって金額は大きく変動します。  
　本シミュレーションはあくまで目安であり、詳細なライフプラン作成時には個別の金額での再試算をおすすめします。

</div>
""",
    unsafe_allow_html=True
)

# ===============================
# 右上固定ロゴ表示
# ===============================
logo_path = Path(__file__).parent / "logo_sh.png"

def load_logo_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

if logo_path.exists():
    logo_b64 = load_logo_base64(logo_path)
    st.markdown(
        f"""
        <style>
        .fixed-logo {{
            position: fixed;
            top: 70px;
            right: 40px;
            width: 100px;
            z-index: 9999;
            pointer-events: none;
        }}
        </style>
        <img src="data:image/png;base64,{logo_b64}" class="fixed-logo">
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("ロゴ画像（logo_sh.png）が見つかりませんでした。")
