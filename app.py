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

st.set_page_config(page_title="ライフプランシミュレーション", layout="wide")
st.title("ライフプランシミュレーション")

# ----------------------------
# 基本情報
# ----------------------------
st.sidebar.header("基本情報")
start_age = st.sidebar.number_input("現在の年齢", 20, 60, 40)
start_year = datetime.today().year
end_age = 65

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
    age_now = st.sidebar.number_input(
        f"子ども{i+1}の現在の年齢", 0, 25, 5, key=f"age_now_{i}"
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
            "age_now": int(age_now),      # 現在の年齢（今年）
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

# インフレ率
inflation = st.sidebar.slider("インフレ率（年%）", 0.0, 5.0, 1.0) / 100

# 賃金上昇率
wage_growth = st.sidebar.slider("賃金上昇率（年%）", 0.0, 5.0, 1.0) / 100

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
    """
    子どもの人数に応じて生活費の基準額を変える。
    子ども3人以上は3と同じ扱い。
    """
    if life_cost_preset == "手入力":
        return life_cost_custom

    n = int(num_children)
    if n < 0:
        n = 0
    if n > 3:
        n = 3

    base = LIFE_TABLE[n][life_cost_preset]
    return base

# ===============================
# 教育費モデル（すべて円）
# ===============================
PUBLIC_MH = 400_000   # 中高 公立（年額）
PRIVATE_MH = 900_000  # 中高 私立（年額）
UNI_COST = {
    "国公立大学": 800_000,
    "私立大学": 1_500_000,
    "専門学校": 1_100_000,
}
UNI_YEARS = {"国公立大学": 4, "私立大学": 4, "専門学校": 2}
ENTRANCE_FEE = 300_000  # 入学金（初年度）
DORM_COST = 800_000     # 下宿費（年額）

def education_cost_for_year(year_index: int) -> int:
    """
    year_index: 0 がシミュレーション開始年、その +1, +2 ... が以降の年
    - 中高の学費（公立/私立）は 12〜18歳で加算
    - 「塾・学費（月額）」はおおよそ 6〜22歳の間で毎年加算（今の年齢から反映）
    """
    total = 0
    for cs in child_settings:
        # 今年の年齢 = 現在の年齢 + 経過年数
        child_age = cs["age_now"] + year_index

        # 中学〜高校の学校費用（公立/私立）
        if 12 <= child_age <= 18:
            total += PRIVATE_MH if cs["school_type"] == "私立" else PUBLIC_MH

        # 小学校高学年〜大学くらいまでは塾・学費（月額）を反映
        if 6 <= child_age <= 22 and cs["cram_month"] > 0:
            total += cs["cram_month"] * 12

        # 大学・専門
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
years = list(range(start_year, start_year + (end_age - start_age) + 1))
records = []

cumulative = initial_savings     # 現在の貯蓄からスタート（現金・預金）
invest_balance = invest_principal
num_children = len(child_settings)

for idx, year in enumerate(years):
    age = start_age + idx
    if has_spouse and spouse_age is not None:
        spouse_age_year = spouse_age + idx  # 現在の年齢＋経過年数
    else:
        spouse_age_year = None

    # 収入（手取り）に賃金上昇率を適用
    annual_income = int(
        income * (1 + wage_growth) ** idx +
        spouse_income * (1 + wage_growth) ** idx
    )

    # 生活費（子どもの人数 & インフレ連動）
    base_life = get_life_cost(num_children)
    life_cost = int(base_life * ((1 + inflation) ** idx))

    # 教育費
    edu_cost = education_cost_for_year(idx)

    # ローンは残り年数を過ぎたら 0 円
    if idx < loan_years:
        annual_debt = debt_year
    else:
        annual_debt = 0

    # 投資積立（年額）
    invest_annual = invest_month * 12

    # 純粋な「消費的支出」の合計（生活費＋住宅＋管理修繕＋教育）
    pure_expense = life_cost + annual_debt + repair + edu_cost

    # 投資積立まで含めたキャッシュアウト（実際に口座から出ていくお金）
    annual_expense = pure_expense + invest_annual

    # 年間収支（投資積立も考慮した後の増減）
    annual_balance = annual_income - annual_expense

    # 現金ベースの累積貯蓄
    cumulative += annual_balance

    # 投資残高：昨年までの残高＋リターン＋今年の積立
    invest_balance = invest_balance * (1 + invest_return) + invest_annual

    # 総資産＝現金（累積貯蓄）＋投資残高
    total_asset = cumulative + invest_balance

    # この年のレコード用 dict を作成
    row = {
        "年齢": age,
        "西暦": year,
        "配偶者年齢": spouse_age_year,
        "収入（手取り）": annual_income,
        "生活費": life_cost,
        "住宅ローン": annual_debt,
        "管理費・修繕費": repair,
        "教育費": edu_cost,
        "投資積立額": invest_annual,
        "支出合計": annual_expense,       # 投資積立込みのキャッシュアウト
        "年間収支": annual_balance,
        "累積貯蓄": cumulative,
        "投資残高": invest_balance,
        "総資産": total_asset,
    }

    # 子どもごとの年齢を追加（子1年齢, 子2年齢, 子3年齢）
    for j, cs in enumerate(child_settings):
        row[f"子{j+1}年齢"] = cs["age_now"] + idx

    records.append(row)

df = pd.DataFrame(records)

# ===============================
# キャッシュフロー表（横向き）
# ===============================
st.header("キャッシュフロー表")

# 子ども年齢列を動的に生成
child_age_cols = [f"子{i+1}年齢" for i in range(num_children)]

# 「配偶者年齢」はいったん含めておき、あとで必要に応じて削除する
base_cols = ["西暦", "年齢", "配偶者年齢"]

rest_cols = [
    "収入（手取り）", "生活費", "住宅ローン", "管理費・修繕費",
    "教育費", "投資積立額", "支出合計", "年間収支",
    "累積貯蓄", "投資残高", "総資産",
]

show_cols = base_cols + child_age_cols + rest_cols

df_show = df[show_cols].copy()

# ★ 配偶者なしの場合は「配偶者年齢」列を完全に削除（表＆CSV両方から消える）
if not has_spouse and "配偶者年齢" in df_show.columns:
    df_show = df_show.drop(columns=["配偶者年齢"])

# 配偶者年齢 と 西暦 以外は数値として整形（子ども年齢も含めて整数化）
numeric_cols = [c for c in df_show.columns if c not in ["配偶者年齢", "西暦"]]
df_show[numeric_cols] = df_show[numeric_cols].round(0).astype(int)

st.caption("※ 収入は手取りベース、金額の単位はすべて円です。投資積立額も支出に含めた後の年間収支・累積貯蓄を表示しています。")

# 強調表示用
def highlight(row):
    name = row.name
    if name == "収入（手取り）":
        return ["background-color: #D7EEFF; font-weight: bold"] * len(row)
    if name in ["生活費", "住宅ローン", "管理費・修繕費", "教育費", "投資積立額", "支出合計"]:
        return ["background-color: #FFE4E1; font-weight: bold"] * len(row)
    if name in ["年間収支", "累積貯蓄", "投資残高", "総資産"]:
        return ["background-color: #E9FFE7; font-weight: bold"] * len(row)
    return [""] * len(row)

# ヘッダー用の西暦
years_header = df_show["西暦"].astype(int).values

# 表からは西暦列を削除（年齢から始まる）
df_body = df_show.drop(columns=["西暦"])

# 転置して 行＝項目、列＝西暦
df_t = df_body.T
df_t.columns = years_header  # 一番上のグレー行が西暦、1行目は「年齢」

st.dataframe(df_t.style.apply(highlight, axis=1))

# ===============================
# CSVダウンロード（CSVには西暦 & 子ども年齢 & 投資積立額も含める）
# ===============================
csv_data = df_show.to_csv(index=False).encode("utf-8-sig")
st.download_button("CSVダウンロード", csv_data, "cashflow.csv", "text/csv")

# ===============================
# 資産・貯蓄・投資残高の推移グラフ（万円表示）
# ===============================
st.header("資産・貯蓄・投資残高の推移")

fig, ax = plt.subplots(figsize=(10, 5))

yen_to_10k = 10_000  # 万円単位に変換

ax.plot(df["年齢"], df["累積貯蓄"] / yen_to_10k, label="累積貯蓄（現金）")
ax.plot(df["年齢"], df["投資残高"] / yen_to_10k, label="投資残高")
ax.plot(df["年齢"], df["総資産"] / yen_to_10k, label="総資産", linewidth=3)
ax.set_xlabel("年齢")
ax.set_ylabel("金額（万円）")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# ===============================
# モデルの前提説明（生活費・教育費）
# ===============================
st.markdown(
    """
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
"""
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
            top: 70px;     /* 上からの距離 */
            right: 40px;   /* 右からの距離 */
            width: 100px;  /* ロゴの大きさ */
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
