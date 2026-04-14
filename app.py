import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import hashlib

from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image, Paragraph
from reportlab.platypus.flowables import KeepInFrame
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import reportlab.pdfbase.pdfdoc as pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ========================
# reportlab の md5 パッチ（usedforsecurity 対策）
# ========================
def _patched_md5(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return hashlib.md5(*args, **kwargs)

pdfdoc.md5 = _patched_md5

# ========================
# 日本語フォント設定（画面 & PDF両方）
# ========================
BASE_DIR = Path(__file__).parent
font_path = BASE_DIR / "fonts" / "ipaexg.ttf"
logo_path = BASE_DIR / "logo_sh.png"

# 生活費プリセットを「統計値ベース + 10%」に上乗せする係数
LIFE_PRESET_UPLIFT = 1.10

# 生活費（消費支出）の内訳（目安）
LIFE_BREAKDOWN_RATIOS = {
    "食料": 0.24,
    "住居（家賃以外）": 0.06,
    "光熱・水道": 0.06,
    "家具・家事用品": 0.04,
    "被服及び履物": 0.04,
    "保健医療": 0.04,
    "交通・通信": 0.14,
    "教養娯楽": 0.12,
    "教育": 0.03,
    "その他": 0.23,
}

# Matplotlib（画面用）
fm.fontManager.addfont(str(font_path))
matplotlib.rcParams["font.family"] = "IPAexGothic"

# ReportLab（PDF用）
pdfmetrics.registerFont(TTFont("IPAexGothic", str(font_path)))

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

has_spouse = st.sidebar.checkbox("配偶者あり")
if has_spouse:
    spouse_age = st.sidebar.number_input("配偶者の現在の年齢", 20, 60, 40)
else:
    spouse_age = None

income = st.sidebar.number_input("本人の手取り年収（円）", 0, 30_000_000, 6_000_000)
if has_spouse:
    spouse_income = st.sidebar.number_input("配偶者の手取り年収（円）", 0, 30_000_000, 3_000_000)
else:
    spouse_income = 0

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

debt_month = st.sidebar.number_input("住宅ローン返済（月額・円）", 0, 2_000_000, 150_000)
debt_year = debt_month * 12
loan_years = st.sidebar.number_input("住宅ローンの残り返済年数（年）", 0, 40, 30)

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
initial_savings = st.sidebar.number_input("現在の貯蓄額（円）", 0, 300_000_000, 3_000_000)
invest_principal = st.sidebar.number_input("投資元本（円）", 0, 100_000_000, 1_000_000)
invest_month = st.sidebar.number_input("毎月積立額（円）", 0, 1_000_000, 30_000)
invest_return = st.sidebar.slider("利回り（年率%）", 0.0, 15.0, 3.0) / 100
savings_rate = st.sidebar.slider("普通預金金利（年率%）", 0.0, 2.0, 0.1, step=0.01) / 100

# ----------------------------
# 老後設定
# ----------------------------
st.sidebar.header("老後設定")

retirement_age = st.sidebar.number_input("退職年齢", 55, 70, 65)
end_age = st.sidebar.number_input("シミュレーション終了年齢", 70, 100, 90)

pension_start_age = st.sidebar.number_input("年金受給開始年齢", 60, 75, 65)
pension_amount = st.sidebar.number_input("年金受給額（年額・円）", 0, 10_000_000, 1_500_000)

if has_spouse:
    spouse_pension_amount = st.sidebar.number_input("配偶者の年金受給額（年額・円）", 0, 10_000_000, 700_000)
    spouse_retirement_age = st.sidebar.number_input("配偶者の退職年齢", 55, 70, 65)
else:
    spouse_pension_amount = 0
    spouse_retirement_age = 65

retirement_lump_sum = st.sidebar.number_input("退職金（円）", 0, 100_000_000, 0)

post_retirement_life_cost_input = st.sidebar.number_input(
    "老後の生活費（年額・円、0=現役時と同じ）", 0, 30_000_000, 0
)

years = list(range(start_year, start_year + (end_age - start_age) + 1))

# ----------------------------
# 特別収入・特別支出（万円単位）
# ----------------------------
st.sidebar.header("特別収入・特別支出（万円）")

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
        "西暦": st.column_config.NumberColumn("西暦", disabled=True, format="%d年", width="small"),
        "特別収入": st.column_config.NumberColumn("特別収入", min_value=0, step=10, format="%d", width="small"),
        "特別支出": st.column_config.NumberColumn("特別支出", min_value=0, step=10, format="%d", width="small"),
    },
    key="special_editor_sidebar",
)

special_income_by_year = {int(k): int(v) * 10_000 for k, v in zip(special_df["西暦"], special_df["特別収入"])}
special_expense_by_year = {int(k): int(v) * 10_000 for k, v in zip(special_df["西暦"], special_df["特別支出"])}

# ----------------------------
# 収入変化の設定（産休・転職等）
# ----------------------------
st.sidebar.header("収入変化の設定（産休・転職等）")

income_change_df = pd.DataFrame({
    "西暦": years,
    "本人収入倍率(%)": [100] * len(years),
    "配偶者収入倍率(%)": [100] * len(years),
})

income_change_df = st.sidebar.data_editor(
    income_change_df,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={
        "西暦": st.column_config.NumberColumn("西暦", disabled=True, format="%d年", width="small"),
        "本人収入倍率(%)": st.column_config.NumberColumn("本人(%)", min_value=0, max_value=300, step=10, format="%d", width="small"),
        "配偶者収入倍率(%)": st.column_config.NumberColumn("配偶者(%)", min_value=0, max_value=300, step=10, format="%d", width="small"),
    },
    key="income_change_editor",
)

person_multiplier_by_year = {
    int(k): float(v) / 100.0
    for k, v in zip(income_change_df["西暦"], income_change_df["本人収入倍率(%)"])
}
spouse_multiplier_by_year = {
    int(k): float(v) / 100.0
    for k, v in zip(income_change_df["西暦"], income_change_df["配偶者収入倍率(%)"])
}

# ===============================
# 生活費プリセット
# ===============================
LIFE_TABLE = {
    0: {"ミニマム": 2_200_000, "標準": 2_800_000, "ゆとり": 3_400_000},
    1: {"ミニマム": 2_600_000, "標準": 3_200_000, "ゆとり": 3_800_000},
    2: {"ミニマム": 3_000_000, "標準": 3_600_000, "ゆとり": 4_200_000},
    3: {"ミニマム": 3_400_000, "標準": 4_000_000, "ゆとり": 4_600_000},
}

def get_life_cost_raw(num_children: int) -> int:
    n = max(0, min(3, int(num_children)))
    return int(LIFE_TABLE[n][life_cost_preset])

def get_life_cost(num_children: int) -> int:
    if life_cost_preset == "手入力":
        return int(life_cost_custom)
    base = get_life_cost_raw(num_children)
    return int(round(base * LIFE_PRESET_UPLIFT))

def build_life_breakdown(annual_life_cost: int) -> pd.DataFrame:
    rows = []
    for k, ratio in LIFE_BREAKDOWN_RATIOS.items():
        yen = int(round(annual_life_cost * ratio))
        rows.append({"費目": k, "年額（円）": yen, "月額（円）": int(round(yen / 12))})
    df_bd = pd.DataFrame(rows)
    df_bd.loc[len(df_bd)] = {
        "費目": "合計",
        "年額（円）": int(df_bd["年額（円）"].sum()),
        "月額（円）": int(round(df_bd["年額（円）"].sum() / 12)),
    }
    return df_bd

# サイドバー: 生活費プリセットの内訳表示
if life_cost_preset != "手入力":
    with st.sidebar.expander("生活費プリセットの内訳（目安）"):
        raw = get_life_cost_raw(len(child_settings))
        uplifted = int(round(raw * LIFE_PRESET_UPLIFT))
        st.caption(
            "※ 統計ベース（年額）に対して、このアプリでは+10%上乗せをデフォルトにしています。"
        )
        st.write(f"統計ベース: {raw:,.0f}円 / 年 → 上乗せ後: {uplifted:,.0f}円 / 年")
        bd = build_life_breakdown(uplifted)
        st.dataframe(
            bd.style.format({"年額（円）": "{:,.0f}", "月額（円）": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

# ===============================
# 教育費モデル
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

# 退職金フラグ
lump_sum_year = start_year + (retirement_age - start_age)

# 老後生活費
if post_retirement_life_cost_input > 0:
    post_retirement_life_cost_base = post_retirement_life_cost_input
else:
    post_retirement_life_cost_base = None  # 現役時と同じ

for idx, year in enumerate(years):
    age = start_age + idx
    spouse_age_year = (spouse_age + idx) if (has_spouse and spouse_age is not None) else None

    is_retired = age >= retirement_age
    is_spouse_retired = has_spouse and (spouse_age is not None) and ((spouse_age + idx) >= spouse_retirement_age)

    # 現役/老後で収入を切り替え
    if is_retired:
        # 退職後: 年金収入のみ
        pension_income = pension_amount if age >= pension_start_age else 0
        person_income = int(pension_income)
    else:
        base_person_income = int(income * (1 + wage_growth) ** idx)
        multiplier = person_multiplier_by_year.get(year, 1.0)
        person_income = int(base_person_income * multiplier)

    if has_spouse:
        if is_spouse_retired:
            spouse_pension_age = spouse_age + idx if spouse_age is not None else 0
            sp_pension = spouse_pension_amount if spouse_pension_age >= pension_start_age else 0
            spouse_income_year = int(sp_pension)
        else:
            base_spouse_income = int(spouse_income * (1 + wage_growth) ** idx)
            sp_multiplier = spouse_multiplier_by_year.get(year, 1.0)
            spouse_income_year = int(base_spouse_income * sp_multiplier)
    else:
        spouse_income_year = 0

    # 退職金（退職年に一括）
    lump_sum_this_year = retirement_lump_sum if (year == lump_sum_year and retirement_lump_sum > 0) else 0

    special_income = int(special_income_by_year.get(year, 0))
    annual_income = person_income + spouse_income_year + special_income + lump_sum_this_year

    # 生活費（老後は別設定可能）
    if is_retired and post_retirement_life_cost_base is not None:
        base_life = post_retirement_life_cost_base
    else:
        base_life = get_life_cost(num_children)
    life_cost = int(base_life * ((1 + inflation) ** idx))

    edu_cost = education_cost_for_year(idx)
    annual_debt = debt_year if idx < loan_years else 0
    invest_annual = invest_month * 12
    special_expense = int(special_expense_by_year.get(year, 0))

    pure_expense = life_cost + annual_debt + repair + edu_cost + special_expense
    annual_expense = pure_expense + invest_annual

    annual_balance = annual_income - annual_expense

    # 普通預金金利を貯蓄残高に適用
    cumulative = cumulative * (1 + savings_rate) + annual_balance
    invest_balance = invest_balance * (1 + invest_return) + invest_annual

    if cumulative < 0:
        deficit = -cumulative
        invest_balance -= deficit
        cumulative = 0

    total_asset = cumulative + invest_balance

    row = {
        "年齢": age,
        "西暦": year,
        "配偶者年齢": spouse_age_year,
        "本人収入": person_income,
        "特別収入": special_income + lump_sum_this_year,
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
# キャッシュフロー表の列整理
# ===============================
child_age_cols = [f"子{i+1}年齢" for i in range(num_children)]
base_cols = ["西暦", "年齢", "配偶者年齢"]

income_cols = ["本人収入"] + (["配偶者収入"] if has_spouse else []) + ["特別収入", "収入合計"]
expense_cols = [
    "生活費", "住宅ローン", "管理費・修繕費",
    "教育費", "投資積立額", "特別支出", "支出合計",
]
asset_cols = ["年間収支", "累積貯蓄", "投資残高", "総資産"]

rest_cols = income_cols + expense_cols + asset_cols
show_cols = base_cols + child_age_cols + rest_cols

df_show = df[show_cols].copy()

if not has_spouse:
    if "配偶者年齢" in df_show.columns:
        df_show = df_show.drop(columns=["配偶者年齢"])
    if "配偶者収入" in df_show.columns:
        df_show = df_show.drop(columns=["配偶者収入"])

numeric_cols = [c for c in df_show.columns if c not in ["配偶者年齢", "西暦"]]
df_show[numeric_cols] = df_show[numeric_cols].round(0).astype(int)

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

# ===============================
# サマリーメトリクス
# ===============================
final_total_asset = int(df["総資産"].iloc[-1])
min_cumulative = int(df["累積貯蓄"].min())
min_cumulative_idx = df["累積貯蓄"].idxmin()
min_cumulative_age = int(df.loc[min_cumulative_idx, "年齢"])
min_cumulative_year = int(df.loc[min_cumulative_idx, "西暦"])

# 65歳時の総資産
age65_rows = df[df["年齢"] == 65]
if len(age65_rows) > 0:
    asset_at_65 = int(age65_rows["総資産"].iloc[0])
else:
    asset_at_65 = None

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("最終年の総資産", f"{final_total_asset // 10_000:,} 万円")
with col_m2:
    st.metric(
        "最低累積貯蓄",
        f"{min_cumulative // 10_000:,} 万円",
        delta=f"{min_cumulative_age}歳（{min_cumulative_year}年）",
        delta_color="off",
    )
with col_m3:
    if asset_at_65 is not None:
        st.metric("65歳時の総資産", f"{asset_at_65 // 10_000:,} 万円")
    else:
        st.metric("65歳時の総資産", "対象外")

# 貯蓄マイナス警告
if (df["累積貯蓄"] < 0).any() or min_cumulative < 0:
    neg_years = df[df["累積貯蓄"] < 0]["年齢"].tolist()
    if neg_years:
        st.warning(f"累積貯蓄がマイナスになる年があります（{neg_years[0]}歳〜）。収入・支出の見直しをご検討ください。")
    else:
        st.warning("累積貯蓄が一時的にマイナスになる可能性があります。")

# ===============================
# タブ構成
# ===============================
tab_cf, tab_graph, tab_premise = st.tabs(["キャッシュフロー表", "グラフ", "モデルの前提"])

# ===============================
# PDF生成関数（A3横・1枚固定）
# ===============================
def create_cashflow_pdf(df_show: pd.DataFrame, sidebar_inputs: dict, logo_path: Path):
    buffer = BytesIO()
    W, H = landscape(A3)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A3),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )
    avail_w = W - doc.leftMargin - doc.rightMargin
    styles = getSampleStyleSheet()

    df_y = df_show.copy()
    if "西暦" not in df_y.columns:
        raise ValueError("df_show に '西暦' 列が必要です。")

    df_y = df_y.rename(columns={"西暦": "年", "年齢": "本人年齢"})
    child_cols = [c for c in df_y.columns if c.startswith("子") and c.endswith("年齢")]

    spouse_col = "配偶者年齢" if "配偶者年齢" in df_y.columns else None

    if "住宅ローン" in df_y.columns and "管理費・修繕費" in df_y.columns:
        df_y["住宅費"] = df_y["住宅ローン"] + df_y["管理費・修繕費"]
    else:
        df_y["住宅費"] = 0

    money_cols = ["収入合計", "生活費", "住宅費", "教育費", "投資積立額", "特別支出", "年間収支", "累積貯蓄", "投資残高", "総資産"]
    cols = ["年", "本人年齢"] + ([spouse_col] if spouse_col else []) + child_cols + [c for c in money_cols if c in df_y.columns]
    out = df_y[cols].copy()

    YEN_TO_10K = 10_000
    for c in out.columns:
        if c in ("年", "本人年齢", "配偶者年齢") or c.endswith("年齢"):
            continue
        out[c] = (pd.to_numeric(out[c], errors="coerce") / YEN_TO_10K).round(0).fillna(0).astype(int)

    n_years = len(out)
    n_children = len(child_cols)

    child_penalty = 0.0
    if n_children >= 3:
        child_penalty = 0.6
    elif n_children == 2:
        child_penalty = 0.3
    elif n_children == 0:
        child_penalty = -0.2

    if n_years <= 22:
        tbl_header_fs, tbl_body_fs = 10.6 - child_penalty, 10.1 - child_penalty
        cell_pad = 5.2
        sb_fs, sb_head_fs = 10.2, 11.2
        sidebar_w = 90 * mm
    elif n_years <= 32:
        tbl_header_fs, tbl_body_fs = 9.8 - child_penalty, 9.4 - child_penalty
        cell_pad = 4.6
        sb_fs, sb_head_fs = 9.8, 10.8
        sidebar_w = 86 * mm
    else:
        tbl_header_fs, tbl_body_fs = 9.2 - child_penalty, 8.8 - child_penalty
        cell_pad = 3.9
        sb_fs, sb_head_fs = 9.6, 10.4
        sidebar_w = 82 * mm

    gap = 6 * mm
    table_w = avail_w - sidebar_w - gap

    sb_style = ParagraphStyle(
        "sb",
        parent=styles["Normal"],
        fontName="IPAexGothic",
        fontSize=sb_fs,
        leading=sb_fs + 2,
        spaceAfter=2,
    )
    sb_head = ParagraphStyle(
        "sb_head",
        parent=styles["Normal"],
        fontName="IPAexGothic",
        fontSize=sb_head_fs,
        leading=sb_head_fs + 2,
        spaceAfter=2,
    )

    def P(txt: str, head: bool = False):
        return Paragraph(txt, sb_head if head else sb_style)

    sidebar_blocks = []
    for section, payload in sidebar_inputs.items():
        sidebar_blocks.append(P(f"<b>{section}</b>", head=True))
        if isinstance(payload, (list, tuple)):
            if len(payload) == 0:
                sidebar_blocks.append(P("なし"))
            else:
                for line in payload:
                    sidebar_blocks.append(P(str(line)))
        elif isinstance(payload, dict):
            if len(payload) == 0:
                sidebar_blocks.append(P("なし"))
            else:
                for k, v in payload.items():
                    sidebar_blocks.append(P(f"{k}：{v}"))
        else:
            sidebar_blocks.append(P(str(payload)))
        sidebar_blocks.append(Spacer(1, 2 * mm))

    sidebar_frame = KeepInFrame(sidebar_w, 9999, sidebar_blocks, mode="shrink", hAlign="LEFT", vAlign="TOP")
    sidebar_box = Table([[sidebar_frame]], colWidths=[sidebar_w])
    sidebar_box.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    header = out.columns.tolist()
    data = [header]
    for _, r in out.iterrows():
        row = []
        for c in header:
            v = r[c]
            if c in ("年", "本人年齢"):
                row.append(str(int(v)))
            elif c.endswith("年齢"):
                row.append("" if pd.isna(v) else str(v))
            else:
                row.append(f"{int(v):,}")
        data.append(row)

    fixed_map = {"年": 14 * mm, "本人年齢": 13 * mm}
    if spouse_col:
        fixed_map[spouse_col] = 13 * mm
    for cc in child_cols:
        fixed_map[cc] = 12 * mm

    fixed_sum = sum(fixed_map.get(c, 0) for c in header if c in fixed_map)
    flex_cols = [c for c in header if c not in fixed_map]
    flex_w = (table_w - fixed_sum) / max(len(flex_cols), 1)
    col_widths = [fixed_map.get(c, flex_w) for c in header]

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    ts = TableStyle([
        ("FONT", (0, 0), (-1, -1), "IPAexGothic"),
        ("FONTSIZE", (0, 0), (-1, 0), tbl_header_fs),
        ("FONTSIZE", (0, 1), (-1, -1), tbl_body_fs),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94A3B8")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1220")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (1 + (1 if spouse_col else 0) + len(child_cols), -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), cell_pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), cell_pad),
    ])

    def col_ix(name: str) -> int:
        return header.index(name)

    if "収入合計" in header:
        ts.add("BACKGROUND", (col_ix("収入合計"), 1), (col_ix("収入合計"), -1), colors.HexColor("#D7EEFF"))
    for c in ("生活費", "住宅費", "教育費", "投資積立額", "特別支出"):
        if c in header:
            ts.add("BACKGROUND", (col_ix(c), 1), (col_ix(c), -1), colors.HexColor("#FFEFEF"))
    for c in ("年間収支", "累積貯蓄", "投資残高", "総資産"):
        if c in header:
            ts.add("BACKGROUND", (col_ix(c), 1), (col_ix(c), -1), colors.HexColor("#E8F7E6"))

    for i in range(1, len(data)):
        if i % 2 == 0:
            ts.add("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC"))

    tbl.setStyle(ts)

    main_row = Table([[tbl, "", sidebar_box]], colWidths=[table_w, gap, sidebar_w])
    main_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    def draw_header(canvas, _doc):
        canvas.saveState()
        canvas.setFont("IPAexGothic", 20)
        canvas.drawString(doc.leftMargin, H - 14 * mm, "ライフプランシミュレーション")
        canvas.setFont("IPAexGothic", 10.5)
        canvas.drawString(doc.leftMargin, H - 21 * mm, f"金額：万円（端数四捨五入） / {n_years}年")

        try:
            if logo_path is not None and Path(logo_path).exists():
                ir = ImageReader(str(logo_path))
                iw, ih = ir.getSize()
                box_w, box_h = 58 * mm, 16 * mm
                scale = min(box_w / iw, box_h / ih)
                dw, dh = iw * scale, ih * scale
                x = W - doc.rightMargin - dw
                y = H - 12 * mm - dh
                canvas.drawImage(ir, x, y, width=dw, height=dh, mask="auto")
        except Exception:
            pass

        canvas.restoreState()

    header_space = 26 * mm
    avail_h = H - doc.topMargin - doc.bottomMargin - header_space

    k = KeepInFrame(avail_w, avail_h, [main_row], mode="shrink", hAlign="LEFT", vAlign="TOP")

    elements = [Spacer(1, header_space), k]
    doc.build(elements, onFirstPage=draw_header)

    buffer.seek(0)
    return buffer

# ===============================
# PDF生成（手動トリガー）
# ===============================
def make_df_key(df: pd.DataFrame) -> str:
    core = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    meta = (str(list(df.index)) + str(list(df.columns))).encode("utf-8")
    return hashlib.sha256(meta + core).hexdigest()

def _yen(v: int) -> str:
    return f"{int(v):,}円"

def _pct(v: float) -> str:
    return f"{v*100:.1f}%"

# 教育費の表現
edu_lines = []
for i, cs in enumerate(child_settings, start=1):
    edu_lines.append(
        f"子ども{i}：誕生年 {cs['birth_year']} / 中高 {cs['school_type']} / 進学先 {cs['uni']} / 下宿 {'あり' if cs['dorm'] else 'なし'} / 塾 {int(cs['cram_month']):,}円/月"
    )

special_income_total = int(sum(special_income_by_year.values()))
special_expense_total = int(sum(special_expense_by_year.values()))

base_life_annual = get_life_cost(num_children)
life_month_for_pdf = int(round(base_life_annual / 12))

sidebar_inputs = {
    "基本情報": [
        f"現在の年齢：{start_age}歳",
        f"退職年齢：{retirement_age}歳",
        f"シミュレーション終了年齢：{end_age}歳",
        f"配偶者：{'あり' if has_spouse else 'なし'}",
        (f"配偶者年齢：{spouse_age}歳" if has_spouse and spouse_age is not None else "配偶者年齢：-"),
        f"世帯手取り年収：{_yen(income + (spouse_income if has_spouse else 0))}",
        f"賃金上昇率：{_pct(wage_growth)}",
        f"インフレ率：{_pct(inflation)}",
    ],
    "老後設定": [
        f"年金受給開始：{pension_start_age}歳",
        f"本人年金：{_yen(pension_amount)}/年",
        *(
            [f"配偶者年金：{_yen(spouse_pension_amount)}/年",
             f"配偶者退職年齢：{spouse_retirement_age}歳"]
            if has_spouse else []
        ),
        f"退職金：{_yen(retirement_lump_sum)}",
        f"老後生活費：{'現役時と同じ' if post_retirement_life_cost_input == 0 else _yen(post_retirement_life_cost_input) + '/年'}",
    ],
    "投資・貯蓄": [
        f"現在の貯蓄：{_yen(initial_savings)}",
        f"投資元本：{_yen(invest_principal)}",
        f"毎月積立：{_yen(invest_month)}/月",
        f"想定利回り：{_pct(invest_return)}",
        f"普通預金金利：{_pct(savings_rate)}",
    ],
    "支出関連": [
        f"生活費：{_yen(life_month_for_pdf)}/月",
        f"住宅ローン：{_yen(debt_month)}/月（残 {loan_years}年）",
        f"管理費・修繕費：{_yen(repair_month)}/月",
    ],
    "教育費": edu_lines if len(edu_lines) > 0 else ["子ども：なし"],
    "特別収入・特別支出": (
        ["なし"] if (special_income_total == 0 and special_expense_total == 0)
        else [f"特別収入 合計：{_yen(special_income_total)}", f"特別支出 合計：{_yen(special_expense_total)}"]
    ),
}

current_key = make_df_key(df_show)

if st.session_state.get("pdf_key") != current_key:
    st.session_state["pdf_key"] = current_key
    st.session_state["pdf_bytes"] = None

# ===============================
# タブ1: キャッシュフロー表
# ===============================
with tab_cf:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("キャッシュフロー表")
    st.caption("※ 収入は手取りベース、金額の単位はすべて円です。投資積立額も支出に含めた後の年間収支・累積貯蓄を表示しています。")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("PDFを生成する", key="btn_make_pdf"):
            with st.spinner("PDFを作成中…"):
                st.session_state["pdf_bytes"] = create_cashflow_pdf(df_show, sidebar_inputs, logo_path).getvalue()
    with col_b:
        if st.button("PDFをクリア", key="btn_clear_pdf"):
            st.session_state["pdf_bytes"] = None
    with col_c:
        if st.session_state.get("pdf_bytes"):
            st.download_button(
                label="PDFをダウンロード",
                data=st.session_state["pdf_bytes"],
                file_name="cashflow_a3_onepage.pdf",
                mime="application/pdf",
                key="btn_download_pdf",
            )
        else:
            st.caption("※ 先に「PDFを生成する」を押してください。")

    # 表表示
    styler = (
        df_t.style
            .apply(highlight_row, axis=1)
            .apply(bold_key_rows, axis=1)
            .map(color_negative)
            .format("{:,.0f}")
    )
    st.dataframe(styler, height=table_height, use_container_width=True)

    # CSV ダウンロード
    csv_buffer = BytesIO()
    df_t.to_csv(csv_buffer, encoding="utf-8-sig")
    st.download_button(
        label="CSVをダウンロード",
        data=csv_buffer.getvalue(),
        file_name="cashflow_table.csv",
        mime="text/csv",
        key="btn_download_csv",
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# タブ2: グラフ
# ===============================
with tab_graph:
    # --- グラフ1: 資産推移 ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("資産・貯蓄・投資残高の推移")

    fig, ax = plt.subplots(figsize=(10, 5))
    yen_to_10k = 10_000

    ax.plot(df["年齢"], df["累積貯蓄"] / yen_to_10k, label="累積貯蓄（現金）")
    ax.plot(df["年齢"], df["投資残高"] / yen_to_10k, label="投資残高")
    ax.plot(df["年齢"], df["総資産"] / yen_to_10k, label="総資産", linewidth=3)

    # 退職年齢に縦線
    if start_age <= retirement_age <= end_age:
        ax.axvline(x=retirement_age, color="gray", linestyle="--", alpha=0.7, label=f"退職（{retirement_age}歳）")

    ax.set_xlabel("年齢")
    ax.set_ylabel("金額（万円）")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- グラフ2: 収支バーチャート ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("年間収入・支出・収支の推移")

    ages = df["年齢"].values
    income_10k = df["収入合計"].values / yen_to_10k
    expense_10k = df["支出合計"].values / yen_to_10k
    balance_10k = df["年間収支"].values / yen_to_10k

    x = np.arange(len(ages))
    bar_width = 0.35

    fig2, ax2 = plt.subplots(figsize=(max(10, len(ages) * 0.4), 5))
    ax2_right = ax2.twinx()

    bars1 = ax2.bar(x - bar_width / 2, income_10k, bar_width, label="収入合計", color="#4A90D9", alpha=0.85)
    bars2 = ax2.bar(x + bar_width / 2, expense_10k, bar_width, label="支出合計", color="#E05C5C", alpha=0.85)
    line1, = ax2_right.plot(x, balance_10k, color="#2DBD78", linewidth=2, marker="o", markersize=3, label="年間収支")

    ax2.set_xlabel("年齢")
    ax2.set_ylabel("金額（万円）")
    ax2_right.set_ylabel("年間収支（万円）")
    ax2_right.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # x軸ラベルは5年おきに表示（多い場合）
    step = max(1, len(ages) // 20)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(ages[::step])

    ax2.grid(True, linestyle="--", alpha=0.3)

    lines_labels = [ax2.get_legend_handles_labels(), ax2_right.get_legend_handles_labels()]
    handles = lines_labels[0][0] + lines_labels[1][0]
    labels = lines_labels[0][1] + lines_labels[1][1]
    ax2.legend(handles, labels, loc="upper left")

    fig2.tight_layout()
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# タブ3: モデルの前提
# ===============================
with tab_premise:
    st.markdown(
        """
    <div class="section-card">

    ### 生活費・教育費モデルの前提

    - **生活費**
      - 総務省「家計調査」による勤労者世帯（2人以上）の消費支出データをベースに、都市部（大阪圏）の水準を参考にしたモデルです。
      - 体感との差が出にくいように、**このアプリでは統計ベースに対して +10% 上乗せした金額**をプリセットの起点にしています。
      - 「ミニマム」：節約寄り / 「標準」：一般的 / 「ゆとり」：外食・レジャー多め
      - 子どもの人数（0〜3人）に応じて世帯の生活費が段階的に増える前提としています。
      - 参考: 費目別の目安内訳（割合モデル）をサイドバー内「生活費プリセットの内訳（目安）」に表示しています。

    - **教育費（中学校〜高校）**
      - 文部科学省「子供の学習費調査」などを参考に、
        公立・私立別の年間学費＋塾・習い事費を目安としてモデル化しています。

    - **大学・専門学校の費用**
      - 各種統計に基づく平均的な授業料・施設費等から、
        「国公立大学」「私立大学」「専門学校」ごとの年間費用を設定。
      - 入学初年度には入学金（30万円）を加算し、
        「下宿あり」の場合は下宿費（年80万円）を上乗せしています。

    - **老後シミュレーション**
      - 退職後は本人収入が年金収入に切り替わります。
      - 退職金は退職年に一括で特別収入として計上されます。
      - 老後の生活費は別途設定可能です（未設定の場合は現役時と同じ生活費が適用されます）。
      - 普通預金金利は累積貯蓄残高に毎年適用されます。

    ※ 実際の家計・学校選択・進路によって金額は大きく変動します。
    　本シミュレーションはあくまで目安であり、詳細なライフプラン作成時には個別の金額での再試算をおすすめします。

    </div>
    """,
        unsafe_allow_html=True
    )

# ===============================
# 右上固定ロゴ表示
# ===============================
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
