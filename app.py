import json
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page setup
st.set_page_config(
    page_title="Bengaluru Home Price",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# light CSS polish
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2.0rem; max-width: 1300px;}
      h1,h2,h3 { line-height: 1.15; }
      .stMetric { text-wrap: balance; }
      .stPlotlyChart { background: transparent !important; }
      .small-note { opacity: 0.75; font-size: 0.9rem; }
      .nowrap { white-space: nowrap; }
    </style>
    """,
    unsafe_allow_html=True
)

# Utilities
def theme_template():
    base = st.get_option("theme.base")
    return "plotly_dark" if base == "dark" else "plotly"

def format_price_lakh_crore(lakh_value: float) -> str:
    if lakh_value is None or pd.isna(lakh_value):
        return "N/A"
    return f"₹ {lakh_value/100:,.2f} Cr" if lakh_value >= 100 else f"₹ {lakh_value:,.2f} Lakhs"

def model_kind(model):
    return type(model).__name__

def get_linear_location_coefs(model, data_columns, top_k=12):
    if not hasattr(model, "coef_"):
        return None, None
    coefs = np.array(model.coef_).ravel()
    loc_names = data_columns[3:]
    loc_coefs = coefs[3:] if len(coefs) >= len(data_columns) else None
    if loc_coefs is None or len(loc_coefs) != len(loc_names):
        return None, None
    s = pd.Series(loc_coefs, index=[ln.title() for ln in loc_names])
    return s.sort_values(ascending=False).head(top_k), s.sort_values(ascending=True).head(top_k)

def predict_price(model, data_columns, location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1
    x = np.zeros(len(data_columns))
    # expected: [total_sqft, bath, bhk, ...locations]
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1.0
    return float(getattr(model, "predict")([x])[0])  # in Lakhs

def rebuild_location_if_needed(df: pd.DataFrame, data_columns: list) -> pd.DataFrame:
    df = df.copy()
    if "location" in df.columns:
        return df
    base_cols = {"total_sqft", "bath", "bhk", "price"}
    onehot_candidates = [c for c in df.columns if c not in base_cols]
    if onehot_candidates:
        # If value is 1 for the true location; idxmax picks the column with max value
        df["location"] = df[onehot_candidates].idxmax(axis=1)
        return df
    # fallback best-effort using columns.json
    loc_cols = [c for c in data_columns[3:] if c in df.columns]
    if loc_cols:
        df["location"] = df[loc_cols].idxmax(axis=1)
        return df
    raise ValueError("Cannot reconstruct 'location'. Provide pre-encoding data or one-hot columns.")

def add_ppsf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # price is in Lakhs
    df["price_per_sqft_inr"] = (df["price"] * 100000) / df["total_sqft"].replace(0, np.nan)
    df["price_per_sqft_inr"] = df["price_per_sqft_inr"].replace([np.inf, -np.inf], np.nan)
    return df

def build_feature_matrix(df_in: pd.DataFrame, data_columns: list) -> np.ndarray:
    X = np.zeros((len(df_in), len(data_columns)))
    X[:, 0] = df_in["total_sqft"].astype(float)
    X[:, 1] = df_in["bath"].astype(float)
    X[:, 2] = df_in["bhk"].astype(float)

    if "location" in df_in.columns:
        for i, loc in enumerate(df_in["location"].astype(str).str.lower().values):
            try:
                j = data_columns.index(loc)
                X[i, j] = 1.0
            except ValueError:
                pass
    else:
        for j, col in enumerate(data_columns[3:], start=3):
            if col in df_in.columns:
                X[:, j] = df_in[col].values
    return X

@st.cache_resource(show_spinner=False)
def load_model_cols(model_path: str, cols_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path, "r") as f:
        data_columns = json.load(f)["data_columns"]
    return model, data_columns

@st.cache_data(show_spinner=False)
def load_default_csv(csv_path: str, data_columns: list, cache_buster: int = 0):
    df = pd.read_csv(csv_path)
    df = rebuild_location_if_needed(df, data_columns)
    df = add_ppsf(df)
    return df

# Sidebar
with st.sidebar:
    st.header("Settings")
    disable_cache = st.toggle("Disable cache (for debugging)", value=False)
    show_outliers = st.toggle("Show outliers in boxplots", value=False)
    top_n = st.slider("Top N locations (charts)", 5, 30, 15, 1)
    st.divider()

    st.subheader("Upload CSV (optional)")
    st.caption("Use your own cleaned dataset for EDA & charts.")
    uploaded_csv = st.file_uploader("CSV with columns: total_sqft, bath, bhk, price, location or one-hot", type=["csv"])

    st.divider()
    st.subheader("Model Files in use")
    st.code("banglore_home_prices_model.pickle\ncolumns.json\nbangalore_home_prices_cleaned.csv", language="bash")

# Load model/columns
t0 = time.perf_counter()
try:
    model, data_columns = load_model_cols("banglore_home_prices_model.pickle", "columns.json")
except Exception as e:
    st.error(f"Failed to load model/columns: {e}")
    st.stop()

# Load dataset (default or uploaded)
try:
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        df = rebuild_location_if_needed(df, data_columns)
        df = add_ppsf(df)
        data_source = "Uploaded CSV"
    else:
        # cache_buster toggles caching off/on
        cache_buster = 0 if not disable_cache else int(time.time())
        df = load_default_csv("bangalore_home_prices_cleaned.csv", data_columns, cache_buster)
        data_source = "Default CSV"
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

template = theme_template()
load_latency_ms = (time.perf_counter() - t0) * 1000

# Header
st.title("🏙️ Bengaluru Home Price")
st.caption(f"Data source: {data_source} • Load time: {load_latency_ms:.0f} ms")

# Tabs
tab_pred, tab_dist, tab_eda, tab_sens, tab_batch, tab_model, tab_report = st.tabs(
    ["🔮 Prediction", "📊 Price Distribution", "📈 EDA", "🧪 Sensitivity", "📦 Batch Predict", "🧠 Model Insights", "📄 Report (PDF)"]
)

# 🔮 Prediction
with tab_pred:
    st.subheader("Enter House Details")
    with st.form("predict_form", clear_on_submit=False):
        sqft = st.number_input("Total Square Feet", min_value=250, max_value=20000, value=1100, step=50, help="Built-up area")
        bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=12, value=2, step=1)
        bath = st.number_input("Number of Bathrooms", min_value=1, max_value=12, value=2, step=1)
        unique_locations = sorted(df["location"].dropna().unique().tolist())
        location = st.selectbox("Location", unique_locations, index=min(10, len(unique_locations)-1))
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        issues = []
        if bath > bhk + 3: issues.append("Bathrooms seem unusually high for the chosen BHK.")
        if sqft / max(bhk, 1) < 200: issues.append("Sqft per BHK is quite low; please verify area.")
        for msg in issues: st.warning(msg)

        t1 = time.perf_counter()
        pred_lakh = predict_price(model, data_columns, location, sqft, bath, bhk)
        pred_ms = (time.perf_counter() - t1) * 1000

        c1, c2 = st.columns(2)
        c1.success(f"Estimated Price: **{format_price_lakh_crore(pred_lakh)}**")
        c2.caption(f"Inference time: {pred_ms:.1f} ms")

        # Scenario: Price vs Sqft for same BHK/Bath/Location
        sizes = np.linspace(max(300, sqft*0.6), min(4000, sqft*1.8), 24).astype(int)
        preds = [predict_price(model, data_columns, location, int(s), bath, bhk) for s in sizes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=preds, mode="lines+markers", name="Predicted Price (Lakhs)"))
        fig.update_layout(
            template=template,
            title=f"Scenario: Price vs Sqft • {location}",
            xaxis_title="Square Feet", yaxis_title="Price (Lakhs)",
            height=420, margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

# 📊 Price Distribution
with tab_dist:
    st.subheader("Average Price & PPSF by Location")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Records", f"{len(df):,}")
    colB.metric("Unique Locations", f"{df['location'].nunique():,}")
    colC.metric("Avg Price", format_price_lakh_crore(df['price'].mean()))
    colD.metric("Median PPSF (₹/sqft)", f"{int(df['price_per_sqft_inr'].median()):,}")

    metric_choice = st.radio("Metric", ["Average Price (Lakhs)", "Median PPSF (₹/sqft)"], horizontal=True)
    selected_locs = st.multiselect("Focus locations (optional)", sorted(df["location"].unique()), max_selections=8)

    df_use = df.copy()
    if selected_locs:
        df_use = df_use[df_use["location"].isin(selected_locs)]

    if metric_choice == "Average Price (Lakhs)":
        agg = df_use.groupby("location")["price"].mean().sort_values(ascending=False).head(top_n)
        y_title = "Average Price (Lakhs)"
    else:
        agg = df_use.groupby("location")["price_per_sqft_inr"].median().sort_values(ascending=False).head(top_n)
        y_title = "Median PPSF (₹/sqft)"

    fig_bar = px.bar(
        agg.reset_index(), x="location", y=agg.name,
        template=template, height=450
    )
    fig_bar.update_layout(
        xaxis={"categoryorder": "total descending"},
        yaxis_title=y_title,
        title=f"Top {len(agg)} Locations by {y_title}",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Price Spread by Location (Boxplot)")
    top_locs = agg.index.tolist()
    df_box = df[df["location"].isin(top_locs)].copy()
    fig_box = px.box(
        df_box, x="location", y="price", template=template,
        points="all" if show_outliers else False, height=450
    )
    fig_box.update_layout(yaxis_title="Price (Lakhs)", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_box, use_container_width=True)

    # Insights
    if len(agg) >= 3:
        top_loc = agg.index[0]; top_val = agg.iloc[0]
        bot_loc = agg.index[-1]; bot_val = agg.iloc[-1]
        st.info(
            f"**Insight:** **{top_loc}** leads at **{top_val:,.2f}** "
            f"{'Lakhs' if metric_choice.startswith('Average') else '₹/sqft'}, "
            f"~{(top_val/bot_val-1)*100:,.0f}% higher than **{bot_loc}** — "
            f"indicative of demand, amenities, and connectivity."
        )

    st.download_button(
        "⬇️ Download aggregated table (CSV)",
        data=agg.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"location_{'avgprice' if metric_choice.startswith('Average') else 'ppsf'}_top{len(agg)}.csv",
        mime="text/csv"
    )

# 📈 EDA
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    loc_filter = st.selectbox("Filter by Location", ["All"] + sorted(df["location"].unique()))
    df_eda = df if loc_filter == "All" else df[df["location"] == loc_filter]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df_eda):,}")
    c2.metric("Avg Price", format_price_lakh_crore(df_eda["price"].mean()))
    c3.metric("Median Price", format_price_lakh_crore(df_eda["price"].median()))
    c4.metric("Median PPSF", f"{int(df_eda['price_per_sqft_inr'].median()):,} ₹/sqft")
    c5.metric("Avg Sqft", f"{int(df_eda['total_sqft'].mean()):,}")

    left, right = st.columns(2)
    with left:
        fig_hist_price = px.histogram(df_eda, x="price", nbins=40, template=template)
        fig_hist_price.update_layout(title="Price Distribution (Lakhs)", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hist_price, use_container_width=True)
    with right:
        fig_hist_ppsf = px.histogram(df_eda, x="price_per_sqft_inr", nbins=40, template=template)
        fig_hist_ppsf.update_layout(title="Price per Sqft (₹/sqft) Distribution", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hist_ppsf, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        # FIXED: build value_counts with explicit names to avoid 'index' key error
        vc_bhk = df_eda["bhk"].value_counts().sort_index().reset_index(name="count").rename(columns={"index":"bhk"})
        fig_bhk = px.bar(vc_bhk, x="bhk", y="count", template=template)
        fig_bhk.update_layout(title="BHK Mix", xaxis_title="BHK", yaxis_title="Count", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bhk, use_container_width=True)
    with right2:
        vc_bath = df_eda["bath"].value_counts().sort_index().reset_index(name="count").rename(columns={"index":"bath"})
        fig_bath = px.bar(vc_bath, x="bath", y="count", template=template)
        fig_bath.update_layout(title="Bathroom Count Mix", xaxis_title="Bathrooms", yaxis_title="Count", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bath, use_container_width=True)

    st.markdown("#### Correlation Matrix (Selected Features)")
    corr_df = df_eda[["price", "total_sqft", "bhk", "bath", "price_per_sqft_inr"]].corr().round(2)
    fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu", template=template, aspect="auto")
    fig_corr.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=df_eda.to_csv(index=False).encode("utf-8"),
        file_name=f"bangalore_prices_{'ALL' if loc_filter=='All' else loc_filter}.csv",
        mime="text/csv"
    )

# 🧪 Sensitivity (Heatmap)
with tab_sens:
    st.subheader("Sensitivity Analysis (Heatmap)")

    # Choose two varying parameters
    axis_choice = st.radio("Vary along axes", ["Sqft vs BHK", "Sqft vs Bath", "BHK vs Bath"], horizontal=True)
    location = st.selectbox("Location", sorted(df["location"].unique()), index=0, key="sens_loc")

    # ranges
    if "Sqft" in axis_choice:
        sqft_min = st.number_input("Sqft min", 300, 20000, 600, step=50, key="sq_min")
        sqft_max = st.number_input("Sqft max", 300, 20000, 3000, step=50, key="sq_max")
        sqft_steps = st.slider("Sqft steps", 5, 40, 20)
        sqft_vals = np.linspace(sqft_min, max(sqft_min+100, sqft_max), sqft_steps, dtype=int)
    if "BHK" in axis_choice:
        bhk_min = st.number_input("BHK min", 1, 12, 1, step=1, key="bk_min")
        bhk_max = st.number_input("BHK max", 1, 12, 6, step=1, key="bk_max")
        bhk_vals = np.arange(bhk_min, bhk_max+1)
    if "Bath" in axis_choice:
        bath_min = st.number_input("Bath min", 1, 12, 1, step=1, key="bt_min")
        bath_max = st.number_input("Bath max", 1, 12, 5, step=1, key="bt_max")
        bath_vals = np.arange(bath_min, bath_max+1)

    # fixed params
    fixed_bhk = st.number_input("Fixed BHK (if not on axis)", 1, 12, 2, step=1)
    fixed_bath = st.number_input("Fixed Bath (if not on axis)", 1, 12, 2, step=1)
    fixed_sqft = st.number_input("Fixed Sqft (if not on axis)", 250, 20000, 1100, step=50)

    # compute grid
    def pred(sq, ba, bk):
        return predict_price(model, data_columns, location, sq, ba, bk)

    if axis_choice == "Sqft vs BHK":
        Z = np.array([[pred(s, fixed_bath, b) for b in bhk_vals] for s in sqft_vals])
        x_vals, y_vals = sqft_vals, bhk_vals
        xlab, ylab = "Sqft", "BHK"
    elif axis_choice == "Sqft vs Bath":
        Z = np.array([[pred(s, bth, fixed_bhk) for bth in bath_vals] for s in sqft_vals])
        x_vals, y_vals = sqft_vals, bath_vals
        xlab, ylab = "Sqft", "Bath"
    else:
        Z = np.array([[pred(fixed_sqft, bth, b) for b in bhk_vals] for bth in bath_vals])
        x_vals, y_vals = bath_vals, bhk_vals
        xlab, ylab = "Bath", "BHK"

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=Z, x=x_vals, y=y_vals, colorscale="Viridis", colorbar_title="Price (Lakhs)"
        )
    )
    fig_hm.update_layout(
        template=template, height=520, title=f"Sensitivity Heatmap • {location}",
        xaxis_title=xlab, yaxis_title=ylab, margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# 📦 Batch Predict
with tab_batch:
    st.subheader("Batch Predictions from CSV")
    st.caption("Upload a CSV with columns: total_sqft, bath, bhk and either a 'location' column or one-hot location columns matching training.")

    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_csv")
    if batch_file is not None:
        try:
            df_in = pd.read_csv(batch_file)
            df_in = rebuild_location_if_needed(df_in, data_columns)
            X = build_feature_matrix(df_in, data_columns)

            t2 = time.perf_counter()
            preds = getattr(model, "predict")(X)  # Lakhs
            elapsed_ms = (time.perf_counter() - t2) * 1000

            df_out = df_in.copy()
            df_out["predicted_price_lakh"] = preds
            df_out["predicted_price_readable"] = [format_price_lakh_crore(p) for p in preds]

            m1, m2, m3 = st.columns(3)
            m1.metric("Rows predicted", f"{len(df_out):,}")
            m2.metric("Mean predicted price", format_price_lakh_crore(df_out["predicted_price_lakh"].mean()))
            m3.caption(f"Inference time: {elapsed_ms:.0f} ms ({elapsed_ms/len(df_out):.2f} ms/row)")

            st.dataframe(df_out.head(25), use_container_width=True)
            st.download_button(
                "⬇️ Download predictions (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
    else:
        st.info("Upload a CSV to run batch predictions.")

# 🧠 Model Insights
with tab_model:
    st.subheader("Model Insights")
    st.write(f"**Model type:** `{model_kind(model)}`")
    st.caption("If the model is linear, we estimate location influence via coefficients (proxy importance).")

    pos, neg = get_linear_location_coefs(model, data_columns, top_k=12)
    if pos is None:
        st.info("Coefficient-based insights are available for linear models exposing `.coef_`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Positive Location Coefficients** *(push price up)*")
            df_pos = pos.reset_index()
            df_pos.columns = ["Location", "Coefficient"]
            fig_pos = px.bar(df_pos, x="Location", y="Coefficient", template=template)
            fig_pos.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pos, use_container_width=True)
        with c2:
            st.markdown("**Top Negative Location Coefficients** *(pull price down)*")
            df_neg = neg.reset_index()
            df_neg.columns = ["Location", "Coefficient"]
            fig_neg = px.bar(df_neg, x="Location", y="Coefficient", template=template)
            fig_neg.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_neg, use_container_width=True)

    st.markdown(
        "<div class='small-note'>Note: Coefficients reflect the fitted training data; "
        "interpretation should be paired with on-ground factors (supply, amenities, commute, schools).</div>",
        unsafe_allow_html=True
    )

# 📄 Report (PDF)
with tab_report:
    st.subheader("Generate PDF Report (KPIs + Charts)")

    # let user choose a slice (All or a specific location)
    loc_sel = st.selectbox("Slice to export", ["All"] + sorted(df["location"].unique()), key="pdf_slice")
    df_slice = df if loc_sel == "All" else df[df["location"] == loc_sel]

    # Build KPIs quickly
    kpis = {
        "Rows": f"{len(df_slice):,}",
        "Avg Price": format_price_lakh_crore(df_slice["price"].mean()),
        "Median Price": format_price_lakh_crore(df_slice["price"].median()),
        "Median PPSF": f"{int(df_slice['price_per_sqft_inr'].median()):,} ₹/sqft",
        "Avg Sqft": f"{int(df_slice['total_sqft'].mean()):,}",
    }

    # Create a couple of figures to embed
    figs = {}
    figs["price_hist"] = px.histogram(df_slice, x="price", nbins=40, template=template, title="Price Distribution (Lakhs)")
    figs["ppsf_hist"]  = px.histogram(df_slice, x="price_per_sqft_inr", nbins=40, template=template, title="Price per Sqft (₹/sqft)")

    st.write("**KPIs**")
    st.json(kpis)

    # Generate PDF
    if st.button("🧾 Build PDF"):
        pdf_path = f"report_{'ALL' if loc_sel=='All' else loc_sel}.pdf"
        saved_imgs = []

        # Try exporting charts as images (requires kaleido)
        try:
            for key, fig in figs.items():
                img_path = f"{key}.png"
                fig.write_image(img_path, scale=2, width=1000, height=600)  # needs kaleido
                saved_imgs.append(img_path)
            charts_ok = True
        except Exception:
            charts_ok = False

        # Try to write PDF with reportlab
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            from reportlab.lib.utils import ImageReader

            c = canvas.Canvas(pdf_path, pagesize=A4)
            W, H = A4
            y = H - 50
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, y, f"Bengaluru Home Price — Report ({loc_sel})")
            y -= 28
            c.setFont("Helvetica", 11)
            for k, v in kpis.items():
                c.drawString(40, y, f"{k}: {v}")
                y -= 18

            if charts_ok and saved_imgs:
                y -= 12
                for img in saved_imgs:
                    if y < 200:  # new page if too low
                        c.showPage()
                        y = H - 50
                    c.drawImage(ImageReader(img), 40, y-260, width=W-80, height=240, preserveAspectRatio=True, mask='auto')
                    y -= 270

            c.showPage()
            c.save()

            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF report", f, file_name=pdf_path, mime="application/pdf")
            if not charts_ok:
                st.warning("PDF built without charts (install `kaleido` for chart images).")
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")
            st.info("Install dependencies: `pip install reportlab kaleido` and ensure write permissions in the folder.")

# Footer note
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: var(--background-color);
    color: var(--text-color);
    text-align: center;
    padding: 12px 0;
    font-size: 14px;
    border-top: 1px solid rgba(49,51,63,0.2);
    box-shadow: 0 -2px 6px rgba(0,0,0,0.1);
}

.footer a {
    text-decoration: none;
    padding: 6px 12px;
    margin: 0 6px;
    border-radius: 20px;
    background: rgba(49, 51, 63, 0.05);
    color: var(--primary-color);
    font-weight: 500;
    transition: all 0.3s ease;
}

.footer a:hover {
    background: var(--primary-color);
    color: white !important;
}
</style>

<div class="footer">
<p>🚀 Created by <a href="https://www.linkedin.com/in/vishal--kapoor/" target="_blank">Vishal Kapoor</a> | 
<a href="https://github.com/Kapoor-Vishal" target="_blank">GitHub</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True
