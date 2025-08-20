import os
import sys
import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from streamlit_plotly_events import plotly_events

# =========================
# Config & static resources
# =========================

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

@st.cache_resource
def load_image(path):
    return Image.open(path)

favicon_path = resource_path("assets/icon.ico")
favicon = load_image(favicon_path) if os.path.exists(favicon_path) else None
st.set_page_config(page_title="Rosettier", page_icon=favicon, layout="centered")

logo_path = resource_path("assets/logo.png")
logo = load_image(logo_path) if os.path.exists(logo_path) else None

# Global Plotly config (for st.plotly_chart when not using plotly_events)
st.session_state.setdefault("plotly_config", {"displayModeBar": False, "responsive": True})

# =========================
# Performance & Sync (sidebar) — ENGLISH
# =========================
st.sidebar.header("Performance")
# Default OFF, as requested
lite_mode = st.sidebar.checkbox("Lightweight mode (recommended for slower PCs)", value=False)
MAX_SELECT = st.sidebar.number_input(
    "Selection limit (wells at once)",
    min_value=24, max_value=384, value=120, step=12
)

st.sidebar.header("Synchronization 96 ↔ 384")
auto_sync = st.sidebar.checkbox("Automatically sync changes between 96 and 384", value=True)

# =========================
# Constants & plate geometry
# =========================
rows_96 = list("ABCDEFGH")
columns_96 = list(range(1, 13))
rows_384 = list("ABCDEFGHIJKLMNOP")
columns_384 = list(range(1, 25))

row_indices_96 = {row: 8 - i for i, row in enumerate(rows_96)}
row_indices_384 = {row: 16 - i for i, row in enumerate(rows_384)}

# Row index maps
row_to_idx_96 = {r: i for i, r in enumerate(rows_96)}
idx_to_row_96 = {i: r for i, r in enumerate(rows_96)}
row_to_idx_384 = {r: i for i, r in enumerate(rows_384)}
idx_to_row_384 = {i: r for i, r in enumerate(rows_384)}

# ==================================
# Session state bootstrapping / init
# ==================================
if 'available_variables' not in st.session_state:
    st.session_state.available_variables = []

# Persist selections
st.session_state.setdefault("sel_wells_96", [])
for p in (1, 2, 3, 4):
    st.session_state.setdefault(f"sel_wells_{p}", [])
st.session_state.setdefault("sel_wells_combined", [])

# History as diffs
if 'history' not in st.session_state:
    st.session_state.history = {
        'plate_96_data': [],
        'plate_1_384_data': [],
        'plate_2_384_data': [],
        'plate_3_384_data': [],
        'plate_4_384_data': [],
        'combined_plate_data': [],
    }
    st.session_state.MAX_HISTORY = 200

def wells_96_df():
    return pd.DataFrame({'Well': [f"{r}{c}" for r in rows_96 for c in columns_96]})

def wells_384_df():
    return pd.DataFrame({'Well': [f"{r}{c}" for r in rows_384 for c in columns_384]})

def initialize_plates():
    if 'plate_96_data' not in st.session_state:
        st.session_state.plate_96_data = wells_96_df()

    for i in range(1, 5):
        key = f'plate_{i}_384_data'
        if key not in st.session_state:
            st.session_state[key] = wells_96_df()

    if 'combined_plate_data' not in st.session_state:
        st.session_state.combined_plate_data = wells_384_df()

    # Ensure variable columns exist
    for key in ['plate_96_data', 'plate_1_384_data', 'plate_2_384_data', 'plate_3_384_data', 'plate_4_384_data', 'combined_plate_data']:
        for var in st.session_state.available_variables:
            if var not in st.session_state[key].columns:
                st.session_state[key][var] = pd.NA

initialize_plates()

# =========================
# Utility helpers
# =========================
def parse_well(well: str):
    well = str(well).strip().upper()
    return well[0], int(well[1:])

def ensure_variables_exist(plate_key: str):
    for var in st.session_state.available_variables:
        if var not in st.session_state[plate_key].columns:
            st.session_state[plate_key][var] = pd.NA

def normalize_well_column(df: pd.DataFrame):
    if "Well" in df.columns:
        df["Well"] = df["Well"].astype(str).str.strip().str.upper()
    return df

def coerce_mixed_columns_to_string(df: pd.DataFrame):
    for c in df.columns:
        if c == 'Well':
            continue
        nonnull = df[c].dropna()
        if not nonnull.empty:
            try:
                pd.to_numeric(nonnull, errors='raise')
            except Exception:
                df[c] = df[c].astype(str)

def value_color(variable, value):
    if pd.isna(value):
        return "lightgray"
    key = f"{variable}:::{str(value)}"
    h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
    return f"#{h:06x}"

def create_plate_figure(df: pd.DataFrame, plate_type="96", current_variable=None, lite_mode=False):
    df_plot = df.copy()

    if plate_type == "96":
        num_cols, num_rows, marker_size = 12, 8, (28 if lite_mode else 40)
        row_indices, tick_text, title = row_indices_96, rows_96, "Rosettier - 96-Well Plate"
    else:
        num_cols, num_rows, marker_size = 24, 16, (10 if lite_mode else 15)
        row_indices, tick_text, title = row_indices_384, rows_384, "Rosettier - 384-Well Plate"

    parsed = df_plot['Well'].apply(parse_well)
    df_plot['Parsed_Row'] = [r for r, _ in parsed]
    df_plot['Parsed_Col'] = [c for _, c in parsed]
    df_plot['Y'] = df_plot['Parsed_Row'].map(row_indices)
    df_plot['X'] = df_plot['Parsed_Col']

    # Colors
    if current_variable and current_variable in df_plot.columns:
        vals = df_plot[current_variable]
        colors = [value_color(current_variable, v) for v in vals]
    else:
        colors = "lightgray"

    # Hover text
    if current_variable and current_variable in df_plot.columns:
        hovertexts = df_plot.apply(
            lambda r: f"Well: {r['Well']}<br>{current_variable}: {r[current_variable]}",
            axis=1
        )
    else:
        hovertexts = df_plot['Well']

    # Use WebGL in lightweight mode
    ScatterClass = go.Scattergl if lite_mode else go.Scatter

    show_text = (plate_type == "96") and (not lite_mode)  # text visible only on 96 & non-lite
    mode = 'markers+text' if show_text else 'markers'

    fig = go.Figure()
    fig.add_trace(ScatterClass(
        x=df_plot['X'],
        y=df_plot['Y'],
        mode=mode,
        marker=dict(
            size=marker_size,
            color=colors,
            line=dict(width=(0 if lite_mode else (0.8 if plate_type=="96" else 0.5)), color='black'),
            opacity=(1.0 if lite_mode else 0.9)
        ),
        text=None if not show_text else df_plot['Well'],
        textposition="middle center" if show_text else None,
        hovertext=hovertexts,
        hovertemplate="%{hovertext}<extra></extra>",
        name='Wells'
    ))

    fig.update_xaxes(range=[0.5, num_cols + 0.5], dtick=1, showgrid=not lite_mode, zeroline=False, showticklabels=True)
    fig.update_yaxes(
        range=[0.5, num_rows + 0.5], dtick=1, showgrid=not lite_mode, zeroline=False,
        showticklabels=True, tickmode='array', tickvals=list(range(num_rows, 0, -1)), ticktext=tick_text
    )

    # Grid only in normal mode
    if not lite_mode:
        line_w = 0.5 if plate_type == "384" else 0.8
        for i in range(1, num_cols + 1):
            fig.add_shape(type="line", x0=i, y0=0.5, x1=i, y1=num_rows + 0.5, line=dict(color="lightgray", width=line_w))
        for i in range(1, num_rows + 1):
            fig.add_shape(type="line", x0=0.5, y0=i, x1=num_cols + 0.5, y1=i, line=dict(color="lightgray", width=line_w))

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=True,
        height=600 if plate_type == "96" else 800,
        title=title,
        clickmode='event+select',
        dragmode='select',
        margin=dict(l=20, r=20, t=80, b=20),
        uirevision="keep"  # keep client-side state
    )
    return fig

# ---------- Mappings 96 <-> 384 ----------
def map_96_to_384_well(plate_num: int, well96: str) -> str:
    """
    plate_num in {1,2,3,4}
    1:(row_off=0,col_off=0), 2:(0,1), 3:(1,0), 4:(1,1)
    """
    row_off = 0 if plate_num in (1, 2) else 1
    col_off = 0 if plate_num in (1, 3) else 1

    r, c = parse_well(well96)  # r='A'..'H', c=1..12
    r96 = row_to_idx_96[r]     # 0..7
    r384_idx = r96 * 2 + row_off
    c384 = (c - 1) * 2 + col_off + 1  # 1..24
    r384 = idx_to_row_384[r384_idx]
    return f"{r384}{c384}"

def map_384_to_96_well(well384: str) -> tuple[int, str]:
    """
    Return: (plate_num, well96), plate_num in {1,2,3,4}
    """
    r, c = parse_well(well384)      # r='A'..'P', c=1..24
    r384_idx = row_to_idx_384[r]    # 0..15
    row_off = r384_idx % 2          # 0 -> plates 1/2, 1 -> 3/4
    col_off = (c % 2 == 0)          # True (1) if column is even -> plates 2/4

    if row_off == 0 and not col_off:
        plate_num = 1
    elif row_off == 0 and col_off:
        plate_num = 2
    elif row_off == 1 and not col_off:
        plate_num = 3
    else:
        plate_num = 4

    r96_idx = (r384_idx - row_off) // 2  # 0..7
    c96 = ((c - 1) - (1 if col_off else 0)) // 2 + 1  # 1..12
    r96 = idx_to_row_96[r96_idx]
    return plate_num, f"{r96}{c96}"

# ================
# History by diffs
# ================
def push_history_diff(plate_key, well, column, old_value, new_value):
    stack = st.session_state.history[plate_key]
    stack.append(("set", str(well), str(column), old_value, new_value))
    if len(stack) > st.session_state.MAX_HISTORY:
        stack.pop(0)

def undo(plate_key):
    stack = st.session_state.history[plate_key]
    if not stack:
        st.warning(f"No actions to undo for {plate_key}.")
        return
    action, well, column, old_value, new_value = stack.pop()
    if action == "set":
        df = st.session_state[plate_key]
        df.loc[df['Well'] == well, column] = old_value
        st.session_state[plate_key] = df
        st.success(f"Undo: {plate_key} — {well}.{column} ← {old_value}")

def safe_assign_value(df, well, column, value, plate_key):
    # Type: force string only if non-numeric
    try:
        float(value)
    except Exception:
        if column in df.columns:
            df[column] = df[column].astype(str)
    mask = (df['Well'] == well)
    if not mask.any():
        return
    old_value = df.loc[mask, column].iloc[0]
    push_history_diff(plate_key, well, column, old_value, value)
    df.loc[mask, column] = value
    st.session_state[plate_key] = df  # no unnecessary copy()

def assign_and_sync_from_96(plate_num: int, well: str, var: str, val):
    """Assign in 96 and (if auto_sync) mirror in 384"""
    plate_key = f'plate_{plate_num}_384_data'
    safe_assign_value(st.session_state[plate_key], well, var, val, plate_key)
    if auto_sync:
        comb_well = map_96_to_384_well(plate_num, well)
        safe_assign_value(st.session_state['combined_plate_data'], comb_well, var, val, 'combined_plate_data')

def assign_and_sync_from_384(well: str, var: str, val):
    """Assign in 384 and (if auto_sync) mirror in corresponding 96"""
    safe_assign_value(st.session_state['combined_plate_data'], well, var, val, 'combined_plate_data')
    if auto_sync:
        plate_num, well96 = map_384_to_96_well(well)
        plate_key = f'plate_{plate_num}_384_data'
        safe_assign_value(st.session_state[plate_key], well96, var, val, plate_key)

def copy_plate_data(source_key, dest_key):
    source_df = st.session_state[source_key].copy(deep=True)
    st.session_state[dest_key] = source_df
    new_cols = [c for c in source_df.columns if c not in ["Well"]]
    for col in new_cols:
        if col not in st.session_state.available_variables:
            st.session_state.available_variables.append(col)
    ensure_variables_exist(dest_key)
    st.success(f"Copied data from {source_key} to {dest_key}.")

# =========================
# Caching for downloads
# =========================
@st.cache_data
def to_excel(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Plate')
    return buf.getvalue()

@st.cache_data
def to_tsv(df):
    return df.to_csv(sep='\t', index=False).encode('utf-8')

# ======
# Header
# ======
if logo is not None:
    col1, col2, col3 = st.columns([1, 2, 1.5])
    with col1:
        pass
    with col2:
        st.markdown("<h1 style='text-align: left;'>Rosettier</h1>", unsafe_allow_html=True)
    with col3:
        st.image(logo, width=300)
else:
    st.title("Rosettier")

# =========================
# Sidebar: Variable manager
# =========================
st.sidebar.header("Variable Management")
new_variable = st.sidebar.text_input("Add new variable")
if st.sidebar.button("Add Variable") and new_variable:
    if new_variable not in st.session_state.available_variables:
        st.session_state.available_variables.append(new_variable)
        # add column across all plates
        for key in ['plate_96_data', 'plate_1_384_data', 'plate_2_384_data', 'plate_3_384_data', 'plate_4_384_data', 'combined_plate_data']:
            if new_variable not in st.session_state[key].columns:
                st.session_state[key][new_variable] = pd.NA
        st.sidebar.success(f"Variable '{new_variable}' added.")
    else:
        st.sidebar.warning("That variable already exists.")

st.sidebar.header("Variables Legend")
max_legend = 30
for var in st.session_state.available_variables:
    st.sidebar.markdown(f"**{var}**")
    # Unique values (capped)
    uniq_vals = []
    for key in ['plate_96_data', 'combined_plate_data']:
        if var in st.session_state[key].columns:
            uniq_vals.extend(st.session_state[key][var].dropna().unique().tolist())
    seen = set()
    shown = 0
    for v in uniq_vals:
        if v in seen:
            continue
        seen.add(v)
        if shown >= max_legend:
            st.sidebar.markdown("_… legend truncated …_")
            break
        st.sidebar.markdown(f"<span style='color:{value_color(var, v)}'>⬤</span> {v}", unsafe_allow_html=True)
        shown += 1
    st.sidebar.markdown("---")

st.sidebar.write("Available Variables:")
for var in st.session_state.available_variables:
    st.sidebar.write(f"- {var}")

# =========================
# Tabs
# =========================
main_tabs = st.tabs(["96-Well Plate", "384-Well Plate"])

# ------------- 96-Well Plate Tab -------------
with main_tabs[0]:
    st.subheader("Manage 96-Well Plate")

    if st.button("Undo Last Action (96-Well Plate)", key='undo_plate_96'):
        undo('plate_96_data')

    st.subheader("Upload 96-Well Plate from Excel/TSV/CSV")
    up_96_file = st.file_uploader("Upload a 96-well plate file", type=["xlsx", "csv", "tsv"], key="upload_96")

    if up_96_file and not st.session_state.get("uploaded_96_processed", False):
        file_extension = up_96_file.name.split('.')[-1].lower()
        try:
            if file_extension == "xlsx":
                uploaded_df = pd.read_excel(up_96_file)
            else:
                sep = "\t" if file_extension == "tsv" else ","
                uploaded_df = pd.read_csv(up_96_file, sep=sep)
            uploaded_df = normalize_well_column(uploaded_df)
            if len(uploaded_df) != 96:
                st.error("Uploaded file must have exactly 96 rows.")
            elif "Well" not in uploaded_df.columns:
                st.error("The file must contain a 'Well' column.")
            else:
                st.session_state.plate_96_data = uploaded_df.copy()
                new_cols = [c for c in uploaded_df.columns if c not in ["Well"]]
                for col in new_cols:
                    if col not in st.session_state.available_variables:
                        st.session_state.available_variables.append(col)
                ensure_variables_exist('plate_96_data')
                st.success("Successfully loaded 96-well data!")
                st.session_state.uploaded_96_processed = True
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    ensure_variables_exist('plate_96_data')
    plate_key_96 = 'plate_96_data'
    plate_df_96 = st.session_state[plate_key_96]

    selected_var_96 = st.selectbox("Select variable to visualize", options=["None"] + st.session_state.available_variables, key='select_var_96')

    # Recreate figure only if var or lite_mode changed
    last_signature = st.session_state.get("last_sig_96")
    this_signature = (selected_var_96, lite_mode)
    if last_signature != this_signature or "fig_96" not in st.session_state:
        st.session_state["fig_96"] = create_plate_figure(
            plate_df_96, plate_type="96",
            current_variable=None if selected_var_96 == "None" else selected_var_96,
            lite_mode=lite_mode
        )
        st.session_state["last_sig_96"] = this_signature

    fig_96 = st.session_state["fig_96"]
    selected_points_96 = plotly_events(fig_96, select_event=True, override_height=600, key='plotly_events_96')

    if selected_points_96:
        if len(selected_points_96) > MAX_SELECT:
            st.info(f"You selected {len(selected_points_96)} wells; only the first {MAX_SELECT} will be used.")
            selected_points_96 = selected_points_96[:MAX_SELECT]
        wells = []
        for p in selected_points_96:
            if isinstance(p, dict) and 'pointIndex' in p:
                idx = int(p['pointIndex'])
                if 0 <= idx < len(plate_df_96):
                    wells.append(plate_df_96.iloc[idx]['Well'])
        st.session_state.sel_wells_96 = wells

    st.caption(f"Selected wells: {', '.join(st.session_state.sel_wells_96) or '—'}")

    if st.session_state.sel_wells_96:
        if st.session_state.available_variables:
            var_to_assign_96 = st.selectbox("Select variable to assign", st.session_state.available_variables, key='assign_var_96')
            val_to_assign_96 = st.text_input(f"Value for '{var_to_assign_96}'", key='val_assign_96')
            if st.button("Assign value to selected wells", key='btn_assign_96'):
                if val_to_assign_96:
                    for w in st.session_state.sel_wells_96:
                        # Assign in the base 96 template (separate from plates 1–4)
                        safe_assign_value(st.session_state[plate_key_96], w, var_to_assign_96, val_to_assign_96, plate_key_96)
                    st.success(f"Assigned '{val_to_assign_96}' to wells in 96 base.")
                    if selected_var_96 == var_to_assign_96:
                        st.session_state["fig_96"] = create_plate_figure(
                            st.session_state[plate_key_96], plate_type="96",
                            current_variable=None if selected_var_96 == "None" else selected_var_96,
                            lite_mode=lite_mode
                        )
                        st.session_state["last_sig_96"] = (selected_var_96, lite_mode)
                else:
                    st.warning("Enter a value first.")
        else:
            st.info("No variables found. Add one in the sidebar to begin.")

    st.subheader("Plate Data")
    display_cols_96 = ['Well'] + st.session_state.available_variables
    st.dataframe(st.session_state[plate_key_96][display_cols_96], height=420, use_container_width=True)

    st.header("Download configuration")
    filename_96 = st.text_input("Base name for 96-well download:", value="Rosettier_Plate_96", key='dl_name_96') or "Rosettier_Plate_96"

    has_na_96 = st.session_state[plate_key_96][st.session_state.available_variables].isna().any().any() if st.session_state.available_variables else False
    proceed_dl_96 = True
    if has_na_96:
        proceed_dl_96 = st.checkbox("File has N/As. Continue anyway?", key='proceed_96')
        st.warning("Your file still contains N/As.")

    if proceed_dl_96:
        final_96 = st.session_state[plate_key_96][display_cols_96].copy()
        coerce_mixed_columns_to_string(final_96)
        excel_96 = to_excel(final_96)
        st.download_button(
            label="Download Excel",
            data=excel_96,
            file_name=f"{filename_96}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        tsv_96 = to_tsv(final_96)
        st.download_button(
            label="Download TSV",
            data=tsv_96,
            file_name=f"{filename_96}.tsv",
            mime='text/tab-separated-values'
        )

# ------------- 384-Well Plate Tab -------------
with main_tabs[1]:
    st.subheader("Manage 384-Well Plate")
    st.info("Manage four separate 96-well plates to combine into a single 384-well plate.")

    # ======= Upload 384 =======
    st.markdown("**Upload 384-well plate (Excel/TSV/CSV)**")
    up_384_file = st.file_uploader("Upload a 384-well plate file", type=["xlsx", "csv", "tsv"], key="upload_384")
    if up_384_file and not st.session_state.get("uploaded_384_processed", False):
        file_extension = up_384_file.name.split('.')[-1].lower()
        try:
            if file_extension == "xlsx":
                uploaded_384 = pd.read_excel(up_384_file)
            else:
                sep = "\t" if file_extension == "tsv" else ","
                uploaded_384 = pd.read_csv(up_384_file, sep=sep)
            uploaded_384 = normalize_well_column(uploaded_384)
            if len(uploaded_384) != 384:
                st.error("Uploaded file must have exactly 384 rows.")
            elif "Well" not in uploaded_384.columns:
                st.error("The file must contain a 'Well' column.")
            else:
                # Expand variables if new ones arrive
                new_cols = [c for c in uploaded_384.columns if c not in ["Well"]]
                for col in new_cols:
                    if col not in st.session_state.available_variables:
                        st.session_state.available_variables.append(col)
                        # Add column to all plates
                        for key in ['plate_96_data', 'plate_1_384_data', 'plate_2_384_data', 'plate_3_384_data', 'plate_4_384_data', 'combined_plate_data']:
                            if col not in st.session_state[key].columns:
                                st.session_state[key][col] = pd.NA
                st.session_state['combined_plate_data'] = uploaded_384.copy()
                st.success("Successfully loaded 384-well data!")
                st.session_state.uploaded_384_processed = True
        except Exception as e:
            st.error(f"Failed to read 384 file: {e}")

    # ======= Split 384 → 4x96 =======
    if st.button("Split current 384 into four 96-well plates (overwrite Plates 1–4)", key='btn_split_384'):
        comb = st.session_state['combined_plate_data']
        new_plates = {i: wells_96_df() for i in [1, 2, 3, 4]}
        # Ensure columns
        for i in [1, 2, 3, 4]:
            for var in st.session_state.available_variables:
                if var not in new_plates[i].columns:
                    new_plates[i][var] = pd.NA
        # Distribute values
        for _, row in comb.iterrows():
            well384 = row['Well']
            plate_num, well96 = map_384_to_96_well(well384)
            for var in st.session_state.available_variables:
                val = row.get(var, pd.NA)
                if pd.isna(val):
                    continue
                mask = (new_plates[plate_num]['Well'] == well96)
                new_plates[plate_num].loc[mask, var] = val
        # Overwrite
        for i in [1, 2, 3, 4]:
            st.session_state[f'plate_{i}_384_data'] = new_plates[i]
        st.success("Split done: Plates 1–4 updated from current 384.")

    plate_subtabs = st.tabs([f"Plate {i}" for i in range(1, 5)])

    def manage_plate(plate_num: int):
        plate_key = f'plate_{plate_num}_384_data'
        df_plate = st.session_state[plate_key]
        st.markdown(f"### Manage Plate {plate_num}")

        # Copy from other plate
        source_opts = [i for i in [1, 2, 3, 4] if i != plate_num]
        chosen_src = st.selectbox(f"Copy from Plate ... to Plate {plate_num}", source_opts, key=f'plate_copy_select_{plate_num}')
        if st.button(f"Copy from Plate {chosen_src}", key=f'btn_copy_{plate_num}'):
            src_key = f'plate_{chosen_src}_384_data'
            copy_plate_data(src_key, plate_key)

        if st.button(f"Undo Last Action (Plate {plate_num})", key=f'undo_plate_{plate_num}'):
            undo(plate_key)

        st.markdown("**Upload data (Excel/TSV/CSV)**")
        up_file = st.file_uploader(f"Upload Plate {plate_num} file", type=["xlsx", "csv", "tsv"], key=f'file_up_{plate_num}')
        if up_file and not st.session_state.get(f"uploaded_plate_{plate_num}_processed", False):
            file_ext = up_file.name.split('.')[-1].lower()
            try:
                if file_ext == "xlsx":
                    uploaded_df = pd.read_excel(up_file)
                else:
                    sep = "\t" if file_ext == "tsv" else ","
                    uploaded_df = pd.read_csv(up_file, sep=sep)
                uploaded_df = normalize_well_column(uploaded_df)
                if len(uploaded_df) != 96:
                    st.error("File must have exactly 96 rows for a 96-well plate.")
                elif "Well" not in uploaded_df.columns:
                    st.error("The file must contain a 'Well' column.")
                else:
                    st.session_state[plate_key] = uploaded_df.copy()
                    new_cols = [c for c in uploaded_df.columns if c not in ["Well"]]
                    for col in new_cols:
                        if col not in st.session_state.available_variables:
                            st.session_state.available_variables.append(col)
                            # Add column to all plates
                            for key in ['plate_96_data', 'plate_1_384_data', 'plate_2_384_data', 'plate_3_384_data', 'plate_4_384_data', 'combined_plate_data']:
                                if col not in st.session_state[key].columns:
                                    st.session_state[key][col] = pd.NA
                    ensure_variables_exist(plate_key)
                    st.success(f"Loaded data into Plate {plate_num}.")
                    st.session_state[f"uploaded_plate_{plate_num}_processed"] = True
            except Exception as e:
                st.error(f"Error reading file: {e}")

        ensure_variables_exist(plate_key)
        var_plate = st.selectbox(f"Select variable to visualize (Plate {plate_num})", ["None"] + st.session_state.available_variables, key=f'var_sel_plate_{plate_num}')

        # Cache fig per plate, sensitive to lite_mode
        last_key = f"last_sig_{plate_num}"
        fig_key = f"fig_{plate_num}"
        sig = (var_plate, lite_mode)
        if st.session_state.get(last_key) != sig or fig_key not in st.session_state:
            st.session_state[fig_key] = create_plate_figure(
                df_plate, plate_type="96",
                current_variable=None if var_plate == "None" else var_plate,
                lite_mode=lite_mode
            )
            st.session_state[last_key] = sig

        fig_p = st.session_state[fig_key]
        points_p = plotly_events(fig_p, select_event=True, override_height=600, key=f'plotly_ev_{plate_num}')
        if points_p:
            if len(points_p) > MAX_SELECT:
                st.info(f"You selected {len(points_p)} wells; only the first {MAX_SELECT} will be used.")
                points_p = points_p[:MAX_SELECT]
            wells = []
            for p in points_p:
                if isinstance(p, dict) and 'pointIndex' in p:
                    idx = int(p['pointIndex'])
                    if 0 <= idx < len(df_plate):
                        wells.append(df_plate.iloc[idx]['Well'])
            st.session_state[f"sel_wells_{plate_num}"] = wells

        st.caption(f"Selected wells (Plate {plate_num}): {', '.join(st.session_state.get(f'sel_wells_{plate_num}', [])) or '—'}")

        if st.session_state.get(f"sel_wells_{plate_num}", []):
            if st.session_state.available_variables:
                var_assign = st.selectbox(f"Assign which variable? (Plate {plate_num})", st.session_state.available_variables, key=f'assign_var_plate_{plate_num}')
                val_assign = st.text_input(f"Value for '{var_assign}' (Plate {plate_num})", key=f'val_assign_{plate_num}')
                if st.button(f"Assign value (Plate {plate_num})", key=f'btn_assign_{plate_num}'):
                    if val_assign:
                        for w in st.session_state[f"sel_wells_{plate_num}"]:
                            assign_and_sync_from_96(plate_num, w, var_assign, val_assign)
                        st.success(f"Assigned '{val_assign}' to wells in Plate {plate_num}.")
                        if var_plate == var_assign:
                            st.session_state[fig_key] = create_plate_figure(
                                st.session_state[plate_key], plate_type="96",
                                current_variable=None if var_plate == "None" else var_plate,
                                lite_mode=lite_mode
                            )
                            st.session_state[last_key] = (var_plate, lite_mode)
                    else:
                        st.warning("Enter a value first.")
            else:
                st.info("No variables available. Add in the sidebar first.")

        st.markdown("**Plate Data**")
        disp_cols_sub = ['Well'] + st.session_state.available_variables
        st.dataframe(st.session_state[plate_key][disp_cols_sub], height=420, use_container_width=True)

    # Subtabs 1–4
    for i in range(1, 5):
        with plate_subtabs[i - 1]:
            manage_plate(i)

    st.info("""
    **Important Note:**  
    When combining plates, the wells are mapped as follows:
    - **A1** of the combined 384-well plate is **A1** from **Plate 1**
    - **A2** of the combined 384-well plate is **A1** from **Plate 2**
    - **B1** of the combined 384-well plate is **A1** from **Plate 3**
    - **B2** of the combined 384-well plate is **A1** from **Plate 4**
    ... and so on.
    """)

    # ======= Combine 4x96 → 384 =======
    if st.button("Combine All Plates into 384-Well Plate", key='btn_combine_384'):
        plate1 = st.session_state['plate_1_384_data']
        plate2 = st.session_state['plate_2_384_data']
        plate3 = st.session_state['plate_3_384_data']
        plate4 = st.session_state['plate_4_384_data']
        # Vectorized combination
        combined_plate_df = st.session_state.combined_plate_data.copy().set_index('Well')
        plate_mappings = {
            1: {'plate_data': plate1, 'row_offset': 0, 'col_offset': 0},
            2: {'plate_data': plate2, 'row_offset': 0, 'col_offset': 1},
            3: {'plate_data': plate3, 'row_offset': 1, 'col_offset': 0},
            4: {'plate_data': plate4, 'row_offset': 1, 'col_offset': 1},
        }
        for plate_num, mapping in plate_mappings.items():
            plate_df = mapping['plate_data'].copy()
            plate_df['row_letter'] = plate_df['Well'].str[0]
            plate_df['col_num'] = plate_df['Well'].str[1:].astype(int)
            row_map = {r: i for i, r in enumerate(rows_96)}
            plate_df['row_idx_96'] = plate_df['row_letter'].map(row_map)
            plate_df['combined_row_idx'] = plate_df['row_idx_96'] * 2 + mapping['row_offset']
            plate_df['combined_col'] = (plate_df['col_num'] - 1) * 2 + mapping['col_offset'] + 1
            valid = (plate_df['combined_row_idx'] < len(rows_384)) & (plate_df['combined_col'] <= max(columns_384))
            plate_df = plate_df[valid].copy()
            plate_df['combined_row'] = plate_df['combined_row_idx'].apply(lambda idx: rows_384[int(idx)])
            plate_df['combined_well'] = plate_df['combined_row'] + plate_df['combined_col'].astype(str)
            for variable in st.session_state.available_variables:
                if variable in plate_df.columns:
                    updates = plate_df.set_index('combined_well')[variable].dropna()
                    idx = combined_plate_df.index.intersection(updates.index)
                    combined_plate_df.loc[idx, variable] = updates.loc[idx]
        st.session_state.combined_plate_data = combined_plate_df.reset_index()
        st.success("Successfully combined the four 96-well plates into a 384-well plate.")

    if st.button("Undo Last Action (Combined Plate)", key='undo_combined'):
        undo('combined_plate_data')

    combined_df = st.session_state['combined_plate_data']
    st.markdown("**Combined Plate Data**")
    ensure_variables_exist('combined_plate_data')
    disp_cols_combined = ['Well'] + st.session_state.available_variables
    st.dataframe(combined_df[disp_cols_combined], height=420, use_container_width=True)

    var_combined = st.selectbox("Select variable to visualize (Combined Plate)", ["None"] + st.session_state.available_variables, key='var_sel_combined')

    last_sig = st.session_state.get("last_sig_combined")
    sig = (var_combined, lite_mode)
    if last_sig != sig or "fig_combined" not in st.session_state:
        st.session_state["fig_combined"] = create_plate_figure(
            combined_df, plate_type="384",
            current_variable=None if var_combined == "None" else var_combined,
            lite_mode=lite_mode
        )
        st.session_state["last_sig_combined"] = sig

    fig_combined = st.session_state["fig_combined"]
    points_combined = plotly_events(fig_combined, select_event=True, override_height=800, key='plotly_ev_combined')
    if points_combined:
        if len(points_combined) > MAX_SELECT:
            st.info(f"You selected {len(points_combined)} wells; only the first {MAX_SELECT} will be used.")
            points_combined = points_combined[:MAX_SELECT]
        wells_c = []
        for p in points_combined:
            if isinstance(p, dict) and 'pointIndex' in p:
                idx = int(p['pointIndex'])
                if 0 <= idx < len(combined_df):
                    wells_c.append(combined_df.iloc[idx]['Well'])
        st.session_state.sel_wells_combined = wells_c

    st.caption(f"Selected wells (Combined): {', '.join(st.session_state.sel_wells_combined) or '—'}")

    if st.session_state.sel_wells_combined:
        if st.session_state.available_variables:
            var_assign_comb = st.selectbox("Assign which variable? (Combined Plate)", st.session_state.available_variables, key='assign_var_combined')
            val_assign_comb = st.text_input(f"Value for '{var_assign_comb}' (Combined Plate)", key='val_assign_comb')
            if st.button("Assign value (Combined Plate)", key='btn_assign_comb'):
                if val_assign_comb:
                    for w in st.session_state.sel_wells_combined:
                        assign_and_sync_from_384(w, var_assign_comb, val_assign_comb)
                    st.success(f"Assigned '{val_assign_comb}' to selected wells in Combined Plate.")
                    if var_combined == var_assign_comb:
                        st.session_state["fig_combined"] = create_plate_figure(
                            st.session_state['combined_plate_data'], plate_type="384",
                            current_variable=None if var_combined == "None" else var_combined,
                            lite_mode=lite_mode
                        )
                        st.session_state["last_sig_combined"] = (var_combined, lite_mode)
                else:
                    st.warning("Enter a value first.")
        else:
            st.info("No variables available. Add in the sidebar first.")

    st.header("Download combined 384-well configuration")
    dl_name_combined = st.text_input("Base name for combined 384-well download:", value="Rosettier_384_Plate", key='dl_name_384') or "Rosettier_384_Plate"

    has_na_combined = combined_df[st.session_state.available_variables].isna().any().any() if st.session_state.available_variables else False
    proceed_combined = True
    if has_na_combined:
        proceed_combined = st.checkbox("File has N/As. Continue anyway?", key='proceed_384')
        st.warning("File still has N/As.")

    if proceed_combined:
        final_384 = combined_df[disp_cols_combined].copy()
        coerce_mixed_columns_to_string(final_384)
        excel_384 = to_excel(final_384)
        st.download_button(
            label="Download Excel",
            data=excel_384,
            file_name=f"{dl_name_combined}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        tsv_384 = to_tsv(final_384)
        st.download_button(
            label="Download TSV",
            data=tsv_384,
            file_name=f"{dl_name_combined}.tsv",
            mime='text/tab-separated-values'
        )

st.markdown("---")
st.markdown("""
**Authorship Statement**  
Developed by Ignacio de Quinto in 2025.  
Contact me at idequintoc@gmail.com for licensing or collaboration inquiries.
""")
