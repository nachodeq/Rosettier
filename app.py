import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from io import BytesIO
import random
from PIL import Image
import os
import sys

############################
# 1. Resource path and setup
############################

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

favicon_path = resource_path("assets/icon.ico")
favicon = Image.open(favicon_path) if os.path.exists(favicon_path) else None

st.set_page_config(
    page_title="Rosettier",
    page_icon=favicon,  # Use the loaded favicon
    layout="centered"
)

############################
# 2. Plate definitions
############################

rows_96 = list("ABCDEFGH")
columns_96 = list(range(1, 13))

rows_384 = list("ABCDEFGHIJKLMNOP")
columns_384 = list(range(1, 25))

############################
# 3. Color management
############################

def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def get_color(variable, value):
    """Retrieves the color assigned to a specific value of a variable.
    If the value does not have an assigned color, generates a new one."""
    if variable not in st.session_state.color_map:
        st.session_state.color_map[variable] = {}
    
    if value not in st.session_state.color_map[variable]:
        st.session_state.color_map[variable][value] = generate_random_color()
    
    return st.session_state.color_map[variable][value]

############################
# 4. Well parsing
############################

def parse_well(well):
    """Parse 'A1' into row='A', column=1 (for 96-well)."""
    row = well[0]
    column = int(well[1:])
    return row, column

def parse_well_384(well):
    """Parse 'A1' into row='A', column=1 (for 384-well)."""
    row = well[0]
    column = int(well[1:])
    return row, column

############################
# 5. Plotting - 96-well
############################
def create_plate_visualization(plate_df, current_variable=None):
    """
    Build a scatter plot for a 96-well plate.
    We do row/col transforms in a local copy so we never store them in session state.
    """
    df_plot = plate_df.copy()

    # Assign colors
    if current_variable and current_variable in df_plot.columns:
        colors = df_plot['Well'].map(
            lambda well: get_color(
                current_variable,
                df_plot.loc[df_plot['Well'] == well, current_variable].values[0]
            )
            if not pd.isna(df_plot.loc[df_plot['Well'] == well, current_variable].values[0])
            else 'lightgray'
        )
    else:
        colors = 'lightgray'

    # Parse row/column in local copy
    df_plot['Parsed_Row'], df_plot['Parsed_Column'] = zip(*df_plot['Well'].map(parse_well))
    row_indices = {row: 8 - idx for idx, row in enumerate(rows_96)}
    df_plot['Y'] = df_plot['Parsed_Row'].map(row_indices)
    df_plot['X'] = df_plot['Parsed_Column']

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot['X'],
            y=df_plot['Y'],
            mode='markers+text',
            marker=dict(
                size=40,
                color=colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=df_plot['Well'],
            customdata=df_plot.index,
            textposition="middle center",
            hoverinfo='text',
            name='Wells'
        )
    )

    # Axes (A..H top to bottom)
    fig.update_xaxes(range=[0.5, 12.5], dtick=1, showgrid=True, zeroline=False, showticklabels=True)
    fig.update_yaxes(
        range=[0.5, 8.5],
        dtick=1,
        showgrid=True,
        zeroline=False,
        showticklabels=True,
        tickmode='array',
        tickvals=[8,7,6,5,4,3,2,1],
        ticktext=rows_96
    )

    # Grid lines
    for i in range(1, 13):
        fig.add_shape(
            type="line",
            x0=i, y0=0.5, x1=i, y1=8.5,
            line=dict(color="black", width=1, dash='dot'),
            opacity=0.5
        )
    for i in range(1, 9):
        fig.add_shape(
            type="line",
            x0=0.5, y0=i, x1=12.5, y1=i,
            line=dict(color="black", width=1, dash='dot'),
            opacity=0.5
        )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=True,
        height=600,
        title="Rosettier - 96-Well Plate",
        clickmode='event+select',
        dragmode='select',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig

############################
# 6. Plotting - 384-well
############################
def create_combined_plate_visualization(combined_plate_df, current_variable=None):
    """
    Build a scatter plot for a 384-well plate (local copy).
    """
    df_plot = combined_plate_df.copy()

    if current_variable and current_variable in df_plot.columns:
        colors = df_plot['Well'].map(
            lambda well: get_color(
                current_variable,
                df_plot.loc[df_plot['Well'] == well, current_variable].values[0]
            )
            if not pd.isna(df_plot.loc[df_plot['Well'] == well, current_variable].values[0])
            else 'lightgray'
        )
    else:
        colors = 'lightgray'

    df_plot['Parsed_Row'], df_plot['Parsed_Column'] = zip(*df_plot['Well'].map(parse_well_384))
    row_indices = {row: 16 - idx for idx, row in enumerate(rows_384)}
    df_plot['Y'] = df_plot['Parsed_Row'].map(row_indices)
    df_plot['X'] = df_plot['Parsed_Column']

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot['X'],
            y=df_plot['Y'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=0.5, color='black'),
                opacity=0.8
            ),
            text=df_plot['Well'],
            customdata=df_plot.index,
            textposition="middle center",
            hoverinfo='text',
            name='Wells'
        )
    )

    fig.update_xaxes(range=[0.5, 24.5], dtick=1, showgrid=True, zeroline=False, showticklabels=True)
    fig.update_yaxes(
        range=[0.5, 16.5],
        dtick=1,
        showgrid=True,
        zeroline=False,
        showticklabels=True,
        tickmode='array',
        tickvals=list(range(16, 0, -1)),
        ticktext=rows_384
    )

    for i in range(1, 25):
        fig.add_shape(
            type="line",
            x0=i, y0=0.5, x1=i, y1=16.5,
            line=dict(color="black", width=0.5, dash='dot'),
            opacity=0.5
        )
    for i in range(1, 17):
        fig.add_shape(
            type="line",
            x0=0.5, y0=i, x1=24.5, y1=i,
            line=dict(color="black", width=0.5, dash='dot'),
            opacity=0.5
        )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=True,
        height=800,
        title="Rosettier - 384-Well Plate",
        clickmode='event+select',
        dragmode='select',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig

############################
# 7. Combining 4x96 into 384
############################
def combine_plates(plate1, plate2, plate3, plate4):
    """
    Combine four 96-well plates into one 384-well plate (interleaved).
    """
    combined_plate_df = st.session_state.combined_plate_data.copy()
    
    plate_mappings = {
        1: {'plate_data': plate1, 'row_offset': 0, 'col_offset': 0},
        2: {'plate_data': plate2, 'row_offset': 0, 'col_offset': 1},
        3: {'plate_data': plate3, 'row_offset': 1, 'col_offset': 0},
        4: {'plate_data': plate4, 'row_offset': 1, 'col_offset': 1},
    }
    
    for plate_num, mapping in plate_mappings.items():
        plate_df = mapping['plate_data']
        row_offset = mapping['row_offset']
        col_offset = mapping['col_offset']
        
        for _, row in plate_df.iterrows():
            well = row['Well']
            row_label, col_label = parse_well(well)
            
            try:
                row_idx_96 = rows_96.index(row_label)
            except ValueError:
                st.warning(f"Invalid row '{row_label}' in Plate {plate_num}. Skipping.")
                continue
            
            if col_label not in columns_96:
                st.warning(f"Invalid column '{col_label}' in Plate {plate_num}. Skipping.")
                continue
            
            combined_row_idx = row_idx_96 * 2 + row_offset
            combined_col_idx = (col_label -1)*2 + col_offset +1

            if combined_row_idx >= len(rows_384):
                st.warning(f"Row idx {combined_row_idx} out of range. Skipping.")
                continue
            combined_row_label = rows_384[combined_row_idx]
            
            if combined_col_idx > max(columns_384):
                st.warning(f"Column idx {combined_col_idx} out of range. Skipping.")
                continue

            combined_well = f"{combined_row_label}{combined_col_idx}"

            # Transfer all user variables
            for variable in st.session_state.available_variables:
                value = row.get(variable, pd.NA)
                combined_plate_df.loc[combined_plate_df['Well'] == combined_well, variable] = value
    
    st.session_state.combined_plate_data = combined_plate_df

############################
# 8. Session state init
############################
def initialize_plates():
    # Single 96
    if 'plate_96_data' not in st.session_state:
        wells_96_list = [f"{r}{c}" for r in rows_96 for c in columns_96]
        df_96 = pd.DataFrame({'Well': wells_96_list})
        st.session_state.plate_96_data = df_96

    # Four sub-plates
    for i in range(1, 5):
        plate_key = f'plate_{i}_384_data'
        if plate_key not in st.session_state:
            wells_96_list = [f"{r}{c}" for r in rows_96 for c in columns_96]
            df_sub = pd.DataFrame({'Well': wells_96_list})
            for var in st.session_state.get('available_variables', []):
                df_sub[var] = pd.NA
            st.session_state[plate_key] = df_sub

    # Combined 384
    if 'combined_plate_data' not in st.session_state:
        comb_wells = [f"{r}{c}" for r in rows_384 for c in columns_384]
        df_comb = pd.DataFrame({'Well': comb_wells})
        for var in st.session_state.get('available_variables', []):
            df_comb[var] = pd.NA
        st.session_state.combined_plate_data = df_comb

initialize_plates()

# List of variables
if 'available_variables' not in st.session_state:
    st.session_state.available_variables = []

# Color map
if 'color_map' not in st.session_state:
    st.session_state.color_map = {}

# Refresh
if 'refresh' not in st.session_state:
    st.session_state.refresh = False

############################
# 9. History / Undo init
############################
if 'history' not in st.session_state:
    st.session_state.history = {
        'plate_96_data': [],
        'plate_1_384_data': [],
        'plate_2_384_data': [],
        'plate_3_384_data': [],
        'plate_4_384_data': [],
        'combined_plate_data': []
    }

def push_history(plate_key):
    """Before modifying, save a snapshot."""
    df_copy = st.session_state[plate_key].copy(deep=True)
    st.session_state.history[plate_key].append(df_copy)

############################
# 10. Remove Unused Colors
############################
def remove_unused_colors():
    """
    Remove color legend entries for variable-value pairs not in ANY plate.
    """
    all_plates = [
        'plate_96_data',
        'plate_1_384_data',
        'plate_2_384_data',
        'plate_3_384_data',
        'plate_4_384_data',
        'combined_plate_data'
    ]
    for var in st.session_state.available_variables:
        used_values = set()
        for pk in all_plates:
            if var in st.session_state[pk].columns:
                plate_vals = st.session_state[pk][var].dropna().unique()
                used_values.update(plate_vals)
        current_map = st.session_state.color_map.get(var, {})
        to_remove = [val for val in current_map if val not in used_values]
        for val in to_remove:
            del st.session_state.color_map[var][val]

############################
# 11. Ensure Variables Exist
############################
def ensure_variables_exist(plate_key):
    """
    Auto-add missing columns for each available variable
    so we never get KeyErrors on display.
    """
    for var in st.session_state.available_variables:
        if var not in st.session_state[plate_key].columns:
            st.session_state[plate_key][var] = pd.NA

############################
# 12. Coerce Mixed Columns to String
############################
def coerce_mixed_columns_to_string(df):
    """
    For every column except 'Well', if there's at least 
    one non-numeric value, cast the entire column to str.
    This ensures Arrow doesn't complain about mixed numeric+text.
    """
    for c in df.columns:
        if c == 'Well':
            continue
        nonnull = df[c].dropna()
        if len(nonnull) > 0:
            try:
                # If to_numeric fails, there's some text
                pd.to_numeric(nonnull, errors='raise')
            except:
                # If we can't parse all as numeric, make column string
                df[c] = df[c].astype(str)

############################
# 13. Safe Assign for Mixed Types
############################
def safe_assign_value(df, well, column, value):
    """
    If 'value' is not numeric, convert the entire column to string first.
    Then assign the value to the row matching 'Well'.
    """
    # Attempt numeric parse
    try:
        float(value)
    except ValueError:
        # Convert entire column to string
        df[column] = df[column].astype(str)

    df.loc[df['Well'] == well, column] = value

############################
# 14. Undo function
############################
def undo(plate_key):
    """
    Undo last change. Then ensure variables exist, 
    coerce columns if needed, remove unused colors, maybe rerun.
    """
    if st.session_state.history[plate_key]:
        last_snapshot = st.session_state.history[plate_key].pop()
        st.session_state[plate_key] = last_snapshot
        st.success(f"Undo successful for {plate_key}.")

        ensure_variables_exist(plate_key)
        coerce_mixed_columns_to_string(st.session_state[plate_key])
        remove_unused_colors()

        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.info("Please refresh or interact again to see updates.")
    else:
        st.warning(f"No actions to undo for {plate_key}.")

############################
# 15. Copy Plate Feature
############################
def copy_plate_data(source_key, dest_key):
    """
    Overwrite dest_key data with source_key data.
    Then ensure variables exist, coerce mixed cols, remove old colors, rerun if possible.
    """
    push_history(dest_key)
    source_df = st.session_state[source_key].copy(deep=True)
    st.session_state[dest_key] = source_df

    # If the source has new columns, add them to available_variables
    new_cols = [c for c in source_df.columns if c not in ["Well"]]
    for col in new_cols:
        if col not in st.session_state.available_variables:
            st.session_state.available_variables.append(col)
            st.session_state.color_map[col] = {}

    ensure_variables_exist(dest_key)
    coerce_mixed_columns_to_string(st.session_state[dest_key])
    remove_unused_colors()
    st.success(f"Copied data from {source_key} to {dest_key}.")

    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.info("Please refresh or interact again to see updates.")

############################
# 16. Header & Logo
############################
logo_path = resource_path("assets/logo.png")
try:
    logo = Image.open(logo_path)
except FileNotFoundError:
    logo = None

if logo:
    col1, col2, col3 = st.columns([1, 2, 1.5])
    with col1:
        pass
    with col2:
        st.markdown("<h1 style='text-align: left;'>Rosettier</h1>", unsafe_allow_html=True)
    with col3:
        st.image(logo, width=300)
else:
    st.title("Rosettier")

############################
# 17. Sidebar
############################
st.sidebar.header("Variable Management")

new_variable = st.sidebar.text_input("Add New Variable")
if st.sidebar.button("Add Variable") and new_variable:
    if new_variable not in st.session_state.available_variables:
        st.session_state.available_variables.append(new_variable)
        st.session_state.color_map[new_variable] = {}
        # Ensure each plate has the new column
        st.session_state.plate_96_data[new_variable] = pd.NA
        for i in range(1, 5):
            pk = f'plate_{i}_384_data'
            st.session_state[pk][new_variable] = pd.NA
        st.session_state.combined_plate_data[new_variable] = pd.NA
        st.sidebar.success(f"Variable '{new_variable}' added.")
    else:
        st.sidebar.warning("That variable already exists.")

st.sidebar.header("Variables Legend")
for var, values in st.session_state.color_map.items():
    st.sidebar.markdown(f"**{var}**")
    for v, color in values.items():
        st.sidebar.markdown(f"<span style='color:{color}'>⬤</span> {v}", unsafe_allow_html=True)
    st.sidebar.markdown("---")

st.sidebar.write("Available Variables:")
for var in st.session_state.available_variables:
    st.sidebar.write(f"- {var}")

############################
# 18. Main Tabs
############################
main_tabs = st.tabs(["96-Well Plate", "384-Well Plate"])

############################
# Tab 1: 96-Well Plate
############################
with main_tabs[0]:
    st.subheader("Manage 96-Well Plate")

    if st.button("Undo Last Action (96-Well Plate)", key='undo_plate_96'):
        undo('plate_96_data')

    st.subheader("Upload 96-Well Plate from Excel/TSV/CSV")
    up_96_file = st.file_uploader("Upload a 96-well plate file", type=["xlsx","csv","tsv"], key="upload_96")
    if up_96_file:
        file_extension = up_96_file.name.split('.')[-1].lower()
        try:
            if file_extension == "xlsx":
                uploaded_df = pd.read_excel(up_96_file)
            else:
                sep = "\t" if file_extension == "tsv" else ","
                uploaded_df = pd.read_csv(up_96_file, sep=sep)

            if len(uploaded_df) != 96:
                st.error("Uploaded file must have exactly 96 rows.")
            elif "Well" not in uploaded_df.columns:
                st.error("The file must contain a 'Well' column.")
            else:
                st.session_state.plate_96_data = uploaded_df.copy()

                # Register any new columns as variables
                new_cols = [c for c in uploaded_df.columns if c not in ["Well"]]
                for col in new_cols:
                    if col not in st.session_state.available_variables:
                        st.session_state.available_variables.append(col)
                        st.session_state.color_map[col] = {}

                # 1) Ensure missing columns are added for known variables
                ensure_variables_exist('plate_96_data')
                # 2) Coerce any column with strings + numbers to string
                coerce_mixed_columns_to_string(st.session_state['plate_96_data'])

                st.success("Successfully loaded 96-well data!")
                st.session_state.refresh = not st.session_state.refresh

        except Exception as e:
            st.error(f"Failed to read file: {e}")

    # Ensure columns for known variables exist in case of previous changes
    ensure_variables_exist('plate_96_data')
    coerce_mixed_columns_to_string(st.session_state['plate_96_data'])

    plate_key_96 = 'plate_96_data'
    plate_df_96 = st.session_state[plate_key_96]
    selected_var_96 = st.selectbox(
        "Select Variable to Visualize",
        options=["None"] + st.session_state.available_variables,
        key='select_var_96'
    )
    fig_96 = create_plate_visualization(plate_df_96, current_variable=None if selected_var_96 == "None" else selected_var_96)
    selected_points_96 = plotly_events(fig_96, select_event=True, override_height=600, key='plotly_events_96')

    # Assigning values
    if selected_points_96:
        try:
            selected_wells_96 = [
                st.session_state[plate_key_96].iloc[p['pointIndex']]['Well']
                for p in selected_points_96
            ]
        except KeyError:
            st.error("Key 'pointIndex' is missing in selected points.")
            selected_wells_96 = []

        if selected_wells_96:
            if st.session_state.available_variables:
                var_to_assign_96 = st.selectbox("Select Variable to Assign", st.session_state.available_variables, key='assign_var_96')
                val_to_assign_96 = st.text_input(f"Value for '{var_to_assign_96}'", key='val_assign_96')
                if st.button("Assign Value to Selected Wells", key='btn_assign_96'):
                    if val_to_assign_96:
                        push_history('plate_96_data')
                        for w in selected_wells_96:
                            safe_assign_value(st.session_state[plate_key_96], w, var_to_assign_96, val_to_assign_96)
                            get_color(var_to_assign_96, val_to_assign_96)
                        # After interactive assignment, re-check for mixed columns
                        coerce_mixed_columns_to_string(st.session_state[plate_key_96])
                        st.success(f"Assigned '{val_to_assign_96}' to wells.")
                        st.session_state.refresh = not st.session_state.refresh
                    else:
                        st.warning("Enter a value first.")
            else:
                st.info("No variables found. Add one in the sidebar to begin.")

    st.subheader("Plate Data")
    display_cols_96 = ['Well'] + st.session_state.available_variables
    st.dataframe(st.session_state[plate_key_96][display_cols_96])

    st.header("Download Configuration")

    def to_excel(df):
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Rosettier_Plate_96')
        return buf.getvalue()

    def to_tsv(df):
        return df.to_csv(sep='\t', index=False).encode('utf-8')

    if st.session_state.available_variables:
        has_na_96 = st.session_state[plate_key_96][st.session_state.available_variables].isna().any().any()
    else:
        has_na_96 = False

    filename_96 = st.text_input("Base name for 96-well download:", value="Rosettier_Plate_96", key='dl_name_96')
    if not filename_96:
        filename_96 = "Rosettier_Plate_96"

    if has_na_96:
        proceed_dl_96 = st.checkbox("File has N/As. Continue anyway?", key='proceed_96')
    else:
        proceed_dl_96 = True

    if proceed_dl_96:
        if has_na_96:
            st.warning("Your file still contains N/As.")
        # Excel
        excel_96 = to_excel(st.session_state[plate_key_96][display_cols_96])
        st.download_button(
            label="Download Excel",
            data=excel_96,
            file_name=f"{filename_96}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        # TSV
        tsv_96 = to_tsv(st.session_state[plate_key_96][display_cols_96])
        st.download_button(
            label="Download TSV",
            data=tsv_96,
            file_name=f"{filename_96}.tsv",
            mime='text/tab-separated-values'
        )

############################
# Tab 2: 384-Well Plate
############################
with main_tabs[1]:
    st.subheader("Manage 384-Well Plate")
    st.info("Manage four separate 96-well plates to combine into a single 384-well plate.")

    plate_subtabs = st.tabs([f"Plate {i}" for i in range(1, 5)])

    def manage_plate(plate_num):
        plate_key = f'plate_{plate_num}_384_data'
        df_plate = st.session_state[plate_key]

        st.markdown(f"### Manage Plate {plate_num}")

        # Copy from another plate
        source_opts = [1, 2, 3, 4]
        if plate_num in source_opts:
            source_opts.remove(plate_num)
        chosen_src = st.selectbox(
            f"Copy from Plate ... to Plate {plate_num}",
            source_opts,
            key=f'plate_copy_select_{plate_num}'
        )
        if st.button(f"Copy from Plate {chosen_src}", key=f'btn_copy_{plate_num}'):
            src_key = f'plate_{chosen_src}_384_data'
            copy_plate_data(src_key, plate_key)

        # Undo
        if st.button(f"Undo Last Action (Plate {plate_num})", key=f'undo_plate_{plate_num}'):
            undo(plate_key)

        # Upload
        st.markdown("**Upload data (Excel/TSV/CSV)**")
        up_file = st.file_uploader(
            f"Upload Plate {plate_num} file",
            type=["xlsx","csv","tsv"],
            key=f'file_up_{plate_num}'
        )
        if up_file:
            file_ext = up_file.name.split('.')[-1].lower()
            try:
                if file_ext == "xlsx":
                    uploaded_df = pd.read_excel(up_file)
                else:
                    sep = "\t" if file_ext == "tsv" else ","
                    uploaded_df = pd.read_csv(up_file, sep=sep)

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
                            st.session_state.color_map[col] = {}

                    ensure_variables_exist(plate_key)
                    coerce_mixed_columns_to_string(st.session_state[plate_key])

                    st.success(f"Loaded data into Plate {plate_num}.")
                    st.session_state.refresh = not st.session_state.refresh
            except Exception as e:
                st.error(f"Error reading file: {e}")

        ensure_variables_exist(plate_key)
        coerce_mixed_columns_to_string(st.session_state[plate_key])

        # Visualization
        var_plate = st.selectbox(
            f"Select Variable to Visualize (Plate {plate_num})",
            ["None"] + st.session_state.available_variables,
            key=f'var_sel_plate_{plate_num}'
        )
        fig_p = create_plate_visualization(
            df_plate,
            current_variable=None if var_plate == "None" else var_plate
        )
        points_p = plotly_events(fig_p, select_event=True, override_height=600, key=f'plotly_ev_{plate_num}')

        # Assignment
        if points_p:
            try:
                sel_wells = [
                    st.session_state[plate_key].iloc[p['pointIndex']]['Well']
                    for p in points_p
                ]
            except KeyError:
                st.error("Key 'pointIndex' missing in points.")
                sel_wells = []

            if sel_wells:
                if st.session_state.available_variables:
                    var_assign = st.selectbox(
                        f"Assign which variable? (Plate {plate_num})",
                        st.session_state.available_variables,
                        key=f'assign_var_plate_{plate_num}'
                    )
                    val_assign = st.text_input(
                        f"Value for '{var_assign}' (Plate {plate_num})",
                        key=f'val_assign_{plate_num}'
                    )
                    if st.button(f"Assign Value (Plate {plate_num})", key=f'btn_assign_{plate_num}'):
                        if val_assign:
                            push_history(plate_key)
                            for w in sel_wells:
                                safe_assign_value(st.session_state[plate_key], w, var_assign, val_assign)
                                get_color(var_assign, val_assign)
                            coerce_mixed_columns_to_string(st.session_state[plate_key])
                            st.success(f"Assigned '{val_assign}' to wells in Plate {plate_num}.")
                            st.session_state.refresh = not st.session_state.refresh
                        else:
                            st.warning("Please enter a value first.")
                else:
                    st.info("No variables available. Add in the sidebar first.")

        # Show DataFrame
        st.markdown("**Plate Data**")
        disp_cols_sub = ['Well'] + st.session_state.available_variables
        st.dataframe(st.session_state[plate_key][disp_cols_sub])

    # Manage each sub-plate
    for i in range(1, 5):
        with plate_subtabs[i-1]:
            manage_plate(i)

    # Combine
    if st.button("Combine All Plates into 384-Well Plate", key='btn_combine_384'):
        plate1 = st.session_state['plate_1_384_data']
        plate2 = st.session_state['plate_2_384_data']
        plate3 = st.session_state['plate_3_384_data']
        plate4 = st.session_state['plate_4_384_data']
        combine_plates(plate1, plate2, plate3, plate4)
        # Also coerce the combined DataFrame
        coerce_mixed_columns_to_string(st.session_state['combined_plate_data'])
        st.success("Successfully combined the four 96-well plates into a 384-well plate.")
        st.session_state.refresh = not st.session_state.refresh

    # Undo for combined
    if st.button("Undo Last Action (Combined Plate)", key='undo_combined'):
        undo('combined_plate_data')

    # Show combined data
    combined_df = st.session_state['combined_plate_data']
    st.markdown("**Combined Plate Data**")
    disp_cols_combined = ['Well'] + st.session_state.available_variables

    ensure_variables_exist('combined_plate_data')
    coerce_mixed_columns_to_string(st.session_state['combined_plate_data'])

    st.dataframe(combined_df[disp_cols_combined])

    # Visualization
    var_combined = st.selectbox(
        "Select Variable to Visualize (Combined Plate)",
        ["None"] + st.session_state.available_variables,
        key='var_sel_combined'
    )
    fig_combined = create_combined_plate_visualization(
        combined_df,
        current_variable=None if var_combined == "None" else var_combined
    )
    points_combined = plotly_events(fig_combined, select_event=True, override_height=800, key=f'plotly_ev_combined')

    # Assign in combined
    if points_combined:
        try:
            sel_wells_combined = [
                st.session_state['combined_plate_data'].iloc[p['pointIndex']]['Well']
                for p in points_combined
            ]
        except KeyError:
            st.error("Key 'pointIndex' is missing in points.")
            sel_wells_combined = []

        if sel_wells_combined:
            if st.session_state.available_variables:
                var_assign_comb = st.selectbox(
                    "Assign which variable? (Combined Plate)",
                    st.session_state.available_variables,
                    key='assign_var_combined'
                )
                val_assign_comb = st.text_input(
                    f"Value for '{var_assign_comb}' (Combined Plate)",
                    key='val_assign_comb'
                )
                if st.button("Assign Value (Combined Plate)", key='btn_assign_comb'):
                    if val_assign_comb:
                        push_history('combined_plate_data')
                        for w in sel_wells_combined:
                            safe_assign_value(st.session_state['combined_plate_data'], w, var_assign_comb, val_assign_comb)
                            get_color(var_assign_comb, val_assign_comb)
                        coerce_mixed_columns_to_string(st.session_state['combined_plate_data'])
                        st.success(f"Assigned '{val_assign_comb}' to selected wells in Combined Plate.")
                        st.session_state.refresh = not st.session_state.refresh
                    else:
                        st.warning("Enter a value first.")
            else:
                st.info("No variables available. Add in the sidebar first.")

    # Download combined
    st.header("Download Combined 384-Well Configuration")

    def to_excel_combined(df):
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Rosettier_384_Plate')
        return buf.getvalue()

    def to_tsv_combined(df):
        return df.to_csv(sep='\t', index=False).encode('utf-8')

    if st.session_state.available_variables:
        has_na_combined = combined_df[st.session_state.available_variables].isna().any().any()
    else:
        has_na_combined = False

    dl_name_combined = st.text_input(
        "Base name for combined 384-well download:",
        value="Rosettier_384_Plate",
        key='dl_name_384'
    )
    if not dl_name_combined:
        dl_name_combined = "Rosettier_384_Plate"

    if has_na_combined:
        proceed_combined = st.checkbox("File has N/As. Continue anyway?", key='proceed_384')
    else:
        proceed_combined = True

    if proceed_combined:
        if has_na_combined:
            st.warning("File still has N/As.")
        excel_384 = to_excel_combined(combined_df[disp_cols_combined])
        st.download_button(
            label="Download Excel",
            data=excel_384,
            file_name=f"{dl_name_combined}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        tsv_384 = to_tsv_combined(combined_df[disp_cols_combined])
        st.download_button(
            label="Download TSV",
            data=tsv_384,
            file_name=f"{dl_name_combined}.tsv",
            mime='text/tab-separated-values'
        )
# Authorship statement
st.markdown("---")
st.markdown("""
**Authorship Statement**  
Developed by Ignacio de Quinto in 2025.  
Please contact me at idequintoc@gmail.com for licensing or collaboration inquiries.
""")
