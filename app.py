import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from io import BytesIO
import random
from PIL import Image
import os
import sys

# Function to get the absolute path to resources, works for dev and for PyInstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load the favicon
favicon_path = resource_path("assets/icon.ico")
favicon = Image.open(favicon_path) if os.path.exists(favicon_path) else None

# Set page config
st.set_page_config(
    page_title="Rosettier",
    page_icon=favicon,  # Use the loaded favicon
    layout="centered"
)

# Initialize rows and columns for 96-well plates
rows_96 = list("ABCDEFGH")
columns_96 = list(range(1, 13))

# Initialize rows and columns for 384-well plates
rows_384 = list("ABCDEFGHIJKLMNOP")
columns_384 = list(range(1, 25))

# Function to generate random hexadecimal colors
def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Function to get or assign a color to a specific value of a variable
def get_color(variable, value):
    """Retrieves the color assigned to a specific value of a variable.
    If the value does not have an assigned color, generates a new one."""
    if variable not in st.session_state.color_map:
        st.session_state.color_map[variable] = {}
    
    if value not in st.session_state.color_map[variable]:
        st.session_state.color_map[variable][value] = generate_random_color()
    
    return st.session_state.color_map[variable][value]

# Function to parse 'Well' into row and column for 96-well plate
def parse_well(well):
    row = well[0]
    column = int(well[1:])
    return row, column

# Function to parse 'Well' into row and column for 384-well plate
def parse_well_384(well):
    row = well[0]
    column = int(well[1:])
    return row, column

# Function to create the plate visualization for 96-well plates
def create_plate_visualization(plate_df, current_variable=None):
    fig = go.Figure()

    # Define colors based on the current variable
    if current_variable and current_variable in plate_df.columns:
        color_map = st.session_state.color_map.get(current_variable, {})
        # Assign colors based on the current variable's values
        colors = plate_df['Well'].map(
            lambda well: get_color(current_variable, plate_df.loc[plate_df['Well'] == well, current_variable].values[0]) 
            if not pd.isna(plate_df.loc[plate_df['Well'] == well, current_variable].values[0]) 
            else 'lightgray'
        )
    else:
        # If no variable is selected, color all wells with a base color
        colors = 'lightgray'

    # Extract row and column for positioning
    plate_df['Parsed_Row'], plate_df['Parsed_Column'] = zip(*plate_df['Well'].map(parse_well))

    # Convert rows to numerical indices for the Y-axis (A=8, B=7, ..., H=1)
    row_indices = {row: 8 - idx for idx, row in enumerate(rows_96)}
    plate_df['Y'] = plate_df['Parsed_Row'].map(row_indices)
    plate_df['X'] = plate_df['Parsed_Column']

    # Add well markers with labels
    fig.add_trace(
        go.Scatter(
            x=plate_df['X'],
            y=plate_df['Y'],
            mode='markers+text',
            marker=dict(
                size=40,
                color=colors,
                line=dict(width=1, color='black'),
                opacity=0.8  # Add transparency to markers
            ),
            text=plate_df['Well'],  # Visible text on the markers
            customdata=plate_df.index,  # Use index to map to DataFrame
            textposition="middle center",
            hoverinfo='text',
            name='Wells'
        )
    )

    # Configure axes with fixed ranges and adjusted margins
    fig.update_xaxes(range=[0.5, 12.5], dtick=1, showgrid=True, zeroline=False, showticklabels=True)
    fig.update_yaxes(range=[0.5, 8.5], dtick=1, showgrid=True, zeroline=False, showticklabels=True)

    # Add vertical lines for columns with thinner, dashed lines and transparency
    for i in range(1, 13):
        fig.add_shape(
            type="line",
            x0=i, y0=0.5, x1=i, y1=8.5,
            line=dict(color="black", width=1, dash='dot'),
            opacity=0.5  # Add transparency to lines
        )

    # Add horizontal lines for rows with thinner, dashed lines and transparency
    for i in range(1, 9):
        fig.add_shape(
            type="line",
            x0=0.5, y0=i, x1=12.5, y1=i,
            line=dict(color="black", width=1, dash='dot'),
            opacity=0.5  # Add transparency to lines
        )

    # Set plotly chart backgrounds to white to eliminate duplication with Streamlit's background
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=True,  # Allow the figure to adjust automatically
        height=600,  # Maintain a reasonable height
        title="Rosettier - 96-Well Plate",
        clickmode='event+select',
        dragmode='select',  # Allow box or lasso selection
        margin=dict(l=20, r=20, t=80, b=20)  # Reduce margins
    )

    return fig

# Function to create the plate visualization for 384-well plates
def create_combined_plate_visualization(combined_plate_df, current_variable=None):
    fig = go.Figure()
    
    # Define colors based on the current variable
    if current_variable and current_variable in combined_plate_df.columns:
        color_map = st.session_state.color_map.get(current_variable, {})
        # Assign colors based on the current variable's values
        colors = combined_plate_df['Well'].map(
            lambda well: get_color(current_variable, combined_plate_df.loc[combined_plate_df['Well'] == well, current_variable].values[0]) 
            if not pd.isna(combined_plate_df.loc[combined_plate_df['Well'] == well, current_variable].values[0]) 
            else 'lightgray'
        )
    else:
        # If no variable is selected, color all wells with a base color
        colors = 'lightgray'
    
    # Extract row and column for positioning
    combined_plate_df['Parsed_Row'], combined_plate_df['Parsed_Column'] = zip(*combined_plate_df['Well'].map(parse_well_384))
    
    # Convert rows to numerical indices for the Y-axis (A=16, B=15, ..., P=1)
    row_indices = {row: 16 - idx for idx, row in enumerate(rows_384)}
    combined_plate_df['Y'] = combined_plate_df['Parsed_Row'].map(row_indices)
    combined_plate_df['X'] = combined_plate_df['Parsed_Column']
    
    # Add well markers with labels
    fig.add_trace(
        go.Scatter(
            x=combined_plate_df['X'],
            y=combined_plate_df['Y'],
            mode='markers+text',
            marker=dict(
                size=15,  # Smaller size for 384-well plate
                color=colors,
                line=dict(width=0.5, color='black'),
                opacity=0.8  # Add transparency to markers
            ),
            text=combined_plate_df['Well'],  # Visible text on the markers
            customdata=combined_plate_df.index,  # Use index to map to DataFrame
            textposition="middle center",
            hoverinfo='text',
            name='Wells'
        )
    )
    
    # Configure axes with fixed ranges and adjusted margins
    fig.update_xaxes(range=[0.5, 24.5], dtick=1, showgrid=True, zeroline=False, showticklabels=True)
    fig.update_yaxes(range=[0.5, 16.5], dtick=1, showgrid=True, zeroline=False, showticklabels=True)
    
    # Add vertical lines for columns with thinner, dashed lines and transparency
    for i in range(1, 25):
        fig.add_shape(
            type="line",
            x0=i, y0=0.5, x1=i, y1=16.5,
            line=dict(color="black", width=0.5, dash='dot'),
            opacity=0.5  # Add transparency to lines
        )
    
    # Add horizontal lines for rows with thinner, dashed lines and transparency
    for i in range(1, 17):
        fig.add_shape(
            type="line",
            x0=0.5, y0=i, x1=24.5, y1=i,
            line=dict(color="black", width=0.5, dash='dot'),
            opacity=0.5  # Add transparency to lines
        )
    
    # Set plotly chart backgrounds to white to eliminate duplication with Streamlit's background
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=True,  # Allow the figure to adjust automatically
        height=800,  # Increased height for 384-well plate
        title="Rosettier - 384-Well Plate",
        clickmode='event+select',
        dragmode='select',  # Allow box or lasso selection
        margin=dict(l=20, r=20, t=80, b=20)  # Reduce margins
    )
    
    return fig

# Function to combine four 96-well plates into one 384-well plate based on interleaved mapping
def combine_plates(plate1, plate2, plate3, plate4):
    """
    Combine four 96-well plates into one 384-well plate based on the interleaved pattern:
    
    - Plate 1: Maps to odd rows (A, C, E, ...) and odd columns (1, 3, 5, ...)
    - Plate 2: Maps to odd rows (A, C, E, ...) and even columns (2, 4, 6, ...)
    - Plate 3: Maps to even rows (B, D, F, ...) and odd columns (1, 3, 5, ...)
    - Plate 4: Maps to even rows (B, D, F, ...) and even columns (2, 4, 6, ...)
    
    This ensures that:
    - Plate1 A1 -> 384 A1
    - Plate2 A1 -> 384 A2
    - Plate3 A1 -> 384 B1
    - Plate4 A1 -> 384 B2
    """
    combined_plate_df = st.session_state.combined_plate_data.copy()
    
    # Define the mapping for each plate
    plate_mappings = {
        1: {
            'rows': rows_96,  # A-H
            'cols': columns_96,  # 1-12
            'plate_data': plate1,
            'row_offset': 0,
            'col_offset': 0
        },
        2: {
            'rows': rows_96,
            'cols': columns_96,
            'plate_data': plate2,
            'row_offset': 0,
            'col_offset': 1  # Even columns
        },
        3: {
            'rows': rows_96,
            'cols': columns_96,
            'plate_data': plate3,
            'row_offset': 1,  # Even rows
            'col_offset': 0
        },
        4: {
            'rows': rows_96,
            'cols': columns_96,
            'plate_data': plate4,
            'row_offset': 1,  # Even rows
            'col_offset': 1  # Even columns
        },
    }
    
    for plate_num, mapping in plate_mappings.items():
        plate_df = mapping['plate_data']
        row_offset = mapping['row_offset']
        col_offset = mapping['col_offset']
        
        for idx, row in plate_df.iterrows():
            well = row['Well']
            plate_row_label, plate_col_label = parse_well(well)
            
            # Calculate numerical indices for rows and columns
            try:
                plate_row_idx = rows_96.index(plate_row_label)
            except ValueError:
                st.warning(f"Invalid row label '{plate_row_label}' in Plate {plate_num}. Skipping.")
                continue
            
            if plate_col_label not in columns_96:
                st.warning(f"Invalid column label '{plate_col_label}' in Plate {plate_num}. Skipping.")
                continue
            
            # Calculate 384-well plate row and column
            # Each 96-well plate corresponds to a 2x2 block in the 384-well plate
            # Rows: A-H in 96 map to A-P in 384 (each 96 row maps to two 384 rows)
            # Columns: 1-12 in 96 map to 1-24 in 384 (each 96 column maps to two 384 columns)
            combined_row_idx = plate_row_idx * 2 + row_offset  # 0-based index
            combined_col_idx = (plate_col_label -1) *2 + col_offset +1  # 1-based index
            
            # Get row label for 384-well plate
            if combined_row_idx >= len(rows_384):
                st.warning(f"Row index {combined_row_idx} out of range for 384-well plate. Skipping.")
                continue
            combined_row_label = rows_384[combined_row_idx]
            
            # Get column label for 384-well plate
            if combined_col_idx > max(columns_384):
                st.warning(f"Column index {combined_col_idx} out of range for 384-well plate. Skipping.")
                continue
            combined_col_label = combined_col_idx
            
            # Construct combined well identifier
            combined_well = f"{combined_row_label}{combined_col_label}"
            
            # Assign all available variables
            for variable in st.session_state.available_variables:
                value = row.get(variable, pd.NA)
                combined_plate_df.loc[combined_plate_df['Well'] == combined_well, variable] = value
    
    # Update the session state with the combined data
    st.session_state.combined_plate_data = combined_plate_df

# Initialize the plate DataFrames in session_state
def initialize_plates():
    # Initialize single 96-well plate
    if 'plate_96_data' not in st.session_state:
        wells_96 = [f"{row}{col}" for row in rows_96 for col in columns_96]
        plate_96_df = pd.DataFrame({'Well': wells_96, 'Value': pd.NA})
        st.session_state.plate_96_data = plate_96_df

    # Initialize four 96-well plates for 384-well combination
    for i in range(1, 5):
        plate_key = f'plate_{i}_384_data'
        if plate_key not in st.session_state:
            wells_96_384 = [f"{row}{col}" for row in rows_96 for col in columns_96]
            plate_df = pd.DataFrame({'Well': wells_96_384, 'Value': pd.NA})
            # Initialize additional variables as NaN
            for var in st.session_state.get('available_variables', []):
                plate_df[var] = pd.NA
            st.session_state[plate_key] = plate_df

    # Initialize the combined 384-well plate
    if 'combined_plate_data' not in st.session_state:
        combined_wells = [f"{row}{col}" for row in rows_384 for col in columns_384]
        combined_plate_df = pd.DataFrame({'Well': combined_wells, 'Value': pd.NA})
        # Initialize additional variables as NaN
        for var in st.session_state.get('available_variables', []):
            combined_plate_df[var] = pd.NA
        st.session_state.combined_plate_data = combined_plate_df

# Call the initialization function
initialize_plates()

# Initialize the list of available variables and their color maps
if 'available_variables' not in st.session_state:
    st.session_state.available_variables = []

if 'color_map' not in st.session_state:
    st.session_state.color_map = {}

# Initialize the refresh trigger (if using the workaround)
if 'refresh' not in st.session_state:
    st.session_state.refresh = False

# Load logo image
logo_path = resource_path("assets/logo.png")
try:
    logo = Image.open(logo_path)
except FileNotFoundError:
    logo = None  # Handle missing logo gracefully

# Application header with larger logo and name "Rosettier" positioned to the right
if logo:
    # Create three columns with relative widths: left, center, right
    col1, col2, col3 = st.columns([1, 2, 1.5])
    with col1:
        pass  # Empty column for spacing
    with col2:
        st.markdown("<h1 style='text-align: left;'>Rosettier</h1>", unsafe_allow_html=True)
    with col3:
        st.image(logo, width=300)  # Increased width for a bigger logo
else:
    st.title("Rosettier")

# Sidebar for variable management
st.sidebar.header("Variable Management")
new_variable = st.sidebar.text_input("Add New Variable")
if st.sidebar.button("Add Variable") and new_variable:
    if new_variable not in st.session_state.available_variables:
        st.session_state.available_variables.append(new_variable)
        # Initialize the color_map for the new variable
        st.session_state.color_map[new_variable] = {}
        # Add the new variable as a column with NaN values to all plates
        st.session_state.plate_96_data[new_variable] = pd.NA
        for i in range(1, 5):
            plate_key = f'plate_{i}_384_data'
            st.session_state[plate_key][new_variable] = pd.NA
        st.session_state.combined_plate_data[new_variable] = pd.NA
        st.sidebar.success(f"Variable '{new_variable}' added.")
    else:
        st.sidebar.warning("The variable already exists.")

# Display the legend of variables with their colors and values
st.sidebar.header("Variables Legend")
for var, values in st.session_state.color_map.items():
    # Display only the variable name in the legend
    st.sidebar.markdown(f"**{var}**")
    for value, color in values.items():
        st.sidebar.markdown(f"<span style='color:{color}'>⬤</span> {value}", unsafe_allow_html=True)
    st.sidebar.markdown("---")

# Display available variables
st.sidebar.write("Available Variables:")
for var in st.session_state.available_variables:
    st.sidebar.write(f"- {var}")

# Create two main tabs: "96-Well Plate" and "384-Well Plate"
main_tabs = st.tabs(["96-Well Plate", "384-Well Plate"])

##########################
# Tab 1: 96-Well Plate
##########################
with main_tabs[0]:
    st.subheader("Manage 96-Well Plate")
    
    # Display the plate visualization and handle interactions using plotly_events
    plate_key = 'plate_96_data'
    plate_df = st.session_state[plate_key]
    
    selected_variable = st.selectbox(
        "Select Variable to Visualize",
        options=["None"] + st.session_state.available_variables,
        key='select_var_96'
    )
    fig = create_plate_visualization(
        plate_df, 
        current_variable=None if selected_variable == "None" else selected_variable
    )
    
    # Use plotly_events to render the figure and capture interactions
    selected_points = plotly_events(fig, select_event=True, override_height=600, key='plotly_events_96')

    if selected_points:
        try:
            selected_wells = [
                st.session_state[plate_key].iloc[point['pointIndex']]['Well']
                for point in selected_points
            ]
        except KeyError:
            st.error("The key 'pointIndex' is not present in the selected points.")
            selected_wells = []

        if selected_wells:
            # Select variable to assign
            if st.session_state.available_variables:
                variable_to_assign = st.selectbox(
                    "Select Variable to Assign",
                    options=st.session_state.available_variables,
                    key='select_assign_var_96'
                )

                # Input for value
                value_to_assign = st.text_input(f"Assign value for '{variable_to_assign}'", key='input_assign_val_96')

                # Assign Value button
                if st.button("Assign Value to Selected Wells", key='assign_btn_96'):
                    if value_to_assign:
                        for well in selected_wells:
                            st.session_state[plate_key].loc[
                                st.session_state[plate_key]['Well'] == well,
                                variable_to_assign
                            ] = value_to_assign
                            # Assign color to the variable's value
                            get_color(variable_to_assign, value_to_assign)
                        st.success(f"Value '{value_to_assign}' assigned to variable '{variable_to_assign}' for selected wells.")
                        # Refresh the plot after assignment
                        st.session_state.refresh = not st.session_state.refresh
                    else:
                        st.warning("Please enter a value before assigning.")
            else:
                st.info("Please add variables from the sidebar to assign values.")

    # Display the updated DataFrame
    st.subheader("Plate Data")
    
    # Show only 'Well' and the added variables
    display_columns = ['Well'] + st.session_state.available_variables
    st.dataframe(st.session_state[plate_key][display_columns])

    # Download the Excel and TSV files for 96-well plate
    st.header("Download Configuration")

    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Rosettier_Plate_96')
        processed_data = output.getvalue()
        return processed_data

    def to_tsv(df):
        return df.to_csv(sep='\t', index=False).encode('utf-8')

    # Check for NAs in the DataFrame (excluding the 'Well' column)
    if st.session_state.available_variables:
        na_exists = st.session_state.plate_96_data[st.session_state.available_variables].isna().any().any()
    else:
        na_exists = False  # No variables added, no NAs

    # Input for download file name
    default_filename = "Rosettier_Plate_96"
    download_filename = st.text_input("Enter the base name for your download files:", value=default_filename, key='download_filename_96')

    if not download_filename:
        download_filename = "Rosettier_Plate_96"  # Fallback to default if empty

    if na_exists:
        proceed = st.checkbox("There are N/As in your file, do you want to continue?", key='proceed_download_96')
    else:
        proceed = True  # No NAs, proceed to show download buttons

    # Display download buttons only if no NAs or user has confirmed to proceed
    if proceed:
        # Display warning only if N/As exist
        if na_exists:
            st.warning("There are N/As in your file.")

        # Excel download
        excel_data = to_excel(st.session_state.plate_96_data[display_columns])
        excel_file_name = f"{download_filename}.xlsx"
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=excel_file_name,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        # TSV download
        tsv_data = to_tsv(st.session_state.plate_96_data[display_columns])
        tsv_file_name = f"{download_filename}.tsv"
        st.download_button(
            label="Download TSV",
            data=tsv_data,
            file_name=tsv_file_name,
            mime='text/tab-separated-values'
        )
    else:
        # Do not show any message; the checkbox serves as the prompt
        pass

##########################
# Tab 2: 384-Well Plate
##########################
with main_tabs[1]:
    st.subheader("Manage 384-Well Plate")
    
    st.info("Manage four separate 96-well plates to combine into a single 384-well plate.")
    
    # Create sub-tabs for each of the four 96-well plates
    plate_subtabs = st.tabs([f"Plate {i}" for i in range(1, 5)])

    # Function to manage individual 96-well plates within the 384-well Plate tab
    def manage_plate(plate_num):
        plate_key = f'plate_{plate_num}_384_data'
        plate_df = st.session_state[plate_key]
        
        st.markdown(f"### Manage Plate {plate_num}")
        
        # Display the plate visualization and handle interactions using plotly_events
        selected_variable = st.selectbox(
            f"Select Variable to Visualize (Plate {plate_num})",
            options=["None"] + st.session_state.available_variables,
            key=f'select_var_plate_{plate_num}'
        )
        fig = create_plate_visualization(
            plate_df, 
            current_variable=None if selected_variable == "None" else selected_variable
        )
        
        # Use plotly_events to render the figure and capture interactions
        selected_points = plotly_events(fig, select_event=True, override_height=600, key=f'plotly_events_plate_{plate_num}')

        if selected_points:
            try:
                selected_wells = [
                    st.session_state[plate_key].iloc[point['pointIndex']]['Well']
                    for point in selected_points
                ]
            except KeyError:
                st.error("The key 'pointIndex' is not present in the selected points.")
                selected_wells = []

            if selected_wells:
                # Select variable to assign
                if st.session_state.available_variables:
                    variable_to_assign = st.selectbox(
                        f"Select Variable to Assign (Plate {plate_num})",
                        options=st.session_state.available_variables,
                        key=f'select_assign_var_plate_{plate_num}'
                    )

                    # Input for value
                    value_to_assign = st.text_input(f"Assign value for '{variable_to_assign}'", key=f'input_assign_val_plate_{plate_num}')

                    # Assign Value button
                    if st.button(f"Assign Value to Selected Wells (Plate {plate_num})", key=f'assign_btn_plate_{plate_num}'):
                        if value_to_assign:
                            for well in selected_wells:
                                st.session_state[plate_key].loc[
                                    st.session_state[plate_key]['Well'] == well,
                                    variable_to_assign
                                ] = value_to_assign
                                # Assign color to the variable's value
                                get_color(variable_to_assign, value_to_assign)
                            st.success(f"Value '{value_to_assign}' assigned to variable '{variable_to_assign}' for selected wells in Plate {plate_num}.")
                            # Refresh the plot after assignment
                            st.session_state.refresh = not st.session_state.refresh
                        else:
                            st.warning("Please enter a value before assigning.")
                else:
                    st.info("Please add variables from the sidebar to assign values.")

        # Display the updated DataFrame
        st.markdown("**Plate Data**")
        
        # Show only 'Well' and the added variables
        display_columns_plate = ['Well'] + st.session_state.available_variables
        st.dataframe(st.session_state[plate_key][display_columns_plate])

    # Manage each plate within its respective sub-tab
    for i in range(1, 5):
        with plate_subtabs[i-1]:
            manage_plate(i)

    # Button to combine plates into 384-well plate
    if st.button("Combine All Plates into 384-Well Plate", key='combine_all_384'):
        # Retrieve the four 96-well plate DataFrames from session_state
        plate1 = st.session_state['plate_1_384_data']
        plate2 = st.session_state['plate_2_384_data']
        plate3 = st.session_state['plate_3_384_data']
        plate4 = st.session_state['plate_4_384_data']
        
        # Call the revised combine_plates function
        combine_plates(plate1, plate2, plate3, plate4)
        
        # Provide feedback to the user
        st.success("Successfully combined the four 96-well plates into a 384-well plate.")
        
        # Refresh the combined plate visualization
        st.session_state.refresh = not st.session_state.refresh

    # Display the combined plate data
    combined_plate_df = st.session_state['combined_plate_data']
    st.markdown("**Combined Plate Data**")
    display_columns_combined = ['Well'] + st.session_state.available_variables
    st.dataframe(combined_plate_df[display_columns_combined])

    # Visualization of the combined 384-well plate
    selected_variable_combined = st.selectbox(
        "Select Variable to Visualize (Combined Plate)",
        options=["None"] + st.session_state.available_variables,
        key='select_var_combined_384'
    )
    fig_combined = create_combined_plate_visualization(
        combined_plate_df, 
        current_variable=None if selected_variable_combined == "None" else selected_variable_combined
    )
    
    # Use plotly_events to render the figure and capture interactions
    selected_points_combined = plotly_events(fig_combined, select_event=True, override_height=800, key='plotly_events_combined_384')

    if selected_points_combined:
        try:
            selected_wells_combined = [
                st.session_state['combined_plate_data'].iloc[point['pointIndex']]['Well']
                for point in selected_points_combined
            ]
        except KeyError:
            st.error("The key 'pointIndex' is not present in the selected points.")
            selected_wells_combined = []

        if selected_wells_combined:
            # Select variable to assign
            if st.session_state.available_variables:
                variable_to_assign_combined = st.selectbox(
                    "Select Variable to Assign (Combined Plate)",
                    options=st.session_state.available_variables,
                    key='select_assign_var_combined_384'
                )

                # Input for value
                value_to_assign_combined = st.text_input(f"Assign value for '{variable_to_assign_combined}'", key='input_assign_val_combined_384')

                # Assign Value button
                if st.button("Assign Value to Selected Wells (Combined Plate)", key='assign_btn_combined_384'):
                    if value_to_assign_combined:
                        for well in selected_wells_combined:
                            st.session_state['combined_plate_data'].loc[
                                st.session_state['combined_plate_data']['Well'] == well,
                                variable_to_assign_combined
                            ] = value_to_assign_combined
                            # Assign color to the variable's value
                            get_color(variable_to_assign_combined, value_to_assign_combined)
                        st.success(f"Value '{value_to_assign_combined}' assigned to variable '{variable_to_assign_combined}' for selected wells in Combined Plate.")
                        # Refresh the plot after assignment
                        st.session_state.refresh = not st.session_state.refresh
                    else:
                        st.warning("Please enter a value before assigning.")
            else:
                st.info("Please add variables from the sidebar to assign values.")

    # Download the combined plate
    st.header("Download Combined 384-Well Configuration")

    def to_excel_combined(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Rosettier_384_Plate')
        processed_data = output.getvalue()
        return processed_data

    def to_tsv_combined(df):
        return df.to_csv(sep='\t', index=False).encode('utf-8')

    # Check for NAs in the DataFrame (excluding the 'Well' column)
    if st.session_state.available_variables:
        na_exists_combined = st.session_state.combined_plate_data[st.session_state.available_variables].isna().any().any()
    else:
        na_exists_combined = False  # No variables added, no NAs

    # Input for download file name
    default_filename_combined = "Rosettier_384_Plate"
    download_filename_combined = st.text_input("Enter the base name for your combined download files:", value=default_filename_combined, key='download_filename_combined_384')

    if not download_filename_combined:
        download_filename_combined = "Rosettier_384_Plate"  # Fallback to default if empty

    if na_exists_combined:
        proceed_combined = st.checkbox("There are N/As in your file, do you want to continue?", key='proceed_download_combined_384')
    else:
        proceed_combined = True  # No NAs, proceed to show download buttons

    # Display download buttons only if no NAs or user has confirmed to proceed
    if proceed_combined:
        # Display warning only if N/As exist
        if na_exists_combined:
            st.warning("There are N/As in your file.")

        # Excel download
        excel_data_combined = to_excel_combined(st.session_state.combined_plate_data[display_columns_combined])
        excel_file_name_combined = f"{download_filename_combined}.xlsx"
        st.download_button(
            label="Download Excel",
            data=excel_data_combined,
            file_name=excel_file_name_combined,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        # TSV download
        tsv_data_combined = to_tsv_combined(st.session_state.combined_plate_data[display_columns_combined])
        tsv_file_name_combined = f"{download_filename_combined}.tsv"
        st.download_button(
            label="Download TSV",
            data=tsv_data_combined,
            file_name=tsv_file_name_combined,
            mime='text/tab-separated-values'
        )
    else:
        # Do not show any message; the checkbox serves as the prompt
        pass
