import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from io import BytesIO
import random
from PIL import Image
import os
import sys

# Function to get the absolute path to resource, works for dev and for PyInstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Initialize rows and columns
rows = list("ABCDEFGH")
columns = list(range(1, 13))

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

# Function to parse 'Well' into row and column
def parse_well(well):
    row = well[0]
    column = int(well[1:])
    return row, column

# Function to create the plate visualization
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
    row_indices = {row: 8 - idx for idx, row in enumerate(rows)}
    plate_df['Y'] = plate_df['Parsed_Row'].map(row_indices)
    plate_df['X'] = plate_df['Parsed_Column']

    # Add well markers
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

    # Adjust layout margins to ensure all wells are visible
    fig.update_layout(
        autosize=True,  # Allow the figure to adjust automatically
        height=600,  # Maintain a reasonable height
        title="Rosettier",
        clickmode='event+select',
        dragmode='select',  # Allow box or lasso selection
        margin=dict(l=20, r=20, t=80, b=20)  # Reduce margins
    )

    return fig

# Initialize the plate DataFrame in session_state
if 'plate_data' not in st.session_state:
    # Create a list of 'Well' from A1 to H12
    wells = [f"{row}{col}" for row in rows for col in columns]
    plate_df = pd.DataFrame({'Well': wells})
    st.session_state.plate_data = plate_df

# Initialize the list of available variables and their color maps
if 'available_variables' not in st.session_state:
    st.session_state.available_variables = []

if 'color_map' not in st.session_state:
    st.session_state.color_map = {}

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
        # Add the new variable as a column with NaN values
        st.session_state.plate_data[new_variable] = pd.NA
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

# Select the variable to visualize
variable_to_visualize = st.selectbox(
    "Select Variable to Visualize",
    options=["None"] + st.session_state.available_variables
)

# Plate header
st.header("96-Well Plate Visualization")

# Create and display the figure
fig = create_plate_visualization(st.session_state.plate_data, 
                                 current_variable=None if variable_to_visualize == "None" else variable_to_visualize)
selected_points = plotly_events(fig, select_event=True, override_height=600)

# Handle selected points
if selected_points:
    # Extract the indices of the selected wells
    try:
        selected_wells = [
            st.session_state.plate_data.iloc[point['pointIndex']]['Well']
            for point in selected_points
        ]
    except KeyError:
        st.error("The key 'pointIndex' is not present in the selected points.")
        selected_wells = []

    if selected_wells:
        # Proceed to variable assignment
        if st.session_state.available_variables:
            variable_to_assign = st.selectbox(
                "Select Variable to Assign",
                options=st.session_state.available_variables
            )

            # Input the value to assign
            value_to_assign = st.text_input(f"Assign value for '{variable_to_assign}'")

            # Assign Value button
            if st.button("Assign Value"):
                if value_to_assign:
                    # Assign the value to the selected wells for the chosen variable
                    for well in selected_wells:
                        st.session_state.plate_data.loc[
                            st.session_state.plate_data['Well'] == well,
                            variable_to_assign
                        ] = value_to_assign
                        # Assign color to the variable's value
                        get_color(variable_to_assign, value_to_assign)
                    st.success(f"Value '{value_to_assign}' assigned to variable '{variable_to_assign}' for selected wells.")
                else:
                    st.warning("Please enter a value before assigning.")

# Display the updated DataFrame
st.subheader("Plate Data")

# Show only 'Well' and the added variables
display_columns = ['Well'] + st.session_state.available_variables
st.dataframe(st.session_state.plate_data[display_columns])

# Download the Excel and TSV files
st.header("Download Configuration")

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Rosettier_Plate')
    processed_data = output.getvalue()
    return processed_data

def to_tsv(df):
    return df.to_csv(sep='\t', index=False).encode('utf-8')

# Check for NAs in the DataFrame (excluding the 'Well' column)
if st.session_state.available_variables:
    na_exists = st.session_state.plate_data[st.session_state.available_variables].isna().any().any()
else:
    na_exists = False  # No variables added, no NAs

# Input for download file name
default_filename = "Rosettier_Plate"
download_filename = st.text_input("Enter the base name for your download files:", value=default_filename)

if not download_filename:
    download_filename = "Rosettier_Plate"  # Fallback to default if empty

if na_exists:
    proceed = st.checkbox("There are N/As in your file, do you want to continue?")
else:
    proceed = True  # No NAs, proceed to show download buttons

# Display download buttons only if no NAs or user has confirmed to proceed
if proceed:
    # Display warning only if N/As exist
    if na_exists:
        st.warning("There are N/As in your file.")
    
    # Excel download
    excel_data = to_excel(st.session_state.plate_data[display_columns])
    excel_file_name = f"{download_filename}.xlsx"
    st.download_button(
        label="Download Excel",
        data=excel_data,
        file_name=excel_file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    
    # TSV download
    tsv_data = to_tsv(st.session_state.plate_data[display_columns])
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

