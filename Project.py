# ID5059 - Knowledge Discovery & Data Mining
# Coursework Assignment 1 - Individual
# Deadline: Friday 24th February 2023 (week 6), 9pm


################################################################################
###################### IMPORT LIBRARIES, DATA FRAME ############################
################################################################################

# Load libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zlib import crc32
# To split the training from testing sections
from sklearn.model_selection import StratifiedShuffleSplit
# Correlation matrix plots
from pandas.plotting import scatter_matrix
# Import to change the character variables to categorical
from sklearn.preprocessing import OrdinalEncoder
# Import to change the character variables to numerical
from sklearn.preprocessing import LabelEncoder
# This line runs on Jupyter Notebook only
#get_ipython().run_line_magic("matplotlib", "inline")

# Read the dataset from a file
# Specify type "string" for the "bed" and "dealer_zip" attributes to avoid errors interpreting them as numbers
cars = pd.read_csv("/Users/ernakuginyte/Documents/ID5059 Knowledge Discovery and Data Mining/Assignments/Project 1/data/small/used_cars_data_small_0.csv", \
    dtype = {"bed": "string", "dealer_zip": "string"})


################################################################################
############################# EXPLORE THE DATA #################################
################################################################################

# Clear the maximum number of columns to be displayed, so that all will be visible
pd.set_option("display.max_columns", None)
# Check the basic statistics of the data
cars.describe()
# Display the data frame
print(cars)
# Explore numeric columns visually
cars.hist(bins = 20, figsize = (20, 15)) 


################################################################################
############################# DATA WRANGLING ###################################
################################################################################

################################ DUPLICATES ####################################

# Firstly, check for duplicates
print(cars.duplicated().sum()) 
# Drop duplicates
cars = cars.drop_duplicates()


########################## DEAL WITH MISSING VALUES ############################ 

# Deal with the missing values.
# Check which columns have only missing values and drop them.
cars = cars.dropna(axis = 1, how = "all")

# Check which columns have some missing values.
# First find the number of missing values in each column.
na_columns = cars.isna().sum()
# Save the column names with missing values
cols_with_na = list(na_columns[na_columns > 0].index)
# Print the column names with number of missing values 
#    out of total number of rows in the data set
for col in cols_with_na:
    print(f"{col}: {na_columns[col]} missing values out of {cars.shape[0]}")

### clean_df - Cleans dataframe columns with missing data;
###            if more than 25% of rows are missing, the whole column will be dropped;
###            else, add median numeric values to the numeric data, string or object data won"t be modified.  
#   INPUT:
#              df - data frame with latent values.
#   OUTPUT:
#              df_cleaned - cleaned data frame.
def clean_df(df):
    
    # Firstly check if the input value is of correct type
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input argument incorrect, it should be a Data Frame! :)")
    # Check if the data frame has any missing values
    if not df.isnull().values.any():
        raise ValueError("Your data frame does not have any missing values, hooray! :)")    
        
    # Find the number of missing values in each column
    na_columns = df.isna().sum()

    # Calculate the percentage of missing values in each column
    na_columns_percent = na_columns / df.shape[0]

    # Get the names of the columns with more than 25% missing values
    cols_to_drop = list(na_columns_percent[na_columns_percent > 0.25].index)

    # Drop the columns with more than 25% missing values
    df_cleaned = df.drop(cols_to_drop, axis = 1)

    # Replace the missing values in the remaining columns with the median value
    # if the column is numeric, otherwise leave the values as is
    for col in df_cleaned.columns:
        # First check if the data type is numeric
        if df_cleaned[col].dtype in ["float64", "int64"]:
            # Fill the missing values with medians
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace = True)
            
    # Return the cleaned data frame
    return df_cleaned

# Apply the function and save the cleaned data frame
cars = clean_df(df = cars)
# Print names of non-numeric columns with the number of missing values.
# Get the number of missing values in each non-numeric column
na_string_columns = cars.isna().sum()

# Get the names of non-numeric columns with missing values
cols_with_na = list(na_string_columns[na_string_columns > 0].index)

# Print the names of non-numeric columns with the number of missing values
for col in cols_with_na:
    print(f"{col}: {na_string_columns[col]} missing values out of {cars.shape[0]}")


####################### ADD MEAN TO THE MISSING VALUES #########################

### fill_na_mean - Take away the " in", "in", " gal" measurement units from data set 
###                variables and converts it to a numeric object;
###                Fills latent variables with mean value.
#   INPUT:  
#                  variables_inches - names of variables that have " in" inches measurement;
#                                     also has some missing values.
#   OUTPUT: 
#                  data - cleaned data frame.
def fill_na_mean(data, vars):

    # Firstly check if the input value is of correct type
    # Check if data is a data frame?
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input argument incorrect, it should be a Data Frame! :)")
    # Check if the data frame has any missing values
    if not data.isnull().values.any():
        raise ValueError("Your data frame does not have any missing values, hooray! :)")
    # Check if the variables_inches vector is not empty
    if len(vars) == 0:
        raise ValueError("Variable list is empty! :)")

        # For each variable in the variables_inches list
    for variable in vars:

        # Check if the variable type is object
        if data[variable].dtype == object:
            # Remove the "in" and "gal" from the values
            data[variable] = data[variable].str.replace(" in", "").str.replace(" gal", "").\
                str.replace("in", "").str.replace("--", "")
            # Convert the string variables to numeric
            data[variable] = pd.to_numeric(data[variable], errors = "coerce")
        # Fill missing values with the mean
        data[variable] = data[variable].fillna(data[variable].mean())

    # Return the wrangled data frame
    return data

# Create a list of variables that can 
variables_inches = ["back_legroom", "front_legroom", "height", "length", \
    "wheelbase", "width", "fuel_tank_volume"]

# Apply the fill_na_mean function and save the cleaned data frame
cars = fill_na_mean(data = cars, vars = variables_inches)

###### FILL THIS ##########
# Fill the latent variables in the "transmission" variable
#cars["transmission"] = cars["transmission"].apply(lambda x: 1 if x == "yes" else 0)

# Check if the engine_type column is exactly the same as the engine_cylinder?
cars["engine_type"] = cars["engine_cylinders"].str.split(" ", expand=True)[0]

# The variables that have too many missing values or cannot be filled manually
#   as they are character variables; main_picture_url useless for this analysis
columns_to_drop = ["engine_cylinders", "fuel_type", "main_picture_url", "trimId", \
    "major_options", "power", "torque", "transmission_display", "trim_name", \
        "wheel_system", "wheel_system_display", "main_picture_url", "description", \
            "maximum_seating", "engine_type", "vin"]
# Drop the unwanted variables
cars.drop(columns_to_drop, axis = 1, inplace = True)

# Delete rows that have a missing value in "body_type"
cars = cars.dropna(subset=["body_type"])

# Deal with missing values in listing_color and exterior_color variables.
# Replace the "UNKNOWN" values in listing_color with values from exterior_color.
cars.loc[cars["listing_color"] == "UNKNOWN", "listing_color"] = cars["exterior_color"]
# Replace the "None" values in exterior_color with values in listing_color
cars.loc[cars["exterior_color"] == "None", "exterior_color"] = cars["listing_color"]
# Delete rows where listing_color is UNKNOWN and exterior_color is None
cars = cars.loc[(cars["listing_color"] != "UNKNOWN") | (cars["exterior_color"].notna())]

# Check which variables have missing values
print(cars.isnull().sum())

################################################################################
############################## ADD EXTRA VARIABLES #############################
################################################################################

# Add an extra variable mileage_per_owner
#cars["mileage_per_owner"] = cars["mileage"] / cars["owner_count"]
# Add an extra variable savings_per_day
#cars_train["savings_per_day"] = cars_train["savings_amount"] / cars_train["daysonmarket"]

################################################################################
###################### DEAL WITH OTHER OBJECT VARIABLES ########################
################################################################################

################################ CATEGORICAL ###################################
# Convert categories from text to numbers
ordinal_encoder = OrdinalEncoder()
# Define which variables to change to categorical
char_variables = ["body_type", "city", "franchise_dealer", \
    "is_new", "listing_color", "make_name", "model_name", \
        "sp_name", "transmission", "dealer_zip"]
# Categorise variables
for variables in char_variables:
    cars[variables] = cars[variables].astype("category")

################################# NUMERICAL #####################################
# Encode character variables to numerical ones
# Create a label encoder object
encoder = LabelEncoder()
# Variables to convert to numerical ones
char_to_num_variables = ["exterior_color", "city", "interior_color", "make_name", "model_name", "sp_name"]
# Convert the character columns to numerical variables using transform function
for var in char_to_num_variables:
    encoder.fit(cars[var])
    cars[var+"_code"] = encoder.transform(cars[var].astype(str)) 

#################################### DATE ######################################
# Convert listed_date from object to Unix timestamp
cars["listed_date"] = pd.to_datetime(cars["listed_date"]).astype(int) // 10 ** 9


################################################################################
################################ CORRELATION ###################################
################################################################################

# Check which columns will be taken in for correlation function
print(cars.dtypes)
# Check for correlation in the data.
# Calculate the Pearson correlation coefficients for all covariates.
cor = cars.corr()
# Print the correlation coefficients
print(cor)

### custom_annotave - Custom annotating function.
#   INPUT:
#                     value - value of correlation to be highlighted;
#                     symbol - symbol to be used to highlight the values.
#   OUTPUT: 
#                     
def custom_annotate(value, symbol = "+"):
    if abs(value) >= 0.75:
        return symbol
    return ""

# Apply the custom annotating function to correlation data
annot = np.vectorize(custom_annotate)(cor)

# Plot the heatmap of correlations between variables
sns.heatmap(cor, annot = annot, fmt = "", cmap = "YlGnBu")


################################################################################
####################### SELECT VARIABLES FOR THE MODEL #########################
################################################################################

### important_variables - Function to select variables for the model.
#   INPUT:  
#                         corr_threshold - threshold of correlation with the price;
#                         data - data frame;
#   OUTPUT:
#                         cars_selected - updated data frame.
def important_variables(data = cars, corr_threshold = 0.5):
    # Get correlation matrix of all pairs of variables
    corr_matrix = data.corr()["price"]
    # Pre-set a variable to store selection
    selected_vars = []
    cars_selected = pd.DataFrame()
    # Select the variables with correlation greater than or equal to the threshold
    for col in corr_matrix.index:
        if abs(corr_matrix[col]) >= corr_threshold and col != "price":
            selected_vars.append(col)

    # Add the prediction variable "price" to the selected variables and create a new data frame
    selected_vars.append('price')
    cars_selected = data[selected_vars]
    
    return cars_selected

# Get the new data frame with selected variables
cars_selected = important_variables(data = cars, corr_threshold = 0.5)







