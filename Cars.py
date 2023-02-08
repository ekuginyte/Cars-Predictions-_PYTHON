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
# This line runs on Jupyter Notebook only
#get_ipython().run_line_magic('matplotlib', 'inline')

# Read the dataset from a file
# Specify type 'string' for the 'bed' and 'dealer_zip' attributes to avoid errors interpreting them as numbers
cars = pd.read_csv("/Users/ernakuginyte/Documents/ID5059 Knowledge Discovery and Data Mining/Assignments/Project 1/data/small/used_cars_data_small_0.csv", \
    dtype = {"bed": "string", "dealer_zip": "string"})

################################################################################
############################# EXPLORE THE DATA #################################
################################################################################

# Clear the maximum number of columns to be displayed, so that all will be visible
pd.set_option('display.max_columns', None)
# Check the basic statistics of the data
cars.describe()
# Display the data frame
print(cars)

# Explore numeric columns visually
cars.hist(bins = 20, figsize = (20,15)) 
plt.show()

################################################################################
############################# DATA WRANGLING ###################################
################################################################################
# Firstly, deal with the missing values.
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
###            else, add median numeric values to the numeric data, string or object data won't be modified.  
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
        if df_cleaned[col].dtype in ['float64', 'int64']:
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

################################################################################
############### DECIDE WHAT TO DO WITH MISSING STRING VALUES ###################
################################################################################


################################################################################
################################ CORRELATION ###################################
################################################################################

# Check which columns will be taken in for correlation function
print(cars.dtypes)
# Check for correlation in the data.
# Calculate the Pearson correlation coefficients for all covariates.
corr = cars.corr()
# Print the correlation coefficients
print(corr)

# Plot the correlation coefficients in a heatmap
sns.heatmap(corr, annot = False)
plt.show()

################################################################################
######################### TRAINING AND TEST SETS ###############################
################################################################################

# Check quantiles to then categorise the data in to 5 sections
cat = cars["price"].describe()

# Create price_categ column to split the data into evenly distributed
#   training and testing sets
cars["price_categ"] = pd.qcut(cars["price"], q = 5, labels = [1, 2, 3, 4, 5])

cars["price_categ"] = pd.cut(cars["price"],
                        bins = [0, 1.5, 3.0, 4.5, 6., np.inf],
                        labels = [1, 2, 3, 4, 5])

# Plot the category values in a histogram to check if it makes sense
cars["price_categ"].hist()
plt.show()

# Split the training and the testing sets
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 7) # 7 being the lucky number
for train_index, test_index in split.split(cars, cars["price_categ"]):
        strat_train_set = cars.loc[train_index]
        strat_test_set = cars.loc[test_index]

# Check if the categorical split makes sense
strat_test_set["price"].value_counts() / len(strat_test_set)

# Drop the category column as it should not predict anything in the model
for set_ in (strat_train_set, strat_test_set): 
    set_.drop("price_categ", axis = 1, inplace = True)

################################################################################
######################## EXPLORE TRAINING DATA SET #############################
################################################################################

# Add an extra variable
cars["mileage_per_owner"] = cars["mileage"] / cars["owner_count"]

# Explore the training data set visually
# First copy the data set.
cars_t = strat_train_set.copy()

# Plot geographical data
cars_t.plot(kind = "scatter", x = "longitude", y = "latitude", 
            c = cars_t["price"], cmap = plt.get_cmap("jet"), colorbar = True)
plt.legend()
plt.show()


# Correlation
corr_matrix = cars_t.corr()
attributes = corr_matrix["price"].sort_values(ascending = False)

# Plot the correlation scatter matrix
col_names = list(cars_t.columns)
attributes = ["year", "horsepower", "latitude", "mileage"]
scatter_matrix(cars[attributes], figsize = (12, 8))
plt.show()

################################################################################
############################# TRAINING MODELS ##################################
################################################################################


################################################################################
########################### EVALUATE THE MODELS ################################
################################################################################








