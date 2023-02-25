################################################################################
######################### FEATURE SCALING. METHOD ##############################
################################################################################

# Check if the data has a lot of outliers to then decide if MINMAX or STANDARDISATION
# Create a box plot for each numeric column
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
# Plots
sns.boxplot(x = "engine_displacement", data = strat_train_set, ax = axes[0, 0])
sns.boxplot(x = "fuel_tank_volume", data = strat_train_set, ax = axes[0, 1])
sns.boxplot(x = "height", data = strat_train_set, ax = axes[0, 2])
sns.boxplot(x = "horsepower", data = strat_train_set, ax = axes[1, 0])
sns.boxplot(x = "latitude", data = strat_train_set, ax = axes[1, 1])
sns.boxplot(x = "length", data = strat_train_set, ax = axes[1, 2])
sns.boxplot(x = "longitude", data = strat_train_set, ax = axes[2, 0])
sns.boxplot(x = "mileage", data = strat_train_set, ax = axes[2, 1])
sns.boxplot(x = "owner_count", data = strat_train_set, ax = axes[2, 2])
sns.boxplot(x = "wheelbase", data = strat_train_set, ax = axes[3, 0])
sns.boxplot(x = "width", data = strat_train_set, ax = axes[3, 1])
sns.boxplot(x = "price", data = strat_train_set, ax = axes[3, 2])
# Layout type
plt.tight_layout()

# There are some outliers, STANDARDISATION method will be used


################################################################################
####################### EXPLORE THE TRAINING DATA SET ##########################
################################################################################

# Explore the training data set visually.
# Check for correlation in the data.
# Use a scatter plot for the attributes, exclude column "index"
cols_to_plot = [col for col in strat_train_set.columns if col != "index"]
data_to_plot = strat_train_set[cols_to_plot]

# Plot the scatter matrix, excluding the "index" column
fig, axes = plt.subplots(nrows = len(data_to_plot.columns), ncols = len(data_to_plot.columns), figsize = (16, 12))
# Subtitle
plt.suptitle("Scatter Plot Matrix of Cars Data", fontsize=14)
pd.plotting.scatter_matrix(data_to_plot, alpha = 0.2, ax = axes)
# Rotate the x-axis labels vertically and y-axis labels horizontally
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("right")
# Set layout
plt.tight_layout()
plt.gcf().subplots_adjust(wspace = 0, hspace = 0)
