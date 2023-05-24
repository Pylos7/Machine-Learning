#1 - Import the data
#2 - Clean the data
#3 - Split data. Training Set/Test Set
#4 - Create a Model
#5 - Check the output
#6 - Improve

# The key distinction between Classification vs Regression algorithms is 
# Regression algorithms are used to determine continuous values such as price, income, age, etc.
# Classification algorithms are used to forecast or classify the distinct values such as Real or False, Male or Female, Spam or Not Spam, etc.

# Classification - New Instance = Mode
# Regression - New Instance = Mean


# Building a KNN Model
# Plot the training dataset
# Locate the new "test" instance
# Calculate distance from all train data points
# Sort the distance list in ascending order
# Choose first K distances from the sorted list Lets say K=5 in this instance

# Determine the value of K - Use elbow method
#   - Choose a range of values for K
#   - For each value, implement a KNN model
#   - Calculate error corresponding to each K value and plot it

# How to calculate the distance between points
#   - Manhattan Distance
#       Sum of Absolute differences between the two points, across all dimensions (not shortest distance between 2 points)
#   - Euclidean Distance
#       Shortest distance between two points
#   - Minkowiski Distance
#       Combination of Manhattan and Euclidean Distance Calculations
#   - Hamming Distance
#       Used to calculate distance when working with categorical variables
#       Total number of differences between two strings of indentical length

# Issues with distance based algorithms
#   - Takes the distance between points into account
#   - Fails when variables have different scales
#   - Relative Distance between points changes
#   - Leads to ambiguous interpretations
#
#       Solution:
#           Scale all features to the same scale