import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the wine data from a CSV file
wine = pd.read_csv("winequality-white.csv", delimiter=';')

# Define a quality threshold
quality_threshold = 6  # Adjust this threshold as needed

# Filter the data for quality above the threshold
filtered_wine = wine[wine["quality"] > quality_threshold]

# Extract the columns for the 3D plot
# xname = "volatile acidity"
# yname = "citric acid"
# zname = "free sulfur dioxide"

xname = "alcohol"
yname = "sulphates"
zname = "total sulfur dioxide"


# Set the plot style
plt.style.use('ggplot')

# Create a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel(xname)
ax.set_ylabel(yname)
ax.set_zlabel(zname)

# Scatter plot the filtered data
ax.scatter(
    filtered_wine[xname],
    filtered_wine[yname],
    filtered_wine[zname],
    c=filtered_wine["quality"],
    s=filtered_wine["quality"]**2,
    cmap="cool"
)

plt.show()
