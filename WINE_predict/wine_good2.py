import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the wine data from a CSV file
wine = pd.read_csv("WINE_predict/winequality-white.csv", delimiter=';')

# Extract the columns for the 3D plot
xname = "volatile acidity"
yname = "citric acid"
zname = "free sulfur dioxide"

# Set the plot style
plt.style.use('ggplot')

# Create a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel(xname)
ax.set_ylabel(yname)
ax.set_zlabel(zname)

# Scatter plot the data
ax.scatter(
    wine[xname],
    wine[yname],
    wine[zname],
    c=wine["quality"],
    s=wine["quality"]**2,
    cmap="cool"
)

plt.show()
