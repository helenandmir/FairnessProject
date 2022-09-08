# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math
#>>>>Airbnb<<<<<
# Using Numpy to create an array X
X_NR = [156,301,460,610]
X_NR_F= [156,301,460,610]
X_NR_D= [115,137,144,149]

# Assign variables to the y axis part of the curve
Y_NR = [1.8340,1.8136,1.8261,1.8149]
Y_NR_F= [2.0081,2.0009,1.8311,1.9912]
Y_NR_D = [1.8928,1.9001,1.8975,2.2276]

# Plotting both the curves simultaneously
plt.plot(X_NR, Y_NR, color='b', label='NR',marker='o')
plt.plot(X_NR_F, Y_NR_F, color='r',linestyle = 'dotted',linewidth = '4', label='Feasable NR',marker='o')
plt.plot(X_NR_D, Y_NR_D, color='y', label='Diverse NR',marker='o')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Number of centers")
plt.ylabel("Maximum alpha")
plt.title("Airbnb:Alpha and number of centers")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
#>>>>Banks<<<<<
# Using Numpy to create an array X
X_NR = [162,314,480,632]
X_NR_F= [162,314,479,632]
X_NR_D= [4,2,2,2]

# Assign variables to the y axis part of the curve
Y_NR = [1.6169,1.7790,1.9023,1.8399]
Y_NR_F= [1.6818,1.7840,1.9023,2.0969]
Y_NR_D = [1.3547,1.8012,1.6653,1.0931]

# Plotting both the curves simultaneously
plt.plot(X_NR, Y_NR, color='b', label='NR',marker='o')
plt.plot(X_NR_F, Y_NR_F, color='r',linestyle = 'dotted',linewidth = '5', label='Feasable NR',marker='o')
plt.plot(X_NR_D, Y_NR_D, color='y', label='Diverse NR',marker='o')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Number of centers")
plt.ylabel("Maximum alpha")
plt.title("Banks: Alpha and number of centers")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
#>>>>Point<<<<<
# Using Numpy to create an array X
X_NR = [165,320,479,648]
X_NR_F= [165,320,479,648]
X_NR_D= [133, 222,234, 247]

# Assign variables to the y axis part of the curve
Y_NR = [1.8621,1.9094,1.8720,1.8296]
Y_NR_F= [1.8621,2.0367,2.0904,2.0357]
Y_NR_D = [1.7439,1.7847,2.1437,2.1469]

# Plotting both the curves simultaneously
plt.plot(X_NR, Y_NR, color='b', label='NR',marker='o')
plt.plot(X_NR_F, Y_NR_F, color='r',linestyle = 'dotted',linewidth = '5', label='Feasable NR',marker='o')
plt.plot(X_NR_D, Y_NR_D, color='y', label='Diverse NR',marker='o')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Number of centers")
plt.ylabel("Maximum alpha")
plt.title("Point: Alpha and number of centers")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()