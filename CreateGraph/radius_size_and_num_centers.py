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
Y_NR = [13.1457,11.1517,10.1275,9.3324]
Y_NR_F= [13.1457,11.1517,10.1275,9.3324]
Y_NR_D = [14.6716,14.6716,14.6716,14.6716]

# Plotting both the curves simultaneously
plt.plot(X_NR, Y_NR, color='b', label='NR',marker='o')
plt.plot(X_NR_F, Y_NR_F, color='r',linestyle = 'dotted',linewidth = '5', label='Feasable NR',marker='o')
plt.plot(X_NR_D, Y_NR_D, color='y', label='Diverse NR',marker='o')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Number of centers")
plt.ylabel("Maximum Radius size")
plt.title("Airbnb: Radius size and number of centers")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
#>>>>Banks<<<<<
# Using Numpy to create an array X
X_NR = [162,314,480,632]
X_NR_F= [162,314,479,632]
X_NR_D= [2,2,2,4]

# Assign variables to the y axis part of the curve
Y_NR = [2156.22,2153.33,2148.77,2148.08]
Y_NR_F= [2156.22,2153.33,2148.77,2148.08]
Y_NR_D = [3589.25,3589.25,3589.25,3004.63]

# Plotting both the curves simultaneously
plt.plot(X_NR, Y_NR, color='b', label='NR',marker='o')
plt.plot(X_NR_F, Y_NR_F, color='r',linestyle = 'dotted',linewidth = '4', label='Feasable NR',marker='o')
plt.plot(X_NR_D, Y_NR_D, color='y', label='Diverse NR',marker='o')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Number of centers")
plt.ylabel("Maximum Radius size")
plt.title("Banks: Radius size and number of centers")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
#>>>>Point<<<<<
# Using Numpy to create an array X
X_NR = [165,320,479,648]
X_NR_F= [165,320,479,648]
X_NR_D= [133, 222,234,247]

# Assign variables to the y axis part of the curve
Y_NR = [225.69,216.91,181.48,155.11]
Y_NR_F= [225.69,216.91,181.48,155.11]
Y_NR_D = [225.69,216.91,215.01,181.48]

# Plotting both the curves simultaneously
plt.plot(X_NR, Y_NR, color='b', label='NR',marker='o')
plt.plot(X_NR_F, Y_NR_F, color='r',linestyle = 'dotted',linewidth = '5', label='Feasable NR',marker='o')
plt.plot(X_NR_D, Y_NR_D, color='y', label='Diverse NR',marker='o')


# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Number of centers")
plt.ylabel("Maximum Radius size")
plt.title("Point: Radius size and number of centers")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()