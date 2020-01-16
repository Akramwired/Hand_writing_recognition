import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


digits = datasets.load_digits()
"""# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()"""
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

"""fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')
for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
  
plt.show()
"""

new_samples = np.array([
[0.00,0.00,0.15,2.59,1.83,0.00,0.00,0.00,0.00,0.00,6.03,5.80,7.25,1.38,0.00,0.00,0.00,0.00,1.53,0.00,5.11,3.05,0.00,0.00,0.00,0.00,0.00,2.67,7.55,5.57,1.22,0.00,0.00,0.00,0.00,1.37,3.05,4.20,5.87,0.00,0.00,0.00,0.00,0.00,0.00,0.92,6.86,0.00,0.00,0.00,0.15,3.59,5.04,7.02,4.58,0.00,0.00,0.00,0.15,3.59,2.98,1.22,0.00,0.00],
[0.00,0.00,0.00,1.45,4.81,0.31,0.00,0.00,0.00,0.00,0.99,6.94,7.62,0.76,0.00,0.00,0.00,0.61,6.64,4.43,7.02,0.00,0.00,0.00,0.00,4.66,7.47,7.40,7.62,6.33,1.15,0.00,0.00,1.15,1.83,1.30,7.02,1.91,0.23,0.00,0.00,0.00,0.00,0.76,6.94,0.00,0.00,0.00,0.00,0.00,0.00,0.23,5.95,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.54,0.76,2.29,7.10,2.98,0.00,0.00,0.00,7.40,6.86,6.18,1.99,0.00,0.00,0.00,0.00,7.32,1.98,2.06,0.00,0.00,0.00,0.00,0.00,6.86,7.02,7.17,2.21,0.00,0.00,0.00,0.00,3.81,0.92,2.29,6.64,0.15,0.00,0.00,0.00,0.00,0.00,0.69,7.32,1.30,0.00,0.00,0.00,0.00,5.64,7.32,3.81,0.00,0.00,0.00,0.00,0.00,1.15,0.54,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.30,4.19,4.12,1.83,0.00,0.00,0.00,0.00,4.43,5.64,5.27,7.47,3.59,0.00,0.00,0.00,3.05,7.63,6.48,5.72,4.88,0.00,0.00,0.00,0.00,0.00,0.00,0.61,7.32,0.46,0.00,0.00,0.00,0.00,0.15,3.13,7.25,0.46,0.00,0.00,0.00,0.31,6.86,6.18,1.60,0.00,0.00,0.00,0.00,0.00,0.61,0.07,0.00,0.00,0.00,0.00]
])


new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')