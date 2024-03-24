import numpy as np
from sklearn.metrics import confusion_matrix
labels = [True, True]
outputs = np.array([1, 1])
res = confusion_matrix(labels, outputs > 0.5, labels=[0, 1]).ravel()
print(res)