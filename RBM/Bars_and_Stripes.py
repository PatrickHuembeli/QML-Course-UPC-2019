import numpy as np
import matplotlib.pyplot as plt

dataset = []
set_size = 10000

for k in range(set_size):
    init = np.zeros((4,4))
    nr_of_stripes = np.random.randint(low = 1, high= 4, size = 1)
    columns = np.random.randint(4, size = nr_of_stripes)

    for i in columns:
        init[:, i] = 1.0

    dataset.append(init)
    dataset.append(init.T)   
    
np.save('bars_and_stripes', dataset)   

# plt.imshow(dataset[0])
# plt.show()  