from ba import *
import matplotlib.pyplot as plt
import random

ba = BarabasiAlbert(5)
ba.drive(100000, 1, method="preferential")
result = random.choices(ba._attachment_nodes, k=100)
# bin the result
plt.hist(result)
# plot the analytical result
