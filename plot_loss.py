import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('./checkpoints/20000.npy', 'rb') as f:
    loss_01= np.load(f)
with open('./checkpoints/20002.npy', 'rb') as f:
    loss_02= np.load(f)


fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(5,5))
line1, =ax.plot(np.linspace(0, len(loss_01)-1, len(loss_01)), loss_01, color='b')
line2, =ax.plot(np.linspace(0, len(loss_02)-1, len(loss_02)), loss_02, color='y')
# plt.plot(np.linspace(0, len(loss_01)-1, len(loss_01)), loss_01)
# plt.title('Training Loss Curve')
# plt.show()

ax.legend([line1, line2],['es+norm', 'vn'])
plt.show()