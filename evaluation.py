import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

accuracies3 = np.load('./metrics/accuracies3_10000.npy')
accuracies24 = np.load('./metrics/accuracies24_10000.npy')
accuracies42 = np.load('./metrics/accuracies42_10000.npy')
accuraciesx = np.load('./metrics/accuracies.npy')

print(accuraciesx)

avg_accuracies = np.empty_like(accuracies3)

for i in range(accuracies3.shape[0]):
    for j in range(accuracies3.shape[1]):
        avg_accuracies[i,j] = np.average(np.array([accuracies3[i,j],accuracies24[i,j],accuracies42[i,j]]))

length = 7500

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(np.arange(100,length+100,100), avg_accuracies[0,:int(length/100)]*100, label='Zeros: '+str(np.round(avg_accuracies[0,int(length/100-1)]*100,1))+'%')
ax.plot(np.arange(100,length+100,100), avg_accuracies[1,:int(length/100)]*100, label='Random: '+str(np.round(avg_accuracies[1,int(length/100-1)]*100,1))+'%')
ax.plot(np.arange(100,length+100,100), avg_accuracies[2,:int(length/100)]*100, label='Sine & cosine: '+str(np.round(avg_accuracies[2,int(length/100-1)]*100,1))+'%')
ax.plot(np.arange(100,length+100,100), avg_accuracies[3,:int(length/100)]*100, label='Arctan: '+str(np.round(avg_accuracies[3,int(length/100-1)]*100,1))+'%')
ax.plot(np.arange(100,length+100,100), avg_accuracies[4,:int(length/100)]*100, label='RPE sin: '+str(np.round(avg_accuracies[4,int(length/100-1)]*100,1))+'%')
ax.plot(np.arange(100,length+100,100), avg_accuracies[5,:int(length/100)]*100, label='Linear: '+str(np.round(avg_accuracies[5,int(length/100-1)]*100,1))+'%')
ax.set_xlabel("Training steps")
ax.set_ylabel("Accuracy")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()

plt.show()