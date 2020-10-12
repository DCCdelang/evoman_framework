from statistics import mean
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import stats

fig, ax = plt.subplots()

with open('EASYgain.txt','r') as f:
    content1 = f.readlines()

content1 = [x.strip() for x in content1]
lists1 = [line.split(",") for line in content1]

with open('HARDgain.txt','r') as f:
    content2 = f.readlines()

content2 = [x.strip() for x in content2]
lists2 = [line.split(",") for line in content2]

lists1 = lists1[2:]
lists2 = lists2[2:]

gains1 = []
meangains1 = []
gains2 = []
meangains2 = []
count = 0
for i in range(len(lists1)):
    count += 1
    gains1.append(float(lists1[i][1]))
    gains2.append(float(lists2[i][1]))
    if count == 5:
        meangains1.append(mean(gains1))
        meangains2.append(mean(gains2))
        gains1 = []
        gains2 = []
        count = 0
# print(meangains)

# print(gaindata)
# Gives this array:
# [array([-10.71475335,  -6.96814981,  -6.09687667,  -2.26566063,
#        -12.32158321, -12.41114519,  -0.4223671 , -10.47659258,
#         -1.66361415, -10.4192869 ]), array([-15.32414623, -16.22658776, -20.196896  , -18.38548418,
#         -5.71713901, -22.04782413, -20.99404968, -23.94261043,
#        -23.16517974, -12.88841963])]
DM_easy = np.array([-16.779999999999934, -1.2099999999999256, -22.354999999999933, 11.610000000000054, -31.829999999999927, -23.67999999999996, 0.0800000000000594, -17.579999999999927, -12.194999999999927, -5.3899999999999375])

DM_hard = np.array([-9.249999999999934, -17.19499999999993, -17.104999999999933, -48.74499999999997, 5.315000000000108, -35.49999999999996, -27.279999999999962, -0.5999999999999261, -4.404999999999931, -7.739999999999934])
gaindata = [DM_easy,DM_hard,np.array(meangains1),np.array(meangains2)]

# tneat, pneat = stats.ttest_ind(meangains1, meangains2)
# print(tneat,pneat, "NEAT easy is significant greater than DM hard")
# tdm, pdm = stats.ttest_ind(DM_easy, DM_hard)
# print(tdm,pdm, "DM easy is significant greater than DM hard")
tdm, pdm = stats.ttest_ind(meangains1,DM_easy)
print(mean(meangains1), np.std(meangains1),mean(DM_easy), np.std(DM_easy))
print(tdm,pdm, "NEAT easy is significant greater than DM easy")
print( mean(meangains2), np.std(meangains2),mean(DM_hard), np.std(DM_hard))
tdm, pdm = stats.ttest_ind( meangains2,DM_hard)
print(tdm,pdm, "NEAT hard is significant smaller than DM hard")


ax.boxplot(gaindata,labels=["DM-e", "DM-h","NEAT-e", "NEAT-h"])
plt.xticks(fontsize=18)
plt.yticks(fontsize=14)
plt.ylabel("Gain",fontsize=18)
ax.set_title('Gains per EA per enemy subset', fontsize=18)
plt.savefig('gains_boxplot.png')
#
fit1 = []
fit2 = []
meanfit1 = []
meanfit2 = []
count = 0
for i in range(len(lists1)):
    count += 1
    fit1.append(float(lists1[i][4]))
    fit2.append(float(lists2[i][4]))
    if count == 5:
        meanfit1.append(mean(fit1))
        meanfit2.append(mean(fit2))
        fit1 = []
        fit2 = []
        count = 0

fitdata = [np.array(meanfit1),np.array(meanfit2)]

# For easy set
EasyMax = np.zeros((30,10))
EasyMean = np.zeros((30,10))

for sol in range(10):
    path = "C:/Users/djdcc_000/Documents/School/UNI/Computational Science/Evolutionary Computing/Assignment/evoman_framework_taskII/neat_sol/EASY/sol_"+str(sol)+"/stats.txt"
    with open(path,'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    lists = [line.split(" ") for line in content]
    lists = lists[9::3]
    for gen in range(len(lists)):
        EasyMax[gen][sol] = float(lists[gen][3])
        EasyMean[gen][sol] = float(lists[gen][4])

mu1 = EasyMax.mean(axis=1)
sigma1 = EasyMax.std(axis=1)
mu2 = EasyMean.mean(axis=1)
sigma2 = EasyMean.std(axis=1)

fig, ax = plt.subplots(1)
t = np.arange(1,31)
ax.tick_params(labelsize = 14)
ax.plot(t, mu1, lw=2, label='Mean Max', color='blue')
ax.plot(t, mu2, lw=2, label='Mean Mean', color='red')
ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)
ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='red', alpha=0.5)
ax.set_title('NEAT: easy enemy subset',fontsize=18)
ax.legend(loc='lower right', fontsize=16)
ax.set_xlabel('Generation', fontsize=16)
ax.set_ylabel('Fitness', fontsize=16)
plt.xlim(1,30)
plt.savefig('meanmaxEasy.png')


# For hard set
HardMax = np.zeros((30,10))
HardMean = np.zeros((30,10))

for sol in range(10):
    path = "C:/Users/djdcc_000/Documents/School/UNI/Computational Science/Evolutionary Computing/Assignment/evoman_framework_taskII/neat_sol/HARD/sol_"+str(sol)+"/stats.txt"
    with open(path,'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    lists = [line.split(" ") for line in content]
    lists = lists[9::3]
    for gen in range(len(lists)):
        HardMax[gen][sol] = float(lists[gen][3])
        HardMean[gen][sol] = float(lists[gen][4])

mu1 = HardMax.mean(axis=1)
sigma1 = HardMax.std(axis=1)
mu2 = HardMean.mean(axis=1)
sigma2 = HardMean.std(axis=1)

fig, ax = plt.subplots(1)
t = np.arange(1,31)
ax.tick_params(labelsize = 14)
ax.plot(t, mu1, lw=2, label='Mean Max', color='blue')
ax.plot(t, mu2, lw=2, label='Mean Mean', color='red')
ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)
ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='red', alpha=0.5)
ax.set_title('NEAT: hard enemy subset',fontsize=18)
ax.legend(loc='lower right', fontsize=16)
ax.set_xlabel('Generation', fontsize=16)
ax.set_ylabel('Fitness', fontsize=16)
plt.xlim(1,30)
plt.savefig('meanmaxHard.png')

# print(EasyMax.mean(axis=1))
