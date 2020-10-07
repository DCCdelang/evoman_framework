import re


# Dit bestand itereerd door besttest.txt bestand om alle bias en weights1_slice
# waardes te extraheren en weer in een soortgelijk bestand te zetten zoals bij
# Karinas implementatie.
text = open('besttest.txt','r')

with open('besttest.txt','r') as f:
    bias = []
    weights = []
    for line in f:
        if "bias=" in line:
            bias.append(line[line.index("bias=")+5:line.index(', response')])
        if "weight=" in line:
            weights.append(line[line.index("weight=")+7:line.index(', enabled')])

    bias1 = bias[5:]
    weights1 = weights[:200]
    bias2 = bias[:5]
    weights2 = weights[200:]
    print(len(bias1), len(weights1),len(bias2), len(weights2))
    testlist = bias1+weights1[::-1]+bias2+weights2[::-1]
    print(testlist)

    file_aux  = open('outputtest.txt','w')
    for value in testlist:
        file_aux.write(value+'\n')
    file_aux.close()
