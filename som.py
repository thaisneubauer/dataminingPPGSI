from random import shuffle
from random import randint
import seaborn as sns
import pandas as pd
import numpy as np
import sys

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

radius = 2
learningRate = 0.9
w = []
dataSet = []
labels = []
minWeight = 1
maxWeight = 10

# retorna todos os elementos de uma coluna de uma matriz
def getColumn(matrix, i):
    return [row[i] for row in matrix]

# diminui o raio de vizinhanca de acordo com um fator, mas nao deixa chegar a 0
def decreaseNeighbourhoodRadius(factor):
    global radius
    radius = radius - factor
    if radius == 0 :
        radius = 1

# diminui o learning rate
def decreaseLearningRate():
    global learningRate
    learningRate -= 0.1

# inicializa os pesos com valores aleatorios
# numero de dimensoes, numero de neuronios, valores minimo e maximos
def initWeigths(dimensions, neuronsNumer, minValue, maxValue):
    w = [[0 for i in range(dimensions)] for j in range(neuronsNumer)]
    for i in range(0, len(w)):
        for j in range(0, len(w[i])):
            w[i][j] = np.random.uniform(minValue, maxValue)

    return w

# distancia euclidiana entre o neuronio e o dado
def dist(weigth, dataPoint):
    soma = 0
    for i in range(0, len(weigth)):
        soma = soma + np.sum((weigth[i]-dataPoint[i])**2)
    return np.sqrt(soma)

# atualiza o peso de um determinado neuronio
def updateWeight(i, datapoint, radius_step):
    if(w[i]):
        for j in range(len(w[i])):
            #print("\t" + str(learningRate * (1/radius) * (datapoint[j] - w[i][j])))
            w[i][j] = w[i][j] + learningRate * (1/radius_step) * (datapoint[j] - w[i][j])

# atualiza os pesos do neuronio e seus vizinhos
def updateWeigths(winner, datapoint, m, radius_step):
    # Ex topologia 2x4:
    # 1-2-3-4
    # | | | |
    # 5-6-7-8
    #

    #Se esta dentro do raio de atualizacao
    if radius_step > 0:
        # atualiza o peso do no vencedor
        updateWeight(winner, datapoint, 1)

        # vizinho da direita. verifica se existe e tambem esta dentro da mesma linha da dimensao (ex. 4 nao eh vizinho do 5)
        if (winner + 1 < len(w) and winner + 1 % m[1] > winner % m[1]):
            #updateWeight(winner + 1, datapoint, radius_step)
            updateWeigths(winner + 1, datapoint, m, radius_step - 1) #atualiza para os outros vizinhos do raio
        # vizinho da esquerda, mesmo raciocinio
        if (winner - 1 > 0 and winner - 1 % m[1] < winner % m[1]):
            #updateWeight(winner - 1, datapoint, radius_step)
            updateWeigths(winner - 1, datapoint, m, radius_step - 1) #atualiza para os outros vizinhos do raio
        # vizinho de baixo. verifica se existe
        if (winner + m[1] < len(w)):
            #updateWeight(winner + m[1], datapoint, radius_step)
            updateWeigths(winner + m[1], datapoint, m, radius_step - 1) #atualiza para os outros vizinhos do raio
        # vizinho de cima, mesmo raciocinio
        if (winner - m[1] > 0):
            #updateWeight(winner - m[1], datapoint, radius_step)
            updateWeigths(winner - m[1], datapoint, m, radius_step - 1) #atualiza para os outros vizinhos do raio

# verifica se um neuronio eh vizinho de outro
def isNeighboor(first, second, m):
    if(abs(first - second) == 1):
        if(first % m[1] < second % m[1]):
            return 1
        else:
            return 0
    else:
        if(abs(first - second) == m[1]):
            return 1
        else:
            return 0

# retorna o erro de quantizacao em um conjunto de dados
def quantizationError(dataSet):
    distSum = 0
    for i in range(0, len(dataSet)):
            #distancia do neuronio vencedor
            winnerDist = sys.float_info.max

            #indice do neuronio vencedor
            winner = 0

            #calcula as distancias entre todos os neuronios e salva a menor
            for neuronIndex in range(0, len(w)):
                d = dist(w[neuronIndex], dataSet[i])
                if(d < winnerDist):
                    winner = neuronIndex
                    winnerDist = d

            distSum += winnerDist
    return distSum/len(dataSet)

# retorna o erro topologico em um conjunto de dados
def topologicalError(dataSet, m):
    distSum = 0
    for i in range(0, len(dataSet)):
            #distancia do neuronio vencedor
            winnerDist = sys.float_info.max
            secondWinnerDist = sys.float_info.max

            #indice do neuronio vencedor
            winner = 0
            secondWinner = 0

            #calcula as distancias entre todos os neuronios e salva a menor
            for neuronIndex in range(0, len(w)):
                d = dist(w[neuronIndex], dataSet[i])
                if(d < winnerDist):
                    winner = neuronIndex
                    winnerDist = d
                else:
                    if(d < secondWinnerDist):
                        secondWinner = neuronIndex
                        secondWinnerDist = d

            distSum += int(isNeighboor(winner, secondWinner, m))

    return distSum/len(dataSet)

def readSpiral():
    global dataSet, labels, maxWeight
    maxWeight = 35
    fileObject = open("data/spiral.txt", "r")
    for line in fileObject:
        sline = line.split("\t")
        dataSet.append([float(sline[0]), float(sline[1])])
        labels.append(int(sline[2]))

def readT48():
    global dataSet, maxWeight
    maxWeight = 650
    fileObject = open("data/t4.8k.txt", "r")
    for line in fileObject:
        sline = line.split(" ")
        dataSet.append([float(sline[0]), float(sline[1])])



def getWinnersColors():
    global dataSet
    winners = []
    #colors = []
    #print(matplotlib.colors.cnames.items()['name'])

    #colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(dataSet)))
    for i in range(0, len(dataSet)):
            #distancia do neuronio vencedor
            winnerDist = sys.float_info.max

            #indice do neuronio vencedor
            winner = 0

            #calcula as distancias entre todos os neuronios e salva a menor
            for neuronIndex in range(0, len(w)):
                d = dist(w[neuronIndex], dataSet[i])
                if(d < winnerDist):
                    winner = neuronIndex
                    winnerDist = d
            winners.append(winner)

    return winners

# metodo inicial
def main():

    global w, dataSet, minWeight, maxWeight
    #readSpiral()
    #readT48()

    dataSet=[[10, 10], [1, 1], [9,9], [2, 2], [10, 9], [2, 1], [9, 10], [1, 2]]
    #dataSet = [[5,5], [5,6

    # Topologia da rede: a quantidade de neurionios em cada dimensao para
    # calculo dos vizinhos. A multiplicacao dos numeros deve ser a qntd de neuronios
    m = [1,2]

    # define os valores dos parametros a serem testados
    # chama SOM com os valores a serem testados

    #     dimensions, neuronsNumber, minValue, maxValue
    # Spiral => 1 ~ 35
    # T48 => 1 ~ 650
    w = initWeigths(2, m[0] * m[1], minWeight, maxWeight)

    print("Initial Weights:")
    print(w)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(getColumn(dataSet,0), getColumn(dataSet,1), "sb", getColumn(w,0), getColumn(w,1), "or")
    #plt.savefig("plot_init.png")
    #plt.clf()

    count = 0
    # enquanto a taxa de aprendizado eh maior que zero
    while(learningRate > 0):
        # para cada item dos dados, calcula o neuronio vencedor
        shuffle(dataSet) # >>>>>>>> NAO PASSAR PELOS DADOS SEMPRE NA MESMA ORDEM <<<<<<< justo, boa
        for i in range(0, len(dataSet)):
            #distancia do neuronio vencedor
            winnerDist = sys.float_info.max

            #indice do neuronio vencedor
            winner = 0

            #calcula as distancias entre todos os neuronios e salva a menor
            for neuronIndex in range(0, len(w)):
                d = dist(w[neuronIndex], dataSet[i])
                if(d < winnerDist):
                    winner = neuronIndex
                    winnerDist = d

            #print(str(w[winner]) + " -> " + str(dataSet[i]))
            #print(str(i) + " " + str(winner) + " " + str(winnerDist))

            #plt.plot(getColumn(dataSet,0), getColumn(dataSet,1), "sb", getColumn(w,0), getColumn(w,1), "or")
            #plt.savefig("plot_"+str(count)+".png")
            #plt.clf()
            #count = count + 1

            #atualiza os pesos do vencedor e seus vizinhos
            #print(str(winner) + "\t ANTES: " + str(w[winner]) )
            updateWeigths(winner, dataSet[i], m, radius)
            #print(str(winner) + "\t DEPOIS: " + str(w[winner]))

        decreaseLearningRate()
        decreaseNeighbourhoodRadius(1)

    print("Final Weights:")
    print(w)

    print("Quantization Error: " + str(quantizationError(dataSet)))
    print("Topological Error: " + str(topologicalError(dataSet, m)))

    plt.subplot(212)

    df = pd.DataFrame(dataSet, columns=list('xy'))
    df['w'] = getWinnersColors()
    sns_plot = sns.pairplot(x_vars=['x'], y_vars=['y'], data=df, hue="w", size=10)
    sns_plot.savefig("output.png")
    #plt.plot(getColumn(dataSet,0), getColumn(dataSet,1), "sb", getColumn(w,0), getColumn(w,1), "or")
    #plt.plot(getColumn(dataSet,0), getColumn(dataSet,1), marker='o', linestyle='', label=getWinnersColors())
    #print(getWinnersColors())
    #plt.scatter(getColumn(dataSet,0), getColumn(dataSet,1), [], linspace(1,10,length(dataSet)))
    #plt.plot(getColumn(dataSet,0), getColumn(dataSet,1), "rgbw")
    #plt.savefig("plot_end.png")

if __name__ == "__main__":
    main()

#Spiral:  txt
#H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203.
#t4.8k: G. Karypis, E.H. Han, V. Kumar, CHAMELEON: A hierarchical 765 clustering algorithm using dynamic modeling, IEEE Trans. on Computers, 32 (8), 68-75, 1999. 
