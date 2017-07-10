
# coding: utf-8

# In[2]:


from random import shuffle
from random import randint
import seaborn as sns
import pandas as pd
import numpy as np
import sys

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

# In[3]:


# retorna todos os elementos de uma coluna de uma matriz
def getColumn(matrix, i):
    return [row[i] for row in matrix]


# In[4]:


# diminui o raio de vizinhanca de acordo com um fator, mas nao deixa chegar a 0
def decreaseNeighbourhoodRadius(factor, radius):
    radius = radius - factor
    if radius == 0 :
        radius = 1
    return radius


# In[4]:


# diminui o learning rate
def decreaseLearningRate(learning_rate, factor):
    return learning_rate - factor


# In[5]:


# inicializa os pesos com valores aleatorios
# numero de dimensoes, numero de neuronios, valores minimo e maximos
def initWeigths(dimensions, neuronsNumer, minValue, maxValue):
    w = [[0 for i in range(dimensions)] for j in range(neuronsNumer)]
    for i in range(0, len(w)):
        for j in range(0, len(w[i])):
            w[i][j] = np.random.uniform(minValue, maxValue)

    return w


# In[6]:


# distancia euclidiana entre o neuronio e o dado
def dist(weigth, dataPoint):
    soma = 0
    for i in range(0, len(weigth)):
        soma = soma + np.sum((weigth[i]-dataPoint[i])**2)
    return np.sqrt(soma)


# In[7]:


# atualiza o peso de um determinado neuronio
def updateWeight(w, i, datapoint, radius_step, learning_rate):
    if(w[i]):
        for j in range(len(w[i])):
            #print("\t" + str(learningRate * (1/radius) * (datapoint[j] - w[i][j])))
            w[i][j] = w[i][j] + learning_rate * (1/radius_step) * (datapoint[j] - w[i][j])


# In[23]:


# atualiza os pesos do neuronio e seus vizinhos
def updateWeigths(w, winner, datapoint, m, radius_step, learning_rate):
    # Ex topologia 2x4:
    # 1-2-3-4
    # | | | |
    # 5-6-7-8
    #

    #Se esta dentro do raio de atualizacao
    if radius_step >= 0:
        # atualiza o peso do no vencedor
        #print('Atualizando neuronio',winner)
        updateWeight(w, winner, datapoint, 1, learning_rate)

        # vizinho da direita. verifica se existe e tambem esta dentro da mesma linha da dimensao (ex. 4 nao eh vizinho do 5)
        if (winner + radius_step < len(w) and (winner + radius_step) % m[1] >= winner % m[1]):
            #updateWeight(winner + 1, datapoint, radius_step)
            updateWeigths(w, winner + radius_step, datapoint, m, radius_step - 1, learning_rate) #atualiza para os outros vizinhos do raio
        # vizinho da esquerda, mesmo raciocinio
        if (winner - radius_step >= 0 and (winner - radius_step) % m[1] <= winner % m[1]):
            #updateWeight(winner - 1, datapoint, radius_step)
            updateWeigths(w, winner - radius_step, datapoint, m, radius_step - 1, learning_rate) #atualiza para os outros vizinhos do raio
        # vizinho de baixo. verifica se existe
        if (winner + (radius_step*m[1]) < len(w)):
            #updateWeight(winner + m[1], datapoint, radius_step)
            updateWeigths(w, winner + (radius_step*m[1]), datapoint, m, radius_step - 1, learning_rate) #atualiza para os outros vizinhos do raio
        # vizinho de cima, mesmo raciocinio
        if (winner - (radius_step*m[1]) >= 0):
            #updateWeight(winner - m[1], datapoint, radius_step)
            updateWeigths(w, winner - (radius_step*m[1]), datapoint, m, radius_step - 1, learning_rate) #atualiza para os outros vizinhos do raio


# In[34]:


# verifica se um neuronio eh vizinho de outro
def isNeighboor(first, second, m):
    if (first == second) : return 0
    if(first > second):
        x = first
        first = second
        second = x

    if(abs(first - second) == 1):
        if(second % m[1] != 0):
            return 1
        else:
            return 0
    else:
        if(abs(first - second) == m[1]):
            return 1
        else:
            return 0


# In[9]:


# retorna o erro de quantizacao em um conjunto de dados
def quantizationError(dataSet, w):
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


# In[36]:


# retorna o erro topologico em um conjunto de dados
def topologicalError(dataSet, m, w):
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

            distSum += isNeighboor(winner, secondWinner, m)

    return distSum/len(dataSet)


# In[11]:


def readSpiral():
    dataSet = []
    maxWeight = 35
    fileObject = open("data/clustering/spiral.txt", "r")
    for line in fileObject:
        sline = line.split("\t")
        dataSet.append([float(sline[0]), float(sline[1])])
        #labels.append(int(sline[2]))
    return dataSet

def readT48():
    dataSet = []
    maxWeight = 650
    fileObject = open("data/clustering/t4.8k.txt", "r")
    for line in fileObject:
        sline = line.split(" ")
        dataSet.append([float(sline[0]), float(sline[1])])
    return dataSet



def getWinnersColors(w, dataSet):
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


# In[12]:

# metodo inicial
def som(path, id_img, dataSet, radius, m, minWeight, maxWeight, learning_rate, learning_rate_decrease_factor):
    #     dimensions, neuronsNumber, minValue, maxValue
    # Spiral => 1 ~ 35
    # T48 => 1 ~ 650
    w = initWeigths(2, m[0] * m[1], minWeight, maxWeight)

    #print("Initial Weights:")
    #print(w)

    #print(dataSet)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(getColumn(dataSet,0), getColumn(dataSet,1), "sb", getColumn(w,0), getColumn(w,1), "or")

    count = 0
    # enquanto a taxa de aprendizado eh maior que zero
    while(learning_rate > 0):
        count += 1
        #print('Epoca:',count)
        # para cada item dos dados, calcula o neuronio vencedor
        shuffle(dataSet)
        for i in range(0, len(dataSet)):
            #print('Dado:',i)
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
            #print('Neuronio vencedor:',winner)
            #atualiza os pesos do vencedor e seus vizinhos
            updateWeigths(w, winner, dataSet[i], m, radius, learning_rate)

        learning_rate = decreaseLearningRate(learning_rate, learning_rate_decrease_factor)
        radius = decreaseNeighbourhoodRadius(1, radius)


    #print("Final Weights:")
    #print(w)

    plt.subplot(212)
    plt.plot(getColumn(dataSet,0), getColumn(dataSet,1), "sb", getColumn(w,0), getColumn(w,1), "or")
    plt.savefig(path + "/neurons_position_" + id_img + ".png")
    plt.close()

    if len(dataSet[0]) == 2:
        df = pd.DataFrame(dataSet, columns=list('xy'))
    else:
        df = pd.DataFrame(dataSet, columns=list('xyz'))
    df['w'] = getWinnersColors(w, dataSet)
    sns_plot = sns.pairplot(x_vars=['x'], y_vars=['y'], data=df, hue="w", size=12)
    sns_plot.savefig(path + "/cluster_result_" + id_img + ".png")

    uMatrix(path, id_img, w, m)

    return [quantizationError(dataSet, w),topologicalError(dataSet, m, w)]

def mediaAoRedorUMatrix(u, i, j):
    soma = 0
    count = 0
    #print(str(i) + " - " + str(j))
    if i - 1 >= 0:
        soma += u[i-1][j]
        count += 1
    if i + 1 < len(u):
        soma += u[i+1][j]
        count += 1
    if j - 1 >= 0:
        soma += u[i][j-1]
        count += 1
    if j + 1 < len(u[i]):
        soma += u[i][j+1]
        count += 1
    #print(" -> " + str(soma) + " / " + str(count) + " " + str(soma/count))
    return soma/count

def uMatrix(path, id_img, w, m):
    u = [[0 for i in range((2 * m[1]) - 1)] for j in range((2 * m[0]) - 1)]

    k = 0
    l = 0
    for i in range(0, len(u)):
        for j in range(0, len(u[i])):
            #print(str(i) + "," + str(j))
            #linha par e coluna impar
            if i % 2 == 0 and j % 2 != 0:
            #    print(" -> " + str(k) + "," + str(k+1))
                u[i][j] = -dist(w[k], w[k+1])
                k = k + 1
            #linha impar e coluna par
            elif i % 2 != 0 and j % 2 == 0:
            #    print(" -> " + str(l) + "," + str(l + (m[1])))
                u[i][j] = -dist(w[l], w[l + (m[1])])
                l = l + 1

    for i in range(0, len(u)):
        for j in range(0, len(u[i])):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0) or (i != 0 and j != 0) or (i != 0 and j != len(u[0])-1) or (i != len(u)-1 and j != 0) or (i != len(u)-1 and j != len(u[0])-1):
                u[i][j] = mediaAoRedorUMatrix(u, i, j)

    u[0][0] = mediaAoRedorUMatrix(u, 0, 0)
    u[0][len(u[0])-1] = mediaAoRedorUMatrix(u, 0, len(u[0])-1)
    u[len(u)-1][0] = mediaAoRedorUMatrix(u, len(u)-1, 0)
    u[len(u)-1][len(u[0])-1] = mediaAoRedorUMatrix(u, len(u)-1, len(u[0])-1)

    plt.clf()
    plt.contourf(u, cmap = "rainbow")
    plt.colorbar()
    plt.savefig(path + "/u_matrix_" + id_img + ".png")

# In[1]:


def main():
    #values to be tested
    data_path = 'data/clustering/'
    extensao = '.txt'
    datasets_min_max = {'spiral': {'min': -10, 'max': 50},
                        't4.8k': {'min': 10, 'max': 600}}
    #topologies = [{'min': 1,'max': 1},{'min': 10,'max': 5}]
    topologiesX = [20]
    topologiesY = [20]
    radius_values = [2]
    learning_rates = [0.9]
    learning_rate_decrease_factors = [0.01]

    columns = ['dataset','radius','learning_rgmate','topology', 'learning_rate_decrease_factor', 'quantization_error', 'topological_error']
    tests = pd.DataFrame(columns=columns)
    path_results = "results_clustering_UMatrix"

    '''for filename in datasets_min_max:
        if filename == 'spiral':
            dataset = pd.read_csv(data_path+filename+extensao, sep='\t', header=None)
        else:
            dataset = pd.read_csv(data_path+filename+extensao, sep=' ', header=None)
        dataset.head(5)
        #for topology_y in range(topologies[0]['min'],topologies[1]['max']+1):
        #    for topology_x in range(topologies[1]['min'],topologies[0]['max']+1):
        for topology_y in topologiesY:
            for topology_x in topologiesX:
                m = [topology_x, topology_y]
                for radius in radius_values:
                    for learning_rate in learning_rates:
                        for learning_rate_decrease_factor in learning_rate_decrease_factors:
                            combination = [filename, "rd:" + str(radius), "lr:" + str(learning_rate), str(topology_x) + 'x' + str(topology_y), "lrdf:" + str(learning_rate_decrease_factor)]
                            img_str = '_'.join(combination)
                            print(img_str)
                            result = som(path_results,img_str, dataset.values.tolist(), radius, m, datasets_min_max[filename]['min'], datasets_min_max[filename]['max'], learning_rate, learning_rate_decrease_factor)
                            combination.append(result[0])
                            combination.append(result[1])
                            tests = tests.append(pd.DataFrame([combination], columns=columns), ignore_index=True)
                            # SOM calls
    print(tests)
    tests.to_csv(path_or_buf=path_results+"/test_results.txt", sep='\t', index=False, header=True)

    '''
    spiral = readSpiral()
    tk = readT48()

    plt.figure(1)
    plt.subplot(211)
    plt.plot(getColumn(spiral,0), getColumn(spiral,1), "sb")

    plt.subplot(212)
    plt.plot(getColumn(tk,0), getColumn(tk,1), "sb")
    plt.savefig("dataFigs.png")
    plt.close()

    # Topologia da rede: a quantidade de neurionios em cada dimensao para
    # calculo dos vizinhos. A multiplicacao dos numeros deve ser a qntd de neuronios
    #m = [2, 5]
    #dataSet=[[10, 10], [1, 1], [9,9], [2, 2], [10, 9], [2, 1], [9, 10], [1, 2]]
    #som("teste0", readSpiral(), 2, m, 1, 1, 0.9)'''

if __name__ == "__main__":
    main()

#Spiral:  txt
#H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203.
#t4.8k: G. Karypis, E.H. Han, V. Kumar, CHAMELEON: A hierarchical 765 clustering algorithm using dynamic modeling, IEEE Trans. on Computers, 32 (8), 68-75, 1999.
