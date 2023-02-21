import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Guilherme\Documents\Projetos\projeto-adaline')
x = np.loadtxt('x.txt')
(amostras, entradas) = np.shape(x)

t = np.genfromtxt('target.txt')
(numclasses, targets) = np.shape(t)

limiar = 0.0
alfa = 0.01
errotolerado = 0.001

v = np.empty((entradas, numclasses))

v0 = np.empty(numclasses)

vetorCiclos = []
vetorErros = []

for i in range(entradas):
    for j in range(numclasses):
        v[i][j] = rd.uniform(-0.1, 0.1)

for j in range(numclasses):
    v0[j] = rd.uniform(-0.1,0.1)

erro = 10
ciclo = 0

yin = np.zeros((numclasses,1))
y = np.zeros((numclasses,1))


# opc = 0

# print('DESEJA EXECUTAR O TREINAMENTO POR ERRO OU POR CICLOS:')
# print('1 - ERROS')
# print('2 - CICLOS')
# opc = input('INSIRA O QUE DESEJA: ')
# if opc==1:
#     erro = (input('QUAL A QUANTIDADE DE ERRO: '))
# elif opc == 2:
#     ciclo = (input('QUAL A QUANTIDADE DE CICLO: '))
#
# print('ESCOLHA A TAXA DE APRENDIZAGEM: ')
# alfa = input('INSIRA O QUE DESEJA: ')

while erro > errotolerado:
    erro = 0
    ciclo = ciclo + 1

    for i in range(amostras):
        xaux = x[i,:]
        for m in range(numclasses):
            soma = 0
            for n in range(entradas):
                soma = soma + xaux[n] * v[n][m]
            yin[m] = soma + v0[m]

        for j in range(numclasses):
            if yin[j] >= limiar:
                y[j] = 1.0
            else:
                y[j] = -1.0

        for j in range(numclasses):
            erro = erro + 0.5 * ((t[j][i] - y[j])**2)
        vanterior = v

        for m in range(entradas):
            for n in range(numclasses):
                v[m][n] = vanterior[m][n] + alfa*(t[n][i] - y[n]) * xaux[m]
        v0anterior = v0


        for j in range(numclasses):
            v0[j] = v0anterior[j] + alfa * (t[j][i] - y[j])

    vetorCiclos.append(ciclo)
    vetorErros.append(erro)

    plt.scatter(vetorCiclos, vetorErros, marker='x', color='blue')
    plt.xlabel('Ciclo')
    plt.ylabel('Erro')
    plt.show()

# print(vetorCiclos)
# print(erro)

#opc = input('DESEJA TESTAR COM QUAL LETRA?')
# while opc < 1 or opc > 21:
#     opc = input('OPÇÃO INVALIDA\n\n')
#     opc = input('DESEJA TESTAR COM QUAL LETRA?')
#
# xteste = x[opc,:]

xteste = x[7,:]
for m2 in range(numclasses):
    soma = 0
    for n2 in range(entradas):
        soma = soma + xteste[n2] * v[n2][m2]
        yin[m2] = soma + v0[m2]

print(yin)
for j in range(numclasses):
    if yin[j] >= limiar:
        y[j] = 1.0
    else:
        y[j] = -1.0
print(y)