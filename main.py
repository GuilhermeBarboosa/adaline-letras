#!/usr/bin/env python
# -*- coding: utf-8 -*-

# aplicativo para verificar se o ser vivo eh quadrupede ou bipede
# quadrupede = 1, bipede = -1
# cao = [-1,-1,1,1] | resposta = 1
# gato = [1,1,1,1] | resposta = 1
# cavalo = [1,1,-1,1] | resposta = 1
# homem = [-1,-1,-1,1] | resposta = -1

# pesos (sinapses)
w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# entradas
x = [
[-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1],
[-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1],
[-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1],
[1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
[1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
[1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
[-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
[-1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
[1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
[1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
[1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,1,-1],
[1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1],
[-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
[-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
[-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
[1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1],
[1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1],
[1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1],
]

# respostas esperadas
t = [1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1]
# bias (ajuste fino)
b = 0
# saida
y = 0
# numero maximo de interacoes
max_int = 1000
# taxa de aprendizado
taxa_aprendizado = 0.0025
# soma
soma = 0

# nome do animal
letra = ""
# resposta = acerto ou falha
resposta = ""

# dicionario de dados
d = {'-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1':'a',
'-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1':'b',
'-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1':'c',
'1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1':'d',
'1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1':'e',
'1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1':'j',
'-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1':'k',
'-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1':'a',
'-1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1':'b',
'1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1':'c',
'1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1':'d',
'1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,1,-1':'e',
'1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1':'j',
'1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1':'k',
'1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1':'a',
'-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1':'b',
'-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1':'c',
'-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1':'d',
'1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1':'e',
'1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1':'j',
'1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1':'k',}

print("Treinando")


# funcao para converter listas em strings
def listToString(list):
    s = str(list).strip('[]')
    s = s.replace(' ', '')
    return s


# inicio do algoritmo
for k in range(1, max_int):
    acertos = 0
    print("INTERACAO " + str(k) + "-------------------------")
    for i in range(0, len(x)):
        soma = 0

        # pega o nome do animal no dicionário
        if d.get(listToString(x[i])):
            letra = d[listToString(x[i])]
        else:
            letra = ""

        # para calcular a saida do adaline, cada entrada de x eh multiplicada
        # pelo seu peso w correspondente
        for j in range(0, len(x[i])):
            soma += x[i][j] * w[j]

        # a saida eh igual a soma anterior
        y_in = soma
        # print("y_in = ",str(y_in))

        # funcao de ativação
        if y_in >= 0:
            y = 1
        else:
            y = -1

        # atualiza os pesos
        for j in range(0, len(w)):
            # regra delta

            w[j] = w[j] + taxa_aprendizado * (t[i] - y_in) * x[i][j]

        if y == t[i]:
            acertos += 1
            resposta = "acerto"
        else:
            resposta = "erro"

        # imprime a resposta
        if y == 1:
            print("Resposta: " + resposta + "\nLetra correta: " + letra)
        elif y == 0:
            print("Padrao nao identificado")
        elif y == -1:
            print("Resposta: " + resposta + "\nLetra errada: " + letra)

    if acertos == len(x):
        print("Funcionalidade aprendida com " + str(k) + " interacoes")
        print("A LETRA ESCOLHIDA FOI" + letra)
        break
    print("")
print("Finished")