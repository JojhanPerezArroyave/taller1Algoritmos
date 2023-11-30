import tabulate as tb
import numpy as np
import string

canales = {}
with open('canales.txt', 'r') as archivo_txt:
    for linea in archivo_txt:
        partes = linea.strip().split()
        if partes:
            canal, valores = partes[0], partes[1:]
            canales[canal] = valores

"""Devuelve una lista de los estados existentes en los canales de comunicación. Los estados se definen como los valores que toman los canales de comunicación en un instante de tiempo determinado.
Los tiempos están definidos por cada posición de cada canal de comunicación. Por ejemplo, el estado 1 es 000 y el estado 2 es 101."""
def estadosExistentes(canales: dict) -> list:
    estados = set()
    num_tiempos = len(list(canales.values())[0]) 
    orden_definido = ['000', '100', '010', '110', '001', '101', '011', '111']  # Define el orden según el comentario

    for i in range(num_tiempos):
        estado = "".join(canales[canal][i] for canal in canales)
        estados.add(estado)

    estados_ordenados = [estado for estado in orden_definido if estado in estados]

    return estados_ordenados



"""Devuelve un diccionario el cual por cada estado existente en la lista de estados, devuelve cuántas veces se encuentra este en los canales de comunicación."""
def frecuenciaEstados(canales: dict, estados: list) -> dict:
    frecuencias = dict()
    num_tiempos = len(list(canales.values())[0]) 
    
    for estado in estados:
        frecuencias[estado] = 0
        for i in range(num_tiempos):
            if estado == "".join(canales[canal][i] for canal in canales):
                frecuencias[estado] += 1
    
    return frecuencias



"""Método que devuelve un diccionario en donde la clave es cada estado de la lista de estados y su valor es un arreglo que tiene tantas posiciones como canales, el valor de la posicion 0 del arreeglo será las veces que despues de ese estado, hay un 1 en el canal A"""
def frecuenciaEstadosSiguientes(canales: dict, estados: list) -> dict:
    num_canales = len(canales)
    frecuencias = {estado: [0] * num_canales for estado in estados}
    num_tiempos = len(list(canales.values())[0])

    for estado in estados:
        for i in range(num_tiempos - 1):
            if estado == "".join(canales[canal][i] for canal in canales):
                for j, canal in enumerate(canales):
                    if canales[canal][i + 1] == "1":
                        frecuencias[estado][j] += 1

    return frecuencias

def probabilidades(frecuenciaEstados: dict, frecuenciaEstadosSiguientes: dict) -> dict:
    probabilidades = dict()
    for estado in frecuenciaEstados:
        probabilidades[estado] = [0] * len(frecuenciaEstadosSiguientes[estado])
        for i, frecuencia in enumerate(frecuenciaEstadosSiguientes[estado]):
            probabilidades[estado][i] = frecuencia / frecuenciaEstados[estado]
    return probabilidades


""" Método que devuelve un diccionario en donde la clave es cada estado de la lista de estados y su valor es un arreglo que tiene tantas posiciones como estados existentes, el valor de la posicion 0 del arreglo será las veces que despues del primer estado esta el primer estado, la posición 1 es las veces que despues del primer estado esta el segundo estado  """
def probabilidadesEstadosSiguientes(canales: dict, estados: list) -> dict:
    probabilidades = {estado: [0] * len(estados) for estado in estados}
    num_tiempos = len(list(canales.values())[0])

    for estado in estados:
        for i in range(num_tiempos - 1):
            if estado == "".join(canales[canal][i] for canal in canales):
                for j, estadoSiguiente in enumerate(estados):
                    if estadoSiguiente == "".join(canales[canal][i + 1] for canal in canales):
                        probabilidades[estado][j] += 1

    return probabilidades


def generateTable1(data: dict, headers: list) -> str:
    table_data = []
    for key, values in data.items():
        table_data.append([key] + values)
    return tb.tabulate(table_data, headers, tablefmt="fancy_grid")


def probabilidadesEstadosAnteriores(canales: dict, estados: list) -> dict:
    probabilidades = {estado: [0] * len(estados) for estado in estados}
    num_tiempos = len(list(canales.values())[0])

    for estado in estados:
        for i in range(num_tiempos - 1, 0, -1):
            if estado == "".join(canales[canal][i] for canal in canales):
                for j, estadoSiguiente in enumerate(estados):
                    if estadoSiguiente == "".join(canales[canal][i - 1] for canal in canales):
                        probabilidades[estado][j] += 1

    return probabilidades

def frecuenciaEstadosAnteriores(canales: dict, estados: list) -> dict:
    num_canales = len(canales)
    frecuencias = {estado: [0] * num_canales for estado in estados}
    num_tiempos = len(list(canales.values())[0])

    for estado in estados:
        for i in range(num_tiempos - 1, 0, -1):
            if estado == "".join(canales[canal][i] for canal in canales):
                for j, canal in enumerate(canales):
                    if canales[canal][i - 1] == "1":
                        frecuencias[estado][j] += 1

    return frecuencias

#Impresión de las tablas

varFrecuenciaEstados = frecuenciaEstados(canales, estadosExistentes(canales));
varEstadosExistentes = estadosExistentes(canales)

# tableEstadoCanalF = generateTable1(probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])
# print(tableEstadoCanalF)
#print(probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])

# tableEstadoEstadoF = generateTable1(probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes)), varEstadosExistentes)
# print(tableEstadoEstadoF)
# print(probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes)), varEstadosExistentes)

diccionarioProb = probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes))
# print(diccionarioProb)
# tableEstadoCanalP = generateTable1(probabilidades(varFrecuenciaEstados, frecuenciaEstadosAnteriores(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])
# print(tableEstadoCanalP)

# tableEstadoEstadoP = generateTable1(probabilidades(varFrecuenciaEstados, probabilidadesEstadosAnteriores(canales, varEstadosExistentes)), varEstadosExistentes)
# print(tableEstadoEstadoP) 


"""Dado el diccionario de probabilidades de estados siguientes, retorna las probabilidades de un estado dado"""
def distribucionEstado(probabilidades: dict, estado: str) -> list:
    return probabilidades[estado]

# print(distribucionEstado(diccionarioProb, '000'))

""""Dado el diccionaro que tiene las probabilidades del canal siguiente dado un estado actual,
usa numpy para generar una distribución de probabilidades de uno de los canales dado un estado actual"""
def distribucionCanal(probabilidades: dict, estado: str, canal: str) -> list:
    return np.random.choice([0, 1], p=[1 - probabilidades[estado][int(canal)], probabilidades[estado][int(canal)]])

#print(distribucionCanal(diccionarioProb, '000', '0'))

"""Dado el diccionario de probabilidades de estados siguientes, necesito marginalizar un canal del estado siguiente, es decir, sumar las probabilidades
de los estados siguientes que queden iguales dado que se quite uno de sus canales, por ejemplo, si el estado actual es 000 y marginalizo el canal 0 de los estados siguientes,
 entonces quedaría asi: 000: [0.0,0.0,0.5,0.5], debido a que sumo los estados siguientes 000 y 100, los 010 y 110 quedan iguales, y los 001 y 101 quedan iguales.
 o por ejemplo, si el estado actual es 000 y marginalizo el canal 1 de los estados siguientes,
 entonces quedaría asi: 000: [0.0,0.0,0.5,0.5], debido a que sumo los estados siguientes 000 y 010, los 100 y 110 quedan iguales, y los 111 y 101 y el 001 y 011 quedan iguales."""
def marginalizarCanal(probabilidades: dict, canal: str) -> dict:
    for estado in probabilidades:
        if canal == '0':
            probabilidades[estado] = [probabilidades[estado][0] + probabilidades[estado][1],
                                      probabilidades[estado][2] + probabilidades[estado][3],
                                      probabilidades[estado][4] + probabilidades[estado][5] + probabilidades[estado][6] + probabilidades[estado][7]]
            probabilidades[estado] = probabilidades[estado][:3]
        elif canal == '1':
            probabilidades[estado] = [probabilidades[estado][0] + probabilidades[estado][2],
                                      probabilidades[estado][1] + probabilidades[estado][3],
                                      probabilidades[estado][4] + probabilidades[estado][6],
                                      probabilidades[estado][5] + probabilidades[estado][7]]
            probabilidades[estado] = probabilidades[estado][:2] + probabilidades[estado][2:4]
        elif canal == '2':
            probabilidades[estado] = [probabilidades[estado][0] + probabilidades[estado][1],
                                      probabilidades[estado][2] + probabilidades[estado][3],
                                      probabilidades[estado][4] + probabilidades[estado][5],
                                      probabilidades[estado][6] + probabilidades[estado][7]]
            probabilidades[estado] = probabilidades[estado][:2] + probabilidades[estado][2:4]
    return probabilidades

# Ejemplo de uso
diccionarioProb = {'000': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
print(marginalizarCanal(diccionarioProb, '1'))


print(marginalizarCanal(diccionarioProb, '0'))

































    