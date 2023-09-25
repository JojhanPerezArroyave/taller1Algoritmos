import numpy as np
import tabulate as tb

canales = {
    "canalA" : ["0", "1", "1", "0", "1", "1", "0", "0", "0", "1", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "1", "0", "0", "0", "1"],
    "canalB" : ["0", "0", "0", "1", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "1", "0", "1", "1", "0", "1", "1", "0", "1", "1", "1", "1", "0", "0", "0"],
    "canalC" : ["0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "1", "0", "1", "1", "0", "1", "1", "1", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0"],
}


"""Devuelve una lista de los estados existentes en los canales de comunicación. Los estados se definen como los valores que toman los canales de comunicación en un instante de tiempo determinado.
Los tiempos están definidos por cada posición de cada canal de comunicación. Por ejemplo, el estado 1 es 000 y el estado 2 es 101."""
def estadosExistentes(canales: dict) -> list:
    estados = set()
    num_tiempos = len(list(canales.values())[0]) 
    
    for i in range(num_tiempos):
        estado = "".join(canales[canal][i] for canal in canales)
        estados.add(estado)
    
    return list(estados)


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
    num_canales = len(canales)
    probabilidades = {estado: [0] * len(estados) for estado in estados}
    num_tiempos = len(list(canales.values())[0])

    for estado in estados:
        for i in range(num_tiempos - 1):
            if estado == "".join(canales[canal][i] for canal in canales):
                for j, estadoSiguiente in enumerate(estados):
                    if estadoSiguiente == "".join(canales[canal][i + 1] for canal in canales):
                        probabilidades[estado][j] += 1

    return probabilidades
"""Imprimir el metodo frecuenciaEstadosSiguientes"""


def generateTable1(data: dict, headers: list) -> str:
    table_data = []
    for key, values in data.items():
        table_data.append([key] + values)
    return tb.tabulate(table_data, headers, tablefmt="fancy_grid")

table_data = []

#prob = probabilidades(frecuenciaEstados(canales, estadosExistentes(canales)), frecuenciaEstadosSiguientes(canales, estadosExistentes(canales)))
#headers = ["1"]*len(canales.values())
#for key, values in prob.items():
#    table_data.append([key] + values)

print(probabilidades(frecuenciaEstados(canales, estadosExistentes(canales)), frecuenciaEstadosSiguientes(canales, estadosExistentes(canales))))

#tabla = tb.tabulate(probabilidades(frecuenciaEstados(canales, estadosExistentes(canales)), frecuenciaEstadosSiguientes(canales, estadosExistentes(canales))), tablefmt='orgtbl', headers='keys')


tableEstadoCanalF = generateTable1(probabilidades(frecuenciaEstados(canales, estadosExistentes(canales)), frecuenciaEstadosSiguientes(canales, estadosExistentes(canales))), ["1"]*len(canales.values()))
print(tableEstadoCanalF)

tableEstadoEstadoF = generateTable1(probabilidades(frecuenciaEstados(canales, estadosExistentes(canales)), probabilidadesEstadosSiguientes(canales, estadosExistentes(canales))), estadosExistentes(canales))
print(tableEstadoEstadoF)




    