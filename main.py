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



"""Devuelve un diccionario el cual por cada estado existente en la lista de estados, devuelve cuántas veces se encuentra este en los canales de comunicación, pero si el estado se encuentra al final, este no se cuenta."""
def frecuenciaEstados(canales: dict, estados: list) -> dict:
    num_canales = len(canales)
    frecuencias = {estado: 0 for estado in estados}
    num_tiempos = len(list(canales.values())[0])

    for estado in estados:
        for i in range(num_tiempos - 1):
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

"""
    Calcula las probabilidades condicionales de transición entre estados.

    Parameters:
        - frecuenciaEstados (dict): Un diccionario que contiene las frecuencias de ocurrencia de estados en canales de comunicación.
        - frecuenciaEstadosSiguientes (dict): Un diccionario que contiene las frecuencias de ocurrencia de estados siguientes en canales de comunicación.

    Returns:
        dict: Un diccionario que representa las probabilidades condicionales de transición entre estados.
              Las claves son estados y los valores son listas de probabilidades correspondientes a cada estado siguiente.
    """
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

"""
    Genera una tabla formateada a partir de un diccionario de datos y una lista de encabezados.

    Parameters:
        - data (dict): Un diccionario donde las claves son etiquetas y los valores son listas de datos asociados.
        - headers (list): Una lista que contiene los encabezados de la tabla.

    Returns:
        str: Una representación en formato de cadena de la tabla generada utilizando la librería 'tabulate'.
    """
def generateTable1(data: dict, headers: list) -> str:
    table_data = []
    for key, values in data.items():
        table_data.append([key] + values)
    return tb.tabulate(table_data, headers, tablefmt="fancy_grid")

"""
    Calcula las probabilidades de transición entre estados anteriores en canales de comunicación.

    Parameters:
        - canales (dict): Un diccionario con claves que representan nombres de canales y valores que son listas de estados en cada instante de tiempo.
        - estados (list): Una lista de estados para los cuales se calcularán las probabilidades de transición desde estados anteriores.

    Returns:
        dict: Un diccionario que representa las probabilidades de transición desde estados anteriores para cada estado en la lista proporcionada.
              Las claves son estados y los valores son listas de probabilidades correspondientes a cada estado siguiente en la lista.
"""
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

"""
    Calcula la frecuencia de estados anteriores en canales de comunicación.

    Parameters:
        - canales (dict): Un diccionario con claves que representan nombres de canales y valores que son listas de estados en cada instante de tiempo.
        - estados (list): Una lista de estados para los cuales se calculará la frecuencia de estados anteriores.

    Returns:
        dict: Un diccionario que representa la frecuencia de estados anteriores para cada estado en la lista proporcionada.
              Las claves son estados y los valores son listas de frecuencias correspondientes a cada canal de comunicación.
"""
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

tableEstadoCanalF = generateTable1(probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])
print(tableEstadoCanalF)
#print(probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])

tableEstadoEstadoF = generateTable1(probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes)), varEstadosExistentes)
print(tableEstadoEstadoF)
# print(probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes)), varEstadosExistentes)

diccionarioProb = probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes))
diccionarioProbCanal = probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes))
# print(diccionarioProb)
# tableEstadoCanalP = generateTable1(probabilidades(varFrecuenciaEstados, frecuenciaEstadosAnteriores(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])
# print(tableEstadoCanalP)

# tableEstadoEstadoP = generateTable1(probabilidades(varFrecuenciaEstados, probabilidadesEstadosAnteriores(canales, varEstadosExistentes)), varEstadosExistentes)
# print(tableEstadoEstadoP) 


"""Dado el diccionario de probabilidades de estados siguientes, retorna las probabilidades de un estado dado"""
def distribucionEstado(probabilidades: dict, estado: str) -> list:
    return probabilidades[estado]

# print(distribucionEstado(diccionarioProb, '000'))

"""
    Calcula la distribución de un canal específico a partir de un diccionario de probabilidades de transición.

    Parameters:
        - probabilidades (dict): Un diccionario con claves que representan estados y valores que son listas de probabilidades de transición.
        - canal (int): Índice del canal para el cual se calculará la distribución.

    Returns:
        dict: Un diccionario que representa la distribución del canal especificado,
              donde las claves son estados y los valores son listas de probabilidades y sus complementos.
"""
def distribucionCanal(probabilidades: dict, canal: int) -> dict:
    distribucion = {}
    for estado in probabilidades:
        distribucion[estado] = [probabilidades[estado][canal], 1 - probabilidades[estado][canal]]
    return distribucion


"""Dado el diccionario de distribuciones de un canal dado un estado actual, conviertelo en una matriz"""
def matrizDistribucionCanal(distribucion: dict) -> list:
    matriz = []
    for estado in distribucion:
        matriz.append(distribucion[estado])
    return matriz

"""
    Marginaliza una fila específica en un diccionario, reduciendo la dimensión de los datos.
    
    Parameters:
        - diccionario (dict): Un diccionario con claves que representan estados y valores que son listas asociadas.
        - indice (int): Índice de la posición que se eliminará en cada clave del diccionario.

    Returns:
        dict: Un nuevo diccionario marginalizado, donde las claves son versiones reducidas de las claves originales,
              y los valores son promedios ponderados de las listas originales en la posición especificada.
    """
def marginalizar_fila(diccionario, indice):
    nuevo_diccionario = {}

    for clave, valores in diccionario.items():
        nueva_clave = clave[:indice] + clave[indice + 1:]
        if nueva_clave in nuevo_diccionario:
            nuevo_diccionario[nueva_clave] = [sum(x)/2 for x in zip(nuevo_diccionario[nueva_clave], valores)]
        else:
            nuevo_diccionario[nueva_clave] = valores

    return nuevo_diccionario

"""
    Marginaliza una columna específica en un diccionario, reduciendo la dimensión de los datos.

    Parameters:
        - diccionario (dict): Un diccionario con claves que representan estados y valores que son listas asociadas.
        - indice (int): Índice de la posición que se eliminará en cada clave del diccionario.

    Returns:
        dict: Un nuevo diccionario marginalizado, donde las claves son versiones reducidas de las claves originales,
              y los valores son sumas acumulativas de las listas originales en la posición especificada.
    """
def marginalizar_columna(diccionario, indice):
    nuevo_diccionario = {}

    for clave, valores in diccionario.items():
        nueva_clave = clave[:indice] + clave[indice + 1:]
        if nueva_clave in nuevo_diccionario:
            nuevo_diccionario[nueva_clave] = [sum(x) for x in zip(nuevo_diccionario[nueva_clave], valores)]
        else:
            nuevo_diccionario[nueva_clave] = valores

    return nuevo_diccionario

"""
    Transpone una matriz representada por un diccionario, intercambiando filas y columnas,
    y reorganiza las claves del diccionario según una nueva lista de claves.

    Parameters:
        - diccionario (dict): Un diccionario donde las claves representan las filas y los valores son listas asociadas.
        - keys (list): Una lista de nuevas claves que se utilizarán para organizar el diccionario resultante.

    Returns:
        dict: Un nuevo diccionario transpuesto, donde las claves son las nuevas claves proporcionadas,
              y los valores son las columnas correspondientes de la matriz original.
    """
def trasponerMatrizDictKeys(diccionario: dict, keys: list) -> dict:
    matriz = np.array(list(diccionario.values()))
    matrizTranspuesta = np.transpose(matriz)
    return {key: matrizTranspuesta[i].tolist() for i, key in enumerate(keys)}

"""
    Calcula la distribución del sistema de partidos políticos a partir de probabilidades de transición.

    Parameters:
        - probabilidades (dict): Un diccionario con claves que representan estados y valores que son listas de probabilidades de transición.
        - canales_futuros (list): Una lista de índices de canales que se eliminarán al calcular la distribución del sistema futuro.
        - canales_actuales (list): Una lista de índices de canales que se eliminarán al calcular la distribución del sistema actual.

    Returns:
        dict: Un diccionario que representa la distribución del sistema de partidos políticos,
              donde las claves son estados reducidos y los valores son listas de probabilidades correspondientes.
"""
def distribucion_sistema_partido(probabilidades: dict, canales_futuros: list, canales_actuales: list) -> dict:
    res = trasponerMatrizDictKeys(probabilidades, list(probabilidades.keys()))
    for i in range(len(canales_futuros)):
        res = marginalizar_columna(res, canales_futuros[i])
    res = trasponerMatrizDictKeys(res, list(probabilidades.keys()))
    for i in range(len(canales_actuales)):
        res = marginalizar_fila(res, canales_actuales[i])
    return res[estado_actual]

"""El primer array sirve para la marginalización de columnas, el segundo para la marginalización de filas"""""
dist_partida = distribucion_sistema_partido(diccionarioProb, [0], [])

print(dist_partida)































    