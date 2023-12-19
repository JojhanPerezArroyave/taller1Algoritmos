import tabulate as tb
import numpy as np
from scipy.stats import wasserstein_distance
import string
import asyncio
import os
import curses

def listar_archivos_txt(ruta):
    archivos_txt = [archivo for archivo in os.listdir(ruta) if archivo.endswith(".txt")]
    return archivos_txt

def interfaz(stdscr):
    curses.curs_set(0)  # Ocultar el cursor
    stdscr.clear()

    ruta_actual = os.getcwd() + "/canalesPrueba"
    archivos_txt_disponibles = listar_archivos_txt(ruta_actual)

    if not archivos_txt_disponibles:
        stdscr.addstr(0, 0, "No hay archivos .txt en la carpeta actual.")
        stdscr.refresh()
        stdscr.getch()
        return

    opcion_seleccionada = 0

    while True:
        stdscr.clear()

        for i, archivo in enumerate(archivos_txt_disponibles):
            if i == opcion_seleccionada:
                stdscr.addstr(i, 0, f"> {archivo}")
            else:
                stdscr.addstr(i, 0, f"  {archivo}")

        stdscr.refresh()

        tecla = stdscr.getch()

        if tecla == curses.KEY_UP and opcion_seleccionada > 0:
            opcion_seleccionada -= 1
        elif tecla == curses.KEY_DOWN and opcion_seleccionada < len(archivos_txt_disponibles) - 1:
            opcion_seleccionada += 1
        elif tecla == ord('\n'):
            archivo_seleccionado = archivos_txt_disponibles[opcion_seleccionada]
            procesar_archivo(archivo_seleccionado)
            break

canales = {}
def procesar_archivo(nombre_archivo):
    with open("./canalesPrueba/" + nombre_archivo, 'r') as archivo_txt:
        for linea in archivo_txt:
            partes = linea.strip().split()
            if partes:
                canal, valores = partes[0], partes[1:]
                canales[canal] = valores

    # Haces lo que necesites con el diccionario 'canales'
    print(f"Contenido del archivo {nombre_archivo} procesado:")
    print(canales)

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
        if indice >= len(clave):
            indice -= 1
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
    Calcula la distribución del sistema a partir de probabilidades de transición.

    Parameters:
        - probabilidades (dict): Un diccionario con claves que representan estados y valores que son listas de probabilidades de transición.
        - canales_futuros (list): Una lista de índices de canales que se eliminarán al calcular la distribución del sistema futuro.
        - canales_actuales (list): Una lista de índices de canales que se eliminarán al calcular la distribución del sistema actual.

    Returns:
        dict: Un diccionario que representa la distribución del sistema,
              donde las claves son estados reducidos y los valores son listas de probabilidades correspondientes.
"""
def distribucion_sistema_partido(probabilidades: dict, canales_futuros: list, canales_actuales: list) -> dict:
    if -1 not in canales_futuros:
        return distribucion_vacia(canales_actuales)
    if -1 not in canales_actuales:
        return distribucion_vacia(canales_futuros)
    res = trasponerMatrizDictKeys(probabilidades, list(probabilidades.keys()))
    for i in range(len(canales_futuros)):
        if canales_futuros[i] != -1:
            res = marginalizar_columna(res, canales_futuros[i])
    res = trasponerMatrizDictKeys(res, list(probabilidades.keys()))
    for i in range(len(canales_actuales)):
        if canales_actuales[i] != -1:
            res = marginalizar_fila(res, canales_actuales[i])
    return res

"""
    Calcula la distribución vacía del sistema.

    Parameters:
        - canales (list): Una lista de índices de canales, donde se contarán los canales existentes para determinar la distribución vacía.

    Returns:
        dict: Un diccionario que representa la distribución vacía del sistema.
              La clave 'E' representa el estado vacío, y el valor es una lista de probabilidades uniformes.

"""
def distribucion_vacia(canales: list):
    existentes = 0
    for element in canales:
        if element == -1:
            existentes += 1
    res = {}
    res['E'] = [1/(2**existentes) for i in range(2**existentes)]
    return res

"""
    Genera recursivamente todas las combinaciones posibles a partir de las posiciones restantes.

    Parameters:
        - posiciones_restantes (list): Una lista de listas, donde cada sublista representa los valores posibles para una posición.
        - combinacion_actual (list): Una lista que representa la combinación actual en construcción.
        - todas_combinaciones (list): Una lista que se actualizará con todas las combinaciones generadas.

    Returns:
        None: La función no devuelve un valor explícito, pero actualiza la lista `todas_combinaciones` con las combinaciones generadas.

    Note:
        Esta función utiliza la recursividad para generar todas las combinaciones posibles a partir de las posiciones restantes.
"""
def generar_combinaciones(posiciones_restantes, combinacion_actual, todas_combinaciones):
    if not posiciones_restantes:
        todas_combinaciones.append(tuple(combinacion_actual))
        return

    # Recursivamente generar combinaciones para cada posición
    for valor in posiciones_restantes[0]:
        generar_combinaciones(posiciones_restantes[1:], combinacion_actual + [valor], todas_combinaciones)

"""
    Calcula y devuelve el estado actual correspondiente a los canales especificados.

    Parameters:
        - estado_actual (str): Una cadena que representa el estado actual completo.
        - canales (list): Una lista de índices de canales, donde se seleccionarán los canales relevantes para el estado.

    Returns:
        str: Una cadena que representa el estado actual correspondiente a los canales especificados.

    Note:
        Esta función construye un nuevo estado excluyendo las posiciones indicadas por los canales con valor -1.
"""
def indice_estado_actual(estado_actual: str, canales: list):
    estado = ''
    for i in range(len(canales)):
        if canales[i] == -1:
            estado += estado_actual[i]
    return estado
            
"""
    Selecciona y devuelve el elemento en la posición opuesta en una lista en relación con el índice dado.

    Parameters:
        - lst (list): Una lista de elementos.
        - index (int): El índice para el cual se seleccionará el elemento opuesto.

    Returns:
        object or None: El elemento en la posición opuesta con respecto al índice dado.
                        Devuelve None si el índice opuesto está fuera del rango de la lista.

    Note:
        Esta función calcula el índice opuesto y verifica si está dentro del rango de la lista antes de devolver el elemento correspondiente.
"""
def opposite_index_select(lst, index):
    # Calcular el indice opuesto
    opposite_index = len(lst) - 1 - index
    
    # Verificar que el indice opuesto esté dentro del rango
    if 0 <= opposite_index < len(lst):
        # Devolver el elemento en el indice opuesto
        return lst[opposite_index]
    else:
        # Devolver None si el indice opuesto está fuera del rango
        return None

"""
    Implementa el algoritmo de ordenación merge sort de manera asíncrona y selecciona el mínimo valor.

    Parameters:
        - arr (list): Una lista de elementos que se ordenarán y de la cual se seleccionará el mínimo.
        - sistema_original: [Describir el tipo]: El sistema original al que se asocian los elementos en la lista.
        - estado_actual: [Describir el tipo]: El estado actual utilizado en la selección del mínimo.
        - combinaciones: [Describir el tipo]: Una estructura de datos para almacenar combinaciones.

    Returns:
        list: Una lista ordenada que resulta de aplicar merge sort al arreglo de entrada.

    Note:
        Esta función utiliza el algoritmo merge sort de manera asíncrona para ordenar la lista y seleccionar el mínimo valor.
"""
async def merge_sort_select_min(arr, sistema_original, estado_actual, combinaciones):
    if len(arr) <= 1:
        return arr

    # Divide la lista en mitades
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Llamada recursiva para ordenar y seleccionar el mínimo en las mitades
    left_half = await merge_sort_select_min(left_half, sistema_original, estado_actual, combinaciones)
    right_half = await merge_sort_select_min(right_half, sistema_original, estado_actual, combinaciones)

    # Fusionar las mitades ordenadas
    result = await merge(left_half, right_half, sistema_original, estado_actual, combinaciones)

    return result

"""
    Fusiona dos listas ordenadas de manera asíncrona y selecciona el menor valor de acuerdo con ciertos criterios.

    Parameters:
        - left (list): Una lista ordenada.
        - right (list): Otra lista ordenada.
        - sistema_original: [Describir el tipo]: El sistema original al que se asocian los elementos en las listas.
        - estado_actual: [Describir el tipo]: El estado actual utilizado en la selección del menor valor.
        - combinaciones: [Describir el tipo]: Una estructura de datos para almacenar combinaciones.

    Returns:
        list: Una lista fusionada y ordenada que resulta de combinar las listas de entrada.

    Note:
        Esta función fusiona dos listas ordenadas de manera asíncrona y selecciona el menor valor utilizando ciertos criterios.
"""
async def merge(left, right, sistema_original, estado_actual, combinaciones):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        # Comparar elementos y seleccionar el menor
        lado_izq = await obtener_emd_sistema(sistema_original, estado_actual, combinaciones, left[i][0], left[i][1])
        lado_der = await obtener_emd_sistema(sistema_original, estado_actual, combinaciones, right[j][0], right[j][1])
        if lado_izq < lado_der :
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Agregar los elementos restantes, si los hay
    result.extend(left[i:])
    result.extend(right[j:])

    return result

"""
    Calcula la distancia de Earth Mover's (EMD) entre dos distribuciones de probabilidad en el contexto de un sistema político.

    Parameters:
        - sistema_original (dict): Un diccionario que representa el sistema político original.
        - estado_actual (str): El estado actual del sistema político.
        - combinaciones (list): Una lista de combinaciones de canales relevantes para el sistema.
        - combinacion_presente (tuple): Una tupla que representa la combinación presente.
        - combinacion_futuro (tuple): Una tupla que representa la combinación futura.

    Returns:
        float: La distancia de Earth Mover's entre las distribuciones de probabilidad asociadas con las combinaciones dadas.

    Note:
        Esta función utiliza el algoritmo de distancia de Earth Mover's para calcular la diferencia entre las distribuciones de probabilidad
        en dos particiones del sistema político, considerando combinaciones presentes y futuras.
"""
async def obtener_emd_sistema(sistema_original: dict, estado_actual: str, combinaciones: list, combinacion_presente: tuple, combinacion_futuro: tuple):
    distribucion_original = sistema_original[estado_actual]
    estado_particion_1 = ''
    estado_particion_2 = ''
    particion_1 = distribucion_sistema_partido(sistema_original, combinacion_presente, combinacion_futuro)
    opuesto_futuro = opposite_index_select(combinaciones, combinaciones.index(combinacion_presente) )
    opuesto_actual = opposite_index_select(combinaciones, combinaciones.index(combinacion_futuro))
    particion_2 = distribucion_sistema_partido(sistema_original, opuesto_futuro, opuesto_actual)
    if 'E' in particion_1.keys():
        estado_particion_1 = 'E'
        estado_particion_2 = indice_estado_actual(estado_actual, opposite_index_select(combinaciones, combinaciones.index(combinacion_futuro)))
    if 'E' in particion_2.keys():
        estado_particion_1 = indice_estado_actual(estado_actual, combinacion_futuro)
        estado_particion_2 = 'E'
    if 'E' not in particion_1.keys() and 'E' not in particion_2.keys():
        estado_particion_1 = indice_estado_actual(estado_actual, combinacion_futuro)
        estado_particion_2 = indice_estado_actual(estado_actual, opposite_index_select(combinaciones, combinaciones.index(combinacion_futuro)))
    distribucion_particion_1 = particion_1[estado_particion_1]
    distribucion_particion_2 = particion_2[estado_particion_2]
    distribucion_combinada = np.kron(distribucion_particion_1, distribucion_particion_2)

    distancia = wasserstein_distance(np.array(distribucion_original), distribucion_combinada)

    return distancia

"""
    Genera y devuelve todas las combinaciones posibles de estados presente y futuro, considerando restricciones específicas.

    Returns:
        list: Una lista de tuplas que representan todas las combinaciones permitidas de estados presente y futuro.

    Note:
        Esta función utiliza restricciones específicas para generar combinaciones posibles de estados presente y futuro,
        excluyendo combinaciones específicas para evitar duplicados o combinaciones que no cumplen con ciertos criterios.
"""
def obtener_combinaciones_presente_futuro():
    combinaciones_totales = []
    # Definir restricciones para cada posición
    restricciones = [[0, -1], [1, -1], [2, -1]]  # Puedes agregar más restricciones según sea necesario

    # Generar todas las combinaciones posibles según las restricciones
    todas_combinaciones = []
    generar_combinaciones(restricciones, [], todas_combinaciones)

    for futuro in todas_combinaciones:
        for presente in todas_combinaciones:
            if not (futuro == (0,1,2) and presente == (0,1,2)) and not (futuro == (-1,-1,-1) and presente == (0,1,2)) and not (futuro == (0,1,2) and presente == (-1,-1,-1)) and not (futuro == (-1,-1,-1) and presente == (-1,-1,-1)):
                combinaciones_totales.append((presente, futuro))
    return combinaciones_totales

"""
    Convierte una solución representada como una tupla de índices en una lista de letras correspondientes.

    Parameters:
        - solucion (tuple): Una tupla de índices que representa una solución.

    Returns:
        list: Una lista de letras correspondientes a los índices de la solución. Los elementos en posiciones no especificadas son excluidos.

    Note:
        Esta función asigna letras a los índices específicos en la solución, excluyendo aquellos que están marcados con el valor -1.
"""
def convertir_solucion_letras(solucion: tuple):
    solucion_letras = []
    for i in range(len(solucion)):
        if solucion[i] == -1:
            if i == 0:
                solucion_letras.append('A')
            if i == 1:
                solucion_letras.append('B')
            if i == 2:
                solucion_letras.append('C')
    if len(solucion_letras) == 0:
        solucion_letras.append('Ø')
    return solucion_letras

async def main():
    #Impresión de las tablas
    curses.wrapper(interfaz)

    varFrecuenciaEstados = frecuenciaEstados(canales, estadosExistentes(canales));
    varEstadosExistentes = estadosExistentes(canales)

    tableEstadoCanalF = generateTable1(probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])
    print(tableEstadoCanalF)
    #print(probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])

    tableEstadoEstadoF = generateTable1(probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes)), varEstadosExistentes)
    print(tableEstadoEstadoF)
    # print(probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes)), varEstadosExistentes)

    diccionarioProb = probabilidades(varFrecuenciaEstados, probabilidadesEstadosSiguientes(canales, varEstadosExistentes))
    #diccionarioProbCanal = probabilidades(varFrecuenciaEstados, frecuenciaEstadosSiguientes(canales, varEstadosExistentes))
    # print(diccionarioProb)
    # tableEstadoCanalP = generateTable1(probabilidades(varFrecuenciaEstados, frecuenciaEstadosAnteriores(canales, varEstadosExistentes)), [f'Canal {letter}' for letter in string.ascii_uppercase[:len(canales.values())]])
    # print(tableEstadoCanalP)

    # tableEstadoEstadoP = generateTable1(probabilidades(varFrecuenciaEstados, probabilidadesEstadosAnteriores(canales, varEstadosExistentes)), varEstadosExistentes)
    # print(tableEstadoEstadoP) 
    """El primer array sirve para la marginalización de columnas, el segundo para la marginalización de filas"""""
    #dist_partida = distribucion_sistema_partido(diccionarioProb, [0, 1, -1], [0, -1, -1])

    #print(dist_partida)
    combinaciones = []
    generar_combinaciones([[0, -1], [1, -1], [2, -1]], [], combinaciones)
    min_emd = await merge_sort_select_min(obtener_combinaciones_presente_futuro(), diccionarioProb, '000', combinaciones)

    print(min_emd[0])
    print(f'Particion 1: {convertir_solucion_letras(min_emd[0][0])} y {convertir_solucion_letras(min_emd[0][1])}') 
    print(f'Particion 2: {convertir_solucion_letras(opposite_index_select(combinaciones, combinaciones.index(min_emd[0][0])))} y {convertir_solucion_letras(opposite_index_select(combinaciones, combinaciones.index(min_emd[0][1])))}') 

asyncio.run(main())

