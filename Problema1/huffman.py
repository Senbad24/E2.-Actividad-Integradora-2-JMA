"""
===============================================================
 Program : huffman.py
 Author  : Juan Manuel Ahuactzin
 Date    : 2025-10-13
 Version : 1.1
===============================================================
Description:
    Implementation of Huffman coding using a min-heap-based 
    approach to build the optimal binary prefix tree.
    Each node represents a character and its frequency, and 
    the resulting tree provides the Huffman code for each symbol.
    The class includes a __str__ method to visualize the tree
    structure and binary code paths (0/1).

    Adapted from: GeeksforGeeks. (s. f.). Huffman Coding | Greedy 
    Algo-3. GeeksforGeeks. 
    https://www.geeksforgeeks.org/dsa/huffman-coding-greedy-algo-3/

Usage:
    python 05_huffman.py

Example:
    python 05_huffman.py

Dependencies:
    - Python 3.x
    - Standard libraries: heapq

Notes:
    - The program constructs the Huffman tree and prints its
      hierarchical representation, followed by the Huffman code
      assigned to each character.
    - The __str__ method allows visualization of the recursive 
      structure, indicating '0' for left edges and '1' for right 
      edges.
===============================================================
"""


import heapq
import json
import subprocess
import pandas as pd
import numpy as np
from bitarray import bitarray
from pathlib import Path
from collections import Counter
import pickle
from statistics import mean, stdev
from time import perf_counter

def show_dataframe(df, decimals=2):
    try:
        from IPython import get_ipython
        from IPython.display import display

        shell = get_ipython()

        if shell is not None and shell.__class__.__name__ == 'ZMQInteractiveShell':
            display(df.round(decimals))
            return

    except ImportError:
        pass

    print(df.round(decimals).to_string(index=False))


def print_huffman_solution(symbols, frequencies, codes):
    
    df, H, L = tabla_huffman(symbols, frequencies, codes)

    show_dataframe(df, 2)

    print(f"Entropía: {H:.4f} bits/símbolo")
    print(f"Longitud promedio Huffman: {L:.4f} bits/símbolo")


# Class to represent huffman tree 
class Node:
    def __init__(self, x, character=""):
        self.data = x
        self.char = character
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.data < other.data
    
    def __str__(self, level=0, prefix="Root: ", cumchain=""):
        """Devuelve una representación jerárquica del árbol."""
        result = "   " * level + f"{prefix}({self.char}:{self.data}):{cumchain}\n"
        if self.left:
            result += self.left.__str__(level + 1, prefix="L-0- ", cumchain=cumchain+'0')
        if self.right:
            result += self.right.__str__(level + 1, prefix="R-1- ", cumchain=cumchain+'1')
        return result
    
# Function to traverse tree in preorder 
# manner and push the huffman representation 
# of each character.
def preOrder(root, ans, curr):
    if root is None:
        return

    # Leaf node represents a character.
    if root.left is None and root.right is None:
        ans[root.char] = curr
        return

    preOrder(root.left, ans, curr + '0')
    preOrder(root.right, ans, curr + '1')

def huffmanCodes(s, freq):
    # Code here
    n = len(s)

    # Min heap for node class.
    pq = []
    for i in range(n):
        tmp = Node(freq[i], s[i])
        heapq.heappush(pq, tmp)

    # Construct huffman tree.
    while len(pq) >= 2:
        # Left node 
        l = heapq.heappop(pq)

        # Right node 
        r = heapq.heappop(pq)

        newNode = Node(l.data + r.data)
        newNode.left = l
        newNode.right = r

        heapq.heappush(pq, newNode)

    root = heapq.heappop(pq)
    codes_dic = {}
    preOrder(root, codes_dic, "")
    return codes_dic, root

# Hecho con ayuda de IA
def visualizar_arbol(root, salida="arbol_huffman.svg"):
    lineas = ["digraph Huffman {", "    graph [ordering=out];"]

    def visitar(nodo, identificador):
        etiqueta = str(nodo.data)
        if nodo.left is None and nodo.right is None:
            etiqueta = f"{nodo.char!r}: {nodo.data}"

        etiqueta_dot = json.dumps(etiqueta, ensure_ascii=False)
        lineas.append(f"    {identificador} [label={etiqueta_dot}];")

        #Recorre hacia la izqueirda y la derecha
        for bit, hijo in (("0", nodo.left), ("1", nodo.right)):
            if hijo is not None:
                id_hijo = identificador + bit
                lineas.append(f'    {identificador} -> {id_hijo} [label="{bit}"];')
                visitar(hijo, id_hijo)

    if root is not None:
        visitar(root, "n")
    lineas.append("}")

    ruta = Path(salida)
    subprocess.run(
        ["dot", "-Tsvg", "-o", str(ruta)],
        input="\n".join(lineas),
        encoding="utf-8",
        check=True,
    )
    return ruta


def tabla_huffman(simbolos, frecuencias, codigos):
    """
    Genera una tabla de análisis de la codificación Huffman.

    Parámetros:
        simbolos: lista de símbolos
        frecuencias: lista de frecuencias
        codigos: diccionario {simbolo: codigo_binario}

    Retorna:
        df: DataFrame con los resultados
        H: entropía de Shannon (bits/símbolo)
        L: longitud promedio de Huffman (bits/símbolo)
    """

    if len(simbolos) != len(frecuencias):
        raise ValueError("Las listas deben tener igual longitud.")

    if (np.array(frecuencias) <= 0).any():
        raise ValueError("Las frecuencias deben ser positivas.")

    if set(simbolos) != set(codigos):
        raise ValueError("Los símbolos y códigos no coinciden.")

    # Probabilidades
    p = np.array(frecuencias) / sum(frecuencias)

    # Longitudes teóricas
    lt = -np.log2(p)

    # Longitudes de los códigos Huffman
    l = np.array([len(codigos[s]) for s in simbolos])

    # Construcción del DataFrame
    df = pd.DataFrame({
        'Caracter': simbolos,
        'Frecuencia': frecuencias,
        'Probabilidad': p,
        'Longitud teórica: lt': lt,
        'p*lt': p * lt,
        'Código': [codigos[s] for s in simbolos],
        'Longitud: l': l,
        'p*l': p * l
    })

    # Entropía y longitud promedio
    H = np.sum(p * lt)
    L = np.sum(p * l)

    return df, H, L

def codificar(ruta_huff, codes_dic, texto, root):
    # Pasar los codigos de strings a bitarrays
    codes = {}
    for char, code in codes_dic.items():
        codes[char] = bitarray(code)

    # bits es el archivo codificado, vamos agregamos la codificacion de cada caracter
    bits = bitarray()
    for char in texto:
        bits.extend(codes[char])

    # Calcular el padding, por si no llega a completar el ultimo byte
    padding = (-len(bits)) % 8

    # Generar el .huff y guardar el diccionario, padding, arbol y bits
    with ruta_huff.open("wb") as archivo:
        pickle.dump((codes_dic, padding, root), archivo)
        bits.tofile(archivo)

def leer_comprimido(ruta_huff):
    with ruta_huff.open("rb") as archivo:
        codes_dic, padding, root = pickle.load(archivo)
        bits = bitarray()
        bits.frombytes(archivo.read())

    # Las versiones anteriores guardaban 8 cuando no habia relleno.
    if 0 < padding < 8:
        del bits[-padding:]

    return bits, codes_dic, root


def decodificar_por_diccionario(bits, codes_dic):
    codes_dic_inv = {}
    for char, code in codes_dic.items():
        codes_dic_inv[code] = char

    codigo_actual = ""
    texto = ""
    for bit in bits:
        codigo_actual += "1" if bit else "0"

        if codigo_actual in codes_dic_inv:
            texto += codes_dic_inv[codigo_actual]
            codigo_actual = ""

    return texto


def decodificar_por_arbol(bits, root):
    texto = ""
    nodo = root
    for bit in bits:
        if bit:
            nodo = nodo.right
        else:
            nodo = nodo.left

        if nodo.right is None and nodo.left is None:
            texto += nodo.char
            nodo = root

    return texto


def comparar_tiempos_decodificacion():
    carpeta = Path(__file__).resolve().parent
    archivos = {"espanol_1": "Español", "ingles_1": "Inglés", "frances_1": "Francés"}
    filas = []

    for nombre, idioma in archivos.items():
        ruta_huff = carpeta / "comprimidos" / f"{nombre}.huff"
        bits, codes_dic, root = leer_comprimido(ruta_huff)
        metodos = (
            ("Diccionario inverso", decodificar_por_diccionario, codes_dic),
            ("Árbol de Huffman", decodificar_por_arbol, root),
        )

        for metodo, decodificador, datos in metodos:

            tiempos_ms = []
            for _ in range(200):
                inicio = perf_counter()
                decodificador(bits, datos)
                tiempos_ms.append((perf_counter() - inicio) * 1000)

            filas.append({
                "Archivo": ruta_huff.name,
                "Idioma": idioma,
                "Método": metodo,
                "Repeticiones": 200,
                "Promedio (ms)": mean(tiempos_ms),
                "Desviación estándar (ms)": stdev(tiempos_ms),
            })

    return pd.DataFrame(filas)


if __name__ == "__main__":
    # Prender para generar las graficas
    # TARDA 25 seg APROXIMADAMENTE
    generar_grafos = False 

    # Prender para generar la comparacion de los decodificadores, 
    # HACE 200 REPETICIONES DE CADA METODO CON 1 ARCHIVO DE CADA IDIOMA
    # TARDA 4 min APROXIMADAMENTE
    comparar_decodificacion = False

    carpeta = Path(__file__).with_name("textos")

    # Genera las carpetas para los .huff y los grafos (por si no llegaran a estar ya)
    (carpeta.parent / "comprimidos").mkdir(exist_ok=True)
    if generar_grafos:
        (carpeta.parent / "grafos").mkdir(exist_ok=True)

    # Para iterar los 3 idiomas
    idiomas = {
        "espanol": "Español",
        "frances": "Francés",
        "ingles": "Inglés",
    }
    # Para las estadisticas
    filas = []

    for prefijo, idioma in idiomas.items():
        for numero in range(1, 11):
            ruta_txt = carpeta / f"{prefijo}_{numero}.txt"
            ruta_huff = carpeta.parent / "comprimidos" / f"{ruta_txt.stem}.huff"
            ruta_grafo = carpeta.parent / "grafos" / f"{ruta_txt.stem}.svg"

            datos = ruta_txt.read_bytes()
            tamano_original = len(datos)

            texto = datos.decode("utf-8")
            conteo = Counter(texto)
            simbolos = list(conteo.keys())
            frecuencias = list(conteo.values())

            codes_dic, root = huffmanCodes(simbolos, frecuencias)

            if generar_grafos:
                visualizar_arbol(root, ruta_grafo)

            _, _, bits_por_simbolo = tabla_huffman(simbolos, frecuencias, codes_dic)

            codificar(ruta_huff, codes_dic, texto, root)
            tamano_comprimido = ruta_huff.stat().st_size

            filas.append({
                "Archivo": ruta_txt.name,
                "Idioma": idioma,
                "Tamaño original (bytes)": tamano_original,
                "Tamaño comprimido (bytes)": tamano_comprimido,
                "Compression factor": tamano_comprimido / tamano_original,
                "Bits promedio por símbolo": bits_por_simbolo,
            })

    por_archivo = pd.DataFrame(filas)
    por_idioma = por_archivo.groupby("Idioma", as_index=False)[
        ["Compression factor", "Bits promedio por símbolo"]
    ].mean()

    por_archivo.to_csv( carpeta.parent / "estadisticas_por_archivo.csv", index=False, encoding="utf-8-sig" )
    por_idioma.to_csv( carpeta.parent / "estadisticas_por_idioma.csv", index=False, encoding="utf-8-sig" )

    if comparar_decodificacion:
        tiempos = comparar_tiempos_decodificacion()
        tiempos.to_csv( carpeta.parent / "tiempos_decodificacion.csv", index=False,encoding="utf-8-sig")
        
