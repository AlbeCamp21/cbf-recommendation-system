"""
Análisis Ético del Dataset de Ofertas de Empleo
Sistema de Recomendación CBF - Fase 5

Este script analiza el dataset para identificar posibles sesgos
y limitaciones que deben documentarse en el informe final.
Incluye gráficos con matplotlib con colores
"""

import pickle
from collections import Counter
import matplotlib.pyplot as plt


def cargar_datos():
    """Carga todas las ofertas desde los archivos procesados."""
    categorias = ['asistente', 'contador', 'desarrollador',
                  'ingeniero', 'marketing', 'programador', 'vendedor']

    todas_ofertas = []
    stats_categorias = {}

    for cat in categorias:
        ruta = f'dataset/processed/vectors_{cat}.pkl'
        try:
            with open(ruta, 'rb') as f:
                data = pickle.load(f)
                ofertas = data['metadata']
                stats_categorias[cat] = len(ofertas)
                for item in ofertas:
                    item['categoria'] = cat
                todas_ofertas.extend(ofertas)
        except FileNotFoundError:
            print(f"Advertencia: No se encontró {ruta}")

    return todas_ofertas, stats_categorias


# -------------------------------------------------------------------------
# 1. SESGO GEOGRÁFICO + GRÁFICO
# -------------------------------------------------------------------------
def analizar_sesgo_geografico(ofertas):
    """Analiza la distribución geográfica de las ofertas y genera gráfico."""
    print("\n" + "="*60)
    print("1. ANÁLISIS DE SESGO GEOGRÁFICO")
    print("="*60)

    ciudades = {
        'lima': 0,
        'arequipa': 0,
        'trujillo': 0,
        'cusco': 0,
        'piura': 0,
        'chiclayo': 0,
        'callao': 0,
        'huancayo': 0,
        'ica': 0,
        'tacna': 0
    }

    ofertas_con_ubicacion = 0

    for oferta in ofertas:
        texto = oferta.get('description', '').lower()
        encontro_ciudad = False

        for ciudad in ciudades:
            if ciudad in texto:
                ciudades[ciudad] += 1
                encontro_ciudad = True

        if encontro_ciudad:
            ofertas_con_ubicacion += 1

    print(f"\nOfertas que mencionan ubicación: {ofertas_con_ubicacion}/{len(ofertas)} "
          f"({ofertas_con_ubicacion/len(ofertas)*100:.1f}%)")

    print("\nDistribución por ciudad:")
    print("-" * 40)

    for ciudad, count in sorted(ciudades.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = count / len(ofertas) * 100
            barra = "█" * int(pct * 2)
            print(f"  {ciudad.title():12} {count:5} ({pct:5.1f}%) {barra}")

    # Gráfico
    labels = [c.title() for c, v in ciudades.items() if v > 0]
    valores = [v for v in ciudades.values() if v > 0]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, valores, color="skyblue")   # COLOR AGREGADO
    plt.title("Distribución Geográfica de Ofertas")
    plt.xlabel("Ciudad")
    plt.ylabel("Cantidad de menciones")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# 2. SESGO DE GÉNERO + GRÁFICO
# -------------------------------------------------------------------------
def analizar_sesgo_genero(ofertas):
    """Analiza el uso de términos de género y genera gráfico."""
    print("\n" + "="*60)
    print("2. ANÁLISIS DE SESGO DE GÉNERO")
    print("="*60)

    terminos_masculinos = [
        'el candidato', 'los candidatos', 'el postulante', 'los postulantes',
        'el profesional', 'interesados', 'egresado', 'graduado',
        'ingeniero', 'contador', 'vendedor', 'programador', 'desarrollador',
        'asistente administrativo', 'ejecutivo'
    ]

    terminos_femeninos = [
        'la candidata', 'las candidatas', 'la postulante', 'las postulantes',
        'la profesional', 'interesadas', 'egresada', 'graduada',
        'ingeniera', 'contadora', 'vendedora', 'programadora', 'desarrolladora',
        'asistente administrativa', 'ejecutiva'
    ]

    terminos_neutros = [
        'candidato/a', 'candidatos/as', 'postulante', 'profesional',
        'persona', 'quien', 'egresado/a', 'profesionales',
        'interesados/as', 'el/la candidato/a'
    ]

    conteo_masc = 0
    conteo_fem = 0
    conteo_neutro = 0

    for oferta in ofertas:
        texto = oferta.get('description', '').lower()

        if any(t in texto for t in terminos_masculinos):
            conteo_masc += 1
        if any(t in texto for t in terminos_femeninos):
            conteo_fem += 1
        if any(t in texto for t in terminos_neutros):
            conteo_neutro += 1

    print(f"\nOfertas con términos masculinos: {conteo_masc}")
    print(f"Ofertas con términos femeninos:  {conteo_fem}")
    print(f"Ofertas con términos neutros:    {conteo_neutro}")

    # Gráfico
    etiquetas = ["Masculino", "Femenino", "Neutro"]
    valores = [conteo_masc, conteo_fem, conteo_neutro]

    plt.figure(figsize=(7, 5))
    plt.bar(etiquetas, valores,
            color=["royalblue", "lightcoral", "gold"])   # COLORES AGREGADOS
    plt.title("Sesgo de Género en Ofertas")
    plt.ylabel("Cantidad de Ofertas")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# 3. SESGO DE EDAD + GRÁFICO
# -------------------------------------------------------------------------
def analizar_sesgo_edad(ofertas):
    """Analiza restricciones de edad y genera gráfico."""
    print("\n" + "="*60)
    print("3. ANÁLISIS DE SESGO POR EDAD")
    print("="*60)

    patrones_edad = [
        'menor de 25', 'menor de 30', 'menor de 35', 'menor de 40',
        'mayor de 25', 'mayor de 30', 'mayor de 35',
        'entre 18 y 25', 'entre 25 y 35', 'entre 18 y 30',
        '18 a 25', '25 a 35', '18 a 30', '20 a 30', '25 a 40',
        'hasta 25 años', 'hasta 30 años', 'hasta 35 años',
        'máximo 25', 'máximo 30', 'máximo 35',
        'edad máxima', 'rango de edad', 'años de edad'
    ]

    ofertas_con_restriccion = 0
    detalles = Counter()

    for oferta in ofertas:
        texto = oferta.get('description', '').lower()

        for patron in patrones_edad:
            if patron in texto:
                ofertas_con_restriccion += 1
                detalles[patron] += 1
                break

    pct = ofertas_con_restriccion / len(ofertas) * 100
    print(f"\nOfertas con restricciones de edad: {ofertas_con_restriccion} ({pct:.1f}%)")

    # Gráfico
    if detalles:
        labels = list(detalles.keys())[:10]
        valores = list(detalles.values())[:10]

        plt.figure(figsize=(12, 5))
        plt.bar(labels, valores, color="mediumseagreen")  # COLOR AGREGADO
        plt.title("Restricciones de Edad Más Frecuentes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# -------------------------------------------------------------------------
# 4. REQUISITOS EDUCATIVOS + GRÁFICO
# -------------------------------------------------------------------------
def analizar_requisitos_educativos(ofertas):
    """Analiza requisitos educativos y genera gráfico."""
    print("\n" + "="*60)
    print("4. ANÁLISIS DE REQUISITOS EDUCATIVOS")
    print("="*60)

    niveles = {
        'secundaria': ['secundaria completa', 'secundaria'],
        'tecnico': ['técnico', 'tecnico', 'instituto'],
        'universitario': ['universitario', 'universidad', 'bachiller', 'licenciado', 'licenciatura'],
        'postgrado': ['maestría', 'maestria', 'postgrado', 'posgrado', 'mba', 'doctorado']
    }

    conteo = {nivel: 0 for nivel in niveles}

    for oferta in ofertas:
        texto = oferta.get('description', '').lower()

        for nivel, palabras in niveles.items():
            if any(p in texto for p in palabras):
                conteo[nivel] += 1

    # Imprimir en terminal
    for nivel, count in conteo.items():
        print(f"{nivel.title():15}: {count}")

    # Gráfico
    labels = list(conteo.keys())
    valores = list(conteo.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, valores, color="mediumpurple")  # COLOR AGREGADO
    plt.title("Requisitos Educativos en Ofertas")
    plt.ylabel("Cantidad de Ofertas")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# 5. DISTRIBUCIÓN POR CATEGORÍAS + GRÁFICO
# -------------------------------------------------------------------------
def analizar_distribucion_categorias(stats_categorias, total):
    """Analiza balance de categorías y genera gráfico."""
    print("\n" + "="*60)
    print("5. DISTRIBUCIÓN POR CATEGORÍA")
    print("="*60)

    for cat, count in stats_categorias.items():
        print(f"{cat.title():12}: {count}")

    # Gráfico
    labels = list(stats_categorias.keys())
    valores = list(stats_categorias.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, valores, color="orange")  # COLOR AGREGADO
    plt.title("Distribución de Ofertas por Categoría")
    plt.xticks(rotation=45)
    plt.ylabel("Cantidad de Ofertas")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# RESUMEN EJECUTIVO (sin cambios)
# -------------------------------------------------------------------------
def generar_resumen(ofertas, stats_categorias):
    print("\n" + "="*60)
    print("RESUMEN EJECUTIVO")
    print("="*60)

    print(f"""
Total ofertas procesadas: {len(ofertas)}
Categorías: {len(stats_categorias)}
Fuentes: Computrabajo y Bumeran

Hallazgos:
 - Ver gráficos y análisis detallado arriba
    """)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    print("="*60)
    print("ANÁLISIS ÉTICO DEL DATASET DE OFERTAS DE EMPLEO")
    print("="*60)

    ofertas, stats = cargar_datos()
    print(f"Ofertas cargadas: {len(ofertas)}")

    analizar_distribucion_categorias(stats, len(ofertas))
    analizar_sesgo_geografico(ofertas)
    analizar_sesgo_genero(ofertas)
    analizar_sesgo_edad(ofertas)
    analizar_requisitos_educativos(ofertas)

    generar_resumen(ofertas, stats)


if __name__ == "__main__":
    main()
