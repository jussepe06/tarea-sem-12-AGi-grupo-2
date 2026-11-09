# Tarea Semana 12 - Reglas de Asociación (Algoritmo Apriori)

**Curso:** Agentes Inteligentes  
**Grupo:** 2  
**Tema:** Análisis de canasta de mercado (Market Basket Analysis) - Caso Ripley  

## Resumen del trabajo
En este repositorio se desarrolla el análisis de los 1,000 tickets de compra de Ripley (campaña oct–dic 2025) aplicando el algoritmo Apriori en Python con `pandas` y `mlxtend`. El objetivo es identificar qué productos se compran juntos mediante las métricas de soporte, confianza y lift, y proponer estrategias de venta cruzada y promociones para subir el ticket promedio.

## Archivos principales
- `Dataset_Ripley_ReglasAsociacion.xlsx`: Base de datos transaccional original (4,434 registros en 1,000 tickets).
- `ripley_apriori.ipynb`: Notebook de Jupyter con el desarrollo paso a paso de las 7 actividades de la guía.
- `ejecutar_analisis.py`: Script en Python que ejecuta todo el cálculo y exporta las tablas de resultados a Excel.
- `Resultados_Ripley_Apriori_Grupo2.xlsx`: Tablas generadas con los KPIs, el top 10 de productos y las mejores reglas.

## Requisitos para ejecutar
```bash
pip install pandas openpyxl mlxtend matplotlib
```
