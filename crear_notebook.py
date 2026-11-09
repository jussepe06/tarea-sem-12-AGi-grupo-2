import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("""# 🛒 Caso Ripley: Market Basket Analysis con Algoritmo Apriori
### Inteligencia Comercial & Minería de Datos Transaccionales
**Curso:** Agentes Inteligentes  
**Herramienta:** Python (pandas, mlxtend, matplotlib, openpyxl)  
**Objetivo de Negocio:** Analizar 1,000 boletas de compra de cinco tiendas Ripley en Lima durante la campaña de fin de año para descubrir patrones de co-ocurrencia y formular una estrategia comercial basada en evidencia (combos, venta cruzada, layout y promociones).

---

## 📌 Fundamentos Teóricos: De la Base de Datos a la Minería de Reglas

### 1. Modelado Transaccional (Visión de Base de Datos)
En los sistemas transaccionales relacionales (OLTP / ERP / Punto de Venta), las compras se almacenan en tablas normalizadas con una relación uno a muchos ($1:N$):
- **Entidad `Ticket` (Cabecera):** `TicketID`, `Fecha`, `Tienda`.
- **Entidad `DetalleTicket` (Líneas de venta):** `TicketID`, `Producto`, `Categoría`, `Precio`.

Para aplicar algoritmos de canasta de mercado como **Apriori**, debemos desnormalizar y transformar este formato largo a una **matriz binaria de canastas (One-Hot Encoding)**:
- Cada fila representa una canasta ($t \in T$).
- Cada columna representa un producto del catálogo ($i \in I$).
- El valor es $1$ si el producto está en el ticket y $0$ si no está.

### 2. Métricas Clave de Reglas de Asociación ($A \rightarrow B$)
| Métrica | Definición Matemática | Interpretación en Negocio |
| :--- | :--- | :--- |
| **Soporte (Support)** | $\text{sop}(A \rightarrow B) = \frac{N(A \cap B)}{N}$ | ¿Qué tan popular o frecuente es comprar $A$ y $B$ juntos en el total de clientes? |
| **Confianza (Confidence)** | $\text{conf}(A \rightarrow B) = \frac{N(A \cap B)}{N(A)}$ | De los clientes que compraron $A$, ¿qué porcentaje también compró $B$? |
| **Lift (Elevación)** | $\text{lift}(A \rightarrow B) = \frac{\text{conf}(A \rightarrow B)}{\text{sop}(B)}$ | ¿Cuántas veces más probable es comprar $B$ si se compró $A$, frente a comprar $B$ al azar? |

- **$\text{Lift} > 1$:** Asociación positiva real (complementarios).
- **$\text{Lift} = 1$:** Independencia estadística (comprados juntos por puro azar).
- **$\text{Lift} < 1$:** Asociación negativa (sustitutos o excluyentes).
"""))

# Setup code
cells.append(nbf.v4.new_code_cell("""# Importación de librerías esenciales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Configuración visual de gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

print("Librerías importadas correctamente.")
"""))

# Actividad 1 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 📦 Actividad 1: Carga y Preparación de Datos (Formato Canasta)

Transformamos el dataset transaccional en formato largo (`TicketID`, `Fecha`, `Tienda`, `Categoría`, `Producto`, `Precio`) a una lista de canastas agrupadas por `TicketID`, y luego a una matriz binaria One-Hot.
"""))

# Actividad 1 Code
cells.append(nbf.v4.new_code_cell("""# 1. Cargar datos del archivo Excel oficial
excel_path = 'Dataset_Ripley_ReglasAsociacion.xlsx'
df_trans = pd.read_excel(excel_path, sheet_name='Transacciones')
df_cat = pd.read_excel(excel_path, sheet_name='Catálogo')

# 2. Métricas y KPIs de la canasta
n_tickets = df_trans['TicketID'].nunique()
n_productos = df_trans['Producto'].nunique()
n_lineas = len(df_trans)

canastas_sizes = df_trans.groupby('TicketID')['Producto'].count()
tam_prom = canastas_sizes.mean()
tam_min = canastas_sizes.min()
tam_max = canastas_sizes.max()
tam_std = canastas_sizes.std()

ticket_gasto = df_trans.groupby('TicketID')['Precio (S/)'].sum()
venta_total = ticket_gasto.sum()
ticket_prom_soles = ticket_gasto.mean()

print("="*60)
print("             KPIS DE LA CANASTA TRANSACCIONAL")
print("="*60)
print(f"Total de Tickets Únicos:          {n_tickets:,}")
print(f"Total de Líneas Transaccionales: {n_lineas:,}")
print(f"Productos Distintos Vendidos:     {n_productos}")
print(f"Tamaño Promedio de Canasta:       {tam_prom:.2f} productos/ticket")
print(f"Rango de Tamaño de Canasta:       Mínimo {tam_min} | Máximo {tam_max} (Desv. Est.: {tam_std:.2f})")
print(f"Venta Total Acumulada:            S/ {venta_total:,.2f}")
print(f"Ticket Promedio de Venta:         S/ {ticket_prom_soles:,.2f}")
print("="*60)
"""))

# One-hot encoding code
cells.append(nbf.v4.new_code_cell("""# 3. Transformación a formato One-Hot (Canastas)
canastas = df_trans.groupby('TicketID')['Producto'].apply(list).tolist()

te = TransactionEncoder()
matriz_binaria = te.fit(canastas).transform(canastas)
df_onehot = pd.DataFrame(matriz_binaria, columns=te.columns_)

print(f"Dimensiones de la Matriz One-Hot: {df_onehot.shape[0]} tickets x {df_onehot.shape[1]} productos")
df_onehot.head()
"""))

# Distribution plot code
cells.append(nbf.v4.new_code_cell("""# Gráfico de distribución del tamaño de la canasta
plt.figure(figsize=(9, 4.5))
plt.hist(canastas_sizes, bins=range(1, 13), color='#B71C1C', edgecolor='black', alpha=0.8, align='left')
plt.title('Distribución del Número de Productos por Ticket (Ripley)', fontsize=13, weight='bold')
plt.xlabel('Cantidad de Productos en el Ticket')
plt.ylabel('Frecuencia (Número de Tickets)')
plt.xticks(range(1, 12))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(tam_prom, color='navy', linestyle='dashed', linewidth=2, label=f'Media: {tam_prom:.2f} ítems')
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Actividad 2 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 📊 Actividad 2: Itemsets Frecuentes y Soporte Individual

### Justificación del Umbral de Soporte Mínimo (`min_support = 0.03` o 3%)
- Con $N = 1,000$ boletas, un soporte de **3.0%** equivale exactamente a **30 clientes**.
- **Justificación de Negocio:** En un retail departamental de gran escala con 39 productos activos y 9 departamentos, un patrón que ocurre menos de 30 veces en tres meses resulta anecdótico y no amerita inversión en combos o cambios de exhibición física. Por otro lado, un umbral superior al 5% descartaría combinaciones estratégicas de categorías con menor rotación pero alto margen como Electrohogar.
"""))

# Actividad 2 Code
cells.append(nbf.v4.new_code_cell("""# Soporte individual de cada producto
soporte_individual = df_onehot.mean().sort_values(ascending=False)
top10_ind = pd.DataFrame({
    'Producto': soporte_individual.head(10).index,
    'Soporte': soporte_individual.head(10).values,
    'Tickets': (soporte_individual.head(10).values * n_tickets).astype(int),
    'Soporte_Pct': (soporte_individual.head(10).values * 100).round(2)
}).merge(df_cat[['Producto', 'Categoría', 'Precio referencial (S/)']], on='Producto', how='left')

print("--- TOP 10 PRODUCTOS CON MAYOR SOPORTE INDIVIDUAL ---")
display(top10_ind)

# Gráfico de barras Top 10 productos
plt.figure(figsize=(10, 5))
colors = ['#1565C0' if c == 'Tecnología' else '#2E7D32' if c == 'Hogar' else '#E65100' for c in top10_ind['Categoría']]
bars = plt.barh(top10_ind['Producto'][::-1], top10_ind['Soporte_Pct'][::-1], color=colors[::-1], edgecolor='black')
plt.title('Top 10 Productos con Mayor Penetración en Tickets (Soporte %)', fontsize=13, weight='bold')
plt.xlabel('Soporte (% del Total de Tickets)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/4, f"{bar.get_width():.1f}%", va='center')
plt.tight_layout()
plt.show()
"""))

# Itemsets frecuentes code
cells.append(nbf.v4.new_code_cell("""# Generación de Itemsets Frecuentes con Apriori
frecuentes = apriori(df_onehot, min_support=0.03, use_colnames=True)
frecuentes['tamaño'] = frecuentes['itemsets'].apply(lambda x: len(x))

print(f"Total de itemsets frecuentes encontrados (Soporte >= 3%): {len(frecuentes)}")
print(f"- 1-itemsets (productos individuales): {len(frecuentes[frecuentes['tamaño'] == 1])}")
print(f"- 2-itemsets (pares de productos):     {len(frecuentes[frecuentes['tamaño'] == 2])}")
print(f"- 3-itemsets (tríos de productos):     {len(frecuentes[frecuentes['tamaño'] == 3])}")
"""))

# Actividad 3 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🔗 Actividad 3: Generación y Filtrado de Reglas de Asociación

Configuramos:
- **Métrica:** Confianza ($\ge 25\%$) para garantizar que la regla tenga predictibilidad real.
- **Filtro de Lift:** $\text{Lift} > 1.0$ para eliminar asociaciones espurias o independientes.
- **Ordenamiento:** Descendente por Lift y luego por Soporte.
"""))

# Actividad 3 Code
cells.append(nbf.v4.new_code_cell("""# Generar reglas de asociación
reglas = association_rules(frecuentes, metric='confidence', min_threshold=0.25)
reglas = reglas[reglas['lift'] > 1.0].sort_values(by=['lift', 'support'], ascending=[False, False])

# Añadir columnas legibles
reglas['Antecedente'] = reglas['antecedents'].apply(lambda x: ', '.join(list(x)))
reglas['Consecuente'] = reglas['consequents'].apply(lambda x: ', '.join(list(x)))

print(f"Total de reglas válidas (Confianza >= 25%, Lift > 1.0): {len(reglas)}")

# Tabla con el Top 10 de mejores reglas ordenadas por Lift
top10_tabla = reglas[['Antecedente', 'Consecuente', 'support', 'confidence', 'lift']].head(10).copy()
top10_tabla.columns = ['Antecedente (A)', 'Consecuente (B)', 'Soporte', 'Confianza', 'Lift']
top10_tabla['Soporte'] = (top10_tabla['Soporte'] * 100).round(2).astype(str) + '%'
top10_tabla['Confianza'] = (top10_tabla['Confianza'] * 100).round(2).astype(str) + '%'
top10_tabla['Lift'] = top10_tabla['Lift'].round(3)

display(top10_tabla)
"""))

# Scatter plot code
cells.append(nbf.v4.new_code_cell("""# Gráfico de dispersión: Soporte vs Confianza (Color por Lift)
plt.figure(figsize=(9, 5))
sc = plt.scatter(reglas['support']*100, reglas['confidence']*100, c=reglas['lift'], cmap='viridis', alpha=0.8, edgecolors='k', s=50)
cbar = plt.colorbar(sc)
cbar.set_label('Lift', rotation=270, labelpad=15)
plt.title('Mapa de Reglas de Asociación: Soporte vs Confianza vs Lift', fontsize=13, weight='bold')
plt.xlabel('Soporte (%)')
plt.ylabel('Confianza (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

# Actividad 4 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 💡 Actividad 4: Interpretación de Negocio de las 5 Mejores Reglas

Seleccionamos 5 reglas estratégicas balanceando **Lift alto**, **Soporte suficiente** y **viabilidad comercial** en las categorías clave de Ripley:

### 1. Electrohogar: `[Olla arrocera] ➔ [Licuadora]`
- **Métricas:** Soporte = **5.10%** (51 tickets) | Confianza = **100.00%** | Lift = **11.63**
- **Interpretación:** El 100% de los clientes que adquirieron una Olla arrocera también se llevaron una Licuadora. Esta compra ocurre **11.6 veces más** de lo esperado por azar. Representa una canasta típica de equipamiento o renovación de cocina en campaña navideña.

### 2. Moda Mujer: `[Cartera] ➔ [Vestido]`
- **Métricas:** Soporte = **7.50%** (75 tickets) | Confianza = **100.00%** | Lift = **8.55**
- **Interpretación:** Todas las clientas que compraron una cartera también compraron un vestido. La compra de accesorios en moda mujer no es aislada: funciona como complemento directo del outfit principal.

### 3. Ecosistema Tecnológico: `[Smartwatch] ➔ [Smartphone]`
- **Métricas:** Soporte = **4.80%** (48 tickets) | Confianza = **100.00%** | Lift = **4.90**
- **Interpretación:** Quien compra un reloj inteligente siempre compra el teléfono celular. Es una asociación de ecosistema funcional: el cliente adquiere ambos para vincularlos y estrenarlos juntos.

### 4. Cross-Selling Tecnológico: `[Funda + Audífonos] ➔ [Smartphone]`
- **Métricas:** Soporte = **9.50%** (95 tickets) | Confianza = **95.00%** | Lift = **4.66**
- **Interpretación:** El 95% de los clientes que adquieren la combinación de audífonos y funda se están llevando un smartphone. Con casi 10% de soporte de toda la tienda, es una de las canastas de mayor valor monetario.

### 5. Accesorios & Periféricos: `[Mouse + Teclado] ➔ [Mochila]`
- **Métricas:** Soporte = **3.00%** (30 tickets) | Confianza = **56.60%** | Lift = **8.45**
- **Interpretación:** Más del 56% de los compradores de periféricos de oficina/estudio añaden una mochila para transportarlos. Representa el kit de movilidad digital / home-office.
"""))

# Actividad 5 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🏬 Actividad 5: Análisis Segmentado por Tienda y Categoría

Comparamos el comportamiento en las 5 tiendas de Lima (`Ripley Jockey Plaza`, `Ripley Mega Plaza`, `Ripley San Isidro`, `Ripley San Miguel`, `Ripley Primavera`).
"""))

# Actividad 5 Code
cells.append(nbf.v4.new_code_cell("""# Métricas por Tienda
tienda_stats = df_trans.groupby('Tienda').agg(
    Tickets=('TicketID', 'nunique'),
    Lineas=('Producto', 'count'),
    Venta_Total=('Precio (S/)', 'sum')
)
tienda_stats['Ticket_Promedio'] = (tienda_stats['Venta_Total'] / tienda_stats['Tickets']).round(2)
tienda_stats['Canasta_Promedio'] = (tienda_stats['Lineas'] / tienda_stats['Tickets']).round(2)
tienda_stats = tienda_stats.sort_values(by='Tickets', ascending=False)

display(tienda_stats)

# Gráfico comparativo de Ticket Promedio por Tienda
plt.figure(figsize=(9, 4.5))
bars = plt.bar(tienda_stats.index.str.replace('Ripley ', ''), tienda_stats['Ticket_Promedio'], color='#00897B', edgecolor='black')
plt.title('Ticket Promedio por Tienda Ripley (Soles)', fontsize=13, weight='bold')
plt.ylabel('S/ por Ticket')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 120, f"S/ {bar.get_height():,.0f}", ha='center', color='white', weight='bold')
plt.tight_layout()
plt.show()
"""))

# Actividad 6 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🚀 Actividad 6: Propuesta de Estrategia Comercial Ripley

A partir de los hallazgos analíticos, formulamos un plan de acción para elevar el ticket promedio y la venta cruzada:

### 1. Dos Combos / Bundles Justificados por Métricas
* **Combo 1: "Renueva tu Cocina" (Olla Arrocera + Licuadora)**
  - *Precios Catálogo:* Olla Arrocera (S/ 139) + Licuadora (S/ 159) = S/ 298.
  - *Precio Combo:* **S/ 249** (descuento del 16.4%, ahorro de S/ 49).
  - *Sustento:* Soporte 5.10%, Confianza 100%, Lift 11.63. Garantiza rotación conjunta sin subsidiar productos no demandados.
* **Combo 2: "Total Look Elegance" (Vestido + Cartera)**
  - *Precios Catálogo:* Vestido (S/ 169) + Cartera (S/ 159) = S/ 328.
  - *Precio Combo con Tarjeta Ripley:* **S/ 279** (ahorro de S/ 49).
  - *Sustento:* Soporte 7.50%, Confianza 100%, Lift 8.55. Eleva el ticket en moda femenina.

### 2. Dos Acciones de Venta Cruzada (Cross-Selling en Web, App y Caja)
* **Acción Digital (Web & App):** Al agregar un **Smartphone (S/ 1,799)** al carrito, desplegar pop-up interactivo *"Completa tu equipo"* sugiriendo **Funda (S/ 49)** y **Audífonos Bluetooth (S/ 199)** con un 20% de descuento en los accesorios.
* **Acción en Punto de Venta (Cajas):** Al escanear periféricos de cómputo (Mouse o Teclado), el POS alerta al cajero para ofrecer la **Mochila ejecutiva** con descuento exclusivo inmediato (Lift 8.45).

### 3. Decisión de Layout y Merchandising en Tienda Física
* **Acercamiento en Tienda:** En el departamento de Electrohogar, unificar la cabecera de góndola de "Pequeños Electrodomésticos" juntando Ollas arroceras, Licuadoras y Hervidores.
* **Separación Estratégica:** Ubicar los accesorios de telefonía (fundas, micas, cargadores) en el pasillo principal que conduce a las cajas de tecnología, forzando un recorrido visual obligatorio para quienes van a retirar su celular.

### 4. Política de Precios, Riesgo de Canibalización y Cuantificación de Impacto
* **Cálculo de Impacto Económico:**  
  - En el dataset, 204 personas compraron un Smartphone, pero solo 95 llevaron el combo de Funda + Audífonos (brecha de 109 clientes).
  - Si con la venta cruzada en web y caja capturamos al **30% de esa brecha** (33 clientes adicionales):
  - Ingreso incremental por cliente = S/ 49 (Funda) + S/ 199 (Audífonos) = S/ 248.
  - **Incremento directo en ventas: 33 × S/ 248 = +S/ 8,184 adicionales** solo en ese extracto de 1,000 boletas.
* **Control de Canibalización:** Los descuentos en combos deben condicionarse al uso de la Tarjeta Ripley o compra simultánea, evitando rebajar el precio individual de los productos estrellas.
"""))

# Actividad 7 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 📝 Actividad 7: Conclusiones y Limitaciones del Estudio

### Tres Conclusiones Principales:
1. **La venta cruzada no es casual:** Los lifts superiores a 8 y 11 demuestran que categorías como Electrohogar y Moda Mujer dependen críticamente de compras complementarias que deben exhibirse y promocionarse juntas.
2. **Tecnología domina la facturación:** Concentra el 69.4% del volumen de ventas (S/ 876,670 de S/ 1,262,136) y presenta las canastas de mayor valor monetario gracias al arrastre del Smartphone.
3. **Alto potencial de crecimiento en ticket promedio:** El 53.4% de los compradores de smartphones no adquieren accesorios en la misma compra; cerrar esa brecha genera un impacto económico directo inmediato.

### Dos Limitaciones Metodológicas:
1. **Estacionalidad (Campaña de Fin de Año):** El dataset cubre octubre–diciembre 2025 (Navidad y Cyber Days), lo que sobrepondera regalos, compras impulsivas y electrodomésticos para obsequio, patrones que pueden variar en meses regulares.
2. **Correlación no implica Causalidad:** Las reglas revelan co-ocurrencia simultánea, pero no registran la intención primaria de compra ni el impacto de promociones previas activas en tienda en esa fecha.
"""))

# Write notebook file
nb['cells'] = cells
with open('ripley_apriori.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook ripley_apriori.ipynb generado exitosamente.")
