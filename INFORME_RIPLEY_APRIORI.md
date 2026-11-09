# INFORME DE INTELIGENCIA COMERCIAL: MARKET BASKET ANALYSIS CON ALGORITMO APRIORI EN RETAIL (CASO RIPLEY)

**Facultad de Ingeniería e Inteligencia Artificial**  
**Curso:** Agentes Inteligentes  
**Semana:** 12 – Práctica Calificada  
**Caso de Negocio:** Ripley Perú – Estrategia de Venta Cruzada y Optimización del Ticket Promedio  
**Herramienta Utilizada:** Python 3.13 (`pandas`, `mlxtend`, `openpyxl`, `matplotlib`)  
**Fecha:** Noviembre 2026  

---

## RESUMEN EJECUTIVO
El área de Inteligencia Comercial de Ripley identificó un estancamiento en el ticket promedio de compra durante las campañas comerciales. El presente estudio aplica técnicas de **Minería de Datos Transaccionales** y **Market Basket Analysis (MBA)** utilizando el **Algoritmo Apriori** sobre un extracto de **1,000 tickets de compra (4,434 líneas de venta)** provenientes de cinco tiendas Ripley en Lima durante la campaña de fin de año (octubre–diciembre 2025).

A través de la transformación de los datos a una matriz binaria *one-hot* y el modelado de reglas de asociación con soporte mínimo de 3.0% y confianza mínima del 25%, se identificaron **364 reglas válidas con Lift > 1.0**. Los hallazgos evidencian patrones de compra altamente complementarios en categorías estratégicas:
1. **Electrohogar:** Sinergia total en equipamiento de cocina (`[Olla arrocera] ➔ [Licuadora]` con Confianza de 100% y Lift de 11.63).
2. **Moda Femenina:** Venta cruzada indisoluble entre accesorios y prendas (`[Cartera] ➔ [Vestido]` con Confianza de 100% y Lift de 8.55).
3. **Ecosistema Tecnológico:** Venta cruzada de alta facturación (`[Funda + Audífonos] ➔ [Smartphone]` con Soporte de 9.50% y Confianza de 95.0%).

Con base en estos descubrimientos, se diseñó una **Estrategia Comercial Cuantificada** basada en combos promocionales, venta cruzada en checkout (web/caja) y optimización de layout físico, proyectando un incremento inmediato de **+S/ 8,184.00** en ingresos incrementales por cada 1,000 boletas capturando únicamente el 30% de la brecha no convertida en telefonía.

---

## 1. MARCO CONCEPTUAL Y DE BASES DE DATOS

### 1.1 De la Base de Datos Relacional a la Matriz de Canasta
En la arquitectura de bases de datos relacionales transaccionales (OLTP / ERP), la venta se almacena con integridad referencial mediante una relación $1:N$:
- **Tabla Cabecera (`Boleta`/`Ticket`):** `TicketID`, `Fecha`, `Tienda`.
- **Tabla Detalle (`DetalleVenta`):** `TicketID`, `ProductoID`, `Precio`.

Para procesar algoritmos de minería de reglas como **Apriori**, es necesario realizar un pivoteo y desnormalización de la tabla detalle agrupando por `TicketID`, convirtiendo la lista de productos adquiridos en una **matriz esparsa One-Hot Encoding**:
$$\mathbf{M} \in \{0, 1\}^{N \times M}$$
donde $N = 1,000$ transacciones y $M = 39$ productos distintos con ventas activas.

### 1.2 Métricas de Asociación: Soporte, Confianza y Lift
Sea $A$ el conjunto antecedente e $B$ el consecuente de una regla $A \rightarrow B$:
1. **Soporte (Support):**  
   $$\text{sop}(A \rightarrow B) = \frac{N(A \cap B)}{N}$$  
   Mide la penetración o frecuencia relativa de la co-ocurrencia dentro del universo total de tickets.
2. **Confianza (Confidence):**  
   $$\text{conf}(A \rightarrow B) = \frac{N(A \cap B)}{N(A)}$$  
   Mide la fiabilidad predictiva de la regla: de todos los clientes que compraron $A$, qué porcentaje también adquirió $B$.
3. **Lift (Fuerza de Elevación):**  
   $$\text{lift}(A \rightarrow B) = \frac{\text{conf}(A \rightarrow B)}{\text{sop}(B)} = \frac{N \cdot N(A \cap B)}{N(A) \cdot N(B)}$$  
   Compara la probabilidad conjunta observada frente a la probabilidad esperada bajo independencia estadística:
   - $\text{Lift} > 1$: Asociación real y complementaria.
   - $\text{Lift} = 1$: Independencia (ocurren juntos por mero azar).
   - $\text{Lift} < 1$: Sustitución o canibalización (el cliente compra uno u otro, pero no ambos).

---

## 2. DESARROLLO DE ACTIVIDADES

### Actividad 1 · Preparación de los Datos y KPIs de Canasta
Se procesaron las 4,434 filas de la hoja `Transacciones`. No se registraron valores nulos ni duplicados inconsistentes. La estructura de canastas arrojó los siguientes indicadores clave:

| Indicador Operativo | Valor Numérico | Interpretación de Negocio |
| :--- | :--- | :--- |
| **Tickets Únicos ($N$)** | **1,000 boletas** | Muestra representativa de 5 sucursales de Lima. |
| **Líneas de Venta** | **4,434 ítems** | Registros individuales de compra. |
| **Variedad de Productos Activos** | **39 productos** | De 43 productos del catálogo (4 sin ventas en el extracto). |
| **Tamaño Promedio de Canasta** | **4.43 productos** | Canasta multi-ítem saludable con desviación estándar de 1.98. |
| **Rango de Productos por Ticket** | **1 a 11 productos** | Mínimo 1 producto, máximo 11 productos por canasta. |
| **Facturación Total Acumulada** | **S/ 1,262,136.00** | Venta bruta de la muestra evaluada. |
| **Ticket Promedio de Venta** | **S/ 1,262.14** | Facturación media por cliente/boleta. |

---

### Actividad 2 · Itemsets Frecuentes y Soporte Individual

#### Justificación del Umbral de Soporte Mínimo (`min_support = 0.03` / 3%)
Para un universo de $N = 1,000$ transacciones, un umbral de **3.0%** exige que cualquier itemset o regla haya sido adquirido como mínimo por **30 clientes diferentes**.
- **Justificación Analítica:** Un soporte menor al 3% (por ejemplo 1% o 10 clientes) generaría sobreajuste hacia combinaciones anecdóticas que no ameritan inversión comercial. Un soporte superior al 5% descartaría por completo categorías de alto valor monetario pero menor rotación transaccional, como **Electrohogar** y **Moda Mujer**.

#### Top 10 Productos con Mayor Soporte Individual
El ranking individual refleja los productos ancla del retailer:

| # | Producto | Categoría | Soporte (%) | Boletas ($N(A)$) | Precio Ref. (S/) |
| :-: | :--- | :--- | :-: | :-: | :-: |
| 1 | **Polo** | Moda hombre | **29.60%** | 296 | S/ 59.00 |
| 2 | **Funda para celular** | Tecnología | **26.10%** | 261 | S/ 49.00 |
| 3 | **Medias deportivas** | Deportes | **22.60%** | 226 | S/ 29.00 |
| 4 | **Cargador portátil** | Tecnología | **21.00%** | 210 | S/ 119.00 |
| 5 | **Smartphone** | Tecnología | **20.40%** | 204 | S/ 1,799.00 |
| 6 | **Zapatillas deportivas** | Calzado | **18.50%** | 185 | S/ 269.00 |
| 7 | **Audífonos Bluetooth** | Tecnología | **17.80%** | 178 | S/ 199.00 |
| 8 | **Toallas** | Hogar | **17.00%** | 170 | S/ 59.00 |
| 9 | **Perfume** | Belleza | **17.00%** | 170 | S/ 199.00 |
| 10 | **Juego de sábanas** | Hogar | **16.40%** | 164 | S/ 149.00 |

*Interpretación:* Los productos de ticket bajo y alta reposición (**Polo**, **Funda para celular**, **Medias deportivas**) lideran en penetración. No obstante, el **Smartphone** logra un impresionante 20.4% de penetración, constituyendo el mayor motor de ingresos brutos de la tienda.

Con este umbral, el algoritmo Apriori descubrió **169 itemsets frecuentes**:
- **1-itemsets:** 38 productos individuales.
- **2-itemsets:** 85 combinaciones binarias.
- **3-itemsets:** 41 tríos de productos frecuentes.

---

### Actividad 3 · Generación y Filtrado de Reglas de Asociación
Fijando una **Confianza Mínima de 25%** y **Lift > 1.0**, el algoritmo generó **364 reglas de asociación**. A continuación se presentan las 10 mejores reglas ordenadas estrictamente por su valor de **Lift**:

| Regla | Antecedente (A) | Consecuente (B) | Soporte | Confianza | Lift |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **R1** | Olla arrocera | Licuadora | 5.10% | 100.00% | **11.628** |
| **R2** | Licuadora | Olla arrocera | 5.10% | 59.30% | **11.628** |
| **R3** | Hervidor eléctrico | Licuadora | 4.30% | 100.00% | **11.628** |
| **R4** | Licuadora | Hervidor eléctrico | 4.30% | 50.00% | **11.628** |
| **R5** | Cartera | Vestido | 7.50% | 100.00% | **8.547** |
| **R6** | Vestido | Cartera | 7.50% | 64.10% | **8.547** |
| **R7** | Blusa | Vestido | 5.40% | 100.00% | **8.547** |
| **R8** | Vestido | Blusa | 5.40% | 46.15% | **8.547** |
| **R9** | Pantalón de tela | Vestido | 3.80% | 100.00% | **8.547** |
| **R10** | Vestido | Pantalón de tela | 3.80% | 32.48% | **8.547** |

---

### Actividad 4 · Interpretación de Negocio de las 5 Reglas Más Valiosas
Para generar valor real en Ripley, no basta con seleccionar las reglas de mayor Lift numérico; se seleccionaron 5 reglas que combinan **significancia estadística, volumen monetario y viabilidad operativa**:

#### 1. Regla Electrohogar: `[Olla arrocera] ➔ [Licuadora]`
- **Métricas:** Soporte = **5.10%** (51 tickets) | Confianza = **100.00%** | Lift = **11.63**
- **Lectura Comercial:** La totalidad (100%) de los clientes que compraron una olla arrocera se llevaron también una licuadora. La co-ocurrencia es 11.6 veces mayor a la probabilidad aleatoria. Esto demuestra un comportamiento de compra enfocado en **equipamiento o remodelación de cocina**.

#### 2. Regla Moda Mujer: `[Cartera] ➔ [Vestido]`
- **Métricas:** Soporte = **7.50%** (75 tickets) | Confianza = **100.00%** | Lift = **8.55**
- **Lectura Comercial:** El 100% de las compradoras de carteras adquirieron simultáneamente un vestido. Representa la compra de **"Total Look"** para eventos de fin de año (fiestas, graduaciones, año nuevo). La cartera actúa como accesorio dependiente de la prenda principal.

#### 3. Regla Ecosistema Tech: `[Smartwatch] ➔ [Smartphone]`
- **Métricas:** Soporte = **4.80%** (48 tickets) | Confianza = **100.00%** | Lift = **4.90**
- **Lectura Comercial:** Cada cliente que adquiere un Smartwatch (reloj inteligente) adquiere también un Smartphone. En la campaña navideña, los clientes no compran el wearable de forma aislada, sino como parte de la renovación integral de sus dispositivos personales.

#### 4. Regla Canasta Tech de Alto Valor: `[Funda + Audífonos] ➔ [Smartphone]`
- **Métricas:** Soporte = **9.50%** (95 tickets) | Confianza = **95.00%** | Lift = **4.66**
- **Lectura Comercial:** El 95% de los clientes que juntan en su canasta audífonos Bluetooth y funda para celular están comprando un Smartphone. Es una regla de altísimo volumen (casi 1 de cada 10 tickets de Ripley) con un valor monetario combinado de **S/ 2,047.00**.

#### 5. Regla Movilidad Digital: `[Mouse + Teclado] ➔ [Mochila]`
- **Métricas:** Soporte = **3.00%** (30 tickets) | Confianza = **56.60%** | Lift = **8.45**
- **Lectura Comercial:** El 56.6% de quienes compran periféricos inalámbricos de cómputo agregan una mochila a su compra. Quienes invierten en periféricos portátiles buscan inmediatamente la solución física para transportarlos.

---

### Actividad 5 · Análisis Comparativo por Tienda y Categoría

#### Comparativa entre Tiendas
El análisis de las 1,000 transacciones distribuidas en las cinco sedes arroja perfiles diferenciados:

| Sucursal Ripley | Tickets | Líneas | Venta Total (S/) | Ticket Promedio | Canasta Media | Categoría Dominante |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ripley Jockey Plaza** | 304 | 1,313 | S/ 391,547.00 | **S/ 1,287.98** | 4.32 | Tecnología (69.2% venta) |
| **Ripley Mega Plaza** | 248 | 1,086 | S/ 317,014.00 | **S/ 1,278.28** | 4.38 | Tecnología (69.8% venta) |
| **Ripley Primavera** | 122 | 553 | S/ 165,357.00 | **S/ 1,355.39** | 4.53 | Tecnología (70.1% venta) |
| **Ripley San Isidro** | 162 | 756 | S/ 206,864.00 | **S/ 1,276.94** | **4.67** | Hogar y Tecnología |
| **Ripley San Miguel** | 164 | 726 | S/ 181,354.00 | **S/ 1,105.82** | 4.43 | Moda y Calzado |

#### Variación de Patrones por Tienda:
- **Jockey Plaza (Lima Este/Sur residencial alto):** Presenta una fuerte asociación en **Electrohogar y Tecnología**. Regla destacada: `[Microondas] ➔ [Licuadora]` con Confianza del 100% y Lift de 12.16. Los clientes exhiben mayor propensión al equipamiento integral de cocina.
- **Mega Plaza (Lima Norte):** Es la tienda con mayor cantidad de reglas de cross-selling de electrodomésticos compactos (559 reglas). Aparecen triples combinaciones intensivas como `[Licuadora + Hervidor] ➔ [Olla arrocera]` con Confianza de 83.3% y Lift de 17.22, evidenciando compras familiares masivas por fiestas navideñas.
- **San Isidro:** Exhibe la canasta más grande (4.67 ítems/ticket) y el mayor arrastre hacia la categoría **Hogar** (sábanas, edredones, toallas).

---

### Actividad 6 · Propuesta de Estrategia de Ventas Basada en Evidencia

#### 1. Dos Combos / Bundles de Productos
* **Combo 1: "Master Chef Home" (Olla Arrocera + Licuadora)**
  - *Justificación Numérica:* Soporte de 5.10%, Confianza del 100% y Lift de 11.63.
  - *Precios de Catálogo:* Olla Arrocera (S/ 139) + Licuadora (S/ 159) = S/ 298.00.
  - *Precio Combo Promocional:* **S/ 249.00** (ahorro del 16.4% o S/ 49.00).
  - *Impacto:* Eleva la conversión en el pasillo de electrodomésticos empujando la compra conjunta inmediata.
* **Combo 2: "Elegancia Total" (Vestido + Cartera)**
  - *Justificación Numérica:* Soporte de 7.50%, Confianza del 100% y Lift de 8.55.
  - *Precios de Catálogo:* Vestido (S/ 169) + Cartera (S/ 159) = S/ 328.00.
  - *Precio Combo Exclusivo Tarjeta Ripley:* **S/ 279.00** (ahorro de S/ 49.00).
  - *Impacto:* Captura la intención de compra para outfits de fiesta de fin de año incrementando la venta cruzada de accesorios de moda.

#### 2. Dos Acciones de Venta Cruzada (Web, App y Cajas)
* **Acción 1: Pop-up Inteligente en E-Commerce / App Ripley:**  
  Al momento en que un usuario añade un **Smartphone** al carrito digital, se dispara una ventana contextual: *"Protege y disfruta tu nuevo Smartphone"*, ofreciendo el bundle de **Funda de Silicona (S/ 49)** + **Audífonos Bluetooth (S/ 199)** con un 20% de descuento automático en los accesorios si se compran en ese instante.
* **Acción 2: Cross-Selling en Terminal de Punto de Venta (POS / Cajas Físicas):**  
  Programar el sistema de cajas para que, al escanear un mouse inalámbrico o teclado, la pantalla del cajero emita un script guiado ofreciendo la **Mochila Porta-Laptop ejecutiva** a S/ 89 (precio regular S/ 129), aprovechando el Lift de 8.45.

#### 3. Decisión de Layout y Merchandising en Tienda Física
* **Acercamiento Físico (Cross-Merchandising en Electrohogar):**  
  Crear una isla central o cabecera de góndola temática de "Pequeños Electrodomésticos de Cocina" donde convivan físicamente las Ollas arroceras, Licuadoras y Hervidores eléctricos, en lugar de mantenerlos en estanterías separadas por marca.
* **Separación Estratégica (Tráfico Dirigido en Telefonía):**  
  Ubicar el módulo de exhibición de Smartphones en el fondo de la zona tecnológica, y colocar las góndolas de fundas, cargadores portátiles y audífonos a lo largo del pasillo de acceso directo a la caja de cobro, forzando la visualización del accesorio complementario.

#### 4. Pricing, Riesgo de Canibalización y Cuantificación de Impacto Económico
* **Riesgo de Canibalización y Mitigación:**  
  El riesgo principal de un bundle es que clientes que ya planeaban pagar el precio completo (S/ 298 por olla y licuadora) reciban el descuento innecesariamente. Para mitigarlo:
  1. El descuento del combo debe ser exclusivo pagando con la **Tarjeta Ripley / Banco Ripley**, financiando parte del descuento a través del margen financiero y fidelización.
  2. Limitar la vigencia temporal de los bundles a ventanas promocionales de fin de semana ("Viernes a Domingo").
* **Cuantificación del Impacto Incremental:**
  - En las 1,000 boletas evaluadas, **204 clientes compraron un Smartphone**, pero únicamente **95 compraron el combo complementario de Funda + Audífonos**.
  - Existe una brecha no capturada de **109 clientes** que salieron de la tienda únicamente con el teléfono.
  - Si la estrategia de venta cruzada en web y caja logra convertir de forma conservadora al **30% de esa brecha** (33 clientes adicionales):
    $$\text{Ingreso adicional por cliente} = \text{S/ 49 (Funda)} + \text{S/ 199 (Audífonos)} = \text{S/ 248.00}$$
    $$\text{Impacto Financiero Directo} = 33 \times \text{S/ 248.00} = \mathbf{+\text{S/ 8,184.00}}$$
  - Este incremento representa un crecimiento inmediato del **+0.65% en la facturación total** de la cadena actuando sobre una sola regla.

---

### Actividad 7 · Conclusiones y Limitaciones

#### Tres Conclusiones Principales:
1. **La venta cruzada responde a necesidades funcionales completas:** Las métricas de confianza perfecta (100%) y lifts superiores a 8.0 en Electrohogar y Moda demuestran que los clientes de Ripley buscan soluciones integrales (cocina lista, look completo) y no artículos aislados.
2. **Tecnología es el motor de valor:** Aunque categorías como Moda Hombre tienen mayor rotación en volumen, la categoría Tecnología representa el 69.4% de la facturación global y es la que mayor valor incremental aporta a la canasta.
3. **El análisis transaccional orienta la inversión física y digital:** El layout de tienda y los algoritmos del e-commerce deben abandonar la división tradicional por departamentos cerrados para organizarse según la co-ocurrencia real demostrada por los clientes.

#### Dos Limitaciones Metodológicas del Estudio:
1. **Sesgo Estacional de Campaña:** Los datos corresponden a los meses de **octubre a diciembre de 2025** (Black Friday, Cyber Ripley y Navidad), periodos con alta concentración de regalos y promociones especiales que no necesariamente reflejan el patrón de consumo en meses como marzo o julio.
2. **Correlación Transaccional vs. Causalidad:** El algoritmo Apriori identifica qué ítems aparecen juntos en la canasta, pero no determina la secuencia cronológica de elección dentro de la tienda ni registra si la co-ocurrencia fue provocada por una promoción previa que estaba activa en ese momento.

---

## 3. FICHA TÉCNICA Y ENTREGABLES ADJUNTOS
- **Script Principal Python:** `ejecutar_analisis.py` (ejecución modular reproducible con exportación automática).
- **Notebook Interactivo Jupyter:** `ripley_apriori.ipynb` (con código, comentarios didácticos y visualizaciones gráficas).
- **Consolidado de Datos en Excel:** `Resultados_Ripley_Apriori_Grupo2.xlsx` (con hojas: `KPIs_Canasta`, `Top10_Productos_Soporte`, `Top10_Reglas_Lift`, `5_Reglas_Estrategicas`, `Analisis_Tiendas`).
- **Dataset Fuente Oficial:** `Dataset_Ripley_ReglasAsociacion.xlsx`.

