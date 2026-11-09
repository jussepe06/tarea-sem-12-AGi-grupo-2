# tarea-sem-12-AGi-grupo-2
1. Logro de la sesión
Al finalizar la actividad, el estudiante aplica el algoritmo Apriori para descubrir reglas de asociación 
a partir de datos transaccionales reales de un retailer, interpreta las métricas de soporte, confianza 
y lift, y formula una estrategia de ventas basada en evidencia (venta cruzada, combos, ubicación de 
productos y promociones).
Capacidades específicas
 Preparar y transformar datos transaccionales al formato de canasta (one-hot por ticket).
 Aplicar Apriori para calcular e interpretar soporte, confianza y lift.
 Usar herramientas de minería de datos para generar y filtrar reglas de asociación en KNIME, 
Python o Excel.
 Comunicar resultados y traducir hallazgos analíticos en decisiones comerciales accionables.
2. Caso de estudio: Ripley quiere vender más
Ripley es una de las cadenas de tiendas por departamento más importantes del Perú. Su área de 
Inteligencia Comercial observa que el ticket promedio se ha estancado y quiere impulsar la venta 
cruzada: lograr que cada cliente que entra a comprar un producto se lleve también productos 
complementarios.
Facultad de Ingeniería e Inteligencia Artificial
Ing. de Sistemas de Información
Ing. de Software 
Ciencia de Datos e IA 
Ing. en Ciberseguridad
Ing. en Inteligencia Artificial
Te incorporas como analista de datos junior. El equipo te entrega un extracto de 1 000 tickets de 
compra (boletas) de cinco tiendas Ripley de Lima, correspondientes a la campaña de fin de año 
(octubre–diciembre 2025). Tu misión es responder con datos a una pregunta de negocio muy 
concreta:
¿Qué productos se compran juntos y cómo aprovechamos ese patrón para vender más?
Para resolverlo aplicarás Market Basket Analysis mediante reglas de asociación: el mismo principio 
detrás del clásico “los clientes que compran X también compran Y”.
3. Marco conceptual: soporte, confianza y lift
Una regla de asociación tiene la forma A → B (si el cliente compra A, tiende a comprar B). A es el 
antecedente y B el consecuente. Con N = número total de tickets, las tres métricas clave son:
Métrica Fórmula Qué mide / cómo se lee
Soporte
(support) sop(A→B) = N(A∩B) / N
Qué tan frecuente es la combinación dentro del 
total de tickets. Mide la popularidad o relevancia 
de la regla.
Confianza
(confidence) conf(A→B) = N(A∩B) / N(A) De los clientes que compran A, qué porcentaje 
también compra B. Mide la fiabilidad de la regla.
Lift lift(A→B) = conf(A→B) / 
sop(B)
Cuántas veces más probable es comprar B cuando 
se compra A frente al azar. lift > 1: asociación 
real; = 1: independencia; < 1: sustitución.
Criterio para una buena regla
Una regla útil combina soporte suficiente (no es anecdótica), confianza alta (predice bien) y lift > 1 (la 
relación es genuina, no producto del azar). El mayor lift no siempre es la mejor regla: prioriza también 
el sentido comercial.
4. El dataset
Trabajarás con el archivo Dataset_Ripley_ReglasAsociacion.xlsx. Está en formato largo (una fila por 
ítem vendido); varias filas con el mismo TicketID conforman una misma canasta. Contiene 1 000 
tickets, 4 434 líneas de venta y 39 productos distintos repartidos en 9 categorías y 5 tiendas.
Hoja Contenido
Transacciones Detalle de cada compra: TicketID, Fecha, Tienda, Categoría, Producto y Precio (S/). Es 
la hoja principal del análisis.
Catálogo Maestro de los 39 productos con su categoría y precio referencial.
Resumen KPIs del dataset y soporte por producto, calculados con fórmulas (se recalculan 
solos).
Diccionario Descripción de cada columna del archivo.
Vistazo a la hoja Transacciones
TicketID Tienda Categoría Producto Precio (S/)
Facultad de Ingeniería e Inteligencia Artificial
Ing. de Sistemas de Información
Ing. de Software 
Ciencia de Datos e IA 
Ing. en Ciberseguridad
Ing. en Inteligencia Artificial
T00001 Ripley Jockey Plaza Tecnología Audífonos Bluetooth 199
T00001 Ripley Jockey Plaza Tecnología Smartphone 1799
T00001 Ripley Jockey Plaza Moda hombre Polo 59
T00002 Ripley Jockey Plaza Hogar Juego de sábanas 149
T00003 Ripley Mega Plaza Tecnología Smartwatch 599
La variable clave para las reglas es Producto, agrupada por TicketID. Para Apriori conviene 
transformarla a formato one-hot (una columna por producto, 1 si aparece en el ticket y 0 si no).
5. Instrucciones según la herramienta elegida
Resuelve la práctica con la herramienta de tu preferencia. El resultado analítico debe ser 
equivalente entre las tres opciones; lo que cambia es el procedimiento. Elige UNA y márcala en el 
encabezado.
Opción A · KNIME Analytics Platform
Construye el siguiente flujo de nodos:
1. Excel Reader: leer la hoja Transacciones del archivo Excel.
2. GroupBy: agrupar por TicketID y agregar la columna Producto con el método de agregación “List” 
o “Set”. Obtendrás una columna de colección = la canasta de cada ticket.
3. Association Rule Learner (Borgelt): conectar la columna de colección; configura Minimum 
Support (p. ej. 3–5 %) y Minimum Confidence (p. ej. 30–40 %).
4. Sorter: ordenar las reglas por Lift en orden descendente.
5. Row Filter: (opcional) quedarte solo con las reglas de Lift > 1.
Tip KNIME
Si prefieres separar los pasos, usa primero Item Set Finder (Borgelt) para los itemsets frecuentes y 
luego Association Rule Learner (Borgelt). Exporta la tabla de reglas con Excel Writer (o una captura) 
para tu informe.
Opción B · Python (pandas + mlxtend)
Entorno sugerido: Jupyter Notebook o Google Colab. Instala mlxtend si hace falta con pip install 
mlxtend. Completa el siguiente código de arranque:
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# 1) Cargar y armar las canastas por ticket
df = pd.read_excel("Dataset_Ripley_ReglasAsociacion.xlsx", sheet_name="Transacciones")
canastas = df.groupby("TicketID")["Producto"].apply(list).tolist()
# 2) Formato one-hot (una columna por producto)
te = TransactionEncoder()
matriz = pd.DataFrame(te.fit(canastas).transform(canastas), columns=te.columns_)
# 3) Itemsets frecuentes (ajusta el soporte minimo)
frecuentes = apriori(matriz, min_support=0.03, use_colnames=True)
Facultad de Ingeniería e Inteligencia Artificial
Ing. de Sistemas de Información
Ing. de Software 
Ciencia de Datos e IA 
Ing. en Ciberseguridad
Ing. en Inteligencia Artificial
# 4) Reglas de asociacion (ajusta la confianza minima)
reglas = association_rules(frecuentes, metric="confidence", min_threshold=0.30)
# 5) Ordenar por lift y mostrar las mejores
reglas = reglas.sort_values("lift", ascending=False)
print(reglas[["antecedents","consequents","support","confidence","lift"]].head(10))
Prueba distintos valores de min_support y min_threshold, y filtra las reglas con lift > 1 para 
quedarte con las accionables.
Opción C · Microsoft Excel
Excel no ejecuta Apriori directamente, así que harás un análisis guiado sobre pares de productos 
candidatos:
1. Soporte individual: en la hoja Transacciones crea una Tabla Dinámica con Producto en Filas y 
Recuento distinto de TicketID en Valores (o usa la hoja Resumen, que ya trae el soporte por 
producto).
2. Co-ocurrencia: pivota TicketID en filas y Producto en columnas (valor = 1 si aparece), o usa 
COUNTIFS / SUMAR.SI.CONJUNTO para construir la matriz one-hot.
3. Pares candidatos: elige 8–10 pares que sospeches relacionados (p. ej. Smartphone–Funda para 
celular, Licuadora–Olla arrocera, Vestido–Cartera) y calcula con fórmulas: N(A∩B), Soporte = 
N(A∩B)/1000, Confianza = N(A∩B)/N(A) y Lift = Confianza ÷ (N(B)/1000).
4. Tabla final: arma una tabla con tus reglas y ordénalas por Lift de mayor a menor.
Tip Excel
Usa la hoja Resumen para obtener N(A) y N(B) rápidamente, y COUNTIFS sobre tu tabla one-hot para 
contar N(A∩B). Recuerda que el total de tickets es 1 000.
6. Actividades a desarrollar
Desarrolla y documenta cada actividad en tu informe; incluye el código o las capturas y, sobre todo, 
los resultados numéricos y su interpretación.
Actividad 1 · Preparación de los datos (canasta)
Transforma las transacciones al formato de canasta por TicketID (one-hot). Reporta el número de 
tickets, el número de productos distintos y el tamaño promedio de la canasta.
Actividad 2 · Itemsets frecuentes y soporte
Genera los itemsets frecuentes con un soporte mínimo razonable y justifica el umbral elegido. Lista 
los 10 productos individuales con mayor soporte e interprétalos brevemente.
Actividad 3 · Generación de reglas
Genera las reglas de asociación con tu umbral de confianza. Presenta una tabla con las 10 mejores 
reglas (antecedente, consecuente, soporte, confianza y lift), ordenadas por lift.
Actividad 4 · Interpretación de las 5 mejores reglas
Facultad de Ingeniería e Inteligencia Artificial
Ing. de Sistemas de Información
Ing. de Software 
Ciencia de Datos e IA 
Ing. en Ciberseguridad
Ing. en Inteligencia Artificial
Selecciona las 5 reglas más valiosas. No te guíes solo por el lift: considera también que tengan 
soporte suficiente y sentido comercial. Para cada una, explica en lenguaje de negocio qué significa 
para Ripley.
Actividad 5 · Análisis por tienda o categoría
Repite o filtra el análisis para una tienda específica o una categoría (por ejemplo Tecnología, Moda 
mujer o Electrohogar). ¿Cambian las reglas? ¿Aparecen patrones propios de una tienda?
Actividad 6 · Propuesta de estrategia de ventas (núcleo de la práctica)
A partir de tus reglas, propón una estrategia comercial concreta para aumentar el ticket promedio 
de Ripley. Debe incluir como mínimo:
 2 combos / bundles de productos (con su justificación por soporte y lift).
 2 acciones de venta cruzada en web, app o caja, del tipo “los clientes que compran X también 
llevan Y”.
 1 decisión de layout o ubicación: qué productos acercar o separar en la tienda y por qué.
 1 acción de pricing o promoción (descuento por combo, 2x1, etc.) y el riesgo a vigilar 
(canibalización).
Cuantifica el impacto esperado cuando sea posible (por ejemplo: “el X % de los tickets que incluyen 
A no llevan B; capturar la mitad incrementaría…”).
Actividad 7 · Conclusiones y limitaciones
Resume tus 3 hallazgos principales y menciona al menos 2 limitaciones del análisis (datos de una 
sola campaña, sensibilidad al soporte mínimo, correlación que no implica causalidad, etc.).
7. Entregables y formato de entrega
 Informe en PDF o Word con el desarrollo de las 7 actividades: portada con tus datos, 
resultados, código o capturas e interpretación.
 Archivo de trabajo: workflow .knwf (KNIME), notebook .ipynb / script .py (Python) o libro .xlsx 
con tus cálculos (Excel).
 Extensión sugerida: 6–10 páginas. La redacción cuidada forma parte de la calificación.
 Entrega por el Aula Virtual antes de la fecha indicada por el docente.
8. Rúbrica de evaluación
Criterio Descripción Pts.
Preparación de datos Formato de canasta correcto; KPIs de la canasta reportados 
(Act. 1). 2
Itemsets y soporte
Itemsets frecuentes correctos; umbral justificado; top de 
productos (Act. 2). 3
Generación de reglas Tabla de reglas con soporte, confianza y lift bien calculados 
(Act. 3). 4
Interpretación Lectura de negocio correcta de las 5 mejores reglas (Act. 4). 3
Facultad de Ingeniería e Inteligencia Artificial
Ing. de Sistemas de Información
Ing. de Software 
Ciencia de Datos e IA 
Ing. en Ciberseguridad
Ing. en Inteligencia Artificial
Análisis por tienda / 
categoría Comparación pertinente y bien argumentada (Act. 5). 2
Estrategia de ventas Propuesta accionable y justificada con métricas (Act. 6). 4
Conclusiones y redacción Hallazgos, limitaciones y calidad de redacción (Act. 7). 2
TOTAL 20
Recuerda
El objetivo no es solo correr el algoritmo, sino convertir los números en decisiones de negocio. Una 
regla con alto lift pero soporte mínimo puede ser una curiosidad; la que combina soporte, confianza 
y lift con sentido comercial es la que hace ganar dinero a Ripley
