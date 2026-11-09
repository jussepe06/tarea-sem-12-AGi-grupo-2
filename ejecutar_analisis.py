import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# Actividad 1: Preparacion de los datos
# ==========================================
df = pd.read_excel("Dataset_Ripley_ReglasAsociacion.xlsx", sheet_name="Transacciones")
catalogo = pd.read_excel("Dataset_Ripley_ReglasAsociacion.xlsx", sheet_name="Catálogo")

# Agrupamos los productos comprados en cada ticket
canastas = df.groupby("TicketID")["Producto"].apply(list).tolist()

# Pasamos a formato one-hot (1/True si el producto esta en el ticket, 0/False si no)
te = TransactionEncoder()
matriz = pd.DataFrame(te.fit(canastas).transform(canastas), columns=te.columns_)

# Calculo de KPIs basicos de la canasta
num_tickets = df["TicketID"].nunique()
num_productos = df["Producto"].nunique()
tam_canasta = df.groupby("TicketID")["Producto"].count()
tam_promedio = tam_canasta.mean()
venta_total = df["Precio (S/)"].sum()
ticket_promedio = df.groupby("TicketID")["Precio (S/)"].sum().mean()

print("--- ACTIVIDAD 1: KPIS DE LA CANASTA ---")
print(f"Numero total de tickets: {num_tickets}")
print(f"Total de lineas de venta: {len(df)}")
print(f"Productos distintos vendidos: {num_productos}")
print(f"Tamano promedio de la canasta: {tam_promedio:.3f} productos por ticket")
print(f"Venta total: S/ {venta_total:,.2f} | Ticket promedio: S/ {ticket_promedio:,.2f}\n")

# ==========================================
# Actividad 2: Itemsets frecuentes y soporte
# ==========================================
# Top 10 productos individuales con mayor soporte
soporte_productos = matriz.mean().sort_values(ascending=False).head(10)
top10_productos = pd.DataFrame({
    "Producto": soporte_productos.index,
    "Soporte": soporte_productos.values,
    "Soporte (%)": (soporte_productos.values * 100).round(1),
    "Tickets": (soporte_productos.values * num_tickets).astype(int)
})

print("--- ACTIVIDAD 2: TOP 10 PRODUCTOS CON MAYOR SOPORTE ---")
print(top10_productos.to_string(index=False))

# Aplicamos Apriori con soporte minimo de 3% (minimo 30 tickets de 1000)
frecuentes = apriori(matriz, min_support=0.03, use_colnames=True)
frecuentes["cantidad_items"] = frecuentes["itemsets"].apply(len)

print(f"\nTotal de itemsets frecuentes (soporte >= 3%): {len(frecuentes)}")
print(f" - De 1 producto: {len(frecuentes[frecuentes['cantidad_items'] == 1])}")
print(f" - De 2 productos: {len(frecuentes[frecuentes['cantidad_items'] == 2])}")
print(f" - De 3 productos: {len(frecuentes[frecuentes['cantidad_items'] == 3])}\n")

# ==========================================
# Actividad 3: Generacion de reglas
# ==========================================
# Generamos las reglas con confianza minima de 30% y filtramos lift > 1
reglas = association_rules(frecuentes, metric="confidence", min_threshold=0.30)
reglas = reglas[reglas["lift"] > 1.0].copy()

# Formateamos antecedentes y consecuentes para que se lean mejor sin el frozenset
reglas["Antecedente"] = reglas["antecedents"].apply(lambda x: ", ".join(list(x)))
reglas["Consecuente"] = reglas["consequents"].apply(lambda x: ", ".join(list(x)))

# Ordenamos por lift de mayor a menor
reglas = reglas.sort_values(["lift", "support"], ascending=[False, False])

print(f"--- ACTIVIDAD 3: TOP 10 REGLAS ORDENADAS POR LIFT (de {len(reglas)} reglas) ---")
tabla_top10 = reglas[["Antecedente", "Consecuente", "support", "confidence", "lift"]].head(10).copy()
tabla_top10["support"] = tabla_top10["support"].round(3)
tabla_top10["confidence"] = tabla_top10["confidence"].round(3)
tabla_top10["lift"] = tabla_top10["lift"].round(3)
print(tabla_top10.to_string(index=False))

# ==========================================
# Actividad 4: Las 5 mejores reglas para el negocio
# ==========================================
# Filtramos 5 reglas representativas de distintas categorias con buen soporte y lift
print("\n--- ACTIVIDAD 4: 5 REGLAS SELECCIONADAS PARA ESTRATEGIA ---")
filtro_5 = [
    ("Olla arrocera", "Licuadora"),
    ("Cartera", "Vestido"),
    ("Smartwatch", "Smartphone"),
    ("Audífonos Bluetooth, Funda para celular", "Smartphone"),
    ("Mouse inalámbrico, Teclado inalámbrico", "Mochila")
]

reglas_5_lista = []
for ant_buscado, con_buscado in filtro_5:
    # Buscamos coincidencia sin importar el orden interno de los items
    set_ant = set(ant_buscado.split(", "))
    set_con = set(con_buscado.split(", "))
    match = reglas[(reglas["antecedents"] == frozenset(set_ant)) & (reglas["consequents"] == frozenset(set_con))]
    if not match.empty:
        r = match.iloc[0]
        reglas_5_lista.append({
            "Antecedente": r["Antecedente"],
            "Consecuente": r["Consecuente"],
            "Soporte": round(r["support"], 3),
            "Confianza": round(r["confidence"], 3),
            "Lift": round(r["lift"], 3)
        })

df_5_reglas = pd.DataFrame(reglas_5_lista)
print(df_5_reglas.to_string(index=False))

# ==========================================
# Actividad 5: Analisis por tienda y categoria
# ==========================================
print("\n--- ACTIVIDAD 5: RESUMEN POR TIENDA ---")
resumen_tiendas = df.groupby("Tienda").agg(
    Tickets=("TicketID", "nunique"),
    Items_Vendidos=("Producto", "count"),
    Venta_Total=("Precio (S/)", "sum")
).reset_index()
resumen_tiendas["Tamano_Canasta_Prom"] = (resumen_tiendas["Items_Vendidos"] / resumen_tiendas["Tickets"]).round(2)
resumen_tiendas["Ticket_Promedio_Soles"] = (resumen_tiendas["Venta_Total"] / resumen_tiendas["Tickets"]).round(2)
resumen_tiendas = resumen_tiendas.sort_values("Tickets", ascending=False)
print(resumen_tiendas.to_string(index=False))

# Ejemplo de reglas filtrando solo una categoria: Tecnologia
df_tec = df[df["Categoría"] == "Tecnología"]
canastas_tec = df_tec.groupby("TicketID")["Producto"].apply(list).tolist()
te_tec = TransactionEncoder()
matriz_tec = pd.DataFrame(te_tec.fit(canastas_tec).transform(canastas_tec), columns=te_tec.columns_)
frec_tec = apriori(matriz_tec, min_support=0.05, use_colnames=True)
reglas_tec = association_rules(frec_tec, metric="confidence", min_threshold=0.40)
reglas_tec = reglas_tec[reglas_tec["lift"] > 1.0].sort_values("lift", ascending=False)
reglas_tec["Antecedente"] = reglas_tec["antecedents"].apply(lambda x: ", ".join(list(x)))
reglas_tec["Consecuente"] = reglas_tec["consequents"].apply(lambda x: ", ".join(list(x)))

print("\n--- TOP 5 REGLAS DENTRO DE LA CATEGORIA TECNOLOGIA ---")
print(reglas_tec[["Antecedente", "Consecuente", "support", "confidence", "lift"]].head(5).round(3).to_string(index=False))

# Exportamos las tablas a un Excel de resultados
with pd.ExcelWriter("Resultados_Ripley_Apriori_Grupo2.xlsx", engine="openpyxl") as writer:
    top10_productos.to_excel(writer, sheet_name="Top10_Productos", index=False)
    tabla_top10.to_excel(writer, sheet_name="Top10_Reglas", index=False)
    df_5_reglas.to_excel(writer, sheet_name="5_Mejores_Reglas", index=False)
    resumen_tiendas.to_excel(writer, sheet_name="Resumen_Tiendas", index=False)

print("\nArchivo 'Resultados_Ripley_Apriori_Grupo2.xlsx' guardado correctamente.")
