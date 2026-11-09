import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar hojas
excel_path = 'Dataset_Ripley_ReglasAsociacion.xlsx'
df_trans = pd.read_excel(excel_path, sheet_name='Transacciones')
df_cat = pd.read_excel(excel_path, sheet_name='Catálogo')
df_res = pd.read_excel(excel_path, sheet_name='Resumen')

print("==================================================")
print("ACTIVIDAD 1: PREPARACIÓN DE LOS DATOS Y KPIS")
print("==================================================")
n_tickets = df_trans['TicketID'].nunique()
n_productos = df_trans['Producto'].nunique()
n_lineas = len(df_trans)
tam_canasta = df_trans.groupby('TicketID')['Producto'].count()
tam_prom = tam_canasta.mean()
tam_min = tam_canasta.min()
tam_max = tam_canasta.max()
tam_std = tam_canasta.std()

venta_total = df_trans['Precio (S/)'].sum()
ticket_prom = df_trans.groupby('TicketID')['Precio (S/)'].sum().mean()

print(f"Total tickets únicos: {n_tickets}")
print(f"Total líneas de venta: {n_lineas}")
print(f"Total productos distintos vendidos: {n_productos}")
print(f"Tamaño canasta: promedio={tam_prom:.3f}, min={tam_min}, max={tam_max}, std={tam_std:.3f}")
print(f"Venta total: S/ {venta_total:,.2f}")
print(f"Ticket promedio en S/: S/ {ticket_prom:,.2f}")

# Canastas a formato one-hot
canastas = df_trans.groupby('TicketID')['Producto'].apply(list).tolist()
te = TransactionEncoder()
te_ary = te.fit(canastas).transform(canastas)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

print("\n==================================================")
print("ACTIVIDAD 2: ITEMSETS FRECUENTES Y SOPORTE")
print("==================================================")
# Soporte individual
soporte_ind = df_onehot.mean().sort_values(ascending=False)
top10_ind = soporte_ind.head(10)
print("--- TOP 10 PRODUCTOS INDIVIDUALES CON MAYOR SOPORTE ---")
for prod, s in top10_ind.items():
    cnt = int(round(s * n_tickets))
    cat = df_cat.loc[df_cat['Producto'] == prod, 'Categoría'].values
    cat_str = cat[0] if len(cat) > 0 else 'N/A'
    print(f"- {prod} ({cat_str}): Soporte = {s:.3f} ({s*100:.1f}%) | {cnt} tickets")

# Itemsets frecuentes con min_support = 0.03 (30 tickets)
frecuentes_03 = apriori(df_onehot, min_support=0.03, use_colnames=True)
frecuentes_03['longitud'] = frecuentes_03['itemsets'].apply(lambda x: len(x))
print(f"\nItemsets frecuentes con soporte >= 3% (30 tickets): {len(frecuentes_03)}")
print(f"- De 1 producto: {len(frecuentes_03[frecuentes_03['longitud'] == 1])}")
print(f"- De 2 productos: {len(frecuentes_03[frecuentes_03['longitud'] == 2])}")
print(f"- De 3 productos: {len(frecuentes_03[frecuentes_03['longitud'] == 3])}")

print("\n==================================================")
print("ACTIVIDAD 3: GENERACIÓN DE REGLAS (TOP 15)")
print("==================================================")
reglas_03 = association_rules(frecuentes_03, metric='confidence', min_threshold=0.25)
reglas_03_lift = reglas_03[reglas_03['lift'] > 1.0].sort_values(by=['lift', 'support'], ascending=[False, False])
print(f"Total reglas generadas con confianza >= 25% y lift > 1.0: {len(reglas_03_lift)}")

top15_reglas = reglas_03_lift.head(15).copy()
for i, (idx, r) in enumerate(top15_reglas.iterrows(), 1):
    ant = ", ".join(list(r['antecedents']))
    con = ", ".join(list(r['consequents']))
    print(f"R{i}: {ant} => {con}")
    print(f"    Soporte: {r['support']:.3f} ({r['support']*100:.1f}%) | Confianza: {r['confidence']:.3f} ({r['confidence']*100:.1f}%) | Lift: {r['lift']:.3f}")

print("\n==================================================")
print("ACTIVIDAD 5: ANÁLISIS POR TIENDA Y POR CATEGORÍA")
print("==================================================")
print("--- DISTRIBUCIÓN POR TIENDA ---")
tienda_stats = df_trans.groupby('Tienda').agg(
    tickets=('TicketID', 'nunique'),
    lineas=('Producto', 'count'),
    venta_total=('Precio (S/)', 'sum')
)
tienda_stats['ticket_promedio'] = tienda_stats['venta_total'] / tienda_stats['tickets']
tienda_stats['canasta_promedio'] = tienda_stats['lineas'] / tienda_stats['tickets']
print(tienda_stats)

print("\n--- DISTRIBUCIÓN POR CATEGORÍA ---")
cat_stats = df_trans.groupby('Categoría').agg(
    lineas=('Producto', 'count'),
    tickets=('TicketID', 'nunique'),
    venta_total=('Precio (S/)', 'sum')
).sort_values(by='lineas', ascending=False)
print(cat_stats)

# Comparación de reglas: Ripley Jockey Plaza vs Ripley Mega Plaza
def obtener_reglas_tienda(tienda_nombre, min_supp=0.03, min_conf=0.25):
    df_t = df_trans[df_trans['Tienda'] == tienda_nombre]
    canastas_t = df_t.groupby('TicketID')['Producto'].apply(list).tolist()
    te_t = TransactionEncoder()
    df_oh_t = pd.DataFrame(te_t.fit(canastas_t).transform(canastas_t), columns=te_t.columns_)
    frec_t = apriori(df_oh_t, min_support=min_supp, use_colnames=True)
    reg_t = association_rules(frec_t, metric='confidence', min_threshold=min_conf)
    reg_t_lift = reg_t[reg_t['lift'] > 1.0].sort_values(by='lift', ascending=False)
    return len(canastas_t), reg_t_lift

n_jp, reg_jp = obtener_reglas_tienda('Ripley Jockey Plaza')
n_mp, reg_mp = obtener_reglas_tienda('Ripley Mega Plaza')
print(f"\nJockey Plaza ({n_jp} tickets) -> Reglas: {len(reg_jp)}")
print("Top 3 Jockey Plaza:")
for idx, r in reg_jp.head(3).iterrows():
    print(f"  {list(r['antecedents'])} => {list(r['consequents'])} | Sop: {r['support']:.3f} | Conf: {r['confidence']:.3f} | Lift: {r['lift']:.3f}")

print(f"\nMega Plaza ({n_mp} tickets) -> Reglas: {len(reg_mp)}")
print("Top 3 Mega Plaza:")
for idx, r in reg_mp.head(3).iterrows():
    print(f"  {list(r['antecedents'])} => {list(r['consequents'])} | Sop: {r['support']:.3f} | Conf: {r['confidence']:.3f} | Lift: {r['lift']:.3f}")

