import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar transacciones
df = pd.read_excel('Dataset_Ripley_ReglasAsociacion.xlsx', sheet_name='Transacciones')

print("=== 1. KPIs BÁSICOS ===")
n_tickets = df['TicketID'].nunique()
n_productos = df['Producto'].nunique()
tam_prom = df.groupby('TicketID')['Producto'].count().mean()
print(f"Número de tickets únicos: {n_tickets}")
print(f"Número de productos distintos vendidos: {n_productos}")
print(f"Tamaño promedio de canasta: {tam_prom:.3f} productos")

# Agrupar canastas
canastas = df.groupby('TicketID')['Producto'].apply(list).tolist()

# One-hot
te = TransactionEncoder()
te_ary = te.fit(canastas).transform(canastas)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

print("\n=== 2. TOP 10 PRODUCTOS CON MAYOR SOPORTE INDIVIDUAL ===")
soporte_ind = df_onehot.mean().sort_values(ascending=False)
for prod, sop in soporte_ind.head(10).items():
    print(f"{prod}: {sop:.3f} ({int(sop * n_tickets)} tickets)")

print("\n=== 3. ITEMSETS FRECUENTES (min_support = 0.03) ===")
frecuentes_03 = apriori(df_onehot, min_support=0.03, use_colnames=True)
print(f"Total itemsets con soporte >= 3%: {len(frecuentes_03)}")

# Reglas
reglas_03 = association_rules(frecuentes_03, metric='confidence', min_threshold=0.25)
reglas_03_lift = reglas_03[reglas_03['lift'] > 1.0].sort_values(by='lift', ascending=False)
print(f"Total reglas con confianza >= 25% y lift > 1: {len(reglas_03_lift)}")

print("\n--- TOP 15 REGLAS ORDENADAS POR LIFT ---")
for idx, r in reglas_03_lift.head(15).iterrows():
    ant = list(r['antecedents'])
    con = list(r['consequents'])
    print(f"{ant} -> {con} | Sop: {r['support']:.3f} | Conf: {r['confidence']:.3f} | Lift: {r['lift']:.3f}")

