import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def main():
    print("="*70)
    print("   CASO RIPLEY: MARKET BASKET ANALYSIS CON APRIORI (PYTHON)")
    print("="*70)

    # 1. Cargar datos
    excel_path = 'Dataset_Ripley_ReglasAsociacion.xlsx'
    df_trans = pd.read_excel(excel_path, sheet_name='Transacciones')
    df_cat = pd.read_excel(excel_path, sheet_name='Catálogo')
    
    # 2. Métricas de Canasta (Actividad 1)
    n_tickets = df_trans['TicketID'].nunique()
    n_productos = df_trans['Producto'].nunique()
    n_lineas = len(df_trans)
    
    canastas_serie = df_trans.groupby('TicketID')['Producto'].count()
    tam_prom = canastas_serie.mean()
    tam_min = canastas_serie.min()
    tam_max = canastas_serie.max()
    tam_std = canastas_serie.std()
    
    gasto_ticket = df_trans.groupby('TicketID')['Precio (S/)'].sum()
    venta_total = gasto_ticket.sum()
    ticket_prom_soles = gasto_ticket.mean()

    print(f"\n[ACTIVIDAD 1] KPIs DE LA CANASTA:")
    print(f"  - Total de boletas/tickets analizados: {n_tickets:,}")
    print(f"  - Total de líneas transaccionales: {n_lineas:,}")
    print(f"  - Variedad de productos con venta: {n_productos} productos")
    print(f"  - Tamaño promedio de canasta: {tam_prom:.2f} ítems/ticket (mín: {tam_min}, máx: {tam_max}, desv: {tam_std:.2f})")
    print(f"  - Venta total acumulada: S/ {venta_total:,.2f}")
    print(f"  - Gasto promedio por ticket: S/ {ticket_prom_soles:,.2f}")

    # 3. Transformación a formato canasta (One-Hot Encoding)
    canastas = df_trans.groupby('TicketID')['Producto'].apply(list).tolist()
    te = TransactionEncoder()
    matriz_binaria = te.fit(canastas).transform(canastas)
    df_onehot = pd.DataFrame(matriz_binaria, columns=te.columns_)

    # 4. Itemsets Frecuentes y Top 10 Soporte (Actividad 2)
    soporte_individual = df_onehot.mean().sort_values(ascending=False)
    top10_ind = pd.DataFrame({
        'Producto': soporte_individual.head(10).index,
        'Soporte': soporte_individual.head(10).values,
        'Tickets': (soporte_individual.head(10).values * n_tickets).astype(int),
        'Porcentaje': (soporte_individual.head(10).values * 100).round(2)
    }).merge(df_cat[['Producto', 'Categoría', 'Precio referencial (S/)']], on='Producto', how='left')

    print(f"\n[ACTIVIDAD 2] TOP 10 PRODUCTOS CON MAYOR SOPORTE:")
    for idx, row in top10_ind.iterrows():
        print(f"  {idx+1}. {row['Producto']} ({row['Categoría']}): Soporte = {row['Soporte']:.3f} ({row['Porcentaje']}%) | {row['Tickets']} tickets | S/ {row['Precio referencial (S/)']}")

    # Itemsets frecuentes con umbral justificado min_support = 0.03 (30 tickets)
    frecuentes = apriori(df_onehot, min_support=0.03, use_colnames=True)
    frecuentes['tamaño'] = frecuentes['itemsets'].apply(lambda x: len(x))

    print(f"\n  Itemsets Frecuentes con Soporte >= 3.0% (mínimo 30 clientes): {len(frecuentes)} conjuntos")
    print(f"    - Conjuntos de 1 producto (1-itemsets): {len(frecuentes[frecuentes['tamaño'] == 1])}")
    print(f"    - Conjuntos de 2 productos (2-itemsets): {len(frecuentes[frecuentes['tamaño'] == 2])}")
    print(f"    - Conjuntos de 3 productos (3-itemsets): {len(frecuentes[frecuentes['tamaño'] == 3])}")

    # 5. Generación de Reglas de Asociación (Actividad 3)
    reglas = association_rules(frecuentes, metric='confidence', min_threshold=0.25)
    reglas = reglas[reglas['lift'] > 1.0].sort_values(by=['lift', 'support'], ascending=[False, False])

    # Convertir frozensets a strings legibles
    reglas['antecedente_str'] = reglas['antecedents'].apply(lambda x: ', '.join(list(x)))
    reglas['consecuente_str'] = reglas['consequents'].apply(lambda x: ', '.join(list(x)))

    print(f"\n[ACTIVIDAD 3] REGLAS GENERADAS (Confianza >= 25%, Lift > 1.0): {len(reglas)} reglas")
    print(f"\n  TOP 10 MEJORES REGLAS ORDENADAS POR LIFT:")
    top10_reglas = reglas.head(10).copy()
    for idx, r in top10_reglas.reset_index().iterrows():
        print(f"  R{idx+1}: [{r['antecedente_str']}]  -->  [{r['consecuente_str']}]")
        print(f"      Soporte: {r['support']:.3f} ({r['support']*100:.1f}%) | Confianza: {r['confidence']:.3f} ({r['confidence']*100:.1f}%) | Lift: {r['lift']:.3f}")

    # 6. Selección de las 5 Reglas más valiosas de negocio (Actividad 4)
    # Seleccionamos reglas con sentido comercial transversal: Electrohogar, Moda Mujer, Tecnología, Accesorios y Deportes
    reglas_seleccionadas = [
        # Regla 1: Olla arrocera -> Licuadora (Electrohogar)
        reglas[(reglas['antecedente_str'] == 'Olla arrocera') & (reglas['consecuente_str'] == 'Licuadora')].iloc[0],
        # Regla 2: Cartera -> Vestido (Moda Mujer)
        reglas[(reglas['antecedente_str'] == 'Cartera') & (reglas['consecuente_str'] == 'Vestido')].iloc[0],
        # Regla 3: Smartwatch -> Smartphone (Tecnología)
        reglas[(reglas['antecedente_str'] == 'Smartwatch') & (reglas['consecuente_str'] == 'Smartphone')].iloc[0],
        # Regla 4: Funda para celular + Audífonos Bluetooth -> Smartphone
        reglas[(reglas['antecedente_str'].isin(['Audífonos Bluetooth, Funda para celular', 'Funda para celular, Audífonos Bluetooth'])) & (reglas['consecuente_str'] == 'Smartphone')].iloc[0],
        # Regla 5: Teclado + Mouse -> Mochila
        reglas[(reglas['antecedente_str'].isin(['Mouse inalámbrico, Teclado inalámbrico', 'Teclado inalámbrico, Mouse inalámbrico'])) & (reglas['consecuente_str'] == 'Mochila')].iloc[0]
    ]

    print(f"\n[ACTIVIDAD 4] 5 REGLAS MÁS VALIOSAS PARA ESTRATEGIA COMERCIAL:")
    for i, r in enumerate(reglas_seleccionadas, 1):
        print(f"  Regla Estratégica {i}: [{r['antecedente_str']}]  ==>  [{r['consecuente_str']}]")
        print(f"      Soporte: {r['support']*100:.2f}% | Confianza: {r['confidence']*100:.2f}% | Lift: {r['lift']:.2f}")

    # 7. Análisis por Tienda (Actividad 5)
    tiendas_res = []
    for t in df_trans['Tienda'].unique():
        df_sub = df_trans[df_trans['Tienda'] == t]
        t_tickets = df_sub['TicketID'].nunique()
        t_lineas = len(df_sub)
        t_venta = df_sub['Precio (S/)'].sum()
        t_prom_ticket = t_venta / t_tickets
        t_prom_items = t_lineas / t_tickets
        
        # Categoría más vendida
        cat_top = df_sub.groupby('Categoría')['Producto'].count().sort_values(ascending=False).index[0]
        
        tiendas_res.append({
            'Tienda': t,
            'Tickets': t_tickets,
            'Líneas': t_lineas,
            'Venta Total (S/)': t_venta,
            'Ticket Promedio (S/)': t_prom_ticket,
            'Canasta Promedio': t_prom_items,
            'Categoría Principal': cat_top
        })
    df_tiendas = pd.DataFrame(tiendas_res).sort_values(by='Tickets', ascending=False)
    print(f"\n[ACTIVIDAD 5] COMPARATIVA POR TIENDA:")
    print(df_tiendas.to_string(index=False))

    # Guardar resultados en Excel para entrega adjunta
    output_excel = 'Resultados_Ripley_Apriori_Grupo2.xlsx'
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        pd.DataFrame([{
            'Indicador': 'Tickets analizados', 'Valor': n_tickets
        }, {
            'Indicador': 'Líneas de venta', 'Valor': n_lineas
        }, {
            'Indicador': 'Productos distintos', 'Valor': n_productos
        }, {
            'Indicador': 'Tamaño medio canasta', 'Valor': round(tam_prom, 2)
        }, {
            'Indicador': 'Venta total acumulada (S/)', 'Valor': round(venta_total, 2)
        }, {
            'Indicador': 'Ticket promedio (S/)', 'Valor': round(ticket_prom_soles, 2)
        }]).to_excel(writer, sheet_name='KPIs_Canasta', index=False)
        
        top10_ind.to_excel(writer, sheet_name='Top10_Productos_Soporte', index=False)
        top10_reglas[['antecedente_str', 'consecuente_str', 'support', 'confidence', 'lift']].rename(columns={
            'antecedente_str': 'Antecedente (A)',
            'consecuente_str': 'Consecuente (B)',
            'support': 'Soporte',
            'confidence': 'Confianza',
            'lift': 'Lift'
        }).to_excel(writer, sheet_name='Top10_Reglas_Lift', index=False)
        
        pd.DataFrame([{
            'Regla': f"[{r['antecedente_str']}] => [{r['consecuente_str']}]",
            'Soporte (%)': round(r['support']*100, 2),
            'Confianza (%)': round(r['confidence']*100, 2),
            'Lift': round(r['lift'], 2)
        } for r in reglas_seleccionadas]).to_excel(writer, sheet_name='5_Reglas_Estrategicas', index=False)

        df_tiendas.to_excel(writer, sheet_name='Analisis_Tiendas', index=False)

    print(f"\n-> Archivo Excel con resultados exportado con éxito: {output_excel}")
    print("="*70)

if __name__ == '__main__':
    main()

