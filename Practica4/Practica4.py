
"""
Práctica 4 - Tests Estadísticos
Detecta columnas y prueba si las clases/etiquetas difieren en la variable numérica.
Ej.: probar si 'Sales' difiere entre 'Category' (ANOVA + post-hoc o Kruskal-Wallis + post-hoc).
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations


# Intenta usar primero CSV; si no existe, busca XLSX y toma la primera hoja.
CSV_NAME = "superstore_clean.csv"
XLSX_NAME = "superstore_clean.practica2_reporte.xlsx"

OUT_SUMMARY = "practica4_resultados.csv"

# CARGA DE DATOS (automática)

base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
csv_path = base_dir / CSV_NAME
xlsx_path = base_dir / XLSX_NAME

if csv_path.exists():
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    source = csv_path.name
elif xlsx_path.exists():
    xls = pd.ExcelFile(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=xls.sheet_names[0])
    source = xlsx_path.name
else:
    raise FileNotFoundError(f"No encontré {CSV_NAME} ni {XLSX_NAME} en {base_dir}")

print(f"Archivo leído: {source}")
print("Columnas disponibles:", list(df.columns))

# Normalizar espacios en nombres de columnas
df.columns = [c.strip() for c in df.columns]


# DETECCIÓN AUTOMÁTICA DE VENTAS (num) Y ETIQUETA (cat)

sales_col = None
cat_col = None
for c in df.columns:
    lc = c.lower()
    if sales_col is None and ("sales" in lc or "venta" in lc or "revenue" in lc):
        sales_col = c
    if cat_col is None and ("category" in lc or "segment" in lc or "categoria" in lc or "type" in lc):
        cat_col = c

# Fallbacks si no encontrados
if sales_col is None:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if not nums:
        raise ValueError("No encontré columnas numéricas para analizar.")
    sales_col = nums[0]
if cat_col is None:
    objs = df.select_dtypes(include=['object']).columns.tolist()
    if not objs:
        raise ValueError("No encontré columnas categóricas para agrupar (labels).")
    cat_col = objs[0]

print(f"Usaré como variable numérica: '{sales_col}'")
print(f"Usaré como etiqueta/categoría: '{cat_col}'")

# Eliminar filas con NA en las columnas utilizadas
df = df[[sales_col, cat_col]].dropna()


# RESUMEN POR GRUPO

grouped = df.groupby(cat_col)[sales_col].agg(['count','mean','median','std','min','max']).reset_index()
print("\nResumen por grupo:")
print(grouped)


# COMPROBAR SUPUESTOS



alpha = 0.05
print("\nComprobando supuestos (alpha = {:.3f}):".format(alpha))

# Normalidad: Shapiro para grupos 
normality_results = {}
for name, group in df.groupby(cat_col)[sales_col]:
    n = len(group)
    if n >= 5000:
        # Si demasiadas observaciones, usar D'Agostino K^2
        stat, p = stats.normaltest(group)
        test_name = "D'Agostino"
    else:
        stat, p = stats.shapiro(group)
        test_name = "Shapiro"
    normality_results[name] = (test_name, stat, p, p > alpha)
    print(f"  {name}: {test_name} stat={stat:.4f}, p={p:.4e} -> normal? {p>alpha}")

# Homogeneidad de varianzas: Levene
groups = [g[sales_col].values for _, g in df.groupby(cat_col)]
lev_stat, lev_p = stats.levene(*groups, center='median')
print(f"\nLevene test: stat={lev_stat:.4f}, p={lev_p:.4e} -> homogeneidad? {lev_p>alpha}")


# ELEGIR TEST: ANOVA  o Kruskal-Wallis 

# Decisión simple: si TODAS las muestras pasan normalidad y Levene pasa -> ANOVA
all_normal = all(res[3] for res in normality_results.values())
homog = (lev_p > alpha)

use_anova = all_normal and homog

print(f"\nDecisión automática: usar ANOVA? {use_anova} (all_normal={all_normal}, homogeneidad={homog})")

results_summary = []

if use_anova:
    # ANOVA vía statsmodels
    formula = f'Q("{sales_col}") ~ C(Q("{cat_col}"))'
    # alternativa: usar ols with safe column names
    df_for_anova = df.rename(columns={sales_col: "target", cat_col: "group"})
    model = ols('target ~ C(group)', data=df_for_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nANOVA (tabla):")
    print(anova_table)
    results_summary.append(("ANOVA", None, anova_table.to_dict()))
    # Post-hoc: pairwise t-tests with Bonferroni correction 
    groups_names = df_for_anova['group'].unique()
    posthoc = []
    for a,b in combinations(groups_names, 2):
        ga = df_for_anova.loc[df_for_anova['group']==a, 'target']
        gb = df_for_anova.loc[df_for_anova['group']==b, 'target']
        tstat, pval = stats.ttest_ind(ga, gb, equal_var=False)  # welch
        posthoc.append((a,b,tstat,pval))
    # ajustar p-values (Bonferroni)
    m = len(posthoc)
    posthoc_adj = [(a,b,t,p, min(1, p*m)) for (a,b,t,p) in posthoc]
    print("\nPost-hoc pairwise t-tests (Welch) con Bonferroni:")
    for a,b,t,p,adj in posthoc_adj:
        print(f"  {a} vs {b}: t={t:.4f}, p={p:.4e}, p_bonf={adj:.4e} -> significant? {adj < alpha}")
    results_summary.append(("posthoc_t_bonf", None, posthoc_adj))
else:
    # Kruskal-Wallis test
    kw_stat, kw_p = stats.kruskal(*groups)
    print("\nKruskal-Wallis test:")
    print(f"  stat={kw_stat:.4f}, p={kw_p:.4e} -> grupos distintos? {kw_p < alpha}")
    results_summary.append(("Kruskal-Wallis", kw_stat, kw_p))
    # Post-hoc: pairwise Mann-Whitney U tests with Bonferroni
    posthoc = []
    group_names = [name for name, _ in df.groupby(cat_col)]
    for a,b in combinations(group_names, 2):
        ga = df.loc[df[cat_col]==a, sales_col]
        gb = df.loc[df[cat_col]==b, sales_col]
        u, p = stats.mannwhitneyu(ga, gb, alternative='two-sided')
        posthoc.append((a,b,u,p))
    m = len(posthoc)
    posthoc_adj = [(a,b,u,p, min(1, p*m)) for (a,b,u,p) in posthoc]
    print("\nPost-hoc pairwise Mann-Whitney U con Bonferroni:")
    for a,b,u,p,adj in posthoc_adj:
        print(f"  {a} vs {b}: U={u:.4f}, p={p:.4e}, p_bonf={adj:.4e} -> significant? {adj < alpha}")
    results_summary.append(("posthoc_mw_bonf", None, posthoc_adj))

# GUARDAR RESULTADOS BÁSICOS A CSV

# preparar salida sencilla
out_rows = []
out_rows.append({"test":"decision", "detail": "used_anova", "value": use_anova})
out_rows.append({"test":"levene_p", "detail":"levene_p", "value": float(lev_p)})

for grp, (tst, stat, p, ok) in zip(normality_results.keys(), [(normality_results[k][0], normality_results[k][1], normality_results[k][2], normality_results[k][3]) for k in normality_results]):
    out_rows.append({"test":"normality", "detail": grp, "value": float(stat)})

# posthoc rows
for rec in results_summary:
    tag = rec[0]
    body = rec[2]
    if isinstance(body, list):
        for item in body:
            if len(item) == 5:  
                a,b,stat,p,adj = item
                out_rows.append({"test":tag, "detail": f"{a}_vs_{b}", "value": float(adj)})
            else:
                # other forms
                out_rows.append({"test":tag, "detail": str(item[:2]), "value": float(item[2]) if len(item)>2 else None})
    else:
        # scalars
        out_rows.append({"test":tag, "detail":"stat_or_p", "value": float(body) if body is not None else None})

out_df = pd.DataFrame(out_rows)
out_df.to_csv(base_dir / OUT_SUMMARY, index=False)
print(f"\n✅ Resultados guardados en {OUT_SUMMARY}")


# INTERPRETACIÓN BÁSICA (en consola)

print("\n=== Interpretación guía ===")
if use_anova:
    p_anova = anova_table["PR(>F)"][0]
    if p_anova < alpha:
        print(f" - ANOVA: p = {p_anova:.4e} -> Rechazamos H0: hay diferencias entre al menos dos grupos en '{sales_col}' según '{cat_col}'.")
    else:
        print(f" - ANOVA: p = {p_anova:.4e} -> No rechazamos H0: no hay evidencia de diferencias entre grupos.")
else:
    if kw_p < alpha:
        print(f" - Kruskal-Wallis: p = {kw_p:.4e} -> Rechazamos H0: hay diferencias entre grupos.")
    else:
        print(f" - Kruskal-Wallis: p = {kw_p:.4e} -> No rechazamos H0: no hay evidencia de diferencias.")

print("\nRevisa el CSV de salida para valores numéricos y los p-values ajustados de las comparaciones pairwise.")
