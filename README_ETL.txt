Repositorio ETL - Dataset de Ventas (integrado)

Archivos incluidos:
- dataset_original.csv  -> Datos simulados con errores (duplicados, formatos mixtos, nulos, valores 'N/A', comas en decimales, etc.)
- dataset_limpio.csv   -> Datos limpios y normalizados. Transformaciones aplicadas:
    * Normalización de formatos de fecha a YYYY-MM-DD
    * Extracción de Día, Mes, Año y Trimestre
    * Conversión de Precio_Unitario a numérico (coma->punto)
    * Imputación de Marca faltante con 'Desconocida'
    * Cálculo de Ingreso_Total cuando faltaba (Precio_Unitario * Cantidad_Vendida)
    * Cálculo de Costo_Total como 60% del Ingreso_Total cuando faltaba
    * Cálculo de Ganancia = Ingreso_Total - Costo_Total
    * Normalización de texto (title case) y corrección de errores comunes
    * Eliminación de duplicados exactos
- README_ETL.txt       -> Este archivo

Ruta de los archivos en el entorno: /mnt/data/