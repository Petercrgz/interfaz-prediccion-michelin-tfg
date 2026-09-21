import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
engine = create_engine(os.getenv("DATABASE_URL"))

# procesa el archivo CSV y adapta las columnas a la base de datos
def procesar_csv(csv):
    df = pd.read_csv(csv)
    df.columns = df.columns.str.lower().str.strip()
    return df

def poblar_tablas_externas(conexion, df, nombre_tabla, nombre_columna):
    inicio = time.time()
    valores= set(df[nombre_columna].dropna()                      # evitar errores si hay NaN
             .str.split(";")                  # dividir cada fila en lista
             .explode()                       # convertir listas en filas individuales
             .str.strip()                     # quitar espacios
             .unique())
    
    if valores:
        lista_valores = list(valores)
        claves_valores = [f"(:valor_{i})" for i in range(len(lista_valores))]
        parametros = {f"valor_{i}": valor for i, valor in enumerate(lista_valores)}
        consulta_inyección = text(f"INSERT INTO {nombre_tabla} (nombre) VALUES {', '.join(claves_valores)} ON CONFLICT DO NOTHING")
        conexion.execute(consulta_inyección, parametros)
        
    consulta_todos_los_valores = text(f"SELECT id, nombre FROM {nombre_tabla} ")
    resultado = conexion.execute(consulta_todos_los_valores).fetchall()
    fin = time.time()
    print(f"Tiempo de ejecución para {nombre_tabla}: {fin - inicio:.2f} segundos")
    return {nombre: id for id, nombre in resultado}

def poblar_ubicaciones(conexion, df):
    paises = sorted({str(pais).strip() for pais in df["pais"].dropna() if str(pais).strip()})
    if paises:
        parametros = {f"pais_{i}": pais for i, pais in enumerate(paises)}
        valores = ", ".join(f"(:pais_{i})" for i in range(len(paises)))
        conexion.execute(
            text(f"INSERT INTO paises (nombre) VALUES {valores} ON CONFLICT (nombre) DO NOTHING"),
            parametros
        )
    paises_db = conexion.execute(text("SELECT id, nombre FROM paises")).fetchall()
    ids_paises = {nombre: id_pais for id_pais, nombre in paises_db}

    ubicaciones = sorted({
        (str(fila.ciudad).strip(), str(fila.pais).strip())
        for fila in df.itertuples(index=False)
        if pd.notna(fila.ciudad) and pd.notna(fila.pais)
        and str(fila.ciudad).strip() and str(fila.pais).strip()
    })
    if ubicaciones:
        parametros = {}
        valores = []
        for i, (ciudad, pais) in enumerate(ubicaciones):
            valores.append(f"(:ciudad_{i}, :pais_id_{i})")
            parametros[f"ciudad_{i}"] = ciudad
            parametros[f"pais_id_{i}"] = ids_paises[pais]
        conexion.execute(
            text(f"INSERT INTO ciudades (nombre, pais_id) VALUES {', '.join(valores)}"),
            parametros
        )

    ciudades_db = conexion.execute(text("""
        SELECT ciudades.id, ciudades.nombre, paises.nombre
        FROM ciudades
        JOIN paises ON paises.id = ciudades.pais_id
    """)).fetchall()
    ids_ciudades = {(nombre_ciudad, nombre_pais): id_ciudad
                    for id_ciudad, nombre_ciudad, nombre_pais in ciudades_db}
    return ids_ciudades


def poblar_tabla_principal_y_relaciones(conexion, df, nombre_tabla, nombre_columna,
                                        diccionario_cocinas, diccionario_ideales, ids_ciudades):
    inicio = time.time()
    valores_restaurantes = []
    for fila in df.itertuples(index=False):
        ciudad = str(fila.ciudad).strip()
        pais = str(fila.pais).strip()
        valores_restaurantes.append({
            "nombre": getattr(fila, "nombre", ""),
            "direccion": getattr(fila, "direccion", ""),
            "ciudad_id": ids_ciudades[(ciudad, pais)],
            "precio": int(getattr(fila, "precio", 0)),
            "valoracion": int(getattr(fila, "valoracion", 0))
        })
    ids_restaurante = []
    tamaño_trozo = 1000
    for i in range(0, len(valores_restaurantes), tamaño_trozo):
        trozo = valores_restaurantes[i:i+tamaño_trozo]
        parametros = {}
        claves_valores = []
        for j, restaurante in enumerate(trozo):
            claves_valores.append(
                f"(:nombre_{j}, :direccion_{j}, :ciudad_id_{j}, :precio_{j}, :valoracion_{j})"
            )
            parametros.update({
                f"nombre_{j}": restaurante["nombre"],
                f"direccion_{j}": restaurante["direccion"],
                f"ciudad_id_{j}": restaurante["ciudad_id"],
                f"precio_{j}": restaurante["precio"],
                f"valoracion_{j}": restaurante["valoracion"]
            })
        consulta = text(f"""
            INSERT INTO restaurantes (nombre, direccion, ciudad_id, precio, valoracion)
            VALUES {', '.join(claves_valores)}
            RETURNING id
        """)
        ids_restaurante.extend(id_restaurante for (id_restaurante,) in conexion.execute(consulta, parametros))
    print(f"Se han insertado {len(ids_restaurante)} restaurantes en la tabla '{nombre_tabla}'.")

    relaciones_cocinas = []
    relaciones_ideales = []
    for fila, id_restaurante in zip(df.itertuples(index=False), ids_restaurante):
        tipos_cocina = getattr(fila, "tipos_cocina", "")
        if pd.notna(tipos_cocina) and tipos_cocina:
            for tipo in str(tipos_cocina).split(";"):
                if tipo.strip() in diccionario_cocinas:
                    relaciones_cocinas.append((id_restaurante, diccionario_cocinas[tipo.strip()]))
        tipos_ideales = getattr(fila, "ideales", "")
        if pd.notna(tipos_ideales) and tipos_ideales:
            for tipo in str(tipos_ideales).split(";"):
                if tipo.strip() in diccionario_ideales:
                    relaciones_ideales.append((id_restaurante, diccionario_ideales[tipo.strip()]))

    # Insertar relaciones en la tabla de cocinas
    relaciones_trozo = 2000
    if relaciones_cocinas:
        for k in range(0, len(relaciones_cocinas), relaciones_trozo):
            trozo = relaciones_cocinas[k:k+relaciones_trozo]
            claves_valores = [f"(:restaurante_id_{i}, :cocina_id_{i})" for i in range(len(trozo))]
            parametros = {f"restaurante_id_{i}": relacion[0] for i, relacion in enumerate(trozo)}
            parametros.update({f"cocina_id_{i}": relacion[1] for i, relacion in enumerate(trozo)})
            consulta_inyeccion_relacion_cocina = text(f"""
                            INSERT INTO restaurantes_cocinas (restaurante_id, cocina_id)
                            VALUES {', '.join(claves_valores)}
                            ON CONFLICT DO NOTHING
                        """)
            conexion.execute(consulta_inyeccion_relacion_cocina, parametros)
    
    # Insertar relaciones en la tabla de ideales
    if relaciones_ideales:
        for k in range(0, len(relaciones_ideales), relaciones_trozo):
            trozo = relaciones_ideales[k:k+relaciones_trozo]
            claves_valores = [f"(:restaurante_id_{i}, :ideal_id_{i})" for i in range(len(trozo))]
            parametros = {f"restaurante_id_{i}": relacion[0] for i, relacion in enumerate(trozo)}
            parametros.update({f"ideal_id_{i}": relacion[1] for i, relacion in enumerate(trozo)})
            consulta_inyeccion_relacion_ideal = text(f"""
                            INSERT INTO restaurantes_ideales (restaurante_id, ideal_id)
                            VALUES {', '.join(claves_valores)}
                            ON CONFLICT DO NOTHING
                        """)
            conexion.execute(consulta_inyeccion_relacion_ideal, parametros)
    fin = time.time()
    print(f"Tiempo de ejecución para poblar la tabla de restaurantes y sus relaciones: {fin - inicio:.2f} segundos")

if __name__ == "__main__":
    csv_nombre = BASE_DIR / "datos" / "restaurantes_michelin_def.csv"
    df = procesar_csv(csv_nombre)

    inicio = time.time()
    with engine.begin() as conexion :
        print("Limpieza de tablas y reinicio de secuencias")
        conexion.execute(text("TRUNCATE TABLE restaurantes, cocinas, ideales, ciudades, paises RESTART IDENTITY CASCADE;"))
        diccionario_cocinas = poblar_tablas_externas(conexion , df, "cocinas", "tipos_cocina")
        diccionario_ideales = poblar_tablas_externas(conexion , df, "ideales", "ideales")
        ids_ciudades = poblar_ubicaciones(conexion, df)
        poblar_tabla_principal_y_relaciones(
            conexion, df, "restaurantes", "nombre", diccionario_cocinas,
            diccionario_ideales, ids_ciudades
        )
    fin = time.time()
    print(f"Tiempo de ejecución total: {fin - inicio:.2f} segundos")
    print("Datos cargados correctamente en la base de datos.")