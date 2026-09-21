import os
import joblib
import time
import math
import numpy as np
import functools
import pandas as pd

from pathlib import Path
from flask import Flask, render_template, request, jsonify, abort, make_response
from dotenv import load_dotenv
from supabase import create_client, Client
from sqlalchemy import text
from carga_csv import (
    engine, procesar_csv, poblar_tablas_externas, poblar_ubicaciones,
    poblar_tabla_principal_y_relaciones
)
env_ruta = Path(__file__).resolve().parent / ".env"
load_dotenv(env_ruta)

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

#configuración de supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

pipeline = joblib.load(BASE_DIR / 'modelo' / 'pipeline_xgboost_final.pkl')


def transformar_restaurante_nuevo(pais, tipos_cocina, ideales, precio, ciudad):
    """Aplica EXACTAMENTE la misma transformación que en entrenamiento --
    mismo orden, mismos encoders ya ajustados (.transform(), nunca
    .fit_transform() aquí)."""
    ohe = pipeline['ohe']
    mlb_tipos = pipeline['mlb_tipos']
    mlb_ideales = pipeline['mlb_ideales']
    densidad_ciudad_train = pipeline['densidad_ciudad_train']

    codificado_pais = ohe.transform(pd.DataFrame({'Pais': [pais]}))
    codificado_tipos = mlb_tipos.transform([tipos_cocina])
    codificado_ideales = mlb_ideales.transform([ideales])
    precio_array = np.array([[precio]])
    densidad = densidad_ciudad_train.get(ciudad, 0)  # ciudad nueva -> 0, mismo criterio que en entrenamiento
    densidad_array = np.array([[densidad]])

    return np.hstack([codificado_pais, codificado_tipos, codificado_ideales, precio_array, densidad_array])

def admin_requerido(vista):
    @functools.wraps(vista)
    def wrapper(*args, **kwargs):
        admin = request.authorization
        if not admin or admin.username != os.environ.get("ADMIN_USER") or admin.password != os.environ.get("ADMIN_PASSWORD"):
            respuesta = make_response(
                'Acceso no autorizado. Introduce las credenciales de administrador.',
                401
            )
            respuesta.headers['WWW-Authenticate'] = 'Basic realm="Administracion"'
            return respuesta
        return vista(*args, **kwargs)
    return wrapper


def normalizar_restaurante(restaurante):
    restaurante = dict(restaurante)
    ciudad = restaurante.pop('ciudades', None) or {}
    pais = ciudad.pop('paises', None) or {}
    restaurante['ciudad'] = ciudad.get('nombre', '')
    restaurante['pais'] = pais.get('nombre', '')
    return restaurante


@app.route('/')
def index():
    paises_response = supabase.table('paises').select('id,nombre').order('nombre').execute()
    cocinas_response = supabase.table('cocinas').select('nombre').order('nombre').execute()
    ideales_response = supabase.table('ideales').select('nombre').order('nombre').execute()

    paises = paises_response.data
    cocinas = [cocina['nombre'] for cocina in cocinas_response.data]
    ideales = [ideal['nombre'] for ideal in ideales_response.data]

    return render_template('index.html', paises=paises, cocinas=cocinas, ideales=ideales)

@app.route('/api/ciudades', methods=['GET'])
def get_ciudades():
    pais_id = request.args.get('pais_id', type=int)
    if pais_id is None:
        return jsonify({'mensaje': 'Debe indicar un pais_id.'}), 400

    response = supabase.table('ciudades').select('id,nombre').eq('pais_id', pais_id).order('nombre').execute()
    return jsonify(response.data)

@app.route('/admin', methods=['GET'])
@admin_requerido
def admin():
    return render_template('admin.html')

@app.route('/restaurantes', methods=['GET'])
def restaurantes():
    try:
        pagina = max(1, int(request.args.get('page', 1)))
    except (TypeError, ValueError):
        pagina = 1

    texto = request.args.get('texto', '').strip()
    pais_filtro = request.args.get('pais', '').strip()
    valoracion_filtro = request.args.get('valoracion', '').strip()
    tamaño_pagina = 12
    paises_response = supabase.table('paises').select('nombre').order('nombre').execute()

    ids_texto = None
    if texto:
        patron_texto = f'*{texto}*'
        nombres_response = supabase.table('restaurantes').select('id').ilike('nombre', patron_texto).execute()
        ciudades_response = supabase.table('ciudades').select('id').ilike('nombre', patron_texto).execute()
        ids_por_nombre = {restaurante['id'] for restaurante in nombres_response.data}
        ids_ciudades = [ciudad['id'] for ciudad in ciudades_response.data]
        ids_por_ciudad = set()
        if ids_ciudades:
            restaurantes_ciudad = supabase.table('restaurantes').select('id').in_('ciudad_id', ids_ciudades).execute()
            ids_por_ciudad = {restaurante['id'] for restaurante in restaurantes_ciudad.data}
        ids_texto = list(ids_por_nombre | ids_por_ciudad) or [-1]

    consulta_total = supabase.table('restaurantes').select(
        'id, ciudades!inner(nombre, paises!inner(nombre))', count='exact'
    )
    if ids_texto is not None:
        consulta_total = consulta_total.in_('id', ids_texto)
    if pais_filtro:
        consulta_total = consulta_total.eq('ciudades.paises.nombre', pais_filtro)
    if valoracion_filtro in {'0', '1', '2', '3'}:
        consulta_total = consulta_total.eq('valoracion', int(valoracion_filtro))

    total_response = consulta_total.limit(1).execute()
    total = total_response.count or 0
    total_paginas = max(1, math.ceil(total / tamaño_pagina))
    pagina = min(pagina, total_paginas)
    inicio = (pagina - 1) * tamaño_pagina
    fin = inicio + tamaño_pagina - 1
    consulta = supabase.table('restaurantes').select(
        '*, ciudades!inner(nombre, paises!inner(nombre)), restaurantes_cocinas(cocinas(nombre)), restaurantes_ideales(ideales(nombre))'
    )
    if ids_texto is not None:
        consulta = consulta.in_('id', ids_texto)
    if pais_filtro:
        consulta = consulta.eq('ciudades.paises.nombre', pais_filtro)
    if valoracion_filtro in {'0', '1', '2', '3'}:
        consulta = consulta.eq('valoracion', int(valoracion_filtro))
    response = consulta.order('nombre').range(inicio, fin).execute()

    precios = {
        0: '€/€€€€ (Ajustado)',
        1: '€/€€€€ (Moderado)',
        2: '€€€/€€€€ (Ocasión especial)',
        3: '€€€€/€€€€ (Sin reparar en gastos)'
    }
    restaurantes_presentacion = []
    for restaurante in response.data:
        restaurante = normalizar_restaurante(restaurante)
        precio = restaurante.get('precio')
        valoracion = restaurante.get('valoracion')
        try:
            precio = int(precio)
        except (TypeError, ValueError):
            precio = None
        try:
            valoracion = max(0, min(3, int(valoracion)))
        except (TypeError, ValueError):
            valoracion = 0

        restaurante['precio_texto'] = precios.get(precio, 'No disponible')
        restaurante['valoracion'] = valoracion
        restaurantes_presentacion.append(restaurante)

    return render_template(
        'restaurantes.html',
        restaurantes=restaurantes_presentacion,
        pagina=pagina,
        total_paginas=total_paginas,
        total=total,
        paginas=range(max(1, pagina - 2), min(total_paginas, pagina + 2) + 1),
        paises=[pais['nombre'] for pais in paises_response.data],
        texto=texto,
        pais_filtro=pais_filtro,
        valoracion_filtro=valoracion_filtro
    )

@app.route('/admin/cargar_csv', methods=['POST'])
@admin_requerido
def cargar_csv():
    csv = BASE_DIR / 'datos' / 'restaurantes_michelin_def.csv'
    df = procesar_csv(csv)

    inicio = time.time()
    try:
        with engine.begin() as conexion:
            conexion.execute(text("TRUNCATE TABLE restaurantes, cocinas, ideales, ciudades, paises RESTART IDENTITY CASCADE;"))
            diccionario_cocinas = poblar_tablas_externas(conexion, df, "cocinas", "tipos_cocina")
            diccionario_ideales = poblar_tablas_externas(conexion, df, "ideales", "ideales")
            ids_ciudades = poblar_ubicaciones(conexion, df)
            poblar_tabla_principal_y_relaciones(
                conexion, df, "restaurantes", "nombre", diccionario_cocinas,
                diccionario_ideales, ids_ciudades
            )
    except Exception as error:
        app.logger.exception("Error al cargar el CSV en la base de datos")
        return jsonify({'mensaje': f'No se pudo cargar el CSV: {error}'}), 500
    fin = time.time()

    return  jsonify({'mensaje': f'Datos cargados correctamente en {fin - inicio:.2f} segundos'})

@app.route('/api/prediccion', methods=['POST'])
def prediccion():
    ciudad = request.form.get('ciudad', '').strip()
    pais = request.form.get('pais', '').strip()
    pais_id = request.form.get('pais_id', type=int)
    ciudad_id = request.form.get('ciudad_id', type=int)
    if not pais_id or not ciudad_id:
        abort(400, description='Debe seleccionar un país y una ciudad.')

    ubicacion = supabase.table('ciudades').select('nombre, paises!inner(nombre)').eq(
        'id', ciudad_id
    ).eq('pais_id', pais_id).single().execute()
    if not ubicacion.data or ubicacion.data['nombre'] != ciudad or ubicacion.data['paises']['nombre'] != pais:
        abort(400, description='La ciudad seleccionada no pertenece al país indicado.')
    precio_recibido = request.form.get('precio')
    try:
        precio = int(precio_recibido)
    except (TypeError, ValueError):
        abort(400, description='El precio debe ser un número entero entre 0 y 3.')
    if precio not in range(4):
        abort(400, description='El precio debe estar entre 0 y 3.')

    tipos_cocina = request.form.getlist('tipos_cocina')
    ideales = request.form.getlist('ideales')

    if not tipos_cocina:
        abort(400, description='Debe seleccionar al menos un tipo de cocina.')

    # Transformar el restaurante nuevo
    nuevo_vector = transformar_restaurante_nuevo(pais, tipos_cocina, ideales, precio, ciudad)

    # Realizar la predicción
    prediccion = int(pipeline['modelo'].predict(nuevo_vector)[0])

    print(f"Predicción para el restaurante en {ciudad}, {pais} con precio {precio}, tipos de cocina {tipos_cocina} e ideales {ideales}: {prediccion}")

    return render_template(
        'prediccion.html', 
        ciudad=ciudad, pais=pais, precio=precio, tipos_cocina=tipos_cocina, ideales=ideales, prediccion=prediccion)
if __name__ == '__main__':
    app.run(use_reloader=False)