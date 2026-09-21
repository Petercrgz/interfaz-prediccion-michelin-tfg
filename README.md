# Interfaz de prediccion Michelin

Aplicacion web desarrollada con Flask para consultar restaurantes Michelin y predecir la valoracion de un restaurante a partir de su ubicacion, precio, tipos de cocina e ideales.

## Despliegue

La aplicacion esta disponible en:

<https://interfaz-prediccion-michelin-tfg.onrender.com/>

## Requisitos

- Python 3.10 o superior.
- Un proyecto de Supabase configurado, que proporcione una base de datos PostgreSQL accesible y su API REST.
- El archivo `modelo/pipeline_xgboost_final.pkl`.

## Instalacion local

1. Clona el repositorio y entra en su directorio:

   ```bash
   git clone https://github.com/Petercrgz/interfaz-prediccion-michelin-tfg.git
   ```

2. Crea y activa un entorno virtual:

   En Windows PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   En macOS o Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Variables de entorno

Copia `.env.example` como `.env` y completa los valores correspondientes:

```bash
cp .env.example .env
```

En Windows PowerShell, puedes usar:

```powershell
Copy-Item .env.example .env
```

El archivo `.env` debe contener:

```dotenv
DATABASE_URL=postgresql+psycopg2://usuario:contraseña@host:puerto/postgres
SUPABASE_URL=https://id_proyecto.supabase.co
SUPABASE_KEY=clave_secreta_anon
ADMIN_USER=admin
ADMIN_PASSWORD=contraseña_personalizable
```

No subas el archivo `.env` al repositorio, ya que contiene credenciales de acceso.

## Ejecucion

Con el entorno virtual activo y las variables configuradas, inicia la aplicacion con:

```bash
python app.py
```

Abre despues <http://127.0.0.1:5000/> en el navegador.

## Funcionalidades principales

### Prediccion

- Seleccionar un pais y una ciudad del país seleccionado.
- Elegir el precio del restaurante.
- Seleccionar uno o varios tipos de cocina.
- Seleccionar los ideales del restaurante.
- Obtener una prediccion de la valoracion mediante el modelo XGBoost guardado en `modelo/pipeline_xgboost_final.pkl`.

### Listado de restaurantes

- Consultar los restaurantes Michelin almacenados en la base de datos.
- Buscar restaurantes por nombre o ciudad.
- Filtrar los resultados por pais y valoracion.
- Navegar por los resultados mediante paginacion.
- Consultar informacion como la direccion, el precio, la valoracion, los tipos de cocina y los ideales.

### Administracion

- Acceder al panel de administracion protegido mediante autenticacion basica en `/admin`.
- Cargar el archivo `datos/restaurantes_michelin_def.csv` en la base de datos.
- Actualizar los restaurantes, paises, ciudades, tipos de cocina e ideales almacenados.

La carga del CSV reemplaza los datos de las tablas principales y sus relaciones, por lo que debe utilizarse con precaucion.

## Tareas y operaciones adicionales por consola (CLI)

Estas operaciones deben ejecutarse desde la carpeta raiz del proyecto, con el entorno virtual activo.

### Entrenamiento del modelo

El script `modelo/modeloXGBoost_4.py` lee el archivo `datos/restaurantes_michelin_def.csv`, entrena un clasificador XGBoost y evalua sus resultados.

Para ejecutarlo:

```bash
python modelo/modeloXGBoost_4.py
```

Durante la ejecucion se muestran por consola:

- El numero total de restaurantes y el porcentaje de cada valoracion.
- Una muestra de los atributos y del objetivo.
- La matriz de confusion.
- El reporte de clasificacion con precision, exhaustividad y puntuacion F1.
- La tabla con las variables mas relevantes segun SHAP.
- El tiempo total de procesamiento.

Ademas, el script genera en `archivos_modelo/`:

- `importancias_restaurantes_michelin_def_xgBoost_final.png`: grafica de importancia de las variables.
- `tabla_shap_restaurantes_michelin_def_xgBoost_final.csv`: tabla con las 15 variables mas relevantes segun SHAP.
- `shap_xgboost.png`: grafica SHAP con todas las variables.
- `shap_xgboost_sin_precio.png`: grafica SHAP excluyendo la variable precio.

Por ultimo, guarda o actualiza `modelo/pipeline_xgboost_final.pkl`, que contiene el modelo entrenado, los codificadores y la informacion necesaria para realizar predicciones desde la aplicacion Flask.

### Carga del CSV en la base de datos

El script `carga_csv.py` procesa `datos/restaurantes_michelin_def.csv` y carga sus datos en las tablas de Supabase mediante la conexion PostgreSQL definida en `DATABASE_URL`.

Para ejecutarlo:

```bash
python carga_csv.py
```

El script:

- Limpia las tablas `restaurantes`, `cocinas`, `ideales`, `ciudades` y `paises`.
- Reinicia sus secuencias y vuelve a insertar los datos del CSV.
- Crea las relaciones entre restaurantes, tipos de cocina e ideales.
- Muestra por consola el tiempo empleado y el numero de restaurantes insertados.

Antes de ejecutarlo, comprueba que `.env` contiene una `DATABASE_URL` valida y que la base de datos incluye las tablas esperadas. Esta operacion sustituye los datos existentes.

## Estructura del proyecto

```text
interfaz-prediccion-michelin-tfg/
|-- app.py                     Aplicacion Flask y rutas web/API
|-- carga_csv.py               Procesamiento y carga del CSV
|-- requirements.txt           Dependencias de Python
|-- .env.example               Plantilla de configuracion
|-- .gitignore                 Archivos y carpetas excluidos de Git
|-- archivos_modelo/           Directorio local de salida para graficas y tablas (ignorado en Git)
|-- datos/
|   `-- restaurantes_michelin_def.csv
|-- modelo/
|   |-- modeloXGBoost_4.py     Codigo de entrenamiento o analisis
|   `-- pipeline_xgboost_final.pkl
|-- static/                    Archivos estaticos
|   `-- img/                   Imagenes de la aplicacion
`-- templates/                 Plantillas HTML Jinja2
    |-- _footer.html
    |-- _header.html
    |-- admin.html
    |-- index.html
    |-- prediccion.html
    `-- restaurantes.html
```