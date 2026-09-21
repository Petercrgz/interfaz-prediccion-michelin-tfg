from matplotlib import pyplot
import pandas as pd
import numpy as np
import time
import shap
import joblib
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def calcular_pesos_suavizados(objetivo, intensidad=0.5):
    """Calcula pesos de muestra entre 'sin balancear' (intensidad=0, todos
    pesan 1) y 'balanced completo' (intensidad=1, el peso normal de
    compute_sample_weight). intensidad=0.5 es la raíz cuadrada del peso
    balanced -- un punto intermedio que reequilibra las clases minoritarias
    sin llegar a la corrección completa, que puede ser demasiado agresiva.

    Se renormaliza para que la media de los pesos siga siendo 1, igual que
    hace compute_sample_weight -- así el único efecto real de 'intensidad'
    es cuánto se reequilibra entre clases, no la intensidad general del
    entrenamiento.
    """
    pesos_balanceados = compute_sample_weight(class_weight='balanced', y=objetivo)
    pesos = pesos_balanceados ** intensidad
    pesos = pesos * (len(pesos) / pesos.sum())
    return pesos


# === Configuración final de XGBoost, ya verificada en el proceso de ajuste ===
# (búsqueda por etapas + comprobación de que ningún valor quedaba pegado al
# borde de la rejilla explorada). Ya NO se vuelve a ejecutar GridSearchCV
# ni el bucle de tasas de aprendizaje: son costosos (la última pasada tardó
# 147 minutos) y ya no aportan nada nuevo -- se usan directamente.
parametros = {
    'colsample_bytree': 0.8,
    'max_depth': 3,
    'min_child_weight': 0,
    'subsample': 0.6,
    'learning_rate': 0.05,
    'n_estimators': 263,
    'random_state': 42,
}


def modelo_xgBoost(nombre_csv):
    inicio = time.perf_counter()
    ruta_csv = Path(nombre_csv)
    carpeta_archivos_modelo = Path(__file__).resolve().parent.parent / 'archivos_modelo'
    carpeta_archivos_modelo.mkdir(parents=True, exist_ok=True)

    restaurantes = pd.read_csv(ruta_csv)
    nombre_csv_base = ruta_csv.stem

    total = len(restaurantes)
    print(f"\nTotal de restaurantes: {total}")

    print("Porcentaje de cada restaurante por valoración:")
    for val in range(4):
        count = len(restaurantes[restaurantes['Valoracion'] == val])
        print(f"Valoración {val}: {count} ({count/total*100:.2f}%)")

    semilla = 42

    atributos_categoricos = ['Pais', 'Tipos_Cocina', 'Ideales']
    atributos_continuos_base = ['Precio']

    atributos = restaurantes.loc[:, atributos_categoricos + atributos_continuos_base + ['Ciudad']]
    print(atributos.head())

    objetivo = restaurantes['Valoracion']
    print(objetivo.head())

    # Separar datos en entrenamiento y prueba ANTES de codificar
    atributos_entrenamiento_raw, atributos_prueba_raw, objetivo_entrenamiento, objetivo_prueba = train_test_split(
        atributos, objetivo,
        test_size=0.2,
        stratify=objetivo,
        random_state=semilla
    )

    # Densidad de restaurantes por ciudad, calculada SOLO con entrenamiento
    densidad_ciudad_train = (
        atributos_entrenamiento_raw.groupby('Ciudad').size()
        .rename('restaurantes_en_ciudad')
    )
    atributos_entrenamiento_raw = atributos_entrenamiento_raw.copy()
    atributos_prueba_raw = atributos_prueba_raw.copy()
    atributos_entrenamiento_raw['restaurantes_en_ciudad'] = (
        atributos_entrenamiento_raw['Ciudad'].map(densidad_ciudad_train).fillna(0)
    )
    atributos_prueba_raw['restaurantes_en_ciudad'] = (
        atributos_prueba_raw['Ciudad'].map(densidad_ciudad_train).fillna(0)
    )

    mlb_tipos = MultiLabelBinarizer()
    mlb_ideales = MultiLabelBinarizer()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def codificar_mlb(serie):
        return serie.fillna('').str.split(';').apply(lambda x: [item.strip() for item in x if item.strip()])

    # Encoders ajustados SOLO con el bloque de entrenamiento
    codificado_tipos_train = mlb_tipos.fit_transform(codificar_mlb(atributos_entrenamiento_raw['Tipos_Cocina']))
    codificado_ideales_train = mlb_ideales.fit_transform(codificar_mlb(atributos_entrenamiento_raw['Ideales']))
    codificado_nombre_train = ohe.fit_transform(atributos_entrenamiento_raw[['Pais']])

    precio_train = atributos_entrenamiento_raw[['Precio']].values
    densidad_train = atributos_entrenamiento_raw[['restaurantes_en_ciudad']].values

    atributos_entrenamiento = np.hstack([
        codificado_nombre_train, codificado_tipos_train, codificado_ideales_train,
        precio_train, densidad_train
    ])

    codificado_tipos_test = mlb_tipos.transform(codificar_mlb(atributos_prueba_raw['Tipos_Cocina']))
    codificado_ideales_test = mlb_ideales.transform(codificar_mlb(atributos_prueba_raw['Ideales']))
    codificado_nombre_test = ohe.transform(atributos_prueba_raw[['Pais']])

    precio_test = atributos_prueba_raw[['Precio']].values
    densidad_test = atributos_prueba_raw[['restaurantes_en_ciudad']].values

    atributos_prueba = np.hstack([
        codificado_nombre_test, codificado_tipos_test, codificado_ideales_test,
        precio_test, densidad_test
    ])

    nombres_caracteristicas = (
        list(ohe.get_feature_names_out(['Pais']))
        + list(mlb_tipos.classes_)
        + list(mlb_ideales.classes_)
        + ['Precio', 'restaurantes_en_ciudad']
    )

    assert len(nombres_caracteristicas) == atributos_entrenamiento.shape[1], (
        f"|N caracteristicas|={len(nombres_caracteristicas)} != "
        f"|Columnas en matriz|={atributos_entrenamiento.shape[1]}"
    )

    peso_muestra = calcular_pesos_suavizados(objetivo_entrenamiento)

    # === Entrenamiento directo con la configuración ya verificada ===
    # Nada de GridSearchCV ni de bucle de tasas de aprendizaje aquí: ya se
    # hizo esa búsqueda y se confirmó esta configuración como la definitiva.
    parametros = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    clasificador = XGBClassifier(**parametros)
    clasificador.fit(atributos_entrenamiento, objetivo_entrenamiento, sample_weight=peso_muestra)

    predicciones = clasificador.predict(atributos_prueba)

    print("Matriz de confusión:")
    print(confusion_matrix(objetivo_prueba, predicciones))
    print("\nReporte de clasificación:")
    print(classification_report(objetivo_prueba, predicciones))

    # === Importancia de variables (Gini/MDI) ===
    importancias = pd.Series(clasificador.feature_importances_, index=nombres_caracteristicas)
    importancias = importancias[importancias > 0].sort_values(ascending=False).head(20)

    pyplot.figure(figsize=(10, 8))
    importancias.sort_values().plot(kind='barh')
    pyplot.xlabel('Importancia (reducción de impureza)')
    pyplot.title(f'Top 20 variables más importantes -- {nombre_csv_base}')
    pyplot.tight_layout()
    pyplot.savefig(carpeta_archivos_modelo / f"importancias_{nombre_csv_base}_xgBoost_final.png",
                   dpi=300, bbox_inches='tight')
    pyplot.close()
    print("Importancia de variables guardada.")

    # === SHAP: una única vez, reutilizado para tabla y las dos figuras ===
    explainer_xgboost = shap.TreeExplainer(clasificador)
    shap_values = explainer_xgboost(atributos_prueba)

    medias_por_clase = np.mean(np.abs(shap_values.values), axis=0)  # (variables, clases)
    tabla_shap = pd.DataFrame(
        medias_por_clase,
        index=nombres_caracteristicas,
        columns=[f'Clase {c}' for c in clasificador.classes_]
    )
    tabla_shap['Total'] = tabla_shap.sum(axis=1)
    tabla_shap = tabla_shap.sort_values('Total', ascending=False)

    print(tabla_shap.head(15).round(4))
    tabla_shap.head(15).round(4).to_csv(
        carpeta_archivos_modelo / f"tabla_shap_{nombre_csv_base}_xgBoost_final.csv"
    )
    print(tabla_shap.head(15).round(4).to_latex(float_format='%.4f'))

    # Figura 1: contexto completo (Precio incluido)
    shap.summary_plot(shap_values, atributos_prueba, feature_names=nombres_caracteristicas, show=False)
    pyplot.savefig(carpeta_archivos_modelo / 'shap_xgboost.png', dpi=300, bbox_inches='tight')
    pyplot.close()

    # Figura 2: sin Precio, para ver bien el resto de variables
    mascara_sin_precio = [i for i in range(len(nombres_caracteristicas)) if i != nombres_caracteristicas.index('Precio')]
    shap.summary_plot(
        shap_values[:, mascara_sin_precio],
        atributos_prueba[:, mascara_sin_precio],
        feature_names=[nombres_caracteristicas[i] for i in mascara_sin_precio],
        show=False
    )
    pyplot.savefig(carpeta_archivos_modelo / 'shap_xgboost_sin_precio.png', dpi=300, bbox_inches='tight')
    pyplot.close()
    print("Gráficas SHAP guardadas.")

    # === Guardado del pipeline completo para la aplicación ===
    # Todo lo necesario para predecir un restaurante nuevo, no solo el
    # modelo: sin los encoders no hay forma de transformar datos nuevos al
    # mismo formato numérico que el modelo espera.
    ruta_pipeline = Path(__file__).resolve().parent / 'pipeline_xgboost_final.pkl'
    joblib.dump({
        'modelo': clasificador,
        'ohe': ohe,
        'mlb_tipos': mlb_tipos,
        'mlb_ideales': mlb_ideales,
        'densidad_ciudad_train': densidad_ciudad_train,
        'nombres_caracteristicas': nombres_caracteristicas,
    }, ruta_pipeline)
    print(f"Pipeline completo guardado en {ruta_pipeline}")

    tiempo_csv = time.perf_counter() - inicio
    print(f"Tiempo total para procesar {nombre_csv}: {tiempo_csv:.2f} segundos "
          f"({tiempo_csv / 60:.2f} minutos)")


if __name__ == "__main__":
    nombre_csv = Path(__file__).resolve().parent.parent / 'datos' / 'restaurantes_michelin_def.csv'
    modelo_xgBoost(nombre_csv)