from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="API Predictor Saber 11 y Saber Pro")

# --- 2. CONFIGURAR CORS (AÑADE ESTO) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Permite que cualquier sitio web acceda
    allow_credentials=True,
    allow_methods=["*"],      # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],      # Permite todos los encabezados
)

# --- CARGA DE MODELOS CON RUTA DINÁMICA ---
# Obtenemos la ruta de la carpeta raíz (un nivel arriba de /api)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

try:
    # Construimos la ruta completa al archivo
    path_saber11 = os.path.join(BASE_DIR, 'modelo_gbr_saber11.joblib')
    path_saberpro = os.path.join(BASE_DIR, 'modelo_gbr_saberpro.joblib')
    
    model_saber11 = joblib.load(path_saber11)
    model_saberpro = joblib.load(path_saberpro)
    print("Modelos cargados exitosamente")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    # Es recomendable que los modelos sean None si fallan para evitar crashes
    model_saber11 = None
    model_saberpro = None


# --- ESQUEMA PARA SABER 11 ---
class DatosSaber11(BaseModel):
    cole_caracter: str; cole_depto_ubicacion: str; cole_jornada: str; cole_mcpio_ubicacion: str; cole_naturaleza: str
    estu_dedicacioninternet: str; estu_dedicacionlecturadiaria: str; estu_depto_presentacion_saber11: str
    estu_depto_reside: str; estu_generacione: str; estu_genero_saber11: str; estu_mcpio_presentacion_saber11: str
    estu_mcpio_reside: str; fami_comecarnepescadohuevo: str; fami_comecerealfrutoslegumbre: str
    fami_comelechederivados: str; fami_cuartoshogar: str; fami_educacionmadre_saber11: str
    fami_educacionpadre_saber11: str; fami_estratovivienda_saber11: str; fami_numlibros: str
    fami_personashogar: str; fami_situacioneconomica: str; fami_tieneautomovil_saber11: str
    fami_tienecomputador_saber11: str; fami_tieneconsolavideojuegos_saber11: str
    fami_tienehornomicroogas_saber11: str; fami_tienemotocicleta_saber11: str
    fami_trabajolabormadre_saber11: str; fami_trabajolaborpadre_saber11: str
    fami_ocupacionmadre: str; fami_ocupacionpadre: str; periodo: int; edad_saber11: int
    # CORREGIDOS A STR:
    estu_nse_establecimiento: str
    estu_nse_individual_saber11: str
    estu_inse_individual_saber11: float # Si este también falla con "NSE", cámbialo a str
    estu_nse_ies: int
# --- ESQUEMA PARA SABER PRO ---
class DatosSaberPro(BaseModel):
    # --- Información del Colegio ---
    cole_caracter: str
    cole_depto_ubicacion: str
    cole_jornada: str
    cole_mcpio_ubicacion: str
    cole_naturaleza: str
    
    # --- DESEMPEÑOS (Corregidos a int según tu error) ---
    desemp_c_naturales: int
    desemp_ingles: str  # Si tu dato de inglés es "A1", cámbialo a str
    desemp_lectura_critica: int
    desemp_matematicas: int
    desemp_sociales_ciudadanas: int
    
    # --- Información Estudiante y Familia (Saber 11) ---
    estu_dedicacioninternet: str
    estu_dedicacionlecturadiaria: str
    estu_depto_presentacion_saber11: str
    estu_depto_reside: str
    estu_generacione: str
    estu_genero_saber11: str
    estu_inse_individual_saber11: float
    estu_mcpio_presentacion_saber11: str
    estu_mcpio_reside: str
    estu_nse_establecimiento: str  # Corregido a str
    estu_nse_individual_saber11: str # Corregido a str
    
    # --- Hábitos y Hogar ---
    fami_comecarnepescadohuevo: str
    fami_comecerealfrutoslegumbre: str
    fami_comelechederivados: str
    fami_cuartoshogar: str
    fami_educacionmadre_saber11: str
    fami_educacionpadre_saber11: str
    fami_estratovivienda_saber11: str
    fami_numlibros: str
    fami_personashogar: str
    fami_situacioneconomica: str
    fami_tieneautomovil_saber11: str
    fami_tienecomputador_saber11: str
    fami_tieneconsolavideojuegos_saber11: str
    fami_tienehornomicroogas_saber11: str
    fami_tienemotocicleta_saber11: str
    fami_trabajolabormadre_saber11: str
    fami_trabajolaborpadre_saber11: str
    
    # --- Percentiles y Puntajes Saber 11 (Numéricos) ---
    percentil_c_naturales: float
    percentil_global_saber11: float
    percentil_ingles: float
    percentil_lectura_critica: float
    percentil_matematicas: float
    percentil_sociales_ciudadanas: float
    punt_c_naturales: float
    punt_global_saber11: float
    punt_ingles: float
    punt_lectura_critica: float
    punt_matematicas: float
    punt_sociales_ciudadanas: float
    periodo: int
    
    # --- Información Específica Saber Pro ---
    estu_depto_presentacion_saberpro: str
    estu_genero_saberpro: str
    estu_horassemanatrabaja_saberpro: str
    estu_inse_individual_saberpro: float
    estu_inst_departamento: str
    estu_inst_municipio: str
    estu_mcpio_presentacion_saberpro: str
    estu_nse_ies: int  # Corregido a str
    estu_nse_individual_saberpro: str # Corregido a str
    estu_nucleo_pregrado: str
    estu_pagomatriculabeca: str
    estu_pagomatriculacredito: str
    estu_pagomatriculapadres: str
    estu_pagomatriculapropio: str
    estu_prgm_academico: str
    estu_prgm_departamento: str
    estu_prgm_municipio: str
    estu_semestrecursa: str
    estu_tituloobtenidobachiller: str
    fami_educacionmadre_saberpro: str
    fami_educacionpadre_saberpro: str
    fami_estratovivienda_saberpro: str
    fami_ocupacionmadre: str
    fami_ocupacionpadre: str
    fami_tieneautomovil_saberpro: str
    fami_tienehornomicroogas_saberpro: str
    fami_tienemotocicleta_saberpro: str
    fami_tieneserviciotv_saberpro: str
    fami_trabajolabormadre_saberpro: str
    fami_trabajolaborpadre_saberpro: str
    inst_caracter_academico: str
    inst_origen: str
    
    # --- Edades y Niveles ---
    edad_saber11: float # Usar float por si hay nulos convertidos
    edad_saberpro: float
    nivel_saber11: str


# --- ENDPOINTS ---

@app.post("/predict/saber11")
def predict_s11(data: DatosSaber11):
    try:
        df = pd.DataFrame([data.model_dump()])
        pred = model_saber11.predict(df)
        
        # CAMBIO AQUÍ: Usamos [0] para sacar el número del array de NumPy
        puntaje_final = float(pred[0]) 
        
        return {
            "examen": "saber11", 
            "prediccion_puntaje_global": round(puntaje_final, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/saberpro")
def predict_spro(data: DatosSaberPro):
    try:
        # 1. Convertimos los datos a DataFrame
        df = pd.DataFrame([data.model_dump()])
        
        # 2. Hacemos la predicción
        pred = model_saberpro.predict(df)
        
        # 3. CORRECCIÓN: Extraemos el valor escalar del array de NumPy
        # Usamos float(pred[0]) para asegurar que tome el primer elemento
        puntaje_final = float(pred[0]) 
        
        return {
            "examen": "saberpro",
            "prediccion_puntaje_global": round(puntaje_final, 2)
        }
    except Exception as e:
        # Esto te ayudará a ver qué pasa en la terminal si algo más falla
        print(f"Error detallado en Saber Pro: {e}")
        raise HTTPException(status_code=500, detail=str(e))


