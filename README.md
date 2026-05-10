# 🎓 EduPredictor.ai - API REST con Flask

Herramienta avanzada de analítica predictiva para estudiantes e instituciones colombianas que utiliza modelos de Machine Learning para predecir rendimiento académico en pruebas Saber 11 y Saber Pro.

## 📋 Contenido del Proyecto

```
Proyecto-de-Analitica/
├── app.py                                 # API Flask principal
├── edupredictor_backend.py               # Backend sin API (solo predicciones)
├── requirements.txt                       # Dependencias Python
├── ESTRUCTURA_DATOS_EDUPREDICTOR.md     # Documentación de estructura de datos
├── best_model_saber11_classification.joblib
├── best_model_saber11_regression.joblib
├── best_model_saberpro_classification.joblib
├── best_model_saberpro_regression.joblib
├── templates/
│   └── index.html                        # Interfaz principal (Stitch design)
├── static/
│   ├── css/
│   │   └── style.css                     # Estilos profesionales
│   └── js/
│       └── script.js                     # Lógica frontend
└── uploads/                               # Carpeta para archivos cargados
```

## 🚀 Instalación Rápida

### 1. Clonar/Descargar el Proyecto
```bash
cd "c:\Users\nesto\OneDrive - Universidad de la Sabana\Universidad\Cata\Proyecto-de-Analitica"
```

### 2. Crear Entorno Virtual (Recomendado)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la API
```bash
python app.py
```

La aplicación estará disponible en: **http://localhost:5000**

## 🐳 Ejecutar con Docker

### Opción 1: Docker Compose
```bash
docker compose up --build
```

La aplicación quedará disponible en: **http://localhost:5000**

Para detenerla:
```bash
docker compose down
```

### Opción 2: Docker directo
```bash
docker build -t edupredictor .
docker run --rm -p 5000:5000 -v "$(pwd)/uploads:/app/uploads" edupredictor
```

En PowerShell:
```powershell
docker run --rm -p 5000:5000 -v "${PWD}/uploads:/app/uploads" edupredictor
```

## 🎯 Características Principales

### 1. **Interfaz Intuitiva**
- Diseño profesional basado en Stitch
- Navegación sin recargas de página
- Responsive (móvil, tablet, desktop)
- Badges SABER11 / SABERPRO

### 2. **Dos Flujos de Predicción**

#### **Uso Institucional**
- Carga masiva de datos (CSV/Excel)
- Procesamiento de múltiples estudiantes
- Descarga de plantillas de ejemplo
- Exportación de resultados

#### **Uso Individual**
- Formularios personalizados por tipo de examen
- Predicción inmediata
- Descarga de reportes

### 3. **Modelos de ML**
- **Saber 11** (Bachillerato)
  - 8 variables de entrada
  - Clasificación + Regresión
  
- **Saber Pro** (Universitario)
  - 11 variables de entrada (incluye campos adicionales)
  - Clasificación + Regresión

## 📊 Endpoints de la API

### Predicción Individual

**Saber 11**
```bash
POST /api/predict/saber11
Content-Type: application/json

{
    "estrato": 2,
    "educacion_madre": "Secundaria",
    "educacion_padre": "Primaria",
    "internet": "Sí",
    "consumo_proteina": "Diariamente",
    "puntaje_matematicas": 65,
    "puntaje_lectura": 70,
    "edad": 17
}
```

**Saber Pro**
```bash
POST /api/predict/saberpro
Content-Type: application/json

{
    "estrato": 3,
    "educacion_madre": "Universitaria",
    "educacion_padre": "Secundaria",
    "internet": "Sí",
    "consumo_proteina": "Regularmente",
    "puntaje_matematicas": 78,
    "puntaje_lectura": 82,
    "edad": 21,
    "caracter_institucion": "UNIVERSIDAD",
    "semestre_actual": 6,
    "horas_trabajo_semanal": "20"
}
```

### Predicción en Lote

```bash
POST /api/predict/bulk
Content-Type: multipart/form-data

Form Data:
- file: [archivo CSV o Excel]
- exam_type: saber11|saberpro
```

### Plantilla de Ejemplo

```bash
GET /api/template/saber11
GET /api/template/saberpro
```

### Health Check

```bash
GET /api/health
```

Respuesta:
```json
{
    "status": "ok",
    "modelos_cargados": 2,
    "total_modelos_esperados": 2
}
```

## 📁 Estructura de Datos

### Archivo CSV para Saber 11
```
Nombre_Estudiante,Estrato,Educacion_Madre,Educacion_Padre,Internet,Consumo_Proteina,Edad
Juan Pérez,2,Secundaria,Primaria,Entre 1 y 3 horas,Diariamente,17
Maria Lopez,3,Universitaria,Secundaria,Más de 3 horas,Regularmente,17
```

### Archivo CSV para Saber Pro
```
Nombre_Estudiante,Estrato,Educacion_Madre,Educacion_Padre,Internet,Consumo_Proteina,Matematicas_Prev,Lectura_Prev,Edad,Caracter_Institucion,Semestre_Actual,Horas_Trabajo_Semanal
Juan Pérez,2,Secundaria,Primaria,Sí,Diariamente,65,70,22,TECNOLÓGICA,6,0
Maria Lopez,3,Universitaria,Secundaria,Sí,Regularmente,78,82,21,UNIVERSIDAD,5,20
```

## 🎨 Diseño Visual

El frontend utiliza:
- **Colores**: Azul (#3B82F6), Cian (#06B6D4)
- **Tipografía**: Inter (sans-serif)
- **Fondo**: Gris claro (bg-slate-50)
- **Radio de bordes**: 8px
- **Sombras**: Suave y profesional

## 📈 Flujo de Navegación

```
Home
├── Uso Institucional
│   ├── Selección Examen
│   ├── Carga Masiva
│   └── Resultados
└── Uso Individual
    ├── Selección Examen
    ├── Formulario (Saber 11 u 8 campos / Saber Pro u 11 campos)
    └── Predicción Individual
```

## 🔧 Configuración Avanzada

### Variables de Entrada Válidas

**Educación:**
- Ninguno, Primaria, Secundaria, Universitaria, Posgrado

**Internet:**
- No Navega Internet, Menos de 1 hora, 1-2 horas, 2-4 horas, Más de 4 horas

**Consumo Proteína:**
- Nunca o rara vez comemos eso, A veces, Regularmente, Frecuentemente, Diariamente

**Institución (SaberPro):**
- TÉCNICA PROFESIONAL, TECNOLÓGICA, UNIVERSIDAD

**Horas Trabajo (SaberPro):**
- 0, 1-10, 11-20, 21-30, 31-40, Más de 40

## 📝 Ejemplos de Uso

### Python - Predicción Individual
```python
import requests

url = "http://localhost:5000/api/predict/saber11"
data = {
    "estrato": 2,
    "educacion_madre": "Secundaria",
    "educacion_padre": "Primaria",
    "internet": "Sí",
    "consumo_proteina": "Diariamente",
    "puntaje_matematicas": 65,
    "puntaje_lectura": 70,
    "edad": 17
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL - Predicción Individual
```bash
curl -X POST http://localhost:5000/api/predict/saber11 \
  -H "Content-Type: application/json" \
  -d '{
    "estrato": 2,
    "educacion_madre": "Secundaria",
    "educacion_padre": "Primaria",
    "internet": "Sí",
    "consumo_proteina": "Diariamente",
    "puntaje_matematicas": 65,
    "puntaje_lectura": 70,
    "edad": 17
  }'
```

### JavaScript - Predicción Individual
```javascript
const data = {
    estrato: 2,
    educacion_madre: "Secundaria",
    educacion_padre: "Primaria",
    internet: "Sí",
    consumo_proteina: "Diariamente",
    puntaje_matematicas: 65,
    puntaje_lectura: 70,
    edad: 17
};

fetch('http://localhost:5000/api/predict/saber11', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
})
.then(r => r.json())
.then(result => console.log(result));
```

## 🐛 Troubleshooting

### Error: "Modelos no encontrados"
```bash
# Asegúrate de que los archivos .joblib estén en el directorio raíz
ls -la best_model_*.joblib
```

### Error: "Puerto 5000 en uso"
```bash
# Cambiar puerto en app.py
app.run(debug=True, port=8000)  # Usar otro puerto

# O terminar proceso actual
lsof -ti:5000 | xargs kill -9  # macOS/Linux
```

### Error: "Módulo no encontrado"
```bash
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

## 📊 Respuesta de Predicción

```json
{
    "success": true,
    "puntaje_estimado": 420,
    "riesgo": "BAJO",
    "color_riesgo": "green",
    "confianza_modelo": 0.87,
    "percentil": 65,
    "desempenio_esperado": "Desempeño Promedio",
    "color_desempenio": "blue"
}
```

## 🔐 Seguridad

- Validación de tipos de archivo (CSV, Excel)
- Límite de tamaño de archivo: 10MB
- Sanitización de nombres de archivo
- CORS habilitado (configurable)
- Manejo de errores robusto

## 📱 Compatibilidad

- ✅ Chrome, Firefox, Safari, Edge (últimas versiones)
- ✅ Responsive: Mobile, Tablet, Desktop
- ✅ Python 3.8+
- ✅ Windows, macOS, Linux

## 📞 Soporte

Para reportar errores o sugerencias, contacta con el equipo de desarrollo.

## 📄 Licencia

Proyecto de Analítica - Universidad de la Sabana (2026)

---

**Desarrollado con ❤️ para mejorar la educación en Colombia**
