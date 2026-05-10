# EduPredictor.ai - Estructura de Datos

## Flujo de Aplicación

### 1. Pantalla de Inicio (Home)
- Dos opciones principales: "Uso Institucional" y "Uso Individual"
- Navegación sin recarga de página

### 2. Flujo: Uso Institucional

#### 2.1 Selección de Examen
- Opción 1: **Saber 11** (Bachillerato)
- Opción 2: **Saber Pro** (Universitario)

#### 2.2 Carga Masiva de Datos
- Formato esperado: **CSV o XLSX**
- Área de drag & drop
- Plantilla descargable con columnas requeridas

### 3. Flujo: Uso Individual

#### 3.1 Selección de Examen
- Opción 1: **Saber 11** (Bachillerato)
- Opción 2: **Saber Pro** (Universitario)

#### 3.2 Formulario Personalizado
- Formularios específicos según examen elegido
- Validación de entrada en tiempo real
- Predicción inmediata

---

## Estructura de Archivos CSV para Carga Masiva

### Para Saber 11

```
Nombre_Estudiante,Estrato,Educacion_Madre,Educacion_Padre,Internet,Consumo_Proteina,Matematicas_Prev,Lectura_Prev,Edad
Juan Pérez,2,Secundaria,Primaria,Sí,Diariamente,65,70,17
Maria Lopez,3,Universitaria,Secundaria,Sí,Regularmente,78,82,17
Carlos Rodríguez,1,Primaria,Primaria,No,A veces,45,50,18
Ana García,4,Posgrado,Universitaria,Sí,Diariamente,89,92,17
```

#### Descripción de Columnas:

| Columna | Tipo | Valores Posibles | Descripción |
|---------|------|-----------------|-------------|
| **Nombre_Estudiante** | Texto | Cualquier nombre | Identificador del estudiante |
| **Estrato** | Numérico | 1-6, "Sin Estrato" | Estrato socioeconómico |
| **Educacion_Madre** | Categórico | Ninguno, Primaria, Secundaria, Universitaria, Posgrado | Nivel educativo de la madre |
| **Educacion_Padre** | Categórico | Ninguno, Primaria, Secundaria, Universitaria, Posgrado | Nivel educativo del padre |
| **Internet** | Categórico | Sí, No | Acceso a internet |
| **Consumo_Proteina** | Categórico | Nunca o rara vez, A veces, Regularmente, Frecuentemente, Diariamente | Frecuencia de consumo de carne/huevo |
| **Matematicas_Prev** | Numérico | 0-100 | Puntaje previo en matemáticas |
| **Lectura_Prev** | Numérico | 0-100 | Puntaje previo en lectura crítica |
| **Edad** | Numérico | 14-25 | Edad del estudiante |

---

### Para Saber Pro

```
Nombre_Estudiante,Estrato,Educacion_Madre,Educacion_Padre,Internet,Consumo_Proteina,Matematicas_Prev,Lectura_Prev,Edad,Caracter_Institucion,Semestre_Actual,Horas_Trabajo_Semanal
Juan Pérez,2,Secundaria,Primaria,Sí,Diariamente,65,70,22,TECNOLÓGICA,06,0
Maria Lopez,3,Universitaria,Secundaria,Sí,Regularmente,78,82,21,UNIVERSIDAD,05,20
```

#### Descripción de Columnas Adicionales (Saber Pro):

| Columna | Tipo | Valores Posibles | Descripción |
|---------|------|-----------------|-------------|
| **Caracter_Institucion** | Categórico | TÉCNICA PROFESIONAL, TECNOLÓGICA, UNIVERSIDAD | Tipo de institución de educación superior |
| **Semestre_Actual** | Numérico | 01-12 | Semestre actual del estudiante |
| **Horas_Trabajo_Semanal** | Categórico | 0, 1-10, 11-20, 21-30, 31-40, Más de 40 | Horas de trabajo semanal |

---

## Campos del Formulario Individual

### Saber 11 (8 campos)
1. **Estrato Socioeconómico** (dropdown): Sin Estrato, 1-6
2. **Nivel Educativo Madre** (dropdown): Ninguno, Primaria, Secundaria, Universitaria, Posgrado
3. **Nivel Educativo Padre** (dropdown): Ninguno, Primaria, Secundaria, Universitaria, Posgrado
4. **Dedicación Internet** (dropdown): No Navega, <1h, 1-2h, 2-4h, >4h
5. **Consumo Carne/Huevo** (dropdown): Nunca/Rara vez, A veces, Regularmente, Frecuentemente, Diariamente
6. **Puntaje Matemáticas** (número): 0-100
7. **Puntaje Lectura Crítica** (número): 0-100
8. **Edad** (número): 14-25

### Saber Pro (11 campos)
- Campos 1-8 (igual a Saber 11)
- 9. **Carácter Institución** (dropdown): TÉCNICA PROFESIONAL, TECNOLÓGICA, UNIVERSIDAD
- 10. **Semestre Actual** (dropdown): 01-12
- 11. **Horas Trabajo Semanal** (dropdown): 0, 1-10, 11-20, 21-30, 31-40, >40

---

## Salida de Predicción

### Resultado Individual
```json
{
  "puntaje_estimado": 420,
  "riesgo": "BAJO",
  "confianza_modelo": 0.87,
  "percentil": 65,
  "desempenio_esperado": "Desempeño Académico Promedio"
}
```

### Resultado Bulk (Tabla)
| Nombre Estudiante | Puntaje Predicho | Riesgo | Desempeño Esperado |
|-------------------|------------------|--------|-------------------|
| Juan Pérez | 420 | BAJO | Desempeño Promedio |
| Maria Lopez | 385 | ALTO | Riesgo de Deserción |
| Carlos Rodríguez | 380 | ALTO | Riesgo de Deserción |
| Ana García | 520 | BAJO | Desempeño Sobresaliente |

---

## Modelos Disponibles

### Saber 11
- **Clasificación**: `best_model_saber11_classification.joblib` (XGBoost)
- **Regresión**: Puntaje predicho

### Saber Pro
- **Clasificación**: `best_model_saberpro_classification.joblib` (XGBoost)
- **Regresión**: `best_model_saberpro_regression.joblib` (Ridge - CUML)

---

## Notas Técnicas

- Los modelos fueron entrenados con GPU (CuML, XGBoost GPU)
- Requiere preprocesamiento de variables categóricas
- Escalado de variables numéricas
- Validación de rangos y tipos de datos

---

## Pantallas del Prototipo en Stitch

1. **Home** - Selección inicial (Institucional/Individual)
2. **Exam Selection (Institucional)** - Elegir Saber 11 o Saber Pro
3. **Bulk Upload** - Carga de archivos CSV/XLSX con plantilla
4. **Bulk Results** - Tabla de resultados con predicciones
5. **Individual Exam Selection** - Elegir Saber 11 o Saber Pro
6. **Saber 11 Form** - Formulario de 8 campos
7. **Saber Pro Form** - Formulario de 11 campos
8. **Prediction Result** - Modal/Pantalla de resultado

---

## Diseño Visual

- **Colores**: Azul (#3B82F6) como color primario, Cian (#06B6D4) como secundario
- **Tipografía**: Inter (sans-serif)
- **Fondo**: Gris claro (bg-slate-50)
- **Tarjetas**: Blanco con bordes redondeados
- **Badges**: SABER11 y SABERPRO en azul en esquina superior derecha

---

## Flujo de Navegación

```
Home
├── Uso Institucional
│   ├── Exam Selection (Saber 11 / Saber Pro)
│   ├── Bulk Upload
│   └── Bulk Results
│       └── Export / Nueva Carga
└── Uso Individual
    ├── Exam Selection (Saber 11 / Saber Pro)
    ├── Individual Form (Saber 11 / Saber Pro)
    └── Prediction Result
        └── Download Report / Back
```
