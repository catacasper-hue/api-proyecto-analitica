/**
 * EduPredictor.ai - Script Principal
 * Logica de navegacion y predicciones
 */

let currentExamType = 'saber11';
let selectedFile = null;
let currentPrediction = null;
let previousScreen = null;
let dropdownData = null;
const referencePayload = window.referencePayload || {};
const modelSchemaPayload = window.modelSchemaPayload || {};
const bulkTemplateDefinitions = {
    saber11: {
        columns: ['Nombre_Estudiante', 'Estrato', 'Educacion_Madre', 'Educacion_Padre', 'Internet', 'Consumo_Proteina', 'Edad'],
        rows: [
            ['Juan Perez', '2', 'Secundaria', 'Primaria', 'Entre 1 y 3 horas', 'Diariamente', '17'],
            ['Maria Lopez', '3', 'Universitaria', 'Secundaria', 'Más de 3 horas', 'Regularmente', '17']
        ]
    },
    saberpro: {
        columns: ['Nombre_Estudiante', 'Estrato', 'Educacion_Madre', 'Educacion_Padre', 'Internet', 'Consumo_Proteina', 'Matematicas_Prev', 'Lectura_Prev', 'Edad', 'Caracter_Institucion', 'Semestre_Actual', 'Horas_Trabajo_Semanal'],
        rows: [
            ['Juan Perez', '2', 'Secundaria', 'Primaria', 'Entre 1 y 3 horas', 'Diariamente', '65', '70', '22', 'INSTITUCION TECNOLOGICA', '06', '0'],
            ['Maria Lopez', '3', 'Universitaria', 'Secundaria', 'Más de 3 horas', 'Regularmente', '78', '82', '21', 'UNIVERSIDAD', '05', 'Entre 11 y 20 horas']
        ]
    }
};
const numericValidationRules = {
    edad_saber11: { min: 14, max: 30, label: 'Edad en Saber 11' },
    edad_saberpro: { min: 16, max: 100, label: 'Edad Saber Pro' },
    punt_global_saber11: { min: 0, max: 500, label: 'Puntaje Global Saber 11' },
    puntaje_global_saber11: { min: 0, max: 500, label: 'Puntaje Global Saber 11' },
    punt_matematicas: { min: 0, max: 100, label: 'Puntaje Matematicas' },
    puntaje_matematicas: { min: 0, max: 100, label: 'Puntaje Matematicas' },
    punt_lectura_critica: { min: 0, max: 100, label: 'Puntaje Lectura Critica' },
    puntaje_lectura: { min: 0, max: 100, label: 'Puntaje Lectura Critica' },
    punt_c_naturales: { min: 0, max: 100, label: 'Puntaje Ciencias Naturales' },
    punt_sociales_ciudadanas: { min: 0, max: 100, label: 'Puntaje Sociales y Ciudadanas' },
    punt_ingles: { min: 0, max: 100, label: 'Puntaje Ingles' }
};

// ========== NAVEGACION ==========

function navigateTo(screenId) {
    const activeScreen = document.querySelector('.screen.active');
    if (activeScreen && activeScreen.id !== 'home-screen') {
        previousScreen = activeScreen.id;
    }

    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });

    const screen = document.getElementById(screenId);
    if (screen) {
        screen.classList.add('active');
        window.scrollTo(0, 0);
    }
}

function goBack() {
    if (previousScreen) {
        navigateTo(previousScreen);
    } else {
        navigateTo('home-screen');
    }
}

function setExamType(type) {
    currentExamType = type;
    updateBadge(type);
    updateBulkTemplatePreview(type);
}

function updateBadge(type) {
    const badge = type === 'saber11' ? 'SABER11' : 'SABERPRO';
    document.querySelectorAll('.badge').forEach(element => {
        element.textContent = badge;
    });
}

function updateBulkTemplatePreview(type = currentExamType) {
    const template = bulkTemplateDefinitions[type] || bulkTemplateDefinitions.saber11;
    const table = document.querySelector('.template-table table');
    if (!table) return;

    const thead = table.querySelector('thead tr');
    const tbody = table.querySelector('tbody');
    if (!thead || !tbody) return;

    thead.innerHTML = template.columns.map(column => `<th>${column}</th>`).join('');
    tbody.innerHTML = template.rows.map(row => (
        `<tr>${row.map(value => `<td>${value}</td>`).join('')}</tr>`
    )).join('');
}

// ========== UPLOAD & FILE HANDLING ==========

function openFileDialog() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput) return;

    fileInput.value = '';
    fileInput.click();
}

function handleUploadAreaKeydown(e) {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openFileDialog();
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    const uploadArea = e.target.closest('#uploadArea');
    if (uploadArea) {
        uploadArea.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    const uploadArea = e.target.closest('#uploadArea');
    if (uploadArea) {
        uploadArea.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();

    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.classList.remove('drag-over');
    }

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        selectedFile = files[0];
        showFileInfo(selectedFile.name);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        selectedFile = files[0];
        showFileInfo(selectedFile.name);
    }
}

function showFileInfo(filename) {
    document.getElementById('fileName').textContent = filename;
    document.getElementById('fileInfo').classList.remove('hidden');
}

function uploadFile() {
    if (!selectedFile) {
        alert('Por favor selecciona un archivo');
        return;
    }

    showLoader();

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('exam_type', currentExamType);

    fetch('/api/predict/bulk', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoader();
        if (data.success) {
            displayBulkResults(data.resultados, data.total_registros);
            navigateTo('bulk-results');
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        hideLoader();
        console.error('Error:', error);
        alert('Error al procesar el archivo');
    });
}

function displayBulkResults(results, total) {
    document.getElementById('results-summary').textContent =
        `Analisis de ${total} estudiante${total !== 1 ? 's' : ''} procesado${total !== 1 ? 's' : ''}`;

    const tbody = document.getElementById('resultsList');
    tbody.innerHTML = '';

    results.forEach(result => {
        const row = document.createElement('tr');
        const riskClass = getRiskClass(result.Riesgo);

        row.innerHTML = `
            <td>${result.Nombre_Estudiante}</td>
            <td><strong>${result.Puntaje_Predicho}</strong></td>
            <td><span class="risk-${riskClass.toLowerCase()}">${result.Riesgo}</span></td>
            <td>${result.Desempenio}</td>
        `;
        tbody.appendChild(row);
    });
}

// ========== FORM SUBMISSION ==========

function submitForm(examType) {
    const formElement = examType === 'saber11'
        ? document.getElementById('saber11FormElement')
        : document.getElementById('saberproFormElement');

    syncParentWorkFields(formElement);
    syncProgramNucleo(formElement);

    const validationErrors = validateForm(formElement, examType);
    if (!formElement.checkValidity() || validationErrors.length) {
        formElement.reportValidity();
        alert(validationErrors.slice(0, 5).join('\n') || 'Por favor completa todos los campos requeridos');
        return;
    }

    const formData = new FormData(formElement);
    const data = Object.fromEntries(formData);

    showLoader();

    const endpoint = examType === 'saber11'
        ? '/api/predict/saber11'
        : '/api/predict/saberpro';

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        hideLoader();
        if (data.success) {
            displayPrediction(data, examType);
            navigateTo('prediction-result');
        } else {
            alert('Error: ' + (data.error || 'Error desconocido'));
        }
    })
    .catch(error => {
        hideLoader();
        console.error('Error:', error);
        alert('Error al procesar la prediccion');
    });
}

function displayPrediction(data, examType) {
    currentPrediction = data;
    currentExamType = examType;

    updateBadge(examType);

    const setText = (id, text) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = text;
        }
        return element;
    };

    setText('scoreValue', data.puntaje_estimado);

    const performanceColors = {
        'Desempeño Sobresaliente': 'green',
        'Desempeño Promedio': 'blue',
        'Riesgo de Deserción': 'red'
    };

    setText('performanceText', data.desempenio_esperado);
    const perfEl = document.getElementById('performanceText');
    if (perfEl) {
        perfEl.style.color = performanceColors[data.desempenio_esperado] || 'inherit';
    }

    setText('riskValue', data.riesgo);
    setText('percentileValue', data.percentil + '%');
    setText('interpretationText', getInterpretation(data, examType));
}

function getInterpretation(data, examType) {
    const score = data.puntaje_estimado;
    const risk = data.riesgo;
    const thresholds = examType === 'saberpro'
        ? { high: 190, medium: 150 }
        : { high: 350, medium: 250 };
    let interpretation = '';

    if (score >= thresholds.high) {
        interpretation = '\u00a1Excelente! Tu desempe\u00f1o predicho es sobresaliente. ';
        interpretation += 'Contin\u00faa desarrollando tus habilidades y considera programas especializados.';
    } else if (score >= thresholds.medium) {
        interpretation = 'Buen desempe\u00f1o predicho. Mant\u00e9n tu dedicaci\u00f3n al estudio ';
        interpretation += 'y contin\u00faa fortaleciendo las \u00e1reas donde tengas debilidades.';
    } else {
        interpretation = 'Tu predicci\u00f3n indica un desempe\u00f1o que requiere atenci\u00f3n. ';
        interpretation += 'Te recomendamos buscar apoyo acad\u00e9mico inmediato y refuerzo en \u00e1reas cr\u00edticas.';
    }

    if (risk === 'ALTO') {
        interpretation += ' Se detecta un riesgo alto. Por favor busca asesor\u00eda acad\u00e9mica profesional.';
    } else if (risk === 'MEDIO') {
        interpretation += ' Mant\u00e9n un seguimiento peri\u00f3dico de tu progreso.';
    }

    if (examType === 'saberpro') {
        interpretation += ' Como estudiante de educaci\u00f3n superior, aprovecha los recursos institucionales de apoyo.';
    }

    return interpretation;
}

function getRiskClass(risk) {
    if (risk === 'BAJO') return 'low';
    if (risk === 'MEDIO') return 'medium';
    if (risk === 'ALTO') return 'high';
    return 'medium';
}

// ========== UTILITY FUNCTIONS ==========

function showLoader() {
    document.getElementById('loader').classList.remove('hidden');
}

function hideLoader() {
    document.getElementById('loader').classList.add('hidden');
}

function downloadReport() {
    if (!currentPrediction) return;

    const exam = currentExamType === 'saber11' ? 'Saber 11' : 'Saber Pro';
    const date = new Date().toLocaleDateString('es-CO');

    const reportContent = `
REPORTE DE PREDICCI\u00d3N ACAD\u00c9MICA - EduPredictor.ai
==================================================

Examen: ${exam}
Fecha: ${date}
Hora: ${new Date().toLocaleTimeString('es-CO')}

RESULTADO PRINCIPAL
------------------
Puntaje Estimado: ${currentPrediction.puntaje_estimado}
Nivel de Riesgo: ${currentPrediction.riesgo}
Desempe\u00f1o Esperado: ${currentPrediction.desempenio_esperado}

M\u00c9TRICAS ADICIONALES
--------------------
Confianza del Modelo: ${(currentPrediction.confianza_modelo * 100).toFixed(1)}%
Percentil Predicho: ${currentPrediction.percentil}%

INTERPRETACI\u00d3N
--------------
${getInterpretation(currentPrediction, currentExamType)}

---
Este reporte ha sido generado autom\u00e1ticamente por EduPredictor.ai
Para m\u00e1s informaci\u00f3n: https://edupredictor.ai
    `;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prediccion_${currentExamType}_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function exportResults() {
    const table = document.getElementById('resultsTable');
    const csv = [];

    const headers = [];
    table.querySelectorAll('th').forEach(th => {
        headers.push(th.textContent);
    });
    csv.push(headers.join(','));

    table.querySelectorAll('tbody tr').forEach(tr => {
        const row = [];
        tr.querySelectorAll('td').forEach(td => {
            row.push('"' + td.textContent.trim() + '"');
        });
        csv.push(row.join(','));
    });

    const blob = new Blob([csv.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `resultados_${currentExamType}_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// ========== TEMPLATE DOWNLOAD ==========

function downloadTemplate() {
    fetch(`/api/template/${currentExamType}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const csv = convertToCSV(data.plantilla, data.columnas);
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `plantilla_${currentExamType}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    })
    .catch(error => console.error('Error:', error));
}

function convertToCSV(data, columns) {
    const csv = [];

    csv.push(columns.join(','));

    data.forEach(row => {
        const values = columns.map(col => row[col]);
        csv.push(values.join(','));
    });

    return csv.join('\n');
}

// ========== INITIALIZATION ==========

document.addEventListener('DOMContentLoaded', function() {
    navigateTo('home-screen');
    setupDependentDropdowns();
    setupSaber11AutoLevel();
    setupParentWorkSync();
    setupProgramNucleoSync();
    setupFieldTooltips();
    loadDropdownData();
    normalizeSaberProLabels();
    updateBulkTemplatePreview();
});

function renderModelDrivenFields() {
    renderAdditionalFields('saberproFormElement', modelSchemaPayload.saberpro, 'Variables del Modelo Saber Pro');
}
function normalizeSaberProLabels() {
    const form = document.getElementById('saberproFormElement');
    if (!form) return;

    const textFixes = [
        ['Situación económica en Saber 11', 'Situacion economica en Saber 11'],
        ['SituaciÃ³n econÃ³mica en Saber 11', 'Situacion economica en Saber 11'],
        ['Educación de la madre en Saber 11', 'Educacion de la madre en Saber 11'],
        ['EducaciÃ³n de la madre en Saber 11', 'Educacion de la madre en Saber 11'],
        ['Educación del padre en Saber 11', 'Educacion del padre en Saber 11'],
        ['EducaciÃ³n del padre en Saber 11', 'Educacion del padre en Saber 11'],
        ['Programa académico', 'Programa academico'],
        ['Programa acadÃ©mico', 'Programa academico'],
        ['Núcleo de pregrado', 'Nucleo de pregrado'],
        ['NÃºcleo de pregrado', 'Nucleo de pregrado'],
        ['Departamento de presentación Saber Pro', 'Departamento de presentacion Saber Pro'],
        ['Departamento de presentaciÃ³n Saber Pro', 'Departamento de presentacion Saber Pro'],
        ['Municipio de presentación Saber Pro', 'Municipio de presentacion Saber Pro'],
        ['Municipio de presentaciÃ³n Saber Pro', 'Municipio de presentacion Saber Pro'],
        ['¿La matrícula la pagan tus padres?', 'La matricula la pagan tus padres?'],
        ['Â¿La matrÃ­cula la pagan tus padres?', 'La matricula la pagan tus padres?'],
        ['¿La matrícula la pagas tú?', 'La matricula la pagas tu?'],
        ['Â¿La matrÃ­cula la pagas tÃº?', 'La matricula la pagas tu?'],
    ];

    form.querySelectorAll('label, option, h3, p').forEach(node => {
        const current = node.textContent.trim();
        const match = textFixes.find(([broken]) => broken === current);
        if (match) {
            node.textContent = match[1];
        }
        if (current === 'SÃ­' || current === 'Sí') {
            node.textContent = 'Si';
        }
        if (current === '9 o mÃ¡s' || current === '9 o más') {
            node.textContent = '9 o mas';
        }
    });
}

function renderAdditionalFields(formId, schema, title) {
    if (!schema) return;

    const form = document.getElementById(formId);
    if (!form) return;

    const fields = [];
    const seen = new Set();
    const syncedParentWorkFields = new Set([
        'fami_trabajolabormadre_saber11',
        'fami_trabajolaborpadre_saber11',
        'fami_trabajolabormadre_saberpro',
        'fami_trabajolaborpadre_saberpro'
    ]);

    (schema.numeric || []).forEach(name => {
        if (!syncedParentWorkFields.has(name) && !hasVisibleField(form, name) && !seen.has(name)) {
            fields.push(createFieldConfig(name, 'numeric'));
            seen.add(name);
        }
    });

    Object.entries(schema.ordinal || {}).forEach(([name, options]) => {
        if (!syncedParentWorkFields.has(name) && !hasVisibleField(form, name) && !seen.has(name)) {
            fields.push(createFieldConfig(name, 'ordinal', options));
            seen.add(name);
        }
    });

    (schema.low_card || []).forEach(name => {
        if (!syncedParentWorkFields.has(name) && !hasVisibleField(form, name) && !seen.has(name)) {
            fields.push(createFieldConfig(name, 'low_card'));
            seen.add(name);
        }
    });

    (schema.high_card || []).forEach(name => {
        if (!syncedParentWorkFields.has(name) && !hasVisibleField(form, name) && !seen.has(name)) {
            fields.push(createFieldConfig(name, 'high_card'));
            seen.add(name);
        }
    });

    if (!fields.length) return;

    const section = document.createElement('div');
    section.className = 'form-section';
    section.innerHTML = `<h3><i class="fas fa-sliders"></i> ${title}</h3><p>Si conoces esta informacion, completala tal como aparece en tus datos o reporte. Asi la prediccion se ajusta mejor a tu caso.</p>`;

    for (let i = 0; i < fields.length; i += 3) {
        const row = document.createElement('div');
        row.className = 'form-row';
        fields.slice(i, i + 3).forEach(field => {
            row.appendChild(buildFieldElement(field));
        });
        section.appendChild(row);
    }

    const submitButton = form.querySelector('.btn-primary.btn-large');
    if (submitButton) {
        form.insertBefore(section, submitButton);
    } else {
        form.appendChild(section);
    }
}

function hasVisibleField(form, name) {
    const fields = form.querySelectorAll(`[name="${name}"]`);
    return Array.from(fields).some(field => field.type !== 'hidden');
}

function createFieldConfig(name, kind, options = []) {
    const booleanFields = [
        'fami_tieneautomovil_saber11', 'fami_tienecomputador_saber11', 'fami_tieneconsolavideojuegos_saber11',
        'fami_tienehornomicroogas_saber11', 'fami_tienemotocicleta_saber11', 'fami_tieneautomovil_saberpro',
        'fami_tienehornomicroogas_saberpro', 'fami_tienemotocicleta_saberpro', 'fami_tieneserviciotv_saberpro',
        'estu_pagomatriculabeca', 'estu_pagomatriculacredito', 'estu_pagomatriculapadres', 'estu_pagomatriculapropio'
    ];

    if (booleanFields.includes(name)) {
        return { name, type: 'select', options: ['Si', 'No'] };
    }

    if (name === 'estu_generacione') {
        return { name, type: 'select', options: ['NO', 'GENERACION E - GRATUIDAD'] };
    }

    if (kind === 'ordinal') {
        return { name, type: 'select', options };
    }

    if (kind === 'numeric') {
        return { name, type: 'number' };
    }

    return { name, type: 'text' };
}

function buildFieldElement(field) {
    const wrapper = document.createElement('div');
    wrapper.className = 'form-group';

    const label = document.createElement('label');
    label.textContent = prettifyFieldName(field.name);
    appendLabelTooltip(label, field.name);
    wrapper.appendChild(label);

    let input;
    if (field.type === 'select') {
        input = document.createElement('select');
        input.innerHTML = '<option value="">Seleccione...</option>';
        (field.options || []).forEach(optionValue => {
            const option = document.createElement('option');
            option.value = optionValue;
            option.textContent = optionValue;
            input.appendChild(option);
        });
    } else {
        input = document.createElement('input');
        input.type = field.type;
        if (field.type === 'number') {
            input.step = '0.1';
        }
    }

    input.name = field.name;
    input.required = true;
    wrapper.appendChild(input);
    return wrapper;
}

const fieldTooltipTexts = {
    estu_inse_individual_saber11: 'INSE es el Indice Socioeconomico Individual. Resume condiciones del hogar y del estudiante usadas por ICFES para contextualizar el desempeno.',
    estu_inse_individual_saberpro: 'INSE es el Indice Socioeconomico Individual. Resume condiciones del hogar y del estudiante usadas por ICFES para contextualizar el desempeno.',
    estu_nse_ies: 'NSE es el Nivel Socioeconomico. IES significa Institucion de Educacion Superior; aqui representa el nivel socioeconomico asociado a la institucion.',
    estu_nse_establecimiento: 'NSE es el Nivel Socioeconomico del colegio o establecimiento educativo.',
    estu_nse_individual_saber11: 'NSE es el Nivel Socioeconomico individual del estudiante, usado para contextualizar condiciones del hogar.',
    estu_nse_individual_saberpro: 'NSE es el Nivel Socioeconomico individual del estudiante, usado para contextualizar condiciones del hogar.'
};

function appendLabelTooltip(label, fieldName) {
    const tooltip = fieldTooltipTexts[fieldName];
    if (!tooltip || label.querySelector('.tooltip-icon')) return;

    label.appendChild(document.createTextNode(' '));
    const icon = document.createElement('span');
    icon.className = 'tooltip-icon';
    icon.tabIndex = 0;
    icon.title = tooltip;
    icon.innerHTML = '<i class="fas fa-circle-question"></i>';
    label.appendChild(icon);
}

function setupFieldTooltips() {
    Object.keys(fieldTooltipTexts).forEach(fieldName => {
        document.querySelectorAll(`[name="${fieldName}"]`).forEach(field => {
            const group = field.closest('.form-group');
            const label = group ? group.querySelector('label') : null;
            if (label) {
                appendLabelTooltip(label, fieldName);
            }
        });
    });
}

function setupParentWorkSync() {
    document.querySelectorAll('[name="fami_ocupacionmadre"], [name="fami_ocupacionpadre"]').forEach(field => {
        if (field.dataset.parentWorkSync === 'true') return;
        const syncScope = () => syncParentWorkFields(field.closest('form') || document);
        field.addEventListener('change', syncScope);
        field.addEventListener('input', syncScope);
        field.dataset.parentWorkSync = 'true';
    });

    document.querySelectorAll('form').forEach(form => syncParentWorkFields(form));
}

function syncParentWorkFields(scope = document) {
    syncParentField(scope, 'fami_ocupacionmadre', [
        'fami_trabajolabormadre_saber11',
        'fami_trabajolabormadre_saberpro'
    ]);
    syncParentField(scope, 'fami_ocupacionpadre', [
        'fami_trabajolaborpadre_saber11',
        'fami_trabajolaborpadre_saberpro'
    ]);
}

function syncParentField(scope, sourceName, targetNames) {
    const source = Array.from(scope.querySelectorAll(`[name="${sourceName}"]`))
        .find(field => field.type !== 'hidden' && field.value);
    if (!source) return;

    targetNames.forEach(targetName => {
        scope.querySelectorAll(`[name="${targetName}"]`).forEach(target => {
            target.value = targetName.includes('trabajolabor')
                ? mapParentWorkValue(source.value)
                : source.value;
        });
    });
}

function mapParentWorkValue(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (normalized.includes('profesional')) {
        return 'Trabaja como profesional (por ejemplo médico, abogado, ingeniero)';
    }
    if (normalized.includes('directivo') || normalized.includes('gerente') || normalized === 'empresario') {
        return 'Es dueño de un negocio grande, tiene un cargo de nivel directivo o gerencial';
    }
    if (normalized.includes('cuenta propia') || normalized.includes('independiente') || normalized.includes('pequeno empresario')) {
        return 'Es dueño de un negocio pequeño (tiene pocos empleados o no tiene, por ejemplo tienda, papelería, etc';
    }
    if (normalized.includes('hogar') || normalized.includes('desempleado') || normalized.includes('otra')) {
        return 'Trabaja en el hogar, no trabaja o estudia';
    }
    if (normalized.includes('pensionado')) {
        return 'Pensionado';
    }
    return 'Tiene un trabajo de tipo auxiliar administrativo (por ejemplo, secretario o asistente)';
}

function prettifyFieldName(name) {
    const labelOverrides = {
        desemp_c_naturales: 'Desempeno en ciencias naturales',
        desemp_lectura_critica: 'Desempeno en lectura critica',
        desemp_matematicas: 'Desempeno en matematicas',
        desemp_sociales_ciudadanas: 'Desempeno en sociales y ciudadanas',
        desemp_ingles: 'Desempeno en ingles',
        estu_inse_individual_saber11: 'INSE individual en Saber 11',
        estu_inse_individual_saberpro: 'INSE individual en Saber Pro',
        estu_nse_ies: 'NSE de la IES',
        estu_nse_establecimiento: 'NSE del establecimiento',
        estu_nse_individual_saber11: 'NSE individual en Saber 11',
        estu_nse_individual_saberpro: 'NSE individual en Saber Pro',
        fami_comecerealfrutoslegumbre: 'Consumo de cereales, frutos y legumbres',
        fami_comelechederivados: 'Consumo de leche y derivados',
        fami_cuartoshogar: 'Cantidad de cuartos en el hogar',
        fami_numlibros: 'Numero de libros en el hogar',
        fami_tieneautomovil_saber11: 'Tiene automovil en el hogar (Saber 11)',
        fami_tienecomputador_saber11: 'Tiene computador en el hogar (Saber 11)',
        fami_tieneconsolavideojuegos_saber11: 'Tiene consola de videojuegos (Saber 11)',
        fami_tienehornomicroogas_saber11: 'Tiene horno microondas en el hogar (Saber 11)',
        fami_tienemotocicleta_saber11: 'Tiene motocicleta en el hogar (Saber 11)',
        fami_trabajolabormadre_saber11: 'Trabajo de la madre (Saber 11)',
        fami_trabajolaborpadre_saber11: 'Trabajo del padre (Saber 11)',
        fami_ocupacionmadre: 'Ocupacion de la madre',
        fami_ocupacionpadre: 'Ocupacion del padre',
        fami_tieneautomovil_saberpro: 'Tiene automovil en el hogar (Saber Pro)',
        fami_tienehornomicroogas_saberpro: 'Tiene horno microondas en el hogar (Saber Pro)',
        fami_tienemotocicleta_saberpro: 'Tiene motocicleta en el hogar (Saber Pro)',
        estu_prgm_departamento: 'Departamento del programa academico',
        estu_prgm_municipio: 'Municipio del programa academico',
        fami_trabajolabormadre_saberpro: 'Trabajo de la madre (Saber Pro)',
        fami_trabajolaborpadre_saberpro: 'Trabajo del padre (Saber Pro)',
    };

    if (labelOverrides[name]) {
        return labelOverrides[name];
    }

    return name
        .replace(/_/g, ' ')
        .replace(/\bsaber11\b/gi, 'Saber 11')
        .replace(/\bsaberpro\b/gi, 'Saber Pro')
        .replace(/\bnse\b/gi, 'NSE')
        .replace(/\binse\b/gi, 'INSE')
        .replace(/\bies\b/gi, 'IES')
        .replace(/\bmcpio\b/gi, 'municipio')
        .replace(/\bdepto\b/gi, 'departamento')
        .replace(/\bprgm\b/gi, 'programa')
        .replace(/\bc\b/gi, 'C')
        .replace(/\bcritica\b/gi, 'critica')
        .replace(/\bmatematicas\b/gi, 'matematicas')
        .replace(/\bnaturales\b/gi, 'naturales')
        .replace(/\bsociales\b/gi, 'sociales')
        .replace(/\bingles\b/gi, 'ingles')
        .replace(/\bpagomatricula\b/gi, 'pago matricula')
        .replace(/\bsemestrecursa\b/gi, 'semestre cursa')
        .replace(/\bhorassemanatrabaja\b/gi, 'horas semana trabaja')
        .replace(/\bocupacion\b/gi, 'ocupacion')
        .replace(/\blectura\b/gi, 'lectura')
        .replace(/\bdesemp\b/gi, 'desempeno')
        .replace(/\bglobal\b/gi, 'global')
        .replace(/\bpercentil\b/gi, 'percentil')
        .replace(/\bpunt\b/gi, 'puntaje')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/^\w/, char => char.toUpperCase());
}

async function loadDropdownData() {
    if (referencePayload.deptos_municipios) {
        dropdownData = referencePayload;
        populateInitialDropdowns();
        return;
    }

    try {
        const response = await fetch('/api/reference/dropdowns');
        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || 'No fue posible cargar los datos de referencia');
        }

        dropdownData = result.data;
        populateInitialDropdowns();
    } catch (error) {
        console.error('Error cargando dropdowns:', error);
    }
}

function populateInitialDropdowns() {
    if (!dropdownData) return;

    const deptos = Object.keys(dropdownData.deptos_municipios || {}).sort();

    document.querySelectorAll('.depto-select').forEach(select => {
        const currentValue = select.value;
        select.innerHTML = '<option value="">Seleccione...</option>';

        deptos.forEach(depto => {
            const option = document.createElement('option');
            option.value = depto;
            option.textContent = depto;
            select.appendChild(option);
        });

        if (currentValue) {
            select.value = resolveDeptoKey(currentValue) || currentValue;
            select.dispatchEvent(new Event('change'));
        }
    });
}

function setupDependentDropdowns() {
    document.querySelectorAll('.depto-select').forEach(select => {
        select.addEventListener('change', function() {
            const targetName = this.getAttribute('data-target');
            const form = this.closest('form');
            const targetSelect = form
                ? Array.from(form.querySelectorAll(`select[name="${targetName}"]`)).find(field => field.type !== 'hidden')
                : null;

            if (!targetSelect) return;

            targetSelect.innerHTML = '<option value="">Seleccione Municipio...</option>';

            if (!dropdownData || !dropdownData.deptos_municipios) return;

            const selectedDepto = resolveDeptoKey(this.value);
            if (selectedDepto && dropdownData.deptos_municipios[selectedDepto]) {
                dropdownData.deptos_municipios[selectedDepto].forEach(mcpio => {
                    const option = document.createElement('option');
                    option.value = mcpio;
                    option.textContent = mcpio;
                    targetSelect.appendChild(option);
                });
            }
        });
    });
}

function normalizeLookupText(value) {
    return String(value || '')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/\s+/g, ' ')
        .trim()
        .toUpperCase();
}

function resolveDeptoKey(value) {
    if (!dropdownData || !dropdownData.deptos_municipios || !value) return '';
    const deptos = Object.keys(dropdownData.deptos_municipios);
    const direct = deptos.find(depto => depto === value);
    if (direct) return direct;

    const normalizedValue = normalizeLookupText(value);
    return deptos.find(depto => normalizeLookupText(depto) === normalizedValue) || '';
}

function setupProgramNucleoSync() {
    document.querySelectorAll('form').forEach(form => {
        const programFields = Array.from(form.querySelectorAll('[name="estu_prgm_academico"]'))
            .filter(field => field.type !== 'hidden');
        programFields.forEach(field => {
            if (field.dataset.nucleoSync === 'true') return;
            const sync = () => syncProgramNucleo(form);
            field.addEventListener('input', sync);
            field.addEventListener('change', sync);
            field.dataset.nucleoSync = 'true';
            sync();
        });
    });
}

function syncProgramNucleo(scope = document) {
    const program = Array.from(scope.querySelectorAll('[name="estu_prgm_academico"]'))
        .reverse()
        .find(field => field.type !== 'hidden' && field.value);
    if (!program) return;

    scope.querySelectorAll('[name="estu_nucleo_pregrado"]').forEach(field => {
        field.value = program.value;
    });
}

function setupSaber11AutoLevel() {
    const puntGlobalInput = document.getElementById('punt_global_saber11');
    const nivelSaber11Select = document.getElementById('nivel_saber11');

    if (puntGlobalInput && nivelSaber11Select) {
        const updateLevel = () => {
            const score = parseFloat(puntGlobalInput.value);
            if (isNaN(score)) return;

            let level = 'Excelente';
            if (score < 200) level = 'Deficiente';
            else if (score < 250) level = 'Bajo';
            else if (score < 300) level = 'Regular';
            else if (score < 350) level = 'Alto';

            nivelSaber11Select.value = level;
        };

        puntGlobalInput.addEventListener('input', updateLevel);
        updateLevel();
    }
}

// ========== VALIDACION DE FORMULARIOS ==========

function validateForm(formElement, examType = currentExamType) {
    const inputs = formElement.querySelectorAll('input[required], select[required]');
    const errors = [];

    inputs.forEach(input => {
        input.setCustomValidity('');
        input.style.borderColor = '#E2E8F0';

        if (!input.value) {
            markInvalid(input, 'Este campo es obligatorio');
            errors.push(`${getFieldLabel(input)} es obligatorio`);
            return;
        }

        if (input.type === 'number') {
            const value = Number(input.value);
            const rule = numericValidationRules[input.name] || {};
            const min = rule.min !== undefined ? rule.min : (input.min !== '' ? Number(input.min) : null);
            const max = rule.max !== undefined ? rule.max : (input.max !== '' ? Number(input.max) : null);
            const label = rule.label || getFieldLabel(input);
            if (!Number.isFinite(value)) {
                markInvalid(input, 'Ingresa un numero valido');
                errors.push(`${label} debe ser un numero valido`);
                return;
            }
            if (min !== null && value < min) {
                markInvalid(input, `El valor minimo es ${min}`);
                errors.push(`${label} debe ser mayor o igual a ${min}`);
                return;
            }
            if (max !== null && value > max) {
                markInvalid(input, `El valor maximo es ${max}`);
                errors.push(`${label} debe ser menor o igual a ${max}`);
            }
        }
    });

    if (examType === 'saberpro') {
        const age11 = getNumericField(formElement, 'edad_saber11');
        const agePro = getNumericField(formElement, 'edad_saberpro');
        if (age11 !== null && agePro !== null && age11 > agePro) {
            const field = formElement.querySelector('[name="edad_saber11"]');
            markInvalid(field, 'La edad en Saber 11 no puede ser mayor que la edad en Saber Pro');
            errors.push('La edad en Saber 11 no puede ser mayor que la edad en Saber Pro');
        }
    }

    return errors;
}

function markInvalid(input, message) {
    if (!input) return;
    input.style.borderColor = '#EF4444';
    input.setCustomValidity(message);
}

function getFieldLabel(input) {
    const group = input.closest('.form-group');
    const label = group ? group.querySelector('label') : null;
    return label ? label.textContent.trim() : input.name;
}

function getNumericField(formElement, name) {
    const field = Array.from(formElement.querySelectorAll(`[name="${name}"]`))
        .reverse()
        .find(element => element.type !== 'hidden' && element.value);
    if (!field) return null;
    const value = Number(field.value);
    return Number.isFinite(value) ? value : null;
}

// ========== EXPORTAR PARA TESTING ==========

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        navigateTo,
        setExamType,
        submitForm,
        downloadReport
    };
}

