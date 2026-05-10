"""
EduPredictor.ai - Flask API Backend
API REST para predicciones académicas usando modelos de ML
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import warnings
import json

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

from cuml_pickle_compat import install as install_cuml_pickle_compat

warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Configuración
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


REFERENCE_PAYLOAD = {
    "quartiles": {
        "q1": {
            "label": "Cuartil 1",
            "percentile": 13,
            "scores": {
                "lectura": 47.0,
                "matematicas": 43.0,
                "sociales": 39.0,
                "ciencias": 43.0,
                "ingles": 43.0,
            },
        },
        "q2": {
            "label": "Cuartil 2",
            "percentile": 38,
            "scores": {
                "lectura": 52.0,
                "matematicas": 50.0,
                "sociales": 45.0,
                "ciencias": 49.0,
                "ingles": 49.0,
            },
        },
        "q3": {
            "label": "Cuartil 3",
            "percentile": 63,
            "scores": {
                "lectura": 56.0,
                "matematicas": 55.0,
                "sociales": 50.0,
                "ciencias": 53.0,
                "ingles": 54.0,
            },
        },
        "q4": {
            "label": "Cuartil 4",
            "percentile": 88,
            "scores": {
                "lectura": 62.0,
                "matematicas": 62.0,
                "sociales": 57.0,
                "ciencias": 59.0,
                "ingles": 63.0,
            },
        },
    },
    "strengths": {
        "balanced": "Equilibrado",
        "math": "Fortaleza en Matemáticas",
        "reading": "Fortaleza en Lectura",
        "english": "Fortaleza en Inglés",
    },
    "saber11": {
        "source_label": "ICFES Saber 11 2025",
        "published_on": "2026-01-22",
        "note": (
            "Los rangos automáticos usan agregados oficiales ICFES Saber 11 2025 "
            "(calendarios A y B, publicados el 2026-01-22)."
        ),
        "global_average": 258.57,
        "subject_averages": {
            "lectura": 54.23,
            "matematicas": 52.56,
            "sociales": 48.09,
            "ciencias": 50.87,
            "ingles": 52.82,
        },
    },
    "saberpro": {
        "source_label": "ICFES Saber Pro 2024",
        "published_on": "2026-04-06",
        "note": (
            "Los puntajes base usan anclajes oficiales de Saber 11 2025. "
            "El agregado nacional Saber Pro 2025 aun no estaba publicado al 2026-04-26; "
            "segun el calendario oficial de difusion del ICFES, esa publicacion esta prevista "
            "para el 2026-07-31, por eso se usa el ultimo agregado oficial disponible de 2024."
        ),
        "global_average": 146.23,
        "module_averages": {
            "competencias_ciudadanas": 145.64,
            "comunicacion_escrita": 133.98,
            "ingles": 155.35,
            "lectura_critica": 152.51,
            "razonamiento_cuantitativo": 143.67,
        },
    },
}


class ModelManager:
    """Gestor de modelos y adaptador de inferencia."""

    def __init__(self, model_dir: str = "."):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}
        self.load_errors: Dict[str, str] = {}

        self.education_map = {
            "Ninguno": "Ninguno",
            "ninguno": "Ninguno",
            "Primaria": "Primaria completa",
            "primaria": "Primaria completa",
            "Secundaria": "Secundaria (Bachillerato) completa",
            "secundaria": "Secundaria (Bachillerato) completa",
            "Tecnica": "Técnica o tecnológica completa",
            "Técnica": "Técnica o tecnológica completa",
            "Tecnológica": "Técnica o tecnológica completa",
            "tecnica": "Técnica o tecnológica completa",
            "tecnologica": "Técnica o tecnológica completa",
            "Universitaria": "Educación profesional completa",
            "universitaria": "Educación profesional completa",
            "Posgrado": "Postgrado",
            "posgrado": "Postgrado",
            "Técnica o tecnológica completa": "Técnica o tecnológica completa",
            "Educación profesional completa": "Educación profesional completa",
            "Postgrado": "Postgrado",
        }
        self.internet_map = {
            "No Navega Internet": "No Navega Internet",
            "Menos de 1 hora": "30 minutos o menos",
            "1-2 horas": "Entre 1 y 3 horas",
            "2-4 horas": "Entre 1 y 3 horas",
            "Más de 4 horas": "Más de 3 horas",
            "Mas de 4 horas": "Más de 3 horas",
            "Más de 3 horas": "Más de 3 horas",
            "Mas de 3 horas": "Más de 3 horas",
            "No": "No Navega Internet",
            "Sí": "Entre 1 y 3 horas",
            "Si": "Entre 1 y 3 horas",
        }
        self.protein_map = {
            "Nunca o rara vez comemos eso": "Nunca o rara vez comemos eso",
            "Nunca": "Nunca o rara vez comemos eso",
            "A veces": "1 o 2 veces por semana",
            "Regularmente": "3 a 5 veces por semana",
            "Frecuentemente": "Todos o casi todos los días",
            "Diariamente": "Todos o casi todos los días",
            "1 o 2 veces por semana": "1 o 2 veces por semana",
            "3 a 5 veces por semana": "3 a 5 veces por semana",
            "Todos o casi todos los días": "Todos o casi todos los días",
            "Todos o casi todos los dias": "Todos o casi todos los días",
        }
        self.institution_map = {
            "TÉCNICA PROFESIONAL": "TÉCNICA PROFESIONAL",
            "TECNICA PROFESIONAL": "TÉCNICA PROFESIONAL",
            "TECNOLÓGICA": "INSTITUCIÓN TECNOLÓGICA",
            "TECNOLOGICA": "INSTITUCIÓN TECNOLÓGICA",
            "INSTITUCIÓN TECNOLÓGICA": "INSTITUCIÓN TECNOLÓGICA",
            "INSTITUCION TECNOLOGICA": "INSTITUCIÓN TECNOLÓGICA",
            "INSTITUCIÓN UNIVERSITARIA": "INSTITUCIÓN UNIVERSITARIA",
            "INSTITUCION UNIVERSITARIA": "INSTITUCIÓN UNIVERSITARIA",
            "UNIVERSIDAD": "UNIVERSIDAD",
        }
        self.bulk_required_columns = {
            "saber11": [
                "Nombre_Estudiante",
                "Estrato",
                "Educacion_Madre",
                "Educacion_Padre",
                "Internet",
                "Consumo_Proteina",
                "Edad",
            ],
            "saberpro": [
                "Nombre_Estudiante",
                "Estrato",
                "Educacion_Madre",
                "Educacion_Padre",
                "Internet",
                "Consumo_Proteina",
                "Matematicas_Prev",
                "Lectura_Prev",
                "Edad",
                "Caracter_Institucion",
                "Semestre_Actual",
                "Horas_Trabajo_Semanal",
            ],
        }

        self.load_models()

    def load_models(self) -> None:
        model_files = {
            "saber11_regression": "best_model_saber11_regression.joblib",
            "saberpro_regression": "best_model_saberpro_regression.joblib",
        }

        self.models = {}
        self.load_errors = {}

        for key, filename in model_files.items():
            try:
                path = self.model_dir / filename
                self.models[key] = self._load_model_artifact(path)
                print(f"{key} cargado correctamente")
            except FileNotFoundError:
                message = f"{filename} no encontrado en {self.model_dir}"
                self.load_errors[key] = message
                print(message)
            except Exception as exc:
                self.load_errors[key] = str(exc)
                print(f"Error al cargar {filename}: {exc}")

    def _load_model_artifact(self, path: Path) -> Any:
        try:
            return joblib.load(path)
        except Exception as exc:
            if "cuml" not in str(exc).lower():
                raise

        install_cuml_pickle_compat()
        return joblib.load(path)

    def _is_bundle(self, artifact: Any) -> bool:
        return isinstance(artifact, dict) and "model" in artifact and "preprocessor" in artifact

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""

        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            return ""

        if "Ã" in text or "Â" in text:
            try:
                text = text.encode("latin1").decode("utf-8")
            except UnicodeError:
                pass

        replacements = {
            "d?as": "días",
            "D?AS": "DÍAS",
            "M?S": "MÁS",
            "m?s": "más",
            "mÃ¡s": "más",
            "SÃ­": "Sí",
            "Â¿": "¿",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return " ".join(text.split())

    def _is_blank(self, value: Any) -> bool:
        return self._clean_text(value) == ""

    def _first_present(self, *values: Any) -> Any:
        for value in values:
            if not self._is_blank(value):
                return value
        return None

    def _map_yes_no(self, value: Any, default: str = "No") -> str:
        cleaned = self._clean_text(value).lower()
        if cleaned in {"si", "sí", "s", "1", "true", "yes"}:
            return "Si"
        if cleaned in {"no", "n", "0", "false"}:
            return "No"
        return default

    def _parse_estrato_number(self, value: Any) -> int:
        mapping = {
            "Sin Estrato": 0,
            "Estrato 1": 1,
            "Estrato 2": 2,
            "Estrato 3": 3,
            "Estrato 4": 4,
            "Estrato 5": 5,
            "Estrato 6": 6,
        }
        value = self._clean_text(value)
        if value in mapping:
            return mapping[value]
        numeric = self._safe_int(value, 2)
        return int(self._clamp(numeric, 0, 6))

    def _format_estrato_label(self, estrato_num: int) -> str:
        return "Sin Estrato" if estrato_num == 0 else f"Estrato {estrato_num}"

    def _map_nse_label(self, estrato_num: int) -> str:
        if estrato_num <= 1:
            return "NSE1"
        if estrato_num == 2:
            return "NSE2"
        if estrato_num <= 4:
            return "NSE3"
        return "NSE4"

    def _estimate_inse(self, estrato_num: int) -> float:
        inse_map = {0: 35.0, 1: 40.0, 2: 50.0, 3: 60.0, 4: 70.0, 5: 80.0, 6: 90.0}
        return inse_map.get(estrato_num, 50.0)

    def _map_parent_education(self, value: Any) -> str:
        return self.education_map.get(self._clean_text(value), "Secundaria (Bachillerato) completa")

    def _map_internet_usage(self, value: Any) -> str:
        cleaned = self._clean_text(value)
        return self.internet_map.get(cleaned, "Entre 1 y 3 horas")

    def _map_protein_consumption(self, value: Any) -> str:
        cleaned = self._clean_text(value)
        return self.protein_map.get(cleaned, "3 a 5 veces por semana")

    def _map_institution_character(self, value: Any) -> str:
        cleaned = self._clean_text(value).upper()
        return self.institution_map.get(cleaned, "UNIVERSIDAD")

    def _map_work_hours(self, value: Any) -> str:
        cleaned = self._clean_text(value)
        exact_map = {
            "0": "0",
            "1-10": "Menos de 10 horas",
            "11-20": "Entre 11 y 20 horas",
            "21-30": "Entre 21 y 30 horas",
            "31-40": "Más de 30 horas",
            "Más de 40": "Más de 30 horas",
            "Mas de 40": "Más de 30 horas",
        }
        if cleaned in exact_map:
            return exact_map[cleaned]

        hours = self._safe_int(value, 0)
        if hours <= 0:
            return "0"
        if hours <= 10:
            return "Menos de 10 horas"
        if hours <= 20:
            return "Entre 11 y 20 horas"
        if hours <= 30:
            return "Entre 21 y 30 horas"
        return "Más de 30 horas"

    def _format_semester(self, value: Any) -> str:
        cleaned = self._clean_text(value)
        if cleaned in {"12 o más", "12 o mas", "Más de 12", "Mas de 12"}:
            return "12 o más"
        semester = self._safe_int(value, 6)
        if semester >= 12:
            return "12 o más"
        return f"{int(self._clamp(semester, 1, 11)):02d}"

    def _map_reading_dedication(self, value: Any, fallback_score: float = 50.0) -> str:
        cleaned = self._clean_text(value)
        mapping = {
            "No leo": "No leo por entretenimiento",
            "No leo por entretenimiento": "No leo por entretenimiento",
            "30 minutos o menos": "30 minutos o menos",
            "Entre 30 y 60 minutos": "Entre 30 y 60 minutos",
            "Entre 1 y 2 horas": "Entre 1 y 2 horas",
            "Más de 2 horas": "Más de 2 horas",
            "Mas de 2 horas": "Más de 2 horas",
        }
        return mapping.get(cleaned, self._estimate_reading_dedication(fallback_score))

    def _map_household_people(self, value: Any, estrato_num: int = 3) -> str:
        cleaned = self._clean_text(value)
        exact = {"1 a 2", "3 a 4", "5 a 6", "7 a 8", "9 o más"}
        if cleaned in exact:
            return cleaned
        if cleaned in {"9 o mas", "9 o m?s"}:
            return "9 o más"

        people = self._safe_int(cleaned, 4)
        if people <= 2:
            return "1 a 2"
        if people <= 4:
            return "3 a 4"
        if people <= 6:
            return "5 a 6"
        if people <= 8:
            return "7 a 8"
        return "9 o más"

    def _map_rooms(self, value: Any, estrato_num: int = 3) -> str:
        cleaned = self._clean_text(value)
        exact = {"Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis o mas"}
        if cleaned in exact:
            return cleaned
        rooms = self._safe_int(cleaned, 3)
        return {
            1: "Uno",
            2: "Dos",
            3: "Tres",
            4: "Cuatro",
            5: "Cinco",
        }.get(int(self._clamp(rooms, 1, 6)), "Seis o mas")

    def _map_books(self, value: Any) -> str:
        cleaned = self._clean_text(value).upper()
        exact = {"0 A 10 LIBROS", "11 A 25 LIBROS", "26 A 100 LIBROS", "MÁS DE 100 LIBROS"}
        if cleaned in exact:
            return cleaned
        if cleaned in {"MAS DE 100 LIBROS", "M?S DE 100 LIBROS"}:
            return "MÁS DE 100 LIBROS"
        books = self._safe_int(cleaned, 25)
        if books <= 10:
            return "0 A 10 LIBROS"
        if books <= 25:
            return "11 A 25 LIBROS"
        if books <= 100:
            return "26 A 100 LIBROS"
        return "MÁS DE 100 LIBROS"

    def _map_economic_situation(self, value: Any) -> str:
        cleaned = self._clean_text(value).capitalize()
        return cleaned if cleaned in {"Peor", "Igual", "Mejor"} else "Igual"

    def _map_gender(self, value: Any) -> str:
        cleaned = self._clean_text(value).upper()
        if cleaned in {"F", "FEMENINO", "MUJER"}:
            return "F"
        return "M"

    def _default_ordinal_value(self, mapping: Dict[Any, float]) -> Any:
        ordered = [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
        return ordered[len(ordered) // 2]

    def get_reference_payload(self) -> Dict[str, Any]:
        return REFERENCE_PAYLOAD

    def get_model_schema_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}

        for exam_type, model_key in {
            "saber11": "saber11_regression",
            "saberpro": "saberpro_regression",
        }.items():
            artifact = self.models.get(model_key)
            if not artifact or not self._is_bundle(artifact):
                payload[exam_type] = {"numeric": [], "ordinal": {}, "low_card": [], "high_card": []}
                continue

            state = artifact["preprocessor"].state
            payload[exam_type] = {
                "numeric": list(state.numeric_columns),
                "ordinal": {col: list(mapping.keys()) for col, mapping in state.ordinal_maps.items()},
                "low_card": list(state.low_card_columns),
                "high_card": list(state.high_card_columns),
            }

        return payload

    def _model_feature_names(self, model_key: str) -> set:
        artifact = self.models.get(model_key)
        if not artifact or not self._is_bundle(artifact):
            return set()

        state = artifact["preprocessor"].state
        return {
            *state.numeric_columns,
            *state.ordinal_maps.keys(),
            *state.low_card_columns,
            *state.high_card_columns,
        }

    def _preserve_model_fields(self, model_key: str, full_data: Dict[str, Any], data_dict: Dict[str, Any]) -> None:
        for key in self._model_feature_names(model_key):
            if key in full_data:
                data_dict[key] = full_data[key]

    def _sync_parent_work_with_occupation(self, data: Dict[str, Any]) -> None:
        mother_occupation = data.get("fami_ocupacionmadre")
        father_occupation = data.get("fami_ocupacionpadre")
        mother_level = self._simplify_education_level(
            data.get("fami_educacionmadre_saber11")
            or data.get("fami_educacionmadre_saberpro")
            or data.get("educacion_madre")
        )
        father_level = self._simplify_education_level(
            data.get("fami_educacionpadre_saber11")
            or data.get("fami_educacionpadre_saberpro")
            or data.get("educacion_padre")
        )

        if mother_occupation not in (None, ""):
            data["fami_ocupacionmadre"] = self._map_parent_occupation(mother_occupation, "mother", mother_level)
            data["fami_trabajolabormadre_saber11"] = self._map_parent_work(mother_occupation, "mother", mother_level)
            data["fami_trabajolabormadre_saberpro"] = data["fami_trabajolabormadre_saber11"]
        if father_occupation not in (None, ""):
            data["fami_ocupacionpadre"] = self._map_parent_occupation(father_occupation, "father", father_level)
            data["fami_trabajolaborpadre_saber11"] = self._map_parent_work(father_occupation, "father", father_level)
            data["fami_trabajolaborpadre_saberpro"] = data["fami_trabajolaborpadre_saber11"]

    def _normalize_inse_inputs(self, data: Dict[str, Any]) -> None:
        inse_labels = {
            "muy bajo": 35.0,
            "bajo": 45.0,
            "medio": 60.0,
            "alto": 75.0,
            "muy alto": 90.0,
        }
        for key in ("estu_inse_individual_saber11", "estu_inse_individual_saberpro"):
            value = data.get(key)
            if value in (None, ""):
                continue
            cleaned = str(value).strip().lower()
            if cleaned in inse_labels:
                data[key] = inse_labels[cleaned]

    def _normalize_quartile(self, value: Any, fallback_score: Optional[float] = None) -> str:
        mapping = {
            "q1": "q1",
            "1": "q1",
            "cuartil1": "q1",
            "cuartil 1": "q1",
            "q2": "q2",
            "2": "q2",
            "cuartil2": "q2",
            "cuartil 2": "q2",
            "q3": "q3",
            "3": "q3",
            "cuartil3": "q3",
            "cuartil 3": "q3",
            "q4": "q4",
            "4": "q4",
            "cuartil4": "q4",
            "cuartil 4": "q4",
        }
        cleaned = self._clean_text(value).lower()
        if cleaned in mapping:
            return mapping[cleaned]

        if fallback_score is not None:
            if fallback_score < 48.5:
                return "q1"
            if fallback_score < 53.0:
                return "q2"
            if fallback_score < 57.5:
                return "q3"
            return "q4"

        return "q2"

    def _normalize_strength(self, value: Any) -> str:
        mapping = {
            "balanced": "balanced",
            "equilibrado": "balanced",
            "math": "math",
            "matematicas": "math",
            "matemáticas": "math",
            "reading": "reading",
            "lectura": "reading",
            "english": "english",
            "ingles": "english",
            "inglés": "english",
        }
        cleaned = self._clean_text(value).lower()
        return mapping.get(cleaned, "balanced")

    def _strength_label(self, strength_key: str) -> str:
        return REFERENCE_PAYLOAD["strengths"].get(strength_key, REFERENCE_PAYLOAD["strengths"]["balanced"])

    def _derive_score_profile(self, quartile_key: str, strength_key: str) -> Dict[str, float]:
        base_scores = dict(REFERENCE_PAYLOAD["quartiles"][quartile_key]["scores"])
        adjustments = {
            "balanced": {"lectura": 0.0, "matematicas": 0.0, "sociales": 0.0, "ciencias": 0.0, "ingles": 0.0},
            "math": {"lectura": -2.0, "matematicas": 4.0, "sociales": -1.0, "ciencias": 2.0, "ingles": 0.0},
            "reading": {"lectura": 4.0, "matematicas": -2.0, "sociales": 2.0, "ciencias": 0.0, "ingles": 3.0},
            "english": {"lectura": 1.0, "matematicas": -1.0, "sociales": 0.0, "ciencias": 0.0, "ingles": 5.0},
        }
        for subject, delta in adjustments.get(strength_key, adjustments["balanced"]).items():
            base_scores[subject] = round(self._clamp(base_scores[subject] + delta, 20.0, 100.0), 2)
        return base_scores

    def _derive_profile_percentiles(self, quartile_key: str, strength_key: str) -> Dict[str, int]:
        base_percentile = REFERENCE_PAYLOAD["quartiles"][quartile_key]["percentile"]
        adjustments = {
            "balanced": {"lectura": 0, "matematicas": 0, "sociales": 0, "ciencias": 0, "ingles": 0},
            "math": {"lectura": -4, "matematicas": 7, "sociales": -2, "ciencias": 3, "ingles": 0},
            "reading": {"lectura": 7, "matematicas": -4, "sociales": 3, "ciencias": 0, "ingles": 5},
            "english": {"lectura": 1, "matematicas": -2, "sociales": 0, "ciencias": 0, "ingles": 8},
        }
        profile = {}
        for subject, delta in adjustments.get(strength_key, adjustments["balanced"]).items():
            profile[subject] = int(self._clamp(base_percentile + delta, 1, 99))
        return profile

    def _infer_internet_input(self, estrato_num: int, quartile_key: str) -> str:
        if estrato_num <= 0:
            return "No Navega Internet"
        if estrato_num <= 1 and quartile_key == "q1":
            return "Menos de 1 hora"
        if estrato_num >= 4 and quartile_key in {"q3", "q4"}:
            return "Más de 4 horas"
        if quartile_key == "q4":
            return "2-4 horas"
        return "1-2 horas"

    def _infer_protein_input(self, estrato_num: int, quartile_key: str) -> str:
        if estrato_num <= 1:
            return "A veces" if quartile_key in {"q1", "q2"} else "Regularmente"
        if estrato_num <= 3:
            return "Regularmente"
        if quartile_key == "q4":
            return "Diariamente"
        return "Frecuentemente"

    def _simplify_education_level(self, value: Any) -> str:
        cleaned = self._clean_text(value).lower()
        if "post" in cleaned:
            return "posgrado"
        if "univers" in cleaned or "profes" in cleaned:
            return "universitaria"
        if "téc" in cleaned or "tec" in cleaned:
            return "tecnica"
        if "sec" in cleaned or "bach" in cleaned:
            return "secundaria"
        if "prim" in cleaned:
            return "primaria"
        return "ninguno"

    def _infer_parent_work(self, education_level: str) -> str:
        mapping = {
            "ninguno": "Trabaja en el hogar, no trabaja o estudia",
            "primaria": "Es agricultor, pesquero o jornalero",
            "secundaria": "Tiene un trabajo de tipo auxiliar administrativo (por ejemplo, secretario o asistente)",
            "tecnica": "Tiene un trabajo de tipo auxiliar administrativo (por ejemplo, secretario o asistente)",
            "universitaria": "Trabaja como profesional (por ejemplo médico, abogado, ingeniero)",
            "posgrado": "Es dueño de un negocio grande, tiene un cargo de nivel directivo o gerencial",
        }
        return mapping.get(education_level, "Tiene un trabajo de tipo auxiliar administrativo (por ejemplo, secretario o asistente)")

    def _infer_parent_occupation(self, education_level: str, parent: str) -> str:
        professional = "Profesional Independiente" if parent == "mother" else "Profesional independiente"
        mapping = {
            "ninguno": "Trabajador por cuenta propia",
            "primaria": "Trabajador por cuenta propia",
            "secundaria": "Empleado de nivel auxiliar o administrativo",
            "tecnica": "Empleado de nivel técnico o profesional",
            "universitaria": professional,
            "posgrado": "Empleado de nivel directivo",
        }
        return mapping.get(education_level, "Empleado de nivel auxiliar o administrativo")

    def _map_parent_occupation(self, value: Any, parent: str, education_level: str = "secundaria") -> str:
        cleaned = self._clean_text(value)
        professional = "Profesional Independiente" if parent == "mother" else "Profesional independiente"
        exact_values = {
            "Empleado con cargo como director o gerente general",
            "Empleado de nivel auxiliar o administrativo",
            "Empleado de nivel directivo",
            "Empleado de nivel técnico o profesional",
            "Empleado obrero u operario",
            "Empresario",
            "Hogar",
            "Otra actividad u ocupación",
            "Pensionado",
            "Pequeño empresario",
            "Profesional Independiente",
            "Profesional independiente",
            "Trabajador por cuenta propia",
        }
        if cleaned in exact_values:
            if cleaned.lower() == "profesional independiente":
                return professional
            return cleaned

        aliases = {
            "profesional": professional,
            "independiente": "Trabajador por cuenta propia",
            "ama de casa": "Hogar",
            "hogar": "Hogar",
            "empleado(a)": "Empleado de nivel auxiliar o administrativo",
            "empleado": "Empleado de nivel auxiliar o administrativo",
            "desempleado(a)": "Otra actividad u ocupación",
            "desempleado": "Otra actividad u ocupación",
            "otra": "Otra actividad u ocupación",
            "pensionado": "Pensionado",
        }
        return aliases.get(cleaned.lower(), self._infer_parent_occupation(education_level, parent))

    def _map_parent_work(self, value: Any, parent: str, education_level: str = "secundaria") -> str:
        cleaned = self._clean_text(value)
        exact_values = {
            "Es agricultor, pesquero o jornalero",
            "Es dueño de un negocio grande, tiene un cargo de nivel directivo o gerencial",
            "Es dueño de un negocio pequeño (tiene pocos empleados o no tiene, por ejemplo tienda, papelería, etc",
            "Es operario de máquinas o conduce vehículos (taxita, chofer)",
            "Es vendedor o trabaja en atención al público",
            "No aplica",
            "No sabe",
            "Pensionado",
            "Tiene un trabajo de tipo auxiliar administrativo (por ejemplo, secretario o asistente)",
            "Trabaja como personal de limpieza, mantenimiento, seguridad o construcción",
            "Trabaja como profesional (por ejemplo médico, abogado, ingeniero)",
            "Trabaja en el hogar, no trabaja o estudia",
        }
        if cleaned in exact_values:
            return cleaned

        occupation = self._map_parent_occupation(value, parent, education_level).lower()
        if "profesional" in occupation:
            return "Trabaja como profesional (por ejemplo médico, abogado, ingeniero)"
        if "directivo" in occupation or "gerente" in occupation or occupation == "empresario":
            return "Es dueño de un negocio grande, tiene un cargo de nivel directivo o gerencial"
        if "técnico" in occupation or "tecnico" in occupation or "auxiliar" in occupation:
            return "Tiene un trabajo de tipo auxiliar administrativo (por ejemplo, secretario o asistente)"
        if "cuenta propia" in occupation or "pequeño empresario" in occupation:
            return "Es dueño de un negocio pequeño (tiene pocos empleados o no tiene, por ejemplo tienda, papelería, etc"
        if occupation == "hogar" or "otra" in occupation:
            return "Trabaja en el hogar, no trabaja o estudia"
        if "pensionado" in occupation:
            return "Pensionado"
        return self._infer_parent_work(education_level)

    def _derive_tuition_payment_flags(
        self,
        estrato_num: int,
        work_hours_band: str,
        institution_character: str,
    ) -> Dict[str, str]:
        choice = "padres"
        if work_hours_band in {"Entre 21 y 30 horas", "Más de 30 horas"}:
            choice = "propio"
        elif estrato_num <= 1:
            choice = "beca"
        elif estrato_num == 2 and institution_character != "UNIVERSIDAD":
            choice = "credito"

        return {
            "estu_pagomatriculabeca": "Si" if choice == "beca" else "No",
            "estu_pagomatriculacredito": "Si" if choice == "credito" else "No",
            "estu_pagomatriculapadres": "Si" if choice == "padres" else "No",
            "estu_pagomatriculapropio": "Si" if choice == "propio" else "No",
        }

    def _prepare_input_data(self, exam_type: str, data: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prepared = dict(data or {})

        has_manual_scores = prepared.get("puntaje_matematicas") not in (None, "") and prepared.get("puntaje_lectura") not in (None, "")
        fallback_average = None
        if has_manual_scores:
            fallback_average = (
                self._safe_float(prepared.get("puntaje_matematicas"), 50.0)
                + self._safe_float(prepared.get("puntaje_lectura"), 50.0)
            ) / 2.0

        quartile_key = self._normalize_quartile(prepared.get("perfil_cuartil"), fallback_average)
        strength_key = self._normalize_strength(prepared.get("fortaleza_area"))
        score_profile = self._derive_score_profile(quartile_key, strength_key)

        if prepared.get("educacion_hogar"):
            prepared.setdefault("educacion_madre", prepared["educacion_hogar"])
            prepared.setdefault("educacion_padre", prepared["educacion_hogar"])

        estrato_num = self._parse_estrato_number(prepared.get("estrato", 2))
        prepared.setdefault("internet", self._infer_internet_input(estrato_num, quartile_key))
        prepared.setdefault("consumo_proteina", self._infer_protein_input(estrato_num, quartile_key))

        if not has_manual_scores:
            prepared["puntaje_matematicas"] = score_profile["matematicas"]
            prepared["puntaje_lectura"] = score_profile["lectura"]

        prepared["_quartile_key"] = quartile_key
        prepared["_strength_key"] = strength_key
        prepared["_subject_profile"] = score_profile
        prepared["_percentile_profile"] = self._derive_profile_percentiles(quartile_key, strength_key)

        profile_label = f"{REFERENCE_PAYLOAD['quartiles'][quartile_key]['label']} · {self._strength_label(strength_key)}"
        assumptions = {
            "perfil": profile_label,
            "cuartil": quartile_key.upper(),
            "fortaleza": self._strength_label(strength_key),
            "puntajes_estimados": score_profile,
            "referencia": REFERENCE_PAYLOAD[exam_type]["source_label"],
            "nota_referencia": REFERENCE_PAYLOAD[exam_type]["note"],
        }
        return prepared, assumptions

    def _estimate_reading_dedication(self, reading_score: float) -> str:
        if reading_score < 40:
            return "No leo por entretenimiento"
        if reading_score < 55:
            return "30 minutos o menos"
        if reading_score < 70:
            return "Entre 30 y 60 minutos"
        if reading_score < 85:
            return "Entre 1 y 2 horas"
        return "Más de 2 horas"

    def _estimate_subject_band(self, score: float) -> float:
        if score < 40:
            return 1.0
        if score < 55:
            return 2.0
        if score < 75:
            return 3.0
        return 4.0

    def _estimate_english_level(self, score: float) -> str:
        if score < 35:
            return "A-"
        if score < 50:
            return "A1"
        if score < 65:
            return "A2"
        if score < 80:
            return "B+"
        return "B1"

    def _categorize_saber11_global(self, score: float) -> str:
        if score < 200:
            return "Deficiente"
        if score < 250:
            return "Bajo"
        if score < 300:
            return "Regular"
        if score < 350:
            return "Alto"
        return "Excelente"

    def _build_base_row(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        state = artifact["preprocessor"].state
        if state is None:
            raise RuntimeError("El preprocesador del modelo no está ajustado.")

        row: Dict[str, Any] = {}
        for col in state.numeric_columns:
            row[col] = float(state.numeric_medians.get(col, 0.0))
        for col, mapping in state.ordinal_maps.items():
            row[col] = self._default_ordinal_value(mapping)
        for col in state.low_card_columns:
            levels = state.low_card_levels.get(col, [])
            row[col] = levels[0] if levels else "Missing"
        for col in state.high_card_columns:
            freq_map = state.high_card_frequency_maps.get(col, {})
            row[col] = max(freq_map, key=freq_map.get) if freq_map else "Missing"
        return row

    def _apply_household_defaults(
        self,
        row: Dict[str, Any],
        data: Dict[str, Any],
        estrato_num: int,
        math_score: float,
        reading_score: float,
    ) -> None:
        quartile_key = data.get("_quartile_key", "q2")
        quartile_rank = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}.get(quartile_key, 2)
        score_profile = data.get("_subject_profile", {})
        percentiles = data.get("_percentile_profile", {})

        mother_level = self._simplify_education_level(data.get("educacion_madre"))
        father_level = self._simplify_education_level(data.get("educacion_padre"))
        highest_level_rank = max(
            {"ninguno": 0, "primaria": 1, "secundaria": 2, "tecnica": 3, "universitaria": 4, "posgrado": 5}.get(mother_level, 0),
            {"ninguno": 0, "primaria": 1, "secundaria": 2, "tecnica": 3, "universitaria": 4, "posgrado": 5}.get(father_level, 0),
        )

        if "cole_caracter" in row:
            row["cole_caracter"] = "ACADÉMICO"
        if "cole_naturaleza" in row:
            row["cole_naturaleza"] = "NO OFICIAL" if estrato_num >= 4 else "OFICIAL"
        if "cole_jornada" in row:
            row["cole_jornada"] = "COMPLETA" if quartile_key == "q4" and estrato_num >= 4 else "MAÑANA"

        if "fami_numlibros" in row:
            if highest_level_rank <= 1 and quartile_rank <= 2:
                row["fami_numlibros"] = "0 A 10 LIBROS"
            elif highest_level_rank <= 2:
                row["fami_numlibros"] = "11 A 25 LIBROS"
            elif highest_level_rank <= 4 or quartile_rank <= 3:
                row["fami_numlibros"] = "26 A 100 LIBROS"
            else:
                row["fami_numlibros"] = "MÁS DE 100 LIBROS"

        if "fami_personashogar" in row:
            if estrato_num <= 1:
                row["fami_personashogar"] = "7 a 8" if quartile_key == "q1" else "5 a 6"
            elif estrato_num == 2:
                row["fami_personashogar"] = "5 a 6"
            elif estrato_num <= 4:
                row["fami_personashogar"] = "3 a 4"
            else:
                row["fami_personashogar"] = "1 a 2"

        if "fami_cuartoshogar" in row:
            if estrato_num <= 0:
                row["fami_cuartoshogar"] = "Uno"
            elif estrato_num == 1:
                row["fami_cuartoshogar"] = "Dos"
            elif estrato_num == 2:
                row["fami_cuartoshogar"] = "Tres"
            elif estrato_num <= 4:
                row["fami_cuartoshogar"] = "Cuatro"
            elif estrato_num == 5:
                row["fami_cuartoshogar"] = "Cinco"
            else:
                row["fami_cuartoshogar"] = "Seis o mas"

        if "fami_situacioneconomica" in row:
            if quartile_key == "q1" and estrato_num <= 2:
                row["fami_situacioneconomica"] = "Peor"
            elif quartile_key == "q4" and estrato_num >= 4:
                row["fami_situacioneconomica"] = "Mejor"
            else:
                row["fami_situacioneconomica"] = "Igual"

        protein = self._map_protein_consumption(data.get("consumo_proteina"))
        food_scale = [
            "Nunca o rara vez comemos eso",
            "1 o 2 veces por semana",
            "3 a 5 veces por semana",
            "Todos o casi todos los días",
        ]
        protein_idx = food_scale.index(protein) if protein in food_scale else 2
        cereal_idx = min(3, protein_idx + (1 if quartile_rank >= 3 else 0))
        dairy_idx = min(3, protein_idx + (1 if estrato_num >= 3 else 0))

        if "fami_comecerealfrutoslegumbre" in row:
            row["fami_comecerealfrutoslegumbre"] = food_scale[cereal_idx]
        if "fami_comelechederivados" in row:
            row["fami_comelechederivados"] = food_scale[dairy_idx]

        internet_usage = self._map_internet_usage(data.get("internet"))
        has_computer = internet_usage != "No Navega Internet" and (estrato_num >= 2 or quartile_rank >= 3)
        has_car = estrato_num >= 4
        has_motorcycle = 2 <= estrato_num <= 4
        has_microwave = estrato_num >= 3
        has_console = quartile_key == "q4" and estrato_num >= 3

        for col, value in {
            "fami_tieneautomovil_saber11": "Si" if has_car else "No",
            "fami_tienecomputador_saber11": "Si" if has_computer else "No",
            "fami_tieneconsolavideojuegos_saber11": "Si" if has_console else "No",
            "fami_tienehornomicroogas_saber11": "Si" if has_microwave else "No",
            "fami_tienemotocicleta_saber11": "Si" if has_motorcycle else "No",
            "fami_tieneautomovil_saberpro": "Si" if has_car else "No",
            "fami_tienehornomicroogas_saberpro": "Si" if has_microwave else "No",
            "fami_tienemotocicleta_saberpro": "Si" if has_motorcycle else "No",
            "fami_tieneserviciotv_saberpro": "Si" if estrato_num >= 1 else "No",
        }.items():
            if col in row:
                row[col] = value

        mother_occupation = self._infer_parent_occupation(mother_level, "mother")
        father_occupation = self._infer_parent_occupation(father_level, "father")
        mother_work = self._infer_parent_work(mother_level)
        father_work = self._infer_parent_work(father_level)
        if "fami_trabajolabormadre_saber11" in row:
            row["fami_trabajolabormadre_saber11"] = mother_work
        if "fami_trabajolaborpadre_saber11" in row:
            row["fami_trabajolaborpadre_saber11"] = father_work
        if "fami_trabajolabormadre_saberpro" in row:
            row["fami_trabajolabormadre_saberpro"] = mother_work
        if "fami_trabajolaborpadre_saberpro" in row:
            row["fami_trabajolaborpadre_saberpro"] = father_work

        if "fami_ocupacionmadre" in row:
            row["fami_ocupacionmadre"] = mother_occupation
        if "fami_ocupacionpadre" in row:
            row["fami_ocupacionpadre"] = father_occupation

        if "estu_generacione" in row:
            row["estu_generacione"] = "GENERACION E - GRATUIDAD" if estrato_num <= 2 and quartile_rank >= 3 else "NO"

        if "inst_origen" in row:
            row["inst_origen"] = "OFICIAL MUNICIPAL" if estrato_num <= 2 else "NO OFICIAL - FUNDACIÓN"

        if "estu_inse_individual_saberpro" in row:
            row["estu_inse_individual_saberpro"] = self._estimate_inse(estrato_num)

        if "punt_c_naturales" in row and score_profile:
            row["punt_c_naturales"] = score_profile.get("ciencias", (math_score + reading_score) / 2.0)
        if "punt_sociales_ciudadanas" in row and score_profile:
            row["punt_sociales_ciudadanas"] = score_profile.get("sociales", (math_score + reading_score) / 2.0)
        if "punt_ingles" in row and score_profile:
            row["punt_ingles"] = score_profile.get("ingles", reading_score)

        for col, subject in {
            "percentil_c_naturales": "ciencias",
            "percentil_matematicas": "matematicas",
            "percentil_lectura_critica": "lectura",
            "percentil_sociales_ciudadanas": "sociales",
            "percentil_ingles": "ingles",
        }.items():
            if col in row and percentiles:
                row[col] = percentiles.get(subject, REFERENCE_PAYLOAD["quartiles"][quartile_key]["percentile"])

    def _apply_common_inputs(self, row: Dict[str, Any], data: Dict[str, Any]) -> Tuple[int, float, float]:
        """Aplica mapeos comunes a Saber 11 y Saber Pro."""
        # Extraer estrato (manejar múltiples nombres posibles)
        val_estrato = self._first_present(
            data.get("estrato"),
            data.get("fami_estratovivienda_saber11"),
            data.get("fami_estratovivienda_saberpro"),
        )
        estrato_num = self._parse_estrato_number(val_estrato if val_estrato is not None else 2)
        estrato_label = self._format_estrato_label(estrato_num)
        nse_label = self._map_nse_label(estrato_num)
        inse_value = self._estimate_inse(estrato_num)
        score_profile = data.get("_subject_profile", {})
        
        # Mapear puntajes (ICFES usa punt_matematicas, internamente se usa puntaje_matematicas)
        math_score = self._clamp(
            self._safe_float(
                data.get("puntaje_matematicas") or data.get("punt_matematicas") or score_profile.get("matematicas", 50), 
                50.0
            ), 
            0.0, 100.0
        )
        reading_score = self._clamp(
            self._safe_float(
                data.get("puntaje_lectura") or data.get("punt_lectura_critica") or score_profile.get("lectura", 50), 
                50.0
            ), 
            0.0, 100.0
        )

        for col in ("fami_estratovivienda_saber11", "fami_estratovivienda_saberpro"):
            if col in row:
                row[col] = estrato_label

        for col in ("estu_nse_establecimiento", "estu_nse_individual_saber11", "estu_nse_individual_saberpro"):
            if col in row:
                row[col] = nse_label

        for col in ("estu_inse_individual_saber11", "estu_inse_individual_saberpro"):
            if col in row:
                row[col] = inse_value

        if "estu_nse_ies" in row:
            row["estu_nse_ies"] = float(nse_label[-1])

        # Educación padres
        mother_education = self._map_parent_education(
            data.get("educacion_madre") or data.get("fami_educacionmadre_saber11") or data.get("fami_educacionmadre_saberpro")
        )
        father_education = self._map_parent_education(
            data.get("educacion_padre") or data.get("fami_educacionpadre_saber11") or data.get("fami_educacionpadre_saberpro")
        )
        for col in ("fami_educacionmadre_saber11", "fami_educacionmadre_saberpro"):
            if col in row:
                row[col] = mother_education
        for col in ("fami_educacionpadre_saber11", "fami_educacionpadre_saberpro"):
            if col in row:
                row[col] = father_education

        if "estu_dedicacioninternet" in row:
            val_internet = self._first_present(data.get("internet"), data.get("estu_dedicacioninternet"))
            row["estu_dedicacioninternet"] = self._map_internet_usage(val_internet)
        
        if "fami_comecarnepescadohuevo" in row:
            val_protein = self._first_present(data.get("consumo_proteina"), data.get("fami_comecarnepescadohuevo"))
            row["fami_comecarnepescadohuevo"] = self._map_protein_consumption(val_protein)
            
        if "estu_dedicacionlecturadiaria" in row:
            val_reading = data.get("estu_dedicacionlecturadiaria")
            row["estu_dedicacionlecturadiaria"] = self._map_reading_dedication(val_reading, reading_score)

        self._apply_household_defaults(row, data, estrato_num, math_score, reading_score)

        return estrato_num, math_score, reading_score

    def _apply_saber11_inputs(self, row: Dict[str, Any], data: Dict[str, Any]) -> None:
        _, _, reading_score = self._apply_common_inputs(row, data)
        age = self._clamp(self._safe_float(self._first_present(data.get("edad_saber11"), data.get("edad")), 17.0), 14.0, 25.0)

        if "edad_saber11" in row:
            row["edad_saber11"] = age
        if "estu_dedicacionlecturadiaria" in row:
            row["estu_dedicacionlecturadiaria"] = self._estimate_reading_dedication(reading_score)

    def _apply_saberpro_inputs(self, row: Dict[str, Any], data: Dict[str, Any]) -> None:
        _, math_score, reading_score = self._apply_common_inputs(row, data)
        age = self._clamp(self._safe_float(self._first_present(data.get("edad_saberpro"), data.get("edad")), 21.0), 16.0, 100.0)
        score_profile = data.get("_subject_profile", {})
        percentile_profile = data.get("_percentile_profile", {})
        science_score = self._safe_float(score_profile.get("ciencias", (math_score + reading_score) / 2.0), (math_score + reading_score) / 2.0)
        social_score = self._safe_float(score_profile.get("sociales", (math_score + reading_score) / 2.0), (math_score + reading_score) / 2.0)
        english_score = self._safe_float(score_profile.get("ingles", reading_score), reading_score)
        average_score = round((math_score + reading_score + science_score + social_score + english_score) / 5.0, 2)
        global_saber11 = round(average_score * 5.0, 2)

        if "edad_saberpro" in row:
            row["edad_saberpro"] = age
        if "edad_saber11" in row:
            row["edad_saber11"] = max(14.0, age - 5.0)

        numeric_updates = {
            "punt_matematicas": math_score,
            "punt_lectura_critica": reading_score,
            "punt_c_naturales": science_score,
            "punt_sociales_ciudadanas": social_score,
            "punt_ingles": english_score,
            "punt_global_saber11": global_saber11,
            "percentil_matematicas": percentile_profile.get("matematicas", round(math_score)),
            "percentil_lectura_critica": percentile_profile.get("lectura", round(reading_score)),
            "percentil_c_naturales": percentile_profile.get("ciencias", round(science_score)),
            "percentil_sociales_ciudadanas": percentile_profile.get("sociales", round(social_score)),
            "percentil_ingles": percentile_profile.get("ingles", round(english_score)),
            "percentil_global_saber11": REFERENCE_PAYLOAD["quartiles"][data.get("_quartile_key", "q2")]["percentile"],
            "desemp_matematicas": self._estimate_subject_band(math_score),
            "desemp_lectura_critica": self._estimate_subject_band(reading_score),
            "desemp_c_naturales": self._estimate_subject_band(science_score),
            "desemp_sociales_ciudadanas": self._estimate_subject_band(social_score),
        }
        for col, value in numeric_updates.items():
            if col in row:
                row[col] = value

        if "desemp_ingles" in row:
            row["desemp_ingles"] = self._estimate_english_level(english_score)
        if "nivel_saber11" in row:
            row["nivel_saber11"] = data.get("nivel_saber11") or self._categorize_saber11_global(global_saber11)
        if "estu_horassemanatrabaja_saberpro" in row:
            val_work = self._first_present(data.get("estu_horassemanatrabaja_saberpro"), data.get("horas_trabajo_semanal"))
            row["estu_horassemanatrabaja_saberpro"] = self._map_work_hours(val_work)
        if "estu_semestrecursa" in row:
            val_sem = self._first_present(data.get("estu_semestrecursa"), data.get("semestre_actual"))
            row["estu_semestrecursa"] = self._format_semester(val_sem)
        if "inst_caracter_academico" in row:
            val_inst = self._first_present(data.get("inst_caracter_academico"), data.get("caracter_institucion"))
            row["inst_caracter_academico"] = self._map_institution_character(val_inst)
        if "inst_origen" in row:
            row["inst_origen"] = data.get("inst_origen") or ("OFICIAL MUNICIPAL" if self._parse_estrato_number(data.get("estrato", 2)) <= 2 else "NO OFICIAL - FUNDACIÓN")

        payment_flags = self._derive_tuition_payment_flags(
            self._parse_estrato_number(data.get("estrato", 2)),
            self._map_work_hours(self._first_present(data.get("horas_trabajo_semanal"), data.get("estu_horassemanatrabaja_saberpro"))),
            self._map_institution_character(self._first_present(data.get("caracter_institucion"), data.get("inst_caracter_academico"))),
        )
        for col, value in payment_flags.items():
            if col in row:
                row[col] = value

    def _build_legacy_feature_frame(self, exam_type: str, data: Dict[str, Any]) -> pd.DataFrame:
        base = {
            "estrato": self._parse_estrato_number(data.get("estrato", 2)),
            "puntaje_matematicas": self._safe_float(data.get("puntaje_matematicas", 50), 50.0),
            "puntaje_lectura": self._safe_float(data.get("puntaje_lectura", 50), 50.0),
            "edad": self._safe_int(data.get("edad", 17 if exam_type == "saber11" else 21), 17 if exam_type == "saber11" else 21),
        }
        if exam_type == "saberpro":
            base["semestre_actual"] = self._safe_int(data.get("semestre_actual", 6), 6)
        return pd.DataFrame([base])

    def _build_feature_frame(self, model_key: str, data: Dict[str, Any]) -> pd.DataFrame:
        artifact = self.models[model_key]
        if not self._is_bundle(artifact):
            exam_type = "saber11" if model_key.startswith("saber11") else "saberpro"
            return self._build_legacy_feature_frame(exam_type, data)

        row = self._build_base_row(artifact)
        if model_key.startswith("saber11"):
            self._apply_saber11_inputs(row, data)
        else:
            self._apply_saberpro_inputs(row, data)

        # Si el usuario diligencia una variable exacta del modelo, esa respuesta
        # debe prevalecer sobre cualquier valor inferido internamente.
        for key, value in data.items():
            if key in row and value not in (None, ""):
                row[key] = value

        return pd.DataFrame([row])

    def _prepare_features_for_prediction(self, artifact: Any, features: pd.DataFrame) -> Any:
        if not self._is_bundle(artifact):
            return features

        transformed = artifact["preprocessor"].transform(features)
        if hasattr(transformed, "to_numpy"):
            transformed = transformed.to_numpy(dtype=np.float32, copy=False)
        else:
            transformed = np.asarray(transformed, dtype=np.float32)

        scaler = artifact.get("scaler")
        if scaler is not None:
            transformed = scaler.transform(transformed).astype(np.float32, copy=False)

        return transformed

    def _predict_with_artifact(self, artifact: Any, features: pd.DataFrame) -> Tuple[np.ndarray, Optional[float]]:
        prepared = self._prepare_features_for_prediction(artifact, features)
        predictor = artifact["model"] if self._is_bundle(artifact) else artifact
        predictions = np.asarray(predictor.predict(prepared)).reshape(-1)

        confidence = None
        if hasattr(predictor, "predict_proba"):
            try:
                probabilities = np.asarray(predictor.predict_proba(prepared))
                if probabilities.ndim == 2 and len(probabilities) > 0:
                    confidence = float(np.max(probabilities[0]))
            except Exception:
                confidence = None

        return predictions, confidence

    def _predict_score(self, model_key: str, data: Dict[str, Any]) -> float:
        artifact = self.models[model_key]
        features = self._build_feature_frame(model_key, data)
        predictions, _ = self._predict_with_artifact(artifact, features)
        return float(predictions[0])

    def _predict_label(self, model_key: str, data: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        artifact = self.models[model_key]
        features = self._build_feature_frame(model_key, data)
        predictions, confidence = self._predict_with_artifact(artifact, features)
        raw_label = predictions[0]

        if self._is_bundle(artifact) and artifact.get("encoder") is not None:
            label = artifact["encoder"].inverse_transform(np.asarray([int(raw_label)], dtype=int))[0]
            return str(label), confidence

        return str(raw_label), confidence

    def _get_exam_config(self, exam_type: str) -> Dict[str, int]:
        if exam_type == "saberpro":
            return {"max_score": 300, "medium": 150, "high": 190}
        return {"max_score": 500, "medium": 250, "high": 350}

    def _map_risk_from_score(self, score: int, exam_type: str) -> str:
        config = self._get_exam_config(exam_type)
        if score >= config["high"]:
            return "BAJO"
        if score >= config["medium"]:
            return "MEDIO"
        return "ALTO"

    def _map_risk_from_label(self, label: Optional[str], score: int, exam_type: str) -> str:
        mapping = {
            "deficiente": "ALTO",
            "bajo": "ALTO",
            "regular": "MEDIO",
            "alto": "BAJO",
            "excelente": "BAJO",
            "bueno": "BAJO",
        }
        if label:
            normalized = str(label).strip().lower()
            if normalized in mapping:
                return mapping[normalized]
        return self._map_risk_from_score(score, exam_type)

    def _calibrate_regression_score(self, exam_type: str, score: float) -> float:
        if exam_type == "saber11" and score < 100:
            return score + 300.0
        return score

    def _build_prediction_response(
        self,
        exam_type: str,
        score: float,
        assumptions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = self._get_exam_config(exam_type)
        score = self._calibrate_regression_score(exam_type, score)
        score = int(round(self._clamp(score, 0, config["max_score"])))

        if score >= config["high"]:
            performance = "Desempeño Sobresaliente"
            color = "green"
        elif score >= config["medium"]:
            performance = "Desempeño Promedio"
            color = "blue"
        else:
            performance = "Riesgo de Deserción"
            color = "red"

        risk = self._map_risk_from_score(score, exam_type)
        
        return {
            "success": True,
            "puntaje_estimado": score,
            "riesgo": risk,
            "color_riesgo": self._get_risk_color(risk),
            "percentil": int(round((score / config["max_score"]) * 100)),
            "desempenio_esperado": performance,
            "color_desempenio": color,
        }

    def _get_risk_color(self, risk: str) -> str:
        colors = {"BAJO": "green", "MEDIO": "yellow", "ALTO": "red"}
        return colors.get(risk, "gray")

    def _get_defaults(self, exam_type: str) -> Dict[str, Any]:
        # No se retornan valores por defecto, el usuario debe ingresar todos los datos
        return {}

    def _map_bachelor_title(self, value: Any) -> str:
        cleaned = self._clean_text(value).lower()
        if "técn" in cleaned or "tecn" in cleaned:
            return "Bachiller técnico"
        if "pedag" in cleaned or "normalista" in cleaned:
            return "Bachiller pedagógico o normalista"
        return "Bachiller académico"

    def _normalize_nse_label(self, value: Any, default: str = "NSE3") -> str:
        cleaned = self._clean_text(value).upper().replace(" ", "")
        if cleaned in {"NSE1", "1"}:
            return "NSE1"
        if cleaned in {"NSE2", "2"}:
            return "NSE2"
        if cleaned in {"NSE3", "3"}:
            return "NSE3"
        if cleaned in {"NSE4", "4"}:
            return "NSE4"
        return default

    def _normalize_school_character(self, value: Any) -> str:
        cleaned = self._clean_text(value).upper()
        return {
            "TECNICO": "TÉCNICO",
            "TÉCNICO": "TÉCNICO",
            "ACADEMICO": "ACADÉMICO",
            "ACADÉMICO": "ACADÉMICO",
            "TECNICO/ACADEMICO": "TÉCNICO/ACADÉMICO",
            "TÉCNICO/ACADÉMICO": "TÉCNICO/ACADÉMICO",
            "NO APLICA": "NO APLICA",
        }.get(cleaned, "ACADÉMICO")

    def _normalize_school_shift(self, value: Any) -> str:
        cleaned = self._clean_text(value).upper()
        return {
            "UNICA": "UNICA",
            "ÚNICA": "UNICA",
            "MANANA": "MAÑANA",
            "MAÑANA": "MAÑANA",
            "TARDE": "TARDE",
            "NOCHE": "NOCHE",
            "SABATINA": "SABATINA",
            "COMPLETA": "COMPLETA",
        }.get(cleaned, "MAÑANA")

    def _normalize_institution_origin(self, value: Any) -> str:
        cleaned = self._clean_text(value).upper()
        mapping = {
            "PRIVADA": "NO OFICIAL - FUNDACIÓN",
            "NO OFICIAL": "NO OFICIAL - FUNDACIÓN",
            "NO OFICIAL (PRIVADA)": "NO OFICIAL - FUNDACIÓN",
            "NO OFICIAL - FUNDACION": "NO OFICIAL - FUNDACIÓN",
            "NO OFICIAL - FUNDACIÓN": "NO OFICIAL - FUNDACIÓN",
            "NO OFICIAL - CORPORACION": "NO OFICIAL - CORPORACIÓN",
            "NO OFICIAL - CORPORACIÓN": "NO OFICIAL - CORPORACIÓN",
            "OFICIAL DEPARTAMENTAL": "OFICIAL DEPARTAMENTAL",
            "OFICIAL NACIONAL": "OFICIAL NACIONAL",
            "OFICIAL MUNICIPAL": "OFICIAL MUNICIPAL",
            "REGIMEN ESPECIAL": "REGIMEN ESPECIAL",
        }
        return mapping.get(cleaned, "NO OFICIAL - FUNDACIÓN")

    def _normalize_generation_e(self, value: Any) -> str:
        cleaned = self._clean_text(value).upper()
        if cleaned in {"SI", "SÍ", "GENERACION E", "GENERACIÓN E", "GENERACION E - GRATUIDAD"}:
            return "GENERACION E - GRATUIDAD"
        if "EXCELENCIA DEPARTAMENTAL" in cleaned:
            return "GENERACION E - EXCELENCIA DEPARTAMENTAL"
        if "EXCELENCIA NACIONAL" in cleaned:
            return "GENERACION E - EXCELENCIA NACIONAL"
        return "NO"

    def _normalize_prediction_payload(self, exam_type: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in (data or {}).items():
            normalized[key] = self._clean_text(value) if isinstance(value, str) else value

        estrato_value = self._first_present(
            normalized.get("estrato"),
            normalized.get("fami_estratovivienda_saber11"),
            normalized.get("fami_estratovivienda_saberpro"),
        )
        if estrato_value is not None:
            estrato_num = self._parse_estrato_number(estrato_value)
            normalized["estrato"] = estrato_num
            if exam_type == "saber11" or "fami_estratovivienda_saber11" in normalized:
                normalized["fami_estratovivienda_saber11"] = self._format_estrato_label(estrato_num)
            if exam_type == "saberpro" or "fami_estratovivienda_saberpro" in normalized:
                normalized["fami_estratovivienda_saberpro"] = self._format_estrato_label(estrato_num)

        for source, target in (
            ("educacion_madre", "fami_educacionmadre_saber11"),
            ("educacion_padre", "fami_educacionpadre_saber11"),
            ("educacion_madre", "fami_educacionmadre_saberpro"),
            ("educacion_padre", "fami_educacionpadre_saberpro"),
        ):
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]

        for field in (
            "fami_educacionmadre_saber11",
            "fami_educacionpadre_saber11",
            "fami_educacionmadre_saberpro",
            "fami_educacionpadre_saberpro",
        ):
            if field in normalized:
                normalized[field] = self._map_parent_education(normalized[field])

        if "internet" in normalized and "estu_dedicacioninternet" not in normalized:
            normalized["estu_dedicacioninternet"] = normalized["internet"]
        if "estu_dedicacioninternet" in normalized:
            normalized["estu_dedicacioninternet"] = self._map_internet_usage(normalized["estu_dedicacioninternet"])

        if "consumo_proteina" in normalized and "fami_comecarnepescadohuevo" not in normalized:
            normalized["fami_comecarnepescadohuevo"] = normalized["consumo_proteina"]
        for field in ("fami_comecarnepescadohuevo", "fami_comecerealfrutoslegumbre", "fami_comelechederivados"):
            if field in normalized:
                normalized[field] = self._map_protein_consumption(normalized[field])

        if "fami_personashogar" in normalized:
            normalized["fami_personashogar"] = self._map_household_people(normalized["fami_personashogar"])
        if "fami_cuartoshogar" in normalized:
            normalized["fami_cuartoshogar"] = self._map_rooms(normalized["fami_cuartoshogar"])
        if "fami_numlibros" in normalized:
            normalized["fami_numlibros"] = self._map_books(normalized["fami_numlibros"])
        if "fami_situacioneconomica" in normalized:
            normalized["fami_situacioneconomica"] = self._map_economic_situation(normalized["fami_situacioneconomica"])

        if "estu_dedicacionlecturadiaria" in normalized:
            lectura = self._first_present(normalized.get("puntaje_lectura"), normalized.get("punt_lectura_critica"), 50)
            normalized["estu_dedicacionlecturadiaria"] = self._map_reading_dedication(normalized["estu_dedicacionlecturadiaria"], self._safe_float(lectura, 50.0))

        age_11 = self._first_present(normalized.get("edad_saber11"), normalized.get("edad"))
        age_pro = self._first_present(normalized.get("edad_saberpro"), normalized.get("edad"))
        if exam_type == "saber11" and age_11 is not None:
            normalized["edad_saber11"] = self._safe_float(age_11, 17.0)
            normalized["edad"] = normalized["edad_saber11"]
        if exam_type == "saberpro":
            if age_pro is not None:
                normalized["edad_saberpro"] = self._safe_float(age_pro, 21.0)
                normalized["edad"] = normalized["edad_saberpro"]
            if age_11 is not None:
                normalized["edad_saber11"] = self._safe_float(age_11, max(14.0, self._safe_float(age_pro, 21.0) - 5.0))

        for field in ("estu_genero_saber11", "estu_genero_saberpro"):
            if field in normalized:
                normalized[field] = self._map_gender(normalized[field])

        for field in (
            "fami_tieneautomovil_saber11",
            "fami_tienecomputador_saber11",
            "fami_tieneconsolavideojuegos_saber11",
            "fami_tienehornomicroogas_saber11",
            "fami_tienemotocicleta_saber11",
            "estu_pagomatriculabeca",
            "estu_pagomatriculacredito",
            "estu_pagomatriculapadres",
            "estu_pagomatriculapropio",
            "fami_tieneautomovil_saberpro",
            "fami_tienehornomicroogas_saberpro",
            "fami_tienemotocicleta_saberpro",
            "fami_tieneserviciotv_saberpro",
        ):
            if field in normalized:
                normalized[field] = self._map_yes_no(normalized[field])

        for field in ("estu_nse_establecimiento", "estu_nse_individual_saber11", "estu_nse_individual_saberpro"):
            if field in normalized:
                normalized[field] = self._normalize_nse_label(normalized[field])
        if "estu_nse_ies" in normalized and not self._is_blank(normalized["estu_nse_ies"]):
            nse_ies = self._clean_text(normalized["estu_nse_ies"]).replace("NSE", "").replace("nse", "")
            normalized["estu_nse_ies"] = self._clamp(self._safe_float(nse_ies, 3.0), 1.0, 4.0)

        if "cole_caracter" in normalized:
            normalized["cole_caracter"] = self._normalize_school_character(normalized["cole_caracter"])
        if "cole_jornada" in normalized:
            normalized["cole_jornada"] = self._normalize_school_shift(normalized["cole_jornada"])
        if "cole_naturaleza" in normalized:
            naturaleza = self._clean_text(normalized["cole_naturaleza"]).upper()
            normalized["cole_naturaleza"] = "OFICIAL" if naturaleza == "OFICIAL" else "NO OFICIAL"
        if "inst_caracter_academico" in normalized or "caracter_institucion" in normalized:
            inst_value = self._first_present(normalized.get("inst_caracter_academico"), normalized.get("caracter_institucion"))
            normalized["inst_caracter_academico"] = self._map_institution_character(inst_value)
            normalized["caracter_institucion"] = normalized["inst_caracter_academico"]
        if "inst_origen" in normalized:
            normalized["inst_origen"] = self._normalize_institution_origin(normalized["inst_origen"])
        if "estu_generacione" in normalized:
            normalized["estu_generacione"] = self._normalize_generation_e(normalized["estu_generacione"])
        if "estu_semestrecursa" in normalized or "semestre_actual" in normalized:
            sem_value = self._first_present(normalized.get("estu_semestrecursa"), normalized.get("semestre_actual"))
            normalized["estu_semestrecursa"] = self._format_semester(sem_value)
            normalized["semestre_actual"] = self._safe_int(sem_value, 6)
        if "estu_horassemanatrabaja_saberpro" in normalized or "horas_trabajo_semanal" in normalized:
            work_value = self._first_present(normalized.get("estu_horassemanatrabaja_saberpro"), normalized.get("horas_trabajo_semanal"))
            normalized["estu_horassemanatrabaja_saberpro"] = self._map_work_hours(work_value)
            normalized["horas_trabajo_semanal"] = normalized["estu_horassemanatrabaja_saberpro"]
        if "estu_tituloobtenidobachiller" in normalized:
            normalized["estu_tituloobtenidobachiller"] = self._map_bachelor_title(normalized["estu_tituloobtenidobachiller"])

        if exam_type == "saberpro":
            program = self._first_present(normalized.get("estu_prgm_academico"), normalized.get("estu_nucleo_pregrado"))
            if program is not None:
                normalized["estu_prgm_academico"] = program
                normalized["estu_nucleo_pregrado"] = program

        if exam_type == "saber11":
            if self._is_blank(normalized.get("estu_depto_presentacion_saber11")):
                normalized["estu_depto_presentacion_saber11"] = self._first_present(
                    normalized.get("estu_depto_reside"),
                    normalized.get("cole_depto_ubicacion"),
                )
            if self._is_blank(normalized.get("estu_mcpio_presentacion_saber11")):
                normalized["estu_mcpio_presentacion_saber11"] = self._first_present(
                    normalized.get("estu_mcpio_reside"),
                    normalized.get("cole_mcpio_ubicacion"),
                )

        for source, target in (
            ("puntaje_matematicas", "punt_matematicas"),
            ("puntaje_lectura", "punt_lectura_critica"),
            ("puntaje_global_saber11", "punt_global_saber11"),
        ):
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
            if target in normalized and source not in normalized:
                normalized[source] = normalized[target]

        if "punt_global_saber11" in normalized and not self._is_blank(normalized["punt_global_saber11"]):
            score = self._safe_float(normalized["punt_global_saber11"], 300.0)
            normalized["punt_global_saber11"] = score
            normalized["puntaje_global_saber11"] = score
            normalized["nivel_saber11"] = self._categorize_saber11_global(score)

        for field in ("fami_ocupacionmadre", "fami_ocupacionpadre"):
            if field in normalized:
                parent = "mother" if field.endswith("madre") else "father"
                edu = normalized.get("fami_educacionmadre_saber11" if parent == "mother" else "fami_educacionpadre_saber11")
                normalized[field] = self._map_parent_occupation(normalized[field], parent, self._simplify_education_level(edu))

        self._normalize_inse_inputs(normalized)
        self._sync_parent_work_with_occupation(normalized)
        return normalized

    def _add_required_errors(self, data: Dict[str, Any], fields: Dict[str, str], errors: List[str]) -> None:
        for key, label in fields.items():
            if self._is_blank(data.get(key)):
                errors.append(f"{label} es obligatorio")

    def _add_numeric_range_error(
        self,
        data: Dict[str, Any],
        key: str,
        label: str,
        minimum: float,
        maximum: float,
        errors: List[str],
        *,
        required: bool = False,
    ) -> None:
        value = data.get(key)
        if self._is_blank(value):
            if required:
                errors.append(f"{label} es obligatorio")
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} debe ser numerico")
            return
        if numeric < minimum or numeric > maximum:
            errors.append(f"{label} debe estar entre {minimum:g} y {maximum:g}")

    def _validate_prediction_payload(self, exam_type: str, data: Dict[str, Any], *, strict: bool) -> List[str]:
        errors: List[str] = []

        if strict and exam_type == "saber11":
            self._add_required_errors(
                data,
                {
                    "cole_caracter": "Caracter del colegio",
                    "cole_depto_ubicacion": "Departamento del colegio",
                    "cole_mcpio_ubicacion": "Municipio del colegio",
                    "cole_jornada": "Jornada del colegio",
                    "cole_naturaleza": "Naturaleza del colegio",
                    "estu_genero_saber11": "Genero",
                    "estu_depto_reside": "Departamento de residencia",
                    "estu_mcpio_reside": "Municipio de residencia",
                    "fami_estratovivienda_saber11": "Estrato de vivienda",
                    "fami_educacionmadre_saber11": "Educacion de la madre",
                    "fami_educacionpadre_saber11": "Educacion del padre",
                    "estu_dedicacioninternet": "Dedicacion a internet",
                    "estu_dedicacionlecturadiaria": "Dedicacion a lectura",
                    "fami_comecarnepescadohuevo": "Consumo de proteina",
                    "fami_personashogar": "Personas en el hogar",
                    "fami_situacioneconomica": "Situacion economica",
                },
                errors,
            )
            self._add_numeric_range_error(data, "edad_saber11", "Edad", 14, 25, errors, required=True)

        if strict and exam_type == "saberpro":
            self._add_required_errors(
                data,
                {
                    "estu_genero_saberpro": "Genero",
                    "estu_semestrecursa": "Semestre",
                    "inst_caracter_academico": "Caracter de la institucion",
                    "inst_origen": "Origen de la institucion",
                    "estu_inst_departamento": "Departamento de la IES",
                    "estu_inst_municipio": "Municipio de la IES",
                    "fami_estratovivienda_saberpro": "Estrato de vivienda",
                    "fami_educacionmadre_saberpro": "Educacion de la madre",
                    "fami_educacionpadre_saberpro": "Educacion del padre",
                    "punt_global_saber11": "Puntaje global Saber 11",
                    "punt_matematicas": "Puntaje de matematicas",
                    "punt_lectura_critica": "Puntaje de lectura critica",
                    "punt_c_naturales": "Puntaje de ciencias naturales",
                    "punt_sociales_ciudadanas": "Puntaje de sociales y ciudadanas",
                    "punt_ingles": "Puntaje de ingles",
                    "estu_prgm_academico": "Programa academico",
                    "estu_nucleo_pregrado": "Nucleo de pregrado",
                    "estu_depto_presentacion_saberpro": "Departamento de presentacion Saber Pro",
                    "estu_mcpio_presentacion_saberpro": "Municipio de presentacion Saber Pro",
                },
                errors,
            )
            self._add_numeric_range_error(data, "edad_saberpro", "Edad Saber Pro", 16, 100, errors, required=True)
            self._add_numeric_range_error(data, "edad_saber11", "Edad en Saber 11", 14, 30, errors, required=True)

        if exam_type == "saber11":
            if not strict:
                self._add_numeric_range_error(data, "edad_saber11", "Edad", 14, 25, errors)
        else:
            if not strict:
                self._add_numeric_range_error(data, "edad_saberpro", "Edad Saber Pro", 16, 100, errors)
                self._add_numeric_range_error(data, "edad_saber11", "Edad en Saber 11", 14, 30, errors)

        global_score_key = "punt_global_saber11" if not self._is_blank(data.get("punt_global_saber11")) else "puntaje_global_saber11"
        self._add_numeric_range_error(data, global_score_key, "Puntaje global Saber 11", 0, 500, errors)

        subject_score_fields = [
            (("punt_matematicas", "puntaje_matematicas"), "Puntaje de matematicas"),
            (("punt_lectura_critica", "puntaje_lectura"), "Puntaje de lectura critica"),
            (("punt_c_naturales",), "Puntaje de ciencias naturales"),
            (("punt_sociales_ciudadanas",), "Puntaje de sociales y ciudadanas"),
            (("punt_ingles",), "Puntaje de ingles"),
        ]
        for keys, label in subject_score_fields:
            key = next((candidate for candidate in keys if not self._is_blank(data.get(candidate))), keys[0])
            self._add_numeric_range_error(data, key, label, 0, 100, errors)

        for field, label in {
            "estu_inse_individual_saber11": "INSE Saber 11",
            "estu_inse_individual_saberpro": "INSE Saber Pro",
        }.items():
            self._add_numeric_range_error(data, field, label, 0, 100, errors)

        for field in (
            "percentil_c_naturales",
            "percentil_global_saber11",
            "percentil_ingles",
            "percentil_lectura_critica",
            "percentil_matematicas",
            "percentil_sociales_ciudadanas",
        ):
            self._add_numeric_range_error(data, field, field.replace("_", " "), 0, 100, errors)

        for field in ("desemp_c_naturales", "desemp_lectura_critica", "desemp_matematicas", "desemp_sociales_ciudadanas"):
            self._add_numeric_range_error(data, field, field.replace("_", " "), 1, 4, errors)

        if not self._is_blank(data.get("estu_nse_ies")):
            self._add_numeric_range_error(data, "estu_nse_ies", "NSE de la IES", 1, 4, errors)

        if exam_type == "saberpro" and not self._is_blank(data.get("edad_saber11")) and not self._is_blank(data.get("edad_saberpro")):
            if self._safe_float(data.get("edad_saber11"), 17.0) > self._safe_float(data.get("edad_saberpro"), 21.0):
                errors.append("La edad en Saber 11 no puede ser mayor que la edad en Saber Pro")

        return errors

    def predict_saber11(self, data: Dict[str, Any], *, strict: bool = True) -> Dict[str, Any]:
        try:
            full_data = self._get_defaults("saber11")
            full_data.update(self._normalize_prediction_payload("saber11", data))

            validation_errors = self._validate_prediction_payload("saber11", full_data, strict=strict)
            if validation_errors:
                return {"success": False, "error": "Corrige estos campos: " + "; ".join(validation_errors)}

            if "saber11_regression" not in self.models:
                detail = self.load_errors.get("saber11_regression")
                suffix = f" ({detail})" if detail else ""
                return {"success": False, "error": "Modelo de regresion Saber 11 no disponible" + suffix}

            score = self._predict_score("saber11_regression", full_data)
            return self._build_prediction_response("saber11", score, full_data)
        except Exception as exc:
            print(f"Error en predict_saber11: {exc}")
            return {"success": False, "error": str(exc)}

    def predict_saberpro(self, data: Dict[str, Any], *, strict: bool = True) -> Dict[str, Any]:
        try:
            full_data = self._get_defaults("saberpro")
            full_data.update(self._normalize_prediction_payload("saberpro", data))

            validation_errors = self._validate_prediction_payload("saberpro", full_data, strict=strict)
            if validation_errors:
                return {"success": False, "error": "Corrige estos campos: " + "; ".join(validation_errors)}

            if "saberpro_regression" not in self.models:
                detail = self.load_errors.get("saberpro_regression")
                suffix = f" ({detail})" if detail else ""
                return {"success": False, "error": "Modelo de regresion Saber Pro no disponible" + suffix}

            score = self._predict_score("saberpro_regression", full_data)
            return self._build_prediction_response("saberpro", score, full_data)
        except Exception as exc:
            print(f"Error en predict_saberpro: {exc}")
            return {"success": False, "error": str(exc)}

    def predict_bulk(self, df: pd.DataFrame, exam_type: str) -> pd.DataFrame:
        predictions = []

        for _, row in df.iterrows():
            try:
                if exam_type == "saber11":
                    data = {
                        "estrato": row.get("Estrato", 2),
                        "fami_educacionmadre_saber11": self._map_parent_education(row.get("Educacion_Madre", "Ninguno")),
                        "fami_educacionpadre_saber11": self._map_parent_education(row.get("Educacion_Padre", "Ninguno")),
                        "estu_dedicacioninternet": self._map_internet_usage(row.get("Internet", "No Navega Internet")),
                        "fami_comecarnepescadohuevo": self._map_protein_consumption(row.get("Consumo_Proteina", "Nunca o rara vez comemos eso")),
                        "edad": self._safe_int(row.get("Edad", 17), 17),
                    }
                    result = self.predict_saber11(data, strict=False)
                else:
                    data = {
                        "estrato": row.get("Estrato", 2),
                        "fami_educacionmadre_saberpro": self._map_parent_education(row.get("Educacion_Madre", "Ninguno")),
                        "fami_educacionpadre_saberpro": self._map_parent_education(row.get("Educacion_Padre", "Ninguno")),
                        "estu_dedicacioninternet": self._map_internet_usage(row.get("Internet", "No Navega Internet")),
                        "fami_comecarnepescadohuevo": self._map_protein_consumption(row.get("Consumo_Proteina", "Nunca o rara vez comemos eso")),
                        "puntaje_matematicas": self._safe_float(row.get("Matematicas_Prev", 50), 50.0),
                        "puntaje_lectura": self._safe_float(row.get("Lectura_Prev", 50), 50.0),
                        "edad": self._safe_int(row.get("Edad", 21), 21),
                        "inst_caracter_academico": self._map_institution_character(row.get("Caracter_Institucion", "UNIVERSIDAD")),
                        "semestre_actual": self._safe_int(row.get("Semestre_Actual", 6), 6),
                        "horas_trabajo_semanal": row.get("Horas_Trabajo_Semanal", "0"),
                    }
                    result = self.predict_saberpro(data, strict=False)

                predictions.append(
                    {
                        "Nombre_Estudiante": row.get("Nombre_Estudiante", "Sin nombre"),
                        "Puntaje_Predicho": result.get("puntaje_estimado", "Error"),
                        "Riesgo": result.get("riesgo", "Desconocido"),
                        "Desempenio": result.get("desempenio_esperado", result.get("error", "Desconocido")),
                    }
                )
            except Exception as exc:
                predictions.append(
                    {
                        "Nombre_Estudiante": row.get("Nombre_Estudiante", "Sin nombre"),
                        "Puntaje_Predicho": "Error",
                        "Riesgo": "Error",
                        "Desempenio": str(exc),
                    }
                )

        return pd.DataFrame(predictions)

    def validate_bulk_upload(self, df: pd.DataFrame, exam_type: str) -> List[str]:
        errors: List[str] = []
        if exam_type not in self.bulk_required_columns:
            return ["Tipo de examen no soportado"]
        if df.empty:
            return ["El archivo no contiene registros"]

        missing = [col for col in self.bulk_required_columns[exam_type] if col not in df.columns]
        if missing:
            errors.append("Faltan columnas obligatorias: " + ", ".join(missing))
            return errors

        if len(df) > 5000:
            errors.append("El archivo supera el maximo de 5000 registros por carga")

        for index, row in df.head(50).iterrows():
            row_number = int(index) + 2
            if self._is_blank(row.get("Nombre_Estudiante")):
                errors.append(f"Fila {row_number}: Nombre_Estudiante es obligatorio")
            row_data = row.to_dict()
            self._add_numeric_range_error(row_data, "Estrato", f"Fila {row_number}: Estrato", 0, 6, errors, required=True)
            if exam_type == "saber11":
                self._add_numeric_range_error(row_data, "Edad", f"Fila {row_number}: Edad", 14, 25, errors, required=True)
            else:
                self._add_numeric_range_error(row_data, "Edad", f"Fila {row_number}: Edad", 16, 100, errors, required=True)
                self._add_numeric_range_error(row_data, "Matematicas_Prev", f"Fila {row_number}: Matematicas_Prev", 0, 100, errors, required=True)
                self._add_numeric_range_error(row_data, "Lectura_Prev", f"Fila {row_number}: Lectura_Prev", 0, 100, errors, required=True)
                self._add_numeric_range_error(row_data, "Semestre_Actual", f"Fila {row_number}: Semestre_Actual", 1, 12, errors, required=True)

            if len(errors) >= 15:
                errors.append("Hay mas errores; corrige el archivo y vuelve a cargarlo")
                break

        return errors


model_manager = ModelManager(model_dir=".")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template(
        "index.html",
        reference_payload=model_manager.get_reference_payload(),
        model_schema_payload=model_manager.get_model_schema_payload(),
    )


@app.route("/api/predict/saber11", methods=["POST"])
def predict_saber11():
    try:
        data = request.get_json()
        result = model_manager.predict_saber11(data)
        return jsonify(result), 200 if result.get("success") else 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/predict/saberpro", methods=["POST"])
def predict_saberpro():
    try:
        data = request.get_json()
        result = model_manager.predict_saberpro(data)
        return jsonify(result), 200 if result.get("success") else 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/predict/bulk", methods=["POST"])
def predict_bulk():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No se recibio ningun archivo"}), 400

        file = request.files["file"]
        exam_type = request.form.get("exam_type", "saber11")
        if exam_type not in {"saber11", "saberpro"}:
            return jsonify({"success": False, "error": "Tipo de examen no soportado"}), 400

        if file.filename == "":
            return jsonify({"success": False, "error": "Selecciona un archivo"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Tipo de archivo no permitido. Usa CSV, XLSX o XLS"}), 400

        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df.columns = [str(col).strip() for col in df.columns]

        validation_errors = model_manager.validate_bulk_upload(df, exam_type)
        if validation_errors:
            return jsonify({"success": False, "error": "; ".join(validation_errors)}), 400

        results_df = model_manager.predict_bulk(df, exam_type)
        results = results_df.to_dict("records")

        return jsonify(
            {
                "success": True,
                "total_registros": len(results),
                "resultados": results,
            }
        ), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/reference/dropdowns", methods=["GET"])
def get_dropdown_data():
    try:
        path = Path("reference_data/dropdown_data.json")
        if not path.exists():
            return jsonify({"success": False, "error": "Datos de referencia no encontrados"}), 404
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data}), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return (
        jsonify(
            {
                "status": "ok",
                "modelos_cargados": len(model_manager.models),
                "total_modelos_esperados": 2,
                "modelos_disponibles": sorted(model_manager.models.keys()),
                "errores_carga": model_manager.load_errors,
            }
        ),
        200,
    )


@app.route("/api/template/<exam_type>", methods=["GET"])
def get_template(exam_type: str):
    try:
        if exam_type == "saber11":
            data = {
                "Nombre_Estudiante": ["Juan Pérez", "Maria Lopez", "Carlos Rodríguez"],
                "Estrato": [2, 3, 1],
                "Educacion_Madre": ["Secundaria", "Universitaria", "Primaria"],
                "Educacion_Padre": ["Primaria", "Secundaria", "Primaria"],
                "Internet": ["1-2 horas", "2-4 horas", "No Navega Internet"],
                "Consumo_Proteina": ["Diariamente", "Regularmente", "A veces"],
                "Edad": [17, 17, 18],
            }
        else:
            data = {
                "Nombre_Estudiante": ["Juan Pérez", "Maria Lopez"],
                "Estrato": [2, 3],
                "Educacion_Madre": ["Secundaria", "Universitaria"],
                "Educacion_Padre": ["Primaria", "Secundaria"],
                "Internet": ["2-4 horas", "1-2 horas"],
                "Consumo_Proteina": ["Diariamente", "Regularmente"],
                "Matematicas_Prev": [65, 78],
                "Lectura_Prev": [70, 82],
                "Edad": [22, 21],
                "Caracter_Institucion": ["TECNOLÓGICA", "UNIVERSIDAD"],
                "Semestre_Actual": [6, 5],
                "Horas_Trabajo_Semanal": ["0", "11-20"],
            }

        df = pd.DataFrame(data)
        return jsonify({"success": True, "plantilla": df.to_dict("records"), "columnas": list(df.columns)}), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    use_reloader = os.getenv("FLASK_USE_RELOADER", "0").lower() in {"1", "true", "yes", "on"}

    print("\n" + "=" * 60)
    print("  EduPredictor.ai - API de Predicción Académica")
    print("=" * 60)
    print(f"\nModelos disponibles: {len(model_manager.models)}/2")
    print(f"Servidor corriendo en: http://localhost:5000")
    print(f"Debug: {debug_mode} | Reloader: {use_reloader}\n")

    app.run(debug=debug_mode, use_reloader=use_reloader, port=5000, host="0.0.0.0")
