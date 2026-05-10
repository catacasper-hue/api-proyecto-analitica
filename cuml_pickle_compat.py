from __future__ import annotations

import sys
import types

import numpy as np


class CumlArrayDescriptorMeta:
    """Compatibilidad mínima para deserializar arrays de cuML en CPU."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)

    def to_numpy(self) -> np.ndarray:
        value = getattr(self, "input_value", None)
        return np.asarray(value, dtype=np.float32)


class CumlArray:
    """Reconstruye el array serializado por cuML usando los frames de joblib."""

    @staticmethod
    def host_deserialize(meta, frames):
        array = np.asarray(frames[0], dtype=np.float32)
        shape = meta.get("constructor-kwargs", {}).get("shape")
        if shape:
            array = array.reshape(shape)
        return array


class Ridge:
    """Implementa solo lo necesario para inferencia de un Ridge serializado por cuML."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _extract_array(self, value) -> np.ndarray:
        if hasattr(value, "to_numpy"):
            return np.asarray(value.to_numpy(), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    def predict(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        coef = self._extract_array(getattr(self, "coef_", 0.0))
        intercept = self._extract_array(getattr(self, "intercept_", 0.0))

        if coef.ndim == 1:
            prediction = x @ coef
        else:
            prediction = x @ coef.T

        if intercept.ndim == 0 or intercept.size == 1:
            prediction = prediction + float(intercept.reshape(-1)[0])
        else:
            prediction = prediction + intercept

        return np.asarray(prediction, dtype=np.float32)


def install() -> bool:
    """Registra módulos stub solo cuando cuML no está instalado."""
    try:
        import cuml  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    if "cuml.linear_model.ridge" in sys.modules:
        return True

    cuml_pkg = types.ModuleType("cuml")
    cuml_pkg.__path__ = []
    linear_model_pkg = types.ModuleType("cuml.linear_model")
    linear_model_pkg.__path__ = []
    ridge_module = types.ModuleType("cuml.linear_model.ridge")
    common_pkg = types.ModuleType("cuml.common")
    common_pkg.__path__ = []
    array_descriptor_module = types.ModuleType("cuml.common.array_descriptor")
    internals_pkg = types.ModuleType("cuml.internals")
    internals_pkg.__path__ = []
    array_module = types.ModuleType("cuml.internals.array")

    ridge_module.Ridge = Ridge
    linear_model_pkg.Ridge = Ridge
    array_descriptor_module.CumlArrayDescriptorMeta = CumlArrayDescriptorMeta
    array_module.CumlArray = CumlArray

    common_pkg.array_descriptor = array_descriptor_module
    internals_pkg.array = array_module
    cuml_pkg.linear_model = linear_model_pkg
    cuml_pkg.common = common_pkg
    cuml_pkg.internals = internals_pkg

    modules = {
        "cuml": cuml_pkg,
        "cuml.linear_model": linear_model_pkg,
        "cuml.linear_model.ridge": ridge_module,
        "cuml.common": common_pkg,
        "cuml.common.array_descriptor": array_descriptor_module,
        "cuml.internals": internals_pkg,
        "cuml.internals.array": array_module,
    }
    sys.modules.update(modules)
    return True
