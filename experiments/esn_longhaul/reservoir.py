"""Frame-local, train-calibrated linear and echo-state symbol equalizers.

All fitted quantities are explicit. A prediction uses received symbols and frame
masks only; transmitted symbols are consulted exclusively by ``fit`` methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from numba import njit
except ImportError:  # The reference implementation remains usable without JIT.
    def njit(*args, **kwargs):
        return lambda function: function


def complex_to_real(symbols: np.ndarray) -> np.ndarray:
    """XY complex symbols -> [XI, XQ, YI, YQ], retaining the time axis."""
    symbols = np.asarray(symbols)
    if symbols.ndim != 2 or symbols.shape[1] != 2:
        raise ValueError("symbols must have shape (time, 2)")
    result = np.empty((len(symbols), 4), dtype=np.float64)
    result[:, 0::2] = symbols.real
    result[:, 1::2] = symbols.imag
    return result


def real_to_complex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("real XY values must have shape (time, 4)")
    return values[:, 0::2] + 1j * values[:, 1::2]


@dataclass(frozen=True)
class SymbolFrame:
    rx_symbols: np.ndarray
    tx_symbols: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    frame_id: str = ""

    def __post_init__(self):
        rx = np.asarray(self.rx_symbols, dtype=np.complex128)
        if rx.ndim != 2 or rx.shape[1] != 2:
            raise ValueError("received symbols must have shape (time, 2)")
        tx = None if self.tx_symbols is None else np.asarray(self.tx_symbols, dtype=np.complex128)
        if tx is not None and tx.shape != rx.shape:
            raise ValueError("transmitted and received frame shapes differ")
        valid = np.ones(len(rx), dtype=bool) if self.valid_mask is None else np.asarray(self.valid_mask, dtype=bool)
        if valid.shape != (len(rx),):
            raise ValueError("valid_mask must have shape (time,)")
        if not np.isfinite(rx[valid]).all() or (tx is not None and not np.isfinite(tx[valid]).all()):
            raise ValueError("valid symbols must be finite")
        object.__setattr__(self, "rx_symbols", rx)
        object.__setattr__(self, "tx_symbols", tx)
        object.__setattr__(self, "valid_mask", valid)


@dataclass(frozen=True)
class ContextConfig:
    past_symbols: int = 2
    future_symbols: int = 2
    washout_symbols: int = 64

    def __post_init__(self):
        for value in asdict(self).values():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("context and washout lengths must be nonnegative integers")

    @property
    def n_features(self) -> int:
        return 4 * (1 + self.past_symbols + self.future_symbols)


@dataclass(frozen=True)
class ReservoirConfig:
    units: int = 128
    spectral_radius: float = 0.8
    leak_rate: float = 0.5
    input_scale: float = 0.5
    density: float = 0.05
    bias_scale: float = 0.2
    ridge: float = 1e-4
    seed: int = 1

    def __post_init__(self):
        if not isinstance(self.units, int) or isinstance(self.units, bool) or self.units < 2:
            raise ValueError("units must be an integer >= 2")
        if not (0 < self.leak_rate <= 1 and 0 < self.density <= 1):
            raise ValueError("leak_rate and density must lie in (0, 1]")
        if self.spectral_radius <= 0 or self.input_scale <= 0 or self.bias_scale < 0 or self.ridge <= 0:
            raise ValueError("radius, input scale, and ridge must be positive; bias scale must be nonnegative")
        if not np.isfinite(list(asdict(self).values())).all() or self.seed < 0:
            raise ValueError("reservoir settings must be finite and seed nonnegative")


@dataclass(frozen=True)
class PreparedFrame:
    indices: np.ndarray
    inputs: np.ndarray
    calibrated: np.ndarray
    score_rows: np.ndarray


class TrainPreprocessor:
    """A complex gain per polarization and four train-only input scales."""

    def __init__(self, context: ContextConfig):
        self.context = context
        self.gain: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, frames: Iterable[SymbolFrame]) -> TrainPreprocessor:
        frames = tuple(frames)
        if not frames or any(frame.tx_symbols is None for frame in frames):
            raise ValueError("calibration requires labeled training frames")
        numerator = np.zeros(2, dtype=np.complex128)
        denominator = np.zeros(2)
        for frame in frames:
            rx, tx = frame.rx_symbols[frame.valid_mask], frame.tx_symbols[frame.valid_mask]
            # Fit the forward channel gain h in rx = h*tx + noise, then invert
            # it. Inverse-regression LS would shrink an AWGN constellation.
            numerator += np.sum(rx * tx.conj(), axis=0)
            denominator += np.sum(np.abs(tx) ** 2, axis=0)
        if np.any(denominator <= 0):
            raise ValueError("training data have no transmitted signal energy")
        forward_gain = numerator / denominator
        if np.any(np.abs(forward_gain) < 1e-12):
            raise ValueError("training forward gain is zero; scalar calibration is undefined")
        self.gain = 1 / forward_gain
        total, squares, count = np.zeros(4), np.zeros(4), 0
        for frame in frames:
            real = complex_to_real(frame.rx_symbols[frame.valid_mask] * self.gain)
            total += real.sum(axis=0)
            squares += (real * real).sum(axis=0)
            count += len(real)
        self.mean = total / count
        self.scale = np.sqrt(np.maximum(squares / count - self.mean**2, 1e-12))
        return self

    def _check_fitted(self):
        if self.gain is None or self.mean is None or self.scale is None:
            raise RuntimeError("preprocessor has not been fit on training frames")

    def calibrate(self, received: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(received) * self.gain

    def prepare(self, frame: SymbolFrame) -> PreparedFrame:
        """Build windows inside this frame and reset/wash out every valid segment."""
        self._check_fitted()
        n = len(frame.rx_symbols)
        past, future = self.context.past_symbols, self.context.future_symbols
        width = past + future + 1
        calibrated = self.calibrate(frame.rx_symbols)
        if n < width:
            raise ValueError("frame is shorter than the declared context")
        starts = np.arange(n - width + 1)
        invalid_cumulative = np.r_[0, np.cumsum(~frame.valid_mask)]
        valid_windows = (invalid_cumulative[starts + width] - invalid_cumulative[starts]) == 0
        indices = starts[valid_windows] + past
        standardized = (complex_to_real(calibrated) - self.mean) / self.scale
        inputs = np.concatenate([standardized[indices + offset] for offset in range(-past, future + 1)], axis=1)
        score_rows = np.ones(len(indices), dtype=bool)
        segment_starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
        segment_ends = np.r_[segment_starts[1:], len(indices)]
        for start, end in zip(segment_starts, segment_ends):
            score_rows[start:min(start + self.context.washout_symbols, end)] = False
        return PreparedFrame(indices, inputs, calibrated, score_rows)

    def common_mask(self, frame: SymbolFrame) -> np.ndarray:
        prepared = self.prepare(frame)
        mask = np.zeros(len(frame.rx_symbols), dtype=bool)
        mask[prepared.indices[prepared.score_rows]] = True
        return mask


@njit(cache=True)
def _states_numba(inputs, indices, input_weights, bias, indptr, recurrent_indices, recurrent_data, leak):
    n, units = len(inputs), len(bias)
    states = np.empty((n, units), dtype=np.float64)
    state, next_state = np.zeros(units), np.zeros(units)
    for sample in range(n):
        if sample == 0 or indices[sample] != indices[sample - 1] + 1:
            state[:] = 0.0
        for row in range(units):
            activation = bias[row]
            for feature in range(inputs.shape[1]):
                activation += input_weights[row, feature] * inputs[sample, feature]
            for entry in range(indptr[row], indptr[row + 1]):
                activation += recurrent_data[entry] * state[recurrent_indices[entry]]
            next_state[row] = (1 - leak) * state[row] + leak * np.tanh(activation)
        state[:] = next_state
        states[sample] = state
    return states


def _ridge_readout(gram: np.ndarray, cross: np.ndarray, count: int, ridge: float) -> np.ndarray:
    if count == 0:
        raise ValueError("no training samples remain after context/mask/washout")
    normal = gram / count
    regularizer = np.eye(len(normal)) * ridge
    regularizer[0, 0] = 0  # Do not penalize the constant offset.
    return np.linalg.solve(normal + regularizer, cross / count)


@dataclass(frozen=True)
class Prediction:
    symbols: np.ndarray
    valid_mask: np.ndarray


class LinearEqualizer:
    """Ridge residual readout with precisely the ESN's explicit context."""

    kind = "linear"

    def __init__(self, preprocessor: TrainPreprocessor, ridge: float = 1e-4):
        if ridge <= 0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and positive")
        preprocessor._check_fitted()
        self.preprocessor, self.ridge = preprocessor, ridge
        self.readout: np.ndarray | None = None
        self.training_samples = 0

    def design(self, frame: SymbolFrame) -> tuple[PreparedFrame, np.ndarray]:
        prepared = self.preprocessor.prepare(frame)
        return prepared, np.column_stack((np.ones(len(prepared.inputs)), prepared.inputs))

    def fit_ridge_path(self, frames: Iterable[SymbolFrame], ridges: Iterable[float]) -> list[LinearEqualizer]:
        dimension = 1 + self.preprocessor.context.n_features
        gram, cross, count = np.zeros((dimension, dimension)), np.zeros((dimension, 4)), 0
        for frame in frames:
            if frame.tx_symbols is None:
                raise ValueError("readout fitting requires training labels")
            prepared, design = self.design(frame)
            design = design[prepared.score_rows]
            indices = prepared.indices[prepared.score_rows]
            residual = complex_to_real(frame.tx_symbols[indices] - prepared.calibrated[indices])
            gram += design.T @ design
            cross += design.T @ residual
            count += len(design)
        models = []
        for ridge in ridges:
            model = LinearEqualizer(self.preprocessor, float(ridge))
            model.readout = _ridge_readout(gram, cross, count, model.ridge)
            model.training_samples = count
            models.append(model)
        return models

    def fit(self, frames: Iterable[SymbolFrame]) -> LinearEqualizer:
        fitted = self.fit_ridge_path(frames, [self.ridge])[0]
        self.readout, self.training_samples = fitted.readout, fitted.training_samples
        return self

    def predict_prepared(self, prepared: PreparedFrame, design: np.ndarray) -> Prediction:
        if self.readout is None:
            raise RuntimeError("readout has not been fit")
        symbols = prepared.calibrated.copy()
        symbols[prepared.indices] += real_to_complex(design @ self.readout)
        valid = np.zeros(len(symbols), dtype=bool)
        valid[prepared.indices[prepared.score_rows]] = True
        return Prediction(symbols, valid)

    def predict(self, frame: SymbolFrame) -> Prediction:
        return self.predict_prepared(*self.design(frame))

    @property
    def model_array_bytes(self) -> int:
        return sum(array.nbytes for array in self._arrays().values())

    def _arrays(self) -> dict[str, np.ndarray]:
        if self.readout is None:
            raise RuntimeError("cannot save an unfit readout")
        return {"gain": self.preprocessor.gain, "mean": self.preprocessor.mean,
                "scale": self.preprocessor.scale, "readout": self.readout}

    def _metadata(self) -> dict:
        return {"schema_version": 1, "kind": self.kind, "context": asdict(self.preprocessor.context),
                "ridge": self.ridge, "training_samples": self.training_samples}

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata=np.array(json.dumps(self._metadata(), sort_keys=True)), **self._arrays())


class ESNEqualizer(LinearEqualizer):
    """Frozen sparse tanh reservoir; fit only a linear residual readout."""

    kind = "esn"

    def __init__(self, preprocessor: TrainPreprocessor, config: ReservoirConfig = ReservoirConfig()):
        super().__init__(preprocessor, config.ridge)
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.input_weights = rng.uniform(-config.input_scale, config.input_scale,
                                         (config.units, preprocessor.context.n_features))
        self.bias = rng.uniform(-config.bias_scale, config.bias_scale, config.units)
        # Exact eigenvalue scaling is inexpensive for this deliberately small ESN.
        for _ in range(8):
            dense = rng.uniform(-1, 1, (config.units, config.units))
            dense *= rng.random(dense.shape) < config.density
            radius = float(np.max(np.abs(np.linalg.eigvals(dense))))
            if radius > 1e-12:
                break
        else:
            raise ValueError("sampled recurrent graph has zero spectral radius; increase density")
        dense *= config.spectral_radius / radius
        rows, columns = np.nonzero(dense)
        self.recurrent_indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=config.units))].astype(np.int64)
        self.recurrent_indices = columns.astype(np.int64)
        self.recurrent_data = dense[rows, columns]

    def design(self, frame: SymbolFrame) -> tuple[PreparedFrame, np.ndarray]:
        prepared = self.preprocessor.prepare(frame)
        states = _states_numba(prepared.inputs, prepared.indices, self.input_weights, self.bias,
                               self.recurrent_indptr, self.recurrent_indices, self.recurrent_data,
                               self.config.leak_rate)
        return prepared, np.column_stack((np.ones(len(states)), prepared.inputs, states))

    def fit_ridge_path(self, frames: Iterable[SymbolFrame], ridges: Iterable[float]) -> list[ESNEqualizer]:
        dimension = 1 + self.preprocessor.context.n_features + self.config.units
        gram, cross, count = np.zeros((dimension, dimension)), np.zeros((dimension, 4)), 0
        for frame in frames:
            if frame.tx_symbols is None:
                raise ValueError("readout fitting requires training labels")
            prepared, design = self.design(frame)
            design = design[prepared.score_rows]
            indices = prepared.indices[prepared.score_rows]
            residual = complex_to_real(frame.tx_symbols[indices] - prepared.calibrated[indices])
            gram += design.T @ design
            cross += design.T @ residual
            count += len(design)
        models = []
        for ridge in ridges:
            if ridge <= 0 or not np.isfinite(ridge):
                raise ValueError("ridge path must contain finite positive values")
            # Avoid regenerating (or refitting) the reservoir for ridge candidates.
            model = object.__new__(ESNEqualizer)
            model.preprocessor, model.ridge = self.preprocessor, float(ridge)
            model.config = replace(self.config, ridge=float(ridge))
            for name in ("input_weights", "bias", "recurrent_indptr", "recurrent_indices", "recurrent_data"):
                setattr(model, name, getattr(self, name))
            model.readout = _ridge_readout(gram, cross, count, model.ridge)
            model.training_samples = count
            models.append(model)
        return models

    def _arrays(self) -> dict[str, np.ndarray]:
        return dict(super()._arrays(), input_weights=self.input_weights, bias=self.bias,
                    recurrent_indptr=self.recurrent_indptr, recurrent_indices=self.recurrent_indices,
                    recurrent_data=self.recurrent_data)

    def _metadata(self) -> dict:
        return dict(super()._metadata(), reservoir=asdict(self.config))


def load_equalizer(path: str | Path) -> LinearEqualizer | ESNEqualizer:
    """Restore a prediction-complete checkpoint; no pickle or training is used."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema_version") != 1 or metadata.get("kind") not in {"linear", "esn"}:
            raise ValueError("unsupported equalizer checkpoint")
        preprocessor = TrainPreprocessor(ContextConfig(**metadata["context"]))
        preprocessor.gain, preprocessor.mean, preprocessor.scale = (archive[key].copy() for key in ("gain", "mean", "scale"))
        if metadata["kind"] == "esn":
            model = object.__new__(ESNEqualizer)
            model.config = ReservoirConfig(**metadata["reservoir"])
            for key in ("input_weights", "bias", "recurrent_indptr", "recurrent_indices", "recurrent_data"):
                setattr(model, key, archive[key].copy())
        else:
            model = object.__new__(LinearEqualizer)
        model.preprocessor, model.ridge = preprocessor, metadata["ridge"]
        model.readout = archive["readout"].copy()
        model.training_samples = metadata["training_samples"]
    return model
