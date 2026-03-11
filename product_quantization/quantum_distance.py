# -*- coding: utf-8 -*-
__author__ = "Christian Jesinghaus"
 
# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation

import numpy as np
from typing import List, Optional
import logging
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit_aer import AerSimulator
try:
    from tqdm import trange
except ImportError:  
    def trange(*args, **kwargs):
        return range(*args, **kwargs)

logger = logging.getLogger(__name__)

_ALLOWED_METRICS = {
    "log_fidelity",
    "one_minus_fidelity",
    "swap_test",
    "1-f",
    "lf",
    "logf",
    "omf",
}


class QuantumDistanceCalculator:
    """
    Unified quantum distance calculator with smoothed log-fidelity.

    For metric == "log_fidelity" we use the zero-normalized smoothed loss

        d_eps(F) = log((1 + eps) / (F + eps))

    This preserves the ordering of -log(F + eps), but now d_eps(1) = 0.
    """
    def __init__(
        self,
        shots: int = 1024,
        backend=None,
        smooth_eps: float = 1e-3,
        circuit_batch_size: int = 256,
        transpile_optimization_level: int = 0,
    ):
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.smooth_eps = smooth_eps
        self.circuit_batch_size = max(1, int(circuit_batch_size))
        self.transpile_optimization_level = int(transpile_optimization_level)

    def distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        *,
        mode: str | None = None,
        metric: str | None = None,
    ) -> float:
        """
        Calculates distance between two vectors

        accepts "mode" and "metric" for legacy reasons
        """
        #Legacy
        if mode is None:
            mode = metric or "log_fidelity"

        mode = self._normalize_mode(mode)
        F = self._fidelity(vec1, vec2)
        return self._smooth_log_distance(F) if mode == "log_fidelity" else 1.0 - F

    def fidelity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return self._fidelity(vec1, vec2)

    #helpers 
    def _smooth_log_distance(self, F: float) -> float:
        """
        Zero-normalized smoothed log-fidelity distance:

            d_eps(F) = log((1 + eps) / (F + eps))

        Properties:
        - finite for F = 0
        - equals 0 for F = 1
        - strictly decreasing in F
        """
        F = float(np.clip(F, 0.0, 1.0))
        return float(np.log((1.0 + self.smooth_eps) / (F + self.smooth_eps)))

    def _normalize_mode(self, mode) -> str:
        """
        Normalize legacy aliases to the two canonical modes:
        'log_fidelity' and 'one_minus_fidelity'.
        """
        if mode is None:
            return "log_fidelity"

        mode = str(mode).lower()
        if mode not in _ALLOWED_METRICS:
            raise ValueError(f"Unknown distance mode '{mode}'. Allowed: {_ALLOWED_METRICS}")

        return (
            "one_minus_fidelity"
            if mode in ("one_minus_fidelity", "swap_test", "1-f", "omf")
            else "log_fidelity"
        )

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec)
        return v / (np.linalg.norm(v) + 1e-12)

    def _classical_fidelity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.abs(np.dot(v1, v2)) ** 2)

    def _counts_to_overlap_sq(self, counts: dict) -> float:
        prob_0 = counts.get("0", 0) / self.shots
        return max(0.0, min(1.0, 2 * prob_0 - 1))

    def _run_swap_test_circuits(self, circuits: List[QuantumCircuit]) -> List[dict]:
        if not circuits:
            return []

        tqcs = transpile(
            circuits,
            self.backend,
            optimization_level=self.transpile_optimization_level,
        )
        if not isinstance(tqcs, list):
            tqcs = [tqcs]

        job = self.backend.run(tqcs, shots=self.shots)
        result = job.result()

        try:
            return [result.get_counts(i) for i in range(len(tqcs))]
        except Exception:
            raw_counts = result.get_counts()
            return raw_counts if isinstance(raw_counts, list) else [raw_counts]

    def _fidelity_from_normalized_pairs(
        self,
        pairs: List[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        fidelities = np.zeros(len(pairs), dtype=float)
        circuits: List[QuantumCircuit] = []
        circuit_pair_indices: List[int] = []

        for pair_idx, (v1, v2) in enumerate(pairs):
            qc = self._create_swap_test_circuit(v1, v2)
            if qc is None:
                fidelities[pair_idx] = self._classical_fidelity(v1, v2)
            else:
                circuits.append(qc)
                circuit_pair_indices.append(pair_idx)

        for batch_start in range(0, len(circuits), self.circuit_batch_size):
            batch_circuits = circuits[batch_start:batch_start + self.circuit_batch_size]
            batch_pair_indices = circuit_pair_indices[
                batch_start:batch_start + self.circuit_batch_size
            ]

            try:
                counts_list = self._run_swap_test_circuits(batch_circuits)
                for local_idx, counts in enumerate(counts_list):
                    pair_idx = batch_pair_indices[local_idx]
                    fidelities[pair_idx] = self._counts_to_overlap_sq(counts)
            except Exception as e:
                logger.warning("Falling back to classical fidelity for circuit batch: %s", e)
                for pair_idx in batch_pair_indices:
                    v1, v2 = pairs[pair_idx]
                    fidelities[pair_idx] = self._classical_fidelity(v1, v2)

        return fidelities

    def _distances_from_fidelities(
        self,
        fidelities: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        metric = self._normalize_mode(metric)
        fidelities = np.clip(np.asarray(fidelities, dtype=float), 0.0, 1.0)

        if metric == "log_fidelity":
            return np.log((1.0 + self.smooth_eps) / (fidelities + self.smooth_eps))

        return 1.0 - fidelities

    
    #Fidelity via Swap‑Test or classical-Fallback
    
    def _fidelity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        v1 = self._normalize_vector(vec1)
        v2 = self._normalize_vector(vec2)
        return float(self._fidelity_from_normalized_pairs([(v1, v2)])[0])

    def _create_swap_test_circuit(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> Optional[QuantumCircuit]:
        try:
            from .util import amplitude_encoding

            d = len(vec1)
            n_qubits = int(np.ceil(np.log2(d))) if d > 1 else 1
            qreg1 = QuantumRegister(n_qubits, "v1")
            qreg2 = QuantumRegister(n_qubits, "v2")
            anc = QuantumRegister(1, "anc")
            creg = ClassicalRegister(1, "c")
            qc = QuantumCircuit(qreg1, qreg2, anc, creg)

            qc.compose(amplitude_encoding(vec1), qreg1, inplace=True)
            qc.compose(amplitude_encoding(vec2), qreg2, inplace=True)

            qc.h(anc[0])
            for i in range(n_qubits):
                qc.cswap(anc[0], qreg1[i], qreg2[i])
            qc.h(anc[0])
            qc.measure(anc[0], creg[0])
            return qc
        except Exception as e:
            logger.warning("Failed to create swap‑test circuit: %s", e)
            return None

    #Distancelist (Test‑Vektor vs. List)
    def quantum_distance_matrix(
        self,
        vectors: List[np.ndarray],
        test_vector: np.ndarray,
        mode: str = "log_fidelity",
    ) -> np.ndarray:
        mode = self._normalize_mode(mode)

        test_vector_norm = self._normalize_vector(test_vector)
        vectors_norm = [self._normalize_vector(v) for v in vectors]
        pairs = [(test_vector_norm, v) for v in vectors_norm]

        fidelities = self._fidelity_from_normalized_pairs(pairs)
        return self._distances_from_fidelities(fidelities, mode)

    #Alias for K‑Means / PQ‑kNN
    def pairwise_distance_matrix(
        self,
        X: List[np.ndarray],
        Y: Optional[List[np.ndarray]] = None,
        metric: str = "log_fidelity",
    ) -> np.ndarray:
        metric = self._normalize_mode(metric)

        X_norm = [self._normalize_vector(x) for x in X]
        Y_norm = X_norm if Y is None else [self._normalize_vector(y) for y in Y]

        D = np.zeros((len(X_norm), len(Y_norm)), dtype=float)

        batch_pairs: List[tuple[np.ndarray, np.ndarray]] = []
        batch_positions: List[tuple[int, int]] = []

        def flush_batch() -> None:
            if not batch_pairs:
                return

            fidelities = self._fidelity_from_normalized_pairs(batch_pairs)
            distances = self._distances_from_fidelities(fidelities, metric)

            for (row_idx, col_idx), dist in zip(batch_positions, distances):
                D[row_idx, col_idx] = dist

            batch_pairs.clear()
            batch_positions.clear()

        for i, x in enumerate(X_norm):
            for j, y in enumerate(Y_norm):
                batch_pairs.append((x, y))
                batch_positions.append((i, j))

                if len(batch_pairs) >= self.circuit_batch_size:
                    flush_batch()

        flush_batch()
        return D


#External Helpers: full Pairwise Distance Matrix
def quantum_pairwise_distances(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = "swap_test",
    *,
    shots: int = 1024,
    smooth_eps: float = 1e-3,
    backend=None,
    circuit_batch_size: int = 256,
    transpile_optimization_level: int = 0,
) -> np.ndarray:
    calc = QuantumDistanceCalculator(
        shots=shots,
        backend=backend,
        smooth_eps=smooth_eps,
        circuit_batch_size=circuit_batch_size,
        transpile_optimization_level=transpile_optimization_level,
    )
    return calc.pairwise_distance_matrix(X, Y, metric)
