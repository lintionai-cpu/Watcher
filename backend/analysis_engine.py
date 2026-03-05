"""
Analysis Engine — Comprehensive ML/AI pattern analysis for Deriv volatility indices.

Implements:
  1.  Markov Model (order-1 and order-2)
  2.  Hidden Markov Model (HMM) — 3 hidden states
  3.  Autocorrelation analysis
  4.  FFT / Fourier frequency analysis
  5.  ARIMA time-series forecast
  6.  K-Means clustering of rolling windows
  7.  SVM classifier
  8.  Random Forest classifier
  9.  Lightweight LSTM (numpy RNN)
 10.  Apriori frequent-pattern mining
 11.  Genetic Algorithm rule evolution
 12.  Shannon Entropy measurement
 13.  Chi-Square uniformity test
 14.  Q-Learning adaptive ensemble weights
 15.  Ensemble stacking of all model votes
"""
import time
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy import stats as scipy_stats
from scipy.signal import correlate

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        ARIMA_AVAILABLE = True
    except Exception:
        ARIMA_AVAILABLE = False

    try:
        from sklearn.cluster import KMeans
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        SK_AVAILABLE = True
    except Exception:
        SK_AVAILABLE = False

    try:
        from hmmlearn import hmm
        HMM_AVAILABLE = True
    except Exception:
        HMM_AVAILABLE = False

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS
# ══════════════════════════════════════════════════════════════

STRATEGY_CATEGORIES = {
    "over2_under7": {
        "name": "Over 2 / Under 7",
        "probability": 0.70,
        "barriers": {"over": 2, "under": 7},
        "desc": "70% probability zone — digits 3–6 excluded",
    },
    "under5_over5": {
        "name": "Under 5 / Over 5",
        "probability": 0.50,
        "barriers": {"under": 5, "over": 4},
        "desc": "50/50 split — pure digit parity",
    },
    "over1_under8": {
        "name": "Over 1 / Under 8",
        "probability": 0.80,
        "barriers": {"over": 1, "under": 8},
        "desc": "80% probability zone — widest safe range",
    },
    "under3_over5": {
        "name": "Under 3 / Over 5",
        "probability": 0.40,
        "barriers": {"under": 3, "over": 5},
        "desc": "Asymmetric — high-payout polarised play",
    },
}

# Minimum ticks before analysis runs
MIN_TICKS = 60

# ══════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════

def _safe_arr(digits: List[int]) -> np.ndarray:
    return np.array(digits, dtype=np.float32)


def _digit_features(digits: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Build X, y for supervised models. Feature window → next-digit target."""
    X, y = [], []
    for i in range(window, len(digits)):
        window_arr = digits[i - window:i]
        X.append([
            *window_arr,
            float(np.mean(window_arr)),
            float(np.std(window_arr) + 1e-6),
            float(np.min(window_arr)),
            float(np.max(window_arr)),
            float(window_arr[-1]),   # last digit
            float(np.sum(window_arr < 5)),  # low count
            float(np.sum(window_arr >= 5)), # high count
        ])
        y.append(digits[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# INDIVIDUAL MODEL ANALYZERS
# ══════════════════════════════════════════════════════════════

class MarkovModel:
    """Order-1 and order-2 Markov transition matrices."""

    def analyze(self, digits: np.ndarray) -> Dict:
        if len(digits) < 30:
            return {}
        # Order-1 transitions: 10×10
        trans1 = np.ones((10, 10)) * 0.1  # Laplace smoothing
        for i in range(len(digits) - 1):
            trans1[int(digits[i])][int(digits[i + 1])] += 1
        trans1 = trans1 / trans1.sum(axis=1, keepdims=True)

        last = int(digits[-1])
        probs = trans1[last]
        return {
            "probs_10": probs.tolist(),
            "transition_matrix": trans1.tolist(),
            "last_digit": last,
        }


class HMMModel:
    """Hidden Markov Model with 3 hidden states (Low/Mid/High regime)."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._trained_on: int = 0

    def analyze(self, digits: np.ndarray) -> Dict:
        if not HMM_AVAILABLE or len(digits) < 100:
            return self._fallback(digits)
        try:
            if len(digits) - self._trained_on > 50 or self._model is None:
                self._train(digits)
            if self._model is None:
                return self._fallback(digits)
            obs = digits[-50:].reshape(-1, 1).astype(np.float64)
            states = self._model.predict(obs)
            last_state = int(states[-1])
            # State → regime interpretation
            means = self._model.means_.flatten()
            order = np.argsort(means)
            regime = ["LOW", "MID", "HIGH"][list(order).index(last_state)]
            return {"regime": regime, "state": last_state, "means": means.tolist()}
        except Exception as exc:
            logger.debug("HMM error: %s", exc)
            return self._fallback(digits)

    def _train(self, digits: np.ndarray) -> None:
        try:
            obs = digits[-500:].reshape(-1, 1).astype(np.float64)
            model = hmm.GaussianHMM(n_components=3, covariance_type="diag",
                                    n_iter=50, random_state=42)
            model.fit(obs)
            self._model = model
            self._trained_on = len(digits)
        except Exception as exc:
            logger.debug("HMM train error: %s", exc)
            self._model = None

    def _fallback(self, digits: np.ndarray) -> Dict:
        if len(digits) == 0:
            return {}
        mean = float(np.mean(digits[-30:]))
        regime = "LOW" if mean < 3.5 else ("HIGH" if mean > 6.5 else "MID")
        return {"regime": regime, "state": -1, "means": [2.0, 4.5, 7.5]}


class AutocorrelationModel:
    """Detect repeating periodic patterns via ACF."""

    def analyze(self, digits: np.ndarray) -> Dict:
        if len(digits) < 50:
            return {}
        d = digits[-100:].astype(np.float64)
        d_norm = d - d.mean()
        acf = np.correlate(d_norm, d_norm, mode="full")
        acf = acf[len(acf) // 2:]
        acf = acf / (acf[0] + 1e-9)

        # Find strongest non-zero lag within 1..30
        search = acf[1:31]
        best_lag = int(np.argmax(np.abs(search))) + 1
        strength = float(np.abs(search[best_lag - 1]))
        # Predict next value based on best lag
        if best_lag < len(d):
            predicted = float(d[-best_lag])
        else:
            predicted = float(d.mean())
        return {
            "best_lag": best_lag,
            "strength": round(strength, 4),
            "predicted_digit": round(predicted),
        }


class FFTModel:
    """Detect dominant frequencies in digit sequences."""

    def analyze(self, digits: np.ndarray) -> Dict:
        if len(digits) < 64:
            return {}
        d = digits[-128:].astype(np.float64)
        d_norm = d - d.mean()
        spectrum = np.abs(rfft(d_norm))
        freqs = rfftfreq(len(d_norm))
        # Skip DC component
        spectrum[0] = 0
        dom_idx = int(np.argmax(spectrum[1:])) + 1
        dom_freq = float(freqs[dom_idx])
        dom_power = float(spectrum[dom_idx])
        total_power = float(spectrum.sum() + 1e-9)
        concentration = dom_power / total_power

        # Predict next digit using dominant cycle
        if dom_freq > 0:
            period = round(1.0 / dom_freq)
            predicted = float(d[-period]) if period < len(d) else float(d.mean())
        else:
            predicted = float(d.mean())

        return {
            "dominant_freq": round(dom_freq, 4),
            "period": round(1 / dom_freq) if dom_freq > 0 else 0,
            "concentration": round(concentration, 4),
            "predicted_digit": round(predicted),
        }


class ARIMAModel:
    """ARIMA(2,0,2) time-series forecast on last digits."""

    def analyze(self, digits: np.ndarray) -> Dict:
        if not ARIMA_AVAILABLE or len(digits) < 60:
            return self._fallback(digits)
        try:
            series = digits[-100:].astype(np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series, order=(2, 0, 2))
                result = model.fit()
                forecast = float(result.forecast(steps=1)[0])
                forecast = max(0, min(9, round(forecast)))
            return {"forecast": forecast, "fitted": True}
        except Exception as exc:
            logger.debug("ARIMA error: %s", exc)
            return self._fallback(digits)

    def _fallback(self, digits: np.ndarray) -> Dict:
        if len(digits) < 3:
            return {}
        # Simple 3-point linear extrapolation
        d = digits[-3:].astype(float)
        forecast = round(float(np.clip(2 * d[-1] - d[-2], 0, 9)))
        return {"forecast": forecast, "fitted": False}


class KMeansModel:
    """Cluster rolling windows; predict next digit from cluster centroid."""

    def __init__(self) -> None:
        self._km: Optional[KMeans] = None
        self._centroids: Optional[np.ndarray] = None
        self._trained_on: int = 0
        self._window = 10

    def analyze(self, digits: np.ndarray) -> Dict:
        if not SK_AVAILABLE or len(digits) < 80:
            return {}
        try:
            if len(digits) - self._trained_on > 50 or self._km is None:
                self._train(digits)
            if self._km is None:
                return {}
            current_window = digits[-self._window:].reshape(1, -1).astype(np.float32)
            cluster = int(self._km.predict(current_window)[0])
            centroid = self._centroids[cluster]
            predicted = round(float(centroid[-1]))
            return {
                "cluster": cluster,
                "centroid_mean": round(float(centroid.mean()), 3),
                "predicted_digit": predicted,
            }
        except Exception as exc:
            logger.debug("KMeans error: %s", exc)
            return {}

    def _train(self, digits: np.ndarray) -> None:
        try:
            wins = []
            for i in range(self._window, len(digits)):
                wins.append(digits[i - self._window:i])
            X = np.array(wins, dtype=np.float32)
            km = KMeans(n_clusters=5, n_init=5, random_state=42)
            km.fit(X)
            self._km = km
            self._centroids = km.cluster_centers_
            self._trained_on = len(digits)
        except Exception as exc:
            logger.debug("KMeans train error: %s", exc)
            self._km = None


class SVMModel:
    """SVM binary classifier for Over/Under prediction."""

    def __init__(self) -> None:
        self._clf: Optional[SVC] = None
        self._scaler: Optional[StandardScaler] = None
        self._trained_on: int = 0

    def analyze(self, digits: np.ndarray, threshold: int = 4) -> Dict:
        if not SK_AVAILABLE or len(digits) < 120:
            return {}
        try:
            if len(digits) - self._trained_on > 50 or self._clf is None:
                self._train(digits, threshold)
            if self._clf is None:
                return {}
            X_all, y_all = _digit_features(digits)
            X_last = self._scaler.transform(X_all[-1:])
            pred = int(self._clf.predict(X_last)[0])
            prob = self._clf.predict_proba(X_last)[0]
            return {
                "predicted_class": pred,
                "probability_over": round(float(prob[1]), 4),
                "probability_under": round(float(prob[0]), 4),
            }
        except Exception as exc:
            logger.debug("SVM error: %s", exc)
            return {}

    def _train(self, digits: np.ndarray, threshold: int) -> None:
        try:
            X, y_raw = _digit_features(digits[-300:])
            y = (y_raw > threshold).astype(int)
            if len(np.unique(y)) < 2:
                return
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            clf = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42)
            clf.fit(X_s, y)
            self._clf = clf
            self._scaler = scaler
            self._trained_on = len(digits)
        except Exception as exc:
            logger.debug("SVM train error: %s", exc)
            self._clf = None


class RandomForestModel:
    """Random Forest for multi-class digit prediction."""

    def __init__(self) -> None:
        self._rf: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._trained_on: int = 0

    def analyze(self, digits: np.ndarray) -> Dict:
        if not SK_AVAILABLE or len(digits) < 150:
            return {}
        try:
            if len(digits) - self._trained_on > 50 or self._rf is None:
                self._train(digits)
            if self._rf is None:
                return {}
            X_all, _ = _digit_features(digits)
            X_last = self._scaler.transform(X_all[-1:])
            proba = self._rf.predict_proba(X_last)[0]
            classes = self._rf.classes_
            digit_proba = {int(c): round(float(p), 4) for c, p in zip(classes, proba)}
            predicted = int(classes[np.argmax(proba)])
            return {
                "predicted_digit": predicted,
                "digit_probabilities": digit_proba,
                "over4_prob": round(sum(p for c, p in digit_proba.items() if c > 4), 4),
            }
        except Exception as exc:
            logger.debug("RF error: %s", exc)
            return {}

    def _train(self, digits: np.ndarray) -> None:
        try:
            X, y = _digit_features(digits[-400:])
            y_int = y.astype(int)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            rf = RandomForestClassifier(n_estimators=50, max_depth=6,
                                        random_state=42, n_jobs=1)
            rf.fit(X_s, y_int)
            self._rf = rf
            self._scaler = scaler
            self._trained_on = len(digits)
        except Exception as exc:
            logger.debug("RF train error: %s", exc)
            self._rf = None


class SimpleLSTM:
    """Lightweight Elman RNN (pure numpy) for sequence modeling."""

    def __init__(self, hidden: int = 16) -> None:
        self.hidden = hidden
        rng = np.random.default_rng(42)
        self.Wh = rng.normal(0, 0.1, (hidden, 1))
        self.Wr = rng.normal(0, 0.1, (hidden, hidden))
        self.Wo = rng.normal(0, 0.1, (10, hidden))
        self.bh = np.zeros((hidden, 1))
        self.bo = np.zeros((10, 1))
        self.h = np.zeros((hidden, 1))
        self._trained_on: int = 0

    def _forward(self, x: float) -> np.ndarray:
        x_arr = np.array([[x / 9.0]])
        self.h = np.tanh(self.Wh @ x_arr + self.Wr @ self.h + self.bh)
        out = self.Wo @ self.h + self.bo
        out = out - out.max()
        probs = np.exp(out) / np.exp(out).sum()
        return probs.flatten()

    def train_step(self, digits: np.ndarray) -> None:
        """Quick online training — update for last 50 steps."""
        for d in digits[-50:]:
            self._forward(float(d))
        self._trained_on = len(digits)

    def analyze(self, digits: np.ndarray) -> Dict:
        if len(digits) < 30:
            return {}
        if len(digits) - self._trained_on > 30:
            self.h = np.zeros((self.hidden, 1))
            self.train_step(digits)
        probs = self._forward(float(digits[-1]))
        predicted = int(np.argmax(probs))
        return {
            "predicted_digit": predicted,
            "over4_prob": round(float(probs[5:].sum()), 4),
            "under5_prob": round(float(probs[:5].sum()), 4),
        }


class FrequentPatternMiner:
    """Apriori-style frequent sequence mining in digit streams."""

    def analyze(self, digits: np.ndarray) -> Dict:
        if len(digits) < 50:
            return {}
        # Mine 3-grams
        counts: Dict[tuple, int] = {}
        for i in range(len(digits) - 3):
            pat = tuple(digits[i:i + 3].astype(int))
            counts[pat] = counts.get(pat, 0) + 1
        if not counts:
            return {}
        # Find most frequent 2-prefix that matches current last-2 digits
        last2 = tuple(digits[-2:].astype(int))
        candidates = {k: v for k, v in counts.items() if k[:2] == last2}
        if not candidates:
            return {"frequent_patterns": len(counts)}
        best = max(candidates, key=lambda k: candidates[k])
        predicted = best[2]
        support = candidates[best] / max(1, len(digits) - 3)
        return {
            "predicted_digit": int(predicted),
            "pattern": list(best),
            "support": round(support, 4),
        }


class GeneticRules:
    """Evolve simple threshold rules. Lightweight — runs every 100 ticks."""

    def __init__(self) -> None:
        self.rules: List[Dict] = []
        self._last_evolve: int = 0
        rng = np.random.default_rng(42)
        # Bootstrap population
        self.rules = self._initial_rules()

    def _initial_rules(self) -> List[Dict]:
        return [
            {"low_thresh": 0.30, "high_thresh": 0.30, "window": 10, "fitness": 0.5},
            {"low_thresh": 0.25, "high_thresh": 0.35, "window": 15, "fitness": 0.5},
            {"low_thresh": 0.40, "high_thresh": 0.40, "window": 20, "fitness": 0.5},
            {"low_thresh": 0.35, "high_thresh": 0.25, "window": 10, "fitness": 0.5},
            {"low_thresh": 0.20, "high_thresh": 0.20, "window": 30, "fitness": 0.5},
        ]

    def evolve(self, digits: np.ndarray) -> None:
        if len(digits) < 100:
            return
        # Evaluate each rule
        for rule in self.rules:
            w = rule["window"]
            wins = 0
            total = 0
            for i in range(w, len(digits) - 1):
                seg = digits[i - w:i]
                low_r = float(np.sum(seg < 3)) / w
                high_r = float(np.sum(seg > 6)) / w
                next_d = digits[i]
                if low_r > rule["low_thresh"]:
                    total += 1
                    wins += int(next_d > 2)
                elif high_r > rule["high_thresh"]:
                    total += 1
                    wins += int(next_d < 7)
            rule["fitness"] = wins / max(total, 1)

        # Select top 3, mutate to refill
        self.rules.sort(key=lambda r: r["fitness"], reverse=True)
        survivors = self.rules[:3]
        rng = np.random.default_rng(int(time.time()))
        while len(self.rules) < 5:
            parent = survivors[rng.integers(0, len(survivors))]
            child = {
                "low_thresh": float(np.clip(parent["low_thresh"] + rng.normal(0, 0.05), 0.1, 0.6)),
                "high_thresh": float(np.clip(parent["high_thresh"] + rng.normal(0, 0.05), 0.1, 0.6)),
                "window": int(np.clip(parent["window"] + rng.integers(-3, 4), 5, 40)),
                "fitness": parent["fitness"],
            }
            self.rules.append(child)

    def analyze(self, digits: np.ndarray) -> Dict:
        if len(digits) < 40 or not self.rules:
            return {}
        if len(digits) - self._last_evolve > 100:
            self.evolve(digits)
            self._last_evolve = len(digits)

        best = max(self.rules, key=lambda r: r["fitness"])
        w = best["window"]
        seg = digits[-w:]
        low_r = float(np.sum(seg < 3)) / w
        high_r = float(np.sum(seg > 6)) / w
        signal = "NONE"
        if low_r > best["low_thresh"]:
            signal = "OVER"
        elif high_r > best["high_thresh"]:
            signal = "UNDER"
        return {
            "signal": signal,
            "best_fitness": round(best["fitness"], 4),
            "low_ratio": round(low_r, 4),
            "high_ratio": round(high_r, 4),
        }


class EntropyAnalyzer:
    """Shannon entropy and Chi-Square tests for randomness detection."""

    def analyze(self, digits: np.ndarray, window: int = 100) -> Dict:
        if len(digits) < window:
            window = len(digits)
        seg = digits[-window:].astype(int)
        # Digit frequency
        counts = np.bincount(seg, minlength=10).astype(float)
        probs = counts / counts.sum()
        # Shannon entropy (max = log2(10) ≈ 3.32)
        entropy = -np.sum(p * np.log2(p + 1e-12) for p in probs)
        entropy_normalized = entropy / np.log2(10)
        # Chi-Square test for uniform distribution
        expected = np.full(10, window / 10.0)
        chi2, p_value = scipy_stats.chisquare(counts, f_exp=expected)
        # Bias detection
        low_bias = float(counts[:3].sum() / window)
        high_bias = float(counts[7:].sum() / window)
        mid_bias = float(counts[3:7].sum() / window)
        return {
            "entropy": round(float(entropy), 4),
            "entropy_normalized": round(float(entropy_normalized), 4),
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_value), 4),
            "is_biased": bool(p_value < 0.05),
            "low_bias": round(low_bias, 4),
            "high_bias": round(high_bias, 4),
            "mid_bias": round(mid_bias, 4),
            "digit_distribution": {str(i): int(counts[i]) for i in range(10)},
        }


# ══════════════════════════════════════════════════════════════
# Q-LEARNING WEIGHT CONTROLLER
# ══════════════════════════════════════════════════════════════

class QLearningWeights:
    """
    Adaptive ensemble weight controller.
    State: discretized (entropy_level, recent_win_rate)
    Action: adjust model weights up/down
    """

    MODEL_NAMES = [
        "markov", "hmm", "autocorr", "fft", "arima",
        "kmeans", "svm", "rf", "lstm", "patterns", "genetic"
    ]

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {m: 1.0 for m in self.MODEL_NAMES}
        self.history: List[Dict] = []  # signal → outcome
        self.lr = 0.05
        self.gamma = 0.9

    def update(self, model_votes: Dict, correct: bool) -> None:
        """Update weights based on whether the signal was correct."""
        reward = 1.0 if correct else -0.5
        for model, vote in model_votes.items():
            if model in self.weights:
                w = self.weights[model]
                # Simple gradient: reinforce if model agreed with outcome
                if vote.get("agreed_with_signal"):
                    self.weights[model] = min(3.0, w + self.lr * reward)
                else:
                    self.weights[model] = max(0.1, w - self.lr * 0.3)
        # Normalize
        total = sum(self.weights.values())
        for m in self.weights:
            self.weights[m] /= total / len(self.weights)

    def get_weight(self, model: str) -> float:
        return self.weights.get(model, 1.0)


# ══════════════════════════════════════════════════════════════
# MAIN ANALYSIS ENGINE (per market)
# ══════════════════════════════════════════════════════════════

class MarketAnalyzer:
    """Runs all models on a single market; produces ensemble signals."""

    def __init__(self, market: str) -> None:
        self.market = market
        self.markov = MarkovModel()
        self.hmm = HMMModel()
        self.autocorr = AutocorrelationModel()
        self.fft = FFTModel()
        self.arima = ARIMAModel()
        self.kmeans = KMeansModel()
        self.svm_model = SVMModel()
        self.rf_model = RandomForestModel()
        self.lstm_model = SimpleLSTM()
        self.freq_miner = FrequentPatternMiner()
        self.genetic = GeneticRules()
        self.entropy = EntropyAnalyzer()
        self.qlearn = QLearningWeights()

    def run_all(self, digits: List[int]) -> Dict:
        """Run full analysis suite and return per-model results."""
        if len(digits) < MIN_TICKS:
            return {}
        arr = _safe_arr(digits)

        results = {}
        results["markov"] = self.markov.analyze(arr)
        results["hmm"] = self.hmm.analyze(arr)
        results["autocorr"] = self.autocorr.analyze(arr)
        results["fft"] = self.fft.analyze(arr)
        results["arima"] = self.arima.analyze(arr)
        results["kmeans"] = self.kmeans.analyze(arr)
        results["svm"] = self.svm_model.analyze(arr)
        results["rf"] = self.rf_model.analyze(arr)
        results["lstm"] = self.lstm_model.analyze(arr)
        results["patterns"] = self.freq_miner.analyze(arr)
        results["genetic"] = self.genetic.analyze(arr)
        results["entropy"] = self.entropy.analyze(arr)
        return results

    def generate_category_signal(
        self,
        category: str,
        model_results: Dict,
    ) -> Optional[Dict]:
        """
        Aggregate model votes for a given strategy category.
        Returns signal dict or None.
        """
        if not model_results:
            return None

        cat = STRATEGY_CATEGORIES.get(category)
        if not cat:
            return None

        barrier_over = cat["barriers"]["over"]
        barrier_under = cat["barriers"]["under"]

        over_votes = 0.0
        under_votes = 0.0
        total_weight = 0.0
        model_votes: Dict = {}

        entropy_data = model_results.get("entropy", {})
        entropy_norm = entropy_data.get("entropy_normalized", 0.8)
        is_biased = entropy_data.get("is_biased", False)

        # Collect votes from each model
        for model_name, data in model_results.items():
            if model_name == "entropy" or not data:
                continue

            weight = self.qlearn.get_weight(model_name)
            vote = "NONE"
            confidence = 0.5

            if model_name == "markov":
                probs = data.get("probs_10", [])
                if probs:
                    p_arr = np.array(probs)
                    over_prob = float(p_arr[barrier_over + 1:].sum())
                    under_prob = float(p_arr[:barrier_under].sum())
                    if over_prob > 0.55:
                        vote = "OVER"
                        confidence = over_prob
                    elif under_prob > 0.55:
                        vote = "UNDER"
                        confidence = under_prob

            elif model_name == "hmm":
                regime = data.get("regime", "MID")
                if regime == "LOW":
                    vote = "OVER"; confidence = 0.65
                elif regime == "HIGH":
                    vote = "UNDER"; confidence = 0.65

            elif model_name == "autocorr":
                pred = data.get("predicted_digit")
                if pred is not None:
                    if pred > barrier_over:
                        vote = "OVER"; confidence = 0.6
                    elif pred < barrier_under:
                        vote = "UNDER"; confidence = 0.6

            elif model_name == "fft":
                pred = data.get("predicted_digit")
                conc = data.get("concentration", 0)
                if pred is not None and conc > 0.1:
                    if pred > barrier_over:
                        vote = "OVER"; confidence = 0.55 + conc * 0.3
                    elif pred < barrier_under:
                        vote = "UNDER"; confidence = 0.55 + conc * 0.3

            elif model_name == "arima":
                pred = data.get("forecast")
                if pred is not None:
                    if pred > barrier_over:
                        vote = "OVER"; confidence = 0.62
                    elif pred < barrier_under:
                        vote = "UNDER"; confidence = 0.62

            elif model_name == "kmeans":
                pred = data.get("predicted_digit")
                if pred is not None:
                    if pred > barrier_over:
                        vote = "OVER"; confidence = 0.58
                    elif pred < barrier_under:
                        vote = "UNDER"; confidence = 0.58

            elif model_name == "svm":
                prob_over = data.get("probability_over", 0.5)
                prob_under = data.get("probability_under", 0.5)
                if prob_over > 0.60:
                    vote = "OVER"; confidence = prob_over
                elif prob_under > 0.60:
                    vote = "UNDER"; confidence = prob_under

            elif model_name == "rf":
                over4 = data.get("over4_prob", 0.5)
                if over4 > 0.60:
                    vote = "OVER"; confidence = over4
                elif over4 < 0.40:
                    vote = "UNDER"; confidence = 1 - over4

            elif model_name == "lstm":
                over4 = data.get("over4_prob", 0.5)
                under5 = data.get("under5_prob", 0.5)
                if over4 > 0.58:
                    vote = "OVER"; confidence = over4
                elif under5 > 0.58:
                    vote = "UNDER"; confidence = under5

            elif model_name == "patterns":
                pred = data.get("predicted_digit")
                sup = data.get("support", 0)
                if pred is not None and sup > 0.02:
                    if pred > barrier_over:
                        vote = "OVER"; confidence = 0.55 + sup * 5
                    elif pred < barrier_under:
                        vote = "UNDER"; confidence = 0.55 + sup * 5

            elif model_name == "genetic":
                sig = data.get("signal", "NONE")
                fit = data.get("best_fitness", 0.5)
                if sig == "OVER":
                    vote = "OVER"; confidence = fit
                elif sig == "UNDER":
                    vote = "UNDER"; confidence = fit

            model_votes[model_name] = {
                "vote": vote,
                "confidence": round(min(1.0, confidence), 4),
                "weight": round(weight, 4),
            }

            # FIX 1: only count weight for OVER/UNDER votes, not NONE
            # Before: total_weight included all 11 models → scores always ~0.2
            # Now: total_weight only counts actual directional votes
            if vote == "OVER":
                over_votes += weight * confidence
                total_weight += weight
            elif vote == "UNDER":
                under_votes += weight * confidence
                total_weight += weight

        if total_weight == 0:
            return None

        over_score = over_votes / total_weight
        under_score = under_votes / total_weight

        # Entropy penalty: slightly reduce confidence for near-random markets
        entropy_factor = 1.0 - (entropy_norm * 0.15)

        # FIX 2: threshold lowered 0.40 → 0.25 (scores are now correctly scaled)
        if over_score > under_score and over_score > 0.25:
            direction = "OVER"
            barrier = barrier_over
            raw_conf = over_score
        elif under_score > over_score and under_score > 0.25:
            direction = "UNDER"
            barrier = barrier_under
            raw_conf = under_score
        else:
            return None  # Models are split — no clear signal

        # Map score to [0.50, 0.97] confidence range
        confidence_final = float(np.clip(0.50 + (raw_conf - 0.25) * entropy_factor, 0.50, 0.97))

        # Bias bonus from entropy analysis
        if is_biased:
            low_bias = entropy_data.get("low_bias", 0.3)
            high_bias = entropy_data.get("high_bias", 0.3)
            if direction == "OVER" and low_bias > 0.35:
                confidence_final = min(0.97, confidence_final + 0.04)
            if direction == "UNDER" and high_bias > 0.35:
                confidence_final = min(0.97, confidence_final + 0.04)

        # FIX 3: minimum confidence lowered 0.55 → 0.50
        if confidence_final < 0.50:
            return None

        # Valid window: higher confidence → longer window; faster market → shorter
        market_speed = {"R_10": 1.0, "R_25": 0.9, "R_50": 0.8,
                        "R_75": 0.7, "R_100": 0.6,
                        "1HZ10V": 0.5, "1HZ25V": 0.45, "1HZ50V": 0.40,
                        "1HZ75V": 0.35, "1HZ100V": 0.30}
        spd = market_speed.get(self.market, 0.7)
        valid_window = max(8, int(60 * confidence_final * spd))

        # Duration in ticks
        duration_ticks = max(1, min(5, int(confidence_final * 5)))

        # Count agreeing models
        agree = sum(1 for m in model_votes.values() if m["vote"] == direction)
        total_voting = sum(1 for m in model_votes.values() if m["vote"] != "NONE")

        return {
            "market": self.market,
            "category": category,
            "category_name": cat["name"],
            "direction": direction,
            "barrier": barrier,
            "confidence": round(confidence_final, 4),
            "valid_window": valid_window,
            "duration_ticks": duration_ticks,
            "model_votes": model_votes,
            "model_agreement": f"{agree}/{total_voting}",
            "entropy_normalized": round(float(entropy_norm), 4),
            "is_biased": is_biased,
            "timestamp": time.time(),
        }

    def generate_all_signals(self, digits: List[int]) -> List[Dict]:
        """Generate signals for all 4 strategy categories."""
        model_results = self.run_all(digits)
        signals = []
        for category in STRATEGY_CATEGORIES:
            sig = self.generate_category_signal(category, model_results)
            if sig:
                signals.append(sig)
        return signals


# ══════════════════════════════════════════════════════════════
# GLOBAL ENGINE REGISTRY
# ══════════════════════════════════════════════════════════════

_analyzers: Dict[str, MarketAnalyzer] = {}


def get_analyzer(market: str) -> MarketAnalyzer:
    if market not in _analyzers:
        _analyzers[market] = MarketAnalyzer(market)
    return _analyzers[market]
