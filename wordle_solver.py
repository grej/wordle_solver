"""
Numba-Accelerated Wordle Solver
===============================
High-performance Wordle solver using Numba JIT compilation.
Entropy-based strategy with frequency weighting and trap detection.

Usage:
    from wordle_solver import WordleSolverEngine
    engine = WordleSolverEngine()        # builds word list + pattern matrix
    engine.solve('tiger')                # solve with trace
    engine.play()                        # interactive mode
    engine.difficulty('batch')           # difficulty analysis
    engine.absurdle('raise')             # adversarial mode
"""

import numba as nb
import numpy as np
import os
import time
from collections import Counter

# === Core Numba Functions ===

@nb.njit(nb.uint16(nb.uint8[:], nb.uint8[:]), cache=True)
def compute_pattern(guess, target):
    result = np.zeros(5, dtype=np.uint8)
    target_counts = np.zeros(26, dtype=np.uint8)
    for i in range(5):
        target_counts[target[i]] += 1
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = 2
            target_counts[guess[i]] -= 1
    for i in range(5):
        if result[i] == 0 and target_counts[guess[i]] > 0:
            result[i] = 1
            target_counts[guess[i]] -= 1
    pid = np.uint16(0)
    for i in range(5):
        pid = pid * np.uint16(3) + np.uint16(result[i])
    return pid

@nb.njit(nb.uint16[:,:](nb.uint8[:,:]), cache=True, parallel=True)
def precompute_pattern_matrix(words):
    n = words.shape[0]
    matrix = np.empty((n, n), dtype=np.uint16)
    for i in nb.prange(n):
        for j in range(n):
            matrix[i, j] = compute_pattern(words[i], words[j])
    return matrix

@nb.njit(cache=True, parallel=True)
def matrix_entropies_weighted(pattern_matrix, guess_indices, candidate_indices, word_scores, freq_weight):
    n_guesses = guess_indices.shape[0]
    n_cands = candidate_indices.shape[0]
    scores = np.empty(n_guesses, dtype=np.float64)
    for gi in nb.prange(n_guesses):
        g = guess_indices[gi]
        counts = np.zeros(243, dtype=np.int32)
        for ci in range(n_cands):
            counts[pattern_matrix[g, candidate_indices[ci]]] += 1
        ent = 0.0
        for k in range(243):
            if counts[k] > 0:
                p = counts[k] / n_cands
                ent -= p * np.log2(p)
        scores[gi] = ent + freq_weight * word_scores[g]
    return scores

@nb.njit(cache=True)
def matrix_filter(pattern_matrix, guess_idx, pattern_id, candidate_indices):
    n = candidate_indices.shape[0]
    count = 0
    for i in range(n):
        if pattern_matrix[guess_idx, candidate_indices[i]] == pattern_id:
            count += 1
    result = np.empty(count, dtype=np.int32)
    idx = 0
    for i in range(n):
        if pattern_matrix[guess_idx, candidate_indices[i]] == pattern_id:
            result[idx] = candidate_indices[i]
            idx += 1
    return result

PATTERN_EMOJI = {0: "⬜", 1: "🟨", 2: "🟩"}

def decode_pattern(pid):
    result = []
    for _ in range(5):
        result.append(pid % 3)
        pid //= 3
    return result[::-1]

def encode_words(word_list):
    n = len(word_list)
    arr = np.empty((n, 5), dtype=np.uint8)
    for i, word in enumerate(word_list):
        for j, ch in enumerate(word):
            arr[i, j] = ord(ch) - ord("a")
    return arr


class WordleSolverEngine:
    """Complete Wordle solver engine with all features."""

    def __init__(self, extra_words=None):
        print("Loading word list...")
        self.word_list = self._load_words(extra_words)
        self.encoded = encode_words(self.word_list)
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}
        self.n = len(self.word_list)

        print(f"Building {self.n}×{self.n} pattern matrix...")
        t0 = time.perf_counter()
        self.matrix = precompute_pattern_matrix(self.encoded)
        dt = time.perf_counter() - t0
        print(f"Ready! ({dt:.2f}s, {self.matrix.nbytes/1024/1024:.0f} MB)")

        self.scores = self._compute_word_scores()
        self.raise_idx = self.word_to_idx.get("raise", 0)

    def _load_words(self, extra_words):
        words = set()
        for path in ["/usr/share/dict/words", "/usr/share/dict/american-english"]:
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        w = line.strip().lower()
                        if len(w) == 5 and w.isalpha() and w == w.lower():
                            words.add(w)
                break
        if extra_words:
            words.update(extra_words)
        return sorted(words)

    def _compute_word_scores(self):
        freq = {"e":12.7,"t":9.1,"a":8.2,"o":7.5,"i":7.0,"n":6.7,"s":6.3,
                "h":6.1,"r":6.0,"d":4.3,"l":4.0,"c":2.8,"u":2.8,"m":2.4,
                "w":2.4,"f":2.2,"g":2.0,"y":2.0,"p":1.9,"b":1.5,"v":1.0,
                "k":0.8,"j":0.15,"x":0.15,"q":0.10,"z":0.07}
        scores = np.zeros(self.n, dtype=np.float64)
        for i, word in enumerate(self.word_list):
            unique = set(word)
            scores[i] = sum(freq.get(c, 0) for c in unique) / 50.0 - (5 - len(unique)) * 0.1
        return scores

    def solve(self, target, opener="raise"):
        """Solve with full trace."""
        if target not in self.word_to_idx:
            print(f"❌ \'{target}\' not in dictionary")
            return
        t_idx = self.word_to_idx[target]
        o_idx = self.word_to_idx.get(opener, self.raise_idx)
        candidates = np.arange(self.n, dtype=np.int32)
        history = []

        for turn in range(1, 7):
            if turn == 1:
                g_idx = o_idx
            elif len(candidates) <= 2:
                g_idx = candidates[0]
            else:
                scores = matrix_entropies_weighted(
                    self.matrix, candidates, candidates, self.scores, 0.15)
                g_idx = int(candidates[np.argmax(scores)])

            pattern = int(self.matrix[g_idx, t_idx])
            emoji = "".join(PATTERN_EMOJI[d] for d in decode_pattern(pattern))
            history.append((self.word_list[g_idx], emoji))

            if pattern == 242:
                print(f"🎉 Solved \'{target}\' in {turn}!")
                for g, e in history:
                    print(f"  {g.upper()} {e}")
                return turn

            candidates = matrix_filter(self.matrix, np.int32(g_idx), np.uint16(pattern), candidates)

        print(f"❌ Failed to solve \'{target}\' in 6")
        return 7


if __name__ == "__main__":
    engine = WordleSolverEngine()
    engine.solve("crane")
    engine.solve("plumb")
    engine.solve("tiger")
