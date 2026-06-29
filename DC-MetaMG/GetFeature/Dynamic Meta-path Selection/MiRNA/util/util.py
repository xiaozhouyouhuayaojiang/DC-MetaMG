from collections import defaultdict
import numpy as np
import random
import torch

from util.getValue import getValue


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def build_adjacency(object_disease_matrix: np.ndarray) -> np.ndarray:
    """
    Build a symmetric adjacency matrix where two nodes are adjacent if they
    share at least one disease association.

    Args:
        object_disease_matrix: binary matrix of shape (n_nodes, n_diseases)

    Returns:
        adj: int8 array of shape (n_nodes, n_nodes)
    """
    n = object_disease_matrix.shape[0]
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if np.any((object_disease_matrix[i] > 0) & (object_disease_matrix[j] > 0)):
                adj[i][j] = adj[j][i] = 1
    return adj


# ---------------------------------------------------------------------------
# DFS exhaustive path enumeration
# ---------------------------------------------------------------------------

def dfs_all_paths(adjacency: np.ndarray, start: int, max_hops: int) -> dict:
    """
    Exhaustive DFS traversal from *start*, collecting every simple path up to
    *max_hops* edges long.

    Args:
        adjacency: symmetric binary adjacency matrix (n_nodes, n_nodes)
        start:     index of the starting node
        max_hops:  maximum number of hops (edges) allowed in a path

    Returns:
        paths: dict mapping (start, end) -> list of paths,
               where each path is a list of node indices [start, ..., end]
    """
    n = adjacency.shape[0]
    paths: dict = defaultdict(list)

    def _dfs(current: int, path: list, visited: set):
        # Record every intermediate/terminal node reached from start
        if len(path) > 1:
            end = path[-1]
            paths[(start, end)].append(list(path))

        # Stop if we've already taken max_hops edges
        if len(path) - 1 >= max_hops:
            return

        for neighbor in range(n):
            if neighbor not in visited and adjacency[current][neighbor]:
                visited.add(neighbor)
                _dfs(neighbor, path + [neighbor], visited)
                visited.remove(neighbor)

    _dfs(start, [start], {start})
    return dict(paths)


# ---------------------------------------------------------------------------
# Path scoring — formulae (5) and (6) from the paper
# ---------------------------------------------------------------------------

def scoring_function(s_ab: float, theta: float, alpha: float = 2.0) -> float:
    """
    Formula (6): F(s_{a,b}) = sign(s_{a,b} - θ) · |s_{a,b} - θ|^α

    Applies a polynomial penalty below θ and a sublinear incentive above θ,
    realising the asymmetric mechanism described in the paper.
    """
    diff = s_ab - theta
    return float(np.sign(diff) * (np.abs(diff) ** alpha))


def score_path(path: list, similarity_matrix: np.ndarray,
               theta: float, alpha: float = 2.0, beta: float = 1.0) -> float:
    """
    Score a single path using formula (5):

        score(p) = Σ_{b ∈ p, b ≠ a}  e^{-β · d_{a,b}} · F(s_{a,b})

    where d_{a,b} is the hop distance from the start node a to b along p,
    and F is the scoring function (formula 6).

    Args:
        path:              list of node indices [a, ..., c]
        similarity_matrix: pairwise similarity matrix s = G · G^T
        theta:             similarity threshold determined by KMeans
        alpha:             exponent in formula (6), paper sets α = 2
        beta:              distance-decay coefficient in formula (5)

    Returns:
        Scalar path score.
    """
    a = path[0]
    total = 0.0
    for hop_idx, b in enumerate(path[1:], start=1):
        s_ab = similarity_matrix[a][b]
        f_s = scoring_function(s_ab, theta, alpha)
        decay = np.exp(-beta * hop_idx)
        total += decay * f_s
    return total


def get_state_value(paths_a_to_c: list, similarity_matrix: np.ndarray,
                    theta: float, alpha: float = 2.0, beta: float = 1.0) -> float:
    """
    Formula (5): ρ_{a,c} = max_{p ∈ P_{a→c}} Σ_{b ∈ p} e^{-β·d_{a,b}} · F(s_{a,b})

    Returns -inf when no paths exist (node pair is unreachable).
    """
    if not paths_a_to_c:
        return -np.inf
    return max(score_path(p, similarity_matrix, theta, alpha, beta)
               for p in paths_a_to_c)


# ---------------------------------------------------------------------------
# Main meta-path extraction
# ---------------------------------------------------------------------------

def getMetaPath(num: int, object_disease_matrix: np.ndarray,
                correlation_file: str, max_hops: int = 3,
                alpha: float = 2.0, beta: float = 1.0) -> np.ndarray:
    """
    Build the homogeneous interaction matrix via DFS path enumeration and
    the path-scoring formulae (5) & (6).

    A pair (a, c) receives an edge (DDI[a][c] = 1) if and only if the state
    value ρ_{a,c} > 0, meaning at least one path from a to c carries a net
    positive score after distance decay and asymmetric similarity scoring.

    Args:
        num:                    number of nodes (drugs or miRNAs)
        object_disease_matrix:  binary association matrix (num × n_diseases)
        correlation_file:       path to the pre-computed correlation/similarity
                                matrix (output of Preprocess scripts)
        max_hops:               DFS depth limit (3 for drugs, 2 for miRNAs)
        alpha:                  exponent for formula (6); paper uses α = 2
        beta:                   distance-decay coefficient for formula (5)

    Returns:
        DDI: symmetric binary numpy array of shape (num, num)
    """
    # --- θ from KMeans clustering on the similarity distribution ---
    theta = getValue(correlation_file)

    # --- Load similarity matrix s = G · G^T ---
    similarity_matrix = np.loadtxt(correlation_file, dtype=float)

    # --- Build shared-disease adjacency ---
    adjacency = build_adjacency(object_disease_matrix)

    DDI = np.zeros((num, num), dtype=int)

    for a in range(num):
        # DFS from node a: enumerate all simple paths within max_hops
        all_paths = dfs_all_paths(adjacency, a, max_hops)

        for c in range(a + 1, num):
            paths_ac = all_paths.get((a, c), [])
            rho = get_state_value(paths_ac, similarity_matrix, theta, alpha, beta)
            if rho > 0:
                DDI[a][c] = 1
                DDI[c][a] = 1

    return DDI
