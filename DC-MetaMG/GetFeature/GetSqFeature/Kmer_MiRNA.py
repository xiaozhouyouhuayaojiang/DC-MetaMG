import argparse
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def generate_all_kmers(k_values=(2, 3, 4, 5)):
    """Return a list of all k-mer strings for the given k values over the RNA alphabet."""
    return [''.join(p) for k in k_values for p in product('ACGU', repeat=k)]


def count_kmers(sequence: str, k_values=(2, 3, 4, 5)) -> dict:
    """Count occurrences of every k-mer in *sequence*."""
    counts = defaultdict(int)
    for k in k_values:
        for i in range(len(sequence) - k + 1):
            counts[sequence[i:i + k]] += 1
    return counts


def create_multi_kmer_matrix(sequences: dict, k_values=(2, 3, 4, 5), normalize=True) -> pd.DataFrame:
    """
    Build a k-mer frequency matrix for a collection of RNA sequences.

    Args:
        sequences: mapping of name -> sequence string
        k_values:  tuple/list of k values to include
        normalize: if True, divide counts by sequence length

    Returns:
        DataFrame of shape (n_valid_sequences, n_kmers)
    """
    # Filter sequences that are too short
    valid_seqs = {}
    n_filtered = 0
    for name, seq in sequences.items():
        cleaned = ''.join(c.upper() for c in seq if c.upper() in 'ACGU')
        if len(cleaned) < max(k_values):
            n_filtered += 1
            continue
        valid_seqs[name] = cleaned

    if n_filtered > 0:
        print(f"Filtered {n_filtered} sequences (too short for k={max(k_values)})")

    # Build frequency matrix
    all_kmers = generate_all_kmers(k_values)
    matrix = pd.DataFrame(0, index=valid_seqs.keys(), columns=all_kmers)

    for name, seq in valid_seqs.items():
        for kmer, count in count_kmers(seq, k_values).items():
            matrix.loc[name, kmer] = count

    # Normalize by sequence length
    if normalize:
        lengths = pd.Series({name: len(seq) for name, seq in valid_seqs.items()})
        assert (lengths > 0).all(), "Zero-length sequences found after filtering"
        matrix = matrix.div(lengths, axis=0)

    # Drop all-zero columns and add small epsilon
    matrix = matrix.loc[:, matrix.sum(axis=0) > 0] + 1e-6

    assert not matrix.isna().any().any(), "NaN values remain in k-mer matrix"
    return matrix


def reduce_dimensionality(matrix: pd.DataFrame, n_components: int = 50):
    """
    Standardise and PCA-reduce the k-mer matrix.

    Returns:
        reduced:         numpy array of shape (n_seq, n_components)
        variance_ratio:  explained variance ratio per component
    """
    scaled = StandardScaler().fit_transform(matrix)
    scaled = SimpleImputer(strategy='mean').fit_transform(scaled)
    reduced = PCA(n_components=n_components).fit_transform(scaled)
    return reduced


def read_mirna_data(file_path: str) -> dict:
    """Read miRNA name-sequence pairs from a tab-separated file."""
    sequences = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                sequences[parts[0]] = parts[1]
    return sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute k-mer sequence features for miRNAs")
    parser.add_argument('--input',       type=str, default='MiRNA_Sq.txt')
    parser.add_argument('--out_dir',     type=str, default='MiRNA')
    parser.add_argument('--k_values',    type=int, nargs='+', default=[3, 4, 5, 6])
    parser.add_argument('--n_components', type=int, default=198)
    args = parser.parse_args()

    mirna_sequences = read_mirna_data(args.input)
    kmer_matrix = create_multi_kmer_matrix(mirna_sequences, k_values=args.k_values, normalize=True)
    print(f"k-mer matrix shape: {kmer_matrix.shape}, NaN count: {kmer_matrix.isna().sum().sum()}")

    reduced = reduce_dimensionality(kmer_matrix, n_components=args.n_components)

    kmer_matrix.to_csv(f"{args.out_dir}/multi_kmer_matrix.csv")
    with open(f"{args.out_dir}/Feature_MiRNA_sq", 'w') as f:
        for row in reduced:
            f.write(' '.join(map(str, row)) + '\n')

    print(f"Saved features to {args.out_dir}/Feature_MiRNA_sq")
