import argparse
import numpy as np
from collections import defaultdict
from itertools import product
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def read_lncrna_ids(file_path: str) -> list:
    """Read lncRNA IDs from a tab-separated file (second column)."""
    with open(file_path, 'r') as f:
        return [line.split('\t')[1].strip() for line in f if line.strip()]


def process_fasta_file(fasta_file: str, lncrna_ids: list) -> dict:
    """
    Parse a FASTA file and collect sequences for the given lncRNA IDs.

    FASTA headers are expected to have the form '>base_id:part_num'.
    All sequence parts belonging to the same base_id are concatenated.

    Returns:
        dict mapping lncRNA ID -> list of sequence strings
    """
    id_set = set(lncrna_ids)
    sequences = defaultdict(list)
    current_id = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id].append(''.join(current_seq))
                header = line[1:].split(':')
                base_id = header[0]
                current_id = base_id if base_id in id_set else None
                current_seq = []
            elif current_id:
                current_seq.append(line)

    if current_id and current_seq:
        sequences[current_id].append(''.join(current_seq))

    return dict(sequences)


def _kmer_freq(sequence: str, k_values=(3, 4, 5)) -> dict:
    """Return normalised k-mer frequency dicts for each k."""
    result = {}
    for k in k_values:
        counts = defaultdict(int)
        for i in range(len(sequence) - k + 1):
            counts[sequence[i:i + k]] += 1
        total = sum(counts.values())
        result[k] = {kmer: cnt / total for kmer, cnt in counts.items()} if total > 0 else {}
    return result


def build_kmer_features(sequences: dict, k_values=(3, 4, 5)) -> tuple:
    """
    Build a k-mer feature matrix over all sequences.

    Returns:
        features: dict mapping lncRNA ID -> feature vector (list of floats)
        sorted_kmers: dict mapping k -> sorted list of k-mer strings
    """
    # Collect all observed k-mers
    kmer_sets = {k: set() for k in k_values}
    for seq_list in sequences.values():
        for seq in seq_list:
            for k, freq in _kmer_freq(seq, k_values).items():
                kmer_sets[k].update(freq.keys())

    sorted_kmers = {k: sorted(kmer_sets[k]) for k in k_values}

    # Build feature vectors
    features = {}
    for lnc_id, seq_list in sequences.items():
        combined = {k: defaultdict(float) for k in k_values}
        for seq in seq_list:
            for k, freq in _kmer_freq(seq, k_values).items():
                for kmer, val in freq.items():
                    combined[k][kmer] += val

        vec = []
        for k in k_values:
            total = sum(combined[k].values())
            if total > 0:
                vec.extend(combined[k].get(kmer, 0) / total for kmer in sorted_kmers[k])
            else:
                vec.extend([0.0] * len(sorted_kmers[k]))
        features[lnc_id] = vec

    return features, sorted_kmers


def reduce_dimensionality(feature_matrix: np.ndarray, n_components: int = 396):
    """Standardise and PCA-reduce the feature matrix."""
    scaled = StandardScaler().fit_transform(feature_matrix)
    scaled = SimpleImputer(strategy='mean').fit_transform(scaled)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled)
    return reduced, pca


def save_features(lncrna_ids: list, reduced: np.ndarray, output_file: str):
    """Write PCA-reduced features to a tab-separated file with a header."""
    with open(output_file, 'w') as f:
        header = ['lncRNA_ID'] + [f'PC{i + 1}' for i in range(reduced.shape[1])]
        f.write('\t'.join(header) + '\n')
        for lnc_id, row in zip(lncrna_ids, reduced):
            f.write('\t'.join([lnc_id] + [str(x) for x in row]) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Compute k-mer sequence features for lncRNAs')
    parser.add_argument('--id_file',        default='LncRNA_ID.txt',
                        help='Tab-separated file containing lncRNA IDs (second column)')
    parser.add_argument('--fasta_file',     default='LncRNA_Sq.fasta',
                        help='FASTA file containing lncRNA sequences')
    parser.add_argument('--output_file',    default='LncRNA/Feature_LncRNA_sq.txt',
                        help='Output path for PCA-reduced features')
    parser.add_argument('--k_values',       nargs='+', type=int, default=[3, 4, 5],
                        help='k-mer lengths to compute (default: 3 4 5)')
    parser.add_argument('--pca_components', type=int, default=396,
                        help='Number of PCA components (default: 396)')
    args = parser.parse_args()

    lncrna_ids = read_lncrna_ids(args.id_file)
    print(f"Loaded {len(lncrna_ids)} lncRNA IDs")

    sequences = process_fasta_file(args.fasta_file, lncrna_ids)
    print(f"Found sequences for {len(sequences)} lncRNAs")

    features, _ = build_kmer_features(sequences, args.k_values)
    print(f"Built k-mer features for {len(features)} lncRNAs")

    valid_ids = [lid for lid in lncrna_ids if lid in features]
    feature_matrix = np.array([features[lid] for lid in valid_ids])

    reduced, pca = reduce_dimensionality(feature_matrix, args.pca_components)
    print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")

    save_features(valid_ids, reduced, args.output_file)
    print(f"Features saved to {args.output_file}")


if __name__ == '__main__':
    main()
