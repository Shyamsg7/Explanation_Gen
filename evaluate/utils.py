from .rouge import rouge
from .bleu import compute_bleu
import mauve

def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def mauve_score(references, generated, num_buckets=25, pca_dim=75, kmeans_n_clusters=20, seed=25):
    """
    Compute MAUVE score to measure distribution similarity between generated and reference texts.
    
    Args:
        references: List of reference texts (strings)
        generated: List of generated texts (strings)
        num_buckets: Number of buckets for histogram (default: 25)
        pca_dim: PCA dimensionality reduction (default: 75)
        kmeans_n_clusters: Number of clusters for K-means (default: 20)
        seed: Random seed for reproducibility (default: 25)
    
    Returns:
        float: MAUVE score (0-100 scale)
    """
    mauve_out = mauve.compute_mauve(
        p_text=generated,
        q_text=references,
        device_id=0,  # Use GPU if available
        max_text_length=256,
        num_buckets=num_buckets,
        pca_max_data=pca_dim,
        kmeans_num_clusters=kmeans_n_clusters,
        seed=seed
    )
    return mauve_out.mauve * 100