"""
Orientation module from Paper: "Balancing Objectives in Counseling Conversations:
Advancing Forwards or Looking Backwards" Zhang et al. 2020, https://convokit.cornell.edu/
"""
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def weight_matrix(phrases: List[str], replies: List[str]) -> np.ndarray:
    X = phrases.extent(replies)
    vectorizer = TfidfVectorizer()

