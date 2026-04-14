"""
text.py

Helper functions for text-related operations for Natural Language Processing.

Course: CSCI 357 - AI and Neural Networks
Author: Chang Min Bark
"""

import pathlib
import urllib.request
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.functional import cosine_similarity
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset


class GloVeVocab:
    """A lightweight container for pretrained GloVe word vectors.

    Stores the vocabulary and embedding matrix so that individual word vectors
    can be retrieved by name. The interface intentionally mirrors the legacy
    torchtext.vocab.GloVe API (.stoi, .itos, .vectors, .dim, __getitem__) so
    that downstream code — nearest-neighbor search, analogy arithmetic, and
    embedding matrix construction — works without modification.

    Attributes:
        stoi (dict[str, int]): Maps each word to its row index in `vectors`.
        itos (list[str]):      Maps each row index back to its word.
        vectors (torch.Tensor): Shape (vocab_size, dim). Row i is the embedding
                                for the word itos[i].
        dim (int): Embedding dimensionality (e.g. 50, 100, 200, or 300).
    """

    def __init__(self, words: List[str], vectors: torch.Tensor) -> None:
        """Build lookup structures from an ordered word list and a vector matrix.

        Args:
            words: Ordered list of vocabulary strings. The position of a word
                in this list is its row index in ``vectors``.
            vectors: Tensor of shape (len(words), dim) containing the embeddings.
        """
        # Store a copy of the word list as self.itos ("index to string").
        #       This lets us map an integer index back to a word.
        self.itos = words

        # Build self.stoi ("string to index") as a dict that maps each word
        #       to its integer position in self.itos.
        #       Hint: use enumerate(self.itos) so that idx matches the row in vectors.
        self.stoi = {}
        for idx, word in enumerate(self.itos):
            self.stoi[word] = idx

        # Store the embedding matrix as self.vectors.
        self.vectors = vectors

        # Store the embedding dimension as self.dim.
        #       Hint: vectors has shape (vocab_size, dim) — use vectors.shape[1].
        self.dim = vectors.shape[1]

    def __len__(self) -> int:
        """Return the vocabulary size (total number of words)."""
        # Return the number of entries in self.itos.
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        """Support the `word in glove` membership test."""
        return word in self.stoi

    def __getitem__(self, word: str) -> torch.Tensor:
        """Return the embedding vector for a word.

        Args:
            word: The word to look up.

        Returns:
            Tensor of shape (dim,) — a 1D vector of floats.

        Raises:
            KeyError: If the word is not in the vocabulary.
        """
        # Look up the word's index with self.stoi, then return the
        #       corresponding row from self.vectors.
        #       Raise a KeyError with a helpful message if the word is missing.
        if word not in self.stoi:
            raise KeyError(f"Word {word} missing from GloVe vocabulary")
        idx = self.stoi[word]

        return self.vectors[idx]


def load_glove_vectors(glove_dir: str = "./data/glove", dim: int = 100) -> GloVeVocab:
    """Download (if needed) and load GloVe 6B word vectors from a plain text file.

    GloVe 6B vectors are distributed by Stanford NLP as a zip archive (~822 MB):
        https://nlp.stanford.edu/data/glove.6B.zip

    The archive contains four text files:
        glove.6B.50d.txt   (50-dimensional vectors)
        glove.6B.100d.txt  (100-dimensional vectors)
        glove.6B.200d.txt  (200-dimensional vectors)
        glove.6B.300d.txt  (300-dimensional vectors)

    Only the file matching `dim` is extracted. Everything is cached under
    `glove_dir` so the download happens only once.

    Text file format — one line per word:
        word  val1  val2  ...  val_dim

    Args:
        glove_dir: Directory where GloVe files will be stored / read from.
                   Defaults to "./data/glove". Created automatically if absent.
        dim:       Embedding dimensionality. One of {50, 100, 200, 300}.

    Returns:
        GloVeVocab instance with .stoi, .itos, .vectors, and .dim populated.
    """
    glove_dir = pathlib.Path(glove_dir)
    glove_dir.mkdir(parents=True, exist_ok=True)

    txt_file = glove_dir / f"glove.6B.{dim}d.txt"
    zip_file = glove_dir / "glove.6B.zip"

    # --- Step 1: Download and extract if the .txt file is not already cached ---
    if not txt_file.exists():
        if not zip_file.exists():
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            print("Downloading GloVe 6B (~822 MB) from Stanford NLP...")
            print(f"  Saving to: {zip_file}")
            print("  (This only happens once — the file will be cached.)")

            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                pct = 100 * downloaded / total_size if total_size > 0 else 0
                print(
                    f"  {downloaded / 1e6:6.1f} / {total_size / 1e6:.1f} MB  ({pct:.1f}%)",
                    end="\r",
                )

            urllib.request.urlretrieve(url, zip_file, reporthook=_progress)
            print("\nDownload complete.")

        target_name = f"glove.6B.{dim}d.txt"
        print(f"Extracting {target_name} from archive...")

        # Open zip_file as a ZipFile and extract only target_name into glove_dir.
        #       Hint: use `with zipfile.ZipFile(zip_file) as zf:` then zf.extract().
        with zipfile.ZipFile(zip_file) as zf:
            zf.extract(target_name, glove_dir)
        print("Extraction complete.")
    else:
        print(f"Found cached {txt_file.name} — loading...")

    # --- Step 2: Parse the text file into word list and vector matrix ---
    #
    # We read line by line to keep memory usage predictable. Each line is split
    # on whitespace: the first token is the word, the rest are float strings.
    # We accumulate 1D tensors in a list, then stack them into a 2D matrix at
    # the end (one torch.stack call is far faster than repeated concatenation).

    words = []
    raw_vectors = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            # Strip the trailing newline and split the line on spaces.
            #       tokens[0] is the word; tokens[1:] are the float-string values.
            values = line.strip().split()
            word = values[0]
            tokens = values[1:]

            # Convert the numeric string tokens to a 1D float32 tensor.
            #       Hint: torch.tensor([float(v) for v in tokens[1:]], dtype=torch.float32)
            vec = torch.tensor([float(v) for v in tokens], dtype=torch.float32)

            # Append `word` to `words` and `vec` to `raw_vectors`.
            words.append(word)
            raw_vectors.append(vec)

    # Use torch.stack() to combine the list of 1D tensors into a single
    #       2D tensor of shape (vocab_size, dim). This is more efficient than
    #       building the matrix row by row — torch.stack() allocates one large
    #       contiguous block of memory and fills it in one shot.
    vectors_tensor = torch.stack(raw_vectors, dim=0)

    print(f"Loaded {len(words):,} vectors of dimension {dim}.")
    return GloVeVocab(words, vectors_tensor)


def build_glove_embedding_matrix(
    vocab: Dict[str, int],
    glove: GloVeVocab,
    embedding_dim: int = 100,
) -> torch.Tensor:
    """Build an embedding matrix aligned to a given vocabulary.

    For each word in the vocabulary, if it exists in the GloVe embeddings, use
    the pretrained GloVe vector. Otherwise, initialize randomly from N(0, 0.1).
    The PAD token (index 0) is always set to the zero vector.

    Args:
        vocab: Vocabulary mapping words to integer indices.
        glove: Pretrained GloVe word vectors container.
        embedding_dim: Dimensionality of the embeddings. Defaults to 100.

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim) where row i
        contains the embedding for the word at index i.
    """
    vocab_size = len(vocab)

    # Initialize an embedding matrix of zeros with shape (vocab_size, embedding_dim).
    #       This will hold the final embeddings for all words in our vocabulary.
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))

    # Initialize a counter to track how many words we find in GloVe.
    found_in_glove = 0

    # Loop through each word and its index in the vocabulary.
    for word, idx in vocab.items():
        # Skip index 0 (the PAD token) — it should remain all zeros.
        if idx == 0:
            continue

        # Check if the word exists in the GloVe vocabulary (glove.stoi).
        #       If it does, copy the pretrained GloVe vector into our matrix.
        if word in glove:
            embedding_matrix[idx] = glove[word]
            found_in_glove += 1

        # If the word is not in GloVe, initialize it randomly.
        #       Use a normal distribution N(0, 0.1) by sampling from torch.randn
        #       and scaling by 0.1.
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1

    print(
        f"GloVe coverage: {found_in_glove}/{vocab_size} words ({100 * found_in_glove / vocab_size:.1f}%)"
    )
    return embedding_matrix


def generate_skip_gram_pairs(
    token_id_sequences: Dataset, window_size: int = 2
) -> TensorDataset:
    """Generate (center, context) skip-gram pairs from encoded token-ID sequences.

    For each token in a sequence (excluding the first and last ``window_size``
    positions), emits one pair for every context token within ``window_size``
    steps of the center token. Both center and context are integer token IDs.

    Args:
        token_id_sequences: TextDataset whose ``.samples`` attribute is a list
            of (token_ids_tensor, label_tensor) tuples. Only the token IDs are used.
        window_size: Number of tokens to the left and right of each center token
            to treat as context. Defaults to 2.

    Returns:
        TensorDataset of two aligned 1-D tensors ``(centers, contexts)``, each
        of length equal to the total number of skip-gram pairs generated.
    """
    # Generate skip gram pairs
    # Generate skip-gram pairs
    all_centers = []
    all_contexts = []
    for sample in token_id_sequences.samples:
        token_ids = sample[0]  # shape: (n,)
        # Use unfold to extract all windows at once
        windows = token_ids.unfold(0, 2 * window_size + 1, 1)  # (n - 2*ws, 2*ws+1)
        centers = windows[:, window_size]  # middle column
        # context = all columns except center
        context_cols = torch.cat(
            [windows[:, :window_size], windows[:, window_size + 1 :]], dim=1
        )
        center_expanded = centers.unsqueeze(1).expand_as(context_cols)
        all_centers.append(center_expanded.reshape(-1))
        all_contexts.append(context_cols.reshape(-1))

    # Wrap pairs in a PyTorch TensorDataset (TensorDataset(centers_tensor, contexts_tensor))
    centers = torch.cat(all_centers)
    contexts = torch.cat(all_contexts)
    return TensorDataset(centers, contexts)


def find_nearest_neighbors(glove: GloVeVocab, word: str, k: int = 10) -> None:
    """Find the k nearest neighbors of a word in GloVe embedding space.

    Computes cosine similarity between the query word's embedding and all other
    word embeddings, then returns the k most similar words (excluding the query
    word itself).

    Args:
        glove: GloVe vocabulary object containing word embeddings.
        word: The query word to find neighbors for.
        k: Number of nearest neighbors to return. Defaults to 10.

    Returns:
        None. Prints the k nearest neighbors with their cosine similarity scores.
    """
    # Check if the word exists in the GloVe vocabulary. If not, print a
    #       message and return early.
    if word not in glove:
        print(f"'{word}' not in GloVe vocabulary")
        return

    # Get the embedding vector for the query word and add a batch dimension
    #       using unsqueeze(0) so it has shape (1, dim).
    query_vec = glove[word].unsqueeze(0)

    # Compute cosine similarity between the query vector and all vectors in
    #       glove.vectors. This returns a 1D tensor of similarity scores.
    cosine_sims = cosine_similarity(query_vec, glove.vectors)

    # Sort the similarity scores in descending order and get the top k+1 indices.
    #       We use k+1 because the first result will be the query word itself (similarity=1.0),
    #       which we want to skip. Slice [1:k+1] to get the k nearest neighbors.
    top_scores, top_indices = torch.topk(cosine_sims, k=k + 1)

    # Print the results in a formatted table showing each neighbor and its similarity.
    print(f"Nearest neighbors of '{word}':")
    for score, idx in zip(top_scores[1:], top_indices[1:]):  # skip the query word
        print(f"  {glove.itos[idx]:<15} {score.item():.4f}")
    print()


def analogy(glove: GloVeVocab, a: str, b: str, c: str, k: int = 5) -> None:
    """Solve word analogy using vector arithmetic in embedding space.

    Computes the vector b - a + c and finds the k nearest neighbors to answer
    the analogy "a is to b as c is to ?". For example, "man is to king as
    woman is to ?" should yield "queen".

    Args:
        glove: GloVeVocab instance containing word embeddings.
        a: First word in the analogy (e.g., ``"man"``).
        b: Second word in the analogy (e.g., ``"king"``).
        c: Third word in the analogy (e.g., ``"woman"``).
        k: Number of nearest neighbors to return. Defaults to 5.

    Returns:
        None. Prints the top k analogical completions with cosine similarities.
    """
    # Compute the analogy vector using vector arithmetic: b - a + c
    #       This represents the direction from a to b, applied to c.
    # Look up vectors for each word
    vec_a, vec_b, vec_c = glove[a], glove[b], glove[c]
    analogy_vec = vec_b - vec_a + vec_c  # shape (dim,)

    # Compute cosine similarities between the analogy vector and all GloVe vectors.
    #       Hint: Use F.cosine_similarity with vec.unsqueeze(0) to add a batch dimension.
    cosine_sims = cosine_similarity(analogy_vec.unsqueeze(0), glove.vectors)

    # Create a set of indices to exclude (the input words a, b, c) from results.
    #       We don't want the analogy to return the words we used as inputs.
    excluded = {glove.stoi[w] for w in (a, b, c) if w in glove.stoi}

    # Set the similarity scores for excluded words to -1 so they won't be selected.
    for idx in excluded:
        cosine_sims[idx] = -1.0

    # Find the top k words with highest similarity scores.
    #       Hint: Use argsort(descending=True) and slice [:k].
    top_scores, top_indices = torch.topk(cosine_sims, k=k)

    # Print the analogy question and the top k results with their similarity scores.
    print(f"'{a}' is to '{b}' as '{c}' is to ...?")
    for score, idx in zip(top_scores, top_indices):
        print(f"  {glove.itos[idx]:<15} {score.item():.4f}")
    print()


def text_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    padding_value: int = 0,
    max_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for variable-length text sequences.

    Pads all sequences in the batch to the length of the longest sequence,
    using 0 (the PAD index) as the padding value.

    Args:
        batch: List of (token_ids_tensor, label_tensor) tuples from TextDataset.
        padding_value: Index used for padding (default 0, the PAD token).
        max_seq_len: If set, each sequence is truncated to at most this many
            tokens (first ``max_seq_len`` tokens are kept). Use this for
            memory-heavy models such as self-attention, whose memory scales
            as ``O(batch * L^2)`` in sequence length ``L``.

    Returns:
        Tuple of (padded_sequences, labels) where padded_sequences has shape
        (batch_size, max_seq_len) and labels has shape (batch_size,).
    """
    texts, labels = zip(*batch)

    # If max_seq_len is specified, truncate each sequence to this maximum length.
    if max_seq_len is not None:
        texts = tuple(t[:max_seq_len] for t in texts)

    # Pad sequences to the same length (that of the longest sequence in the batch),
    # using padding_value (default is 0 for <PAD> token). Result is (batch_size, max_seq_len).
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=padding_value)

    # Stack the labels into a single tensor of shape (batch_size,).
    labels = torch.stack(labels)

    # Return the padded text sequences and the labels as a tuple.
    return padded_texts, labels


def build_vocab(
    tokenized_texts: List[List[str]], max_vocab_size: int = 25000, min_freq: int = 2
) -> Dict[str, int]:
    """Build a word-to-index vocabulary from tokenized texts.

    Reserves index 0 for <PAD> and index 1 for <UNK>. Words are ranked by
    frequency and only the top max_vocab_size words with at least min_freq
    occurrences are included.

    Args:
        tokenized_texts: List of lists of tokens (strings).
        max_vocab_size: Maximum number of words in the vocabulary (excluding special tokens).
        min_freq: Minimum frequency a word must have to be included.

    Returns:
        Dictionary mapping words to integer indices.
    """
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common(max_vocab_size):
        if freq < min_freq:
            break
        vocab[word] = idx
        idx += 1

    return vocab


def print_similar_words(
    embedding_layer: torch.nn.Embedding,
    vocab: Dict[str, int],
    target_word: str,
    k: int = 5,
) -> None:
    """Print the k most similar words to a target word using cosine similarity.

    Looks up the target word's learned embedding and computes cosine similarity
    against every other embedding in the layer, then prints the top-k results
    (excluding the query word itself).

    Args:
        embedding_layer: Trained nn.Embedding layer whose weight matrix provides
            the word vectors.
        vocab: Dictionary mapping words to their integer indices (must match the
            embedding layer's index space).
        target_word: The word to find similar words for.
        k: Number of similar words to return. Defaults to 5.

    Returns:
        None. Prints the top-k most similar words and their cosine similarity scores.
    """
    if target_word not in vocab:
        print(f"'{target_word}' not in vocabulary")
        return

    # Build a reverse mapping from index to word
    itos = {idx: word for word, idx in vocab.items()}

    # Retrieve the full embedding weight matrix (vocab_size, embed_dim)
    weights = embedding_layer.weight.data

    # Look up the target word's vector and add a batch dimension
    target_idx = vocab[target_word]
    query_vec = weights[target_idx].unsqueeze(0)  # (1, embed_dim)

    # Compute cosine similarity between query and all embeddings
    sims = cosine_similarity(query_vec, weights)  # (vocab_size,)

    # Mask out the query word so it doesn't appear in results
    sims[target_idx] = -1.0

    # Get top-k results
    top_scores, top_indices = torch.topk(sims, k=k)

    print(f"Words most similar to '{target_word}':")
    for score, idx in zip(top_scores, top_indices):
        print(f"  {itos.get(idx.item(), '<UNK>'):<15} {score.item():.4f}")


# ============================ EXPERIMENTAL ============================


def mlm_collate_fn(
    batch,
    mask_id,
    vocab_size,
    padding_value,
    max_seq_len,
    mask_prob,
):
    """Collate function for Masked Language Modeling.

    Pads sequences, then randomly masks 15% of non-padding tokens following the
    BERT masking strategy (80% [MASK], 10% random, 10% unchanged).

    Args:
        batch: List of (token_ids_tensor, label_tensor) tuples.
        padding_value: PAD token index.
        max_seq_len: Maximum sequence length (truncate longer sequences).
        mask_prob: Fraction of non-padding tokens to mask.
        mask_id: Token ID for the <MASK> token.
        vocab_size: Total vocabulary size (for random replacement).

    Returns:
        Tuple of (masked_input, mlm_labels) where:
            masked_input: (batch, seq_len) — input with masked tokens
            mlm_labels: (batch, seq_len) — original token IDs at masked positions,
                        -100 elsewhere (ignored by CrossEntropyLoss)
    """
    texts, _ = zip(*batch)  # ignore original task labels — MLM is self-supervised

    # Truncate sequences to max_seq_len
    if max_seq_len is not None:
        texts = tuple(t[:max_seq_len] for t in texts)

    # Pad to uniform length within the batch
    padded = pad_sequence(texts, batch_first=True, padding_value=padding_value)
    # (batch, seq_len)

    # ── Create masks ──
    # Only mask non-padding positions
    non_pad_mask = padded != padding_value  # True where there are real tokens

    # Randomly select mask_prob fraction of non-padding positions
    rand_matrix = torch.rand(padded.shape)  # uniform [0, 1)
    mask_positions = (rand_matrix < mask_prob) & non_pad_mask  # True = will be masked

    # ── Build MLM labels ──
    # -100 at positions we don't predict; original token ID at masked positions
    mlm_labels = torch.full(padded.shape, -100, dtype=torch.long)
    mlm_labels[mask_positions] = padded[
        mask_positions
    ]  # ground truth at masked positions

    # ── Apply the BERT masking strategy to the input ──
    masked_input = padded.clone()

    # Generate a random number for each masked position to decide the replacement type
    replacement_probs = torch.rand(mask_positions.sum())

    # Get indices of masked positions
    mask_indices = mask_positions.nonzero(as_tuple=False)  # (N_masked, 2)

    # 80% of masked positions → replaced with <MASK> token
    mask_replace = replacement_probs < 0.8
    mask_replace_idx = mask_indices[mask_replace]
    masked_input[mask_replace_idx[:, 0], mask_replace_idx[:, 1]] = mask_id

    # 10% of masked positions → replaced with a random token (avoid PAD=0)
    random_replace = (replacement_probs >= 0.8) & (replacement_probs < 0.9)
    random_replace_idx = mask_indices[random_replace]
    random_tokens = torch.randint(2, vocab_size, (random_replace_idx.shape[0],))
    masked_input[random_replace_idx[:, 0], random_replace_idx[:, 1]] = random_tokens

    # 10% of masked positions → left unchanged (but still predicted via mlm_labels)

    return masked_input, mlm_labels
