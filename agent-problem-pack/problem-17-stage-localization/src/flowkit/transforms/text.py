"""Text helpers for token-level pipelines (not registered as stages)."""


def tokenize(text):
    return [part for part in text.lower().split() if part]


def ngrams(tokens, n=2):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
