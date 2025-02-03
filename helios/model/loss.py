from .model import TokensAndMasks


def loss(target: TokensAndMasks, preds: TokensAndMasks) -> float:
    raise NotImplementedError
