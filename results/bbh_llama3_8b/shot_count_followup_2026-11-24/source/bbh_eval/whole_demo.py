"""Choose complete demonstrations under an exact retained-token ceiling."""
import random


def choose_whole_demos(demo_lengths, target_demo_tokens, seed):
    """Return original-order demo indices with the largest achievable token sum.

    The caller supplies token counts for the same serialized demonstration spans
    used by the KV selector. The fixed instruction prefix is outside the budget.
    Ties use a seeded ordering. For the ten-shot pilot there are at most 1,024
    possible subsets.
    """
    if any(type(length) is not int or length <= 0 for length in demo_lengths):
        raise ValueError("Each demonstration length must be a positive integer.")
    if type(target_demo_tokens) is not int or target_demo_tokens < 0:
        raise ValueError("The token ceiling must be a nonnegative integer.")
    order = list(range(len(demo_lengths)))
    random.Random(seed).shuffle(order)
    by_sum = {0: ()}
    for index in order:
        length = demo_lengths[index]
        for previous, subset in list(by_sum.items()):
            total = previous + length
            if total <= target_demo_tokens and total not in by_sum:
                by_sum[total] = subset + (index,)
    return sorted(by_sum[max(by_sum)])
