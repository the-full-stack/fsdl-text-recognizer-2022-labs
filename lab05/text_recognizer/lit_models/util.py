from typing import Union

import torch


def first_appearance(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    """Return indices of first appearance of element in x, collapsing along dim.

    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9

    Parameters
    ----------
    x
        One or two-dimensional Tensor to search for element.
    element
        Item to search for inside x.
    dim
        Dimension of Tensor to collapse over.

    Returns
    -------
    torch.Tensor
        Indices where element occurs in x. If element is not found,
        return length of x along dim. One dimension smaller than x.

    Raises
    ------
    ValueError
        if x is not a 1 or 2 dimensional Tensor

    Examples
    --------
    >>> first_appearance(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1], [3, 1, 1]]), 3)
    tensor([2, 1, 3, 0])
    >>> first_appearance(torch.tensor([1, 2, 3]), 1, dim=0)
    tensor(0)
    """
    if x.dim() > 2 or x.dim() == 0:
        raise ValueError(f"only 1 or 2 dimensional Tensors allowed, got Tensor with dim {x.dim()}")
    matches = x == element
    first_appearance_mask = (matches.cumsum(dim) == 1) & matches
    does_match, match_index = first_appearance_mask.max(dim)
    first_inds = torch.where(does_match, match_index, x.shape[dim])
    return first_inds


def replace_after(x: torch.Tensor, element: Union[int, float], replace: Union[int, float]) -> torch.Tensor:
    """Replace all values in each row of 2d Tensor x after the first appearance of element with replace.

    Parameters
    ----------
    x
        Two-dimensional Tensor (shape denoted (B, S)) to replace values in.
    element
        Item to search for inside x.
    replace
        Item that replaces entries that appear after element.

    Returns
    -------
    outs
        New Tensor of same shape as x with values after element replaced.

    Examples
    --------
    >>> replace_after(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1], [3, 1, 1]]), 3, 4)
    tensor([[1, 2, 3],
            [2, 3, 4],
            [1, 1, 1],
            [3, 4, 4]])
    """
    first_appearances = first_appearance(x, element, dim=1)  # (B,)
    indices = torch.arange(0, x.shape[-1]).type_as(x)  # (S,)
    outs = torch.where(
        indices[None, :] <= first_appearances[:, None],  # if index is before first appearance
        x,  # return the value from x
        replace,  # otherwise, return the replacement value
    )
    return outs  # (B, S)
