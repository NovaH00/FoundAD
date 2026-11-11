from torch import Tensor


def assert_shape(t: Tensor, expected_shape: str) -> None:
    if (
        len(expected_shape) < 2
        or expected_shape[0] != "("
        or expected_shape[-1] != ")"
    ):
        raise ValueError(
            f"Incorrect expected_shape format: `{expected_shape}`.\n"
            "Example of correct format: (_, _, 3, 224, 224)"
        )

    values = [v.strip() for v in expected_shape[1:-1].split(",")]

    # Allow a trailing comma, e.g. (2, 3,)
    if values and values[-1] == "":
        values.pop()

    if any(v != "_" and not v.isdecimal() for v in values):
        raise ValueError(
            f"Incorrect expected_shape format: `{expected_shape}`."
        )

    expected_dims = [
        None if v == "_" else int(v)
        for v in values
    ]

    actual_shape = tuple(t.shape)

    if t.ndim != len(expected_dims):
        raise ValueError(
            f"Tensor shape mismatch: expected {expected_shape}, "
            f"got {actual_shape}\n"
            f"Reason: Rank mismatch\n"
            f"Detail: Expected rank {len(expected_dims)}, got {t.ndim}"
        )

    for i, (expected, actual) in enumerate(zip(expected_dims, actual_shape)):
        if expected is not None and expected != actual:
            raise ValueError(
                f"Tensor shape mismatch: expected {expected_shape}, "
                f"got {actual_shape}\n"
                f"Reason: Dimension size mismatch\n"
                f"Detail: At axis {i}, expected size {expected}, got {actual}"
            )
