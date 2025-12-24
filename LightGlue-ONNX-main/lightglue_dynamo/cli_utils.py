import typer


def check_multiple_of(value: int, k: int) -> None:
    if value % k != 0:
        raise typer.BadParameter(f"Value must be a multiple of {k}.")
