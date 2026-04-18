import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(data, handle, indent=4)
