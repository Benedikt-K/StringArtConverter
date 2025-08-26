from typing import Tuple, List

# def Segment
Segment = Tuple[int, int]

def save_path_txt(path: List[Segment], out_txt: str) -> None:
    """
    Save path to file
    """
    with open(out_txt, "w", encoding="utf-8") as f:
        for a, b in path:
            f.write(f"{a} {b}\n")