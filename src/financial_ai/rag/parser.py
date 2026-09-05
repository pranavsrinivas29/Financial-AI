import re
from pathlib import Path

from bs4 import BeautifulSoup


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()


def parse_sec_html(
    file_path: Path,
) -> str:
    """
    Parse downloaded SEC HTML into clean text.

    We intentionally avoid OCR because SEC filings
    are normally available as HTML.
    """

    html = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    # Remove irrelevant document elements.
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):
        tag.decompose()

    text = soup.get_text(
        separator="\n"
    )

    lines = []

    for line in text.splitlines():

        line = clean_text(line)

        if not line:
            continue

        lines.append(line)

    return "\n".join(lines)