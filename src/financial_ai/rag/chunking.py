import re
import uuid

from financial_ai.rag.schemas import (
    RAGChunk,
)


ITEM_PATTERN = re.compile(
    r"(?im)^ITEM\s+"
    r"(\d+[A-Z]?(?:\.\d+)?)"
    r"\.?\s*"
    r"(?:[-–—:]\s*)?"
    r"([^\n]{0,100})$"
)

def clean_section_title(
    item_number: str,
    title: str,
) -> str:

    title = title.strip(
        " .:-–—"
    )

    if not title:
        return f"Item {item_number}"

    # Avoid accidentally treating long body text
    # as a heading.
    if len(title.split()) > 15:
        return f"Item {item_number}"

    return (
        f"Item {item_number} - "
        f"{title}"
    )
    

def split_sec_sections(
    text: str,
) -> list[tuple[str, str]]:
    """
    Split SEC filing primarily around ITEM headings.

    Examples:
        ITEM 1. Financial Statements
        ITEM 1A. Risk Factors
        ITEM 2. Management's Discussion...
        ITEM 2.02 Results of Operations...
    """

    matches = list(
        ITEM_PATTERN.finditer(text)
    )

    if not matches:
        return [
            (
                "Full Filing",
                text,
            )
        ]

    sections = []

    for index, match in enumerate(
        matches
    ):

        start = match.start()

        if (
            index + 1
            < len(matches)
        ):
            end = matches[
                index + 1
            ].start()

        else:
            end = len(text)

        item_number = match.group(1).strip()
        title = match.group(2).strip()

        section_name = clean_section_title(
            item_number,
            title,
        )

        if title:
            section_name += (
                f" - {title}"
            )

        section_text = (
            text[start:end]
            .strip()
        )

        if section_text:
            sections.append(
                (
                    section_name,
                    section_text,
                )
            )

    return sections


def split_into_chunks(
    text: str,
    chunk_size: int = 350,
    overlap: int = 50,
)-> list[str]:
    """
    Simple word-boundary chunking inside each SEC section.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        if end == len(words):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def build_rag_chunks(
    text: str,
    ticker: str,
    filing_type: str,
    filed_date: str,
    accession_number: str,
    source_url: str,
) -> list[RAGChunk]:

    sections = split_sec_sections(
        text
    )

    output = []

    global_index = 0

    for (
        section_name,
        section_text,
    ) in sections:

        chunks = split_into_chunks(
            section_text
        )

        for chunk_text in chunks:

            output.append(
                RAGChunk(
                    chunk_id=str(
                        uuid.uuid4()
                    ),
                    ticker=ticker,
                    filing_type=(
                        filing_type
                    ),
                    filed_date=(
                        filed_date
                    ),
                    accession_number=(
                        accession_number
                    ),
                    section=(
                        section_name
                    ),
                    text=chunk_text,
                    source_url=(
                        source_url
                    ),
                    chunk_index=(
                        global_index
                    ),
                )
            )

            global_index += 1

    return output