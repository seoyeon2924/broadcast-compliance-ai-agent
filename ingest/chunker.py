"""
Text chunker — splits extracted text into fixed-size overlapping chunks.
Stage 2 implementation.
"""

from config import settings


class Chunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk(self, text: str) -> list[str]:
        """
        Split *text* into chunks of ``chunk_size`` characters
        with ``chunk_overlap`` overlap.

        Returns:
            list of chunk strings
        """
        # TODO: Stage 2
        raise NotImplementedError("Chunking will be implemented in Stage 2")
