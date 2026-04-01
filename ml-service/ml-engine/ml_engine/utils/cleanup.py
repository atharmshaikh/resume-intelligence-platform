"""
cleanup.py
==========
Service to manage automated file house-keeping.
Deletes raw PDF/DOCX resumes after X hours of retention.
"""

import time
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class CleanupService:
    """
    Electronic shredder for temporary resume storage.
    Ensures PII (Personally Identifiable Information) isn't 
    stored longer than necessary.
    """

    def __init__(self, directory: str | Path, retention_hours: int = 24) -> None:
        self.directory = Path(directory)
        self.retention_seconds = retention_hours * 3600

    def purge_stale_files(self, extensions: List[str] = [".pdf", ".docx"]) -> int:
        """
        Find and delete files older than the retention threshold.
        Returns the number of files deleted.
        """
        if not self.directory.exists():
            logger.warning("Cleanup directory not found: %s", self.directory)
            return 0

        now = time.time()
        deleted_count = 0

        # Scan directory for resume extensions
        for item in self.directory.iterdir():
            if item.is_file() and item.suffix.lower() in extensions:
                
                # Check modification time
                file_age = now - item.stat().st_mtime
                
                if file_age > self.retention_seconds:
                    try:
                        item.unlink()
                        deleted_count += 1
                        logger.info("Auto-Purged stale resume: %s", item.name)
                    except Exception as exc:
                        logger.error("Failed to delete stale file %s: %s", item.name, exc)

        return deleted_count
