#!/usr/bin/env python3
"""
Enterprise Workspace Optimizer & Duplicate Detection Utility
Adheres to SOLID, KISS, and PEP 8 standards with concurrent hashing,
robust error handling, and structured JSON reporting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WorkspaceOptimizer")


@dataclass(frozen=True)
class FileRecord:
    """Immutable data record representing a scanned file."""
    path: Path
    size_bytes: int


class DiskScanner:
    """Responsible for traversing directory trees and filtering files by size."""

    def __init__(self, root_dir: Path, min_size_mb: int = 20) -> None:
        self.root_dir = root_dir
        self.min_size_bytes = min_size_mb * 1024 * 1024

    def scan(self) -> Generator[FileRecord, None, None]:
        """Recursively yield files greater than or equal to the minimum size threshold."""
        if not self.root_dir.exists():
            logger.warning("Root directory %s does not exist.", self.root_dir)
            return

        try:
            for entry in self.root_dir.rglob("*"):
                if entry.is_file() and not entry.is_symlink():
                    try:
                        stat = entry.stat()
                        if stat.st_size >= self.min_size_bytes:
                            yield FileRecord(path=entry, size_bytes=stat.st_size)
                    except (PermissionError, FileNotFoundError) as e:
                        logger.debug("Skipping inaccessible file %s: %s", entry, e)
        except Exception as e:
            logger.error("Error traversing directory %s: %s", self.root_dir, e)


class ParallelDuplicateDetector:
    """Identifies byte-identical duplicate files using size grouping and concurrent SHA-256 hashing."""

    def __init__(self, records: List[FileRecord], max_workers: int = 4) -> None:
        self.records = records
        self.max_workers = max_workers

    @staticmethod
    def _compute_sha256(path: Path, chunk_size: int = 65536) -> Optional[Tuple[str, Path]]:
        """Compute SHA-256 checksum of a file efficiently using buffered chunk reads."""
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest(), path
        except (PermissionError, FileNotFoundError) as e:
            logger.debug("Permission or access error hashing %s: %s", path, e)
            return None
        except Exception as e:
            logger.error("Unexpected error hashing %s: %s", path, e)
            return None

    def find_duplicates(self) -> Dict[str, List[Path]]:
        """Group records by size, then concurrently compute hashes for size collisions."""
        size_map: Dict[int, List[Path]] = {}
        for record in self.records:
            size_map.setdefault(record.size_bytes, []).append(record.path)

        # Filter candidate paths sharing identical file sizes (> 1 file)
        candidate_paths: List[Path] = [
            path for paths in size_map.values() if len(paths) > 1 for path in paths
        ]

        if not candidate_paths:
            return {}

        hash_map: Dict[str, List[Path]] = {}
        logger.info("Computing SHA-256 hashes for %d candidate files concurrently...", len(candidate_paths))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._compute_sha256, path): path for path in candidate_paths}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    file_hash, path = result
                    hash_map.setdefault(file_hash, []).append(path)

        # Retain only hashes with multiple distinct file paths
        return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


class ReportManager:
    """Responsible for formatting and atomically persisting analysis reports."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def save_report(self, duplicates: Dict[str, List[Path]], total_scanned: int) -> None:
        """Atomically serialize duplicate findings to JSON."""
        report_data = {
            "metadata": {
                "total_files_scanned": total_scanned,
                "duplicate_groups_count": len(duplicates)
            },
            "duplicates": {
                file_hash: [str(path) for path in paths]
                for file_hash, paths in duplicates.items()
            }
        }
        
        try:
            temp_path = self.output_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
            os.replace(temp_path, self.output_path)
            logger.info("Report successfully written to %s", self.output_path)
        except Exception as e:
            logger.error("Failed to write report to %s: %s", self.output_path, e)


class WorkspaceOptimizerService:
    """Orchestrates the end-to-end workspace scanning and optimization pipeline."""

    def __init__(self, root_dir: Path, output_path: Path, min_size_mb: int = 20) -> None:
        self.root_dir = root_dir
        self.output_path = output_path
        self.min_size_mb = min_size_mb

    def run(self) -> Dict[str, Any]:
        """Execute the scan, duplicate detection, and reporting workflow."""
        logger.info("Starting workspace scan on %s (Threshold: >= %d MiB)", self.root_dir, self.min_size_mb)
        
        scanner = DiskScanner(self.root_dir, self.min_size_mb)
        records = list(scanner.scan())
        logger.info("Discovered %d qualifying files.", len(records))

        detector = ParallelDuplicateDetector(records)
        duplicates = detector.find_duplicates()
        logger.info("Identified %d duplicate file groups.", len(duplicates))

        reporter = ReportManager(self.output_path)
        reporter.save_report(duplicates, len(records))

        summary = {
            "status": "success",
            "root_directory": str(self.root_dir),
            "total_files_scanned": len(records),
            "duplicate_groups_found": len(duplicates),
            "report_path": str(self.output_path)
        }
        print(json.dumps(summary, indent=2))
        return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the optimizer service."""
    parser = argparse.ArgumentParser(description="Enterprise Workspace Optimizer & Duplicate Detector")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home(),
        help="Root directory to scan (default: user home directory)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "workspace_duplication_report.json",
        help="Path to output JSON report"
    )
    parser.add_argument(
        "--min-size-mb",
        type=int,
        default=20,
        help="Minimum file size threshold in MiB (default: 20)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    service = WorkspaceOptimizerService(
        root_dir=args.root,
        output_path=args.output,
        min_size_mb=args.min_size_mb
    )
    service.run()
    
#!/usr/bin/env python3
"""
Workspace Optimizer & Maintenance Utility
Adheres to SOLID, KISS, and PEP 8 standards with rigorous error handling,
type hinting, and clean modular architecture for Termux and Linux environments.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Tuple

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WorkspaceOptimizer")


@dataclass(frozen=True)
class FileRecord:
    """Immutable data record representing a scanned file."""
    path: Path
    size_bytes: int


class DiskScanner:
    """Responsible for scanning directory trees and identifying files exceeding size thresholds."""

    def __init__(self, root_dir: Path, min_size_mb: int = 20) -> None:
        self.root_dir = root_dir
        self.min_size_bytes = min_size_mb * 1024 * 1024

    def scan(self) -> Generator[FileRecord, None, None]:
        """Recursively yield files greater than or equal to the minimum size threshold."""
        try:
            for entry in self.root_dir.rglob("*"):
                if entry.is_file() and not entry.is_symlink():
                    try:
                        stat = entry.stat()
                        if stat.st_size >= self.min_size_bytes:
                            yield FileRecord(path=entry, size_bytes=stat.st_size)
                    except (PermissionError, FileNotFoundError) as e:
                        logger.debug("Skipping inaccessible file %s: %s", entry, e)
        except Exception as e:
            logger.error("Error during directory traversal of %s: %s", self.root_dir, e)


class DuplicateDetector:
    """Identifies byte-identical duplicate files using size pre-filtering and SHA-256 hashing."""

    def __init__(self, records: List[FileRecord]) -> None:
        self.records = records

    @staticmethod
    def _compute_sha256(path: Path, chunk_size: int = 65536) -> str:
        """Compute the SHA-256 checksum of a file efficiently using buffered chunk reads."""
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error("Failed to compute hash for %s: %s", path, e)
            return ""

    def find_duplicates(self) -> Dict[str, List[Path]]:
        """Group records by size first, then verify byte-identical content via SHA-256."""
        size_map: Dict[int, List[Path]] = {}
        for record in self.records:
            size_map.setdefault(record.size_bytes, []).append(record.path)

        # Filter candidate groups where more than one file shares the exact same size
        candidate_groups = {size: paths for size, paths in size_map.items() if len(paths) > 1}
        hash_map: Dict[str, List[Path]] = {}

        for size, paths in candidate_groups.items():
            for path in paths:
                file_hash = self._compute_sha256(path)
                if file_hash:
                    hash_map.setdefault(file_hash, []).append(path)

        # Retain only hashes that map to multiple distinct file paths
        return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


class ReportGenerator:
    """Responsible for formatting and persisting analysis results to disk."""

    def __init__(self, duplicates: Dict[str, List[Path]]) -> None:
        self.duplicates = duplicates

    def save_json(self, output_path: Path) -> None:
        """Serialize duplicate findings into structured JSON format."""
        report_data = {
            file_hash: [str(path) for path in paths]
            for file_hash, paths in self.duplicates.items()
        }
        try:
            output_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
            logger.info("Duplicate analysis report successfully saved to %s", output_path)
        except Exception as e:
            logger.error("Failed to write report to %s: %s", output_path, e)


def main() -> None:
    """Main execution entry point for the workspace optimization utility."""
    home_dir = Path.home()
    logger.info("Initializing workspace storage audit on %s", home_dir)

    scanner = DiskScanner(home_dir, min_size_mb=20)
    scanned_records = list(scanner.scan())
    logger.info("Discovered %d files >= 20 MiB", len(scanned_records))

    detector = DuplicateDetector(scanned_records)
    duplicates = detector.find_duplicates()
    logger.info("Identified %d unique duplicate file groups", len(duplicates))

    report_path = home_dir / "workspace_duplication_report.json"
    reporter = ReportGenerator(duplicates)
    reporter.save_json(report_path)

    summary = {
        "status": "success",
        "total_files_scanned": len(scanned_records),
        "duplicate_groups_found": len(duplicates),
        "report_location": str(report_path)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    
