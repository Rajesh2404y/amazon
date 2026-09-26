import os
import json
import logging
from src.config import config

logger = logging.getLogger("entity_resolution")

def run_profiling(dataset_dir: str = "dataset", reports_dir: str = "reports"):
    """Runs dataset profiling and ensures reports are generated."""
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "dataset_profile.json")
    md_path = os.path.join(reports_dir, "dataset_profile.md")
    
    if os.path.exists(json_path) and os.path.exists(md_path):
        logger.info(f"Dataset profiling reports already exist at:\n  {json_path}\n  {md_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Train Source 1 Entities: {data['train_sources']['train_source1.tsv']['total_rows']:,}")
        logger.info(f"Test Source 1 Entities:  {data['test_sources']['test_source1.tsv']['total_rows']:,}")
        logger.info(f"Total Ground Truth Matches: {data['ground_truth']['total_matches_across_all']:,}")
        return json_path, md_path
    else:
        logger.warning("Reports not found. Please run the profile generation script.")
        return None, None

if __name__ == "__main__":
    run_profiling()
