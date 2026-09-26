import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

@dataclass
class AppConfig:
    raw_config: Dict[str, Any] = field(default_factory=load_config)

    @property
    def random_seed(self) -> int:
        return self.raw_config.get("system", {}).get("random_seed", 42)

    @property
    def max_ram_gb(self) -> float:
        return self.raw_config.get("system", {}).get("max_ram_gb", 12.0)

    @property
    def train_dir(self) -> str:
        return self.raw_config.get("paths", {}).get("train_dir", "dataset/train")

    @property
    def test_dir(self) -> str:
        return self.raw_config.get("paths", {}).get("test_dir", "dataset/test")

    @property
    def output_dir(self) -> str:
        return self.raw_config.get("paths", {}).get("output_dir", "output")

    @property
    def models_dir(self) -> str:
        return self.raw_config.get("paths", {}).get("models_dir", "models")

    @property
    def reports_dir(self) -> str:
        return self.raw_config.get("paths", {}).get("reports_dir", "reports")

    @property
    def blocking_cfg(self) -> Dict[str, Any]:
        return self.raw_config.get("blocking", {})

    @property
    def training_cfg(self) -> Dict[str, Any]:
        return self.raw_config.get("training_data", {})

    @property
    def xgboost_cfg(self) -> Dict[str, Any]:
        return self.raw_config.get("xgboost", {})

    @property
    def decision_cfg(self) -> Dict[str, Any]:
        return self.raw_config.get("decision", {})

config = AppConfig()
