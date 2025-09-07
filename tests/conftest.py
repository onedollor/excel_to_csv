"""Pytest configuration and shared fixtures for Excel-to-CSV converter tests."""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Generator

import pandas as pd
import yaml


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict() -> dict:
    """Sample configuration dictionary for testing."""
    return {
        "monitoring": {
            "folders": ["./input", "./data"],
            "file_patterns": ["*.xlsx", "*.xls"],
            "process_existing": True,
            "max_file_size": 100,
        },
        "confidence": {
            "threshold": 0.9,
            "weights": {
                "data_density": 0.4,
                "header_quality": 0.3,
                "consistency": 0.3,
            },
        },
        "output": {
            "folder": "./output",
            "naming_pattern": "{filename}_{worksheet}.csv",
            "include_timestamp": True,
            "encoding": "utf-8",
        },
        "processing": {
            "max_concurrent": 5,
        },
        "logging": {
            "level": "INFO",
            "file": {
                "enabled": True,
                "path": "./logs/test.log",
            },
        },
    }


@pytest.fixture
def sample_config_file(temp_dir: Path, sample_config_dict: dict) -> Path:
    """Create a sample configuration file for testing."""
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config_dict, f)
    return config_file


@pytest.fixture
def sample_excel_data() -> pd.DataFrame:
    """Create sample Excel data for testing."""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 22],
        'Department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Marketing'],
        'Salary': [75000, 65000, 55000, 70000, 60000]
    })


@pytest.fixture
def sample_sparse_data() -> pd.DataFrame:
    """Create sparse Excel data for testing confidence analysis."""
    data = pd.DataFrame(index=range(10), columns=range(5))
    # Only fill a few cells to create sparse data
    data.iloc[0, 0] = "Header1"
    data.iloc[0, 1] = "Header2"
    data.iloc[2, 0] = "Value1"
    data.iloc[4, 1] = "Value2"
    return data


@pytest.fixture
def sample_excel_file(temp_dir: Path, sample_excel_data: pd.DataFrame) -> Path:
    """Create a sample Excel file for testing."""
    excel_file = temp_dir / "test_data.xlsx"
    sample_excel_data.to_excel(excel_file, index=False)
    return excel_file


@pytest.fixture
def invalid_excel_file(temp_dir: Path) -> Path:
    """Create an invalid Excel file for testing."""
    invalid_file = temp_dir / "invalid.xlsx"
    invalid_file.write_text("This is not an Excel file")
    return invalid_file


@pytest.fixture
def env_override():
    """Context manager for environment variable testing."""
    class EnvOverride:
        def __init__(self):
            self.original_env = {}
        
        def set(self, key: str, value: str):
            if key in os.environ:
                self.original_env[key] = os.environ[key]
            os.environ[key] = value
        
        def clear(self):
            for key in list(os.environ.keys()):
                if key.startswith('EXCEL_TO_CSV_'):
                    if key in self.original_env:
                        os.environ[key] = self.original_env[key]
                    else:
                        del os.environ[key]
    
    override = EnvOverride()
    yield override
    override.clear()