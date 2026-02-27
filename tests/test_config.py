"""Unit tests for Config dataclass and YAML I/O."""

import tempfile
from pathlib import Path

import pytest

from musae_inv.config import Config


class TestConfigDefaults:
    """Test that default configuration values are correct."""

    def test_default_seed(self):
        cfg = Config()
        assert cfg.seed == 42

    def test_default_model_id(self):
        cfg = Config()
        assert cfg.model_id == "google/gemma-2-2b"

    def test_default_target_layers(self):
        cfg = Config()
        assert cfg.target_layers == [6, 12, 18, 25]

    def test_default_icfs_top_k(self):
        cfg = Config()
        assert cfg.icfs_top_k == 128

    def test_default_musae_C(self):
        cfg = Config()
        assert cfg.musae_C == 0.3

    def test_default_sae_width(self):
        cfg = Config()
        assert cfg.sae_width == 16384


class TestConfigDerived:
    """Test derived properties."""

    def test_icfs_dim(self):
        cfg = Config(icfs_top_k=128, target_layers=[6, 12, 18, 25])
        assert cfg.icfs_dim == 512  # 4 layers × 128

    def test_icfs_dim_custom(self):
        cfg = Config(icfs_top_k=64, target_layers=[6, 25])
        assert cfg.icfs_dim == 128  # 2 layers × 64


class TestConfigIO:
    """Test YAML save/load round-trip."""

    def test_roundtrip(self):
        cfg = Config(seed=123, icfs_top_k=256, musae_C=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            cfg.save(path)
            loaded = Config.load(path)
            assert loaded.seed == 123
            assert loaded.icfs_top_k == 256
            assert loaded.musae_C == 1.0

    def test_save_creates_parent_dirs(self):
        cfg = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "config.yaml"
            cfg.save(path)
            assert path.exists()


class TestConfigMakeDirs:
    """Test directory creation."""

    def test_make_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(output_dir=str(Path(tmpdir) / "test_outputs"))
            cfg.make_dirs()
            assert cfg.features_dir.exists()
            assert cfg.results_dir.exists()
            assert cfg.plots_dir.exists()
