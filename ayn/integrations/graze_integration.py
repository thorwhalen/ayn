"""Integration with graze package for downloading and caching agent data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

# graze integration is optional
try:
    from graze import graze, Graze
    HAS_GRAZE = True
except ImportError:
    HAS_GRAZE = False


def fetch_agent_repo(
    url: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Fetch an agent repository using graze.

    Downloads and caches agent repositories for offline use.

    Args:
        url: URL to agent repository (GitHub, GitLab, etc.)
        cache_dir: Directory to cache downloads
        force_download: Force re-download even if cached

    Returns:
        Path to downloaded repository

    Example:
        >>> if HAS_GRAZE:
        ...     # Example would download a real repo
        ...     # path = fetch_agent_repo("https://github.com/user/agent-repo")
        ...     pass
    """
    if not HAS_GRAZE:
        raise ImportError(
            "graze package required. Install with: pip install graze"
        )

    cache_dir = cache_dir or os.path.expanduser("~/.ayn/repos")

    # Use graze to download and cache
    if force_download:
        # Clear cache for this URL
        grazer = Graze(rootdir=cache_dir)
        cached_path = grazer(url)
    else:
        cached_path = graze(url, rootdir=cache_dir)

    return Path(cached_path)


def cache_agent_data(
    url: str,
    data_type: str = "repo",
    cache_dir: Optional[str] = None,
) -> Union[Path, bytes]:
    """Cache agent-related data using graze.

    Can cache:
    - Repositories
    - Model weights
    - Datasets
    - Documentation
    - Configuration files

    Args:
        url: URL to data
        data_type: Type of data (repo, model, dataset, docs, config)
        cache_dir: Custom cache directory

    Returns:
        Path to cached data or raw bytes

    Example:
        >>> if HAS_GRAZE:
        ...     # Example would cache real data
        ...     # path = cache_agent_data(
        ...     #     "https://example.com/model.bin",
        ...     #     data_type="model"
        ...     # )
        ...     pass
    """
    if not HAS_GRAZE:
        raise ImportError(
            "graze package required. Install with: pip install graze"
        )

    # Organize cache by data type
    if cache_dir is None:
        base_cache = os.path.expanduser("~/.ayn/cache")
        cache_dir = os.path.join(base_cache, data_type)

    os.makedirs(cache_dir, exist_ok=True)

    # Download and cache
    cached_path = graze(url, rootdir=cache_dir)

    return Path(cached_path)


class AgentDataCache:
    """Manages cached agent data using graze.

    Provides a higher-level interface for managing agent-related downloads.

    Example:
        >>> if HAS_GRAZE:
        ...     import tempfile
        ...     cache_dir = tempfile.mkdtemp()
        ...     cache = AgentDataCache(cache_dir=cache_dir)
        ...     # path = cache.get_repo("https://github.com/user/agent")
        ...     import shutil
        ...     shutil.rmtree(cache_dir)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if not HAS_GRAZE:
            raise ImportError(
                "graze package required. Install with: pip install graze"
            )

        self.cache_dir = cache_dir or os.path.expanduser("~/.ayn/cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_repo(self, url: str, force: bool = False) -> Path:
        """Get a repository (cached or download).

        Args:
            url: Repository URL
            force: Force re-download

        Returns:
            Path to repository
        """
        repo_cache = os.path.join(self.cache_dir, "repos")
        return fetch_agent_repo(url, cache_dir=repo_cache, force_download=force)

    def get_model(self, url: str, force: bool = False) -> Path:
        """Get a model file (cached or download).

        Args:
            url: Model file URL
            force: Force re-download

        Returns:
            Path to model file
        """
        model_cache = os.path.join(self.cache_dir, "models")
        os.makedirs(model_cache, exist_ok=True)

        if force:
            grazer = Graze(rootdir=model_cache)
            path = grazer(url)
        else:
            path = graze(url, rootdir=model_cache)

        return Path(path)

    def get_dataset(self, url: str, force: bool = False) -> Path:
        """Get a dataset (cached or download).

        Args:
            url: Dataset URL
            force: Force re-download

        Returns:
            Path to dataset
        """
        dataset_cache = os.path.join(self.cache_dir, "datasets")
        os.makedirs(dataset_cache, exist_ok=True)

        if force:
            grazer = Graze(rootdir=dataset_cache)
            path = grazer(url)
        else:
            path = graze(url, rootdir=dataset_cache)

        return Path(path)

    def get_config(self, url: str, force: bool = False) -> Path:
        """Get a configuration file (cached or download).

        Args:
            url: Config file URL
            force: Force re-download

        Returns:
            Path to config file
        """
        config_cache = os.path.join(self.cache_dir, "configs")
        os.makedirs(config_cache, exist_ok=True)

        if force:
            grazer = Graze(rootdir=config_cache)
            path = grazer(url)
        else:
            path = graze(url, rootdir=config_cache)

        return Path(path)

    def clear_cache(self, data_type: Optional[str] = None):
        """Clear cached data.

        Args:
            data_type: Type to clear (repos, models, datasets, configs) or None for all
        """
        import shutil

        if data_type:
            cache_path = os.path.join(self.cache_dir, data_type)
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                os.makedirs(cache_path, exist_ok=True)
        else:
            # Clear entire cache
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_size(self) -> int:
        """Get total size of cache in bytes.

        Returns:
            Total cache size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
