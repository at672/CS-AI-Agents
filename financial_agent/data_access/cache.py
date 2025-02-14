from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
from pathlib import Path
import json
import hashlib
from datetime import datetime, timedelta
import logging
from functools import wraps
import diskcache as dc

logger = logging.getLogger(__name__)

class CacheKey:
    """Handles creation and management of cache keys"""
    
    @staticmethod
    def create_key(company: str, statement_type: str, period: str, **kwargs) -> str:
        """Create a deterministic cache key from parameters"""
        key_dict = {
            'company': company,
            'statement_type': statement_type,
            'period': period,
            **kwargs
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

class CacheEntry:
    """Represents a cached data entry with metadata"""
    
    def __init__(self, data: Any, expires_at: Optional[datetime] = None):
        self.data = data
        self.created_at = datetime.now()
        self.expires_at = expires_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def increment_access(self):
        """Track cache hit"""
        self.access_count += 1

class MemoryCache:
    """In-memory cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[CacheEntry]:
        entry = self._cache.get(key)
        if entry and entry.is_expired():
            self.delete(key)
            return None
        if entry:
            entry.increment_access()
        return entry
    
    def set(self, key: str, entry: CacheEntry) -> None:
        if len(self._cache) >= self._max_size:
            self._evict_entries()
        self._cache[key] = entry
    
    def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        self._cache.clear()
    
    def _evict_entries(self):
        """Evict least accessed entries when cache is full"""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].access_count, x[1].created_at)
        )
        entries_to_remove = max(1, len(self._cache) // 10)
        for key, _ in sorted_entries[:entries_to_remove]:
            self.delete(key)

class DiskCache:
    """Disk-based cache implementation"""
    
    def __init__(self, cache_dir: Union[str, Path]):
        self._cache = dc.Cache(str(cache_dir))
    
    def get(self, key: str) -> Optional[CacheEntry]:
        try:
            entry = self._cache.get(key)
            if entry and entry.is_expired():
                self.delete(key)
                return None
            if entry:
                entry.increment_access()
                self._cache.set(key, entry)
            return entry
        except Exception as e:
            logger.error(f"Error retrieving from disk cache: {e}")
            return None
    
    def set(self, key: str, entry: CacheEntry) -> None:
        try:
            self._cache.set(key, entry)
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
    
    def delete(self, key: str) -> None:
        try:
            self._cache.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from disk cache: {e}")
    
    def clear(self) -> None:
        try:
            self._cache.clear()
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")

class TieredCache:
    """Implements a tiered caching strategy"""
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        disk_cache_dir: Optional[Path] = None
    ):
        self.memory_cache = MemoryCache(max_size=memory_cache_size)
        self.disk_cache = DiskCache(disk_cache_dir) if disk_cache_dir else None
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache, trying memory first then disk"""
        entry = self.memory_cache.get(key)
        if entry:
            logger.debug(f"Memory cache hit for key: {key}")
            return entry.data
        
        if self.disk_cache:
            entry = self.disk_cache.get(key)
            if entry:
                logger.debug(f"Disk cache hit for key: {key}")
                self.memory_cache.set(key, entry)
                return entry.data
        
        return None
    
    def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[timedelta] = None
    ) -> None:
        """Store data in cache with optional TTL"""
        expires_at = datetime.now() + ttl if ttl else None
        entry = CacheEntry(data, expires_at)
        
        self.memory_cache.set(key, entry)
        if self.disk_cache:
            self.disk_cache.set(key, entry)
    
    def clear(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()

def cached_financial_data(ttl: Optional[timedelta] = None):
    """Decorator for caching financial data method calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, company: str, statement_type: str, period: str, *args, **kwargs):
            cache_key = CacheKey.create_key(company, statement_type, period, **kwargs)
            
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
            
            data = func(self, company, statement_type, period, *args, **kwargs)
            self.cache.set(cache_key, data, ttl)
            
            return data
        return wrapper
    return decorator
