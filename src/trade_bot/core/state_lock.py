"""State locking utilities to prevent race conditions."""

import asyncio
from typing import Any, Optional
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


class StateLock:
    """Async lock for managing state access."""
    
    def __init__(self, name: str):
        """Initialize state lock."""
        self.name = name
        self._lock = asyncio.Lock()
        self._lock_count = 0
        logger.debug(f"State lock '{name}' created")
    
    async def acquire(self) -> None:
        """Acquire the lock."""
        await self._lock.acquire()
        self._lock_count += 1
        logger.debug(f"State lock '{self.name}' acquired (count: {self._lock_count})")
    
    def release(self) -> None:
        """Release the lock."""
        if self._lock.locked():
            self._lock.release()
            self._lock_count -= 1
            logger.debug(f"State lock '{self.name}' released (count: {self._lock_count})")
    
    @asynccontextmanager
    async def locked(self):
        """Context manager for locked access."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()
    
    def locked_count(self) -> int:
        """Get current lock count."""
        return self._lock_count


class StateManager:
    """Manages multiple state locks."""
    
    def __init__(self):
        """Initialize state manager."""
        self._locks: dict[str, StateLock] = {}
        logger.info("State manager initialized")
    
    def get_lock(self, name: str) -> StateLock:
        """Get or create a state lock."""
        if name not in self._locks:
            self._locks[name] = StateLock(name)
        return self._locks[name]
    
    @asynccontextmanager
    async def lock_state(self, lock_name: str):
        """Context manager for locking specific state."""
        lock = self.get_lock(lock_name)
        async with lock.locked():
            yield
    
    def get_lock_status(self) -> dict[str, int]:
        """Get status of all locks."""
        return {name: lock.locked_count() for name, lock in self._locks.items()}


# Global state manager instance
state_manager = StateManager()
