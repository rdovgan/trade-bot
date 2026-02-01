"""Unit tests for state locking and concurrency control (Phase 3)."""

import pytest
import asyncio
from trade_bot.core.state_lock import StateLock, StateManager


class TestStateLock:
    """Tests for StateLock class."""

    @pytest.mark.asyncio
    async def test_lock_acquire_release(self):
        """Test basic lock acquire and release."""
        lock = StateLock("test_lock")

        await lock.acquire()
        assert lock._lock.locked()

        lock.release()
        assert not lock._lock.locked()

    @pytest.mark.asyncio
    async def test_lock_context_manager(self):
        """Test lock as context manager."""
        lock = StateLock("test_lock")

        async with lock.locked():
            assert lock._lock.locked()

        assert not lock._lock.locked()

    @pytest.mark.asyncio
    async def test_lock_count(self):
        """Test lock count tracking."""
        lock = StateLock("test_lock")

        assert lock.locked_count() == 0

        await lock.acquire()
        assert lock.locked_count() == 1

        lock.release()
        assert lock.locked_count() == 0

    @pytest.mark.asyncio
    async def test_concurrent_access_blocked(self):
        """Test that concurrent access is properly blocked."""
        lock = StateLock("test_lock")
        results = []

        async def worker(worker_id, delay):
            async with lock.locked():
                results.append(f"start_{worker_id}")
                await asyncio.sleep(delay)
                results.append(f"end_{worker_id}")

        # Start two workers concurrently
        await asyncio.gather(
            worker(1, 0.1),
            worker(2, 0.1)
        )

        # Results should show that workers executed serially, not concurrently
        # One worker should completely finish before the other starts
        assert results in [
            ["start_1", "end_1", "start_2", "end_2"],
            ["start_2", "end_2", "start_1", "end_1"]
        ]


class TestStateManager:
    """Tests for StateManager class."""

    def test_get_lock_creates_new(self):
        """Test that get_lock creates new lock if doesn't exist."""
        manager = StateManager()

        lock = manager.get_lock("test_lock")
        assert isinstance(lock, StateLock)
        assert lock.name == "test_lock"

    def test_get_lock_returns_same(self):
        """Test that get_lock returns same lock for same name."""
        manager = StateManager()

        lock1 = manager.get_lock("test_lock")
        lock2 = manager.get_lock("test_lock")

        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_lock_state_context_manager(self):
        """Test lock_state context manager."""
        manager = StateManager()

        async with manager.lock_state("test_lock"):
            # Lock should be held
            lock = manager.get_lock("test_lock")
            assert lock._lock.locked()

        # Lock should be released after context
        assert not lock._lock.locked()

    @pytest.mark.asyncio
    async def test_multiple_locks_independent(self):
        """Test that different locks are independent."""
        manager = StateManager()

        lock1_held = False
        lock2_held = False

        async with manager.lock_state("lock1"):
            lock1_held = True
            # Should be able to acquire lock2 while holding lock1
            async with manager.lock_state("lock2"):
                lock2_held = True

        assert lock1_held and lock2_held

    def test_get_lock_status(self):
        """Test getting status of all locks."""
        manager = StateManager()

        manager.get_lock("lock1")
        manager.get_lock("lock2")

        status = manager.get_lock_status()
        assert "lock1" in status
        assert "lock2" in status
        assert status["lock1"] == 0
        assert status["lock2"] == 0


class TestConcurrencyControl:
    """Integration tests for concurrency control."""

    @pytest.mark.asyncio
    async def test_account_state_update_serialization(self):
        """Test that account state updates are serialized."""
        from trade_bot.core.state_lock import state_manager

        results = []
        update_order = []

        async def update_account(worker_id):
            async with state_manager.lock_state("account"):
                update_order.append(f"start_{worker_id}")
                await asyncio.sleep(0.05)  # Simulate work
                results.append(worker_id)
                update_order.append(f"end_{worker_id}")

        # Run 3 concurrent updates
        await asyncio.gather(
            update_account(1),
            update_account(2),
            update_account(3)
        )

        # All updates should complete
        assert len(results) == 3
        assert set(results) == {1, 2, 3}

        # Updates should be serialized - each should completely finish before next starts
        assert len(update_order) == 6
        # Check that for each worker, start and end are adjacent
        for i in range(0, 6, 2):
            worker_id = update_order[i].split('_')[1]
            assert update_order[i] == f"start_{worker_id}"
            assert update_order[i+1] == f"end_{worker_id}"

    @pytest.mark.asyncio
    async def test_order_tracking_serialization(self):
        """Test that order tracking updates are serialized."""
        from trade_bot.core.state_lock import state_manager

        order_dict = {}
        results = []

        async def add_order(order_id):
            async with state_manager.lock_state("orders"):
                # Simulate reading, modifying, writing
                current = dict(order_dict)
                await asyncio.sleep(0.01)
                current[order_id] = f"order_{order_id}"
                order_dict.clear()
                order_dict.update(current)
                results.append(order_id)

        # Add 5 orders concurrently
        await asyncio.gather(*[add_order(i) for i in range(5)])

        # All orders should be added
        assert len(order_dict) == 5
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_position_update_serialization(self):
        """Test that position updates are serialized."""
        from trade_bot.core.state_lock import state_manager

        position_state = {"size": 0}
        results = []

        async def update_position(delta):
            async with state_manager.lock_state("positions"):
                current = position_state["size"]
                await asyncio.sleep(0.01)
                position_state["size"] = current + delta
                results.append(delta)

        # Run concurrent updates
        await asyncio.gather(
            update_position(1),
            update_position(2),
            update_position(3)
        )

        # Final size should be sum of all deltas (no race condition)
        assert position_state["size"] == 6
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_risk_validation_serialization(self):
        """Test that risk validation is serialized."""
        from trade_bot.core.state_lock import state_manager

        approved_count = 0
        decisions = []

        async def validate_trade(trade_id):
            nonlocal approved_count
            async with state_manager.lock_state("risk_validation"):
                # Simulate checking exposure and approving
                current_exposure = approved_count
                await asyncio.sleep(0.01)
                if current_exposure < 3:  # Max 3 positions
                    approved_count += 1
                    decisions.append((trade_id, "approved"))
                else:
                    decisions.append((trade_id, "rejected"))

        # Try to approve 5 trades concurrently
        await asyncio.gather(*[validate_trade(i) for i in range(5)])

        # Only first 3 should be approved
        approved = [d for d in decisions if d[1] == "approved"]
        rejected = [d for d in decisions if d[1] == "rejected"]

        assert len(approved) == 3
        assert len(rejected) == 2
        assert approved_count == 3
