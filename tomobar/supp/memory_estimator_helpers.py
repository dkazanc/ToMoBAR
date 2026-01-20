ALLOCATION_UNIT_SIZE = 512


class _DeviceMemStack:
    def __init__(self) -> None:
        self.allocations = []
        self.current = 0
        self.highwater = 0

    def malloc(self, bytes):
        self.allocations.append(bytes)
        allocated = self._round_up(bytes)
        self.current += allocated
        self.highwater = max(self.current, self.highwater)

    def free(self, bytes):
        assert bytes in self.allocations
        self.allocations.remove(bytes)
        self.current -= self._round_up(bytes)
        assert self.current >= 0

    def _round_up(self, size):
        size = (size + ALLOCATION_UNIT_SIZE - 1) // ALLOCATION_UNIT_SIZE
        return size * ALLOCATION_UNIT_SIZE
