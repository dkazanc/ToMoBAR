ALLOCATION_UNIT_SIZE = 512


class DeviceMemStack:
    _instance = None
    _stack_count = 0

    def __enter__(self):
        if DeviceMemStack._stack_count == 0:
            DeviceMemStack._instance = self

        DeviceMemStack._stack_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        DeviceMemStack._stack_count -= 1

        if DeviceMemStack._stack_count == 0:
            DeviceMemStack._instance = None

    @classmethod
    def instance(cls):
        return cls._instance

    def __init__(self) -> None:
        self.allocations = []
        self.current = 0
        self.highwater = 0

    def malloc(self, byte_count):
        self.allocations.append(byte_count)
        allocated = self._round_up(byte_count)
        self.current += allocated
        self.highwater = max(self.current, self.highwater)

    def free(self, byte_count):
        assert byte_count in self.allocations
        self.allocations.remove(byte_count)
        self.current -= self._round_up(byte_count)
        assert self.current >= 0

    def _round_up(self, size):
        size = (size + ALLOCATION_UNIT_SIZE - 1) // ALLOCATION_UNIT_SIZE
        return size * ALLOCATION_UNIT_SIZE
