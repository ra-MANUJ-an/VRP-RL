```python
from typing import *
from collections import *
from random import *
from pyvrp import *

class CostEvaluator:
    def __init__(
        self,
        load_penalties: list[float],
        tw_penalty: float,
        dist_penalty: float,
    ) -> None:
        self.load_penalties = load_penalties
        self.tw_penalty = tw_penalty
        self.dist_penalty = dist_penalty

    def load_penalty(
        self, load: int, capacity: int, dimension: int
    ) -> int:
        return load * capacity * dimension

    def tw_penalty(self, time_warp: int) -> int:
        return time_warp * 2

    def dist_penalty(self, distance: int, max_distance: int) -> int:
        return distance * (max_distance - distance)

    def penalised_cost(self, solution: Solution) -> int:
        return sum(
            self.load_penalty(client.load, client.capacity, client.dimension)
            for client in solution.clients
        ) + sum(
            self.tw_penalty(client.tw_early, client.tw_late)
            for client in solution.clients
        ) + sum(
            self.dist_penalty(client.distance, max_distance)
            for client in solution.clients
        )

class DynamicBitset:
    def __init__(self, num_bits: int) -> None:
        self.num_bits = num_bits
        self.bitset = DynamicBitset(num_bits)
        self.reset()

    def __eq__(self, other: object) -> bool:
        return self.bitset == other.bitset

    def __getitem__(self, idx: int) -> bool:
        return self.bitset[idx]

    def __setitem__(self, idx: int, value: bool) -> None:
        self.bitset[idx] = value

    def all(self) -> bool:
        return all(self.bitset)

    def any(self) -> bool:
        return any(self.bitset)

    def none(self) -> bool:
        return not self.bitset

    def count(self) -> int:
        return sum(self.bitset)

    def __len__(self) -> int:
        return self.num_bits

    def __or__(self, other: DynamicBitset) -> DynamicBitset:
        return DynamicBitset(self.num_bits | other.num_bits)

    def __and__(self, other: DynamicBitset) -> DynamicBitset:
        return DynamicBitset(self.num_bits & other.num_bits)

    def __xor__(self, other: DynamicBitset) -> DynamicBitset:
        return DynamicBitset(self.num_bits ^ other.num_bits)

    def __invert__(self) -> DynamicBitset:
        return DynamicBitset(self.num_bits ^ 0)

    def reset(self) -> DynamicBitset:
        self.bitset = DynamicBitset(self.num_bits)

class Client:
    x: int
    y: int
    delivery: list[int]
    pickup: list[int]
    service_duration: int
    tw_early: int
    tw_late: int
    release_time: int
    prize: int
    required: bool
    group: int | None
    name: str
    def __init__(
        self,
        x: int,
        y: int,
        delivery: list[int] = [],
        pickup: list[int] = [],
        service_duration: int = 0,
        tw_early: int = 0,
        tw_late: int = ...,
        release_time: int = 0,
        prize: int = 0,
        required: bool = True,
        group: int | None = None,
        *,
        name: str = "",
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple, /) -> None: ...

class ClientGroup:
    required: bool
    mutually_exclusive: bool
    def __init__(
        self,
        clients: list[int] = [],
        required: bool = True,
    ) -> None: ...
    @property
    def clients(self) -> list[int]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def add_client(self, client: int) -> None: ...
    def clear(self) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple, /) -> None: ...

class Depot:
    x: int
    y: int
    service_duration: int
    tw_early: int
    tw_late: int
    name: str
    def __init__(
        self,
        x: int,
        y: int,
        service_duration