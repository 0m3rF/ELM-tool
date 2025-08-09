
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ContainerState(Enum):
    """Container states."""
    NOT_EXISTS = "not_exists"
    STOPPED = "stopped"
    RUNNING = "running"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

@dataclass
class DatabaseConfig:
    """Configuration for database containers."""
    name: str
    image: str
    port: int
    env_vars: Dict[str, str]
    health_check_cmd: List[str]
    wait_time: int = 30
    startup_priority: int = 0  # Lower numbers start first
    depends_on: List[str] = field(default_factory=list)   # List of container names this depends on

@dataclass
class ContainerResult:
    """Result of container operations."""
    name: str
    success: bool
    state: ContainerState
    message: str
    startup_time: float = 0.0
