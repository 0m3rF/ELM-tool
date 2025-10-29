
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


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
    target_port: int  # Host port to expose the container on
    env_vars: Dict[str, str]
    health_check_cmd: List[str]
    default_port: Optional[int] = None  # Default port inside the container (if None, uses target_port)
    wait_time: int = 30
    startup_priority: int = 0  # Lower numbers start first
    depends_on: List[str] = field(default_factory=list)   # List of container names this depends on

    def __post_init__(self):
        """Set default_port to target_port if not specified."""
        if self.default_port is None:
            self.default_port = self.target_port

@dataclass
class ContainerResult:
    """Result of container operations."""
    name: str
    success: bool
    state: ContainerState
    message: str
    startup_time: float = 0.0
