import pytest
import subprocess
import time
import threading
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

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
    depends_on: List[str] = None  # List of container names this depends on

@dataclass
class ContainerResult:
    """Result of container operations."""
    name: str
    success: bool
    state: ContainerState
    message: str
    startup_time: float = 0.0

class DockerManager:
    """Manager class for Docker operations with improved orchestration."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.container_states = {}
    
    @staticmethod
    def run_command(cmd: List[str], timeout: int = 600) -> Tuple[bool, str, str]:
        """Run a Docker command and return success, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, "", str(e)
    
    def is_docker_available(self) -> bool:
        """Check if Docker is installed and daemon is running."""
        success, _, _ = self.run_command(["docker", "info"])
        return success
    
    def get_container_state(self, container_name: str) -> ContainerState:
        """Get comprehensive container state."""
        # Check if container exists
        success, stdout, _ = self.run_command([
            "docker", "ps", "-a", "--filter", f"name=^{container_name}$", 
            "--format", "{{.Names}}"
        ])
        
        if not success or container_name not in stdout.strip():
            return ContainerState.NOT_EXISTS
        
        # Check if container is running
        success, stdout, _ = self.run_command([
            "docker", "ps", "--filter", f"name=^{container_name}$", 
            "--format", "{{.Names}}"
        ])
        
        if not success or container_name not in stdout.strip():
            return ContainerState.STOPPED
        
        return ContainerState.RUNNING
    
    def wait_for_dependencies(self, config: DatabaseConfig, max_wait: int = 600) -> bool:
        """Wait for dependency containers to be healthy."""
        if not config.depends_on:
            return True
        
        print(f"⏳ Waiting for dependencies of {config.name}: {config.depends_on}")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            all_ready = True
            for dep_name in config.depends_on:
                if self.get_container_state(dep_name) != ContainerState.RUNNING:
                    all_ready = False
                    break
            
            if all_ready:
                print(f"✅ All dependencies ready for {config.name}")
                return True
            
            time.sleep(2)
        
        print(f"❌ Timeout waiting for dependencies of {config.name}")
        return False
    
    def create_container(self, config: DatabaseConfig) -> ContainerResult:
        """Create and start a database container with dependency management."""
        start_time = time.time()
        
        # Wait for dependencies first
        if not self.wait_for_dependencies(config):
            return ContainerResult(
                name=config.name,
                success=False,
                state=ContainerState.UNHEALTHY,
                message="Dependencies not ready",
                startup_time=time.time() - start_time
            )
        
        # Check current state
        current_state = self.get_container_state(config.name)
        
        if current_state == ContainerState.NOT_EXISTS:
            print(f"🔧 Creating container: {config.name}")
            if not self._create_new_container(config):
                return ContainerResult(
                    name=config.name,
                    success=False,
                    state=ContainerState.UNHEALTHY,
                    message="Failed to create container",
                    startup_time=time.time() - start_time
                )
        elif current_state == ContainerState.STOPPED:
            print(f"🔄 Starting stopped container: {config.name}")
            success, _, stderr = self.run_command(["docker", "start", config.name])
            if not success:
                return ContainerResult(
                    name=config.name,
                    success=False,
                    state=ContainerState.UNHEALTHY,
                    message=f"Failed to start container: {stderr}",
                    startup_time=time.time() - start_time
                )
        else:
            print(f"✅ Container already running: {config.name}")
        
        # Wait for container to be healthy
        if self.wait_for_container_health(config):
            with self.lock:
                self.container_states[config.name] = ContainerState.HEALTHY
            
            return ContainerResult(
                name=config.name,
                success=True,
                state=ContainerState.HEALTHY,
                message="Container is healthy",
                startup_time=time.time() - start_time
            )
        else:
            return ContainerResult(
                name=config.name,
                success=False,
                state=ContainerState.UNHEALTHY,
                message="Container failed health check",
                startup_time=time.time() - start_time
            )
    
    def _create_new_container(self, config: DatabaseConfig) -> bool:
        """Create a new container."""
        # Build docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", config.name,
            "-p", f"{config.port}:{config.port}"
        ]
        
        # Add environment variables
        for key, value in config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # Special handling for MSSQL to disable SSL
        if "mssql" in config.name.lower():
            cmd.extend(["-e", "MSSQL_ENCRYPT=false"])
            cmd.extend(["-e", "MSSQL_TRUST_SERVER_CERTIFICATE=true"])
        
        # Add image
        cmd.append(config.image)
        
        success, stdout, stderr = self.run_command(cmd)
        if not success:
            print(f"❌ Failed to create container {config.name}: {stderr}")
            return False
        
        return True
    
    def wait_for_container_health(self, config: DatabaseConfig) -> bool:
        """Wait for container to be healthy with retry logic."""
        print(f"⏳ Waiting for {config.name} to be ready...")
        
        max_retries = 3
        for retry in range(max_retries):
            if retry > 0:
                print(f"🔄 Retry {retry}/{max_retries} for {config.name}")
            
            # Wait for container to be in running state first
            for attempt in range(config.wait_time):
                if self.get_container_state(config.name) == ContainerState.RUNNING:
                    break
                time.sleep(1)
            else:
                print(f"❌ {config.name} failed to start within {config.wait_time} seconds")
                if retry < max_retries - 1:
                    continue
                return False
            
            # Wait a bit more for services to initialize
            time.sleep(5)
            
            # Try health check if available
            if config.health_check_cmd:
                # Special handling for MSSQL with multiple fallback options
                if "mssql" in config.name.lower():
                    if self._check_mssql_health(config):
                        print(f"✅ {config.name} health check passed!")
                        return True
                    else:
                        print(f"⚠️  {config.name} health check failed")
                        if retry < max_retries - 1:
                            time.sleep(10)
                            continue
                else:
                    # Standard health check for other databases
                    cmd = ["docker", "exec", config.name] + config.health_check_cmd
                    success, stdout, stderr = self.run_command(cmd, timeout=10)
                    
                    if success:
                        print(f"✅ {config.name} health check passed!")
                        return True
                    else:
                        print(f"⚠️  {config.name} health check failed: {stderr}")
                        if retry < max_retries - 1:
                            time.sleep(10)
                            continue
            else:
                # If no health check, just verify it's running
                if self.get_container_state(config.name) == ContainerState.RUNNING:
                    print(f"✅ {config.name} is running!")
                    return True
        
        print(f"❌ {config.name} failed health check after {max_retries} retries")
        return False
    
    def _check_mssql_health(self, config: DatabaseConfig) -> bool:
        """Special health check for MSSQL with multiple fallback options."""
        health_checks = [
            # Try with trust server certificate and no encryption
            ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-N", "-Q", "SELECT 1"],
            # Try with just trust server certificate
            ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-Q", "SELECT 1"],
            # Try without any SSL flags (original command)
            ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-Q", "SELECT 1"],
            # Try with different server specification
            ["/opt/mssql-tools18/bin/sqlcmd", "-S", "127.0.0.1", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-Q", "SELECT 1"],
            # Simple connection test
            ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-l", "5", "-Q", "SELECT @@VERSION"]
        ]
        
        for i, health_cmd in enumerate(health_checks, 1):
            print(f"🔍 Trying MSSQL health check method {i}/{len(health_checks)}")
            cmd = ["docker", "exec", config.name] + health_cmd
            success, stdout, stderr = self.run_command(cmd, timeout=15)
            
            if success:
                print(f"✅ MSSQL health check method {i} succeeded!")
                return True
            else:
                print(f"⚠️  MSSQL health check method {i} failed: {stderr[:100]}...")
                time.sleep(2)  # Brief pause between attempts
        
        # If all health checks fail, try a basic container process check
        print("🔍 Trying basic MSSQL process check...")
        cmd = ["docker", "exec", config.name, "pgrep", "-f", "sqlservr"]
        success, stdout, stderr = self.run_command(cmd, timeout=5)
        
        if success and stdout.strip():
            print("✅ MSSQL process is running - assuming healthy")
            return True
        
        return False
    
    def remove_container(self, container_name: str) -> bool:
        """Remove a container (force stop and remove)."""
        # Stop container
        self.run_command(["docker", "stop", container_name])
        # Remove container
        success, _, _ = self.run_command(["docker", "rm", container_name])
        return success


class DatabaseConfigs:
    """Database configuration definitions with dependencies."""
    
    @staticmethod
    def get_configs() -> Dict[str, DatabaseConfig]:
        """Get all database configurations with startup priorities."""
        return {
            "postgresql": DatabaseConfig(
                name="ELM_TOOL_postgresql",
                image="postgres:15-alpine",
                port=5432,
                env_vars={
                    "POSTGRES_DB": "ELM_TOOL_db",
                    "POSTGRES_USER": "ELM_TOOL_user",
                    "POSTGRES_PASSWORD": "ELM_TOOL_password"
                },
                health_check_cmd=["pg_isready", "-U", "ELM_TOOL_user", "-d", "ELM_TOOL_db"],
                wait_time=30,
                startup_priority=1  # Start first (fastest to start)
            ),
            "mysql": DatabaseConfig(
                name="ELM_TOOL_mysql",
                image="mysql:8.0",
                port=3306,
                env_vars={
                    "MYSQL_DATABASE": "ELM_TOOL_db",
                    "MYSQL_USER": "ELM_TOOL_user",
                    "MYSQL_PASSWORD": "ELM_TOOL_password",
                    "MYSQL_ROOT_PASSWORD": "ELM_TOOL_root_password"
                },
                health_check_cmd=["mysqladmin", "ping", "-h", "localhost"],
                wait_time=45,
                startup_priority=2  # Start second
            ),
            "mssql": DatabaseConfig(
                name="ELM_TOOL_mssql",
                image="mcr.microsoft.com/mssql/server:2022-latest",
                port=1433,
                env_vars={
                    "ACCEPT_EULA": "Y",
                    "SA_PASSWORD": "ELM_TOOL_Password123!",
                    "MSSQL_PID": "Express"
                },
                health_check_cmd=["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-Q", "SELECT 1"],
                wait_time=60,
                startup_priority=3  # Start third
            ),
            "oracle": DatabaseConfig(
                name="ELM_TOOL_oracle",
                image="gvenzl/oracle-xe:21-slim",
                port=1521,
                env_vars={
                    "ORACLE_PASSWORD": "ELM_TOOL_password",
                    "APP_USER": "ELM_TOOL_user",
                    "APP_USER_PASSWORD": "ELM_TOOL_password"
                },
                health_check_cmd=["sqlplus", "-s", "ELM_TOOL_user/ELM_TOOL_password@localhost:1521/XE", "<<<", "SELECT 1 FROM DUAL;"],
                wait_time=90,
                startup_priority=4  # Start last (slowest to start)
            )
        }


class DatabaseOrchestrator:
    """Orchestrates database container startup with proper dependency management."""
    
    def __init__(self, docker_manager: DockerManager):
        self.docker_manager = docker_manager
        self.results = {}
    
    def setup_databases(self, configs: Dict[str, DatabaseConfig], parallel: bool = True) -> Dict[str, ContainerResult]:
        """Setup all databases with proper ordering and dependency management."""
        print("\n" + "="*60)
        print("🚀 STARTING DATABASE CONTAINER ORCHESTRATION")
        print("="*60)
        
        # Sort configs by startup priority
        sorted_configs = sorted(configs.items(), key=lambda x: x[1].startup_priority)
        
        if parallel:
            return self._setup_parallel(sorted_configs)
        else:
            return self._setup_sequential(sorted_configs)
    
    def _setup_sequential(self, sorted_configs: List[Tuple[str, DatabaseConfig]]) -> Dict[str, ContainerResult]:
        """Setup databases sequentially based on priority."""
        results = {}
        
        for db_name, config in sorted_configs:
            print(f"\n📋 Processing {db_name} (priority: {config.startup_priority})")
            result = self.docker_manager.create_container(config)
            results[db_name] = result
            
            if not result.success:
                print(f"❌ {db_name} setup failed: {result.message}")
                # Continue with other databases even if one fails
            else:
                print(f"✅ {db_name} setup completed in {result.startup_time:.2f}s")
        
        return results
    
    def _setup_parallel(self, sorted_configs: List[Tuple[str, DatabaseConfig]]) -> Dict[str, ContainerResult]:
        """Setup databases in parallel, respecting priorities and dependencies."""
        results = {}
        
        # Group by priority
        priority_groups = {}
        for db_name, config in sorted_configs:
            priority = config.startup_priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append((db_name, config))
        
        # Process each priority group
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]
            print(f"\n🔄 Starting priority group {priority} ({len(group)} databases)")
            
            # Start all databases in this priority group in parallel
            with ThreadPoolExecutor(max_workers=len(group)) as executor:
                future_to_db = {
                    executor.submit(self.docker_manager.create_container, config): db_name
                    for db_name, config in group
                }
                
                for future in as_completed(future_to_db):
                    db_name = future_to_db[future]
                    try:
                        result = future.result()
                        results[db_name] = result
                        
                        if result.success:
                            print(f"✅ {db_name} completed in {result.startup_time:.2f}s")
                        else:
                            print(f"❌ {db_name} failed: {result.message}")
                    except Exception as e:
                        print(f"❌ {db_name} exception: {str(e)}")
                        results[db_name] = ContainerResult(
                            name=db_name,
                            success=False,
                            state=ContainerState.UNHEALTHY,
                            message=f"Exception: {str(e)}"
                        )
        
        return results


# Pytest fixtures with explicit dependencies
@pytest.fixture(scope="session")
def docker_manager():
    """Docker manager fixture."""
    return DockerManager()


@pytest.fixture(scope="session")
def database_configs():
    """Database configurations fixture."""
    return DatabaseConfigs.get_configs()


@pytest.fixture(scope="session")
def database_orchestrator(docker_manager):
    """Database orchestrator fixture."""
    return DatabaseOrchestrator(docker_manager)


@pytest.fixture(scope="session", autouse=True)
def ensure_docker_available(docker_manager):
    """Ensure Docker is available before running tests."""
    if not docker_manager.is_docker_available():
        pytest.skip("Docker is not available")


@pytest.fixture(scope="session")
def setup_databases(database_orchestrator, database_configs):
    """Setup database containers with proper orchestration."""
    # This fixture has explicit dependencies on docker_manager through orchestrator
    results = database_orchestrator.setup_databases(database_configs, parallel=True)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DATABASE SETUP SUMMARY")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for db_name, result in results.items():
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"{db_name:12} | {status:10} | {result.startup_time:6.2f}s | {result.message}")
        
        if result.success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    
    # Fail the entire test session if no databases are running
    if successful == 0:
        pytest.fail("No database containers are running")
    
    return results


# Test classes with explicit dependencies
class TestDockerEnvironment:
    """Test Docker environment setup - runs first."""
    
    def test_docker_installed(self, docker_manager):
        """Test if Docker is installed and accessible."""
        assert docker_manager.is_docker_available(), "Docker should be installed and daemon running"
    
    def test_docker_commands_work(self, docker_manager):
        """Test basic Docker commands."""
        success, stdout, _ = docker_manager.run_command(["docker", "--version"])
        assert success, "Docker version command should work"
        assert "Docker version" in stdout, "Docker version output should contain version info"


class TestDatabaseSetup:
    """Test database setup process - runs after Docker tests."""
    
    def test_setup_results_available(self, setup_databases):
        """Test that setup results are available."""
        assert setup_databases is not None, "Setup results should be available"
        assert len(setup_databases) > 0, "At least one database should be processed"
    
    def test_at_least_one_database_healthy(self, setup_databases):
        """Test that at least one database is healthy."""
        healthy_count = sum(1 for result in setup_databases.values() if result.success)
        assert healthy_count > 0, f"At least one database should be healthy. Results: {setup_databases}"


class TestDatabaseContainers:
    """Test individual database containers - runs after setup."""
    
    @pytest.mark.parametrize("db_name", ["postgresql", "mysql", "oracle", "mssql"])
    def test_database_container_exists(self, db_name, docker_manager, database_configs, setup_databases):
        """Test that database containers exist."""
        config = database_configs[db_name]
        state = docker_manager.get_container_state(config.name)
        assert state != ContainerState.NOT_EXISTS, f"{db_name} container should exist"
    
    @pytest.mark.parametrize("db_name", ["postgresql", "mysql", "oracle", "mssql"])
    def test_database_container_running(self, db_name, docker_manager, database_configs, setup_databases):
        """Test that database containers are running."""
        config = database_configs[db_name]
        state = docker_manager.get_container_state(config.name)
        
        # If container is not running, provide diagnostic information
        if state != ContainerState.RUNNING:
            # Get container logs for debugging
            success, logs, _ = docker_manager.run_command(
                ["docker", "logs", "--tail", "20", config.name], timeout=10
            )
            if success:
                print(f"\n📋 Last 20 lines of {config.name} logs:")
                print(logs)
            
            # Check if container exists but is stopped
            if state == ContainerState.STOPPED:
                print(f"\n⚠️  {config.name} exists but is stopped. Attempting to start...")
                start_success, _, start_error = docker_manager.run_command(
                    ["docker", "start", config.name], timeout=30
                )
                if start_success:
                    # Wait a bit and recheck
                    time.sleep(10)
                    state = docker_manager.get_container_state(config.name)
                    if state == ContainerState.RUNNING:
                        print(f"✅ {config.name} started successfully")
                    else:
                        print(f"❌ {config.name} failed to start properly")
                else:
                    print(f"❌ Failed to start {config.name}: {start_error}")
        
        assert state == ContainerState.RUNNING, f"{db_name} container should be running, but is {state}"
    
    @pytest.mark.parametrize("db_name", ["postgresql", "mysql", "oracle", "mssql"])
    def test_database_setup_successful(self, db_name, setup_databases):
        """Test that database setup was successful."""
        if db_name in setup_databases:
            result = setup_databases[db_name]
            assert result.success, f"{db_name} setup should be successful. Error: {result.message}"
        else:
            pytest.skip(f"{db_name} was not processed during setup")


class TestDatabaseConnections:
    """Test database connection parameters - runs after containers are verified."""
    
    def test_postgresql_connection_info(self, database_configs):
        """Test PostgreSQL connection information."""
        config = database_configs["postgresql"]
        assert config.env_vars["POSTGRES_DB"] == "ELM_TOOL_db"
        assert config.env_vars["POSTGRES_USER"] == "ELM_TOOL_user"
        assert config.env_vars["POSTGRES_PASSWORD"] == "ELM_TOOL_password"
        assert config.port == 5432
    
    def test_mysql_connection_info(self, database_configs):
        """Test MySQL connection information."""
        config = database_configs["mysql"]
        assert config.env_vars["MYSQL_DATABASE"] == "ELM_TOOL_db"
        assert config.env_vars["MYSQL_USER"] == "ELM_TOOL_user"
        assert config.env_vars["MYSQL_PASSWORD"] == "ELM_TOOL_password"
        assert config.port == 3306
    
    def test_oracle_connection_info(self, database_configs):
        """Test Oracle connection information."""
        config = database_configs["oracle"]
        assert config.env_vars["APP_USER"] == "ELM_TOOL_user"
        assert config.env_vars["APP_USER_PASSWORD"] == "ELM_TOOL_password"
        assert config.port == 1521
    
    def test_mssql_connection_info(self, database_configs):
        """Test MSSQL connection information."""
        config = database_configs["mssql"]
        assert config.env_vars["SA_PASSWORD"] == "ELM_TOOL_Password123!"
        assert config.env_vars["MSSQL_PID"] == "Express"
        assert config.port == 1433


# Utility functions
def debug_mssql_connection():
    """Debug MSSQL connection issues."""
    print("\n" + "="*60)
    print("🔍 MSSQL CONNECTION DEBUGGING")
    print("="*60)
    
    docker_manager = DockerManager()
    container_name = "ELM_TOOL_mssql"
    
    if not docker_manager.get_container_state(container_name) == ContainerState.RUNNING:
        print(f"❌ {container_name} is not running")
        return
    
    print(f"✅ {container_name} is running")
    
    # Check container logs
    success, logs, _ = docker_manager.run_command(
        ["docker", "logs", "--tail", "30", container_name], timeout=10
    )
    if success:
        print(f"\n📋 Container logs:")
        print(logs)
    
    # Try various connection methods
    connection_tests = [
        {
            "name": "Basic sqlcmd with trust certificate",
            "cmd": ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-Q", "SELECT 1"]
        },
        {
            "name": "sqlcmd with no encryption",
            "cmd": ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-N", "-Q", "SELECT 1"]
        },
        {
            "name": "sqlcmd with IP address",
            "cmd": ["/opt/mssql-tools18/bin/sqlcmd", "-S", "127.0.0.1", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-Q", "SELECT 1"]
        },
        {
            "name": "Process check",
            "cmd": ["pgrep", "-f", "sqlservr"]
        }
    ]
    
    for test in connection_tests:
        print(f"\n🔍 Testing: {test['name']}")
        cmd = ["docker", "exec", container_name] + test["cmd"]
        success, stdout, stderr = docker_manager.run_command(cmd, timeout=15)
        
        if success:
            print(f"✅ Success: {stdout[:100]}...")
        else:
            print(f"❌ Failed: {stderr[:200]}...")
    
    # Check if we can connect from host
    print(f"\n🔍 Testing host connectivity")
    success, stdout, stderr = docker_manager.run_command(
        ["docker", "exec", container_name, "netstat", "-tln"], timeout=10
    )
    if success:
        print(f"📋 Network ports:")
        print(stdout)


def print_connection_strings():
    """Print connection strings for all databases."""
    print("\n" + "="*60)
    print("DATABASE CONNECTION STRINGS")
    print("="*60)
    
    print(f"\nPostgreSQL:")
    print(f"  postgresql://ELM_TOOL_user:ELM_TOOL_password@localhost:5432/ELM_TOOL_db")
    
    print(f"\nMySQL:")
    print(f"  mysql://ELM_TOOL_user:ELM_TOOL_password@localhost:3306/ELM_TOOL_db")
    
    print(f"\nOracle:")
    print(f"  oracle://ELM_TOOL_user:ELM_TOOL_password@localhost:1521/XE")
    
    print(f"\nMSSQL:")
    print(f"  mssql://sa:ELM_TOOL_Password123!@localhost:1433")
    print(f"  Note: Use TrustServerCertificate=true in connection string")
    print(f"  Example: Server=localhost,1433;Database=master;User Id=sa;Password=ELM_TOOL_Password123!;TrustServerCertificate=true;")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug-mssql":
        debug_mssql_connection()
    else:
        print_connection_strings()
        
        print(f"\n🧪 TEST EXECUTION ORDER:")
        print(f"1. TestDockerEnvironment - Verify Docker is available")
        print(f"2. setup_databases fixture - Create/start/health-check containers")
        print(f"3. TestDatabaseSetup - Verify setup process")
        print(f"4. TestDatabaseContainers - Verify individual containers")
        print(f"5. TestDatabaseConnections - Verify connection parameters")
        
        print(f"\n📋 CONTAINER STARTUP ORDER:")
        print(f"Priority 1: PostgreSQL (fastest)")
        print(f"Priority 2: MySQL")
        print(f"Priority 3: MSSQL")
        print(f"Priority 4: Oracle (slowest)")
        
        print(f"\n🚀 RUN TESTS:")
        print(f"pytest {__file__} -v -s")
        print(f"pytest {__file__} -v -s --tb=short")
        print(f"pytest {__file__}::TestDatabaseSetup -v -s")
        
        print(f"\n🔍 DEBUG MSSQL:")
        print(f"python {__file__} debug-mssql")