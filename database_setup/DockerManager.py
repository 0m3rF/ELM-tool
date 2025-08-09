
import platform
import subprocess
import threading
import time
from typing import List, Tuple
import click

from elm.models import ContainerResult, ContainerState, DatabaseConfig
from TestDatabaseConfigs import MSSQL_PASSWORD

MSSQL_CMD_BIN = "/opt/mssql-tools18/bin/sqlcmd" # to resolve sonarqube issue
SELECT_1_QUERY = "SELECT 1"

class DockerManager:
    """Manager class for Docker operations with improved orchestration."""
    
    def __init__(self, timeout: int = 600):
        self.lock = threading.Lock()
        self.container_states = {}
        self.default_timeout = timeout
    
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
        success, _, _ = self.run_command(["docker", "info"], timeout=10)
        return success
    
    def check_docker_installation(self) -> Tuple[bool, str]:
        """Check Docker installation and provide installation guidance."""
        # Check if docker command exists
        success, _, stderr = self.run_command(["docker", "--version"], timeout=5)
        if not success:
            return False, self._get_docker_install_instructions()
        
        # Check if Docker daemon is running
        success, _, stderr = self.run_command(["docker", "info"], timeout=10)
        if not success:
            if "permission denied" in stderr.lower():
                return False, "Docker is installed but permission denied. Try running with sudo or add your user to docker group."
            elif "cannot connect" in stderr.lower() or "daemon" in stderr.lower():
                return False, "Docker is installed but daemon is not running. Please start Docker service."
            else:
                return False, f"Docker daemon issue: {stderr}"
        
        return True, "Docker is available and running"
    
    def _get_docker_install_instructions(self) -> str:
        """Get Docker installation instructions based on OS."""
        system = platform.system().lower()
        
        if system == "windows":
            return """
Docker is not installed. Please install Docker Desktop:
1. Download from: https://www.docker.com/products/docker-desktop
2. Run the installer and follow the setup wizard
3. Restart your computer if prompted
4. Start Docker Desktop from the Start menu
"""
        elif system == "darwin":  # macOS
            return """
Docker is not installed. Please install Docker Desktop:
1. Download from: https://www.docker.com/products/docker-desktop
2. Drag Docker.app to Applications folder
3. Launch Docker from Applications
4. Follow the setup instructions
"""
        else:  # Linux
            return """
Docker is not installed. Install using your package manager:

Ubuntu/Debian:
  sudo apt update
  sudo apt install docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  sudo usermod -aG docker $USER

CentOS/RHEL/Fedora:
  sudo dnf install docker
  sudo systemctl start docker
  sudo systemctl enable docker
  sudo usermod -aG docker $USER

After installation, log out and back in for group changes to take effect.
"""
    
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

        click.echo(f"⏳ Waiting for dependencies of {config.name}: {config.depends_on}")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            all_ready = True
            for dep_name in config.depends_on:
                if self.get_container_state(dep_name) != ContainerState.RUNNING:
                    all_ready = False
                    break

            if all_ready:
                click.echo(f"✅ All dependencies ready for {config.name}")
                return True

            time.sleep(2)

        click.echo(f"❌ Timeout waiting for dependencies of {config.name}")
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
            click.echo(f"🔧 Creating container: {config.name}")
            if not self._create_new_container(config):
                return ContainerResult(
                    name=config.name,
                    success=False,
                    state=ContainerState.UNHEALTHY,
                    message="Failed to create container",
                    startup_time=time.time() - start_time
                )
        elif current_state == ContainerState.STOPPED:
            click.echo(f"🔄 Starting stopped container: {config.name}")
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
            click.echo(f"✅ Container already running: {config.name}")

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
            "-p", f"{config.port+1}:{config.port}"  # in case of there is already deployed db use next port.
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

        success, _, stderr = self.run_command(cmd, timeout=self.default_timeout)
        if not success:
            click.echo(f"❌ Failed to create container {config.name}: {stderr}")
            return False

        return True

    def wait_for_container_health(self, config: DatabaseConfig) -> bool:
        """Wait for container to be healthy with retry logic."""
        click.echo(f"⏳ Waiting for {config.name} to be ready...")

        max_retries = 3
        for retry in range(max_retries):
            if retry > 0:
                click.echo(f"🔄 Retry {retry}/{max_retries} for {config.name}")

            # Wait for container to be in running state first
            for _ in range(config.wait_time):
                if self.get_container_state(config.name) == ContainerState.RUNNING:
                    break
                time.sleep(1)
            else:
                click.echo(f"❌ {config.name} failed to start within {config.wait_time} seconds")
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
                        click.echo(f"✅ {config.name} health check passed!")
                        return True
                    else:
                        click.echo(f"⚠️  {config.name} health check failed")
                        if retry < max_retries - 1:
                            time.sleep(10)
                else:
                    # Standard health check for other databases
                    cmd = ["docker", "exec", config.name] + config.health_check_cmd
                    success, _, stderr = self.run_command(cmd, timeout=10)

                    if success:
                        click.echo(f"✅ {config.name} health check passed!")
                        return True
                    else:
                        click.echo(f"⚠️  {config.name} health check failed: {stderr}")
                        if retry < max_retries - 1:
                            time.sleep(10)
            else:
                # If no health check, just verify it's running
                if self.get_container_state(config.name) == ContainerState.RUNNING:
                    click.echo(f"✅ {config.name} is running!")
                    return True

        click.echo(f"❌ {config.name} failed health check after {max_retries} retries")
        return False

    def _check_mssql_health(self, config: DatabaseConfig) -> bool:
        """Special health check for MSSQL with multiple fallback options."""
        health_checks = [
            # Try with trust server certificate and no encryption
            [MSSQL_CMD_BIN, "-S", "localhost", "-U", "sa", "-P", MSSQL_PASSWORD, "-C", "-N", "-Q", SELECT_1_QUERY],
            # Try with just trust server certificate
            [MSSQL_CMD_BIN, "-S", "localhost", "-U", "sa", "-P", MSSQL_PASSWORD, "-C", "-Q", SELECT_1_QUERY],
            # Try without any SSL flags (original command)
            [MSSQL_CMD_BIN, "-S", "localhost", "-U", "sa", "-P", MSSQL_PASSWORD, "-Q", SELECT_1_QUERY],
            # Try with different server specification
            [MSSQL_CMD_BIN, "-S", "127.0.0.1", "-U", "sa", "-P", MSSQL_PASSWORD, "-C", "-Q", SELECT_1_QUERY],
            # Simple connection test
            [MSSQL_CMD_BIN, "-S", "localhost", "-U", "sa", "-P", MSSQL_PASSWORD, "-l", "5", "-Q", "SELECT @@VERSION"]
        ]

        for i, health_cmd in enumerate(health_checks, 1):
            click.echo(f"🔍 Trying MSSQL health check method {i}/{len(health_checks)}")
            cmd = ["docker", "exec", config.name] + health_cmd
            success, _, stderr = self.run_command(cmd, timeout=15)

            if success:
                click.echo(f"✅ MSSQL health check method {i} succeeded!")
                return True
            else:
                click.echo(f"⚠️  MSSQL health check method {i} failed: {stderr[:100]}...")
                time.sleep(2)  # Brief pause between attempts

        # If all health checks fail, try a basic container process check
        click.echo("🔍 Trying basic MSSQL process check...")
        cmd = ["docker", "exec", config.name, "pgrep", "-f", "sqlservr"]
        success, stdout, _ = self.run_command(cmd, timeout=5)

        if success and stdout.strip():
            click.echo("✅ MSSQL process is running - assuming healthy")
            return True

        return False

    def remove_container(self, container_name: str) -> bool:
        """Remove a container (force stop and remove)."""
        # Stop container
        self.run_command(["docker", "stop", container_name])
        # Remove container
        success, _, _ = self.run_command(["docker", "rm", container_name])
        return success

    def pull_image(self, image: str) -> bool:
        """Pull a Docker image."""
        click.echo(f"📥 Pulling image: {image}")
        success, _, stderr = self.run_command(["docker", "pull", image], timeout=self.default_timeout)
        if not success:
            click.echo(f"❌ Failed to pull image {image}: {stderr}")
            return False
        click.echo(f"✅ Successfully pulled image: {image}")
        return True
