import subprocess
import unittest
import pytest

def is_docker_installed() -> bool:
    """
    Check if Docker is installed.
    Returns True if Docker is installed, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def is_docker_daemon_running() -> bool:
    """
    Check if Docker daemon is running.
    Returns True if daemon is accessible, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

# Example usage and test functions
def run_docker_tests():
    """
    Test function that checks Docker availability.
    """
    print("=== Docker Availability Check ===")

    # Check if Docker is installed
    if not is_docker_installed():
        print("❌ Docker is not installed or not in PATH")
        return False
    else:
        print("✅ Docker is installed")

    # Check if Docker daemon is running
    if not is_docker_daemon_running():
        print("❌ Docker daemon is not running")
        return False
    else:
        print("✅ Docker daemon is running")

    print("✅ Docker is available and ready to use")
    return True


class DockerTestCase(unittest.TestCase):
    """
    Test case class for Docker availability tests.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        pass

    @pytest.mark.dependency(name="docker_installation")
    def test_docker_installation(self):
        """Test if Docker is properly installed."""
        self.assertTrue(is_docker_installed(), "Docker should be installed")

    @pytest.mark.dependency(name="docker_daemon_running",depends=["test_docker_installation"])
    def test_docker_daemon_running(self):
        """Test if Docker daemon is running."""
        self.assertTrue(is_docker_daemon_running(), "Docker daemon should be running")

    @pytest.mark.dependency(name="docker_availability",depends=["test_docker_installation","test_docker_daemon_running"])
    def test_docker_availability(self):
        """Test overall Docker availability."""
        docker_installed = is_docker_installed()
        daemon_running = is_docker_daemon_running()

        self.assertTrue(docker_installed, "Docker should be installed")
        self.assertTrue(daemon_running, "Docker daemon should be running")

        if docker_installed and daemon_running:
            print("✅ Docker is fully available and ready to use")
        else:
            print("❌ Docker is not properly available")


if __name__ == "__main__":
    # Run the example tests
    print("Running Docker availability tests...")
    run_docker_tests()

    print("\n" + "="*50)
    print("Running unittest tests...")

    # Run unittest tests
    unittest.main(verbosity=2)