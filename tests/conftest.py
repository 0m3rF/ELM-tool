from elm.elm_utils import variables
import pytest

@pytest.fixture
def temp_env_dir():
    return variables.ENVS_FILE