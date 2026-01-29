# Requirements

## Runtime Dependencies

The following packages are required for `coreason-optimizer` to function:

*   `python >= 3.12`
*   `click >= 8.0.0`
*   `jinja2 >= 3.0.0`
*   `loguru >= 0.6.0`
*   `numpy >= 1.20.0`
*   `openai >= 1.0.0`
*   `pydantic >= 2.0.0`
*   `scikit-learn >= 1.0.0`
*   `coreason-identity >= 0.4.1`

### Server Mode (Microservice)

For running the optimization microservice (Server Mode), the following additional dependencies are required:

*   `fastapi >= 0.100.0`
*   `uvicorn >= 0.20.0`
*   `httpx`
*   `anyio`

## Development Dependencies

For development and testing:

*   `pytest`
*   `ruff`
*   `pre-commit`
*   `pytest-cov`
*   `mkdocs`
*   `mkdocs-material`
