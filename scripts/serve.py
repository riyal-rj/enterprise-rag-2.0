"""Production entrypoint for the container (Docker ``CMD``).

Not used for local dev — ``make api`` runs uvicorn directly with
``--reload``. This script is the fixed, no-reload entrypoint the Docker
image's ``CMD`` points at; ``log_config=None`` leaves logging to the app's
own configuration instead of uvicorn's default dictConfig.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_config=None,
    )
