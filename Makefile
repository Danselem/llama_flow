install-python:
	uv python install


init:
	uv init
	uv tool install black

install:
	uv venv
	. .venv/bin/activate
	uv pip install --all-extras --requirement pyproject.toml
	# uv pip sync requirements.txt
	# uv add -r requirements.txt

runapp:
	uv run src/app.py