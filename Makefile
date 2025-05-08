sync-notebooks:
	uv run jupytext --set-formats docs//notebooks//ipynb,docs//notebooks//scripts//py:percent --sync docs/notebooks/*.ipynb
	uv run ruff format "docs/notebooks/"

run-notebooks:
	uv run jupytext --execute docs/notebooks/*ipynb