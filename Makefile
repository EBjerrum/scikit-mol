sync-py:
	uv run jupytext --sync "docs/notebooks/*.ipynb"

sync-ipynb:
	uv run jupytext --sync "docs/notebooks/*.py"

run-notebooks:
	# Execute the notebooks, gives a .nbconvert.ipynb extension
	jupyter nbconvert --to notebook --execute *ipynb
	# move the .nbconvert.ipynb to the original .ipynb
	for file in *.nbconvert.ipynb; do 
		fname=${file/.nbconvert.ipynb/};
		rm $fname.ipynb
		mv $file $fname.ipynb
	done