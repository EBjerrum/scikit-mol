# Sync the .py and .ipynb
source ./sync_notebooks.sh
# Execute the notebooks, gives a .nbconvert.ipynb extension
jupyter nbconvert --to notebook --execute *ipynb
# move the .nbconvert.ipynb to the original .ipynb
for file in *.nbconvert.ipynb; do 
    fname=${file/.nbconvert.ipynb/};
    rm $fname.ipynb
    mv $file $fname.ipynb
done

