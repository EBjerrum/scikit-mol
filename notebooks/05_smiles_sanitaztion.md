# SMILES sanitation
Sometimes we are faced with datasets which has SMILES that rdkit doesn't want to sanitize. This can be human entry errors, or differences between RDKits more strict sanitazion and other toolkits implementations of the parser. e.g. RDKit will not handle a tetravalent nitrogen when it has no charge, where other toolkits may simply build the graph anyway, disregarding the issues with the valence rules or guessing that the nitrogen should have a charge, where it could also by accident instead have a methyl group too many.


```python
import pandas as pd
from rdkit.Chem import PandasTools

csv_file = "../tests/data/SLC6A4_active_excapedb_subset.csv" # Hmm, maybe better to download directly
data = pd.read_csv(csv_file)


```

Now, this example dataset contain all sanitizable SMILES, so for demonstration purposes, we will corrupt one of them


```python
data.loc[1,'SMILES'] = 'CN(C)(C)(C)'
```


```python

PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
print(f'Dataset contains {data.ROMol.isna().sum()} unparsable mols')

```

    Dataset contains 1 unparsable mols


    [19:45:39] Explicit valence for atom # 1 N, 4, is greater than permitted


If we use these SMILES for the scikit-learn pipeline, we would face an error, so we need to check and clean the dataset first. The CheckSmilesSanitation can help us with that.


```python
from scikit_mol.sanitizer import CheckSmilesSanitazion
smileschecker = CheckSmilesSanitazion()

smiles_list_valid, y_valid, smiles_errors, y_errors = smileschecker.sanitize(list(data.SMILES), list(data.pXC50))
```

    Error in parsing 1 SMILES. Unparsable SMILES can be found in self.errors


    [19:45:39] Explicit valence for atom # 1 N, 4, is greater than permitted


Now the smiles_list_valid should be all valid and the y_values filtered as well. Errors are returned, but also accesible after the call to .sanitize() in the .errors property


```python
smileschecker.errors
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CN(C)(C)(C)</td>
      <td>7.18046</td>
    </tr>
  </tbody>
</table>
</div>



The checker can also be used only on X


```python
smiles_list_valid, X_errors = smileschecker.sanitize(list(data.SMILES))
smileschecker.errors
```

    Error in parsing 1 SMILES. Unparsable SMILES can be found in self.errors


    [19:45:39] Explicit valence for atom # 1 N, 4, is greater than permitted





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CN(C)(C)(C)</td>
    </tr>
  </tbody>
</table>
</div>


