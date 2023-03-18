import pytest
import numpy as np
from rdkit import Chem
from fixtures import mols_list, smiles_list
from test_fptransformers import morgan_transformer, atompair_transformer, topologicaltorsion_transformer, rdkit_transformer


def test_Transformer_exotic_types(mols_list, morgan_transformer,atompair_transformer, topologicaltorsion_transformer):
    for transformer in [morgan_transformer, atompair_transformer, topologicaltorsion_transformer]:
        params = transformer.get_params()

        for useCounts in [np.bool_(True), np.bool_(False)]:
            
            for key, value in params.items():
                if isinstance(value, int):
                    exotic_type_value = np.int64(value)
                elif isinstance(value, bool):
                    exotic_type_value = np.bool_(value)
                else:
                    print(f'{key}:{value}:{type(value)}')
                    exotic_type_value = value

                exotic_params = {key:exotic_type_value, 'useCounts':useCounts}
                print(exotic_params)    
                transformer.set_params(**exotic_params)
                transformer.transform(mols_list)


def test_RDKFp_exotic_types(mols_list, rdkit_transformer):
        transformer = rdkit_transformer
        params = transformer.get_params()

        for key, value in params.items():
            if isinstance(value, int):
                exotic_type_value = np.int64(value)
            elif isinstance(value, bool):
                exotic_type_value = np.bool_(value)
            else:
                print(f'{key}:{value}:{type(value)}')
                exotic_type_value = value

            exotic_params = {key:exotic_type_value}
            print(exotic_params)    
            transformer.set_params(**exotic_params)
            transformer.transform(mols_list)

