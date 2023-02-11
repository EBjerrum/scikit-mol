# Safer to instantiate the transformer object in the thread, and only transfer the parameters
def parallel_helper(args):
    from scikit_mol.descriptors import Desc2DTransformer
    #from rdkit import Chem
    params, x = args
    transformer = Desc2DTransformer(**params)
    #print(Chem.MolToSmiles(x[0]), flush=True)
    #print(type(x), flush=True)
    #transformer._transform_mol(Chem.MolFromSmiles('c1ccccc1'))
    y = transformer._transform(x)
    return y
    return np.zeros((10,10))

    del transformer
    return y
