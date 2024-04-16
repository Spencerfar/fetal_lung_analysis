import scanpy as sc
import scvelo as scv
import latentvelo as ltv
from preprocess import preprocessed_clean
import numpy as np
import pandas as pd

adata = sc.read('../../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_c2/c2_anndata.h5ad')
adata.obs['annotations'] = adata.obsm['Clusters'][:,0]
adata.obs.annotations = adata.obs.annotations.astype('category')

meta = pd.read_csv('../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_c2/c2_final_metadata.csv',
                  index_col=0)

umap = pd.read_csv('../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_c2/c2_final_updated_umap.csv',
                  index_col=0)

adata.obs['id'] = [str(x.split('-')[0]) +'-'+ str(x.split('-')[1]) for x in adata.obs.index.values]
adata.obs = adata.obs.set_index('id')

adata.obs['cell_type'] = meta['cell_type_Aug']
adata.obs['epithelial_ident'] = meta['epithelial_ident']
adata.obs['Newepithelialident'] = meta['Newepithelialident']
adata.obs['Clusters'] = meta['Clusters']
adata.obs['UMAP1'] = umap['UMAP_1']
adata.obs['UMAP2'] = umap['UMAP_2']
adata.obsm['X_umap'] = np.array([adata.obs.UMAP1, adata.obs.UMAP2]).T

adata = adata[~adata.obs.cell_type.isin(['SMG basal cells', 'SMG secretory cells',
                                          'NRGN+ cells', 'Schwann'])]
adata = adata[~adata.obs['cell_type'].isna()]

import gc
gc.collect()

scv.pp.filter_genes(adata, min_shared_cells=30)
scv.pp.filter_and_normalize(adata)
scv.pp.filter_genes_dispersion(adata, n_top_genes = 3000, flavor='cell_ranger', log=False)
gc.collect()

preprocessed_clean(adata, celltype_key='cell_type', normalize_library=False, batch_key='batch',
                  recompute_pca=True, recompute_neighbors=True, recompute_moments=True,
                  n_neighbors=100)

model = ltv.models.VAE(observed=3000, latent_dim=100, encoder_hidden=125, zr_dim=4, h_dim=4,
                            batch_correction=True, 
                            batches = len(adata.obs.batch.unique()))

epochs, val_ae, val_traj = ltv.train(model, adata, epochs=0, batch_size = 500,
                                     name='epithelial',
                                     grad_clip=100,
                                     learning_rate=1e-2)


import torch as th
#model.load_state_dict(th.load('epithelial/model_state_epoch72.params',map_location=th.device('cuda')))

latent_adata = ltv.output_results(model, adata)

scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')


refine_model = ltv.models.RefineODE(model,observed=3000, latent_dim=100, encoder_hidden=125, zr_dim=4,
                            h_dim=4,batch_correction=True, 
                           batches = len(adata.obs.batch.unique()),
                          max_sigma_z=0.0025, path_reg=True)

epochs, val_ae, val_traj,_,_ = ltv.train(refine_model, adata, epochs=0, batch_size = 500, name='epithelial_refine', 
                                     grad_clip=100, learning_rate=5e-2, timing=True, shuffle=False)


#refine_model.load_state_dict(th.load('epithelial_refine/model_state_epoch98.params',map_location=th.device('cuda')))


latent_adata, adata = ltv.output_results(refine_model, adata, decoded=True, gene_velocity=True, 
                                        R2_scores=True, R2_unspliced=True, celltype_R2=True, 
                                     num_greater_zero=25, celltype_key='cell_type')


scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')

latent_adata.write('../../../../../../../media/spencer/Extra HDD/fits/latentvelo_C2.h5ad', compression='gzip')

# write adata
adata.write('../../../../../../../media/spencer/Extra HDD/fits/adata_C2.h5ad')
