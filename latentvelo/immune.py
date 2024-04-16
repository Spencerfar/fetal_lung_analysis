import scanpy as sc
import scvelo as scv
import latentvelo as ltv
from preprocess import preprocessed_clean
import numpy as np
import pandas as pd

c5 = sc.read('../../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_others/c5_anndata.h5ad')
c6 = sc.read('../../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_others/c6_anndata.h5ad')

import anndata as ad
adata = ad.concat([c5, c6], label='compartment')



meta = pd.read_csv('../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_c2/immune_metadata.csv',
                  index_col=0)

umap = pd.read_csv('../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_c2/immune_cell_embeddings.csv',
                  index_col=0)

adata.obs['id'] = [str(x.split('-')[0]) +'-'+ str(x.split('-')[1]) for x in adata.obs.index.values]
adata.obs = adata.obs.set_index('id')

adata.obs['cell_type'] = meta['cell_type']
adata.obs['UMAP1'] = umap['UMAP_1']
adata.obs['UMAP2'] = umap['UMAP_2']
adata.obsm['X_umap'] = np.array([adata.obs.UMAP1, adata.obs.UMAP2]).T

scv.pp.filter_genes(adata, min_shared_cells=100)
scv.pp.filter_and_normalize(adata)

scv.pp.filter_genes_dispersion(adata, n_top_genes = 3000, flavor='cell_ranger', log=False)

preprocessed_clean(adata, celltype_key='cell_type', normalize_library=True, batch_key='batch',
                  recompute_pca=True, recompute_neighbors=True, recompute_moments=True,
                  n_neighbors=100)


model = ltv.models.VAE(observed=3000, latent_dim=50, encoder_hidden=75, zr_dim=3, h_dim=3,
                           batch_correction=True, 
                           batches = len(adata.obs.batch.unique()))

epochs, val_ae, val_traj = ltv.train(model, adata, epochs=100, batch_size = 500,
                                     name='immune_parameters',
                                     grad_clip=100,
                                     learning_rate=1e-2)

import torch as th
#model.load_state_dict(th.load('immune_parameters_50/model_state_epoch64.params',map_location=th.device('cuda')))


latent_adata, adata = ltv.output_results(model, adata, decoded=True, gene_velocity=True, 
                                        R2_scores=True, R2_unspliced=True, celltype_R2=True, 
                                     num_greater_zero=25, celltype_key='cell_type')

scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')


refine_model = ltv.models.RefineODE(model,observed=3000, latent_dim=50, encoder_hidden=75, zr_dim=3, h_dim=3,
                           batch_correction=True, 
                           batches = len(adata.obs.batch.unique()),
                          max_sigma_z=0.0025, 
                                    path_reg=True)

epochs, val_ae, val_traj,_,_ = ltv.train(refine_model, adata, epochs=100, batch_size = 500, name='immune_parameters_refine', 
                                     grad_clip=100, learning_rate=5e-2, timing=True, shuffle=False)

#model.load_state_dict(th.load('immune_parameters_refine/model_state_epoch96.params',map_location=th.device('cuda')))

latent_adata, adata = ltv.output_results(refine_model, adata, decoded=True, gene_velocity=True, 
                                        R2_scores=True, R2_unspliced=True, celltype_R2=True, 
                                     num_greater_zero=25, celltype_key='cell_type')

scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')


latent_adata.write('fits/latentvelo_immune_50.h5ad')
adata.write('../../../../../../../media/spencer/Extra HDD/fits/adata_immune_50.h5ad')
