import scanpy as sc
import scvelo as scv
import latentvelo as ltv
from preprocess import preprocessed_clean
import numpy as np

adata = sc.read('../../../../../../../media/spencer/Seagate Backup Plus Drive/scRNAseq/lung/raw_others/c3_anndata.h5ad')
adata.obs['annotations'] = adata.obsm['Clusters'][:,0]
adata.obs.annotations = adata.obs.annotations.astype('category')

scv.pp.filter_genes(adata, min_shared_cells=100)
scv.pp.filter_and_normalize(adata)


scv.pp.filter_genes_dispersion(adata, n_top_genes = 3000, flavor='cell_ranger', log=False)
preprocessed_clean(adata, celltype_key='annotations', normalize_library=False, batch_key='batch',
                  recompute_pca=True, recompute_neighbors=True, recompute_moments=True,
                  n_neighbors=100)


model = ltv.models.VAE(observed=3000, latent_dim=50, encoder_hidden=75, zr_dim=3, h_dim=3,
                           batch_correction=True, 
                           batches = len(adata.obs.batch.unique()))

epochs, val_ae, val_traj = ltv.train(model, adata, epochs=100, batch_size = 500,
                                     name='endothelial_parameters',
                                     grad_clip=100,
                                     learning_rate=1e-2)


import torch as th
#model.load_state_dict(th.load('endothelial_parameters1/model_state_epoch93.params',map_location=th.device('cuda')))

refine_model = ltv.models.RefineODE(model,observed=3000, latent_dim=50, encoder_hidden=75, zr_dim=3, h_dim=3,
                           batch_correction=True, 
                           batches = len(adata.obs.batch.unique()),
                          max_sigma_z=0.0025, 
                                    path_reg=True)

epochs, val_ae, val_traj,_,_ = ltv.train(refine_model, adata, epochs=100, batch_size = 500, name='endothelial_parameters_refine', 
                                     grad_clip=100, learning_rate=5e-2, timing=True, shuffle=False)

latent_adata, adata = ltv.output_results(refine_model, adata, decoded=True, gene_velocity=True, 
                                        R2_scores=True, R2_unspliced=True, celltype_R2=True, 
                                     num_greater_zero=25, celltype_key='annotations')

scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')

latent_adata.write('fits/latentvelo_endothelial_Sept11.h5ad')
adata.write('../../../../../../../media/spencer/Extra HDD/fits/adata_endothelial_Sept11.h5ad')
