import os
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata._core.views import ArrayView
import anndata
import scipy
from scipy import sparse
from typing import Union
from .nn import SDNE
from .stat import null_test
sc.settings.verbosity = 0
import warnings
warnings.filterwarnings("ignore")

class GRN:
    def __init__(self,
                 name: str = None,
                 data: anndata.AnnData = None,
                 GRN_file_dir: PathLike = None,
                 rebuild_GRN: bool = False,
                 verbose: bool = True,
                 **kwargs):
        """
        A gene regulatory network object containing a gene list and a sparse network

        Parameters
        ----------
        name: str
            The name of this object, it will be used to name the saved GRN
        data: anndata.AnnData
            The adata used to construct this object
        GRN_file_dir: PathLike
            The dir path containing the saved GRN
        rebuild_GRN: bool
            To rebuild the GRN using the input data
        verbose: bool
            To print the messages during processing
        kwargs: dict
            The keyword arguments for rebuilding the GRN using pcnet reconstruction method
        """
        self.kws = kwargs
        if GRN_file_dir is not None:
            self._pc_net_file_name = (Path(GRN_file_dir) / Path(f"pcnet_{name}.npz"))
        # load pcnet

        if verbose:
            print(f'load GRN {name}')
        if GRN_file_dir is not None:
            li = data.var_names.values.tolist()  # index转换为列表，将字母全部改为大写
            s = [item.upper() for item in li]
            index_data = pd.Index(s)
            self._gene_names = index_data
            num_nodes = len(self._gene_names)
            adj_matrix = np.zeros((num_nodes, num_nodes))
            adj_matrix = pd.DataFrame(adj_matrix)
            adj_matrix.index = self._gene_names
            adj_matrix.columns = self._gene_names
            with open(Path(GRN_file_dir) / Path(f"{name}.tsv"), 'r') as file:
                next(file)  # 跳过第一行
                for line in file:
                    start_point, end_point, weight = line.strip().split('\t')
                    if start_point and end_point and weight:  # 检查是否有缺失值
                        # 执行对没有缺失值的行的操作
                        adj_matrix.loc[start_point.upper(), end_point.upper()] = abs(float(weight))
                        # 例如，可以在这里处理数据或者存储到新的数据结构中
                        pass
            adj_matrix = pd.DataFrame(np.maximum(adj_matrix.values, adj_matrix.transpose().values),
                                        index=adj_matrix.index, columns=adj_matrix.columns)
            self._net = sparse.csr_matrix(adj_matrix.values)


    @classmethod
    def from_sparse(cls, name, sparse_matrix, gene_names):
        """
        Build the GRN object from a sparse matrix and a list of gene names

        Parameters
        ----------
        name: str
        sparse_matrix
        gene_names

        Returns
        -------
        grn_obj: GRN
        """
        obj = cls(name)
        obj.set_value(sparse_matrix, gene_names)
        return obj

    @classmethod
    def load(cls, dir_name, pcnet_name):
        return cls(name=pcnet_name, GRN_file_dir=dir_name)

    @property
    def net(self) -> sparse.coo_matrix:
        return self._net

    @property
    def shape(self):
        return self._net.shape

    @property
    def gene_names(self):
        return self._gene_names

    def set_value(self, sparse_matrix: sparse.coo_matrix, gene_names):
        if sparse_matrix.shape[0] != sparse_matrix.shape[1]:
            raise ValueError("sparse_matrix should be a square sparse matrix"
                             f"({sparse_matrix.shape[0]} != {sparse_matrix.shape[1]})")
        if sparse_matrix.shape[0] != len(gene_names):
            raise ValueError(f"gene_names should have the same length as the sparse_matrix "
                             f"({sparse_matrix.shape[0]} != {len(gene_names)})")

        self._net = sparse_matrix
        self._gene_names = gene_names

    def set_rows_as(self, gene_names, value):   
        self._net[self._gene_names.isin(gene_names), :] = value

    def set_cols_as(self, gene_names, value):
        self._net[:, self._gene_names.isin(gene_names)] = value
    
    def set_rows_and_cols_as(self, gene_names, value):
        mask = self._gene_names.isin(gene_names)[:, None] @ self._gene_names.isin(gene_names)[None, :]
        self._net[mask] = value

    def copy(self):
        new_net = GRN()
        new_net.set_value(self._net.copy(), self._gene_names.copy())
        return new_net

    def save(self, dir_name):
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        sparse.save_npz(self._pc_net_file_name, self._net)
        pd.DataFrame({"gene_name": self._gene_names}).to_csv(Path(dir_name) / Path("gene_name.tsv"), sep='\t')

    def subset_in(self, values, copy=True):
        if copy:
            new_net = self.copy()
            new_net.subset_in(values=values, copy=False)
            return new_net

        bool_ind = self._gene_names.isin(values) # np.array
        self._net = self._net.tocsr()[bool_ind, :][:, bool_ind]
        self._gene_names = self._gene_names[bool_ind]

    def concat(self, grn, axis=0):
        if axis not in [0, 1]:
            raise ValueError("axis should be either 0 or 1.")
        concat_method = sparse.vstack if axis == 0 else sparse.hstack
        self._net = concat_method([self._net, grn.net])
        self._gene_names = pd.Index(pd.concat([self.gene_names.to_series(), grn.gene_names.to_series()]))


def concat_grns(grns, axis: int = 0):
    """
    Concatenate multiple GRNs along an axis

    Parameters
    ----------
    grns: A list of GRN
        An array or list of GRNs
    axis: int
        The axis to concatenate along (0: row, 1: col)

    Returns
    -------
    grn_obj: GRN

    """
    if axis not in [0, 1]:
        raise ValueError("axis should be either 0 or 1.")
    concat_method = sparse.vstack if axis == 0 else sparse.hstack
    obj = GRN()
    obj.set_value(concat_method([grn.net for grn in grns]),
                  gene_names=pd.Index(pd.concat([grn.gene_names.to_series() for grn in grns])))
    return obj


class scSDNE:
    def __init__(self,
                 data: anndata.AnnData,
                 source_celltype: str,
                 target_celltype: str,
                 obs_label: str,  # ident
                 GRN_file_dir: Union[str, PathLike] = None,
                 rebuild_GRN: bool = False,
                 query_DB: str = None,
                 alpha: float = 0.5,
                 mu: float = 1.,
                 scale_w: bool = True,
                 n_dim: int = 2,
                 verbose=True,
                 **kwargs):
        """
        The main object used to do analysis.

        Parameters
        ----------
        data: anndata.AnnData
            The data used to generate GRNs, manifold alignment results
        source_celltype:str
            The sender cell type
        target_celltype:str
            The receiver cell type
        obs_label: str
        GRN_file_dir
        rebuild_GRN
        query_DB
        alpha
        mu
        scale_w
        n_dim
        verbose
        """

        if query_DB is not None and query_DB not in ['comb', 'pairs']:
            raise ValueError('queryDB using the keyword None, \'comb\' or \'pairs\'')

        self._metrics = ["mean", "var"]
        self.verbose = verbose
        self.X = data.X
        self._cell_names = [source_celltype, target_celltype]
        self._cell_data_dic, self._cell_metric_dict = {}, {}
        self._genes = {}
        data.var_names = data.var_names.str.upper() # all species use upper case genes
        for name in self._cell_names:
            self.load_data(data, name, obs_label)

        import pkg_resources
        self._LRs = self._load_db_data(pkg_resources.resource_filename('scSDNE', 'database/LR.csv'),
                                       ['ligand', 'receptor'])

        # fill metrics
        self._LR_metrics = self.fill_metric()
        self._candidates = self._get_candidates(self._LR_metrics)

        self._net_A = GRN(name=self._cell_names[0],
                          data=self._cell_data_dic[self._cell_names[0]],
                          GRN_file_dir=GRN_file_dir,
                          rebuild_GRN=rebuild_GRN,
                          verbose=self.verbose,
                          **kwargs)
        self._net_B = GRN(name=self._cell_names[1],
                          data=self._cell_data_dic[self._cell_names[1]],
                          GRN_file_dir=GRN_file_dir,
                          rebuild_GRN=rebuild_GRN,
                          verbose=self.verbose,
                          **kwargs)
        if self.verbose:
            print("build correspondence and initiate a trainer")

        # cal w
        self._w, self.w12_shape, W, self.W12, N_A, N_B, self.geneX, self.geneY = self._build_w(alpha=alpha,
                                                query_DB=query_DB,
                                                scale_w=scale_w,
                                                mu=mu)
        dimension,_ = W.shape
        self._nn_trainer = SDNE(W, dimension, n_dim, layers=None)
        self._aligned_result = None
        if self.verbose:
            print("scSDNE init completed\n")

    @property
    def candidates(self):
        return self._candidates

    @property
    def trainer(self):
        return self._nn_trainer

    @property
    def w(self):
        return self._w

    @property
    def net_A(self):
        return self._net_A

    @property
    def net_B(self):
        return self._net_B

    @property
    def aligned_dist(self):
        if self._aligned_result is None:
            raise AttributeError("No aligned_dist created yet. "
                                 "Please call train_nn() to train the neural network to get embeddings first.")

        return self._aligned_result

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def Knk(self, ko_gene_list: Union[str, list]):
        if not isinstance(ko_gene_list, list):
            ko_gene_list = [ko_gene_list]
        gene_idx = pd.concat([self._genes[self._cell_names[0]].to_series(),
                              self._genes[self._cell_names[1]].to_series()])
        assert len(gene_idx) == self._net_A.net.shape[0] + self._net_B.net.shape[0]

        bool_idx = gene_idx.isin(ko_gene_list)
        self_knk = self.copy()
        self_knk._w = self_knk._w.tolil()
        self_knk._w[bool_idx, :] = 0
        self_knk._w[:, bool_idx] = 0
        self_knk._w = self_knk._w.tocoo()
        if self.verbose:
            print(f"remove edges and correspondence of gene {ko_gene_list}")
        return self_knk

    def load_data(self, data, cell_name, obs_label):
        if isinstance(data, anndata.AnnData):
            self._genes[cell_name] = data.var_names
            self._cell_data_dic[cell_name] = data[data.obs[obs_label] == cell_name, :]
            self._cell_metric_dict[cell_name] = {}
            self._cell_metric_dict[cell_name] = self._get_metric(self._cell_data_dic[cell_name], cell_name)

    def _load_db_data(self, file_path, subsets):
        df = pd.read_csv(file_path)
        df = df.loc[:, subsets] if subsets is not None else df
        return df

    def _get_metric(self, adata: ArrayView, name):  # require normalized data
        '''compute metrics for each gene'''
        data_norm = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X.copy()  # adata.layers['log1p']
        if self.verbose:
            print("(cell, feature):", data_norm.shape)
        if (data_norm % 1 != 0).any():  # check space: True for log (float), False for counts (int)
            mean = np.mean(data_norm, axis=0)  # .toarray()
            var = np.var(data_norm, axis=0)  # .toarray()
            return {"mean": dict(zip(self._genes[name], mean)),
                    "var": dict(zip(self._genes[name], var))} # , dispersion, cv
        raise ValueError("require log data")

    def fill_metric(self):
        val_df = pd.DataFrame()
        for m in self._metrics:
            val_df[f"{m}_L"] = self._LRs["ligand"].map(self._cell_metric_dict[self._cell_names[0]][m]).fillna(0.)
            val_df[f"{m}_R"] = self._LRs["receptor"].map(self._cell_metric_dict[self._cell_names[1]][m]).fillna(0.)
        df = pd.concat([self._LRs, val_df], axis=1)  # concat 1:1 since sharing same index
        df = df[(df['mean_L'] > 0) & (df['mean_R'] > 0)]  # filter 0 (none or zero expression) of LR
        if self.verbose:
            print(f"selected {df.shape[0]} LR pairs")

        return df

    def _get_candidates(self, df_filtered):
        '''selected L-R candidates'''
        df_filtered = df_filtered.drop_duplicates(['ligand', 'receptor'])
        candidates = [a + '_' + b for a, b in zip(np.asarray(df_filtered["ligand"], dtype=str),
                                                  np.asarray(df_filtered["receptor"], dtype=str))]
        return candidates

    @staticmethod
    def _zero_out_w(w, mask_lig, mask_rec):
        w = w.tolil()
        w[mask_lig, :] = 0
        w[:, mask_rec] = 0
        assert np.count_nonzero(w) == sum(mask_lig) * sum(mask_rec)
        return w.tocoo()

    @staticmethod
    def _build_metric_vec(dic, gene_names):
        return np.array([dic[g] if g in dic else np.nan for g in gene_names])

    def _build_w(self, alpha, query_DB=None, scale_w=True, mu: float = 1.): # -> (sparse.coo_matrix, (int, int)):
        '''build w: 3 modes, default None will not query the DB and use all pair-wise corresponding scores'''
        # (1-a)*u^2 + a*var
        ligand, receptor = self._cell_names[0], self._cell_names[1]

        metric_a_temp = self._build_metric_vec(dic=self._cell_metric_dict[ligand]["mean"],
                                                                        gene_names=self._genes[ligand])[:, None]
        metric_b_temp = self._build_metric_vec(dic=self._cell_metric_dict[receptor]["mean"],
                                                                        gene_names=self._genes[ligand])[:, None]
        #w12 = metric_a_temp @ metric_b_temp.T
        # w12 = np.sqrt(metric_a_temp @ metric_b_temp.T)
        mean_value = np.mean(self.X)
        metric = np.sqrt(metric_a_temp @ metric_b_temp.T)
        w12 = np.divide(metric, metric + mean_value)
        nonzero_indices = self._net_A.net.nonzero()
        sum_elements_A = 0
        for row, col, value in zip(nonzero_indices[0], nonzero_indices[1], self._net_A.net.data):
            element = w12[row, col]
            sum_elements_A += element
        nonzero_indices = self._net_B.net.nonzero()
        sum_elements_B = 0
        for row, col, value in zip(nonzero_indices[0], nonzero_indices[1], self._net_B.net.data):
            element = w12[row, col]
            sum_elements_B += element

        net_A, net_B = self._net_A.net.toarray(), self._net_B.net.toarray()

        mu = (net_A.sum() + net_B.sum()) / (sum_elements_A + sum_elements_B)
        net_A = net_A / mu
        net_B = net_B / mu
        del metric_a_temp
        del metric_b_temp
        if query_DB is not None:
            if query_DB == "comb":
                # ada.var index of LR genes (the intersect of DB and object genes, no pair relationship maintained)
                used_row_index = np.isin(self._genes[ligand], self._LRs["ligand"])
                used_col_index = np.isin(self._genes[receptor], self._LRs["receptor"])
            elif query_DB == "pairs":
                # maintain L-R pair relationship, both > 0
                selected_LR = self._LR_metrics[(self._LR_metrics[f"mean_L"] > 0) & (self._LR_metrics[f"mean_R"] > 0)]
                used_row_index = np.isin(self._genes[ligand], selected_LR["ligand"])
                used_col_index = np.isin(self._genes[receptor], selected_LR["receptor"])
            else:
                raise ValueError("queryDB must be: [None, \'comb\' or \'pairs\']")
            w12 = self._zero_out_w(w12, used_row_index, used_col_index)

        w = np.block([[net_A, w12], [w12.T, net_B]])
        gene_names_x = self._genes[self._cell_names[0]]
        gene_names_y = self._genes[self._cell_names[1]]
        return sparse.coo_matrix(w), w12.shape, w, w12, net_A, net_B, gene_names_x, gene_names_y

    def _get_data_arr(self):  # change the name (not always count data)
        '''return a list of counts in numpy array, gene by cell'''
        data_arr = [cell_data.X.T.toarray() if scipy.sparse.issparse(cell_data.X) else cell_data.X.T  # gene by cell
                    for _, cell_data in self._cell_data_dic.items()]
        return data_arr  # a list
    def get_embeds(self,
                 n_steps = 200,
                 lr = 0.001,
                 plot_losses: bool = False,
                 losses_file_name: str = None,
                 **optim_kwargs
                 ):
        projections = self._nn_trainer.train(n_steps=n_steps, lr=lr, **optim_kwargs)
        
        if plot_losses:
            self._nn_trainer.plot_losses(losses_file_name)

        n_samples = projections.shape[0]
        X1 = projections[:n_samples // 2, :]
        Y = projections[n_samples // 2:, :]
        dist = scipy.spatial.distance.cdist(X1, Y, metric='euclidean')
        dist_df = pd.DataFrame(dist, index=self.geneX, columns=self.geneY)
        dist_df = pd.DataFrame(dist_df.stack())  # multi_index, colname 0 for dist
        dist_df = dist_df.rename_axis([1, 2]).reset_index(level=[1, 2])
        dist_df.columns = ['ligand', 'receptor', 'dist']
        dist_df.index = dist_df['ligand'] + '_' + dist_df['receptor']
        dist_df['correspondence'] = self.W12.reshape((self.W12.size,))
        self._aligned_result = dist_df
        return projections
    
    def plot_losses(self, **kwargs):
        self._nn_trainer.plot_losses(**kwargs)

    def null_test(self,
                  filter_zeros: bool = True,
                  pval=0.05,
                  plot_result=False):
        return null_test(self._aligned_result, self._candidates,
                         filter_zeros=filter_zeros,
                         pval=pval,
                         plot=plot_result)
    
def main(args):
    # workpath = Path.joinpath(Path(__file__).parent.parent, 'tutorials/data')
    from time import time
    if args.eva:
        adata = sc.datasets.pbmc3k()
        adata = adata[
            np.random.choice(adata.shape[0], args.n_sample, replace=False), 
            np.random.choice(adata.shape[1], args.n_feature, replace=False)].copy()
        adata.obs["ident"] = ["cell_A"] * (len(adata)//2) + ["cell_B"] * (args.n_sample-len(adata)//2)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X 
    else:
        from .dataLoader import build_adata
        adata = build_adata(counts_path = args.file)
        print(adata)

    SDNE = scSDNE(data = adata,
                        source_celltype = args.sender,
                        target_celltype = args.receiver,
                        obs_label = args.label,
                        rebuild_GRN = args.rebuild, # timer
                        GRN_file_dir = args.workdir,  
                        verbose = args.verbose,
                        n_cpus = args.n_cpus)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type = str)
    parser.add_argument('-w', '--workdir', type = str, default = 'xct_results')
    parser.add_argument('-o', '--output', type = str, default = 'xct_enriched')
    parser.add_argument('-s', '--sender', type = str, default = 'cell_A')
    parser.add_argument('-r', '--receiver', type = str, default = 'cell_B')
    parser.add_argument('-l', '--label', type = str, default = 'ident')
    parser.add_argument('--n_cpus', type = int, default = -1)
    parser.add_argument('-v', '--verbose', action = 'store_true')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--rebuild', dest = 'rebuild', action = 'store_true')
    feature_parser.add_argument('--no-rebuild', dest = 'rebuild', action ='store_false')
    parser.set_defaults(rebuild = True)

    parser.add_argument('--eva', action = 'store_true')
    parser.add_argument('--n_sample', type = int, default = 100)
    parser.add_argument('--n_feature', type = int, default = 3000)
    
    args = parser.parse_args()
    main(args)


