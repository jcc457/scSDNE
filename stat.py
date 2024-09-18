import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def null_test(df_nn: pd.DataFrame, 
            candidates, 
            filter_zeros=True, 
            pval=0.05, 
            plot=False):
    '''nonparametric left tail test to have enriched pairs'''
    if ('dist' or 'correspondence') not in df_nn.columns:
        raise IndexError('require resulted dataframe with column \'dist\' and \'correspondence\'')

    else:
        dist_test = df_nn[df_nn.index.isin(candidates)].copy()
        # filter pairs with correspondence_score zero
        if filter_zeros:
            mask = df_nn['correspondence'] != 0
        else:
            mask = np.ones(len(df_nn), dtype=bool)
        dist_null = df_nn[(~df_nn.index.isin(candidates)) & (mask)]
        dist_test['p_val'] = dist_test['dist'].apply(
            lambda x: scipy.stats.percentileofscore(dist_null['dist'], x) / 100)
        df_enriched = dist_test[dist_test['p_val'] < pval].sort_values(by=['dist'])
        print(f'\nTotal enriched: {len(df_enriched)} / {len(df_nn)}')
        df_enriched['enriched_rank'] = np.arange(len(df_enriched)) + 1

        if plot:
            cut = np.percentile(dist_null['dist'].values, pval)  # left tail
            plt.hist(dist_null['dist'], bins=1000, color='royalblue')
            for d in dist_test['dist']:
                if d < cut:
                    c = 'red'
                else:
                    c = 'gray'
                plt.axvline(d, ls=(0, (1, 1)), linewidth=0.5, alpha=0.8, c=c)
            plt.xlabel('distance')
            plt.show()
        del dist_test
    return df_enriched
