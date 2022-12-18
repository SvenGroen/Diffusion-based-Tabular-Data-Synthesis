from pathlib import Path
from table_evaluator import TableEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dython.nominal import compute_associations

class TableEvaluatorFix(TableEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_correlation_difference(self, plot_diff=True, fname=None, **kwargs):
        """
        Plot the association matrices for each table and, if chosen, the difference between them.

        :param plot_diff: whether to plot the difference
        :param fname: If not none, saves the plot with this file name.
        :param kwargs: kwargs for sns.heatmap
        """
        plot_correlation_difference(self.real, self.fake, cat_cols=self.categorical_columns, plot_diff=plot_diff, fname=fname,
                                    **kwargs)


    
    def visual_evaluation(self, save_dir=None, **kwargs):
        """
        Plot all visual evaluation metrics. Includes plotting the mean and standard deviation, cumulative sums, correlation differences and the PCA transform.
        :save_dir: directory path to save images 
        :param kwargs: any kwargs for matplotlib.
        """
        if save_dir is None: 
            super().plot_mean_std()
            super().plot_cumsums()
            super().plot_distributions()
            self.plot_correlation_difference(**kwargs)
            super().plot_pca()    
        else: 
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            super().plot_mean_std(fname=save_dir/'mean_std.png')
            super().plot_cumsums(fname=save_dir/'cumsums.png')
            super().plot_distributions(fname=save_dir/'distributions.png')
            self.plot_correlation_difference(fname=save_dir/'correlation_difference.png', **kwargs)
            super().plot_pca(fname=save_dir/'pca.png') 

def plot_correlation_difference(real: pd.DataFrame, fake: pd.DataFrame, plot_diff: bool = True, cat_cols: list = None, annot=False, fname=None):
        """
        Plot the association matrices for the `real` dataframe, `fake` dataframe and plot the difference between them. Has support for continuous and Categorical
        (Male, Female) data types. All Object and Category dtypes are considered to be Categorical columns if `dis_cols` is not passed.

        - Continuous - Continuous: Uses Pearson's correlation coefficient
        - Continuous - Categorical: Uses so called correlation ratio (https://en.wikipedia.org/wiki/Correlation_ratio) for both continuous - categorical and categorical - continuous.
        - Categorical - Categorical: Uses Theil's U, an asymmetric correlation metric for Categorical associations

        :param real: DataFrame with real data
        :param fake: DataFrame with synthetic data
        :param plot_diff: Plot difference if True, else not
        :param cat_cols: List of Categorical columns
        :param boolean annot: Whether to annotate the plot with numbers indicating the associations.
        """
        assert isinstance(real, pd.DataFrame), f'`real` parameters must be a Pandas DataFrame'
        assert isinstance(fake, pd.DataFrame), f'`fake` parameters must be a Pandas DataFrame'
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        if cat_cols is None:
            cat_cols = real.select_dtypes(['object', 'category'])
        if plot_diff:
            fig, ax = plt.subplots(1, 3, figsize=(24, 7))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        
        # cols = ["age", "education", "occupation","capital-gain", "capital-loss", "hours-per-week","income"]
        # real=real[cols]
        # fake=fake[cols]
        # cat_cols = [col for col in cat_cols if col in real.columns]
        # compute the associations
        real_corr = compute_associations(real, nominal_columns=cat_cols, theil_u=True) 
        fake_corr = compute_associations(fake, nominal_columns=cat_cols, theil_u=True)
        # fake_corr = associations(fake, nominal_columns=cat_cols, plot=False, theil_u=True,
        #                          mark_columns=True, annot=annot, ax=ax[1], cmap=cmap)['corr']
        # convert to float to avoid issues with the heatmap
        real_corr = real_corr.astype(float)
        fake_corr = fake_corr.astype(float)
        cols = ["age", "education", "occupation","capital-gain", "capital-loss", "hours-per-week","income"]
        
        # add real corr to ax[0] and fake corr to ax[1]
        sns.heatmap(real_corr, ax=ax[0], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')
        sns.heatmap(fake_corr, ax=ax[1], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

        if plot_diff:
            diff = abs(real_corr - fake_corr)
            sns.set(style="white")
            sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,
                        linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

        titles = ['Real', 'Fake', 'Difference'] if plot_diff else ['Real', 'Fake']
        for i, label in enumerate(titles):
            title_font = {'size': '18'}
            ax[i].set_title(label, **title_font)
        plt.tight_layout()

        if fname is not None: 
            plt.savefig(fname)

        plt.show()
