from pathlib import Path
from table_evaluator import TableEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

    def plot_cumsums(self, nr_cols=4, fname=None):
        """
        Plot the cumulative sums for all columns in the real and fake dataset. Height of each row scales with the length of the labels. Each plot contains the
        values of a real columns and the corresponding fake column.
        :param fname: If not none, saves the plot with this file name. 
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            r = self.real[col]
            f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
            cdf(r, f, col, 'Cumsum', ax=axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if fname is not None: 
            plt.savefig(fname)

        plt.show()
    
    def visual_evaluation(self, save_dir=None, **kwargs):
        """
        Plot all visual evaluation metrics. Includes plotting the mean and standard deviation, cumulative sums, correlation differences and the PCA transform.
        :save_dir: directory path to save images 
        :param kwargs: any kwargs for matplotlib.
        """
        if save_dir is None: 
            super().plot_mean_std()
            self.plot_cumsums()
            super().plot_distributions()
            self.plot_correlation_difference(**kwargs)
            super().plot_pca()    
        else: 
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            super().plot_mean_std(fname=save_dir/'mean_std.png')
            self.plot_cumsums(fname=save_dir/'cumsums.png')
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

def cdf(data_r, data_f, xlabel: str = 'Values', ylabel: str = 'Cumulative Sum', ax=None):
    """
    Plot continous density function on optionally given ax. If no ax, cdf is plotted and shown.

    :param data_r: Series with real data
    :param data_f: Series with fake data
    :param xlabel: Label to put on the x-axis
    :param ylabel: Label to put on the y-axis
    :param ax: The axis to plot on. If ax=None, a new figure is created.
    """
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    ax = ax if ax else plt.subplots()[1]

    axis_font = {'size': '14'}
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)

    ax.grid()
    ax.plot(x1, y, marker='o', linestyle='none', label='Real', ms=8)
    ax.plot(x2, y, marker='o', linestyle='none', label='Fake', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    import matplotlib.ticker as mticker

    # If labels are strings, rotate them vertical
    if isinstance(data_r, pd.Series) and data_r.dtypes == 'object':
        ticks_loc = ax.get_xticks()
        ticks_loc = ticks_loc[:len(data_r.sort_values().unique())]# MANUAL FIX --> not in original library
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(data_r.sort_values().unique(), rotation='vertical')

    if ax is None:
        plt.show()
