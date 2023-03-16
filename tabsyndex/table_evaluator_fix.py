from pathlib import Path
from table_evaluator import TableEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from dython.nominal import compute_associations

# in some cases, sns bins="auto" function takes forever to plot histograms for non-categorical data.
# in that case, use the "sturges" bins function (see plot_distributions())
NO_AUTO_BIN =["capital-gain", "capital-loss"]

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

        plt.show(block=False)
    
    def plot_pca(self, fname=None, compares_with_train=False):
        """
        Plot the first two components of a PCA of real and fake data.
        :param fname: If not none, saves the plot with this file name.
        """
        real, fake = self.convert_numerical()

        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('First two components of PCA', fontsize=16)
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        if not compares_with_train:
            ax[0].set_title('Real data')
            ax[1].set_title('Synthetic data')
        else:
            ax[0].set_title('Real Test data')
            ax[1].set_title('Real Train data')

        if fname is not None: 
            plt.savefig(fname)
        plt.show(block=False)

    def plot_mean_std(self, fname=None):
        """
        Class wrapper function for plotting the mean and std using `viz.plot_mean_std`.
        :param fname: If not none, saves the plot with this file name. 
        """
        plot_mean_std(self.real, self.fake, fname=fname)


    def visual_evaluation(self, save_dir=None, **kwargs):
        """
        Plot all visual evaluation metrics. Includes plotting the mean and standard deviation, cumulative sums, correlation differences and the PCA transform.
        :save_dir: directory path to save images 
        :param kwargs: any kwargs for matplotlib.
        """
        if save_dir is None: 
            self.plot_mean_std()
            self.plot_cumsums()
            self.plot_distributions()
            self.plot_correlation_difference(**kwargs)
            self.plot_pca()    
        else: 
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            print("Mean_std")
            self.plot_mean_std(fname=save_dir/'mean_std.png')
            print("Cumsums")
            self.plot_cumsums(fname=save_dir/'cumsums.png')
            print("Distributions")
            self.plot_distributions(fname=save_dir/'distributions.png')
            print("Corr")
            self.plot_correlation_difference(fname=save_dir/'correlation_difference.png', **kwargs)
            print("PCA")
            self.plot_pca(fname=save_dir/'pca.png', compares_with_train="real" in str(save_dir)) 
    
    def plot_distributions(self, nr_cols=3, fname=None):
        """
        Plot the distribution plots for all columns in the real and fake dataset. Height of each row of plots scales with the length of the labels. Each plot
        contains the values of a real columns and the corresponding fake column.
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
        fig.suptitle('Distribution per feature', fontsize=16)
        axes = ax.flatten()
        handles, labels = None, None
        for i, col in enumerate(self.real.columns):
            if col not in self.categorical_columns:
                plot_df = pd.DataFrame({col: pd.concat([self.real[col], self.fake[col]]), 'kind': ['Synthetic'] * self.n_samples +  ['Real'] * self.n_samples})
                
                bins = "auto" if col not in NO_AUTO_BIN else "sturges"
                fig = sns.histplot(plot_df, bins = bins, x=col, hue='kind',ax=axes[i], stat='probability', legend=True, kde=False)
                axes[i].set_autoscaley_on(True)
                if handles is None:
                    axes[i].legend(['Real', 'Synthetic'], fancybox=False, title="kind")
                else:
                    axes[i].legend(handles, labels, title="kind")

            else:
                real = self.real.copy()
                fake = self.fake.copy()
                real['kind'] = 'Real'
                fake['kind'] = 'Synthetic'
                concat = pd.concat([fake, real])
                palette = sns.color_palette(
                    [(0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                     (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
                x, y, hue = col, "proportion", "kind"
                ax = (concat[x]
                      .groupby(concat[hue])
                      .value_counts(normalize=True)
                      .rename(y)
                      .reset_index()
                      .pipe((sns.barplot, "data"), x=x, y=y, hue=hue, ax=axes[i], saturation=0.8, palette=palette))
                ax.set_xticklabels(axes[i].get_xticklabels(), rotation='vertical')
                handles, labels = axes[i].get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        if fname is not None: 
            plt.savefig(fname)
        plt.show(block=False)

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

        titles = ['Real', 'Synthetic', 'Difference'] if plot_diff else ['Real', 'Synthetic']
        for i, label in enumerate(titles):
            title_font = {'size': '18'}
            ax[i].set_title(label, **title_font)
        plt.tight_layout()

        if fname is not None: 
            plt.savefig(fname)

        plt.show(block=False)

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
    ax.plot(x2, y, marker='o', linestyle='none', label='Synthetic', alpha=0.5)
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
        plt.show(block=False)


def plot_mean_std(real: pd.DataFrame, fake: pd.DataFrame, ax=None, fname=None):
    """
    Plot the means and standard deviations of each dataset.

    :param real: DataFrame containing the real data
    :param fake: DataFrame containing the fake data
    :param ax: Axis to plot on. If none, a new figure is made.
    :param fname: If not none, saves the plot with this file name. 
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Absolute Log Mean and STDs of numeric data\n', fontsize=16)

    ax[0].grid(True)
    ax[1].grid(True)
    real = real._get_numeric_data()
    fake = fake._get_numeric_data()
    real_mean = np.log(np.add(abs(real.mean()).values, 1e-5))
    fake_mean = np.log(np.add(abs(fake.mean()).values, 1e-5))
    min_mean = min(real_mean) - 1
    max_mean = max(real_mean) + 1
    line = np.arange(min_mean, max_mean)
    sns.lineplot(x=line, y=line, ax=ax[0])
    sns.scatterplot(x=real_mean,
                    y=fake_mean,
                    ax=ax[0])
    ax[0].set_title('Means of real and synthetic data')
    ax[0].set_xlabel('real data mean (log)')
    ax[0].set_ylabel('synthetic data mean (log)')

    real_std = np.log(np.add(real.std().values, 1e-5))
    fake_std = np.log(np.add(fake.std().values, 1e-5))
    min_std = min(real_std) - 1
    max_std = max(real_std) + 1
    line = np.arange(min_std, max_std)
    sns.lineplot(x=line, y=line, ax=ax[1])
    sns.scatterplot(x=real_std,
                    y=fake_std,
                    ax=ax[1])
    ax[1].set_title('Stds of real and synthetic data')
    ax[1].set_xlabel('real data std (log)')
    ax[1].set_ylabel('synthetic data std (log)')

    if fname is not None:
        plt.savefig(fname)

    if ax is None:
        plt.show(block=False)