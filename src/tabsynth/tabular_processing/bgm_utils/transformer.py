'''
Credits: 
Processing is based upon the work of the authors of:
https://github.com/Team-TUD/CTAB-GAN-Plus
https://github.com/Team-TUD/CTAB-GAN

No changes have been made to the code below.
Only documentation has been added.
'''

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture


class DataTransformer:
    """
    A class to transform and inverse transform data for machine learning applications.

    Parameters
    ----------
    meta : list of dicts
        metadata for each column of the input data
    model : list
        trained models for each non-categorical column
    components : list of np.ndarray
        component information for each non-categorical column
    ordering : list of np.ndarray
        ordering information for each mixed column
    non_categorical_columns : list of int
        column indices of non-categorical columns
    general_columns : list of int
        column indices of general columns
    n_clusters : int
        number of clusters in the trained models for each non-categorical column

    Attributes
    ----------
    reorder_info : list of tuples
        information on how the data was split into categorical and numerical parts
    cat_style : str
        the style used to encode categorical variables
    x_cat : np.ndarray
        categorical part of the data
    x_num : np.ndarray
        numerical part of the data

    Methods
    -------
    inverse_transform(data)
        Inverse transforms the given data to the original input space.
        Parameters:
            data : np.ndarray
                the transformed data to be inverse transformed
        Returns:
            np.ndarray
                the inverse transformed data
            int
                the number of invalid rows in the input data
    split_cat_num(data, cat_style="one-hot")
        Splits the input data into categorical and numerical parts.
        Parameters:
            data : np.ndarray
                the data to be split
            cat_style : str, optional
                the style used to encode categorical variables (default is "one-hot")
        Returns:
            np.ndarray
                the categorical part of the data
            np.ndarray
                the numerical part of the data
    """
    def __init__(self, train_data=pd.DataFrame, categorical_list=[], mixed_dict={}, general_list=[],
                 non_categorical_list=[], n_clusters=10, eps=0.005):
        np.random.seed(42)
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        self.train_data = train_data
        self.categorical_columns = categorical_list
        self.mixed_columns = mixed_dict
        self.general_columns = general_list
        self.non_categorical_columns = non_categorical_list
        self.description_long = ""
        self.description_list = []
        

    def get_metadata(self):

        meta = []
        self.description_long = ""
        self.description_list = []
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:, index]
            if index in self.categorical_columns:
                if index in self.non_categorical_columns:
                    meta.append({
                        "name": index,
                        "name_str": self.train_data.columns[index],
                        "type": "continuous",
                        "min": column.min(),
                        "max": column.max(),
                    })
                else:
                    mapper = column.value_counts().index.tolist()
                    meta.append({
                        "name": index,
                        "name_str": self.train_data.columns[index],
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper
                    })

            elif index in self.mixed_columns.keys():
                meta.append({
                    "name": index,
                    "name_str": self.train_data.columns[index],
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index]
                })
            else:
                meta.append({
                    "name": index,
                    "name_str": self.train_data.columns[index],
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })
        return meta

    def fit(self):
        data = self.train_data.values
        self.meta = self.get_metadata()
        model = []
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                if id_ not in self.general_columns:
                    gm = BayesianGaussianMixture(
                        n_components=self.n_clusters,
                        weight_concentration_prior_type='dirichlet_process',
                        weight_concentration_prior=0.001,
                        max_iter=100, n_init=1, random_state=42)
                    gm.fit(data[:, id_].reshape([-1, 1]))
                    mode_freq = (pd.Series(gm.predict(data[:, id_].reshape([-1, 1]))).value_counts().keys())
                    model.append(gm)
                    old_comp = gm.weights_ > self.eps
                    comp = []
                    for i in range(self.n_clusters):
                        if (i in (mode_freq)) & old_comp[i]:
                            comp.append(True)
                        else:
                            comp.append(False)
                    self.components.append(comp)
                    self.output_info += [(1, 'tanh', 'no_g', info['name_str'], info["type"]), (np.sum(comp), 'softmax', info['name_str'], info["type"])]
                    self.output_dim += 1 + np.sum(comp)
                else:
                    model.append(None)
                    self.components.append(None)
                    self.output_info += [(1, 'tanh', 'yes_g',info['name_str'], info["type"])]
                    self.output_dim += 1

            elif info['type'] == "mixed":

                gm1 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001, max_iter=100,
                    n_init=1, random_state=42)
                gm2 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001, max_iter=100,
                    n_init=1, random_state=42)

                gm1.fit(data[:, id_].reshape([-1, 1]))

                filter_arr = []
                for element in data[:, id_]:
                    if element not in info['modal']:
                        filter_arr.append(True)
                    else:
                        filter_arr.append(False)

                gm2.fit(data[:, id_][filter_arr].reshape([-1, 1]))
                mode_freq = (pd.Series(gm2.predict(data[:, id_][filter_arr].reshape([-1, 1]))).value_counts().keys())
                self.filter_arr.append(filter_arr)
                model.append((gm1, gm2))

                old_comp = gm2.weights_ > self.eps

                comp = []

                for i in range(self.n_clusters):
                    if (i in (mode_freq)) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)

                self.components.append(comp)

                self.output_info += [(1, 'tanh', "no_g", info['name_str'], info["type"]), (np.sum(comp) + len(info['modal']), 'softmax', info['name_str'], info["type"])]
                self.output_dim += 1 + np.sum(comp) + len(info['modal'])
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info['size'], 'softmax', info['name_str'], info["type"])]
                self.output_dim += info['size']
            description = f"Column number {info['name']} is " \
                          f"{info['name_str']}," \
                          f" of type {info['type']}, " \
                          f"and has length {self.output_info[-1][0]}"
            self.description_long += description + "\n"
            self.description_list.append(description)


        self.model = model

    def transform(self, data, ispositive=False, positive_list=None):
        values = []
        mixed_counter = 0
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == "continuous":
                if id_ not in self.general_columns:
                    current = current.reshape([-1, 1])
                    means = self.model[id_].means_.reshape((1, self.n_clusters))
                    stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                    features = np.empty(shape=(len(current), self.n_clusters))
                    if ispositive == True:
                        if id_ in positive_list:
                            features = np.abs(current - means) / (4 * stds)
                    else:
                        features = (current - means) / (4 * stds)

                    probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                    n_opts = sum(self.components[id_])
                    features = features[:, self.components[id_]]
                    probs = probs[:, self.components[id_]]

                    opt_sel = np.zeros(len(data), dtype='int')
                    for i in range(len(data)):
                        pp = probs[i] + 1e-6
                        pp = pp / sum(pp)
                        opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                    idx = np.arange((len(features)))
                    features = features[idx, opt_sel].reshape([-1, 1])
                    features = np.clip(features, -.99, .99)
                    probs_onehot = np.zeros_like(probs)
                    probs_onehot[np.arange(len(probs)), opt_sel] = 1

                    re_ordered_phot = np.zeros_like(probs_onehot)

                    col_sums = probs_onehot.sum(axis=0)

                    n = probs_onehot.shape[1]
                    largest_indices = np.argsort(-1 * col_sums)[:n]
                    self.ordering.append(largest_indices)
                    for id, val in enumerate(largest_indices):
                        re_ordered_phot[:, id] = probs_onehot[:, val]

                    values += [features, re_ordered_phot]

                else:

                    self.ordering.append(None)

                    if id_ in self.non_categorical_columns:
                        info['min'] = -1e-3
                        info['max'] = info['max'] + 1e-3

                    current = (current - (info['min'])) / (info['max'] - info['min'])
                    current = current * 2 - 1
                    current = current.reshape([-1, 1])
                    values.append(current)

            elif info['type'] == "mixed":

                means_0 = self.model[id_][0].means_.reshape([-1])
                stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])

                zero_std_list = []
                means_needed = []
                stds_needed = []

                for mode in info['modal']:
                    if mode != -9999999:
                        dist = []
                        for idx, val in enumerate(list(means_0.flatten())):
                            dist.append(abs(mode - val))
                        index_min = np.argmin(np.array(dist))
                        zero_std_list.append(index_min)
                    else:
                        continue

                for idx in zero_std_list:
                    means_needed.append(means_0[idx])
                    stds_needed.append(stds_0[idx])

                mode_vals = []

                for i, j, k in zip(info['modal'], means_needed, stds_needed):
                    this_val = np.abs(i - j) / (4 * k)
                    mode_vals.append(this_val)

                if -9999999 in info["modal"]:
                    mode_vals.append(0)

                # ADDED --> calc filter array everytime new so transform works also for not fitted data
                _filter_arr = []
                for element in current:
                    if element not in info['modal']:
                        _filter_arr.append(True)
                    else:
                        _filter_arr.append(False)
                # ADDED END

                current = current.reshape([-1, 1])
                filter_arr = self.filter_arr[mixed_counter]
                current = current[_filter_arr] # CHANGED filter_arr to _filter_arr

                means = self.model[id_][1].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_][1].covariances_).reshape((1, self.n_clusters))
                features = np.empty(shape=(len(current), self.n_clusters))
                if ispositive == True:
                    if id_ in positive_list:
                        features = np.abs(current - means) / (4 * stds)
                else:
                    features = (current - means) / (4 * stds)

                probs = self.model[id_][1].predict_proba(current.reshape([-1, 1]))

                n_opts = sum(self.components[id_])  # 8
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(current), dtype='int')
                for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                extra_bits = np.zeros([len(current), len(info['modal'])])
                temp_probs_onehot = np.concatenate([extra_bits, probs_onehot], axis=1)
                final = np.zeros([len(data), 1 + probs_onehot.shape[1] + len(info['modal'])])
                features_curser = 0
                for idx, val in enumerate(data[:, id_]):
                    if val in info['modal']:
                        category_ = list(map(info['modal'].index, [val]))[0]
                        final[idx, 0] = mode_vals[category_]
                        final[idx, (category_ + 1)] = 1

                    else:
                        final[idx, 0] = features[features_curser]
                        final[idx, (1 + len(info['modal'])):] = temp_probs_onehot[features_curser][len(info['modal']):]
                        features_curser = features_curser + 1

                just_onehot = final[:, 1:]
                re_ordered_jhot = np.zeros_like(just_onehot)
                n = just_onehot.shape[1]
                col_sums = just_onehot.sum(axis=0)
                largest_indices = np.argsort(-1 * col_sums)[:n]
                self.ordering.append(largest_indices)
                for id, val in enumerate(largest_indices):
                    re_ordered_jhot[:, id] = just_onehot[:, val]
                final_features = final[:, 0].reshape([-1, 1])
                values += [final_features, re_ordered_jhot]
                mixed_counter = mixed_counter + 1

            else:
                self.ordering.append(None)
                col_t = np.zeros([len(data), info['size']])
                idx = list(map(info['i2s'].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])
        invalid_ids = []
        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                if id_ not in self.general_columns:
                    u = data[:, st]
                    v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]
                    order = self.ordering[id_]
                    v_re_ordered = np.zeros_like(v)

                    for id, val in enumerate(order):
                        v_re_ordered[:, val] = v[:, id]

                    v = v_re_ordered

                    u = np.clip(u, -1, 1)
                    v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                    v_t[:, self.components[id_]] = v
                    v = v_t
                    st += 1 + np.sum(self.components[id_])
                    means = self.model[id_].means_.reshape([-1])
                    stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                    p_argmax = np.argmax(v, axis=1)
                    std_t = stds[p_argmax]
                    mean_t = means[p_argmax]
                    tmp = u * 4 * std_t + mean_t

                    for idx, val in enumerate(tmp):
                        if (val < info["min"]) | (val > info['max']):
                            invalid_ids.append(idx)

                    if id_ in self.non_categorical_columns:
                        tmp = np.round(tmp)

                    data_t[:, id_] = tmp

                else:
                    u = data[:, st]
                    u = (u + 1) / 2
                    u = np.clip(u, 0, 1)
                    u = u * (info['max'] - info['min']) + info['min']
                    if id_ in self.non_categorical_columns:
                        data_t[:, id_] = np.round(u)
                    else:
                        data_t[:, id_] = u

                    st += 1

            elif info['type'] == "mixed":

                u = data[:, st]
                full_v = data[:, (st + 1):(st + 1) + len(info['modal']) + np.sum(self.components[id_])]
                order = self.ordering[id_]
                full_v_re_ordered = np.zeros_like(full_v)

                for id, val in enumerate(order):
                    full_v_re_ordered[:, val] = full_v[:, id]

                full_v = full_v_re_ordered

                mixed_v = full_v[:, :len(info['modal'])]
                v = full_v[:, -np.sum(self.components[id_]):]

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = np.concatenate([mixed_v, v_t], axis=1)

                st += 1 + np.sum(self.components[id_]) + len(info['modal'])
                means = self.model[id_][1].means_.reshape([-1])
                stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)

                result = np.zeros_like(u)

                for idx in range(len(data)):
                    if p_argmax[idx] < len(info['modal']):
                        argmax_value = p_argmax[idx]
                        result[idx] = float(list(map(info['modal'].__getitem__, [argmax_value]))[0])
                    else:
                        std_t = stds[(p_argmax[idx] - len(info['modal']))]
                        mean_t = means[(p_argmax[idx] - len(info['modal']))]
                        result[idx] = u[idx] * 4 * std_t + mean_t

                for idx, val in enumerate(result):
                    if (val < info["min"]) | (val > info['max']):
                        invalid_ids.append(idx)

                data_t[:, id_] = result

            else:
                current = data[:, st:st + info['size']]
                st += info['size']
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))

        invalid_ids = np.unique(np.array(invalid_ids))
        all_ids = np.arange(0, len(data))
        valid_ids = list(set(all_ids) - set(invalid_ids))

        # return data_t[valid_ids], len(invalid_ids)
        return data_t, len(invalid_ids)

    def split_cat_num(self, data, cat_style="one-hot"):
        assert cat_style in ["one-hot", "labels"]
        self.reorder_info = []
        cat = []
        num = []
        self.cat_style = cat_style
        if cat_style == "one-hot":
            cat_function = lambda x, **kwargs: x 
        elif cat_style == 'labels':
            cat_function = lambda x: np.argmax(x, axis=1)
        
        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                if id_ not in self.general_columns:
                    u = data[:, st]
                    v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]
                    num.append(u)
                    self.reorder_info.append(("num",1, 1)) # numerical columns always have 1 component
                    len_v = v.shape[1]
                    v=cat_function(v)
                    v = v.reshape(-1, 1) if len(v.shape) == 1 else v
                    cat.append(v)
                    self.reorder_info.append(("cat",v.shape[1], len_v))
                    st += 1 + np.sum(self.components[id_])

                else:
                    u = data[:, st]
                    num.append(u)
                    self.reorder_info.append(("num",1, 1))
                    st += 1

            elif info['type'] == "mixed":
                u = data[:, st]
                v = data[:, (st + 1):(st + 1) + len(info['modal']) + np.sum(self.components[id_])]
                num.append(u)
                self.reorder_info.append(("num",1, 1))
                len_v = v.shape[1]
                v=cat_function(v)
                v = v.reshape(-1, 1) if len(v.shape) == 1 else v
                cat.append(v)
                self.reorder_info.append(("cat",v.shape[1], len_v))

                st += 1 + np.sum(self.components[id_]) + len(info['modal'])

            else:
                v = data[:, st:st + info['size']]
                len_v = v.shape[1]
                v=cat_function(v)
                v = v.reshape(-1, 1) if len(v.shape) == 1 else v
                cat.append(v)
                self.reorder_info.append(("cat", v.shape[1], len_v))
                st += info['size']

        self.x_cat = np.concatenate(cat, axis=-1)
        num = [np.expand_dims(x, axis=-1) for x in num]
        self.x_num = np.concatenate(num, axis=-1)
        return self.x_cat, self.x_num
    

    def inverse_split_cat_num(self, x_cat, x_num):
        assert self.reorder_info is not None and len(self.reorder_info) > 0, "data has not been split yet"
        assert self.cat_style is not None, "data has not been split yet"


        total = []
        if self.cat_style == "one-hot":
            cat_function = lambda x, **kwargs: x 
        elif self.cat_style == 'labels':
            cat_function = lambda x, length: np.eye(int(length))[x.squeeze().astype(int)] # x needs to be oh shape (n,) #+1
        for info, st, length in self.reorder_info:
            if info == "num":
                total.append(x_num[:, :st]) # needs to be [:,:st] not [:, st] because we need (n,1) and not (n,)
                x_num = np.delete(x_num, range(st), axis=1)
            else:
                total.append(cat_function(x_cat[:, :st], length))
                x_cat = np.delete(x_cat, range(st), axis=1)
        return np.concatenate(total, axis=-1)


class ImageTransformer:

    def __init__(self, side):
        self.height = side

    def transform(self, data, padding="zero"):
        if padding not in ["zero", "same"]:
            raise ValueError("Padding must be 'same' or 'zero'")
        if padding == "zero":
            if self.height * self.height > len(data[0]):
                padding = torch.zeros((len(data), self.height * self.height - len(data[0]))).to(data.device)
                data = torch.cat([data, padding], axis=1)
            return data.view(-1, 1, self.height, self.height)
        if padding == "same":
            while self.height * self.height > len(data[0]):
                data = torch.cat([data, data], axis=1)
            # cut of the end:
            data = data[:, :self.height * self.height]
            data = data.view(-1, 1, self.height, self.height)
            return data


    def inverse_transform(self, data):
        data = data.view(-1, self.height * self.height)

        return data
