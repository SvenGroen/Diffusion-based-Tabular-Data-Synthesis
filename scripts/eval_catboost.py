from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report, r2_score
import numpy as np
import os
from sklearn.utils import shuffle
import zero
from pathlib import Path
import lib
from pprint import pprint
from lib import concat_features, read_pure_data, get_catboost_config, read_changed_val

def train_catboost(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed = 0,
    params = None,
    change_val = True,
    device = None # dummy
):
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    info = lib.load_json(os.path.join(real_data_path, 'info.json'))
    T = lib.Transformations(**T_dict)
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(real_data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)

        ###
        # dists = privacy_metrics(real_data_path, synthetic_data_path)
        # bad_fakes = dists.argsort()[:int(0.25 * len(y_fake))]
        # X_num_fake = np.delete(X_num_fake, bad_fakes, axis=0)
        # X_cat_fake = np.delete(X_cat_fake, bad_fakes, axis=0) if X_cat_fake is not None else None
        # y_fake = np.delete(y_fake, bad_fakes, axis=0)
        ###

        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {parent_dir}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path)

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(real_data_path)
    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test')

    D = lib.Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        lib.TaskType(info['task_type']),
        info.get('n_classes')
    )

    D = lib.transform_dataset(D, T, None)
    X = concat_features(D)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    if params is None:
        catboost_config = get_catboost_config(real_data_path, is_cv=True)
    else:
        catboost_config = params

    if 'cat_features' not in catboost_config:
        catboost_config['cat_features'] = list(range(D.n_num_features, D.n_features))

    for col in range(D.n_features):
        for split in X.keys():
            if col in catboost_config['cat_features']:
                X[split][col] = X[split][col].astype(str)
            else:
                X[split][col] = X[split][col].astype(float)
    print(T_dict)
    pprint(catboost_config, width=100)
    print('-'*100)
    
    if D.is_regression:
        model = CatBoostRegressor(
            **catboost_config,
            eval_metric='RMSE',
            random_seed=seed
        )
        predict = model.predict
    else:
        model = CatBoostClassifier(
            loss_function="MultiClass" if D.is_multiclass else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=seed,
            class_names=[str(i) for i in range(D.n_classes)] if D.is_multiclass else ["0", "1"]
        )
        predict = (
            model.predict_proba
            if D.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )

    model.fit(
        X['train'], D.y['train'],
        eval_set=(X['val'], D.y['val']),
        verbose=100
    )
    # ---- added ----     
    import numpy as np
    print(f"Sampling finished for seed {seed}")
    print(f" x_num shape: {X['train'].shape} \t y_gen shape: {D.y['train'].shape}\t")
    try:
        total = np.concatenate((X['train'],  np.expand_dims(D.y['train'],-1)), axis=-1)
        import pandas as pd
        df = pd.DataFrame(total, columns =["0","1","2","3","4","5","6","7","8","9","10","11","12","13","y"])
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        df.to_csv(f'outputs/output_syn{seed}.csv', index=False)
    except Exception as e:
        print("Could not save synthetic dataset. Error:")
        print(e)

    
    print('trying to load real data...')

    X_num_r, X_cat_r, y_r = read_pure_data(real_data_path)
    X_num_test_r, X_cat_test_r, y_test_r = read_pure_data(real_data_path, 'test')
    D_r = lib.Dataset(
    {'train': X_num_r, 'val': X_num_val, 'test': X_num_test_r} if X_num_r is not None else None,
    {'train': X_cat_r, 'val': X_cat_val, 'test': X_cat_test_r} if X_cat_r is not None else None,
    {'train': y_r, 'val': y_val, 'test': y_test_r},
    {},
    lib.TaskType(info['task_type']),
    info.get('n_classes')
    )
    D_r = lib.transform_dataset(D_r, T, None)
    X_r = concat_features(D_r)

    print(f" x_num shape: {X_r['train'].shape} \t y_gen shape: {D_r.y['train'].shape}\t")
    try:
        total = np.concatenate((X_r['train'],  np.expand_dims(D_r.y['train'],-1)), axis=-1)
        import pandas as pd
        df = pd.DataFrame(total, columns =["0","1","2","3","4","5","6","7","8","9","10","11","12","13","y"])
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        df.to_csv(f'outputs/output_r{seed}.csv', index=False)
    except Exception as e:
        print("Could not save real dataset. Error:")
        print(e)



    # ---- ---- ----

    predictions = {k: predict(v) for k, v in X.items()}
    print(predictions['train'].shape)

    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = lib.MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_catboost.json"))

    return metrics_report

    