def get_column_names(cat_cols, num_cols, target_col):
    """
    Returns a list of column names in the order they appear in the dataset.
    Expects that cat_columns and num_columns are in order.
    """
    all_cols = cat_cols + num_cols 
    all_cols += [all_cols.pop(all_cols.index(target_col))] # moves the target column to the end
    return all_cols