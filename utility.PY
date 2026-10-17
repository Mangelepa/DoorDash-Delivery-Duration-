"""Helper functions for the Delivery Duration Prediction project."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def add_time_features(df):
    """Create time features using the local (US/Pacific) time of the order."""
    df = df.copy()
    local_time = df['created_at'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
    df['hour'] = local_time.dt.hour
    df['hour_frac'] = local_time.dt.hour + local_time.dt.minute / 60
    df['day_of_week'] = local_time.dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def add_marketplace_features(df):
    """Create features that describe how busy the marketplace is."""
    df = df.copy()
    df['dashers_missing'] = df['total_onshift_dashers'].isna().astype(int)
    df['busy_gt_onshift'] = (df['total_busy_dashers'] > df['total_onshift_dashers']).astype(int)
    df['free_dashers'] = df['total_onshift_dashers'] - df['total_busy_dashers']
    onshift = df['total_onshift_dashers'].replace(0, np.nan)
    df['busy_ratio'] = df['total_busy_dashers'] / onshift
    df['orders_per_onshift'] = df['total_outstanding_orders'] / onshift
    df['orders_per_free'] = df['total_outstanding_orders'] / df['free_dashers'].where(df['free_dashers'] > 0)
    return df


def add_order_features(df):
    """Create features that describe the size and complexity of the order."""
    df = df.copy()
    df['avg_item_price'] = df['subtotal'] / df['total_items']
    df['price_range'] = df['max_item_price'] - df['min_item_price']
    df['items_per_distinct'] = df['total_items'] / df['num_distinct_items']
    return df


def target_encode(train, test, key, target='delivery_duration', n_splits=5, smoothing=20):
    """Smoothed mean-target encoding.

    Train rows are encoded out-of-fold so a row never sees its own target.
    Test rows are encoded with the statistics of the full training set.
    """
    global_mean = train[target].mean()
    train_encoded = np.zeros(len(train))

    for fit_idx, enc_idx in KFold(n_splits, shuffle=True, random_state=42).split(train):
        stats = train.iloc[fit_idx].groupby(key)[target].agg(['sum', 'count'])
        mapping = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
        train_encoded[enc_idx] = train.iloc[enc_idx][key].map(mapping).fillna(global_mean).values

    stats = train.groupby(key)[target].agg(['sum', 'count'])
    mapping = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
    test_encoded = test[key].map(mapping).fillna(global_mean).values
    return train_encoded, test_encoded


def one_hot_encode(train, test, columns):
    """One-hot encode columns in train and test, keeping the same dummy columns in both."""
    train_dummies = pd.get_dummies(train[columns].astype('Int64').astype(str), dtype=int)
    test_dummies = pd.get_dummies(test[columns].astype('Int64').astype(str), dtype=int)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    return (pd.concat([train, train_dummies], axis=1),
            pd.concat([test, test_dummies], axis=1))


def evaluate_model(name, y_true, y_pred):
    """Return the main regression metrics for one model as a dictionary."""
    return {
        'Model': name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Within_10_min_%': np.mean(np.abs(y_true - y_pred) <= 600) * 100,
    }


class OffsetRegressor(RegressorMixin, BaseEstimator):
    """Prep-time decomposition as a scikit-learn estimator.

    The wrapped model predicts the remainder  y - X[offset_col]  and the
    offset (already-known stage times) is added back when predicting.
    """

    def __init__(self, estimator=None, offset_col='known_time'):
        self.estimator = estimator
        self.offset_col = offset_col

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y - X[self.offset_col])
        return self

    def predict(self, X):
        return self.estimator_.predict(X) + X[self.offset_col].values