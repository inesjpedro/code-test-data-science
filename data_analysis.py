"""
    Data Analysis
"""


import datetime
import json
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def train_classifier(file_name, valid_pc, test_pc, lr, log_folder, model_name, patience, epochs, l2_reg):
    """Analysis the dataset"""

    # Load all data into flatten dataframe
    df = multiple_json_objects_to_df(file_name)
    # Remove users without any activity
    df = remove_users_without_activity_info(df)

    # Number of payments that don't have orders nor transactions associated
    cols = get_activity_no_payment_cols()
    df['is_payment_without_correspondence'] = pd.isnull(df[cols]).all(axis=1)
    # Number of characters in paymentMethodIssuer
    df['nr_chars_payment_method_issuer'] = df.apply(lambda row: count_nr_chars(row['paymentMethodIssuer']), axis=1)

    # Group data by user
    customers_df = get_customer_df(df)

    # Convert boolean column to int (0, 1)
    customers_df['duplicated_email'] = customers_df['duplicated_email'].astype(int)

    # Split dataset into train, validation and test
    train_df, valid_df, test_df = split_dataset(customers_df, valid_pc, test_pc)

    # Split predictive from output variables
    train_dataset = dataframe_to_tf_dataset(train_df)
    valid_dataset = dataframe_to_tf_dataset(valid_df)
    test_dataset = dataframe_to_tf_dataset(test_df)

    # Shuffle and batch the dataset
    train_dataset = train_dataset.shuffle(len(train_df)).batch(1)
    valid_dataset = valid_dataset.batch(1)
    test_dataset = test_dataset.batch(1)

    # Create model
    model = tf.keras.Sequential()
    if l2_reg is None:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))

    # Compile model
    metrics = ['acc',
               tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalseNegatives(name='fn'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall'),
               tf.keras.metrics.AUC(name='auc')]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss='binary_crossentropy',
                  metrics=metrics)

    # Get callbacks
    callbacks = get_callbacks(log_folder, model_name, patience)

    hist = model.fit(train_dataset,
                     epochs=epochs,
                     validation_data=valid_dataset,
                     callbacks=callbacks,
                     verbose=0)

    # Evaluate the dataset
    metric_names = ['loss', 'Accuracy', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'AUC']
    train_res = model.evaluate(train_dataset)
    print_results(train_res, 'Training', metric_names)
    valid_res = model.evaluate(valid_dataset)
    print_results(valid_res, 'Validation', metric_names)
    if test_pc > 0:
        test_res = model.evaluate(test_dataset)
        print_results(test_res, 'Test', metric_names)

    plot_metrics(hist, model_name)


def train_rf_classifier(file_name, test_pc):

    # Load all data into flatten dataframe
    df = multiple_json_objects_to_df(file_name)
    # Remove users without any activity
    df = remove_users_without_activity_info(df)

    # Number of payments that don't have orders nor transactions associated
    cols = get_activity_no_payment_cols()
    df['is_payment_without_correspondence'] = pd.isnull(df[cols]).all(axis=1)
    # Number of characters in paymentMethodIssuer
    df['nr_chars_payment_method_issuer'] = df.apply(lambda row: count_nr_chars(row['paymentMethodIssuer']), axis=1)

    # Group data by user
    customers_df = get_customer_df(df)

    # Convert boolean column to int (0, 1)
    customers_df['duplicated_email'] = customers_df['duplicated_email'].astype(int)

    # Split dataset into train and test
    train_df, test_df, _ = split_dataset(customers_df, test_pc, 0)

    # Dataframe to numpy array
#    x_train, y_train = dataframe_to_numpy(train_df)
#    x_test, y_test = dataframe_to_numpy(test_df)
    train_df['fraudulent'] = train_df['fraudulent'].astype(int)
    test_df['fraudulent'] = test_df['fraudulent'].astype(int)
    y_train = train_df.pop('fraudulent')
    y_test = test_df.pop('fraudulent')

    # Train model
    rf, train_df, sc = rand_forest_model(train_df, y_train)
    y_train_pred = rf.predict(train_df)

    # Test model
    x_test = sc.transform(test_df)
    y_test_pred = rf.predict(x_test)

    print("\n\nEvaluating model on training set")
    evaluate_rand_forest(y_train, y_train_pred)
    print("\n\nEvaluating model on test set")
    evaluate_rand_forest(y_test, y_test_pred)


def dataframe_to_numpy(df):
    """Transforms data frame to numpy X and Y arrays"""

    nr_ex = df.shape[0]
    y = df['fraudulent'].to_numpy(int).reshape(nr_ex, 1)

    pred_cols = [col for col in df.columns if col != 'fraudulent']
    x = df[pred_cols].to_numpy(float)

    return x, y


def get_customer_df(df):
    """Groups data by customer and adds new features"""

    # Group reviews by customer ID
    gb = df.groupby('customer.id')
    customers_df = gb.size().to_frame(name='nr_rows')

    unique_cols = ['customer.id', 'fraudulent', 'customer.customerEmail',
                   'customer.customerPhone', 'customer.customerDevice',
                   'customer.customerIPAddress', 'customer.customerBillingAddress']
    customers_df = pd.merge(customers_df,
                            df[unique_cols].groupby('customer.id').first().reset_index(),
                            on='customer.id')

    # Add new features
    customers_df = pd.merge(customers_df,
                            gb.agg({'is_payment_without_correspondence': 'sum'}).rename(
                                columns={'is_payment_without_correspondence': 'nr_payments_without_correspondence'}),
                            on='customer.id')

    # Number of unique orders
    customers_df['nr_unique_orders'] = gb['orderId'].nunique()

    # Percentage of failed transactions
    customers_df['percent_failed_transactions'] = \
        gb.apply(lambda row:
                 row['transactionFailed'].sum()/row['transactionFailed'].count()
                 if row['transactionFailed'].count() > 0 else 0)

    # Number of different payment methods type used per user
    customers_df['nr_different_payment_types'] = gb['paymentMethodType'].nunique()
    # Minimum number of characters in payment method issuer
    customers_df = pd.merge(customers_df,
                            gb.agg({'nr_chars_payment_method_issuer': 'min'}).rename(
                                columns={'nr_chars_payment_method_issuer': 'min_nr_chars_payment_method_issuer'}),
                            on='customer.id')

    # Median and maximum transaction amount
    customers_df['median_transaction_amount'] = gb['transactionAmount'].median()
    customers_df = pd.merge(customers_df,
                            gb.agg({'transactionAmount': 'max'}).rename(
                                columns={'transactionAmount': 'max_transaction_amount'}),
                            on='customer.id')

    # Is this email shared by other users
    duplicated_emails = set(customers_df[
        customers_df.duplicated(subset='customer.customerEmail')]['customer.customerEmail'].tolist())
    customers_df['duplicated_email'] = customers_df.apply(
        lambda row: row['customer.customerEmail'] in duplicated_emails, axis=1)

    # Get customer email domain
    customers_df['email_domain'] = customers_df.apply(lambda row: get_email_domain(row['customer.customerEmail']),
                                                      axis=1)
    # Count number of dots in email domain
    customers_df['nr_dots_email_domain'] = customers_df.apply(lambda row: row['email_domain'].count('.'), axis=1)

    print("Number of fraudulent users: {} (~ {:.2f}%)".format(
        customers_df[customers_df['fraudulent']].shape[0],
        customers_df[customers_df['fraudulent']].shape[0] / customers_df.shape[0] * 100
    ))

    # Drop columns that are not going to be used
    irrelevant_cols = unique_cols[2:] + ['customer.id', 'nr_rows', 'email_domain']
    customers_df = customers_df.drop(irrelevant_cols, axis=1)

    # Replace NaN values with 0
    customers_df = customers_df.fillna(0)

    return customers_df


def split_dataset(customers_df, valid_pc, test_pc):
    """Splits the dataset into train, valid and test sets."""

    # Split data by class
    neg_df = customers_df[~customers_df['fraudulent']]
    pos_df = customers_df[customers_df['fraudulent']]
    # Shuffle rows
    neg_df = neg_df.sample(frac=1, random_state=123).reset_index(drop=True)
    pos_df = pos_df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Compute split indexes
    nr_pos_examples = pos_df.shape[0]
    nr_neg_examples = neg_df.shape[0]
    valid_pos_ind = int(nr_pos_examples * (1 - (valid_pc + test_pc)))
    test_pos_ind = int(nr_pos_examples * (1 - test_pc))
    valid_neg_ind = int(nr_neg_examples * (1 - (valid_pc + test_pc)))
    test_neg_ind = int(nr_neg_examples * (1 - test_pc))

    # Split data
    train_df = pd.concat([neg_df.iloc[:valid_neg_ind, :], pos_df.iloc[:valid_pos_ind, :]])
    valid_df = pd.concat([neg_df.iloc[valid_neg_ind:test_neg_ind, :], pos_df.iloc[valid_pos_ind:test_pos_ind, :]])
    test_df = pd.concat([neg_df.iloc[test_neg_ind:, :], pos_df.iloc[test_pos_ind:, :]])

    # Shuffle positive and negative examples
    train_df = train_df.sample(frac=1).reset_index(drop=True)
#    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
#    test_df = test_df.sample(frac=1).reset_index(drop=True)

    return train_df, valid_df, test_df


def dataframe_to_tf_dataset(df):
    """Converts dataframe to TensorFlow dataset"""

    target = df.pop('fraudulent')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    return dataset


def get_callbacks(log_folder, model_name, patience):
    """Creates list with TensorBoard and Early Stopping callbacks"""

    # TensorBoard callback
    log_dir = os.path.join(log_folder, '{}_{}'.format(model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
    callbacks = [tb_callback]

    # Early Stopping callback
    if patience > 0:
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        callbacks.append(es_callback)

    return callbacks


def print_results(res, split, metrics):
    """Prints the model evaluation on a data split"""

    p = -1
    r = -1

    print("\n\n{} results:".format(split))
    for i, m in enumerate(metrics):
        print("{}: {:.4f}".format(m, res[i]))
        if m == 'Precision':
            p = res[i]
        if m == 'Recall':
            r = res[i]

    if all(m >= 0 for m in [p, r]):
        f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
        print("F1-score: {:.4f}".format(f1))


def plot_metrics(history, model_name):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

    plt.savefig('plots/{}.png'.format(model_name))


def rand_forest_model(X_train, y_train):
    """Random forest"""

    sc = StandardScaler()
    train_norm_data = sc.fit_transform(X_train)
    rf = RandomForestClassifier(random_state=0)
    rf.fit(train_norm_data, y_train)

    return rf, train_norm_data, sc


def evaluate_rand_forest(y_gt, y_pred):
    print("Confusion matrix:")
    print(confusion_matrix(y_gt, y_pred))
    print("Classification report:")
    print(classification_report(y_gt, y_pred))
    print("Accuracy:")
    print(accuracy_score(y_gt, y_pred))


def multiple_json_objects_to_df(json_file):
    """Converts a JSON with multiple objects to a flat data frame"""

    with open(json_file) as f:
        for i, json_obj in enumerate(f):
            customer_dict = json.loads(json_obj)
            customer_dict['customer.id'] = i

            # Create basic customer profile
            customer_df = pd.json_normalize(
                {k: customer_dict[k] for k in ['customer.id', 'fraudulent', 'customer']}
            )

            # Get sub dataframe with transactions info
            sub_df = pd.json_normalize(customer_dict['transactions'])
            # Add orders info
            tmp_df = pd.json_normalize(customer_dict['orders'])
            if sub_df.shape[0] > 0:
                sub_df = pd.merge(sub_df, tmp_df, on='orderId', how="outer")
            else:
                sub_df = tmp_df.copy()
            # Add payment methods info
            tmp_df = pd.json_normalize(customer_dict['paymentMethods'])
            if sub_df.shape[0] > 0:
                sub_df = pd.merge(sub_df, tmp_df, on='paymentMethodId', how="outer")
            elif tmp_df.shape[0] > 0:
                sub_df = tmp_df.copy()
            else:
                sub_df = sub_df.append(pd.Series(), ignore_index=True)
            sub_df['customer.id'] = i

            # Merge customer info with transaction info
            customer_df = pd.merge(customer_df, sub_df, on='customer.id')

            if i == 0:
                final_df = customer_df.copy()
            else:
                final_df = pd.concat([final_df, customer_df])

    return final_df


def remove_users_without_activity_info(df):
    """Removes users that don't contain any activity info"""

    activity_cols = ['transactionId', 'orderId', 'paymentMethodId',
                     'transactionAmount', 'transactionFailed', 'orderAmount',
                     'orderState', 'orderShippingAddress',
                     'paymentMethodRegistrationFailure',
                     'paymentMethodType', 'paymentMethodProvider',
                     'paymentMethodIssuer']

    tmp = df.shape[0]
    df = df.dropna(subset=activity_cols, how='all')
    print("Number of eliminated rows of customers without any activity information: {} (~ {:.2f}%)".format(
        tmp - df.shape[0], (tmp - df.shape[0]) / tmp * 100
    ))

    return df


def get_email_domain(email):
    """Returns the domain of the email"""

    email_split = email.split("@")
    if len(email_split) == 2:
        return email_split[1]
    else:
        return ''


def count_nr_chars(s):
    """Counts number of characters in string"""

    stripped_s = "".join(s.split())
    return len(stripped_s)


def get_activity_no_payment_cols():
    """Gets column names that are related with activity but not payment methods"""

    return ['transactionId', 'orderId', 'transactionAmount', 'transactionFailed', 'orderAmount', 'orderState',
            'orderShippingAddress']


if __name__ == '__main__':
    file_name = 'customers.json'
    valid_pc = 0.15
    test_pc = 0.15
    lr = 0.001
    log_folder = 'logs'
    model_name = 'fraud_classifier_lr_001_es_40'
    patience = 40
    epochs = 500
    l2_reg = None

#    train_classifier(file_name, valid_pc, test_pc, lr, log_folder, model_name, patience, epochs, l2_reg)
    train_rf_classifier(file_name, 0.2)