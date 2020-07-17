"""
    Provides analysis of the customers dataset, engineers features, and detect which customers are fraudulent.
"""


import argparse
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score


def perform_fraud_detection(file_name, test_pc):
    """Takes the customers dataset and performs fraud detection"""

    # Load all data into flatten dataframe
    df = multiple_json_objects_to_df(file_name)
    # Remove users without any activity
    df = remove_users_without_activity_info(df)
    # Add new columns
    df = add_flat_columns(df)

    # Group data by user and create summary features for each user
    customers_df = get_customer_df(df)

    # Split data into train and test
    train_df, test_df = split_dataset(customers_df, test_pc)

    # Convert bool output to int
    train_df.loc[:, 'fraudulent'] = train_df['fraudulent'].astype(int)
    test_df.loc[:, 'fraudulent'] = test_df['fraudulent'].astype(int)
    # Separate predictive from output variables
    y_train = train_df.pop('fraudulent')
    y_test = test_df.pop('fraudulent')

    # Train model
    rf, train_df, sc = rand_forest_model(train_df, y_train)

    # Test model
    x_test = sc.transform(test_df)
    print("\n\nEvaluating model on training set")
    evaluate_model(rf, train_df, y_train)
    print("\n\nEvaluating model on test set")
    evaluate_model(rf, x_test, y_test)


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


def get_customer_df(df):
    """Groups data by customer and creates features that summarize users"""

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
    customers_df.loc[:, 'nr_unique_orders'] = gb['orderId'].nunique()

    # Percentage of failed transactions
    customers_df.loc[:, 'percent_failed_transactions'] = \
        gb.apply(lambda row:
                 row['transactionFailed'].sum()/row['transactionFailed'].count()
                 if row['transactionFailed'].count() > 0 else 0)

    # Number of different payment methods type used per user
    customers_df.loc[:, 'nr_different_payment_types'] = gb['paymentMethodType'].nunique()
    # Minimum number of characters in payment method issuer
    customers_df = pd.merge(customers_df,
                            gb.agg({'nr_chars_payment_method_issuer': 'min'}).rename(
                                columns={'nr_chars_payment_method_issuer': 'min_nr_chars_payment_method_issuer'}),
                            on='customer.id')

    # Median and maximum transaction amount
    customers_df.loc[:, 'median_transaction_amount'] = gb['transactionAmount'].median()
    customers_df = pd.merge(customers_df,
                            gb.agg({'transactionAmount': 'max'}).rename(
                                columns={'transactionAmount': 'max_transaction_amount'}),
                            on='customer.id')

    # Is this email shared by other users
    duplicated_emails = set(customers_df[
        customers_df.duplicated(subset='customer.customerEmail')]['customer.customerEmail'].tolist())
    customers_df.loc[:, 'duplicated_email'] = customers_df.apply(
        lambda row: row['customer.customerEmail'] in duplicated_emails, axis=1)

    # Get customer email domain
    customers_df.loc[:, 'email_domain'] = customers_df.apply(lambda row: get_email_domain(row['customer.customerEmail']),
                                                      axis=1)
    # Count number of dots in email domain
    customers_df.loc[:, 'nr_dots_email_domain'] = customers_df.apply(lambda row: row['email_domain'].count('.'), axis=1)

    # Convert boolean duplicated email column to int (0, 1)
    customers_df.loc[:, 'duplicated_email'] = customers_df['duplicated_email'].astype(int)

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


def add_flat_columns(df):
    """Adds new columns to the flat data frame"""

    # Number of payments that don't have orders nor transactions associated
    cols = get_activity_no_payment_cols()
    df['is_payment_without_correspondence'] = pd.isnull(df[cols]).all(axis=1)
    df.loc[:, 'is_payment_without_correspondence'] = pd.isnull(df[cols]).all(axis=1)
    # Number of characters in paymentMethodIssuer
    df.loc[:, 'nr_chars_payment_method_issuer'] = df.apply(lambda row:
                                                           count_nr_chars(row['paymentMethodIssuer']), axis=1)

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


def split_dataset(customers_df, test_pc):
    """Splits the dataset into train and test sets."""

    # Split data by class
    neg_df = customers_df[~customers_df['fraudulent']]
    pos_df = customers_df[customers_df['fraudulent']]
    # Shuffle rows
    neg_df = neg_df.sample(frac=1, random_state=123).reset_index(drop=True)
    pos_df = pos_df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Compute split indexes
    nr_pos_examples = pos_df.shape[0]
    nr_neg_examples = neg_df.shape[0]
    test_pos_ind = int(nr_pos_examples * (1 - test_pc))
    test_neg_ind = int(nr_neg_examples * (1 - test_pc))

    # Split data
    train_df = pd.concat([neg_df.iloc[:test_neg_ind, :], pos_df.iloc[:test_pos_ind, :]])
    test_df = pd.concat([neg_df.iloc[test_neg_ind:, :], pos_df.iloc[test_pos_ind:, :]])

    # Shuffle positive and negative examples from the training set
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    return train_df, test_df


def rand_forest_model(x_train, y_train):
    """Fits a random forest model on the training data"""

    sc = StandardScaler()
    train_norm_data = sc.fit_transform(x_train)
    rf = RandomForestClassifier(random_state=0)
    rf.fit(train_norm_data, y_train)

    return rf, train_norm_data, sc


def evaluate_model(model, x, y_gt):
    """Evaluates a model"""

    # Compute predictions
    y_pred = model.predict(x)

    print("Confusion matrix:")
    print(confusion_matrix(y_gt, y_pred))
    print("Classification report:")
    print(classification_report(y_gt, y_pred))
    print("Accuracy:")
    print(accuracy_score(y_gt, y_pred))
    print("AUC:")
    print(roc_auc_score(y_gt, model.predict_proba(x)[:, 1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", type=str, default='customers.json', help='Path to the customers dataset.')
    parser.add_argument(
        "--test_pc", type=float, default=0.2, help='Percentage of the dataset used for test.')
    args = parser.parse_args()

    perform_fraud_detection(args.data_file, args.test_pc)
