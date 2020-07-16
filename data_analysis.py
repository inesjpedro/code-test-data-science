"""
    Data Analysis
"""


import json
import pandas as pd


def analyze_data(file_name):
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
    customers_df = customers_df.drop(unique_cols[2:], axis=1)

    # Replace NaN values with 0
    customers_df = customers_df.fillna(0)

    return customers_df


def split_dataset(customers_df, split_pc):
    """Splits the dataset into train, valid and test sets."""


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
    analyze_data(file_name)