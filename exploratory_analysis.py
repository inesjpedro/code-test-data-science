"""
    This script is dedicated to perform exploratory analysis to the customer dataset.
"""


import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from fraud_detection import multiple_json_objects_to_df, remove_users_without_activity_info, add_flat_columns, \
    get_customer_df


def dataset_analysis(file_name):
    """Performs exploratory analysis on the dataset"""

    # Load all data into flatten dataframe
    df = multiple_json_objects_to_df(file_name)
    # Remove users without any activity
    df = remove_users_without_activity_info(df)
    # Add new columns
    df = add_flat_columns(df)

    # Is it always true that transactionAmount == orderAmount?
    are_amounts_equal(df)

    # Are there duplicated users?
    duplicated_users(df)

    # Is there a strong relation between the type of payment method
    # used and the type of customer (fraudulent or not)?
    plot_hist_per_class(df, 'paymentMethodType', 'Payment method type')

    # Is there a strong relation between the order state and the type of customer?
    df['orderState'].fillna('NaN', inplace=True)
    plot_hist_per_class(df, 'orderState', 'Order state')

    # Group data by user and create summary features for each user
    customers_df = get_customer_df(df)

    # Plot data distribution
    plot_label_distr(customers_df)

    # Plot the correlation matrix
    correlation_matrix(customers_df, 'fraudulent')

    # Plot the histogram of given features
    plot_hist_per_class(customers_df, 'min_nr_chars_payment_method_issuer',
                        'Minimum number of characters in payment method issuer')
    plot_hist_per_class(customers_df, 'nr_payments_without_correspondence', 'Number of payments without correspondence')
    plot_hist_per_class(customers_df, 'duplicated_email', 'Duplicated email')
    plot_hist_per_class(customers_df, 'max_transaction_amount', 'Maximum transaction amount')


def plot_label_distr(customers_df):
    """Plots histogram of labels"""

    nr_fraud = customers_df[customers_df['fraudulent']].shape[0]
    nr_non_fraud = customers_df[~customers_df['fraudulent']].shape[0]

    print("Number of fraudulent customers: {} ({:.2f}%)".format(
        nr_fraud, nr_fraud / customers_df.shape[0] * 100))
    print("Number of non-fraudulent customers: {} ({:.2f}%)".format(
        nr_non_fraud, nr_non_fraud / customers_df.shape[0] * 100))

    plt.clf()
    barlist = plt.bar(['fraud', 'non-fraud'], [nr_fraud, nr_non_fraud])
    barlist[0].set_color('r')
    barlist[1].set_color('g')
    plt.show()


def correlation_matrix(df, col):
    """Plots correlation matrix"""

    # Correlation matrix
    corr = df.corr()
    col_corr = corr[col]
    print(col_corr)
    plt.clf()
    col_corr.drop(col, inplace=True)
    col_corr.plot.barh()
    plt.tight_layout()
    plt.show()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(5, 5))
    sns_plot = sns.heatmap(corr,
                           xticklabels=corr.columns,
                           yticklabels=corr.columns,
                           cmap="YlGnBu",
                           linewidths=.5, ax=ax)


def plot_hist_per_class(customers_df, feature, title):
    """Plots the histogram of a given feature, separated by class"""

    plt.clf()
    plt.hist([customers_df.loc[customers_df['fraudulent'], feature],
              customers_df.loc[~customers_df['fraudulent'], feature]],
             color=['r', 'g'])
    plt.legend(['Fraud', 'Non-fraud'])
    plt.title('{} by type of user'.format(title))
    plt.show()


def are_amounts_equal(df):
    """Answers the question: Is it always true that transactionAmount == orderAmount?"""

    tmp_df = df[(df['orderAmount'].notnull()) | (df['transactionAmount'].notnull())]

    tmp_df.loc[:, 'amount_match'] = tmp_df.apply(lambda row: row['orderAmount'] == row['transactionAmount'], axis=1)
    tmp_df = tmp_df[~tmp_df['amount_match']][['customer.id', 'fraudulent', 'orderAmount', 'transactionAmount', 'amount_match']]

    if tmp_df.shape[0] > 0:
        print("There are {} cases in which transactionAmount != orderAmount!".format(tmp_df.shape[0]))
    else:
        print("Is it always the case that transactionAmount == orderAmount")


def duplicated_users(df):
    """Detects whether or not there are users with
       duplicated emails ou phone numbers"""

    # Group reviews by customer ID
    gb = df.groupby('customer.id')
    customers_df = gb.size().to_frame(name='nr_rows')

    unique_cols = ['customer.id', 'fraudulent', 'customer.customerEmail',
                   'customer.customerPhone', 'customer.customerDevice',
                   'customer.customerIPAddress', 'customer.customerBillingAddress']
    customers_df = pd.merge(customers_df,
                            df[unique_cols].groupby('customer.id').first().reset_index(),
                            on='customer.id')

    # Check if there are duplicated emails
    duplicated_email_df = customers_df[customers_df.duplicated(subset='customer.customerEmail', keep=False)]
    if duplicated_email_df.shape[0] > 0:
        print("\nThere are {} duplicated emails across {} customers!".format(
            len(set(duplicated_email_df['customer.customerEmail'].tolist())),
            duplicated_email_df.shape[0]
        ))
    else:
        print("\nThere are NO duplicated emails across customers.")

    # Check if there are duplicated phone numbers
    duplicated_phone_nr_df = customers_df[customers_df.duplicated(subset='customer.customerPhone', keep=False)]
    if duplicated_phone_nr_df.shape[0] > 0:
        print("\nThere are duplicated emails across customers!")
        print(duplicated_phone_nr_df)
    else:
        print("\nThere are NO duplicated emails across customers.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", type=str, default='customers.json', help='Path to the customers dataset.')
    args = parser.parse_args()

    dataset_analysis(args.data_file)
