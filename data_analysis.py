"""
    Data Analysis
"""


import json
import pandas as pd
from pandas.io.json import json_normalize


def analyze_data(file_name):
    """Analysis the dataset"""

    customers_df = multiple_json_objects_to_df(file_name)
    customers_df.to_csv('customers_flatten.csv')


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


if __name__ == '__main__':
    file_name = 'customers.json'
    analyze_data(file_name)