import pandas as pd

def generate_features(orders, prior, products, aisles, departments):
    # Merge prior orders with product info and user info
    df = pd.merge(prior, orders, on='order_id', how='left')
    df = pd.merge(df, products, on='product_id', how='left')
    df = pd.merge(df, aisles, on='aisle_id', how='left')
    df = pd.merge(df, departments, on='department_id', how='left')

    # User-level features
    user_features = orders.groupby('user_id').agg({
        'order_number': 'max',
        'days_since_prior_order': 'mean'
    }).rename(columns={
        'order_number': 'user_total_orders',
        'days_since_prior_order': 'avg_days_between_orders'
    })

    # Product-level features
    product_features = prior.groupby('product_id').agg({
        'reordered': ['sum', 'count']
    })
    product_features.columns = ['times_reordered', 'times_purchased']
    product_features['reorder_ratio'] = product_features['times_reordered'] / product_features['times_purchased']

    # User-product interaction features
    user_product = df.groupby(['user_id', 'product_id']).agg({
        'add_to_cart_order': 'count',
        'reordered': 'sum'
    }).rename(columns={
        'add_to_cart_order': 'up_order_count',
        'reordered': 'up_reorder_count'
    })

    # Combine all features
    final = user_product.merge(user_features, on='user_id')
    final = final.merge(product_features, on='product_id')
    final.reset_index(inplace=True)
    return final