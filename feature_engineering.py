import pandas as pd

def generate_features(orders, prior, products, aisles, departments):
    print("Merging prior orders with product info...")
     # Step 1: Merge prior with products
    prior = prior.merge(products, on='product_id', how='left')
    prior = prior.merge(aisles, on='aisle_id', how='left')
    prior = prior.merge(departments, on='department_id', how='left')

    # Step 2: Merge prior with orders to get user_id
    prior = prior.merge(orders[['order_id', 'user_id']], on='order_id', how='left')

    # data = prior.merge(products, on='product_id', how='left')
    # data = data.merge(orders, on='order_id', how='left')
    data = prior.copy()

    #  Product-level features
    print(" Creating product-level features...")
    product_features = data.groupby('product_id').agg({
        'reordered': ['mean', 'sum'],
        'order_id': 'count'
    }).reset_index()
    product_features.columns = ['product_id', 'reorder_ratio', 'times_reordered', 'times_purchased']

    #  User-Product interaction features
    print(" Creating user-product interaction features...")
    user_product = data.groupby(['user_id', 'product_id']).agg({
        'order_id': 'count',
        'reordered': 'sum'
    }).reset_index()
    user_product.columns = ['user_id', 'product_id', 'up_order_count', 'up_reorder_count']

    #  User-level features
    print(" Creating user-level features...")
    user_features = orders.groupby('user_id').agg({
        'order_number': 'max',
        'days_since_prior_order': 'mean'
    }).reset_index()
    user_features.columns = ['user_id', 'user_total_orders', 'avg_days_between_orders']

    #  Merge features
    print(" Merging all features...")
    final = user_product.merge(user_features, on='user_id', how='left')
    final = final.merge(product_features, on='product_id', how='left')

    #  Generate label (reordered or not)
    print(" Merging labels from prior orders...")
    labels = prior[['user_id', 'product_id', 'reordered']].drop_duplicates()
    final = final.merge(labels, on=['user_id', 'product_id'], how='left')
    final['reordered'] = final['reordered'].fillna(0)

    print("Feature generation complete.")
    print("Final feature columns:", final.columns.tolist())
    print("Final shape:", final.shape)

    return final
