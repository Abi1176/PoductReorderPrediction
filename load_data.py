import pandas as pd

def load_all_data():
    """
    Load all data from the CSV files in the 'data' directory.
    
    Returns:
        dict: A dictionary with DataFrames for each dataset.
    """
    # datasets = {}
    # datasets['train'] = pd.read_csv('data/train.csv')
    # datasets['test'] = pd.read_csv('data/test.csv')
    # datasets['validation'] = pd.read_csv('data/validation.csv')
    # orders= pd.read_csv('data/orders.csv')
    # products = pd.read_csv('data/products.csv')
    # customers = pd.read_csv('data/customers.csv')
    # datasets = {
    #     'orders': orders,
    #     'products': products,
    #     'customers': customers
    # }
    orders = pd.read_csv('data/orders.csv')
    prior = pd.read_csv('data/order_products__prior.csv')
    train = pd.read_csv('data/order_products__train.csv')
    products = pd.read_csv('data/products.csv')
    aisles = pd.read_csv('data/aisles.csv')
    departments = pd.read_csv('data/departments.csv')
    return orders, prior, train, products, aisles, departments
