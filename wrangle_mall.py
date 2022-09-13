import pandas as pd
import env
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_mallcustomer_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
    return df.set_index('customer_id')

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_missing = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 
                                 'perc_cols_missing': prnt_missing}).\
    reset_index().groupby(['num_cols_missing', 
                           'perc_cols_missing']).\
    count().reset_index().rename(columns={'customer_id':'count'})
        
    return rows_missing

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    prnt_missing = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing,
                                'perc_rows_missing': prnt_missing})
    
    return cols_missing


def summarize(df):
    print('DataFrame head: ')
    print(df.head())
    print()
    print()
    print('DataFrame info: ')
    print(df.info())
    print()
    print()
    print('DataFrame describe: ')
    print(df.describe().T)
    print()
    print()
    print('DataFrame nulls by col: ', nulls_by_col(df))
    print()
    print()
    print('DataFrame nulls by row: ', nulls_by_row(df))
    print()
    print()
    nums = [col for col in df.columns if df[col].dtype != 'O']
    cats = [col for col in df.columns if col not in nums]
    print('Value Counts: ')
    for col in df.columns:
        print('Column Name: '+ col)
        if col in cats:
            print(df[col].value_counts())
            print()
        else:
            print(df[col].value_counts(bins=10, sort=False))
            print()
    print()
    print()
    print()
    print('Report Finished')
    

def handle_outliers(df):
    nums = [col for col in df.columns if df[col].dtype != 'O']
    cats = [col for col in df.columns if col not in nums]

    for col in nums:
        q1,q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr

        df = df[(df[col] > lower) & (df[col] < upper)]
    return df

    
def split_data(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    # split the data into train and test. 
    train, test = train_test_split(df, test_size = .2, random_state=123)
    
    # split the train data into train and validate
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test


def scale_data(train, val, test, cols_to_scale):
    '''
    This function takes in train, validate, and test dataframes as well as a
    list of features to be scaled via the MinMaxScalar. It then returns the 
    scaled versions of train, validate, and test in new dataframes. 
    '''
    # create copies to not mess with the original dataframes
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    # create the scaler and fit it
    scaler = MinMaxScaler()
    scaler.fit(train[cols_to_scale])
    
    # use the scaler to scale the data and resave
    train_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(train[cols_to_scale]),
                                               columns = train[cols_to_scale].columns.values).set_index([train.index.values])
    val_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(val[cols_to_scale]),
                                               columns = val[cols_to_scale].columns.values).set_index([val.index.values])
    test_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(test[cols_to_scale]),
                                               columns = test[cols_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, val_scaled, test_scaled