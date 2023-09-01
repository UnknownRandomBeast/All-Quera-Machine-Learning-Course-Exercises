
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class Preprocessor : 
    def __init__(self, df):
        self.df = df.copy()

    def handling_cat_cols(self) :
        one_hot_encoded_df = pd.get_dummies(self.df['Gender'], prefix='Gender')
        self.df = self.df.join(one_hot_encoded_df)
        self.df.drop('Gender', axis=1, inplace=True)

    def remove_junk_cols(self):
        self.df.drop('CustomerID', axis=1, inplace=True)

    def normalize(self):
        numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

        for col in numerical_cols:
            self.df[col] = self.df[col] - self.df[col].min()
            self.df[col] = self.df[col] / self.df[col].max()

    def reduce_dimensions(self):
        pca = PCA(n_components=2)
        self.df = pca.fit_transform(self.df.to_numpy())
        self.df = pd.DataFrame(self.df)
        self.df.columns = self.df.columns.astype(str)

    def transform (self) : 
        self.handling_cat_cols()
        self.remove_junk_cols()
        self.normalize()
        self.reduce_dimensions()

        return self.df
