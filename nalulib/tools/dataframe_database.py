import numpy as np
import pandas as pd
import pickle

class DataFrameDatabase:
    def __init__(self, filename=None, dfs=None, configs=None, name='', validate_columns=False):
        """
        Initializes the DataFrameDatabase.

        INPUTS:
          - filename (str): Path to a file to load the database from.
          - dfs (list): List of dataframes to initialize the database with.
          - configs (pd.DataFrame or list of dicts): DataFrame of configurations.
          - name (str): Name of the database.
          - validate_columns (bool): If True, ensures all dataframes have the same columns.
        """
        self.dfs = []
        self.configs = pd.DataFrame()
        self._common = {}
        self.validate_columns = validate_columns
        self.name = name
        if filename is not None:
            self.load(filename)
            return
        if dfs is not None:
            if not isinstance(dfs, list):
                raise Exception('dfs must be a list of dataframes')
            self.dfs = dfs 
        if configs is not None:
            if isinstance(configs, list):
                configs=pd.DataFrame(configs)
            if not isinstance(configs, pd.DataFrame):
                raise Exception('configs must be a dataframe')
            self.configs = configs 

    def insert(self, config, df):
        """
        Inserts a new dataframe and its associated config into the database.

        INPUTS:
          - config (dict): Configuration dictionary for the dataframe.
          - df (pd.DataFrame): DataFrame to insert.

        RAISES:
          - ValueError: If validate_columns is True and the dataframe's columns do not match.

        EXAMPLE:
          db = DataFrameDatabase()
          db.insert({'airfoil': 'S809', 'Re': 0.75, 'Mach': 0.3}, df1)
          db.insert({'airfoil': 'NACA0012', 'Re': 0.85, 'Mach': 0.4}, df2)
        """
        if self.validate_columns and self.dfs:
            if not all(df.columns == self.dfs[0].columns):
                raise ValueError("All dataframes must have the same columns when validate_columns is True.")

        if not isinstance(df, pd.DataFrame):
            raise Exception('df must be DataFrame')

        self.dfs.append(df)
        config_index = len(self.dfs) - 1
        try:
            config_df = pd.DataFrame([config], index=[config_index])
        except:
            import pdb; pdb.set_trace()
        self.configs = pd.concat([self.configs, config_df])

    def insert_multiple(self, configs, dfs):
        for c, df in zip(configs, dfs):
            self.insert(c, df)


    def save(self, filename):
        """
        Saves the database to a pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump({'dfs': self.dfs, 'configs': self.configs, 'name': self.name}, f)

    def load(self, filename):
        """
        Loads the database from a pickle file.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.dfs = data['dfs']
            self.configs = data['configs']
            self.name = data['name']

    def copy(self):
        dfs = [df.copy() for df in self.dfs]
        configs = self.configs.copy()
        return DataFrameDatabase(name=self.name, dfs=dfs, configs=configs)

    def _newdb(self, selected_configs=None, selected_indices=None):
        selected_configs = selected_configs.reset_index(drop=True) # New index
        new_db = DataFrameDatabase(name=self.name)
        new_db.dfs = [self.dfs[i] for i in selected_indices]
        new_db.configs = selected_configs
        return new_db

    def query(self, query):
        """
        Selects rows from the configs dataframe based on a query and returns a new database.

        INPUTS:
          - query (str): A query string to filter the configs dataframe.

        OUTPUTS:
          - DataFrameDatabase: A new database containing the filtered rows.

        EXAMPLE:
          # Query the database for rows where Re > 0.8
          filtered_db = db.query("Re > 0.8")
        """
        selected_configs = self.configs.query(query)
        selected_indices = selected_configs.index.tolist()
        return self._newdb(selected_configs, selected_indices)

    def select(self, config):
        """
        Selects rows from the database that match the given config.
        If keys are missing, return all entries that match the provided keys.

        INPUTS:
          - config (dict): A dictionary of key-value pairs to filter the configs dataframe.

        OUTPUTS:
          - DataFrameDatabase: A new database containing the filtered rows.

        EXAMPLE:
          filtered_db = db.select({'airfoil': 'S809'})

        """
        mask = pd.Series(True, index=self.configs.index)
        for key, value in config.items():
            if key in self.configs.columns:
                mask &= self.configs[key] == value
        selected_configs = self.configs[mask]
        selected_indices = selected_configs.index.tolist()
        return self._newdb(selected_configs, selected_indices)

    def select_approximate(self, key, value, tolerance):
        """
        Selects rows from the database where the given key is within a tolerance of the value.

        INPUTS:
          - key (str): The column name to filter on.
          - value (float): The target value to filter around.
          - tolerance (float): The tolerance within which the values should match.

        OUTPUTS:
          - DataFrameDatabase: A new database containing the filtered rows.

        EXAMPLE:
          # Select rows where Re is approximately 0.8 with a tolerance of 0.1
          filtered_db = db.select_approximate('Re', 0.8, 0.1)
        """
        if key not in self.configs.columns:
            raise KeyError(f"Key '{key}' not found in configs.")
        mask = self.configs[key].apply(lambda x: abs(x - value) <= tolerance if pd.notnull(x) else False)
        selected_configs = self.configs[mask]
        selected_indices = selected_configs.index.tolist()
        return self._newdb(selected_configs, selected_indices)

    def select_closest(self, key, value):
        """
        Selects rows from the database where the given key is the closest to the value provided.
        INPUTS:
          - key (str): The column name to filter on.
          - value (float): The target value to filter around.

        OUTPUTS:
          - DataFrameDatabase: A new database containing the filtered rows.

        EXAMPLE:
          # Select rows where Re is approximately 0.8 with a tolerance of 0.1
          filtered_db = db.select_closest('Re', 0.8)
        """
        if key not in self.configs.columns:
            raise KeyError(f"Key '{key}' not found in configs.")
        values = self.configs[key].values
        i = np.argmin(abs(values-value))
        val = values[i]
        #print('Value', value, 'closest', val)
        return self.select({key:val})

    
    def select_minor(self, key, value):
      """
      Selects rows from the database where the given key is <= the provided value.

      INPUTS:
        - key (str): The column name to filter on.
        - value (float): The upper threshold value to compare.

      OUTPUTS:
        - DataFrameDatabase: A new database containing the filtered rows.

      EXAMPLE:
        # Select rows where Frequency <= 1.0
        filtered_db = db.select_minor('Frequency', 1.0)
      """
      if key not in self.configs.columns:
          raise KeyError(f"Key '{key}' not found in configs.")
      
      mask = self.configs[key].apply(lambda x: x <= value if pd.notnull(x) else False)
      selected_configs = self.configs[mask]
      selected_indices = selected_configs.index.tolist()
      return self._newdb(selected_configs, selected_indices)


    @staticmethod
    def concatenate(databases, name="ConcatenatedDatabase"):
        """
        Concatenates multiple DataFrameDatabase objects into a single database.

        INPUTS:
          - databases (list): A list of DataFrameDatabase objects to concatenate.
          - name (str): Name for the new concatenated database.

        OUTPUTS:
          - DataFrameDatabase: A new database containing all entries from the input databases.
        """
        concatenated_dfs = []
        concatenated_configs = []

        for db in databases:
            concatenated_dfs.extend(db.dfs)
            concatenated_configs.append(db.configs)

        # Combine all configs into a single DataFrame
        combined_configs = pd.concat(concatenated_configs, ignore_index=True)

        # Create a new database
        new_db = DataFrameDatabase(name=name)
        new_db.dfs = concatenated_dfs
        new_db.configs = combined_configs

        return new_db

    def concat(self, db, inplace=True):
        if inplace:
            db_loc = self
        else:
            db_loc = self.copy()
            db_loc.name+='_'+db.name
            for c, df in db:
                db_loc.insert(c, df)
        return db_loc



    def simplify(self):
        """
        Returns a new database with the same dataframes but a simplified configs dataframe.
        Columns in the configs dataframe with constant values across all rows are dropped.

        OUTPUTS:
          - DataFrameDatabase: A new database with simplified configs.
        """
        ## Identify columns with constant values
        constant_columns = [col for col in self.configs.columns if self.configs[col].nunique(dropna=False) == 1]

        ## Extract the common config (constant values)
        common_config = {col: self.configs[col].iloc[0] for col in constant_columns}

        # Drop constant columns
        simplified_configs = self.configs.drop(columns=constant_columns)

        # Create a new database with the simplified configs
        new_db = DataFrameDatabase(name=self.name)
        new_db.dfs = self.dfs.copy()
        new_db.configs = simplified_configs
        new_db._common = self._common | common_config

        return new_db
        #, common_config

    @property
    def common(self):
        return self._common

    @property
    def common_config(self):
        # Identify columns with constant values
        constant_columns = [col for col in self.configs.columns if self.configs[col].nunique(dropna=False) == 1]
        # Extract the common config (constant values)
        common_config = {col: self.configs[col].iloc[0] for col in constant_columns}
        return common_config


    @property
    def configs_dict(self):
        return self.configs.to_dict(orient='records')

    @property
    def df_columns(self):
        df_keys = list(self.dfs[0].columns) if len(self)>0 else []
        return df_keys

    def keys(self):
        return self.configs.keys()

    def __getitem__(self, index_or_column):
        """
        Allows indexing into the database using db[index].

        Returns the config and dataframe at the specified index.
        """
        if type(index_or_column) is int:
            index = index_or_column
            if index < 0 or index >= len(self.dfs):
                raise IndexError("Index out of range.")
            return self.configs.loc[index].to_dict(), self.dfs[index]
        else:
            column = index_or_column
            return self.configs[column]

    def __len__(self):
        """
        Returns the number of entries in the database.
        """
        return len(self.dfs)

    def __iter__(self):
        """
        Allows iteration over configs and dataframes in the database.
        """
        for config, df in zip(self.configs.iterrows(), self.dfs):
            yield config[1].to_dict(), df

    def get_singleton(self, label='', extra_args={}):
        n = len(self)
        if n == 0:
            print(f'[FAIL] Database is empty. Cannot return a single {label}', extra_args)
            return None, None
        elif n >1:
            print(f'[WARN] Database has {n} items. Cannot return a single {label}', extra_args)
            c, df = self[0]
        else:
            c, df = self[0]
        return c, df

    def toDict(self, key=None):
        """ 
        return a dictionary 
          {v:df}
        only if there is one column
        """
        d = {}
        if key is None:
            if len(self.keys())!=1:
                raise Exception(f'toDict only works if a single column is present in the configs (e.g. after running simplify). Currently there are {len(self.keys())} columns.')
            k = self.keys()[0]
            for c, df in self:
                d[c[k]] = df
        vals = self.configs[key]
        if len(vals.unique()) != len(vals):
            raise Exception(f'toDict only works if the values of the column {key} are unique')
        for v, (c, df) in zip(vals, self):
            d[v] = df
        return d



    def __repr__(self):
        """
        Returns a string representation of the database with basic information.
        """
        num_entries = len(self)
        config_keys = list(self.configs.columns)
        df_keys = list(self.dfs[0].columns) if len(self)>0 else []
        s = f"<DataFrameDatabase {self.name}>\n"
        s+= f" - db.len()         : {num_entries}\n"
        s+= f" - db.configs.keys(): {config_keys}\n"
        s+= f" - db.df_columns    : {df_keys}\n"
        s+= f" * db.common:         {self.common}\n"
        s+= f" * db.common_config:  {self.common_config}\n"
        s+= f" - db.configs: <pandas DataFrame>:\n {self.configs}\n"
        s+= f"useful_functions: \n"
        s+=  " - newdb = db.select(config) , e.g. config={'key1':'val1'}\n"
        s+= f" - newdb = db.select_approximate(key, value, tol)\n"
        s+= f" - c, df  = db.get_singelton(config)\n"
        s+= f"useful_usage: \n"
        s+=  "   c, df = db[0]\n"
        s+=  "   for config, df in db:"
        return s
