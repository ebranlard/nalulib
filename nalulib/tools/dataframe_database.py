import pandas as pd
import pickle

class DataFrameDatabase:
    def __init__(self, filename=None, dfs=None, configs=None, name='', validate_columns=False):
        """
        Initializes the DataFrameDatabase.

        INPUTS:
          - filename (str): Path to a file to load the database from.
          - dfs (list): List of dataframes to initialize the database with.
          - configs (pd.DataFrame): DataFrame of configurations.
          - name (str): Name of the database.
          - validate_columns (bool): If True, ensures all dataframes have the same columns.
        """
        self.validate_columns = validate_columns
        if filename is not None:
            self.load(filename)
            return
        self.dfs = dfs if dfs is not None else []
        self.configs = configs if configs is not None else pd.DataFrame()
        self.name = name

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
        self.dfs.append(df)
        config_index = len(self.dfs) - 1
        config_df = pd.DataFrame([config], index=[config_index])
        self.configs = pd.concat([self.configs, config_df])

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

    def _newdb(self, selected_configs, selected_indices):
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

    def simplify(self):
        """
        Returns a new database with the same dataframes but a simplified configs dataframe.
        Columns in the configs dataframe with constant values across all rows are dropped.

        OUTPUTS:
          - DataFrameDatabase: A new database with simplified configs.
        """
        # Identify columns with constant values
        constant_columns = [col for col in self.configs.columns if self.configs[col].nunique() == 1]

        # Extract the common config (constant values)
        common_config = {col: self.configs[col].iloc[0] for col in constant_columns}

        # Drop constant columns
        simplified_configs = self.configs.drop(columns=constant_columns)

        # Create a new database with the simplified configs
        new_db = DataFrameDatabase(name=self.name)
        new_db.dfs = self.dfs.copy()
        new_db.configs = simplified_configs

        return new_db, common_config

    @property
    def configs_dict(self):
        return self.configs.to_dict(orient='records')

    @property
    def df_columns(self):
        df_keys = list(self.dfs[0].columns) if len(self)>0 else []
        return df_keys

    def __getitem__(self, index):
        """
        Allows indexing into the database using db[index].

        Returns the config and dataframe at the specified index.
        """
        if index < 0 or index >= len(self.dfs):
            raise IndexError("Index out of range.")
        return self.configs.loc[index].to_dict(), self.dfs[index]

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
        s+= f" - db.configs:\n {self.configs}\n"
        s+= f" - useful_functions: select, select_approximate\n"
        return s
