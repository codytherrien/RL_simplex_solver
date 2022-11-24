import pandas as pd
import numpy as np
import random
import math
import warnings
warnings.filterwarnings("ignore")

def generate_LP():
    # Generates LP for training if no LP is given
    variables = random.randint(2,100)
    constraints = random.randint(2,100)

    lp = np.random.random_integers(low=-500, high=500, size=(variables, constraints))
    lp[0][0] = 0

    return lp

class Simplex:
    def __init__(self, file=None):
        self.df = self.load_lp(file)
        self.entering_column = None
        self.leaving_row = None
        self.origional_vector = self.df.iloc[0].copy()
        self.solution = None
        self.reward = 0

    def load_lp(self, file):
        if file is None:
            df = pd.DataFrame(generate_LP(), dtype=float)
        else:
            lines = file.readlines()
            lines[0] = lines[0].strip() + ' 0'  
            temp_array = []
            for line in lines:
                line = line.strip().split()
                line = [line[-1]] + line[:-1]
                temp_array.append(line)

            df = pd.DataFrame(temp_array, dtype=float)

        df.columns = ['x' +str(col) for col in df.columns]
        df.index = ['z' + str(row) for row in df.index]
        df.iloc[1:,1:] = df.iloc[1:,1:]*-1

        return df

    def dansig_get_entering_column(self):
        self.entering_column = int(self.df.iloc[0, 1:].argmax() + 1)

    def bland_get_entering_column(self):
        temp = self.df.iloc[0,1:].copy()
        temp.drop(temp[temp <= 0].index, inplace=True)
        temp = temp.index

        if min(temp) == 'Oh':
            temp.remove('Oh')
        if min(temp).startswith('x'):
            possible_entering = []
            for word in temp:
                if word.startswith('x'):
                    possible_entering.append(int(float(word[1:])))
            bland_entering = 'x' + str(min(possible_entering))
        else:
            possible_entering = []
            for word in temp:
                possible_entering.append(int(float(word[1:])))
            bland_entering = 'z' + str(min(possible_entering))

        self.entering_column = self.df.columns.get_loc(bland_entering)
    
    def largest_increase_get_entering_column(self):
        temp = self.df.iloc[1:,1:].copy()
        temp[temp >= 0] = -9999999999
        temp = temp.divide(self.df.iloc[1:,0], axis=0)
        temp = temp.max(axis=0)
        temp = pd.concat([self.df.iloc[0, 1:] > 0, temp], axis=1)
        temp = temp.iloc[:,0]*temp.iloc[:,1]

        self.entering_column = temp.argmin() + 1

    def get_leaving_row(self):
        temp = self.df.iloc[1:,0] / self.df.iloc[1:,self.entering_column]
        if min(temp) > 0:
            self.solution = 'unbounded'
            self.reward = 1
        else:
            temp.loc[(temp > 0),] = -999999999999 

            if max(temp) == 0 and self.df.iloc[int(temp.argmax() + 1),self.entering_column] > 0:
                potential_leaving = self.df.iloc[1:,self.entering_column][self.df.iloc[1:,self.entering_column] < 0].index
                
                try:
                    leaving_id = temp[potential_leaving].idxmax()
                    self.leaving_row = self.df.index.get_loc(leaving_id)
                except:
                    self.solution = 'unbounded'
                    self.reward = 1
                
            else:
                self.leaving_row = int(temp.argmax() + 1)

    def pivot_row_equation(self):
        divisor = -self.df.iloc[self.leaving_row, self.entering_column]
        self.df.iloc[self.leaving_row, self.entering_column] = -1
        self.df.iloc[self.leaving_row] = self.df.iloc[self.leaving_row] / divisor
    
    def non_pivot_rows_equation(self):
        multiplyer = self.df.iloc[:self.leaving_row, self.entering_column].copy()
        self.df.iloc[:self.leaving_row, self.entering_column] = 0
        try:
            self.df.iloc[:self.leaving_row] = self.df.iloc[:self.leaving_row] + \
                np.outer(multiplyer, self.df.iloc[self.leaving_row])
        except: 
            self.df.iloc[:self.leaving_row] = self.df.iloc[:self.leaving_row] + \
                self.df.iloc[self.leaving_row]*multiplyer

        multiplyer = self.df.iloc[self.leaving_row+1:, self.entering_column].copy()
        self.df.iloc[self.leaving_row+1:, self.entering_column] = 0
        try:
            self.df.iloc[self.leaving_row+1:] = self.df.iloc[self.leaving_row+1:] + \
                np.outer(multiplyer, self.df.iloc[self.leaving_row])
        except: 
            self.df.iloc[self.leaving_row+1:] = self.df.iloc[self.leaving_row+1:] + \
                self.df.iloc[self.leaving_row]*multiplyer
        self.df[abs(self.df) < 1e-10] = 0

    def naive_pivot(self):
        if max(self.df.iloc[0,1:]) <= 0:
            self.solution = 'optimal'
            self.reward = 1
        
        else:
            if min(self.df.iloc[1:,0]) == 0:
                self.bland_get_entering_column()
            else:
                self.largest_increase_get_entering_column()
            self.get_leaving_row()
            self.pivot_row_equation()
            self.non_pivot_rows_equation()
            self.df.rename(
                columns={self.df.columns[self.entering_column]: self.df.index[self.leaving_row]}, 
                index={self.df.index[self.leaving_row]: self.df.columns[self.entering_column]}, 
                inplace=True
            )
            if max(self.df.iloc[0,1:]) <= 0:
                self.solution = 'optimal'
                self.reward = 1

    def rl_pivot(self, pivot):
        if max(self.df.iloc[0,1:]) <= 0:
            self.solution = 'optimal'
            self.reward = 1

        else:    
            self.reward = 0
            if pivot == 0:
                self.bland_get_entering_column()
            elif pivot == 1:
                self.dansig_get_entering_column()
            elif pivot == 2:
                self.largest_increase_get_entering_column()
            else:
                # This is only here if model breaks, which hasnt happend in testing
                self.bland_get_entering_column()
            self.get_leaving_row()
            self.pivot_row_equation()
            self.non_pivot_rows_equation()
            self.df.rename(
                columns={self.df.columns[self.entering_column]: self.df.index[self.leaving_row]}, 
                index={self.df.index[self.leaving_row]: self.df.columns[self.entering_column]}, 
                inplace=True
            )

            if max(self.df.iloc[0,1:]) <= 0:
                self.solution = 'optimal'
                self.reward = 1

        return self.df, self.reward

    def get_auxiliary_entering_column(self):
        self.entering_column = self.df.columns.get_loc('Oh')

    def get_auxiliary_leaving_row(self):
        self.leaving_row = self.df.iloc[:,0].argmin()

    def auxiliary_pivot(self):
        self.get_auxiliary_entering_column()
        self.get_auxiliary_leaving_row()
        self.pivot_row_equation()
        self.non_pivot_rows_equation()
        self.df.rename(
            columns={self.df.columns[self.entering_column]: self.df.index[self.leaving_row]}, 
            index={self.df.index[self.leaving_row]: self.df.columns[self.entering_column]}, 
            inplace=True
        )

    def auxiliary_setup(self):
        self.df.iloc[0] = 0
        self.df['Oh'] = 1
        self.df['Oh'][0] = -1
        self.auxiliary_pivot()

    def remove_auxiliary_column(self):
        self.solution = None
        self.df.drop('Oh', axis=1, inplace=True)

        for key, val in self.origional_vector.iteritems():
            if key in self.df.columns:
                self.df[key][0] += val
            else:
                self.df.iloc[0] += self.df.loc[key]*val

    def get_oh_entering_column(self):
        possible_columns = abs(self.df.iloc[0,1:]) < 1e-20
        possible_columns = possible_columns[possible_columns].index.values

        possible_columns = abs(self.df.loc['Oh'][possible_columns]) > abs(self.df.loc['Oh'][0])
        possible_columns = possible_columns[possible_columns].index.values
        possible_columns = abs(self.df.loc['Oh'][1:]) > abs(self.df.loc['Oh'][0])
        possible_columns = possible_columns[possible_columns].index.values
        col_name = self.df[possible_columns].iloc[0].abs().idxmin()

        self.entering_column = self.df.columns.get_loc(col_name)
    
    def smallest_increase_get_entering_column(self):
        # This is only used in auxiliary problem when omega is in basis
        temp = self.df.iloc[1:,1:].copy()
        temp = temp.divide(self.df.iloc[1:,0], axis=0)
        temp = temp.max(axis=0)
        temp = pd.concat([self.df.iloc[0, 1:] > 0, temp], axis=1)
        temp = temp.iloc[:,0]*temp.iloc[:,1]
        self.entering_column = temp.argmin() + 1

    def get_oh_leaving_column(self):
        self.leaving_row = self.df.index.get_loc('Oh')
    
    def final_auxiliary_pivot(self):
        self.get_oh_entering_column()
        self.get_oh_leaving_column()
        self.pivot_row_equation()
        self.non_pivot_rows_equation()
        self.df.rename(
            columns={self.df.columns[self.entering_column]: self.df.index[self.leaving_row]}, 
            index={self.df.index[self.leaving_row]: self.df.columns[self.entering_column]}, 
            inplace=True
        )

    def check_auxiliary(self):
        if abs(self.df.iloc[0,0]) > 1e-15:
            self.solution = 'infeasible'
            self.reward = 1
        elif self.solution == 'optimal' and 'Oh' in self.df.columns:
            self.remove_auxiliary_column()
        elif self.solution == 'optimal':
            self.final_auxiliary_pivot()
            self.remove_auxiliary_column()

    def get_observation(self):
        return self.df
    
    def get_results(self):
        endpoint = ' '
        print(self.solution)
        if self.solution == 'optimal':
            if self.df.iloc[0,0] == 0:
                rounded_output = 0
            else:
                # This rounding is done to reproduce results format given in test cases
                rounded_output =  round(self.df.iloc[0,0], 7 - int(math.floor(math.log10(abs(self.df.iloc[0,0])))) - 1)
            if rounded_output % 1 == 0:
                print(f"{int(rounded_output)}")
            else:
                print(f"{self.df.iloc[0,0]:.7}")
            optimization_variables = {}
            for col in self.df.iloc[:,1:]:
                if col.startswith('x'):
                    optimization_variables[col] = 0
            for i, row in self.df.iterrows():
                if i.startswith('x'):
                    optimization_variables[i] = row[0]
            for i in range(1,len(optimization_variables) + 1):
                x = float(optimization_variables[f'x{i}'])
                if i == len(optimization_variables):
                    endpoint = ''
                if x == 0:
                    rounded_x = 0
                else:
                    rounded_x =  round(x, 7 - int(math.floor(math.log10(abs(x)))) - 1)
                if rounded_x % 1 == 0:
                    print(int(rounded_x), end= endpoint)
                else:
                    print(rounded_x, end= endpoint)
            print()