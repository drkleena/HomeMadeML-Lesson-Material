import pandas as pd
from collections import defaultdict as dd
import numpy as np

class NaiveBayes():
    
    priors = dd(int)
    posteriors = {}
    df = pd.DataFrame()
    
    def __init__(self, df):
        self.df = df
        self.priors = self.get_priors()
        self.posteriors = self.get_posteriors()
    
    def get_priors(self):
        df = self.df
        class_freq = dd(int)
        for item in df['class']:
            class_freq[item] += 1 
        sum_ = sum(class_freq.values())
        for item in class_freq.keys():
            class_freq[item] = class_freq[item]/sum_
        return class_freq

    def get_posteriors(self):
        df = self.df
        posterior_dict2 = {}
        class_freq = dd(int)
        for item in df['class']:
            class_freq[item] += 1
        for attribute in list(df)[0:-1]:
            posterior_dict2[attribute] = {}
        dictinct_class_values = df["class"].unique()


        for x in posterior_dict2:
            for values in dictinct_class_values:
                distinct_cell_values = df[x].unique()
                posterior_dict2[x].update({values:{}})
                for things in distinct_cell_values:
                    posterior_dict2[x][values].update({things:0})

        for row in range(len(df.index)):
            for attribute in list(df)[0:-1]:
                # do not contribute missing data to probability counts
                if(posterior_dict2[attribute][df["class"][row]][df[attribute][row]]!="?"):
                    posterior_dict2[attribute][df["class"][row]][df[attribute][row]]+=1
        for key, value in posterior_dict2.items():
            for key1, value1 in value.items():
                for key2, value2 in value1.items():
                    posterior_dict2[key][key1][key2] = value2/class_freq[key1]
        return posterior_dict2
    def test_instance(self,instance, df):
        posterior_dict = self.posteriors
        priors_dict = self.priors
        
        # make epsilon smaller than 1/n
        epsilon = (1/len(df.index))*0.50
        result_list = {}
        # loop through distinct classes
        for cj in df["class"].unique():
            a=1
            # priors x ∏ (cell | class)
            for x in range(len(instance)-1):
                item = instance[x]
                # dont consider instances that are missing
                if(item != "?"):
                    try:
                        a *=posterior_dict[x][cj][instance[x]] 
                    # if we test an instance that we havent seen, perform epsilon smoothing
                    except KeyError:
                        a*=epsilon
            prob = priors_dict[cj] * a
            result_list.update({cj:prob})
    
        return result_list
    
    def predict(self, test_data):
    
        df_results_appended = test_data.copy(deep=True)
        
        df_results_appended["predicted"] = ""
        
        for x in range(len(test_data.index)):

            instance = test_data.values.tolist()[x]
            predicted_dict = self.test_instance(instance, df_results_appended)
            
            max_prob = sorted(predicted_dict, key=predicted_dict.get, reverse = True)[0]
           
            df_results_appended["predicted"].set_value(x, max_prob)
        return df_results_appended