import numpy as np
import pandas as pd
import math

from scipy.stats import shapiro
from scipy.stats import bartlett
from scipy.stats import friedmanchisquare
from pingouin import sphericity
from pingouin import rm_anova
from pingouin import wilcoxon


# Encapsulates a data set and provides functions for evaluation.
class DatasetEvaluation:
    
    def __init__(self, df):
        self.df = df
        self.alpha = 0.05
        self.order_table = {}
    
    
    ####################
    ### MANIPULATION ###
    ####################
    
    # Excludes data which fulfills the condition.
    # condition: (column:string, value)
    def exclude(self, condition):
        self.df = self.df[self.df[condition[0]] != condition[1]]
        
    # Replaces values in a column.
    # column: string
    # dict: old -> new value
    def replace(self, column, dict):
        self.df[column].replace(dict)
        
    # Adds a new column which is the mean of given columns.
    # columns: list of strings
    # name: for new column
    def add_mean(self, columns, name):
        self.df[name] = self.df[columns].mean(axis=1)
        
     
    ###################
    ### QUICK STATS ###
    ###################
    
    # Displays the data set as data frame.
    def display(self):
        df = self.df
        display(df)
        
    # Displays the data set in a sorted fashion.
    # column: string
    # ascending: bool
    def display_sorted(self, column, ascending):
        df = self.df
        display(df.sort_values(column, ascending=ascending))
        
    # Returns the mean value of a column.
    # column: string
    def mean(self, column):
        df = self.df
        return df[column].mean()
        
    # Returns the std value of a column.
    # column: string
    def std(self, column):
        df = self.df
        return df[column].std()
        
    # Returns the count of a property's values.
    # property: column to count the available values
    # index: column to use as index
    def countsByProperty(self, property, index):
        df = self.df
        return df.groupby([property]).count()[index]
        
       
    ##############
    ### CONFIG ###
    ##############
    
    # Saves an order of values for a column.
    # column: string
    # order: ordered list of values
    def save_order(self, column, order):
        self.order_table[column] = order

    # Sets the alpha value used for hypothesis testing.
    # alpha: new value
    # default is 0.05
    def set_alpha(self, alpha):
        self.alpha = alpha


    ##############
    ### CHECKS ###
    ##############

    # Checks for a normal distribution of the values in a column
    # which fulfill a given condition.
    # column: string
    # condition: (column:string, value)
    # display_result: bool if the result should be displayed
    # returns test statistic, p value
    ###
    # Uses scipy.stats.shapiro test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
    # "The Shapiro-Wilk test tests the null hypothesis that the data was 
    # drawn from a normal distribution."
    ###
    def check_normal_distribution(self, column, condition=None, display_result=True):
        data = self.__get_condition(self.df, condition)[column]
        stat, p = shapiro(data)
        if display_result:
            print("### Normal Distribution ###")
            if not condition:
                print("{0:}: stat={1:.5}, p={2:.5}".format(column, stat, p))
            else:
                print("{0:} with ({1:} = {2:}): stat={3:.5}, p={4:.5}".format( 
                        column, condition[0], condition[1], stat, p))
            if p > self.alpha:
                print('--> Gaussian-like')
            else:
                print('--> Non-Gaussian')
            print("")

        return stat, p


    # Checks for homogene variances of the values in a column
    # separated into groups depending on values in group column.
    # value_col: string for column with values
    # group_col: string for column with groups/conditions to compare
    # display_result: bool if the result should be displayed
    # returns test statistic, p value
    ###
    # Uses scipy.stats.bartlett test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html
    # "Bartlett’s test tests the null hypothesis that all input samples
    # are from populations with equal variances."
    ###
    def check_homogene_variances(self, value_col, group_col, display_result=True):
        # collect data
        data = self.__get_condition_sets(self.df, value_col, group_col)

        # perform test
        stat, p = bartlett(*data)
        if display_result:
            print("### Homogeneity of Variances ###")
            print("{0:}: stat={1:.5}, p={2:.5}".format(value_col, stat, p))
            if p > self.alpha:
                print('--> Homogene Variances')
            else:
                print('--> Non-Homogene Variances')
            print("")

        return stat, p

    # Checks sphericity of the values in a column
    # separated into groups/conditions and individuals.
    # value_col: string for column with values
    # group_col: string for column with groups/conditions to compare
    # subject_col: string for column with subject/participant ids
    # display_result: bool if the result should be displayed
    # returns spher (bool), W test statistic, chi2 effect size, dof, p value
    ###
    # Uses pingouin.sphericity test
    # https://pingouin-stats.org/generated/pingouin.sphericity.html
    # "Mauchly and JNS test for sphericity."
    ###
    def check_sphericity(self, value_col, group_col, subject_col, display_result=True):
        # perform test
        spher, W, chi2, dof, p = sphericity(self.df, value_col, group_col, subject_col)

        if display_result:
            print("### Sphericity ###")
            print("{0:} between {1:} for {2:}: W={3:.5}, chi2={4:.5}, dof={5:}, p={6:.5}".format(
                value_col, group_col, subject_col, W, chi2, dof, p))
            if spher:
                print('--> Sphericity given')
            else:
                print('--> No sphericity given')
            print("")

        return spher, W, chi2, dof, p


    #################################
    ### OVERALL VARIANCE ANALYSIS ###
    #################################

    # Compares the values obtained in different groups/conditions
    # with a Friedman test. Ignore data not matching the given condition.
    # Use this for repeated-measures data without normal distribution.
    # https://yatani.jp/teaching/doku.php?id=hcistats:kruskalwallis
    # With following assumptions:
    # https://accendoreliability.com/non-parametric-friedman-test/
    ###
    # value_col: string for column with values
    # group_col: string for column with groups/conditions to compare
    # condition: (column:string, value)
    # display_result: bool if the result should be displayed
    # returns test statistic, p value
    ###
    # Uses scipy.stats.friedmanchisquare test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html
    # "Compute the Friedman test for repeated measurements.
    # The Friedman test tests the null hypothesis that repeated
    # measurements of the same individuals have the same distribution."
    ###
    def friedman_test(self, value_col, group_col, condition=False, display_result=True):
        # collect data
        data = self.__get_condition_sets(self.__get_condition(self.df, condition), value_col, group_col)

        # perform test
        stat, p = friedmanchisquare(*data)
        if display_result:
            print("################")
            print("### Friedman ###")
            print("################")
            if not condition is False:
                print(self.__condition_to_string(condition))
            print("{0:} between {1:}: stat={2:.5}, p={3:.5}".format(
                value_col, group_col, stat, p))
            if p > self.alpha:
                print('--> No significant effects')
            else:
                print('--> Significant effects')
            print("")

        return stat, p


    # Compares the values obtained in different groups/conditions
    # with an ANOVA repeated-measures test. Ignores data not matching
    # the given condition.
    # Use this for repeated-measures data with normal distribution.
    # https://yatani.jp/teaching/doku.php?id=hcistats:anova
    ###
    # value_col: string for column with values
    # group_col: string for column with groups/conditions to compare
    # subject_col: string for column with subject/participant ids
    # condition: (column:string, value)
    # display_result: bool if the result should be displayed
    # returns a summary table with attributes like:
    # F (test statistic), p-unc, p-GG-corr (for lack of sphericity),
    # n2 (effect size)
    ###
    # Uses pingouin.rm_anova test
    # https://pingouin-stats.org/generated/pingouin.rm_anova.html
    ###
    def anova_test(self, value_col, group_col, subject_col, condition=False, display_result=True):
        # collect data
        data = self.__get_condition(self.df, condition)
        
        # perform test
        summary = rm_anova(data, value_col, group_col, subject_col, correction=True, effsize='n2')
        if display_result:
            print("#############")
            print("### ANOVA ###")
            print("#############")
            if not condition is False:
                print(self.__condition_to_string(condition))
            display(summary)
            print("")

        return summary


    ######################
    ### POST-HOC TESTS ###
    ######################

    # Compares subgroups of the data to each other and determines
    # how significant their differences are.
    # If baseline parameter is given, all groups only compared to baseline.
    # Else all groups are compared pairwise with each other.
    # Use this as a post-hoc when Friedman's test for repeated-measures
    # data without normal distributions hints at significant differences.
    # https://yatani.jp/teaching/doku.php?id=hcistats:kruskalwallis#post-hoc_test
    # Appropriate for paired/dependant samples:
    # https://yatani.jp/teaching/doku.php?id=hcistats:wilcoxonsigned
    # With following assumptions:
    # https://www.statisticssolutions.com/assumptions-of-the-wilcox-sign-test/
    ###
    # value_col: string for column with values
    # group_col: string for column with groups/conditions to compare
    # condition: (column:string, value)
    # baseline: optional value of group_col treated as a baseline
    # display_result: bool if the result should be displayed
    # file: string path to location if csv should be saved
    # returns a summary table with content depending on baseline
    ###
    # Uses pingouin.wilcoxon:
    # https://pingouin-stats.org/generated/pingouin.wilcoxon.html
    # "The Wilcoxon signed-rank test [1] tests the null hypothesis 
    # that two related paired samples come from the same distribution."
    ###
    def wilcoxon_test(self, value_col, group_col, condition=False, 
                     baseline=None, display_result=True, file=None):
        # collect data
        df = self.__get_condition(self.df, condition)

        # collect group values to compare
        groups = self.__ordered_values(group_col)

        # baseline gets special treatment
        if baseline is not None:
            groups = [x for x in groups if x != baseline]
            groups.append(baseline)

        # setup dict to construct dataframe
        if baseline == None:
            results = {'A':[], 'B':[], 'W':[], 'p':[], 'bonf':[], 'RBC':[], 'CLES':[]}
        else:
            results = {}
            for group in groups:
                results[group] = []

        # collect all pairs to compare
        to_compare = []
        for g1 in groups:
            for g2 in groups:
                if g1 != g2 and not (g2,g1) in to_compare:
                    to_compare.append((g1,g2))

        # compute results
        if baseline == None:
            # compare all groups to each other
            for (g1, g2) in to_compare:
                # perform wilcoxon
                s1 = df[df[group_col]==g1][value_col]
                s2 = df[df[group_col]==g2][value_col]
                stats = wilcoxon(s1, s2)
                # read results
                W = stats['W-val'].values[0]
                p = stats['p-val'].values[0]
                bonf = self.__apply_bonferroni(p, len(to_compare))
                rbc = stats['RBC'].values[0]
                cles = stats['CLES'].values[0]
                # results
                results['A'].append(g1)
                results['B'].append(g2)
                results['W'].append(W)
                results['p'].append(self.__check_p(p))
                results['bonf'].append(self.__check_p(bonf))
                results['RBC'].append(round(rbc, 5))
                results['CLES'].append(round(cles, 5))
            
            # create dataframe
            df_res = pd.DataFrame(results)

        else:
            # only compare to baseline
            for (g1, g2) in to_compare:
                # check if this is compared to baseline
                if g2 != baseline:
                    continue
                
                # perform wilcoxon
                s1 = df[df[group_col]==g1][value_col]
                s2 = df[df[group_col]==g2][value_col]
                stats = wilcoxon(s1, s2)
                # read results
                W = stats['W-val'].values[0]
                p = stats['p-val'].values[0]
                bonf = self.__apply_bonferroni(p, len(groups))
                rbc = stats['RBC'].values[0]
                cles = stats['CLES'].values[0]
                # results
                results[g1].append(self.__check_p(p))
                results[g1].append(self.__check_p(bonf))
                results[g1].append(W)
                results[g1].append(round(rbc, 5))
            
            df_res = pd.DataFrame(results, index=pd.Index(['p', 'bonf', 'W', 'r'], name='value'), columns=pd.Index(['XXS', 'XS', 'S', 'L', 'XL', 'XXL'], name='group'))

        if display_result:
            print("################")
            print("### Wilcoxon ###")
            print("################")
            if not condition is False:
                print(self.__condition_to_string(condition))
            display(df_res)
            print("")

        if file is not None:
            df_res.to_csv(file)

    # TODO
    def paired_t_test(self):
        pass

    ##############
    ### HELPER ###
    ##############
    
    # Returns all possible values in a column ordered.
    # column: string
    def __ordered_values(self, column):
        if column in self.order_table:
            return self.order_table[column]
        else:
            return self.__possible_values(column)

    # Returns all possible values in a column.
    # column: string
    def __possible_values(self, column):
        return pd.unique(self.df[column])
        
    # Returns the rows which fulfills the condition.
    # data: dataframe
    # conditions: (column:string, value)
    def __get_condition(self, data, condition):
        if not condition:
            return data
        else:
            return data[data[condition[0]]==condition[1]]

    # Returns a string representation of a condition.
    # condition: (column:string, value)
    def __condition_to_string(self, condition):
        return "{} = {}".format(condition[0], condition[1])

    # Returns the subset of values from a value column
    # separated by their value a group column.
    # data: dataframe
    # value_col: string for column with values
    # group_col: string for column with groups/conditions to compare
    # condition: (column:string, value)
    def __get_condition_sets(self, data, value_col, group_col):
        result = []
        groups = self.__ordered_values(group_col)
        for group in groups:
            result.append(self.__get_condition(data, (group_col, group))[value_col])
            
        return result

    # Returns the bonferroni corrected version of a p-value.
    # p: p-significance value
    # num: number of comparisons besides (including) this
    def __apply_bonferroni(self, p, num):
        return min(p*num, 1)

    # Returns an annotated version of a p-value.
    # p: p-significance value
    def __check_p(self, p):
        if type(p) == str:
            return p
        elif abs(p) <= 0.001:
            return str(round(p, 5)) + " ***"
        elif abs(p) <= 0.01:
            return str(round(p, 3)) + " **"
        elif abs(p) <= 0.05:
            return str(round(p, 3)) + " *"
        else:
            return str(round(p, 5))