import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Encapsulates a data set and provides functions for exploration.
# Source: https://github.com/MarcRuble/experiment-evaluation
###
class DatasetExploration:
    
    def __init__(self, df):
        self.df = df
        self.order_table = {}
        self.color_table = {}
    
    
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
        self.df[column].replace(dict, inplace=True)
        
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
    def display_sorted(self, column, ascending=True):
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
    def counts_by_property(self, property, index):
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
    
    # Saves a color scheme for a column.
    # colors: ordered list of colors
    # column: string
    def save_colors(self, colors, column=None):
        if column is None:
            column = '__default'
            
        self.color_table[column] = colors
            
            
    #############
    ### PLOTS ###
    #############  

    # Creates a barplot and boxplot next to each other.
    # x: column name for x axis
    # y: column names for y axis (string[])
    # max_y: max value for y axis
    # func: x[] -> y
    # condition: (column:string, value) or list of (column:string, value)
    # x_label: label at x axis
    # y_legend: label in legend (if legend is used, multiple y) for y-axes
    # y_label: label for y-axes
    # axes_color: color string
    # hatches: list of hatch identifiers
    # bar_width: width of the bars
    # file: string path to file location to save plot
    # show: whether to display the output in the notebook
    ###
    # Prerequisites:
    # Order and color scheme for x column must be specified.
    ###
    def bothplot(self, x, y, max_y=None, func=np.mean, condition=False, 
                x_label=None, y_legend=None, y_label=None,
                axes_color='black', hatches=['', '.', '/', '..', '//'], 
                bar_width=None, file=None, show=True):
        plt.rcParams["figure.figsize"] = (8,5)

        plt.subplot(1, 2, 1)
        self.barplot(x, y, max_y=max_y, func=func, condition=condition, 
            x_label=x_label, y_legend=y_legend, y_label=y_label,
            axes_color=axes_color, hatches=hatches, bar_width=bar_width,
            file="bar_{}".format(file) if file is not None else None, show=show)
        plt.subplot(1, 2, 2)
        self.boxplot(x, y, max_y=max_y,condition=condition, 
            x_label=x_label, y_legend=y_legend, y_label=y_label,
            axes_color=axes_color, hatches=hatches, bar_width=bar_width,
            file="box_{}".format(file) if file is not None else None, show=show)

        plt.show()

    
    # Creates a barplot with given parameters.
    # x: column name for x axis
    # y: column names for y axis (string[])
    # max_y: max value for y axis
    # func: x[] -> y
    # condition: (column:string, value) or list of (column:string, value)
    # x_label: label at x axis
    # y_legend: label in legend (if legend is used, multiple y) for y-axes
    # y_label: label for y-axes
    # axes_color: color string
    # hatches: list of hatch identifiers
    # bar_width: width of the bars
    # file: string path to file location to save plot
    # show: whether to display the output in the notebook
    ###
    # Prerequisites:
    # Order and color scheme for x column must be specified.
    ###
    def barplot(self, x, y, max_y=None, func=np.mean, condition=False, 
                x_label=None, y_legend=None, y_label=None,
                axes_color='black', hatches=['', '.', '/', '..', '//'], 
                bar_width=None, file=None, show=True):
                                             
        df = self.df
        if isinstance(y, str):
            y = [y]

        if x_label is None:
            x_label = x
        if y_legend is None:
            y_legend = y

        if isinstance(y_legend, str):
            y_legend = [y_legend]
        
        # collect y values
        x_vals = self.order_table[x]
        y_vals = []
        
        for yi in y:
            # setup new empty list for this y-type
            values = []
            for x_val in x_vals:
                df_c = self.__get_condition(df, condition)
                values.append(func(df_c[df_c[x]==x_val][yi]))
            
            # save list
            y_vals.append(values)
            
        # determine x positions for every y-type
        if bar_width is None:
            bar_width = 1 / len(y) - 0.05
        x_ticks = []
        
        for i in range(0, len(y)):
            if i == 0:
                x_ticks.append(np.arange(len(x_vals)))
            else:
                x_ticks.append([xj + bar_width for xj in x_ticks[i-1]])
        
        # make the plot
        plt.xlabel(x_label)
        plt.xticks([r + (len(y)-1)*(bar_width/2) for r in range(len(x_vals))], x_vals)
        
        if max_y is not None:
            plt.yticks(np.arange(max_y+1))

        for i in range(0, len(y)):
            plt.bar(x_ticks[i], y_vals[i], color=self.__get_colors(x),
                    width=bar_width, edgecolor='white', label=y[i], hatch=hatches[i])
        
        # set colors
        axes = plt.gca()
        axes.spines['bottom'].set_color(axes_color)
        axes.spines['top'].set_color(axes_color)
        axes.spines['left'].set_color(axes_color)
        axes.spines['right'].set_color(axes_color)
        axes.xaxis.label.set_color(axes_color)
        axes.yaxis.label.set_color(axes_color)
        axes.tick_params(axis='x', colors=axes_color)
        axes.tick_params(axis='y', colors=axes_color)
        
        # configure legend or y label
        if len(y_legend) > 1:
            plt.legend(y_legend)
            if y_label != None:
                plt.ylabel(y_label)
        else:
            if y_label != None:
                plt.ylabel(y_label)
            else:
                plt.ylabel(y_legend[0])
                
        # save the plot
        if file != None:
            plt.savefig(file, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        
       
    # Creates a boxplot with given parameters.
    # x: column name for x axis
    # y: column names for y axis (string[]) or single column name
    # max_y: max value for y axis
    # condition: (column:string, value) or list of (column:string, value)
    # x_label: label at x axis
    # y_legend: label in legend (if legend is used, multiple y) for y-axes
    # y_label: label for y-axes
    # axes_color: color string
    # hatches: list of hatch identifiers
    # bar_width: width of the boxes
    # file: string path to file location to save plot
    # show: whether to display the output in the notebook
    ###
    # Prerequisites:
    # Order for x must be specified.
    ###
    def boxplot(self, x, y, max_y=None, condition=False, 
                x_label=None, y_legend=None, y_label=None,
                axes_color='black', hatches=['', '.', '/', '..', '//'], 
                bar_width=None, file=None, show=True):

        df = self.df
        if isinstance(y, str):
            y = [y]

        if x_label is None:
            x_label = x
        if y_legend is None:
            y_legend = y

        if isinstance(y_legend, str):
            y_legend = [y_legend]

        # collect y values
        x_vals = self.order_table[x]
        y_vals = []
        
        for yi in y:
            # setup new empty list for this y-type
            values = []
            for x_val in x_vals:
                df_c = self.__get_condition(df, condition)
                values.append(df_c[df_c[x]==x_val][yi])
            
            # save list
            y_vals.append(values)

        # determine x positions for every y-type
        if bar_width is None:
            bar_width = 1 / len(y) - 0.05
        x_ticks = []
        
        for i in range(0, len(y)):
            if i == 0:
                x_ticks.append(np.arange(len(x_vals)))
            else:
                x_ticks.append([xj + bar_width for xj in x_ticks[i-1]])
        
        # define boxplot median style
        medianprops = dict(linestyle='-', linewidth=1.5, color='firebrick')

        # make the plot
        plt.xlabel(x_label)
        plt.xticks([r + (len(y)-1)*(bar_width/2) for r in range(len(x_vals))], x_vals)
        
        if max_y is not None:
            plt.yticks(np.arange(max_y+1))

        bplots = []
        for i in range(0, len(y)):
            bplots.append(plt.boxplot(y_vals[i], positions=x_ticks[i], widths=bar_width, 
                    patch_artist=True, medianprops=medianprops, manage_ticks=False))
        
        # fill with colors
        for i in range(len(bplots)):
            for patch, color in zip(bplots[i]['boxes'], self.__get_colors(x)):
                patch.set_facecolor(color)
                patch.set(hatch=hatches[i])

        # set colors
        axes = plt.gca()
        axes.spines['bottom'].set_color(axes_color)
        axes.spines['top'].set_color(axes_color)
        axes.spines['left'].set_color(axes_color)
        axes.spines['right'].set_color(axes_color)
        axes.xaxis.label.set_color(axes_color)
        axes.yaxis.label.set_color(axes_color)
        axes.tick_params(axis='x', colors=axes_color)
        axes.tick_params(axis='y', colors=axes_color)

        # configure legend or y label
        if len(y_legend) > 1:
            artists = []
            for i in range(len(bplots)):
                artists.append(bplots[i]['boxes'][0])
            plt.gca().legend(artists, y_legend)
            if y_label != None:
                plt.ylabel(y_label)
        else:
            if y_label != None:
                plt.ylabel(y_label)
            else:
                plt.ylabel(y_legend[0])
        
        # save the plot
        if file != None:
            plt.savefig(file, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        
                    
    
    ##############
    ### HELPER ###
    ##############
    
    # Returns all possible values in a column.
    # column: string
    def __possible_values(self, column):
        return pd.unique(self.df[column])
    
    # Returns the specified color scheme or a default.
    # column: string
    def __get_colors(self, column):
        if column in self.color_table:
            return self.color_table[column]
        else:
            return self.color_table['__default']
        
    # Returns the rows which fulfills the condition.
    # data: dataframe
    # condition: (column:string, value) or list of (column:string, value)
    def __get_condition(self, data, condition):
        if not condition:
            return data
        elif isinstance(condition, list):
            for single_cond in condition:
                data = self.__get_condition(data, single_cond)
            return data
        else:
            return data[data[condition[0]]==condition[1]]