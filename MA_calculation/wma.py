# Function for weighted moving average calculation
import pandas as pd
from scipy.stats import norm
from scipy.signal.windows import boxcar
from matplotlib import pyplot as plt

# PARAMETERS: 
#   df (type: pandas.DataFrame) - Time series pandas dataframe
#   end_date (type: <class 'datetime.date'>) (optional, default=None) - Provide the end date of the data. If None, reindex dates between min and max date range. Otherwise, use date provided
#   date_col (type: <class 'str'>) (optional, default='date') - Name of column containing date
#   grouping (type: <class 'list'>) (optional, default=['Ticker', 'COMPANY']) - List of columns to use groupby() function on
# RETURNS:
#   df (type: pandas.DataFrame) - Returns time series pandas dataframe with complete range of dates
# USE:
# This allows our Weighted Moving Average function to fill-in the last available moving average value on missing dates
def fill_dates(df, end_date=None, date_col='date', grouping=['Ticker','COMPANY']):
    # Convert date column to a pandas datetime index
    df[date_col] = pd.DatetimeIndex(df[date_col])

    # Set the datetime index as the index for the data (Needed to apply reindex function)
    df = df.set_index(date_col)

    # Set end_date
    end_date = df.index.max() if end_date == None else end_date

    # Generate a complete list of dates between the starting and the end date
    idx = pd.date_range(df.index.min(), end_date)

    # Group data based on a given criteria as same date ranges will be repeated for all groups
    # For instance - Each company in the SDG data has ticks from feb 19 2015 to present day
    #               So, for each company we need to fill in the missing dates
    # Then, apply() is used to apply reindex() function on each subset ('Companies')
    # reindex() adds the missing dates as index and places NaNs for all coluumns
    df = df.groupby(by=grouping).apply(lambda x:x.reindex(idx).rename_axis(date_col)).drop(grouping,axis=1).reset_index()
    return df

# PARAMETERS:
#   sigma (type: <class 'float'>) (optional, default=2.5) - Scale for the scipy libraries norm.pdf function
#   size (type: <class 'int'>) (optional, default=7) - The size for normal weights window
# RETURNS:
#   default_weight (type: <class 'list'>) - Returns normal weight window as a list
# USE:
# Generates window of the size specified, which is used for rolling window moving average
def window_weights(sigma=2.5, size=7):
    # Generate weight window using norm.pdf() of the specified size and scale
    default_weight = [norm.pdf(i-(size-1), scale = sigma) for i in range(size)]

    # Normalize the weight so they sum up to one
    default_weight = default_weight/sum(default_weight)
    return default_weight

# PARAMETERS:
#   dataframe (type: pandas.DataFrame) - DataFrame containing columns for which weighted moving average is to be calculated
#   ma_col (type: <class 'list'>) - List of columns for which weighted moving average is needed
#   window_size (type: <class 'int'>) (optional, default=7) - Size parameter for the window_weight() function
#   window_sigma (type: <class 'float'>) (optional, default=2.5) - Sigma parameter for the window_weight() function
#   minimum_periods (type <class 'int'>) (optional, default=7) - Minimum number of days required to calculate moving average. Less than or equal to window size
#   extend (type: <class 'bool'>) (optional, default=True) - Specify whether to add to the existing dataframe as columns or return new
#   merge_col (type: <class 'list'>) (optional, default=['COMPANY']) - Columns to use as basis for grouping datasets
# RETURNS:
#   dataframe (type: pandas.DataFrame) - If extend parameter is True, returns original dataframe with additional moving average columns
#   SDGi (type: pandas.DataFrame) - If sxtend parameter is False, returns a dataframe containing moving averages
#   company_nan (type: pandas.DataFrame) - DataFrame consisting of information on number of days it takes to get the first moving average for all companies
# USE:
# Calculates weighted moving average for mutiple columns with specifications for the window weights
def weighted_MA(dataframe, ma_col, window_size=7, window_sigma=2.5, minimum_periods=7, extend=True, merge_col=['COMPANY']):
    #Create adataframe with Company names to add calculation results to
    #Getting index is important for reference
    #Comapany names are required for grouping dataframe
    SDGi = pd.DataFrame(dataframe[merge_col], index=dataframe.index)

    # Calculate 7-day MA of each z-score:
    default_weight = window_weights(sigma=window_sigma, size=window_size)

    #Dataframe to store number of days needed to get first rating for a company
    company_nan = pd.DataFrame(dataframe[merge_col[0]].unique(), columns=merge_col)
    company_nan.set_index(merge_col, inplace=True)

    print('Calculating ' + str(window_size) + '-Day moving average.')
    #Calculating moving average for all the columns
    for i in range(1,len(ma_col)+1):
        # Create name for the new column
        col_str = 'MA_'+str(window_size)+'day_'+str(i) if len(ma_col)>1 else 'MA_'+str(window_size)+'day'

        # Grouping dataframe by companies so we can use a rolling window over individual datasets
        # Calculate moving average using rolling function of pandas dataframe
        # We use notnull() as we need 7/180 days of news to calculate moving average
        # 'boxcar' window type allows us to use our custom window
        # min_periods specifies how many entries are needed at a minimum for computation
        SDGi[col_str] = dataframe[dataframe[ma_col[i-1]].notnull()].groupby(merge_col).apply(lambda x : \
            x[ma_col[i-1]].rolling(window=default_weight, min_periods=minimum_periods, win_type='boxcar').mean()).reset_index(0,drop=True)
        
        #Fill days without news with last time moving average was available
        SDGi[col_str]=SDGi.groupby(merge_col, sort=False)[col_str].ffill()
        
        #Store number of days it takes to get MA
        holder = pd.DataFrame(SDGi.groupby(merge_col, sort=False).apply(lambda x : x[col_str].isna().sum()), columns=[col_str])
        company_nan=pd.concat([company_nan, holder], axis=1, sort=False)

    if extend:
        # Append moving averages to the original dataframe
        dataframe = pd.concat([dataframe, SDGi.drop(columns=merge_col)], axis=1)
        return dataframe, company_nan
    else:
        # Return a seperate dataframe with the weighted moving averages
        return SDGi, company_nan

# PARAMETERS:
#   data (type: pandas.DataFrame) - DataFrame containing columns to be rescaled
#   cols (type: <class 'list'>) - List of columns to be scaled
#   binning (type <class 'bool'>) (optional, default=True) - Whether to rescale using bins or rescale based on whole data
#   bin_col (type <class 'str'>) (optional, default='Bin') - Name of column with bins
#   date_col (type: <class 'str'>) (optional, default='date') - Name of the date column to be used for filtering data
#   lookback (type: <class 'int'>) (optional, default=5) - Number of past years to be used to calculate priors for the present year
#   initial_years (type: <class 'int'>) (optional, default=3) - Number of initial years to that are in sample. This number should be less than the lookback period.
# RETURNS:
#   data (type: pandas.DataFrame) - Returns dataframe with rescaled columns
# USE:
# Rescales given columns to make standard deviation 1 with or without binning such that different columns are comparable
def rescale(data, cols, binning=True, bin_col='Bin', date_col='date', lookback=5, initial_years=3):
    # Get moving scaling factor
    scalar_yr = scalar_yearly(data, cols, binning=binning, bin_col=bin_col, date_col=date_col, lookback=lookback, initial_years=initial_years)

    # Apply scaling based on bins
    data[cols] = data[cols]*scalar_yr.values
    # data[cols] = data.groupby(id_col).apply(lambda x : x[cols]*scalar_yr[x[date_col].dt.year].loc[scalar_yr[x[date_col].dt.year].index==x.Bin.unique()[0], cols].values if not (scalar_yr[x[date_col].dt.year][scalar_yr[x[date_col].dt.year].index==x.Bin.unique()[0]].empty) else x[cols])

    return data

# PARAMETERS:
#   data (type: pandas.DataFrame) - DataFrame containing columns for which standard deviation is to be calculated
#   cols (type: <class 'list'>) - List of columns for which standard deviation is to be calculated
#   bin_col (type <class 'str'>) (optional, default='Bin') - Name of the column with bins
# RETURNS:
#   stats (type: pandas.DataFrame) - Dataframe of standard deviations of select columns for each bin
# USE:
# Get standard deviation of given ccolumns with bins. This function can also display the standard deviations for the user
def std_binned(data, cols, bin_col='Bin'):
    # Calculate statistics for each bin
    stats = data.groupby(bin_col)[cols].std()

    # Display statistics to user
    # fig = plt.figure(figsize=(16, 4))
    # cell_text = []
    # for row in range(len(stats)):
    #     cell_text.append(stats.iloc[row].round(4))
    
    # table = plt.table(cellText=cell_text, colLabels=stats.columns, rowLabels=stats.index, loc='center')
    # table.auto_set_font_size(False)
    
    # #table.auto_set_column_width(col=[i+1 for i in range(len(cols))])
    # table.set_fontsize(6)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    # for pos in ['right','top','bottom','left']:
    #     plt.gca().spines[pos].set_visible(False)
    # # plt.savefig(name_file)
    # plt.show()
    # plt.close()
    return stats

# PARAMETERS:
#   data (type: pandas.DataFrame) - DataFrame containing columns for which standard deviation is to be calculated
#   cols (type: <class 'list'>) - List of columns for which standard deviation is to be calculated
#   binning (type <class 'bool'>) (optional, default=True) - Whether to rescale using bins or rescale based on whole data
#   bin_col (type <class 'str'>) (optional, default='Bin') - Name of the column with bins
#   date_col (type: <class 'str'>) (optional, default='date') - Name of the date column to be used for filtering data
#   lookback (type: <class 'int'>) (optional, default=5) - Number of past years to be used to calculate priors for the present year
#   initial_years (type: <class 'int'>) (optional, default=3) - Number of initial years to that are in sample. This number should be less than the lookback period.
# RETURNS:
#   scalar_yr (type: pandas.DataFrame) - DataFrame of the size of given columns, containing multiplier for scaling
# USE:
# Get scaling factors yearly based on the standard deviation, looking back given number of years.
def scalar_yearly(data, cols, binning=True, bin_col='Bin', date_col='date', lookback=5, initial_years=3):
    # Initialize
    scalar_yr = {}
    t = pd.DataFrame(index=data.index)
    start_yr = data[date_col].dt.year.min()
    end_yr = data[date_col].dt.year.max()
    
    initial_range = [start_yr + i for i in range(initial_years)]
    temp1 = data[data[date_col].dt.year.isin(initial_range)]
    
    if binning:
        temp2 = std_binned(temp1, cols, bin_col=bin_col)
        scalar = 1/temp2
    
        # Add next year out of sample
        initial_range.append(initial_range[-1]+1)
    
        for year in initial_range:
            scalar_yr[year] = scalar

        # Now, for the remaining period
        for year in range(initial_range[-1]+1, end_yr+1):
            new_range = [j for j in range(max(start_yr, year-lookback), year)]

            temp1 = data[data[date_col].dt.year.isin(new_range)]
            temp2 = std_binned(temp1, cols, bin_col=bin_col)
            scalar = 1/temp2
            scalar_yr[year] = scalar
            
        scalar_yr = pd.concat(scalar_yr).reset_index().rename({'level_0':'Year'}, axis=1)
        
        for c in cols:
            t[c] = pd.Series(list(zip(data[date_col].dt.year, data[bin_col]))).map(dict(zip(zip(scalar_yr.Year, scalar_yr[bin_col]), scalar_yr[c])))
        
    else:
        temp2 = temp1[cols].std()
        scalar = 1/temp2
        
        # Add next year out of sample
        initial_range.append(initial_range[-1]+1)
    
        for year in initial_range:
            scalar_yr[year] = scalar

        # Now, for the remaining period
        for year in range(initial_range[-1]+1, end_yr+1):
            new_range = [j for j in range(max(start_yr, year-lookback), year)]

            temp1 = data[data[date_col].dt.year.isin(new_range)]
            temp2 = temp1[cols].std()
            scalar = 1/temp2
            scalar_yr[year] = scalar
            
        scalar_yr = pd.DataFrame.from_dict(scalar_yr).T
        
        for c in cols:
            t[c] = data[date_col].dt.year.map(dict(zip(scalar_yr.index, scalar_yr[c])))

    return t