# Get the necessary libraries to work with data
import pandas as pd
import numpy as np

# # Store parent and data directory
# parent = os.path.dirname(os.path.dirname(os.getcwd()))+'/'
# sys.path.append(parent + 'Attribution_Compiled/Python')

import wma
# import attrib
import bayesian
from matplotlib import pyplot as plt

# Get runtime variables
# variables = pd.read_csv(path + 'combine_v1v2_Fiona/SPY/Readme.txt', nrows=2, names=['Variable', 'Value'], skipinitialspace=True)
# variables.set_index('Variable', drop=True, inplace=True)

# Create a list of columns that will be modified for easier referencing
SDG_cols=['SDG_'+str(i) for i in range(1,18)]
BW_cols = ['BW_'+i for i in SDG_cols]
SDG_std_cols=['SDG_'+str(i)+'_std' for i in range(1,18)]
SDG_count_cols=['SDG_'+str(i)+'_count' for i in range(1,18)]
MA7_cols=['MA_7day_'+str(i) for i in range(1,18)]
MA60_cols=['MA_60day_'+str(i) for i in range(1,18)]

# Load data in a pandas dataframe and remove the csv index
# Convert date column to datetime object
# dataframe = pd.read_csv(path+variables.loc['Data_File_Name', 'Value'])
dataframe = pd.read_csv('BHOOY.csv')
dataframe = dataframe.drop(['Unnamed: 0'], axis=1)
dataframe['date']=dataframe['date'].astype('datetime64[ns]')

#Adding back missing days with no news for moving average fill
dataframe = wma.fill_dates(dataframe)

# Bayesian weighting data to give less weight to noisy data points
dataframe = bayesian.bayesian_weights(dataframe, SDG_cols, SDG_count_cols, SDG_std_cols)

# Calculate 7-day MA of each z-score:
dataframe, company_nan = wma.weighted_MA(dataframe=dataframe, ma_col=BW_cols, window_size=7, window_sigma=2.5, minimum_periods=7, extend=True, merge_col=['COMPANY'])

# Calculate 60-day MA of each z-score:
dataframe, company_nan_60 = wma.weighted_MA(dataframe=dataframe, ma_col=BW_cols, window_size=60, window_sigma=25, minimum_periods=15, extend=True, merge_col=['COMPANY'])

# Drop Bayesian-weighted columns as they hold no further value
# MA-7 and MA-60 incorporated their characteristics
dataframe.drop(BW_cols, axis=1, inplace=True)

# Calculate averages for each data-group
dataframe['SDG_Mean']=dataframe[SDG_cols].mean(axis=1)
dataframe['SDG_std_Mean']=dataframe[SDG_std_cols].mean(axis=1)
dataframe['SDG_count_Mean']=dataframe[SDG_count_cols].mean(axis=1)
dataframe['MA_7day_Mean']=dataframe[MA7_cols].mean(axis=1)
dataframe['MA_60day_Mean']=dataframe[MA60_cols].mean(axis=1)

# GICS Mapping
# gsector = {10 : 'Energy',
# 15 : 'Materials',
# 20 : 'Industrials',
# 25 : 'Consumer Discretionary',
# 30 : 'Consumer Staples',
# 35 : 'Health Care',
# 40 : 'Financials',
# 45 : 'Information Technology',
# 50 : 'Telecommunication Services',
# 55 : 'Utilities',
# 60 : 'Real Estate'}
# linking_table = pd.read_csv(path + 'combine_v1v2_Fiona/SPY/WRDS/SPY_Link.csv')
# dataframe['GICS Sector'] = dataframe.Ticker.map(dict(zip(linking_table.tic, linking_table.GSECTOR)))

# Re-arrange columns for consistent formating
# columns_rearrange = ['date', 'Ticker', 'COMPANY', 'GICS Sector']+SDG_cols+['SDG_Mean']+MA7_cols+['MA_7day_Mean']+MA60_cols+['MA_60day_Mean']+SDG_std_cols+['SDG_std_Mean']+SDG_count_cols+['SDG_count_Mean']
# dataframe = dataframe[columns_rearrange]

########## Re-scale data to avoid different ranges for different SDGs ##########
# # A -1 score in SDG-1 should indicate same level of concern for SDG-1 as a -1 for SDG-4 does for it
# print('Rescaling...')
#
# # Get Market capitalization file
# print('Using ' + path + variables.loc['Market_Caps', 'Value'])
# cap = pd.read_csv(path + variables.loc['Market_Caps', 'Value']).drop('Unnamed: 0', axis=1)
#
# # Fill in missing caps
# cap.loc[:, 'datadate'] = cap.datadate.astype('datetime64[ns]')
# idx = pd.date_range(cap.datadate.min(), cap.datadate.max(), freq='12M')
# cap.datadate = pd.DatetimeIndex(cap.datadate)
# cap = cap.set_index('datadate')
# # We only forward-fill as back-fill will over-estimate the market capitalization. for initial missing caps, we can safely assume that they belong to the smallest bin
# cap = cap.groupby(['tic', 'conm']).apply(lambda x : x.reindex(idx).rename_axis('datadate').ffill()).drop(['tic', 'conm'], axis=1).reset_index()
# cap['cmth'] = cap.datadate.dt.month
# cap['cyear'] = cap.datadate.dt.year
# # Assign Bins
# cap['Bin'] = cap.groupby('cyear').apply(lambda x : pd.qcut(x.Cap_BB, [0, 0.025, 0.175, 0.475, 0.725, 0.875, 0.95, 1.],\
#     labels=['0-2.5%', '2.5-17.5%', '17.5-47.5%', '47.5-72.5%', '72.5-87.5%', '87.5-95%', '95-100%'])).reset_index(0, drop=True)
# cap['Bin'].fillna('0-2.5%', inplace=True)
#
# # Add column to data-frame
# dataframe['Bin'] = pd.Series(list(zip(dataframe.Ticker, dataframe.date.dt.year-1))).map(dict(zip(zip(cap.tic, cap.cyear), cap.Bin)))
# dataframe['Bin'] = dataframe['Bin'].astype('category')
# dataframe['Bin'].cat.reorder_categories(['0-2.5%', '2.5-17.5%', '17.5-47.5%', '47.5-72.5%', '72.5-87.5%', '87.5-95%', '95-100%'], inplace=True, ordered=True)
#
# # Add Mean column. We need to rescale it too.
# MA7_cols.append('MA_7day_Mean')
# MA60_cols.append('MA_60day_Mean')
#
# # Call re-scale function with binning
# dataframe = wma.rescale(dataframe, MA7_cols, lookback=5, initial_years=3)
# dataframe = wma.rescale(dataframe, MA60_cols, lookback=5, initial_years=3)
#
# # Undo changes made to data for rescaling
# MA7_cols.pop(-1)
# MA60_cols.pop(-1)
# dataframe.drop('Bin', axis=1, inplace=True)

# Write file for storage
# Output data to CSV
dataframe.to_csv('data_rated_v1v2.csv')
print('Saved - Rated Data')