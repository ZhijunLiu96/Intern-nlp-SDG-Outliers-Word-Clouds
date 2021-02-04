# Function to get bayesian weighted data
import pandas as pd

# PARAMETERS: 
#   data (type: pandas.DataFrame) - DataFrame cotaining relavent data for bayesian weighting
#   count_cols (type: <class 'list'>) - List of columns containing news volume information for each SDG ([SDG_1_count - SDG_17_count])
#   std_cols (type: <class 'list'>) - List of columns containing standard deviation in the news for each day for all SDGs ([SDG_1_std - SDG_17_std])
# RETURNS:
#   prior (type: pandas.DataFrame) - Returns a DataFrame with base value of news counts, base value of standard deviation and sigma prior for each SDG
# USE:
# Helps obtain the sigma priors for each SDG to be used in Bayesian weight calculation
def priors(data, count_cols, std_cols):
    # Create an empty DataFrame to store results
    prior = pd.DataFrame()

    # For each SDG-
    for i in range(1, len(count_cols)+1):
        # Might not be necessary as we omit rows where count = 0
        # data[count_cols[i-1]].fillna(0, inplace=True)
        # data[std_cols[i-1]].fillna(0, inplace=True)
        # For all days where news volume is 1, we don't have enough information to get standard deviation between articles. Thus, a reasonable assumption of '2' is made for computing the sigma prior
        data.loc[data[count_cols[i-1]]==1, std_cols[i-1]]=2.0

        # First, we get the base number for news volumes
        # We do this by getting the median of all the news volumes (counts) given that volume > 1
        prior.loc[str(i), 'n_base'] =  data.loc[data[count_cols[i-1]]>1, [count_cols[i-1]]].median().values

        # Second, we get the 
        prior.loc[str(i), 's_base'] =  (data.loc[data[count_cols[i-1]]>1, [std_cols[i-1]]].mean().values \
                                        + data.loc[data[count_cols[i-1]]>0, [std_cols[i-1]]].median().values)/2
        prior.loc[str(i), 's_prior'] = prior.loc[str(i), 's_base'] * ((3/prior.loc[str(i), 'n_base'])**(0.5))
    return prior

# PARAMETERS: 
#   data (type: pandas.DataFrame) - DataFrame cotaining relavent data for bayesian weighting
#   sdg_cols (type: <class 'list'>) - List of columns containing score information for each SDG ([SDG_1 - SDG_17])
#   count_cols (type: <class 'list'>) - List of columns containing news volume information for each SDG ([SDG_1_count - SDG_17_count])
#   std_cols (type: <class 'list'>) - List of columns containing standard deviation in the news for each day for all SDGs ([SDG_1_std - SDG_17_std])
#   date_col (type: <class 'str'>) (optional, default='date') - Name of the date column to be used for filtering data
#   lookback (type: <class 'int'>) (optional, default=5) - Number of past years to be used to calculate priors for the present year
#   initial_years (type: <class 'int'>) (optional, default=3) - Number of initial years to that are in sample. This number should be less than the lookback period.
# RETURNS:
#   data (type: pandas.DataFrame) - Returns original data with STD updated to '2' for rows with count=1 as it is a resonable assumption
#   prior (type: pandas.DataFrame) - Returns sigma prior to be used for each SDG for each year
# USE:
# Helps obtain the sigma priors for each SDG to be used in Bayesian weight calculation for each year. Priors for each year are calculated based on the lookback period.
def priors_yearly(data, count_cols, std_cols, date_col='date', lookback=5, initial_years=3):
    # Initialize
    yearly_priors = {}
    min_year = data[date_col].dt.year.min()
    max_year = data[date_col].dt.year.max()
    
    for i in range(1, len(count_cols)+1):
        # For all days where news volume is 1, we don't have enough information to get standard deviation between articles. Thus, a reasonable assumption of '2' is made for computing the sigma prior
        data.loc[data[count_cols[i-1]]==1, std_cols[i-1]]=2.0
        
    initial_range = [min_year + i for i in range(initial_years)]
    
    temp = priors(data[data[date_col].dt.year.isin(initial_range)], count_cols, std_cols)
    initial_range.append(initial_range[-1]+1)
    
    for i in initial_range:
        yearly_priors[i] = temp
    
    # Now, we do this for the remaining years
    for i in range(initial_range[-1]+1, max_year+1):
        new_range = [j for j in range(max(min_year, i-lookback), i)]
        temp = priors(data[data[date_col].dt.year.isin(new_range)], count_cols, std_cols)
        yearly_priors[i] = temp
        
    # Format and return yearly priors
    p = pd.DataFrame(columns=yearly_priors.keys())
    for yr in yearly_priors.keys():
        p[yr] = yearly_priors[yr]['s_prior']
    
    return data, p

# PARAMETERS: 
#   data (type: pandas.DataFrame) - DataFrame cotaining relavent data for bayesian weighting
#   sdg_cols (type: <class 'list'>) - List of columns containing score information for each SDG ([SDG_1 - SDG_17])
#   count_cols (type: <class 'list'>) - List of columns containing news volume information for each SDG ([SDG_1_count - SDG_17_count])
#   std_cols (type: <class 'list'>) - List of columns containing standard deviation in the news for each day for all SDGs ([SDG_1_std - SDG_17_std])
#   date_col (type: <class 'str'>) (optional, default='date') - Name of the date column to be used for filtering data
#   lookback (type: <class 'int'>) (optional, default=5) - Number of past years to be used to calculate priors for the present year
#   initial_years (type: <class 'int'>) (optional, default=3) - Number of initial years to that are in sample. This number should be less than the lookback period.
# RETURNS:
#   data (type: pandas.DataFrame) - Returns original data with STD updated to '2' for rows with count=1 as it is a resonable assumption
#                                   Weights SDG columns according to the bayesian scheme and returns it as additional columns with 'BW_' prefix
# USE:
# Applies bayesian weighting scheme to SDG columns based on News Volume and daily standard deviation
# This helps assign higher weight to a score that is a result of higher number of articles and have lower standard deviation (i.e. The different number of news articles agree on a given score)
# It also assings lower weight to a score that is reported by fewer sources or that vary too much (i.e. high standard deviation) or both
def bayesian_weights(data, sdg_cols, count_cols, std_cols, date_col='date', lookback=5, initial_years=3):
    # Calculate bayesian prior 
    data, prior = priors_yearly(data, count_cols, std_cols, date_col=date_col, lookback=lookback, initial_years=initial_years)
    
    col_name = ['BW_'+i for i in sdg_cols]

    r = data[date_col].dt.year.map(prior.to_dict(orient='list'))
    data[col_name] = pd.DataFrame(r.tolist(), index=r.index)

    # Generate weights using bayesian priors
    data[col_name] = (data[count_cols]*(data[col_name]**2).values)/(data[count_cols]*(data[col_name]**2).values+(data[std_cols]**2).values)

    # Weight data and store in additional columns
    data[ col_name] = data[sdg_cols]*data[col_name].values
    return data