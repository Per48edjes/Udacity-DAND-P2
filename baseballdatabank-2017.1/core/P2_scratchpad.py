import glob
import pandas as pd
from scipy import stats
import numpy as np


# 0. Import all the data to a dictionary where keys are file names and values
# are dataframes ###
filename_list = glob.glob('*.csv')
data_dict = {f[:-4]: pd.read_csv(f) for f in filename_list}

## Show all the columns ##
pd.set_option('display.max_columns', None)
## Show top and bottom 3 rows ##
pd.set_option('display.max_rows', 10)


### 1. Rankings of WS-winning Teams ###

## 1.1 Adding compound team metrics ##

# On-base Percentage, "OBP"
data_dict["Teams"]["OBP"] = (data_dict[
    "Teams"]["H"] + data_dict[
        "Teams"]["BB"] + data_dict[
            "Teams"]["HBP"]) / (data_dict[
                "Teams"]["AB"] + data_dict[
                    "Teams"]["SF"] + data_dict[
                        "Teams"]["BB"] + data_dict[
                            "Teams"]["HBP"])

# Slugging Percentage, "SLG"
TB_weights = pd.DataFrame([1, 2, 3, 4], index=['S', '2B', '3B', 'HR'])
data_dict["Teams"]["S"] = data_dict["Teams"][
    "H"] - data_dict["Teams"].ix[:, "2B": "HR"].sum(axis=1)

new_col_list = ['yearID', 'lgID', 'teamID', 'franchID', 'divID', 'Rank', 'G', 'Ghome', 'W', 'L', 'DivWin', 'WCWin', 'LgWin', 'WSWin', 'R', 'AB', 'H', 'S', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS',
                'HBP', 'SF', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'name', 'park', 'attendance', 'BPF', 'PPF', 'teamIDBR', 'teamIDlahman45', 'teamIDretro', 'OBP']

data_dict["Teams"] = data_dict["Teams"][new_col_list]

data_dict["Teams"]["TB"] = data_dict["Teams"].ix[:, "S":"HR"].dot(TB_weights)

data_dict["Teams"]["SLG"] = data_dict["Teams"]["TB"] / data_dict[
    "Teams"]["AB"]

# On-base plus slugging, "OPS"
data_dict["Teams"]["OPS"] = data_dict[
    "Teams"]["OBP"] + data_dict["Teams"]["SLG"]

# Team Batting Average, "teamBA"
data_dict["Teams"]["teamBA"] = data_dict[
    "Teams"]["H"] / data_dict["Teams"]["AB"]

# Making WSWin column summarizable
data_dict["Teams"].replace({'WSWin': {'Y': 1, 'N': 0}}, inplace=True)

## Computing average age of teams per year ##

# Create dataframe of all players on each team in each year; player might
# show up on 2+ different teams in a given year
tables_of_players = ["Batting", "Pitching", "Fielding"]
tables_to_merge = []
for table in tables_of_players:
    df = data_dict[table]
    tables_to_merge.append(df.groupby(
        ["yearID", "teamID", "playerID"]).mean())

all_players_byteam = reduce(lambda left, right: pd.merge(
    left, right, left_index=True, right_index=True, how="outer"), tables_to_merge)

all_players_byteam.reset_index(inplace=True)
all_players_byteam = all_players_byteam[["yearID", "teamID", "playerID"]]
player_info = data_dict["Master"]
age_df = all_players_byteam.merge(
    player_info[["playerID", "birthYear"]], on="playerID")
age_df["age"] = age_df["yearID"] - age_df["birthYear"]

# Players with missing 'birthYear's == 130
# missing_age_info = age_df[age_df.isnull().birthYear]
# print len(set(missing_age_info.playerID))

# Add "age" column to "Teams" table
grouped_Ages = age_df.groupby(["yearID", "teamID"]).mean()[["age"]]
grouped_Teams = data_dict["Teams"].groupby(["yearID", "teamID"]).mean()
data_dict["Teams"] = grouped_Teams.merge(
    grouped_Ages, left_index=True, right_index=True, how="inner")

### 1.2 Assemble dataframe of percentile rankings for selected statistics ###

# Begin exploring where the WS team falls in distribution for various
# statistics vs. competitors in a given year ###

# Ignore gnore RuntimeWarning: invalid value when computing percentiles
np.seterr(divide='ignore', invalid='ignore')


# Create dataframe containing all stats of interest
stats_of_interest = ["R", "H", "teamBA",
                     "SLG", "OBP", "OPS", "SB", "RA", "ERA", "HA", "SO", "BBA", "age"]

pctile_calc = lambda x: [stats.percentileofscore(x, a, 'strict') for a in x]
pctile_table = data_dict["Teams"].apply(pctile_calc)
WSWinner_pctile_table = pctile_table[pctile_table["WSWin"] > 0.0]
print pctile_table
