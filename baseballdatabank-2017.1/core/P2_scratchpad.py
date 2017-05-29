import glob
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## Show all the columns ##
pd.set_option('display.max_columns', None)
## Show limited rows ##
pd.set_option('display.max_rows', 50)

## Ignore gnore RuntimeWarning: invalid value when computing percentiles ##
np.seterr(divide='ignore', invalid='ignore')


### 0. Import data ###
filename_list = glob.glob('*.csv')
data_dict = {f[:-4]: pd.read_csv(f) for f in filename_list}

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

# Walks plus hits per inning pitched, "WHIP"

data_dict["Teams"]["WHIP"] = (data_dict["Teams"][
                              "BBA"] + data_dict["Teams"]["HA"]) / (data_dict["Teams"]["IPouts"] / 3)

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

# Add "age", "salaries" column to "Teams" table
grouped_Ages = age_df.groupby(["yearID", "teamID"]).mean()[["age"]]
grouped_Teams = data_dict["Teams"].groupby(["yearID", "teamID"]).mean()
salary_info = data_dict["Salaries"].groupby(
    ["yearID", "teamID"]).median()[["salary"]]

team_summary_per_year = grouped_Teams.join(
    grouped_Ages, how="left").join(salary_info, how="left")


## Fixing year 2016's missing salary summary stats due to mismatched teamIDs ##

# Find the rows in the team_summary_per_year table missing salary info
missing_team_sals = list(zip(
    *team_summary_per_year[team_summary_per_year.salary.isnull()].salary.ix[1985:].index.values)[1])

# Find the teamIDs in salary_info but NOT in team_summary_per_year
teamIDs_from_salary_info = list(set(salary_info.ix[
                                2016].index.values) - set(team_summary_per_year.ix[2016].index.values))

# Reconcile teamIDs; create a mapping to do look-up
teamID_df = data_dict["Teams"][data_dict["Teams"].loc[
    :, "yearID"] == 2016].loc[:, ["teamIDBR", "teamID"]]

teamID_mapping = teamID_df[teamID_df["teamID"] !=
                           teamID_df["teamIDBR"]]

teamID_mapping.set_index(["teamID"], drop=True, inplace=True)

# Replace the missing 'salary' values in team_summary_per_year
df = team_summary_per_year.ix[2016,
                              'salary'].ix[team_summary_per_year.ix[2016, 'salary'].isnull()]

df1 = df.reset_index()['teamID'].map(
    lambda x: salary_info.ix[(2016, teamID_mapping.ix[x, 'teamIDBR']), 'salary'])

team_summary_per_year.ix[2016,
                         'salary'].ix[team_summary_per_year.ix[2016, 'salary'].isnull()] = df1.astype(np.float64).values


### 1.2 Assemble dataframe of rankings for selected statistics ###

# Begin exploring where the WS team falls in distribution for various
# statistics vs. competitors in a given year ###


def pctile_calc(x):
    return [stats.percentileofscore(x, a, 'mean') for a in x]


def zscore(x):
    return stats.zscore(x)


def stdizer(df, names, *args):
    std_tables = {}
    name_picker = 0
    for f in args:
        temp_dic = {}
        for year in set(team_summary_per_year.index.get_level_values("yearID")):
            df_temp = df.loc[year].apply(f)
            df_temp["yearID"] = [year for teams in df_temp.index.values]
            temp_dic[year] = df_temp.reset_index().groupby(
                ["yearID", "teamID"]).max()
        std_tables[names[name_picker]] = pd.concat(list(temp_dic.values()))
        name_picker += 1
    return std_tables


std_tables_names = ["percentiles", "zscores"]
std_data_library = stdizer(team_summary_per_year,
                           std_tables_names, pctile_calc, zscore)

zscores_wswinners = std_data_library["zscores"].reset_index().groupby("yearID").apply(
    lambda x: x.sort_values("WSWin", ascending=False).head(1)).set_index(["yearID", "teamID"])


### 1.3 Charting of pairwise comparisons ###

stats_of_interest = ["R", "H", "teamBA",
                     "SLG", "OBP", "OPS", "SB", "RA", "ERA", "HA", "SO", "BBA", "age", "salary"]

# Experimenting by creating 1 pairwise scatter plot(quadrant delineation
# important!)
# fig = plt.figure()
# for i in range(1, 10):
#     x = zscores_wswinners["teamBA"]
#     y = zscores_wswinners["ERA"]
#
#     ax = fig.add_subplot(3, 3, i)
#     ax.scatter(x, y)
#     plot_axes_range = [-3, 3, -3, 3]
#     ax.axis(plot_axes_range, 'equal')
#
#     ax.spines['left'].set_position('center')
#     ax.spines['right'].set_color('none')
#     ax.spines['bottom'].set_position('zero')
#     ax.spines['top'].set_color('none')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#
#     ax.axhline(linewidth=1, color='black')
#     ax.axvline(linewidth=1, color='black')
#
# plt.show()

# Heatmap for correlations
# corrmat = zscores_wswinners.corr()
# print corrmat
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)

# k = 10  # number of variables for heatmap
# cols = corrmat.nlargest(k, 'WSWin')['WSWin'].index
# cm = np.corrcoef(zscores_wswinners[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
#                  'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()

# Experimenting with many pairwise plots

# Select the years which have information on across all dimensions
# df = zscores_wswinners[-zscores_wswinners["OPS"].isnull()]
# sns.set()
# cols = ["WHIP", "OPS", "ERA", "teamBA", "age", "salary"]
# sns.pairplot(df[cols])
# plt.show()
