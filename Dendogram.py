import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform, pdist
import git

#get the Git root location
git_repo = git.Repo('.', search_parent_directories=True)
#Load the data
df = pd.read_csv(f"{git_repo.working_tree_dir}\\Data\\data.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)

#Creating a list of elements
start_index = df.columns.get_loc('A')
end_index = df.columns.get_loc('T')
ELEMENTS = np.array(df.iloc[:, start_index:end_index + 1].columns)

df_data = df.iloc[:, start_index:end_index + 1]

#Calculating the correlation
corr = df_data.corr(method='spearman')

#Distance matriz
dissimilarity = 1 - corr
Z = linkage(squareform(dissimilarity), 'complete')


#Plot:
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 5), layout='constrained')
hierarchy.set_link_color_palette(['#000000', '#000000', '#000000', '#000000','#000000', '#000000', '#000000', '#000000'])
dendrogram(Z, labels=ELEMENTS, orientation='top', leaf_rotation=0, above_threshold_color='grey')
fig.savefig(f"{git_repo.working_tree_dir}\\Dendogram Plot\\Dendogram.jpeg", dpi=300, bbox_inches='tight')
