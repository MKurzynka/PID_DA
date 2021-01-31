import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def plot_p_vs_tpc(data_frame, fig_size=(15,8)):
    fig = plt.figure(figsize=fig_size, dpi=80)
    sns.scatterplot(x="P", y="tpc_signal", hue="pdg", data = data_frame, palette="deep")
    plt.xscale('log')