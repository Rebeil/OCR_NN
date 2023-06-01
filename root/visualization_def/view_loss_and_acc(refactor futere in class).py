from matplotlib.ticker import AutoMinorLocator
import matplotlib as plt

def view_2_grahic(data:tuple=None, name_gra:str = 'None_name', figsize:tuple=(10,7),epochs:int=9):
    """_summary_

    Args:
        data (tuple, optional): потери и точность. Defaults to None.
        name_gra (str, optional): название графика. Defaults to 'None_name'.
        figsize (tuple, optional): размер фигуры. Defaults to (10,7).
        epochs (int, optional): кол-во эпох. Defaults to 9.

    Raises:
        Exception: если данные(data) не переданы
    """
    if data is None:
        raise Exception("Data is none")
    loss_epochs_list, acc_epochs_list = data
    fig, ax = plt.subplots(len(data), figsize=figsize)
    fig.suptitle(name_gra,x=0.52, y=0.99)

    fig.tight_layout(h_pad= 2 )
    #define subplot titles
    ax[0].set_title('loss', fontsize=16)
    ax[1].set_title('acc',fontsize=16)

    #add overall title and adjust it so that it doesn't overlap with subplot titles
    plt.subplots_adjust(wspace=0, hspace=0.8 )

    for i in ax:
        i.set_xlabel("epochs", fontsize=18)        
        i.xaxis.set_label_coords (.0, -.3)
        i.set_ylabel("y", fontsize=12)
        i.grid(which="major", linewidth=1.2)
        i.grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
        
        i.xaxis.set_minor_locator(AutoMinorLocator())
        i.yaxis.set_minor_locator(AutoMinorLocator())
        i.tick_params(which='major', length=10, width=2)
        i.tick_params(which='minor', length=5, width=1)
    ax[0].annotate(round(loss_epochs_list[-1],4), (epochs,loss_epochs_list[-1]))
    ax[0].plot(loss_epochs_list,'o-r')
    
    ax[1].annotate(round(acc_epochs_list[-1],4), (epochs,acc_epochs_list[-1]))
    ax[1].plot(acc_epochs_list,'o-.g')