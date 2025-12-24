import nalulib.pyplot as plt



if __name__ == '__main__':
    #figt, ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,5.8))
    #figt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
    #if 'x' in var:
    #    plt.plot(dft['Time'].values, dft['Cx'].values, label=FC+'x')
    #if 'y' in var:
    #    plt.plot(dft['Time'].values, dft['Cy'].values, label=FC+'y')
    #ax.set_ylabel(label)
    #ax.set_xlabel('Time [s]')
    #ax.legend()




    fig, ax= plt.subplots(1, 1)

    ax.plot([0,1], [0,1], '-', label='Hello')
    ax.plot([0,1], [-1,0],'-', label='You')
    plt.show()
