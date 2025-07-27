import os
import numpy as np
import plotext as _plt

__all__ = [name for name in dir(_plt) if not name.startswith('_')]
globals().update({name: getattr(_plt, name) for name in __all__})

# Add your custom functions/classes
globals()['subplots'] = subplots

MARKERS=['dot', 'cross', '+', 'fhd', 'braille', 'sd']
COLORS=['blue', 'orange', 'green', 'red', 'gray','magenta', 'cyan']

def get_plotext_default_size():
    try:
        size = os.get_terminal_size()
    except OSError:
        size = (80, 24)  # Default size if terminal size cannot be determined
    width = size.columns - 10
    height = int(size.lines * 0.7)
    height100 = int(size.lines)
    return width, height, height100


class DummyAx():
    def __init__(self, ia=None, ib=None):
        self.iPlot=0
        self.ia=ia
        self.ib=ib
        pass

    def plot(self, x, y, *args, color=None, ms=None, markerfacecolor=None, markersize=None, **kwargs):
        if len(args)==1:
            sty = args[0]
            #print('>>> sty', sty)
        if not hasattr(x, '__len__'):
            x = [x]
        if not hasattr(y, '__len__'):
            y = [y]
        #print(kwargs.keys())

        if color is None:
            color = COLORS[self.iPlot % len(COLORS)]
        marker = MARKERS[self.iPlot % len(MARKERS)]

        if self.ia is not None and self.ib is not None:
            _plt.subplot(self.ia+1, self.ib+1)
        _plt.plot(x, y, marker=marker, color=color, **kwargs)


        self.iPlot+=1


    def set_xlabel(self, *args, **kwargs):
        if self.ia is not None and self.ib is not None:
            _plt.subplot(self.ia+1, self.ib+1)
        _plt.xlabel(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        if self.ia is not None and self.ib is not None:
            _plt.subplot(self.ia+1, self.ib+1)
        _plt.ylabel(*args, **kwargs)

    def set_aspect(self, *args, **kwargs):
        #print('[pyplott] aspect skipped')
        pass
        #_plt.ylabel(*args, **kwargs)

    def set_title(self, *args, **kwargs):
        if self.ia is not None and self.ib is not None:
            _plt.subplot(self.ia+1, self.ib+1)
        _plt.title(*args, **kwargs)

    def legend(self, *args, **kwargs):
        pass
        # _plt.legend(*args, **kwargs)

    def arrow(self, *args, **kwargs):
        pass
        #print('[pyplott] arrow skipped')

    def grid(self, *args, **kwargs):
        _plt.grid(*args, **kwargs)


class DummyFig():
    def __init__(self, axes=None, *args, figsize=None, **kwargs):
        if axes is None:
            axes = [DummyAx()]
        self.axes = axes

        w, h, h100 = get_plotext_default_size()
        ratio_def = h/w
        #print('>>> w, h', w, h)
        if figsize is not None:
            w0, h0 = figsize
            ratio = h0/w0
            #print('>>>> figsize')
            #print('Ratio:', ratio)
            h = int(h100 * ratio)
        else:
            w=None
            h=None
        #print('>>> w, h', w, h)
        _plt.plotsize(width=w, height=h)
    
    def subplots_adjust(*args, **kwargs):
        pass

def subplots(a, b, figsize=None, sharey=None, **kwargs):
    _plt.subplots(a, b, **kwargs)
    axes = np.empty((a,b), dtype=object)
    for i in range(a):
        for j in range(b):
           axes[i,j] = DummyAx(ia=i, ib=j)

#    print('[pyplott] dummy suplots')
#    if len(args)==2:
    fig = DummyFig(axes=axes, figsize=figsize)
    if a==1 and b==1:
        axes=axes[0,0]
    elif a==1 or b==1:
        axes=axes.flatten()
    return fig, axes
# 
#def show():
#    _plt.show()



if __name__ == '__main__':
    fig, ax= subplots()

    ax.plot([0,1], [0,1])
    _plt.show()
