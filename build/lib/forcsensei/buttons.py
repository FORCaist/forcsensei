from ipywidgets import widgets
from IPython.display import display
from traitlets import traitlets
import numpy as np

import forcsensei.preprocess as tpp
import forcsensei.regress as tre
import forcsensei.results as res
import forcsensei.utils as uts

def ini0(X):
    X['fn'] = X['fc'].selected
    del X['fc']
    X = uts.header_keywords(X)
    display(widgets.HTML(value="FORCsensei Initialization complete"))
    return X

def ini(ex):
    ex.value = ini0(ex.value)

def ppo(ex):    
    ex.value = tpp.options(ex.value)

def ppe(ex):    
    ex.value = tpp.execute(ex.value)

def ro(ex):    
    ex.value = tre.options(ex.value)

def rc(ex):    
    ex.value = tre.compare(ex.value)

def re(ex):    
    ex.value = tre.execute(ex.value)

def rf(ex):    
    res.forc(ex.value)

def rpo(ex):    
    res.profile_options(ex.value)

def rpp(ex):    
    res.profile_plot(ex.value)

class LoadedButton(widgets.Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))

