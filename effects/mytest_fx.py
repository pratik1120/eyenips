"""An effect you made in the Create Effect window (expression mode)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vizstudio.exprfx import ExpressionEffectBase


class CustomEffect(ExpressionEffectBase):
    name = 'Mytest FX'
    BRIGHT = 'sin(x*10+t)+bass*4'
    HUE = 'x*0.5+t*0.05'
