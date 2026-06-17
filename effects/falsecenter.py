"""An effect you made in the Create Effect window."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vizstudio.exprfx import ExpressionEffectBase


class CustomEffect(ExpressionEffectBase):
    name = 'falsecenter'
    BRIGHT = '(((sin((x+y)*14 + t*5.3)) * (1 + bass*2.3)) + ((sin(r*29 - t*6)) * (1 + beat*2.3))) * 0.7'
    HUE = 'theta*0.5 + t*0.1'
    PALETTE = 'ocean'
