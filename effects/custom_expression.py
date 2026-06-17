"""The live "Custom (expression)" effect.

This is the effect the Expression tab of the Create Effect window edits in real
time. It's just a thin subclass of ExpressionEffectBase with default formulas.
When you click "Save as effect", a new file like this one is written for you.
"""

# Let this file find the vizstudio package whether run directly or imported.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vizstudio.exprfx import ExpressionEffectBase


class CustomExpression(ExpressionEffectBase):
    name = "Custom (expression)"
    description = "Type math formulas in the Create Effect window to shape this."


if __name__ == "__main__":
    import app
    app.main(prefer=CustomExpression.name)
