#-----------------------------------------------------------------------------
# Copyright (c) 2012 - 2019, Anaconda, Inc., and Bokeh Contributors.
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Boilerplate
#-----------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest ; pytest

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Standard library imports

# External imports

# Bokeh imports
from bokeh.models import FixedTicker

# Module under test
import bokeh.models.grids as bmg

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

def test_ticker_accepts_number_sequences():
    g = bmg.Grid(ticker=[-10, 0, 10, 20.7])
    assert isinstance(g.ticker, FixedTicker)
    assert g.ticker.ticks == [-10, 0, 10, 20.7]

    g = bmg.Grid()
    g.ticker = [-10, 0, 10, 20.7]
    assert isinstance(g.ticker, FixedTicker)
    assert g.ticker.ticks == [-10, 0, 10, 20.7]

#-----------------------------------------------------------------------------
# Dev API
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Private API
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------
