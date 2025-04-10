# Copyright 2015-2020 Mathieu Bernard
#
# This file is part of phonologizer: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Phonologizer is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with phonologizer. If not, see <http://www.gnu.org/licenses/>.
"""Multilingual text to phonemes converter"""

from g2p.phonemizer_api.backend.espeak import EspeakBackend, EspeakMbrolaBackend
from g2p.phonemizer_api.backend.festival import FestivalBackend
from g2p.phonemizer_api.backend.segments import SegmentsBackend
