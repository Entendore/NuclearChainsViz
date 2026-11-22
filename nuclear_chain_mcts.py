#!/usr/bin/env python3
"""
Nuclear Chain Reaction Explorer with MCTS
==========================================
Visualizes decay chains, fission chains, and fusion chains using Monte Carlo Tree Search.
Single-file application — requires: pip install PySide6

Usage: python nuclear_chain_mcts.py
"""

from __future__ import annotations
import math
import random
import time
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Ensure libEGL is findable (some Linux systems need this)
_lib_dir = os.path.expanduser("~/.local/lib")
if os.path.isdir(_lib_dir):
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    if _lib_dir not in _existing:
        os.environ["LD_LIBRARY_PATH"] = f"{_lib_dir}:{_existing}"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QTextEdit, QSplitter, QFrame, QProgressBar,
    QScrollArea, QCheckBox, QSlider, QGridLayout, QSizePolicy,
    QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem, QGraphicsPathItem,
    QToolBar, QStatusBar, QMenu, QMenuBar
)
from PySide6.QtCore import (
    Qt, QTimer, QRectF, QPointF, QSize, Signal, QThread
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QLinearGradient, QPainterPath,
    QRadialGradient, QAction, QKeySequence, QPalette
)


##############################################################################
# 1. NUCLEAR PHYSICS DATA
##############################################################################

class DecayMode(Enum):
    ALPHA = "alpha"
    BETA_MINUS = "beta-"
    BETA_PLUS = "beta+"
    ELECTRON_CAPTURE = "EC"
    NEUTRON_EMISSION = "n"
    PROTON_EMISSION = "p"
    SPONTANEOUS_FISSION = "SF"
    STABLE = "stable"
    ISOMERIC_TRANSITION = "IT"


class ChainType(Enum):
    DECAY = "Decay Chain"
    FISSION = "Fission Chain"
    FUSION = "Fusion Chain"


@dataclass
class Isotope:
    name: str
    symbol: str
    Z: int
    A: int
    N: int
    half_life: float
    half_life_str: str
    decay_mode: DecayMode
    decay_energy: float
    is_stable: bool = False

    @property
    def binding_energy_per_nucleon(self) -> float:
        Z, A = self.Z, self.A
        if A <= 0:
            return 0
        N = A - Z
        a_v, a_s, a_c, a_a, a_p = 15.56, 17.23, 0.697, 23.285, 12.0
        BE = (a_v * A
              - a_s * A ** (2 / 3)
              - a_c * Z * (Z - 1) / A ** (1 / 3)
              - a_a * (A - 2 * Z) ** 2 / A
              + a_p / A ** (1 / 4) * (1 if N % 2 == 0 and Z % 2 == 0 else -1 if N % 2 == 1 and Z % 2 == 1 else 0))
        return max(BE / A, 0)


@dataclass
class Reaction:
    reactants: list[str]
    products: list[str]
    Q_value: float
    probability: float = 1.0
    description: str = ""
    reaction_type: ChainType = ChainType.DECAY


# ---------------------------------------------------------------------------
# Isotope database
# ---------------------------------------------------------------------------
ISOTOPES: dict[str, Isotope] = {}


def _add(name, symbol, Z, A, hl, hl_str, mode, energy):
    N = A - Z
    is_stable = (mode == DecayMode.STABLE)
    ISOTOPES[symbol] = Isotope(name, symbol, Z, A, N, hl, hl_str, mode, energy, is_stable)


# --- Uranium-238 Decay Series (4n+2) ---
_add("Uranium-238",    "U-238",  92, 238, 1.41e17,  "4.468 Gy",    DecayMode.ALPHA,        4.270)
_add("Thorium-234",    "Th-234", 90, 234, 2.455e6,  "24.10 d",     DecayMode.BETA_MINUS,   0.273)
_add("Protactinium-234","Pa-234m",91,234, 7.0e4,    "1.17 min",    DecayMode.BETA_MINUS,   2.197)
_add("Uranium-234",    "U-234",  92, 234, 2.455e13, "245.5 ky",    DecayMode.ALPHA,        4.859)
_add("Thorium-230",    "Th-230", 90, 230, 2.536e13, "75.38 ky",    DecayMode.ALPHA,        4.770)
_add("Radium-226",     "Ra-226", 88, 226, 5.045e11, "1600 y",      DecayMode.ALPHA,        4.871)
_add("Radon-222",      "Rn-222", 86, 222, 3.823e5,  "3.823 d",     DecayMode.ALPHA,        5.590)
_add("Polonium-218",   "Po-218", 84, 218, 183.0,    "3.10 min",    DecayMode.ALPHA,        6.115)
_add("Lead-214",       "Pb-214", 82, 214, 1.605e4,  "26.8 min",    DecayMode.BETA_MINUS,   1.024)
_add("Bismuth-214",    "Bi-214", 83, 214, 1197.0,   "19.9 min",    DecayMode.BETA_MINUS,   3.272)
_add("Polonium-214",   "Po-214", 84, 214, 1.643e-4, "164.3 us",    DecayMode.ALPHA,        7.835)
_add("Lead-210",       "Pb-210", 82, 210, 7.0e8,    "22.3 y",      DecayMode.BETA_MINUS,   0.064)
_add("Bismuth-210",    "Bi-210", 83, 210, 1.201e6,  "5.012 d",     DecayMode.BETA_MINUS,   1.161)
_add("Polonium-210",   "Po-210", 84, 210, 1.199e7,  "138.4 d",     DecayMode.ALPHA,        5.407)
_add("Lead-206",       "Pb-206", 82, 206, 0,         "Stable",      DecayMode.STABLE,        0.0)

# --- Thorium-232 Decay Series (4n) ---
_add("Thorium-232",    "Th-232", 90, 232, 4.053e17, "14.05 Gy",    DecayMode.ALPHA,        4.083)
_add("Radium-228",     "Ra-228", 88, 228, 1.736e9,  "5.75 y",      DecayMode.BETA_MINUS,   0.046)
_add("Actinium-228",   "Ac-228", 89, 228, 3.635e5,  "6.15 h",      DecayMode.BETA_MINUS,   2.125)
_add("Thorium-228",    "Th-228", 90, 228, 6.047e7,  "1.913 y",     DecayMode.ALPHA,        5.520)
_add("Radium-224",     "Ra-224", 88, 224, 3.176e6,  "3.632 d",     DecayMode.ALPHA,        5.789)
_add("Radon-220",      "Rn-220", 86, 220, 55.6,     "55.6 s",      DecayMode.ALPHA,        6.405)
_add("Polonium-216",   "Po-216", 84, 216, 0.145,    "0.145 s",     DecayMode.ALPHA,        6.906)
_add("Lead-212",       "Pb-212", 82, 212, 3.836e5,  "10.64 h",     DecayMode.BETA_MINUS,   0.574)
_add("Bismuth-212",    "Bi-212", 83, 212, 3600.0,   "60.55 min",   DecayMode.ALPHA,        6.207)
_add("Thallium-208",   "Tl-208", 81, 208, 183.0,    "3.053 min",   DecayMode.BETA_MINUS,   1.796)
_add("Lead-208",       "Pb-208", 82, 208, 0,         "Stable",      DecayMode.STABLE,        0.0)

# --- Actinium-235 Series (4n+3) ---
_add("Uranium-235",    "U-235",  92, 235, 2.222e16, "703.8 My",    DecayMode.ALPHA,        4.679)
_add("Thorium-231",    "Th-231", 90, 231, 1.016e5,  "25.52 h",     DecayMode.BETA_MINUS,   0.392)
_add("Protactinium-231","Pa-231",91,231, 1.026e13, "32.76 ky",    DecayMode.ALPHA,        5.150)
_add("Actinium-227",   "Ac-227", 89, 227, 7.294e7,  "21.772 y",    DecayMode.BETA_MINUS,   0.045)
_add("Thorium-227",    "Th-227", 90, 227, 1.204e7,  "18.72 d",     DecayMode.ALPHA,        6.147)
_add("Radium-223",     "Ra-223", 88, 223, 8.353e5,  "11.43 d",     DecayMode.ALPHA,        5.979)
_add("Radon-219",      "Rn-219", 86, 219, 3.96,     "3.96 s",      DecayMode.ALPHA,        6.946)
_add("Polonium-215",   "Po-215", 84, 215, 1.781e-3, "1.781 ms",    DecayMode.ALPHA,        7.527)
_add("Lead-211",       "Pb-211", 82, 211, 2.161e3,  "36.1 min",    DecayMode.BETA_MINUS,   1.367)
_add("Bismuth-211",    "Bi-211", 83, 211, 720.0,    "2.14 min",    DecayMode.ALPHA,        6.751)
_add("Thallium-207",   "Tl-207", 81, 207, 4.77,     "4.77 min",    DecayMode.BETA_MINUS,   1.418)
_add("Lead-207",       "Pb-207", 82, 207, 0,         "Stable",      DecayMode.STABLE,        0.0)

# --- Additional important isotopes ---
_add("Carbon-14",      "C-14",   6,  14, 1.798e11, "5,730 y",     DecayMode.BETA_MINUS,   0.156)
_add("Carbon-12",      "C-12",   6,  12, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Carbon-13",      "C-13",   6,  13, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Nitrogen-14",    "N-14",   7,  14, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Nitrogen-15",    "N-15",   7,  15, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Oxygen-16",      "O-16",   8,  16, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Oxygen-17",      "O-17",   8,  17, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Helium-3",       "He-3",   2,   3, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Helium-4",       "He-4",   2,   4, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Hydrogen-1",     "H-1",    1,   1, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Hydrogen-2",     "H-2",    1,   2, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Lithium-7",      "Li-7",   3,   7, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Beryllium-7",    "Be-7",   4,   7, 7.014e6,  "53.22 d",     DecayMode.ELECTRON_CAPTURE,0.862)
_add("Beryllium-8",    "Be-8",   4,   8, 7.0e-17,  "7e-17 s",     DecayMode.ALPHA,        0.092)
_add("Boron-8",        "B-8",    5,   8, 0.77,     "0.77 s",      DecayMode.BETA_PLUS,    17.979)
_add("Iron-56",        "Fe-56",  26,  56, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Iodine-131",     "I-131",  53, 131, 5.616e5, "8.02 d",      DecayMode.BETA_MINUS,   0.971)
_add("Cesium-137",     "Cs-137", 55, 137, 9.473e8, "30.08 y",     DecayMode.BETA_MINUS,   1.176)
_add("Strontium-90",   "Sr-90",  38,  90, 9.137e8, "28.9 y",      DecayMode.BETA_MINUS,   0.546)
_add("Plutonium-239",  "Pu-239", 94, 239, 7.57e11, "24,110 y",    DecayMode.ALPHA,        5.244)
_add("Neutron",        "n",       0,   1, 880.0,    "880 s",       DecayMode.BETA_MINUS,   0.782)
_add("Proton",         "p",       1,   1, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Deuteron",       "D",       1,   2, 0,         "Stable",      DecayMode.STABLE,        0.0)
_add("Tritium",        "T",       1,   3, 3.888e8, "12.32 y",     DecayMode.BETA_MINUS,   0.019)
_add("Positron",       "e+",     -1,   0, 0,         "antimatter",  DecayMode.STABLE,        0.0)
_add("Neutrino",       "v_e",     0,   0, 0,         "Stable",      DecayMode.STABLE,        0.0)


# ---------------------------------------------------------------------------
# Reaction databases
# ---------------------------------------------------------------------------
DECAY_REACTIONS: list[Reaction] = []
FISSION_REACTIONS: list[Reaction] = []
FUSION_REACTIONS: list[Reaction] = []


def _add_decay(parent, daughter, mode, energy, prob, desc):
    DECAY_REACTIONS.append(Reaction(
        reactants=[parent], products=[daughter],
        Q_value=energy, probability=prob,
        description=f"{parent} -> {daughter} ({desc})",
        reaction_type=ChainType.DECAY))


def _define_decay_reactions():
    for p, d, e, desc in [
        ("U-238","Th-234",4.270,"alpha: He-4 emission"),("Th-234","Pa-234m",0.273,"beta-: n -> p + e- + v"),
        ("Pa-234m","U-234",2.197,"beta-: n -> p + e- + v"),("U-234","Th-230",4.859,"alpha: He-4 emission"),
        ("Th-230","Ra-226",4.770,"alpha: He-4 emission"),("Ra-226","Rn-222",4.871,"alpha: He-4 emission"),
        ("Rn-222","Po-218",5.590,"alpha: He-4 emission"),("Po-218","Pb-214",6.115,"alpha: He-4 emission"),
        ("Pb-214","Bi-214",1.024,"beta-: n -> p + e- + v"),("Bi-214","Po-214",3.272,"beta-: n -> p + e- + v"),
        ("Po-214","Pb-210",7.835,"alpha: He-4 emission"),("Pb-210","Bi-210",0.064,"beta-: n -> p + e- + v"),
        ("Bi-210","Po-210",1.161,"beta-: n -> p + e- + v"),("Po-210","Pb-206",5.407,"alpha: He-4 emission"),
        ("Th-232","Ra-228",4.083,"alpha: He-4 emission"),("Ra-228","Ac-228",0.046,"beta-: n -> p + e- + v"),
        ("Ac-228","Th-228",2.125,"beta-: n -> p + e- + v"),("Th-228","Ra-224",5.520,"alpha: He-4 emission"),
        ("Ra-224","Rn-220",5.789,"alpha: He-4 emission"),("Rn-220","Po-216",6.405,"alpha: He-4 emission"),
        ("Po-216","Pb-212",6.906,"alpha: He-4 emission"),("Pb-212","Bi-212",0.574,"beta-: n -> p + e- + v"),
        ("Bi-212","Tl-208",6.207,"alpha: He-4 emission"),("Tl-208","Pb-208",1.796,"beta-: n -> p + e- + v"),
        ("U-235","Th-231",4.679,"alpha: He-4 emission"),("Th-231","Pa-231",0.392,"beta-: n -> p + e- + v"),
        ("Pa-231","Ac-227",5.150,"alpha: He-4 emission"),("Ac-227","Th-227",0.045,"beta-: n -> p + e- + v"),
        ("Th-227","Ra-223",6.147,"alpha: He-4 emission"),("Ra-223","Rn-219",5.979,"alpha: He-4 emission"),
        ("Rn-219","Po-215",6.946,"alpha: He-4 emission"),("Po-215","Pb-211",7.527,"alpha: He-4 emission"),
        ("Pb-211","Bi-211",1.367,"beta-: n -> p + e- + v"),("Bi-211","Tl-207",6.751,"alpha: He-4 emission"),
        ("Tl-207","Pb-207",1.418,"beta-: n -> p + e- + v"),
    ]:
        m = DecayMode.ALPHA if "alpha" in desc else DecayMode.BETA_MINUS
        _add_decay(p, d, m, e, 1.0, desc)


def _define_fission_reactions():
    for target, products in [
        ("U-235", [("Ba-141","Kr-92",0.058,200.4),("Xe-140","Sr-94",0.065,201.8),
                    ("Cs-137","Rb-96",0.062,205.3),("I-131","Y-103",0.029,202.5),
                    ("Ba-144","Kr-89",0.059,198.7),("La-147","Br-87",0.042,195.6),
                    ("Te-134","Zr-100",0.055,203.1),("Xe-134","Sr-100",0.048,201.2)]),
        ("U-238", [("Ba-144","Kr-90",0.052,195.2),("Xe-140","Sr-94",0.058,197.8),
                    ("Cs-137","Rb-96",0.048,198.4),("Nd-147","Ge-87",0.040,192.1)]),
        ("Pu-239",[("Ba-141","Kr-93",0.060,207.1),("Xe-140","Sr-94",0.063,210.5),
                    ("Cs-137","Rb-97",0.057,208.9),("Mo-99","Sn-134",0.051,209.3)]),
    ]:
        for p1, p2, prob, qval in products:
            FISSION_REACTIONS.append(Reaction(
                reactants=[target, "n"], products=[p1, p2, "2n"],
                Q_value=qval, probability=prob,
                description=f"{target} + n -> {p1} + {p2} + 2n ({qval:.1f} MeV)",
                reaction_type=ChainType.FISSION))


def _define_fusion_reactions():
    # pp-chain
    FUSION_REACTIONS.append(Reaction(["H-1","H-1"],["D","e+","v_e"],1.442,1.0,
        "p + p -> D + e+ + v_e  (pp-I first step, very slow)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["D","H-1"],["He-3"],5.493,1.0,
        "D + p -> He-3 + gamma  (pp-I second step)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["He-3","He-3"],["He-4","H-1","H-1"],12.860,0.83,
        "He-3 + He-3 -> He-4 + 2p  (pp-I, 83%)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["He-3","He-4"],["Be-7"],1.586,0.17,
        "He-3 + He-4 -> Be-7 + gamma  (pp-II/III start, 17%)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["Be-7","e+"],["Li-7","v_e"],0.862,0.89,
        "Be-7 + e- -> Li-7 + v_e  (electron capture)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["Li-7","H-1"],["He-4","He-4"],17.348,1.0,
        "Li-7 + p -> 2 He-4  (pp-II final, 17%)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["Be-7","H-1"],["B-8"],0.137,0.11,
        "Be-7 + p -> B-8 + gamma  (pp-III, 0.02% total)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["B-8"],["Be-8","e+","v_e"],17.979,1.0,
        "B-8 -> Be-8 + e+ + v_e",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["Be-8"],["He-4","He-4"],0.092,1.0,
        "Be-8 -> 2 He-4  (unstable, 7e-17 s)",ChainType.FUSION))
    # CNO cycle
    FUSION_REACTIONS.append(Reaction(["C-12","H-1"],["N-13"],1.943,1.0,
        "C-12 + p -> N-13 + gamma  (CNO-I first step)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["N-13"],["C-13","e+","v_e"],2.221,1.0,
        "N-13 -> C-13 + e+ + v_e",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["C-13","H-1"],["N-14"],7.550,1.0,
        "C-13 + p -> N-14 + gamma",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["N-14","H-1"],["O-15"],7.293,1.0,
        "N-14 + p -> O-15 + gamma",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["O-15"],["N-15","e+","v_e"],2.754,1.0,
        "O-15 -> N-15 + e+ + v_e",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["N-15","H-1"],["C-12","He-4"],4.966,1.0,
        "N-15 + p -> C-12 + He-4  (CNO cycle complete)",ChainType.FUSION))
    # Triple-alpha
    FUSION_REACTIONS.append(Reaction(["He-4","He-4"],["Be-8"],-0.092,0.05,
        "He-4 + He-4 -> Be-8 (resonance, short-lived)",ChainType.FUSION))
    FUSION_REACTIONS.append(Reaction(["Be-8","He-4"],["C-12"],7.275,1.0,
        "Be-8 + He-4 -> C-12 + gamma  (Hoyle resonance)",ChainType.FUSION))


_define_decay_reactions()
_define_fission_reactions()
_define_fusion_reactions()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def get_decay_products(isotope_symbol: str) -> list[Reaction]:
    return [r for r in DECAY_REACTIONS if r.reactants[0] == isotope_symbol]

def get_fission_products(isotope_symbol: str) -> list[Reaction]:
    return [r for r in FISSION_REACTIONS if r.reactants[0] == isotope_symbol]

def get_fusion_reactions(available: list[str]) -> list[Reaction]:
    return [r for r in FUSION_REACTIONS if all(reactant in available for reactant in r.reactants)]

def get_reactions_for(chain_type: ChainType, current_nuclei: list[str]) -> list[Reaction]:
    if chain_type == ChainType.DECAY:
        reactions = []
        for nuc in current_nuclei:
            reactions.extend(get_decay_products(nuc))
        return reactions
    elif chain_type == ChainType.FISSION:
        reactions, seen = [], set()
        for nuc in current_nuclei:
            if nuc == "n":
                continue
            for r in get_fission_products(nuc):
                rid = id(r)
                if rid not in seen:
                    seen.add(rid)
                    reactions.append(r)
        return reactions
    elif chain_type == ChainType.FUSION:
        reactions = get_fusion_reactions(current_nuclei)
        seen, unique = set(), []
        for r in reactions:
            rid = id(r)
            if rid not in seen:
                seen.add(rid)
                unique.append(r)
        return unique
    return []

def get_starting_isotopes(chain_type: ChainType) -> list[str]:
    if chain_type == ChainType.DECAY:
        return ["U-238", "U-235", "Th-232", "C-14", "Cs-137"]
    elif chain_type == ChainType.FISSION:
        return ["U-235", "U-238", "Pu-239"]
    elif chain_type == ChainType.FUSION:
        return ["H-1", "H-1"]
    return []


ELEMENT_COLORS = {
    "U":"#e74c3c","Th":"#e67e22","Pa":"#f39c12","Ra":"#d35400",
    "Rn":"#c0392b","Po":"#e74c3c","Pb":"#95a5a6","Bi":"#bdc3c7",
    "Tl":"#7f8c8d","Ac":"#d4ac0d","Pu":"#9b59b6",
    "H":"#3498db","He":"#2ecc71","Li":"#1abc9c","Be":"#16a085",
    "B":"#27ae60","C":"#2c3e50","N":"#2980b9","O":"#e74c3c",
    "F":"#8e44ad","Ne":"#f1c40f",
    "Sr":"#e67e22","Y":"#f1c40f","Zr":"#3498db","Mo":"#9b59b6",
    "Ru":"#1abc9c","Rh":"#e74c3c","Pd":"#95a5a6","Ag":"#bdc3c7",
    "Cd":"#34495e","In":"#7f8c8d","Sn":"#2c3e50","Sb":"#d35400",
    "Te":"#e67e22","I":"#9b59b6","Xe":"#3498db","Cs":"#e74c3c",
    "Ba":"#27ae60","La":"#16a085","Ce":"#2980b9","Pr":"#8e44ad",
    "Nd":"#f39c12","Ge":"#34495e","Kr":"#2ecc71","Br":"#e74c3c",
    "Rb":"#9b59b6","Fe":"#e67e22","n":"#f1c40f",
}
DEFAULT_COLOR = "#7f8c8d"

def get_element_color(symbol: str) -> str:
    elem = symbol.split("-")[0].replace("+", "").strip()
    for key, color in ELEMENT_COLORS.items():
        if elem.startswith(key):
            return color
    return DEFAULT_COLOR


##############################################################################
# 2. MCTS ENGINE
##############################################################################

@dataclass
class MCTSNode:
    state: NuclearState
    parent: Optional[MCTSNode]
    action: Optional[Reaction]
    children: list[MCTSNode]
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[Reaction] = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) == 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.avg_reward
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploitation + exploration_term


@dataclass
class NuclearState:
    nuclei: list[str]
    chain_type: ChainType
    depth: int = 0
    total_energy: float = 0.0
    history: list[Reaction] = field(default_factory=list)
    visited_symbols: set[str] = field(default_factory=set)

    def copy(self) -> NuclearState:
        return NuclearState(
            nuclei=list(self.nuclei), chain_type=self.chain_type,
            depth=self.depth + 1, total_energy=self.total_energy,
            history=list(self.history), visited_symbols=set(self.visited_symbols))

    def get_available_reactions(self) -> list[Reaction]:
        max_d = 15 if self.chain_type == ChainType.FUSION else 50
        if self.depth >= max_d:
            return []
        return get_reactions_for(self.chain_type, self.nuclei)

    def apply_reaction(self, reaction: Reaction) -> NuclearState:
        new_state = self.copy()
        new_state.history.append(reaction)
        new_state.total_energy += reaction.Q_value
        for reactant in reaction.reactants:
            if reactant in new_state.nuclei:
                new_state.nuclei.remove(reactant)
        skip_tokens = {"e+", "v_e", "n", "gamma", "2n", "3n"}
        for product in reaction.products:
            if product not in skip_tokens:
                new_state.nuclei.append(product)
                new_state.visited_symbols.add(product)
        if self.chain_type == ChainType.FISSION and "n" in reaction.products:
            new_state.nuclei.append("n")
        return new_state

    def is_terminal(self) -> bool:
        return len(self.get_available_reactions()) == 0

    def reward(self) -> float:
        r = 0.0
        r += self.total_energy * (0.1 if self.chain_type == ChainType.FUSION else 0.01)
        for nuc in self.nuclei:
            if nuc in ISOTOPES and ISOTOPES[nuc].is_stable:
                r += 10.0
                if ISOTOPES[nuc].Z == 26 and ISOTOPES[nuc].A == 56:
                    r += 50.0
        r -= self.depth * (0.8 if self.chain_type == ChainType.FUSION else 0.1)
        if self.chain_type == ChainType.FUSION:
            if "C-12" in self.nuclei and "He-4" in self.nuclei:
                r += 30.0
            if "He-4" in self.nuclei:
                r += 20.0
            has_triple_alpha = ("C-12" in self.nuclei and len(self.history) > 0
                                and "He-4" in self.history[0].reactants)
            if has_triple_alpha:
                r += 25.0
        r += len(self.visited_symbols) * 0.5
        if self.chain_type == ChainType.FUSION and "He-4" in self.nuclei:
            r += 20.0
        if self.chain_type == ChainType.DECAY:
            for nuc in self.nuclei:
                if nuc in ISOTOPES and ISOTOPES[nuc].is_stable and ISOTOPES[nuc].Z == 82:
                    r += 30.0
        return r


class MCTSEngine:
    def __init__(self, chain_type: ChainType, initial_nuclei: list[str],
                 max_iterations: int = 2000, max_time: float = 5.0,
                 exploration_constant: float = 1.414):
        self.chain_type = chain_type
        self.initial_nuclei = list(initial_nuclei)
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.exploration = exploration_constant
        self.root: Optional[MCTSNode] = None
        self.best_path: list[Reaction] = []
        self.best_reward: float = float('-inf')
        self.iterations_done = 0

    def search(self) -> list[Reaction]:
        initial_state = NuclearState(
            nuclei=list(self.initial_nuclei), chain_type=self.chain_type,
            visited_symbols=set(self.initial_nuclei))
        self.root = MCTSNode(state=initial_state, parent=None, action=None,
                             children=[], untried_actions=initial_state.get_available_reactions())
        self.iterations_done = 0
        start_time = time.time()
        while self.iterations_done < self.max_iterations:
            if time.time() - start_time > self.max_time:
                break
            node = self._select(self.root)
            reward = self._evaluate(node)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_path = self._extract_path(node)
            self.iterations_done += 1
        if self.root and self.root.children:
            best_leaf = self._get_best_leaf(self.root)
            self.best_path = self._extract_path(best_leaf)
            self.best_reward = best_leaf.state.reward()
        return self.best_path

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal and node.is_fully_expanded:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration))
        if not node.is_fully_expanded and not node.is_terminal:
            return self._expand(node)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            return node
        weights = [r.probability for r in node.untried_actions]
        total = sum(weights)
        if total == 0:
            action = random.choice(node.untried_actions)
        else:
            r = random.random() * total
            cumulative = 0
            action = node.untried_actions[-1]
            for a, w in zip(node.untried_actions, weights):
                cumulative += w
                if r <= cumulative:
                    action = a
                    break
        node.untried_actions.remove(action)
        new_state = node.state.apply_reaction(action)
        child = MCTSNode(state=new_state, parent=node, action=action, children=[],
                         untried_actions=new_state.get_available_reactions())
        node.children.append(child)
        return child

    def _evaluate(self, node: MCTSNode) -> float:
        reward = self._rollout(node.state)
        self._backpropagate(node, reward)
        return reward

    def _rollout(self, state: NuclearState, max_depth: int = 30) -> float:
        current = state
        depth = 0
        while depth < max_depth:
            reactions = current.get_available_reactions()
            if not reactions:
                break
            weights = [r.probability for r in reactions]
            total = sum(weights)
            if total == 0:
                break
            r = random.random() * total
            cumulative = 0
            action = reactions[-1]
            for react, w in zip(reactions, weights):
                cumulative += w
                if r <= cumulative:
                    action = react
                    break
            current = current.apply_reaction(action)
            depth += 1
        return current.reward()

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _get_best_leaf(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
        return node

    def _extract_path(self, node: MCTSNode) -> list[Reaction]:
        path = []
        while node and node.action is not None:
            path.append(node.action)
            node = node.parent
        path.reverse()
        return path

    def get_tree_stats(self) -> dict:
        if not self.root:
            return {"iterations": 0, "total_nodes": 0, "max_depth": 0,
                    "best_reward": 0, "path_length": 0, "total_energy_MeV": 0}
        total_nodes, max_depth = [0], [0]
        def count(node, d):
            total_nodes[0] += 1
            if d > max_depth[0]: max_depth[0] = d
            for c in node.children: count(c, d + 1)
        count(self.root, 0)
        return {"iterations": self.iterations_done, "total_nodes": total_nodes[0],
                "max_depth": max_depth[0], "best_reward": self.best_reward,
                "path_length": len(self.best_path),
                "total_energy_MeV": sum(r.Q_value for r in self.best_path)}

    def get_tree_for_visualization(self) -> dict:
        if not self.root:
            return {"nodes": [], "edges": []}
        nodes, edges, max_vis_depth = [], [], 12
        queue = [(self.root, 0)]
        node_id = {id(self.root): 0}
        next_id = [1]
        while queue:
            node, depth = queue.pop(0)
            if depth > max_vis_depth or node.visits < 1:
                continue
            nid = node_id[id(node)]
            label = ", ".join(node.state.nuclei[:3])
            if len(node.state.nuclei) > 3:
                label += f" (+{len(node.state.nuclei)-3} more)"
            nodes.append({"id": nid, "label": label or "root", "depth": depth,
                          "visits": node.visits, "avg_reward": round(node.avg_reward, 2),
                          "is_root": node.parent is None,
                          "energy": round(node.state.total_energy, 2),
                          "n_nuclei": len(node.state.nuclei)})
            for child in sorted(node.children, key=lambda c: c.visits, reverse=True)[:4]:
                if child.visits < 1:
                    continue
                cid = next_id[0]; next_id[0] += 1
                node_id[id(child)] = cid
                action_label = child.action.description[:50] if child.action else ""
                if child.action and child.action.reaction_type:
                    action_label = f"[{child.action.reaction_type.value}] {action_label}"
                edges.append({"source": nid, "target": cid, "action": action_label,
                              "visits": child.visits,
                              "probability": child.action.probability if child.action else 0,
                              "Q_value": child.action.Q_value if child.action else 0})
                queue.append((child, depth + 1))
        return {"nodes": nodes, "edges": edges}


##############################################################################
# 3. MCTS WORKER THREAD
##############################################################################

class MCTSWorker(QThread):
    progress = Signal(int, int)
    finished_signal = Signal(dict)
    tree_ready = Signal(dict)
    path_ready = Signal(list, float)

    def __init__(self, chain_type: ChainType, initial_nuclei: list[str],
                 max_iterations: int = 2000, max_time: float = 5.0,
                 exploration: float = 1.414):
        super().__init__()
        self.chain_type = chain_type
        self.initial_nuclei = initial_nuclei
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.exploration = exploration
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        engine = MCTSEngine(self.chain_type, self.initial_nuclei,
                            self.max_iterations, self.max_time, self.exploration)
        path = engine.search()
        stats = engine.get_tree_stats()
        tree = engine.get_tree_for_visualization()
        self.path_ready.emit(path, engine.best_reward)
        self.tree_ready.emit(tree)
        self.finished_signal.emit(stats)


##############################################################################
# 4. GUI COMPONENTS
##############################################################################

class IsotopeNodeItem(QGraphicsEllipseItem):
    def __init__(self, symbol, x, y, radius=30, is_start=False, is_stable=False,
                 visit_count=0, energy=0):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.symbol = symbol
        self.radius = radius
        self.is_start = is_start
        self.is_stable = is_stable
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        color = get_element_color(symbol)
        base_color = QColor(color)
        if is_start:
            g = QRadialGradient(0, 0, radius)
            g.setColorAt(0, QColor("#ffeb3b")); g.setColorAt(0.7, QColor("#ff9800"))
            g.setColorAt(1, QColor("#e65100"))
            self.setBrush(QBrush(g)); self.setPen(QPen(QColor("#fff"), 3))
        elif is_stable:
            g = QRadialGradient(0, 0, radius)
            g.setColorAt(0, QColor("#e8f5e9")); g.setColorAt(0.6, QColor("#66bb6a"))
            g.setColorAt(1, QColor("#2e7d32"))
            self.setBrush(QBrush(g)); self.setPen(QPen(QColor("#1b5e20"), 3))
        else:
            g = QRadialGradient(0, -radius * 0.3, radius * 1.2)
            g.setColorAt(0, base_color.lighter(150)); g.setColorAt(0.6, base_color)
            g.setColorAt(1, base_color.darker(150))
            self.setBrush(QBrush(g)); self.setPen(QPen(base_color.darker(180), 2))

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        painter.setPen(QPen(QColor("white")))
        painter.drawText(QRectF(-self.radius, -self.radius, self.radius*2, self.radius*2),
                         Qt.AlignCenter, self.symbol)


class ReactionArrowItem(QGraphicsPathItem):
    def __init__(self, start, end, label="", color="#90a4ae", q_value=0, visits=0):
        super().__init__()
        self._start, self._end, self.label = start, end, label
        self.q_value, self.visits = q_value, visits
        path = QPainterPath(); path.moveTo(start)
        dx, dy = end.x() - start.x(), end.y() - start.y()
        dist = math.sqrt(dx*dx + dy*dy) or 1
        offset = min(dist * 0.3, 60)
        ctrl = QPointF((start.x()+end.x())/2 + (-dy/dist*offset),
                       (start.y()+end.y())/2 + (dx/dist*offset))
        path.cubicTo(ctrl, ctrl, end); self.setPath(path)
        alpha = min(80 + visits * 3, 255)
        c = QColor(color); c.setAlpha(alpha)
        self.setPen(QPen(c, min(1.5 + visits * 0.2, 6)))
        self.setBrush(QBrush(Qt.NoBrush))

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if not self.label:
            return
        mid = QPointF((self._start.x()+self._end.x())/2,
                      (self._start.y()+self._end.y())/2 - 12)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#546e7a")))
        text = self.label + (f" ({self.q_value:.1f} MeV)" if self.q_value > 0 else "")
        painter.drawText(mid, text)


class ChainPathView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#1a1a2e")))
        self.setMinimumHeight(200)

    def clear_view(self):
        self._scene.clear()

    def display_path(self, path, chain_type, initial_nuclei):
        self.clear_view()
        if not path:
            self._scene.addText("No path found. Run MCTS search.", QFont("Segoe UI", 14))
            self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-50,-50,50,50))
            return
        x_offset, y_center, node_spacing, node_radius = 80, 100, 180, 32
        for i, nuc in enumerate(initial_nuclei):
            self._scene.addItem(IsotopeNodeItem(nuc, x_offset + i*50, y_center, node_radius, is_start=True))
        current_x, all_nodes, nuclei_positions = x_offset + max(len(initial_nuclei)*50, 50), [], {}
        colors = {ChainType.DECAY:"#ff7043", ChainType.FISSION:"#42a5f5", ChainType.FUSION:"#66bb6a"}
        for step_idx, reaction in enumerate(path):
            y = y_center + (40 if step_idx % 2 == 0 else -40)
            for i, product in enumerate(reaction.products):
                if product in ("e+","v_e","gamma","2n","3n","n"):
                    continue
                px = current_x + i * 70
                while f"{product}_{int(px)}" in nuclei_positions:
                    px += 70
                nuclei_positions[f"{product}_{int(px)}"] = True
                is_stable = (product in ISOTOPES and ISOTOPES[product].is_stable)
                node = IsotopeNodeItem(product, px, y, node_radius, is_stable=is_stable)
                self._scene.addItem(node); all_nodes.append((node, px, y))
            if all_nodes:
                prev_node, prev_x, prev_py = all_nodes[-1]
                mode = reaction.description.split("(")[-1].rstrip(")") if "(" in reaction.description else ""
                if chain_type != ChainType.DECAY:
                    mode = reaction.reaction_type.value
                self._scene.addItem(ReactionArrowItem(QPointF(prev_x, prev_py), QPointF(current_x, y),
                    label=mode, color=colors.get(chain_type,"#90a4ae"), q_value=reaction.Q_value, visits=10))
            current_x += node_spacing + max(len(reaction.products)-1, 0) * 70
        title = self._scene.addText(
            f"{chain_type.value}  -  {len(path)} reactions  |  "
            f"Total Energy: {sum(r.Q_value for r in path):.2f} MeV",
            QFont("Segoe UI", 11, QFont.Bold))
        title.setDefaultTextColor(QColor("#e0e0e0")); title.setPos(x_offset - 40, 15)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-80,-60,80,60))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


class MCTSTreeView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#0d1117")))
        self.setMinimumHeight(300)

    def clear_view(self):
        self._scene.clear()

    def display_tree(self, tree_data, chain_type):
        self.clear_view()
        if not tree_data or "nodes" not in tree_data or not tree_data["nodes"]:
            self._scene.addText("No tree data. Run MCTS search first.", QFont("Segoe UI", 12))
            return
        nodes, edges = tree_data["nodes"], tree_data["edges"]
        chain_colors = {ChainType.DECAY:"#ff7043", ChainType.FISSION:"#42a5f5", ChainType.FUSION:"#66bb6a"}
        h_spacing, v_spacing = 200, 100
        max_visits = max((n["visits"] for n in nodes), default=1) or 1
        depth_groups = defaultdict(list)
        for n in nodes: depth_groups[n["depth"]].append(n)
        node_positions = {}
        for depth, group in depth_groups.items():
            width = len(group) * h_spacing; start_x = -width/2 + h_spacing/2
            for i, node in enumerate(group):
                node_positions[node["id"]] = QPointF(start_x + i*h_spacing, depth*v_spacing + 60)
        node_map = {n["id"]: n for n in nodes}
        for edge in edges:
            src_pos, tgt_pos = node_positions.get(edge["source"]), node_positions.get(edge["target"])
            if not src_pos or not tgt_pos: continue
            base_c = QColor(chain_colors.get(chain_type,"#90a4ae"))
            alpha = min(60 + int(edge["visits"]/max_visits*150), 220)
            base_c.setAlpha(alpha)
            self._scene.addLine(src_pos.x(),src_pos.y(),tgt_pos.x(),tgt_pos.y(),
                QPen(base_c, max(1, min(edge["visits"]/max_visits*5, 5))))
            label = self._scene.addText(
                f"{edge.get('Q_value',0):.1f} MeV (v={edge['visits']})", QFont("Segoe UI", 6))
            label.setDefaultTextColor(QColor("#78909c"))
            label.setPos(QPointF((src_pos.x()+tgt_pos.x())/2,(src_pos.y()+tgt_pos.y())/2-8))
        for node in nodes:
            pos = node_positions[node["id"]]
            v_ratio = node["visits"] / max_visits
            radius = 18 + v_ratio * 20
            reward = node.get("avg_reward", 0)
            nc = QColor("#4caf50" if reward>20 else "#ff9800" if reward>10 else "#2196f3" if reward>0 else "#607d8b")
            g = QRadialGradient(pos, radius)
            g.setColorAt(0, nc.lighter(160)); g.setColorAt(0.7, nc); g.setColorAt(1, nc.darker(150))
            self._scene.addEllipse(pos.x()-radius, pos.y()-radius, radius*2, radius*2,
                QPen(nc.darker(130),2), QBrush(g))
            label_text = node["label"][:12] + (".." if len(node["label"])>12 else "")
            text = self._scene.addText(label_text, QFont("Consolas", 7, QFont.Bold))
            text.setDefaultTextColor(QColor("white"))
            r = text.boundingRect(); text.setPos(pos.x()-r.width()/2, pos.y()-r.height()/2)
            vt = self._scene.addText(f"v={node['visits']}", QFont("Segoe UI", 6))
            vt.setDefaultTextColor(QColor("#90a4ae")); vr = vt.boundingRect()
            vt.setPos(pos.x()-vr.width()/2, pos.y()+radius+2)
        title = self._scene.addText(
            f"MCTS Search Tree  |  Nodes: {len(nodes)}  |  Edges: {len(edges)}",
            QFont("Segoe UI", 10, QFont.Bold))
        title.setDefaultTextColor(QColor("#b0bec5")); title.setPos(-200, 10)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-100,-50,100,100))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


class ReactionLogPanel(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit { background-color: #1a1a2e; color: #e0e0e0;
                font-family: 'Consolas','Courier New',monospace; font-size: 11px;
                border: 1px solid #2a2a4a; border-radius: 4px; padding: 8px; }""")

    def clear(self):
        self.setHtml("<span style='color:#546e7a;'>Reaction log will appear here after MCTS search...</span>")

    def display_path(self, path, chain_type, reward):
        if not path:
            self.setHtml("<span style='color:#ef5350;'>No reaction path found.</span>"); return
        colors = {ChainType.DECAY:("#ff7043","#ffab91"), ChainType.FISSION:("#42a5f5","#90caf9"),
                  ChainType.FUSION:("#66bb6a","#a5d6a7")}
        mc, lc = colors.get(chain_type, ("#90a4ae","#cfd8dc"))
        html = (f"<h3 style='color:#42a5f5;margin:4px 0;'>{chain_type.value} - Best Path</h3>"
                f"<p style='color:#78909c;'>MCTS Reward: {reward:.2f}</p><hr style='border-color:#2a2a4a;'>")
        total_energy = 0
        for i, reaction in enumerate(path):
            total_energy += reaction.Q_value
            html += (f"<div style='margin:6px 0;padding:6px;background-color:{lc}11;"
                     f"border-left:3px solid {mc};border-radius:2px;'>"
                     f"<b style='color:{mc};'>Step {i+1}:</b> "
                     f"<span style='color:#e0e0e0;'>{reaction.description}</span><br>"
                     f"<span style='color:#78909c;font-size:10px;'>"
                     f"Q = {reaction.Q_value:.3f} MeV  |  "
                     f"Reactants: {' + '.join(reaction.reactants)}  &rarr;  "
                     f"Products: {' + '.join(reaction.products)}</span></div>")
        html += (f"<hr style='border-color:#2a2a4a;'>"
                 f"<p style='color:#66bb6a;font-weight:bold;'>"
                 f"Total Steps: {len(path)}  |  Total Energy: {total_energy:.3f} MeV</p>")
        self.setHtml(html)


class ControlPanel(QWidget):
    search_requested = Signal(ChainType, list, int, float, float)
    abort_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280); self.setMaximumWidth(350)
        layout = QVBoxLayout(self); layout.setSpacing(8); layout.setContentsMargins(8,8,8,8)

        type_group = QGroupBox("Chain Type"); tl = QVBoxLayout(type_group)
        self.chain_combo = QComboBox()
        self.chain_combo.addItems([ct.value for ct in ChainType])
        self.chain_combo.currentIndexChanged.connect(self._on_chain_type_changed)
        tl.addWidget(self.chain_combo); layout.addWidget(type_group)

        start_group = QGroupBox("Starting Nuclei"); sl = QGridLayout(start_group)
        self.start_combo = QComboBox()
        sl.addWidget(QLabel("Primary:"), 0, 0); sl.addWidget(self.start_combo, 0, 1)
        self.start_combo2 = QComboBox(); self.start_combo2.addItem("(none)")
        sl.addWidget(QLabel("Secondary:"), 1, 0); sl.addWidget(self.start_combo2, 1, 1)
        self._populate_starting_isotopes(ChainType.DECAY)
        layout.addWidget(start_group)

        mcts_group = QGroupBox("MCTS Parameters"); ml = QGridLayout(mcts_group)
        ml.addWidget(QLabel("Iterations:"), 0, 0)
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(100,50000)
        self.iter_spin.setValue(2000); self.iter_spin.setSingleStep(500)
        ml.addWidget(self.iter_spin, 0, 1)
        ml.addWidget(QLabel("Time Limit (s):"), 1, 0)
        self.time_spin = QDoubleSpinBox(); self.time_spin.setRange(0.5,60.0)
        self.time_spin.setValue(5.0); self.time_spin.setSingleStep(0.5)
        ml.addWidget(self.time_spin, 1, 1)
        ml.addWidget(QLabel("Exploration (C):"), 2, 0)
        self.explore_spin = QDoubleSpinBox(); self.explore_spin.setRange(0.1,5.0)
        self.explore_spin.setValue(1.414); self.explore_spin.setSingleStep(0.1)
        self.explore_spin.setToolTip("UCB1 exploration constant (sqrt(2) ~ 1.414)")
        ml.addWidget(self.explore_spin, 2, 1)
        layout.addWidget(mcts_group)

        self.run_btn = QPushButton("  Run MCTS Search"); self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet(
            "QPushButton{background:#1976d2;color:white;font-size:14px;font-weight:bold;"
            "border-radius:6px;border:none}QPushButton:hover{background:#1565c0}"
            "QPushButton:pressed{background:#0d47a1}QPushButton:disabled{background:#546e7a}")
        self.run_btn.clicked.connect(self._on_run); layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("  Stop"); self.stop_btn.setEnabled(False); self.stop_btn.setMinimumHeight(32)
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#e53935;color:white;font-size:12px;font-weight:bold;"
            "border-radius:6px;border:none}QPushButton:hover{background:#c62828}"
            "QPushButton:disabled{background:#546e7a}")
        self.stop_btn.clicked.connect(self._on_stop); layout.addWidget(self.stop_btn)

        stats_group = QGroupBox("Search Statistics"); stl = QVBoxLayout(stats_group)
        self.stats_label = QLabel("Run a search to see statistics.")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color:#b0bec5;font-size:11px;")
        stl.addWidget(self.stats_label); layout.addWidget(stats_group)

        info_group = QGroupBox("Isotope Information"); il = QVBoxLayout(info_group)
        self.info_label = QLabel("Select an isotope in the path view.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color:#90a4ae;font-size:10px;")
        il.addWidget(self.info_label); layout.addWidget(info_group)
        layout.addStretch()

    def _populate_starting_isotopes(self, chain_type):
        self.start_combo.clear(); self.start_combo2.clear()
        self.start_combo.addItems(get_starting_isotopes(chain_type))
        if chain_type == ChainType.FUSION:
            for iso in ["H-1","He-4","C-12","C-13","N-14"]: self.start_combo2.addItem(iso)
        elif chain_type == ChainType.FISSION:
            self.start_combo2.addItem("n")
        else:
            self.start_combo2.addItem("(none)")

    def _on_chain_type_changed(self, index):
        cts = list(ChainType)
        if 0 <= index < len(cts): self._populate_starting_isotopes(cts[index])

    def _on_run(self):
        idx = self.chain_combo.currentIndex()
        if idx < 0: return
        cts = list(ChainType); chain_type = cts[idx]
        nuclei = [self.start_combo.currentText()]
        sec = self.start_combo2.currentText()
        if sec and sec != "(none)": nuclei.append(sec)
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.search_requested.emit(chain_type, nuclei, self.iter_spin.value(),
                                  self.time_spin.value(), self.explore_spin.value())

    def _on_stop(self):
        self.abort_requested.emit()

    def set_search_finished(self):
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    def update_stats(self, stats):
        self.stats_label.setText(
            f"<b>MCTS Results:</b><br>Iterations: {stats.get('iterations',0)}<br>"
            f"Tree Nodes: {stats.get('total_nodes',0)}<br>"
            f"Max Depth: {stats.get('max_depth',0)}<br>"
            f"Best Reward: {stats.get('best_reward',0):.2f}<br>"
            f"Path Length: {stats.get('path_length',0)} steps<br>"
            f"Total Energy: {stats.get('total_energy_MeV',0):.2f} MeV")

    def show_isotope_info(self, symbol):
        if symbol not in ISOTOPES:
            self.info_label.setText(f"<b>{symbol}</b><br>Not in database"); return
        iso = ISOTOPES[symbol]
        self.info_label.setText(
            f"<b>{iso.name}</b><br>Symbol: {iso.symbol}<br>"
            f"Z = {iso.Z}  |  A = {iso.A}  |  N = {iso.N}<br>"
            f"Half-life: {iso.half_life_str}<br>Decay Mode: {iso.decay_mode.value}<br>"
            f"Decay Energy: {iso.decay_energy:.3f} MeV<br>"
            f"BE/A (est): {iso.binding_energy_per_nucleon:.3f} MeV<br>"
            f"Stable: {'Yes' if iso.is_stable else 'No'}")


##############################################################################
# 5. MAIN WINDOW
##############################################################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuclear Chain Reaction Explorer with MCTS")
        self.setMinimumSize(1200, 800); self.resize(1400, 900)
        self._worker: Optional[MCTSWorker] = None
        self._current_path: list[Reaction] = []
        self._current_chain_type = ChainType.DECAY
        self._current_nuclei: list[str] = []
        self._setup_style(); self._build_ui(); self._build_menu()

    def _setup_style(self):
        self.setStyleSheet("""
            QMainWindow{background:#121212}
            QGroupBox{color:#b0bec5;font-weight:bold;border:1px solid #2a2a4a;
                border-radius:6px;margin-top:10px;padding-top:14px}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px}
            QLabel{color:#cfd8dc;font-size:11px}
            QComboBox{background:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a;
                border-radius:4px;padding:4px 8px;min-height:24px}
            QComboBox::drop-down{border:none;width:24px}
            QComboBox QAbstractItemView{background:#1e1e3a;color:#e0e0e0;selection-background-color:#1976d2}
            QSpinBox,QDoubleSpinBox{background:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a;
                border-radius:4px;padding:2px 6px}
            QTabWidget::pane{border:1px solid #2a2a4a;border-radius:4px;background:#0d1117}
            QTabBar::tab{background:#1a1a2e;color:#78909c;padding:8px 20px;border:1px solid #2a2a4a;
                border-bottom:none;border-top-left-radius:6px;border-top-right-radius:6px;margin-right:2px}
            QTabBar::tab:selected{background:#0d1117;color:#42a5f5;font-weight:bold}
            QTabBar::tab:hover:!selected{background:#252545;color:#b0bec5}
            QSplitter::handle{background:#2a2a4a;width:3px}
            QStatusBar{color:#546e7a;background:#0a0a14}
            QMenuBar{background:#0d1117;color:#b0bec5}
            QMenuBar::item:selected{background:#1976d2}
            QMenu{background:#1a1a2e;color:#e0e0e0;border:1px solid #2a2a4a}
            QMenu::item:selected{background:#1976d2}
            QToolBar{background:#0d1117;border-bottom:1px solid #2a2a4a;spacing:6px}""")

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6,6,6,6); main_layout.setSpacing(6)
        self.control_panel = ControlPanel()
        self.control_panel.search_requested.connect(self._start_search)
        self.control_panel.abort_requested.connect(self._stop_search)
        main_layout.addWidget(self.control_panel)
        right_splitter = QSplitter(Qt.Vertical)
        self.viz_tabs = QTabWidget()
        self.chain_path_view = ChainPathView()
        self.mcts_tree_view = MCTSTreeView()
        self.viz_tabs.addTab(self.chain_path_view, "  Reaction Chain Path")
        self.viz_tabs.addTab(self.mcts_tree_view, "  MCTS Search Tree")
        right_splitter.addWidget(self.viz_tabs)
        self.reaction_log = ReactionLogPanel(); self.reaction_log.setMaximumHeight(300)
        right_splitter.addWidget(self.reaction_log)
        right_splitter.setStretchFactor(0, 3); right_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(right_splitter, stretch=1)
        self.statusBar().showMessage("Ready. Select a chain type and run MCTS search.")
        self.progress_bar = QProgressBar(); self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:3px}"
            "QProgressBar::chunk{background:#1976d2;border-radius:2px}")
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        exit_action = QAction("E&xit", self); exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)
        view_menu = menubar.addMenu("&View")
        for name, slot in [("Zoom &In", self._zoom_in), ("Zoom &Out", self._zoom_out),
                           ("&Fit View", self._fit_view)]:
            a = QAction(name, self)
            if "In" in name: a.setShortcut(QKeySequence.ZoomIn)
            elif "Out" in name: a.setShortcut(QKeySequence.ZoomOut)
            else: a.setShortcut(QKeySequence("Ctrl+0"))
            a.triggered.connect(slot); view_menu.addAction(a)
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self); about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        toolbar = QToolBar("Quick Start"); toolbar.setMovable(False); self.addToolBar(toolbar)
        for name, ct, nuclei in [
            ("U-238 Decay", ChainType.DECAY, ["U-238"]),
            ("U-235 Decay", ChainType.DECAY, ["U-235"]),
            ("Th-232 Decay", ChainType.DECAY, ["Th-232"]),
            ("U-235 Fission", ChainType.FISSION, ["U-235","n"]),
            ("Pu-239 Fission", ChainType.FISSION, ["Pu-239","n"]),
            ("PP-Chain (Sun)", ChainType.FUSION, ["H-1"]*6),
            ("CNO Cycle", ChainType.FUSION, ["C-12","H-1","H-1","H-1","H-1"]),
            ("Triple-Alpha", ChainType.FUSION, ["He-4","He-4","He-4"]),
        ]:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, c=ct, n=nuclei: self._quick_start(c, n))
            toolbar.addAction(action)

    def _quick_start(self, chain_type, nuclei):
        self.control_panel.chain_combo.blockSignals(True)
        self.control_panel.chain_combo.setCurrentIndex(list(ChainType).index(chain_type))
        self.control_panel.chain_combo.blockSignals(False)
        self.control_panel.start_combo.blockSignals(True)
        self.control_panel.start_combo.setCurrentText(nuclei[0] if nuclei else "")
        self.control_panel.start_combo.blockSignals(False)
        self.control_panel.start_combo2.blockSignals(True)
        if len(nuclei) > 1:
            self.control_panel.start_combo2.setCurrentText(nuclei[1])
        else:
            self.control_panel.start_combo2.setCurrentIndex(0)
        self.control_panel.start_combo2.blockSignals(False)
        self._start_search(chain_type, nuclei, 2000, 5.0, 1.414)

    def _start_search(self, chain_type, nuclei, iterations, time_limit, exploration):
        if self._worker and self._worker.isRunning():
            self._worker.abort(); self._worker.wait()
        self._current_chain_type = chain_type; self._current_nuclei = nuclei
        self.statusBar().showMessage(f"Running MCTS for {chain_type.value} with {iterations} iterations...")
        self.progress_bar.setValue(0)
        self._worker = MCTSWorker(chain_type, nuclei, iterations, time_limit, exploration)
        self._worker.path_ready.connect(self._on_path_found, Qt.QueuedConnection)
        self._worker.tree_ready.connect(self._on_tree_ready, Qt.QueuedConnection)
        self._worker.finished_signal.connect(self._on_search_finished, Qt.QueuedConnection)
        self._worker.finished.connect(self.control_panel.set_search_finished, Qt.QueuedConnection)
        self._worker.start()

    def _stop_search(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self.statusBar().showMessage("Search stopped by user.")

    def _on_path_found(self, path, reward):
        self._current_path = path
        self.chain_path_view.display_path(path, self._current_chain_type, self._current_nuclei)
        self.reaction_log.display_path(path, self._current_chain_type, reward)

    def _on_tree_ready(self, tree_data):
        self.mcts_tree_view.display_tree(tree_data, self._current_chain_type)

    def _on_search_finished(self, stats):
        self.control_panel.update_stats(stats); self.progress_bar.setValue(100)
        self.statusBar().showMessage(
            f"MCTS complete: {stats.get('path_length',0)} steps, "
            f"{stats.get('total_energy_MeV',0):.2f} MeV total energy.")

    def _zoom_in(self):
        c = self.viz_tabs.currentWidget()
        if isinstance(c, (ChainPathView, MCTSTreeView)): c.scale(1.2, 1.2)

    def _zoom_out(self):
        c = self.viz_tabs.currentWidget()
        if isinstance(c, (ChainPathView, MCTSTreeView)): c.scale(1/1.2, 1/1.2)

    def _fit_view(self):
        c = self.viz_tabs.currentWidget()
        if isinstance(c, (ChainPathView, MCTSTreeView)):
            c.fitInView(c.sceneRect(), Qt.KeepAspectRatio)

    def _show_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "About Nuclear Chain Explorer",
            "<h2>Nuclear Chain Reaction Explorer</h2>"
            "<p>Visualizes nuclear decay chains, fission chains, and fusion chains "
            "using <b>Monte Carlo Tree Search (MCTS)</b>.</p>"
            "<h3>Features:</h3><ul>"
            "<li><b>Decay Chains:</b> U-238, U-235, Th-232 series with alpha/beta decay</li>"
            "<li><b>Fission Chains:</b> Neutron-induced fission of U-235, U-238, Pu-239</li>"
            "<li><b>Fusion Chains:</b> pp-chain, CNO cycle, triple-alpha process</li>"
            "<li><b>MCTS:</b> Explores reaction space with UCB1 selection and weighted simulation</li>"
            "</ul><p>Built with Python & PySide6</p>")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.abort(); self._worker.wait(3000)
        event.accept()


##############################################################################
# 6. ENTRY POINT
##############################################################################

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#121212"))
    palette.setColor(QPalette.WindowText, QColor("#e0e0e0"))
    palette.setColor(QPalette.Base, QColor("#0d1117"))
    palette.setColor(QPalette.AlternateBase, QColor("#1a1a2e"))
    palette.setColor(QPalette.Text, QColor("#e0e0e0"))
    palette.setColor(QPalette.Button, QColor("#1e1e3a"))
    palette.setColor(QPalette.ButtonText, QColor("#e0e0e0"))
    palette.setColor(QPalette.Highlight, QColor("#1976d2"))
    palette.setColor(QPalette.HighlightedText, QColor("white"))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()