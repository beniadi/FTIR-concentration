# -*- coding: utf-8 -*-
"""HITRAN line lists and partition sums through HAPI (pip install hitran-api).

    molecules()                         {name: [(global_id, local_iso, iso_name, abundance), ...]}
    fetch(molecule, numin, numax, isos) download into Input/HITRAN, return the .data path
    molecule_name(mol)                  HITRAN molecule number -> name
    partition_ratio(mol, iso, T)        Q(296 K) / Q(T) from TIPS (exact, per isotopologue)
    iso_mass(mol, iso)                  molecular mass (amu), for the Doppler width

The downloaded .data file is the native 160-character HITRAN format, which
concentration.load_hitran reads.  HAPI writes a .header next to it.  HAPI is
imported on first use only, so the rest of the program runs without it.
"""

import contextlib
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
HITRAN_DIR = os.path.join(HERE, "Input", "HITRAN")

_hapi = None


def hapi():
    """The hapi module, imported once."""
    global _hapi
    if _hapi is None:
        try:
            with contextlib.redirect_stdout(io.StringIO()):    # HAPI prints a banner on import
                import hapi as h
        except ImportError as e:
            raise ImportError("HITRAN download needs HAPI:  pip install hitran-api") from e
        _hapi = h
    return _hapi


def available():
    try:
        hapi(); return True
    except ImportError:
        return False


def molecules():
    """Molecule name -> its isotopologues, from HAPI's ISO_ID table, sorted by
    HITRAN molecule number."""
    h = hapi(); ix = h.ISO_ID_INDEX
    out = {}
    for gid, rec in sorted(h.ISO_ID.items(), key=lambda kv: (kv[1][ix["M"]], kv[1][ix["I"]])):
        out.setdefault(rec[ix["mol_name"]], []).append(
            (gid, rec[ix["I"]], rec[ix["iso_name"]], rec[ix["abundance"]]))
    return out


def molecule_name(mol):
    """HITRAN molecule number -> name ('H2O' for 1), None if unknown."""
    h = hapi(); ix = h.ISO_ID_INDEX
    for rec in h.ISO_ID.values():
        if rec[ix["M"]] == int(mol):
            return rec[ix["mol_name"]]
    return None


def table_name(molecule, numin, numax):
    """'N2O_2216.50-2219.00' - safe as a file name."""
    return "%s_%.2f-%.2f" % (re.sub(r"[^A-Za-z0-9+]+", "", molecule), numin, numax)


def fetch(molecule, numin, numax, isos=None, dest=HITRAN_DIR, name=None):
    """Download the lines of one molecule between numin and numax (cm-1).

    isos: local isotopologue numbers to include (1 = most abundant); None = all.
    Returns the path of the .data file in dest."""
    h = hapi()
    table = molecules().get(molecule)
    if not table:
        raise ValueError("unknown HITRAN molecule %r" % molecule)
    ids = [gid for gid, loc, _n, _a in table if isos is None or loc in isos]
    if not ids:
        raise ValueError("no isotopologue %s for %s" % (isos, molecule))
    if not numax > numin:
        raise ValueError("the wavenumber range is empty")
    os.makedirs(dest, exist_ok=True)
    name = name or table_name(molecule, numin, numax)
    # Point HAPI at dest without db_begin, which would also parse every table
    # already in the folder.
    h.VARIABLES["BACKEND_DATABASE_NAME"] = dest
    with contextlib.redirect_stdout(io.StringIO()):
        h.fetch_by_ids(name, ids, float(numin), float(numax))
    h.LOCAL_TABLE_CACHE.pop(name, None)          # keep only the file, not a copy in memory
    path = os.path.join(dest, name + ".data")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError("HITRAN returned no lines for %s in %.4f-%.4f cm-1" % (molecule, numin, numax))
    return path


# HITRAN's one-character isotopologue field: 1-9, then 0 = 10, A = 11, B = 12.
def iso_number(c):
    c = str(c).strip()
    return {"0": 10, "A": 11, "B": 12}.get(c, int(c) if c.isdigit() else 1)


def iso_mass(mol, iso):
    """Molecular mass (amu) of one isotopologue, from HAPI's ISO table."""
    h = hapi()
    return float(h.ISO[(int(mol), iso_number(iso))][h.ISO_INDEX["mass"]])


def partition_ratio(mol, iso, T, T_ref=296.0):
    """Q(T_ref) / Q(T) from the TIPS partition sums shipped with HAPI."""
    h = hapi()
    return float(h.partitionSum(int(mol), iso_number(iso), T_ref) / h.partitionSum(int(mol), iso_number(iso), T))
