"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from boostedhh.processors.utils import (
    GEN_FLAGS,
    P4,
    PDGID,
    add_selection,
    pad_val,
)
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray


def _iterate_children(children, parent_pdgId):
    """Iterates through the children of a particle in case of photon scattering to find the final daughter particles"""

    for _ in range(5):
        # check if any of the children are the same as the parent pdgId
        mask = ak.any(np.abs(children.pdgId) == parent_pdgId, axis=2)
        if not ak.any(mask):
            break

        children_children = ak.flatten(
            children[np.abs(children.pdgId) == parent_pdgId].children, axis=3
        )

        # get next layer of children
        children = ak.where(mask, children_children, children)

    return children


def _sum_taus(taut):
    return ak.sum(taut, axis=1)

def gen_selection_Zll(
    events: NanoEventsArray,
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,
):
    """Gets Z -> ee / mumu 4-vectors + lepton decay information"""
    print("-----------------------GEN FOR LEP---------------------------")

    genparts = events.GenPart[events.GenPart.hasFlags(GEN_FLAGS)]

    # Select Z bosons
    Z = genparts[genparts.pdgId == PDGID.Z]

    # Save Z 4-vector info
    GenZVars = {
        f"GenZ{key}": ak.to_numpy(ak.pad_none(Z[var], 1, clip=True))
        for (var, key) in P4.items()
    }

    # Z children
    Z_children = Z.children
    # Save children pdgIds
    GenZVars["GenZChildren"] = pad_val(
        Z_children.pdgId[:, :, 0], 2, axis=1
    )

    # Identify electrons and muons
    is_ee = np.abs(Z_children.pdgId) == PDGID.e
    is_mumu = np.abs(Z_children.pdgId) == PDGID.mu

    # Event selections
    has_ee = ak.sum(ak.flatten(is_ee, axis=2), axis=1) == 2
    has_mumu = ak.sum(ak.flatten(is_mumu, axis=2), axis=1) == 2
    has_ll = has_ee | has_mumu
    if selection_args is not None:
        add_selection("has_ll", has_ll, *selection_args)

    # Extract leptons
    electrons = Z_children[is_ee]
    muons = Z_children[is_mumu]

    flat_electrons = ak.flatten(electrons, axis=2)
    flat_muons = ak.flatten(muons, axis=2)

    # Save electron variables
    GenElectronVars = {
        f"GenElectron{key}": pad_val(flat_electrons[var], 2, axis=1)
        for (var, key) in P4.items()
    }

    # Save muon variables
    GenMuonVars = {
        f"GenMuon{key}": pad_val(flat_muons[var], 2, axis=1)
        for (var, key) in P4.items()
    }

    # Z->ee object for matching
    Zee = Z[ak.sum(is_ee, axis=2) == 2]
    Zee = ak.pad_none(Zee, 1, axis=1, clip=True)[:, 0]

    # Z->mumu object for matching
    Zmumu = Z[ak.sum(is_mumu, axis=2) == 2]
    Zmumu = ak.pad_none(Zmumu, 1, axis=1, clip=True)[:, 0]

    # deltaR matching with AK8 fatjets
    eedr = pad_val(fatjets[:, :2].delta_r(Zee), 2, axis=1)
    mumudr = pad_val(fatjets[:, :2].delta_r(Zmumu), 2, axis=1)

    GenMatchingVars = {
        "ak8FatJetZeedR": eedr,
        "ak8FatJetZmumudR": mumudr,
    }

    return {
        **GenZVars,
        **GenElectronVars,
        **GenMuonVars,
        **GenMatchingVars,
    }

def gen_selection_Ztautau(
    events: NanoEventsArray,
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,
):
    """Gets Z and tautau 4-vectors + tau decay information"""
    print("-----------------------GEN FOR TAU TAU---------------------------")
    genparts = events.GenPart[events.GenPart.hasFlags(GEN_FLAGS)]


    Z = genparts[genparts.pdgId == PDGID.Z]
    # saving 4-vector info
    #GenZVars = {f"GenZ{key}": Z[var].to_numpy() for (var, key) in P4.items()}
    GenZVars = {f"GenZ{key}": ak.to_numpy(ak.pad_none(Z[var], 1, clip=True))  for (var, key) in P4.items()}
   
    Z_children = Z.children
    # pad_val is necessary to avoid a numpy MaskedArray even though all events have exactly 2 Higgs'
    GenZVars["GenZChildren"] = pad_val(Z_children.pdgId[:, :, 0], 2, axis=1)

    # finding bb and VV children
    is_tt = np.abs(Z_children.pdgId) == PDGID.tau
    print("a",np.any(is_tt))
    # checking that there are 2 bs and 2 taus

    has_tt = ak.sum(ak.flatten(is_tt, axis=2), axis=1) == 2
    if selection_args is not None:
        add_selection("has_tautau",  has_tt, *selection_args)

    taus = Z_children[is_tt]
    flat_taus = ak.flatten(taus, axis=2)
    GenTauVars = {f"GenTau{key}": pad_val(flat_taus[var], 2, axis=1) for (var, key) in P4.items()}

    tau_children = ak.flatten(taus.children, axis=2)
    tau_children = _iterate_children(tau_children, PDGID.tau)

    # check if tau children are leptons or hadrons
    # check neutral and charged pion IDs for hadronic taus
    tauh = _sum_taus(
        ak.any([ak.any(np.abs(tau_children.pdgId) == pid, axis=2) for pid in PDGID.pions], axis=0)
    )
    taumu = _sum_taus(ak.any(np.abs(tau_children.pdgId) == PDGID.mu, axis=2))
    taue = _sum_taus(ak.any(np.abs(tau_children.pdgId) == PDGID.e, axis=2))

    GenTauVars["GenTauhh"] = (tauh == 2).to_numpy()
    GenTauVars["GenTauhm"] = ((tauh == 1) & (taumu == 1)).to_numpy()
    GenTauVars["GenTauhe"] = ((tauh == 1) & (taue == 1)).to_numpy()
    #dR fatjet and gen tau
    Ztt = Z[ak.sum(is_tt, axis=2) == 2]
    Ztt = ak.pad_none(Ztt, 1, axis=1, clip=True)[:, 0]
    ttdr = pad_val(fatjets[:, :2].delta_r(Ztt),2, axis=1)
    GenMatchingVars = {
         "ak8FatJetZttdR": ttdr,
     }

    return {**GenZVars,  **GenTauVars , **GenMatchingVars}





def gen_selection_HHbbtautau(
    events: NanoEventsArray,
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,
):
    """Gets HH, bb, and tautau 4-vectors + tau decay information"""

    genparts = events.GenPart[events.GenPart.hasFlags(GEN_FLAGS)]

    # finding the two gen higgs
    higgs = genparts[genparts.pdgId == PDGID.H]
    print("H",higgs)
    # saving 4-vector info
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in P4.items()}

    # saving whether H->bb or H->tautau
    higgs_children = higgs.children
    # pad_val is necessary to avoid a numpy MaskedArray even though all events have exactly 2 Higgs'
    GenHiggsVars["GenHiggsChildren"] = pad_val(higgs_children.pdgId[:, :, 0], 2, axis=1)

    # finding bb and VV children
    is_bb = np.abs(higgs_children.pdgId) == PDGID.b
    is_tt = np.abs(higgs_children.pdgId) == PDGID.tau

    # checking that there are 2 bs and 2 taus
    has_bb = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 2
    has_tt = ak.sum(ak.flatten(is_tt, axis=2), axis=1) == 2
    if selection_args is not None:
        add_selection("has_bbtautau", has_bb * has_tt, *selection_args)

    bb = ak.flatten(higgs_children[is_bb], axis=2)
    GenbbVars = {f"Genbb{key}": pad_val(bb[var], 2, axis=1) for (var, key) in P4.items()}

    taus = higgs_children[is_tt]
    flat_taus = ak.flatten(taus, axis=2)
    GenTauVars = {f"GenTau{key}": pad_val(flat_taus[var], 2, axis=1) for (var, key) in P4.items()}

    tau_children = ak.flatten(taus.children, axis=2)
    tau_children = _iterate_children(tau_children, PDGID.tau)

    # check if tau children are leptons or hadrons
    # check neutral and charged pion IDs for hadronic taus
    tauh = _sum_taus(
        ak.any([ak.any(np.abs(tau_children.pdgId) == pid, axis=2) for pid in PDGID.pions], axis=0)
    )
    taumu = _sum_taus(ak.any(np.abs(tau_children.pdgId) == PDGID.mu, axis=2))
    taue = _sum_taus(ak.any(np.abs(tau_children.pdgId) == PDGID.e, axis=2))

    GenTauVars["GenTauhh"] = (tauh == 2).to_numpy()
    GenTauVars["GenTauhm"] = ((tauh == 1) & (taumu == 1)).to_numpy()
    GenTauVars["GenTauhe"] = ((tauh == 1) & (taue == 1)).to_numpy()

    #fatjet gen matching
    Hbb = higgs[ak.sum(is_bb, axis=2) == 2]
    Hbb = ak.pad_none(Hbb, 1, axis=1, clip=True)[:, 0]

    Htt = higgs[ak.sum(is_tt, axis=2) == 2]
    Htt = ak.pad_none(Htt, 1, axis=1, clip=True)[:, 0]

    #TODO: check more than just the leading two fatjets!
    bbdr = fatjets.delta_r(Hbb)
    ttdr = fatjets.delta_r(Htt)

    match_dR = 0.8
    Hbb_match = bbdr <= match_dR
    Htt_match = ttdr <= match_dR

    # overlap removal - in the case where fatjet is matched to both, match it only to the closest Higgs
    Hbb_match = (Hbb_match * ~Htt_match) + (bbdr <= ttdr) * (Hbb_match * Htt_match)
    Htt_match = (Htt_match * ~Hbb_match) + (bbdr > ttdr) * (Hbb_match * Htt_match)

    GenMatchingVars = {
        "ak8FatJetHbb": pad_val(Hbb_match, 3, axis=1),
        "ak8FatJetHtt": pad_val(Htt_match, 3, axis=1),
    }

    return {**GenHiggsVars, **GenbbVars, **GenTauVars, **GenMatchingVars}


def gen_selection_HH4b(
    events: NanoEventsArray,
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,  # noqa: ARG001
):
    """
    Save GenVars for HH(4b) events
    """
    genparts = events.GenPart[events.GenPart.hasFlags(GEN_FLAGS)]

    # finding the two gen higgs
    higgs = genparts[genparts.pdgId == PDGID.H]
    print("H", higgs)
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in P4.items()}
    higgs_children = higgs.children
    is_bb = np.abs(higgs_children.pdgId) == PDGID.b
    bs = ak.flatten(higgs_children[is_bb], axis=2)
    GenbVars = {f"Genb{key}": pad_val(bs[var], 4, axis=1) for (var, key) in P4.items()}

    return {**GenHiggsVars, **GenbVars}
