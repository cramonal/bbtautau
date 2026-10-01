"""
Collection of utilities for corrections and systematics in processors.

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import contextlib

import awkward as ak
import dask_awkward as dak
import numpy as np
import correctionlib
import correctionlib.schemav2
import pickle
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.nanoaod import JetArray, MuonArray
from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
from coffea.lookup_tools import extractor

from hbb.MuonScaRe import pt_resol, pt_scale, pt_resol_var, pt_scale_var 
from hbb.jerc_eras import jec_eras,jer_eras, jec_mc, jer_mc, jec_data, fatjet_jerc_keys, jet_jerc_keys
from hbb.taggers import b_taggers
from hbb.EWHiggs_corrections import theory_xs, xs_ewkcorr, ewh_ptbin

ak.behavior.update(vector.behavior)
package_path = str(pathlib.Path(__file__).parent.parent.resolve())

# Important Run3 start of Run
FirstRun_2022C = 355794
FirstRun_2022D = 357487
LastRun_2022D = 359021
FirstRun_2022E = 359022
LastRun_2022F = 362180

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms-griddata.cern.ch/cat/metadata/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "muon_pt" : ["MUO", "muon_scalesmearing.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "photon": ["EGM", "photon.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "pileup2024": ["LUM", "puWeights_CDEFGHI.json.gz"], # file excludes 2024B 
    "fatjet_jec": ["JME", "fatJet_jerc.json.gz"],
    "jet_jec": ["JME", "jet_jerc.json.gz"],
    "jetveto": ["JME", "jetvetomaps.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
    "jetid" : ["JME", "jetid.json.gz"],
}

years = {
    "2022": "Run3-22CDSep23-Summer22-NanoAODv12",
    "2022EE": "Run3-22EFGSep23-Summer22EE-NanoAODv12",
    "2023": "Run3-23CSep23-Summer23-NanoAODv12",
    "2023BPix": "Run3-23DSep23-Summer23BPix-NanoAODv12",
    "2024": "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15",
}


def add_VJets_corrections(weights: Weights, dataset: str, genpart):

    boson = ak.firsts(genpart[
            ((genpart.pdgId == 23)|(abs(genpart.pdgId) == 24))
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ])
    vpt = ak.fill_none(boson.pt, 0.)

    Zjets_pref = ["DYto2L-2Jets_MLL-50", "DYto2E-2Jets_MLL-50", "DYto2Mu-2Jets_MLL-50", "DYto2Tau-2Jets_MLL-50", "Zto2Q-4Jets_Bin-HT", "Zto2Q-4Jets_HT"]
    Wjets_pref = ["WtoLNu-2Jets", "Wto2Q-3Jets_Bin-HT", "Wto2Q-3Jets_HT"]
    isZ_dataset = any(ds in dataset for ds in Zjets_pref)
    isW_dataset = any(ds in dataset for ds in Wjets_pref)

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = ak.ones_like(vpt)
        for syst in systlist:
            weights.add(syst, ones, ewkcorr.evaluate(syst + "_up", vpt) / ewknom, ewkcorr.evaluate(syst + "_down", vpt) / ewknom)

    file = f"{package_path}/hbb/data/vjets/vjets_corrections_2flavorDY.json"
    vjets_kfactors = correctionlib.CorrectionSet.from_file(file)

    if isZ_dataset:
        qcdcorr = vjets_kfactors["Z_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr)

    elif isW_dataset:
        qcdcorr = vjets_kfactors["W_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr)

    return
