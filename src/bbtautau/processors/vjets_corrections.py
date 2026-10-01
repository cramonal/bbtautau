from __future__ import annotations
 
import pathlib
 
import awkward as ak
import correctionlib
from coffea.analysis_tools import Weights
 
package_path = str(pathlib.Path(__file__).parent.parent.resolve())
 
# fix this to wherever you land the vendored vjets_corrections_*.json - see
# module docstring above
vjets_json_path = f"{package_path}/../../data/vjets/vjets_corrections_2flavorDY.json"
 
# dataset-name prefixes that should get the Z (DY) vs W k-factor - matches
# hbb-run3's lists verbatim. DYto2Mu-2Jets_MLL-50 is in here, so this
# applies directly to the dymumuSkimmer's DY samples.
Zjets_pref = [
    "DYto2L-2Jets_MLL-50",
    "DYto2E-2Jets_MLL-50",
    "DYto2Mu-2Jets_MLL-50",
    "DYto2Tau-2Jets_MLL-50",
    "Zto2Q-4Jets_Bin-HT",
    "Zto2Q-4Jets_HT",
]
Wjets_pref = [
    "WtoLNu-2Jets",
    "Wto2Q-3Jets_Bin-HT",
    "Wto2Q-3Jets_HT",
]
 
common_systs = [
    "d1K_NLO",
    "d2K_NLO",
    "d3K_NLO",
    "d1kappa_EW",
]
zsysts = common_systs + ["Z_d2kappa_EW", "Z_d3kappa_EW"]
wsysts = common_systs + ["W_d2kappa_EW", "W_d3kappa_EW"]
 
def add_VJets_corrections(weights: Weights, dataset: str, genpart):
    """
    Adds the NLO EWK+QCD k-factor weight ("vjets_nominal") plus its
    systematic variations, for DY/Z+jets and W+jets datasets identified by
    name via Zjets_pref/Wjets_pref. No-ops (returns without adding
    anything) for any other dataset.
    """
    isZ_dataset = any(ds in dataset for ds in Zjets_pref)
    isW_dataset = any(ds in dataset for ds in Wjets_pref)
 
    if not (isZ_dataset or isW_dataset):
        return
 
    boson = ak.firsts(
        genpart[
            ((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24))
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
    )
    vpt = ak.fill_none(boson.pt, 0.0)
 
    def add_systs(systlist, qcdcorr, ewkcorr):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = ak.ones_like(vpt)
        for syst in systlist:
            weights.add(
                syst,
                ones,
                ewkcorr.evaluate(syst + "_up", vpt) / ewknom,
                ewkcorr.evaluate(syst + "_down", vpt) / ewknom,
            )
 
    vjets_kfactors = correctionlib.CorrectionSet.from_file(vjets_json_path)
 
    if isZ_dataset:
        qcdcorr = vjets_kfactors["Z_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr)
    elif isW_dataset:
        qcdcorr = vjets_kfactors["W_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr)

