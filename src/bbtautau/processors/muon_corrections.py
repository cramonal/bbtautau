"""
Muon pT scale/resolution and ID/ISO scale-factor corrections for the
DY->mumu skimmer, ported from:
    https://github.com/DAZSLE/hbb-run3/blob/main/src/hbb/corrections.py
    (correct_muons(), add_muon_weights(), mupt_variations)
"""
from __future__ import annotations
import gzip
import json
import awkward as ak
import correctionlib
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.nanoaod import MuonArray
 
# vendor this file from https://gitlab.cern.ch/cms-muonPOG/muonscarekit
# (see module docstring above) and fix this import to match its location
from bbtautau.MuonScaRe import pt_resol, pt_resol_var, pt_scale, pt_scale_var


pog_correction_path = "/cvmfs/cms-griddata.cern.ch/cat/metadata/"

pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "muon_pt": ["MUO", "muon_scalesmearing.json.gz"],
}

years = {
    "2022": "Run3-22CDSep23-Summer22-NanoAODv12",
    "2022EE": "Run3-22EFGSep23-Summer22EE-NanoAODv12",
    "2023": "Run3-23CSep23-Summer23-NanoAODv12",
    "2023BPix": "Run3-23DSep23-Summer23BPix-NanoAODv12",
    "2024": "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15",
}

def _sanitize_inf(obj):
    """
    Recursively replaces literal "inf"/"-inf" *strings* with real (large,
    finite) numbers. Confirmed bug in the muon_Z.json.gz served from
    /cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/.../muon_Z.json.gz: the
    open-ended top pT-bin edge was serialized as the Python str(float("inf"))
    ("inf") instead of a JSON numeric value, which correctionlib's schema
    (edges must be numbers) rejects with "Invalid edges array type". Uses
    1e6, the standard correctionlib convention for an open upper bin edge.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_inf(v) for v in obj]
    if obj == "inf":
        return 1e6
    if obj == "-inf":
        return -1e6
    return obj
 
 
def load_corrections(path: str) -> correctionlib.CorrectionSet:
    """
    correctionlib.CorrectionSet.from_file(path), but loads+patches the raw
    JSON first (see _sanitize_inf) rather than handing the file straight to
    correctionlib. Safe no-op for files that don't have the bug. Use this
    instead of correctionlib.CorrectionSet.from_file() everywhere in this
    module, since we can't be sure which POG jsons this mirror affects.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return correctionlib.CorrectionSet.from_string(json.dumps(_sanitize_inf(data)))
 


def get_pog_json(obj: str, year: str) -> str:
    pog_json = pog_jsons[obj]
    year_tag = years[year]
    print("corlib", f"{pog_correction_path}/{pog_json[0]}/{year_tag}/latest/{pog_json[1]}")
    return f"{pog_correction_path}/{pog_json[0]}/{year_tag}/latest/{pog_json[1]}"


# name of the pT-scale/resolution variation -> the field name it lands in on
# the corrected muon collection (matches hbb-run3's mupt_variations)
mupt_variations = {
    "MuonPTScale": "ptscalecorr",
    "MuonPTRes": "ptcorr_resol",
}


def correct_muons(muons: MuonArray, events, year: str, isRealData: bool):
    """
    Applies the central MUON POG pT scale + resolution correction.

    Adds a "ptcorr" field (and the Up/Down variation fields used by
    mupt_variations) to the muon collection. Use `getattr(muons, "ptcorr")`
    (or the appropriate variation field name) instead of `muons.pt`
    everywhere downstream - object selection, dimuon mass/pT, etc.
    """
    #cset = correctionlib.CorrectionSet.from_file(get_pog_json("muon_pt", year))
    cset = load_corrections(get_pog_json("muon_pt", year))
 

    if isRealData:
        muons["ptcorr"] = pt_scale(
            1, muons.pt, muons.eta, muons.phi, muons.charge, cset, nested=True
        )
    else:
        muons["ptscalecorr"] = pt_scale(
            0, muons.pt, muons.eta, muons.phi, muons.charge, cset, nested=True
        )
        muons["ptcorr"] = pt_resol(
            muons.ptscalecorr,
            muons.eta,
            muons.phi,
            muons.nTrackerLayers,
            events.event,
            events.luminosityBlock,
            cset,
            nested=True,
        )

        muons["ptscalecorr_up"] = pt_scale_var( muons.ptcorr, muons.eta, muons.phi, muons.charge, "up", cset, nested=True)
        muons["ptscalecorr_down"] = pt_scale_var( muons.ptcorr, muons.eta, muons.phi, muons.charge, "dn", cset, nested=True)
        muons["ptcorr_resol_up"] = pt_resol_var(muons.ptscalecorr, muons.ptcorr, muons.eta, "up", cset, nested=True)
        muons["ptcorr_resol_down"] = pt_resol_var(muons.ptscalecorr, muons.ptcorr, muons.eta, "dn", cset, nested=True)

    return muons


def add_muon_weights(
    weights: Weights,
    year: str,
    muons: MuonArray,
    pt_type: str,
    muon_type: str,
    alt_str: str,
):
    """
    Muon ID/ISO scale factors, ported from hbb-run3's add_muon_weights().

    `muon_type` selects the working point the SFs correspond to - use
    "highpt" for the Tight-ID, pT >= 20 GeV DY->mumu selection (the closest
    of the two WPs hbb-run3 defines to a Tight-ID, isolated selection);
    switch to "loose" if you instead want SFs for a looser selection.
    `pt_type` should be "ptcorr" (the field correct_muons() adds) so the
    SFs are evaluated at the corrected momentum, matching what's used in
    the analysis selection itself.
    """
    if muon_type == "loose":
        id_key = "NUM_LooseID_DEN_TrackerMuons"
        iso_key = "NUM_LoosePFIso_DEN_LooseID"
    elif muon_type == "highpt":
        id_key = "NUM_HighPtID_DEN_TrackerMuons"
        iso_key = "NUM_LooseRelTkIso_DEN_HighPtID"
    else:
        raise ValueError(f"Unknown muon_type {muon_type!r}")

    #cset = correctionlib.CorrectionSet.from_file(get_pog_json("muon", year))
    cset = load_corrections(get_pog_json("muon", year)) 
    m, nm = ak.flatten(muons), ak.num(muons, axis=1)
    def get_sf(cset_key, syst):
        sf = cset[cset_key].evaluate(abs(m.eta), getattr(m, pt_type), syst)
        return ak.prod(ak.unflatten(sf, nm), axis=-1)
    id_nom = get_sf(id_key, "nominal")
    id_up = get_sf(id_key, "systup")
    id_down = get_sf(id_key, "systdown")

    iso_nom = get_sf(iso_key, "nominal")
    iso_up = get_sf(iso_key, "systup")
    iso_down = get_sf(iso_key, "systdown")

    weights.add(f"{alt_str}muon_ID", id_nom, id_up, id_down)
    weights.add(f"{alt_str}muon_ISO", iso_nom, iso_up, iso_down)

