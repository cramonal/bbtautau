"""
Skimmer for a DY->mumu control region, for the bbtautau analysis.

Selection:
    - 2 opposite-sign muons, pT >= 20 GeV, |eta| < 2.4, Tight ID,
      miniPFRelIso_all < 0.1
    - dR(mu, mu) < 0.8
    - m(mumu) in (70, 110) GeV
    - veto additional loose leptons
    - pT(mumu) > 200 GeV, |eta(mumu)| < 2.4
"""

from __future__ import annotations

import logging
import pathlib
import time
from collections import OrderedDict

import awkward as ak
import numpy as np
from boostedhh import hh_vars
from boostedhh.processors import SkimmerABC, utils
from boostedhh.processors.corrections import (
    JECs,
    add_pileup_weight,
    add_ps_weight,
    get_jetveto_event,
    get_pdf_weights,
    get_scale_weights,
)
from boostedhh.processors.utils import (
    P4,
    PAD_VAL,
    add_selection,
    pad_val,
)
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

from bbtautau.HLTs import HLTs
 
from bbtautau.processors.muon_corrections import add_muon_weights, correct_muons
from bbtautau.processors.vjets_corrections import add_VJets_corrections

from . import GenSelection, objects

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "DYto2L": GenSelection.gen_selection_DYleptonic,  # inclusive dilepton (2022/2023)
    "DYto2Mu": GenSelection.gen_selection_DYleptonic,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

package_path = str(pathlib.Path(__file__).parent.parent.resolve())


class dymumuSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing a
    boosted Z(->mumu) control-region selection.
    """

    # name in nano files: name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "Muon": {
            **P4,
            "charge": "charge",
            "tightId": "tightId",
            "miniPFRelIso_all": "miniPFRelIsoAll",
            "pt_raw": "PtRaw",
            },
        "Jet": {
            **P4,
            "rawFactor": "rawFactor",
            "btagPNetB": "btagPNetB",
        },
        "MET": {
            "pt": "Pt",
            "phi": "Phi",
        },
        "Event": {
            "run": "run",
            "event": "event",
            "luminosityBlock": "luminosityBlock",
        },
        "Pileup": {
            "nPU",
        },
        "TriggerObject": {
            "pt": "Pt",
            "eta": "Eta",
            "phi": "Phi",
            "filterBits": "Bit",
        },
    }

    # DY->mumu selection parameters
    muon_selection = {  # noqa: RUF012
        "pt": 20,
        "eta_max": 2.4,
        "miniiso_max": 0.1,
        "dr_min": 0.8,
        "mass_min": 70,
        "mass_max": 110,
    }


    def __init__(
        self,
        xsecs: dict = None,
        save_systematics: bool = False,
        region: str = "signal",
        nano_version: str = "v12_private",
    ):
        super().__init__()
        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        # HLT selection - TODO: point this at a dimuon (or single-muon) HLT
        # list once available; left as-is for now to match tautauSkimmer.py's
        # interface.
        self.HLTs = {"signal": HLTs.hlt_list(hlt_prefix=False)}
        self.HLTs = self.HLTs[region]

        self._systematics = save_systematics
        self._nano_version = nano_version
        self._region = region
        self._accumulator = processor.dict_accumulator({})

        logger.info(
            f"Running DY->mumu skimmer with:\nsystematics {self._systematics}\nregion {self._region}"
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for the DY->mumu control region"""
        start = time.time()
        logging.info(f"# events {len(events)}")

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])
        isData = not hasattr(events, "genWeight")

        gen_weights = events["genWeight"].to_numpy() if not isData else None
        n_events = len(events) if isData else np.sum(gen_weights)

        # selection and cutflow
        selection = PackedSelection()
        cutflow = OrderedDict()
        cutflow["all"] = n_events
        selection_args = (selection, cutflow, isData, gen_weights)

        JEC_loader = JECs(year)

        #########################
        # Object definitions
        #########################
        print("starting object selection", f"{time.time() - start:.2f}")

        # loose leptons, used only for the extra-lepton veto
        muons_corrected = correct_muons(events.Muon, events, year, isData) #scale corrections 
        veto_electrons, _ = objects.loose_electrons(events, events.Electron, year)
        veto_muons, _ = objects.loose_muons(events, muons_corrected, year)

        # tight, isolated signal muons for the DY->mumu selection
        # (mu_trigvars is per-candidate-muon, before pairing; add trigger
        # matching for the chosen pair here if/when you need it)
        sig_muons, trigsig_muons = objects.good_muons_dymumu(events, muons_corrected, year, _type="ptcorr")
        sig_muons["pt_raw"] = sig_muons.pt
        sig_muons["pt"] = sig_muons.ptcorr
        print("sig_muons multiplicity:", ak.to_list(ak.num(sig_muons, axis=1)[:20]))
        print("sig_muons charge:", ak.to_list(sig_muons.charge[:20]))
        # AK4 jets (kept for cross-checks / normalization against the signal region)
        jets, _ = JEC_loader.get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            jecs=utils.jecs,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )

        if JEC_loader.met_factory is not None:
            met = JEC_loader.met_factory.build(events.MET, jets, {}) if isData else events.MET
        else:
            met = events.MET

        jets = objects.good_ak4jets(jets, nano_version=self._nano_version)
        ht = ak.sum(jets.pt, axis=1)


        print("Objects", f"{time.time() - start:.2f}")

        #########################
        # Dimuon system
        #########################

        # all unique muon pairs among the tight, isolated muons
        mu_pairs = ak.combinations(sig_muons, 2, fields=["mu1", "mu2"])
        
        # opposite sign
        os_pairs = mu_pairs[mu_pairs.mu1.charge != mu_pairs.mu2.charge]
        print("mu_pairs:", ak.to_list(ak.num(mu_pairs, axis=1)[:20]))
        print("os_pairs:", ak.to_list(ak.num(os_pairs, axis=1)[:20]))              
        dr = os_pairs.mu1.delta_r(os_pairs.mu2)
        dimuon = os_pairs.mu1 + os_pairs.mu2
        print("mass:", ak.to_list(dimuon.mass[:20]))
        print("dr:", ak.to_list(dr[:20]))
        # dR and mass-window requirements applied at the pair level
        pair_sel = (
            (dr > self.muon_selection["dr_min"])
            & (dimuon.mass > self.muon_selection["mass_min"])
            & (dimuon.mass < self.muon_selection["mass_max"])
        )
        os_pairs = os_pairs[pair_sel]
        dimuon = dimuon[pair_sel]
        dr = dr[pair_sel]
        print("pairpair:", ak.to_list(pair_sel)[:20])
        print("pairpair dr:", ak.to_list(dr)[:20])
        has_pair = ak.num(dimuon, axis=1) > 0
        print("after pair selection:", ak.to_list(ak.num(dimuon, axis=1)[:20]))
        print("has_pair:", ak.to_list(has_pair[:20]))
        # if more than one valid pair survives, take the one closest to mZ
        best_idx = ak.argmin(np.abs(dimuon.mass - 91.1876), axis=1, keepdims=True)
        dimuon_best = ak.firsts(dimuon[best_idx])
        pair_best = ak.firsts(os_pairs[best_idx])
        dr_best = ak.fill_none(ak.firsts(dr[best_idx]), PAD_VAL)
        mu1 = pair_best.mu1
        mu2 = pair_best.mu2
        dimuon_scalar_pt = ak.fill_none(mu1.pt + mu2.pt, PAD_VAL)
        print("pairpair:", ak.to_list(mu1.pt)[:20])
        print("pairpair:", ak.to_list(mu2.pt)[:20])
        print("pairpa bestr:", ak.to_list(dimuon_best)[:20])
        print("scalar:", ak.to_list(dimuon_scalar_pt)[:20])
        reco_muon_pair = ak.concatenate([ak.singletons(mu1), ak.singletons(mu2)], axis=1)
        print("Dimuon", f"{time.time() - start:.2f}")
        
        #########################
        # Save / derive variables
        #########################

        # Gen variables
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, reco_muon_pair, selection_args)
                genVars = {**genVars, **vars_dict}

        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )
        logging.info(f"Passing gen selection: {np.sum(gen_selected)} / {len(events)}")

        # per-muon variables, saved as Muon1<var> / Muon2<var>
        muonVars = {}
        for var, key in self.skim_vars["Muon"].items():
            muonVars[f"Muon1{key}"] = ak.fill_none(mu1[var], PAD_VAL).to_numpy()
            muonVars[f"Muon2{key}"] = ak.fill_none(mu2[var], PAD_VAL).to_numpy()

        # dimuon system variables
        dimuonVars = {
            "DimuonMass": ak.fill_none(dimuon_best.mass, PAD_VAL).to_numpy(),
            "DimuonscalarPt": ak.fill_none(dimuon_scalar_pt, PAD_VAL).to_numpy(),
            "DimuonPt": ak.fill_none(dimuon_best.pt, PAD_VAL).to_numpy(),
            "DimuonEta": ak.fill_none(dimuon_best.eta, PAD_VAL).to_numpy(),
            "DimuonPhi": ak.fill_none(dimuon_best.phi, PAD_VAL).to_numpy(),
            "DimuonDR": dr_best.to_numpy() if hasattr(dr_best, "to_numpy") else dr_best,
        }

        # AK4 Jet variables
        num_ak4_jets = 4
        jet_skimvars = self.skim_vars["Jet"]
        if not isData:
            jet_skimvars = {**jet_skimvars, "pt_gen": "MatchedGenJetPt"}
        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_ak4_jets, axis=1)
            for (var, key) in jet_skimvars.items()
        }


        # MET
        metVars = {f"MET{key}": met[var].to_numpy() for (var, key) in self.skim_vars["MET"].items()}

        # Event variables
        eventVars = {
            key: events[val].to_numpy()
            for key, val in self.skim_vars["Event"].items()
            if key in events.fields
        }
        eventVars["ht"] = ht.to_numpy()
        eventVars["nElectrons"] = ak.num(events.Electron).to_numpy()
        eventVars["nMuons"] = ak.num(events.Muon).to_numpy()
        eventVars["nLooseMuons"] = ak.num(veto_muons).to_numpy()
        eventVars["nLooseElectrons"] = ak.num(veto_electrons).to_numpy()
        eventVars["nJets"] = ak.num(jets).to_numpy()
        eventVars["nGoodMuons"] = ak.num(sig_muons).to_numpy()
        # generator-level Z pT (LHE_Vpt), for stitching together the DY
        # samples - only meaningful for DY MC, not data or other samples.
        if not isData and "DYto2Mu" in dataset:
            eventVars["LHEVpt"] = events.LHE.Vpt.to_numpy()
        else:
            eventVars["LHEVpt"] = np.ones(len(events)) * PAD_VAL
 

        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}
        pileupVars = {**pileupVars, "nPV": events.PV["npvs"].to_numpy()}

        # Trigger variables
        HLTVars = {}
        zeros = np.zeros(len(events), dtype="int")
        for trigger in self.HLTs[year]:
            if trigger in events.HLT.fields:
                HLTVars[f"HLT_{trigger}"] = events.HLT[trigger].to_numpy().astype(int)
            else:
                logger.warning(f"Missing {trigger}!")
                HLTVars[f"HLT_{trigger}"] = zeros

        skimmed_events = {
            **genVars,
            **eventVars,
            **pileupVars,
            **HLTVars,
            **muonVars,
            **dimuonVars,
            **ak4JetVars,
            **metVars,
        }

        print("Vars", f"{time.time() - start:.2f}")

        ######################
        # Selection
        ######################

        HLT_triggered = np.any(
            np.array(
                [events.HLT[trigger] for trigger in self.HLTs[year] if trigger in events.HLT.fields]
            ),
            axis=0,
        )
        apply_trigger = False
        if apply_trigger:
            add_selection("trigger", HLT_triggered, *selection_args)

        # metfilters
        cut_metfilters = np.ones(len(events), dtype="bool")
        for mf in utils.met_filters:
            if mf in events.Flag.fields:
                cut_metfilters = cut_metfilters & events.Flag[mf]
        add_selection("met_filters", cut_metfilters, *selection_args)


        # exactly 2 tight, isolated, opposite-sign muons forming a valid pair
        add_selection("has_os_dimuon", has_pair, *selection_args)

        # dR(mu, mu) < 0.8 and m(mumu) in (70, 110) - already enforced at the
        # pair level above (pair_sel); has_os_dimuon captures whether any
        # pair survived those cuts.

        # veto additional loose leptons: exactly the 2 signal muons should be
        # loose (assuming Tight ID implies Loose ID in objects.py), and no
        # extra loose electrons/muons.
        add_selection(
            "electron_veto",
            (ak.num(veto_electrons, axis=1) == 0),
            *selection_args,
        )

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

        totals_dict = {"nevents": n_events}
        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weight_muons = ak.concatenate(
                [ak.singletons(mu1), ak.singletons(mu2)], axis=1
            )

            weights_dict, totals_temp = self.add_weights(
                events,
                year,
                dataset,
                gen_weights,
                gen_selected,
                weight_muons,
            )
            skimmed_events = {**skimmed_events, **weights_dict}
            totals_dict = {**totals_dict, **totals_temp}

        ##############################
        # Reshape and apply selections
        ##############################

        sel_all = selection.all(*selection.names)
        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        dataframe = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(dataframe, fname)

        logger.info(f"Cutflow:\n{cutflow}")
        print("Return ", f"{time.time() - start:.2f}")

        return {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(
        self,
        events,
        year,
        dataset,
        gen_weights,
        gen_selected,
        weight_muons,
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy(), dataset)
        add_ps_weight(weights, events.PSWeight)

        add_VJets_corrections(weights, dataset, events.GenPart) #ewk corrections for V+jets
        add_muon_weights(weights, year, weight_muons, "pt", "highpt", "")  #muon ID SF
        logger.debug("weights", extra=weights._weights.keys())

        weights_dict = {}
        totals_dict = {}

        weights_dict["weight"] = weights.weight()
        norm_preserving_weights = hh_vars.norm_preserving_weights
        weight_np = weights.partial_weight(include=norm_preserving_weights)
        totals_dict["np_nominal"] = np.sum(weight_np[gen_selected])

        if self._systematics:
            for systematic in list(weights.variations):
                weights_dict[f"weight_{systematic}"] = weights.weight(modifier=systematic)

            for key in weights._weights:
                weights_dict[f"single_weight_{key}"] = weights.partial_weight([key])

        ###################### Normalization (Step 1) ######################
        weight_norm = self.get_dataset_norm(year, dataset)
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        weights_dict["weight_noxsec"] = weights.weight()

        return weights_dict, totals_dict

