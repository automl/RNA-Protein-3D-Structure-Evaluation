#!/usr/bin/env python3
"""
OpenStructure's `compare-structures` without the SEQRES records of the reference mmCIF.

The command-line action builds its chain mapping on the SEQRES of the reference. For a few complexes (e.g. 7QDY with
OpenStructure 2.12) its heuristic chain mapping then fails on a chain without aligned residues (IndexError in
bb_lddt), although the structures themselves map and score without problems. Called with the same arguments as
`ost compare-structures`, this script runs the action's own loading and processing on the structures alone (as
OpenStructure's Python Scorer does) and writes the same JSON, so the callers parse it unchanged. The ost_* scripts
use it as a retry when the action fails. Run it with the python of the `ost` environment:

    python evaluation/ost_compare_noseqres.py -m model.cif -r reference.cif -o out.json --dockq --qs-score
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import traceback

import ost


def _action():
    """The compare-structures action of this OpenStructure installation, loaded as a module."""
    path = os.path.join(sys.prefix, "libexec", "openstructure", "ost-compare-structures")
    loader = importlib.machinery.SourceFileLoader("ost_compare_structures_action", path)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def main():
    act = _action()
    sys.argv = ["ost compare-structures"] + sys.argv[1:]
    args = act._ParseArgs()
    sink = act._LogSink()
    ost.PushLogSink(sink)
    ost.PushVerbosityLevel(args.verbosity)
    act._CheckCompoundLib()
    try:
        ref_format = act._GetStructureFormat(args.reference, sformat=args.reference_format)
        reference, _seqres, _mapping = act._LoadStructure(args.reference, sformat=ref_format,
                                                          bu_id=args.reference_biounit,
                                                          fault_tolerant=args.fault_tolerant)
        model_format = act._GetStructureFormat(args.model, sformat=args.model_format)
        model, _, _ = act._LoadStructure(args.model, sformat=model_format, bu_id=args.model_biounit,
                                         fault_tolerant=args.fault_tolerant)
        out = act._Process(model, reference, args, model_format, ref_format)
        out["arguments"] = args.__dict__
        out["log"] = sink.messages
        out["ost_version"] = ost.__version__
        out["status"] = "SUCCESS"
        out["seqres"] = "not used (ost_compare_noseqres.py)"
    except Exception:
        etype, evalue, _tb = sys.exc_info()
        out = {"status": "FAILURE", "traceback": traceback.format_exc(limit=1000),
               "exception": " ".join(traceback.format_exception_only(etype, evalue))}
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=4, sort_keys=False)
        raise
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()
