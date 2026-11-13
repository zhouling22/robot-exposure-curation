#!/usr/bin/env python3
"""Run upstream robomimic train.py without constructing unused CLIP language state.

Can-MH configs contain no language observations. The upstream script currently
constructs a CLIP LangEncoder unconditionally; this shim supplies the same
interface with a null encoder and then executes the unmodified train.py.
"""
import argparse
import runpy
import sys
import types


class NullLangEncoder:
    def __init__(self, device=None):
        self.device = device

    def get_lang_emb(self, lang):
        return None


module = types.ModuleType('robomimic.utils.lang_utils')
module.LangEncoder = NullLangEncoder
sys.modules['robomimic.utils.lang_utils'] = module
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--robomimic-train', required=True)
known, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
runpy.run_path(known.robomimic_train, run_name='__main__')
