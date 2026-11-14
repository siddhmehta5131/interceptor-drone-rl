"""Audit every decided value in the code base (plan P2 / professor request).

Writes docs/full_value_audit.csv with one row per:
  * sourced config leaf ({value, unit, source} YAML),
  * plain config leaf (other YAML files),
  * Python named constant, dict entry, function default, numeric keyword
    argument and remaining inline numeric literal (interceptor/, training/,
    scripts/, run.py, notebooks),
  * modelling decision (from build_decision_register.DECISIONS).

Status:
  CITED             an external reference is named next to the value
  CITED_PARTIAL     an external reference is named but the value is also
                    described as an estimate / fit / project choice
  DERIVED_INTERNAL  points to an internal derivation (EQ-id, plan decision,
                    config key); no external reference
  CITED_SECONDARY   backed by a web source listed in data/*.csv (encyclopedic,
                    press or aggregator; not a primary datasheet)
  DATA_DISAGREES    the collected data contradicts the value; value unchanged
  MISSING           nothing is recorded

Run: python scripts/build_full_value_audit.py
"""
from __future__ import annotations

import ast
import csv
import io
import json
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_decision_register import DECISIONS, walk  # noqa: E402
from reference_library import ANSWER, LIB, apply_rules, DECISION_RULES  # noqa: E402

OUT = ROOT / "docs" / "full_value_audit.csv"
PY_DIRS = ["interceptor", "training", "scripts"]
PY_FILES = ["run.py", "conftest.py"]
PLAIN_YAML = ["training.yaml",
              "visualiser.yaml", "ablations.yaml"]

# token -> full reference.
REFS = [
    (r"F(o|ö)rster", "Förster, J. (2015). System Identification of the Crazyflie 2.0 Nano Quadrocopter. BSc thesis, ETH Zürich."),
    (r"Faessler", "Faessler, M., Franchi, A., Scaramuzza, D. (2018). Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag. IEEE RA-L 3(2), arXiv:1712.02402 (Sec. VII-C: d_x 0.49-0.54, d_y 0.24-0.39 1/s on a 610 g quadrotor; d_z not identified)."),
    (r"NanoBench", "Ullah, S. I., Baca, J. (2026). NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification, Control, and State Estimation. arXiv:2603.09908 (flying mass 40.85 g confirmed; other coefficients not found in the paper, BUGS #27)."),
    (r"Bitcraze|Crazyflie firmware|platform_defaults", "Bitcraze AB. Crazyflie 2.x datasheet and crazyflie-firmware (platform_defaults_cf2.h), github.com/bitcraze/crazyflie-firmware."),
    (r"cf2x\.urdf|gym-pybullet|Panerati", "Panerati, J. et al. (2021). Learning to Fly - a Gym Environment with PyBullet Physics for RL of Multi-agent Quadcopter Control. IROS. (cf2x.urdf)"),
    (r"Beard|McLain|mavsim|Aerosonde", "Beard, R. W., McLain, T. W. (2012). Small Unmanned Aircraft: Theory and Practice. Princeton Univ. Press; mavsim_public aerosonde_parameters.py."),
    (r"Zipfel|CADAC|AIM5", "Zipfel, P. H. (2014). Modeling and Simulation of Aerospace Vehicle Dynamics, 3rd ed. AIAA; CADAC++ AIM5 example."),
    (r"MIL-HDBK|MIL-F|Dryden", "MIL-HDBK-1797 (1997) / MIL-F-8785C (1980). Flying Qualities of Piloted Aircraft (Dryden turbulence model). US DoD."),
    (r"14 CFR 23\.337", "14 CFR 23.337 Limit maneuvering load factors (2011 ed.), https://www.govinfo.gov/content/pkg/CFR-2011-title14-vol1/pdf/CFR-2011-title14-vol1-sec23-337.pdf"),
    (r"stabilizer_types\.h", "Bitcraze crazyflie-firmware stabilizer_types.h (ATTITUDE_RATE 500 Hz, RATE_MAIN_LOOP 1000 Hz), https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/interface/stabilizer_types.h"),
    (r"Standard gravity", "Standard gravity g0 = 9.80665 m/s^2 (defined value), rounded to 9.81 as in mavsim_public aerosonde_parameters.py"),
    (r"US Air Force fact sheet", "US Air Force (Air Combat Command). F-16 Fighting Falcon fact sheet: 1,500 mph (Mach 2 at altitude); up to nine G's with full internal fuel; wingspan 32 ft 8 in (9.8 m). https://www.af.mil/About-Us/Fact-Sheets/Display/Article/104505/f-16-fighting-falcon/"),
    (r"IRGC confirms specs|IRGC exhibition poster", "Binnie, J. (2023, 17 May). IRGC confirms specs for Shahed-136 attack UAV. Janes. (IRGC poster: 185 km/h, 200 kg, 50 kg warhead, 2,500 km.) https://www.janes.com/osint-insights/defence-news/air/irgc-confirms-specs-for-shahed-136-attack-uav"),
    (r"Echodyne", "Echodyne Corp. company statement hosted by the US International Trade Administration (2024): EchoShield <0.5 deg az x el; EchoGuard ~1 deg az, ~1.5 deg el, ~1-2 m range accuracy. https://www.trade.gov/sites/default/files/2024-08/ECHODYNE.pdf"),
    (r"Schulman", "Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347 (App. A Table 3 p. 10, Table 1 p. 6, Sec. 6.1 read 2026-11-15)."),
    (r"Raffin|Stable-Baselines|SB3", "Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR 22(268); installed stable_baselines3 2.9.0, constructor defaults in ppo/ppo.py, sac/sac.py, td3/td3.py (read 2026-11-15)."),
    (r"Haarnoja(?! 2018b)|1801\.01290","Haarnoja, T. et al. (2018a). Soft Actor-Critic. ICML, arXiv:1801.01290 (App. D Table 1 and Sec. 4.2 read 2026-11-15)."),
    (r"Haarnoja 2018b|1812\.05905", "Haarnoja, T. et al. (2018b). Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905 (Sec. 5 automatic temperature, App. D Table 1 entropy target -dim(A), read 2026-11-15)."),
    (r"Fujimoto", "Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML, arXiv:1802.09477 (Sec. 6.1 and App. C Table 3 read 2026-11-15)."),
    (r"sfujim", "Fujimoto, S. TD3 reference implementation, github.com/sfujim/TD3 commit 34770cc (2023-07-14): TD3.py, main.py, utils.py (read 2026-11-15)."),
    (r"Andrychowicz", "Andrychowicz, M. et al. (2021). What Matters In On-Policy Reinforcement Learning? ICLR, arXiv:2006.05990 (Sec. 3.2 pp. 5-6 networks, Sec. 3.3 normalisation; read 2026-11-15)."),
    (r"Chadwick", "Chadwick, W. R., Zarchan, P. (1995). Interception of spiraling ballistic missiles. Proc. American Control Conference, vol. 6, p. 4476. doi:10.1109/ACC.1995.532784 (abstract only: weave frequencies of 0.5 to 1 Hz are most critical for fast-response PN interceptors)."),
    (r"(?<!& )Zarchan", "Zarchan, P. Tactical and Strategic Missile Guidance. AIAA Progress in Astronautics and Aeronautics."),
    (r"Wilson", "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA 22(158)."),
    (r"McNemar", "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. Psychometrika 12."),
    (r"\bNg\b.*199|Harada", "Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations. ICML."),
    (r"Kalman 1960|Bar-Shalom", "Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. J. Basic Eng. 82; Bar-Shalom, Li, Kirubarajan (2001). Estimation with Applications to Tracking and Navigation. Wiley."),
]
REFS = [(re.compile(p, re.I), full) for p, full in REFS]
SOFT = re.compile(r"estimate|analytic|baseline|notebook v2|hand|tuned|fit\b|fitted|anchored|"
                  r"replaced|rounded|approx|assum|placeholder|legacy|unsourced|TODO", re.I)
INTERNAL = re.compile(r"EQ-[A-Z]+|decision [SQ]\d|\b[SQF]\d{1,2}\b|\bP\d+\.\d+|BUGS|configs/|"
                      r"mathematics\.md|implementation_plan|derived|measured|sweep|plan ", re.I)


def _data_rows():
    out = {}
    for name in ("threat_platforms", "interceptor_platforms", "interceptor_sensors", "lethality_and_gload"):
        f = ROOT / "data" / f"{name}.csv"
        if f.exists():
            for r in csv.DictReader(f.open(encoding="utf8")):
                label = r.get("name") or r.get("interceptor") or r.get("item")
                out[r["id"]] = f"{label}: {r['source_url']} [{r['source_tier']}]"
    return out


DATA_ROWS = _data_rows()
DATA_TAG = re.compile(r"secondary source, data/\w+\.csv ((?:[TISLG]\d\d ?)+)")


def classify(src: str):
    if src.startswith("OWN_FIT"):
        # fitted by us to a cited data set: the data set is cited, the number is not
        return "OWN_FIT", " | ".join(full for rx, full in REFS if rx.search(src))
    if src.startswith("PROJECT_CHOICE"):
        # an honest project choice; any named reference is a cross-check, not the value's source
        return "PROJECT_CHOICE", " | ".join(full for rx, full in REFS if rx.search(src))
    ids = [i for grp in DATA_TAG.findall(src) for i in grp.split()]
    if ids:
        ref = " | ".join(DATA_ROWS.get(i, i) for i in dict.fromkeys(ids))
        return ("DATA_DISAGREES" if "DATA DISAGREES" in src else "DATA_GAP" if "DATA GAP" in src else "CITED_SECONDARY"), ref
    refs = [full for rx, full in REFS if rx.search(src)]
    if refs:
        hard = re.search(r"14 CFR 23\.337|stabilizer_types\.h|Standard gravity", src)
        return ("CITED_PARTIAL" if SOFT.search(src) and not hard else "CITED"), " | ".join(refs)
    if src and INTERNAL.search(src):
        return "DERIVED_INTERNAL", ""
    return "MISSING", ""


FILE_DOC_REFS = {}

# --------------------------------------------------------------- python --
TRIVIAL = {0, 1, 2, -1}


def is_num(n):
    return isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex)) \
        and not isinstance(n.value, bool)


def has_num(node):
    return any(is_num(n) for n in ast.walk(node))


def comments_of(text: str):
    out = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string.lstrip("# ").strip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


class Extractor(ast.NodeVisitor):
    def __init__(self, text, rel):
        self.lines = text.splitlines()
        self.rel = rel
        self.cm = comments_of(text)
        self.scope = []
        self.rows = []
        self.used = set()

    def src(self, lo, hi):
        parts = [self.cm[i] for i in range(lo, hi + 1) if i in self.cm]
        if not parts:  # comment block directly above
            i = lo - 1
            block = []
            while i >= 1 and i in self.cm and self.lines[i - 1].lstrip().startswith("#") and len(block) < 3:
                block.insert(0, self.cm[i])
                i -= 1
            parts = block
        return " ".join(parts)[:300]

    def add(self, kind, node, name, value):
        hi = getattr(node, "end_lineno", node.lineno)
        self.rows.append((kind, f"{self.rel}:{node.lineno}", name[:120], str(value)[:240],
                          ".".join(self.scope), self.src(node.lineno, hi)))
        for n in ast.walk(node):
            self.used.add(id(n))

    def scoped(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_ClassDef = scoped

    def visit_FunctionDef(self, node):
        a = node.args
        pos = a.posonlyargs + a.args
        for arg, d in list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
                [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]:
            if isinstance(d, ast.Constant) and d.value is None:
                continue
            if has_num(d) or isinstance(d, ast.Constant):
                self.add("code_default", d, f"{node.name}({arg.arg}=)", ast.unparse(d))
        self.scoped(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _assign(self, node, targets, value):
        name = ", ".join(ast.unparse(t) for t in targets)
        if name == "__all__" or value is None:
            return
        top = len(self.scope) == 0 or not any(c[:1].islower() for c in self.scope[-1:])
        if isinstance(value, ast.Dict) and has_num(value) and all(
                isinstance(k, ast.Constant) for k in value.keys if k is not None):
            for k, v in zip(value.keys, value.values):
                if k is None:
                    continue
                if isinstance(v, ast.Dict) and all(isinstance(kk, ast.Constant) for kk in v.keys if kk):
                    for kk, vv in zip(v.keys, v.values):
                        if kk is not None and (has_num(vv) or isinstance(vv, ast.Constant)):
                            self.add("code_constant", vv, f"{name}[{k.value!r}][{kk.value!r}]", ast.unparse(vv))
                elif has_num(v) or isinstance(v, ast.Constant):
                    self.add("code_constant", v, f"{name}[{k.value!r}]", ast.unparse(v))
            return
        if has_num(value):
            nums = [n.value for n in ast.walk(value) if is_num(n)]
            in_func = any(c[:1].islower() or c.startswith("_") for c in self.scope[-1:])
            if all(v in TRIVIAL for v in nums) and in_func and not name.isupper():
                return  # counters / zero initialisers inside functions
            text = ast.unparse(value)
            if not isinstance(value, (ast.Constant, ast.UnaryOp, ast.Tuple, ast.List)):
                text = text[:150] + "  {literals: " + ", ".join(map(str, nums)) + "}"
            self.add("code_constant", node, name, text)
        elif isinstance(value, ast.Constant) and isinstance(value.value, (str, bool)) and top \
                and not name.startswith("__") and "\n" not in str(value.value):
            self.add("code_choice", node, name, repr(value.value))

    def visit_Assign(self, node):
        self._assign(node, node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._assign(node, [node.target], node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._assign(node, [node.target], node.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        for kw in node.keywords:
            if kw.arg and id(kw.value) not in self.used and has_num(kw.value) \
                    and len(ast.unparse(kw.value)) < 80:
                fn = ast.unparse(node.func)[-60:]
                self.add("code_kwarg", kw.value, f"{fn}({kw.arg}=)", ast.unparse(kw.value))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        for n in ast.walk(node.slice):
            self.used.add(id(n))
        self.generic_visit(node)

    def finish(self, tree):
        by_line = {}
        for n in ast.walk(tree):
            if is_num(n) and id(n) not in self.used and n.value not in TRIVIAL:
                by_line.setdefault(n.lineno, []).append(n.value)
        for ln, vals in sorted(by_line.items()):
            code = self.lines[ln - 1].split("#")[0].strip()[:120] if ln <= len(self.lines) else ""
            self.rows.append(("code_inline", f"{self.rel}:{ln}", code, ", ".join(map(str, vals)),
                              "", self.src(ln, ln)))
        return self.rows


def extract_py(text, rel):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    ex = Extractor(text, rel)
    ex.visit(tree)
    doc = " ".join(ast.get_docstring(n) or "" for n in ast.walk(tree)
                   if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)))
    FILE_DOC_REFS[rel] = classify(doc)[1]
    return ex.finish(tree)


# ----------------------------------------------------------------- yaml --
LEAF = re.compile(r"^(\s*)(- )?([\w\-\.]+):\s*(.*?)\s*$")


def split_comment(line):
    m = re.search(r"\s#\s?(.*)$", line)
    if m and line[:m.start()].count('"') % 2 == 0:
        return line[:m.start()], m.group(1).strip()
    return line, ""


def extract_plain_yaml(path, rel):
    rows, stack, block = [], [], []
    header = []
    lines = path.read_text(encoding="utf8").splitlines()
    for ln in lines:
        if ln.startswith("#"):
            header.append(ln.lstrip("# "))
        elif ln.strip():
            break
    header = " ".join(header)
    for i, raw in enumerate(lines, 1):
        if raw.lstrip().startswith("#"):
            block.append(raw.lstrip("# ").strip())
            continue
        body, com = split_comment(raw)
        m = LEAF.match(body)
        if not m:
            if raw.strip() == "":
                block = []
            continue
        indent = len(m.group(1)) + (2 if m.group(2) else 0)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key, val = m.group(3), m.group(4)
        if val in ("", "|", ">"):
            stack.append((indent, key))
        else:
            p = ".".join([k for _, k in stack] + [key])
            src = com or " ".join(block)[-300:]
            if not src and rel.endswith("training.yaml") and p.startswith(("algorithms.", "network.")):
                src = "file header: " + header[:260]
            rows.append(("config_plain", f"{rel}:{i}", p, val, "", src))
        block = []
    return rows


def main() -> None:
    rows = []
    # sourced config leaves
    for f in sorted((ROOT / "configs").rglob("*.yaml")):
        leaves = []
        walk(yaml.safe_load(f.read_text(encoding="utf8")), [], leaves, f.relative_to(ROOT).as_posix())
        for fname, p, v, u, s in leaves:
            rows.append(("config_sourced", fname, p, f"{v} {u}".strip(), "", s))
    for name in PLAIN_YAML:
        f = ROOT / "configs" / name
        if f.exists():
            rows += extract_plain_yaml(f, f"configs/{name}")
    # python
    files = [ROOT / p for p in PY_FILES if (ROOT / p).exists()]
    for d in PY_DIRS:
        files += sorted(p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts
                        and p.name not in ("reference_library.py", "build_full_value_audit.py"))
    for f in files:
        rows += extract_py(f.read_text(encoding="utf8"), f.relative_to(ROOT).as_posix())
    # notebooks
    for f in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(f.read_text(encoding="utf8"))
        for ci, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            code = "".join(c["source"])
            code = "\n".join("pass" if l.lstrip().startswith(("!", "%")) else l for l in code.splitlines())
            for r in extract_py(code, f"notebooks/{f.name}#cell{ci}"):
                rows.append(("notebook_" + r[0][5:],) + r[1:])

    with OUT.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "kind", "location", "name", "value", "scope", "recorded_source",
                    "status", "reference", "reference_url", "url_checked", "suggested_reference", "panel_risk", "panel_answer"])
        stat = Counter()
        n = 0
        # config sources that name a code constant (e.g. "constants.py PH_M")
        link = {}
        for kind, loc, name, value, scope, src in rows:
            if kind == "config_sourced":
                for ident in re.findall(r"[A-Z][A-Z0-9]*_[A-Z0-9_]+", src):
                    link.setdefault(ident, f"{src} (via {loc}::{name})")
        cfg_val = {}
        for kind, loc, name, value, scope, src in rows:
            if kind == "config_sourced":
                try:
                    cfg_val.setdefault(float(value.split()[0]), []).append(f"{loc}::{name} (source: {src})")
                except (ValueError, IndexError):
                    pass
        for kind, loc, name, value, scope, src in rows:
            n += 1
            status, ref = classify(src)
            hint = ""
            fdoc = FILE_DOC_REFS.get(loc.rsplit(":", 1)[0], "")
            if fdoc and not status.startswith("CITED"):
                hint = "file docstrings cite (not tied to this value): " + fdoc
            if not kind.startswith("config"):
                try:
                    fv = float(value)
                    if len(value.strip("-")) >= 4 and len(cfg_val.get(fv, [])) == 1:
                        hint = "same value as " + cfg_val[fv][0] + ("; " + hint if hint else "")
                except ValueError:
                    pass
            if kind.startswith("code") and status in ("MISSING", "DERIVED_INTERNAL") and name in link:
                s2, r2 = classify(link[name])
                if s2.startswith("CITED"):
                    src, status, ref = (src + " || " if src else "") + link[name], s2, r2
            rurl, rchk = "", ""
            if (status == "MISSING" and kind == "config_sourced") or \
                    (status in ("MISSING", "DERIVED_INTERNAL") and kind != "config_sourced"):
                st2, keys, note = apply_rules(kind, loc, name, status, value)
                if st2 != status or keys:
                    status = st2
                    if keys:
                        ref = " | ".join(LIB[k][0] for k in keys)
                        rurl = " | ".join(LIB[k][1] for k in keys if LIB[k][1])
                        rchk = " | ".join(f"{k}: {LIB[k][2]}" for k in keys)
                    if note:
                        hint = note + ("; " + hint if hint else "")
            stat[(kind.split("_")[0], status)] += 1
            w.writerow([f"V{n:05d}", kind, loc, name, value, scope, src or "MISSING", status, ref or
                        ("MISSING" if (not ref and status == "MISSING") else ""), rurl, rchk, hint, *ANSWER[status]])
        smap = {"cited": "CITED", "partly_cited": "CITED_PARTIAL",
                "derived_or_internal": "DERIVED_INTERNAL", "NEEDS_CITATION": "MISSING"}
        for j, d in enumerate(DECISIONS, 1):
            status = smap[d[4]]
            _, ref = classify(d[3] + " " + (d[5] if status.startswith("CITED") else ""))
            rurl, rchk, note = "", "", ""
            if status == "MISSING" and f"D{j:03d}" in DECISION_RULES:
                status, keys, note = DECISION_RULES[f"D{j:03d}"]
                if keys:
                    ref = " | ".join(LIB[k][0] for k in keys)
                    rurl = " | ".join(LIB[k][1] for k in keys if LIB[k][1])
                    rchk = " | ".join(f"{k}: {LIB[k][2]}" for k in keys)
                else:
                    ref = note
            stat[("decision", status)] += 1
            w.writerow([f"D{j:03d}", "decision", d[0], d[1], d[2], "", d[3], status,
                        ref or "MISSING", rurl, rchk, f"{note} {d[5]} [{d[6]}] {d[7]}".strip(), *ANSWER[status]])
    print(f"wrote {OUT}: {n} values + {len(DECISIONS)} decisions")
    lines = ["# References", "",
             "Generated by `scripts/build_full_value_audit.py`. `checked` says how the link was verified on 2026-11-14.", "",
             "## Papers, standards, datasheets and source files", "",
             "| Key | Reference | Link | Checked |", "|---|---|---|---|"]
    for k in sorted(LIB):
        c, u, chk = LIB[k]
        lines.append(f"| {k} | {c} | {u or 'no URL (book)'} | {chk} |")
    lines += ["", "## Vehicle, sensor and lethality data (web sources; see `data/*.csv` for values)", "",
              "| Id | Item and link [source tier] |", "|---|---|"]
    for k in sorted(DATA_ROWS):
        lines.append(f"| {k} | {DATA_ROWS[k]} |")
    (ROOT / "docs" / "references.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("wrote docs/references.md")
    for k in sorted(stat):
        print(f"  {k[0]:9s} {k[1]:17s} {stat[k]}")


if __name__ == "__main__":
    main()
