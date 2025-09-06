# app.py
# ChemWeaver backend: resolve query→SMILES, build 3D SDF, build Gaussian .gjf
#
# Requirements:
#   pip install flask flask-cors rdkit-pypi requests
# Run:
#   python app.py    (http://127.0.0.1:5000)

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
import re
import requests

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# --------- PubChem helpers ----------
def _pc_json(url, timeout=10):
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if "Fault" in js:
        det = js["Fault"].get("Details") or []
        msg = det[0] if det else js["Fault"].get("Message", "PubChem fault")
        raise ValueError(str(msg))
    return js

def _props_smiles_from_properties(js):
    props = (js.get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return None
    p = props[0]
    return p.get("IsomericSMILES") or p.get("CanonicalSMILES")

# --------- Core chemistry helpers ----------
def gaussian_bond_order(bond) -> float:
    bt = bond.GetBondType()
    if bt == BondType.SINGLE:   return 1.0
    if bt == BondType.DOUBLE:   return 2.0
    if bt == BondType.TRIPLE:   return 3.0
    if bt == BondType.AROMATIC or bond.GetIsAromatic():
        return 1.5
    return 1.0

def smiles_to_rdkit_with_H(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    mol = Chem.AddHs(mol)
    return mol

def embed_3d_inplace(mol):
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def mol_to_sdf_text(mol) -> str:
    from io import StringIO
    sio = StringIO()
    w = Chem.SDWriter(sio)
    w.write(mol)
    w.flush()
    return sio.getvalue()

def build_connectivity_lines(mol) -> str:
    lines = []
    nat = mol.GetNumAtoms()
    nbrs = {i: [] for i in range(nat)}
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        order = gaussian_bond_order(b)
        nbrs[i].append((j, order))
        nbrs[j].append((i, order))
    for i in range(nat):
        if not nbrs[i]:
            lines.append(f" {i+1}")
            continue
        parts = [f" {i+1}"]
        for j, o in sorted(nbrs[i], key=lambda t: t[0]):
            parts.append(f"{j+1} {o:.1f}")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"

def format_gaussian(mol, title: str, header: str, include_conn: bool = True) -> str:
    conf = mol.GetConformer()
    coords = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        pos = conf.GetAtomPosition(idx)
        coords.append((a.GetSymbol(), pos.x, pos.y, pos.z))

    out = []
    if header:
        out.append(header.rstrip() + "\n\n")
    out.append(f"{title}\n\n")
    out.append("0 1\n")
    for (sym, x, y, z) in coords:
        out.append(f" {sym:<2} {x:>10.6f} {y:>10.6f} {z:>10.6f}\n")
    if include_conn:
        out.append("\n")
        out.append(build_connectivity_lines(mol))
    return "".join(out)

# --------- Resolver: query (SMILES or Name/CAS/InChIKey) → SMILES ----------
def resolve_query_to_smiles(q: str) -> str:
    q = (q or "").strip()
    if not q:
        raise ValueError("missing query")

    # 0) If it's valid SMILES already, canonicalize and return
    try:
        mol = Chem.MolFromSmiles(q)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        pass  # not SMILES, fall through

    enc = requests.utils.quote(q, safe="")
    is_inchikey = re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", q, flags=re.I) is not None
    is_cas      = re.match(r"^\d{2,7}-\d{2}-\d$", q) is not None

    # 1) InChIKey
    if is_inchikey:
        js = _pc_json(f"{PUBCHEM}/compound/inchikey/{enc}/property/IsomericSMILES,CanonicalSMILES/JSON")
        smi = _props_smiles_from_properties(js)
        if smi: return smi
        raise ValueError("no SMILES for that InChIKey")

    # 2) CAS RN
    if is_cas:
        try:
            js = _pc_json(f"{PUBCHEM}/compound/xref/rn/{enc}/property/IsomericSMILES,CanonicalSMILES/JSON")
            smi = _props_smiles_from_properties(js)
            if smi: return smi
        except Exception:
            pass
        cj = _pc_json(f"{PUBCHEM}/compound/xref/rn/{enc}/cids/JSON")
        cids = ((cj.get("IdentifierList") or {}).get("CID") or [])
        if cids:
            cid = cids[0]
            pj = _pc_json(f"{PUBCHEM}/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON")
            smi = _props_smiles_from_properties(pj)
            if smi: return smi
        raise ValueError("no SMILES for that CAS")

    # 3) Name → property, else Name → CID → property
    try:
        js = _pc_json(f"{PUBCHEM}/compound/name/{enc}/property/IsomericSMILES,CanonicalSMILES/JSON")
        smi = _props_smiles_from_properties(js)
        if smi: return smi
    except Exception:
        pass

    cj = _pc_json(f"{PUBCHEM}/compound/name/{enc}/cids/JSON")
    cids = ((cj.get("IdentifierList") or {}).get("CID") or [])
    if cids:
        cid = cids[0]
        pj = _pc_json(f"{PUBCHEM}/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON")
        smi = _props_smiles_from_properties(pj)
        if smi: return smi

    raise ValueError("could not resolve SMILES from that name")

# --------- Routes ----------
@app.route("/api/resolve_any", methods=["POST", "GET", "OPTIONS"])
def api_resolve_any():
    """
    Resolve a query (SMILES or name/CAS/InChIKey) → SMILES
      POST JSON: {"query": "..."}
      GET  ?query=...
    Returns: {"smiles": "..."} or 4xx on failure
    """
    try:
        if request.method == "GET":
            q = (request.args.get("query") or "").strip()
        else:
            data = request.get_json(force=True, silent=True) or {}
            q = (data.get("query") or "").strip()
        smi = resolve_query_to_smiles(q)
        return jsonify({"smiles": smi})
    except requests.HTTPError as e:
        return jsonify({"error": f"HTTP {e.response.status_code} from PubChem"}), 502
    except Exception as e:
        print("resolve_any error:", repr(e))
        return jsonify({"error": str(e)}), 404

@app.post("/api/smiles2sdf")
def api_smiles2sdf():
    data = request.get_json(force=True)
    smiles = (data.get("smiles") or "").trim()
    if not smiles:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        mol = smiles_to_rdkit_with_H(smiles)
        embed_3d_inplace(mol)
        sdf = mol_to_sdf_text(mol)
        return Response(sdf, mimetype="chemical/x-mdl-sdfile")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/api/gjf")
def api_gjf():
    data = request.get_json(force=True)
    smiles = (data.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"error": "Missing 'smiles'"}), 400
    header = data.get("header", "").strip()
    title  = data.get("title", smiles)
    include_conn = bool(data.get("include_connectivity", True))
    try:
        mol = smiles_to_rdkit_with_H(smiles)
        embed_3d_inplace(mol)
        gjf = format_gaussian(mol, title, header, include_conn)
        return Response(gjf, mimetype="text/plain")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/")
def ok():
    return "ChemWeaver backend OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
