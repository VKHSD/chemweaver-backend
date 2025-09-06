# app.py
# ChemWeaver backend: RDKit-powered SMILES → 3D SDF and Gaussian .gjf
#
# Requirements:
#   pip install flask flask-cors rdkit-pypi
# Run:
#   python app.py    (serves at http://127.0.0.1:5000)

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

app = Flask(__name__)
CORS(app)
import re
import requests

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def _pc_json(url, timeout=10):
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    # PubChem sometimes returns 200 with a Fault payload
    fault = js.get("Fault")
    if fault:
        details = fault.get("Details") or []
        msg = details[0] if details else fault.get("Message", "PubChem fault")
        raise ValueError(str(msg))
    return js

def _props_smiles_from_properties(js):
    p = (js.get("PropertyTable") or {}).get("Properties") or []
    if not p:
        return None
    p = p[0]
    return p.get("IsomericSMILES") or p.get("CanonicalSMILES")

@app.post("/api/resolve_name")
def api_resolve_name():
    """
    Input JSON: {"query": "<name | CAS | InChIKey>"}
    Output JSON: {"smiles": "..."}  (400 on failure)
    """
    data = request.get_json(force=True)
    q = (data.get("query") or "").strip()
    if not q:
        return jsonify({"error": "missing query"}), 400

    enc = requests.utils.quote(q, safe="")
    is_inchikey = re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", q, flags=re.I) is not None
    is_cas      = re.match(r"^\d{2,7}-\d{2}-\d$", q) is not None

    try:
        # 1) InChIKey → SMILES
        if is_inchikey:
            js = _pc_json(f"{PUBCHEM}/compound/inchikey/{enc}/property/IsomericSMILES,CanonicalSMILES/JSON")
            smi = _props_smiles_from_properties(js)
            if smi: return jsonify({"smiles": smi})
            return jsonify({"error": "no SMILES for that InChIKey"}), 404

        # 2) CAS → (properties | CID → properties)
        if is_cas:
            try:
                js = _pc_json(f"{PUBCHEM}/compound/xref/rn/{enc}/property/IsomericSMILES,CanonicalSMILES/JSON")
                smi = _props_smiles_from_properties(js)
                if smi: return jsonify({"smiles": smi})
            except Exception:
                pass
            cj = _pc_json(f"{PUBCHEM}/compound/xref/rn/{enc}/cids/JSON")
            cids = ((cj.get("IdentifierList") or {}).get("CID") or [])
            if cids:
                cid = cids[0]
                pj = _pc_json(f"{PUBCHEM}/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON")
                smi = _props_smiles_from_properties(pj)
                if smi: return jsonify({"smiles": smi})
            return jsonify({"error": "no SMILES for that CAS"}), 404

        # 3) Name → (properties | CID → properties)
        try:
            js = _pc_json(f"{PUBCHEM}/compound/name/{enc}/property/IsomericSMILES,CanonicalSMILES/JSON")
            smi = _props_smiles_from_properties(js)
            if smi: return jsonify({"smiles": smi})
        except Exception:
            pass

        cj = _pc_json(f"{PUBCHEM}/compound/name/{enc}/cids/JSON")
        cids = ((cj.get("IdentifierList") or {}).get("CID") or [])
        if cids:
            cid = cids[0]
            pj = _pc_json(f"{PUBCHEM}/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON")
            smi = _props_smiles_from_properties(pj)
            if smi: return jsonify({"smiles": smi})

        return jsonify({"error": "could not resolve SMILES from that name"}), 404

    except requests.HTTPError as e:
        return jsonify({"error": f"HTTP {e.response.status_code} from PubChem"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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
    # ETKDG + quick UFF relaxation
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def mol_to_sdf_text(mol) -> str:
    # Make a temp writer to string
    # (RDKit's SDWriter expects a file-like; use in-memory)
    from io import StringIO
    sio = StringIO()
    w = Chem.SDWriter(sio)
    w.write(mol)
    w.flush()
    return sio.getvalue()

def build_connectivity_lines(mol) -> str:
    # Symmetric neighbor listing with true bond orders
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
    # Coordinates
    conf = mol.GetConformer()
    coords = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        pos = conf.GetAtomPosition(idx)
        coords.append((a.GetSymbol(), pos.x, pos.y, pos.z))

    # Build text
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

@app.route("/api/smiles2sdf", methods=["POST"])
def api_smiles2sdf():
    """
    Input JSON: {"smiles": "..."}
    Output: SDF text (3D, hydrogens added)
    """
    data = request.get_json(force=True)
    smiles = (data.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        mol = smiles_to_rdkit_with_H(smiles)
        embed_3d_inplace(mol)
        sdf = mol_to_sdf_text(mol)
        return Response(sdf, mimetype="chemical/x-mdl-sdfile")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/gjf", methods=["POST"])
def api_gjf():
    """
    Input JSON:
      {
        "smiles": "...",
        "header": "%chk=<name>.chk\n# opt freq B3LYP/6-31G(d) geom=connectivity",
        "title":  "C6H6 generated by ChemWeaver",
        "include_connectivity": true
      }
    Output: plain text .gjf
    """
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

@app.route("/")
def ok():
    return "ChemWeaver backend OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



