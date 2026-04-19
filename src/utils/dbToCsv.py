
import csv, argparse, mysql.connector, os.path as osp



parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=3306)
parser.add_argument("--user", required=True)
parser.add_argument("--password", required=True)
parser.add_argument("--database", required=True)
parser.add_argument("--results_root", required=True, help="Local folder that contains your Data/results")
parser.add_argument("--out_csv", default="rows.csv")
args = parser.parse_args()

conn = mysql.connector.connect(
    host=args.host, port=args.port, user=args.user,
    password=args.password, database=args.database
)
cur = conn.cursor(dictionary=True)


cur.execute("""
    SELECT filename, filepath, diameter, thickness, ratio, ref_index
    FROM ImageData
""")

rows = []
root = osp.abspath(args.results_root)
for r in cur:
    fp = (r.get("filepath") or "").replace("\\","/")
    rel = ""
    if fp:
        try:
            rel = osp.relpath(fp, root).replace("\\","/")
        except Exception:

            rel = ""
    rows.append({
        "relative_path": rel,
        "filepath": fp,
        "filename": (r.get("filename") or ""),
        "diameter": r["diameter"],
        "thickness": r["thickness"],
        "ratio": r["ratio"],
        "ref_index": r["ref_index"],
    })

cur.close(); conn.close()

with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[
        "relative_path","filepath","filename",
        "diameter","thickness","ratio","ref_index"
    ])
    w.writeheader(); w.writerows(rows)

print(f"Wrote {len(rows)} rows to {args.out_csv}")
