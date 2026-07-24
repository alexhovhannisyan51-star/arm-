import os
import re
import math
import requests
import pandas as pd
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
LOCAL_FILE = "variant_summary.txt.gz"

ACCEPTED_SIGNIFICANCE = [
    "Pathogenic",
    "Likely pathogenic",
    "Uncertain significance",
    "Likely benign",
    "Benign",
]

USECOLS = [
    "VariationID",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "ReviewStatus",
    "RS# (dbSNP)",
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "Assembly",
    "PhenotypeList",
]

HGVS_C_RE = re.compile(r"(c\.[^\s)]+)")
HGVS_P_RE = re.compile(r"\((p\.[^)]+)\)")


def download_clinvar():
    print("Downloading ClinVar variant_summary.txt.gz ...")
    with requests.get(CLINVAR_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(LOCAL_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    print("Download complete.")


def classify_significance(raw):
    if not isinstance(raw, str):
        return None
    for term in ACCEPTED_SIGNIFICANCE:
        if term in raw:
            return term
    return None


def extract_hgvs(name):
    if not isinstance(name, str):
        return None, None
    c_match = HGVS_C_RE.search(name)
    p_match = HGVS_P_RE.search(name)
    hgvs_c = c_match.group(1).rstrip(",") if c_match else None
    hgvs_p = p_match.group(1) if p_match else None
    return hgvs_c, hgvs_p


def main():
    download_clinvar()

    print("Loading into pandas (this may take a few minutes)...")
    df = pd.read_csv(
        LOCAL_FILE,
        sep="\t",
        usecols=USECOLS,
        dtype=str,
        compression="gzip",
        low_memory=False,
    )
    print(f"Raw rows loaded: {len(df)}")

    df = df[df["Assembly"] == "GRCh38"]
    print(f"After GRCh38 filter: {len(df)}")

    df["clinvar_significance"] = df["ClinicalSignificance"].apply(classify_significance)
    df = df[df["clinvar_significance"].notna()]
    print(f"After significance filter: {len(df)}")

    hgvs = df["Name"].apply(extract_hgvs)
    df["hgvs_c"] = hgvs.apply(lambda x: x[0])
    df["hgvs_p"] = hgvs.apply(lambda x: x[1])

    df["variant_id"] = "clinvar_" + df["VariationID"].astype(str)
    df["rsid"] = df["RS# (dbSNP)"].apply(
        lambda x: f"rs{x}" if isinstance(x, str) and x not in ("-1", "") else None
    )
    df["clinvar_conditions"] = df["PhenotypeList"].apply(
        lambda x: "; ".join([p.strip() for p in x.split("|") if p.strip() and p.strip().lower() not in ("not provided", "not specified")])
        if isinstance(x, str) else None
    )

    out = pd.DataFrame({
        "variant_id": df["variant_id"],
        "chrom": df["Chromosome"],
        "pos": pd.to_numeric(df["PositionVCF"], errors="coerce"),
        "ref": df["ReferenceAlleleVCF"],
        "alt": df["AlternateAlleleVCF"],
        "gene": df["GeneSymbol"],
        "rsid": df["rsid"],
        "hgvs_c": df["hgvs_c"],
        "hgvs_p": df["hgvs_p"],
        "clinvar_significance": df["clinvar_significance"],
        "clinvar_review_status": df["ReviewStatus"],
        "clinvar_conditions": df["clinvar_conditions"],
        "gnomad_af_exome": None,
        "gnomad_af_genome": None,
        "cadd_phred": None,
    }).drop_duplicates(subset=["variant_id"]).reset_index(drop=True)

    print(f"Final clean dataset: {len(out)} variants")
    print(out["clinvar_significance"].value_counts())

    records = out.where(pd.notnull(out), None).to_dict(orient="records")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    BATCH_SIZE = 500
    n_batches = math.ceil(len(records) / BATCH_SIZE)
    for i in range(n_batches):
        batch = records[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        supabase.table("variants").upsert(batch, on_conflict="variant_id").execute()
        if i % 20 == 0:
            print(f"Uploaded batch {i + 1}/{n_batches}")

    print(f"Done. Upserted {len(records)} variants into Supabase.")


if __name__ == "__main__":
    main()
