#!/usr/bin/env python3
"""
Import `scripts/Patient_Diabetes_Records.csv` into a Postgres/Supabase `Patients` table.

Usage:
  python3 scripts/import_patients_to_postgres.py

Environment:
- Uses `DATABASE_URL` if set, otherwise uses `user`, `password`, `host`, `port`, `dbname` from `.env` (same as `src/main.py`).

Behavior:
- Reads `scripts/Patient_Diabetes_Records.csv` by default.
- Upserts rows into `Patients` on conflict of `Patient_ID`.
"""
import os
import csv
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values


load_dotenv()

CSV_PATH = os.path.join(os.path.dirname(__file__), "Patient_Diabetes_Records.csv")
DATABASE_URL = os.getenv("DATABASE_URL")
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")


def get_conn():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    return psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME)


def parse_row(row: dict):
    def try_cast(v, typ):
        if v is None or v == "":
            return None
        try:
            return typ(v)
        except Exception:
            return None

    return (
        row.get("Patient_ID"),
        row.get("Name") or None,
        try_cast(row.get("Age"), int),
        row.get("Gender") or None,
        try_cast(row.get("Height_cm"), float),
        try_cast(row.get("Weight_kg"), float),
        row.get("Diabetes_Type") or None,
        try_cast(row.get("Duration_Years"), float),
        row.get("Comorbidities") or None,
        try_cast(row.get("Latest_HbA1c"), float),
        row.get("Current_Meds") or None,
        try_cast(row.get("eGFR_ml_min"), float),
        row.get("Recent_Symptoms") or None,
    )


def import_csv(conn, csv_path=CSV_PATH):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [parse_row(r) for r in reader]

    cur = conn.cursor()
    sql = (
        "INSERT INTO Patients (Patient_ID, Name, Age, Gender, Height_cm, Weight_kg, Diabetes_Type, Duration_Years, Comorbidities, Latest_HbA1c, Current_Meds, eGFR_ml_min, Recent_Symptoms) "
        "VALUES %s "
        "ON CONFLICT (Patient_ID) DO UPDATE SET "
        "Name = EXCLUDED.Name, Age = EXCLUDED.Age, Gender = EXCLUDED.Gender, Height_cm = EXCLUDED.Height_cm, Weight_kg = EXCLUDED.Weight_kg, "
        "Diabetes_Type = EXCLUDED.Diabetes_Type, Duration_Years = EXCLUDED.Duration_Years, Comorbidities = EXCLUDED.Comorbidities, "
        "Latest_HbA1c = EXCLUDED.Latest_HbA1c, Current_Meds = EXCLUDED.Current_Meds, eGFR_ml_min = EXCLUDED.eGFR_ml_min, Recent_Symptoms = EXCLUDED.Recent_Symptoms"
    )

    # Use execute_values for efficient bulk insert
    execute_values(cur, sql, rows, template=None)
    conn.commit()
    cur.close()
    return len(rows)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return

    conn = get_conn()
    try:
        print("Connected to Postgres — importing CSV...")
        count = import_csv(conn, CSV_PATH)
        print(f"Imported/updated {count} rows into Patients table.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
