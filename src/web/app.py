import os
import sys
from pathlib import Path

# Add src to path to allow imports from sibling modules
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from agent import MedicineAssistantAgent

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-key-medicine-assistant")

# Database connection
def get_db_connection():
    # Use DATABASE_URL if available, otherwise individual params
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        conn = psycopg2.connect(database_url)
    else:
        conn = psycopg2.connect(
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port"),
            dbname=os.getenv("dbname")
        )
    return conn

# Initialize Agent
# We initialize it lazily or globally. 
# Note: In a production app, you might want to handle this differently.
try:
    agent = MedicineAssistantAgent()
except Exception as e:
    print(f"Warning: Could not initialize Agent: {e}")
    agent = None

@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute('SELECT * FROM Patients ORDER BY created_at DESC')
        patients = cur.fetchall()
    except Exception as e:
        flash(f"Error fetching patients: {e}", "danger")
        patients = []
    finally:
        cur.close()
        conn.close()
    return render_template('index.html', patients=patients)

@app.route('/patient/add', methods=('GET', 'POST'))
def add_patient():
    if request.method == 'POST':
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO Patients (
                    Patient_ID, Name, Age, Gender, Height_cm, Weight_kg, 
                    Diabetes_Type, Duration_Years, Comorbidities, Latest_HbA1c, 
                    Current_Meds, eGFR_ml_min, Recent_Symptoms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                request.form['patient_id'],
                request.form['name'],
                request.form['age'],
                request.form['gender'],
                request.form['height_cm'],
                request.form['weight_kg'],
                request.form['diabetes_type'],
                request.form['duration_years'],
                request.form['comorbidities'],
                request.form['latest_hba1c'],
                request.form['current_meds'],
                request.form['egfr_ml_min'],
                request.form['recent_symptoms']
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            flash('Patient added successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Error adding patient: {e}", 'danger')
            
    return render_template('patient_form.html', action='Add', patient={})

@app.route('/patient/edit/<patient_id>', methods=('GET', 'POST'))
def edit_patient(patient_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    if request.method == 'POST':
        try:
            cur.execute("""
                UPDATE Patients SET
                    Name = %s, Age = %s, Gender = %s, Height_cm = %s, Weight_kg = %s,
                    Diabetes_Type = %s, Duration_Years = %s, Comorbidities = %s,
                    Latest_HbA1c = %s, Current_Meds = %s, eGFR_ml_min = %s, Recent_Symptoms = %s
                WHERE Patient_ID = %s
            """, (
                request.form['name'],
                request.form['age'],
                request.form['gender'],
                request.form['height_cm'],
                request.form['weight_kg'],
                request.form['diabetes_type'],
                request.form['duration_years'],
                request.form['comorbidities'],
                request.form['latest_hba1c'],
                request.form['current_meds'],
                request.form['egfr_ml_min'],
                request.form['recent_symptoms'],
                patient_id
            ))
            conn.commit()
            flash('Patient updated successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Error updating patient: {e}", 'danger')
        finally:
            cur.close()
            conn.close()
    else:
        cur.execute('SELECT * FROM Patients WHERE Patient_ID = %s', (patient_id,))
        patient = cur.fetchone()
        cur.close()
        conn.close()
        if not patient:
            flash('Patient not found', 'danger')
            return redirect(url_for('index'))
        return render_template('patient_form.html', action='Edit', patient=patient)

@app.route('/patient/delete/<patient_id>', methods=('POST',))
def delete_patient(patient_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('DELETE FROM Patients WHERE Patient_ID = %s', (patient_id,))
        conn.commit()
        flash('Patient deleted successfully!', 'success')
    except Exception as e:
        flash(f"Error deleting patient: {e}", 'danger')
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('index'))

@app.route('/consult', methods=('GET', 'POST'))
def consult():
    patient_id = request.args.get('patient_id') or request.form.get('patient_id')
    patient = None
    
    if patient_id:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM Patients WHERE Patient_ID = %s', (patient_id,))
        patient = cur.fetchone()
        cur.close()
        conn.close()
    
    response = None
    if request.method == 'POST':
        if not agent:
            flash("Agent not initialized. Check configuration.", "danger")
        else:
            # Gather data from form
            consult_data = {
                "patient_id": patient_id,
                "name": patient['Name'] if patient else request.form.get('name', 'Unknown'),
                "latest_hba1c": request.form.get('latest_hba1c'),
                "blood_glucose": request.form.get('blood_glucose'),
                "blood_pressure": request.form.get('blood_pressure'),
                "egfr": request.form.get('egfr'),
                "lipid_panel": request.form.get('lipid_panel'),
                "symptoms_notes": request.form.get('symptoms_notes'),
                "treatment_adjustments": request.form.get('treatment_adjustments'),
                "current_meds": patient['Current_Meds'] if patient else request.form.get('current_meds', '')
            }
            
            # Construct a detailed query for the agent
            query = f"""
            Please analyze the following patient data and provide recommendations:
            
            Patient: {consult_data['name']} (ID: {consult_data['patient_id']})
            
            Current Vitals & Labs:
            - Latest HbA1c: {consult_data['latest_hba1c']}%
            - Immediate Blood Glucose: {consult_data['blood_glucose']} mg/dL
            - Blood Pressure: {consult_data['blood_pressure']} mmHg
            - eGFR: {consult_data['egfr']} ml/min
            - Lipid Panel: {consult_data['lipid_panel']}
            
            Clinical Notes:
            - Symptoms/Notes: {consult_data['symptoms_notes']}
            - Recent Treatment Adjustments: {consult_data['treatment_adjustments']}
            - Current Medications (from record): {consult_data['current_meds']}
            
            Based on this, please provide:
            1. Assessment of current control.
            2. Recommendations for medication adjustment if needed.
            3. Suggested monitoring plan.
            """
            
            try:
                # Invoke the agent
                response = agent.invoke(query, patient_info=str(consult_data))
            except Exception as e:
                flash(f"Error generating report: {e}", "danger")
                
    return render_template('consult.html', patient=patient, response=response)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000 for local development
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
