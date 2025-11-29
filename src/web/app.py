import io
import os
import re
import sys
import tempfile
from pathlib import Path
from flask import Flask, flash, redirect, render_template, request, url_for, send_file, jsonify
from markdown import markdown as md_to_html
try:
    from weasyprint import HTML, CSS
except Exception:
    HTML = None
    CSS = None
from utils.translate import translate_en_to_ar
from utils.stt import stt as stt_tool
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Ensure `src` package is importable when running this script directly
sys.path.append(str(Path(__file__).parent.parent))

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
    return render_template('index.html')

@app.route('/patients')
def patients():
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
    return render_template('patients.html', patients=patients)

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
            return redirect(url_for('patients'))
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
            return redirect(url_for('patients'))
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
            return redirect(url_for('patients'))
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
    
    result = None
    if request.method == 'POST':
        if not agent:
            flash("Agent not initialized. Check configuration.", "danger")
        else:
            try:
                # Gather structured patient data from form and database
                consult_data = {
                    # Basic identification
                    "patient_id": patient_id,
                    "name": patient.get('name') if patient else request.form.get('name', 'Unknown'),
                    
                    # Demographics
                    "age": request.form.get('age') or (patient.get('age') if patient else None),
                    "gender": request.form.get('gender') or (patient.get('gender') if patient else None),
                    "weight": request.form.get('weight') or (patient.get('weight_kg') if patient else None),
                    
                    # Diabetes profile
                    "diabetes_type": request.form.get('diabetes_type') or (patient.get('diabetes_type') if patient else None),
                    "duration_years": request.form.get('duration_years') or (patient.get('duration_years') if patient else None),
                    
                    # Lab values
                    "latest_hba1c": request.form.get('latest_hba1c') or (patient.get('latest_hba1c') if patient else None),
                    "blood_glucose": request.form.get('blood_glucose'),
                    "blood_pressure": request.form.get('blood_pressure'),
                    "egfr": request.form.get('egfr') or (patient.get('egfr_ml_min') if patient else None),
                    "lipid_panel": request.form.get('lipid_panel'),
                    
                    # Clinical information
                    "symptoms_notes": request.form.get('symptoms_notes') or (patient.get('recent_symptoms') if patient else ''),
                    "treatment_adjustments": request.form.get('treatment_adjustments'),
                    "current_meds": patient.get('current_meds') if patient else request.form.get('current_meds', ''),
                    
                    # Additional fields
                    "comorbidities": request.form.get('comorbidities') or (patient.get('comorbidities') if patient else None),
                    "allergies": request.form.get('allergies') or (patient.get('allergies') if patient else None),
                }
                
                # Clean up None values - convert to empty string for better display
                consult_data = {k: (v if v is not None else '') for k, v in consult_data.items()}
                
                # Construct a simplified query - the agent will use the structured data
                query = f"""
Please analyze the patient data and provide comprehensive physician and patient reports.

Patient: {consult_data['name']} (ID: {consult_data['patient_id']})

Latest vitals and labs provided in structured data.
"""
                
                # Invoke the agent with structured patient_info
                result = agent.invoke(query, patient_info=consult_data)
                
                # Check if clarification is needed
                if result.get("needs_clarification"):
                    flash("Additional patient information is required for a complete analysis.", "warning")
                
                # Display safety alerts prominently
                if result.get("safety_alerts"):
                    for alert in result["safety_alerts"]:
                        flash(alert, "danger")
                        
            except Exception as e:
                flash(f"Error generating report: {str(e)}", "danger")
                # Log the full error for debugging
                import traceback
                print(f"Error in consult route: {traceback.format_exc()}")
                
    # Convert returned markdown to HTML server-side so the template receives HTML
    phys_md = result.get('physician_report', '') if result else ''
    pat_md = result.get('patient_report', '') if result else ''

    # Translate patient report to Arabic synchronously (done prior to rendering)
    pat_md_ar = ''
    try:
        if pat_md:
            pat_md_ar = translate_en_to_ar(pat_md)
    except Exception as e:
        # If translation fails, log and continue with empty Arabic report
        print(f"Translation error: {e}")

    phys_html = md_to_html(phys_md, extensions=['extra', 'nl2br']) if phys_md else ''
    pat_html = md_to_html(pat_md, extensions=['extra', 'nl2br']) if pat_md else ''
    pat_html_ar = md_to_html(pat_md_ar, extensions=['extra', 'nl2br']) if pat_md_ar else ''

    return render_template('consult.html', patient=patient, result=result, physician_html=phys_html, patient_html=pat_html, patient_html_ar=pat_html_ar)


@app.route('/consult/pdf/<report_type>', methods=('POST',))
def consult_pdf(report_type: str):
    """Generate a PDF for a given report type (physician|patient).

    Accepts JSON body: { "md": "<markdown>", "html": "<optional full html>", "filename": "optional.pdf" }
    If `html` is provided, it will be used directly. Otherwise `md` will be converted to HTML.
    Uses WeasyPrint for HTML->PDF rendering. Returns PDF as attachment.
    """
    try:
        data = request.get_json(force=True, silent=True) or request.form or {}
        filename = data.get('filename') or f"{report_type}_report.pdf"

        html_input = data.get('html')
        md_input = data.get('md') or data.get('report')

        if html_input:
            body_html = html_input
        elif md_input:
            # Convert markdown to HTML (allowing common extensions)
            body_html = md_to_html(md_input, extensions=['extra', 'nl2br'])
        else:
            body_html = ''

        if HTML is None:
            return (jsonify({'error': 'WeasyPrint is not installed on the server. Please install weasyprint and required system packages.'}), 501)

        # Wrap the body_html in a minimal HTML document and include some basic styles
        css_text = """
            body { font-family: Arial, Helvetica, sans-serif; font-size: 12px; color: #111827; }
            h1 { font-size: 18px; }
            h2 { font-size: 16px; }
            p { margin: 0 0 8px 0; }
            ul { margin: 0 0 8px 18px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 8px; }
            table, th, td { border: 1px solid #ddd; padding: 6px; }
            strong { font-weight: bold; }
            br {line-height: 1.2}
        """

        full_html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>{report_type.title()} Report</title>
        </head>
        <body>
          {body_html}
        </body>
        </html>
        """

        # Render PDF with WeasyPrint
        html_obj = HTML(string=full_html, base_url=request.base_url)
        pdf_bytes = html_obj.write_pdf(stylesheets=[CSS(string=css_text)])

        bio = io.BytesIO(pdf_bytes)
        bio.seek(0)
        return send_file(bio, as_attachment=True, download_name=filename, mimetype='application/pdf')
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return (jsonify({'error': str(e)}), 500)


    @app.route('/consult/stt', methods=('POST',))
    def consult_stt():
        """Accept an audio file upload (form field 'audio') and return the STT tool output as JSON.

        Currently the `utils.stt.stt()` function is a placeholder and does not accept audio bytes.
        This endpoint accepts the uploaded file for future use and returns the dict from the STT tool.
        """
        # Basic validation
        if 'audio' not in request.files:
            return jsonify({'error': "Missing 'audio' file in request"}), 400

        audio_file = request.files['audio']
        # We don't currently process the audio bytes; pass through to the STT placeholder
        try:
            # If in future stt_tool accepts bytes, we can pass audio_file.read()
            stt_result = stt_tool()
            return jsonify(stt_result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# Alternative: Async version if using async Flask (e.g., with Quart)
@app.route('/consult', methods=('GET', 'POST'))
async def consult_async():
    patient_id = request.args.get('patient_id') or request.form.get('patient_id')
    patient = None
    
    if patient_id:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM Patients WHERE Patient_ID = %s', (patient_id,))
        patient = cur.fetchone()
        cur.close()
        conn.close()
    
    result = None
    if request.method == 'POST':
        if not agent:
            flash("Agent not initialized. Check configuration.", "danger")
        else:
            try:
                # Gather structured patient data
                consult_data = {
                    "patient_id": patient_id,
                    "name": patient.get('name') if patient else request.form.get('name', 'Unknown'),
                    "age": request.form.get('age') or (patient.get('age') if patient else None),
                    "gender": request.form.get('gender') or (patient.get('gender') if patient else None),
                    "weight": request.form.get('weight') or (patient.get('weight_kg') if patient else None),
                    "diabetes_type": request.form.get('diabetes_type') or (patient.get('diabetes_type') if patient else None),
                    "duration_years": request.form.get('duration_years') or (patient.get('duration_years') if patient else None),
                    "latest_hba1c": request.form.get('latest_hba1c') or (patient.get('latest_hba1c') if patient else None),
                    "blood_glucose": request.form.get('blood_glucose'),
                    "blood_pressure": request.form.get('blood_pressure'),
                    "egfr": request.form.get('egfr') or (patient.get('egfr_ml_min') if patient else None),
                    "lipid_panel": request.form.get('lipid_panel'),
                    "symptoms_notes": request.form.get('symptoms_notes') or (patient.get('recent_symptoms') if patient else ''),
                    "treatment_adjustments": request.form.get('treatment_adjustments'),
                    "current_meds": patient.get('current_meds') if patient else request.form.get('current_meds', ''),
                    "comorbidities": request.form.get('comorbidities') or (patient.get('comorbidities') if patient else None),
                    "allergies": request.form.get('allergies') or (patient.get('allergies') if patient else None),
                }
                
                consult_data = {k: (v if v is not None else '') for k, v in consult_data.items()}
                
                query = f"Please analyze patient {consult_data['name']} (ID: {consult_data['patient_id']}) and provide comprehensive reports."
                
                # Use async invoke
                result = await agent.ainvoke(query, patient_info=consult_data)
                
                if result.get("needs_clarification"):
                    flash("Additional patient information is required for a complete analysis.", "warning")
                
                if result.get("safety_alerts"):
                    for alert in result["safety_alerts"]:
                        flash(alert, "danger")
                        
            except Exception as e:
                flash(f"Error generating report: {str(e)}", "danger")
                import traceback
                print(f"Error in consult route: {traceback.format_exc()}")
                
    # Convert returned markdown to HTML server-side so the template receives HTML
    phys_md = result.get('physician_report', '') if result else ''
    pat_md = result.get('patient_report', '') if result else ''

    # Translate patient report to Arabic synchronously (done prior to rendering)
    pat_md_ar = ''
    try:
        if pat_md:
            pat_md_ar = translate_en_to_ar(pat_md)
    except Exception as e:
        print(f"Translation error: {e}")

    phys_html = md_to_html(phys_md, extensions=['extra', 'nl2br']) if phys_md else ''
    pat_html = md_to_html(pat_md, extensions=['extra', 'nl2br']) if pat_md else ''
    pat_html_ar = md_to_html(pat_md_ar, extensions=['extra', 'nl2br']) if pat_md_ar else ''

    return render_template('consult.html', patient=patient, result=result, physician_html=phys_html, patient_html=pat_html, patient_html_ar=pat_html_ar)


@app.route('/consult/stt', methods=['POST'])
def consult_stt():
    """
    API endpoint for speech-to-text processing in consult form.
    Accepts audio file upload and returns extracted patient data as JSON.
    
    Expected form data:
        - audio: Audio file (webm, mp3, wav, etc.)
    
    Returns:
        JSON with keys matching consult form fields:
        - HbA1c, Blood Glucose, eGFR, Lipid Panel, Blood Pressure, 
        - Symptoms & Notes, Treatments Adjustments
    """
    # Check if audio file is in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OpenRouter API key not configured"}), 500
    
    # Save uploaded audio to a temporary file
    try:
        # Determine file extension from original filename or default to webm
        original_ext = os.path.splitext(audio_file.filename)[1] or '.webm'
        
        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)
            tmp_audio_path = tmp_audio.name
        
        try:
            # Process audio with STT - returns dict with form field keys
            try:
                result = stt_tool(
                    audio_path=tmp_audio_path,
                    api_key=api_key
                )
            except Exception as inner_e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"STT processing failed: {str(inner_e)}"}), 500

            # Return the result directly (keys match what frontend expects)
            return jsonify(result)
            
        finally:
            # Clean up temporary audio file
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
                
    except Exception as e:
        print(f"STT API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Get port from environment variable or default to 5000 for local development
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
