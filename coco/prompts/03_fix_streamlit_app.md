Running streamlit_app/streamlit_app.py against the live data fails on startup with:

    SQL compilation error: invalid identifier 'SCORED_AT'

Check every query in streamlit_app/streamlit_app.py against the real columns (DESCRIBE the
tables it reads on the pdm connection) and fix all mismatches, not only this one. Also
replace the f-string SQL that inserts a machine id or sensor name with bound parameters
(session.sql(query, params=[...])) where possible, and validate the sensor column name
against the known SENSOR_1..SENSOR_21 columns. Edit only streamlit_app/streamlit_app.py.
Do not change any tables. Reply with the list of fixes.
