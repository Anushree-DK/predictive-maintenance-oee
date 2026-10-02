Add a new analysis query to sql/002_analysis_worksheet.sql and run it.

Question: when the model puts an in-service engine in a risk class, how often does
that engine really fail within the class's window?

- PDM.PUBLIC.FLEET_PREDICTIONS has the latest RISK_CLASS per in-service engine
  (CRITICAL / HIGH / MEDIUM / LOW, based on RUL_LOWER: <= 15, <= 40, <= 90, above).
- PDM.PUBLIC.FLEET_GROUND_TRUTH has NASA's TRUE_RUL for the same engines.
- Per risk class: number of engines, average predicted RUL, average true RUL, and the
  share of engines whose TRUE_RUL is within that class's threshold (CRITICAL 15,
  HIGH 40, MEDIUM 90; LOW has no window, so report NULL).

Steps:
1. Write the query and run it against the pdm connection to check it works.
2. Append it to sql/002_analysis_worksheet.sql as query "6." with a one-line comment,
   in the same style as the existing queries.
3. Reply with the result table.

Do not change any tables or any other file.
