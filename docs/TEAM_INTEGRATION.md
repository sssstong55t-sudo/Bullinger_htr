# Shared website integration review

Reviewed on November 14, 2026 against `ethanganz/ase-manuscript-project` commit
`62ab207aad1db1c1f99d19e3dabbd46ac4385bbe` and this repository's default pipeline.
This is a source and contract review, not a production deployment acceptance test.

## Current state

The two repositories contain the core recognition and website components, but
upload-to-recognition-to-map is not connected in the shared repository yet.
Its README describes a live website. Its [test workflow](https://github.com/ethanganz/ase-manuscript-project/actions/runs/42689563966)
and [Docker image build](https://github.com/ethanganz/ase-manuscript-project/actions/runs/42689564227)
succeeded at the reviewed commit; those checks do not test neural inference or
prove which image version the server currently runs.

| Component | Evidence in shared repository | Remaining work |
| --- | --- | --- |
| Uploads and page images | `backend/services/ingest.py` creates documents and pages in Supabase, with pages marked `pending` | Enqueue and process those pages |
| Transcription display | `frontend/src/components/CollectionViewer.jsx` calls `getMockTranscription` and `getMockPlaces` | Fetch actual page results and show pending/failed/complete states |
| Page API | `backend/routes/documents.py` serves metadata and images | Add a permission-checked page-result endpoint |
| Place lookup | `backend/services/place_lookup.py` and `candidates.py` resolve candidates and countries | Adapt the field names and preserve the original candidate list |
| Database | `002_place_names.sql` stores places and mentions linked to one non-null place UUID | Store line text, geometry, offsets, unresolved mentions, and full candidate arrays; keep reviewed selections separately |
| Packaging | Compose runs only frontend and backend; backend build context is `./backend` | Add a recognition worker with this repository, its dependencies and installed models |
| Vendored models/code | `HTR-NLP/` describes the older TrOCR/GLiNER flow | Pin a reviewed version of this repository and use public TrOCR plus mBERT/aliases |

The existing backend image neither copies the sibling `HTR-NLP/` directory nor
installs the recognition dependencies. Copying a JSON example into the frontend
will not make later user uploads run inference.

## Worker invocation

Provision the README dependencies and model weights once in the worker image or
persistent model volume. For each stored page, download its image into a job
working directory, then invoke the script with an argument list (not a shell
command assembled from uploaded filenames):

```python
subprocess.run([
    worker_python, str(pipeline_root / "scripts/run_page_pipeline.py"),
    "--image", str(local_page_image),
    "--output-dir", str(new_job_directory),
    "--page-id", str(database_page_uuid),
    "--device", "auto",
], check=True)
```

Use pending/running/complete/failed states, a timeout, and an idempotency key such
as page UUID plus pipeline version. Persist failure details for operators.
Store `page_response.json`, crops and preview before marking the page complete.
A retry should use a fresh directory and avoid duplicate database mentions.
Use a background worker rather than running a long model job in the upload
HTTP request. This script loads models per invocation; a long-lived model
service can be added later for throughput.

## JSON and candidate contract

`page_response.json` contains `page`, `pipeline`, `summary`, and ordered `lines`.
Each line contains `line_id`, relative `image`, `bbox`, `polygon`, `text`, and
`places`. Each mention has `start`, `end`, `text`,
`official_place_candidates`, and `place_link_status`.

The shared resolver currently expects different keys. This is the minimal
mapping for its existing `resolve_mention` function:

```python
team_mention = {
    "observed_text": mention["text"],
    "place_candidates": mention["official_place_candidates"],
}
choice = resolve_mention(team_mention, lookup)
```

Pass `lookup` loaded from the shared backend's `resources/place_variants.json`
and `resources/localities.xml`. The compact candidates in this repository do
not include `official_name_fields.country`; the shared lookup can supply
country data. Prefer enrichment by `official_place_id` where available, and
retain provenance when its word-based lookup adds further candidates.

A contract check on page `12982-1` gave:

| Literal mention | Direct unadapted call | Mapped call with team lookup |
| --- | --- | --- |
| Zurich | no_candidates | ambiguous |
| Baden | no_candidates | resolved |
| Costantz (two mentions) | no_candidates | resolved |

The shared resolver's `resolved` status is an automatic lookup decision, not a
human verification or an accuracy score. It filters candidates without
coordinates and may merge nearby same-name entries. Keep the original array
and `candidate_for_review` / `ambiguous` / `unmatched` status in stored results;
do not discard unmatched or coordinate-less mentions. Store a later selected
place and its review status separately. Keep official IDs such as `l587`
separate from the database's own place UUIDs.

For the UI:

- Replace relative image paths with authenticated storage/API URLs.
- Keep Python Unicode code-point offsets (end exclusive). In JavaScript use
  `Array.from(text).slice(start, end).join("")` to match them; ordinary string
  slicing uses UTF-16 units and can differ for non-BMP characters.
- Use the normalized `input/page.png` coordinate frame for overlays. If the
  website displays the original EXIF-rotated image instead, apply the same
  orientation and scaling before drawing boxes/polygons.
- Display multiple candidates and missing names/coordinates explicitly. Only
  draw pins when both coordinates are available; do not invent `(0, 0)`.
- Do not interpret a single candidate as a confirmed historical location.

## Deployment acceptance

The existing frontend/backend Docker, nginx API proxy, Supabase integration,
and CI provide a useful deployment base. Before accepting the complete product:

1. Add the worker and result persistence/API described above.
2. Replace the mock frontend calls and test a real uploaded page end to end.
3. Configure the actual Supabase project/bucket/migrations, fixed JWT secret,
   mail settings, HTTPS/cookies and deployed image versions on the server.
4. Verify owner/private-collection access for result JSON and crops, restart
   recovery, retries, model availability, runtime/memory and concurrent jobs.
5. Download a result archive from the deployed app and validate it after
   extraction. The local browser-download check of the Python output is not
   a check of the shared website's future download endpoint.

No shared-repository files or live server configuration were modified in this
review. Its deployment credentials, running server version and a real upload
round trip were not inspected.
