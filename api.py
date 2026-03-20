"""
Call Match API - Backend endpoints for Call Match functionality

Provides two main endpoints:
1. POST /api/call-match/step1 - Write call_opening from CSV to database
2. POST /api/call-match/step2 - Analyze calls and compute match scores

Both support real-time progress via WebSocket.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Path to this package (call_match_package)
CALL_MATCH_PACKAGE_PATH = Path(__file__).parent
if str(CALL_MATCH_PACKAGE_PATH) not in sys.path:
    sys.path.insert(0, str(CALL_MATCH_PACKAGE_PATH))

from loguru import logger

router = APIRouter(prefix="/api/call-match", tags=["call-match"])

# Progress tracking
_progress: Dict[str, Dict[str, Any]] = {}


class ProgressUpdate(BaseModel):
    """Progress update model"""

    step: str
    current: int
    total: int
    message: str
    status: str  # "running", "completed", "error"


def _update_progress(
    task_id: str,
    step: str,
    current: int,
    total: int,
    message: str,
    status: str = "running",
):
    """Update progress for a task"""
    _progress[task_id] = {
        "step": step,
        "current": current,
        "total": total,
        "message": message,
        "status": status,
        "percentage": int(current / total * 100) if total > 0 else 0,
    }


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "message": "Call Match API is running"}


@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Get progress for a specific task"""
    if task_id not in _progress:
        return {"status": "not_found", "message": "Task not found"}
    return _progress[task_id]


class Step1Request(BaseModel):
    """Step 1 request - no parameters needed, uses config"""

    pass


class Step1Response(BaseModel):
    """Step 1 response"""

    success: bool
    task_id: str
    message: str
    stats: Optional[Dict[str, Any]] = None


@router.post("/step1/start")
async def start_step1():
    """
    Start Step 1: Write call_opening from CSV to database

    This runs csv_to_call_opening.py in background and returns a task_id
    for progress tracking.
    """
    import uuid

    task_id = str(uuid.uuid4())

    # Start background task
    asyncio.create_task(_run_step1(task_id))

    return Step1Response(
        success=True,
        task_id=task_id,
        message="Step 1 started - Writing call_opening to database",
    )


async def _run_step1(task_id: str):
    """Run Step 1 in background"""
    try:
        _update_progress(task_id, "step1", 0, 100, "Loading CSV file...", "running")

        # Import and run the actual script logic
        import pandas as pd
        import pymysql

        # Import config from call_match_package
        sys.path.insert(0, str(CALL_MATCH_PACKAGE_PATH))
        import importlib

        config = importlib.import_module("config")

        # Load CSV
        _update_progress(task_id, "step1", 10, 100, "Reading CSV file...", "running")

        csv_file = config.CSV_FILE
        encodings = ["utf-8", "gbk", "latin1"]
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=enc)
                break
            except Exception:
                continue

        if df is None:
            _update_progress(
                task_id, "step1", 0, 100, "Failed to read CSV file", "error"
            )
            return

        required_cols = {"Company Name", "cold_call_opening_line"}
        if not required_cols.issubset(df.columns):
            _update_progress(
                task_id,
                "step1",
                0,
                100,
                f"CSV missing columns: {required_cols - set(df.columns)}",
                "error",
            )
            return

        df_valid = df[df["cold_call_opening_line"].notna()].copy()
        total_rows = len(df_valid)

        _update_progress(
            task_id,
            "step1",
            20,
            100,
            f"Found {total_rows} valid rows to process",
            "running",
        )

        # Connect to database
        cfg = config.DB_CONFIG.copy()
        cfg["cursorclass"] = pymysql.cursors.DictCursor
        conn = pymysql.connect(**cfg)

        updated = 0
        skipped = 0

        try:
            with conn.cursor() as cur:
                for idx, (_, row) in enumerate(df_valid.iterrows()):
                    company = row["Company Name"]
                    opening = row["cold_call_opening_line"]

                    if pd.isna(company) or pd.isna(opening):
                        skipped += 1
                        continue

                    cur.execute(
                        """
                        UPDATE sdr_cdr_inout_data
                        SET call_opening = %s
                        WHERE contact_company = %s
                        """,
                        (opening, company),
                    )
                    updated += cur.rowcount

                    # Update progress every 10 rows
                    if idx % 10 == 0:
                        progress = 20 + int((idx / total_rows) * 70)
                        _update_progress(
                            task_id,
                            "step1",
                            progress,
                            100,
                            f"Processing row {idx + 1}/{total_rows}...",
                            "running",
                        )

                    if updated % 100 == 0:
                        conn.commit()

                conn.commit()
        finally:
            conn.close()

        _update_progress(
            task_id,
            "step1",
            100,
            100,
            f"Completed! Updated {updated} records, skipped {skipped}",
            "completed",
        )

        # Store stats
        _progress[task_id]["stats"] = {
            "total_rows": total_rows,
            "updated": updated,
            "skipped": skipped,
        }

    except Exception as e:
        logger.error(f"Step 1 error: {e}")
        _update_progress(task_id, "step1", 0, 100, f"Error: {str(e)}", "error")


class Step2Request(BaseModel):
    """Step 2 request - configurable thresholds"""

    usage_threshold: int = 50
    opening_threshold: int = 70
    answer_threshold: int = 70


class Step2Response(BaseModel):
    """Step 2 response"""

    success: bool
    task_id: str
    message: str
    stats: Optional[Dict[str, Any]] = None


@router.post("/step2/start")
async def start_step2(request: Step2Request):
    """
    Start Step 2: Analyze calls and compute match scores

    This runs analyze_calls.py logic in background with the specified thresholds.
    """
    import uuid

    task_id = str(uuid.uuid4())

    # Start background task
    asyncio.create_task(
        _run_step2(
            task_id,
            request.usage_threshold,
            request.opening_threshold,
            request.answer_threshold,
        )
    )

    return Step2Response(
        success=True,
        task_id=task_id,
        message="Step 2 started - Analyzing call match scores",
    )


async def _run_step2(
    task_id: str, usage_threshold: int, opening_threshold: int, answer_threshold: int
):
    """Run Step 2 in background"""
    try:
        _update_progress(
            task_id, "step2", 0, 100, "Loading configuration...", "running"
        )

        # Import the analyze_calls module
        sys.path.insert(0, str(CALL_MATCH_PACKAGE_PATH))
        import importlib

        # Force reimport config and analyze_calls
        if "config" in sys.modules:
            importlib.reload(sys.modules["config"])
        config = importlib.import_module("config")

        # Override thresholds
        config.USAGE_THRESHOLD = usage_threshold
        config.OPENING_THRESHOLD = opening_threshold
        config.ANSWER_THRESHOLD = answer_threshold

        _update_progress(
            task_id, "step2", 5, 100, "Connecting to database...", "running"
        )

        from datetime import date

        import pymysql

        # Get database connection
        cfg = config.DB_CONFIG.copy()
        cfg["cursorclass"] = pymysql.cursors.DictCursor

        DATE_CUTOFF = date(2026, 2, 2)
        MIN_TALKING_TIME = 45

        # Fetch records
        conn = pymysql.connect(**cfg)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, call_opening, transcription, date, talking_time, real_person
                    FROM sdr_cdr_inout_data
                    WHERE transcription IS NOT NULL
                      AND date >= %s
                      AND (talking_time IS NULL OR talking_time >= %s)
                    """,
                    (DATE_CUTOFF, MIN_TALKING_TIME),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        total_records = len(rows)
        _update_progress(
            task_id,
            "step2",
            10,
            100,
            f"Found {total_records} records to analyze",
            "running",
        )

        if total_records == 0:
            _update_progress(
                task_id, "step2", 100, 100, "No records found to analyze", "completed"
            )
            _progress[task_id]["stats"] = {
                "total_records": 0,
                "processed": 0,
                "skipped_not_real": 0,
            }
            return

        # Import analyze_calls functions
        if "analyze_calls" in sys.modules:
            importlib.reload(sys.modules["analyze_calls"])
        analyze_calls = importlib.import_module("analyze_calls")

        # Load Q&A config
        qas = analyze_calls._load_questions_answers()

        thresholds = {
            "usage": float(usage_threshold),
            "opening": float(opening_threshold),
            "answer": float(answer_threshold),
        }

        results = []
        skipped_not_real = 0

        for idx, row in enumerate(rows):
            # Create CallRecord
            rec = analyze_calls.CallRecord(
                id=row["id"],
                call_opening=row.get("call_opening"),
                transcription=row["transcription"],
                date=row.get("date"),
                talking_time=row.get("talking_time"),
                real_person=row.get("real_person"),
            )

            # Update progress
            progress = 10 + int((idx / total_records) * 80)
            _update_progress(
                task_id,
                "step2",
                progress,
                100,
                f"Analyzing record {idx + 1}/{total_records}...",
                "running",
            )

            # Skip if not real person
            if rec.real_person is False:
                skipped_not_real += 1
                continue

            # Process record
            try:
                res = await analyze_calls._process_record(rec, qas, thresholds)
                results.append(res)
            except Exception as e:
                logger.warning(f"Error processing record {rec.id}: {e}")
                continue

        # Write back results
        _update_progress(
            task_id, "step2", 90, 100, "Writing results to database...", "running"
        )

        await asyncio.to_thread(analyze_calls._write_back, results)

        # Collect detailed stats
        newly_processed = sum(1 for r in results if r.get("processed"))
        skipped_already_complete = sum(
            1 for r in results if r.get("skipped_incremental")
        )
        skipped_parse_error = sum(
            1
            for r in results
            if not r.get("processed") and not r.get("skipped_incremental")
        )

        # Count newly calculated fields
        opening_calculated = sum(
            1
            for r in results
            if r.get("processed") and "opening" in r.get("calculated_fields", [])
        )
        question_calculated = sum(
            1
            for r in results
            if r.get("processed") and "question" in r.get("calculated_fields", [])
        )
        answer_calculated = sum(
            1
            for r in results
            if r.get("processed") and "answer" in r.get("calculated_fields", [])
        )

        _update_progress(
            task_id,
            "step2",
            100,
            100,
            f"Completed! Processed {newly_processed} new records, skipped {skipped_already_complete} already complete",
            "completed",
        )

        _progress[task_id]["stats"] = {
            "total_records": total_records,
            "processed": newly_processed,
            "skipped_already_complete": skipped_already_complete,
            "skipped_parse_error": skipped_parse_error,
            "skipped_not_real": skipped_not_real,
            "opening_calculated": opening_calculated,
            "question_calculated": question_calculated,
            "answer_calculated": answer_calculated,
            "tokens_saved": skipped_already_complete * 100,  # Rough estimate
        }

    except Exception as e:
        logger.error(f"Step 2 error: {e}", exc_info=True)
        _update_progress(task_id, "step2", 0, 100, f"Error: {str(e)}", "error")


@router.websocket("/progress/ws/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()

    try:
        last_progress = None
        while True:
            current = _progress.get(task_id)

            if current != last_progress:
                await websocket.send_json(current or {"status": "not_found"})
                last_progress = current

                if current and current.get("status") in ["completed", "error"]:
                    break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
