from __future__ import annotations

from uuid import uuid4


class FactorApplication:
    """Application facade; the legacy engine migrates behind these ports."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}

    def create_mining_job(self, payload: dict) -> dict:
        job_id = uuid4().hex
        result = {"job_id": job_id, "status": "PENDING", "request": payload}
        self._jobs[job_id] = result
        return result

    def get_mining_job(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def cancel_mining_job(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job["status"] == "PENDING":
            job["status"] = "CANCELLED"
        return {"job_id": job_id, "cancelled": job["status"] == "CANCELLED", "status": job["status"]}
