from __future__ import annotations

import os
import queue
import json
import shutil
import smtplib
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


BASE_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(BASE_DIR / ".env")


DEFAULT_DATA_ROOT = BASE_DIR if os.name == "nt" else Path(os.getenv("MODEL_DATA_ROOT", str(Path.home() / ".local/share/amb82-model-convert")))
DATA_ROOT = Path(os.getenv("MODEL_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

JOB_ROOT = DATA_ROOT / "webui_jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)
SERVICE_STATE_PATH = DATA_ROOT / "service_state.json"

WSL_DISTRO = os.getenv("MODEL_WSL_DISTRO", "AMB_Model").strip() or "AMB_Model"
WEBUI_HOST = os.getenv("MODEL_WEBUI_HOST", "127.0.0.1")
WEBUI_PORT = int(os.getenv("MODEL_WEBUI_PORT", "8891"))
WEBUI_BASE_URL = os.getenv("MODEL_WEBUI_BASE_URL", f"http://{WEBUI_HOST}:{WEBUI_PORT}").rstrip("/")
ACUITY_OUTPUT_NB_FILENAME = "network_binary.nb"
PUBLIC_OUTPUT_NB_FILENAME = "imgclassification.nb"
YOLO_PUBLIC_OUTPUT_NB_FILENAME = "yolov4_tiny.nb"
SITE_ASSET_DIR = BASE_DIR / "site_assets"
FAVICON_PATHS = {
    "ico": SITE_ASSET_DIR / "favicon.ico",
    "16": SITE_ASSET_DIR / "favicon-16x16.png",
    "32": SITE_ASSET_DIR / "favicon-32x32.png",
    "apple": SITE_ASSET_DIR / "apple-touch-icon.png",
}
SERVICE_ICON_PATHS = {
    "mqttgo": BASE_DIR / "service_icons" / "mqttgo_thumb.png",
    "mqttgovip": BASE_DIR / "service_icons" / "mqttgovip_thumb.png",
    "nmking": BASE_DIR / "service_icons" / "nmking.jpg",
}
EXAMPLE_DOWNLOAD_PATHS = {
    "arduino_imgclassification": BASE_DIR / "example_downloads" / "ameba_imgclassification_final.zip",
}


@dataclass
class MailSettings:
    host: str = os.getenv("SMTP_HOST", "").strip()
    port: int = int(os.getenv("SMTP_PORT", "587"))
    username: str = os.getenv("SMTP_USERNAME", "").strip()
    password: str = os.getenv("SMTP_PASSWORD", "").strip()
    from_email: str = os.getenv("SMTP_FROM_EMAIL", "").strip()
    use_tls: bool = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
    use_ssl: bool = os.getenv("SMTP_USE_SSL", "false").strip().lower() in {"1", "true", "yes", "on"}

    def enabled(self) -> bool:
        return bool(self.host and self.from_email)


@dataclass
class JobRecord:
    job_id: str
    filename: str
    email: str
    created_at: str
    model_type: str = "teachable"
    status: str = "queued"
    message: str = "等待開始"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    elapsed_seconds: float | None = None
    log_path: str | None = None
    output_path: str | None = None
    work_dir: str | None = None
    calibration_count: int = 0
    queue_position: int | None = None
    notification_status: str = "pending"
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "email": self.email,
            "created_at": self.created_at,
            "model_type": self.model_type,
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "elapsed_seconds": self.elapsed_seconds,
            "log_path": self.log_path,
            "output_path": self.output_path,
            "work_dir": self.work_dir,
            "calibration_count": self.calibration_count,
            "queue_position": self.queue_position,
            "notification_status": self.notification_status,
        }


app = FastAPI(title="Model Convert WebUI", version="0.1.0")
app.state.jobs: dict[str, JobRecord] = {}
app.state.job_queue: queue.Queue[str] = queue.Queue()
app.state.captcha_store: dict[str, str] = {}
app.state.metrics_lock = threading.Lock()
app.state.metrics = {"total_completed_count": 0}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def bootstrap_total_completed_count() -> int:
    return sum(1 for path in JOB_ROOT.glob("*/work/out_nbg_unify/network_binary.nb") if path.exists())


def load_service_state() -> dict[str, int]:
    if SERVICE_STATE_PATH.exists():
        try:
            data = json.loads(SERVICE_STATE_PATH.read_text(encoding="utf-8"))
            total = int(data.get("total_completed_count", 0))
            return {"total_completed_count": max(total, 0)}
        except Exception:
            pass

    data = {"total_completed_count": bootstrap_total_completed_count()}
    save_service_state(data)
    return data


def save_service_state(data: dict[str, int]) -> None:
    SERVICE_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def increment_total_completed_count() -> None:
    with app.state.metrics_lock:
        app.state.metrics["total_completed_count"] = int(app.state.metrics.get("total_completed_count", 0)) + 1
        save_service_state(app.state.metrics)


def create_captcha() -> tuple[str, str]:
    captcha_id = uuid.uuid4().hex[:12]
    code = str(uuid.uuid4().int % 9000 + 1000)
    app.state.captcha_store[captcha_id] = code
    return captcha_id, code


def build_download_url(job_id: str) -> str:
    return f"{WEBUI_BASE_URL}/api/jobs/{job_id}/download"


def acuity_output_path(work_dir: Path) -> Path:
    return work_dir / "out_nbg_unify" / ACUITY_OUTPUT_NB_FILENAME


def public_output_path(work_dir: Path) -> Path:
    return work_dir / "out_nbg_unify" / PUBLIC_OUTPUT_NB_FILENAME


def public_output_path_for_model(work_dir: Path, model_type: str) -> Path:
    if model_type == "yolo_darknet":
        return work_dir / "out_nbg_unify" / YOLO_PUBLIC_OUTPUT_NB_FILENAME
    return public_output_path(work_dir)


def public_output_filename_for_model(model_type: str) -> str:
    return YOLO_PUBLIC_OUTPUT_NB_FILENAME if model_type == "yolo_darknet" else PUBLIC_OUTPUT_NB_FILENAME


def build_received_mail_subject(job: JobRecord) -> str:
    return f"[MQTTGO] 已收到模型轉換工作 - {job.job_id}"


def build_received_mail_text_body(job: JobRecord) -> str:
    model_label = "Teachable Machine" if job.model_type == "teachable" else "YOLO Darknet"
    return (
        "您好，\n\n"
        "我們已收到您的模型轉換工作，系統將依序排隊處理。\n\n"
        f"工作編號：{job.job_id}\n"
        f"模型類型：{model_label}\n"
        f"上傳檔案：{job.filename}\n"
        f"收到時間：{job.created_at}\n"
        f"校正圖片：{job.calibration_count} 張\n\n"
        "目前工作已建立，請等候轉換完成後，我們會再寄送結果通知與下載連結。\n"
    )


def build_received_mail_html_body(job: JobRecord) -> str:
    model_label = "Teachable Machine" if job.model_type == "teachable" else "YOLO Darknet"
    return f"""\
<!doctype html>
<html lang="zh-Hant">
<body style="margin:0;padding:24px;background:#f6f6f6;font-family:Segoe UI,Microsoft JhengHei,sans-serif;color:#2b2b2b;">
  <div style="max-width:720px;margin:0 auto;background:#ffffff;border:1px solid #d7d7d7;">
    <div style="background:#f5bf2c;padding:18px 22px;font-size:24px;font-weight:700;">NMKING小霸王實驗室</div>
    <div style="padding:24px 22px;">
      <h2 style="margin:0 0 14px;font-size:26px;">已收到您的模型轉換工作。</h2>
      <p style="margin:0 0 10px;">工作編號：{job.job_id}</p>
      <p style="margin:0 0 10px;">模型類型：{model_label}</p>
      <p style="margin:0 0 10px;">上傳檔案：{job.filename}</p>
      <p style="margin:0 0 10px;">收到時間：{job.created_at}</p>
      <p style="margin:0 0 18px;">校正圖片：{job.calibration_count} 張</p>
      <p style="margin:0 0 18px;">目前工作已建立，請等候轉換完成後，我們會再寄送結果通知與下載連結。</p>
    </div>
  </div>
</body>
</html>
"""


def build_mail_subject(job: JobRecord) -> str:
    return f"[MQTTGO] 模型轉換{'完成' if job.status == 'completed' else '失敗'} - {job.job_id}"


def build_mail_text_body(job: JobRecord) -> str:
    links = (
        "\n其他服務：\n"
        "mqttgo.io：https://mqttgo.io\n"
        "mqttgo.vip：https://mqttgo.vip\n"
        "nmking.io：https://www.nmking.io\n"
        "twgo.io：https://twgo.io\n"
    )
    if job.status == "completed":
        output_name = public_output_filename_for_model(job.model_type)
        return (
            "您好，\n\n"
            "您的模型轉換已完成。\n\n"
            f"工作編號：{job.job_id}\n"
            f"上傳檔案：{job.filename}\n"
            f"完成時間：{job.finished_at}\n"
            f"耗時：約 {job.elapsed_seconds} 秒\n"
            f"下載連結：{build_download_url(job.job_id)}\n\n"
            f"請使用上方連結下載 {output_name}。\n"
            f"{links}"
        )
    return (
        "您好，\n\n"
        "您的模型轉換未能完成。\n\n"
        f"工作編號：{job.job_id}\n"
        f"上傳檔案：{job.filename}\n"
        f"失敗時間：{job.finished_at}\n"
        "系統狀態：轉換失敗，請查看系統記錄。\n\n"
        "請稍後重新上傳，或聯繫管理員協助查看後端記錄。\n"
        f"{links}"
    )


def build_mail_html_body(job: JobRecord) -> str:
    output_name = public_output_filename_for_model(job.model_type)
    status_line = (
        f"""
        <p style="margin:0 0 10px;">下載連結：
          <a href="{build_download_url(job.job_id)}" style="color:#2563eb;">{build_download_url(job.job_id)}</a>
        </p>
        <p style="margin:0 0 18px;">請使用上方連結下載 <code>{output_name}</code>。</p>
        """
        if job.status == "completed"
        else """
        <p style="margin:0 0 18px;">系統狀態：轉換失敗，請查看系統記錄。</p>
        <p style="margin:0 0 18px;">請稍後重新上傳，或聯繫管理員協助查看後端記錄。</p>
        """
    )
    title = "您的模型轉換已完成。" if job.status == "completed" else "您的模型轉換未能完成。"
    time_label = "完成時間" if job.status == "completed" else "失敗時間"
    elapsed_html = f'<p style="margin:0 0 10px;">耗時：約 {job.elapsed_seconds} 秒</p>' if job.status == "completed" else ""
    return f"""\
<!doctype html>
<html lang="zh-Hant">
<body style="margin:0;padding:24px;background:#f6f6f6;font-family:Segoe UI,Microsoft JhengHei,sans-serif;color:#2b2b2b;">
  <div style="max-width:720px;margin:0 auto;background:#ffffff;border:1px solid #d7d7d7;">
    <div style="background:#f5bf2c;padding:18px 22px;font-size:24px;font-weight:700;">MQTTGO</div>
    <div style="padding:24px 22px;">
      <h2 style="margin:0 0 14px;font-size:26px;">{title}</h2>
      <p style="margin:0 0 10px;">工作編號：{job.job_id}</p>
      <p style="margin:0 0 10px;">上傳檔案：{job.filename}</p>
      <p style="margin:0 0 10px;">{time_label}：{job.finished_at}</p>
      {elapsed_html}
      {status_line}
      <div style="margin-top:22px;padding-top:18px;border-top:1px solid #e5e7eb;">
        <p style="margin:0 0 14px;font-weight:700;">其他服務</p>
        <table role="presentation" cellpadding="0" cellspacing="0" border="0" style="width:100%;">
          <tr>
            <td style="padding:6px;">
              <a href="https://mqttgo.io" style="display:block;border:1px solid #d1d5db;background:#fafafa;padding:12px;text-decoration:none;color:#111827;">
                <div style="width:56px;height:56px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;margin:0 0 10px 0;overflow:hidden;">
                  <img src="cid:mqttgo_icon" alt="mqttgo.io" style="width:56px;height:56px;display:block;object-fit:contain;">
                </div>
                <div style="font-weight:700;margin-bottom:4px;">mqttgo.io</div>
                <div style="font-size:12px;color:#6b7280;">免費匿名的 mqtt 服務</div>
              </a>
            </td>
            <td style="padding:6px;">
              <a href="https://mqttgo.vip" style="display:block;border:1px solid #d1d5db;background:#fafafa;padding:12px;text-decoration:none;color:#111827;">
                <div style="width:56px;height:56px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;margin:0 0 10px 0;overflow:hidden;">
                  <img src="cid:mqttgovip_icon" alt="mqttgo.vip" style="width:56px;height:56px;display:block;object-fit:contain;">
                </div>
                <div style="font-weight:700;margin-bottom:4px;">mqttgo.vip</div>
                <div style="font-size:12px;color:#6b7280;">專業的 mqtt 服務</div>
              </a>
            </td>
          </tr>
          <tr>
            <td style="padding:6px;">
              <a href="https://www.nmking.io" style="display:block;border:1px solid #d1d5db;background:#fafafa;padding:12px;text-decoration:none;color:#111827;">
                <div style="width:56px;height:56px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;margin:0 0 10px 0;overflow:hidden;">
                  <img src="cid:nmking_icon" alt="nmking.io" style="width:56px;height:56px;display:block;object-fit:contain;">
                </div>
                <div style="font-weight:700;margin-bottom:4px;">nmking.io</div>
                <div style="font-size:12px;color:#6b7280;">教學網站</div>
              </a>
            </td>
            <td style="padding:6px;">
              <a href="https://twgo.io" style="display:block;border:1px solid #d1d5db;background:#fafafa;padding:12px;text-decoration:none;color:#111827;">
                <div style="width:56px;height:56px;border-radius:999px;background:#E62457;color:#ffffff;font-weight:700;display:flex;align-items:center;justify-content:center;margin:0 0 10px 0;font-size:20px;">T</div>
                <div style="font-weight:700;margin-bottom:4px;">twgo.io</div>
                <div style="font-size:12px;color:#6b7280;">簡單免費轉址服務</div>
              </a>
            </td>
          </tr>
        </table>
      </div>
    </div>
  </div>
</body>
</html>
"""


def _add_service_icons_to_html_part(message: EmailMessage) -> None:
    html_part = message.get_payload()[-1]
    for cid, icon_name in [
        ("mqttgo_icon", "mqttgo"),
        ("mqttgovip_icon", "mqttgovip"),
        ("nmking_icon", "nmking"),
    ]:
        path = SERVICE_ICON_PATHS.get(icon_name)
        if not path or not path.exists():
            continue
        suffix = path.suffix.lower()
        subtype = "png" if suffix == ".png" else "jpeg"
        html_part.add_related(path.read_bytes(), maintype="image", subtype=subtype, cid=f"<{cid}>")


def send_email_message(subject: str, to_email: str, text_body: str, html_body: str, include_service_icons: bool = False) -> None:
    settings = MailSettings()
    if not settings.enabled():
        raise RuntimeError("mail_not_configured")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr(("NMKING小霸王實驗室", settings.from_email))
    message["To"] = to_email
    message.set_content(text_body, charset="utf-8")
    message.add_alternative(html_body, subtype="html", charset="utf-8")
    if include_service_icons:
        _add_service_icons_to_html_part(message)

    if settings.use_ssl:
        with smtplib.SMTP_SSL(settings.host, settings.port, timeout=30) as server:
            if settings.username:
                server.login(settings.username, settings.password)
            server.send_message(message)
    else:
        with smtplib.SMTP(settings.host, settings.port, timeout=30) as server:
            if settings.use_tls:
                server.starttls()
            if settings.username:
                server.login(settings.username, settings.password)
            server.send_message(message)


def send_received_email(job: JobRecord) -> None:
    send_email_message(
        build_received_mail_subject(job),
        job.email,
        build_received_mail_text_body(job),
        build_received_mail_html_body(job),
        include_service_icons=False,
    )


def send_job_email(job: JobRecord) -> None:
    settings = MailSettings()
    if not settings.enabled():
        job.notification_status = "mail_not_configured"
        return

    send_email_message(
        build_mail_subject(job),
        job.email,
        build_mail_text_body(job),
        build_mail_html_body(job),
        include_service_icons=True,
    )

    job.notification_status = "sent"


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = str(resolved).replace("\\", "/").split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_wsl_command(zip_path: Path, work_dir: Path, calibration_dir: Path) -> list[str]:
    if os.name == "nt":
        script_path = to_wsl_path(BASE_DIR / "zip_to_nb_wsl.sh")
        zip_wsl = to_wsl_path(zip_path)
        work_wsl = to_wsl_path(work_dir)
        calib_wsl = to_wsl_path(calibration_dir)
        command = f"bash {shell_quote(script_path)} {shell_quote(zip_wsl)} {shell_quote(work_wsl)} {shell_quote(calib_wsl)}"
        return ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", command]

    return [
        "bash",
        str((BASE_DIR / "zip_to_nb_wsl.sh").resolve()),
        str(zip_path.resolve()),
        str(work_dir.resolve()),
        str(calibration_dir.resolve()),
    ]


def build_yolo_darknet_stub_command(work_dir: Path) -> list[str]:
    input_dir = work_dir.parent / "input"
    cfg_path = input_dir / "model.cfg"
    weights_path = input_dir / "model.weights"
    if os.name == "nt":
        script_path = to_wsl_path(BASE_DIR / "darknet_to_nb_wsl.sh")
        cfg_wsl = to_wsl_path(cfg_path)
        weights_wsl = to_wsl_path(weights_path)
        work_wsl = to_wsl_path(work_dir)
        calib_wsl = to_wsl_path(work_dir.parent / "calibration")
        command = f"bash {shell_quote(script_path)} {shell_quote(cfg_wsl)} {shell_quote(weights_wsl)} {shell_quote(work_wsl)} {shell_quote(calib_wsl)}"
        return ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", command]

    return [
        "bash",
        str((BASE_DIR / "darknet_to_nb_wsl.sh").resolve()),
        str(cfg_path.resolve()),
        str(weights_path.resolve()),
        str(work_dir.resolve()),
        str((work_dir.parent / "calibration").resolve()),
    ]


def run_job(job: JobRecord, zip_path: Path, work_dir: Path, calibration_dir: Path, log_path: Path, output_path: Path) -> None:
    if job.model_type == "teachable":
        command = build_wsl_command(zip_path, work_dir, calibration_dir)
    else:
        command = build_yolo_darknet_stub_command(work_dir)
    generated_output_path = acuity_output_path(work_dir)
    started = datetime.now()
    with job.lock:
        job.status = "running"
        job.message = "WSL 轉換中" if job.model_type == "teachable" else "YOLO Darknet 流程檢查中"
        job.started_at = now_text()
        job.queue_position = None

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[{now_text()}] Starting job {job.job_id}\n")
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            text=True,
        )
        return_code = process.wait()

    elapsed = round((datetime.now() - started).total_seconds(), 2)
    with job.lock:
        job.return_code = return_code
        job.finished_at = now_text()
        job.elapsed_seconds = elapsed
        if job.model_type == "teachable" and return_code == 0 and generated_output_path.exists():
            if generated_output_path != output_path:
                shutil.copyfile(generated_output_path, output_path)
            job.status = "completed"
            job.message = "轉換完成"
            job.output_path = str(output_path)
            job.notification_status = "ready_to_notify_success"
            increment_total_completed_count()
        elif job.model_type == "yolo_darknet" and return_code == 0:
            yolo_output_path = public_output_path_for_model(work_dir, job.model_type)
            if yolo_output_path.exists():
                job.status = "completed"
                job.message = "轉換完成"
                job.output_path = str(yolo_output_path)
                job.notification_status = "ready_to_notify_success"
                increment_total_completed_count()
            else:
                job.status = "failed"
                job.message = "YOLO Darknet 轉換失敗，請查看 log"
                job.notification_status = "ready_to_notify_failed"
        else:
            job.status = "failed"
            job.message = "轉換失敗，請查看 log"
            job.notification_status = "ready_to_notify_failed"

    try:
        send_job_email(job)
    except Exception:
        job.notification_status = "mail_send_failed"


def refresh_queue_positions() -> None:
    pending_ids = list(app.state.job_queue.queue)
    for job in app.state.jobs.values():
        if job.status == "queued":
            try:
                job.queue_position = pending_ids.index(job.job_id) + 1
                job.message = f"排隊中，第 {job.queue_position} 位"
            except ValueError:
                job.queue_position = None


def service_summary() -> dict[str, object]:
    jobs = list(app.state.jobs.values())
    running = [job for job in jobs if job.status == "running"]
    queued = [job for job in jobs if job.status == "queued"]
    completed = [job for job in jobs if job.status == "completed"]
    failed = [job for job in jobs if job.status == "failed"]
    return {
        "status_label": "轉換中" if running else "閒置",
        "queue_count": len(queued),
        "queue_job_ids": [job.job_id for job in queued],
        "running_count": len(running),
        "running_job_ids": [job.job_id for job in running],
        "completed_count": len(completed),
        "total_completed_count": int(app.state.metrics.get("total_completed_count", 0)),
        "failed_count": len(failed),
    }


def worker_loop() -> None:
    while True:
        job_id = app.state.job_queue.get()
        job = app.state.jobs.get(job_id)
        if not job:
            app.state.job_queue.task_done()
            continue

        job_dir = JOB_ROOT / job_id
        zip_path = job_dir / "input" / "converted_keras.zip"
        work_dir = job_dir / "work"
        calibration_dir = job_dir / "calibration"
        log_path = job_dir / "job.log"
        output_path = public_output_path_for_model(work_dir, job.model_type)

        refresh_queue_positions()
        run_job(job, zip_path, work_dir, calibration_dir, log_path, output_path)
        app.state.job_queue.task_done()
        refresh_queue_positions()


def _send_received_email_background(job: JobRecord) -> None:
    try:
        send_received_email(job)
    except Exception:
        pass


def html_page() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>8735(AMB82) Teachable Machine 模型轉換</title>
  <link rel="icon" href="/favicon.ico" sizes="any">
  <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
  <style>
    :root {
      --bg: #f6f6f6;
      --header: #f5bf2c;
      --panel: #ffffff;
      --ink: #2b2b2b;
      --muted: #737373;
      --line: #d7d7d7;
      --accent: #3b82f6;
      --accent-2: #6b7280;
      --ok: #2f8a58;
      --warn: #cb7a12;
      --bad: #b83a32;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    #header {
      height: 80px;
      width: 100%;
      background: linear-gradient(to bottom, var(--header) 0%, var(--header) 100%);
      border-bottom: 1px solid rgba(0,0,0,0.08);
    }
    #header > div {
      max-width: 1100px;
      margin: 0 auto;
      height: 100%;
      padding: 0 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      font-weight: 700;
      font-size: 24px;
    }
    .brand-mark {
      width: 44px;
      height: 44px;
      border-radius: 10px;
      background: rgba(255,255,255,0.9);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
    }
    .header-note {
      color: rgba(0,0,0,0.72);
      font-size: 14px;
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 22px 20px 36px;
      display: grid;
      gap: 20px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 0;
      padding: 18px;
      box-shadow: none;
    }
    h1, h2, h3, p { margin: 0; }
    .hero {
      display: grid;
      gap: 8px;
      border-left: 4px solid var(--header);
    }
    .hero h1 {
      font-size: 26px;
      font-weight: 700;
    }
    .hero p {
      color: var(--muted);
      line-height: 1.7;
    }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }
    .section-title h2 {
      font-size: 23px;
      font-weight: 700;
    }
    form { display: grid; gap: 14px; }
    input[type=email] {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid #cccccc;
      border-radius: 0;
      background: #fff;
      font: inherit;
    }
    select {
      width: 100%;
      min-height: 52px;
      padding: 12px 44px 12px 14px;
      border: 1px solid #b8bec7;
      border-radius: 0;
      background: #f8fafc;
      color: #111827;
      font: inherit;
      font-size: 16px;
      line-height: 1.4;
      appearance: auto;
    }
    select:focus,
    input[type=email]:focus,
    input[type=text]:focus {
      outline: 2px solid #bfdbfe;
      outline-offset: 0;
      border-color: #93c5fd;
    }
    .dropzone {
      border: 1px dashed #bcbcbc;
      background: #fafafa;
      padding: 14px;
      display: grid;
      gap: 8px;
      cursor: pointer;
    }
    .dropzone strong {
      font-size: 15px;
    }
    .dropzone span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .selected-file {
      display: none;
      margin-top: 10px;
      padding: 12px 14px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: #1e3a8a;
    }
    .selected-file strong {
      display: block;
      font-size: 12px;
      margin-bottom: 4px;
      color: #1d4ed8;
    }
    .selected-file code {
      font-size: 16px;
      font-weight: 700;
      color: #1e40af;
      word-break: break-all;
    }
    .captcha-box {
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 12px;
      align-items: stretch;
    }
    .captcha-code {
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #d1d5db;
      background: #111827;
      color: #f9fafb;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: 0.18em;
      min-height: 48px;
      user-select: none;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(88px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .preview-item {
      border: 1px solid #d1d5db;
      background: #fff;
      padding: 6px;
      display: grid;
      gap: 6px;
    }
    .preview-item img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
      background: #f3f4f6;
    }
    .preview-item span {
      font-size: 12px;
      color: var(--muted);
      word-break: break-word;
      line-height: 1.4;
    }
    button {
      border: 0;
      border-radius: 0;
      padding: 12px 16px;
      font: inherit;
      font-weight: 600;
      color: white;
      background: var(--accent);
      cursor: pointer;
    }
    button:disabled {
      background: #9ca3af;
      cursor: not-allowed;
      opacity: 0.8;
    }
    button.secondary { background: var(--accent-2); }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }
    .status-board {
      display: grid;
      gap: 12px;
    }
    .status-main {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px;
      border: 1px solid #d1d5db;
      background: #fafafa;
    }
    .status-main strong {
      font-size: 20px;
      color: #111827;
    }
    .status-resting strong {
      color: #374151;
    }
    .status-busy strong {
      color: #b45309;
    }
    .status-note {
      color: var(--muted);
      font-size: 13px;
    }
    .status-completed { color: var(--ok); font-weight: 700; }
    .status-running { color: var(--warn); font-weight: 700; }
    .status-failed { color: var(--bad); font-weight: 700; }
    .status-queued { color: var(--accent-2); font-weight: 700; }
    .empty {
      color: var(--muted);
      padding: 14px;
      border: 1px dashed var(--line);
      background: #fafafa;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      font-size: 12px;
      border-radius: 999px;
      background: #f3f4f6;
      color: #374151;
      border: 1px solid #e5e7eb;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .form-message {
      display: none;
      padding: 12px 14px;
      border: 1px solid #fecaca;
      background: #fef2f2;
      color: #991b1b;
      font-size: 14px;
      line-height: 1.5;
    }
    .form-message.show {
      display: block;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .queue-list {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid #e5e7eb;
    }
    .queue-list strong {
      display: block;
      margin-bottom: 10px;
      color: #111827;
      font-size: 14px;
    }
    .queue-job-ids {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .queue-job-id {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #dbeafe;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .queue-empty {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .service-links {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }
    .service-link {
      position: relative;
      display: block;
      padding: 16px;
      padding-top: 88px;
      min-height: 168px;
      border: 1px solid #d1d5db;
      background: #fafafa;
      color: inherit;
      text-decoration: none;
    }
    .service-link:hover {
      background: #f3f4f6;
      text-decoration: none;
    }
    .service-icon {
      position: absolute;
      top: 16px;
      left: 16px;
      width: 64px;
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 12px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      color: #f9fafb;
      font-size: 20px;
      font-weight: 700;
      overflow: hidden;
    }
    .service-icon img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .service-icon.twgo {
      border-radius: 50%;
      background: #E62457;
      border: 0;
      font-size: 1.1rem;
    }
    .service-link strong {
      display: block;
      font-size: 16px;
      color: #111827;
      margin-bottom: 8px;
    }
    .service-link span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (max-width: 880px) {
      .grid { grid-template-columns: 1fr; }
      .meta { grid-template-columns: 1fr; }
      .service-links { grid-template-columns: 1fr; }
      #header > div { padding: 0 14px; }
      .brand { font-size: 20px; }
      .header-note { display: none; }
    }
  </style>
</head>
<body>
  <div id="header">
    <div>
      <div class="brand">
        <div class="brand-mark">M</div>
        <div>MQTTGO</div>
      </div>
    </div>
  </div>
  <main>
    <section class="card hero">
      <h1>8735(AMB82) 模型轉換 <a href="https://github.com/youjunjer/8735-AMB82--model-convert" target="_blank" rel="noreferrer">(Github repo)</a></h1>
      <p>目前提供 <strong>Teachable Machine</strong> 與 <strong>YOLO Darknet</strong> 兩個模型類型選項。</p>
      <p id="hero-guide-1">1. 請先到 <a href="https://teachablemachine.withgoogle.com/" target="_blank" rel="noreferrer">Google Teachable Machine網站</a> 建立並訓練自己的模型，完成後匯出模型檔 <code>converted_keras.zip</code>。</p>
      <p id="hero-guide-2">2. 上傳 <code>converted_keras.zip</code> 與至少一張校正圖片後，系統呼叫轉換流程，產出 <code>imgclassification.nb</code> 的下載路徑，回復到指定的 Mail。</p>
    </section>
    <section class="grid">
      <section class="card">
        <div class="section-title">
          <h2>建立轉換工作</h2>
          <span class="pill">單一 worker 排隊</span>
        </div>
        <p id="form-hint" class="hint">目前要求填 email、模型 zip 與至少一張校正圖片。轉換完成後的 email 通知流程先保留欄位，寄信設定下一步再接。</p>
        <form id="upload-form" style="margin-top:14px;" novalidate>
          <select id="model-type" name="model_type" required>
            <option value="" selected>請選擇</option>
            <option value="teachable">Teachable Machine</option>
            <option value="yolo_darknet">YOLO Darknet</option>
          </select>
          <input id="email" type="email" name="email" placeholder="請輸入通知 email" required>
          <div id="teachable-fields">
            <label class="dropzone" for="zip-file">
              <strong>模型壓縮檔</strong>
              <span id="zip-help-text">請選取 <code>converted_keras.zip</code></span>
              <input id="zip-file" type="file" name="file" accept=".zip" style="display:none;">
              <span id="zip-file-name">尚未選擇檔案</span>
              <div id="selected-zip-file" class="selected-file">
                <strong>目前選擇的模型</strong>
                <code id="selected-zip-text"></code>
              </div>
            </label>
          </div>
          <div id="yolo-darknet-fields" style="display:none;">
            <label class="dropzone" for="yolo-cfg-file">
              <strong>YOLO 設定檔</strong>
              <span>請選取 <code>.cfg</code></span>
              <input id="yolo-cfg-file" type="file" name="yolo_cfg_file" accept=".cfg" style="display:none;">
              <span id="yolo-cfg-file-name">尚未選擇檔案</span>
            </label>
            <label class="dropzone" for="yolo-weights-file">
              <strong>YOLO 權重檔</strong>
              <span>請選取 <code>.weights</code></span>
              <input id="yolo-weights-file" type="file" name="yolo_weights_file" accept=".weights" style="display:none;">
              <span id="yolo-weights-file-name">尚未選擇檔案</span>
            </label>
            <label class="dropzone" for="yolo-classes-file">
              <strong>YOLO 類別名稱</strong>
              <span>可選，請選取 <code>classes.txt</code></span>
              <input id="yolo-classes-file" type="file" name="yolo_classes_file" accept=".txt" style="display:none;">
              <span id="yolo-classes-file-name">尚未選擇檔案</span>
            </label>
          </div>
          <label class="dropzone" for="calibration-files">
            <strong>校正圖片</strong>
            <span>至少 1 張，可多選 jpg / jpeg / png</span>
            <input id="calibration-files" type="file" name="calibration_files" accept=".jpg,.jpeg,.png" multiple style="display:none;">
            <span id="calibration-file-name">尚未選擇檔案</span>
          </label>
          <div id="calibration-preview" class="preview-grid" style="display:none;"></div>
          <div class="captcha-box">
            <div id="captcha-code" class="captcha-code">----</div>
            <input id="captcha-input" type="text" inputmode="numeric" pattern="[0-9]*" maxlength="4" placeholder="請輸入數字驗證碼" required>
          </div>
          <button type="submit" disabled>開始轉換</button>
          <div id="submit-hint" class="hint" style="margin-top:10px;">請先選擇模型類型</div>
        </form>
        <div id="submit-result" class="form-message" style="margin-top:12px;"></div>
      </section>
      <section class="card">
        <div class="section-title">
          <h2>服務狀態</h2>
          <span class="pill">通知暫不寄送</span>
        </div>
        <div class="status-board" style="margin-top:14px;">
          <div id="service-status-main" class="status-main status-resting">
            <div>
              <div class="status-note">目前伺服器狀態</div>
              <strong id="service-status-label">閒置</strong>
            </div>
            <span class="pill" id="service-queue-pill">排隊 0 筆</span>
          </div>
          <div class="meta">
            <div>WSL 發行版：<strong>AMB_Model</strong></div>
            <div>處理模式：<strong>單一 worker 排隊</strong></div>
            <div>目前轉換中：<strong id="service-running-count">0</strong> 筆</div>
            <div>目前排隊中：<strong id="service-queue-count">0</strong> 筆</div>
            <div>累計完成：<strong id="service-total-completed-count">0</strong> 筆</div>
          </div>
          <div class="queue-list">
            <strong>轉換中的工作編號</strong>
            <div id="service-running-job-ids" class="queue-job-ids">
              <span class="queue-empty">目前沒有轉換中的工作</span>
            </div>
          </div>
          <div class="queue-list">
            <strong>排隊中的工作編號</strong>
            <div id="service-queue-job-ids" class="queue-job-ids">
              <span class="queue-empty">目前沒有排隊工作</span>
            </div>
          </div>
        </div>
        <div class="actions" style="margin-top:16px;">
          <button id="refresh-jobs" class="secondary" type="button">重新整理工作列表</button>
        </div>
      </section>
    </section>
    <section class="card">
      <div class="section-title">
        <h2>其他服務</h2>
      </div>
      <div class="service-links">
        <a class="service-link" href="https://mqttgo.io" target="_blank" rel="noreferrer">
          <div class="service-icon">
            <img src="/api/service-icon/mqttgo" alt="mqttgo.io logo">
          </div>
          <strong>mqttgo.io</strong>
          <span>免費匿名的 mqtt 服務</span>
        </a>
        <a class="service-link" href="https://mqttgo.vip" target="_blank" rel="noreferrer">
          <div class="service-icon">
            <img src="/api/service-icon/mqttgovip" alt="mqttgo.vip logo">
          </div>
          <strong>mqttgo.vip</strong>
          <span>專業的 mqtt 服務</span>
        </a>
        <a class="service-link" href="https://nmking.io" target="_blank" rel="noreferrer">
          <div class="service-icon">
            <img src="/api/service-icon/nmking" alt="nmking.io logo">
          </div>
          <strong>nmking.io</strong>
          <span>教學網站</span>
        </a>
        <a class="service-link" href="https://twgo.io" target="_blank" rel="noreferrer">
          <div class="service-icon twgo">T</div>
          <strong>twgo.io</strong>
          <span>簡單免費轉址服務</span>
        </a>
      </div>
    </section>
    <section class="card">
      <div class="section-title">
        <h2>範例下載</h2>
      </div>
      <div class="service-links">
        <a class="service-link" href="/api/examples/arduino-imgclassification">
          <div class="service-icon twgo">A</div>
          <strong>8735(AMB82) Image Classification Arduino 範例</strong>
          <span>下載 Arduino 範例專案，將模型放到 <code>NN_MDL/imgclassification.nb</code> 後即可測試。</span>
        </a>
      </div>
    </section>
  </main>
  <script>
    function showFormMessage(message, type = "error") {
      const result = document.getElementById("submit-result");
      result.textContent = message;
      result.classList.add("show");
      if (type === "error") {
        result.style.borderColor = "#fecaca";
        result.style.background = "#fef2f2";
        result.style.color = "#991b1b";
      } else {
        result.style.borderColor = "#bfdbfe";
        result.style.background = "#eff6ff";
        result.style.color = "#1d4ed8";
      }
    }

    function clearFormMessage() {
      const result = document.getElementById("submit-result");
      result.textContent = "";
      result.classList.remove("show");
    }

    function renderService(summary) {
      const main = document.getElementById("service-status-main");
      const label = document.getElementById("service-status-label");
      const queuePill = document.getElementById("service-queue-pill");
      const runningCount = document.getElementById("service-running-count");
      const queueCount = document.getElementById("service-queue-count");
      const totalCompletedCount = document.getElementById("service-total-completed-count");
      const runningJobIds = document.getElementById("service-running-job-ids");
      const queueJobIds = document.getElementById("service-queue-job-ids");

      label.textContent = summary.status_label;
      queuePill.textContent = `排隊 ${summary.queue_count} 筆`;
      runningCount.textContent = summary.running_count;
      queueCount.textContent = summary.queue_count;
      totalCompletedCount.textContent = summary.total_completed_count;
      runningJobIds.innerHTML = "";
      if (summary.running_job_ids && summary.running_job_ids.length) {
        for (const jobId of summary.running_job_ids) {
          const chip = document.createElement("span");
          chip.className = "queue-job-id";
          chip.textContent = jobId;
          runningJobIds.appendChild(chip);
        }
      } else {
        const empty = document.createElement("span");
        empty.className = "queue-empty";
        empty.textContent = "目前沒有轉換中的工作";
        runningJobIds.appendChild(empty);
      }
      queueJobIds.innerHTML = "";
      if (summary.queue_job_ids && summary.queue_job_ids.length) {
        for (const jobId of summary.queue_job_ids) {
          const chip = document.createElement("span");
          chip.className = "queue-job-id";
          chip.textContent = jobId;
          queueJobIds.appendChild(chip);
        }
      } else {
        const empty = document.createElement("span");
        empty.className = "queue-empty";
        empty.textContent = "目前沒有排隊工作";
        queueJobIds.appendChild(empty);
      }

      main.classList.remove("status-resting", "status-busy");
      main.classList.add(summary.status_label === "轉換中" ? "status-busy" : "status-resting");
    }

    async function refreshJobs() {
      const response = await fetch("/api/jobs");
      const data = await response.json();
      renderService(data.service);
    }

    let currentCaptchaId = null;

    async function refreshCaptcha() {
      const response = await fetch("/api/captcha");
      const data = await response.json();
      currentCaptchaId = data.captcha_id;
      document.getElementById("captcha-code").textContent = data.captcha_code;
      document.getElementById("captcha-input").value = "";
    }

    function updateModelTypeUI() {
      const modelType = document.getElementById("model-type").value;
      const heroGuide1 = document.getElementById("hero-guide-1");
      const heroGuide2 = document.getElementById("hero-guide-2");
      const formHint = document.getElementById("form-hint");
      const submitHint = document.getElementById("submit-hint");
      const submitButton = document.querySelector('#upload-form button[type="submit"]');
      const zipHelpText = document.getElementById("zip-help-text");
      const teachableFields = document.getElementById("teachable-fields");
      const yoloFields = document.getElementById("yolo-darknet-fields");

      if (modelType === "yolo_darknet") {
        submitHint.textContent = "已選擇 YOLO Darknet，可繼續填寫資料並開始轉換";
        submitButton.disabled = false;
        heroGuide1.innerHTML = '1. 請準備 YOLO Darknet 模型檔，至少需要 <code>.cfg</code> 與 <code>.weights</code>；<code>classes.txt</code> 可選。';
        heroGuide2.innerHTML = '2. 系統會嘗試將 YOLO Darknet 模型轉為 <code>yolov4_tiny.nb</code>。這是第一版 beta 流程。';
        formHint.textContent = "YOLO Darknet 第一版 beta：目前使用 .cfg / .weights 與校正圖片進行轉換，classes.txt 為選填。";
        teachableFields.style.display = "none";
        yoloFields.style.display = "block";
      } else if (modelType === "teachable") {
        submitHint.textContent = "已選擇 Teachable Machine，可繼續填寫資料並開始轉換";
        submitButton.disabled = false;
        heroGuide1.innerHTML = '1. 請先到 <a href="https://teachablemachine.withgoogle.com/" target="_blank" rel="noreferrer">Google Teachable Machine網站</a> 建立並訓練自己的模型，完成後匯出模型檔 <code>converted_keras.zip</code>。';
        heroGuide2.innerHTML = '2. 上傳 <code>converted_keras.zip</code> 與至少一張校正圖片後，系統呼叫轉換流程，產出 <code>imgclassification.nb</code> 的下載路徑，回復到指定的 Mail。';
        formHint.textContent = "目前要求填 email、模型 zip 與至少一張校正圖片。轉換完成後的 email 通知流程先保留欄位，寄信設定下一步再接。";
        zipHelpText.innerHTML = '請選取 <code>converted_keras.zip</code>';
        teachableFields.style.display = "block";
        yoloFields.style.display = "none";
      } else {
        submitHint.textContent = "請先選擇模型類型";
        submitButton.disabled = true;
        heroGuide1.innerHTML = '1. 請先選擇模型類型。';
        heroGuide2.innerHTML = '2. 選擇後系統會顯示對應的上傳欄位與轉換說明。';
        formHint.textContent = "請先選擇模型類型，再進行檔案上傳。";
        teachableFields.style.display = "none";
        yoloFields.style.display = "none";
      }
    }

    function resetUploadForm() {
      document.getElementById("model-type").value = "";
      document.getElementById("email").value = "";
      document.getElementById("zip-file").value = "";
      document.getElementById("yolo-cfg-file").value = "";
      document.getElementById("yolo-weights-file").value = "";
      document.getElementById("yolo-classes-file").value = "";
      document.getElementById("calibration-files").value = "";
      document.getElementById("captcha-input").value = "";
      document.getElementById("zip-file-name").textContent = "尚未選擇檔案";
      document.getElementById("yolo-cfg-file-name").textContent = "尚未選擇檔案";
      document.getElementById("yolo-weights-file-name").textContent = "尚未選擇檔案";
      document.getElementById("yolo-classes-file-name").textContent = "尚未選擇檔案";
      document.getElementById("calibration-file-name").textContent = "尚未選擇檔案";
      document.getElementById("selected-zip-text").textContent = "";
      document.getElementById("selected-zip-file").style.display = "none";
      const preview = document.getElementById("calibration-preview");
      preview.innerHTML = "";
      preview.style.display = "none";
      updateModelTypeUI();
    }

    document.getElementById("refresh-jobs").addEventListener("click", refreshJobs);
    document.getElementById("model-type").addEventListener("change", updateModelTypeUI);
    document.getElementById("upload-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const modelTypeInput = document.getElementById("model-type");
      const fileInput = document.getElementById("zip-file");
      const yoloCfgInput = document.getElementById("yolo-cfg-file");
      const yoloWeightsInput = document.getElementById("yolo-weights-file");
      const yoloClassesInput = document.getElementById("yolo-classes-file");
      const calibrationInput = document.getElementById("calibration-files");
      const emailInput = document.getElementById("email");
      const captchaInput = document.getElementById("captcha-input");
      if (!modelTypeInput.value) {
        showFormMessage("請先選擇模型類型。");
        modelTypeInput.focus();
        return;
      }
      const isTeachable = modelTypeInput.value === "teachable";
      const modelReady = isTeachable
        ? fileInput.files.length
        : (yoloCfgInput.files.length && yoloWeightsInput.files.length);
      if (!modelReady || !calibrationInput.files.length || !emailInput.value.trim() || !captchaInput.value.trim()) {
        if (!emailInput.value.trim()) {
          showFormMessage("請先輸入通知 email。");
          emailInput.focus();
          return;
        }
        if (!modelReady && isTeachable) {
          showFormMessage("請先選擇模型壓縮檔 converted_keras.zip。");
          fileInput.click();
          return;
        }
        if (!modelReady && !isTeachable) {
          showFormMessage("請先提供 YOLO Darknet 的 .cfg 與 .weights。");
          yoloCfgInput.click();
          return;
        }
        if (!calibrationInput.files.length) {
          showFormMessage("請至少選擇 1 張校正圖片。");
          calibrationInput.click();
          return;
        }
        if (!captchaInput.value.trim()) {
          showFormMessage("請先輸入數字驗證碼。");
          captchaInput.focus();
          return;
        }
        return;
      }
      showFormMessage("上傳中...", "info");
      const formData = new FormData();
      formData.append("model_type", modelTypeInput.value);
      formData.append("email", emailInput.value.trim());
      formData.append("captcha_id", currentCaptchaId || "");
      formData.append("captcha_input", captchaInput.value.trim());
      if (isTeachable) {
        formData.append("file", fileInput.files[0]);
      } else {
        formData.append("yolo_cfg_file", yoloCfgInput.files[0]);
        formData.append("yolo_weights_file", yoloWeightsInput.files[0]);
        if (yoloClassesInput.files.length) {
          formData.append("yolo_classes_file", yoloClassesInput.files[0]);
        }
      }
      for (const item of calibrationInput.files) {
        formData.append("calibration_files", item);
      }
      const response = await fetch("/api/jobs", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        showFormMessage(data.detail || "建立工作失敗");
        await refreshCaptcha();
        return;
      }
      showFormMessage(`工作已建立：${data.job_id}`, "info");
      resetUploadForm();
      await refreshCaptcha();
      await refreshJobs();
    });

    document.getElementById("zip-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      document.getElementById("zip-file-name").textContent = file ? file.name : "尚未選擇檔案";
      const selectedBox = document.getElementById("selected-zip-file");
      const selectedText = document.getElementById("selected-zip-text");
      clearFormMessage();
      if (file) {
        selectedText.textContent = file.name;
        selectedBox.style.display = "block";
      } else {
        selectedText.textContent = "";
        selectedBox.style.display = "none";
      }
    });

    document.getElementById("yolo-cfg-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      document.getElementById("yolo-cfg-file-name").textContent = file ? file.name : "尚未選擇檔案";
      clearFormMessage();
    });

    document.getElementById("yolo-weights-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      document.getElementById("yolo-weights-file-name").textContent = file ? file.name : "尚未選擇檔案";
      clearFormMessage();
    });

    document.getElementById("yolo-classes-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      document.getElementById("yolo-classes-file-name").textContent = file ? file.name : "尚未選擇檔案";
      clearFormMessage();
    });

    document.getElementById("calibration-files").addEventListener("change", (event) => {
      const files = Array.from(event.target.files);
      const count = files.length;
      document.getElementById("calibration-file-name").textContent = count ? `已選擇 ${count} 張圖片` : "尚未選擇檔案";
      clearFormMessage();

      const preview = document.getElementById("calibration-preview");
      preview.innerHTML = "";
      if (!count) {
        preview.style.display = "none";
        return;
      }

      preview.style.display = "grid";
      for (const file of files) {
        const item = document.createElement("div");
        item.className = "preview-item";

        const image = document.createElement("img");
        image.alt = file.name;
        image.src = URL.createObjectURL(file);
        image.onload = () => URL.revokeObjectURL(image.src);

        const label = document.createElement("span");
        label.textContent = file.name;

        item.appendChild(image);
        item.appendChild(label);
        preview.appendChild(item);
      }
    });

    document.getElementById("email").addEventListener("input", clearFormMessage);
    document.getElementById("captcha-input").addEventListener("input", clearFormMessage);

    refreshCaptcha();
    refreshJobs();
    updateModelTypeUI();
    setInterval(refreshJobs, 4000);
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return html_page()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "wsl_distro": WSL_DISTRO}


@app.get("/api/jobs")
async def list_jobs() -> dict[str, object]:
    jobs = sorted(app.state.jobs.values(), key=lambda item: item.created_at, reverse=True)
    return {"jobs": [job.to_dict() for job in jobs], "service": service_summary()}


@app.get("/api/captcha")
async def get_captcha() -> dict[str, str]:
    captcha_id, code = create_captcha()
    return {"captcha_id": captcha_id, "captcha_code": code}


@app.get("/api/service-icon/{icon_name}")
async def get_service_icon(icon_name: str) -> FileResponse:
    path = SERVICE_ICON_PATHS.get(icon_name)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="找不到圖示")
    return FileResponse(path)


@app.get("/api/examples/arduino-imgclassification")
async def download_arduino_imgclassification_example() -> FileResponse:
    path = EXAMPLE_DOWNLOAD_PATHS["arduino_imgclassification"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example archive not found.")
    return FileResponse(
        path,
        media_type="application/zip",
        filename="ameba_imgclassification_final.zip",
    )


@app.get("/favicon.ico")
async def get_favicon_ico() -> FileResponse:
    return FileResponse(FAVICON_PATHS["ico"], media_type="image/x-icon")


@app.get("/favicon-16x16.png")
async def get_favicon_16() -> FileResponse:
    return FileResponse(FAVICON_PATHS["16"], media_type="image/png")


@app.get("/favicon-32x32.png")
async def get_favicon_32() -> FileResponse:
    return FileResponse(FAVICON_PATHS["32"], media_type="image/png")


@app.get("/apple-touch-icon.png")
async def get_apple_touch_icon() -> FileResponse:
    return FileResponse(FAVICON_PATHS["apple"], media_type="image/png")


@app.post("/api/jobs")
async def create_job(
    model_type: str = Form(...),
    email: str = Form(...),
    captcha_id: str = Form(...),
    captcha_input: str = Form(...),
    file: UploadFile | None = File(None),
    yolo_cfg_file: UploadFile | None = File(None),
    yolo_weights_file: UploadFile | None = File(None),
    yolo_classes_file: UploadFile | None = File(None),
    calibration_files: list[UploadFile] = File(...),
) -> JSONResponse:
    model_type = model_type.strip()
    if model_type not in {"teachable", "yolo_darknet"}:
        raise HTTPException(status_code=400, detail="不支援的模型類型")
    email = email.strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="請輸入有效 email")
    expected_code = app.state.captcha_store.pop(captcha_id, None)
    if not expected_code or captcha_input.strip() != expected_code:
        raise HTTPException(status_code=400, detail="數字驗證碼錯誤，請重新輸入")
    if not calibration_files:
        raise HTTPException(status_code=400, detail="至少需要一張校正圖片")

    if model_type == "teachable":
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="缺少檔名")
        if not file.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="目前只接受 .zip 檔")
    else:
        if not yolo_cfg_file or not yolo_cfg_file.filename:
            raise HTTPException(status_code=400, detail="請提供 YOLO Darknet 的 .cfg 檔")
        if not yolo_weights_file or not yolo_weights_file.filename:
            raise HTTPException(status_code=400, detail="請提供 YOLO Darknet 的 .weights 檔")
        if not yolo_cfg_file.filename.lower().endswith(".cfg"):
            raise HTTPException(status_code=400, detail="YOLO 設定檔必須是 .cfg")
        if not yolo_weights_file.filename.lower().endswith(".weights"):
            raise HTTPException(status_code=400, detail="YOLO 權重檔必須是 .weights")
        if yolo_classes_file and yolo_classes_file.filename and not yolo_classes_file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="YOLO 類別名稱檔必須是 .txt")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOB_ROOT / job_id
    input_dir = job_dir / "input"
    work_dir = job_dir / "work"
    calibration_dir = job_dir / "calibration"
    input_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    zip_path = input_dir / "converted_keras.zip"
    log_path = job_dir / "job.log"
    output_path = public_output_path_for_model(work_dir, model_type)

    saved_filename = ""
    if model_type == "teachable":
        with zip_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
        saved_filename = file.filename or "converted_keras.zip"
    else:
        cfg_path = input_dir / "model.cfg"
        weights_path = input_dir / "model.weights"
        classes_path = input_dir / "classes.txt"
        with cfg_path.open("wb") as target:
            shutil.copyfileobj(yolo_cfg_file.file, target)
        with weights_path.open("wb") as target:
            shutil.copyfileobj(yolo_weights_file.file, target)
        if yolo_classes_file and yolo_classes_file.filename:
            with classes_path.open("wb") as target:
                shutil.copyfileobj(yolo_classes_file.file, target)
        saved_filename = f"{yolo_cfg_file.filename} + {yolo_weights_file.filename}"

    saved_count = 0
    for index, image in enumerate(calibration_files, start=1):
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            continue
        target_name = f"img_{index:02d}{'.jpg' if suffix == '.jpeg' else suffix}"
        with (calibration_dir / target_name).open("wb") as target:
            shutil.copyfileobj(image.file, target)
        saved_count += 1

    if saved_count == 0:
        raise HTTPException(status_code=400, detail="校正圖片只接受 jpg、jpeg、png")

    record = JobRecord(
        job_id=job_id,
        filename=saved_filename,
        email=email,
        created_at=now_text(),
        model_type=model_type,
        log_path=str(log_path),
        output_path=str(output_path) if output_path.exists() else None,
        work_dir=str(work_dir),
        calibration_count=saved_count,
    )
    app.state.jobs[job_id] = record
    app.state.job_queue.put(job_id)
    refresh_queue_positions()
    threading.Thread(target=_send_received_email_background, args=(record,), daemon=True).start()
    return JSONResponse({"ok": True, "job_id": job_id})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, object]:
    job = app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到工作")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/log")
async def get_job_log(job_id: str) -> HTMLResponse:
    job = app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到工作")
    if not job.log_path or not Path(job.log_path).exists():
        return HTMLResponse("", status_code=200)
    return HTMLResponse(Path(job.log_path).read_text(encoding="utf-8", errors="replace"))


@app.get("/api/jobs/{job_id}/download")
async def download_output(job_id: str) -> FileResponse:
    job = app.state.jobs.get(job_id)
    if job:
        output_name = public_output_filename_for_model(job.model_type)
    else:
        output_name = PUBLIC_OUTPUT_NB_FILENAME
    fallback_output = JOB_ROOT / job_id / "work" / "out_nbg_unify" / output_name
    if not job and fallback_output.exists():
        return FileResponse(
            fallback_output,
            media_type="application/octet-stream",
            filename=output_name,
        )
    if not job:
        raise HTTPException(status_code=404, detail="找不到工作")
    output_path = Path(job.output_path) if job.output_path else fallback_output
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="nb 尚未產生")
    return FileResponse(
        output_path,
        media_type="application/octet-stream",
        filename=output_name,
    )


def main() -> None:
    uvicorn.run("model_convert_webui:app", host=WEBUI_HOST, port=WEBUI_PORT, reload=False)


app.state.metrics = load_service_state()
worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()


if __name__ == "__main__":
    main()
