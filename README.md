# 8735(AMB82) Teachable Machine 模型轉換

這個專案提供一個可部署的 Web UI，讓使用者上傳 `converted_keras.zip` 與校正圖片，背景呼叫 Acuity Docker 轉換流程，最後產出 `network_binary.nb`，並透過 email 通知下載連結。

## 功能

- 上傳 `converted_keras.zip`
- 至少 1 張校正圖片
- 單一 worker 排隊轉換
- 累計完成次數持久保存
- Gmail SMTP 成功/失敗通知
- `model.mqttgo.io` HTTPS 反向代理部署

## 專案結構

- `model_convert_webui.py`
  Web UI 與 API 主程式
- `zip_to_nb_wsl.sh`
  `converted_keras.zip -> network_binary.nb` 主轉換流程
- `zip_to_nb_wsl_gpu.sh`
  GPU 版包裝腳本
- `run_model_convert_webui_wsl.sh`
  WSL / systemd 啟動腳本
- `service_icons/`
  前台與 mail 使用的品牌圖示
- `site_assets/`
  favicon 與網站資產

## 環境需求

- Windows + WSL2
- WSL 發行版：`AMB_Model`
- Docker Engine 可在 WSL 內使用
- 可拉取 `ghcr.io/ameba-aiot/acuity-toolkit:6.18.8`
- Python 3.10+

## 本機啟動

在 PowerShell：

```powershell
.\run_model_convert_webui.ps1
```

在 WSL：

```bash
bash ./run_model_convert_webui_wsl.sh
```

## 設定檔

將 `.env.example` 複製成 `.env`，填入：

- `MODEL_WEBUI_BASE_URL`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`

## systemd 部署

目前正式服務名稱：

```text
amb82-model-convert.service
```

常用指令：

```bash
systemctl status amb82-model-convert.service
sudo systemctl restart amb82-model-convert.service
journalctl -u amb82-model-convert.service -n 100 --no-pager
```

## HTTPS

目前以 Caddy 反向代理：

```text
https://model.mqttgo.io -> http://127.0.0.1:8891
```

## 注意事項

- `.env` 不應同步到 GitHub
- `webui_jobs/` 與 `service_state.json` 屬於執行期資料
- 如果要保留累計完成次數，請備份 `service_state.json`
